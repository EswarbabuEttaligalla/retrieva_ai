"""Automated tests for Supabase JWT authentication and endpoint protection."""

import base64
import json
import logging
import time
from typing import Any, Dict, Tuple
from unittest.mock import patch
from uuid import UUID, uuid4

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jwt import PyJWKClientConnectionError

from backend.app.core.auth import (
    AuthenticatedUser,
    clear_jwks_client_cache,
    verify_supabase_jwt,
)
from backend.app.core.config import Settings, get_settings
from backend.app.main import app


def _generate_ec_keypair(kid: str = "ec-key-1") -> Tuple[Any, Dict[str, Any]]:
    """Generate an ephemeral ES256 (P-256) private key and public JWK dict."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk_dict = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(private_key.public_key()))
    jwk_dict["kid"] = kid
    jwk_dict["use"] = "sig"
    jwk_dict["alg"] = "ES256"
    return private_key, jwk_dict


def _generate_rsa_keypair(kid: str = "rsa-key-1") -> Tuple[Any, Dict[str, Any]]:
    """Generate an ephemeral RS256 (2048-bit) private key and public JWK dict."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk_dict = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk_dict["kid"] = kid
    jwk_dict["use"] = "sig"
    jwk_dict["alg"] = "RS256"
    return private_key, jwk_dict


def _build_claims(
    sub: Any = None,
    iss: str = "https://zblfcosunplvnnwjfyjz.supabase.co/auth/v1",
    aud: str = "authenticated",
    role: str = "authenticated",
    exp_offset: int = 3600,
    include_sub: bool = True,
) -> Dict[str, Any]:
    now = int(time.time())
    claims: Dict[str, Any] = {
        "iss": iss,
        "aud": aud,
        "role": role,
        "email": "user@example.com",
        "iat": now - 10,
        "exp": now + exp_offset,
    }
    if include_sub:
        claims["sub"] = str(sub if sub is not None else uuid4())
    return claims


def _make_auth_settings(**overrides: Any) -> Settings:
    defaults: Dict[str, Any] = {
        "APP_ENV": "development",
        "SUPABASE_URL": "https://zblfcosunplvnnwjfyjz.supabase.co",
        "SUPABASE_ANON_KEY": "test-public-anon-key",
        "SUPABASE_JWT_AUDIENCE": "authenticated",
        "SUPABASE_JWT_ALGORITHMS": ["ES256", "RS256"],
        "SUPABASE_JWKS_CACHE_TTL_SECONDS": 300,
        "SUPABASE_JWKS_TIMEOUT_SECONDS": 5.0,
        "AUTH_ENABLED": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture(autouse=True)
def reset_jwks_cache():
    """Ensure PyJWKClient cache is isolated per test."""
    clear_jwks_client_cache()
    yield
    clear_jwks_client_cache()


class TestJWTVerification:
    """Unit tests for Supabase JWT signature, claim, JWKS, and config validation."""

    @pytest.mark.asyncio
    async def test_valid_es256_jwt(self):
        priv, jwk = _generate_ec_keypair("ec-1")
        user_uuid = uuid4()
        token = jwt.encode(
            _build_claims(sub=user_uuid),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-1"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            user = await verify_supabase_jwt(token, settings=settings)

        assert isinstance(user, AuthenticatedUser)
        assert user.user_id == user_uuid
        assert user.email == "user@example.com"
        assert user.role == "authenticated"

    @pytest.mark.asyncio
    async def test_valid_rs256_jwt(self):
        priv, jwk = _generate_rsa_keypair("rsa-1")
        user_uuid = uuid4()
        token = jwt.encode(
            _build_claims(sub=user_uuid),
            priv,
            algorithm="RS256",
            headers={"kid": "rsa-1"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            user = await verify_supabase_jwt(token, settings=settings)

        assert user.user_id == user_uuid
        assert user.role == "authenticated"

    @pytest.mark.asyncio
    async def test_expired_jwt_rejected_401(self):
        priv, jwk = _generate_ec_keypair("ec-exp")
        token = jwt.encode(
            _build_claims(exp_offset=-120),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-exp"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(token, settings=settings)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_issuer_rejected_401(self):
        priv, jwk = _generate_ec_keypair("ec-iss")
        token = jwt.encode(
            _build_claims(iss="https://evil.example.com/auth/v1"),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-iss"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(token, settings=settings)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_audience_rejected_401(self):
        priv, jwk = _generate_ec_keypair("ec-aud")
        token = jwt.encode(
            _build_claims(aud="service_role"),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-aud"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(token, settings=settings)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected_401(self):
        priv_signer, _ = _generate_ec_keypair("ec-sig")
        _, jwk_other = _generate_ec_keypair("ec-sig")
        token = jwt.encode(
            _build_claims(),
            priv_signer,
            algorithm="ES256",
            headers={"kid": "ec-sig"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk_other]}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(token, settings=settings)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_tampered_payload_rejected_401(self):
        priv, jwk = _generate_ec_keypair("ec-tamper")
        token = jwt.encode(
            _build_claims(sub=uuid4()),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-tamper"},
        )
        header_b64, _, sig_b64 = token.split(".")
        tampered_claims = _build_claims(sub=uuid4())
        tampered_b64 = base64.urlsafe_b64encode(
            json.dumps(tampered_claims).encode("utf-8")
        ).rstrip(b"=").decode("ascii")
        tampered_token = f"{header_b64}.{tampered_b64}.{sig_b64}"

        settings = _make_auth_settings()
        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(tampered_token, settings=settings)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_unsupported_algorithm_rejected_401(self):
        _, jwk = _generate_ec_keypair("ec-hs")
        # Sign with symmetric HS256 while server only allows ES256/RS256
        hs_token = jwt.encode(
            _build_claims(),
            "super-secret-symmetric-key-for-test-only-32bytes",
            algorithm="HS256",
            headers={"kid": "ec-hs"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(hs_token, settings=settings)

        assert exc_info.value.status_code == 401

        with pytest.raises(ValueError):
            _make_auth_settings(SUPABASE_JWT_ALGORITHMS=["HS256"])

        with pytest.raises(ValueError):
            _make_auth_settings(SUPABASE_JWT_ALGORITHMS=["none"])

    @pytest.mark.asyncio
    async def test_missing_subject_rejected_401(self):
        priv, jwk = _generate_ec_keypair("ec-nosub")
        token = jwt.encode(
            _build_claims(include_sub=False),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-nosub"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(token, settings=settings)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_uuid_subject_rejected_401(self):
        priv, jwk = _generate_ec_keypair("ec-badsub")
        token = jwt.encode(
            _build_claims(sub="not-a-valid-uuid"),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-badsub"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(token, settings=settings)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_anonymous_role_rejected_403(self):
        priv, jwk = _generate_ec_keypair("ec-anon")
        token = jwt.encode(
            _build_claims(role="anon"),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-anon"},
        )
        settings = _make_auth_settings()

        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(token, settings=settings)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_jwks_unknown_key_and_key_rotation(self):
        priv_old, jwk_old = _generate_ec_keypair("kid-old")
        priv_new, jwk_new = _generate_ec_keypair("kid-new")
        user_uuid = uuid4()
        settings = _make_auth_settings()

        token_old = jwt.encode(
            _build_claims(sub=user_uuid),
            priv_old,
            algorithm="ES256",
            headers={"kid": "kid-old"},
        )
        token_new = jwt.encode(
            _build_claims(sub=user_uuid),
            priv_new,
            algorithm="ES256",
            headers={"kid": "kid-new"},
        )
        token_unknown = jwt.encode(
            _build_claims(sub=user_uuid),
            priv_new,
            algorithm="ES256",
            headers={"kid": "kid-nonexistent"},
        )

        # First call caches kid-old; second call sees kid-new, refreshes JWKS, and succeeds
        with patch(
            "jwt.PyJWKClient.fetch_data",
            side_effect=[
                {"keys": [jwk_old]},
                {"keys": [jwk_old, jwk_new]},
                {"keys": [jwk_old, jwk_new]},
            ],
        ):
            user1 = await verify_supabase_jwt(token_old, settings=settings)
            user2 = await verify_supabase_jwt(token_new, settings=settings)
            assert user1.user_id == user_uuid
            assert user2.user_id == user_uuid

            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(token_unknown, settings=settings)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_jwks_unavailable_returns_503(self):
        priv, _ = _generate_ec_keypair("ec-down")
        token = jwt.encode(
            _build_claims(),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-down"},
        )
        settings = _make_auth_settings()

        with patch(
            "jwt.PyJWKClient.fetch_data",
            side_effect=PyJWKClientConnectionError("JWKS endpoint unreachable"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await verify_supabase_jwt(token, settings=settings)

        assert exc_info.value.status_code == 503

    def test_missing_production_config_fails_closed(self):
        with pytest.raises(ValueError):
            _make_auth_settings(
                APP_ENV="production",
                SUPABASE_URL=None,
                SUPABASE_JWKS_URL=None,
                SUPABASE_JWT_ISSUER=None,
            )

        with pytest.raises(ValueError):
            _make_auth_settings(
                APP_ENV="production",
                AUTH_ENABLED=False,
            )


class TestEndpointAuthentication:
    """Integration tests verifying route protection and log safety."""

    def test_public_endpoints_accessible_without_auth(self):
        with TestClient(app) as client:
            root_res = client.get("/")
            assert root_res.status_code == 200

            health_res = client.get("/api/v1/health")
            assert health_res.status_code == 200

    @pytest.mark.parametrize(
        "method,path,kwargs",
        [
            ("GET", "/api/v1/documents", {}),
            ("POST", "/api/v1/documents/upload", {}),
            ("DELETE", "/api/v1/documents/doc_123", {}),
            ("POST", "/api/v1/retrieval/search", {"json": {"query": "test"}}),
            ("POST", "/api/v1/answer", {"json": {"query": "test"}}),
            ("POST", "/api/v1/answer/stream", {"json": {"query": "test"}}),
            ("GET", "/api/v1/conversations", {}),
            ("POST", "/api/v1/conversations", {"json": {"title": "Test"}}),
            ("GET", "/api/v1/conversations/conv_123", {}),
            ("PATCH", "/api/v1/conversations/conv_123", {"json": {"title": "New"}}),
            ("DELETE", "/api/v1/conversations/conv_123", {}),
            ("POST", "/api/v1/conversations/conv_123/answer", {"json": {"query": "test"}}),
            ("POST", "/api/v1/conversations/conv_123/answer/stream", {"json": {"query": "test"}}),
        ],
    )
    def test_protected_endpoints_reject_missing_bearer_token(self, method: str, path: str, kwargs: Dict[str, Any]):
        settings = _make_auth_settings()
        prev_settings_override = app.dependency_overrides.get(get_settings)
        app.dependency_overrides[get_settings] = lambda: settings
        try:
            with TestClient(app) as client:
                res = client.request(method, path, **kwargs)
                assert res.status_code == 401
                assert res.headers.get("www-authenticate") == "Bearer"
        finally:
            if prev_settings_override is None:
                app.dependency_overrides.pop(get_settings, None)
            else:
                app.dependency_overrides[get_settings] = prev_settings_override

    def test_protected_endpoint_accepts_valid_jwt_and_never_logs_token(self, caplog):
        priv, jwk = _generate_ec_keypair("ec-http")
        user_uuid = uuid4()
        valid_token = jwt.encode(
            _build_claims(sub=user_uuid),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-http"},
        )
        expired_token = jwt.encode(
            _build_claims(sub=user_uuid, exp_offset=-60),
            priv,
            algorithm="ES256",
            headers={"kid": "ec-http"},
        )
        settings = _make_auth_settings()
        prev_settings_override = app.dependency_overrides.get(get_settings)
        app.dependency_overrides[get_settings] = lambda: settings

        try:
            import httpx
            mock_pool = httpx.AsyncClient(
                transport=httpx.MockTransport(lambda req: httpx.Response(200, json=[]))
            )
            with caplog.at_level(logging.DEBUG, logger="retrieva_ai"):
                with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}), patch(
                    "backend.app.core.supabase_client.get_supabase_http_client",
                    return_value=mock_pool,
                ):
                    with TestClient(app) as client:
                        ok_res = client.get(
                            "/api/v1/documents",
                            headers={"Authorization": f"Bearer {valid_token}"},
                        )
                        assert ok_res.status_code == 200

                        bad_res = client.get(
                            "/api/v1/documents",
                            headers={"Authorization": f"Bearer {expired_token}"},
                        )
                        assert bad_res.status_code == 401

            log_text = caplog.text
            assert valid_token not in log_text
            assert expired_token not in log_text
            assert "Bearer " not in log_text
        finally:
            if prev_settings_override is None:
                app.dependency_overrides.pop(get_settings, None)
            else:
                app.dependency_overrides[get_settings] = prev_settings_override
