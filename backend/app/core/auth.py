"""Supabase JWT authentication and FastAPI dependencies for Retrieva AI.

Verifies Supabase-issued JWT access tokens asymmetrically against the project's
JWKS endpoint using PyJWT. Never trusts unverified claims, never logs raw tokens,
and never uses a Supabase service-role key.
"""

import asyncio
import threading
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import (
    ExpiredSignatureError,
    InvalidAlgorithmError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidSignatureError,
    InvalidTokenError,
    MissingRequiredClaimError,
    PyJWKClient,
    PyJWKClientConnectionError,
    PyJWKClientError,
)
from pydantic import BaseModel, Field

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

logger = get_logger("core.auth")

bearer_scheme = HTTPBearer(auto_error=False)

_jwks_client_lock = threading.Lock()
_jwks_clients: Dict[Tuple[str, int, float], PyJWKClient] = {}

_DEV_BYPASS_USER_ID = UUID("00000000-0000-0000-0000-000000000000")


class AuthenticatedUser(BaseModel):
    """Verified caller identity extracted from a validated Supabase JWT."""

    user_id: UUID
    email: Optional[str] = None
    role: str = "authenticated"
    claims: Dict[str, Any] = Field(default_factory=dict)


def clear_jwks_client_cache() -> None:
    """Clear cached PyJWKClient instances (primarily for test isolation)."""
    with _jwks_client_lock:
        _jwks_clients.clear()


def _get_jwks_client(jwks_url: str, cache_ttl: int, timeout: float) -> PyJWKClient:
    """Return a cached, bounded PyJWKClient for the given JWKS configuration."""
    cache_key = (jwks_url, cache_ttl, timeout)
    with _jwks_client_lock:
        client = _jwks_clients.get(cache_key)
        if client is None:
            client = PyJWKClient(
                uri=jwks_url,
                cache_keys=True,
                max_cached_keys=16,
                cache_jwk_set=True,
                lifespan=cache_ttl,
                timeout=timeout,
            )
            _jwks_clients[cache_key] = client
        return client


def _unauthorized(detail: str) -> HTTPException:
    """Build a standardized 401 Unauthorized exception."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _forbidden(detail: str) -> HTTPException:
    """Build a standardized 403 Forbidden exception."""
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
    )


def _service_unavailable(detail: str) -> HTTPException:
    """Build a standardized 503 Service Unavailable exception."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
    )


def _decode_and_verify_sync(token: str, settings: Settings) -> AuthenticatedUser:
    """Synchronously fetch the signing key (cached/rotating) and verify the JWT."""
    jwks_url = settings.resolved_jwks_url
    issuer = settings.resolved_issuer

    if not jwks_url or not issuer:
        logger.error("Supabase JWT authentication configuration is missing (JWKS URL or issuer).")
        raise _service_unavailable("Authentication service is not configured.")

    jwks_client = _get_jwks_client(
        jwks_url=jwks_url,
        cache_ttl=settings.SUPABASE_JWKS_CACHE_TTL_SECONDS,
        timeout=settings.SUPABASE_JWKS_TIMEOUT_SECONDS,
    )

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
    except PyJWKClientConnectionError as exc:
        logger.warning("JWKS verification endpoint unavailable: %s", exc)
        raise _service_unavailable("Authentication verification service is temporarily unavailable.") from exc
    except PyJWKClientError as exc:
        logger.warning("JWT signing key lookup failed: %s", exc)
        raise _unauthorized("Authentication token signing key is invalid or unrecognized.") from exc
    except InvalidTokenError as exc:
        logger.warning("Malformed JWT header during key lookup.")
        raise _unauthorized("Authentication token is malformed.") from exc

    try:
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=settings.SUPABASE_JWT_ALGORITHMS,
            issuer=issuer,
            audience=settings.SUPABASE_JWT_AUDIENCE,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
                "require": ["exp", "iat", "sub", "aud", "iss"],
            },
        )
    except ExpiredSignatureError as exc:
        logger.info("Rejected expired JWT.")
        raise _unauthorized("Authentication token has expired.") from exc
    except InvalidIssuerError as exc:
        logger.warning("Rejected JWT with invalid issuer.")
        raise _unauthorized("Authentication token issuer is invalid.") from exc
    except InvalidAudienceError as exc:
        logger.warning("Rejected JWT with invalid audience.")
        raise _unauthorized("Authentication token audience is invalid.") from exc
    except InvalidAlgorithmError as exc:
        logger.warning("Rejected JWT with unsupported signing algorithm.")
        raise _unauthorized("Authentication token algorithm is not supported.") from exc
    except InvalidSignatureError as exc:
        logger.warning("Rejected JWT with invalid signature.")
        raise _unauthorized("Authentication token signature is invalid.") from exc
    except MissingRequiredClaimError as exc:
        logger.warning("Rejected JWT missing required claim.")
        raise _unauthorized("Authentication token is missing required claims.") from exc
    except InvalidTokenError as exc:
        logger.warning("Rejected invalid JWT.")
        raise _unauthorized("Authentication token is invalid.") from exc

    raw_sub = payload.get("sub")
    if not raw_sub or not isinstance(raw_sub, str):
        logger.warning("Rejected JWT missing valid string subject claim.")
        raise _unauthorized("Authentication token subject is missing or invalid.")

    try:
        user_id = UUID(raw_sub)
    except ValueError as exc:
        logger.warning("Rejected JWT with non-UUID subject claim.")
        raise _unauthorized("Authentication token subject is not a valid UUID.") from exc

    role = payload.get("role")
    if role != "authenticated":
        logger.warning("Rejected JWT with insufficient role '%s'.", role)
        raise _forbidden("Authenticated user role is required.")

    email = payload.get("email")
    return AuthenticatedUser(
        user_id=user_id,
        email=email if isinstance(email, str) else None,
        role=role,
        claims=payload,
    )


async def verify_supabase_jwt(
    token: str,
    settings: Optional[Settings] = None,
) -> AuthenticatedUser:
    """Verify a Supabase JWT access token without blocking the asyncio event loop."""
    active_settings = settings or get_settings()
    clean_token = (token or "").strip()
    if not clean_token:
        raise _unauthorized("Authentication credentials were not provided.")
    return await asyncio.to_thread(_decode_and_verify_sync, clean_token, active_settings)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedUser:
    """FastAPI dependency that authenticates the caller via Supabase Bearer JWT."""
    env = settings.APP_ENV.strip().lower()
    if not settings.AUTH_ENABLED:
        if env == "development":
            return AuthenticatedUser(
                user_id=_DEV_BYPASS_USER_ID,
                email="dev-local@retrieva.local",
                role="authenticated",
                claims={"sub": str(_DEV_BYPASS_USER_ID), "role": "authenticated"},
            )
        logger.error("AUTH_ENABLED=False is forbidden outside development (APP_ENV=%s).", settings.APP_ENV)
        raise _service_unavailable("Authentication configuration is invalid for this environment.")

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Missing or invalid Bearer authentication credentials.")

    token = (credentials.credentials or "").strip()
    if not token:
        raise _unauthorized("Missing Bearer authentication token.")

    return await verify_supabase_jwt(token, settings=settings)
