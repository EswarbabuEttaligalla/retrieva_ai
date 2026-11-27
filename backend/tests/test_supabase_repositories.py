"""Automated tests for request-scoped SupabaseClient and owner-scoped repositories."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch
from urllib.parse import parse_qs
from uuid import UUID, uuid4

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend.app.core.auth import clear_jwks_client_cache
from backend.app.core.config import Settings, get_settings
from backend.app.core.supabase_client import (
    SupabaseClient,
    SupabaseConfigError,
    SupabaseForbiddenError,
    get_supabase_client,
    get_supabase_http_client,
)
from backend.app.repositories import (
    DocumentRepository,
    SupabaseConversationRepository,
    SupabaseMessageRepository,
    get_document_repository,
    get_message_repository,
    get_supabase_conversation_repository,
)
from backend.app.core.rpc_auth import (
    build_backend_rpc_proof,
    canonical_jsonb,
    compute_create_document_payload_digest,
    compute_delete_document_payload_digest,
    compute_record_assistant_message_payload_digest,
    compute_update_status_payload_digest,
)

USER_A_ID = UUID("10000000-0000-4000-8000-000000000001")
USER_B_ID = UUID("10000000-0000-4000-8000-000000000002")
ANON_KEY = "test-public-anon-key"
SUPABASE_URL = "https://zblfcosunplvnnwjfyjz.supabase.co"


def _make_keypair(kid: str = "repo-ec-1") -> Tuple[Any, Dict[str, Any]]:
    priv = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(priv.public_key()))
    jwk["kid"] = kid
    jwk["use"] = "sig"
    jwk["alg"] = "ES256"
    return priv, jwk


def _issue_jwt(priv: Any, user_id: UUID, kid: str = "repo-ec-1") -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": str(user_id),
            "aud": "authenticated",
            "iss": f"{SUPABASE_URL}/auth/v1",
            "role": "authenticated",
            "email": f"{user_id}@example.test",
            "iat": now - 5,
            "exp": now + 3600,
        },
        priv,
        algorithm="ES256",
        headers={"kid": kid},
    )


def _make_settings() -> Settings:
    return Settings(
        APP_ENV="development",
        SUPABASE_URL=SUPABASE_URL,
        SUPABASE_ANON_KEY=ANON_KEY,
        SUPABASE_REST_TIMEOUT_SECONDS=5.0,
    )


class MockSupabaseRLSEngine:
    """In-memory PostgREST + RLS simulator enforcing owner_id from Bearer JWT and backend HMAC proofs."""

    def __init__(self, token_to_user: Dict[str, UUID], rpc_hmac_secret: Optional[str] = None) -> None:
        self.token_to_user = token_to_user
        self.rpc_hmac_secret = rpc_hmac_secret or os.environ.get(
            "BACKEND_RPC_HMAC_SECRET", "pytest-backend-rpc-hmac-secret-0001"
        )
        self.used_nonces: set[str] = set()
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.messages: List[Dict[str, Any]] = []
        self.observed_requests: List[Dict[str, str]] = []

    def _verify_backend_proof(
        self,
        uid: UUID,
        operation: str,
        doc_id: str,
        payload_digest: str,
        body: Dict[str, Any],
    ) -> Optional[httpx.Response]:
        proof = body.get("p_backend_proof")
        nonce = body.get("p_proof_nonce")
        expires_at = body.get("p_proof_expires_at")
        if not proof or not nonce or expires_at is None or not (16 <= len(str(nonce).strip()) <= 128):
            return httpx.Response(403, json={"code": "42501", "message": "Backend authorization proof required"})
        now_epoch = int(time.time())
        if int(expires_at) < now_epoch or int(expires_at) > now_epoch + 300:
            return httpx.Response(403, json={"code": "42501", "message": "Backend authorization proof expired"})
        message = f"{uid}|{operation}|{doc_id}|{payload_digest}|{expires_at}|{nonce}"
        expected = hmac.new(
            self.rpc_hmac_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, str(proof).strip().lower()):
            return httpx.Response(403, json={"code": "42501", "message": "Invalid backend authorization proof"})
        if nonce in self.used_nonces:
            return httpx.Response(403, json={"code": "42501", "message": "Backend authorization proof has already been used"})
        self.used_nonces.add(nonce)
        return None

    def _authenticate(self, request: httpx.Request) -> Optional[UUID]:
        if request.headers.get("apikey") != ANON_KEY:
            return None
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return None
        token = auth_header[7:]
        return self.token_to_user.get(token)

    def handle(self, request: httpx.Request) -> httpx.Response:
        auth_header = request.headers.get("authorization", "")
        self.observed_requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "authorization": auth_header,
                "apikey": request.headers.get("apikey", ""),
            }
        )

        uid = self._authenticate(request)
        if uid is None:
            return httpx.Response(401, json={"message": "Invalid JWT"})

        path = request.url.path
        qs = parse_qs(request.url.query.decode("utf-8")) if request.url.query else {}
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        now = datetime.now(timezone.utc).isoformat()

        # ---------------- RPCs ----------------
        if path == "/rest/v1/rpc/create_user_document" and request.method == "POST":
            doc_id = body.get("p_document_id")
            if not doc_id:
                return httpx.Response(400, json={"code": "22023", "message": "Document ID is required"})
            storage_path = body["p_storage_path"]
            digest = compute_create_document_payload_digest(
                storage_path=storage_path,
                filename=body["p_filename"],
                file_extension=body["p_file_extension"],
                mime_type=body["p_mime_type"],
                content_hash=body["p_content_hash"],
                file_size_bytes=body["p_file_size_bytes"],
                processing_status=body.get("p_processing_status", "pending"),
                embedding_model=body.get("p_embedding_model"),
            )
            proof_err = self._verify_backend_proof(uid, "create_user_document", str(doc_id), digest, body)
            if proof_err is not None:
                return proof_err
            if storage_path.split("/")[0] != str(uid):
                return httpx.Response(400, json={"code": "23514", "message": "documents_storage_path_scoped"})
            for existing in self.documents.values():
                if existing["owner_id"] == str(uid) and existing["content_hash"] == body["p_content_hash"]:
                    return httpx.Response(409, json={"code": "23505", "message": "documents_owner_hash_unique"})
            row = {
                "id": doc_id,
                "owner_id": str(uid),
                "storage_path": storage_path,
                "filename": body["p_filename"],
                "file_extension": body["p_file_extension"],
                "mime_type": body["p_mime_type"],
                "content_hash": body["p_content_hash"],
                "file_size_bytes": body["p_file_size_bytes"],
                "processing_status": body.get("p_processing_status", "pending"),
                "embedding_model": body.get("p_embedding_model"),
                "total_pages": 0,
                "total_chunks": 0,
                "error_message": None,
                "created_at": now,
                "updated_at": now,
            }
            self.documents[doc_id] = row
            return httpx.Response(200, json=row)

        if path == "/rest/v1/rpc/update_user_document_status" and request.method == "POST":
            doc_id = body["p_document_id"]
            digest = compute_update_status_payload_digest(
                processing_status=body["p_processing_status"],
                total_pages=body.get("p_total_pages"),
                total_chunks=body.get("p_total_chunks"),
                embedding_model=body.get("p_embedding_model"),
                error_message=body.get("p_error_message"),
            )
            proof_err = self._verify_backend_proof(uid, "update_user_document_status", str(doc_id), digest, body)
            if proof_err is not None:
                return proof_err
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != str(uid):
                return httpx.Response(200, json=[])
            row["processing_status"] = body["p_processing_status"]
            if body.get("p_total_pages") is not None:
                row["total_pages"] = body["p_total_pages"]
            if body.get("p_total_chunks") is not None:
                row["total_chunks"] = body["p_total_chunks"]
            if body.get("p_embedding_model") is not None:
                row["embedding_model"] = body["p_embedding_model"]
            row["error_message"] = None if row["processing_status"] == "ready" else body.get("p_error_message")
            row["updated_at"] = now
            return httpx.Response(200, json=[row])

        if path == "/rest/v1/rpc/delete_user_document" and request.method == "POST":
            doc_id = body.get("p_document_id")
            if not doc_id:
                return httpx.Response(400, json={"code": "22023", "message": "Document ID is required"})
            digest = compute_delete_document_payload_digest(UUID(str(doc_id)))
            proof_err = self._verify_backend_proof(uid, "delete_user_document", str(doc_id), digest, body)
            if proof_err is not None:
                return proof_err
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != str(uid):
                return httpx.Response(200, json=[])
            del self.documents[doc_id]
            return httpx.Response(200, json=[row])

        if path == "/rest/v1/rpc/record_assistant_message" and request.method == "POST":
            cid = body.get("p_conversation_id")
            if not cid:
                return httpx.Response(400, json={"code": "22023", "message": "Conversation ID is required"})
            content = body.get("p_content")
            if not isinstance(content, str) or not content.strip():
                return httpx.Response(400, json={"code": "22023", "message": "Assistant message content cannot be empty"})
            metadata = body.get("p_metadata", {})
            citations = body.get("p_citations", [])
            if not isinstance(metadata, dict):
                return httpx.Response(400, json={"code": "22023", "message": "Message metadata must be a JSON object"})
            if not isinstance(citations, list):
                return httpx.Response(400, json={"code": "22023", "message": "Message citations must be a JSON array"})
            digest = compute_record_assistant_message_payload_digest(
                content=content,
                metadata=metadata,
                citations=citations,
            )
            proof_err = self._verify_backend_proof(uid, "record_assistant_message", str(cid), digest, body)
            if proof_err is not None:
                return proof_err
            conv = self.conversations.get(cid)
            if not conv or conv["owner_id"] != str(uid):
                return httpx.Response(403, json={"code": "42501", "message": "Conversation not found or not owned by caller"})
            msg = {
                "id": str(uuid4()),
                "conversation_id": cid,
                "owner_id": str(uid),
                "role": "assistant",
                "content": content,
                "raw_query": None,
                "rewritten_query": None,
                "metadata": metadata,
                "citations": citations,
                "created_at": now,
            }
            self.messages.append(msg)
            conv["updated_at"] = now
            return httpx.Response(200, json=msg)

        # ---------------- Documents Table (SELECT only) ----------------
        if path == "/rest/v1/documents":
            if request.method != "GET":
                return httpx.Response(403, json={"code": "42501", "message": "permission denied for table documents"})
            rows = [r for r in self.documents.values() if r["owner_id"] == str(uid)]
            if "id" in qs:
                target_id = qs["id"][0].removeprefix("eq.")
                rows = [r for r in rows if r["id"] == target_id]
            return httpx.Response(200, json=rows)

        # ---------------- Conversations Table ----------------
        if path == "/rest/v1/conversations":
            if request.method == "POST":
                if body.get("owner_id") != str(uid):
                    return httpx.Response(403, json={"code": "42501", "message": "RLS violation"})
                cid = str(uuid4())
                row = {
                    "id": cid,
                    "owner_id": str(uid),
                    "title": body.get("title", "New Conversation"),
                    "created_at": now,
                    "updated_at": now,
                }
                self.conversations[cid] = row
                return httpx.Response(201, json=[row])

            rows = [r for r in self.conversations.values() if r["owner_id"] == str(uid)]
            if "id" in qs:
                target_id = qs["id"][0].removeprefix("eq.")
                rows = [r for r in rows if r["id"] == target_id]

            if request.method == "GET":
                return httpx.Response(200, json=rows)
            if request.method == "PATCH":
                updated = []
                for r in rows:
                    r["title"] = body["title"]
                    r["updated_at"] = now
                    updated.append(r)
                return httpx.Response(200, json=updated)
            if request.method == "DELETE":
                deleted = list(rows)
                for r in deleted:
                    self.conversations.pop(r["id"], None)
                    self.messages = [m for m in self.messages if m["conversation_id"] != r["id"]]
                return httpx.Response(200, json=deleted)

        # ---------------- Messages Table ----------------
        if path == "/rest/v1/messages":
            if request.method == "GET":
                rows = [m for m in self.messages if m["owner_id"] == str(uid)]
                if "conversation_id" in qs:
                    cid = qs["conversation_id"][0].removeprefix("eq.")
                    rows = [m for m in rows if m["conversation_id"] == cid]
                return httpx.Response(200, json=rows)

            if request.method == "POST":
                cid = body.get("conversation_id")
                conv = self.conversations.get(cid)
                # Enforce RLS: owner_id == uid, role == 'user', and owned conversation exists
                if (
                    body.get("owner_id") != str(uid)
                    or body.get("role") != "user"
                    or not conv
                    or conv["owner_id"] != str(uid)
                ):
                    return httpx.Response(403, json={"code": "42501", "message": "new row violates row-level security policy for table messages"})
                msg = {
                    "id": str(uuid4()),
                    "conversation_id": cid,
                    "owner_id": str(uid),
                    "role": "user",
                    "content": body["content"],
                    "raw_query": body.get("raw_query"),
                    "rewritten_query": body.get("rewritten_query"),
                    "metadata": body.get("metadata", {}),
                    "citations": body.get("citations", []),
                    "created_at": now,
                }
                self.messages.append(msg)
                return httpx.Response(201, json=[msg])

        return httpx.Response(404, json={"message": "Not found"})


@pytest.fixture(autouse=True)
def _clear_jwks():
    clear_jwks_client_cache()
    yield
    clear_jwks_client_cache()


class TestSupabaseClientAndDI:
    """Authentication, JWT forwarding, pooling isolation, and secret hygiene tests."""

    def test_dependency_rejects_missing_and_invalid_jwt_and_forwards_user_jwt(self, caplog):
        priv, jwk = _make_keypair("di-ec-1")
        token_a = _issue_jwt(priv, USER_A_ID, kid="di-ec-1")
        token_b = _issue_jwt(priv, USER_B_ID, kid="di-ec-1")
        engine = MockSupabaseRLSEngine({token_a: USER_A_ID, token_b: USER_B_ID})
        settings = _make_settings()

        mini_app = FastAPI()
        mini_app.dependency_overrides[get_settings] = lambda: settings

        @mini_app.get("/_test/docs")
        async def list_docs(repo: DocumentRepository = Depends(get_document_repository)):
            docs = await repo.list_documents(repo._client.owner_id)
            return {"owner": str(repo._client.owner_id), "count": len(docs)}

        async_mock_client = httpx.AsyncClient(transport=httpx.MockTransport(engine.handle))

        with caplog.at_level(logging.DEBUG, logger="retrieva_ai"):
            with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}), patch(
                "backend.app.core.supabase_client.get_supabase_http_client",
                return_value=async_mock_client,
            ):
                with TestClient(mini_app) as client:
                    # Missing JWT
                    res_missing = client.get("/_test/docs")
                    assert res_missing.status_code == 401

                    # Invalid JWT
                    res_invalid = client.get(
                        "/_test/docs",
                        headers={"Authorization": "Bearer invalid.jwt.token"},
                    )
                    assert res_invalid.status_code == 401

                    # Valid User A JWT
                    res_a = client.get(
                        "/_test/docs",
                        headers={"Authorization": f"Bearer {token_a}"},
                    )
                    assert res_a.status_code == 200
                    assert res_a.json()["owner"] == str(USER_A_ID)

                    # Valid User B JWT
                    res_b = client.get(
                        "/_test/docs",
                        headers={"Authorization": f"Bearer {token_b}"},
                    )
                    assert res_b.status_code == 200
                    assert res_b.json()["owner"] == str(USER_B_ID)

        assert engine.observed_requests[0]["authorization"] == f"Bearer {token_a}"
        assert engine.observed_requests[1]["authorization"] == f"Bearer {token_b}"
        assert token_a not in caplog.text
        assert token_b not in caplog.text
        assert ANON_KEY not in caplog.text

    @pytest.mark.asyncio
    async def test_concurrent_requests_never_exchange_authorization_headers(self):
        engine = MockSupabaseRLSEngine({"jwt-user-a": USER_A_ID, "jwt-user-b": USER_B_ID})
        async with httpx.AsyncClient(transport=httpx.MockTransport(engine.handle)) as shared_pool:
            # Shared pool client must carry zero auth/apikey headers
            assert "authorization" not in {k.lower() for k in shared_pool.headers}
            assert "apikey" not in {k.lower() for k in shared_pool.headers}

            client_a = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-user-a", USER_A_ID, http_client=shared_pool)
            client_b = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-user-b", USER_B_ID, http_client=shared_pool)

            assert "jwt-user-a" not in repr(client_a)
            assert ANON_KEY not in repr(client_a)

            conv_repo_a = SupabaseConversationRepository(client_a)
            conv_repo_b = SupabaseConversationRepository(client_b)

            async def run_user_a(i: int):
                c = await conv_repo_a.create_conversation(USER_A_ID, f"A-{i}")
                fetched = await conv_repo_a.get_conversation(c.id, USER_A_ID)
                assert fetched is not None
                assert fetched.owner_id == USER_A_ID

            async def run_user_b(i: int):
                c = await conv_repo_b.create_conversation(USER_B_ID, f"B-{i}")
                fetched = await conv_repo_b.get_conversation(c.id, USER_B_ID)
                assert fetched is not None
                assert fetched.owner_id == USER_B_ID

            await asyncio.gather(*[run_user_a(i) for i in range(10)], *[run_user_b(i) for i in range(10)])

            for req in engine.observed_requests:
                assert req["apikey"] == ANON_KEY
                assert req["authorization"] in {"Bearer jwt-user-a", "Bearer jwt-user-b"}

        # Every conversation created under User A belongs exclusively to User A, and vice versa
        a_convs = [c for c in engine.conversations.values() if c["owner_id"] == str(USER_A_ID)]
        b_convs = [c for c in engine.conversations.values() if c["owner_id"] == str(USER_B_ID)]
        assert len(a_convs) == 10
        assert len(b_convs) == 10

    def test_no_service_role_credential_referenced_in_backend_code(self):
        root = Path("backend/app")
        for py_file in root.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "service_role" not in content.lower(), f"service_role found in {py_file}"

        with pytest.raises(SupabaseConfigError):
            SupabaseClient(SUPABASE_URL, "", "jwt-user-a", USER_A_ID)


class TestDocumentRepositoryIsolation:
    """Owner-scoped DocumentRepository CRUD and cross-tenant isolation tests."""

    @pytest.mark.asyncio
    async def test_document_crud_and_cross_tenant_isolation(self):
        engine = MockSupabaseRLSEngine({"jwt-a": USER_A_ID, "jwt-b": USER_B_ID})
        async with httpx.AsyncClient(transport=httpx.MockTransport(engine.handle)) as pool:
            client_a = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-a", USER_A_ID, http_client=pool)
            client_b = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-b", USER_B_ID, http_client=pool)
            repo_a = DocumentRepository(client_a)
            repo_b = DocumentRepository(client_b)

            doc_a = await repo_a.create_document(
                owner_id=USER_A_ID,
                filename="report.pdf",
                file_extension=".pdf",
                mime_type="application/pdf",
                content_hash="a" * 64,
                file_size_bytes=2048,
                processing_status="uploading",
            )
            assert isinstance(doc_a.id, UUID)
            assert doc_a.owner_id == USER_A_ID

            # User A can get & list own document
            fetched_a = await repo_a.get_document(doc_a.id, USER_A_ID)
            assert fetched_a is not None and fetched_a.id == doc_a.id
            assert [d.id for d in await repo_a.list_documents(USER_A_ID)] == [doc_a.id]

            # User B cannot see, update, list, or delete User A's document
            assert await repo_b.get_document(doc_a.id, USER_B_ID) is None
            assert await repo_b.list_documents(USER_B_ID) == []
            assert await repo_b.update_document_status(doc_a.id, USER_B_ID, "failed", error_message="pwned") is None
            assert await repo_b.delete_document(doc_a.id, USER_B_ID) is False

            # User A's document remains intact and updatable by User A
            updated_a = await repo_a.update_document_status(
                doc_a.id,
                USER_A_ID,
                "ready",
                total_pages=3,
                total_chunks=12,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            )
            assert updated_a is not None
            assert updated_a.processing_status == "ready"
            assert updated_a.total_chunks == 12

            # Mismatched owner_id in repository call is rejected immediately
            with pytest.raises(SupabaseForbiddenError):
                await repo_b.get_document(doc_a.id, USER_A_ID)

            # User A can delete own document
            assert await repo_a.delete_document(doc_a.id, USER_A_ID) is True
            assert await repo_a.get_document(doc_a.id, USER_A_ID) is None


class TestConversationAndMessageRepositoryIsolation:
    """Owner-scoped conversation & message repository isolation and RLS enforcement tests."""

    @pytest.mark.asyncio
    async def test_conversation_and_message_isolation_and_rls_enforcement(self):
        engine = MockSupabaseRLSEngine({"jwt-a": USER_A_ID, "jwt-b": USER_B_ID})
        async with httpx.AsyncClient(transport=httpx.MockTransport(engine.handle)) as pool:
            client_a = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-a", USER_A_ID, http_client=pool)
            client_b = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-b", USER_B_ID, http_client=pool)

            conv_a_repo = SupabaseConversationRepository(client_a)
            conv_b_repo = SupabaseConversationRepository(client_b)
            msg_a_repo = SupabaseMessageRepository(client_a)
            msg_b_repo = SupabaseMessageRepository(client_b)

            conv_a = await conv_a_repo.create_conversation(USER_A_ID, "Private A")
            conv_b = await conv_b_repo.create_conversation(USER_B_ID, "Private B")

            # User A inserts and lists own messages
            msg_a = await msg_a_repo.insert_user_message(
                conversation_id=conv_a.id,
                owner_id=USER_A_ID,
                content="What is in my PDF?",
                raw_query="What is in my PDF?",
                rewritten_query="What is in my PDF?",
            )
            assert msg_a.role == "user"
            assert len(await msg_a_repo.list_messages(conv_a.id, USER_A_ID)) == 1

            # User A cannot read, list, rename, or delete User B's conversation
            assert await conv_a_repo.get_conversation(conv_b.id, USER_A_ID) is None
            assert [c.id for c in await conv_a_repo.list_conversations(USER_A_ID)] == [conv_a.id]
            assert await conv_a_repo.update_conversation_title(conv_b.id, USER_A_ID, "Hijacked") is None
            assert await conv_a_repo.delete_conversation(conv_b.id, USER_A_ID) is False
            assert await msg_a_repo.list_messages(conv_b.id, USER_A_ID) == []

            # User B's conversation title is unaffected
            fetched_b = await conv_b_repo.get_conversation(conv_b.id, USER_B_ID)
            assert fetched_b is not None and fetched_b.title == "Private B"

            # User A cannot insert a user message into User B's conversation (RLS 403)
            with pytest.raises(SupabaseForbiddenError):
                await msg_a_repo.insert_user_message(
                    conversation_id=conv_b.id,
                    owner_id=USER_A_ID,
                    content="Cross-tenant injection",
                )

            # Direct assistant-message insertion cannot bypass existing RLS (RLS 403)
            with pytest.raises(SupabaseForbiddenError):
                await client_a.insert(
                    "messages",
                    payload={
                        "conversation_id": str(conv_a.id),
                        "owner_id": str(USER_A_ID),
                        "role": "assistant",
                        "content": "Spoofed assistant reply",
                        "citations": [],
                    },
                )

            # Dependency factories wire repositories cleanly
            assert isinstance(get_document_repository(client_a), DocumentRepository)
            assert isinstance(get_supabase_conversation_repository(client_a), SupabaseConversationRepository)
            assert isinstance(get_message_repository(client_a), SupabaseMessageRepository)


class TestBackendOnlyRPCAuthorization:
    """Tests verifying backend-only HMAC proof enforcement on document lifecycle RPCs."""

    @pytest.mark.asyncio
    async def test_direct_client_rpc_calls_denied_and_proof_bindings_enforced(self, caplog):
        secret = "unit-test-backend-rpc-hmac-secret-001"
        engine = MockSupabaseRLSEngine({"jwt-a": USER_A_ID, "jwt-b": USER_B_ID}, rpc_hmac_secret=secret)

        async with httpx.AsyncClient(transport=httpx.MockTransport(engine.handle)) as pool:
            client_a = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-a", USER_A_ID, http_client=pool)
            client_b = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-b", USER_B_ID, http_client=pool)
            repo_a = DocumentRepository(client_a, rpc_hmac_secret=secret)
            repo_b = DocumentRepository(client_b, rpc_hmac_secret=secret)

            doc_id = uuid4()
            storage_path = f"{USER_A_ID}/{doc_id}/report.pdf"
            base_create_payload = {
                "p_document_id": str(doc_id),
                "p_storage_path": storage_path,
                "p_filename": "report.pdf",
                "p_file_extension": ".pdf",
                "p_mime_type": "application/pdf",
                "p_content_hash": "b" * 64,
                "p_file_size_bytes": 1024,
                "p_processing_status": "uploading",
                "p_embedding_model": None,
            }

            # 1. Direct authenticated RPC calls without backend proof are denied
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc("create_user_document", dict(base_create_payload))

            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "update_user_document_status",
                    {"p_document_id": str(doc_id), "p_processing_status": "ready"},
                )

            # 2. Invalid / altered proof is denied
            create_digest = compute_create_document_payload_digest(
                storage_path=storage_path,
                filename="report.pdf",
                file_extension=".pdf",
                mime_type="application/pdf",
                content_hash="b" * 64,
                file_size_bytes=1024,
                processing_status="uploading",
                embedding_model=None,
            )
            proof_a, nonce_a, exp_a = build_backend_rpc_proof(
                secret=secret,
                owner_id=USER_A_ID,
                operation="create_user_document",
                document_id=doc_id,
                payload_digest=create_digest,
            )
            tampered_proof = ("0" if proof_a[0] != "0" else "1") + proof_a[1:]
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "create_user_document",
                    {
                        **base_create_payload,
                        "p_backend_proof": tampered_proof,
                        "p_proof_nonce": nonce_a,
                        "p_proof_expires_at": exp_a,
                    },
                )

            # 3. Proof for User A cannot authorize User B
            with pytest.raises(SupabaseForbiddenError):
                await client_b.rpc(
                    "create_user_document",
                    {
                        **base_create_payload,
                        "p_storage_path": f"{USER_B_ID}/{doc_id}/report.pdf",
                        "p_backend_proof": proof_a,
                        "p_proof_nonce": nonce_a,
                        "p_proof_expires_at": exp_a,
                    },
                )

            # 4. Proof for create_user_document cannot authorize update_user_document_status
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "update_user_document_status",
                    {
                        "p_document_id": str(doc_id),
                        "p_processing_status": "ready",
                        "p_backend_proof": proof_a,
                        "p_proof_nonce": nonce_a,
                        "p_proof_expires_at": exp_a,
                    },
                )

            # 5. Expired proof is denied
            expired_proof, expired_nonce, expired_at = build_backend_rpc_proof(
                secret=secret,
                owner_id=USER_A_ID,
                operation="create_user_document",
                document_id=doc_id,
                payload_digest=create_digest,
                ttl_seconds=-10,
            )
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "create_user_document",
                    {
                        **base_create_payload,
                        "p_backend_proof": expired_proof,
                        "p_proof_nonce": expired_nonce,
                        "p_proof_expires_at": expired_at,
                    },
                )

            # 6. Valid single-use proof succeeds once and replay is denied
            with caplog.at_level(logging.DEBUG, logger="retrieva_ai"):
                created = await client_a.rpc(
                    "create_user_document",
                    {
                        **base_create_payload,
                        "p_backend_proof": proof_a,
                        "p_proof_nonce": nonce_a,
                        "p_proof_expires_at": exp_a,
                    },
                )
            assert created["owner_id"] == str(USER_A_ID)
            assert secret not in caplog.text
            assert proof_a not in caplog.text
            assert nonce_a not in caplog.text

            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "create_user_document",
                    {
                        **base_create_payload,
                        "p_backend_proof": proof_a,
                        "p_proof_nonce": nonce_a,
                        "p_proof_expires_at": exp_a,
                    },
                )

            # 7. Tampering status payload invalidates update proof
            update_digest = compute_update_status_payload_digest(
                processing_status="indexing",
                total_pages=None,
                total_chunks=None,
                embedding_model=None,
                error_message=None,
            )
            upd_proof, upd_nonce, upd_exp = build_backend_rpc_proof(
                secret=secret,
                owner_id=USER_A_ID,
                operation="update_user_document_status",
                document_id=doc_id,
                payload_digest=update_digest,
            )
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "update_user_document_status",
                    {
                        "p_document_id": str(doc_id),
                        "p_processing_status": "ready",
                        "p_backend_proof": upd_proof,
                        "p_proof_nonce": upd_nonce,
                        "p_proof_expires_at": upd_exp,
                    },
                )

            # 8. Backend-mediated repository call succeeds with JWT owner isolation preserved
            updated = await repo_a.update_document_status(doc_id, USER_A_ID, "ready", total_pages=2, total_chunks=6)
            assert updated is not None and updated.owner_id == USER_A_ID and updated.processing_status == "ready"
            assert await repo_b.get_document(doc_id, USER_B_ID) is None

    @pytest.mark.asyncio
    async def test_missing_backend_hmac_secret_fails_closed(self):
        engine = MockSupabaseRLSEngine({"jwt-a": USER_A_ID})
        async with httpx.AsyncClient(transport=httpx.MockTransport(engine.handle)) as pool:
            client_a = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-a", USER_A_ID, http_client=pool)
            repo = DocumentRepository(client_a, rpc_hmac_secret="")
            assert "jwt-a" not in repr(repo)

            with pytest.raises(SupabaseConfigError):
                await repo.create_document(
                    owner_id=USER_A_ID,
                    filename="missing-secret.pdf",
                    file_extension=".pdf",
                    mime_type="application/pdf",
                    content_hash="c" * 64,
                    file_size_bytes=512,
                )

            with pytest.raises(SupabaseConfigError):
                await repo.update_document_status(uuid4(), USER_A_ID, "ready")

            with pytest.raises(SupabaseConfigError):
                await repo.delete_document(uuid4(), USER_A_ID)

            msg_repo = SupabaseMessageRepository(client_a, rpc_hmac_secret="")
            assert "jwt-a" not in repr(msg_repo)
            with pytest.raises(SupabaseConfigError):
                await msg_repo.insert_assistant_message(uuid4(), USER_A_ID, "Blocked without secret")

        with pytest.raises(ValueError):
            Settings(
                APP_ENV="production",
                SUPABASE_URL=SUPABASE_URL,
                SUPABASE_ANON_KEY=ANON_KEY,
                BACKEND_RPC_HMAC_SECRET=None,
            )

    @pytest.mark.asyncio
    async def test_delete_and_assistant_message_rpcs_require_valid_bound_proofs(self, caplog):
        secret = "unit-test-complete-rpc-hmac-secret-002"
        engine = MockSupabaseRLSEngine({"jwt-a": USER_A_ID, "jwt-b": USER_B_ID}, rpc_hmac_secret=secret)

        async with httpx.AsyncClient(transport=httpx.MockTransport(engine.handle)) as pool:
            client_a = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-a", USER_A_ID, http_client=pool)
            client_b = SupabaseClient(SUPABASE_URL, ANON_KEY, "jwt-b", USER_B_ID, http_client=pool)
            doc_repo_a = DocumentRepository(client_a, rpc_hmac_secret=secret)
            doc_repo_b = DocumentRepository(client_b, rpc_hmac_secret=secret)
            conv_repo_a = SupabaseConversationRepository(client_a)
            conv_repo_b = SupabaseConversationRepository(client_b)
            msg_repo_a = SupabaseMessageRepository(client_a, rpc_hmac_secret=secret)
            msg_repo_b = SupabaseMessageRepository(client_b, rpc_hmac_secret=secret)

            doc_a = await doc_repo_a.create_document(
                owner_id=USER_A_ID,
                filename="contract.pdf",
                file_extension=".pdf",
                mime_type="application/pdf",
                content_hash="d" * 64,
                file_size_bytes=2048,
                processing_status="ready",
            )
            conv_a = await conv_repo_a.create_conversation(USER_A_ID, "User A Chat")
            conv_b = await conv_repo_b.create_conversation(USER_B_ID, "User B Chat")

            # 1. Direct delete_user_document without proof is denied
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc("delete_user_document", {"p_document_id": str(doc_a.id)})

            # 2. Direct record_assistant_message without proof is denied
            base_msg_payload = {
                "p_conversation_id": str(conv_a.id),
                "p_content": "Answer [Source 1].",
                "p_metadata": {"retrieval_mode": "hybrid"},
                "p_citations": [{"document_id": str(doc_a.id), "page_number": 1}],
            }
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc("record_assistant_message", dict(base_msg_payload))

            # 3. Invalid, expired, cross-user, and cross-operation proofs fail on delete_user_document
            del_digest = compute_delete_document_payload_digest(doc_a.id)
            del_proof, del_nonce, del_exp = build_backend_rpc_proof(
                secret=secret,
                owner_id=USER_A_ID,
                operation="delete_user_document",
                target_id=doc_a.id,
                payload_digest=del_digest,
            )
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "delete_user_document",
                    {
                        "p_document_id": str(doc_a.id),
                        "p_backend_proof": "f" * 64,
                        "p_proof_nonce": del_nonce,
                        "p_proof_expires_at": del_exp,
                    },
                )
            exp_del_proof, exp_del_nonce, exp_del_exp = build_backend_rpc_proof(
                secret=secret,
                owner_id=USER_A_ID,
                operation="delete_user_document",
                target_id=doc_a.id,
                payload_digest=del_digest,
                ttl_seconds=-5,
            )
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "delete_user_document",
                    {
                        "p_document_id": str(doc_a.id),
                        "p_backend_proof": exp_del_proof,
                        "p_proof_nonce": exp_del_nonce,
                        "p_proof_expires_at": exp_del_exp,
                    },
                )
            with pytest.raises(SupabaseForbiddenError):
                await client_b.rpc(
                    "delete_user_document",
                    {
                        "p_document_id": str(doc_a.id),
                        "p_backend_proof": del_proof,
                        "p_proof_nonce": del_nonce,
                        "p_proof_expires_at": del_exp,
                    },
                )
            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "update_user_document_status",
                    {
                        "p_document_id": str(doc_a.id),
                        "p_processing_status": "failed",
                        "p_backend_proof": del_proof,
                        "p_proof_nonce": del_nonce,
                        "p_proof_expires_at": del_exp,
                    },
                )

            # 4. Modifying content, metadata, or citations invalidates record_assistant_message proof
            msg_digest = compute_record_assistant_message_payload_digest(
                content=base_msg_payload["p_content"],
                metadata=base_msg_payload["p_metadata"],
                citations=base_msg_payload["p_citations"],
            )
            msg_proof, msg_nonce, msg_exp = build_backend_rpc_proof(
                secret=secret,
                owner_id=USER_A_ID,
                operation="record_assistant_message",
                target_id=conv_a.id,
                payload_digest=msg_digest,
            )
            for tampered_field in (
                {"p_content": "Tampered answer"},
                {"p_metadata": {"retrieval_mode": "dense"}},
                {"p_citations": [{"document_id": str(doc_a.id), "page_number": 99}]},
            ):
                with pytest.raises(SupabaseForbiddenError):
                    await client_a.rpc(
                        "record_assistant_message",
                        {
                            **base_msg_payload,
                            **tampered_field,
                            "p_backend_proof": msg_proof,
                            "p_proof_nonce": msg_nonce,
                            "p_proof_expires_at": msg_exp,
                        },
                    )

            # Cross-user and cross-operation proof reuse denied for record_assistant_message
            with pytest.raises(SupabaseForbiddenError):
                await client_b.rpc(
                    "record_assistant_message",
                    {
                        **base_msg_payload,
                        "p_backend_proof": msg_proof,
                        "p_proof_nonce": msg_nonce,
                        "p_proof_expires_at": msg_exp,
                    },
                )

            # 5. Valid record_assistant_message proof succeeds once and replay is denied; secrets never logged
            with caplog.at_level(logging.DEBUG, logger="retrieva_ai"):
                recorded = await client_a.rpc(
                    "record_assistant_message",
                    {
                        **base_msg_payload,
                        "p_backend_proof": msg_proof,
                        "p_proof_nonce": msg_nonce,
                        "p_proof_expires_at": msg_exp,
                    },
                )
            assert recorded["owner_id"] == str(USER_A_ID)
            assert recorded["role"] == "assistant"
            assert secret not in caplog.text
            assert msg_proof not in caplog.text
            assert msg_nonce not in caplog.text

            with pytest.raises(SupabaseForbiddenError):
                await client_a.rpc(
                    "record_assistant_message",
                    {
                        **base_msg_payload,
                        "p_backend_proof": msg_proof,
                        "p_proof_nonce": msg_nonce,
                        "p_proof_expires_at": msg_exp,
                    },
                )

            # 6. Repository callers succeed for owner and preserve cross-tenant isolation
            msg_a2 = await msg_repo_a.insert_assistant_message(
                conversation_id=conv_a.id,
                owner_id=USER_A_ID,
                content="Repository-signed assistant reply.",
                metadata={"retrieval_mode": "bm25"},
                citations=[{"document_id": str(doc_a.id), "page_number": 2}],
            )
            assert msg_a2.owner_id == USER_A_ID
            assert msg_a2.role == "assistant"

            with pytest.raises(SupabaseForbiddenError):
                await msg_repo_b.insert_assistant_message(
                    conversation_id=conv_a.id,
                    owner_id=USER_B_ID,
                    content="Cross-tenant write attempt",
                )
            assert await msg_repo_a.list_messages(conv_b.id, USER_A_ID) == []

            assert await doc_repo_b.delete_document(doc_a.id, USER_B_ID) is False
            assert await doc_repo_a.delete_document(doc_a.id, USER_A_ID) is True
            assert await doc_repo_a.get_document(doc_a.id, USER_A_ID) is None

    def test_canonical_jsonb_handles_edge_cases_and_key_order_independently(self):
        payload_1 = {
            "z_unicode": "Héllo 🚀\n\"quoted\"\\slash/\x7f",
            "a_numbers": [1, 1.0, 0.00001, 1e20, -0.0, 0],
            "m_flags": {"b": False, "a": True, "n": None},
            "nested": [{"k2": "v2", "k1": "v1"}],
        }
        payload_2 = {
            "nested": [{"k1": "v1", "k2": "v2"}],
            "m_flags": {"n": None, "a": True, "b": False},
            "a_numbers": [1, 1, 1e-5, 100000000000000000000, 0.0, 0],
            "z_unicode": "Héllo 🚀\n\"quoted\"\\slash/\x7f",
        }
        c1 = canonical_jsonb(payload_1)
        c2 = canonical_jsonb(payload_2)
        assert c1 == c2
        assert "1e-05" not in c1
        assert "1e+20" not in c1
        assert "d:-0" not in c1
        assert (
            compute_record_assistant_message_payload_digest(
                content="Answer",
                metadata=payload_1,
                citations=payload_1["nested"],
            )
            == compute_record_assistant_message_payload_digest(
                content="Answer",
                metadata=payload_2,
                citations=payload_2["nested"],
            )
        )
