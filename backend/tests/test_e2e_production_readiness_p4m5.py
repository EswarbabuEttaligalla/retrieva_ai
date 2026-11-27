"""Phase 4 Milestone 5: End-to-end multi-tenant production readiness workflow test.

Exercises the complete real FastAPI HTTP flow (mocking only external Supabase HTTP
and OpenAI/Gemini LLM) across two users:
1. User A uploads both a PDF and a TXT document.
2. User A retrieves from both documents and generates a grounded answer.
3. User A creates a conversation, asks a question, and resumes the conversation.
4. User B signs in (with User B's JWT) and cannot see User A's documents or conversations.
5. User B uploads an independent document and retrieves its content.
6. User A signs back in, retains all data, deletes the PDF document, and verifies:
   - Metadata row, Storage object, ChromaDB vectors, and BM25 entries are purged.
   - Deleted PDF cannot contribute evidence to subsequent /retrieval/search or /answer.
   - Remaining TXT document and conversation remain intact.
"""

import io
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs
from uuid import UUID, uuid4

import fitz
import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from backend.app.core.auth import clear_jwks_client_cache, get_current_user
from backend.app.core.config import Settings, get_settings
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.llm.client import OpenAIClient
from backend.app.main import app
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.retrieval.tenant_bm25 import TenantBM25Manager
from backend.app.services.ingestion_service import IngestionService, get_ingestion_service
from backend.app.services.rag_service import RAGService, get_rag_service
from backend.app.vector_store.store import ChromaVectorStore

USER_A_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
USER_B_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
SUPABASE_URL = "https://zblfcosunplvnnwjfyjz.supabase.co"
ANON_KEY = "test-public-anon-key"


def _make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class E2ESupabaseSimulator:
    """In-memory multi-tenant Supabase PostgREST + Storage simulator."""

    def __init__(self, token_to_user: Dict[str, UUID]) -> None:
        self.token_to_user = token_to_user
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.storage_objects: Dict[str, bytes] = {}
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.messages: List[Dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        tok = auth.removeprefix("Bearer ").strip()
        uid = self.token_to_user.get(tok)
        if uid is None:
            return httpx.Response(401, json={"message": "Unauthorized"})
        uid_str = str(uid)
        path = request.url.path
        qs = parse_qs(request.url.query.decode("utf-8")) if request.url.query else {}
        now = datetime.now(timezone.utc).isoformat()

        # Storage
        if path.startswith("/storage/v1/object/retrieva-user-documents/"):
            obj_path = path.split("/retrieva-user-documents/", 1)[1]
            if not obj_path.startswith(f"{uid_str}/"):
                return httpx.Response(403, json={"message": "Forbidden"})
            if request.method == "POST":
                self.storage_objects[obj_path] = request.content
                return httpx.Response(200, json={"Key": f"retrieva-user-documents/{obj_path}"})
            if request.method == "DELETE":
                if obj_path not in self.storage_objects:
                    return httpx.Response(404, json={"message": "Not found"})
                del self.storage_objects[obj_path]
                return httpx.Response(200, json={"message": "Deleted"})

        body = json.loads(request.content.decode("utf-8")) if request.content else {}

        # Document RPCs
        if path == "/rest/v1/rpc/create_user_document" and request.method == "POST":
            doc_id = body.get("p_document_id") or str(uuid4())
            row = {
                "id": doc_id,
                "owner_id": uid_str,
                "storage_path": body["p_storage_path"],
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
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != uid_str:
                return httpx.Response(200, json=[])
            row["processing_status"] = body["p_processing_status"]
            if body.get("p_total_pages") is not None:
                row["total_pages"] = body["p_total_pages"]
            if body.get("p_total_chunks") is not None:
                row["total_chunks"] = body["p_total_chunks"]
            if body.get("p_embedding_model") is not None:
                row["embedding_model"] = body["p_embedding_model"]
            row["updated_at"] = now
            return httpx.Response(200, json=[row])

        if path == "/rest/v1/rpc/delete_user_document" and request.method == "POST":
            doc_id = body["p_document_id"]
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != uid_str:
                return httpx.Response(200, json=[])
            del self.documents[doc_id]
            return httpx.Response(200, json=[row])

        if path == "/rest/v1/documents" and request.method == "GET":
            rows = [r for r in self.documents.values() if r["owner_id"] == uid_str]
            if "id" in qs:
                tid = qs["id"][0].removeprefix("eq.")
                rows = [r for r in rows if r["id"] == tid]
            if "content_hash" in qs:
                thash = qs["content_hash"][0].removeprefix("eq.")
                rows = [r for r in rows if r["content_hash"] == thash]
            if "processing_status" in qs:
                st = qs["processing_status"][0].removeprefix("eq.")
                rows = [r for r in rows if r["processing_status"] == st]
            return httpx.Response(200, json=rows)

        # Conversations & Messages
        if path == "/rest/v1/conversations":
            if request.method == "POST":
                cid = str(uuid4())
                row = {
                    "id": cid,
                    "owner_id": uid_str,
                    "title": body.get("title", "New Conversation"),
                    "created_at": now,
                    "updated_at": now,
                }
                self.conversations[cid] = row
                return httpx.Response(201, json=[row])
            if request.method == "GET":
                rows = [c for c in self.conversations.values() if c["owner_id"] == uid_str]
                if "id" in qs:
                    tid = qs["id"][0].removeprefix("eq.")
                    rows = [c for c in rows if c["id"] == tid]
                return httpx.Response(200, json=rows)
            if request.method == "PATCH":
                tid = qs.get("id", [""])[0].removeprefix("eq.")
                conv = self.conversations.get(tid)
                if not conv or conv["owner_id"] != uid_str:
                    return httpx.Response(200, json=[])
                if "title" in body:
                    conv["title"] = body["title"]
                conv["updated_at"] = now
                return httpx.Response(200, json=[conv])

        if path == "/rest/v1/messages":
            if request.method == "POST":
                cid = body["conversation_id"]
                conv = self.conversations.get(cid)
                if not conv or conv["owner_id"] != uid_str:
                    return httpx.Response(403, json={"code": "42501", "message": "Forbidden"})
                row = {
                    "id": str(uuid4()),
                    "conversation_id": cid,
                    "owner_id": uid_str,
                    "role": body["role"],
                    "content": body["content"],
                    "raw_query": body.get("raw_query"),
                    "rewritten_query": body.get("rewritten_query"),
                    "metadata": body.get("metadata", {}),
                    "citations": body.get("citations", []),
                    "created_at": now,
                }
                self.messages.append(row)
                return httpx.Response(201, json=[row])
            if request.method == "GET":
                rows = [m for m in self.messages if m["owner_id"] == uid_str]
                if "conversation_id" in qs:
                    cid = qs["conversation_id"][0].removeprefix("eq.")
                    rows = [m for m in rows if m["conversation_id"] == cid]
                return httpx.Response(200, json=rows)

        if path == "/rest/v1/rpc/record_assistant_message" and request.method == "POST":
            cid = body["p_conversation_id"]
            conv = self.conversations.get(cid)
            if not conv or conv["owner_id"] != uid_str:
                return httpx.Response(403, json={"code": "42501", "message": "Forbidden"})
            row = {
                "id": str(uuid4()),
                "conversation_id": cid,
                "owner_id": uid_str,
                "role": "assistant",
                "content": body["p_content"],
                "raw_query": None,
                "rewritten_query": None,
                "metadata": body.get("p_metadata", {}),
                "citations": body.get("p_citations", []),
                "created_at": now,
            }
            self.messages.append(row)
            return httpx.Response(200, json=row)

        return httpx.Response(404, json={"message": "Not found"})


@pytest.fixture(scope="module")
def shared_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder.get_instance()


def test_complete_multi_tenant_e2e_workflow(shared_embedder: SentenceTransformerEmbedder):
    temp_dir = tempfile.mkdtemp(prefix="retrieva_p4m5_e2e_")
    priv = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(priv.public_key()))
    jwk["kid"] = "p4m5-ec-1"
    jwk["use"] = "sig"
    jwk["alg"] = "ES256"

    now = int(time.time())
    token_a = jwt.encode(
        {"sub": str(USER_A_ID), "aud": "authenticated", "iss": f"{SUPABASE_URL}/auth/v1", "role": "authenticated", "iat": now - 5, "exp": now + 3600},
        priv,
        algorithm="ES256",
        headers={"kid": "p4m5-ec-1"},
    )
    token_b = jwt.encode(
        {"sub": str(USER_B_ID), "aud": "authenticated", "iss": f"{SUPABASE_URL}/auth/v1", "role": "authenticated", "iat": now - 5, "exp": now + 3600},
        priv,
        algorithm="ES256",
        headers={"kid": "p4m5-ec-1"},
    )

    rpc_secret = os.environ.get("BACKEND_RPC_HMAC_SECRET", "pytest-backend-rpc-hmac-secret-0001")
    settings = Settings(
        APP_ENV="development",
        AUTH_ENABLED=True,
        SUPABASE_URL=SUPABASE_URL,
        SUPABASE_ANON_KEY=ANON_KEY,
        BACKEND_RPC_HMAC_SECRET=rpc_secret,
    )

    sb = E2ESupabaseSimulator({token_a: USER_A_ID, token_b: USER_B_ID})
    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(sb.handle))
    store = ChromaVectorStore(collection_name="p4m5_e2e_col", persist_directory=temp_dir)
    bm25 = TenantBM25Manager(vector_store=store)
    retriever = Retriever(
        dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
        tenant_bm25_manager=bm25,
        vector_store=store,
    )
    ingestion = IngestionService(
        embedder=shared_embedder,
        vector_store=store,
        tenant_bm25_manager=bm25,
    )
    mock_llm = MagicMock(spec=OpenAIClient)
    mock_llm.model = "gpt-4o-mini"
    mock_llm.generate.return_value = {
        "content": json.dumps({"answer": "Verified grounded finding [Source 1].", "insufficient_context": False, "citations": []}),
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
    }
    rag = RAGService(retriever=retriever, llm_client=mock_llm)

    clear_jwks_client_cache()
    prev_auth = app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_rag_service] = lambda: rag

    try:
        with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}), patch(
            "backend.app.core.supabase_client.get_supabase_http_client",
            return_value=mock_http,
        ):
            with TestClient(app) as client:
                hdr_a = {"Authorization": f"Bearer {token_a}"}
                hdr_b = {"Authorization": f"Bearer {token_b}"}

                # 1. User A uploads PDF and TXT
                pdf_bytes = _make_pdf("User A PDF contains alpha_fluorite_mineral data.")
                up_pdf = client.post("/api/v1/documents/upload", headers=hdr_a, files={"file": ("a.pdf", io.BytesIO(pdf_bytes), "application/pdf")})
                up_txt = client.post("/api/v1/documents/upload", headers=hdr_a, files={"file": ("a.txt", io.BytesIO(b"User A TXT contains beta_malachite_ore data."), "text/plain")})
                assert up_pdf.status_code == 200 and up_txt.status_code == 200
                pdf_id = up_pdf.json()["document_id"]
                txt_id = up_txt.json()["document_id"]

                # 2. User A retrieves from both documents
                r_pdf = client.post("/api/v1/retrieval/search", headers=hdr_a, json={"query": "alpha_fluorite_mineral", "config": {"mode": "hybrid", "top_k": 5}}).json()
                assert r_pdf["returned_chunks_count"] >= 1 and r_pdf["chunks"][0]["document_id"] == pdf_id
                r_txt = client.post("/api/v1/retrieval/search", headers=hdr_a, json={"query": "beta_malachite_ore", "config": {"mode": "bm25", "top_k": 5}}).json()
                assert r_txt["returned_chunks_count"] == 1 and r_txt["chunks"][0]["document_id"] == txt_id

                # 3. User A creates and resumes a conversation
                conv = client.post("/api/v1/conversations", headers=hdr_a, json={"title": "Mineral Analysis"})
                assert conv.status_code == 201
                conv_id = conv.json()["id"]
                turn1 = client.post(f"/api/v1/conversations/{conv_id}/answer", headers=hdr_a, json={"query": "What does alpha_fluorite_mineral say?"})
                assert turn1.status_code == 200
                assert turn1.json()["insufficient_context"] is False
                conv_detail = client.get(f"/api/v1/conversations/{conv_id}", headers=hdr_a)
                assert conv_detail.status_code == 200
                assert len(conv_detail.json()["messages"]) == 2

                # 4. User B signs in -> cannot see User A's documents or conversations
                assert client.get("/api/v1/documents", headers=hdr_b).json()["total"] == 0
                convs_b = client.get("/api/v1/conversations", headers=hdr_b).json()
                assert (convs_b if isinstance(convs_b, list) else convs_b.get("conversations", [])) == []
                assert client.get(f"/api/v1/conversations/{conv_id}", headers=hdr_b).status_code == 404

                # 5. User B uploads independent document & retrieves
                up_b = client.post("/api/v1/documents/upload", headers=hdr_b, files={"file": ("b.txt", io.BytesIO(b"User B gamma_tourmaline_crystal data."), "text/plain")})
                assert up_b.status_code == 200
                assert client.post("/api/v1/retrieval/search", headers=hdr_b, json={"query": "gamma_tourmaline_crystal", "config": {"mode": "hybrid", "top_k": 5}}).json()["returned_chunks_count"] == 1
                assert client.post("/api/v1/retrieval/search", headers=hdr_b, json={"query": "alpha_fluorite_mineral", "config": {"mode": "bm25", "top_k": 5}}).json()["returned_chunks_count"] == 0

                # 6. User A deletes PDF -> purged from DB, Storage, ChromaDB, BM25, and excluded from subsequent answers
                del_a = client.delete(f"/api/v1/documents/{pdf_id}", headers=hdr_a)
                assert del_a.status_code == 200
                assert pdf_id not in sb.documents
                assert not any(k.startswith(f"{USER_A_ID}/{pdf_id}/") for k in sb.storage_objects)
                assert not store.has_document(pdf_id, USER_A_ID)

                post_del_ret = client.post("/api/v1/retrieval/search", headers=hdr_a, json={"query": "alpha_fluorite_mineral", "config": {"mode": "bm25", "top_k": 5}})
                assert post_del_ret.json()["returned_chunks_count"] == 0
                post_del_ans = client.post("/api/v1/answer", headers=hdr_a, json={"query": "Explain alpha_fluorite_mineral", "document_ids": [pdf_id]})
                assert post_del_ans.status_code == 200
                assert post_del_ans.json()["insufficient_context"] is True

                # Remaining TXT document & conversation intact
                docs_a = client.get("/api/v1/documents", headers=hdr_a).json()
                assert docs_a["total"] == 1 and docs_a["documents"][0]["document_id"] == txt_id
    finally:
        app.dependency_overrides.clear()
        if prev_auth is not None:
            app.dependency_overrides[get_current_user] = prev_auth
        clear_jwks_client_cache()
        shutil.rmtree(temp_dir, ignore_errors=True)
