"""Phase 4 Milestone: Ingestion failure-path and recovery audit tests.

Covers all 10 failure scenarios in the multi-tenant ingestion pipeline:
1. Unsupported file type and invalid file content.
2. Empty or corrupted PDF/TXT.
3. PDF extraction failure (corrupted body, encrypted, image-only, unexpected error).
4. Chunking failure.
5. Embedding model failure.
6. Chroma vector insertion failure (including partial-write rollback).
7. Supabase Storage upload failure.
8. Supabase document metadata creation/update RPC failure (create, indexing, ready, missing row).
9. Multi-stage partial failure cleanup & cross-tenant isolation during rollback.
10. Retry after failed ingestion (no duplicate records or vectors, searchable via Dense + BM25).
"""

import io
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch
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
from backend.app.core.supabase_client import (
    SupabaseClient,
    SupabaseUnavailableError,
    get_supabase_client,
)
from backend.app.document_processing.chunker import TextChunker
from backend.app.document_processing.exceptions import ChunkingError
from backend.app.document_processing.extractor import DocumentExtractor
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.main import app
from backend.app.repositories.document_repository import DocumentRepository, get_document_repository
from backend.app.repositories.storage_repository import SupabaseStorageRepository, get_storage_repository
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.retrieval.tenant_bm25 import TenantBM25Manager
from backend.app.schemas.retrieval import RetrievalConfiguration, RetrievalMode, RetrievalRequest
from backend.app.services.ingestion_service import IngestionService, get_ingestion_service
from backend.app.vector_store.store import ChromaVectorStore

USER_A_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
USER_B_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
SUPABASE_URL = "https://zblfcosunplvnnwjfyjz.supabase.co"
ANON_KEY = "test-public-anon-key"


def _make_pdf(text: str = "Phase 4 failure audit sample PDF content.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_encrypted_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Secret encrypted content.")
    pdf_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    doc.close()
    return pdf_bytes


class InMemoryMultiTenantSupabase:
    """In-memory PostgREST + Storage simulator with configurable RPC/Storage failure injection."""

    def __init__(self, token_to_user: Dict[str, UUID]) -> None:
        self.token_to_user = token_to_user
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.storage_objects: Dict[str, Dict[str, Any]] = {}
        self.fail_create_document: bool = False
        self.fail_status_on: Optional[str] = None
        self.return_empty_on_ready: bool = False
        self.fail_storage_upload: bool = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        tok = auth.removeprefix("Bearer ").strip()
        uid = self.token_to_user.get(tok)
        if uid is None:
            return httpx.Response(401, json={"message": "Invalid JWT"})
        owner_hdr = str(uid)
        path = request.url.path
        qs = parse_qs(request.url.query.decode("utf-8")) if request.url.query else {}
        now = datetime.now(timezone.utc).isoformat()

        # Storage routes
        if path.startswith("/storage/v1/object/authenticated/retrieva-user-documents/"):
            obj_path = path.split("/retrieva-user-documents/", 1)[1]
            if not obj_path.startswith(f"{owner_hdr}/"):
                return httpx.Response(403, json={"message": "Forbidden"})
            obj = self.storage_objects.get(obj_path)
            if not obj:
                return httpx.Response(404, json={"message": "Object not found"})
            return httpx.Response(200, content=obj["bytes"])

        if path.startswith("/storage/v1/object/retrieva-user-documents/"):
            obj_path = path.split("/retrieva-user-documents/", 1)[1]
            if not obj_path.startswith(f"{owner_hdr}/"):
                return httpx.Response(403, json={"message": "Forbidden"})
            if request.method == "POST":
                if self.fail_storage_upload:
                    return httpx.Response(503, json={"message": "Simulated Storage outage"})
                if obj_path in self.storage_objects and request.headers.get("x-upsert") != "true":
                    return httpx.Response(409, json={"message": "Already exists"})
                self.storage_objects[obj_path] = {
                    "bytes": request.content,
                    "content_type": request.headers.get("content-type"),
                }
                return httpx.Response(200, json={"Key": f"retrieva-user-documents/{obj_path}"})
            if request.method == "DELETE":
                if obj_path not in self.storage_objects:
                    return httpx.Response(404, json={"message": "Object not found"})
                del self.storage_objects[obj_path]
                return httpx.Response(200, json={"message": "Deleted"})

        # PostgREST RPCs
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        if path == "/rest/v1/rpc/create_user_document" and request.method == "POST":
            if self.fail_create_document:
                return httpx.Response(503, json={"message": "Simulated create_user_document RPC outage"})
            doc_id = body.get("p_document_id") or str(uuid4())
            row = {
                "id": doc_id,
                "owner_id": owner_hdr,
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
            target_status = body.get("p_processing_status")
            if self.fail_status_on and target_status == self.fail_status_on:
                return httpx.Response(503, json={"message": f"Simulated RPC failure on status={target_status}"})
            doc_id = body["p_document_id"]
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != owner_hdr:
                return httpx.Response(200, json=[])
            if self.return_empty_on_ready and target_status == "ready":
                return httpx.Response(200, json=[])
            row["processing_status"] = target_status
            if body.get("p_total_pages") is not None:
                row["total_pages"] = body["p_total_pages"]
            if body.get("p_total_chunks") is not None:
                row["total_chunks"] = body["p_total_chunks"]
            if body.get("p_embedding_model") is not None:
                row["embedding_model"] = body["p_embedding_model"]
            if body.get("p_error_message") is not None:
                row["error_message"] = body["p_error_message"]
            row["updated_at"] = now
            return httpx.Response(200, json=[row])

        if path == "/rest/v1/rpc/delete_user_document" and request.method == "POST":
            doc_id = body["p_document_id"]
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != owner_hdr:
                return httpx.Response(200, json=[])
            del self.documents[doc_id]
            return httpx.Response(200, json=[row])

        if path == "/rest/v1/documents" and request.method == "GET":
            rows = [r for r in self.documents.values() if r["owner_id"] == owner_hdr]
            if "id" in qs:
                tid = qs["id"][0].removeprefix("eq.")
                rows = [r for r in rows if r["id"] == tid]
            if "content_hash" in qs:
                thash = qs["content_hash"][0].removeprefix("eq.")
                rows = [r for r in rows if r["content_hash"] == thash]
            return httpx.Response(200, json=rows)

        return httpx.Response(404, json={"message": "Not found"})


@pytest.fixture(scope="module")
def shared_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder.get_instance()


@pytest.fixture()
def audit_env(shared_embedder: SentenceTransformerEmbedder):
    temp_dir = tempfile.mkdtemp(prefix="retrieva_p4_audit_")
    priv = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(priv.public_key()))
    jwk["kid"] = "p4-ec-1"
    jwk["use"] = "sig"
    jwk["alg"] = "ES256"

    now = int(time.time())
    token_a = jwt.encode(
        {
            "sub": str(USER_A_ID),
            "aud": "authenticated",
            "iss": f"{SUPABASE_URL}/auth/v1",
            "role": "authenticated",
            "email": "user_a@example.test",
            "iat": now - 5,
            "exp": now + 3600,
        },
        priv,
        algorithm="ES256",
        headers={"kid": "p4-ec-1"},
    )
    token_b = jwt.encode(
        {
            "sub": str(USER_B_ID),
            "aud": "authenticated",
            "iss": f"{SUPABASE_URL}/auth/v1",
            "role": "authenticated",
            "email": "user_b@example.test",
            "iat": now - 5,
            "exp": now + 3600,
        },
        priv,
        algorithm="ES256",
        headers={"kid": "p4-ec-1"},
    )

    rpc_secret = os.environ.get("BACKEND_RPC_HMAC_SECRET", "pytest-backend-rpc-hmac-secret-0001")
    test_settings = Settings(
        APP_ENV="development",
        AUTH_ENABLED=True,
        SUPABASE_URL=SUPABASE_URL,
        SUPABASE_ANON_KEY=ANON_KEY,
        BACKEND_RPC_HMAC_SECRET=rpc_secret,
    )

    sb_engine = InMemoryMultiTenantSupabase({token_a: USER_A_ID, token_b: USER_B_ID})
    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(sb_engine.handle))
    store = ChromaVectorStore(collection_name="p4_audit_col", persist_directory=temp_dir)
    bm25_manager = TenantBM25Manager(vector_store=store)
    dense = DenseRetriever(embedder=shared_embedder, vector_store=store)
    retriever = Retriever(dense_retriever=dense, tenant_bm25_manager=bm25_manager, vector_store=store)
    ingestion = IngestionService(embedder=shared_embedder, vector_store=store, tenant_bm25_manager=None)

    clear_jwks_client_cache()
    prev_auth = app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_retriever] = lambda: retriever

    with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}), patch(
        "backend.app.core.supabase_client.get_supabase_http_client",
        return_value=mock_http,
    ):
        with TestClient(app) as client:
            yield {
                "client": client,
                "token_a": token_a,
                "token_b": token_b,
                "sb": sb_engine,
                "store": store,
                "bm25": bm25_manager,
                "retriever": retriever,
                "ingestion": ingestion,
            }

    app.dependency_overrides.clear()
    if prev_auth is not None:
        app.dependency_overrides[get_current_user] = prev_auth
    clear_jwks_client_cache()
    shutil.rmtree(temp_dir, ignore_errors=True)


def _upload(client: TestClient, token: str, filename: str, content: bytes, mime: str):
    return client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, io.BytesIO(content), mime)},
    )


def _assert_user_a_clean(env: Dict[str, Any], *, expect_db_failed: bool) -> None:
    """Assert User A has zero Storage objects, zero Chroma vectors, zero BM25 chunks, and no 'ready' DB rows."""
    sb: InMemoryMultiTenantSupabase = env["sb"]
    store: ChromaVectorStore = env["store"]
    bm25: TenantBM25Manager = env["bm25"]

    a_storage = [k for k in sb.storage_objects if k.startswith(f"{USER_A_ID}/")]
    assert a_storage == [], f"Orphaned Storage objects for User A: {a_storage}"
    assert store.count_for_owner(USER_A_ID) == 0

    idx = bm25.get_index(USER_A_ID)
    if idx is not None:
        assert idx.doc_count == 0

    a_docs = [r for r in sb.documents.values() if r["owner_id"] == str(USER_A_ID)]
    assert all(r["processing_status"] != "ready" for r in a_docs)
    if expect_db_failed:
        assert any(r["processing_status"] == "failed" for r in a_docs)
    else:
        assert a_docs == []


class TestPhase4IngestionFailureRecovery:
    """Failure-injection tests for every ingestion stage and retry path."""

    def test_1_and_2_invalid_empty_and_corrupted_inputs_create_no_resources(self, audit_env):
        client = audit_env["client"]
        tok = audit_env["token_a"]

        # Unsupported extension
        assert _upload(client, tok, "malware.exe", b"MZ...", "application/octet-stream").status_code == 400
        # Disallowed MIME for .pdf
        assert _upload(client, tok, "fake.pdf", b"%PDF-1.4", "image/png").status_code == 415
        # Empty 0-byte file
        assert _upload(client, tok, "empty.txt", b"", "text/plain").status_code == 400
        # Fake PDF without %PDF magic bytes
        assert _upload(client, tok, "bad_magic.pdf", b"NOT_A_PDF", "application/pdf").status_code == 400
        # TXT with binary null byte & invalid UTF-8
        assert _upload(client, tok, "nulls.txt", b"hello\x00world", "text/plain").status_code == 400
        assert _upload(client, tok, "badutf8.txt", b"\xff\xfe\xfd", "text/plain").status_code == 400
        _assert_user_a_clean(audit_env, expect_db_failed=False)

        # Whitespace-only TXT passes endpoint check, fails at DocumentExtractor -> Storage cleaned up
        res_ws = _upload(client, tok, "whitespace.txt", b"   \n\t  ", "text/plain")
        assert res_ws.status_code == 400
        _assert_user_a_clean(audit_env, expect_db_failed=True)

    def test_3_pdf_extraction_failures_clean_storage_and_mark_failed(self, audit_env):
        client = audit_env["client"]
        tok = audit_env["token_a"]

        # Corrupted PDF with valid %PDF header
        res_corrupt = _upload(client, tok, "corrupt.pdf", b"%PDF-1.4\n%%corrupted-garbage", "application/pdf")
        assert res_corrupt.status_code == 400
        _assert_user_a_clean(audit_env, expect_db_failed=True)

        # Encrypted PDF
        res_enc = _upload(client, tok, "encrypted.pdf", _make_encrypted_pdf(), "application/pdf")
        assert res_enc.status_code == 400
        _assert_user_a_clean(audit_env, expect_db_failed=True)

        # Scanned / image-only PDF (0 extractable text)
        res_empty_pdf = _upload(client, tok, "scanned.pdf", _make_pdf(""), "application/pdf")
        assert res_empty_pdf.status_code == 400
        _assert_user_a_clean(audit_env, expect_db_failed=True)

        # Unexpected extractor exception
        with patch.object(DocumentExtractor, "extract", side_effect=RuntimeError("PyMuPDF crash")):
            res_crash = _upload(client, tok, "crash.pdf", _make_pdf("valid text"), "application/pdf")
            assert res_crash.status_code == 500
        _assert_user_a_clean(audit_env, expect_db_failed=True)

    def test_4_chunking_failure_cleans_storage_and_marks_failed(self, audit_env):
        client = audit_env["client"]
        tok = audit_env["token_a"]

        with patch.object(TextChunker, "chunk_page", side_effect=ChunkingError("Simulated chunking failure")):
            res = _upload(client, tok, "chunk_fail.txt", b"Chunking failure test payload.", "text/plain")
            assert res.status_code == 500
        _assert_user_a_clean(audit_env, expect_db_failed=True)

    def test_5_embedding_failure_cleans_storage_and_marks_failed(self, audit_env):
        client = audit_env["client"]
        tok = audit_env["token_a"]

        with patch.object(audit_env["ingestion"].embedder, "embed_batch", side_effect=RuntimeError("Model OOM")):
            res = _upload(client, tok, "embed_fail.txt", b"Embedding failure test payload.", "text/plain")
            assert res.status_code == 500
        _assert_user_a_clean(audit_env, expect_db_failed=True)

    def test_6_chroma_insertion_failure_rolls_back_vectors_and_storage(self, audit_env):
        client = audit_env["client"]
        tok = audit_env["token_a"]
        store: ChromaVectorStore = audit_env["store"]

        with patch.object(store.collection, "add", side_effect=RuntimeError("Chroma HNSW write failed")):
            res = _upload(client, tok, "chroma_fail.txt", b"Chroma insertion failure payload.", "text/plain")
            assert res.status_code == 500
        _assert_user_a_clean(audit_env, expect_db_failed=True)

    def test_7_storage_upload_failure_leaves_zero_vectors_and_failed_row(self, audit_env):
        client = audit_env["client"]
        tok = audit_env["token_a"]
        sb: InMemoryMultiTenantSupabase = audit_env["sb"]

        sb.fail_storage_upload = True
        try:
            res = _upload(client, tok, "storage_fail.txt", b"Storage outage payload.", "text/plain")
            assert res.status_code == 503, res.text
        finally:
            sb.fail_storage_upload = False
        _assert_user_a_clean(audit_env, expect_db_failed=True)

    def test_8_supabase_rpc_failures_create_indexing_ready_and_missing_row(self, audit_env):
        client = audit_env["client"]
        tok = audit_env["token_a"]
        sb: InMemoryMultiTenantSupabase = audit_env["sb"]

        # 8a. create_user_document failure
        sb.fail_create_document = True
        try:
            res_create = _upload(client, tok, "rpc_create.txt", b"Create RPC fail.", "text/plain")
            assert res_create.status_code == 503
        finally:
            sb.fail_create_document = False
        _assert_user_a_clean(audit_env, expect_db_failed=False)

        # 8b. update_user_document_status('indexing') failure
        sb.fail_status_on = "indexing"
        try:
            res_idx = _upload(client, tok, "rpc_idx.txt", b"Indexing RPC fail.", "text/plain")
            assert res_idx.status_code == 503
        finally:
            sb.fail_status_on = None
        _assert_user_a_clean(audit_env, expect_db_failed=True)

        # 8c. update_user_document_status('ready') failure after vectors inserted
        sb.fail_status_on = "ready"
        try:
            res_ready = _upload(client, tok, "rpc_ready.txt", b"Ready RPC fail.", "text/plain")
            assert res_ready.status_code == 503
        finally:
            sb.fail_status_on = None
        _assert_user_a_clean(audit_env, expect_db_failed=True)

        # 8d. update_user_document_status('ready') returning empty row (missing/deleted concurrently)
        sb.return_empty_on_ready = True
        try:
            res_missing = _upload(client, tok, "rpc_missing.txt", b"Missing ready row.", "text/plain")
            assert res_missing.status_code == 500
        finally:
            sb.return_empty_on_ready = False
        _assert_user_a_clean(audit_env, expect_db_failed=True)

    def test_9_tenant_b_data_unaffected_by_tenant_a_failures(self, audit_env):
        client = audit_env["client"]
        tok_a = audit_env["token_a"]
        tok_b = audit_env["token_b"]
        sb: InMemoryMultiTenantSupabase = audit_env["sb"]
        store: ChromaVectorStore = audit_env["store"]
        retriever: Retriever = audit_env["retriever"]

        # Tenant B uploads a valid document first
        ok_b = _upload(
            client,
            tok_b,
            "tenant_b_safe.txt",
            b"Tenant B confidential zephyr baseline document.",
            "text/plain",
        )
        assert ok_b.status_code == 200
        b_doc_id = ok_b.json()["document_id"]
        b_chunks_before = store.count_for_owner(USER_B_ID)
        assert b_chunks_before > 0

        # Tenant A fails at ready-status update (late failure after Storage + ChromaDB)
        sb.fail_status_on = "ready"
        try:
            fail_a = _upload(client, tok_a, "tenant_a_fail.txt", b"Tenant A failing payload.", "text/plain")
            assert fail_a.status_code == 503
        finally:
            sb.fail_status_on = None

        _assert_user_a_clean(audit_env, expect_db_failed=True)

        # Verify Tenant B Storage, ChromaDB, DB catalog, and Hybrid retrieval are intact
        assert any(k.startswith(f"{USER_B_ID}/{b_doc_id}/") for k in sb.storage_objects)
        assert store.count_for_owner(USER_B_ID) == b_chunks_before
        b_list = client.get("/api/v1/documents", headers={"Authorization": f"Bearer {tok_b}"}).json()
        assert b_list["total"] == 1
        ret = retriever.retrieve(
            RetrievalRequest(
                query="zephyr",
                mode=RetrievalMode.HYBRID,
                config=RetrievalConfiguration(top_k=5),
            ),
            owner_id=USER_B_ID,
        )
        assert ret.returned_chunks_count > 0

    def test_10_retry_after_failed_ingestion_succeeds_without_duplicates(self, audit_env):
        client = audit_env["client"]
        tok = audit_env["token_a"]
        sb: InMemoryMultiTenantSupabase = audit_env["sb"]
        store: ChromaVectorStore = audit_env["store"]
        retriever: Retriever = audit_env["retriever"]

        payload = b"Unique retryable kaleidoscope spectroscopy findings for Phase 4."

        # First attempt fails after vectors inserted
        sb.fail_status_on = "ready"
        try:
            first_try = _upload(client, tok, "retry_doc.txt", payload, "text/plain")
            assert first_try.status_code == 503
        finally:
            sb.fail_status_on = None

        _assert_user_a_clean(audit_env, expect_db_failed=True)
        failed_docs = [r for r in sb.documents.values() if r["owner_id"] == str(USER_A_ID)]
        assert len(failed_docs) == 1
        old_doc_id = failed_docs[0]["id"]

        # Retry exact same payload
        second_try = _upload(client, tok, "retry_doc.txt", payload, "text/plain")
        assert second_try.status_code == 200
        body = second_try.json()
        assert body["status"] == "success"
        assert body["is_duplicate"] is False
        new_doc_id = body["document_id"]
        assert new_doc_id != old_doc_id

        # Old failed record purged; exactly 1 ready record, 1 Storage object, exact chunk count
        a_docs = [r for r in sb.documents.values() if r["owner_id"] == str(USER_A_ID)]
        assert len(a_docs) == 1
        assert a_docs[0]["id"] == new_doc_id
        assert a_docs[0]["processing_status"] == "ready"
        a_storage = [k for k in sb.storage_objects if k.startswith(f"{USER_A_ID}/")]
        assert len(a_storage) == 1
        assert a_storage[0].startswith(f"{USER_A_ID}/{new_doc_id}/")
        assert store.count_for_owner(USER_A_ID) == body["total_chunks"]

        # Searchable via both Dense and BM25
        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25):
            ret = retriever.retrieve(
                RetrievalRequest(
                    query="kaleidoscope spectroscopy",
                    mode=mode,
                    config=RetrievalConfiguration(top_k=5),
                ),
                owner_id=USER_A_ID,
            )
            assert ret.returned_chunks_count >= 1
            assert all(c.document_id == new_doc_id for c in ret.chunks)
