"""Phase 4 Milestone 2: Stale ingestion recovery, concurrency safety, and partial rollback tests."""

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from urllib.parse import parse_qs
from uuid import UUID, uuid4

import httpx
import pytest

from backend.app.core.config import Settings, get_settings
from backend.app.core.supabase_client import SupabaseClient, SupabaseConflictError
from backend.app.document_processing.identifier import compute_content_hash
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.storage_repository import SupabaseStorageRepository
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.retriever import Retriever
from backend.app.retrieval.tenant_bm25 import TenantBM25Manager
from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import RetrievalConfiguration, RetrievalMode, RetrievalRequest
from backend.app.services.ingestion_service import (
    DocumentInProgressError,
    IngestionService,
)
from backend.app.vector_store.store import ChromaVectorStore

USER_A_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
USER_B_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
SUPABASE_URL = "https://zblfcosunplvnnwjfyjz.supabase.co"
ANON_KEY = "test-public-anon-key"


class ConcurrencySupabaseMock:
    """Thread/async-safe in-memory Supabase PostgREST + Storage simulator with unique (owner_id, content_hash)."""

    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.storage_objects: Dict[str, Dict[str, Any]] = {}
        self.fail_status_update: bool = False
        self.fail_status_on: Optional[str] = None
        self.fail_storage_delete: bool = False
        self.force_hash_conflict_once: bool = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        owner_hdr = request.headers.get("x-test-owner") or str(USER_A_ID)
        path = request.url.path
        qs = parse_qs(request.url.query.decode("utf-8")) if request.url.query else {}
        now = datetime.now(timezone.utc).isoformat()

        # Storage endpoints
        if path.startswith("/storage/v1/object/retrieva-user-documents/"):
            obj_path = path.split("/retrieva-user-documents/", 1)[1]
            if not obj_path.startswith(f"{owner_hdr}/"):
                return httpx.Response(403, json={"message": "Forbidden"})
            if request.method == "POST":
                if obj_path in self.storage_objects and request.headers.get("x-upsert") != "true":
                    return httpx.Response(409, json={"message": "Already exists"})
                self.storage_objects[obj_path] = {
                    "bytes": request.content,
                    "content_type": request.headers.get("content-type"),
                }
                return httpx.Response(200, json={"Key": f"retrieva-user-documents/{obj_path}"})
            if request.method == "DELETE":
                if self.fail_storage_delete:
                    return httpx.Response(503, json={"message": "Simulated Storage delete failure"})
                if obj_path not in self.storage_objects:
                    return httpx.Response(404, json={"message": "Object not found"})
                del self.storage_objects[obj_path]
                return httpx.Response(200, json={"message": "Deleted"})

        # PostgREST RPCs
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        if path == "/rest/v1/rpc/create_user_document" and request.method == "POST":
            if self.force_hash_conflict_once:
                self.force_hash_conflict_once = False
                return httpx.Response(
                    409,
                    json={"code": "23505", "message": "duplicate key value violates unique constraint \"documents_owner_hash_unique\""},
                )
            for row in self.documents.values():
                if row["owner_id"] == owner_hdr and row["content_hash"] == body["p_content_hash"]:
                    return httpx.Response(
                        409,
                        json={"code": "23505", "message": "duplicate key value violates unique constraint \"documents_owner_hash_unique\""},
                    )
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
            if self.fail_status_update or (self.fail_status_on and target_status == self.fail_status_on):
                return httpx.Response(503, json={"message": "Simulated DB status update failure"})
            doc_id = body["p_document_id"]
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != owner_hdr:
                return httpx.Response(200, json=[])
            row["processing_status"] = body["p_processing_status"]
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
def env(shared_embedder: SentenceTransformerEmbedder):
    temp_dir = tempfile.mkdtemp(prefix="retrieva_p4m2_")
    sb = ConcurrencySupabaseMock()
    store = ChromaVectorStore(collection_name="p4m2_col", persist_directory=temp_dir)
    bm25 = TenantBM25Manager(vector_store=store)
    ingestion = IngestionService(embedder=shared_embedder, vector_store=store, tenant_bm25_manager=bm25)
    retriever = Retriever(
        dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
        tenant_bm25_manager=bm25,
        vector_store=store,
    )

    def make_repos(owner_id: UUID):
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(sb.handle),
            headers={"x-test-owner": str(owner_id)},
        )
        client = SupabaseClient(
            supabase_url=SUPABASE_URL,
            anon_key=ANON_KEY,
            access_token=f"jwt-{owner_id}",
            owner_id=owner_id,
            http_client=http_client,
        )
        return DocumentRepository(client), SupabaseStorageRepository(client)

    yield {
        "sb": sb,
        "store": store,
        "bm25": bm25,
        "ingestion": ingestion,
        "retriever": retriever,
        "make_repos": make_repos,
        "embedder": shared_embedder,
    }
    shutil.rmtree(temp_dir, ignore_errors=True)


def _seed_doc_row(
    sb: ConcurrencySupabaseMock,
    owner_id: UUID,
    doc_id: UUID,
    filename: str,
    payload: bytes,
    status: str,
    age_seconds: int,
) -> str:
    ts = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
    storage_path = f"{owner_id}/{doc_id}/{filename}"
    sb.documents[str(doc_id)] = {
        "id": str(doc_id),
        "owner_id": str(owner_id),
        "storage_path": storage_path,
        "filename": filename,
        "file_extension": ".txt",
        "mime_type": "text/plain",
        "content_hash": compute_content_hash(payload),
        "file_size_bytes": len(payload),
        "processing_status": status,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "total_pages": 1 if status in {"indexing", "ready"} else 0,
        "total_chunks": 1 if status in {"indexing", "ready"} else 0,
        "error_message": None,
        "created_at": ts,
        "updated_at": ts,
    }
    sb.storage_objects[storage_path] = {"bytes": payload, "content_type": "text/plain"}
    return storage_path


class TestStaleIngestionRecoveryAndConcurrency:
    """Tests for Part A (stale recovery), Part B (concurrent duplicate uploads), and Part C (partial cleanup)."""

    @pytest.mark.asyncio
    async def test_stale_uploading_and_indexing_recovery_and_reupload(self, env):
        sb: ConcurrencySupabaseMock = env["sb"]
        store: ChromaVectorStore = env["store"]
        ingestion: IngestionService = env["ingestion"]
        doc_repo_a, storage_repo_a = env["make_repos"](USER_A_ID)
        doc_repo_b, storage_repo_b = env["make_repos"](USER_B_ID)

        payload_up = b"Stale uploading document payload."
        payload_idx = b"Stale indexing document payload."
        doc_up = uuid4()
        doc_idx = uuid4()
        doc_b = uuid4()

        _seed_doc_row(sb, USER_A_ID, doc_up, "stale_up.txt", payload_up, "uploading", age_seconds=600)
        _seed_doc_row(sb, USER_A_ID, doc_idx, "stale_idx.txt", payload_idx, "indexing", age_seconds=600)
        # User B also has a stale doc; User A's recovery must NEVER touch User B
        _seed_doc_row(sb, USER_B_ID, doc_b, "b_stale.txt", b"User B stale.", "uploading", age_seconds=600)

        # Seed ChromaDB vectors for doc_idx to verify vector cleanup
        chunk = DocumentChunk(
            chunk_id=f"{doc_idx}_p1_c0",
            document_id=str(doc_idx),
            filename="stale_idx.txt",
            page_number=1,
            page_references=[1],
            chunk_index=0,
            chunk_text=payload_idx.decode(),
            start_offset=0,
            end_offset=len(payload_idx),
            char_count=len(payload_idx),
            word_count=4,
        )
        emb = env["embedder"].embed_batch([chunk.chunk_text])
        store.add_chunks([chunk], emb, USER_A_ID, compute_content_hash(payload_idx))
        assert store.count_for_owner(USER_A_ID) == 1

        results = await ingestion.recover_stale_documents_for_owner(
            owner_id=USER_A_ID,
            doc_repo=doc_repo_a,
            storage_repo=storage_repo_a,
            stale_timeout_seconds=300,
        )
        assert len(results) == 2
        assert all(r.recovered and not r.cleanup_errors for r in results)
        assert store.count_for_owner(USER_A_ID) == 0
        assert [k for k in sb.storage_objects if k.startswith(f"{USER_A_ID}/")] == []
        assert sb.documents[str(doc_up)]["processing_status"] == "failed"
        assert sb.documents[str(doc_idx)]["processing_status"] == "failed"

        # User B untouched
        assert sb.documents[str(doc_b)]["processing_status"] == "uploading"
        assert any(k.startswith(f"{USER_B_ID}/{doc_b}/") for k in sb.storage_objects)

        # Re-uploading payload_idx now succeeds cleanly
        ok = await ingestion.ingest_document_for_user(
            file_bytes=payload_idx,
            filename="stale_idx.txt",
            owner_id=USER_A_ID,
            doc_repo=doc_repo_a,
            storage_repo=storage_repo_a,
        )
        assert ok.status == "success"
        assert ok.document_id != str(doc_idx)
        assert str(doc_idx) not in sb.documents

    @pytest.mark.asyncio
    async def test_active_ready_and_deleting_documents_protected_from_recovery(self, env):
        sb: ConcurrencySupabaseMock = env["sb"]
        ingestion: IngestionService = env["ingestion"]
        doc_repo, storage_repo = env["make_repos"](USER_A_ID)

        active_payload = b"Active uploading payload."
        doc_active = uuid4()
        doc_ready = uuid4()
        doc_deleting = uuid4()

        _seed_doc_row(sb, USER_A_ID, doc_active, "active.txt", active_payload, "uploading", age_seconds=10)
        _seed_doc_row(sb, USER_A_ID, doc_ready, "ready.txt", b"Ready payload.", "ready", age_seconds=3600)
        _seed_doc_row(sb, USER_A_ID, doc_deleting, "deleting.txt", b"Deleting payload.", "deleting", age_seconds=3600)

        results = await ingestion.recover_stale_documents_for_owner(
            owner_id=USER_A_ID,
            doc_repo=doc_repo,
            storage_repo=storage_repo,
            stale_timeout_seconds=300,
        )
        assert results == []
        assert sb.documents[str(doc_active)]["processing_status"] == "uploading"
        assert sb.documents[str(doc_ready)]["processing_status"] == "ready"
        assert sb.documents[str(doc_deleting)]["processing_status"] == "deleting"

        # Active upload must still reject concurrent duplicate with DocumentInProgressError
        with pytest.raises(DocumentInProgressError):
            await ingestion.ingest_document_for_user(
                file_bytes=active_payload,
                filename="active.txt",
                owner_id=USER_A_ID,
                doc_repo=doc_repo,
                storage_repo=storage_repo,
            )

    @pytest.mark.asyncio
    async def test_concurrent_and_repeated_recovery_idempotency(self, env):
        sb: ConcurrencySupabaseMock = env["sb"]
        ingestion: IngestionService = env["ingestion"]
        doc_repo, storage_repo = env["make_repos"](USER_A_ID)

        doc_id = uuid4()
        _seed_doc_row(sb, USER_A_ID, doc_id, "race_stale.txt", b"Race stale payload.", "indexing", age_seconds=900)

        r1, r2 = await asyncio.gather(
            ingestion.recover_stale_document(doc_id, USER_A_ID, doc_repo, storage_repo, stale_timeout_seconds=300),
            ingestion.recover_stale_document(doc_id, USER_A_ID, doc_repo, storage_repo, stale_timeout_seconds=300),
        )
        recovered_count = sum(1 for r in (r1, r2) if r.recovered)
        already_handled_count = sum(1 for r in (r1, r2) if r.already_handled)
        assert recovered_count == 1
        assert already_handled_count == 1

        # Repeated call is idempotent no-op
        r3 = await ingestion.recover_stale_document(doc_id, USER_A_ID, doc_repo, storage_repo, stale_timeout_seconds=300)
        assert r3.recovered is False
        assert r3.already_handled is True
        assert r3.final_status == "failed"

    @pytest.mark.asyncio
    async def test_same_owner_concurrent_duplicate_uploads_and_db_unique_conflict(self, env):
        sb: ConcurrencySupabaseMock = env["sb"]
        store: ChromaVectorStore = env["store"]
        ingestion: IngestionService = env["ingestion"]
        doc_repo, storage_repo = env["make_repos"](USER_A_ID)

        payload = b"Simultaneous duplicate upload content for Phase 4 Milestone 2."
        res1, res2 = await asyncio.gather(
            ingestion.ingest_document_for_user(payload, "dup.txt", USER_A_ID, doc_repo, storage_repo),
            ingestion.ingest_document_for_user(payload, "dup.txt", USER_A_ID, doc_repo, storage_repo),
        )
        statuses = sorted([res1.status, res2.status])
        assert statuses == ["duplicate", "success"]
        assert res1.document_id == res2.document_id
        assert len([r for r in sb.documents.values() if r["owner_id"] == str(USER_A_ID)]) == 1
        assert len([k for k in sb.storage_objects if k.startswith(f"{USER_A_ID}/")]) == 1
        expected_chunks = res1.total_chunks if res1.status == "success" else res2.total_chunks
        assert store.count_for_owner(USER_A_ID) == expected_chunks

        # Multi-worker DB race simulation: unique constraint (23505 -> SupabaseConflictError 409)
        sb.force_hash_conflict_once = True
        with pytest.raises(SupabaseConflictError):
            await ingestion.ingest_document_for_user(
                b"Race loser unique constraint payload.",
                "loser.txt",
                USER_A_ID,
                doc_repo,
                storage_repo,
            )
        # Loser created zero Storage objects or ChromaDB vectors
        assert len([k for k in sb.storage_objects if k.startswith(f"{USER_A_ID}/")]) == 1
        assert store.count_for_owner(USER_A_ID) == expected_chunks

    @pytest.mark.asyncio
    async def test_cross_owner_identical_concurrent_uploads_succeed_independently(self, env):
        sb: ConcurrencySupabaseMock = env["sb"]
        store: ChromaVectorStore = env["store"]
        ingestion: IngestionService = env["ingestion"]
        doc_repo_a, storage_repo_a = env["make_repos"](USER_A_ID)
        doc_repo_b, storage_repo_b = env["make_repos"](USER_B_ID)

        shared_payload = b"Identical cross-tenant document uploaded concurrently."
        res_a, res_b = await asyncio.gather(
            ingestion.ingest_document_for_user(shared_payload, "shared.txt", USER_A_ID, doc_repo_a, storage_repo_a),
            ingestion.ingest_document_for_user(shared_payload, "shared.txt", USER_B_ID, doc_repo_b, storage_repo_b),
        )
        assert res_a.status == "success"
        assert res_b.status == "success"
        assert res_a.document_id != res_b.document_id
        assert store.count_for_owner(USER_A_ID) == res_a.total_chunks
        assert store.count_for_owner(USER_B_ID) == res_b.total_chunks
        assert len([k for k in sb.storage_objects if k.startswith(f"{USER_A_ID}/")]) == 1
        assert len([k for k in sb.storage_objects if k.startswith(f"{USER_B_ID}/")]) == 1

    @pytest.mark.asyncio
    async def test_partial_cleanup_failures_observable_and_retryable(self, env):
        sb: ConcurrencySupabaseMock = env["sb"]
        store: ChromaVectorStore = env["store"]
        ingestion: IngestionService = env["ingestion"]
        doc_repo, storage_repo = env["make_repos"](USER_A_ID)

        # 1. Multi-cleanup failure during stale recovery (ChromaDB + Storage both fail)
        doc_multi = uuid4()
        payload_multi = b"Multi cleanup failure payload."
        storage_path = _seed_doc_row(sb, USER_A_ID, doc_multi, "multi_fail.txt", payload_multi, "indexing", age_seconds=700)

        sb.fail_storage_delete = True
        try:
            with patch.object(store, "delete_document", side_effect=RuntimeError("Chroma delete down")):
                res = await ingestion.recover_stale_document(
                    doc_multi, USER_A_ID, doc_repo, storage_repo, stale_timeout_seconds=300
                )
        finally:
            sb.fail_storage_delete = False

        assert res.recovered is False
        assert "chromadb:RuntimeError" in res.cleanup_errors
        assert "storage:SupabaseUnavailableError" in res.cleanup_errors
        assert sb.documents[str(doc_multi)]["processing_status"] == "failed"
        assert "chromadb:RuntimeError" in (sb.documents[str(doc_multi)]["error_message"] or "")
        # Storage object & failed DB row preserved so retry can clean them up
        assert storage_path in sb.storage_objects

        # Retry upload now purges the failed row + Storage object and succeeds
        ok = await ingestion.ingest_document_for_user(
            payload_multi, "multi_fail.txt", USER_A_ID, doc_repo, storage_repo
        )
        assert ok.status == "success"
        assert storage_path not in sb.storage_objects
        assert str(doc_multi) not in sb.documents

        # 2. Rollback failure when status update fails during ingestion failure
        sb.fail_status_on = "failed"
        try:
            with patch.object(ingestion.embedder, "embed_batch", side_effect=RuntimeError("Embedder crash")):
                with pytest.raises(RuntimeError, match="Embedder crash"):
                    await ingestion.ingest_document_for_user(
                        b"Rollback status fail payload.",
                        "status_fail.txt",
                        USER_A_ID,
                        doc_repo,
                        storage_repo,
                    )
        finally:
            sb.fail_status_on = None

        # The failed ingestion's DB row stayed 'indexing' (since DB was down during rollback),
        # and Storage was cleaned up. Once aged past timeout, stale recovery recovers it!
        stuck_rows = [
            r for r in sb.documents.values()
            if r["owner_id"] == str(USER_A_ID) and r["processing_status"] == "indexing"
        ]
        assert len(stuck_rows) == 1
        stuck_id = UUID(stuck_rows[0]["id"])
        stuck_rows[0]["updated_at"] = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()

        rec_after = await ingestion.recover_stale_document(
            stuck_id, USER_A_ID, doc_repo, storage_repo, stale_timeout_seconds=300
        )
        assert rec_after.recovered is True
        assert sb.documents[str(stuck_id)]["processing_status"] == "failed"
