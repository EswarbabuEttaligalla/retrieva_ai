"""Phase 4 Milestone 3: BM25 consistency and retrieval integrity tests."""

import asyncio
import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from unittest.mock import patch
from urllib.parse import parse_qs
from uuid import UUID, uuid4

import httpx
import pytest

from backend.app.core.supabase_client import SupabaseClient
from backend.app.document_processing.identifier import compute_content_hash
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.storage_repository import SupabaseStorageRepository
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.retriever import Retriever
from backend.app.retrieval.tenant_bm25 import TenantBM25Manager
from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import RetrievalConfiguration, RetrievalMode, RetrievalRequest
from backend.app.services.ingestion_service import IngestionService
from backend.app.vector_store.store import ChromaVectorStore

USER_A_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
USER_B_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
SUPABASE_URL = "https://zblfcosunplvnnwjfyjz.supabase.co"
ANON_KEY = "test-public-anon-key"


class InMemoryBM25AuditSupabase:
    """Minimal owner-scoped Supabase PostgREST + Storage simulator for BM25 consistency tests."""

    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.storage_objects: Dict[str, bytes] = {}
        self.fail_status_on: Optional[str] = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        owner_hdr = request.headers.get("x-test-owner") or str(USER_A_ID)
        path = request.url.path
        qs = parse_qs(request.url.query.decode("utf-8")) if request.url.query else {}
        now = datetime.now(timezone.utc).isoformat()

        if path.startswith("/storage/v1/object/retrieva-user-documents/"):
            obj_path = path.split("/retrieva-user-documents/", 1)[1]
            if request.method == "POST":
                self.storage_objects[obj_path] = request.content
                return httpx.Response(200, json={"Key": f"retrieva-user-documents/{obj_path}"})
            if request.method == "DELETE":
                if obj_path not in self.storage_objects:
                    return httpx.Response(404, json={"message": "Not found"})
                del self.storage_objects[obj_path]
                return httpx.Response(200, json={"message": "Deleted"})

        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        if path == "/rest/v1/rpc/create_user_document" and request.method == "POST":
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
                return httpx.Response(503, json={"message": f"Simulated failure on {target_status}"})
            doc_id = body["p_document_id"]
            row = self.documents.get(doc_id)
            if not row or row["owner_id"] != owner_hdr:
                return httpx.Response(200, json=[])
            row["processing_status"] = target_status
            if body.get("p_total_pages") is not None:
                row["total_pages"] = body["p_total_pages"]
            if body.get("p_total_chunks") is not None:
                row["total_chunks"] = body["p_total_chunks"]
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
def bm25_env(shared_embedder: SentenceTransformerEmbedder):
    temp_dir = tempfile.mkdtemp(prefix="retrieva_p4m3_")
    sb = InMemoryBM25AuditSupabase()
    store = ChromaVectorStore(collection_name="p4m3_col", persist_directory=temp_dir)
    bm25 = TenantBM25Manager(vector_store=store, max_cached_tenants=2)
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
        "temp_dir": temp_dir,
        "sb": sb,
        "store": store,
        "bm25": bm25,
        "ingestion": ingestion,
        "retriever": retriever,
        "make_repos": make_repos,
        "embedder": shared_embedder,
    }
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestBM25ConsistencyAndRetrievalIntegrity:
    """Tests covering all 9 BM25 consistency and retrieval integrity scenarios."""

    @pytest.mark.asyncio
    async def test_ingest_retrieve_delete_disappears_from_bm25(self, bm25_env):
        ingestion: IngestionService = bm25_env["ingestion"]
        retriever: Retriever = bm25_env["retriever"]
        bm25: TenantBM25Manager = bm25_env["bm25"]
        doc_repo, storage_repo = bm25_env["make_repos"](USER_A_ID)

        res = await ingestion.ingest_document_for_user(
            b"Unique chrysanthemum bioluminescence telemetry.",
            "flora.txt",
            USER_A_ID,
            doc_repo,
            storage_repo,
        )
        assert res.status == "success"

        hit = retriever.retrieve(
            RetrievalRequest(query="chrysanthemum bioluminescence", mode=RetrievalMode.BM25),
            owner_id=USER_A_ID,
        )
        assert hit.returned_chunks_count == 1
        assert hit.chunks[0].document_id == res.document_id

        deleted = await ingestion.delete_document_for_user(
            UUID(res.document_id), USER_A_ID, doc_repo, storage_repo
        )
        assert deleted == 1

        after = retriever.retrieve(
            RetrievalRequest(query="chrysanthemum bioluminescence", mode=RetrievalMode.BM25),
            owner_id=USER_A_ID,
        )
        assert after.returned_chunks_count == 0
        idx = bm25.get_index(USER_A_ID)
        assert idx is not None and idx.doc_count == 0 and idx.inverted_index == {}

    @pytest.mark.asyncio
    async def test_failed_ingestion_and_stale_recovery_leave_no_bm25_entries(self, bm25_env):
        sb: InMemoryBM25AuditSupabase = bm25_env["sb"]
        store: ChromaVectorStore = bm25_env["store"]
        bm25: TenantBM25Manager = bm25_env["bm25"]
        ingestion: IngestionService = bm25_env["ingestion"]
        retriever: Retriever = bm25_env["retriever"]
        doc_repo, storage_repo = bm25_env["make_repos"](USER_A_ID)

        # 1. Failed ingestion at ready-status update
        sb.fail_status_on = "ready"
        try:
            with pytest.raises(Exception):
                await ingestion.ingest_document_for_user(
                    b"Uncommitted periwinkle failure token.",
                    "fail.txt",
                    USER_A_ID,
                    doc_repo,
                    storage_repo,
                )
        finally:
            sb.fail_status_on = None

        assert retriever.retrieve(
            RetrievalRequest(query="periwinkle", mode=RetrievalMode.BM25),
            owner_id=USER_A_ID,
        ).returned_chunks_count == 0

        # 2. Stale 'indexing' doc that was hydrated into BM25 cache from ChromaDB
        stale_id = uuid4()
        stale_text = b"Stale uncommitted rhododendron indexing chunk."
        stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        sb.documents[str(stale_id)] = {
            "id": str(stale_id),
            "owner_id": str(USER_A_ID),
            "storage_path": f"{USER_A_ID}/{stale_id}/stale.txt",
            "filename": "stale.txt",
            "file_extension": ".txt",
            "mime_type": "text/plain",
            "content_hash": compute_content_hash(stale_text),
            "file_size_bytes": len(stale_text),
            "processing_status": "indexing",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "total_pages": 1,
            "total_chunks": 1,
            "error_message": None,
            "created_at": stale_ts,
            "updated_at": stale_ts,
        }
        sb.storage_objects[f"{USER_A_ID}/{stale_id}/stale.txt"] = stale_text
        chunk = DocumentChunk(
            chunk_id=f"{stale_id}_p1_c0",
            document_id=str(stale_id),
            filename="stale.txt",
            page_number=1,
            page_references=[1],
            chunk_index=0,
            chunk_text=stale_text.decode(),
            start_offset=0,
            end_offset=len(stale_text),
            char_count=len(stale_text),
            word_count=5,
        )
        store.add_chunks([chunk], bm25_env["embedder"].embed_batch([chunk.chunk_text]), USER_A_ID, compute_content_hash(stale_text))
        # Hydrate cache while stale doc is in ChromaDB
        bm25.refresh_owner(USER_A_ID)
        assert bm25.get_index(USER_A_ID).doc_count == 1

        rec = await ingestion.recover_stale_document(stale_id, USER_A_ID, doc_repo, storage_repo, stale_timeout_seconds=300)
        assert rec.recovered is True
        assert retriever.retrieve(
            RetrievalRequest(query="rhododendron", mode=RetrievalMode.BM25),
            owner_id=USER_A_ID,
        ).returned_chunks_count == 0

    @pytest.mark.asyncio
    async def test_cache_eviction_rehydration_and_backend_restart_reconstruction(self, bm25_env):
        ingestion: IngestionService = bm25_env["ingestion"]
        retriever: Retriever = bm25_env["retriever"]
        bm25: TenantBM25Manager = bm25_env["bm25"]
        doc_repo_a, storage_repo_a = bm25_env["make_repos"](USER_A_ID)
        doc_repo_b, storage_repo_b = bm25_env["make_repos"](USER_B_ID)
        user_c = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
        doc_repo_c, storage_repo_c = bm25_env["make_repos"](user_c)

        # Identical content for User A and User B
        shared = b"Identical cross-owner bougainvillea corpus."
        res_a = await ingestion.ingest_document_for_user(shared, "shared.txt", USER_A_ID, doc_repo_a, storage_repo_a)
        res_b = await ingestion.ingest_document_for_user(shared, "shared.txt", USER_B_ID, doc_repo_b, storage_repo_b)
        await ingestion.ingest_document_for_user(b"Tenant C marigolds.", "c.txt", user_c, doc_repo_c, storage_repo_c)

        # max_cached_tenants=2 -> User A was evicted when User C loaded
        assert bm25.get_index(USER_A_ID) is None

        # Rehydration on query restores User A cleanly
        rehydrated = retriever.retrieve(
            RetrievalRequest(query="bougainvillea", mode=RetrievalMode.BM25),
            owner_id=USER_A_ID,
        )
        assert rehydrated.returned_chunks_count == 1
        assert rehydrated.chunks[0].document_id == res_a.document_id

        # Delete User A's doc -> User B's Dense/BM25/Hybrid remain intact
        await ingestion.delete_document_for_user(UUID(res_a.document_id), USER_A_ID, doc_repo_a, storage_repo_a)
        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            assert retriever.retrieve(
                RetrievalRequest(query="bougainvillea", mode=mode),
                owner_id=USER_A_ID,
            ).returned_chunks_count == 0
            b_hit = retriever.retrieve(
                RetrievalRequest(query="bougainvillea", mode=mode),
                owner_id=USER_B_ID,
            )
            assert b_hit.returned_chunks_count == 1
            assert b_hit.chunks[0].document_id == res_b.document_id

        # Simulate full backend restart: brand-new ChromaVectorStore, TenantBM25Manager, Retriever over temp_dir
        restarted_store = ChromaVectorStore(collection_name="p4m3_col", persist_directory=bm25_env["temp_dir"])
        restarted_bm25 = TenantBM25Manager(vector_store=restarted_store)
        restarted_retriever = Retriever(
            dense_retriever=DenseRetriever(embedder=bm25_env["embedder"], vector_store=restarted_store),
            tenant_bm25_manager=restarted_bm25,
            vector_store=restarted_store,
        )
        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            assert restarted_retriever.retrieve(
                RetrievalRequest(query="bougainvillea", mode=mode),
                owner_id=USER_A_ID,
            ).returned_chunks_count == 0
            b_post_restart = restarted_retriever.retrieve(
                RetrievalRequest(query="bougainvillea", mode=mode),
                owner_id=USER_B_ID,
            )
            assert b_post_restart.returned_chunks_count == 1
            assert b_post_restart.chunks[0].document_id == res_b.document_id

    @pytest.mark.asyncio
    async def test_concurrent_ingestion_and_deletion_consistency(self, bm25_env):
        ingestion: IngestionService = bm25_env["ingestion"]
        retriever: Retriever = bm25_env["retriever"]
        store: ChromaVectorStore = bm25_env["store"]
        bm25: TenantBM25Manager = bm25_env["bm25"]
        doc_repo, storage_repo = bm25_env["make_repos"](USER_A_ID)

        d1 = await ingestion.ingest_document_for_user(b"Delete target hibiscus alpha.", "d1.txt", USER_A_ID, doc_repo, storage_repo)
        d2 = await ingestion.ingest_document_for_user(b"Keep baseline lavender beta.", "d2.txt", USER_A_ID, doc_repo, storage_repo)

        # Concurrently delete d1 while ingesting d3
        del_res, ing_res = await asyncio.gather(
            ingestion.delete_document_for_user(UUID(d1.document_id), USER_A_ID, doc_repo, storage_repo),
            ingestion.ingest_document_for_user(b"Concurrent ingest jasmine gamma.", "d3.txt", USER_A_ID, doc_repo, storage_repo),
        )
        assert del_res == 1
        assert ing_res.status == "success"

        # BM25 index corpus must match ChromaDB chunks exactly (d2 + d3, no d1)
        chroma_ids = {c["chunk_id"] for c in store.get_owner_chunks(USER_A_ID)}
        bm25_ids = set(bm25.get_or_create_index(USER_A_ID).corpus_chunks.keys())
        assert bm25_ids == chroma_ids
        assert retriever.retrieve(RetrievalRequest(query="hibiscus", mode=RetrievalMode.BM25), owner_id=USER_A_ID).returned_chunks_count == 0
        assert retriever.retrieve(RetrievalRequest(query="lavender", mode=RetrievalMode.BM25), owner_id=USER_A_ID).returned_chunks_count == 1
        assert retriever.retrieve(RetrievalRequest(query="jasmine", mode=RetrievalMode.BM25), owner_id=USER_A_ID).returned_chunks_count == 1
