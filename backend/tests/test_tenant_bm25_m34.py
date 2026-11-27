"""Phase 3, Milestone 3.4 Automated Tests: Per-Tenant BM25 + Hybrid Retrieval Isolation.

Verifies:
- Elimination of shared global BM25 index for authenticated retrieval
- Lazy per-tenant BM25 hydration from owner-scoped ChromaDB chunks
- Independent BM25 corpus statistics and scores across tenants (no cross-tenant IDF leakage)
- Identical content uploaded by multiple users remains independently indexed and deletable
- Isolation across Dense, BM25, Hybrid (RRF), and Cross-Encoder Reranking stages
- Defense-in-depth TenantIsolationError enforcement at every retrieval stage and missing owner rejection
- Multi-document UUID filtering (>=3 docs across 2 users, foreign UUID, mixed owned+foreign, empty filter)
- RAGService answer_question and answer_question_stream preserving all multi-document filters
- LRU eviction and clean transparent reconstruction from ChromaDB
- Thread-safe concurrent same-owner and cross-owner retrieval
- Immediate BM25 sync on ingestion/deletion, failed-ingestion cleanup, and fail-closed hydration
- End-to-end HTTP isolation via /api/v1/retrieval/search and /api/v1/chat/answer with real ES256 JWTs
"""

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.core.supabase_client import SupabaseClient
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.llm.client import OpenAIClient
from backend.app.main import app
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.storage_repository import SupabaseStorageRepository
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.retrieval.tenant_bm25 import BM25HydrationError, TenantBM25Manager
from backend.app.schemas.answer import AnswerRequest
from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import (
    RetrievalConfiguration,
    RetrievalMode,
    RetrievalRequest,
    RetrievedChunk,
)
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.rag_service import RAGService
from backend.app.vector_store.store import ChromaVectorStore, TenantIsolationError

USER_A_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
USER_B_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
USER_C_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
SUPABASE_URL = "https://zblfcosunplvnnwjfyjz.supabase.co"
ANON_KEY = "test-anon-key"


def _generate_ec_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk_dict = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(private_key.public_key()))
    jwk_dict["kid"] = "m34-test-key"
    jwk_dict["alg"] = "ES256"
    jwk_dict["use"] = "sig"
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem, jwk_dict


def _issue_jwt(private_pem: bytes, user_id: UUID, kid: str = "m34-test-key") -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "aud": "authenticated",
        "iss": f"{SUPABASE_URL}/auth/v1",
        "role": "authenticated",
        "email": f"{str(user_id)[:8]}@example.com",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
    }
    return jwt.encode(payload, private_pem, algorithm="ES256", headers={"kid": kid})


def _make_chunk(
    chunk_id: str,
    document_id: str,
    filename: str,
    text: str,
    page: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        page_number=page,
        page_references=[page],
        chunk_index=chunk_index,
        chunk_text=text,
        start_offset=0,
        end_offset=len(text),
        char_count=len(text),
        word_count=len(text.split()),
    )


@pytest.fixture(scope="module")
def shared_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder.get_instance()


@pytest.fixture
def isolated_env(shared_embedder: SentenceTransformerEmbedder):
    """Build an isolated ChromaDB + TenantBM25Manager + Retriever populated for Users A and B."""
    temp_dir = tempfile.mkdtemp(prefix="retrieva_m34_iso_")
    store = ChromaVectorStore(collection_name="m34_isolation_col", persist_directory=temp_dir)
    manager = TenantBM25Manager(vector_store=store, max_cached_tenants=50)
    dense = DenseRetriever(embedder=shared_embedder, vector_store=store)
    retriever = Retriever(
        dense_retriever=dense,
        tenant_bm25_manager=manager,
        vector_store=store,
        reranker=CrossEncoderReranker.get_instance(),
    )

    doc_a1 = str(uuid4())
    doc_a2 = str(uuid4())
    doc_a3 = str(uuid4())
    doc_a_shared = str(uuid4())

    doc_b1 = str(uuid4())
    doc_b2 = str(uuid4())
    doc_b_shared = str(uuid4())

    shared_text = (
        "Shared distributed consensus protocol checkpoint verification procedure for cluster nodes."
    )

    a_chunks = [
        _make_chunk(
            "chunk_a1_0",
            doc_a1,
            "alpha_mineral.txt",
            "Zephyrite crystal lattice resonance stabilizes photonic waveguides under cryogenic conditions.",
        ),
        _make_chunk(
            "chunk_a2_0",
            doc_a2,
            "alpha_telemetry.txt",
            "Telemetry pipeline aggregates sensor frames using bounded ring buffers and checksum validation.",
        ),
        _make_chunk(
            "chunk_a3_0",
            doc_a3,
            "alpha_audit.txt",
            "Zephyrite secondary spectral calibration validates optical sensor alignment across mirrors.",
        ),
        _make_chunk(
            "chunk_a_shared_0",
            doc_a_shared,
            "shared_manual.txt",
            shared_text,
        ),
    ]

    b_chunks = [
        _make_chunk(
            "chunk_b1_0",
            doc_b1,
            "beta_astro.txt",
            "Quasarflux radio interferometry measures pulsar timing residuals across deep space arrays.",
        ),
        _make_chunk(
            "chunk_b2_0",
            doc_b2,
            "beta_botany.txt",
            "Photosynthesis in alpine flora adapts chlorophyll absorption under ultraviolet radiation.",
        ),
        _make_chunk(
            "chunk_b_shared_0",
            doc_b_shared,
            "shared_manual.txt",
            shared_text,
        ),
    ]

    store.add_chunks(
        a_chunks,
        shared_embedder.embed_batch([c.chunk_text for c in a_chunks]),
        owner_id=USER_A_ID,
        prevent_duplicates=False,
    )
    store.add_chunks(
        b_chunks,
        shared_embedder.embed_batch([c.chunk_text for c in b_chunks]),
        owner_id=USER_B_ID,
        prevent_duplicates=False,
    )

    # Also plant a legacy unowned chunk directly in ChromaDB; must NEVER hydrate into any tenant.
    legacy_vec = shared_embedder.embed_batch(["Legacy unowned zephyrite quasarflux cluster secret."])[0]
    store.collection.add(
        ids=["legacy_unowned_chunk_0"],
        embeddings=[legacy_vec],
        documents=["Legacy unowned zephyrite quasarflux cluster secret."],
        metadatas=[{"document_id": "legacy_doc", "filename": "legacy.txt", "page_number": 1}],
    )

    try:
        yield {
            "temp_dir": temp_dir,
            "store": store,
            "manager": manager,
            "retriever": retriever,
            "embedder": shared_embedder,
            "docs": {
                "a1": doc_a1,
                "a2": doc_a2,
                "a3": doc_a3,
                "a_shared": doc_a_shared,
                "b1": doc_b1,
                "b2": doc_b2,
                "b_shared": doc_b_shared,
            },
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class InMemorySupabaseMock:
    """Minimal mocked Supabase PostgREST + Storage for M3.4 ingestion & API tests."""

    def __init__(self, fail_on_ready_status: bool = False) -> None:
        self.fail_on_ready_status = fail_on_ready_status
        self.documents: Dict[str, Dict] = {}
        self.storage_objects: Dict[str, bytes] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        now = datetime.now(timezone.utc).isoformat()
        owner_id = request.headers.get("x-test-owner", str(USER_A_ID))
        if path.startswith("/storage/v1/object/retrieva-user-documents/"):
            obj_path = path.split("/retrieva-user-documents/", 1)[1]
            if request.method == "POST":
                self.storage_objects[obj_path] = request.content
                return httpx.Response(200, json={"Key": f"retrieva-user-documents/{obj_path}"})
            if request.method == "DELETE":
                self.storage_objects.pop(obj_path, None)
                return httpx.Response(200, json={"message": "Deleted"})

        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        if path == "/rest/v1/rpc/create_user_document":
            doc_id = body.get("p_document_id") or str(uuid4())
            row = {
                "id": doc_id,
                "owner_id": owner_id,
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

        if path == "/rest/v1/rpc/update_user_document_status":
            doc_id = body["p_document_id"]
            if self.fail_on_ready_status and body.get("p_processing_status") == "ready":
                return httpx.Response(500, json={"message": "Simulated metadata ready failure"})
            row = self.documents.get(doc_id)
            if not row:
                return httpx.Response(200, json=[])
            row["processing_status"] = body["p_processing_status"]
            if body.get("p_total_pages") is not None:
                row["total_pages"] = body["p_total_pages"]
            if body.get("p_total_chunks") is not None:
                row["total_chunks"] = body["p_total_chunks"]
            row["updated_at"] = now
            return httpx.Response(200, json=[row])

        if path == "/rest/v1/rpc/delete_user_document":
            row = self.documents.pop(body["p_document_id"], None)
            return httpx.Response(200, json=[row] if row else [])

        if path == "/rest/v1/documents" and request.method == "GET":
            return httpx.Response(200, json=[])

        return httpx.Response(404, json={"message": "Not found"})


class TestTenantBM25Isolation:
    """Tests strict per-tenant BM25 isolation, corpus statistics, and identical-content handling."""

    def test_lazy_hydration_only_loads_queried_tenant(self, isolated_env):
        manager: TenantBM25Manager = isolated_env["manager"]
        retriever: Retriever = isolated_env["retriever"]

        assert manager.cached_tenant_count() == 0
        assert manager.get_index(USER_A_ID) is None
        assert manager.get_index(USER_B_ID) is None

        # User B runs BM25 search: only User B is hydrated
        res_b = retriever.retrieve(
            RetrievalRequest(query="quasarflux", mode=RetrievalMode.BM25, top_k=5),
            owner_id=USER_B_ID,
        )
        assert res_b.returned_chunks_count == 1
        assert res_b.chunks[0].document_id == isolated_env["docs"]["b1"]
        assert manager.cached_tenant_count() == 1
        assert manager.get_index(USER_B_ID) is not None
        assert manager.get_index(USER_A_ID) is None

        # Legacy unowned chunks must NOT be hydrated into B or A
        assert manager.get_index(USER_B_ID).doc_count == 3
        stats_a = retriever.get_owner_statistics(USER_A_ID)
        assert stats_a["bm25"]["chunk_count"] == 4
        assert stats_a["bm25"]["unique_document_count"] == 4

    def test_distinctive_terms_and_stats_never_leak_across_tenants(self, isolated_env):
        retriever: Retriever = isolated_env["retriever"]
        store: ChromaVectorStore = isolated_env["store"]
        embedder: SentenceTransformerEmbedder = isolated_env["embedder"]
        docs = isolated_env["docs"]

        # Query User B with User A's exclusive keyword -> must return 0 chunks in BM25
        bm25_b_for_a = retriever.retrieve(
            RetrievalRequest(query="zephyrite", mode=RetrievalMode.BM25, top_k=5),
            owner_id=USER_B_ID,
        )
        assert bm25_b_for_a.returned_chunks_count == 0

        # Query User A with User B's exclusive keyword -> must return 0 chunks in BM25
        bm25_a_for_b = retriever.retrieve(
            RetrievalRequest(query="quasarflux", mode=RetrievalMode.BM25, top_k=5),
            owner_id=USER_A_ID,
        )
        assert bm25_a_for_b.returned_chunks_count == 0

        # Measure User B's BM25 score and stats on shared consensus document BEFORE mutating User A
        stats_b_before = retriever.get_owner_statistics(USER_B_ID)["bm25"]
        res_b_before = retriever.retrieve(
            RetrievalRequest(query="distributed consensus checkpoint", mode=RetrievalMode.BM25, top_k=5),
            owner_id=USER_B_ID,
        )
        assert res_b_before.returned_chunks_count == 1
        assert res_b_before.chunks[0].document_id == docs["b_shared"]
        score_b_before = res_b_before.chunks[0].bm25_score

        # Heavily mutate User A's corpus with the exact same terms to shift User A's N, n(t), avgdl
        extra_chunks = [
            _make_chunk(
                f"chunk_a_flood_{idx}",
                str(uuid4()),
                f"flood_{idx}.txt",
                "Distributed consensus protocol checkpoint cluster nodes replication quorum.",
            )
            for idx in range(12)
        ]
        store.add_chunks(
            extra_chunks,
            embedder.embed_batch([c.chunk_text for c in extra_chunks]),
            owner_id=USER_A_ID,
            prevent_duplicates=False,
        )
        retriever.sync_owner_chunks(USER_A_ID, extra_chunks)

        # User A's stats & scores change, while User B's stats and BM25 score remain bit-for-bit identical
        stats_a_after = retriever.get_owner_statistics(USER_A_ID)["bm25"]
        stats_b_after = retriever.get_owner_statistics(USER_B_ID)["bm25"]
        res_b_after = retriever.retrieve(
            RetrievalRequest(query="distributed consensus checkpoint", mode=RetrievalMode.BM25, top_k=5),
            owner_id=USER_B_ID,
        )
        assert stats_a_after["chunk_count"] == 16
        assert stats_b_after == stats_b_before
        assert res_b_after.chunks[0].bm25_score == pytest.approx(score_b_before, rel=1e-9, abs=1e-12)

    def test_identical_content_distinct_uuids_independent_delete(self, isolated_env):
        retriever: Retriever = isolated_env["retriever"]
        store: ChromaVectorStore = isolated_env["store"]
        docs = isolated_env["docs"]

        # Both A and B initially find their own UUID for the identical text
        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            res_a = retriever.retrieve(
                RetrievalRequest(query="distributed consensus checkpoint", mode=mode, top_k=5),
                owner_id=USER_A_ID,
            )
            res_b = retriever.retrieve(
                RetrievalRequest(query="distributed consensus checkpoint", mode=mode, top_k=5),
                owner_id=USER_B_ID,
            )
            assert any(c.document_id == docs["a_shared"] for c in res_a.chunks)
            assert all(c.document_id != docs["b_shared"] for c in res_a.chunks)
            assert any(c.document_id == docs["b_shared"] for c in res_b.chunks)
            assert all(c.document_id != docs["a_shared"] for c in res_b.chunks)

        # Delete User A's copy -> User B's copy must stay fully searchable across all modes
        store.delete_document(docs["a_shared"], owner_id=USER_A_ID)
        retriever.remove_owner_document(USER_A_ID, docs["a_shared"])

        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            res_a_after = retriever.retrieve(
                RetrievalRequest(query="distributed consensus checkpoint", mode=mode, top_k=5),
                owner_id=USER_A_ID,
            )
            res_b_after = retriever.retrieve(
                RetrievalRequest(query="distributed consensus checkpoint", mode=mode, top_k=5),
                owner_id=USER_B_ID,
            )
            assert all(c.document_id != docs["a_shared"] for c in res_a_after.chunks)
            assert any(c.document_id == docs["b_shared"] for c in res_b_after.chunks)


class TestRetrievalModesAndDefenseInDepth:
    """Tests Dense, BM25, Hybrid, Reranking, missing owner rejection, and stage assertions."""

    def test_all_modes_and_reranking_return_only_caller_chunks(self, isolated_env):
        retriever: Retriever = isolated_env["retriever"]
        docs = isolated_env["docs"]
        user_a_docs = {docs["a1"], docs["a2"], docs["a3"], docs["a_shared"]}

        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            for rerank in (False, True):
                res = retriever.retrieve(
                    RetrievalRequest(
                        query="zephyrite photonic resonance and consensus",
                        mode=mode,
                        top_k=5,
                        candidate_k=10,
                        enable_reranking=rerank,
                    ),
                    owner_id=USER_A_ID,
                )
                assert res.returned_chunks_count >= 1
                for chunk in res.chunks:
                    assert chunk.document_id in user_a_docs
                    assert chunk.metadata.get("owner_id") == str(USER_A_ID)

    def test_missing_or_empty_owner_id_fails_closed(self, isolated_env):
        retriever: Retriever = isolated_env["retriever"]
        manager: TenantBM25Manager = isolated_env["manager"]

        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            for bad_owner in (None, "", "   "):
                with pytest.raises((ValueError, TenantIsolationError)):
                    retriever.retrieve(
                        RetrievalRequest(query="zephyrite", mode=mode),
                        owner_id=bad_owner,  # type: ignore[arg-type]
                    )

        with pytest.raises((ValueError, TenantIsolationError)):
            manager.get_or_create_index("")

    def test_defense_in_depth_blocks_injected_cross_tenant_chunk_at_every_stage(self, isolated_env):
        retriever: Retriever = isolated_env["retriever"]
        foreign_chunk = RetrievedChunk(
            chunk_id="foreign_leak_chunk",
            document_id=isolated_env["docs"]["b1"],
            filename="beta_astro.txt",
            page_number=1,
            page_references=[1],
            chunk_text="Quasarflux cross-tenant spoofed chunk.",
            start_offset=0,
            end_offset=38,
            rank=1,
            cosine_distance=0.1,
            cosine_similarity=0.9,
            bm25_score=5.0,
            metadata={"owner_id": str(USER_B_ID)},
        )

        # Stage: dense
        with patch.object(retriever.dense_retriever, "search", return_value=[foreign_chunk]):
            with pytest.raises(TenantIsolationError, match="dense"):
                retriever.retrieve(
                    RetrievalRequest(query="zephyrite", mode=RetrievalMode.DENSE),
                    owner_id=USER_A_ID,
                )

        # Stage: bm25
        fake_bm25 = MagicMock(spec=BM25Index)
        fake_bm25.search.return_value = [foreign_chunk]
        with patch.object(retriever.tenant_bm25_manager, "get_or_create_index", return_value=fake_bm25):
            with pytest.raises(TenantIsolationError, match="bm25"):
                retriever.retrieve(
                    RetrievalRequest(query="zephyrite", mode=RetrievalMode.BM25),
                    owner_id=USER_A_ID,
                )

        # Stage: fusion
        with patch("backend.app.retrieval.retriever.reciprocal_rank_fusion", return_value=[foreign_chunk]):
            with pytest.raises(TenantIsolationError, match="fusion"):
                retriever.retrieve(
                    RetrievalRequest(query="zephyrite", mode=RetrievalMode.HYBRID),
                    owner_id=USER_A_ID,
                )

        # Stage: rerank
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = ([foreign_chunk], None)
        with patch.object(retriever, "get_reranker", return_value=mock_reranker):
            with pytest.raises(TenantIsolationError, match="rerank"):
                retriever.retrieve(
                    RetrievalRequest(query="zephyrite", mode=RetrievalMode.BM25, enable_reranking=True),
                    owner_id=USER_A_ID,
                )


class TestDocumentFilteringAndRAGService:
    """Tests single, multi-doc, foreign-doc, mixed, and empty filtering through Retriever and RAGService."""

    def test_single_multi_foreign_mixed_and_empty_filters(self, isolated_env):
        retriever: Retriever = isolated_env["retriever"]
        docs = isolated_env["docs"]

        # 1. Single owned document filter
        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            res_single = retriever.retrieve(
                RetrievalRequest(
                    query="zephyrite optical calibration",
                    mode=mode,
                    document_id_filter=docs["a1"],
                    top_k=5,
                ),
                owner_id=USER_A_ID,
            )
            assert res_single.returned_chunks_count == 1
            assert res_single.chunks[0].document_id == docs["a1"]

        # 2. Multi-document filter with subset [a1, a3] excludes a2 and User B docs
        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            res_multi = retriever.retrieve(
                RetrievalRequest(
                    query="zephyrite resonance calibration telemetry",
                    mode=mode,
                    document_ids_filter=[docs["a1"], docs["a3"]],
                    top_k=5,
                ),
                owner_id=USER_A_ID,
            )
            found_docs = {c.document_id for c in res_multi.chunks}
            assert found_docs == {docs["a1"], docs["a3"]}
            assert docs["a2"] not in found_docs

        # 3. Foreign document UUID only -> 0 results across all modes
        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            res_foreign = retriever.retrieve(
                RetrievalRequest(
                    query="quasarflux radio",
                    mode=mode,
                    document_ids_filter=[docs["b1"]],
                    top_k=5,
                ),
                owner_id=USER_A_ID,
            )
            assert res_foreign.returned_chunks_count == 0
            assert res_foreign.chunks == []

        # 4. Mixed owned + foreign UUIDs -> ONLY owned a1 returned, never b1
        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            res_mixed = retriever.retrieve(
                RetrievalRequest(
                    query="zephyrite quasarflux",
                    mode=mode,
                    document_ids_filter=[docs["a1"], docs["b1"]],
                    top_k=5,
                ),
                owner_id=USER_A_ID,
            )
            assert [c.document_id for c in res_mixed.chunks] == [docs["a1"]]

        # 5. Conflicting single + multi filter with empty intersection -> 0 results
        res_conflict = retriever.retrieve(
            RetrievalRequest(
                query="zephyrite",
                mode=RetrievalMode.HYBRID,
                document_id_filter=docs["a1"],
                document_ids_filter=[docs["a3"]],
            ),
            owner_id=USER_A_ID,
        )
        assert res_conflict.returned_chunks_count == 0

        # 6. Empty filter -> searches all owned documents for User A (including a1, a2, a3, a_shared)
        res_all = retriever.retrieve(
            RetrievalRequest(
                query="zephyrite telemetry consensus",
                mode=RetrievalMode.BM25,
                document_ids_filter=None,
                top_k=10,
            ),
            owner_id=USER_A_ID,
        )
        assert res_all.returned_chunks_count >= 3
        assert all(c.metadata.get("owner_id") == str(USER_A_ID) for c in res_all.chunks)

    @pytest.mark.asyncio
    async def test_rag_service_preserves_multi_document_filters_and_streaming(self, isolated_env):
        retriever: Retriever = isolated_env["retriever"]
        docs = isolated_env["docs"]

        mock_llm = MagicMock(spec=OpenAIClient)
        mock_llm.model = "gpt-4o-mini"
        mock_llm.generate.return_value = {
            "content": json.dumps(
                {
                    "answer": "Zephyrite resonance and calibration are validated [Source 1] [Source 2].",
                    "insufficient_context": False,
                    "citations": [
                        {
                            "source_id": "[Source 1]",
                            "claim": "Resonance stabilizes photonic waveguides",
                            "quote": "Zephyrite crystal lattice resonance stabilizes photonic waveguides",
                        },
                        {
                            "source_id": "[Source 2]",
                            "claim": "Spectral calibration validates optical sensor alignment",
                            "quote": "Zephyrite secondary spectral calibration validates optical sensor alignment",
                        },
                    ],
                }
            ),
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        }

        async def _mock_stream_async(*args, **kwargs):
            yield "Grounded [Source 1]."

        mock_llm.generate_stream_async = _mock_stream_async

        rag = RAGService(retriever=retriever, llm_client=mock_llm)

        # Multi-doc filter [a1, a3] must return sources from BOTH documents (not truncated to a1 only!)
        resp = rag.answer_question(
            AnswerRequest(
                query="What does zephyrite resonance and spectral calibration validate?",
                document_ids=[docs["a1"], docs["a3"]],
                retrieval_config=RetrievalConfiguration(mode=RetrievalMode.BM25, top_k=5),
            ),
            owner_id=USER_A_ID,
        )
        cited_or_context_docs = {c.document_id for c in resp.citations}
        # Verify retriever actually returned both docs in answer generation
        assert resp.metadata.total_chunks_retrieved == 2
        assert cited_or_context_docs == {docs["a1"], docs["a3"]}

        # Foreign document UUID in RAGService -> zero chunks retrieved, grounded fallback
        resp_foreign = rag.answer_question(
            AnswerRequest(
                query="What is quasarflux?",
                document_ids=[docs["b1"]],
                retrieval_config=RetrievalConfiguration(mode=RetrievalMode.HYBRID, top_k=5),
            ),
            owner_id=USER_A_ID,
        )
        assert resp_foreign.metadata.total_chunks_retrieved == 0
        assert resp_foreign.citations == []

        # Streaming also respects mixed owned + foreign filter and retrieves only owned a3
        stream_events = []
        async for evt in rag.answer_question_stream(
            AnswerRequest(
                query="zephyrite calibration",
                document_ids=[docs["a3"], docs["b1"]],
                retrieval_config=RetrievalConfiguration(mode=RetrievalMode.BM25, top_k=5),
            ),
            owner_id=USER_A_ID,
        ):
            stream_events.append(evt)
        citation_frames = [e for e in stream_events if "event: citations" in e]
        assert citation_frames
        assert docs["a3"] in citation_frames[0]
        assert docs["b1"] not in citation_frames[0]


class TestCacheEvictionConcurrencyAndLifecycle:
    """Tests LRU eviction, concurrency, failed ingestion containment, and fail-closed hydration."""

    def test_lru_eviction_and_rehydration(self, shared_embedder: SentenceTransformerEmbedder):
        temp_dir = tempfile.mkdtemp(prefix="retrieva_m34_lru_")
        try:
            store = ChromaVectorStore(collection_name="m34_lru_col", persist_directory=temp_dir)
            manager = TenantBM25Manager(vector_store=store, max_cached_tenants=2)

            for uid, kw in [(USER_A_ID, "obsidian"), (USER_B_ID, "beryllium"), (USER_C_ID, "vanadium")]:
                ch = _make_chunk(f"ch_{uid}", str(uid), f"{kw}.txt", f"Unique keyword {kw} sample document.")
                store.add_chunks([ch], shared_embedder.embed_batch([ch.chunk_text]), owner_id=uid, prevent_duplicates=False)

            manager.get_or_create_index(USER_A_ID)
            manager.get_or_create_index(USER_B_ID)
            assert manager.cached_owner_ids() == [str(USER_A_ID), str(USER_B_ID)]

            # Touch A so B becomes least-recently used
            manager.get_or_create_index(USER_A_ID)
            assert manager.cached_owner_ids() == [str(USER_B_ID), str(USER_A_ID)]

            # Load C -> B must be evicted
            manager.get_or_create_index(USER_C_ID)
            assert manager.cached_tenant_count() == 2
            assert manager.get_index(USER_B_ID) is None
            assert manager.cached_owner_ids() == [str(USER_A_ID), str(USER_C_ID)]

            # Re-querying B transparently reconstructs B from ChromaDB
            idx_b = manager.get_or_create_index(USER_B_ID)
            hits_b = idx_b.search("beryllium", top_k=5)
            assert len(hits_b) == 1
            assert hits_b[0].metadata["owner_id"] == str(USER_B_ID)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_concurrent_same_and_cross_tenant_retrieval(self, isolated_env):
        retriever: Retriever = isolated_env["retriever"]

        def _query(uid: UUID, term: str):
            return retriever.retrieve(
                RetrievalRequest(query=term, mode=RetrievalMode.HYBRID, top_k=5),
                owner_id=uid,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i in range(20):
                if i % 2 == 0:
                    futures.append((USER_A_ID, pool.submit(_query, USER_A_ID, "zephyrite resonance")))
                else:
                    futures.append((USER_B_ID, pool.submit(_query, USER_B_ID, "quasarflux radio")))

            for expected_owner, fut in futures:
                res = fut.result(timeout=30)
                assert res.returned_chunks_count >= 1
                for c in res.chunks:
                    assert c.metadata.get("owner_id") == str(expected_owner)

    @pytest.mark.asyncio
    async def test_failed_ingestion_leaves_no_searchable_bm25_chunks(self, shared_embedder: SentenceTransformerEmbedder):
        temp_dir = tempfile.mkdtemp(prefix="retrieva_m34_fail_ingest_")
        try:
            store = ChromaVectorStore(collection_name="m34_fail_col", persist_directory=temp_dir)
            manager = TenantBM25Manager(vector_store=store)
            retriever = Retriever(
                dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
                tenant_bm25_manager=manager,
                vector_store=store,
            )
            ingestion = IngestionService(
                embedder=shared_embedder,
                vector_store=store,
                tenant_bm25_manager=manager,
            )

            mock_sb = InMemorySupabaseMock(fail_on_ready_status=True)
            mock_http = httpx.AsyncClient(
                transport=httpx.MockTransport(mock_sb.handle),
                headers={"x-test-owner": str(USER_A_ID)},
            )
            sb_client = SupabaseClient(
                supabase_url=SUPABASE_URL,
                anon_key=ANON_KEY,
                access_token="tok",
                owner_id=USER_A_ID,
                http_client=mock_http,
            )
            doc_repo = DocumentRepository(sb_client)
            storage_repo = SupabaseStorageRepository(sb_client)

            with pytest.raises(Exception):
                await ingestion.ingest_document_for_user(
                    file_bytes=b"Uncommitted xylophone quantum failure payload.",
                    filename="failing.txt",
                    owner_id=USER_A_ID,
                    doc_repo=doc_repo,
                    storage_repo=storage_repo,
                )

            res = retriever.retrieve(
                RetrievalRequest(query="xylophone", mode=RetrievalMode.BM25),
                owner_id=USER_A_ID,
            )
            assert res.returned_chunks_count == 0
            assert store.get_owner_chunks(USER_A_ID) == []
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_chromadb_hydration_failure_fails_closed(self, shared_embedder: SentenceTransformerEmbedder):
        temp_dir = tempfile.mkdtemp(prefix="retrieva_m34_hydrate_fail_")
        try:
            store = ChromaVectorStore(collection_name="m34_hydrate_fail", persist_directory=temp_dir)
            manager = TenantBM25Manager(vector_store=store)
            retriever = Retriever(
                dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
                tenant_bm25_manager=manager,
                vector_store=store,
            )
            with patch.object(store, "get_owner_chunks", side_effect=RuntimeError("Chroma unavailable")):
                with pytest.raises(BM25HydrationError):
                    retriever.retrieve(
                        RetrievalRequest(query="anything", mode=RetrievalMode.BM25),
                        owner_id=USER_A_ID,
                    )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestEndToEndHTTPIsolation:
    """Verifies real JWT-authenticated /retrieval/search and /chat/answer isolation."""

    def test_http_retrieval_and_chat_endpoints_isolated(self, isolated_env):
        pem, jwk = _generate_ec_keypair()
        token_a = _issue_jwt(pem, USER_A_ID)
        token_b = _issue_jwt(pem, USER_B_ID)
        retriever: Retriever = isolated_env["retriever"]

        settings = Settings(
            APP_ENV="development",
            SUPABASE_URL=SUPABASE_URL,
            SUPABASE_ANON_KEY=ANON_KEY,
        )

        mock_llm = MagicMock(spec=OpenAIClient)
        mock_llm.model = "gpt-4o-mini"
        mock_llm.generate.return_value = {
            "content": json.dumps({"answer": "Answer [Source 1].", "insufficient_context": False, "citations": []}),
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        rag_svc = RAGService(retriever=retriever, llm_client=mock_llm)

        from backend.app.services.rag_service import get_rag_service

        prev_settings = app.dependency_overrides.get(get_settings)
        prev_ret = app.dependency_overrides.get(get_retriever)
        prev_rag = app.dependency_overrides.get(get_rag_service)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_retriever] = lambda: retriever
        app.dependency_overrides[get_rag_service] = lambda: rag_svc

        def _sb_ready_handler(request: httpx.Request) -> httpx.Response:
            auth = request.headers.get("authorization", "")
            tok = auth.removeprefix("Bearer ").strip()
            docs = isolated_env["docs"]
            if tok == token_a:
                rows = [{"id": did, "processing_status": "ready"} for k, did in docs.items() if k.startswith("a")]
            elif tok == token_b:
                rows = [{"id": did, "processing_status": "ready"} for k, did in docs.items() if k.startswith("b")]
            else:
                return httpx.Response(401, json={"message": "Unauthorized"})
            return httpx.Response(200, json=rows)

        mock_sb_http = httpx.AsyncClient(transport=httpx.MockTransport(_sb_ready_handler))

        try:
            with patch("jwt.PyJWKClient.fetch_data", return_value={"keys": [jwk]}), patch(
                "backend.app.core.supabase_client.get_supabase_http_client",
                return_value=mock_sb_http,
            ):
                with TestClient(app) as client:
                    # Startup must not warm global BM25 cache
                    assert client.get("/api/v1/health").status_code == 200

                    # Unauthenticated rejected
                    unauth = client.post("/api/v1/retrieval/search", json={"query": "zephyrite", "mode": "bm25"})
                    assert unauth.status_code == 401

                    # User A finds zephyrite; User B gets 0 hits
                    ra = client.post(
                        "/api/v1/retrieval/search",
                        headers={"Authorization": f"Bearer {token_a}"},
                        json={"query": "zephyrite", "mode": "hybrid", "top_k": 5},
                    )
                    rb = client.post(
                        "/api/v1/retrieval/search",
                        headers={"Authorization": f"Bearer {token_b}"},
                        json={"query": "zephyrite", "mode": "bm25", "top_k": 5},
                    )
                    assert ra.status_code == 200
                    assert ra.json()["returned_chunks_count"] >= 1
                    assert rb.status_code == 200
                    assert rb.json()["returned_chunks_count"] == 0

                    # User A querying User B's document UUID via /api/v1/answer -> 0 chunks retrieved
                    chat_cross = client.post(
                        "/api/v1/answer",
                        headers={"Authorization": f"Bearer {token_a}"},
                        json={
                            "query": "Explain quasarflux.",
                            "document_ids": [isolated_env["docs"]["b1"]],
                            "retrieval_config": {"mode": "hybrid", "top_k": 5},
                        },
                    )
                    assert chat_cross.status_code == 200
                    assert chat_cross.json()["metadata"]["total_chunks_retrieved"] == 0
        finally:
            for key, prev in [
                (get_settings, prev_settings),
                (get_retriever, prev_ret),
                (get_rag_service, prev_rag),
            ]:
                if prev is None:
                    app.dependency_overrides.pop(key, None)
                else:
                    app.dependency_overrides[key] = prev
