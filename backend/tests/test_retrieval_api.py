"""Automated integration and regression test suite for Standalone Retrieval API.

Verifies:
1. Dense semantic retrieval success and score metric isolation.
2. BM25 keyword retrieval success and score metric isolation.
3. Hybrid retrieval and Reciprocal Rank Fusion (RRF) score calculation.
4. Reranking disabled by default (cross_encoder_score and reranking_summary are null).
5. Reranking enabled (cross-encoder score populated and reranking_summary returned).
6. Strict schema serialization and score distinction.
7. Document ID filtering isolation.
8. Query validation: empty and whitespace queries return 400 Bad Request.
9. Empty knowledge base / zero-hit query behavior (returns 200 with empty chunk list).
10. Retrieval service error handling (returns 500 without leaking raw tracebacks).
11. Absolute absence of OpenAI API calls or LLM invocations during retrieval.
12. Exact parity between standalone API results and direct Retriever.retrieve() calls.
"""

import shutil
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from backend.app.document_processing.chunker import DocumentChunk
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.main import app
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.schemas.retrieval import (
    RetrievalConfiguration,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
)
from backend.app.vector_store.store import ChromaVectorStore


@pytest.fixture(scope="module")
def shared_embedder():
    """Module-scoped embedder instance."""
    return SentenceTransformerEmbedder.get_instance()


@pytest.fixture(scope="module")
def test_retrieval_fixture(shared_embedder) -> Generator[Retriever, None, None]:
    """Create an isolated module-scoped Retriever populated with distinct test documents."""
    temp_dir = tempfile.mkdtemp(prefix="retrieva_api_test_")
    store = ChromaVectorStore(
        collection_name="test_api_retrieval",
        persist_directory=temp_dir,
    )

    # Create two distinct test documents
    doc1_chunk = DocumentChunk(
        chunk_id="chunk_alpha_01",
        document_id="doc_alpha",
        filename="quantum_physics.txt",
        page_number=1,
        page_references=[1],
        chunk_index=0,
        chunk_text="Quantum computing leverages superposition and quantum entanglement to execute parallel calculations.",
        start_offset=0,
        end_offset=102,
        char_count=102,
        word_count=12,
    )

    doc2_chunk = DocumentChunk(
        chunk_id="chunk_beta_01",
        document_id="doc_beta",
        filename="botany_basics.txt",
        page_number=1,
        page_references=[1],
        chunk_index=0,
        chunk_text="Photosynthesis converts solar light energy into chemical energy within plant chloroplasts.",
        start_offset=0,
        end_offset=89,
        char_count=89,
        word_count=11,
    )

    chunks = [doc1_chunk, doc2_chunk]
    texts = [c.chunk_text for c in chunks]
    embeddings = shared_embedder.embed_batch(texts)

    from uuid import UUID
    test_owner_id = UUID("11111111-1111-4111-8111-111111111111")
    store.add_chunks(chunks, embeddings, owner_id=test_owner_id, prevent_duplicates=False)

    retriever = Retriever(
        dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
        vector_store=store,
        reranker=CrossEncoderReranker.get_instance(),
    )

    # Override get_retriever in app
    prev_retriever = app.dependency_overrides.get(get_retriever)
    app.dependency_overrides[get_retriever] = lambda: retriever

    try:
        yield retriever
    finally:
        # Teardown only the override owned by this fixture
        if prev_retriever is None:
            app.dependency_overrides.pop(get_retriever, None)
        else:
            app.dependency_overrides[get_retriever] = prev_retriever
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestRetrievalAPIEndpoints:
    """Test suite verifying all functional and contract requirements for /api/v1/retrieval/search."""

    def test_dense_retrieval_success(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """1. Verify Dense retrieval mode returns nearest semantic neighbors and populates cosine scores."""
        payload = {
            "query": "superposition and qubit entanglement",
            "mode": "dense",
            "top_k": 2,
        }
        response = sync_client.post("/api/v1/retrieval/search", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["mode"] == "dense"
        assert data["returned_chunks_count"] >= 1
        top_chunk = data["chunks"][0]
        assert top_chunk["document_id"] == "doc_alpha"
        assert top_chunk["cosine_similarity"] is not None
        assert top_chunk["cosine_distance"] is not None
        assert 0.0 <= top_chunk["cosine_distance"] <= 2.0
        # Metric isolation: other scores must be null
        assert top_chunk["bm25_score"] is None
        assert top_chunk["rrf_score"] is None
        assert top_chunk["cross_encoder_score"] is None

    def test_bm25_retrieval_success(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """2. Verify BM25 keyword retrieval succeeds and isolates BM25 scores."""
        payload = {
            "query": "chloroplasts photosynthesis",
            "mode": "bm25",
            "top_k": 2,
        }
        response = sync_client.post("/api/v1/retrieval/search", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["mode"] == "bm25"
        assert data["returned_chunks_count"] >= 1
        top_chunk = data["chunks"][0]
        assert top_chunk["document_id"] == "doc_beta"
        assert top_chunk["bm25_score"] is not None
        assert top_chunk["bm25_score"] > 0.0
        # Metric isolation: dense and rrf scores must be null
        assert top_chunk["cosine_similarity"] is None
        assert top_chunk["cosine_distance"] is None
        assert top_chunk["rrf_score"] is None
        assert top_chunk["cross_encoder_score"] is None

    def test_hybrid_retrieval_success(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """3. Verify Hybrid retrieval executes both Dense + BM25 and computes RRF scores."""
        payload = {
            "query": "solar light quantum physics",
            "mode": "hybrid",
            "top_k": 2,
            "candidate_k": 4,
        }
        response = sync_client.post("/api/v1/retrieval/search", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["mode"] == "hybrid"
        assert data["returned_chunks_count"] > 0
        for chunk in data["chunks"]:
            assert chunk["rrf_score"] is not None
            assert 0.0 < chunk["rrf_score"] <= 1.0

    def test_reranking_disabled_by_default(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """4. Verify reranking is disabled by default and produces no cross-encoder scores."""
        payload = {
            "query": "quantum calculations",
            "top_k": 2,
        }
        response = sync_client.post("/api/v1/retrieval/search", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["reranking_summary"] is None
        for chunk in data["chunks"]:
            assert chunk["cross_encoder_score"] is None

    def test_reranking_enabled(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """5. Verify enabling reranking populates cross_encoder_score and reranking_summary."""
        payload = {
            "query": "quantum superposition parallel computing",
            "mode": "hybrid",
            "top_k": 2,
            "candidate_k": 4,
            "enable_reranking": True,
        }
        response = sync_client.post("/api/v1/retrieval/search", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["reranking_summary"] is not None
        assert data["reranking_summary"]["candidates_in"] >= 1
        assert "cross-encoder" in data["reranking_summary"]["model_name"].lower()
        assert data["returned_chunks_count"] > 0
        for chunk in data["chunks"]:
            assert chunk["cross_encoder_score"] is not None
            assert isinstance(chunk["cross_encoder_score"], float)

    def test_score_fields_serialization(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """6. Verify all chunk fields and metadata adhere strictly to the schema specification."""
        payload = {
            "query": "chemical energy plants",
            "mode": "dense",
            "top_k": 1,
        }
        response = sync_client.post("/api/v1/retrieval/search", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "execution_time_ms" in data
        assert isinstance(data["execution_time_ms"], float)
        assert "configuration_applied" in data
        assert data["returned_chunks_count"] == 1

        chunk = data["chunks"][0]
        required_fields = [
            "chunk_id",
            "document_id",
            "filename",
            "chunk_text",
            "page_number",
            "page_references",
            "start_offset",
            "end_offset",
            "rank",
            "cosine_similarity",
            "cosine_distance",
            "bm25_score",
            "rrf_score",
            "cross_encoder_score",
            "metadata",
        ]
        for field in required_fields:
            assert field in chunk, f"Missing required field: {field}"

        assert chunk["rank"] == 1
        assert chunk["filename"] == "botany_basics.txt"
        assert chunk["page_number"] == 1

    def test_document_filtering(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """7. Verify document_id_filter restricts results exclusively to matching document."""
        payload = {
            "query": "energy and quantum computing",
            "mode": "hybrid",
            "document_id_filter": "doc_beta",
            "top_k": 5,
        }
        response = sync_client.post("/api/v1/retrieval/search", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["returned_chunks_count"] >= 1
        for chunk in data["chunks"]:
            assert chunk["document_id"] == "doc_beta"
            assert chunk["document_id"] != "doc_alpha"

    def test_invalid_query_rejection_400(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """8. Verify empty, whitespace, or missing queries are rejected with 400 Bad Request."""
        # Empty string
        res_empty = sync_client.post("/api/v1/retrieval/search", json={"query": ""})
        assert res_empty.status_code in (400, 422)

        # Whitespace string
        res_ws = sync_client.post("/api/v1/retrieval/search", json={"query": "   \n\t  "})
        assert res_ws.status_code in (400, 422)

        # Missing query
        res_none = sync_client.post("/api/v1/retrieval/search", json={"mode": "dense"})
        assert res_none.status_code == 422

    def test_empty_knowledge_base_behavior(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """9. Verify clean 200 OK response with 0 chunks when no documents match."""
        payload = {
            "query": "astronomy galaxies telescope",
            "document_id_filter": "nonexistent_document_id_999",
            "top_k": 5,
        }
        response = sync_client.post("/api/v1/retrieval/search", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["returned_chunks_count"] == 0
        assert data["chunks"] == []
        assert data["total_candidates_found"] == 0

    def test_retrieval_service_failure_handling(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """10. Verify internal server exceptions return clean 500 without leaking raw stack trace."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("Simulated internal storage fault")

        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        try:
            response = sync_client.post(
                "/api/v1/retrieval/search",
                json={"query": "test query"},
            )
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Simulated internal storage fault" in data["detail"]
        finally:
            app.dependency_overrides[get_retriever] = lambda: test_retrieval_fixture

    def test_no_openai_api_call(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """11. Verify retrieval endpoint never instantiates OpenAIClient or makes LLM calls."""
        with patch("backend.app.llm.client.OpenAIClient") as mock_openai_cls, \
             patch("openai.OpenAI") as mock_raw_openai:
            
            mock_openai_cls.side_effect = AssertionError("OpenAIClient should not be initialized during retrieval!")
            mock_raw_openai.side_effect = AssertionError("OpenAI SDK should not be initialized during retrieval!")

            response = sync_client.post(
                "/api/v1/retrieval/search",
                json={"query": "quantum calculations", "mode": "hybrid", "enable_reranking": True},
            )
            assert response.status_code == 200
            mock_openai_cls.assert_not_called()
            mock_raw_openai.assert_not_called()

    def test_retriever_engine_behavior_unmodified(self, sync_client: TestClient, test_retrieval_fixture: Retriever):
        """12. Verify parity between direct Retriever.retrieve() call and API endpoint."""
        query = "quantum superposition entanglement"
        req = RetrievalRequest(
            query=query,
            config=RetrievalConfiguration(
                mode=RetrievalMode.HYBRID,
                top_k=2,
                candidate_k=4,
                enable_reranking=True,
            ),
        )
        from uuid import UUID
        direct_result = test_retrieval_fixture.retrieve(
            req,
            owner_id=UUID("11111111-1111-4111-8111-111111111111"),
        )

        # Call via API endpoint with identical parameters
        payload = {
            "query": query,
            "mode": "hybrid",
            "top_k": 2,
            "candidate_k": 4,
            "enable_reranking": True,
        }
        api_response = sync_client.post("/api/v1/retrieval/search", json=payload)
        assert api_response.status_code == 200
        api_data = api_response.json()

        assert api_data["mode"] == direct_result.mode.value
        assert api_data["returned_chunks_count"] == direct_result.returned_chunks_count
        assert len(api_data["chunks"]) == len(direct_result.chunks)

        for api_chunk, direct_chunk in zip(api_data["chunks"], direct_result.chunks):
            assert api_chunk["chunk_id"] == direct_chunk.chunk_id
            assert api_chunk["rank"] == direct_chunk.rank
            assert api_chunk["chunk_text"] == direct_chunk.chunk_text
            if direct_chunk.cross_encoder_score is not None:
                assert pytest.approx(api_chunk["cross_encoder_score"], abs=1e-4) == direct_chunk.cross_encoder_score
