"""Comprehensive automated test suite for Phase 4: Retrieval and Semantic Search Engine.

Covers:
1. Query embedding validation, dimensionality, normalization, model reuse.
2. Dense semantic retrieval, top-k limits, cosine distance/similarity, threshold filtering.
3. Custom BM25 keyword index, TF-IDF calculation, length normalization, synchronization.
4. Hybrid retrieval and Reciprocal Rank Fusion (RRF) math and deduplication.
5. Cross-encoder reranking behavior and metadata preservation.
6. Retrieval orchestrator end-to-end integration and benchmark evaluation metrics.
"""

import math
import shutil
import tempfile
from typing import Any, Dict, List

import pytest

from backend.app.document_processing.chunker import DocumentChunk
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.retrieval.bm25 import BM25Index, tokenize_text
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.evaluation import (
    EVALUATION_DOCUMENTS,
    EVALUATION_QUERIES,
    calculate_hit_rate_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    run_retrieval_benchmark,
)
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.retriever import Retriever
from backend.app.schemas.retrieval import (
    RetrievalConfiguration,
    RetrievalMode,
    RetrievalRequest,
    RetrievedChunk,
)
from backend.app.vector_store.client import reset_client_cache
from backend.app.vector_store.store import ChromaVectorStore


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def shared_embedder():
    return SentenceTransformerEmbedder.get_instance()


@pytest.fixture
def isolated_vector_store():
    """Create an isolated persistent vector store in a temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="retrieva_test_retrieval_")
    store = ChromaVectorStore(
        collection_name="test_retrieval_collection",
        persist_directory=temp_dir,
    )
    yield store, temp_dir
    reset_client_cache()
    shutil.rmtree(temp_dir, ignore_errors=True)


def populate_store_with_sample_chunks(store: ChromaVectorStore, embedder: SentenceTransformerEmbedder):
    """Seed the vector store with distinct known chunks for testing."""
    sample_texts = [
        ("c1", "doc_python", "Python is an interpreted, high-level, general-purpose programming language.", 1),
        ("c2", "doc_python", "Python uses dynamic typing and garbage collection with reference counting.", 2),
        ("c3", "doc_rag", "Retrieval-Augmented Generation enhances language models with external database knowledge.", 1),
        ("c4", "doc_rag", "Dense vector embeddings represent text semantics in continuous geometric vector spaces.", 2),
        ("c5", "doc_chroma", "ChromaDB is an open-source vector database designed for AI application embeddings.", 1),
    ]

    chunks = []
    texts = []
    for cid, doc_id, txt, page in sample_texts:
        chunk = DocumentChunk(
            chunk_id=cid,
            document_id=doc_id,
            filename=f"{doc_id}.pdf",
            page_number=page,
            page_references=[page],
            chunk_index=len(chunks),
            chunk_text=txt,
            start_offset=0,
            end_offset=len(txt),
            char_count=len(txt),
            word_count=len(txt.split()),
        )
        chunks.append(chunk)
        texts.append(txt)

    embeddings = embedder.embed_batch(texts)
    store.add_chunks(chunks, embeddings, prevent_duplicates=False)
    return chunks


# ============================================================================
# 1. Query Embedding Tests
# ============================================================================

class TestQueryEmbedding:
    """Test suite for query embedding generation and validation."""

    def test_query_embedding_dimension_and_norm(self, shared_embedder):
        query = "What is Retrieval-Augmented Generation?"
        vec = shared_embedder.embed_text(query)

        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)
        # Verify L2 normalization
        norm = math.sqrt(sum(x ** 2 for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_query_validation_empty_string(self, shared_embedder):
        with pytest.raises(ValueError):
            shared_embedder.embed_text("")
        with pytest.raises(ValueError):
            shared_embedder.embed_text("   \n\t  ")

    def test_model_reuse_singleton(self):
        e1 = SentenceTransformerEmbedder.get_instance()
        e2 = SentenceTransformerEmbedder.get_instance()
        assert e1 is e2  # Verifies zero duplicate model weights allocated


# ============================================================================
# 2. Dense Semantic Retrieval Tests
# ============================================================================

class TestDenseRetrieval:
    """Test suite for ChromaDB dense vector semantic retrieval."""

    def test_dense_retrieval_nearest_neighbor(self, isolated_vector_store, shared_embedder):
        store, _ = isolated_vector_store
        populate_store_with_sample_chunks(store, shared_embedder)

        dense = DenseRetriever(embedder=shared_embedder, vector_store=store)
        results = dense.search(query="How does RAG help language models?", top_k=2)

        assert len(results) == 2
        # The top result should be the RAG chunk (c3)
        assert results[0].chunk_id == "c3"
        assert "Retrieval-Augmented Generation" in results[0].chunk_text
        assert results[0].rank == 1
        assert results[0].cosine_similarity is not None
        assert results[0].cosine_distance is not None
        # Cosine distance = 1 - cosine similarity
        assert results[0].cosine_distance == pytest.approx(1.0 - results[0].cosine_similarity, abs=1e-4)
        assert results[0].cosine_similarity > 0.35

    def test_dense_retrieval_top_k_bounds(self, isolated_vector_store, shared_embedder):
        store, _ = isolated_vector_store
        populate_store_with_sample_chunks(store, shared_embedder)

        dense = DenseRetriever(embedder=shared_embedder, vector_store=store)
        res_k1 = dense.search(query="Python programming", top_k=1)
        assert len(res_k1) == 1

        res_k3 = dense.search(query="Python programming", top_k=3)
        assert len(res_k3) == 3

    def test_dense_retrieval_empty_collection(self, isolated_vector_store, shared_embedder):
        store, _ = isolated_vector_store
        dense = DenseRetriever(embedder=shared_embedder, vector_store=store)
        results = dense.search(query="Anything", top_k=5)
        assert results == []

    def test_dense_retrieval_document_id_filter(self, isolated_vector_store, shared_embedder):
        store, _ = isolated_vector_store
        populate_store_with_sample_chunks(store, shared_embedder)

        dense = DenseRetriever(embedder=shared_embedder, vector_store=store)
        # Filter strictly for doc_chroma
        results = dense.search(query="programming language", top_k=5, document_id_filter="doc_chroma")
        assert len(results) == 1
        assert results[0].document_id == "doc_chroma"
        assert results[0].chunk_id == "c5"

    def test_dense_retrieval_similarity_threshold_filtering(self, isolated_vector_store, shared_embedder):
        store, _ = isolated_vector_store
        populate_store_with_sample_chunks(store, shared_embedder)

        dense = DenseRetriever(embedder=shared_embedder, vector_store=store)
        # High threshold should filter out less relevant chunks
        high_thresh_results = dense.search(
            query="Python programming language",
            top_k=5,
            similarity_threshold=0.85,  # Very strict
        )
        # Should return fewer results than unrestricted top_k
        assert len(high_thresh_results) < 5
        for r in high_thresh_results:
            assert r.cosine_similarity >= 0.85


# ============================================================================
# 3. Custom BM25 Keyword Retrieval Tests
# ============================================================================

class TestBM25KeywordRetrieval:
    """Test suite for Okapi BM25 implementation."""

    def test_tokenization(self):
        tokens = tokenize_text("Okapi BM25: Fast, Deterministic & Exact (2026)!")
        assert "okapi" in tokens
        assert "bm25" in tokens
        assert "fast" in tokens
        assert "deterministic" in tokens
        assert "exact" in tokens
        assert "2026" in tokens

    def test_bm25_exact_keyword_ranking(self):
        index = BM25Index(k1=1.5, b=0.75)
        chunks = [
            {"chunk_id": "b1", "document_id": "d1", "text": "PostgreSQL database relational ACID transactions."},
            {"chunk_id": "b2", "document_id": "d1", "text": "Vector database ChromaDB stores embeddings with HNSW."},
            {"chunk_id": "b3", "document_id": "d2", "text": "Python programming language with dynamic typing."},
        ]
        index.build_index(chunks)
        assert index.doc_count == 3

        results = index.search(query="PostgreSQL ACID", top_k=2)
        assert len(results) == 1
        assert results[0].chunk_id == "b1"
        assert results[0].bm25_score > 0.0

    def test_bm25_idf_properties(self):
        index = BM25Index()
        chunks = [
            {"chunk_id": "1", "text": "apple banana cherry"},
            {"chunk_id": "2", "text": "apple banana"},
            {"chunk_id": "3", "text": "apple date"},
        ]
        index.build_index(chunks)

        # 'cherry' appears in 1 doc, 'banana' in 2 docs, 'apple' in all 3 docs
        idf_cherry = index.calculate_idf("cherry")
        idf_banana = index.calculate_idf("banana")
        idf_apple = index.calculate_idf("apple")

        # Rarer terms must have higher IDF weight
        assert idf_cherry > idf_banana
        assert idf_banana > idf_apple
        assert idf_apple > 0.0  # +1 floor prevents zero or negative IDF

    def test_bm25_empty_query_and_corpus(self):
        index = BM25Index()
        assert index.search("query", top_k=5) == []

        index.build_index([{"chunk_id": "1", "text": "valid chunk"}])
        assert index.search("", top_k=5) == []
        assert index.search("   ", top_k=5) == []

    def test_bm25_document_removal(self):
        index = BM25Index()
        chunks = [
            {"chunk_id": "c1", "document_id": "doc_A", "text": "Chunk from document A."},
            {"chunk_id": "c2", "document_id": "doc_B", "text": "Chunk from document B."},
        ]
        index.build_index(chunks)
        assert index.doc_count == 2

        removed = index.remove_document("doc_A")
        assert removed == 1
        assert index.doc_count == 1
        # Terms unique to doc_A should yield no results
        assert index.search("A", top_k=5) == []
        # Chunks for doc_A must not be in search results
        retrieved_ids = [c.chunk_id for c in index.search("document A", top_k=5)]
        assert "c1" not in retrieved_ids
        assert "c2" in retrieved_ids


# ============================================================================
# 4. Hybrid Retrieval & Reciprocal Rank Fusion Tests
# ============================================================================

class TestReciprocalRankFusion:
    """Test suite for Reciprocal Rank Fusion aggregation."""

    def test_rrf_scoring_math(self):
        dense_chunk = RetrievedChunk(
            chunk_id="doc1",
            document_id="d1",
            filename="file.txt",
            page_number=1,
            page_references=[1],
            chunk_text="text",
            start_offset=0,
            end_offset=4,
            rank=1,
            cosine_similarity=0.90,
        )
        bm25_chunk = RetrievedChunk(
            chunk_id="doc1",
            document_id="d1",
            filename="file.txt",
            page_number=1,
            page_references=[1],
            chunk_text="text",
            start_offset=0,
            end_offset=4,
            rank=2,
            bm25_score=4.5,
        )

        fused = reciprocal_rank_fusion([dense_chunk], [bm25_chunk], rrf_k=60, top_k=1)
        assert len(fused) == 1
        # RRF = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639344 + 0.01612903 = 0.03252247
        expected_rrf = (1.0 / 61.0) + (1.0 / 62.0)
        assert fused[0].rrf_score == pytest.approx(expected_rrf, abs=1e-6)
        assert fused[0].cosine_similarity == 0.90
        assert fused[0].bm25_score == 4.5

    def test_rrf_deduplication_and_order(self):
        c_a = RetrievedChunk(chunk_id="A", document_id="d", filename="f", page_number=1, chunk_text="A", start_offset=0, end_offset=1, rank=1)
        c_b = RetrievedChunk(chunk_id="B", document_id="d", filename="f", page_number=1, chunk_text="B", start_offset=0, end_offset=1, rank=2)
        c_c = RetrievedChunk(chunk_id="C", document_id="d", filename="f", page_number=1, chunk_text="C", start_offset=0, end_offset=1, rank=1)

        # Dense: [A (rank 1), B (rank 2)]
        # BM25:  [C (rank 1), A (rank 2)]
        # RRF for A = 1/61 + 1/62 = 0.03252
        # RRF for C = 1/61 = 0.01639
        # RRF for B = 1/62 = 0.01613
        fused = reciprocal_rank_fusion([c_a, c_b], [c_c, c_a], rrf_k=60, top_k=3)

        assert len(fused) == 3
        assert fused[0].chunk_id == "A"  # Highest combined rank
        assert fused[1].chunk_id == "C"
        assert fused[2].chunk_id == "B"
        assert fused[0].rank == 1
        assert fused[1].rank == 2
        assert fused[2].rank == 3


# ============================================================================
# 5. Cross-Encoder Reranking Tests
# ============================================================================

class TestCrossEncoderReranking:
    """Test suite for CrossEncoder reranking."""

    @pytest.fixture(scope="class")
    def reranker(self):
        return CrossEncoderReranker.get_instance()

    def test_reranker_reordering(self, reranker):
        query = "What is the capital of France?"
        c_relevant = RetrievedChunk(
            chunk_id="c_rel",
            document_id="d1",
            filename="geography.txt",
            page_number=1,
            chunk_text="Paris is the capital and most populous city of France.",
            start_offset=0,
            end_offset=54,
            rank=2,  # Originally ranked second
        )
        c_irrelevant = RetrievedChunk(
            chunk_id="c_irrel",
            document_id="d2",
            filename="fruits.txt",
            page_number=1,
            chunk_text="The French horn is a brass musical instrument made of tubing wrapped into a coil.",
            start_offset=0,
            end_offset=82,
            rank=1,  # Originally ranked first by lexical overlap
        )

        reranked, summary = reranker.rerank(query, [c_irrelevant, c_relevant], top_k=2)

        assert len(reranked) == 2
        # The cross-encoder must promote the factual answer to rank 1
        assert reranked[0].chunk_id == "c_rel"
        assert reranked[0].rank == 1
        assert reranked[0].cross_encoder_score > reranked[1].cross_encoder_score
        assert summary.candidates_in == 2
        assert summary.candidates_out == 2


# ============================================================================
# 6. End-to-End Retrieval Orchestrator Tests
# ============================================================================

class TestRetrievalOrchestrator:
    """Test suite for the central Retriever service coordinating all modes."""

    @pytest.fixture
    def test_retriever(self, isolated_vector_store, shared_embedder):
        store, _ = isolated_vector_store
        populate_store_with_sample_chunks(store, shared_embedder)

        retriever = Retriever(
            dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
            bm25_index=BM25Index(),
            vector_store=store,
            auto_sync_bm25=True,
        )
        return retriever

    def test_dense_mode_retrieval(self, test_retriever):
        req = RetrievalRequest(
            query="vector database storage",
            config=RetrievalConfiguration(mode=RetrievalMode.DENSE, top_k=2),
        )
        res = test_retriever.retrieve(req)

        assert res.mode == RetrievalMode.DENSE
        assert len(res.chunks) == 2
        assert res.chunks[0].cosine_similarity is not None
        assert res.execution_time_ms > 0.0

    def test_bm25_mode_retrieval(self, test_retriever):
        req = RetrievalRequest(
            query="dynamic typing reference counting",
            config=RetrievalConfiguration(mode=RetrievalMode.BM25, top_k=2),
        )
        res = test_retriever.retrieve(req)

        assert res.mode == RetrievalMode.BM25
        assert len(res.chunks) >= 1
        assert res.chunks[0].chunk_id == "c2"  # Exact terms match c2
        assert res.chunks[0].bm25_score > 0.0

    def test_hybrid_mode_retrieval(self, test_retriever):
        req = RetrievalRequest(
            query="Python language garbage collection",
            config=RetrievalConfiguration(
                mode=RetrievalMode.HYBRID,
                top_k=3,
                candidate_k=5,
            ),
        )
        res = test_retriever.retrieve(req)

        assert res.mode == RetrievalMode.HYBRID
        assert len(res.chunks) <= 3
        for c in res.chunks:
            assert c.rrf_score is not None

    def test_hybrid_with_reranking(self, test_retriever):
        req = RetrievalRequest(
            query="vector embeddings semantic space",
            config=RetrievalConfiguration(
                mode=RetrievalMode.HYBRID,
                top_k=2,
                candidate_k=4,
                enable_reranking=True,
            ),
        )
        res = test_retriever.retrieve(req)

        assert res.mode == RetrievalMode.HYBRID
        assert len(res.chunks) == 2
        assert res.reranking_summary is not None
        assert res.chunks[0].cross_encoder_score is not None


# ============================================================================
# 7. Retrieval Evaluation Benchmark Tests
# ============================================================================

class TestRetrievalBenchmark:
    """Test suite executing the benchmark evaluation across Dense, BM25, and Hybrid."""

    def test_benchmark_execution(self, isolated_vector_store, shared_embedder):
        store, _ = isolated_vector_store

        # Populate evaluation documents into vector store
        chunks = []
        texts = []
        for d in EVALUATION_DOCUMENTS:
            meta = dict(d["metadata"])
            txt = d["text"]
            chunk = DocumentChunk(
                chunk_id=d["chunk_id"],
                document_id=d["document_id"],
                filename=d["filename"],
                page_number=meta.get("page_number", 1),
                page_references=meta.get("page_references", [1]),
                chunk_index=len(chunks),
                chunk_text=txt,
                start_offset=0,
                end_offset=len(txt),
                char_count=len(txt),
                word_count=len(txt.split()),
            )
            chunks.append(chunk)
            texts.append(txt)

        embeddings = shared_embedder.embed_batch(texts)
        store.add_chunks(chunks, embeddings, prevent_duplicates=False)

        retriever = Retriever(
            dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
            bm25_index=BM25Index(),
            vector_store=store,
            auto_sync_bm25=True,
        )

        results = run_retrieval_benchmark(retriever, queries=EVALUATION_QUERIES, eval_ks=[3, 5])

        assert results["dataset_size"] == len(EVALUATION_QUERIES)
        assert results["corpus_size"] == len(EVALUATION_DOCUMENTS)
        assert results["evaluated_ks"] == [3, 5]
        for mode_key in ["dense", "bm25", "hybrid"]:
            metrics = results["modes"][mode_key]
            assert "precision@3" in metrics
            assert "precision@5" in metrics
            assert "recall@3" in metrics
            assert "recall@5" in metrics
            assert "mrr" in metrics
            assert "hit_rate@3" in metrics
            assert "hit_rate@5" in metrics
            # Hit rate should be substantial (> 0.50) across all modes
            assert metrics["hit_rate@5"] >= 0.50
            assert metrics["mrr"] > 0.40

    def test_manual_metric_verification_example_1(self):
        """Verify metric calculation formulas manually on Example 1:
        1 relevant document ('c1').
        Retrieved order: ['c2', 'c1', 'c3', 'c4', 'c5'].
        """
        retrieved = ["c2", "c1", "c3", "c4", "c5"]
        relevant = {"c1"}

        # At K=3: retrieved = ['c2', 'c1', 'c3'] -> 1 hit
        assert calculate_precision_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3, rel=1e-4)
        assert calculate_recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0, rel=1e-4)
        assert calculate_hit_rate_at_k(retrieved, relevant, k=3) == 1.0

        # At K=5: retrieved = ['c2', 'c1', 'c3', 'c4', 'c5'] -> 1 hit
        assert calculate_precision_at_k(retrieved, relevant, k=5) == pytest.approx(1 / 5, rel=1e-4)
        assert calculate_recall_at_k(retrieved, relevant, k=5) == pytest.approx(1.0, rel=1e-4)
        assert calculate_hit_rate_at_k(retrieved, relevant, k=5) == 1.0

        # Reciprocal Rank: 'c1' is at 1-based rank 2 -> 1 / 2 = 0.5
        assert calculate_reciprocal_rank(retrieved, relevant) == pytest.approx(0.5, rel=1e-4)

    def test_manual_metric_verification_example_2(self):
        """Verify metric calculation formulas manually on Example 2:
        3 relevant documents: {'c1', 'c3', 'c5'}.
        Retrieved order: ['c2', 'c4', 'c6', 'c1', 'c7'].
        """
        retrieved = ["c2", "c4", "c6", "c1", "c7"]
        relevant = {"c1", "c3", "c5"}

        # At K=3: retrieved = ['c2', 'c4', 'c6'] -> 0 hits
        assert calculate_precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.0, abs=1e-4)
        assert calculate_recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.0, abs=1e-4)
        assert calculate_hit_rate_at_k(retrieved, relevant, k=3) == 0.0

        # At K=5: retrieved = ['c2', 'c4', 'c6', 'c1', 'c7'] -> 1 hit ('c1' at rank 4)
        assert calculate_precision_at_k(retrieved, relevant, k=5) == pytest.approx(1 / 5, rel=1e-4)
        assert calculate_recall_at_k(retrieved, relevant, k=5) == pytest.approx(1 / 3, rel=1e-4)
        assert calculate_hit_rate_at_k(retrieved, relevant, k=5) == 1.0

        # Reciprocal Rank: first hit 'c1' is at rank 4 -> 1 / 4 = 0.25
        assert calculate_reciprocal_rank(retrieved, relevant) == pytest.approx(0.25, rel=1e-4)
