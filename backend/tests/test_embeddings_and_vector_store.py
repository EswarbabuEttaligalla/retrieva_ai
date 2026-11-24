"""Comprehensive automated test suite for Phase 3: Dense Embeddings and Vector Storage.

Covers:
1. SentenceTransformerEmbedder loading, dimensionality (384), normalization, finite values, batching.
2. Embedding validation (dimension, finite values, L2 norm, empty string handling).
3. ChromaVectorStore client creation, collection with cosine distance, metadata serialization.
4. Duplicate prevention (document ID & content hash).
5. Document deletion and collection statistics.
6. Persistence verification across client restarts.
7. End-to-end integration: PDF/TXT extraction -> cleaning -> chunking -> embedding -> ChromaDB storage.
"""

import math
import shutil
import tempfile
from pathlib import Path
from typing import List

import pytest
import fitz  # PyMuPDF

from backend.app.document_processing.chunker import TextChunker, DocumentChunk
from backend.app.document_processing.identifier import compute_content_hash, generate_document_id
from backend.app.embeddings.embedder import (
    SentenceTransformerEmbedder,
    EmbeddingModelLoadError,
    EmbeddingGenerationError,
)
from backend.app.embeddings.validator import (
    validate_embedding,
    validate_batch_embeddings,
    validate_embedding_dimension,
    validate_finite_vector,
    calculate_l2_norm,
    is_l2_normalized,
    EmbeddingDimensionError,
    EmbeddingValueError,
    EmbeddingNormalizationError,
)
from backend.app.vector_store.client import get_chroma_client, reset_client_cache
from backend.app.vector_store.store import (
    ChromaVectorStore,
    DocumentAlreadyExistsError,
    VectorStoreInsertionError,
)
from backend.app.services.ingestion_service import IngestionService


# ============================================================================
# Helpers
# ============================================================================

def make_test_pdf(page_texts: List[str]) -> bytes:
    """Create an in-memory PDF with specified text per page."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page(width=595, height=842)
        if text:
            page.insert_text((50, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ============================================================================
# 1. Embedding Validator Tests
# ============================================================================

class TestEmbeddingValidator:
    """Test suite for numerical and dimensional embedding vector validators."""

    def test_validate_dimension_success(self):
        vec = [0.1] * 384
        validate_embedding_dimension(vec, expected_dim=384)

    def test_validate_dimension_failure(self):
        vec = [0.1] * 128
        with pytest.raises(EmbeddingDimensionError) as exc_info:
            validate_embedding_dimension(vec, expected_dim=384)
        assert "expected 384, got 128" in str(exc_info.value)

    def test_validate_finite_vector_success(self):
        vec = [0.0, -1.5, 3.14159, 100.0]
        validate_finite_vector(vec)

    def test_validate_finite_vector_nan(self):
        vec = [1.0, float("nan"), 2.0]
        with pytest.raises(EmbeddingValueError) as exc_info:
            validate_finite_vector(vec)
        assert "contains NaN" in str(exc_info.value)

    def test_validate_finite_vector_infinity(self):
        vec = [1.0, float("inf"), 2.0]
        with pytest.raises(EmbeddingValueError) as exc_info:
            validate_finite_vector(vec)
        assert "contains infinite value" in str(exc_info.value)

    def test_calculate_l2_norm(self):
        # 3-4-5 triangle in 2D
        vec = [3.0, 4.0]
        assert calculate_l2_norm(vec) == pytest.approx(5.0, abs=1e-6)

        # Unit vector
        unit_vec = [1.0, 0.0, 0.0]
        assert calculate_l2_norm(unit_vec) == pytest.approx(1.0, abs=1e-6)

    def test_is_l2_normalized(self):
        norm_val = 1.0 / math.sqrt(384)
        normalized_vec = [norm_val] * 384
        assert is_l2_normalized(normalized_vec, tolerance=1e-4) is True

        unnormalized_vec = [1.0] * 384
        assert is_l2_normalized(unnormalized_vec, tolerance=1e-4) is False

    def test_validate_embedding_comprehensive_success(self):
        norm_val = 1.0 / math.sqrt(384)
        vec = [norm_val] * 384
        validate_embedding(vec, expected_dim=384, check_normalization=True)

    def test_validate_embedding_fails_unnormalized(self):
        vec = [0.5] * 384
        with pytest.raises(EmbeddingNormalizationError):
            validate_embedding(vec, expected_dim=384, check_normalization=True)


# ============================================================================
# 2. Embedding Model Tests (Sentence Transformers)
# ============================================================================

class TestSentenceTransformerEmbedder:
    """Test suite for SentenceTransformerEmbedder using all-MiniLM-L6-v2."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Provide a shared embedder instance for fast class execution."""
        return SentenceTransformerEmbedder.get_instance()

    def test_model_loading_and_attributes(self, embedder):
        assert embedder is not None
        assert embedder.get_embedding_dimension() == 384
        assert embedder.get_max_sequence_length() == 256
        assert "all-MiniLM-L6-v2" in embedder.model_name

    def test_embed_single_text_dimension_and_norm(self, embedder):
        text = "Retrieval-Augmented Generation improves LLM factual accuracy with external knowledge."
        vector = embedder.embed_text(text)

        assert isinstance(vector, list)
        assert len(vector) == 384
        assert all(isinstance(x, float) for x in vector)
        # Verify L2 normalization: ||v|| == 1.0
        norm = calculate_l2_norm(vector)
        assert norm == pytest.approx(1.0, abs=1e-4)
        # Verify all numbers are finite
        assert all(math.isfinite(x) for x in vector)

    def test_embed_empty_input_rejected(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed_text("")
        with pytest.raises(ValueError):
            embedder.embed_text("   \n\t  ")

    def test_embed_batch(self, embedder):
        texts = [
            "Dense vector embeddings capture semantic meaning.",
            "ChromaDB stores high-dimensional vectors on disk.",
            "Cosine similarity measures the angle between two unit vectors.",
        ]
        vectors = embedder.embed_batch(texts, batch_size=2)

        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 384
            assert calculate_l2_norm(vec) == pytest.approx(1.0, abs=1e-4)
            assert all(math.isfinite(x) for x in vec)

    def test_semantic_similarity_property(self, embedder):
        """Verify that semantically similar sentences have higher cosine similarity."""
        v1 = embedder.embed_text("The fast brown dog jumped over the lazy canine.")
        v2 = embedder.embed_text("A quick auburn hound leaped above the sleeping pup.")
        v3 = embedder.embed_text("Quantum electrodynamics explains photon-electron interactions in vacuum.")

        # Dot product of normalized vectors equals cosine similarity
        sim_12 = sum(a * b for a, b in zip(v1, v2))
        sim_13 = sum(a * b for a, b in zip(v1, v3))

        # Related sentences should have significantly higher similarity than physics sentence
        assert sim_12 > sim_13
        assert sim_12 > 0.55
        assert sim_13 < 0.35

    def test_sequence_length_inspection_and_token_count(self, embedder):
        max_seq = embedder.get_max_sequence_length()
        assert max_seq == 256
        sample_text = "This is a short sentence for token counting."
        token_count = embedder.count_tokens(sample_text)
        assert token_count > 0
        assert token_count < max_seq


# ============================================================================
# 3. Persistent ChromaDB Vector Store Tests
# ============================================================================

class TestChromaVectorStore:
    """Test suite for ChromaVectorStore persistence, metadata, and operations."""

    @pytest.fixture
    def temp_store(self):
        """Provide a clean isolated persistent vector store in a temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix="retrieva_chroma_test_")
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir,
        )
        yield store, temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def make_dummy_chunk(
        self,
        doc_id: str,
        chunk_idx: int,
        text: str,
        page_num: int = 1,
    ) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
            document_id=doc_id,
            filename="sample.pdf",
            page_number=page_num,
            page_references=[page_num],
            chunk_index=chunk_idx,
            chunk_text=text,
            start_offset=0,
            end_offset=len(text),
            char_count=len(text),
            word_count=len(text.split()),
        )

    def test_collection_creation_and_attributes(self, temp_store):
        store, _ = temp_store
        assert store.count() == 0
        assert store.collection_name == "test_collection"
        stats = store.get_statistics()
        assert stats["distance_metric"] == "cosine"
        assert stats["embedding_dimension"] == 384
        assert stats["total_chunks"] == 0

    def test_add_chunks_with_explicit_embeddings(self, temp_store):
        store, _ = temp_store
        doc_id = "doc_test_1"
        chunks = [
            self.make_dummy_chunk(doc_id, 0, "First chunk text about document indexing."),
            self.make_dummy_chunk(doc_id, 1, "Second chunk text about vector similarity."),
        ]
        norm_val = 1.0 / math.sqrt(384)
        embeddings = [[norm_val] * 384, [norm_val] * 384]

        count = store.add_chunks(chunks, embeddings, content_hash="hash_123")
        assert count == 2
        assert store.count() == 2
        assert store.has_document(doc_id) is True
        assert store.has_content_hash("hash_123") is True

    def test_metadata_serialization_and_deserialization(self, temp_store):
        store, _ = temp_store
        doc_id = "doc_meta_test"
        chunk = DocumentChunk(
            chunk_id=f"{doc_id}_chunk_0",
            document_id=doc_id,
            filename="multi_page.pdf",
            page_number=1,
            page_references=[1, 2],  # Multi-page reference
            chunk_index=0,
            chunk_text="A chunk spanning page one and two.",
            start_offset=10,
            end_offset=44,
            char_count=34,
            word_count=7,
        )
        norm_val = 1.0 / math.sqrt(384)
        store.add_chunks([chunk], [[norm_val] * 384], content_hash="hash_multi")

        retrieved_chunks = store.get_document_chunks(doc_id)
        assert len(retrieved_chunks) == 1
        meta = retrieved_chunks[0]["metadata"]

        assert meta["document_id"] == doc_id
        assert meta["filename"] == "multi_page.pdf"
        assert meta["page_number"] == 1
        assert meta["page_references"] == [1, 2]  # Reconstructed List[int]
        assert meta["content_hash"] == "hash_multi"
        assert meta["start_offset"] == 10
        assert meta["end_offset"] == 44

    def test_duplicate_document_prevention(self, temp_store):
        store, _ = temp_store
        doc_id = "doc_dup"
        chunk = self.make_dummy_chunk(doc_id, 0, "Duplicate test text.")
        norm_val = 1.0 / math.sqrt(384)

        store.add_chunks([chunk], [[norm_val] * 384], content_hash="hash_dup")

        # Second attempt with same doc_id should raise DocumentAlreadyExistsError
        with pytest.raises(DocumentAlreadyExistsError):
            store.add_chunks([chunk], [[norm_val] * 384], content_hash="different_hash")

        # Third attempt with same content_hash should raise DocumentAlreadyExistsError
        chunk2 = self.make_dummy_chunk("different_doc_id", 0, "Duplicate test text.")
        with pytest.raises(DocumentAlreadyExistsError):
            store.add_chunks([chunk2], [[norm_val] * 384], content_hash="hash_dup")

    def test_delete_document(self, temp_store):
        store, _ = temp_store
        doc_id = "doc_to_delete"
        chunks = [
            self.make_dummy_chunk(doc_id, 0, "Chunk 1 to delete"),
            self.make_dummy_chunk(doc_id, 1, "Chunk 2 to delete"),
        ]
        norm_val = 1.0 / math.sqrt(384)
        store.add_chunks(chunks, [[norm_val] * 384, [norm_val] * 384])

        assert store.count() == 2
        deleted = store.delete_document(doc_id)
        assert deleted == 2
        assert store.count() == 0
        assert store.has_document(doc_id) is False

    def test_persistence_across_client_restarts(self):
        """Verify data persists on disk and is accessible after closing and reopening Chroma client."""
        temp_dir = tempfile.mkdtemp(prefix="retrieva_chroma_restart_")
        try:
            # 1. Initialize store and add records
            store1 = ChromaVectorStore(collection_name="persist_col", persist_directory=temp_dir)
            chunk = DocumentChunk(
                chunk_id="restart_chunk_0",
                document_id="doc_restart",
                filename="restart.pdf",
                page_number=1,
                page_references=[1],
                chunk_index=0,
                chunk_text="Persistent content must survive application shutdown and restart.",
                start_offset=0,
                end_offset=65,
                char_count=65,
                word_count=8,
            )
            norm_val = 1.0 / math.sqrt(384)
            store1.add_chunks([chunk], [[norm_val] * 384], content_hash="hash_restart")
            assert store1.count() == 1

            # 2. Simulate complete shutdown: reset client cache and delete object
            reset_client_cache()
            del store1

            # 3. Simulate process restart: initialize completely new client pointing to same folder
            store2 = ChromaVectorStore(collection_name="persist_col", persist_directory=temp_dir)
            assert store2.count() == 1
            assert store2.has_document("doc_restart") is True

            chunks = store2.get_document_chunks("doc_restart")
            assert len(chunks) == 1
            assert chunks[0]["text"] == "Persistent content must survive application shutdown and restart."
            assert chunks[0]["metadata"]["content_hash"] == "hash_restart"
            assert chunks[0]["metadata"]["page_references"] == [1]

        finally:
            reset_client_cache()
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 4. End-to-End Ingestion Integration Tests
# ============================================================================

class TestIngestionServiceIntegration:
    """End-to-end integration tests connecting extraction, cleaning, chunking, embedding, and storage."""

    @pytest.fixture
    def test_service(self):
        temp_dir = tempfile.mkdtemp(prefix="retrieva_ingest_service_")
        embedder = SentenceTransformerEmbedder.get_instance()
        vector_store = ChromaVectorStore(
            collection_name="e2e_collection",
            persist_directory=temp_dir,
        )
        service = IngestionService(embedder=embedder, vector_store=vector_store)
        yield service, temp_dir
        reset_client_cache()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_e2e_pdf_ingestion(self, test_service):
        service, _ = test_service
        pdf_pages = [
            "Chapter 1: Foundations of Dense Retrieval.\nDense vector representations project textual chunks into a continuous mathematical vector space.",
            "Chapter 2: Vector Storage and Indexing.\nChromaDB implements Hierarchical Navigable Small World (HNSW) graphs for approximate nearest neighbor retrieval.",
        ]
        pdf_bytes = make_test_pdf(pdf_pages)

        result = service.ingest_document(pdf_bytes, filename="rag_foundations.pdf")

        assert result.status == "success"
        assert result.is_duplicate is False
        assert result.total_pages == 2
        assert result.total_chunks >= 2
        assert result.vector_count == result.total_chunks
        assert result.embedding_dimension == 384
        assert result.collection_name == "e2e_collection"

        # Verify chunks exist in ChromaDB
        chunks = service.get_document_chunks(result.document_id)
        assert len(chunks) == result.total_chunks
        for c in chunks:
            assert c["metadata"]["filename"] == "rag_foundations.pdf"
            assert isinstance(c["metadata"]["page_references"], list)
            assert c["metadata"]["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"

    def test_e2e_txt_ingestion(self, test_service):
        service, _ = test_service
        txt_content = (
            "System Architecture Overview.\n\n"
            "Retrieva AI implements a modular RAG pipeline without external framework abstractions.\n\n"
            "Each component maintains single responsibility: extraction, cleaning, chunking, embedding, and vector storage."
        ).encode("utf-8")

        result = service.ingest_document(txt_content, filename="architecture.txt")

        assert result.status == "success"
        assert result.total_chunks >= 1
        assert result.vector_count >= 1

    def test_e2e_duplicate_prevention(self, test_service):
        service, _ = test_service
        pdf_bytes = make_test_pdf(["Identical document content for duplicate checking."])

        # First ingestion
        res1 = service.ingest_document(pdf_bytes, filename="original.pdf")
        assert res1.status == "success"
        assert res1.is_duplicate is False

        # Second ingestion with same bytes
        res2 = service.ingest_document(pdf_bytes, filename="uploaded_again.pdf")
        assert res2.status == "duplicate"
        assert res2.is_duplicate is True
        assert res2.total_chunks == 0
        assert res2.vector_count == 0

    def test_e2e_delete_document(self, test_service):
        service, _ = test_service
        pdf_bytes = make_test_pdf(["Document content that will be deleted soon."])
        res = service.ingest_document(pdf_bytes, filename="to_delete.pdf")
        assert service.vector_store.count() > 0

        deleted_chunks = service.delete_document(res.document_id)
        assert deleted_chunks == res.total_chunks
        assert service.vector_store.count() == 0
        assert service.vector_store.has_document(res.document_id) is False
