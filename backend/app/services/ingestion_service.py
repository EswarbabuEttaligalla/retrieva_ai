"""Integrated ingestion service for Retrieva AI.

Connects document extraction, text cleaning, boundary-aware chunking,
batch dense embedding generation, and persistent vector storage in ChromaDB.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.document_processing.cleaner import TextCleaner
from backend.app.document_processing.chunker import TextChunker
from backend.app.document_processing.extractor import DocumentExtractor
from backend.app.document_processing.identifier import (
    DocumentRegistry,
    compute_content_hash,
    generate_document_id,
    global_document_registry,
)
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.schemas.document import (
    DocumentChunk,
    DocumentMetadata,
    ExtractedPage,
    IngestionResult,
)
from backend.app.vector_store.store import ChromaVectorStore, DocumentAlreadyExistsError

logger = get_logger("services.ingestion")


class IngestionService:
    """Production service coordinating end-to-end ingestion and vector persistence."""

    def __init__(
        self,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        registry: Optional[DocumentRegistry] = None,
        chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = TextChunker.DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Initialize service components.

        Args:
            embedder: SentenceTransformerEmbedder instance (uses singleton if None).
            vector_store: ChromaVectorStore instance (initialized if None).
            registry: DocumentRegistry instance (uses global registry if None).
            chunk_size: Configurable chunk size in characters.
            chunk_overlap: Configurable chunk overlap in characters.
        """
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = embedder or SentenceTransformerEmbedder.get_instance()
        self.vector_store = vector_store or ChromaVectorStore()
        self.registry = registry or global_document_registry

    def ingest_document(
        self,
        file_input: Union[bytes, str, Path],
        filename: str = "",
        prevent_duplicates: bool = True,
    ) -> IngestionResult:
        """Execute complete document ingestion, embedding generation, and vector persistence.

        Flow:
        1. Extract bytes and infer filename.
        2. Compute SHA-256 digest and document ID.
        3. Check duplicate status against both in-memory registry and ChromaDB persistent store.
        4. Extract text per page using PyMuPDF / text decoder.
        5. Clean and normalize text, conservatively stripping multi-page headers/footers.
        6. Divide text into overlapping boundary-aware chunks with page provenance.
        7. Generate normalized 384-dimensional dense vectors in batches.
        8. Persist chunks, embeddings, and metadata into ChromaDB.
        9. Handle errors atomically with rollback.

        Args:
            file_input: Raw byte buffer or path to document.
            filename: Original document filename.
            prevent_duplicates: Reject duplicate content hashes if True.

        Returns:
            IngestionResult with execution status, metadata, chunks, and vector metrics.
        """
        # Resolve bytes and filename
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not filename:
                filename = path.name
            file_bytes = path.read_bytes()
        else:
            file_bytes = file_input

        if not filename:
            filename = "document.pdf"

        # Step 1: Content Hashing & Document ID
        content_hash = compute_content_hash(file_bytes)
        document_id = generate_document_id(content_hash)
        ext = Path(filename).suffix.lower()

        logger.info(
            "Starting document ingestion for '%s' (id=%s, hash=%s, bytes=%d)",
            filename,
            document_id,
            content_hash[:16],
            len(file_bytes),
        )

        # Step 2: Check Duplicate Status in Registry AND ChromaDB
        is_duplicate = False
        if prevent_duplicates:
            if self.vector_store.has_content_hash(content_hash) or self.registry.is_duplicate(content_hash):
                is_duplicate = True

        if is_duplicate:
            logger.info("Duplicate document detected for hash %s, skipping ingestion", content_hash[:16])
            existing_meta = self.registry.get_by_hash(content_hash)
            return IngestionResult(
                document_id=document_id,
                filename=filename,
                content_hash=content_hash,
                status="duplicate",
                total_pages=existing_meta.total_pages if existing_meta else 0,
                total_chunks=0,
                embedding_dimension=self.embedder.get_embedding_dimension(),
                vector_count=0,
                collection_name=self.vector_store.collection_name,
                is_duplicate=True,
                metadata=existing_meta,
                chunks=[],
                raw_pages=[],
                warnings=[f"Document with identical SHA-256 hash ({content_hash[:12]}...) is already stored."],
            )

        # Step 3: Extract Raw Text & Metadata
        raw_pages, extracted_meta = DocumentExtractor.extract(file_bytes, filename=filename)

        # Step 4: Clean Text across all pages
        cleaned_pages: List[ExtractedPage] = []
        raw_page_texts = [p.raw_text for p in raw_pages]
        cleaned_page_texts = [TextCleaner.clean(t) for t in raw_page_texts]

        if len(cleaned_page_texts) > 2:
            cleaned_page_texts = TextCleaner.strip_repetitive_headers_footers(cleaned_page_texts)

        warnings: List[str] = []
        for p, cleaned_txt in zip(raw_pages, cleaned_page_texts):
            p.cleaned_text = cleaned_txt
            p.char_count = len(cleaned_txt)
            p.word_count = len(cleaned_txt.split())
            p.has_text = bool(cleaned_txt.strip())

            if not p.has_text:
                warnings.append(f"Page {p.page_number} contains no extractable or cleaned text.")

            cleaned_pages.append(p)

        # Step 5: Chunking Page-by-Page with Exact Offset Tracking
        all_chunks: List[DocumentChunk] = []
        current_chunk_idx = 0

        for page in cleaned_pages:
            if not page.has_text:
                continue

            page_chunks = self.chunker.chunk_page(
                page_text=page.cleaned_text,
                document_id=document_id,
                filename=filename,
                page_number=page.page_number,
                global_start_index=current_chunk_idx,
            )
            all_chunks.extend(page_chunks)
            current_chunk_idx += len(page_chunks)

        if not all_chunks:
            warnings.append("Document yielded zero text chunks.")
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_extension=ext,
                file_size_bytes=len(file_bytes),
                content_hash=content_hash,
                total_pages=len(cleaned_pages),
                extracted_metadata=extracted_meta,
            )
            return IngestionResult(
                document_id=document_id,
                filename=filename,
                content_hash=content_hash,
                status="success",
                total_pages=len(cleaned_pages),
                total_chunks=0,
                embedding_dimension=self.embedder.get_embedding_dimension(),
                vector_count=0,
                collection_name=self.vector_store.collection_name,
                is_duplicate=False,
                metadata=metadata,
                chunks=[],
                raw_pages=cleaned_pages,
                warnings=warnings,
            )

        # Step 5b: Enforce Model Token Limits to Prevent Silent Information Truncation
        max_allowed_tokens = self.embedder.get_max_sequence_length() - 4
        all_chunks = self.chunker.enforce_token_limit(
            all_chunks,
            max_tokens=max_allowed_tokens,
            token_overlap=32,
            tokenizer=self.embedder.get_tokenizer(),
        )

        # Step 6: Dense Embedding Generation (Sentence Transformers)
        chunk_texts = [c.chunk_text for c in all_chunks]
        logger.info(
            "Generating dense embeddings for %d chunks (dim=%d)...",
            len(chunk_texts),
            self.embedder.get_embedding_dimension(),
        )

        embeddings = self.embedder.embed_batch(chunk_texts)

        # Step 7: Store Chunks, Explicit Embeddings, and Metadata in ChromaDB
        stored_count = self.vector_store.add_chunks(
            chunks=all_chunks,
            embeddings=embeddings,
            content_hash=content_hash,
            embedding_model=self.embedder.model_name,
            prevent_duplicates=prevent_duplicates,
        )

        # Step 8: Construct and Register Document Metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_extension=ext,
            file_size_bytes=len(file_bytes),
            content_hash=content_hash,
            total_pages=len(cleaned_pages),
            total_chunks=len(all_chunks),
            extracted_metadata=extracted_meta,
        )
        self.registry.register(metadata)

        logger.info(
            "Ingestion completed for '%s': %d chunks stored in collection '%s'",
            document_id,
            stored_count,
            self.vector_store.collection_name,
        )

        return IngestionResult(
            document_id=document_id,
            filename=filename,
            content_hash=content_hash,
            status="success",
            total_pages=len(cleaned_pages),
            total_chunks=len(all_chunks),
            embedding_dimension=self.embedder.get_embedding_dimension(),
            vector_count=stored_count,
            collection_name=self.vector_store.collection_name,
            is_duplicate=False,
            metadata=metadata,
            chunks=all_chunks,
            raw_pages=cleaned_pages,
            warnings=warnings,
        )

    def delete_document(self, document_id: str) -> int:
        """Remove a document from both vector store and in-memory registry.

        Args:
            document_id: Unique document identifier.

        Returns:
            Number of chunks removed from vector store.
        """
        deleted_count = self.vector_store.delete_document(document_id)
        self.registry.remove(document_id)
        return deleted_count

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve stored chunks and deserialized metadata for a document."""
        return self.vector_store.get_document_chunks(document_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Return operational statistics for ingestion and vector storage."""
        stats = self.vector_store.get_statistics()
        stats["registry_documents_count"] = self.registry.count()
        stats["embedding_model"] = self.embedder.model_name
        stats["max_sequence_length"] = self.embedder.get_max_sequence_length()
        return stats


_ingestion_service_instance: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    """Get or create the singleton IngestionService instance."""
    global _ingestion_service_instance
    if _ingestion_service_instance is None:
        _ingestion_service_instance = IngestionService()
    return _ingestion_service_instance
