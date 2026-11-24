"""ChromaDB vector store implementation for Retrieva AI.

Provides persistent storage, metadata serialization/deserialization,
duplicate prevention, atomic chunk ingestion, and document lifecycle management.
"""

from typing import Any, Dict, List, Optional, Set
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.document import DocumentChunk
from backend.app.embeddings.validator import validate_batch_embeddings, EmbeddingValidationError
from backend.app.vector_store.client import get_chroma_client

logger = get_logger("vector_store.store")


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class DocumentAlreadyExistsError(VectorStoreError):
    """Raised when attempting to add a document that is already ingested."""
    pass


class VectorStoreInsertionError(VectorStoreError):
    """Raised when chunk insertion into vector store fails."""
    pass


class ChromaVectorStore:
    """Manages persistent document vector storage in ChromaDB.

    Enforces explicit embeddings generation (Sentence Transformers) and bypasses
    ChromaDB's internal embedding functions.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        client: Optional[ClientAPI] = None,
    ) -> None:
        """Initialize the vector store with persistent storage and cosine distance.

        Args:
            collection_name: Name of the ChromaDB collection. Defaults to config setting.
            persist_directory: Directory for persistent Chroma storage.
            client: Optional pre-configured ChromaDB ClientAPI instance.
        """
        settings = get_settings()
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.expected_dim = settings.EMBEDDING_DIMENSION

        # Initialize persistent client
        self.client: ClientAPI = client or get_chroma_client(self.persist_directory)

        # Get or create collection with explicit cosine similarity configuration
        # Note: We do NOT pass an embedding_function to ensure raw embeddings are supplied
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # STRICT: Manual embeddings only!
        )

        logger.info(
            "ChromaVectorStore initialized for '%s' (persist_dir=%s, count=%d)",
            self.collection_name,
            self.persist_directory,
            self.collection.count(),
        )

    def count(self) -> int:
        """Return total number of chunks currently stored in the collection."""
        return self.collection.count()

    def has_document(self, document_id: str) -> bool:
        """Check if any chunk belonging to the given document_id exists.

        Args:
            document_id: Unique SHA-256 or string identifier of the document.

        Returns:
            True if the document is present, False otherwise.
        """
        if not document_id:
            return False

        results = self.collection.get(
            where={"document_id": document_id},
            limit=1,
            include=[],
        )
        return bool(results and results.get("ids") and len(results["ids"]) > 0)

    def has_content_hash(self, content_hash: str) -> bool:
        """Check if a document with the exact content hash already exists.

        Args:
            content_hash: SHA-256 hash of the cleaned or raw document content.

        Returns:
            True if content hash exists, False otherwise.
        """
        if not content_hash:
            return False

        results = self.collection.get(
            where={"content_hash": content_hash},
            limit=1,
            include=[],
        )
        return bool(results and results.get("ids") and len(results["ids"]) > 0)

    def serialize_metadata(
        self,
        chunk: DocumentChunk,
        content_hash: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert chunk metadata into ChromaDB-compatible scalar types.

        ChromaDB metadata strictly accepts: str, int, float, bool.
        Lists such as page_references must be serialized to comma-delimited strings.

        Args:
            chunk: DocumentChunk instance.
            content_hash: Content hash for duplicate tracking.
            embedding_model: Identifier of the embedding model used.

        Returns:
            Dict containing scalar metadata fields.
        """
        settings = get_settings()
        page_refs_str = ",".join(str(p) for p in chunk.page_references)

        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "page_references": page_refs_str,
            "chunk_index": chunk.chunk_index,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "char_count": chunk.char_count,
            "word_count": chunk.word_count,
            "content_hash": content_hash or "",
            "embedding_model": embedding_model or settings.EMBEDDING_MODEL_NAME,
        }

    def deserialize_metadata(self, raw_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize scalar metadata fields back to structured Python types.

        Reconstructs page_references list from comma-delimited string.

        Args:
            raw_meta: Raw metadata dictionary from ChromaDB.

        Returns:
            Dictionary with parsed page_references as List[int].
        """
        meta = dict(raw_meta)
        page_refs_str = meta.get("page_references", "")
        if isinstance(page_refs_str, str) and page_refs_str.strip():
            try:
                meta["page_references"] = [
                    int(p.strip()) for p in page_refs_str.split(",") if p.strip()
                ]
            except ValueError:
                meta["page_references"] = [meta.get("page_number", 1)]
        elif isinstance(page_refs_str, list):
            meta["page_references"] = [int(p) for p in page_refs_str]
        else:
            meta["page_references"] = [meta.get("page_number", 1)]

        return meta

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        content_hash: Optional[str] = None,
        embedding_model: Optional[str] = None,
        prevent_duplicates: bool = True,
    ) -> int:
        """Add document chunks and their precomputed embeddings into ChromaDB.

        Performs:
        1. Length and dimensional validation.
        2. Duplicate detection check.
        3. Metadata serialization.
        4. Atomic-like insertion with automatic rollback on error.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: Corresponding list of 384-dimensional float vectors.
            content_hash: SHA-256 hash of the source document content.
            embedding_model: Name of the embedding model used.
            prevent_duplicates: If True, raises DocumentAlreadyExistsError if doc exists.

        Returns:
            Total number of chunks successfully stored.

        Raises:
            ValueError: If chunks and embeddings lengths do not match or are empty.
            DocumentAlreadyExistsError: If document already exists and prevent_duplicates is True.
            EmbeddingValidationError: If any embedding vector fails validation.
            VectorStoreInsertionError: If ChromaDB insertion fails.
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty.")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )

        document_id = chunks[0].document_id

        # Duplicate checking
        if prevent_duplicates:
            if self.has_document(document_id):
                raise DocumentAlreadyExistsError(
                    f"Document with ID '{document_id}' already exists in vector store."
                )
            if content_hash and self.has_content_hash(content_hash):
                raise DocumentAlreadyExistsError(
                    f"Document with identical content hash '{content_hash}' already exists in vector store."
                )

        # Validate all embeddings before attempting insertion
        validate_batch_embeddings(embeddings, expected_dim=self.expected_dim)

        # Prepare payloads for ChromaDB
        ids: List[str] = [chunk.chunk_id for chunk in chunks]
        documents: List[str] = [chunk.text for chunk in chunks]
        metadatas: List[Dict[str, Any]] = [
            self.serialize_metadata(chunk, content_hash=content_hash, embedding_model=embedding_model)
            for chunk in chunks
        ]

        logger.info(
            "Inserting %d chunks into ChromaDB for document '%s'...",
            len(chunks),
            document_id,
        )

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                "Persisted %d chunks to ChromaDB for document '%s' (collection total=%d)",
                len(chunks),
                document_id,
                self.collection.count(),
            )
            return len(chunks)

        except Exception as exc:
            logger.error("ChromaDB chunk insertion failed for '%s': %s", document_id, exc)
            # Rollback: Clean up any partially inserted chunks for this document
            try:
                self.delete_document(document_id)
            except Exception as rollback_err:
                logger.critical("Rollback cleanup failed for '%s': %s", document_id, rollback_err)
            raise VectorStoreInsertionError(
                f"Failed to insert chunks into ChromaDB for document '{document_id}': {str(exc)}"
            ) from exc

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all stored chunks for a given document_id ordered by chunk_index.

        Args:
            document_id: Unique document identifier.

        Returns:
            List of dictionaries containing chunk details and deserialized metadata.
        """
        if not document_id:
            return []

        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )

        if not results or not results.get("ids"):
            return []

        chunks: List[Dict[str, Any]] = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for idx, chunk_id in enumerate(ids):
            doc_text = docs[idx] if idx < len(docs) else ""
            raw_meta = metas[idx] if idx < len(metas) else {}
            deserialized_meta = self.deserialize_metadata(raw_meta)

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": doc_text,
                "metadata": deserialized_meta,
                "chunk_index": deserialized_meta.get("chunk_index", 0),
            })

        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document from ChromaDB.

        Args:
            document_id: Identifier of the document to remove.

        Returns:
            Number of chunks removed.
        """
        if not document_id:
            return 0

        results = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        existing_ids = results.get("ids", [])
        if not existing_ids:
            return 0

        self.collection.delete(where={"document_id": document_id})
        logger.info(
            "Deleted document '%s' from ChromaDB (%d chunks removed, %d remaining)",
            document_id,
            len(existing_ids),
            self.collection.count(),
        )
        return len(existing_ids)

    def get_unique_document_ids(self) -> Set[str]:
        """Return the set of unique document_ids currently stored in the collection."""
        all_records = self.collection.get(include=["metadatas"])
        metadatas = all_records.get("metadatas", []) or []
        doc_ids = set()
        for m in metadatas:
            if m and "document_id" in m:
                doc_ids.add(m["document_id"])
        return doc_ids

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive operational statistics for the vector store.

        Returns:
            Dictionary with collection metadata, counts, and directory paths.
        """
        unique_docs = self.get_unique_document_ids()
        total_chunks = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "persist_directory": str(Path(self.persist_directory).resolve()),
            "total_chunks": total_chunks,
            "unique_documents_count": len(unique_docs),
            "unique_document_ids": sorted(list(unique_docs)),
            "distance_metric": "cosine",
            "embedding_dimension": self.expected_dim,
        }

    def reset(self) -> None:
        """Delete all records and reset the collection. Useful for tests."""
        logger.warning("Resetting ChromaDB collection '%s'...", self.collection_name)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
