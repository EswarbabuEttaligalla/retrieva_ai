"""ChromaDB vector store implementation for Retrieva AI.

Provides persistent storage, metadata serialization/deserialization,
per-owner duplicate prevention, atomic chunk ingestion, mandatory owner_id
filtering on all lookups/queries/deletions, and defensive post-retrieval checks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.embeddings.validator import EmbeddingValidationError, validate_batch_embeddings
from backend.app.schemas.document import DocumentChunk
from backend.app.vector_store.client import get_chroma_client

logger = get_logger("vector_store.store")


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class DocumentAlreadyExistsError(VectorStoreError):
    """Raised when attempting to add a document that is already ingested for this owner."""
    pass


class VectorStoreInsertionError(VectorStoreError):
    """Raised when chunk insertion into vector store fails."""
    pass


class TenantIsolationError(VectorStoreError):
    """Raised when a vector store operation detects missing or mismatched owner_id."""
    pass


def normalize_owner_id(owner_id: Union[UUID, str]) -> str:
    """Validate and normalize owner_id as a canonical lowercase UUID string."""
    if isinstance(owner_id, UUID):
        return str(owner_id)
    if isinstance(owner_id, str) and owner_id.strip():
        try:
            return str(UUID(owner_id.strip()))
        except ValueError as exc:
            raise ValueError("owner_id must be a valid UUID.") from exc
    raise ValueError("A valid UUID owner_id is required.")


class ChromaVectorStore:
    """Manages persistent document vector storage in ChromaDB with mandatory tenant isolation.

    Enforces explicit embeddings generation (Sentence Transformers) and bypasses
    ChromaDB's internal embedding functions. Every chunk carries `owner_id` metadata
    and every read, query, and delete operation requires an explicit `owner_id`.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        client: Optional[ClientAPI] = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.expected_dim = settings.EMBEDDING_DIMENSION

        self.client: ClientAPI = client or get_chroma_client(self.persist_directory)
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

    @staticmethod
    def _build_owner_where(
        owner_str: str,
        extra_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a ChromaDB `where` clause always scoped to `owner_id`."""
        owner_clause = {"owner_id": owner_str}
        if not extra_filter:
            return owner_clause
        return {"$and": [owner_clause, extra_filter]}

    @staticmethod
    def _assert_records_owned(
        metadatas: List[Optional[Dict[str, Any]]],
        owner_str: str,
        operation: str,
    ) -> None:
        """Defensive post-retrieval check: reject any chunk lacking matching owner_id."""
        for raw_meta in metadatas:
            meta = raw_meta or {}
            record_owner = meta.get("owner_id")
            if record_owner != owner_str:
                logger.error(
                    "SECURITY ALERT: Cross-tenant vector blocked during %s (expected_owner=%s, actual_owner=%s, document_id=%s)",
                    operation,
                    owner_str,
                    record_owner,
                    meta.get("document_id"),
                )
                raise TenantIsolationError(
                    "Cross-tenant vector access prevented by ownership verification."
                )

    def count(self) -> int:
        """Return total number of chunks across the collection (system metric only)."""
        return self.collection.count()

    def count_for_owner(self, owner_id: Union[UUID, str]) -> int:
        """Return total number of chunks stored for a specific owner_id."""
        owner_str = normalize_owner_id(owner_id)
        results = self.collection.get(
            where={"owner_id": owner_str},
            include=[],
        )
        return len(results.get("ids") or [])

    def has_document(self, document_id: str, owner_id: Union[UUID, str]) -> bool:
        """Check if any chunk belonging to (owner_id, document_id) exists."""
        owner_str = normalize_owner_id(owner_id)
        if not document_id:
            return False

        results = self.collection.get(
            where=self._build_owner_where(owner_str, {"document_id": document_id}),
            limit=1,
            include=[],
        )
        return bool(results and results.get("ids") and len(results["ids"]) > 0)

    def has_content_hash(self, content_hash: str, owner_id: Union[UUID, str]) -> bool:
        """Check if a document with content_hash exists for this specific owner_id."""
        owner_str = normalize_owner_id(owner_id)
        if not content_hash:
            return False

        results = self.collection.get(
            where=self._build_owner_where(owner_str, {"content_hash": content_hash}),
            limit=1,
            include=[],
        )
        return bool(results and results.get("ids") and len(results["ids"]) > 0)

    def serialize_metadata(
        self,
        chunk: DocumentChunk,
        owner_id: Union[UUID, str],
        content_hash: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert chunk metadata into ChromaDB-compatible scalar types including owner_id."""
        owner_str = normalize_owner_id(owner_id)
        settings = get_settings()
        page_refs_str = ",".join(str(p) for p in chunk.page_references)

        return {
            "owner_id": owner_str,
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
        """Deserialize scalar metadata fields back to structured Python types."""
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
        owner_id: Union[UUID, str],
        content_hash: Optional[str] = None,
        embedding_model: Optional[str] = None,
        prevent_duplicates: bool = True,
    ) -> int:
        """Add document chunks and precomputed embeddings into ChromaDB under owner_id."""
        owner_str = normalize_owner_id(owner_id)

        if not chunks:
            raise ValueError("Chunks list cannot be empty.")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )

        document_id = chunks[0].document_id

        if prevent_duplicates:
            if self.has_document(document_id, owner_id=owner_str):
                raise DocumentAlreadyExistsError(
                    f"Document with ID '{document_id}' already exists for this owner."
                )
            if content_hash and self.has_content_hash(content_hash, owner_id=owner_str):
                raise DocumentAlreadyExistsError(
                    f"Document with identical content hash '{content_hash}' already exists for this owner."
                )

        validate_batch_embeddings(embeddings, expected_dim=self.expected_dim)

        ids: List[str] = [chunk.chunk_id for chunk in chunks]
        documents: List[str] = [chunk.text for chunk in chunks]
        metadatas: List[Dict[str, Any]] = [
            self.serialize_metadata(
                chunk,
                owner_id=owner_str,
                content_hash=content_hash,
                embedding_model=embedding_model,
            )
            for chunk in chunks
        ]

        logger.info(
            "Inserting %d chunks into ChromaDB for document '%s' (owner=%s)...",
            len(chunks),
            document_id,
            owner_str,
        )

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                "Persisted %d chunks to ChromaDB for document '%s' (owner=%s)",
                len(chunks),
                document_id,
                owner_str,
            )
            return len(chunks)

        except Exception as exc:
            logger.error("ChromaDB chunk insertion failed for '%s': %s", document_id, exc)
            try:
                self.delete_document(document_id, owner_id=owner_str)
            except Exception as rollback_err:
                logger.critical("Rollback cleanup failed for '%s': %s", document_id, rollback_err)
            raise VectorStoreInsertionError(
                f"Failed to insert chunks into ChromaDB for document '{document_id}': {str(exc)}"
            ) from exc

    def get_document_chunks(
        self,
        document_id: str,
        owner_id: Union[UUID, str],
    ) -> List[Dict[str, Any]]:
        """Retrieve all stored chunks for (owner_id, document_id) ordered by chunk_index."""
        owner_str = normalize_owner_id(owner_id)
        if not document_id:
            return []

        results = self.collection.get(
            where=self._build_owner_where(owner_str, {"document_id": document_id}),
            include=["documents", "metadatas"],
        )

        ids = results.get("ids") or []
        if not ids:
            return []

        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        self._assert_records_owned(metas, owner_str, "get_document_chunks")

        chunks: List[Dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            doc_text = docs[idx] if idx < len(docs) else ""
            raw_meta = metas[idx] if idx < len(metas) else {}
            deserialized_meta = self.deserialize_metadata(raw_meta)

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "owner_id": owner_str,
                "text": doc_text,
                "metadata": deserialized_meta,
                "chunk_index": deserialized_meta.get("chunk_index", 0),
            })

        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks

    def get_owner_chunks(self, owner_id: Union[UUID, str]) -> List[Dict[str, Any]]:
        """Retrieve all chunks owned by owner_id."""
        owner_str = normalize_owner_id(owner_id)
        results = self.collection.get(
            where={"owner_id": owner_str},
            include=["documents", "metadatas"],
        )
        ids = results.get("ids") or []
        if not ids:
            return []

        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        self._assert_records_owned(metas, owner_str, "get_owner_chunks")

        chunks: List[Dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            doc_text = docs[idx] if idx < len(docs) else ""
            raw_meta = metas[idx] if idx < len(metas) else {}
            deserialized_meta = self.deserialize_metadata(raw_meta)
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": deserialized_meta.get("document_id", ""),
                "filename": deserialized_meta.get("filename", ""),
                "text": doc_text,
                "chunk_text": doc_text,
                "metadata": deserialized_meta,
            })
        return chunks

    def query_similar(
        self,
        query_embedding: List[float],
        owner_id: Union[UUID, str],
        top_k: int = 5,
        document_id_filter: Optional[str] = None,
        document_ids_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a mandatory owner-scoped ChromaDB similarity query."""
        owner_str = normalize_owner_id(owner_id)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        extra: Optional[Dict[str, Any]] = None
        allowed_ids: Optional[List[str]] = None
        if document_ids_filter is not None:
            cleaned_ids = [str(d).strip() for d in document_ids_filter if str(d).strip()]
            if cleaned_ids:
                allowed_ids = list(dict.fromkeys(cleaned_ids))
        if document_id_filter:
            clean_single = str(document_id_filter).strip()
            if clean_single:
                if allowed_ids is not None:
                    allowed_ids = [d for d in allowed_ids if d == clean_single]
                    if not allowed_ids:
                        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
                else:
                    allowed_ids = [clean_single]

        if allowed_ids is not None:
            if len(allowed_ids) == 1:
                extra = {"document_id": allowed_ids[0]}
            else:
                extra = {"document_id": {"$in": allowed_ids}}

        where_clause = self._build_owner_where(owner_str, extra)

        matching = self.collection.get(where=where_clause, include=[])
        matching_ids = matching.get("ids") or []
        if not matching_ids:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        effective_n_results = min(top_k, len(matching_ids))
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        metas = (results.get("metadatas") or [[]])[0]
        self._assert_records_owned(metas, owner_str, "query_similar")
        return results

    def delete_document(self, document_id: str, owner_id: Union[UUID, str]) -> int:
        """Delete all chunks belonging to (owner_id, document_id) from ChromaDB."""
        owner_str = normalize_owner_id(owner_id)
        if not document_id:
            return 0

        where_clause = self._build_owner_where(owner_str, {"document_id": document_id})
        results = self.collection.get(
            where=where_clause,
            include=["metadatas"],
        )
        existing_ids = results.get("ids") or []
        if not existing_ids:
            return 0

        self._assert_records_owned(results.get("metadatas") or [], owner_str, "delete_document")
        self.collection.delete(where=where_clause)
        logger.info(
            "Deleted document '%s' for owner '%s' from ChromaDB (%d chunks removed)",
            document_id,
            owner_str,
            len(existing_ids),
        )
        return len(existing_ids)

    def get_unique_document_ids(self, owner_id: Union[UUID, str]) -> Set[str]:
        """Return unique document_ids stored for the specified owner_id."""
        owner_str = normalize_owner_id(owner_id)
        records = self.collection.get(
            where={"owner_id": owner_str},
            include=["metadatas"],
        )
        metadatas = records.get("metadatas") or []
        self._assert_records_owned(metadatas, owner_str, "get_unique_document_ids")

        doc_ids: Set[str] = set()
        for m in metadatas:
            if m and "document_id" in m:
                doc_ids.add(m["document_id"])
        return doc_ids

    def get_statistics(self, owner_id: Optional[Union[UUID, str]] = None) -> Dict[str, Any]:
        """Return vector store statistics. Document IDs are returned only when owner_id is supplied."""
        if owner_id is not None:
            owner_str = normalize_owner_id(owner_id)
            unique_docs = self.get_unique_document_ids(owner_str)
            total_chunks = self.count_for_owner(owner_str)
            return {
                "collection_name": self.collection_name,
                "persist_directory": str(Path(self.persist_directory).resolve()),
                "owner_id": owner_str,
                "total_chunks": total_chunks,
                "unique_documents_count": len(unique_docs),
                "unique_document_ids": sorted(list(unique_docs)),
                "distance_metric": "cosine",
                "embedding_dimension": self.expected_dim,
            }

        return {
            "collection_name": self.collection_name,
            "persist_directory": str(Path(self.persist_directory).resolve()),
            "total_chunks": self.collection.count(),
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
