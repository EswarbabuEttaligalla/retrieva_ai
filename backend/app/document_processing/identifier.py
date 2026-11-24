"""Document identification and duplicate detection using cryptographic hashing."""

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.app.core.logging import get_logger
from backend.app.document_processing.exceptions import RegistryPersistenceError
from backend.app.schemas.document import DocumentMetadata

logger = get_logger("document_processing.identifier")


def compute_content_hash(content: bytes) -> str:
    """Calculate the cryptographic SHA-256 hash of raw document bytes.
    
    Args:
        content: The raw byte sequence of the document.
        
    Returns:
        Hexadecimal string digest of the SHA-256 hash.
    """
    hasher = hashlib.sha256()
    hasher.update(content)
    return hasher.hexdigest()


def generate_document_id(content_hash: str) -> str:
    """Generate a readable, deterministic document ID from a SHA-256 hash.
    
    Args:
        content_hash: The SHA-256 hex string.
        
    Returns:
        A unique document identifier prefixed with 'doc_'.
    """
    return f"doc_{content_hash[:16]}"


class DocumentRegistry:
    """Thread-safe document registry with optional JSON persistence for tracking ingested documents and duplicates.
    
    Stores document metadata indexed by SHA-256 content hash. Prevents redundant ingestion
    and avoids accidental overwrites of existing documents.
    """

    def __init__(self, persist_path: Optional[Union[str, Path]] = None) -> None:
        self._storage: Dict[str, DocumentMetadata] = {}
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _save(self, raise_on_error: bool = False) -> None:
        """Atomically persist storage to disk if persist_path is configured.
        
        Args:
            raise_on_error: If True, re-raises RegistryPersistenceError on failure
                            instead of merely logging.
        """
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {h: meta.model_dump(mode="json") for h, meta in self._storage.items()}
            temp_file = self._persist_path.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_file.replace(self._persist_path)
        except Exception as exc:
            logger.error("Failed to persist document registry to %s: %s", self._persist_path, exc)
            if raise_on_error:
                raise RegistryPersistenceError(
                    f"Failed to persist document registry to {self._persist_path}", str(exc)
                )

    def _load(self) -> None:
        """Load persisted registry from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            content = self._persist_path.read_text(encoding="utf-8")
            if not content.strip():
                return
            data = json.loads(content)
            for h, meta_dict in data.items():
                self._storage[h] = DocumentMetadata.model_validate(meta_dict)
            logger.info("Loaded %d documents from registry at %s", len(self._storage), self._persist_path)
        except Exception as exc:
            logger.error("Failed to load document registry from %s: %s", self._persist_path, exc)

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if a document with the exact content hash has already been registered."""
        with self._lock:
            return content_hash in self._storage

    def register(self, metadata: DocumentMetadata, raise_on_error: bool = False) -> bool:
        """Register a document if it does not already exist.
        
        Args:
            metadata: DocumentMetadata to record.
            raise_on_error: If True, raises RegistryPersistenceError if disk write fails.
            
        Returns:
            True if registration succeeded, False if a duplicate already exists.
        """
        with self._lock:
            if metadata.content_hash in self._storage:
                return False
            self._storage[metadata.content_hash] = metadata
            try:
                self._save(raise_on_error=raise_on_error)
            except Exception:
                # Rollback in-memory mutation on disk persistence failure
                del self._storage[metadata.content_hash]
                raise
            return True

    def get_by_hash(self, content_hash: str) -> Optional[DocumentMetadata]:
        """Retrieve existing document metadata by content hash."""
        with self._lock:
            return self._storage.get(content_hash)

    def get_by_id(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve existing document metadata by unique document_id."""
        with self._lock:
            for meta in self._storage.values():
                if meta.document_id == document_id:
                    return meta
            return None

    def count(self) -> int:
        """Return the number of unique registered documents."""
        with self._lock:
            return len(self._storage)

    def remove(self, document_id: str, raise_on_error: bool = False) -> bool:
        """Remove a document from registry by its document_id.
        
        Args:
            document_id: Unique identifier of the document to remove.
            raise_on_error: If True, raises RegistryPersistenceError if disk write fails.
        """
        with self._lock:
            target_hash = None
            for h, meta in self._storage.items():
                if meta.document_id == document_id:
                    target_hash = h
                    break
            if target_hash:
                removed_meta = self._storage[target_hash]
                del self._storage[target_hash]
                try:
                    self._save(raise_on_error=raise_on_error)
                except Exception:
                    # Rollback in-memory deletion on disk persistence failure
                    self._storage[target_hash] = removed_meta
                    raise
                return True
            return False

    def get_all(self) -> List[DocumentMetadata]:
        """Return a copy of all registered documents sorted by creation date descending."""
        with self._lock:
            docs = list(self._storage.values())
        docs.sort(key=lambda d: d.created_at, reverse=True)
        return docs

    def reconcile_from_vector_store(self, vector_store: Any, prune_orphans: bool = True) -> int:
        """Reconstruct document metadata from ChromaDB chunk records if missing from registry,
        and optionally prune registry documents that no longer exist in ChromaDB."""
        if not vector_store:
            return 0
        if vector_store.count() == 0:
            reconciled_count = 0
            if prune_orphans:
                with self._lock:
                    if self._storage:
                        reconciled_count = len(self._storage)
                        self._storage.clear()
                        self._save()
                        logger.info("Pruned %d orphaned documents from registry (vector store is empty)", reconciled_count)
            return reconciled_count
        reconciled_count = 0
        try:
            records = vector_store.collection.get(include=["metadatas"])
            metas = records.get("metadatas") or []
            doc_chunks_count: Dict[str, int] = {}
            doc_info: Dict[str, Dict[str, Any]] = {}

            for m in metas:
                if not m or "document_id" not in m:
                    continue
                doc_id = m["document_id"]
                doc_chunks_count[doc_id] = doc_chunks_count.get(doc_id, 0) + 1
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        "filename": m.get("filename", "unknown"),
                        "content_hash": m.get("content_hash", ""),
                        "page_number": m.get("page_number", 1),
                    }
                else:
                    if m.get("page_number", 1) > doc_info[doc_id]["page_number"]:
                        doc_info[doc_id]["page_number"] = m["page_number"]

            with self._lock:
                existing_doc_ids = {m.document_id for m in self._storage.values()}
                for doc_id, info in doc_info.items():
                    if doc_id not in existing_doc_ids:
                        content_hash = info["content_hash"] or doc_id.replace("doc_", "")
                        ext = Path(info["filename"]).suffix.lower() or ".pdf"
                        meta = DocumentMetadata(
                            document_id=doc_id,
                            filename=info["filename"],
                            file_extension=ext,
                            file_size_bytes=0,
                            content_hash=content_hash,
                            total_pages=info["page_number"],
                            total_chunks=doc_chunks_count[doc_id],
                        )
                        self._storage[content_hash] = meta
                        reconciled_count += 1

                # Prune orphaned documents in registry that have 0 chunks in ChromaDB
                if prune_orphans and doc_info:
                    orphans = [h for h, m in self._storage.items() if m.document_id not in doc_info]
                    for orphan_hash in orphans:
                        del self._storage[orphan_hash]
                        reconciled_count += 1

                if reconciled_count > 0:
                    self._save()
                    logger.info("Reconciled %d documents from vector store records", reconciled_count)
        except Exception as exc:
            logger.warning("Reconciliation with vector store skipped or failed: %s", exc)
        return reconciled_count

    def clear(self) -> None:
        """Clear all registered documents (primarily used in test teardown)."""
        with self._lock:
            self._storage.clear()
            self._save()


# Default global registry singleton with persistence in chroma_data
global_document_registry = DocumentRegistry(
    persist_path=Path("chroma_data") / "documents_metadata.json"
)
