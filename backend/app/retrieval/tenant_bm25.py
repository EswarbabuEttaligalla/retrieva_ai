"""Per-tenant BM25 index manager with LRU eviction and lazy ChromaDB hydration."""

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

try:
    from backend.app.core.config import get_settings
    from backend.app.core.logging import get_logger
    from backend.app.retrieval.bm25 import BM25Index
    from backend.app.schemas.document import DocumentChunk
    from backend.app.vector_store.store import (
        ChromaVectorStore,
        TenantIsolationError,
        normalize_owner_id,
    )
except ImportError:
    from app.core.config import get_settings
    from app.core.logging import get_logger
    from app.retrieval.bm25 import BM25Index
    from app.schemas.document import DocumentChunk
    from app.vector_store.store import (
        ChromaVectorStore,
        TenantIsolationError,
        normalize_owner_id,
    )

logger = get_logger("retrieval.tenant_bm25")


class BM25HydrationError(RuntimeError):
    """Raised when lazy hydration of an owner's BM25 index fails."""


class TenantBM25Manager:
    """Manages isolated per-owner BM25Index partitions with LRU cache bounds.

    Guarantees:
    - Each authenticated owner_id receives an independent BM25Index with isolated
      N, n(t), avgdl, IDF weights, and postings.
    - No global fallback index exists; hydration queries ChromaDB strictly by owner_id.
    - Per-tenant locks serialize concurrent mutations/hydration for the same owner
      without serializing retrieval across different owners.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        max_cached_tenants: Optional[int] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        settings = get_settings()
        resolved_max = max_cached_tenants if max_cached_tenants is not None else settings.BM25_MAX_CACHED_TENANTS
        if resolved_max < 1:
            raise ValueError(f"max_cached_tenants must be at least 1, got {resolved_max}")

        self.vector_store = vector_store or ChromaVectorStore()
        self.max_cached_tenants = resolved_max
        self.k1 = k1
        self.b = b

        self._cache: OrderedDict[str, BM25Index] = OrderedDict()
        self._cache_mutex = threading.Lock()
        self._tenant_locks: Dict[str, threading.Lock] = {}

    def _get_tenant_lock(self, owner_str: str) -> threading.Lock:
        with self._cache_mutex:
            lock = self._tenant_locks.get(owner_str)
            if lock is None:
                lock = threading.Lock()
                self._tenant_locks[owner_str] = lock
            return lock

    def _put_cached(self, owner_str: str, index: BM25Index) -> None:
        with self._cache_mutex:
            self._cache[owner_str] = index
            self._cache.move_to_end(owner_str)
            while len(self._cache) > self.max_cached_tenants:
                evicted_owner, _ = self._cache.popitem(last=False)
                logger.info("Evicted LRU BM25 index for owner '%s' (cache_limit=%d)", evicted_owner, self.max_cached_tenants)

    def _hydrate_from_vector_store(self, owner_str: str) -> BM25Index:
        """Load strictly owner_str's chunks from ChromaDB and construct a fresh BM25Index."""
        try:
            chunks = self.vector_store.get_owner_chunks(owner_str)
        except TenantIsolationError:
            raise
        except Exception as exc:
            logger.error("Failed hydrating BM25 index for owner '%s': %s", owner_str, exc)
            raise BM25HydrationError(f"Failed to hydrate BM25 index for owner '{owner_str}'.") from exc

        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            if meta.get("owner_id") != owner_str:
                logger.error(
                    "SECURITY ALERT: Hydration blocked cross-tenant chunk (expected_owner=%s, doc=%s)",
                    owner_str,
                    chunk.get("document_id"),
                )
                raise TenantIsolationError("Cross-tenant chunk rejected during BM25 hydration.")

        index = BM25Index(k1=self.k1, b=self.b)
        index.build_index(chunks)
        logger.debug("Hydrated BM25 index for owner '%s' (%d chunks)", owner_str, index.doc_count)
        return index

    def get_index(self, owner_id: Union[UUID, str]) -> Optional[BM25Index]:
        """Return cached BM25Index for owner_id without hydrating, or None if uncached."""
        owner_str = normalize_owner_id(owner_id)
        with self._cache_mutex:
            index = self._cache.get(owner_str)
            if index is not None:
                self._cache.move_to_end(owner_str)
            return index

    def get_or_create_index(self, owner_id: Union[UUID, str]) -> BM25Index:
        """Return cached BM25Index for owner_id or lazily hydrate it from ChromaDB."""
        owner_str = normalize_owner_id(owner_id)
        cached = self.get_index(owner_str)
        if cached is not None:
            return cached

        tenant_lock = self._get_tenant_lock(owner_str)
        with tenant_lock:
            cached = self.get_index(owner_str)
            if cached is not None:
                return cached
            index = self._hydrate_from_vector_store(owner_str)
            self._put_cached(owner_str, index)
            return index

    def refresh_owner(self, owner_id: Union[UUID, str]) -> BM25Index:
        """Rebuild and cache owner_id's BM25Index directly from ChromaDB."""
        owner_str = normalize_owner_id(owner_id)
        tenant_lock = self._get_tenant_lock(owner_str)
        with tenant_lock:
            index = self._hydrate_from_vector_store(owner_str)
            self._put_cached(owner_str, index)
            return index

    def add_chunks(
        self,
        owner_id: Union[UUID, str],
        chunks: Sequence[Union[DocumentChunk, Dict[str, Any]]],
    ) -> int:
        """Add chunks exclusively to owner_id's BM25Index."""
        owner_str = normalize_owner_id(owner_id)
        if not chunks:
            return 0

        normalized_chunks: List[Dict[str, Any]] = []
        for item in chunks:
            if isinstance(item, DocumentChunk):
                meta = {
                    "owner_id": owner_str,
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "filename": item.filename,
                    "page_number": item.page_number,
                    "page_references": list(item.page_references),
                    "chunk_index": item.chunk_index,
                    "start_offset": item.start_offset,
                    "end_offset": item.end_offset,
                    "char_count": item.char_count,
                    "word_count": item.word_count,
                }
                normalized_chunks.append({
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "filename": item.filename,
                    "text": item.chunk_text,
                    "chunk_text": item.chunk_text,
                    "metadata": meta,
                })
            else:
                meta = dict(item.get("metadata") or {})
                record_owner = meta.get("owner_id")
                if record_owner is not None and record_owner != owner_str:
                    raise TenantIsolationError("Cannot add cross-tenant chunk to owner BM25 index.")
                meta["owner_id"] = owner_str
                copy_item = dict(item)
                copy_item["metadata"] = meta
                normalized_chunks.append(copy_item)

        tenant_lock = self._get_tenant_lock(owner_str)
        with tenant_lock:
            with self._cache_mutex:
                index = self._cache.get(owner_str)
                if index is not None:
                    self._cache.move_to_end(owner_str)
            if index is None:
                index = self._hydrate_from_vector_store(owner_str)
                self._put_cached(owner_str, index)

            added = 0
            for chunk_dict in normalized_chunks:
                if chunk_dict["chunk_id"] not in index.corpus_chunks:
                    index.add_chunk(chunk_dict)
                    added += 1
            return added

    def remove_document(self, owner_id: Union[UUID, str], document_id: str) -> int:
        """Remove a document's chunks from owner_id's cached BM25Index if present."""
        owner_str = normalize_owner_id(owner_id)
        if not document_id:
            return 0

        tenant_lock = self._get_tenant_lock(owner_str)
        with tenant_lock:
            with self._cache_mutex:
                index = self._cache.get(owner_str)
                if index is not None:
                    self._cache.move_to_end(owner_str)
            if index is None:
                return 0
            return index.remove_document(document_id)

    def invalidate_owner(self, owner_id: Union[UUID, str]) -> bool:
        """Evict owner_id's cached BM25Index immediately."""
        owner_str = normalize_owner_id(owner_id)
        with self._cache_mutex:
            existed = self._cache.pop(owner_str, None) is not None
            return existed

    def get_owner_statistics(self, owner_id: Union[UUID, str]) -> Dict[str, Any]:
        """Return isolated BM25 corpus statistics exclusively for owner_id."""
        owner_str = normalize_owner_id(owner_id)
        index = self.get_or_create_index(owner_str)
        stats = index.get_statistics()
        stats["owner_id"] = owner_str
        return stats

    def cached_tenant_count(self) -> int:
        """Return number of tenant BM25 indexes currently held in LRU cache."""
        with self._cache_mutex:
            return len(self._cache)

    def cached_owner_ids(self) -> List[str]:
        """Return owner_ids currently cached (ordered oldest to most recently used)."""
        with self._cache_mutex:
            return list(self._cache.keys())
