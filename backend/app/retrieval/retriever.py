"""Retrieval orchestrator service for Retrieva AI.

Coordinates owner-isolated Dense Semantic Retrieval, per-tenant BM25 Keyword Search,
Reciprocal Rank Fusion, and Cross-Encoder Reranking with defense-in-depth ownership checks.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Set, Union
from uuid import UUID

try:
    from backend.app.core.logging import get_logger
    from backend.app.retrieval.dense import DenseRetriever
    from backend.app.retrieval.fusion import reciprocal_rank_fusion
    from backend.app.retrieval.reranker import CrossEncoderReranker
    from backend.app.retrieval.tenant_bm25 import TenantBM25Manager
    from backend.app.schemas.document import DocumentChunk
    from backend.app.schemas.retrieval import (
        RetrievalConfiguration,
        RetrievalMode,
        RetrievalRequest,
        RetrievalResult,
        RetrievedChunk,
    )
    from backend.app.vector_store.store import (
        ChromaVectorStore,
        TenantIsolationError,
        normalize_owner_id,
    )
except ImportError:
    from app.core.logging import get_logger
    from app.retrieval.dense import DenseRetriever
    from app.retrieval.fusion import reciprocal_rank_fusion
    from app.retrieval.reranker import CrossEncoderReranker
    from app.retrieval.tenant_bm25 import TenantBM25Manager
    from app.schemas.document import DocumentChunk
    from app.schemas.retrieval import (
        RetrievalConfiguration,
        RetrievalMode,
        RetrievalRequest,
        RetrievalResult,
        RetrievedChunk,
    )
    from app.vector_store.store import (
        ChromaVectorStore,
        TenantIsolationError,
        normalize_owner_id,
    )

logger = get_logger("retrieval.retriever")


class Retriever:
    """Tenant-isolated retrieval orchestrator supporting Dense, BM25, and Hybrid strategies."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        tenant_bm25_manager: Optional[TenantBM25Manager] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.dense_retriever = dense_retriever or DenseRetriever(vector_store=self.vector_store)
        self.tenant_bm25_manager = tenant_bm25_manager or TenantBM25Manager(vector_store=self.vector_store)
        self._reranker = reranker

    def get_reranker(self, model_name: str) -> CrossEncoderReranker:
        """Lazy load or return CrossEncoderReranker singleton."""
        if self._reranker is None or self._reranker.model_name != model_name:
            self._reranker = CrossEncoderReranker.get_instance(model_name=model_name)
        return self._reranker

    def sync_owner_chunks(
        self,
        owner_id: Union[UUID, str],
        chunks: Sequence[Union[DocumentChunk, Dict[str, Any]]],
    ) -> int:
        """Add newly ingested chunks exclusively to owner_id's BM25 partition."""
        return self.tenant_bm25_manager.add_chunks(owner_id, chunks)

    def remove_owner_document(self, owner_id: Union[UUID, str], document_id: str) -> int:
        """Remove a deleted document exclusively from owner_id's cached BM25 partition."""
        return self.tenant_bm25_manager.remove_document(owner_id, document_id)

    def refresh_owner_bm25(self, owner_id: Union[UUID, str]):
        """Rebuild owner_id's BM25 index strictly from their ChromaDB chunks."""
        return self.tenant_bm25_manager.refresh_owner(owner_id)

    def get_owner_statistics(self, owner_id: Union[UUID, str]) -> Dict[str, Any]:
        """Return isolated ChromaDB + BM25 statistics exclusively for owner_id."""
        owner_str = normalize_owner_id(owner_id)
        vector_stats = self.vector_store.get_statistics(owner_id=owner_str)
        bm25_stats = self.tenant_bm25_manager.get_owner_statistics(owner_str)
        return {
            "owner_id": owner_str,
            "vector_store": vector_stats,
            "bm25": bm25_stats,
        }

    @staticmethod
    def _resolve_allowed_document_ids(
        cfg: RetrievalConfiguration,
        authorized_ready_doc_ids: Optional[Set[str]] = None,
    ) -> Optional[Set[str]]:
        """Resolve combined filters intersected with authorized ready document IDs."""
        allowed: Optional[Set[str]] = None
        if cfg.document_ids_filter is not None:
            cleaned = {str(d).strip() for d in cfg.document_ids_filter if str(d).strip()}
            if cleaned:
                allowed = cleaned
        if cfg.document_id_filter:
            single = str(cfg.document_id_filter).strip()
            if single:
                if allowed is not None:
                    allowed = allowed & {single}
                else:
                    allowed = {single}
        if authorized_ready_doc_ids is not None:
            ready_set = {str(d).strip() for d in authorized_ready_doc_ids if str(d).strip()}
            allowed = (allowed & ready_set) if allowed is not None else ready_set
        return allowed

    @staticmethod
    def _assert_chunks_owned_and_filtered(
        chunks: List[RetrievedChunk],
        owner_str: str,
        allowed_doc_ids: Optional[Set[str]],
        stage: str,
    ) -> None:
        """Defense-in-depth assertion verifying every candidate belongs to owner_id and allowed docs."""
        for chunk in chunks:
            chunk_owner = (chunk.metadata or {}).get("owner_id")
            if chunk_owner != owner_str:
                logger.error(
                    "SECURITY ALERT: Cross-tenant chunk rejected at stage '%s' (expected_owner=%s, actual_owner=%s, doc_id=%s)",
                    stage,
                    owner_str,
                    chunk_owner,
                    chunk.document_id,
                )
                raise TenantIsolationError(
                    f"Cross-tenant chunk detected during '{stage}' stage."
                )
            if allowed_doc_ids is not None and chunk.document_id not in allowed_doc_ids:
                logger.error(
                    "SECURITY ALERT: Unrequested document chunk rejected at stage '%s' (doc_id=%s)",
                    stage,
                    chunk.document_id,
                )
                raise TenantIsolationError(
                    f"Document filter violation detected during '{stage}' stage."
                )

    def retrieve(
        self,
        request: RetrievalRequest,
        owner_id: Union[UUID, str],
        authorized_ready_doc_ids: Optional[Set[str]] = None,
    ) -> RetrievalResult:
        """Execute the configured retrieval pipeline strictly scoped to owner_id and ready docs."""
        start_time = time.perf_counter()
        query = request.query
        cfg = request.config
        owner_str = normalize_owner_id(owner_id)
        allowed_doc_ids = self._resolve_allowed_document_ids(cfg, authorized_ready_doc_ids)

        logger.info(
            "Executing tenant-isolated retrieval (owner=%s, mode=%s, top_k=%d, rerank=%s)...",
            owner_str,
            cfg.mode.value,
            cfg.top_k,
            cfg.enable_reranking,
        )

        # Short-circuit conflicting filters (e.g. non-overlapping single + multi filter)
        if allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return RetrievalResult(
                query=query,
                mode=cfg.mode,
                total_candidates_found=0,
                returned_chunks_count=0,
                chunks=[],
                execution_time_ms=round(duration_ms, 2),
                configuration_applied=cfg.model_dump(),
                reranking_summary=None,
            )

        doc_ids_list = sorted(allowed_doc_ids) if allowed_doc_ids is not None else None
        rerank_summary = None
        candidates: List[RetrievedChunk] = []

        # -------------------------------------------------------------
        # Mode 1: DENSE Retrieval (Mandatory owner_id in ChromaDB)
        # -------------------------------------------------------------
        if cfg.mode == RetrievalMode.DENSE:
            fetch_k = cfg.candidate_k if cfg.enable_reranking else cfg.top_k
            candidates = self.dense_retriever.search(
                query=query,
                owner_id=owner_str,
                top_k=fetch_k,
                similarity_threshold=cfg.similarity_threshold,
                document_id_filter=cfg.document_id_filter,
                document_ids_filter=doc_ids_list,
            )
            self._assert_chunks_owned_and_filtered(candidates, owner_str, allowed_doc_ids, "dense")

        # -------------------------------------------------------------
        # Mode 2: BM25 Retrieval (Per-owner BM25Index via TenantBM25Manager)
        # -------------------------------------------------------------
        elif cfg.mode == RetrievalMode.BM25:
            fetch_k = cfg.candidate_k if cfg.enable_reranking else cfg.top_k
            owner_bm25 = self.tenant_bm25_manager.get_or_create_index(owner_str)
            candidates = owner_bm25.search(
                query=query,
                top_k=fetch_k,
                document_id_filter=cfg.document_id_filter,
                document_ids_filter=doc_ids_list,
            )
            self._assert_chunks_owned_and_filtered(candidates, owner_str, allowed_doc_ids, "bm25")

        # -------------------------------------------------------------
        # Mode 3: HYBRID Retrieval (Owner Dense + Owner BM25 + RRF)
        # -------------------------------------------------------------
        elif cfg.mode == RetrievalMode.HYBRID:
            dense_candidates = self.dense_retriever.search(
                query=query,
                owner_id=owner_str,
                top_k=cfg.candidate_k,
                similarity_threshold=cfg.similarity_threshold,
                document_id_filter=cfg.document_id_filter,
                document_ids_filter=doc_ids_list,
            )
            self._assert_chunks_owned_and_filtered(dense_candidates, owner_str, allowed_doc_ids, "hybrid_dense")

            owner_bm25 = self.tenant_bm25_manager.get_or_create_index(owner_str)
            bm25_candidates = owner_bm25.search(
                query=query,
                top_k=cfg.candidate_k,
                document_id_filter=cfg.document_id_filter,
                document_ids_filter=doc_ids_list,
            )
            self._assert_chunks_owned_and_filtered(bm25_candidates, owner_str, allowed_doc_ids, "hybrid_bm25")

            fusion_limit = cfg.candidate_k if cfg.enable_reranking else cfg.top_k
            candidates = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                bm25_results=bm25_candidates,
                rrf_k=cfg.rrf_k,
                top_k=fusion_limit,
            )
            self._assert_chunks_owned_and_filtered(candidates, owner_str, allowed_doc_ids, "fusion")

        total_candidates_found = len(candidates)

        # -------------------------------------------------------------
        # Optional Cross-Encoder Reranking
        # -------------------------------------------------------------
        if cfg.enable_reranking and candidates:
            reranker = self.get_reranker(cfg.reranker_model)
            candidates, rerank_summary = reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=cfg.top_k,
            )
            self._assert_chunks_owned_and_filtered(candidates, owner_str, allowed_doc_ids, "rerank")
        else:
            candidates = candidates[:cfg.top_k]

        self._assert_chunks_owned_and_filtered(candidates, owner_str, allowed_doc_ids, "final")

        for idx, chunk in enumerate(candidates, start=1):
            chunk.rank = idx

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=query,
            mode=cfg.mode,
            total_candidates_found=total_candidates_found,
            returned_chunks_count=len(candidates),
            chunks=candidates,
            execution_time_ms=round(duration_ms, 2),
            configuration_applied=cfg.model_dump(),
            reranking_summary=rerank_summary,
        )

    @staticmethod
    def _has_markdown_table_rows(text: str) -> bool:
        """Return True if text contains at least one Markdown table row ('| ... |')."""
        for line in (text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("|") and stripped.count("|") >= 2:
                return True
        return False

    def expand_table_continuations(
        self,
        chunks: List[RetrievedChunk],
        owner_id: Union[UUID, str],
        max_extra_chunks: int = 10,
        authorized_ready_doc_ids: Optional[Set[str]] = None,
    ) -> List[RetrievedChunk]:
        """Include contiguous same-document Markdown table chunks so split tables remain complete."""
        if authorized_ready_doc_ids is not None:
            chunks = [c for c in chunks if c.document_id in authorized_ready_doc_ids]
        if not chunks or max_extra_chunks <= 0:
            return chunks
        if not any(self._has_markdown_table_rows(c.chunk_text) for c in chunks):
            return chunks

        owner_str = normalize_owner_id(owner_id)
        if self.vector_store is None or not hasattr(self.vector_store, "get_document_chunks"):
            return chunks

        seen_ids: Set[str] = {c.chunk_id for c in chunks}
        doc_chunks_cache: Dict[str, List[Dict[str, Any]]] = {}
        extra_chunks: List[RetrievedChunk] = []

        for chunk in chunks:
            if not self._has_markdown_table_rows(chunk.chunk_text):
                continue
            doc_id = chunk.document_id
            if authorized_ready_doc_ids is not None and doc_id not in authorized_ready_doc_ids:
                continue
            if doc_id not in doc_chunks_cache:
                try:
                    doc_chunks_cache[doc_id] = self.vector_store.get_document_chunks(
                        document_id=doc_id,
                        owner_id=owner_str,
                    )
                except TenantIsolationError:
                    raise
                except Exception as exc:
                    logger.debug("Skipping table continuation expansion for doc=%s: %s", doc_id, exc)
                    doc_chunks_cache[doc_id] = []

            doc_chunks = doc_chunks_cache[doc_id]
            if not doc_chunks:
                continue

            idx_in_doc = next(
                (i for i, item in enumerate(doc_chunks) if item["chunk_id"] == chunk.chunk_id),
                None,
            )
            if idx_in_doc is None:
                continue

            neighbor_indices: List[int] = []
            # Walk backward across contiguous table chunks
            pos = idx_in_doc - 1
            while pos >= 0 and self._has_markdown_table_rows(doc_chunks[pos].get("text", "")):
                neighbor_indices.append(pos)
                pos -= 1
            neighbor_indices.reverse()

            # Walk forward across contiguous table chunks
            pos = idx_in_doc + 1
            while pos < len(doc_chunks) and self._has_markdown_table_rows(doc_chunks[pos].get("text", "")):
                neighbor_indices.append(pos)
                pos += 1

            for n_idx in neighbor_indices:
                if len(extra_chunks) >= max_extra_chunks:
                    break
                raw = doc_chunks[n_idx]
                cid = raw["chunk_id"]
                if cid in seen_ids:
                    continue
                meta = dict(raw.get("metadata") or {})
                if meta.get("owner_id") != owner_str or raw.get("document_id") != doc_id:
                    raise TenantIsolationError("Cross-tenant or cross-document chunk rejected during table expansion.")
                page_refs = meta.get("page_references", [meta.get("page_number", 1)])
                if not isinstance(page_refs, list):
                    page_refs = [int(page_refs)]
                text = raw.get("text", "")
                extra_chunks.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        document_id=doc_id,
                        filename=meta.get("filename", chunk.filename),
                        page_number=int(meta.get("page_number", chunk.page_number)),
                        page_references=page_refs,
                        chunk_text=text,
                        start_offset=int(meta.get("start_offset", 0)),
                        end_offset=int(meta.get("end_offset", len(text))),
                        rank=len(chunks) + len(extra_chunks) + 1,
                        cosine_distance=chunk.cosine_distance,
                        cosine_similarity=chunk.cosine_similarity,
                        bm25_score=chunk.bm25_score,
                        rrf_score=chunk.rrf_score,
                        cross_encoder_score=chunk.cross_encoder_score,
                        metadata=meta,
                    )
                )
                seen_ids.add(cid)

        if not extra_chunks:
            return chunks

        allowed_docs = {c.document_id for c in chunks}
        self._assert_chunks_owned_and_filtered(extra_chunks, owner_str, allowed_docs, "table_expansion")
        return list(chunks) + extra_chunks


_retriever_instance: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get or create singleton Retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
