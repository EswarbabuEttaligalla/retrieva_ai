"""Retrieval orchestrator service for Retrieva AI.

Coordinates Dense Semantic Retrieval, BM25 Keyword Search, Reciprocal Rank Fusion,
and Cross-Encoder Reranking across configurable retrieval modes.
"""

import time
from typing import Any, Dict, List, Optional

try:
    from backend.app.core.logging import get_logger
    from backend.app.retrieval.bm25 import BM25Index
    from backend.app.retrieval.dense import DenseRetriever
    from backend.app.retrieval.fusion import reciprocal_rank_fusion
    from backend.app.retrieval.reranker import CrossEncoderReranker
    from backend.app.schemas.retrieval import (
        RetrievalConfiguration,
        RetrievalMode,
        RetrievalRequest,
        RetrievalResult,
        RetrievedChunk,
    )
    from backend.app.vector_store.store import ChromaVectorStore
except ImportError:
    from app.core.logging import get_logger
    from app.retrieval.bm25 import BM25Index
    from app.retrieval.dense import DenseRetriever
    from app.retrieval.fusion import reciprocal_rank_fusion
    from app.retrieval.reranker import CrossEncoderReranker
    from app.schemas.retrieval import (
        RetrievalConfiguration,
        RetrievalMode,
        RetrievalRequest,
        RetrievalResult,
        RetrievedChunk,
    )
    from app.vector_store.store import ChromaVectorStore

logger = get_logger("retrieval.retriever")


class Retriever:
    """Production-grade retrieval orchestrator supporting Dense, BM25, and Hybrid strategies."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_index: Optional[BM25Index] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        auto_sync_bm25: bool = True,
    ) -> None:
        """Initialize retrieval components.

        Args:
            dense_retriever: DenseRetriever instance.
            bm25_index: BM25Index instance.
            vector_store: ChromaVectorStore instance.
            reranker: CrossEncoderReranker instance (lazy-loaded if None and requested).
            auto_sync_bm25: Whether to synchronize BM25 from ChromaDB on initialization.
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.dense_retriever = dense_retriever or DenseRetriever(vector_store=self.vector_store)
        self.bm25_index = bm25_index or BM25Index()
        self._reranker = reranker

        if auto_sync_bm25:
            self.sync_bm25_with_vector_store()

    def sync_bm25_with_vector_store(self) -> int:
        """Synchronize the in-memory BM25 index with all chunks stored in ChromaDB.

        Returns:
            Count of indexed chunks.
        """
        logger.info("Synchronizing BM25 index with ChromaDB persistent store...")
        try:
            records = self.vector_store.collection.get(include=["documents", "metadatas"])
            ids = records.get("ids", [])
            docs = records.get("documents", [])
            metas = records.get("metadatas", [])

            chunk_list: List[Dict[str, Any]] = []
            for idx, chunk_id in enumerate(ids):
                doc_text = docs[idx] if idx < len(docs) else ""
                raw_meta = metas[idx] if idx < len(metas) else {}
                deserialized_meta = self.vector_store.deserialize_metadata(raw_meta)

                chunk_list.append({
                    "chunk_id": chunk_id,
                    "document_id": deserialized_meta.get("document_id", ""),
                    "filename": deserialized_meta.get("filename", ""),
                    "text": doc_text,
                    "chunk_text": doc_text,
                    "metadata": deserialized_meta,
                })

            self.bm25_index.build_index(chunk_list)
            logger.info("BM25 index synchronized: %d chunks indexed", len(chunk_list))
            return len(chunk_list)
        except Exception as exc:
            logger.error("Failed to sync BM25 index with ChromaDB: %s", exc)
            return 0

    def get_reranker(self, model_name: str) -> CrossEncoderReranker:
        """Lazy load or return CrossEncoderReranker singleton."""
        if self._reranker is None or self._reranker.model_name != model_name:
            self._reranker = CrossEncoderReranker.get_instance(model_name=model_name)
        return self._reranker

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Execute the configured retrieval pipeline for a query.

        Args:
            request: Validated RetrievalRequest payload.

        Returns:
            RetrievalResult containing ranked chunks and stage execution metrics.
        """
        start_time = time.perf_counter()
        query = request.query
        cfg = request.config

        logger.info(
            "Executing retrieval (mode=%s, top_k=%d, rerank=%s) for query: '%s'...",
            cfg.mode.value,
            cfg.top_k,
            cfg.enable_reranking,
            query[:60],
        )

        rerank_summary = None
        candidates: List[RetrievedChunk] = []

        # -------------------------------------------------------------
        # Mode 1: DENSE Retrieval
        # -------------------------------------------------------------
        if cfg.mode == RetrievalMode.DENSE:
            fetch_k = cfg.candidate_k if cfg.enable_reranking else cfg.top_k
            candidates = self.dense_retriever.search(
                query=query,
                top_k=fetch_k,
                similarity_threshold=cfg.similarity_threshold,
                document_id_filter=cfg.document_id_filter,
            )

        # -------------------------------------------------------------
        # Mode 2: BM25 Retrieval
        # -------------------------------------------------------------
        elif cfg.mode == RetrievalMode.BM25:
            fetch_k = cfg.candidate_k if cfg.enable_reranking else cfg.top_k
            candidates = self.bm25_index.search(
                query=query,
                top_k=fetch_k,
                document_id_filter=cfg.document_id_filter,
            )

        # -------------------------------------------------------------
        # Mode 3: HYBRID Retrieval (Dense + BM25 + RRF)
        # -------------------------------------------------------------
        elif cfg.mode == RetrievalMode.HYBRID:
            # 1. Fetch dense candidates
            dense_candidates = self.dense_retriever.search(
                query=query,
                top_k=cfg.candidate_k,
                similarity_threshold=cfg.similarity_threshold,
                document_id_filter=cfg.document_id_filter,
            )

            # 2. Fetch BM25 candidates
            bm25_candidates = self.bm25_index.search(
                query=query,
                top_k=cfg.candidate_k,
                document_id_filter=cfg.document_id_filter,
            )

            # 3. Fuse with Reciprocal Rank Fusion
            fusion_limit = cfg.candidate_k if cfg.enable_reranking else cfg.top_k
            candidates = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                bm25_results=bm25_candidates,
                rrf_k=cfg.rrf_k,
                top_k=fusion_limit,
            )

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
        else:
            candidates = candidates[:cfg.top_k]

        # Ensure final ranks are 1-based sequential integers
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


_retriever_instance: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get or create singleton Retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
