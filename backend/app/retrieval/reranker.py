"""Cross-encoder reranking engine for Retrieva AI.

Performs joint query-chunk cross-attention scoring using ms-marco-MiniLM-L-6-v2
to re-rank candidate chunks retrieved from initial bi-encoder or BM25 stages.
"""

import time
from typing import List, Optional, Tuple

try:
    from backend.app.core.logging import get_logger
    from backend.app.schemas.retrieval import RetrievedChunk, RerankingResult
except ImportError:
    from app.core.logging import get_logger
    from app.schemas.retrieval import RetrievedChunk, RerankingResult

logger = get_logger("retrieval.reranker")


class CrossEncoderReranker:
    """Evaluates query and candidate text jointly using a pretrained CrossEncoder."""

    _instance: Optional["CrossEncoderReranker"] = None

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        """Initialize the cross-encoder model.

        Args:
            model_name: HuggingFace model identifier.
            device: Compute device ('cpu', 'cuda', etc.). If None, auto-detected.
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load cross-encoder weights into memory."""
        try:
            logger.info("Loading CrossEncoder '%s' (device: %s)...", self.model_name, self.device)
            from sentence_transformers import CrossEncoder

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device

            self._model = CrossEncoder(self.model_name, **kwargs)
            logger.info("CrossEncoder '%s' loaded successfully", self.model_name)
        except Exception as exc:
            logger.error("Failed to load CrossEncoder '%s': %s", self.model_name, exc)
            raise RuntimeError(f"CrossEncoder load failed: {str(exc)}") from exc

    @classmethod
    def get_instance(
        cls,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> "CrossEncoderReranker":
        """Get or create singleton instance of the CrossEncoder."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievedChunk], RerankingResult]:
        """Score query-candidate pairs and re-order chunks by cross-encoder relevance.

        Args:
            query: Natural language query.
            candidates: Retrieved candidate chunks from initial stage.
            top_k: Optional slice limit for returned re-ordered results.

        Returns:
            Tuple of (reranked_chunks, reranking_summary_metadata).
        """
        if not candidates:
            return [], RerankingResult(
                model_name=self.model_name,
                candidates_in=0,
                candidates_out=0,
                duration_ms=0.0,
            )

        start_time = time.perf_counter()

        # Build joint input pairs
        pairs = [(query, chunk.chunk_text) for chunk in candidates]

        logger.info(
            "Reranking %d candidates with CrossEncoder '%s'...",
            len(pairs),
            self.model_name,
        )

        # Predict raw logits/scores
        raw_scores = self._model.predict(pairs)

        # Attach scores to candidate copies
        scored_candidates: List[RetrievedChunk] = []
        for idx, chunk in enumerate(candidates):
            score = float(raw_scores[idx])
            chunk.cross_encoder_score = round(score, 6)
            scored_candidates.append(chunk)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda c: c.cross_encoder_score or -9999.0, reverse=True)

        final_limit = top_k or len(scored_candidates)
        reranked_chunks = scored_candidates[:final_limit]

        # Re-assign 1-based ranks
        for new_rank, chunk in enumerate(reranked_chunks, start=1):
            chunk.rank = new_rank

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        summary = RerankingResult(
            model_name=self.model_name,
            candidates_in=len(candidates),
            candidates_out=len(reranked_chunks),
            duration_ms=round(duration_ms, 2),
        )

        logger.info(
            "Reranked %d -> %d candidates in %.2f ms",
            len(candidates),
            len(reranked_chunks),
            duration_ms,
        )

        return reranked_chunks, summary
