"""Reciprocal Rank Fusion (RRF) for hybrid dense and keyword retrieval.

Implements rank-based score aggregation that avoids arbitrary score normalization
across disparate scales (cosine similarity vs. unbounded BM25 scores).
"""

from typing import Dict, List, Optional

try:
    from backend.app.core.logging import get_logger
    from backend.app.schemas.retrieval import RetrievedChunk
except ImportError:
    from app.core.logging import get_logger
    from app.schemas.retrieval import RetrievedChunk

logger = get_logger("retrieval.fusion")


def reciprocal_rank_fusion(
    dense_results: List[RetrievedChunk],
    bm25_results: List[RetrievedChunk],
    rrf_k: int = 60,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    """Merge ranked candidate lists from dense and BM25 retrievers using RRF.

    Formula:
        RRF_score(d) = sum_{m in {dense, bm25}} 1 / (rrf_k + rank_m(d))

    Args:
        dense_results: Ordered candidates from dense semantic retrieval.
        bm25_results: Ordered candidates from BM25 keyword retrieval.
        rrf_k: Smoothing constant preventing high ranks from dominating (default: 60).
        top_k: Final number of merged results to return.

    Returns:
        Deduplicated list of RetrievedChunk objects ordered descending by rrf_score.
    """
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be at least 1, got {rrf_k}")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    # chunk_id -> aggregate RRF score
    rrf_scores: Dict[str, float] = {}
    # chunk_id -> merged RetrievedChunk
    chunk_map: Dict[str, RetrievedChunk] = {}

    # Process Dense candidates
    for idx, chunk in enumerate(dense_results, start=1):
        chunk_id = chunk.chunk_id
        rank = getattr(chunk, "rank", None) or idx
        score_component = 1.0 / (rrf_k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_component
        chunk_map[chunk_id] = chunk

    # Process BM25 candidates
    for idx, chunk in enumerate(bm25_results, start=1):
        chunk_id = chunk.chunk_id
        rank = getattr(chunk, "rank", None) or idx
        score_component = 1.0 / (rrf_k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_component

        if chunk_id in chunk_map:
            # Merge BM25 score into existing candidate from dense
            existing = chunk_map[chunk_id]
            existing.bm25_score = chunk.bm25_score
        else:
            chunk_map[chunk_id] = chunk

    # Sort all candidates descending by aggregate RRF score
    sorted_chunk_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: rrf_scores[cid],
        reverse=True,
    )

    top_chunk_ids = sorted_chunk_ids[:top_k]

    fused_results: List[RetrievedChunk] = []
    for final_rank, chunk_id in enumerate(top_chunk_ids, start=1):
        candidate = chunk_map[chunk_id]
        final_score = rrf_scores[chunk_id]

        candidate.rrf_score = round(final_score, 8)
        candidate.rank = final_rank
        fused_results.append(candidate)

    logger.info(
        "Reciprocal Rank Fusion merged %d dense + %d bm25 candidates -> %d unique results",
        len(dense_results),
        len(bm25_results),
        len(fused_results),
    )

    return fused_results
