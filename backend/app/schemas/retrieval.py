"""Pydantic data models and schemas for retrieval and semantic search."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class RetrievalMode(str, Enum):
    """Supported retrieval execution strategies."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RetrievedChunk(BaseModel):
    """A single retrieved evidence chunk with multi-stage ranking provenance."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original filename of the source document")
    page_number: int = Field(..., ge=1, description="Primary 1-indexed page number")
    page_references: List[int] = Field(
        default_factory=list,
        description="All page numbers referenced by this chunk",
    )
    chunk_text: str = Field(..., description="Verbatim chunk text")
    start_offset: int = Field(..., ge=0, description="Start offset in cleaned page text")
    end_offset: int = Field(..., ge=1, description="End offset in cleaned page text")
    rank: int = Field(..., ge=1, description="1-indexed rank position in final result list")

    # Score metrics across stages
    cosine_distance: Optional[float] = Field(
        default=None,
        description="Cosine distance from ChromaDB (0.0 = identical, 1.0 = orthogonal)",
    )
    cosine_similarity: Optional[float] = Field(
        default=None,
        description="Calculated cosine similarity (1.0 - cosine_distance)",
    )
    bm25_score: Optional[float] = Field(
        default=None,
        description="BM25 keyword relevance score",
    )
    rrf_score: Optional[float] = Field(
        default=None,
        description="Reciprocal Rank Fusion score (in hybrid mode)",
    )
    cross_encoder_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder relevance logit/score (if reranking was applied)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk metadata from vector store",
    )


class RetrievalConfiguration(BaseModel):
    """Runtime configuration parameters for a retrieval operation."""

    mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID,
        description="Retrieval strategy: 'dense', 'bm25', or 'hybrid'",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Final number of top-ranked chunks to return",
    )
    candidate_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of candidate chunks retrieved from each initial retriever before fusion/reranking",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity filter (chunks below this are discarded)",
    )
    enable_reranking: bool = Field(
        default=False,
        description="Whether to apply cross-encoder reranking on top candidates",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace model identifier for cross-encoder reranking",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Smoothing constant for Reciprocal Rank Fusion formula",
    )
    bm25_k1: float = Field(
        default=1.5,
        ge=0.0,
        description="BM25 term frequency saturation parameter",
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 document length normalization parameter",
    )
    document_id_filter: Optional[str] = Field(
        default=None,
        description="Optional filter restricting retrieval to a specific document ID",
    )
    document_ids_filter: Optional[List[str]] = Field(
        default=None,
        description="Optional filter restricting retrieval to a list of document IDs",
    )

    @field_validator("candidate_k")
    @classmethod
    def validate_candidate_k(cls, v: int, info) -> int:
        top_k = info.data.get("top_k", 5)
        if v < top_k:
            return top_k
        return v


class RetrievalRequest(BaseModel):
    """Input payload for a retrieval query."""

    query: str = Field(..., min_length=1, description="Natural language search query")
    config: RetrievalConfiguration = Field(
        default_factory=RetrievalConfiguration,
        description="Retrieval options and parameters",
    )

    @model_validator(mode="before")
    @classmethod
    def assemble_configuration(cls, data: Any) -> Any:
        """Allow convenience flat payloads (e.g. {query: '...', mode: 'dense', top_k: 5})
        by bundling top-level configuration fields into the 'config' sub-object."""
        if isinstance(data, dict):
            # If 'config' is not explicitly provided, look for config parameters at top-level
            if "config" not in data:
                config_fields = {
                    "mode",
                    "top_k",
                    "candidate_k",
                    "similarity_threshold",
                    "enable_reranking",
                    "reranker_model",
                    "rrf_k",
                    "bm25_k1",
                    "bm25_b",
                    "document_id_filter",
                    "document_ids_filter",
                }
                extracted_config = {k: v for k, v in data.items() if k in config_fields}
                if "document_ids" in data and "document_ids_filter" not in extracted_config:
                    extracted_config["document_ids_filter"] = data.get("document_ids")
                if extracted_config:
                    data = {k: v for k, v in data.items() if k not in config_fields and k != "document_ids"}
                    data["config"] = extracted_config
        return data

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Query string cannot be empty or whitespace-only.")
        return clean


class RerankingResult(BaseModel):
    """Result details from a cross-encoder reranking pass."""

    model_name: str
    candidates_in: int
    candidates_out: int
    duration_ms: float


class RetrievalResult(BaseModel):
    """Comprehensive output payload for a completed retrieval operation."""

    query: str = Field(..., description="The query string executed")
    mode: RetrievalMode = Field(..., description="Retrieval mode used ('dense', 'bm25', 'hybrid')")
    total_candidates_found: int = Field(..., ge=0, description="Total candidates identified before filtering")
    returned_chunks_count: int = Field(..., ge=0, description="Number of chunks returned in this result")
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="Ranked list of retrieved chunks")
    execution_time_ms: float = Field(..., ge=0.0, description="End-to-end retrieval execution duration in milliseconds")
    configuration_applied: Dict[str, Any] = Field(default_factory=dict, description="Resolved retrieval settings")
    reranking_summary: Optional[RerankingResult] = Field(default=None, description="Details if reranking was performed")
