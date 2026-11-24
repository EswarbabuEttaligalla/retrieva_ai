"""Pydantic schemas for LLM Answer Generation and RAG Pipeline Orchestration."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from backend.app.schemas.retrieval import RetrievalConfiguration


class ContextSource(BaseModel):
    """Represents a retrieved document chunk selected into the bounded LLM context."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Stable prompt reference identifier, e.g. '[Source 1]'")
    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Parent document SHA-256 identifier")
    filename: str = Field(description="Source document filename")
    page_number: int = Field(description="Primary page number")
    page_references: List[int] = Field(default_factory=list, description="All pages spanned by this chunk")
    chunk_text: str = Field(description="Exact verbatim text of the chunk")
    retrieval_rank: int = Field(description="1-based rank from retrieval engine")
    retrieval_score: float = Field(description="Similarity or fusion score from retrieval")
    token_count: int = Field(description="Estimated token count of chunk text")


class ExcludedContextChunk(BaseModel):
    """Represents a retrieved chunk excluded from LLM context due to budget or filtering."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Identifier of excluded chunk")
    document_id: str = Field(description="Document ID")
    filename: str = Field(description="Document filename")
    retrieval_rank: int = Field(description="Original retrieval rank")
    token_count: int = Field(description="Token count of excluded chunk")
    reason: str = Field(description="Detailed reason for exclusion (e.g., token budget exhausted)")


class SourceCitation(BaseModel):
    """Verified citation tying an answer claim to a specific retrieved source."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Cited source label matching context, e.g. '[Source 1]'")
    chunk_id: str = Field(description="Unique ID of the cited chunk")
    document_id: str = Field(description="Parent document ID")
    filename: str = Field(description="Source filename")
    page_references: List[int] = Field(default_factory=list, description="Source page numbers")
    quote: Optional[str] = Field(default=None, description="Direct supporting quote if available")
    claim: Optional[str] = Field(default=None, description="Specific claim supported by this source")


class AnswerMetadata(BaseModel):
    """Diagnostic execution metadata for answer generation and token economics."""

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(description="OpenAI model name used")
    retrieval_mode: str = Field(description="Retrieval strategy: dense, bm25, or hybrid")
    total_chunks_retrieved: int = Field(description="Total chunks returned by retriever")
    context_chunks_used: int = Field(description="Number of chunks fitting within token budget")
    context_chunks_excluded: int = Field(description="Number of chunks discarded due to token budget")
    context_tokens_used: int = Field(description="Total tokens consumed by included chunks")
    max_context_tokens: int = Field(description="Configured maximum context token ceiling")
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt tokens reported by OpenAI API")
    completion_tokens: Optional[int] = Field(default=None, description="Completion tokens reported by OpenAI API")
    total_tokens: Optional[int] = Field(default=None, description="Total API tokens consumed")
    request_tokens_budget: Optional[int] = Field(default=None, description="Configured total request token limit ceiling")
    has_hallucinated_sources: bool = Field(default=False, description="True if any cited source was fabricated or excluded")
    citation_errors: List[str] = Field(default_factory=list, description="Diagnostic descriptions of citation errors")
    latency_ms: float = Field(description="Total end-to-end pipeline latency in milliseconds")


class AnswerRequest(BaseModel):
    """Client request schema for RAG answer generation."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000, description="Natural language user question")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document filter")
    retrieval_config: Optional[RetrievalConfiguration] = Field(
        default=None,
        description="Optional custom retrieval parameters (mode, top_k, threshold)",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (defaults to settings.DEFAULT_TEMPERATURE)",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=50,
        le=4000,
        description="Maximum generation tokens (defaults to settings.DEFAULT_MAX_COMPLETION_TOKENS)",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional persistent conversation session ID for multi-turn history and query rewriting",
    )


class AnswerResponse(BaseModel):
    """Comprehensive response schema containing grounded answer and verified provenance."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Original user question")
    answer: str = Field(description="Context-grounded natural language answer")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Associated persistent conversation session ID",
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Standalone rewritten query if conversational rewriting was applied",
    )
    citations: List[SourceCitation] = Field(
        default_factory=list,
        description="Verified citations directly mapped to context chunks",
    )
    insufficient_context: bool = Field(
        default=False,
        description="True if available documents do not contain sufficient evidence to answer",
    )
    sources_used: List[ContextSource] = Field(
        default_factory=list,
        description="Chunks provided in LLM context",
    )
    sources_excluded: List[ExcludedContextChunk] = Field(
        default_factory=list,
        description="Chunks retrieved but excluded due to token budget limitations",
    )
    metadata: AnswerMetadata = Field(description="Performance and pipeline execution telemetry")
