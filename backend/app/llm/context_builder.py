"""Context Construction Engine for Retrieva AI.

Transforms retrieved chunks from Phase 4 into a structured, bounded, and
provenance-tagged context payload for LLM prompt synthesis.
"""

from typing import List, NamedTuple, Optional
from pydantic import BaseModel, ConfigDict

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.llm.token_counter import TokenBudgetReport, TokenCounter
from backend.app.schemas.answer import ContextSource, ExcludedContextChunk
from backend.app.schemas.retrieval import RetrievedChunk

logger = get_logger("llm.context_builder")


class ContextBuildResult(BaseModel):
    """Result of bounded context construction."""

    model_config = ConfigDict(frozen=True)

    formatted_context: str
    sources_used: List[ContextSource]
    sources_excluded: List[ExcludedContextChunk]
    total_tokens_used: int
    max_budget: int


class ContextBuilder:
    """Constructs token-bounded, provenance-attributed context for LLM generation.

    Core Principles:
    1. Rank-Ordered Prioritization: Evaluates candidate chunks in order of retrieval rank (Rank #1 first).
    2. Strict Token Accounting: Computes full token usage (chunk text + provenance header) via tiktoken BPE.
    3. No Silent Truncation: Chunks that cannot fit fully within the remaining budget are excluded
       as complete units rather than truncated midway, avoiding partial statements or offset corruption.
    4. Stable Source Identifiers: Every included chunk receives a stable label '[Source N]'
       referenced directly in generated citations.

    Token Estimation Qualification:
    Text-token counts for document chunks and prompt strings are exact using tiktoken BPE.
    ChatML envelope framing (3 tokens/message + 3 priming) and response formatting allowances
    are documented conservative approximations, not observable exact API token serialization.
    """

    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize the context builder.

        Args:
            max_context_tokens: Maximum allowed tokens for retrieved context.
                Defaults to Settings.MAX_CONTEXT_TOKENS (default 2500).
            model_name: Configured model name used for exact tokenizer resolution.
        """
        settings = get_settings()
        self.max_context_tokens = (
            max_context_tokens if max_context_tokens is not None else settings.MAX_CONTEXT_TOKENS
        )
        self.model_name = model_name or settings.active_model

    @classmethod
    def calculate_request_budget_report(
        cls,
        query: str,
        system_prompt: Optional[str] = None,
        reserved_completion_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        total_request_limit: Optional[int] = None,
        max_context_limit: Optional[int] = None,
        safety_margin: int = 50,
    ) -> TokenBudgetReport:
        """Calculate detailed token budget breakdown using tiktoken and ChatML overhead."""
        return TokenCounter.calculate_request_token_budget(
            query=query,
            system_prompt=system_prompt,
            reserved_completion_tokens=reserved_completion_tokens,
            model_name=model_name,
            total_request_limit=total_request_limit,
            max_context_limit=max_context_limit,
            safety_margin=safety_margin,
        )

    @classmethod
    def calculate_allowed_context_budget(
        cls,
        query: str,
        system_prompt: Optional[str] = None,
        reserved_completion_tokens: Optional[int] = None,
        total_request_limit: Optional[int] = None,
        max_context_limit: Optional[int] = None,
        safety_margin: int = 50,
        model_name: Optional[str] = None,
    ) -> int:
        """Dynamically compute the token budget available for retrieved document context.

        Uses tiktoken where supported, reserves completion tokens, and includes a documented
        ChatML/API overhead allowance and safety margin.
        """
        report = cls.calculate_request_budget_report(
            query=query,
            system_prompt=system_prompt,
            reserved_completion_tokens=reserved_completion_tokens,
            model_name=model_name,
            total_request_limit=total_request_limit,
            max_context_limit=max_context_limit,
            safety_margin=safety_margin,
        )
        return report.allowed_context_budget

    def build_context(
        self,
        retrieved_chunks: List[RetrievedChunk],
        max_context_tokens: Optional[int] = None,
    ) -> ContextBuildResult:
        """Construct the bounded formatted context string and provenance records.

        Args:
            retrieved_chunks: Ranked list of retrieved chunks from Phase 4.
            max_context_tokens: Optional per-call token budget override.

        Returns:
            ContextBuildResult containing formatted text, included sources, and excluded chunks.
        """
        budget = max_context_tokens if max_context_tokens is not None else self.max_context_tokens
        sources_used: List[ContextSource] = []
        sources_excluded: List[ExcludedContextChunk] = []

        if not retrieved_chunks:
            logger.info("Context builder received 0 chunks; returning empty context.")
            return ContextBuildResult(
                formatted_context="",
                sources_used=[],
                sources_excluded=[],
                total_tokens_used=0,
                max_budget=budget,
            )

        accumulated_tokens = 0
        context_blocks: List[str] = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source_id = f"[Source {idx}]"
            pages_str = ", ".join(map(str, chunk.page_references)) if chunk.page_references else str(chunk.page_number)

            # Sanitize any delimiter collisions within untrusted document text
            sanitized_text = (
                chunk.chunk_text.replace("=== BEGIN RETRIEVED CONTEXT", "--- BEGIN RETRIEVED CONTEXT")
                .replace("=== END RETRIEVED CONTEXT ===", "--- END RETRIEVED CONTEXT ---")
                .strip()
            )

            # Construct clean structured block with immutable provenance tag
            header = f"--- {source_id} (Document: {chunk.filename}, Pages: [{pages_str}], ChunkID: {chunk.chunk_id}) ---\n"
            block_text = f"{header}{sanitized_text}\n\n"

            # Measure token count of full block using tiktoken model encoder
            token_res = TokenCounter.count_tokens(block_text, model_name=self.model_name)
            block_tokens = token_res.token_count

            # Check if block fits within remaining budget
            if accumulated_tokens + block_tokens <= budget:
                accumulated_tokens += block_tokens
                context_blocks.append(block_text)

                score = (
                    chunk.cross_encoder_score
                    if chunk.cross_encoder_score is not None
                    else (
                        chunk.rrf_score
                        if chunk.rrf_score is not None
                        else (chunk.cosine_similarity or chunk.bm25_score or 0.0)
                    )
                )
                sources_used.append(
                    ContextSource(
                        source_id=source_id,
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        page_number=chunk.page_number,
                        page_references=chunk.page_references,
                        chunk_text=chunk.chunk_text,
                        retrieval_rank=chunk.rank,
                        retrieval_score=round(score, 4),
                        token_count=block_tokens,
                    )
                )
            else:
                reason = (
                    f"Token budget exhausted (chunk requires {block_tokens} tokens, "
                    f"remaining budget is {budget - accumulated_tokens} of {budget})"
                )
                logger.debug(
                    "Excluding chunk %s (rank %d) from context: %s",
                    chunk.chunk_id,
                    chunk.rank,
                    reason,
                )
                sources_excluded.append(
                    ExcludedContextChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        retrieval_rank=chunk.rank,
                        token_count=block_tokens,
                        reason=reason,
                    )
                )

        formatted_context = "".join(context_blocks).strip()
        logger.info(
            "Context built: %d sources included (%d tokens), %d sources excluded (budget=%d)",
            len(sources_used),
            accumulated_tokens,
            len(sources_excluded),
            budget,
        )

        return ContextBuildResult(
            formatted_context=formatted_context,
            sources_used=sources_used,
            sources_excluded=sources_excluded,
            total_tokens_used=accumulated_tokens,
            max_budget=budget,
        )
