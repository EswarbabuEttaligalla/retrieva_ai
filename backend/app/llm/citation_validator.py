"""Citation Validation Engine for Retrieva AI.

Verifies that all citations produced by the LLM strictly correspond to
chunks that were actually admitted into the prompt context, rejecting
hallucinated source IDs, excluded chunk references, or malformed markers.
"""

import re
from typing import Dict, List, NamedTuple, Optional, Set
from pydantic import BaseModel, ConfigDict

from backend.app.core.logging import get_logger
from backend.app.schemas.answer import ContextSource, ExcludedContextChunk, SourceCitation

logger = get_logger("llm.citation_validator")


class CitationValidationResult(BaseModel):
    """Structured report of citation validation."""

    model_config = ConfigDict(frozen=True)

    valid_citations: List[SourceCitation]
    invalid_citations: List[str]
    has_hallucinated_sources: bool
    citation_coverage: float  # Proportion of cited sources that were valid (0.0 to 1.0)


class CitationValidator:
    """Validates source citations against the bounded context provided to the LLM.

    PROVENANCE VALIDATION VS SEMANTIC ENTAILMENT:
    ---------------------------------------------
    1. What this validator verifies:
       - Provenance Authenticity: Guarantees that every cited source identifier
         (e.g., '[Source 1]' or chunk ID) corresponds to an authentic document chunk
         that was actually admitted into the LLM prompt context payload.
       - Rejection of Hallucinations: Identifies and flags fabricated identifiers
         (e.g., '[Source 99]') and references to chunks that were excluded due to
         token budgeting constraints.
       - Non-destructive Metadata Preservation: Preserves exact chunk IDs, document IDs,
         filenames, page numbers, and quotes without synthetic ID remapping.

    2. What this validator DOES NOT verify:
       - Semantic Entailment / NLI: Syntactic validity does NOT mathematically guarantee
         that the cited chunk logically entails or substantiates the natural-language claim
         made by the LLM. Detecting unsupported factual claims paired with valid citations
         or claims without citations requires an auxiliary Natural Language Inference (NLI)
         cross-encoder model or secondary judge LLM pass.
       - Retrieva AI makes NO claim of semantic entailment verification in Phase 5.
    """

    INLINE_CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

    @classmethod
    def extract_inline_citations(cls, text: str) -> List[str]:
        """Extract all '[Source N]' citation markers appearing in the text string."""
        if not text:
            return []
        matches = cls.INLINE_CITATION_PATTERN.findall(text)
        return [f"[Source {m}]" for m in matches]

    @classmethod
    def validate_citations(
        cls,
        raw_citations: Optional[List[Dict[str, Optional[str]]]],
        answer_text: str,
        sources_used: List[ContextSource],
        sources_excluded: Optional[List[ExcludedContextChunk]] = None,
    ) -> CitationValidationResult:
        """Validate all proposed citations against the included context sources.

        Args:
            raw_citations: Optional list of citation dicts from model JSON output.
            answer_text: Full generated answer text (inspected for inline citations).
            sources_used: Verified list of chunks that were included in the context.
            sources_excluded: Chunks that were excluded due to token budgeting.

        Returns:
            CitationValidationResult with validated citations and rejected identifiers.
        """
        sources_map: Dict[str, ContextSource] = {
            src.source_id.strip().lower(): src for src in sources_used
        }
        chunk_id_map: Dict[str, ContextSource] = {
            src.chunk_id: src for src in sources_used
        }
        excluded_chunk_ids: Set[str] = {
            c.chunk_id for c in (sources_excluded or [])
        }

        valid_citations: List[SourceCitation] = []
        invalid_citations: List[str] = []
        seen_chunk_ids: Set[str] = set()

        # 1. Process explicit citations provided in structured JSON
        if raw_citations:
            for item in raw_citations:
                raw_id = (item.get("source_id") or "").strip()
                normalized_id = raw_id.lower()
                quote = item.get("quote")
                claim = item.get("claim")

                if not raw_id:
                    invalid_citations.append("Empty source identifier")
                    continue

                if normalized_id in sources_map:
                    matched_source = sources_map[normalized_id]
                    if matched_source.chunk_id not in seen_chunk_ids:
                        valid_citations.append(
                            SourceCitation(
                                source_id=matched_source.source_id,
                                chunk_id=matched_source.chunk_id,
                                document_id=matched_source.document_id,
                                filename=matched_source.filename,
                                page_references=matched_source.page_references,
                                quote=quote,
                                claim=claim,
                            )
                        )
                        seen_chunk_ids.add(matched_source.chunk_id)
                else:
                    is_excluded = (
                        raw_id in excluded_chunk_ids
                        or any(c.chunk_id == raw_id for c in (sources_excluded or []))
                        or any(
                            f"[source {c.retrieval_rank}]" == normalized_id
                            for c in (sources_excluded or [])
                        )
                    )
                    if is_excluded:
                        msg = f"Cited source {raw_id} was excluded from context due to token budget"
                        logger.warning(msg)
                        invalid_citations.append(msg)
                    else:
                        msg = f"Fabricated source identifier: {raw_id} (not present in context)"
                        logger.warning(msg)
                        invalid_citations.append(msg)

        # 2. Extract and reconcile inline citations from the answer text
        inline_ids = cls.extract_inline_citations(answer_text)
        for inline_id in inline_ids:
            normalized_inline = inline_id.lower()
            if normalized_inline in sources_map:
                matched_source = sources_map[normalized_inline]
                if matched_source.chunk_id not in seen_chunk_ids:
                    valid_citations.append(
                        SourceCitation(
                            source_id=matched_source.source_id,
                            chunk_id=matched_source.chunk_id,
                            document_id=matched_source.document_id,
                            filename=matched_source.filename,
                            page_references=matched_source.page_references,
                            quote=None,
                            claim=None,
                        )
                    )
                    seen_chunk_ids.add(matched_source.chunk_id)
            else:
                is_excluded = any(
                    f"[source {c.retrieval_rank}]" == normalized_inline
                    for c in (sources_excluded or [])
                )
                if is_excluded:
                    msg = f"Cited inline source {inline_id} was excluded from context due to token budget"
                else:
                    msg = f"Fabricated inline citation in text: {inline_id}"

                if msg not in invalid_citations:
                    logger.warning(msg)
                    invalid_citations.append(msg)

        total_cited = len(valid_citations) + len(invalid_citations)
        coverage = (len(valid_citations) / total_cited) if total_cited > 0 else 1.0

        return CitationValidationResult(
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            has_hallucinated_sources=len(invalid_citations) > 0,
            citation_coverage=round(coverage, 4),
        )
