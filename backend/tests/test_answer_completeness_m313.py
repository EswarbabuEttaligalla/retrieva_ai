"""Regression tests for Milestone 3.13: Multi-Part Answer Completeness and Missing-Evidence Handling."""

import json
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

import pytest

from backend.app.llm.context_builder import ContextBuilder
from backend.app.llm.prompt_builder import PromptBuilder, STREAMING_SYSTEM_PROMPT, SYSTEM_PROMPT
from backend.app.schemas.answer import AnswerRequest
from backend.app.schemas.retrieval import RetrievalRequest, RetrievalResult, RetrievedChunk
from backend.app.services.rag_service import RAGService
from backend.app.vector_store.store import TenantIsolationError

USER_A_ID = UUID("10000000-0000-4000-8000-000000000001")
USER_B_ID = UUID("10000000-0000-4000-8000-000000000002")
DOC_REPORT_ID = "doc-project-topic4-report"
DOC_OTHER_ID = "doc-unrelated-report"


def _build_topic4_passages_with_page13(owner_id: UUID = USER_A_ID) -> List[RetrievedChunk]:
    """Grounded Project_Topic4_Report.pdf fixture chunks including Page 12, 13, 16, 17."""
    owner_str = str(owner_id)
    meta = {"owner_id": owner_str}
    return [
        RetrievedChunk(
            chunk_id="p13-sensitivity",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=13,
            page_references=[13],
            chunk_text=(
                "Sensitivity Analysis (Zero-Carbohydrate Beverages Removed): After removing the nine "
                "zero-carbohydrate beverages, the beverage mean carbohydrate content is 27.6 g and the "
                "sensitivity point-biserial correlation is r_pb = -0.403."
            ),
            start_offset=0,
            end_offset=230,
            rank=1,
            cross_encoder_score=5.6,
            rrf_score=0.033,
            metadata=meta,
        ),
        RetrievedChunk(
            chunk_id="p12-main",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=12,
            page_references=[12],
            chunk_text=(
                "Original Main Analysis: The main point-biserial correlation is r_pb = -0.4694. "
                "The beverage mean carbohydrate content is 24.26 g, the food mean carbohydrate content "
                "is 41.49 g, and the population standard deviation is 17.95 g."
            ),
            start_offset=0,
            end_offset=225,
            rank=2,
            cross_encoder_score=5.2,
            rrf_score=0.031,
            metadata=meta,
        ),
        RetrievedChunk(
            chunk_id="p17-significance",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=17,
            page_references=[17],
            chunk_text=(
                "Significance Testing: Main correlation r_pb = -0.4694, sensitivity correlation "
                "r_pb = -0.403, t-statistic = -7.231, df = 185, p < 0.001, and R^2 = 0.220."
            ),
            start_offset=0,
            end_offset=160,
            rank=3,
            cross_encoder_score=4.9,
            rrf_score=0.029,
            metadata=meta,
        ),
        RetrievedChunk(
            chunk_id="p16-table",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=16,
            page_references=[16],
            chunk_text=(
                "Significance-Test Summary Table: t = -7.231, df = 185, p < 0.001, R^2 = 0.220."
            ),
            start_offset=0,
            end_offset=85,
            rank=4,
            cross_encoder_score=4.5,
            rrf_score=0.027,
            metadata=meta,
        ),
    ]


class PassageFixtureRetriever:
    """In-memory retriever enforcing owner_id and document filters."""

    def __init__(self, chunks_by_owner: Dict[str, List[RetrievedChunk]]) -> None:
        self._chunks_by_owner = chunks_by_owner

    def retrieve(self, request: RetrievalRequest, owner_id: UUID) -> RetrievalResult:
        owner_str = str(owner_id)
        pool = list(self._chunks_by_owner.get(owner_str, []))
        allowed = set(request.config.document_ids_filter or [])
        if request.config.document_id_filter:
            allowed = allowed & {request.config.document_id_filter} if allowed else {request.config.document_id_filter}
        if allowed:
            pool = [c for c in pool if c.document_id in allowed]
        for chunk in pool:
            if (chunk.metadata or {}).get("owner_id") != owner_str:
                raise TenantIsolationError("Cross-tenant chunk rejected.")
        selected = pool[: request.config.top_k]
        return RetrievalResult(
            query=request.query,
            mode=request.config.mode,
            total_candidates_found=len(pool),
            returned_chunks_count=len(selected),
            chunks=selected,
            execution_time_ms=1.8,
            configuration_applied=request.config.model_dump(),
        )


class MultiPartCompletenessLLM:
    """Deterministic grounded LLM double modeling multi-part completeness & missing-evidence rules."""

    model = "gpt-4o-mini"

    @staticmethod
    def _tag(user_prompt: str, chunk_id: str) -> str:
        for line in user_prompt.splitlines():
            if chunk_id in line and "[Source " in line:
                start = line.index("[Source ")
                end = line.index("]", start) + 1
                return line[start:end]
        return "[Source 1]"

    def _synthesize(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        sys_prompt = messages[0]["content"]
        user_prompt = messages[-1]["content"]
        # Verify updated prompt directives are present on every call
        assert "Not reported in the provided document evidence." in sys_prompt
        assert "Not reported in the provided document evidence." in user_prompt

        q_lower = user_prompt.split("USER QUESTION:", 1)[-1].lower()
        p13 = self._tag(user_prompt, "p13-sensitivity")
        p12 = self._tag(user_prompt, "p12-main")
        p17 = self._tag(user_prompt, "p17-significance")
        p16 = self._tag(user_prompt, "p16-table")

        # 1. Exact verified partial-evidence sensitivity question
        if "removing the nine zero-carbohydrate beverages" in q_lower and "foods in that same sensitivity analysis" in q_lower:
            if "27.6 g" not in user_prompt:
                return {
                    "answer": "The provided documents do not contain sufficient information to answer this question.",
                    "insufficient_context": True,
                    "citations": [],
                }
            answer = (
                f"1. Beverage mean carbohydrate content after removing the nine zero-carbohydrate beverages: "
                f"27.6 g {p13}.\n"
                f"2. Food mean carbohydrate content in that same sensitivity analysis: "
                f"Not reported in the provided document evidence."
            )
            return {
                "answer": answer,
                "insufficient_context": False,
                "citations": [
                    {
                        "source_id": p13,
                        "quote": "After removing the nine zero-carbohydrate beverages, the beverage mean carbohydrate content is 27.6 g",
                        "claim": "Sensitivity beverage mean carbohydrate content is 27.6 g",
                    }
                ],
            }

        # 2. Complete multi-part main + significance question
        if "main point-biserial correlation" in q_lower and "r^2" in q_lower:
            answer = (
                f"1. Main point-biserial correlation: -0.4694 {p12}.\n"
                f"2. Main group means: beverage mean is 24.26 g {p12} and food mean is 41.49 g {p12} (SD = 17.95 g {p12}).\n"
                f"3. Significance statistics and R^2: t = -7.231 {p17}, df = 185 {p17}, p < 0.001 {p17}, and R^2 = 0.220 {p16}."
            )
            return {
                "answer": answer,
                "insufficient_context": False,
                "citations": [
                    {"source_id": p12, "quote": "main point-biserial correlation is r_pb = -0.4694", "claim": "Main correlation and group means"},
                    {"source_id": p17, "quote": "t-statistic = -7.231, df = 185, p < 0.001", "claim": "Significance test statistics"},
                    {"source_id": p16, "quote": "R^2 = 0.220", "claim": "Explained variance R^2"},
                ],
            }

        # 3. Unsupported numerical question
        return {
            "answer": "Not reported in the provided document evidence.",
            "insufficient_context": True,
            "citations": [],
        }

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        payload = self._synthesize(messages)
        return {
            "content": json.dumps(payload),
            "model": self.model,
            "prompt_tokens": 500,
            "completion_tokens": 110,
            "total_tokens": 610,
        }

    async def generate_stream_async(self, messages: List[Dict[str, str]], **kwargs):
        payload = self._synthesize(messages)
        yield payload["answer"]


class TestAnswerCompletenessM313:
    """Tests for multi-part completeness, missing-evidence disclosure, and analysis distinction."""

    def test_prompts_enforce_completeness_missing_evidence_and_analysis_distinction(self):
        for prompt in (SYSTEM_PROMPT, STREAMING_SYSTEM_PROMPT):
            assert "Not reported in the provided document evidence." in prompt
            assert "sensitivity" in prompt.lower()
            assert "every" in prompt.lower()
            assert "[Source N]" in prompt or "[Source 1]" in prompt
        user_json = PromptBuilder.build_user_prompt("Q1?", "--- [Source 1] ---")
        user_stream = PromptBuilder.build_streaming_user_prompt("Q1?", "--- [Source 1] ---")
        assert "Not reported in the provided document evidence." in user_json
        assert "Not reported in the provided document evidence." in user_stream

    def test_complete_multipart_answer_reports_all_metrics_with_citations(self):
        retriever = PassageFixtureRetriever({str(USER_A_ID): _build_topic4_passages_with_page13(USER_A_ID)})
        svc = RAGService(
            retriever=retriever,
            llm_client=MultiPartCompletenessLLM(),
            context_builder=ContextBuilder(max_context_tokens=2500, model_name="gpt-4o-mini"),
        )
        resp = svc.answer_question(
            AnswerRequest(
                query="What are the main point-biserial correlation, beverage and food means, SD, t, df, p, and R^2?",
                document_ids=[DOC_REPORT_ID],
            ),
            owner_id=USER_A_ID,
        )
        assert resp.insufficient_context is False
        for expected in ("-0.4694", "24.26 g", "41.49 g", "17.95 g", "-7.231", "185", "0.001", "0.220"):
            assert expected in resp.answer
        # Every numerical line has a [Source N] citation
        for line in resp.answer.splitlines():
            if re.search(r"\d", line):
                assert "[Source " in line
        assert {c.chunk_id for c in resp.citations}.issubset({s.chunk_id for s in resp.sources_used})

    def test_partial_sensitivity_evidence_discloses_missing_food_mean_without_substituting_main_mean(self):
        retriever = PassageFixtureRetriever({str(USER_A_ID): _build_topic4_passages_with_page13(USER_A_ID)})
        svc = RAGService(
            retriever=retriever,
            llm_client=MultiPartCompletenessLLM(),
            context_builder=ContextBuilder(max_context_tokens=2500, model_name="gpt-4o-mini"),
        )
        query = (
            "What is the exact mean carbohydrate content of beverages after removing the nine "
            "zero-carbohydrate beverages, and what is the exact mean carbohydrate content of foods "
            "in that same sensitivity analysis?"
        )
        resp = svc.answer_question(
            AnswerRequest(query=query, document_ids=[DOC_REPORT_ID]),
            owner_id=USER_A_ID,
        )
        assert resp.insufficient_context is False
        assert "27.6 g" in resp.answer
        assert "Not reported in the provided document evidence." in resp.answer
        # Must NOT substitute original main food mean (41.49 g) or main beverage mean (24.26 g)
        assert "41.49" not in resp.answer
        assert "24.26" not in resp.answer
        assert "nowhere in the document" not in resp.answer.lower()
        assert len(resp.citations) == 1
        assert resp.citations[0].chunk_id == "p13-sensitivity"

    @pytest.mark.asyncio
    async def test_streaming_and_non_streaming_consistency_and_unsupported_numerical_refusal(self):
        retriever = PassageFixtureRetriever({str(USER_A_ID): _build_topic4_passages_with_page13(USER_A_ID)})
        svc = RAGService(
            retriever=retriever,
            llm_client=MultiPartCompletenessLLM(),
            context_builder=ContextBuilder(max_context_tokens=2500, model_name="gpt-4o-mini"),
        )
        partial_q = (
            "What is the exact mean carbohydrate content of beverages after removing the nine "
            "zero-carbohydrate beverages, and what is the exact mean carbohydrate content of foods "
            "in that same sensitivity analysis?"
        )
        stream_events = [
            ev
            async for ev in svc.answer_question_stream(
                AnswerRequest(query=partial_q, document_ids=[DOC_REPORT_ID]),
                owner_id=USER_A_ID,
            )
        ]
        joined = "".join(stream_events)
        assert "27.6 g" in joined
        assert "Not reported in the provided document evidence." in joined
        assert "41.49" not in joined
        assert '"insufficient_context": false' in joined
        assert '"p13-sensitivity"' in joined

        # Unsupported numerical question in both non-streaming and streaming
        unsupported_q = "What is the exact ANOVA F-statistic for sugar content across retail brands?"
        non_stream_refusal = svc.answer_question(
            AnswerRequest(query=unsupported_q, document_ids=[DOC_REPORT_ID]),
            owner_id=USER_A_ID,
        )
        assert non_stream_refusal.insufficient_context is True
        assert non_stream_refusal.citations == []

        stream_refusal_events = [
            ev
            async for ev in svc.answer_question_stream(
                AnswerRequest(query=unsupported_q, document_ids=[DOC_REPORT_ID]),
                owner_id=USER_A_ID,
            )
        ]
        joined_refusal = "".join(stream_refusal_events)
        assert "Not reported in the provided document evidence." in joined_refusal
        assert '"insufficient_context": true' in joined_refusal

    def test_tenant_and_document_isolation_preserved(self):
        retriever = PassageFixtureRetriever({str(USER_A_ID): _build_topic4_passages_with_page13(USER_A_ID)})
        svc = RAGService(retriever=retriever, llm_client=MultiPartCompletenessLLM())

        # Different tenant cannot access User A's report
        other_user = svc.answer_question(
            AnswerRequest(query="What is 27.6 g after removing the nine zero-carbohydrate beverages?", document_ids=[DOC_REPORT_ID]),
            owner_id=USER_B_ID,
        )
        assert other_user.insufficient_context is True
        assert other_user.sources_used == []

        # Different document filter excludes DOC_REPORT_ID
        wrong_doc = svc.answer_question(
            AnswerRequest(query="What is 27.6 g after removing the nine zero-carbohydrate beverages?", document_ids=[DOC_OTHER_ID]),
            owner_id=USER_A_ID,
        )
        assert wrong_doc.insufficient_context is True
        assert wrong_doc.sources_used == []
