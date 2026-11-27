"""Regression tests for Milestone 3.12: Retrieval-to-Generation diagnostic (Project_Topic4_Report.pdf)."""

import json
from typing import Any, Dict, List, Optional
from uuid import UUID

import pytest

from backend.app.llm.context_builder import ContextBuilder
from backend.app.schemas.answer import AnswerRequest
from backend.app.schemas.retrieval import (
    RetrievalConfiguration,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)
from backend.app.services.rag_service import RAGService
from backend.app.vector_store.store import TenantIsolationError

USER_A_ID = UUID("10000000-0000-4000-8000-000000000001")
USER_B_ID = UUID("10000000-0000-4000-8000-000000000002")
DOC_REPORT_ID = "doc-project-topic4-report"
DOC_OTHER_ID = "doc-other-report"


def _build_project_topic4_chunks(owner_id: UUID = USER_A_ID) -> List[RetrievedChunk]:
    """Build deterministic ranked chunks reflecting Project_Topic4_Report.pdf evidence."""
    owner_str = str(owner_id)
    base_meta = {"owner_id": owner_str}
    return [
        RetrievedChunk(
            chunk_id="chunk-p2",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=2,
            page_references=[2],
            chunk_text=(
                "Executive Summary: This report investigates sugar content differences between "
                "beverage and food items. The main point-biserial correlation is r_pb = -0.4694, "
                "indicating a statistically significant negative association (p < 0.001)."
            ),
            start_offset=0,
            end_offset=220,
            rank=1,
            cross_encoder_score=5.4,
            rrf_score=0.032,
            metadata=base_meta,
        ),
        RetrievedChunk(
            chunk_id="chunk-p17",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=17,
            page_references=[17],
            chunk_text=(
                "Sensitivity and Significance Analysis: While the main point-biserial correlation "
                "is -0.4694, the sensitivity correlation after outlier trimming is -0.403. "
                "Hypothesis testing yields t-statistic = -7.231, df = 185, p < 0.001, and R^2 = 0.220."
            ),
            start_offset=0,
            end_offset=240,
            rank=2,
            cross_encoder_score=5.1,
            rrf_score=0.031,
            metadata=base_meta,
        ),
        RetrievedChunk(
            chunk_id="chunk-p16",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=16,
            page_references=[16],
            chunk_text=(
                "Table 4: Significance-Test Summary. t-statistic: -7.231, degrees of freedom (df): 185, "
                "p-value: < 0.001, explained variance R^2: 0.220."
            ),
            start_offset=0,
            end_offset=140,
            rank=3,
            cross_encoder_score=4.8,
            rrf_score=0.029,
            metadata=base_meta,
        ),
        RetrievedChunk(
            chunk_id="chunk-p5",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=5,
            page_references=[5],
            chunk_text=(
                "Data Collection Methodology: Nutritional labels were sampled across retail categories "
                "and standardized per 100g/100mL serving size prior to correlation testing."
            ),
            start_offset=0,
            end_offset=160,
            rank=4,
            cross_encoder_score=4.2,
            rrf_score=0.027,
            metadata=base_meta,
        ),
        RetrievedChunk(
            chunk_id="chunk-p9",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=9,
            page_references=[9],
            chunk_text=(
                "Assumption Checks: Normality diagnostics and Levene homogeneity checks were reviewed "
                "before computing group-level descriptive statistics."
            ),
            start_offset=0,
            end_offset=140,
            rank=5,
            cross_encoder_score=3.9,
            rrf_score=0.025,
            metadata=base_meta,
        ),
        RetrievedChunk(
            chunk_id="chunk-p12",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=12,
            page_references=[12],
            chunk_text=(
                "Main Group Statistics: The main point-biserial correlation is -0.4694. "
                "Beverage mean sugar content is 24.26 g, food mean sugar content is 41.49 g, "
                "and population standard deviation is 17.95 g."
            ),
            start_offset=0,
            end_offset=210,
            rank=6,
            cross_encoder_score=3.7,
            rrf_score=0.024,
            metadata=base_meta,
        ),
        RetrievedChunk(
            chunk_id="chunk-p19",
            document_id=DOC_REPORT_ID,
            filename="Project_Topic4_Report.pdf",
            page_number=19,
            page_references=[19],
            chunk_text=(
                "Limitations: Observational cross-sectional packaging data cannot establish "
                "causal dietary intake patterns."
            ),
            start_offset=0,
            end_offset=110,
            rank=7,
            cross_encoder_score=3.2,
            rrf_score=0.021,
            metadata=base_meta,
        ),
    ]


class FixtureRetriever:
    """In-memory retriever honoring owner_id, document filters, and top_k truncation."""

    def __init__(self, chunks_by_owner: Dict[str, List[RetrievedChunk]]) -> None:
        self._chunks_by_owner = chunks_by_owner
        self.observed_requests: List[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest, owner_id: UUID) -> RetrievalResult:
        self.observed_requests.append(request)
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
            execution_time_ms=2.5,
            configuration_applied=request.config.model_dump(),
        )


class EvidenceAwareLLMClient:
    """Deterministic LLM double that inspects the bounded prompt context like a grounded model."""

    model = "gpt-4o-mini"

    def __init__(self, *, force_fabricated_citation: bool = False) -> None:
        self.force_fabricated_citation = force_fabricated_citation
        self.last_user_prompt = ""

    def _synthesize(self, user_prompt: str) -> Dict[str, Any]:
        self.last_user_prompt = user_prompt
        if self.force_fabricated_citation:
            return {
                "answer": "The main correlation is -0.4694 [Source 99].",
                "insufficient_context": False,
                "citations": [{"source_id": "[Source 99]", "quote": "fabricated", "claim": "main correlation"}],
            }

        has_means = "24.26 g" in user_prompt and "41.49 g" in user_prompt
        has_main_corr = "-0.4694" in user_prompt
        has_sig = "-7.231" in user_prompt and "185" in user_prompt and "0.220" in user_prompt
        has_sens = "-0.403" in user_prompt

        if "median household income" in user_prompt.lower():
            return {
                "answer": "The provided documents do not contain sufficient information to answer this question.",
                "insufficient_context": True,
                "citations": [],
            }

        if not (has_means and has_main_corr and has_sig):
            return {
                "answer": "The provided documents do not contain sufficient information to answer this question.",
                "insufficient_context": True,
                "citations": [],
            }

        # Map actual [Source N] tags present in formatted context
        p12_tag = self._find_source_tag(user_prompt, "chunk-p12") or "[Source 6]"
        p17_tag = self._find_source_tag(user_prompt, "chunk-p17") or "[Source 2]"
        p16_tag = self._find_source_tag(user_prompt, "chunk-p16") or "[Source 3]"

        ans = (
            f"The main point-biserial correlation is -0.4694 {p12_tag}{p17_tag}, with a beverage mean of "
            f"24.26 g, food mean of 41.49 g, and population standard deviation of 17.95 g {p12_tag}. "
            f"Sensitivity analysis reports a correlation of -0.403 (distinct from the main result) {p17_tag}. "
            f"Significance testing shows t = -7.231, df = 185, p < 0.001, and R^2 = 0.220 {p17_tag}{p16_tag}."
        )
        return {
            "answer": ans,
            "insufficient_context": False,
            "citations": [
                {"source_id": p12_tag, "quote": "Beverage mean sugar content is 24.26 g, food mean sugar content is 41.49 g", "claim": "Group means and main correlation"},
                {"source_id": p17_tag, "quote": "sensitivity correlation after outlier trimming is -0.403", "claim": "Sensitivity correlation and significance"},
                {"source_id": p16_tag, "quote": "t-statistic: -7.231, degrees of freedom (df): 185", "claim": "Significance summary"},
            ],
        }

    @staticmethod
    def _find_source_tag(user_prompt: str, chunk_id: str) -> Optional[str]:
        for line in user_prompt.splitlines():
            if chunk_id in line and "[Source " in line:
                start = line.index("[Source ")
                end = line.index("]", start) + 1
                return line[start:end]
        return None

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        payload = self._synthesize(messages[-1]["content"])
        return {
            "content": json.dumps(payload),
            "model": self.model,
            "prompt_tokens": 600,
            "completion_tokens": 120,
            "total_tokens": 720,
        }

    async def generate_stream_async(self, messages: List[Dict[str, str]], **kwargs):
        payload = self._synthesize(messages[-1]["content"])
        yield payload["answer"]


class TestRetrievalToGenerationDiagnosticM312:
    """Reproduces the top_k=5 truncation vs top_k=7 recovery and verifies grounding + isolation."""

    def test_top_k_5_drops_rank_6_means_while_default_qa_top_k_7_answers_all_metrics(self):
        chunks = _build_project_topic4_chunks(USER_A_ID)
        retriever = FixtureRetriever({str(USER_A_ID): chunks})
        llm = EvidenceAwareLLMClient()
        svc = RAGService(
            retriever=retriever,
            llm_client=llm,
            context_builder=ContextBuilder(max_context_tokens=2500, model_name="gpt-4o-mini"),
        )
        question = (
            "What are the main point-biserial correlation, beverage and food group means, "
            "population SD, sensitivity correlation, t-statistic, df, p-value, and R^2?"
        )

        # Reproduce pre-fix top_k=5 behavior: 5/5 chunks used, but rank 6 (Page 12) is cut off
        resp_k5 = svc.answer_question(
            AnswerRequest(
                query=question,
                document_ids=[DOC_REPORT_ID],
                retrieval_config=RetrievalConfiguration(mode=RetrievalMode.HYBRID, top_k=5, enable_reranking=True),
            ),
            owner_id=USER_A_ID,
        )
        assert resp_k5.insufficient_context is True
        assert resp_k5.metadata.total_chunks_retrieved == 5
        assert resp_k5.metadata.context_chunks_used == 5
        assert resp_k5.metadata.context_chunks_excluded == 0

        # Post-fix default QA (retrieval_config=None -> top_k=7): all 7 chunks fit within 2500 tokens
        resp_default = svc.answer_question(
            AnswerRequest(query=question, document_ids=[DOC_REPORT_ID]),
            owner_id=USER_A_ID,
        )
        assert retriever.observed_requests[-1].config.top_k == 7
        assert resp_default.insufficient_context is False
        assert resp_default.metadata.context_chunks_used == 7
        assert resp_default.metadata.context_chunks_excluded == 0
        assert resp_default.metadata.context_tokens_used <= 2500

        # Verify main correlation, both group means, sensitivity distinction, significance, and R^2
        assert "-0.4694" in resp_default.answer
        assert "24.26 g" in resp_default.answer
        assert "41.49 g" in resp_default.answer
        assert "17.95 g" in resp_default.answer
        assert "-0.403" in resp_default.answer
        assert "-7.231" in resp_default.answer
        assert "185" in resp_default.answer
        assert "0.220" in resp_default.answer

        # Citations only refer to chunks included in sources_used
        used_ids = {s.chunk_id for s in resp_default.sources_used}
        assert len(resp_default.citations) >= 3
        assert {c.chunk_id for c in resp_default.citations}.issubset(used_ids)

    @pytest.mark.asyncio
    async def test_streaming_qa_recovers_all_evidence_and_unsupported_query_still_refuses(self):
        chunks = _build_project_topic4_chunks(USER_A_ID)
        retriever = FixtureRetriever({str(USER_A_ID): chunks})
        llm = EvidenceAwareLLMClient()
        svc = RAGService(
            retriever=retriever,
            llm_client=llm,
            context_builder=ContextBuilder(max_context_tokens=2500, model_name="gpt-4o-mini"),
        )

        # Streaming QA for supported multi-page question
        events: List[str] = []
        async for ev in svc.answer_question_stream(
            AnswerRequest(query="Report main correlation, group means, sensitivity correlation, t, df, p, R^2.", document_ids=[DOC_REPORT_ID]),
            owner_id=USER_A_ID,
        ):
            events.append(ev)
        joined = "".join(events)
        assert '"insufficient_context": false' in joined
        assert "24.26 g" in joined and "41.49 g" in joined and "-0.403" in joined and "0.220" in joined

        # Genuinely unsupported question still returns insufficient_context=True
        unsupported = svc.answer_question(
            AnswerRequest(query="What was the participants' median household income?", document_ids=[DOC_REPORT_ID]),
            owner_id=USER_A_ID,
        )
        assert unsupported.insufficient_context is True
        assert unsupported.citations == []

    def test_citation_validation_rejects_out_of_context_sources_and_preserves_tenant_isolation(self):
        chunks_a = _build_project_topic4_chunks(USER_A_ID)
        chunks_b = [
            RetrievedChunk(
                chunk_id="chunk-b1",
                document_id=DOC_OTHER_ID,
                filename="UserB_Secret.pdf",
                page_number=1,
                page_references=[1],
                chunk_text="Confidential User B data.",
                start_offset=0,
                end_offset=25,
                rank=1,
                metadata={"owner_id": str(USER_B_ID)},
            )
        ]
        retriever = FixtureRetriever({str(USER_A_ID): chunks_a, str(USER_B_ID): chunks_b})

        # Fabricated/excluded citation still triggers safe rejection
        bad_llm = EvidenceAwareLLMClient(force_fabricated_citation=True)
        bad_svc = RAGService(retriever=retriever, llm_client=bad_llm)
        rejected = bad_svc.answer_question(
            AnswerRequest(query="What is the main correlation?", document_ids=[DOC_REPORT_ID]),
            owner_id=USER_A_ID,
        )
        assert rejected.insufficient_context is True
        assert rejected.metadata.has_hallucinated_sources is True
        assert rejected.citations == []

        # User B filtering for User A's document receives 0 chunks and never sees User A's evidence
        good_svc = RAGService(retriever=retriever, llm_client=EvidenceAwareLLMClient())
        isolated = good_svc.answer_question(
            AnswerRequest(query="What is the main correlation?", document_ids=[DOC_REPORT_ID]),
            owner_id=USER_B_ID,
        )
        assert isolated.insufficient_context is True
        assert isolated.sources_used == []
        assert isolated.metadata.total_chunks_retrieved == 0
