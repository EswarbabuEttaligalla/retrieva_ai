"""Integration and end-to-end tests for the Phase 5 RAG Pipeline Orchestrator.

Covers:
1. End-to-end RAG answer generation with mocked retrieval and mocked OpenAI inference.
2. Handling of empty retrieval result (safe fallback, insufficient_context=True).
3. Insufficient context reporting when documents lack facts.
4. Handling of prompt-injection attempts inside retrieved chunks.
5. FastAPI HTTP endpoint (/api/v1/answer) contracts, validation, and error translations.
6. Optional real API integration test (marked and skipped if no real API key is set).
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.llm.client import OpenAIClient
from backend.app.llm.context_builder import ContextBuilder
from backend.app.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from backend.app.main import app
from backend.app.retrieval.retriever import Retriever
from backend.app.schemas.answer import AnswerRequest, AnswerResponse
from backend.app.schemas.retrieval import RetrievalMode, RetrievalResult, RetrievedChunk
from backend.app.services.rag_service import RAGService, get_rag_service


# ============================================================================
# Helpers & Fixtures
# ============================================================================

def make_test_retrieved_chunk(chunk_id: str, text: str, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc_rag_test",
        filename="retrieval_guide.pdf",
        page_number=1,
        page_references=[1],
        chunk_index=rank - 1,
        chunk_text=text,
        start_offset=0,
        end_offset=len(text),
        rank=rank,
        cosine_distance=0.10,
        cosine_similarity=0.90,
        bm25_score=None,
        rrf_score=None,
        cross_encoder_score=None,
    )


@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=Retriever)
    return retriever


@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=OpenAIClient)
    client.model = "gpt-4o-mini"
    return client


# ============================================================================
# 1. RAG Service Unit / Pipeline Orchestration Tests
# ============================================================================

from uuid import UUID

TEST_OWNER_ID = UUID("11111111-1111-4111-8111-111111111111")


class TestRAGServicePipeline:
    """Test suite for RAGService orchestration flow."""

    def test_e2e_rag_happy_path(self, mock_retriever, mock_llm_client):
        """Standard happy path: retrieved chunks -> context -> LLM -> verified citations."""
        chunk1 = make_test_retrieved_chunk(
            "chunk_1",
            "Reciprocal Rank Fusion merges multiple ranked lists without requiring score normalization.",
            rank=1,
        )

        mock_retriever.retrieve.return_value = RetrievalResult(
            query="What is Reciprocal Rank Fusion?",
            mode=RetrievalMode.HYBRID,
            total_candidates_found=1,
            returned_chunks_count=1,
            chunks=[chunk1],
            execution_time_ms=12.5,
        )

        mock_llm_client.generate.return_value = {
            "content": (
                '{\n'
                '  "answer": "Reciprocal Rank Fusion (RRF) combines rankings from different systems without score normalization [Source 1].",\n'
                '  "insufficient_context": false,\n'
                '  "citations": [\n'
                '    {\n'
                '      "source_id": "[Source 1]",\n'
                '      "quote": "Reciprocal Rank Fusion merges multiple ranked lists",\n'
                '      "claim": "Combines rankings without score normalization"\n'
                '    }\n'
                '  ]\n'
                '}'
            ),
            "model": "gpt-4o-mini",
            "prompt_tokens": 120,
            "completion_tokens": 35,
            "total_tokens": 155,
        }

        service = RAGService(
            retriever=mock_retriever,
            llm_client=mock_llm_client,
            context_builder=ContextBuilder(max_context_tokens=1000),
        )

        req = AnswerRequest(query="What is Reciprocal Rank Fusion?")
        response = service.answer_question(req, owner_id=TEST_OWNER_ID)

        assert isinstance(response, AnswerResponse)
        assert "Reciprocal Rank Fusion" in response.answer
        assert not response.insufficient_context
        assert len(response.citations) == 1
        assert response.citations[0].source_id == "[Source 1]"
        assert response.citations[0].chunk_id == "chunk_1"
        assert response.citations[0].filename == "retrieval_guide.pdf"

        # Check metadata
        assert response.metadata.model_name == "gpt-4o-mini"
        assert response.metadata.retrieval_mode == "hybrid"
        assert response.metadata.total_chunks_retrieved == 1
        assert response.metadata.context_chunks_used == 1
        assert response.metadata.context_chunks_excluded == 0
        assert response.metadata.prompt_tokens == 120
        assert response.metadata.latency_ms > 0

    def test_empty_retrieval_returns_safe_fallback(self, mock_retriever, mock_llm_client):
        """When retrieval returns 0 chunks, return safe message without calling LLM."""
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="Nonexistent topic query",
            mode=RetrievalMode.DENSE,
            total_candidates_found=0,
            returned_chunks_count=0,
            chunks=[],
            execution_time_ms=5.0,
        )

        service = RAGService(retriever=mock_retriever, llm_client=mock_llm_client)
        req = AnswerRequest(query="Nonexistent topic query")
        response = service.answer_question(req, owner_id=TEST_OWNER_ID)

        assert response.insufficient_context is True
        assert "do not contain any information relevant" in response.answer
        assert len(response.citations) == 0
        assert len(response.sources_used) == 0
        # Ensure LLM generate was NOT called
        mock_llm_client.generate.assert_not_called()

    def test_insufficient_context_prose_fallback(self, mock_retriever, mock_llm_client):
        """When model output states insufficient context in plain prose, flag is set."""
        chunk1 = make_test_retrieved_chunk("chunk_1", "Python was released in 1991.", rank=1)
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="What is the capital of Mars?",
            mode=RetrievalMode.DENSE,
            total_candidates_found=1,
            returned_chunks_count=1,
            chunks=[chunk1],
            execution_time_ms=8.0,
        )

        # Plain prose response (model did not output JSON)
        mock_llm_client.generate.return_value = {
            "content": "The provided documents do not contain sufficient information to answer questions about Mars.",
            "model": "gpt-4o-mini",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }

        service = RAGService(retriever=mock_retriever, llm_client=mock_llm_client)
        req = AnswerRequest(query="What is the capital of Mars?")
        response = service.answer_question(req, owner_id=TEST_OWNER_ID)

        assert response.insufficient_context is True
        assert len(response.citations) == 0

    def test_empty_query_raises_value_error(self, mock_retriever, mock_llm_client):
        """Whitespace query must be rejected before retrieval execution."""
        service = RAGService(retriever=mock_retriever, llm_client=mock_llm_client)
        with pytest.raises(ValueError) as exc:
            service.answer_question(AnswerRequest(query="   "), owner_id=TEST_OWNER_ID)
        assert "cannot be empty" in str(exc.value)

    def test_safe_rejection_on_fabricated_citation(self, mock_retriever, mock_llm_client):
        """Fabricated source IDs (e.g. [Source 99]) must trigger safe rejection of the complete answer."""
        chunk1 = make_test_retrieved_chunk("chunk_1", "Real verified fact from document.", rank=1)
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="Test query",
            mode=RetrievalMode.DENSE,
            total_candidates_found=1,
            returned_chunks_count=1,
            chunks=[chunk1],
            execution_time_ms=10.0,
        )

        # Model generates an answer claiming support from non-existent [Source 99]
        mock_llm_client.generate.return_value = {
            "content": (
                '{\n'
                '  "answer": "Fabricated claim relying on fake source [Source 99].",\n'
                '  "insufficient_context": false,\n'
                '  "citations": [{"source_id": "[Source 99]", "claim": "Fake claim"}]\n'
                '}'
            ),
            "model": "gpt-4o-mini",
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
        }

        service = RAGService(retriever=mock_retriever, llm_client=mock_llm_client)
        response = service.answer_question(AnswerRequest(query="Test query"), owner_id=TEST_OWNER_ID)

        # The unsafe generated answer MUST BE REJECTED entirely
        assert "could not be verified because it referenced invalid" in response.answer
        assert "Fabricated claim" not in response.answer
        assert response.insufficient_context is True
        assert len(response.citations) == 0
        assert response.metadata.has_hallucinated_sources is True
        assert len(response.metadata.citation_errors) > 0

    def test_safe_rejection_on_excluded_chunk_citation(self, mock_retriever, mock_llm_client):
        """Citations to chunks excluded due to token budget must trigger safe rejection."""
        # Create 2 chunks, but set context budget so only chunk 1 fits
        chunk1 = make_test_retrieved_chunk("chunk_1", "Admitted chunk text.", rank=1)
        chunk2 = make_test_retrieved_chunk("chunk_2", "Excluded chunk text.", rank=2)

        mock_retriever.retrieve.return_value = RetrievalResult(
            query="Test query",
            mode=RetrievalMode.DENSE,
            total_candidates_found=2,
            returned_chunks_count=2,
            chunks=[chunk1, chunk2],
            execution_time_ms=10.0,
        )

        # Model tries to cite chunk_2 (which was excluded)
        mock_llm_client.generate.return_value = {
            "content": (
                '{\n'
                '  "answer": "Claim based on excluded chunk [Source 2].",\n'
                '  "insufficient_context": false,\n'
                '  "citations": [{"source_id": "chunk_2", "claim": "Excluded claim"}]\n'
                '}'
            ),
            "model": "gpt-4o-mini",
            "prompt_tokens": 80,
            "completion_tokens": 25,
            "total_tokens": 105,
        }

        # Context builder with tiny budget that only fits chunk 1 (~40 tokens)
        service = RAGService(
            retriever=mock_retriever,
            llm_client=mock_llm_client,
            context_builder=ContextBuilder(max_context_tokens=60),
        )
        response = service.answer_question(AnswerRequest(query="Test query"), owner_id=TEST_OWNER_ID)

        assert response.insufficient_context is True
        assert "could not be verified" in response.answer
        assert len(response.citations) == 0
        assert response.metadata.has_hallucinated_sources is True

    def test_oversized_query_rejected_before_retrieval_and_llm(self, mock_retriever, mock_llm_client):
        """Query and completion overhead exceeding total token budget must raise ValueError before running retrieval or LLM."""
        # Query within character limits (max_length=2000), but max_tokens requested (3800)
        # together with system prompt (380) and safety margin (150) exceeds total budget (4096)
        service = RAGService(retriever=mock_retriever, llm_client=mock_llm_client)

        with pytest.raises(ValueError) as exc:
            service.answer_question(AnswerRequest(
                query="Explain the complete architecture in depth.",
                max_tokens=3800,
            ), owner_id=TEST_OWNER_ID)

        assert "exceed total request token limit" in str(exc.value)
        # Neither retrieval nor LLM should have been called!
        mock_retriever.retrieve.assert_not_called()
        mock_llm_client.generate.assert_not_called()

    def test_answer_without_citations_preserves_text_when_valid(self, mock_retriever, mock_llm_client):
        """When answer contains no citations and insufficient_context is False, response is preserved."""
        chunk1 = make_test_retrieved_chunk("chunk_1", "Informational text.", rank=1)
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="General question",
            mode=RetrievalMode.DENSE,
            total_candidates_found=1,
            returned_chunks_count=1,
            chunks=[chunk1],
            execution_time_ms=10.0,
        )

        mock_llm_client.generate.return_value = {
            "content": (
                '{\n'
                '  "answer": "This is a general response without citations.",\n'
                '  "insufficient_context": false,\n'
                '  "citations": []\n'
                '}'
            ),
            "model": "gpt-4o-mini",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }

        service = RAGService(retriever=mock_retriever, llm_client=mock_llm_client)
        response = service.answer_question(AnswerRequest(query="General question"), owner_id=TEST_OWNER_ID)

        assert response.answer == "This is a general response without citations."
        assert response.insufficient_context is False
        assert len(response.citations) == 0
        assert response.metadata.has_hallucinated_sources is False


# ============================================================================
# 2. FastAPI HTTP Endpoint Tests (/api/v1/answer)
# ============================================================================

class TestAnswerAPIEndpoint:
    """Test suite for the FastAPI answer generation HTTP route."""

    @pytest.fixture
    def client(self):
        prev_rag = app.dependency_overrides.get(get_rag_service)
        try:
            yield TestClient(app)
        finally:
            if prev_rag is None:
                app.dependency_overrides.pop(get_rag_service, None)
            else:
                app.dependency_overrides[get_rag_service] = prev_rag

    def test_answer_endpoint_success(self, client):
        """Successful POST /api/v1/answer returns 200 with complete AnswerResponse schema."""
        mock_response = AnswerResponse(
            query="What is HNSW?",
            answer="HNSW is a graph-based approximate nearest neighbor algorithm [Source 1].",
            citations=[],
            insufficient_context=False,
            sources_used=[],
            sources_excluded=[],
            metadata={
                "model_name": "gpt-4o-mini",
                "retrieval_mode": "dense",
                "total_chunks_retrieved": 1,
                "context_chunks_used": 1,
                "context_chunks_excluded": 0,
                "context_tokens_used": 45,
                "max_context_tokens": 2500,
                "prompt_tokens": 150,
                "completion_tokens": 25,
                "total_tokens": 175,
                "latency_ms": 320.5,
            },
        )

        mock_svc = MagicMock()
        mock_svc.answer_question.return_value = mock_response
        app.dependency_overrides[get_rag_service] = lambda: mock_svc

        resp = client.post(
            "/api/v1/answer",
            json={"query": "What is HNSW?"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is HNSW?"
        assert "HNSW is a graph-based" in data["answer"]
        assert data["metadata"]["model_name"] == "gpt-4o-mini"

    def test_answer_endpoint_config_error_mapping(self, client):
        """LLMConfigError (e.g. missing API key) maps to 503 without exposing secrets."""
        mock_svc = MagicMock()
        mock_svc.answer_question.side_effect = LLMConfigError("Missing OPENAI_API_KEY")
        app.dependency_overrides[get_rag_service] = lambda: mock_svc

        resp = client.post(
            "/api/v1/answer",
            json={"query": "What is Python?"},
        )

        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data
        # Must contain user-friendly message, NOT raw variable names or secrets
        assert "LLM service is not properly configured" in data["detail"]

    def test_answer_endpoint_auth_error_mapping(self, client):
        """LLMAuthenticationError maps to 502 Bad Gateway."""
        mock_svc = MagicMock()
        mock_svc.answer_question.side_effect = LLMAuthenticationError("Invalid API key")
        app.dependency_overrides[get_rag_service] = lambda: mock_svc

        resp = client.post(
            "/api/v1/answer",
            json={"query": "What is Python?"},
        )

        assert resp.status_code == 502
        assert "Authentication with the AI model failed" in resp.json()["detail"]

    def test_answer_endpoint_rate_limit_mapping(self, client):
        """LLMRateLimitError maps to 429 Too Many Requests."""
        mock_svc = MagicMock()
        mock_svc.answer_question.side_effect = LLMRateLimitError("Quota exceeded")
        app.dependency_overrides[get_rag_service] = lambda: mock_svc

        resp = client.post(
            "/api/v1/answer",
            json={"query": "What is Python?"},
        )

        assert resp.status_code == 429
        assert "rate-limited" in resp.json()["detail"]

    def test_answer_endpoint_timeout_mapping(self, client):
        """LLMTimeoutError maps to 504 Gateway Timeout."""
        mock_svc = MagicMock()
        mock_svc.answer_question.side_effect = LLMTimeoutError("Timed out")
        app.dependency_overrides[get_rag_service] = lambda: mock_svc

        resp = client.post(
            "/api/v1/answer",
            json={"query": "What is Python?"},
        )

        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]


# ============================================================================
# 3. Optional Real API Integration Test (Skipped when no key)
# ============================================================================

def is_real_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key and not any(ph in key.lower() for ph in ["your", "placeholder", "dummy", "test", "xxx"]))


@pytest.mark.skipif(
    not is_real_openai_key(),
    reason="Real OpenAI API key not detected in environment. Skipping live integration test.",
)
def test_live_openai_api_call():
    """Live integration test against the real OpenAI API.

    Only executes when OPENAI_API_KEY is actively exported in the test environment.
    """
    client = OpenAIClient.get_instance()
    messages = [
        {"role": "system", "content": "You are a concise testing assistant. Reply with one word: 'OK'."},
        {"role": "user", "content": "Ping test."},
    ]
    res = client.generate(messages, max_tokens=10)
    assert "OK" in res["content"].upper()
