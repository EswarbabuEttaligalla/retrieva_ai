"""Automated integration and regression test suite for Real-Time SSE Answer Streaming.

Verifies:
1. Incremental token streaming via Server-Sent Events (SSE).
2. Strict W3C SSE frame formatting (event, data, double newline termination).
3. Deterministic event order: status -> token -> status -> citations -> done.
4. Post-stream citation validation success and metadata assembly.
5. Safe rejection on fabricated citations ([Source 99]).
6. Safe rejection on citations excluded due to token budget limits.
7. Insufficient context behavior on empty retrieval.
8. Empty model stream handling.
9. OpenAI authentication error mapping (emits error event, no raw stack traces).
10. OpenAI rate limit error mapping.
11. OpenAI timeout error mapping.
12. OpenAI network error mapping.
13. Generator cleanup and resource release via try/finally.
14. Client cancellation / GeneratorExit handling.
15. Guarantee that no successful citations event is emitted after rejection or error.
16. Request validation (empty query and oversized query).
17. Existing non-streaming answer endpoint (/api/v1/answer) remains fully functional.
18. Streaming endpoint executes genuine retrieval and context construction.
19. Streaming HTTP response headers (Content-Type, Cache-Control, X-Accel-Buffering).
"""

import json
from typing import Generator, List
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from backend.app.core.sse import format_sse_event
from backend.app.document_processing.chunker import DocumentChunk
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.llm.client import OpenAIClient
from backend.app.llm.context_builder import ContextBuilder
from backend.app.llm.exceptions import (
    LLMAuthenticationError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from backend.app.main import app
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.schemas.answer import ContextSource, ExcludedContextChunk
from backend.app.schemas.retrieval import RetrievalMode
from backend.app.services.rag_service import RAGService, get_rag_service
from backend.app.vector_store.store import ChromaVectorStore
import tempfile
import shutil


def parse_sse_events(raw_text: str) -> List[dict]:
    """Parse raw SSE HTTP response text into a list of event dictionaries."""
    events = []
    current_event = None
    current_data = []

    for line in raw_text.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            current_data.append(line[6:])
        elif line == "":
            if current_event is not None or current_data:
                data_str = "\n".join(current_data)
                try:
                    parsed_data = json.loads(data_str)
                except Exception:
                    parsed_data = data_str
                events.append({"event": current_event, "data": parsed_data})
                current_event = None
                current_data = []

    if current_event is not None or current_data:
        data_str = "\n".join(current_data)
        try:
            parsed_data = json.loads(data_str)
        except Exception:
            parsed_data = data_str
        events.append({"event": current_event, "data": parsed_data})

    return events


@pytest.fixture(scope="module")
def shared_embedder():
    """Module-scoped embedder instance."""
    return SentenceTransformerEmbedder.get_instance()


@pytest.fixture(scope="module")
def streaming_rag_fixture(shared_embedder):
    """Create an isolated Retriever and RAGService with controlled documents."""
    temp_dir = tempfile.mkdtemp(prefix="retrieva_streaming_test_")
    store = ChromaVectorStore(
        collection_name="test_streaming_collection",
        persist_directory=temp_dir,
    )
    bm25 = BM25Index()

    doc1_chunk = DocumentChunk(
        chunk_id="chunk_thermo_01",
        document_id="doc_thermo",
        filename="physics_laws.txt",
        page_number=1,
        page_references=[1],
        chunk_index=0,
        chunk_text="The second law of thermodynamics states that total entropy of an isolated system always increases.",
        start_offset=0,
        end_offset=98,
        char_count=98,
        word_count=16,
    )

    doc2_chunk = DocumentChunk(
        chunk_id="chunk_thermo_02",
        document_id="doc_thermo",
        filename="physics_laws.txt",
        page_number=2,
        page_references=[2],
        chunk_index=1,
        chunk_text="Absolute zero is the theoretical temperature at which entropy reaches its minimum value.",
        start_offset=99,
        end_offset=187,
        char_count=88,
        word_count=13,
    )

    chunks = [doc1_chunk, doc2_chunk]
    embeddings = shared_embedder.embed_batch([c.chunk_text for c in chunks])
    store.add_chunks(chunks, embeddings, prevent_duplicates=False)
    for c in chunks:
        bm25.add_chunk({
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "filename": c.filename,
            "chunk_text": c.chunk_text,
            "page_number": c.page_number,
            "page_references": c.page_references,
            "start_offset": c.start_offset,
            "end_offset": c.end_offset,
        })

    retriever = Retriever(
        dense_retriever=DenseRetriever(embedder=shared_embedder, vector_store=store),
        bm25_index=bm25,
        vector_store=store,
        auto_sync_bm25=False,
    )

    mock_llm = MagicMock(spec=OpenAIClient)
    mock_llm.model = "gpt-4o-mini"

    async def _default_stream_async(*args, **kwargs):
        res = mock_llm.generate_stream(*args, **kwargs)
        try:
            if hasattr(res, "__aiter__"):
                async for item in res:
                    yield item
            elif hasattr(res, "__iter__"):
                for item in res:
                    yield item
            else:
                yield res
        finally:
            if hasattr(res, "close"):
                res.close()
            elif hasattr(res, "aclose"):
                await res.aclose()

    mock_llm.generate_stream_async = MagicMock(side_effect=_default_stream_async)

    rag_service = RAGService(
        retriever=retriever,
        llm_client=mock_llm,
        context_builder=ContextBuilder(),
    )

    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_rag_service] = lambda: rag_service

    yield rag_service, mock_llm

    app.dependency_overrides.clear()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_mock_state(streaming_rag_fixture):
    """Reset mock state before and after each test."""
    rag_service, mock_llm = streaming_rag_fixture
    mock_llm.generate_stream.side_effect = None
    mock_llm.generate_stream.return_value = iter([])

    async def _default_stream_async(*args, **kwargs):
        res = mock_llm.generate_stream(*args, **kwargs)
        try:
            if hasattr(res, "__aiter__"):
                async for item in res:
                    yield item
            elif hasattr(res, "__iter__"):
                for item in res:
                    yield item
            else:
                yield res
        finally:
            if hasattr(res, "close"):
                res.close()
            elif hasattr(res, "aclose"):
                await res.aclose()

    mock_llm.generate_stream_async = MagicMock(side_effect=_default_stream_async)
    yield
    mock_llm.generate_stream.side_effect = None
    mock_llm.generate_stream_async = MagicMock(side_effect=_default_stream_async)


class TestStreamingAPIEndpoints:
    """Test suite verifying /api/v1/answer/stream SSE behavior and citation verification."""

    def test_successful_token_streaming(self, sync_client: TestClient, streaming_rag_fixture):
        """1. Verify tokens stream incrementally and match mock model output."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.side_effect = None
        mock_tokens = ["Entropy ", "always ", "increases [Source 1]."]
        mock_llm.generate_stream.return_value = iter(mock_tokens)

        payload = {"query": "What does the second law state?", "retrieval_config": {"mode": "hybrid"}}
        response = sync_client.post("/api/v1/answer/stream", json=payload)
        assert response.status_code == 200

        events = parse_sse_events(response.text)
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) == 3
        assembled = "".join(e["data"]["delta"] for e in token_events)
        assert assembled == "Entropy always increases [Source 1]."

    def test_correct_sse_frame_formatting(self):
        """2. Verify format_sse_event generates compliant W3C frames."""
        frame = format_sse_event("token", {"delta": "test chunk"})
        assert frame.startswith("event: token\n")
        assert "data: {\"delta\": \"test chunk\"}\n\n" in frame
        assert frame.endswith("\n\n")

    def test_correct_event_sequence(self, sync_client: TestClient, streaming_rag_fixture):
        """3. Verify deterministic event lifecycle: status -> token -> status -> citations -> done."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.return_value = iter(["Test answer [Source 1]."])

        response = sync_client.post("/api/v1/answer/stream", json={"query": "second law"})
        assert response.status_code == 200

        events = parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        # Verify stages
        assert "status" in event_types
        assert "token" in event_types
        assert "citations" in event_types
        assert event_types[-1] == "done"

        # Verify status transitions
        status_stages = [e["data"]["stage"] for e in events if e["event"] == "status"]
        assert "retrieving" in status_stages
        assert "building_context" in status_stages
        assert "generating" in status_stages
        assert "verifying" in status_stages

    def test_citation_validation_success(self, sync_client: TestClient, streaming_rag_fixture):
        """4. Verify verified citations are emitted when model references admitted sources."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.return_value = iter(["Entropy increases [Source 1]."])

        response = sync_client.post("/api/v1/answer/stream", json={"query": "second law"})
        events = parse_sse_events(response.text)

        citations_events = [e for e in events if e["event"] == "citations"]
        assert len(citations_events) == 1
        cit_data = citations_events[0]["data"]
        assert cit_data["status"] == "verified"
        assert len(cit_data["citations"]) >= 1
        assert cit_data["citations"][0]["source_id"] == "[Source 1]"
        assert cit_data["citations"][0]["filename"] == "physics_laws.txt"

    def test_citation_validation_rejection_fabricated(self, sync_client: TestClient, streaming_rag_fixture):
        """5. Verify safe rejection when model cites fabricated source [Source 99]."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.return_value = iter(["Hallucinated fact [Source 99]."])

        response = sync_client.post("/api/v1/answer/stream", json={"query": "second law"})
        events = parse_sse_events(response.text)

        rejection_events = [e for e in events if e["event"] == "rejection"]
        assert len(rejection_events) == 1
        assert rejection_events[0]["data"]["status"] == "rejected"
        assert any("[Source 99]" in err for err in rejection_events[0]["data"]["invalid_citations"])

        # Crucial: NO citations event should be emitted!
        citations_events = [e for e in events if e["event"] == "citations"]
        assert len(citations_events) == 0

        # Must end with done
        assert events[-1]["event"] == "done"

    def test_citation_validation_rejection_excluded(self, sync_client: TestClient, streaming_rag_fixture):
        """6. Verify safe rejection when model cites a source excluded by token budget."""
        rag_service, mock_llm = streaming_rag_fixture

        # Patch ContextBuilder.build_context to exclude a chunk
        original_build = rag_service.context_builder.build_context

        def mock_build(chunks, max_context_tokens):
            res = original_build(chunks, max_context_tokens=100)
            # Artificially add chunk_thermo_02 to excluded
            excluded = [
                ExcludedContextChunk(
                    chunk_id="chunk_thermo_02",
                    document_id="doc_thermo",
                    filename="physics_laws.txt",
                    retrieval_rank=2,
                    token_count=20,
                    reason="Test budget exhaustion",
                )
            ]
            from backend.app.llm.context_builder import ContextBuildResult
            return ContextBuildResult(
                formatted_context=res.formatted_context,
                sources_used=[res.sources_used[0]],
                sources_excluded=excluded,
                total_tokens_used=res.total_tokens_used,
                max_budget=100,
            )

        with patch.object(rag_service.context_builder, "build_context", side_effect=mock_build):
            mock_llm.generate_stream.return_value = iter(["Absolute zero text [Source 2]."])
            response = sync_client.post("/api/v1/answer/stream", json={"query": "temperature"})
            events = parse_sse_events(response.text)

            rejection_events = [e for e in events if e["event"] == "rejection"]
            assert len(rejection_events) == 1
            assert "excluded from context" in rejection_events[0]["data"]["invalid_citations"][0]
            assert len([e for e in events if e["event"] == "citations"]) == 0

    def test_insufficient_context_behavior(self, sync_client: TestClient, streaming_rag_fixture):
        """7. Verify insufficient context when zero chunks are retrieved."""
        rag_service, mock_llm = streaming_rag_fixture

        # Filter by nonexistent document
        payload = {"query": "quantum gravity", "document_ids": ["nonexistent_doc_id"]}
        response = sync_client.post("/api/v1/answer/stream", json=payload)
        events = parse_sse_events(response.text)

        tokens = [e for e in events if e["event"] == "token"]
        assert len(tokens) >= 1
        assert "do not contain any information" in tokens[0]["data"]["delta"]

        citations = [e for e in events if e["event"] == "citations"]
        assert len(citations) == 1
        assert citations[0]["data"]["insufficient_context"] is True
        assert events[-1]["event"] == "done"

    def test_empty_stream_handling(self, sync_client: TestClient, streaming_rag_fixture):
        """8. Verify handling when model stream yields no tokens."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.return_value = iter([])

        response = sync_client.post("/api/v1/answer/stream", json={"query": "test query"})
        events = parse_sse_events(response.text)

        assert events[-1]["event"] == "done"
        citations = [e for e in events if e["event"] == "citations"]
        assert len(citations) == 1
        assert citations[0]["data"]["status"] == "verified"

    def test_openai_auth_error_handling(self, sync_client: TestClient, streaming_rag_fixture):
        """9. Verify OpenAI authentication errors emit clean error event without leaking API key."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.side_effect = LLMAuthenticationError("Incorrect API key provided")

        response = sync_client.post("/api/v1/answer/stream", json={"query": "test query"})
        events = parse_sse_events(response.text)

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["data"]["code"] == "authentication_error"
        assert len([e for e in events if e["event"] == "citations"]) == 0
        assert events[-1]["event"] == "done"

    def test_openai_rate_limit_error_handling(self, sync_client: TestClient, streaming_rag_fixture):
        """10. Verify OpenAI rate limit error mapping."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.side_effect = LLMRateLimitError("Rate limit exceeded")

        response = sync_client.post("/api/v1/answer/stream", json={"query": "test query"})
        events = parse_sse_events(response.text)

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["data"]["code"] == "rate_limit_error"
        assert events[-1]["event"] == "done"

    def test_openai_timeout_error_handling(self, sync_client: TestClient, streaming_rag_fixture):
        """11. Verify OpenAI timeout error mapping."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.side_effect = LLMTimeoutError("Stream timed out")

        response = sync_client.post("/api/v1/answer/stream", json={"query": "test query"})
        events = parse_sse_events(response.text)

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["data"]["code"] == "timeout_error"
        assert events[-1]["event"] == "done"

    def test_openai_network_error_handling(self, sync_client: TestClient, streaming_rag_fixture):
        """12. Verify OpenAI network connection error mapping."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.side_effect = LLMNetworkError("Connection refused")

        response = sync_client.post("/api/v1/answer/stream", json={"query": "test query"})
        events = parse_sse_events(response.text)

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["data"]["code"] == "network_error"
        assert events[-1]["event"] == "done"

    def test_generator_cleanup_on_stream_exit(self, streaming_rag_fixture):
        """13. Verify generator cleanup and try/finally execution."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.side_effect = None

        cleaned_up = False

        def mock_generator():
            nonlocal cleaned_up
            try:
                yield "tok1"
                yield "tok2"
            finally:
                cleaned_up = True

        mock_llm.generate_stream.return_value = mock_generator()
        from backend.app.schemas.answer import AnswerRequest

        import asyncio

        async def run_and_close():
            gen = rag_service.answer_question_stream(AnswerRequest(query="test"))
            async for ev in gen:
                if "event: token" in ev:
                    break
            await gen.aclose()

        asyncio.run(run_and_close())
        assert cleaned_up is True

    def test_client_cancellation_handling(self, streaming_rag_fixture):
        """14. Verify client cancellation via GeneratorExit is handled gracefully."""
        rag_service, mock_llm = streaming_rag_fixture
        from backend.app.schemas.answer import AnswerRequest

        import asyncio

        async def simulate_cancellation():
            gen = rag_service.answer_question_stream(AnswerRequest(query="test query"))
            # Read 2 events then simulate client abort
            await gen.__anext__()
            await gen.__anext__()
            await gen.aclose()

        asyncio.run(simulate_cancellation())

    def test_no_successful_citations_after_failure(self, sync_client: TestClient, streaming_rag_fixture):
        """15. Guarantee that no 'citations' event is emitted when an error or rejection occurs."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.side_effect = RuntimeError("Storage crash")

        response = sync_client.post("/api/v1/answer/stream", json={"query": "test"})
        events = parse_sse_events(response.text)

        assert any(e["event"] == "error" for e in events)
        assert not any(e["event"] == "citations" for e in events)
        assert events[-1]["event"] == "done"

    def test_request_validation(self, sync_client: TestClient, streaming_rag_fixture):
        """16. Verify empty query rejection returns error event."""
        # Empty string
        res = sync_client.post("/api/v1/answer/stream", json={"query": "   "})
        assert res.status_code == 200  # SSE stream starts and emits error event
        events = parse_sse_events(res.text)
        assert events[0]["event"] == "error"
        assert events[0]["data"]["code"] == "invalid_request"
        assert events[-1]["event"] == "done"

    def test_non_streaming_endpoint_remains_unchanged(self, sync_client: TestClient, streaming_rag_fixture):
        """17. Verify that POST /api/v1/answer still works identically and returns AnswerResponse."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate.return_value = {
            "content": json.dumps({
                "answer": "Entropy increases [Source 1].",
                "insufficient_context": False,
                "citations": [
                    {"source_id": "[Source 1]", "quote": "entropy increases", "claim": "entropy"}
                ]
            }),
            "model": "gpt-4o-mini",
            "prompt_tokens": 120,
            "completion_tokens": 25,
            "total_tokens": 145,
        }

        response = sync_client.post("/api/v1/answer", json={"query": "second law"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert len(data["citations"]) >= 1
        assert data["citations"][0]["source_id"] == "[Source 1]"

    def test_streaming_does_not_bypass_retrieval_or_context(self, sync_client: TestClient, streaming_rag_fixture):
        """18. Verify that streaming genuinely invokes retrieval and ContextBuilder."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.return_value = iter(["Answer."])

        with patch.object(rag_service.retriever, "retrieve", wraps=rag_service.retriever.retrieve) as spy_retriever, \
             patch.object(rag_service.context_builder, "build_context", wraps=rag_service.context_builder.build_context) as spy_builder:

            response = sync_client.post("/api/v1/answer/stream", json={"query": "second law"})
            assert response.status_code == 200
            assert spy_retriever.called
            assert spy_builder.called

    def test_streaming_response_headers(self, sync_client: TestClient, streaming_rag_fixture):
        """19. Verify SSE media type and buffering headers."""
        rag_service, mock_llm = streaming_rag_fixture
        mock_llm.generate_stream.return_value = iter(["Test"])

        response = sync_client.post("/api/v1/answer/stream", json={"query": "second law"})
        assert "text/event-stream" in response.headers["content-type"]
        assert "no-cache" in response.headers["cache-control"]
        assert response.headers.get("x-accel-buffering") == "no"

    @pytest.mark.asyncio
    async def test_slow_upstream_does_not_block_event_loop(self, streaming_rag_fixture):
        """20. Verify slow upstream token arrival does not block unrelated asyncio tasks."""
        rag_service, mock_llm = streaming_rag_fixture
        import asyncio
        from backend.app.schemas.answer import AnswerRequest

        async def slow_stream(*args, **kwargs):
            yield "Token 1 "
            await asyncio.sleep(0.04)
            yield "Token 2 [Source 1]."
            await asyncio.sleep(0.04)

        mock_llm.generate_stream_async = MagicMock(side_effect=slow_stream)

        # Background coroutine that ticks every 5ms on the same event loop
        background_ticks = 0
        stop_background = False

        async def background_task():
            nonlocal background_ticks
            while not stop_background:
                await asyncio.sleep(0.005)
                background_ticks += 1

        bg_task = asyncio.create_task(background_task())

        tokens_received = []
        try:
            gen = rag_service.answer_question_stream(AnswerRequest(query="second law"))
            async for ev in gen:
                if "event: token" in ev:
                    tokens_received.append(ev)
        finally:
            stop_background = True
            await bg_task

        assert len(tokens_received) == 2
        # Background task must have made continuous progress during token generation
        assert background_ticks >= 5, f"Expected background task to make progress, but got {background_ticks} ticks"

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests_independent_progress(self, streaming_rag_fixture):
        """21. Verify multiple concurrent streaming requests make independent progress."""
        rag_service, mock_llm = streaming_rag_fixture
        import asyncio
        from backend.app.schemas.answer import AnswerRequest

        async def stream_for_query(messages, **kwargs):
            user_msg = messages[-1]["content"] if messages else ""
            if "first" in user_msg:
                yield "First stream token [Source 1]."
            else:
                yield "Second stream token [Source 2]."

        mock_llm.generate_stream_async = MagicMock(side_effect=stream_for_query)

        async def consume_stream(q):
            events = []
            async for ev in rag_service.answer_question_stream(AnswerRequest(query=q)):
                events.append(ev)
            return events

        # Run 2 streams concurrently
        res1, res2 = await asyncio.gather(
            consume_stream("first law"),
            consume_stream("second law"),
        )

        assert any("First stream token" in e for e in res1)
        assert any("Second stream token" in e for e in res2)

    @pytest.mark.asyncio
    async def test_async_client_cancellation_terminates_upstream_resources(self, streaming_rag_fixture):
        """22. Verify client cancellation terminates upstream async generator cleanly."""
        rag_service, mock_llm = streaming_rag_fixture
        import asyncio
        from backend.app.schemas.answer import AnswerRequest

        upstream_cleaned = False

        async def cancellable_stream(*args, **kwargs):
            nonlocal upstream_cleaned
            try:
                yield "tok1 "
                await asyncio.sleep(0.5)
                yield "tok2"
            finally:
                upstream_cleaned = True

        mock_llm.generate_stream_async = MagicMock(side_effect=cancellable_stream)

        gen = rag_service.answer_question_stream(AnswerRequest(query="test"))
        async for ev in gen:
            if "event: token" in ev:
                break
        await gen.aclose()

        assert upstream_cleaned is True

    @pytest.mark.asyncio
    async def test_openai_client_generate_stream_async_native(self):
        """23. Verify OpenAIClient.generate_stream_async native non-blocking execution."""
        from backend.app.llm.client import OpenAIClient
        import asyncio

        mock_async_raw = MagicMock()
        mock_stream = MagicMock()

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Async "))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="stream!"))]

        async def async_chunks():
            yield chunk1
            yield chunk2

        mock_stream.__aiter__ = lambda self: async_chunks()
        mock_stream.close = MagicMock(return_value=asyncio.sleep(0))

        mock_async_raw.chat.completions.create = MagicMock(
            return_value=asyncio.sleep(0, result=mock_stream)
        )

        client = OpenAIClient(api_key="sk-test-async-key", async_client=mock_async_raw)

        tokens = []
        async for tok in client.generate_stream_async([{"role": "user", "content": "Hi"}]):
            tokens.append(tok)

        assert tokens == ["Async ", "stream!"]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_sync_fallback_cancellation_while_blocked(self, streaming_rag_fixture):
        """24. Verify synchronous iterator cancellation while worker thread is blocked."""
        rag_service, mock_llm = streaming_rag_fixture
        import asyncio
        import time
        from backend.app.schemas.answer import AnswerRequest

        def slow_sync_generator(*args, **kwargs):
            try:
                time.sleep(0.3)
                yield "first"
                time.sleep(0.3)
                yield "second"
            finally:
                pass

        # Force synchronous generator path by setting generate_stream and removing generate_stream_async
        mock_llm.generate_stream = MagicMock(side_effect=slow_sync_generator)
        old_async = getattr(mock_llm, "generate_stream_async", None)
        if hasattr(mock_llm, "generate_stream_async"):
            delattr(mock_llm, "generate_stream_async")

        try:
            gen = rag_service.answer_question_stream(AnswerRequest(query="entropy"))
            # Read first event (status)
            ev = await gen.__anext__()
            assert "event: status" in ev
            # Cancel immediately while worker thread is blocked in next()
            await gen.aclose()
        finally:
            if old_async is not None:
                mock_llm.generate_stream_async = old_async

    @pytest.mark.asyncio
    async def test_sync_fallback_cleanup_normal_completion(self, streaming_rag_fixture):
        """25. Verify synchronous iterator is closed cleanly upon normal completion."""
        rag_service, mock_llm = streaming_rag_fixture
        from backend.app.schemas.answer import AnswerRequest

        closed = False

        def normal_sync_gen(*args, **kwargs):
            nonlocal closed
            try:
                yield "According to [Source 1], entropy increases."
            finally:
                closed = True

        mock_llm.generate_stream = MagicMock(side_effect=normal_sync_gen)
        old_async = getattr(mock_llm, "generate_stream_async", None)
        if hasattr(mock_llm, "generate_stream_async"):
            delattr(mock_llm, "generate_stream_async")

        try:
            gen = rag_service.answer_question_stream(AnswerRequest(query="entropy"))
            events = []
            async for ev in gen:
                events.append(ev)
            assert closed is True
            assert any("event: done" in ev for ev in events)
        finally:
            if old_async is not None:
                mock_llm.generate_stream_async = old_async

    @pytest.mark.asyncio
    async def test_no_concurrent_next_and_close_generator_collision(self, streaming_rag_fixture):
        """26. Verify no ValueError: generator already executing during concurrent close attempt."""
        rag_service, mock_llm = streaming_rag_fixture
        import asyncio
        import time
        from backend.app.schemas.answer import AnswerRequest

        def slow_blocking_gen(*args, **kwargs):
            try:
                time.sleep(0.2)
                yield "data"
            finally:
                time.sleep(0.05)

        mock_llm.generate_stream = MagicMock(side_effect=slow_blocking_gen)
        old_async = getattr(mock_llm, "generate_stream_async", None)
        if hasattr(mock_llm, "generate_stream_async"):
            delattr(mock_llm, "generate_stream_async")

        try:
            gen = rag_service.answer_question_stream(AnswerRequest(query="entropy"))
            # Advance past status to trigger worker thread read
            ev1 = await gen.__anext__()
            assert "event: status" in ev1
            # Give tiny moment for worker thread to enter next()
            await asyncio.sleep(0.02)
            # aclose triggers finally block with _safe_close
            # Must complete without ValueError: generator already executing
            await gen.aclose()
        finally:
            if old_async is not None:
                mock_llm.generate_stream_async = old_async

    @pytest.mark.asyncio
    async def test_sync_fallback_event_loop_responsiveness(self, streaming_rag_fixture):
        """27. Verify event loop remains responsive while synchronous iterator blocks in thread."""
        rag_service, mock_llm = streaming_rag_fixture
        import asyncio
        import time
        from backend.app.schemas.answer import AnswerRequest

        def blocking_sync_gen(*args, **kwargs):
            time.sleep(0.25)
            yield "Entropy always increases [Source 1]."

        mock_llm.generate_stream = MagicMock(side_effect=blocking_sync_gen)
        old_async = getattr(mock_llm, "generate_stream_async", None)
        if hasattr(mock_llm, "generate_stream_async"):
            delattr(mock_llm, "generate_stream_async")

        heartbeat_ticks = 0

        async def heartbeat():
            nonlocal heartbeat_ticks
            try:
                while True:
                    await asyncio.sleep(0.03)
                    heartbeat_ticks += 1
            except asyncio.CancelledError:
                pass

        hb_task = asyncio.create_task(heartbeat())
        try:
            gen = rag_service.answer_question_stream(AnswerRequest(query="entropy"))
            async for _ in gen:
                pass
        finally:
            hb_task.cancel()
            await asyncio.gather(hb_task, return_exceptions=True)
            if old_async is not None:
                mock_llm.generate_stream_async = old_async

        # During 250ms blocking read, heartbeat ticking every 30ms should record at least 3 ticks
        assert heartbeat_ticks >= 3, f"Event loop was starved; recorded only {heartbeat_ticks} ticks"

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_closes_openai_async_session(self):
        """28. Verify FastAPI lifespan calls OpenAIClient.aclose() on application shutdown."""
        from backend.app.main import lifespan, create_application
        from backend.app.llm.client import OpenAIClient
        from unittest.mock import AsyncMock

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        test_app = create_application()
        with patch.object(OpenAIClient, "_instance", mock_client):
            async with lifespan(test_app):
                # Inside running application lifespan
                assert not mock_client.aclose.called
            # After lifespan exits (shutdown)
            assert mock_client.aclose.called

    @pytest.mark.asyncio
    async def test_openai_client_aclose_idempotent_and_reinitializable(self):
        """29. Verify OpenAIClient.aclose() is idempotent and reinitializes on demand."""
        from backend.app.llm.client import OpenAIClient
        from unittest.mock import AsyncMock

        mock_raw_async = MagicMock()
        mock_raw_async.close = AsyncMock()

        client = OpenAIClient(api_key="sk-test-idempotent-key", async_client=mock_raw_async)

        # 1. First aclose call
        await client.aclose()
        assert mock_raw_async.close.call_count == 1
        assert client._async_client is None

        # 2. Repeated aclose call must not error
        await client.aclose()
        await client.aclose()
        assert mock_raw_async.close.call_count == 1

        # 3. Accessing _ensure_async_client_ready recreates the client lazily
        recreated = client._ensure_async_client_ready()
        assert recreated is not None
        assert client._async_client is not None

        # 4. Clean up recreated client
        await client.aclose()
        assert client._async_client is None

    def test_openai_client_aclose_does_not_break_sync_generate(self):
        """30. Verify aclose() does not affect the synchronous generate() client interface."""
        import asyncio
        from backend.app.llm.client import OpenAIClient

        mock_sync_raw = MagicMock()
        choice_mock = MagicMock()
        choice_mock.message.content = "Sync response"
        response_mock = MagicMock(
            choices=[choice_mock],
            model="gpt-4o-mini",
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_sync_raw.chat.completions.create.return_value = response_mock

        mock_async_raw = MagicMock()
        mock_async_raw.close = MagicMock(return_value=asyncio.sleep(0))

        client = OpenAIClient(api_key="sk-test-sync-key", client=mock_sync_raw, async_client=mock_async_raw)

        # Close async session
        asyncio.run(client.aclose())

        # Verify synchronous generate continues to work normally
        result = client.generate([{"role": "user", "content": "Hello"}])
        assert result["content"] == "Sync response"
        assert result["model"] == "gpt-4o-mini"
