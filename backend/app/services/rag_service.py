"""RAG Pipeline Orchestration Service for Retrieva AI.

Connects the Phase 4 Retrieval Engine with Phase 5 Context Construction,
Prompt Engineering, OpenAI Inference, and Citation Validation.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.core.sse import format_sse_event
from backend.app.llm.citation_validator import CitationValidator
from backend.app.llm.client import OpenAIClient
from backend.app.llm.context_builder import ContextBuilder
from backend.app.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from backend.app.llm.prompt_builder import PromptBuilder
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.schemas.answer import (
    AnswerMetadata,
    AnswerRequest,
    AnswerResponse,
    SourceCitation,
)
from backend.app.schemas.retrieval import RetrievalConfiguration, RetrievalRequest

logger = get_logger("services.rag_service")


class RAGService:
    """Production orchestrator executing the 10-step RAG answer generation lifecycle.

    Guarantees:
    - Step-by-step pipeline execution from query validation to verified citations.
    - Full grounding verification without fabricating unsupported claims.
    - Safe error containment preventing secret leakage or cryptic stack traces.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[OpenAIClient] = None,
        context_builder: Optional[ContextBuilder] = None,
        conversation_service: Optional[Any] = None,
        lock_manager: Optional[Any] = None,
    ):
        """Initialize the RAG service with required component engines."""
        self.retriever = retriever or get_retriever()
        self.llm_client = llm_client or OpenAIClient.get_instance()
        self.context_builder = context_builder or ContextBuilder()
        self._conversation_service = conversation_service
        from backend.app.conversations.locks import get_conversation_lock_manager
        self._lock_manager = lock_manager or get_conversation_lock_manager()

    @property
    def lock_manager(self):
        """Return the ConversationLockManager instance."""
        return self._lock_manager

    def answer_question(self, request: AnswerRequest) -> AnswerResponse:
        """Execute the complete end-to-end RAG answer generation pipeline.

        Pipeline Stages:
        1. Validate the user question.
        2. Retrieve relevant document chunks via Phase 4 Retrieval Engine.
        3. Check for empty retrieval and trigger safe fallback if empty.
        4. Build bounded context with strict token budgeting and stable IDs.
        5. Construct grounded prompt with prompt-injection defense rules.
        6. Dispatch request to OpenAI client.
        7. Parse structured response (JSON or robust fallback).
        8. Validate citations against admitted context chunks.
        9. Assemble AnswerMetadata (token economics, latency, model info).
        10. Return verified AnswerResponse.

        Args:
            request: Validated AnswerRequest model.

        Returns:
            AnswerResponse model with grounded text, citations, and metadata.

        Raises:
            ValueError: If query is invalid.
            LLMError: If LLM service fails.
        """
        start_time = time.perf_counter()

        # Step 1: Validate User Question & Pre-flight Request Token Budgeting
        raw_query = (request.query or "").strip()
        if not raw_query:
            raise ValueError("Query string cannot be empty or whitespace-only.")

        query = raw_query
        rewritten_query = raw_query
        conversation_service = None

        if request.conversation_id:
            from backend.app.conversations.service import get_conversation_service
            conversation_service = self._conversation_service or get_conversation_service()
            rewritten_query, _ = conversation_service.prepare_conversational_turn_sync(
                conversation_id=request.conversation_id,
                raw_query=raw_query,
            )
            conversation_service.record_user_message(
                conversation_id=request.conversation_id,
                raw_query=raw_query,
                rewritten_query=rewritten_query,
            )
            query = rewritten_query

        settings = get_settings()
        reserved_completion_tokens = request.max_tokens or settings.DEFAULT_MAX_COMPLETION_TOKENS
        total_request_limit = settings.TOTAL_REQUEST_TOKEN_LIMIT

        # Calculate dynamic allowed context budget before retrieval/LLM
        # This will raise ValueError if question + base prompt exceeds total budget
        budget_report = self.context_builder.calculate_request_budget_report(
            query=query,
            reserved_completion_tokens=reserved_completion_tokens,
            model_name=self.llm_client.model,
            total_request_limit=total_request_limit,
            max_context_limit=self.context_builder.max_context_tokens,
        )
        allowed_context_tokens = budget_report.allowed_context_budget

        logger.info(
            "Executing RAG answer generation for query: '%s'... (dynamic context budget: %d tokens, model limit: %d)",
            query[:60],
            allowed_context_tokens,
            budget_report.model_context_limit,
        )

        # Step 2: Retrieve Relevant Document Chunks via Phase 4 Retriever
        retrieval_config = request.retrieval_config or RetrievalConfiguration()
        if not retrieval_config.document_id_filter and request.document_ids:
            retrieval_config.document_id_filter = request.document_ids[0]

        retrieval_req = RetrievalRequest(
            query=query,
            document_ids=request.document_ids,
            config=retrieval_config,
        )
        retrieval_result = self.retriever.retrieve(retrieval_req)
        retrieved_chunks = retrieval_result.chunks

        logger.info(
            "Retrieved %d candidate chunks (mode=%s)",
            len(retrieved_chunks),
            retrieval_result.mode.value,
        )

        # Step 3: Handle Empty Retrieval Result
        if not retrieved_chunks:
            logger.warning("Retrieval returned 0 chunks for query '%s'", query[:60])
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            fallback_ans = "The available documents do not contain any information relevant to this question."
            if conversation_service and request.conversation_id:
                conversation_service.record_assistant_message(
                    conversation_id=request.conversation_id,
                    content=fallback_ans,
                    metadata={"insufficient_context": True, "citations": []},
                )
            return AnswerResponse(
                query=raw_query,
                answer=fallback_ans,
                conversation_id=request.conversation_id,
                rewritten_query=rewritten_query if rewritten_query != raw_query else None,
                citations=[],
                insufficient_context=True,
                sources_used=[],
                sources_excluded=[],
                metadata=AnswerMetadata(
                    model_name=self.llm_client.model,
                    retrieval_mode=retrieval_result.mode.value,
                    total_chunks_retrieved=0,
                    context_chunks_used=0,
                    context_chunks_excluded=0,
                    context_tokens_used=0,
                    max_context_tokens=allowed_context_tokens,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    request_tokens_budget=budget_report.model_context_limit,
                    has_hallucinated_sources=False,
                    citation_errors=[],
                    latency_ms=round(latency_ms, 2),
                ),
            )

        # Step 4: Build Bounded Context with Strict Token Accounting
        context_result = self.context_builder.build_context(
            retrieved_chunks,
            max_context_tokens=allowed_context_tokens,
        )

        if not context_result.sources_used:
            logger.warning("No chunks fit into context budget for query '%s'", query[:60])
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            fallback_ans = "The available document context is too large to process within the allocated token budget."
            if conversation_service and request.conversation_id:
                conversation_service.record_assistant_message(
                    conversation_id=request.conversation_id,
                    content=fallback_ans,
                    metadata={"insufficient_context": True, "citations": []},
                )
            return AnswerResponse(
                query=raw_query,
                answer=fallback_ans,
                conversation_id=request.conversation_id,
                rewritten_query=rewritten_query if rewritten_query != raw_query else None,
                citations=[],
                insufficient_context=True,
                sources_used=[],
                sources_excluded=context_result.sources_excluded,
                metadata=AnswerMetadata(
                    model_name=self.llm_client.model,
                    retrieval_mode=retrieval_result.mode.value,
                    total_chunks_retrieved=len(retrieved_chunks),
                    context_chunks_used=0,
                    context_chunks_excluded=len(context_result.sources_excluded),
                    context_tokens_used=0,
                    max_context_tokens=context_result.max_budget,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    request_tokens_budget=budget_report.model_context_limit,
                    has_hallucinated_sources=False,
                    citation_errors=[],
                    latency_ms=round(latency_ms, 2),
                ),
            )

        # Step 5: Construct Grounded Prompt with Injection Defense
        messages = PromptBuilder.build_messages(
            query=query,
            formatted_context=context_result.formatted_context,
        )

        # Step 6: Dispatch Request to OpenAI Client
        llm_output = self.llm_client.generate(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        raw_content = llm_output["content"]

        # Step 7: Parse Structured Response (JSON or Prose Fallback)
        answer_text, insufficient_context, raw_citations = self._parse_model_output(raw_content)

        # Step 8: Validate Citations Against Admitted Context Chunks
        validation_result = CitationValidator.validate_citations(
            raw_citations=raw_citations,
            answer_text=answer_text,
            sources_used=context_result.sources_used,
            sources_excluded=context_result.sources_excluded,
        )

        # Enforce Safe Failure Policy for Invalid Citations
        if validation_result.has_hallucinated_sources:
            logger.warning(
                "Safe rejection triggered: LLM produced invalid or fabricated citations: %s",
                validation_result.invalid_citations,
            )
            answer_text = (
                "The generated answer could not be verified because it referenced invalid, "
                "fabricated, or excluded source citations. The available documents do not "
                "substantiate this response."
            )
            insufficient_context = True
            validated_citations = []
            if conversation_service and request.conversation_id:
                conversation_service.record_assistant_message(
                    conversation_id=request.conversation_id,
                    content=answer_text,
                    metadata={
                        "rejected": True,
                        "reason": "Invalid or fabricated citations",
                        "invalid_citations": validation_result.invalid_citations,
                    },
                )
        else:
            validated_citations = validation_result.valid_citations
            if conversation_service and request.conversation_id:
                conversation_service.record_assistant_message(
                    conversation_id=request.conversation_id,
                    content=answer_text,
                    metadata={
                        "citations": [c.model_dump() for c in validated_citations],
                        "insufficient_context": insufficient_context,
                        "retrieval_mode": retrieval_result.mode.value,
                    },
                )

        # Warning when no citations are provided for non-insufficient response
        if not validated_citations and not insufficient_context and context_result.sources_used:
            logger.warning("Answer generated without explicit source citations despite non-empty context.")

        # Step 9: Assemble Diagnostic Metadata
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        metadata = AnswerMetadata(
            model_name=llm_output.get("model", self.llm_client.model),
            retrieval_mode=retrieval_result.mode.value,
            total_chunks_retrieved=len(retrieved_chunks),
            context_chunks_used=len(context_result.sources_used),
            context_chunks_excluded=len(context_result.sources_excluded),
            context_tokens_used=context_result.total_tokens_used,
            max_context_tokens=context_result.max_budget,
            prompt_tokens=llm_output.get("prompt_tokens"),
            completion_tokens=llm_output.get("completion_tokens"),
            total_tokens=llm_output.get("total_tokens"),
            request_tokens_budget=budget_report.model_context_limit,
            has_hallucinated_sources=validation_result.has_hallucinated_sources,
            citation_errors=validation_result.invalid_citations,
            latency_ms=round(latency_ms, 2),
        )

        # Step 10: Return Final AnswerResponse
        return AnswerResponse(
            query=raw_query,
            answer=answer_text,
            conversation_id=request.conversation_id,
            rewritten_query=rewritten_query if rewritten_query != raw_query else None,
            citations=validated_citations,
            insufficient_context=insufficient_context,
            sources_used=context_result.sources_used,
            sources_excluded=context_result.sources_excluded,
            metadata=metadata,
        )

    def _parse_model_output(self, content: str) -> tuple[str, bool, List[Dict[str, Any]]]:
        """Parse raw completion output into answer text, insufficient_context flag, and citations."""
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                answer = parsed.get("answer", "").strip() or cleaned
                insufficient = bool(parsed.get("insufficient_context", False))
                raw_cits = parsed.get("citations", [])
                if not isinstance(raw_cits, list):
                    raw_cits = []
                return answer, insufficient, raw_cits
        except json.JSONDecodeError:
            pass

        # Fallback for plain prose completion
        lower_content = content.lower()
        insufficient_phrases = [
            "not contain sufficient information",
            "does not provide sufficient information",
            "insufficient information to answer",
            "do not contain enough information",
            "cannot answer this question based on the provided",
        ]
        insufficient = any(phrase in lower_content for phrase in insufficient_phrases)
        return content.strip(), insufficient, []

    async def answer_question_stream(self, request: AnswerRequest) -> AsyncIterator[str]:
        """Execute the real-time RAG streaming pipeline emitting SSE events.

        Stages:
        1. Request validation and pre-flight token budgeting.
        2. Status: retrieving -> retrieve document chunks.
        3. Optional status: reranking -> cross-encoder candidate scoring.
        4. Status: building_context -> assemble bounded context.
        5. Status: generating -> stream tokens from OpenAI model.
        6. Status: verifying -> validate inline citations against admitted context.
        7. If valid: emit 'citations' and terminal 'done'.
        8. If rejected: emit 'rejection' and terminal 'done' (discarding unverified prose).
        9. On error: emit 'error' and terminal 'done'.

        Yields:
            W3C-compliant SSE text frames.
        """
        start_time = time.perf_counter()
        acquired_cid = await self._lock_manager.acquire(request.conversation_id)

        try:
            # 1. Validation & Pre-flight Token Budgeting
            raw_query = (request.query or "").strip()
            if not raw_query:
                yield format_sse_event(
                    "error",
                    {"code": "invalid_request", "message": "Query string cannot be empty or whitespace-only."},
                )
                yield format_sse_event("done", {"finished": True})
                return

            query = raw_query
            rewritten_query = raw_query
            conversation_service = None

            if request.conversation_id:
                from backend.app.conversations.service import get_conversation_service, ConversationNotFoundError
                conversation_service = self._conversation_service or get_conversation_service()
                try:
                    rewritten_query, _ = await conversation_service.prepare_conversational_turn(
                        conversation_id=request.conversation_id,
                        raw_query=raw_query,
                    )
                    await conversation_service.record_user_message_async(
                        conversation_id=request.conversation_id,
                        raw_query=raw_query,
                        rewritten_query=rewritten_query,
                    )
                    query = rewritten_query
                except ConversationNotFoundError as not_found_err:
                    yield format_sse_event("error", {"code": "not_found", "message": str(not_found_err)})
                    yield format_sse_event("done", {"finished": True})
                    return

            settings = get_settings()
            reserved_completion_tokens = request.max_tokens or settings.DEFAULT_MAX_COMPLETION_TOKENS
            total_request_limit = settings.TOTAL_REQUEST_TOKEN_LIMIT

            try:
                budget_report = self.context_builder.calculate_request_budget_report(
                    query=query,
                    system_prompt=PromptBuilder.get_streaming_system_prompt(),
                    reserved_completion_tokens=reserved_completion_tokens,
                    model_name=self.llm_client.model,
                    total_request_limit=total_request_limit,
                    max_context_limit=self.context_builder.max_context_tokens,
                )
            except ValueError as exc:
                logger.warning("Query rejected by token budget guard: %s", exc)
                yield format_sse_event(
                    "error",
                    {"code": "query_too_large", "message": str(exc)},
                )
                yield format_sse_event("done", {"finished": True})
                return

            allowed_context_tokens = budget_report.allowed_context_budget

            # 2. Retrieval Stage
            yield format_sse_event("status", {"stage": "retrieving", "message": "Searching the knowledge base..."})
            await asyncio.sleep(0)

            retrieval_config = request.retrieval_config or RetrievalConfiguration()
            if not retrieval_config.document_id_filter and request.document_ids:
                retrieval_config.document_id_filter = request.document_ids[0]

            retrieval_req = RetrievalRequest(
                query=query,
                document_ids=request.document_ids,
                config=retrieval_config,
            )

            retrieval_start = time.perf_counter()
            retrieval_result = self.retriever.retrieve(retrieval_req)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
            retrieved_chunks = retrieval_result.chunks

            # Reranking status notification if reranking was performed
            if retrieval_config.enable_reranking and retrieved_chunks:
                yield format_sse_event("status", {"stage": "reranking", "message": "Applying Cross-Encoder neural reranker..."})
                await asyncio.sleep(0)

            # 3. Handle Empty Retrieval
            if not retrieved_chunks:
                logger.warning("Streaming retrieval returned 0 chunks for query '%s'", query[:60])
                fallback_text = "The available documents do not contain any information relevant to this question."
                yield format_sse_event("status", {"stage": "generating", "message": "Synthesizing answer..."})
                yield format_sse_event("token", {"delta": fallback_text})
                await asyncio.sleep(0)

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                metadata = AnswerMetadata(
                    model_name=self.llm_client.model,
                    retrieval_mode=retrieval_result.mode.value,
                    total_chunks_retrieved=0,
                    context_chunks_used=0,
                    context_chunks_excluded=0,
                    context_tokens_used=0,
                    max_context_tokens=allowed_context_tokens,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    request_tokens_budget=budget_report.model_context_limit,
                    has_hallucinated_sources=False,
                    citation_errors=[],
                    latency_ms=round(latency_ms, 2),
                )

                if conversation_service and request.conversation_id:
                    await conversation_service.record_assistant_message_async(
                        conversation_id=request.conversation_id,
                        content=fallback_text,
                        metadata={"insufficient_context": True, "citations": []},
                    )

                yield format_sse_event(
                    "citations",
                    {
                        "status": "verified",
                        "citations": [],
                        "insufficient_context": True,
                        "metadata": metadata.model_dump(),
                    },
                )
                yield format_sse_event("done", {"finished": True})
                return

            # 4. Context Construction Stage
            yield format_sse_event("status", {"stage": "building_context", "message": "Assembling bounded context payload..."})
            await asyncio.sleep(0)

            context_start = time.perf_counter()
            context_result = self.context_builder.build_context(
                retrieved_chunks,
                max_context_tokens=allowed_context_tokens,
            )
            context_ms = (time.perf_counter() - context_start) * 1000.0

            if not context_result.sources_used:
                logger.warning("No chunks fit into context budget for streaming query '%s'", query[:60])
                fallback_text = "The available document context is too large to process within the allocated token budget."
                yield format_sse_event("status", {"stage": "generating", "message": "Synthesizing answer..."})
                yield format_sse_event("token", {"delta": fallback_text})
                await asyncio.sleep(0)

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                metadata = AnswerMetadata(
                    model_name=self.llm_client.model,
                    retrieval_mode=retrieval_result.mode.value,
                    total_chunks_retrieved=len(retrieved_chunks),
                    context_chunks_used=0,
                    context_chunks_excluded=len(context_result.sources_excluded),
                    context_tokens_used=0,
                    max_context_tokens=context_result.max_budget,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    request_tokens_budget=budget_report.model_context_limit,
                    has_hallucinated_sources=False,
                    citation_errors=[],
                    latency_ms=round(latency_ms, 2),
                )

                if conversation_service and request.conversation_id:
                    await conversation_service.record_assistant_message_async(
                        conversation_id=request.conversation_id,
                        content=fallback_text,
                        metadata={"insufficient_context": True, "citations": []},
                    )

                yield format_sse_event(
                    "citations",
                    {
                        "status": "verified",
                        "citations": [],
                        "insufficient_context": True,
                        "metadata": metadata.model_dump(),
                    },
                )
                yield format_sse_event("done", {"finished": True})
                return

            # 5. Build Grounded Streaming Prompt
            messages = PromptBuilder.build_streaming_messages(
                query=query,
                formatted_context=context_result.formatted_context,
            )

            # 6. Stream Generation Stage
            yield format_sse_event("status", {"stage": "generating", "message": "Streaming grounded answer tokens..."})
            await asyncio.sleep(0)

            full_answer_buffer: List[str] = []
            gen_start = time.perf_counter()
            first_token_time: Optional[float] = None

            # Obtain token stream: prefer generate_stream_async if available, fallback to generate_stream
            token_stream = None
            if hasattr(self.llm_client, "generate_stream_async"):
                token_stream = self.llm_client.generate_stream_async(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            elif hasattr(self.llm_client, "generate_stream"):
                token_stream = self.llm_client.generate_stream(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            else:
                raise LLMError("Configured LLM client does not provide a streaming interface.")

            sync_lock = None
            try:
                # If stream is an asynchronous generator/iterator, consume natively without blocking
                if hasattr(token_stream, "__anext__") or hasattr(token_stream, "__aiter__"):
                    async for chunk_str in token_stream:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_answer_buffer.append(chunk_str)
                        yield format_sse_event("token", {"delta": chunk_str})
                        await asyncio.sleep(0)
                else:
                    # Synchronous iterator fallback: read each item in a worker thread via asyncio.to_thread
                    # ensuring that slow network reads never freeze the FastAPI asyncio event loop.
                    # Safe synchronization is used to prevent concurrent next() and close() collisions
                    # (ValueError: generator already executing).
                    import threading
                    sync_lock = threading.Lock()
                    sentinel = object()

                    def _safe_next():
                        with sync_lock:
                            return next(token_stream, sentinel)

                    while True:
                        chunk_str = await asyncio.to_thread(_safe_next)
                        if chunk_str is sentinel:
                            break
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_answer_buffer.append(chunk_str)
                        yield format_sse_event("token", {"delta": chunk_str})
                        await asyncio.sleep(0)
            finally:
                if token_stream is not None:
                    if hasattr(token_stream, "aclose"):
                        try:
                            await token_stream.aclose()
                        except Exception as close_err:
                            logger.debug("Closing async token_stream notice: %s", close_err)
                    elif hasattr(token_stream, "close"):
                        try:
                            def _safe_close():
                                # In Python, OS worker threads cannot be forcibly killed by asyncio.
                                # If a worker thread is still executing inside next(), calling close()
                                # concurrently raises ValueError: generator already executing.
                                acquired = sync_lock.acquire(timeout=0.05) if sync_lock is not None else True
                                if acquired:
                                    try:
                                        token_stream.close()
                                    finally:
                                        if sync_lock is not None:
                                            sync_lock.release()
                                else:
                                    logger.debug("Synchronous iterator worker thread is still active; skipped concurrent close to avoid generator execution race.")

                            await asyncio.to_thread(_safe_close)
                        except ValueError as val_err:
                            if "generator already executing" in str(val_err):
                                logger.debug("Synchronous generator was executing concurrently during cleanup: %s", val_err)
                            else:
                                logger.debug("ValueError during synchronous token_stream close: %s", val_err)
                        except Exception as close_err:
                            logger.debug("Closing sync token_stream notice: %s", close_err)

            accumulated_answer = "".join(full_answer_buffer)

            # 7. Post-Generation Citation Validation
            yield format_sse_event("status", {"stage": "verifying", "message": "Validating citation provenance against admitted sources..."})
            await asyncio.sleep(0)

            validation_result = CitationValidator.validate_citations(
                raw_citations=None,
                answer_text=accumulated_answer,
                sources_used=context_result.sources_used,
                sources_excluded=context_result.sources_excluded,
            )

            lower_answer = accumulated_answer.lower()
            insufficient_phrases = [
                "not contain sufficient information",
                "does not provide sufficient information",
                "insufficient information to answer",
                "do not contain enough information",
                "cannot answer this question based on the provided",
            ]
            insufficient_context = any(phrase in lower_answer for phrase in insufficient_phrases)

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            ttft_ms = round((first_token_time - start_time) * 1000.0, 2) if first_token_time else None

            # Enforce Safe Rejection Policy: If citations are hallucinated, discard provisional text
            if validation_result.has_hallucinated_sources:
                logger.warning(
                    "Streaming safe rejection triggered: invalid/fabricated citations: %s",
                    validation_result.invalid_citations,
                )
                safe_fallback_prose = (
                    "The generated answer could not be verified because it referenced invalid, "
                    "fabricated, or excluded source citations. The available documents do not "
                    "substantiate this response."
                )
                if conversation_service and request.conversation_id:
                    await conversation_service.record_assistant_message_async(
                        conversation_id=request.conversation_id,
                        content=safe_fallback_prose,
                        metadata={
                            "rejected": True,
                            "reason": "Citation verification failed due to invalid, fabricated, or excluded source references.",
                            "invalid_citations": validation_result.invalid_citations,
                        },
                    )
                yield format_sse_event(
                    "rejection",
                    {
                        "status": "rejected",
                        "reason": "Citation verification failed due to invalid, fabricated, or excluded source references.",
                        "invalid_citations": validation_result.invalid_citations,
                        "safe_fallback": safe_fallback_prose,
                    },
                )
                yield format_sse_event("done", {"finished": True})
                return

            # Successful Citation Validation
            validated_citations = validation_result.valid_citations
            if conversation_service and request.conversation_id:
                await conversation_service.record_assistant_message_async(
                    conversation_id=request.conversation_id,
                    content=accumulated_answer,
                    metadata={
                        "citations": [c.model_dump() for c in validated_citations],
                        "insufficient_context": insufficient_context,
                        "retrieval_mode": retrieval_result.mode.value,
                        "latency_ms": round(latency_ms, 2),
                    },
                )
            metadata = AnswerMetadata(
                model_name=self.llm_client.model,
                retrieval_mode=retrieval_result.mode.value,
                total_chunks_retrieved=len(retrieved_chunks),
                context_chunks_used=len(context_result.sources_used),
                context_chunks_excluded=len(context_result.sources_excluded),
                context_tokens_used=context_result.total_tokens_used,
                max_context_tokens=context_result.max_budget,
                prompt_tokens=None,
                completion_tokens=len(accumulated_answer.split()),
                total_tokens=None,
                request_tokens_budget=budget_report.model_context_limit,
                has_hallucinated_sources=False,
                citation_errors=[],
                latency_ms=round(latency_ms, 2),
            )

            yield format_sse_event(
                "citations",
                {
                    "status": "verified",
                    "citations": [c.model_dump() for c in validated_citations],
                    "insufficient_context": insufficient_context,
                    "metadata": metadata.model_dump(),
                    "ttft_ms": ttft_ms,
                },
            )
            yield format_sse_event("done", {"finished": True})

        except GeneratorExit:
            logger.info("Streaming client disconnected; generator exited.")
            raise

        except LLMAuthenticationError as exc:
            logger.error("Streaming LLM auth error: %s", exc.message)
            yield format_sse_event("error", {"code": "authentication_error", "message": exc.user_message})
            yield format_sse_event("done", {"finished": True})

        except LLMRateLimitError as exc:
            logger.warning("Streaming LLM rate limit: %s", exc.message)
            yield format_sse_event("error", {"code": "rate_limit_error", "message": exc.user_message})
            yield format_sse_event("done", {"finished": True})

        except LLMTimeoutError as exc:
            logger.error("Streaming LLM timeout: %s", exc.message)
            yield format_sse_event("error", {"code": "timeout_error", "message": exc.user_message})
            yield format_sse_event("done", {"finished": True})

        except LLMNetworkError as exc:
            logger.error("Streaming LLM network error: %s", exc.message)
            yield format_sse_event("error", {"code": "network_error", "message": exc.user_message})
            yield format_sse_event("done", {"finished": True})

        except LLMConfigError as exc:
            logger.error("Streaming LLM config error: %s", exc.message)
            yield format_sse_event("error", {"code": "config_error", "message": exc.user_message})
            yield format_sse_event("done", {"finished": True})

        except LLMError as exc:
            logger.error("Streaming LLM general error: %s", exc.message)
            yield format_sse_event("error", {"code": "llm_error", "message": exc.user_message})
            yield format_sse_event("done", {"finished": True})

        except Exception as exc:
            logger.error("Unexpected error in answer_question_stream: %s", exc, exc_info=True)
            yield format_sse_event("error", {"code": "internal_error", "message": "An unexpected error occurred during answer generation."})
            yield format_sse_event("done", {"finished": True})

        finally:
            if acquired_cid:
                await self._lock_manager.release(acquired_cid)


_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Return the global RAGService instance."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
