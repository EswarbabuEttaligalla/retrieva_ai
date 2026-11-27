"""RAG Pipeline Orchestration Service for Retrieva AI.

Connects the Phase 4 Retrieval Engine with Phase 5 Context Construction,
Prompt Engineering, OpenAI Inference, Citation Validation, and Supabase
owner-scoped conversation persistence.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from backend.app.conversations.service import ConversationNotFoundError
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.core.sse import format_sse_event
from backend.app.core.supabase_client import SupabaseClientError, SupabaseUnavailableError
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
from backend.app.vector_store.store import normalize_owner_id

logger = get_logger("services.rag_service")


class RAGService:
    """Production orchestrator executing the grounded RAG answer generation lifecycle."""

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

    @staticmethod
    def _build_retrieval_request(request: AnswerRequest, query: str) -> RetrievalRequest:
        """Assemble a RetrievalRequest preserving single and multi-document filters."""
        retrieval_config = (
            request.retrieval_config.model_copy()
            if request.retrieval_config is not None
            else RetrievalConfiguration(top_k=7)
        )
        if request.document_ids:
            cleaned_doc_ids = [doc_id.strip() for doc_id in request.document_ids if doc_id and doc_id.strip()]
            if (
                len(cleaned_doc_ids) == 1
                and not retrieval_config.document_id_filter
                and not retrieval_config.document_ids_filter
            ):
                retrieval_config.document_id_filter = cleaned_doc_ids[0]
            elif cleaned_doc_ids:
                if retrieval_config.document_ids_filter:
                    existing_set = {
                        doc_id.strip()
                        for doc_id in retrieval_config.document_ids_filter
                        if doc_id and doc_id.strip()
                    }
                    retrieval_config.document_ids_filter = [
                        doc_id for doc_id in cleaned_doc_ids if doc_id in existing_set
                    ]
                else:
                    retrieval_config.document_ids_filter = cleaned_doc_ids
        return RetrievalRequest(query=query, config=retrieval_config)

    @staticmethod
    async def _persist_conversational_turn(
        conv_svc: Any,
        conversation_id: str,
        owner_uuid: UUID,
        raw_query: str,
        rewritten_query: str,
        assistant_content: Optional[str],
        assistant_metadata: Optional[Dict[str, Any]] = None,
        assistant_citations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist the user message and optional verified assistant message to Supabase."""
        await conv_svc.record_user_message(
            conversation_id=conversation_id,
            owner_id=owner_uuid,
            raw_query=raw_query,
            rewritten_query=rewritten_query,
        )
        if assistant_content is not None:
            await conv_svc.record_assistant_message(
                conversation_id=conversation_id,
                owner_id=owner_uuid,
                content=assistant_content,
                metadata=assistant_metadata,
                citations=assistant_citations,
            )

    def _assemble_non_conversational_answer(
        self,
        request: AnswerRequest,
        owner_id: Union[UUID, str],
        raw_query: str,
        query: str,
        rewritten_query: str,
        start_time: float,
        authorized_ready_doc_ids: Optional[Set[str]] = None,
    ) -> Tuple[AnswerResponse, Optional[Dict[str, Any]], List[Dict[str, Any]], bool]:
        """Execute synchronous retrieval, context assembly, generation, and citation verification."""
        settings = get_settings()
        reserved_completion_tokens = request.max_tokens or settings.DEFAULT_MAX_COMPLETION_TOKENS
        total_request_limit = settings.TOTAL_REQUEST_TOKEN_LIMIT

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

        retrieval_req = self._build_retrieval_request(request, query)
        if authorized_ready_doc_ids is not None:
            retrieval_result = self.retriever.retrieve(
                retrieval_req,
                owner_id=owner_id,
                authorized_ready_doc_ids=authorized_ready_doc_ids,
            )
            retrieved_chunks = [c for c in retrieval_result.chunks if c.document_id in authorized_ready_doc_ids]
        else:
            retrieval_result = self.retriever.retrieve(retrieval_req, owner_id=owner_id)
            retrieved_chunks = retrieval_result.chunks
        if retrieved_chunks and hasattr(self.retriever, "expand_table_continuations"):
            if authorized_ready_doc_ids is not None:
                expanded = self.retriever.expand_table_continuations(
                    retrieved_chunks,
                    owner_id=owner_id,
                    authorized_ready_doc_ids=authorized_ready_doc_ids,
                )
            else:
                expanded = self.retriever.expand_table_continuations(
                    retrieved_chunks,
                    owner_id=owner_id,
                )
            if isinstance(expanded, list):
                retrieved_chunks = (
                    [c for c in expanded if c.document_id in authorized_ready_doc_ids]
                    if authorized_ready_doc_ids is not None
                    else expanded
                )

        logger.info(
            "Retrieved %d candidate chunks (mode=%s)",
            len(retrieved_chunks),
            retrieval_result.mode.value,
        )

        # Empty retrieval -> verified insufficient-context fallback
        if not retrieved_chunks:
            logger.warning("Retrieval returned 0 chunks for query '%s'", query[:60])
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            fallback_ans = "The available documents do not contain any information relevant to this question."
            resp = AnswerResponse(
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
            return resp, {"insufficient_context": True, "citations": [], "retrieval_mode": retrieval_result.mode.value}, [], True

        # Context construction
        context_result = self.context_builder.build_context(
            retrieved_chunks,
            max_context_tokens=allowed_context_tokens,
        )

        if not context_result.sources_used:
            logger.warning("No chunks fit into context budget for query '%s'", query[:60])
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            fallback_ans = "The available document context is too large to process within the allocated token budget."
            resp = AnswerResponse(
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
            return resp, {"insufficient_context": True, "citations": [], "retrieval_mode": retrieval_result.mode.value}, [], True

        # Grounded prompt & generation
        messages = PromptBuilder.build_messages(
            query=query,
            formatted_context=context_result.formatted_context,
        )
        llm_output = self.llm_client.generate(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        raw_content = llm_output["content"]
        answer_text, insufficient_context, raw_citations = self._parse_model_output(raw_content)

        validation_result = CitationValidator.validate_citations(
            raw_citations=raw_citations,
            answer_text=answer_text,
            sources_used=context_result.sources_used,
            sources_excluded=context_result.sources_excluded,
        )

        should_persist_assistant = not validation_result.has_hallucinated_sources
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
            validated_citations: List[SourceCitation] = []
            persist_meta: Optional[Dict[str, Any]] = None
            cit_dicts: List[Dict[str, Any]] = []
        else:
            validated_citations = validation_result.valid_citations
            cit_dicts = [c.model_dump() for c in validated_citations]
            persist_meta = {
                "citations": cit_dicts,
                "insufficient_context": insufficient_context,
                "retrieval_mode": retrieval_result.mode.value,
            }

        if not validated_citations and not insufficient_context and context_result.sources_used:
            logger.warning("Answer generated without explicit source citations despite non-empty context.")

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

        resp = AnswerResponse(
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
        return resp, persist_meta, cit_dicts, should_persist_assistant

    def answer_question(
        self,
        request: AnswerRequest,
        owner_id: Union[UUID, str],
        authorized_ready_doc_ids: Optional[Set[str]] = None,
    ) -> AnswerResponse:
        """Synchronous RAG answer generation for standalone non-conversational callers."""
        raw_query = (request.query or "").strip()
        if not raw_query:
            raise ValueError("Query string cannot be empty or whitespace-only.")
        if request.conversation_id:
            raise RuntimeError(
                "Conversational answering requires Supabase async pipeline (answer_question_async); "
                "SQLite fallback is disabled."
            )

        start_time = time.perf_counter()
        resp, _, _, _ = self._assemble_non_conversational_answer(
            request=request,
            owner_id=owner_id,
            raw_query=raw_query,
            query=raw_query,
            rewritten_query=raw_query,
            start_time=start_time,
            authorized_ready_doc_ids=authorized_ready_doc_ids,
        )
        return resp

    async def answer_question_async(
        self,
        request: AnswerRequest,
        owner_id: Union[UUID, str],
        conversation_service: Optional[Any] = None,
        authorized_ready_doc_ids: Optional[Set[str]] = None,
    ) -> AnswerResponse:
        """Execute full async RAG pipeline with Supabase owner-scoped conversation persistence."""
        start_time = time.perf_counter()
        raw_query = (request.query or "").strip()
        if not raw_query:
            raise ValueError("Query string cannot be empty or whitespace-only.")

        owner_uuid = UUID(normalize_owner_id(owner_id))
        query = raw_query
        rewritten_query = raw_query
        conv_svc = conversation_service or self._conversation_service

        if request.conversation_id:
            if conv_svc is None:
                raise SupabaseUnavailableError(
                    "Supabase conversation service is unavailable.",
                    status_code=503,
                    code="conversation_service_unavailable",
                )
            rewritten_query, _ = await conv_svc.prepare_conversational_turn(
                conversation_id=request.conversation_id,
                owner_id=owner_uuid,
                raw_query=raw_query,
            )
            query = rewritten_query

        resp, persist_meta, cit_dicts, should_persist_assistant = await asyncio.to_thread(
            self._assemble_non_conversational_answer,
            request,
            owner_uuid,
            raw_query,
            query,
            rewritten_query,
            start_time,
            authorized_ready_doc_ids,
        )

        if request.conversation_id and conv_svc is not None:
            await self._persist_conversational_turn(
                conv_svc=conv_svc,
                conversation_id=request.conversation_id,
                owner_uuid=owner_uuid,
                raw_query=raw_query,
                rewritten_query=rewritten_query,
                assistant_content=resp.answer if should_persist_assistant else None,
                assistant_metadata=persist_meta if should_persist_assistant else None,
                assistant_citations=cit_dicts if should_persist_assistant else None,
            )

        return resp

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

    async def answer_question_stream(
        self,
        request: AnswerRequest,
        owner_id: Union[UUID, str],
        conversation_service: Optional[Any] = None,
        authorized_ready_doc_ids: Optional[Set[str]] = None,
    ) -> AsyncIterator[str]:
        """Execute the real-time RAG streaming pipeline emitting SSE events with Supabase persistence."""
        start_time = time.perf_counter()
        acquired_cid = await self._lock_manager.acquire(request.conversation_id)

        try:
            raw_query = (request.query or "").strip()
            if not raw_query:
                yield format_sse_event(
                    "error",
                    {"code": "invalid_request", "message": "Query string cannot be empty or whitespace-only."},
                )
                yield format_sse_event("done", {"finished": True})
                return

            owner_uuid = UUID(normalize_owner_id(owner_id))
            query = raw_query
            rewritten_query = raw_query
            conv_svc = conversation_service or self._conversation_service

            if request.conversation_id:
                if conv_svc is None:
                    yield format_sse_event(
                        "error",
                        {"code": "service_unavailable", "message": "Supabase conversation service is unavailable."},
                    )
                    yield format_sse_event("done", {"finished": True})
                    return
                try:
                    rewritten_query, _ = await conv_svc.prepare_conversational_turn(
                        conversation_id=request.conversation_id,
                        owner_id=owner_uuid,
                        raw_query=raw_query,
                    )
                    query = rewritten_query
                except ConversationNotFoundError as not_found_err:
                    yield format_sse_event("error", {"code": "not_found", "message": str(not_found_err)})
                    yield format_sse_event("done", {"finished": True})
                    return
                except SupabaseClientError as sb_err:
                    logger.error("Supabase failure preparing conversational stream: %s", sb_err.message)
                    yield format_sse_event(
                        "error",
                        {"code": "service_unavailable", "message": sb_err.message},
                    )
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
                yield format_sse_event("error", {"code": "query_too_large", "message": str(exc)})
                yield format_sse_event("done", {"finished": True})
                return

            allowed_context_tokens = budget_report.allowed_context_budget

            # 2. Retrieval Stage
            yield format_sse_event("status", {"stage": "retrieving", "message": "Searching the knowledge base..."})
            await asyncio.sleep(0)

            retrieval_req = self._build_retrieval_request(request, query)
            if authorized_ready_doc_ids is not None:
                retrieval_result = await asyncio.to_thread(
                    self.retriever.retrieve,
                    retrieval_req,
                    owner_uuid,
                    authorized_ready_doc_ids,
                )
                retrieved_chunks = [c for c in retrieval_result.chunks if c.document_id in authorized_ready_doc_ids]
            else:
                retrieval_result = await asyncio.to_thread(self.retriever.retrieve, retrieval_req, owner_uuid)
                retrieved_chunks = retrieval_result.chunks
            if retrieved_chunks and hasattr(self.retriever, "expand_table_continuations"):
                if authorized_ready_doc_ids is not None:
                    expanded = await asyncio.to_thread(
                        self.retriever.expand_table_continuations,
                        retrieved_chunks,
                        owner_uuid,
                        10,
                        authorized_ready_doc_ids,
                    )
                else:
                    expanded = await asyncio.to_thread(
                        self.retriever.expand_table_continuations,
                        retrieved_chunks,
                        owner_uuid,
                    )
                if isinstance(expanded, list):
                    retrieved_chunks = (
                        [c for c in expanded if c.document_id in authorized_ready_doc_ids]
                        if authorized_ready_doc_ids is not None
                        else expanded
                    )

            if retrieval_req.config.enable_reranking and retrieved_chunks:
                yield format_sse_event("status", {"stage": "reranking", "message": "Applying Cross-Encoder neural reranker..."})
                await asyncio.sleep(0)

            # 3. Empty Retrieval Fallback
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

                if conv_svc and request.conversation_id:
                    try:
                        await self._persist_conversational_turn(
                            conv_svc=conv_svc,
                            conversation_id=request.conversation_id,
                            owner_uuid=owner_uuid,
                            raw_query=raw_query,
                            rewritten_query=rewritten_query,
                            assistant_content=fallback_text,
                            assistant_metadata={
                                "insufficient_context": True,
                                "citations": [],
                                "retrieval_mode": retrieval_result.mode.value,
                            },
                            assistant_citations=[],
                        )
                    except SupabaseClientError as sb_err:
                        logger.error("Failed persisting empty-retrieval stream turn: %s", sb_err.message)
                        yield format_sse_event(
                            "error",
                            {"code": "persistence_error", "message": "Failed to save conversation turn to Supabase."},
                        )
                        yield format_sse_event("done", {"finished": True})
                        return

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

            context_result = self.context_builder.build_context(
                retrieved_chunks,
                max_context_tokens=allowed_context_tokens,
            )

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

                if conv_svc and request.conversation_id:
                    try:
                        await self._persist_conversational_turn(
                            conv_svc=conv_svc,
                            conversation_id=request.conversation_id,
                            owner_uuid=owner_uuid,
                            raw_query=raw_query,
                            rewritten_query=rewritten_query,
                            assistant_content=fallback_text,
                            assistant_metadata={
                                "insufficient_context": True,
                                "citations": [],
                                "retrieval_mode": retrieval_result.mode.value,
                            },
                            assistant_citations=[],
                        )
                    except SupabaseClientError as sb_err:
                        logger.error("Failed persisting budget-overflow stream turn: %s", sb_err.message)
                        yield format_sse_event(
                            "error",
                            {"code": "persistence_error", "message": "Failed to save conversation turn to Supabase."},
                        )
                        yield format_sse_event("done", {"finished": True})
                        return

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
            first_token_time: Optional[float] = None

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
                if hasattr(token_stream, "__anext__") or hasattr(token_stream, "__aiter__"):
                    async for chunk_str in token_stream:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_answer_buffer.append(chunk_str)
                        yield format_sse_event("token", {"delta": chunk_str})
                        await asyncio.sleep(0)
                else:
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
                                acquired = sync_lock.acquire(timeout=0.05) if sync_lock is not None else True
                                if acquired:
                                    try:
                                        token_stream.close()
                                    finally:
                                        if sync_lock is not None:
                                            sync_lock.release()

                            await asyncio.to_thread(_safe_close)
                        except ValueError as val_err:
                            logger.debug("Concurrent close notice on sync token_stream: %s", val_err)
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
            insufficient_context = any(phrase in lower_answer for phrase in insufficient_phrases) or (
                not validation_result.valid_citations
                and "not reported in the provided document evidence" in lower_answer
            )

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            ttft_ms = round((first_token_time - start_time) * 1000.0, 2) if first_token_time else None

            # Enforce Safe Rejection Policy: NEVER persist assistant message with invalid/fabricated citations
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
                if conv_svc and request.conversation_id:
                    try:
                        await self._persist_conversational_turn(
                            conv_svc=conv_svc,
                            conversation_id=request.conversation_id,
                            owner_uuid=owner_uuid,
                            raw_query=raw_query,
                            rewritten_query=rewritten_query,
                            assistant_content=None,
                        )
                    except SupabaseClientError as sb_err:
                        logger.error("Failed recording rejected stream user turn: %s", sb_err.message)
                        yield format_sse_event(
                            "error",
                            {"code": "persistence_error", "message": "Failed to persist conversation state."},
                        )
                        yield format_sse_event("done", {"finished": True})
                        return

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

            # Successful Citation Validation -> Persist user + verified assistant message
            validated_citations = validation_result.valid_citations
            cit_dicts = [c.model_dump() for c in validated_citations]
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

            if conv_svc and request.conversation_id:
                try:
                    await self._persist_conversational_turn(
                        conv_svc=conv_svc,
                        conversation_id=request.conversation_id,
                        owner_uuid=owner_uuid,
                        raw_query=raw_query,
                        rewritten_query=rewritten_query,
                        assistant_content=accumulated_answer,
                        assistant_metadata={
                            "citations": cit_dicts,
                            "insufficient_context": insufficient_context,
                            "retrieval_mode": retrieval_result.mode.value,
                            "latency_ms": round(latency_ms, 2),
                        },
                        assistant_citations=cit_dicts,
                    )
                except SupabaseClientError as sb_err:
                    logger.error("Failed persisting verified streaming assistant turn: %s", sb_err.message)
                    yield format_sse_event(
                        "error",
                        {"code": "persistence_error", "message": "Failed to persist verified assistant response to Supabase."},
                    )
                    yield format_sse_event("done", {"finished": True})
                    return

            yield format_sse_event(
                "citations",
                {
                    "status": "verified",
                    "citations": cit_dicts,
                    "insufficient_context": insufficient_context,
                    "metadata": metadata.model_dump(),
                    "ttft_ms": ttft_ms,
                },
            )
            yield format_sse_event("done", {"finished": True})

        except GeneratorExit:
            logger.info("Streaming client disconnected; generator exited without persisting partial output.")
            raise

        except asyncio.CancelledError:
            logger.info("Streaming request cancelled; skipping partial persistence.")
            raise

        except ValueError as exc:
            logger.warning("Streaming validation/isolation failure: %s", exc)
            yield format_sse_event("error", {"code": "invalid_request", "message": str(exc)})
            yield format_sse_event("done", {"finished": True})

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
