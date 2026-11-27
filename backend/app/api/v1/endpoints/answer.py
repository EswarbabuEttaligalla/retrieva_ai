"""Answer generation endpoint for Retrieva AI RAG pipeline."""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials

from backend.app.conversations.query_rewriter import get_query_rewriter
from backend.app.conversations.service import (
    ConversationNotFoundError,
    ConversationService,
    get_conversation_service,
)
from backend.app.core.auth import AuthenticatedUser, bearer_scheme, get_current_user
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.core.supabase_client import (
    SupabaseClient,
    SupabaseClientError,
    SupabaseConfigError,
    raise_for_supabase_error,
)
from backend.app.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from backend.app.repositories.conversation_repository import SupabaseConversationRepository
from backend.app.repositories.document_repository import resolve_ready_document_ids_for_request
from backend.app.repositories.message_repository import SupabaseMessageRepository
from backend.app.schemas.answer import AnswerRequest, AnswerResponse
from backend.app.services.rag_service import RAGService, get_rag_service

router = APIRouter()
logger = get_logger("api.endpoints.answer")


def _resolve_optional_conversation_service(
    http_request: Request,
    conversation_id: Optional[str],
    current_user: AuthenticatedUser,
    credentials: Optional[HTTPAuthorizationCredentials],
    settings: Settings,
) -> Optional[ConversationService]:
    """Resolve a request-scoped Supabase ConversationService only when conversation_id is supplied."""
    if not conversation_id:
        return None

    override_fn = http_request.app.dependency_overrides.get(get_conversation_service)
    if override_fn is not None:
        return override_fn()

    if credentials is None or credentials.scheme.lower() != "bearer" or not (credentials.credentials or "").strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer authentication credentials for conversation persistence.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase conversation service is not configured.",
        )

    try:
        client = SupabaseClient(
            supabase_url=settings.SUPABASE_URL,
            anon_key=settings.SUPABASE_ANON_KEY,
            access_token=credentials.credentials.strip(),
            owner_id=current_user.user_id,
            timeout_seconds=settings.SUPABASE_REST_TIMEOUT_SECONDS,
        )
    except SupabaseConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        ) from exc

    return ConversationService(
        conversation_repo=SupabaseConversationRepository(client),
        message_repo=SupabaseMessageRepository(client),
        query_rewriter=get_query_rewriter(),
    )


@router.post(
    "/answer",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Grounded RAG Answer",
    description="Accepts a natural language query, retrieves relevant document chunks, synthesizes a grounded answer, and returns verified source citations.",
)
async def generate_answer(
    request: AnswerRequest,
    http_request: Request,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> AnswerResponse:
    """Generate a context-grounded answer with verified source citations."""
    conv_service = _resolve_optional_conversation_service(
        http_request=http_request,
        conversation_id=request.conversation_id,
        current_user=current_user,
        credentials=credentials,
        settings=settings,
    )

    ready_doc_ids = await resolve_ready_document_ids_for_request(
        http_request=http_request,
        current_user=current_user,
        credentials=credentials,
        settings=settings,
    )

    try:
        if request.conversation_id and conv_service is not None:
            conv = await conv_service.get_conversation(request.conversation_id, owner_id=current_user.user_id)
            if not conv:
                raise ConversationNotFoundError("Conversation not found.")
            async with rag_service.lock_manager.lock(request.conversation_id):
                if isinstance(rag_service, RAGService):
                    return await rag_service.answer_question_async(
                        request,
                        owner_id=current_user.user_id,
                        conversation_service=conv_service,
                        authorized_ready_doc_ids=ready_doc_ids,
                    )
                return await asyncio.to_thread(
                    rag_service.answer_question,
                    request,
                    current_user.user_id,
                )
        else:
            if isinstance(rag_service, RAGService):
                return await rag_service.answer_question_async(
                    request,
                    owner_id=current_user.user_id,
                    conversation_service=None,
                    authorized_ready_doc_ids=ready_doc_ids,
                )
            return await asyncio.to_thread(
                rag_service.answer_question,
                request,
                current_user.user_id,
            )

    except ValueError as exc:
        logger.warning("Invalid answer request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except ConversationNotFoundError as exc:
        logger.warning("Conversation not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        ) from exc

    except SupabaseClientError as exc:
        raise_for_supabase_error(exc, "Failed communicating with Supabase conversation service.")

    except LLMConfigError as exc:
        logger.error("LLM configuration error: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.user_message,
        ) from exc

    except LLMAuthenticationError as exc:
        logger.error("LLM authentication failure: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.user_message,
        ) from exc

    except LLMRateLimitError as exc:
        logger.warning("LLM rate limit reached: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=exc.user_message,
        ) from exc

    except LLMTimeoutError as exc:
        logger.error("LLM timeout: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=exc.user_message,
        ) from exc

    except LLMNetworkError as exc:
        logger.error("LLM network connection failure: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.user_message,
        ) from exc

    except LLMError as exc:
        logger.error("LLM answer generation failed: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.user_message,
        ) from exc

    except Exception as exc:
        logger.error("Unexpected error during RAG orchestration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while generating the answer.",
        ) from exc


@router.post(
    "/answer/stream",
    status_code=status.HTTP_200_OK,
    summary="Generate Grounded RAG Answer via Real-Time SSE Stream",
    description="Streams answer tokens progressively with SSE, performing citation provenance validation upon completion.",
)
async def generate_answer_stream(
    request: AnswerRequest,
    http_request: Request,
    rag_service: RAGService = Depends(get_rag_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream grounded answer tokens in real-time with post-stream citation validation."""
    conv_service = _resolve_optional_conversation_service(
        http_request=http_request,
        conversation_id=request.conversation_id,
        current_user=current_user,
        credentials=credentials,
        settings=settings,
    )

    if request.conversation_id and conv_service is not None:
        try:
            conv = await conv_service.get_conversation(request.conversation_id, owner_id=current_user.user_id)
        except SupabaseClientError as exc:
            raise_for_supabase_error(exc, "Failed verifying conversation ownership.")
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )

    ready_doc_ids = await resolve_ready_document_ids_for_request(
        http_request=http_request,
        current_user=current_user,
        credentials=credentials,
        settings=settings,
    )

    return StreamingResponse(
        rag_service.answer_question_stream(
            request,
            owner_id=current_user.user_id,
            conversation_service=conv_service,
            authorized_ready_doc_ids=ready_doc_ids,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
