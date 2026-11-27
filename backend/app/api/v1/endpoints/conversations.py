"""API endpoints for Supabase-backed persistent conversations, multi-turn chat, and streaming."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials

from backend.app.conversations.models import (
    ConversationAnswerRequest,
    ConversationDetail,
    ConversationRecord,
    CreateConversationRequest,
    UpdateConversationRequest,
)
from backend.app.conversations.service import (
    ConversationNotFoundError,
    ConversationService,
    get_conversation_service,
)
from backend.app.core.auth import AuthenticatedUser, bearer_scheme, get_current_user
from backend.app.core.config import Settings, get_settings
from backend.app.core.supabase_client import SupabaseClientError, raise_for_supabase_error
from backend.app.repositories.document_repository import resolve_ready_document_ids_for_request
from backend.app.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from backend.app.schemas.answer import AnswerRequest, AnswerResponse
from backend.app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/conversations", tags=["Conversations"])
logger = logging.getLogger("retrieva_ai.api.endpoints.conversations")


@router.post(
    "",
    response_model=ConversationRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Create New Conversation",
    description="Creates a new persistent conversation session owned by the authenticated user.",
)
async def create_conversation(
    request: Optional[CreateConversationRequest] = None,
    service: ConversationService = Depends(get_conversation_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> ConversationRecord:
    """Create a new owner-scoped conversation in Supabase."""
    title = request.title if request else None
    try:
        return await service.create_conversation(owner_id=current_user.user_id, title=title)
    except SupabaseClientError as exc:
        raise_for_supabase_error(exc, "Failed to create conversation.")


@router.get(
    "",
    response_model=List[ConversationRecord],
    status_code=status.HTTP_200_OK,
    summary="List Conversations",
    description="Returns all conversations owned by the authenticated user ordered by most recently updated.",
)
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=200, description="Max conversations to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    service: ConversationService = Depends(get_conversation_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> List[ConversationRecord]:
    """List owner-scoped conversations from Supabase ordered by updated_at descending."""
    try:
        return await service.list_conversations(
            owner_id=current_user.user_id,
            limit=limit,
            offset=offset,
        )
    except SupabaseClientError as exc:
        raise_for_supabase_error(exc, "Failed to list conversations.")


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Conversation Details and Messages",
    description="Returns owned conversation metadata and chronological message history.",
)
async def get_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> ConversationDetail:
    """Retrieve an owned conversation and its messages; returns uniform 404 if missing or foreign."""
    try:
        detail = await service.get_conversation_with_messages(
            conversation_id=conversation_id,
            owner_id=current_user.user_id,
        )
    except SupabaseClientError as exc:
        raise_for_supabase_error(exc, "Failed to retrieve conversation.")

    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )
    return detail


@router.patch(
    "/{conversation_id}",
    response_model=ConversationRecord,
    status_code=status.HTTP_200_OK,
    summary="Update Conversation Title",
    description="Updates the title of an owned conversation.",
)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    service: ConversationService = Depends(get_conversation_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> ConversationRecord:
    """Rename an owned conversation; returns uniform 404 for missing or foreign IDs."""
    try:
        return await service.update_conversation_title(
            conversation_id=conversation_id,
            owner_id=current_user.user_id,
            title=request.title,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        ) from exc
    except SupabaseClientError as exc:
        raise_for_supabase_error(exc, "Failed to update conversation title.")


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Conversation",
    description="Deletes an owned conversation session and cascades deletion to its messages.",
)
async def delete_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Delete an owned conversation; returns uniform 404 for missing or foreign IDs."""
    try:
        await service.delete_conversation(
            conversation_id=conversation_id,
            owner_id=current_user.user_id,
        )
        return {"deleted": True, "conversation_id": conversation_id}
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        ) from exc
    except SupabaseClientError as exc:
        raise_for_supabase_error(exc, "Failed to delete conversation.")


@router.post(
    "/{conversation_id}/answer",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Conversation-Aware Answer",
    description="Executes grounded RAG question answering within the context of an owned conversation session.",
)
async def answer_conversation(
    conversation_id: str,
    request: ConversationAnswerRequest,
    http_request: Request,
    service: ConversationService = Depends(get_conversation_service),
    rag_service: RAGService = Depends(get_rag_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
):
    """Verify conversation ownership and execute conversational RAG in JSON or SSE stream mode."""
    try:
        conv = await service.get_conversation(conversation_id, owner_id=current_user.user_id)
    except SupabaseClientError as exc:
        raise_for_supabase_error(exc, "Failed to verify conversation ownership.")

    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    rag_request = AnswerRequest(
        query=request.query,
        document_ids=request.document_ids,
        retrieval_config=request.retrieval_config,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        conversation_id=conversation_id,
    )

    ready_doc_ids = await resolve_ready_document_ids_for_request(
        http_request=http_request,
        current_user=current_user,
        credentials=credentials,
        settings=settings,
    )

    if request.stream:
        return StreamingResponse(
            rag_service.answer_question_stream(
                rag_request,
                owner_id=current_user.user_id,
                conversation_service=service,
                authorized_ready_doc_ids=ready_doc_ids,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async with rag_service.lock_manager.lock(conversation_id):
        try:
            return await rag_service.answer_question_async(
                rag_request,
                owner_id=current_user.user_id,
                conversation_service=service,
                authorized_ready_doc_ids=ready_doc_ids,
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            ) from exc
        except SupabaseClientError as exc:
            raise_for_supabase_error(exc, "Failed persisting conversation turn.")
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except LLMConfigError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=exc.user_message,
            ) from exc
        except LLMAuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=exc.user_message,
            ) from exc
        except LLMRateLimitError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=exc.user_message,
            ) from exc
        except LLMTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=exc.user_message,
            ) from exc
        except LLMNetworkError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=exc.user_message,
            ) from exc
        except LLMError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=exc.user_message,
            ) from exc


@router.post(
    "/{conversation_id}/answer/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream Conversation-Aware Answer via SSE",
    description="Streams grounded answer tokens progressively with owner-isolated query rewriting and Supabase persistence.",
)
async def answer_conversation_stream(
    conversation_id: str,
    request: ConversationAnswerRequest,
    http_request: Request,
    service: ConversationService = Depends(get_conversation_service),
    rag_service: RAGService = Depends(get_rag_service),
    current_user: AuthenticatedUser = Depends(get_current_user),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Verify ownership before starting SSE stream and persist verified turns via Supabase."""
    try:
        conv = await service.get_conversation(conversation_id, owner_id=current_user.user_id)
    except SupabaseClientError as exc:
        raise_for_supabase_error(exc, "Failed to verify conversation ownership.")

    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    rag_request = AnswerRequest(
        query=request.query,
        document_ids=request.document_ids,
        retrieval_config=request.retrieval_config,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        conversation_id=conversation_id,
    )

    ready_doc_ids = await resolve_ready_document_ids_for_request(
        http_request=http_request,
        current_user=current_user,
        credentials=credentials,
        settings=settings,
    )

    return StreamingResponse(
        rag_service.answer_question_stream(
            rag_request,
            owner_id=current_user.user_id,
            conversation_service=service,
            authorized_ready_doc_ids=ready_doc_ids,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
