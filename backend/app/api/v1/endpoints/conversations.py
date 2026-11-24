"""API endpoints for persistent conversations, multi-turn chat sessions, and conversational answering."""

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

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
from backend.app.schemas.answer import AnswerRequest, AnswerResponse
from backend.app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/conversations", tags=["Conversations"])
logger = logging.getLogger("retrieva_ai.api.endpoints.conversations")


@router.post(
    "",
    response_model=ConversationRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Create New Conversation",
    description="Creates a new persistent conversation session.",
)
async def create_conversation(
    request: Optional[CreateConversationRequest] = None,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationRecord:
    """Create a new persistent conversation asynchronously without blocking the event loop."""
    title = request.title if request else None
    return await service.create_conversation_async(title=title)


@router.get(
    "",
    response_model=List[ConversationRecord],
    status_code=status.HTTP_200_OK,
    summary="List Conversations",
    description="Returns all conversation sessions ordered by most recently updated.",
)
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=200, description="Max conversations to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    service: ConversationService = Depends(get_conversation_service),
) -> List[ConversationRecord]:
    """List persistent conversations ordered by updated_at descending asynchronously."""
    return await service.list_conversations_async(limit=limit, offset=offset)


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Conversation Details and Messages",
    description="Returns conversation metadata and its full chronological message history.",
)
async def get_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetail:
    """Retrieve conversation and its messages in chronological order asynchronously."""
    detail = await service.get_conversation_with_messages_async(conversation_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return detail


@router.patch(
    "/{conversation_id}",
    response_model=ConversationRecord,
    status_code=status.HTTP_200_OK,
    summary="Update Conversation Title",
    description="Updates the title of an existing conversation.",
)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationRecord:
    """Update conversation title asynchronously."""
    try:
        await service.update_conversation_title_async(conversation_id, title=request.title)
        conv = await service.get_conversation_async(conversation_id)
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        return conv
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Conversation",
    description="Deletes a conversation session and cascades deletion to all associated messages.",
)
async def delete_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> dict:
    """Delete conversation and cascade associated messages asynchronously."""
    try:
        await service.delete_conversation_async(conversation_id)
        return {"deleted": True, "conversation_id": conversation_id}
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.post(
    "/{conversation_id}/answer",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Conversation-Aware Answer",
    description="Executes grounded RAG question answering within the context of an existing conversation session.",
)
async def answer_conversation(
    conversation_id: str,
    request: ConversationAnswerRequest,
    service: ConversationService = Depends(get_conversation_service),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Generate answer within conversation, supporting both JSON and streaming modes with per-conversation lock."""
    # Verify conversation exists asynchronously
    conv = await service.get_conversation_async(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    rag_request = AnswerRequest(
        query=request.query,
        document_ids=request.document_ids,
        retrieval_config=request.retrieval_config,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        conversation_id=conversation_id,
    )

    if request.stream:
        return StreamingResponse(
            rag_service.answer_question_stream(rag_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async with rag_service.lock_manager.lock(conversation_id):
        try:
            return await asyncio.to_thread(rag_service.answer_question, rag_request)
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )


@router.post(
    "/{conversation_id}/answer/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream Conversation-Aware Answer via SSE",
    description="Streams grounded answer tokens progressively with conversation-aware query rewriting and persistent memory.",
)
async def answer_conversation_stream(
    conversation_id: str,
    request: ConversationAnswerRequest,
    service: ConversationService = Depends(get_conversation_service),
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """Stream conversation-aware answer tokens in real-time via Server-Sent Events (SSE)."""
    conv = await service.get_conversation_async(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    rag_request = AnswerRequest(
        query=request.query,
        document_ids=request.document_ids,
        retrieval_config=request.retrieval_config,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        conversation_id=conversation_id,
    )

    return StreamingResponse(
        rag_service.answer_question_stream(rag_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
