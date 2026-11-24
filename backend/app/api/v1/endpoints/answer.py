"""Answer generation endpoint for Retrieva AI RAG pipeline."""

import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.app.conversations.service import ConversationNotFoundError, get_conversation_service
from backend.app.core.logging import get_logger
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

router = APIRouter()
logger = get_logger("api.endpoints.answer")


@router.post(
    "/answer",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Grounded RAG Answer",
    description="Accepts a natural language query, retrieves relevant document chunks, synthesizes a grounded answer, and returns verified source citations.",
)
async def generate_answer(
    request: AnswerRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> AnswerResponse:
    """Generate a context-grounded answer with verified source citations."""
    try:
        if request.conversation_id:
            async with rag_service.lock_manager.lock(request.conversation_id):
                return await asyncio.to_thread(rag_service.answer_question, request)
        else:
            return await asyncio.to_thread(rag_service.answer_question, request)

    except ValueError as exc:
        logger.warning("Invalid answer request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except ConversationNotFoundError as exc:
        logger.warning("Conversation not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    except LLMConfigError as exc:
        logger.error("LLM configuration error: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.user_message,
        )

    except LLMAuthenticationError as exc:
        logger.error("LLM authentication failure: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.user_message,
        )

    except LLMRateLimitError as exc:
        logger.warning("LLM rate limit reached: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=exc.user_message,
        )

    except LLMTimeoutError as exc:
        logger.error("LLM timeout: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=exc.user_message,
        )

    except LLMNetworkError as exc:
        logger.error("LLM network connection failure: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.user_message,
        )

    except LLMError as exc:
        logger.error("LLM answer generation failed: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.user_message,
        )

    except Exception as exc:
        logger.error("Unexpected error during RAG orchestration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while generating the answer.",
        )


@router.post(
    "/answer/stream",
    status_code=status.HTTP_200_OK,
    summary="Generate Grounded RAG Answer via Real-Time SSE Stream",
    description="Streams answer tokens progressively with SSE, performing citation provenance validation upon completion.",
)
async def generate_answer_stream(
    request: AnswerRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """Stream grounded answer tokens in real-time with post-stream citation validation."""
    if request.conversation_id:
        if not get_conversation_service().get_conversation(request.conversation_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{request.conversation_id}' not found.",
            )

    return StreamingResponse(
        rag_service.answer_question_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
