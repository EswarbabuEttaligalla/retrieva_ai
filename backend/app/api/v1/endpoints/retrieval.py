"""Standalone retrieval and semantic search endpoint for Retrieva AI.

Exposes the Phase 4 retrieval engine directly for inspection, testing,
and diagnostic relevance analysis without invoking the LLM generation pipeline.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from backend.app.core.auth import AuthenticatedUser, bearer_scheme, get_current_user
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.repositories.document_repository import resolve_ready_document_ids_for_request
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.schemas.retrieval import RetrievalRequest, RetrievalResult

router = APIRouter(prefix="/retrieval")
logger = get_logger("api.endpoints.retrieval")


@router.post(
    "/search",
    response_model=RetrievalResult,
    status_code=status.HTTP_200_OK,
    summary="Search Document Knowledge Base",
    description="Execute standalone Dense, BM25, or Hybrid retrieval with optional Cross-Encoder reranking.",
)
@router.post(
    "",
    response_model=RetrievalResult,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def search_documents(
    request: RetrievalRequest,
    http_request: Request,
    retriever: Retriever = Depends(get_retriever),
    current_user: AuthenticatedUser = Depends(get_current_user),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> RetrievalResult:
    """Execute standalone multi-stage document retrieval without LLM generation.

    Guarantees:
    - Never invokes the OpenAI API or external LLM providers.
    - Exposes candidate chunks, provenance (filename, page numbers, offsets), and scores.
    - Returns independent score metrics: cosine similarity/distance, BM25 score, RRF score,
      and Cross-Encoder score (null when not applicable for the selected mode).
    """
    clean_query = (request.query or "").strip()
    if not clean_query:
        logger.warning("Rejected retrieval search: empty query string")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace-only.",
        )

    ready_doc_ids = await resolve_ready_document_ids_for_request(
        http_request=http_request,
        current_user=current_user,
        credentials=credentials,
        settings=settings,
    )

    try:
        if ready_doc_ids is not None:
            result = retriever.retrieve(
                request,
                owner_id=current_user.user_id,
                authorized_ready_doc_ids=ready_doc_ids,
            )
        else:
            result = retriever.retrieve(request, owner_id=current_user.user_id)
        logger.info(
            "Standalone retrieval completed: mode=%s, query='%s', returned=%d chunks (took %.2f ms)",
            result.mode.value,
            clean_query[:50],
            result.returned_chunks_count,
            result.execution_time_ms,
        )
        return result

    except ValueError as exc:
        logger.warning("Invalid retrieval parameters: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception as exc:
        logger.error("Unexpected error during retrieval search: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing retrieval search: {str(exc)}",
        )
