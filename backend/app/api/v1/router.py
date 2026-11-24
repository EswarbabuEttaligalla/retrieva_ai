"""API v1 root router aggregator."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import answer, conversations, documents, health, retrieval

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(documents.router, tags=["Documents"])
api_router.include_router(retrieval.router, tags=["Retrieval"])
api_router.include_router(answer.router, tags=["Answer Generation"])
api_router.include_router(conversations.router, tags=["Conversations"])
