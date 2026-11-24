"""Retrieva AI - Main Application Entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.router import api_router
from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging

settings = get_settings()
logger = setup_logging(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager handling startup and shutdown events."""
    logger.info("Starting up %s (version: %s, env: %s)...", settings.PROJECT_NAME, settings.APP_VERSION, settings.APP_ENV)
    logger.info("CORS allowed origins: %s", settings.BACKEND_CORS_ORIGINS)
    
    # Reconcile Document Registry and BM25 index with persistent ChromaDB store on startup
    try:
        from backend.app.services.ingestion_service import get_ingestion_service
        from backend.app.retrieval.retriever import get_retriever
        from backend.app.conversations.database import init_conversation_db
        svc = get_ingestion_service()
        svc.registry.reconcile_from_vector_store(svc.vector_store, prune_orphans=True)
        retriever = get_retriever()
        retriever.sync_bm25_with_vector_store()
        init_conversation_db()
        logger.info("Startup reconciliation complete: Registry, ChromaDB, BM25, and Conversation DB synchronized.")
    except Exception as exc:
        logger.warning("Startup reconciliation warning: %s", exc)

    yield
    
    logger.info("Shutting down %s...", settings.PROJECT_NAME)
    try:
        from backend.app.llm.client import OpenAIClient
        if OpenAIClient._instance is not None:
            await OpenAIClient._instance.aclose()
            logger.info("OpenAIClient async session closed.")
    except Exception as exc:
        logger.warning("Error during OpenAIClient async session shutdown: %s", exc)


def create_application() -> FastAPI:
    """Application factory for Retrieva AI FastAPI instance."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.APP_VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        lifespan=lifespan,
    )

    # Configure Cross-Origin Resource Sharing (CORS)
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.BACKEND_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Register API Routers
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/", summary="Root Status Endpoint", tags=["Root"])
    async def root() -> Dict[str, Any]:
        """Root endpoint displaying service metadata."""
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.APP_VERSION,
            "status": "online",
            "docs_url": f"{settings.API_V1_STR}/docs",
            "health_check": f"{settings.API_V1_STR}/health",
        }

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
