"""Health check endpoint definition."""

import logging
from fastapi import APIRouter, status
from backend.app.core.config import get_settings
from backend.app.schemas.health import HealthResponse

router = APIRouter()
logger = logging.getLogger("retrieva_ai.health")


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Status",
    description="Returns the current operational status, version, and environment of Retrieva AI.",
)
async def get_health() -> HealthResponse:
    """Return application health information."""
    settings = get_settings()
    logger.debug("Health check probe requested.")
    return HealthResponse(
        status="healthy",
        app_name=settings.PROJECT_NAME,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )
