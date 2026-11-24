"""Unit and integration tests for the health check endpoint."""

import pytest
from starlette.testclient import TestClient
from backend.app.core.config import get_settings


def test_root_endpoint(sync_client: TestClient):
    """Verify that root endpoint responds with metadata and 200 OK."""
    response = sync_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Retrieva AI"
    assert data["status"] == "online"
    assert "health_check" in data
    assert "docs_url" in data


def test_health_check_endpoint(sync_client: TestClient):
    """Verify health check endpoint returns 200, valid schema, and healthy status."""
    settings = get_settings()
    response = sync_client.get(f"{settings.API_V1_STR}/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == settings.PROJECT_NAME
    assert data["version"] == settings.APP_VERSION
    assert data["environment"] == settings.APP_ENV
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_async_health_check(async_client):
    """Verify health check responds appropriately under async client."""
    settings = get_settings()
    response = await async_client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == settings.APP_VERSION
