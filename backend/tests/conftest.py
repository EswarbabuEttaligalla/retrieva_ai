"""Shared pytest fixtures for Retrieva AI backend tests."""

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="session")
def sync_client():
    """Synchronous test client fixture for standard endpoint testing."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client():
    """Asynchronous HTTP test client fixture for async endpoint testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
