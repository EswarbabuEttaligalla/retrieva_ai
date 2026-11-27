"""Shared pytest fixtures for Retrieva AI backend tests."""

import os
from uuid import UUID

os.environ.setdefault("BACKEND_RPC_HMAC_SECRET", "pytest-backend-rpc-hmac-secret-0001")

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient

from backend.app.core.auth import AuthenticatedUser, get_current_user
from backend.app.main import app


@pytest.fixture(autouse=True)
def default_auth_override(request):
    """Provide a default authenticated user for existing non-auth test suites."""
    if (
        "test_auth" in request.node.nodeid
        or "test_tenant_isolation_m33" in request.node.nodeid
        or "test_tenant_bm25_m34" in request.node.nodeid
        or "test_supabase_conversations_m35" in request.node.nodeid
    ):
        yield
        return

    previous = app.dependency_overrides.get(get_current_user)
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id=UUID("11111111-1111-4111-8111-111111111111"),
        email="test@retrieva.local",
        role="authenticated",
    )
    try:
        yield
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_current_user, None)
        else:
            app.dependency_overrides[get_current_user] = previous


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
