"""Request-scoped asynchronous Supabase PostgREST client for Retrieva AI.

Uses a shared httpx.AsyncClient connection pool with strictly per-request
Authorization and apikey headers so concurrent users never share credentials.
Never uses a service-role key and never logs tokens, keys, or auth headers.
"""

import threading
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union
from uuid import UUID

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from backend.app.core.auth import AuthenticatedUser, bearer_scheme, get_current_user
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

logger = get_logger("core.supabase_client")

_pool_lock = threading.Lock()
_http_pool: Optional[httpx.AsyncClient] = None
_http_pool_timeout: Optional[float] = None


class SupabaseClientError(Exception):
    """Base exception for sanitized Supabase API and transport errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class SupabaseConfigError(SupabaseClientError):
    """Raised when required Supabase client configuration is missing."""

    def __init__(self, message: str = "Supabase client configuration is missing.") -> None:
        super().__init__(message=message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code="config_error")


class SupabaseUnauthorizedError(SupabaseClientError):
    """Raised when Supabase rejects caller authentication."""


class SupabaseForbiddenError(SupabaseClientError):
    """Raised when Supabase RLS or grants deny the requested operation."""


class SupabaseConflictError(SupabaseClientError):
    """Raised on unique constraint or conflict errors."""


class SupabaseValidationError(SupabaseClientError):
    """Raised when PostgreSQL check or foreign-key constraints reject input."""


class SupabaseNotFoundError(SupabaseClientError):
    """Raised when an expected RPC or resource endpoint is not found."""


class SupabaseUnavailableError(SupabaseClientError):
    """Raised on Supabase network, timeout, or upstream 5xx errors."""


def get_supabase_http_client(timeout_seconds: float = 10.0) -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient connection pool with NO auth headers."""
    global _http_pool, _http_pool_timeout
    with _pool_lock:
        if _http_pool is None or _http_pool.is_closed or _http_pool_timeout != timeout_seconds:
            _http_pool = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
            _http_pool_timeout = timeout_seconds
        return _http_pool


async def close_supabase_http_client() -> None:
    """Close the shared httpx.AsyncClient pool during FastAPI lifespan shutdown."""
    global _http_pool, _http_pool_timeout
    with _pool_lock:
        client = _http_pool
        _http_pool = None
        _http_pool_timeout = None
    if client is not None and not client.is_closed:
        await client.aclose()


def _parse_postgrest_error(response: httpx.Response) -> Tuple[Optional[str], str]:
    """Extract sanitized PostgREST error code and message from an error response."""
    code: Optional[str] = None
    message = f"Supabase request failed with status {response.status_code}."
    try:
        payload = response.json()
        if isinstance(payload, dict):
            raw_code = payload.get("code")
            code = str(raw_code) if raw_code is not None else None
            raw_msg = payload.get("message") or payload.get("error_description") or payload.get("error")
            if isinstance(raw_msg, str) and raw_msg.strip():
                message = raw_msg.strip()
    except Exception:
        pass
    return code, message


class SupabaseClient:
    """Immutable-credential, request-scoped Supabase PostgREST client."""

    def __init__(
        self,
        supabase_url: str,
        anon_key: str,
        access_token: str,
        owner_id: UUID,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        clean_url = (supabase_url or "").strip().rstrip("/")
        clean_anon = (anon_key or "").strip()
        clean_token = (access_token or "").strip()

        if not clean_url or not clean_anon:
            raise SupabaseConfigError("SUPABASE_URL and SUPABASE_ANON_KEY must be configured.")
        if not clean_token:
            raise SupabaseUnauthorizedError(
                "Missing caller access token for Supabase request.",
                status_code=status.HTTP_401_UNAUTHORIZED,
                code="missing_token",
            )

        self.owner_id: UUID = owner_id
        self.supabase_url: str = clean_url
        self.rest_base_url: str = f"{clean_url}/rest/v1"
        self.storage_base_url: str = f"{clean_url}/storage/v1"
        self._anon_key: str = clean_anon
        self._access_token: str = clean_token
        self._timeout_seconds: float = timeout_seconds
        self._http_client: httpx.AsyncClient = http_client or get_supabase_http_client(timeout_seconds)

    def __repr__(self) -> str:
        return f"SupabaseClient(owner_id={self.owner_id!s}, rest_base_url={self.rest_base_url!r})"

    __str__ = __repr__

    def _build_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Build fresh per-request headers; never stored on the shared pool client."""
        headers = {
            "apikey": self._anon_key,
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }
        if extra_headers:
            for key, value in extra_headers.items():
                if key.lower() in {"authorization", "apikey"}:
                    continue
                headers[key] = value
        return headers

    async def request(
        self,
        method: str,
        resource_path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Union[Dict[str, Any], List[Any]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Execute an authenticated PostgREST request under the caller's JWT."""
        url = f"{self.rest_base_url}/{resource_path.lstrip('/')}"
        headers = self._build_headers(extra_headers)
        if json_body is not None and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        try:
            response = await self._http_client.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            logger.warning("Supabase request timed out (%s %s).", method.upper(), resource_path)
            raise SupabaseUnavailableError(
                "Supabase database request timed out.",
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                code="timeout",
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("Supabase network error (%s %s): %s", method.upper(), resource_path, type(exc).__name__)
            raise SupabaseUnavailableError(
                "Supabase database service is temporarily unavailable.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="network_error",
            ) from exc

        if 200 <= response.status_code < 300:
            if response.status_code == 204 or not response.content:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise SupabaseUnavailableError(
                    "Supabase returned an invalid JSON response.",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    code="invalid_response",
                ) from exc

        pg_code, pg_msg = _parse_postgrest_error(response)
        logger.warning(
            "Supabase PostgREST error (%s %s -> HTTP %d, code=%s).",
            method.upper(),
            resource_path,
            response.status_code,
            pg_code,
        )

        if response.status_code == 401:
            raise SupabaseUnauthorizedError(pg_msg, status_code=401, code=pg_code)
        if response.status_code == 403 or pg_code == "42501":
            raise SupabaseForbiddenError(pg_msg, status_code=403, code=pg_code)
        if response.status_code == 409 or pg_code == "23505":
            raise SupabaseConflictError(pg_msg, status_code=409, code=pg_code)
        if response.status_code == 404 or pg_code == "PGRST202":
            raise SupabaseNotFoundError(pg_msg, status_code=404, code=pg_code)
        if response.status_code in {400, 422} or pg_code in {"23514", "23503", "22P02"}:
            raise SupabaseValidationError(pg_msg, status_code=400, code=pg_code)
        if response.status_code >= 500:
            raise SupabaseUnavailableError(pg_msg, status_code=502, code=pg_code)
        raise SupabaseClientError(pg_msg, status_code=response.status_code, code=pg_code)

    async def select(
        self,
        table: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        result = await self.request("GET", table, params=params)
        return result if isinstance(result, list) else []

    async def insert(
        self,
        table: str,
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        result = await self.request(
            "POST",
            table,
            json_body=payload,
            extra_headers={"Prefer": "return=representation"},
        )
        return result if isinstance(result, list) else []

    async def update(
        self,
        table: str,
        params: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        result = await self.request(
            "PATCH",
            table,
            params=params,
            json_body=payload,
            extra_headers={"Prefer": "return=representation"},
        )
        return result if isinstance(result, list) else []

    async def delete(
        self,
        table: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        result = await self.request(
            "DELETE",
            table,
            params=params,
            extra_headers={"Prefer": "return=representation"},
        )
        return result if isinstance(result, list) else []

    async def rpc(
        self,
        function_name: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return await self.request(
            "POST",
            f"rpc/{function_name}",
            json_body=payload or {},
        )

    async def request_raw(
        self,
        method: str,
        absolute_url: str,
        content: Optional[bytes] = None,
        json_body: Optional[Union[Dict[str, Any], List[Any]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        accept: str = "*/*",
    ) -> httpx.Response:
        """Execute an authenticated raw HTTP request (e.g., Supabase Storage) with per-request credentials."""
        headers = self._build_headers(extra_headers)
        headers["Accept"] = accept
        if json_body is not None and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        try:
            response = await self._http_client.request(
                method=method.upper(),
                url=absolute_url,
                content=content,
                json=json_body,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            logger.warning("Supabase raw request timed out (%s %s).", method.upper(), absolute_url.split("?")[0])
            raise SupabaseUnavailableError(
                "Supabase storage request timed out.",
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                code="timeout",
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("Supabase raw network error (%s): %s", method.upper(), type(exc).__name__)
            raise SupabaseUnavailableError(
                "Supabase storage service is temporarily unavailable.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="network_error",
            ) from exc

        if 200 <= response.status_code < 300:
            return response

        pg_code, pg_msg = _parse_postgrest_error(response)
        logger.warning(
            "Supabase Storage error (%s -> HTTP %d, code=%s).",
            method.upper(),
            response.status_code,
            pg_code,
        )
        if response.status_code == 401:
            raise SupabaseUnauthorizedError(pg_msg, status_code=401, code=pg_code)
        if response.status_code == 403 or pg_code == "42501":
            raise SupabaseForbiddenError(pg_msg, status_code=403, code=pg_code)
        if response.status_code == 409 or "duplicate" in pg_msg.lower() or "already exists" in pg_msg.lower():
            raise SupabaseConflictError(pg_msg, status_code=409, code=pg_code)
        if response.status_code == 404 or "not found" in pg_msg.lower():
            raise SupabaseNotFoundError(pg_msg, status_code=404, code=pg_code)
        if response.status_code in {400, 413, 415, 422}:
            raise SupabaseValidationError(pg_msg, status_code=response.status_code, code=pg_code)
        if response.status_code >= 500:
            raise SupabaseUnavailableError(pg_msg, status_code=502, code=pg_code)
        raise SupabaseClientError(pg_msg, status_code=response.status_code, code=pg_code)


async def get_supabase_client(
    current_user: AuthenticatedUser = Depends(get_current_user),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> SupabaseClient:
    """FastAPI dependency returning a request-scoped SupabaseClient for the authenticated caller."""
    if credentials is None or credentials.scheme.lower() != "bearer" or not (credentials.credentials or "").strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database client is not configured.",
        )

    try:
        return SupabaseClient(
            supabase_url=settings.SUPABASE_URL,
            anon_key=settings.SUPABASE_ANON_KEY,
            access_token=credentials.credentials.strip(),
            owner_id=current_user.user_id,
            timeout_seconds=settings.SUPABASE_REST_TIMEOUT_SECONDS,
        )
    except SupabaseConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        ) from exc


def raise_for_supabase_error(exc: SupabaseClientError, default_detail: str) -> NoReturn:
    """Translate sanitized SupabaseClientError subclasses into controlled HTTPException statuses."""
    if isinstance(exc, SupabaseUnauthorizedError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Supabase rejected caller authentication.",
        ) from exc
    if isinstance(exc, SupabaseForbiddenError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied by tenant ownership policy.",
        ) from exc
    if isinstance(exc, SupabaseValidationError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    if isinstance(exc, (SupabaseConfigError, SupabaseUnavailableError)):
        raise HTTPException(
            status_code=exc.status_code or status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=default_detail,
    ) from exc
