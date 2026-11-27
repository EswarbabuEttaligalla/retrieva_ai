"""Owner-scoped repository for Supabase public.documents records."""

import inspect
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict

from backend.app.core.auth import AuthenticatedUser
from backend.app.core.config import Settings
from backend.app.core.rpc_auth import (
    RPC_PROOF_TTL_SECONDS,
    build_backend_rpc_proof,
    compute_create_document_payload_digest,
    compute_delete_document_payload_digest,
    compute_update_status_payload_digest,
    resolve_backend_rpc_hmac_secret,
)
from backend.app.core.supabase_client import (
    SupabaseClient,
    SupabaseClientError,
    SupabaseConfigError,
    SupabaseForbiddenError,
    SupabaseValidationError,
    get_supabase_client,
    raise_for_supabase_error,
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
CANONICAL_MIME_BY_EXTENSION = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}
SUPPORTED_PROCESSING_STATUSES = {
    "pending",
    "uploading",
    "indexing",
    "ready",
    "failed",
    "deleting",
}
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


class DocumentRecord(BaseModel):
    """Domain model mirroring a row in Supabase public.documents."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    owner_id: UUID
    storage_path: str
    filename: str
    file_extension: str
    mime_type: str
    content_hash: str
    file_size_bytes: int
    processing_status: str
    embedding_model: Optional[str] = None
    total_pages: int = 0
    total_chunks: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentRepository:
    """Owner-scoped repository for public.documents.

    Reads execute directly through PostgREST under caller RLS (`SELECT`).
    Writes execute through narrowly scoped SECURITY DEFINER RPCs that validate
    `auth.uid()` inside PostgreSQL because direct table INSERT/UPDATE/DELETE
    grants are intentionally withheld from `authenticated`.
    """

    def __init__(self, client: SupabaseClient, rpc_hmac_secret: Optional[str] = None) -> None:
        self._client = client
        self._rpc_hmac_secret_override = rpc_hmac_secret

    def __repr__(self) -> str:
        return f"DocumentRepository(owner_id={self._client.owner_id})"

    def _require_rpc_hmac_secret(self) -> str:
        return resolve_backend_rpc_hmac_secret(self._rpc_hmac_secret_override)

    def _verify_owner_context(self, owner_id: UUID) -> None:
        if owner_id != self._client.owner_id:
            raise SupabaseForbiddenError(
                "Requested owner context does not match authenticated caller.",
                status_code=403,
                code="owner_mismatch",
            )

    @staticmethod
    def _extract_single_row(result: Any) -> Optional[Dict[str, Any]]:
        if isinstance(result, list):
            return result[0] if result else None
        if isinstance(result, dict):
            return result
        return None

    async def create_document(
        self,
        owner_id: UUID,
        filename: str,
        file_extension: str,
        mime_type: str,
        content_hash: str,
        file_size_bytes: int,
        storage_path: Optional[str] = None,
        document_id: Optional[UUID] = None,
        processing_status: str = "pending",
        embedding_model: Optional[str] = None,
    ) -> DocumentRecord:
        """Create a new owner-scoped document metadata row in public.documents."""
        self._verify_owner_context(owner_id)

        clean_filename = Path((filename or "").replace("\0", "")).name.strip()
        if not clean_filename or len(clean_filename) > 255:
            raise SupabaseValidationError("Document filename must be 1-255 characters.", status_code=400)

        ext = (file_extension or "").strip().lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise SupabaseValidationError(f"Unsupported file extension '{ext}'.", status_code=400)

        if not clean_filename.lower().endswith(ext):
            raise SupabaseValidationError("Filename suffix must match file_extension.", status_code=400)

        expected_mime = CANONICAL_MIME_BY_EXTENSION[ext]
        normalized_mime = (mime_type or "").split(";")[0].strip().lower()
        if normalized_mime != expected_mime:
            raise SupabaseValidationError(
                f"Canonical MIME type for '{ext}' must be '{expected_mime}'.",
                status_code=400,
            )

        clean_hash = (content_hash or "").strip().lower()
        if not SHA256_HEX_RE.match(clean_hash):
            raise SupabaseValidationError("content_hash must be a valid 64-char lowercase SHA-256 hex digest.", status_code=400)

        if file_size_bytes < 1 or file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise SupabaseValidationError("file_size_bytes must be between 1 and 10485760.", status_code=400)

        if processing_status not in SUPPORTED_PROCESSING_STATUSES:
            raise SupabaseValidationError(f"Unsupported processing_status '{processing_status}'.", status_code=400)

        doc_uuid = document_id or uuid4()
        resolved_storage_path = (storage_path or f"{owner_id}/{doc_uuid}/{clean_filename}").strip()
        if not resolved_storage_path.startswith(f"{owner_id}/"):
            raise SupabaseValidationError("storage_path must be scoped under the owner's UUID prefix.", status_code=400)

        secret = self._require_rpc_hmac_secret()
        payload_digest = compute_create_document_payload_digest(
            storage_path=resolved_storage_path,
            filename=clean_filename,
            file_extension=ext,
            mime_type=expected_mime,
            content_hash=clean_hash,
            file_size_bytes=file_size_bytes,
            processing_status=processing_status,
            embedding_model=embedding_model,
        )
        proof, nonce, expires_at = build_backend_rpc_proof(
            secret=secret,
            owner_id=owner_id,
            operation="create_user_document",
            document_id=doc_uuid,
            payload_digest=payload_digest,
        )

        payload: Dict[str, Any] = {
            "p_document_id": str(doc_uuid),
            "p_storage_path": resolved_storage_path,
            "p_filename": clean_filename,
            "p_file_extension": ext,
            "p_mime_type": expected_mime,
            "p_content_hash": clean_hash,
            "p_file_size_bytes": file_size_bytes,
            "p_processing_status": processing_status,
            "p_embedding_model": embedding_model,
            "p_backend_proof": proof,
            "p_proof_nonce": nonce,
            "p_proof_expires_at": expires_at,
        }
        result = await self._client.rpc("create_user_document", payload)
        row = self._extract_single_row(result)
        if not row:
            raise SupabaseValidationError("Document creation RPC returned no record.", status_code=500)
        return DocumentRecord.model_validate(row)

    async def get_document(self, document_id: UUID, owner_id: UUID) -> Optional[DocumentRecord]:
        """Fetch a single document owned by owner_id; returns None if missing or foreign."""
        self._verify_owner_context(owner_id)
        rows = await self._client.select(
            "documents",
            params={
                "id": f"eq.{document_id}",
                "owner_id": f"eq.{owner_id}",
                "limit": "1",
            },
        )
        if not rows:
            return None
        return DocumentRecord.model_validate(rows[0])

    async def get_document_by_hash(self, content_hash: str, owner_id: UUID) -> Optional[DocumentRecord]:
        """Fetch an existing document for owner_id by SHA-256 content_hash."""
        self._verify_owner_context(owner_id)
        clean_hash = (content_hash or "").strip().lower()
        if not SHA256_HEX_RE.match(clean_hash):
            return None
        rows = await self._client.select(
            "documents",
            params={
                "owner_id": f"eq.{owner_id}",
                "content_hash": f"eq.{clean_hash}",
                "limit": "1",
            },
        )
        if not rows:
            return None
        return DocumentRecord.model_validate(rows[0])

    async def list_documents(
        self,
        owner_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentRecord]:
        """List all documents owned by owner_id ordered newest first."""
        self._verify_owner_context(owner_id)
        rows = await self._client.select(
            "documents",
            params={
                "owner_id": f"eq.{owner_id}",
                "order": "created_at.desc",
                "limit": str(limit),
                "offset": str(offset),
            },
        )
        return [DocumentRecord.model_validate(row) for row in rows]

    async def list_ready_document_ids(self, owner_id: UUID) -> Set[str]:
        """Return the set of document UUID strings owned by owner_id with status 'ready'."""
        self._verify_owner_context(owner_id)
        rows = await self._client.select(
            "documents",
            params={
                "owner_id": f"eq.{owner_id}",
                "processing_status": "eq.ready",
                "select": "id,processing_status",
            },
        )
        return {
            str(row["id"])
            for row in rows
            if row.get("id") and (row.get("processing_status") in {None, "ready"})
        }

    async def update_document_status(
        self,
        document_id: UUID,
        owner_id: UUID,
        status: str,
        total_pages: Optional[int] = None,
        total_chunks: Optional[int] = None,
        embedding_model: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[DocumentRecord]:
        """Update processing status/counters for an owned document; returns None if not found/foreign."""
        self._verify_owner_context(owner_id)
        if status not in SUPPORTED_PROCESSING_STATUSES:
            raise SupabaseValidationError(f"Unsupported processing_status '{status}'.", status_code=400)

        truncated_error = error_message[:500] if error_message else None
        secret = self._require_rpc_hmac_secret()
        payload_digest = compute_update_status_payload_digest(
            processing_status=status,
            total_pages=total_pages,
            total_chunks=total_chunks,
            embedding_model=embedding_model,
            error_message=truncated_error,
        )
        proof, nonce, expires_at = build_backend_rpc_proof(
            secret=secret,
            owner_id=owner_id,
            operation="update_user_document_status",
            document_id=document_id,
            payload_digest=payload_digest,
        )

        payload: Dict[str, Any] = {
            "p_document_id": str(document_id),
            "p_processing_status": status,
            "p_total_pages": total_pages,
            "p_total_chunks": total_chunks,
            "p_embedding_model": embedding_model,
            "p_error_message": truncated_error,
            "p_backend_proof": proof,
            "p_proof_nonce": nonce,
            "p_proof_expires_at": expires_at,
        }
        result = await self._client.rpc("update_user_document_status", payload)
        row = self._extract_single_row(result)
        if not row:
            return None
        return DocumentRecord.model_validate(row)

    async def delete_document(self, document_id: UUID, owner_id: UUID) -> bool:
        """Delete an owned document record; returns False if missing or owned by another user."""
        self._verify_owner_context(owner_id)
        secret = self._require_rpc_hmac_secret()
        payload_digest = compute_delete_document_payload_digest(document_id)
        proof, nonce, expires_at = build_backend_rpc_proof(
            secret=secret,
            owner_id=owner_id,
            operation="delete_user_document",
            target_id=document_id,
            payload_digest=payload_digest,
        )
        result = await self._client.rpc(
            "delete_user_document",
            {
                "p_document_id": str(document_id),
                "p_backend_proof": proof,
                "p_proof_nonce": nonce,
                "p_proof_expires_at": expires_at,
            },
        )
        row = self._extract_single_row(result)
        return bool(row)


def get_document_repository(
    client: SupabaseClient = Depends(get_supabase_client),
) -> DocumentRepository:
    """FastAPI dependency providing a request-scoped DocumentRepository."""
    return DocumentRepository(client=client)


async def resolve_ready_document_ids_for_request(
    http_request: Request,
    current_user: AuthenticatedUser,
    credentials: Optional[HTTPAuthorizationCredentials],
    settings: Settings,
) -> Optional[Set[str]]:
    """Fetch owner-scoped 'ready' document IDs from Supabase, failing closed on errors.

    Returns None only in unauthenticated/dev test mode when no DocumentRepository
    override and no Bearer credentials are present.
    """
    doc_repo: Optional[DocumentRepository] = None
    override_repo_fn = http_request.app.dependency_overrides.get(get_document_repository)
    if override_repo_fn is not None:
        candidate = override_repo_fn()
        doc_repo = await candidate if inspect.isawaitable(candidate) else candidate
    elif (
        credentials is not None
        and credentials.scheme.lower() == "bearer"
        and (credentials.credentials or "").strip()
        and settings.SUPABASE_URL
        and settings.SUPABASE_ANON_KEY
    ):
        try:
            client = SupabaseClient(
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
        doc_repo = DocumentRepository(client=client)

    if doc_repo is None:
        if settings.AUTH_ENABLED and settings.APP_ENV.strip().lower() != "development":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document authorization service is unavailable.",
            )
        return None

    try:
        return await doc_repo.list_ready_document_ids(current_user.user_id)
    except SupabaseClientError as exc:
        raise_for_supabase_error(exc, "Failed to verify authorized ready documents.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document authorization verification is temporarily unavailable.",
        ) from exc
