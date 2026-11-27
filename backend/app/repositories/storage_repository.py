"""Owner-scoped repository for Supabase private Storage bucket operations."""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote
from uuid import UUID

from fastapi import Depends

from backend.app.core.config import Settings, get_settings
from backend.app.core.supabase_client import (
    SupabaseClient,
    SupabaseForbiddenError,
    SupabaseNotFoundError,
    SupabaseValidationError,
    get_supabase_client,
)

PRIVATE_BUCKET_NAME = "retrieva-user-documents"
MAX_STORAGE_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB matching bucket & schema limits
ALLOWED_STORAGE_EXTENSIONS = {".pdf", ".txt"}
ALLOWED_INPUT_MIME_TYPES = {
    ".pdf": {
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
        "applications/vnd.pdf",
        "text/pdf",
    },
    ".txt": {
        "text/plain",
        "text/markdown",
        "text/csv",
    },
}
CANONICAL_STORAGE_MIME = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}
_SAFE_FILENAME_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


class SupabaseStorageRepository:
    """Owner-scoped operations against the private `retrieva-user-documents` bucket.

    Enforces `{owner_id}/{document_id}/{sanitized_filename}` path scoping,
    caller-JWT authentication, and strict file content/type validation.
    """

    def __init__(
        self,
        client: SupabaseClient,
        bucket_name: str = PRIVATE_BUCKET_NAME,
        max_file_bytes: int = MAX_STORAGE_FILE_BYTES,
    ) -> None:
        self._client = client
        self.bucket_name = bucket_name
        self.max_file_bytes = max_file_bytes

    def _verify_owner_context(self, owner_id: UUID) -> None:
        if owner_id != self._client.owner_id:
            raise SupabaseForbiddenError(
                "Requested owner context does not match authenticated caller.",
                status_code=403,
                code="owner_mismatch",
            )

    @staticmethod
    def sanitize_filename(raw_filename: str) -> str:
        """Sanitize filename to a single safe basename with preserved extension."""
        cleaned = Path((raw_filename or "").replace("\0", "")).name.strip()
        if not cleaned or cleaned in {".", ".."}:
            raise SupabaseValidationError("A valid document filename is required.", status_code=400)

        stem = Path(cleaned).stem
        ext = Path(cleaned).suffix.lower()
        safe_stem = _SAFE_FILENAME_CHARS_RE.sub("_", stem).strip("._-") or "document"
        safe_name = f"{safe_stem}{ext}"
        if len(safe_name) > 255:
            max_stem = 255 - len(ext)
            safe_name = f"{safe_stem[:max_stem]}{ext}"
        return safe_name

    def validate_file(
        self,
        file_bytes: bytes,
        filename: str,
        declared_mime: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Validate extension, declared MIME, size, and actual bytes; return (safe_name, ext, canonical_mime)."""
        safe_name = self.sanitize_filename(filename)
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_STORAGE_EXTENSIONS:
            raise SupabaseValidationError(
                f"Unsupported file format '{ext}'. Allowed extensions: {sorted(ALLOWED_STORAGE_EXTENSIONS)}.",
                status_code=400,
            )

        if declared_mime:
            mime_base = declared_mime.split(";")[0].strip().lower()
            if mime_base and mime_base not in ALLOWED_INPUT_MIME_TYPES[ext]:
                raise SupabaseValidationError(
                    f"Unsupported media type '{declared_mime}' for '{ext}' documents.",
                    status_code=415,
                )

        if not file_bytes:
            raise SupabaseValidationError("Uploaded file is empty (0 bytes).", status_code=400)
        if len(file_bytes) > self.max_file_bytes:
            raise SupabaseValidationError(
                f"File size exceeds the maximum allowed limit of {self.max_file_bytes // (1024 * 1024)} MB.",
                status_code=413,
            )

        if ext == ".pdf":
            if not file_bytes.startswith(b"%PDF"):
                raise SupabaseValidationError(
                    "Invalid PDF file format: missing '%PDF' header.",
                    status_code=400,
                )
        elif ext == ".txt":
            if b"\x00" in file_bytes:
                raise SupabaseValidationError(
                    "Invalid TXT file format: binary null bytes detected.",
                    status_code=400,
                )
            try:
                file_bytes.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise SupabaseValidationError(
                    "Invalid TXT file format: text must be valid UTF-8 encoded.",
                    status_code=400,
                ) from exc

        return safe_name, ext, CANONICAL_STORAGE_MIME[ext]

    @classmethod
    def build_storage_path(cls, owner_id: UUID, document_id: UUID, filename: str) -> str:
        """Build strict `{owner_id}/{document_id}/{safe_filename}` Storage key."""
        safe_name = cls.sanitize_filename(filename)
        return f"{owner_id}/{document_id}/{safe_name}"

    def _validate_owned_storage_path(self, owner_id: UUID, storage_path: str) -> str:
        """Ensure storage_path has no traversal segments and belongs to owner_id prefix."""
        self._verify_owner_context(owner_id)
        normalized = (storage_path or "").strip().lstrip("/")
        parts = normalized.split("/")
        if (
            not normalized
            or "\\" in normalized
            or ".." in parts
            or "." in parts
            or len(parts) < 3
            or parts[0] != str(owner_id)
        ):
            raise SupabaseForbiddenError(
                "Storage path is invalid or outside the authenticated owner's namespace.",
                status_code=403,
                code="invalid_storage_path",
            )
        return normalized

    def _object_url(self, storage_path: str, prefix: str = "object") -> str:
        encoded_path = quote(storage_path, safe="/")
        return f"{self._client.storage_base_url}/{prefix}/{self.bucket_name}/{encoded_path}"

    async def upload_document(
        self,
        owner_id: UUID,
        document_id: UUID,
        filename: str,
        file_bytes: bytes,
        declared_mime: Optional[str] = None,
        upsert: bool = False,
    ) -> Dict[str, Any]:
        """Validate and upload document bytes into the caller's private Storage prefix."""
        self._verify_owner_context(owner_id)
        safe_name, ext, canonical_mime = self.validate_file(
            file_bytes=file_bytes,
            filename=filename,
            declared_mime=declared_mime,
        )
        storage_path = f"{owner_id}/{document_id}/{safe_name}"
        url = self._object_url(storage_path)

        response = await self._client.request_raw(
            method="POST",
            absolute_url=url,
            content=file_bytes,
            extra_headers={
                "Content-Type": canonical_mime,
                "x-upsert": "true" if upsert else "false",
            },
            accept="application/json",
        )
        body = {}
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            pass

        return {
            "storage_path": storage_path,
            "filename": safe_name,
            "file_extension": ext,
            "mime_type": canonical_mime,
            "file_size_bytes": len(file_bytes),
            "key": body.get("Key", f"{self.bucket_name}/{storage_path}"),
        }

    async def download_document(self, owner_id: UUID, storage_path: str) -> bytes:
        """Download document bytes from the caller's private Storage prefix."""
        scoped_path = self._validate_owned_storage_path(owner_id, storage_path)
        url = self._object_url(scoped_path, prefix="object/authenticated")
        response = await self._client.request_raw(
            method="GET",
            absolute_url=url,
            accept="*/*",
        )
        return response.content

    async def delete_document(self, owner_id: UUID, storage_path: str) -> bool:
        """Delete a Storage object inside the caller's prefix (idempotent on missing object)."""
        scoped_path = self._validate_owned_storage_path(owner_id, storage_path)
        url = self._object_url(scoped_path)
        try:
            await self._client.request_raw(
                method="DELETE",
                absolute_url=url,
                accept="application/json",
            )
            return True
        except SupabaseNotFoundError:
            return False

    async def get_document_metadata(
        self,
        owner_id: UUID,
        storage_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve private Storage object metadata for an owned path; returns None if absent."""
        scoped_path = self._validate_owned_storage_path(owner_id, storage_path)
        url = self._object_url(scoped_path, prefix="object/info/authenticated")
        try:
            response = await self._client.request_raw(
                method="GET",
                absolute_url=url,
                accept="application/json",
            )
            parsed = response.json()
            return parsed if isinstance(parsed, dict) else None
        except SupabaseNotFoundError:
            return None


def get_storage_repository(
    client: SupabaseClient = Depends(get_supabase_client),
    settings: Settings = Depends(get_settings),
) -> SupabaseStorageRepository:
    """FastAPI dependency providing a request-scoped SupabaseStorageRepository."""
    return SupabaseStorageRepository(
        client=client,
        bucket_name=settings.SUPABASE_STORAGE_BUCKET,
        max_file_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
    )
