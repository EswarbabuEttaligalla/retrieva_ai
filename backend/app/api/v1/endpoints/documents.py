"""Tenant-isolated document management endpoints for Retrieva AI.

Provides REST APIs for uploading documents to private Supabase Storage,
persisting metadata in Supabase public.documents, indexing owner-scoped chunks
into ChromaDB, listing user-owned documents, and atomically deleting documents.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.core.auth import AuthenticatedUser, get_current_user
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.core.supabase_client import (
    SupabaseClientError,
    SupabaseConflictError,
    SupabaseForbiddenError,
    SupabaseNotFoundError,
    SupabaseUnauthorizedError,
    SupabaseUnavailableError,
    SupabaseValidationError,
)
from backend.app.document_processing.exceptions import (
    CorruptedPDFError,
    EmptyDocumentError,
    EncodingError,
    EncryptedPDFError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from backend.app.repositories.document_repository import (
    DocumentRepository,
    get_document_repository,
)
from backend.app.repositories.storage_repository import (
    SupabaseStorageRepository,
    get_storage_repository,
)
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
)
from backend.app.services.ingestion_service import (
    DocumentInProgressError,
    IngestionService,
    get_ingestion_service,
)
from backend.app.vector_store.store import VectorStoreInsertionError

router = APIRouter(prefix="/documents")
logger = get_logger("api.endpoints.documents")


ALLOWED_MIME_TYPES = {
    ".pdf": [
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
        "applications/vnd.pdf",
        "text/pdf",
    ],
    ".txt": [
        "text/plain",
        "text/markdown",
        "text/csv",
    ],
}


def _raise_for_supabase_error(exc: SupabaseClientError, fallback_detail: str) -> None:
    if isinstance(exc, SupabaseUnauthorizedError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    if isinstance(exc, SupabaseForbiddenError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=exc.message,
        ) from exc
    if isinstance(exc, SupabaseConflictError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        ) from exc
    if isinstance(exc, SupabaseValidationError):
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    if isinstance(exc, SupabaseNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        ) from exc
    if isinstance(exc, SupabaseUnavailableError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=fallback_detail,
    ) from exc


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and Index Document",
    description="Accepts PDF or TXT documents, stores the raw file in private Supabase Storage, and indexes owner-scoped chunks into ChromaDB.",
)
async def upload_document(
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    retriever: Retriever = Depends(get_retriever),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    storage_repo: SupabaseStorageRepository = Depends(get_storage_repository),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> DocumentUploadResponse:
    """Handle tenant-isolated document upload, Storage persistence, and vector indexing."""
    settings = get_settings()

    # 1. Filename & Path Traversal Protection
    raw_filename = file.filename or ""
    clean_filename = Path(raw_filename).name.replace("\0", "").strip()
    if not clean_filename or clean_filename in {".", ".."}:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid document filename is required.",
        )

    # 2. Extension Allowlist
    file_ext = Path(clean_filename).suffix.lower()
    if file_ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        await file.close()
        logger.warning("Rejected file '%s': unsupported extension '%s'", clean_filename, file_ext)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Retrieva AI supports {settings.ALLOWED_UPLOAD_EXTENSIONS}.",
        )

    # 3. MIME Content-Type Validation
    if file.content_type:
        mime = file.content_type.split(";")[0].strip().lower()
        allowed_mimes = ALLOWED_MIME_TYPES.get(file_ext, [])
        if mime and mime not in allowed_mimes:
            await file.close()
            logger.warning(
                "Rejected file '%s': unsupported MIME type '%s' for extension '%s'",
                clean_filename,
                mime,
                file_ext,
            )
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported media type '{file.content_type}'. Expected {allowed_mimes} for '{file_ext}' documents.",
            )

    # 4. Streaming Content Reading & Proactive Size Enforcement
    chunk_size = 64 * 1024
    total_bytes = 0
    buffer_list = []
    max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)

    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                logger.warning(
                    "Rejected file '%s': size exceeded %d MB limit during streaming upload",
                    clean_filename,
                    max_mb,
                )
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File size exceeds the maximum allowed limit of {max_mb} MB.",
                )
            buffer_list.append(chunk)
        file_bytes = b"".join(buffer_list)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to read uploaded file '%s': %s", clean_filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        )
    finally:
        await file.close()

    # 5. Empty File Validation
    if total_bytes == 0:
        logger.warning("Rejected file '%s': empty file (0 bytes)", clean_filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid document.",
        )

    # 6. Format & Magic Byte Validation
    if file_ext == ".pdf":
        if not file_bytes.startswith(b"%PDF"):
            logger.warning("Rejected file '%s': missing '%%PDF' header", clean_filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file format: missing '%PDF' header or file is corrupted.",
            )
    elif file_ext == ".txt":
        if b"\x00" in file_bytes:
            logger.warning("Rejected file '%s': binary null bytes detected", clean_filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid TXT file format: binary null bytes detected.",
            )
        try:
            file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            logger.warning("Rejected file '%s': invalid UTF-8 encoding: %s", clean_filename, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid TXT file format: text must be valid UTF-8 encoded ({str(exc)}).",
            )

    # 7. Execute Tenant-Isolated Ingestion Pipeline
    try:
        result = await ingestion_service.ingest_document_for_user(
            file_bytes=file_bytes,
            filename=clean_filename,
            owner_id=current_user.user_id,
            doc_repo=doc_repo,
            storage_repo=storage_repo,
            declared_mime=file.content_type,
        )
    except DocumentInProgressError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except SupabaseClientError as exc:
        _raise_for_supabase_error(exc, "Failed to persist document metadata or storage object.")
    except (
        EmptyDocumentError,
        EncryptedPDFError,
        CorruptedPDFError,
        NoExtractableTextError,
        UnsupportedFileTypeError,
        EncodingError,
    ) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except VectorStoreInsertionError as exc:
        logger.error("ChromaDB vector insertion failed for '%s': %s", clean_filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist document chunks in vector store: {str(exc)}",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected ingestion failure for '%s': %s", clean_filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while processing document: {str(exc)}",
        ) from exc

    # 8. Synchronize Owner-Scoped BM25 Index on Successful Ingestion
    if (
        result.status == "success"
        and result.chunks
        and ingestion_service.tenant_bm25_manager is not retriever.tenant_bm25_manager
    ):
        try:
            added_bm25 = retriever.sync_owner_chunks(current_user.user_id, result.chunks)
            logger.info(
                "Synchronized owner BM25 index (%d chunks) after ingesting '%s'",
                added_bm25,
                result.document_id,
            )
        except Exception as exc:
            logger.warning("BM25 sync after ingestion warning: %s", exc)

    if result.is_duplicate or result.status == "duplicate":
        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            content_hash=result.content_hash,
            file_size_bytes=len(file_bytes),
            total_pages=result.total_pages,
            total_chunks=result.total_chunks,
            status="duplicate",
            created_at=result.metadata.created_at if result.metadata else datetime.now(timezone.utc),
            is_duplicate=True,
            message=f"Document '{result.filename}' with identical SHA-256 hash is already indexed in your knowledge base.",
        )

    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        content_hash=result.content_hash,
        file_size_bytes=len(file_bytes),
        total_pages=result.total_pages,
        total_chunks=result.total_chunks,
        status="success",
        created_at=result.metadata.created_at if result.metadata else datetime.now(timezone.utc),
        is_duplicate=False,
        message=f"Document '{result.filename}' was successfully indexed into {result.total_chunks} chunks.",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Ingested Documents",
    description="Retrieve catalog of all ready documents owned by the authenticated user.",
)
async def list_documents(
    doc_repo: DocumentRepository = Depends(get_document_repository),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> DocumentListResponse:
    """Return only ready documents belonging to the authenticated user."""
    try:
        docs = await doc_repo.list_documents(owner_id=current_user.user_id)
    except SupabaseClientError as exc:
        _raise_for_supabase_error(exc, "Failed to retrieve document catalog.")

    ready_docs = [doc for doc in docs if doc.processing_status == "ready"]
    items = [
        DocumentListItem(
            document_id=str(doc.id),
            filename=doc.filename,
            file_extension=doc.file_extension,
            file_size_bytes=doc.file_size_bytes,
            total_pages=doc.total_pages,
            total_chunks=doc.total_chunks,
            created_at=doc.created_at,
        )
        for doc in ready_docs
    ]
    return DocumentListResponse(documents=items, total=len(items))


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Permanently delete an owned document from ChromaDB, Supabase Storage, and Supabase metadata.",
)
async def delete_document(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    retriever: Retriever = Depends(get_retriever),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    storage_repo: SupabaseStorageRepository = Depends(get_storage_repository),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> DocumentDeleteResponse:
    """Delete an owned document by UUID; returns 404 for nonexistent or foreign documents."""
    clean_id = (document_id or "").strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID cannot be empty.",
        )

    try:
        doc_uuid = UUID(clean_id)
    except ValueError as exc:
        # Uniform 404 avoids leaking ID structure or existence across tenants
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        ) from exc

    try:
        deleted_chunks = await ingestion_service.delete_document_for_user(
            document_id=doc_uuid,
            owner_id=current_user.user_id,
            doc_repo=doc_repo,
            storage_repo=storage_repo,
        )
    except SupabaseClientError as exc:
        _raise_for_supabase_error(exc, "Failed to delete document.")
    except Exception as exc:
        logger.error("Failed deleting document '%s' for owner '%s': %s", doc_uuid, current_user.user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document cleanly; please retry.",
        ) from exc

    if deleted_chunks is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    try:
        removed_bm25 = retriever.remove_owner_document(current_user.user_id, str(doc_uuid))
        logger.info(
            "Synchronized owner BM25 index (removed %d chunks) after deleting document '%s'",
            removed_bm25,
            doc_uuid,
        )
    except Exception as exc:
        logger.error("BM25 sync after deletion failed for document '%s': %s", doc_uuid, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document was removed from persistent stores, but BM25 keyword index synchronization failed: {str(exc)}",
        ) from exc

    return DocumentDeleteResponse(
        document_id=str(doc_uuid),
        deleted_chunks=deleted_chunks,
        status="deleted",
        message=f"Document '{doc_uuid}' and all associated {deleted_chunks} chunks were successfully deleted.",
    )
