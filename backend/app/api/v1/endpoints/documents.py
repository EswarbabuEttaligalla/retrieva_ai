"""Document management endpoints for Retrieva AI.

Provides REST APIs for document uploading, listing ingested documents,
and deleting documents with atomic ChromaDB and BM25 index synchronization.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.document_processing.exceptions import (
    CorruptedPDFError,
    EmptyDocumentError,
    EncryptedPDFError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
    EncodingError,
)
from backend.app.retrieval.retriever import Retriever, get_retriever
from backend.app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
)
from backend.app.services.ingestion_service import (
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


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and Index Document",
    description="Accepts PDF or TXT documents, validates file integrity and size, extracts text, generates chunk embeddings, and indexes into persistent vector store.",
)
async def upload_document(
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    retriever: Retriever = Depends(get_retriever),
) -> DocumentUploadResponse:
    """Handle document file upload, validation, chunking, and vector indexing."""
    settings = get_settings()

    # 1. Filename & Path Traversal Protection
    raw_filename = file.filename or ""
    clean_filename = Path(raw_filename).name.replace("\0", "").strip()
    if not clean_filename or clean_filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid document filename is required.",
        )

    # 2. Extension Allowlist
    file_ext = Path(clean_filename).suffix.lower()
    if file_ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
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
    chunk_size = 64 * 1024  # 64 KB streaming buffer
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
        # Reject binary null bytes
        if b"\x00" in file_bytes:
            logger.warning("Rejected file '%s': binary null bytes detected", clean_filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid TXT file format: binary null bytes detected.",
            )
        # Enforce valid UTF-8 (supporting standard UTF-8 and UTF-8 with BOM)
        try:
            file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            logger.warning("Rejected file '%s': invalid UTF-8 encoding: %s", clean_filename, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid TXT file format: text must be valid UTF-8 encoded ({str(exc)}).",
            )

    # 7. Execute Ingestion Pipeline
    try:
        result = ingestion_service.ingest_document(
            file_input=file_bytes,
            filename=clean_filename,
            prevent_duplicates=True,
        )
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except EncryptedPDFError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except CorruptedPDFError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except NoExtractableTextError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except EncodingError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except VectorStoreInsertionError as exc:
        logger.error("ChromaDB vector insertion failed for '%s': %s", clean_filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist document chunks in vector store: {str(exc)}",
        )
    except Exception as exc:
        logger.error("Unexpected ingestion failure for '%s': %s", clean_filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while processing document: {str(exc)}",
        )

    # 6. Synchronize BM25 Index on Successful Ingestion
    if result.status == "success":
        try:
            retriever.sync_bm25_with_vector_store()
            logger.info("Synchronized BM25 index after ingesting '%s'", result.document_id)
        except Exception as exc:
            logger.warning("BM25 sync after ingestion warning: %s", exc)

    # 7. Formulate Response
    if result.is_duplicate or result.status == "duplicate":
        stored_chunks = 0
        if result.metadata and getattr(result.metadata, "total_chunks", 0):
            stored_chunks = result.metadata.total_chunks
        else:
            try:
                stored_chunks = len(ingestion_service.vector_store.get_document_chunks(result.document_id))
            except Exception:
                stored_chunks = 0

        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            content_hash=result.content_hash,
            file_size_bytes=len(file_bytes),
            total_pages=result.total_pages,
            total_chunks=stored_chunks,
            status="duplicate",
            created_at=result.metadata.created_at if result.metadata else datetime.now(timezone.utc),
            is_duplicate=True,
            message=f"Document '{clean_filename}' with identical SHA-256 hash is already indexed in the system.",
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
        message=f"Document '{clean_filename}' was successfully indexed into {result.total_chunks} chunks.",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Ingested Documents",
    description="Retrieve catalog of all documents currently ingested, including file size, page count, and chunk counts.",
)
async def list_documents(
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> DocumentListResponse:
    """Return all registered documents in the knowledge base."""
    # Ensure in-memory registry is reconciled with ChromaDB if needed
    if ingestion_service.registry.count() == 0 and ingestion_service.vector_store.count() > 0:
        ingestion_service.registry.reconcile_from_vector_store(ingestion_service.vector_store)

    docs = ingestion_service.registry.get_all()
    items = [
        DocumentListItem(
            document_id=doc.document_id,
            filename=doc.filename,
            file_extension=doc.file_extension,
            file_size_bytes=doc.file_size_bytes,
            total_pages=doc.total_pages,
            total_chunks=doc.total_chunks,
            created_at=doc.created_at,
        )
        for doc in docs
    ]
    return DocumentListResponse(documents=items, total=len(items))


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Permanently delete a document and its vector chunks from the vector store and registry, synchronizing BM25 index.",
)
async def delete_document(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    retriever: Retriever = Depends(get_retriever),
) -> DocumentDeleteResponse:
    """Delete a document by its document_id."""
    clean_id = document_id.strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID cannot be empty.",
        )

    has_in_registry = bool(ingestion_service.registry.get_by_id(clean_id))
    has_in_store = ingestion_service.vector_store.has_document(clean_id)

    if not has_in_registry and not has_in_store:
        logger.warning("Attempted to delete nonexistent document '%s'", clean_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{clean_id}' not found.",
        )

    # Step 1: Delete chunks from ChromaDB persistent vector store
    try:
        deleted_chunks = ingestion_service.vector_store.delete_document(clean_id)
    except Exception as exc:
        logger.error("Failed to delete document '%s' from vector store: %s", clean_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document chunks from vector store: {str(exc)}",
        )

    # Step 2: Remove from Document Registry with disk persistence
    try:
        ingestion_service.registry.remove(clean_id, raise_on_error=True)
    except Exception as exc:
        logger.error(
            "ChromaDB chunks deleted for '%s', but metadata persistence failed: %s. Initiating reconciliation recovery...",
            clean_id,
            exc,
        )
        try:
            ingestion_service.registry.reconcile_from_vector_store(
                ingestion_service.vector_store, prune_orphans=True
            )
        except Exception as rec_exc:
            logger.error("Automated reconciliation recovery failed: %s", rec_exc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector chunks for '{clean_id}' were deleted, but metadata registry persistence failed: {str(exc)}. State reconciliation was initiated.",
        )

    # Step 3: Rebuild BM25 in-memory keyword index
    try:
        retriever.sync_bm25_with_vector_store()
        logger.info("Synchronized BM25 index after deleting document '%s'", clean_id)
    except Exception as exc:
        logger.error("BM25 sync after deletion failed for document '%s': %s", clean_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document was removed from persistent stores, but BM25 keyword index synchronization failed: {str(exc)}",
        )

    logger.info("Successfully deleted document '%s' (%d chunks removed)", clean_id, deleted_chunks)
    return DocumentDeleteResponse(
        document_id=clean_id,
        deleted_chunks=deleted_chunks,
        status="deleted",
        message=f"Document '{clean_id}' and all associated {deleted_chunks} chunks were successfully deleted.",
    )
