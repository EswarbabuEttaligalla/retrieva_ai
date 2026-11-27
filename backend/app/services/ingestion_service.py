"""Integrated tenant-aware ingestion service for Retrieva AI.

Coordinates Supabase DocumentRepository metadata, private Storage persistence,
document extraction, text cleaning, boundary-aware chunking, dense embeddings,
and mandatory owner-scoped ChromaDB vector persistence.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.document_processing.chunker import TextChunker
from backend.app.document_processing.cleaner import TextCleaner
from backend.app.document_processing.extractor import DocumentExtractor
from backend.app.document_processing.identifier import (
    DocumentRegistry,
    compute_content_hash,
)
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.repositories.document_repository import DocumentRecord, DocumentRepository
from backend.app.repositories.storage_repository import SupabaseStorageRepository
from backend.app.retrieval.tenant_bm25 import TenantBM25Manager
from backend.app.schemas.document import (
    DocumentChunk,
    DocumentMetadata,
    ExtractedPage,
    IngestionResult,
)
from backend.app.vector_store.store import ChromaVectorStore, normalize_owner_id

logger = get_logger("services.ingestion")


STALE_ELIGIBLE_STATUSES = {"uploading", "indexing"}


class DocumentInProgressError(Exception):
    """Raised when an identical document upload or deletion is already in progress."""


@dataclass(frozen=True)
class StaleRecoveryResult:
    """Observable outcome of a single stale-document recovery attempt."""

    document_id: str
    recovered: bool
    already_handled: bool = False
    vectors_deleted: int = 0
    storage_deleted: bool = False
    cleanup_errors: List[str] = field(default_factory=list)
    final_status: str = "failed"


class IngestionService:
    """Production service coordinating tenant-isolated ingestion and vector persistence."""

    def __init__(
        self,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        registry: Optional[DocumentRegistry] = None,
        tenant_bm25_manager: Optional[TenantBM25Manager] = None,
        chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = TextChunker.DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = embedder or SentenceTransformerEmbedder.get_instance()
        self.vector_store = vector_store or ChromaVectorStore()
        self.registry = registry
        self.tenant_bm25_manager = tenant_bm25_manager
        self._upload_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._recovery_locks: Dict[Tuple[str, str], asyncio.Lock] = {}

    def _get_upload_lock(self, owner_id: UUID, content_hash: str) -> asyncio.Lock:
        key = (str(owner_id), content_hash)
        lock = self._upload_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._upload_locks[key] = lock
        return lock

    def _get_recovery_lock(self, owner_id: UUID, document_id: UUID) -> asyncio.Lock:
        key = (str(owner_id), str(document_id))
        lock = self._recovery_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._recovery_locks[key] = lock
        return lock

    @staticmethod
    def is_stale_record(
        record: DocumentRecord,
        stale_timeout_seconds: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Return True iff record is in an in-progress ingestion state older than timeout."""
        if record.processing_status not in STALE_ELIGIBLE_STATUSES:
            return False
        timeout = stale_timeout_seconds if stale_timeout_seconds is not None else get_settings().STALE_INGESTION_TIMEOUT_SECONDS
        current_time = now or datetime.now(timezone.utc)
        updated_at = record.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return (current_time - updated_at).total_seconds() >= timeout

    async def recover_stale_document(
        self,
        document_id: UUID,
        owner_id: UUID,
        doc_repo: DocumentRepository,
        storage_repo: SupabaseStorageRepository,
        stale_timeout_seconds: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> StaleRecoveryResult:
        """Idempotently recover a single owner-scoped stale uploading/indexing document."""
        timeout = stale_timeout_seconds if stale_timeout_seconds is not None else get_settings().STALE_INGESTION_TIMEOUT_SECONDS
        lock = self._get_recovery_lock(owner_id, document_id)
        async with lock:
            fresh = await doc_repo.get_document(document_id, owner_id=owner_id)
            if fresh is None:
                return StaleRecoveryResult(
                    document_id=str(document_id),
                    recovered=False,
                    already_handled=True,
                    final_status="missing",
                )
            if not self.is_stale_record(fresh, stale_timeout_seconds=timeout, now=now):
                return StaleRecoveryResult(
                    document_id=str(document_id),
                    recovered=False,
                    already_handled=True,
                    final_status=fresh.processing_status,
                )

            prior_status = fresh.processing_status
            # Claim in DB first so concurrent workers observe 'failed' and do not double-recover
            claimed = await doc_repo.update_document_status(
                document_id,
                owner_id,
                status="failed",
                error_message=f"Stale '{prior_status}' ingestion timed out after {timeout}s; recovery in progress.",
            )
            if claimed is None:
                return StaleRecoveryResult(
                    document_id=str(document_id),
                    recovered=False,
                    already_handled=True,
                    final_status="missing",
                )

            cleanup_errors: List[str] = []
            vectors_deleted = 0
            storage_deleted = False

            try:
                vectors_deleted = await asyncio.to_thread(
                    self.vector_store.delete_document,
                    str(document_id),
                    owner_id,
                )
            except Exception as exc:
                logger.error("Stale recovery vector cleanup failed for '%s': %s", document_id, exc)
                cleanup_errors.append(f"chromadb:{type(exc).__name__}")

            if self.tenant_bm25_manager is not None:
                try:
                    self.tenant_bm25_manager.remove_document(owner_id, str(document_id))
                except Exception as exc:
                    logger.warning("Stale recovery BM25 cleanup failed for '%s': %s", document_id, exc)
                    cleanup_errors.append(f"bm25:{type(exc).__name__}")

            try:
                storage_deleted = await storage_repo.delete_document(owner_id, fresh.storage_path)
            except Exception as exc:
                logger.warning("Stale recovery storage cleanup failed for '%s': %s", fresh.storage_path, exc)
                cleanup_errors.append(f"storage:{type(exc).__name__}")

            final_msg = (
                f"Stale '{prior_status}' recovery partial failure: {','.join(cleanup_errors)}"
                if cleanup_errors
                else f"Recovered stale '{prior_status}' ingestion after {timeout}s timeout."
            )[:500]
            try:
                await doc_repo.update_document_status(
                    document_id,
                    owner_id,
                    status="failed",
                    error_message=final_msg,
                )
            except Exception as exc:
                cleanup_errors.append(f"metadata_status:{type(exc).__name__}")
                logger.warning("Failed updating final recovery error_message for '%s': %s", document_id, exc)

            return StaleRecoveryResult(
                document_id=str(document_id),
                recovered=len(cleanup_errors) == 0,
                already_handled=False,
                vectors_deleted=vectors_deleted,
                storage_deleted=storage_deleted,
                cleanup_errors=cleanup_errors,
                final_status="failed",
            )

    async def recover_stale_documents_for_owner(
        self,
        owner_id: UUID,
        doc_repo: DocumentRepository,
        storage_repo: SupabaseStorageRepository,
        stale_timeout_seconds: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> List[StaleRecoveryResult]:
        """Scan and recover all stale uploading/indexing documents belonging to owner_id."""
        docs = await doc_repo.list_documents(owner_id=owner_id)
        results: List[StaleRecoveryResult] = []
        for doc in docs:
            if self.is_stale_record(doc, stale_timeout_seconds=stale_timeout_seconds, now=now):
                res = await self.recover_stale_document(
                    document_id=doc.id,
                    owner_id=owner_id,
                    doc_repo=doc_repo,
                    storage_repo=storage_repo,
                    stale_timeout_seconds=stale_timeout_seconds,
                    now=now,
                )
                results.append(res)
        return results

    @staticmethod
    def _chunks_to_bm25_dicts(chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Serialize DocumentChunk models into BM25-ready metadata dictionaries."""
        return [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "chunk_text": chunk.chunk_text,
                "page_number": chunk.page_number,
                "page_references": chunk.page_references,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
            }
            for chunk in chunks
        ]

    def _prepare_chunks_and_embeddings(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: str,
    ) -> tuple[List[ExtractedPage], Dict[str, Any], List[DocumentChunk], List[List[float]], List[str]]:
        """Execute extraction, cleaning, chunking, token-limit enforcement, and embeddings."""
        raw_pages, extracted_meta = DocumentExtractor.extract(file_bytes, filename=filename)

        cleaned_pages: List[ExtractedPage] = []
        raw_page_texts = [p.raw_text for p in raw_pages]
        cleaned_page_texts = [TextCleaner.clean(t) for t in raw_page_texts]

        if len(cleaned_page_texts) > 2:
            cleaned_page_texts = TextCleaner.strip_repetitive_headers_footers(cleaned_page_texts)

        warnings: List[str] = []
        for p, cleaned_txt in zip(raw_pages, cleaned_page_texts):
            p.cleaned_text = cleaned_txt
            p.char_count = len(cleaned_txt)
            p.word_count = len(cleaned_txt.split())
            p.has_text = bool(cleaned_txt.strip())

            if not p.has_text:
                warnings.append(f"Page {p.page_number} contains no extractable or cleaned text.")

            cleaned_pages.append(p)

        all_chunks: List[DocumentChunk] = []
        current_chunk_idx = 0
        for page in cleaned_pages:
            if not page.has_text:
                continue

            page_chunks = self.chunker.chunk_page(
                page_text=page.cleaned_text,
                document_id=document_id,
                filename=filename,
                page_number=page.page_number,
                global_start_index=current_chunk_idx,
            )
            all_chunks.extend(page_chunks)
            current_chunk_idx += len(page_chunks)

        if not all_chunks:
            return cleaned_pages, extracted_meta, [], [], warnings

        max_allowed_tokens = self.embedder.get_max_sequence_length() - 4
        all_chunks = self.chunker.enforce_token_limit(
            all_chunks,
            max_tokens=max_allowed_tokens,
            token_overlap=32,
            tokenizer=self.embedder.get_tokenizer(),
        )

        chunk_texts = [c.chunk_text for c in all_chunks]
        logger.info(
            "Generating dense embeddings for %d chunks (dim=%d)...",
            len(chunk_texts),
            self.embedder.get_embedding_dimension(),
        )
        embeddings = self.embedder.embed_batch(chunk_texts)
        return cleaned_pages, extracted_meta, all_chunks, embeddings, warnings

    async def ingest_document_for_user(
        self,
        file_bytes: bytes,
        filename: str,
        owner_id: UUID,
        doc_repo: DocumentRepository,
        storage_repo: SupabaseStorageRepository,
        declared_mime: Optional[str] = None,
    ) -> IngestionResult:
        """Execute full tenant-isolated ingestion:

        AuthenticatedUser -> DocumentRepository (uploading) -> Supabase Storage
        -> Extraction/Chunking/Embedding -> ChromaDB (owner_id) -> DocumentRepository (ready).
        """
        safe_name, ext, canonical_mime = storage_repo.validate_file(
            file_bytes=file_bytes,
            filename=filename,
            declared_mime=declared_mime,
        )
        content_hash = compute_content_hash(file_bytes)

        async with self._get_upload_lock(owner_id, content_hash):
            return await self._ingest_document_for_user_locked(
                file_bytes=file_bytes,
                safe_name=safe_name,
                ext=ext,
                canonical_mime=canonical_mime,
                content_hash=content_hash,
                owner_id=owner_id,
                doc_repo=doc_repo,
                storage_repo=storage_repo,
            )

    async def _ingest_document_for_user_locked(
        self,
        file_bytes: bytes,
        safe_name: str,
        ext: str,
        canonical_mime: str,
        content_hash: str,
        owner_id: UUID,
        doc_repo: DocumentRepository,
        storage_repo: SupabaseStorageRepository,
    ) -> IngestionResult:
        """Execute serialized per-(owner, content_hash) ingestion and stale recovery."""
        # Per-owner duplicate / in-flight check
        existing: Optional[DocumentRecord] = await doc_repo.get_document_by_hash(
            content_hash=content_hash,
            owner_id=owner_id,
        )
        if existing is not None:
            if existing.processing_status in STALE_ELIGIBLE_STATUSES and self.is_stale_record(existing):
                recovery = await self.recover_stale_document(
                    document_id=existing.id,
                    owner_id=owner_id,
                    doc_repo=doc_repo,
                    storage_repo=storage_repo,
                )
                if recovery.cleanup_errors:
                    raise RuntimeError(
                        f"Stale ingestion cleanup incomplete for '{existing.id}': {','.join(recovery.cleanup_errors)}"
                    )
                existing = await doc_repo.get_document(existing.id, owner_id=owner_id)

        if existing is not None:
            if existing.processing_status == "ready":
                existing_meta = DocumentMetadata(
                    document_id=str(existing.id),
                    filename=existing.filename,
                    file_extension=existing.file_extension,
                    file_size_bytes=existing.file_size_bytes,
                    content_hash=existing.content_hash,
                    total_pages=max(1, existing.total_pages),
                    total_chunks=existing.total_chunks,
                    created_at=existing.created_at,
                )
                return IngestionResult(
                    document_id=str(existing.id),
                    filename=existing.filename,
                    content_hash=existing.content_hash,
                    status="duplicate",
                    total_pages=existing.total_pages,
                    total_chunks=existing.total_chunks,
                    embedding_dimension=self.embedder.get_embedding_dimension(),
                    vector_count=existing.total_chunks,
                    collection_name=self.vector_store.collection_name,
                    is_duplicate=True,
                    metadata=existing_meta,
                    chunks=[],
                    raw_pages=[],
                    warnings=[f"Document with identical SHA-256 hash ({content_hash[:12]}...) is already stored."],
                )

            if existing.processing_status in {"pending", "uploading", "indexing", "deleting"}:
                raise DocumentInProgressError(
                    f"Document operation ('{existing.processing_status}') is already in progress for this file."
                )

            if existing.processing_status == "failed":
                # Idempotently purge failed residue for this owner before retrying cleanly
                await asyncio.to_thread(self.vector_store.delete_document, str(existing.id), owner_id)
                if self.tenant_bm25_manager is not None:
                    self.tenant_bm25_manager.remove_document(owner_id, str(existing.id))
                try:
                    await storage_repo.delete_document(owner_id, existing.storage_path)
                except Exception as cleanup_exc:
                    logger.warning("Cleanup of failed Storage object notice: %s", cleanup_exc)
                await doc_repo.delete_document(existing.id, owner_id)

        doc_uuid = uuid4()
        storage_path = storage_repo.build_storage_path(owner_id, doc_uuid, safe_name)

        # Stage 1: Create metadata row in 'uploading' state
        created_row = await doc_repo.create_document(
            owner_id=owner_id,
            filename=safe_name,
            file_extension=ext,
            mime_type=canonical_mime,
            content_hash=content_hash,
            file_size_bytes=len(file_bytes),
            storage_path=storage_path,
            document_id=doc_uuid,
            processing_status="uploading",
            embedding_model=self.embedder.model_name,
        )

        # Stage 2: Upload original file bytes to private Supabase Storage
        storage_uploaded = False
        try:
            await storage_repo.upload_document(
                owner_id=owner_id,
                document_id=doc_uuid,
                filename=safe_name,
                file_bytes=file_bytes,
                declared_mime=canonical_mime,
                upsert=False,
            )
            storage_uploaded = True
        except Exception as storage_exc:
            logger.error("Private Storage upload failed for doc '%s' (owner=%s): %s", doc_uuid, owner_id, type(storage_exc).__name__)
            try:
                await doc_repo.update_document_status(
                    doc_uuid,
                    owner_id,
                    status="failed",
                    error_message="Private file storage upload failed.",
                )
            except Exception as status_exc:
                logger.warning("Failed updating status to 'failed' after storage error: %s", status_exc)
            raise

        # Stage 3: Transition to 'indexing', extract, chunk, embed, and insert into ChromaDB
        vectors_stored = False
        try:
            await doc_repo.update_document_status(doc_uuid, owner_id, status="indexing")

            cleaned_pages, extracted_meta, all_chunks, embeddings, warnings = await asyncio.to_thread(
                self._prepare_chunks_and_embeddings,
                file_bytes,
                safe_name,
                str(doc_uuid),
            )

            stored_count = 0
            if all_chunks:
                stored_count = await asyncio.to_thread(
                    self.vector_store.add_chunks,
                    all_chunks,
                    embeddings,
                    owner_id,
                    content_hash,
                    self.embedder.model_name,
                    True,
                )
                vectors_stored = True

            ready_row = await doc_repo.update_document_status(
                doc_uuid,
                owner_id,
                status="ready",
                total_pages=len(cleaned_pages),
                total_chunks=len(all_chunks),
                embedding_model=self.embedder.model_name,
            )
            if ready_row is None:
                raise RuntimeError(
                    f"Failed to transition document '{doc_uuid}' to 'ready' status: metadata record not found."
                )

            if self.tenant_bm25_manager is not None and all_chunks:
                self.tenant_bm25_manager.add_chunks(owner_id, self._chunks_to_bm25_dicts(all_chunks))

            metadata = DocumentMetadata(
                document_id=str(doc_uuid),
                filename=safe_name,
                file_extension=ext,
                file_size_bytes=len(file_bytes),
                content_hash=content_hash,
                total_pages=max(1, len(cleaned_pages)),
                total_chunks=len(all_chunks),
                created_at=ready_row.created_at if ready_row else created_row.created_at,
                extracted_metadata=extracted_meta,
            )
            return IngestionResult(
                document_id=str(doc_uuid),
                filename=safe_name,
                content_hash=content_hash,
                status="success",
                total_pages=len(cleaned_pages),
                total_chunks=len(all_chunks),
                embedding_dimension=self.embedder.get_embedding_dimension(),
                vector_count=stored_count,
                collection_name=self.vector_store.collection_name,
                is_duplicate=False,
                metadata=metadata,
                chunks=all_chunks,
                raw_pages=cleaned_pages,
                warnings=warnings,
            )

        except Exception as exc:
            logger.error("Ingestion failed for doc '%s' (owner=%s): %s", doc_uuid, owner_id, type(exc).__name__)
            cleanup_errors: List[str] = []
            if self.tenant_bm25_manager is not None:
                try:
                    self.tenant_bm25_manager.remove_document(owner_id, str(doc_uuid))
                except Exception as bm25_cleanup_exc:
                    logger.warning("BM25 cleanup failed for '%s': %s", doc_uuid, bm25_cleanup_exc)
                    cleanup_errors.append(f"bm25:{type(bm25_cleanup_exc).__name__}")
            if vectors_stored:
                try:
                    await asyncio.to_thread(self.vector_store.delete_document, str(doc_uuid), owner_id)
                except Exception as vec_cleanup_exc:
                    logger.critical("Vector cleanup failed for '%s': %s", doc_uuid, vec_cleanup_exc)
                    cleanup_errors.append(f"chromadb:{type(vec_cleanup_exc).__name__}")
            if storage_uploaded:
                try:
                    await storage_repo.delete_document(owner_id, storage_path)
                except Exception as storage_cleanup_exc:
                    logger.warning(
                        "Storage cleanup failed for '%s' (owner=%s): %s",
                        storage_path,
                        owner_id,
                        storage_cleanup_exc,
                    )
                    cleanup_errors.append(f"storage:{type(storage_cleanup_exc).__name__}")

            err_msg = str(exc)[:400]
            if cleanup_errors:
                err_msg = f"{err_msg} [cleanup_errors={','.join(cleanup_errors)}]"[:500]
                logger.error(
                    "Ingestion rollback for doc '%s' (owner=%s) was INCOMPLETE: %s",
                    doc_uuid,
                    owner_id,
                    cleanup_errors,
                )
            try:
                await doc_repo.update_document_status(
                    doc_uuid,
                    owner_id,
                    status="failed",
                    error_message=err_msg,
                )
            except Exception as status_exc:
                cleanup_errors.append(f"metadata_status:{type(status_exc).__name__}")
                logger.warning(
                    "Failed marking document '%s' as failed (cleanup_errors=%s): %s",
                    doc_uuid,
                    cleanup_errors,
                    status_exc,
                )
            raise

    async def delete_document_for_user(
        self,
        document_id: UUID,
        owner_id: UUID,
        doc_repo: DocumentRepository,
        storage_repo: SupabaseStorageRepository,
    ) -> Optional[int]:
        """Verify ownership, mark deleting, remove ChromaDB vectors and Storage object, then delete row."""
        doc = await doc_repo.get_document(document_id, owner_id=owner_id)
        if doc is None:
            return None

        await doc_repo.update_document_status(document_id, owner_id=owner_id, status="deleting")

        deleted_chunks = await asyncio.to_thread(
            self.vector_store.delete_document,
            str(document_id),
            owner_id,
        )
        if self.tenant_bm25_manager is not None:
            self.tenant_bm25_manager.remove_document(owner_id, str(document_id))

        await storage_repo.delete_document(owner_id=owner_id, storage_path=doc.storage_path)

        deleted_row = await doc_repo.delete_document(document_id, owner_id=owner_id)
        if not deleted_row:
            raise RuntimeError("Failed to delete document metadata record after purging vectors and storage.")

        return deleted_chunks

    def ingest_document(
        self,
        file_input: Union[bytes, str, Path],
        owner_id: Union[UUID, str],
        filename: str = "",
        prevent_duplicates: bool = True,
        document_id: Optional[str] = None,
    ) -> IngestionResult:
        """Synchronous owner-scoped ingestion helper for local/unit pipeline testing."""
        owner_str = normalize_owner_id(owner_id)

        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not filename:
                filename = path.name
            file_bytes = path.read_bytes()
        else:
            file_bytes = file_input

        if not filename:
            filename = "document.pdf"

        content_hash = compute_content_hash(file_bytes)
        resolved_doc_id = document_id or str(uuid4())
        ext = Path(filename).suffix.lower()

        if prevent_duplicates and self.vector_store.has_content_hash(content_hash, owner_id=owner_str):
            return IngestionResult(
                document_id=resolved_doc_id,
                filename=filename,
                content_hash=content_hash,
                status="duplicate",
                total_pages=0,
                total_chunks=0,
                embedding_dimension=self.embedder.get_embedding_dimension(),
                vector_count=0,
                collection_name=self.vector_store.collection_name,
                is_duplicate=True,
                chunks=[],
                raw_pages=[],
                warnings=[f"Document with identical SHA-256 hash ({content_hash[:12]}...) is already stored."],
            )

        cleaned_pages, extracted_meta, all_chunks, embeddings, warnings = self._prepare_chunks_and_embeddings(
            file_bytes=file_bytes,
            filename=filename,
            document_id=resolved_doc_id,
        )

        if not all_chunks:
            metadata = DocumentMetadata(
                document_id=resolved_doc_id,
                filename=filename,
                file_extension=ext,
                file_size_bytes=len(file_bytes),
                content_hash=content_hash,
                total_pages=len(cleaned_pages),
                extracted_metadata=extracted_meta,
            )
            return IngestionResult(
                document_id=resolved_doc_id,
                filename=filename,
                content_hash=content_hash,
                status="success",
                total_pages=len(cleaned_pages),
                total_chunks=0,
                embedding_dimension=self.embedder.get_embedding_dimension(),
                vector_count=0,
                collection_name=self.vector_store.collection_name,
                is_duplicate=False,
                metadata=metadata,
                chunks=[],
                raw_pages=cleaned_pages,
                warnings=warnings,
            )

        stored_count = self.vector_store.add_chunks(
            chunks=all_chunks,
            embeddings=embeddings,
            owner_id=owner_str,
            content_hash=content_hash,
            embedding_model=self.embedder.model_name,
            prevent_duplicates=prevent_duplicates,
        )
        if self.tenant_bm25_manager is not None and all_chunks:
            self.tenant_bm25_manager.add_chunks(owner_str, self._chunks_to_bm25_dicts(all_chunks))

        metadata = DocumentMetadata(
            document_id=resolved_doc_id,
            filename=filename,
            file_extension=ext,
            file_size_bytes=len(file_bytes),
            content_hash=content_hash,
            total_pages=len(cleaned_pages),
            total_chunks=len(all_chunks),
            extracted_metadata=extracted_meta,
        )
        if self.registry is not None:
            self.registry.register(metadata)

        return IngestionResult(
            document_id=resolved_doc_id,
            filename=filename,
            content_hash=content_hash,
            status="success",
            total_pages=len(cleaned_pages),
            total_chunks=len(all_chunks),
            embedding_dimension=self.embedder.get_embedding_dimension(),
            vector_count=stored_count,
            collection_name=self.vector_store.collection_name,
            is_duplicate=False,
            metadata=metadata,
            chunks=all_chunks,
            raw_pages=cleaned_pages,
            warnings=warnings,
        )

    def delete_document(self, document_id: str, owner_id: Union[UUID, str]) -> int:
        """Remove a document for owner_id from vector store (and optional registry)."""
        owner_str = normalize_owner_id(owner_id)
        deleted_count = self.vector_store.delete_document(document_id, owner_id=owner_str)
        if self.tenant_bm25_manager is not None:
            self.tenant_bm25_manager.remove_document(owner_str, document_id)
        if self.registry is not None:
            self.registry.remove(document_id)
        return deleted_count

    def get_document_chunks(self, document_id: str, owner_id: Union[UUID, str]) -> List[Dict[str, Any]]:
        """Retrieve stored chunks and deserialized metadata for (owner_id, document_id)."""
        return self.vector_store.get_document_chunks(document_id, owner_id=owner_id)

    def get_statistics(self, owner_id: Optional[Union[UUID, str]] = None) -> Dict[str, Any]:
        """Return operational statistics for ingestion and vector storage."""
        stats = self.vector_store.get_statistics(owner_id=owner_id)
        stats["embedding_model"] = self.embedder.model_name
        stats["max_sequence_length"] = self.embedder.get_max_sequence_length()
        return stats


_ingestion_service_instance: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    """Get or create the singleton IngestionService instance."""
    global _ingestion_service_instance
    if _ingestion_service_instance is None:
        from backend.app.retrieval.retriever import get_retriever

        retriever = get_retriever()
        _ingestion_service_instance = IngestionService(
            vector_store=retriever.vector_store,
            tenant_bm25_manager=retriever.tenant_bm25_manager,
        )
    return _ingestion_service_instance
