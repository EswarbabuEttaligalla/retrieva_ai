"""Document ingestion and processing pipeline orchestrator."""

import os
from pathlib import Path
from typing import List, Optional, Union

from backend.app.document_processing.cleaner import TextCleaner
from backend.app.document_processing.chunker import TextChunker
from backend.app.document_processing.extractor import DocumentExtractor
from backend.app.document_processing.identifier import (
    DocumentRegistry,
    compute_content_hash,
    generate_document_id,
    global_document_registry,
)
from backend.app.schemas.document import (
    DocumentChunk,
    DocumentMetadata,
    ExtractedPage,
    IngestionResult,
)


class DocumentIngestionPipeline:
    """Orchestrates end-to-end document extraction, cleaning, chunking, and identification."""

    def __init__(
        self,
        registry: Optional[DocumentRegistry] = None,
        chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = TextChunker.DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.registry = registry or global_document_registry
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process(
        self,
        file_input: Union[bytes, str, Path],
        filename: str = "",
    ) -> IngestionResult:
        """Execute the complete ingestion pipeline for a document.
        
        Args:
            file_input: Raw byte buffer or path to file.
            filename: Original filename (inferred if file_input is a path).
            
        Returns:
            An IngestionResult containing metadata, extracted pages, and generated chunks.
        """
        # Resolve bytes and filename
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not filename:
                filename = path.name
            file_bytes = path.read_bytes()
        else:
            file_bytes = file_input

        if not filename:
            filename = "document.pdf"

        # Step 1: Content Hashing & Document ID
        content_hash = compute_content_hash(file_bytes)
        document_id = generate_document_id(content_hash)
        ext = os.path.splitext(filename)[1].lower()

        # Step 2: Check Duplicate Status
        if self.registry.is_duplicate(content_hash):
            existing_meta = self.registry.get_by_hash(content_hash)
            return IngestionResult(
                document_id=document_id,
                filename=filename,
                content_hash=content_hash,
                status="duplicate",
                total_pages=existing_meta.total_pages if existing_meta else 0,
                total_chunks=0,
                is_duplicate=True,
                metadata=existing_meta,
                chunks=[],
                raw_pages=[],
                warnings=[f"Document with identical SHA-256 hash ({content_hash[:12]}...) is already ingested."],
            )

        # Step 3: Extract Raw Text & Metadata
        raw_pages, extracted_meta = DocumentExtractor.extract(file_bytes, filename=filename)

        # Step 4: Clean Text across all pages
        cleaned_pages: List[ExtractedPage] = []
        raw_page_texts = [p.raw_text for p in raw_pages]
        cleaned_page_texts = [TextCleaner.clean(t) for t in raw_page_texts]

        # Apply multi-page header/footer stripping if applicable
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

        # Step 5: Chunking Page-by-Page with Page Provenance
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

        # Step 6: Construct Document Metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_extension=ext,
            file_size_bytes=len(file_bytes),
            content_hash=content_hash,
            total_pages=len(cleaned_pages),
            extracted_metadata=extracted_meta,
        )

        # Step 7: Register in Document Registry
        self.registry.register(metadata)

        return IngestionResult(
            document_id=document_id,
            filename=filename,
            content_hash=content_hash,
            status="success",
            total_pages=len(cleaned_pages),
            total_chunks=len(all_chunks),
            is_duplicate=False,
            metadata=metadata,
            chunks=all_chunks,
            raw_pages=cleaned_pages,
            warnings=warnings,
        )
