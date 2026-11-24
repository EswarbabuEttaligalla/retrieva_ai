"""Pydantic schemas and data contracts for document ingestion and chunking."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    """Metadata describing a processed document."""

    document_id: str = Field(..., description="Unique deterministic identifier for the document")
    filename: str = Field(..., description="Original filename with extension")
    file_extension: str = Field(..., description="Normalized lower-case extension (e.g. .pdf, .txt)")
    file_size_bytes: int = Field(..., ge=0, description="Size of the raw document in bytes")
    content_hash: str = Field(..., description="SHA-256 cryptographic digest of document bytes")
    total_pages: int = Field(..., ge=1, description="Total number of extracted pages")
    total_chunks: int = Field(
        default=0,
        ge=0,
        description="Total number of chunks stored for the document",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of ingestion",
    )
    extracted_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Format-specific metadata extracted from file header (author, title, etc.)",
    )


class ExtractedPage(BaseModel):
    """Data representation of a single extracted document page."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    raw_text: str = Field(..., description="Verbatim raw text extracted from the document")
    cleaned_text: str = Field(default="", description="Normalized, cleaned text ready for chunking")
    char_count: int = Field(default=0, ge=0, description="Character count of cleaned text")
    word_count: int = Field(default=0, ge=0, description="Word count of cleaned text")
    has_text: bool = Field(default=False, description="Flag indicating whether text was found")

    def model_post_init(self, __context: Any) -> None:
        """Derive counts and has_text if cleaned_text is populated."""
        if self.cleaned_text:
            self.char_count = len(self.cleaned_text)
            self.word_count = len(self.cleaned_text.split())
            self.has_text = bool(self.cleaned_text.strip())
        elif self.raw_text:
            self.has_text = bool(self.raw_text.strip())


class DocumentChunk(BaseModel):
    """An individual text chunk with strict offset tracking and page provenance."""

    chunk_id: str = Field(..., description="Globally unique chunk identifier (<doc_id>_p<page>_c<idx>)")
    document_id: str = Field(..., description="ID of the parent document")
    filename: str = Field(..., description="Filename of the source document")
    page_number: int = Field(..., ge=1, description="Primary 1-indexed page number")
    page_references: List[int] = Field(
        default_factory=list,
        description="All page numbers referenced by this chunk",
    )
    chunk_index: int = Field(..., ge=0, description="0-indexed position of chunk in the document")
    chunk_text: str = Field(..., min_length=1, description="The textual content of the chunk")
    start_offset: int = Field(..., ge=0, description="Start character index in the normalized page text")
    end_offset: int = Field(..., ge=1, description="End character index in the normalized page text")
    char_count: int = Field(..., ge=1, description="Number of characters in the chunk text")
    word_count: int = Field(..., ge=1, description="Number of words in the chunk text")

    @property
    def text(self) -> str:
        """Alias for chunk_text for convenience and vector store compatibility."""
        return self.chunk_text

    @field_validator("end_offset")
    @classmethod
    def validate_offsets(cls, end_offset: int, info) -> int:
        """Ensure end_offset is strictly greater than start_offset."""
        start_offset = info.data.get("start_offset")
        if start_offset is not None and end_offset <= start_offset:
            raise ValueError(f"end_offset ({end_offset}) must be strictly greater than start_offset ({start_offset})")
        return end_offset


class IngestionResult(BaseModel):
    """Result summary of the document ingestion and chunking operation."""

    document_id: str = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Original filename")
    content_hash: str = Field(..., description="SHA-256 content hash")
    status: str = Field(default="success", description="Status ('success', 'duplicate', 'failed')")
    total_pages: int = Field(default=0, ge=0, description="Number of pages ingested")
    total_chunks: int = Field(default=0, ge=0, description="Number of chunks generated")
    embedding_dimension: Optional[int] = Field(default=None, description="Dimension of generated vector embeddings")
    vector_count: Optional[int] = Field(default=0, ge=0, description="Number of vectors stored in vector database")
    collection_name: Optional[str] = Field(default=None, description="Name of vector collection where vectors are stored")
    is_duplicate: bool = Field(default=False, description="Whether document was already ingested")
    metadata: Optional[DocumentMetadata] = Field(default=None, description="Detailed document metadata")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="List of generated chunks")
    raw_pages: Optional[List[ExtractedPage]] = Field(
        default=None,
        description="Preserved raw and cleaned pages for debugging and provenance verification",
    )
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings encountered")


class DocumentUploadResponse(BaseModel):
    """Response returned upon completing document upload and indexing."""

    document_id: str = Field(..., description="Unique deterministic identifier for the document")
    filename: str = Field(..., description="Original filename with extension")
    content_hash: str = Field(..., description="SHA-256 cryptographic digest")
    file_size_bytes: int = Field(..., ge=0, description="Size of the raw document in bytes")
    total_pages: int = Field(..., ge=0, description="Total number of processed pages")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks created and stored")
    status: str = Field(..., description="Status ('success', 'duplicate', 'failed')")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of upload",
    )
    is_duplicate: bool = Field(default=False, description="Whether document was already ingested")
    message: str = Field(..., description="Human-readable outcome summary")


class DocumentListItem(BaseModel):
    """Individual document entry in the document catalog list."""

    document_id: str = Field(..., description="Unique deterministic identifier for the document")
    filename: str = Field(..., description="Original filename with extension")
    file_extension: str = Field(..., description="Normalized lower-case extension (e.g. .pdf, .txt)")
    file_size_bytes: int = Field(..., ge=0, description="Size of the raw document in bytes")
    total_pages: int = Field(..., ge=0, description="Total number of processed pages")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks stored in vector database")
    created_at: datetime = Field(..., description="UTC timestamp of upload")


class DocumentListResponse(BaseModel):
    """Response returned for listing all ingested documents."""

    documents: List[DocumentListItem] = Field(
        default_factory=list,
        description="List of all stored documents in the system",
    )
    total: int = Field(..., ge=0, description="Total number of registered documents")


class DocumentDeleteResponse(BaseModel):
    """Response returned upon deleting a document."""

    document_id: str = Field(..., description="ID of the deleted document")
    deleted_chunks: int = Field(..., ge=0, description="Number of vector chunks removed from store")
    status: str = Field(default="deleted", description="Operation status ('deleted')")
    message: str = Field(..., description="Human-readable outcome summary")
