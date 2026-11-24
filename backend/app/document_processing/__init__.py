"""Document processing module: extraction, cleaning, chunking, identification, and pipeline."""

from backend.app.document_processing.chunker import TextChunker
from backend.app.document_processing.cleaner import TextCleaner
from backend.app.document_processing.exceptions import (
    ChunkingError,
    CorruptedPDFError,
    DocumentProcessingError,
    EmptyDocumentError,
    EncryptedPDFError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
    EncodingError,
)
from backend.app.document_processing.extractor import DocumentExtractor
from backend.app.document_processing.identifier import (
    DocumentRegistry,
    compute_content_hash,
    generate_document_id,
    global_document_registry,
)
from backend.app.document_processing.pipeline import DocumentIngestionPipeline

__all__ = [
    "TextChunker",
    "TextCleaner",
    "DocumentExtractor",
    "DocumentRegistry",
    "DocumentIngestionPipeline",
    "compute_content_hash",
    "generate_document_id",
    "global_document_registry",
    "DocumentProcessingError",
    "UnsupportedFileTypeError",
    "CorruptedPDFError",
    "EncryptedPDFError",
    "EmptyDocumentError",
    "NoExtractableTextError",
    "EncodingError",
    "ChunkingError",
]
