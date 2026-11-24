"""Domain exceptions for document processing and ingestion."""


class DocumentProcessingError(Exception):
    """Base exception for all document processing errors."""

    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when an uploaded document has an unsupported file extension or MIME type."""
    pass


class CorruptedPDFError(DocumentProcessingError):
    """Raised when a PDF file is corrupted, malformed, or cannot be parsed."""
    pass


class EncryptedPDFError(DocumentProcessingError):
    """Raised when a PDF file is password-protected or encrypted."""
    pass


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a document contains 0 bytes or has no textual content."""
    pass


class NoExtractableTextError(DocumentProcessingError):
    """Raised when a PDF contains valid pages but zero extractable text (e.g. scanned or image-only)."""
    pass


class EncodingError(DocumentProcessingError):
    """Raised when a text document cannot be decoded with supported encodings."""
    pass


class ChunkingError(DocumentProcessingError):
    """Raised when chunking parameters or execution violates invariants."""
    pass


class RegistryPersistenceError(DocumentProcessingError):
    """Raised when atomic metadata registry file serialization fails."""
    pass
