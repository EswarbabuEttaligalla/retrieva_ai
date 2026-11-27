"""Document extraction engine supporting PDF and TXT documents using PyMuPDF."""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import fitz  # PyMuPDF

from backend.app.document_processing.exceptions import (
    CorruptedPDFError,
    EmptyDocumentError,
    EncryptedPDFError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
    EncodingError,
)
from backend.app.schemas.document import ExtractedPage


class DocumentExtractor:
    """Extracts raw text and metadata from PDF and TXT files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    @classmethod
    def extract(
        cls,
        file_input: Union[bytes, str, Path],
        filename: str = "",
    ) -> Tuple[List[ExtractedPage], Dict[str, Any]]:
        """Extract pages and metadata from a document provided as bytes or file path.
        
        Args:
            file_input: Raw byte buffer or path to the file.
            filename: Name of the file (required if file_input is bytes).
            
        Returns:
            Tuple of (list of ExtractedPage objects, metadata dictionary).
            
        Raises:
            UnsupportedFileTypeError: If file extension is not supported.
            EmptyDocumentError: If file has 0 bytes or no content.
            CorruptedPDFError: If PDF format is damaged or invalid.
            EncryptedPDFError: If PDF is password protected.
            NoExtractableTextError: If PDF has pages but zero text (e.g. scanned).
            EncodingError: If TXT file cannot be decoded.
        """
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if not filename:
                filename = path.name
            try:
                content = path.read_bytes()
            except Exception as e:
                raise CorruptedPDFError("Failed to read file from disk", str(e))
        else:
            content = file_input

        if not filename:
            raise UnsupportedFileTypeError("Filename must be specified for byte streams")

        ext = os.path.splitext(filename)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file extension '{ext}'. Retrieva AI supports {cls.SUPPORTED_EXTENSIONS}"
            )

        if len(content) == 0:
            raise EmptyDocumentError(f"Document '{filename}' is empty (0 bytes)")

        if ext == ".pdf":
            return cls._extract_pdf(content, filename)
        elif ext == ".txt":
            return cls._extract_txt(content, filename)
        else:
            raise UnsupportedFileTypeError(f"Unsupported file format: {ext}")

    @staticmethod
    def _format_markdown_table(rows: List[List[Any]]) -> str:
        """Format extracted table rows as a clean row-preserving Markdown table."""
        cleaned_rows: List[List[str]] = []
        max_cols = 0
        for row in rows:
            if not row:
                continue
            cells = [" ".join(str(cell or "").split()) for cell in row]
            if any(cells):
                cleaned_rows.append(cells)
                max_cols = max(max_cols, len(cells))

        if not cleaned_rows or max_cols == 0:
            return ""

        padded = [r + [""] * (max_cols - len(r)) for r in cleaned_rows]
        lines = ["| " + " | ".join(padded[0]) + " |"]
        if len(padded) > 1:
            lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
            for data_row in padded[1:]:
                lines.append("| " + " | ".join(data_row) + " |")
        return "\n".join(lines)

    @classmethod
    def _extract_pdf_page_text(cls, page: Any) -> str:
        """Extract PDF page text while preserving table rows and column headers."""
        try:
            table_finder = page.find_tables() if hasattr(page, "find_tables") else None
            tables = list(getattr(table_finder, "tables", None) or [])
        except Exception:
            tables = []

        if not tables:
            return page.get_text("text") or ""

        try:
            blocks = page.get_text("blocks") or []
        except Exception:
            return page.get_text("text") or ""

        table_rects = []
        positioned_elements: List[Tuple[float, float, str]] = []
        for tbl in tables:
            bbox = getattr(tbl, "bbox", None)
            if bbox is None:
                continue
            rect = fitz.Rect(bbox)
            md_table = cls._format_markdown_table(tbl.extract() or [])
            if md_table:
                table_rects.append(rect)
                positioned_elements.append((rect.y0, rect.x0, md_table))

        if not positioned_elements:
            return page.get_text("text") or ""

        for blk in blocks:
            if len(blk) < 7 or blk[6] != 0:
                continue
            blk_rect = fitz.Rect(blk[:4])
            blk_text = (blk[4] or "").strip()
            if not blk_text:
                continue
            blk_area = max(blk_rect.get_area(), 1e-6)
            overlaps_table = any(
                (blk_rect & t_rect).get_area() / blk_area > 0.5
                for t_rect in table_rects
            )
            if not overlaps_table:
                positioned_elements.append((blk_rect.y0, blk_rect.x0, blk_text))

        positioned_elements.sort(key=lambda item: (item[0], item[1]))
        combined = "\n\n".join(item[2] for item in positioned_elements if item[2])
        return combined if combined.strip() else (page.get_text("text") or "")

    @classmethod
    def _extract_pdf(cls, content: bytes, filename: str) -> Tuple[List[ExtractedPage], Dict[str, Any]]:
        """Extract pages and metadata from PDF bytes using PyMuPDF."""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as e:
            raise CorruptedPDFError(f"Failed to parse PDF document '{filename}'", str(e))

        # Check encryption
        if doc.is_encrypted or doc.needs_pass:
            doc.close()
            raise EncryptedPDFError(
                f"PDF document '{filename}' is password-protected or encrypted"
            )

        if doc.page_count == 0:
            doc.close()
            raise EmptyDocumentError(f"PDF document '{filename}' contains 0 pages")

        extracted_metadata: Dict[str, Any] = {
            "format": doc.metadata.get("format", "PDF"),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "page_count": doc.page_count,
        }

        pages: List[ExtractedPage] = []
        total_chars = 0

        for i in range(doc.page_count):
            page = doc[i]
            page_num = i + 1
            # Extract text while preserving PDF table row/column structure
            text = cls._extract_pdf_page_text(page)
            total_chars += len(text.strip())

            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    raw_text=text,
                    cleaned_text="",  # Populated downstream in cleaning step
                    char_count=len(text),
                    word_count=len(text.split()),
                    has_text=bool(text.strip()),
                )
            )

        doc.close()

        # Check if the PDF has any extractable text at all
        if total_chars == 0:
            raise NoExtractableTextError(
                f"PDF document '{filename}' contains {len(pages)} pages but zero extractable text. "
                "This document appears to be scanned or image-only. "
                "Retrieva AI does not claim to perform OCR without an active OCR subsystem."
            )

        return pages, extracted_metadata

    @classmethod
    def _extract_txt(cls, content: bytes, filename: str) -> Tuple[List[ExtractedPage], Dict[str, Any]]:
        """Extract text from TXT bytes enforcing valid UTF-8/UTF-8-BOM encoding."""
        # 1. Binary check: reject null bytes indicating non-text binary streams
        if b"\x00" in content:
            raise EncodingError(
                f"TXT document '{filename}' contains binary null bytes and cannot be parsed as text."
            )

        # 2. Strict UTF-8 / UTF-8-BOM decoding
        # Priority: utf-8, utf-8-sig
        decoded_text = ""
        used_encoding = ""
        for enc in ["utf-8", "utf-8-sig"]:
            try:
                decoded_text = content.decode(enc)
                used_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if not used_encoding:
            raise EncodingError(
                f"TXT document '{filename}' is not valid UTF-8 encoded text."
            )

        if not decoded_text.strip():
            raise EmptyDocumentError(f"TXT document '{filename}' contains only whitespace")

        metadata: Dict[str, Any] = {
            "format": "TXT",
            "encoding": used_encoding,
            "char_count": len(decoded_text),
            "line_count": len(decoded_text.splitlines()),
        }

        page = ExtractedPage(
            page_number=1,
            raw_text=decoded_text,
            cleaned_text="",
            char_count=len(decoded_text),
            word_count=len(decoded_text.split()),
            has_text=True,
        )

        return [page], metadata
