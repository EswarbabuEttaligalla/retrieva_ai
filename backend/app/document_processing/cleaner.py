"""Text cleaning and normalization pipeline for Retrieva AI."""

import re
import unicodedata
from typing import List


class TextCleaner:
    """Normalizes and cleans raw extracted text while preserving semantic structure."""

    # Control characters to strip (excluding tab \x09 and newline \x0a)
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    # Line-end hyphenation pattern: word followed by hyphen and newline followed by word
    _HYPHENATION_RE = re.compile(r"(\b[a-zA-Z]{2,})-\n([a-zA-Z]{2,}\b)")

    # Multiple horizontal whitespaces (spaces/tabs) on a single line
    _HORIZONTAL_WS_RE = re.compile(r"[^\S\n\r]+")

    # Three or more consecutive newlines collapsed to paragraph breaks (\n\n)
    _EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")

    @classmethod
    def clean(cls, raw_text: str) -> str:
        """Apply comprehensive text normalization to a string of text.
        
        Steps:
        1. Unicode normalization (NFKC) for consistent representation of ligatures & accents.
        2. Replace carriage returns (\r\n and \r) with standard \n.
        3. Remove non-printable control characters.
        4. Reconstruct words broken across line-ending hyphens.
        5. Normalize horizontal whitespace (tabs and multiple spaces) within lines.
        6. Collapse excessive blank lines to preserve clean paragraph boundaries (\n\n).
        7. Strip leading and trailing whitespace from the overall page text.
        """
        if not raw_text:
            return ""

        # Step 1: Unicode Normalization (NFKC converts ligatures like 'fi' -> 'fi', '½' -> '1/2', etc.)
        text = unicodedata.normalize("NFKC", raw_text)

        # Step 2: Standardize line endings to Unix style \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Step 3: Remove control characters that disrupt tokenization or storage
        text = cls._CONTROL_CHAR_RE.sub("", text)

        # Step 4: Repair words split by line-wrap hyphens (e.g. "concur-\nrent" -> "concurrent")
        text = cls._HYPHENATION_RE.sub(r"\1\2", text)

        # Step 5: Normalize horizontal whitespace line by line
        lines: List[str] = []
        for line in text.split("\n"):
            # Collapse multiple spaces and tabs into a single space, then strip edges
            cleaned_line = cls._HORIZONTAL_WS_RE.sub(" ", line).strip()
            lines.append(cleaned_line)

        # Recombine lines
        text = "\n".join(lines)

        # Step 6: Collapse excessive vertical spacing (keep at most 2 newlines for paragraph breaks)
        text = cls._EXCESSIVE_NEWLINES_RE.sub("\n\n", text)

        # Step 7: Final trim
        return text.strip()

    @classmethod
    def strip_repetitive_headers_footers(cls, page_texts: List[str]) -> List[str]:
        """Conservatively detect and remove exact running headers/footers repeated across multi-page documents.
        
        Args:
            page_texts: List of cleaned page texts in sequential page order.
            
        Returns:
            List of page texts with detected repeating header/footer lines removed.
        """
        if len(page_texts) < 3:
            # Need at least 3 pages to confidently identify running headers/footers
            return page_texts

        # Extract candidates: first line (header) and last line (footer) from each non-empty page
        first_lines = []
        last_lines = []
        valid_indices = []

        for i, pt in enumerate(page_texts):
            lines = [line.strip() for line in pt.split("\n") if line.strip()]
            if lines:
                first_lines.append(lines[0])
                last_lines.append(lines[-1])
                valid_indices.append(i)

        if len(valid_indices) < 3:
            return page_texts

        # Determine if the same header appears in > 70% of valid pages
        header_to_remove = None
        for candidate in set(first_lines):
            # Only consider short candidate lines (< 80 chars, e.g. "Retrieva AI Manual - Confidential")
            if len(candidate) <= 80 and first_lines.count(candidate) / len(valid_indices) >= 0.70:
                header_to_remove = candidate
                break

        # Determine if the same footer appears in > 70% of valid pages
        footer_to_remove = None
        for candidate in set(last_lines):
            if len(candidate) <= 80 and last_lines.count(candidate) / len(valid_indices) >= 0.70:
                footer_to_remove = candidate
                break

        if not header_to_remove and not footer_to_remove:
            return page_texts

        cleaned_pages = []
        for pt in page_texts:
            lines = pt.split("\n")
            if header_to_remove and lines and lines[0].strip() == header_to_remove:
                lines = lines[1:]
            if footer_to_remove and lines and lines[-1].strip() == footer_to_remove:
                lines = lines[:-1]
            cleaned_pages.append("\n".join(lines).strip())

        return cleaned_pages
