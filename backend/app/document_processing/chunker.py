"""Custom text chunking engine with semantic boundary preservation and provenance tracking."""

import re
from typing import Any, List, Optional, Tuple
from backend.app.document_processing.exceptions import ChunkingError
from backend.app.schemas.document import DocumentChunk


def count_tokens_text(text: str, tokenizer: Optional[Any] = None) -> int:
    """Accurately count subword tokens using tokenizer if available, or regex word/punct estimation.

    Args:
        text: Input string.
        tokenizer: Optional HuggingFace tokenizer instance.

    Returns:
        Token count as an integer.
    """
    if not text or not text.strip():
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Regex fallback: matches words and standalone punctuation marks
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return max(1, len(tokens))


class TextChunker:
    """Manual text chunking engine designed for Retrieval-Augmented Generation.
    
    Unit of Measurement:
        Characters (with semantic boundary awareness).
        
    Justification:
        1. Character offsets provide deterministic, exact substring tracking:
           `page_text[start_offset:end_offset] == chunk_text` is verifiable to the exact byte.
        2. Zero dependency on third-party tokenizer models, guaranteeing standalone reliability.
        3. 500 characters roughly translates to 75-100 English words, well within the 256-token
           sweet spot of Sentence-Transformers (e.g. all-MiniLM-L6-v2) for optimal dense vector fidelity.
    """

    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 100

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ChunkingError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(
        self,
        page_text: str,
        document_id: str,
        filename: str,
        page_number: int,
        global_start_index: int = 0,
    ) -> List[DocumentChunk]:
        """Divide a single normalized page of text into overlapping, boundary-aware chunks.
        
        Args:
            page_text: The normalized cleaned text of the page.
            document_id: Unique identifier of the source document.
            filename: Original document filename.
            page_number: 1-indexed page number.
            global_start_index: Offset for chunk indexing across multi-page documents.
            
        Returns:
            A list of DocumentChunk objects with verified offsets and provenance.
        """
        if not page_text or not page_text.strip():
            return []

        text_len = len(page_text)

        # Base case: text fits entirely within one chunk
        if text_len <= self.chunk_size:
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_p{page_number}_c{global_start_index}",
                document_id=document_id,
                filename=filename,
                page_number=page_number,
                page_references=[page_number],
                chunk_index=global_start_index,
                chunk_text=page_text,
                start_offset=0,
                end_offset=text_len,
                char_count=text_len,
                word_count=len(page_text.split()),
            )
            return [chunk]

        chunks: List[DocumentChunk] = []
        start = 0
        chunk_idx = global_start_index

        while start < text_len:
            # Determine end position for the candidate window
            if start + self.chunk_size >= text_len:
                end = text_len
            else:
                end = self._find_boundary(page_text, start, start + self.chunk_size)

            chunk_content = page_text[start:end]
            
            # Create the chunk
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_p{page_number}_c{chunk_idx}",
                document_id=document_id,
                filename=filename,
                page_number=page_number,
                page_references=[page_number],
                chunk_index=chunk_idx,
                chunk_text=chunk_content,
                start_offset=start,
                end_offset=end,
                char_count=len(chunk_content),
                word_count=len(chunk_content.split()),
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end >= text_len:
                break

            # Calculate next starting position with overlap
            next_start = end - self.chunk_overlap

            # Ensure strict forward progress
            if next_start <= start:
                next_start = start + 1

            # Avoid starting the next chunk mid-table-row or mid-word
            if next_start < end:
                line_start = page_text.rfind("\n", start, next_start) + 1
                if page_text[line_start:].lstrip().startswith("|"):
                    next_nl = page_text.find("\n", next_start, end)
                    if line_start > start:
                        next_start = line_start
                    elif next_nl != -1 and next_nl + 1 < end:
                        next_start = next_nl + 1
                elif not page_text[next_start].isspace() and next_start > 0 and not page_text[next_start - 1].isspace():
                    next_space = page_text.find(" ", next_start, end)
                    if next_space != -1 and next_space + 1 < end:
                        next_start = next_space + 1

            start = next_start

        return chunks

    def _find_boundary(self, text: str, start: int, max_end: int) -> int:
        """Find the most natural linguistic boundary near max_end.
        
        Priority:
        1. Paragraph boundary: \n\n
        2. Sentence boundaries: '. ', '! ', '? ', '.\n', or newline '\n'
        3. Word boundary: ' '
        4. Fallback: hard cut at max_end
        """
        # Search zone: look within the last (chunk_overlap * 2) characters or at least 40% of chunk_size
        min_search_start = max(start + 1, max_end - max(self.chunk_overlap * 2, int(self.chunk_size * 0.4)))
        search_slice = text[min_search_start:max_end]

        # 1. Paragraph boundary (\n\n)
        p_idx = search_slice.rfind("\n\n")
        if p_idx != -1:
            return min_search_start + p_idx + 2

        # 2. Sentence boundary (. , ! , ? , .\n)
        for punct in [". ", "! ", "? ", ".\n"]:
            s_idx = search_slice.rfind(punct)
            if s_idx != -1:
                return min_search_start + s_idx + len(punct)

        # Single newline (\n)
        nl_idx = search_slice.rfind("\n")
        if nl_idx != -1:
            return min_search_start + nl_idx + 1

        # 3. Word boundary (space)
        w_idx = search_slice.rfind(" ")
        if w_idx != -1:
            return min_search_start + w_idx + 1

        # 4. Fallback: hard cut at max_end if no natural boundary exists
        return max_end

    @staticmethod
    def reconstruct_page_text(chunks: List[DocumentChunk]) -> str:
        """Reconstruct the original normalized page text from a sequence of overlapping chunks.
        
        This verifies that no text was lost between chunks during the chunking process.
        
        Algorithm:
            Starts with Chunk 0. For each subsequent chunk, appends only the non-overlapping
            suffix from `prev_chunk.end_offset` to `curr_chunk.end_offset`.
        """
        if not chunks:
            return ""

        reconstructed_parts: List[str] = [chunks[0].chunk_text]
        prev_end = chunks[0].end_offset

        for chunk in chunks[1:]:
            if chunk.end_offset <= prev_end:
                # Completely covered by previous chunk
                continue
            # Extract only the newly uncovered text slice
            overlap_len = max(0, prev_end - chunk.start_offset)
            new_text_slice = chunk.chunk_text[overlap_len:]
            reconstructed_parts.append(new_text_slice)
            prev_end = chunk.end_offset

        return "".join(reconstructed_parts)

    @classmethod
    def split_oversized_chunk(
        cls,
        chunk: DocumentChunk,
        max_tokens: int = 250,
        token_overlap: int = 32,
        tokenizer: Optional[Any] = None,
    ) -> List[DocumentChunk]:
        """Split a chunk exceeding max_tokens into boundary-aware subchunks.

        Preserves:
        - document_id, filename, page_number, page_references.
        - Exact character offsets relative to original page text:
          `page_text[subchunk.start_offset : subchunk.end_offset] == subchunk.chunk_text`
        - Sequential subchunk numbering: f"{chunk.chunk_id}_s{idx}".
        - Strict forward progress and overlap continuity.

        Args:
            chunk: The DocumentChunk to check and potentially split.
            max_tokens: Maximum allowed tokens per subchunk (default: 250).
            token_overlap: Overlap in tokens between consecutive subchunks (default: 32).
            tokenizer: Optional tokenizer instance.

        Returns:
            List of DocumentChunk instances, each with <= max_tokens.
        """
        chunk_text = chunk.chunk_text
        total_tokens = count_tokens_text(chunk_text, tokenizer=tokenizer)

        if total_tokens <= max_tokens:
            return [chunk]

        # Partition text into natural linguistic spans: sentences or paragraphs
        spans: List[Tuple[int, int]] = []
        sentence_delims = list(re.finditer(r"(\n\n+|\.\s+|\!\s+|\?\s+|\n)", chunk_text))
        prev_pos = 0
        for match in sentence_delims:
            end_pos = match.end()
            if end_pos > prev_pos:
                spans.append((prev_pos, end_pos))
                prev_pos = end_pos
        if prev_pos < len(chunk_text):
            spans.append((prev_pos, len(chunk_text)))

        # Break any individual span that exceeds max_tokens down into words
        refined_spans: List[Tuple[int, int]] = []
        for s_start, s_end in spans:
            span_text = chunk_text[s_start:s_end]
            if count_tokens_text(span_text, tokenizer=tokenizer) > max_tokens:
                word_matches = list(re.finditer(r"\S+\s*", span_text))
                if word_matches:
                    for wm in word_matches:
                        refined_spans.append((s_start + wm.start(), s_start + wm.end()))
                else:
                    refined_spans.append((s_start, s_end))
            else:
                refined_spans.append((s_start, s_end))

        subchunks: List[DocumentChunk] = []
        current_span_idx = 0
        sub_idx = 0
        num_spans = len(refined_spans)

        while current_span_idx < num_spans:
            start_char = refined_spans[current_span_idx][0]
            end_char = refined_spans[current_span_idx][1]
            last_valid_span_idx = current_span_idx

            for next_idx in range(current_span_idx + 1, num_spans):
                candidate_end = refined_spans[next_idx][1]
                candidate_text = chunk_text[start_char:candidate_end]
                if count_tokens_text(candidate_text, tokenizer=tokenizer) <= max_tokens:
                    end_char = candidate_end
                    last_valid_span_idx = next_idx
                else:
                    break

            sub_text = chunk_text[start_char:end_char]
            sub_start_offset = chunk.start_offset + start_char
            sub_end_offset = chunk.start_offset + end_char

            sub_chunk = DocumentChunk(
                chunk_id=f"{chunk.chunk_id}_s{sub_idx}",
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                page_references=list(chunk.page_references),
                chunk_index=chunk.chunk_index + sub_idx,
                chunk_text=sub_text,
                start_offset=sub_start_offset,
                end_offset=sub_end_offset,
                char_count=len(sub_text),
                word_count=len(sub_text.split()),
            )
            subchunks.append(sub_chunk)
            sub_idx += 1

            if last_valid_span_idx >= num_spans - 1:
                break

            # Calculate overlap for next subchunk:
            # Step back from last_valid_span_idx such that the overlap in tokens is roughly token_overlap
            next_span_idx = last_valid_span_idx + 1
            overlap_tokens = 0
            back_idx = last_valid_span_idx
            while back_idx > current_span_idx and overlap_tokens < token_overlap:
                span_t = chunk_text[refined_spans[back_idx][0] : refined_spans[back_idx][1]]
                overlap_tokens += count_tokens_text(span_t, tokenizer=tokenizer)
                next_span_idx = back_idx
                back_idx -= 1

            if next_span_idx <= current_span_idx:
                next_span_idx = current_span_idx + 1

            current_span_idx = next_span_idx

        return subchunks

    @classmethod
    def enforce_token_limit(
        cls,
        chunks: List[DocumentChunk],
        max_tokens: int = 250,
        token_overlap: int = 32,
        tokenizer: Optional[Any] = None,
    ) -> List[DocumentChunk]:
        """Validate and split any chunk in a list that exceeds max_tokens.

        Re-indexes the entire resulting chunk list so chunk_index values are strictly
        sequential (0, 1, 2, ...).

        Args:
            chunks: Initial list of DocumentChunk instances.
            max_tokens: Maximum allowed tokens per chunk.
            token_overlap: Overlap in tokens for split chunks.
            tokenizer: Optional tokenizer instance.

        Returns:
            List of token-safe DocumentChunk instances with strictly sequential indices.
        """
        output_chunks: List[DocumentChunk] = []

        for chunk in chunks:
            if count_tokens_text(chunk.chunk_text, tokenizer=tokenizer) <= max_tokens:
                output_chunks.append(chunk)
            else:
                split_pieces = cls.split_oversized_chunk(
                    chunk,
                    max_tokens=max_tokens,
                    token_overlap=token_overlap,
                    tokenizer=tokenizer,
                )
                output_chunks.extend(split_pieces)

        # Re-index to ensure chunk_index is strictly continuous 0..N-1
        for idx, c in enumerate(output_chunks):
            c.chunk_index = idx

        return output_chunks


# Backward-compatible alias
DocumentChunker = TextChunker
