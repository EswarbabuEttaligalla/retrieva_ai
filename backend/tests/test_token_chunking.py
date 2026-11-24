"""Unit and integration tests for Token-Aware Chunking and Truncation Prevention.

Verifies:
1. Short chunk below limit remains intact.
2. Chunk at exact limit remains intact.
3. Oversized chunk (> 250 tokens) is safely subchunked.
4. Subchunks do not exceed max_tokens.
5. Exact character offset reconstruction: page_text[c.start_offset:c.end_offset] == c.chunk_text.
6. Provenance (document_id, page_number, page_references) is preserved across subchunks.
7. Sequential re-indexing ensures chunk_index is strictly continuous 0..N-1.
8. EmbeddingTokenLimitError is raised by SentenceTransformerEmbedder when oversized text is passed.
9. auto_truncate=True permits truncated embedding generation without raising.
"""

import pytest

from backend.app.document_processing.chunker import (
    DocumentChunk,
    TextChunker,
    count_tokens_text,
)
from backend.app.embeddings.embedder import (
    EmbeddingTokenLimitError,
    SentenceTransformerEmbedder,
)


@pytest.fixture(scope="module")
def shared_tokenizer():
    embedder = SentenceTransformerEmbedder.get_instance()
    return embedder.get_tokenizer()


class TestTokenAwareChunking:
    """Test suite for token-level validation and subchunking."""

    def test_chunk_below_token_limit(self):
        """Chunk with ~30 tokens should remain untouched."""
        text = "This is a short chunk containing roughly twenty words to ensure it stays well below the limit."
        chunk = DocumentChunk(
            chunk_id="doc1_c0",
            document_id="doc1",
            filename="sample.pdf",
            page_number=1,
            page_references=[1],
            chunk_index=0,
            chunk_text=text,
            start_offset=0,
            end_offset=len(text),
            char_count=len(text),
            word_count=len(text.split()),
        )

        result = TextChunker.enforce_token_limit([chunk], max_tokens=250)
        assert len(result) == 1
        assert result[0].chunk_id == "doc1_c0"
        assert result[0].chunk_text == text
        assert result[0].start_offset == 0
        assert result[0].end_offset == len(text)
        assert result[0].chunk_index == 0

    def test_chunk_at_token_limit(self, shared_tokenizer):
        """Chunk precisely at the token threshold should not be split."""
        words = ["token"] * 240
        text = " ".join(words)
        token_count = count_tokens_text(text, tokenizer=shared_tokenizer)
        assert token_count <= 250

        chunk = DocumentChunk(
            chunk_id="doc1_c0",
            document_id="doc1",
            filename="sample.pdf",
            page_number=1,
            page_references=[1],
            chunk_index=0,
            chunk_text=text,
            start_offset=10,
            end_offset=10 + len(text),
            char_count=len(text),
            word_count=len(text.split()),
        )

        result = TextChunker.enforce_token_limit([chunk], max_tokens=250)
        assert len(result) == 1
        assert result[0].chunk_id == "doc1_c0"
        assert result[0].start_offset == 10
        assert result[0].end_offset == 10 + len(text)

    def test_oversized_chunk_split_and_offset_reconstruction(self, shared_tokenizer):
        """Oversized chunk (e.g., 400 tokens) must be split into subchunks <= 250 tokens.

        Every subchunk's offsets must exactly reconstruct its slice from the parent page text.
        """
        prefix = "Base offset padding text before chunk starts. "
        sentences = [
            f"Sentence number {i} discusses retrieval architecture, vector databases, and indexing."
            for i in range(1, 40)
        ]
        chunk_text = " ".join(sentences)
        full_page_text = prefix + chunk_text + " Extra trailing text on the page."

        start_offset = len(prefix)
        end_offset = start_offset + len(chunk_text)

        initial_tokens = count_tokens_text(chunk_text, tokenizer=shared_tokenizer)
        assert initial_tokens > 300, f"Expected >300 tokens, got {initial_tokens}"

        parent_chunk = DocumentChunk(
            chunk_id="parent_doc_c5",
            document_id="doc_xyz",
            filename="manual.pdf",
            page_number=3,
            page_references=[3],
            chunk_index=5,
            chunk_text=chunk_text,
            start_offset=start_offset,
            end_offset=end_offset,
            char_count=len(chunk_text),
            word_count=len(chunk_text.split()),
        )

        subchunks = TextChunker.split_oversized_chunk(
            parent_chunk,
            max_tokens=250,
            token_overlap=30,
            tokenizer=shared_tokenizer,
        )

        assert len(subchunks) > 1, "Oversized chunk must be split into multiple subchunks"

        for idx, sub in enumerate(subchunks):
            # Token count constraint
            sub_tokens = count_tokens_text(sub.chunk_text, tokenizer=shared_tokenizer)
            assert sub_tokens <= 250, f"Subchunk {idx} exceeded 250 tokens ({sub_tokens})"

            # Provenance constraints
            assert sub.document_id == "doc_xyz"
            assert sub.filename == "manual.pdf"
            assert sub.page_number == 3
            assert sub.page_references == [3]

            # Offset reconstruction: full_page_text[start:end] == sub.chunk_text
            reconstructed_slice = full_page_text[sub.start_offset : sub.end_offset]
            assert reconstructed_slice == sub.chunk_text, (
                f"Offset mismatch on subchunk {idx}: "
                f"expected '{sub.chunk_text[:30]}...', got '{reconstructed_slice[:30]}...'"
            )

    def test_enforce_token_limit_sequential_reindexing(self, shared_tokenizer):
        """Mixed list with small, oversized, and small chunks must be re-indexed sequentially."""
        long_text = " ".join(["algorithm", "vector", "search", "dimension", "similarity"] * 70)  # ~350 tokens

        c0 = DocumentChunk(
            chunk_id="c0",
            document_id="d1",
            filename="f.txt",
            page_number=1,
            page_references=[1],
            chunk_index=0,
            chunk_text="Small chunk one.",
            start_offset=0,
            end_offset=16,
            char_count=16,
            word_count=3,
        )
        c1 = DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            filename="f.txt",
            page_number=1,
            page_references=[1],
            chunk_index=1,
            chunk_text=long_text,
            start_offset=20,
            end_offset=20 + len(long_text),
            char_count=len(long_text),
            word_count=len(long_text.split()),
        )
        c2 = DocumentChunk(
            chunk_id="c2",
            document_id="d1",
            filename="f.txt",
            page_number=1,
            page_references=[1],
            chunk_index=2,
            chunk_text="Small chunk three.",
            start_offset=1000,
            end_offset=1018,
            char_count=18,
            word_count=3,
        )

        processed = TextChunker.enforce_token_limit(
            [c0, c1, c2],
            max_tokens=250,
            token_overlap=30,
            tokenizer=shared_tokenizer,
        )

        assert len(processed) > 3
        # Indices must be strictly 0, 1, 2, 3, ...
        for expected_idx, chunk in enumerate(processed):
            assert chunk.chunk_index == expected_idx


class TestEmbedderTokenGuard:
    """Test suite verifying embedder catches oversized inputs."""

    def test_embed_text_raises_on_token_overflow(self):
        """Passing text with > 256 tokens must raise EmbeddingTokenLimitError when auto_truncate=False."""
        embedder = SentenceTransformerEmbedder.get_instance()
        long_text = " ".join(["word"] * 300)

        with pytest.raises(EmbeddingTokenLimitError) as exc_info:
            embedder.embed_text(long_text, auto_truncate=False)

        assert "exceeds the model limit of 256" in str(exc_info.value)
        assert exc_info.value.token_count is not None
        assert exc_info.value.token_count > 256

    def test_embed_text_auto_truncate_succeeds(self):
        """Passing text with > 256 tokens succeeds when auto_truncate=True is explicitly set."""
        embedder = SentenceTransformerEmbedder.get_instance()
        long_text = " ".join(["word"] * 300)

        embedding = embedder.embed_text(long_text, auto_truncate=True)
        assert len(embedding) == 384
        assert pytest.approx(sum(x * x for x in embedding), rel=1e-3) == 1.0
