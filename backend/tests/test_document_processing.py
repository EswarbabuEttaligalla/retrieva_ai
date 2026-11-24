"""Comprehensive test suite for Phase 2 Document Ingestion and Chunking Engine."""

import os
import pytest
import fitz  # PyMuPDF

from backend.app.document_processing.chunker import TextChunker
from backend.app.document_processing.cleaner import TextCleaner
from backend.app.document_processing.exceptions import (
    ChunkingError,
    CorruptedPDFError,
    EmptyDocumentError,
    EncryptedPDFError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from backend.app.document_processing.extractor import DocumentExtractor
from backend.app.document_processing.identifier import (
    DocumentRegistry,
    compute_content_hash,
    generate_document_id,
)
from backend.app.document_processing.pipeline import DocumentIngestionPipeline


# ============================================================================
# Programmatic PDF Helpers for Reproducible Testing
# ============================================================================

def make_pdf(page_texts: list[str], password: str | None = None) -> bytes:
    """Programmatically generate a real in-memory PDF with PyMuPDF."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page(width=595, height=842)
        if text:
            # Insert text at standard margin
            page.insert_text((50, 72), text, fontsize=11)
    
    if password:
        pdf_bytes = doc.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="admin",
            user_pw=password,
        )
    else:
        pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ============================================================================
# Task 7.1: Single-Page PDF Extraction
# ============================================================================

def test_single_page_pdf_extraction():
    """Verify single-page PDF extracts correct text, page number, and metadata."""
    sample_text = "Retrieva AI provides accurate document question answering."
    pdf_bytes = make_pdf([sample_text])

    pages, meta = DocumentExtractor.extract(pdf_bytes, filename="single.pdf")

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert sample_text in pages[0].raw_text
    assert pages[0].has_text is True
    assert meta["page_count"] == 1
    assert "PDF" in meta["format"]


# ============================================================================
# Task 7.2: Multi-Page PDF Extraction
# ============================================================================

def test_multi_page_pdf_extraction():
    """Verify multi-page PDF extracts each page independently preserving order."""
    page_1 = "This is content from Page One of the architectural blueprint."
    page_2 = "This is content from Page Two detailing vector search algorithms."
    page_3 = "This is content from Page Three covering evaluation benchmarks."

    pdf_bytes = make_pdf([page_1, page_2, page_3])

    pages, meta = DocumentExtractor.extract(pdf_bytes, filename="multipage.pdf")

    assert len(pages) == 3
    assert meta["page_count"] == 3
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert pages[2].page_number == 3

    assert page_1 in pages[0].raw_text
    assert page_2 in pages[1].raw_text
    assert page_3 in pages[2].raw_text


# ============================================================================
# Task 7.3: TXT Extraction
# ============================================================================

def test_txt_extraction():
    """Verify UTF-8 TXT extraction preserves paragraphs and line breaks."""
    raw_txt = (
        "First paragraph introducing Retrieva AI.\n\n"
        "Second paragraph with technical details:\n"
        "- Vector databases\n"
        "- Cosine similarity\n"
    )
    txt_bytes = raw_txt.encode("utf-8")

    pages, meta = DocumentExtractor.extract(txt_bytes, filename="notes.txt")

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert meta["format"] == "TXT"
    assert "Vector databases" in pages[0].raw_text
    assert meta["encoding"] == "utf-8"


# ============================================================================
# Task 7.4: Empty Documents
# ============================================================================

def test_empty_documents():
    """Verify 0-byte files and whitespace-only files raise EmptyDocumentError."""
    # 0-byte PDF
    with pytest.raises(EmptyDocumentError):
        DocumentExtractor.extract(b"", filename="empty.pdf")

    # 0-byte TXT
    with pytest.raises(EmptyDocumentError):
        DocumentExtractor.extract(b"", filename="empty.txt")

    # Whitespace-only TXT
    with pytest.raises(EmptyDocumentError):
        DocumentExtractor.extract(b"   \n\n\t  ", filename="blank.txt")


# ============================================================================
# Task 7.5: Invalid and Corrupted PDF Files
# ============================================================================

def test_invalid_and_corrupted_pdf_files():
    """Verify malformed or corrupted byte streams raise CorruptedPDFError."""
    corrupted_bytes = b"%PDF-1.4\nGARBAGE_BYTES_THAT_CANNOT_BE_PARSED_AS_PDF"
    with pytest.raises(CorruptedPDFError):
        DocumentExtractor.extract(corrupted_bytes, filename="corrupted.pdf")


# ============================================================================
# Task 7.6: Password-Protected / Encrypted PDFs
# ============================================================================

def test_password_protected_pdf():
    """Verify encrypted/password-protected PDFs raise EncryptedPDFError."""
    encrypted_bytes = make_pdf(["Confidential executive summary"], password="SecretPassword123")

    with pytest.raises(EncryptedPDFError):
        DocumentExtractor.extract(encrypted_bytes, filename="confidential.pdf")


# ============================================================================
# Task 7.7: Unicode and Multilingual Content
# ============================================================================

def test_unicode_and_multilingual_content():
    """Verify handling of accents, mathematical symbols, and multilingual text."""
    unicode_content = (
        "Retrieva AI handles café, résumé, Zürich, and naïve.\n\n"
        "Greek symbols: α, β, γ, λ, and mathematical symbols: ∑, ∏, √x, ∫ f(x)dx.\n\n"
        "Multilingual phrases: こんにちは, 你好, مرحبا."
    )
    txt_bytes = unicode_content.encode("utf-8")

    pages, _ = DocumentExtractor.extract(txt_bytes, filename="unicode.txt")
    cleaned = TextCleaner.clean(pages[0].raw_text)

    assert "café" in cleaned
    assert "résumé" in cleaned
    assert "α, β, γ" in cleaned
    assert "∑" in cleaned
    assert "こんにちは" in cleaned


# ============================================================================
# Task 7.8: Text Cleaning and Normalization
# ============================================================================

def test_text_cleaning_pipeline():
    """Verify whitespace collapse, hyphenation repair, and excessive newline reduction."""
    messy_text = (
        "Paragraph   one   with    excessive     spaces.\t\t\n"
        "\n\n\n\n"
        "Para-\ngraph two demonstrates hyphen-\nation repair.\r\n"
        "Control \x00char\x07acters removed."
    )

    cleaned = TextCleaner.clean(messy_text)

    # 1. Multiple spaces collapsed
    assert "Paragraph one with excessive spaces." in cleaned
    # 2. Hyphenation repaired
    assert "Paragraph two demonstrates hyphenation repair." in cleaned
    # 3. Control characters stripped
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    # 4. Excessive newlines capped to \n\n
    assert "\n\n\n" not in cleaned


# ============================================================================
# Task 7.9: Chunk Size Constraints
# ============================================================================

def test_chunk_size_constraints():
    """Verify that generated chunks do not exceed the configured maximum chunk size."""
    long_text = (
        "Retrieval-Augmented Generation enhances generative AI by grounding model responses "
        "in authoritative, verifiable source documents. " * 15
    )
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)

    chunks = chunker.chunk_page(
        page_text=long_text,
        document_id="doc_test",
        filename="test.txt",
        page_number=1,
    )

    assert len(chunks) > 1
    for chunk in chunks:
        # Every chunk must respect the chunk size limit
        assert chunk.char_count <= 300
        assert len(chunk.chunk_text) <= 300


# ============================================================================
# Task 7.10: Chunk Overlap Correctness
# ============================================================================

def test_chunk_overlap_correctness():
    """Verify that successive chunks share overlapping text."""
    sample_text = (
        "Sentence one establishes the foundation. "
        "Sentence two introduces the vector space. "
        "Sentence three computes the cosine similarity. "
        "Sentence four generates the final prompt. "
        "Sentence five synthesizes the grounded answer."
    )
    chunker = TextChunker(chunk_size=120, chunk_overlap=40)

    chunks = chunker.chunk_page(
        page_text=sample_text,
        document_id="doc_test",
        filename="test.txt",
        page_number=1,
    )

    assert len(chunks) >= 3

    for i in range(len(chunks) - 1):
        curr_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        # The end of the current chunk must overlap with the start of the next chunk
        assert next_chunk.start_offset < curr_chunk.end_offset
        overlap_len = curr_chunk.end_offset - next_chunk.start_offset
        assert overlap_len > 0

        # Verify the shared text slice is identical
        shared_from_curr = curr_chunk.chunk_text[-overlap_len:]
        shared_from_next = next_chunk.chunk_text[:overlap_len]
        assert shared_from_curr == shared_from_next


# ============================================================================
# Task 7.11: Paragraph Boundary Handling
# ============================================================================

def test_paragraph_boundary_handling():
    """Verify chunker prefers splitting at paragraph boundaries (\n\n) over sentences."""
    p1 = "First cohesive paragraph detailing the mathematical properties of dense vector embeddings."
    p2 = "Second paragraph covering Approximate Nearest Neighbor search with HNSW graphs."
    p3 = "Third paragraph summarizing evaluation metrics including Precision@K and MRR."
    text = f"{p1}\n\n{p2}\n\n{p3}"

    # Set chunk size large enough to hold ~1.5 paragraphs
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_page(text, document_id="doc_p", filename="doc.txt", page_number=1)

    # Chunk 0 should break cleanly after paragraph 1
    assert chunks[0].chunk_text.endswith("\n\n") or chunks[0].chunk_text.strip() == p1


# ============================================================================
# Task 7.12: No Lost Text Across Chunk Boundaries (Reconstruction)
# ============================================================================

def test_no_lost_text_across_chunk_boundaries():
    """Verify complete original page text is reconstructed from chunks without omissions."""
    text = (
        "Natural language processing has transitioned toward dense retrieval models. "
        "Traditional inverted indices suffer from lexical mismatch when queries use synonyms. "
        "Dense retrieval maps queries and documents into a shared continuous semantic space. "
        "In this space, semantic similarity correlates with geometric proximity. "
        "Cosine similarity measures the angle between vectors independent of vector magnitude."
    )

    chunker = TextChunker(chunk_size=140, chunk_overlap=35)
    chunks = chunker.chunk_page(text, document_id="doc_recon", filename="recon.txt", page_number=1)

    reconstructed = TextChunker.reconstruct_page_text(chunks)
    assert reconstructed == text


# ============================================================================
# Task 7.13: Correct Page Provenance Tracking
# ============================================================================

def test_correct_page_provenance():
    """Verify chunks maintain strict page numbers and multi-page indexing."""
    page_1 = "Alpha content on page one explaining document ingestion."
    page_2 = "Beta content on page two explaining vector similarity search."

    pdf_bytes = make_pdf([page_1, page_2])
    registry = DocumentRegistry()
    pipeline = DocumentIngestionPipeline(registry=registry, chunk_size=200, chunk_overlap=40)

    result = pipeline.process(pdf_bytes, filename="provenance.pdf")

    assert result.status == "success"
    assert len(result.chunks) >= 2

    # Check page numbers match source
    page_1_chunks = [c for c in result.chunks if c.page_number == 1]
    page_2_chunks = [c for c in result.chunks if c.page_number == 2]

    assert len(page_1_chunks) >= 1
    assert len(page_2_chunks) >= 1

    for c in page_1_chunks:
        assert c.page_number == 1
        assert 1 in c.page_references
        assert "Alpha" in c.chunk_text

    for c in page_2_chunks:
        assert c.page_number == 2
        assert 2 in c.page_references
        assert "Beta" in c.chunk_text


# ============================================================================
# Task 7.14: Correct Start and End Offsets
# ============================================================================

def test_correct_start_and_end_offsets():
    """Verify page_text[start_offset:end_offset] == chunk_text for every generated chunk."""
    page_text = (
        "Modern RAG architecture decouples retrieval from generation. "
        "The retriever identifies relevant passages from a vector database. "
        "The generator conditions its output on the retrieved context. "
        "This prevents hallucinations and provides verifiable citations."
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=25)
    chunks = chunker.chunk_page(page_text, document_id="doc_off", filename="offsets.txt", page_number=1)

    for chunk in chunks:
        # Strict invariant check
        extracted_slice = page_text[chunk.start_offset : chunk.end_offset]
        assert extracted_slice == chunk.chunk_text
        assert chunk.char_count == len(chunk.chunk_text)
        assert chunk.end_offset > chunk.start_offset


# ============================================================================
# Task 7.15: Duplicate Document Detection
# ============================================================================

def test_duplicate_document_detection():
    """Verify SHA-256 duplicate detection rejects redundant document ingestion."""
    content = "Identical document content for testing SHA-256 collision and duplicate rejection."
    txt_bytes = content.encode("utf-8")

    registry = DocumentRegistry()
    pipeline = DocumentIngestionPipeline(registry=registry)

    # First ingestion -> success
    result_1 = pipeline.process(txt_bytes, filename="report_v1.txt")
    assert result_1.status == "success"
    assert result_1.is_duplicate is False
    assert registry.count() == 1

    # Second ingestion with same content but different filename -> detected as duplicate
    result_2 = pipeline.process(txt_bytes, filename="report_copy.txt")
    assert result_2.status == "duplicate"
    assert result_2.is_duplicate is True
    assert result_2.content_hash == result_1.content_hash
    assert result_2.document_id == result_1.document_id
    assert registry.count() == 1  # Registry count unchanged


# ============================================================================
# Task 7.16: Error Handling & Scanned PDF Detection
# ============================================================================

def test_error_handling_and_scanned_pdf():
    """Verify unsupported extensions and scanned/image-only PDFs raise appropriate errors."""
    # 1. Unsupported extension
    with pytest.raises(UnsupportedFileTypeError):
        DocumentExtractor.extract(b"Some text", filename="spreadsheet.xlsx")

    # 2. PDF with 0 extractable text (simulating scanned document)
    doc = fitz.open()
    doc.new_page()  # Page created with zero text
    scanned_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(NoExtractableTextError) as exc_info:
        DocumentExtractor.extract(scanned_bytes, filename="scanned.pdf")
    assert "scanned or image-only" in str(exc_info.value)

    # 3. Invalid chunking parameters
    with pytest.raises(ChunkingError):
        TextChunker(chunk_size=100, chunk_overlap=100)  # overlap == size

    with pytest.raises(ChunkingError):
        TextChunker(chunk_size=100, chunk_overlap=120)  # overlap > size

    with pytest.raises(ChunkingError):
        TextChunker(chunk_size=-10, chunk_overlap=5)  # negative size
