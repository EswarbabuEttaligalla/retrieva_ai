"""Regression tests for Milestone 3.15: Table-Aware PDF Extraction, Chunking, and Answer Synthesis."""

import json
from typing import Any, Dict, List, Tuple
from uuid import UUID

import fitz
import pytest

from backend.app.document_processing.chunker import TextChunker
from backend.app.document_processing.pipeline import DocumentIngestionPipeline
from backend.app.llm.context_builder import ContextBuilder
from backend.app.llm.prompt_builder import STREAMING_SYSTEM_PROMPT, SYSTEM_PROMPT
from backend.app.retrieval.bm25 import BM25Index
from backend.app.schemas.answer import AnswerRequest
from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import (
    RetrievalConfiguration,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)
from backend.app.services.rag_service import RAGService
from backend.app.vector_store.store import TenantIsolationError

USER_A_ID = UUID("10000000-0000-4000-8000-000000000001")
USER_B_ID = UUID("10000000-0000-4000-8000-000000000002")

EXPECTED_ALGORITHMS = [
    "XGBoost",
    "Random Forest",
    "Multi-Layer Perceptron (MLP)",
    "SVM (RBF Kernel)",
    "Decision Tree",
    "K-Nearest Neighbors (KNN)",
    "Logistic Regression",
    "Gaussian Naive Bayes",
]


def _draw_pdf_table(page: fitz.Page, x0: float, y0: float, col_widths: List[float], row_height: float, rows: List[List[str]]) -> None:
    """Draw a bordered table with text cells so PyMuPDF find_tables() detects its structure."""
    total_w = sum(col_widths)
    total_h = row_height * len(rows)
    # Horizontal lines
    for r_idx in range(len(rows) + 1):
        y = y0 + r_idx * row_height
        page.draw_line((x0, y), (x0 + total_w, y))
    # Vertical lines
    x = x0
    for w in col_widths:
        page.draw_line((x, y0), (x, y0 + total_h))
        x += w
    page.draw_line((x0 + total_w, y0), (x0 + total_w, y0 + total_h))

    for r_idx, row in enumerate(rows):
        cx = x0
        cy = y0 + r_idx * row_height + row_height * 0.68
        for c_idx, cell in enumerate(row):
            page.insert_text((cx + 4, cy), cell, fontsize=9)
            cx += col_widths[c_idx]


def _build_sic_report_pdf_bytes() -> bytes:
    """Create a 5-page PDF mirroring Section 5 (pages 4-5) of the SIC Decoding Report."""
    doc = fitz.open()
    for p_idx, intro in enumerate(
        [
            "ML-Based SIC Decoding Order Selection Report\nExecutive Overview.",
            "Section 2: NOMA System Model and Successive Interference Cancellation.",
            "Section 3: Channel Gain Feature Engineering and Dataset Preparation.",
        ],
        start=1,
    ):
        page = doc.new_page()
        page.insert_text((50, 60), intro, fontsize=11)

    # Page 4: prose mentioning only MLP and SVM + Table 1 (rows 1-4)
    p4 = doc.new_page()
    p4.insert_text(
        (50, 50),
        "Section 5: Machine Learning Models Evaluated\n"
        "In preliminary tuning, MLP and SVM showed strong non-linear decision boundaries.\n"
        "Table 1 summarizes the first four evaluated ML algorithms:",
        fontsize=10,
    )
    _draw_pdf_table(
        p4,
        x0=50,
        y0=110,
        col_widths=[180, 120, 90],
        row_height=24,
        rows=[
            ["Algorithm", "Family", "Accuracy"],
            ["XGBoost", "Boosting", "94.2%"],
            ["Random Forest", "Bagging", "93.1%"],
            ["Multi-Layer Perceptron (MLP)", "Neural Net", "92.8%"],
            ["SVM (RBF Kernel)", "Kernel", "91.9%"],
        ],
    )

    # Page 5: Table 1 continued (rows 5-8)
    p5 = doc.new_page()
    p5.insert_text(
        (50, 50),
        "Section 5 (continued): Remaining ML Algorithms Evaluated in Table 1:",
        fontsize=10,
    )
    _draw_pdf_table(
        p5,
        x0=50,
        y0=85,
        col_widths=[180, 120, 90],
        row_height=24,
        rows=[
            ["Algorithm", "Family", "Accuracy"],
            ["Decision Tree", "Tree", "89.4%"],
            ["K-Nearest Neighbors (KNN)", "Instance", "88.7%"],
            ["Logistic Regression", "Linear", "86.5%"],
            ["Gaussian Naive Bayes", "Probabilistic", "84.1%"],
        ],
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class ChunkBackedRetriever:
    """BM25-backed fixture retriever over real DocumentIngestionPipeline chunks with owner isolation."""

    def __init__(self, chunks_by_owner: Dict[str, List[DocumentChunk]]) -> None:
        self._chunks_by_owner = chunks_by_owner
        self._indexes: Dict[str, BM25Index] = {}
        for owner_str, chunks in chunks_by_owner.items():
            idx = BM25Index()
            idx.build_index(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "filename": c.filename,
                        "page_number": c.page_number,
                        "page_references": c.page_references,
                        "chunk_text": c.chunk_text,
                        "start_offset": c.start_offset,
                        "end_offset": c.end_offset,
                        "metadata": {
                            "owner_id": owner_str,
                            "document_id": c.document_id,
                            "page_number": c.page_number,
                            "page_references": c.page_references,
                        },
                    }
                    for c in chunks
                ]
            )
            self._indexes[owner_str] = idx

    def retrieve(self, request: RetrievalRequest, owner_id: UUID) -> RetrievalResult:
        owner_str = str(owner_id)
        index = self._indexes.get(owner_str)
        if index is None:
            return RetrievalResult(
                query=request.query,
                mode=request.config.mode,
                total_candidates_found=0,
                returned_chunks_count=0,
                chunks=[],
                execution_time_ms=1.0,
                configuration_applied=request.config.model_dump(),
            )
        hits = index.search(
            query=request.query,
            top_k=request.config.top_k,
            document_id_filter=request.config.document_id_filter,
            document_ids_filter=request.config.document_ids_filter,
        )
        for h in hits:
            if (h.metadata or {}).get("owner_id") != owner_str:
                raise TenantIsolationError("Cross-tenant chunk rejected.")
        return RetrievalResult(
            query=request.query,
            mode=request.config.mode,
            total_candidates_found=len(hits),
            returned_chunks_count=len(hits),
            chunks=hits,
            execution_time_ms=1.5,
            configuration_applied=request.config.model_dump(),
        )


class TableEnumeratingLLM:
    """Grounded LLM double that extracts markdown table rows from prompt context."""

    model = "gpt-4o-mini"

    @staticmethod
    def _extract_algorithms_with_sources(user_prompt: str) -> List[Tuple[str, str]]:
        found: List[Tuple[str, str]] = []
        current_source = "[Source 1]"
        for line in user_prompt.splitlines():
            if line.startswith("--- [Source ") and "]" in line:
                end = line.index("]") + 1
                current_source = line[4:end]
                continue
            stripped = line.strip()
            if stripped.startswith("|") and "---" not in stripped:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if cells and cells[0] and cells[0].lower() != "algorithm":
                    found.append((cells[0], current_source))
        return found

    def _synthesize(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        sys_prompt = messages[0]["content"]
        assert "enumerate ALL matching items" in sys_prompt

        user_prompt = messages[-1]["content"]
        items = self._extract_algorithms_with_sources(user_prompt)
        if not items:
            return {
                "answer": "The provided documents do not contain sufficient information to answer this question.",
                "insufficient_context": True,
                "citations": [],
            }

        lines = ["The 8 ML algorithms stated in the document are:"]
        citations: List[Dict[str, str]] = []
        seen_sources = set()
        for idx, (algo, src_tag) in enumerate(items, start=1):
            lines.append(f"{idx}. {algo} {src_tag}")
            if src_tag not in seen_sources:
                citations.append(
                    {
                        "source_id": src_tag,
                        "quote": algo,
                        "claim": f"Algorithm {algo} listed in Table 1",
                    }
                )
                seen_sources.add(src_tag)

        return {
            "answer": "\n".join(lines),
            "insufficient_context": False,
            "citations": citations,
        }

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        payload = self._synthesize(messages)
        return {
            "content": json.dumps(payload),
            "model": self.model,
            "prompt_tokens": 600,
            "completion_tokens": 140,
            "total_tokens": 740,
        }

    async def generate_stream_async(self, messages: List[Dict[str, str]], **kwargs):
        payload = self._synthesize(messages)
        yield payload["answer"]


class TestTableAwareRetrievalM315:
    """Verifies PDF table extraction, row-preserving chunking, and full 8-algorithm synthesis."""

    def test_pdf_table_extraction_preserves_all_8_algorithms_and_chunk_offsets(self):
        pdf_bytes = _build_sic_report_pdf_bytes()
        pipeline = DocumentIngestionPipeline()
        result = pipeline.process(pdf_bytes, filename="ML-Based SIC Decoding Order Selection Report.pdf")

        p4_text = next(p.cleaned_text for p in result.raw_pages if p.page_number == 4)
        p5_text = next(p.cleaned_text for p in result.raw_pages if p.page_number == 5)

        # Verify Markdown table structure preserved on pages 4 and 5
        assert "| Algorithm | Family | Accuracy |" in p4_text
        assert "| XGBoost | Boosting | 94.2% |" in p4_text
        assert "| Decision Tree | Tree | 89.4% |" in p5_text

        # Verify all 8 algorithms survived extraction and chunking without truncation
        p4_p5_chunks = [c for c in result.chunks if c.page_number in {4, 5}]
        combined_chunk_text = "\n".join(c.chunk_text for c in p4_p5_chunks)
        for algo in EXPECTED_ALGORITHMS:
            assert algo in combined_chunk_text

        # Verify exact offset & reconstruction invariants remain unbroken
        for page in result.raw_pages:
            page_chunks = [c for c in result.chunks if c.page_number == page.page_number]
            for c in page_chunks:
                assert page.cleaned_text[c.start_offset : c.end_offset] == c.chunk_text
            assert TextChunker.reconstruct_page_text(page_chunks) == page.cleaned_text

    @pytest.mark.asyncio
    async def test_qa_enumerates_all_8_algorithms_with_pages_4_and_5_citations_and_isolation(self):
        pdf_bytes = _build_sic_report_pdf_bytes()
        pipeline = DocumentIngestionPipeline()
        ingested = pipeline.process(pdf_bytes, filename="ML-Based SIC Decoding Order Selection Report.pdf")

        retriever = ChunkBackedRetriever({str(USER_A_ID): ingested.chunks})
        svc = RAGService(
            retriever=retriever,
            llm_client=TableEnumeratingLLM(),
            context_builder=ContextBuilder(max_context_tokens=2500, model_name="gpt-4o-mini"),
        )
        query = "What are the 8 ML algorithms that are stated?"

        # Non-streaming QA
        resp = svc.answer_question(
            AnswerRequest(
                query=query,
                document_ids=[ingested.document_id],
                retrieval_config=RetrievalConfiguration(mode=RetrievalMode.BM25, top_k=7),
            ),
            owner_id=USER_A_ID,
        )
        assert resp.insufficient_context is False
        for algo in EXPECTED_ALGORITHMS:
            assert algo in resp.answer
        assert "CatBoost" not in resp.answer
        assert "LightGBM" not in resp.answer

        cited_pages = {p for c in resp.citations for p in c.page_references}
        assert {4, 5}.issubset(cited_pages)
        assert {c.chunk_id for c in resp.citations}.issubset({s.chunk_id for s in resp.sources_used})

        # Streaming QA consistency
        stream_events = [
            ev
            async for ev in svc.answer_question_stream(
                AnswerRequest(
                    query=query,
                    document_ids=[ingested.document_id],
                    retrieval_config=RetrievalConfiguration(mode=RetrievalMode.BM25, top_k=7),
                ),
                owner_id=USER_A_ID,
            )
        ]
        joined = "".join(stream_events)
        for algo in EXPECTED_ALGORITHMS:
            assert algo in joined
        assert '"insufficient_context": false' in joined

        # Prompt rules check
        assert "enumerate ALL matching items" in SYSTEM_PROMPT
        assert "enumerate ALL matching items" in STREAMING_SYSTEM_PROMPT

        # Tenant and document isolation
        other_user = svc.answer_question(
            AnswerRequest(query=query, document_ids=[ingested.document_id]),
            owner_id=USER_B_ID,
        )
        assert other_user.insufficient_context is True
        assert other_user.sources_used == []

        wrong_doc = svc.answer_question(
            AnswerRequest(query=query, document_ids=["unrelated-doc-id"]),
            owner_id=USER_A_ID,
        )
        assert wrong_doc.insufficient_context is True
        assert wrong_doc.sources_used == []
