"""Regression tests for Milestone 3.16: 6-column multi-page PDF table extraction & continuation expansion."""

import json
from typing import Any, Dict, List
from uuid import UUID

import fitz
import pytest

from backend.app.document_processing.pipeline import DocumentIngestionPipeline
from backend.app.embeddings.embedder import SentenceTransformerEmbedder
from backend.app.llm.context_builder import ContextBuilder
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.retriever import Retriever
from backend.app.retrieval.tenant_bm25 import TenantBM25Manager
from backend.app.schemas.answer import AnswerRequest
from backend.app.schemas.retrieval import RetrievalConfiguration, RetrievalMode, RetrievalRequest
from backend.app.services.rag_service import RAGService
from backend.app.vector_store.store import ChromaVectorStore, TenantIsolationError

USER_A_ID = UUID("10000000-0000-4000-8000-000000000001")
USER_B_ID = UUID("10000000-0000-4000-8000-000000000002")

GROUND_TRUTH = [
    ("XGBoost", "Boosted Trees"),
    ("Random Forest", "Bagged Trees"),
    ("Multi-Layer Perceptron (MLP)", "Neural Network"),
    ("SVM (RBF Kernel)", "Kernel Margin"),
    ("Decision Tree", "Single Tree"),
    ("K-Nearest Neighbors (KNN)", "Instance-Based"),
    ("Logistic Regression", "Linear Model"),
    ("Gaussian Naive Bayes", "Probabilistic"),
]


def _draw_wide_table(page: fitz.Page, x0: float, y0: float, col_widths: List[float], row_height: float, rows: List[List[str]]) -> None:
    total_w = sum(col_widths)
    total_h = row_height * len(rows)
    for r_idx in range(len(rows) + 1):
        y = y0 + r_idx * row_height
        page.draw_line((x0, y), (x0 + total_w, y))
    x = x0
    for w in col_widths:
        page.draw_line((x, y0), (x, y0 + total_h))
        x += w
    page.draw_line((x0 + total_w, y0), (x0 + total_w, y0 + total_h))

    for r_idx, row in enumerate(rows):
        cx = x0
        cy = y0 + r_idx * row_height + row_height * 0.65
        for c_idx, cell in enumerate(row):
            page.insert_text((cx + 3, cy), cell[:28], fontsize=7)
            cx += col_widths[c_idx]


def _build_faithful_sic_pdf_bytes() -> bytes:
    """Faithful 5-page fixture reproducing the 6-column Section 5 Table 2 layout across pages 4-5."""
    doc = fitz.open()
    for page_text in (
        "TECHNICAL PROJECT REPORT: ML-Based SIC Decoding Order Selection.",
        "Section 2: NOMA System Model and SINR Formulation.",
        "Section 3: Dataset Generation and Feature Engineering.",
    ):
        p = doc.new_page()
        p.insert_text((40, 60), page_text, fontsize=11)

    # Page 4: Section 5 intro + rows 1-2 in wide 6-column Table 2
    p4 = doc.new_page()
    p4.insert_text(
        (40, 50),
        "5. Comparative Analysis of Candidate ML Models\n"
        "Eight distinct ML algorithms are selected across linear, probabilistic, instance-based,\n"
        "kernel, tree-ensemble, and neural network families (Table 2).",
        fontsize=10,
    )
    _draw_wide_table(
        p4,
        x0=30,
        y0=110,
        col_widths=[25, 110, 80, 120, 110, 75],
        row_height=26,
        rows=[
            ["#", "ML Algorithm", "Family", "Strengths for This Problem", "Limitations", "Suitability"],
            ["1", "XGBoost", "Boosted Trees", "Captures feature interactions", "Hyperparameter tuning", "Best Overall"],
            ["2", "Random Forest", "Bagged Trees", "High accuracy out-of-the-box", "Larger memory footprint", "Top Tier"],
        ],
    )

    # Page 5: rows 3-8 in wide 6-column Table 2
    p5 = doc.new_page()
    _draw_wide_table(
        p5,
        x0=30,
        y0=60,
        col_widths=[25, 110, 80, 120, 110, 75],
        row_height=26,
        rows=[
            ["#", "ML Algorithm", "Family", "Strengths for This Problem", "Limitations", "Suitability"],
            ["3", "Multi-Layer Perceptron (MLP)", "Neural Network", "Smooth continuous boundaries", "Sensitive to scaling", "Top Tier"],
            ["4", "SVM (RBF Kernel)", "Kernel Margin", "Optimal separating surfaces", "Scales poorly >30k rows", "Strong"],
            ["5", "Decision Tree", "Single Tree", "Exposes explicit thresholds", "Prone to overfitting", "Baseline"],
            ["6", "K-Nearest Neighbors (KNN)", "Instance-Based", "Zero training phase", "Slow test inference", "Moderate"],
            ["7", "Logistic Regression", "Linear Model", "Ultra-fast linear benchmark", "Cannot model curves", "Baseline"],
            ["8", "Gaussian Naive Bayes", "Probabilistic", "Minimal compute overhead", "Assumes independence", "Baseline"],
        ],
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TableRowLLM:
    """Deterministic grounded LLM double reading numbered Markdown table rows from context."""

    model = "gpt-4o-mini"

    def _synthesize(self, user_prompt: str) -> Dict[str, Any]:
        rows: List[tuple[str, str, str, str]] = []
        current_src = "[Source 1]"
        seen_algos = set()
        for line in user_prompt.splitlines():
            if line.startswith("--- [Source ") and "]" in line:
                current_src = line[4 : line.index("]") + 1]
                continue
            stripped = line.strip()
            if stripped.startswith("|") and "---" not in stripped:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cells) >= 3 and cells[0].isdigit():
                    algo, family = cells[1], cells[2]
                    if algo not in seen_algos:
                        seen_algos.add(algo)
                        rows.append((cells[0], algo, family, current_src))

        if not rows:
            return {
                "answer": "The provided documents do not contain sufficient information to answer this question.",
                "insufficient_context": True,
                "citations": [],
            }

        lines = ["| # | ML Algorithm | Family | Source |", "| --- | --- | --- | --- |"]
        citations: List[Dict[str, str]] = []
        seen_src = set()
        for num, algo, fam, src in sorted(rows, key=lambda r: int(r[0])):
            lines.append(f"| {num} | {algo} | {fam} | {src} |")
            if src not in seen_src:
                citations.append({"source_id": src, "quote": algo, "claim": f"{algo} ({fam})"})
                seen_src.add(src)
        return {
            "answer": "\n".join(lines),
            "insufficient_context": False,
            "citations": citations,
        }

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        payload = self._synthesize(messages[-1]["content"])
        return {"content": json.dumps(payload), "model": self.model, "prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620}

    async def generate_stream_async(self, messages: List[Dict[str, str]], **kwargs):
        yield self._synthesize(messages[-1]["content"])["answer"]


class TestActualPdfTableDiagnosticM316:
    """Verifies multi-page wide table retrieval, continuation expansion, and full 8-row synthesis."""

    @pytest.mark.asyncio
    async def test_wide_table_survives_extraction_retrieval_expansion_and_synthesis(self, tmp_path):
        pdf_bytes = _build_faithful_sic_pdf_bytes()
        pipeline = DocumentIngestionPipeline()
        ingested = pipeline.process(pdf_bytes, filename="ML-Based_SIC_Decoding_Order_Selection_Report.pdf")

        # Verify all 8 algorithms and families survived extraction, cleaning, and chunking
        combined = "\n".join(c.chunk_text for c in ingested.chunks if c.page_number in {4, 5})
        for algo, family in GROUND_TRUTH:
            assert algo in combined
            assert family in combined

        # Index into isolated ChromaDB + TenantBM25
        store = ChromaVectorStore(persist_directory=str(tmp_path / "chroma_m316"), collection_name="m316_test")
        embedder = SentenceTransformerEmbedder.get_instance()
        embeddings = embedder.embed_batch([c.chunk_text for c in ingested.chunks])
        store.add_chunks(ingested.chunks, embeddings, owner_id=USER_A_ID)

        dense = DenseRetriever(embedder=embedder, vector_store=store)
        bm25_mgr = TenantBM25Manager(vector_store=store)
        retriever = Retriever(dense_retriever=dense, tenant_bm25_manager=bm25_mgr, vector_store=store)

        query = (
            "What are all eight machine learning algorithms listed in Section 5? "
            "Give their names and ML families in a numbered table, and cite the relevant pages."
        )

        # Verify DENSE, BM25, and HYBRID + table continuation expansion all cover all 8 rows
        for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
            res = retriever.retrieve(
                RetrievalRequest(
                    query=query,
                    config=RetrievalConfiguration(mode=mode, top_k=7, candidate_k=20, document_id_filter=ingested.document_id),
                ),
                owner_id=USER_A_ID,
            )
            expanded = retriever.expand_table_continuations(res.chunks, owner_id=USER_A_ID)
            ctx = ContextBuilder(max_context_tokens=2500, model_name="gpt-4o-mini").build_context(expanded)
            for algo, family in GROUND_TRUTH:
                assert algo in ctx.formatted_context, f"{algo} missing in {mode.value}"
                assert family in ctx.formatted_context, f"{family} missing in {mode.value}"

        # Verify RAGService non-streaming & streaming enumerate all 8 algorithms and cite pages 4 & 5
        svc = RAGService(retriever=retriever, llm_client=TableRowLLM(), context_builder=ContextBuilder(max_context_tokens=2500, model_name="gpt-4o-mini"))
        resp = svc.answer_question(
            AnswerRequest(
                query=query,
                document_ids=[ingested.document_id],
                retrieval_config=RetrievalConfiguration(mode=RetrievalMode.HYBRID, top_k=7, candidate_k=20),
            ),
            owner_id=USER_A_ID,
        )
        assert resp.insufficient_context is False
        for algo, family in GROUND_TRUTH:
            assert algo in resp.answer
            assert family in resp.answer
        cited_pages = {p for c in resp.citations for p in c.page_references}
        assert {4, 5}.issubset(cited_pages)

        stream_out = "".join(
            [
                ev
                async for ev in svc.answer_question_stream(
                    AnswerRequest(
                        query=query,
                        document_ids=[ingested.document_id],
                        retrieval_config=RetrievalConfiguration(mode=RetrievalMode.HYBRID, top_k=7, candidate_k=20),
                    ),
                    owner_id=USER_A_ID,
                )
            ]
        )
        for algo, family in GROUND_TRUTH:
            assert algo in stream_out
            assert family in stream_out
        assert '"insufficient_context": false' in stream_out

        # Tenant isolation: User B cannot retrieve or expand User A's table
        isolated = svc.answer_question(AnswerRequest(query=query, document_ids=[ingested.document_id]), owner_id=USER_B_ID)
        assert isolated.insufficient_context is True
        assert isolated.sources_used == []
