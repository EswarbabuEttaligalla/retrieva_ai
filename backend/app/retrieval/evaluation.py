"""Retrieval evaluation engine and benchmark suite for Retrieva AI.

Calculates Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and Hit Rate@K
across Dense, BM25, and Hybrid retrieval strategies over a 48-chunk, 20-query ground-truth benchmark.
"""

from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID
from pydantic import BaseModel

EVAL_OWNER_ID = UUID("00000000-0000-4000-8000-000000000000")

try:
    from backend.app.core.logging import get_logger
    from backend.app.retrieval.retriever import Retriever
    from backend.app.schemas.retrieval import RetrievalConfiguration, RetrievalMode, RetrievalRequest
except ImportError:
    from app.core.logging import get_logger
    from app.retrieval.retriever import Retriever
    from app.schemas.retrieval import RetrievalConfiguration, RetrievalMode, RetrievalRequest

logger = get_logger("retrieval.evaluation")


class EvaluationQuery(BaseModel):
    """Ground truth query with relevant chunk targets and evaluation intent."""

    query_id: str
    query_text: str
    expected_chunk_ids: List[str]
    intent_category: str  # "exact_keyword", "semantic_paraphrase", "technical_fact", "conceptual"


# ============================================================================
# Labeled Evaluation Corpus (48 Chunks across 6 Technical Domains)
# ============================================================================

EVALUATION_DOCUMENTS: List[Dict[str, Any]] = [
    # Domain 1: Vector Databases & Indexing (c1 - c8)
    {
        "chunk_id": "c1",
        "document_id": "doc_vectordb",
        "filename": "vector_databases.txt",
        "text": "ChromaDB provides a persistent vector store client utilizing an SQLite database for scalar metadata storage and an embedded HNSW library for nearest neighbor vector search.",
        "metadata": {"document_id": "doc_vectordb", "filename": "vector_databases.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c2",
        "document_id": "doc_vectordb",
        "filename": "vector_databases.txt",
        "text": "Hierarchical Navigable Small World graphs achieve logarithmic approximate nearest neighbor search complexity O(log N) by maintaining multi-layer geometric skip-list structures.",
        "metadata": {"document_id": "doc_vectordb", "filename": "vector_databases.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c3",
        "document_id": "doc_vectordb",
        "filename": "vector_databases.txt",
        "text": "ChromaDB cosine distance measures directional separation: cosine_distance = 1.0 - cosine_similarity. Lower distance values indicate greater semantic relevance.",
        "metadata": {"document_id": "doc_vectordb", "filename": "vector_databases.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c4",
        "document_id": "doc_vectordb",
        "filename": "vector_databases.txt",
        "text": "When vector embeddings are normalized to unit L2 norm, the Euclidean distance squared is directly proportional to cosine distance: ||u - v||^2 = 2 * (1 - cos(theta)).",
        "metadata": {"document_id": "doc_vectordb", "filename": "vector_databases.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c5",
        "document_id": "doc_distractor_db",
        "filename": "faiss_clustering.txt",
        "text": "Faiss Inverted File with Product Quantization compresses high dimensional vectors into compact byte codes by clustering vector space into Voronoi cells.",
        "metadata": {"document_id": "doc_distractor_db", "filename": "faiss_clustering.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c6",
        "document_id": "doc_distractor_db",
        "filename": "graph_databases.txt",
        "text": "Neo4j graph database executes declarative Cypher queries traversing nodes and relationship edges for knowledge graph entity resolution.",
        "metadata": {"document_id": "doc_distractor_db", "filename": "graph_databases.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c7",
        "document_id": "doc_hardware",
        "filename": "hardware_acceleration.txt",
        "text": "SIMD AVX-512 hardware vectorization instructions accelerate floating point matrix multiplication and dot products across CPU vector registers.",
        "metadata": {"document_id": "doc_hardware", "filename": "hardware_acceleration.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c8",
        "document_id": "doc_vectordb",
        "filename": "vector_databases.txt",
        "text": "ChromaDB metadata filtering restricts search spaces using scalar equality clauses such as where={'document_id': 'doc_123'} prior to distance sorting.",
        "metadata": {"document_id": "doc_vectordb", "filename": "vector_databases.txt", "page_number": 3, "page_references": [3]},
    },

    # Domain 2: BM25 & Sparse Information Retrieval (c9 - c16)
    {
        "chunk_id": "c9",
        "document_id": "doc_bm25",
        "filename": "bm25_theory.txt",
        "text": "The Okapi BM25 ranking algorithm controls term frequency saturation through parameter k1, ensuring repeated keyword occurrences yield diminishing score returns.",
        "metadata": {"document_id": "doc_bm25", "filename": "bm25_theory.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c10",
        "document_id": "doc_bm25",
        "filename": "bm25_theory.txt",
        "text": "BM25 document length normalization parameter b penalizes verbose text chunks while rewarding concise documents matching query terms relative to avgdl.",
        "metadata": {"document_id": "doc_bm25", "filename": "bm25_theory.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c11",
        "document_id": "doc_bm25",
        "filename": "bm25_theory.txt",
        "text": "Robertson-Spärck Jones Inverse Document Frequency assigns higher statistical weight to rare keywords using formula ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5)).",
        "metadata": {"document_id": "doc_bm25", "filename": "bm25_theory.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c12",
        "document_id": "doc_bm25",
        "filename": "bm25_theory.txt",
        "text": "An inverted index maps normalized vocabulary tokens to posting lists containing document identifiers and within-document term frequencies.",
        "metadata": {"document_id": "doc_bm25", "filename": "bm25_theory.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c13",
        "document_id": "doc_distractor_ir",
        "filename": "pagerank.txt",
        "text": "Google PageRank models random surfer link transitions across directed web hyperlinks, assigning static authority scores based on eigenvector centrality.",
        "metadata": {"document_id": "doc_distractor_ir", "filename": "pagerank.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c14",
        "document_id": "doc_distractor_ir",
        "filename": "elasticsearch_arch.txt",
        "text": "ElasticSearch distributes search shard replicas across cluster nodes while background Lucene merge threads consolidate immutable segment files.",
        "metadata": {"document_id": "doc_distractor_ir", "filename": "elasticsearch_arch.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c15",
        "document_id": "doc_bm25",
        "filename": "bm25_theory.txt",
        "text": "Alphanumeric lowercasing tokenization strips irrelevant punctuation symbols while preserving compound technical identifiers and numerical digits.",
        "metadata": {"document_id": "doc_bm25", "filename": "bm25_theory.txt", "page_number": 3, "page_references": [3]},
    },
    {
        "chunk_id": "c16",
        "document_id": "doc_fusion",
        "filename": "rank_fusion.txt",
        "text": "Reciprocal Rank Fusion merges rankings from disparate dense and sparse search retrievers using rank reciprocals RRF(d) = sum(1 / (k + rank_m(d))).",
        "metadata": {"document_id": "doc_fusion", "filename": "rank_fusion.txt", "page_number": 1, "page_references": [1]},
    },

    # Domain 3: Cross-Encoders & Transformers (c17 - c24)
    {
        "chunk_id": "c17",
        "document_id": "doc_crossencoder",
        "filename": "cross_encoders.txt",
        "text": "Cross-encoders evaluate query and document candidate pairs jointly with full multi-head cross-attention, providing superior reranking precision.",
        "metadata": {"document_id": "doc_crossencoder", "filename": "cross_encoders.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c18",
        "document_id": "doc_crossencoder",
        "filename": "cross_encoders.txt",
        "text": "Bi-encoders encode queries and documents independently into fixed-length vectors, enabling scalable sub-millisecond retrieval across millions of indexed passages.",
        "metadata": {"document_id": "doc_crossencoder", "filename": "cross_encoders.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c19",
        "document_id": "doc_crossencoder",
        "filename": "cross_encoders.txt",
        "text": "The cross-encoder model ms-marco-MiniLM-L-6-v2 was trained on MS MARCO passage ranking to predict direct query-document relevance logits.",
        "metadata": {"document_id": "doc_crossencoder", "filename": "cross_encoders.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c20",
        "document_id": "doc_transformer",
        "filename": "transformer_attention.txt",
        "text": "Standard Transformer multi-head self-attention exhibits quadratic computational complexity O(N^2) in memory and runtime relative to token sequence length N.",
        "metadata": {"document_id": "doc_transformer", "filename": "transformer_attention.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c21",
        "document_id": "doc_distractor_ml",
        "filename": "cnn_vision.txt",
        "text": "Convolutional neural network feature maps apply spatial 2D filters and max pooling operations to extract translation-invariant visual edges.",
        "metadata": {"document_id": "doc_distractor_ml", "filename": "cnn_vision.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c22",
        "document_id": "doc_distractor_ml",
        "filename": "lstm_rnn.txt",
        "text": "Long Short-Term Memory networks employ multiplicative input, forget, and output gates to regulate gradient backpropagation across long temporal recurrence steps.",
        "metadata": {"document_id": "doc_distractor_ml", "filename": "lstm_rnn.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c23",
        "document_id": "doc_transformer",
        "filename": "transformer_attention.txt",
        "text": "The all-MiniLM-L6-v2 model supports a maximum sequence length of 256 tokens; exceeding inputs are truncated by the tokenizer, causing content past token 256 to disappear.",
        "metadata": {"document_id": "doc_transformer", "filename": "transformer_attention.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c24",
        "document_id": "doc_transformer",
        "filename": "transformer_attention.txt",
        "text": "Subword tokenizers decompose rare technical terminology into frequent constituent WordPiece or Byte-Pair subword tokens to maintain a bounded vocabulary.",
        "metadata": {"document_id": "doc_transformer", "filename": "transformer_attention.txt", "page_number": 2, "page_references": [2]},
    },

    # Domain 4: Document Ingestion & Chunking (c25 - c32)
    {
        "chunk_id": "c25",
        "document_id": "doc_ingestion",
        "filename": "document_processing.txt",
        "text": "PyMuPDF accesses internal PDF character glyph streams and bounding box coordinates directly, extracting clean raw text without invoking optical character recognition.",
        "metadata": {"document_id": "doc_ingestion", "filename": "document_processing.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c26",
        "document_id": "doc_ingestion",
        "filename": "document_processing.txt",
        "text": "Unicode NFKC normalization decomposes ligature glyphs and converts irregular carriage returns and non-breaking spaces into standardized UTF-8 characters.",
        "metadata": {"document_id": "doc_ingestion", "filename": "document_processing.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c27",
        "document_id": "doc_ingestion",
        "filename": "document_processing.txt",
        "text": "Sliding window text chunking prioritizes natural paragraph and sentence punctuation boundaries, preventing arbitrary mid-word or mid-sentence semantic truncation.",
        "metadata": {"document_id": "doc_ingestion", "filename": "document_processing.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c28",
        "document_id": "doc_ingestion",
        "filename": "document_processing.txt",
        "text": "SHA-256 cryptographic content digests uniquely identify ingested files, enabling instant detection and deduplication of identically uploaded document content.",
        "metadata": {"document_id": "doc_ingestion", "filename": "document_processing.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c29",
        "document_id": "doc_distractor_ingest",
        "filename": "tesseract_ocr.txt",
        "text": "Tesseract OCR binarizes scanned raster bitmaps using Otsu thresholding before feeding character contours into an LSTM optical character classifier.",
        "metadata": {"document_id": "doc_distractor_ingest", "filename": "tesseract_ocr.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c30",
        "document_id": "doc_distractor_ingest",
        "filename": "markdown_ast.txt",
        "text": "CommonMark AST parsers construct an Abstract Syntax Tree representing nested markdown header, blockquote, and unordered list hierarchy.",
        "metadata": {"document_id": "doc_distractor_ingest", "filename": "markdown_ast.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c31",
        "document_id": "doc_ingestion",
        "filename": "document_processing.txt",
        "text": "Provenance offset tracking preserves start_offset and end_offset character coordinates, allowing lossless reconstruction of original page text from overlapping chunks.",
        "metadata": {"document_id": "doc_ingestion", "filename": "document_processing.txt", "page_number": 3, "page_references": [3]},
    },
    {
        "chunk_id": "c32",
        "document_id": "doc_ingestion",
        "filename": "document_processing.txt",
        "text": "Token-aware subchunking splits oversized chunks into subchunks guaranteed to fit within the 256-token embedding capacity while preserving parent page offsets.",
        "metadata": {"document_id": "doc_ingestion", "filename": "document_processing.txt", "page_number": 3, "page_references": [3]},
    },

    # Domain 5: Programming Languages & Systems Architecture (c33 - c40)
    {
        "chunk_id": "c33",
        "document_id": "doc_python_mem",
        "filename": "python_memory.txt",
        "text": "Python memory management relies primarily on deterministic reference counting, supplemented by a cyclic generational garbage collector that detects isolated reference cycles.",
        "metadata": {"document_id": "doc_python_mem", "filename": "python_memory.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c34",
        "document_id": "doc_distractor_sys",
        "filename": "java_garbage_collection.txt",
        "text": "Java JVM ZGC and G1GC garbage collectors perform concurrent colored-pointer mark and sweep phases to eliminate stop-the-world application latency pauses.",
        "metadata": {"document_id": "doc_distractor_sys", "filename": "java_garbage_collection.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c35",
        "document_id": "doc_fastapi",
        "filename": "fastapi_async.txt",
        "text": "FastAPI leverages Python asynchronous asyncio event loops and ASGI lifespan context managers to handle non-blocking concurrent HTTP requests.",
        "metadata": {"document_id": "doc_fastapi", "filename": "fastapi_async.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c36",
        "document_id": "doc_distractor_sys",
        "filename": "flask_wsgi.txt",
        "text": "Flask WSGI synchronous applications process incoming client connections by allocating separate operating system threads per concurrent HTTP request.",
        "metadata": {"document_id": "doc_distractor_sys", "filename": "flask_wsgi.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c37",
        "document_id": "doc_fastapi",
        "filename": "fastapi_async.txt",
        "text": "Pydantic v2 accelerates schema data validation and JSON serialization through compiled pydantic-core Rust routines with static type annotations.",
        "metadata": {"document_id": "doc_fastapi", "filename": "fastapi_async.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c38",
        "document_id": "doc_python_mem",
        "filename": "python_memory.txt",
        "text": "Thread-safe data structures in Python utilize threading.Lock primitives to serialize concurrent thread access and avoid race conditions on shared memory dictionaries.",
        "metadata": {"document_id": "doc_python_mem", "filename": "python_memory.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c39",
        "document_id": "doc_distractor_sys",
        "filename": "rust_memory.txt",
        "text": "The Rust compiler enforces memory safety without a garbage collector through static ownership rules, linear lifetimes, and compile-time borrow checker verification.",
        "metadata": {"document_id": "doc_distractor_sys", "filename": "rust_memory.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c40",
        "document_id": "doc_fastapi",
        "filename": "fastapi_async.txt",
        "text": "Uvicorn manages production worker processes, binding asynchronous socket connections to the uvloop libuv event loop implementation.",
        "metadata": {"document_id": "doc_fastapi", "filename": "fastapi_async.txt", "page_number": 3, "page_references": [3]},
    },

    # Domain 6: Retrieval Evaluation & Information Theory (c41 - c48)
    {
        "chunk_id": "c41",
        "document_id": "doc_eval",
        "filename": "evaluation_metrics.txt",
        "text": "Precision@K evaluates retrieval quality by computing the ratio of relevant items retrieved within the top K results divided by K.",
        "metadata": {"document_id": "doc_eval", "filename": "evaluation_metrics.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c42",
        "document_id": "doc_eval",
        "filename": "evaluation_metrics.txt",
        "text": "Recall@K evaluates retrieval coverage by computing the proportion of all known relevant ground-truth documents that appear in the top K results.",
        "metadata": {"document_id": "doc_eval", "filename": "evaluation_metrics.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c43",
        "document_id": "doc_eval",
        "filename": "evaluation_metrics.txt",
        "text": "Mean Reciprocal Rank computes the average of reciprocal ranks 1 / rank_i for the first relevant document retrieved across all evaluation test queries.",
        "metadata": {"document_id": "doc_eval", "filename": "evaluation_metrics.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c44",
        "document_id": "doc_eval",
        "filename": "evaluation_metrics.txt",
        "text": "Hit Rate@K is a binary metric indicating whether at least one ground-truth relevant document was successfully retrieved in the top K candidates.",
        "metadata": {"document_id": "doc_eval", "filename": "evaluation_metrics.txt", "page_number": 2, "page_references": [2]},
    },
    {
        "chunk_id": "c45",
        "document_id": "doc_eval",
        "filename": "evaluation_metrics.txt",
        "text": "Ground truth relevance judgments define authoritative binary or graded mappings between specific user queries and truly relevant corpus documents.",
        "metadata": {"document_id": "doc_eval", "filename": "evaluation_metrics.txt", "page_number": 3, "page_references": [3]},
    },
    {
        "chunk_id": "c46",
        "document_id": "doc_distractor_eval",
        "filename": "classification_eval.txt",
        "text": "Area Under the Receiver Operating Characteristic curve AUC-ROC plots the true positive rate against false positive rate across classification thresholds.",
        "metadata": {"document_id": "doc_distractor_eval", "filename": "classification_eval.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c47",
        "document_id": "doc_distractor_eval",
        "filename": "ndcg_ranking.txt",
        "text": "Normalized Discounted Cumulative Gain NDCG discounts relevant documents appearing lower in ranked lists using logarithmic position penalties.",
        "metadata": {"document_id": "doc_distractor_eval", "filename": "ndcg_ranking.txt", "page_number": 1, "page_references": [1]},
    },
    {
        "chunk_id": "c48",
        "document_id": "doc_eval",
        "filename": "evaluation_metrics.txt",
        "text": "The lost in the middle phenomenon observes that language models attend effectively to prompt beginnings and endings but frequently ignore facts placed in the center.",
        "metadata": {"document_id": "doc_eval", "filename": "evaluation_metrics.txt", "page_number": 3, "page_references": [3]},
    },
]


# ============================================================================
# Labeled Evaluation Queries (20 Diverse Benchmark Queries)
# ============================================================================

EVALUATION_QUERIES: List[EvaluationQuery] = [
    EvaluationQuery(
        query_id="q1",
        query_text="How does Okapi BM25 prevent term frequency saturation?",
        expected_chunk_ids=["c9"],
        intent_category="exact_keyword",
    ),
    EvaluationQuery(
        query_id="q2",
        query_text="What vector database uses persistent disk storage with SQLite and HNSW?",
        expected_chunk_ids=["c1"],
        intent_category="exact_keyword",
    ),
    EvaluationQuery(
        query_id="q3",
        query_text="How is cosine distance converted to cosine similarity in ChromaDB?",
        expected_chunk_ids=["c3"],
        intent_category="technical_fact",
    ),
    EvaluationQuery(
        query_id="q4",
        query_text="Explain the difference between bi-encoders and cross-encoders for retrieval.",
        expected_chunk_ids=["c17", "c18"],  # Multiple relevant chunks!
        intent_category="conceptual",
    ),
    EvaluationQuery(
        query_id="q5",
        query_text="What algorithm combines rankings from different search systems without normalizing scores?",
        expected_chunk_ids=["c16"],
        intent_category="semantic_paraphrase",
    ),
    EvaluationQuery(
        query_id="q6",
        query_text="How does Python manage memory and clean up cyclic references?",
        expected_chunk_ids=["c33"],
        intent_category="technical_fact",
    ),
    EvaluationQuery(
        query_id="q7",
        query_text="Why does Transformer self-attention scale quadratically with sequence length?",
        expected_chunk_ids=["c20"],
        intent_category="conceptual",
    ),
    EvaluationQuery(
        query_id="q8",
        query_text="What is the token limit of all-MiniLM-L6-v2 and how are oversized chunks prevented?",
        expected_chunk_ids=["c23", "c32"],  # Multiple relevant chunks!
        intent_category="technical_fact",
    ),
    EvaluationQuery(
        query_id="q9",
        query_text="How does BM25 adjust scoring for documents of different lengths?",
        expected_chunk_ids=["c10"],
        intent_category="exact_keyword",
    ),
    EvaluationQuery(
        query_id="q10",
        query_text="How do we verify that chunk text was not lost during document splitting?",
        expected_chunk_ids=["c27", "c31"],  # Multiple relevant chunks!
        intent_category="semantic_paraphrase",
    ),
    EvaluationQuery(
        query_id="q11",
        query_text="What graph algorithm achieves O(log N) approximate nearest neighbor vector search?",
        expected_chunk_ids=["c2"],
        intent_category="technical_fact",
    ),
    EvaluationQuery(
        query_id="q12",
        query_text="How are duplicate uploaded files detected using cryptographic hashes?",
        expected_chunk_ids=["c28"],
        intent_category="exact_keyword",
    ),
    EvaluationQuery(
        query_id="q13",
        query_text="What metric evaluates the rank of the first relevant document retrieved?",
        expected_chunk_ids=["c43"],
        intent_category="exact_keyword",
    ),
    EvaluationQuery(
        query_id="q14",
        query_text="Explain the trade-off between Precision@K and Recall@K in search.",
        expected_chunk_ids=["c41", "c42"],  # Multiple relevant chunks!
        intent_category="conceptual",
    ),
    EvaluationQuery(
        query_id="q15",
        query_text="What library extracts text glyphs and coordinates from PDF pages directly?",
        expected_chunk_ids=["c25"],
        intent_category="technical_fact",
    ),
    EvaluationQuery(
        query_id="q16",
        query_text="How does Unicode NFKC normalization clean raw text?",
        expected_chunk_ids=["c26"],
        intent_category="exact_keyword",
    ),
    EvaluationQuery(
        query_id="q17",
        query_text="What hardware instructions accelerate floating point vector multiplications on modern processors?",
        expected_chunk_ids=["c7"],
        intent_category="semantic_paraphrase",
    ),
    EvaluationQuery(
        query_id="q18",
        query_text="How does FastAPI manage server lifecycles asynchronously?",
        expected_chunk_ids=["c35"],
        intent_category="technical_fact",
    ),
    EvaluationQuery(
        query_id="q19",
        query_text="Why do long prompt contexts degrade language model reasoning?",
        expected_chunk_ids=["c48"],
        intent_category="conceptual",
    ),
    EvaluationQuery(
        query_id="q20",
        query_text="What model performs passage reranking using joint cross-attention on MS MARCO?",
        expected_chunk_ids=["c17", "c19"],  # Multiple relevant chunks!
        intent_category="technical_fact",
    ),
]


# ============================================================================
# Metric Calculators
# ============================================================================

def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Precision@K = |Retrieved_K cap Relevant| / K.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Recall@K = |Retrieved_K cap Relevant| / |Relevant|.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def calculate_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Compute Reciprocal Rank = 1 / first_relevant_rank (or 0.0 if not found).

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.

    Returns:
        Reciprocal rank between 0.0 and 1.0.
    """
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Hit Rate@K = 1.0 if any relevant chunk is in top K, else 0.0.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        1.0 if at least one hit in top k, else 0.0.
    """
    top_k = retrieved_ids[:k]
    for cid in top_k:
        if cid in relevant_ids:
            return 1.0
    return 0.0


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_retrieval_benchmark(
    retriever: Retriever,
    queries: Optional[List[EvaluationQuery]] = None,
    eval_ks: Optional[List[int]] = None,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute evaluation benchmark comparing Dense, BM25, and Hybrid retrieval modes.

    Calculates Precision@K, Recall@K, Hit Rate@K (for k in eval_ks, default [3, 5]), and MRR.

    Args:
        retriever: Configured Retriever instance.
        queries: Benchmark queries list (defaults to EVALUATION_QUERIES).
        eval_ks: List of K thresholds to evaluate (defaults to [3, 5]).
        k: Optional single K parameter for backward-compatibility.

    Returns:
        Structured evaluation metrics across all strategies.
    """
    eval_queries = queries or EVALUATION_QUERIES
    if k is not None and eval_ks is None:
        ks = [k]
    else:
        ks = eval_ks or [3, 5]
    max_k = max(ks)
    modes = [RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID]

    results: Dict[str, Any] = {
        "dataset_size": len(eval_queries),
        "corpus_size": len(EVALUATION_DOCUMENTS),
        "evaluated_ks": ks,
        "modes": {},
    }

    for mode in modes:
        mode_metrics: Dict[str, Any] = {}
        reciprocal_ranks: List[float] = []
        precision_tracker: Dict[int, List[float]] = {k: [] for k in ks}
        recall_tracker: Dict[int, List[float]] = {k: [] for k in ks}
        hit_rate_tracker: Dict[int, List[float]] = {k: [] for k in ks}

        for eq in eval_queries:
            relevant_set = set(eq.expected_chunk_ids)

            request = RetrievalRequest(
                query=eq.query_text,
                config=RetrievalConfiguration(
                    mode=mode,
                    top_k=max_k,
                    candidate_k=max_k * 3,
                    enable_reranking=False,
                ),
            )
            res = retriever.retrieve(request, owner_id=EVAL_OWNER_ID)
            retrieved_ids = [c.chunk_id for c in res.chunks]

            reciprocal_ranks.append(calculate_reciprocal_rank(retrieved_ids, relevant_set))

            for k in ks:
                precision_tracker[k].append(calculate_precision_at_k(retrieved_ids, relevant_set, k))
                recall_tracker[k].append(calculate_recall_at_k(retrieved_ids, relevant_set, k))
                hit_rate_tracker[k].append(calculate_hit_rate_at_k(retrieved_ids, relevant_set, k))

        num_q = len(eval_queries)
        for k in ks:
            mode_metrics[f"precision@{k}"] = round(sum(precision_tracker[k]) / num_q, 4)
            mode_metrics[f"recall@{k}"] = round(sum(recall_tracker[k]) / num_q, 4)
            mode_metrics[f"hit_rate@{k}"] = round(sum(hit_rate_tracker[k]) / num_q, 4)

        mode_metrics["mrr"] = round(sum(reciprocal_ranks) / num_q, 4)
        results["modes"][mode.value] = mode_metrics

    return results
