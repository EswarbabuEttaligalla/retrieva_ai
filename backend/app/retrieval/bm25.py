"""Custom implementation of the Okapi BM25 keyword retrieval algorithm.

Implements term frequency (TF), Robertson-Spärck Jones inverse document frequency (IDF),
and document length normalization from first principles without external RAG frameworks.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from backend.app.core.logging import get_logger
    from backend.app.schemas.retrieval import RetrievedChunk
except ImportError:
    from app.core.logging import get_logger
    from app.schemas.retrieval import RetrievedChunk

logger = get_logger("retrieval.bm25")

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words.

    Args:
        text: Input string.

    Returns:
        List of normalized lowercase token strings.
    """
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Okapi BM25 inverted index with document length normalization.

    Formula:
        Score(D, Q) = sum_{t in Q} IDF(t) * (TF(t, D) * (k1 + 1)) / (TF(t, D) + k1 * (1 - b + b * (|D| / avgdl)))

    Where:
        - TF(t, D): Term frequency of t in document D.
        - |D|: Length of document D in tokens.
        - avgdl: Average document length across the indexed corpus.
        - k1: Controls term frequency saturation (default: 1.5).
        - b: Controls document length penalization (default: 0.75).
        - IDF(t): ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 index with tuning parameters.

        Args:
            k1: Term frequency saturation constant (k1 >= 0).
            b: Length normalization parameter (0 <= b <= 1).
        """
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be between 0.0 and 1.0, got {b}")

        self.k1 = k1
        self.b = b

        # Corpus state
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.total_tokens: int = 0

        # Storage structures
        # chunk_id -> token count
        self.doc_lengths: Dict[str, int] = {}
        # term -> document frequency (count of chunks containing term)
        self.doc_frequencies: Dict[str, int] = {}
        # term -> {chunk_id: term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        # chunk_id -> raw chunk metadata dict
        self.corpus_chunks: Dict[str, Dict[str, Any]] = {}

    def clear(self) -> None:
        """Reset and wipe the BM25 index."""
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.total_tokens = 0
        self.doc_lengths.clear()
        self.doc_frequencies.clear()
        self.inverted_index.clear()
        self.corpus_chunks.clear()

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build or replace the entire index from a list of chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries containing 'chunk_id', 'text', and metadata.
        """
        self.clear()
        for chunk in chunks:
            self.add_chunk(chunk)

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """Index a single document chunk.

        Args:
            chunk: Dictionary containing 'chunk_id', 'text' (or 'chunk_text'), and metadata.
        """
        chunk_id = chunk["chunk_id"]
        # Support both 'text' and 'chunk_text'
        text = chunk.get("text") or chunk.get("chunk_text") or ""

        # Remove previous version if already indexed
        if chunk_id in self.corpus_chunks:
            self.remove_chunk(chunk_id)

        tokens = tokenize_text(text)
        doc_len = len(tokens)

        # Store metadata
        self.corpus_chunks[chunk_id] = chunk
        self.doc_lengths[chunk_id] = doc_len
        self.doc_count += 1
        self.total_tokens += doc_len
        self.avg_doc_len = self.total_tokens / self.doc_count if self.doc_count > 0 else 0.0

        # Term counts in this chunk
        term_counts: Dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        # Update inverted index and document frequencies
        for term, freq in term_counts.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
                self.doc_frequencies[term] = 0

            self.inverted_index[term][chunk_id] = freq
            self.doc_frequencies[term] += 1

    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a single chunk from the index.

        Args:
            chunk_id: ID of the chunk to remove.

        Returns:
            True if chunk was found and removed, False otherwise.
        """
        if chunk_id not in self.corpus_chunks:
            return False

        doc_len = self.doc_lengths.pop(chunk_id, 0)
        del self.corpus_chunks[chunk_id]

        self.doc_count -= 1
        self.total_tokens -= doc_len
        self.avg_doc_len = self.total_tokens / self.doc_count if self.doc_count > 0 else 0.0

        # Clean inverted index and frequencies
        empty_terms: List[str] = []
        for term, postings in self.inverted_index.items():
            if chunk_id in postings:
                del postings[chunk_id]
                self.doc_frequencies[term] -= 1
                if self.doc_frequencies[term] <= 0:
                    empty_terms.append(term)

        for term in empty_terms:
            del self.inverted_index[term]
            del self.doc_frequencies[term]

        return True

    def remove_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document_id.

        Args:
            document_id: Identifier of the document to remove.

        Returns:
            Count of removed chunks.
        """
        target_chunk_ids = [
            cid for cid, c in self.corpus_chunks.items()
            if c.get("document_id") == document_id or c.get("metadata", {}).get("document_id") == document_id
        ]
        for cid in target_chunk_ids:
            self.remove_chunk(cid)
        return len(target_chunk_ids)

    def calculate_idf(self, term: str) -> float:
        """Calculate Robertson-Spärck Jones Inverse Document Frequency (IDF) with +1 floor.

        Formula:
            IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

        The +1 floor ensures IDF is strictly positive even for terms appearing in all documents.

        Args:
            term: Normalized token string.

        Returns:
            IDF weight as a float.
        """
        n_t = self.doc_frequencies.get(term, 0)
        if n_t == 0 or self.doc_count == 0:
            return 0.0

        # Positive Robertson-Spärck Jones formula
        numerator = self.doc_count - n_t + 0.5
        denominator = n_t + 0.5
        return math.log(1.0 + (numerator / denominator))

    def score_chunk(self, query_tokens: List[str], chunk_id: str) -> float:
        """Compute the BM25 relevance score of a chunk for a given tokenized query.

        Args:
            query_tokens: List of normalized query terms.
            chunk_id: Identifier of the candidate chunk.

        Returns:
            Non-negative BM25 score.
        """
        if chunk_id not in self.corpus_chunks or self.doc_count == 0:
            return 0.0

        doc_len = self.doc_lengths.get(chunk_id, 0)
        avg_len = self.avg_doc_len if self.avg_doc_len > 0 else 1.0

        # Length normalization factor
        len_norm = 1.0 - self.b + self.b * (doc_len / avg_len)

        score = 0.0
        for term in query_tokens:
            postings = self.inverted_index.get(term)
            if not postings or chunk_id not in postings:
                continue

            tf = postings[chunk_id]
            idf = self.calculate_idf(term)

            # Okapi BM25 TF saturation
            term_score = idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm))
            score += term_score

        return score

    def get_statistics(self) -> Dict[str, Any]:
        """Return corpus statistics strictly for this BM25Index instance."""
        unique_docs = {
            c.get("document_id") or c.get("metadata", {}).get("document_id")
            for c in self.corpus_chunks.values()
        }
        unique_docs.discard(None)
        unique_docs.discard("")
        return {
            "chunk_count": self.doc_count,
            "unique_document_count": len(unique_docs),
            "total_tokens": self.total_tokens,
            "avg_doc_len": round(self.avg_doc_len, 4),
            "vocabulary_size": len(self.doc_frequencies),
            "k1": self.k1,
            "b": self.b,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id_filter: Optional[str] = None,
        document_ids_filter: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Execute BM25 keyword search over indexed chunks.

        Args:
            query: Natural language query string.
            top_k: Maximum number of ranked results to return.
            document_id_filter: Optional single document ID to filter candidates.
            document_ids_filter: Optional list of document IDs to filter candidates.

        Returns:
            Ranked list of RetrievedChunk objects sorted descending by bm25_score.
        """
        if not query or not query.strip() or self.doc_count == 0 or top_k <= 0:
            return []

        allowed_doc_ids: Optional[Set[str]] = None
        if document_ids_filter is not None:
            cleaned_ids = {str(d).strip() for d in document_ids_filter if str(d).strip()}
            if cleaned_ids:
                allowed_doc_ids = cleaned_ids
        if document_id_filter:
            clean_single = str(document_id_filter).strip()
            if clean_single:
                if allowed_doc_ids is not None:
                    allowed_doc_ids = allowed_doc_ids & {clean_single}
                    if not allowed_doc_ids:
                        return []
                else:
                    allowed_doc_ids = {clean_single}

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        # Find all chunks containing at least one query term
        candidate_chunk_ids: Set[str] = set()
        for term in query_tokens:
            postings = self.inverted_index.get(term, {})
            candidate_chunk_ids.update(postings.keys())

        if not candidate_chunk_ids:
            return []

        # Filter and score candidates
        scored_candidates: List[Tuple[float, str]] = []
        for chunk_id in candidate_chunk_ids:
            chunk_meta = self.corpus_chunks[chunk_id]
            doc_id = chunk_meta.get("document_id") or chunk_meta.get("metadata", {}).get("document_id")

            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue

            score = self.score_chunk(query_tokens, chunk_id)
            if score > 0.0:
                scored_candidates.append((score, chunk_id))

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        # Construct typed RetrievedChunk models
        results: List[RetrievedChunk] = []
        for rank_idx, (score, chunk_id) in enumerate(top_candidates, start=1):
            raw_chunk = self.corpus_chunks[chunk_id]
            meta = dict(raw_chunk.get("metadata", {}))

            # Resolve page references
            page_refs = meta.get("page_references", [meta.get("page_number", 1)])
            if isinstance(page_refs, str):
                try:
                    page_refs = [int(p.strip()) for p in page_refs.split(",") if p.strip()]
                except ValueError:
                    page_refs = [meta.get("page_number", 1)]

            chunk_text = raw_chunk.get("text") or raw_chunk.get("chunk_text") or ""

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=raw_chunk.get("document_id") or meta.get("document_id", ""),
                    filename=raw_chunk.get("filename") or meta.get("filename", ""),
                    page_number=meta.get("page_number", 1),
                    page_references=page_refs,
                    chunk_text=chunk_text,
                    start_offset=meta.get("start_offset", 0),
                    end_offset=meta.get("end_offset", len(chunk_text)),
                    rank=rank_idx,
                    bm25_score=round(score, 6),
                    metadata=meta,
                )
            )

        return results
