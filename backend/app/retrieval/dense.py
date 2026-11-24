"""Dense semantic retrieval engine for Retrieva AI.

Converts natural language queries into 384-dimensional dense vectors using the
shared SentenceTransformerEmbedder and queries ChromaDB with cosine distance.
"""

from typing import Any, Dict, List, Optional

try:
    from backend.app.core.logging import get_logger
    from backend.app.embeddings.embedder import SentenceTransformerEmbedder
    from backend.app.schemas.retrieval import RetrievedChunk
    from backend.app.vector_store.store import ChromaVectorStore
except ImportError:
    from app.core.logging import get_logger
    from app.embeddings.embedder import SentenceTransformerEmbedder
    from app.schemas.retrieval import RetrievedChunk
    from app.vector_store.store import ChromaVectorStore

logger = get_logger("retrieval.dense")


class DenseRetriever:
    """Performs dense vector semantic search against the persistent ChromaDB collection.

    Reuses the existing SentenceTransformerEmbedder singleton to avoid duplicate model loads.
    """

    def __init__(
        self,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ) -> None:
        """Initialize dense retriever with embedder and vector store.

        Args:
            embedder: Pre-initialized SentenceTransformerEmbedder (singleton used if None).
            vector_store: Pre-initialized ChromaVectorStore (initialized if None).
        """
        self.embedder = embedder or SentenceTransformerEmbedder.get_instance()
        self.vector_store = vector_store or ChromaVectorStore()

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        document_id_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Execute semantic nearest-neighbor retrieval for a query.

        Args:
            query: Non-empty natural language query string.
            top_k: Maximum number of chunks to return (top_k >= 1).
            similarity_threshold: Optional minimum cosine similarity filter [-1.0, 1.0].
            document_id_filter: Optional document ID to restrict search scope.

        Returns:
            Ranked list of RetrievedChunk objects ordered by ascending distance / descending similarity.

        Raises:
            ValueError: If query is blank or top_k < 1.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")
        if top_k < 1:
            raise ValueError(f"top_k must be greater than or equal to 1, got {top_k}")

        total_chunks = self.vector_store.count()
        if total_chunks == 0:
            logger.info("ChromaDB collection is empty, returning zero dense candidates")
            return []

        # 1. Query Embedding Generation (384-dim, L2-normalized)
        query_vector = self.embedder.embed_text(query)

        # 2. Build ChromaDB query filters
        where_clause: Optional[Dict[str, Any]] = None
        if document_id_filter:
            where_clause = {"document_id": document_id_filter}

        # ChromaDB n_results cannot exceed total documents in collection
        effective_n_results = min(top_k, total_chunks)

        logger.info(
            "Executing ChromaDB dense vector query (n_results=%d, filter=%s)...",
            effective_n_results,
            document_id_filter,
        )

        # 3. Query ChromaDB collection
        results = self.vector_store.collection.query(
            query_embeddings=[query_vector],
            n_results=effective_n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not ids:
            return []

        retrieved_chunks: List[RetrievedChunk] = []

        for idx, chunk_id in enumerate(ids):
            chunk_text = docs[idx] if idx < len(docs) else ""
            raw_meta = metas[idx] if idx < len(metas) else {}
            deserialized_meta = self.vector_store.deserialize_metadata(raw_meta)

            # ChromaDB with hnsw:space='cosine' returns cosine distance:
            # cosine_distance = 1.0 - cosine_similarity
            cosine_dist = float(distances[idx]) if idx < len(distances) else 0.0
            cosine_sim = 1.0 - cosine_dist

            # 4. Relevance Filtering
            if similarity_threshold is not None:
                if cosine_sim < similarity_threshold:
                    logger.debug(
                        "Skipping chunk %s with similarity %.4f below threshold %.4f",
                        chunk_id,
                        cosine_sim,
                        similarity_threshold,
                    )
                    continue

            page_refs = deserialized_meta.get("page_references", [deserialized_meta.get("page_number", 1)])
            if not isinstance(page_refs, list):
                page_refs = [int(page_refs)]

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=deserialized_meta.get("document_id", ""),
                    filename=deserialized_meta.get("filename", ""),
                    page_number=deserialized_meta.get("page_number", 1),
                    page_references=page_refs,
                    chunk_text=chunk_text,
                    start_offset=deserialized_meta.get("start_offset", 0),
                    end_offset=deserialized_meta.get("end_offset", len(chunk_text)),
                    rank=len(retrieved_chunks) + 1,
                    cosine_distance=round(cosine_dist, 6),
                    cosine_similarity=round(cosine_sim, 6),
                    metadata=deserialized_meta,
                )
            )

        return retrieved_chunks
