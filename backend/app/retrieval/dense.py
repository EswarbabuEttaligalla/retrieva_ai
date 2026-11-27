"""Dense semantic retrieval engine for Retrieva AI.

Converts natural language queries into 384-dimensional dense vectors using the
shared SentenceTransformerEmbedder and executes mandatory owner-scoped ChromaDB queries.
"""

from typing import List, Optional, Union
from uuid import UUID

try:
    from backend.app.core.logging import get_logger
    from backend.app.embeddings.embedder import SentenceTransformerEmbedder
    from backend.app.schemas.retrieval import RetrievedChunk
    from backend.app.vector_store.store import (
        ChromaVectorStore,
        TenantIsolationError,
        normalize_owner_id,
    )
except ImportError:
    from app.core.logging import get_logger
    from app.embeddings.embedder import SentenceTransformerEmbedder
    from app.schemas.retrieval import RetrievedChunk
    from app.vector_store.store import (
        ChromaVectorStore,
        TenantIsolationError,
        normalize_owner_id,
    )

logger = get_logger("retrieval.dense")


class DenseRetriever:
    """Performs mandatory owner-scoped dense vector semantic search against ChromaDB."""

    def __init__(
        self,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder.get_instance()
        self.vector_store = vector_store or ChromaVectorStore()

    def search(
        self,
        query: str,
        owner_id: Union[UUID, str],
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        document_id_filter: Optional[str] = None,
        document_ids_filter: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Execute semantic nearest-neighbor retrieval strictly scoped to owner_id."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")
        if top_k < 1:
            raise ValueError(f"top_k must be greater than or equal to 1, got {top_k}")

        owner_str = normalize_owner_id(owner_id)

        query_vector = self.embedder.embed_text(query)

        results = self.vector_store.query_similar(
            query_embedding=query_vector,
            owner_id=owner_str,
            top_k=top_k,
            document_id_filter=document_id_filter,
            document_ids_filter=document_ids_filter,
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

            if deserialized_meta.get("owner_id") != owner_str:
                logger.error(
                    "SECURITY ALERT: DenseRetriever rejected cross-tenant chunk (expected_owner=%s, doc=%s)",
                    owner_str,
                    deserialized_meta.get("document_id"),
                )
                raise TenantIsolationError("Cross-tenant chunk rejected after ChromaDB query.")

            cosine_dist = float(distances[idx]) if idx < len(distances) else 0.0
            cosine_sim = 1.0 - cosine_dist

            if similarity_threshold is not None and cosine_sim < similarity_threshold:
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
