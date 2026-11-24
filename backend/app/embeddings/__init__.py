"""Embeddings package for Retrieva AI."""

from .embedder import (
    SentenceTransformerEmbedder,
    EmbeddingEngineError,
    EmbeddingModelLoadError,
    EmbeddingGenerationError,
    EmbeddingTokenLimitError,
)
from .validator import (
    validate_embedding,
    validate_batch_embeddings,
    validate_embedding_dimension,
    validate_finite_vector,
    calculate_l2_norm,
    is_l2_normalized,
    EmbeddingValidationError,
    EmbeddingDimensionError,
    EmbeddingValueError,
    EmbeddingNormalizationError,
)

__all__ = [
    "SentenceTransformerEmbedder",
    "EmbeddingEngineError",
    "EmbeddingModelLoadError",
    "EmbeddingGenerationError",
    "EmbeddingTokenLimitError",
    "validate_embedding",
    "validate_batch_embeddings",
    "validate_embedding_dimension",
    "validate_finite_vector",
    "calculate_l2_norm",
    "is_l2_normalized",
    "EmbeddingValidationError",
    "EmbeddingDimensionError",
    "EmbeddingValueError",
    "EmbeddingNormalizationError",
]
