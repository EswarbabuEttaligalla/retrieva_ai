"""Embedding validation utilities for Retrieva AI.

Ensures that generated dense vector embeddings satisfy dimensionality,
finiteness, and normalization constraints before insertion into vector storage.
"""

import math
from typing import List, Sequence, Union


class EmbeddingValidationError(Exception):
    """Base exception for embedding validation errors."""
    pass


class EmbeddingDimensionError(EmbeddingValidationError):
    """Raised when an embedding vector has an unexpected dimension."""
    pass


class EmbeddingValueError(EmbeddingValidationError):
    """Raised when an embedding vector contains non-finite values (NaN, Inf)."""
    pass


class EmbeddingNormalizationError(EmbeddingValidationError):
    """Raised when an embedding vector fails L2 normalization checks."""
    pass


def validate_embedding_dimension(
    vector: Sequence[Union[float, int]],
    expected_dim: int = 384,
) -> None:
    """Validate that the vector length matches the expected dimension.

    Args:
        vector: Numerical sequence representing the embedding.
        expected_dim: Expected dimensionality (default: 384 for all-MiniLM-L6-v2).

    Raises:
        EmbeddingDimensionError: If the vector length does not match expected_dim.
    """
    actual_dim = len(vector)
    if actual_dim != expected_dim:
        raise EmbeddingDimensionError(
            f"Embedding dimensionality mismatch: expected {expected_dim}, got {actual_dim}."
        )


def validate_finite_vector(vector: Sequence[Union[float, int]]) -> None:
    """Validate that all values in the vector are finite (no NaN, Inf, or -Inf).

    Args:
        vector: Numerical sequence representing the embedding.

    Raises:
        EmbeddingValueError: If any component is NaN, positive infinity, or negative infinity.
    """
    for idx, val in enumerate(vector):
        if math.isnan(val):
            raise EmbeddingValueError(f"Embedding contains NaN at index {idx}.")
        if math.isinf(val):
            raise EmbeddingValueError(f"Embedding contains infinite value at index {idx}.")


def calculate_l2_norm(vector: Sequence[Union[float, int]]) -> float:
    """Compute the Euclidean (L2) norm of a vector.

    Formula: ||v||_2 = sqrt(sum(v_i^2))

    Args:
        vector: Numerical sequence representing the embedding.

    Returns:
        The Euclidean norm as a float.
    """
    return math.sqrt(sum(float(x) ** 2 for x in vector))


def is_l2_normalized(
    vector: Sequence[Union[float, int]],
    tolerance: float = 1e-4,
) -> bool:
    """Check if the vector is L2-normalized within a specified tolerance.

    Args:
        vector: Numerical sequence representing the embedding.
        tolerance: Allowed deviation from 1.0 (default: 1e-4).

    Returns:
        True if |norm - 1.0| <= tolerance, False otherwise.
    """
    norm = calculate_l2_norm(vector)
    return abs(norm - 1.0) <= tolerance


def validate_embedding(
    vector: Sequence[Union[float, int]],
    expected_dim: int = 384,
    check_normalization: bool = True,
    norm_tolerance: float = 1e-4,
) -> None:
    """Comprehensive validation of a single embedding vector.

    Checks:
    1. Dimension matches expected_dim.
    2. All components are finite numbers.
    3. Vector is L2-normalized (if check_normalization=True).

    Args:
        vector: Numerical sequence representing the embedding.
        expected_dim: Expected dimension (384 for all-MiniLM-L6-v2).
        check_normalization: Whether to verify L2 normalization.
        norm_tolerance: Tolerance for L2 normalization check.

    Raises:
        EmbeddingDimensionError: If vector dimension is invalid.
        EmbeddingValueError: If vector has NaN or Inf.
        EmbeddingNormalizationError: If vector is not L2 normalized.
    """
    validate_embedding_dimension(vector, expected_dim=expected_dim)
    validate_finite_vector(vector)

    if check_normalization:
        norm = calculate_l2_norm(vector)
        if abs(norm - 1.0) > norm_tolerance:
            raise EmbeddingNormalizationError(
                f"Embedding vector is not L2 normalized: norm is {norm:.6f}, expected 1.0 ± {norm_tolerance}."
            )


def validate_batch_embeddings(
    vectors: List[List[float]],
    expected_dim: int = 384,
    check_normalization: bool = True,
    norm_tolerance: float = 1e-4,
) -> None:
    """Validate a batch of embedding vectors.

    Args:
        vectors: List of embedding vectors.
        expected_dim: Expected dimension for each vector.
        check_normalization: Whether to check L2 normalization.
        norm_tolerance: Tolerance for L2 normalization check.

    Raises:
        EmbeddingValidationError: If any vector fails validation.
    """
    if not vectors:
        raise EmbeddingValueError("Batch embedding list cannot be empty.")

    for idx, vec in enumerate(vectors):
        try:
            validate_embedding(
                vec,
                expected_dim=expected_dim,
                check_normalization=check_normalization,
                norm_tolerance=norm_tolerance,
            )
        except EmbeddingValidationError as e:
            raise EmbeddingValidationError(
                f"Batch validation failed at vector index {idx}: {str(e)}"
            ) from e
