"""Sentence Transformers embedding engine for Retrieva AI.

Generates dense numerical vector representations for document chunks using
sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) with L2 normalization.
"""

from typing import List, Optional

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.embeddings.validator import (
    validate_embedding,
    validate_batch_embeddings,
    EmbeddingValidationError,
)

logger = get_logger("embeddings.embedder")


class EmbeddingEngineError(Exception):
    """Base exception for embedding engine errors."""
    pass


class EmbeddingModelLoadError(EmbeddingEngineError):
    """Raised when the embedding model fails to initialize."""
    pass


class EmbeddingGenerationError(EmbeddingEngineError):
    """Raised when embedding generation fails."""
    pass


class EmbeddingTokenLimitError(EmbeddingEngineError):
    """Raised when input text exceeds the embedding model's maximum sequence length."""
    def __init__(self, message: str, token_count: Optional[int] = None):
        super().__init__(message)
        self.token_count = token_count


class SentenceTransformerEmbedder:
    """Production-grade embedding engine utilizing Sentence Transformers.

    Uses `sentence-transformers/all-MiniLM-L6-v2` producing 384-dimensional
    dense vectors with unit L2 norm.
    """

    _instance: Optional["SentenceTransformerEmbedder"] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier. Defaults to config setting.
            device: Compute device ('cpu', 'cuda', etc.). If None, auto-detected.

        Raises:
            EmbeddingModelLoadError: If the model cannot be loaded.
        """
        settings = get_settings()
        self.model_name: str = model_name or settings.EMBEDDING_MODEL_NAME
        self.expected_dim: int = settings.EMBEDDING_DIMENSION
        self.device: Optional[str] = device
        self._model = None
        self._max_seq_length: int = 256

        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model into memory."""
        try:
            logger.info("Loading embedding model '%s' (device: %s)...", self.model_name, self.device)
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device

            self._model = SentenceTransformer(self.model_name, **kwargs)
            self._max_seq_length = getattr(self._model, "max_seq_length", 256)
            logger.info(
                "Embedding model loaded: %s (dim=%d, max_seq=%d)",
                self.model_name,
                self.expected_dim,
                self._max_seq_length,
            )
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", self.model_name, exc)
            raise EmbeddingModelLoadError(
                f"Failed to load embedding model '{self.model_name}': {str(exc)}"
            ) from exc

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "SentenceTransformerEmbedder":
        """Get or create singleton instance of the embedder."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    @property
    def max_sequence_length(self) -> int:
        """Return the maximum supported token sequence length."""
        return self._max_seq_length

    def get_embedding_dimension(self) -> int:
        """Return the expected embedding dimension (384)."""
        return self.expected_dim

    def get_max_sequence_length(self) -> int:
        """Inspect the maximum token length supported before truncation."""
        return self._max_seq_length

    def get_tokenizer(self):
        """Return the HuggingFace tokenizer instance."""
        if self._model is not None:
            return getattr(self._model, "tokenizer", None)
        return None

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string using the model tokenizer.

        Args:
            text: Input string.

        Returns:
            Number of tokens.
        """
        if not text or not self._model:
            return 0
        try:
            tokenizer = self._model.tokenizer
            return len(tokenizer.encode(text, add_special_tokens=True))
        except Exception:
            return len(text.split())

    def embed_text(self, text: str, auto_truncate: bool = False) -> List[float]:
        """Convert a single text string into a normalized 384-dimensional dense vector.

        Args:
            text: Non-empty text string.
            auto_truncate: If False (default), raises EmbeddingTokenLimitError when text
                exceeds max sequence length to prevent silent information loss.
                If True, logs warning and allows tokenizer to truncate.

        Returns:
            List of 384 Python floats representing the L2-normalized embedding.

        Raises:
            ValueError: If input is empty or whitespace-only.
            EmbeddingTokenLimitError: If text exceeds max sequence length and auto_truncate is False.
            EmbeddingGenerationError: If model inference fails.
            EmbeddingValidationError: If the generated embedding fails validation checks.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text for embedding must be a non-empty string.")

        token_count = self.count_tokens(text)
        if token_count > self._max_seq_length:
            if not auto_truncate:
                raise EmbeddingTokenLimitError(
                    f"Input text has {token_count} tokens, which exceeds the model limit of {self._max_seq_length}. "
                    f"Token-aware subchunking required to prevent silent content loss.",
                    token_count=token_count,
                )
            logger.warning(
                "Text (%d chars, ~%d tokens) exceeds max sequence length (%d) and will be truncated",
                len(text),
                token_count,
                self._max_seq_length,
            )

        try:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            vector: List[float] = [float(x) for x in embedding.tolist()]
            validate_embedding(vector, expected_dim=self.expected_dim, check_normalization=True)
            return vector

        except (EmbeddingValidationError, EmbeddingTokenLimitError):
            raise
        except Exception as exc:
            logger.error("Failed to generate embedding: %s", exc)
            raise EmbeddingGenerationError(f"Embedding generation failed: {str(exc)}") from exc

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        auto_truncate: bool = False,
    ) -> List[List[float]]:
        """Generate normalized embeddings for a batch of text chunks.

        Args:
            texts: List of non-empty text strings.
            batch_size: Chunk batch size for inference (default from settings: 32).
            show_progress: Whether to display a progress bar.
            auto_truncate: If False (default), raises EmbeddingTokenLimitError if any
                item in the batch exceeds max_seq_length.

        Returns:
            List of 384-dimensional float vectors.

        Raises:
            ValueError: If texts list is empty or contains blank strings.
            EmbeddingTokenLimitError: If any item exceeds max sequence length without auto_truncate.
            EmbeddingGenerationError: If model inference fails.
            EmbeddingValidationError: If any output vector fails validation.
        """
        if not texts:
            raise ValueError("Texts list for batch embedding cannot be empty.")

        settings = get_settings()
        effective_batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        for idx, item in enumerate(texts):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Batch item at index {idx} is empty or not a string.")
            if not auto_truncate:
                tc = self.count_tokens(item)
                if tc > self._max_seq_length:
                    raise EmbeddingTokenLimitError(
                        f"Batch item at index {idx} has {tc} tokens, exceeding model maximum length of {self._max_seq_length}. "
                        f"Token-aware subchunking required to prevent silent content loss.",
                        token_count=tc,
                    )

        logger.info(
            "Generating embeddings for batch of %d items (batch_size=%d)...",
            len(texts),
            effective_batch_size,
        )

        try:
            raw_embeddings = self._model.encode(
                texts,
                batch_size=effective_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )

            vectors: List[List[float]] = [
                [float(val) for val in vec] for vec in raw_embeddings.tolist()
            ]

            validate_batch_embeddings(
                vectors,
                expected_dim=self.expected_dim,
                check_normalization=True,
            )

            logger.info("Generated %d normalized embeddings successfully", len(vectors))
            return vectors

        except EmbeddingValidationError:
            raise
        except Exception as exc:
            logger.error("Batch embedding generation failed: %s", exc)
            raise EmbeddingGenerationError(f"Batch embedding generation failed: {str(exc)}") from exc
