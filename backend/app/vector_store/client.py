"""ChromaDB persistent client provider for Retrieva AI.

Manages the lifecycle and initialization of the ChromaDB PersistentClient,
ensuring vectors are persisted to the local filesystem without relying on
in-memory transient state.
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings

try:
    from backend.app.core.config import get_settings
    from backend.app.core.logging import get_logger
except ImportError:
    from app.core.config import get_settings
    from app.core.logging import get_logger

logger = get_logger("vector_store.client")

_persistent_client: Optional[ClientAPI] = None


def get_chroma_client(persist_directory: Optional[str] = None) -> ClientAPI:
    """Return a ChromaDB PersistentClient instance.

    Args:
        persist_directory: Optional filesystem path for Chroma persistence.
            Defaults to the path configured in application settings.

    Returns:
        chromadb.ClientAPI: Configured persistent client instance.
    """
    global _persistent_client

    settings = get_settings()
    target_dir = persist_directory or settings.CHROMA_PERSIST_DIRECTORY

    # Ensure absolute path or resolved relative path
    persist_path = Path(target_dir).resolve()
    persist_path.mkdir(parents=True, exist_ok=True)

    # Return cached client if already initialized with the same directory
    if _persistent_client is not None and persist_directory is None:
        return _persistent_client

    logger.info("Initializing ChromaDB PersistentClient at %s", persist_path)

    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )

    if persist_directory is None:
        _persistent_client = client

    return client


def reset_client_cache() -> None:
    """Reset the cached client instance, useful for tests and client restart simulations."""
    global _persistent_client
    _persistent_client = None
