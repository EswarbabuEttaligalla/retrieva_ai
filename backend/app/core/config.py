"""Application configuration management using Pydantic Settings."""

import json
from functools import lru_cache
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Retrieva AI backend service.
    
    Loads configuration from environment variables and optionally from an .env file.
    """

    # Project Information
    PROJECT_NAME: str = "Retrieva AI"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"

    # Server Bindings
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    # Logging Settings
    LOG_LEVEL: str = "INFO"

    # Document Upload & Processing Settings (Phase 6)
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = [".pdf", ".txt"]

    # Embedding Settings (Phase 3)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Vector Storage Settings (Phase 3)
    CHROMA_PERSIST_DIRECTORY: str = "chroma_data"
    CHROMA_COLLECTION_NAME: str = "retrieva_documents"

    # LLM Answer Generation & RAG Orchestration Settings (Phase 5)
    LLM_PROVIDER: str = "gemini"
    OPENAI_API_KEY: Union[str, None] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: Union[str, None] = None
    GEMINI_API_KEY: Union[str, None] = None
    GEMINI_MODEL: str = "gemini-flash-lite-latest"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    OPENAI_REQUEST_TIMEOUT: float = 30.0
    OPENAI_MAX_RETRIES: int = 2
    MAX_CONTEXT_TOKENS: int = 2500
    TOTAL_REQUEST_TOKEN_LIMIT: int = 4096
    DEFAULT_TEMPERATURE: float = 0.1
    DEFAULT_MAX_COMPLETION_TOKENS: int = 800

    # Persistent Conversation & Chat Memory Settings (Phase 7 Milestone 3)
    CONVERSATION_DB_PATH: str = "chroma_data/conversation_history.db"
    CONVERSATION_HISTORY_WINDOW: int = 5  # Number of recent turns (query/answer pairs)
    QUERY_REWRITE_ENABLED: bool = True
    QUERY_REWRITE_TIMEOUT: float = 10.0

    @field_validator("LLM_PROVIDER", mode="before")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        """Normalize and validate the configured LLM provider."""
        provider = str(value).strip().lower()
        if provider not in {"openai", "gemini"}:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{value}'. Supported providers are: openai, gemini."
            )
        return provider

    @property
    def active_llm_provider(self) -> str:
        """Return the normalized provider selected for LLM requests."""
        return self.LLM_PROVIDER

    @property
    def active_api_key(self) -> Union[str, None]:
        """Return the API key for the selected provider."""
        return self.GEMINI_API_KEY if self.active_llm_provider == "gemini" else self.OPENAI_API_KEY

    @property
    def active_model(self) -> str:
        """Return the model configured for the selected provider."""
        return self.GEMINI_MODEL if self.active_llm_provider == "gemini" else self.OPENAI_MODEL

    @property
    def active_base_url(self) -> Union[str, None]:
        """Return the optional provider endpoint override."""
        return self.GEMINI_BASE_URL if self.active_llm_provider == "gemini" else self.OPENAI_BASE_URL

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins whether supplied as JSON string or Python list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, str) and v.startswith("["):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        return v  # type: ignore

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
