"""Application configuration management using Pydantic Settings."""

import json
from functools import lru_cache
from typing import List, Optional, Union

from pydantic import field_validator, model_validator
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
    STALE_INGESTION_TIMEOUT_SECONDS: int = 300  # 5 minutes

    # Embedding Settings (Phase 3)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Vector Storage Settings (Phase 3)
    CHROMA_PERSIST_DIRECTORY: str = "chroma_data"
    CHROMA_COLLECTION_NAME: str = "retrieva_documents"
    BM25_MAX_CACHED_TENANTS: int = 50

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

    # Supabase JWT Authentication Settings (Phase 2 - never include service-role key)
    SUPABASE_URL: Optional[str] = None
    SUPABASE_ANON_KEY: Optional[str] = None
    SUPABASE_STORAGE_BUCKET: str = "retrieva-user-documents"
    SUPABASE_JWKS_URL: Optional[str] = None
    SUPABASE_JWT_ISSUER: Optional[str] = None
    SUPABASE_JWT_AUDIENCE: str = "authenticated"
    SUPABASE_JWT_ALGORITHMS: List[str] = ["ES256", "RS256"]
    SUPABASE_JWKS_CACHE_TTL_SECONDS: int = 300
    SUPABASE_JWKS_TIMEOUT_SECONDS: float = 5.0
    SUPABASE_REST_TIMEOUT_SECONDS: float = 10.0
    BACKEND_RPC_HMAC_SECRET: Optional[str] = None
    AUTH_ENABLED: bool = True

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

    @field_validator("SUPABASE_JWT_ALGORITHMS", mode="before")
    @classmethod
    def validate_jwt_algorithms(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse and strictly allowlist asymmetric JWT signing algorithms."""
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    items = [str(item).strip() for item in parsed] if isinstance(parsed, list) else []
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid SUPABASE_JWT_ALGORITHMS JSON: {exc}") from exc
            else:
                items = [item.strip() for item in stripped.split(",") if item.strip()]
        else:
            items = [str(item).strip() for item in v]

        allowed = {"ES256", "RS256"}
        if not items:
            raise ValueError("SUPABASE_JWT_ALGORITHMS must not be empty.")
        disallowed = [alg for alg in items if alg not in allowed]
        if disallowed:
            raise ValueError(
                f"Unsupported JWT algorithm(s) {disallowed}. Allowed asymmetric algorithms: {sorted(allowed)}."
            )
        return items

    @field_validator("SUPABASE_JWKS_CACHE_TTL_SECONDS")
    @classmethod
    def validate_jwks_cache_ttl(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("SUPABASE_JWKS_CACHE_TTL_SECONDS must be positive.")
        return value

    @field_validator("SUPABASE_JWKS_TIMEOUT_SECONDS")
    @classmethod
    def validate_jwks_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("SUPABASE_JWKS_TIMEOUT_SECONDS must be positive.")
        return value

    @field_validator("SUPABASE_REST_TIMEOUT_SECONDS")
    @classmethod
    def validate_rest_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("SUPABASE_REST_TIMEOUT_SECONDS must be positive.")
        return value

    @field_validator("BM25_MAX_CACHED_TENANTS")
    @classmethod
    def validate_bm25_max_cached_tenants(cls, value: int) -> int:
        if value < 1:
            raise ValueError("BM25_MAX_CACHED_TENANTS must be at least 1.")
        return value

    @property
    def resolved_jwks_url(self) -> Optional[str]:
        """Return explicit JWKS URL or derive it from SUPABASE_URL."""
        if self.SUPABASE_JWKS_URL and self.SUPABASE_JWKS_URL.strip():
            return self.SUPABASE_JWKS_URL.strip()
        if self.SUPABASE_URL and self.SUPABASE_URL.strip():
            return f"{self.SUPABASE_URL.strip().rstrip('/')}/auth/v1/.well-known/jwks.json"
        return None

    @property
    def resolved_issuer(self) -> Optional[str]:
        """Return explicit JWT issuer or derive it from SUPABASE_URL."""
        if self.SUPABASE_JWT_ISSUER and self.SUPABASE_JWT_ISSUER.strip():
            return self.SUPABASE_JWT_ISSUER.strip()
        if self.SUPABASE_URL and self.SUPABASE_URL.strip():
            return f"{self.SUPABASE_URL.strip().rstrip('/')}/auth/v1"
        return None

    @model_validator(mode="after")
    def validate_production_auth_fail_closed(self) -> "Settings":
        """Fail closed when authentication configuration is missing or disabled outside development."""
        env = self.APP_ENV.strip().lower()
        if env != "development":
            if not self.AUTH_ENABLED:
                raise ValueError(
                    f"AUTH_ENABLED cannot be False when APP_ENV='{self.APP_ENV}'."
                )
            if not self.resolved_jwks_url or not self.resolved_issuer:
                raise ValueError(
                    "Supabase authentication configuration is required outside development "
                    "(set SUPABASE_URL, or both SUPABASE_JWKS_URL and SUPABASE_JWT_ISSUER)."
                )
            if not self.SUPABASE_JWT_AUDIENCE or not self.SUPABASE_JWT_AUDIENCE.strip():
                raise ValueError("SUPABASE_JWT_AUDIENCE must be configured outside development.")
            if not self.BACKEND_RPC_HMAC_SECRET or not self.BACKEND_RPC_HMAC_SECRET.strip():
                raise ValueError("BACKEND_RPC_HMAC_SECRET must be configured outside development.")
        return self

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
