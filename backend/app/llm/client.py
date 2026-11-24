"""OpenAI-compatible LLM client integration module for Retrieva AI.

Provides isolated, production-grade LLM communication with strict credential protection,
configurable timeouts, error mapping, and streaming architecture hooks.
"""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
import openai
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMResponseParsingError,
    LLMTimeoutError,
)

logger = get_logger("llm.client")


class OpenAIClient:
    """Encapsulated OpenAI API client for Retrieva AI.

    Guarantees:
    1. Credentials are read solely from environment settings and never hardcoded or logged.
    2. Exceptions from openai SDK are mapped to domain-specific LLMError subclasses.
    3. Exposes synchronous completion and non-blocking asynchronous streaming interfaces.
    """

    _instance: Optional["OpenAIClient"] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        client: Optional[Any] = None,
        async_client: Optional[Any] = None,
    ):
        """Initialize the selected OpenAI-compatible provider client.

        Args:
            api_key: Optional explicit API key. If omitted, fetched for the active provider.
            model: Model identifier. If omitted, fetched for the active provider.
            timeout: Request timeout in seconds.
            max_retries: Automatic SDK retries for transient HTTP errors.
            client: Optional mock or pre-configured openai.OpenAI client for synchronous operations.
            async_client: Optional mock or pre-configured openai.AsyncOpenAI client for non-blocking streaming.
        """
        settings = get_settings()
        self.provider = settings.active_llm_provider
        self.provider_display = "Google Gemini" if self.provider == "gemini" else "OpenAI"
        if api_key is not None:
            raw_key = api_key
        else:
            raw_key = settings.active_api_key

        # Reject empty strings or common environment template placeholders as unconfigured
        if raw_key and any(
            ph in raw_key.lower()
            for ph in [
                "your_api_key",
                "your-api-key",
                "your-openai",
                "your-gemini",
                "your-google",
                "placeholder",
                "xxx",
            ]
        ):
            self.api_key = None
        else:
            self.api_key = raw_key.strip() if isinstance(raw_key, str) else raw_key

        self.model = model or settings.active_model
        self.base_url = settings.active_base_url
        self.timeout = timeout if timeout is not None else settings.OPENAI_REQUEST_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else settings.OPENAI_MAX_RETRIES

        # Never print or log the API key directly
        logger.info(
            "Initializing LLM client (provider=%s, model=%s, timeout=%.1fs, max_retries=%d, key_present=%s)",
            self.provider,
            self.model,
            self.timeout,
            self.max_retries,
            bool(self.api_key),
        )

        if client is not None:
            self._client = client
        elif self.api_key:
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**client_kwargs)
        else:
            self._client = None

        if async_client is not None:
            self._async_client = async_client
        elif self.api_key:
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._async_client = openai.AsyncOpenAI(**client_kwargs)
        else:
            self._async_client = None

    @classmethod
    def get_instance(cls, **kwargs) -> "OpenAIClient":
        """Return the shared singleton OpenAIClient instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for test isolation)."""
        cls._instance = None

    def _sanitize_error_message(self, message: str) -> str:
        """Sanitize error messages to prevent leakage of credentials or sensitive tokens."""
        if not message:
            return ""
        sanitized = str(message)
        if self.api_key and len(self.api_key) > 5:
            sanitized = sanitized.replace(self.api_key, "[REDACTED_API_KEY]")
        import re
        sanitized = re.sub(r"sk-[a-zA-Z0-9_\-]{20,}", "[REDACTED_API_KEY]", sanitized)
        sanitized = re.sub(r"AIza[a-zA-Z0-9_\-]{20,}", "[REDACTED_API_KEY]", sanitized)
        sanitized = re.sub(r"AQ\.[a-zA-Z0-9_\-]{20,}", "[REDACTED_API_KEY]", sanitized)
        return sanitized

    def _ensure_client_ready(self) -> Any:
        """Verify that an active OpenAI client with valid configuration exists."""
        if not self.api_key or self._client is None:
            raise LLMConfigError(
                f"{self.provider_display} API key is missing. Set the configured provider API key in the environment or .env file."
            )
        return self._client

    def _ensure_async_client_ready(self) -> Any:
        """Verify that an active AsyncOpenAI client with valid configuration exists.

        Recreates the underlying AsyncOpenAI client lazily if it was previously closed.
        """
        if not self.api_key:
            raise LLMConfigError(
                f"{self.provider_display} API key is missing. Set the configured provider API key in the environment or .env file."
            )
        if self._async_client is None:
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._async_client = openai.AsyncOpenAI(**client_kwargs)
        return self._async_client

    async def aclose(self) -> None:
        """Close the underlying asynchronous client session if active.

        This method is idempotent and can be safely called repeatedly.
        """
        client_to_close = self._async_client
        self._async_client = None
        if client_to_close is not None and hasattr(client_to_close, "close"):
            try:
                await client_to_close.close()
            except Exception as exc:
                logger.debug("Error closing async OpenAI client: %s", exc)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a standard chat completion request.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens in generated completion.

        Returns:
            Dictionary containing:
                - 'content': Generated response string.
                - 'model': Actual model string reported by API.
                - 'prompt_tokens': Tokens in prompt.
                - 'completion_tokens': Tokens in completion.
                - 'total_tokens': Total tokens consumed.

        Raises:
            LLMConfigError: If API key is missing.
            LLMAuthenticationError: On 401 Unauthorized.
            LLMRateLimitError: On 429 Rate Limit / Quota Exceeded.
            LLMTimeoutError: On request timeout.
            LLMNetworkError: On network connection failure.
            LLMResponseParsingError: On malformed or empty response.
        """
        client = self._ensure_client_ready()
        settings = get_settings()
        effective_temp = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        effective_max_tokens = max_tokens if max_tokens is not None else settings.DEFAULT_MAX_COMPLETION_TOKENS

        try:
            logger.debug("Dispatching %s completion request to model %s", self.provider_display, self.model)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=effective_temp,
                max_tokens=effective_max_tokens,
            )

            if not response.choices:
                raise LLMResponseParsingError(f"{self.provider_display} returned an empty choice list.")

            choice = response.choices[0]
            content = choice.message.content
            if content is None:
                raise LLMResponseParsingError(f"{self.provider_display} returned null content in message.")

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None

            return {
                "content": content.strip(),
                "model": response.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        except AuthenticationError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s authentication failed: %s", self.provider_display, msg)
            raise LLMAuthenticationError(msg) from exc
        except RateLimitError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s rate limit or quota exceeded: %s", self.provider_display, msg)
            raise LLMRateLimitError(msg) from exc
        except APITimeoutError as exc:
            logger.error("OpenAI API call timed out after %.1fs: %s", self.timeout, exc)
            raise LLMTimeoutError(f"Request timed out after {self.timeout}s") from exc
        except APIConnectionError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s connection failed: %s", self.provider_display, msg)
            raise LLMNetworkError(f"Failed to connect to {self.provider_display}: {msg}") from exc
        except APIError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s API returned error: %s", self.provider_display, msg)
            raise LLMError(f"{self.provider_display} API error: {msg}") from exc
        except (LLMResponseParsingError, LLMConfigError):
            raise
        except Exception as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("Unexpected error in %s client: %s", self.provider_display, msg)
            raise LLMError(f"Unexpected LLM generation error: {msg}") from exc

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Streaming generation interface (prepared for Phase 6 / future streaming endpoints).

        Yields string tokens incrementally as they arrive over the wire from OpenAI.
        Guarantees deterministic stream teardown and resource release via try...finally.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in generated completion.

        Yields:
            Token text strings incrementally.

        Raises:
            LLMError subclasses on failure.
        """
        client = self._ensure_client_ready()
        settings = get_settings()
        effective_temp = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        effective_max_tokens = max_tokens if max_tokens is not None else settings.DEFAULT_MAX_COMPLETION_TOKENS

        stream = None
        try:
            logger.debug("Opening %s completion stream with model %s", self.provider_display, self.model)
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=effective_temp,
                max_tokens=effective_max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except AuthenticationError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s authentication failed during stream: %s", self.provider_display, msg)
            raise LLMAuthenticationError(msg) from exc
        except RateLimitError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s rate limit exceeded during stream: %s", self.provider_display, msg)
            raise LLMRateLimitError(msg) from exc
        except APITimeoutError as exc:
            logger.error("OpenAI stream timed out: %s", exc)
            raise LLMTimeoutError("Stream timed out") from exc
        except APIConnectionError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s stream connection error: %s", self.provider_display, msg)
            raise LLMNetworkError(f"{self.provider_display} stream connection failed: {msg}") from exc
        except APIError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s API error during stream: %s", self.provider_display, msg)
            raise LLMError(f"{self.provider_display} API error: {msg}") from exc
        except Exception as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("Error during streaming generation: %s", msg)
            raise LLMError(f"Streaming error: {msg}") from exc
        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception as close_exc:
                    logger.debug("Stream close raised notice: %s", close_exc)

    async def generate_stream_async(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Asynchronous streaming generation interface using non-blocking socket I/O.

        Yields string tokens incrementally as they arrive over the wire from OpenAI
        via AsyncOpenAI, preserving the FastAPI event loop for concurrent tasks.
        Guarantees deterministic stream teardown and resource release via try...finally.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in generated completion.

        Yields:
            Token text strings incrementally.

        Raises:
            LLMError subclasses on failure.
        """
        client = self._ensure_async_client_ready()
        settings = get_settings()
        effective_temp = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        effective_max_tokens = max_tokens if max_tokens is not None else settings.DEFAULT_MAX_COMPLETION_TOKENS

        stream = None
        try:
            logger.debug("Opening async %s completion stream with model %s", self.provider_display, self.model)
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=effective_temp,
                max_tokens=effective_max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except AuthenticationError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s authentication failed during async stream: %s", self.provider_display, msg)
            raise LLMAuthenticationError(msg) from exc
        except RateLimitError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s rate limit exceeded during async stream: %s", self.provider_display, msg)
            raise LLMRateLimitError(msg) from exc
        except APITimeoutError as exc:
            logger.error("OpenAI async stream timed out: %s", exc)
            raise LLMTimeoutError("Stream timed out") from exc
        except APIConnectionError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s async stream connection error: %s", self.provider_display, msg)
            raise LLMNetworkError(f"{self.provider_display} stream connection failed: {msg}") from exc
        except APIError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s API error during async stream: %s", self.provider_display, msg)
            raise LLMError(f"{self.provider_display} API error: {msg}") from exc
        except Exception as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("Error during async streaming generation: %s", msg)
            raise LLMError(f"Streaming error: {msg}") from exc
        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    await stream.close()
                except Exception as close_exc:
                    logger.debug("Error closing async stream: %s", close_exc)

    async def rewrite_query_async(
        self,
        messages: List[Dict[str, str]],
        timeout: Optional[float] = None,
    ) -> str:
        """Execute conversational query rewriting asynchronously.

        Args:
            messages: Prompt messages containing instructions, history, and latest query.
            timeout: Optional maximum duration in seconds for the rewriting call.

        Returns:
            The standalone rewritten query string.

        Raises:
            LLMError subclasses on API failure or timeout.
        """
        client = self._ensure_async_client_ready()
        eff_timeout = timeout or get_settings().QUERY_REWRITE_TIMEOUT

        try:
            logger.debug("Dispatching async %s query rewriting request to model %s", self.provider_display, self.model)
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=150,
                timeout=eff_timeout,
            )

            if not response.choices or not response.choices[0].message.content:
                raise LLMResponseParsingError(
                    f"{self.provider_display} returned empty completion for query rewrite."
                )

            rewritten = response.choices[0].message.content.strip()
            # Remove any wrapping quotes or redundant prefixes if present
            if (rewritten.startswith('"') and rewritten.endswith('"')) or (rewritten.startswith("'") and rewritten.endswith("'")):
                rewritten = rewritten[1:-1].strip()
            return rewritten

        except AuthenticationError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s authentication failed during query rewrite: %s", self.provider_display, msg)
            raise LLMAuthenticationError(msg) from exc
        except RateLimitError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.error("%s rate limit exceeded during query rewrite: %s", self.provider_display, msg)
            raise LLMRateLimitError(msg) from exc
        except APITimeoutError as exc:
            logger.warning("OpenAI query rewrite timed out after %.1fs: %s", eff_timeout, exc)
            raise LLMTimeoutError(f"Query rewrite timed out after {eff_timeout}s") from exc
        except APIConnectionError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.warning("%s connection error during query rewrite: %s", self.provider_display, msg)
            raise LLMNetworkError(f"{self.provider_display} query rewrite connection failed: {msg}") from exc
        except APIError as exc:
            msg = self._sanitize_error_message(str(exc))
            logger.warning("%s API error during query rewrite: %s", self.provider_display, msg)
            raise LLMError(f"{self.provider_display} query rewrite API error: {msg}") from exc

    def rewrite_query(
        self,
        messages: List[Dict[str, str]],
        timeout: Optional[float] = None,
    ) -> str:
        """Execute conversational query rewriting synchronously."""
        client = self._ensure_client_ready()
        eff_timeout = timeout or get_settings().QUERY_REWRITE_TIMEOUT

        try:
            logger.debug("Dispatching sync %s query rewriting request to model %s", self.provider_display, self.model)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=150,
                timeout=eff_timeout,
            )

            if not response.choices or not response.choices[0].message.content:
                raise LLMResponseParsingError(
                    f"{self.provider_display} returned empty completion for query rewrite."
                )

            rewritten = response.choices[0].message.content.strip()
            if (rewritten.startswith('"') and rewritten.endswith('"')) or (rewritten.startswith("'") and rewritten.endswith("'")):
                rewritten = rewritten[1:-1].strip()
            return rewritten

        except AuthenticationError as exc:
            msg = self._sanitize_error_message(str(exc))
            raise LLMAuthenticationError(msg) from exc
        except RateLimitError as exc:
            msg = self._sanitize_error_message(str(exc))
            raise LLMRateLimitError(msg) from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"Query rewrite timed out after {eff_timeout}s") from exc
        except APIConnectionError as exc:
            msg = self._sanitize_error_message(str(exc))
            raise LLMNetworkError(f"{self.provider_display} query rewrite connection failed: {msg}") from exc
        except APIError as exc:
            msg = self._sanitize_error_message(str(exc))
            raise LLMError(f"{self.provider_display} query rewrite API error: {msg}") from exc
