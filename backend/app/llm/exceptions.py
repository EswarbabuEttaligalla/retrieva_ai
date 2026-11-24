"""Exceptions for LLM integration and generation in Retrieva AI."""


class LLMError(Exception):
    """Base exception for all LLM client and answer generation errors."""

    def __init__(self, message: str, user_message: str = "An error occurred during answer generation."):
        super().__init__(message)
        self.message = message
        self.user_message = user_message


class LLMConfigError(LLMError):
    """Raised when the LLM configuration or API key is missing or invalid."""

    def __init__(self, message: str = "OpenAI API key is not configured."):
        super().__init__(
            message=message,
            user_message="LLM service is not properly configured. Please check server settings.",
        )


class LLMAuthenticationError(LLMError):
    """Raised when authentication with OpenAI fails (401)."""

    def __init__(self, message: str = "Authentication with OpenAI API failed."):
        super().__init__(
            message=message,
            user_message="Authentication with the AI model failed. Please verify credentials.",
        )


class LLMRateLimitError(LLMError):
    """Raised when rate limits or quotas are exceeded (429)."""

    def __init__(self, message: str = "OpenAI rate limit or quota exceeded."):
        super().__init__(
            message=message,
            user_message="The AI service is currently busy or rate-limited. Please try again shortly.",
        )


class LLMTimeoutError(LLMError):
    """Raised when the OpenAI API call times out."""

    def __init__(self, message: str = "OpenAI API request timed out."):
        super().__init__(
            message=message,
            user_message="The request to the AI model timed out. Please try again.",
        )


class LLMNetworkError(LLMError):
    """Raised when network connectivity to OpenAI fails."""

    def __init__(self, message: str = "Network connection to OpenAI failed."):
        super().__init__(
            message=message,
            user_message="Unable to reach the AI model service due to a network error.",
        )


class LLMResponseParsingError(LLMError):
    """Raised when the model response is malformed, empty, or unparseable."""

    def __init__(self, message: str = "Failed to parse model response."):
        super().__init__(
            message=message,
            user_message="The AI model returned an unexpected response format.",
        )
