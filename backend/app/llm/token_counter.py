"""OpenAI Token Estimation and Budget Accounting Module for Retrieva AI.

Provides accurate token counting using tiktoken where available, model context
limit resolution, ChatML message framing allowances, and robust fallback handling.

CRITICAL DISTINCTION — EXACT BPE COUNTING VS. ESTIMATED OVERHEAD:
-----------------------------------------------------------------
1. Exact Text-Token Counting:
   Raw text strings (system prompt, user query, retrieved chunk text, and template
   strings) are encoded to exact token counts using tiktoken Byte-Pair Encoding (BPE)
   for the target model (e.g., 'o200k_base' for gpt-4o / gpt-4o-mini; 'cl100k_base'
   for gpt-4 / gpt-3.5-turbo).

2. Estimated Allowances for API Framing:
   OpenAI Chat Completions request serialization occurs server-side. Per OpenAI
   documentation, ChatML message envelope tokens (3 tokens per message for '<|im_start|>{role}\\n',
   3 priming tokens for '<|im_start|>assistant<|message|>') and JSON response format
   schema overhead cannot be directly observed on the client prior to HTTP transmission.
   These fixed overhead allowances are documented conservative approximations, NOT
   guaranteed exact API server tokens.

3. Safety Margin:
   A conservative safety margin (default 50 tokens) is included to absorb any server-side
   ChatML variation or tokenization boundary artifacts, ensuring the prompt payload never
   exceeds model or configured context window ceilings.
"""

from typing import Any, Dict, List, NamedTuple, Optional
import tiktoken

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.document_processing.chunker import count_tokens_text

logger = get_logger("llm.token_counter")


class TokenCountResult(NamedTuple):
    """Result of token estimation."""

    token_count: int
    encoding_name: str
    is_approximate: bool
    fallback_used: bool


class TokenBudgetReport(NamedTuple):
    """Structured report of request token budget breakdown."""

    system_tokens: int
    query_tokens: int
    formatting_overhead: int
    reserved_completion_tokens: int
    safety_margin: int
    total_overhead_tokens: int
    allowed_context_budget: int
    model_context_limit: int
    max_context_ceiling: int
    is_approximate: bool


class TokenCounter:
    """Manages token estimation, tokenizer selection, and context budget allocation."""

    # Standard context window limits for recognized OpenAI chat models
    MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4o-2024-08-06": 128000,
        "gpt-4o-mini-2024-07-18": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        "gemini-2.5-flash": 1_048_576,
        "gemini-flash-lite-latest": 1_048_576,
    }

    _cached_encodings: Dict[str, Any] = {}
    _gemini_model_prefixes = ("gemini-",)

    @classmethod
    def get_encoding(cls, model_name: Optional[str] = None) -> tuple[Any, str, bool, bool]:
        """Resolve the appropriate tiktoken encoding for a given model.

        Returns:
            Tuple of (encoding_object, encoding_name, is_approximate, fallback_used).
        """
        settings = get_settings()
        target_model = (model_name or settings.active_model).strip()

        if target_model in cls._cached_encodings:
            enc, enc_name, is_approx, fallback = cls._cached_encodings[target_model]
            return enc, enc_name, is_approx, fallback

        if target_model.lower().startswith(cls._gemini_model_prefixes):
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                result = (encoding, "cl100k_base", True, True)
                cls._cached_encodings[target_model] = result
                logger.info(
                    "Using approximate cl100k_base token estimates for Gemini model '%s'; "
                    "tiktoken does not provide Gemini-native tokenization.",
                    target_model,
                )
                return result
            except Exception as exc:
                logger.error("Failed to load fallback encoding for Gemini model '%s': %s", target_model, exc)
                result = (None, "heuristic_fallback", True, True)
                cls._cached_encodings[target_model] = result
                return result

        try:
            encoding = tiktoken.encoding_for_model(target_model)
            result = (encoding, encoding.name, False, False)
            cls._cached_encodings[target_model] = result
            return result
        except KeyError:
            logger.warning(
                "Model '%s' not recognized by tiktoken. Falling back to 'cl100k_base' encoding (approximate).",
                target_model,
            )
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                result = (encoding, "cl100k_base", True, True)
                cls._cached_encodings[target_model] = result
                return result
            except Exception as exc:
                logger.error("Failed to load 'cl100k_base' encoding: %s", exc)
                result = (None, "heuristic_fallback", True, True)
                cls._cached_encodings[target_model] = result
                return result
        except Exception as exc:
            logger.error("Unexpected error obtaining tokenizer for model '%s': %s", target_model, exc)
            result = (None, "heuristic_fallback", True, True)
            cls._cached_encodings[target_model] = result
            return result

    @classmethod
    def count_tokens(cls, text: str, model_name: Optional[str] = None) -> TokenCountResult:
        """Count tokens in a string using the model's exact tokenizer or documented fallback.

        Args:
            text: Text string to count.
            model_name: OpenAI model identifier (e.g., 'gpt-4o-mini').

        Returns:
            TokenCountResult with token count, encoding name, and approximation flags.
        """
        if not text:
            return TokenCountResult(
                token_count=0,
                encoding_name="none",
                is_approximate=False,
                fallback_used=False,
            )

        encoding, enc_name, is_approx, fallback = cls.get_encoding(model_name)

        if encoding is not None:
            try:
                tokens = len(encoding.encode(text))
                return TokenCountResult(
                    token_count=tokens,
                    encoding_name=enc_name,
                    is_approximate=is_approx,
                    fallback_used=fallback,
                )
            except Exception as exc:
                logger.warning("tiktoken encode failed: %s; using conservative fallback", exc)

        # Conservative fallback using SentenceTransformers tokenizer + 15% safety buffer
        base_tokens = count_tokens_text(text)
        conservative_count = int(base_tokens * 1.15) + 1
        return TokenCountResult(
            token_count=conservative_count,
            encoding_name="sentence_transformers_heuristic",
            is_approximate=True,
            fallback_used=True,
        )

    @classmethod
    def count_chat_tokens(
        cls,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        api_overhead_allowance: int = 25,
    ) -> TokenCountResult:
        """Count tokens in a chat completion payload following OpenAI ChatML conventions.

        OpenAI ChatML Convention:
        - Each message has 3 tokens for framing: `<|im_start|>{role}\\n{content}<|im_end|>\\n`
        - Assistant response priming: 3 tokens (`<|im_start|>assistant<|message|>`)
        - `api_overhead_allowance`: Allowance for response_format JSON schemas and system headers.

        Qualification:
        Server-side API serialization overhead is not directly observable on the client.
        Therefore, this calculation combines exact BPE token counts for text contents with
        documented estimated allowances for API message envelopes.
        """
        total = 0
        is_approx = False
        fallback = False
        enc_name = "unknown"

        for message in messages:
            # 3 tokens per message framing
            total += 3
            content = message.get("content", "")
            res = cls.count_tokens(content, model_name=model_name)
            total += res.token_count
            enc_name = res.encoding_name
            if res.is_approximate:
                is_approx = True
            if res.fallback_used:
                fallback = True

        # Priming tokens (3) + API overhead allowance
        total += 3 + api_overhead_allowance

        return TokenCountResult(
            token_count=total,
            encoding_name=enc_name,
            is_approximate=is_approx,
            fallback_used=fallback,
        )

    @classmethod
    def get_model_context_limit(
        cls,
        model_name: Optional[str] = None,
        override_limit: Optional[int] = None,
    ) -> int:
        """Resolve the effective total context ceiling for the request.

        Guarantees that the context ceiling never exceeds the model's supported
        hardware context limit, even if a larger override is passed.
        """
        settings = get_settings()
        target_model = model_name or settings.active_model
        known_window = cls.MODEL_CONTEXT_WINDOWS.get(target_model, 128000)

        configured_limit = (
            override_limit
            if (override_limit is not None and override_limit > 0)
            else settings.TOTAL_REQUEST_TOKEN_LIMIT
        )
        if configured_limit and configured_limit > 0:
            return min(configured_limit, known_window)
        return known_window

    @classmethod
    def calculate_request_token_budget(
        cls,
        query: str,
        system_prompt: Optional[str] = None,
        reserved_completion_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        total_request_limit: Optional[int] = None,
        max_context_limit: Optional[int] = None,
        safety_margin: int = 50,
    ) -> TokenBudgetReport:
        """Dynamically compute the token budget available for retrieved document context.

        Formula:
            TotalCeiling = min(ConfiguredLimit, ModelContextWindow)
            Overhead = SystemTokens + QueryTokens + FormattingAndChatMLOverhead + ReservedCompletionTokens + SafetyMargin
            AllowedContext = max(0, min(MAX_CONTEXT_TOKENS, TotalCeiling - Overhead))

        Guarantees:
            1. Allowed context NEVER exceeds MAX_CONTEXT_TOKENS ceiling.
            2. Total request budget NEVER exceeds TotalCeiling (clamped to model hardware window).
            3. Fallback tokenizers are explicitly flagged as is_approximate=True.
            4. Message framing and structured output allowances are qualified as estimated allowances.

        Args:
            query: User's natural language question.
            system_prompt: System prompt string.
            reserved_completion_tokens: Tokens reserved for model generation.
            model_name: OpenAI model identifier.
            total_request_limit: Total token ceiling for prompt + completion.
            max_context_limit: Configured upper bound for retrieved context tokens.
            safety_margin: Buffer absorbing API serialization variations (~50 tokens).

        Returns:
            TokenBudgetReport with detailed breakdown.

        Raises:
            ValueError: If fixed overhead (system prompt + query + formatting + completion)
                        exceeds total_request_limit.
        """
        from backend.app.llm.prompt_builder import PromptBuilder

        settings = get_settings()
        eff_model = model_name or settings.active_model
        eff_sys_prompt = system_prompt or PromptBuilder.get_system_prompt()
        eff_total_limit = cls.get_model_context_limit(eff_model, override_limit=total_request_limit)
        eff_context_limit = max_context_limit or settings.MAX_CONTEXT_TOKENS
        eff_completion = (
            reserved_completion_tokens
            if reserved_completion_tokens is not None
            else settings.DEFAULT_MAX_COMPLETION_TOKENS
        )

        sys_res = cls.count_tokens(eff_sys_prompt, model_name=eff_model)
        query_res = cls.count_tokens(query, model_name=eff_model)

        # Message framing overhead (2 messages: system + user = 6 tokens, + 3 priming = 9 tokens)
        chatml_overhead = 9

        # Prompt formatting template overhead (delimiters + JSON instructions)
        fmt_template = (
            "=== BEGIN RETRIEVED CONTEXT (UNTRUSTED DATA) ===\n\n"
            "=== END RETRIEVED CONTEXT ===\n\n"
            "USER QUESTION: \n\n"
            "Provide your grounded answer and citations in the specified JSON format:"
        )
        fmt_res = cls.count_tokens(fmt_template, model_name=eff_model)
        formatting_overhead = chatml_overhead + fmt_res.token_count + 10  # 10 tokens buffer for JSON schema

        total_overhead = (
            sys_res.token_count
            + query_res.token_count
            + formatting_overhead
            + eff_completion
            + safety_margin
        )

        is_approximate = sys_res.is_approximate or query_res.is_approximate or fmt_res.is_approximate

        if total_overhead >= eff_total_limit:
            raise ValueError(
                f"Question and completion overhead exceed total request token limit of {eff_total_limit} "
                f"(requires {total_overhead} tokens: system={sys_res.token_count}, query={query_res.token_count}, "
                f"format={formatting_overhead}, completion={eff_completion}, safety={safety_margin}). "
                f"Please submit a shorter question."
            )

        available_for_context = eff_total_limit - total_overhead
        allowed_budget = max(0, min(eff_context_limit, available_for_context))

        return TokenBudgetReport(
            system_tokens=sys_res.token_count,
            query_tokens=query_res.token_count,
            formatting_overhead=formatting_overhead,
            reserved_completion_tokens=eff_completion,
            safety_margin=safety_margin,
            total_overhead_tokens=total_overhead,
            allowed_context_budget=allowed_budget,
            model_context_limit=eff_total_limit,
            max_context_ceiling=eff_context_limit,
            is_approximate=is_approximate,
        )
        chatml_overhead = 9

        # Prompt formatting template overhead (delimiters + JSON instructions)
        fmt_template = (
            "=== BEGIN RETRIEVED CONTEXT (UNTRUSTED DATA) ===\n\n"
            "=== END RETRIEVED CONTEXT ===\n\n"
            "USER QUESTION: \n\n"
            "Provide your grounded answer and citations in the specified JSON format:"
        )
        fmt_res = cls.count_tokens(fmt_template, model_name=eff_model)
        formatting_overhead = chatml_overhead + fmt_res.token_count + 10  # 10 tokens buffer for JSON schema

        total_overhead = (
            sys_res.token_count
            + query_res.token_count
            + formatting_overhead
            + eff_completion
            + safety_margin
        )

        is_approximate = sys_res.is_approximate or query_res.is_approximate or fmt_res.is_approximate

        if total_overhead >= eff_total_limit:
            raise ValueError(
                f"Question and completion overhead exceed total request token limit of {eff_total_limit} "
                f"(requires {total_overhead} tokens: system={sys_res.token_count}, query={query_res.token_count}, "
                f"format={formatting_overhead}, completion={eff_completion}, safety={safety_margin}). "
                f"Please submit a shorter question."
            )

        available_for_context = eff_total_limit - total_overhead
        allowed_budget = max(0, min(eff_context_limit, available_for_context))

        return TokenBudgetReport(
            system_tokens=sys_res.token_count,
            query_tokens=query_res.token_count,
            formatting_overhead=formatting_overhead,
            reserved_completion_tokens=eff_completion,
            safety_margin=safety_margin,
            total_overhead_tokens=total_overhead,
            allowed_context_budget=allowed_budget,
            model_context_limit=eff_total_limit,
            max_context_ceiling=eff_context_limit,
            is_approximate=is_approximate,
        )

