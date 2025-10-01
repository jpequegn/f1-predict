"""Anthropic Claude LLM provider implementation.

This module provides integration with Anthropic's Claude API including
Claude 3.5 Sonnet and other Claude models with proper error handling.
"""

from typing import Any, AsyncIterator, Optional

import structlog
from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic, AnthropicError

from f1_predict.llm.base import BaseLLMProvider, LLMConfig, LLMResponse
from f1_predict.llm.exceptions import (
    AuthenticationError,
    InvalidResponseError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)

logger = structlog.get_logger(__name__)

# Anthropic pricing per 1K tokens (as of 2024)
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider implementation.

    Supports Claude 3.5 Sonnet, Claude 3 Opus, and other Claude models
    with streaming, token counting, and cost estimation.
    """

    def __init__(self, config: LLMConfig, api_key: str):
        """Initialize Anthropic provider.

        Args:
            config: Provider configuration
            api_key: Anthropic API key

        Raises:
            AuthenticationError: If API key is invalid
        """
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=api_key)
        self.logger.info("anthropic_provider_initialized", model=config.model)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text completion using Anthropic Claude API.

        Args:
            prompt: User prompt text
            system_prompt: Optional system instruction
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            RateLimitError: When rate limits are exceeded
            AuthenticationError: For authentication failures
            TimeoutError: When request times out
            ProviderUnavailableError: When service is unavailable
            InvalidResponseError: For malformed responses
        """
        try:
            # Build request parameters
            params = {
                "model": self.config.model,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                params["system"] = system_prompt

            self.logger.debug("anthropic_request_started", **params)

            # Make API call
            response = await self.client.messages.create(**params)

            # Extract response data
            content = response.content[0].text
            if not content:
                msg = "Empty response from Anthropic"
                raise InvalidResponseError(msg)

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            estimated_cost = self.estimate_cost(input_tokens, output_tokens)

            llm_response = LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
                metadata={
                    "stop_reason": response.stop_reason,
                    "model_used": response.model,
                },
            )

            self.logger.info(
                "anthropic_response_received",
                tokens=total_tokens,
                cost=estimated_cost,
                stop_reason=response.stop_reason,
            )

            return llm_response

        except AnthropicError as e:
            self.logger.error(
                "anthropic_error", error=str(e), error_type=type(e).__name__
            )
            error_message = str(e)
            if "rate_limit" in error_message.lower() or "429" in error_message:
                raise RateLimitError(f"Anthropic rate limit exceeded: {e}") from e
            elif "authentication" in error_message.lower() or "401" in error_message:
                raise AuthenticationError(f"Anthropic authentication failed: {e}") from e
            elif "timeout" in error_message.lower():
                raise TimeoutError(f"Anthropic request timed out: {e}") from e
            elif "500" in error_message or "503" in error_message:
                raise ProviderUnavailableError(
                    f"Anthropic service unavailable: {e}"
                ) from e
            else:
                raise InvalidResponseError(f"Anthropic API error: {e}") from e

    async def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate streaming text completion.

        Args:
            prompt: User prompt text
            system_prompt: Optional system instruction
            **kwargs: Additional Anthropic-specific parameters

        Yields:
            Content chunks as they are generated

        Raises:
            RateLimitError: When rate limits are exceeded
            AuthenticationError: For authentication failures
            TimeoutError: When request times out
            ProviderUnavailableError: When service is unavailable
        """
        try:
            # Build request parameters
            params = {
                "model": self.config.model,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }

            if system_prompt:
                params["system"] = system_prompt

            self.logger.debug("anthropic_streaming_started", **params)

            # Make streaming API call
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except AnthropicError as e:
            self.logger.error(
                "anthropic_streaming_error", error=str(e), error_type=type(e).__name__
            )
            error_message = str(e)
            if "rate_limit" in error_message.lower():
                raise RateLimitError(f"Anthropic rate limit exceeded: {e}") from e
            elif "authentication" in error_message.lower():
                raise AuthenticationError(f"Anthropic authentication failed: {e}") from e
            elif "timeout" in error_message.lower():
                raise TimeoutError(f"Anthropic request timed out: {e}") from e
            else:
                raise ProviderUnavailableError(f"Anthropic service error: {e}") from e

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate number of tokens
        """
        # Rough approximation: 1 token ≈ 4 characters
        # Anthropic uses similar tokenization to GPT
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Get pricing for model (fallback to Claude 3 Haiku if not found)
        pricing = ANTHROPIC_PRICING.get(
            self.config.model, ANTHROPIC_PRICING["claude-3-haiku-20240307"]
        )

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost

    @property
    def name(self) -> str:
        """Get provider name.

        Returns:
            Provider name identifier
        """
        return "anthropic"

    @property
    def available_models(self) -> list[str]:
        """Get list of available Anthropic models.

        Returns:
            List of model identifiers
        """
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
