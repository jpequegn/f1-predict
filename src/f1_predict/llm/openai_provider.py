"""OpenAI LLM provider implementation.

This module provides integration with OpenAI's API including GPT-4, GPT-3.5-turbo,
and other OpenAI models with proper error handling and cost tracking.
"""

from typing import Any, AsyncIterator, Optional

import structlog
from openai import AsyncOpenAI, OpenAIError

from f1_predict.llm.base import BaseLLMProvider, LLMConfig, LLMResponse
from f1_predict.llm.exceptions import (
    AuthenticationError,
    InvalidResponseError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)

logger = structlog.get_logger(__name__)

# OpenAI pricing per 1K tokens (as of 2024)
OPENAI_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
}


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation.

    Supports GPT-4, GPT-3.5-turbo and other OpenAI models with
    streaming, token counting, and cost estimation.
    """

    def __init__(self, config: LLMConfig, api_key: str):
        """Initialize OpenAI provider.

        Args:
            config: Provider configuration
            api_key: OpenAI API key

        Raises:
            AuthenticationError: If API key is invalid
        """
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=api_key)
        self.logger.info("openai_provider_initialized", model=config.model)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text completion using OpenAI API.

        Args:
            prompt: User prompt text
            system_prompt: Optional system instruction
            **kwargs: Additional OpenAI-specific parameters

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
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Merge config with kwargs
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "frequency_penalty": kwargs.get(
                    "frequency_penalty", self.config.frequency_penalty
                ),
                "presence_penalty": kwargs.get(
                    "presence_penalty", self.config.presence_penalty
                ),
            }

            self.logger.debug("openai_request_started", **params)

            # Make API call
            response = await self.client.chat.completions.create(**params)

            # Extract response data
            content = response.choices[0].message.content
            if not content:
                msg = "Empty response from OpenAI"
                raise InvalidResponseError(msg)

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

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
                    "finish_reason": response.choices[0].finish_reason,
                    "model_used": response.model,
                },
            )

            self.logger.info(
                "openai_response_received",
                tokens=total_tokens,
                cost=estimated_cost,
                finish_reason=response.choices[0].finish_reason,
            )

            return llm_response

        except OpenAIError as e:
            self.logger.error("openai_error", error=str(e), error_type=type(e).__name__)
            # Map OpenAI errors to our custom exceptions
            error_message = str(e)
            if "rate_limit" in error_message.lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}") from e
            elif "authentication" in error_message.lower() or "api_key" in error_message.lower():
                raise AuthenticationError(f"OpenAI authentication failed: {e}") from e
            elif "timeout" in error_message.lower():
                raise TimeoutError(f"OpenAI request timed out: {e}") from e
            elif "server_error" in error_message.lower() or "service_unavailable" in error_message.lower():
                raise ProviderUnavailableError(f"OpenAI service unavailable: {e}") from e
            else:
                raise InvalidResponseError(f"OpenAI API error: {e}") from e

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
            **kwargs: Additional OpenAI-specific parameters

        Yields:
            Content chunks as they are generated

        Raises:
            RateLimitError: When rate limits are exceeded
            AuthenticationError: For authentication failures
            TimeoutError: When request times out
            ProviderUnavailableError: When service is unavailable
        """
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Merge config with kwargs
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "stream": True,
            }

            self.logger.debug("openai_streaming_started", **params)

            # Make streaming API call
            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIError as e:
            self.logger.error(
                "openai_streaming_error", error=str(e), error_type=type(e).__name__
            )
            error_message = str(e)
            if "rate_limit" in error_message.lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}") from e
            elif "authentication" in error_message.lower():
                raise AuthenticationError(f"OpenAI authentication failed: {e}") from e
            elif "timeout" in error_message.lower():
                raise TimeoutError(f"OpenAI request timed out: {e}") from e
            else:
                raise ProviderUnavailableError(f"OpenAI service error: {e}") from e

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate number of tokens
        """
        # Rough approximation: 1 token ≈ 4 characters
        # For production, use tiktoken library for accurate counting
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Get pricing for model (fallback to gpt-3.5-turbo if not found)
        pricing = OPENAI_PRICING.get(
            self.config.model, OPENAI_PRICING["gpt-3.5-turbo"]
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
        return "openai"

    @property
    def available_models(self) -> list[str]:
        """Get list of available OpenAI models.

        Returns:
            List of model identifiers
        """
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
