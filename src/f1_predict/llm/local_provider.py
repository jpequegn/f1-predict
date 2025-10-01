"""Local LLM provider implementation using Ollama.

This module provides integration with locally hosted models via Ollama,
enabling free, private LLM usage without API costs.
"""

from typing import Any, AsyncIterator, Optional

import httpx
import structlog

from f1_predict.llm.base import BaseLLMProvider, LLMConfig, LLMResponse
from f1_predict.llm.exceptions import ProviderUnavailableError, TimeoutError

logger = structlog.get_logger(__name__)


class LocalProvider(BaseLLMProvider):
    """Local LLM provider using Ollama.

    Supports local models like Llama 3.1, Mistral, Mixtral with
    no API costs and full privacy.
    """

    def __init__(self, config: LLMConfig, endpoint: str = "http://localhost:11434"):
        """Initialize local provider.

        Args:
            config: Provider configuration
            endpoint: Ollama API endpoint

        Raises:
            ProviderUnavailableError: If Ollama is not running
        """
        super().__init__(config)
        self.endpoint = endpoint
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        self.logger.info("local_provider_initialized", model=config.model)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text completion using local Ollama model.

        Args:
            prompt: User prompt text
            system_prompt: Optional system instruction
            **kwargs: Additional model-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            TimeoutError: When request times out
            ProviderUnavailableError: When Ollama is unavailable
        """
        try:
            # Build prompt with system message if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            payload = {
                "model": self.config.model,
                "prompt": full_prompt,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "stream": False,
            }

            self.logger.debug("local_request_started", **payload)

            # Make API call to Ollama
            response = await self.client.post(
                f"{self.endpoint}/api/generate", json=payload
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("response", "")

            # Local models are free, so cost is 0
            input_tokens = self.count_tokens(full_prompt)
            output_tokens = self.count_tokens(content)
            total_tokens = input_tokens + output_tokens

            llm_response = LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=0.0,  # Local models are free
                metadata={
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                },
            )

            self.logger.info(
                "local_response_received",
                tokens=total_tokens,
                eval_count=data.get("eval_count"),
            )

            return llm_response

        except httpx.TimeoutException as e:
            msg = f"Local model request timed out: {e}"
            raise TimeoutError(msg) from e
        except httpx.HTTPError as e:
            msg = f"Ollama service unavailable: {e}"
            raise ProviderUnavailableError(msg) from e

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
            **kwargs: Additional model-specific parameters

        Yields:
            Content chunks as they are generated

        Raises:
            TimeoutError: When request times out
            ProviderUnavailableError: When Ollama is unavailable
        """
        try:
            # Build prompt with system message if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            payload = {
                "model": self.config.model,
                "prompt": full_prompt,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "stream": True,
            }

            self.logger.debug("local_streaming_started", **payload)

            # Make streaming API call
            async with self.client.stream(
                "POST", f"{self.endpoint}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json

                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

        except httpx.TimeoutException as e:
            msg = f"Local model streaming timed out: {e}"
            raise TimeoutError(msg) from e
        except httpx.HTTPError as e:
            msg = f"Ollama service unavailable during streaming: {e}"
            raise ProviderUnavailableError(msg) from e

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate number of tokens
        """
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD (always 0 for local models)
        """
        return 0.0  # Local models are free

    @property
    def name(self) -> str:
        """Get provider name.

        Returns:
            Provider name identifier
        """
        return "local"

    @property
    def available_models(self) -> list[str]:
        """Get list of available local models.

        Returns:
            List of model identifiers (common Ollama models)
        """
        return [
            "llama3.1",
            "llama3.1:70b",
            "mistral",
            "mixtral",
            "gemma2",
        ]

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
