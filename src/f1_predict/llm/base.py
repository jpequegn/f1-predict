"""Base abstract class for LLM providers.

This module defines the interface that all LLM providers must implement,
ensuring consistent behavior across different providers (OpenAI, Anthropic, local models).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM provider.

    Attributes:
        content: Generated text content
        model: Model used for generation
        provider: Provider name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens used
        estimated_cost: Estimated cost in USD
        metadata: Additional provider-specific metadata
    """

    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    metadata: dict[str, Any]


@dataclass
class LLMConfig:
    """Configuration for LLM provider.

    Attributes:
        model: Model name/identifier
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
    """

    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM provider implementations must inherit from this class
    and implement the required abstract methods.
    """

    def __init__(self, config: LLMConfig):
        """Initialize LLM provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text completion from prompt.

        Args:
            prompt: User prompt text
            system_prompt: Optional system instruction
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            LLMError: For provider-specific errors
        """
        pass

    @abstractmethod
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
            **kwargs: Additional provider-specific parameters

        Yields:
            Content chunks as they are generated

        Raises:
            LLMError: For provider-specific errors
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get provider name.

        Returns:
            Provider name identifier
        """
        pass

    @property
    @abstractmethod
    def available_models(self) -> list[str]:
        """Get list of available models.

        Returns:
            List of model identifiers
        """
        pass

    def __str__(self) -> str:
        """String representation of provider.

        Returns:
            Provider name and model
        """
        return f"{self.name} ({self.config.model})"

    def __repr__(self) -> str:
        """Detailed string representation.

        Returns:
            Full provider information
        """
        return f"<{self.__class__.__name__} model={self.config.model}>"
