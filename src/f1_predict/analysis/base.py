"""Base analyzer class for all analysis generators."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import structlog

from f1_predict.llm.base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class BaseAnalyzer(ABC):
    """Base class for all analysis generators.

    Provides common functionality for LLM-powered analysis generation including
    template management, validation, and output formatting.

    Attributes:
        llm_provider: LLM provider for text generation
        logger: Structured logger instance
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        """Initialize base analyzer.

        Args:
            llm_provider: LLM provider instance for text generation
        """
        self.llm_provider = llm_provider
        self.logger = logger.bind(analyzer=self.__class__.__name__)

    @abstractmethod
    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        """Generate analysis content.

        Args:
            **kwargs: Analysis-specific parameters

        Returns:
            Dictionary containing generated analysis content and metadata

        Raises:
            ValueError: If required parameters are missing
            LLMError: If LLM generation fails
        """
        pass

    def _add_metadata(self, content: dict[str, Any]) -> dict[str, Any]:
        """Add standard metadata to generated content.

        Args:
            content: Generated content dictionary

        Returns:
            Content with added metadata
        """
        content["generated_at"] = datetime.now().isoformat()
        content["generator"] = self.__class__.__name__
        content["llm_provider"] = self.llm_provider.name
        content["llm_model"] = self.llm_provider.config.model

        if "content" in content and isinstance(content["content"], str):
            content["word_count"] = len(content["content"].split())
            content["estimated_read_time"] = (
                f"{max(1, content['word_count'] // 200)} minutes"
            )

        return content

    def _validate_output(
        self, content: dict[str, Any], required_fields: list[str]
    ) -> bool:
        """Validate generated content has required fields.

        Args:
            content: Generated content to validate
            required_fields: List of required field names

        Returns:
            True if all required fields present, False otherwise
        """
        missing = [field for field in required_fields if field not in content]
        if missing:
            self.logger.warning("output_validation_failed", missing_fields=missing)
            return False
        return True

    def _calculate_readability(self, text: str) -> dict[str, float]:
        """Calculate readability metrics for generated text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with readability metrics
        """
        words = text.split()
        sentences = text.count(".") + text.count("!") + text.count("?")
        sentences = max(1, sentences)  # Avoid division by zero

        avg_words_per_sentence = len(words) / sentences

        # Simple readability approximation
        # Real implementation would use textstat or similar
        readability_score = 100 - (avg_words_per_sentence * 1.5)

        return {
            "words": len(words),
            "sentences": sentences,
            "avg_words_per_sentence": avg_words_per_sentence,
            "readability_score": max(0, min(100, readability_score)),
        }
