"""LLM integration module for F1 prediction system.

This module provides a unified interface for working with multiple LLM providers
(OpenAI, Anthropic, local models) with error handling, cost tracking, and
prompt template management.
"""

from f1_predict.llm.anthropic_provider import AnthropicProvider
from f1_predict.llm.base import BaseLLMProvider, LLMConfig, LLMResponse
from f1_predict.llm.cost_tracker import CostTracker, UsageRecord
from f1_predict.llm.exceptions import (
    AuthenticationError,
    BudgetExceededError,
    InvalidResponseError,
    LLMError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    TemplateError,
    TimeoutError,
)
from f1_predict.llm.local_provider import LocalProvider
from f1_predict.llm.openai_provider import OpenAIProvider
from f1_predict.llm.templates import PromptTemplateManager, get_default_template

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMConfig",
    "LLMResponse",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "LocalProvider",
    # Templates
    "PromptTemplateManager",
    "get_default_template",
    # Cost tracking
    "CostTracker",
    "UsageRecord",
    # Exceptions
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "QuotaExceededError",
    "TimeoutError",
    "InvalidResponseError",
    "ProviderUnavailableError",
    "TemplateError",
    "BudgetExceededError",
]
