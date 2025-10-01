"""Custom exceptions for LLM module.

This module defines exception hierarchy for LLM-related errors including
API failures, rate limiting, authentication issues, and provider unavailability.
"""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    pass


class RateLimitError(LLMError):
    """Exception raised when API rate limits are exceeded (429 errors)."""

    pass


class AuthenticationError(LLMError):
    """Exception raised for authentication failures (401/403 errors)."""

    pass


class QuotaExceededError(LLMError):
    """Exception raised when API quota is exhausted."""

    pass


class TimeoutError(LLMError):
    """Exception raised when API request times out."""

    pass


class InvalidResponseError(LLMError):
    """Exception raised when API returns malformed or invalid response."""

    pass


class ProviderUnavailableError(LLMError):
    """Exception raised when LLM provider service is unavailable."""

    pass


class TemplateError(LLMError):
    """Exception raised for prompt template errors."""

    pass


class BudgetExceededError(LLMError):
    """Exception raised when budget limits are exceeded."""

    pass
