"""Tests for base LLM provider classes."""

from dataclasses import asdict

import pytest

from f1_predict.llm.base import LLMConfig, LLMResponse


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test creating LLMResponse instance."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            estimated_cost=0.001,
            metadata={"temperature": 0.7},
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.total_tokens == 30
        assert response.estimated_cost == 0.001
        assert response.metadata == {"temperature": 0.7}

    def test_llm_response_to_dict(self):
        """Test converting LLMResponse to dictionary."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            estimated_cost=0.001,
            metadata={},
        )

        data = asdict(response)
        assert data["content"] == "Test"
        assert data["model"] == "gpt-4"
        assert data["provider"] == "openai"


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_llm_config_defaults(self):
        """Test LLMConfig with default values."""
        config = LLMConfig(model="gpt-4")

        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.timeout == 30

    def test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=2000,
            timeout=60,
        )

        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.timeout == 60

    def test_llm_config_validation(self):
        """Test LLMConfig validates temperature range."""
        # Temperature should be between 0 and 2
        config = LLMConfig(model="gpt-4", temperature=1.5)
        assert 0 <= config.temperature <= 2

    def test_llm_config_to_dict(self):
        """Test converting LLMConfig to dictionary."""
        config = LLMConfig(model="gpt-4", temperature=0.8)
        data = asdict(config)

        assert data["model"] == "gpt-4"
        assert data["temperature"] == 0.8
