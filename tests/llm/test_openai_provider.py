"""Tests for OpenAI provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIError, AuthenticationError as OpenAIAuthError, RateLimitError as OpenAIRateLimitError

from f1_predict.llm.base import LLMConfig
from f1_predict.llm.exceptions import AuthenticationError, InvalidResponseError, RateLimitError
from f1_predict.llm.openai_provider import OPENAI_PRICING, OpenAIProvider


@pytest.fixture
def config():
    """Create test configuration."""
    return LLMConfig(model="gpt-3.5-turbo", temperature=0.7, max_tokens=500)


@pytest.fixture
def provider(config):
    """Create OpenAI provider instance."""
    return OpenAIProvider(config=config, api_key="test-api-key")


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider.name == "openai"
        assert provider.config.model == "gpt-3.5-turbo"
        assert provider.client is not None

    def test_available_models(self, provider):
        """Test available models list."""
        models = provider.available_models
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models
        assert "gpt-4-turbo" in models

    def test_count_tokens_approximation(self, provider):
        """Test token counting approximation."""
        text = "Hello, world! This is a test."
        tokens = provider.count_tokens(text)
        # Rough approximation: 1 token ≈ 4 characters
        assert tokens > 0
        assert tokens == len(text) // 4

    def test_estimate_cost_gpt4(self, provider):
        """Test cost estimation for GPT-4."""
        provider.config.model = "gpt-4"
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)

        expected = (1000 / 1000 * OPENAI_PRICING["gpt-4"]["input"] +
                   500 / 1000 * OPENAI_PRICING["gpt-4"]["output"])
        assert cost == expected

    def test_estimate_cost_gpt35(self, provider):
        """Test cost estimation for GPT-3.5."""
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)

        expected = (1000 / 1000 * OPENAI_PRICING["gpt-3.5-turbo"]["input"] +
                   500 / 1000 * OPENAI_PRICING["gpt-3.5-turbo"]["output"])
        assert cost == expected

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            response = await provider.generate(
                prompt="Test prompt",
                system_prompt="Test system"
            )

            assert response.content == "Generated response"
            assert response.model == "gpt-3.5-turbo"
            assert response.provider == "openai"
            assert response.input_tokens == 10
            assert response.output_tokens == 20
            assert response.total_tokens == 30
            assert response.estimated_cost > 0

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, provider):
        """Test generation with custom kwargs."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await provider.generate(
                prompt="Test",
                temperature=0.9,
                max_tokens=1000
            )

            # Verify custom parameters were passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.9
            assert call_kwargs["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_generate_rate_limit_error(self, provider):
        """Test rate limit error handling."""
        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = OpenAIRateLimitError("Rate limit exceeded", response=MagicMock(), body={})

            with pytest.raises(RateLimitError, match="rate limit"):
                await provider.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_auth_error(self, provider):
        """Test authentication error handling."""
        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = OpenAIAuthError("Invalid API key", response=MagicMock(), body={})

            with pytest.raises(AuthenticationError, match="authentication"):
                await provider.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_invalid_response(self, provider):
        """Test invalid response handling."""
        mock_response = MagicMock()
        mock_response.choices = []  # Empty choices

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            with pytest.raises(InvalidResponseError, match="No response choices"):
                await provider.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_streaming(self, provider):
        """Test streaming generation."""
        async def mock_stream():
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                mock_chunk = MagicMock()
                mock_chunk.choices = [MagicMock()]
                mock_chunk.choices[0].delta.content = chunk
                yield mock_chunk

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream()

            collected = []
            async for chunk in provider.generate_streaming(prompt="Test"):
                collected.append(chunk)

            assert "".join(collected) == "Hello world!"

    @pytest.mark.asyncio
    async def test_generate_no_system_prompt(self, provider):
        """Test generation without system prompt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await provider.generate(prompt="Test prompt")

            # Verify messages structure
            call_kwargs = mock_create.call_args.kwargs
            assert len(call_kwargs["messages"]) == 1
            assert call_kwargs["messages"][0]["role"] == "user"
