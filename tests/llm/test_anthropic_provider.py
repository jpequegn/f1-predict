"""Tests for Anthropic provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import APIError, AuthenticationError as AnthropicAuthError, RateLimitError as AnthropicRateLimitError

from f1_predict.llm.anthropic_provider import ANTHROPIC_PRICING, AnthropicProvider
from f1_predict.llm.base import LLMConfig
from f1_predict.llm.exceptions import AuthenticationError, InvalidResponseError, RateLimitError


@pytest.fixture
def config():
    """Create test configuration."""
    return LLMConfig(model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=500)


@pytest.fixture
def provider(config):
    """Create Anthropic provider instance."""
    return AnthropicProvider(config=config, api_key="test-api-key")


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider.name == "anthropic"
        assert provider.config.model == "claude-3-5-sonnet-20241022"
        assert provider.client is not None

    def test_available_models(self, provider):
        """Test available models list."""
        models = provider.available_models
        assert "claude-3-5-sonnet-20241022" in models
        assert "claude-3-opus-20240229" in models
        assert "claude-3-haiku-20240307" in models

    def test_count_tokens_approximation(self, provider):
        """Test token counting approximation."""
        text = "Hello, world! This is a test."
        tokens = provider.count_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4

    def test_estimate_cost_sonnet(self, provider):
        """Test cost estimation for Claude Sonnet."""
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)

        expected = (1000 / 1000 * ANTHROPIC_PRICING["claude-3-5-sonnet-20241022"]["input"] +
                   500 / 1000 * ANTHROPIC_PRICING["claude-3-5-sonnet-20241022"]["output"])
        assert cost == expected

    def test_estimate_cost_opus(self, provider):
        """Test cost estimation for Claude Opus."""
        provider.config.model = "claude-3-opus-20240229"
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)

        expected = (1000 / 1000 * ANTHROPIC_PRICING["claude-3-opus-20240229"]["input"] +
                   500 / 1000 * ANTHROPIC_PRICING["claude-3-opus-20240229"]["output"])
        assert cost == expected

    def test_estimate_cost_haiku(self, provider):
        """Test cost estimation for Claude Haiku."""
        provider.config.model = "claude-3-haiku-20240307"
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)

        expected = (1000 / 1000 * ANTHROPIC_PRICING["claude-3-haiku-20240307"]["input"] +
                   500 / 1000 * ANTHROPIC_PRICING["claude-3-haiku-20240307"]["output"])
        assert cost == expected

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Generated response"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        with patch.object(provider.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            response = await provider.generate(
                prompt="Test prompt",
                system_prompt="Test system"
            )

            assert response.content == "Generated response"
            assert response.model == "claude-3-5-sonnet-20241022"
            assert response.provider == "anthropic"
            assert response.input_tokens == 10
            assert response.output_tokens == 20
            assert response.total_tokens == 30
            assert response.estimated_cost > 0

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, provider):
        """Test generation with custom kwargs."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Response"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        with patch.object(provider.client.messages, "create", new_callable=AsyncMock) as mock_create:
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
        with patch.object(provider.client.messages, "create", new_callable=AsyncMock) as mock_create:
            # Create proper APIStatusError with required arguments
            error = AnthropicRateLimitError("Rate limit exceeded", response=MagicMock(), body={})
            mock_create.side_effect = error

            with pytest.raises(RateLimitError, match="rate limit"):
                await provider.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_auth_error(self, provider):
        """Test authentication error handling."""
        with patch.object(provider.client.messages, "create", new_callable=AsyncMock) as mock_create:
            # Create proper APIStatusError with required arguments
            error = AnthropicAuthError("Invalid API key", response=MagicMock(), body={})
            mock_create.side_effect = error

            with pytest.raises(AuthenticationError, match="authentication"):
                await provider.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_invalid_response(self, provider):
        """Test invalid response handling."""
        mock_response = MagicMock()
        # Simulate response with empty content
        mock_content_obj = MagicMock()
        mock_content_obj.text = ""  # Empty text
        mock_response.content = [mock_content_obj]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 0

        with patch.object(provider.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            with pytest.raises(InvalidResponseError, match="Empty response"):
                await provider.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_streaming(self, provider):
        """Test streaming generation."""
        async def mock_stream():
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                mock_event = MagicMock()
                mock_event.type = "content_block_delta"
                mock_event.delta.text = chunk
                yield mock_event

        with patch.object(provider.client.messages, "stream", new_callable=AsyncMock) as mock_stream_method:
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_stream())
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_stream_method.return_value = mock_context

            collected = []
            async for chunk in provider.generate_streaming(prompt="Test"):
                collected.append(chunk)

            assert "".join(collected) == "Hello world!"

    @pytest.mark.asyncio
    async def test_generate_no_system_prompt(self, provider):
        """Test generation without system prompt."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Response"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 10

        with patch.object(provider.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await provider.generate(prompt="Test prompt")

            # Verify system parameter was not included
            call_kwargs = mock_create.call_args.kwargs
            assert "system" not in call_kwargs or call_kwargs.get("system") == ""

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, provider):
        """Test generation with system prompt."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Response"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        with patch.object(provider.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await provider.generate(
                prompt="Test prompt",
                system_prompt="You are a helpful assistant"
            )

            # Verify system prompt was passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["system"] == "You are a helpful assistant"
