"""Tests for local Ollama provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from f1_predict.llm.base import LLMConfig
from f1_predict.llm.exceptions import InvalidResponseError, ProviderUnavailableError, TimeoutError
from f1_predict.llm.local_provider import LocalProvider


@pytest.fixture
def config():
    """Create test configuration."""
    return LLMConfig(model="llama3.1", temperature=0.7, max_tokens=500)


@pytest.fixture
def provider(config):
    """Create local provider instance."""
    return LocalProvider(config=config, endpoint="http://localhost:11434")


class TestLocalProvider:
    """Test local Ollama provider implementation."""

    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider.name == "local"
        assert provider.config.model == "llama3.1"
        assert provider.endpoint == "http://localhost:11434"
        assert provider.client is not None

    def test_available_models(self, provider):
        """Test available models list."""
        models = provider.available_models
        assert "llama3.1" in models
        assert "mistral" in models
        assert "mixtral" in models

    def test_count_tokens_approximation(self, provider):
        """Test token counting approximation."""
        text = "Hello, world! This is a test."
        tokens = provider.count_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4

    def test_estimate_cost_is_zero(self, provider):
        """Test cost estimation for local models is zero."""
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Generated response from Ollama",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            response = await provider.generate(
                prompt="Test prompt",
                system_prompt="Test system"
            )

            assert response.content == "Generated response from Ollama"
            assert response.model == "llama3.1"
            assert response.provider == "local"
            assert response.input_tokens == 10
            assert response.output_tokens == 20
            assert response.total_tokens == 30
            assert response.estimated_cost == 0.0

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, provider):
        """Test generation with custom kwargs."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Response",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await provider.generate(
                prompt="Test",
                temperature=0.9,
                max_tokens=1000
            )

            # Verify custom parameters were passed in options
            call_kwargs = mock_post.call_args.kwargs
            json_data = call_kwargs["json"]
            assert json_data["options"]["temperature"] == 0.9
            assert json_data["options"]["num_predict"] == 1000

    @pytest.mark.asyncio
    async def test_generate_timeout_error(self, provider):
        """Test timeout error handling."""
        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timeout")

            with pytest.raises(TimeoutError, match="timeout"):
                await provider.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, provider):
        """Test connection error handling."""
        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(ProviderUnavailableError, match="unavailable"):
                await provider.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_invalid_response(self, provider):
        """Test invalid response handling."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}  # Missing 'response' key

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(InvalidResponseError, match="Invalid response"):
                await provider.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_generate_streaming(self, provider):
        """Test streaming generation."""
        async def mock_stream():
            responses = [
                b'{"response": "Hello"}\n',
                b'{"response": " world"}\n',
                b'{"response": "!"}\n',
            ]
            for resp in responses:
                yield resp

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_stream

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            collected = []
            async for chunk in provider.generate_streaming(prompt="Test"):
                collected.append(chunk)

            assert "".join(collected) == "Hello world!"

    @pytest.mark.asyncio
    async def test_generate_no_system_prompt(self, provider):
        """Test generation without system prompt."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Response",
            "prompt_eval_count": 5,
            "eval_count": 10,
        }

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await provider.generate(prompt="Test prompt")

            # Verify request structure
            call_kwargs = mock_post.call_args.kwargs
            json_data = call_kwargs["json"]
            assert json_data["model"] == "llama3.1"
            assert json_data["prompt"] == "Test prompt"
            assert json_data["system"] == ""

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, provider):
        """Test generation with system prompt."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Response",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await provider.generate(
                prompt="Test prompt",
                system_prompt="You are a helpful assistant"
            )

            # Verify system prompt was included
            call_kwargs = mock_post.call_args.kwargs
            json_data = call_kwargs["json"]
            assert json_data["system"] == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_generate_missing_token_counts(self, provider):
        """Test handling of missing token counts in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Response without token counts"
        }

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            response = await provider.generate(prompt="Test")

            # Should use approximation for missing tokens
            assert response.input_tokens > 0
            assert response.output_tokens > 0
            assert response.total_tokens == response.input_tokens + response.output_tokens

    def test_custom_endpoint(self, config):
        """Test provider with custom endpoint."""
        provider = LocalProvider(config=config, endpoint="http://custom:8080")
        assert provider.endpoint == "http://custom:8080"
