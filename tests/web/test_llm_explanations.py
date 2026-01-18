"""Tests for web LLM explanation utilities."""

import pytest
from unittest.mock import MagicMock, patch

from f1_predict.web.utils.llm_explanations import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    _format_simple_comparison,
    _format_simple_explanation,
    _format_simple_post_race,
    _format_simple_preview,
    _get_provider,
    _run_async,
    check_llm_availability,
    generate_driver_comparison,
    generate_post_race_analysis,
    generate_prediction_explanation,
    generate_race_preview,
)


class TestDefaults:
    """Test default configuration."""

    def test_default_model(self):
        """Test default model is set."""
        assert DEFAULT_MODEL == "claude-3-haiku-20240307"

    def test_default_provider(self):
        """Test default provider is set."""
        assert DEFAULT_PROVIDER == "anthropic"


class TestGetProvider:
    """Test provider initialization."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("f1_predict.llm.anthropic_provider.AnthropicProvider")
    def test_get_anthropic_provider(self, mock_provider):
        """Test getting Anthropic provider."""
        mock_provider.return_value = MagicMock()
        provider = _get_provider("anthropic")
        assert provider is not None
        mock_provider.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    def test_get_anthropic_no_key(self):
        """Test Anthropic provider without API key."""
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        provider = _get_provider("anthropic")
        assert provider is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("f1_predict.llm.openai_provider.OpenAIProvider")
    def test_get_openai_provider(self, mock_provider):
        """Test getting OpenAI provider."""
        mock_provider.return_value = MagicMock()
        provider = _get_provider("openai")
        assert provider is not None

    @patch("f1_predict.llm.local_provider.LocalProvider")
    def test_get_local_provider(self, mock_provider):
        """Test getting local provider."""
        mock_provider.return_value = MagicMock()
        provider = _get_provider("local")
        assert provider is not None

    def test_get_unknown_provider(self):
        """Test getting unknown provider."""
        provider = _get_provider("unknown")
        assert provider is None

    @patch("f1_predict.llm.local_provider.LocalProvider")
    def test_get_provider_with_model_override(self, mock_provider):
        """Test getting provider with model override."""
        mock_provider.return_value = MagicMock()
        _get_provider("local", model="custom-model")
        call_args = mock_provider.call_args
        assert call_args[0][0].model == "custom-model"


class TestRunAsync:
    """Test async runner utility."""

    def test_run_async_simple(self):
        """Test running simple async coroutine."""
        async def simple_coro():
            return "result"

        result = _run_async(simple_coro())
        assert result == "result"


class TestFormatSimpleExplanation:
    """Test simple explanation formatting."""

    def test_empty_podium(self):
        """Test with empty podium."""
        result = _format_simple_explanation([])
        assert result == "No prediction available."

    def test_single_driver(self):
        """Test with single driver."""
        podium = [{"driver": "Verstappen", "confidence": 0.85, "position": 1}]
        result = _format_simple_explanation(podium)
        assert "Verstappen" in result
        assert "85%" in result
        assert "Position 1" in result

    def test_full_podium(self):
        """Test with full podium."""
        podium = [
            {"driver": "Verstappen", "confidence": 0.85, "position": 1},
            {"driver": "Norris", "confidence": 0.75, "position": 2},
            {"driver": "Leclerc", "confidence": 0.65, "position": 3},
        ]
        result = _format_simple_explanation(podium)
        assert "Verstappen" in result
        assert "Norris" in result
        assert "Leclerc" in result


class TestFormatSimpleComparison:
    """Test simple comparison formatting."""

    def test_basic_comparison(self):
        """Test basic comparison without stats."""
        result = _format_simple_comparison("Verstappen", "Hamilton", None, None)
        assert "Verstappen vs Hamilton" in result
        assert "statistics not available" in result

    def test_comparison_with_wins(self):
        """Test comparison with wins data."""
        stats1 = {"wins": 5}
        stats2 = {"wins": 3}
        result = _format_simple_comparison("Verstappen", "Hamilton", stats1, stats2)
        assert "Wins" in result
        assert "5" in result
        assert "3" in result

    def test_comparison_with_points(self):
        """Test comparison with points data."""
        stats1 = {"points": 300}
        stats2 = {"points": 250}
        result = _format_simple_comparison("Verstappen", "Hamilton", stats1, stats2)
        assert "Points" in result
        assert "300" in result
        assert "250" in result


class TestFormatSimplePreview:
    """Test simple preview formatting."""

    def test_basic_preview(self):
        """Test basic preview."""
        result = _format_simple_preview("Monaco GP", "Monte Carlo", ["Verstappen", "Norris"])
        assert "Monaco GP" in result
        assert "Monte Carlo" in result
        assert "Verstappen" in result

    def test_preview_empty_drivers(self):
        """Test preview with empty drivers list."""
        result = _format_simple_preview("Monaco GP", "Monte Carlo", [])
        assert "Monaco GP" in result
        assert "No drivers available" in result


class TestFormatSimplePostRace:
    """Test simple post-race formatting."""

    def test_empty_results(self):
        """Test with empty results."""
        result = _format_simple_post_race("Monaco GP", [])
        assert "Monaco GP" in result
        assert "Results not available" in result

    def test_with_results(self):
        """Test with race results."""
        results = [
            {"driver": "Verstappen"},
            {"driver": "Norris"},
            {"driver": "Leclerc"},
        ]
        result = _format_simple_post_race("Monaco GP", results)
        assert "Monaco GP" in result
        assert "Verstappen" in result
        assert "Winner" in result


class TestGeneratePredictionExplanation:
    """Test prediction explanation generation."""

    def test_fallback_without_provider(self):
        """Test fallback when no provider available."""
        prediction = {
            "podium": [
                {"driver": "Verstappen", "confidence": 0.85, "position": 1}
            ],
            "race_name": "Monaco GP",
            "overall_confidence": 0.8,
        }
        # Without mocking provider, it should use fallback
        with patch("f1_predict.web.utils.llm_explanations._get_provider", return_value=None):
            result = generate_prediction_explanation(prediction)
            assert "Verstappen" in result

    @patch("f1_predict.web.utils.llm_explanations._get_provider")
    @patch("f1_predict.web.utils.llm_explanations._run_async")
    def test_with_llm_provider(self, mock_run_async, mock_get_provider):
        """Test with LLM provider."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_async.return_value = "LLM explanation"

        prediction = {
            "podium": [
                {"driver": "Verstappen", "confidence": 0.85, "position": 1}
            ],
            "race_name": "Monaco GP",
            "overall_confidence": 0.8,
        }
        result = generate_prediction_explanation(prediction)
        assert result == "LLM explanation"


class TestGenerateDriverComparison:
    """Test driver comparison generation."""

    def test_fallback_without_provider(self):
        """Test fallback when no provider available."""
        with patch("f1_predict.web.utils.llm_explanations._get_provider", return_value=None):
            result = generate_driver_comparison("Verstappen", "Hamilton")
            assert "Verstappen vs Hamilton" in result

    @patch("f1_predict.web.utils.llm_explanations._get_provider")
    @patch("f1_predict.web.utils.llm_explanations._run_async")
    def test_with_llm_provider(self, mock_run_async, mock_get_provider):
        """Test with LLM provider."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_response = MagicMock()
        mock_response.content = "LLM comparison"
        mock_run_async.return_value = mock_response

        result = generate_driver_comparison("Verstappen", "Hamilton")
        assert result == "LLM comparison"


class TestGenerateRacePreview:
    """Test race preview generation."""

    def test_fallback_without_provider(self):
        """Test fallback when no provider available."""
        with patch("f1_predict.web.utils.llm_explanations._get_provider", return_value=None):
            result = generate_race_preview(
                "Monaco GP",
                "Monte Carlo",
                ["Verstappen", "Norris", "Leclerc"],
            )
            assert "Monaco GP" in result
            assert "Monte Carlo" in result

    @patch("f1_predict.web.utils.llm_explanations._get_provider")
    @patch("f1_predict.web.utils.llm_explanations._run_async")
    def test_with_llm_provider(self, mock_run_async, mock_get_provider):
        """Test with LLM provider."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_response = MagicMock()
        mock_response.content = "LLM preview"
        mock_run_async.return_value = mock_response

        result = generate_race_preview(
            "Monaco GP",
            "Monte Carlo",
            ["Verstappen", "Norris"],
        )
        assert result == "LLM preview"


class TestGeneratePostRaceAnalysis:
    """Test post-race analysis generation."""

    def test_fallback_without_provider(self):
        """Test fallback when no provider available."""
        results = [
            {"driver": "Verstappen", "position": 1, "gap": "Winner"},
            {"driver": "Norris", "position": 2, "gap": "+5.2s"},
        ]
        with patch("f1_predict.web.utils.llm_explanations._get_provider", return_value=None):
            result = generate_post_race_analysis("Monaco GP", results)
            assert "Monaco GP" in result
            assert "Verstappen" in result

    @patch("f1_predict.web.utils.llm_explanations._get_provider")
    @patch("f1_predict.web.utils.llm_explanations._run_async")
    def test_with_llm_provider(self, mock_run_async, mock_get_provider):
        """Test with LLM provider."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_response = MagicMock()
        mock_response.content = "LLM analysis"
        mock_run_async.return_value = mock_response

        results = [
            {"driver": "Verstappen", "position": 1, "gap": "Winner"},
        ]
        result = generate_post_race_analysis("Monaco GP", results)
        assert result == "LLM analysis"


class TestCheckLLMAvailability:
    """Test LLM availability checker."""

    @patch("f1_predict.web.utils.llm_explanations._get_provider")
    def test_provider_available(self, mock_get_provider):
        """Test when provider is available."""
        mock_provider = MagicMock()
        mock_provider.config.model = "test-model"
        mock_get_provider.return_value = mock_provider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            result = check_llm_availability("anthropic")
            assert result["available"] is True
            assert result["api_key_set"] is True
            assert result["model"] == "test-model"

    @patch("f1_predict.web.utils.llm_explanations._get_provider")
    def test_provider_unavailable(self, mock_get_provider):
        """Test when provider is unavailable."""
        mock_get_provider.return_value = None

        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = check_llm_availability("anthropic")
            assert result["available"] is False
            assert result["api_key_set"] is False
            assert result["error"] is not None

    def test_local_provider_no_key_needed(self):
        """Test local provider doesn't need API key."""
        with patch("f1_predict.web.utils.llm_explanations._get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.config.model = "llama3.2"
            mock_get.return_value = mock_provider

            result = check_llm_availability("local")
            assert result["api_key_set"] is True  # Always true for local


class TestErrorHandling:
    """Test error handling in explanation functions."""

    @patch("f1_predict.web.utils.llm_explanations._get_provider")
    @patch("f1_predict.web.utils.llm_explanations._run_async")
    def test_rate_limit_error_fallback(self, mock_run_async, mock_get_provider):
        """Test fallback on rate limit error."""
        from f1_predict.llm.exceptions import RateLimitError

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_async.side_effect = RateLimitError("Rate limited")

        prediction = {
            "podium": [{"driver": "Verstappen", "confidence": 0.85, "position": 1}],
            "race_name": "Monaco GP",
        }
        result = generate_prediction_explanation(prediction)
        # Should fallback to simple format
        assert "Verstappen" in result

    @patch("f1_predict.web.utils.llm_explanations._get_provider")
    @patch("f1_predict.web.utils.llm_explanations._run_async")
    def test_auth_error_fallback(self, mock_run_async, mock_get_provider):
        """Test fallback on authentication error."""
        from f1_predict.llm.exceptions import AuthenticationError

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_async.side_effect = AuthenticationError("Auth failed")

        result = generate_driver_comparison("Verstappen", "Hamilton")
        # Should fallback to simple format
        assert "Verstappen vs Hamilton" in result

    @patch("f1_predict.web.utils.llm_explanations._get_provider")
    @patch("f1_predict.web.utils.llm_explanations._run_async")
    def test_generic_exception_fallback(self, mock_run_async, mock_get_provider):
        """Test fallback on generic exception."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_run_async.side_effect = Exception("Something went wrong")

        results = [{"driver": "Verstappen", "position": 1, "gap": "Winner"}]
        result = generate_post_race_analysis("Monaco GP", results)
        # Should fallback to simple format
        assert "Monaco GP" in result
