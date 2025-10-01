"""Tests for PredictionExplainer class."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from f1_predict.analysis.prediction_explainer import PredictionExplainer
from f1_predict.llm.base import LLMConfig, LLMResponse


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = MagicMock()
    provider.name = "test_provider"
    provider.config = LLMConfig(model="test-model")
    provider.generate = AsyncMock()
    return provider


@pytest.fixture
def explainer(mock_llm_provider):
    """Create PredictionExplainer instance."""
    return PredictionExplainer(llm_provider=mock_llm_provider)


@pytest.fixture
def sample_features():
    """Sample feature importance data."""
    return [
        ("qualifying_position", 0.35, "Started from pole"),
        ("driver_form_score", 0.25, "Won last 3 races"),
        ("circuit_performance_score", 0.20, "Strong history at this track"),
        ("team_reliability_score", 0.12, "No DNFs this season"),
        ("championship_position", 0.08, "Leading the championship"),
    ]


class TestPredictionExplainer:
    """Test PredictionExplainer functionality."""

    @pytest.mark.asyncio
    async def test_generate_simple_explanation(self, explainer, sample_features):
        """Test simple explanation generation."""
        result = await explainer.generate(
            driver_name="Max Verstappen",
            position=1,
            confidence=85.0,
            model_name="XGBoost",
            top_features=sample_features,
            detail_level="simple",
        )

        assert result["driver_name"] == "Max Verstappen"
        assert result["predicted_position"] == 1
        assert result["confidence"] == 85.0
        assert result["detail_level"] == "simple"
        assert "explanation" in result
        assert "Verstappen" in result["explanation"]
        assert "win" in result["explanation"]

    @pytest.mark.asyncio
    async def test_generate_detailed_explanation(self, explainer, mock_llm_provider, sample_features):
        """Test detailed explanation with LLM."""
        # Mock LLM response
        mock_llm_provider.generate.return_value = LLMResponse(
            content="Max Verstappen is predicted to win with 85% confidence based on his pole position and recent form.",
            model="test-model",
            provider="test",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost=0.001,
            metadata={},
        )

        result = await explainer.generate(
            driver_name="Max Verstappen",
            position=1,
            confidence=85.0,
            model_name="XGBoost",
            top_features=sample_features,
            circuit="Monaco",
            detail_level="detailed",
        )

        assert result["detail_level"] == "detailed"
        assert "explanation" in result
        assert len(result["explanation"]) > 50
        mock_llm_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_technical_explanation(self, explainer, sample_features):
        """Test technical explanation generation."""
        result = await explainer.generate(
            driver_name="Lewis Hamilton",
            position=2,
            confidence=75.0,
            model_name="RandomForest",
            top_features=sample_features,
            detail_level="technical",
        )

        assert result["detail_level"] == "technical"
        assert "RandomForest" in result["explanation"]
        assert "Feature Importance" in result["explanation"]

    @pytest.mark.asyncio
    async def test_feature_translation(self, explainer, sample_features):
        """Test feature name translation."""
        result = await explainer.generate(
            driver_name="Charles Leclerc",
            position=3,
            confidence=65.0,
            model_name="Ensemble",
            top_features=sample_features,
            detail_level="simple",
        )

        # Check that technical names are translated
        assert "starting grid position" in result["explanation"] or "pole" in result["explanation"].lower()

    @pytest.mark.asyncio
    async def test_confidence_message(self, explainer, sample_features):
        """Test confidence level messaging."""
        # High confidence
        result_high = await explainer.generate(
            driver_name="Max Verstappen",
            position=1,
            confidence=92.0,
            model_name="XGBoost",
            top_features=sample_features,
            detail_level="simple",
        )
        assert result_high["confidence_message"] == "Very strong prediction - high probability"

        # Medium confidence
        result_medium = await explainer.generate(
            driver_name="Sergio Perez",
            position=2,
            confidence=68.0,
            model_name="XGBoost",
            top_features=sample_features,
            detail_level="simple",
        )
        assert result_medium["confidence_message"] == "Moderate prediction - reasonable chance"

        # Low confidence
        result_low = await explainer.generate(
            driver_name="George Russell",
            position=5,
            confidence=45.0,
            model_name="XGBoost",
            top_features=sample_features,
            detail_level="simple",
        )
        assert result_low["confidence_message"] == "Uncertain prediction - many variables at play"

    @pytest.mark.asyncio
    async def test_invalid_driver_name(self, explainer, sample_features):
        """Test error handling for invalid driver name."""
        with pytest.raises(ValueError, match="Driver name is required"):
            await explainer.generate(
                driver_name="",
                position=1,
                confidence=85.0,
                model_name="XGBoost",
                top_features=sample_features,
            )

    @pytest.mark.asyncio
    async def test_invalid_position(self, explainer, sample_features):
        """Test error handling for invalid position."""
        with pytest.raises(ValueError, match="Position must be between 1 and 20"):
            await explainer.generate(
                driver_name="Max Verstappen",
                position=25,
                confidence=85.0,
                model_name="XGBoost",
                top_features=sample_features,
            )

    @pytest.mark.asyncio
    async def test_invalid_confidence(self, explainer, sample_features):
        """Test error handling for invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 100"):
            await explainer.generate(
                driver_name="Max Verstappen",
                position=1,
                confidence=150.0,
                model_name="XGBoost",
                top_features=sample_features,
            )

    @pytest.mark.asyncio
    async def test_metadata_generation(self, explainer, sample_features):
        """Test metadata is added to output."""
        result = await explainer.generate(
            driver_name="Max Verstappen",
            position=1,
            confidence=85.0,
            model_name="XGBoost",
            top_features=sample_features,
            detail_level="simple",
        )

        assert "generated_at" in result
        assert "generator" in result
        assert "llm_provider" in result
        assert "llm_model" in result
        assert result["generator"] == "PredictionExplainer"

    @pytest.mark.asyncio
    async def test_formatted_features(self, explainer, sample_features):
        """Test feature formatting in output."""
        result = await explainer.generate(
            driver_name="Max Verstappen",
            position=1,
            confidence=85.0,
            model_name="XGBoost",
            top_features=sample_features,
            detail_level="simple",
        )

        assert "top_features" in result
        assert len(result["top_features"]) == len(sample_features)
        assert all("feature" in f for f in result["top_features"])
        assert all("importance" in f for f in result["top_features"])
        assert all("explanation" in f for f in result["top_features"])

    @pytest.mark.asyncio
    async def test_llm_fallback(self, explainer, mock_llm_provider, sample_features):
        """Test fallback to simple explanation if LLM fails."""
        # Make LLM raise an exception
        mock_llm_provider.generate.side_effect = Exception("LLM error")

        result = await explainer.generate(
            driver_name="Max Verstappen",
            position=1,
            confidence=85.0,
            model_name="XGBoost",
            top_features=sample_features,
            detail_level="detailed",  # Request detailed but should fallback
        )

        assert "explanation" in result
        assert len(result["explanation"]) > 0  # Should have fallback explanation
