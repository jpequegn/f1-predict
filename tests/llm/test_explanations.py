"""Tests for natural language explanation generation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from f1_predict.llm.base import LLMConfig, LLMResponse
from f1_predict.llm.chat_session import ChatSession
from f1_predict.llm.explanations import F1PredictionExplainer


@pytest.fixture
def mock_provider():
    """Create mock LLM provider."""
    provider = MagicMock()
    provider.name = "test-provider"
    provider.config = LLMConfig(model="test-model")
    return provider


@pytest.fixture
def explainer(mock_provider):
    """Create F1PredictionExplainer instance."""
    return F1PredictionExplainer(provider=mock_provider)


class TestF1PredictionExplainer:
    """Test F1PredictionExplainer class."""

    def test_explainer_initialization(self, explainer, mock_provider):
        """Test explainer initialization."""
        assert explainer.provider == mock_provider
        assert explainer.template_manager is not None

    @pytest.mark.asyncio
    async def test_explain_race_prediction_success(self, explainer, mock_provider):
        """Test successful race prediction explanation."""
        mock_response = LLMResponse(
            content="This driver is predicted to win because...",
            model="test-model",
            provider="test-provider",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost=0.005,
            metadata={},
        )

        mock_provider.generate = AsyncMock(return_value=mock_response)

        explanation = await explainer.explain_race_prediction(
            race_name="Monaco GP",
            drivers=["Driver A", "Driver B", "Driver C"],
            predicted_winner="Driver A",
            confidence=0.85,
            key_factors={"qualifying_position": "pole", "car_pace": "strong"},
        )

        assert explanation == "This driver is predicted to win because..."
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_explain_race_prediction_invalid_confidence(self, explainer):
        """Test explanation with invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            await explainer.explain_race_prediction(
                race_name="Monaco GP",
                drivers=["Driver A"],
                predicted_winner="Driver A",
                confidence=1.5,  # Invalid: > 1.0
                key_factors={},
            )

    @pytest.mark.asyncio
    async def test_explain_race_prediction_with_session(
        self, explainer, mock_provider
    ):
        """Test explanation with chat session."""
        mock_response = LLMResponse(
            content="Driver A will win the race.",
            model="test-model",
            provider="test-provider",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            estimated_cost=0.004,
            metadata={},
        )

        mock_provider.generate = AsyncMock(return_value=mock_response)
        session = ChatSession()

        await explainer.explain_race_prediction(
            race_name="Monaco GP",
            drivers=["Driver A"],
            predicted_winner="Driver A",
            confidence=0.9,
            key_factors={},
            session=session,
        )

        # Verify session has messages
        assert session.message_count == 2
        assert session.user_message_count == 1
        assert session.assistant_message_count == 1

    @pytest.mark.asyncio
    async def test_explain_driver_performance_success(
        self, explainer, mock_provider
    ):
        """Test successful driver performance explanation."""
        mock_response = LLMResponse(
            content="Driver will perform well due to recent form.",
            model="test-model",
            provider="test-provider",
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            estimated_cost=0.0045,
            metadata={},
        )

        mock_provider.generate = AsyncMock(return_value=mock_response)

        explanation = await explainer.explain_driver_performance(
            driver_name="Driver A",
            recent_form={"last_race": "1st", "avg_position": "2nd"},
            circuit_factors={"experience": 0.9, "pace": 0.85},
            weather_forecast={"rain": "50%", "temp": "25C"},
        )

        assert "Driver will perform well" in explanation
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_explain_prediction_uncertainty(self, explainer, mock_provider):
        """Test uncertainty explanation."""
        mock_response = LLMResponse(
            content="Uncertainty comes from weather and competition.",
            model="test-model",
            provider="test-provider",
            input_tokens=100,
            output_tokens=30,
            total_tokens=130,
            estimated_cost=0.004,
            metadata={},
        )

        mock_provider.generate = AsyncMock(return_value=mock_response)

        explanation = await explainer.explain_prediction_uncertainty(
            prediction="Driver A to finish in top 3",
            confidence=0.65,
            uncertainty_factors=["Weather changes", "Tire strategy", "Safety cars"],
        )

        assert "Uncertainty comes from" in explanation

    @pytest.mark.asyncio
    async def test_generate_detailed_analysis(self, explainer, mock_provider):
        """Test detailed analysis generation."""
        mock_response = LLMResponse(
            content="Paragraph 1...\nParagraph 2...\nParagraph 3...",
            model="test-model",
            provider="test-provider",
            input_tokens=200,
            output_tokens=150,
            total_tokens=350,
            estimated_cost=0.011,
            metadata={},
        )

        mock_provider.generate = AsyncMock(return_value=mock_response)

        analysis = await explainer.generate_detailed_analysis(
            race_name="Monaco GP",
            data={
                "top_contender": "Driver A",
                "track_type": "street circuit",
                "weather": "clear",
            },
        )

        assert "Paragraph" in analysis
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_detailed_analysis_with_session(self, explainer, mock_provider):
        """Test detailed analysis with session context."""
        mock_response = LLMResponse(
            content="Analysis text",
            model="test-model",
            provider="test-provider",
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            estimated_cost=0.01,
            metadata={},
        )

        mock_provider.generate = AsyncMock(return_value=mock_response)
        session = ChatSession()

        await explainer.generate_detailed_analysis(
            race_name="Monaco GP",
            data={"key": "value"},
            session=session,
        )

        assert session.message_count == 2

    def test_create_explanation_summary(self, explainer):
        """Test explanation summary creation."""
        summary = explainer.create_explanation_summary(
            explanation="Driver A will win the race.",
            confidence=0.85,
            metadata={"race_id": "001"},
        )

        assert summary["explanation"] == "Driver A will win the race."
        assert summary["confidence"] == 0.85
        assert summary["provider"] == "test-provider"
        assert summary["model"] == "test-model"
        assert summary["metadata"]["race_id"] == "001"

    def test_explanation_summary_without_metadata(self, explainer):
        """Test explanation summary without metadata."""
        summary = explainer.create_explanation_summary(
            explanation="Test explanation",
            confidence=0.75,
        )

        assert summary["metadata"] == {}

    @pytest.mark.asyncio
    async def test_explain_race_prediction_error_handling(
        self, explainer, mock_provider
    ):
        """Test error handling in race prediction."""
        mock_provider.generate = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(Exception, match="API error"):
            await explainer.explain_race_prediction(
                race_name="Monaco GP",
                drivers=["Driver A"],
                predicted_winner="Driver A",
                confidence=0.8,
                key_factors={},
            )

    @pytest.mark.asyncio
    async def test_driver_performance_with_session(self, explainer, mock_provider):
        """Test driver performance explanation with session."""
        mock_response = LLMResponse(
            content="Driver performance analysis",
            model="test-model",
            provider="test-provider",
            input_tokens=100,
            output_tokens=30,
            total_tokens=130,
            estimated_cost=0.004,
            metadata={},
        )

        mock_provider.generate = AsyncMock(return_value=mock_response)
        session = ChatSession(model="test-model", provider="test-provider")

        await explainer.explain_driver_performance(
            driver_name="Driver A",
            recent_form={},
            circuit_factors={},
            weather_forecast={},
            session=session,
        )

        assert session.message_count == 2


class TestExplanationIntegration:
    """Integration tests for explanation generation."""

    @pytest.mark.asyncio
    async def test_multiple_explanations_same_session(self, explainer, mock_provider):
        """Test generating multiple explanations in same session."""
        mock_response = LLMResponse(
            content="Explanation text",
            model="test-model",
            provider="test-provider",
            input_tokens=100,
            output_tokens=30,
            total_tokens=130,
            estimated_cost=0.004,
            metadata={},
        )

        mock_provider.generate = AsyncMock(return_value=mock_response)
        session = ChatSession()

        # Generate first explanation
        await explainer.explain_race_prediction(
            race_name="Monaco GP",
            drivers=["Driver A"],
            predicted_winner="Driver A",
            confidence=0.8,
            key_factors={},
            session=session,
        )

        # Generate second explanation
        await explainer.explain_driver_performance(
            driver_name="Driver A",
            recent_form={},
            circuit_factors={},
            weather_forecast={},
            session=session,
        )

        # Session should have 4 messages (2 per explanation)
        assert session.message_count == 4

    @pytest.mark.asyncio
    async def test_confidence_ranges(self, explainer, mock_provider):
        """Test explanation generation with various confidence levels."""
        mock_response = LLMResponse(
            content="Explanation",
            model="test-model",
            provider="test-provider",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            estimated_cost=0.004,
            metadata={},
        )

        mock_provider.generate = AsyncMock(return_value=mock_response)

        for confidence in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = await explainer.explain_race_prediction(
                race_name="Test GP",
                drivers=["Driver"],
                predicted_winner="Driver",
                confidence=confidence,
                key_factors={},
            )
            assert result is not None
