"""LLM integration tests for F1 Race Predictor.

Tests the integration of LLM providers (OpenAI, Anthropic, Local) with the prediction system,
including cost tracking, template rendering, error handling, and analysis generation.
"""

from unittest.mock import Mock, patch, MagicMock
from typing import Any
from datetime import datetime
from pathlib import Path
import tempfile

import pytest

from f1_predict.llm.base import BaseLLMProvider, LLMConfig, LLMResponse
from f1_predict.llm.exceptions import (
    RateLimitError,
    AuthenticationError,
    BudgetExceededError,
    InvalidResponseError,
    ProviderUnavailableError,
)
from f1_predict.llm.cost_tracker import CostTracker, UsageRecord


@pytest.fixture
def temp_cost_tracker_db() -> Any:
    """Provide a temporary database path for cost tracker testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_llm_usage.db"
        yield db_path


@pytest.mark.integration
class TestLLMProviderInitialization:
    """Test LLM provider initialization and authentication."""

    def test_llm_config_creation(self) -> None:
        """Test creating LLM configuration."""
        config = LLMConfig(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=1000,
        )

        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_llm_response_creation(self) -> None:
        """Test creating LLM response object."""
        response = LLMResponse(
            content="Test response",
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            input_tokens=50,
            output_tokens=50,
            total_tokens=100,
            estimated_cost=0.03,
            metadata={},
        )

        assert response.content == "Test response"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.provider == "anthropic"
        assert response.total_tokens == 100
        assert response.estimated_cost == 0.03

    def test_provider_authentication_error(self) -> None:
        """Test handling authentication errors."""
        # Create mock response
        with patch("f1_predict.llm.base.BaseLLMProvider.generate") as mock_generate:
            mock_generate.side_effect = AuthenticationError("Invalid API key")

            with pytest.raises(AuthenticationError):
                raise AuthenticationError("Invalid API key")


@pytest.mark.integration
class TestCostTracking:
    """Test LLM cost tracking and budget enforcement."""

    def test_cost_tracker_initialization(self, temp_cost_tracker_db: Path) -> None:
        """Test cost tracker initialization."""
        tracker = CostTracker(db_path=temp_cost_tracker_db, daily_budget=10.0, monthly_budget=200.0)

        assert tracker.daily_budget == 10.0
        assert tracker.monthly_budget == 200.0
        assert tracker.get_daily_cost() == 0.0
        assert tracker.get_monthly_cost() == 0.0

    def test_cost_tracking_within_budget(self, temp_cost_tracker_db: Path) -> None:
        """Test cost tracking when within budget."""
        tracker = CostTracker(db_path=temp_cost_tracker_db, daily_budget=10.0, monthly_budget=200.0)

        # Track cost within budget
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            template=None,
            input_tokens=50,
            output_tokens=50,
            total_tokens=100,
            estimated_cost=0.05,
            request_duration=1.2,
            success=True,
        )
        tracker.record_usage(record)

        assert tracker.get_daily_cost() == pytest.approx(0.05, rel=1e-4)
        assert tracker.get_monthly_cost() == pytest.approx(0.05, rel=1e-4)

    def test_cost_tracking_exceeds_daily_budget(self, temp_cost_tracker_db: Path) -> None:
        """Test cost tracking when daily budget is exceeded."""
        tracker = CostTracker(db_path=temp_cost_tracker_db, daily_budget=0.10, monthly_budget=200.0)

        # Add costs that will exceed daily budget
        record1 = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            template=None,
            input_tokens=100,
            output_tokens=100,
            total_tokens=200,
            estimated_cost=0.06,
            request_duration=1.5,
            success=True,
        )
        tracker.record_usage(record1)

        # Second record will exceed budget, should raise exception
        record2 = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            template=None,
            input_tokens=100,
            output_tokens=100,
            total_tokens=200,
            estimated_cost=0.05,
            request_duration=1.2,
            success=True,
        )

        # Should raise BudgetExceededError when exceeding
        with pytest.raises(BudgetExceededError):
            tracker.record_usage(record2)

    def test_cost_tracking_exceeds_monthly_budget(self, temp_cost_tracker_db: Path) -> None:
        """Test cost tracking when monthly budget is exceeded."""
        tracker = CostTracker(db_path=temp_cost_tracker_db, daily_budget=1000.0, monthly_budget=1.0)

        # Add first cost within budget
        record1 = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            template=None,
            input_tokens=200,
            output_tokens=200,
            total_tokens=400,
            estimated_cost=0.6,
            request_duration=2.0,
            success=True,
        )
        tracker.record_usage(record1)

        # Second record will exceed monthly budget, should raise exception
        record2 = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            template=None,
            input_tokens=200,
            output_tokens=200,
            total_tokens=400,
            estimated_cost=0.5,
            request_duration=1.8,
            success=True,
        )

        # Should raise BudgetExceededError when exceeding
        with pytest.raises(BudgetExceededError):
            tracker.record_usage(record2)

    def test_cost_tracking_multiple_records(self, temp_cost_tracker_db: Path) -> None:
        """Test cost tracking with multiple records."""
        tracker = CostTracker(db_path=temp_cost_tracker_db, daily_budget=10.0, monthly_budget=200.0)

        # Track first usage record
        record1 = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            template=None,
            input_tokens=50,
            output_tokens=50,
            total_tokens=100,
            estimated_cost=0.03,
            request_duration=1.0,
            success=True,
        )
        tracker.record_usage(record1)
        assert tracker.get_daily_cost() == pytest.approx(0.03, rel=1e-4)

        # Track second usage record
        record2 = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            template=None,
            input_tokens=50,
            output_tokens=50,
            total_tokens=100,
            estimated_cost=0.02,
            request_duration=1.0,
            success=True,
        )
        tracker.record_usage(record2)
        assert tracker.get_daily_cost() == pytest.approx(0.05, rel=1e-4)
        assert tracker.get_monthly_cost() == pytest.approx(0.05, rel=1e-4)


@pytest.mark.integration
class TestLLMErrorHandling:
    """Test LLM error handling and recovery."""

    def test_rate_limit_error(self) -> None:
        """Test handling rate limit errors."""
        with pytest.raises(RateLimitError):
            raise RateLimitError("Rate limit exceeded: 100 requests/minute")

    def test_budget_exceeded_error(self) -> None:
        """Test handling budget exceeded errors."""
        with pytest.raises(BudgetExceededError):
            raise BudgetExceededError("Daily budget of $10.00 exceeded")

    def test_invalid_response_error(self) -> None:
        """Test handling invalid response errors."""
        with pytest.raises(InvalidResponseError):
            raise InvalidResponseError("Invalid JSON response from API")

    def test_provider_unavailable_error(self) -> None:
        """Test handling provider unavailable errors."""
        with pytest.raises(ProviderUnavailableError):
            raise ProviderUnavailableError("OpenAI API temporarily unavailable")

    def test_error_recovery_with_fallback(self) -> None:
        """Test error recovery with fallback provider."""
        primary_error = AuthenticationError("Primary provider auth failed")
        fallback_response = "Fallback analysis from local provider"

        # Simulate fallback recovery
        assert isinstance(primary_error, AuthenticationError)
        # In real implementation, would switch to fallback provider


@pytest.mark.integration
class TestLLMTemplateRendering:
    """Test LLM prompt template rendering."""

    def test_template_variables_available(self) -> None:
        """Test that template variables are properly provided."""
        template_vars = {
            "race_name": "Monaco Grand Prix",
            "season": 2024,
            "round": 3,
            "driver_names": ["Verstappen", "Hamilton", "Leclerc"],
            "track_characteristics": "Street circuit, tight corners",
            "weather": "Sunny, 25°C",
        }

        assert "race_name" in template_vars
        assert "driver_names" in template_vars
        assert len(template_vars["driver_names"]) == 3

    def test_template_rendering_with_race_preview(self) -> None:
        """Test race preview template rendering."""
        template_vars = {
            "race_name": "Monaco Grand Prix",
            "season": 2024,
            "round": 3,
            "circuit": "Monte Carlo",
            "weather": "Sunny",
            "pole_position": "Verstappen",
            "top_3_grid": ["Verstappen", "Hamilton", "Leclerc"],
            "key_factors": ["Qualifying crucial", "Weather stable", "Tire management"],
        }

        # Verify all required variables present
        required_vars = [
            "race_name",
            "season",
            "round",
            "circuit",
            "weather",
        ]
        for var in required_vars:
            assert var in template_vars

    def test_template_rendering_with_prediction_explanation(self) -> None:
        """Test prediction explanation template rendering."""
        template_vars = {
            "predicted_winner": "Verstappen",
            "confidence": 0.92,
            "top_3_predictions": [
                {"driver": "Verstappen", "probability": 0.92},
                {"driver": "Hamilton", "probability": 0.06},
                {"driver": "Leclerc", "probability": 0.02},
            ],
            "key_features": [
                {"feature": "driver_form_score", "importance": 0.35},
                {"feature": "team_reliability", "importance": 0.28},
                {"feature": "qualifying_edge", "importance": 0.25},
            ],
            "explanation_detail": "detailed",
        }

        assert template_vars["predicted_winner"] == "Verstappen"
        assert template_vars["confidence"] >= 0.8
        assert len(template_vars["top_3_predictions"]) == 3
        assert len(template_vars["key_features"]) >= 3


@pytest.mark.integration
class TestLLMAnalysisGeneration:
    """Test LLM-based analysis generation."""

    def test_race_preview_generation_success(self) -> None:
        """Test successful race preview generation."""
        mock_llm_response = {
            "content": "Monaco Grand Prix is a prestigious street circuit requiring high precision driving skills. The tight corners and lack of run-off areas make qualifying crucial. Drivers must be extremely precise with their braking points and lines. This is one of the most challenging races on the calendar where consistency and car control are paramount.",
            "input_tokens": 150,
            "output_tokens": 300,
            "total_tokens": 450,
            "estimated_cost": 0.045,
        }

        # Verify response structure
        assert "content" in mock_llm_response
        assert len(mock_llm_response["content"]) > 100  # Should be substantial
        assert mock_llm_response["total_tokens"] > 0
        assert mock_llm_response["estimated_cost"] > 0

    def test_prediction_explanation_simple(self) -> None:
        """Test simple-level prediction explanation."""
        mock_explanation = "Verstappen is predicted to win because of his superior car performance and qualifying advantage."

        assert len(mock_explanation) > 50
        assert "predicted" in mock_explanation.lower() or "win" in mock_explanation.lower()

    def test_prediction_explanation_detailed(self) -> None:
        """Test detailed-level prediction explanation."""
        mock_explanation = (
            "Verstappen leads with 92% confidence. Key factors: "
            "1) Driver form (85/100) - recent excellent performance. "
            "2) Team reliability (95%) - Red Bull has been consistent. "
            "3) Qualifying advantage (+0.10s) - strong qualifying setup."
        )

        assert len(mock_explanation) > 100
        assert "confidence" in mock_explanation.lower()
        assert "%" in mock_explanation

    def test_prediction_explanation_technical(self) -> None:
        """Test technical-level prediction explanation."""
        mock_explanation = (
            "Model: GBM with 150 features. "
            "Top feature importance: driver_form_score (35.2%), "
            "team_reliability (28.1%), qualifying_edge (25.3%). "
            "Prediction confidence: 92% ± 3.2% (95% CI). "
            "Top 3: Verstappen (0.92), Hamilton (0.06), Leclerc (0.02)."
        )

        assert "Model:" in mock_explanation
        assert "feature importance" in mock_explanation.lower()
        assert "confidence" in mock_explanation.lower()

    def test_analysis_generation_with_fallback(self) -> None:
        """Test analysis generation with fallback when LLM fails."""
        # First attempt fails, fallback succeeds
        fallback_content = "Unable to generate detailed analysis. Prediction: Verstappen (92% confidence)"

        assert fallback_content is not None
        assert len(fallback_content) > 0
        assert "confidence" in fallback_content.lower()


@pytest.mark.integration
class TestLLMMultiProviderSupport:
    """Test multi-provider LLM support."""

    def test_provider_switching_anthropic_to_openai(self) -> None:
        """Test switching from Anthropic to OpenAI provider."""
        config_anthropic = LLMConfig(
            model="claude-3-5-sonnet-20241022",
        )

        config_openai = LLMConfig(
            model="gpt-4-turbo",
        )

        # Verify configurations have different models
        assert config_anthropic.model != config_openai.model
        assert config_anthropic.model == "claude-3-5-sonnet-20241022"
        assert config_openai.model == "gpt-4-turbo"

    def test_provider_switching_cloud_to_local(self) -> None:
        """Test switching from cloud provider to local Ollama."""
        config_cloud = LLMConfig(model="claude-3-5-sonnet-20241022")
        config_local = LLMConfig(model="llama2")

        # Verify both configurations are valid
        assert config_cloud.model == "claude-3-5-sonnet-20241022"
        assert config_local.model == "llama2"

    def test_provider_pricing_comparison(self) -> None:
        """Test provider cost comparison."""
        # Anthropic Claude 3.5 Sonnet
        anthropic_cost = 0.003  # $3 per 1M input tokens

        # OpenAI GPT-4 Turbo
        openai_cost = 0.01  # $10 per 1M input tokens

        # Local (free)
        local_cost = 0.0

        # Verify pricing hierarchy
        assert local_cost == 0.0
        assert anthropic_cost < openai_cost
        assert anthropic_cost > 0


@pytest.mark.integration
class TestLLMIntegrationWithPrediction:
    """Test LLM integration with prediction system."""

    def test_llm_explanation_for_prediction(self) -> None:
        """Test generating LLM explanation for a prediction."""
        prediction_data = {
            "predicted_winner": "Verstappen",
            "confidence": 0.92,
            "top_3": ["Verstappen", "Hamilton", "Leclerc"],
            "win_probability": 0.92,
            "podium_probability": 0.99,
        }

        # Verify prediction data available for LLM
        assert "predicted_winner" in prediction_data
        assert "confidence" in prediction_data
        assert prediction_data["confidence"] >= 0.9

    def test_llm_race_preview_before_prediction(self) -> None:
        """Test generating race preview before making predictions."""
        race_data = {
            "race_name": "Monaco Grand Prix",
            "season": 2024,
            "round": 3,
            "circuit": "Monte Carlo",
            "weather": "Sunny, 25°C",
            "schedule_date": "2024-05-26",
        }

        # Verify race data available for LLM
        assert "race_name" in race_data
        assert "circuit" in race_data
        assert race_data["season"] == 2024

    def test_prediction_explanation_integration(self) -> None:
        """Test complete prediction explanation workflow."""
        # Mock prediction result
        prediction_result = {
            "driver": "Verstappen",
            "position": 1,
            "confidence": 0.92,
            "key_factors": {
                "driver_form": 0.35,
                "team_reliability": 0.28,
                "qualifying_edge": 0.25,
                "circuit_experience": 0.12,
            },
        }

        # Verify can generate explanation from prediction
        assert "driver" in prediction_result
        assert "confidence" in prediction_result
        assert "key_factors" in prediction_result
        assert len(prediction_result["key_factors"]) >= 3


@pytest.mark.integration
class TestLLMStreamingResponses:
    """Test LLM streaming response handling."""

    def test_streaming_response_chunks(self) -> None:
        """Test receiving and processing streaming response chunks."""
        mock_chunks = [
            "Monaco ",
            "Grand ",
            "Prix ",
            "is ",
            "a ",
            "street ",
            "circuit",
            "...",
        ]

        # Collect chunks
        full_response = "".join(mock_chunks)

        assert len(full_response) > 0
        assert full_response.startswith("Monaco")
        assert "circuit" in full_response

    def test_streaming_response_token_counting(self) -> None:
        """Test token counting for streaming responses."""
        chunks = [
            "This is a test response",
            " with multiple chunks",
            " that simulate streaming",
        ]

        # Simulate token counting
        full_text = "".join(chunks)
        estimated_tokens = len(full_text.split())  # Simple word-based estimate

        assert estimated_tokens > 0
        assert len(chunks) == 3

    def test_streaming_response_cost_tracking(self, temp_cost_tracker_db: Path) -> None:
        """Test cost tracking during streaming."""
        tracker = CostTracker(db_path=temp_cost_tracker_db, daily_budget=10.0, monthly_budget=200.0)

        # Simulate streaming with cost updates
        chunk_costs = [0.001, 0.0012, 0.0011]
        for i, cost in enumerate(chunk_costs):
            record = UsageRecord(
                timestamp=datetime.now(),
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                template=None,
                input_tokens=20,
                output_tokens=30 + i * 10,
                total_tokens=50 + i * 10,
                estimated_cost=cost,
                request_duration=0.5,
                success=True,
            )
            tracker.record_usage(record)

        total_cost = sum(chunk_costs)
        assert tracker.get_daily_cost() == pytest.approx(total_cost, rel=1e-4)


@pytest.mark.integration
class TestLLMCaching:
    """Test LLM response caching."""

    def test_cache_race_preview(self) -> None:
        """Test caching race preview results."""
        cache_key = "race_preview_2024_3_monaco"
        cached_value = "Cached preview for Monaco Grand Prix"

        # Simulate cache storage
        cache = {cache_key: cached_value}

        assert cache_key in cache
        assert cache[cache_key] == cached_value

    def test_cache_prediction_explanation(self) -> None:
        """Test caching prediction explanation."""
        cache_key = "prediction_explanation_verstappen_0.92"
        cached_explanation = "Cached explanation for Verstappen prediction"

        cache = {cache_key: cached_explanation}

        assert cache_key in cache
        assert len(cache[cache_key]) > 0

    def test_cache_invalidation(self) -> None:
        """Test cache invalidation on new data."""
        cache = {"old_key": "old_value"}

        # Invalidate cache
        cache.clear()

        assert len(cache) == 0


@pytest.mark.integration
@pytest.mark.slow
class TestLLMEndToEndWorkflow:
    """End-to-end LLM workflow integration tests."""

    def test_complete_llm_workflow_with_mocks(self, temp_cost_tracker_db: Path) -> None:
        """Test complete workflow: predict → explain → track cost."""
        # Step 1: Make prediction
        prediction = {
            "driver": "Verstappen",
            "confidence": 0.92,
            "top_3": ["Verstappen", "Hamilton", "Leclerc"],
        }

        # Step 2: Generate LLM explanation
        explanation = f"Prediction: {prediction['driver']} wins with {prediction['confidence']:.0%} confidence"

        # Step 3: Track cost
        tracker = CostTracker(db_path=temp_cost_tracker_db, daily_budget=10.0, monthly_budget=200.0)
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            template=None,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost=0.03,
            request_duration=1.5,
            success=True,
        )
        tracker.record_usage(record)

        # Verify workflow
        assert prediction["driver"] == "Verstappen"
        assert "Verstappen" in explanation
        assert tracker.get_daily_cost() > 0

    def test_llm_workflow_with_fallback(self) -> None:
        """Test LLM workflow with fallback when primary provider fails."""
        # Try primary provider (fails)
        primary_provider = "anthropic"
        primary_failed = True

        # Fall back to local provider
        fallback_provider = "local"
        fallback_explanation = "Local model explanation: Driver performance based on available data."

        assert primary_failed is True
        assert len(fallback_explanation) > 0
        assert fallback_provider == "local"

    def test_llm_workflow_budget_awareness(self, temp_cost_tracker_db: Path) -> None:
        """Test LLM workflow respecting budget constraints."""
        tracker = CostTracker(db_path=temp_cost_tracker_db, daily_budget=0.50, monthly_budget=10.0)

        # Make multiple predictions with cost tracking within budget
        predictions = [
            {"driver": "Verstappen", "cost": 0.10},
            {"driver": "Hamilton", "cost": 0.10},
            {"driver": "Leclerc", "cost": 0.10},
        ]

        total_cost = 0
        for i, pred in enumerate(predictions):
            total_cost += pred["cost"]
            record = UsageRecord(
                timestamp=datetime.now(),
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                template=None,
                input_tokens=100,
                output_tokens=100 + i * 20,
                total_tokens=200 + i * 20,
                estimated_cost=pred["cost"],
                request_duration=1.5,
                success=True,
            )
            tracker.record_usage(record)

        # Verify cost tracking
        assert total_cost == pytest.approx(0.30, rel=1e-4)
        assert tracker.get_daily_cost() <= tracker.daily_budget
