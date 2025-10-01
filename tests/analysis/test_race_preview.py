"""Tests for RacePreviewGenerator class."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from f1_predict.analysis.race_preview import RacePreviewGenerator
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
def mock_historical_provider():
    """Create mock historical context provider."""
    provider = MagicMock()
    provider.generate = AsyncMock()
    provider.generate.return_value = {
        "circuit_facts": [
            {
                "fact": "Lewis Hamilton has won 8 times at Silverstone",
                "relevance_score": 0.95,
                "category": "driver_record",
            }
        ],
        "interesting_patterns": ["Pole sitter wins 60% of races here"],
        "relevant_milestones": [],
    }
    return provider


@pytest.fixture
def preview_generator(mock_llm_provider, mock_historical_provider):
    """Create RacePreviewGenerator instance."""
    return RacePreviewGenerator(
        llm_provider=mock_llm_provider,
        historical_provider=mock_historical_provider,
    )


@pytest.fixture
def sample_race_data():
    """Sample race data for testing."""
    return {
        "race_name": "Monaco Grand Prix",
        "circuit_name": "Circuit de Monaco",
        "race_date": "2024-05-26",
        "round_number": 8,
        "circuit_characteristics": "Tight street circuit with 19 corners and limited overtaking",
        "past_winner": "Max Verstappen",
        "lap_record": "1:10.166 (Lewis Hamilton, 2021)",
        "top_drivers": [
            {"name": "Max Verstappen", "points": 150},
            {"name": "Sergio Perez", "points": 120},
            {"name": "Charles Leclerc", "points": 110},
        ],
        "predicted_winner": "Max Verstappen",
        "confidence": 85.0,
        "predicted_podium": "Verstappen, Leclerc, Perez",
        "key_factors": [
            "Strong qualifying performance",
            "Excellent tire management",
            "Track position critical",
        ],
    }


class TestRacePreviewGenerator:
    """Test RacePreviewGenerator functionality."""

    @pytest.mark.asyncio
    async def test_generate_preview_success(
        self,
        preview_generator,
        mock_llm_provider,
        sample_race_data,
    ):
        """Test successful preview generation."""
        # Mock LLM response
        mock_llm_provider.generate.return_value = LLMResponse(
            content="The Monaco Grand Prix returns to the iconic streets of Monte Carlo. "
            "Max Verstappen starts as favorite with 85% confidence based on his strong qualifying form. "
            "The tight street circuit demands precision and patience. "
            * 30,  # Repeat to meet word count
            model="test-model",
            provider="test",
            input_tokens=200,
            output_tokens=400,
            total_tokens=600,
            estimated_cost=0.01,
            metadata={},
        )

        result = await preview_generator.generate(**sample_race_data)

        assert result["race_name"] == "Monaco Grand Prix"
        assert result["circuit_name"] == "Circuit de Monaco"
        assert "preview_text" in result
        assert len(result["preview_text"]) > 100
        assert "generated_at" in result
        assert "readability" in result  # Word count is in readability metrics

    @pytest.mark.asyncio
    async def test_generate_with_historical_context(
        self,
        preview_generator,
        mock_llm_provider,
        mock_historical_provider,
        sample_race_data,
    ):
        """Test preview generation with historical context."""
        mock_llm_provider.generate.return_value = LLMResponse(
            content="Monaco preview with historical context" * 50,
            model="test-model",
            provider="test",
            input_tokens=200,
            output_tokens=400,
            total_tokens=600,
            estimated_cost=0.01,
            metadata={},
        )

        result = await preview_generator.generate(
            **sample_race_data,
            include_historical=True,
        )

        # Verify historical provider was called
        mock_historical_provider.generate.assert_called_once()
        assert "circuit_facts" in result
        assert len(result["circuit_facts"]) > 0

    @pytest.mark.asyncio
    async def test_generate_without_historical_context(
        self,
        preview_generator,
        mock_llm_provider,
        mock_historical_provider,
        sample_race_data,
    ):
        """Test preview generation without historical context."""
        mock_llm_provider.generate.return_value = LLMResponse(
            content="Monaco preview without history" * 50,
            model="test-model",
            provider="test",
            input_tokens=200,
            output_tokens=400,
            total_tokens=600,
            estimated_cost=0.01,
            metadata={},
        )

        result = await preview_generator.generate(
            **sample_race_data,
            include_historical=False,
        )

        # Verify historical provider was NOT called
        mock_historical_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_race_name(self, preview_generator, sample_race_data):
        """Test error handling for invalid race name."""
        sample_race_data["race_name"] = ""
        with pytest.raises(ValueError, match="Race name and circuit name are required"):
            await preview_generator.generate(**sample_race_data)

    @pytest.mark.asyncio
    async def test_invalid_circuit_name(self, preview_generator, sample_race_data):
        """Test error handling for invalid circuit name."""
        sample_race_data["circuit_name"] = ""
        with pytest.raises(ValueError, match="Race name and circuit name are required"):
            await preview_generator.generate(**sample_race_data)

    @pytest.mark.asyncio
    async def test_fallback_preview_on_llm_failure(
        self,
        preview_generator,
        mock_llm_provider,
        sample_race_data,
    ):
        """Test fallback preview when LLM fails."""
        # Make LLM raise exception
        mock_llm_provider.generate.side_effect = Exception("LLM service unavailable")

        # Should not raise, but use fallback
        result = await preview_generator.generate(**sample_race_data)

        assert "preview_text" in result
        assert len(result["preview_text"]) > 0
        assert "Monaco Grand Prix" in result["preview_text"]

    @pytest.mark.asyncio
    async def test_preview_word_count_validation(
        self,
        preview_generator,
        mock_llm_provider,
        sample_race_data,
    ):
        """Test word count validation and truncation."""
        # Create very long response
        long_text = "word " * 1000
        mock_llm_provider.generate.return_value = LLMResponse(
            content=long_text,
            model="test-model",
            provider="test",
            input_tokens=200,
            output_tokens=800,
            total_tokens=1000,
            estimated_cost=0.02,
            metadata={},
        )

        result = await preview_generator.generate(**sample_race_data)

        # Should be truncated
        word_count = len(result["preview_text"].split())
        assert word_count <= 700

    @pytest.mark.asyncio
    async def test_readability_metrics(
        self,
        preview_generator,
        mock_llm_provider,
        sample_race_data,
    ):
        """Test readability metrics calculation."""
        mock_llm_provider.generate.return_value = LLMResponse(
            content="Short preview text. " * 100,
            model="test-model",
            provider="test",
            input_tokens=200,
            output_tokens=400,
            total_tokens=600,
            estimated_cost=0.01,
            metadata={},
        )

        result = await preview_generator.generate(**sample_race_data)

        assert "readability" in result
        assert "words" in result["readability"]
        assert "sentences" in result["readability"]
        assert "readability_score" in result["readability"]

    @pytest.mark.asyncio
    async def test_generate_drivers_to_watch(self, preview_generator):
        """Test drivers to watch generation."""
        predictions = [
            {
                "driver_name": "Max Verstappen",
                "position": 1,
                "confidence": 85,
                "reasoning": "Strong qualifying form",
            },
            {
                "driver_name": "Charles Leclerc",
                "position": 2,
                "confidence": 75,
                "reasoning": "Home race advantage",
            },
            {
                "driver_name": "Sergio Perez",
                "position": 3,
                "confidence": 65,
                "reasoning": "Consistent performer",
            },
        ]

        result = await preview_generator.generate_drivers_to_watch(predictions, max_drivers=3)

        assert len(result) == 3
        assert result[0]["name"] == "Max Verstappen"
        assert result[0]["predicted_finish"] == "P1"
        assert result[0]["confidence"] == 85

    @pytest.mark.asyncio
    async def test_metadata_generation(
        self,
        preview_generator,
        mock_llm_provider,
        sample_race_data,
    ):
        """Test metadata is properly added."""
        mock_llm_provider.generate.return_value = LLMResponse(
            content="Preview text " * 100,
            model="test-model",
            provider="test",
            input_tokens=200,
            output_tokens=400,
            total_tokens=600,
            estimated_cost=0.01,
            metadata={},
        )

        result = await preview_generator.generate(**sample_race_data)

        assert "generated_at" in result
        assert "generator" in result
        assert "llm_provider" in result
        assert "llm_model" in result
        assert result["generator"] == "RacePreviewGenerator"
        assert result["llm_provider"] == "test_provider"

    @pytest.mark.asyncio
    async def test_minimal_data_preview(
        self,
        preview_generator,
        mock_llm_provider,
    ):
        """Test preview generation with minimal data."""
        mock_llm_provider.generate.return_value = LLMResponse(
            content="Minimal preview " * 100,
            model="test-model",
            provider="test",
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            estimated_cost=0.005,
            metadata={},
        )

        result = await preview_generator.generate(
            race_name="British Grand Prix",
            circuit_name="Silverstone",
            race_date="2024-07-07",
            round_number=12,
            circuit_characteristics="Fast flowing circuit",
        )

        assert result["race_name"] == "British Grand Prix"
        assert "preview_text" in result
        assert len(result["preview_text"]) > 0
