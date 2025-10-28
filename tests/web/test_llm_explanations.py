"""Tests for LLM explanation utilities."""
import pytest
from unittest.mock import MagicMock, patch

from f1_predict.web.utils.llm_explanations import (
    generate_prediction_explanation,
    generate_driver_comparison,
)


def test_generate_prediction_explanation_returns_string(mock_llm_explanation):
    """Test that generate_prediction_explanation returns a string."""
    prediction = {
        "podium": [
            {"position": 1, "driver": "Verstappen", "confidence": 0.87},
            {"position": 2, "driver": "Hamilton", "confidence": 0.72},
        ],
        "metadata": {"race_id": "2025_21"},
    }

    with patch('f1_predict.web.utils.llm_explanations.LLMClient', create=True) as MockLLM:
        mock_llm = MagicMock()
        mock_llm.generate_explanation.return_value = mock_llm_explanation
        MockLLM.return_value = mock_llm

        result = generate_prediction_explanation(prediction)

        assert isinstance(result, str)
        assert len(result) > 0


def test_generate_driver_comparison_returns_string():
    """Test that generate_driver_comparison returns comparison text."""
    driver1 = "Verstappen"
    driver2 = "Hamilton"

    with patch('f1_predict.web.utils.llm_explanations.LLMClient', create=True) as MockLLM:
        mock_llm = MagicMock()
        mock_llm.compare_drivers.return_value = "Comparison text"
        MockLLM.return_value = mock_llm

        result = generate_driver_comparison(driver1, driver2)

        assert isinstance(result, str)
        assert len(result) > 0
