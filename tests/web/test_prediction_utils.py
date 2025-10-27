"""Tests for prediction utilities."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import pandas as pd

from f1_predict.web.utils.prediction import (
    get_upcoming_races,
    prepare_race_features,
    get_ensemble_prediction,
)


def test_get_upcoming_races_returns_dataframe(mock_upcoming_races):
    """Test that get_upcoming_races returns a properly formatted DataFrame."""
    with patch('f1_predict.web.utils.prediction.fetch_upcoming_races',
               return_value=mock_upcoming_races):
        result = get_upcoming_races()

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ['round', 'race_name', 'race_date'])
        assert len(result) > 0


def test_prepare_race_features_creates_valid_input(mock_upcoming_races):
    """Test that prepare_race_features creates ML-ready input."""
    race = mock_upcoming_races.iloc[0]
    features = prepare_race_features(race)

    assert isinstance(features, dict)
    assert 'season' in features
    assert 'round' in features
    assert 'circuit_id' in features
    assert features['season'] == 2025


def test_get_ensemble_prediction_returns_prediction_dict(
    mock_upcoming_races,
    mock_ensemble_predictions,
):
    """Test that get_ensemble_prediction returns properly formatted prediction."""
    race = mock_upcoming_races.iloc[0]

    with patch('f1_predict.web.utils.prediction.EnsemblePredictor') as MockPredictor:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_ensemble_predictions
        MockPredictor.return_value = mock_predictor

        result = get_ensemble_prediction(race)

        assert isinstance(result, dict)
        assert 'podium' in result
        assert 'full_order' in result
        assert 'feature_importance' in result
        assert len(result['podium']) == 3
