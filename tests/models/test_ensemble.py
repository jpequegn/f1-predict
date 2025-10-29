"""Tests for ensemble prediction model.

Comprehensive test coverage for the EnsemblePredictor class including
soft voting, hard voting, weight handling, and model agreement scoring.
"""

import pickle
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from f1_predict.models.ensemble import EnsemblePredictor


@pytest.fixture
def sample_features():
    """Create sample race features for testing."""
    return pd.DataFrame(
        {
            "driver_id": ["verstappen", "hamilton", "leclerc"],
            "qualifying_position": [1, 2, 3],
            "driver_form_score": [85.0, 78.0, 72.0],
            "team_reliability": [0.95, 0.88, 0.82],
            "circuit_performance": [90.0, 85.0, 75.0],
        }
    )


@pytest.fixture
def mock_models():
    """Create mock models for ensemble testing."""
    model1 = Mock()
    model2 = Mock()

    # Model 1 predictions: high confidence
    model1.predict_proba.return_value = np.array([0.85, 0.72, 0.60])
    model1.predict.return_value = pd.DataFrame(
        {
            "driver_id": ["verstappen", "hamilton", "leclerc"],
            "predicted_outcome": [1, 1, 0],
            "confidence": [85.0, 72.0, 60.0],
        }
    )
    model1.save = Mock()

    # Model 2 predictions: different pattern
    model2.predict_proba.return_value = np.array([0.80, 0.65, 0.55])
    model2.predict.return_value = pd.DataFrame(
        {
            "driver_id": ["verstappen", "hamilton", "leclerc"],
            "predicted_outcome": [1, 0, 0],
            "confidence": [80.0, 65.0, 55.0],
        }
    )
    model2.save = Mock()

    return [model1, model2]


class TestEnsemblePredictorInit:
    """Tests for EnsemblePredictor initialization."""

    def test_init_with_valid_models(self, mock_models):
        """Test initialization with valid models."""
        ensemble = EnsemblePredictor(models=mock_models, voting="soft")

        assert len(ensemble.models) == 2
        assert ensemble.voting == "soft"
        assert ensemble.weights == [0.5, 0.5]

    def test_init_with_empty_models(self):
        """Test that empty model list raises ValueError."""
        with pytest.raises(ValueError, match="At least one model required"):
            EnsemblePredictor(models=[], voting="soft")

    def test_init_with_invalid_voting(self, mock_models):
        """Test that invalid voting strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid voting strategy"):
            EnsemblePredictor(models=mock_models, voting="invalid")

    def test_init_with_custom_weights(self, mock_models):
        """Test initialization with custom weights."""
        weights = [0.6, 0.4]
        ensemble = EnsemblePredictor(models=mock_models, weights=weights, voting="soft")

        assert ensemble.weights == [0.6, 0.4]

    def test_init_with_mismatched_weights(self, mock_models):
        """Test that mismatched weights raise ValueError."""
        weights = [0.5, 0.3, 0.2]  # 3 weights for 2 models
        with pytest.raises(ValueError, match="Number of weights"):
            EnsemblePredictor(models=mock_models, weights=weights, voting="soft")

    def test_init_normalizes_weights(self, mock_models):
        """Test that weights are normalized."""
        weights = [1.0, 1.0]  # Sum = 2.0
        ensemble = EnsemblePredictor(models=mock_models, weights=weights, voting="soft")

        # Should be normalized to sum to 1.0
        assert sum(ensemble.weights) == pytest.approx(1.0)
        assert ensemble.weights == [0.5, 0.5]

    def test_init_with_unnormalized_weights(self, mock_models):
        """Test initialization with unnormalized weights."""
        weights = [2.0, 1.0]  # Sum = 3.0, should normalize to [2/3, 1/3]
        ensemble = EnsemblePredictor(models=mock_models, weights=weights, voting="soft")

        assert ensemble.weights == pytest.approx([2 / 3, 1 / 3])


class TestSoftVoting:
    """Tests for soft voting (probability-based) prediction."""

    def test_soft_voting_predict_proba(self, mock_models, sample_features):
        """Test soft voting probability prediction."""
        ensemble = EnsemblePredictor(models=mock_models, voting="soft")
        probs = ensemble.predict_proba(sample_features)

        # Should be average of the two models
        expected = np.array([0.825, 0.685, 0.575])  # Average of [0.85, 0.80], [0.72, 0.65], [0.60, 0.55]
        assert probs == pytest.approx(expected)

    def test_soft_voting_predict_with_default_threshold(self, mock_models, sample_features):
        """Test soft voting prediction with default threshold."""
        ensemble = EnsemblePredictor(models=mock_models, voting="soft")
        predictions = ensemble.predict(sample_features)

        assert len(predictions) == 3
        assert "driver_id" in predictions.columns
        assert "predicted_outcome" in predictions.columns
        assert "confidence" in predictions.columns

    def test_soft_voting_predict_with_custom_threshold(self, mock_models, sample_features):
        """Test soft voting with custom probability threshold."""
        ensemble = EnsemblePredictor(models=mock_models, voting="soft")

        # Low threshold
        predictions_low = ensemble.predict(sample_features, threshold=0.5)
        # High threshold
        predictions_high = ensemble.predict(sample_features, threshold=0.8)

        # Higher threshold should result in fewer positive predictions
        assert (predictions_high["predicted_outcome"].sum() <=
                predictions_low["predicted_outcome"].sum())

    def test_soft_voting_confidence_values(self, mock_models, sample_features):
        """Test that confidence values are in valid range."""
        ensemble = EnsemblePredictor(models=mock_models, voting="soft")
        predictions = ensemble.predict(sample_features)

        assert (predictions["confidence"] >= 0).all()
        assert (predictions["confidence"] <= 100).all()

    def test_soft_voting_with_three_models(self):
        """Test soft voting with more than two models."""
        model1 = Mock()
        model1.predict_proba.return_value = np.array([0.9, 0.5])
        model1.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "predicted_outcome": [1, 0],
                "confidence": [90.0, 50.0],
            }
        )

        model2 = Mock()
        model2.predict_proba.return_value = np.array([0.8, 0.6])
        model2.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "predicted_outcome": [1, 1],
                "confidence": [80.0, 60.0],
            }
        )

        model3 = Mock()
        model3.predict_proba.return_value = np.array([0.7, 0.4])
        model3.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "predicted_outcome": [1, 0],
                "confidence": [70.0, 40.0],
            }
        )

        features = pd.DataFrame({"driver_id": ["d1", "d2"]})
        ensemble = EnsemblePredictor(models=[model1, model2, model3], voting="soft")
        probs = ensemble.predict_proba(features)

        # Average: [0.8, 0.5]
        assert probs == pytest.approx([0.8, 0.5])


class TestHardVoting:
    """Tests for hard voting (decision-based) prediction."""

    def test_hard_voting_predict(self, mock_models, sample_features):
        """Test hard voting prediction."""
        ensemble = EnsemblePredictor(models=mock_models, voting="hard")
        predictions = ensemble.predict(sample_features)

        assert len(predictions) == 3
        assert "predicted_outcome" in predictions.columns

    def test_hard_voting_majority_decision(self, mock_models, sample_features):
        """Test hard voting uses majority decision."""
        ensemble = EnsemblePredictor(models=mock_models, voting="hard")
        predictions = ensemble.predict(sample_features, threshold=0.5)

        # Model 1: [1, 1, 0], Model 2: [1, 0, 0]
        # Majority: [1, 0.5, 0]
        # With weights [0.5, 0.5]: [0.5, 0.25, 0]
        assert predictions["predicted_outcome"].iloc[0] == 1  # Unanimous 1
        assert predictions["predicted_outcome"].iloc[2] == 0  # Unanimous 0

    def test_hard_voting_with_weighted_votes(self):
        """Test hard voting with weighted votes."""
        model1 = Mock()
        model1.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1"],
                "predicted_outcome": [1],
                "confidence": [100.0],
            }
        )

        model2 = Mock()
        model2.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1"],
                "predicted_outcome": [0],
                "confidence": [50.0],
            }
        )

        features = pd.DataFrame({"driver_id": ["d1"]})

        # Model 1 weighted 0.7, Model 2 weighted 0.3
        ensemble = EnsemblePredictor(
            models=[model1, model2],
            weights=[0.7, 0.3],
            voting="hard",
        )
        predictions = ensemble.predict(features, threshold=0.5)

        # Weighted vote: 1*0.7 + 0*0.3 = 0.7 > 0.5, so prediction is 1
        assert predictions["predicted_outcome"].iloc[0] == 1


class TestModelAgreement:
    """Tests for model agreement scoring."""

    def test_model_agreement_perfect(self):
        """Test model agreement when models always agree."""
        model1 = Mock()
        model1.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2", "d3"],
                "predicted_outcome": [1, 1, 0],
                "confidence": [90.0, 85.0, 80.0],
            }
        )

        model2 = Mock()
        model2.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2", "d3"],
                "predicted_outcome": [1, 1, 0],
                "confidence": [88.0, 87.0, 82.0],
            }
        )

        features = pd.DataFrame({"driver_id": ["d1", "d2", "d3"]})
        ensemble = EnsemblePredictor(models=[model1, model2])
        agreement = ensemble.get_model_agreement(features)

        assert agreement == 1.0  # Perfect agreement

    def test_model_agreement_partial(self):
        """Test model agreement with partial disagreement."""
        model1 = Mock()
        model1.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "predicted_outcome": [1, 0],
                "confidence": [90.0, 80.0],
            }
        )

        model2 = Mock()
        model2.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "predicted_outcome": [1, 1],
                "confidence": [85.0, 70.0],
            }
        )

        features = pd.DataFrame({"driver_id": ["d1", "d2"]})
        ensemble = EnsemblePredictor(models=[model1, model2])
        agreement = ensemble.get_model_agreement(features)

        # 1 agreement out of 2 samples = 0.5
        assert agreement == 0.5

    def test_model_agreement_no_agreement(self):
        """Test model agreement when models disagree on all samples."""
        model1 = Mock()
        model1.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "predicted_outcome": [1, 0],
                "confidence": [90.0, 80.0],
            }
        )

        model2 = Mock()
        model2.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "predicted_outcome": [0, 1],
                "confidence": [85.0, 70.0],
            }
        )

        features = pd.DataFrame({"driver_id": ["d1", "d2"]})
        ensemble = EnsemblePredictor(models=[model1, model2])
        agreement = ensemble.get_model_agreement(features)

        assert agreement == 0.0  # No agreement

    def test_model_agreement_empty_features(self, mock_models):
        """Test model agreement with empty features."""
        ensemble = EnsemblePredictor(models=mock_models)
        agreement = ensemble.get_model_agreement(pd.DataFrame())

        assert agreement == 0.0


class TestSaveLoad:
    """Tests for ensemble serialization."""

    def test_save_ensemble(self, mock_models, tmp_path):
        """Test saving ensemble to disk."""
        ensemble = EnsemblePredictor(models=mock_models, weights=[0.6, 0.4], voting="soft")

        save_path = tmp_path / "test_ensemble.pkl"
        ensemble.save(save_path)

        assert save_path.exists()
        # Check that model directory was created
        assert (tmp_path / "test_ensemble_models").exists()

    def test_load_ensemble(self, mock_models, tmp_path):
        """Test loading ensemble from disk."""
        # Create and save ensemble
        ensemble = EnsemblePredictor(models=mock_models, weights=[0.6, 0.4], voting="hard")
        save_path = tmp_path / "test_ensemble.pkl"
        ensemble.save(save_path)

        # Verify that metadata was saved correctly
        with open(save_path, "rb") as f:
            data = pickle.load(f)
            assert data["weights"] == pytest.approx([0.6, 0.4])
            assert data["voting"] == "hard"
            assert len(data["model_paths"]) == 2

    def test_load_nonexistent_ensemble(self, tmp_path):
        """Test loading ensemble from nonexistent path."""
        nonexistent_path = tmp_path / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError, match="Ensemble file not found"):
            EnsemblePredictor.load(nonexistent_path)


class TestEmptyData:
    """Tests for handling empty data."""

    def test_predict_proba_empty_features(self, mock_models):
        """Test predict_proba with empty features."""
        ensemble = EnsemblePredictor(models=mock_models, voting="soft")
        result = ensemble.predict_proba(pd.DataFrame())

        assert len(result) == 0

    def test_predict_empty_features(self, mock_models):
        """Test predict with empty features."""
        ensemble = EnsemblePredictor(models=mock_models, voting="soft")
        result = ensemble.predict(pd.DataFrame())

        assert len(result) == 0
        assert "driver_id" in result.columns
        assert "predicted_outcome" in result.columns


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_model_ensemble(self):
        """Test ensemble with single model."""
        model = Mock()
        model.predict_proba.return_value = np.array([0.8, 0.6])

        features = pd.DataFrame({"driver_id": ["d1", "d2"]})
        ensemble = EnsemblePredictor(models=[model], voting="soft")
        probs = ensemble.predict_proba(features)

        assert probs == pytest.approx([0.8, 0.6])

    def test_ensemble_with_extreme_weights(self, mock_models):
        """Test ensemble with extreme weight values."""
        weights = [0.99, 0.01]
        features = pd.DataFrame(
            {
                "driver_id": ["d1"],
                "feature1": [1.0],
            }
        )

        ensemble = EnsemblePredictor(models=mock_models, weights=weights, voting="soft")
        probs = ensemble.predict_proba(features)

        # Result should be closer to model1 (0.85 * 0.99 + 0.80 * 0.01)
        expected = 0.85 * 0.99 + 0.80 * 0.01
        assert probs[0] == pytest.approx(expected)

    def test_ensemble_consistency(self):
        """Test that ensemble predictions are deterministic."""
        # Create models that return correct number of predictions
        model1 = Mock()
        model1.predict_proba.return_value = np.array([0.85, 0.72])
        model1.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "predicted_outcome": [1, 1],
                "confidence": [85.0, 72.0],
            }
        )

        model2 = Mock()
        model2.predict_proba.return_value = np.array([0.80, 0.65])
        model2.predict.return_value = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "predicted_outcome": [1, 0],
                "confidence": [80.0, 65.0],
            }
        )

        features = pd.DataFrame(
            {
                "driver_id": ["d1", "d2"],
                "feature1": [1.0, 2.0],
            }
        )

        ensemble = EnsemblePredictor(models=[model1, model2], voting="soft")

        # Make single prediction and verify structure
        predictions = ensemble.predict(features)

        assert len(predictions) == 2
        assert all(col in predictions.columns for col in ["driver_id", "predicted_outcome", "confidence"])
        assert (predictions["confidence"] >= 0).all()
        assert (predictions["confidence"] <= 100).all()
