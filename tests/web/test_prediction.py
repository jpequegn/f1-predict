"""Tests for web prediction utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from f1_predict.web.utils.prediction import PredictionManager


@pytest.fixture
def prediction_manager():
    """Create prediction manager instance for testing."""
    return PredictionManager()


@pytest.fixture
def sample_race_data():
    """Create sample race data."""
    return {
        "race_id": "race_1",
        "name": "Monaco Grand Prix",
        "circuit": "Monaco",
        "date": "2024-05-26",
        "location": "Monte Carlo",
        "round": 5,
        "season": 2024,
    }


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    return pd.DataFrame(
        {
            "driver_id": [f"driver_{i}" for i in range(20)],
            "qualifying_position": list(range(1, 21)),
            "driver_form_score": [90.0 - i * 3 for i in range(20)],
            "team_reliability_score": [88.0 - i * 2 for i in range(20)],
            "circuit_performance_score": [85.0 - i * 2.5 for i in range(20)],
        }
    )


@pytest.fixture
def sample_prediction():
    """Create sample prediction result."""
    return {
        "race": "Monaco Grand Prix",
        "podium": [
            ("driver_0", 0.85),
            ("driver_1", 0.78),
            ("driver_2", 0.72),
        ],
        "predictions": [
            {"position": i + 1, "driver_id": f"driver_{i}", "confidence": 0.9 - i * 0.03}
            for i in range(20)
        ],
    }


class TestPredictionManager:
    """Tests for PredictionManager class."""

    def test_initialization(self, prediction_manager):
        """Test prediction manager initialization."""
        assert prediction_manager is not None
        assert hasattr(prediction_manager, "logger")
        assert hasattr(prediction_manager, "collector")
        assert hasattr(prediction_manager, "form_calculator")

    def test_get_upcoming_races_empty(self, prediction_manager):
        """Test getting upcoming races when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            races = prediction_manager.get_upcoming_races()
            assert isinstance(races, list)
            assert len(races) == 0

    def test_get_upcoming_races_with_data(self, prediction_manager, tmp_path):
        """Test getting upcoming races from CSV file."""
        # Create temporary schedule file
        schedule_df = pd.DataFrame(
            {
                "race_id": ["race_1", "race_2"],
                "name": ["Monaco Grand Prix", "Canadian Grand Prix"],
                "circuit": ["Monaco", "Montreal"],
                "date": ["2024-05-26", "2024-06-09"],
                "location": ["Monte Carlo", "Montreal"],
                "round": [5, 6],
                "season": [2024, 2024],
            }
        )

        schedule_file = tmp_path / "race_schedules.csv"
        schedule_df.to_csv(schedule_file, index=False)

        # Patch the Path constructor in the prediction module
        with patch(
            "f1_predict.web.utils.prediction.Path",
            return_value=schedule_file,
        ):
            with patch(
                "f1_predict.web.utils.prediction.pd.read_csv",
                return_value=schedule_df,
            ):
                races = prediction_manager.get_upcoming_races()
                assert len(races) == 2
                assert races[0]["name"] == "Monaco Grand Prix"
                assert races[1]["name"] == "Canadian Grand Prix"

    def test_load_model_ensemble(self, prediction_manager):
        """Test loading ensemble model."""
        model, metadata = prediction_manager.load_model("ensemble")
        assert model is not None
        assert metadata["type"] == "ensemble"
        assert "accuracy" in metadata
        assert "training_date" in metadata

    def test_load_model_xgboost(self, prediction_manager):
        """Test loading XGBoost model."""
        model, metadata = prediction_manager.load_model("xgboost")
        assert model is not None
        assert metadata["type"] == "xgboost"

    def test_load_model_lightgbm(self, prediction_manager):
        """Test loading LightGBM model."""
        model, metadata = prediction_manager.load_model("lightgbm")
        assert model is not None
        assert metadata["type"] == "lightgbm"

    def test_load_model_random_forest(self, prediction_manager):
        """Test loading Random Forest model."""
        model, metadata = prediction_manager.load_model("random_forest")
        assert model is not None
        assert metadata["type"] == "random_forest"

    def test_load_model_invalid_type(self, prediction_manager):
        """Test loading invalid model type raises error."""
        with pytest.raises(ValueError, match="Invalid model type"):
            prediction_manager.load_model("invalid_model")

    def test_prepare_race_features_success(
        self, prediction_manager, sample_race_data, tmp_path
    ):
        """Test preparing race features."""
        # Create temporary race results file with date column for form calculator
        results_df = pd.DataFrame(
            {
                "driver_id": [f"driver_{i}" for i in range(20)],
                "position": list(range(1, 21)),
                "points": [25 - i for i in range(20)],
                "date": pd.date_range("2024-01-01", periods=20),
            }
        )

        results_file = tmp_path / "race_results_2024.csv"
        results_df.to_csv(results_file, index=False)

        # Patch the Path constructor and read_csv in the prediction module
        def mock_path_constructor(path_str):
            """Mock Path constructor."""
            if "race_results" in str(path_str):
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.__truediv__ = lambda self, other: results_file
                return mock_path
            return Path(path_str)

        with patch(
            "f1_predict.web.utils.prediction.Path",
            side_effect=lambda x: (
                results_file
                if "race_results" in str(x)
                else MagicMock(exists=lambda: False)
            ),
        ):
            with patch(
                "f1_predict.web.utils.prediction.pd.read_csv",
                return_value=results_df,
            ):
                # Mock the form_calculator to avoid date issues
                with patch.object(
                    prediction_manager.form_calculator,
                    "calculate_form_score",
                    return_value=85.0,
                ):
                    features = prediction_manager.prepare_race_features(
                        sample_race_data["race_id"], sample_race_data["season"]
                    )

                    assert isinstance(features, pd.DataFrame)
                    assert len(features) > 0
                    assert "driver_id" in features.columns
                    assert "driver_form_score" in features.columns

    def test_prepare_race_features_missing_file(self, prediction_manager):
        """Test preparing race features with missing file."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                prediction_manager.prepare_race_features("race_1", 2024)

    def test_generate_prediction(
        self, prediction_manager, sample_features, sample_race_data
    ):
        """Test generating predictions."""
        mock_model = Mock()
        mock_model.predict_proba.return_value = {
            "driver_0": 0.85,
            "driver_1": 0.78,
            "driver_2": 0.72,
        }

        prediction = prediction_manager.generate_prediction(
            mock_model, sample_features, sample_race_data["name"]
        )

        assert "race" in prediction
        assert "predictions" in prediction
        assert "podium" in prediction
        assert prediction["race"] == sample_race_data["name"]
        assert len(prediction["predictions"]) > 0
        assert len(prediction["podium"]) == 3

    def test_format_prediction_results(
        self, prediction_manager, sample_prediction
    ):
        """Test formatting prediction results."""
        results_df = prediction_manager.format_prediction_results(sample_prediction)

        assert isinstance(results_df, pd.DataFrame)
        assert "Position" in results_df.columns
        assert "Driver" in results_df.columns
        assert "Team" in results_df.columns
        assert "Confidence" in results_df.columns
        assert len(results_df) == 20

    def test_format_prediction_results_with_driver_info(
        self, prediction_manager, sample_prediction
    ):
        """Test formatting prediction results with driver information."""
        driver_info = {
            "driver_0": {"name": "Lewis Hamilton", "team": "Mercedes"},
            "driver_1": {"name": "Max Verstappen", "team": "Red Bull"},
        }

        results_df = prediction_manager.format_prediction_results(
            sample_prediction, driver_info
        )

        assert results_df.iloc[0]["Driver"] == "Lewis Hamilton"
        assert results_df.iloc[0]["Team"] == "Mercedes"

    def test_export_prediction_csv(self, prediction_manager, sample_prediction):
        """Test exporting prediction as CSV."""
        csv_data = prediction_manager.export_prediction(
            sample_prediction, format="csv"
        )

        assert csv_data is not None
        assert isinstance(csv_data, bytes)
        assert b"position" in csv_data.lower()

    def test_export_prediction_json(self, prediction_manager, sample_prediction):
        """Test exporting prediction as JSON."""
        json_data = prediction_manager.export_prediction(
            sample_prediction, format="json"
        )

        assert json_data is not None
        assert isinstance(json_data, bytes)

        # Verify it's valid JSON
        parsed = json.loads(json_data.decode("utf-8"))
        assert "race" in parsed
        assert "predictions" in parsed

    def test_export_prediction_invalid_format(
        self, prediction_manager, sample_prediction
    ):
        """Test exporting prediction with invalid format."""
        result = prediction_manager.export_prediction(
            sample_prediction, format="xml"
        )
        assert result is None

    def test_export_prediction_error_handling(self, prediction_manager):
        """Test export handles errors gracefully."""
        invalid_prediction = None
        result = prediction_manager.export_prediction(invalid_prediction, format="csv")
        assert result is None


class TestPredictionPageIntegration:
    """Integration tests for prediction page."""

    def test_prediction_workflow(self, prediction_manager, sample_race_data):
        """Test complete prediction workflow."""
        # 1. Get races
        with patch.object(
            prediction_manager, "get_upcoming_races", return_value=[sample_race_data]
        ):
            races = prediction_manager.get_upcoming_races()
            assert len(races) > 0

        # 2. Load model
        model, metadata = prediction_manager.load_model("xgboost")
        assert model is not None

        # 3. Prepare features
        sample_features = pd.DataFrame(
            {
                "driver_id": [f"driver_{i}" for i in range(20)],
                "qualifying_position": list(range(1, 21)),
                "driver_form_score": [90.0 - i * 3 for i in range(20)],
                "team_reliability_score": [88.0 - i * 2 for i in range(20)],
                "circuit_performance_score": [85.0 - i * 2.5 for i in range(20)],
            }
        )

        # 4. Generate prediction
        mock_model = Mock()
        mock_model.predict_proba.return_value = {
            f"driver_{i}": 0.9 - i * 0.03 for i in range(20)
        }

        prediction = prediction_manager.generate_prediction(
            mock_model, sample_features, sample_race_data["name"]
        )

        assert "predictions" in prediction
        assert len(prediction["predictions"]) > 0

        # 5. Format results
        results_df = prediction_manager.format_prediction_results(prediction)
        assert not results_df.empty

        # 6. Export
        csv_data = prediction_manager.export_prediction(prediction, format="csv")
        assert csv_data is not None
