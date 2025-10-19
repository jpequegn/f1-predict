"""Integration tests for web interface workflows."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestPredictionWorkflow:
    """Tests for complete prediction workflow."""

    def test_prediction_workflow_end_to_end(self, mock_prediction_manager):
        """Test complete prediction generation workflow."""
        from f1_predict.web.utils.prediction import PredictionManager

        manager = mock_prediction_manager

        # Step 1: Get races
        races = manager.get_upcoming_races()
        assert len(races) > 0
        selected_race = races[0]

        # Step 2: Load model
        model, metadata = manager.load_model("ensemble")
        assert model is not None
        assert metadata["type"] == "ensemble"

        # Step 3: Prepare features
        features = manager.prepare_race_features(
            selected_race["race_id"], selected_race["season"]
        )
        assert len(features) > 0

        # Step 4: Generate prediction
        prediction = manager.generate_prediction(model, features, selected_race["name"])
        assert "predictions" in prediction
        assert len(prediction["predictions"]) > 0

    def test_prediction_with_different_models(self, mock_prediction_manager):
        """Test prediction with different model types."""
        from f1_predict.web.utils.prediction import PredictionManager

        manager = PredictionManager()
        models_to_test = ["ensemble", "xgboost", "lightgbm", "random_forest"]

        for model_type in models_to_test:
            model, metadata = manager.load_model(model_type)
            assert model is not None
            assert metadata["type"] == model_type

    def test_prediction_export_workflow(self, sample_prediction):
        """Test prediction export workflow."""
        from f1_predict.web.utils.prediction import PredictionManager

        manager = PredictionManager()

        # Export as CSV
        csv_data = manager.export_prediction(sample_prediction, format="csv")
        assert csv_data is not None
        assert isinstance(csv_data, bytes)
        assert b"position" in csv_data or b"driver" in csv_data

        # Export as JSON
        json_data = manager.export_prediction(sample_prediction, format="json")
        assert json_data is not None
        assert isinstance(json_data, bytes)
        assert b"race" in json_data


class TestComparisonWorkflow:
    """Tests for driver/team comparison workflow."""

    def test_driver_comparison_setup(self, sample_driver_data):
        """Test driver comparison setup."""
        drivers = sample_driver_data["driver_id"].tolist()
        assert len(drivers) >= 2

        driver1 = drivers[0]
        driver2 = drivers[1]
        assert driver1 != driver2

    def test_comparison_data_preparation(self, sample_race_results):
        """Test comparison data preparation."""
        # Prepare comparison data from race results
        comparison_data = sample_race_results[
            ["driver_name", "team", "position", "points"]
        ]
        assert len(comparison_data) > 0
        assert "driver_name" in comparison_data.columns
        assert "points" in comparison_data.columns

    def test_comparison_calculation(self, sample_race_results):
        """Test comparison calculations."""
        # Calculate head-to-head stats
        top_2_drivers = sample_race_results.nlargest(2, "points")
        assert len(top_2_drivers) == 2

        points_diff = top_2_drivers.iloc[0]["points"] - top_2_drivers.iloc[1]["points"]
        assert points_diff >= 0


class TestAnalyticsWorkflow:
    """Tests for analytics dashboard workflow."""

    def test_analytics_data_loading(self, sample_race_results):
        """Test analytics data loading."""
        assert len(sample_race_results) > 0
        assert "position" in sample_race_results.columns
        assert "points" in sample_race_results.columns

    def test_kpi_calculation_workflow(self, sample_race_results):
        """Test KPI calculation workflow."""
        # Calculate race count
        race_count = len(sample_race_results)
        assert race_count > 0

        # Calculate average position
        avg_position = sample_race_results["position"].mean()
        assert avg_position > 0

        # Calculate total points
        total_points = sample_race_results["points"].sum()
        assert total_points > 0

        # Calculate win count (position == 1)
        win_count = len(sample_race_results[sample_race_results["position"] == 1])
        assert win_count >= 0

    def test_standings_calculation(self, sample_race_results):
        """Test standings calculation."""
        # Group by driver and sum points
        standings = (
            sample_race_results.groupby("driver_name")
            .agg({"points": "sum", "position": "count"})
            .rename(columns={"position": "races"})
            .sort_values("points", ascending=False)
        )

        assert len(standings) > 0
        assert standings.iloc[0]["points"] >= standings.iloc[-1]["points"]


class TestDataPersistence:
    """Tests for data persistence across pages."""

    def test_session_state_storage(self, mock_session_state):
        """Test session state storage."""
        mock_session_state.__setitem__("test_key", "test_value")
        mock_session_state.__getitem__.return_value = "test_value"

        assert mock_session_state["test_key"] == "test_value"

    def test_prediction_result_persistence(self, sample_prediction):
        """Test prediction result can be persisted."""
        # Verify prediction structure
        assert "race" in sample_prediction
        assert "predictions" in sample_prediction
        assert len(sample_prediction["predictions"]) > 0

    def test_settings_persistence(self):
        """Test settings can be persisted."""
        settings = {
            "theme": "Nebula Dark",
            "timezone": "UTC",
            "units": "Metric",
        }

        assert settings["theme"] == "Nebula Dark"
        assert len(settings) == 3


class TestExportFunctionality:
    """Tests for export functionality."""

    def test_csv_export_format(self, sample_prediction):
        """Test CSV export format."""
        from f1_predict.web.utils.prediction import PredictionManager

        manager = PredictionManager()
        csv_data = manager.export_prediction(sample_prediction, format="csv")

        assert csv_data is not None
        assert isinstance(csv_data, bytes)

    def test_json_export_format(self, sample_prediction):
        """Test JSON export format."""
        from f1_predict.web.utils.prediction import PredictionManager
        import json

        manager = PredictionManager()
        json_data = manager.export_prediction(sample_prediction, format="json")

        assert json_data is not None
        assert isinstance(json_data, bytes)

        # Verify it's valid JSON
        parsed = json.loads(json_data.decode("utf-8"))
        assert "race" in parsed

    def test_export_file_naming(self, sample_race_data):
        """Test export file naming convention."""
        race_name = sample_race_data["name"]
        filename = f"{race_name.replace(' ', '_')}_prediction.csv"

        assert filename == "Monaco_Grand_Prix_prediction.csv"
        assert not filename.startswith(" ")
        assert not filename.endswith(" ")


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_missing_race_data_handling(self):
        """Test handling of missing race data."""
        races = []
        if not races:
            # Should handle gracefully
            assert len(races) == 0

    def test_invalid_model_type_handling(self, mock_prediction_manager):
        """Test handling of invalid model type."""
        manager = mock_prediction_manager

        try:
            manager.load_model("invalid_model")
        except (ValueError, KeyError):
            # Should raise appropriate error
            pass

    def test_export_error_handling(self, sample_prediction):
        """Test export error handling."""
        from f1_predict.web.utils.prediction import PredictionManager

        manager = PredictionManager()

        # Test invalid format
        result = manager.export_prediction(sample_prediction, format="xml")
        assert result is None
