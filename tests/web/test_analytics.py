"""Unit tests for web analytics module.

Tests cover:
- KPI calculations
- Championship standings
- Performance metrics
- Data quality assessment
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from f1_predict.web.utils.analytics import AnalyticsCalculator


class TestAnalyticsCalculatorKPIs:
    """Tests for KPI calculation."""

    @pytest.fixture
    def sample_race_data(self):
        """Create sample race data."""
        return pd.DataFrame({
            "race_id": [1, 1, 1, 2, 2, 2],
            "driver_id": ["driver_1", "driver_2", "driver_3", "driver_1", "driver_2", "driver_3"],
            "position": [1, 2, 3, 1, 3, 2],
            "points": [25, 18, 15, 25, 15, 18],
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "confidence_score": [0.95, 0.87, 0.75, 0.92, 0.88, 0.79],
        })

    def test_calculate_kpis_basic(self, sample_race_data):
        """Test basic KPI calculation."""
        kpis = AnalyticsCalculator.calculate_kpis(sample_race_data)

        assert "total_races" in kpis
        assert "prediction_accuracy" in kpis
        assert "avg_confidence" in kpis
        assert "data_quality_score" in kpis
        assert "last_sync" in kpis

    def test_total_races_count(self, sample_race_data):
        """Test total races calculation."""
        kpis = AnalyticsCalculator.calculate_kpis(sample_race_data)
        assert kpis["total_races"] == 2

    def test_avg_confidence(self, sample_race_data):
        """Test average confidence calculation."""
        kpis = AnalyticsCalculator.calculate_kpis(sample_race_data)
        expected_avg = np.mean([0.95, 0.87, 0.75, 0.92, 0.88, 0.79])
        assert abs(kpis["avg_confidence"] - expected_avg) < 0.01

    def test_data_quality_score(self, sample_race_data):
        """Test data quality score calculation."""
        kpis = AnalyticsCalculator.calculate_kpis(sample_race_data)
        assert 0 <= kpis["data_quality_score"] <= 100

    def test_last_sync_timestamp(self, sample_race_data):
        """Test last sync timestamp extraction."""
        kpis = AnalyticsCalculator.calculate_kpis(sample_race_data)
        assert kpis["last_sync"] is not None
        assert isinstance(kpis["last_sync"], pd.Timestamp)

    def test_kpis_empty_dataframe(self):
        """Test KPI calculation with empty dataframe."""
        empty_df = pd.DataFrame()
        kpis = AnalyticsCalculator.calculate_kpis(empty_df)

        assert kpis["total_races"] == 0
        assert kpis["prediction_accuracy"] == 0.0
        assert kpis["avg_confidence"] == 0.0
        assert kpis["data_quality_score"] == 0.0
        assert kpis["last_sync"] is None

    def test_kpis_with_missing_columns(self):
        """Test KPI calculation with missing columns."""
        df = pd.DataFrame({
            "position": [1, 2, 3],
            "points": [25, 18, 15],
        })
        kpis = AnalyticsCalculator.calculate_kpis(df)

        assert "total_races" in kpis
        assert kpis["total_races"] == 0  # No race_id column


class TestAnalyticsDataQuality:
    """Tests for data quality assessment."""

    def test_data_quality_perfect(self):
        """Test data quality with perfect (no missing) data."""
        df = pd.DataFrame({
            "race_id": [1, 1, 1],
            "position": [1, 2, 3],
            "points": [25, 18, 15],
            "date": pd.date_range("2024-01-01", periods=3),
        })
        kpis = AnalyticsCalculator.calculate_kpis(df)
        assert kpis["data_quality_score"] == 100.0

    def test_data_quality_with_missing(self):
        """Test data quality with missing values."""
        df = pd.DataFrame({
            "race_id": [1, 1, None],
            "position": [1, 2, 3],
            "points": [25, 18, 15],
            "date": pd.date_range("2024-01-01", periods=3),
        })
        kpis = AnalyticsCalculator.calculate_kpis(df)
        assert kpis["data_quality_score"] < 100.0
        assert kpis["data_quality_score"] > 0.0


class TestAnalyticsPerformanceMetrics:
    """Tests for performance metrics calculation."""

    @pytest.fixture
    def multi_race_data(self):
        """Create multi-race sample data."""
        data_list = []
        for race_id in range(1, 6):
            for position in [1, 2, 3, 4, 5]:
                data_list.append({
                    "race_id": race_id,
                    "driver_id": f"driver_{position}",
                    "position": position,
                    "points": max(0, 26 - (position * 5)),
                    "date": datetime(2024, 1, 1) + timedelta(days=race_id * 7),
                    "confidence_score": 0.5 + (position * 0.05),
                })
        return pd.DataFrame(data_list)

    def test_multi_race_kpis(self, multi_race_data):
        """Test KPI calculation with multiple races."""
        kpis = AnalyticsCalculator.calculate_kpis(multi_race_data)
        assert kpis["total_races"] == 5
        assert kpis["avg_confidence"] > 0.5

    def test_consistency_metric(self, multi_race_data):
        """Test consistency of KPI calculations."""
        kpis1 = AnalyticsCalculator.calculate_kpis(multi_race_data)
        kpis2 = AnalyticsCalculator.calculate_kpis(multi_race_data)

        assert kpis1["total_races"] == kpis2["total_races"]
        assert kpis1["avg_confidence"] == kpis2["avg_confidence"]


class TestAnalyticsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_race(self):
        """Test with single race data."""
        df = pd.DataFrame({
            "race_id": [1, 1, 1],
            "driver_id": ["driver_1", "driver_2", "driver_3"],
            "position": [1, 2, 3],
            "points": [25, 18, 15],
            "date": [datetime(2024, 1, 1)] * 3,
            "confidence_score": [0.95, 0.88, 0.75],
        })
        kpis = AnalyticsCalculator.calculate_kpis(df)
        assert kpis["total_races"] == 1

    def test_single_driver_multiple_races(self):
        """Test with single driver across multiple races."""
        data = []
        for race_id in range(1, 6):
            data.append({
                "race_id": race_id,
                "driver_id": "driver_1",
                "position": (race_id % 5) + 1,
                "points": 25 - (race_id % 5) * 5,
                "date": datetime(2024, 1, 1) + timedelta(days=race_id),
                "confidence_score": 0.9 - (race_id * 0.01),
            })
        df = pd.DataFrame(data)
        kpis = AnalyticsCalculator.calculate_kpis(df)
        assert kpis["total_races"] == 5

    def test_high_confidence_data(self):
        """Test with high confidence predictions."""
        df = pd.DataFrame({
            "race_id": [1, 1, 1],
            "position": [1, 2, 3],
            "points": [25, 18, 15],
            "date": pd.date_range("2024-01-01", periods=3),
            "confidence_score": [0.99, 0.98, 0.97],
        })
        kpis = AnalyticsCalculator.calculate_kpis(df)
        assert kpis["avg_confidence"] > 0.97

    def test_low_confidence_data(self):
        """Test with low confidence predictions."""
        df = pd.DataFrame({
            "race_id": [1, 1, 1],
            "position": [1, 2, 3],
            "points": [25, 18, 15],
            "date": pd.date_range("2024-01-01", periods=3),
            "confidence_score": [0.51, 0.52, 0.53],
        })
        kpis = AnalyticsCalculator.calculate_kpis(df)
        assert 0.5 < kpis["avg_confidence"] < 0.6


class TestAnalyticsDataTypes:
    """Tests for data type handling."""

    def test_date_column_string(self):
        """Test with date column as string."""
        df = pd.DataFrame({
            "race_id": [1, 1, 1],
            "position": [1, 2, 3],
            "points": [25, 18, 15],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "confidence_score": [0.95, 0.88, 0.75],
        })
        kpis = AnalyticsCalculator.calculate_kpis(df)
        assert kpis["last_sync"] is not None

    def test_confidence_as_integer(self):
        """Test confidence score as integer values."""
        df = pd.DataFrame({
            "race_id": [1, 1, 1],
            "position": [1, 2, 3],
            "points": [25, 18, 15],
            "date": pd.date_range("2024-01-01", periods=3),
            "confidence_score": [1, 1, 1],  # As integers
        })
        kpis = AnalyticsCalculator.calculate_kpis(df)
        assert kpis["avg_confidence"] == 1.0

    def test_numeric_race_id(self):
        """Test race_id as numeric values."""
        df = pd.DataFrame({
            "race_id": [1, 2, 3],
            "position": [1, 2, 3],
            "points": [25, 18, 15],
            "date": pd.date_range("2024-01-01", periods=3),
        })
        kpis = AnalyticsCalculator.calculate_kpis(df)
        assert kpis["total_races"] == 3
