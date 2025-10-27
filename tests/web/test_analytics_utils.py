"""Tests for analytics utilities."""
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch

from f1_predict.web.utils.analytics import (
    calculate_kpi_metrics,
    get_championship_standings,
    calculate_driver_performance,
)


def test_calculate_kpi_metrics_returns_dict(mock_kpi_metrics):
    """Test that calculate_kpi_metrics returns KPI dictionary."""
    result = calculate_kpi_metrics()

    assert isinstance(result, dict)
    assert 'races_analyzed' in result
    assert 'prediction_accuracy' in result
    assert 'avg_confidence' in result


def test_get_championship_standings_returns_dataframe(mock_championship_standings):
    """Test that get_championship_standings returns DataFrame."""
    result = get_championship_standings()

    assert isinstance(result, pd.DataFrame)
    assert 'driver' in result.columns
    assert 'points' in result.columns


def test_calculate_driver_performance_returns_metrics():
    """Test that calculate_driver_performance returns performance metrics."""
    driver = "Verstappen"

    result = calculate_driver_performance(driver)

    assert isinstance(result, dict)
    assert 'wins' in result
    assert 'podiums' in result
