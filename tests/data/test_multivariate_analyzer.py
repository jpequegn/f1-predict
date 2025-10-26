"""Tests for multivariate anomaly detection using Isolation Forest."""

import pandas as pd
import pytest

from f1_predict.data.multivariate_analyzer import MultivariateAnalyzer


def test_analyzer_initialization():
    """Test MultivariateAnalyzer initializes with correct defaults."""
    analyzer = MultivariateAnalyzer()

    assert analyzer is not None
    assert analyzer.contamination == 0.1
    assert analyzer.n_estimators == 100
    assert analyzer.random_state == 42
    assert analyzer.is_fitted is False


def test_analyzer_custom_initialization():
    """Test MultivariateAnalyzer with custom parameters."""
    analyzer = MultivariateAnalyzer(
        contamination=0.05, n_estimators=200, random_state=123
    )

    assert analyzer.contamination == 0.05
    assert analyzer.n_estimators == 200
    assert analyzer.random_state == 123
    assert analyzer.is_fitted is False


def test_isolation_forest_detection():
    """Test Isolation Forest detects multivariate anomalies."""
    analyzer = MultivariateAnalyzer(contamination=0.1, random_state=42)

    # Create dataset with clear multivariate outliers
    # Normal cluster around (50, 50), outliers far away
    normal_data = pd.DataFrame(
        {
            "feature1": [50, 52, 48, 51, 49, 50, 51, 49, 52, 48] * 5,  # 50 points
            "feature2": [50, 51, 49, 52, 48, 50, 49, 51, 50, 52] * 5,
        }
    )

    # Add 5 clear outliers (10% contamination)
    outlier_data = pd.DataFrame(
        {
            "feature1": [200, 300, -100, 250, -50],
            "feature2": [200, 300, -100, 250, -50],
        }
    )

    data = pd.concat([normal_data, outlier_data], ignore_index=True)

    # Fit and detect
    analyzer.fit(data)
    result = analyzer.detect(data)

    # Check columns added
    assert "anomaly_flag" in result.columns
    assert "anomaly_score" in result.columns
    assert "anomaly_method" in result.columns

    # Check that some anomalies detected
    assert result["anomaly_flag"].sum() > 0

    # Check anomaly_method is set correctly
    anomaly_rows = result[result["anomaly_flag"]]
    assert all(anomaly_rows["anomaly_method"] == "isolation_forest")

    # Check scores are normalized to 0-1
    assert (result["anomaly_score"] >= 0).all()
    assert (result["anomaly_score"] <= 1).all()


def test_fit_and_predict():
    """Test fit() and detect() workflow."""
    analyzer = MultivariateAnalyzer(random_state=42)

    # Create training data
    train_data = pd.DataFrame(
        {
            "lap_time": [90.5, 91.0, 90.8, 91.2, 90.9, 91.1, 90.7, 90.6, 91.3, 90.4],
            "tire_age": [5, 6, 5, 7, 5, 6, 5, 6, 7, 5],
        }
    )

    # Should not be fitted initially
    assert analyzer.is_fitted is False

    # Fit the model
    analyzer.fit(train_data)

    # Should be fitted after fit()
    assert analyzer.is_fitted is True

    # Create test data with outlier
    test_data = pd.DataFrame(
        {
            "lap_time": [90.8, 91.0, 150.0],  # 150.0 is outlier
            "tire_age": [5, 6, 50],  # 50 is outlier
        }
    )

    # Detect anomalies
    result = analyzer.detect(test_data)

    # Should detect the outlier
    assert result.iloc[2]["anomaly_flag"] == True
    assert result.iloc[2]["anomaly_score"] > 0


def test_handles_missing_data():
    """Test analyzer handles missing values by filling with mean."""
    analyzer = MultivariateAnalyzer(random_state=42)

    # Create data with missing values
    data = pd.DataFrame(
        {
            "feature1": [10, 11, None, 10, 12, 9, 11, 10, 9, 10],
            "feature2": [50, 51, 49, None, 48, 50, 49, 51, 50, 52],
        }
    )

    # Should not crash
    analyzer.fit(data)
    result = analyzer.detect(data)

    # Should return same number of rows
    assert len(result) == len(data)
    assert "anomaly_flag" in result.columns


def test_detect_before_fit_raises_error():
    """Test that calling detect() before fit() raises ValueError."""
    analyzer = MultivariateAnalyzer()

    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
        }
    )

    with pytest.raises(ValueError, match="must be fitted"):
        analyzer.detect(data)


def test_empty_dataframe():
    """Test analyzer with empty DataFrame."""
    analyzer = MultivariateAnalyzer()

    data = pd.DataFrame()

    # fit() with empty data should raise ValueError
    with pytest.raises(ValueError, match="empty"):
        analyzer.fit(data)


def test_single_row_dataframe():
    """Test analyzer with single row (insufficient for Isolation Forest)."""
    analyzer = MultivariateAnalyzer()

    data = pd.DataFrame(
        {
            "feature1": [10],
            "feature2": [20],
        }
    )

    # Should raise ValueError - not enough samples
    with pytest.raises(ValueError, match="at least 2 samples"):
        analyzer.fit(data)


def test_single_feature_column():
    """Test analyzer with only one numeric column."""
    analyzer = MultivariateAnalyzer(random_state=42)

    data = pd.DataFrame(
        {
            "feature1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100],
        }
    )

    # Should work with single feature
    analyzer.fit(data)
    result = analyzer.detect(data)

    assert len(result) == len(data)
    assert "anomaly_flag" in result.columns


def test_non_numeric_columns_ignored():
    """Test that non-numeric columns are ignored."""
    analyzer = MultivariateAnalyzer(random_state=42)

    data = pd.DataFrame(
        {
            "driver": ["Hamilton", "Verstappen", "Leclerc"] * 5,
            "team": ["Mercedes", "Red Bull", "Ferrari"] * 5,
            "feature1": list(range(15)),
            "feature2": list(range(15, 30)),
        }
    )

    # Should ignore string columns and work with numeric ones
    analyzer.fit(data)
    result = analyzer.detect(data)

    assert len(result) == len(data)
    assert "driver" in result.columns  # Original columns preserved
    assert "team" in result.columns


def test_score_normalization():
    """Test that anomaly scores are normalized to 0-1 range."""
    analyzer = MultivariateAnalyzer(random_state=42)

    # Create data with varying degrees of anomalies
    data = pd.DataFrame(
        {
            "feature1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100, 200],
            "feature2": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 300, 400],
        }
    )

    analyzer.fit(data)
    result = analyzer.detect(data)

    # All scores should be in 0-1 range
    assert (result["anomaly_score"] >= 0).all()
    assert (result["anomaly_score"] <= 1).all()

    # More extreme outliers should have higher scores
    assert result.iloc[-1]["anomaly_score"] > result.iloc[0]["anomaly_score"]


def test_refit_updates_model():
    """Test that refitting updates the model."""
    analyzer = MultivariateAnalyzer(random_state=42)

    # First fit
    data1 = pd.DataFrame(
        {
            "feature1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "feature2": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        }
    )

    analyzer.fit(data1)
    assert analyzer.is_fitted is True

    # Refit with different data
    data2 = pd.DataFrame(
        {
            "feature1": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "feature2": [200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
        }
    )

    analyzer.fit(data2)
    assert analyzer.is_fitted is True

    # Model should work with new data
    result = analyzer.detect(data2)
    assert len(result) == len(data2)


def test_contamination_affects_detection():
    """Test that contamination parameter affects number of anomalies detected."""
    # Higher contamination = more anomalies expected
    analyzer_high = MultivariateAnalyzer(contamination=0.3, random_state=42)
    analyzer_low = MultivariateAnalyzer(contamination=0.05, random_state=42)

    data = pd.DataFrame(
        {
            "feature1": list(range(50)),
            "feature2": list(range(50, 100)),
        }
    )

    # Fit both
    analyzer_high.fit(data)
    analyzer_low.fit(data)

    # Detect
    result_high = analyzer_high.detect(data)
    result_low = analyzer_low.detect(data)

    # Higher contamination should flag more anomalies
    # Note: This is probabilistic, but should generally hold
    high_count = result_high["anomaly_flag"].sum()
    low_count = result_low["anomaly_flag"].sum()

    # At minimum, high contamination should detect at least as many as low
    assert high_count >= low_count
