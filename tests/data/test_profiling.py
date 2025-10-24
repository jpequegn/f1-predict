"""Tests for data profiling module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1_predict.data.profiling import DataProfiler, DatasetProfile, NumericProfile, CategoricalProfile


class TestNumericProfile:
    """Test numeric profile statistics."""

    def test_numeric_profile_creation(self):
        """Test creating numeric profile."""
        profile = NumericProfile(
            column_name="test_col",
            count=100,
            missing_count=5,
            missing_pct=5.0,
            mean=50.0,
            median=48.0,
            std=10.0,
            min_val=10.0,
            max_val=90.0,
            q1=40.0,
            q3=60.0,
            iqr=20.0,
            skewness=0.5,
            kurtosis=1.2,
            cv=0.2,
            range_val=80.0,
            outlier_count=3,
            outlier_pct=3.0,
        )

        assert profile.column_name == "test_col"
        assert profile.count == 100
        assert profile.mean == 50.0

    def test_numeric_profile_to_dict(self):
        """Test converting numeric profile to dictionary."""
        profile = NumericProfile(
            column_name="test",
            count=50,
            missing_count=0,
            missing_pct=0.0,
            mean=25.0,
            median=24.0,
            std=5.0,
            min_val=10.0,
            max_val=40.0,
            q1=20.0,
            q3=30.0,
            iqr=10.0,
            skewness=0.1,
            kurtosis=0.5,
            cv=0.2,
            range_val=30.0,
            outlier_count=1,
            outlier_pct=2.0,
        )

        d = profile.to_dict()
        assert d["column"] == "test"
        assert d["count"] == 50
        assert d["mean"] == 25.0
        assert "skewness" in d


class TestCategoricalProfile:
    """Test categorical profile statistics."""

    def test_categorical_profile_creation(self):
        """Test creating categorical profile."""
        profile = CategoricalProfile(
            column_name="test_cat",
            count=100,
            missing_count=5,
            missing_pct=5.0,
            unique_count=10,
            unique_pct=10.0,
            mode="A",
            mode_freq=50,
            mode_pct=50.0,
        )

        assert profile.column_name == "test_cat"
        assert profile.unique_count == 10
        assert profile.mode == "A"

    def test_categorical_profile_to_dict(self):
        """Test converting categorical profile to dictionary."""
        profile = CategoricalProfile(
            column_name="test",
            count=50,
            missing_count=0,
            missing_pct=0.0,
            unique_count=5,
            unique_pct=10.0,
            mode="X",
            mode_freq=30,
            mode_pct=60.0,
            top_values={"X": 30, "Y": 15, "Z": 5},
        )

        d = profile.to_dict()
        assert d["unique"] == 5
        assert d["mode"] == "X"
        assert "top_values" in d


class TestDataProfiler:
    """Test data profiling functionality."""

    @pytest.fixture
    def profiler(self):
        """Create profiler instance."""
        return DataProfiler()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for profiling."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "id": range(1, 101),
                "value": np.random.normal(50, 10, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
                "score": np.random.uniform(0, 100, 100),
            }
        )

    def test_profiler_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.iqr_multiplier == 1.5

    def test_profile_numeric_column(self, profiler, sample_dataframe):
        """Test profiling numeric column."""
        profile = profiler._profile_numeric_column(sample_dataframe["value"])

        assert profile.column_name == "value"
        assert profile.count > 0
        assert profile.mean is not None
        assert profile.std is not None
        assert profile.skewness is not None
        assert profile.kurtosis is not None

    def test_profile_categorical_column(self, profiler, sample_dataframe):
        """Test profiling categorical column."""
        profile = profiler._profile_categorical_column(sample_dataframe["category"])

        assert profile.column_name == "category"
        assert profile.unique_count > 0
        assert profile.mode is not None
        assert profile.mode_freq > 0

    def test_profile_dataframe(self, profiler, sample_dataframe):
        """Test profiling entire DataFrame."""
        profile = profiler.profile_dataframe(sample_dataframe, name="test_data")

        assert profile.name == "test_data"
        assert profile.shape == (100, 4)
        assert len(profile.numeric_profiles) == 3  # id, value, score
        assert len(profile.categorical_profiles) == 1  # category
        assert profile.correlation_matrix is not None

    def test_profile_with_missing_values(self, profiler):
        """Test profiling with missing values."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4, 5],
                "col2": ["a", None, "c", "d", "e"],
            }
        )

        profile = profiler.profile_dataframe(df)

        numeric_profile = profile.numeric_profiles["col1"]
        assert numeric_profile.missing_count == 1
        assert numeric_profile.missing_pct == 20.0

        categorical_profile = profile.categorical_profiles["col2"]
        assert categorical_profile.missing_count == 1
        assert categorical_profile.missing_pct == 20.0

    def test_profile_with_outliers(self, profiler):
        """Test outlier detection in profiling."""
        data = [1, 2, 3, 4, 5] + [1000]  # Last value is outlier
        df = pd.DataFrame({"values": data})

        profile = profiler.profile_dataframe(df)
        numeric_profile = profile.numeric_profiles["values"]

        assert numeric_profile.outlier_count > 0
        assert numeric_profile.outlier_pct > 0

    def test_profile_empty_dataframe(self, profiler):
        """Test profiling empty DataFrame."""
        df = pd.DataFrame({"col1": [], "col2": []})

        profile = profiler.profile_dataframe(df)
        assert profile.shape[0] == 0

    def test_profile_single_column(self, profiler):
        """Test profiling single column DataFrame."""
        df = pd.DataFrame({"single": [1, 2, 3, 4, 5]})

        profile = profiler.profile_dataframe(df)
        assert len(profile.numeric_profiles) == 1
        assert len(profile.categorical_profiles) == 0

    def test_dataset_profile_to_dict(self, profiler, sample_dataframe):
        """Test converting dataset profile to dictionary."""
        profile = profiler.profile_dataframe(sample_dataframe)
        d = profile.to_dict()

        assert d["rows"] == 100
        assert d["columns"] == 4
        assert "memory_mb" in d
        assert "numeric_columns" in d
        assert "categorical_columns" in d

    def test_compare_profiles(self, profiler):
        """Test comparing two profiles."""
        df1 = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": ["x", "y", "z", "x", "y"],
            }
        )

        df2 = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["x", "y", "z", "x", "y", "w"],
                "c": [10, 20, 30, 40, 50, 60],
            }
        )

        profile1 = profiler.profile_dataframe(df1, name="df1")
        profile2 = profiler.profile_dataframe(df2, name="df2")

        comparison = profiler.compare_profiles(profile1, profile2)

        assert comparison["row_count_diff"] == 1
        assert comparison["column_count_diff"] == 1
        assert "c" in comparison["new_columns"]
        assert "a" in comparison["numeric_stat_changes"]

    def test_numeric_profile_with_all_nan(self, profiler):
        """Test numeric profiling with all NaN values."""
        series = pd.Series([np.nan, np.nan, np.nan], name="all_nan")
        profile = profiler._profile_numeric_column(series)

        assert profile.count == 0
        assert profile.missing_pct == 100.0
        assert profile.mean == 0.0

    def test_categorical_profile_single_value(self, profiler):
        """Test categorical profiling with single repeated value."""
        series = pd.Series(["A", "A", "A", "A"], name="constant")
        profile = profiler._profile_categorical_column(series)

        assert profile.unique_count == 1
        assert profile.mode == "A"
        assert profile.mode_pct == 100.0

    def test_coefficient_of_variation(self, profiler):
        """Test coefficient of variation calculation."""
        series = pd.Series([1, 2, 3, 4, 5], name="cv_test")
        profile = profiler._profile_numeric_column(series)

        # CV = std / mean
        assert profile.cv > 0
        assert 0 < profile.cv < 1  # For this data, cv should be moderate


class TestDatasetProfile:
    """Test dataset profile structure."""

    def test_dataset_profile_creation(self):
        """Test creating dataset profile."""
        profile = DatasetProfile(
            name="test",
            shape=(100, 5),
            memory_usage=1000,
            missing_pct=2.5,
            duplicate_rows=3,
        )

        assert profile.name == "test"
        assert profile.shape == (100, 5)
        assert profile.duplicate_rows == 3

    def test_dataset_profile_with_correlations(self):
        """Test dataset profile with correlation matrix."""
        corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        profile = DatasetProfile(
            name="test",
            shape=(50, 2),
            memory_usage=500,
            missing_pct=0.0,
            duplicate_rows=0,
            correlation_matrix=corr_matrix,
            correlation_columns=["col1", "col2"],
        )

        assert profile.correlation_matrix is not None
        assert len(profile.correlation_columns) == 2


class TestDataProfilerIntegration:
    """Integration tests for data profiling."""

    def test_profile_race_results_like_data(self):
        """Test profiling F1 race-like data."""
        profiler = DataProfiler()

        df = pd.DataFrame(
            {
                "race_id": range(1, 21),
                "driver_id": np.random.randint(1, 20, 20),
                "position": np.random.randint(1, 21, 20),
                "points": np.random.choice([0, 1, 2, 4, 6, 8, 10, 12, 15, 18, 25], 20),
                "status": np.random.choice(["Finished", "DNF", "DSQ", "WD"], 20),
                "lap_time": np.random.uniform(60.0, 90.0, 20),
            }
        )

        profile = profiler.profile_dataframe(df, name="race_results")

        assert profile.shape[0] == 20
        assert profile.shape[1] == 6
        assert "position" in profile.numeric_profiles
        assert "status" in profile.categorical_profiles

    def test_profile_persistence(self):
        """Test saving and loading profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = DataProfiler()

            df = pd.DataFrame(
                {
                    "x": np.random.randn(100),
                    "y": np.random.choice(["A", "B"], 100),
                }
            )

            profile = profiler.profile_dataframe(df, name="test")
            profile_dict = profile.to_dict()

            # Save
            profile_file = Path(tmpdir) / "profile.json"
            import json

            with open(profile_file, "w") as f:
                json.dump(profile_dict, f)

            # Load
            with open(profile_file) as f:
                loaded = json.load(f)

            assert loaded["name"] == "test"
            assert loaded["rows"] == 100
            assert loaded["columns"] == 2
