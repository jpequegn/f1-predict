"""Data profiling and exploratory data analysis utilities.

Provides comprehensive data profiling for F1 datasets including:
- Descriptive statistics (mean, std, percentiles, skewness, kurtosis)
- Distribution analysis (histograms, normality tests)
- Correlation analysis and multivariate relationships
- Categorical variable profiling
- Missing value patterns
- Outlier summary
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog
from scipy import stats

# Type aliases for cleaner type hints
NDArray = np.ndarray[Any, np.dtype[np.floating[Any]]]

logger = structlog.get_logger(__name__)


class DataType(Enum):
    """Data type classification."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    OTHER = "other"


@dataclass
class NumericProfile:
    """Profile statistics for numeric variables."""

    column_name: str
    count: int
    missing_count: int
    missing_pct: float
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    cv: float  # Coefficient of variation
    range_val: float
    outlier_count: int
    outlier_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column_name,
            "count": int(self.count),
            "missing": int(self.missing_count),
            "missing_pct": round(self.missing_pct, 2),
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "std": round(self.std, 6),
            "min": round(self.min_val, 6),
            "max": round(self.max_val, 6),
            "q1": round(self.q1, 6),
            "q3": round(self.q3, 6),
            "iqr": round(self.iqr, 6),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "cv": round(self.cv, 4),
            "range": round(self.range_val, 6),
            "outliers": int(self.outlier_count),
            "outlier_pct": round(self.outlier_pct, 2),
        }


@dataclass
class CategoricalProfile:
    """Profile statistics for categorical variables."""

    column_name: str
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    mode: Optional[str]
    mode_freq: int
    mode_pct: float
    top_values: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column_name,
            "count": int(self.count),
            "missing": int(self.missing_count),
            "missing_pct": round(self.missing_pct, 2),
            "unique": int(self.unique_count),
            "unique_pct": round(self.unique_pct, 2),
            "mode": self.mode,
            "mode_freq": int(self.mode_freq),
            "mode_pct": round(self.mode_pct, 2),
            "top_values": dict(sorted(self.top_values.items(), key=lambda x: x[1], reverse=True)[:10]),
        }


@dataclass
class DatasetProfile:
    """Overall dataset profile."""

    name: str
    shape: tuple[int, int]  # (rows, columns)
    memory_usage: int
    missing_pct: float
    duplicate_rows: int
    numeric_profiles: dict[str, NumericProfile] = field(default_factory=dict)
    categorical_profiles: dict[str, CategoricalProfile] = field(default_factory=dict)
    correlation_matrix: Optional[NDArray] = None
    correlation_columns: list[str] = field(default_factory=list)
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "rows": int(self.shape[0]),
            "columns": int(self.shape[1]),
            "memory_mb": round(self.memory_usage / 1024 / 1024, 2),
            "missing_pct": round(self.missing_pct, 2),
            "duplicate_rows": int(self.duplicate_rows),
            "numeric_columns": len(self.numeric_profiles),
            "categorical_columns": len(self.categorical_profiles),
            "created_at": self.created_at,
        }


class DataProfiler:
    """Comprehensive data profiling for exploratory data analysis."""

    def __init__(self, iqr_multiplier: float = 1.5):
        """Initialize data profiler.

        Args:
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
        """
        self.iqr_multiplier = iqr_multiplier
        self.logger = logger.bind(component="data_profiler")

    def profile_dataframe(self, df: pd.DataFrame, name: str = "dataset") -> DatasetProfile:
        """Generate comprehensive profile for a DataFrame.

        Args:
            df: DataFrame to profile
            name: Name for the profile

        Returns:
            DatasetProfile with complete analysis
        """
        import time

        try:
            start_time = time.time()

            # Basic info
            rows, cols = df.shape
            memory_usage = df.memory_usage(deep=True).sum()
            missing_pct = (df.isna().sum().sum() / (rows * cols)) * 100 if (rows * cols) > 0 else 0.0
            duplicate_rows = df.duplicated().sum()

            # Profile numeric columns
            numeric_profiles: dict[str, NumericProfile] = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                numeric_profiles[col] = self._profile_numeric_column(df[col])

            # Profile categorical columns
            categorical_profiles: dict[str, CategoricalProfile] = {}
            for col in df.select_dtypes(include=["object", "category"]).columns:
                categorical_profiles[col] = self._profile_categorical_column(df[col])

            # Correlation matrix for numeric columns
            correlation_matrix = None
            correlation_columns = []
            if numeric_profiles:
                numeric_df = df[list(numeric_profiles.keys())].dropna()
                if len(numeric_df) > 0:
                    correlation_matrix = numeric_df.corr().values
                    correlation_columns = list(numeric_df.columns)

            profile = DatasetProfile(
                name=name,
                shape=(rows, cols),
                memory_usage=memory_usage,
                missing_pct=missing_pct,
                duplicate_rows=duplicate_rows,
                numeric_profiles=numeric_profiles,
                categorical_profiles=categorical_profiles,
                correlation_matrix=correlation_matrix,
                correlation_columns=correlation_columns,
                created_at=time.time(),
            )

            elapsed = time.time() - start_time
            self.logger.info("dataset_profiled", name=name, rows=rows, cols=cols, elapsed_sec=round(elapsed, 2))

            return profile

        except Exception as e:
            self.logger.error("profile_dataframe_failed", error=str(e), name=name)
            raise

    def _profile_numeric_column(self, series: pd.Series) -> NumericProfile:
        """Profile a numeric column.

        Args:
            series: Series to profile

        Returns:
            NumericProfile with statistics
        """
        data = series.dropna()
        missing_count = series.isna().sum()
        missing_pct = (missing_count / len(series)) * 100 if len(series) > 0 else 0.0

        # Calculate statistics
        mean = float(data.mean()) if len(data) > 0 else 0.0
        median = float(data.median()) if len(data) > 0 else 0.0
        std = float(data.std()) if len(data) > 0 else 0.0
        min_val = float(data.min()) if len(data) > 0 else 0.0
        max_val = float(data.max()) if len(data) > 0 else 0.0

        q1 = float(data.quantile(0.25)) if len(data) > 0 else 0.0
        q3 = float(data.quantile(0.75)) if len(data) > 0 else 0.0
        iqr = q3 - q1

        # Distribution metrics
        skewness = float(stats.skew(data)) if len(data) > 0 else 0.0
        kurtosis = float(stats.kurtosis(data)) if len(data) > 0 else 0.0

        # Outlier detection using IQR
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_pct = (outlier_count / len(data)) * 100 if len(data) > 0 else 0.0

        # Coefficient of variation
        cv = (std / abs(mean)) if mean != 0 else 0.0
        range_val = max_val - min_val

        return NumericProfile(
            column_name=series.name or "unknown",
            count=len(data),
            missing_count=missing_count,
            missing_pct=missing_pct,
            mean=mean,
            median=median,
            std=std,
            min_val=min_val,
            max_val=max_val,
            q1=q1,
            q3=q3,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurtosis,
            cv=cv,
            range_val=range_val,
            outlier_count=outlier_count,
            outlier_pct=outlier_pct,
        )

    def _profile_categorical_column(self, series: pd.Series) -> CategoricalProfile:
        """Profile a categorical column.

        Args:
            series: Series to profile

        Returns:
            CategoricalProfile with statistics
        """
        missing_count = series.isna().sum()
        missing_pct = (missing_count / len(series)) * 100

        # Value counts
        value_counts = series.value_counts()
        unique_count = len(value_counts)
        unique_pct = (unique_count / len(series)) * 100

        # Mode
        mode = value_counts.index[0] if len(value_counts) > 0 else None
        mode_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
        mode_pct = (mode_freq / len(series)) * 100 if len(series) > 0 else 0.0

        # Top values
        top_values = value_counts.head(10).to_dict()

        return CategoricalProfile(
            column_name=series.name or "unknown",
            count=len(series) - missing_count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            mode=str(mode) if mode is not None else None,
            mode_freq=mode_freq,
            mode_pct=mode_pct,
            top_values=top_values,
        )

    def compare_profiles(self, profile1: DatasetProfile, profile2: DatasetProfile) -> dict[str, Any]:
        """Compare two dataset profiles.

        Args:
            profile1: First profile
            profile2: Second profile

        Returns:
            Dictionary with comparison metrics
        """
        new_columns: list[str] = []
        removed_columns: list[str] = []
        numeric_stat_changes: dict[str, dict[str, float]] = {}
        categorical_changes: dict[str, Any] = {}

        # Check for new/removed numeric columns
        prof1_numeric = set(profile1.numeric_profiles.keys())
        prof2_numeric = set(profile2.numeric_profiles.keys())
        new_columns.extend(list(prof2_numeric - prof1_numeric))
        removed_columns.extend(list(prof1_numeric - prof2_numeric))

        # Check for new/removed categorical columns
        prof1_cat = set(profile1.categorical_profiles.keys())
        prof2_cat = set(profile2.categorical_profiles.keys())
        new_columns.extend(list(prof2_cat - prof1_cat))
        removed_columns.extend(list(prof1_cat - prof2_cat))

        # Compare common numeric columns
        for col in prof1_numeric & prof2_numeric:
            p1 = profile1.numeric_profiles[col]
            p2 = profile2.numeric_profiles[col]
            numeric_stat_changes[col] = {
                "mean_change": round(p2.mean - p1.mean, 6),
                "std_change": round(p2.std - p1.std, 6),
                "missing_pct_change": round(p2.missing_pct - p1.missing_pct, 2),
            }

        changes: dict[str, Any] = {
            "row_count_diff": profile2.shape[0] - profile1.shape[0],
            "column_count_diff": profile2.shape[1] - profile1.shape[1],
            "missing_pct_change": profile2.missing_pct - profile1.missing_pct,
            "new_columns": new_columns,
            "removed_columns": removed_columns,
            "numeric_stat_changes": numeric_stat_changes,
            "categorical_changes": categorical_changes,
        }

        return changes
