"""Fixtures for integration tests."""

from pathlib import Path
from typing import Generator, Any
import tempfile

import pandas as pd
import pytest


@pytest.fixture
def integration_tmp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for integration test data.

    Yields:
        Temporary directory path
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_race_results() -> pd.DataFrame:
    """Sample race results data for testing.

    Returns:
        DataFrame with race result columns
    """
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "round": [1, 1, 1],
            "race_id": ["monaco_2024", "monaco_2024", "monaco_2024"],
            "driver_id": [1, 2, 3],
            "driver_name": ["Verstappen", "Hamilton", "Leclerc"],
            "constructor_id": [1, 2, 3],
            "team_id": [1, 1, 2],
            "team_name": ["Red Bull", "Mercedes", "Ferrari"],
            "position": [1, 2, 3],
            "points": [25, 18, 15],
            "qualifying_position": [1, 2, 3],
            "status": ["Finished", "Finished", "Finished"],
            "lap_time": [5400.123, 5405.456, 5410.789],
        }
    )


@pytest.fixture
def sample_qualifying_results() -> pd.DataFrame:
    """Sample qualifying results data for testing.

    Returns:
        DataFrame with qualifying result columns
    """
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "round": [1, 1, 1],
            "race_id": ["monaco_2024", "monaco_2024", "monaco_2024"],
            "driver_id": [1, 2, 3],
            "driver_name": ["Verstappen", "Hamilton", "Leclerc"],
            "constructor_id": [1, 2, 3],
            "team_id": [1, 1, 2],
            "position": [1, 2, 3],
            "q1_time": [73.456, 73.789, 74.123],
            "q2_time": [73.123, 73.456, 73.789],
            "q3_time": [72.890, 73.123, 73.456],
        }
    )


@pytest.fixture
def sample_race_schedule() -> pd.DataFrame:
    """Sample race schedule data for testing.

    Returns:
        DataFrame with race schedule columns
    """
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "round": [1, 2, 3],
            "race_name": ["Bahrain", "Saudi Arabia", "Monaco"],
            "circuit_id": [1, 2, 3],
            "circuit_name": ["Bahrain International", "Jeddah", "Monte-Carlo"],
            "date": ["2024-02-25", "2024-03-03", "2024-03-24"],
        }
    )


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Sample engineered features for testing.

    Returns:
        DataFrame with feature columns
    """
    return pd.DataFrame(
        {
            "race_id": ["monaco_2024", "monaco_2024", "monaco_2024"],
            "driver_id": [1, 2, 3],
            "driver_form_score": [85.5, 78.3, 72.1],
            "team_reliability": [0.95, 0.92, 0.88],
            "qualifying_edge": [0.1, -0.05, -0.15],
            "circuit_experience": [8, 10, 12],
            "weather_adjustment": [0.0, -0.05, 0.1],
            "historical_win_rate": [0.35, 0.25, 0.15],
        }
    )


@pytest.fixture
def sample_predictions() -> pd.DataFrame:
    """Sample model predictions for testing.

    Returns:
        DataFrame with prediction columns
    """
    return pd.DataFrame(
        {
            "race_id": ["monaco_2024", "monaco_2024", "monaco_2024"],
            "driver_id": [1, 2, 3],
            "driver_name": ["Verstappen", "Hamilton", "Leclerc"],
            "predicted_position": [1, 2, 3],
            "confidence": [92.5, 78.3, 65.2],
            "win_probability": [0.92, 0.05, 0.03],
            "podium_probability": [0.99, 0.85, 0.70],
        }
    )


@pytest.fixture
def integration_data_dir(integration_tmp_dir: Path) -> Path:
    """Create data directory structure for integration tests.

    Args:
        integration_tmp_dir: Temporary directory fixture

    Returns:
        Path to data directory with subdirectories
    """
    data_dir = integration_tmp_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)

    return data_dir


@pytest.fixture
def mock_models_dir(integration_tmp_dir: Path) -> Path:
    """Create models directory for integration tests.

    Args:
        integration_tmp_dir: Temporary directory fixture

    Returns:
        Path to models directory
    """
    models_dir = integration_tmp_dir / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


class MockModel:
    """Mock ML model for testing without requiring trained models."""

    def __init__(self, seed: int = 42):
        """Initialize mock model.

        Args:
            seed: Random seed for reproducibility
        """
        import numpy as np
        self.seed = seed
        np.random.seed(seed)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate mock predictions.

        Args:
            features: Input features DataFrame

        Returns:
            Predictions DataFrame
        """
        import numpy as np
        n_samples = len(features)

        return pd.DataFrame(
            {
                "predicted_position": np.random.randint(1, 21, n_samples),
                "confidence": np.random.uniform(50, 99, n_samples),
                "win_probability": np.random.uniform(0, 1, n_samples),
            }
        )

    def predict_proba(self, features: pd.DataFrame) -> dict:
        """Generate mock probability predictions.

        Args:
            features: Input features DataFrame

        Returns:
            Dictionary with probability distributions
        """
        import numpy as np
        n_samples = len(features)

        return {
            "podium_proba": np.random.uniform(0.3, 1.0, n_samples),
            "top10_proba": np.random.uniform(0.6, 1.0, n_samples),
        }


@pytest.fixture
def mock_model() -> MockModel:
    """Provide mock ML model for integration tests.

    Returns:
        MockModel instance
    """
    return MockModel()


@pytest.fixture
def temp_cost_tracker_db() -> Generator[Path, None, None]:
    """Provide a temporary database path for cost tracker testing.

    Yields:
        Temporary database path
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_llm_usage.db"
        yield db_path


# Pytest configuration for integration tests
def pytest_configure(config):
    """Register custom markers for integration tests."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance benchmark"
    )
