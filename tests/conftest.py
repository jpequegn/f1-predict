"""Root-level pytest configuration and shared fixtures for all test phases.

This file consolidates fixtures needed across all test modules:
- Phase 1: Pytest infrastructure
- Phase 2: API client tests
- Phase 3: Data collection tests
- Phase 4: Feature engineering tests
- Phase 5: Model prediction tests
- Phase 6: Performance & CLI tests
"""

import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import structlog


# ============================================================================
# Phase 1: Pytest Infrastructure & Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers and configure test environment."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "ml: mark test as ML model test")
    config.addinivalue_line("markers", "data: mark test as data test")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "performance: mark test as performance")
    config.addinivalue_line("markers", "cli: mark test as CLI test")

    # Configure structlog for testing (avoid stdlib logger processors that expect 'name' attr)
    _configure_test_logging()

    # Configure PyTorch for test safety - disable threading/multiprocessing
    _configure_pytorch_for_testing()


def _configure_test_logging() -> None:
    """Configure structlog for test environment with compatible processors."""
    # Use simpler processors that don't require stdlib logger attributes
    processors = [
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=False,  # Disable caching in tests
    )


def _configure_pytorch_for_testing() -> None:
    """Configure PyTorch for safe test execution.

    Disables threading and multiprocessing to prevent segfaults when running
    many tests sequentially. PyTorch's threading can accumulate state across
    test runs, leading to crashes in torch/optim/adam.py.
    """
    try:
        import torch
        import os

        # Disable threading to prevent state accumulation
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # Disable OpenMP (used by PyTorch internals)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        # Force CPU only to avoid CUDA/thread issues
        torch.set_default_device("cpu")

    except ImportError:
        # PyTorch not installed, skip configuration
        pass


# ============================================================================
# Phase 2: API Client Fixtures
# ============================================================================


@pytest.fixture
def mock_http_response():
    """Mock successful HTTP 200 response.

    Returns:
        Mock response object with status_code=200 and JSON data
    """
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "MRData": {
            "RaceTable": {
                "Races": [
                    {
                        "season": "2024",
                        "round": "5",
                        "raceName": "Monaco Grand Prix",
                        "Circuit": {
                            "circuitId": "monaco",
                            "circuitName": "Circuit de Monaco",
                            "Location": {
                                "lat": "43.3347",
                                "long": "7.4216",
                                "locality": "Monte Carlo",
                                "country": "Monaco",
                            },
                        },
                        "date": "2024-05-26",
                        "time": "14:00:00Z",
                        "Results": [
                            {
                                "position": "1",
                                "points": "25",
                                "Driver": {
                                    "driverId": "hamilton",
                                    "givenName": "Lewis",
                                    "familyName": "Hamilton",
                                },
                                "Constructor": {
                                    "constructorId": "mercedes",
                                    "name": "Mercedes",
                                },
                                "grid": "1",
                                "laps": "78",
                                "status": "Finished",
                                "Time": {"millis": "5400123"},
                            }
                        ],
                        "QualifyingResults": [
                            {
                                "position": "1",
                                "Driver": {"driverId": "hamilton"},
                                "Constructor": {"constructorId": "mercedes"},
                                "Q1": "72.345",
                                "Q2": "71.234",
                                "Q3": "70.123",
                            }
                        ],
                    }
                ]
            }
        }
    }
    return response


@pytest.fixture
def mock_api_error():
    """Mock HTTP 500 error response.

    Returns:
        Mock response object with status_code=500
    """
    response = MagicMock()
    response.status_code = 500
    response.json.side_effect = ValueError("Invalid JSON")
    return response


@pytest.fixture
def mock_api_timeout():
    """Mock API timeout exception.

    Returns:
        Exception instance for timeout scenario
    """
    return TimeoutError("API request timed out after 10 seconds")


@pytest.fixture
def mock_api_rate_limit():
    """Mock HTTP 429 rate limit response.

    Returns:
        Mock response object with status_code=429
    """
    response = MagicMock()
    response.status_code = 429
    response.headers = {"Retry-After": "60"}
    return response


# ============================================================================
# Phase 3: Data Collection & Fixtures
# ============================================================================


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Sample feature DataFrame with 20 drivers and typical race features.

    Returns:
        DataFrame with engineered features for 20 drivers
    """
    return pd.DataFrame(
        {
            "driver_id": [i for i in range(1, 21)],
            "driver_name": [
                "Hamilton",
                "Verstappen",
                "Leclerc",
                "Russell",
                "Perez",
                "Alonso",
                "Norris",
                "Gasly",
                "Magnussen",
                "Tsunoda",
                "Stroll",
                "Albon",
                "Ocon",
                "Bottas",
                "Zhou",
                "Ricciardo",
                "Hulkenberg",
                "Sargeant",
                "Piastri",
                "De Vries",
            ],
            "qualifying_position": list(range(1, 21)),
            "driver_form_score": [90.0 - i * 3 for i in range(20)],
            "team_reliability_score": [88.0 - i * 2 for i in range(20)],
            "circuit_performance_score": [85.0 - i * 2.5 for i in range(20)],
            "dnf_rate": [0.05 * i / 20 for i in range(20)],
            "avg_finish_position": [5.0 + i * 0.5 for i in range(20)],
            "season": [2024] * 20,
            "race_round": [5] * 20,
        }
    )


@pytest.fixture
def sample_race_results() -> pd.DataFrame:
    """Sample race results DataFrame with corresponding positions and outcomes.

    Returns:
        DataFrame with actual race results for training
    """
    return pd.DataFrame(
        {
            "season": [2024] * 20,
            "round": [5] * 20,
            "driver_id": [i for i in range(1, 21)],
            "driver_name": [
                "Hamilton",
                "Verstappen",
                "Leclerc",
                "Russell",
                "Perez",
                "Alonso",
                "Norris",
                "Gasly",
                "Magnussen",
                "Tsunoda",
                "Stroll",
                "Albon",
                "Ocon",
                "Bottas",
                "Zhou",
                "Ricciardo",
                "Hulkenberg",
                "Sargeant",
                "Piastri",
                "De Vries",
            ],
            "position": list(range(1, 21)),
            "points": [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10,
            "qualifying_position": list(range(1, 21)),
            "status": ["Finished"] * 18 + ["DNF", "DNF"],
            "lap_time": [5400.123 + i * 10 for i in range(20)],
            "laps_completed": [78 - i for i in range(20)],
        }
    )


@pytest.fixture
def sample_historical_data() -> dict:
    """Sample multi-season historical data structure.

    Returns:
        Dictionary with race results and qualifying data for multiple seasons
    """
    seasons = [2020, 2021, 2022, 2023, 2024]
    data = {
        "race_results": pd.DataFrame(
            {
                "season": [s for s in seasons for _ in range(5)],
                "round": [i for _ in seasons for i in range(1, 6)],
                "driver_id": [1 for _ in range(len(seasons) * 5)],
                "position": [3, 1, 2, 4, 5] * 5,
                "points": [15, 25, 18, 12, 10] * 5,
            }
        ),
        "qualifying_results": pd.DataFrame(
            {
                "season": [s for s in seasons for _ in range(5)],
                "round": [i for _ in seasons for i in range(1, 6)],
                "driver_id": [1 for _ in range(len(seasons) * 5)],
                "position": [2, 1, 3, 1, 2] * 5,
            }
        ),
        "schedule": pd.DataFrame(
            {
                "season": [2024] * 24,
                "round": list(range(1, 25)),
                "race_name": [f"Race {i}" for i in range(1, 25)],
                "date": pd.date_range("2024-02-25", periods=24, freq="2W"),
            }
        ),
    }
    return data


@pytest.fixture
def sample_race_results_web() -> pd.DataFrame:
    """Sample race results for web interface testing (from web conftest).

    Returns:
        DataFrame with race result columns
    """
    return pd.DataFrame(
        {
            "position": [1, 2, 3, 4, 5],
            "driver_id": ["driver_1", "driver_2", "driver_3", "driver_4", "driver_5"],
            "driver_name": [
                "Lewis Hamilton",
                "Max Verstappen",
                "Charles Leclerc",
                "George Russell",
                "Sergio Perez",
            ],
            "team": ["Mercedes", "Red Bull", "Ferrari", "Mercedes", "Red Bull"],
            "points": [25, 18, 15, 12, 10],
            "laps_completed": [78, 78, 77, 76, 75],
        }
    )


@pytest.fixture
def sample_qualifying_results() -> pd.DataFrame:
    """Sample qualifying results DataFrame.

    Returns:
        DataFrame with qualifying session data
    """
    return pd.DataFrame(
        {
            "season": [2024] * 20,
            "round": [5] * 20,
            "driver_id": [i for i in range(1, 21)],
            "position": list(range(1, 21)),
            "q1_time": [73.456 + i * 0.1 for i in range(20)],
            "q2_time": [73.123 + i * 0.1 for i in range(20)],
            "q3_time": [72.890 + i * 0.1 for i in range(20)],
        }
    )


@pytest.fixture
def sample_race_schedule() -> pd.DataFrame:
    """Sample race schedule data.

    Returns:
        DataFrame with race schedule for season
    """
    return pd.DataFrame(
        {
            "season": [2024] * 24,
            "round": list(range(1, 25)),
            "race_name": [
                "Bahrain",
                "Saudi Arabia",
                "Monaco",
                "Canada",
                "Spain",
                "Austria",
                "British",
                "Hungary",
                "Netherlands",
                "Monza",
                "Singapore",
                "Japan",
                "Mexico",
                "USA",
                "Brazil",
                "Abu Dhabi",
            ]
            + [f"Race {i}" for i in range(17, 25)],
            "circuit_id": list(range(1, 25)),
            "date": pd.date_range("2024-02-25", periods=24, freq="2W"),
        }
    )


# ============================================================================
# Phase 4: Feature Engineering Fixtures
# ============================================================================


@pytest.fixture
def engineered_features() -> pd.DataFrame:
    """Sample engineered features for testing feature calculations.

    Returns:
        DataFrame with calculated feature values
    """
    return pd.DataFrame(
        {
            "driver_form": [85.0, 78.0, 72.0, 65.0, 58.0],
            "team_reliability": [0.95, 0.92, 0.88, 0.85, 0.80],
            "qualifying_advantage": [0.1, 0.05, 0.0, -0.05, -0.10],
            "circuit_experience": [8, 10, 12, 14, 16],
            "historical_win_rate": [0.35, 0.25, 0.15, 0.10, 0.05],
        }
    )


# ============================================================================
# Phase 5: Model Fixtures
# ============================================================================


@pytest.fixture
def trained_random_forest():
    """Fixture providing a trained RandomForestRacePredictor.

    Returns:
        Trained RandomForestRacePredictor instance
    """
    from f1_predict.models.random_forest import RandomForestRacePredictor

    model = RandomForestRacePredictor(n_estimators=10, use_optimized_params=False)

    # Create sample training data
    X = pd.DataFrame(
        {
            "qualifying_position": list(range(1, 21)),
            "driver_form_score": [90.0 - i * 3 for i in range(20)],
            "team_reliability_score": [88.0 - i * 2 for i in range(20)],
        }
    )
    y = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 2)

    model.fit(X, y)
    return model


@pytest.fixture
def trained_xgboost():
    """Fixture providing a trained XGBoostRacePredictor.

    Returns:
        Trained XGBoostRacePredictor instance
    """
    from f1_predict.models.xgboost_model import XGBoostRacePredictor

    model = XGBoostRacePredictor(n_estimators=10, use_optimized_params=False)

    # Create sample training data
    X = pd.DataFrame(
        {
            "qualifying_position": list(range(1, 21)),
            "driver_form_score": [90.0 - i * 3 for i in range(20)],
            "team_reliability_score": [88.0 - i * 2 for i in range(20)],
        }
    )
    y = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 2)

    model.fit(X, y, verbose=False)
    return model


@pytest.fixture
def trained_lightgbm():
    """Fixture providing a trained LightGBMRacePredictor.

    Returns:
        Trained LightGBMRacePredictor instance
    """
    from f1_predict.models.lightgbm_model import LightGBMRacePredictor

    model = LightGBMRacePredictor(n_estimators=10, use_optimized_params=False)

    # Create sample training data
    X = pd.DataFrame(
        {
            "qualifying_position": list(range(1, 21)),
            "driver_form_score": [90.0 - i * 3 for i in range(20)],
            "team_reliability_score": [88.0 - i * 2 for i in range(20)],
        }
    )
    y = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 2)

    model.fit(X, y, verbose=False)
    return model


# ============================================================================
# Phase 6: File & Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Temporary directory for model files.

    Args:
        tmp_path: pytest's tmp_path fixture

    Returns:
        Path to temporary model directory
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for data files.

    Args:
        tmp_path: pytest's tmp_path fixture

    Returns:
        Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)

    return data_dir


@pytest.fixture
def integration_tmp_dir() -> Generator[Path, None, None]:
    """Temporary directory for integration test data.

    Yields:
        Temporary directory path
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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


# ============================================================================
# Mocking & Web Fixtures
# ============================================================================


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state.

    Returns:
        MagicMock configured for session state behavior
    """
    state = MagicMock()
    state.__getitem__ = Mock(return_value=None)
    state.__setitem__ = Mock()
    state.__contains__ = Mock(return_value=False)
    return state


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing.

    Returns:
        Dictionary of mocked Streamlit functions
    """
    with patch("streamlit.title") as mock_title, patch(
        "streamlit.markdown"
    ) as mock_markdown, patch("streamlit.subheader") as mock_subheader, patch(
        "streamlit.success"
    ) as mock_success, patch(
        "streamlit.error"
    ) as mock_error:
        return {
            "title": mock_title,
            "markdown": mock_markdown,
            "subheader": mock_subheader,
            "success": mock_success,
            "error": mock_error,
        }


@pytest.fixture
def mock_prediction_manager():
    """Mock PredictionManager for web testing.

    Returns:
        MagicMock configured as PredictionManager
    """
    manager = MagicMock()
    manager.get_upcoming_races.return_value = [
        {
            "race_id": "race_1",
            "name": "Monaco Grand Prix",
            "circuit": "Monaco",
            "date": "2024-05-26",
            "location": "Monte Carlo",
            "round": 5,
            "season": 2024,
        }
    ]
    manager.load_model.return_value = (
        MagicMock(),
        {
            "type": "ensemble",
            "accuracy": 0.72,
            "training_date": "2024-10-01",
            "version": "1.0.0",
        },
    )
    manager.prepare_race_features.return_value = pd.DataFrame(
        {
            "driver_id": [f"driver_{i}" for i in range(20)],
            "qualifying_position": list(range(1, 21)),
            "driver_form_score": [90.0 - i * 3 for i in range(20)],
            "team_reliability_score": [88.0 - i * 2 for i in range(20)],
            "circuit_performance_score": [85.0 - i * 2.5 for i in range(20)],
        }
    )
    manager.generate_prediction.return_value = {
        "race": "Monaco Grand Prix",
        "podium": [
            ("driver_0", 0.85),
            ("driver_1", 0.78),
            ("driver_2", 0.72),
        ],
        "predictions": [
            {
                "position": i + 1,
                "driver_id": f"driver_{i}",
                "confidence": 0.9 - i * 0.03,
            }
            for i in range(20)
        ],
    }
    return manager


@pytest.fixture(autouse=True)
def reset_streamlit_session():
    """Reset Streamlit session state between tests.

    Yields:
        None
    """
    try:
        import streamlit as st

        if hasattr(st, "session_state"):
            st.session_state.clear()
        yield
        if hasattr(st, "session_state"):
            st.session_state.clear()
    except ImportError:
        # Streamlit not available, skip reset
        yield


@pytest.fixture(autouse=True)
def cleanup_pytorch():
    """Clean up PyTorch state between tests to prevent segfaults.

    PyTorch can accumulate state (threads, memory, optimizer state) across
    sequential test runs, leading to segmentation faults in torch/optim/adam.py.
    This fixture ensures clean state before and after each test.

    Yields:
        None
    """
    try:
        import torch
        import gc

        # Pre-test cleanup: garbage collection
        gc.collect()

        yield

        # Post-test cleanup
        # 1. Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 2. Reset threads to prevent accumulation
        torch.set_num_threads(torch.get_num_threads())

        # 3. Garbage collection
        gc.collect()

    except ImportError:
        # PyTorch not available, skip cleanup
        yield
