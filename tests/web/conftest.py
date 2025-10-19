"""Pytest configuration and shared fixtures for web tests."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state."""
    state = MagicMock()
    state.__getitem__ = Mock(return_value=None)
    state.__setitem__ = Mock()
    state.__contains__ = Mock(return_value=False)
    return state


@pytest.fixture
def sample_race_data():
    """Sample race data for testing."""
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
def sample_races_list():
    """Sample list of races for testing."""
    return [
        {
            "race_id": "race_1",
            "name": "Monaco Grand Prix",
            "circuit": "Monaco",
            "date": "2024-05-26",
            "location": "Monte Carlo",
            "round": 5,
            "season": 2024,
        },
        {
            "race_id": "race_2",
            "name": "Canadian Grand Prix",
            "circuit": "Montreal",
            "date": "2024-06-09",
            "location": "Montreal",
            "round": 6,
            "season": 2024,
        },
    ]


@pytest.fixture
def sample_driver_data():
    """Sample driver data for testing."""
    return pd.DataFrame(
        {
            "driver_id": ["driver_1", "driver_2", "driver_3"],
            "name": ["Lewis Hamilton", "Max Verstappen", "Charles Leclerc"],
            "team": ["Mercedes", "Red Bull", "Ferrari"],
            "number": [44, 1, 16],
        }
    )


@pytest.fixture
def sample_race_results():
    """Sample race results for testing."""
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
def sample_prediction():
    """Sample prediction result for testing."""
    return {
        "race": "Monaco Grand Prix",
        "podium": [
            ("driver_1", 0.85),
            ("driver_2", 0.78),
            ("driver_3", 0.72),
        ],
        "predictions": [
            {
                "position": i + 1,
                "driver_id": f"driver_{i+1}",
                "confidence": 0.9 - i * 0.03,
            }
            for i in range(20)
        ],
    }


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing."""
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
    """Mock PredictionManager for testing."""
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
        "podium": [("driver_0", 0.85), ("driver_1", 0.78), ("driver_2", 0.72)],
        "predictions": [
            {"position": i + 1, "driver_id": f"driver_{i}", "confidence": 0.9 - i * 0.03}
            for i in range(20)
        ],
    }
    return manager


@pytest.fixture(autouse=True)
def reset_streamlit_session():
    """Reset Streamlit session state between tests."""
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
