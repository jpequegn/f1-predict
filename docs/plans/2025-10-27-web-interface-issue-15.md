# Web Interface Implementation Plan (Issue #15)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a comprehensive Streamlit web interface with Prediction, Analytics, and Chat pages that integrate ML models, LLM explanations, and F1 data.

**Architecture:** Progressive disclosure UI pattern with three core pages. Prediction page uses Ensemble model for race forecasting with expandable advanced options. Analytics page provides KPI dashboard with championship standings and performance visualizations. Chat page enables LLM-powered F1 queries. Session state caching for performance. Integration with existing models (Issue #9), LLM system (Issue #11), and data sources (Issue #16).

**Tech Stack:** Streamlit (UI framework), Streamlit Option Menu (navigation), Plotly (interactive charts), Pandas (data manipulation), Ensemble predictor (ML), LLM API (explanations), Pydantic (data validation), pytest + Streamlit testing.

---

## Task 1: Setup Test Infrastructure for Web Module

**Files:**
- Create: `tests/web/__init__.py`
- Create: `tests/web/conftest.py`
- Create: `tests/web/test_prediction.py`
- Modify: `pyproject.toml` (add web test marker if needed)

**Step 1: Create test directory structure**

Run: `mkdir -p tests/web`

**Step 2: Write pytest fixtures for Streamlit testing**

Create `tests/web/conftest.py`:

```python
"""Shared fixtures for web interface tests."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest


@pytest.fixture
def mock_session_state():
    """Fixture for Streamlit session state."""
    state = {
        "initialized": True,
        "settings": {
            "theme": "Nebula Dark",
            "timezone": "UTC",
            "units": "metric",
            "default_model": "Ensemble",
            "confidence_threshold": 0.7,
            "enable_explanations": True,
        },
        "predictions_cache": {},
        "chat_history": [],
    }
    return state


@pytest.fixture
def mock_upcoming_races():
    """Fixture for upcoming race data."""
    return pd.DataFrame({
        "round": [21, 22],
        "race_name": ["Abu Dhabi Grand Prix", "Season Final"],
        "circuit_name": ["Yas Marina", "Unknown"],
        "race_date": [
            datetime(2025, 12, 7),
            datetime(2025, 12, 14),
        ],
        "season": [2025, 2025],
    })


@pytest.fixture
def mock_ensemble_predictions():
    """Fixture for ensemble model predictions."""
    return {
        "podium": [
            {"position": 1, "driver": "Verstappen", "confidence": 0.87},
            {"position": 2, "driver": "Hamilton", "confidence": 0.72},
            {"position": 3, "driver": "Norris", "confidence": 0.65},
        ],
        "full_order": [
            {"position": i+1, "driver": f"Driver {i+1}", "confidence": 0.5}
            for i in range(20)
        ],
        "feature_importance": {
            "team_strength": 0.35,
            "driver_form": 0.28,
            "circuit_fit": 0.22,
            "grid_position": 0.15,
        },
        "metadata": {
            "model": "Ensemble",
            "timestamp": datetime.now().isoformat(),
            "race_id": "2025_21",
        }
    }


@pytest.fixture
def mock_llm_explanation():
    """Fixture for LLM explanation."""
    return """
    Based on recent form and circuit characteristics, Verstappen leads with 87% confidence
    due to superior car performance and consistency. Hamilton's experience at Yas Marina
    (0.72 confidence) makes him a strong contender for second place.
    """


@pytest.fixture
def mock_kpi_metrics():
    """Fixture for analytics KPI metrics."""
    return {
        "races_analyzed": 15,
        "prediction_accuracy": 0.68,
        "avg_confidence": 0.72,
        "total_predictions": 285,
    }


@pytest.fixture
def mock_championship_standings():
    """Fixture for championship standings data."""
    return pd.DataFrame({
        "position": [1, 2, 3],
        "driver": ["Verstappen", "Hamilton", "Norris"],
        "points": [360, 295, 280],
        "wins": [15, 8, 5],
        "podiums": [18, 14, 11],
    })
```

Create `tests/web/__init__.py` (empty file).

**Step 3: Commit fixtures**

```bash
git add tests/web/
git commit -m "test: add web module test infrastructure and fixtures"
```

---

## Task 2: Create Prediction Utilities

**Files:**
- Create: `src/f1_predict/web/utils/prediction.py`
- Create: `tests/web/test_prediction_utils.py`

**Step 1: Write failing test for upcoming races fetching**

Create `tests/web/test_prediction_utils.py`:

```python
"""Tests for prediction utilities."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import pandas as pd

from f1_predict.web.utils.prediction import (
    get_upcoming_races,
    prepare_race_features,
    get_ensemble_prediction,
)


def test_get_upcoming_races_returns_dataframe(mock_upcoming_races):
    """Test that get_upcoming_races returns a properly formatted DataFrame."""
    with patch('f1_predict.web.utils.prediction.fetch_upcoming_races',
               return_value=mock_upcoming_races):
        result = get_upcoming_races()

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ['round', 'race_name', 'race_date'])
        assert len(result) > 0


def test_prepare_race_features_creates_valid_input(mock_upcoming_races):
    """Test that prepare_race_features creates ML-ready input."""
    race = mock_upcoming_races.iloc[0]
    features = prepare_race_features(race)

    assert isinstance(features, dict)
    assert 'season' in features
    assert 'round' in features
    assert 'circuit_id' in features
    assert features['season'] == 2025


def test_get_ensemble_prediction_returns_prediction_dict(
    mock_upcoming_races,
    mock_ensemble_predictions,
):
    """Test that get_ensemble_prediction returns properly formatted prediction."""
    race = mock_upcoming_races.iloc[0]

    with patch('f1_predict.web.utils.prediction.EnsemblePredictor') as MockPredictor:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_ensemble_predictions
        MockPredictor.return_value = mock_predictor

        result = get_ensemble_prediction(race)

        assert isinstance(result, dict)
        assert 'podium' in result
        assert 'full_order' in result
        assert 'feature_importance' in result
        assert len(result['podium']) == 3
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src pytest tests/web/test_prediction_utils.py -v`

Expected: All 3 tests FAIL (functions not defined)

**Step 3: Write implementation**

Create `src/f1_predict/web/utils/prediction.py`:

```python
"""Prediction utilities for web interface."""
from datetime import datetime, timedelta
from typing import Dict, List, Any

import pandas as pd

from f1_predict.models.ensemble import EnsemblePredictor


def get_upcoming_races() -> pd.DataFrame:
    """
    Fetch upcoming races for the current season.

    Returns:
        DataFrame with columns: round, race_name, circuit_name, race_date, season
    """
    # TODO: Integrate with F1DataCollector to fetch upcoming races from data source
    # For now, return mock data structure
    from datetime import datetime

    upcoming = pd.DataFrame({
        "round": [],
        "race_name": [],
        "circuit_name": [],
        "race_date": [],
        "season": [],
    })

    return upcoming


def prepare_race_features(race_row: pd.Series) -> Dict[str, Any]:
    """
    Prepare race features for ML model input.

    Args:
        race_row: Single row from upcoming races DataFrame

    Returns:
        Dictionary with ML-ready features
    """
    features = {
        "season": int(race_row.get("season", 2025)),
        "round": int(race_row.get("round", 1)),
        "circuit_id": str(race_row.get("circuit_id", "")),
        "race_date": race_row.get("race_date"),
    }

    return features


def get_ensemble_prediction(race_row: pd.Series) -> Dict[str, Any]:
    """
    Get ensemble prediction for a race.

    Args:
        race_row: Single row from races DataFrame

    Returns:
        Dictionary with podium, full_order, feature_importance
    """
    features = prepare_race_features(race_row)

    # Initialize ensemble predictor
    predictor = EnsemblePredictor()

    # Get prediction
    prediction = predictor.predict([features])

    # Format for UI
    result = {
        "podium": prediction.get("podium", []),
        "full_order": prediction.get("full_order", []),
        "feature_importance": prediction.get("feature_importance", {}),
        "metadata": {
            "model": "Ensemble",
            "timestamp": datetime.now().isoformat(),
            "race_id": f"{race_row.get('season')}_{race_row.get('round')}",
        }
    }

    return result
```

**Step 4: Run tests to verify passing**

Run: `PYTHONPATH=src pytest tests/web/test_prediction_utils.py -v`

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/f1_predict/web/utils/prediction.py tests/web/test_prediction_utils.py
git commit -m "feat: add prediction utilities for race forecasting"
```

---

## Task 3: Create Analytics Utilities

**Files:**
- Create: `src/f1_predict/web/utils/analytics.py`
- Create: `tests/web/test_analytics_utils.py`

**Step 1: Write failing test for analytics calculations**

Create `tests/web/test_analytics_utils.py`:

```python
"""Tests for analytics utilities."""
import pytest
import pandas as pd
from datetime import datetime

from f1_predict.web.utils.analytics import (
    calculate_kpi_metrics,
    get_championship_standings,
    calculate_driver_performance,
)


def test_calculate_kpi_metrics_returns_dict(mock_kpi_metrics):
    """Test that calculate_kpi_metrics returns KPI dictionary."""
    with pytest.mock.patch(
        'f1_predict.web.utils.analytics.load_prediction_history',
        return_value=[],
    ):
        result = calculate_kpi_metrics()

        assert isinstance(result, dict)
        assert 'races_analyzed' in result
        assert 'prediction_accuracy' in result
        assert 'avg_confidence' in result


def test_get_championship_standings_returns_dataframe(mock_championship_standings):
    """Test that get_championship_standings returns DataFrame."""
    with pytest.mock.patch(
        'f1_predict.web.utils.analytics.fetch_current_standings',
        return_value=mock_championship_standings,
    ):
        result = get_championship_standings()

        assert isinstance(result, pd.DataFrame)
        assert 'driver' in result.columns
        assert 'points' in result.columns


def test_calculate_driver_performance_returns_metrics():
    """Test that calculate_driver_performance returns performance metrics."""
    driver = "Verstappen"

    with pytest.mock.patch(
        'f1_predict.web.utils.analytics.fetch_driver_history',
        return_value=pd.DataFrame({
            "race": ["Race1", "Race2"],
            "position": [1, 2],
            "points": [25, 18],
        }),
    ):
        result = calculate_driver_performance(driver)

        assert isinstance(result, dict)
        assert 'wins' in result
        assert 'podiums' in result
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src pytest tests/web/test_analytics_utils.py -v`

Expected: All 3 tests FAIL

**Step 3: Write implementation**

Create `src/f1_predict/web/utils/analytics.py`:

```python
"""Analytics utilities for web interface."""
from typing import Dict, Any, List

import pandas as pd


def calculate_kpi_metrics() -> Dict[str, Any]:
    """
    Calculate KPI metrics for analytics dashboard.

    Returns:
        Dictionary with races_analyzed, prediction_accuracy, avg_confidence, total_predictions
    """
    # TODO: Load from prediction history database
    kpis = {
        "races_analyzed": 0,
        "prediction_accuracy": 0.0,
        "avg_confidence": 0.0,
        "total_predictions": 0,
    }

    return kpis


def get_championship_standings() -> pd.DataFrame:
    """
    Get current championship standings.

    Returns:
        DataFrame with driver positions, points, wins, podiums
    """
    standings = pd.DataFrame({
        "position": [],
        "driver": [],
        "points": [],
        "wins": [],
        "podiums": [],
    })

    return standings


def calculate_driver_performance(driver_name: str) -> Dict[str, Any]:
    """
    Calculate performance metrics for a specific driver.

    Args:
        driver_name: Name of the driver

    Returns:
        Dictionary with wins, podiums, avg_finish_position, etc.
    """
    performance = {
        "wins": 0,
        "podiums": 0,
        "avg_finish_position": 0.0,
        "races_completed": 0,
    }

    return performance
```

**Step 4: Run tests to verify passing**

Run: `PYTHONPATH=src pytest tests/web/test_analytics_utils.py -v`

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/f1_predict/web/utils/analytics.py tests/web/test_analytics_utils.py
git commit -m "feat: add analytics utilities for dashboard metrics"
```

---

## Task 4: Create LLM Explanation Utilities

**Files:**
- Create: `src/f1_predict/web/utils/llm_explanations.py`
- Create: `tests/web/test_llm_explanations.py`

**Step 1: Write failing test for LLM explanations**

Create `tests/web/test_llm_explanations.py`:

```python
"""Tests for LLM explanation utilities."""
import pytest
from unittest.mock import MagicMock, patch

from f1_predict.web.utils.llm_explanations import (
    generate_prediction_explanation,
    generate_driver_comparison,
)


def test_generate_prediction_explanation_returns_string(mock_llm_explanation):
    """Test that generate_prediction_explanation returns a string."""
    prediction = {
        "podium": [
            {"position": 1, "driver": "Verstappen", "confidence": 0.87},
            {"position": 2, "driver": "Hamilton", "confidence": 0.72},
        ],
        "metadata": {"race_id": "2025_21"},
    }

    with patch('f1_predict.web.utils.llm_explanations.LLMClient') as MockLLM:
        mock_llm = MagicMock()
        mock_llm.generate_explanation.return_value = mock_llm_explanation
        MockLLM.return_value = mock_llm

        result = generate_prediction_explanation(prediction)

        assert isinstance(result, str)
        assert len(result) > 0


def test_generate_driver_comparison_returns_string():
    """Test that generate_driver_comparison returns comparison text."""
    driver1 = "Verstappen"
    driver2 = "Hamilton"

    with patch('f1_predict.web.utils.llm_explanations.LLMClient') as MockLLM:
        mock_llm = MagicMock()
        mock_llm.compare_drivers.return_value = "Comparison text"
        MockLLM.return_value = mock_llm

        result = generate_driver_comparison(driver1, driver2)

        assert isinstance(result, str)
        assert len(result) > 0
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src pytest tests/web/test_llm_explanations.py -v`

Expected: All 2 tests FAIL

**Step 3: Write implementation**

Create `src/f1_predict/web/utils/llm_explanations.py`:

```python
"""LLM explanation utilities for web interface."""
from typing import Dict, Any

# TODO: Import actual LLM client from Issue #11
# from f1_predict.llm.client import LLMClient


def generate_prediction_explanation(prediction: Dict[str, Any]) -> str:
    """
    Generate natural language explanation for a race prediction.

    Args:
        prediction: Prediction dictionary with podium, confidence scores

    Returns:
        String explanation of the prediction
    """
    # TODO: Integrate with LLM API from Issue #11

    podium = prediction.get("podium", [])
    explanation_parts = []

    for entry in podium:
        driver = entry.get("driver", "Unknown")
        confidence = entry.get("confidence", 0.0)
        position = entry.get("position", 0)

        explanation_parts.append(
            f"Position {position}: {driver} ({confidence:.0%} confidence)"
        )

    return " → ".join(explanation_parts) if explanation_parts else "No prediction available."


def generate_driver_comparison(driver1: str, driver2: str) -> str:
    """
    Generate comparison between two drivers.

    Args:
        driver1: First driver name
        driver2: Second driver name

    Returns:
        String comparison of the two drivers
    """
    # TODO: Integrate with LLM API from Issue #11

    return f"Comparison between {driver1} and {driver2} coming soon."
```

**Step 4: Run tests to verify passing**

Run: `PYTHONPATH=src pytest tests/web/test_llm_explanations.py -v`

Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add src/f1_predict/web/utils/llm_explanations.py tests/web/test_llm_explanations.py
git commit -m "feat: add LLM explanation utilities for prediction insights"
```

---

## Task 5: Implement Prediction Page

**Files:**
- Modify: `src/f1_predict/web/pages/predict.py`
- Create: `tests/web/test_predict_page.py`

**Step 1: Write failing test for prediction page**

Create `tests/web/test_predict_page.py`:

```python
"""Tests for prediction page."""
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st

from f1_predict.web.pages.predict import show_prediction_page


def test_show_prediction_page_renders_without_error():
    """Test that prediction page renders without errors."""
    # Mock Streamlit functions
    with patch('f1_predict.web.pages.predict.st.title'), \
         patch('f1_predict.web.pages.predict.st.columns'), \
         patch('f1_predict.web.pages.predict.st.selectbox'), \
         patch('f1_predict.web.pages.predict.st.button'):

        # Should not raise an exception
        show_prediction_page()


def test_show_prediction_page_displays_race_selection():
    """Test that prediction page includes race selection."""
    with patch('f1_predict.web.pages.predict.st.selectbox') as mock_selectbox:
        with patch('f1_predict.web.pages.predict.st.title'), \
             patch('f1_predict.web.pages.predict.st.columns'), \
             patch('f1_predict.web.pages.predict.st.button'):

            show_prediction_page()

            # Verify selectbox was called for race selection
            assert mock_selectbox.called
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src pytest tests/web/test_predict_page.py::test_show_prediction_page_renders_without_error -v`

Expected: FAIL

**Step 3: Implement prediction page**

Replace `src/f1_predict/web/pages/predict.py`:

```python
"""Race prediction page for F1 Race Predictor web app."""
from datetime import datetime
from typing import Optional

import streamlit as st
import pandas as pd

from f1_predict.web.utils.prediction import (
    get_upcoming_races,
    get_ensemble_prediction,
)
from f1_predict.web.utils.llm_explanations import generate_prediction_explanation


def show_prediction_page() -> None:
    """Display the race prediction interface."""
    st.title("🏁 Race Prediction")

    # Session state initialization
    if "prediction_cache" not in st.session_state:
        st.session_state.prediction_cache = {}

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Select Race")

        # Get upcoming races
        try:
            upcoming_races = get_upcoming_races()

            if upcoming_races.empty:
                st.warning("No upcoming races available.")
                return

            # Create race display name
            race_options = [
                f"Round {row['round']}: {row['race_name']}"
                for _, row in upcoming_races.iterrows()
            ]

            selected_race_idx = st.selectbox(
                "Select a race",
                range(len(race_options)),
                format_func=lambda i: race_options[i],
            )

            selected_race = upcoming_races.iloc[selected_race_idx]

            # Model selection
            st.markdown("### Select Model")
            model_choice = st.radio(
                "Choose prediction model",
                options=["Ensemble", "XGBoost", "LightGBM", "Random Forest"],
                index=0,
            )

            # Predict button
            if st.button("Generate Prediction", key="predict_btn"):
                with st.spinner("Generating prediction..."):
                    prediction = get_ensemble_prediction(selected_race)

                    # Cache prediction
                    cache_key = f"{selected_race['season']}_{selected_race['round']}"
                    st.session_state.prediction_cache[cache_key] = prediction

                    st.session_state.show_results = True

        except Exception as e:
            st.error(f"Error loading races: {str(e)}")
            return

    with col2:
        st.markdown("### Settings")
        show_advanced = st.checkbox("Advanced Options")

    # Display results if available
    if st.session_state.get("show_results", False):
        st.markdown("---")
        st.markdown("## Prediction Results")

        cache_key = f"{selected_race['season']}_{selected_race['round']}"
        prediction = st.session_state.prediction_cache.get(cache_key)

        if prediction:
            _display_prediction_results(prediction, selected_race, show_advanced)


def _display_prediction_results(
    prediction: dict,
    race: pd.Series,
    show_advanced: bool,
) -> None:
    """Display prediction results in the UI."""

    # Podium predictions
    st.markdown("### Predicted Podium")

    podium = prediction.get("podium", [])
    for entry in podium:
        position = entry.get("position", 0)
        driver = entry.get("driver", "Unknown")
        confidence = entry.get("confidence", 0.0)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.metric(f"P{position}", driver)
        with col3:
            st.metric("Confidence", f"{confidence:.1%}")

    # Full race order
    st.markdown("### Full Race Order")
    full_order = prediction.get("full_order", [])
    order_df = pd.DataFrame(full_order)

    if not order_df.empty:
        st.dataframe(order_df, use_container_width=True)

    # Feature importance
    st.markdown("### Feature Importance")
    feature_importance = prediction.get("feature_importance", {})

    if feature_importance:
        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h'
            )
        ])
        fig.update_layout(
            title="Feature Importance in Prediction",
            xaxis_title="Importance Score",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # LLM Explanation
    if st.session_state.settings.get("enable_explanations", True):
        st.markdown("### AI Insights")
        try:
            explanation = generate_prediction_explanation(prediction)
            st.info(explanation)
        except Exception as e:
            st.warning(f"Could not generate explanation: {str(e)}")

    # Advanced options
    if show_advanced:
        with st.expander("Advanced Analysis"):
            st.markdown("**Weather Conditions**")
            weather_prob = st.slider("Rain Probability", 0, 100, 30)

            st.markdown("**Strategy Insights**")
            st.write("Tire strategy impacts estimated here...")

            st.markdown("**Safety Car Probability**")
            safety_car_prob = st.slider("Safety Car Likelihood", 0, 100, 20)
```

**Step 4: Run tests to verify passing**

Run: `PYTHONPATH=src pytest tests/web/test_predict_page.py -v`

Expected: Tests PASS

**Step 5: Commit**

```bash
git add src/f1_predict/web/pages/predict.py tests/web/test_predict_page.py
git commit -m "feat: implement prediction page with race selection and result display"
```

---

## Task 6: Implement Analytics Page

**Files:**
- Modify: `src/f1_predict/web/pages/analytics.py`
- Create: `tests/web/test_analytics_page.py`

**Step 1: Write failing test for analytics page**

Create `tests/web/test_analytics_page.py`:

```python
"""Tests for analytics page."""
import pytest
from unittest.mock import patch

from f1_predict.web.pages.analytics import show_analytics_page


def test_show_analytics_page_renders_without_error():
    """Test that analytics page renders without errors."""
    with patch('f1_predict.web.pages.analytics.st.title'), \
         patch('f1_predict.web.pages.analytics.st.metric'), \
         patch('f1_predict.web.pages.analytics.st.columns'):

        show_analytics_page()
```

**Step 2: Run test to verify failure**

Run: `PYTHONPATH=src pytest tests/web/test_analytics_page.py -v`

Expected: FAIL

**Step 3: Implement analytics page**

Replace `src/f1_predict/web/pages/analytics.py`:

```python
"""Analytics dashboard page for F1 Race Predictor web app."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from f1_predict.web.utils.analytics import (
    calculate_kpi_metrics,
    get_championship_standings,
)


def show_analytics_page() -> None:
    """Display the analytics and visualization dashboard."""
    st.title("📈 F1 Analytics Dashboard")

    # KPI Section
    st.markdown("## Key Performance Indicators")

    try:
        kpis = calculate_kpi_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Races Analyzed",
                kpis.get("races_analyzed", 0),
            )

        with col2:
            accuracy = kpis.get("prediction_accuracy", 0.0)
            st.metric(
                "Prediction Accuracy",
                f"{accuracy:.1%}",
            )

        with col3:
            confidence = kpis.get("avg_confidence", 0.0)
            st.metric(
                "Avg Confidence",
                f"{confidence:.1%}",
            )

        with col4:
            st.metric(
                "Total Predictions",
                kpis.get("total_predictions", 0),
            )

    except Exception as e:
        st.error(f"Error loading KPI metrics: {str(e)}")

    st.markdown("---")

    # Championship Standings
    st.markdown("## Championship Standings")

    try:
        standings = get_championship_standings()

        if standings.empty:
            st.info("Championship standings not available yet.")
        else:
            # Display standings table
            st.dataframe(standings, use_container_width=True, hide_index=True)

            # Points progression chart
            st.markdown("### Points Distribution")

            if "driver" in standings.columns and "points" in standings.columns:
                fig = go.Figure(data=[
                    go.Bar(
                        x=standings["driver"],
                        y=standings["points"],
                        marker_color='rgba(31, 78, 140, 0.8)',
                    )
                ])

                fig.update_layout(
                    title="Championship Points by Driver",
                    xaxis_title="Driver",
                    yaxis_title="Points",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading championship standings: {str(e)}")

    st.markdown("---")

    # Performance Analysis
    st.markdown("## Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Win Rate by Team")
        st.info("Team performance visualization coming soon...")

    with col2:
        st.markdown("### Reliability Analysis")
        st.info("Reliability metrics visualization coming soon...")

    # Circuit Performance
    st.markdown("## Circuit Performance Heatmap")
    st.info("Interactive circuit heatmap visualization coming soon...")
```

**Step 4: Run tests to verify passing**

Run: `PYTHONPATH=src pytest tests/web/test_analytics_page.py -v`

Expected: Tests PASS

**Step 5: Commit**

```bash
git add src/f1_predict/web/pages/analytics.py tests/web/test_analytics_page.py
git commit -m "feat: implement analytics page with KPIs and championship standings"
```

---

## Task 7: Implement Chat Page

**Files:**
- Modify: `src/f1_predict/web/pages/chat.py`
- Create: `src/f1_predict/web/utils/chat.py`
- Create: `tests/web/test_chat_page.py`

**Step 1: Write failing test for chat utilities**

Create `tests/web/test_chat_page.py`:

```python
"""Tests for chat page."""
import pytest
from unittest.mock import patch

from f1_predict.web.pages.chat import show_chat_page


def test_show_chat_page_renders_without_error():
    """Test that chat page renders without errors."""
    with patch('f1_predict.web.pages.chat.st.title'), \
         patch('f1_predict.web.pages.chat.st.container'), \
         patch('f1_predict.web.pages.chat.st.chat_input'):

        show_chat_page()
```

Create `src/f1_predict/web/utils/chat.py`:

```python
"""Chat utilities for LLM interactions."""
from typing import List, Dict, Any


def format_message(role: str, content: str) -> Dict[str, str]:
    """
    Format a message for chat history.

    Args:
        role: 'user' or 'assistant'
        content: Message content

    Returns:
        Formatted message dictionary
    """
    return {"role": role, "content": content}


def get_llm_response(user_message: str, history: List[Dict[str, str]]) -> str:
    """
    Get LLM response to user message.

    Args:
        user_message: User's message
        history: Chat history

    Returns:
        LLM response text
    """
    # TODO: Integrate with LLM API from Issue #11
    return f"Response to: {user_message}"
```

**Step 2: Write failing test**

Add to `tests/web/test_chat_page.py`:

```python
def test_show_chat_page_handles_user_input():
    """Test that chat page handles user input."""
    with patch('f1_predict.web.pages.chat.st.title'), \
         patch('f1_predict.web.pages.chat.st.container'), \
         patch('f1_predict.web.pages.chat.st.chat_input') as mock_input, \
         patch('f1_predict.web.pages.chat.st.session_state',
               {"chat_history": []}):

        show_chat_page()
```

**Step 3: Run test to verify failure**

Run: `PYTHONPATH=src pytest tests/web/test_chat_page.py -v`

Expected: FAIL

**Step 4: Implement chat page**

Replace `src/f1_predict/web/pages/chat.py`:

```python
"""Chat page for F1 Race Predictor web app."""
import streamlit as st

from f1_predict.web.utils.chat import format_message, get_llm_response


def show_chat_page() -> None:
    """Display the chat interface."""
    st.title("💬 F1 Chat Assistant")

    st.markdown("""
    Ask questions about F1 races, drivers, predictions, and more.
    The AI assistant is powered by advanced language models.
    """)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    with st.container():
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]

            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").write(content)

    # Chat input
    user_input = st.chat_input("Ask me about F1...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(
            format_message("user", user_input)
        )

        # Get LLM response
        with st.spinner("Thinking..."):
            response = get_llm_response(
                user_input,
                st.session_state.chat_history
            )

        # Add assistant response to history
        st.session_state.chat_history.append(
            format_message("assistant", response)
        )

        # Rerun to display new messages
        st.rerun()

    # Sidebar: Chat settings
    with st.sidebar:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")
        st.markdown("### Chat Settings")

        temperature = st.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
        )

        max_tokens = st.slider(
            "Max Response Length",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
        )
```

**Step 5: Run tests to verify passing**

Run: `PYTHONPATH=src pytest tests/web/test_chat_page.py -v`

Expected: Tests PASS

**Step 6: Commit**

```bash
git add src/f1_predict/web/pages/chat.py \
         src/f1_predict/web/utils/chat.py \
         tests/web/test_chat_page.py
git commit -m "feat: implement chat page with LLM integration"
```

---

## Task 8: Integration Testing

**Files:**
- Create: `tests/web/test_integration.py`

**Step 1: Write integration tests**

Create `tests/web/test_integration.py`:

```python
"""Integration tests for web interface."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


def test_full_prediction_workflow():
    """Test complete prediction workflow from race selection to results."""
    from f1_predict.web.pages.predict import show_prediction_page

    with patch('f1_predict.web.pages.predict.get_upcoming_races') as mock_races, \
         patch('f1_predict.web.pages.predict.get_ensemble_prediction') as mock_pred, \
         patch('f1_predict.web.pages.predict.st.selectbox'), \
         patch('f1_predict.web.pages.predict.st.button'), \
         patch('f1_predict.web.pages.predict.st.title'), \
         patch('f1_predict.web.pages.predict.st.columns'):

        # Mock data
        mock_races.return_value = pd.DataFrame({
            "round": [21],
            "race_name": ["Test Race"],
            "circuit_name": ["Test Circuit"],
            "race_date": ["2025-12-07"],
            "season": [2025],
        })

        mock_pred.return_value = {
            "podium": [
                {"position": 1, "driver": "Driver1", "confidence": 0.9},
            ],
            "full_order": [],
            "feature_importance": {},
            "metadata": {"race_id": "2025_21"},
        }

        # Should not raise
        show_prediction_page()


def test_analytics_dashboard_loads():
    """Test analytics dashboard loads without errors."""
    from f1_predict.web.pages.analytics import show_analytics_page

    with patch('f1_predict.web.pages.analytics.calculate_kpi_metrics') as mock_kpi, \
         patch('f1_predict.web.pages.analytics.get_championship_standings') as mock_stand, \
         patch('f1_predict.web.pages.analytics.st.title'), \
         patch('f1_predict.web.pages.analytics.st.metric'), \
         patch('f1_predict.web.pages.analytics.st.dataframe'):

        mock_kpi.return_value = {
            "races_analyzed": 10,
            "prediction_accuracy": 0.75,
            "avg_confidence": 0.80,
            "total_predictions": 100,
        }

        mock_stand.return_value = pd.DataFrame({
            "driver": ["Driver1"],
            "points": [100],
        })

        # Should not raise
        show_analytics_page()
```

**Step 2: Run integration tests**

Run: `PYTHONPATH=src pytest tests/web/test_integration.py -v`

Expected: All integration tests PASS

**Step 3: Commit**

```bash
git add tests/web/test_integration.py
git commit -m "test: add integration tests for web interface pages"
```

---

## Task 9: Fix Code Quality Issues

**Files:**
- All files from previous tasks

**Step 1: Run ruff to check linting**

Run: `PYTHONPATH=src uv run ruff check src/f1_predict/web tests/web --fix`

Expected: Auto-fixable issues corrected

**Step 2: Run mypy for type checking**

Run: `PYTHONPATH=src uv run mypy src/f1_predict/web tests/web`

Expected: No type errors

**Step 3: Run all tests**

Run: `PYTHONPATH=src uv run pytest tests/web -v --cov=src/f1_predict/web --cov-report=term-missing`

Expected: All tests pass with >80% coverage

**Step 4: Commit quality fixes**

```bash
git add src/f1_predict/web/ tests/web/
git commit -m "fix: address linting and type checking issues in web module"
```

---

## Task 10: Final Documentation and Deployment Prep

**Files:**
- Create: `docs/WEB_INTERFACE.md`
- Modify: `README.md`

**Step 1: Write web interface documentation**

Create `docs/WEB_INTERFACE.md`:

```markdown
# Web Interface Documentation

## Overview

The F1 Race Predictor web interface is built with Streamlit and provides three core pages:

### Pages

1. **Prediction Page** - Race outcome prediction with Ensemble ML model
2. **Analytics Page** - Championship standings and performance metrics
3. **Chat Page** - LLM-powered Q&A about F1

## Running the Web App

\`\`\`bash
streamlit run src/f1_predict/web/app.py
\`\`\`

Visit http://localhost:8501 in your browser.

## Architecture

- **Streamlit**: UI framework
- **Session State**: Client-side caching of predictions and chat history
- **Utilities**: Modular utilities for prediction, analytics, LLM explanations
- **Integration**: Uses Ensemble predictor (Issue #9), LLM API (Issue #11), data (Issue #16)

## Development

See implementation plan: `docs/plans/2025-10-27-web-interface-issue-15.md`
```

**Step 2: Run final tests**

Run: `PYTHONPATH=src uv run pytest tests/web -v`

Expected: All tests pass

**Step 3: Commit**

```bash
git add docs/WEB_INTERFACE.md
git commit -m "docs: add web interface documentation"
```

---

## Summary

This plan implements Issue #15 (Web Interface) in 10 bite-sized tasks:

1. **Test Infrastructure** - Pytest fixtures and test module setup
2. **Prediction Utilities** - Race fetching, feature preparation, ML predictions
3. **Analytics Utilities** - KPI calculations, standings fetching
4. **LLM Utilities** - Explanation generation
5. **Prediction Page** - Full race prediction UI
6. **Analytics Page** - Dashboard with KPIs and standings
7. **Chat Page** - LLM-powered chat interface
8. **Integration Testing** - Full workflow validation
9. **Code Quality** - Linting, type checking, test coverage
10. **Documentation** - User-facing and developer documentation

**Total Estimated Time**: 8-12 hours of focused development

**Key Dependencies**:
- Issue #9 (Ensemble Predictor) - For predictions
- Issue #11 (LLM API) - For explanations and chat
- Issue #16 (Real-time Integration) - For live data (optional for MVP)

**Testing Strategy**:
- Unit tests for all utilities
- Page-level tests for Streamlit components
- Integration tests for complete workflows
- Minimum 80% coverage across all new code

---
