"""Tests for home page."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from f1_predict.web.pages import home


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions."""
    with patch("f1_predict.web.pages.home.st") as mock_st:
        # Set up default return values for Streamlit components
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_st.container.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        mock_st.button.return_value = False
        mock_st.metric.return_value = None
        mock_st.selectbox.return_value = None
        mock_st.markdown.return_value = None
        mock_st.info.return_value = None
        mock_st.caption.return_value = None
        yield mock_st


def test_show_home_page_renders_without_error(mock_streamlit):
    """Test home page renders without errors."""
    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = []

        # Should not raise any exception
        home.show_home_page()

        # Verify title was set
        mock_streamlit.title.assert_called_once()


def test_show_home_page_displays_quick_stats(mock_streamlit):
    """Test home page displays quick stats."""
    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = []

        home.show_home_page()

        # Verify metrics were displayed (4 KPI metrics)
        assert mock_streamlit.metric.call_count >= 4


def test_show_home_page_displays_upcoming_races(mock_streamlit):
    """Test home page displays upcoming races."""
    upcoming_races = [
        {
            "name": "Monaco Grand Prix",
            "location": "Monaco",
            "circuit": "Circuit de Monaco",
            "date": "2024-05-26",
            "round": 6,
        },
        {
            "name": "Canadian Grand Prix",
            "location": "Montreal",
            "circuit": "Circuit Gilles Villeneuve",
            "date": "2024-06-09",
            "round": 7,
        },
    ]

    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = upcoming_races

        home.show_home_page()

        # Verify next 3 races are displayed
        # Each race displays name, location, date, and predict button
        assert mock_streamlit.markdown.call_count > 0


def test_show_home_page_predict_button(mock_streamlit):
    """Test home page has predict button for upcoming races."""
    upcoming_races = [
        {
            "name": "Monaco Grand Prix",
            "location": "Monaco",
            "circuit": "Circuit de Monaco",
            "date": "2024-05-26",
            "round": 6,
        },
    ]

    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = upcoming_races
        mock_streamlit.button.return_value = False

        home.show_home_page()

        # Verify predict buttons were created
        assert mock_streamlit.button.call_count >= 1


def test_show_home_page_quick_actions(mock_streamlit):
    """Test home page displays quick action buttons."""
    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = []
        mock_streamlit.button.return_value = False

        home.show_home_page()

        # Verify action buttons were created
        assert mock_streamlit.button.call_count >= 1


def test_show_home_page_no_upcoming_races(mock_streamlit):
    """Test home page handles no upcoming races gracefully."""
    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = []

        home.show_home_page()

        # Verify info message was displayed
        mock_streamlit.info.assert_called()


def test_show_home_page_ai_assistant_section(mock_streamlit):
    """Test home page displays AI assistant section."""
    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = []

        home.show_home_page()

        # Verify sections were created
        assert mock_streamlit.markdown.call_count > 0


def test_show_home_page_three_upcoming_races_max(mock_streamlit):
    """Test home page shows max 3 upcoming races."""
    upcoming_races = [
        {
            "name": f"Race {i}",
            "location": f"Location {i}",
            "circuit": f"Circuit {i}",
            "date": f"2024-0{i}-01",
            "round": i,
        }
        for i in range(1, 8)  # Create 7 races
    ]

    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = upcoming_races

        home.show_home_page()

        # Only 3 races should be shown
        # Count button calls for predict buttons (one per race shown)
        predict_buttons = [
            call for call in mock_streamlit.button.call_args_list
            if "Predict" in str(call)
        ]
        assert len(predict_buttons) <= 3


def test_get_upcoming_races_function():
    """Test get_upcoming_races function exists and is callable."""
    # Verify the function exists
    assert hasattr(home, "get_upcoming_races")
    assert callable(home.get_upcoming_races)


def test_show_home_page_columns_layout(mock_streamlit):
    """Test home page uses proper column layout."""
    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = []
        mock_streamlit.columns.return_value = (MagicMock(), MagicMock())

        home.show_home_page()

        # Verify columns were created for layout
        assert mock_streamlit.columns.call_count >= 1


def test_show_home_page_navigation_buttons(mock_streamlit):
    """Test home page has navigation buttons to other pages."""
    with patch("f1_predict.web.pages.home.get_upcoming_races") as mock_races:
        mock_races.return_value = []
        mock_streamlit.button.return_value = False

        home.show_home_page()

        # Verify action buttons (Generate Prediction, Compare, Analytics) exist
        button_calls = mock_streamlit.button.call_args_list
        assert len(button_calls) >= 1
