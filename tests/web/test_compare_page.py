"""Tests for comparison page."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import streamlit as st

from f1_predict.web.pages import compare


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions."""
    with patch("f1_predict.web.pages.compare.st") as mock_st:
        # Set up default return values for Streamlit components
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_st.tabs.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_st.radio.return_value = "Drivers"
        mock_st.selectbox.return_value = "Max Verstappen"
        mock_st.button.return_value = False
        mock_st.metric.return_value = None
        mock_st.dataframe.return_value = None
        mock_st.markdown.return_value = None
        yield mock_st


def test_show_comparison_page_renders_without_error(mock_streamlit):
    """Test comparison page renders without errors."""
    mock_streamlit.radio.return_value = "Drivers"

    # Should not raise any exception
    compare.show_comparison_page()

    # Verify title was set
    mock_streamlit.title.assert_called_once()


def test_show_comparison_page_radio_selector(mock_streamlit):
    """Test comparison page displays comparison type selector."""
    mock_streamlit.radio.return_value = "Drivers"

    compare.show_comparison_page()

    # Verify radio button was created
    mock_streamlit.radio.assert_called_once()
    call_args = mock_streamlit.radio.call_args
    assert "Drivers" in str(call_args) or "Teams" in str(call_args)


def test_show_driver_comparison(mock_streamlit):
    """Test driver comparison interface."""
    mock_streamlit.selectbox.side_effect = [
        "Max Verstappen",
        "Lewis Hamilton",
        "All Circuits",
        2024,
    ]
    mock_streamlit.button.return_value = False

    compare.show_driver_comparison()

    # Verify selectboxes were created for driver selection
    assert mock_streamlit.selectbox.call_count >= 2


def test_show_driver_comparison_filters(mock_streamlit):
    """Test driver comparison has circuit and season filters."""
    mock_streamlit.selectbox.side_effect = [
        "Max Verstappen",
        "Lewis Hamilton",
        "Monaco",
        2024,
    ]
    mock_streamlit.button.return_value = False

    compare.show_driver_comparison()

    # Verify all selectboxes were called (drivers, circuit, season)
    assert mock_streamlit.selectbox.call_count >= 4


def test_show_team_comparison(mock_streamlit):
    """Test team comparison interface."""
    mock_streamlit.selectbox.side_effect = [
        "Red Bull Racing",
        "Mercedes",
        2024,
    ]
    mock_streamlit.button.return_value = False

    compare.show_team_comparison()

    # Verify selectboxes were created for team selection
    assert mock_streamlit.selectbox.call_count >= 2


def test_display_driver_comparison_shows_statistics(mock_streamlit):
    """Test driver comparison displays head-to-head statistics."""
    compare._display_driver_comparison(
        "Max Verstappen",
        "Lewis Hamilton",
        "All Circuits",
        2024,
    )

    # Verify metrics were displayed
    assert mock_streamlit.metric.call_count >= 3


def test_display_driver_comparison_shows_tabs(mock_streamlit):
    """Test driver comparison displays performance tabs."""
    mock_streamlit.tabs.return_value = (MagicMock(), MagicMock(), MagicMock())

    compare._display_driver_comparison(
        "Max Verstappen",
        "Lewis Hamilton",
        "All Circuits",
        2024,
    )

    # Verify tabs were created for different performance metrics
    mock_streamlit.tabs.assert_called()


def test_display_driver_comparison_circuit_specific_stats(mock_streamlit):
    """Test driver comparison shows circuit-specific stats when filtered."""
    compare._display_driver_comparison(
        "Max Verstappen",
        "Lewis Hamilton",
        "Monaco",
        2024,
    )

    # Verify dataframe was displayed for circuit stats
    mock_streamlit.dataframe.assert_called()


def test_display_team_comparison_shows_statistics(mock_streamlit):
    """Test team comparison displays team statistics."""
    compare._display_team_comparison(
        "Red Bull Racing",
        "Mercedes",
        2024,
    )

    # Verify metrics were displayed (constructor points, wins, podiums)
    assert mock_streamlit.metric.call_count >= 3


def test_display_team_comparison_shows_driver_breakdown(mock_streamlit):
    """Test team comparison shows driver breakdown."""
    compare._display_team_comparison(
        "Red Bull Racing",
        "Mercedes",
        2024,
    )

    # Verify dataframes were displayed for each team's drivers
    assert mock_streamlit.dataframe.call_count >= 2


def test_display_team_comparison_reliability(mock_streamlit):
    """Test team comparison displays reliability metrics."""
    compare._display_team_comparison(
        "Red Bull Racing",
        "Mercedes",
        2024,
    )

    # Verify DNF rate metrics were displayed
    assert mock_streamlit.metric.call_count >= 5


def test_comparison_page_constants_defined():
    """Test that comparison data constants are defined."""
    # Verify drivers list is defined
    assert hasattr(compare, "DRIVERS")
    assert len(compare.DRIVERS) > 0

    # Verify teams list is defined
    assert hasattr(compare, "TEAMS")
    assert len(compare.TEAMS) > 0

    # Verify circuits list is defined
    assert hasattr(compare, "CIRCUITS")
    assert len(compare.CIRCUITS) > 0

    # Verify seasons list is defined
    assert hasattr(compare, "SEASONS")
    assert len(compare.SEASONS) > 0
