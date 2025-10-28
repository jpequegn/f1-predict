"""Tests for analytics page."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, Mock
import streamlit as st

from f1_predict.web.pages import analytics


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions."""
    with patch("f1_predict.web.pages.analytics.st") as mock_st:
        # Set up default return values for Streamlit components
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_st.tabs.return_value = (MagicMock(), MagicMock())
        mock_st.select_slider.return_value = "Current Season"
        mock_st.checkbox.return_value = False
        mock_st.metric.return_value = None
        mock_st.selectbox.return_value = None
        mock_st.dataframe.return_value = None
        mock_st.info.return_value = None
        yield mock_st


def test_show_analytics_page_renders_without_error(mock_streamlit):
    """Test analytics page renders without errors."""
    with patch("f1_predict.web.pages.analytics.calculate_kpi_metrics") as mock_kpis:
        with patch("f1_predict.web.pages.analytics.get_championship_standings") as mock_standings:
            mock_kpis.return_value = {
                "races_analyzed": 10,
                "prediction_accuracy": 0.85,
                "avg_confidence": 0.78,
                "total_predictions": 100,
            }
            mock_standings.return_value = pd.DataFrame()

            # Should not raise any exception
            analytics.show_analytics_page()

            # Verify title was set
            mock_streamlit.title.assert_called_once()


def test_show_analytics_page_displays_kpi_metrics(mock_streamlit):
    """Test analytics page displays KPI metrics."""
    with patch("f1_predict.web.pages.analytics.calculate_kpi_metrics") as mock_kpis:
        with patch("f1_predict.web.pages.analytics.get_championship_standings"):
            mock_kpis.return_value = {
                "races_analyzed": 25,
                "prediction_accuracy": 0.87,
                "avg_confidence": 0.82,
                "total_predictions": 250,
            }

            analytics.show_analytics_page()

            # Verify metrics were displayed
            assert mock_streamlit.metric.call_count >= 4


def test_analytics_page_displays_championship_standings(mock_streamlit):
    """Test analytics page displays championship standings."""
    standings_df = pd.DataFrame({
        "position": [1, 2, 3],
        "driver": ["Driver A", "Driver B", "Driver C"],
        "team": ["Team A", "Team B", "Team C"],
        "points": [250, 200, 180],
        "wins": [5, 4, 3],
        "podiums": [12, 10, 8],
    })

    with patch("f1_predict.web.pages.analytics.calculate_kpi_metrics"):
        with patch("f1_predict.web.pages.analytics.get_championship_standings") as mock_standings:
            mock_standings.return_value = standings_df

            analytics.show_analytics_page()

            # Verify dataframe was displayed
            mock_streamlit.dataframe.assert_called()


def test_analytics_page_displays_performance_charts(mock_streamlit):
    """Test analytics page displays performance analysis charts."""
    with patch("f1_predict.web.pages.analytics.calculate_kpi_metrics"):
        with patch("f1_predict.web.pages.analytics.get_championship_standings"):
            with patch("f1_predict.web.pages.analytics.px.bar"):
                analytics.show_analytics_page()

                # Verify selectbox for time period was called
                mock_streamlit.select_slider.assert_called_once()


def test_analytics_page_time_period_selector(mock_streamlit):
    """Test analytics page has time period selector."""
    with patch("f1_predict.web.pages.analytics.calculate_kpi_metrics"):
        with patch("f1_predict.web.pages.analytics.get_championship_standings"):
            mock_streamlit.select_slider.return_value = "Current Season"

            analytics.show_analytics_page()

            # Verify time period selector was created
            mock_streamlit.select_slider.assert_called_once()
            call_args = mock_streamlit.select_slider.call_args
            assert "Time Period" in str(call_args)


def test_analytics_page_displays_tabs(mock_streamlit):
    """Test analytics page displays tabs for standings."""
    with patch("f1_predict.web.pages.analytics.calculate_kpi_metrics"):
        with patch("f1_predict.web.pages.analytics.get_championship_standings"):
            mock_streamlit.tabs.return_value = (MagicMock(), MagicMock())

            analytics.show_analytics_page()

            # Verify tabs were created
            mock_streamlit.tabs.assert_called()


def test_analytics_page_auto_refresh_option(mock_streamlit):
    """Test analytics page has auto-refresh checkbox."""
    with patch("f1_predict.web.pages.analytics.calculate_kpi_metrics"):
        with patch("f1_predict.web.pages.analytics.get_championship_standings"):
            mock_streamlit.checkbox.return_value = False

            analytics.show_analytics_page()

            # Verify checkbox for auto-refresh exists
            mock_streamlit.checkbox.assert_called()
