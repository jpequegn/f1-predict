"""Tests for prediction page."""
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd

from f1_predict.web.pages.predict import show_prediction_page


def test_show_prediction_page_renders_without_error():
    """Test that prediction page renders without errors."""
    # Mock Streamlit functions
    mock_col1 = MagicMock()
    mock_col2 = MagicMock()
    mock_session_state = MagicMock()
    mock_session_state.settings = {}
    mock_session_state.get.return_value = False

    with patch('f1_predict.web.pages.predict.st.title'), \
         patch('f1_predict.web.pages.predict.st.columns', return_value=[mock_col1, mock_col2]), \
         patch('f1_predict.web.pages.predict.st.selectbox'), \
         patch('f1_predict.web.pages.predict.st.button'), \
         patch('f1_predict.web.pages.predict.st.session_state', mock_session_state), \
         patch('f1_predict.web.pages.predict.get_upcoming_races') as mock_races, \
         patch('f1_predict.web.pages.predict.st.warning'):

        # Mock empty races
        mock_races.return_value = pd.DataFrame({
            'round': [],
            'race_name': [],
            'circuit_name': [],
            'race_date': [],
            'season': [],
        })

        # Should not raise an exception
        show_prediction_page()


def test_show_prediction_page_displays_race_selection():
    """Test that prediction page includes race selection."""
    mock_col1 = MagicMock()
    mock_col2 = MagicMock()
    mock_session_state = MagicMock()
    mock_session_state.settings = {}
    mock_session_state.get.return_value = False

    with patch('f1_predict.web.pages.predict.st.selectbox') as mock_selectbox:
        with patch('f1_predict.web.pages.predict.st.title'), \
             patch('f1_predict.web.pages.predict.st.columns', return_value=[mock_col1, mock_col2]), \
             patch('f1_predict.web.pages.predict.st.button'), \
             patch('f1_predict.web.pages.predict.st.session_state', mock_session_state), \
             patch('f1_predict.web.pages.predict.get_upcoming_races') as mock_races, \
             patch('f1_predict.web.pages.predict.st.markdown'), \
             patch('f1_predict.web.pages.predict.st.radio'), \
             patch('f1_predict.web.pages.predict.st.checkbox'):

            # Mock races with data
            mock_races.return_value = pd.DataFrame({
                'round': [21],
                'race_name': ['Test Race'],
                'circuit_name': ['Test Circuit'],
                'race_date': ['2025-12-07'],
                'season': [2025],
            })

            show_prediction_page()

            # Verify selectbox was called for race selection
            assert mock_selectbox.called
