"""Tests for settings page."""

import pytest
from unittest.mock import patch, MagicMock

from f1_predict.web.pages import settings


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions."""
    with patch("f1_predict.web.pages.settings.st") as mock_st:
        # Set up session state as a proper object with attribute access
        session_state = MagicMock()
        session_state.__setitem__ = MagicMock()
        session_state.__getitem__ = MagicMock()
        session_state.get = lambda key, default=None: default

        # Set up default return values for Streamlit components
        mock_st.session_state = session_state
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_st.tabs.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "Nebula Dark"
        mock_st.radio.return_value = "Metric (km/h, °C)"
        mock_st.slider.return_value = 0.7
        mock_st.checkbox.return_value = False
        mock_st.number_input.return_value = 5
        mock_st.text_input.return_value = "https://ergast.com/api/f1"
        mock_st.button.return_value = False
        mock_st.metric.return_value = None
        mock_st.info.return_value = None
        mock_st.success.return_value = None
        mock_st.markdown.return_value = None
        yield mock_st


def test_show_settings_page_renders_without_error(mock_streamlit):
    """Test settings page renders without errors."""
    # Should not raise any exception
    settings.show_settings_page()

    # Verify title was set
    mock_streamlit.title.assert_called_once()


def test_show_settings_page_displays_tabs(mock_streamlit):
    """Test settings page displays settings tabs."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    settings.show_settings_page()

    # Verify tabs were created
    mock_streamlit.tabs.assert_called_once()
    call_args = mock_streamlit.tabs.call_args
    # Should have 4 tabs: General, Predictions, Display, API
    assert "General" in str(call_args)


def test_show_settings_page_general_tab(mock_streamlit):
    """Test settings page general tab."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mock_streamlit.selectbox.side_effect = [
        "Nebula Dark",
        "UTC",
        "Metric (km/h, °C)",
        "English",
    ]
    mock_streamlit.radio.return_value = "Metric (km/h, °C)"

    settings.show_settings_page()

    # Verify general settings were configured
    assert mock_streamlit.selectbox.call_count >= 3


def test_show_settings_page_prediction_settings(mock_streamlit):
    """Test settings page prediction settings tab."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mock_streamlit.selectbox.side_effect = [
        "Nebula Dark",  # theme
        "UTC",  # timezone
        "English",  # language
        "Ensemble",  # default model
    ]
    mock_streamlit.slider.return_value = 0.7
    mock_streamlit.checkbox.side_effect = [True, False, False, False]
    mock_streamlit.radio.return_value = "Metric (km/h, °C)"
    mock_streamlit.button.return_value = False

    settings.show_settings_page()

    # Verify prediction settings were configured
    assert mock_streamlit.slider.call_count >= 1


def test_show_settings_page_display_settings(mock_streamlit):
    """Test settings page display settings tab."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mock_streamlit.selectbox.side_effect = [
        "Nebula Dark",
        "UTC",
        "English",
        "Interactive (Plotly)",
    ]
    mock_streamlit.number_input.side_effect = [5, 10]
    mock_streamlit.checkbox.side_effect = [True, False, False, True]
    mock_streamlit.radio.return_value = "Metric (km/h, °C)"
    mock_streamlit.button.return_value = False

    settings.show_settings_page()

    # Verify display settings were configured
    assert mock_streamlit.number_input.call_count >= 2


def test_show_settings_page_api_settings(mock_streamlit):
    """Test settings page API configuration tab."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mock_streamlit.selectbox.side_effect = [
        "Nebula Dark",
        "UTC",
        "English",
        "Interactive (Plotly)",
    ]
    mock_streamlit.text_input.return_value = "https://ergast.com/api/f1"
    mock_streamlit.number_input.side_effect = [4, 60, 30, 5, 10]
    mock_streamlit.checkbox.side_effect = [True, False, False, False]
    mock_streamlit.radio.return_value = "Metric (km/h, °C)"
    mock_streamlit.button.return_value = False

    settings.show_settings_page()

    # Verify API settings were configured
    assert mock_streamlit.text_input.call_count >= 1


def test_show_settings_page_save_button(mock_streamlit):
    """Test settings page has save button."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mock_streamlit.selectbox.side_effect = [
        "Nebula Dark",
        "UTC",
        "English",
        "Ensemble",
        "Interactive (Plotly)",
    ]
    mock_streamlit.radio.return_value = "Metric (km/h, °C)"
    mock_streamlit.slider.return_value = 0.7
    mock_streamlit.checkbox.return_value = False
    mock_streamlit.number_input.return_value = 5
    mock_streamlit.text_input.return_value = "https://ergast.com/api/f1"
    mock_streamlit.button.return_value = False

    settings.show_settings_page()

    # Verify buttons were created (Save, Reset, Help)
    assert mock_streamlit.button.call_count >= 1


def test_show_settings_page_save_settings(mock_streamlit):
    """Test settings page saves settings correctly."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mock_streamlit.selectbox.side_effect = [
        "Nebula Dark",
        "UTC",
        "English",
        "Ensemble",
        "Interactive (Plotly)",
    ]
    mock_streamlit.radio.return_value = "Metric (km/h, °C)"
    mock_streamlit.slider.return_value = 0.7
    mock_streamlit.checkbox.return_value = False
    mock_streamlit.number_input.return_value = 5
    mock_streamlit.text_input.return_value = "https://ergast.com/api/f1"
    # Make save button click
    def button_side_effect(label, **kwargs):
        return "Save Settings" in label
    mock_streamlit.button.side_effect = button_side_effect

    settings.show_settings_page()

    # Verify success message was shown
    mock_streamlit.success.assert_called()


def test_show_settings_page_reset_button(mock_streamlit):
    """Test settings page has reset to defaults button."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mock_streamlit.selectbox.side_effect = [
        "Nebula Dark",
        "UTC",
        "English",
        "Ensemble",
        "Interactive (Plotly)",
    ]
    mock_streamlit.radio.return_value = "Metric (km/h, °C)"
    mock_streamlit.slider.return_value = 0.7
    mock_streamlit.checkbox.return_value = False
    mock_streamlit.number_input.return_value = 5
    mock_streamlit.text_input.return_value = "https://ergast.com/api/f1"
    mock_streamlit.button.return_value = False

    settings.show_settings_page()

    # Verify reset button was created
    assert mock_streamlit.button.call_count >= 1


def test_show_settings_page_help_button(mock_streamlit):
    """Test settings page has help button."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mock_streamlit.selectbox.side_effect = [
        "Nebula Dark",
        "UTC",
        "English",
        "Ensemble",
        "Interactive (Plotly)",
    ]
    mock_streamlit.radio.return_value = "Metric (km/h, °C)"
    mock_streamlit.slider.return_value = 0.7
    mock_streamlit.checkbox.return_value = False
    mock_streamlit.number_input.return_value = 5
    mock_streamlit.text_input.return_value = "https://ergast.com/api/f1"
    mock_streamlit.button.return_value = False

    settings.show_settings_page()

    # Verify help button was created
    assert mock_streamlit.button.call_count >= 1


def test_show_settings_page_about_section(mock_streamlit):
    """Test settings page displays about section."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    mock_streamlit.selectbox.side_effect = [
        "Nebula Dark",
        "UTC",
        "English",
        "Ensemble",
        "Interactive (Plotly)",
    ]
    mock_streamlit.radio.return_value = "Metric (km/h, °C)"
    mock_streamlit.slider.return_value = 0.7
    mock_streamlit.checkbox.return_value = False
    mock_streamlit.number_input.return_value = 5
    mock_streamlit.text_input.return_value = "https://ergast.com/api/f1"
    mock_streamlit.button.return_value = False

    settings.show_settings_page()

    # Verify metrics were displayed in about section
    assert mock_streamlit.metric.call_count >= 3


def test_show_settings_page_theme_options(mock_streamlit):
    """Test settings page displays available themes."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    settings.show_settings_page()

    # Verify theme selectbox was called
    selectbox_calls = [
        call for call in mock_streamlit.selectbox.call_args_list
        if "Color Theme" in str(call) or "Nebula" in str(call)
    ]
    assert len(selectbox_calls) > 0


def test_show_settings_page_timezone_options(mock_streamlit):
    """Test settings page displays timezone options."""
    mock_streamlit.tabs.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    settings.show_settings_page()

    # Verify timezone selectbox was called with timezone options
    timezone_calls = [
        call for call in mock_streamlit.selectbox.call_args_list
        if "Timezone" in str(call) or "UTC" in str(call)
    ]
    assert len(timezone_calls) > 0
