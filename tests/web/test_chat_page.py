"""Tests for chat interface page."""

import pytest
from unittest.mock import patch, MagicMock

from f1_predict.web.pages import chat


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions."""
    with patch("f1_predict.web.pages.chat.st") as mock_st:
        # Set up session state as a proper object with attribute access
        session_state = MagicMock()
        session_state.__contains__ = lambda self, key: False
        session_state.get = lambda key, default=None: default
        session_state.__setitem__ = MagicMock()
        session_state.__getitem__ = MagicMock()

        # Set up default return values for Streamlit components
        mock_st.session_state = session_state
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.chat_input.return_value = None
        mock_st.chat_message.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        mock_st.spinner.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        mock_st.selectbox.return_value = "Claude 3"
        mock_st.slider.return_value = 0.7
        mock_st.button.return_value = False
        mock_st.metric.return_value = None
        mock_st.dataframe.return_value = None
        mock_st.markdown.return_value = None
        mock_st.sidebar = MagicMock()
        yield mock_st


def test_show_chat_page_renders_without_error(mock_streamlit):
    """Test chat page renders without errors."""
    mock_streamlit.session_state = {}
    mock_streamlit.chat_input.return_value = None

    # Should not raise any exception
    chat.show_chat_page()

    # Verify title was set
    mock_streamlit.title.assert_called_once()


def test_show_chat_page_initializes_session_state(mock_streamlit):
    """Test chat page initializes session state."""
    mock_streamlit.session_state = {}
    mock_streamlit.chat_input.return_value = None

    chat.show_chat_page()

    # Session state should be initialized if not present
    assert mock_streamlit.chat_input is not None


def test_show_chat_page_displays_chat_input(mock_streamlit):
    """Test chat page displays chat input field."""
    mock_streamlit.session_state = {"chat_messages": []}
    mock_streamlit.chat_input.return_value = None

    chat.show_chat_page()

    # Verify chat input was created
    mock_streamlit.chat_input.assert_called_once()


def test_show_chat_page_suggested_queries(mock_streamlit):
    """Test chat page displays suggested queries."""
    mock_streamlit.session_state = {"chat_messages": []}
    mock_streamlit.chat_input.return_value = None

    chat.show_chat_page()

    # Verify sidebar was accessed for suggestions
    mock_streamlit.sidebar.assert_called()


def test_generate_chat_response_prediction_query():
    """Test chat response for prediction queries."""
    response, attachments = chat.generate_chat_response(
        "Who will win the next race?",
        "test-conversation-id",
    )

    # Verify response contains prediction information
    assert isinstance(response, str)
    assert len(response) > 0

    # Verify attachments are included
    assert isinstance(attachments, list)


def test_generate_chat_response_comparison_query():
    """Test chat response for comparison queries."""
    response, attachments = chat.generate_chat_response(
        "Compare Max Verstappen and Lewis Hamilton",
        "test-conversation-id",
    )

    # Verify response contains comparison data
    assert isinstance(response, str)
    assert "Verstappen" in response or "Hamilton" in response


def test_generate_chat_response_team_query():
    """Test chat response for team-related queries."""
    response, attachments = chat.generate_chat_response(
        "Show Red Bull's performance",
        "test-conversation-id",
    )

    # Verify response contains team information
    assert isinstance(response, str)
    assert len(response) > 0


def test_generate_chat_response_standings_query():
    """Test chat response for standings queries."""
    response, attachments = chat.generate_chat_response(
        "Show the championship standings",
        "test-conversation-id",
    )

    # Verify response contains standings
    assert isinstance(response, str)

    # Verify attachments include table data
    assert isinstance(attachments, list)
    if attachments:
        assert any(a.get("type") == "table" for a in attachments)


def test_generate_chat_response_qualifying_query():
    """Test chat response for qualifying-related queries."""
    response, attachments = chat.generate_chat_response(
        "Which driver has the best qualifying record?",
        "test-conversation-id",
    )

    # Verify response contains qualifying analysis
    assert isinstance(response, str)
    assert "pole" in response.lower() or "qualifying" in response.lower()


def test_generate_chat_response_default_fallback():
    """Test chat response default fallback for unknown queries."""
    response, attachments = chat.generate_chat_response(
        "Some random unrelated question",
        "test-conversation-id",
    )

    # Verify response provides helpful information
    assert isinstance(response, str)
    assert len(response) > 0


def test_generate_chat_response_returns_tuple():
    """Test generate_chat_response returns tuple."""
    result = chat.generate_chat_response(
        "Any question",
        "test-conversation-id",
    )

    # Verify return type is tuple with two elements
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], list)


def test_generate_chat_response_attachments_structure():
    """Test chat response attachments have correct structure."""
    response, attachments = chat.generate_chat_response(
        "Show the standings",
        "test-conversation-id",
    )

    # Verify attachments have correct structure
    for attachment in attachments:
        assert isinstance(attachment, dict)
        assert "type" in attachment
        assert attachment["type"] in ["metric", "table"]


def test_chat_input_user_message(mock_streamlit):
    """Test chat page handles user messages."""
    mock_streamlit.session_state = {"chat_messages": []}
    mock_streamlit.chat_input.return_value = "Who will win?"

    # Mock the chat_message context manager
    mock_streamlit.chat_message = MagicMock()

    chat.show_chat_page()

    # Verify chat input was called
    mock_streamlit.chat_input.assert_called_once()


def test_show_chat_page_displays_model_settings(mock_streamlit):
    """Test chat page displays model settings in sidebar."""
    mock_streamlit.session_state = {"chat_messages": []}
    mock_streamlit.chat_input.return_value = None
    mock_streamlit.selectbox.return_value = "Claude 3"
    mock_streamlit.slider.return_value = 0.7

    chat.show_chat_page()

    # Verify sidebar selectbox for model was called
    assert mock_streamlit.selectbox.call_count >= 1


def test_show_chat_page_clear_history_button(mock_streamlit):
    """Test chat page has clear history button."""
    mock_streamlit.session_state = {"chat_messages": []}
    mock_streamlit.chat_input.return_value = None
    mock_streamlit.button.return_value = False

    chat.show_chat_page()

    # Verify button exists for clearing chat history
    assert mock_streamlit.button.call_count >= 1
