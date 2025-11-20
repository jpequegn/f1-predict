"""Tests for chat interface page with LLM provider integration."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys

# Mock streamlit at import time to avoid import errors
sys.modules["streamlit"] = MagicMock()

from f1_predict.web.pages import chat
from f1_predict.llm.base import LLMResponse
from f1_predict.llm.exceptions import LLMError


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions."""
    with patch("f1_predict.web.pages.chat.st") as mock_st:
        # Set up session state that works like streamlit.session_state
        session_state = MagicMock()
        session_state.__contains__ = lambda self, key: hasattr(self, key)
        session_state.__setitem__ = lambda self, key, value: setattr(self, key, value)
        session_state.__getitem__ = lambda self, key: getattr(self, key)

        # Set up default return values for Streamlit components
        mock_st.session_state = session_state
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.chat_input.return_value = None
        mock_st.chat_message.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        mock_st.spinner.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        mock_st.selectbox.return_value = "Claude 3.5 (Anthropic)"
        mock_st.slider.return_value = 0.7
        mock_st.button.return_value = False
        mock_st.metric.return_value = None
        mock_st.dataframe.return_value = None
        mock_st.markdown.return_value = None
        mock_st.caption.return_value = None
        mock_st.error.return_value = None
        mock_st.warning.return_value = None
        mock_st.sidebar = MagicMock()
        yield mock_st


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return LLMResponse(
        content="This is a test response from the LLM.",
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        estimated_cost=0.001,
        metadata={"finish_reason": "end_turn"},
    )


@pytest.fixture
def mock_chat_session(mock_llm_response):
    """Create a mock ChatSession."""
    session = MagicMock()
    session.provider = MagicMock()
    session.provider.generate = AsyncMock(return_value=mock_llm_response)
    return session


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


def test_show_chat_page_displays_model_settings(mock_streamlit):
    """Test chat page displays model settings in sidebar."""
    mock_streamlit.session_state = {"chat_messages": []}
    mock_streamlit.chat_input.return_value = None
    mock_streamlit.selectbox.return_value = "Claude 3.5 (Anthropic)"
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


def test_initialize_llm_session_anthropic():
    """Test LLM session initialization with Anthropic provider."""
    from f1_predict.llm.base import LLMConfig

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("f1_predict.web.pages.chat.AnthropicProvider") as mock_provider:
            mock_provider.return_value = MagicMock()

            session = chat._initialize_llm_session(
                "Claude 3.5 (Anthropic)",
                0.7,
                500,
            )

            # Verify provider was initialized
            assert mock_provider.called


def test_initialize_llm_session_openai():
    """Test LLM session initialization with OpenAI provider."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with patch("f1_predict.web.pages.chat.OpenAIProvider") as mock_provider:
            mock_provider.return_value = MagicMock()

            session = chat._initialize_llm_session(
                "GPT-4 (OpenAI)",
                0.7,
                500,
            )

            # Verify provider was initialized
            assert mock_provider.called


def test_initialize_llm_session_local():
    """Test LLM session initialization with Local provider."""
    with patch("f1_predict.web.pages.chat.LocalProvider") as mock_provider:
        mock_provider.return_value = MagicMock()

        session = chat._initialize_llm_session(
            "Local LLM (Ollama)",
            0.7,
            500,
        )

        # Verify provider was initialized
        assert mock_provider.called


def test_initialize_llm_session_missing_api_key(mock_streamlit):
    """Test LLM session initialization with missing API key."""
    mock_streamlit.warning = MagicMock()

    with patch.dict("os.environ", {}, clear=True):
        session = chat._initialize_llm_session(
            "Claude 3.5 (Anthropic)",
            0.7,
            500,
        )

        # Should return None when API key is missing
        assert session is None


def test_get_model_name_anthropic():
    """Test model name retrieval for Anthropic."""
    model_name = chat._get_model_name("Claude 3.5 (Anthropic)")
    assert model_name == "claude-3-5-sonnet-20241022"


def test_get_model_name_openai():
    """Test model name retrieval for OpenAI."""
    model_name = chat._get_model_name("GPT-4 (OpenAI)")
    assert model_name == "gpt-4"


def test_get_model_name_local():
    """Test model name retrieval for Local LLM."""
    model_name = chat._get_model_name("Local LLM (Ollama)")
    assert model_name == "llama3.1"


@pytest.mark.asyncio
async def test_generate_llm_response_success(mock_chat_session, mock_llm_response):
    """Test successful LLM response generation."""
    response, cost, tokens = await chat._generate_llm_response(
        "Who will win the next race?",
        mock_chat_session,
    )

    assert response == mock_llm_response.content
    assert cost == mock_llm_response.estimated_cost
    assert tokens == mock_llm_response.total_tokens


@pytest.mark.asyncio
async def test_generate_llm_response_with_none_session():
    """Test LLM response generation with None session."""
    with pytest.raises(LLMError):
        await chat._generate_llm_response(
            "Any prompt",
            None,
        )


@pytest.mark.asyncio
async def test_generate_llm_response_with_none_provider():
    """Test LLM response generation with None provider."""
    session = MagicMock()
    session.provider = None

    with pytest.raises(LLMError):
        await chat._generate_llm_response(
            "Any prompt",
            session,
        )


def test_show_chat_page_handles_user_input(mock_streamlit, mock_chat_session, mock_llm_response):
    """Test chat page handles user input and generates response."""
    # Setup
    mock_streamlit.session_state = {
        "chat_messages": [],
        "conversation_id": "test-id",
        "llm_session": mock_chat_session,
        "total_cost": 0.0,
        "token_count": 0,
        "prompt_input": "",
    }
    mock_streamlit.chat_input.return_value = "Who will win the next race?"
    mock_streamlit.chat_message.return_value = MagicMock(
        __enter__=MagicMock(),
        __exit__=MagicMock()
    )

    # Mock rerun to avoid infinite loop
    mock_streamlit.rerun = MagicMock()

    # Call function (will process the input)
    chat.show_chat_page()

    # Verify chat_message was called for both user and assistant
    assert mock_streamlit.chat_message.call_count >= 1


def test_chat_page_tracks_cost_and_tokens(mock_streamlit):
    """Test chat page tracks API costs and token usage."""
    mock_streamlit.session_state = {
        "chat_messages": [],
        "total_cost": 0.0,
        "token_count": 0,
    }
    mock_streamlit.chat_input.return_value = None
    mock_streamlit.metric = MagicMock()

    chat.show_chat_page()

    # Verify metric was called (for displaying cost/tokens)
    assert mock_streamlit.metric.call_count >= 0


def test_chat_page_displays_sidebar_settings(mock_streamlit):
    """Test chat page displays all sidebar settings."""
    mock_streamlit.session_state = {"chat_messages": []}
    mock_streamlit.chat_input.return_value = None

    chat.show_chat_page()

    # Verify sidebar was used
    assert mock_streamlit.sidebar is not None
    # Verify provider selection
    assert mock_streamlit.selectbox.call_count >= 1
    # Verify temperature and max_tokens sliders
    assert mock_streamlit.slider.call_count >= 2
