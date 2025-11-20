"""Chat interface page for F1 Race Predictor web app with LLM integration."""

import asyncio
import uuid
from typing import Optional

import streamlit as st

from f1_predict.llm.base import LLMConfig
from f1_predict.llm.chat_session import ChatSession, ChatSessionManager
from f1_predict.llm.explanations import F1PredictionExplainer
from f1_predict.llm.anthropic_provider import AnthropicProvider
from f1_predict.llm.openai_provider import OpenAIProvider
from f1_predict.llm.local_provider import LocalProvider
from f1_predict.llm.exceptions import LLMError


def show_chat_page() -> None:
    """Display the LLM-powered chat interface with real LLM integration."""
    st.title("💬 F1 Chat Assistant")

    # Sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ LLM Settings")

        # Model selection
        model_provider = st.selectbox(
            "AI Model Provider",
            options=["Claude 3.5 (Anthropic)", "GPT-4 (OpenAI)", "Local LLM (Ollama)"],
            index=0,
            help="Select which LLM provider to use"
        )

        # Temperature control
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values = more creative, lower = more focused"
        )

        # Max tokens
        max_tokens = st.slider(
            "Response Length",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
        )

        st.markdown("---")
        st.markdown("### 💡 Suggested Queries")

        suggestions = [
            "Who will win the next race?",
            "Compare Max Verstappen and Lewis Hamilton",
            "Show Red Bull's 2024 season performance",
            "Explain this driver's qualifying record",
            "Analyze the championship standings",
            "Which driver performs best in wet conditions?",
        ]

        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
                st.session_state.prompt_input = suggestion
                st.rerun()

        st.markdown("---")

        # Session management
        st.markdown("### 📋 Session Management")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()

        # Cost tracking
        if "total_cost" in st.session_state:
            st.metric("Session Cost", f"${st.session_state.total_cost:.4f}")

        if "token_count" in st.session_state:
            st.metric("Tokens Used", st.session_state.token_count)

    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.llm_session = None
        st.session_state.total_cost = 0.0
        st.session_state.token_count = 0
        st.session_state.prompt_input = ""

    # Initialize LLM provider if not done
    if st.session_state.llm_session is None:
        st.session_state.llm_session = _initialize_llm_session(
            model_provider, temperature, max_tokens
        )

    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("---")

    # Chat input
    prompt = st.chat_input(
        "Ask about F1 predictions, statistics, or race analysis...",
        key="chat_input"
    )

    # Handle suggested query
    if st.session_state.prompt_input:
        prompt = st.session_state.prompt_input
        st.session_state.prompt_input = ""

    if prompt:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Run async function in Streamlit context
                response, cost, tokens = asyncio.run(_generate_llm_response(
                    prompt,
                    st.session_state.llm_session,
                ))

                st.markdown(response)

                # Add assistant message with metadata
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "cost": cost,
                    "tokens": tokens,
                })

                # Update session cost and token tracking
                st.session_state.total_cost += cost
                st.session_state.token_count += tokens

                # Show cost and tokens if available
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"💰 Cost: ${cost:.4f}")
                with col2:
                    st.caption(f"📊 Tokens: {tokens}")

            except LLMError as e:
                st.error(f"LLM Error: {str(e)}")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

        st.rerun()


def _initialize_llm_session(
    model_provider: str,
    temperature: float,
    max_tokens: int,
) -> Optional[ChatSession]:
    """Initialize LLM provider and ChatSession.

    Args:
        model_provider: Selected LLM provider
        temperature: Temperature setting
        max_tokens: Maximum tokens for response

    Returns:
        Initialized ChatSession or None if provider unavailable
    """
    try:
        # Create configuration
        config = LLMConfig(
            model=_get_model_name(model_provider),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Initialize provider
        if "Claude" in model_provider:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.warning("Anthropic API key not found. Please set ANTHROPIC_API_KEY.")
                return None
            provider = AnthropicProvider(config, api_key)
        elif "GPT" in model_provider:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.warning("OpenAI API key not found. Please set OPENAI_API_KEY.")
                return None
            provider = OpenAIProvider(config, api_key)
        elif "Local" in model_provider:
            provider = LocalProvider(config, endpoint="http://localhost:11434")
        else:
            return None

        # Create ChatSession
        return ChatSession(session_id=st.session_state.conversation_id, provider=provider)

    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None


def _get_model_name(model_provider: str) -> str:
    """Get model name from provider selection.

    Args:
        model_provider: User's provider selection

    Returns:
        Model name string
    """
    if "Claude" in model_provider:
        return "claude-3-5-sonnet-20241022"
    elif "GPT-4" in model_provider:
        return "gpt-4"
    elif "Local" in model_provider:
        return "llama3.1"
    return "gpt-3.5-turbo"


async def _generate_llm_response(
    prompt: str,
    llm_session: ChatSession,
) -> tuple[str, float, int]:
    """Generate response using actual LLM provider via ChatSession.

    Args:
        prompt: User's chat message
        llm_session: Initialized ChatSession with provider

    Returns:
        Tuple of (response_text, estimated_cost, total_tokens)
    """
    if not llm_session or not llm_session.provider:
        raise LLMError("LLM session not initialized")

    # Generate response using provider
    response = await llm_session.provider.generate(prompt=prompt)

    # Extract metadata
    content = response.content
    cost = response.estimated_cost
    tokens = response.total_tokens

    return content, cost, tokens
