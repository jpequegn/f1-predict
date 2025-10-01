"""Chat interface page for F1 Race Predictor web app."""

import streamlit as st


def show_chat_page() -> None:
    """Display the LLM-powered chat interface."""
    st.title("💬 F1 Chat Assistant")
    st.info("🚧 Chat interface coming soon!")

    st.markdown("""
    **Features:**
    - Natural language F1 queries
    - Conversational predictions
    - Statistical analysis
    - Race insights and explanations
    - Interactive charts and tables
    - Suggested queries
    """)

    # Placeholder chat interface
    st.markdown("### Try asking:")
    st.markdown("- 'Who will win the next race?'")
    st.markdown("- 'Compare Hamilton and Verstappen'")
    st.markdown("- 'Show me Red Bull's performance this season'")
