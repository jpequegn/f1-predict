"""Settings page for F1 Race Predictor web app."""

import streamlit as st


def show_settings_page() -> None:
    """Display the application settings and preferences."""
    st.title("⚙️ Settings")
    st.info("🚧 Settings page coming soon!")

    st.markdown("""
    **Features:**
    - Color theme selection
    - Timezone configuration
    - Unit preferences (metric/imperial)
    - Default ML model selection
    - Confidence threshold adjustment
    - API configuration
    - Cache settings
    """)
