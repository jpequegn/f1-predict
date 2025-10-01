"""Driver and team comparison page for F1 Race Predictor web app."""

import streamlit as st


def show_comparison_page() -> None:
    """Display the driver/team comparison interface."""
    st.title("📊 Driver & Team Comparison")
    st.info("🚧 Comparison tools coming soon!")

    st.markdown("""
    **Features:**
    - Compare two drivers head-to-head
    - Compare team performance
    - Filter by circuit or season
    - View race results, qualifying, and points trends
    - Circuit-specific performance analysis
    """)
