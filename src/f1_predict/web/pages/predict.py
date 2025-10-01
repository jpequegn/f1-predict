"""Race prediction page for F1 Race Predictor web app."""

import streamlit as st


def show_prediction_page() -> None:
    """Display the race prediction interface."""
    st.title("🏁 Race Prediction")
    st.info("🚧 Prediction interface coming soon!")

    st.markdown("""
    **Features:**
    - Select upcoming race
    - Choose ML model (Ensemble, XGBoost, LightGBM, Random Forest)
    - Generate predictions with confidence scores
    - View predicted podium and full race order
    - Explore feature importance and explanations
    """)
