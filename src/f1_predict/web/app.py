"""F1 Race Predictor - Streamlit Web Application.

Modern, interactive web interface for F1 race predictions, data visualization,
and analysis. Built with Streamlit and Nebula UI design system.
"""

import streamlit as st
from streamlit_option_menu import option_menu

from f1_predict.web.pages import (
    analytics,
    chat,
    compare,
    explainability,
    home,
    predict,
    settings,
)
from f1_predict.web.utils.theme import apply_nebula_theme


def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="F1 Race Predictor",
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/jpequegn/f1-predict",
            "Report a bug": "https://github.com/jpequegn/f1-predict/issues",
            "About": "# F1 Race Predictor\nML-powered F1 race predictions",
        },
    )

    # Apply Nebula UI theme
    apply_nebula_theme()

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.settings = {
            "theme": "Nebula Dark",
            "timezone": "UTC",
            "units": "metric",
            "default_model": "Ensemble",
            "confidence_threshold": 0.7,
            "enable_explanations": True,
        }

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🏎️ F1 Race Predictor")
        st.markdown("---")

        selected = option_menu(
            menu_title=None,
            options=["Home", "Predict", "Compare", "Analytics", "Explainability", "Chat", "Settings"],
            icons=[
                "house-fill",
                "trophy-fill",
                "bar-chart-fill",
                "graph-up",
                "search",
                "chat-dots-fill",
                "gear-fill",
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "#1E2130"},
                "icon": {"color": "#A3A9BF", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "4px",
                    "padding": "8px 12px",
                    "border-radius": "4px",
                    "color": "#E0E6F0",
                    "background-color": "transparent",
                },
                "nav-link-selected": {
                    "background-color": "#1F4E8C",
                    "color": "#E0E6F0",
                    "font-weight": "500",
                },
            },
        )

        # Footer
        st.markdown("---")
        st.markdown(
            '<small style="color: #A3A9BF;">Powered by ML & F1 Data</small>',
            unsafe_allow_html=True,
        )

    # Route to pages
    if selected == "Home":
        home.show_home_page()
    elif selected == "Predict":
        predict.show_prediction_page()
    elif selected == "Compare":
        compare.show_comparison_page()
    elif selected == "Analytics":
        analytics.show_analytics_page()
    elif selected == "Explainability":
        explainability.show_explainability_page()
    elif selected == "Chat":
        chat.show_chat_page()
    elif selected == "Settings":
        settings.show_settings_page()


if __name__ == "__main__":
    main()
