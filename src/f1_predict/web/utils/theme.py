"""Nebula UI theme configuration for Streamlit app.

This module provides theme styling following the Nebula UI design system:
- Dark, moody aesthetics with dark blue primary accent (#1F4E8C)
- Compact layouts with minimal whitespace
- Sharp, angular components with subtle rounded corners
- Modern, futuristic, elegant appearance
"""

from typing import Dict


def get_nebula_css() -> str:
    """Get Nebula UI CSS styles for Streamlit.

    Returns:
        CSS string with Nebula UI theme styles
    """
    return """
        <style>
        /* ============================================
           Nebula UI Theme - Color Variables
           ============================================ */
        :root {
            --background: #121317;
            --surface: #1E2130;
            --surface-hover: #2A2F45;
            --text-primary: #E0E6F0;
            --text-secondary: #A3A9BF;
            --accent-primary: #1F4E8C;
            --accent-hover: #2762B3;
            --divider: #333A56;
            --success: #28A745;
            --warning: #FFC107;
            --danger: #DC3545;
        }

        /* ============================================
           Global Styles
           ============================================ */
        .stApp {
            background-color: var(--background);
            color: var(--text-primary);
            font-family: 'Inter', 'Roboto', sans-serif;
        }

        /* ============================================
           Typography
           ============================================ */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary);
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        h1 { font-size: 24px; }
        h2 { font-size: 20px; }
        h3 { font-size: 16px; }

        p, div, span, label {
            color: var(--text-primary);
            font-size: 14px;
            line-height: 1.5;
        }

        .stMarkdown small, .stCaption {
            font-size: 12px;
            color: var(--text-secondary);
        }

        /* ============================================
           Buttons
           ============================================ */
        .stButton > button {
            background-color: var(--accent-primary);
            color: var(--text-primary);
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 14px;
            transition: background-color 0.15s ease;
        }

        .stButton > button:hover {
            background-color: var(--accent-hover);
            border: none;
        }

        .stButton > button:active {
            background-color: var(--accent-primary);
            transform: scale(0.98);
        }

        .stButton > button:disabled {
            background-color: #555D7A;
            color: var(--text-secondary);
            cursor: not-allowed;
        }

        /* ============================================
           Input Fields & Forms
           ============================================ */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div {
            background-color: var(--surface);
            border: 1px solid var(--divider);
            border-radius: 2px;
            color: var(--text-primary);
            font-size: 14px;
        }

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div:focus-within {
            border: 2px solid var(--accent-primary);
            outline: none;
            box-shadow: 0 0 0 1px var(--accent-primary);
        }

        /* ============================================
           Cards & Panels
           ============================================ */
        .stMetric {
            background-color: var(--surface);
            border-radius: 4px;
            padding: 16px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
        }

        .stMetric label {
            color: var(--text-secondary);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stMetric > div {
            color: var(--text-primary);
            font-size: 24px;
            font-weight: 700;
        }

        /* ============================================
           Tables & DataFrames
           ============================================ */
        .dataframe {
            background-color: var(--surface);
            color: var(--text-primary);
            border: 1px solid var(--divider);
        }

        .dataframe thead tr th {
            background-color: var(--surface-hover);
            color: var(--text-primary);
            border-bottom: 2px solid var(--divider);
            font-weight: 600;
            padding: 12px 8px;
        }

        .dataframe tbody tr td {
            border-bottom: 1px solid var(--divider);
            padding: 8px;
        }

        .dataframe tbody tr:hover {
            background-color: var(--surface-hover);
        }

        /* ============================================
           Sidebar
           ============================================ */
        [data-testid="stSidebar"] {
            background-color: var(--surface);
            border-right: 1px solid var(--divider);
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: var(--text-primary);
        }

        /* ============================================
           Tabs
           ============================================ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: var(--surface);
            border-radius: 4px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: var(--text-secondary);
            border-radius: 2px;
            padding: 8px 16px;
            font-weight: 500;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: var(--surface-hover);
            color: var(--text-primary);
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--accent-primary);
            color: var(--text-primary);
        }

        /* ============================================
           Expanders
           ============================================ */
        .streamlit-expanderHeader {
            background-color: var(--surface);
            border: 1px solid var(--divider);
            border-radius: 4px;
            color: var(--text-primary);
            font-weight: 500;
        }

        .streamlit-expanderHeader:hover {
            background-color: var(--surface-hover);
        }

        .streamlit-expanderContent {
            background-color: var(--surface);
            border: 1px solid var(--divider);
            border-top: none;
            border-radius: 0 0 4px 4px;
        }

        /* ============================================
           Progress Bars & Sliders
           ============================================ */
        .stProgress > div > div > div {
            background-color: var(--accent-primary);
        }

        .stSlider > div > div > div > div {
            background-color: var(--accent-primary);
        }

        /* ============================================
           Chat Interface
           ============================================ */
        .stChatMessage {
            background-color: var(--surface);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }

        .stChatMessage[data-testid="user-message"] {
            border-left: 3px solid var(--accent-primary);
        }

        .stChatMessage[data-testid="assistant-message"] {
            border-left: 3px solid var(--text-secondary);
        }

        /* ============================================
           Alerts & Notifications
           ============================================ */
        .stSuccess {
            background-color: rgba(40, 167, 69, 0.1);
            border-left: 4px solid var(--success);
            color: var(--text-primary);
        }

        .stWarning {
            background-color: rgba(255, 193, 7, 0.1);
            border-left: 4px solid var(--warning);
            color: var(--text-primary);
        }

        .stError {
            background-color: rgba(220, 53, 69, 0.1);
            border-left: 4px solid var(--danger);
            color: var(--text-primary);
        }

        .stInfo {
            background-color: rgba(31, 78, 140, 0.1);
            border-left: 4px solid var(--accent-primary);
            color: var(--text-primary);
        }

        /* ============================================
           Spinners & Loading States
           ============================================ */
        .stSpinner > div {
            border-top-color: var(--accent-primary);
        }

        /* ============================================
           Tooltips
           ============================================ */
        [data-baseweb="tooltip"] {
            background-color: var(--surface-hover);
            color: var(--text-primary);
            border: 1px solid var(--divider);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
        }

        /* ============================================
           Responsive Design
           ============================================ */
        @media (max-width: 768px) {
            h1 { font-size: 20px; }
            h2 { font-size: 18px; }
            h3 { font-size: 16px; }

            .stButton > button {
                width: 100%;
                padding: 10px 16px;
            }

            .stMetric {
                padding: 12px;
            }
        }

        /* ============================================
           Utility Classes
           ============================================ */
        .nebula-card {
            background-color: var(--surface);
            border-radius: 4px;
            padding: 16px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
        }

        .nebula-divider {
            border-top: 1px solid var(--divider);
            margin: 16px 0;
        }

        .nebula-text-secondary {
            color: var(--text-secondary);
        }

        .nebula-compact {
            padding: 8px;
            margin: 4px 0;
        }
        </style>
    """


def get_theme_config() -> Dict[str, any]:
    """Get Streamlit theme configuration.

    Returns:
        Dictionary with theme configuration for .streamlit/config.toml
    """
    return {
        "primaryColor": "#1F4E8C",
        "backgroundColor": "#121317",
        "secondaryBackgroundColor": "#1E2130",
        "textColor": "#E0E6F0",
        "font": "sans serif",
    }


def apply_nebula_theme() -> None:
    """Apply Nebula UI theme to Streamlit app.

    This function should be called early in the app initialization.
    It injects custom CSS to style the application according to the
    Nebula UI design system.
    """
    import streamlit as st

    st.markdown(get_nebula_css(), unsafe_allow_html=True)
