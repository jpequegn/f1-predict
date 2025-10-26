"""Settings page for F1 Race Predictor web app."""

import logging

import streamlit as st

from f1_predict.web.utils.settings import SettingsManager

logger = logging.getLogger(__name__)


@st.cache_resource  # type: ignore[misc]
def get_settings_manager() -> SettingsManager:
    """Get cached settings manager instance."""
    return SettingsManager()


def _display_general_settings(settings_mgr: SettingsManager) -> None:
    """Display general settings section."""
    st.subheader("⚙️ General Settings")

    col1, col2 = st.columns(2)

    with col1:
        theme = st.selectbox(
            "Color Theme",
            options=["Nebula Dark", "Light", "High Contrast"],
            index=["Nebula Dark", "Light", "High Contrast"].index(
                settings_mgr.get("general", "theme")
            ),
            help="Select your preferred color theme",
        )
        settings_mgr.set("general", "theme", theme)

        units = st.radio(
            "Units",
            options=["Metric (km/h, °C)", "Imperial (mph, °F)"],
            index=0
            if settings_mgr.get("general", "units") == "metric"
            else 1,
            horizontal=True,
            help="Choose your preferred unit system",
        )
        units_value = "metric" if "Metric" in units else "imperial"
        settings_mgr.set("general", "units", units_value)

    with col2:
        timezone = st.selectbox(
            "Timezone",
            options=[
                "UTC",
                "EST",
                "CST",
                "MST",
                "PST",
                "CET",
                "IST",
                "JST",
                "AEST",
            ],
            index=["UTC", "EST", "CST", "MST", "PST", "CET", "IST", "JST", "AEST"].index(
                settings_mgr.get("general", "timezone")
            ),
            help="Select your timezone",
        )
        settings_mgr.set("general", "timezone", timezone)

        date_format = st.selectbox(
            "Date Format",
            options=["ISO", "US", "EU"],
            index=["ISO", "US", "EU"].index(
                settings_mgr.get("general", "date_format")
            ),
            help="Choose date format (ISO: 2024-10-19, US: 10/19/2024, EU: 19/10/2024)",
        )
        settings_mgr.set("general", "date_format", date_format)


def _display_prediction_settings(settings_mgr: SettingsManager) -> None:
    """Display prediction settings section."""
    st.subheader("🎯 Prediction Settings")

    col1, col2 = st.columns(2)

    with col1:
        model = st.selectbox(
            "Default Prediction Model",
            options=["Ensemble", "XGBoost", "LightGBM", "Random Forest"],
            index=["Ensemble", "XGBoost", "LightGBM", "Random Forest"].index(
                settings_mgr.get("predictions", "default_model")
            ),
            help="Default ML model for predictions",
        )
        settings_mgr.set("predictions", "default_model", model)

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=settings_mgr.get("predictions", "confidence_threshold"),
            step=0.05,
            help="Minimum confidence for displaying predictions",
        )
        settings_mgr.set("predictions", "confidence_threshold", confidence_threshold)

    with col2:
        show_importance = st.checkbox(
            "Show Feature Importance by Default",
            value=settings_mgr.get("predictions", "show_feature_importance"),
        )
        settings_mgr.set("predictions", "show_feature_importance", show_importance)

        show_explanations = st.checkbox(
            "Always Show Explanations",
            value=settings_mgr.get("predictions", "always_show_explanations"),
            help="Display SHAP explanations by default",
        )
        settings_mgr.set("predictions", "always_show_explanations", show_explanations)

        export_format = st.selectbox(
            "Default Export Format",
            options=["csv", "json", "both"],
            index=["csv", "json", "both"].index(
                settings_mgr.get("predictions", "export_format_default")
            ),
        )
        settings_mgr.set("predictions", "export_format_default", export_format)


def _display_comparison_settings(settings_mgr: SettingsManager) -> None:
    """Display comparison settings section."""
    st.subheader("🏁 Comparison Settings")

    col1, col2 = st.columns(2)

    with col1:
        season = st.number_input(
            "Default Season",
            min_value=2020,
            max_value=2050,
            value=settings_mgr.get("comparisons", "default_season"),
            help="Default season for comparisons",
        )
        is_valid, msg = settings_mgr.validate_setting("comparisons", "default_season", season)
        if not is_valid:
            st.error(msg)
        else:
            settings_mgr.set("comparisons", "default_season", season)

        races_to_display = st.number_input(
            "Races to Display",
            min_value=1,
            max_value=100,
            value=settings_mgr.get("comparisons", "races_to_display"),
            help="Number of races to display in comparisons",
        )
        settings_mgr.set("comparisons", "races_to_display", races_to_display)

    with col2:
        chart_type = st.selectbox(
            "Default Chart Type",
            options=["line", "bar", "scatter"],
            index=["line", "bar", "scatter"].index(
                settings_mgr.get("comparisons", "default_chart_type")
            ),
            help="Default visualization type for comparisons",
        )
        settings_mgr.set("comparisons", "default_chart_type", chart_type)

        color_scheme = st.selectbox(
            "Color Scheme",
            options=["team", "driver", "gradient"],
            index=["team", "driver", "gradient"].index(
                settings_mgr.get("comparisons", "color_scheme")
            ),
            help="Color scheme for comparison charts",
        )
        settings_mgr.set("comparisons", "color_scheme", color_scheme)

    st.markdown("**Display Options**")

    col1, col2 = st.columns(2)

    with col1:
        include_sprints = st.checkbox(
            "Include Sprint Races",
            value=settings_mgr.get("comparisons", "include_sprint_races"),
            help="Include sprint races in comparisons",
        )
        settings_mgr.set("comparisons", "include_sprint_races", include_sprints)

    with col2:
        enable_animation = st.checkbox(
            "Enable Chart Animation",
            value=settings_mgr.get("comparisons", "enable_animation"),
            help="Animate chart transitions",
        )
        settings_mgr.set("comparisons", "enable_animation", enable_animation)

    col1, col2 = st.columns(2)

    with col1:
        show_trend_lines = st.checkbox(
            "Show Trend Lines",
            value=settings_mgr.get("comparisons", "show_trend_lines"),
            help="Display trend lines in charts",
        )
        settings_mgr.set("comparisons", "show_trend_lines", show_trend_lines)


def _display_analytics_settings(settings_mgr: SettingsManager) -> None:
    """Display analytics settings section."""
    st.subheader("📊 Analytics Settings")

    col1, col2 = st.columns(2)

    with col1:
        time_period = st.selectbox(
            "Default Time Period",
            options=["Last 5 Races", "Current Season", "Last 2 Seasons", "All Time"],
            index=[
                "Last 5 Races",
                "Current Season",
                "Last 2 Seasons",
                "All Time",
            ].index(settings_mgr.get("analytics", "default_time_period")),
        )
        settings_mgr.set("analytics", "default_time_period", time_period)

        standings_size = st.number_input(
            "Standings Page Size",
            min_value=5,
            max_value=50,
            value=settings_mgr.get("analytics", "standings_page_size"),
            step=5,
            help="Number of entries per standings page",
        )
        settings_mgr.set("analytics", "standings_page_size", standings_size)

    with col2:
        auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=settings_mgr.get("analytics", "auto_refresh_enabled"),
        )
        settings_mgr.set("analytics", "auto_refresh_enabled", auto_refresh)

        if auto_refresh:
            refresh_interval = st.number_input(
                "Refresh Interval (seconds)",
                min_value=60,
                max_value=3600,
                value=settings_mgr.get("analytics", "refresh_interval"),
                step=60,
            )
            settings_mgr.set("analytics", "refresh_interval", refresh_interval)

        show_historical = st.checkbox(
            "Show Historical Data",
            value=settings_mgr.get("analytics", "show_historical_data"),
        )
        settings_mgr.set("analytics", "show_historical_data", show_historical)


def _display_api_settings(settings_mgr: SettingsManager) -> None:
    """Display API configuration settings."""
    st.subheader("🔌 API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        endpoint = st.text_input(
            "Ergast API Endpoint",
            value=settings_mgr.get("api", "ergast_endpoint"),
            help="F1 API endpoint URL",
        )
        is_valid, msg = settings_mgr.validate_setting("api", "ergast_endpoint", endpoint)
        if endpoint:
            settings_mgr.set("api", "ergast_endpoint", endpoint)

        rate_limit = st.number_input(
            "Rate Limit (requests/second)",
            min_value=1,
            max_value=10,
            value=settings_mgr.get("api", "rate_limit"),
            help="API request rate limit",
        )
        is_valid, msg = settings_mgr.validate_setting("api", "rate_limit", rate_limit)
        if not is_valid:
            st.error(msg)
        else:
            settings_mgr.set("api", "rate_limit", rate_limit)

    with col2:
        timeout = st.number_input(
            "Request Timeout (seconds)",
            min_value=5,
            max_value=120,
            value=settings_mgr.get("api", "request_timeout"),
            help="API request timeout in seconds",
        )
        is_valid, msg = settings_mgr.validate_setting(
            "api", "request_timeout", timeout
        )
        if not is_valid:
            st.error(msg)
        else:
            settings_mgr.set("api", "request_timeout", timeout)

        retries = st.number_input(
            "Retry Attempts",
            min_value=0,
            max_value=5,
            value=settings_mgr.get("api", "retry_attempts"),
            help="Number of retry attempts on failure",
        )
        is_valid, msg = settings_mgr.validate_setting(
            "api", "retry_attempts", retries
        )
        if not is_valid:
            st.error(msg)
        else:
            settings_mgr.set("api", "retry_attempts", retries)

    st.markdown("**Caching**")

    col1, col2 = st.columns(2)

    with col1:
        enable_cache = st.checkbox(
            "Enable API Caching",
            value=settings_mgr.get("api", "enable_caching"),
        )
        settings_mgr.set("api", "enable_caching", enable_cache)

    with col2:
        if enable_cache:
            cache_duration = st.number_input(
                "Cache Duration (minutes)",
                min_value=5,
                max_value=1440,
                value=settings_mgr.get("api", "cache_duration"),
            )
            is_valid, msg = settings_mgr.validate_setting(
                "api", "cache_duration", cache_duration
            )
            if not is_valid:
                st.error(msg)
            else:
                settings_mgr.set("api", "cache_duration", cache_duration)


def _display_advanced_settings(settings_mgr: SettingsManager) -> None:
    """Display advanced settings section."""
    st.subheader("🔧 Advanced Settings")

    col1, col2 = st.columns(2)

    with col1:
        debug_logging = st.checkbox(
            "Enable Debug Logging",
            value=settings_mgr.get("advanced", "enable_debug_logging"),
            help="Enable verbose logging for debugging",
        )
        settings_mgr.set("advanced", "enable_debug_logging", debug_logging)

        show_metrics = st.checkbox(
            "Show Performance Metrics",
            value=settings_mgr.get("advanced", "show_performance_metrics"),
            help="Display performance timing information",
        )
        settings_mgr.set("advanced", "show_performance_metrics", show_metrics)

    with col2:
        aggressive_cache = st.checkbox(
            "Enable Aggressive Caching",
            value=settings_mgr.get("advanced", "enable_aggressive_caching"),
            help="Cache more aggressively for speed",
        )
        settings_mgr.set("advanced", "enable_aggressive_caching", aggressive_cache)

        quality = st.selectbox(
            "Chart Rendering Quality",
            options=["low", "medium", "high"],
            index=["low", "medium", "high"].index(
                settings_mgr.get("advanced", "chart_rendering_quality")
            ),
            help="Higher quality = slower rendering",
        )
        settings_mgr.set("advanced", "chart_rendering_quality", quality)


def _display_data_management(settings_mgr: SettingsManager) -> None:
    """Display data management options."""
    st.subheader("💾 Data Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache cleared!")

    with col2:
        if st.button("📥 Export Settings", use_container_width=True):
            settings_json = settings_mgr.export_settings()
            st.download_button(
                label="Download settings.json",
                data=settings_json,
                file_name="f1_predict_settings.json",
                mime="application/json",
                use_container_width=True,
            )

    with col3:
        uploaded_file = st.file_uploader(
            "📤 Import Settings",
            type=["json"],
            accept_single_file=True,
        )
        if uploaded_file:
            try:
                settings_content = uploaded_file.read().decode("utf-8")
                if settings_mgr.import_settings(settings_content):
                    st.success("✅ Settings imported successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to import settings")
            except Exception as e:
                st.error(f"❌ Error importing settings: {e}")


def show_settings_page() -> None:
    """Display the application settings and preferences."""
    st.title("⚙️ Settings")
    st.markdown(
        "Configure application preferences, defaults, and advanced options."
    )

    # Get settings manager
    settings_mgr = get_settings_manager()

    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "General",
        "Predictions",
        "Comparisons",
        "Analytics",
        "API",
        "Advanced",
    ])

    with tab1:
        _display_general_settings(settings_mgr)

    with tab2:
        _display_prediction_settings(settings_mgr)

    with tab3:
        _display_comparison_settings(settings_mgr)

    with tab4:
        _display_analytics_settings(settings_mgr)

    with tab5:
        _display_api_settings(settings_mgr)

    with tab6:
        _display_advanced_settings(settings_mgr)
        st.divider()
        _display_data_management(settings_mgr)

    # Save buttons at the bottom
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            if settings_mgr.save():
                st.success("✅ Settings saved successfully!")
                # Store in session state for other pages
                st.session_state.settings = settings_mgr.get_all()
            else:
                st.error("❌ Failed to save settings")

    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            if "confirm_reset" not in st.session_state:
                st.session_state.confirm_reset = False

            if st.session_state.confirm_reset:
                if settings_mgr.reset_to_defaults():
                    st.session_state.confirm_reset = False
                    st.success("✅ Settings reset to defaults!")
                    st.rerun()
                else:
                    st.error("❌ Failed to reset settings")
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")
