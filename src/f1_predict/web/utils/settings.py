"""Settings management utilities for F1 Race Predictor.

Provides persistent configuration management for:
- General preferences (theme, timezone, units)
- Prediction settings (model, confidence, explanations)
- Comparison settings (filters, visualization)
- Analytics settings (dashboard config)
- API configuration (endpoints, rate limits)
- Advanced settings (performance, debug)
"""

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def get_settings_dir() -> Path:
    """Get settings directory path."""
    return Path.home() / ".f1_predict"


def get_settings_file() -> Path:
    """Get settings file path."""
    return get_settings_dir() / "settings.json"


def get_default_settings() -> dict[str, dict[str, Any]]:
    """Get default settings configuration.

    Returns:
        Dictionary with all default settings
    """
    return {
        # General Settings
        "general": {
            "theme": "Nebula Dark",
            "timezone": "UTC",
            "units": "metric",
            "language": "en",
            "date_format": "ISO",
        },
        # Prediction Settings
        "predictions": {
            "default_model": "Ensemble",
            "confidence_threshold": 0.7,
            "show_feature_importance": True,
            "always_show_explanations": True,
            "predictions_to_cache": 50,
            "show_podium_only": False,
            "confidence_chart_type": "bar",
            "export_format_default": "csv",
        },
        # Comparison Settings
        "comparisons": {
            "default_season": 2024,
            "include_sprint_races": True,
            "default_chart_type": "line",
            "races_to_display": 10,
            "color_scheme": "team",
            "enable_animation": True,
            "show_trend_lines": True,
        },
        # Analytics Settings
        "analytics": {
            "default_time_period": "Current Season",
            "auto_refresh_enabled": False,
            "refresh_interval": 300,
            "standings_page_size": 10,
            "show_historical_data": True,
            "include_retired_drivers": False,
            "kpis_to_display": ["races", "accuracy", "confidence", "quality"],
        },
        # API Configuration
        "api": {
            "ergast_endpoint": "https://ergast.com/api/f1",
            "rate_limit": 4,
            "request_timeout": 30,
            "retry_attempts": 3,
            "enable_caching": True,
            "cache_duration": 60,
            "auto_update_data": False,
            "update_frequency": "weekly",
            "data_quality_threshold": 85,
            "enable_validation": True,
        },
        # Advanced Settings
        "advanced": {
            "enable_aggressive_caching": False,
            "preload_common_queries": True,
            "chart_rendering_quality": "medium",
            "max_concurrent_calls": 5,
            "enable_debug_logging": False,
            "show_performance_metrics": False,
            "developer_mode": False,
        },
    }


class SettingsManager:
    """Manages application settings with persistence."""

    def __init__(self) -> None:
        """Initialize settings manager."""
        self.logger = logger.bind(component="settings_manager")
        self.settings_file = get_settings_file()
        self.settings: dict[str, Any] = {}
        self._load_settings()

    def _load_settings(self) -> None:
        """Load settings from file or use defaults."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file) as f:
                    loaded = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    defaults = get_default_settings()
                    self.settings = self._merge_settings(defaults, loaded)
                    self.logger.info("Settings loaded from file")
            else:
                self.settings = get_default_settings()
                self.logger.info("Using default settings")
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            self.settings = get_default_settings()

    def _merge_settings(self, defaults: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
        """Merge loaded settings with defaults.

        Args:
            defaults: Default settings
            loaded: Loaded settings

        Returns:
            Merged settings (loaded overrides defaults)
        """
        merged = defaults.copy()
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    def get(self, section: str, key: str) -> Any:
        """Get a setting value.

        Args:
            section: Settings section (general, predictions, etc.)
            key: Setting key

        Returns:
            Setting value or None if not found
        """
        try:
            return self.settings.get(section, {}).get(key)
        except Exception as e:
            self.logger.error(f"Error getting setting {section}.{key}: {e}")
            return None

    def get_section(self, section: str) -> dict[str, Any]:
        """Get all settings in a section.

        Args:
            section: Settings section

        Returns:
            Dictionary of section settings
        """
        return self.settings.get(section, {}) or {}

    def get_all(self) -> dict[str, Any]:
        """Get all settings.

        Returns:
            Complete settings dictionary
        """
        return self.settings.copy()

    def set(self, section: str, key: str, value: Any) -> bool:
        """Set a setting value.

        Args:
            section: Settings section
            key: Setting key
            value: New value

        Returns:
            True if successful, False otherwise
        """
        try:
            if section not in self.settings:
                self.settings[section] = {}
            self.settings[section][key] = value
            return True
        except Exception as e:
            self.logger.error(f"Error setting {section}.{key}: {e}")
            return False

    def set_section(self, section: str, values: dict[str, Any]) -> bool:
        """Set multiple settings in a section.

        Args:
            section: Settings section
            values: Dictionary of key-value pairs

        Returns:
            True if successful, False otherwise
        """
        try:
            if section not in self.settings:
                self.settings[section] = {}
            self.settings[section].update(values)
            return True
        except Exception as e:
            self.logger.error(f"Error setting section {section}: {e}")
            return False

    def save(self) -> bool:
        """Save settings to file.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)

            # Write settings to file
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=2)

            self.logger.info("Settings saved to file")
            return True
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            return False

    def reset_to_defaults(self) -> bool:
        """Reset all settings to defaults.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.settings = get_default_settings()
            return self.save()
        except Exception as e:
            self.logger.error(f"Error resetting settings: {e}")
            return False

    def export_settings(self) -> str:
        """Export settings as JSON string.

        Returns:
            JSON string representation of settings
        """
        try:
            return json.dumps(self.settings, indent=2)
        except Exception as e:
            self.logger.error(f"Error exporting settings: {e}")
            return "{}"

    def import_settings(self, json_str: str) -> bool:
        """Import settings from JSON string.

        Args:
            json_str: JSON string with settings

        Returns:
            True if successful, False otherwise
        """
        try:
            imported = json.loads(json_str)
            defaults = get_default_settings()
            self.settings = self._merge_settings(defaults, imported)
            return self.save()
        except Exception as e:
            self.logger.error(f"Error importing settings: {e}")
            return False

    def validate_setting(  # noqa: C901, PLR0911, PLR0912
        self, section: str, key: str, value: Any
    ) -> tuple[bool, str]:
        """Validate a setting value.

        Args:
            section: Settings section
            key: Setting key
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Numeric validations
            if section == "predictions" and key == "confidence_threshold":
                if not 0.0 <= value <= 1.0:
                    return False, "Confidence threshold must be between 0.0 and 1.0"

            if section == "api" and key == "rate_limit":
                if not 1 <= value <= 10:
                    return False, "Rate limit must be between 1 and 10"

            if section == "api" and key == "request_timeout" and value < 5:
                return False, "Request timeout must be at least 5 seconds"

            if section == "api" and key == "retry_attempts" and value < 0:
                return False, "Retry attempts must be non-negative"

            if section == "api" and key == "cache_duration" and value < 5:
                return False, "Cache duration must be at least 5 minutes"

            if section == "analytics" and key == "refresh_interval":
                if value < 60:
                    return False, "Refresh interval must be at least 60 seconds"

            if section == "analytics" and key == "standings_page_size":
                if value < 1:
                    return False, "Page size must be at least 1"

            if section == "analytics" and key == "data_quality_threshold":
                if not 0 <= value <= 100:
                    return False, "Data quality threshold must be between 0 and 100"

            # Enum validations
            if section == "general" and key == "theme":
                valid_themes = ["Nebula Dark", "Light", "High Contrast"]
                if value not in valid_themes:
                    return False, f"Theme must be one of {valid_themes}"

            if section == "general" and key == "units":
                if value not in ["metric", "imperial"]:
                    return False, "Units must be 'metric' or 'imperial'"

            if section == "general" and key == "date_format":
                valid_formats = ["ISO", "US", "EU"]
                if value not in valid_formats:
                    return False, f"Date format must be one of {valid_formats}"

            if section == "predictions" and key == "default_model":
                valid_models = ["Ensemble", "XGBoost", "LightGBM", "Random Forest"]
                if value not in valid_models:
                    return False, f"Model must be one of {valid_models}"

            if section == "predictions" and key == "confidence_chart_type":
                valid_types = ["bar", "line", "scatter"]
                if value not in valid_types:
                    return False, f"Chart type must be one of {valid_types}"

            if section == "predictions" and key == "export_format_default":
                valid_formats = ["csv", "json", "both"]
                if value not in valid_formats:
                    return False, f"Export format must be one of {valid_formats}"

            if section == "analytics" and key == "default_time_period":
                valid_periods = ["Last 5 Races", "Current Season", "Last 2 Seasons", "All Time"]
                if value not in valid_periods:
                    return False, f"Time period must be one of {valid_periods}"

            if section == "api" and key == "update_frequency":
                valid_freq = ["daily", "weekly", "manual"]
                if value not in valid_freq:
                    return False, f"Update frequency must be one of {valid_freq}"

            if section == "advanced" and key == "chart_rendering_quality":
                valid_quality = ["low", "medium", "high"]
                if value not in valid_quality:
                    return False, f"Quality must be one of {valid_quality}"

            if section == "comparisons" and key == "default_season":
                if not 2020 <= value <= 2050:
                    return False, "Default season must be between 2020 and 2050"

            if section == "comparisons" and key == "races_to_display":
                if value < 1 or value > 100:
                    return False, "Races to display must be between 1 and 100"

            if section == "comparisons" and key == "default_chart_type":
                valid_types = ["line", "bar", "scatter"]
                if value not in valid_types:
                    return False, f"Chart type must be one of {valid_types}"

            if section == "comparisons" and key == "color_scheme":
                valid_schemes = ["team", "driver", "gradient"]
                if value not in valid_schemes:
                    return False, f"Color scheme must be one of {valid_schemes}"

            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"
