"""Unit tests for settings utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from f1_predict.web.utils.settings import (
    SettingsManager,
    get_default_settings,
)


@pytest.fixture
def temp_settings_file(monkeypatch):
    """Create temporary settings file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_file = Path(tmpdir) / "settings.json"
        monkeypatch.setattr(
            "f1_predict.web.utils.settings.get_settings_file",
            lambda: settings_file,
        )
        yield settings_file


class TestSettingsManager:
    """Tests for SettingsManager class."""

    def test_default_settings_structure(self):
        """Test default settings have correct structure."""
        defaults = get_default_settings()

        # Check main sections exist
        assert "general" in defaults
        assert "predictions" in defaults
        assert "comparisons" in defaults
        assert "analytics" in defaults
        assert "api" in defaults
        assert "advanced" in defaults

    def test_default_settings_values(self):
        """Test default settings have correct values."""
        defaults = get_default_settings()

        assert defaults["general"]["theme"] == "Nebula Dark"
        assert defaults["general"]["units"] == "metric"
        assert defaults["predictions"]["default_model"] == "Ensemble"
        assert defaults["api"]["rate_limit"] == 4

    def test_settings_manager_initialization(self, temp_settings_file):
        """Test settings manager initializes correctly."""
        manager = SettingsManager()

        assert manager.settings is not None
        assert len(manager.settings) > 0

    def test_get_setting(self, temp_settings_file):
        """Test getting individual settings."""
        manager = SettingsManager()

        value = manager.get("general", "theme")
        assert value == "Nebula Dark"

    def test_get_section(self, temp_settings_file):
        """Test getting entire settings section."""
        manager = SettingsManager()

        section = manager.get_section("general")
        assert "theme" in section
        assert "units" in section
        assert "timezone" in section

    def test_get_all_settings(self, temp_settings_file):
        """Test getting all settings."""
        manager = SettingsManager()

        all_settings = manager.get_all()
        assert isinstance(all_settings, dict)
        assert len(all_settings) > 0

    def test_set_setting(self, temp_settings_file):
        """Test setting individual setting."""
        manager = SettingsManager()

        result = manager.set("general", "theme", "Light")
        assert result is True

        value = manager.get("general", "theme")
        assert value == "Light"

    def test_set_section(self, temp_settings_file):
        """Test setting multiple settings in section."""
        manager = SettingsManager()

        new_values = {"theme": "High Contrast", "units": "imperial"}
        result = manager.set_section("general", new_values)

        assert result is True
        assert manager.get("general", "theme") == "High Contrast"
        assert manager.get("general", "units") == "imperial"

    def test_save_settings(self, temp_settings_file):
        """Test saving settings to file."""
        manager = SettingsManager()

        manager.set("general", "theme", "Light")
        result = manager.save()

        assert result is True
        assert temp_settings_file.exists()

        # Verify file content
        with open(temp_settings_file) as f:
            saved = json.load(f)
            assert saved["general"]["theme"] == "Light"

    def test_load_saved_settings(self, temp_settings_file):
        """Test loading previously saved settings."""
        # Save settings
        manager1 = SettingsManager()
        manager1.set("general", "theme", "Light")
        manager1.save()

        # Load settings in new manager instance
        manager2 = SettingsManager()
        value = manager2.get("general", "theme")

        assert value == "Light"

    def test_reset_to_defaults(self, temp_settings_file):
        """Test resetting settings to defaults."""
        manager = SettingsManager()

        manager.set("general", "theme", "Light")
        result = manager.reset_to_defaults()

        assert result is True
        assert manager.get("general", "theme") == "Nebula Dark"

    def test_export_settings(self, temp_settings_file):
        """Test exporting settings as JSON string."""
        manager = SettingsManager()

        exported = manager.export_settings()
        assert isinstance(exported, str)

        # Verify it's valid JSON
        parsed = json.loads(exported)
        assert "general" in parsed

    def test_import_settings(self, temp_settings_file):
        """Test importing settings from JSON string."""
        manager = SettingsManager()

        new_settings = {
            "general": {"theme": "Light", "units": "imperial"},
            "predictions": {"default_model": "XGBoost"},
        }

        json_str = json.dumps(new_settings)
        result = manager.import_settings(json_str)

        assert result is True
        assert manager.get("general", "theme") == "Light"
        assert manager.get("predictions", "default_model") == "XGBoost"

    def test_merge_settings(self, temp_settings_file):
        """Test merging new settings with defaults."""
        manager = SettingsManager()

        defaults = get_default_settings()
        loaded = {"general": {"theme": "Light"}}

        merged = manager._merge_settings(defaults, loaded)

        assert merged["general"]["theme"] == "Light"
        assert merged["general"]["units"] == "metric"  # From defaults


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_validate_confidence_threshold(self, temp_settings_file):
        """Test confidence threshold validation."""
        manager = SettingsManager()

        # Valid values
        assert manager.validate_setting("predictions", "confidence_threshold", 0.5)[0]
        assert manager.validate_setting("predictions", "confidence_threshold", 0.0)[0]
        assert manager.validate_setting("predictions", "confidence_threshold", 1.0)[0]

        # Invalid values
        assert not manager.validate_setting(
            "predictions", "confidence_threshold", -0.1
        )[0]
        assert not manager.validate_setting(
            "predictions", "confidence_threshold", 1.1
        )[0]

    def test_validate_rate_limit(self, temp_settings_file):
        """Test rate limit validation."""
        manager = SettingsManager()

        # Valid values
        assert manager.validate_setting("api", "rate_limit", 1)[0]
        assert manager.validate_setting("api", "rate_limit", 10)[0]

        # Invalid values
        assert not manager.validate_setting("api", "rate_limit", 0)[0]
        assert not manager.validate_setting("api", "rate_limit", 11)[0]

    def test_validate_timeout(self, temp_settings_file):
        """Test request timeout validation."""
        manager = SettingsManager()

        # Valid values
        assert manager.validate_setting("api", "request_timeout", 30)[0]

        # Invalid values
        assert not manager.validate_setting("api", "request_timeout", 3)[0]

    def test_validate_theme(self, temp_settings_file):
        """Test theme validation."""
        manager = SettingsManager()

        # Valid values
        assert manager.validate_setting("general", "theme", "Nebula Dark")[0]
        assert manager.validate_setting("general", "theme", "Light")[0]

        # Invalid values
        assert not manager.validate_setting("general", "theme", "InvalidTheme")[0]

    def test_validate_units(self, temp_settings_file):
        """Test units validation."""
        manager = SettingsManager()

        # Valid values
        assert manager.validate_setting("general", "units", "metric")[0]
        assert manager.validate_setting("general", "units", "imperial")[0]

        # Invalid values
        assert not manager.validate_setting("general", "units", "celsius")[0]

    def test_validate_model(self, temp_settings_file):
        """Test model validation."""
        manager = SettingsManager()

        # Valid values
        for model in ["Ensemble", "XGBoost", "LightGBM", "Random Forest"]:
            assert manager.validate_setting(
                "predictions", "default_model", model
            )[0]

        # Invalid values
        assert not manager.validate_setting(
            "predictions", "default_model", "InvalidModel"
        )[0]

    def test_validation_error_messages(self, temp_settings_file):
        """Test that validation returns meaningful error messages."""
        manager = SettingsManager()

        is_valid, msg = manager.validate_setting(
            "api", "rate_limit", 0
        )

        assert not is_valid
        assert len(msg) > 0
        assert "must be" in msg.lower() or "invalid" in msg.lower()


class TestSettingsPersistence:
    """Tests for settings persistence across sessions."""

    def test_settings_persist_across_instances(self, temp_settings_file):
        """Test that settings persist across different manager instances."""
        # Create and modify settings
        manager1 = SettingsManager()
        manager1.set("general", "theme", "Light")
        manager1.set("predictions", "confidence_threshold", 0.85)
        manager1.save()

        # Create new manager instance
        manager2 = SettingsManager()

        # Verify persisted values
        assert manager2.get("general", "theme") == "Light"
        assert manager2.get("predictions", "confidence_threshold") == 0.85

    def test_settings_file_created_on_demand(self, temp_settings_file):
        """Test that settings file is created when needed."""
        assert not temp_settings_file.exists()

        manager = SettingsManager()
        manager.set("general", "theme", "Light")
        manager.save()

        assert temp_settings_file.exists()

    def test_settings_directory_created_on_demand(self, monkeypatch):
        """Test that settings directory is created when needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "subdir" / "settings.json"
            monkeypatch.setattr(
                "f1_predict.web.utils.settings.get_settings_file",
                lambda: settings_file,
            )

            assert not settings_file.parent.exists()

            manager = SettingsManager()
            manager.save()

            assert settings_file.parent.exists()


class TestSettingsImportExport:
    """Tests for settings import/export functionality."""

    def test_export_then_import(self, temp_settings_file):
        """Test exporting and importing settings."""
        # Set up initial settings
        manager1 = SettingsManager()
        manager1.set("general", "theme", "Light")
        manager1.set("predictions", "confidence_threshold", 0.85)

        # Export settings
        exported = manager1.export_settings()

        # Import into new manager
        manager2 = SettingsManager()
        manager2.reset_to_defaults()
        result = manager2.import_settings(exported)

        assert result is True
        assert manager2.get("general", "theme") == "Light"
        assert manager2.get("predictions", "confidence_threshold") == 0.85

    def test_import_invalid_json(self, temp_settings_file):
        """Test importing invalid JSON."""
        manager = SettingsManager()

        result = manager.import_settings("invalid json")
        assert result is False

    def test_import_partial_settings(self, temp_settings_file):
        """Test importing partial settings (should merge with defaults)."""
        manager = SettingsManager()

        partial_settings = {"general": {"theme": "Light"}}
        json_str = json.dumps(partial_settings)

        result = manager.import_settings(json_str)

        assert result is True
        assert manager.get("general", "theme") == "Light"
        # Should still have defaults for missing keys
        assert manager.get("general", "units") is not None
