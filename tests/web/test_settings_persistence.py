"""Tests for settings persistence functionality."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from f1_predict.web.utils.settings import (
    SettingsManager,
    get_default_settings,
    validate_settings,
    export_settings_json,
    import_settings_json,
    save_settings,
    load_settings,
    reset_to_defaults,
    ensure_settings_dir,
)


class TestDefaultSettings:
    """Test default settings generation."""

    def test_get_default_settings_returns_dict(self):
        """Test that default settings returns a dictionary."""
        defaults = get_default_settings()
        assert isinstance(defaults, dict)
        assert len(defaults) > 0

    def test_default_settings_contain_required_keys(self):
        """Test that default settings contain all required keys."""
        defaults = get_default_settings()

        required_keys = [
            "theme",
            "timezone",
            "units",
            "default_model",
            "confidence_threshold",
            "enable_explanations",
            "api_endpoint",
            "rate_limit",
            "cache_duration",
        ]

        for key in required_keys:
            assert key in defaults, f"Missing required key: {key}"

    def test_default_theme_is_nebula_dark(self):
        """Test that default theme is Nebula Dark."""
        defaults = get_default_settings()
        assert defaults["theme"] == "Nebula Dark"

    def test_default_confidence_threshold_is_valid(self):
        """Test that default confidence threshold is valid."""
        defaults = get_default_settings()
        threshold = defaults["confidence_threshold"]
        assert 0.0 <= threshold <= 1.0
        assert threshold == 0.7


class TestValidateSettings:
    """Test settings validation."""

    def test_validate_valid_settings(self):
        """Test validation of valid settings."""
        settings = get_default_settings()
        is_valid, error = validate_settings(settings)
        assert is_valid
        assert error is None

    def test_validate_invalid_rate_limit_too_low(self):
        """Test validation rejects rate limit < 1."""
        settings = {"rate_limit": 0}
        is_valid, error = validate_settings(settings)
        assert not is_valid
        assert error is not None
        assert "rate limit" in error.lower()

    def test_validate_invalid_rate_limit_too_high(self):
        """Test validation rejects rate limit > 10."""
        settings = {"rate_limit": 11}
        is_valid, error = validate_settings(settings)
        assert not is_valid
        assert error is not None

    def test_validate_invalid_confidence_threshold(self):
        """Test validation rejects confidence threshold outside 0-1."""
        settings = {"confidence_threshold": 1.5}
        is_valid, error = validate_settings(settings)
        assert not is_valid

    def test_validate_invalid_theme(self):
        """Test validation rejects invalid theme."""
        settings = {"theme": "InvalidTheme"}
        is_valid, error = validate_settings(settings)
        assert not is_valid
        assert "theme" in error.lower()

    def test_validate_valid_request_timeout(self):
        """Test validation accepts valid request timeout."""
        settings = {"request_timeout": 30}
        is_valid, error = validate_settings(settings)
        assert is_valid

    def test_validate_invalid_request_timeout_too_high(self):
        """Test validation rejects request timeout > 300."""
        settings = {"request_timeout": 301}
        is_valid, error = validate_settings(settings)
        assert not is_valid


class TestExportImportSettings:
    """Test settings export and import."""

    def test_export_settings_json_valid(self):
        """Test exporting settings to JSON."""
        settings = get_default_settings()
        json_str = export_settings_json(settings)

        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_import_settings_json_valid(self):
        """Test importing valid settings JSON."""
        settings = get_default_settings()
        json_str = export_settings_json(settings)

        success, imported, error = import_settings_json(json_str)

        assert success
        assert error is None
        assert imported is not None
        assert isinstance(imported, dict)

    def test_import_settings_json_invalid(self):
        """Test importing invalid JSON."""
        success, imported, error = import_settings_json("invalid json")

        assert not success
        assert imported is None
        assert error is not None

    def test_import_settings_json_invalid_dict(self):
        """Test importing JSON that's not a dict."""
        success, imported, error = import_settings_json('["list", "not", "dict"]')

        assert not success
        assert error is not None

    def test_import_settings_json_invalid_values(self):
        """Test importing JSON with invalid values."""
        invalid_settings = json.dumps({"rate_limit": 100})
        success, imported, error = import_settings_json(invalid_settings)

        assert not success
        assert error is not None


class TestSettingsManager:
    """Test SettingsManager class."""

    def test_settings_manager_initialization(self):
        """Test manager initializes correctly."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            mock_load.return_value = get_default_settings()

            manager = SettingsManager()

            assert manager.settings is not None
            assert isinstance(manager.settings, dict)
            assert manager.is_changed is False

    def test_settings_manager_get(self):
        """Test getting a setting value."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            defaults = get_default_settings()
            mock_load.return_value = defaults

            manager = SettingsManager()

            theme = manager.get("theme")
            assert theme == "Nebula Dark"

    def test_settings_manager_get_with_default(self):
        """Test getting a setting with default value."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            mock_load.return_value = {}

            manager = SettingsManager()

            value = manager.get("nonexistent", "default_value")
            assert value == "default_value"

    def test_settings_manager_set(self):
        """Test setting a value."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            mock_load.return_value = get_default_settings()

            manager = SettingsManager()
            assert manager.is_changed is False

            manager.set("theme", "Light")

            assert manager.get("theme") == "Light"
            assert manager.is_changed is True

    def test_settings_manager_update(self):
        """Test updating multiple values."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            mock_load.return_value = get_default_settings()

            manager = SettingsManager()

            updates = {"theme": "Light", "timezone": "US/Eastern"}
            manager.update(updates)

            assert manager.get("theme") == "Light"
            assert manager.get("timezone") == "US/Eastern"
            assert manager.is_changed is True

    def test_settings_manager_save_valid(self):
        """Test saving valid settings."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            with patch("f1_predict.web.utils.settings.save_settings") as mock_save:
                mock_load.return_value = get_default_settings()
                mock_save.return_value = True

                manager = SettingsManager()
                manager.set("theme", "Light")

                success, error = manager.save()

                assert success
                assert error is None
                assert manager.is_changed is False
                mock_save.assert_called_once()

    def test_settings_manager_save_invalid(self):
        """Test saving invalid settings fails."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            mock_load.return_value = get_default_settings()

            manager = SettingsManager()
            manager.settings["rate_limit"] = 100  # Invalid

            success, error = manager.save()

            assert not success
            assert error is not None

    def test_settings_manager_reset(self):
        """Test resetting to defaults."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            with patch("f1_predict.web.utils.settings.reset_to_defaults") as mock_reset:
                mock_load.return_value = get_default_settings()
                mock_reset.return_value = True

                manager = SettingsManager()
                manager.set("theme", "Light")

                success = manager.reset()

                assert success
                assert manager.get("theme") == "Nebula Dark"

    def test_settings_manager_export(self):
        """Test exporting settings."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            defaults = get_default_settings()
            mock_load.return_value = defaults

            manager = SettingsManager()
            json_str = manager.export()

            assert isinstance(json_str, str)
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)

    def test_settings_manager_import_valid(self):
        """Test importing valid settings."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            defaults = get_default_settings()
            mock_load.return_value = defaults

            manager = SettingsManager()
            json_str = manager.export()

            manager2 = SettingsManager()
            success, error = manager2.import_json(json_str)

            assert success
            assert error is None
            assert manager2.is_changed is True

    def test_settings_manager_import_invalid(self):
        """Test importing invalid settings."""
        with patch("f1_predict.web.utils.settings.load_settings") as mock_load:
            mock_load.return_value = get_default_settings()

            manager = SettingsManager()
            success, error = manager.import_json("invalid json")

            assert not success
            assert error is not None


class TestSettingsPersistence:
    """Test actual file-based persistence."""

    def test_ensure_settings_dir_creates_directory(self):
        """Test that ensure_settings_dir creates the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("f1_predict.web.utils.settings.SETTINGS_DIR", Path(tmpdir) / ".f1_predict"):
                settings_dir = ensure_settings_dir()

                assert settings_dir.exists()
                assert settings_dir.is_dir()

    def test_save_and_load_settings(self):
        """Test saving and loading settings from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("f1_predict.web.utils.settings.SETTINGS_FILE", Path(tmpdir) / "settings.json"):
                settings = get_default_settings()
                settings["theme"] = "Light"

                with patch("f1_predict.web.utils.settings.SETTINGS_DIR", Path(tmpdir)):
                    success = save_settings(settings)
                    assert success

                    loaded = load_settings()
                    assert loaded["theme"] == "Light"

    def test_load_settings_returns_defaults_if_file_missing(self):
        """Test that load_settings returns defaults if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("f1_predict.web.utils.settings.SETTINGS_FILE", Path(tmpdir) / "nonexistent.json"):
                settings = load_settings()

                assert isinstance(settings, dict)
                assert settings["theme"] == "Nebula Dark"

    def test_reset_to_defaults_removes_file(self):
        """Test that reset_to_defaults removes the settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"

            with patch("f1_predict.web.utils.settings.SETTINGS_FILE", settings_file):
                with patch("f1_predict.web.utils.settings.SETTINGS_DIR", Path(tmpdir)):
                    # Create a file
                    save_settings({"theme": "Light"})
                    assert settings_file.exists()

                    # Reset
                    reset_to_defaults()
                    assert not settings_file.exists()
