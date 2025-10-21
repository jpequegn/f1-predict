"""Tests for alert channels (email and Slack)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from f1_predict.web.utils.alert_channels import (
    EmailAlertChannel,
    SlackAlertChannel,
)
from f1_predict.web.utils.alert_config import (
    AlertChannelConfig,
    EmailConfig,
    SlackConfig,
    load_alert_config_from_dict,
    load_alert_config_from_env,
    validate_alert_config,
)


class TestEmailAlertChannel:
    """Test email alert channel."""

    @pytest.fixture
    def email_channel(self):
        """Create email channel instance."""
        return EmailAlertChannel(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="test@example.com",
            sender_password="password123",
            recipients=["recipient@example.com"],
        )

    def test_email_channel_init(self, email_channel):
        """Test email channel initialization."""
        assert email_channel.smtp_server == "smtp.gmail.com"
        assert email_channel.smtp_port == 587
        assert email_channel.sender_email == "test@example.com"
        assert len(email_channel.recipients) == 1

    def test_email_channel_validate_config_valid(self, email_channel):
        """Test validation of valid email configuration."""
        assert email_channel.validate_config() is True

    def test_email_channel_validate_config_missing_smtp(self):
        """Test validation fails without SMTP server."""
        channel = EmailAlertChannel(
            smtp_server="",
            smtp_port=587,
            sender_email="test@example.com",
            sender_password="password123",
            recipients=["recipient@example.com"],
        )
        assert channel.validate_config() is False

    def test_email_channel_validate_config_no_recipients(self):
        """Test validation fails without recipients."""
        channel = EmailAlertChannel(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="test@example.com",
            sender_password="password123",
            recipients=[],
        )
        assert channel.validate_config() is False

    def test_email_format_text_email(self):
        """Test plain text email formatting."""
        alert = {
            "title": "Test Alert",
            "severity": "critical",
            "component": "performance",
            "message": "Test message",
            "metric_name": "accuracy",
            "metric_value": 0.75,
            "threshold": 0.8,
            "model_version": "v1.0",
        }
        text = EmailAlertChannel._format_text_email(alert)
        assert "Test Alert" in text
        assert "CRITICAL" in text
        assert "accuracy" in text

    def test_email_format_html_email(self):
        """Test HTML email formatting."""
        alert = {
            "title": "Test Alert",
            "severity": "warning",
            "component": "drift",
            "message": "Drift detected",
            "metric_name": "psi",
            "metric_value": 0.3,
            "threshold": 0.25,
            "model_version": "v1.0",
        }
        html = EmailAlertChannel._format_html_email(alert)
        assert "<html>" in html
        assert "Test Alert" in html
        assert "#FFC107" in html  # Warning color

    @patch("smtplib.SMTP")
    def test_email_send_success(self, mock_smtp, email_channel):
        """Test successful email sending."""
        mock_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_instance

        alert = {
            "alert_id": "alert_1",
            "title": "Test",
            "severity": "info",
            "component": "test",
            "message": "Test message",
            "metric_name": "test_metric",
            "metric_value": 0.5,
            "threshold": 0.7,
            "model_version": "v1",
        }

        result = email_channel.send(alert)
        assert result is True
        mock_instance.starttls.assert_called_once()
        mock_instance.login.assert_called_once()

    @patch("smtplib.SMTP")
    def test_email_send_failure(self, mock_smtp, email_channel):
        """Test email send failure handling."""
        mock_smtp.side_effect = Exception("SMTP Error")

        alert = {
            "alert_id": "alert_1",
            "title": "Test",
            "severity": "critical",
            "component": "test",
            "message": "Test",
            "metric_name": "test",
            "metric_value": 0.5,
            "threshold": 0.7,
            "model_version": "v1",
        }

        result = email_channel.send(alert)
        assert result is False


class TestSlackAlertChannel:
    """Test Slack alert channel."""

    @pytest.fixture
    def slack_channel(self):
        """Create Slack channel instance."""
        return SlackAlertChannel(
            bot_token="xoxb-test-token",
            channel_id="C123456",
        )

    def test_slack_channel_init(self, slack_channel):
        """Test Slack channel initialization."""
        assert slack_channel.bot_token == "xoxb-test-token"
        assert slack_channel.channel_id == "C123456"

    def test_slack_format_message(self):
        """Test Slack message formatting."""
        alert = {
            "title": "Performance Alert",
            "severity": "critical",
            "component": "model",
            "message": "Model accuracy degraded",
            "metric_name": "accuracy",
            "metric_value": 0.71,
            "threshold": 0.8,
            "model_version": "v1.0",
        }

        blocks = SlackAlertChannel._format_slack_message(alert)
        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"
        assert "Performance Alert" in blocks[0]["text"]["text"]

    def test_slack_message_severity_colors(self):
        """Test Slack message colors for different severities."""
        severities = {
            "critical": "#DC3545",
            "warning": "#FFC107",
            "info": "#28A745",
        }

        for severity, color in severities.items():
            alert = {
                "title": "Test",
                "severity": severity,
                "component": "test",
                "message": "Test",
                "metric_name": "test",
                "metric_value": 0.5,
                "threshold": 0.7,
                "model_version": "v1",
            }
            blocks = SlackAlertChannel._format_slack_message(alert)
            assert blocks  # Should create valid message

    @patch("slack_sdk.WebClient.chat_postMessage")
    def test_slack_send_success(self, mock_post, slack_channel):
        """Test successful Slack message sending."""
        mock_post.return_value = {"ok": True}

        alert = {
            "alert_id": "alert_1",
            "title": "Test",
            "severity": "warning",
            "component": "test",
            "message": "Test message",
            "metric_name": "test_metric",
            "metric_value": 0.5,
            "threshold": 0.7,
            "model_version": "v1",
        }

        with patch.object(slack_channel.client, "chat_postMessage", return_value={"ok": True}):
            result = slack_channel.send(alert)
            assert result is True


class TestAlertConfig:
    """Test alert configuration."""

    def test_email_config_init(self):
        """Test email config initialization."""
        config = EmailConfig(
            enabled=True,
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="test@example.com",
            sender_password="password",
            recipients=["test@example.com"],
        )
        assert config.enabled is True
        assert config.smtp_server == "smtp.gmail.com"

    def test_email_config_parse_recipients_string(self):
        """Test email config parses recipients from comma-separated string."""
        config = EmailConfig(
            enabled=True,
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="test@example.com",
            sender_password="password",
            recipients="test1@example.com, test2@example.com",
        )
        assert len(config.recipients) == 2

    def test_email_config_is_valid(self):
        """Test email config validation."""
        config = EmailConfig(
            enabled=True,
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="test@example.com",
            sender_password="password",
            recipients=["test@example.com"],
        )
        assert config.is_valid() is True

    def test_email_config_disabled(self):
        """Test disabled email config is always valid."""
        config = EmailConfig(enabled=False)
        assert config.is_valid() is True

    def test_slack_config_init(self):
        """Test Slack config initialization."""
        config = SlackConfig(
            enabled=True,
            bot_token="xoxb-token",
            channel_id="C123456",
        )
        assert config.enabled is True
        assert config.bot_token == "xoxb-token"

    def test_slack_config_is_valid(self):
        """Test Slack config validation."""
        config = SlackConfig(
            enabled=True,
            bot_token="xoxb-token",
            channel_id="C123456",
        )
        assert config.is_valid() is True

    def test_alert_channel_config_load_from_dict(self):
        """Test loading alert config from dictionary."""
        config_dict = {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "test@example.com",
                "sender_password": "password",
                "recipients": ["test@example.com"],
            },
            "slack": {
                "enabled": True,
                "bot_token": "xoxb-token",
                "channel_id": "C123456",
            },
        }

        config = load_alert_config_from_dict(config_dict)
        assert config.email.enabled is True
        assert config.slack.enabled is True

    def test_validate_alert_config_valid(self):
        """Test validation of valid alert config."""
        config = AlertChannelConfig(
            email=EmailConfig(
                enabled=True,
                smtp_server="smtp.gmail.com",
                smtp_port=587,
                sender_email="test@example.com",
                sender_password="password",
                recipients=["test@example.com"],
            ),
            slack=SlackConfig(
                enabled=True,
                bot_token="xoxb-token",
                channel_id="C123456",
            ),
        )

        is_valid, msg = validate_alert_config(config)
        assert is_valid is True

    def test_validate_alert_config_invalid_email(self):
        """Test validation fails with invalid email config."""
        config = AlertChannelConfig(
            email=EmailConfig(enabled=True, smtp_server="", sender_email="test@example.com"),
            slack=SlackConfig(enabled=False),
        )

        is_valid, msg = validate_alert_config(config)
        assert is_valid is False
        assert "Email" in msg

    @patch.dict("os.environ", {
        "ALERT_EMAIL_ENABLED": "true",
        "ALERT_EMAIL_SMTP_SERVER": "smtp.gmail.com",
        "ALERT_EMAIL_SMTP_PORT": "587",
        "ALERT_EMAIL_SENDER": "test@example.com",
        "ALERT_EMAIL_PASSWORD": "password",
        "ALERT_EMAIL_RECIPIENTS": "test@example.com",
    })
    def test_load_config_from_env(self):
        """Test loading config from environment variables."""
        config = load_alert_config_from_env()
        assert config.email.enabled is True
        assert config.email.smtp_server == "smtp.gmail.com"
        assert "test@example.com" in config.email.recipients
