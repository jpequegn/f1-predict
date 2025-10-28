"""Configuration management for alert channels.

Handles configuration loading and validation for email and Slack alert channels
from environment variables.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger(__name__)


class EmailConfig(BaseModel):
    """Email alert channel configuration."""

    enabled: bool = False
    smtp_server: str = Field(default="", description="SMTP server address")
    smtp_port: int = Field(default=587, description="SMTP port (usually 587 for TLS)")
    sender_email: str = Field(default="", description="Sender email address")
    sender_password: str = Field(default="", description="Sender email password/app password")
    recipients: list[str] = Field(default_factory=list, description="Recipient email addresses")
    use_tls: bool = Field(default=True, description="Use TLS for SMTP connection")

    class Config:
        """Pydantic config."""

        extra = "allow"

    @validator("recipients", pre=True)
    def parse_recipients(cls, v: Any) -> list[str]:
        """Parse recipients from comma-separated string or list.

        Args:
            v: Recipients value (str or list)

        Returns:
            List of recipient emails
        """
        if isinstance(v, str):
            return [email.strip() for email in v.split(",") if email.strip()]
        if isinstance(v, list):
            return v
        return []

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns:
            True if valid configuration
        """
        if not self.enabled:
            return True

        required_fields = [
            self.smtp_server,
            self.sender_email,
            self.sender_password,
            self.recipients,
        ]
        return all(required_fields) and len(self.recipients) > 0


class SlackConfig(BaseModel):
    """Slack alert channel configuration."""

    enabled: bool = False
    bot_token: str = Field(default="", description="Slack bot token (xoxb-...)")
    channel_id: str = Field(default="", description="Default Slack channel ID")
    app_token: Optional[str] = Field(default=None, description="Slack app token (optional)")

    class Config:
        """Pydantic config."""

        extra = "allow"

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns:
            True if valid configuration
        """
        if not self.enabled:
            return True

        return bool(self.bot_token and self.channel_id)


class AlertChannelConfig(BaseModel):
    """Overall alert channel configuration."""

    email: EmailConfig = Field(default_factory=EmailConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)

    class Config:
        """Pydantic config."""

        extra = "allow"


def load_alert_config_from_env() -> AlertChannelConfig:
    """Load alert channel configuration from environment variables.

    Environment variables:
    - ALERT_EMAIL_ENABLED: Enable email alerts (true/false)
    - ALERT_EMAIL_SMTP_SERVER: SMTP server address
    - ALERT_EMAIL_SMTP_PORT: SMTP port
    - ALERT_EMAIL_SENDER: Sender email address
    - ALERT_EMAIL_PASSWORD: Sender password
    - ALERT_EMAIL_RECIPIENTS: Comma-separated recipient emails
    - ALERT_SLACK_ENABLED: Enable Slack alerts (true/false)
    - ALERT_SLACK_BOT_TOKEN: Slack bot token
    - ALERT_SLACK_CHANNEL: Default Slack channel ID

    Returns:
        AlertChannelConfig instance
    """
    import os

    try:
        # Email configuration
        email_config = EmailConfig(
            enabled=os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true",
            smtp_server=os.getenv("ALERT_EMAIL_SMTP_SERVER", ""),
            smtp_port=int(os.getenv("ALERT_EMAIL_SMTP_PORT", "587")),
            sender_email=os.getenv("ALERT_EMAIL_SENDER", ""),
            sender_password=os.getenv("ALERT_EMAIL_PASSWORD", ""),
            recipients=os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(","),
            use_tls=os.getenv("ALERT_EMAIL_USE_TLS", "true").lower() == "true",
        )

        # Slack configuration
        slack_config = SlackConfig(
            enabled=os.getenv("ALERT_SLACK_ENABLED", "false").lower() == "true",
            bot_token=os.getenv("ALERT_SLACK_BOT_TOKEN", ""),
            channel_id=os.getenv("ALERT_SLACK_CHANNEL", ""),
        )

        config = AlertChannelConfig(email=email_config, slack=slack_config)

        logger.info(
            "alert_config_loaded",
            email_enabled=email_config.enabled,
            slack_enabled=slack_config.enabled,
        )

        return config

    except Exception as e:
        logger.error("alert_config_load_failed", error=str(e))
        return AlertChannelConfig()


def load_alert_config_from_dict(config_dict: dict[str, Any]) -> AlertChannelConfig:
    """Load alert channel configuration from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        AlertChannelConfig instance
    """
    try:
        email_data = config_dict.get("email", {})
        slack_data = config_dict.get("slack", {})

        email_config = EmailConfig(**email_data) if email_data else EmailConfig()
        slack_config = SlackConfig(**slack_data) if slack_data else SlackConfig()

        return AlertChannelConfig(email=email_config, slack=slack_config)

    except Exception as e:
        logger.error("alert_config_parse_failed", error=str(e))
        return AlertChannelConfig()


def validate_alert_config(config: AlertChannelConfig) -> tuple[bool, str]:
    """Validate alert channel configuration.

    Args:
        config: AlertChannelConfig instance

    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []

    if config.email.enabled and not config.email.is_valid():
        errors.append("Email configuration is incomplete or invalid")

    if config.slack.enabled and not config.slack.is_valid():
        errors.append("Slack configuration is incomplete or invalid")

    if errors:
        return False, "; ".join(errors)

    return True, ""
