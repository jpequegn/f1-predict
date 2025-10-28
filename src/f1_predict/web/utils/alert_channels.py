"""Alert delivery channels for monitoring system.

Implements email, Slack, and other notification channels for alert delivery.
"""

from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from typing import Any, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import structlog

logger = structlog.get_logger(__name__)


class AlertChannel(ABC):
    """Abstract base class for alert delivery channels."""

    @abstractmethod
    def send(self, alert: dict[str, Any]) -> bool:
        """Send an alert through the channel.

        Args:
            alert: Alert dictionary with title, message, severity, etc.

        Returns:
            True if sent successfully, False otherwise
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate channel configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        pass


class EmailAlertChannel(AlertChannel):
    """Email alert delivery channel."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipients: list[str],
    ):
        """Initialize email channel.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port (usually 587 for TLS)
            sender_email: Sender email address
            sender_password: Sender email password/app password
            recipients: List of recipient email addresses
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipients = recipients
        self.logger = logger.bind(component="email_channel")

    def validate_config(self) -> bool:
        """Validate email configuration.

        Returns:
            True if configuration is valid
        """
        if not self.smtp_server or not self.smtp_port:
            self.logger.error("email_config_invalid", error="Missing SMTP config")
            return False

        if not self.sender_email or not self.sender_password:
            self.logger.error("email_config_invalid", error="Missing credentials")
            return False

        if not self.recipients or len(self.recipients) == 0:
            self.logger.error("email_config_invalid", error="No recipients configured")
            return False

        return True

    def send(self, alert: dict[str, Any]) -> bool:
        """Send alert via email.

        Args:
            alert: Alert dictionary

        Returns:
            True if sent successfully
        """
        try:
            if not self.validate_config():
                return False

            # Create email message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"[{alert.get('severity', 'INFO').upper()}] {alert.get('title', 'Alert')}"
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.recipients)

            # Plain text version
            text = self._format_text_email(alert)

            # HTML version
            html = self._format_html_email(alert)

            part1 = MIMEText(text, "plain")
            part2 = MIMEText(html, "html")

            message.attach(part1)
            message.attach(part2)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    self.recipients,
                    message.as_string(),
                )

            self.logger.info("alert_sent_via_email", alert_id=alert.get("alert_id"))
            return True

        except smtplib.SMTPAuthenticationError as e:
            self.logger.error("email_authentication_failed", error=str(e))
            return False
        except smtplib.SMTPException as e:
            self.logger.error("email_send_failed", error=str(e))
            return False
        except Exception as e:
            self.logger.error("email_unexpected_error", error=str(e))
            return False

    @staticmethod
    def _format_text_email(alert: dict[str, Any]) -> str:
        """Format alert as plain text email.

        Args:
            alert: Alert dictionary

        Returns:
            Formatted text email body
        """
        return (
            f"Alert: {alert.get('title', 'Monitoring Alert')}\n"
            f"Severity: {alert.get('severity', 'INFO').upper()}\n"
            f"Component: {alert.get('component', 'Unknown')}\n"
            f"Message: {alert.get('message', '')}\n"
            f"Metric: {alert.get('metric_name', 'N/A')}\n"
            f"Current Value: {alert.get('metric_value', 'N/A')}\n"
            f"Threshold: {alert.get('threshold', 'N/A')}\n"
            f"Model Version: {alert.get('model_version', 'N/A')}\n"
        )

    @staticmethod
    def _format_html_email(alert: dict[str, Any]) -> str:
        """Format alert as HTML email.

        Args:
            alert: Alert dictionary

        Returns:
            Formatted HTML email body
        """
        severity_color = {
            "critical": "#DC3545",
            "warning": "#FFC107",
            "info": "#28A745",
        }.get(alert.get("severity", "info").lower(), "#A3A9BF")

        return f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
                <div style="background-color: white; border-radius: 8px; padding: 20px; max-width: 600px; margin: 0 auto;">
                    <div style="border-left: 4px solid {severity_color}; padding-left: 16px; margin-bottom: 20px;">
                        <h2 style="margin: 0 0 10px 0; color: #121317;">
                            {alert.get('title', 'Monitoring Alert')}
                        </h2>
                        <p style="margin: 0; color: #666; font-size: 14px;">
                            Severity: <strong style="color: {severity_color};">
                                {alert.get('severity', 'INFO').upper()}
                            </strong>
                        </p>
                    </div>

                    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Message:</strong> {alert.get('message', '')}
                        </p>
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Component:</strong> {alert.get('component', 'Unknown')}
                        </p>
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Metric:</strong> {alert.get('metric_name', 'N/A')}
                        </p>
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Current Value:</strong> {alert.get('metric_value', 'N/A')}
                        </p>
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Threshold:</strong> {alert.get('threshold', 'N/A')}
                        </p>
                        <p style="margin: 10px 0; color: #333;">
                            <strong>Model Version:</strong> {alert.get('model_version', 'N/A')}
                        </p>
                    </div>

                    <p style="font-size: 12px; color: #999; margin: 20px 0 0 0;">
                        This is an automated alert from the F1 Model Monitoring System.
                    </p>
                </div>
            </body>
        </html>
        """


class SlackAlertChannel(AlertChannel):
    """Slack alert delivery channel."""

    def __init__(self, bot_token: str, channel_id: Optional[str] = None):
        """Initialize Slack channel.

        Args:
            bot_token: Slack bot token (starts with xoxb-)
            channel_id: Default channel ID (can be overridden per alert)
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.client = WebClient(token=bot_token)
        self.logger = logger.bind(component="slack_channel")

    def validate_config(self) -> bool:
        """Validate Slack configuration.

        Returns:
            True if configuration is valid
        """
        if not self.bot_token:
            self.logger.error("slack_config_invalid", error="Missing bot token")
            return False

        # Try to validate token
        try:
            self.client.auth_test()
            return True
        except SlackApiError as e:
            self.logger.error("slack_auth_failed", error=str(e))
            return False

    def send(self, alert: dict[str, Any]) -> bool:
        """Send alert via Slack.

        Args:
            alert: Alert dictionary

        Returns:
            True if sent successfully
        """
        try:
            if not self.validate_config():
                return False

            channel = alert.get("slack_channel", self.channel_id)
            if not channel:
                self.logger.error("slack_send_failed", error="No channel specified")
                return False

            # Create Slack message
            message = self._format_slack_message(alert)

            # Send message
            self.client.chat_postMessage(
                channel=channel,
                blocks=message,
                text=alert.get("title", "Alert"),  # Fallback text
            )

            self.logger.info(
                "alert_sent_via_slack",
                alert_id=alert.get("alert_id"),
                channel=channel,
            )
            return True

        except SlackApiError as e:
            self.logger.error("slack_send_failed", error=str(e))
            return False
        except Exception as e:
            self.logger.error("slack_unexpected_error", error=str(e))
            return False

    @staticmethod
    def _format_slack_message(alert: dict[str, Any]) -> list[dict[str, Any]]:
        """Format alert for Slack.

        Args:
            alert: Alert dictionary

        Returns:
            List of Slack block kit dictionaries
        """
        severity_emoji = {
            "critical": "🔴",
            "warning": "🟡",
            "info": "🟢",
        }.get(alert.get("severity", "info").lower(), "ℹ️")

        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity_emoji} {alert.get('title', 'Alert')}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{alert.get('severity', 'INFO').upper()}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Component:*\n{alert.get('component', 'Unknown')}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Metric:*\n{alert.get('metric_name', 'N/A')}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Current Value:*\n{alert.get('metric_value', 'N/A')}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Message:*\n{alert.get('message', '')}",
                },
            },
            {
                "type": "divider",
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📍 Threshold: `{alert.get('threshold', 'N/A')}` | Model: `{alert.get('model_version', 'N/A')}`",
                },
            },
        ]
