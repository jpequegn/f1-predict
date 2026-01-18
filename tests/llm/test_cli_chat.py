"""Tests for CLI chatbot interface."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from click.testing import CliRunner

from f1_predict.cli_chat import (
    COMMANDS,
    F1_SYSTEM_PROMPT,
    PROVIDER_CONFIGS,
    SUGGESTED_QUERIES,
    chat,
    ask,
    cli,
    get_provider,
    handle_command,
)


class TestProviderConfigs:
    """Test provider configuration."""

    def test_provider_configs_exist(self):
        """Test all expected providers are configured."""
        assert "anthropic" in PROVIDER_CONFIGS
        assert "openai" in PROVIDER_CONFIGS
        assert "local" in PROVIDER_CONFIGS

    def test_provider_configs_have_required_keys(self):
        """Test each provider config has required keys."""
        required_keys = {"model", "env_key", "description"}
        for provider, config in PROVIDER_CONFIGS.items():
            assert required_keys.issubset(config.keys()), f"{provider} missing keys"

    def test_local_provider_no_env_key(self):
        """Test local provider doesn't require API key."""
        assert PROVIDER_CONFIGS["local"]["env_key"] is None


class TestGetProvider:
    """Test provider initialization."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("f1_predict.llm.anthropic_provider.AnthropicProvider")
    def test_get_anthropic_provider(self, mock_provider):
        """Test getting Anthropic provider with valid key."""
        mock_provider.return_value = MagicMock()
        provider = get_provider("anthropic")
        assert provider is not None
        mock_provider.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    def test_get_anthropic_provider_no_key(self):
        """Test Anthropic provider fails without API key."""
        # Clear any existing key
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        provider = get_provider("anthropic")
        assert provider is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("f1_predict.llm.openai_provider.OpenAIProvider")
    def test_get_openai_provider(self, mock_provider):
        """Test getting OpenAI provider with valid key."""
        mock_provider.return_value = MagicMock()
        provider = get_provider("openai")
        assert provider is not None
        mock_provider.assert_called_once()

    @patch("f1_predict.llm.local_provider.LocalProvider")
    def test_get_local_provider(self, mock_provider):
        """Test getting local provider (no key required)."""
        mock_provider.return_value = MagicMock()
        provider = get_provider("local")
        assert provider is not None
        mock_provider.assert_called_once()

    def test_get_unknown_provider(self):
        """Test getting unknown provider returns None."""
        provider = get_provider("unknown_provider")
        assert provider is None

    @patch("f1_predict.llm.local_provider.LocalProvider")
    def test_get_provider_with_custom_model(self, mock_provider):
        """Test getting provider with custom model override."""
        mock_provider.return_value = MagicMock()
        get_provider("local", model="custom-model")
        # Verify config was created with custom model
        call_args = mock_provider.call_args
        assert call_args[0][0].model == "custom-model"


class TestHandleCommand:
    """Test command handling."""

    @pytest.fixture
    def mock_session(self):
        """Create mock chat session."""
        session = MagicMock()
        session.get_messages.return_value = []
        return session

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        provider = MagicMock()
        provider.config.model = "test-model"
        return provider

    def test_help_command(self, mock_session, mock_provider):
        """Test /help command."""
        should_continue, output, new_provider = handle_command(
            "/help", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "Available commands" in output
        assert new_provider is None

    def test_quit_command(self, mock_session, mock_provider):
        """Test /quit command."""
        should_continue, output, new_provider = handle_command(
            "/quit", mock_session, "anthropic", mock_provider
        )
        assert should_continue is False
        assert "Goodbye" in output

    def test_exit_command(self, mock_session, mock_provider):
        """Test /exit command."""
        should_continue, output, new_provider = handle_command(
            "/exit", mock_session, "anthropic", mock_provider
        )
        assert should_continue is False

    def test_clear_command(self, mock_session, mock_provider):
        """Test /clear command."""
        should_continue, output, new_provider = handle_command(
            "/clear", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "cleared" in output.lower()
        mock_session.clear_history.assert_called_once()

    def test_history_command_empty(self, mock_session, mock_provider):
        """Test /history command with no messages."""
        mock_session.get_messages.return_value = []
        should_continue, output, new_provider = handle_command(
            "/history", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "No messages" in output

    def test_history_command_with_messages(self, mock_session, mock_provider):
        """Test /history command with messages."""
        mock_msg = MagicMock()
        mock_msg.role = "user"
        mock_msg.content = "Test message"
        mock_session.get_messages.return_value = [mock_msg]

        should_continue, output, new_provider = handle_command(
            "/history", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "history" in output.lower()

    def test_provider_command_list(self, mock_session, mock_provider):
        """Test /provider command without args lists providers."""
        should_continue, output, new_provider = handle_command(
            "/provider", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "anthropic" in output.lower()
        assert new_provider is None

    def test_provider_command_switch(self, mock_session, mock_provider):
        """Test /provider command with valid provider."""
        should_continue, output, new_provider = handle_command(
            "/provider openai", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "Switching" in output
        assert new_provider == "openai"

    def test_provider_command_invalid(self, mock_session, mock_provider):
        """Test /provider command with invalid provider."""
        should_continue, output, new_provider = handle_command(
            "/provider invalid", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "Unknown provider" in output
        assert new_provider is None

    def test_model_command(self, mock_session, mock_provider):
        """Test /model command."""
        should_continue, output, new_provider = handle_command(
            "/model", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "test-model" in output

    def test_suggest_command(self, mock_session, mock_provider):
        """Test /suggest command."""
        should_continue, output, new_provider = handle_command(
            "/suggest", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "Suggested queries" in output

    def test_unknown_command(self, mock_session, mock_provider):
        """Test unknown command."""
        should_continue, output, new_provider = handle_command(
            "/unknown", mock_session, "anthropic", mock_provider
        )
        assert should_continue is True
        assert "Unknown command" in output


class TestCommands:
    """Test CLI command configuration."""

    def test_commands_defined(self):
        """Test all commands are defined."""
        expected = {"/help", "/clear", "/history", "/provider", "/model", "/suggest", "/quit", "/exit"}
        assert expected.issubset(COMMANDS.keys())

    def test_suggested_queries_exist(self):
        """Test suggested queries are defined."""
        assert len(SUGGESTED_QUERIES) > 0
        for query in SUGGESTED_QUERIES:
            assert isinstance(query, str)
            assert len(query) > 0

    def test_system_prompt_defined(self):
        """Test F1 system prompt is defined."""
        assert F1_SYSTEM_PROMPT
        assert "F1" in F1_SYSTEM_PROMPT
        assert "analyst" in F1_SYSTEM_PROMPT.lower()


class TestCLICommands:
    """Test CLI entry points."""

    def test_cli_group_exists(self):
        """Test CLI group is defined."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "F1 Predict CLI" in result.output

    def test_chat_command_help(self):
        """Test chat command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--model" in result.output

    def test_ask_command_help(self):
        """Test ask command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ask", "--help"])
        assert result.exit_code == 0
        assert "QUESTION" in result.output

    @patch("f1_predict.cli_chat.get_provider")
    def test_chat_command_no_provider(self, mock_get_provider):
        """Test chat command fails gracefully without provider."""
        mock_get_provider.return_value = None
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "-p", "anthropic"])
        assert result.exit_code == 1
        assert "Failed to initialize" in result.output

    @patch("f1_predict.cli_chat.get_provider")
    def test_ask_command_no_provider(self, mock_get_provider):
        """Test ask command fails gracefully without provider."""
        mock_get_provider.return_value = None
        runner = CliRunner()
        result = runner.invoke(cli, ["ask", "Who will win?", "-p", "anthropic"])
        assert result.exit_code == 1
        assert "Failed to initialize" in result.output

    @patch("f1_predict.cli_chat.get_provider")
    @patch("f1_predict.cli_chat.asyncio.run")
    def test_ask_command_success(self, mock_async_run, mock_get_provider):
        """Test ask command with successful response."""
        mock_provider = MagicMock()
        mock_provider.config.model = "test-model"
        mock_get_provider.return_value = mock_provider

        mock_response = MagicMock()
        mock_response.content = "Test answer"
        mock_async_run.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(cli, ["ask", "Who will win?", "-p", "local"])
        assert result.exit_code == 0
        assert "Test answer" in result.output
