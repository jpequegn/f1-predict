"""CLI chatbot interface for F1 predictions.

Provides an interactive command-line chat interface for querying
F1 predictions and getting natural language explanations.
"""

import asyncio
import os
import sys
from typing import Any, Optional

import click
import structlog

from f1_predict.llm.base import LLMConfig
from f1_predict.llm.chat_session import ChatSession, ChatSessionManager
from f1_predict.llm.exceptions import (
    AuthenticationError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
)

logger = structlog.get_logger(__name__)

# Provider configurations
PROVIDER_CONFIGS = {
    "anthropic": {
        "model": "claude-3-haiku-20240307",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Anthropic Claude (fast and cost-effective)",
    },
    "openai": {
        "model": "gpt-3.5-turbo",
        "env_key": "OPENAI_API_KEY",
        "description": "OpenAI GPT-3.5 Turbo",
    },
    "local": {
        "model": "llama3.2",
        "env_key": None,
        "description": "Local Ollama (no API key required)",
    },
}

# F1 chat system prompt
F1_SYSTEM_PROMPT = """You are an expert F1 (Formula 1) analyst assistant. You help users with:
- Race predictions and analysis
- Driver and team performance comparisons
- Circuit analysis and strategy insights
- Historical F1 data and statistics
- Explaining race strategies and regulations

Be concise but informative. Use data and statistics when available.
If you don't know something specific, say so clearly.
Format your responses for easy reading in a terminal."""

# Suggested queries for help
SUGGESTED_QUERIES = [
    "Who will win the next race?",
    "Compare Max Verstappen and Lewis Hamilton",
    "What are the key factors for Monaco GP?",
    "Explain DRS and how it affects overtaking",
    "What is the tire degradation strategy?",
    "Who are the top championship contenders?",
]

# Command prefixes
COMMAND_PREFIX = "/"
COMMANDS = {
    "/help": "Show available commands",
    "/clear": "Clear chat history",
    "/history": "Show conversation history",
    "/provider": "Change LLM provider",
    "/model": "Show current model info",
    "/suggest": "Show suggested queries",
    "/quit": "Exit the chat",
    "/exit": "Exit the chat",
}


def get_provider(
    provider_name: str,
    model: Optional[str] = None,
) -> Optional[Any]:
    """Get an LLM provider instance.

    Args:
        provider_name: Name of provider
        model: Optional model override

    Returns:
        Provider instance or None if not available
    """
    config_info = PROVIDER_CONFIGS.get(provider_name)
    if not config_info:
        click.echo(f"Unknown provider: {provider_name}", err=True)
        return None

    model = model or config_info["model"]
    config = LLMConfig(
        model=model,
        temperature=0.7,
        max_tokens=1024,
    )

    try:
        if provider_name == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                click.echo("ANTHROPIC_API_KEY not set", err=True)
                return None
            from f1_predict.llm.anthropic_provider import AnthropicProvider

            return AnthropicProvider(config, api_key)

        if provider_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                click.echo("OPENAI_API_KEY not set", err=True)
                return None
            from f1_predict.llm.openai_provider import OpenAIProvider

            return OpenAIProvider(config, api_key)

        if provider_name == "local":
            from f1_predict.llm.local_provider import LocalProvider

            return LocalProvider(config)

        return None

    except Exception as e:
        click.echo(f"Failed to initialize provider: {e}", err=True)
        return None


async def chat_response(
    provider: Any,
    message: str,
    session: ChatSession,
) -> str:
    """Get chat response from LLM.

    Args:
        provider: LLM provider instance
        message: User message
        session: Chat session for context

    Returns:
        Assistant response
    """
    # Add user message to session
    session.add_message("user", message)

    # Build conversation history for context
    history = session.get_messages()
    context = "\n".join(
        [
            f"{msg.role}: {msg.content}"
            for msg in history[-10:]  # Last 10 messages for context
        ]
    )

    prompt = f"""Previous conversation:
{context}

Current question: {message}

Provide a helpful response based on the conversation context."""

    try:
        response = await provider.generate(
            prompt=prompt,
            system_prompt=F1_SYSTEM_PROMPT,
        )
        assistant_message = response.content
        session.add_message("assistant", assistant_message)
        return assistant_message

    except RateLimitError:
        return "Rate limit exceeded. Please wait a moment and try again."
    except AuthenticationError:
        return "Authentication failed. Please check your API key."
    except ProviderUnavailableError:
        return "Service temporarily unavailable. Please try again later."
    except LLMError as e:
        return f"LLM error: {e}"
    except Exception as e:
        logger.error("chat_error", error=str(e))
        return f"An error occurred: {e}"


def handle_command(
    command: str,
    session: ChatSession,
    provider_name: str,
    provider: Any,
) -> tuple[bool, str, Optional[str]]:
    """Handle chat command.

    Args:
        command: Command string (e.g., '/help')
        session: Current chat session
        provider_name: Current provider name
        provider: Current provider instance

    Returns:
        Tuple of (should_continue, output_message, new_provider_name)
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return False, "Goodbye! Thanks for chatting.", None

    if cmd == "/help":
        help_text = "Available commands:\n"
        for cmd_name, description in COMMANDS.items():
            help_text += f"  {cmd_name:12} - {description}\n"
        return True, help_text, None

    if cmd == "/clear":
        session.clear_history()
        return True, "Chat history cleared.", None

    if cmd == "/history":
        messages = session.get_messages()
        if not messages:
            return True, "No messages in history.", None
        history_text = "Conversation history:\n"
        for i, msg in enumerate(messages[-20:], 1):  # Last 20 messages
            role = "You" if msg.role == "user" else "Assistant"
            content = (
                msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            )
            history_text += f"  {i}. [{role}] {content}\n"
        return True, history_text, None

    if cmd == "/provider":
        if args:
            if args.lower() in PROVIDER_CONFIGS:
                return True, f"Switching to {args} provider...", args.lower()
            available = ", ".join(PROVIDER_CONFIGS.keys())
            return True, f"Unknown provider '{args}'. Available: {available}", None
        current = f"Current provider: {provider_name}"
        available = "\nAvailable providers:\n"
        for name, info in PROVIDER_CONFIGS.items():
            marker = " (current)" if name == provider_name else ""
            available += f"  - {name}: {info['description']}{marker}\n"
        return True, current + available, None

    if cmd == "/model":
        if provider:
            return (
                True,
                f"Current model: {provider.config.model}\nProvider: {provider_name}",
                None,
            )
        return True, f"No provider initialized. Current setting: {provider_name}", None

    if cmd == "/suggest":
        suggestions = "Suggested queries:\n"
        for i, query in enumerate(SUGGESTED_QUERIES, 1):
            suggestions += f"  {i}. {query}\n"
        return True, suggestions, None

    return True, f"Unknown command: {cmd}. Type /help for available commands.", None


def print_welcome():
    """Print welcome message."""
    click.echo("\n" + "=" * 60)
    click.echo("  F1 Predict - AI Chat Assistant")
    click.echo("=" * 60)
    click.echo("\nAsk me anything about F1 predictions, drivers, or races!")
    click.echo("Type /help for commands, /quit to exit.\n")


def print_response(response: str):
    """Print assistant response with formatting."""
    click.echo()
    click.echo(click.style("Assistant: ", fg="green", bold=True), nl=False)
    click.echo(response)
    click.echo()


@click.command()
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "openai", "local"]),
    default="anthropic",
    help="LLM provider to use",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Model to use (overrides provider default)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def chat(provider: str, model: Optional[str], verbose: bool):
    """Interactive F1 prediction chat assistant.

    Start a chat session to ask questions about F1 predictions,
    driver comparisons, race analysis, and more.

    Example:
        f1-predict chat --provider anthropic
        f1-predict chat -p local -m llama3.2
    """
    # Initialize provider
    current_provider_name = provider
    current_provider = get_provider(current_provider_name, model)

    if not current_provider:
        click.echo(
            "\nFailed to initialize provider. Check your API key or try 'local' provider."
        )
        click.echo("Example: f1-predict chat --provider local")
        sys.exit(1)

    # Initialize session manager and session
    session_manager = ChatSessionManager()
    session = session_manager.create_session(metadata={"user_id": "cli_user"})

    print_welcome()
    click.echo(
        f"Using provider: {current_provider_name} ({current_provider.config.model})"
    )
    click.echo()

    # Main chat loop
    while True:
        try:
            user_input = click.prompt(
                click.style("You", fg="blue", bold=True),
                prompt_suffix=": ",
            )

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith(COMMAND_PREFIX):
                should_continue, output, new_provider = handle_command(
                    user_input, session, current_provider_name, current_provider
                )
                click.echo(output)

                if new_provider:
                    new_provider_instance = get_provider(new_provider, model)
                    if new_provider_instance:
                        current_provider_name = new_provider
                        current_provider = new_provider_instance
                        click.echo(f"Switched to {current_provider_name}")
                    else:
                        click.echo(f"Failed to switch to {new_provider}")

                if not should_continue:
                    break
                continue

            # Get chat response
            click.echo(click.style("Thinking...", fg="yellow"), nl=False)

            response = asyncio.run(chat_response(current_provider, user_input, session))

            # Clear "Thinking..." and print response
            click.echo("\r" + " " * 20 + "\r", nl=False)
            print_response(response)

        except KeyboardInterrupt:
            click.echo("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            click.echo("\n\nGoodbye!")
            break
        except Exception as e:
            if verbose:
                logger.exception("chat_error")
            click.echo(f"\nError: {e}")


@click.command()
@click.argument("question")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "openai", "local"]),
    default="anthropic",
    help="LLM provider to use",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Model to use",
)
def ask(question: str, provider: str, model: Optional[str]):
    """Ask a single question about F1.

    For interactive chat, use the 'chat' command instead.

    Example:
        f1-predict ask "Who will win the next race?"
        f1-predict ask "Compare Verstappen and Hamilton" -p openai
    """
    llm_provider = get_provider(provider, model)
    if not llm_provider:
        click.echo("Failed to initialize provider.", err=True)
        sys.exit(1)

    click.echo(click.style("Thinking...", fg="yellow"))

    try:
        response = asyncio.run(
            llm_provider.generate(
                prompt=question,
                system_prompt=F1_SYSTEM_PROMPT,
            )
        )
        click.echo("\r" + " " * 20 + "\r", nl=False)
        click.echo(click.style("\nAnswer: ", fg="green", bold=True))
        click.echo(response.content)
        click.echo()

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)


# Main CLI group
@click.group()
def cli():
    """F1 Predict CLI with AI chat capabilities."""
    pass


cli.add_command(chat)
cli.add_command(ask)


if __name__ == "__main__":
    cli()
