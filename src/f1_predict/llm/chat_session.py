"""Chat session state management for multi-turn conversations.

This module provides session management for maintaining conversation state
across multiple turns with LLM providers, enabling context-aware interactions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ChatMessage:
    """Single message in a chat session.

    Attributes:
        role: Message role ('user', 'assistant', 'system')
        content: Message content text
        timestamp: When message was created
        metadata: Additional message metadata
    """

    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary.

        Returns:
            Dictionary representation of message
        """
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ChatSession:
    """Manages conversation state across multiple turns.

    Maintains chat history, session metadata, and provides
    message management utilities.

    Attributes:
        session_id: Unique session identifier
        messages: List of chat messages
        created_at: Session creation timestamp
        last_updated: Last message timestamp
        model: LLM model being used
        provider: LLM provider name
        metadata: Session-level metadata
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    model: Optional[str] = None
    provider: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def last_updated(self) -> Optional[datetime]:
        """Get timestamp of last message.

        Returns:
            Timestamp of most recent message or None if no messages
        """
        return self.messages[-1].timestamp if self.messages else None

    @property
    def message_count(self) -> int:
        """Get total number of messages.

        Returns:
            Number of messages in session
        """
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        """Get count of user messages.

        Returns:
            Number of user messages
        """
        return sum(1 for msg in self.messages if msg.role == "user")

    @property
    def assistant_message_count(self) -> int:
        """Get count of assistant messages.

        Returns:
            Number of assistant messages
        """
        return sum(1 for msg in self.messages if msg.role == "assistant")

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ChatMessage:
        """Add message to session.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional message metadata

        Returns:
            Created ChatMessage instance

        Raises:
            ValueError: If role is invalid
        """
        if role not in ("user", "assistant", "system"):
            msg = f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'"
            raise ValueError(msg)

        if not content or not isinstance(content, str):
            msg = "Content must be non-empty string"
            raise ValueError(msg)

        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        self.messages.append(message)
        logger.debug(
            "message_added",
            session_id=self.session_id,
            role=role,
            message_count=self.message_count,
        )

        return message

    def get_messages(self, role: Optional[str] = None) -> list[ChatMessage]:
        """Get messages, optionally filtered by role.

        Args:
            role: Optional role filter ('user', 'assistant', 'system')

        Returns:
            List of ChatMessage objects
        """
        if role is None:
            return self.messages.copy()

        return [msg for msg in self.messages if msg.role == role]

    def get_conversation_text(self, include_metadata: bool = False) -> str:
        """Get formatted conversation text.

        Args:
            include_metadata: Whether to include message metadata in output

        Returns:
            Formatted conversation string
        """
        lines = []
        for msg in self.messages:
            role_label = msg.role.upper()
            if include_metadata and msg.metadata:
                metadata_str = f" {msg.metadata}"
            else:
                metadata_str = ""
            lines.append(f"{role_label}{metadata_str}: {msg.content}")

        return "\n".join(lines)

    def get_context_for_llm(self, max_messages: Optional[int] = None) -> list[dict[str, str]]:
        """Get messages formatted for LLM API call.

        Args:
            max_messages: Maximum recent messages to include (for context window limits)

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = self.messages
        if max_messages is not None:
            messages = messages[-max_messages:]

        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def clear_history(self) -> None:
        """Clear all messages from session."""
        old_count = len(self.messages)
        self.messages.clear()
        logger.info(
            "session_cleared",
            session_id=self.session_id,
            messages_cleared=old_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary.

        Returns:
            Dictionary representation of session
        """
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "model": self.model,
            "provider": self.provider,
            "metadata": self.metadata,
        }


class ChatSessionManager:
    """Manage multiple chat sessions.

    Provides session lifecycle management, storage, and retrieval.
    """

    def __init__(self) -> None:
        """Initialize session manager."""
        self.sessions: dict[str, ChatSession] = {}
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("chat_session_manager_initialized")

    def create_session(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ChatSession:
        """Create new chat session.

        Args:
            model: LLM model identifier
            provider: LLM provider name
            metadata: Session-level metadata

        Returns:
            New ChatSession instance
        """
        session = ChatSession(
            model=model,
            provider=provider,
            metadata=metadata or {},
        )

        self.sessions[session.session_id] = session
        self.logger.info(
            "session_created",
            session_id=session.session_id,
            model=model,
            provider=provider,
        )

        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get existing session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ChatSession if found, None otherwise
        """
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete session and its history.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info("session_deleted", session_id=session_id)
            return True

        return False

    def get_all_sessions(self) -> list[ChatSession]:
        """Get all sessions.

        Returns:
            List of ChatSession objects
        """
        return list(self.sessions.values())

    def session_count(self) -> int:
        """Get total number of sessions.

        Returns:
            Number of active sessions
        """
        return len(self.sessions)

    def clear_all_sessions(self) -> int:
        """Clear all sessions.

        Returns:
            Number of sessions cleared
        """
        count = len(self.sessions)
        self.sessions.clear()
        self.logger.info("all_sessions_cleared", count=count)
        return count
