"""Tests for chat session management."""

from datetime import datetime, timezone

import pytest

from f1_predict.llm.chat_session import ChatMessage, ChatSession, ChatSessionManager


class TestChatMessage:
    """Test ChatMessage class."""

    def test_message_creation(self):
        """Test creating ChatMessage."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)

    def test_message_with_metadata(self):
        """Test message with metadata."""
        metadata = {"source": "web", "user_id": "123"}
        msg = ChatMessage(role="user", content="Test", metadata=metadata)
        assert msg.metadata == metadata

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = ChatMessage(role="assistant", content="Response")
        data = msg.to_dict()
        assert data["role"] == "assistant"
        assert data["content"] == "Response"
        assert "timestamp" in data


class TestChatSession:
    """Test ChatSession class."""

    def test_session_creation(self):
        """Test creating ChatSession."""
        session = ChatSession(model="gpt-4", provider="openai")
        assert session.model == "gpt-4"
        assert session.provider == "openai"
        assert session.message_count == 0

    def test_add_user_message(self):
        """Test adding user message."""
        session = ChatSession()
        msg = session.add_message("user", "Hello assistant")
        assert msg.role == "user"
        assert session.message_count == 1
        assert session.user_message_count == 1

    def test_add_assistant_message(self):
        """Test adding assistant message."""
        session = ChatSession()
        msg = session.add_message("assistant", "Hello user")
        assert msg.role == "assistant"
        assert session.assistant_message_count == 1

    def test_add_system_message(self):
        """Test adding system message."""
        session = ChatSession()
        msg = session.add_message("system", "You are helpful")
        assert msg.role == "system"
        assert session.message_count == 1

    def test_add_invalid_role(self):
        """Test adding message with invalid role."""
        session = ChatSession()
        with pytest.raises(ValueError, match="Invalid role"):
            session.add_message("invalid", "Content")

    def test_add_empty_content(self):
        """Test adding message with empty content."""
        session = ChatSession()
        with pytest.raises(ValueError, match="non-empty string"):
            session.add_message("user", "")

    def test_add_non_string_content(self):
        """Test adding message with non-string content."""
        session = ChatSession()
        with pytest.raises(ValueError):
            session.add_message("user", 12345)  # type: ignore

    def test_get_all_messages(self):
        """Test retrieving all messages."""
        session = ChatSession()
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.add_message("user", "Q2")

        messages = session.get_messages()
        assert len(messages) == 3

    def test_get_messages_by_role(self):
        """Test retrieving messages filtered by role."""
        session = ChatSession()
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.add_message("user", "Q2")

        user_msgs = session.get_messages(role="user")
        assert len(user_msgs) == 2
        assert all(msg.role == "user" for msg in user_msgs)

    def test_get_conversation_text(self):
        """Test getting formatted conversation."""
        session = ChatSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")

        text = session.get_conversation_text()
        assert "USER: Hello" in text
        assert "ASSISTANT: Hi there" in text

    def test_get_context_for_llm(self):
        """Test getting context for LLM API."""
        session = ChatSession()
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")

        context = session.get_context_for_llm()
        assert len(context) == 2
        assert context[0] == {"role": "user", "content": "Q1"}
        assert context[1] == {"role": "assistant", "content": "A1"}

    def test_get_context_with_max_messages(self):
        """Test getting context with message limit."""
        session = ChatSession()
        for i in range(5):
            session.add_message("user", f"Q{i}")

        context = session.get_context_for_llm(max_messages=3)
        assert len(context) == 3

    def test_last_updated_property(self):
        """Test last_updated property."""
        session = ChatSession()
        assert session.last_updated is None

        session.add_message("user", "First")
        time1 = session.last_updated

        session.add_message("user", "Second")
        time2 = session.last_updated

        assert time1 is not None
        assert time2 is not None
        assert time2 >= time1

    def test_message_counts(self):
        """Test message counting methods."""
        session = ChatSession()
        session.add_message("system", "System")
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.add_message("user", "Q2")
        session.add_message("assistant", "A2")

        assert session.message_count == 5
        assert session.user_message_count == 2
        assert session.assistant_message_count == 2

    def test_clear_history(self):
        """Test clearing session history."""
        session = ChatSession()
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        assert session.message_count == 2

        session.clear_history()
        assert session.message_count == 0

    def test_session_to_dict(self):
        """Test converting session to dict."""
        session = ChatSession(model="gpt-4", provider="openai")
        session.add_message("user", "Hello")

        data = session.to_dict()
        assert data["model"] == "gpt-4"
        assert data["provider"] == "openai"
        assert len(data["messages"]) == 1
        assert "session_id" in data
        assert "created_at" in data

    def test_session_metadata(self):
        """Test session-level metadata."""
        metadata = {"project": "f1predict", "version": "1.0"}
        session = ChatSession(metadata=metadata)
        assert session.metadata == metadata

    def test_message_with_metadata_in_conversation(self):
        """Test message metadata in conversation."""
        session = ChatSession()
        session.add_message("user", "Query", metadata={"tokens": 5})
        session.add_message("assistant", "Response", metadata={"tokens": 10})

        text = session.get_conversation_text(include_metadata=True)
        assert "tokens" in text


class TestChatSessionManager:
    """Test ChatSessionManager class."""

    def test_manager_initialization(self):
        """Test creating ChatSessionManager."""
        manager = ChatSessionManager()
        assert manager.session_count() == 0

    def test_create_session(self):
        """Test creating session through manager."""
        manager = ChatSessionManager()
        session = manager.create_session(model="gpt-4", provider="openai")
        assert session is not None
        assert manager.session_count() == 1

    def test_get_session(self):
        """Test retrieving session by ID."""
        manager = ChatSessionManager()
        created = manager.create_session()
        retrieved = manager.get_session(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session(self):
        """Test retrieving non-existent session."""
        manager = ChatSessionManager()
        session = manager.get_session("nonexistent-id")
        assert session is None

    def test_delete_session(self):
        """Test deleting session."""
        manager = ChatSessionManager()
        session = manager.create_session()
        assert manager.session_count() == 1

        deleted = manager.delete_session(session.session_id)
        assert deleted is True
        assert manager.session_count() == 0

    def test_delete_nonexistent_session(self):
        """Test deleting non-existent session."""
        manager = ChatSessionManager()
        deleted = manager.delete_session("nonexistent-id")
        assert deleted is False

    def test_get_all_sessions(self):
        """Test retrieving all sessions."""
        manager = ChatSessionManager()
        manager.create_session(model="gpt-4")
        manager.create_session(model="gpt-3.5")
        manager.create_session(model="claude-3")

        sessions = manager.get_all_sessions()
        assert len(sessions) == 3

    def test_clear_all_sessions(self):
        """Test clearing all sessions."""
        manager = ChatSessionManager()
        manager.create_session()
        manager.create_session()
        manager.create_session()

        count = manager.clear_all_sessions()
        assert count == 3
        assert manager.session_count() == 0

    def test_session_persistence(self):
        """Test session state is maintained."""
        manager = ChatSessionManager()
        session1 = manager.create_session()
        session1.add_message("user", "Hello")

        retrieved = manager.get_session(session1.session_id)
        assert retrieved is not None
        assert retrieved.message_count == 1
        assert retrieved.get_messages()[0].content == "Hello"

    def test_multiple_sessions_independent(self):
        """Test multiple sessions are independent."""
        manager = ChatSessionManager()
        session1 = manager.create_session()
        session2 = manager.create_session()

        session1.add_message("user", "Question 1")
        session2.add_message("user", "Question 2")

        assert session1.message_count == 1
        assert session2.message_count == 1
        assert session1.get_messages()[0].content != session2.get_messages()[0].content
