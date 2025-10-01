"""Tests for LLM cost tracking system."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from f1_predict.llm.cost_tracker import CostTracker, UsageRecord
from f1_predict.llm.exceptions import BudgetExceededError


@pytest.fixture
def temp_db(tmp_path):
    """Create temporary database path."""
    return tmp_path / "test_usage.db"


@pytest.fixture
def tracker(temp_db):
    """Create cost tracker with test database."""
    return CostTracker(
        db_path=temp_db,
        daily_budget=10.0,
        monthly_budget=200.0,
        alert_threshold=0.8,
    )


@pytest.fixture
def sample_record():
    """Create sample usage record."""
    return UsageRecord(
        timestamp=datetime.now(),
        provider="openai",
        model="gpt-4",
        template="race_preview",
        input_tokens=100,
        output_tokens=200,
        total_tokens=300,
        estimated_cost=0.015,
        request_duration=1.5,
        success=True,
        user_id="test_user",
    )


class TestUsageRecord:
    """Test UsageRecord dataclass."""

    def test_usage_record_creation(self, sample_record):
        """Test creating usage record."""
        assert sample_record.provider == "openai"
        assert sample_record.model == "gpt-4"
        assert sample_record.input_tokens == 100
        assert sample_record.output_tokens == 200
        assert sample_record.total_tokens == 300
        assert sample_record.estimated_cost == 0.015
        assert sample_record.success is True

    def test_usage_record_optional_fields(self):
        """Test usage record with minimal fields."""
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="local",
            model="llama3.1",
            template=None,
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            estimated_cost=0.0,
            request_duration=2.0,
            success=True,
        )
        assert record.template is None
        assert record.user_id is None


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_tracker_initialization(self, tracker, temp_db):
        """Test tracker initializes correctly."""
        assert tracker.db_path == temp_db
        assert tracker.daily_budget == 10.0
        assert tracker.monthly_budget == 200.0
        assert tracker.alert_threshold == 0.8
        assert temp_db.exists()

    def test_database_schema(self, tracker):
        """Test database schema is created correctly."""
        conn = sqlite3.connect(str(tracker.db_path))
        cursor = conn.cursor()

        # Check table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_usage'"
        )
        assert cursor.fetchone() is not None

        # Check columns
        cursor.execute("PRAGMA table_info(llm_usage)")
        columns = [col[1] for col in cursor.fetchall()]
        expected_columns = [
            "id", "timestamp", "provider", "model", "template",
            "input_tokens", "output_tokens", "total_tokens",
            "estimated_cost", "request_duration", "success", "user_id"
        ]
        for col in expected_columns:
            assert col in columns

        conn.close()

    def test_record_usage_success(self, tracker, sample_record):
        """Test recording successful usage."""
        tracker.record_usage(sample_record)

        # Verify record was stored
        conn = sqlite3.connect(str(tracker.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM llm_usage")
        count = cursor.fetchone()[0]
        assert count == 1

        # Verify data
        cursor.execute("SELECT provider, model, estimated_cost FROM llm_usage")
        row = cursor.fetchone()
        assert row[0] == "openai"
        assert row[1] == "gpt-4"
        assert row[2] == 0.015

        conn.close()

    def test_record_multiple_usages(self, tracker):
        """Test recording multiple usage records."""
        records = [
            UsageRecord(
                timestamp=datetime.now(),
                provider="openai",
                model="gpt-4",
                template=None,
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                estimated_cost=0.015,
                request_duration=1.5,
                success=True,
            ),
            UsageRecord(
                timestamp=datetime.now(),
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                template=None,
                input_tokens=150,
                output_tokens=300,
                total_tokens=450,
                estimated_cost=0.009,
                request_duration=2.0,
                success=True,
            ),
        ]

        for record in records:
            tracker.record_usage(record)

        conn = sqlite3.connect(str(tracker.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM llm_usage")
        count = cursor.fetchone()[0]
        assert count == 2
        conn.close()

    def test_get_daily_cost(self, tracker):
        """Test getting daily cost."""
        # Record usage for today
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            template=None,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            estimated_cost=0.5,
            request_duration=1.5,
            success=True,
        )
        tracker.record_usage(record)

        daily_cost = tracker.get_daily_cost()
        assert daily_cost == 0.5

    def test_get_monthly_cost(self, tracker):
        """Test getting monthly cost."""
        # Record multiple usages
        for _ in range(3):
            record = UsageRecord(
                timestamp=datetime.now(),
                provider="openai",
                model="gpt-4",
                template=None,
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                estimated_cost=1.0,
                request_duration=1.5,
                success=True,
            )
            tracker.record_usage(record)

        monthly_cost = tracker.get_monthly_cost()
        assert monthly_cost == 3.0

    def test_daily_budget_exceeded(self, tracker):
        """Test daily budget enforcement."""
        # Record usage that exceeds daily budget
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            template=None,
            input_tokens=1000,
            output_tokens=2000,
            total_tokens=3000,
            estimated_cost=11.0,  # Exceeds 10.0 daily budget
            request_duration=1.5,
            success=True,
        )

        with pytest.raises(BudgetExceededError, match="Daily budget exceeded"):
            tracker.record_usage(record)

    def test_monthly_budget_exceeded(self, tracker):
        """Test monthly budget enforcement."""
        # First, add some usage
        for _ in range(19):
            record = UsageRecord(
                timestamp=datetime.now(),
                provider="openai",
                model="gpt-4",
                template=None,
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                estimated_cost=10.0,
                request_duration=1.5,
                success=True,
            )
            tracker.record_usage(record)

        # Now try to add one that exceeds monthly budget
        record = UsageRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            template=None,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            estimated_cost=11.0,  # Would exceed 200.0 monthly budget
            request_duration=1.5,
            success=True,
        )

        with pytest.raises(BudgetExceededError, match="Monthly budget exceeded"):
            tracker.record_usage(record)

    def test_get_usage_stats(self, tracker):
        """Test usage statistics generation."""
        # Record various usages
        providers = ["openai", "anthropic", "local"]
        for i, provider in enumerate(providers):
            record = UsageRecord(
                timestamp=datetime.now(),
                provider=provider,
                model="test-model",
                template=None,
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                estimated_cost=0.5 * (i + 1),
                request_duration=1.5,
                success=True,
            )
            tracker.record_usage(record)

        stats = tracker.get_usage_stats(days=7)

        assert stats["period_days"] == 7
        assert stats["total_cost"] == 3.0  # 0.5 + 1.0 + 1.5
        assert stats["request_count"] == 3
        assert stats["total_tokens"] == 900  # 300 * 3
        assert stats["success_rate"] == 100.0
        assert stats["avg_cost_per_request"] == 1.0

    def test_get_usage_stats_by_provider(self, tracker):
        """Test usage statistics breakdown by provider."""
        # Record usages for different providers
        for _ in range(2):
            tracker.record_usage(UsageRecord(
                timestamp=datetime.now(),
                provider="openai",
                model="gpt-4",
                template=None,
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                estimated_cost=1.0,
                request_duration=1.5,
                success=True,
            ))

        tracker.record_usage(UsageRecord(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            template=None,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            estimated_cost=0.5,
            request_duration=2.0,
            success=True,
        ))

        stats = tracker.get_usage_stats(days=7)
        cost_by_provider = stats["cost_by_provider"]

        assert cost_by_provider["openai"] == 2.0
        assert cost_by_provider["anthropic"] == 0.5

    def test_get_usage_stats_with_failures(self, tracker):
        """Test usage statistics with failed requests."""
        # Success
        tracker.record_usage(UsageRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            template=None,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            estimated_cost=1.0,
            request_duration=1.5,
            success=True,
        ))

        # Failure
        tracker.record_usage(UsageRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            template=None,
            input_tokens=100,
            output_tokens=0,
            total_tokens=100,
            estimated_cost=0.1,
            request_duration=0.5,
            success=False,
        ))

        stats = tracker.get_usage_stats(days=7)
        assert stats["success_rate"] == 50.0  # 1 success, 1 failure

    def test_cost_tracking_over_time(self, tracker):
        """Test cost tracking across different time periods."""
        # Record old usage (outside current month)
        old_record = UsageRecord(
            timestamp=datetime.now() - timedelta(days=60),
            provider="openai",
            model="gpt-4",
            template=None,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            estimated_cost=5.0,
            request_duration=1.5,
            success=True,
        )

        # Manually insert to bypass budget check
        conn = sqlite3.connect(str(tracker.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO llm_usage (
                timestamp, provider, model, template,
                input_tokens, output_tokens, total_tokens,
                estimated_cost, request_duration, success, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                old_record.timestamp,
                old_record.provider,
                old_record.model,
                old_record.template,
                old_record.input_tokens,
                old_record.output_tokens,
                old_record.total_tokens,
                old_record.estimated_cost,
                old_record.request_duration,
                old_record.success,
                old_record.user_id,
            ),
        )
        conn.commit()
        conn.close()

        # Record recent usage
        recent_record = UsageRecord(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            template=None,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            estimated_cost=2.0,
            request_duration=1.5,
            success=True,
        )
        tracker.record_usage(recent_record)

        # Monthly cost should only include recent
        monthly_cost = tracker.get_monthly_cost()
        assert monthly_cost == 2.0

    def test_default_database_location(self, tmp_path, monkeypatch):
        """Test default database location."""
        # This test ensures the default path is created correctly
        tracker = CostTracker()
        assert tracker.db_path.exists()
        assert tracker.db_path.name == "llm_usage.db"
