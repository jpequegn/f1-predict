"""Cost tracking and monitoring for LLM usage.

This module tracks LLM API costs, enforces budgets, and provides
analytics on usage patterns and spending.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import structlog

from f1_predict.llm.exceptions import BudgetExceededError

logger = structlog.get_logger(__name__)


@dataclass
class UsageRecord:
    """Record of LLM API usage.

    Attributes:
        timestamp: When the request was made
        provider: Provider name (openai, anthropic, local)
        model: Model identifier
        template: Template used (if any)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens used
        estimated_cost: Estimated cost in USD
        request_duration: Request duration in seconds
        success: Whether request succeeded
        user_id: User/session identifier
    """

    timestamp: datetime
    provider: str
    model: str
    template: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    request_duration: float
    success: bool
    user_id: Optional[str] = None


class CostTracker:
    """Track and monitor LLM API costs.

    Maintains a SQLite database of usage records and enforces
    budget limits with configurable alert thresholds.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        daily_budget: float = 10.0,
        monthly_budget: float = 200.0,
        alert_threshold: float = 0.8,
    ):
        """Initialize cost tracker.

        Args:
            db_path: Path to SQLite database file
            daily_budget: Daily budget limit in USD
            monthly_budget: Monthly budget limit in USD
            alert_threshold: Threshold for budget alerts (0.0-1.0)
        """
        if db_path is None:
            # Default to data directory
            db_path = Path(__file__).parents[3] / "data" / "llm_usage.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.alert_threshold = alert_threshold

        self.logger = logger.bind(db_path=str(db_path))

        # Initialize database
        self._init_database()

        self.logger.info(
            "cost_tracker_initialized",
            daily_budget=daily_budget,
            monthly_budget=monthly_budget,
        )

    def _init_database(self):
        """Initialize SQLite database with usage table."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                provider VARCHAR(50) NOT NULL,
                model VARCHAR(100) NOT NULL,
                template VARCHAR(100),
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                estimated_cost REAL NOT NULL,
                request_duration REAL NOT NULL,
                success BOOLEAN NOT NULL,
                user_id VARCHAR(100)
            )
        """)

        # Create indexes for common queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON llm_usage(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_provider ON llm_usage(provider)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_model ON llm_usage(model)"
        )

        conn.commit()
        conn.close()

    def record_usage(self, record: UsageRecord):
        """Record LLM usage.

        Args:
            record: Usage record to store

        Raises:
            BudgetExceededError: If recording would exceed budget
        """
        # Check budget before recording
        self._check_budget(record.estimated_cost)

        conn = sqlite3.connect(str(self.db_path))
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
                record.timestamp,
                record.provider,
                record.model,
                record.template,
                record.input_tokens,
                record.output_tokens,
                record.total_tokens,
                record.estimated_cost,
                record.request_duration,
                record.success,
                record.user_id,
            ),
        )

        conn.commit()
        conn.close()

        self.logger.info(
            "usage_recorded",
            provider=record.provider,
            model=record.model,
            tokens=record.total_tokens,
            cost=record.estimated_cost,
        )

    def _check_budget(self, additional_cost: float):
        """Check if additional cost would exceed budget.

        Args:
            additional_cost: Cost to add

        Raises:
            BudgetExceededError: If budget would be exceeded
        """
        daily_cost = self.get_daily_cost()
        monthly_cost = self.get_monthly_cost()

        # Check daily budget
        if daily_cost + additional_cost > self.daily_budget:
            msg = f"Daily budget exceeded: ${daily_cost + additional_cost:.4f} > ${self.daily_budget}"
            self.logger.error("daily_budget_exceeded", cost=daily_cost + additional_cost)
            raise BudgetExceededError(msg)

        # Check monthly budget
        if monthly_cost + additional_cost > self.monthly_budget:
            msg = f"Monthly budget exceeded: ${monthly_cost + additional_cost:.4f} > ${self.monthly_budget}"
            self.logger.error("monthly_budget_exceeded", cost=monthly_cost + additional_cost)
            raise BudgetExceededError(msg)

        # Alert at threshold
        if daily_cost + additional_cost > self.daily_budget * self.alert_threshold:
            self.logger.warning(
                "daily_budget_threshold_reached",
                cost=daily_cost + additional_cost,
                threshold=self.daily_budget * self.alert_threshold,
            )

        if monthly_cost + additional_cost > self.monthly_budget * self.alert_threshold:
            self.logger.warning(
                "monthly_budget_threshold_reached",
                cost=monthly_cost + additional_cost,
                threshold=self.monthly_budget * self.alert_threshold,
            )

    def get_daily_cost(self) -> float:
        """Get total cost for current day.

        Returns:
            Total cost in USD for today
        """
        today = datetime.now().date()
        return self._get_cost_for_date_range(today, today)

    def get_monthly_cost(self) -> float:
        """Get total cost for current month.

        Returns:
            Total cost in USD for this month
        """
        now = datetime.now()
        start_of_month = datetime(now.year, now.month, 1).date()
        return self._get_cost_for_date_range(start_of_month, now.date())

    def _get_cost_for_date_range(self, start_date, end_date) -> float:
        """Get total cost for a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Total cost in USD
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT SUM(estimated_cost)
            FROM llm_usage
            WHERE DATE(timestamp) BETWEEN ? AND ?
        """,
            (start_date, end_date),
        )

        result = cursor.fetchone()[0]
        conn.close()

        return result if result is not None else 0.0

    def get_usage_stats(self, days: int = 7) -> dict:
        """Get usage statistics for recent period.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with usage statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).date()

        # Total cost
        cursor.execute(
            """
            SELECT SUM(estimated_cost)
            FROM llm_usage
            WHERE DATE(timestamp) >= ?
        """,
            (since_date,),
        )
        total_cost = cursor.fetchone()[0] or 0.0

        # Cost by provider
        cursor.execute(
            """
            SELECT provider, SUM(estimated_cost) as cost
            FROM llm_usage
            WHERE DATE(timestamp) >= ?
            GROUP BY provider
            ORDER BY cost DESC
        """,
            (since_date,),
        )
        cost_by_provider = dict(cursor.fetchall())

        # Total tokens
        cursor.execute(
            """
            SELECT SUM(total_tokens)
            FROM llm_usage
            WHERE DATE(timestamp) >= ?
        """,
            (since_date,),
        )
        total_tokens = cursor.fetchone()[0] or 0

        # Request count
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM llm_usage
            WHERE DATE(timestamp) >= ?
        """,
            (since_date,),
        )
        request_count = cursor.fetchone()[0] or 0

        # Success rate
        cursor.execute(
            """
            SELECT
                SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
            FROM llm_usage
            WHERE DATE(timestamp) >= ?
        """,
            (since_date,),
        )
        success_rate = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "period_days": days,
            "total_cost": total_cost,
            "cost_by_provider": cost_by_provider,
            "total_tokens": total_tokens,
            "request_count": request_count,
            "success_rate": success_rate,
            "avg_cost_per_request": total_cost / request_count if request_count > 0 else 0.0,
        }
