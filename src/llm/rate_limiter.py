import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RateLimitConfig:
    requests_per_minute: int
    requests_per_hour: int
    max_daily_tokens: int
    alert_threshold: float


class QuotaStorage(ABC):
    """Abstract base class for quota storage implementations"""

    @abstractmethod
    async def get_daily_usage(self, date: str) -> int:
        """Get token usage for specified date"""
        pass

    @abstractmethod
    async def update_token_usage(self, date: str, tokens: int):
        """Update token usage for specified date"""
        pass

    @abstractmethod
    async def cleanup_old_records(self, before_date: str):
        """Clean up old usage records"""
        pass


class SQLiteQuotaStorage(QuotaStorage):
    def __init__(self, db_path: str = "rate_limiter.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    date TEXT PRIMARY KEY,
                    tokens_used INTEGER
                )
            """
            )

    async def get_daily_usage(self, date: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("SELECT tokens_used FROM token_usage WHERE date = ?", (date,)).fetchone()
            return result[0] if result else 0

    async def update_token_usage(self, date: str, tokens: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO token_usage (date, tokens_used)
                VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET
                tokens_used = tokens_used + ?
            """,
                (date, tokens, tokens),
            )
            conn.commit()

    async def cleanup_old_records(self, before_date: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM token_usage WHERE date < ?", (before_date,))
            conn.commit()


class DjangoQuotaStorage(QuotaStorage):
    """Example Django model-based storage"""

    def __init__(self, token_usage_model):
        self.model = token_usage_model

    async def get_daily_usage(self, date: str) -> int:
        try:
            usage = await self.model.objects.filter(date=date).afirst()
            return usage.tokens_used if usage else 0
        except Exception:
            # Log error and return 0 as fallback
            return 0

    async def update_token_usage(self, date: str, tokens: int):
        await self.model.objects.update_or_create(date=date, defaults={"tokens_used": tokens})

    async def cleanup_old_records(self, before_date: str):
        await self.model.objects.filter(date__lt=before_date).delete()


class RateLimiter:
    def __init__(self, config: RateLimitConfig, storage: Optional[QuotaStorage] = None):
        self.config = config
        self.storage = storage or SQLiteQuotaStorage()  # Default to SQLite
        self._minute_bucket = self.config.requests_per_minute
        self._hour_bucket = self.config.requests_per_hour
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def _refill_buckets(self):
        """Refill rate limit buckets based on elapsed time"""
        now = time.time()
        elapsed = now - self._last_refill

        # Refill minute bucket
        minute_tokens = int(elapsed * (self.config.requests_per_minute / 60))
        self._minute_bucket = min(self._minute_bucket + minute_tokens, self.config.requests_per_minute)

        # Refill hour bucket
        hour_tokens = int(elapsed * (self.config.requests_per_hour / 3600))
        self._hour_bucket = min(self._hour_bucket + hour_tokens, self.config.requests_per_hour)

        self._last_refill = now

    async def check_and_update(self, tokens: int) -> tuple[bool, Optional[str]]:
        """Check if the request can proceed and update counters"""
        today = datetime.now().strftime("%Y-%m-%d")

        with self._lock:
            self._refill_buckets()

            # Check rate limits
            if self._minute_bucket < 1:
                return False, "Rate limit exceeded: Too many requests per minute"
            if self._hour_bucket < 1:
                return False, "Rate limit exceeded: Too many requests per hour"

            # Check daily token quota
            daily_usage = await self.storage.get_daily_usage(today)
            if daily_usage + tokens > self.config.max_daily_tokens:
                return False, "Daily token quota exceeded"

            # Update counters if all checks pass
            self._minute_bucket -= 1
            self._hour_bucket -= 1
            await self.storage.update_token_usage(today, tokens)

            return True, None

    async def get_quota_status(self) -> dict:
        """Get current quota and rate limit status"""
        today = datetime.now().strftime("%Y-%m-%d")

        with self._lock:
            self._refill_buckets()
            daily_usage = await self.storage.get_daily_usage(today)

            return {
                "minute_requests_remaining": self._minute_bucket,
                "hour_requests_remaining": self._hour_bucket,
                "daily_tokens_used": daily_usage,
                "daily_tokens_remaining": self.config.max_daily_tokens - daily_usage,
                "quota_used_percentage": (daily_usage / self.config.max_daily_tokens) * 100,
            }
