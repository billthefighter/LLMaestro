import threading
import time
from datetime import datetime, date, timedelta
from typing import Dict, Optional

from pydantic import ConfigDict, Field

from llmaestro.config.base import RateLimitConfig
from llmaestro.core.persistence import PersistentModel


class TokenBucket(PersistentModel):
    """Token bucket implementation for rate limiting and quota tracking.

    This model maintains token usage data and provides methods for checking
    and updating quotas across different time periods.
    """

    # Configuration
    rate_limit_config: RateLimitConfig = Field(description="Rate limiting configuration")

    # Usage tracking
    daily_usage: Dict[date, int] = Field(default_factory=dict, description="Daily token usage mapped by date")

    # Bucket state
    minute_tokens: int = Field(default=0, description="Current number of tokens in the minute bucket")
    last_refill_timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp(), description="Last time the buckets were refilled"
    )

    model_config = ConfigDict(validate_assignment=True)

    async def initialize(self) -> None:
        """Initialize the token bucket.

        This method sets up initial state and cleans up any old usage data.
        """
        # Reset minute tokens to max
        self.minute_tokens = self.rate_limit_config.requests_per_minute

        # Update timestamp
        self.last_refill_timestamp = datetime.now().timestamp()

        # Clean up old records
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=30
        )  # Keep last 30 days

        self.daily_usage = {
            date_key: usage for date_key, usage in self.daily_usage.items() if date_key >= cutoff.date()
        }

    def _get_date_key(self, dt: datetime) -> date:
        """Convert datetime to date key for storage.

        Args:
            dt: Datetime to convert

        Returns:
            Date object for storage key
        """
        return dt.date()

    async def get_daily_usage(self, dt: datetime) -> int:
        """Get token usage for specified date.

        Args:
            dt: Datetime to get usage for

        Returns:
            Number of tokens used on that date
        """
        date_key = self._get_date_key(dt)
        return self.daily_usage.get(date_key, 0)

    async def update_token_usage(self, dt: datetime, tokens: int) -> None:
        """Update token usage for specified date.

        Args:
            dt: Datetime to update usage for
            tokens: Number of tokens to add to usage
        """
        date_key = self._get_date_key(dt)
        current = self.daily_usage.get(date_key, 0)
        self.daily_usage[date_key] = current + tokens

    async def cleanup_old_records(self, before_dt: datetime) -> None:
        """Clean up usage records before specified date.

        Args:
            before_dt: Datetime cutoff for cleanup
        """
        cutoff_date = self._get_date_key(before_dt)
        self.daily_usage = {date_key: usage for date_key, usage in self.daily_usage.items() if date_key >= cutoff_date}

    async def check_quota(self, dt: datetime, tokens: int) -> tuple[bool, Optional[str]]:
        """Check if requested tokens are within quota limits.

        Args:
            dt: Datetime to check quota for
            tokens: Number of tokens to check

        Returns:
            Tuple of (allowed, error_message)
        """
        daily_usage = await self.get_daily_usage(dt)
        if daily_usage + tokens > self.rate_limit_config.max_daily_tokens:
            return False, "Daily token quota exceeded"
        return True, None

    def refill_minute_bucket(self, current_time: float) -> None:
        """Refill the minute token bucket based on elapsed time.

        Args:
            current_time: Current timestamp
        """
        elapsed = current_time - self.last_refill_timestamp
        minute_tokens = int(elapsed * (self.rate_limit_config.requests_per_minute / 60))
        self.minute_tokens = min(self.minute_tokens + minute_tokens, self.rate_limit_config.requests_per_minute)
        self.last_refill_timestamp = current_time

    async def get_quota_status(self, dt: datetime) -> dict:
        """Get current quota and rate limit status.

        Args:
            dt: Datetime to get status for

        Returns:
            Dictionary with quota status information
        """
        daily_usage = await self.get_daily_usage(dt)
        return {
            "minute_requests_remaining": self.minute_tokens,
            "daily_tokens_used": daily_usage,
            "daily_tokens_remaining": self.rate_limit_config.max_daily_tokens - daily_usage,
            "quota_used_percentage": (daily_usage / self.rate_limit_config.max_daily_tokens) * 100,
        }


class RateLimiter:
    """Rate limiter implementation using token bucket algorithm."""

    def __init__(self, config: RateLimitConfig, storage: Optional[TokenBucket] = None):
        """Initialize rate limiter.

        Args:
            config: Rate limiting configuration
            storage: Optional token bucket for storage
        """
        self.config = config
        self.storage = storage or TokenBucket(rate_limit_config=config)
        self._lock = threading.Lock()

    async def initialize(self) -> None:
        """Initialize async components of the rate limiter.

        This method should be called after construction to set up any async resources
        like storage backends or cleanup tasks.
        """
        # Clean up old records on initialization
        await self.cleanup_old_records()

        # Initialize storage if needed
        if hasattr(self.storage, "initialize"):
            await self.storage.initialize()

    async def check_and_update(self, tokens: int) -> tuple[bool, Optional[str]]:
        """Check if the request can proceed and update counters.

        Args:
            tokens: Number of tokens to consume

        Returns:
            Tuple of (allowed, error_message)
        """
        now = datetime.now()
        current_time = time.time()

        with self._lock:
            # Refill minute bucket
            self.storage.refill_minute_bucket(current_time)

            # Check minute rate limit
            if self.storage.minute_tokens < 1:
                return False, "Rate limit exceeded: Too many requests per minute"

            # Check daily quota
            quota_ok, error = await self.storage.check_quota(now, tokens)
            if not quota_ok:
                return False, error

            # Update counters
            self.storage.minute_tokens -= 1
            await self.storage.update_token_usage(now, tokens)

            return True, None

    async def get_quota_status(self) -> dict:
        """Get current quota and rate limit status.

        Returns:
            Dictionary with quota status information
        """
        now = datetime.now()

        with self._lock:
            self.storage.refill_minute_bucket(time.time())
            return await self.storage.get_quota_status(now)

    async def cleanup_old_records(self, days_to_keep: int = 30) -> None:
        """Clean up old usage records.

        Args:
            days_to_keep: Number of days of history to retain
        """
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_to_keep)

        with self._lock:
            await self.storage.cleanup_old_records(cutoff)
