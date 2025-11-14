"""Rate limiter for LLM API calls with cost tracking (H-3).

Implements token bucket algorithm with:
- Per-minute request limits
- Per-hour cost limits
- Sliding window tracking
"""

import time
from collections import deque
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class LLMRateLimiter:
    """Rate limiter for LLM API calls with cost tracking.

    Implements token bucket algorithm with:
    - Per-minute request limits
    - Per-hour cost limits
    - Sliding window tracking
    """

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_cost_per_hour: float = 10.0,
        cost_per_request: float = 0.01
    ):
        """Initialize rate limiter with configurable limits.

        Args:
            max_requests_per_minute: Maximum API requests per minute
            max_cost_per_hour: Maximum cost per hour in USD
            cost_per_request: Default cost per request in USD
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_cost_per_hour = max_cost_per_hour
        self.cost_per_request = cost_per_request

        # Sliding windows for tracking
        # Using deque with maxlen to automatically drop old entries
        self.request_times: deque = deque(maxlen=1000)
        self.hourly_costs: deque = deque(maxlen=1000)

        logger.info(
            "rate_limiter_initialized",
            max_requests_per_minute=max_requests_per_minute,
            max_cost_per_hour=max_cost_per_hour
        )

    def check_rate_limit(self) -> tuple[bool, Optional[str]]:
        """Check if request is within rate limits.

        Returns:
            (allowed, reason) - (True, None) if allowed, (False, reason) if denied
        """
        now = time.time()

        # Check per-minute request limit
        minute_ago = now - 60
        recent_requests = sum(
            1 for t in self.request_times if t > minute_ago
        )

        if recent_requests >= self.max_requests_per_minute:
            reason = (
                f"Rate limit exceeded: {recent_requests}/{self.max_requests_per_minute} "
                f"requests in the last minute"
            )
            logger.warning(
                "rate_limit_exceeded_requests",
                recent_requests=recent_requests,
                max_requests=self.max_requests_per_minute
            )
            return False, reason

        # Check per-hour cost limit
        hour_ago = now - 3600
        recent_cost = sum(
            cost for timestamp, cost in self.hourly_costs
            if timestamp > hour_ago
        )

        if recent_cost + self.cost_per_request > self.max_cost_per_hour:
            reason = (
                f"Cost limit exceeded: ${recent_cost:.4f}/${self.max_cost_per_hour:.2f} "
                f"in the last hour"
            )
            logger.warning(
                "rate_limit_exceeded_cost",
                recent_cost=recent_cost,
                max_cost=self.max_cost_per_hour
            )
            return False, reason

        return True, None

    def record_request(self, cost: Optional[float] = None) -> None:
        """Record successful API request.

        Args:
            cost: Actual cost of the request (uses default if not provided)
        """
        now = time.time()
        # Use 'if cost is not None' to properly handle cost=0.0
        actual_cost = cost if cost is not None else self.cost_per_request

        self.request_times.append(now)
        self.hourly_costs.append((now, actual_cost))

        logger.debug(
            "llm_request_recorded",
            cost=actual_cost,
            requests_last_minute=len([t for t in self.request_times if t > now - 60]),
            cost_last_hour=sum(c for t, c in self.hourly_costs if t > now - 3600)
        )

    def get_stats(self) -> dict:
        """Get current rate limiter statistics.

        Returns:
            Dictionary with current usage stats
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        recent_requests = sum(1 for t in self.request_times if t > minute_ago)
        recent_cost = sum(c for t, c in self.hourly_costs if t > hour_ago)

        return {
            "requests_last_minute": recent_requests,
            "max_requests_per_minute": self.max_requests_per_minute,
            "requests_remaining": max(0, self.max_requests_per_minute - recent_requests),
            "cost_last_hour": recent_cost,
            "max_cost_per_hour": self.max_cost_per_hour,
            "cost_remaining": max(0, self.max_cost_per_hour - recent_cost),
            "utilization_percent": (recent_requests / self.max_requests_per_minute) * 100
        }

    def reset(self) -> None:
        """Reset rate limiter (primarily for testing)."""
        self.request_times.clear()
        self.hourly_costs.clear()
        logger.info("rate_limiter_reset")
