"""A clock you can control."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


class FrozenClock:
    """Time that only moves when a test moves it.

    Every sampling window, lease expiry and cost-per-day rollup in this system
    is a function of time. Testing those against the wall clock means either
    sleeping or accepting flakiness, and both are worse than injecting a port.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        if self._now.tzinfo is None:
            raise ValueError("FrozenClock must start from a timezone-aware datetime")

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float = 0.0, **kwargs: float) -> datetime:
        self._now += timedelta(seconds=seconds, **kwargs)
        return self._now
