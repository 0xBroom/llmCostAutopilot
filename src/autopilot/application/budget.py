"""The daily spend guard.

Ships "assembled but not wired" — nothing calls this yet. Phase 2's request
pipeline is the named wiring point (consulting `check()` before a request is
served). Independently unit-testable and independently `make check`-green in
the meantime, matching every other slice in this change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal

from autopilot.application.ports import Clock, RequestStore
from autopilot.domain.errors import BudgetExceededError


@dataclass(frozen=True, slots=True)
class DailySpendGuard:
    """Refuses a request whose cost would meet or exceed the daily cap.

    Takes `daily_budget_usd: Decimal` directly, never `Settings` — the
    `config-is-a-leaf` import-linter contract forbids `application` from
    importing `autopilot.config`, and this guard has no use for anything else
    `Settings` carries.
    """

    store: RequestStore
    clock: Clock
    daily_budget_usd: Decimal
    max_rows: int = 100_000

    async def spend_today(self) -> Decimal:
        """Sum of every served request's cost since the start of today, UTC.

        Fails closed rather than undercounting: if `list_since` hands back
        exactly `max_rows` rows, the page may have been truncated and the
        true total may be larger than what was summed — raising here is
        safer than approving more spend on a number known to be unreliable.
        """
        since = self.clock.now().astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        records = await self.store.list_since(since, limit=self.max_rows)
        if len(records) >= self.max_rows:
            raise BudgetExceededError(
                f"spend window truncated at {self.max_rows} rows since {since.isoformat()}; "
                "refusing to compute a total that may be an undercount"
            )
        total = Decimal(0)
        for record in records:
            if record.response is not None:
                total += record.response.cost.total
        return total

    async def remaining(self) -> Decimal:
        """Budget left for today. May be negative if already over."""
        return self.daily_budget_usd - await self.spend_today()

    async def check(self, *, projected_cost: Decimal) -> None:
        """Raise if today's spend plus `projected_cost` would meet or exceed
        the daily cap. Never raises on a strictly-under total."""
        spend = await self.spend_today()
        projected_total = spend + projected_cost
        if projected_total >= self.daily_budget_usd:
            raise BudgetExceededError(
                f"today's spend ({spend}) plus this request's projected cost "
                f"({projected_cost}) = {projected_total}, which would meet or exceed "
                f"the daily budget ({self.daily_budget_usd})"
            )
