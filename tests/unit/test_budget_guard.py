"""The daily spend guard: `RequestStore` + `Clock` in, a refusal or nothing out.

Every scenario here is a `datetime` scenario in disguise — daily spend is
entirely about which requests fall inside "today". `FrozenClock` (Phase 0) is
the only source of "now" the guard is allowed to consult; a test that reads
`datetime.now()` anywhere in this file would depend on when it happens to run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from autopilot.application.budget import DailySpendGuard
from autopilot.domain.errors import BudgetExceededError
from autopilot.domain.models import ComplexityTier, ModelConfig, PriceSource, TokenUsage
from tests.factories import make_record
from tests.fakes import FrozenClock, InMemoryRequestStore

# One cent per prompt token, zero output cost, so an exact dollar-and-cents
# spend is reachable with an exact integer token count — no rounding
# ambiguity between the amount a test asks for and the amount CostBreakdown
# actually computes.
_PRICED_MODEL = ModelConfig(
    key="priced-for-tests",
    provider="ollama",
    provider_model_id="ollama/priced-for-tests",
    input_cost_per_token=Decimal("0.01"),
    output_cost_per_token=Decimal("0"),
    max_context_tokens=8192,
    quality_tier=ComplexityTier.SIMPLE,
    price_source=PriceSource.PRICE_MAP,
)


def _guard(
    *,
    store: InMemoryRequestStore,
    clock: FrozenClock,
    daily_budget_usd: Decimal,
    max_rows: int = 100_000,
) -> DailySpendGuard:
    return DailySpendGuard(
        store=store, clock=clock, daily_budget_usd=daily_budget_usd, max_rows=max_rows
    )


async def _seed(store: InMemoryRequestStore, *, spend: Decimal, at: datetime) -> None:
    """Record a single request whose total cost is exactly `spend`."""
    prompt_tokens = int(spend / _PRICED_MODEL.input_cost_per_token)
    usage = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=0)
    record = make_record(chosen=_PRICED_MODEL, usage=usage, at=at)
    assert record.response is not None
    assert record.response.cost.total == spend
    await store.save(record)


async def test_under_budget_check_does_not_raise() -> None:
    store = InMemoryRequestStore()
    clock = FrozenClock(datetime(2026, 8, 7, 23, 59, 0, tzinfo=UTC))
    await _seed(store, spend=Decimal("42.13"), at=datetime(2026, 8, 7, 1, 0, tzinfo=UTC))
    guard = _guard(store=store, clock=clock, daily_budget_usd=Decimal("50.00"))

    await guard.check(projected_cost=Decimal("0"))  # must not raise


async def test_at_budget_exactly_raises_naming_both_values() -> None:
    store = InMemoryRequestStore()
    clock = FrozenClock(datetime(2026, 8, 7, 23, 59, 0, tzinfo=UTC))
    await _seed(store, spend=Decimal("50.00"), at=datetime(2026, 8, 7, 1, 0, tzinfo=UTC))
    guard = _guard(store=store, clock=clock, daily_budget_usd=Decimal("50.00"))

    with pytest.raises(BudgetExceededError, match=r"50\.00.*50\.00|50\.00"):
        await guard.check(projected_cost=Decimal("0"))


async def test_projected_cost_that_would_tip_it_over_raises() -> None:
    store = InMemoryRequestStore()
    clock = FrozenClock(datetime(2026, 8, 7, 23, 59, 0, tzinfo=UTC))
    await _seed(store, spend=Decimal("49.00"), at=datetime(2026, 8, 7, 1, 0, tzinfo=UTC))
    guard = _guard(store=store, clock=clock, daily_budget_usd=Decimal("50.00"))

    with pytest.raises(BudgetExceededError):
        await guard.check(projected_cost=Decimal("1.50"))


async def test_guard_uses_the_injected_clock_never_wall_clock() -> None:
    """Fixed at 23:59 on 2026-08-07. A record from the previous UTC day must
    not count toward today's spend, even one microsecond before midnight."""
    store = InMemoryRequestStore()
    clock = FrozenClock(datetime(2026, 8, 7, 23, 59, 0, tzinfo=UTC))
    await _seed(
        store,
        spend=Decimal("999.00"),
        at=datetime(2026, 8, 6, 23, 59, 59, tzinfo=UTC),
    )
    await _seed(store, spend=Decimal("10.00"), at=datetime(2026, 8, 7, 0, 0, 0, tzinfo=UTC))
    guard = _guard(store=store, clock=clock, daily_budget_usd=Decimal("50.00"))

    assert await guard.spend_today() == Decimal("10.00")


async def test_remaining_is_budget_minus_spend_today() -> None:
    store = InMemoryRequestStore()
    clock = FrozenClock(datetime(2026, 8, 7, 12, 0, 0, tzinfo=UTC))
    await _seed(store, spend=Decimal("30.00"), at=datetime(2026, 8, 7, 1, 0, tzinfo=UTC))
    guard = _guard(store=store, clock=clock, daily_budget_usd=Decimal("50.00"))

    assert await guard.remaining() == Decimal("20.00")


async def test_truncated_page_fails_closed() -> None:
    """`list_since` returning exactly `max_rows` rows means the true total
    could be larger than what was summed. A guard that returned the summed
    total anyway would silently let a >max_rows-request day undercount its
    own spend and keep approving requests it shouldn't."""
    store = InMemoryRequestStore()
    clock = FrozenClock(datetime(2026, 8, 7, 12, 0, 0, tzinfo=UTC))
    for _ in range(3):
        await _seed(store, spend=Decimal("1.00"), at=datetime(2026, 8, 7, 1, 0, tzinfo=UTC))
    guard = _guard(store=store, clock=clock, daily_budget_usd=Decimal("50.00"), max_rows=3)

    with pytest.raises(BudgetExceededError, match="truncat"):
        await guard.spend_today()
