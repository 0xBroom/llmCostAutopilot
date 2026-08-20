"""The SQLite `RequestStore` adapter, exercised against a real file database.

A file, not `:memory:`, on purpose: WAL and the busy-timeout only mean anything
with a file on disk and more than one connection, and the concurrency guarantee
is the whole reason those pragmas exist.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from autopilot.domain.models import (
    CostBreakdown,
    RequestRecord,
    TokenUsage,
)
from autopilot.infrastructure.persistence import SqliteRequestStore, create_engine, metadata
from autopilot.infrastructure.persistence.schema import requests
from tests.factories import CHEAP, EXPENSIVE, LOCAL, make_decision, make_record, make_response

pytestmark = pytest.mark.integration


async def _make_store(db_path: Path) -> tuple[SqliteRequestStore, AsyncEngine]:
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    return SqliteRequestStore(engine), engine


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteRequestStore]:
    adapter, engine = await _make_store(tmp_path / "test.db")
    try:
        yield adapter
    finally:
        await engine.dispose()


async def _row_count(engine: AsyncEngine) -> int:
    async with engine.connect() as conn:
        result = await conn.execute(select(func.count()).select_from(requests))
        return int(result.scalar_one())


# --- round-trip fidelity ------------------------------------------------------


async def test_save_then_get_roundtrips_a_served_record(store: SqliteRequestStore) -> None:
    record = make_record(chosen=LOCAL, baseline=EXPENSIVE)
    await store.save(record)

    fetched = await store.get(record.request_id)

    assert fetched == record


async def test_get_missing_returns_none(store: SqliteRequestStore) -> None:
    assert await store.get(uuid4()) is None


async def test_error_record_roundtrips(store: SqliteRequestStore) -> None:
    request_id = uuid4()
    when = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    record = RequestRecord(
        request_id=request_id,
        received_at=when,
        decision=make_decision(request_id=request_id, at=when),
        response=None,
        baseline_cost=None,
        error="provider timed out",
    )
    await store.save(record)

    fetched = await store.get(request_id)

    assert fetched == record
    assert fetched is not None
    assert fetched.response is None
    assert fetched.error == "provider timed out"


# --- upsert semantics ---------------------------------------------------------


async def test_save_is_upsert_last_write_wins(store: SqliteRequestStore) -> None:
    request_id = uuid4()
    when = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    first = RequestRecord(
        request_id=request_id,
        received_at=when,
        decision=make_decision(request_id=request_id, chosen=LOCAL),
        response=make_response(model=LOCAL, content="first"),
        baseline_cost=CostBreakdown.compute(TokenUsage(100, 50), EXPENSIVE),
    )
    second = RequestRecord(
        request_id=request_id,
        received_at=when,
        decision=make_decision(request_id=request_id, chosen=LOCAL),
        response=make_response(model=LOCAL, content="second"),
        baseline_cost=CostBreakdown.compute(TokenUsage(100, 50), EXPENSIVE),
    )

    await store.save(first)
    await store.save(second)

    fetched = await store.get(request_id)
    assert fetched is not None
    assert fetched.response is not None
    assert fetched.response.content == "second"
    assert await _row_count(store._engine) == 1  # asserting no duplicate row


# --- list_since ordering, filtering, limit ------------------------------------


async def test_list_since_orders_by_time_and_respects_limit(store: SqliteRequestStore) -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    old = make_record(at=base - timedelta(hours=2))
    mid = make_record(at=base)
    new = make_record(at=base + timedelta(hours=2))
    for record in (new, old, mid):  # inserted out of order on purpose
        await store.save(record)

    since = base - timedelta(minutes=30)
    listed = await store.list_since(since, limit=10)

    # `old` is before `since` and must be filtered out; the rest sorted ascending.
    assert [r.request_id for r in listed] == [mid.request_id, new.request_id]

    limited = await store.list_since(base - timedelta(days=1), limit=1)
    assert [r.request_id for r in limited] == [old.request_id]


async def test_list_since_orders_by_instant_across_timezones(store: SqliteRequestStore) -> None:
    # 12:00-05:00 is 17:00 UTC — chronologically AFTER 13:00 UTC, even though its
    # raw ISO string sorts earlier. Correct ordering proves UTC normalisation.
    earlier_utc = make_record(at=datetime(2026, 1, 1, 13, 0, tzinfo=UTC))
    later_but_string_smaller = make_record(
        at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=-5)))
    )
    await store.save(later_but_string_smaller)
    await store.save(earlier_utc)

    listed = await store.list_since(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), limit=10)

    assert [r.request_id for r in listed] == [
        earlier_utc.request_id,
        later_but_string_smaller.request_id,
    ]


# --- money handling -----------------------------------------------------------


async def test_money_survives_as_decimal_not_float(store: SqliteRequestStore) -> None:
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
    record = make_record(chosen=CHEAP, baseline=EXPENSIVE, usage=usage)
    assert record.response is not None
    await store.save(record)

    fetched = await store.get(record.request_id)
    assert fetched is not None
    assert fetched.response is not None

    cost = fetched.response.cost
    assert isinstance(cost.total, Decimal)
    assert isinstance(cost.prompt_cost, Decimal)
    assert cost.prompt_cost == record.response.cost.prompt_cost  # exact, not approx
    assert cost.total == record.response.cost.total


async def test_baseline_cost_is_frozen_against_a_later_price_change(
    store: SqliteRequestStore,
) -> None:
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
    record = make_record(chosen=LOCAL, baseline=EXPENSIVE, usage=usage)
    assert record.baseline_cost is not None
    frozen_baseline_total = record.baseline_cost.total
    await store.save(record)

    # Simulate the catalog re-pricing the baseline model AFTER the row was written.
    repriced = replace(EXPENSIVE, input_cost_per_token=EXPENSIVE.input_cost_per_token * 2)
    recomputed = CostBreakdown.compute(usage, repriced).total

    fetched = await store.get(record.request_id)
    assert fetched is not None
    assert fetched.baseline_cost is not None
    # The stored counterfactual is what was true at write time, not the new price.
    assert fetched.baseline_cost.total == frozen_baseline_total
    assert fetched.baseline_cost.total != recomputed


# --- pragmas and concurrency --------------------------------------------------


async def test_wal_and_busy_timeout_are_configured(store: SqliteRequestStore) -> None:
    async with store._engine.connect() as conn:  # inspecting pragmas
        journal_mode = (await conn.exec_driver_sql("PRAGMA journal_mode")).scalar_one()
        busy_timeout = (await conn.exec_driver_sql("PRAGMA busy_timeout")).scalar_one()

    assert str(journal_mode).lower() == "wal"
    assert int(busy_timeout) == 5000


async def test_concurrent_writers_both_land(tmp_path: Path) -> None:
    # Two independent engines against the same file, writing at the same time:
    # WAL + busy_timeout is what keeps this from raising "database is locked".
    db_path = tmp_path / "concurrent.db"
    writer_a, engine_a = await _make_store(db_path)
    _, engine_b = await _make_store(db_path)  # schema already exists; create_all is a no-op
    writer_b = SqliteRequestStore(engine_b)
    try:
        records = [make_record() for _ in range(20)]
        await asyncio.gather(
            *(writer_a.save(r) for r in records[::2]),
            *(writer_b.save(r) for r in records[1::2]),
        )
        assert await _row_count(engine_a) == len(records)
    finally:
        await engine_a.dispose()
        await engine_b.dispose()


# --- write-path performance (measured and reported, NOT CI-gated) -------------


async def test_write_path_latency_is_reported(store: SqliteRequestStore) -> None:
    # The #20 budget is p95 < 15 ms, but a shared CI runner's wall-clock is too
    # noisy to gate on (see #5). So this measures and prints the number, and only
    # asserts a loose sanity ceiling that catches a real regression (e.g. an
    # accidental fsync-per-row), not the tight budget.
    samples: list[float] = []
    for _ in range(50):
        record = make_record()
        start = time.perf_counter()
        await store.save(record)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    p95 = samples[int(0.95 * (len(samples) - 1))]
    print(f"\nwrite-path latency: p50={samples[len(samples) // 2]:.2f}ms p95={p95:.2f}ms")

    assert p95 < 200.0  # sanity ceiling, not the 15ms budget
