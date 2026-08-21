"""SQLite adapter for the ``RequestStore`` port.

Upserts on ``request_id`` (last-write-wins): a retried save is idempotent rather
than a duplicate row. The in-memory fake documents and mirrors this exactly, so
a test written against the fake behaves the same against this adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from autopilot.domain.models import RequestRecord
from autopilot.infrastructure.persistence.schema import requests
from autopilot.infrastructure.persistence.serialization import record_to_row, row_to_record


class SqliteRequestStore:
    """Durable ``RequestStore`` backed by SQLite over aiosqlite."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save(self, record: RequestRecord) -> None:
        row = record_to_row(record)
        stmt = sqlite_insert(requests).values(**row)
        overwrite = {
            column.name: stmt.excluded[column.name]
            for column in requests.columns
            if column.name != "id"
        }
        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=overwrite)
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def get(self, request_id: UUID) -> RequestRecord | None:
        stmt = select(requests).where(requests.c.id == str(request_id))
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            row = result.mappings().first()
        if row is None:
            return None
        return row_to_record(dict(row))

    async def list_since(self, since: datetime, *, limit: int) -> Sequence[RequestRecord]:
        if since.tzinfo is None:
            raise ValueError("list_since requires a timezone-aware `since`")
        if limit <= 0:
            # SQLite reads LIMIT -1 as "no limit"; a non-positive limit is almost
            # always a caller bug (an off-by-one page size going negative), and
            # silently returning the entire table is the worst possible answer.
            raise ValueError("list_since requires a positive `limit`")
        # Compare against the same UTC-normalised text the rows were stored with,
        # so the range filter and ordering are by instant, not by wall-clock string.
        since_text = since.astimezone(UTC).isoformat()
        stmt = (
            select(requests)
            .where(requests.c.received_at >= since_text)
            .order_by(requests.c.received_at, requests.c.id)
            .limit(limit)
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            rows = result.mappings().all()
        return [row_to_record(dict(row)) for row in rows]
