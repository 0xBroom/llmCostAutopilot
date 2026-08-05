"""An in-memory request store."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from autopilot.domain.models import RequestRecord


@dataclass
class InMemoryRequestStore:
    """A dict with the `RequestStore` shape.

    Deliberately preserves last-write-wins on `request_id` rather than
    appending, because that is what the SQLite adapter's upsert does, and a
    fake that behaves differently from the real thing is worse than no fake.
    """

    records: dict[UUID, RequestRecord] = field(default_factory=dict)

    async def save(self, record: RequestRecord) -> None:
        self.records[record.request_id] = record

    async def get(self, request_id: UUID) -> RequestRecord | None:
        return self.records.get(request_id)

    async def list_since(self, since: datetime, *, limit: int) -> Sequence[RequestRecord]:
        rows = [r for r in self.records.values() if r.received_at >= since]
        rows.sort(key=lambda r: r.received_at)
        return rows[:limit]

    # --- assertion helpers ----------------------------------------------------

    @property
    def saved_count(self) -> int:
        return len(self.records)
