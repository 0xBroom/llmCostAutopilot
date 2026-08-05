"""An in-memory verification queue with real lease semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID

from autopilot.domain.models import VerificationJob
from tests.fakes.clock import FrozenClock


@dataclass
class InMemoryVerificationQueue:
    """Leases, does not delete on read.

    The lease is the whole point. A queue fake that pops on dequeue makes it
    impossible to test the case the real system actually has to survive — a
    worker dying mid-job — and would let a redelivery bug ship green.
    """

    clock: FrozenClock = field(default_factory=FrozenClock)
    pending: list[VerificationJob] = field(default_factory=list)
    leased: dict[UUID, datetime] = field(default_factory=dict)
    completed: list[UUID] = field(default_factory=list)
    failed: dict[UUID, str] = field(default_factory=dict)

    async def enqueue(self, job: VerificationJob) -> None:
        self.pending.append(job)

    async def lease(self, *, limit: int, lease_seconds: int) -> Sequence[VerificationJob]:
        now = self.clock.now()
        expiry = now + timedelta(seconds=lease_seconds)
        available = [
            j
            for j in self.pending
            if j.job_id not in self.leased or self.leased[j.job_id] <= now  # lease expired
        ]
        taken = available[:limit]
        for job in taken:
            self.leased[job.job_id] = expiry
        return taken

    async def complete(self, job_id: UUID) -> None:
        self.pending = [j for j in self.pending if j.job_id != job_id]
        self.leased.pop(job_id, None)
        self.completed.append(job_id)

    async def fail(self, job_id: UUID, error: str) -> None:
        self.leased.pop(job_id, None)  # release for redelivery
        self.failed[job_id] = error

    # --- assertion helpers ----------------------------------------------------

    @property
    def depth(self) -> int:
        return len(self.pending)
