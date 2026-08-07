"""Ports: the interfaces the application layer is written against.

`typing.Protocol`, not ABCs. Structural typing means an adapter never inherits
from anything in this module, so `infrastructure` stays free of framework-shaped
base classes and a test fake is just a class with the right methods.

Nothing here may import from `infrastructure`, `interfaces` or any vendor SDK.
`.importlinter` enforces that, and CI fails on violation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from autopilot.domain.models import (
    ClassificationResult,
    CompletionRequest,
    LLMResponse,
    Message,
    ModelConfig,
    PromptFeatures,
    RequestRecord,
    VerificationJob,
)


@runtime_checkable
class Clock(Protocol):
    """Time as a dependency.

    `datetime.now()` inside the core makes sampling windows, retry backoff and
    cost-per-day aggregation untestable without sleeping. Always tz-aware.
    """

    def now(self) -> datetime: ...


@runtime_checkable
class TokenCounter(Protocol):
    """Token counting, behind a port.

    Counting accurately requires a model-family tokenizer, which lives in a
    vendor package the core is forbidden from importing. So it is a port. The
    production adapter delegates to the gateway's tokenizer; tests use a
    deterministic heuristic and never need one installed.

    Takes a `ModelConfig` rather than a bare string on purpose: tokenization
    depends on the *provider* model, and passing the catalog key by mistake
    would produce silently wrong counts.
    """

    def count(self, messages: Sequence[Message], model: ModelConfig) -> int: ...


@runtime_checkable
class LLMGateway(Protocol):
    """The provider boundary. The only thing that talks to an LLM.

    Implementations must translate vendor exceptions into
    `autopilot.domain.errors` types, and must report the model that *actually*
    answered in `LLMResponse.model_key` — which may differ from `model` when a
    fallback chain fires.
    """

    async def complete(
        self,
        request: CompletionRequest,
        model: ModelConfig,
        *,
        timeout_s: float,
    ) -> LLMResponse: ...


@runtime_checkable
class ComplexityClassifier(Protocol):
    """Prompt features → complexity tier, with a calibrated confidence.

    `version` is not decoration: it is written onto every `RoutingDecision`, so
    that a change in routing behaviour can be attributed to a model swap rather
    than guessed at.
    """

    @property
    def version(self) -> str: ...

    def classify(self, features: PromptFeatures) -> ClassificationResult: ...


@runtime_checkable
class RequestStore(Protocol):
    """Durable record of every served request and its decision."""

    async def save(self, record: RequestRecord) -> None: ...

    async def get(self, request_id: UUID) -> RequestRecord | None: ...

    async def list_since(self, since: datetime, *, limit: int) -> Sequence[RequestRecord]: ...


@runtime_checkable
class VerificationQueue(Protocol):
    """Work queue for the async quality checks.

    `dequeue` leases rather than deletes: a worker that dies mid-job must not
    take the job with it. Redelivery after the lease expires is the storage
    adapter's responsibility.
    """

    async def enqueue(self, job: VerificationJob) -> None: ...

    async def lease(self, *, limit: int, lease_seconds: int) -> Sequence[VerificationJob]: ...

    async def complete(self, job_id: UUID) -> None: ...

    async def fail(self, job_id: UUID, error: str) -> None: ...
