"""Every fake must satisfy its port.

This file is the reason the fakes are hand-written classes instead of mocks.
The assignments below are checked by **mypy**, not just at runtime: if someone
adds a parameter to `LLMGateway.complete` and forgets the fake, `make typecheck`
fails here. A `MagicMock` would have absorbed the change and kept the suite
green while production broke.

The `isinstance` assertions are a weaker, runtime-only backstop — a
`runtime_checkable` Protocol only checks that the members exist, never their
signatures. Both layers, because they catch different mistakes.
"""

from __future__ import annotations

from autopilot.application.ports import (
    Clock,
    ComplexityClassifier,
    LLMGateway,
    RequestStore,
    TokenCounter,
    VerificationQueue,
)
from tests.fakes import (
    FakeClassifier,
    FakeLLMGateway,
    FrozenClock,
    HeuristicTokenCounter,
    InMemoryRequestStore,
    InMemoryVerificationQueue,
)


def test_fakes_satisfy_their_ports_statically() -> None:
    # Static: mypy verifies each assignment against the Protocol's signatures.
    _clock: Clock = FrozenClock()
    _gateway: LLMGateway = FakeLLMGateway()
    _classifier: ComplexityClassifier = FakeClassifier()
    _tokens: TokenCounter = HeuristicTokenCounter()
    _store: RequestStore = InMemoryRequestStore()
    _queue: VerificationQueue = InMemoryVerificationQueue()

    # Runtime: member presence only.
    assert isinstance(_clock, Clock)
    assert isinstance(_gateway, LLMGateway)
    assert isinstance(_classifier, ComplexityClassifier)
    assert isinstance(_tokens, TokenCounter)
    assert isinstance(_store, RequestStore)
    assert isinstance(_queue, VerificationQueue)
