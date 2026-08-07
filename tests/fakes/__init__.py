"""Typed in-memory implementations of every port.

Every fake in this package satisfies its `Protocol` under mypy. That check is
enforced in `tests/unit/test_ports_contract.py`, which is the whole reason
these are hand-written classes and not mocks.
"""

from tests.fakes.classifier import FakeClassifier
from tests.fakes.clock import FrozenClock
from tests.fakes.gateway import FakeLLMGateway, GatewayCall
from tests.fakes.queue import InMemoryVerificationQueue
from tests.fakes.store import InMemoryRequestStore
from tests.fakes.tokens import HeuristicTokenCounter

__all__ = [
    "FakeClassifier",
    "FakeLLMGateway",
    "FrozenClock",
    "GatewayCall",
    "HeuristicTokenCounter",
    "InMemoryRequestStore",
    "InMemoryVerificationQueue",
]
