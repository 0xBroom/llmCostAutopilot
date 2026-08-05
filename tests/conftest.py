"""Shared fixtures and the guarantees that hold for the whole suite."""

from __future__ import annotations

import os
import random

import pytest

from tests.factories import CATALOG
from tests.fakes import (
    FakeClassifier,
    FakeLLMGateway,
    FrozenClock,
    HeuristicTokenCounter,
    InMemoryRequestStore,
    InMemoryVerificationQueue,
)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip provider credentials from the environment for every test.

    If a test can pass on a developer laptop because a key happened to be
    exported, it will fail in CI where none exists — or worse, it will pass in
    CI by silently spending money. Neither is acceptable, so the default suite
    runs with no keys at all, exactly like a fork's pull request does.
    """
    for name in list(os.environ):
        if name.startswith(("AUTOPILOT_", "OPENAI_", "ANTHROPIC_", "LITELLM_")):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def rng() -> random.Random:
    """Seeded randomness. The verification sampler must be replayable."""
    return random.Random(1337)  # noqa: S311 — sampling, not cryptography


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def catalog() -> dict[str, object]:
    return dict(CATALOG)


@pytest.fixture
def gateway() -> FakeLLMGateway:
    return FakeLLMGateway()


@pytest.fixture
def classifier() -> FakeClassifier:
    return FakeClassifier()


@pytest.fixture
def token_counter() -> HeuristicTokenCounter:
    return HeuristicTokenCounter()


@pytest.fixture
def store() -> InMemoryRequestStore:
    return InMemoryRequestStore()


@pytest.fixture
def queue(clock: FrozenClock) -> InMemoryVerificationQueue:
    return InMemoryVerificationQueue(clock=clock)


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, object]:
    """Cassette hygiene, applied at record time.

    Scrubbing on the way in rather than reviewing on the way out: a secret that
    reaches disk is already in the reflog, and `git rm` does not remove it.
    """
    return {
        "record_mode": "none",  # CI replays only. Recording is an explicit local act.
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("api-key", "REDACTED"),
            ("x-api-key", "REDACTED"),
            ("openai-organization", "REDACTED"),
            ("set-cookie", "REDACTED"),
            ("cookie", "REDACTED"),
        ],
        "filter_query_parameters": [("key", "REDACTED"), ("api_key", "REDACTED")],
        "decode_compressed_response": True,
    }


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
