"""The fakes are test infrastructure, and test infrastructure that lies is
worse than none. These check that they behave like the things they stand in for.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from autopilot.domain.errors import ProviderRateLimitError
from autopilot.domain.models import ComplexityTier, PromptFeatures, SamplingStratum
from tests.factories import CHEAP, EXPENSIVE, LOCAL, make_job, make_record, make_request
from tests.fakes import (
    FakeClassifier,
    FakeLLMGateway,
    FrozenClock,
    HeuristicTokenCounter,
    InMemoryRequestStore,
    InMemoryVerificationQueue,
)

# --- clock --------------------------------------------------------------------


def test_the_clock_only_moves_when_told() -> None:
    clock = FrozenClock()
    first = clock.now()
    assert clock.now() == first

    clock.advance(seconds=90)
    assert (clock.now() - first).total_seconds() == 90


def test_the_clock_is_timezone_aware() -> None:
    """Domain types reject naive datetimes, so a naive fake clock would make
    every test that persists a record fail for the wrong reason."""
    assert FrozenClock().now().tzinfo is not None


# --- gateway ------------------------------------------------------------------


async def test_the_gateway_records_which_model_was_asked() -> None:
    gateway = FakeLLMGateway(replies={"llama3-local": "4"})

    response = await gateway.complete(make_request(), LOCAL, timeout_s=5.0)

    assert response.content == "4"
    assert gateway.called_model_keys == ["llama3-local"]
    assert gateway.calls[0].timeout_s == 5.0


async def test_the_gateway_prices_the_response_with_the_real_catalog_entry() -> None:
    """A fake that returns a zero cost would let every accounting test pass
    while the arithmetic was wrong."""
    gateway = FakeLLMGateway()

    cheap = await gateway.complete(make_request(), CHEAP, timeout_s=5.0)
    dear = await gateway.complete(make_request(), EXPENSIVE, timeout_s=5.0)

    assert cheap.cost.total > 0
    assert dear.cost.total > cheap.cost.total


async def test_the_gateway_can_simulate_a_fallback_changing_the_answering_model() -> None:
    """The real Router falls through to another model after a failure. The
    response then reports a `model_key` the policy never chose, and the cost
    accounting has to survive that."""
    gateway = FakeLLMGateway(fallback_to={"llama3-local": CHEAP})

    response = await gateway.complete(make_request(), LOCAL, timeout_s=5.0)

    assert gateway.called_model_keys == ["llama3-local"]
    assert response.model_key == "haiku-3-5"
    assert response.cost.total > 0  # it fell through to a paid model


async def test_the_gateway_raises_domain_errors_not_vendor_ones() -> None:
    gateway = FakeLLMGateway(errors={"gpt-4o": ProviderRateLimitError("429", model_key="gpt-4o")})

    with pytest.raises(ProviderRateLimitError):
        await gateway.complete(make_request(), EXPENSIVE, timeout_s=5.0)


# --- classifier ---------------------------------------------------------------


def test_the_classifier_returns_a_valid_distribution() -> None:
    """`ClassificationResult` validates hard; a sloppy fake would blow up in
    every test that used it."""
    result = FakeClassifier(tier=ComplexityTier.MODERATE, confidence=0.7).classify(
        PromptFeatures(names=("n",), values=(1.0,))
    )

    assert result.tier is ComplexityTier.MODERATE
    assert result.confidence == 0.7
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_the_classifier_can_be_driven_by_a_rule() -> None:
    classifier = FakeClassifier(
        rule=lambda f: (
            (ComplexityTier.COMPLEX, 0.95)
            if f.as_mapping()["token_count"] > 500
            else (ComplexityTier.SIMPLE, 0.88)
        )
    )

    long = classifier.classify(PromptFeatures(names=("token_count",), values=(900.0,)))
    short = classifier.classify(PromptFeatures(names=("token_count",), values=(12.0,)))

    assert long.tier is ComplexityTier.COMPLEX
    assert short.tier is ComplexityTier.SIMPLE
    assert len(classifier.seen) == 2


# --- token counter ------------------------------------------------------------


def test_the_token_counter_is_deterministic_and_needs_no_tokenizer() -> None:
    counter = HeuristicTokenCounter()
    messages = make_request(prompt="a" * 400).messages

    assert counter.count(messages, LOCAL) == counter.count(messages, LOCAL)
    assert counter.count(messages, LOCAL) == 104  # 400/4 + 4 overhead


# --- store --------------------------------------------------------------------


async def test_the_store_upserts_like_the_real_adapter() -> None:
    store = InMemoryRequestStore()
    record = make_record()

    await store.save(record)
    await store.save(record)

    assert store.saved_count == 1
    assert await store.get(record.request_id) == record


async def test_the_store_filters_and_orders_by_time() -> None:
    store = InMemoryRequestStore()
    early = make_record(at=datetime(2026, 1, 1, tzinfo=UTC))
    late = make_record(at=datetime(2026, 3, 1, tzinfo=UTC))
    await store.save(late)
    await store.save(early)

    rows = await store.list_since(datetime(2026, 2, 1, tzinfo=UTC), limit=10)

    assert [r.request_id for r in rows] == [late.request_id]


# --- queue --------------------------------------------------------------------


async def test_leasing_does_not_remove_the_job() -> None:
    """A worker that dies mid-job must not take the job with it."""
    clock = FrozenClock()
    queue = InMemoryVerificationQueue(clock=clock)
    await queue.enqueue(make_job())

    leased = await queue.lease(limit=10, lease_seconds=60)

    assert len(leased) == 1
    assert queue.depth == 1  # still there, just held


async def test_a_leased_job_is_not_handed_out_twice() -> None:
    clock = FrozenClock()
    queue = InMemoryVerificationQueue(clock=clock)
    await queue.enqueue(make_job())

    await queue.lease(limit=10, lease_seconds=60)
    second = await queue.lease(limit=10, lease_seconds=60)

    assert second == []


async def test_an_expired_lease_is_redelivered() -> None:
    clock = FrozenClock()
    queue = InMemoryVerificationQueue(clock=clock)
    await queue.enqueue(make_job())
    await queue.lease(limit=10, lease_seconds=60)

    clock.advance(seconds=61)
    redelivered = await queue.lease(limit=10, lease_seconds=60)

    assert len(redelivered) == 1


async def test_completing_a_job_removes_it() -> None:
    queue = InMemoryVerificationQueue()
    job = make_job(stratum=SamplingStratum.LOW_CONFIDENCE)
    await queue.enqueue(job)

    await queue.complete(job.job_id)

    assert queue.depth == 0
    assert queue.completed == [job.job_id]


async def test_failing_a_job_releases_it_for_retry() -> None:
    clock = FrozenClock()
    queue = InMemoryVerificationQueue(clock=clock)
    job = make_job()
    await queue.enqueue(job)
    await queue.lease(limit=1, lease_seconds=600)

    await queue.fail(job.job_id, "judge unavailable")

    assert queue.failed[job.job_id] == "judge unavailable"
    assert len(await queue.lease(limit=1, lease_seconds=600)) == 1


async def test_the_stratum_survives_the_round_trip() -> None:
    """The parity estimator reads this field. Losing it in the queue would
    silently pool enriched samples into the headline number."""
    queue = InMemoryVerificationQueue()
    await queue.enqueue(make_job(stratum=SamplingStratum.TIER_BOUNDARY))

    (leased,) = await queue.lease(limit=1, lease_seconds=60)

    assert leased.stratum is SamplingStratum.TIER_BOUNDARY
    assert not leased.stratum.is_representative
