"""The domain invariants. These are the rules the rest of the system leans on."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from autopilot.domain.models import (
    ClassificationResult,
    CompletionRequest,
    ComplexityTier,
    CostBreakdown,
    DecisionReason,
    Message,
    ModelConfig,
    PriceSource,
    PromptFeatures,
    RequestRecord,
    RoutingDecision,
    SamplingStratum,
    TokenUsage,
)
from tests.factories import CHEAP, EXPENSIVE, LOCAL, make_decision, make_record, make_response

# --- money --------------------------------------------------------------------


def test_money_arithmetic_is_exact_where_float_is_not() -> None:
    """A hundred thousand prompt tokens at 8e-7 USD each is exactly 0.08 USD.

    Accumulated in binary floating point it is not, and the error compounds
    across a month of traffic until the savings report disagrees with the
    provider invoice. That disagreement is the fastest possible way to lose
    trust in this system, so money is `Decimal` and this test is the tripwire.

    Note the accumulation is written as an explicit loop. Python's builtin
    `sum()` has used compensated (Neumaier) summation for floats since 3.12,
    and SQLite's `SUM()` has used Kahan-Babuska since 3.42 — so a naive
    demonstration through either of those *passes* and proves nothing. Real
    aggregation code is a loop, in a report, in pandas, or across a paginated
    query, and none of that is compensated. Relying on someone else's
    compensation as a correctness strategy is not a strategy.
    """
    price = Decimal("0.0000008")
    total = Decimal(0)
    for _ in range(100_000):
        total += price
    assert total == Decimal("0.08")

    float_total = 0.0
    for _ in range(100_000):
        float_total += 8e-7
    assert float_total != 0.08


def test_cost_breakdown_carries_the_prices_that_produced_it() -> None:
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
    cost = CostBreakdown.compute(usage, EXPENSIVE)

    assert cost.prompt_cost == Decimal("0.0000025") * 1000
    assert cost.completion_cost == Decimal("0.00001") * 500
    assert cost.total == cost.prompt_cost + cost.completion_cost

    # The prices travel with the number. A row written today can be re-derived
    # in six months even after the provider changes its rate card.
    assert cost.input_cost_per_token == EXPENSIVE.input_cost_per_token
    assert cost.output_cost_per_token == EXPENSIVE.output_cost_per_token


def test_no_quantisation_at_the_domain_layer() -> None:
    """A single cheap request costs far less than a cent. Rounding here would
    silently floor thousands of requests to zero."""
    cost = CostBreakdown.compute(TokenUsage(10, 5), CHEAP)
    assert cost.total > 0
    assert cost.total < Decimal("0.0001")


def test_local_models_are_free_at_the_point_of_use() -> None:
    assert LOCAL.is_free
    assert not CHEAP.is_free
    assert CostBreakdown.compute(TokenUsage(10_000, 5_000), LOCAL).total == 0


def test_model_config_rejects_negative_prices() -> None:
    with pytest.raises(ValueError, match="negative unit price"):
        replace(CHEAP, input_cost_per_token=Decimal("-0.1"))


def test_model_config_rejects_unknown_provider() -> None:
    """A typo'd provider must surface as a spelling error here, not as a
    confusing MissingCredentialsError once it reaches the Router."""
    with pytest.raises(ValueError, match="unknown provider"):
        replace(CHEAP, provider="cohere")


def test_model_config_rejects_non_positive_max_output_tokens() -> None:
    with pytest.raises(ValueError, match="max_output_tokens must be positive"):
        replace(CHEAP, max_output_tokens=0)


def test_model_config_rejects_self_referential_judge_fallback() -> None:
    with pytest.raises(ValueError, match="cannot be its own judge fallback"):
        replace(CHEAP, judge_fallback=CHEAP.key)


# --- tiers and escalation -----------------------------------------------------


def test_tiers_are_ordered_so_escalation_is_a_comparison() -> None:
    assert ComplexityTier.SIMPLE < ComplexityTier.MODERATE < ComplexityTier.COMPLEX
    assert max(ComplexityTier.SIMPLE, ComplexityTier.COMPLEX) is ComplexityTier.COMPLEX


def test_escalation_must_move_upward() -> None:
    """Recording a downgrade as an escalation would corrupt the analytics that
    attribute spend to the fail-toward-quality rule."""
    with pytest.raises(ValueError, match="upward"):
        make_decision(
            tier=ComplexityTier.SIMPLE,
            escalated_from=ComplexityTier.COMPLEX,
            reason=DecisionReason.LOW_CONFIDENCE_ESCALATION,
        )


def test_a_valid_escalation_is_accepted() -> None:
    decision = make_decision(
        tier=ComplexityTier.MODERATE,
        escalated_from=ComplexityTier.SIMPLE,
        reason=DecisionReason.LOW_CONFIDENCE_ESCALATION,
    )
    assert decision.escalated_from is ComplexityTier.SIMPLE


# --- audit trail --------------------------------------------------------------


def test_decisions_reject_naive_timestamps() -> None:
    """A naive timestamp in a system that aggregates spend per day is a bug
    waiting for a timezone boundary."""
    with pytest.raises(ValueError, match="timezone-aware"):
        RoutingDecision(
            request_id=uuid4(),
            decided_at=datetime(2026, 1, 1, 12, 0),  # naive — the point of the test
            tier=ComplexityTier.SIMPLE,
            confidence=0.9,
            reason=DecisionReason.POLICY_MATCH,
            chosen_model=LOCAL,
            baseline_model=EXPENSIVE,
            policy_version="v1",
            classifier_version="v1",
        )


def test_decision_is_frozen() -> None:
    decision = make_decision()
    with pytest.raises(AttributeError):
        decision.confidence = 0.1  # type: ignore[misc]


def test_a_served_request_must_carry_its_frozen_baseline() -> None:
    """Invariant: the counterfactual is captured at call time, not recomputed
    later from a price map that may have moved."""
    request_id = uuid4()
    with pytest.raises(ValueError, match="frozen baseline"):
        RequestRecord(
            request_id=request_id,
            received_at=datetime(2026, 1, 1, tzinfo=UTC),
            decision=make_decision(request_id=request_id),
            response=make_response(),
            baseline_cost=None,
        )


def test_a_record_needs_a_response_or_an_error() -> None:
    with pytest.raises(ValueError, match="response or an error"):
        RequestRecord(
            request_id=uuid4(),
            received_at=datetime(2026, 1, 1, tzinfo=UTC),
            decision=make_decision(),
        )


def test_savings_is_baseline_minus_actual() -> None:
    record = make_record(chosen=LOCAL, baseline=EXPENSIVE)
    assert record.savings == record.baseline_cost.total  # type: ignore[union-attr]
    assert record.savings > 0


def test_savings_is_none_when_the_call_failed() -> None:
    record = RequestRecord(
        request_id=uuid4(),
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        decision=make_decision(),
        error="provider timeout",
    )
    assert record.savings is None


def test_routing_to_the_baseline_model_saves_nothing() -> None:
    """The honest zero. A tier-3 request routed to the baseline is not a win,
    and the accounting must not pretend otherwise."""
    record = make_record(chosen=EXPENSIVE, baseline=EXPENSIVE)
    assert record.savings == 0


# --- responses ----------------------------------------------------------------


def test_truncation_is_detectable() -> None:
    """A truncated answer is a quality event. Verification needs to see it."""
    assert make_response().was_truncated is False
    assert replace(make_response(), finish_reason="length").was_truncated is True


def test_response_reports_the_model_that_actually_answered() -> None:
    """A fallback can move the call after the decision was made. Intent and
    outcome are separate fields precisely so the cost can be attributed."""
    decision = make_decision(chosen=LOCAL)
    response = make_response(model=CHEAP)  # the chain fell through to a paid model
    assert decision.chosen_model.key != response.model_key


# --- classification -----------------------------------------------------------


def test_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        ClassificationResult(
            tier=ComplexityTier.SIMPLE,
            confidence=0.5,
            probabilities={ComplexityTier.SIMPLE: 0.5, ComplexityTier.MODERATE: 0.2},
            classifier_version="v1",
        )


def test_confidence_must_be_the_probability_of_the_predicted_tier() -> None:
    """Otherwise the low-confidence escalation rule is thresholding a number
    that means nothing."""
    with pytest.raises(ValueError, match="probability of the predicted tier"):
        ClassificationResult(
            tier=ComplexityTier.SIMPLE,
            confidence=0.9,
            probabilities={
                ComplexityTier.SIMPLE: 0.4,
                ComplexityTier.MODERATE: 0.3,
                ComplexityTier.COMPLEX: 0.3,
            },
            classifier_version="v1",
        )


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_confidence_must_be_a_probability(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence out of range"):
        ClassificationResult(
            tier=ComplexityTier.SIMPLE,
            confidence=bad,
            probabilities={},
            classifier_version="v1",
        )


# --- features -----------------------------------------------------------------


def test_feature_vector_names_and_values_must_line_up() -> None:
    """A classifier trained on one column order and served with another fails
    silently. The type refuses to let that happen."""
    with pytest.raises(ValueError, match="feature vector mismatch"):
        PromptFeatures(names=("a", "b"), values=(1.0,))


def test_feature_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="duplicate feature names"):
        PromptFeatures(names=("a", "a"), values=(1.0, 2.0))


def test_features_expose_a_mapping_for_debugging() -> None:
    features = PromptFeatures(names=("token_count", "has_code"), values=(120.0, 1.0))
    assert features.as_mapping() == {"token_count": 120.0, "has_code": 1.0}


# --- sampling -----------------------------------------------------------------


def test_only_the_random_stratum_is_representative() -> None:
    """Quality parity may only be estimated from RANDOM.

    The targeted strata are deliberately enriched with requests we already
    suspect. Pooling them biases parity downward and invalidates its confidence
    interval — on the single headline number this whole project reports.
    """
    assert SamplingStratum.RANDOM.is_representative
    for stratum in SamplingStratum:
        if stratum is not SamplingStratum.RANDOM:
            assert not stratum.is_representative


# --- requests -----------------------------------------------------------------


def test_a_request_needs_at_least_one_message() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        CompletionRequest(request_id=uuid4(), messages=())


def test_max_tokens_must_be_positive_when_set() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        CompletionRequest(
            request_id=uuid4(),
            messages=(Message(role="user", content="hi"),),
            max_tokens=0,
        )


def test_token_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        TokenUsage(prompt_tokens=-1, completion_tokens=0)


def test_catalog_entries_need_a_positive_context_window() -> None:
    with pytest.raises(ValueError, match="max_context_tokens"):
        ModelConfig(
            key="broken",
            provider="openai",
            provider_model_id="openai/nope",
            input_cost_per_token=Decimal("0"),
            output_cost_per_token=Decimal("0"),
            max_context_tokens=0,
            quality_tier=ComplexityTier.SIMPLE,
            price_source=PriceSource.PRICE_MAP,
        )
