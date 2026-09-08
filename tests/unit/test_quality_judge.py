"""`QualityJudge` — the application service orchestrating deterministic
checks, the judge invocation gate, judge selection, the position-swapped
pairwise comparison, and the resolved `Verdict`.

Every judge call goes through `FakeLLMGateway`, never the network. Position
swap and flip detection are exercised with the seeded `rng` fixture
(`random.Random(1337)`, `tests/conftest.py`) so the test is replayable.
"""

from __future__ import annotations

import json
from decimal import Decimal
from random import Random

import pytest

from autopilot.application.quality_judge import QualityJudge
from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.models import ComplexityTier, ModelConfig, PriceSource
from autopilot.domain.quality import (
    JUDGE_PROMPT_VERSION,
    JudgeIndependence,
    JudgeMode,
    PairPosition,
    VerdictLabel,
    check_not_refused,
    check_not_truncated,
)
from tests.factories import make_request, make_response
from tests.fakes import FakeLLMGateway, FrozenClock

_REFERENCE = ModelConfig(
    key="tier3-reference",
    provider="openai",
    provider_model_id="openai/tier3-reference",
    input_cost_per_token=Decimal("0.0000025"),
    output_cost_per_token=Decimal("0.00001"),
    max_context_tokens=128_000,
    quality_tier=ComplexityTier.COMPLEX,
    price_source=PriceSource.PRICE_MAP,
    baseline=True,
)

_JUDGE = ModelConfig(
    key="static-judge",
    provider="anthropic",
    provider_model_id="anthropic/static-judge",
    input_cost_per_token=Decimal("0.0000008"),
    output_cost_per_token=Decimal("0.000004"),
    max_context_tokens=200_000,
    quality_tier=ComplexityTier.COMPLEX,
    price_source=PriceSource.PRICE_MAP,
    judge=True,
    supports_json_mode=True,
)

_CANDIDATE = ModelConfig(
    key="cheap-candidate",
    provider="ollama",
    provider_model_id="ollama_chat/cheap-candidate",
    input_cost_per_token=Decimal("0"),
    output_cost_per_token=Decimal("0"),
    max_context_tokens=8192,
    quality_tier=ComplexityTier.SIMPLE,
    price_source=PriceSource.PRICE_MAP,
)


def _catalog() -> ModelCatalog:
    return ModelCatalog(models=(_REFERENCE, _JUDGE, _CANDIDATE))


def _tie_verdict_json() -> str:
    return json.dumps(
        {
            "winner": "tie",
            "margin": "negligible",
            "dimension_scores": {
                "correctness": 4,
                "completeness": 4,
                "instruction_following": 4,
                "format": 4,
            },
            "rationale": "Both answers are equivalent.",
        }
    )


def _fixed_position_a_wins_json() -> str:
    """The SAME raw text for both calls of a position-swapped pair — a judge
    exhibiting position bias, not tracking content across the swap."""
    return json.dumps(
        {
            "winner": "A",
            "margin": "moderate",
            "dimension_scores": {
                "correctness": 5,
                "completeness": 4,
                "instruction_following": 5,
                "format": 5,
            },
            "rationale": "Answer A is better.",
        }
    )


def _judge(gateway: FakeLLMGateway, *, rng: Random) -> QualityJudge:
    return QualityJudge(
        gateway=gateway,
        clock=FrozenClock(),
        catalog=_catalog(),
        reference_model=_REFERENCE,
        rng=rng,
    )


async def test_not_sampled_gate_makes_no_judge_call(rng: Random) -> None:
    """Spec: Judge Invocation Gate — checks pass but not sampled -> no judge call."""
    gateway = FakeLLMGateway()
    quality_judge = _judge(gateway, rng=rng)
    response = make_response(model=_CANDIDATE, content="a fine answer")

    verdict = await quality_judge.evaluate(
        request=make_request(),
        response=response,
        candidate_model=_CANDIDATE,
        checks=[check_not_refused, check_not_truncated],
        mode=JudgeMode.PAIRWISE,
        sampled=False,
        fail_reference_margin_min="moderate",
    )

    assert gateway.call_count == 0
    assert verdict.judged_by_llm is False
    assert verdict.label is VerdictLabel.PASS
    assert verdict.judge_independence is JudgeIndependence.UNAVAILABLE
    assert "not sampled" in verdict.reason


async def test_deterministic_failure_short_circuits_before_any_judge_call(rng: Random) -> None:
    """Spec: Deterministic Suite Precedes Any Judge Call — a FAIL means the
    judge, including the reference rerun, is never invoked at all."""
    gateway = FakeLLMGateway()
    quality_judge = _judge(gateway, rng=rng)
    response = make_response(model=_CANDIDATE, content="   ")  # empty -> not_refused fails

    verdict = await quality_judge.evaluate(
        request=make_request(),
        response=response,
        candidate_model=_CANDIDATE,
        checks=[check_not_refused, check_not_truncated],
        mode=JudgeMode.PAIRWISE,
        sampled=True,
        fail_reference_margin_min="moderate",
    )

    assert gateway.call_count == 0
    assert verdict.label is VerdictLabel.FAIL
    assert verdict.judged_by_llm is False


async def test_deterministic_mode_never_invokes_the_judge(rng: Random) -> None:
    """Spec: extraction/classification decide on deterministic checks alone."""
    gateway = FakeLLMGateway()
    quality_judge = _judge(gateway, rng=rng)
    response = make_response(model=_CANDIDATE, content="a fine answer")

    verdict = await quality_judge.evaluate(
        request=make_request(),
        response=response,
        candidate_model=_CANDIDATE,
        checks=[check_not_refused, check_not_truncated],
        mode=JudgeMode.DETERMINISTIC,
        sampled=True,
        fail_reference_margin_min=None,
    )

    assert gateway.call_count == 0
    assert verdict.label is VerdictLabel.PASS
    assert verdict.judged_by_llm is False


async def test_position_swap_is_recorded_and_json_mode_requested_for_the_judge(
    rng: Random,
) -> None:
    """Spec: Blind Position-Randomized Labelling + Verdict Provenance Fields.

    A tie every time (position-independent outcome) proves the swap is real
    without needing to distinguish which call happened first.
    """
    gateway = FakeLLMGateway(
        replies={_REFERENCE.key: "reference answer", _JUDGE.key: _tie_verdict_json()}
    )
    quality_judge = _judge(gateway, rng=rng)
    response = make_response(model=_CANDIDATE, content="candidate answer")

    verdict = await quality_judge.evaluate(
        request=make_request(),
        response=response,
        candidate_model=_CANDIDATE,
        checks=[check_not_refused, check_not_truncated],
        mode=JudgeMode.PAIRWISE,
        sampled=True,
        fail_reference_margin_min="moderate",
    )

    assert gateway.call_count == 3  # 1 reference rerun + 2 swapped judge calls
    assert verdict.judged_by_llm is True
    assert len(verdict.judgments) == 2
    positions = {j.position_of_cheap_answer for j in verdict.judgments}
    assert positions == {PairPosition.A, PairPosition.B}
    assert verdict.flip_detected is False
    assert verdict.label is VerdictLabel.PASS

    # Provenance fields present on every produced Verdict.
    assert verdict.judge_model == _JUDGE.key
    assert verdict.position_of_cheap_answer is not None
    assert verdict.judge_prompt_version == JUDGE_PROMPT_VERSION

    judge_calls = [c for c in gateway.calls if c.model.key == _JUDGE.key]
    reference_calls = [c for c in gateway.calls if c.model.key == _REFERENCE.key]
    assert len(reference_calls) == 1
    assert reference_calls[0].response_format is None
    assert len(judge_calls) == 2
    assert all(c.response_format == {"type": "json_object"} for c in judge_calls)


async def test_flip_detection_resolves_to_tie_and_pass(rng: Random) -> None:
    """Task 2.10: the two swapped calls disagree in normalized terms (same
    raw position twice) -> flip_detected=True, winner=tie/margin=negligible,
    a conservative PASS."""
    gateway = FakeLLMGateway(
        replies={_REFERENCE.key: "reference answer", _JUDGE.key: _fixed_position_a_wins_json()}
    )
    quality_judge = _judge(gateway, rng=rng)
    response = make_response(model=_CANDIDATE, content="candidate answer")

    verdict = await quality_judge.evaluate(
        request=make_request(),
        response=response,
        candidate_model=_CANDIDATE,
        checks=[check_not_refused, check_not_truncated],
        mode=JudgeMode.PAIRWISE,
        sampled=True,
        fail_reference_margin_min="moderate",
    )

    assert verdict.flip_detected is True
    assert verdict.winner == "tie"
    assert verdict.margin == "negligible"
    assert verdict.label is VerdictLabel.PASS


async def test_position_swap_determinism_is_seeded() -> None:
    """The same seed produces the same first-position choice across runs —
    replayable, not merely "probably random"."""
    gateway_a = FakeLLMGateway(
        replies={_REFERENCE.key: "reference answer", _JUDGE.key: _tie_verdict_json()}
    )
    gateway_b = FakeLLMGateway(
        replies={_REFERENCE.key: "reference answer", _JUDGE.key: _tie_verdict_json()}
    )
    response = make_response(model=_CANDIDATE, content="candidate answer")

    verdict_a = await _judge(gateway_a, rng=Random(1337)).evaluate(  # noqa: S311 — replay, not crypto
        request=make_request(),
        response=response,
        candidate_model=_CANDIDATE,
        checks=[check_not_refused, check_not_truncated],
        mode=JudgeMode.PAIRWISE,
        sampled=True,
        fail_reference_margin_min="moderate",
    )
    verdict_b = await _judge(gateway_b, rng=Random(1337)).evaluate(  # noqa: S311 — replay, not crypto
        request=make_request(),
        response=response,
        candidate_model=_CANDIDATE,
        checks=[check_not_refused, check_not_truncated],
        mode=JudgeMode.PAIRWISE,
        sampled=True,
        fail_reference_margin_min="moderate",
    )

    assert [j.position_of_cheap_answer for j in verdict_a.judgments] == [
        j.position_of_cheap_answer for j in verdict_b.judgments
    ]


async def test_judge_unavailable_when_no_independent_model_qualifies(rng: Random) -> None:
    """Spec: no different-family model -> judge not run, judge_model=None.

    The candidate shares the static judge's own family (anthropic), and no
    ollama model is in this catalog at all -- with `reference` (openai) and
    `candidate` (anthropic) forbidding both families that exist, nothing
    qualifies, including the static judge itself.
    """
    gateway = FakeLLMGateway()
    same_family_candidate = ModelConfig(
        key="same-family-candidate",
        provider="anthropic",
        provider_model_id="anthropic/same-family-candidate",
        input_cost_per_token=Decimal("0.0000008"),
        output_cost_per_token=Decimal("0.000004"),
        max_context_tokens=200_000,
        quality_tier=ComplexityTier.MODERATE,
        price_source=PriceSource.PRICE_MAP,
    )
    catalog_two_families = ModelCatalog(models=(_REFERENCE, _JUDGE, same_family_candidate))
    quality_judge = QualityJudge(
        gateway=gateway,
        clock=FrozenClock(),
        catalog=catalog_two_families,
        reference_model=_REFERENCE,
        rng=rng,
    )
    response = make_response(model=same_family_candidate, content="a fine answer")

    verdict = await quality_judge.evaluate(
        request=make_request(),
        response=response,
        candidate_model=same_family_candidate,
        checks=[check_not_refused, check_not_truncated],
        mode=JudgeMode.PAIRWISE,
        sampled=True,
        fail_reference_margin_min="moderate",
    )

    assert gateway.call_count == 0
    assert verdict.judged_by_llm is False
    assert verdict.judge_independence is JudgeIndependence.UNAVAILABLE
    assert verdict.judge_model is None


async def test_judge_unavailable_response_raises_judge_unavailable_error(rng: Random) -> None:
    """Spec: the verdict is never invented — garbage judge output propagates
    as `JudgeUnavailableError`, not a fabricated Verdict."""
    from autopilot.domain.errors import JudgeUnavailableError

    gateway = FakeLLMGateway(
        replies={_REFERENCE.key: "reference answer", _JUDGE.key: "not json at all"}
    )
    quality_judge = _judge(gateway, rng=rng)
    response = make_response(model=_CANDIDATE, content="candidate answer")

    with pytest.raises(JudgeUnavailableError):
        await quality_judge.evaluate(
            request=make_request(),
            response=response,
            candidate_model=_CANDIDATE,
            checks=[check_not_refused, check_not_truncated],
            mode=JudgeMode.PAIRWISE,
            sampled=True,
            fail_reference_margin_min="moderate",
        )
