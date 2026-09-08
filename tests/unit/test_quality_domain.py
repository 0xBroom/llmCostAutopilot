"""The quality-contract domain: deterministic checks and verdict value objects.

Deterministic checks are pure and free — a failure here is confidence 1.0,
by construction, because nothing probabilistic decided it. `run_deterministic
_checks` short-circuits on the first failure so a caller never pays an LLM
call for a check whose outcome the deterministic suite already decided.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest

from autopilot.domain.models import LLMResponse
from autopilot.domain.quality import (
    DIMENSIONS,
    DeterministicCheckResult,
    DeterministicOutcome,
    JudgeIndependence,
    PairPosition,
    PairwiseJudgment,
    TaskType,
    Verdict,
    VerdictLabel,
    check_json_valid,
    check_label_in_set,
    check_length_anomaly,
    check_not_refused,
    check_not_truncated,
    check_required_fields,
    run_deterministic_checks,
)
from tests.factories import make_response

# --- helpers --------------------------------------------------------------


def _valid_dimension_scores() -> dict[str, int]:
    return dict.fromkeys(DIMENSIONS, 5)


def _make_pairwise_judgment(**overrides: Any) -> PairwiseJudgment:
    fields: dict[str, Any] = {
        "position_of_cheap_answer": PairPosition.A,
        "winner": "tie",
        "margin": "negligible",
        "dimension_scores": _valid_dimension_scores(),
        "rationale": "Both answers are equivalent.",
        "judge_model": "sonnet-4-5",
        "judge_prompt_version": "qj-1.0",
    }
    fields.update(overrides)
    return PairwiseJudgment(**fields)


def _make_verdict(**overrides: Any) -> Verdict:
    fields: dict[str, Any] = {
        "request_id": uuid4(),
        "label": VerdictLabel.PASS,
        "task_type": TaskType.DEFAULT,
        "deterministic": DeterministicOutcome(results=()),
        "judged_by_llm": False,
        "judge_model": None,
        "reference_model": None,
        "judgments": (),
        "judge_independence": JudgeIndependence.UNAVAILABLE,
        "flip_detected": None,
        "winner": "tie",
        "margin": "negligible",
        "dimension_scores": _valid_dimension_scores(),
        "rationale": "",
        "position_of_cheap_answer": None,
        "judge_prompt_version": None,
        "reason": "no judge invoked",
    }
    fields.update(overrides)
    return Verdict(**fields)


# --- JSON / schema validity -------------------------------------------------


def test_valid_json_passes() -> None:
    response = make_response(content='{"answer": 4}')
    assert check_json_valid(response).passed is True


def test_malformed_json_fails() -> None:
    """Spec: JSON/Schema Validity — malformed JSON FAILs, judge never invoked."""
    response = make_response(content="not json at all {")
    result = check_json_valid(response)
    assert result.passed is False
    assert "not valid JSON" in result.detail


# --- required-field presence (extraction) -----------------------------------


def test_required_fields_present_passes() -> None:
    response = make_response(content='{"name": "Alice", "age": 30}')
    result = check_required_fields(response, required=("name", "age"))
    assert result.passed is True


def test_required_field_missing_fails() -> None:
    """Spec: Required-Field Presence — a missing/null required field FAILs."""
    response = make_response(content='{"name": "Alice", "age": null}')
    result = check_required_fields(response, required=("name", "age"))
    assert result.passed is False
    assert "age" in result.detail


def test_required_fields_on_non_object_json_fails() -> None:
    response = make_response(content="[1, 2, 3]")
    result = check_required_fields(response, required=("name",))
    assert result.passed is False


def test_required_fields_on_malformed_json_fails() -> None:
    response = make_response(content="{broken")
    result = check_required_fields(response, required=("name",))
    assert result.passed is False


# --- label-in-allowed-set (classification) ----------------------------------


def test_label_in_allowed_set_passes() -> None:
    response = make_response(content="positive")
    result = check_label_in_set(response, allowed=frozenset({"positive", "negative"}))
    assert result.passed is True


def test_label_outside_allowed_set_fails() -> None:
    """Spec: Label-in-Allowed-Set — a label outside the configured set FAILs."""
    response = make_response(content="mixed")
    result = check_label_in_set(response, allowed=frozenset({"positive", "negative"}))
    assert result.passed is False
    assert "mixed" in result.detail


# --- refusal / empty / truncation -------------------------------------------


def test_not_refused_passes_on_ordinary_content() -> None:
    response = make_response(content="The answer is 4.")
    assert check_not_refused(response).passed is True


def test_empty_content_fails_not_refused() -> None:
    """Spec: Refusal/Empty/Truncation — empty content FAILs, judge never invoked."""
    response = make_response(content="   ")
    result = check_not_refused(response)
    assert result.passed is False
    assert "empty" in result.detail


def test_refusal_text_fails_not_refused() -> None:
    """Spec: Refusal/Empty/Truncation — refusal content FAILs, judge never invoked."""
    response = make_response(content="I cannot help with that request.")
    result = check_not_refused(response)
    assert result.passed is False
    assert "refusal" in result.detail


def test_finish_reason_length_sets_was_truncated_and_fails() -> None:
    """Spec: Refusal/Empty/Truncation — truncation FAILs at confidence 1.0."""
    response = replace(make_response(), finish_reason="length")
    assert response.was_truncated is True
    result = check_not_truncated(response)
    assert result.passed is False
    assert "length" in result.detail


def test_finish_reason_stop_passes_not_truncated() -> None:
    response = replace(make_response(), finish_reason="stop")
    assert response.was_truncated is False
    assert check_not_truncated(response).passed is True


# --- gross length anomaly ----------------------------------------------------


def test_length_within_bounds_passes() -> None:
    response = make_response(content="a reasonably sized answer")
    assert check_length_anomaly(response, bounds=(1, 1000)).passed is True


def test_length_gross_outlier_fails() -> None:
    """Spec: Gross Length Anomaly — a length far outside bounds FAILs."""
    response = make_response(content="hi")
    result = check_length_anomaly(response, bounds=(50, 1000))
    assert result.passed is False
    assert "outside bounds" in result.detail


# --- deterministic suite: short-circuit -------------------------------------


def test_deterministic_outcome_passed_requires_every_check_to_pass() -> None:
    outcome = DeterministicOutcome(
        results=(
            DeterministicCheckResult(name="a", passed=True, detail=""),
            DeterministicCheckResult(name="b", passed=True, detail=""),
        )
    )
    assert outcome.passed is True

    failing_outcome = DeterministicOutcome(
        results=(
            DeterministicCheckResult(name="a", passed=True, detail=""),
            DeterministicCheckResult(name="b", passed=False, detail="nope"),
        )
    )
    assert failing_outcome.passed is False


def test_run_deterministic_checks_short_circuits_on_first_failure() -> None:
    """Spec: Deterministic Suite Precedes Any Judge Call — the suite stops at
    the first failure. Zero LLM cost by construction: a check that never runs
    can never invoke a judge."""
    calls: list[str] = []

    def failing(response: LLMResponse) -> DeterministicCheckResult:
        calls.append("failing")
        return DeterministicCheckResult(name="failing", passed=False, detail="boom")

    def never_called(response: LLMResponse) -> DeterministicCheckResult:
        calls.append("never_called")
        return DeterministicCheckResult(name="never_called", passed=True, detail="")

    outcome = run_deterministic_checks(make_response(), [failing, never_called])

    assert outcome.passed is False
    assert calls == ["failing"]
    assert len(outcome.results) == 1


def test_run_deterministic_checks_runs_every_check_when_all_pass() -> None:
    def a(response: LLMResponse) -> DeterministicCheckResult:
        return DeterministicCheckResult(name="a", passed=True, detail="")

    def b(response: LLMResponse) -> DeterministicCheckResult:
        return DeterministicCheckResult(name="b", passed=True, detail="")

    outcome = run_deterministic_checks(make_response(), [a, b])

    assert outcome.passed is True
    assert [r.name for r in outcome.results] == ["a", "b"]


# --- PairwiseJudgment / Verdict validation -----------------------------------


def test_pairwise_judgment_accepts_valid_fields() -> None:
    judgment = _make_pairwise_judgment()
    assert judgment.winner == "tie"


@pytest.mark.parametrize("bad_winner", ["C", "", "TIE", "a"])
def test_pairwise_judgment_rejects_non_member_winner(bad_winner: str) -> None:
    with pytest.raises(ValueError, match="winner must be one of"):
        _make_pairwise_judgment(winner=bad_winner)


@pytest.mark.parametrize("bad_margin", ["extreme", "", "Moderate"])
def test_pairwise_judgment_rejects_non_member_margin(bad_margin: str) -> None:
    with pytest.raises(ValueError, match="margin must be one of"):
        _make_pairwise_judgment(margin=bad_margin)


def test_pairwise_judgment_rejects_missing_dimension_key() -> None:
    scores = _valid_dimension_scores()
    del scores["format"]
    with pytest.raises(ValueError, match="dimension_scores keys must be exactly"):
        _make_pairwise_judgment(dimension_scores=scores)


def test_pairwise_judgment_rejects_extra_dimension_key() -> None:
    scores = _valid_dimension_scores()
    scores["extra"] = 5
    with pytest.raises(ValueError, match="dimension_scores keys must be exactly"):
        _make_pairwise_judgment(dimension_scores=scores)


@pytest.mark.parametrize("bad_score", [0, 6, -1])
def test_pairwise_judgment_rejects_out_of_range_score(bad_score: int) -> None:
    scores = _valid_dimension_scores()
    scores["correctness"] = bad_score
    with pytest.raises(ValueError, match="score must be 1-5"):
        _make_pairwise_judgment(dimension_scores=scores)


def test_pairwise_judgment_rejects_rationale_over_400_chars() -> None:
    with pytest.raises(ValueError, match="rationale must be"):
        _make_pairwise_judgment(rationale="x" * 401)


def test_pairwise_judgment_accepts_rationale_at_exactly_400_chars() -> None:
    judgment = _make_pairwise_judgment(rationale="x" * 400)
    assert len(judgment.rationale) == 400


def test_verdict_accepts_valid_fields() -> None:
    verdict = _make_verdict()
    assert verdict.is_pass is True


def test_verdict_tie_label_is_a_pass() -> None:
    verdict = _make_verdict(label=VerdictLabel.TIE)
    assert verdict.is_pass is True


def test_verdict_fail_label_is_not_a_pass() -> None:
    verdict = _make_verdict(label=VerdictLabel.FAIL)
    assert verdict.is_pass is False


@pytest.mark.parametrize("bad_winner", ["C", "", "win"])
def test_verdict_rejects_non_member_winner(bad_winner: str) -> None:
    with pytest.raises(ValueError, match="winner must be one of"):
        _make_verdict(winner=bad_winner)


@pytest.mark.parametrize("bad_margin", ["huge", ""])
def test_verdict_rejects_non_member_margin(bad_margin: str) -> None:
    with pytest.raises(ValueError, match="margin must be one of"):
        _make_verdict(margin=bad_margin)


def test_verdict_rejects_dimension_scores_key_mismatch() -> None:
    scores = _valid_dimension_scores()
    del scores["correctness"]
    with pytest.raises(ValueError, match="dimension_scores keys must be exactly"):
        _make_verdict(dimension_scores=scores)


@pytest.mark.parametrize("bad_score", [0, 6])
def test_verdict_rejects_out_of_range_score(bad_score: int) -> None:
    scores = _valid_dimension_scores()
    scores["completeness"] = bad_score
    with pytest.raises(ValueError, match="score must be 1-5"):
        _make_verdict(dimension_scores=scores)


def test_verdict_rejects_rationale_over_400_chars() -> None:
    with pytest.raises(ValueError, match="rationale must be"):
        _make_verdict(rationale="x" * 401)


def test_verdict_is_frozen() -> None:
    verdict = _make_verdict()
    with pytest.raises(AttributeError):
        verdict.label = VerdictLabel.FAIL  # type: ignore[misc]
