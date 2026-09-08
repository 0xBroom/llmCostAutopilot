"""The quality-contract domain: deterministic checks and verdict value objects.

Deterministic checks are pure and free — a failure here is confidence 1.0,
by construction, because nothing probabilistic decided it. `run_deterministic
_checks` short-circuits on the first failure so a caller never pays an LLM
call for a check whose outcome the deterministic suite already decided.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.errors import JudgeUnavailableError
from autopilot.domain.models import ComplexityTier, LLMResponse, Message, ModelConfig, PriceSource
from autopilot.domain.quality import (
    DIMENSIONS,
    JUDGE_PROMPT_VERSION,
    MARGIN_ORDER,
    DeterministicCheckResult,
    DeterministicOutcome,
    JudgeIndependence,
    JudgeMode,
    JudgeSelection,
    PairPosition,
    PairwiseJudgment,
    TaskType,
    Verdict,
    VerdictLabel,
    build_judge_prompt,
    check_json_valid,
    check_label_in_set,
    check_length_anomaly,
    check_not_refused,
    check_not_truncated,
    check_required_fields,
    combine_swapped_judgments,
    parse_verdict,
    resolve_verdict_label,
    run_deterministic_checks,
    select_judge,
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


# --- select_judge -------------------------------------------------------------
#
# Only 3 provider families exist system-wide (`KNOWN_PROVIDERS`), so a scenario
# that leaves exactly one qualifying family is the natural way to force each
# branch: the static judge either belongs to that one remaining family or it
# does not.

_JUDGE_GPT = ModelConfig(
    key="static-judge-gpt",
    provider="openai",
    provider_model_id="openai/static-judge-gpt",
    input_cost_per_token=Decimal("0.0000025"),
    output_cost_per_token=Decimal("0.00001"),
    max_context_tokens=128_000,
    quality_tier=ComplexityTier.COMPLEX,
    price_source=PriceSource.PRICE_MAP,
    baseline=True,
    judge=True,
)

_LOCAL_MODEL = ModelConfig(
    key="local-model",
    provider="ollama",
    provider_model_id="ollama_chat/local-model",
    input_cost_per_token=Decimal("0"),
    output_cost_per_token=Decimal("0"),
    max_context_tokens=8192,
    quality_tier=ComplexityTier.SIMPLE,
    price_source=PriceSource.PRICE_MAP,
)

_CLOUD_ANTHROPIC = ModelConfig(
    key="cloud-anthropic",
    provider="anthropic",
    provider_model_id="anthropic/cloud-anthropic",
    input_cost_per_token=Decimal("0.0000008"),
    output_cost_per_token=Decimal("0.000004"),
    max_context_tokens=200_000,
    quality_tier=ComplexityTier.MODERATE,
    price_source=PriceSource.PRICE_MAP,
)

_OTHER_OPENAI = ModelConfig(
    key="other-openai",
    provider="openai",
    provider_model_id="openai/other-openai",
    input_cost_per_token=Decimal("0.000001"),
    output_cost_per_token=Decimal("0.000002"),
    max_context_tokens=128_000,
    quality_tier=ComplexityTier.MODERATE,
    price_source=PriceSource.PRICE_MAP,
)


def test_select_judge_uses_static_judge_when_its_family_qualifies() -> None:
    """Branch 1. reference=ollama, candidate=anthropic -> only openai (the
    static judge's family) qualifies."""
    catalog = ModelCatalog(models=(_LOCAL_MODEL, _CLOUD_ANTHROPIC, _JUDGE_GPT))

    selection = select_judge(catalog, reference=_LOCAL_MODEL, candidate=_CLOUD_ANTHROPIC)

    assert selection.judge_model == _JUDGE_GPT
    assert selection.independence == JudgeIndependence.INDEPENDENT
    assert isinstance(selection, JudgeSelection)


def test_select_judge_falls_back_to_strongest_qualifying_cloud_model() -> None:
    """Branch 2. The static judge (openai) is forbidden because the reference
    is also openai; anthropic is the only remaining family and qualifies."""
    catalog = ModelCatalog(models=(_OTHER_OPENAI, _LOCAL_MODEL, _JUDGE_GPT, _CLOUD_ANTHROPIC))

    selection = select_judge(catalog, reference=_OTHER_OPENAI, candidate=_LOCAL_MODEL)

    assert selection.judge_model == _CLOUD_ANTHROPIC
    assert selection.independence == JudgeIndependence.INDEPENDENT
    assert selection.judge_model != _JUDGE_GPT


def test_select_judge_degrades_to_local_when_only_a_local_model_qualifies() -> None:
    """Branch 3. Both cloud families (openai, anthropic) are forbidden by
    reference/candidate; only the local model is left."""
    catalog = ModelCatalog(models=(_OTHER_OPENAI, _CLOUD_ANTHROPIC, _JUDGE_GPT, _LOCAL_MODEL))

    selection = select_judge(catalog, reference=_CLOUD_ANTHROPIC, candidate=_OTHER_OPENAI)

    assert selection.judge_model == _LOCAL_MODEL
    assert selection.independence == JudgeIndependence.DEGRADED_LOCAL


def test_select_judge_is_unavailable_when_no_different_family_model_qualifies() -> None:
    """Branch 4. No ollama model exists in this catalog at all, and both
    cloud families are forbidden -> judge not run."""
    catalog = ModelCatalog(models=(_OTHER_OPENAI, _CLOUD_ANTHROPIC, _JUDGE_GPT))

    selection = select_judge(catalog, reference=_CLOUD_ANTHROPIC, candidate=_OTHER_OPENAI)

    assert selection.judge_model is None
    assert selection.independence == JudgeIndependence.UNAVAILABLE


@pytest.mark.parametrize(
    ("catalog_models", "reference", "candidate"),
    [
        ((_LOCAL_MODEL, _CLOUD_ANTHROPIC, _JUDGE_GPT), _LOCAL_MODEL, _CLOUD_ANTHROPIC),
        ((_OTHER_OPENAI, _LOCAL_MODEL, _JUDGE_GPT, _CLOUD_ANTHROPIC), _OTHER_OPENAI, _LOCAL_MODEL),
        (
            (_OTHER_OPENAI, _CLOUD_ANTHROPIC, _JUDGE_GPT, _LOCAL_MODEL),
            _CLOUD_ANTHROPIC,
            _OTHER_OPENAI,
        ),
        ((_OTHER_OPENAI, _CLOUD_ANTHROPIC, _JUDGE_GPT), _CLOUD_ANTHROPIC, _OTHER_OPENAI),
    ],
)
def test_select_judge_never_selects_the_reference_model(
    catalog_models: tuple[ModelConfig, ...], reference: ModelConfig, candidate: ModelConfig
) -> None:
    """Spec: Cross-Family Judge Selection — across all 4 branches, the judge
    is never the reference model, by identity."""
    catalog = ModelCatalog(models=catalog_models)

    selection = select_judge(catalog, reference=reference, candidate=candidate)

    assert selection.judge_model is None or selection.judge_model.key != reference.key


def test_select_judge_never_mutates_the_catalog() -> None:
    catalog = ModelCatalog(models=(_LOCAL_MODEL, _CLOUD_ANTHROPIC, _JUDGE_GPT))
    before = tuple(catalog.models)

    select_judge(catalog, reference=_LOCAL_MODEL, candidate=_CLOUD_ANTHROPIC)

    assert catalog.models == before


# --- resolve_verdict_label -----------------------------------------------------


@pytest.mark.parametrize("mode", list(JudgeMode))
@pytest.mark.parametrize("margin", MARGIN_ORDER)
def test_resolve_verdict_label_deterministic_failure_always_fails(
    mode: JudgeMode, margin: str
) -> None:
    """A deterministic FAIL short-circuits regardless of mode/margin."""
    assert (
        resolve_verdict_label(
            deterministic_passed=False,
            mode=mode,
            reference_won=True,
            margin=margin,
            fail_reference_margin_min="moderate",
        )
        is VerdictLabel.FAIL
    )


def test_resolve_verdict_label_deterministic_mode_ignores_margin() -> None:
    """Spec: extraction/classification decide on deterministic checks alone."""
    for margin in MARGIN_ORDER:
        assert (
            resolve_verdict_label(
                deterministic_passed=True,
                mode=JudgeMode.DETERMINISTIC,
                reference_won=True,
                margin=margin,
                fail_reference_margin_min=None,
            )
            is VerdictLabel.PASS
        )


def test_resolve_verdict_label_tie_or_candidate_win_is_always_pass() -> None:
    for margin in MARGIN_ORDER:
        assert (
            resolve_verdict_label(
                deterministic_passed=True,
                mode=JudgeMode.PAIRWISE,
                reference_won=False,
                margin=margin,
                fail_reference_margin_min="moderate",
            )
            is VerdictLabel.PASS
        )


@pytest.mark.parametrize(
    ("margin", "expected"),
    [
        ("negligible", VerdictLabel.PASS),
        ("moderate", VerdictLabel.FAIL),
        ("decisive", VerdictLabel.FAIL),
    ],
)
def test_resolve_verdict_label_reference_win_gated_by_margin(
    margin: str, expected: VerdictLabel
) -> None:
    """Spec: Summarization/default — FAIL iff reference wins at margin >= moderate."""
    assert (
        resolve_verdict_label(
            deterministic_passed=True,
            mode=JudgeMode.PAIRWISE,
            reference_won=True,
            margin=margin,
            fail_reference_margin_min="moderate",
        )
        is expected
    )


def test_resolve_verdict_label_missing_threshold_raises_for_pairwise_reference_win() -> None:
    with pytest.raises(ValueError, match="fail_reference_margin_min is required"):
        resolve_verdict_label(
            deterministic_passed=True,
            mode=JudgeMode.PAIRWISE,
            reference_won=True,
            margin="moderate",
            fail_reference_margin_min=None,
        )


@pytest.mark.parametrize("margin", ["negligible", "moderate", "decisive"])
def test_summarization_and_reasoning_collapse_to_the_same_boolean_outcome(margin: str) -> None:
    """Task 2.14: summarization/default (`fail_reference_margin_min=moderate`)
    and reasoning ("reference wins margin != negligible") are the SAME
    threshold under the current 3-level `MARGIN_ORDER` scale — this is a
    property of the ordered comparator, not an independently coded rule per
    task type, and not a masked divergence between them."""
    summarization_result = resolve_verdict_label(
        deterministic_passed=True,
        mode=JudgeMode.PAIRWISE,
        reference_won=True,
        margin=margin,
        fail_reference_margin_min="moderate",
    )
    # The reasoning rule ("margin != negligible") expressed as an explicit
    # boolean, independent of resolve_verdict_label's own implementation.
    reasoning_fails = margin != "negligible"
    reasoning_result = VerdictLabel.FAIL if reasoning_fails else VerdictLabel.PASS

    assert summarization_result == reasoning_result


# --- combine_swapped_judgments -------------------------------------------------


def _judgment(
    *, position: PairPosition, winner: str, margin: str = "moderate", **overrides: Any
) -> PairwiseJudgment:
    return _make_pairwise_judgment(
        position_of_cheap_answer=position, winner=winner, margin=margin, **overrides
    )


def test_combine_swapped_judgments_agrees_when_candidate_wins_both_times() -> None:
    """Call 1: cheap at A, "A" (candidate) wins. Call 2: cheap at B, "B"
    (candidate) wins. Same real-world outcome -> no flip."""
    first = _judgment(position=PairPosition.A, winner="A")
    second = _judgment(position=PairPosition.B, winner="B")

    winner, margin, reference_won, flip_detected = combine_swapped_judgments(first, second)

    assert flip_detected is False
    assert reference_won is False
    assert winner == "A"  # relative to first.position_of_cheap_answer == A
    assert margin == "moderate"


def test_combine_swapped_judgments_agrees_when_reference_wins_both_times() -> None:
    """Call 1: cheap at A, "B" (reference) wins. Call 2: cheap at B, "A"
    (reference) wins. Same real-world outcome -> no flip, reference_won=True."""
    first = _judgment(position=PairPosition.A, winner="B")
    second = _judgment(position=PairPosition.B, winner="A")

    winner, _margin, reference_won, flip_detected = combine_swapped_judgments(first, second)

    assert flip_detected is False
    assert reference_won is True
    assert winner == "B"  # the reference's position relative to first's cheap position A


def test_combine_swapped_judgments_agrees_on_tie() -> None:
    first = _judgment(position=PairPosition.A, winner="tie", margin="negligible")
    second = _judgment(position=PairPosition.B, winner="tie", margin="negligible")

    winner, margin, reference_won, flip_detected = combine_swapped_judgments(first, second)

    assert flip_detected is False
    assert reference_won is False
    assert winner == "tie"
    assert margin == "negligible"


def test_combine_swapped_judgments_flips_to_tie_on_disagreement() -> None:
    """Spec: a judge reporting the SAME raw position twice (position bias, not
    content-tracking) is a real disagreement in normalized terms."""
    first = _judgment(position=PairPosition.A, winner="A")
    second = _judgment(position=PairPosition.B, winner="A")

    winner, margin, reference_won, flip_detected = combine_swapped_judgments(first, second)

    assert flip_detected is True
    assert winner == "tie"
    assert margin == "negligible"
    assert reference_won is False


# --- build_judge_prompt / parse_verdict ----------------------------------------


def test_build_judge_prompt_renders_1_3_5_anchors_for_every_dimension() -> None:
    """Spec: Anchored Rubric — every dimension shows concrete 1/3/5 anchor text."""
    messages = build_judge_prompt(
        "reference answer",
        "candidate answer",
        cheap_position=PairPosition.A,
        task_type=TaskType.DEFAULT,
    )
    rendered = "\n".join(m.content for m in messages)

    for dimension in DIMENSIONS:
        assert dimension in rendered
    assert "1 = " in rendered
    assert "3 = " in rendered
    assert "5 = " in rendered
    assert JUDGE_PROMPT_VERSION in rendered


def test_build_judge_prompt_places_the_cheap_answer_at_the_requested_position() -> None:
    messages = build_judge_prompt(
        "REFERENCE-TEXT",
        "CANDIDATE-TEXT",
        cheap_position=PairPosition.A,
        task_type=TaskType.DEFAULT,
    )
    user_message = next(m for m in messages if m.role == "user")
    cand_index = user_message.content.index("CANDIDATE-TEXT")
    ref_index = user_message.content.index("REFERENCE-TEXT")
    assert cand_index < ref_index

    swapped = build_judge_prompt(
        "REFERENCE-TEXT",
        "CANDIDATE-TEXT",
        cheap_position=PairPosition.B,
        task_type=TaskType.DEFAULT,
    )
    swapped_user = next(m for m in swapped if m.role == "user")
    swapped_ref_index = swapped_user.content.index("REFERENCE-TEXT")
    swapped_cand_index = swapped_user.content.index("CANDIDATE-TEXT")
    assert swapped_ref_index < swapped_cand_index


def test_build_judge_prompt_returns_message_instances() -> None:
    messages = build_judge_prompt(
        "ref", "cand", cheap_position=PairPosition.A, task_type=TaskType.DEFAULT
    )
    assert all(isinstance(m, Message) for m in messages)


def test_parse_verdict_happy_path() -> None:
    text = (
        '{"winner": "A", "margin": "moderate", '
        '"dimension_scores": {"correctness": 5, "completeness": 4, '
        '"instruction_following": 5, "format": 5}, "rationale": "Answer A is more complete."}'
    )
    judgment = parse_verdict(text, cheap_position=PairPosition.A, judge_model="static-judge-gpt")

    assert judgment.winner == "A"
    assert judgment.margin == "moderate"
    assert judgment.judge_model == "static-judge-gpt"
    assert judgment.judge_prompt_version == JUDGE_PROMPT_VERSION
    assert judgment.position_of_cheap_answer == PairPosition.A


def test_parse_verdict_tolerates_markdown_code_fences() -> None:
    text = (
        "```json\n"
        '{"winner": "tie", "margin": "negligible", '
        '"dimension_scores": {"correctness": 4, "completeness": 4, '
        '"instruction_following": 4, "format": 4}, "rationale": "Equivalent."}'
        "\n```"
    )
    judgment = parse_verdict(text, cheap_position=PairPosition.B, judge_model="j")
    assert judgment.winner == "tie"


def test_parse_verdict_tolerates_surrounding_prose() -> None:
    text = (
        "Here is my assessment:\n"
        '{"winner": "B", "margin": "decisive", '
        '"dimension_scores": {"correctness": 2, "completeness": 2, '
        '"instruction_following": 2, "format": 2}, "rationale": "B is much better."}\n'
        "Let me know if you need anything else."
    )
    judgment = parse_verdict(text, cheap_position=PairPosition.A, judge_model="j")
    assert judgment.winner == "B"


def test_parse_verdict_raises_judge_unavailable_on_garbage_text() -> None:
    with pytest.raises(JudgeUnavailableError):
        parse_verdict("this is not JSON at all", cheap_position=PairPosition.A, judge_model="j")


def test_parse_verdict_raises_judge_unavailable_on_missing_field() -> None:
    text = '{"winner": "A", "margin": "moderate"}'
    with pytest.raises(JudgeUnavailableError, match="missing a required field"):
        parse_verdict(text, cheap_position=PairPosition.A, judge_model="j")


def test_parse_verdict_raises_judge_unavailable_on_invalid_verdict_shape() -> None:
    """Well-formed JSON, but an invalid `winner` -- PairwiseJudgment's own
    `__post_init__` rejects it, and parse_verdict never lets that ValueError
    escape as anything but JudgeUnavailableError."""
    text = (
        '{"winner": "C", "margin": "moderate", '
        '"dimension_scores": {"correctness": 5, "completeness": 5, '
        '"instruction_following": 5, "format": 5}, "rationale": "x"}'
    )
    with pytest.raises(JudgeUnavailableError, match="failed verdict validation"):
        parse_verdict(text, cheap_position=PairPosition.A, judge_model="j")


def test_parse_verdict_handles_escaped_quotes_inside_a_string_value() -> None:
    text = (
        '{"winner": "tie", "margin": "negligible", '
        '"dimension_scores": {"correctness": 4, "completeness": 4, '
        '"instruction_following": 4, "format": 4}, '
        '"rationale": "Both mention \\"the answer\\" explicitly."}'
    )
    judgment = parse_verdict(text, cheap_position=PairPosition.A, judge_model="j")
    assert judgment.rationale == 'Both mention "the answer" explicitly.'


def test_parse_verdict_raises_judge_unavailable_on_unbalanced_json() -> None:
    text = '{"winner": "tie", "margin": "negligible"'  # missing closing brace
    with pytest.raises(JudgeUnavailableError, match="not parseable"):
        parse_verdict(text, cheap_position=PairPosition.A, judge_model="j")


def test_parse_verdict_tolerates_a_code_fence_with_no_closing_marker() -> None:
    text = '```json\n{"winner": "tie", "margin": "negligible", ' + (
        '"dimension_scores": {"correctness": 4, "completeness": 4, '
        '"instruction_following": 4, "format": 4}, "rationale": "x"}'
    )
    judgment = parse_verdict(text, cheap_position=PairPosition.A, judge_model="j")
    assert judgment.winner == "tie"


def test_parse_verdict_never_invents_a_verdict_on_malformed_dimension_scores() -> None:
    text = (
        '{"winner": "A", "margin": "moderate", '
        '"dimension_scores": "not an object", "rationale": "x"}'
    )
    with pytest.raises(JudgeUnavailableError):
        parse_verdict(text, cheap_position=PairPosition.A, judge_model="j")
