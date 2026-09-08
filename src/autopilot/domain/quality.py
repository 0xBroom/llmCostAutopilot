"""Quality-contract value objects and deterministic checks.

Pure domain: a `Verdict` is a value produced from a `CompletionRequest` and the
`LLMResponse` that answered it, plus (optionally) a pair of judged answers. No
I/O happens in this module. The deterministic checks below run at zero LLM
cost and always precede any judge call — a failure here short-circuits the
whole quality contract at confidence 1.0, exactly like a routing failure.

Judge selection (`select_judge`), the pass/fail policy (`resolve_verdict_label`),
and the orchestrating `QualityJudge` application service are later slices —
this module only defines the shapes and the free checks they will use.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Literal
from uuid import UUID

from autopilot.domain.errors import JudgeUnavailableError

if TYPE_CHECKING:
    from autopilot.domain.catalog import ModelCatalog
    from autopilot.domain.models import LLMResponse, Message, ModelConfig


class VerdictLabel(StrEnum):
    """The resolved outcome of a quality contract check. `TIE` is a PASS."""

    PASS = "pass"  # noqa: S105 -- an outcome label, not a credential
    FAIL = "fail"
    TIE = "tie"


class JudgeIndependence(StrEnum):
    """How independent the selected judge was from the reference model.

    Recorded on every `Verdict` so flip-rate and agreement can be subset by
    independence rather than pretending every judge call was equally clean.
    Selection logic that produces this value is a later slice; this module
    only defines the type it is carried on.
    """

    INDEPENDENT = "independent"
    DEGRADED_LOCAL = "degraded_local"
    UNAVAILABLE = "unavailable"


class TaskType(StrEnum):
    """The closed set of task shapes a quality profile can be configured for.

    `DEFAULT` is the anchor profile applied when a request carries no
    explicit `task_type`. Never LLM-inferred — a client either states its
    task type or gets the default profile.
    """

    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    DEFAULT = "default"


class JudgeMode(StrEnum):
    """Whether a task type's profile decides on deterministic checks alone,
    or additionally runs a pairwise judge comparison."""

    DETERMINISTIC = "deterministic"
    PAIRWISE = "pairwise"


class PairPosition(StrEnum):
    """Which side of a blind pairwise comparison an answer was shown at."""

    A = "a"
    B = "b"


DIMENSIONS: Final[tuple[str, ...]] = (
    "correctness",
    "completeness",
    "instruction_following",
    "format",
)
"""The exact, closed set of dimensions a pairwise judgment scores. A
`dimension_scores` mapping with any other key set is rejected — see
`PairwiseJudgment.__post_init__` and `Verdict.__post_init__`."""

MARGIN_ORDER: Final[tuple[str, ...]] = ("negligible", "moderate", "decisive")
"""Ordered, not just a membership set: the margin-gated pass/fail policy
(a later slice) compares positions in this tuple, never `!=`, so a future
sub-moderate level can be inserted without silently changing existing
threshold semantics."""

JUDGE_PROMPT_VERSION: Final[str] = "qj-1.0"
"""Bump on any rubric-wording change. Recorded on every judgment/verdict so a
scoring drift can be traced to the prompt version that produced it."""

_ALLOWED_WINNERS: Final[frozenset[str]] = frozenset({"A", "B", "tie"})

RATIONALE_MAX_LEN: Final[int] = 400


def _validate_winner(winner: str) -> None:
    if winner not in _ALLOWED_WINNERS:
        raise ValueError(f"winner must be one of {sorted(_ALLOWED_WINNERS)}, got {winner!r}")


def _validate_margin(margin: str) -> None:
    if margin not in MARGIN_ORDER:
        raise ValueError(f"margin must be one of {MARGIN_ORDER}, got {margin!r}")


def _validate_dimension_scores(dimension_scores: Mapping[str, int]) -> None:
    if frozenset(dimension_scores) != frozenset(DIMENSIONS):
        raise ValueError(
            f"dimension_scores keys must be exactly {DIMENSIONS}, got {tuple(dimension_scores)}"
        )
    for dimension, score in dimension_scores.items():
        if not 1 <= score <= 5:
            raise ValueError(f"dimension {dimension!r} score must be 1-5, got {score}")


def _validate_rationale(rationale: str) -> None:
    if len(rationale) > RATIONALE_MAX_LEN:
        raise ValueError(f"rationale must be <= {RATIONALE_MAX_LEN} chars, got {len(rationale)}")


@dataclass(frozen=True, slots=True)
class DeterministicCheckResult:
    """The outcome of one free, zero-LLM-cost check."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class DeterministicOutcome:
    """The full suite's outcome. `passed` is true only if every check passed."""

    results: tuple[DeterministicCheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


@dataclass(frozen=True, slots=True)
class PairwiseJudgment:
    """The structured output of ONE judge call.

    A `Verdict` (below) carries zero or two of these — two because the judge
    is called once per swapped position to measure position bias, per the
    later `QualityJudge` orchestration.
    """

    position_of_cheap_answer: PairPosition
    winner: Literal["A", "B", "tie"]
    margin: Literal["negligible", "moderate", "decisive"]
    dimension_scores: Mapping[str, int]
    rationale: str
    judge_model: str
    judge_prompt_version: str

    def __post_init__(self) -> None:
        _validate_winner(self.winner)
        _validate_margin(self.margin)
        _validate_dimension_scores(self.dimension_scores)
        _validate_rationale(self.rationale)


@dataclass(frozen=True, slots=True)
class Verdict:
    """The resolved quality outcome for one served request.

    Storage is a later concern (#17); this is the value that concern will
    persist. Judge selection and the pass/fail policy that fill in
    `judge_model`, `judge_independence`, `winner`, `margin` etc. from a real
    comparison are later slices — this type only fixes the shape and its
    invariants so those slices have a value object to fill in and validate.
    """

    request_id: UUID
    label: VerdictLabel
    task_type: TaskType
    deterministic: DeterministicOutcome
    judged_by_llm: bool
    judge_model: str | None
    reference_model: str | None
    judgments: tuple[PairwiseJudgment, ...]
    judge_independence: JudgeIndependence
    flip_detected: bool | None
    winner: Literal["A", "B", "tie"]
    margin: Literal["negligible", "moderate", "decisive"]
    dimension_scores: Mapping[str, int]
    rationale: str
    position_of_cheap_answer: PairPosition | None
    judge_prompt_version: str | None
    reason: str

    def __post_init__(self) -> None:
        _validate_winner(self.winner)
        _validate_margin(self.margin)
        _validate_dimension_scores(self.dimension_scores)
        _validate_rationale(self.rationale)

    @property
    def is_pass(self) -> bool:
        return self.label in {VerdictLabel.PASS, VerdictLabel.TIE}


# --- deterministic checks -------------------------------------------------
#
# Each check is pure and free: no network call, no LLM invocation. A failure
# here means confidence 1.0 — there is nothing probabilistic about "the JSON
# does not parse" or "the label is not in the allowed set". `run_deterministic
# _checks` short-circuits on the first failure so a caller never pays for a
# check that cannot change an already-decided outcome.


def check_json_valid(response: LLMResponse) -> DeterministicCheckResult:
    """The response content MUST parse as JSON."""
    try:
        json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        return DeterministicCheckResult(
            name="json_valid", passed=False, detail="response content is not valid JSON"
        )
    return DeterministicCheckResult(name="json_valid", passed=True, detail="")


def check_required_fields(
    response: LLMResponse, required: tuple[str, ...]
) -> DeterministicCheckResult:
    """Every field in `required` MUST be present and non-null in the parsed JSON object."""
    try:
        payload = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        return DeterministicCheckResult(
            name="required_fields", passed=False, detail="response content is not valid JSON"
        )
    if not isinstance(payload, Mapping):
        return DeterministicCheckResult(
            name="required_fields",
            passed=False,
            detail="response content is not a JSON object",
        )
    missing = [field for field in required if payload.get(field) is None]
    if missing:
        return DeterministicCheckResult(
            name="required_fields",
            passed=False,
            detail=f"missing required field(s): {', '.join(missing)}",
        )
    return DeterministicCheckResult(name="required_fields", passed=True, detail="")


def check_label_in_set(response: LLMResponse, allowed: frozenset[str]) -> DeterministicCheckResult:
    """The (trimmed) response content MUST be one of the configured allowed labels."""
    label = response.content.strip()
    if label not in allowed:
        return DeterministicCheckResult(
            name="label_in_set",
            passed=False,
            detail=f"label {label!r} is not in the allowed set",
        )
    return DeterministicCheckResult(name="label_in_set", passed=True, detail="")


_REFUSAL_MARKERS: Final[tuple[str, ...]] = (
    "i cannot help with that",
    "i can't help with that",
    "i cannot assist with that request",
    "i'm unable to assist with that",
    "i am unable to assist with that",
)


def check_not_refused(response: LLMResponse) -> DeterministicCheckResult:
    """The response MUST NOT be empty and MUST NOT read as an outright refusal."""
    content = response.content.strip()
    if not content:
        return DeterministicCheckResult(
            name="not_refused", passed=False, detail="response content is empty"
        )
    lowered = content.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        return DeterministicCheckResult(
            name="not_refused", passed=False, detail="response content reads as a refusal"
        )
    return DeterministicCheckResult(name="not_refused", passed=True, detail="")


def check_not_truncated(response: LLMResponse) -> DeterministicCheckResult:
    """`finish_reason` MUST NOT indicate the response was cut off mid-stream."""
    if response.was_truncated:
        return DeterministicCheckResult(
            name="not_truncated",
            passed=False,
            detail=f"finish_reason={response.finish_reason!r} indicates truncation",
        )
    return DeterministicCheckResult(name="not_truncated", passed=True, detail="")


def check_length_anomaly(
    response: LLMResponse, bounds: tuple[int, int]
) -> DeterministicCheckResult:
    """The response length MUST fall within the configured `(min, max)` character bounds."""
    minimum, maximum = bounds
    length = len(response.content)
    if length < minimum or length > maximum:
        return DeterministicCheckResult(
            name="length_anomaly",
            passed=False,
            detail=f"response length {length} is outside bounds [{minimum}, {maximum}]",
        )
    return DeterministicCheckResult(name="length_anomaly", passed=True, detail="")


def run_deterministic_checks(
    response: LLMResponse,
    checks: Sequence[Callable[[LLMResponse], DeterministicCheckResult]],
) -> DeterministicOutcome:
    """Run `checks` in order, stopping at the first failure.

    A failed check already decides the outcome at confidence 1.0, so nothing
    is gained — and LLM budget could be lost, once a caller wires a judge
    call after this — by running the remaining checks. The short-circuit is
    the zero-cost guarantee the quality contract makes.
    """
    results: list[DeterministicCheckResult] = []
    for check in checks:
        result = check(response)
        results.append(result)
        if not result.passed:
            break
    return DeterministicOutcome(results=tuple(results))


# --- judge selection ---------------------------------------------------------
#
# Pure: reads only the catalog's public views (`.enabled`, `.judge`). Provider
# is "family" for this purpose. The reference model is excluded by identity in
# every branch below, on top of being excluded by family via `forbidden` —
# belt and braces, because "the judge is never the reference" is exactly the
# invariant a later regression must never quietly break.


@dataclass(frozen=True, slots=True)
class JudgeSelection:
    """The outcome of `select_judge`.

    `judge_model` is `None` only when `independence` is `UNAVAILABLE` — no
    enabled model exists from a family different than both the reference and
    the candidate, so the judge is simply not run.
    """

    judge_model: ModelConfig | None
    independence: JudgeIndependence
    reason: str


def _judge_rank_key(model: ModelConfig) -> tuple[int, str]:
    """Tier-descending, then key. The strongest qualifying model sorts first.

    Mirrors `ModelConfig.fallback_sort_key`'s "tier, then a tiebreaker, then
    key" shape, but ranks the opposite direction: `fallback_chain` wants the
    closest/cheapest fallback, this wants the STRONGEST qualifying judge.
    """
    return (-int(model.quality_tier), model.key)


def select_judge(
    catalog: ModelCatalog, *, reference: ModelConfig, candidate: ModelConfig
) -> JudgeSelection:
    """Choose an independent judge for one comparison.

    Sits ABOVE the catalog's own exactly-one-judge invariant: that static
    `judge` designation is the default/anchor, never mutated here — mirrors
    `ModelCatalog.fallback_chain`, a pure ranking derived from tier, never
    stored. Fallback order, per Design Decision Q2:

      1. the static judge, if its family qualifies             -> INDEPENDENT
      2. else the strongest qualifying CLOUD (non-ollama) model -> INDEPENDENT
      3. else the only different-family model is a local model  -> DEGRADED_LOCAL
      4. else no different-family model exists at all           -> UNAVAILABLE
    """
    forbidden = {reference.provider, candidate.provider}
    qualifying = tuple(
        model
        for model in catalog.enabled
        if model.provider not in forbidden and model.key != reference.key
    )

    if not qualifying:
        return JudgeSelection(
            judge_model=None,
            independence=JudgeIndependence.UNAVAILABLE,
            reason=(
                "no enabled model belongs to a provider family different from "
                "both the reference and the candidate; judge not run"
            ),
        )

    static_judge = catalog.judge
    if static_judge in qualifying:
        return JudgeSelection(
            judge_model=static_judge,
            independence=JudgeIndependence.INDEPENDENT,
            reason="the catalog's static judge belongs to a qualifying family",
        )

    ranked = sorted(qualifying, key=_judge_rank_key)
    cloud_ranked = [model for model in ranked if model.provider != "ollama"]
    if cloud_ranked:
        return JudgeSelection(
            judge_model=cloud_ranked[0],
            independence=JudgeIndependence.INDEPENDENT,
            reason="the static judge does not qualify; using the strongest qualifying cloud model",
        )

    return JudgeSelection(
        judge_model=ranked[0],
        independence=JudgeIndependence.DEGRADED_LOCAL,
        reason=(
            "only a local model qualifies as a different-family judge; independence is degraded"
        ),
    )


# --- pass/fail policy ---------------------------------------------------------


def resolve_verdict_label(
    *,
    deterministic_passed: bool,
    mode: JudgeMode,
    reference_won: bool,
    margin: str,
    fail_reference_margin_min: str | None,
) -> VerdictLabel:
    """Margin-gated pass/fail. Pure — thresholds are data, this is policy.

    A deterministic failure always decides the outcome, before any judge
    result is even consulted. A tie or a candidate win is always a PASS. When
    the reference wins, FAIL iff the margin is at least as large as
    `fail_reference_margin_min`, compared by ORDERED position in
    `MARGIN_ORDER` — never `!=` — so a future sub-moderate level does not
    silently change an existing threshold's meaning (Design Decision Q5).
    """
    if not deterministic_passed:
        return VerdictLabel.FAIL
    if mode is JudgeMode.DETERMINISTIC:
        return VerdictLabel.PASS
    if not reference_won:
        return VerdictLabel.PASS
    if fail_reference_margin_min is None:
        raise ValueError("fail_reference_margin_min is required when mode is PAIRWISE")
    _validate_margin(margin)
    _validate_margin(fail_reference_margin_min)
    if MARGIN_ORDER.index(margin) >= MARGIN_ORDER.index(fail_reference_margin_min):
        return VerdictLabel.FAIL
    return VerdictLabel.PASS


# --- combining a position-swapped pair of judgments ---------------------------


def _judgment_outcome(judgment: PairwiseJudgment) -> Literal["candidate", "reference", "tie"]:
    """Translate one call's position-relative `winner` (A/B/tie) into the
    position-INDEPENDENT frame the pass/fail policy needs: did the candidate
    (the cheap answer) win, did the reference win, or was it a tie?"""
    if judgment.winner == "tie":
        return "tie"
    winning_position = PairPosition.A if judgment.winner == "A" else PairPosition.B
    return "candidate" if winning_position == judgment.position_of_cheap_answer else "reference"


def combine_swapped_judgments(
    first: PairwiseJudgment, second: PairwiseJudgment
) -> tuple[Literal["A", "B", "tie"], Literal["negligible", "moderate", "decisive"], bool, bool]:
    """Combine two position-swapped judgments of the SAME comparison into the
    `(winner, margin, reference_won, flip_detected)` a `Verdict` needs.

    Returns `winner`/`margin` expressed relative to `first.position_of_cheap
    _answer` — the position a `Verdict` itself records — so a caller never has
    to re-derive which position a Verdict-level `winner` refers to.

    Disagreement is measured on the POSITION-INDEPENDENT outcome (candidate /
    reference / tie), never on the raw A/B label: the two calls swap which
    position holds the cheap answer, so comparing raw labels directly would
    call a "flip" on every ordinary, unbiased swap. A judge that reports the
    SAME raw position twice (e.g. "A" both times) is exactly the position-bias
    case this is meant to catch — it means the position it preferred did not
    track the content the swap moved between positions.
    """
    first_outcome = _judgment_outcome(first)
    second_outcome = _judgment_outcome(second)

    if first_outcome != second_outcome:
        return "tie", "negligible", False, True

    reference_won = first_outcome == "reference"
    canonical_position = first.position_of_cheap_answer
    winner: Literal["A", "B", "tie"]
    if first_outcome == "tie":
        winner = "tie"
    elif first_outcome == "candidate":
        winner = "A" if canonical_position is PairPosition.A else "B"
    else:
        winner = "B" if canonical_position is PairPosition.A else "A"

    return winner, first.margin, reference_won, False


# --- anchored rubric prompt + defensive parsing -------------------------------
#
# Note (orchestrator flag): the anchored rubric content below is normally a
# Phase 3 task (tasks.md 3.7), pulled forward into this slice on explicit
# instruction so `build_judge_prompt` is never left unanchored between PR 2
# and PR 3.

_DIMENSION_ANCHORS: Final[Mapping[str, Mapping[int, str]]] = {
    "correctness": {
        1: "Contains factual errors, or contradicts a fact stated in the prompt.",
        3: "Mostly correct, but has one minor inaccuracy or an unsupported claim.",
        5: "Every factual claim is accurate and consistent with the prompt.",
    },
    "completeness": {
        1: "Ignores most of what the prompt actually asked for.",
        3: "Covers the main ask, but omits a secondary part of the request.",
        5: "Fully addresses every part of the prompt, nothing left out.",
    },
    "instruction_following": {
        1: "Disregards an explicit formatting or behavioral instruction in the prompt.",
        3: "Follows most instructions, but deviates on one explicit constraint.",
        5: "Follows every explicit instruction in the prompt exactly.",
    },
    "format": {
        1: "Output structure is unusable: wrong shape, unreadable, or missing required fields.",
        3: "Structure is mostly right, but has a formatting inconsistency.",
        5: "Output matches the expected structure and shape exactly.",
    },
}


def _render_rubric() -> str:
    lines = [f"Rubric ({JUDGE_PROMPT_VERSION}). Score each dimension 1-5 using these anchors:"]
    for dimension in DIMENSIONS:
        anchors = _DIMENSION_ANCHORS[dimension]
        lines.append(f"- {dimension}:")
        lines.append(f"    1 = {anchors[1]}")
        lines.append(f"    3 = {anchors[3]}")
        lines.append(f"    5 = {anchors[5]}")
    return "\n".join(lines)


_JUDGE_RESPONSE_SCHEMA: Final[str] = (
    '{"winner": "A"|"B"|"tie", "margin": "negligible"|"moderate"|"decisive", '
    '"dimension_scores": {"correctness": 1-5, "completeness": 1-5, '
    '"instruction_following": 1-5, "format": 1-5}, "rationale": "<= 400 chars"}'
)


def build_judge_prompt(
    reference_answer: str,
    candidate_answer: str,
    *,
    cheap_position: PairPosition,
    task_type: TaskType,
) -> tuple[Message, ...]:
    """Build the blind, anchored-rubric pairwise comparison prompt.

    `cheap_position` decides which of Answer A / Answer B is the candidate
    (cheap) answer — the caller swaps this across the two calls of one
    comparison to measure position bias (`combine_swapped_judgments`).
    """
    from autopilot.domain.models import Message  # deferred: avoids the models<->quality cycle

    if cheap_position is PairPosition.A:
        answer_a, answer_b = candidate_answer, reference_answer
    else:
        answer_a, answer_b = reference_answer, candidate_answer

    system = Message(
        role="system",
        content=(
            "You are a blind quality judge comparing two candidate answers to "
            f"the same {task_type.value} prompt. You do not know which answer "
            "came from which model. Judge only the two answers below.\n\n"
            f"{_render_rubric()}\n\n"
            "Respond with ONLY a single JSON object, no other text, no markdown "
            f"fences: {_JUDGE_RESPONSE_SCHEMA}"
        ),
    )
    user = Message(role="user", content=f"Answer A:\n{answer_a}\n\nAnswer B:\n{answer_b}")
    return (system, user)


def _extract_json_object(text: str) -> Mapping[str, Any]:
    """Extract the first balanced top-level JSON object in `text`, tolerating
    surrounding markdown code fences. Raises `ValueError` if none is found —
    never returns a guess."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start == -1:
        raise ValueError("no JSON object found in judge output")

    depth = 0
    in_string = False
    escape = False
    end: int | None = None
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end is None:
        raise ValueError("unbalanced JSON object in judge output")

    parsed = json.loads(stripped[start:end])
    if not isinstance(parsed, Mapping):
        raise ValueError("parsed JSON is not an object")
    return parsed


def parse_verdict(
    text: str,
    *,
    cheap_position: PairPosition,
    judge_model: str,
    judge_prompt_version: str = JUDGE_PROMPT_VERSION,
) -> PairwiseJudgment:
    """Defensively parse one judge call's raw text into a `PairwiseJudgment`.

    Never trusts that JSON mode was honoured: extracts the first balanced
    JSON object, tolerating markdown code fences. Raises
    `JudgeUnavailableError` on anything that does not parse or does not
    validate — the verdict is never invented (see `domain.errors
    .JudgeUnavailableError`).
    """
    try:
        payload = _extract_json_object(text)
    except ValueError as exc:
        raise JudgeUnavailableError(f"judge output was not parseable: {exc}") from exc

    try:
        winner = payload["winner"]
        margin = payload["margin"]
        dimension_scores = payload["dimension_scores"]
        rationale = payload["rationale"]
    except (KeyError, TypeError) as exc:
        raise JudgeUnavailableError(f"judge output missing a required field: {exc}") from exc

    if not isinstance(dimension_scores, Mapping):
        raise JudgeUnavailableError("judge output's dimension_scores is not an object")

    try:
        return PairwiseJudgment(
            position_of_cheap_answer=cheap_position,
            winner=winner,
            margin=margin,
            dimension_scores=dimension_scores,
            rationale=rationale,
            judge_model=judge_model,
            judge_prompt_version=judge_prompt_version,
        )
    except (ValueError, TypeError) as exc:
        raise JudgeUnavailableError(f"judge output failed verdict validation: {exc}") from exc
