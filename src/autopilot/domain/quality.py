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
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID

if TYPE_CHECKING:
    from autopilot.domain.models import LLMResponse


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
