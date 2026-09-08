"""`QualityJudge` — orchestrates the quality contract for one served request.

Ships "assembled but not wired", mirroring `application.budget.DailySpendGuard`:
ports and plain values only, never `Settings` — the `config-is-a-leaf`
import-linter contract forbids `application` from importing `autopilot.config`.
`checks`, `mode`, and `fail_reference_margin_min` are supplied per call to
`evaluate()`, not fixed on the instance: they come from a task type's
profile, and loading that profile from YAML is a later slice's concern
(`infrastructure.quality_thresholds_loader`, Phase 3) — this service knows
nothing about config.

Judge selection, rubric-prompt building, verdict parsing, and the pass/fail
policy are ALL pure domain functions (`domain.quality`); this service only
orchestrates the I/O around them: run the free deterministic checks, gate on
sampling and mode, pick an independent judge, re-run the reference prompt on
the tier-3 model, call the judge twice with the cheap answer's position
swapped, and resolve the two judgments into one `Verdict`.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from autopilot.application.ports import Clock, LLMGateway
from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.models import CompletionRequest, LLMResponse, ModelConfig
from autopilot.domain.quality import (
    DIMENSIONS,
    JUDGE_PROMPT_VERSION,
    DeterministicCheckResult,
    DeterministicOutcome,
    JudgeIndependence,
    JudgeMode,
    PairPosition,
    PairwiseJudgment,
    Verdict,
    VerdictLabel,
    build_judge_prompt,
    combine_swapped_judgments,
    parse_verdict,
    resolve_verdict_label,
    run_deterministic_checks,
    select_judge,
)


def _neutral_dimension_scores() -> dict[str, int]:
    """A neutral placeholder for the Verdicts that never reach a judge call —
    there is no judgment to score, only a reason."""
    return dict.fromkeys(DIMENSIONS, 3)


@dataclass(frozen=True, slots=True)
class QualityJudge:
    """The quality contract's application service."""

    gateway: LLMGateway
    clock: Clock
    catalog: ModelCatalog
    reference_model: ModelConfig
    timeout_s: float = 30.0
    rng: random.Random = field(default_factory=random.Random)

    async def evaluate(
        self,
        *,
        request: CompletionRequest,
        response: LLMResponse,
        candidate_model: ModelConfig,
        checks: Sequence[Callable[[LLMResponse], DeterministicCheckResult]],
        mode: JudgeMode,
        sampled: bool,
        fail_reference_margin_min: str | None,
    ) -> Verdict:
        """Evaluate one served (request, response) pair.

        Deterministic checks always run first, at zero LLM cost; a failure
        there short-circuits before the judge is ever considered. The judge
        is invoked only when checks pass, the task type's mode is pairwise,
        the request was already sampled, and an independent judge can be
        selected — otherwise a Verdict is returned with `judged_by_llm=False`
        and a `reason` naming exactly which gate stayed closed.
        """
        deterministic = run_deterministic_checks(response, checks)

        if not deterministic.passed:
            return self._short_circuit_verdict(
                request=request,
                deterministic=deterministic,
                independence=JudgeIndependence.UNAVAILABLE,
                reason="a deterministic check failed; the judge was never invoked",
            )

        if mode is JudgeMode.DETERMINISTIC:
            return self._short_circuit_verdict(
                request=request,
                deterministic=deterministic,
                independence=JudgeIndependence.UNAVAILABLE,
                reason=(
                    "deterministic mode: no pairwise comparison is configured for this task type"
                ),
            )

        if not sampled:
            return self._short_circuit_verdict(
                request=request,
                deterministic=deterministic,
                independence=JudgeIndependence.UNAVAILABLE,
                reason="request was not sampled; the judge invocation gate stayed closed",
            )

        selection = select_judge(
            self.catalog, reference=self.reference_model, candidate=candidate_model
        )
        if selection.judge_model is None:
            return self._short_circuit_verdict(
                request=request,
                deterministic=deterministic,
                independence=selection.independence,
                reason=selection.reason,
            )

        reference_response = await self.gateway.complete(
            request, self.reference_model, timeout_s=self.timeout_s
        )

        first_position = PairPosition.A if self.rng.random() < 0.5 else PairPosition.B
        second_position = PairPosition.B if first_position is PairPosition.A else PairPosition.A

        response_format = (
            {"type": "json_object"} if selection.judge_model.supports_json_mode else None
        )

        judgments: list[PairwiseJudgment] = []
        for position in (first_position, second_position):
            messages = build_judge_prompt(
                reference_response.content,
                response.content,
                cheap_position=position,
                task_type=request.task_type,
            )
            judge_request = replace(request, messages=messages)
            judge_response = await self.gateway.complete(
                judge_request,
                selection.judge_model,
                timeout_s=self.timeout_s,
                response_format=response_format,
            )
            judgments.append(
                parse_verdict(
                    judge_response.content,
                    cheap_position=position,
                    judge_model=selection.judge_model.key,
                    judge_prompt_version=JUDGE_PROMPT_VERSION,
                )
            )

        winner, margin, reference_won, flip_detected = combine_swapped_judgments(
            judgments[0], judgments[1]
        )
        label = resolve_verdict_label(
            deterministic_passed=True,
            mode=mode,
            reference_won=reference_won,
            margin=margin,
            fail_reference_margin_min=fail_reference_margin_min,
        )
        representative = judgments[0]

        return Verdict(
            request_id=request.request_id,
            label=label,
            task_type=request.task_type,
            deterministic=deterministic,
            judged_by_llm=True,
            judge_model=selection.judge_model.key,
            reference_model=self.reference_model.key,
            judgments=tuple(judgments),
            judge_independence=selection.independence,
            flip_detected=flip_detected,
            winner=winner,
            margin=margin,
            dimension_scores=representative.dimension_scores,
            rationale=representative.rationale,
            position_of_cheap_answer=representative.position_of_cheap_answer,
            judge_prompt_version=JUDGE_PROMPT_VERSION,
            reason=(
                "judge disagreed across the position swap; resolved to tie"
                if flip_detected
                else "pairwise judge comparison"
            ),
        )

    def _short_circuit_verdict(
        self,
        *,
        request: CompletionRequest,
        deterministic: DeterministicOutcome,
        independence: JudgeIndependence,
        reason: str,
    ) -> Verdict:
        label = VerdictLabel.PASS if deterministic.passed else VerdictLabel.FAIL
        return Verdict(
            request_id=request.request_id,
            label=label,
            task_type=request.task_type,
            deterministic=deterministic,
            judged_by_llm=False,
            judge_model=None,
            reference_model=None,
            judgments=(),
            judge_independence=independence,
            flip_detected=None,
            winner="tie",
            margin="negligible",
            dimension_scores=_neutral_dimension_scores(),
            rationale=reason[:400],
            position_of_cheap_answer=None,
            judge_prompt_version=None,
            reason=reason,
        )
