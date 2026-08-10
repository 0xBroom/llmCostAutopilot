"""Provider baseline harness (#9): `run_matrix`'s failure survival and
concurrency bound, the pure aggregation helpers, the artifact writers, the
pre-flight spend gate and its cost estimate, and corpus loading.

Every scenario here uses `FakeLLMGateway` or a thin wrapper around it —
composed, never modified, per the fake's own docstring on why it is
hand-written and checked against `LLMGateway` by mypy.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from autopilot.domain.errors import ProviderUnavailableError
from autopilot.domain.models import CompletionRequest, ModelConfig
from autopilot.interfaces.cli.baseline import (
    CallResult,
    ModelStats,
    Prompt,
    aggregate_by_model,
    baseline_output_dir,
    check_spend_gate,
    estimate_run_cost,
    load_prompts,
    output_token_ratio,
    percentile,
    run_matrix,
    write_artifacts,
    write_outputs_md,
    write_raw_jsonl,
    write_summary_md,
)
from tests.factories import CHEAP, EXPENSIVE, LOCAL
from tests.fakes import FrozenClock
from tests.fakes.gateway import FakeLLMGateway

# --- test helpers ---------------------------------------------------------------


def _prompt(*, id: str = "p1", tier: int | None = 1, max_tokens: int = 64) -> Prompt:
    return Prompt(
        id=id, tier=tier, category="test", text=f"prompt text for {id}", max_tokens=max_tokens
    )


def _call(
    *,
    prompt_id: str = "p1",
    model_key: str = "m",
    tier: int = 1,
    repeat_index: int = 0,
    ok: bool = True,
    output: str | None = "ans",
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 20,
    total_cost: Decimal = Decimal("0.01"),
    latency_ms: int | None = 100,
    error: str | None = None,
) -> CallResult:
    return CallResult(
        prompt_id=prompt_id,
        model_key=model_key,
        tier=tier,
        repeat_index=repeat_index,
        ok=ok,
        output=output,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_cost=total_cost,
        latency_ms=latency_ms,
        error=error,
    )


class _ConcurrencyTrackingGateway:
    """Wraps a `FakeLLMGateway` by composition, recording the maximum number
    of calls in flight at once. Never subclasses or mutates the fake — the
    point of the composition is to leave `tests/fakes/gateway.py` untouched.
    """

    def __init__(self, inner: FakeLLMGateway, *, delay_s: float = 0.02) -> None:
        self._inner = inner
        self._delay_s = delay_s
        self._lock = asyncio.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(
        self, request: CompletionRequest, model: ModelConfig, *, timeout_s: float
    ) -> object:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self._delay_s)
            return await self._inner.complete(request, model, timeout_s=timeout_s)
        finally:
            async with self._lock:
                self.in_flight -= 1


# --- run_matrix: execution and failure survival ---------------------------------


async def test_run_matrix_executes_every_prompt_model_repeat_cell() -> None:
    gateway = FakeLLMGateway()
    models = [LOCAL, CHEAP, EXPENSIVE]
    prompts = [_prompt(id="p1"), _prompt(id="p2")]

    results = await run_matrix(
        gateway=gateway, models=models, prompts=prompts, repeats=2, timeout_s=5.0, max_concurrency=4
    )

    assert len(results) == len(models) * len(prompts) * 2
    assert gateway.call_count == len(results)
    assert all(r.ok for r in results)


async def test_run_matrix_survives_a_failure_on_one_model_without_aborting() -> None:
    gateway = FakeLLMGateway(
        errors={CHEAP.key: ProviderUnavailableError("provider down", model_key=CHEAP.key)}
    )
    models = [LOCAL, CHEAP]
    prompts = [_prompt(id="p1")]

    results = await run_matrix(
        gateway=gateway, models=models, prompts=prompts, repeats=3, timeout_s=5.0, max_concurrency=4
    )

    assert len(results) == 6  # matrix length unchanged despite the failure
    cheap_results = [r for r in results if r.model_key == CHEAP.key]
    local_results = [r for r in results if r.model_key == LOCAL.key]
    assert len(cheap_results) == 3
    assert len(local_results) == 3
    assert all(not r.ok and r.error is not None and r.output is None for r in cheap_results)
    assert all(r.ok and r.error is None for r in local_results)


async def test_run_matrix_survives_a_timeout_error() -> None:
    gateway = FakeLLMGateway(errors={CHEAP.key: TimeoutError("provider took too long")})
    results = await run_matrix(
        gateway=gateway,
        models=[CHEAP],
        prompts=[_prompt()],
        repeats=1,
        timeout_s=5.0,
        max_concurrency=4,
    )

    assert len(results) == 1
    assert results[0].ok is False
    assert "too long" in (results[0].error or "")


async def test_run_matrix_bounds_concurrency() -> None:
    inner = FakeLLMGateway()
    tracker = _ConcurrencyTrackingGateway(inner, delay_s=0.02)
    models = [LOCAL, CHEAP, EXPENSIVE]
    prompts = [_prompt(id="p1"), _prompt(id="p2"), _prompt(id="p3")]

    await run_matrix(
        gateway=tracker,  # type: ignore[arg-type]
        models=models,
        prompts=prompts,
        repeats=2,
        timeout_s=5.0,
        max_concurrency=3,
    )

    assert tracker.max_in_flight <= 3
    assert tracker.max_in_flight >= 2  # proves the bound was actually exercised, not serial


# --- percentile helper ------------------------------------------------------------


def test_percentile_p50_on_a_known_list() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0


def test_percentile_p95_on_a_known_list() -> None:
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


def test_percentile_of_empty_sequence_is_zero() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_of_single_value() -> None:
    assert percentile([7.0], 90) == 7.0


# --- output-token ratio -----------------------------------------------------------


def _stats(*, model_key: str, mean_tokens: float, p95_tokens: float) -> ModelStats:
    return ModelStats(
        model_key=model_key,
        tier=1,
        p50_latency_ms=100.0,
        p95_latency_ms=150.0,
        avg_cost_per_req=Decimal("0.01"),
        total_tokens_out=int(mean_tokens),
        mean_completion_tokens=mean_tokens,
        p95_completion_tokens=p95_tokens,
        failures=0,
        total_calls=3,
    )


def test_output_token_ratio_of_baseline_against_itself_is_one() -> None:
    baseline = _stats(model_key="baseline", mean_tokens=80.0, p95_tokens=90.0)
    assert output_token_ratio(baseline, baseline) == (1.0, 1.0)


def test_output_token_ratio_computed_against_baseline() -> None:
    baseline = _stats(model_key="baseline", mean_tokens=100.0, p95_tokens=120.0)
    other = _stats(model_key="other", mean_tokens=50.0, p95_tokens=60.0)
    assert output_token_ratio(other, baseline) == (0.5, 0.5)


# --- aggregate_by_model -----------------------------------------------------------


def test_aggregate_by_model_groups_and_computes_failures() -> None:
    results = [
        _call(model_key="a", repeat_index=0, ok=True, latency_ms=100, completion_tokens=20),
        _call(model_key="a", repeat_index=1, ok=True, latency_ms=200, completion_tokens=30),
        _call(
            model_key="a",
            repeat_index=2,
            ok=False,
            error="boom",
            latency_ms=None,
            completion_tokens=None,
        ),
        _call(model_key="b", repeat_index=0, ok=True, latency_ms=50, completion_tokens=10),
    ]

    stats = aggregate_by_model(results)
    by_key = {s.model_key: s for s in stats}

    assert set(by_key) == {"a", "b"}
    assert by_key["a"].total_calls == 3
    assert by_key["a"].failures == 1
    assert by_key["a"].total_tokens_out == 50  # only the two ok calls
    assert by_key["b"].failures == 0
    assert by_key["b"].total_calls == 1


# --- estimate_run_cost -------------------------------------------------------------


def test_estimate_run_cost_is_a_conservative_upper_bound() -> None:
    # $1/token in and out makes the arithmetic legible.
    model = replace(CHEAP, input_cost_per_token=Decimal("1"), output_cost_per_token=Decimal("1"))
    prompt = _prompt(id="p1", max_tokens=10)  # text "prompt text for p1" is 18 chars
    est = estimate_run_cost(models=[model], prompts=[prompt], repeats=2)
    # input ceil(18/4)=5 * $1, output 10 * $1 -> $15 per call, * 2 repeats -> $30
    assert est == Decimal("30")


def test_estimate_run_cost_ignores_zero_priced_models() -> None:
    free = replace(CHEAP, input_cost_per_token=Decimal("0"), output_cost_per_token=Decimal("0"))
    est = estimate_run_cost(models=[free], prompts=[_prompt()], repeats=3)
    assert est == Decimal("0")


# --- spend gate --------------------------------------------------------------------


def test_spend_gate_blocks_when_over_threshold_and_not_confirmed() -> None:
    error = check_spend_gate(Decimal("2.00"), threshold=Decimal("1.00"), confirmed=False)
    assert error is not None
    assert "confirm-spend" in error


def test_spend_gate_allows_when_over_threshold_and_confirmed() -> None:
    assert check_spend_gate(Decimal("2.00"), threshold=Decimal("1.00"), confirmed=True) is None


def test_spend_gate_allows_when_under_threshold() -> None:
    assert check_spend_gate(Decimal("0.50"), threshold=Decimal("1.00"), confirmed=False) is None


# --- baseline_output_dir -----------------------------------------------------------


def test_baseline_output_dir_derives_the_date_from_the_clock() -> None:
    clock = FrozenClock()  # 2026-01-01
    result = baseline_output_dir(Path("/artifacts"), clock=clock)
    assert result == Path("/artifacts/baseline/2026-01-01")


def test_baseline_output_dir_accepts_an_injected_iso_date() -> None:
    clock = FrozenClock()
    result = baseline_output_dir(Path("/artifacts"), clock=clock, iso_date="2020-05-05")
    assert result == Path("/artifacts/baseline/2020-05-05")


# --- artifact writers ---------------------------------------------------------------


def test_write_raw_jsonl_one_line_per_result_with_decimal_as_string(tmp_path: Path) -> None:
    results = [
        _call(model_key="a", total_cost=Decimal("0.0000123")),
        _call(model_key="b", ok=False, output=None, total_cost=Decimal("0")),
    ]
    path = tmp_path / "raw.jsonl"

    write_raw_jsonl(results, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(results)
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["model_key"] == "a"
    assert parsed[0]["total_cost"] == "0.0000123"
    assert isinstance(parsed[0]["total_cost"], str)
    assert parsed[1]["ok"] is False


def test_write_summary_md_has_exact_header_and_total_line(tmp_path: Path) -> None:
    baseline_stats = _stats(model_key="gpt-4o", mean_tokens=100.0, p95_tokens=120.0)
    other_stats = _stats(model_key="haiku", mean_tokens=50.0, p95_tokens=60.0)
    path = tmp_path / "summary.md"

    write_summary_md(
        [baseline_stats, other_stats],
        baseline_key="gpt-4o",
        total_spend=Decimal("3.140000"),
        path=path,
    )

    content = path.read_text(encoding="utf-8")
    assert (
        "| model | tier | p50 latency | p95 latency | avg cost/req | "
        "total tokens out | out-tok ratio (mean/p95) | failures |"
    ) in content
    assert "TOTAL run spend: $3.140000" in content
    assert "gpt-4o" in content
    assert "haiku" in content


def test_write_outputs_md_has_a_section_per_prompt_and_the_learnings_template(
    tmp_path: Path,
) -> None:
    prompts = [_prompt(id="p1"), _prompt(id="p2")]
    results = [
        _call(prompt_id="p1", model_key="a", repeat_index=0, output="answer-a"),
        _call(prompt_id="p1", model_key="b", repeat_index=0, ok=False, output=None, error="boom"),
        _call(prompt_id="p2", model_key="a", repeat_index=0, output="answer-p2"),
    ]
    path = tmp_path / "outputs.md"

    write_outputs_md(results, prompts, path=path)

    content = path.read_text(encoding="utf-8")
    assert "## p1" in content
    assert "## p2" in content
    assert "answer-a" in content
    assert "FAILED: boom" in content
    assert "## What we learned" in content
    assert content.count("TODO") >= 2


def test_write_artifacts_creates_the_expected_files(tmp_path: Path) -> None:
    prompts = [_prompt(id="p1")]
    results = [_call(prompt_id="p1", model_key="gpt-4o", tier=3)]
    stats = aggregate_by_model(results)
    output_dir = tmp_path / "baseline" / "2026-01-01"

    write_artifacts(
        results,
        stats,
        prompts,
        baseline_key="gpt-4o",
        total_spend=Decimal("0.01"),
        output_dir=output_dir,
    )

    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "outputs.md").exists()


# --- load_prompts -----------------------------------------------------------------


def test_load_prompts_resolves_inline_and_prompt_file_entries(tmp_path: Path) -> None:
    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    (fixture / "doc.txt").write_text("the long document body", encoding="utf-8")

    corpus = tmp_path / "prompts.yaml"
    corpus.write_text(
        """
version: 1
defaults:
  max_tokens: 256
  temperature: 0
prompts:
  - id: inline-one
    tier: 1
    category: reformat
    prompt: "an inline prompt"
  - id: file-one
    tier: 3
    category: long-context
    prompt_file: fixtures/doc.txt
    max_tokens: 999
  - id: overflow-one
    tier: null
    category: overflow
    prompt_file: fixtures/doc.txt
""",
        encoding="utf-8",
    )

    prompts = load_prompts(corpus)
    by_id = {p.id: p for p in prompts}

    assert by_id["inline-one"].text == "an inline prompt"
    assert by_id["inline-one"].max_tokens == 256  # falls back to defaults.max_tokens
    assert by_id["file-one"].text == "the long document body"
    assert by_id["file-one"].max_tokens == 999  # explicit override wins
    assert by_id["overflow-one"].tier is None


def test_load_prompts_corpus_loads_from_the_real_config_file() -> None:
    """The actual shipped corpus (#9's 14 prompts) must load without error."""
    path = Path(__file__).resolve().parents[2] / "config" / "baseline_prompts.yaml"
    prompts = load_prompts(path)

    assert len(prompts) == 14
    ids = {p.id for p in prompts}
    assert "of1-overflow" in ids
    assert "lc1-synthesis" in ids
    overflow = next(p for p in prompts if p.id == "of1-overflow")
    assert overflow.tier is None
    assert len(overflow.text) > 40_000
