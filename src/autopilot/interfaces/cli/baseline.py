"""Provider baseline harness (#9).

Runs a fixed prompt corpus across every enabled catalog model, several times
each, and writes latency/cost/token artifacts a human can read before trusting
the router's routing decisions. Invoke as:

    python -m autopilot.interfaces.cli.baseline

Every matrix cell (one prompt, one model, one repeat) is independently
guarded: a domain `AutopilotError` or a timeout is recorded as a failed
`CallResult` rather than aborting the run. One model or one prompt behaving
badly must never cost the harness every other cell's data — that survival
property is the whole point of `run_matrix`, and is exercised directly by
`tests/unit/test_baseline_harness.py`.

**The spend gate is pre-flight.** Before a single provider call is made, the
run's cost is estimated conservatively — input tokens from a character
heuristic (no tokenizer is wired yet; `TokenCounter` is only a port), output
tokens charged at each prompt's `max_tokens` cap rather than the smaller
amount a model actually emits. If that upper bound exceeds
`Settings.baseline_confirm_spend_threshold_usd` without `--confirm-spend`, the
CLI exits non-zero having spent nothing. This refuses the *spend itself*, not
merely the artifacts: `--confirm-spend` then runs the matrix exactly once. (An
earlier design gated *post-hoc*, after the matrix had already spent the money —
which neither prevented the spend nor avoided paying twice on the confirming
re-run.) The actual total is printed next to the estimate on every run, so the
gap between the two is always visible — and is the raw material for tightening
the estimate over time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
import yaml
from structlog.stdlib import BoundLogger

from autopilot.application.ports import Clock, LLMGateway
from autopilot.config.settings import Settings, load_settings
from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.models import CompletionRequest, Message, ModelConfig
from autopilot.infrastructure.litellm_gateway import LiteLLMGateway
from autopilot.infrastructure.model_catalog_loader import load_model_catalog
from autopilot.infrastructure.router_factory import build_router

# --- corpus -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prompt:
    """One corpus entry, already resolved to concrete text.

    `tier` describes the PROMPT's own intended complexity (1/2/3), and is
    `None` for `of1-overflow`, which is not a routing-tier exercise at all —
    it exists purely to prove the harness survives a per-cell provider
    failure (a local model's context window) without aborting the run.
    """

    id: str
    tier: int | None
    category: str
    text: str
    max_tokens: int


def load_prompts(path: Path) -> list[Prompt]:
    """Read the YAML corpus at `path`.

    `prompt_file` entries resolve relative to `path`'s own parent directory,
    never the process's current working directory — the corpus loads
    identically no matter where `python -m autopilot.interfaces.cli.baseline`
    is invoked from.
    """
    data: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults: Mapping[str, Any] = data.get("defaults") or {}
    default_max_tokens = int(defaults.get("max_tokens", 512))

    prompts: list[Prompt] = []
    for raw_entry in data["prompts"]:
        entry: Mapping[str, Any] = raw_entry
        text = entry.get("prompt")
        if text is None:
            file_path = path.parent / entry["prompt_file"]
            text = file_path.read_text(encoding="utf-8")
        prompts.append(
            Prompt(
                id=entry["id"],
                tier=entry.get("tier"),
                category=entry["category"],
                text=text,
                max_tokens=int(entry.get("max_tokens", default_max_tokens)),
            )
        )
    return prompts


# --- matrix execution ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallResult:
    """One matrix cell: one prompt, against one model, one repeat.

    `tier` is the MODEL's catalog `quality_tier` (always a concrete 1/2/3,
    never `None`) — not the prompt's. Every cell for a given model shares the
    same value, which is what lets `aggregate_by_model` read it off the first
    row per model rather than needing to re-derive it from the catalog.
    """

    prompt_id: str
    model_key: str
    response_model_key: str | None  # who ACTUALLY answered; None when the cell failed
    tier: int
    repeat_index: int
    ok: bool
    output: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_cost: Decimal
    latency_ms: int | None
    error: str | None


async def run_matrix(
    *,
    gateway: LLMGateway,
    models: Sequence[ModelConfig],
    prompts: Sequence[Prompt],
    repeats: int,
    timeout_s: float,
    max_concurrency: int,
    logger: BoundLogger | None = None,
) -> list[CallResult]:
    """Run every (prompt, model, repeat) cell, bounded to `max_concurrency`
    concurrent provider calls via an `asyncio.Semaphore`.

    A cell that raises ANY exception — a domain `AutopilotError`, a timeout, or
    an unexpected `IndexError`/`AttributeError` from a malformed provider
    payload the gateway did not translate — is recorded as a failed
    `CallResult` with the error captured as text. It never aborts the other
    cells: that survival property (the whole point of the harness) must hold
    against every failure mode, not only the ones the domain names. Catching
    broad `Exception` is deliberate here; `BaseException` (KeyboardInterrupt,
    SystemExit) is intentionally left to propagate.
    """
    log = logger if logger is not None else structlog.get_logger(__name__)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(prompt: Prompt, model: ModelConfig, repeat_index: int) -> CallResult:
        tier = int(model.quality_tier)
        request = CompletionRequest(
            request_id=uuid4(),
            messages=(Message(role="user", content=prompt.text),),
            max_tokens=prompt.max_tokens,
        )
        async with semaphore:
            try:
                response = await gateway.complete(request, model, timeout_s=timeout_s)
            except Exception as exc:  # per-cell survival is the contract (see docstring)
                log.warning(
                    "baseline.cell_failed",
                    prompt_id=prompt.id,
                    model_key=model.key,
                    repeat_index=repeat_index,
                    error=str(exc),
                )
                return CallResult(
                    prompt_id=prompt.id,
                    model_key=model.key,
                    response_model_key=None,
                    tier=tier,
                    repeat_index=repeat_index,
                    ok=False,
                    output=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_cost=Decimal(0),
                    latency_ms=None,
                    error=str(exc),
                )

        answering = response.model_key
        if answering != model.key:
            # A baseline runs with fallbacks OFF (see main), so this must not
            # happen. If it ever does, a *different* model answered and this
            # cell's tokens/cost/latency are not the requested model's at all —
            # record the substitution as a failure rather than silently
            # mislabelling another model's numbers as this one's.
            log.warning(
                "baseline.substituted",
                prompt_id=prompt.id,
                requested=model.key,
                answered=answering,
                repeat_index=repeat_index,
            )
            return CallResult(
                prompt_id=prompt.id,
                model_key=model.key,
                response_model_key=answering,
                tier=tier,
                repeat_index=repeat_index,
                ok=False,
                output=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_cost=Decimal(0),
                latency_ms=None,
                error=f"answered by {answering!r}, not the requested model (fallback disabled)",
            )

        return CallResult(
            prompt_id=prompt.id,
            model_key=model.key,
            response_model_key=answering,
            tier=tier,
            repeat_index=repeat_index,
            ok=True,
            output=response.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_cost=response.cost.total,
            latency_ms=response.latency_ms,
            error=None,
        )

    cells = [
        run_one(prompt, model, repeat_index)
        for prompt in prompts
        for model in models
        for repeat_index in range(repeats)
    ]
    return list(await asyncio.gather(*cells))


# --- aggregation ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelStats:
    """Aggregated numbers for one model, across every prompt and repeat it
    was asked to answer in this run."""

    model_key: str
    tier: int
    p50_latency_ms: float
    p95_latency_ms: float
    avg_cost_per_req: Decimal
    total_tokens_out: int
    mean_completion_tokens: float
    p95_completion_tokens: float
    failures: int
    total_calls: int


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile over `values` (0 <= pct <= 100),
    matching the convention `numpy.percentile` defaults to. `0.0` for an
    empty sequence rather than raising — a model with zero successful calls
    is a valid (if unfortunate) row, not an error."""
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * (pct / 100)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return data[lower_index]
    lower_weight = upper_index - rank
    upper_weight = rank - lower_index
    return data[lower_index] * lower_weight + data[upper_index] * upper_weight


def aggregate_by_model(results: Sequence[CallResult]) -> list[ModelStats]:
    """One `ModelStats` row per distinct `model_key` in `results`, in the
    order each model key first appears."""
    order: list[str] = []
    by_model: dict[str, list[CallResult]] = {}
    for r in results:
        if r.model_key not in by_model:
            by_model[r.model_key] = []
            order.append(r.model_key)
        by_model[r.model_key].append(r)

    stats: list[ModelStats] = []
    for key in order:
        rows = by_model[key]
        ok_rows = [r for r in rows if r.ok]
        latencies = [float(r.latency_ms) for r in ok_rows if r.latency_ms is not None]
        completions = [
            float(r.completion_tokens) for r in ok_rows if r.completion_tokens is not None
        ]
        # Cost per SUCCESSFUL request: dividing real spend by all attempts
        # (including zero-cost failures) would understate the per-call cost the
        # baseline exists to compare. Failed cells cost nothing and are not asks
        # that returned an answer.
        ok_cost = sum((r.total_cost for r in ok_rows), Decimal(0))
        total_calls = len(rows)
        avg_cost = ok_cost / len(ok_rows) if ok_rows else Decimal(0)
        stats.append(
            ModelStats(
                model_key=key,
                tier=rows[0].tier,
                p50_latency_ms=percentile(latencies, 50),
                p95_latency_ms=percentile(latencies, 95),
                avg_cost_per_req=avg_cost,
                total_tokens_out=sum(r.completion_tokens or 0 for r in ok_rows),
                mean_completion_tokens=(sum(completions) / len(completions))
                if completions
                else 0.0,
                p95_completion_tokens=percentile(completions, 95),
                failures=total_calls - len(ok_rows),
                total_calls=total_calls,
            )
        )
    return stats


def output_token_ratio(stats: ModelStats, baseline: ModelStats) -> tuple[float, float]:
    """`(mean ratio, p95 ratio)` of `stats`'s completion-token volume against
    the baseline model's. The baseline compared against itself always yields
    `(1.0, 1.0)`."""
    mean_ratio = (
        stats.mean_completion_tokens / baseline.mean_completion_tokens
        if baseline.mean_completion_tokens
        else 0.0
    )
    p95_ratio = (
        stats.p95_completion_tokens / baseline.p95_completion_tokens
        if baseline.p95_completion_tokens
        else 0.0
    )
    return mean_ratio, p95_ratio


# --- spend gate -----------------------------------------------------------------

# Rough characters-per-token ratio. There is no concrete `TokenCounter` yet
# (only the port), so the pre-flight estimate cannot tokenise exactly; 4 is the
# usual English-text approximation. The estimate only feeds a safety threshold,
# not any billed figure, so approximate is fine — and output is over-counted at
# the `max_tokens` cap anyway, which keeps the total on the safe (high) side.
CHARS_PER_TOKEN = 4


def estimate_run_cost(
    *, models: Sequence[ModelConfig], prompts: Sequence[Prompt], repeats: int
) -> Decimal:
    """Conservative pre-flight upper bound on total run cost, used to gate the
    matrix *before* it spends anything.

    Input tokens are estimated from character count (`len(text) /
    CHARS_PER_TOKEN`); output tokens are charged at each prompt's `max_tokens`
    hard cap, not the smaller amount a model will actually emit. Both choices
    push the estimate *up* on purpose: a spend gate must fail safe, so a run it
    lets through never surprises with a bigger bill than it was warned about.
    Zero-priced local models contribute nothing.
    """
    total = Decimal(0)
    for prompt in prompts:
        est_input_tokens = math.ceil(len(prompt.text) / CHARS_PER_TOKEN)
        for model in models:
            per_call = (
                model.input_cost_per_token * est_input_tokens
                + model.output_cost_per_token * prompt.max_tokens
            )
            total += per_call * repeats
    return total


def check_spend_gate(
    estimated_spend: Decimal, *, threshold: Decimal, confirmed: bool
) -> str | None:
    """`None` means proceed; a `str` is the error to print to stderr and the
    reason the matrix must not run.

    A *pre-flight* guard: `estimated_spend` is the conservative upper bound from
    `estimate_run_cost`, checked before any provider call. Refusing here refuses
    the spend itself — re-running with `--confirm-spend` then executes the
    matrix exactly once, rather than paying a second time as a post-hoc gate
    would have forced.
    """
    if estimated_spend > threshold and not confirmed:
        return (
            f"estimated baseline run cost is ${estimated_spend} (conservative upper bound), "
            f"over the ${threshold} threshold, without --confirm-spend. Re-run with "
            "--confirm-spend to execute the matrix."
        )
    return None


# --- artifacts --------------------------------------------------------------------


def baseline_output_dir(artifacts_dir: Path, *, clock: Clock, iso_date: str | None = None) -> Path:
    """`<artifacts_dir>/baseline/<iso-date>/`. `iso_date` is injectable so
    tests are deterministic; production wiring leaves it `None` and derives
    the date from `clock`."""
    date = iso_date if iso_date is not None else clock.now().date().isoformat()
    return artifacts_dir / "baseline" / date


def _result_to_json(result: CallResult) -> dict[str, Any]:
    return {
        "prompt_id": result.prompt_id,
        "model_key": result.model_key,
        "response_model_key": result.response_model_key,
        "tier": result.tier,
        "repeat_index": result.repeat_index,
        "ok": result.ok,
        "output": result.output,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_cost": str(result.total_cost),  # Decimal is not JSON-serialisable
        "latency_ms": result.latency_ms,
        "error": result.error,
    }


def write_raw_jsonl(results: Sequence[CallResult], path: Path) -> None:
    """One JSON object per line, one line per `CallResult`. `Decimal` fields
    are serialised as strings — `json.dumps` cannot handle `Decimal` at all,
    and a float round-trip is exactly the money-precision loss this codebase
    refuses everywhere else."""
    lines = [json.dumps(_result_to_json(r)) for r in results]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


_SUMMARY_HEADER = (
    "| model | tier | p50 latency | p95 latency | avg cost/req | "
    "total tokens out | out-tok ratio (mean/p95) | failures |"
)
_SUMMARY_SEPARATOR = "|" + "|".join(["---"] * 8) + "|"


def write_summary_md(
    stats: Sequence[ModelStats],
    *,
    baseline_key: str,
    total_spend: Decimal,
    path: Path,
) -> None:
    """The headline report: one row per model, plus a trailing TOTAL spend
    line summing `total_cost` across every ok call in the run."""
    baseline = next((s for s in stats if s.model_key == baseline_key), None)
    lines = [_SUMMARY_HEADER, _SUMMARY_SEPARATOR]
    for s in stats:
        if baseline is not None:
            mean_ratio, p95_ratio = output_token_ratio(s, baseline)
            ratio_cell = f"{mean_ratio:.2f}/{p95_ratio:.2f}"
        else:
            ratio_cell = "n/a"
        lines.append(
            f"| {s.model_key} | {s.tier} | {s.p50_latency_ms:.0f}ms | "
            f"{s.p95_latency_ms:.0f}ms | ${s.avg_cost_per_req:.6f} | "
            f"{s.total_tokens_out} | {ratio_cell} | {s.failures} |"
        )
    lines.append("")
    lines.append(f"TOTAL run spend: ${total_spend}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs_md(
    results: Sequence[CallResult], prompts: Sequence[Prompt], *, path: Path
) -> None:
    """Side-by-side outputs, one section per prompt, showing each model's
    repeat-0 answer (or its error, if that cell failed). Ends with a
    `## What we learned` template — a placeholder, not a finding, to be
    filled in by a human after reading the real run's outputs."""
    by_prompt: dict[str, list[CallResult]] = {}
    for r in results:
        by_prompt.setdefault(r.prompt_id, []).append(r)

    lines: list[str] = []
    for prompt in prompts:
        lines.append(f"## {prompt.id}")
        lines.append("")
        for r in by_prompt.get(prompt.id, []):
            if r.repeat_index != 0:
                continue
            lines.append(f"### {r.model_key}")
            lines.append("")
            if r.ok:
                # An empty string is a real (if unusual) success — a refusal, or
                # a prompt whose correct answer is empty. Do NOT render it as a
                # failure; label it explicitly so the reader sees the model
                # returned nothing rather than errored.
                lines.append(r.output if r.output else "(empty output)")
            else:
                lines.append(f"FAILED: {r.error}")
            lines.append("")

    lines.append("## What we learned")
    lines.append("")
    lines.append(
        "<!-- TEMPLATE: fill in after reviewing the real run above. Replace both "
        "bullets with concrete prompt ids and model keys — do not leave the "
        "placeholders in a committed report. -->"
    )
    lines.append(
        "- TODO: on prompt `<prompt-id>`, `<cheap-model-key>` got it wrong "
        "(specifically: `<what it got wrong>`) while `<baseline-model-key>` "
        "answered correctly."
    )
    lines.append(
        "- TODO: on prompt `<prompt-id>`, `<cheap-model-key>` <succeeded/failed/"
        "was noticeably slower or more expensive>, which changes how confidently "
        "it can be routed for that prompt's category."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(
    results: Sequence[CallResult],
    stats: Sequence[ModelStats],
    prompts: Sequence[Prompt],
    *,
    baseline_key: str,
    total_spend: Decimal,
    output_dir: Path,
) -> None:
    """Write `raw.jsonl`, `summary.md` and `outputs.md` into `output_dir`,
    creating it (and any missing parents) first."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_raw_jsonl(results, output_dir / "raw.jsonl")
    write_summary_md(
        stats, baseline_key=baseline_key, total_spend=total_spend, path=output_dir / "summary.md"
    )
    write_outputs_md(results, prompts, path=output_dir / "outputs.md")


# --- wiring ------------------------------------------------------------------------


class _SystemClock:
    """The real clock, structurally satisfying `application.ports.Clock`.
    Lives here rather than in `infrastructure` because nothing else in the
    codebase needs a production `Clock` yet — this harness is its first
    caller."""

    def now(self) -> datetime:
        return datetime.now(UTC)


def _select_models(
    catalog: ModelCatalog, settings: Settings, models_arg: str | None
) -> list[ModelConfig]:
    """Enabled models, minus local models when `settings.disable_local` is set
    (env `AUTOPILOT_DISABLE_LOCAL`; there is no CLI flag for it), minus anything
    not named by `--models` (when given)."""
    models = list(catalog.enabled)
    if settings.disable_local:
        models = [m for m in models if m.provider != "ollama"]
    if models_arg:
        wanted = {key.strip() for key in models_arg.split(",") if key.strip()}
        models = [m for m in models if m.key in wanted]
    return models


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autopilot.interfaces.cli.baseline",
        description=(
            "Provider baseline harness: run a fixed prompt corpus against every "
            "enabled catalog model, several times each, and write latency/cost/"
            "token artifacts."
        ),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("config/baseline_prompts.yaml"),
        help="path to the prompt corpus YAML (default: config/baseline_prompts.yaml)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="repeats per prompt x model cell (default: 3)"
    )
    parser.add_argument(
        "--confirm-spend",
        action="store_true",
        help="required to write artifacts when total spend exceeds the confirmation threshold",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated catalog keys to restrict the matrix to (default: all enabled)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the artifacts directory (default: <artifacts_dir>/baseline/<iso-date>)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    settings = load_settings()
    catalog = load_model_catalog(settings.model_catalog_path)
    models = _select_models(catalog, settings, args.models)
    prompts = load_prompts(args.prompts)

    if not models:
        # Refuse rather than run an empty matrix: write_artifacts would otherwise
        # overwrite <artifacts_dir>/baseline/<today>/ with empty files and exit
        # 0, silently destroying a committed reference run dated today.
        print(
            "no models selected — check --models and AUTOPILOT_DISABLE_LOCAL. "
            "Refusing to run an empty matrix.",
            file=sys.stderr,
        )
        return 1

    if catalog.baseline.key not in {m.key for m in models}:
        # The out-tok ratio column normalises against the baseline model; without
        # it every ratio is 'n/a'. Surface that instead of shipping a silently
        # meaningless summary.
        print(
            f"warning: baseline model {catalog.baseline.key!r} is not in the "
            "selected matrix — the out-tok ratio column will be 'n/a'.",
            file=sys.stderr,
        )

    estimated_spend = estimate_run_cost(models=models, prompts=prompts, repeats=args.repeats)
    print(f"estimated run cost (conservative upper bound): ${estimated_spend}")

    guard_error = check_spend_gate(
        estimated_spend,
        threshold=settings.baseline_confirm_spend_threshold_usd,
        confirmed=args.confirm_spend,
    )
    if guard_error is not None:
        print(guard_error, file=sys.stderr)
        return 1

    # Fallbacks OFF: a baseline measures each model in isolation. With them on,
    # a model that times out or errors is silently answered by a costlier one
    # and its numbers get recorded under the wrong key (see run_matrix's
    # substitution guard) — which is exactly the contamination this harness
    # must not produce.
    router = build_router(catalog, settings, with_fallbacks=False)
    gateway = LiteLLMGateway(router=router, catalog=catalog)

    results = asyncio.run(
        run_matrix(
            gateway=gateway,
            models=models,
            prompts=prompts,
            repeats=args.repeats,
            timeout_s=settings.request_timeout_s,
            max_concurrency=settings.baseline_max_concurrency,
        )
    )

    total_spend = sum((r.total_cost for r in results if r.ok), Decimal(0))
    print(f"actual run spend: ${total_spend} (estimated ${estimated_spend})")

    output_dir = args.output_dir or baseline_output_dir(
        settings.artifacts_dir, clock=_SystemClock()
    )
    stats = aggregate_by_model(results)
    write_artifacts(
        results,
        stats,
        prompts,
        baseline_key=catalog.baseline.key,
        total_spend=total_spend,
        output_dir=output_dir,
    )
    print(f"wrote artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
