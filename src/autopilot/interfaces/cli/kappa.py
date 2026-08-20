"""Inter-annotator agreement for the complexity taxonomy (#10).

Reads two label passes over the same calibration set and reports Cohen's
kappa, the confusion matrix, and every disagreement by prompt id. Invoke as:

    python -m autopilot.interfaces.cli.kappa \
        --pass-a artifacts/calibration/<date>/pass-a.yaml \
        --pass-b artifacts/calibration/<date>/pass-b.yaml

**Kappa is reported, not gated.** This CLI never exits non-zero because a
number came out low: `docs/complexity-taxonomy.md` section 9 explains why
gating on kappa is an unbounded relabelling loop that also corrupts the
measurement. It exits non-zero only for things that make the number
meaningless — a prompt labelled in one pass and not the other, an unlabelled
row, a tier outside 1..3.

The disagreement list is the actual product. Kappa is one number to put in a
README; the per-prompt disagreements are what get turned into rubric
clarifications, and a clarification is the only thing that moves the next
kappa.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml

VALID_TIERS: Final[frozenset[int]] = frozenset({1, 2, 3})

# Landis & Koch (1977). Printed for orientation only — see the module
# docstring. Ordered from the top down so the first match wins.
KAPPA_BANDS: Final[tuple[tuple[float, str], ...]] = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.00, "slight"),
)


def interpret(kappa: float) -> str:
    """Name the Landis & Koch band `kappa` falls in."""
    for floor, name in KAPPA_BANDS:
        if kappa >= floor:
            return name
    return "poor (worse than chance)"


# --- label passes -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelPass:
    """One annotator's labels over the calibration set.

    `reasons` is carried alongside `labels` because a disagreement is only
    actionable with both annotators' stated reasoning next to it — that is
    what turns "cal-13: 2 vs 3" into a rubric clarification.
    """

    name: str
    labels: Mapping[str, int]
    reasons: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError(f"label pass {self.name!r} contains no labels")


def load_label_pass(path: Path, *, name: str | None = None) -> LabelPass:
    """Read a filled-in calibration worksheet.

    Refuses a pass with any unlabelled row rather than silently computing
    agreement over the subset that happens to be filled in: kappa over 11 of
    25 prompts is not kappa over the calibration set, and nothing in the
    output would have said so.
    """
    data: Mapping[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels: dict[str, int] = {}
    reasons: dict[str, str] = {}
    unlabelled: list[str] = []

    for raw_entry in data["prompts"]:
        entry: Mapping[str, Any] = raw_entry
        prompt_id = str(entry["id"])
        tier = entry.get("tier")
        if tier is None:
            unlabelled.append(prompt_id)
            continue
        tier_int = int(tier)
        if tier_int not in VALID_TIERS:
            raise ValueError(f"{path}: {prompt_id} has tier {tier!r}, expected one of 1, 2, 3")
        if prompt_id in labels:
            raise ValueError(f"{path}: duplicate prompt id {prompt_id!r}")
        labels[prompt_id] = tier_int
        reasons[prompt_id] = str(entry.get("label_reason") or "")

    if unlabelled:
        raise ValueError(
            f"{path}: {len(unlabelled)} prompt(s) still unlabelled: {', '.join(unlabelled)}"
        )
    return LabelPass(name=name or path.stem, labels=labels, reasons=reasons)


# --- agreement ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One prompt the two passes labelled differently."""

    prompt_id: str
    tier_a: int
    tier_b: int
    reason_a: str
    reason_b: str


@dataclass(frozen=True, slots=True)
class Agreement:
    """The full result of comparing two passes.

    `kappa` is `None` when it is *undefined*, not when it is zero: if both
    annotators used exactly one category for every prompt, expected agreement
    is 1.0 and the statistic divides by zero. That case is reported as
    undefined rather than as 0.0 or 1.0, both of which would be a lie.
    """

    pass_a: str
    pass_b: str
    n: int
    observed: float
    expected: float
    kappa: float | None
    categories: tuple[int, ...]
    matrix: Mapping[tuple[int, int], int]
    disagreements: tuple[Disagreement, ...]

    @property
    def agreed(self) -> int:
        return self.n - len(self.disagreements)


def cohens_kappa(pass_a: LabelPass, pass_b: LabelPass) -> Agreement:
    """Compare two passes over the *same* prompt ids.

    Refuses passes whose id sets differ. Cohen's kappa is defined for two
    raters scoring the same items; intersecting the ids silently would answer
    a different question than the one asked, on a subset chosen by whichever
    rows happened to be present.
    """
    only_a = sorted(set(pass_a.labels) - set(pass_b.labels))
    only_b = sorted(set(pass_b.labels) - set(pass_a.labels))
    if only_a or only_b:
        raise ValueError(
            "the two passes label different prompts — "
            f"only in {pass_a.name}: {only_a or 'none'}; only in {pass_b.name}: {only_b or 'none'}"
        )

    prompt_ids = sorted(pass_a.labels)
    n = len(prompt_ids)
    pairs = [(pass_a.labels[i], pass_b.labels[i]) for i in prompt_ids]

    matrix = Counter(pairs)
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    categories = tuple(sorted(set(counts_a) | set(counts_b)))

    observed = sum(1 for a, b in pairs if a == b) / n
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)
    kappa = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)

    disagreements = tuple(
        Disagreement(
            prompt_id=prompt_id,
            tier_a=pass_a.labels[prompt_id],
            tier_b=pass_b.labels[prompt_id],
            reason_a=pass_a.reasons.get(prompt_id, ""),
            reason_b=pass_b.reasons.get(prompt_id, ""),
        )
        for prompt_id in prompt_ids
        if pass_a.labels[prompt_id] != pass_b.labels[prompt_id]
    )

    return Agreement(
        pass_a=pass_a.name,
        pass_b=pass_b.name,
        n=n,
        observed=observed,
        expected=expected,
        kappa=kappa,
        categories=categories,
        matrix=dict(matrix),
        disagreements=disagreements,
    )


# --- rendering ----------------------------------------------------------------


def format_kappa(agreement: Agreement) -> str:
    """`kappa` as a display string, band included, `undefined` when it is."""
    if agreement.kappa is None:
        return "undefined (both passes used a single category; expected agreement is 1.0)"
    return f"{agreement.kappa:.3f} ({interpret(agreement.kappa)})"


def render_markdown(
    agreement: Agreement,
    *,
    label: str,
    generated_at: str,
    prompts_path: Path | None = None,
) -> str:
    """The committed artifact: numbers, matrix, and every disagreement.

    `label` is prose the caller supplies to describe *what was measured* —
    "two independent LLM annotators", "human blind self-relabel 48h apart".
    It is mandatory precisely because the procedure is the part a reader can
    otherwise only guess at, and guessing generously is how a calibration
    gets oversold.
    """
    lines: list[str] = [
        f"# Annotator agreement — {label}",
        "",
        f"- Generated: {generated_at}",
        f"- Pass A: `{agreement.pass_a}`",
        f"- Pass B: `{agreement.pass_b}`",
        f"- Prompts: {agreement.n}"
        + (f", from `{prompts_path}`" if prompts_path is not None else ""),
        "",
        "| Metric | Value |",
        "| :-- | :-- |",
        f"| Cohen's κ | **{format_kappa(agreement)}** |",
        f"| Raw agreement | {agreement.agreed} / {agreement.n} ({agreement.observed * 100:.1f}%) |",
        f"| Expected agreement by chance | {agreement.expected * 100:.1f}% |",
        "",
        "κ is reported, never gated — see `docs/complexity-taxonomy.md` section 9.",
        "",
        "## Confusion matrix",
        "",
        "Rows are pass A, columns are pass B.",
        "",
        "| A \\ B | " + " | ".join(str(c) for c in agreement.categories) + " |",
        "| :-- | " + " | ".join("--:" for _ in agreement.categories) + " |",
    ]
    for row in agreement.categories:
        cells = " | ".join(str(agreement.matrix.get((row, col), 0)) for col in agreement.categories)
        lines.append(f"| **{row}** | {cells} |")

    lines += ["", "## Disagreements", ""]
    if not agreement.disagreements:
        lines.append("None. Every prompt received the same tier in both passes.")
    else:
        lines.append("Each of these owes the rubric a clarification (section 10).")
        lines.append("")
        for d in agreement.disagreements:
            lines += [
                f"### `{d.prompt_id}` — tier {d.tier_a} vs {d.tier_b}",
                "",
                f"- **{agreement.pass_a}** (tier {d.tier_a}): {d.reason_a or '_no reason given_'}",
                f"- **{agreement.pass_b}** (tier {d.tier_b}): {d.reason_b or '_no reason given_'}",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def render_console(agreement: Agreement) -> str:
    """Short form for the terminal."""
    lines = [
        f"n = {agreement.n}   agreement = {agreement.agreed}/{agreement.n} "
        f"({agreement.observed * 100:.1f}%)   expected = {agreement.expected * 100:.1f}%",
        f"Cohen's kappa = {format_kappa(agreement)}",
    ]
    if agreement.disagreements:
        lines.append(f"{len(agreement.disagreements)} disagreement(s):")
        lines += [
            f"  {d.prompt_id}: {agreement.pass_a}={d.tier_a}  {agreement.pass_b}={d.tier_b}"
            for d in agreement.disagreements
        ]
    else:
        lines.append("no disagreements")
    return "\n".join(lines)


# --- wiring -------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autopilot.interfaces.cli.kappa",
        description=(
            "Cohen's kappa between two label passes over the complexity-taxonomy "
            "calibration set. Reports; never gates."
        ),
    )
    parser.add_argument("--pass-a", type=Path, required=True, help="first filled-in worksheet")
    parser.add_argument("--pass-b", type=Path, required=True, help="second filled-in worksheet")
    parser.add_argument(
        "--label",
        required=True,
        help=(
            "what was actually measured, in prose — e.g. 'human blind self-relabel, 48h "
            "apart'. Required: the procedure is the part a reader cannot infer."
        ),
    )
    parser.add_argument("--markdown", type=Path, help="also write the full report here")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("config/calibration_prompts.yaml"),
        help="calibration set the passes came from, recorded in the report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    pass_a = load_label_pass(args.pass_a)
    pass_b = load_label_pass(args.pass_b)
    agreement = cohens_kappa(pass_a, pass_b)

    print(render_console(agreement))

    if args.markdown is not None:
        report = render_markdown(
            agreement,
            label=args.label,
            generated_at=datetime.now(UTC).date().isoformat(),
            prompts_path=args.prompts,
        )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
        print(f"wrote {args.markdown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
