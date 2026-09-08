"""The judge-agreement reporting harness for `tests/fixtures/quality/validation_set_v1.jsonl`.

This is NOT a product code path — nothing in `domain` or `application` reads
this module or its fixture at runtime. It exists to compute the one figure
`docs/quality.md` reports: how often the LLM judge's resolved outcome agrees
with a human's label on a small, committed validation set (spec: Judge
Validation Reporting).

HUMAN-IN-THE-LOOP BOUNDARY, stated plainly: nothing in this codebase may
invent a human label. `tests/fixtures/quality/validation_set_v1.jsonl` ships
with `human_label: null` placeholder rows only, each carrying an explicit
`"TODO": "awaiting human label"` marker. `compute_agreement` reflects that
faithfully — it returns `agreement_rate=None` whenever there is nothing
labelled to compute agreement over, rather than reporting 0% or 100% as if a
real figure existed. `docs/quality.md` states the agreement figure itself is
PENDING until a human completes that committed set to `n=15`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """The result of comparing the judge's resolved outcome against a
    human's label, over whatever rows in the set are actually labelled.

    `agreement_rate` is `None` — never `0.0` — when `n_labelled` is zero:
    "no agreement was measured" and "the judge agreed with nobody" are not
    the same claim, and this type never lets the former be read as the
    latter.
    """

    n_total: int
    n_labelled: int
    n_pending: int
    agreement_rate: float | None


def load_validation_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Read one JSON object per non-blank line. Pure I/O, no validation of
    row shape beyond "is this a JSON object" — `compute_agreement` is
    tolerant of whatever optional fields a row does or does not carry."""
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, Mapping):
            raise ValueError(f"{path}: each line must be a JSON object, got {parsed!r}")
        rows.append(parsed)
    return tuple(rows)


def compute_agreement(rows: Sequence[Mapping[str, Any]]) -> AgreementReport:
    """Compute judge/human agreement over `rows`.

    Each row is expected to carry `judge_winner` (the judge's resolved,
    position-independent outcome: `"candidate"` / `"reference"` / `"tie"`)
    and `human_label` (the same domain, or `None`/absent when a row is still
    a pending placeholder awaiting a human annotator). A row whose
    `human_label` is `None` or absent contributes to `n_pending`, never to
    the agreement count — there is no label to agree or disagree with yet.
    """
    n_total = len(rows)
    labelled = [row for row in rows if row.get("human_label") is not None]
    n_labelled = len(labelled)
    n_pending = n_total - n_labelled

    if n_labelled == 0:
        return AgreementReport(
            n_total=n_total, n_labelled=0, n_pending=n_pending, agreement_rate=None
        )

    agreeing = sum(1 for row in labelled if row.get("human_label") == row.get("judge_winner"))
    return AgreementReport(
        n_total=n_total,
        n_labelled=n_labelled,
        n_pending=n_pending,
        agreement_rate=agreeing / n_labelled,
    )
