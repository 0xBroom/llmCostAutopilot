"""`infrastructure.quality_validation_report` — the judge-agreement harness —
plus a guard on `docs/quality.md` and the committed validation-set fixture.

HUMAN-IN-THE-LOOP BOUNDARY: every "labelled" row used below is explicitly
SYNTHETIC, written inline in this test file, and is never presented as a real
human judgment. The committed fixture
(`tests/fixtures/quality/validation_set_v1.jsonl`) is asserted to still be
100% pending placeholders — this is a deliberate guard so nobody can slip a
fabricated agreement figure into `docs/quality.md` without this test (and a
human reviewer) noticing.
"""

from __future__ import annotations

from pathlib import Path

from autopilot.infrastructure.quality_validation_report import (
    compute_agreement,
    load_validation_rows,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALIDATION_SET_PATH = _REPO_ROOT / "tests" / "fixtures" / "quality" / "validation_set_v1.jsonl"
_DOCS_PATH = _REPO_ROOT / "docs" / "quality.md"


# --- compute_agreement, against clearly SYNTHETIC in-test rows -----------------


def test_agreement_on_fully_labelled_synthetic_rows() -> None:
    """SYNTHETIC rows, invented for this test only — not real judgments."""
    synthetic_rows = [
        {"id": "synthetic-1", "judge_winner": "tie", "human_label": "tie"},
        {"id": "synthetic-2", "judge_winner": "candidate", "human_label": "candidate"},
        {"id": "synthetic-3", "judge_winner": "reference", "human_label": "candidate"},
        {"id": "synthetic-4", "judge_winner": "tie", "human_label": "tie"},
    ]

    report = compute_agreement(synthetic_rows)

    assert report.n_total == 4
    assert report.n_labelled == 4
    assert report.n_pending == 0
    assert report.agreement_rate == 3 / 4


def test_agreement_ignores_pending_rows_and_only_counts_labelled_ones() -> None:
    """SYNTHETIC mixed set: 2 labelled (1 agree, 1 disagree), 2 still pending."""
    synthetic_rows: list[dict[str, str | None]] = [
        {"id": "synthetic-a", "judge_winner": "candidate", "human_label": "candidate"},
        {"id": "synthetic-b", "judge_winner": "reference", "human_label": "tie"},
        {"id": "synthetic-c", "judge_winner": "tie", "human_label": None},
        {"id": "synthetic-d", "judge_winner": "candidate"},  # human_label absent entirely
    ]

    report = compute_agreement(synthetic_rows)

    assert report.n_total == 4
    assert report.n_labelled == 2
    assert report.n_pending == 2
    assert report.agreement_rate == 1 / 2


def test_agreement_is_none_not_zero_when_nothing_is_labelled_yet() -> None:
    """The critical honesty property: "nothing labelled" must never render as
    a fabricated 0% agreement — that would misreport a rubric failure that
    was never actually measured."""
    synthetic_rows = [
        {"id": "synthetic-x", "judge_winner": "tie", "human_label": None},
        {"id": "synthetic-y", "judge_winner": "candidate", "human_label": None},
    ]

    report = compute_agreement(synthetic_rows)

    assert report.n_total == 2
    assert report.n_labelled == 0
    assert report.n_pending == 2
    assert report.agreement_rate is None


def test_agreement_on_empty_rows_is_none() -> None:
    report = compute_agreement([])

    assert report.n_total == 0
    assert report.agreement_rate is None


# --- the committed fixture: still pending, not fabricated ----------------------


def test_committed_validation_set_rows_are_still_pending_placeholders() -> None:
    """Every row in the committed fixture MUST still be an unmistakable
    placeholder: `human_label` unset, `TODO` marker present. If this test
    ever fails because a row now has a real `human_label`, that is exactly
    the moment `docs/quality.md`'s agreement section must be rewritten with
    a real, human-reviewed figure — not before."""
    rows = load_validation_rows(_VALIDATION_SET_PATH)

    assert len(rows) >= 1
    for row in rows:
        assert row.get("human_label") is None, (
            f"row {row.get('id')!r} appears to already carry a human_label; "
            "docs/quality.md's PENDING agreement section must be updated by a "
            "human reviewer before this test may be relaxed"
        )
        assert row.get("TODO"), f"row {row.get('id')!r} is missing its TODO marker"


def test_committed_validation_set_reports_no_agreement_yet() -> None:
    rows = load_validation_rows(_VALIDATION_SET_PATH)

    report = compute_agreement(rows)

    assert report.n_labelled == 0
    assert report.agreement_rate is None


# --- docs/quality.md: the PENDING caveat is unmistakable -----------------------


def test_docs_state_the_agreement_figure_is_pending() -> None:
    text = _DOCS_PATH.read_text(encoding="utf-8")

    assert "PENDING" in text
    assert "n=15" in text
    assert "human_label" in text
    assert "TODO" in text


def test_docs_state_the_wilson_ci_caveat_as_computed_not_cited() -> None:
    text = _DOCS_PATH.read_text(encoding="utf-8")

    assert "Wilson" in text
    assert "55%" in text
    assert "93%" in text
    assert "this project's own" in text
