"""Inter-annotator agreement (#10): the kappa statistic, the guards that keep
it meaningful, worksheet loading, and the committed report's rendering.

The arithmetic cases use hand-computed expectations rather than a library, so
a wrong formula fails here instead of agreeing with itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autopilot.interfaces.cli.kappa import (
    LabelPass,
    cohens_kappa,
    format_kappa,
    interpret,
    load_label_pass,
    main,
    render_console,
    render_markdown,
)

CALIBRATION_SET = Path("config/calibration_prompts.yaml")


def make_pass(
    name: str, labels: dict[str, int], reasons: dict[str, str] | None = None
) -> LabelPass:
    return LabelPass(name=name, labels=labels, reasons=reasons or {})


def write_worksheet(path: Path, rows: list[tuple[str, int | None, str]]) -> Path:
    lines = ["version: 1", "prompts:"]
    for prompt_id, tier, reason in rows:
        lines += [
            f"  - id: {prompt_id}",
            f"    tier: {'null' if tier is None else tier}",
            f'    label_reason: "{reason}"',
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- the statistic --------------------------------------------------------------


def test_perfect_agreement_over_several_categories_is_one() -> None:
    labels = {"a": 1, "b": 2, "c": 3, "d": 1}
    result = cohens_kappa(make_pass("x", labels), make_pass("y", dict(labels)))

    assert result.kappa == pytest.approx(1.0)
    assert result.observed == pytest.approx(1.0)
    assert result.disagreements == ()
    assert result.agreed == 4


def test_textbook_two_by_two_table() -> None:
    """A hand-computed case: p_o = 0.70, p_e = 0.50, kappa = 0.40.

    Cells: both-1 = 20, A=1/B=2 = 5, A=2/B=1 = 10, both-2 = 15, n = 50.
    Marginals A: 25/25. Marginals B: 30/20.
    p_e = 0.5*0.6 + 0.5*0.4 = 0.50.
    """
    a: dict[str, int] = {}
    b: dict[str, int] = {}
    for index in range(50):
        if index < 20:
            a[f"p{index}"], b[f"p{index}"] = 1, 1
        elif index < 25:
            a[f"p{index}"], b[f"p{index}"] = 1, 2
        elif index < 35:
            a[f"p{index}"], b[f"p{index}"] = 2, 1
        else:
            a[f"p{index}"], b[f"p{index}"] = 2, 2

    result = cohens_kappa(make_pass("a", a), make_pass("b", b))

    assert result.observed == pytest.approx(0.70)
    assert result.expected == pytest.approx(0.50)
    assert result.kappa == pytest.approx(0.40)
    assert result.agreed == 35
    assert len(result.disagreements) == 15


def test_agreement_no_better_than_chance_is_about_zero() -> None:
    # A alternates 1,2; B is 1 for the first half and 2 for the second. Each
    # rater uses each category 50% of the time, and they agree 50% of the time
    # — exactly what chance predicts.
    a = {f"p{i}": 1 + (i % 2) for i in range(20)}
    b = {f"p{i}": 1 if i < 10 else 2 for i in range(20)}

    result = cohens_kappa(make_pass("a", a), make_pass("b", b))

    assert result.observed == pytest.approx(0.50)
    assert result.expected == pytest.approx(0.50)
    assert result.kappa == pytest.approx(0.0)


def test_systematic_disagreement_is_negative() -> None:
    a = {"p1": 1, "p2": 1, "p3": 2, "p4": 2}
    b = {"p1": 2, "p2": 2, "p3": 1, "p4": 1}

    result = cohens_kappa(make_pass("a", a), make_pass("b", b))

    assert result.kappa is not None
    assert result.kappa < 0
    assert interpret(result.kappa) == "poor (worse than chance)"


def test_kappa_is_undefined_when_both_passes_use_one_category() -> None:
    """Not 0.0, not 1.0 — undefined. Expected agreement is 1.0, so the
    statistic divides by zero, and reporting a number there is a lie."""
    labels = {"p1": 2, "p2": 2, "p3": 2}
    result = cohens_kappa(make_pass("a", labels), make_pass("b", dict(labels)))

    assert result.kappa is None
    assert result.observed == pytest.approx(1.0)
    assert "undefined" in format_kappa(result)


def test_disagreements_carry_both_reasons() -> None:
    a = make_pass("a", {"cal-01": 1, "cal-02": 2}, {"cal-02": "single-hop condensation"})
    b = make_pass("b", {"cal-01": 1, "cal-02": 3}, {"cal-02": "needs synthesis"})

    result = cohens_kappa(a, b)

    assert len(result.disagreements) == 1
    disagreement = result.disagreements[0]
    assert (disagreement.prompt_id, disagreement.tier_a, disagreement.tier_b) == ("cal-02", 2, 3)
    assert disagreement.reason_a == "single-hop condensation"
    assert disagreement.reason_b == "needs synthesis"


def test_mismatched_prompt_sets_are_refused() -> None:
    """Intersecting silently would answer a different question, over a subset
    chosen by whichever rows happened to be present."""
    a = make_pass("a", {"cal-01": 1, "cal-02": 2})
    b = make_pass("b", {"cal-01": 1, "cal-03": 2})

    with pytest.raises(ValueError, match="label different prompts"):
        cohens_kappa(a, b)


def test_empty_pass_is_refused() -> None:
    with pytest.raises(ValueError, match="no labels"):
        make_pass("a", {})


@pytest.mark.parametrize(
    ("kappa", "band"),
    [
        (0.95, "almost perfect"),
        (0.81, "almost perfect"),
        (0.70, "substantial"),
        (0.50, "moderate"),
        (0.30, "fair"),
        (0.10, "slight"),
        (0.00, "slight"),
        (-0.20, "poor (worse than chance)"),
    ],
)
def test_landis_koch_bands(kappa: float, band: str) -> None:
    assert interpret(kappa) == band


# --- worksheet loading ----------------------------------------------------------


def test_load_worksheet_reads_tiers_and_reasons(tmp_path: Path) -> None:
    path = write_worksheet(
        tmp_path / "pass-a.yaml",
        [("cal-01", 1, "answer is present in the input"), ("cal-02", 3, "multi-hop")],
    )

    label_pass = load_label_pass(path)

    assert label_pass.name == "pass-a"
    assert label_pass.labels == {"cal-01": 1, "cal-02": 3}
    assert label_pass.reasons["cal-01"] == "answer is present in the input"


def test_unlabelled_rows_are_refused_by_id(tmp_path: Path) -> None:
    """Kappa over 1 of 3 prompts is not kappa over the calibration set, and
    nothing in the output would have said so."""
    path = write_worksheet(
        tmp_path / "half.yaml", [("cal-01", 1, ""), ("cal-02", None, ""), ("cal-03", None, "")]
    )

    with pytest.raises(ValueError, match="cal-02, cal-03"):
        load_label_pass(path)


def test_tier_outside_the_enum_is_refused(tmp_path: Path) -> None:
    path = write_worksheet(tmp_path / "bad.yaml", [("cal-01", 4, "")])

    with pytest.raises(ValueError, match="expected one of 1, 2, 3"):
        load_label_pass(path)


def test_duplicate_prompt_id_is_refused(tmp_path: Path) -> None:
    path = write_worksheet(tmp_path / "dupe.yaml", [("cal-01", 1, ""), ("cal-01", 2, "")])

    with pytest.raises(ValueError, match="duplicate prompt id"):
        load_label_pass(path)


# --- the committed calibration set ----------------------------------------------


def test_calibration_set_is_shipped_unlabelled() -> None:
    """The worksheet must never carry labels: copying it is how an annotator
    starts, and a pre-filled tier would leak one."""
    with pytest.raises(ValueError, match="still unlabelled"):
        load_label_pass(CALIBRATION_SET)


def test_calibration_set_is_disjoint_from_the_baseline_corpus() -> None:
    """Section 9 of the taxonomy: the #9 prompts have their rulings argued in
    the document, so labelling them would measure recall, not clarity."""
    import yaml

    calibration = yaml.safe_load(CALIBRATION_SET.read_text(encoding="utf-8"))
    baseline = yaml.safe_load(Path("config/baseline_prompts.yaml").read_text(encoding="utf-8"))

    calibration_ids = {p["id"] for p in calibration["prompts"]}
    baseline_ids = {p["id"] for p in baseline["prompts"]}

    assert len(calibration_ids) == 25
    assert calibration_ids & baseline_ids == set()
    assert calibration["taxonomy_version"] == 1


# --- rendering ------------------------------------------------------------------


def test_markdown_report_states_the_procedure_and_every_disagreement() -> None:
    a = make_pass("pass-a", {"cal-01": 1, "cal-02": 2, "cal-03": 3}, {"cal-03": "synthesis"})
    b = make_pass("pass-b", {"cal-01": 1, "cal-02": 3, "cal-03": 3}, {"cal-02": "second hop"})

    report = render_markdown(
        cohens_kappa(a, b),
        label="two independent LLM annotators",
        generated_at="2026-08-20",
        prompts_path=CALIBRATION_SET,
    )

    assert "two independent LLM annotators" in report
    assert "2026-08-20" in report
    assert "cal-02" in report
    assert "second hop" in report
    # A disagreement without a stated reason is still listed, marked as such.
    assert "_no reason given_" in report
    assert "never gated" in report


def test_markdown_report_says_so_when_the_passes_agree_completely() -> None:
    labels = {"cal-01": 1, "cal-02": 3}
    report = render_markdown(
        cohens_kappa(make_pass("a", labels), make_pass("b", dict(labels))),
        label="smoke",
        generated_at="2026-08-20",
    )

    assert "Every prompt received the same tier in both passes." in report


def test_console_output_lists_disagreeing_prompt_ids() -> None:
    a = make_pass("a", {"cal-01": 1, "cal-02": 2})
    b = make_pass("b", {"cal-01": 1, "cal-02": 3})

    console = render_console(cohens_kappa(a, b))

    assert "cal-02" in console
    assert "1 disagreement(s)" in console


# --- CLI ------------------------------------------------------------------------


def test_cli_writes_the_report_and_never_gates_on_a_low_kappa(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Systematic disagreement — a negative kappa — still exits 0. Gating is
    an unbounded relabelling loop; see the module docstring."""
    pass_a = write_worksheet(tmp_path / "a.yaml", [("cal-01", 1, ""), ("cal-02", 2, "")])
    pass_b = write_worksheet(tmp_path / "b.yaml", [("cal-01", 2, ""), ("cal-02", 1, "")])
    report = tmp_path / "nested" / "agreement.md"

    exit_code = main(
        [
            "--pass-a",
            str(pass_a),
            "--pass-b",
            str(pass_b),
            "--label",
            "deliberate disagreement",
            "--markdown",
            str(report),
        ]
    )

    assert exit_code == 0
    assert report.exists()
    assert "deliberate disagreement" in report.read_text(encoding="utf-8")
    assert "kappa" in capsys.readouterr().out.lower()
