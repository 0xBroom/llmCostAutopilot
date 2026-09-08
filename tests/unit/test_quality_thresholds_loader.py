"""`infrastructure.quality_thresholds_loader.load_quality_thresholds` — the
fail-loud versioned-YAML loader for per-task-type quality-contract profiles.

Mirrors `test_model_catalog_loader.py`'s structure: fixtures written to
`tmp_path`, one test per fail-loud path, plus a dedicated test proving the
real, shipped `config/quality.yaml` loads cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from autopilot.domain.errors import ConfigurationError
from autopilot.domain.quality import JudgeMode, TaskType
from autopilot.infrastructure.quality_thresholds_loader import (
    QUALITY_SCHEMA_VERSION,
    load_quality_thresholds,
)


def _valid_task_types() -> dict[str, Any]:
    return {
        "extraction": {
            "mode": "deterministic",
            "checks": ["json_valid", "required_fields", "not_truncated"],
            "required_fields": ["value"],
        },
        "classification": {
            "mode": "deterministic",
            "checks": ["not_refused", "not_truncated", "label_in_set"],
            "allowed_labels": ["positive", "negative", "neutral"],
        },
        "summarization": {
            "mode": "pairwise",
            "checks": ["not_refused", "not_truncated", "length_anomaly"],
            "length_bounds": [20, 4000],
            "fail_reference_margin_min": "moderate",
        },
        "reasoning": {
            "mode": "pairwise",
            "checks": ["not_refused", "not_truncated"],
            "fail_reference_margin_min": "moderate",
        },
        "default": {
            "mode": "pairwise",
            "checks": ["not_refused", "not_truncated", "length_anomaly"],
            "length_bounds": [1, 8000],
            "fail_reference_margin_min": "moderate",
        },
    }


def _write_thresholds(path: Path, task_types: dict[str, Any], *, version: int | None = 1) -> Path:
    payload: dict[str, Any] = {"task_types": task_types}
    if version is not None:
        payload["version"] = version
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# --- happy path ---------------------------------------------------------------


def test_valid_thresholds_load_and_resolve_every_task_type(tmp_path: Path) -> None:
    path = _write_thresholds(tmp_path / "quality.yaml", _valid_task_types())

    thresholds = load_quality_thresholds(path)

    assert set(thresholds.profiles) == set(TaskType)

    extraction = thresholds.get(TaskType.EXTRACTION)
    assert extraction.mode is JudgeMode.DETERMINISTIC
    assert extraction.check_names == ("json_valid", "required_fields", "not_truncated")
    assert extraction.required_fields == ("value",)
    assert extraction.fail_reference_margin_min is None

    classification = thresholds.get(TaskType.CLASSIFICATION)
    assert classification.allowed_labels == frozenset({"positive", "negative", "neutral"})

    summarization = thresholds.get(TaskType.SUMMARIZATION)
    assert summarization.mode is JudgeMode.PAIRWISE
    assert summarization.fail_reference_margin_min == "moderate"
    assert summarization.length_bounds == (20, 4000)

    reasoning = thresholds.get(TaskType.REASONING)
    assert reasoning.fail_reference_margin_min == "moderate"
    assert reasoning.length_bounds is None

    default = thresholds.get(TaskType.DEFAULT)
    assert default.fail_reference_margin_min == "moderate"


# --- schema version -------------------------------------------------------------


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = _write_thresholds(tmp_path / "quality.yaml", _valid_task_types(), version=2)

    with pytest.raises(ConfigurationError, match="schema version"):
        load_quality_thresholds(path)


def test_missing_schema_version_raises(tmp_path: Path) -> None:
    path = _write_thresholds(tmp_path / "quality.yaml", _valid_task_types(), version=None)

    with pytest.raises(ConfigurationError, match="schema version"):
        load_quality_thresholds(path)


# --- task-type key cross-check (bidirectional) ---------------------------------


def test_missing_task_type_key_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    del task_types["reasoning"]
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="missing="):
        load_quality_thresholds(path)


def test_unknown_task_type_key_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    task_types["translation"] = task_types["default"]
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="unknown="):
        load_quality_thresholds(path)


# --- per-profile validation -----------------------------------------------------


def test_unknown_check_name_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    task_types["reasoning"]["checks"] = ["not_refused", "telepathy"]
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="unknown check"):
        load_quality_thresholds(path)


def test_mode_outside_judge_mode_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    task_types["reasoning"]["mode"] = "vibes"
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="mode"):
        load_quality_thresholds(path)


def test_pairwise_profile_missing_margin_threshold_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    del task_types["reasoning"]["fail_reference_margin_min"]
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="fail_reference_margin_min"):
        load_quality_thresholds(path)


def test_margin_threshold_outside_margin_order_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    task_types["reasoning"]["fail_reference_margin_min"] = "catastrophic"
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="fail_reference_margin_min"):
        load_quality_thresholds(path)


def test_margin_threshold_on_a_deterministic_profile_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    task_types["extraction"]["fail_reference_margin_min"] = "moderate"
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="must not set fail_reference_margin_min"):
        load_quality_thresholds(path)


def test_required_fields_check_without_required_fields_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    del task_types["extraction"]["required_fields"]
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="required_fields"):
        load_quality_thresholds(path)


def test_required_fields_declared_without_the_check_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    task_types["reasoning"]["required_fields"] = ["oops"]
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="does not use the required_fields check"):
        load_quality_thresholds(path)


def test_label_in_set_check_without_allowed_labels_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    del task_types["classification"]["allowed_labels"]
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="allowed_labels"):
        load_quality_thresholds(path)


def test_length_anomaly_check_without_length_bounds_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    del task_types["summarization"]["length_bounds"]
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="length_bounds"):
        load_quality_thresholds(path)


def test_length_bounds_min_not_less_than_max_raises(tmp_path: Path) -> None:
    task_types = _valid_task_types()
    task_types["summarization"]["length_bounds"] = [500, 10]
    path = _write_thresholds(tmp_path / "quality.yaml", task_types)

    with pytest.raises(ConfigurationError, match="0 <= min < max"):
        load_quality_thresholds(path)


# --- the real project config -----------------------------------------------------


def test_the_real_project_quality_config_loads_cleanly() -> None:
    """The shipped `config/quality.yaml` — not a fixture — actually loads
    against the real `TaskType` enum and this loader's full validation."""
    repo_root = Path(__file__).resolve().parents[2]
    thresholds = load_quality_thresholds(repo_root / "config" / "quality.yaml")

    assert set(thresholds.profiles) == set(TaskType)
    assert thresholds.get(TaskType.EXTRACTION).mode is JudgeMode.DETERMINISTIC
    assert thresholds.get(TaskType.CLASSIFICATION).mode is JudgeMode.DETERMINISTIC
    for pairwise_type in (TaskType.SUMMARIZATION, TaskType.REASONING, TaskType.DEFAULT):
        profile = thresholds.get(pairwise_type)
        assert profile.mode is JudgeMode.PAIRWISE
        assert profile.fail_reference_margin_min == "moderate"


def test_quality_schema_version_constant_matches_the_shipped_config() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load((repo_root / "config" / "quality.yaml").read_text(encoding="utf-8"))
    assert raw["version"] == QUALITY_SCHEMA_VERSION
