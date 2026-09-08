"""Reads `config/quality.yaml` and resolves it into validated, frozen
per-task-type quality profiles.

Mirrors `model_catalog_loader.py`: a versioned schema, fail-loud
`ConfigurationError` on anything unexpected, and never a silent default. This
lives in `infrastructure` — not `domain` or `application` — specifically so it
MAY import `autopilot.config`-adjacent I/O concerns (reading a path, parsing
YAML) while `domain.quality` and `application.quality_judge` stay clean of
config entirely (`config-is-a-leaf`).

The loader's only job is validation and shape: turning YAML into
`QualityProfile` value objects that `domain.quality.resolve_verdict_label`
and `application.quality_judge.QualityJudge` can consume (a `mode`, an
ordered tuple of check names, and the pairwise margin threshold). Binding
`check_names` to the actual bound `Callable[[LLMResponse],
DeterministicCheckResult]` values `QualityJudge.evaluate()` expects is a
wiring concern for whichever caller assembles a live `QualityJudge` call —
this loader is assembled-but-not-wired, exactly like `QualityJudge` itself
and `LiteLLMGateway`'s `response_format` kwarg were in the prior slice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from autopilot.domain.errors import ConfigurationError
from autopilot.domain.quality import MARGIN_ORDER, JudgeMode, TaskType

QUALITY_SCHEMA_VERSION: Final = 1

KNOWN_CHECK_NAMES: Final[frozenset[str]] = frozenset(
    {
        "json_valid",
        "required_fields",
        "label_in_set",
        "not_refused",
        "not_truncated",
        "length_anomaly",
    }
)
"""The exact set of deterministic check names a YAML profile may name under
`checks:` — the loader's own mirror of the check functions defined in
`domain.quality`. A name outside this set is a config typo, never a silent
no-op check."""

_KNOWN_MODES: Final[frozenset[str]] = frozenset(mode.value for mode in JudgeMode)


class _Missing:
    """A private sentinel distinct from `None`.

    `entry.get(key)` cannot tell "the key is present and set to `None`" apart
    from "the key is absent" — both return `None`. The per-check parameter
    presence/absence checks below need that distinction (a key present but
    empty, e.g. `required_fields: []`, is still a config error, not merely
    absent), so every lookup in this module goes through `_get` instead of a
    bare `.get(...)`.
    """


_MISSING: Final = _Missing()


def _get(entry: Mapping[str, Any], key: str) -> Any:
    return entry.get(key, _MISSING)


@dataclass(frozen=True, slots=True)
class QualityProfile:
    """One task type's validated quality-contract configuration.

    `fail_reference_margin_min` is `None` iff `mode` is `DETERMINISTIC` — a
    deterministic profile never runs the pairwise judge, so it has no margin
    threshold to gate on. `required_fields` / `allowed_labels` /
    `length_bounds` are each set iff the corresponding check
    (`required_fields` / `label_in_set` / `length_anomaly`) is present in
    `check_names` — never populated for a check that is not requested, and
    never left unset for one that is.
    """

    task_type: TaskType
    mode: JudgeMode
    check_names: tuple[str, ...]
    fail_reference_margin_min: str | None
    required_fields: tuple[str, ...] | None = None
    allowed_labels: frozenset[str] | None = None
    length_bounds: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """The whole loaded set: one validated `QualityProfile` per `TaskType`."""

    profiles: Mapping[TaskType, QualityProfile]

    def get(self, task_type: TaskType) -> QualityProfile:
        return self.profiles[task_type]


def load_quality_thresholds(path: Path) -> QualityThresholds:
    """Read the YAML thresholds file, validate it exhaustively, return
    `QualityThresholds`. Every failure path raises `ConfigurationError` (or a
    subclass) — never a default, and never a silently-ignored unknown key."""
    data = _read_yaml(path)

    version = data.get("version")
    if version != QUALITY_SCHEMA_VERSION:
        raise ConfigurationError(
            f"quality thresholds at {path} declare schema version {version!r}, "
            f"but this build only supports version {QUALITY_SCHEMA_VERSION}"
        )

    task_types_raw = data.get("task_types")
    if not isinstance(task_types_raw, Mapping) or not task_types_raw:
        raise ConfigurationError(f"quality thresholds at {path} declare no task_types")

    _check_task_type_keys(path, task_types_raw)

    profiles = {
        task_type: _build_profile(path, task_type, task_types_raw[task_type.value])
        for task_type in TaskType
    }
    return QualityThresholds(profiles=profiles)


# --- internals ---------------------------------------------------------------


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read quality thresholds at {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"quality thresholds at {path} are not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"quality thresholds at {path} must be a YAML mapping at the top level"
        )
    return data


def _check_task_type_keys(path: Path, task_types_raw: Mapping[str, Any]) -> None:
    """Bidirectional cross-check against `TaskType`: every enum member MUST
    have a YAML profile, and every YAML key MUST name an enum member. Either
    direction failing is a fail-loud `ConfigurationError` — a typo'd task
    type key would otherwise silently fall through to no profile at all, and
    a stale/removed enum member would otherwise leave an orphaned profile no
    one is validating."""
    declared = frozenset(task_types_raw)
    known = frozenset(task_type.value for task_type in TaskType)

    missing = known - declared
    unknown = declared - known
    if missing or unknown:
        raise ConfigurationError(
            f"quality thresholds at {path} task_types keys do not match TaskType exactly: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _build_profile(path: Path, task_type: TaskType, entry: Mapping[str, Any]) -> QualityProfile:
    if not isinstance(entry, Mapping):
        raise ConfigurationError(
            f"quality thresholds at {path}: profile {task_type.value!r} must be a mapping"
        )

    mode_raw = entry.get("mode")
    if mode_raw not in _KNOWN_MODES:
        raise ConfigurationError(
            f"quality thresholds at {path}: profile {task_type.value!r} has mode {mode_raw!r}, "
            f"must be one of {sorted(_KNOWN_MODES)}"
        )
    mode = JudgeMode(mode_raw)

    checks_raw = entry.get("checks")
    if not checks_raw or not isinstance(checks_raw, list):
        raise ConfigurationError(
            f"quality thresholds at {path}: profile {task_type.value!r} declares no checks"
        )
    unknown_checks = [name for name in checks_raw if name not in KNOWN_CHECK_NAMES]
    if unknown_checks:
        raise ConfigurationError(
            f"quality thresholds at {path}: profile {task_type.value!r} names unknown "
            f"check(s) {unknown_checks}, known checks are {sorted(KNOWN_CHECK_NAMES)}"
        )
    check_names = tuple(checks_raw)

    fail_reference_margin_min = _resolve_margin_threshold(path, task_type, mode, entry)
    required_fields = _resolve_required_fields(path, task_type, check_names, entry)
    allowed_labels = _resolve_allowed_labels(path, task_type, check_names, entry)
    length_bounds = _resolve_length_bounds(path, task_type, check_names, entry)

    return QualityProfile(
        task_type=task_type,
        mode=mode,
        check_names=check_names,
        fail_reference_margin_min=fail_reference_margin_min,
        required_fields=required_fields,
        allowed_labels=allowed_labels,
        length_bounds=length_bounds,
    )


def _resolve_margin_threshold(
    path: Path, task_type: TaskType, mode: JudgeMode, entry: Mapping[str, Any]
) -> str | None:
    margin = _get(entry, "fail_reference_margin_min")
    if mode is JudgeMode.PAIRWISE:
        if margin is _MISSING or margin is None:
            raise ConfigurationError(
                f"quality thresholds at {path}: pairwise profile {task_type.value!r} "
                "must set fail_reference_margin_min"
            )
        if margin not in MARGIN_ORDER:
            raise ConfigurationError(
                f"quality thresholds at {path}: profile {task_type.value!r} "
                f"fail_reference_margin_min {margin!r} is not one of {MARGIN_ORDER}"
            )
        return str(margin)

    if margin is not _MISSING:
        raise ConfigurationError(
            f"quality thresholds at {path}: deterministic profile {task_type.value!r} "
            "must not set fail_reference_margin_min"
        )
    return None


def _resolve_required_fields(
    path: Path, task_type: TaskType, check_names: tuple[str, ...], entry: Mapping[str, Any]
) -> tuple[str, ...] | None:
    raw = _get(entry, "required_fields")
    uses_check = "required_fields" in check_names
    if uses_check:
        if raw is _MISSING or not raw:
            raise ConfigurationError(
                f"quality thresholds at {path}: profile {task_type.value!r} uses the "
                "required_fields check but declares no required_fields"
            )
        return tuple(raw)
    if raw is not _MISSING:
        raise ConfigurationError(
            f"quality thresholds at {path}: profile {task_type.value!r} declares "
            "required_fields but does not use the required_fields check"
        )
    return None


def _resolve_allowed_labels(
    path: Path, task_type: TaskType, check_names: tuple[str, ...], entry: Mapping[str, Any]
) -> frozenset[str] | None:
    raw = _get(entry, "allowed_labels")
    uses_check = "label_in_set" in check_names
    if uses_check:
        if raw is _MISSING or not raw:
            raise ConfigurationError(
                f"quality thresholds at {path}: profile {task_type.value!r} uses the "
                "label_in_set check but declares no allowed_labels"
            )
        return frozenset(raw)
    if raw is not _MISSING:
        raise ConfigurationError(
            f"quality thresholds at {path}: profile {task_type.value!r} declares "
            "allowed_labels but does not use the label_in_set check"
        )
    return None


def _resolve_length_bounds(
    path: Path, task_type: TaskType, check_names: tuple[str, ...], entry: Mapping[str, Any]
) -> tuple[int, int] | None:
    raw = _get(entry, "length_bounds")
    uses_check = "length_anomaly" in check_names
    if uses_check:
        if raw is _MISSING or not isinstance(raw, list) or len(raw) != 2:
            raise ConfigurationError(
                f"quality thresholds at {path}: profile {task_type.value!r} uses the "
                "length_anomaly check but declares no valid [min, max] length_bounds"
            )
        minimum, maximum = int(raw[0]), int(raw[1])
        if minimum < 0 or maximum <= minimum:
            raise ConfigurationError(
                f"quality thresholds at {path}: profile {task_type.value!r} length_bounds "
                f"{raw!r} must satisfy 0 <= min < max"
            )
        return (minimum, maximum)
    if raw is not _MISSING:
        raise ConfigurationError(
            f"quality thresholds at {path}: profile {task_type.value!r} declares "
            "length_bounds but does not use the length_anomaly check"
        )
    return None
