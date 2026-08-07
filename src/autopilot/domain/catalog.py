"""`ModelCatalog` — the collection-level invariants over `ModelConfig` entries.

Single-instance rules live on `ModelConfig.__post_init__` (unknown provider,
non-positive `max_output_tokens`, self-referential `judge_fallback`). Anything
that needs to see every model at once — uniqueness, exactly-one-baseline,
exactly-one-judge, upward-only fallback derivation — lives here. That split is
what lets `ModelConfig` stay a plain value object.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from autopilot.domain.errors import ConfigurationError, ModelNotInCatalogError
from autopilot.domain.models import ComplexityTier, ModelConfig


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """The whole catalog, with the rules that only make sense across entries."""

    models: tuple[ModelConfig, ...]

    def __post_init__(self) -> None:
        if not self.models:
            raise ConfigurationError("model catalog cannot be empty")

        seen: set[str] = set()
        for m in self.models:
            if m.key in seen:
                raise ConfigurationError(f"duplicate model key {m.key!r} in catalog")
            seen.add(m.key)

        enabled = [m for m in self.models if m.enabled]
        if not enabled:
            raise ConfigurationError("catalog has no enabled models")

        baselines = [m for m in enabled if m.baseline]
        if len(baselines) != 1:
            raise ConfigurationError(
                f"exactly one baseline model required, found {len(baselines)}: "
                f"{[m.key for m in baselines]}"
            )

        judges = [m for m in enabled if m.judge]
        if len(judges) != 1:
            raise ConfigurationError(
                f"exactly one judge model required, found {len(judges)}: {[m.key for m in judges]}"
            )
        judge = judges[0]

        for m in self.models:
            if m.judge_fallback is not None and m is not judge:
                raise ConfigurationError(
                    f"judge_fallback is set on {m.key!r}, which is not the judge model"
                )

        if judge.judge_fallback is not None:
            by_key = {m.key: m for m in self.models}
            fallback = by_key.get(judge.judge_fallback)
            if fallback is None or not fallback.enabled:
                raise ConfigurationError(
                    f"judge_fallback {judge.judge_fallback!r} does not name an "
                    "existing enabled catalog model"
                )
            if fallback.quality_tier < judge.quality_tier:
                raise ConfigurationError(
                    f"judge_fallback {judge.judge_fallback!r} (tier "
                    f"{int(fallback.quality_tier)}) must not rank below the judge's "
                    f"tier ({int(judge.quality_tier)})"
                )

    # --- lookup -----------------------------------------------------------
    def __iter__(self) -> Iterator[ModelConfig]:
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    def __contains__(self, key: object) -> bool:
        return any(m.key == key for m in self.models)

    def get(self, key: str) -> ModelConfig:
        found = self.find(key)
        if found is None:
            raise ModelNotInCatalogError(key)
        return found

    def find(self, key: str) -> ModelConfig | None:
        for m in self.models:
            if m.key == key:
                return m
        return None

    # --- views --------------------------------------------------------------
    @property
    def enabled(self) -> tuple[ModelConfig, ...]:
        return tuple(m for m in self.models if m.enabled)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(m.key for m in self.models)

    @property
    def baseline(self) -> ModelConfig:
        return next(m for m in self.enabled if m.baseline)

    @property
    def judge(self) -> ModelConfig:
        return next(m for m in self.enabled if m.judge)

    def by_tier(self, tier: ComplexityTier) -> tuple[ModelConfig, ...]:
        return tuple(m for m in self.models if m.quality_tier == tier)

    # --- the rule that makes upward-only true ------------------------------
    def fallback_chain(self, key: str) -> tuple[str, ...]:
        """Keys that may answer for `key` after a failure, best-value first.

        Strictly greater tier only. Same-tier is lateral, not upward, and is
        never a fallback target. This is the ONLY producer of fallback data
        in the system; there is no YAML field and no other function — that
        absence is what makes "a fallback never goes downward" structurally
        true instead of merely validated.
        """
        source = self.get(key)
        candidates = [
            m for m in self.enabled if m.quality_tier > source.quality_tier and m.key != key
        ]
        return tuple(m.key for m in sorted(candidates, key=lambda m: m.fallback_sort_key))
