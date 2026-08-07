"""`ModelCatalog` — cross-instance invariants and upward-only fallback derivation.

Single-instance rules (unknown provider, non-positive `max_output_tokens`,
self-referential `judge_fallback`) are covered by `test_domain_models.py`.
Everything here needs to see the whole catalog at once.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.errors import ConfigurationError, ModelNotInCatalogError
from autopilot.domain.models import ComplexityTier, ModelConfig, PriceSource
from tests.factories import CHEAP, EXPENSIVE, LOCAL, make_catalog


def _model(
    key: str,
    tier: ComplexityTier,
    *,
    input_cost: str = "0",
    output_cost: str = "0",
    enabled: bool = True,
    baseline: bool = False,
    judge: bool = False,
    judge_fallback: str | None = None,
) -> ModelConfig:
    return ModelConfig(
        key=key,
        provider="openai",
        provider_model_id=f"openai/{key}",
        input_cost_per_token=Decimal(input_cost),
        output_cost_per_token=Decimal(output_cost),
        max_context_tokens=100_000,
        quality_tier=tier,
        price_source=PriceSource.PRICE_MAP,
        enabled=enabled,
        baseline=baseline,
        judge=judge,
        judge_fallback=judge_fallback,
    )


# --- catalog-level invariants ---------------------------------------------


def test_duplicate_key_rejected() -> None:
    a = _model("dup", ComplexityTier.SIMPLE, baseline=True)
    b = _model("dup", ComplexityTier.COMPLEX, judge=True)
    with pytest.raises(ConfigurationError, match=r"duplicate.*'dup'"):
        ModelCatalog(models=(a, b))


def test_zero_baselines_rejected() -> None:
    models = (
        _model("a", ComplexityTier.SIMPLE, judge=True),
        _model("b", ComplexityTier.MODERATE),
        _model("c", ComplexityTier.COMPLEX),
    )
    with pytest.raises(ConfigurationError, match="exactly one baseline"):
        ModelCatalog(models=models)


def test_two_baselines_rejected() -> None:
    models = (
        _model("a", ComplexityTier.SIMPLE, baseline=True, judge=True),
        _model("b", ComplexityTier.COMPLEX, baseline=True),
    )
    with pytest.raises(ConfigurationError, match="exactly one baseline"):
        ModelCatalog(models=models)


def test_unresolvable_judge_fallback_rejected() -> None:
    judge = _model(
        "gpt-4o",
        ComplexityTier.COMPLEX,
        baseline=True,
        judge=True,
        judge_fallback="nonexistent-key",
    )
    with pytest.raises(ConfigurationError, match="judge_fallback"):
        ModelCatalog(models=(judge,))


def test_empty_catalog_rejected() -> None:
    with pytest.raises(ConfigurationError, match="cannot be empty"):
        ModelCatalog(models=())


def test_catalog_with_no_enabled_models_rejected() -> None:
    models = (_model("a", ComplexityTier.SIMPLE, baseline=True, judge=True, enabled=False),)
    with pytest.raises(ConfigurationError, match="no enabled models"):
        ModelCatalog(models=models)


def test_judge_fallback_only_allowed_on_the_judge_model() -> None:
    models = (
        _model("a", ComplexityTier.SIMPLE, baseline=True, judge_fallback="b"),
        _model("b", ComplexityTier.COMPLEX, judge=True),
    )
    with pytest.raises(ConfigurationError, match="not the judge model"):
        ModelCatalog(models=models)


def test_judge_fallback_must_not_rank_below_the_judge() -> None:
    """Belt-and-braces: even a resolvable judge_fallback naming a lower-tier
    model is rejected — the same upward-only rule that governs fallback_chain
    applies to the judge's own escalation path."""
    models = (
        _model("lower", ComplexityTier.SIMPLE, baseline=True),
        _model("judge", ComplexityTier.COMPLEX, judge=True, judge_fallback="lower"),
    )
    with pytest.raises(ConfigurationError, match="must not rank below"):
        ModelCatalog(models=models)


# --- lookup and views ------------------------------------------------------


def test_lookup_and_views() -> None:
    catalog = make_catalog()
    assert catalog.get("llama3-local") is LOCAL
    assert catalog.find("nonexistent") is None
    assert "haiku-3-5" in catalog
    assert len(catalog) == 3
    assert catalog.enabled == catalog.models
    assert set(catalog.keys) == {"llama3-local", "haiku-3-5", "gpt-4o"}
    assert catalog.baseline.key == "gpt-4o"
    assert catalog.judge.key == "gpt-4o"
    assert catalog.by_tier(ComplexityTier.SIMPLE) == (LOCAL,)
    assert list(catalog) == [LOCAL, CHEAP, EXPENSIVE]


def test_get_raises_for_unknown_key() -> None:
    catalog = make_catalog()
    with pytest.raises(ModelNotInCatalogError):
        catalog.get("nonexistent")


# --- upward-only fallback derivation ---------------------------------------


def test_upward_chain_sorted_by_tier_then_cost() -> None:
    models = (
        _model(
            "haiku-3-5",
            ComplexityTier.SIMPLE,
            input_cost="0.0000008",
            output_cost="0.000004",
            baseline=True,
        ),
        _model(
            "gpt-4o-mini",
            ComplexityTier.MODERATE,
            input_cost="0.00000075",
            output_cost="0.00000075",
        ),
        _model(
            "sonnet-4",
            ComplexityTier.MODERATE,
            input_cost="0.0000015",
            output_cost="0.0000015",
        ),
        _model(
            "opus-4",
            ComplexityTier.COMPLEX,
            input_cost="0.00003",
            output_cost="0.00003",
            judge=True,
        ),
    )
    catalog = ModelCatalog(models=models)
    assert catalog.fallback_chain("haiku-3-5") == ("gpt-4o-mini", "sonnet-4", "opus-4")


def test_same_tier_models_excluded() -> None:
    models = (
        _model("sonnet-4", ComplexityTier.MODERATE, baseline=True),
        _model("gpt-4o-mini", ComplexityTier.MODERATE, judge=True),
    )
    catalog = ModelCatalog(models=models)
    assert "gpt-4o-mini" not in catalog.fallback_chain("sonnet-4")


def test_disabled_models_excluded_from_any_chain() -> None:
    models = (
        _model("haiku-3-5", ComplexityTier.SIMPLE, baseline=True),
        _model("sonnet-4", ComplexityTier.MODERATE, judge=True),
        _model("opus-4", ComplexityTier.COMPLEX, enabled=False),
    )
    catalog = ModelCatalog(models=models)
    assert "opus-4" not in catalog.fallback_chain("haiku-3-5")


def test_top_tier_model_has_no_fallback() -> None:
    catalog = make_catalog()
    assert catalog.fallback_chain("gpt-4o") == ()


def test_every_fallback_target_outranks_its_source() -> None:
    """Design-mandated property test: upward-only is a postcondition of
    `fallback_chain`, not a field validated at construction. This is its
    primary enforcement, over a catalog spanning all three tiers."""
    models = (
        _model(
            "t1",
            ComplexityTier.SIMPLE,
            input_cost="0.0000001",
            output_cost="0.0000001",
            baseline=True,
        ),
        _model("t2a", ComplexityTier.MODERATE, input_cost="0.000001", output_cost="0.000001"),
        _model(
            "t2b",
            ComplexityTier.MODERATE,
            input_cost="0.0000015",
            output_cost="0.0000015",
            judge=True,
        ),
        _model("t3", ComplexityTier.COMPLEX, input_cost="0.00003", output_cost="0.00003"),
    )
    catalog = ModelCatalog(models=models)
    for source_key in catalog.keys:
        source_tier = catalog.get(source_key).quality_tier
        for target_key in catalog.fallback_chain(source_key):
            assert catalog.get(target_key).quality_tier > source_tier
