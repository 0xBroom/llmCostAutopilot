"""Reads `config/models.yaml` and resolves every price to `Decimal`, once.

Takes a path, not `Settings` — this function has no environment concerns and
no credentials. Deployment concerns (`api_key`, `api_base`) are injected
later, at Router construction. Keeping them out here is what lets a future
harness build a catalog without ever touching credentials or a Router.

The hard part is `litellm.model_cost` itself: it is **not** keyed by
`provider_model_id`. `"gpt-4o"` exists; `"openai/gpt-4o"` does not. There are
zero `"anthropic/..."` keys at all — `"claude-sonnet-4-5"` exists only bare.
`_resolve_cost_entry` does the two-step lookup this requires, and a genuine
miss is always fatal (`ConfigurationError`), never a silent zero price — a
model that looks free but is not corrupts savings accounting in the one
direction this system must never be wrong in.

This is the price-map half only: reading YAML, resolving against the bundled
map, applying a YAML `price:` override. Teaching the map a model it has never
heard of (`litellm.register_model()`) and detecting when an override diverges
from the map are a separate, later commit — registration must happen
*before* resolution, so a model that needs it cannot be priced at all until
that support exists. That ordering constraint is what makes this a real
bisectability boundary rather than an arbitrary split.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import yaml

from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.errors import ConfigurationError
from autopilot.domain.models import ComplexityTier, ModelConfig, PriceSource
from autopilot.infrastructure.litellm_env import litellm
from autopilot.infrastructure.pricing import to_price

CATALOG_SCHEMA_VERSION: Final = 1


def load_model_catalog(path: Path) -> ModelCatalog:
    """Read the YAML catalog, resolve every price to `Decimal`, return domain
    data. Every failure path raises `ConfigurationError` (or a subclass) —
    never a default, and never a silent zero price."""
    data = _read_yaml(path)

    version = data.get("version")
    if version != CATALOG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"model catalog at {path} declares schema version {version!r}, "
            f"but this build only supports version {CATALOG_SCHEMA_VERSION}"
        )

    entries = data.get("models")
    if not entries:
        raise ConfigurationError(f"model catalog at {path} declares no models")

    models = tuple(_build_model(entry) for entry in entries)
    return ModelCatalog(models=models)


# --- internals ---------------------------------------------------------------


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read model catalog at {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"model catalog at {path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(f"model catalog at {path} must be a YAML mapping at the top level")
    return data


def _resolve_cost_entry(provider_model_id: str) -> Mapping[str, Any] | None:
    """Two-step key resolution. `litellm.model_cost` is not keyed by
    `provider_model_id`: `"gpt-4o"` exists, `"openai/gpt-4o"` does not; there
    are zero `"anthropic/..."` keys at all. Read through the handle, never
    bound — a future `register_model` call's in-place update must always be
    visible here.
    """
    cost_map: Mapping[str, Any] = litellm.model_cost
    if provider_model_id in cost_map:
        return cast("Mapping[str, Any]", cost_map[provider_model_id])
    _, _, bare = provider_model_id.partition("/")
    if bare and bare in cost_map:
        return cast("Mapping[str, Any]", cost_map[bare])
    return None


def _resolve_price_field(
    *,
    model_key: str,
    field: str,
    override: object | None,
    map_entry: Mapping[str, Any] | None,
) -> tuple[Any, PriceSource]:
    """Resolve one price field: an explicit YAML override wins outright, else
    the bundled map, else a fatal `ConfigurationError` — never a silent zero.
    Returns the exact `Decimal` (via the single `to_price` chokepoint) and
    where it came from.
    """
    map_raw = map_entry.get(field) if map_entry is not None else None

    if override is not None:
        return to_price(override, field=field, model_key=model_key), PriceSource.CATALOG_OVERRIDE

    if map_raw is not None:
        return to_price(map_raw, field=field, model_key=model_key), PriceSource.PRICE_MAP

    raise ConfigurationError(
        f"{field} for model {model_key!r} is present in neither the price map "
        "nor a YAML override — refusing to price it at zero"
    )


def _combine_price_sources(*sources: PriceSource) -> PriceSource:
    return PriceSource.CATALOG_OVERRIDE if PriceSource.CATALOG_OVERRIDE in sources else sources[0]


def _build_model(entry: Mapping[str, Any]) -> ModelConfig:
    key = entry["key"]
    provider_model_id = entry["provider_model_id"]

    map_entry = _resolve_cost_entry(provider_model_id)
    price_overrides = entry.get("price") or {}

    input_cost, input_source = _resolve_price_field(
        model_key=key,
        field="input_cost_per_token",
        override=price_overrides.get("input_cost_per_token"),
        map_entry=map_entry,
    )
    output_cost, output_source = _resolve_price_field(
        model_key=key,
        field="output_cost_per_token",
        override=price_overrides.get("output_cost_per_token"),
        map_entry=map_entry,
    )

    max_context_tokens = entry.get("max_context_tokens")
    if max_context_tokens is None:
        map_max_input = map_entry.get("max_input_tokens") if map_entry is not None else None
        if map_max_input is None:
            raise ConfigurationError(
                f"model {key!r} declares no max_context_tokens, and the price map "
                "has no max_input_tokens to fall back to"
            )
        max_context_tokens = map_max_input

    return ModelConfig(
        key=key,
        provider=entry["provider"],
        provider_model_id=provider_model_id,
        input_cost_per_token=input_cost,
        output_cost_per_token=output_cost,
        max_context_tokens=max_context_tokens,
        quality_tier=ComplexityTier(entry["quality_tier"]),
        price_source=_combine_price_sources(input_source, output_source),
        max_output_tokens=entry.get("max_output_tokens"),
        api_base=entry.get("api_base"),
        enabled=entry.get("enabled", True),
        supports_json_mode=entry.get("supports_json_mode", False),
        expected_latency_ms=entry.get("expected_latency_ms"),
        baseline=entry.get("baseline", False),
        judge=entry.get("judge", False),
        judge_fallback=entry.get("judge_fallback"),
    )
