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

Models the map has never heard of (a local ollama deployment, typically) are
taught via `litellm.register_model()` *before* resolution — `register_model`
can re-key an entry under a builtin key it happens to match, so the loader
re-runs the exact same two-step resolution afterwards and raises if it still
misses, rather than assuming the key it passed landed.

When a YAML `price:` override disagrees with the map, the override wins and
the loader logs a structured warning naming the model, the field, both
values and their ratio — disagreeing with the map is the entire point of an
override (a negotiated rate, an enterprise price), so this is a signal to
read, not a reason to fail the load.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import structlog
import yaml
from structlog.stdlib import BoundLogger

from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.errors import ConfigurationError
from autopilot.domain.models import ComplexityTier, ModelConfig, PriceSource
from autopilot.infrastructure.litellm_env import litellm
from autopilot.infrastructure.pricing import to_price

CATALOG_SCHEMA_VERSION: Final = 1

# Priority order when a model's two price fields disagree on where they came
# from (e.g. one overridden, one from the map). Higher index wins.
_SOURCE_PRIORITY: Final[tuple[PriceSource, ...]] = (
    PriceSource.PRICE_MAP,
    PriceSource.REGISTERED,
    PriceSource.CATALOG_OVERRIDE,
)


def load_model_catalog(path: Path, *, logger: BoundLogger | None = None) -> ModelCatalog:
    """Read the YAML catalog, resolve every price to `Decimal`, return domain
    data. Every failure path raises `ConfigurationError` (or a subclass) —
    never a default, and never a silent zero price."""
    log = logger if logger is not None else structlog.get_logger(__name__)
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

    models = tuple(_build_model(entry, logger=log) for entry in entries)
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


def _register_pricing(provider_model_id: str, register_pricing: Mapping[str, Any]) -> None:
    """Teach the gateway a price for a model it has never heard of, so the
    lookup that follows is uniform instead of branching on "is this in the
    map". `register_model` mutates `litellm.model_cost` in place — see
    `litellm_env.py`'s docstring on why nothing here binds that dict by name.
    """
    litellm.register_model({provider_model_id: dict(register_pricing)})


def _resolve_cost_entry(provider_model_id: str) -> Mapping[str, Any] | None:
    """Two-step key resolution. `litellm.model_cost` is not keyed by
    `provider_model_id`: `"gpt-4o"` exists, `"openai/gpt-4o"` does not; there
    are zero `"anthropic/..."` keys at all. Read through the handle, never
    bound — `register_model`'s in-place update must always be visible here.
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
    registered: bool,
    logger: BoundLogger,
) -> tuple[Any, PriceSource]:
    """Resolve one price field, in the precedence order the design fixes:
    an explicit override always wins (and is checked for divergence against
    the map, if the map has an opinion); otherwise the map (marked
    `REGISTERED` if we just taught it that price, `PRICE_MAP` if it already
    knew); otherwise a fatal `ConfigurationError` — never a silent zero.
    Returns the exact `Decimal` (via the single `to_price` chokepoint) and
    where it came from.
    """
    map_raw = map_entry.get(field) if map_entry is not None else None

    if override is not None:
        resolved = to_price(override, field=field, model_key=model_key)
        if map_raw is not None:
            _warn_if_diverges(
                model_key=model_key,
                field=field,
                map_raw=map_raw,
                resolved=resolved,
                logger=logger,
            )
        return resolved, PriceSource.CATALOG_OVERRIDE

    if map_raw is not None:
        source = PriceSource.REGISTERED if registered else PriceSource.PRICE_MAP
        return to_price(map_raw, field=field, model_key=model_key), source

    raise ConfigurationError(
        f"{field} for model {model_key!r} is present in neither the price map "
        "nor a YAML override — refusing to price it at zero"
    )


def _warn_if_diverges(
    *, model_key: str, field: str, map_raw: object, resolved: Any, logger: BoundLogger
) -> None:
    """Disagreeing with the map is the entire purpose of an override — a
    negotiated rate, an enterprise price, a stale map entry. This never
    fails the load; it makes the disagreement visible so a stale override
    surviving a provider price cut doesn't quietly overstate savings
    forever."""
    map_price = to_price(map_raw, field=field, model_key=model_key)
    if map_price == resolved:
        return
    ratio = None if map_price == 0 else resolved / map_price
    logger.warning(
        "model_catalog.price_override_diverges_from_map",
        model_key=model_key,
        field=field,
        map_value=str(map_price),
        override_value=str(resolved),
        ratio=str(ratio) if ratio is not None else None,
    )


def _combine_price_sources(*sources: PriceSource) -> PriceSource:
    return max(sources, key=_SOURCE_PRIORITY.index)


def _build_model(entry: Mapping[str, Any], *, logger: BoundLogger) -> ModelConfig:
    key = entry["key"]
    provider_model_id = entry["provider_model_id"]

    register_pricing = entry.get("register_pricing")
    if register_pricing is not None:
        _register_pricing(provider_model_id, register_pricing)

    map_entry = _resolve_cost_entry(provider_model_id)
    if register_pricing is not None and map_entry is None:
        raise ConfigurationError(
            f"model {key!r}: registered pricing for {provider_model_id!r} did not "
            "resolve after litellm.register_model() — it may have been re-keyed "
            "under a different builtin key"
        )

    price_overrides = entry.get("price") or {}
    registered = register_pricing is not None

    input_cost, input_source = _resolve_price_field(
        model_key=key,
        field="input_cost_per_token",
        override=price_overrides.get("input_cost_per_token"),
        map_entry=map_entry,
        registered=registered,
        logger=logger,
    )
    output_cost, output_source = _resolve_price_field(
        model_key=key,
        field="output_cost_per_token",
        override=price_overrides.get("output_cost_per_token"),
        map_entry=map_entry,
        registered=registered,
        logger=logger,
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
