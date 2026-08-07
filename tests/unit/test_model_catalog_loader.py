"""`infrastructure.model_catalog_loader.load_model_catalog` — YAML I/O and
price-map resolution (the basic path: no `register_model`, no divergence
detection — those land in a later commit, see the module docstring).

All fixtures are written to `tmp_path`. Prices are resolved against the real,
locally-bundled `litellm==1.95.0` price map (`LITELLM_LOCAL_MODEL_COST_MAP`
pins it — see `infrastructure/litellm_env.py`), never over the network. Every
`provider_model_id` used below was verified this session to resolve against
that exact map via the two-step lookup (`_resolve_cost_entry`) — see
apply-progress for the verification log.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from autopilot.domain.errors import ConfigurationError
from autopilot.domain.models import PriceSource
from autopilot.infrastructure.litellm_env import litellm
from autopilot.infrastructure.model_catalog_loader import (
    CATALOG_SCHEMA_VERSION,
    load_model_catalog,
)


def _write_catalog(path: Path, models: list[dict[str, Any]], *, version: int | None = 1) -> Path:
    payload: dict[str, Any] = {"models": models}
    if version is not None:
        payload["version"] = version
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _gpt_4o_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": "gpt-4o",
        "provider": "openai",
        "provider_model_id": "openai/gpt-4o",
        "quality_tier": 3,
        "baseline": True,
        "judge": True,
    }
    entry.update(overrides)
    return entry


def test_prices_from_the_map_are_exact(tmp_path: Path) -> None:
    """The map default resolves to the human-readable Decimal a person would
    write, not the float's binary expansion — this is `to_price`'s job,
    exercised end to end through the loader."""
    path = _write_catalog(tmp_path / "models.yaml", [_gpt_4o_entry()])

    catalog = load_model_catalog(path)

    model = catalog.get("gpt-4o")
    map_entry = litellm.model_cost["gpt-4o"]
    assert model.input_cost_per_token == Decimal(str(map_entry["input_cost_per_token"]))
    assert model.output_cost_per_token == Decimal(str(map_entry["output_cost_per_token"]))
    assert model.price_source == PriceSource.PRICE_MAP


def test_yaml_price_override_wins_for_the_overridden_field_only(tmp_path: Path) -> None:
    """An override replaces one field's value; the sibling field still comes
    from the map."""
    path = _write_catalog(
        tmp_path / "models.yaml",
        [_gpt_4o_entry(price={"output_cost_per_token": "0.000005"})],
    )

    catalog = load_model_catalog(path)

    model = catalog.get("gpt-4o")
    map_entry = litellm.model_cost["gpt-4o"]
    assert model.output_cost_per_token == Decimal("0.000005")
    assert model.input_cost_per_token == Decimal(str(map_entry["input_cost_per_token"]))
    assert model.price_source == PriceSource.CATALOG_OVERRIDE


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path / "models.yaml", [_gpt_4o_entry()], version=2)

    with pytest.raises(ConfigurationError, match="schema version"):
        load_model_catalog(path)


def test_missing_schema_version_raises(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path / "models.yaml", [_gpt_4o_entry()], version=None)

    with pytest.raises(ConfigurationError, match="schema version"):
        load_model_catalog(path)


def test_max_context_tokens_falls_back_to_the_maps_max_input_tokens(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path / "models.yaml", [_gpt_4o_entry()])

    catalog = load_model_catalog(path)

    expected = litellm.model_cost["gpt-4o"]["max_input_tokens"]
    assert catalog.get("gpt-4o").max_context_tokens == expected


def test_max_context_tokens_raises_when_neither_yaml_nor_map_has_it(tmp_path: Path) -> None:
    """A `provider_model_id` absent from the map, with both prices supplied via
    an explicit override (so price resolution itself does not fail first) and
    no `max_context_tokens` in YAML, has nothing to fall back to."""
    path = _write_catalog(
        tmp_path / "models.yaml",
        [
            {
                "key": "nowhere",
                "provider": "openai",
                "provider_model_id": "openai/does-not-exist-anywhere",
                "quality_tier": 1,
                "baseline": True,
                "judge": True,
                "price": {
                    "input_cost_per_token": "0.000001",
                    "output_cost_per_token": "0.000002",
                },
            }
        ],
    )

    with pytest.raises(ConfigurationError, match="max_context_tokens"):
        load_model_catalog(path)


def test_missing_price_in_neither_map_nor_overlay_raises(tmp_path: Path) -> None:
    path = _write_catalog(
        tmp_path / "models.yaml",
        [
            {
                "key": "nowhere",
                "provider": "openai",
                "provider_model_id": "openai/does-not-exist-anywhere",
                "quality_tier": 1,
                "baseline": True,
                "judge": True,
                "max_context_tokens": 8192,
            }
        ],
    )

    with pytest.raises(ConfigurationError, match="input_cost_per_token"):
        load_model_catalog(path)


def test_negative_price_override_raises_before_any_model_config_is_constructed(
    tmp_path: Path,
) -> None:
    path = _write_catalog(
        tmp_path / "models.yaml",
        [_gpt_4o_entry(price={"input_cost_per_token": "-0.0001"})],
    )

    with pytest.raises(ConfigurationError, match="cannot be negative"):
        load_model_catalog(path)


def test_schema_constant_is_one() -> None:
    assert CATALOG_SCHEMA_VERSION == 1
