"""`infrastructure.model_catalog_loader.load_model_catalog` — YAML I/O,
price-map resolution, and registration for models the map has never heard of.

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
from structlog.testing import capture_logs

from autopilot.domain.errors import ConfigurationError
from autopilot.domain.models import ComplexityTier, PriceSource
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


# --- price-map resolution (part 1) -----------------------------------------


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


# --- registration and divergence (part 2) -----------------------------------


def test_local_model_registers_before_lookup(tmp_path: Path) -> None:
    """`ollama_chat/llama3.1` is absent from litellm 1.95.0's bundled map
    (verified this session) — the loader must teach it via
    `litellm.register_model()` before resolving its price."""
    path = _write_catalog(
        tmp_path / "models.yaml",
        [
            {
                "key": "llama3-local",
                "provider": "ollama",
                "provider_model_id": "ollama_chat/llama3.1",
                "quality_tier": 1,
                "max_context_tokens": 8192,
                "register_pricing": {
                    "litellm_provider": "ollama",
                    "mode": "chat",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                    "max_input_tokens": 8192,
                    "max_output_tokens": 8192,
                },
            },
            _gpt_4o_entry(),
        ],
    )

    catalog = load_model_catalog(path)

    model = catalog.get("llama3-local")
    assert model.input_cost_per_token == Decimal("0")
    assert model.output_cost_per_token == Decimal("0")
    assert model.price_source == PriceSource.REGISTERED
    assert model.quality_tier == ComplexityTier.SIMPLE


def test_registered_model_re_resolves_and_raises_if_it_still_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4: `register_model` can re-key an entry under a builtin key. Assuming
    the key we passed landed is the bug — the loader must re-run the same
    two-step resolution after registering, and fail loudly if it still
    misses, rather than silently pricing the model at zero."""
    monkeypatch.setattr(litellm, "register_model", lambda model_cost: None)
    path = _write_catalog(
        tmp_path / "models.yaml",
        [
            {
                "key": "phantom-local",
                "provider": "ollama",
                "provider_model_id": "ollama_chat/definitely-not-registered",
                "quality_tier": 1,
                "max_context_tokens": 8192,
                "register_pricing": {
                    "litellm_provider": "ollama",
                    "mode": "chat",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
            }
        ],
    )

    with pytest.raises(ConfigurationError, match="phantom-local"):
        load_model_catalog(path)


def test_override_divergence_is_recorded(tmp_path: Path) -> None:
    """When the overlay and the map disagree, the overlay wins and the loader
    emits a structured warning naming the model, the field, both values, and
    the ratio between them."""
    path = _write_catalog(
        tmp_path / "models.yaml",
        [_gpt_4o_entry(price={"input_cost_per_token": "0.0000027"})],
    )

    with capture_logs() as logs:
        catalog = load_model_catalog(path)

    assert catalog.get("gpt-4o").input_cost_per_token == Decimal("0.0000027")

    warnings = [entry for entry in logs if entry["log_level"] == "warning"]
    assert len(warnings) == 1
    warning = warnings[0]
    map_input = litellm.model_cost["gpt-4o"]["input_cost_per_token"]
    assert warning["model_key"] == "gpt-4o"
    assert warning["field"] == "input_cost_per_token"
    assert warning["map_value"] == str(Decimal(str(map_input)))
    assert warning["override_value"] == str(Decimal("0.0000027"))
    assert warning["ratio"] is not None


def test_no_divergence_warning_when_override_matches_the_map(tmp_path: Path) -> None:
    map_input = litellm.model_cost["gpt-4o"]["input_cost_per_token"]
    path = _write_catalog(
        tmp_path / "models.yaml",
        [_gpt_4o_entry(price={"input_cost_per_token": str(map_input)})],
    )

    with capture_logs() as logs:
        load_model_catalog(path)

    assert not [entry for entry in logs if entry["log_level"] == "warning"]


# --- the real project catalog ------------------------------------------------


def test_the_real_project_catalog_loads_and_derives_the_expected_fallbacks() -> None:
    """A dedicated, deliberate exception to "no test reads the real
    config/models.yaml": this is the one place that proves the shipped
    catalog — not a fixture — actually loads against the installed litellm
    map, resolves every provider_model_id (directly, via register_pricing, or
    via an explicit override), and derives the fallback chains the design's
    worked example names."""
    repo_root = Path(__file__).resolve().parents[2]
    catalog = load_model_catalog(repo_root / "config" / "models.yaml")

    assert set(catalog.keys) == {"llama3-local", "haiku-4-5", "gpt-4o", "sonnet-4-5"}
    assert catalog.baseline.key == "gpt-4o"
    assert catalog.judge.key == "sonnet-4-5"
    assert catalog.fallback_chain("llama3-local") == ("haiku-4-5", "gpt-4o", "sonnet-4-5")
    assert catalog.fallback_chain("haiku-4-5") == ("gpt-4o", "sonnet-4-5")
    assert catalog.fallback_chain("gpt-4o") == ()
    assert catalog.fallback_chain("sonnet-4-5") == ()
