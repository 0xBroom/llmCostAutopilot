"""Router deployment shape — `build_model_list`.

Everything here asserts over the plain dicts and lists `build_model_list`
returns, never over a constructed `litellm.Router` instance. That split is
deliberate: litellm merges its own module-level globals into whatever Router
you build (`router.py:594-599` appends a wildcard fallback from
`litellm.default_fallbacks`; `router.py:603` falls back to
`litellm.content_policy_fallbacks` when the argument is empty), so "we did
not pass it" only proves something about *our own return value*, never about
a constructed Router. The assertions that must observe litellm's own merging
belong in `test_router_factory.py` (slice 6), against a real instance.

Zero `litellm` import in this file — confirmed by `make check`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import SecretStr

from autopilot.config.settings import Settings
from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.errors import MissingCredentialsError
from autopilot.domain.models import ComplexityTier, ModelConfig, PriceSource
from autopilot.infrastructure.router_factory import build_model_list
from tests.factories import CHEAP, EXPENSIVE, LOCAL, make_catalog


def _settings(**overrides: object) -> Settings:
    """No `.env`, no ambient environment — same idiom `test_settings.py`
    already established. `make_settings` (tests/factories.py) exists for
    later test files in this slice; this one predates it in the ordered task
    checklist and keeps its own copy rather than reaching forward for one.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


def _credentialed() -> Settings:
    """Every provider key the three-tier factory catalog (LOCAL/CHEAP/
    EXPENSIVE) needs, so `build_model_list` never raises
    `MissingCredentialsError` for a reason unrelated to the scenario under
    test."""
    return _settings(
        openai_api_key=SecretStr("sk-openai-test"),
        anthropic_api_key=SecretStr("sk-anthropic-test"),
    )


def _disabled_model(*, key: str = "retired", provider: str = "openai") -> ModelConfig:
    return ModelConfig(
        key=key,
        provider=provider,
        provider_model_id=f"{provider}/{key}",
        input_cost_per_token=Decimal("0.000001"),
        output_cost_per_token=Decimal("0.000002"),
        max_context_tokens=8_000,
        quality_tier=ComplexityTier.SIMPLE,
        price_source=PriceSource.PRICE_MAP,
        enabled=False,
    )


# --- build_model_list: grouping shape ---------------------------------------


def test_one_deployment_group_per_catalog_key() -> None:
    catalog = make_catalog()
    model_list = build_model_list(catalog, _credentialed())
    names = [entry["model_name"] for entry in model_list]
    assert len(names) == len(set(names))
    assert set(names) == {m.key for m in catalog.enabled}


def test_each_group_points_at_exactly_its_own_provider_model_id() -> None:
    """The subtle bug ADR-0001 warns about: two keys pointing at one
    underlying model is fine; one `model_name` covering two
    `provider_model_id`s is the bug."""
    catalog = make_catalog()
    model_list = build_model_list(catalog, _credentialed())
    for entry in model_list:
        model = catalog.get(entry["model_name"])
        assert entry["litellm_params"]["model"] == model.provider_model_id


def test_deployment_id_is_the_catalog_key() -> None:
    """F3: `model_info["id"]` is how the answering model is recovered after a
    fallback — `router.py:8038-8041` preserves a caller-supplied id."""
    catalog = make_catalog()
    model_list = build_model_list(catalog, _credentialed())
    for entry in model_list:
        assert entry["model_info"]["id"] == entry["model_name"]


def test_disabled_models_get_no_deployment() -> None:
    catalog = ModelCatalog(models=(LOCAL, CHEAP, EXPENSIVE, _disabled_model()))
    model_list = build_model_list(catalog, _credentialed())
    assert "retired" not in {entry["model_name"] for entry in model_list}


# --- build_model_list: credential enforcement -------------------------------


def test_missing_credentials_raise_at_build_time_not_first_request() -> None:
    """Same underlying behaviour the spec's model-catalog domain states under
    'Refusal to route' — one test satisfies both domains."""
    catalog = make_catalog()  # CHEAP is anthropic, EXPENSIVE is openai
    settings = _settings()  # no keys at all -> configured_providers == {"ollama"}

    with pytest.raises(MissingCredentialsError) as exc_info:
        build_model_list(catalog, settings)

    assert exc_info.value.provider in {"anthropic", "openai"}


def test_enabled_false_is_the_sanctioned_credential_opt_out() -> None:
    """Same dedup as the missing-credentials test above: `enabled: false`
    lets an uncredentialed provider's model sit in the catalog without
    blocking every other model from routing."""
    uncredentialed_but_disabled = _disabled_model(key="haiku-4-5", provider="anthropic")
    catalog = ModelCatalog(models=(LOCAL, EXPENSIVE, uncredentialed_but_disabled))
    settings = _settings(openai_api_key=SecretStr("sk-openai-test"))  # no anthropic key

    model_list = build_model_list(catalog, settings)

    assert {entry["model_name"] for entry in model_list} == {"llama3-local", "gpt-4o"}


def test_key_baked_into_the_deployment_entry() -> None:
    settings = _credentialed()
    catalog = make_catalog()
    model_list = build_model_list(catalog, settings)
    cheap_entry = next(entry for entry in model_list if entry["model_name"] == "haiku-3-5")
    assert cheap_entry["litellm_params"]["api_key"] == "sk-anthropic-test"


def test_no_dependency_on_ambient_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key that only exists in `os.environ` — never passed into `Settings`
    — must not leak into the deployment entry. `build_model_list` reads
    credentials exclusively from the injected `Settings` object, never from
    `os.environ`."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ambient-trap")
    settings = _credentialed()  # explicit key, unrelated to the env var above
    catalog = make_catalog()

    model_list = build_model_list(catalog, settings)

    cheap_entry = next(entry for entry in model_list if entry["model_name"] == "haiku-3-5")
    assert cheap_entry["litellm_params"]["api_key"] == "sk-anthropic-test"
    assert cheap_entry["litellm_params"]["api_key"] != "sk-ambient-trap"
