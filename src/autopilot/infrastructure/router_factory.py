"""Deployment builders for `litellm.Router`, driven entirely by the catalog.

Two hazards make this module worth reading closely before touching it, both
found by reading litellm 1.95.0's own source rather than assuming:

1. **Grouping is structural, not a parameter.** `_deployment_for` takes one
   `ModelConfig` and returns one deployment dict. There is no expression in
   this module capable of emitting two deployments under one `model_name` —
   the comprehension in `build_model_list` iterates models, the name is
   `model.key`, and `ModelCatalog.__post_init__` guarantees keys are unique.
   Grouping by tier instead of by catalog key would require changing
   `_deployment_for`'s signature to accept a collection, which is a visible,
   deliberate act, not something a future edit does by accident. ADR-0001
   states the corollary: one deployment group per catalog key, never per
   tier, or model selection quietly reverts to litellm's own
   `simple-shuffle`.
2. **litellm merges its own module-level globals into the Router it builds**
   (`router.py:594-599` appends `{"*": litellm.default_fallbacks}` to
   `self.fallbacks`; `router.py:603` falls back to
   `litellm.content_policy_fallbacks` when the argument is empty). So "we did
   not pass a wildcard" or "we did not pass content_policy_fallbacks" proves
   nothing about a *constructed Router* — only about the argument. Those
   assertions belong in `test_router_factory.py` (the next slice), against a
   real instance. What this module's own tests can trust is its own return
   values: plain dicts and lists litellm has not yet had a chance to touch.

`build_model_list` and `build_fallbacks` are public and deliberately split
from `build_router` (not yet written) for exactly that reason: every shape
assertion here runs with zero `litellm` import, in the fast unit tier.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

from pydantic import SecretStr

from autopilot.config.settings import Settings
from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.errors import ConfigurationError, MissingCredentialsError
from autopilot.domain.models import ModelConfig

_KEY_ACCESSORS: Final[Mapping[str, Callable[[Settings], SecretStr | None]]] = {
    "openai": lambda s: s.openai_api_key,
    "anthropic": lambda s: s.anthropic_api_key,
    "ollama": lambda _s: None,  # local, no credential
}
"""One accessor per known provider. `tests/unit/test_credential_boundary.py::
test_every_known_provider_has_a_credential_accessor` asserts this set equals
`KNOWN_PROVIDERS` exactly, so adding a provider to the domain without
teaching this module how to fetch its credential fails immediately rather
than at the first request."""


def _api_key_for(provider: str, settings: Settings) -> str | None:
    """The only `.get_secret_value()` call site in `src/` — asserted by
    `tests/unit/test_credential_boundary.py::
    test_get_secret_value_is_called_in_exactly_one_module`."""
    secret = _KEY_ACCESSORS[provider](settings)
    return None if secret is None else secret.get_secret_value()


def _api_base_for(model: ModelConfig, settings: Settings) -> str | None:
    """A catalog-level override wins; otherwise the ollama default from
    `Settings`. `api_base` is deployment config, the same class of thing as
    `api_key` — never stored in the catalog for any other provider, which is
    why `load_model_catalog` needs no `Settings` at all."""
    if model.api_base is not None:
        return model.api_base
    return settings.ollama_api_base if model.provider == "ollama" else None


def _require_credentials(catalog: ModelCatalog, settings: Settings) -> None:
    """Refuse at build time, not on the first request. `enabled: false` is
    the sanctioned opt-out — a disabled entry never reaches this loop, since
    it iterates `catalog.enabled`."""
    for model in catalog.enabled:
        if model.provider not in settings.configured_providers:
            raise MissingCredentialsError(model.provider)


def _deployment_for(model: ModelConfig, settings: Settings) -> dict[str, Any]:
    """ONE `ModelConfig` in, ONE deployment out. This signature is the
    enforcement: grouping by tier would require this function to accept a
    collection, which is a change nobody makes by accident."""
    return {
        "model_name": model.key,  # the group name IS the catalog key
        "litellm_params": {
            "model": model.provider_model_id,
            "api_key": _api_key_for(model.provider, settings),
            "api_base": _api_base_for(model, settings),
            "timeout": settings.request_timeout_s,  # a floor; the per-call value wins
        },
        "model_info": {
            "id": model.key,  # F3 — how the answering model is recovered after a fallback
            "base_model": model.provider_model_id,
        },
    }


def build_model_list(catalog: ModelCatalog, settings: Settings) -> list[dict[str, Any]]:
    """One deployment per enabled catalog entry. Raises
    `MissingCredentialsError` before returning if any enabled model's
    provider has no configured credential — failing at build time is the
    entire point."""
    _require_credentials(catalog, settings)
    return [_deployment_for(model, settings) for model in catalog.enabled]


def build_fallbacks(catalog: ModelCatalog) -> list[dict[str, list[str]]]:
    """litellm's fallbacks shape, derived exclusively from
    `catalog.fallback_chain`.

    The inner loop is belt-and-braces: `fallback_chain` cannot itself
    produce a downward chain — there is no field to corrupt, which is what
    makes "a fallback never goes downward" structurally true rather than
    merely validated — but the Router is where a mistake would become a
    wrong provider call, so this re-checks anyway rather than trusting the
    guarantee transitively.
    """
    out: list[dict[str, list[str]]] = []
    for model in catalog.enabled:
        chain = catalog.fallback_chain(model.key)
        if not chain:
            continue  # top tier: nothing above it
        for target_key in chain:
            target = catalog.get(target_key)
            if target.quality_tier <= model.quality_tier:
                raise ConfigurationError(
                    f"fallback {model.key!r} -> {target_key!r} is not upward "
                    f"(tier {int(model.quality_tier)} -> {int(target.quality_tier)})"
                )
        out.append({model.key: list(chain)})
    return out
