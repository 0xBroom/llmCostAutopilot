"""`build_router` — assertions over a REAL constructed `litellm.Router`.

`tests/unit/test_router_deployments.py` (slice 5) deliberately asserts only
over the plain dicts/lists `build_model_list`/`build_fallbacks` return, never
over a constructed Router — because litellm merges its own module-level
globals into whatever Router you build:

- `router.py:594-599` appends `{"*": litellm.default_fallbacks}` to
  `self.fallbacks` whenever `default_fallbacks` (the constructor arg) or
  `litellm.default_fallbacks` (the module global) is not `None`.
- `router.py:601` sets `self.context_window_fallbacks` to
  `context_window_fallbacks or litellm.context_window_fallbacks` — the
  constructor default is `[]` (falsy), so an empty argument falls through to
  the module global.
- `router.py:603` does the identical thing for `content_policy_fallbacks`.

So "we did not pass a wildcard" or "we did not pass content_policy_fallbacks"
proves nothing about a *constructed Router* — only about the argument we
handed it. Every assertion in this file is made on the instance `build_router`
returns, never on the arguments passed into it.

Imports litellm through the shim (`autopilot.infrastructure.litellm_env`) —
the sanctioned single entrypoint. `.importlinter`'s `litellm-single-entrypoint`
contract only polices `src/autopilot/**`, not `tests/**`, but this file keeps
to the same one-import convention as `test_model_catalog_loader.py` (slice 4)
anyway: there is exactly one place in the whole repo, including tests, that
writes `import litellm`.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import SecretStr

from autopilot.config.settings import Settings
from autopilot.domain.errors import ConfigurationError
from autopilot.infrastructure.router_factory import (
    ROUTER_ALLOWED_FAILS,
    ROUTER_COOLDOWN_SECONDS,
    ROUTER_NUM_RETRIES,
    ROUTER_RETRY_AFTER_SECONDS,
    ROUTING_STRATEGY,
    _verify_model_ids_round_tripped,
    build_router,
)
from tests.factories import make_catalog


def _credentialed() -> Settings:
    """Every provider the three-tier factory catalog needs, with two
    DISTINCT keys — required by `test_cross_provider_fallback_uses_the_correct_key`
    to tell the two providers' baked-in credentials apart."""
    return Settings(
        _env_file=None,
        openai_api_key=SecretStr("sk-openai-test"),
        anthropic_api_key=SecretStr("sk-anthropic-test"),
    )  # type: ignore[call-arg]


# --- the two safety-critical fallback settings ------------------------------


def test_content_policy_fallbacks_stay_unset() -> None:
    """If set, a content-policy refusal is silently retried on another model
    and `ContentFilteredError` never surfaces. Asserted on the constructed
    instance — `router.py:603` reads litellm's own global when the argument
    is empty, so "we did not pass it" would prove nothing."""
    router = build_router(make_catalog(), _credentialed())
    assert not router.content_policy_fallbacks


def test_context_window_fallbacks_stay_unset() -> None:
    """Same failure class as content-policy, not named by the proposal but
    explicitly design-mandated (design-2 §5.3): a silent context-overflow
    retry means the audit trail never records why a bigger model answered.
    `router.py:601`, same reasoning."""
    router = build_router(make_catalog(), _credentialed())
    assert not router.context_window_fallbacks


def test_no_wildcard_fallback_entry() -> None:
    """`router.py:594-599` appends `{"*": litellm.default_fallbacks}` when
    either the constructor arg or `litellm.default_fallbacks` is not `None`.
    A wildcard would route every model to a hand-configured list and defeat
    upward-only entirely. This assertion exists only because that branch was
    read, not because a scenario named it."""
    router = build_router(make_catalog(), _credentialed())
    assert all("*" not in entry for entry in router.fallbacks or [])


def test_retry_and_cooldown_configuration() -> None:
    router = build_router(make_catalog(), _credentialed())
    assert router.num_retries == ROUTER_NUM_RETRIES
    assert router.retry_after == ROUTER_RETRY_AFTER_SECONDS
    assert router.allowed_fails == ROUTER_ALLOWED_FAILS
    assert router.cooldown_time == ROUTER_COOLDOWN_SECONDS
    assert router.routing_strategy == ROUTING_STRATEGY


# --- credentials survive real construction ----------------------------------


def test_cross_provider_fallback_uses_the_correct_key() -> None:
    """`haiku-3-5` (anthropic) falls back to `gpt-4o` (openai) — the upward
    chain routinely crosses providers. `router.py:8230`'s per-deployment
    credential precedence means the fallback target's baked-in key must be
    its OWN provider's, not the source's."""
    router = build_router(make_catalog(), _credentialed())
    by_name = {entry["model_name"]: entry for entry in router.model_list}

    source = by_name["haiku-3-5"]
    target = by_name["gpt-4o"]

    assert source["litellm_params"]["api_key"] == "sk-anthropic-test"
    assert target["litellm_params"]["api_key"] == "sk-openai-test"
    assert target["litellm_params"]["api_key"] != source["litellm_params"]["api_key"]


# --- F3: model_info["id"] round-trip, checked for real ----------------------


def test_model_info_id_round_trips_through_a_real_router() -> None:
    """`router.py:8038-8041` (`set_model_list`): `if "id" not in _model_info:
    _model_info["id"] = self._generate_model_id(...)`. Ours is never missing,
    so it must never be replaced — verified by reading source until now; this
    is the first real Router this codebase has ever constructed, so it is the
    first place the assumption can be checked for real rather than read."""
    catalog = make_catalog()
    router = build_router(catalog, _credentialed())

    ids = {entry["model_info"]["id"] for entry in router.model_list}
    assert ids == {m.key for m in catalog.enabled}


def test_round_trip_guard_raises_if_litellm_ever_stops_preserving_the_id() -> None:
    """The positive test above can only prove litellm behaves *today*. A real
    `ModelCatalog` + real `litellm.Router` cannot be coerced into violating
    the assumption — there is nothing to corrupt on our side — so proving the
    guard itself fires requires a stand-in object shaped like a Router.

    This is the loud-failure half of F3: if a future litellm version ever
    regenerates a deployment id instead of honouring the caller-supplied one,
    `_answering_model` (slice 7) would silently attribute a response — and
    its price — to the wrong catalog model. Raising here, at build time, is
    strictly better than a wrong number that looks right.
    """

    class _RouterStub:
        model_list: ClassVar[list[dict[str, object]]] = [
            {"model_info": {"id": "not-the-expected-key"}}
        ]

    with pytest.raises(ConfigurationError, match="did not preserve"):
        _verify_model_ids_round_tripped(_RouterStub(), expected_ids={"the-real-key"})
