"""`LiteLLMGateway` — the adapter's own scenarios, over a hand-written
`FakeRouter`. Never a real `litellm.Router`, never the network.

Imports litellm through the shim only to construct real vendor exception
instances for the error-translation scenarios (same convention as
`test_router_factory.py`) — `FakeRouter`/`FakeModelResponse` themselves stay
vendor-free.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest
from structlog.testing import capture_logs

from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.errors import (
    AuthenticationFailedError,
    ContentFilteredError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from autopilot.domain.models import (
    ComplexityTier,
    CostBreakdown,
    LLMResponse,
    ModelConfig,
    PriceSource,
    RequestRecord,
)
from autopilot.infrastructure.litellm_env import litellm
from autopilot.infrastructure.litellm_gateway import LiteLLMGateway
from tests.factories import CHEAP, EXPENSIVE, LOCAL, make_catalog, make_decision, make_request
from tests.fakes.response import FakeChoice, FakeMessage, FakeModelResponse, FakeUsage
from tests.fakes.router import FakeRouter
from tests.fakes.store import InMemoryRequestStore

_PROMPT_TOKENS = 120
_COMPLETION_TOKENS = 45

# A model priced wildly differently from the real bundled map (gpt-4o's real
# input price is ~2.5e-6, not 1e-3) — used only by the divergence test, which
# needs the canary's litellm-side computation to actually succeed (unlike
# CHEAP's provider_model_id, which the installed map does not resolve at all,
# verified empirically; see litellm_gateway._check_price_agreement's
# best-effort except-and-skip for that case).
_MISPRICED_GPT4O = ModelConfig(
    key="gpt-4o",
    provider="openai",
    provider_model_id="openai/gpt-4o",
    input_cost_per_token=Decimal("0.001"),
    output_cost_per_token=Decimal("0.001"),
    max_context_tokens=128_000,
    quality_tier=ComplexityTier.COMPLEX,
    price_source=PriceSource.CATALOG_OVERRIDE,
    baseline=True,
    judge=True,
)


def _response(
    *,
    model_id: str,
    content: str | None = "the answer",
    finish_reason: str = "stop",
    prompt_tokens: int = _PROMPT_TOKENS,
    completion_tokens: int = _COMPLETION_TOKENS,
    hidden_params: dict[str, object] | None = None,
    response_id: str = "fake-response-id",
) -> FakeModelResponse:
    return FakeModelResponse(
        model=model_id,
        choices=[FakeChoice(message=FakeMessage(content=content), finish_reason=finish_reason)],
        usage=FakeUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        id=response_id,
        _hidden_params=hidden_params or {},
    )


# --- cost, frozen from the catalog --------------------------------------------


async def test_cost_computed_from_frozen_catalog_decimal_prices() -> None:
    router = FakeRouter(responses={CHEAP.key: _response(model_id=CHEAP.provider_model_id)})
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    response = await gateway.complete(make_request(), CHEAP, timeout_s=30)

    assert response.cost.prompt_cost == CHEAP.input_cost_per_token * _PROMPT_TOKENS
    assert response.cost.completion_cost == CHEAP.output_cost_per_token * _COMPLETION_TOKENS


# --- who actually answered -----------------------------------------------------


async def test_model_key_reflects_the_actual_responder_after_fallback() -> None:
    """The request targets `llama3-local`, but litellm's fallback chain moved
    the call to `haiku-3-5` — F3's round-trip, recovered via
    `_hidden_params["model_id"]` (route 1), even though the raw response's
    own `model` field still names the originally requested provider_model_id."""
    router = FakeRouter(
        responses={LOCAL.key: _response(model_id=LOCAL.provider_model_id)},
        answered_by={LOCAL.key: CHEAP.key},
    )
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    response = await gateway.complete(make_request(), LOCAL, timeout_s=30)

    assert response.model_key == CHEAP.key
    assert response.provider_model_id == CHEAP.provider_model_id


async def test_unattributable_response_raises_instead_of_guessing() -> None:
    """Neither `_hidden_params["model_id"]` nor `model` resolves to a catalog
    entry. Raising here is the point: attributing the answer to the requested
    model anyway would freeze a wrong price onto the row, permanently."""
    router = FakeRouter(responses={CHEAP.key: _response(model_id="totally-unknown-model-id")})
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    with pytest.raises(ProviderResponseError, match="cannot attribute"):
        await gateway.complete(make_request(), CHEAP, timeout_s=30)


# --- content normalisation, empty-string-safe -----------------------------------


async def test_length_truncated_completion_returns_partial_text() -> None:
    router = FakeRouter(
        responses={
            CHEAP.key: _response(
                model_id=CHEAP.provider_model_id,
                content="The answer is 4",
                finish_reason="length",
            )
        }
    )
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    response = await gateway.complete(make_request(), CHEAP, timeout_s=30)

    assert response.content == "The answer is 4"
    assert response.finish_reason == "length"


async def test_null_content_coerces_to_empty_string() -> None:
    router = FakeRouter(
        responses={
            CHEAP.key: _response(
                model_id=CHEAP.provider_model_id, content=None, finish_reason="tool_calls"
            )
        }
    )
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    response = await gateway.complete(make_request(), CHEAP, timeout_s=30)

    assert response.content == ""


# --- error translation, wired for the first time --------------------------------


@pytest.mark.parametrize(
    "vendor_exc_name",
    ["ServiceUnavailableError", "BadGatewayError", "InternalServerError", "APIConnectionError"],
)
async def test_many_to_one_mapping_for_provider_unavailability(vendor_exc_name: str) -> None:
    vendor_exc_cls = getattr(litellm.exceptions, vendor_exc_name)
    exc = vendor_exc_cls(message="boom", llm_provider=CHEAP.provider, model=CHEAP.provider_model_id)
    router = FakeRouter(errors={CHEAP.key: exc})
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    with pytest.raises(ProviderUnavailableError):
        await gateway.complete(make_request(), CHEAP, timeout_s=30)


async def test_authentication_failed_error_stays_distinct_from_missing_credentials_error() -> None:
    exc = litellm.exceptions.AuthenticationError(
        message="invalid api key", llm_provider=CHEAP.provider, model=CHEAP.provider_model_id
    )
    router = FakeRouter(errors={CHEAP.key: exc})
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    with pytest.raises(AuthenticationFailedError):
        await gateway.complete(make_request(), CHEAP, timeout_s=30)


async def test_content_filtered_exception_path() -> None:
    exc = litellm.exceptions.ContentPolicyViolationError(
        message="content flagged",
        model=EXPENSIVE.provider_model_id,
        llm_provider=EXPENSIVE.provider,
    )
    router = FakeRouter(errors={EXPENSIVE.key: exc})
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    with pytest.raises(ContentFilteredError):
        await gateway.complete(make_request(), EXPENSIVE, timeout_s=30)


async def test_content_filtered_success_path() -> None:
    """A nominally-200 response with a filtered `finish_reason` raises
    `ContentFilteredError` instead of returning a normal `LLMResponse` — the
    hand-built stand-in this scenario is unit-tested against; the real-payload
    half is explicitly deferred (see the contract-tier README)."""
    router = FakeRouter(
        responses={
            CHEAP.key: _response(
                model_id=CHEAP.provider_model_id, content="", finish_reason="content_filter"
            )
        }
    )
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    with pytest.raises(ContentFilteredError):
        await gateway.complete(make_request(), CHEAP, timeout_s=30)


# --- the price-agreement canary, wired into complete() --------------------------


async def test_price_agreement_divergence_logs_never_raises() -> None:
    catalog = ModelCatalog(models=(_MISPRICED_GPT4O,))
    router = FakeRouter(
        responses={_MISPRICED_GPT4O.key: _response(model_id=_MISPRICED_GPT4O.provider_model_id)}
    )
    gateway = LiteLLMGateway(router=router, catalog=catalog)

    with capture_logs() as logs:
        response = await gateway.complete(make_request(), _MISPRICED_GPT4O, timeout_s=30)

    expected_cost = (
        _MISPRICED_GPT4O.input_cost_per_token * _PROMPT_TOKENS
        + _MISPRICED_GPT4O.output_cost_per_token * _COMPLETION_TOKENS
    )
    assert response.cost.total == expected_cost  # never litellm's number

    warnings = [entry for entry in logs if entry["log_level"] == "warning"]
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning["model_key"] == _MISPRICED_GPT4O.key
    assert warning["catalog_cost"] == str(expected_cost)
    assert warning["litellm_cost"] != warning["catalog_cost"]


# --- response_format passthrough (quality-judge JSON mode wiring) ---------------


async def test_response_format_forwarded_to_router_when_set() -> None:
    router = FakeRouter(responses={CHEAP.key: _response(model_id=CHEAP.provider_model_id)})
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    await gateway.complete(
        make_request(), CHEAP, timeout_s=30, response_format={"type": "json_object"}
    )

    assert router.calls[-1].kwargs["response_format"] == {"type": "json_object"}


async def test_response_format_omitted_when_none_preserves_current_behavior() -> None:
    router = FakeRouter(responses={CHEAP.key: _response(model_id=CHEAP.provider_model_id)})
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    await gateway.complete(make_request(), CHEAP, timeout_s=30)

    assert "response_format" not in router.calls[-1].kwargs


# --- the raw payload never becomes a field, by construction ---------------------


def test_llm_response_has_no_raw_field() -> None:
    """Slice 2's handoff, reconfirmed here: Phase 0's `LLMResponse` has no
    `raw` field and does not get one in this slice. The repr/persistence
    scenarios below are satisfied by construction, not by a repr filter."""
    assert "raw" not in {f.name for f in dataclasses.fields(LLMResponse)}


async def test_repr_excludes_the_raw_provider_payload() -> None:
    sentinel = "RAW-PAYLOAD-MARKER-must-never-reach-llmresponse"
    router = FakeRouter(
        responses={
            CHEAP.key: _response(
                model_id=CHEAP.provider_model_id, hidden_params={"raw_marker": sentinel}
            )
        }
    )
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())

    response = await gateway.complete(make_request(), CHEAP, timeout_s=30)

    assert sentinel not in repr(response)


async def test_persisted_record_never_carries_the_raw_provider_object() -> None:
    store = InMemoryRequestStore()
    router = FakeRouter(responses={CHEAP.key: _response(model_id=CHEAP.provider_model_id)})
    gateway = LiteLLMGateway(router=router, catalog=make_catalog())
    response = await gateway.complete(make_request(), CHEAP, timeout_s=30)

    decision = make_decision(chosen=CHEAP, baseline=EXPENSIVE)
    full_record = RequestRecord(
        request_id=decision.request_id,
        received_at=decision.decided_at,
        decision=decision,
        response=response,
        baseline_cost=CostBreakdown.compute(response.usage, EXPENSIVE),
    )
    await store.save(full_record)

    fetched = await store.get(full_record.request_id)
    assert fetched is not None
    assert fetched.response is not None
    assert "raw" not in {f.name for f in dataclasses.fields(fetched.response)}
