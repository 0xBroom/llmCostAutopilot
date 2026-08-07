"""`LiteLLMGateway` — implements `application.ports.LLMGateway` over a
`litellm.Router`, calling `router.acompletion()` and never
`litellm.acompletion()` directly: the Router is where the deployment
grouping, the credentials and the upward-only fallback chain (all built by
`router_factory.py`) actually live.

Three properties this module holds itself to, all load-bearing:

1. **Never sees a secret.** Credentials are baked into the Router's
   `model_list` at build time (`router_factory.py`); this class takes a
   `RouterLike` and a `ModelCatalog`, never a `Settings`, never a `SecretStr`.
   `tests/unit/test_credential_boundary.py` confirms `.get_secret_value()`
   has exactly one call site, and it is not here.
2. **The answering model is resolved, never assumed.** A fallback chain can
   move the call to a different deployment than the one requested;
   `_answering_model` recovers which one actually answered from
   `_hidden_params["model_id"]` (F3), falling back to an unambiguous
   `provider_model_id` suffix match, and raises rather than silently pricing
   the wrong model's rates onto the row.
3. **Cost is always computed from the catalog's frozen Decimal prices**, via
   `CostBreakdown.compute` — never from anything litellm reports.
   `_check_price_agreement` cross-checks the two and logs on divergence; it
   never feeds its own number back into the returned `LLMResponse`.

`raw` — whatever `router.acompletion()` returns — is a local variable for the
lifetime of `complete()` and is never assigned to any field. `LLMResponse` has
no `raw` attribute (see `domain/models.py`) and does not get one here: the
"excluded from repr, never persisted unmodified" requirement is satisfied by
construction, not by a repr filter.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import structlog
from structlog.stdlib import BoundLogger

from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.errors import ContentFilteredError, ProviderResponseError
from autopilot.domain.models import (
    CompletionRequest,
    CostBreakdown,
    LLMResponse,
    ModelConfig,
    TokenUsage,
)
from autopilot.infrastructure.error_translation import CONTENT_FILTER_FINISH_REASONS, translate
from autopilot.infrastructure.litellm_env import litellm
from autopilot.infrastructure.pricing import PRICE_AGREEMENT_RELATIVE_TOLERANCE, to_price


@runtime_checkable
class RouterLike(Protocol):
    """What the gateway needs from a Router. Not a port — the application
    layer has never heard of this; it is an adapter-local seam that exists so
    `LiteLLMGateway` can be unit-tested against a hand-written `FakeRouter`
    instead of a real `litellm.Router`. See
    `tests/unit/test_adapter_seams_contract.py`."""

    async def acompletion(
        self, model: str, messages: list[dict[str, str]], **kwargs: Any
    ) -> Any: ...


class LiteLLMGateway:
    """Implements `application.ports.LLMGateway`, structurally — never by
    inheritance."""

    def __init__(
        self,
        *,
        router: RouterLike,
        catalog: ModelCatalog,
        logger: BoundLogger | None = None,
    ) -> None:
        self._router = router
        self._catalog = catalog
        self._logger = logger if logger is not None else structlog.get_logger(__name__)

    async def complete(
        self,
        request: CompletionRequest,
        model: ModelConfig,
        *,
        timeout_s: float,
    ) -> LLMResponse:
        messages = self._to_messages(request)
        started = time.perf_counter()
        try:
            raw = await self._router.acompletion(
                model=model.key, messages=messages, timeout=timeout_s
            )
        except Exception as exc:
            raise translate(exc, model_key=model.key) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        answering = self._answering_model(raw, model)
        response = self._normalise(raw, answering, latency_ms)
        self._check_price_agreement(raw, response.cost, answering)
        return response

    # --- internals -------------------------------------------------------

    @staticmethod
    def _to_messages(request: CompletionRequest) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in request.messages]

    def _answering_model(self, raw: Any, requested: ModelConfig) -> ModelConfig:
        """Resolve the model that actually answered, after any fallback.

        Two independent routes, then a hard failure. Never a silent guess:
        attributing a gpt-4o answer to llama3-local's price is a wrong number
        that looks right, which is the one failure mode this system cannot
        have.
        """
        hidden = getattr(raw, "_hidden_params", None) or {}
        by_id = hidden.get("model_id")
        if isinstance(by_id, str) and (found := self._catalog.find(by_id)) is not None:
            return found  # route 1 — F3

        by_model = getattr(raw, "model", None)
        if isinstance(by_model, str):
            matches = [m for m in self._catalog if m.provider_model_id.endswith(by_model)]
            if len(matches) == 1:
                return matches[0]  # route 2 — unambiguous only

        raise ProviderResponseError(
            "cannot attribute the answer to a catalog model "
            f"(requested {requested.key!r}, model_id={by_id!r}, model={by_model!r})",
            model_key=requested.key,
        )

    def _normalise(self, raw: Any, answering: ModelConfig, latency_ms: int) -> LLMResponse:
        choice = raw.choices[0]
        finish_reason = choice.finish_reason
        if finish_reason in CONTENT_FILTER_FINISH_REASONS:
            raise ContentFilteredError(
                f"provider filtered the response (finish_reason={finish_reason!r})",
                model_key=answering.key,
            )

        content = choice.message.content
        usage = TokenUsage(
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
        )
        return LLMResponse(
            content=content if content is not None else "",
            model_key=answering.key,
            provider_model_id=answering.provider_model_id,
            usage=usage,
            cost=CostBreakdown.compute(usage, answering),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            provider_response_id=getattr(raw, "id", None),
        )

    def _check_price_agreement(self, raw: Any, cost: CostBreakdown, answering: ModelConfig) -> None:
        """Cross-check the catalog's frozen Decimal cost against litellm's
        own float arithmetic. Never produces money: `LLMResponse.cost` is
        always `CostBreakdown.compute` from the catalog — this number is
        compared and discarded; log-and-continue on divergence, never raise.

        Computed from token counts alone via `litellm.cost_per_token`, not
        from `raw` itself: `litellm.completion_cost()`'s own duck-typing only
        recognises a `pydantic.BaseModel` or a `dict` as a completion
        response (verified empirically against a hand-written fake, which it
        silently prices at zero), so calling it on `raw` here would either
        require `raw` to be a real vendor object or silently produce a
        meaningless comparison against our own fakes in tests.
        """
        usage = getattr(raw, "usage", None)
        if usage is None:
            return
        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=answering.provider_model_id,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        except Exception:
            return  # best-effort canary; a model litellm cannot price is not a gateway failure

        theirs = to_price(
            input_cost, field="litellm_input_cost", model_key=answering.key
        ) + to_price(output_cost, field="litellm_output_cost", model_key=answering.key)
        ours = cost.total
        if ours == 0 and theirs == 0:
            return

        tolerance = PRICE_AGREEMENT_RELATIVE_TOLERANCE * max(ours, theirs)
        if abs(ours - theirs) > tolerance:
            self._logger.warning(
                "litellm_gateway.price_agreement_diverges",
                model_key=answering.key,
                catalog_cost=str(ours),
                litellm_cost=str(theirs),
            )
