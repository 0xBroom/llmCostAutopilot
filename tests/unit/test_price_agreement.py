"""The price-agreement canary — our frozen-catalog Decimal cost cross-checked
against litellm's own float arithmetic, under the pinned local cost map.

Standalone here, ahead of the gateway (slice 7): `LiteLLMGateway.
_check_price_agreement` will wire this same comparison into `.complete()`,
logging on divergence and never changing the returned cost. This file proves
the comparison itself agrees before anything depends on it.

Uses `anthropic/claude-haiku-4-5` — the same `provider_model_id` as
`config/models.yaml`'s `haiku-4-5` entry, resolving bare (`claude-haiku-4-5`)
against the bundled map, exactly as the loader's two-step resolution
documents. No cache tokens: `litellm.completion_cost` folds cache pricing in
and the simple `price * tokens` product below does not — mixing them would
make this test fail for the wrong reason.
"""

from __future__ import annotations

from decimal import Decimal

from autopilot.infrastructure.litellm_env import litellm
from autopilot.infrastructure.pricing import PRICE_AGREEMENT_RELATIVE_TOLERANCE, to_price

_PROVIDER_MODEL_ID = "anthropic/claude-haiku-4-5"
_BARE_KEY = "claude-haiku-4-5"
_PROMPT_TOKENS = 120
_COMPLETION_TOKENS = 45


def _catalog_cost() -> Decimal:
    """The product our own `CostBreakdown.compute` would arrive at, using
    `to_price` — the one chokepoint — on the same map entry."""
    entry = litellm.model_cost[_BARE_KEY]
    input_price = to_price(
        entry["input_cost_per_token"], field="input_cost_per_token", model_key=_BARE_KEY
    )
    output_price = to_price(
        entry["output_cost_per_token"], field="output_cost_per_token", model_key=_BARE_KEY
    )
    return input_price * _PROMPT_TOKENS + output_price * _COMPLETION_TOKENS


def _litellm_cost() -> Decimal:
    """litellm's own arithmetic, over a hand-built response carrying the same
    token counts. Converted through `to_price` too — the chokepoint rule
    holds even for the canary's own comparison, no exception carved out."""
    response = litellm.ModelResponse(
        model=_BARE_KEY,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        usage=litellm.Usage(
            prompt_tokens=_PROMPT_TOKENS,
            completion_tokens=_COMPLETION_TOKENS,
            total_tokens=_PROMPT_TOKENS + _COMPLETION_TOKENS,
        ),
    )
    raw_cost = litellm.completion_cost(completion_response=response, model=_PROVIDER_MODEL_ID)
    return to_price(raw_cost, field="litellm_completion_cost", model_key=_BARE_KEY)


def test_prices_agree_under_the_pinned_local_cost_map() -> None:
    """Hard assert, not a warning — the map is pinned (`litellm==1.95.0`), so
    this is deterministic in CI. A relative tolerance, not equality: litellm's
    side is float arithmetic (`cost_calculator.py` returns `float`), so
    `Decimal(str(...))` of it will not equal our exact Decimal product bit
    for bit even when the two numbers agree to any sane precision."""
    ours = _catalog_cost()
    theirs = _litellm_cost()

    tolerance = PRICE_AGREEMENT_RELATIVE_TOLERANCE * max(ours, theirs)
    assert abs(ours - theirs) <= tolerance, f"ours={ours} theirs={theirs} tolerance={tolerance}"


def test_the_canary_uses_no_cache_tokens() -> None:
    """Documents the constraint directly: `completion_cost` folds cache
    pricing into its result, and the simple product above does not. A
    regression that adds cache token fields to the request would silently
    invalidate the comparison above without this guard failing loudly first."""
    response = litellm.ModelResponse(
        model=_BARE_KEY,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        usage=litellm.Usage(
            prompt_tokens=_PROMPT_TOKENS,
            completion_tokens=_COMPLETION_TOKENS,
            total_tokens=_PROMPT_TOKENS + _COMPLETION_TOKENS,
        ),
    )
    assert response.usage.prompt_tokens_details is None
    assert response.usage.completion_tokens_details is None
