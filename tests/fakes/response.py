"""Hand-written stand-ins for litellm's `ModelResponse` shape.

Not `unittest.mock`, and not a drop-in replacement for a real
`litellm.ModelResponse` either: `litellm.completion_cost()`'s own duck-typing
only recognises a `pydantic.BaseModel` or a `dict` as a `completion_response`
(see `litellm/cost_calculator.py`), so a hand-written dataclass shaped like
this silently prices at zero if handed to it — verified empirically, not
assumed. That is exactly why `LiteLLMGateway._check_price_agreement` never
calls `completion_cost()` on the raw response at all; it calls
`litellm.cost_per_token(model=..., prompt_tokens=..., completion_tokens=...)`
instead, which needs only two token counts and a model name, never the
response object's own shape.

`tests/unit/test_response_shape.py` is what keeps this fake honest without a
cassette: it checks that the real `litellm.ModelResponse` has every attribute
this fake claims to have, so silently widening the fake to make a gateway
test pass shows up as a diff there instead of silently drifting from the
vendor's real shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class FakeMessage:
    content: str | None
    role: str = "assistant"


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str
    index: int = 0


@dataclass
class FakeModelResponse:
    """Mirrors the subset of `litellm.ModelResponse` the gateway reads:
    `model`, `choices`, `usage`, `id`, `_hidden_params`."""

    model: str
    choices: list[FakeChoice]
    usage: FakeUsage
    id: str = "fake-response-id"
    _hidden_params: dict[str, Any] = field(default_factory=dict)
