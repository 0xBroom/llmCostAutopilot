"""A `litellm.Router`-shaped object that never touches the network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RouterCall:
    """One recorded `acompletion` invocation, so a test can assert exactly
    what the gateway asked the Router for."""

    model: str
    messages: list[dict[str, str]]
    kwargs: dict[str, Any]


@dataclass
class FakeRouter:
    """A Router-shaped object satisfying `litellm_gateway.RouterLike`.

    Hand-written, not a `MagicMock`, for the same reason `FakeLLMGateway` is:
    a mock accepts any call signature, so the day `acompletion`'s shape
    changes, a mock keeps every test green while production breaks. This
    class is checked against `RouterLike` by mypy in
    `tests/unit/test_adapter_seams_contract.py`.

    Configure by the *requested* catalog key — the `model` argument
    `acompletion` receives, which is `model.key` from `router_factory`'s
    one-deployment-group-per-catalog-key design:

      responses    key -> the object `acompletion` returns
      errors       key -> the exception `acompletion` raises instead
      answered_by  key -> the key whose id gets stamped onto the returned
                   response's `_hidden_params["model_id"]` before it is
                   handed back, simulating litellm's F3 preservation of a
                   caller-supplied deployment id after a real fallback moved
                   the call to a different deployment than the one requested
    """

    responses: Mapping[str, Any] = field(default_factory=dict)
    errors: Mapping[str, Exception] = field(default_factory=dict)
    answered_by: Mapping[str, str] = field(default_factory=dict)
    calls: list[RouterCall] = field(default_factory=list)

    async def acompletion(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self.calls.append(RouterCall(model=model, messages=messages, kwargs=kwargs))

        if (err := self.errors.get(model)) is not None:
            raise err

        response = self.responses[model]
        answering_key = self.answered_by.get(model)
        if answering_key is not None:
            hidden = dict(getattr(response, "_hidden_params", {}) or {})
            hidden["model_id"] = answering_key
            response._hidden_params = hidden
        return response
