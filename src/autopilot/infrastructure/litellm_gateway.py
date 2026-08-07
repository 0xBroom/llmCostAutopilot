"""`LiteLLMGateway` — implements `application.ports.LLMGateway` over a
`litellm.Router`, calling `router.acompletion()` and never
`litellm.acompletion()` directly: the Router is where the deployment
grouping, the credentials and the upward-only fallback chain (all built by
`router_factory.py`) actually live.

This module starts with just the seam: `RouterLike` is not a port — the
application layer has never heard of it. It is an adapter-local Protocol that
exists so `LiteLLMGateway` (added next) can be unit-tested against a
hand-written `FakeRouter` instead of a real `litellm.Router`. See
`tests/unit/test_adapter_seams_contract.py`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RouterLike(Protocol):
    """What the gateway needs from a Router."""

    async def acompletion(
        self, model: str, messages: list[dict[str, str]], **kwargs: Any
    ) -> Any: ...
