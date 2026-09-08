"""A gateway that never touches the network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from autopilot.domain.errors import AllProvidersFailedError
from autopilot.domain.models import (
    CompletionRequest,
    CostBreakdown,
    LLMResponse,
    ModelConfig,
    TokenUsage,
)


@dataclass(frozen=True, slots=True)
class GatewayCall:
    """One recorded invocation, so a test can assert *which model was asked*."""

    request: CompletionRequest
    model: ModelConfig
    timeout_s: float
    response_format: Mapping[str, Any] | None = None


@dataclass
class FakeLLMGateway:
    """Deterministic, scriptable, hand-written.

    Hand-written and not a `MagicMock` for one specific reason: a mock accepts
    any call signature. The day someone adds a parameter to `LLMGateway.complete`,
    a mock keeps every test green while production breaks. This class is checked
    against the Protocol by mypy, so the same change fails the type check.

    Configure by catalog key:
      replies      key -> answer text
      errors       key -> exception to raise instead of answering
      fallback_to  key -> the key that *actually* answers, simulating the
                   gateway's fallback chain. The response then reports a
                   `model_key` different from the one requested, which is a
                   real behaviour the cost accounting has to survive.
    """

    replies: Mapping[str, str] = field(default_factory=dict)
    errors: Mapping[str, Exception] = field(default_factory=dict)
    fallback_to: Mapping[str, ModelConfig] = field(default_factory=dict)
    default_reply: str = "fake answer"
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(100, 50))
    latency_ms: int = 5
    finish_reason: str = "stop"
    calls: list[GatewayCall] = field(default_factory=list)

    async def complete(
        self,
        request: CompletionRequest,
        model: ModelConfig,
        *,
        timeout_s: float,
        response_format: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            GatewayCall(
                request=request, model=model, timeout_s=timeout_s, response_format=response_format
            )
        )

        if (err := self.errors.get(model.key)) is not None:
            raise err

        answering = self.fallback_to.get(model.key, model)
        if answering.key in self.errors:
            raise AllProvidersFailedError(
                f"fake chain exhausted from {model.key!r}", model_key=model.key
            )

        return LLMResponse(
            content=self.replies.get(answering.key, self.default_reply),
            model_key=answering.key,
            provider_model_id=answering.provider_model_id,
            usage=self.usage,
            cost=CostBreakdown.compute(self.usage, answering),
            latency_ms=self.latency_ms,
            finish_reason=self.finish_reason,
            provider_response_id=f"fake-{len(self.calls)}",
        )

    # --- assertions helpers ---------------------------------------------------

    @property
    def called_model_keys(self) -> list[str]:
        return [c.model.key for c in self.calls]

    @property
    def call_count(self) -> int:
        return len(self.calls)
