"""Value objects. Pure data and pure arithmetic — no I/O, no vendor SDKs.

Two rules govern this module and are load-bearing for the rest of the system:

1. **Money is `Decimal`, never `float`.** Per-token prices are around 1e-7 USD.
   Summing millions of those in binary floating point loses money in a way that
   is invisible until the savings report disagrees with the provider invoice.
2. **Everything is frozen.** A `RoutingDecision` that can be mutated after the
   fact is not an audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import Final, Literal
from uuid import UUID

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    """One chat message.

    V1 is text-only. Multimodal content blocks are a provider-shaped concern
    and would belong in the adapter, not here.
    """

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """What enters the system, normalised away from any provider's wire format."""

    request_id: UUID
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    # A client may pin a model. The policy records that it was overridden
    # rather than silently ignoring it — see DecisionReason.CLIENT_PINNED_MODEL.
    requested_model: str | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a completion request needs at least one message")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when set")


class ComplexityTier(IntEnum):
    """How much model a request deserves.

    `IntEnum` on purpose: escalation is an ordering question ("route up"),
    and comparison operators should just work.
    """

    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3


class DecisionReason(StrEnum):
    """Machine-readable *why*. Free-text notes go alongside, never instead.

    Analytics needs to answer "how much did low-confidence escalation cost us
    this month?" with a GROUP BY, not with a LIKE over a prose column.
    """

    POLICY_MATCH = "policy_match"
    LOW_CONFIDENCE_ESCALATION = "low_confidence_escalation"
    CONTEXT_OVERFLOW_ESCALATION = "context_overflow_escalation"
    VERIFICATION_ESCALATION = "verification_escalation"
    CLIENT_PINNED_MODEL = "client_pinned_model"
    BUDGET_CAP_DOWNGRADE = "budget_cap_downgrade"
    CACHE_HIT = "cache_hit"
    FALLBACK_AFTER_FAILURE = "fallback_after_failure"


KNOWN_PROVIDERS: Final[frozenset[str]] = frozenset({"anthropic", "openai", "ollama"})


class PriceSource(StrEnum):
    """Where this model's unit prices came from. Written onto every row's
    provenance so a price can be traced without re-running the loader.

    Values are vendor-neutral on purpose: they get persisted onto rows via
    provenance, and swapping the provider layer later must not require a
    data migration of an enum value.
    """

    PRICE_MAP = "price_map"  # the gateway's bundled map
    CATALOG_OVERRIDE = "catalog_override"  # an explicit `price:` block in models.yaml
    REGISTERED = "registered"  # we taught the gateway this model's price


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One entry in the model catalog.

    `provider_model_id` is deliberately *not* named after LiteLLM. The catalog
    describes models; translating an entry into whatever string a particular
    gateway wants is the adapter's job. That naming is what keeps the
    "swap the provider layer without touching business logic" claim honest.
    """

    key: str  # stable catalog key, e.g. "haiku-3-5". Also the Router group name.
    provider: str  # "anthropic" | "openai" | "ollama"
    provider_model_id: str  # e.g. "anthropic/claude-3-5-haiku-20241022"
    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    max_context_tokens: int
    quality_tier: ComplexityTier
    price_source: PriceSource
    max_output_tokens: int | None = None
    api_base: str | None = None
    enabled: bool = True
    supports_json_mode: bool = False
    # Declared for #9 (provider baseline harness), read by nothing in this change.
    expected_latency_ms: int | None = None
    baseline: bool = False
    judge: bool = False
    judge_fallback: str | None = None

    def __post_init__(self) -> None:
        if self.input_cost_per_token < 0 or self.output_cost_per_token < 0:
            raise ValueError(f"negative unit price on model {self.key!r}")
        if self.max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be positive on model {self.key!r}")
        if self.provider not in KNOWN_PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r} on model {self.key!r}")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive on model {self.key!r}")
        if self.judge_fallback is not None and self.judge_fallback == self.key:
            raise ValueError(f"model {self.key!r} cannot be its own judge fallback")

    @property
    def is_free(self) -> bool:
        """True for locally hosted models. Free at the point of use, not free."""
        return self.input_cost_per_token == 0 and self.output_cost_per_token == 0

    @property
    def fallback_sort_key(self) -> tuple[int, Decimal, str]:
        """Ordering key for fallback derivation. NOT a price — it is a rank.

        `key` is the final component so the order is total and tests are
        stable even when two models have identical tier and blended cost.
        """
        return (
            int(self.quality_tier),
            self.input_cost_per_token + self.output_cost_per_token,
            self.key,
        )


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts cannot be negative")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """USD cost *together with the unit prices that produced it*.

    Carrying the prices is not redundancy. Provider pricing changes, and the
    price map ships inside a pinned dependency. Storing only the total means a
    row from March cannot be re-derived or audited in June. This type is what
    makes "prices are frozen per request" a fact rather than an intention.

    No quantisation happens here. A single request can cost 2.4e-5 USD;
    rounding to cents at this layer destroys the number. Round for display.
    """

    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    prompt_cost: Decimal
    completion_cost: Decimal

    @property
    def total(self) -> Decimal:
        return self.prompt_cost + self.completion_cost

    @classmethod
    def compute(cls, usage: TokenUsage, model: ModelConfig) -> CostBreakdown:
        return cls(
            input_cost_per_token=model.input_cost_per_token,
            output_cost_per_token=model.output_cost_per_token,
            prompt_cost=model.input_cost_per_token * usage.prompt_tokens,
            completion_cost=model.output_cost_per_token * usage.completion_tokens,
        )

    @classmethod
    def zero(cls) -> CostBreakdown:
        """A cache hit costs nothing but still occupies a row, with a baseline."""
        z = Decimal(0)
        return cls(z, z, z, z)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A normalised provider response.

    `model_key` is the model that *actually answered*, which is not always the
    model the policy chose: the gateway's fallback chain can move the call
    after a failure. Recording the intent and the outcome separately is the
    only way the cost accounting can be trusted.
    """

    content: str
    model_key: str
    provider_model_id: str
    usage: TokenUsage
    cost: CostBreakdown
    latency_ms: int
    finish_reason: str
    provider_response_id: str | None = None

    @property
    def was_truncated(self) -> bool:
        """A truncated answer is a quality event, not a success."""
        return self.finish_reason in {"length", "max_tokens"}


@dataclass(frozen=True, slots=True)
class PromptFeatures:
    """A deterministic, ordered, named numeric view of a prompt.

    The concrete feature set belongs to the extractor, not to the domain. What
    the domain fixes is the contract: names and values are parallel, ordered,
    and stable — because a classifier trained on one column order and served
    with another fails silently rather than loudly.
    """

    names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(
                f"feature vector mismatch: {len(self.names)} names, {len(self.values)} values"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate feature names")

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """A calibrated prediction.

    `confidence` must come from a *calibrated* model. An uncalibrated
    `predict_proba` is a ranking, not a probability, and the whole
    low-confidence-escalates-upward rule is built on treating it as one.
    """

    tier: ComplexityTier
    confidence: float
    probabilities: dict[ComplexityTier, float]
    classifier_version: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        total = sum(self.probabilities.values())
        if self.probabilities and abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1.0, got {total}")
        predicted = self.probabilities.get(self.tier)
        if predicted is not None and abs(predicted - self.confidence) > 1e-9:
            raise ValueError("confidence must be the probability of the predicted tier")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Why this request went to this model. The audit record.

    Every field here exists to answer a question after the fact: which tier,
    how sure, under which policy, priced how, and what it would have cost on
    the baseline. `baseline_model` is captured *at decision time* so that
    changing the baseline later cannot rewrite history.
    """

    request_id: UUID
    decided_at: datetime
    tier: ComplexityTier
    confidence: float
    reason: DecisionReason
    chosen_model: ModelConfig
    baseline_model: ModelConfig
    policy_version: str
    classifier_version: str
    notes: str = ""
    escalated_from: ComplexityTier | None = None

    def __post_init__(self) -> None:
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        if self.escalated_from is not None and self.escalated_from >= self.tier:
            raise ValueError("escalation must move the tier upward")


class SamplingStratum(StrEnum):
    """Why a request was picked for verification.

    This is not bookkeeping. Quality parity may only be estimated from the
    RANDOM stratum: the targeted strata are deliberately enriched with requests
    we already suspect, so pooling them biases parity downward and invalidates
    its confidence interval. Making the stratum a required field on the job is
    what turns that rule into something the type system helps enforce.
    """

    RANDOM = "random"  # the only stratum admissible for parity estimation
    LOW_CONFIDENCE = "low_confidence"  # targeted, enriched — diagnostic only
    TIER_BOUNDARY = "tier_boundary"  # targeted, enriched — diagnostic only
    MANUAL = "manual"  # human-requested — diagnostic only

    @property
    def is_representative(self) -> bool:
        return self is SamplingStratum.RANDOM


@dataclass(frozen=True, slots=True)
class VerificationJob:
    """A queued request to check whether a routing decision held up."""

    job_id: UUID
    request_id: UUID
    stratum: SamplingStratum
    enqueued_at: datetime
    attempts: int = 0

    def __post_init__(self) -> None:
        if self.enqueued_at.tzinfo is None:
            raise ValueError("enqueued_at must be timezone-aware")
        if self.attempts < 0:
            raise ValueError("attempts cannot be negative")


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """One served request, as persisted.

    Composed of the value objects above rather than flattened into columns:
    the storage schema is the adapter's problem, the invariants are not.

    `baseline_cost` is the counterfactual — the same token usage priced at the
    baseline model's rates, frozen here. It is `None` only when the call failed
    and there is no usage to price.
    """

    request_id: UUID
    received_at: datetime
    decision: RoutingDecision
    response: LLMResponse | None = None
    baseline_cost: CostBreakdown | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.received_at.tzinfo is None:
            raise ValueError("received_at must be timezone-aware")
        if self.response is None and self.error is None:
            raise ValueError("a record needs either a response or an error")
        if self.response is not None and self.baseline_cost is None:
            raise ValueError("a served response must carry its frozen baseline cost")
        if self.response is None and self.baseline_cost is not None:
            raise ValueError(
                "an error record cannot carry a baseline cost: there is no usage to price"
            )

    @property
    def savings(self) -> Decimal | None:
        """Gross savings for this request. Gross — verification and escalation
        overhead are system-level and are netted off in the savings report,
        not here."""
        if self.response is None or self.baseline_cost is None:
            return None
        return self.baseline_cost.total - self.response.cost.total
