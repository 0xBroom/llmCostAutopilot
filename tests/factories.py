"""Builders for domain objects.

The domain types validate hard on construction, which is correct in production
and tedious in tests. These builders supply valid defaults so a test only has
to state the one field it actually cares about.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from autopilot.config.settings import Settings
from autopilot.domain.catalog import ModelCatalog
from autopilot.domain.models import (
    CompletionRequest,
    ComplexityTier,
    CostBreakdown,
    DecisionReason,
    LLMResponse,
    Message,
    ModelConfig,
    PriceSource,
    RequestRecord,
    RoutingDecision,
    SamplingStratum,
    TokenUsage,
    VerificationJob,
)

# A three-tier catalog with realistic price shapes: free local, cheap hosted,
# expensive frontier. Prices are illustrative — the real catalog is loaded from
# YAML and overrides the gateway's price map.
LOCAL = ModelConfig(
    key="llama3-local",
    provider="ollama",
    provider_model_id="ollama_chat/llama3.1",
    input_cost_per_token=Decimal("0"),
    output_cost_per_token=Decimal("0"),
    max_context_tokens=8192,
    api_base="http://localhost:11434",
    quality_tier=ComplexityTier.SIMPLE,
    price_source=PriceSource.PRICE_MAP,
)

CHEAP = ModelConfig(
    key="haiku-3-5",
    provider="anthropic",
    provider_model_id="anthropic/claude-3-5-haiku-20241022",
    input_cost_per_token=Decimal("0.0000008"),
    output_cost_per_token=Decimal("0.000004"),
    max_context_tokens=200_000,
    quality_tier=ComplexityTier.MODERATE,
    price_source=PriceSource.PRICE_MAP,
)

EXPENSIVE = ModelConfig(
    key="gpt-4o",
    provider="openai",
    provider_model_id="openai/gpt-4o",
    input_cost_per_token=Decimal("0.0000025"),
    output_cost_per_token=Decimal("0.00001"),
    max_context_tokens=128_000,
    quality_tier=ComplexityTier.COMPLEX,
    price_source=PriceSource.PRICE_MAP,
    baseline=True,
    judge=True,
)

CATALOG = {m.key: m for m in (LOCAL, CHEAP, EXPENSIVE)}


def make_catalog(*, models: tuple[ModelConfig, ...] | None = None) -> ModelCatalog:
    """A three-tier catalog (LOCAL/CHEAP/EXPENSIVE) ready for fallback-
    derivation and invariant tests. Override `models` to test a specific
    catalog shape — every later slice's tests use this."""
    return ModelCatalog(models=models if models is not None else (LOCAL, CHEAP, EXPENSIVE))


def make_settings(**overrides: object) -> Settings:
    """`Settings` for tests, with the `.env` door closed too.

    `_no_ambient_credentials` (conftest, autouse) strips `os.environ`. Only
    `_env_file=None` also stops pydantic-settings reading the developer's
    `.env`, which it does as a file and therefore independently of the
    environment. See `test_make_settings_ignores_the_dotenv_file`. No test in
    this suite may call `load_settings()` directly — that is the one
    function allowed to see a real `.env`.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


def make_request(
    *,
    prompt: str = "What is 2 + 2?",
    request_id: UUID | None = None,
    **kwargs: object,
) -> CompletionRequest:
    return CompletionRequest(
        request_id=request_id or uuid4(),
        messages=(Message(role="user", content=prompt),),
        **kwargs,  # type: ignore[arg-type]
    )


def make_decision(
    *,
    request_id: UUID | None = None,
    tier: ComplexityTier = ComplexityTier.SIMPLE,
    confidence: float = 0.92,
    reason: DecisionReason = DecisionReason.POLICY_MATCH,
    chosen: ModelConfig = LOCAL,
    baseline: ModelConfig = EXPENSIVE,
    at: datetime | None = None,
    **kwargs: object,
) -> RoutingDecision:
    return RoutingDecision(
        request_id=request_id or uuid4(),
        decided_at=at or datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        tier=tier,
        confidence=confidence,
        reason=reason,
        chosen_model=chosen,
        baseline_model=baseline,
        policy_version="test-policy-1",
        classifier_version="test-clf-1",
        **kwargs,  # type: ignore[arg-type]
    )


def make_response(
    *,
    model: ModelConfig = LOCAL,
    usage: TokenUsage | None = None,
    content: str = "4",
    finish_reason: str = "stop",
    **kwargs: object,
) -> LLMResponse:
    u = usage or TokenUsage(prompt_tokens=100, completion_tokens=50)
    return LLMResponse(
        content=content,
        model_key=model.key,
        provider_model_id=model.provider_model_id,
        usage=u,
        cost=CostBreakdown.compute(u, model),
        latency_ms=42,
        finish_reason=finish_reason,
        **kwargs,  # type: ignore[arg-type]
    )


def make_record(
    *,
    chosen: ModelConfig = LOCAL,
    baseline: ModelConfig = EXPENSIVE,
    usage: TokenUsage | None = None,
    at: datetime | None = None,
) -> RequestRecord:
    """A fully served request, with its counterfactual baseline frozen in."""
    u = usage or TokenUsage(prompt_tokens=100, completion_tokens=50)
    when = at or datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    request_id = uuid4()
    return RequestRecord(
        request_id=request_id,
        received_at=when,
        decision=make_decision(request_id=request_id, chosen=chosen, baseline=baseline, at=when),
        response=make_response(model=chosen, usage=u),
        baseline_cost=CostBreakdown.compute(u, baseline),
    )


def make_job(
    *,
    request_id: UUID | None = None,
    stratum: SamplingStratum = SamplingStratum.RANDOM,
    at: datetime | None = None,
) -> VerificationJob:
    return VerificationJob(
        job_id=uuid4(),
        request_id=request_id or uuid4(),
        stratum=stratum,
        enqueued_at=at or datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
