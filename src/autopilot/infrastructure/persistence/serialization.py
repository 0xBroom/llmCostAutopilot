"""Mapping between the domain's ``RequestRecord`` and a ``requests`` row.

Kept apart from the store so the two concerns stay separate: the store owns the
connection and the SQL, this module owns the shape of a row.

Money handling follows the codebase's single-chokepoint rule (see
``infrastructure/pricing.py`` and ``tests/unit/test_decimal_boundary.py``): the
only value that ever crosses back into ``Decimal`` from storage is a *per-token
price*, and it does so through ``to_price`` — never a bare ``Decimal(row[...])``.
Totals (``prompt_cost``/``completion_cost`` and their baseline counterparts) are
therefore not stored as columns at all; they are recomputed as
``frozen_price * frozen_token_count``, exactly as ``CostBreakdown.compute`` did
at write time. Both factors are frozen in the row, so the recomputed total
cannot drift with the live catalog — the freeze invariant holds without a
second Decimal-parsing site that would break the chokepoint.

Timestamps are normalised to UTC before serialisation. Two reasons, both load
bearing: ``received_at`` is compared and ordered as text in ``list_since``, so a
mixed-offset corpus would sort by wall-clock string rather than by instant; and
an aware ``datetime`` compares equal by instant, so normalising to UTC changes
the representation without breaking round-trip equality of the record.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from autopilot.domain.models import (
    ComplexityTier,
    CostBreakdown,
    DecisionReason,
    LLMResponse,
    ModelConfig,
    PriceSource,
    RequestRecord,
    RoutingDecision,
    TokenUsage,
)
from autopilot.infrastructure.pricing import to_price


def _utc_iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _model_to_json(model: ModelConfig) -> str:
    return json.dumps(
        {
            "key": model.key,
            "provider": model.provider,
            "provider_model_id": model.provider_model_id,
            "input_cost_per_token": str(model.input_cost_per_token),
            "output_cost_per_token": str(model.output_cost_per_token),
            "max_context_tokens": model.max_context_tokens,
            "quality_tier": int(model.quality_tier),
            "price_source": model.price_source.value,
            "max_output_tokens": model.max_output_tokens,
            "api_base": model.api_base,
            "enabled": model.enabled,
            "supports_json_mode": model.supports_json_mode,
            "expected_latency_ms": model.expected_latency_ms,
            "baseline": model.baseline,
            "judge": model.judge,
            "judge_fallback": model.judge_fallback,
        }
    )


def _model_from_json(text: str) -> ModelConfig:
    data = json.loads(text)
    key = data["key"]
    return ModelConfig(
        key=key,
        provider=data["provider"],
        provider_model_id=data["provider_model_id"],
        input_cost_per_token=to_price(
            data["input_cost_per_token"], field="input_cost_per_token", model_key=key
        ),
        output_cost_per_token=to_price(
            data["output_cost_per_token"], field="output_cost_per_token", model_key=key
        ),
        max_context_tokens=data["max_context_tokens"],
        quality_tier=ComplexityTier(data["quality_tier"]),
        price_source=PriceSource(data["price_source"]),
        max_output_tokens=data["max_output_tokens"],
        api_base=data["api_base"],
        enabled=data["enabled"],
        supports_json_mode=data["supports_json_mode"],
        expected_latency_ms=data["expected_latency_ms"],
        baseline=data["baseline"],
        judge=data["judge"],
        judge_fallback=data["judge_fallback"],
    )


def record_to_row(record: RequestRecord) -> dict[str, Any]:
    """Flatten a ``RequestRecord`` into a column mapping for insert/upsert."""
    decision = record.decision
    row: dict[str, Any] = {
        "id": str(record.request_id),
        "received_at": _utc_iso(record.received_at),
        "decided_at": _utc_iso(decision.decided_at),
        "tier_effective": int(decision.tier),
        "escalated_from": (
            int(decision.escalated_from) if decision.escalated_from is not None else None
        ),
        "confidence": decision.confidence,
        "routing_reason": decision.reason.value,
        "policy_version": decision.policy_version,
        "classifier_version": decision.classifier_version,
        "notes": decision.notes,
        "intended_model_key": decision.chosen_model.key,
        "baseline_model_key": decision.baseline_model.key,
        "chosen_model_json": _model_to_json(decision.chosen_model),
        "baseline_model_json": _model_to_json(decision.baseline_model),
        "status": "ok" if record.response is not None else "error",
        "error": record.error,
        # Execution columns are NULL unless a response is present.
        "actual_model_key": None,
        "provider_model_id": None,
        "provider_response_id": None,
        "response_content": None,
        "tokens_in": None,
        "tokens_out": None,
        "finish_reason": None,
        "latency_ms": None,
        # Frozen per-token prices; totals are recomputed from these on read.
        "input_price_snapshot": None,
        "output_price_snapshot": None,
        "baseline_input_price": None,
        "baseline_output_price": None,
    }

    if record.response is not None:
        response = record.response
        row.update(
            {
                "actual_model_key": response.model_key,
                "provider_model_id": response.provider_model_id,
                "provider_response_id": response.provider_response_id,
                "response_content": response.content,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
                "finish_reason": response.finish_reason,
                "latency_ms": response.latency_ms,
                "input_price_snapshot": str(response.cost.input_cost_per_token),
                "output_price_snapshot": str(response.cost.output_cost_per_token),
            }
        )

    if record.baseline_cost is not None:
        baseline = record.baseline_cost
        row.update(
            {
                "baseline_input_price": str(baseline.input_cost_per_token),
                "baseline_output_price": str(baseline.output_cost_per_token),
            }
        )

    return row


def row_to_record(row: dict[str, Any]) -> RequestRecord:
    """Rebuild a ``RequestRecord`` from a column mapping read back from storage."""
    escalated_from_raw = row["escalated_from"]
    decision = RoutingDecision(
        request_id=UUID(row["id"]),
        decided_at=datetime.fromisoformat(row["decided_at"]),
        tier=ComplexityTier(row["tier_effective"]),
        confidence=row["confidence"],
        reason=DecisionReason(row["routing_reason"]),
        chosen_model=_model_from_json(row["chosen_model_json"]),
        baseline_model=_model_from_json(row["baseline_model_json"]),
        policy_version=row["policy_version"],
        classifier_version=row["classifier_version"],
        notes=row["notes"],
        escalated_from=(
            ComplexityTier(escalated_from_raw) if escalated_from_raw is not None else None
        ),
    )

    response: LLMResponse | None = None
    if row["status"] == "ok":
        model_key = row["actual_model_key"]
        usage = TokenUsage(
            prompt_tokens=row["tokens_in"],
            completion_tokens=row["tokens_out"],
        )
        input_price = to_price(
            row["input_price_snapshot"], field="input_price_snapshot", model_key=model_key
        )
        output_price = to_price(
            row["output_price_snapshot"], field="output_price_snapshot", model_key=model_key
        )
        response = LLMResponse(
            content=row["response_content"],
            model_key=model_key,
            provider_model_id=row["provider_model_id"],
            usage=usage,
            cost=CostBreakdown(
                input_cost_per_token=input_price,
                output_cost_per_token=output_price,
                prompt_cost=input_price * usage.prompt_tokens,
                completion_cost=output_price * usage.completion_tokens,
            ),
            latency_ms=row["latency_ms"],
            finish_reason=row["finish_reason"],
            provider_response_id=row["provider_response_id"],
        )

    baseline_cost: CostBreakdown | None = None
    if row["baseline_input_price"] is not None:
        baseline_key = row["baseline_model_key"]
        prompt_tokens = int(row["tokens_in"])
        completion_tokens = int(row["tokens_out"])
        baseline_input = to_price(
            row["baseline_input_price"], field="baseline_input_price", model_key=baseline_key
        )
        baseline_output = to_price(
            row["baseline_output_price"], field="baseline_output_price", model_key=baseline_key
        )
        # The counterfactual prices the SAME token usage at the baseline model's
        # frozen rates — mirroring how it was computed at write time.
        baseline_cost = CostBreakdown(
            input_cost_per_token=baseline_input,
            output_cost_per_token=baseline_output,
            prompt_cost=baseline_input * prompt_tokens,
            completion_cost=baseline_output * completion_tokens,
        )

    return RequestRecord(
        request_id=UUID(row["id"]),
        received_at=datetime.fromisoformat(row["received_at"]),
        decision=decision,
        response=response,
        baseline_cost=baseline_cost,
        error=row["error"],
    )
