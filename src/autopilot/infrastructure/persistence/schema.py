"""SQLAlchemy Core table definitions for the audit trail.

Core, not the ORM. The domain already owns the types (`RequestRecord` and its
value objects); an ORM identity map would be a second, competing model of the
same data. This adapter maps rows to those frozen dataclasses by hand, which
keeps the storage schema a detail of this layer and nothing the core can see.

Money columns are ``TEXT`` holding ``Decimal`` strings. SQLite has no decimal
type and ``REAL`` drifts on aggregation; the rule (issue #20, and the module
docstring of ``domain/models.py``) is to convert at this boundary and aggregate
in Python — never ``SUM()`` money in SQL.

Scope note (#20 as re-sequenced into Phase 2): this migration owns the
``requests`` table only. ``verifications``, ``escalations``,
``training_candidates`` and ``config_audit`` land with their owning issues
(#17/#18/#19/#24), each with its own migration. Retention, the sweep and the
analytics indices are Phase-4 work; the single index here
(``received_at``) is the one the adapter's own ``list_since`` needs, not a
speculative one.
"""

from __future__ import annotations

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text

# A stable naming convention so Alembic autogenerate produces named constraints
# instead of backend-chosen names that churn across migrations.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

requests = Table(
    "requests",
    metadata,
    # --- identity -------------------------------------------------------------
    Column("id", String, primary_key=True),  # request_id, uuid string
    # Stored normalised to UTC so lexicographic string ordering equals
    # chronological ordering — see serialization.py.
    Column("received_at", String, nullable=False, index=True),
    # --- decision (RoutingDecision) -------------------------------------------
    Column("decided_at", String, nullable=False),
    Column("tier_effective", Integer, nullable=False),
    Column("escalated_from", Integer, nullable=True),  # predicted tier, if escalated
    Column("confidence", Float, nullable=False),
    Column("routing_reason", String, nullable=False),  # DecisionReason value
    Column("policy_version", String, nullable=False),
    Column("classifier_version", String, nullable=False),
    Column("notes", Text, nullable=False),
    Column("intended_model_key", String, nullable=False),
    Column("baseline_model_key", String, nullable=False),
    # The chosen and baseline models are frozen as JSON so a historical row is
    # self-contained: reading March's data in June must never depend on the
    # live catalog, which may have re-priced or dropped a model since.
    Column("chosen_model_json", Text, nullable=False),
    Column("baseline_model_json", Text, nullable=False),
    # --- outcome --------------------------------------------------------------
    Column("status", String, nullable=False),  # 'ok' | 'error'
    Column("error", Text, nullable=True),
    # --- execution (LLMResponse; NULL on an error record) ---------------------
    Column("actual_model_key", String, nullable=True),
    Column("provider_model_id", String, nullable=True),
    Column("provider_response_id", String, nullable=True),
    Column("response_content", Text, nullable=True),
    Column("tokens_in", Integer, nullable=True),
    Column("tokens_out", Integer, nullable=True),
    Column("finish_reason", String, nullable=True),
    Column("latency_ms", Integer, nullable=True),
    # --- money: per-token unit prices FROZEN at write time, Decimal-as-TEXT ----
    # Totals are NOT stored: they are recomputed as price * token_count on read
    # (serialization.py), which keeps every Decimal conversion inside the single
    # `to_price` chokepoint. Both factors are frozen here, so the recomputed
    # total cannot drift with the live catalog. NULL on an error record.
    Column("input_price_snapshot", String, nullable=True),
    Column("output_price_snapshot", String, nullable=True),
    # The baseline counterfactual's frozen per-token prices, applied to the same
    # token usage on read. NULL only on an error record, where there is nothing
    # to price.
    Column("baseline_input_price", String, nullable=True),
    Column("baseline_output_price", String, nullable=True),
)
