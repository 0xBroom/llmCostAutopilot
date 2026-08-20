"""create requests table

Revision ID: 0001
Revises:
Create Date: 2026-08-20

Owns the ``requests`` table only. Sibling tables (``verifications``,
``escalations``, ``training_candidates``, ``config_audit``) arrive with their
owning issues, each in its own revision. Retention and the analytics indices are
Phase-4 work; the single index here is the one ``RequestStore.list_since``
needs.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "requests",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("received_at", sa.String(), nullable=False),
        sa.Column("decided_at", sa.String(), nullable=False),
        sa.Column("tier_effective", sa.Integer(), nullable=False),
        sa.Column("escalated_from", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("routing_reason", sa.String(), nullable=False),
        sa.Column("policy_version", sa.String(), nullable=False),
        sa.Column("classifier_version", sa.String(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("intended_model_key", sa.String(), nullable=False),
        sa.Column("baseline_model_key", sa.String(), nullable=False),
        sa.Column("chosen_model_json", sa.Text(), nullable=False),
        sa.Column("baseline_model_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("actual_model_key", sa.String(), nullable=True),
        sa.Column("provider_model_id", sa.String(), nullable=True),
        sa.Column("provider_response_id", sa.String(), nullable=True),
        sa.Column("response_content", sa.Text(), nullable=True),
        sa.Column("tokens_in", sa.Integer(), nullable=True),
        sa.Column("tokens_out", sa.Integer(), nullable=True),
        sa.Column("finish_reason", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("input_price_snapshot", sa.String(), nullable=True),
        sa.Column("output_price_snapshot", sa.String(), nullable=True),
        sa.Column("baseline_input_price", sa.String(), nullable=True),
        sa.Column("baseline_output_price", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_requests")),
    )
    op.create_index(op.f("ix_requests_received_at"), "requests", ["received_at"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_requests_received_at"), table_name="requests")
    op.drop_table("requests")
