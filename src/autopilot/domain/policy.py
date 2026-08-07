"""Tier → model resolution rules.

Intentionally empty in Phase 0. The routing policy engine — YAML schema,
versioning, hot reload, budget-cap interaction and the explainability
contract — is a milestone of its own, and stubbing its shape here would
create a second source of truth for it to drift against.

What Phase 0 fixes is the *output* type the engine must produce:
`RoutingDecision` in `autopilot.domain.models`. Anything the engine emits has
to satisfy that record's invariants, including that an escalation moves the
tier strictly upward and that the baseline model is captured at decision time.
"""
