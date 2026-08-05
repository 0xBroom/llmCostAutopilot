# ADR-002: Hexagonal package layout with a mechanically enforced dependency rule

- **Status:** Accepted
- **Deciders:** project architect
- **Related:** [ADR-001](0001-litellm-as-provider-abstraction.md)

## Context

Four collaborators in this system must be swappable independently:

| Collaborator | Today | Plausible tomorrow |
| :-- | :-- | :-- |
| LLM provider | LiteLLM | anything |
| Classifier | calibrated sklearn | a fine-tuned encoder |
| Store | SQLite | Postgres |
| Queue | SQLite outbox | Redis, SQS |

If those four leak into each other, the claim this project is built to
demonstrate — *swap the provider layer without touching business logic* —
becomes false, and worse, becomes false quietly.

## Decision

Ports and adapters, with a strict one-way dependency rule.

```
src/autopilot/
├── domain/                 pure. no I/O, no litellm, no fastapi, no sqlite
│   ├── models.py           value objects: ModelConfig, ComplexityTier,
│   │                       RoutingDecision, LLMResponse, CostBreakdown, …
│   ├── policy.py           tier → model resolution rules
│   └── errors.py           the domain error taxonomy
├── application/            use cases. orchestrates ports, still no vendor imports
│   └── ports.py            Protocols: LLMGateway, ComplexityClassifier,
│                           TokenCounter, RequestStore, VerificationQueue, Clock
├── infrastructure/         adapters. the only layer that may import a vendor SDK
├── interfaces/             delivery: http/, worker/, cli/
└── config/                 pydantic-settings, YAML loaders
```

**The rule.** `domain` imports nothing from this project. `application` imports
only `domain`. `infrastructure` and `interfaces` may import `application` and
`domain`. Nothing ever imports outward.

`config` is a leaf: the edges read it and inject the result. The core never
reaches for configuration, because a use case that reads settings is a use case
you cannot test twice with different settings.

### Ports are `Protocol`, not ABC

Structural typing, so an adapter never inherits from anything in
`application/`. Two consequences that matter:

- `infrastructure` stays free of framework-shaped base classes.
- A test fake is a plain class with the right methods.

```python
class LLMGateway(Protocol):
    async def complete(
        self, request: CompletionRequest, model: ModelConfig, *, timeout_s: float
    ) -> LLMResponse: ...
```

### Errors are the dependency rule's payoff

`domain/errors.py` exists so that adapters can translate
`litellm.exceptions.RateLimitError` into `ProviderRateLimitError`, and the
application layer can write its `except` clauses without importing a vendor
package. Writing an `except` against a vendor exception is the single most
common way a "clean" architecture quietly stops being one — it is an import
that no one notices because it is not at the top of the file they are reading.

**Adapters translate. The core reacts.**

## Why this is not over-engineering at this size

Two payoffs, both of which appear in the demo rather than in a diagram:

1. The HTTP API and the verification worker are **different processes** that
   must share the exact same routing and escalation logic. That logic living in
   `application/`, free of FastAPI, is what makes that possible without
   duplicating it — and duplicated escalation logic that drifts is how the
   savings report and the live system start disagreeing.
2. Every test in the core suite runs without network access because
   `LLMGateway` is a port with an in-memory fake. That is not a testing
   nicety; it is what lets CI stay green on a pull request from a fork, where
   no secrets exist.

## Enforcement

Advisory rules rot inside a month. `.importlinter` runs in `make lint` and in
CI, and fails the build:

| Contract | What it forbids |
| :-- | :-- |
| `layers` | Any outward import between the four layers |
| `no-vendor-in-core` | `litellm`, `fastapi`, `sqlite3`, `sqlalchemy`, `sklearn`, `streamlit`, `httpx`, `openai`, `anthropic` inside `domain` or `application` |
| `litellm-single-entrypoint` | Any module except `infrastructure/litellm_env.py` importing `litellm` — see ADR-001 |
| `config-is-a-leaf` | `domain` or `application` importing `autopilot.config` |

This was verified rather than assumed. Two violations were deliberately
injected — `import litellm` in `application/ports.py`, and an import of
`infrastructure` inside `domain/errors.py` — and `lint-imports` failed with
exit code 1, reporting both, including the **transitive** path
`domain.errors → infrastructure.litellm_env → litellm`. Then reverted.

## Decisions taken during implementation

**`TokenCounter` is a port.** Counting tokens accurately needs a model-family
tokenizer, which lives in a package the core is forbidden to import. Rather
than weaken the rule, token counting became the sixth port, with a production
adapter and a heuristic implementation for tests. Without this, the feature
extractor and the dependency rule are in direct contradiction and CI fails on
the first run.

**`TokenCounter.count` takes a `ModelConfig`, not a model-name string.**
Tokenization depends on the provider model, and the catalog key and the
provider model id are both strings — so a signature taking `str` accepts the
wrong one happily and returns silently wrong counts. Passing the whole config
makes that mistake unrepresentable.

**`ModelConfig.provider_model_id`, not `litellm_model`.** A field named after
the current vendor bakes it into the domain vocabulary even though no import
crosses the line. The catalog describes models; translating an entry into
whatever string a particular gateway wants is the adapter's job.

**`SamplingStratum` is a required field on `VerificationJob`.** Quality parity
may only be estimated from the random stratum; the targeted strata are
deliberately enriched with requests already suspected of being mis-routed, so
pooling them biases the headline number downward and invalidates its confidence
interval. Making the stratum a required field turns that rule from a convention
someone has to remember into something the type system carries.

**`domain/policy.py` is intentionally empty.** The routing policy engine — YAML
schema, versioning, hot reload, budget-cap interaction — is a milestone of its
own. Stubbing its shape here would create a second definition for the real one
to drift against. What Phase 0 fixes is the *output* type it must produce:
`RoutingDecision`, with its invariants.

## Out of scope

Splitting into multiple distributable packages. One package, enforced layers.
