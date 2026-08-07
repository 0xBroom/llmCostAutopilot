# ADR-001: Adopt the LiteLLM SDK as the provider abstraction (embedded, not Proxy)

- **Status:** Accepted
- **Deciders:** project architect
- **Supersedes:** nothing
- **Related:** [ADR-002](0002-hexagonal-layout.md)

## Context

This system routes requests across at least three LLM providers. Something has
to normalise their wire formats, their token-usage fields, their error taxonomy
and their pricing. There were three ways to get that.

A reviewer will look at this repo and ask, within about ninety seconds:
*"if LiteLLM already routes, what did you actually build?"* The answer has to be
crisp, and the architecture has to make it obviously true.

| Option | What it is | Verdict |
| :-- | :-- | :-- |
| **A. Hand-rolled adapters** | One client class per provider, a hand-maintained price table, manual usage normalisation | **Rejected.** Roughly 600 lines of undifferentiated plumbing that demonstrates nothing, plus a price table that is stale within weeks and quietly corrupts every cost figure downstream. |
| **B. LiteLLM Proxy Server** | Run LiteLLM's standalone OpenAI-compatible gateway as a service, configure it with `config.yaml` | **Rejected.** See below. |
| **C. LiteLLM Python SDK, embedded** | Import `litellm` inside our own FastAPI app; we own the request path | **Accepted.** |

## Decision

Use the **LiteLLM Python SDK embedded in our own service**.

### Why not the Proxy

1. **We need to own the request path.** The value of this system lives strictly
   *between* "request arrives" and "provider is called": feature extraction,
   classification, policy lookup, and the decision record that comes out of it.
   The Proxy owns exactly that stretch and exposes only config-level hooks into
   it. We would spend the project fighting the tool for the one thing the
   project exists to demonstrate.
2. **It would obscure authorship.** A repository that is mostly a `config.yaml`
   for someone else's gateway reads as configuration work.

### What we take from LiteLLM

Verified against its documentation before being written down here:

- `litellm.acompletion(model=..., messages=..., **params)` — async-native.
- `litellm.completion_cost(completion_response=resp)` and
  `litellm.cost_per_token(model, prompt_tokens, completion_tokens)`.
- `litellm.model_cost` — the price and context-window map, as the *default*
  source of per-token pricing.
- `litellm.register_model({...})` — to inject pricing for models the map does
  not cover, notably local Ollama models:

  ```python
  litellm.register_model({
      "ollama_chat/llama3.1": {
          "max_tokens": 8192,
          "input_cost_per_token": 0.0,
          "output_cost_per_token": 0.0,
          "litellm_provider": "ollama",
          "mode": "chat",
      },
  })
  ```

- `litellm.Router` — retries, fallbacks, cooldowns, context-window fallbacks.
- `litellm.integrations.custom_logger.CustomLogger` with
  `async_log_success_event` / `async_log_failure_event`, registered through
  `litellm.callbacks = [...]`. Per-request metadata comes back at
  `kwargs["litellm_params"]["metadata"]`.

### What LiteLLM does not do, and we therefore build

Its routing strategies — `simple-shuffle`, `latency-based-routing`,
`usage-based-routing-v2`, `least-busy`, `cost-based-routing` — all reason about
**deployment metrics**. Latency, load, spend. Not one of them reads the prompt.

> LiteLLM answers *"which deployment of the model I chose?"*.
> This system answers *"which model does this request actually deserve?"*.

So we build: complexity classification of the prompt, the tier → model policy
with an explainable decision record, sampled asynchronous quality verification
with an LLM judge, auto-escalation and its accounting, the failure → retraining
flywheel, and counterfactual net-savings accounting.

There is a corollary that is easy to get wrong and expensive to notice: the
Router must be configured with **one deployment group per catalog key, never one
per tier**. Grouping several models under a single `model_name` and letting
`simple-shuffle` pick between them hands model selection straight back to
LiteLLM, and the policy decision becomes decorative while still being logged as
if it were real.

## Consequences

**Positive.** The provider matrix becomes a configuration concern — a fourth
provider is a YAML entry, not a code change. Pricing tracks upstream instead of
rotting in a table we maintain.

**Negative — dependency weight.** We inherit a large package and its release
cadence. Mitigated by pinning `litellm` to an **exact** version in
`pyproject.toml`, not a range, because the price map ships *inside* the package
and a minor bump can silently change every number in the savings report.
Upgrades are a reviewed change accompanied by a re-run of the benchmark.

**Negative — the price map can be wrong.** It is community-maintained.
Mitigated in two layers: our YAML catalog can override any entry, and the price
actually used is frozen onto each request row (`CostBreakdown` carries the unit
prices that produced it), so a row is auditable long after the map moves.

**Risk, and how it is closed.** LiteLLM may consult a remote price map at
import time. If it does, cost figures depend on when the process started and
whether it had network access — which makes the headline savings number
irreproducible. `LITELLM_LOCAL_MODEL_COST_MAP=True` pins it to the packaged
copy, but the variable is only read **when `litellm` is first imported**. Set it
afterwards and it does nothing, silently.

Documenting that is not enough, because the failure mode is someone adding an
`import litellm` higher in the import graph a month from now and nothing going
red. So the guarantee is structural:

- `src/autopilot/infrastructure/litellm_env.py` sets the variable and is the
  only module in the repository that imports `litellm`.
- An `.importlinter` contract fails the build if any other module imports it —
  including transitively.
- `tests/unit/test_litellm_boundary.py` asserts the *source order* inside the
  shim, which no type checker or import graph can see.

## Compliance

- [x] `pyproject.toml` pins `litellm==1.95.0` exactly, with the reason in a comment.
- [x] The README states in one sentence what LiteLLM does and what this project adds.
- [x] The single-entrypoint contract is enforced by `make lint` and by CI.
- [x] A deliberate violation was injected and confirmed to fail with exit code 1.
- [x] One Router deployment group per catalog key, never per tier — asserted by
      `tests/unit/test_router_deployments.py::test_one_deployment_group_per_catalog_key`
      and `::test_each_group_points_at_exactly_its_own_provider_model_id`.
- [x] `content_policy_fallbacks` and `context_window_fallbacks` are left unset, so
      a safety refusal or a context overflow surfaces as a domain error instead of
      being silently retried elsewhere — asserted by
      `tests/unit/test_router_factory.py::test_content_policy_fallbacks_stay_unset`
      and `::test_context_window_fallbacks_stay_unset`.
- [x] No wildcard `{"*": [...]}` fallback entry reaches the Router — asserted by
      `tests/unit/test_router_factory.py::test_no_wildcard_fallback_entry`.
