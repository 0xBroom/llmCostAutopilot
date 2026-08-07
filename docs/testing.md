# Test strategy

Every interesting component in this system talks to a paid, non-deterministic,
rate-limited external service. That fact drives everything below.

If tests call real providers: the suite is slow, flaky, costs money on every
push, and **cannot run on a pull request from a fork**, because forks do not
get repository secrets. If tests mock everything at the SDK boundary: they stay
green while the real integration is broken, which is worse than having no tests
because it produces confidence.

So the tiers are deliberate, and the boundary each one sits on is deliberate.

## The rule

**The default suite never spends money and never touches the network.**

A green CI run on a fork's pull request is the proof. No provider secrets are
configured on this repository — not as a precaution, but as a forcing function.
If a test needs a credential to pass, the test is in the wrong tier.

## The tiers

### 1. Unit — no I/O, milliseconds, the bulk of the suite

Targets `domain/` and `application/`. Every port is replaced with an in-memory
fake. Routing policy, escalation rules, cost arithmetic and the sampling
decision are tested exhaustively here, because here they are cheap to test.

The fakes live in `tests/fakes/` and are **hand-written classes, not
`unittest.mock`**. This is not taste. A `MagicMock` accepts any call signature,
so on the day someone adds a parameter to `LLMGateway.complete`, the mock
absorbs it and every test stays green while production breaks.
`tests/unit/test_ports_contract.py` assigns each fake to its `Protocol`, so
**mypy** catches exactly that drift:

```python
_gateway: LLMGateway = FakeLLMGateway()   # fails `make typecheck` if the port moves
```

The fakes are also tested themselves (`tests/unit/test_fakes.py`). Test
infrastructure that lies is worse than none — a queue fake that pops on read
instead of leasing would make it impossible to test the case the real system
has to survive, a worker dying mid-job, and would let a redelivery bug ship
green.

### 2. Contract — the adapter against a recorded provider

Proves the LiteLLM adapter parses what a provider actually sends. Record once
with real credentials, scrub, commit, replay forever. `record_mode` is `none`,
so a missing cassette fails rather than silently reaching for the network.

Scrubbing is configured at **record** time (`tests/conftest.py::vcr_config`),
not reviewed at commit time, because a key that reaches disk is already in the
reflog and `git rm` does not remove it.
`tests/contract/test_cassette_hygiene.py` is the check that the scrubbing
actually worked; it passes vacuously with zero cassettes, which is intentional —
the guard has to exist before the first recording, not after the first leak.

**V1 scope:** success plus one failure scenario, for one provider. Provoking
genuine 429s, context-window overflows and malformed responses across three
providers means deliberately abusing rate limits, and it is a day of work that
the typed fakes already cover behaviourally. The full matrix is deferred; see
`tests/contract/cassettes/README.md` for what that costs us.

### 3. Integration — our own services, no external network

FastAPI through `httpx.ASGITransport`, real SQLite on a temporary file, the
real worker loop, a fake gateway. Asserts the full path: request → decision →
row persisted → verification job enqueued.

### 4. Live — opt-in, never in the default suite

Marked `@pytest.mark.live` and deselected by `addopts = "-m 'not live'"`. Run
by hand with real keys before a release or a demo:

```bash
make test-live
```

## Determinism

Three sources of non-determinism, three ports or seeds:

| Source | How it is controlled |
| :-- | :-- |
| Time | the `Clock` port. No `datetime.now()` in `domain`/`application`, and ruff's `DTZ` rules enforce timezone-awareness everywhere. |
| Randomness | an injected `random.Random(seed)`. The verification sampler must be replayable, or a savings report cannot be reproduced. |
| The classifier | a small fixture artifact committed under `tests/fixtures/`, never trained at test time. |

A `conftest.py` autouse fixture strips `AUTOPILOT_*`, `OPENAI_*`, `ANTHROPIC_*`
and `LITELLM_*` from the environment for every test. A test that passes on a
laptop because a key happened to be exported will fail in CI — or worse, pass
in CI by silently spending money.

## Gates

**Coverage: 75%, on `domain/` and `application/` only.**

No global gate, on purpose. A repository-wide percentage is trivially satisfied
by testing adapters that contract tests should be covering, which makes the
number go up while confidence goes down. Gating only the core means the number
measures the thing worth measuring.

**No performance assertions in CI.**

The latency budgets in this system are real and are measured — locally, and
printed to the benchmark report. Shared runners vary too much to gate on, and a
flaky gate teaches the whole team to ignore CI, which is strictly worse than
having no gate.

**CI runs the same three commands as `make check`.** Split into separate steps
only so a red build says *which* gate failed without opening the log.
