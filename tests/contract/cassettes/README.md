# Cassettes

Recorded provider payloads. Committed, replayed in CI, never re-recorded automatically.

## Why they exist

The unit tier proves our logic is right against a fake. It cannot prove the
LiteLLM adapter parses what a provider actually sends. That is what this tier
is for, and it is the only tier that has ever seen a real payload.

## Recording

`record_mode` is `none` everywhere by default, so a missing cassette fails
rather than silently reaching for the network. Recording is an explicit,
manual, local act:

```bash
cp .env.example .env      # then fill in the provider key you are recording
uv run pytest tests/contract -m contract --record-mode=once
```

Then, **before committing**:

1. Read the diff. Not skim — read it. The scrubber in
   `tests/conftest.py::vcr_config` removes known auth headers, but it cannot
   know that a provider decided to echo your organisation id in a new field.
2. Run `uv run pytest tests/contract/test_cassette_hygiene.py`.

If a credential ever reaches a cassette, editing the file is not the fix. It is
in the git history and in every clone. **Rotate the key.**

## Scope

V1 records **success plus one failure scenario for one provider**. Provoking
genuine 429s, context-window overflows and malformed responses across three
providers means deliberately abusing rate limits and is a day of work that the
typed fakes already cover behaviourally.

The full matrix — every provider × {success, 429, overflow, timeout, malformed}
— is deferred. What is lost by deferring it: we would not catch a provider
changing the *shape* of an error response. What makes that acceptable for now:
the adapter funnels every unrecognised failure into `GatewayError`, so an
unexpected shape degrades to a retry and a fallback rather than a crash.

## Deferred from `unified-model-interface` (issue #7, D8)

Two scenarios in `LiteLLMGateway`'s error taxonomy are unit-tested only
against hand-built stand-ins, not against a recorded real payload:

- `ContentFiltered` on a genuinely successful (200) real-provider response.
- A genuinely malformed real-provider payload (reuses `ProviderResponseError`;
  no new type).

Both require real credentials and, for the first one, deliberately provoking
a policy violation — out of scope for this change per the proposal's D8.
`LiteLLMGateway` now exists to record against (it did not before slice 7), so
recording is possible for the first time; it is owned by the named follow-up
cassette-recording issue this change unblocks, not by this change itself.
