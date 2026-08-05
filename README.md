# LLM Cost Autopilot

A routing layer that reads each incoming LLM request, decides how much model it
actually deserves, sends it to the cheapest model that can handle it — and then
keeps checking whether that decision was right.

> **LiteLLM answers *"which deployment of the model I chose?"*.
> This project answers *"which model does this request actually deserve?"*.**

That sentence is the whole scope boundary. LiteLLM is the provider plumbing: one
async call signature across OpenAI, Anthropic and Ollama, a maintained price
map, normalised token usage, retries and fallbacks. What it does **not** do —
and this is confirmed against its routing documentation — is pick a model based
on the *content* of the request. Every one of its strategies (`simple-shuffle`,
`latency-based-routing`, `usage-based-routing-v2`, `least-busy`,
`cost-based-routing`) reasons about deployment metrics: latency, load, spend.
None of them read the prompt.

Reading the prompt, pricing the decision, and proving it was correct is what
this project builds.

## Getting started

```bash
git clone https://github.com/0xBroom/llmCostAutopilot.git
cd llmCostAutopilot
make install
cp .env.example .env
make check
```

`make check` runs lint, type checking and the test suite.

Everything else is a `make` target:

```
make help        list all targets
make format      apply ruff fixes and formatting
make test        the test suite
```

## Architecture

Ports and adapters, with the dependency rule enforced mechanically rather than
by convention:

```
src/autopilot/
├── domain/           pure types and rules. no I/O, no vendor SDKs
├── application/      use cases + the port Protocols
├── infrastructure/   adapters. the only layer that may import a vendor SDK
├── interfaces/       HTTP, worker, CLI
└── config/           typed settings and YAML loaders
```

`import-linter` fails the build if `domain` or `application` import a vendor
SDK or a framework. That contract is what makes the claim "swap the provider
layer without touching business logic" verifiable instead of aspirational — and
it is the cheapest way to keep it true as the codebase grows.

One contract is worth calling out because it looks odd: **only
`infrastructure/litellm_env.py` may import `litellm`**. LiteLLM reads
`LITELLM_LOCAL_MODEL_COST_MAP` once, at first import, to decide whether to pin
its price map to the packaged copy or fetch a newer one over the network. Set it
too late and it does nothing, silently, and cost figures stop being
reproducible. Funnelling the import through one module puts the environment
variable structurally above it, and a test asserts the source order.

Decisions with reasoning are in [`docs/adr/`](docs/adr/).

## Configuration

Every variable the service reads is listed in `.env.example`, which is
committed and must stay complete. They are parsed into a single typed
`Settings` object (`src/autopilot/config/settings.py`); there is no
`os.getenv()` anywhere else in the codebase, because scattered environment
reads make it impossible to answer "what does this service need to run?"
without grepping.

Provider credentials are `SecretStr`, so a settings dump in an exception
handler prints `**********` instead of an API key.

## Status

Under construction. Work is tracked in
[the issues](https://github.com/0xBroom/llmCostAutopilot/issues); the epic is
[#1](https://github.com/0xBroom/llmCostAutopilot/issues/1) and is the source of
truth for scope and sequencing.

## License

MIT.
