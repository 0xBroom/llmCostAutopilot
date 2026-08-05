# LLM Cost Autopilot

A routing layer that reads each incoming LLM request, decides how much model it
actually deserves, sends it to the cheapest model that can handle it — and then
keeps checking whether that decision was right.

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
