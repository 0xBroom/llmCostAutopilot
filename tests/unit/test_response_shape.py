"""Keeping `FakeModelResponse` honest without a cassette.

`FakeModelResponse` has no Protocol to check against — `litellm.ModelResponse`
is the vendor's type, not ours. This is the honest half of what a cassette
would prove: not that a real payload has exactly this shape (that is the
cassette's job, see `tests/contract/cassettes/README.md`), but that every
attribute our fake claims a response has, litellm's own real type actually
has too. Silently widening the fake to make a gateway test pass would show up
here as a diff in `READ_ATTRIBUTES`.

Imports litellm through the shim, same as `test_router_factory.py` and
`test_price_agreement.py`.
"""

from __future__ import annotations

import dataclasses
from typing import Final

from autopilot.infrastructure.litellm_env import litellm
from tests.fakes.response import FakeModelResponse

READ_ATTRIBUTES: Final[tuple[str, ...]] = ("model", "choices", "usage", "id", "_hidden_params")


def test_the_real_response_type_has_every_attribute_our_fake_provides() -> None:
    real = litellm.ModelResponse(
        model="gpt-4o",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        usage=litellm.Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    for attr in READ_ATTRIBUTES:
        assert hasattr(real, attr)

    fake_fields = {f.name for f in dataclasses.fields(FakeModelResponse)}
    assert set(READ_ATTRIBUTES) <= fake_fields | {"_hidden_params"}
