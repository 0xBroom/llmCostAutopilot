"""`RouterLike` is not a port.

`tests/unit/test_ports_contract.py`'s own docstring says "every fake must
satisfy its port" — `RouterLike` does not qualify: the application layer has
never heard of it. It is an adapter-local seam that exists so
`LiteLLMGateway` can be unit-tested against a hand-written `FakeRouter`
instead of a real `litellm.Router`. Same mypy-assignment technique as the
ports contract, in its own file so that docstring stays accurate.
"""

from __future__ import annotations

from autopilot.infrastructure.litellm_gateway import RouterLike
from tests.fakes.router import FakeRouter


def test_fake_router_satisfies_router_like() -> None:
    # Static: mypy verifies this assignment against RouterLike's signature.
    _router: RouterLike = FakeRouter()

    # Runtime: member presence only — a runtime_checkable Protocol never
    # checks signatures, only that the members exist.
    assert isinstance(_router, RouterLike)
