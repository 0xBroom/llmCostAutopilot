"""The single Decimal money-conversion chokepoint.

Per-token prices are around 1e-7 USD, and litellm's own price map — along with
every YAML override and every canary comparison against
`litellm.completion_cost()` — hands them over as native Python floats. Calling
`Decimal(x)` directly on one of those floats re-encodes its binary
approximation exactly (`Decimal(0.0000008) == Decimal('7.99999999999999...e-7')`,
not `Decimal("0.0000008")`), which is invisible until the savings report
disagrees with the provider invoice after being summed a few thousand times.

`Decimal(str(x))` re-encodes the decimal string Python already printed for
that float instead, which is what a human — and an invoice — actually means
by the number. `to_price` is the only place in `src/` this happens; see
`tests/unit/test_decimal_boundary.py` for the AST rule that keeps it that way.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Final

from autopilot.domain.errors import ConfigurationError

MAX_PLAUSIBLE_PRICE_PER_TOKEN: Final[Decimal] = Decimal("1")
"""A per-token price above one US dollar is not a real price — it is a
misconfiguration (a botched unit, a per-1K/per-1M value pasted in raw). Reject
it loudly at load time rather than silently pricing every request wrong."""

_ACCEPTED_TYPES: Final[tuple[type, ...]] = (int, float, str, Decimal)


def to_price(value: object, *, field: str, model_key: str) -> Decimal:
    """Convert an untrusted price value into an exact `Decimal`.

    `value` may be an `int`, `float`, `str` (including YAML-style padded
    scalars), or `Decimal` already. Anything else — notably `bool` (a `bool`
    is numerically an `int` in Python, but a price of `True` is a type error,
    not a price of 1) — is rejected.

    Raises `ConfigurationError`, naming both `field` and `model_key`, for:
    non-numeric input, `NaN`/infinite values, negative values, and values
    above `MAX_PLAUSIBLE_PRICE_PER_TOKEN`.
    """
    if isinstance(value, bool) or not isinstance(value, _ACCEPTED_TYPES):
        raise ConfigurationError(
            f"{field} for model {model_key!r} must be a number, got {type(value).__name__}"
        )

    try:
        price = Decimal(str(value))
    except InvalidOperation as exc:
        raise ConfigurationError(
            f"{field} for model {model_key!r} is not a valid number: {value!r}"
        ) from exc

    if not price.is_finite():
        raise ConfigurationError(
            f"{field} for model {model_key!r} must be a finite number, got {value!r}"
        )
    if price < 0:
        raise ConfigurationError(
            f"{field} for model {model_key!r} cannot be negative, got {value!r}"
        )
    if price > MAX_PLAUSIBLE_PRICE_PER_TOKEN:
        raise ConfigurationError(
            f"{field} for model {model_key!r} exceeds the plausible per-token price "
            f"ceiling ({MAX_PLAUSIBLE_PRICE_PER_TOKEN}): {value!r}"
        )
    return price
