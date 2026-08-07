"""`infrastructure.pricing.to_price` — the single money-conversion chokepoint.

Demonstrates the bug it exists to avoid, not just forbids it: `Decimal(x)` on
a bare float re-encodes the float's binary approximation, while
`Decimal(str(x))` re-encodes the decimal string Python already printed for
that float, which is what a human — and a provider invoice — actually means
by that number.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from autopilot.domain.errors import ConfigurationError
from autopilot.infrastructure.pricing import MAX_PLAUSIBLE_PRICE_PER_TOKEN, to_price


def test_str_conversion_is_what_makes_the_price_exact() -> None:
    assert Decimal(str(0.0000008)) == Decimal("0.0000008")
    assert Decimal(0.0000008) != Decimal("0.0000008")  # noqa: RUF032 — the bug, demonstrated
    assert to_price(0.0000008, field="input_cost_per_token", model_key="haiku-3-5") == Decimal(
        "0.0000008"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0000008, Decimal("0.0000008")),
        ("0.0000008", Decimal("0.0000008")),
        ("  0.0000008  ", Decimal("0.0000008")),  # YAML-style padded scalar
        (1, Decimal("1")),
        (0, Decimal("0")),
        (0.0, Decimal("0")),
        (Decimal("0.0000035"), Decimal("0.0000035")),
    ],
)
def test_to_price_handles_int_float_str_and_decimal_passthrough(
    value: object, expected: Decimal
) -> None:
    assert to_price(value, field="input_cost_per_token", model_key="x") == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        -0.0001,
        Decimal("-1"),
        "not-a-number",
        [],
    ],
)
def test_to_price_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ConfigurationError):
        to_price(value, field="input_cost_per_token", model_key="x")


def test_to_price_rejects_a_price_above_the_plausible_ceiling() -> None:
    with pytest.raises(ConfigurationError):
        to_price(
            MAX_PLAUSIBLE_PRICE_PER_TOKEN + Decimal("0.01"),
            field="input_cost_per_token",
            model_key="x",
        )


def test_to_price_accepts_the_ceiling_price_exactly() -> None:
    assert (
        to_price(MAX_PLAUSIBLE_PRICE_PER_TOKEN, field="input_cost_per_token", model_key="x")
        == MAX_PLAUSIBLE_PRICE_PER_TOKEN
    )


def test_to_price_error_names_field_and_model_key() -> None:
    with pytest.raises(ConfigurationError, match="input_cost_per_token"):
        to_price(-1, field="input_cost_per_token", model_key="haiku-3-5")
    with pytest.raises(ConfigurationError, match="haiku-3-5"):
        to_price(-1, field="input_cost_per_token", model_key="haiku-3-5")
