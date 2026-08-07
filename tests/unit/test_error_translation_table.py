"""Table-integrity tests for `infrastructure.error_translation`.

Zero vendor import here — `ERROR_MAP` is keyed by string class name, and every
assertion below can be made without ever importing litellm or openai. That is
the whole point of the string-keyed design: the mapping table's own
correctness is testable in the fast unit tier, at zero vendor-import cost.

The completeness check against the *actual* installed litellm exception
surface lives in a separate file, `test_error_translation_completeness.py`,
because that one does need to read (not import) litellm's source.
"""

from __future__ import annotations

from autopilot.domain.errors import (
    GATEWAY_TAXONOMY,
    GatewayError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from autopilot.infrastructure.error_translation import (
    DELIBERATELY_UNMAPPED,
    ERROR_MAP,
    translate,
)


def test_every_rule_targets_a_gateway_error_subclass() -> None:
    for rule in ERROR_MAP.values():
        assert issubclass(rule.domain_error, GatewayError)


def test_every_taxonomy_member_is_the_target_of_at_least_one_rule() -> None:
    """Catches "added a taxonomy row, forgot the mapping" — exactly the miss
    a 75% line-coverage gate would wave through."""
    targeted = {rule.domain_error for rule in ERROR_MAP.values()}
    assert targeted >= GATEWAY_TAXONOMY


def test_error_map_keys_are_unique_non_empty_and_self_describing() -> None:
    for key, rule in ERROR_MAP.items():
        assert key, "an ERROR_MAP key must not be empty"
        assert rule.litellm_class_name == key, (
            "ErrorRule.litellm_class_name must match the dict key it's stored under"
        )
    # a dict cannot have duplicate keys by construction, but guard against a
    # future refactor to a list-of-rules shape losing that guarantee silently
    assert len(ERROR_MAP) == len({rule.litellm_class_name for rule in ERROR_MAP.values()})


def test_deliberately_unmapped_entries_carry_a_nonempty_reason() -> None:
    for class_name, reason in DELIBERATELY_UNMAPPED.items():
        assert class_name
        assert reason


def test_error_map_and_deliberately_unmapped_do_not_overlap() -> None:
    assert not (set(ERROR_MAP) & set(DELIBERATELY_UNMAPPED))


class _NotAVendorError(Exception):
    """A plain local exception. No provider SDK raises this."""


def test_translate_on_a_synthetic_non_vendor_exception_falls_back_to_provider_response_error() -> (
    None
):
    exc = _NotAVendorError("weird")

    result = translate(exc)

    assert isinstance(result, ProviderResponseError)
    assert result.__cause__ is exc
    assert "_NotAVendorError" in str(result)


def test_translate_propagates_model_key_on_the_fallback_path() -> None:
    result = translate(_NotAVendorError("x"), model_key="haiku-4-5")
    assert result.model_key == "haiku-4-5"


class RateLimitError(Exception):
    """Locally-defined, same NAME as litellm's `RateLimitError`, wrong module.

    This is the M2 gate: `translate` must not match on class name alone, or
    any exception a test (or a bug) happens to name the same as a vendor
    class would be silently translated as if it came from the provider.
    """


def test_a_locally_defined_class_named_like_a_vendor_exception_does_not_match() -> None:
    exc = RateLimitError("not actually litellm's")

    result = translate(exc)

    assert isinstance(result, ProviderResponseError)
    assert not isinstance(result, ProviderRateLimitError)
