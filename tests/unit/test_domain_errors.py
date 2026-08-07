"""The #7 gateway error taxonomy: new subclasses, `retryable` flags, and the
frozen set `error_translation.py`'s completeness test polices.

`retryable` is a `ClassVar` on the domain error, not a column on the mapping
table (design §4.2 M3): four distinct litellm classes map to
`ProviderUnavailableError`, and a per-rule column would let those four
disagree with each other about whether the *same domain error* is retryable.
Retryability is a property of the failure kind, so these tests assert it
directly on the class.
"""

from __future__ import annotations

from autopilot.domain.errors import (
    GATEWAY_TAXONOMY,
    AllProvidersFailedError,
    AuthenticationFailedError,
    ConfigurationError,
    ContentFilteredError,
    ContextWindowExceededError,
    GatewayError,
    MissingCredentialsError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def test_gateway_error_base_defaults_to_not_retryable() -> None:
    assert GatewayError.retryable is False


def test_existing_retryable_flags_unchanged() -> None:
    assert ProviderTimeoutError.retryable is True
    assert ProviderRateLimitError.retryable is True
    assert ContextWindowExceededError.retryable is False
    assert ProviderResponseError.retryable is False
    assert AllProvidersFailedError.retryable is False


def test_provider_unavailable_is_retryable() -> None:
    assert ProviderUnavailableError.retryable is True


def test_authentication_failed_is_not_retryable() -> None:
    assert AuthenticationFailedError.retryable is False


def test_content_filtered_is_not_retryable() -> None:
    assert ContentFilteredError.retryable is False


def test_new_gateway_errors_carry_model_key_and_are_gateway_errors() -> None:
    for cls in (ProviderUnavailableError, AuthenticationFailedError, ContentFilteredError):
        err = cls("boom", model_key="haiku-4-5")
        assert err.model_key == "haiku-4-5"
        assert isinstance(err, GatewayError)


def test_authentication_failed_is_distinct_from_missing_credentials() -> None:
    """MissingCredentialsError fires at Router-construction time because no
    credential was configured at all. AuthenticationFailedError fires at call
    time because a credential that *was* configured was rejected by the
    provider. Different time, different actor, different remedy — they must
    never collapse into one type."""
    assert not issubclass(MissingCredentialsError, GatewayError)
    assert not issubclass(AuthenticationFailedError, ConfigurationError)
    assert MissingCredentialsError is not AuthenticationFailedError


def test_gateway_taxonomy_names_exactly_the_seven_translate_targets() -> None:
    """The seven rows of the spec's error-taxonomy table — the exact set
    `error_translation.py`'s completeness test uses to confirm nothing was
    added to the taxonomy without a rule pointing at it.

    `AllProvidersFailedError` is deliberately excluded: it is raised by the
    gateway itself after exhausting a fallback chain, never by `translate()`
    on a single vendor exception.
    """
    expected = frozenset(
        {
            ProviderTimeoutError,
            ProviderRateLimitError,
            ProviderUnavailableError,
            AuthenticationFailedError,
            ContextWindowExceededError,
            ContentFilteredError,
            ProviderResponseError,
        }
    )
    assert expected == GATEWAY_TAXONOMY
