"""Vendor exception -> domain error, by name, without a vendor import.

`ERROR_MAP` is keyed by **string class name**, not by the litellm class
itself — this module imports nothing from `litellm` or `openai`. Matching
walks `type(exc).__mro__` and takes the first entry whose `__name__` is in
the table *and* whose `__module__` starts with a member of
`VENDOR_MODULE_PREFIXES`. Two consequences of that design:

1. The table is **order-independent**. `ContextWindowExceededError`,
   `ContentPolicyViolationError` and `RejectedRequestError` all subclass
   `BadRequestError`, which is also in the table — an ordered `isinstance`
   chain would only be correct if the specific entries preceded the general
   one, a property that lives in line order and that a merge can silently
   break. Walking the MRO makes specificity a property of Python's own class
   hierarchy instead.
2. The module-prefix gate (M2) stops a locally-defined class that merely
   *shares a name* with a vendor exception (e.g. a test double named
   `RateLimitError`) from being translated as if it came from the provider.

`translate()` is total: it always returns a `GatewayError`, falling back to
`ProviderResponseError` (with `__cause__` set and the original class name in
the message) for anything it does not recognise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from autopilot.domain.errors import (
    AuthenticationFailedError,
    ContentFilteredError,
    ContextWindowExceededError,
    GatewayError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

VENDOR_MODULE_PREFIXES: Final[tuple[str, ...]] = ("litellm.", "openai.")

CONTENT_FILTER_FINISH_REASONS: Final[frozenset[str]] = frozenset({"content_filter", "refusal"})
"""`finish_reason` values that mean the provider filtered a nominally-200
response. Unverified against real provider payloads (no recorded cassette
yet); an unrecognised value falls through to a normal success, the safe
direction. Owned by the follow-up cassette-recording issue."""


@dataclass(frozen=True, slots=True)
class ErrorRule:
    """One row of the mapping table.

    `litellm_class_name` is a plain string — never a class reference — which
    is what lets this whole module skip the vendor import entirely.
    """

    litellm_class_name: str
    domain_error: type[GatewayError]
    note: str


def _rule(litellm_class_name: str, domain_error: type[GatewayError], note: str) -> ErrorRule:
    return ErrorRule(litellm_class_name=litellm_class_name, domain_error=domain_error, note=note)


ERROR_MAP: Final[dict[str, ErrorRule]] = {
    r.litellm_class_name: r
    for r in (
        _rule("Timeout", ProviderTimeoutError, "openai.APITimeoutError subclass"),
        _rule(
            "RateLimitError",
            ProviderRateLimitError,
            "carries .category; we do not branch on it",
        ),
        _rule("ServiceUnavailableError", ProviderUnavailableError, ""),
        _rule("BadGatewayError", ProviderUnavailableError, ""),
        _rule("InternalServerError", ProviderUnavailableError, ""),
        _rule("APIConnectionError", ProviderUnavailableError, ""),
        _rule(
            "MidStreamFallbackError",
            ProviderUnavailableError,
            "subclasses ServiceUnavailableError; listed explicitly so the "
            "completeness test accounts for it",
        ),
        _rule("APIError", ProviderUnavailableError, "generic transport base; reached last via MRO"),
        _rule("AuthenticationError", AuthenticationFailedError, "401 — a key we have was rejected"),
        _rule("PermissionDeniedError", AuthenticationFailedError, "403"),
        _rule(
            "ContextWindowExceededError",
            ContextWindowExceededError,
            "escalate, never truncate",
        ),
        _rule("ContentPolicyViolationError", ContentFilteredError, ""),
        _rule("RejectedRequestError", ContentFilteredError, "provider refused"),
        _rule(
            "BadRequestError",
            ProviderResponseError,
            "base for ContextWindowExceededError/ContentPolicyViolationError/"
            "RejectedRequestError/others; MRO order handles the specificity",
        ),
        _rule("InvalidRequestError", ProviderResponseError, ""),
        _rule("UnprocessableEntityError", ProviderResponseError, ""),
        _rule("UnsupportedParamsError", ProviderResponseError, ""),
        _rule("ImageFetchError", ProviderResponseError, ""),
        _rule(
            "APIResponseValidationError",
            ProviderResponseError,
            "closest thing to malformed",
        ),
        _rule("JSONSchemaValidationError", ProviderResponseError, ""),
        _rule(
            "NotFoundError",
            ProviderResponseError,
            "a provider 404 is a different time/actor/remedy than "
            "ModelNotInCatalogError, which is our own lookup-miss error — keep separate",
        ),
        _rule("OpenAIError", ProviderResponseError, ""),
        _rule("LiteLLMUnknownProvider", ProviderResponseError, ""),
    )
}

DELIBERATELY_UNMAPPED: Final[dict[str, str]] = {
    "BudgetExceededError": (
        "litellm's own spend limiter. We never set max_budget, so it cannot fire. "
        "Name-collides with autopilot.domain.errors.BudgetExceededError; mapping it "
        "would make two unrelated concepts indistinguishable in logs"
    ),
    "MockException": "litellm's test scaffolding",
    "GuardrailRaisedException": "guardrails not configured",
    "BlockedPiiEntityError": "guardrails not configured",
    "ModifyResponseException": "proxy-server concern",
    "SensitiveDataRouteException": "proxy-server concern",
}


def translate(exc: BaseException, *, model_key: str | None = None) -> GatewayError:
    """Vendor exception -> domain error. Total: always returns a GatewayError."""
    for klass in type(exc).__mro__:
        rule = ERROR_MAP.get(klass.__name__)
        if rule is not None and klass.__module__.startswith(VENDOR_MODULE_PREFIXES):
            translated = rule.domain_error(str(exc), model_key=model_key)
            translated.__cause__ = exc
            return translated

    fallback = ProviderResponseError(
        f"unmapped provider exception {type(exc).__name__}: {exc}",
        model_key=model_key,
    )
    fallback.__cause__ = exc
    return fallback
