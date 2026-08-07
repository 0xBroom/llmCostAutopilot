"""The domain error taxonomy.

This module is what makes the dependency rule pay for itself. Adapters catch
vendor exceptions — `litellm.exceptions.RateLimitError`, `sqlite3.OperationalError`
— and translate them into the types below. The application layer therefore
never imports a vendor package to write an `except` clause, which is the most
common way a "clean" architecture quietly stops being one.

Adapters translate. The core reacts.
"""

from __future__ import annotations

from typing import ClassVar, Final


class AutopilotError(Exception):
    """Base class. Catching this catches everything the system raises on purpose."""


# --- Configuration ------------------------------------------------------------


class ConfigurationError(AutopilotError):
    """The system cannot start, or cannot honour a request, because of config."""


class ModelNotInCatalogError(ConfigurationError):
    def __init__(self, key: str) -> None:
        super().__init__(f"model {key!r} is not in the catalog")
        self.key = key


class MissingCredentialsError(ConfigurationError):
    """No credential was configured for this provider at all.

    Fires at Router-construction time (`build_router`), before any request is
    routed. Distinct from `AuthenticationFailedError`: that one fires at call
    time, when a credential that *was* configured is rejected by the provider.
    Different time, different actor, different remedy — never collapse them.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(f"no credentials configured for provider {provider!r}")
        self.provider = provider


# --- Policy -------------------------------------------------------------------


class PolicyError(AutopilotError):
    """The routing policy could not produce a decision."""


class NoModelForTierError(PolicyError):
    def __init__(self, tier: int) -> None:
        super().__init__(f"routing policy has no model for tier {tier}")
        self.tier = tier


class BudgetExceededError(AutopilotError):
    """A spend cap would be breached by serving this request."""


# --- Gateway ------------------------------------------------------------------


class GatewayError(AutopilotError):
    """Something went wrong on the provider side of the boundary.

    `retryable` is a ClassVar, not a field on `infrastructure.error_translation
    .ErrorRule`. Several distinct litellm exception classes map to the same
    domain error (e.g. four map to `ProviderUnavailableError`); retryability is
    a property of the failure *kind*, so it lives once on the kind, not once
    per mapping rule where four rules could silently disagree with each other.
    """

    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, model_key: str | None = None) -> None:
        super().__init__(message)
        self.model_key = model_key


class ProviderTimeoutError(GatewayError):
    """The provider did not answer inside the request timeout."""

    retryable: ClassVar[bool] = True


class ProviderRateLimitError(GatewayError):
    """429 or equivalent. Retryable, and a signal to the fallback chain."""

    retryable: ClassVar[bool] = True


class ContextWindowExceededError(GatewayError):
    """The prompt does not fit the chosen model. Escalate, do not truncate.

    Silently trimming a prompt to make it fit produces a plausible answer to a
    different question, which is the worst possible failure mode for a system
    whose whole claim is that quality is preserved.
    """


class ProviderUnavailableError(GatewayError):
    """The provider (or the transport to it) is down: 503/502/500 or a
    connection failure. Retryable — a different deployment, or the same one
    after cooldown, may well succeed."""

    retryable: ClassVar[bool] = True


class AuthenticationFailedError(GatewayError):
    """A credential that *was* configured was rejected by the provider
    (401/403). See `MissingCredentialsError` for the distinct, earlier
    failure of no credential being configured at all."""


class ContentFilteredError(GatewayError):
    """The provider refused or filtered the response on policy grounds,
    either as an exception at call time or via a filtered `finish_reason` on
    an otherwise-200 response. Not retryable: retrying the same request
    against the same policy produces the same refusal."""


class ProviderResponseError(GatewayError):
    """The provider answered, but not with something we can use."""


class AllProvidersFailedError(GatewayError):
    """Every model in the fallback chain failed. Nothing left to try."""


GATEWAY_TAXONOMY: Final[frozenset[type[GatewayError]]] = frozenset(
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
"""The exact set of domain errors `infrastructure.error_translation.translate()`
may produce from a vendor exception. `AllProvidersFailedError` is deliberately
excluded — it is raised by the gateway after exhausting a fallback chain, never
by `translate()` on a single exception. This is the set the completeness test
in `error_translation.py` uses to confirm every taxonomy row has a rule
pointing at it."""


# --- Classification -----------------------------------------------------------


class ClassificationError(AutopilotError):
    """The classifier could not produce a usable prediction."""


class ClassifierNotLoadedError(ClassificationError):
    """No trained artifact is available. Fail loudly rather than routing blind."""


# --- Verification -------------------------------------------------------------


class VerificationError(AutopilotError):
    """The async quality check could not complete."""


class JudgeUnavailableError(VerificationError):
    """The judge model failed. The job is retryable; the verdict is not invented."""
