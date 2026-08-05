"""The domain error taxonomy.

This module is what makes the dependency rule pay for itself. Adapters catch
vendor exceptions — `litellm.exceptions.RateLimitError`, `sqlite3.OperationalError`
— and translate them into the types below. The application layer therefore
never imports a vendor package to write an `except` clause, which is the most
common way a "clean" architecture quietly stops being one.

Adapters translate. The core reacts.
"""

from __future__ import annotations


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
    """Something went wrong on the provider side of the boundary."""

    def __init__(self, message: str, *, model_key: str | None = None) -> None:
        super().__init__(message)
        self.model_key = model_key


class ProviderTimeoutError(GatewayError):
    """The provider did not answer inside the request timeout."""


class ProviderRateLimitError(GatewayError):
    """429 or equivalent. Retryable, and a signal to the fallback chain."""


class ContextWindowExceededError(GatewayError):
    """The prompt does not fit the chosen model. Escalate, do not truncate.

    Silently trimming a prompt to make it fit produces a plausible answer to a
    different question, which is the worst possible failure mode for a system
    whose whole claim is that quality is preserved.
    """


class ProviderResponseError(GatewayError):
    """The provider answered, but not with something we can use."""


class AllProvidersFailedError(GatewayError):
    """Every model in the fallback chain failed. Nothing left to try."""


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
