"""Typed configuration. The single place the process reads its environment.

There is no `os.getenv()` anywhere else in this codebase, and that is a rule,
not a habit. Scattered env reads make it impossible to answer "what does this
service need to run?" without grepping, and they make tests depend on ambient
state.

`Settings` is constructed once at process start and injected. Deliberately not
a module-level singleton evaluated at import time — that pattern freezes the
environment before a test can change it.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AUTOPILOT_",
        extra="ignore",
        # Several settings legitimately start with `model_`. Without this,
        # pydantic warns about its own protected namespace on every import.
        protected_namespaces=(),
    )

    # --- Provider credentials -------------------------------------------------
    # SecretStr so a stray f-string or a logged settings dump prints
    # `**********` instead of the key. Never call .get_secret_value() outside
    # an adapter.
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    ollama_api_base: str = "http://localhost:11434"

    # --- Storage --------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./data/autopilot.db"
    data_dir: Path = Path("./data")
    artifacts_dir: Path = Path("./artifacts")

    # --- Configuration files --------------------------------------------------
    model_catalog_path: Path = Path("config/models.yaml")
    routing_policy_path: Path = Path("config/routing.yaml")

    # --- Routing and verification ---------------------------------------------
    verification_sample_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    low_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    escalation_latency_budget_ms: int = Field(default=4000, gt=0)

    # --- Observability --------------------------------------------------------
    log_prompt_text: bool = False
    log_level: str = "INFO"
    log_json: bool = True

    # --- Determinism ----------------------------------------------------------
    random_seed: int = 1337

    # --- Budget -----------------------------------------------------------
    # `None` means no cap is configured — the guard that reads this field is
    # simply never consulted (Phase 2's wiring decision, not this field's).
    daily_budget_usd: Decimal | None = None

    @field_validator("daily_budget_usd", mode="before")
    @classmethod
    def _budget_is_never_a_float(cls, v: object) -> object:
        """A float budget is a float in the money path, and pydantic would
        coerce it silently otherwise. Strings and Decimals only — this is the
        fourth Decimal-boundary door; the AST rule and the pricing.py
        behaviour test cannot reach a pydantic field coercion."""
        if isinstance(v, float):
            raise ValueError("daily_budget_usd must be a string or Decimal, never a float")
        return v

    @field_validator("verification_sample_rate")
    @classmethod
    def _sampling_must_stay_economic(cls, v: float) -> float:
        """Guard rail, not a preference.

        Verifying a request costs a reference completion on a stronger model
        plus a judge call. Push the rate high enough and the verification bill
        exceeds the savings it is measuring — the system pays to prove it is
        saving money while losing it. 0.5 is a generous ceiling; production
        values live near 0.1.
        """
        if v > 0.5:
            raise ValueError(
                f"verification_sample_rate={v} is not economic: verifying more than half "
                "of traffic costs more than routing everything to the baseline model"
            )
        return v

    @property
    def configured_providers(self) -> frozenset[str]:
        """Which providers this process can actually reach.

        The catalog uses this to refuse to route to a model whose provider has
        no credentials, at startup, instead of failing on the first request
        that happens to classify into that tier.
        """
        providers = {"ollama"}  # local, no credentials needed
        if self.openai_api_key is not None:
            providers.add("openai")
        if self.anthropic_api_key is not None:
            providers.add("anthropic")
        return frozenset(providers)


def load_settings(**overrides: object) -> Settings:
    """Build the settings object. Call once, at process start, and inject it."""
    return Settings(**overrides)  # type: ignore[arg-type]
