"""Configuration is typed, overridable and refuses to hide secrets in logs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autopilot.config.settings import Settings


def _settings(**kwargs: object) -> Settings:
    """Build settings with no `.env` file, so a developer's local file cannot
    change the outcome of a test.

    `_env_file` is a pydantic-settings runtime keyword that its generated
    `__init__` signature does not advertise, hence the ignore.
    """
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg,arg-type]


def test_defaults_are_usable_without_any_environment() -> None:
    s = _settings()
    assert s.verification_sample_rate == 0.15
    assert s.log_prompt_text is False
    assert s.model_catalog_path == Path("config/models.yaml")


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOPILOT_VERIFICATION_SAMPLE_RATE", "0.42")
    monkeypatch.setenv("AUTOPILOT_LOG_PROMPT_TEXT", "true")
    monkeypatch.setenv("AUTOPILOT_MODEL_CATALOG_PATH", "/etc/models.yaml")

    s = _settings()

    assert s.verification_sample_rate == 0.42
    assert s.log_prompt_text is True
    assert s.model_catalog_path == Path("/etc/models.yaml")


def test_the_prefix_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unprefixed `LOG_LEVEL` in the shell must not reconfigure this service."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert _settings().log_level == "INFO"


def test_secrets_do_not_print(monkeypatch: pytest.MonkeyPatch) -> None:
    """The most common way an API key reaches a log file is a settings dump in
    an exception handler. `SecretStr` makes that impossible by default."""
    monkeypatch.setenv("AUTOPILOT_OPENAI_API_KEY", "sk-do-not-leak-me")

    s = _settings()

    assert "do-not-leak-me" not in repr(s)
    assert "do-not-leak-me" not in str(s)
    assert "do-not-leak-me" not in str(s.model_dump())
    assert s.openai_api_key is not None
    assert s.openai_api_key.get_secret_value() == "sk-do-not-leak-me"


def test_sample_rate_above_half_is_rejected() -> None:
    """Not a style preference. Verifying more than half of traffic costs a
    reference completion plus a judge call on every other request, which
    exceeds the savings being measured. The config refuses to be set into a
    regime where the system loses money proving it saves money.
    """
    with pytest.raises(ValidationError, match="not economic"):
        _settings(verification_sample_rate=0.8)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_sample_rate_must_be_a_fraction(bad: float) -> None:
    with pytest.raises(ValidationError):
        _settings(verification_sample_rate=bad)


def test_ollama_is_always_available_and_paid_providers_are_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catalog uses this to refuse an unroutable configuration at startup
    rather than on the first request that classifies into that tier."""
    assert _settings().configured_providers == frozenset({"ollama"})

    monkeypatch.setenv("AUTOPILOT_ANTHROPIC_API_KEY", "sk-ant-test")
    assert _settings().configured_providers == frozenset({"ollama", "anthropic"})


def test_unknown_variables_are_ignored_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docker and CI inject plenty of unrelated env vars. Crashing on them
    would make the service undeployable."""
    monkeypatch.setenv("AUTOPILOT_SOMETHING_WE_REMOVED_LAST_MONTH", "1")
    assert _settings().log_level == "INFO"
