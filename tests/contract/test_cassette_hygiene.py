"""Cassettes are recorded against real providers with real credentials.

A key that reaches a committed cassette is in the reflog forever; `git rm` does
not remove it and rotating it is the only fix. Scrubbing is configured at
record time in `conftest.vcr_config`, and this test is the check that the
configuration actually worked.

It passes vacuously when there are no cassettes yet. That is intentional: the
guard must exist *before* the first recording, not after.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CASSETTE_DIR = Path(__file__).parent / "cassettes"

# Shapes of real credentials. Deliberately not a bare "sk-" — the scrubbed
# placeholder "REDACTED" and prose mentioning key formats must not trip this.
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "OpenAI key": re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Bearer token": re.compile(r"[Bb]earer\s+[A-Za-z0-9._\-]{20,}"),
    "Set-Cookie": re.compile(r"(?i)set-cookie:\s*\S+"),
}

REQUIRED_SCRUBBED_HEADERS = ("authorization", "api-key", "x-api-key")


def _cassettes() -> list[Path]:
    if not CASSETTE_DIR.exists():
        return []
    return sorted(CASSETTE_DIR.rglob("*.yaml")) + sorted(CASSETTE_DIR.rglob("*.yml"))


def test_the_cassette_directory_is_discoverable() -> None:
    """A typo in the path would make every check below pass on zero files."""
    assert CASSETTE_DIR.parent.exists()


@pytest.mark.parametrize("cassette", _cassettes(), ids=lambda p: p.name)
def test_no_credentials_in_committed_cassettes(cassette: Path) -> None:
    content = cassette.read_text(encoding="utf-8", errors="replace")
    leaks = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(content)]
    assert not leaks, (
        f"{cassette.name} contains what looks like a live credential ({', '.join(leaks)}). "
        f"Do not just edit the file — the key is already in the git history. Rotate it, "
        f"then re-record with the scrubbing in tests/conftest.py::vcr_config."
    )


@pytest.mark.parametrize("cassette", _cassettes(), ids=lambda p: p.name)
def test_auth_headers_are_redacted_not_absent(cassette: Path) -> None:
    """A cassette with the auth header stripped entirely still replays, so its
    absence proves nothing about whether the scrubber ran. A cassette with
    `REDACTED` proves it did."""
    content = cassette.read_text(encoding="utf-8", errors="replace").lower()
    for header in REQUIRED_SCRUBBED_HEADERS:
        if header in content:
            window_start = content.index(header)
            window = content[window_start : window_start + 200]
            assert "redacted" in window, (
                f"{cassette.name} carries a {header!r} header that was not redacted"
            )
