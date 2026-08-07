"""The credential boundary, checked as source rather than as behaviour.

`.get_secret_value()` unwraps a `SecretStr` into a plain string that can end
up in a log line, an f-string or a `litellm_params` dict — the failure mode
is silent, not a crash, which is exactly why a review can miss it creeping
into a second module. Same source-scan technique as
`tests/unit/test_litellm_boundary.py` and `tests/unit/test_decimal_boundary.py`:
read files as text, never import the vendor package to police the boundary
around it.

Nothing in this module imports `litellm`.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from autopilot.domain.models import KNOWN_PROVIDERS
from autopilot.infrastructure.router_factory import _KEY_ACCESSORS

_ALLOWED_ENV_READERS: frozenset[str] = frozenset({"settings.py", "litellm_env.py"})
"""`settings.py` is the one module pydantic-settings reads the environment
for. `litellm_env.py` is allow-listed for a single, already-policed reason:
setting `LITELLM_LOCAL_MODEL_COST_MAP` before the vendor import is the one
legitimate env access outside `settings.py`, and
`tests/unit/test_litellm_boundary.py` already asserts its ordering."""


def _autopilot_root() -> Path:
    spec = importlib.util.find_spec("autopilot")
    assert spec is not None and spec.submodule_search_locations, "autopilot package not found"
    return Path(next(iter(spec.submodule_search_locations)))


def _source_files() -> list[Path]:
    return sorted(_autopilot_root().rglob("*.py"))


def _calls_get_secret_value(path: Path) -> bool:
    """AST, not substring. `config/settings.py` carries a comment reading
    'Never call .get_secret_value() outside an adapter' right next to the
    `SecretStr` fields it is warning about — a plain substring search trips
    on its own warning. Comments are not part of the AST, so a real `Call`
    node is what this test needs to find."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_secret_value"
        for node in ast.walk(tree)
    )


def test_get_secret_value_is_called_in_exactly_one_module() -> None:
    """`router_factory.py` is the only place `src/` unwraps a `SecretStr` —
    the structural payoff of D3: the adapter that makes network calls
    (`LiteLLMGateway`, slice 7) never sees a secret at all."""
    users = {path.name for path in _source_files() if _calls_get_secret_value(path)}
    assert users == {"router_factory.py"}


def _reads_environment(path: Path) -> bool:
    """AST, not substring — same reasoning as `_calls_get_secret_value` above.

    A substring search for the literal text `"os.getenv("` is defeated by
    anything that doesn't spell the call that way: `from os import getenv`
    then a bare `getenv(...)`, `import os as o` then `o.environ[...]`, or
    `from os import environ as e` then `e.get(...)`. It is also defeated in
    the other direction — a docstring that merely *mentions* `os.getenv()`
    (see `config/settings.py`'s own module docstring) trips a substring
    search without a single line of code ever running. Tracking the local
    names bound to `os`, `os.environ` and `os.getenv` — however they were
    imported or aliased — and matching real `Call`/`Subscript` nodes against
    those names is what closes both gaps at once.

    Catches: `os.getenv(...)`, `os.environ[...]` (read or write, same AST
    node either way), `os.environ.get(...)`, and each of those reached via a
    rename (`from os import getenv`, `import os as o`,
    `from os import environ as e`, and any `as` alias on top of those).

    Deliberately does not catch `os.environ.setdefault(...)` — the one form
    `litellm_env.py` (itself allow-listed below) uses, and not one of the
    three forms the rule was ever written to name.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    os_names: set[str] = set()
    environ_names: set[str] = set()
    getenv_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "os"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name == "environ":
                    environ_names.add(alias.asname or alias.name)
                elif alias.name == "getenv":
                    getenv_names.add(alias.asname or alias.name)

    def is_os(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Name) and expr.id in os_names

    def is_environ(expr: ast.expr) -> bool:
        if isinstance(expr, ast.Attribute) and expr.attr == "environ" and is_os(expr.value):
            return True
        return isinstance(expr, ast.Name) and expr.id in environ_names

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "getenv" and is_os(func.value):
                return True
            if isinstance(func, ast.Name) and func.id in getenv_names:
                return True
            if isinstance(func, ast.Attribute) and func.attr == "get" and is_environ(func.value):
                return True
        if isinstance(node, ast.Subscript) and is_environ(node.value):
            return True
    return False


def test_no_environment_reads_outside_settings_and_the_litellm_shim() -> None:
    """`os.getenv`, `os.environ[...]`, `os.environ.get`. Closes the gap
    `sdd-init` flagged as stated-but-never-verified: the rule that there is
    no `os.getenv()` outside `config/settings.py` was documented, not
    checked, until this test — and AST-checked, not substring-checked, since
    verification found this test was the one guard in this file still using
    a substring search next to its AST-based sibling above."""
    offenders = [
        path.name
        for path in _source_files()
        if path.name not in _ALLOWED_ENV_READERS and _reads_environment(path)
    ]
    assert not offenders, f"environment read outside the sanctioned modules: {offenders}"


def test_every_known_provider_has_a_credential_accessor() -> None:
    """Adding a provider to the domain without teaching `router_factory` how
    to fetch its credential would otherwise surface as a confusing failure
    at the first request instead of at startup — the same class of hole
    `MissingCredentialsError` exists to close for a *missing* credential."""
    assert set(_KEY_ACCESSORS) == KNOWN_PROVIDERS
