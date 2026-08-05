"""The vendor boundary, checked as source rather than as behaviour.

`.importlinter` enforces *who* may import `litellm`. It cannot enforce *where
in the file* the import sits — and for this particular dependency the ordering
is the guarantee. So that part is checked here.

Nothing in this module imports `litellm`; it reads the file. Importing it would
cost seconds of the unit suite for no additional confidence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = "autopilot.infrastructure.litellm_env"


def _source() -> str:
    spec = importlib.util.find_spec(MODULE)
    assert spec is not None and spec.origin is not None, f"{MODULE} not found"
    return Path(spec.origin).read_text(encoding="utf-8")


def test_the_cost_map_is_pinned_before_litellm_is_imported() -> None:
    """`LITELLM_LOCAL_MODEL_COST_MAP` is read once, when `litellm` is first
    imported. Set it afterwards and it does nothing — silently. The savings
    report then depends on whether the process had internet at boot, which is
    not a property a reproducible cost figure can have.
    """
    source = _source()

    env_var = source.index("LITELLM_LOCAL_MODEL_COST_MAP")
    import_stmt = source.index("import litellm")

    assert env_var < import_stmt, (
        "the environment variable must be set BEFORE `import litellm`, "
        "otherwise the local price map is not pinned and costs stop being reproducible"
    )


def test_the_shim_uses_setdefault_so_an_operator_can_override() -> None:
    """Pinned by default, not welded shut. Someone debugging a stale price
    needs to be able to unpin it from the environment."""
    assert "setdefault" in _source()
