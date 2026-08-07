"""Closure over litellm's actual exception surface — by AST, not by import.

Same technique as `tests/unit/test_litellm_boundary.py`: read the vendor
source as text and `ast.parse` it, rather than importing the module, for two
reasons. It is fast, and it keeps this test free of a vendor import even
though its whole job is to police the vendor boundary of the mapping table.

This is the substitute D5 promised for not widening the coverage gate onto
`infrastructure/`, and it is strictly stronger than a percentage: a version
bump that adds a new litellm exception class fails this test by *name*,
rather than the class merely going untested.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from autopilot.infrastructure.error_translation import DELIBERATELY_UNMAPPED, ERROR_MAP


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _litellm_exception_class_names() -> frozenset[str]:
    """Class names defined in `litellm/exceptions.py`, read as source.

    Enum classes (`RateLimitErrorCategory`, `RateLimitType`) are excluded —
    they are not exceptions, and `translate()` is never called with one.
    """
    spec = importlib.util.find_spec("litellm.exceptions")
    assert spec is not None and spec.origin is not None, "litellm.exceptions not found"
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and not any(_base_name(b).endswith("Enum") for b in node.bases)
    )


def test_no_import_statement_names_litellm_in_this_file() -> None:
    """Guards the guard: this file's own job is to stay vendor-import-free
    while reading litellm's source. `importlib.util.find_spec` is a runtime
    call, not a static import statement, so `lint-imports`'s static graph is
    unaffected by it — but only as long as nothing here writes a real import
    of litellm. Checked by AST, not by substring search, so this test's own
    prose (which necessarily mentions the phrase) cannot trip itself up."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] == "litellm" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or node.module.split(".")[0] != "litellm"


def test_every_litellm_exception_is_mapped_or_deliberately_not() -> None:
    discovered = _litellm_exception_class_names()
    accounted = frozenset(ERROR_MAP) | frozenset(DELIBERATELY_UNMAPPED)

    unaccounted = discovered - accounted
    assert not unaccounted, (
        f"litellm defines exception classes nobody decided about: {sorted(unaccounted)}. "
        "Add each to ERROR_MAP or to DELIBERATELY_UNMAPPED with a reason."
    )

    stale = accounted - discovered
    assert not stale, f"the table names classes litellm no longer defines: {sorted(stale)}"
