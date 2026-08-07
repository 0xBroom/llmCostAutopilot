"""The Decimal chokepoint rule, enforced by AST — not by counting call sites.

The proposal's original rule ("`Decimal(...)` occurs at exactly one call site
in `src/`") is already false before this slice adds a single line:
`domain/models.py`'s `CostBreakdown.zero()` has called `Decimal(0)` since
Phase 0, and `Decimal(0)` is exact — an int literal loses no precision. "Exactly
one call site" was never the property that mattered. The property that matters
is *argument shape*: a `Decimal(...)` call is safe exactly when its argument is
a literal (str/int) or a `str(...)` call, and dangerous exactly when it is a
bare `Name`/`Attribute`/`Subscript`/`BinOp` or a float literal — because a
float already lost precision before `Decimal` ever sees it.

Three rules, each with its own test:

- **R1** — every `Decimal(...)` / `decimal.Decimal(...)` call in
  `src/autopilot/**` has exactly one positional argument, and that argument is
  a `str` literal, an `int` literal, or a call to `str(...)`. Never a `Name`,
  `Attribute`, `Subscript`, `BinOp`, or a float literal.
- **R2** — the set of enclosing function names where the argument is a
  `str(...)` call is exactly `{"to_price"}`, and every such site lives in
  `infrastructure/pricing.py`. This is the chokepoint claim: there is exactly
  one place in the codebase that ever converts an untrusted number into money.
- **R3** — no aliased import of `Decimal` (`from decimal import Decimal as D`)
  anywhere in `src/`. R1's detection is name-based; an alias would walk right
  past it.

This file is written before `infrastructure/pricing.py` exists. R1 already
passes against the pre-existing `Decimal(0)` — nothing here should regress
existing code. R2 fails until `to_price` exists, because the actual set of
str(...)-converting functions is empty, not `{"to_price"}`. That failure is
the point: the rule precedes the code it will govern.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def _autopilot_root() -> Path:
    spec = importlib.util.find_spec("autopilot")
    assert spec is not None and spec.submodule_search_locations, "autopilot package not found"
    return Path(next(iter(spec.submodule_search_locations)))


def _source_files() -> list[Path]:
    return sorted(_autopilot_root().rglob("*.py"))


def _decimal_call_marker(func: ast.expr) -> bool:
    """True if `func` is the callee of a `Decimal(...)` or `decimal.Decimal(...)` call."""
    if isinstance(func, ast.Name):
        return func.id == "Decimal"
    if isinstance(func, ast.Attribute):
        return func.attr == "Decimal"
    return False


class _DecimalCallVisitor(ast.NodeVisitor):
    """Walks one module, recording R1 violations and every `str(...)`-argument
    call site (name of the enclosing function, keyed by this file)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[str] = []
        self.str_arg_sites: set[str] = set()
        self._func_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if _decimal_call_marker(node.func):
            self._check_decimal_call(node)
        self.generic_visit(node)

    def _check_decimal_call(self, node: ast.Call) -> None:
        if len(node.args) != 1 or node.keywords:
            self.violations.append(
                f"{self.path}:{node.lineno}: Decimal(...) must take exactly one "
                "positional argument and no keywords"
            )
            return

        arg = node.args[0]

        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str | int)
            and not isinstance(arg.value, bool)
        ):
            return  # str/int literal — legal under R1

        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "str":
            enclosing = self._func_stack[-1] if self._func_stack else "<module>"
            self.str_arg_sites.add(enclosing)
            return

        self.violations.append(
            f"{self.path}:{node.lineno}: Decimal(...) argument must be a str/int "
            f"literal or a str(...) call, got {ast.dump(arg)}"
        )


def _walk_all() -> tuple[list[str], dict[Path, set[str]]]:
    violations: list[str] = []
    sites_by_file: dict[Path, set[str]] = {}
    for path in _source_files():
        visitor = _DecimalCallVisitor(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor.visit(tree)
        violations.extend(visitor.violations)
        if visitor.str_arg_sites:
            sites_by_file[path] = visitor.str_arg_sites
    return violations, sites_by_file


def test_every_decimal_call_takes_a_literal_or_a_str_conversion() -> None:
    """R1. Must already pass against the pre-existing `Decimal(0)` in
    `domain/models.py::CostBreakdown.zero()` — an int literal is exact."""
    violations, _ = _walk_all()
    assert not violations, "\n".join(violations)


def test_the_only_str_conversion_chokepoint_is_to_price_in_pricing_py() -> None:
    """R2. Fails until `infrastructure/pricing.py::to_price` exists."""
    _, sites_by_file = _walk_all()

    function_names = {name for names in sites_by_file.values() for name in names}
    assert function_names == {"to_price"}, (
        "expected the only function converting via Decimal(str(...)) to be "
        f"'to_price', found {sorted(function_names)}"
    )

    file_names = {path.name for path in sites_by_file}
    assert file_names == {"pricing.py"}, (
        f"expected the str(...) chokepoint only in pricing.py, found {sorted(file_names)}"
    )


def test_no_aliased_decimal_import() -> None:
    """R3. An aliased import would let a bad Decimal(...) call slip past R1's
    name-based detection under a different name."""
    violations: list[str] = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "decimal":
                for alias in node.names:
                    if alias.name == "Decimal" and alias.asname is not None:
                        violations.append(f"{path}: aliased Decimal import ({alias.asname})")
    assert not violations, "\n".join(violations)
