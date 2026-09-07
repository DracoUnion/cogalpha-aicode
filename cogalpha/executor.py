"""Static code checks and execution of generated factor functions.

The prompts demand that generated factors are executable Python functions that
take a DataFrame (a single-stock time series, indexed by (date, ticker)) and
return a Series with the same name as the function. We enforce several hard
constraints statically with the AST (nested-loop ban, `while True` ban,
`def` presence) and then execute each function per ticker, merging the results
back into a panel-indexed Series.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import Factor, ParsedFunction

logger = logging.getLogger(__name__)

# Matches the `[function-N] ... [/function-N]` blocks the LLM emits.
_FUNC_BLOCK = re.compile(
    r"\[function\-\d+\]([\s\S]*?)\[/function-\d+\]", re.I
)
_DEF = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(df\)\s*:", re.M)
_DOCSTRING = re.compile(r"\"\"\"(.*?)\"\"\"", re.S)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_generated_code(raw: str) -> List[ParsedFunction]:
    """Extract `[function-N]` blocks from a generation response."""
    out: List[ParsedFunction] = []
    seen = set()
    for match in _FUNC_BLOCK.finditer(raw):
        block = match.group(1).strip()
        dm = _DEF.search(block)
        name = dm.group(1) if dm else _guess_name(block)
        if not name or name in seen:
            continue
        seen.add(name)
        doc = _extract_docstring(block)
        out.append(ParsedFunction(name=name, code=block, docstring=doc))
    return out


def _guess_name(block: str) -> str:
    m = re.search(r"return\s+df_copy\[['\"]([^'\"]+)['\"]\]", block)
    if m:
        return m.group(1)
    m = re.search(r"[A-Za-z_][A-Za-z0-9_]*\(", block)
    return m.group(0).rstrip("(") if m else "factor_unknown"


def _extract_docstring(code: str) -> str:
    m = _DOCSTRING.search(code)
    return m.group(1).strip() if m else ""


# --------------------------------------------------------------------------- #
# Static checks
# --------------------------------------------------------------------------- #
class StaticCheckError(Exception):
    pass


def check_code_static(code: str, name: str) -> List[str]:
    """Return a list of constraint violations (empty means OK).

    Mirrors the "Hard Complexity Constraints" in the prompts.
    """
    issues: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]

    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        issues.append("No `def` function found.")

    for node in ast.walk(tree):
        # Nested-loop detection: any loop whose body contains another loop.
        if isinstance(node, (ast.For, ast.While)):
            child_loops = _count_loops(node)
            if child_loops > 0:
                issues.append(
                    f"Nested loop detected at line {node.lineno} (forbidden)."
                )
        # Infinite-loop detection.
        if isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and bool(node.test.value):
                issues.append(f"`while True` (infinite loop) at line {node.lineno} (forbidden).")
    return issues


def _count_loops(node: ast.AST) -> int:
    count = 0
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.For, ast.While)):
            count += 1
            count += _count_loops(child)
    return count


def validate_name(code: str, name: str) -> List[str]:
    """Ensure the returned Series is named exactly like the function."""
    issues: List[str] = []
    # The function must assign/return a Series with the same name somewhere.
    if f"'{name}'" not in code and f'"{name}"' not in code:
        issues.append(f"Returned Series name '{name}' not referenced in the function.")
    return issues


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def compile_factor(code: str):
    """Compile a factor function into a callable."""
    namespace: dict = {
        "__name__": "<factor>",
        "np": np,
        "pd": pd,
        "talib": _safe_talib(),
        "stats": _safe_stats(),
        "math": __import__("math"),
    }
    exec(compile(code, "<factor>", "exec"), namespace)
    candidates = [
        v for v in namespace.values()
        if callable(v) and getattr(v, "__module__", None) == "<factor>"
    ]
    if not candidates:
        raise ValueError("No callable factor function found in the code.")
    # Prefer a function whose name matches a column we reference; take the first
    # defined function otherwise.
    return candidates[0]


def apply_factor(df: pd.DataFrame, func, factor_name: str) -> pd.Series:
    """Apply a factor function per ticker and return a panel-indexed Series."""
    results: List[pd.Series] = []
    for ticker, sub in df.groupby(level="ticker", sort=False):
        try:
            s = func(sub.copy())
            if isinstance(s, pd.DataFrame):
                # Some models return a frame; take the matching column.
                s = s[factor_name] if factor_name in s.columns else s.iloc[:, 0]
            s = pd.Series(s)
            s.index = sub.index
            results.append(s)
        except Exception as exc:  # pragma: no cover - runtime errors are expected
            raise RuntimeError(f"ticker={ticker}: {exc}") from exc

    if not results:
        raise ValueError("No results produced.")
    out = pd.concat(results).sort_index()
    out.name = factor_name
    return out


def _safe_talib():
    try:
        import talib  # type: ignore
        return talib
    except Exception:
        return None


def _safe_stats():
    try:
        from scipy import stats  # noqa: F401
        return stats
    except Exception:
        return None
