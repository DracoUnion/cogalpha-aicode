"""Shared utility helpers for CogAlpha.

Consolidated from the former data_loader / evaluator / executor / feedback /
selection modules:

  - build_column_desc_manual      (was data_loader)
  - forward_returns, nan_ratio    (was evaluator)
  - StaticCheckError, check_code_static, validate_name,
    compile_factor, apply_factor  (was executor)
  - build_feedback, deterministic_build_feedback, _render_examples  (was feedback)
  - rank_factors, _safe           (was selection)
"""

from __future__ import annotations

import ast
import logging
import random
import textwrap
from typing import List

import numpy as np
import pandas as pd

from .llm import LLMClient
from .models import Factor, FeedbackSummary
from .prompts import _EFFECTIVE_SUMMARY, _INEFFECTIVE_SUMMARY, _SYSTEM_MESSAGE

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Column descriptions (was data_loader)
# --------------------------------------------------------------------------- #
def build_column_desc_manual(columns: List[str]) -> str:
    """A deterministic fallback description block (no LLM required)."""
    return "\n".join(f"- `{c}`: raw daily {c} feature" for c in columns)


# --------------------------------------------------------------------------- #
# Labels / NaN ratio (was evaluator)
# --------------------------------------------------------------------------- #
def forward_returns(df: pd.DataFrame, horizon: int, price_col: str = "open") -> pd.Series:
    """Forward h-day open-to-open return per ticker."""
    open_ = df[price_col]
    fwd = open_.groupby(level="ticker").shift(-horizon)
    return (fwd / open_ - 1.0).rename("ret_fwd")


def nan_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return 1.0
    return float(s.isna().mean())


# --------------------------------------------------------------------------- #
# Static code checks + execution (was executor)
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


# --------------------------------------------------------------------------- #
# Adaptive feedback (was feedback)
# --------------------------------------------------------------------------- #
def _render_examples(factors: List[Factor]) -> str:
    lines = []
    for i, f in enumerate(factors, 1):
        lines.append(f"[factor-{i}]")
        lines.append("State: valid")
        lines.append(f"Metrics: IC / RankIC / ICIR / RankICIR")
        lines.append("Code:")
        lines.append(f"[function-{i}]")
        lines.append(textwrap.indent(f.code, "    ") if f.code else "")
        lines.append(f"[/function-{i}]")
        lines.append(f"[/factor-{i}]")
    return "\n".join(lines)


def build_feedback(
    effective: List[Factor],
    ineffective: List[Factor],
    llm: LLMClient,
) -> FeedbackSummary:
    """Produce effective/ineffective CoT summaries from sample factors."""
    effective_CoT, ineffective_CoT = "", ""

    if effective:
        size = min(len(effective), 6)
        sample = random.sample(effective, size)
        names = ", ".join(f.name for f in sample)
        user = _EFFECTIVE_SUMMARY.replace("{factor_names}", names).replace(
            "{factor_examples}", _render_examples(sample)
        )
        try:
            effective_CoT = llm.complete_quality(_SYSTEM_MESSAGE, user)
        except Exception as exc:  # pragma: no cover - network/API dependent
            logger.warning("effective summary failed: %s", exc)
            effective_CoT = "; ".join(f.name for f in sample)

    if ineffective:
        size = min(len(ineffective), 8)
        sample = random.sample(ineffective, size)
        names = ", ".join(f.name for f in sample)
        user = _INEFFECTIVE_SUMMARY.replace("{factor_names}", names).replace(
            "{factor_examples}", _render_examples(sample)
        )
        try:
            ineffective_CoT = llm.complete_quality(_SYSTEM_MESSAGE, user)
        except Exception as exc:  # pragma: no cover - network/API dependent
            logger.warning("ineffective summary failed: %s", exc)
            ineffective_CoT = "; ".join(f.name for f in sample)

    effective_CoT = effective_CoT[:2000]
    ineffective_CoT = ineffective_CoT[:2000]
    return FeedbackSummary(effective=effective_CoT, ineffective=ineffective_CoT)


def deterministic_build_feedback(
    effective: List[Factor], ineffective: List[Factor]
) -> FeedbackSummary:
    """A no-LLM fallback that still wires feedback into generation prompts."""
    eff = "\n".join(
        f"{f.name}: IC={f.ic:.4f}, window/theme={f.theme}; code={f.code[:120]}"
        for f in effective[:2]
    )
    ineff = "\n".join(
        f"{f.name}: IC={f.ic if f.ic is not None else float('nan'):.4f} theme={f.theme}"
        for f in ineffective[:2]
    )
    return FeedbackSummary(effective=eff, ineffective=ineff)


# --------------------------------------------------------------------------- #
# Pool ranking (was selection)
# --------------------------------------------------------------------------- #
def rank_factors(factors: List[Factor], key: str = "ic") -> List[Factor]:
    return sorted(factors, key=lambda f: _safe(f, key), reverse=True)


def _safe(f: Factor, key: str) -> float:
    v = getattr(f, key, None)
    return float(v) if v is not None else -1e9