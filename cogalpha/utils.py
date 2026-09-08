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
import re
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .models import Factor, FeedbackSummary, ParsedFunction
from .prompts import _EFFECTIVE_SUMMARY, _INEFFECTIVE_SUMMARY, _SYSTEM_MESSAGE

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prompt / parsing helpers (moved from agent)
# --------------------------------------------------------------------------- #
MAX_REPAIR_ATTEMPTS_DEFAULT = 3
_FUNC_BLOCK = re.compile(r"\[function\-\d+\]([\s\S]*?)\[/function-\d+\]", re.I)
_DEF = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(df\)\s*:", re.M)
_DOCSTRING = re.compile(r"\"\"\"(.*?)\"\"\"", re.S)


def render_placeholders(prompt: str, **kw) -> str:
    """Substitute `{placeholder}` tokens; unknown tokens remain unchanged."""
    def _repl(match: re.Match) -> str:
        name = match.group(1)
        if name in kw:
            value = kw[name]
            return "" if value is None else str(value)
        return match.group(0)

    return re.sub(r"{(\w+)}", _repl, prompt)


def render_cot_block(template: str, cot: str) -> str:
    """Render an optional CoT analysis block, or an empty string."""
    if not cot:
        return ""
    return template.replace("{effective_CoT}", cot).replace("{ineffective_CoT}", cot)


def quality_issues(raw: str) -> List[str]:
    """Extract the bulleted issue list from a code-quality review."""
    issues = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("-") or stripped.startswith("*") or stripped.startswith("1."):
            issues.append(stripped.lstrip("-*0123456789. ").strip())
            if len(issues) >= 8:
                break
    return issues


def guess_name(block: str) -> str:
    match = re.search(r"return\s+df_copy\[['\"]([^'\"]+)['\"]\]", block)
    if match:
        return match.group(1)
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*\(", block)
    return match.group(0).rstrip("(") if match else "factor_unknown"


def extract_docstring(code: str) -> str:
    match = _DOCSTRING.search(code)
    return match.group(1).strip() if match else ""


def parse_generated_code(raw: str) -> List[ParsedFunction]:
    """Extract `[function-N]` blocks from a generation response."""
    out: List[ParsedFunction] = []
    seen = set()
    for match in _FUNC_BLOCK.finditer(raw):
        block = match.group(1).strip()
        definition = _DEF.search(block)
        name = definition.group(1) if definition else guess_name(block)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(ParsedFunction(name=name, code=block, docstring=extract_docstring(block)))
    return out


def quality_corrected(raw: str) -> str:
    functions = parse_generated_code(raw)
    return functions[0].code if functions else ""


# --------------------------------------------------------------------------- #
# Evaluation / classification helpers (moved from main)
# --------------------------------------------------------------------------- #
def daily_corr(factor: pd.Series, label: pd.Series, method: str) -> np.ndarray:
    dates = factor.index.get_level_values("date")
    out = []
    for date in pd.unique(dates):
        mask = factor.index.get_level_values("date") == date
        x = factor[mask].to_numpy(dtype=float)
        y = label[mask].to_numpy(dtype=float)
        if len(x) < 5:
            continue
        if method == "pearson":
            correlation = np.corrcoef(x, y)[0, 1]
        else:
            with np.errstate(invalid="ignore"):
                correlation = stats.spearmanr(x, y).statistic
        if np.isfinite(correlation):
            out.append(correlation)
    return np.asarray(out, dtype=float)


def mutual_info(contingency: np.ndarray) -> float:
    try:
        from sklearn.metrics import mutual_info_score

        return float(mutual_info_score(None, None, contingency=contingency))
    except Exception:  # pragma: no cover - sklearn optional
        cont = contingency.astype(float) + 1e-12
        total = cont.sum()
        pxy = cont / total
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        value = (pxy * np.log(pxy / (px * py + 1e-12))).sum()
        return float(max(0.0, value))


def normalized_mi(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    """Normalized mutual information in [0, 1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    size = min(a.size, b.size)
    a, b = a[:size], b[:size]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 20 or a.std() == 0 or b.std() == 0:
        return 0.0
    contingency, _, _ = np.histogram2d(a, b, bins=bins)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0
    value = mutual_info(contingency)
    entropy_a = stats.entropy(contingency.sum(axis=1))
    entropy_b = stats.entropy(contingency.sum(axis=0))
    denominator = np.sqrt(entropy_a * entropy_b)
    return float(value / denominator) if denominator > 0 else 0.0


def metric_thresholds(eval_cfg) -> Dict[str, Tuple[float, float]]:
    return {
        "ic": eval_cfg.ic,
        "rank_ic": eval_cfg.rank_ic,
        "icir": eval_cfg.icir,
        "rank_icir": eval_cfg.rank_icir,
        "mi": eval_cfg.mi,
    }


def threshold_ok(value: float, threshold: float) -> bool:
    return threshold is None or value >= threshold

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


def render_prompt(prompt: str, **kw):
    return re.sub(r"{(\w+)}", lambda g: kw.get(g.group(1), g.group(0)), prompt)