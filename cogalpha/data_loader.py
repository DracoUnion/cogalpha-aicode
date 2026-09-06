"""Data loading and column-description helpers.

CogAlpha expects a daily OHLCV panel with a (date, ticker) MultiIndex. The
`column_description` helper prompt turns raw column names into the concise
English descriptions that are injected into every generation prompt via the
`{columns_desc}` placeholder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .llm_client import LLMClient
from .prompt_loader import PromptLibrary

logger = logging.getLogger(__name__)


def load_panel(path: str) -> pd.DataFrame:
    """Load an OHLCV panel from parquet or CSV.

    Expected index: MultiIndex (date, ticker). Columns typically include
    open/high/low/close/volume. If the file is a CSV, the first two columns are
    assumed to be date and ticker.
    """
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, parse_dates=False)
        if isinstance(df.index, pd.MultiIndex):
            df = df
        elif not isinstance(df.index, pd.MultiIndex):
            # Assume first two columns are date, ticker.
            cols = list(df.columns)
            df = df.set_index([cols[0], cols[1]])
            df.index = df.index.set_names(["date", "ticker"])

    if not isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        if "date" in df.columns and "ticker" in df.columns:
            df = df.set_index(["date", "ticker"]).sort_index()
        else:
            raise ValueError("Panel must have a (date, ticker) MultiIndex, or date+ticker columns.")

    # Normalize the date level to a datetime index.
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values("date")):
        df.index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(df.index.get_level_values("date")), df.index.get_level_values("ticker")],
            names=["date", "ticker"],
        )
    return df.sort_index()


def describe_columns(
    df: pd.DataFrame,
    llm: LLMClient,
    lib: PromptLibrary,
) -> str:
    """Turn column names into a `columns_desc` block.

    Uses the `column_description` helper prompt; on any failure, falls back to
    a plain list of the available columns.
    """
    factor_cols = [c for c in df.columns if c.lower() not in {"label", "y", "target", "ret_fwd"}]
    if not factor_cols:
        raise ValueError("The panel has no factor columns to describe.")

    if llm is None or lib is None:
        return "\n".join(f"- {c}" for c in factor_cols)

    prompt = lib.column_description.replace(
        "{', '.join(factor_names)}", "', '".join(factor_cols)
    )
    try:
        raw = llm.complete_quality(lib.system_message, prompt)
    except Exception as exc:  # pragma: no cover - network/API dependent
        logger.warning("column_description LLM call failed (%s); using plain names", exc)
        return "\n".join(f"- {c}" for c in factor_cols)

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            col, desc = line.split(":", 1)
            lines.append(f"- `{col.strip()}`: {desc.strip()}")
        else:
            lines.append(f"- {line}")
    return "\n".join(lines) or "\n".join(f"- {c}" for c in factor_cols)


def build_column_desc_manual(columns: List[str]) -> str:
    """A deterministic fallback description block (no LLM required)."""
    return "\n".join(f"- `{c}`: raw daily {c} feature" for c in columns)
