"""Column-description helpers.

The `column_description` helper prompt turns raw column names into the concise
English descriptions that are injected into every generation prompt via the
`{columns_desc}` placeholder. (Panel loading itself lives in
`CogAlpha._load` in `search.py`.)
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from .llm_client import LLMClient
from .prompt_loader import _COLUMN_DESCRIPTION, _SYSTEM_MESSAGE

logger = logging.getLogger(__name__)


def describe_columns(
    df: pd.DataFrame,
    llm: LLMClient,
) -> str:
    """Turn column names into a `columns_desc` block.

    Uses the `column_description` helper prompt; on any failure, falls back to
    a plain list of the available columns.
    """
    factor_cols = [c for c in df.columns if c.lower() not in {"label", "y", "target", "ret_fwd"}]
    if not factor_cols:
        raise ValueError("The panel has no factor columns to describe.")

    if llm is None:
        return "\n".join(f"- {c}" for c in factor_cols)

    prompt = _COLUMN_DESCRIPTION.replace("{factor_names}", ", ".join(factor_cols))
    try:
        raw = llm.complete_quality(_SYSTEM_MESSAGE, prompt)
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
