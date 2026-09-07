"""Manual (no-LLM) column-description helper.

`build_column_desc_manual` builds a deterministic ``columns_desc`` block from
the raw column names; the LLM-driven variant lives in `agent.Agent.describe_columns`.
"""

from __future__ import annotations

from typing import List


def build_column_desc_manual(columns: List[str]) -> str:
    """A deterministic fallback description block (no LLM required)."""
    return "\n".join(f"- `{c}`: raw daily {c} feature" for c in columns)