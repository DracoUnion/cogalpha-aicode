"""Adaptive generation feedback.

After each fitness evaluation the framework distills the best effective and
worst ineffective factors into CoT-style summaries (via the
effective/ineffective summary prompts) that are injected into the next
generation's generation prompts through the `effective_factor_analysis` /
`ineffective_factor_analysis` blocks.
"""

from __future__ import annotations

import logging
import random
import textwrap
from typing import List

from .llm import LLMClient
from .prompts import _EFFECTIVE_SUMMARY, _INEFFECTIVE_SUMMARY, _SYSTEM_MESSAGE
from .models import Factor, FeedbackSummary

logger = logging.getLogger(__name__)


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