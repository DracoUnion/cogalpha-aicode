"""Turn a raw parsed function into a validated, evaluated Factor.

This is the bridge between the quality checker and the evaluator: compile ->
execute on the panel -> NaN filter -> fitness (IC/RankIC/ICIR/RankICIR/MI) ->
qualified/elite classification.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from . import executor, evaluator, selection
from .config import CogAlphaConfig
from .quality import QualityGate
from .schemas import Factor, ParsedFunction

logger = logging.getLogger(__name__)


def compute_label(data: pd.DataFrame, horizon: int, price_col: str = "open") -> pd.Series:
    return evaluator.forward_returns(data, horizon, price_col)


def produce_factor(
    pf: ParsedFunction,
    gate: QualityGate,
    data: pd.DataFrame,
    label: pd.Series,
    cfg: CogAlphaConfig,
    *,
    theme: str = "",
    level: str = "",
    agent_id: str = "",
    generation: int = 0,
    source: str = "generated",
) -> Optional[Factor]:
    """Run the quality gate, execute, and evaluate a parsed function."""
    factor = Factor(
        name=pf.name,
        code=pf.code,
        docstring=pf.docstring,
        theme=theme,
        level=level,
        agent_id=agent_id,
        generation=generation,
        source=source,
    )

    # 1) Quality checker (repair + judge + logic improvement).
    final_code = _run_gate(gate, factor)
    if final_code is None:
        return None
    factor.code = final_code

    # 2) Execute.
    series = _execute(factor, data)
    if series is None:
        factor.executable = False
        factor.accepted_by_judge = False
        return None
    factor.executable = True
    factor.nan_ratio = evaluator.nan_ratio(series)
    if factor.nan_ratio > cfg.generation.max_nan_ratio:
        logger.debug("Factor %s NaN ratio %.2f too high; dropping.", factor.name, factor.nan_ratio)
        return None

    # 3) Evaluate.
    metrics = evaluator.evaluate_factor(series, label)
    factor.set_metrics(metrics)
    factor = selection.classify_factor(factor, cfg.evaluation)
    return factor


def _run_gate(gate: QualityGate, factor: Factor) -> Optional[str]:
    # Delayed import to avoid a circular import at module load.
    from .quality import gate_factor

    return gate_factor(gate, factor.code, factor.name)


def _execute(factor: Factor, data: pd.DataFrame) -> Optional[pd.Series]:
    try:
        func = executor.compile_factor(factor.code)
    except Exception as exc:  # pragma: no cover - compile errors
        logger.info("Factor %s failed to compile: %s", factor.name, exc)
        return None
    try:
        return executor.apply_factor(data, func, factor.name)
    except Exception as exc:
        logger.info("Factor %s failed to execute: %s", factor.name, exc)
        return None