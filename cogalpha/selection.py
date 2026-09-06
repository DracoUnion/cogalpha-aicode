"""Pool and selection management.

Implements the population bookkeeping from the paper: an initial pool, a parent
pool, a child pool, a candidate pool (qualified alphas), and an elite pool.
Realized fitness decides qualification (a factor that clears the qualified
thresholds is a candidate; clearing the elite thresholds makes it elite).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .config import EvaluationSettings
from .schemas import Factor

logger = logging.getLogger(__name__)


def _have(eval_cfg: EvaluationSettings) -> Dict[str, Tuple[float, float]]:
    return {
        "ic": eval_cfg.ic,
        "rank_ic": eval_cfg.rank_ic,
        "icir": eval_cfg.icir,
        "rank_icir": eval_cfg.rank_icir,
        "mi": eval_cfg.mi,
    }


def classify_factor(f: Factor, eval_cfg: EvaluationSettings) -> Factor:
    """Set `qualified`/`elite` based on the configured thresholds."""
    thresholds = _have(eval_cfg)
    qual_pass = 0
    elite_pass = 0
    total = 0

    for metric, (q_th, e_th) in thresholds.items():
        v = getattr(f, metric, None)
        if v is None:
            # A missing metric fails to qualify.
            qual_pass -= 100
            continue
        total += 1
        if _ok(v, q_th):
            qual_pass += 1
            if _ok(v, e_th):
                elite_pass += 1

    if total == 0:
        f.qualified = False
        f.elite = False
    else:
        # All present metrics must clear their thresholds.
        f.qualified = qual_pass == total
        f.elite = elite_pass == total
    return f


def _ok(v: float, th: float) -> bool:
    """Thresholds are interpreted as directional cutoffs (positive = good)."""
    if th is None:
        return True
    return v >= th


def rank_factors(factors: List[Factor], key: str = "ic") -> List[Factor]:
    return sorted(factors, key=lambda f: _safe(f, key), reverse=True)


def _safe(f: Factor, key: str) -> float:
    v = getattr(f, key, None)
    return float(v) if v is not None else -1e9