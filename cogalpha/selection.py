"""Pool ranking.

`rank_factors` orders a pool by a fitness metric (IC by default).
Qualification / elite classification is handled by `CogAlpha.classify_factor`.
"""

from __future__ import annotations

import logging
from typing import List

from .models import Factor

logger = logging.getLogger(__name__)


def rank_factors(factors: List[Factor], key: str = "ic") -> List[Factor]:
    return sorted(factors, key=lambda f: _safe(f, key), reverse=True)


def _safe(f: Factor, key: str) -> float:
    v = getattr(f, key, None)
    return float(v) if v is not None else -1e9