"""Fitness evaluation for alpha factors.

Labels are forward returns over the configured forecast horizon (paper:
10-day open returns). Per-factor IC / RankIC / ICIR / RankICIR / MI are
computed by `CogAlpha.evaluate_factor`; this module keeps the label builder
(`forward_returns`) and the NaN-ratio helper.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def forward_returns(df: pd.DataFrame, horizon: int, price_col: str = "open") -> pd.Series:
    """Forward h-day open-to-open return per ticker."""
    open_ = df[price_col]
    fwd = open_.groupby(level="ticker").shift(-horizon)
    return (fwd / open_ - 1.0).rename("ret_fwd")


def nan_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return 1.0
    return float(s.isna().mean())