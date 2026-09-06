"""Fitness evaluation for alpha factors.

Implements the five metrics described in the paper: IC, RankIC, ICIR, RankICIR
(linear relationships) and MI (mutual information, non-linear). Labels are
forward returns over the configured forecast horizon (paper: 10-day open
returns). Cross-sectional correlations are averaged across dates.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def forward_returns(df: pd.DataFrame, horizon: int, price_col: str = "open") -> pd.Series:
    """Forward h-day open-to-open return per ticker."""
    open_ = df[price_col]
    fwd = open_.groupby(level="ticker").shift(-horizon)
    return (fwd / open_ - 1.0).rename("ret_fwd")


def _daily_corr(factor: pd.Series, label: pd.Series, method: str) -> np.ndarray:
    dates = factor.index.get_level_values("date")
    out = []
    for d in pd.unique(dates):
        mask = factor.index.get_level_values("date") == d
        x = factor[mask].to_numpy(dtype=float)
        y = label[mask].to_numpy(dtype=float)
        if len(x) < 5:
            continue
        if method == "pearson":
            r = np.corrcoef(x, y)[0, 1]
        else:
            with np.errstate(invalid="ignore"):
                r = stats.spearmanr(x, y).statistic
        if np.isfinite(r):
            out.append(r)
    return np.asarray(out, dtype=float)


def _mi(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    """Normalized mutual information in [0, 1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 20 or a.std() == 0 or b.std() == 0:
        return 0.0
    cxy, _, _ = np.histogram2d(a, b, bins=bins)
    if cxy.shape[0] < 2 or cxy.shape[1] < 2:
        return 0.0
    mi = _mutual_info(cxy)
    hx = stats.entropy(cxy.sum(axis=1))
    hy = stats.entropy(cxy.sum(axis=0))
    denom = np.sqrt(hx * hy)
    return float(mi / denom) if denom > 0 else 0.0


def _mutual_info(contingency: np.ndarray) -> float:
    try:
        from sklearn.metrics import mutual_info_score

        return float(mutual_info_score(None, None, contingency=contingency))
    except Exception:  # pragma: no cover - sklearn optional
        # Fallback: manual plug-in with a tiny epsilon to avoid log(0).
        cont = contingency.astype(float) + 1e-12
        total = cont.sum()
        pxy = cont / total
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        mi = (pxy * np.log(pxy / (px * py + 1e-12))).sum()
        return float(max(0.0, mi))


def evaluate_factor(
    factor: pd.Series,
    label: pd.Series,
    mi: bool = True,
) -> Dict[str, Optional[float]]:
    """Compute IC / RankIC / ICIR / RankICIR / MI for one factor."""
    common = factor.index.intersection(label.index)
    f = factor.reindex(common)
    l = label.reindex(common)

    ic_daily = _daily_corr(f, l, "pearson")
    rank_daily = _daily_corr(f, l, "spearman")

    ic = float(np.mean(ic_daily)) if ic_daily.size else None
    rank_ic = float(np.mean(rank_daily)) if rank_daily.size else None
    icir = float(np.mean(ic_daily) / np.std(ic_daily)) if ic_daily.size > 1 and np.std(ic_daily) > 0 else None
    rank_icir = (
        float(np.mean(rank_daily) / np.std(rank_daily))
        if rank_daily.size > 1 and np.std(rank_daily) > 0
        else None
    )

    mi_val = None
    if mi:
        mi_val = _mi(f.to_numpy(dtype=float), l.to_numpy(dtype=float))

    return {
        "ic": ic,
        "rank_ic": rank_ic,
        "icir": icir,
        "rank_icir": rank_icir,
        "mi": mi_val,
    }


def nan_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return 1.0
    return float(s.isna().mean())
