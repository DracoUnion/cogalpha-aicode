"""CogAlpha command-line entry point.

Examples
--------
# Full run against your OHLCV panel (needs OPENAI_API_KEY):
    python run.py --data data.parquet --horizon 10 --output ./out

# Offline demo on synthetic data (no API key, exercises the pipeline):
    python run.py --demo

# Quick import/self-test of the static checker & executor:
    python run.py --self-test
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogalpha.models import CogAlphaConfig  # noqa: E402
from cogalpha.main import CogAlpha  # noqa: E402
from cogalpha.openai import set_openai_props


def build_config(args: argparse.Namespace) -> CogAlphaConfig:
    cfg = CogAlphaConfig()
    if args.api_key:
        cfg.llm.api_key = args.api_key
    if args.model:
        cfg.llm.model = args.model
    if args.base_url:
        cfg.llm.base_url = args.base_url
    cfg.data_path = args.data
    cfg.forecast_horizon = args.horizon
    cfg.output_dir = args.output
    if args.parent_pool:
        cfg.generation.parent_pool_size = args.parent_pool
    if args.child_pool:
        cfg.generation.child_pool_size = args.child_pool
    if args.searches:
        cfg.generation.evolution_searches = args.searches
    return cfg


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #
_DEMO_FACTORS = [
    "def factor_mom_ret5(df):\n"
    "    \"\"\"5-day momentum on close price. One clear idea.\"\"\"\n"
    "    df_copy = df.copy()\n"
    "    df_copy['factor_mom_ret5'] = df_copy['close'].pct_change(5)\n"
    "    return df_copy['factor_mom_ret5']",

    "def factor_vol_rng20(df):\n"
    "    \"\"\"20-day rolling range scaled by close. One clear idea.\"\"\"\n"
    "    df_copy = df.copy()\n"
    "    df_copy['factor_vol_rng20'] = (df_copy['high'] - df_copy['low']).rolling(20).mean() / df_copy['close']\n"
    "    return df_copy['factor_vol_rng20']",

    "def factor_liquidity_impact(df):\n"
    "    \"\"\"Unit-volume price impact: (high-close)/volume. One clear idea.\"\"\"\n"
    "    df_copy = df.copy()\n"
    "    df_copy['factor_liquidity_impact'] = (df_copy['high'] - df_copy['close']) / (df_copy['volume'] + 1e-9)\n"
    "    return df_copy['factor_liquidity_impact']",
]


def synthetic_panel(days: int = 600, tickers: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days)
    rows = []
    for t in range(tickers):
        mu = rng.normal(0, 0.001)
        sigma = 0.02 * (1 + 0.5 * np.sin(t))
        rets = rng.normal(mu, sigma, days)
        close = 20 * np.exp(np.cumsum(rets))
        open_ = close * (1 + rng.normal(0, 0.003, days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, days)))
        volume = rng.lognormal(14, 0.4, days)
        sub = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.MultiIndex.from_product([dates, [f"T{t:03d}"]], names=["date", "ticker"]),
        )
        rows.append(sub)
    return pd.concat(rows).sort_index()


def run_demo(args: argparse.Namespace) -> None:
    """Offline end-to-end exercise using a stub LLM and synthetic data."""
    from cogalpha import utils
    from cogalpha.models import ParsedFunction

    set_openai_props(args)
    cfg = build_config(args)
    cfg.data_path = ""
    cfg.use_llm = False  # offline demo: static checks only
    random.seed(cfg.generation.random_state)

    print("== CogAlpha demo (offline) ==")
    df = synthetic_panel()
    engine = CogAlpha(cfg)  # no API calls made (cfg.use_llm is False)
    label = engine.compute_label(df, cfg.forecast_horizon)
    columns_desc = utils.build_column_desc_manual(list(df.columns))
    columns_num = len(df.columns)
    engine.agent.columns_desc = columns_desc
    engine.agent.columns_num = columns_num

    # Build a tiny parent pool from the deterministic demo factors.
    import re
    parent_pool = []
    for code in _DEMO_FACTORS:
        m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
        name = m.group(1) if m else "factor_demo"
        pf = ParsedFunction(name=name, code=code)
        f = engine.produce_factor(
            pf, df, label, theme="demo", level="demo",
            agent_id="demo", generation=0, source="generated",
        )
        if f is not None:
            parent_pool.append(f)
            print(f"  [ok] {f.name:24s} {f.summary()}")

    parent_pool = utils.rank_factors(parent_pool)
    print(f"\nParent pool: {len(parent_pool)} factors")
    print("Top-3 by IC:")
    for f in parent_pool[:3]:
        print(f"  - {f.name}: {f.summary()}")
    print("\nDemo finished. The pipeline (compile -> quality -> execute -> evaluate -> classify) works offline.")


def run_selftest() -> int:
    from cogalpha import utils
    from cogalpha.agent import Agent

    code = _DEMO_FACTORS[0]
    issues = utils.check_code_static(code, "factor_mom_ret5")
    assert not issues, issues
    bad = "def f(df):\n    for i in range(3):\n        for j in range(3):\n            pass"
    assert any("Nested loop" in i for i in utils.check_code_static(bad, "f"))
    bad2 = "def g(df):\n    while True:\n        pass"
    assert any("infinite" in i for i in utils.check_code_static(bad2, "g"))

    parsed = Agent.parse_generated_code(
        "[function-1]\ndef x(df):\n    df_copy = df.copy()\n    return df_copy['x']\n[/function-1]"
    )
    assert len(parsed) == 1 and parsed[0].name == "x", parsed
    print(f"Self-test passed. Parsed {len(parsed)} function(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CogAlpha: LLM-driven code-based alpha evolution")
    parser.add_argument("--data", default="", help="Path to OHLCV panel (parquet/csv) with (date,ticker) index.")
    parser.add_argument("--horizon", type=int, default=10, help="Forecast horizon in days (paper: 10).")
    parser.add_argument("--output", default="./cogalpha_output", help="Output directory.")
    parser.add_argument("--model", default="", help="Overrides the LLM model.")
    parser.add_argument("--base-url", default="", dest="base_url", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="", help="OpenAI API key (or set OPENAI_API_KEY).")
    parser.add_argument("--parent-pool", type=int, default=0, dest="parent_pool")
    parser.add_argument("--child-pool", type=int, default=0, dest="child_pool")
    parser.add_argument("--searches", type=int, default=0, dest="searches")
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--demo", action="store_true", help="Run the offline demo on synthetic data.")
    parser.add_argument("--self-test", action="store_true", help="Run the import/static-check self test.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.self_test:
        return run_selftest()
    if args.demo:
        run_demo(args)
        return 0
    if not args.data:
        parser.error("Provide --data, or use --demo / --self-test.")

    set_openai_props(args)
    cfg = build_config(args)
    engine = CogAlpha(cfg)
    result = engine.run()
    print("\n=== Best factors ===")
    for f in result.best(k=10):
        print(f"  {f.name:28s} IC={f.ic:.4f} RankIC={f.rank_ic:.4f} elite={f.elite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())