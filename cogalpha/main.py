"""CogAlpha evolution-search orchestrator.

Implements the paper's main loop:

  initial pool -> parent pool -> (per task agent, per search, per generation)
    breed (generation / mutation / crossover) -> quality check -> execute ->
    evaluate -> adaptive feedback -> inject top alphas into the parent pool ->
    carry elite alphas forward.

Pools:
  - initial pool   (paper: 80)
  - parent pool    (paper: 32)
  - child pool     (paper: 96)
  - candidate pool (qualified alphas)
  - elite pool     (best alphas, carried across generations)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import utils
from .agent import Agent
from .models import CogAlphaConfig
from .llm import LLMClient
from .prompts import _AGENTS
from .models import Factor, FeedbackSummary, ParsedFunction, SearchResult

logger = logging.getLogger(__name__)


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


def _have(eval_cfg) -> Dict[str, Tuple[float, float]]:
    return {
        "ic": eval_cfg.ic,
        "rank_ic": eval_cfg.rank_ic,
        "icir": eval_cfg.icir,
        "rank_icir": eval_cfg.rank_icir,
        "mi": eval_cfg.mi,
    }


def _ok(v: float, th: float) -> bool:
    """Thresholds are interpreted as directional cutoffs (positive = good)."""
    if th is None:
        return True
    return v >= th


class CogAlpha:
    """The framework entry point: data in, evolved alphas out."""

    def __init__(self, cfg: CogAlphaConfig) -> None:
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.agent = Agent(self.llm, self.cfg)
        self._logger = logger

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _load(self) -> pd.DataFrame:
        """Load the configured OHLCV panel (parquet or CSV) as (date, ticker).

        The panel is expected as a (date, ticker) MultiIndex with OHLCV columns;
        CSV input may also carry `trade_date` / `ts_code` columns that are
        normalized to a datetime date index here.
        """
        if not self.cfg.data_path:
            raise ValueError("config.data_path must point to an OHLCV panel.")

        p = Path(self.cfg.data_path)
        if p.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, parse_dates=False)

        df.rename(columns={"trade_date": "date", "ts_code": "ticker"}, inplace=True)
        if not isinstance(df.index, pd.MultiIndex):
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

    def _describe(self, df: pd.DataFrame) -> str:
        try:
            return self.agent.describe_columns(df)
        except Exception as exc:  # pragma: no cover - API dependent
            self._logger.warning("column_description failed: %s; using manual block", exc)
        return utils.build_column_desc_manual(list(df.columns))

    # ------------------------------------------------------------------ #
    # Pipeline (gate -> execute -> evaluate -> classify)
    # ------------------------------------------------------------------ #
    def compute_label(self, data: pd.DataFrame, horizon: int, price_col: str = "open") -> pd.Series:
        """Forward-return labels used as the fitness target."""
        return utils.forward_returns(data, horizon, price_col)

    def evaluate_factor(
        self, factor: pd.Series, label: pd.Series, mi: bool = True
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

    def classify_factor(self, factor: Factor) -> Factor:
        """Set `qualified`/`elite` based on the configured thresholds."""
        thresholds = _have(self.cfg.evaluation)
        qual_pass = 0
        elite_pass = 0
        total = 0

        for metric, (q_th, e_th) in thresholds.items():
            v = getattr(factor, metric, None)
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
            factor.qualified = False
            factor.elite = False
        else:
            # All present metrics must clear their thresholds.
            factor.qualified = qual_pass == total
            factor.elite = elite_pass == total
        return factor

    def produce_factor(
        self,
        pf: ParsedFunction,
        data: pd.DataFrame,
        label: pd.Series,
        *,
        theme: str = "",
        level: str = "",
        agent_id: str = "",
        generation: int = 0,
        source: str = "generated",
    ) -> Optional[Factor]:
        """Run quality, execute, and evaluate a parsed function."""
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
        final_code = self.agent.gate_factor(factor.code, factor.name)
        if final_code is None:
            return None
        factor.code = final_code

        # 2) Execute.
        series = self._execute(factor, data)
        if series is None:
            factor.executable = False
            factor.accepted_by_judge = False
            return None
        factor.executable = True
        factor.nan_ratio = utils.nan_ratio(series)
        if factor.nan_ratio > self.cfg.generation.max_nan_ratio:
            logger.debug("Factor %s NaN ratio %.2f too high; dropping.", factor.name, factor.nan_ratio)
            return None

        # 3) Evaluate.
        metrics = self.evaluate_factor(series, label)
        factor.set_metrics(metrics)
        factor = self.classify_factor(factor)
        return factor

    def _execute(self, factor: Factor, data: pd.DataFrame) -> Optional[pd.Series]:
        try:
            func = utils.compile_factor(factor.code)
        except Exception as exc:  # pragma: no cover - compile errors
            logger.info("Factor %s failed to compile: %s", factor.name, exc)
            return None
        try:
            return utils.apply_factor(data, func, factor.name)
        except Exception as exc:
            logger.info("Factor %s failed to execute: %s", factor.name, exc)
            return None

    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #
    def run(self) -> SearchResult:
        random.seed(self.cfg.generation.random_state)
        df = self._load()
        label = self.compute_label(df, self.cfg.forecast_horizon)
        columns_desc = self._describe(df)
        columns_num = len(df.columns)
        self.agent.columns_desc = columns_desc
        self.agent.columns_num = columns_num

        gen = self.cfg.generation
        agent_ids = self.agent.list_agents()

        result = SearchResult()

        # --- Phase 1: build the initial parent pool. ---
        parent_pool: List[Factor] = []
        attempts = 0
        while len(parent_pool) < gen.initial_pool_size and attempts < gen.initial_pool_size * 4:
            attempts += 1
            agent_id = random.choice(agent_ids)
            level, _, _ = _AGENTS[agent_id]
            pfs = self.agent.generate_code(
                agent_id, columns_desc, columns_num,
                gen.num_per_request, self.cfg.forecast_horizon, None,
            )
            for pf in pfs:
                f = self.produce_factor(
                    pf, df, label,
                    theme=agent_id, level=level, agent_id=agent_id,
                    generation=0, source="generated",
                )
                if f is not None and f.executable:
                    parent_pool.append(f)
            if attempts % 10 == 0:
                self._logger.info("initial pool: %d/%d", len(parent_pool), gen.initial_pool_size)

        parent_pool = utils.rank_factors(parent_pool)[: gen.parent_pool_size]
        result.candidates = list(parent_pool)
        result.elite = list(parent_pool)
        self._log_pool("initial", parent_pool)

        # --- Phase 2: evolution searches over each agent. ---
        for search_idx in range(gen.evolution_searches):
            for agent_id in agent_ids:
                self._logger.info("== evolution search %d/%d, agent %s ==",
                                  search_idx + 1, gen.evolution_searches, agent_id)
                level, _, _ = _AGENTS[agent_id]
                parent_pool, result = self._evolve_agent(
                    df, label, columns_desc, columns_num, agent_id, level, parent_pool, result
                )

        result.candidates = utils.rank_factors(result.candidates)
        result.elite = utils.rank_factors(result.elite)
        self._save(result)
        return result

    # ------------------------------------------------------------------ #
    # Evolution over one agent
    # ------------------------------------------------------------------ #
    def _evolve_agent(
        self,
        df: pd.DataFrame,
        label: pd.Series,
        columns_desc: str,
        columns_num: int,
        agent_id: str,
        level: str,
        parent_pool: List[Factor],
        result: SearchResult,
    ) -> tuple[List[Factor], SearchResult]:
        gen = self.cfg.generation
        prev_elite = utils.rank_factors(result.elite)[: gen.carry_elite_top]
        feedback: FeedbackSummary = FeedbackSummary()

        for sub in range(gen.sub_loops):
            for g in range(gen.generations_per_sub_loop):
                generation_idx = sub * gen.generations_per_sub_loop + g + 1

                # Breed the child pool from the parent pool.
                children = self._breed(
                    df, label, columns_desc, columns_num,
                    agent_id, level, parent_pool, feedback, generation_idx,
                )

                # Inject qualified/elite children into the parent pool.
                if children:
                    parent_pool = self._inject(parent_pool, children, gen.parent_pool_size)
                    for c in children:
                        result.candidates.append(c)
                        if c.elite:
                            result.elite.append(c)

                # Refresh adaptive feedback every injection window.
                if generation_idx % gen.inject_every == 0:
                    feedback = self._refresh_feedback(result.candidates)

                self._logger.info(
                    "  gen %d/%d: children=%d qualified=%d elite_total=%d",
                    generation_idx, gen.sub_loops * gen.generations_per_sub_loop,
                    len(children), sum(1 for c in children if c.qualified), len(result.elite),
                )

        # Carry forward the previous elites to seed the next search.
        current_top = utils.rank_factors(result.elite)[: gen.carry_elite_top]
        parent_pool = utils.rank_factors(parent_pool + prev_elite + current_top)[: gen.parent_pool_size]
        return parent_pool, result

    # ------------------------------------------------------------------ #
    # Breeding
    # ------------------------------------------------------------------ #
    def _breed(
        self,
        df, label, columns_desc, columns_num,
        agent_id, level, parent_pool, feedback, generation_idx,
    ) -> List[Factor]:
        cfg = self.cfg
        gen = cfg.generation
        num_per = gen.num_per_request
        horizon = cfg.forecast_horizon
        new_factors: List[Factor] = []
        pool = utils.rank_factors(parent_pool)

        for _ in range(max(1, gen.child_pool_size // max(1, len(parent_pool)))):
            # Choose an evolution operation.
            op = random.choice(["generate", "mutation", "crossover", "crossover_then_mutation"])
            candidates: List = []

            if op == "generate":
                pfs = self.agent.generate_code(
                    agent_id, columns_desc, columns_num,
                    num_per, horizon, feedback,
                )
                candidates = [(pf, "generated") for pf in pfs]

            elif op == "mutation" and pool:
                parent = random.choice(pool[: max(1, len(pool) // 2)])
                pfs = self.agent.mutate(
                    columns_desc, columns_num, num_per, horizon,
                    parent.code, extra_guidance=feedback.effective,
                )
                candidates = [(pf, "mutation") for pf in pfs]

            elif op in ("crossover", "crossover_then_mutation") and len(pool) >= 2:
                p1, p2 = random.sample(pool[: max(2, len(pool) // 2)], 2)
                pfs = self.agent.crossover(
                    columns_desc, columns_num, num_per, horizon,
                    p1.code, p2.code, extra_guidance=feedback.effective,
                )
                candidates = [(pf, "crossover") for pf in pfs]

            for pf, source in candidates:
                f = self.produce_factor(
                    pf, df, label,
                    theme=agent_id, level=level, agent_id=agent_id,
                    generation=generation_idx, source=source,
                )
                if f is not None and f.executable and f.qualified:
                    new_factors.append(f)

        return new_factors[: gen.child_pool_size]

    # ------------------------------------------------------------------ #
    # Pool / feedback helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _inject(parent_pool: List[Factor], children: List[Factor], cap: int) -> List[Factor]:
        merged = utils.rank_factors(parent_pool + children)
        return merged[:cap]

    def _refresh_feedback(self, candidates: List[Factor]) -> FeedbackSummary:
        cfg = self.cfg
        if not candidates:
            return FeedbackSummary()
        valid = [c for c in candidates if c.ic is not None]
        effective = utils.rank_factors(valid)[: cfg.feedback.top_effective]
        ineffective = utils.rank_factors(valid, key="ic")[-cfg.feedback.worst_ineffective:] if valid else []
        try:
            return utils.build_feedback(effective, ineffective, self.llm)
        except Exception as exc:  # pragma: no cover - API dependent
            self._logger.warning("feedback build failed: %s", exc)
            return utils.deterministic_build_feedback(effective, ineffective)

    def _log_pool(self, tag: str, pool: List[Factor]) -> None:
        top = utils.rank_factors(pool)[:3]
        self._logger.info(
            "[%s] pool size=%d top_ic=[%s]",
            tag, len(pool), ", ".join(f"{f.ic:.4f}" for f in top),
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _save(self, result: SearchResult) -> None:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        import json

        def dump(factors: List[Factor], filename: str) -> None:
            rows = []
            for f in factors:
                rows.append({
                    "name": f.name, "theme": f.theme, "level": f.level, "source": f.source,
                    "generation": f.generation, "ic": f.ic, "rank_ic": f.rank_ic,
                    "icir": f.icir, "rank_icir": f.rank_icir, "mi": f.mi,
                    "qualified": f.qualified, "elite": f.elite, "code": f.code,
                })
            (out / filename).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        dump(result.candidates, "candidates.json")
        dump(result.elite, "elite.json")
        self._logger.info("Saved %d candidates and %d elites to %s", len(result.candidates), len(result.elite), out)