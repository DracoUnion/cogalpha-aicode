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
import re
import yaml
from pathlib import Path
from pydantic import parse_obj_as
from os import path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, Future
import os
import numpy as np
import pandas as pd

from . import utils
from .agent import Agent
from .models import CogAlphaConfig
from .prompts import _AGENTS
from .models import Factor, FeedbackSummary, ParsedFunction, SearchResult

logger = logging.getLogger(__name__)


class CogAlpha:
    """The framework entry point: data in, evolved alphas out."""

    def __init__(self, cfg: CogAlphaConfig) -> None:
        self.cfg = cfg
        self.agent = Agent(self.cfg)
        self._logger = logger
        self.proj_dir = re.sub(r'\.\w+$', '', cfg.data_path)

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
            coldesc_fname = path.join(self.proj_dir, 'colum_desc.txt')
            if path.isfile(coldesc_fname) and \
               path.getsize(coldesc_fname):
               return open(coldesc_fname, encoding='utf8').read()
            else:
                coldesc = self.agent.describe_columns(df)
                open(coldesc_fname, 'w', encoding='utf8').write(coldesc)
                return coldesc
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

        ic_daily = utils.daily_corr(f, l, "pearson")
        rank_daily = utils.daily_corr(f, l, "spearman")

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
            mi_val = utils.normalized_mi(f.to_numpy(dtype=float), l.to_numpy(dtype=float))

        return {
            "ic": ic,
            "rank_ic": rank_ic,
            "icir": icir,
            "rank_icir": rank_icir,
            "mi": mi_val,
        }

    def classify_factor(self, factor: Factor) -> Factor:
        """Set `qualified`/`elite` based on the configured thresholds."""
        thresholds = utils.metric_thresholds(self.cfg.evaluation)
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
            if utils.threshold_ok(v, q_th):
                qual_pass += 1
                if utils.threshold_ok(v, e_th):
                    elite_pass += 1

        if total == 0:
            factor.qualified = False
            factor.elite = False
        else:
            # All present metrics must clear their thresholds.
            factor.qualified = qual_pass == total
            factor.elite = elite_pass == total
        return factor

    def validate_factor(
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
        step = "",
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
        self._step(f"{step}.1", f"检查因子质量：{pf.name}")
        final_code = self.agent.gate_factor(factor.code, factor.name)
        if final_code is None:
            return None
        factor.code = final_code

        # 2) Execute.
        self._step(f"{step}.2", f"执行因子：{pf.name}")
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
        self._step(f"{step}.3", f"评估因子：{pf.name}")
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


    def _tr_build_parent_pool(
        self,
        idx: int,
        columns_desc: str,
        columns_num: int,
        df: pd.DataFrame, 
        label: pd.Series,
    ) -> List[Factor]:
        self._step(f"1.{idx+1}", "构建初始父池")
        pool: List[Factor] = []
        agent_ids = self.agent.list_agents()
        while True:
            agent_id = random.choice(agent_ids)
            self._step(f"1.{idx+1}.1", f"生成因子，Agent：{agent_id}")
            level, _, _ = _AGENTS[agent_id]
            pfs = self.agent.generate_code(
                agent_id, columns_desc, columns_num,
                self.cfg.generation.num_per_request, 
                self.cfg.forecast_horizon, None,
            )
            pf_names = ', '.join(pf.name for pf in pfs)
            self._step(f"1.{idx+1}.1", f"生成因子完毕：{pf_names}")
            for pf in pfs:
                self._step(f"1.{idx+1}.2", f"验证因子：{pf.name}")
                f = self.validate_factor(
                    pf, df, label,
                    theme=agent_id, level=level, agent_id=agent_id,
                    generation=0, source="generated",
                    step=f"1.{idx+1}.2",
                )
                if f is not None and f.executable:
                    pool.append(f)
                    self._step(f"1.{idx+1}.2", f"验证因子完毕：{f.name}")
                else:
                    self._step(f"1.{idx+1}.2", f"验证因子失败：{pf.name}")
            if pool: break

        return pool

    def _step_build_initial_parent_pool(
        self,
        columns_desc, columns_num,
        df, label, result,
    ):
        parent_pool_fname = path.join(self.proj_dir, 'parent_pool.yaml')
        if path.isfile(parent_pool_fname) and \
           path.getsize(parent_pool_fname):
           parent_pool = yaml.safe_load(open(parent_pool_fname, encoding='utf8').read())
           parent_pool = parse_obj_as(List[Factor], parent_pool)
        else:
            parent_pool: List[Factor] = []
            utils.write_yaml(parent_pool_fname, parent_pool)

        self._step(
            "1", "构建初始父池（目标 %d 个因子，已有 %d 个）", 
            self.cfg.generation.initial_pool_size,
            len(parent_pool),
        )

        trpool = ThreadPoolExecutor(self.cfg.init_threads)
        hdls: List[Future] = []
        rest_num = max(0, self.cfg.generation.initial_pool_size - len(parent_pool))
        for i in range(rest_num):
            h = trpool.submit(
                self._tr_build_parent_pool,
                i, columns_desc, columns_num,
                df, label,
            )
            hdls.append(h)
            if len(hdls) > self.cfg.init_threads:
                for h in hdls:
                    parent_pool += h.result()
                hdls = []
            if i % 10 == 0:
                utils.write_yaml(parent_pool_fname, parent_pool)
        for h in hdls:
            parent_pool += h.result()
        hdls = []
        utils.write_yaml(parent_pool_fname, parent_pool)
  
        parent_pool = utils.rank_factors(parent_pool)[: self.cfg.generation.parent_pool_size]
        result.candidates = list(parent_pool)
        result.elite = list(parent_pool)
        self._step(f"1", "初始父池完成：%d 个因子", len(parent_pool))
        self._log_pool("initial", parent_pool)
        return parent_pool
    
    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #
    def run(self) -> SearchResult:
        self._step("0.1", "初始化随机种子（seed=%d）", self.cfg.generation.random_state)
        random.seed(self.cfg.generation.random_state)

        self._step("0.2", "加载 OHLCV 数据面板")
        df = self._load()
        os.makedirs(self.proj_dir, exist_ok=True)

        self._step("0.3", "计算 %d 日前向收益标签", self.cfg.forecast_horizon)
        label = self.compute_label(df, self.cfg.forecast_horizon)

        self._step("0.4", "生成列描述")
        columns_desc = self._describe(df)
        columns_num = len(df.columns)

        self._step("0.5", "配置 Agent（%d 列）", columns_num)
        self.agent.columns_desc = columns_desc
        self.agent.columns_num = columns_num

        gen = self.cfg.generation
        agent_ids = self.agent.list_agents()
        self._step("0.6", "枚举生成 agent（%d 个）", len(agent_ids))

        result = SearchResult()

        # --- Phase 1: build the initial parent pool. ---
        parent_pool = self._step_build_initial_parent_pool(columns_desc, columns_num, df, label, result)

        # --- Phase 2: evolution searches over each agent. ---
        self._step("2", "进化搜索（%d 次搜索）", gen.evolution_searches)
        for evo_idx in range(gen.evolution_searches):
            self._step(
                f"2.{evo_idx}", 
                "进化搜索 %d/%d",
                evo_idx + 1, gen.evolution_searches
            )
            parent_pool, result = self._evolve_agent(
                df, label, columns_desc, columns_num, parent_pool, result,
                step=f"2.{evo_idx}",
            )

        self._step("3", "排序最终候选 / 精英")
        result.candidates = utils.rank_factors(result.candidates)
        result.elite = utils.rank_factors(result.elite)

        self._step("4", "保存结果")
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
        parent_pool: List[Factor],
        result: SearchResult,
        *,
        step: str,
    ) -> tuple[List[Factor], SearchResult]:
        gen = self.cfg.generation

        self._step(f"{step}.1", "携带前代精英（top %d）", gen.carry_elite_top)
        prev_elite = utils.rank_factors(result.elite)[: gen.carry_elite_top]
        feedback: FeedbackSummary = FeedbackSummary()

        for gen_idx in range(gen.generations_per_evo):
            # Breed the child pool from the parent pool.
            children = self._breed(
                df, label, columns_desc, columns_num,
                parent_pool, feedback, gen_idx,
                step=f"{step}.2.{gen_idx+1}",
            )

            # Inject qualified/elite children into the parent pool.
            if children:
                parent_pool = self._inject(parent_pool, children, gen.parent_pool_size)
                for c in children:
                    result.candidates.append(c)
                    if c.elite:
                        result.elite.append(c)

            # Refresh adaptive feedback every injection window.
            if gen_idx % gen.inject_every == 0:
                feedback = self._refresh_feedback(result.candidates)

            self._step(
                f"{step}.2.{gen_idx+1}", "第 %d/%d 代：children=%d qualified=%d elite=%d",
                gen_idx, gen.generations_per_evo,
                len(children), sum(1 for c in children if c.qualified), len(result.elite),
            )

        # Carry forward the previous elites to seed the next search.
        self._step(f"{step}.3", "推进精英到下一轮搜索")
        current_top = utils.rank_factors(result.elite)[: gen.carry_elite_top]
        parent_pool = utils.rank_factors(parent_pool + prev_elite + current_top)[: gen.parent_pool_size]
        return parent_pool, result

    def _tr_breed(
        self,
        round_idx,
        df, label, columns_desc, columns_num,
        parent_pool, feedback, generation_idx,
        *,
        step: str,
    ):
        new_factors = []
        num_per = self.cfg.generation.num_per_request
        horizon = self.cfg.forecast_horizon
        # Choose an evolution operation.
        op = random.choice(["generate", "mutation", "crossover", "crossover_then_mutation"])
        self._step(f"{step}.{round_idx+1}.1", "繁殖操作：%s", op)
        candidates: List = []
        source = ""

        while True:
            if op == "generate":
                agent_id = random.choice(list(_AGENTS.keys()))
                candidates = self.agent.generate_code(
                    agent_id, columns_desc, columns_num,
                    num_per, horizon, feedback,
                )
                source = "generated"

            elif op == "mutation" and parent_pool:
                parent = random.choice(parent_pool[: max(1, len(parent_pool) // 2)])
                candidates = self.agent.mutate(
                    columns_desc, columns_num, num_per, horizon,
                    parent.code, extra_guidance=feedback.effective,
                )
                source = "mutation"

            elif op in ("crossover", "crossover_then_mutation") and len(pool) >= 2:
                p1, p2 = random.sample(parent_pool[: max(2, len(parent_pool) // 2)], 2)
                candidates = self.agent.crossover(
                    columns_desc, columns_num, num_per, horizon,
                    p1.code, p2.code, extra_guidance=feedback.effective,
                )
                source = "crossover"
            cand_names = ', '.join(pf.name for pf in candidates)
            self._step(f"{step}.{round_idx+1}.1", "繁殖操作完毕：%s", cand_names)

            for pf in candidates:
                self._step(f"{step}.{round_idx+1}.2", "验证因子：%s", pf.name)
                f = self.validate_factor(
                    pf, df, label,
                    theme=agent_id, level="breed", agent_id=agent_id,
                    generation=generation_idx, source=source,
                    step=f"{step}.{round_idx+1}.2",
                )
                if f is not None and f.executable and f.qualified:
                    self._step(f"{step}.{round_idx+1}.2", "检验因子成功：%s", f.name)
                    new_factors.append(f)
                else:
                    self._step(f"{step}.{round_idx+1}.2", "检验因子失败：%s", pf.name)
            if new_factors: break
        return new_factors

    # ------------------------------------------------------------------ #
    # Breeding
    # ------------------------------------------------------------------ #
    def _breed(
        self,
        df, label, columns_desc, columns_num,
        parent_pool, feedback, generation_idx,
        *,
        step: str,
    ) -> List[Factor]:
        cfg = self.cfg
        gen = cfg.generation
        new_factors: List[Factor] = []
        parent_pool = utils.rank_factors(parent_pool)

        trpool = ThreadPoolExecutor(cfg.breed_threads)
        hdls = []
        total = max(1, gen.child_pool_size // max(1, len(parent_pool)))
        for round_idx in range(total):
            h = trpool.submit(
                self._tr_breed,
                round_idx,
                df, label, columns_desc, columns_num,
                parent_pool, feedback, 
                generation_idx,
                step=step,
            )
            hdls.append(h)
            if len(hdls) > cfg.breed_threads:
                for h in hdls:
                    new_factors += h.result()
                hdls = []
        for h in hdls:
            new_factors += h.result()
        hdls = []
            
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
            return self.agent.build_feedback(effective, ineffective)
        except Exception as exc:  # pragma: no cover - API dependent
            self._logger.warning("feedback build failed: %s", exc)
            return utils.deterministic_build_feedback(effective, ineffective)

    def _log_pool(self, tag: str, pool: List[Factor]) -> None:
        top = utils.rank_factors(pool)[:3]
        self._logger.info(
            "[%s] pool size=%d top_ic=[%s]",
            tag, len(pool), ", ".join(f"{f.ic:.4f}" if f.ic is not None else "N/A" for f in top),
        )

    def _step(self, seq: str, msg: str, *args) -> None:
        """Log a run step as ``[seq] message``."""
        if args:
            self._logger.info("[%s] %s", seq, msg % args)
        else:
            self._logger.info("[%s] %s", seq, msg)

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