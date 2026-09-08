"""Agent: prompts, factor generation/evolution, and the quality checker.

A single `Agent` holds the LLM client plus the panel context (`columns_desc` /
`columns_num`) and the config, and owns:

  - prompt construction: render `{placeholder}` tokens from the `prompt_loader`
    constants (build_generation / build_quality / build_evolution prompts)
  - factor generation / evolution: `generate_code` / `mutate` / `crossover`
    via `llm.complete`
  - quality checking (formerly `QualityGate`): static checks, LLM quality
    review, repair, judge, and logic improvement, orchestrated by `gate_factor`
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple
from pydantic import BaseModel

from . import utils
from .models import CogAlphaConfig, FeedbackSummary, JudgeResult, ParsedFunction, QualityResult, Factor
from .openai import call_llm_retry
from .utils import _render_examples, ext_code_block
from .prompts import (
    _AGENTS,
    _COLUMN_DESCRIPTION,
    _EFFECTIVE_ANALYSIS,
    _EVOLUTION,
    _GENERATION_TEMPLATES,
    _INEFFECTIVE_ANALYSIS,
    _QUALITY,
    _SYSTEM_MESSAGE,
    _EFFECTIVE_SUMMARY,
    _INEFFECTIVE_SUMMARY,
)

logger = logging.getLogger(__name__)


class Agent:
    """Holds the LLM client and performs prompt build, factor ops, and quality."""

    def __init__(
        self,
        cfg: CogAlphaConfig,
        columns_desc: str = "",
        columns_num: int = 0,
    ) -> None:
        self.cfg = cfg
        self.columns_desc = columns_desc
        self.columns_num = columns_num
        self.max_repair = cfg.generation.max_repair_attempts or utils.MAX_REPAIR_ATTEMPTS_DEFAULT

    # ------------------------------------------------------------------ #
    # Prompt building
    # ------------------------------------------------------------------ #
    def build_generation_prompt(
        self,
        agent_id: str,
        columns_desc: str,
        columns_num: int,
        num_per_request: int,
        forecast_horizon: int,
        effective_CoT: str = "",
        ineffective_CoT: str = "",
    ) -> Tuple[str, str]:
        """Assemble the full user prompt for one seven-level generation agent.

        The per-agent template is a single complete multi-line prompt with
        `{placeholder}` tokens (see `_GENERATION_TEMPLATES`); substitution uses
        `re.sub`/`.replace` only — no formatting or joining.
        Returns (system_message, user_message).
        """
        user = _GENERATION_TEMPLATES[agent_id]
        user = utils.render_placeholders(
            user,
            columns_desc=columns_desc,
            columns_num=columns_num,
            num_per_request=num_per_request,
            forecast_horizon=forecast_horizon,
        )
        user = user.replace(
            "{effective_block}", utils.render_cot_block(_EFFECTIVE_ANALYSIS, effective_CoT)
        ).replace(
            "{ineffective_block}", utils.render_cot_block(_INEFFECTIVE_ANALYSIS, ineffective_CoT)
        )
        return _SYSTEM_MESSAGE, user

    def build_quality_prompt(self, agent_name: str, **kwargs) -> str:
        """Fill one quality-checker template's `{...}` placeholders."""
        return utils.render_placeholders(_QUALITY[agent_name], **kwargs)

    def build_evolution_prompt(self, agent_name: str, **kwargs) -> str:
        """Fill one evolution template's `{...}` placeholders."""
        return utils.render_placeholders(_EVOLUTION[agent_name], **kwargs)

    # ------------------------------------------------------------------ #
    # LLM driving (llm.complete)
    # ------------------------------------------------------------------ #
    def generate_code(
        self,
        agent_id: str,
        columns_desc: str,
        columns_num: int,
        num_per_request: int,
        forecast_horizon: int,
        feedback: Optional[FeedbackSummary] = None,
        temperature: Optional[float] = None,
    ) -> List[ParsedFunction]:
        """Call the LLM for one generation agent and parse the factor functions."""
        eff = feedback.effective if feedback else ""
        ineff = feedback.ineffective if feedback else ""
        system, user = self.build_generation_prompt(
            agent_id=agent_id,
            columns_desc=columns_desc,
            columns_num=columns_num,
            num_per_request=num_per_request,
            forecast_horizon=forecast_horizon,
            effective_CoT=eff,
            ineffective_CoT=ineff,
        )
        try:
            raw = self.complete(system, user, temperature=temperature, model=self.cfg.llm.model)
        except Exception as exc:  # pragma: no cover - API dependent
            logger.error("generation call for %s failed: %s", agent_id, exc)
            return []
        return self.parse_generated_code(raw)

    def _intro(
        self,
        columns_desc: str,
        columns_num: int,
        num_per_request: int,
        forecast_horizon: int,
    ) -> str:
        """The compact column-schema intro shared by the evolution prompts."""
        return (
            f"Below is the schema of the input DataFrame and a list of {columns_num} existing factors:\n\n"
            f"{columns_desc}\n\n"
            f"Please generate new factor functions to forecast {forecast_horizon}-day forward returns."
        )

    def mutate(
        self,
        columns_desc: str,
        columns_num: int,
        num_per_request: int,
        forecast_horizon: int,
        original_code: str,
        extra_guidance: str = "",
        temperature: Optional[float] = None,
    ) -> List[ParsedFunction]:
        user = self.build_evolution_prompt(
            "mutation_agent",
            intro=self._intro(columns_desc, columns_num, num_per_request, forecast_horizon),
            original_factor_code=original_code,
            extra_guidance=extra_guidance,
        )
        try:
            raw = self.complete(_SYSTEM_MESSAGE, user, temperature=temperature)
        except Exception as exc:  # pragma: no cover
            logger.error("mutation call failed: %s", exc)
            return []
        return self.parse_generated_code(raw)

    def crossover(
        self,
        columns_desc: str,
        columns_num: int,
        num_per_request: int,
        forecast_horizon: int,
        parent_1: str,
        parent_2: str,
        extra_guidance: str = "",
        temperature: Optional[float] = None,
    ) -> List[ParsedFunction]:
        user = self.build_evolution_prompt(
            "crossover_agent",
            intro=self._intro(columns_desc, columns_num, num_per_request, forecast_horizon),
            parent_factor_1_code=parent_1,
            parent_factor_2_code=parent_2,
            extra_guidance=extra_guidance,
        )
        try:
            raw = self.complete(_SYSTEM_MESSAGE, user, temperature=temperature)
        except Exception as exc:  # pragma: no cover
            logger.error("crossover call failed: %s", exc)
            return []
        return self.parse_generated_code(raw)

    # ------------------------------------------------------------------ #
    # Quality checker (static + LLM agents)
    # ------------------------------------------------------------------ #
    def static_check(self, code: str, name: str) -> List[str]:
        """Deterministic AST checks: nested loops, syntax, return column name."""
        issues = utils.check_code_static(code, name)
        issues += utils.validate_name(code, name)
        return issues

    def judge(self, code: str) -> JudgeResult:
        """Judge Agent: decide Accept/Reject + improvement feedback."""
        prompt = self.build_quality_prompt("judge_agent", new_factor_code=code)
        return self.complete_json(_SYSTEM_MESSAGE, prompt, JudgeResult)

    def code_quality(self, code: str) -> QualityResult:
        """Code Quality Agent: LLM review that complements static checks."""
        prompt = self.build_quality_prompt("code_quality_agent", code=code)
        try:
            raw = self.complete_quality(_SYSTEM_MESSAGE, prompt)
        except Exception as exc:  # pragma: no cover - API dependent
            logger.warning("code_quality LLM call failed: %s", exc)
            return QualityResult(status="correct")
        issues = utils.quality_issues(raw)
        corrected = utils.quality_corrected(raw)
        status = "needs adjustments" if issues else "correct"
        return QualityResult(status=status, issues=issues, corrected_code=corrected)

    def repair(self, old_code: str, error: str) -> str:
        """Code Repair Agent: fix an execution/static failure."""
        prompt = self.build_quality_prompt(
            "code_repair_agent",
            columns_num=self.columns_num,
            columns_desc=self.columns_desc,
            old_code=old_code,
            error=error,
        )
        raw = self.complete_quality(_SYSTEM_MESSAGE, prompt)
        funcs = self.parse_generated_code(raw)
        return funcs[0].code if funcs else old_code

    def logic_improve(self, old_code: str, feedback: str) -> str:
        """Logic Improvement Agent: improve a rejected factor."""
        prompt = self.build_quality_prompt(
            "logic_improvement_agent",
            columns_num=self.columns_num,
            columns_desc=self.columns_desc,
            old_code=old_code,
            dynamic_feedback=feedback,
        )
        raw = self.complete_quality(_SYSTEM_MESSAGE, prompt)
        funcs = self.parse_generated_code(raw)
        return funcs[0].code if funcs else old_code

    def check_code(self, code: str, name: str) -> str:
        """Run quality checks and (optionally) the LLM quality agent.

        Returns the possibly-repaired code. Static violations trigger the
        repair agent; nested-loop/infinite-loop violations are hard failures
        that cannot be auto-repaired (returned unrepaired so the caller may
        discard). When `self.cfg.use_llm` is False, only the deterministic
        static checks run (no network calls).
        """
        issues = self.static_check(code, name)
        hard_fail = any("Nested loop" in i or "infinite loop" in i or "SyntaxError" in i for i in issues)
        if hard_fail:
            logger.debug("Hard static violation for %s: %s", name, issues)
            return code

        if not self.cfg.use_llm:
            return code

        # LLM quality review for style/compliance issues.
        try:
            qr = self.code_quality(code)
            code = qr.corrected_code or code
        except Exception as exc:  # pragma: no cover
            logger.warning("code_quality failed for %s: %s", name, exc)

        return code

    def gate_factor(self, code: str, name: str, *, use_judge: bool = True) -> Optional[str]:
        """Run the full quality pipeline for one factor.

        Quality (with repair rounds) -> judge -> logic improvement.
        Returns the final acceptable code, or None if the factor was discarded.
        """
        current = code
        # 1) Quality check (+ repair rounds for non-hard failures).
        for attempt in range(self.max_repair):
            current = self.check_code(current, name)
            issues = self.static_check(current, name)
            if not issues:
                break
            if not self.cfg.use_llm:
                # Offline mode cannot call the repair agent.
                logger.info("Factor %s has static issues and LLM repair is disabled; discarding.", name)
                return None
            if attempt == self.max_repair - 1:
                logger.info("Factor %s failed quality after %d repairs; discarding.", name, self.max_repair)
                return None
            # Get an error message to feed the repair agent.
            error = "; ".join(issues[:4]) or "unknown static issue"
            current = self.repair(current, error)

        # 2) Judge.
        if use_judge and self.cfg.use_llm:
            try:
                jr = self.judge(current)
            except Exception as exc:  # pragma: no cover
                logger.warning("judge failed for %s: %s", name, exc)
                jr = JudgeResult(recommendation="Accept")
            if jr.recommendation != "Accept":
                # 3) Logic improvement.
                improved = self.logic_improve(current, jr.feedback)
                # Re-run static checks on the improved version.
                if not self.static_check(improved, name):
                    current = improved
                else:
                    logger.info("Logic-improved factor %s still invalid; keeping original.", name)

        return current

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_generated_code(raw: str) -> List[ParsedFunction]:
        """Extract `[function-N]` blocks from a generation response."""
        return utils.parse_generated_code(raw)

    def describe_columns(self, df) -> str:
        """Turn the panel's column names into a `columns_desc` block.

        Uses the `_COLUMN_DESCRIPTION` helper prompt through `llm.complete_quality`
        (only when `self.cfg.use_llm` is on); on any failure it falls back to a
        plain list of the available columns.
        """
        factor_cols = [c for c in df.columns if c.lower() not in {"label", "y", "target", "ret_fwd"}]
        if not factor_cols:
            raise ValueError("The panel has no factor columns to describe.")

        if not self.cfg.use_llm:
            return "\n".join(f"- {c}" for c in factor_cols)

        prompt = _COLUMN_DESCRIPTION.replace("{factor_names}", ", ".join(factor_cols))
        try:
            raw = self.complete_quality(_SYSTEM_MESSAGE, prompt)
        except Exception as exc:  # pragma: no cover - network/API dependent
            logger.warning("column_description LLM call failed (%s); using plain names", exc)
            return "\n".join(f"- {c}" for c in factor_cols)

        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                col, desc = line.split(":", 1)
                lines.append(f"- `{col.strip()}`: {desc.strip()}")
            else:
                lines.append(f"- {line}")
        return "\n".join(lines) or "\n".join(f"- {c}" for c in factor_cols)

    def list_agents(self) -> List[str]:
        """Sorted ids of the seven-level generation agents."""
        return sorted(_AGENTS.keys())


    def build_feedback(
        self,
        effective: List[Factor],
        ineffective: List[Factor],
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
                effective_CoT = self.complete_quality(_SYSTEM_MESSAGE, user)
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
                ineffective_CoT = self.complete_quality(_SYSTEM_MESSAGE, user)
            except Exception as exc:  # pragma: no cover - network/API dependent
                logger.warning("ineffective summary failed: %s", exc)
                ineffective_CoT = "; ".join(f.name for f in sample)

        effective_CoT = effective_CoT[:2000]
        ineffective_CoT = ineffective_CoT[:2000]
        return FeedbackSummary(effective=effective_CoT, ineffective=ineffective_CoT)

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """A single chat completion; temperature defaults to a random gen value."""
        if temperature is None:
            temperature = random.choice(self.cfg.llm.gen_temperatures)
        return call_llm_retry(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model or self.cfg.llm.model,
            temp=temperature,
        )

    def complete_quality(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """A chat completion from a quality-checker agent (fixed temperature)."""
        return call_llm_retry(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model or self.cfg.llm.quality_model,
            temp=temperature if temperature is not None else self.cfg.llm.quality_temperature,
        )

    def complete_json(
        self,
        system: str,
        user: str,
        model: BaseModel,
        *,
        temperature: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> T:
        """Ask the model to return a JSON object and coerce it into `model`."""
        parse_output = lambda s: \
            model.model_validate_json(ext_code_block(s))
        return call_llm_retry(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model_name or self.cfg.llm.quality_model,
            temp=temperature if temperature is not None else self.cfg.llm.quality_temperature,
            parse_output=parse_output,
        )