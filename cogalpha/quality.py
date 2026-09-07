"""Multi-agent quality checker.

Implements the four quality-checker agents from the paper:

- Code Quality Agent  : static + LLM review of correctness/compliance.
- Code Repair Agent   : fixes execution/static failures.
- Judge Agent         : evaluates logical / technical / economic soundness.
- Logic Improvement Agent : improves factors that fail the judge.

A generated factor passes through: static/LLM quality check -> (repair if
needed) -> judge -> (logic improvement if rejected). After that it is executed
and evaluated by the caller.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from . import executor
from .agent import Agent
from .models import CogAlphaConfig
from .llm_client import LLMClient
from .prompt_loader import PromptLibrary
from .models import JudgeResult, QualityResult

logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS_DEFAULT = 3


class QualityGate:
    """Runs the checker agents over a single factor's code and metadata."""

    def __init__(
        self,
        llm: LLMClient,
        lib: PromptLibrary,
        columns_desc: str,
        columns_num: int,
        cfg: CogAlphaConfig,
        agent: Optional[Agent] = None,
    ) -> None:
        self.llm = llm
        self.lib = lib
        self.agent = agent or Agent(llm, lib)
        self.columns_desc = columns_desc
        self.columns_num = columns_num
        self.cfg = cfg
        self.max_repair = cfg.generation.max_repair_attempts or MAX_REPAIR_ATTEMPTS_DEFAULT

    # ------------------------------------------------------------------ #
    # Static (deterministic) checks
    # ------------------------------------------------------------------ #
    def static_check(self, code: str, name: str) -> List[str]:
        issues = executor.check_code_static(code, name)
        issues += executor.validate_name(code, name)
        return issues

    # ------------------------------------------------------------------ #
    # LLM agents
    # ------------------------------------------------------------------ #
    def judge(self, code: str) -> JudgeResult:
        """Judge Agent: decide Accept/Reject + improvement feedback."""
        prompt = self.agent.build_quality_prompt(
            "judge_agent", new_factor_code=code
        )
        return self.llm.complete_json(
            self.lib.system_message, prompt, JudgeResult
        )

    def code_quality(self, code: str) -> QualityResult:
        """Code Quality Agent: LLM review that complements static checks."""
        prompt = self.agent.build_quality_prompt("code_quality_agent", code=code)
        try:
            raw = self.llm.complete_quality(self.lib.system_message, prompt)
        except Exception as exc:  # pragma: no cover - API dependent
            logger.warning("code_quality LLM call failed: %s", exc)
            return QualityResult(status="correct")
        issues = _quality_issues(raw)
        corrected = _quality_corrected(raw)
        status = "needs adjustments" if issues else "correct"
        return QualityResult(status=status, issues=issues, corrected_code=corrected)

    def repair(self, old_code: str, error: str) -> str:
        """Code Repair Agent: fix an execution/static failure."""
        prompt = self.agent.build_quality_prompt(
            "code_repair_agent",
            columns_num=self.columns_num,
            columns_desc=self.columns_desc,
            old_code=old_code,
            error=error,
        )
        raw = self.llm.complete_quality(self.lib.system_message, prompt)
        funcs = executor.parse_generated_code(raw)
        return funcs[0].code if funcs else old_code

    def logic_improve(self, old_code: str, feedback: str) -> str:
        """Logic Improvement Agent: improve a rejected factor."""
        prompt = self.agent.build_quality_prompt(
            "logic_improvement_agent",
            columns_num=self.columns_num,
            columns_desc=self.columns_desc,
            old_code=old_code,
            dynamic_feedback=feedback,
        )
        raw = self.llm.complete_quality(self.lib.system_message, prompt)
        funcs = executor.parse_generated_code(raw)
        return funcs[0].code if funcs else old_code

    # ------------------------------------------------------------------ #
    # Full gate
    # ------------------------------------------------------------------ #
    def check_code(self, code: str, name: str) -> str:
        """Run quality checks and (optionally) the LLM quality agent.

        Returns the possibly-repaired code. Static violations trigger the
        repair agent; nested-loop/infinite-loop violations are hard failures
        that cannot be auto-repaired (returned unrepaired so the caller may
        discard). When `cfg.use_llm` is False, only the deterministic static
        checks run (no network calls).
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


def _quality_issues(raw: str) -> List[str]:
    lines = [l for l in raw.splitlines()]
    issues = []
    for l in lines:
        s = l.strip()
        if s.startswith("-") or s.startswith("*") or s.startswith("1."):
            issues.append(s.lstrip("-*0123456789. ").strip())
            if len(issues) >= 8:
                break
    return issues


def _quality_corrected(raw: str) -> str:
    funcs = executor.parse_generated_code(raw)
    return funcs[0].code if funcs else ""


def gate_factor(
    gate: QualityGate,
    code: str,
    name: str,
    *,
    use_judge: bool = True,
) -> Optional[str]:
    """Run the full quality pipeline.

    Returns the final acceptable code, or None if the factor was discarded.
    """
    current = code
    # 1) Quality check (+ repair rounds for non-hard failures).
    for attempt in range(gate.max_repair):
        current = gate.check_code(current, name)
        issues = gate.static_check(current, name)
        if not issues:
            break
        if not gate.cfg.use_llm:
            # Offline mode cannot call the repair agent.
            logger.info("Factor %s has static issues and LLM repair is disabled; discarding.", name)
            return None
        if attempt == gate.max_repair - 1:
            logger.info("Factor %s failed quality after %d repairs; discarding.", name, gate.max_repair)
            return None
        # Get an error message to feed the repair agent.
        error = "; ".join(issues[:4]) or "unknown static issue"
        current = gate.repair(current, error)

    # 2) Judge.
    if use_judge and gate.cfg.use_llm:
        from .models import Factor

        try:
            jr = gate.judge(current)
        except Exception as exc:  # pragma: no cover
            logger.warning("judge failed for %s: %s", name, exc)
            jr = JudgeResult(recommendation="Accept")
        if jr.recommendation != "Accept":
            # 3) Logic improvement.
            improved = gate.logic_improve(current, jr.feedback)
            # Re-run static checks on the improved version.
            if not gate.static_check(improved, name):
                current = improved
            else:
                logger.info("Logic-improved factor %s still invalid; keeping original.", name)

    return current