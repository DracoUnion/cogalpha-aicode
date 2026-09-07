"""Agent: drives the LLM to run factor agents.

A single `Agent` owns the prompt construction (rendering `{placeholder}` tokens
from the `prompt_loader` constants) and the `llm.complete` calls for the
generation / mutation / crossover agents, plus the quality-prompt builders.

Exposes:
  - prompt builders: build_generation_prompt / build_quality_prompt /
                     build_evolution_prompt    (render `{placeholder}` tokens)
  - LLM drivers:     generate_code / mutate / crossover    (call `llm.complete`)
  - helpers:         list_agents / _intro
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from . import executor
from .llm_client import LLMClient
from .models import FeedbackSummary, ParsedFunction
from .prompt_loader import (
    _AGENTS,
    _EFFECTIVE_ANALYSIS,
    _EVOLUTION,
    _GENERATION_TEMPLATES,
    _INEFFECTIVE_ANALYSIS,
    _QUALITY,
    _SYSTEM_MESSAGE,
)

logger = logging.getLogger(__name__)


def _render_placeholders(prompt: str, **kw) -> str:
    """Substitute `{placeholder}` tokens in `prompt` via ``re.sub``.

    Every `{name}` token whose name appears as a keyword is replaced with that
    value; unknown tokens are left untouched. `None` values render empty and
    everything else is stringified (so integer placeholders like `{columns_num}`
    work). This is the single substitution point for every agent prompt.
    """
    def _repl(match: re.Match) -> str:
        name = match.group(1)
        if name in kw:
            value = kw[name]
            return "" if value is None else str(value)
        return match.group(0)

    return re.sub(r"{(\w+)}", _repl, prompt)


def _render_cot_block(template: str, cot: str) -> str:
    """Render an optional (``---``-prefixed) CoT analysis block, or '' if empty."""
    if not cot:
        return ""
    return template.replace("{effective_CoT}", cot).replace("{ineffective_CoT}", cot)


class Agent:
    """Holds an LLM client and performs the factor operations.

    References the prompt constants from `prompt_loader` directly and calls
    `llm.complete` for the generation / mutation / crossover agents.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

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
        user = _render_placeholders(
            user,
            columns_desc=columns_desc,
            columns_num=columns_num,
            num_per_request=num_per_request,
            forecast_horizon=forecast_horizon,
        )
        user = user.replace(
            "{effective_block}", _render_cot_block(_EFFECTIVE_ANALYSIS, effective_CoT)
        ).replace(
            "{ineffective_block}", _render_cot_block(_INEFFECTIVE_ANALYSIS, ineffective_CoT)
        )
        return _SYSTEM_MESSAGE, user

    def build_quality_prompt(self, agent_name: str, **kwargs) -> str:
        """Fill one quality-checker template's `{...}` placeholders."""
        return _render_placeholders(_QUALITY[agent_name], **kwargs)

    def build_evolution_prompt(self, agent_name: str, **kwargs) -> str:
        """Fill one evolution template's `{...}` placeholders."""
        return _render_placeholders(_EVOLUTION[agent_name], **kwargs)

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
            raw = self.llm.complete(system, user, temperature=temperature, model=self.llm.model)
        except Exception as exc:  # pragma: no cover - API dependent
            logger.error("generation call for %s failed: %s", agent_id, exc)
            return []
        return executor.parse_generated_code(raw)

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
            raw = self.llm.complete(_SYSTEM_MESSAGE, user, temperature=temperature)
        except Exception as exc:  # pragma: no cover
            logger.error("mutation call failed: %s", exc)
            return []
        return executor.parse_generated_code(raw)

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
            raw = self.llm.complete(_SYSTEM_MESSAGE, user, temperature=temperature)
        except Exception as exc:  # pragma: no cover
            logger.error("crossover call failed: %s", exc)
            return []
        return executor.parse_generated_code(raw)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def list_agents(self) -> List[str]:
        """Sorted ids of the seven-level generation agents."""
        return sorted(_AGENTS.keys())