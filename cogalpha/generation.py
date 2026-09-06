"""Seven-level generation agents.

`PromptLibrary.agents` maps each agent id (e.g. `agent_liquidity`) to its
(level, intro, guidance). This module assembles the generation prompt and calls
the LLM, returning the raw parsed factor functions tagged with the agent's
theme/level metadata.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from . import executor
from .llm_client import LLMClient
from .prompt_loader import PromptLibrary
from .schemas import FeedbackSummary, ParsedFunction

logger = logging.getLogger(__name__)


def list_agents(lib: PromptLibrary) -> List[str]:
    return sorted(lib.agents.keys())


def generate_code(
    llm: LLMClient,
    lib: PromptLibrary,
    agent_id: str,
    columns_desc: str,
    columns_num: int,
    num_per_request: int,
    forecast_horizon: int,
    feedback: Optional[FeedbackSummary] = None,
    temperature: Optional[float] = None,
) -> List[ParsedFunction]:
    """Call the LLM for one agent and parse the returned factor functions."""
    eff = feedback.effective if feedback else ""
    ineff = feedback.ineffective if feedback else ""

    system, user = lib.build_generation_prompt(
        agent_id=agent_id,
        columns_desc=columns_desc,
        columns_num=columns_num,
        num_per_request=num_per_request,
        forecast_horizon=forecast_horizon,
        effective_CoT=eff,
        ineffective_CoT=ineff,
    )
    try:
        raw = llm.complete(
            system, user, temperature=temperature, model=llm.model
        )
    except Exception as exc:  # pragma: no cover - API dependent
        logger.error("generation call for %s failed: %s", agent_id, exc)
        return []
    return executor.parse_generated_code(raw)