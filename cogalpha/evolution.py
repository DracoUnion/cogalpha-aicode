"""Thinking evolution agents.

Implements the mutation and crossover operations from the paper, executed by
the mutation / crossover sub-agents, plus the crossover-then-mutation variant.
The evolution templates take `{intro}`, `{extra_guidance}`, and the parent
factor code as placeholders.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from . import executor
from .models import CogAlphaConfig
from .llm_client import LLMClient
from .prompt_loader import PromptLibrary
from .models import ParsedFunction

logger = logging.getLogger(__name__)

EVOLUTION_TYPES = ("mutation", "crossover", "crossover_then_mutation")


def _intro(columns_desc: str, columns_num: int, num_per_request: int, forecast_horizon: int) -> str:
    return (
        f"Below is the schema of the input DataFrame and a list of {columns_num} existing factors:\n\n"
        f"{columns_desc}\n\n"
        f"Please generate new factor functions to forecast {forecast_horizon}-day forward returns."
    )


def _mutation_prompt(
    lib: PromptLibrary,
    intro: str,
    original_code: str,
    extra_guidance: str = "",
) -> str:
    return lib.build_evolution_prompt(
        "mutation_agent",
        intro=intro,
        original_factor_code=original_code,
        extra_guidance=extra_guidance,
    )


def _crossover_prompt(
    lib: PromptLibrary,
    intro: str,
    parent_1: str,
    parent_2: str,
    extra_guidance: str = "",
) -> str:
    return lib.build_evolution_prompt(
        "crossover_agent",
        intro=intro,
        parent_factor_1_code=parent_1,
        parent_factor_2_code=parent_2,
        extra_guidance=extra_guidance,
    )


def mutate(
    llm: LLMClient,
    lib: PromptLibrary,
    columns_desc: str,
    columns_num: int,
    num_per_request: int,
    forecast_horizon: int,
    original_code: str,
    extra_guidance: str = "",
    temperature: Optional[float] = None,
) -> List[ParsedFunction]:
    user = _mutation_prompt(
        lib,
        _intro(columns_desc, columns_num, num_per_request, forecast_horizon),
        original_code,
        extra_guidance,
    )
    try:
        raw = llm.complete(lib.system_message, user, temperature=temperature)
    except Exception as exc:  # pragma: no cover
        logger.error("mutation call failed: %s", exc)
        return []
    return executor.parse_generated_code(raw)


def crossover(
    llm: LLMClient,
    lib: PromptLibrary,
    columns_desc: str,
    columns_num: int,
    num_per_request: int,
    forecast_horizon: int,
    parent_1: str,
    parent_2: str,
    extra_guidance: str = "",
    temperature: Optional[float] = None,
) -> List[ParsedFunction]:
    user = _crossover_prompt(
        lib,
        _intro(columns_desc, columns_num, num_per_request, forecast_horizon),
        parent_1,
        parent_2,
        extra_guidance,
    )
    try:
        raw = llm.complete(lib.system_message, user, temperature=temperature)
    except Exception as exc:  # pragma: no cover
        logger.error("crossover call failed: %s", exc)
        return []
    return executor.parse_generated_code(raw)