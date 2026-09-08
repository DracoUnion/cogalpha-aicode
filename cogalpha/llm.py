"""Thin wrapper around the OpenAI Python SDK.

CogAlpha calls the OpenAI-compatible chat-completions API. Generation and
evolution agents sample a temperature uniformly from the configured set;
quality-checker agents use a fixed temperature. Structured outputs (for the
quality-checker / summary agents) are requested as JSON objects and coerced
into pydantic models.
"""

from __future__ import annotations

import json
import logging
import random
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel

from .models import CogAlphaConfig
from .utils import ext_code_block
from .openai import call_llm_retry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, cfg: CogAlphaConfig) -> None:
        from openai import OpenAI

        llm = cfg.llm
        self._client = OpenAI(
            api_key=llm.api_key,
            base_url=llm.base_url,
            timeout=llm.timeout,
            max_retries=llm.max_retries,
        )
        self.cfg = cfg
        self.model = llm.model
        self.quality_model = llm.quality_model
        self.max_tokens = llm.max_tokens
        self.gen_temperatures = llm.gen_temperatures
        self.quality_temperature = llm.quality_temperature

    # ------------------------------------------------------------------ #
    # Low level
    # ------------------------------------------------------------------ #
    def _chat(
        self,
        messages: List[dict],
        model: str,
        temperature: float,
        response_format: Optional[dict] = None,
    ) -> str:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        return content.strip()



