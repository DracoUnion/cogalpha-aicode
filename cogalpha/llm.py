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
            temperature = random.choice(self.gen_temperatures)
        return call_llm_retry(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model or self.model,
            temperature=temperature,
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
            model=model or self.quality_model,
            temperature=temperature if temperature is not None else self.quality_temperature,
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
            model=model_name or self.quality_model,
            temperature=temperature if temperature is not None else self.quality_temperature,
            parse_output=parse_output,
        )

