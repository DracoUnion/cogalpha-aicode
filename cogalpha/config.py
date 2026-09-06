"""Runtime configuration for CogAlpha.

All knobs from the paper's experimental setup are exposed here as pydantic
settings so they can be tuned from the CLI, an env file, or programmatically.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """OpenAI-compatible endpoint settings."""

    model_config = SettingsConfigDict(env_prefix="COGALPHA_", env_file=".env", extra="ignore")

    api_key: Optional[str] = Field(default=None, description="OpenAI API key (or set OPENAI_API_KEY).")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for a compatible endpoint.")
    model: str = Field(default="gpt-4o-mini", description="Default generation/evolution model.")
    quality_model: str = Field(default="gpt-4o-mini", description="Model used by the multi-agent quality checker.")
    max_tokens: int = Field(default=4096, description="Max tokens per completion (paper: 4096).")
    timeout: float = Field(default=120.0, description="Request timeout in seconds.")
    max_retries: int = Field(default=3, description="OpenAI client retry count.")

    # Generation / evolution agents sample temperature uniformly from this set (paper).
    gen_temperatures: List[float] = Field(default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    # Quality checker agents use a fixed temperature (paper: 0.8).
    quality_temperature: float = Field(default=0.8)


class GenerationSettings(BaseModel):
    """Population and evolution-loop sizes from the paper."""

    initial_pool_size: int = Field(default=80, description="Initial pool size (paper: 80).")
    parent_pool_size: int = Field(default=32, description="Parent pool size (paper: 32).")
    child_pool_size: int = Field(default=96, description="Child pool size = 3x parent (paper: 96).")
    num_per_request: int = Field(default=4, description="Factors requested per LLM call.")
    evolution_searches: int = Field(default=3, description="Evolution searches per task agent (paper: 3).")
    generations_per_search: int = Field(default=24, description="Generations per search (paper: 24).")
    sub_loops: int = Field(default=3, description="Sub-loops inside a search (paper: 3).")
    generations_per_sub_loop: int = Field(default=8, description="Generations per sub-loop (paper: 8).")
    inject_every: int = Field(default=2, description="Inject new alphas into the parent pool every N generations.")
    keep_elite_top: int = Field(default=2, description="Keep the top-N elite alphas each generation.")
    carry_elite_top: int = Field(default=2, description="Carry the previous top-N elites into the next generation.")
    max_repair_attempts: int = Field(default=3, description="Max repair attempts before discarding a factor.")
    max_nan_ratio: float = Field(default=0.30, description="Drop factors with NaN ratio above this (paper: 30%).")
    random_state: int = Field(default=42, description="Random seed for reproducibility.")


class FeedbackSettings(BaseModel):
    """Adaptive-generation feedback sampling."""

    top_effective: int = Field(default=2, description="Number of best effective factors to summarize.")
    worst_ineffective: int = Field(default=2, description="Number of worst ineffective factors to summarize.")
    use_effective_feedback: bool = Field(default=True)
    use_ineffective_feedback: bool = Field(default=True)


class EvaluationSettings(BaseModel):
    """Fitness metrics and qualification thresholds."""

    forward_returns: bool = Field(default=True, description="Compute forward returns as labels.")
    # Each metric: (qualified_threshold, elite_threshold).
    ic: Tuple[float, float] = Field(default=(0.006, 0.010))
    rank_ic: Tuple[float, float] = Field(default=(0.006, 0.010))
    icir: Tuple[float, float] = Field(default=(0.10, 0.20))
    rank_icir: Tuple[float, float] = Field(default=(0.10, 0.20))
    # MI (non-linear) threshold, same for qualified/elite by default.
    mi: Tuple[float, float] = Field(default=(0.01, 0.02))


class CogAlphaConfig(BaseModel):
    """Top-level configuration object."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    forecast_horizon: int = Field(default=10, description="Prediction horizon in days (paper: 10-day open returns).")
    prompts_dir: str = Field(default="", description="Path to the prompts/ directory (auto if empty).")
    data_path: str = Field(default="", description="Path to the OHLCV dataset (required for execution/evaluation).")
    output_dir: str = Field(default="./cogalpha_output", description="Where results/pools are persisted.")
    device: str = Field(default="cpu", description="Ignored except for logging; LightGBM runs on CPU.")
    use_llm: bool = Field(default=True, description="Run LLM agents. Set False for a fully offline static-only run.")
