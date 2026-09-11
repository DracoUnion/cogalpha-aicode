"""All dataclass and pydantic model definitions for CogAlpha.

Single home for every data model in the framework:

  - configuration models (pydantic-settings / pydantic)            <- config.py
  - domain + quality / feedback models (pydantic)                  <- schemas.py
  - the evolution-search result dataclass                          <- search.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class LLMSettings(BaseSettings):
    """OpenAI-compatible endpoint settings.

    Project-specific ``COGALPHA_*`` env vars take precedence; the standard
    ``OPENAI_BASE_URL`` / ``OPENAI_CHAT_MODEL`` / ``OPENAI_API_KEY`` vars act
    as fallbacks so the standard OpenAI client env vars work unchanged.
    """

    model_config = SettingsConfigDict(env_prefix="COGALPHA_", env_file=".env", extra="ignore")

    api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("COGALPHA_API_KEY", "OPENAI_API_KEY"),
        description="OpenAI API key (COGALPHA_API_KEY overrides OPENAI_API_KEY).",
    )
    base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("COGALPHA_BASE_URL", "OPENAI_BASE_URL"),
        description="Custom base URL for a compatible endpoint (COGALPHA_BASE_URL overrides OPENAI_BASE_URL).",
    )
    model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("COGALPHA_MODEL", "OPENAI_CHAT_MODEL"),
        description="Default generation/evolution model (COGALPHA_MODEL overrides OPENAI_CHAT_MODEL).",
    )
    quality_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("COGALPHA_QUALITY_MODEL", "OPENAI_CHAT_MODEL"),
        description="Model used by the multi-agent quality checker (COGALPHA_QUALITY_MODEL overrides OPENAI_CHAT_MODEL).",
    )
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
    generations_per_evo: int = Field(default=8, description="Generations per sub-loop (paper: 8).")
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
    init_threads: int = 8
    breed_threads: int = 8


# --------------------------------------------------------------------------- #
# Factors
# --------------------------------------------------------------------------- #
class Factor(BaseModel):
    """A single alpha factor: its code, metadata, and measured fitness."""

    name: str
    code: str
    theme: str = ""        # mechanism/agent theme that generated it
    level: str = ""        # level (e.g. "Level III - Price-Volume Dynamics")
    agent_id: str = ""     # agent template file stem
    generation: int = 0
    source: str = "generated"  # generated | repaired | logic_improved | mutation | crossover
    docstring: str = ""

    # Fitness metrics.
    ic: Optional[float] = None
    rank_ic: Optional[float] = None
    icir: Optional[float] = None
    rank_icir: Optional[float] = None
    mi: Optional[float] = None

    qualified: bool = False
    elite: bool = False
    nan_ratio: float = 0.0
    executable: bool = False
    accepted_by_judge: bool = True

    @property
    def metrics(self) -> Dict[str, Optional[float]]:
        return {
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "icir": self.icir,
            "rank_icir": self.rank_icir,
            "mi": self.mi,
        }

    def set_metrics(self, metrics: Dict[str, float]) -> None:
        self.ic = metrics.get("ic")
        self.rank_ic = metrics.get("rank_ic")
        self.icir = metrics.get("icir")
        self.rank_icir = metrics.get("rank_icir")
        self.mi = metrics.get("mi")

    def summary(self) -> str:
        return (
            f"factor {self.name}: IC={self.ic:.4f} RankIC={self.rank_ic:.4f} "
            f"ICIR={self.icir:.3f} RankICIR={self.rank_icir:.3f} MI={self.mi:.4f} "
            f"qualified={self.qualified} elite={self.elite}"
        )


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
class ParsedFunction(BaseModel):
    name: str
    code: str
    docstring: str = ""


# --------------------------------------------------------------------------- #
# Quality-checker structured outputs
# --------------------------------------------------------------------------- #
class QualityResult(BaseModel):
    status: Literal["correct", "needs adjustments"] = "correct"
    issues: List[str] = Field(default_factory=list)
    corrected_code: str = ""


class JudgeResult(BaseModel):
    practical_soundness: str = "no analysis"
    recommendation: Literal["Accept", "Reject"] = "Reject"
    feedback: str = ""


class RepairResult(BaseModel):
    revised_code: str = ""
    notes: str = ""


class LogicImproveResult(BaseModel):
    revised_code: str = ""
    notes: str = ""


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
class FactorAnalysis(BaseModel):
    name: str
    one_clear_idea: str = ""
    short_formula: str = ""
    efficiency_analysis: str = ""   # or failure_analysis

    def chain_line(self) -> str:
        return (
            f"- **{self.name}**: idea='{self.one_clear_idea}' formula='{self.short_formula}' "
            f"eff='{self.efficiency_analysis}'"
        )


class FeedbackSummary(BaseModel):
    """CoT-style summaries injected into the next generation's prompts."""

    effective: str = ""
    ineffective: str = ""


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
@dataclass
class SearchResult:
    candidates: List[Factor] = field(default_factory=list)
    elite: List[Factor] = field(default_factory=list)
    history: Dict[str, List[float]] = field(default_factory=dict)

    def best(self, key: str = "ic", k: int = 10) -> List[Factor]:
        # Imported lazily so `models` never imports `utils` at module load
        # (utils itself imports Factor from this module).
        from .utils import rank_factors

        return rank_factors(self.elite or self.candidates, key)[:k]