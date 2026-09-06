"""Pydantic data models shared across CogAlpha.

These capture the objects that flow through the framework — factors, their
fitness, LLM structured outputs from the quality-checker agents, and the
feedback summaries fed back into generation.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


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
