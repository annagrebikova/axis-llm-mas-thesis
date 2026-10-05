"""Data models for the three-condition explanation experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

from ..artifacts.models import ArtifactRef

ConditionName = Literal["trace", "trace_provenance", "axis"]

EXPERIMENT_SCHEMA_VERSION = 1

EVIDENCE_ACCESS = {
    "trace": {
        "execution_trace": True,
        "provenance": False,
        "counterfactual_replay": False,
    },
    "trace_provenance": {
        "execution_trace": True,
        "provenance": True,
        "counterfactual_replay": False,
    },
    "axis": {
        "execution_trace": True,
        "provenance": True,
        "counterfactual_replay": True,
    },
}


@dataclass(frozen=True)
class AnswerOptionView:
    option_id: str
    text: str
    is_correct: bool = False


@dataclass(frozen=True)
class ExplanationConditionResult:
    condition: ConditionName
    explanation: str
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    raw_evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    status: str = "ok"


@dataclass(frozen=True)
class MCQExplanationComparison:
    run_id: str
    branch_id: str
    candidate_ref: ArtifactRef
    assessment_ref: ArtifactRef
    decision_ref: ArtifactRef
    question_stem: str
    answer_options: Tuple[AnswerOptionView, ...]
    correct_option_id: Optional[str]
    factual_decision: str
    user_question: str
    trace: ExplanationConditionResult
    trace_provenance: ExplanationConditionResult
    axis: ExplanationConditionResult
    blinding: Dict[str, ConditionName] = field(default_factory=dict)
    axis_status: str = "ok"
    axis_causal_evidence_available: bool = False
    counterfactual_count: int = 0
    successful_replay_count: int = 0
    visualization_data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuizExplanationExperiment:
    schema_version: int
    run_id: str
    branch_id: str
    experiment_seed: int
    user_question: str
    cases: Tuple[MCQExplanationComparison, ...]
    evidence_access: Dict[str, Dict[str, bool]] = field(
        default_factory=lambda: dict(EVIDENCE_ACCESS)
    )
    synthesis_model: str = ""
    models: Dict[str, Any] = field(default_factory=dict)
    include_historical_versions: bool = False
    warnings: Tuple[str, ...] = ()
    workflow_llm_diagnostics: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    axis_interrogation_mode: str = "constrained"
    axis_interrogation_policy: str = "counterfactual_only"
