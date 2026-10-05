from .entities import (
    CandidateQuestion,
    CoveragePlan,
    CoverageStats,
    DocSection,
    EvidenceChunk,
    KnowledgeAtom,
    LearningObjective,
    QualityMetrics,
    QuizPack,
    RejectedQuestion,
    ValidatedQuestion,
    WorkflowStepLog,
)
from .outcomes import CriterionValue, DecisionArtifact, OutcomeAssessment, OutcomeEvaluator
from .privileges import Privilege, agent_can
from .protocols import NeedsEvaluatorProtocol, StateProtocol, WorkflowOrchestratorProtocol
from .runtime import AxisCaseDefinition, CaseDefinition, WorkflowRuntime
from .state_base import BaseCooperativeState
from .workflow_executor import WorkflowExecutor

__all__ = [
    "BaseCooperativeState",
    "CandidateQuestion",
    "AxisCaseDefinition",
    "CaseDefinition",
    "CoveragePlan",
    "CoverageStats",
    "CriterionValue",
    "DecisionArtifact",
    "DocSection",
    "EvidenceChunk",
    "KnowledgeAtom",
    "LearningObjective",
    "NeedsEvaluatorProtocol",
    "OutcomeAssessment",
    "OutcomeEvaluator",
    "Privilege",
    "QualityMetrics",
    "QuizPack",
    "RejectedQuestion",
    "StateProtocol",
    "ValidatedQuestion",
    "WorkflowExecutor",
    "WorkflowOrchestratorProtocol",
    "WorkflowRuntime",
    "WorkflowStepLog",
    "agent_can",
    "CooperativeState",
]


def __getattr__(name: str):
    """Lazy export of MCQ ``CooperativeState`` to keep generic ``core`` import clean."""
    if name == "CooperativeState":
        from ..cases.mcq.cooperative_state import CooperativeState

        return CooperativeState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
