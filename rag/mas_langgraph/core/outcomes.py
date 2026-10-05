"""Generic outcome assessment and decision artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ..artifacts.models import ArtifactRef


@dataclass(frozen=True)
class CriterionValue:
    """One measured criterion for an artifact version (component of Gamma_alpha)."""

    name: str
    value: Any
    value_type: str = "bool"
    label: str = ""
    explanation: str = ""
    decision_relevant: bool = True
    # Compatibility alias — prefer value_type.
    expected_type: str = ""

    def __post_init__(self) -> None:
        if self.expected_type and not self.value_type:
            object.__setattr__(self, "value_type", self.expected_type)
        elif self.value_type and not self.expected_type:
            object.__setattr__(self, "expected_type", self.value_type)


@dataclass(frozen=True)
class OutcomeAssessment:
    """Gamma(alpha, x) — per exact artifact-version criterion vector.

    Identity: assessment_ref = ArtifactRef("assessment:{q_id}", version).
    Producer / branch / run identity belongs on ArtifactEnvelope / ProducedBy,
    not as authoritative fields here.
    """

    assessment_ref: ArtifactRef
    target_ref: ArtifactRef
    criteria: tuple
    notes: tuple = ()


@dataclass(frozen=True)
class DecisionArtifact:
    """Generic computational representation of delta_alpha(Gamma_alpha).

    Persisted MCQ terminal decisions remain SupervisorDecision under
    ArtifactRef("decision:{q_id}", version). Do not also register DecisionArtifact.
    """

    decision_ref: ArtifactRef
    target_ref: ArtifactRef
    assessment_ref: ArtifactRef
    decision: str  # accept | revise | reject
    reasons: tuple
    required_fixes: tuple = ()


@runtime_checkable
class OutcomeEvaluator(Protocol):
    """Assess and decide outcomes for a single artifact version."""

    def assess(
        self,
        state: Dict[str, Any],
        artifact_ref: ArtifactRef,
    ) -> OutcomeAssessment: ...

    def decide(
        self,
        assessment: OutcomeAssessment,
    ) -> DecisionArtifact: ...
