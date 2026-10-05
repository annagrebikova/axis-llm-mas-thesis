"""Agent privilege matrix — edit which role may R/C/U/L/M each entity type."""

from enum import Enum
from typing import Dict, Set


class Privilege(str, Enum):
    R = "R"
    C = "C"
    U = "U"
    L = "L"
    M = "M"


ENTITY_TYPES = (
    "DocSection",
    "LearningObjective",
    "EvidenceChunk",
    "KnowledgeAtom",
    "CandidateQuestion",
    "ValidatedQuestion",
    "RejectedQuestion",
    "CoverageStats",
    "QualityMetrics",
    "QuizPack",
    "CooperativeState",
)

# Agent role id -> entity -> set of privileges
AGENT_PRIVILEGES: Dict[str, Dict[str, Set[str]]] = {
    "structure_outline": {
        "DocSection": {"C", "U"},
        "LearningObjective": {"L"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"L"},
        "CooperativeState": {"U"},
    },
    "curriculum_lo": {
        "DocSection": {"R"},
        "LearningObjective": {"C", "U"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"L"},
        "CooperativeState": {"U"},
    },
    "retriever": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"C"},
        "CooperativeState": {"U"},
    },
    "evidence_curator": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R", "U"},
        "KnowledgeAtom": {"L"},
        "CandidateQuestion": {"L"},
        "ValidatedQuestion": {"L"},
        "RejectedQuestion": {"L"},
        "QualityMetrics": {"U"},
        "CooperativeState": {"U"},
    },
    "atomizer": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"C", "U", "L"},
        "CooperativeState": {"U"},
    },
    "quiz_generator": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R", "L"},
        "KnowledgeAtom": {"R", "L"},
        "CandidateQuestion": {"C", "U"},
        "CoverageStats": {"U"},
        "QualityMetrics": {"U"},
        "CooperativeState": {"U"},
    },
    "verifier": {
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"R"},
        "CandidateQuestion": {"R", "M"},
        "ValidatedQuestion": {"C"},
        "RejectedQuestion": {"C"},
        "CoverageStats": {"U"},
        "QualityMetrics": {"U"},
        "CooperativeState": {"U"},
    },
    "coverage_controller": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"R"},
        "CandidateQuestion": {"R", "U"},
        "ValidatedQuestion": {"R"},
        "RejectedQuestion": {"R", "U"},
        "CoverageStats": {"R", "U"},
        "QualityMetrics": {"R", "U"},
        "CooperativeState": {"U"},
    },
    "quality_duplicate_detector": {
        "EvidenceChunk": {"R", "U"},
        "KnowledgeAtom": {"R"},
        "CandidateQuestion": {"R", "U"},
        "ValidatedQuestion": {"R", "U"},
        "RejectedQuestion": {"R"},
        "CoverageStats": {"R"},
        "QualityMetrics": {"R", "U"},
        "CooperativeState": {"U"},
    },
    "packager_export": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"R"},
        "ValidatedQuestion": {"R"},
        "RejectedQuestion": {"R"},
        "CoverageStats": {"R"},
        "QualityMetrics": {"R"},
        "QuizPack": {"C"},
        "CooperativeState": {"U"},
    },
}
