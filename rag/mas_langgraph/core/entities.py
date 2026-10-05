"""Shared dataclasses for the cooperative blackboard."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocSection:
    section_id: str
    title: str
    slide_range: tuple
    importance_weight: float
    content_summary: Optional[str] = None


@dataclass
class LearningObjective:
    lo_id: str
    section_id: str
    objective_text: str
    difficulty_level: str


@dataclass
class KnowledgeAtom:
    atom_id: str
    content: str
    atom_type: str
    citations: List[Dict[str, Any]]
    section_id: Optional[str] = None
    lo_id: Optional[str] = None


@dataclass
class EvidenceChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float = 0.0
    quality_score: float = 0.0
    relevance_tags: List[str] = field(default_factory=list)
    used_in_atoms: List[str] = field(default_factory=list)
    duplicate_of: Optional[str] = None
    flags: List[str] = field(default_factory=list)


@dataclass
class CandidateQuestion:
    q_id: str
    question_text: str
    question_type: str
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    difficulty: str = "intermediate"
    topic_tags: List[str] = field(default_factory=list)
    related_atoms: List[str] = field(default_factory=list)
    evidence_chunks: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    duplicate_of: Optional[str] = None


@dataclass
class ValidatedQuestion:
    q_id: str
    question_text: str
    question_type: str
    correct_answer: str
    difficulty: str
    options: Optional[List[str]] = None
    topic_tags: List[str] = field(default_factory=list)
    related_atoms: List[str] = field(default_factory=list)
    evidence_chunks: List[str] = field(default_factory=list)
    validation_notes: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    duplicate_of: Optional[str] = None


@dataclass
class RejectedQuestion:
    q_id: str
    question_text: str
    rejection_reason: str
    fix_hints: List[str] = field(default_factory=list)
    original_question: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoverageStats:
    topic: str
    difficulty: str
    question_type: str
    count: int = 0
    validated_count: int = 0
    rejected_count: int = 0


@dataclass
class CoveragePlan:
    default_per_difficulty_type: Dict[Tuple[str, str], int] = field(default_factory=dict)
    per_topic: Optional[Dict[str, Dict[Tuple[str, str], int]]] = None
    per_lo: Optional[Dict[str, Dict[Tuple[str, str], int]]] = None


@dataclass
class QualityMetrics:
    rejection_rate: float = 0.0
    duplication_score: float = 0.0
    avg_evidence_strength: float = 0.0
    total_questions: int = 0
    validated_questions: int = 0
    rejected_questions: int = 0


@dataclass
class QuizPack:
    pack_id: str
    title: str
    questions: List[ValidatedQuestion]
    metadata: Dict[str, Any] = field(default_factory=dict)
    coverage_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStepLog:
    """One orchestration step — extend for AXIS-style trace / checkpoints."""

    step: int
    agent_id: Optional[str]
    reason: str
    input_text: Optional[str]
    result_summary: str
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
