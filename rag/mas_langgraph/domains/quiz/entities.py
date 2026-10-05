"""Literature-grounded MCQ datamodel for PDF-grounded quiz generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ...artifacts.models import ArtifactRef

# Controlled question-type vocabulary for MCQBlueprint planning.
MCQ_QUESTION_TYPES = (
    "basic_definition",
    "key_characteristic",
    "difference_identification",
    "advantage_disadvantage",
    "example_scenario",
)

BLOOM_BY_QUESTION_TYPE = {
    "basic_definition": "Remembering",
    "key_characteristic": "Understanding",
    "difference_identification": "Analyzing",
    "advantage_disadvantage": "Evaluating",
    "example_scenario": "Applying",
}


@dataclass
class SourceRef:
    doc_id: str
    page: Optional[int]
    chunk_id: Optional[str]
    atom_id: Optional[str] = None
    quote: Optional[str] = None


@dataclass
class MCQBlueprint:
    blueprint_id: str
    section_id: str
    lo_id: str
    target_atom_ids: List[str]
    evidence_chunk_ids: List[str]
    question_type: str
    difficulty: str
    bloom_level: str
    generation_goal: str
    status: str = "planned"


@dataclass
class AnswerOption:
    option_id: str
    text: str
    is_correct: bool
    explanation: Optional[str] = None
    misconception: Optional[str] = None
    source_refs: List[SourceRef] = field(default_factory=list)


@dataclass
class CandidateMCQ:
    q_id: str
    blueprint_id: str
    stem: str
    options: List[AnswerOption]
    correct_option_id: str
    answer_explanation: str
    source_refs: List[SourceRef] = field(default_factory=list)
    status: str = "draft"
    revision_count: int = 0
    flags: List[str] = field(default_factory=list)


@dataclass
class LanguageCheckReport:
    q_id: str
    clear: bool
    readable: bool
    grade_level_ok: Optional[bool] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class EvidenceCheckReport:
    q_id: str
    supported_by_pdf: bool
    unsupported_claims: List[str]
    source_refs: List[SourceRef]
    confidence: float
    candidate_version: int = 1
    answer_supported: bool = False
    stem_supported: bool = False
    distractors_valid: bool = False
    explanation_supported: bool = False
    required_fixes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class CrossCoVeReport:
    q_id: str
    generator_answer: str
    verifier_answers: List[str]
    majority_answer: str
    passed: bool
    candidate_version: int = 1
    confidence: float = 0.0
    verifier_model: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class SupervisorDecision:
    q_id: str
    decision: str  # accept, revise, reject
    reasons: List[str]
    candidate_version: int = 1
    required_fixes: List[str] = field(default_factory=list)
    # Exact OutcomeAssessment that produced this decision (Prompt 8).
    assessment_ref: Optional["ArtifactRef"] = None


@dataclass
class SectionEvidenceStatus:
    section_id: str
    section_title: str
    lo_ids: List[str] = field(default_factory=list)
    evidence_chunk_ids: List[str] = field(default_factory=list)
    atom_ids: List[str] = field(default_factory=list)
    retrieval_attempts: int = 0
    has_sufficient_evidence: bool = False
    unsupported: bool = False
    reason: Optional[str] = None


@dataclass
class RevisionRecord:
    q_id: str
    revision_number: int
    previous_stem: str
    previous_options: List[AnswerOption]
    reason: str
    changed_fields: List[str]
    source_report_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
