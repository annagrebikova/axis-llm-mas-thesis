"""Quiz/RAG domain — MCQ model exports (no orchestrator imports to avoid cycles)."""

from .adapters import candidate_mcq_to_candidate_question, candidate_question_to_candidate_mcq
from .blueprint_planner import create_mcq_blueprints_from_coverage
from .entities import (
    AnswerOption,
    BLOOM_BY_QUESTION_TYPE,
    CandidateMCQ,
    CrossCoVeReport,
    EvidenceCheckReport,
    LanguageCheckReport,
    MCQ_QUESTION_TYPES,
    MCQBlueprint,
    RevisionRecord,
    SourceRef,
    SupervisorDecision,
)
from .export import export_mcqs_for_external_iwf_check
from .generation import (
    build_generation_context_from_blueprint,
    generate_candidate_from_blueprint,
    register_candidate_mcq,
    validate_candidate_mcq_structure,
)
from .cross_cove import (
    build_cross_cove_prompt,
    majority_vote,
    parse_verifier_answer,
    run_cross_cove_for_all_candidates,
    run_cross_cove_for_candidate,
)
from .acceptance_policy import (
    HARD_REJECTION_FLAGS,
    STRUCTURAL_REJECT_FLAGS,
    decide_acceptance,
)
from .coverage import (
    apply_acceptance_gate,
    apply_acceptance_gate_to_all,
    build_coverage_requirements,
    compute_mcq_coverage,
    create_repair_blueprints,
    find_coverage_gaps,
    select_final_mcqs_for_quiz,
)
from .packaging import package_final_quiz
from .revision import (
    build_revision_prompt,
    collect_revision_feedback,
    reset_reports_after_revision,
    revise_all_needed_candidates,
    revise_candidate_mcq,
)
from .verification import (
    build_evidence_verification_context,
    verify_all_candidate_mcqs,
    verify_candidate_against_evidence,
)

__all__ = [
    "AnswerOption",
    "BLOOM_BY_QUESTION_TYPE",
    "CandidateMCQ",
    "CrossCoVeReport",
    "EvidenceCheckReport",
    "LanguageCheckReport",
    "MCQ_QUESTION_TYPES",
    "MCQBlueprint",
    "RevisionRecord",
    "SourceRef",
    "SupervisorDecision",
    "candidate_mcq_to_candidate_question",
    "candidate_question_to_candidate_mcq",
    "build_generation_context_from_blueprint",
    "build_cross_cove_prompt",
    "build_coverage_requirements",
    "build_evidence_verification_context",
    "build_generation_context_from_blueprint",
    "compute_mcq_coverage",
    "create_mcq_blueprints_from_coverage",
    "create_repair_blueprints",
    "export_mcqs_for_external_iwf_check",
    "find_coverage_gaps",
    "generate_candidate_from_blueprint",
    "HARD_REJECTION_FLAGS",
    "STRUCTURAL_REJECT_FLAGS",
    "decide_acceptance",
    "majority_vote",
    "package_final_quiz",
    "parse_verifier_answer",
    "register_candidate_mcq",
    "run_cross_cove_for_all_candidates",
    "run_cross_cove_for_candidate",
    "select_final_mcqs_for_quiz",
    "build_revision_prompt",
    "collect_revision_feedback",
    "reset_reports_after_revision",
    "revise_all_needed_candidates",
    "revise_candidate_mcq",
    "validate_candidate_mcq_structure",
    "verify_all_candidate_mcqs",
    "verify_candidate_against_evidence",
    "apply_acceptance_gate",
    "apply_acceptance_gate_to_all",
]
