"""Tool names per agent — edit when adding tools; implementations live in mas_langgraph/tools/."""

from typing import Dict, List

# Maps agent_id -> tool names (privilege-filtered at runtime in full implementation)
TOOLS_BY_AGENT: Dict[str, List[str]] = {
    "structure_outline": [
        "get_outline",
        "build_outline_section",
        "set_full_outline",
        "build_outline_from_document",
        "get_state_summary",
    ],
    "curriculum_lo": [
        "get_learning_objectives",
        "add_learning_objective",
        "generate_learning_objectives_for_outline",
        "get_state_summary",
    ],
    "retriever": [
        "retrieve_chunks",
        "get_evidence_chunk",
        "get_state_summary",
    ],
    "evidence_curator": [
        "get_evidence_chunk",
        "update_evidence_chunk",
        "evaluate_chunk_quality",
        "evaluate_all_chunks_quality",
        "link_chunk_to_atom",
        "link_chunk_to_question",
        "mark_duplicate",
        "curate_evidence",
        "get_state_summary",
    ],
    "atomizer": [
        "add_knowledge_atom",
        "get_atom",
        "bind_atom_to_lo",
        "atomize_chunks",
        "link_chunk_to_atom",
        "get_state_summary",
    ],
    "quiz_generator": [
        "add_candidate_question",
        "get_candidate_questions",
        "generate_quiz_draft",
        "get_coverage",
        "get_state_summary",
    ],
    "verifier": [
        "get_candidate_questions",
        "verify_candidate_question",
        "verify_all_candidates",
        "verify_statement",
        "validate_question",
        "reject_question",
        "get_state_summary",
    ],
    "coverage_controller": [
        "get_coverage",
        "get_next_targets",
        "get_quality_metrics",
        "update_quality_metrics",
        "get_state_summary",
    ],
    "quality_duplicate_detector": [
        "detect_duplicate_chunks",
        "detect_duplicate_questions",
        "get_duplicate_report",
        "get_state_summary",
    ],
    "packager_export": [
        "create_quiz_pack",
        "add_questions_to_pack",
        "export_pack",
        "get_state_summary",
    ],
}
