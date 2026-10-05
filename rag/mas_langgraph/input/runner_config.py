"""Agent runner dispatch config — edit when adding agents or custom invoke paths."""

from typing import Dict

# Primary tool per agent for deterministic stub runs (no LangChain)
AGENT_PRIMARY_TOOL: Dict[str, str] = {
    "structure_outline": "build_outline_from_document",
    "curriculum_lo": "generate_learning_objectives_for_outline",
    "retriever": "retrieve_chunks",
    "quality_duplicate_detector": "detect_duplicate_chunks",
    "evidence_curator": "curate_evidence",
    "atomizer": "atomize_chunks",
    "quiz_generator": "generate_quiz_draft",
    "verifier": "verify_all_candidates",
    "coverage_controller": "get_next_targets",
    "packager_export": "export_pack",
}

# Optional per-agent invoke overrides: agent_id -> method name on AgentRunner
AGENT_INVOKE_METHODS: Dict[str, str] = {
    "retriever": "retrieve",
    "quiz_generator": "generate_quiz",
    "verifier": "verify_all_candidates",
    "packager_export": "export_quiz_pack",
}
