"""Orchestration routing rules — edit order, conditions, and default inputs.

Inputs are derived from document material (outline sections, LOs, coverage gaps)
via input_template keys in orchestrator/material_context.py — not from a user topic.
"""

from typing import Any, Dict, List

ORCHESTRATION_RULES: List[Dict[str, Any]] = [
    {
        "id": "A_outline",
        "condition": "needs_outline",
        "agent": "structure_outline",
        "reason": "No document outline; build outline first.",
        "input": "Build document outline: section_id, title, slide range, importance weight per section.",
    },
    {
        "id": "B_learning_objectives",
        "condition": "needs_learning_objectives",
        "agent": "curriculum_lo",
        "reason": "No or insufficient learning objectives; create LOs.",
        "input": "Create learning objectives for each section with difficulty levels.",
    },
    {
        "id": "C_evidence",
        "condition": "needs_more_evidence",
        "agent": "retriever",
        "reason": "Evidence pool too small; retrieve from next document section.",
        "input_template": "section_query",
    },
    {
        "id": "C_prime_after_retriever",
        "condition": "always",
        "after_agent": "retriever",
        "requires": "has_min_evidence",
        "agent": "quality_duplicate_detector",
        "reason": "Evidence just retrieved; run duplicate detector before curator.",
        "input": "Use detect_duplicate_chunks to flag duplicate evidence chunks in the pool.",
    },
    {
        "id": "C_prime_after_dedup",
        "condition": "always",
        "after_agent": "quality_duplicate_detector",
        "agent": "evidence_curator",
        "reason": "Run curator after duplicate detection to score and tag chunks.",
        "input": "Use curate_evidence to score and tag all chunks in the evidence pool.",
    },
    {
        "id": "E_atomization",
        "condition": "needs_atomization",
        "agent": "atomizer",
        "reason": "Too few knowledge atoms; atomize chunks.",
        "input": "Use atomize_chunks to extract knowledge atoms from all evidence chunks.",
    },
    {
        "id": "D_deduplication",
        "condition": "needs_deduplication",
        "agent": "quality_duplicate_detector",
        "reason": "Duplication score or low-quality chunk ratio too high; run duplicate detector.",
        "input": "Use detect_duplicate_chunks to flag duplicate evidence chunks in the pool.",
        "skip_if_last_agent": ["quality_duplicate_detector"],
        "alternate": {
            "agent": "evidence_curator",
            "reason": "Run curator after duplicate detection.",
            "input": "Use curate_evidence to score and tag all chunks in the evidence pool.",
        },
    },
    {
        "id": "G_verification",
        "condition": "needs_verification",
        "agent": "verifier",
        "reason": "Candidate questions pending; verify against evidence.",
        "input": "Verify next batch of candidate questions against evidence.",
    },
    {
        "id": "F_questions_default",
        "condition": "needs_more_questions",
        "agent": "quiz_generator",
        "reason": "Coverage gap; generate questions for outline section.",
        "input_template": "quiz_target",
        "skip_if_last_agent": ["quiz_generator"],
    },
    {
        "id": "H_high_rejection",
        "condition": "rejection_rate_too_high",
        "agent": "coverage_controller",
        "reason": "Rejection rate too high; check coverage and consider more generation.",
        "input": "Use get_next_targets and get_coverage to analyze gaps and suggest next section focus.",
    },
    {
        "id": "I_coverage_repair",
        "condition": "needs_coverage_repair",
        "agent": "coverage_controller",
        "reason": "Coverage gaps exist; analyze and steer next targets.",
        "input": "Use get_next_targets to list coverage gaps and suggest retrieval or quiz focus.",
    },
    {
        "id": "J_packaging",
        "condition": "ready_for_packaging",
        "agent": "packager_export",
        "reason": "Enough validated questions; export quiz pack.",
        "input_template": "pack_title",
    },
    {
        "id": "Z_done_exported",
        "condition": "has_final_quiz_packs",
        "agent": None,
        "reason": "Workflow complete: quiz pack(s) exported.",
    },
    {
        "id": "Z_stuck",
        "condition": "always",
        "agent": None,
        "reason": "No further action (state may be incomplete or at max iterations).",
    },
]

PROGRESS_METRIC_KEYS = (
    "validated_questions_len",
    "knowledge_atoms_len",
    "canonical_chunks",
    "candidate_questions_len",
)
