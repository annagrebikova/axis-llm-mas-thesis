"""Compatibility adapters — re-export existing quiz adapters + versioning helpers."""

from __future__ import annotations

# Preserve existing adapter behavior
from ...domains.quiz.adapters import (  # noqa: F401
    candidate_mcq_to_candidate_question,
    candidate_question_to_candidate_mcq,
)
from .artifacts import (  # noqa: F401
    candidate_artifact_version,
    revise_candidate_version,
    unwrap,
    wrap_blueprint,
    wrap_candidate_mcq,
    wrap_cross_cove_report,
    wrap_doc_section,
    wrap_evidence_chunk,
    wrap_evidence_report,
    wrap_knowledge_atom,
    wrap_learning_objective,
    wrap_payload,
    wrap_quiz_pack,
    wrap_supervisor_decision,
)
from .case_definition import MCQCaseDefinition  # noqa: F401
