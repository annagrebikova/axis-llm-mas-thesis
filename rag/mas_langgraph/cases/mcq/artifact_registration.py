"""Register MCQ domain payloads into the generic artifact catalog."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ...artifacts.models import ArtifactRef, DerivedFrom
from ...artifacts.registry import ArtifactRegistry
from ...artifacts.resolver import ArtifactResolver
from .artifacts import (
    candidate_artifact_version,
    wrap_blueprint,
    wrap_candidate_mcq,
    wrap_cross_cove_report,
    wrap_doc_section,
    wrap_evidence_chunk,
    wrap_evidence_report,
    wrap_knowledge_atom,
    wrap_learning_objective,
    wrap_outcome_assessment,
    wrap_quiz_pack,
    wrap_supervisor_decision,
)


def _register_and_make_available(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    envelope: Any,
) -> None:
    """Register a newly produced artifact; never overwrite inherited immutables.

    If the branch catalog already holds this ref (e.g. inherited at fork), skip
    store so projected node-local reference rewrites cannot conflict with the
    immutable historical envelope.
    """
    branch_id = getattr(envelope, "branch_id", None) or resolver.branch_id
    existing = registry.get_by_ref(envelope.ref, branch_id)
    if existing is not None:
        resolver.make_available(existing.ref)
        return
    stored = registry.store(envelope)
    resolver.make_available(stored.ref)


def register_doc_sections(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for section in getattr(state, "doc_outline", []) or []:
        env = wrap_doc_section(
            section,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


def register_learning_objectives(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for lo in getattr(state, "learning_objectives", []) or []:
        env = wrap_learning_objective(
            lo,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


def register_evidence_chunks(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for chunk in list((getattr(state, "evidence_pool", {}) or {}).values()):
        env = wrap_evidence_chunk(
            chunk,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


def register_knowledge_atoms(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for atom in list((getattr(state, "knowledge_atoms", {}) or {}).values()):
        env = wrap_knowledge_atom(
            atom,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


def register_mcq_blueprints(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for blueprint in list((getattr(state, "mcq_blueprints", {}) or {}).values()):
        env = wrap_blueprint(
            blueprint,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


def _record_candidate_derivation(
    registry: ArtifactRegistry,
    *,
    logical_id: str,
    new_version: int,
    branch_id: str,
) -> None:
    if new_version <= 1:
        return
    prev_ref = ArtifactRef(logical_id, new_version - 1)
    new_ref = ArtifactRef(logical_id, new_version)
    if registry.get_by_ref(prev_ref, branch_id) is None:
        return
    link = DerivedFrom(new_artifact_ref=new_ref, previous_artifact_ref=prev_ref)
    for existing in registry.derived_from:
        if (
            existing.new_artifact_ref == link.new_artifact_ref
            and existing.previous_artifact_ref == link.previous_artifact_ref
        ):
            return
    registry.derived_from.append(link)


def register_candidate_mcqs(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for candidate in list((getattr(state, "candidate_mcqs", {}) or {}).values()):
        version = candidate_artifact_version(candidate)
        env = wrap_candidate_mcq(
            candidate,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)
        _record_candidate_derivation(
            registry,
            logical_id=str(candidate.q_id),
            new_version=version,
            branch_id=branch_id,
        )


def register_evidence_reports(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for report in list((getattr(state, "evidence_reports", {}) or {}).values()):
        version = int(getattr(report, "candidate_version", 1) or 1)
        env = wrap_evidence_report(
            report,
            version=version,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


def register_cross_cove_reports(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for report in list((getattr(state, "cross_cove_reports", {}) or {}).values()):
        version = int(getattr(report, "candidate_version", 1) or 1)
        env = wrap_cross_cove_report(
            report,
            version=version,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


def register_outcome_assessments(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for assessment in list((getattr(state, "outcome_assessments", {}) or {}).values()):
        env = wrap_outcome_assessment(
            assessment,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


def register_supervisor_decisions(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    # Assessments are produced in the same acceptance occurrence as decisions.
    register_outcome_assessments(
        registry, resolver, state, run_id=run_id, branch_id=branch_id
    )
    for decision in list((getattr(state, "supervisor_decisions", {}) or {}).values()):
        version = int(getattr(decision, "candidate_version", 1) or 1)
        env = wrap_supervisor_decision(
            decision,
            version=version,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


def register_quiz_packs(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    run_id: str,
    branch_id: str,
) -> None:
    for pack in getattr(state, "final_quiz_packs", []) or []:
        env = wrap_quiz_pack(
            pack,
            branch_id=branch_id,
            created_in_run=run_id,
        )
        _register_and_make_available(registry, resolver, env)


_STAGE_HANDLERS = {
    "outline": register_doc_sections,
    "learning_objectives": register_learning_objectives,
    "evidence": register_evidence_chunks,
    "atoms": register_knowledge_atoms,
    "blueprints": register_mcq_blueprints,
    "candidates": register_candidate_mcqs,
    "evidence_reports": register_evidence_reports,
    "cross_cove_reports": register_cross_cove_reports,
    "supervisor_decisions": register_supervisor_decisions,
    "outcome_assessments": register_outcome_assessments,
    "quiz_packs": register_quiz_packs,
}


def register_mcq_artifacts_for_stage(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    state: Any,
    *,
    stage: str,
    run_id: str,
    branch_id: str = "factual",
) -> None:
    """Register observable MCQ artifacts produced at one workflow boundary."""
    handler = _STAGE_HANDLERS.get(stage)
    if handler is None:
        raise KeyError(f"Unknown MCQ artifact registration stage: {stage!r}")
    handler(registry, resolver, state, run_id=run_id, branch_id=branch_id)
