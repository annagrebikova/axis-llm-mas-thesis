"""MCQ-specific provenance input bindings — no generic provenance imports of MCQ types."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from ...artifacts.models import ArtifactRef
from ...artifacts.registry import ArtifactRegistry
from ...artifacts.resolver import ArtifactResolver
from ...provenance.models import ArtifactBinding
from .artifacts import candidate_artifact_version


def _latest_ref(
    registry: ArtifactRegistry,
    logical_id: str,
    *,
    branch_id: str,
    version: Optional[int] = None,
) -> Optional[ArtifactRef]:
    if version is not None:
        ref = ArtifactRef(str(logical_id), int(version))
        if registry.exists(ref, branch_id):
            return ref
        # Fall back to requested version even if not yet in catalog (binding intent)
        return ref
    env = registry.latest(str(logical_id), branch_id)
    if env is not None:
        return env.ref
    return ArtifactRef(str(logical_id), 1)


def _bind(
    bindings: List[ArtifactBinding],
    seen: Set[tuple],
    ref: Optional[ArtifactRef],
    relation: str,
    *,
    resolver: Optional[ArtifactResolver] = None,
) -> None:
    if ref is None:
        return
    if resolver is not None:
        resolved = resolver.resolve_ref(ref)
        if resolved is None:
            return
        ref = resolved
    key = (ref.logical_id, ref.version, relation)
    if key in seen:
        return
    seen.add(key)
    bindings.append(ArtifactBinding(artifact_ref=ref, relation=relation))


def _candidate_ref(candidate: Any) -> ArtifactRef:
    return ArtifactRef(str(candidate.q_id), candidate_artifact_version(candidate))


def _report_version(report: Any, candidate: Any) -> int:
    return int(getattr(report, "candidate_version", None) or candidate_artifact_version(candidate))


def _bind_blueprint_context(
    bindings: List[ArtifactBinding],
    seen: Set[tuple],
    registry: ArtifactRegistry,
    state: Any,
    blueprint: Any,
    *,
    branch_id: str,
    blueprint_relation: str = "generation_blueprint",
    atom_relation: str = "knowledge_atom",
    evidence_relation: str = "supporting_evidence",
    lo_relation: str = "learning_objective",
    section_relation: str = "document_section",
) -> None:
    if blueprint is None:
        return
    bp_ref = _latest_ref(
        registry, getattr(blueprint, "blueprint_id", ""), branch_id=branch_id, version=1
    )
    _bind(bindings, seen, bp_ref, blueprint_relation)
    for atom_id in getattr(blueprint, "target_atom_ids", []) or []:
        _bind(
            bindings,
            seen,
            _latest_ref(registry, atom_id, branch_id=branch_id, version=1),
            atom_relation,
        )
    for chunk_id in getattr(blueprint, "evidence_chunk_ids", []) or []:
        _bind(
            bindings,
            seen,
            _latest_ref(registry, chunk_id, branch_id=branch_id, version=1),
            evidence_relation,
        )
    lo_id = getattr(blueprint, "lo_id", None)
    if lo_id:
        _bind(
            bindings,
            seen,
            _latest_ref(registry, lo_id, branch_id=branch_id, version=1),
            lo_relation,
        )
    section_id = getattr(blueprint, "section_id", None)
    if section_id:
        _bind(
            bindings,
            seen,
            _latest_ref(registry, section_id, branch_id=branch_id, version=1),
            section_relation,
        )


def input_bindings_for_node(
    node_id: str,
    state: Any,
    registry: ArtifactRegistry,
    *,
    branch_id: str = "factual",
    resolver: Optional[ArtifactResolver] = None,
) -> Sequence[ArtifactBinding]:
    """Resolve exact ArtifactRef bindings actually used by ``node_id``.

    Availability alone is insufficient — bindings follow structured domain fields.
    When ``resolver`` is provided, each binding ref is resolved (substitutions /
    blocks) so REPLACE/REMOVE are reflected in provenance.
    """
    cs = state.get("cooperative_state") if isinstance(state, dict) else state
    if cs is None:
        return ()
    bindings: List[ArtifactBinding] = []
    seen: Set[tuple] = set()
    blueprints = getattr(cs, "mcq_blueprints", {}) or {}
    candidates = getattr(cs, "candidate_mcqs", {}) or {}

    if node_id == "plan_mcq_blueprints":
        for section in getattr(cs, "doc_outline", []) or []:
            _bind(
                bindings,
                seen,
                _latest_ref(registry, section.section_id, branch_id=branch_id, version=1),
                "document_section",
            )
        for lo in getattr(cs, "learning_objectives", []) or []:
            _bind(
                bindings,
                seen,
                _latest_ref(registry, lo.lo_id, branch_id=branch_id, version=1),
                "learning_objective",
            )
        for atom in (getattr(cs, "knowledge_atoms", {}) or {}).values():
            _bind(
                bindings,
                seen,
                _latest_ref(registry, atom.atom_id, branch_id=branch_id, version=1),
                "planning_atom",
            )
        for chunk in (getattr(cs, "evidence_pool", {}) or {}).values():
            if getattr(chunk, "duplicate_of", None) is not None:
                continue
            _bind(
                bindings,
                seen,
                _latest_ref(registry, chunk.chunk_id, branch_id=branch_id, version=1),
                "planning_evidence",
            )

    elif node_id in (
        "generate_candidate_mcqs_from_blueprints",
        "generate_repair_blueprints",
    ):
        planned = [
            bp for bp in blueprints.values() if getattr(bp, "status", "") == "planned"
        ]
        # Generation reads planned blueprints; repair creates new ones from gaps
        for bp in planned if node_id.startswith("generate_candidate") else blueprints.values():
            _bind_blueprint_context(
                bindings, seen, registry, cs, bp, branch_id=branch_id
            )

    elif node_id == "verify_candidate_mcqs_against_evidence":
        for candidate in candidates.values():
            if getattr(candidate, "status", "") == "rejected":
                continue
            _bind(bindings, seen, _candidate_ref(candidate), "candidate_question")
            bp = blueprints.get(getattr(candidate, "blueprint_id", ""))
            _bind_blueprint_context(
                bindings,
                seen,
                registry,
                cs,
                bp,
                branch_id=branch_id,
                blueprint_relation="verification_blueprint",
                atom_relation="grounding_atom",
                evidence_relation="supporting_evidence",
            )

    elif node_id == "cross_cove_verify_candidate_mcqs":
        evidence_reports = getattr(cs, "evidence_reports", {}) or {}
        for candidate in candidates.values():
            if getattr(candidate, "status", "") == "rejected":
                continue
            _bind(bindings, seen, _candidate_ref(candidate), "candidate_question")
            report = evidence_reports.get(candidate.q_id)
            if report is not None:
                ver = _report_version(report, candidate)
                if ver == candidate_artifact_version(candidate):
                    _bind(
                        bindings,
                        seen,
                        ArtifactRef(f"evidence-report:{candidate.q_id}", ver),
                        "evidence_verification_gate",
                    )

    elif node_id == "apply_acceptance_gate":
        evidence_reports = getattr(cs, "evidence_reports", {}) or {}
        cross_cove = getattr(cs, "cross_cove_reports", {}) or {}
        for candidate in candidates.values():
            _bind(bindings, seen, _candidate_ref(candidate), "candidate_question")
            bp = blueprints.get(getattr(candidate, "blueprint_id", ""))
            if bp is not None:
                _bind(
                    bindings,
                    seen,
                    _latest_ref(
                        registry, bp.blueprint_id, branch_id=branch_id, version=1
                    ),
                    "blueprint_constraints",
                )
            er = evidence_reports.get(candidate.q_id)
            if er is not None and _report_version(er, candidate) == candidate_artifact_version(
                candidate
            ):
                _bind(
                    bindings,
                    seen,
                    ArtifactRef(f"evidence-report:{candidate.q_id}", _report_version(er, candidate)),
                    "evidence_report",
                )
            cr = cross_cove.get(candidate.q_id)
            if cr is not None and _report_version(cr, candidate) == candidate_artifact_version(
                candidate
            ):
                _bind(
                    bindings,
                    seen,
                    ArtifactRef(
                        f"cross-cove-report:{candidate.q_id}",
                        _report_version(cr, candidate),
                    ),
                    "cross_cove_report",
                )

    elif node_id == "revise_candidate_mcqs":
        evidence_reports = getattr(cs, "evidence_reports", {}) or {}
        cross_cove = getattr(cs, "cross_cove_reports", {}) or {}
        decisions = getattr(cs, "supervisor_decisions", {}) or {}
        for candidate in candidates.values():
            if getattr(candidate, "status", "") != "needs_revision":
                continue
            if "max_revisions_reached" in (getattr(candidate, "flags", []) or []):
                continue
            _bind(bindings, seen, _candidate_ref(candidate), "candidate_to_revise")
            bp = blueprints.get(getattr(candidate, "blueprint_id", ""))
            _bind_blueprint_context(
                bindings,
                seen,
                registry,
                cs,
                bp,
                branch_id=branch_id,
                blueprint_relation="revision_blueprint",
                atom_relation="knowledge_atom",
                evidence_relation="supporting_evidence",
            )
            er = evidence_reports.get(candidate.q_id)
            if er is not None and _report_version(er, candidate) == candidate_artifact_version(
                candidate
            ):
                _bind(
                    bindings,
                    seen,
                    ArtifactRef(f"evidence-report:{candidate.q_id}", _report_version(er, candidate)),
                    "evidence_feedback",
                )
            cr = cross_cove.get(candidate.q_id)
            if cr is not None and _report_version(cr, candidate) == candidate_artifact_version(
                candidate
            ):
                _bind(
                    bindings,
                    seen,
                    ArtifactRef(
                        f"cross-cove-report:{candidate.q_id}",
                        _report_version(cr, candidate),
                    ),
                    "cross_cove_feedback",
                )
            dec = decisions.get(candidate.q_id)
            if dec is not None and _report_version(dec, candidate) == candidate_artifact_version(
                candidate
            ):
                _bind(
                    bindings,
                    seen,
                    ArtifactRef(f"decision:{candidate.q_id}", _report_version(dec, candidate)),
                    "revision_decision",
                )

    elif node_id == "package_final_quiz":
        decisions = getattr(cs, "supervisor_decisions", {}) or {}
        for candidate in candidates.values():
            if getattr(candidate, "status", "") != "accepted":
                continue
            _bind(bindings, seen, _candidate_ref(candidate), "accepted_candidate")
            dec = decisions.get(candidate.q_id)
            if dec is not None and getattr(dec, "decision", "") == "accept":
                if _report_version(dec, candidate) == candidate_artifact_version(candidate):
                    _bind(
                        bindings,
                        seen,
                        ArtifactRef(
                            f"decision:{candidate.q_id}",
                            _report_version(dec, candidate),
                        ),
                        "acceptance_decision",
                    )
            bp = blueprints.get(getattr(candidate, "blueprint_id", ""))
            if bp is not None:
                _bind(
                    bindings,
                    seen,
                    _latest_ref(
                        registry, bp.blueprint_id, branch_id=branch_id, version=1
                    ),
                    "packaging_blueprint",
                )

    elif node_id in (
        "atomize_evidence",
        "atomize_new_evidence",
    ):
        for chunk in (getattr(cs, "evidence_pool", {}) or {}).values():
            if getattr(chunk, "duplicate_of", None) is not None:
                continue
            _bind(
                bindings,
                seen,
                _latest_ref(registry, chunk.chunk_id, branch_id=branch_id, version=1),
                "evidence_to_atomize",
            )

    elif node_id in (
        "deduplicate_evidence",
        "curate_evidence",
        "update_section_evidence_status",
        "check_coverage",
        "repair_missing_evidence",
        "initial_retrieve_evidence_for_all_sections",
        "retrieve_evidence",
    ):
        # Consume evidence / outline context when present without inventing producers
        for section in getattr(cs, "doc_outline", []) or []:
            if node_id in (
                "initial_retrieve_evidence_for_all_sections",
                "repair_missing_evidence",
                "update_section_evidence_status",
                "check_coverage",
            ):
                _bind(
                    bindings,
                    seen,
                    _latest_ref(registry, section.section_id, branch_id=branch_id, version=1),
                    "document_section",
                )
        for chunk in (getattr(cs, "evidence_pool", {}) or {}).values():
            if node_id in ("deduplicate_evidence", "curate_evidence"):
                _bind(
                    bindings,
                    seen,
                    _latest_ref(registry, chunk.chunk_id, branch_id=branch_id, version=1),
                    "evidence_to_curate"
                    if node_id == "curate_evidence"
                    else "evidence_to_deduplicate",
                )

    elif node_id == "generate_learning_objectives":
        for section in getattr(cs, "doc_outline", []) or []:
            _bind(
                bindings,
                seen,
                _latest_ref(registry, section.section_id, branch_id=branch_id, version=1),
                "document_section",
            )

    if resolver is None:
        return tuple(bindings)

    resolved: List[ArtifactBinding] = []
    resolved_seen: Set[tuple] = set()
    for binding in bindings:
        new_ref = resolver.resolve_ref(binding.artifact_ref)
        if new_ref is None:
            continue
        key = (new_ref.logical_id, new_ref.version, binding.relation)
        if key in resolved_seen:
            continue
        resolved_seen.add(key)
        resolved.append(
            ArtifactBinding(artifact_ref=new_ref, relation=binding.relation)
        )
    return tuple(resolved)


def compact_state_delta(
    before_counts: Optional[Dict[str, Any]],
    after_counts: Optional[Dict[str, Any]],
    *,
    new_artifact_refs: Sequence[ArtifactRef] = (),
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compact occurrence state_delta without embedding full CooperativeState."""
    delta: Dict[str, Any] = {}
    before_counts = before_counts or {}
    after_counts = after_counts or {}
    keys = set(before_counts) | set(after_counts)
    for key in sorted(keys):
        b = before_counts.get(key)
        a = after_counts.get(key)
        if b != a:
            delta[key] = {"before": b, "after": a}
    if new_artifact_refs:
        delta["new_artifacts"] = [str(r) for r in new_artifact_refs]
    if extra:
        delta.update(extra)
    return delta
