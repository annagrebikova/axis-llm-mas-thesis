"""MCQ resolver-backed CooperativeState projection for interventions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence

from ...artifacts.channel_state import load_artifact_channels
from ...artifacts.models import ArtifactRef
from ...artifacts.resolver import ArtifactResolver
from ...core.state_cloning import clone_cooperative_state
from ...replay.branch_artifacts import intervention_supplied_refs


def _candidate_version(candidate: Any) -> int:
    return int(getattr(candidate, "revision_count", 0) or 0) + 1


def _as_mapping(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _resolve_id_list(
    ids: Sequence[str],
    resolver: ArtifactResolver,
    *,
    version: int = 1,
) -> List[str]:
    """Rewrite logical IDs through the resolver; drop blocked; dedupe preserving order."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in ids or []:
        ref = ArtifactRef(str(raw), version)
        resolved = resolver.resolve_ref(ref)
        if resolved is None:
            continue
        lid = str(resolved.logical_id)
        if lid in seen:
            continue
        seen.add(lid)
        out.append(lid)
    return out


def _materialize_payload(working: Any, env: Any, artifact_type: str) -> None:
    payload = deepcopy(getattr(env, "payload", None))
    if payload is None:
        return
    ref = env.ref
    if artifact_type == "EvidenceChunk":
        pool = _as_mapping(getattr(working, "evidence_pool", None))
        working.evidence_pool = dict(pool)
        if ref.logical_id not in working.evidence_pool:
            working.evidence_pool[ref.logical_id] = payload
    elif artifact_type == "KnowledgeAtom":
        atoms = _as_mapping(getattr(working, "knowledge_atoms", None))
        working.knowledge_atoms = dict(atoms)
        if ref.logical_id not in working.knowledge_atoms:
            working.knowledge_atoms[ref.logical_id] = payload
    elif artifact_type == "MCQBlueprint":
        bps = _as_mapping(getattr(working, "mcq_blueprints", None))
        working.mcq_blueprints = dict(bps)
        if ref.logical_id not in working.mcq_blueprints:
            working.mcq_blueprints[ref.logical_id] = payload
    elif artifact_type == "CandidateMCQ":
        cands = _as_mapping(getattr(working, "candidate_mcqs", None))
        working.candidate_mcqs = dict(cands)
        if ref.logical_id not in working.candidate_mcqs:
            working.candidate_mcqs[ref.logical_id] = payload


def _apply_blueprint_substitutions(working: Any, resolver: ArtifactResolver) -> None:
    blueprints = _as_mapping(getattr(working, "mcq_blueprints", None))
    working.mcq_blueprints = dict(blueprints)
    for bp_id, bp in list(working.mcq_blueprints.items()):
        evidence_ids = list(getattr(bp, "evidence_chunk_ids", None) or [])
        atom_ids = list(getattr(bp, "target_atom_ids", None) or [])
        new_evidence = _resolve_id_list(evidence_ids, resolver)
        new_atoms = _resolve_id_list(atom_ids, resolver)
        if new_evidence == evidence_ids and new_atoms == atom_ids:
            continue
        try:
            working.mcq_blueprints[bp_id] = replace(
                bp,
                evidence_chunk_ids=new_evidence,
                target_atom_ids=new_atoms,
            )
        except TypeError:
            # Non-dataclass blueprints: mutate a shallow copy when possible
            cloned = deepcopy(bp)
            if hasattr(cloned, "evidence_chunk_ids"):
                cloned.evidence_chunk_ids = new_evidence
            if hasattr(cloned, "target_atom_ids"):
                cloned.target_atom_ids = new_atoms
            working.mcq_blueprints[bp_id] = cloned


def _materialize_substitution_targets(
    working: Any,
    resolver: ArtifactResolver,
    registry: Any,
) -> None:
    """Ensure substitution targets exist in operational collections."""
    for old, new in (resolver.substitutions or {}).items():
        if not resolver.is_available(new):
            continue
        env = registry.get_by_ref(new, resolver.branch_id) or resolver.get_historical(new)
        if env is None:
            continue
        atype = str(getattr(env, "artifact_type", "") or "")
        if atype == "EvidenceChunk":
            pool = _as_mapping(getattr(working, "evidence_pool", None))
            working.evidence_pool = dict(pool)
            old_id = old.logical_id
            if old_id in working.evidence_pool and old_id != new.logical_id:
                working.evidence_pool.pop(old_id, None)
            if new.logical_id not in working.evidence_pool:
                working.evidence_pool[new.logical_id] = deepcopy(env.payload)
        elif atype == "KnowledgeAtom":
            atoms = _as_mapping(getattr(working, "knowledge_atoms", None))
            working.knowledge_atoms = dict(atoms)
            old_id = old.logical_id
            if old_id in working.knowledge_atoms and old_id != new.logical_id:
                working.knowledge_atoms.pop(old_id, None)
            if new.logical_id not in working.knowledge_atoms:
                working.knowledge_atoms[new.logical_id] = deepcopy(env.payload)


def project_cooperative_state_for_resolution(
    state: Mapping[str, Any],
    *,
    node_id: str = "",
) -> Any:
    """Deep-copy CooperativeState and apply resolver availability / substitutions.

    Does not rebuild payloads from envelopes when a ref resolves to itself —
    keeps the checkpoint's operational domain objects. Intervention-supplied
    artifacts absent from operational collections are materialized from the
    branch catalog.
    """
    del node_id
    cs = state.get("cooperative_state")
    if cs is None:
        return None
    working = clone_cooperative_state(cs)
    try:
        registry, resolver = load_artifact_channels(dict(state))
    except Exception:
        return working

    supplied = intervention_supplied_refs(state)
    has_intervention = bool(
        resolver.blocked or resolver.substitutions or supplied
    )
    if not has_intervention and not resolver._availability_enforced:
        return working

    # Drop blocked operational entries
    if resolver.blocked or resolver.substitutions or supplied:
        evidence_pool = _as_mapping(getattr(working, "evidence_pool", None))
        working.evidence_pool = dict(evidence_pool)
        for chunk_id in list(working.evidence_pool):
            ref = ArtifactRef(str(chunk_id), 1)
            if not resolver.is_available(ref):
                working.evidence_pool.pop(chunk_id, None)

        atoms = _as_mapping(getattr(working, "knowledge_atoms", None))
        working.knowledge_atoms = dict(atoms)
        for atom_id in list(working.knowledge_atoms):
            ref = ArtifactRef(str(atom_id), 1)
            if not resolver.is_available(ref):
                working.knowledge_atoms.pop(atom_id, None)

        blueprints = _as_mapping(getattr(working, "mcq_blueprints", None))
        working.mcq_blueprints = dict(blueprints)
        for bp_id in list(working.mcq_blueprints):
            ref = ArtifactRef(str(bp_id), 1)
            if not resolver.is_available(ref):
                working.mcq_blueprints.pop(bp_id, None)

        candidates = _as_mapping(getattr(working, "candidate_mcqs", None))
        working.candidate_mcqs = dict(candidates)
        for q_id, candidate in list(working.candidate_mcqs.items()):
            ref = ArtifactRef(str(q_id), _candidate_version(candidate))
            if not resolver.is_available(ref):
                working.candidate_mcqs.pop(q_id, None)

    if resolver.substitutions:
        _apply_blueprint_substitutions(working, resolver)
        _materialize_substitution_targets(working, resolver, registry)

    # Materialize intervention-supplied available artifacts missing operationally
    for ref in supplied:
        if not resolver.is_available(ref):
            continue
        env = registry.get_by_ref(ref, resolver.branch_id) or resolver.get_historical(ref)
        if env is None:
            continue
        atype = str(getattr(env, "artifact_type", "") or "")
        _materialize_payload(working, env, atype)

    return working


def project_state_for_artifact_resolution(
    state: Mapping[str, Any],
    *,
    node_id: str = "",
) -> MutableMapping[str, Any]:
    """Return a shallow-copied graph state with a filtered CooperativeState."""
    projected = dict(state)
    filtered = project_cooperative_state_for_resolution(state, node_id=node_id)
    if filtered is not None:
        projected["cooperative_state"] = filtered
    return projected
