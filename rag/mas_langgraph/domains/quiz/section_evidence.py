"""Section-level evidence status tracking for full-outline coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from .entities import SectionEvidenceStatus
from .generation import log_ok, log_warn

if TYPE_CHECKING:
    from ...core.state import CooperativeState

MIN_CHUNKS_PER_SECTION = 2


def _lo_ids_for_section(state: "CooperativeState", section_id: str) -> List[str]:
    return [
        lo.lo_id
        for lo in state.learning_objectives
        if lo.section_id == section_id
    ]


def _chunks_for_section(state: "CooperativeState", section_id: str) -> List[str]:
    """Chunks from section atoms or explicit section tags — no global fallback."""
    from .blueprint_planner import _chunks_for_atoms

    atom_ids = _atoms_for_section(state, section_id)
    chunk_ids = _chunks_for_atoms(state, atom_ids)
    if chunk_ids:
        return chunk_ids
    ids: List[str] = []
    for cid, chunk in list(state.evidence_pool.items()):
        if getattr(chunk, "duplicate_of", None) is not None:
            continue
        meta = getattr(chunk, "metadata", {}) or {}
        if meta.get("section_id") == section_id:
            ids.append(cid)
    return sorted(ids)


def _atoms_for_section(state: "CooperativeState", section_id: str) -> List[str]:
    return [
        atom.atom_id
        for atom in list(state.knowledge_atoms.values())
        if atom.section_id == section_id
    ]


def update_section_evidence_status(
    state: "CooperativeState",
    *,
    min_chunks: int = MIN_CHUNKS_PER_SECTION,
) -> Dict[str, SectionEvidenceStatus]:
    """Refresh section_evidence_status from outline + pool + atoms."""
    if not hasattr(state, "section_evidence_status"):
        state.section_evidence_status = {}

    for section in state.doc_outline:
        chunk_ids = _chunks_for_section(state, section.section_id)
        atom_ids = _atoms_for_section(state, section.section_id)
        lo_ids = _lo_ids_for_section(state, section.section_id)
        if not lo_ids:
            lo_ids = [f"lo_{section.section_id}"]

        existing = state.section_evidence_status.get(section.section_id)
        attempts = existing.retrieval_attempts if existing else 0
        unsupported = existing.unsupported if existing else False
        reason = existing.reason if existing else None

        has_sufficient = len(chunk_ids) >= min_chunks and bool(atom_ids)
        if len(chunk_ids) == 1 and atom_ids:
            has_sufficient = True
            if not reason:
                reason = "single_chunk_acceptable_for_short_section"

        if not chunk_ids and not unsupported:
            reason = reason or "missing_evidence"
        elif chunk_ids and not atom_ids:
            reason = reason or "missing_atoms"

        status = SectionEvidenceStatus(
            section_id=section.section_id,
            section_title=section.title,
            lo_ids=lo_ids,
            evidence_chunk_ids=chunk_ids,
            atom_ids=atom_ids,
            retrieval_attempts=attempts,
            has_sufficient_evidence=has_sufficient,
            unsupported=unsupported,
            reason=reason,
        )
        state.section_evidence_status[section.section_id] = status

    return state.section_evidence_status


def get_sections_without_evidence(state: "CooperativeState") -> List[SectionEvidenceStatus]:
    """Sections lacking sufficient evidence that are not yet marked unsupported."""
    update_section_evidence_status(state)
    return [
        s
        for s in state.section_evidence_status.values()
        if not s.has_sufficient_evidence and not s.unsupported
    ]


def get_sections_with_sufficient_evidence(state: "CooperativeState") -> List[SectionEvidenceStatus]:
    update_section_evidence_status(state)
    return [s for s in state.section_evidence_status.values() if s.has_sufficient_evidence]


def mark_section_unsupported(
    state: "CooperativeState",
    section_id: str,
    reason: str,
) -> None:
    update_section_evidence_status(state)
    status = state.section_evidence_status.get(section_id)
    if status is None:
        return
    status.unsupported = True
    status.has_sufficient_evidence = False
    status.reason = reason
    state.section_evidence_status[section_id] = status


def increment_section_retrieval_attempts(
    state: "CooperativeState",
    section_id: str,
) -> None:
    update_section_evidence_status(state)
    status = state.section_evidence_status.get(section_id)
    if status is None:
        return
    status.retrieval_attempts += 1
    state.section_evidence_status[section_id] = status


def log_section_evidence_summary(state: "CooperativeState") -> None:
    """Emit outline vs evidence coverage logs."""
    update_section_evidence_status(state)
    total = len(state.doc_outline)
    with_evidence = len(get_sections_with_sufficient_evidence(state))
    missing = get_sections_without_evidence(state)
    unsupported = [
        s for s in state.section_evidence_status.values() if s.unsupported
    ]

    log_ok(f"Outline sections: {total}")
    log_ok(f"Learning objectives: {len(state.learning_objectives)}")
    log_ok(f"Initial retrieval covered {with_evidence}/{total} sections")
    if missing:
        labels = ", ".join(f"{s.section_id} {s.section_title}" for s in missing[:8])
        log_warn(f"Missing evidence for sections: {labels}")
    if unsupported:
        labels = ", ".join(f"{s.section_id} {s.section_title}" for s in unsupported[:8])
        log_warn(f"Unsupported after retrieval: {labels}")
    log_ok(f"Evidence coverage: {with_evidence}/{total} sections")
