"""Deterministic MCQBlueprint planning from document coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

from .entities import (
    BLOOM_BY_QUESTION_TYPE,
    MCQ_QUESTION_TYPES,
    MCQBlueprint,
)

if TYPE_CHECKING:
    from ...core.state import CooperativeState


def _normalize_difficulty(level: str) -> str:
    normalized = (level or "").strip().lower()
    if normalized in ("beginner", "basic", "easy", "introductory"):
        return "beginner"
    if normalized in ("advanced", "hard", "expert"):
        return "advanced"
    return "intermediate"


def _infer_bloom_level(question_type: str, lo_text: str = "") -> str:
    if question_type in BLOOM_BY_QUESTION_TYPE:
        return BLOOM_BY_QUESTION_TYPE[question_type]
    lowered = lo_text.lower()
    if any(k in lowered for k in ("compare", "contrast", "differentiate", "analyze")):
        return "Analyzing"
    if any(k in lowered for k in ("evaluate", "justify", "assess")):
        return "Evaluating"
    if any(k in lowered for k in ("apply", "use", "implement", "solve")):
        return "Applying"
    if any(k in lowered for k in ("explain", "describe", "summarize")):
        return "Understanding"
    return "Remembering"


def _generation_goal(question_type: str, lo_text: str, section_title: str) -> str:
    lo_snippet = (lo_text or section_title)[:120]
    templates = {
        "basic_definition": f"Assess recall of a core definition related to: {lo_snippet}",
        "key_characteristic": f"Assess understanding of key characteristics for: {lo_snippet}",
        "difference_identification": f"Assess ability to distinguish concepts for: {lo_snippet}",
        "advantage_disadvantage": f"Assess evaluation of trade-offs for: {lo_snippet}",
        "example_scenario": f"Assess application in a scenario for: {lo_snippet}",
    }
    return templates.get(question_type, f"Generate an MCQ aligned to: {lo_snippet}")


def _atoms_for_lo(state: "CooperativeState", lo_id: str, section_id: str) -> List[str]:
    atom_ids: List[str] = []
    for atom in list(state.knowledge_atoms.values()):
        if atom.lo_id == lo_id:
            atom_ids.append(atom.atom_id)
        elif atom.section_id == section_id and not atom.lo_id:
            atom_ids.append(atom.atom_id)
    return atom_ids


def _chunks_for_atoms(state: "CooperativeState", atom_ids: List[str]) -> List[str]:
    chunk_ids: Set[str] = set()
    for atom_id in atom_ids:
        atom = state.knowledge_atoms.get(atom_id)
        if not atom:
            continue
        for citation in atom.citations or []:
            cid = citation.get("chunk_id")
            if cid and cid in state.evidence_pool:
                chunk_ids.add(cid)
    if chunk_ids:
        return sorted(chunk_ids)

    for atom_id in atom_ids:
        atom = state.knowledge_atoms.get(atom_id)
        if not atom:
            continue
        for cid, chunk in list(state.evidence_pool.items()):
            if atom_id in getattr(chunk, "used_in_atoms", []):
                chunk_ids.add(cid)
    return sorted(chunk_ids)


def _section_tagged_chunks(state: "CooperativeState", section_id: str, limit: int = 5) -> List[str]:
    """Return chunks explicitly tagged to a section — never global fallback."""
    ids: List[str] = []
    for cid, chunk in list(state.evidence_pool.items()):
        if getattr(chunk, "duplicate_of", None) is not None:
            continue
        meta = getattr(chunk, "metadata", {}) or {}
        if meta.get("section_id") == section_id:
            ids.append(cid)
    return ids[:limit]


def create_mcq_blueprints_from_coverage(state: "CooperativeState") -> List[MCQBlueprint]:
    """Plan MCQBlueprint objects from outline, LOs, atoms, and evidence — no generic topics."""
    blueprints: List[MCQBlueprint] = []
    los = list(state.learning_objectives)
    sections = {s.section_id: s for s in state.doc_outline}

    planning_units: List[tuple] = []
    if los:
        for lo in los:
            section = sections.get(lo.section_id)
            section_title = section.title if section else lo.section_id
            planning_units.append((lo.section_id, lo.lo_id, lo.objective_text, lo.difficulty_level, section_title))
    elif state.doc_outline:
        for sec in state.doc_outline:
            synthetic_lo = f"lo_{sec.section_id}"
            planning_units.append(
                (sec.section_id, synthetic_lo, sec.content_summary or sec.title, "intermediate", sec.title)
            )

    for idx, (section_id, lo_id, lo_text, lo_difficulty, section_title) in enumerate(planning_units):
        if (section_title or "").strip().lower() == "general":
            continue

        question_type = MCQ_QUESTION_TYPES[idx % len(MCQ_QUESTION_TYPES)]
        difficulty = _normalize_difficulty(lo_difficulty)
        bloom_level = _infer_bloom_level(question_type, lo_text)
        atom_ids = _atoms_for_lo(state, lo_id, section_id)
        chunk_ids = _chunks_for_atoms(state, atom_ids)
        if not chunk_ids:
            chunk_ids = _section_tagged_chunks(state, section_id)

        if not atom_ids or not chunk_ids:
            continue

        blueprint_id = state.generate_id("bp")
        blueprints.append(
            MCQBlueprint(
                blueprint_id=blueprint_id,
                section_id=section_id,
                lo_id=lo_id,
                target_atom_ids=atom_ids,
                evidence_chunk_ids=chunk_ids,
                question_type=question_type,
                difficulty=difficulty,
                bloom_level=bloom_level,
                generation_goal=_generation_goal(question_type, lo_text, section_title),
                status="planned",
            )
        )

    return blueprints
