"""Information-role findings from one AXIS ReplayComparison (MCQ question content).

Causal support remains ReplayComparison / NOOP / timing. This module only
classifies and verbalizes the *role* of information for participants.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROLE_FOCUS_SHAPING = "focus_shaping"
ROLE_ANSWER_ANCHORING = "answer_anchoring"
ROLE_DISTRACTOR_SHAPING = "distractor_shaping"
ROLE_FORMULATION_SHAPING = "formulation_shaping"
ROLE_NECESSARY_FOR_REPRODUCTION = "necessary_for_reproduction"
ROLE_INFLUENTIAL_NON_DECISIVE = "influential_non_decisive"
ROLE_ROBUST = "robust_to_tested_information"
ROLE_NO_DEMONSTRATED = "no_demonstrated_role"
ROLE_INCONCLUSIVE = "inconclusive"
ROLE_ARTIFACT_ONLY = "artifact_level_only"


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def changed_distractor_count_from_lists(
    *,
    removed: Sequence[str],
    added: Sequence[str],
) -> int:
    """Number of distractor contents replaced (set-aware; order ignored)."""
    rem = {_norm(x).lower() for x in removed if _norm(x)}
    add = {_norm(x).lower() for x in added if _norm(x)}
    if not rem and not add:
        return 0
    return max(len(rem), len(add))


def infer_semantic_intervention_validity(
    *,
    artifact_type: str = "",
    subject_generation_relation: Optional[str] = None,
    intervention_before_generation: Optional[bool] = None,
    boundary_label: str = "",
    provenance_relations: Sequence[Any] = (),
    target_ref: Any = None,
) -> Dict[str, Any]:
    """Classify whether an intervention supports information-level claims.

    EvidenceChunk after atomization: semantic content may persist in KnowledgeAtoms.
    EvidenceChunk before atomize / KnowledgeAtom before planning: information-level OK.
    """
    a_type = str(artifact_type or "").strip()
    boundary = str(boundary_label or "").lower()
    _ = subject_generation_relation

    before_atomize = "before" in boundary and "atomize" in boundary
    after_atomize = ("after" in boundary and "atomize" in boundary) or (
        "before" in boundary
        and any(x in boundary for x in ("plan_mcq", "generate_candidate", "verify"))
        and "atomize" not in boundary
        and a_type in {"EvidenceChunk", "EvidenceChunkPayload"}
    )

    target_key = str(target_ref or "")
    has_atomize_edge = False
    for r in provenance_relations or ():
        relation = str(
            getattr(r, "relation", None)
            or (r.get("relation") if isinstance(r, Mapping) else "")
            or ""
        )
        if relation != "evidence_to_atomize":
            continue
        pref = getattr(r, "artifact_ref", None)
        if pref is None and isinstance(r, Mapping):
            pref = r.get("artifact_ref")
        if pref is not None and (
            str(pref) == target_key
            or str(getattr(pref, "logical_id", "") or "") in target_key
        ):
            has_atomize_edge = True
            break

    if a_type in {"KnowledgeAtom", "KnowledgeAtomPayload"}:
        semantic_absent = bool(
            intervention_before_generation is True
            or (boundary and "before" in boundary and "plan" in boundary)
            or (boundary and "before" in boundary and "generate" in boundary)
        )
        return {
            "intervention_scope": "semantic_information"
            if semantic_absent
            else "artifact_only",
            "semantic_information_absent": semantic_absent,
            "derived_information_may_persist": False,
            "preferred_boundary_hint": "before plan_mcq_blueprints",
        }

    if a_type in {"EvidenceChunk", "EvidenceChunkPayload"}:
        if before_atomize:
            return {
                "intervention_scope": "semantic_information",
                "semantic_information_absent": True,
                "derived_information_may_persist": False,
                "preferred_boundary_hint": "before atomize_evidence",
            }
        if after_atomize or has_atomize_edge or (
            intervention_before_generation is True and not before_atomize
        ):
            return {
                "intervention_scope": "artifact_only",
                "semantic_information_absent": False,
                "derived_information_may_persist": True,
                "preferred_boundary_hint": "before atomize_evidence",
            }
        return {
            "intervention_scope": "artifact_only",
            "semantic_information_absent": False,
            "derived_information_may_persist": True,
            "preferred_boundary_hint": "before atomize_evidence",
        }

    # Unknown / other types: if intervention is before generation and we have
    # no evidence of derived persistence, allow information-level claims.
    if intervention_before_generation is True and not has_atomize_edge:
        return {
            "intervention_scope": "semantic_information",
            "semantic_information_absent": True,
            "derived_information_may_persist": False,
            "preferred_boundary_hint": "",
        }

    return {
        "intervention_scope": "artifact_only",
        "semantic_information_absent": False,
        "derived_information_may_persist": False,
        "preferred_boundary_hint": "",
    }


@dataclass(frozen=True)
class InformationRoleFinding:
    """Role of tested information for one selected ReplayComparison."""

    role_type: str
    target_ref: str = ""
    information_description: str = ""
    artifact_type: str = ""

    semantic_information_absent: bool = False
    derived_information_may_persist: bool = False
    intervention_scope: str = "artifact_only"
    intervention_before_generation: Optional[bool] = None

    stem_effect: str = "unavailable"
    correct_answer_effect: str = "unavailable"
    distractor_effect: str = "unavailable"
    answer_explanation_effect: str = "unavailable"
    decision_effect: str = "unchanged"

    factual_stem: Optional[str] = None
    counterfactual_stem: Optional[str] = None
    factual_correct_answer: Optional[str] = None
    counterfactual_correct_answer: Optional[str] = None
    changed_distractor_count: Optional[int] = None

    decision_before: Optional[str] = None
    decision_after: Optional[str] = None
    causal_classification: str = ROLE_INCONCLUSIVE
    content_effect: bool = False
    decision_effect_bool: bool = False

    participant_lines: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "information_role": self.role_type,
            "information_role_supported": self.role_type
            not in {ROLE_INCONCLUSIVE, ROLE_NO_DEMONSTRATED, ROLE_ARTIFACT_ONLY},
            "semantic_information_absent": self.semantic_information_absent,
            "derived_information_may_persist": self.derived_information_may_persist,
            "intervention_scope": self.intervention_scope,
            "changed_distractor_count": self.changed_distractor_count,
            "content_effect": self.content_effect,
            "decision_effect": self.decision_effect_bool,
            "causal_classification": self.causal_classification,
            "stem_effect": self.stem_effect,
            "correct_answer_effect": self.correct_answer_effect,
            "distractor_effect": self.distractor_effect,
            "answer_explanation_effect": self.answer_explanation_effect,
        }


def _decision_phrase(decision: Any) -> str:
    d = str(decision or "").strip().lower()
    mapping = {
        "accept": "ACCEPT",
        "accepted": "ACCEPT",
        "revise": "REVISE",
        "needs_revision": "REVISE",
        "reject": "REJECT",
        "rejected": "REJECT",
    }
    return mapping.get(d, d.upper() or "UNKNOWN")


def _quote(text: Optional[str], *, max_len: int = 120) -> str:
    t = " ".join(str(text or "").split()).strip()
    if len(t) > max_len:
        t = t[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return f'"{t}"' if t else '""'


def classify_stem_semantic_relation(
    factual_stem: Optional[str],
    counterfactual_stem: Optional[str],
) -> str:
    """Conservative stem relation: prefer formulation over focus when unsure."""
    f = _norm(factual_stem).lower()
    c = _norm(counterfactual_stem).lower()
    if not f or not c:
        return "unavailable"
    if f == c:
        return "equivalent_rephrasing"

    stop = {
        "what",
        "is",
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "for",
        "and",
        "or",
        "on",
        "with",
        "by",
        "as",
        "from",
        "which",
        "when",
        "how",
        "does",
        "do",
        "are",
        "its",
        "this",
        "that",
    }
    f_toks = {t for t in re.findall(r"[a-z0-9]+", f) if t not in stop}
    c_toks = {t for t in re.findall(r"[a-z0-9]+", c) if t not in stop}
    if not f_toks or not c_toks:
        return "unavailable"
    inter = len(f_toks & c_toks)
    union = len(f_toks | c_toks)
    jaccard = inter / union if union else 0.0
    shared_core = f_toks & c_toks
    only_f = f_toks - c_toks
    only_c = c_toks - f_toks

    # Near paraphrase (goal↔purpose, etc.): high overlap or tiny lexical swap.
    if jaccard >= 0.65:
        return "equivalent_rephrasing"
    if jaccard >= 0.55 and len(only_f | only_c) <= 2:
        return "equivalent_rephrasing"

    # Clear topic shift.
    if jaccard < 0.35:
        return "different_focus"

    # Added/changed aspect while retaining a shared core concept.
    if shared_core and (only_f or only_c) and jaccard >= 0.4:
        return "same_concept_different_aspect"

    # Conservative default: formulation, not focus.
    return "equivalent_rephrasing"


def classify_information_role(
    *,
    stem_effect: str,
    correct_answer_effect: str,
    distractor_effect: str,
    answer_explanation_effect: str,
    question_content_status: str,
    decision_before: str,
    decision_after: str,
    semantic_information_absent: bool,
    intervention_before_generation: Optional[bool],
    stem_semantic_relation: str = "unavailable",
) -> Tuple[str, str]:
    """Return (role_type, causal_classification)."""
    decision_changed = bool(
        decision_before
        and decision_after
        and decision_before != decision_after
    )
    content_changed = question_content_status in {
        "question_content_changed",
        "question_content_partially_supported",
    } or stem_effect == "changed" or correct_answer_effect == "changed" or distractor_effect == "changed"

    if question_content_status == "question_content_inconclusive":
        return ROLE_INCONCLUSIVE, ROLE_INCONCLUSIVE

    if not semantic_information_absent and content_changed:
        return ROLE_ARTIFACT_ONLY, "artifact_level_downstream"

    if question_content_status == "question_not_reproduced" and semantic_information_absent:
        return ROLE_NECESSARY_FOR_REPRODUCTION, ROLE_NECESSARY_FOR_REPRODUCTION

    if (
        question_content_status == "question_content_unchanged"
        and semantic_information_absent
        and intervention_before_generation is True
    ):
        return ROLE_ROBUST, ROLE_ROBUST

    if not content_changed and not decision_changed:
        return ROLE_NO_DEMONSTRATED, ROLE_NO_DEMONSTRATED

    role = ROLE_NO_DEMONSTRATED
    if (
        stem_effect == "unchanged"
        and correct_answer_effect == "unchanged"
        and distractor_effect == "changed"
    ):
        role = ROLE_DISTRACTOR_SHAPING
    elif (
        stem_effect == "unchanged"
        and correct_answer_effect == "changed"
        and intervention_before_generation is True
    ):
        role = ROLE_ANSWER_ANCHORING
    elif stem_effect == "changed" and intervention_before_generation is True:
        # Stem change alone is not focus_shaping; require stronger semantic shift.
        if stem_semantic_relation in {"different_focus", "same_concept_different_aspect"}:
            role = ROLE_FOCUS_SHAPING
        else:
            role = ROLE_FORMULATION_SHAPING
    elif answer_explanation_effect == "changed" and not (
        stem_effect == "changed"
        or correct_answer_effect == "changed"
        or distractor_effect == "changed"
    ):
        role = ROLE_FORMULATION_SHAPING
    elif content_changed:
        role = ROLE_FORMULATION_SHAPING

    if content_changed and not decision_changed:
        return role, ROLE_INFLUENTIAL_NON_DECISIVE
    if decision_changed:
        return role, "acceptance_relevant"
    return role, role


def verbalize_information_role(finding: InformationRoleFinding) -> Tuple[str, ...]:
    """Participant-facing role lines — no REMOVE/NOOP/checkpoint language."""
    lines: List[str] = []
    role = finding.role_type
    cf_dec = finding.decision_after or ""
    noop_dec = finding.decision_before or ""
    info = _norm(finding.information_description)
    if len(info) > 160:
        cut = info[:159].rsplit(" ", 1)[0].rstrip(".,;:")
        info = (cut or info[:159]) + "…"
    info_ref = f'The information stating "{info}"' if info else "This information"

    if role == ROLE_INCONCLUSIVE:
        lines.append(
            "A controlled check of this particular supporting information could not "
            "reliably show what role it played in the selected question."
        )
        return tuple(lines)

    if role == ROLE_ARTIFACT_ONLY:
        lines.append(
            "A technical change to one source record affected later processing, "
            "but this does not show that the underlying teaching idea itself was absent, "
            "because related derived statements may still have been available."
        )
        return tuple(lines)

    if role == ROLE_NECESSARY_FOR_REPRODUCTION:
        visible_unchanged = (
            finding.stem_effect == "unchanged"
            and finding.correct_answer_effect == "unchanged"
            and finding.distractor_effect == "unchanged"
        )
        if visible_unchanged:
            lines.append(
                f"{info_ref} was required to reproduce the recorded result, "
                "although the visible stem, correct answer, and distractors "
                "remained unchanged."
            )
        else:
            lines.append(
                f"{info_ref} is important for reproducing this particular selected "
                "question. Without it, the remaining support did not yield a "
                "corresponding question with the same educational focus."
            )
        return tuple(lines)

    if role in {ROLE_ROBUST, ROLE_NO_DEMONSTRATED}:
        if finding.answer_explanation_effect == "changed":
            lines.append(
                f"{info_ref} left the question and its answer options stable, "
                "although the accompanying explanation changed."
            )
        else:
            lines.append(
                f"{info_ref} left the question wording, correct answer, and "
                "incorrect alternatives unchanged in this test."
            )
        return tuple(lines)

    if role == ROLE_DISTRACTOR_SHAPING:
        n = finding.changed_distractor_count
        if n == 1:
            lines.append(
                f"{info_ref} did not affect what the selected question asked or "
                "its correct answer, but it influenced one of the incorrect "
                "alternatives."
            )
        else:
            lines.append(
                f"{info_ref} did not affect what the selected question asked or "
                "its correct answer, but it influenced some of the incorrect "
                "alternatives."
            )
    elif role == ROLE_ANSWER_ANCHORING:
        lines.append(
            f"{info_ref} helped determine what counted as the correct answer in "
            "the selected question. The main question remains focused on the same "
            "concept, but the expected answer depends on this specific supporting idea."
        )
    elif role == ROLE_FOCUS_SHAPING:
        lines.append(
            f"{info_ref} helped keep the selected question focused on its "
            "particular conceptual target."
        )
        # Only quote alternative stems when they clarify a genuine focus shift.
        if finding.factual_stem and finding.counterfactual_stem:
            rel = classify_stem_semantic_relation(
                finding.factual_stem, finding.counterfactual_stem
            )
            if rel in {"same_concept_different_aspect", "different_focus"}:
                lines.append(
                    f"With this support the question asks {_quote(finding.factual_stem)}; "
                    f"with the remaining support the available content instead favors "
                    f"{_quote(finding.counterfactual_stem)}."
                )
    elif role == ROLE_FORMULATION_SHAPING:
        if finding.stem_effect == "changed" or finding.correct_answer_effect == "changed":
            lines.append(
                f"{info_ref} influenced how the selected question and its expected "
                "answer were phrased, while the underlying concept remained the same."
            )
        else:
            lines.append(
                f"{info_ref} influences how the answer explanation is formulated, "
                "while the core selected question and options remain the same."
            )
    else:
        lines.append(
            f"{info_ref} influenced part of the exact selected question."
        )

    if finding.causal_classification == ROLE_INFLUENTIAL_NON_DECISIVE:
        # Distinguish question-identity change from mere acceptance stability.
        identity_changed = any(
            getattr(finding, eff, None) == "changed"
            for eff in (
                "stem_effect",
                "correct_answer_effect",
                "distractor_effect",
            )
        )
        if identity_changed:
            lines.append(
                "This information influenced which question was produced."
            )
    elif finding.decision_effect_bool and cf_dec and noop_dec:
        lines.append(
            f"The final selection outcome also changed from {noop_dec} to {cf_dec}."
        )

    return tuple(lines)


def _looks_like_uuid(value: str) -> bool:
    return bool(
        re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            str(value or "").strip(),
            re.I,
        )
    )


def _has_semantic_timing(label: str) -> bool:
    """True when a label encodes workflow timing (not a bare alias like c_6)."""
    low = str(label or "").strip().lower()
    if not low or _looks_like_uuid(low):
        return False
    if re.fullmatch(r"c_\d+", low):
        return False
    return any(
        token in low
        for token in (
            "before ",
            "after ",
            "atomize",
            "plan_mcq",
            "generate_candidate",
            "verify",
            "retrieve",
            "acceptance",
            "cove",
        )
    )


def _resolve_boundary_label(sig: Mapping[str, Any], cmp: Mapping[str, Any]) -> str:
    """Prefer semantic boundary_raw over internal aliases like c_6."""
    for key in ("boundary_raw", "semantic_boundary", "boundary", "boundary_label"):
        val = sig.get(key)
        if isinstance(val, str) and val.strip() and _has_semantic_timing(val):
            return val.strip()
        if isinstance(val, str) and val.strip() and not _looks_like_uuid(val) and not re.fullmatch(
            r"c_\d+", val.strip(), re.I
        ):
            return val.strip()
    ckpt = sig.get("checkpoint")
    if isinstance(ckpt, Mapping):
        for key in ("boundary_raw", "semantic_boundary", "alias", "checkpoint_alias", "label"):
            val = ckpt.get(key)
            if isinstance(val, str) and val.strip() and _has_semantic_timing(val):
                return val.strip()
    # checkpoint_alias only if it itself encodes semantic timing.
    alias = sig.get("checkpoint_alias")
    if isinstance(alias, str) and _has_semantic_timing(alias):
        return alias.strip()
    # Generic subject_generation_relation as fallback.
    rel = cmp.get("subject_generation_relation")
    if isinstance(rel, str) and rel.strip():
        return rel.strip()
    # Last resort: alias/id only when no better label exists (may be non-semantic).
    for key in ("checkpoint_alias", "checkpoint_id"):
        val = sig.get(key)
        if isinstance(val, str) and val.strip() and not _looks_like_uuid(val):
            return val.strip()
    return ""


def _resolve_information_description(
    *,
    target_key: str,
    relevant_artifacts: Sequence[Any] = (),
) -> str:
    """Describe the intervention TARGET content, not the MCQ stem.

    For noisy EvidenceChunks, prefer a clean semantic rendering from descendant
    KnowledgeAtoms when available in ``relevant_artifacts``.
    """
    if not target_key:
        return ""
    try:
        from .lay_facts import (
            _is_noisy_extraction_snippet,
            _lookup_artifact,
            _snippet,
            is_metadata_only_snippet,
        )

        arts = [a for a in relevant_artifacts if isinstance(a, Mapping)]
        row = _lookup_artifact(arts, target_key)
        if row is None:
            lid = target_key.split(":")[0]
            for a in arts:
                ref = str(a.get("ref") or "")
                if ref == lid or ref.startswith(lid + ":"):
                    row = a
                    break
        if row is not None:
            atype = str(row.get("artifact_type") or "")
            snippet = _snippet(row, max_chars=180)
            noisy = bool(snippet and _is_noisy_extraction_snippet(snippet))
            if atype == "EvidenceChunk" and (noisy or not snippet):
                # Build participant description from clean KnowledgeAtoms.
                atom_bits: List[Tuple[int, str]] = []
                for a in arts:
                    if str(a.get("artifact_type") or "") != "KnowledgeAtom":
                        continue
                    # Prefer full clean atoms; avoid mid-sentence ellipsis cuts.
                    atom = _snippet(a, max_chars=220)
                    if not atom or is_metadata_only_snippet(atom):
                        continue
                    if _is_noisy_extraction_snippet(atom):
                        continue
                    # Prefer longer, more definition-like atoms.
                    atom_bits.append((len(atom), atom.rstrip("…").strip()))
                atom_bits.sort(key=lambda x: -x[0])
                clean = [t for _, t in atom_bits[:3]]
                if clean:
                    # One clean primary atom is clearer in participant prose than a
                    # long join (complementary atoms appear as separate component
                    # findings when supported). Still expose multi-atom semantics
                    # when the primary alone is very short.
                    if len(clean) == 1 or len(clean[0]) >= 80:
                        return clean[0]
                    if len(clean) == 2:
                        return f"{clean[0].rstrip('.')}, and {clean[1].rstrip('.')}"
                    return (
                        f"{clean[0].rstrip('.')}, with related points such as "
                        f"{clean[1].rstrip('.')} and {clean[2].rstrip('.')}"
                    )
            if snippet and not is_metadata_only_snippet(snippet) and not noisy:
                return snippet
            if snippet and not is_metadata_only_snippet(snippet) and not noisy:
                return snippet
    except Exception:
        pass
    for a in relevant_artifacts or ():
        if not isinstance(a, Mapping):
            continue
        ref = str(a.get("ref") or "")
        if ref != target_key and not ref.startswith(str(target_key).split(":")[0]):
            continue
        preview = _norm(a.get("content_preview") or a.get("content") or "")
        if preview:
            # Last resort: still avoid obvious OCR noise when possible.
            try:
                from .lay_facts import _is_noisy_extraction_snippet

                if _is_noisy_extraction_snippet(preview):
                    continue
            except Exception:
                pass
            return preview
    return ""


def build_information_role_finding(
    cmp: Mapping[str, Any],
    *,
    factual_decision: str = "ACCEPT",
    provenance_relations: Sequence[Any] = (),
    relevant_artifacts: Sequence[Any] = (),
) -> Optional[InformationRoleFinding]:
    """Build a role finding from one comparison mapping (export or in-memory)."""
    evidence_status = str(cmp.get("evidence_status") or "")
    invalid_control = {
        "noop_error",
        "replay_unstable",
        "outcome_unavailable",
        "intervention_error",
    }
    if evidence_status in invalid_control:
        # Hard epistemic gate: invalid NOOP / replay must not yield a supported
        # causal role (e.g. necessary_for_reproduction from subject absence).
        sig = cmp.get("intervention_signature")
        if not isinstance(sig, Mapping):
            sig = {}
        target = sig.get("target_ref")
        if isinstance(target, Mapping):
            target_key = str(target.get("key") or target.get("logical_id") or "")
        else:
            target_key = str(target or "")
        return InformationRoleFinding(
            role_type=ROLE_INCONCLUSIVE,
            target_ref=target_key,
            information_description="",
            artifact_type=str(sig.get("artifact_type") or ""),
            causal_classification=ROLE_INCONCLUSIVE,
            participant_lines=(
                "Causal counterfactual evidence was unavailable for this intervention "
                f"because the control replay was invalid ({evidence_status}).",
            ),
        )

    qcc = cmp.get("question_content_comparison")
    det: Dict[str, Any] = {}
    if isinstance(qcc, Mapping):
        inner = qcc.get("details")
        det = dict(inner) if isinstance(inner, Mapping) else dict(qcc)
    scd = cmp.get("subject_content_delta")
    if not det and isinstance(scd, Mapping):
        inner = scd.get("details")
        if isinstance(inner, Mapping):
            det = dict(inner)
    if not det and "stem_effect" in cmp:
        det = dict(cmp)
    if not det:
        return None

    sig = cmp.get("intervention_signature")
    if not isinstance(sig, Mapping):
        sig = {}
    target = sig.get("target_ref")
    if isinstance(target, Mapping):
        target_key = str(target.get("key") or target.get("logical_id") or "")
    else:
        target_key = str(target or "")

    artifact_type = str(
        sig.get("artifact_type")
        or cmp.get("artifact_type")
        or det.get("artifact_type")
        or ""
    )
    if not artifact_type:
        if target_key.startswith("atom_") or "atom_" in target_key:
            artifact_type = "KnowledgeAtom"
        elif target_key.startswith("chunk_") or "chunk_" in target_key:
            artifact_type = "EvidenceChunk"
        elif target_key.startswith("bp_") or "bp_" in target_key:
            artifact_type = "MCQBlueprint"
        elif target_key.startswith("q_") or "q_" in target_key:
            artifact_type = "CandidateMCQ"

    before = cmp.get(
        "intervention_before_subject_generation",
        det.get("intervention_before_subject_generation"),
    )
    boundary = _resolve_boundary_label(sig, cmp)

    sem = infer_semantic_intervention_validity(
        artifact_type=artifact_type,
        subject_generation_relation=str(
            cmp.get("subject_generation_relation")
            or det.get("subject_generation_relation")
            or ""
        ),
        intervention_before_generation=before if isinstance(before, bool) else None,
        boundary_label=boundary,
        provenance_relations=provenance_relations,
        target_ref=target_key,
    )

    fact = det.get("factual") if isinstance(det.get("factual"), Mapping) else {}
    cf = det.get("counterfactual") if isinstance(det.get("counterfactual"), Mapping) else {}
    removed = list(det.get("removed_distractors") or [])
    added = list(det.get("added_distractors") or [])
    n_changed = det.get("changed_distractor_count")
    if n_changed is None:
        n_changed = changed_distractor_count_from_lists(removed=removed, added=added)

    noop_dec = _decision_phrase(
        (cmp.get("noop") or {}).get("decision")
        if isinstance(cmp.get("noop"), Mapping)
        else factual_decision
    )
    cf_dec = _decision_phrase(
        (cmp.get("counterfactual") or {}).get("decision")
        if isinstance(cmp.get("counterfactual"), Mapping)
        else None
    )
    fact_dec = _decision_phrase(factual_decision)

    stem_effect = str(det.get("stem_effect") or "unavailable")
    correct_effect = str(det.get("correct_answer_effect") or "unavailable")
    dist_effect = str(det.get("distractor_effect") or "unavailable")
    expl_effect = str(det.get("answer_explanation_effect") or "unavailable")
    status = str(det.get("question_content_effect_status") or "")

    factual_stem = (fact.get("stem") if isinstance(fact, Mapping) else None) or det.get(
        "factual_question_stem"
    )
    counterfactual_stem = (
        (cf.get("stem") if isinstance(cf, Mapping) else None)
        or det.get("counterfactual_question_stem")
    )
    stem_rel = classify_stem_semantic_relation(factual_stem, counterfactual_stem)

    role, causal = classify_information_role(
        stem_effect=stem_effect,
        correct_answer_effect=correct_effect,
        distractor_effect=dist_effect,
        answer_explanation_effect=expl_effect,
        question_content_status=status,
        decision_before=noop_dec or fact_dec,
        decision_after=cf_dec or noop_dec,
        semantic_information_absent=bool(sem["semantic_information_absent"]),
        intervention_before_generation=before if isinstance(before, bool) else None,
        stem_semantic_relation=stem_rel,
    )

    content_effect = status in {
        "question_content_changed",
        "question_content_partially_supported",
        "question_not_reproduced",
    } or stem_effect == "changed" or correct_effect == "changed" or dist_effect == "changed"
    decision_effect_bool = bool(noop_dec and cf_dec and noop_dec != cf_dec)

    info_desc = _resolve_information_description(
        target_key=target_key,
        relevant_artifacts=relevant_artifacts,
    )

    base = InformationRoleFinding(
        role_type=role,
        target_ref=target_key,
        information_description=info_desc,
        artifact_type=artifact_type,
        semantic_information_absent=bool(sem["semantic_information_absent"]),
        derived_information_may_persist=bool(sem["derived_information_may_persist"]),
        intervention_scope=str(sem["intervention_scope"]),
        intervention_before_generation=before if isinstance(before, bool) else None,
        stem_effect=stem_effect,
        correct_answer_effect=correct_effect,
        distractor_effect=dist_effect,
        answer_explanation_effect=expl_effect,
        decision_effect="changed" if decision_effect_bool else "unchanged",
        factual_stem=factual_stem,
        counterfactual_stem=counterfactual_stem,
        factual_correct_answer=fact.get("correct_answer")
        if isinstance(fact, Mapping)
        else None,
        counterfactual_correct_answer=cf.get("correct_answer")
        if isinstance(cf, Mapping)
        else None,
        changed_distractor_count=int(n_changed) if n_changed is not None else None,
        decision_before=noop_dec or fact_dec,
        decision_after=cf_dec or noop_dec,
        causal_classification=causal,
        content_effect=content_effect,
        decision_effect_bool=decision_effect_bool,
    )
    lines = verbalize_information_role(base)
    return InformationRoleFinding(
        role_type=base.role_type,
        target_ref=base.target_ref,
        information_description=base.information_description,
        artifact_type=base.artifact_type,
        semantic_information_absent=base.semantic_information_absent,
        derived_information_may_persist=base.derived_information_may_persist,
        intervention_scope=base.intervention_scope,
        intervention_before_generation=base.intervention_before_generation,
        stem_effect=base.stem_effect,
        correct_answer_effect=base.correct_answer_effect,
        distractor_effect=base.distractor_effect,
        answer_explanation_effect=base.answer_explanation_effect,
        decision_effect=base.decision_effect,
        factual_stem=base.factual_stem,
        counterfactual_stem=base.counterfactual_stem,
        factual_correct_answer=base.factual_correct_answer,
        counterfactual_correct_answer=base.counterfactual_correct_answer,
        changed_distractor_count=base.changed_distractor_count,
        decision_before=base.decision_before,
        decision_after=base.decision_after,
        causal_classification=base.causal_classification,
        content_effect=base.content_effect,
        decision_effect_bool=base.decision_effect_bool,
        participant_lines=lines,
    )


def resolve_interrogation_goal(user_question: str) -> str:
    """Lightweight goal: question_selection | question_content | decision | general."""
    q = (user_question or "").lower()
    selection_markers = (
        "why was this question selected",
        "why this question was selected",
        "why was this question chosen",
        "why this question was chosen",
        "why was this mcq selected",
        "why this mcq was selected",
        "why was this mcq chosen",
        "why was the question selected",
        "why was the question chosen",
    )
    if any(m in q for m in selection_markers):
        return "question_selection"
    if "selected" in q and "question" in q:
        return "question_selection"
    if "chosen" in q and "question" in q:
        return "question_selection"
    content_markers = (
        "generate this question",
        "generated this question",
        "question wording",
        "would it have been different",
        "supporting information",
        "why this question",
        "why did the system generate",
        "without the supporting",
        "shape the question",
        "focus of the question",
    )
    decision_markers = (
        "accept",
        "reject",
        "decision",
        "why was it accepted",
        "why was this accepted",
    )
    if any(m in q for m in content_markers):
        return "question_content"
    if any(m in q for m in decision_markers) and "question" not in q:
        return "decision"
    if "question" in q and any(m in q for m in ("different", "without", "information")):
        return "question_content"
    return "general"
