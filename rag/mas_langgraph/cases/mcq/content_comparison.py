"""MCQ-specific CandidateMCQ content comparison for AXIS replays.

These comparisons detect **normalized text differences** (stem, correct
answer, distractors, explanation). The broad content-effect flag can
include question non-reproduction. They do not measure whether the tested
knowledge changed; optional LLM-assessed concept change lives in
``mas_langgraph.evaluation.concept_change``. Same tested concept does not
imply that all other semantic aspects of the question are unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...axis.comparison import CaseRunObservation, SubjectContentDelta


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _norm_key(text: Optional[str]) -> str:
    return _norm(text).lower()


def correct_answer_text(candidate: Any) -> Optional[str]:
    if candidate is None:
        return None
    opts = list(getattr(candidate, "options", None) or [])
    cid = str(getattr(candidate, "correct_option_id", "") or "")
    for o in opts:
        oid = str(getattr(o, "option_id", "") or getattr(o, "id", "") or "")
        if oid == cid:
            return _norm(getattr(o, "text", None) or getattr(o, "option_text", None) or "")
    return None


def distractor_texts(candidate: Any) -> Tuple[str, ...]:
    if candidate is None:
        return ()
    opts = list(getattr(candidate, "options", None) or [])
    cid = str(getattr(candidate, "correct_option_id", "") or "")
    out: List[str] = []
    for o in opts:
        oid = str(getattr(o, "option_id", "") or getattr(o, "id", "") or "")
        if oid == cid:
            continue
        t = _norm(getattr(o, "text", None) or getattr(o, "option_text", None) or "")
        if t:
            out.append(t)
    return tuple(out)


def project_mcq(candidate: Any) -> Optional[Dict[str, Any]]:
    if candidate is None:
        return None
    stem = _norm(getattr(candidate, "stem", None))
    if not stem and not getattr(candidate, "options", None):
        return None
    return {
        "stem": stem or None,
        "correct_answer": correct_answer_text(candidate),
        "distractors": list(distractor_texts(candidate)),
        "answer_explanation": _norm(getattr(candidate, "answer_explanation", None))
        or None,
    }


def _component_effect(
    *,
    control_stable: Optional[bool],
    changed_under_intervention: Optional[bool],
) -> str:
    if control_stable is None:
        return "unavailable"
    if control_stable is False:
        return "inconclusive"
    if changed_under_intervention is None:
        return "unavailable"
    return "changed" if changed_under_intervention else "unchanged"


@dataclass(frozen=True)
class MCQContentDelta:
    control_stable: bool
    question_present_in_factual: bool
    question_present_in_noop: bool
    question_present_in_counterfactual: bool
    stem_changed: bool
    correct_answer_changed: bool
    distractors_changed: bool
    answer_explanation_changed: bool
    content_changed: bool
    question_content_effect_status: str
    factual_stem: Optional[str] = None
    noop_stem: Optional[str] = None
    counterfactual_stem: Optional[str] = None
    factual_correct_answer: Optional[str] = None
    noop_correct_answer: Optional[str] = None
    counterfactual_correct_answer: Optional[str] = None
    removed_distractors: Tuple[str, ...] = ()
    added_distractors: Tuple[str, ...] = ()
    intervention_before_subject_generation: Optional[bool] = None
    subject_generation_relation: Optional[str] = None
    stem_control_stable: Optional[bool] = None
    correct_answer_control_stable: Optional[bool] = None
    distractors_control_stable: Optional[bool] = None
    answer_explanation_control_stable: Optional[bool] = None
    stem_changed_under_intervention: Optional[bool] = None
    correct_answer_changed_under_intervention: Optional[bool] = None
    distractors_changed_under_intervention: Optional[bool] = None
    answer_explanation_changed_under_intervention: Optional[bool] = None
    core_question_control_stable: Optional[bool] = None


def compare_mcq_subject_content(
    *,
    factual: CaseRunObservation,
    noop: CaseRunObservation,
    counterfactual: CaseRunObservation,
    intervention_before_subject_generation: Optional[bool] = None,
    subject_generation_relation: Optional[str] = None,
) -> SubjectContentDelta:
    f_proj = project_mcq(factual.subject_payload) if factual.subject_present else None
    n_proj = project_mcq(noop.subject_payload) if noop.subject_present else None
    c_proj = (
        project_mcq(counterfactual.subject_payload)
        if counterfactual.subject_present
        else None
    )

    fact_present = bool(f_proj) and bool(factual.subject_present)
    noop_present = bool(n_proj) and bool(noop.subject_present)
    cf_present = bool(c_proj) and bool(counterfactual.subject_present)

    stem_control_stable: Optional[bool] = None
    correct_control_stable: Optional[bool] = None
    distractors_control_stable: Optional[bool] = None
    expl_control_stable: Optional[bool] = None
    stem_changed_ui: Optional[bool] = None
    correct_changed_ui: Optional[bool] = None
    distractors_changed_ui: Optional[bool] = None
    expl_changed_ui: Optional[bool] = None

    stem_changed = False
    correct_changed = False
    distractors_changed = False
    expl_changed = False
    removed: Tuple[str, ...] = ()
    added: Tuple[str, ...] = ()
    content_changed = False

    if fact_present and noop_present and f_proj is not None and n_proj is not None:
        stem_control_stable = _norm_key(f_proj.get("stem")) == _norm_key(
            n_proj.get("stem")
        )
        correct_control_stable = _norm_key(f_proj.get("correct_answer")) == _norm_key(
            n_proj.get("correct_answer")
        )
        distractors_control_stable = sorted(
            _norm_key(x) for x in (f_proj.get("distractors") or [])
        ) == sorted(_norm_key(x) for x in (n_proj.get("distractors") or []))
        expl_control_stable = _norm_key(f_proj.get("answer_explanation")) == _norm_key(
            n_proj.get("answer_explanation")
        )

    # Core MCQ for "same quiz question" excludes answer_explanation.
    core_control_stable = bool(
        fact_present
        and noop_present
        and stem_control_stable
        and correct_control_stable
        and distractors_control_stable
    )
    # Legacy alias: full equality including explanation (kept for diagnostics).
    full_control_stable = bool(
        core_control_stable and expl_control_stable is True
    )
    # Participant-facing control_stable means core question reproduced.
    control_stable = core_control_stable

    if not fact_present:
        status = "question_content_unavailable"
    elif not noop_present:
        status = "question_not_reproduced"
    elif not cf_present:
        if core_control_stable:
            status = "question_not_reproduced"
            content_changed = True
        else:
            status = "question_content_inconclusive"
    else:
        assert n_proj is not None and c_proj is not None
        noop_cf_stem = _norm_key(n_proj.get("stem")) != _norm_key(c_proj.get("stem"))
        noop_cf_correct = _norm_key(n_proj.get("correct_answer")) != _norm_key(
            c_proj.get("correct_answer")
        )
        n_d = {_norm_key(x): x for x in (n_proj.get("distractors") or [])}
        c_d = {_norm_key(x): x for x in (c_proj.get("distractors") or [])}
        noop_cf_dist = set(n_d) != set(c_d)
        removed = tuple(n_d[k] for k in sorted(set(n_d) - set(c_d)))
        added = tuple(c_d[k] for k in sorted(set(c_d) - set(n_d)))
        noop_cf_expl = _norm_key(n_proj.get("answer_explanation")) != _norm_key(
            c_proj.get("answer_explanation")
        )

        if stem_control_stable is True:
            stem_changed_ui = noop_cf_stem
            stem_changed = bool(stem_changed_ui)
        elif stem_control_stable is False:
            stem_changed_ui = None  # inconclusive; do not attribute
            stem_changed = False

        if correct_control_stable is True:
            correct_changed_ui = noop_cf_correct
            correct_changed = bool(correct_changed_ui)
        elif correct_control_stable is False:
            correct_changed_ui = None
            correct_changed = False

        if distractors_control_stable is True:
            distractors_changed_ui = noop_cf_dist
            distractors_changed = bool(distractors_changed_ui)
        elif distractors_control_stable is False:
            distractors_changed_ui = None
            distractors_changed = False

        if expl_control_stable is True:
            expl_changed_ui = noop_cf_expl
            expl_changed = bool(expl_changed_ui)
        elif expl_control_stable is False:
            expl_changed_ui = None
            expl_changed = False

        supported_core_change = bool(
            stem_changed or correct_changed or distractors_changed
        )
        supported_any_change = bool(supported_core_change or expl_changed)
        content_changed = supported_core_change

        any_component_inconclusive = (
            stem_control_stable is False
            or correct_control_stable is False
            or distractors_control_stable is False
        )

        if supported_core_change and core_control_stable:
            status = "question_content_changed"
        elif supported_core_change and not core_control_stable:
            # Partial: e.g. stem stable+changed while distractors unstable.
            status = "question_content_partially_supported"
            content_changed = True
        elif supported_any_change and not supported_core_change and core_control_stable:
            # Explanation-only change with stable core question.
            status = "question_content_unchanged"
            content_changed = False
            # Keep expl_changed True for separate reporting.
        elif core_control_stable and not supported_any_change:
            status = "question_content_unchanged"
        elif any_component_inconclusive and not supported_core_change:
            status = "question_content_inconclusive"
        else:
            status = "question_content_inconclusive"

    stem_effect = _component_effect(
        control_stable=stem_control_stable,
        changed_under_intervention=stem_changed_ui,
    )
    correct_effect = _component_effect(
        control_stable=correct_control_stable,
        changed_under_intervention=correct_changed_ui,
    )
    distractor_effect = _component_effect(
        control_stable=distractors_control_stable,
        changed_under_intervention=distractors_changed_ui,
    )
    expl_effect = _component_effect(
        control_stable=expl_control_stable,
        changed_under_intervention=expl_changed_ui,
    )

    details: Dict[str, Any] = {
        "kind": "mcq_content",
        "control_stable": control_stable,
        "core_question_control_stable": core_control_stable,
        "full_control_stable_including_explanation": full_control_stable,
        "question_present_in_factual": fact_present,
        "question_present_in_noop": noop_present,
        "question_present_in_counterfactual": cf_present,
        "stem_control_stable": stem_control_stable,
        "correct_answer_control_stable": correct_control_stable,
        "distractors_control_stable": distractors_control_stable,
        "answer_explanation_control_stable": expl_control_stable,
        "stem_changed_under_intervention": stem_changed_ui,
        "correct_answer_changed_under_intervention": correct_changed_ui,
        "distractors_changed_under_intervention": distractors_changed_ui,
        "answer_explanation_changed_under_intervention": expl_changed_ui,
        "stem_effect": stem_effect,
        "correct_answer_effect": correct_effect,
        "distractor_effect": distractor_effect,
        "answer_explanation_effect": expl_effect,
        "stem_changed": stem_changed,
        "correct_answer_changed": correct_changed,
        "distractors_changed": distractors_changed,
        "answer_explanation_changed": expl_changed,
        "content_changed": content_changed,
        "question_content_effect_status": status,
        "intervention_before_subject_generation": intervention_before_subject_generation,
        "subject_generation_relation": subject_generation_relation,
        "factual": {
            "stem": (f_proj or {}).get("stem"),
            "correct_answer": (f_proj or {}).get("correct_answer"),
            "distractors": list((f_proj or {}).get("distractors") or []),
            "answer_explanation": (f_proj or {}).get("answer_explanation"),
        },
        "noop": {
            "stem": (n_proj or {}).get("stem"),
            "correct_answer": (n_proj or {}).get("correct_answer"),
            "distractors": list((n_proj or {}).get("distractors") or []),
            "answer_explanation": (n_proj or {}).get("answer_explanation"),
        },
        "counterfactual": {
            "stem": (c_proj or {}).get("stem"),
            "correct_answer": (c_proj or {}).get("correct_answer"),
            "distractors": list((c_proj or {}).get("distractors") or []),
            "answer_explanation": (c_proj or {}).get("answer_explanation"),
        },
        "removed_distractors": list(removed),
        "added_distractors": list(added),
        # Set-aware replacement count (order ignored); equal-size MCQs → replaced options.
        "changed_distractor_count": max(len(removed), len(added)) if (removed or added) else 0,
        "factual_question_stem": (f_proj or {}).get("stem"),
        "noop_question_stem": (n_proj or {}).get("stem"),
        "counterfactual_question_stem": (c_proj or {}).get("stem"),
    }

    return SubjectContentDelta(
        kind="mcq_content",
        factual_available=fact_present,
        noop_available=noop_present,
        counterfactual_available=cf_present,
        control_stable=control_stable,
        counterfactual_differs_from_noop=(
            content_changed
            if status
            in {
                "question_content_changed",
                "question_content_partially_supported",
                "question_not_reproduced",
            }
            else (False if status == "question_content_unchanged" else None)
        ),
        effect_status=status,
        details=details,
    )


def infer_subject_generation_relation(
    *,
    signature: Any,
    explained_case: Any,
    case_slice: Any = None,
    checkpoint_index: Any = None,
    occurrences: Sequence[Any] = (),
    produced_by: Sequence[Any] = (),
) -> Tuple[Optional[bool], str]:
    """Return (intervention_before_generation, relation_label)."""
    target = getattr(explained_case, "artifact_ref", None)
    if target is None:
        return None, "unknown"

    producer_occ_id = None
    for pb in produced_by or ():
        pref = getattr(pb, "artifact_ref", None)
        if pref is None:
            continue
        if str(pref) == str(target) or (
            getattr(pref, "logical_id", None) == getattr(target, "logical_id", None)
            and int(getattr(pref, "version", -1) or -1)
            == int(getattr(target, "version", -1) or -1)
        ):
            producer_occ_id = getattr(pb, "producer_occurrence_id", None)
            break
    if producer_occ_id is None and case_slice is not None:
        for pb in getattr(case_slice, "produced_by", ()) or ():
            pref = getattr(pb, "artifact_ref", None)
            if pref is not None and str(getattr(pref, "logical_id", "")) == str(
                getattr(target, "logical_id", "")
            ):
                producer_occ_id = getattr(pb, "producer_occurrence_id", None)
                break

    producer_exec: Optional[int] = None
    producer_node = ""
    occ_list = list(occurrences or ())
    if not occ_list and case_slice is not None:
        occ_list = list(getattr(case_slice, "relevant_occurrences", ()) or ())
    for occ in occ_list:
        if str(getattr(occ, "occurrence_id", "")) == str(producer_occ_id or ""):
            producer_exec = getattr(occ, "execution_index", None)
            producer_node = str(getattr(occ, "node_id", "") or "")
            break
        produced = list(getattr(occ, "produced_artifacts", ()) or [])
        if target in produced or str(target) in {str(x) for x in produced}:
            producer_exec = getattr(occ, "execution_index", None)
            producer_node = str(getattr(occ, "node_id", "") or "")
            producer_occ_id = getattr(occ, "occurrence_id", None)
            break

    ckpt_id = getattr(signature, "checkpoint_id", None)
    if ckpt_id is None:
        return None, "unknown"

    if checkpoint_index is not None and producer_occ_id:
        records = list(getattr(checkpoint_index, "records", None) or [])
        ckpt_rec = next(
            (r for r in records if getattr(r, "checkpoint_id", None) == ckpt_id),
            None,
        )
        if ckpt_rec is not None:
            after = set(getattr(ckpt_rec, "after_occurrences", ()) or ())
            if str(producer_occ_id) in after:
                return False, "after_generation"
            if producer_exec is not None:
                occ_by_id = {
                    str(getattr(o, "occurrence_id", "")): o for o in occ_list
                }
                max_after_exec = -1
                for oid in after:
                    o = occ_by_id.get(str(oid))
                    if o is not None:
                        max_after_exec = max(
                            max_after_exec,
                            int(getattr(o, "execution_index", -1) or -1),
                        )
                if max_after_exec >= 0:
                    if max_after_exec < int(producer_exec):
                        return True, "before_generation"
                    return False, "after_generation"

    # Boundary alias heuristic when only labels are known via checkpoint after_labels.
    if checkpoint_index is not None:
        records = list(getattr(checkpoint_index, "records", None) or [])
        ckpt_rec = next(
            (r for r in records if getattr(r, "checkpoint_id", None) == ckpt_id),
            None,
        )
        if ckpt_rec is not None:
            labels = " ".join(getattr(ckpt_rec, "after_occurrence_labels", ()) or [])
            low = labels.lower()
            if "generate_candidate_mcqs" in low:
                return False, "after_generation"
            if producer_node and "generate_candidate_mcqs" in producer_node:
                # Producer not yet in after_labels → treat as before.
                if "generate_candidate_mcqs" not in low:
                    return True, "before_generation"

    return None, "unknown"


def prefer_pre_generation_boundaries(
    boundaries: Sequence[str],
    *,
    artifact_type: str = "",
) -> Tuple[str, ...]:
    """Stable re-order: semantically preferred pre-generation boundaries first."""
    a_type = str(artifact_type or "")

    def _rank(b: str) -> Tuple[int, str]:
        low = str(b).lower()
        # EvidenceChunk: prefer before atomize so descendants are recomputed.
        if a_type in {"EvidenceChunk", "EvidenceChunkPayload"}:
            if "before" in low and "atomize" in low:
                return (0, low)
            if "before" in low and "plan_mcq" in low:
                return (2, low)
            if "before" in low and "generate_candidate" in low:
                return (3, low)
        # KnowledgeAtom: prefer before planning, then before generation.
        if a_type in {"KnowledgeAtom", "KnowledgeAtomPayload"}:
            if "before" in low and "plan_mcq" in low:
                return (0, low)
            if "before" in low and "generate_candidate" in low:
                return (1, low)
            if "before" in low and "atomize" in low:
                return (3, low)
        # Blueprint: before generation.
        if a_type in {"MCQBlueprint", "MCQBlueprintPayload"}:
            if "before" in low and "generate_candidate" in low:
                return (0, low)
            if "before" in low and "plan_mcq" in low:
                return (1, low)
        # CandidateMCQ: post-generation is low priority for question-content goals.
        if a_type in {"CandidateMCQ", "CandidateMCQPayload"}:
            if "before" in low and any(
                x in low for x in ("verify", "acceptance", "cross_cove")
            ):
                return (8, low)
            if "after" in low and "generate" in low:
                return (9, low)

        if low.startswith("before ") and any(
            x in low
            for x in (
                "atomize",
                "plan_mcq",
                "curate",
                "retrieve",
                "outline",
                "learning_objective",
                "deduplicate",
                "generate_candidate",
            )
        ):
            return (4, low)
        if low.startswith("after ") and any(
            x in low
            for x in ("generate_candidate", "acceptance", "verify", "cross_cove")
        ):
            return (10, low)
        return (5, low)

    return tuple(sorted((str(b) for b in boundaries), key=_rank))
