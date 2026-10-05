"""Participant-facing MCQ question-content findings from AXIS comparisons."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .information_role import (
    build_information_role_finding,
    changed_distractor_count_from_lists,
)


def _details(cmp: Mapping[str, Any]) -> Dict[str, Any]:
    qcc = cmp.get("question_content_comparison")
    if isinstance(qcc, Mapping):
        det = qcc.get("details")
        if isinstance(det, Mapping) and det:
            return dict(det)
        if "stem_changed" in qcc or "factual" in qcc:
            return dict(qcc)
    # In-memory SubjectContentDelta serialized loosely
    scd = cmp.get("subject_content_delta")
    if isinstance(scd, Mapping):
        det = scd.get("details")
        if isinstance(det, Mapping):
            return dict(det)
    return {}


def _supported_stem_change(det: Mapping[str, Any]) -> bool:
    if det.get("stem_changed_under_intervention") is True:
        return True
    return bool(
        det.get("stem_control_stable") is True
        and det.get("stem_changed")
        and det.get("stem_effect") == "changed"
    )


def _supported_correct_change(det: Mapping[str, Any]) -> bool:
    if det.get("correct_answer_changed_under_intervention") is True:
        return True
    return bool(
        det.get("correct_answer_control_stable") is True
        and det.get("correct_answer_changed")
        and det.get("correct_answer_effect") == "changed"
    )


def _supported_distractor_change(det: Mapping[str, Any]) -> bool:
    if det.get("distractors_changed_under_intervention") is True:
        return True
    return bool(
        det.get("distractors_control_stable") is True
        and det.get("distractors_changed")
        and det.get("distractor_effect") == "changed"
    )


def _meta_from_details(
    det: Mapping[str, Any],
    cmp: Mapping[str, Any],
    *,
    role_diag: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build researcher/participant metadata solely from one comparison."""
    stem_effect = det.get("stem_effect")
    correct_effect = det.get("correct_answer_effect")
    dist_effect = det.get("distractor_effect")
    expl_effect = det.get("answer_explanation_effect")
    stem_changed = stem_effect == "changed"
    correct_changed = correct_effect == "changed"
    dist_changed = dist_effect == "changed"
    expl_changed = expl_effect == "changed"
    parts: List[str] = []
    if stem_changed:
        parts.append("stem")
    if correct_changed:
        parts.append("correct_answer")
    if dist_changed:
        parts.append("distractors")
    if expl_changed:
        parts.append("answer_explanation")
    sig = cmp.get("intervention_signature")
    if not isinstance(sig, Mapping):
        sig = {}
    target = sig.get("target_ref")
    if isinstance(target, Mapping):
        target_key = target.get("key") or target.get("logical_id")
    else:
        target_key = target
    ckpt = sig.get("checkpoint")
    checkpoint_id = sig.get("checkpoint_id")
    if checkpoint_id is None and isinstance(ckpt, Mapping):
        checkpoint_id = ckpt.get("checkpoint_id")
    removed = list(det.get("removed_distractors") or [])
    added = list(det.get("added_distractors") or [])
    n_changed = det.get("changed_distractor_count")
    if n_changed is None:
        n_changed = changed_distractor_count_from_lists(removed=removed, added=added)
    meta: Dict[str, Any] = {
        "primary_question_content_status": det.get("question_content_effect_status"),
        "question_content_control_stable": det.get("control_stable"),
        "question_changed_parts": parts,
        "counterfactual_question_stem": (det.get("counterfactual") or {}).get("stem")
        if isinstance(det.get("counterfactual"), Mapping)
        else det.get("counterfactual_question_stem"),
        "factual_question_stem": (det.get("factual") or {}).get("stem")
        if isinstance(det.get("factual"), Mapping)
        else det.get("factual_question_stem"),
        "intervention_before_question_generation": cmp.get(
            "intervention_before_subject_generation",
            det.get("intervention_before_subject_generation"),
        ),
        "stem_changed": stem_changed,
        "correct_answer_changed": correct_changed,
        "distractors_changed": dist_changed,
        "answer_explanation_changed": expl_changed,
        "stem_control_stable": det.get("stem_control_stable"),
        "correct_answer_control_stable": det.get("correct_answer_control_stable"),
        "distractors_control_stable": det.get("distractors_control_stable"),
        "answer_explanation_control_stable": det.get(
            "answer_explanation_control_stable"
        ),
        "stem_effect": stem_effect,
        "correct_answer_effect": correct_effect,
        "distractor_effect": dist_effect,
        "answer_explanation_effect": expl_effect,
        "changed_distractor_count": int(n_changed) if n_changed is not None else None,
        "primary_question_content_intervention": {
            "operation": sig.get("operation"),
            "target_ref": target_key,
            "checkpoint_id": checkpoint_id,
            "subject_generation_relation": cmp.get(
                "subject_generation_relation",
                det.get("subject_generation_relation"),
            ),
            "intervention_before_subject_generation": cmp.get(
                "intervention_before_subject_generation",
                det.get("intervention_before_subject_generation"),
            ),
        },
    }
    if role_diag:
        meta.update(dict(role_diag))
    return meta


def assert_primary_content_meta_consistent(meta: Mapping[str, Any]) -> None:
    """Fail loudly if primary content fields disagree with each other."""
    parts = set(meta.get("question_changed_parts") or ())
    mapping = (
        ("stem", meta.get("stem_effect"), "stem"),
        ("correct_answer", meta.get("correct_answer_effect"), "correct_answer"),
        ("distractors", meta.get("distractor_effect"), "distractors"),
    )
    for part, effect, _ in mapping:
        if effect == "changed" and part not in parts:
            raise AssertionError(
                f"inconsistent primary content meta: {part} effect=changed "
                f"but not in question_changed_parts={sorted(parts)}"
            )
        if effect in {"unchanged", "inconclusive", "unavailable"} and part in parts:
            raise AssertionError(
                f"inconsistent primary content meta: {part} in parts but "
                f"effect={effect!r}"
            )


def verbalize_question_content_comparison(
    cmp: Mapping[str, Any],
    *,
    factual_decision: str = "ACCEPT",
    provenance_relations: Sequence[Any] = (),
    relevant_artifacts: Sequence[Any] = (),
) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    """Return participant lines + metadata for one comparison's MCQ content delta.

    Participant lines describe the *role of information*, not REMOVE/replay mechanics.
    """
    det = _details(cmp)
    finding = None
    if det:
        finding = build_information_role_finding(
            cmp,
            factual_decision=factual_decision,
            provenance_relations=provenance_relations,
            relevant_artifacts=relevant_artifacts,
        )
    role_diag = finding.diagnostics() if finding is not None else {}
    if finding is not None:
        role_diag = {
            **role_diag,
            "primary_information_role_target": finding.target_ref,
            "primary_information_description": finding.information_description,
            "stem_semantic_relation": None,
        }
    meta = (
        _meta_from_details(det, cmp, role_diag=role_diag)
        if det
        else {
            "primary_question_content_status": None,
            "question_content_control_stable": None,
            "question_changed_parts": [],
            "counterfactual_question_stem": None,
            "factual_question_stem": None,
            "intervention_before_question_generation": cmp.get(
                "intervention_before_subject_generation"
            ),
            "stem_changed": False,
            "correct_answer_changed": False,
            "distractors_changed": False,
            "answer_explanation_changed": False,
            "stem_control_stable": None,
            "correct_answer_control_stable": None,
            "distractors_control_stable": None,
            "answer_explanation_control_stable": None,
            "stem_effect": None,
            "correct_answer_effect": None,
            "distractor_effect": None,
            "answer_explanation_effect": None,
            "changed_distractor_count": None,
            "primary_question_content_intervention": {},
            "primary_information_role_target": None,
            "primary_information_description": None,
        }
    )
    if det:
        assert_primary_content_meta_consistent(meta)
    if not det or finding is None:
        return (), meta

    before = meta["intervention_before_question_generation"]
    if before is False:
        lines = [
            "This information was checked only after the question had already been "
            "created, so this test speaks to later assessment rather than to how the "
            "question was shaped.",
        ]
        if finding.decision_effect_bool and finding.decision_before and finding.decision_after:
            lines.append(
                f"In the later assessment, the final decision changed from "
                f"{finding.decision_before} to {finding.decision_after}."
            )
        elif finding.decision_after:
            lines.append(
                f"The later assessment still recorded {finding.decision_after}."
            )
        return tuple(lines), meta

    lines = list(finding.participant_lines)
    return tuple(lines), meta


def select_primary_question_content_finding(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    factual_decision: str = "ACCEPT",
    provenance_relations: Sequence[Any] = (),
    relevant_artifacts: Sequence[Any] = (),
) -> Tuple[Tuple[str, ...], Dict[str, Any], Optional[Mapping[str, Any]]]:
    """Pick the strongest pre-generation content finding among comparisons."""

    def rank(cmp: Mapping[str, Any]) -> Tuple[int, int, int, int]:
        det = _details(cmp)
        before = cmp.get(
            "intervention_before_subject_generation",
            det.get("intervention_before_subject_generation"),
        )
        status = str(det.get("question_content_effect_status") or "")
        before_rank = 0 if before is True else (1 if before is None else 2)

        stem_ok = _supported_stem_change(det)
        correct_ok = _supported_correct_change(det)
        dist_ok = _supported_distractor_change(det)
        core_stable = bool(
            det.get("core_question_control_stable")
            if det.get("core_question_control_stable") is not None
            else det.get("control_stable")
        )

        finding = build_information_role_finding(
            cmp,
            factual_decision=factual_decision,
            provenance_relations=provenance_relations,
            relevant_artifacts=relevant_artifacts,
        )
        semantic_rank = 0
        if finding is not None:
            if finding.semantic_information_absent:
                semantic_rank = 0
            elif finding.derived_information_may_persist:
                semantic_rank = 2
            else:
                semantic_rank = 1
        else:
            semantic_rank = 3

        if before is True and stem_ok:
            effect_rank = 0
        elif before is True and correct_ok:
            effect_rank = 1
        elif before is True and dist_ok:
            effect_rank = 2
        elif status == "question_content_unchanged" and core_stable and before is True:
            effect_rank = 3
        elif status == "question_not_reproduced" and core_stable:
            effect_rank = 4
        elif status == "question_content_partially_supported" or (
            stem_ok or correct_ok or dist_ok
        ):
            effect_rank = 5
        elif status == "question_content_changed" and core_stable:
            effect_rank = 6
        elif status == "question_content_inconclusive":
            effect_rank = 7
        else:
            effect_rank = 8
        return (semantic_rank, before_rank, effect_rank, 0)

    usable = [c for c in comparisons if _details(c)]
    if not usable:
        return (), {}, None
    ordered = sorted(usable, key=rank)
    primary = ordered[0]
    lines, meta = verbalize_question_content_comparison(
        primary,
        factual_decision=factual_decision,
        provenance_relations=provenance_relations,
        relevant_artifacts=relevant_artifacts,
    )
    assert_primary_content_meta_consistent(meta)
    return lines, meta, primary


def select_component_role_findings(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    factual_decision: str = "ACCEPT",
    provenance_relations: Sequence[Any] = (),
    relevant_artifacts: Sequence[Any] = (),
) -> Tuple[Tuple[str, ...], Dict[str, Any], Optional[Mapping[str, Any]]]:
    """Select up to three complementary component-level role findings.

    At most one of each:
    - question/focus finding
    - correct-answer-specific finding
    - incorrect-answer-specific finding

    A single broad focus-shift comparison does not also fill answer/distractor
    slots from the same comparison.
    """
    focus_cmp: Optional[Mapping[str, Any]] = None
    answer_cmp: Optional[Mapping[str, Any]] = None
    distractor_cmp: Optional[Mapping[str, Any]] = None
    focus_finding = None
    answer_finding = None
    distractor_finding = None

    for cmp in comparisons or ():
        det = _details(cmp)
        if not det:
            continue
        finding = build_information_role_finding(
            cmp,
            factual_decision=factual_decision,
            provenance_relations=provenance_relations,
            relevant_artifacts=relevant_artifacts,
        )
        if finding is None:
            continue
        if finding.role_type in {
            "inconclusive",
            "artifact_level_only",
            "no_demonstrated_effect",
            "no_demonstrated_role",
        }:
            continue
        before = cmp.get(
            "intervention_before_subject_generation",
            det.get("intervention_before_subject_generation"),
        )
        if before is False:
            continue

        role = finding.role_type
        stem_ch = finding.stem_effect == "changed"
        correct_ch = finding.correct_answer_effect == "changed"
        dist_ch = finding.distractor_effect == "changed"
        broad_multi = stem_ch and correct_ch and dist_ch

        if role in {
            "focus_shaping",
            "formulation_shaping",
            "necessary_for_reproduction",
        } or (stem_ch and role not in {"answer_anchoring", "distractor_shaping"}):
            if focus_cmp is None:
                focus_cmp = cmp
                focus_finding = finding
            continue

        if role == "answer_anchoring" or (correct_ch and not stem_ch):
            if broad_multi:
                if focus_cmp is None:
                    focus_cmp = cmp
                    focus_finding = finding
                continue
            if answer_cmp is None:
                answer_cmp = cmp
                answer_finding = finding
            continue

        if role == "distractor_shaping" or (dist_ch and not stem_ch and not correct_ch):
            if broad_multi:
                if focus_cmp is None:
                    focus_cmp = cmp
                    focus_finding = finding
                continue
            if distractor_cmp is None:
                distractor_cmp = cmp
                distractor_finding = finding
            continue

    selected_cmps: List[Mapping[str, Any]] = []
    lines: List[str] = []
    metas: List[Dict[str, Any]] = []
    for cmp, finding in (
        (focus_cmp, focus_finding),
        (answer_cmp, answer_finding),
        (distractor_cmp, distractor_finding),
    ):
        if cmp is None or finding is None:
            continue
        selected_cmps.append(cmp)
        part_lines, meta = verbalize_question_content_comparison(
            cmp,
            factual_decision=factual_decision,
            provenance_relations=provenance_relations,
            relevant_artifacts=relevant_artifacts,
        )
        lines.extend(part_lines)
        metas.append(meta)

    if not lines:
        return select_primary_question_content_finding(
            comparisons,
            factual_decision=factual_decision,
            provenance_relations=provenance_relations,
            relevant_artifacts=relevant_artifacts,
        )

    primary_meta: Dict[str, Any] = dict(metas[0]) if metas else {}
    primary_meta["selected_component_finding_count"] = len(selected_cmps)
    primary_meta["selected_component_roles"] = [
        m.get("information_role") for m in metas if m.get("information_role")
    ]
    primary = focus_finding or answer_finding or distractor_finding
    if primary is not None:
        identity = bool(
            primary.stem_effect == "changed"
            or primary.correct_answer_effect == "changed"
            or primary.distractor_effect == "changed"
            or primary.role_type == "necessary_for_reproduction"
        )
        primary_meta["question_identity_effect"] = identity
        primary_meta["question_identity_effect_known"] = True
        primary_meta["acceptance_effect"] = bool(primary.decision_effect_bool)
        primary_meta["acceptance_effect_known"] = (
            primary.decision_before is not None and primary.decision_after is not None
        ) or ("decision_effect" in primary_meta)
        if "decision_effect" not in primary_meta:
            primary_meta["decision_effect"] = primary.decision_effect_bool

    # Deduplicate lines while preserving order.
    deduped: List[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)

    return tuple(deduped), primary_meta, selected_cmps[0] if selected_cmps else None
