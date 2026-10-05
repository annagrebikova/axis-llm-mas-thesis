"""AXIS mechanism-level metrics from in-memory experiment payloads.

Grounded only in recorded fields. Missing is not False. Rates with a zero
denominator are NA (None), never coerced to 0.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

NA = None

INVALID_EVIDENCE = {
    "noop_error",
    "intervention_error",
    "outcome_unavailable",
    "replay_unstable",
}

CF_OPERATIONS = {"REMOVE", "ADD", "REPLACE"}
FACTUAL_TARGET_OPS = {"REMOVE", "REPLACE"}
RECOGNIZED_QUERY_OPS = {"REMOVE", "ADD", "REPLACE", "WHAT"}
KNOWN_PREFIX_TYPES = {
    "evidencechunk": "EvidenceChunk",
    "evidencechunkpayload": "EvidenceChunk",
    "knowledgeatom": "KnowledgeAtom",
    "knowledgeatompayload": "KnowledgeAtom",
    "mcqblueprint": "MCQBlueprint",
    "blueprint": "MCQBlueprint",
    "candidatemcq": "CandidateMCQ",
    "candidate": "CandidateMCQ",
}

_ALIAS_RE = re.compile(r"^c_(\d+)$", re.IGNORECASE)
_OCC_NODE_RE = re.compile(r"(?:^|:)([^:#]+)(?:#\d+)?$")


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return NA
    den = float(denominator)
    if den == 0:
        return NA
    return float(numerator) / den


def pct(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    value = ratio(numerator, denominator)
    if value is None:
        return NA
    return 100.0 * value


def tribool(value: Any) -> Optional[bool]:
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return None


def normalize_operation(raw: Any) -> Tuple[str, str]:
    """Return (canonical, raw_preserved). WHATIF → REPLACE; missing → UNPARSED."""
    preserved = "" if raw is None else str(raw).strip()
    token = preserved.upper().replace("-", "_").replace(" ", "")
    if not token:
        return "UNPARSED", preserved
    if token in {"WHATIF", "WHAT_IF"}:
        return "REPLACE", preserved or "WHATIF"
    if token in {"REMOVE", "ADD", "REPLACE", "WHAT", "DONE"}:
        return token, preserved
    return "UNPARSED", preserved


def axis_figure4_label(canonical: str) -> Optional[str]:
    return {
        "REMOVE": "remove",
        "ADD": "add",
        "REPLACE": "whatif",
        "WHAT": "what",
    }.get(str(canonical or "").upper())


def checkpoint_position(
    *,
    alias: Any = None,
    checkpoint_id: Any = None,
    checkpoint_index: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Map a checkpoint alias onto chronological position.

    Aliases start at ``c_0``. The 8th checkpoint of 10 (``c_7``) is 80%
    ``checkpoint_position_pct``. ``checkpoint_progress_pct`` maps first=0%
    and last=100%. A single checkpoint yields NA progress rather than /0.
    """
    records = list((checkpoint_index or {}).get("records") or [])
    total = len(records)
    alias_s = str(alias or "").strip()
    ckpt_id = str(checkpoint_id or "").strip()
    pos: Optional[int] = None
    resolved_alias = alias_s
    resolved_id = ckpt_id

    for i, rec in enumerate(records):
        rec_alias = str(rec.get("alias") or "")
        rec_id = str(rec.get("checkpoint_id") or "")
        if alias_s and rec_alias == alias_s:
            pos = i
            resolved_alias = rec_alias
            resolved_id = rec_id or ckpt_id
            break
    if pos is None and ckpt_id:
        for i, rec in enumerate(records):
            rec_id = str(rec.get("checkpoint_id") or "")
            if rec_id and rec_id == ckpt_id:
                pos = i
                resolved_alias = str(rec.get("alias") or resolved_alias)
                resolved_id = rec_id
                break
    if pos is None and alias_s:
        match = _ALIAS_RE.fullmatch(alias_s)
        if match:
            pos = int(match.group(1))
    alias_index: Optional[int] = None
    if resolved_alias:
        match = _ALIAS_RE.fullmatch(resolved_alias)
        if match:
            alias_index = int(match.group(1))
        elif pos is not None:
            alias_index = pos
    if pos is None and alias_index is not None:
        pos = alias_index

    ordinal = (pos + 1) if pos is not None else None
    position_pct = pct(ordinal, total) if ordinal is not None and total else NA
    progress_pct = NA
    if pos is not None and total > 1:
        progress_pct = pct(pos, total - 1)
    elif pos is not None and total == 1:
        progress_pct = NA

    return {
        "checkpoint_alias": resolved_alias or None,
        "checkpoint_id": resolved_id or None,
        "checkpoint_alias_index": alias_index,
        "checkpoint_ordinal_1based": ordinal,
        "checkpoint_total": total if records else (total or None),
        "checkpoint_position_pct": position_pct,
        "checkpoint_progress_pct": progress_pct,
        "zero_based_chronological_position": pos,
    }


def occurrence_node_name(occurrence_id: Any) -> str:
    text = str(occurrence_id or "").strip()
    if not text:
        return ""
    match = _OCC_NODE_RE.search(text)
    if match:
        return match.group(1)
    if "#" in text:
        return text.split("#", 1)[0].split(":")[-1]
    return text.split(":")[-1]


def artifact_ref_key(ref: Any) -> str:
    if ref is None:
        return ""
    if isinstance(ref, Mapping):
        return str(ref.get("key") or ref.get("logical_id") or ref.get("ref") or "")
    return str(ref)


def _axis_condition(case: Mapping[str, Any]) -> Dict[str, Any]:
    conds = case.get("conditions") if isinstance(case.get("conditions"), Mapping) else {}
    axis = conds.get("axis") if isinstance(conds, Mapping) else {}
    return dict(axis or {})


def _axis_raw(case: Mapping[str, Any]) -> Dict[str, Any]:
    axis = _axis_condition(case)
    raw = axis.get("raw_evidence")
    if isinstance(raw, Mapping) and raw:
        return dict(raw)
    summary = axis.get("evidence_summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return {}


def _candidate_ref(case: Mapping[str, Any]) -> str:
    ref = case.get("candidate_ref")
    key = artifact_ref_key(ref)
    return key or str(ref or "")


def extract_experiment_metrics(
    experiment: Mapping[str, Any],
    checkpoint_index: Optional[Mapping[str, Any]] = None,
    *,
    lecturebank_id: str = "",
) -> Dict[str, Any]:
    """Extract per-document AXIS metrics from an in-memory experiment dict."""
    cases = list(experiment.get("cases") or [])
    attempt_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    provenance_rows: List[Dict[str, Any]] = []
    dist_counter: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    criterion_name_counter: Counter = Counter()
    provenance_slicing_errors = 0
    synthesis_rows: List[Dict[str, Any]] = []

    for case in cases:
        cand = _candidate_ref(case)
        raw = _axis_raw(case)
        axis = _axis_condition(case)
        summary = dict(axis.get("evidence_summary") or {})
        _collect_attempts(
            lecturebank_id,
            cand,
            raw.get("interrogation_attempts") or [],
            attempt_rows,
            dist_counter,
        )
        _collect_comparisons(
            lecturebank_id,
            cand,
            raw.get("replay_comparisons") or [],
            experiment.get("timing") or {},
            checkpoint_index,
            raw,
            comparison_rows,
            dist_counter,
            criterion_name_counter,
        )
        slice_error = _collect_provenance(
            lecturebank_id,
            cand,
            raw,
            provenance_rows,
            dist_counter,
        )
        if slice_error:
            provenance_slicing_errors += 1
        term = str(raw.get("termination_reason") or summary.get("termination_reason") or "")
        if term:
            dist_counter[("document", "termination_reason")][term] += 1
        synthesis_rows.append(_synthesis_from_summary(summary or axis.get("evidence_summary") or {}))
        # evidence_summary may duplicate synthesis keys
        syn_src = summary if summary else dict(axis.get("evidence_summary") or {})
        if not synthesis_rows[-1].get("_filled"):
            synthesis_rows[-1] = _synthesis_from_summary(syn_src)

    query_stats = _query_distribution(attempt_rows)
    replay_stats = _replay_stability(comparison_rows, cases)
    effect_stats = _intervention_effects(comparison_rows)
    ckpt_stats = _checkpoint_aggregates(comparison_rows)
    target_stats = _target_artifact_distribution(comparison_rows)
    prov_stats = _provenance_integrity(
        provenance_rows, comparison_rows, cases, provenance_slicing_errors
    )
    llm_stats = _replay_determinism(experiment.get("workflow_llm_diagnostics") or {})
    timing_stats = _timing_metrics(experiment.get("timing") or {})
    synthesis_stats = _synthesis_aggregates(synthesis_rows)
    role_stats = _distribution_from_rows(comparison_rows, "information_role")
    causal_stats = _distribution_from_rows(comparison_rows, "causal_classification")
    round_rows = _round_metrics(comparison_rows)

    for row in comparison_rows:
        for dim, key in (
            ("evidence_status", "evidence_status"),
            ("causal_classification", "causal_classification"),
            ("information_role", "information_role"),
            ("question_content_effect_status", "question_content_effect_status"),
            ("replay_completion_mode", "replay_completion_mode"),
            ("target_artifact_type", "target_artifact_type"),
        ):
            val = row.get(key)
            if val not in (None, ""):
                dist_counter[("document", dim)][str(val)] += 1

    document_metrics = {
        "lecturebank_id": lecturebank_id,
        "explained_question_count": len(cases),
        "accepted_question_count": experiment.get("question_count"),
        "run_id": experiment.get("run_id"),
        "axis_interrogation_mode": experiment.get("axis_interrogation_mode")
        or (
            (attempt_rows[0].get("axis_interrogation_mode") if attempt_rows else None)
            or "constrained"
        ),
        "axis_interrogation_policy": experiment.get("axis_interrogation_policy")
        or (
            (attempt_rows[0].get("axis_interrogation_policy") if attempt_rows else None)
            or "counterfactual_only"
        ),
        "provenance_slicing_error_count": provenance_slicing_errors,
        **query_stats,
        **replay_stats,
        **effect_stats,
        **ckpt_stats,
        **target_stats,
        **prov_stats,
        **llm_stats,
        **timing_stats,
        **synthesis_stats,
        **_prefixed_dist("causal_classification", causal_stats),
        **_prefixed_dist("information_role", role_stats),
    }
    from mas_langgraph.evaluation.outcome_criteria import extract_outcome_criteria_changes

    criteria_out = extract_outcome_criteria_changes(
        experiment, lecturebank_id=lecturebank_id
    )
    return {
        "document_metrics": document_metrics,
        "interrogation_attempts": attempt_rows,
        "replay_comparisons": comparison_rows,
        "provenance_edges": provenance_rows,
        "distributions": _flatten_distributions(lecturebank_id, dist_counter, attempt_rows, comparison_rows, provenance_rows),
        "round_metrics": round_rows,
        "query_operation_counts": query_stats,
        "criterion_effect_names": dict(criterion_name_counter),
        "outcome_criteria_changes": list(criteria_out.get("rows") or []),
        "outcome_criteria_diagnostics": list(criteria_out.get("diagnostics") or []),
    }


def extract_document_metrics(
    experiment: Mapping[str, Any],
    checkpoint_index: Optional[Mapping[str, Any]] = None,
    *,
    lecturebank_id: str = "",
) -> Dict[str, Any]:
    return extract_experiment_metrics(
        experiment, checkpoint_index, lecturebank_id=lecturebank_id
    )


def _collect_attempts(
    lecturebank_id: str,
    candidate_ref: str,
    attempts: Sequence[Any],
    rows: List[Dict[str, Any]],
    dist_counter: Dict[Tuple[str, str], Counter],
) -> None:
    for att in attempts or []:
        if not isinstance(att, Mapping):
            continue
        raw_op = att.get("parsed_operation")
        canon, preserved = normalize_operation(raw_op)
        status = str(att.get("validation_status") or "") or "unknown"
        executed = bool(att.get("executed"))
        produced = bool(att.get("produced_counterfactual"))
        row = {
            "lecturebank_id": lecturebank_id,
            "candidate_ref": candidate_ref,
            "attempt_index": att.get("attempt_index"),
            "generated_query": att.get("generated_query"),
            "parsed_operation": preserved or raw_op,
            "parsed_operation_normalized": canon,
            "axis_figure4_operation": axis_figure4_label(canon),
            "validation_status": status,
            "executed": executed,
            "produced_counterfactual": produced,
            "counted_as_successful_round": bool(att.get("counted_as_successful_round")),
            "parser_error": att.get("parser_error"),
            "resolution_error": att.get("resolution_error"),
            "duplicate_error": att.get("duplicate_error"),
            "contract_error": att.get("contract_error"),
            "execution_error": att.get("execution_error"),
            "vocabulary_error": att.get("vocabulary_error"),
            "available_what_count": att.get("available_what_count"),
            "available_remove_count": att.get("available_remove_count"),
            "available_add_count": att.get("available_add_count"),
            "available_whatif_count": att.get("available_whatif_count"),
            "axis_interrogation_mode": att.get("axis_interrogation_mode"),
            "axis_interrogation_policy": att.get("axis_interrogation_policy"),
        }
        rows.append(row)
        dist_counter[("document", "proposed_query_operation")][canon] += 1
        dist_counter[("document", "validation_status")][status] += 1
        if executed:
            dist_counter[("document", "executed_query_operation")][canon] += 1


def _collect_comparisons(
    lecturebank_id: str,
    candidate_ref: str,
    comparisons: Sequence[Any],
    timing: Mapping[str, Any],
    checkpoint_index: Optional[Mapping[str, Any]],
    raw: Mapping[str, Any],
    rows: List[Dict[str, Any]],
    dist_counter: Dict[Tuple[str, str], Counter],
    criterion_name_counter: Counter,
) -> None:
    artifacts = list(raw.get("relevant_artifacts") or [])
    for i, cmp in enumerate(comparisons or []):
        if not isinstance(cmp, Mapping):
            continue
        sig = cmp.get("intervention_signature") if isinstance(cmp.get("intervention_signature"), Mapping) else {}
        op_raw = sig.get("operation")
        canon, preserved = normalize_operation(op_raw)
        target_ref = artifact_ref_key(sig.get("target_ref"))
        replacement_ref = artifact_ref_key(sig.get("replacement_ref"))
        finding = cmp.get("information_role_finding") if isinstance(cmp.get("information_role_finding"), Mapping) else {}
        qcc = cmp.get("question_content_comparison") if isinstance(cmp.get("question_content_comparison"), Mapping) else {}
        details = qcc.get("details") if isinstance(qcc.get("details"), Mapping) else {}
        ckpt = checkpoint_position(
            alias=sig.get("checkpoint_alias"),
            checkpoint_id=sig.get("checkpoint_id"),
            checkpoint_index=checkpoint_index,
        )
        art_type, art_src = resolve_target_artifact_type(
            finding=finding,
            target_ref=target_ref,
            relevant_artifacts=artifacts,
            signature=sig,
        )
        decision_cs = tribool(cmp.get("decision_control_stable"))
        criteria_cs = tribool(cmp.get("criteria_control_stable"))
        content_cs = tribool(cmp.get("subject_content_control_stable"))
        known_controls = [v for v in (decision_cs, criteria_cs, content_cs) if v is not None]
        full_cs: Optional[bool]
        if len(known_controls) < 3:
            full_cs = None
        else:
            full_cs = all(v is True for v in (decision_cs, criteria_cs, content_cs))

        evidence_status = cmp.get("evidence_status")
        valid = _is_valid_comparison(cmp)
        deltas = list(cmp.get("criterion_deltas") or [])
        criterion_any, decision_relevant_crit, changed_names = _criterion_effects(deltas)
        for name in changed_names:
            criterion_name_counter[name] += 1

        noop = cmp.get("noop") if isinstance(cmp.get("noop"), Mapping) else {}
        cf = cmp.get("counterfactual") if isinstance(cmp.get("counterfactual"), Mapping) else {}
        decision_effect_raw, decision_effect_supported = _decision_effect(
            noop, cf, decision_cs
        )
        content_fields = _content_effects(finding, qcc, details, content_cs)
        completion_mode, executed_nodes = match_replay_timing(
            timing,
            candidate_ref=candidate_ref,
            operation=preserved or canon,
            checkpoint_alias=ckpt.get("checkpoint_alias"),
            round_index=i + 1,
            target_ref=target_ref,
        )
        if not completion_mode:
            llm = cmp.get("llm_replay_diagnostics") if isinstance(cmp.get("llm_replay_diagnostics"), Mapping) else {}
            cf_llm = llm.get("counterfactual") if isinstance(llm.get("counterfactual"), Mapping) else {}
            completion_mode = str(cf_llm.get("replay_completion_mode") or "") or None
            executed_nodes = cf_llm.get("executed_node_count") if executed_nodes is None else executed_nodes

        round_index = i + 1
        if cmp.get("round") not in (None, ""):
            try:
                round_index = int(cmp.get("round"))
            except (TypeError, ValueError):
                pass

        row = {
            "lecturebank_id": lecturebank_id,
            "candidate_ref": candidate_ref,
            "round": round_index,
            "operation": canon,
            "operation_raw": preserved or op_raw,
            "target_ref": target_ref,
            "target_artifact_type": art_type,
            "target_artifact_type_source": art_src,
            "replacement_ref": replacement_ref,
            "boundary_raw": sig.get("boundary_raw"),
            **ckpt,
            "evidence_status": evidence_status,
            "decision_control_stable": decision_cs,
            "criteria_control_stable": criteria_cs,
            "subject_content_control_stable": content_cs,
            "full_control_stable": full_cs,
            "valid_comparison": valid,
            "criterion_effect_any": criterion_any,
            "decision_relevant_criterion_effect": decision_relevant_crit,
            "criterion_effect_any_supported": (
                True if criterion_any is True and criteria_cs is True
                else (False if criterion_any is False and criteria_cs is True else NA)
            ),
            "decision_relevant_criterion_effect_supported": (
                True if decision_relevant_crit is True and criteria_cs is True
                else (False if decision_relevant_crit is False and criteria_cs is True else NA)
            ),
            "content_effect_raw": content_fields["content_effect_raw"],
            "content_effect": content_fields["content_effect_supported"],
            "decision_effect_raw": decision_effect_raw,
            "decision_effect": decision_effect_supported,
            "question_content_effect_status": content_fields["question_content_effect_status"],
            "stem_effect": content_fields["stem_effect"],
            "correct_answer_effect": content_fields["correct_answer_effect"],
            "distractor_effect": content_fields["distractor_effect"],
            "answer_explanation_effect": content_fields["answer_explanation_effect"],
            "question_not_reproduced": content_fields["question_not_reproduced"],
            "information_role": finding.get("information_role"),
            "information_role_supported": finding.get("information_role_supported"),
            "causal_classification": finding.get("causal_classification"),
            "semantic_information_absent": finding.get("semantic_information_absent"),
            "derived_information_may_persist": finding.get("derived_information_may_persist"),
            "intervention_scope": finding.get("intervention_scope")
            or ((cmp.get("semantic_intervention_validity") or {}) or {}).get("intervention_scope"),
            "replay_completion_mode": completion_mode,
            "executed_node_count": executed_nodes,
        }
        rows.append(row)


def _is_valid_comparison(cmp: Mapping[str, Any]) -> bool:
    status = cmp.get("evidence_status")
    if status is None or str(status).strip() == "":
        return False
    return str(status) not in INVALID_EVIDENCE


def _criterion_effects(deltas: Sequence[Any]) -> Tuple[Optional[bool], Optional[bool], List[str]]:
    if not deltas:
        return NA, NA, []
    any_true = False
    any_known = False
    dr_true = False
    dr_known = False
    names: List[str] = []
    for delta in deltas:
        if not isinstance(delta, Mapping):
            continue
        flag = tribool(delta.get("counterfactual_differs_from_noop"))
        if flag is not None:
            any_known = True
            if flag:
                any_true = True
                names.append(str(delta.get("name") or ""))
        if tribool(delta.get("decision_relevant")) is True:
            if flag is not None:
                dr_known = True
                if flag:
                    dr_true = True
    return (
        True if any_true else (False if any_known else NA),
        True if dr_true else (False if dr_known else NA),
        [n for n in names if n],
    )


def _decision_effect(
    noop: Mapping[str, Any],
    cf: Mapping[str, Any],
    decision_control_stable: Optional[bool],
) -> Tuple[Optional[bool], Optional[bool]]:
    n_dec = noop.get("decision")
    c_dec = cf.get("decision")
    n_avail = n_dec not in (None, "")
    c_avail = c_dec not in (None, "")
    if not n_avail or not c_avail:
        raw = NA
    else:
        raw = str(n_dec) != str(c_dec)
    if decision_control_stable is True:
        supported = raw
    elif decision_control_stable is False:
        supported = NA
    else:
        supported = NA
    return raw, supported


def _content_effects(
    finding: Mapping[str, Any],
    qcc: Mapping[str, Any],
    details: Mapping[str, Any],
    content_control_stable: Optional[bool],
) -> Dict[str, Any]:
    status = (
        details.get("question_content_effect_status")
        or qcc.get("effect_status")
        or finding.get("question_content_effect_status")
    )
    stem = details.get("stem_effect") or finding.get("stem_effect")
    correct = details.get("correct_answer_effect") or finding.get("correct_answer_effect")
    dist = details.get("distractor_effect") or finding.get("distractor_effect")
    expl = details.get("answer_explanation_effect") or finding.get("answer_explanation_effect")
    not_reproduced = status == "question_not_reproduced" or details.get(
        "question_present_in_counterfactual"
    ) is False
    raw = finding.get("content_effect")
    if raw is None:
        if status in {
            "question_content_changed",
            "question_content_partially_supported",
            "question_not_reproduced",
        }:
            raw = True
        elif status == "question_content_unchanged":
            raw = False
        elif qcc.get("counterfactual_differs_from_noop") is True:
            raw = True
        elif qcc.get("counterfactual_differs_from_noop") is False:
            raw = False
        else:
            raw = NA
    raw_b = tribool(raw) if raw is not True and raw is not False else raw
    if content_control_stable is True:
        supported = raw_b
    else:
        supported = NA
    return {
        "question_content_effect_status": status,
        "stem_effect": stem,
        "correct_answer_effect": correct,
        "distractor_effect": dist,
        "answer_explanation_effect": expl,
        "question_not_reproduced": not_reproduced,
        "content_effect_raw": raw_b,
        "content_effect_supported": supported,
    }


def resolve_target_artifact_type(
    *,
    finding: Mapping[str, Any],
    target_ref: str,
    relevant_artifacts: Sequence[Any],
    signature: Mapping[str, Any],
) -> Tuple[Optional[str], str]:
    for src, value in (
        ("information_role_finding", finding.get("artifact_type")),
        ("signature", signature.get("target_artifact_type")),
    ):
        if value:
            return str(value), src
    tkey = str(target_ref or "")
    for art in relevant_artifacts or []:
        if not isinstance(art, Mapping):
            continue
        ref = str(art.get("ref") or artifact_ref_key(art.get("artifact_ref")) or "")
        if ref and tkey and (ref == tkey or tkey in ref or ref in tkey):
            atype = art.get("artifact_type") or art.get("type")
            if atype:
                return str(atype), "relevant_artifacts"
    if tkey:
        logical = tkey.split(":")[0] if ":" in tkey else tkey
        stem = re.split(r"[#/]", logical)[0]
        compact = re.sub(r"[^a-z0-9]", "", stem.lower())
        for prefix, label in KNOWN_PREFIX_TYPES.items():
            if compact.startswith(prefix) or prefix in compact:
                return label, "name_prefix_fallback"
    return None, "missing"


def match_replay_timing(
    timing: Mapping[str, Any],
    *,
    candidate_ref: str,
    operation: str,
    checkpoint_alias: Any,
    round_index: int,
    target_ref: str = "",
) -> Tuple[Optional[str], Any]:
    axis = timing.get("axis") if isinstance(timing.get("axis"), Mapping) else {}
    cases = list((axis or {}).get("cases") or [])
    alias = str(checkpoint_alias or "")
    op = str(operation or "").upper()
    cand = str(candidate_ref or "")
    tgt = str(target_ref or "")

    def _score(rd: Mapping[str, Any], case_target: str) -> int:
        score = 0
        blob = " ".join(
            [
                case_target,
                str(rd.get("target") or ""),
                str(rd.get("intervention") or ""),
                str(rd.get("checkpoint") or ""),
                str(rd.get("checkpoint_alias") or ""),
            ]
        )
        if cand and cand in blob:
            score += 4
        if tgt and tgt in blob:
            score += 3
        if alias and alias == str(rd.get("checkpoint_alias") or rd.get("checkpoint") or ""):
            score += 3
        if op and op in str(rd.get("intervention") or "").upper():
            score += 2
        try:
            if int(rd.get("round") or -1) == int(round_index):
                score += 2
        except (TypeError, ValueError):
            pass
        return score

    best = None
    best_score = 0
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        case_target = str(case.get("target") or "")
        for rd in case.get("rounds") or []:
            if not isinstance(rd, Mapping):
                continue
            score = _score(rd, case_target)
            if score > best_score:
                best_score = score
                best = rd
    if best is None or best_score < 2:
        for ev in (timing.get("replay_events") or []):
            if not isinstance(ev, Mapping):
                continue
            blob = " ".join(str(ev.get(k) or "") for k in ("target", "intervention", "checkpoint", "checkpoint_alias"))
            if cand and cand in blob:
                return (
                    str(ev.get("replay_completion_mode") or ev.get("completion_mode") or "") or None,
                    ev.get("executed_node_count"),
                )
        return None, None
    mode = str(best.get("replay_completion_mode") or best.get("completion_mode") or "") or None
    return mode, best.get("executed_node_count")


def _collect_provenance(
    lecturebank_id: str,
    candidate_ref: str,
    raw: Mapping[str, Any],
    rows: List[Dict[str, Any]],
    dist_counter: Dict[Tuple[str, str], Counter],
) -> bool:
    relations = raw.get("provenance_relations")
    slicing_error = False
    warnings = [str(w).lower() for w in (raw.get("warnings") or [])]
    if raw.get("provenance_included") is False:
        slicing_error = True
    if any("slic" in w and "fail" in w for w in warnings) or any(
        "provenance" in w and "fail" in w for w in warnings
    ):
        slicing_error = True
    if relations is None and slicing_error:
        dist_counter[("document", "provenance_slicing_error")]["error"] += 1
        return True
    artifacts = {
        str(a.get("ref") or artifact_ref_key(a)): a
        for a in (raw.get("relevant_artifacts") or [])
        if isinstance(a, Mapping)
    }
    for rel in relations or []:
        if not isinstance(rel, Mapping):
            continue
        art_ref = artifact_ref_key(rel.get("artifact_ref")) or str(rel.get("artifact_ref") or "")
        art = artifacts.get(art_ref) or {}
        producer = rel.get("producer_occurrence_id")
        consumer = rel.get("consumer_occurrence_id")
        pnode = occurrence_node_name(producer)
        cnode = occurrence_node_name(consumer)
        relation = rel.get("relation")
        row = {
            "lecturebank_id": lecturebank_id,
            "candidate_ref": candidate_ref,
            "relation": relation,
            "artifact_ref": art_ref,
            "artifact_type": art.get("artifact_type"),
            "producer_occurrence_id": producer,
            "consumer_occurrence_id": consumer,
            "producer_node": pnode,
            "consumer_node": cnode,
        }
        rows.append(row)
        if relation:
            dist_counter[("document", "provenance_relation")][str(relation)] += 1
        if pnode or cnode:
            dist_counter[("document", "producer_consumer_pair")][f"{pnode}->{cnode}"] += 1
        atype = art.get("artifact_type")
        if atype:
            dist_counter[("document", "relevant_artifact_type")][str(atype)] += 1
    return slicing_error


def _query_distribution(attempts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    proposed = Counter(str(a.get("parsed_operation_normalized") or "UNPARSED") for a in attempts)
    executed = Counter(
        str(a.get("parsed_operation_normalized") or "UNPARSED")
        for a in attempts
        if a.get("executed")
    )
    proposed_recog = sum(proposed[op] for op in RECOGNIZED_QUERY_OPS)
    executed_recog = sum(executed[op] for op in RECOGNIZED_QUERY_OPS)
    proposals = sum(
        1
        for a in attempts
        if a.get("parsed_operation_normalized") in CF_OPERATIONS
    )
    executed_cf = sum(
        1
        for a in attempts
        if a.get("parsed_operation_normalized") in CF_OPERATIONS
        and a.get("executed")
        and a.get("produced_counterfactual")
    )
    fail_counts = Counter()
    for a in attempts:
        status = str(a.get("validation_status") or "unknown")
        fail_counts[status] += 1
        for key in (
            "parser_error",
            "resolution_error",
            "duplicate_error",
            "contract_error",
            "execution_error",
        ):
            if a.get(key):
                fail_counts[key] += 1
        if status == "not_currently_executable":
            fail_counts["not_currently_executable"] += 1
    out: Dict[str, Any] = {
        "interrogation_attempt_count": len(attempts),
        "proposed_remove_count": proposed["REMOVE"],
        "proposed_add_count": proposed["ADD"],
        "proposed_replace_count": proposed["REPLACE"],
        "proposed_what_count": proposed["WHAT"],
        "proposed_done_count": proposed["DONE"],
        "proposed_unparsed_count": proposed["UNPARSED"],
        "proposed_recognized_non_done_count": proposed_recog,
        "proposed_remove_pct": pct(proposed["REMOVE"], proposed_recog),
        "proposed_add_pct": pct(proposed["ADD"], proposed_recog),
        "proposed_replace_pct": pct(proposed["REPLACE"], proposed_recog),
        "proposed_what_pct": pct(proposed["WHAT"], proposed_recog),
        "executed_remove_count": executed["REMOVE"],
        "executed_add_count": executed["ADD"],
        "executed_replace_count": executed["REPLACE"],
        "executed_what_count": executed["WHAT"],
        "executed_done_count": executed["DONE"],
        "executed_unparsed_count": executed["UNPARSED"],
        "executed_recognized_non_done_count": executed_recog,
        "executed_remove_pct": pct(executed["REMOVE"], executed_recog),
        "executed_add_pct": pct(executed["ADD"], executed_recog),
        "executed_replace_pct": pct(executed["REPLACE"], executed_recog),
        "executed_what_pct": pct(executed["WHAT"], executed_recog),
        "counterfactual_intervention_proposals": proposals,
        "executed_counterfactual_interventions": executed_cf,
        "intervention_validity_rate": ratio(executed_cf, proposals),
        "validation_parser_error_count": fail_counts.get("parser_error", 0),
        "validation_resolution_error_count": fail_counts.get("resolution_error", 0),
        "validation_duplicate_error_count": fail_counts.get("duplicate_error", 0),
        "validation_contract_error_count": fail_counts.get("contract_error", 0),
        "validation_execution_error_count": fail_counts.get("execution_error", 0),
        "validation_not_currently_executable_count": fail_counts.get("not_currently_executable", 0),
    }
    known_fail_status = {
        "ok",
        "valid",
        "executed",
        "success",
        "parser_error",
        "resolution_error",
        "duplicate_error",
        "contract_error",
        "execution_error",
        "not_currently_executable",
    }
    status_other = sum(
        1
        for a in attempts
        if str(a.get("validation_status") or "unknown") not in known_fail_status
    )
    out["validation_other_status_count"] = status_other
    return out


def _replay_stability(
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    recorded_cf = 0
    recorded_success = 0
    recorded_valid = 0
    have_recorded = False
    for case in cases:
        raw = _axis_raw(case)
        summary = (_axis_condition(case).get("evidence_summary") or {}) if isinstance(_axis_condition(case).get("evidence_summary"), Mapping) else {}
        src = raw if raw.get("counterfactual_executions") is not None or raw.get("counterfactual_count") is not None else summary
        if src.get("counterfactual_executions") is not None or src.get("counterfactual_count") is not None:
            have_recorded = True
            recorded_cf += int(src.get("counterfactual_executions") or src.get("counterfactual_count") or 0)
        if src.get("successful_replay_count") is not None:
            have_recorded = True
            recorded_success += int(src.get("successful_replay_count") or 0)
        if src.get("valid_replay_comparisons") is not None:
            have_recorded = True
            recorded_valid += int(src.get("valid_replay_comparisons") or 0)
        elif case.get("successful_replay_count") is not None:
            recorded_success += int(case.get("successful_replay_count") or 0)
            have_recorded = True
        if case.get("counterfactual_count") is not None and not (
            src.get("counterfactual_executions") is not None or src.get("counterfactual_count") is not None
        ):
            recorded_cf += int(case.get("counterfactual_count") or 0)
            have_recorded = True

    valid_rows = [r for r in rows if r.get("valid_comparison")]
    cf_from_rows = len(rows)
    success_from_rows = len(valid_rows)
    cf_n = recorded_cf if have_recorded and recorded_cf else cf_from_rows
    success_n = recorded_success if have_recorded and recorded_success else success_from_rows
    valid_n = recorded_valid if have_recorded and recorded_valid else success_from_rows

    def _stab(field: str) -> Tuple[int, int, int, Optional[float]]:
        true_n = sum(1 for r in rows if r.get(field) is True)
        false_n = sum(1 for r in rows if r.get(field) is False)
        miss_n = sum(1 for r in rows if r.get(field) is not True and r.get(field) is not False)
        return true_n, false_n, miss_n, ratio(true_n, true_n + false_n)

    d_t, d_f, d_m, d_r = _stab("decision_control_stable")
    c_t, c_f, c_m, c_r = _stab("criteria_control_stable")
    s_t, s_f, s_m, s_r = _stab("subject_content_control_stable")
    f_t, f_f, f_m, f_r = _stab("full_control_stable")

    modes = Counter(str(r.get("replay_completion_mode") or "unknown") for r in rows)
    statuses = Counter(str(r.get("evidence_status") or "missing") for r in rows)
    node_vals = [
        int(r["executed_node_count"])
        for r in rows
        if r.get("executed_node_count") not in (None, "")
    ]
    replay_fail = sum(1 for r in rows if not r.get("valid_comparison"))
    return {
        "counterfactual_executions": cf_n,
        "successful_replay_count": success_n,
        "valid_replay_comparisons": valid_n,
        "completed_replay_rate": ratio(success_n, cf_n),
        "valid_comparison_rate": ratio(valid_n, cf_n),
        "noop_decision_stability_true": d_t,
        "noop_decision_stability_false": d_f,
        "noop_decision_stability_unavailable": d_m,
        "noop_decision_stability_rate": d_r,
        "noop_criteria_stability_true": c_t,
        "noop_criteria_stability_false": c_f,
        "noop_criteria_stability_unavailable": c_m,
        "noop_criteria_stability_rate": c_r,
        "noop_content_stability_true": s_t,
        "noop_content_stability_false": s_f,
        "noop_content_stability_unavailable": s_m,
        "noop_content_stability_rate": s_r,
        "full_noop_stability_true": f_t,
        "full_noop_stability_false": f_f,
        "full_noop_stability_unavailable": f_m,
        "full_noop_stability_rate": f_r,
        "replay_failure_count": replay_fail,
        "replay_completion_mode_target_observable": modes.get("target_observable", 0),
        "replay_completion_mode_graph_end": modes.get("graph_end", 0),
        "replay_completion_mode_fallback_full": modes.get("fallback_full", 0) + modes.get("fallback/full", 0),
        "replay_completion_mode_error_unknown": modes.get("error", 0) + modes.get("unknown", 0) + modes.get("None", 0),
        "executed_node_count_mean": _mean(node_vals),
        "executed_node_count_median": _median(node_vals),
        "evidence_status_counts": dict(statuses),
    }


def _intervention_effects(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("valid_comparison")]
    def _count(field: str, *, supported: bool = False) -> Tuple[int, int, Optional[float]]:
        if supported:
            true_n = sum(1 for r in valid if r.get(field) is True)
            known = [r for r in valid if r.get(field) is True or r.get(field) is False]
            return true_n, len(known), ratio(true_n, len(known))
        true_n = sum(1 for r in valid if r.get(field) is True)
        known = [r for r in valid if r.get(field) is True or r.get(field) is False]
        return true_n, len(known), ratio(true_n, len(known))

    dec_raw_n, dec_raw_d, dec_raw_r = _count("decision_effect_raw")
    dec_n, dec_d, dec_r = _count("decision_effect")
    crit_n, crit_d, crit_r = _count("criterion_effect_any")
    crit_s_n, crit_s_d, crit_s_r = _count("criterion_effect_any_supported")
    dcrit_n, dcrit_d, dcrit_r = _count("decision_relevant_criterion_effect")
    dcrit_s_n, dcrit_s_d, dcrit_s_r = _count("decision_relevant_criterion_effect_supported")
    cont_raw_n, cont_raw_d, cont_raw_r = _count("content_effect_raw")
    cont_n, cont_d, cont_r = _count("content_effect")

    def _changed_rate(field: str) -> Tuple[int, int, Optional[float]]:
        true_n = sum(1 for r in valid if str(r.get(field) or "") == "changed")
        known = [
            r
            for r in valid
            if str(r.get(field) or "") in {"changed", "unchanged"}
        ]
        return true_n, len(known), ratio(true_n, len(known))

    stem_n, stem_d, stem_r = _changed_rate("stem_effect")
    corr_n, corr_d, corr_r = _changed_rate("correct_answer_effect")
    dist_n, dist_d, dist_r = _changed_rate("distractor_effect")
    expl_n, expl_d, expl_r = _changed_rate("answer_explanation_effect")
    not_repro = sum(1 for r in valid if r.get("question_not_reproduced") is True)
    not_repro_d = len(valid)

    stable = [r for r in valid if r.get("full_control_stable") is True]
    yield_n = 0
    for r in stable:
        if (
            r.get("content_effect") is True
            or r.get("criterion_effect_any_supported") is True
            or r.get("decision_effect") is True
        ):
            yield_n += 1
    return {
        "decision_effect_count_raw": dec_raw_n,
        "decision_effect_rate_raw": dec_raw_r,
        "decision_effect_count": dec_n,
        "decision_effect_rate": dec_r,
        "decision_effect_denominator": dec_d,
        "decision_effect_raw_denominator": dec_raw_d,
        "criterion_effect_count_raw": crit_n,
        "criterion_effect_rate_raw": crit_r,
        "criterion_effect_count": crit_s_n,
        "criterion_effect_rate": crit_s_r,
        "criterion_effect_denominator": crit_s_d,
        "criterion_effect_raw_denominator": crit_d,
        "decision_relevant_criterion_effect_count_raw": dcrit_n,
        "decision_relevant_criterion_effect_rate_raw": dcrit_r,
        "decision_relevant_criterion_effect_count": dcrit_s_n,
        "decision_relevant_criterion_effect_rate": dcrit_s_r,
        "decision_relevant_criterion_effect_denominator": dcrit_s_d,
        "content_effect_count_raw": cont_raw_n,
        "content_effect_rate_raw": cont_raw_r,
        "content_effect_count": cont_n,
        "content_effect_rate": cont_r,
        "content_effect_denominator": cont_d,
        "content_effect_raw_denominator": cont_raw_d,
        "stem_change_count": stem_n,
        "stem_change_rate": stem_r,
        "stem_change_denominator": stem_d,
        "correct_answer_change_count": corr_n,
        "correct_answer_change_rate": corr_r,
        "correct_answer_change_denominator": corr_d,
        "distractor_change_count": dist_n,
        "distractor_change_rate": dist_r,
        "distractor_change_denominator": dist_d,
        "explanation_change_count": expl_n,
        "explanation_change_rate": expl_r,
        "explanation_change_denominator": expl_d,
        "question_not_reproduced_count": not_repro,
        "question_not_reproduced_rate": ratio(not_repro, not_repro_d),
        "effect_yield_any_count": yield_n,
        "effect_yield_any": ratio(yield_n, len(stable)),
        "effect_yield_any_denominator": len(stable),
        "valid_control_stable_comparisons": len(stable),
        "effect_discovery_recall_at_5": NA,
        "causal_precision": NA,
        "causal_recall": NA,
        "causal_classification_accuracy": NA,
        "macro_f1": NA,
    }


def _checkpoint_aggregates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    executed = [
        r
        for r in rows
        if r.get("operation") in CF_OPERATIONS
        and r.get("checkpoint_position_pct") is not None
    ]
    pos = [float(r["checkpoint_position_pct"]) for r in executed]
    prog = [
        float(r["checkpoint_progress_pct"])
        for r in executed
        if r.get("checkpoint_progress_pct") is not None
    ]
    by_op: Dict[str, List[float]] = defaultdict(list)
    by_type: Dict[str, List[float]] = defaultdict(list)
    for r in executed:
        by_op[str(r.get("operation") or "")].append(float(r["checkpoint_position_pct"]))
        by_type[str(r.get("target_artifact_type") or "unknown")].append(
            float(r["checkpoint_position_pct"])
        )
    out = {
        "checkpoint_position_n": len(pos),
        "checkpoint_position_pct_mean": _mean(pos),
        "checkpoint_position_pct_median": _median(pos),
        "checkpoint_position_pct_min": min(pos) if pos else NA,
        "checkpoint_position_pct_max": max(pos) if pos else NA,
        "checkpoint_progress_n": len(prog),
        "checkpoint_progress_pct_mean": _mean(prog),
        "checkpoint_progress_pct_median": _median(prog),
        "checkpoint_progress_pct_min": min(prog) if prog else NA,
        "checkpoint_progress_pct_max": max(prog) if prog else NA,
    }
    for op, vals in by_op.items():
        out[f"checkpoint_position_pct_mean_op_{op}"] = _mean(vals)
    for atype, vals in by_type.items():
        safe = re.sub(r"[^A-Za-z0-9]+", "_", atype)
        out[f"checkpoint_position_pct_mean_type_{safe}"] = _mean(vals)
    return out


def _target_artifact_distribution(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    executed = [r for r in rows if r.get("operation") in CF_OPERATIONS]
    counts = Counter(str(r.get("target_artifact_type") or "unknown") for r in executed)
    n = len(executed)
    out: Dict[str, Any] = {"executed_counterfactual_comparison_count": n}
    for label in ("EvidenceChunk", "KnowledgeAtom", "MCQBlueprint", "CandidateMCQ", "unknown"):
        out[f"target_{label}_count"] = counts.get(label, 0)
        out[f"target_{label}_pct"] = pct(counts.get(label, 0), n)
    other = n - sum(counts.get(l, 0) for l in ("EvidenceChunk", "KnowledgeAtom", "MCQBlueprint", "CandidateMCQ"))
    # other includes unknown already in unknown; keep residual named types
    named = {"EvidenceChunk", "KnowledgeAtom", "MCQBlueprint", "CandidateMCQ", "unknown"}
    named_other = sum(v for k, v in counts.items() if k not in named)
    out["target_other_count"] = named_other
    out["target_other_pct"] = pct(named_other, n)
    out["target_artifact_type_counts"] = dict(counts)
    return out


def _provenance_integrity(
    edges: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    slicing_errors: int,
) -> Dict[str, Any]:
    complete = 0
    for e in edges:
        if (
            str(e.get("relation") or "").strip()
            and str(e.get("artifact_ref") or "").strip()
            and str(e.get("producer_occurrence_id") or "").strip()
            and str(e.get("consumer_occurrence_id") or "").strip()
        ):
            complete += 1
    arts = set()
    producers = set()
    consumers = set()
    pnodes = set()
    cnodes = set()
    for e in edges:
        if e.get("artifact_ref"):
            arts.add(str(e.get("artifact_ref")))
        if e.get("producer_occurrence_id"):
            producers.add(str(e.get("producer_occurrence_id")))
        if e.get("consumer_occurrence_id"):
            consumers.add(str(e.get("consumer_occurrence_id")))
        if e.get("producer_node"):
            pnodes.add(str(e.get("producer_node")))
        if e.get("consumer_node"):
            cnodes.add(str(e.get("consumer_node")))
    # also unique relevant artifacts from cases
    relevant = set()
    for case in cases:
        raw = _axis_raw(case)
        for a in raw.get("relevant_artifacts") or []:
            if isinstance(a, Mapping):
                key = str(a.get("ref") or artifact_ref_key(a) or "")
            else:
                key = str(a)
            if key:
                relevant.add(key)
    in_slice_n = 0
    in_slice_d = 0
    has_use_n = 0
    has_use_d = 0
    edge_refs = {str(e.get("artifact_ref") or "") for e in edges}
    for case in cases:
        raw = _axis_raw(case)
        rel_keys = set()
        for a in raw.get("relevant_artifacts") or []:
            if isinstance(a, Mapping):
                rel_keys.add(str(a.get("ref") or artifact_ref_key(a) or ""))
            else:
                rel_keys.add(str(a))
        case_edges = {
            str(e.get("artifact_ref") or "")
            for e in (raw.get("provenance_relations") or [])
            if isinstance(e, Mapping)
        }
        for cmp in raw.get("replay_comparisons") or []:
            if not isinstance(cmp, Mapping):
                continue
            sig = cmp.get("intervention_signature") if isinstance(cmp.get("intervention_signature"), Mapping) else {}
            canon, _ = normalize_operation(sig.get("operation"))
            if canon not in FACTUAL_TARGET_OPS:
                continue
            if not cmp.get("evidence_status") and not sig:
                continue
            # executed factual-target interventions
            tref = artifact_ref_key(sig.get("target_ref"))
            if not tref:
                continue
            in_slice_d += 1
            if tref in rel_keys or any(tref in k or k in tref for k in rel_keys if k):
                in_slice_n += 1
            has_use_d += 1
            if tref in case_edges or any(tref in k or k in tref for k in case_edges if k):
                has_use_n += 1
    return {
        "provenance_relation_count": len(edges),
        "unique_relevant_artifact_count": len(relevant) or len(arts),
        "unique_producer_occurrence_count": len(producers),
        "unique_consumer_occurrence_count": len(consumers),
        "unique_producer_node_count": len(pnodes),
        "unique_consumer_node_count": len(cnodes),
        "provenance_relation_completeness_rate": ratio(complete, len(edges)),
        "provenance_complete_relation_count": complete,
        "intervention_target_in_slice_count": in_slice_n,
        "intervention_target_in_slice_rate": ratio(in_slice_n, in_slice_d),
        "intervention_target_in_slice_denominator": in_slice_d,
        "intervention_target_has_recorded_use_count": has_use_n,
        "intervention_target_has_recorded_use_rate": ratio(has_use_n, has_use_d),
        "intervention_target_has_recorded_use_denominator": has_use_d,
        "provenance_slicing_error_count": slicing_errors,
    }


def _replay_determinism(diag: Mapping[str, Any]) -> Dict[str, Any]:
    def _i(name: str) -> Optional[int]:
        if name not in diag or diag.get(name) is None:
            return None
        try:
            return int(diag.get(name))
        except (TypeError, ValueError):
            return None

    facts = {
        "factual_llm_request_count": _i("factual_llm_request_count"),
        "factual_llm_unique_request_count": _i("factual_llm_unique_request_count"),
        "factual_llm_memo_hits": _i("factual_llm_memo_hits"),
        "noop_llm_request_count": _i("noop_llm_request_count"),
        "noop_factual_response_hits": _i("noop_factual_response_hits"),
        "noop_llm_request_misses": _i("noop_llm_request_misses"),
        "noop_live_provider_calls": _i("noop_live_provider_calls"),
        "counterfactual_llm_request_count": _i("counterfactual_llm_request_count"),
        "counterfactual_factual_response_hits": _i("counterfactual_factual_response_hits"),
        "counterfactual_novel_request_count": _i("counterfactual_novel_request_count"),
        "counterfactual_memo_hits": _i("counterfactual_memo_hits"),
        "counterfactual_live_provider_calls": _i("counterfactual_live_provider_calls"),
    }
    facts.update(
        {
            "noop_factual_ledger_hit_rate": ratio(
                facts["noop_factual_response_hits"], facts["noop_llm_request_count"]
            ),
            "noop_miss_rate": ratio(
                facts["noop_llm_request_misses"], facts["noop_llm_request_count"]
            ),
            "noop_live_provider_rate": ratio(
                facts["noop_live_provider_calls"], facts["noop_llm_request_count"]
            ),
            "cf_factual_reuse_rate": ratio(
                facts["counterfactual_factual_response_hits"],
                facts["counterfactual_llm_request_count"],
            ),
            "cf_novel_request_rate": ratio(
                facts["counterfactual_novel_request_count"],
                facts["counterfactual_llm_request_count"],
            ),
            "cf_memo_hit_rate": ratio(
                facts["counterfactual_memo_hits"],
                facts["counterfactual_llm_request_count"],
            ),
            "cf_live_provider_rate": ratio(
                facts["counterfactual_live_provider_calls"],
                facts["counterfactual_llm_request_count"],
            ),
        }
    )
    return facts


def _timing_metrics(timing: Mapping[str, Any]) -> Dict[str, Any]:
    exp = timing.get("experiment") if isinstance(timing.get("experiment"), Mapping) else {}
    axis = timing.get("axis") if isinstance(timing.get("axis"), Mapping) else {}
    totals = axis.get("totals") if isinstance(axis.get("totals"), Mapping) else {}
    conds = timing.get("conditions") if isinstance(timing.get("conditions"), Mapping) else {}

    def _g(*keys: str) -> Optional[float]:
        for src in (exp, totals, conds, timing):
            if not isinstance(src, Mapping):
                continue
            for k in keys:
                if k in src and src.get(k) not in (None, ""):
                    try:
                        return float(src.get(k))
                    except (TypeError, ValueError):
                        continue
        return NA

    return {
        "factual_workflow_s": _g("factual_workflow_s"),
        "experiment_total_s": _g("experiment_total_s", "total_s"),
        "trace_total_s": _g("trace_s", "trace_total_s"),
        "trace_provenance_total_s": _g("trace_provenance_s", "trace_provenance_total_s"),
        "axis_total_s": _g("axis_total_s"),
        "interrogation_total_s": _g("interrogation_total_s", "interrogation_s"),
        "intervention_resolution_total_s": _g("intervention_resolution_total_s"),
        "noop_total_s": _g("noop_total_s", "noop_s"),
        "counterfactual_replay_total_s": _g("counterfactual_replay_total_s", "replay_s", "axis_replay_s"),
        "comparison_total_s": _g("comparison_total_s", "comparison_s"),
        "intermediate_synthesis_total_s": _g("intermediate_synthesis_total_s"),
        "final_synthesis_s": _g("final_synthesis_s"),
        "explanation_llm_s": _g("explanation_llm_s", "axis_explanation_llm_s"),
        "workflow_replay_s": _g("workflow_replay_s", "axis_replay_s"),
        "axis_replay_pct": _g("axis_replay_pct", "workflow_replay_pct"),
        "axis_explanation_llm_pct": _g("axis_explanation_llm_pct", "explanation_layer_pct"),
        "other_processing_pct": _g("other_processing_pct", "axis_non_llm_non_replay_pct"),
    }


def _synthesis_from_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(summary, Mapping):
        return {"_filled": False}
    keys = (
        "synthesis_initial_valid",
        "synthesis_repair_attempted",
        "synthesis_repair_valid",
        "synthesis_used_fallback",
        "synthesis_used_mechanical_fallback",
        "synthesis_faithfulness_warning_count",
        "participant_word_count",
        "participant_sentence_count",
        "participant_presentation_valid",
        "participant_presentation_violations",
    )
    row = {k: summary.get(k) for k in keys}
    row["_filled"] = any(summary.get(k) is not None for k in keys if k != "participant_presentation_violations")
    return row


def _synthesis_aggregates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    filled = [r for r in rows if r.get("_filled")]
    n = len(filled)

    def _bool_rate(name: str) -> Tuple[int, int, Optional[float]]:
        true_n = sum(1 for r in filled if r.get(name) is True)
        known = sum(1 for r in filled if r.get(name) is True or r.get(name) is False)
        return true_n, known, ratio(true_n, known)

    init_n, init_d, init_r = _bool_rate("synthesis_initial_valid")
    rep_n, rep_d, rep_r = _bool_rate("synthesis_repair_attempted")
    reps_n, reps_d, reps_r = _bool_rate("synthesis_repair_valid")
    fb_n, fb_d, fb_r = _bool_rate("synthesis_used_fallback")
    if fb_d == 0:
        fb_n, fb_d, fb_r = _bool_rate("synthesis_used_mechanical_fallback")
    pres_n, pres_d, pres_r = _bool_rate("participant_presentation_valid")
    warn_n = sum(
        1
        for r in filled
        if (r.get("synthesis_faithfulness_warning_count") or 0) not in (None, 0)
        and int(r.get("synthesis_faithfulness_warning_count") or 0) > 0
    )
    words = [
        int(r["participant_word_count"])
        for r in filled
        if r.get("participant_word_count") not in (None, "")
    ]
    sents = [
        int(r["participant_sentence_count"])
        for r in filled
        if r.get("participant_sentence_count") not in (None, "")
    ]
    return {
        "synthesis_case_count": n,
        "initial_synthesis_valid_count": init_n,
        "initial_synthesis_valid_rate": init_r,
        "repair_attempt_count": rep_n,
        "repair_attempt_rate": rep_r,
        "repair_success_count": reps_n,
        "repair_success_rate": ratio(reps_n, rep_n) if rep_n else (ratio(reps_n, reps_d) if reps_d else NA),
        "fallback_count": fb_n,
        "fallback_rate": fb_r,
        "presentation_valid_count": pres_n,
        "presentation_valid_rate": pres_r,
        "cases_with_faithfulness_warning_count": warn_n,
        "cases_with_faithfulness_warning_rate": ratio(warn_n, n),
        "participant_word_count_mean": _mean(words),
        "participant_word_count_median": _median(words),
        "participant_sentence_count_mean": _mean(sents),
        "participant_sentence_count_median": _median(sents),
    }


def _distribution_from_rows(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, int]:
    return dict(Counter(str(r.get(field) or "missing") for r in rows if r.get(field) not in (None, "")))


def _prefixed_dist(prefix: str, counts: Mapping[str, int]) -> Dict[str, Any]:
    total = sum(counts.values())
    out: Dict[str, Any] = {f"{prefix}_total_labeled": total}
    for key, n in counts.items():
        safe = re.sub(r"[^A-Za-z0-9]+", "_", str(key))
        out[f"{prefix}_{safe}_count"] = n
        out[f"{prefix}_{safe}_pct"] = pct(n, total)
    return out


def _round_metrics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_round: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        try:
            rnd = int(r.get("round") or 0)
        except (TypeError, ValueError):
            rnd = 0
        by_round[rnd].append(r)
    out = []
    for rnd in sorted(by_round):
        group = by_round[rnd]
        valid = [g for g in group if g.get("valid_comparison")]
        ops = Counter(str(g.get("operation") or "") for g in group)
        types = Counter(str(g.get("target_artifact_type") or "unknown") for g in group)
        pos = [
            float(g["checkpoint_position_pct"])
            for g in group
            if g.get("checkpoint_position_pct") is not None
        ]
        full_true = sum(1 for g in group if g.get("full_control_stable") is True)
        full_known = sum(
            1 for g in group if g.get("full_control_stable") is True or g.get("full_control_stable") is False
        )
        roles = Counter(str(g.get("information_role") or "") for g in group if g.get("information_role"))
        causal = Counter(
            str(g.get("causal_classification") or "")
            for g in group
            if g.get("causal_classification")
        )
        out.append(
            {
                "round": rnd,
                "comparison_count": len(group),
                "valid_comparison_count": len(valid),
                "valid_comparison_rate": ratio(len(valid), len(group)),
                "operation_distribution": dict(ops),
                "target_artifact_type_distribution": dict(types),
                "mean_checkpoint_position_pct": _mean(pos),
                "full_noop_stability_rate": ratio(full_true, full_known),
                "content_effect_rate": ratio(
                    sum(1 for g in valid if g.get("content_effect") is True),
                    sum(1 for g in valid if g.get("content_effect") is True or g.get("content_effect") is False),
                ),
                "criterion_effect_rate": ratio(
                    sum(1 for g in valid if g.get("criterion_effect_any_supported") is True),
                    sum(
                        1
                        for g in valid
                        if g.get("criterion_effect_any_supported") is True
                        or g.get("criterion_effect_any_supported") is False
                    ),
                ),
                "decision_effect_rate": ratio(
                    sum(1 for g in valid if g.get("decision_effect") is True),
                    sum(1 for g in valid if g.get("decision_effect") is True or g.get("decision_effect") is False),
                ),
                "causal_classification_distribution": dict(causal),
                "information_role_distribution": dict(roles),
            }
        )
    return out


def _flatten_distributions(
    lecturebank_id: str,
    dist_counter: Mapping[Tuple[str, str], Counter],
    attempts: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    proposed_recog = sum(
        1
        for a in attempts
        if a.get("parsed_operation_normalized") in RECOGNIZED_QUERY_OPS
    )
    executed_recog = sum(
        1
        for a in attempts
        if a.get("executed") and a.get("parsed_operation_normalized") in RECOGNIZED_QUERY_OPS
    )
    denoms = {
        "proposed_query_operation": proposed_recog,
        "executed_query_operation": executed_recog,
        "validation_status": len(attempts),
        "target_artifact_type": sum(1 for c in comparisons if c.get("operation") in CF_OPERATIONS),
        "evidence_status": len(comparisons),
        "causal_classification": sum(1 for c in comparisons if c.get("causal_classification")),
        "information_role": sum(1 for c in comparisons if c.get("information_role")),
        "question_content_effect_status": sum(
            1 for c in comparisons if c.get("question_content_effect_status")
        ),
        "replay_completion_mode": sum(1 for c in comparisons if c.get("replay_completion_mode")),
        "provenance_relation": len(edges),
        "producer_consumer_pair": len(edges),
        "termination_reason": None,
    }
    for (scope, dim), counter in dist_counter.items():
        denom = denoms.get(dim)
        if denom is None:
            denom = sum(counter.values())
        # DONE excluded from proposed/executed recognized denominators
        for value, count in counter.items():
            use_den = denom
            if dim in {"proposed_query_operation", "executed_query_operation"} and value == "DONE":
                use_den = NA
            rows.append(
                {
                    "scope": scope,
                    "lecturebank_id": lecturebank_id,
                    "dimension": dim,
                    "value": value,
                    "count": count,
                    "denominator": use_den,
                    "percentage": pct(count, use_den) if use_den not in (None, 0) else NA,
                }
            )
    return rows


def pool_rates(rows: Sequence[Mapping[str, Any]], num_key: str, den_key: str) -> Optional[float]:
    nums = []
    dens = []
    for r in rows:
        n = r.get(num_key)
        d = r.get(den_key)
        if n is None or d is None:
            continue
        try:
            nums.append(float(n))
            dens.append(float(d))
        except (TypeError, ValueError):
            continue
    return ratio(sum(nums), sum(dens))


def describe(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {"mean": NA, "stdev": NA, "min": NA, "max": NA, "n": 0}
    stdev = statistics.stdev(clean) if len(clean) >= 2 else NA
    return {
        "mean": statistics.mean(clean),
        "stdev": stdev,
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else NA


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else NA
