"""Per-criterion AXIS replay export from stored criterion_deltas.

Evaluation-only. Does not change AXIS interrogation, replay, acceptance rules,
or existing criterion-effect aggregates.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mas_langgraph.evaluation.axis_metrics import (
    NA,
    _axis_raw,
    _candidate_ref,
    _is_valid_comparison,
    artifact_ref_key,
    normalize_operation,
    ratio,
    tribool,
)
from mas_langgraph.evaluation.concept_change import (
    _all_comparison_copies,
    _concept_signature,
    _primary_comparisons,
)

OUTCOME_CRITERIA_FIELDS = [
    "lecturebank_id",
    "run_id",
    "candidate_ref",
    "intervention_signature",
    "round",
    "operation",
    "target_ref",
    "replacement_ref",
    "checkpoint_id",
    "checkpoint_alias",
    "criterion_name",
    "factual_value",
    "noop_value",
    "counterfactual_value",
    "decision_relevant",
    "noop_differs_from_factual",
    "counterfactual_differs_from_noop",
    "valid_comparison",
    "full_control_stable",
    "criterion_change_supported",
]


def comparison_signature(payload: Optional[Mapping[str, Any]]) -> str:
    """Full intervention signature: operation|target|replacement|checkpoint."""
    return _concept_signature(payload)


def _full_control_stable(cmp: Mapping[str, Any]) -> Optional[bool]:
    decision_cs = tribool(cmp.get("decision_control_stable"))
    criteria_cs = tribool(cmp.get("criteria_control_stable"))
    content_cs = tribool(cmp.get("subject_content_control_stable"))
    known = [v for v in (decision_cs, criteria_cs, content_cs) if v is not None]
    if len(known) < 3:
        return NA
    return all(v is True for v in (decision_cs, criteria_cs, content_cs))


def _has_criterion_deltas(cmp: Mapping[str, Any]) -> bool:
    return "criterion_deltas" in cmp and isinstance(cmp.get("criterion_deltas"), list)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _field_present(delta: Mapping[str, Any], *names: str) -> bool:
    return any(name in delta for name in names)


def _delta_value(delta: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in delta:
            return delta[name]
    return NA


def _delta_snapshot(delta: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "criterion_name": str(delta.get("name") or ""),
        "factual_value": _delta_value(delta, "factual", "factual_value"),
        "noop_value": _delta_value(delta, "noop", "noop_value"),
        "counterfactual_value": _delta_value(delta, "counterfactual", "counterfactual_value"),
        "decision_relevant": (
            tribool(delta.get("decision_relevant"))
            if _field_present(delta, "decision_relevant")
            else NA
        ),
        "noop_differs_from_factual": (
            tribool(delta.get("noop_differs_from_factual"))
            if _field_present(delta, "noop_differs_from_factual")
            else NA
        ),
        "counterfactual_differs_from_noop": (
            tribool(delta.get("counterfactual_differs_from_noop"))
            if _field_present(delta, "counterfactual_differs_from_noop")
            else NA
        ),
    }


def _snapshots_conflict(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "factual_value",
        "noop_value",
        "counterfactual_value",
        "decision_relevant",
        "noop_differs_from_factual",
        "counterfactual_differs_from_noop",
    )
    for key in keys:
        a = left.get(key)
        b = right.get(key)
        if a is NA or b is NA:
            continue
        if _canonical_json(a) != _canonical_json(b):
            return True
    return False


def _merge_snapshots(copies: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], bool]:
    merged = dict(copies[0])
    conflict = False
    keys = (
        "factual_value",
        "noop_value",
        "counterfactual_value",
        "decision_relevant",
        "noop_differs_from_factual",
        "counterfactual_differs_from_noop",
    )
    for key in keys:
        present = [snap[key] for snap in copies if snap.get(key) is not NA]
        if not present:
            merged[key] = NA
            continue
        unique = {_canonical_json(v) for v in present}
        if len(unique) > 1:
            conflict = True
            merged[key] = NA
        else:
            merged[key] = present[0]
    return merged, conflict


def _merge_flags(values: Sequence[Any]) -> Tuple[Any, bool]:
    known = [v for v in values if v is not NA]
    if not known:
        return NA, False
    unique = set(known)
    if len(unique) > 1:
        return NA, True
    return known[0], False


def extract_outcome_criteria_changes(
    experiment: Mapping[str, Any],
    *,
    lecturebank_id: str = "",
) -> Dict[str, Any]:
    """One row per stored criterion-delta on unique AXIS replay comparisons."""
    run_id = str(experiment.get("run_id") or "")
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    seen_export: set[Tuple[str, str, str, str, str]] = set()

    for case in experiment.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        candidate_ref = _candidate_ref(case)
        raw = _axis_raw(case)
        primary = _primary_comparisons(raw)
        copies = _all_comparison_copies(raw)
        round_by_sig: Dict[str, int] = {}
        for index, cmp in enumerate(primary):
            sig = comparison_signature(cmp)
            if sig and sig not in round_by_sig:
                round_by_sig[sig] = index + 1
        extra_round = len(round_by_sig)
        copies_by_sig: Dict[str, List[Mapping[str, Any]]] = {}
        for cmp in copies:
            sig = comparison_signature(cmp)
            if not sig:
                if _has_criterion_deltas(cmp) and cmp.get("criterion_deltas"):
                    diagnostics.append(
                        {
                            "kind": "unresolved_signature",
                            "candidate_ref": candidate_ref,
                            "reason": "comparison lacked a complete intervention signature",
                        }
                    )
                continue
            copies_by_sig.setdefault(sig, []).append(cmp)
            if sig not in round_by_sig:
                extra_round += 1
                round_by_sig[sig] = extra_round

        for sig, group in copies_by_sig.items():
            with_deltas = [cmp for cmp in group if _has_criterion_deltas(cmp)]
            if not with_deltas:
                continue
            valid_vals = [_is_valid_comparison(cmp) for cmp in with_deltas]
            full_vals = [_full_control_stable(cmp) for cmp in with_deltas]
            valid, valid_conflict = _merge_flags(valid_vals)
            full_cs, full_conflict = _merge_flags(full_vals)
            if valid_conflict or full_conflict:
                diagnostics.append(
                    {
                        "kind": "conflicting_comparison_flags",
                        "candidate_ref": candidate_ref,
                        "intervention_signature": sig,
                        "valid_conflict": valid_conflict,
                        "full_control_stable_conflict": full_conflict,
                    }
                )
            first = with_deltas[0]
            sig_map = (
                first.get("intervention_signature")
                if isinstance(first.get("intervention_signature"), Mapping)
                else {}
            )
            op_raw = sig_map.get("operation") or first.get("operation")
            op_canon, _preserved = normalize_operation(op_raw)
            case_run = run_id
            factual = first.get("factual") if isinstance(first.get("factual"), Mapping) else {}
            if factual.get("run_id"):
                case_run = str(factual.get("run_id"))
            by_name: Dict[str, List[Dict[str, Any]]] = {}
            for cmp in with_deltas:
                names_here = []
                for delta in cmp.get("criterion_deltas") or []:
                    if not isinstance(delta, Mapping):
                        continue
                    snap = _delta_snapshot(delta)
                    name = snap["criterion_name"]
                    if not name:
                        diagnostics.append(
                            {
                                "kind": "unnamed_criterion",
                                "candidate_ref": candidate_ref,
                                "intervention_signature": sig,
                            }
                        )
                        continue
                    names_here.append(name)
                    by_name.setdefault(name, []).append(snap)
                if len(names_here) != len(set(names_here)):
                    diagnostics.append(
                        {
                            "kind": "duplicate_criterion_name",
                            "candidate_ref": candidate_ref,
                            "intervention_signature": sig,
                        }
                    )
            delta_counts = {
                id(cmp): len(list(cmp.get("criterion_deltas") or []))
                for cmp in with_deltas
            }
            if len(set(delta_counts.values())) > 1:
                diagnostics.append(
                    {
                        "kind": "conflicting_delta_count",
                        "candidate_ref": candidate_ref,
                        "intervention_signature": sig,
                        "counts": sorted(set(delta_counts.values())),
                    }
                )
            for name, snaps in by_name.items():
                conflict = False
                for left, right in zip(snaps, snaps[1:]):
                    if _snapshots_conflict(left, right):
                        conflict = True
                        break
                merged, merge_conflict = _merge_snapshots(snaps)
                value_conflict = conflict or merge_conflict
                if value_conflict:
                    diagnostics.append(
                        {
                            "kind": "conflicting_criterion_values",
                            "candidate_ref": candidate_ref,
                            "intervention_signature": sig,
                            "criterion_name": name,
                        }
                    )
                raw_diff = merged["counterfactual_differs_from_noop"]
                if value_conflict:
                    supported = NA
                elif valid is True and full_cs is True and raw_diff is not NA:
                    supported = raw_diff
                else:
                    supported = NA
                export_key = (str(lecturebank_id), case_run, candidate_ref, sig, name)
                if export_key in seen_export:
                    continue
                seen_export.add(export_key)
                rows.append(
                    {
                        "lecturebank_id": lecturebank_id,
                        "run_id": case_run,
                        "candidate_ref": candidate_ref,
                        "intervention_signature": sig,
                        "round": round_by_sig.get(sig),
                        "operation": op_canon,
                        "target_ref": artifact_ref_key(sig_map.get("target_ref") or first.get("target_ref")),
                        "replacement_ref": artifact_ref_key(
                            sig_map.get("replacement_ref") or first.get("replacement_ref")
                        ),
                        "checkpoint_id": sig_map.get("checkpoint_id") or first.get("checkpoint_id"),
                        "checkpoint_alias": sig_map.get("checkpoint_alias")
                        or first.get("checkpoint_alias"),
                        "criterion_name": name,
                        "factual_value": merged["factual_value"],
                        "noop_value": merged["noop_value"],
                        "counterfactual_value": merged["counterfactual_value"],
                        "decision_relevant": merged["decision_relevant"],
                        "noop_differs_from_factual": merged["noop_differs_from_factual"],
                        "counterfactual_differs_from_noop": merged[
                            "counterfactual_differs_from_noop"
                        ],
                        "valid_comparison": valid,
                        "full_control_stable": full_cs,
                        "criterion_change_supported": supported,
                    }
                )
    return {
        "rows": rows,
        "diagnostics": diagnostics,
        "by_criterion": aggregate_outcome_criteria(rows),
    }


def aggregate_outcome_criteria(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per-criterion supported-change rates. Denominator is evaluable rows only."""
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        name = str(row.get("criterion_name") or "")
        if not name:
            continue
        acc = counts.setdefault(name, {"supported_changes": 0, "evaluable": 0})
        flag = tribool(row.get("criterion_change_supported"))
        if flag is True:
            acc["evaluable"] += 1
            acc["supported_changes"] += 1
        elif flag is False:
            acc["evaluable"] += 1
    out = []
    for name in sorted(counts):
        acc = counts[name]
        out.append(
            {
                "criterion_name": name,
                "supported_changes": acc["supported_changes"],
                "evaluable_comparisons": acc["evaluable"],
                "supported_change_rate": ratio(acc["supported_changes"], acc["evaluable"]),
            }
        )
    return out


def _counterfactual_value_available(value: Any) -> bool:
    """True when a counterfactual value is present. False/0/[] are available."""
    if value is NA or value is None:
        return False
    if isinstance(value, str) and value.strip().upper() in {"", "NA"}:
        return False
    return True


def filter_outcome_criteria_changed(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Supported changes with an available counterfactual value.

    Uses tribool for flags so CSV ``false`` is not treated as True.
    Does not filter on ``decision_relevant``.
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        differs = tribool(row.get("counterfactual_differs_from_noop"))
        supported = tribool(row.get("criterion_change_supported"))
        if differs is not True or supported is not True:
            continue
        if not _counterfactual_value_available(row.get("counterfactual_value")):
            continue
        out.append(dict(row))
    return out


def hydrate_outcome_criteria_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Restore JSON list/object cells after CSV reload."""
    out = dict(row)
    for key in ("factual_value", "noop_value", "counterfactual_value"):
        val = out.get(key)
        if isinstance(val, str) and val[:1] in "[{":
            try:
                out[key] = json.loads(val)
            except json.JSONDecodeError:
                pass
    rnd = out.get("round")
    if isinstance(rnd, str) and rnd.isdigit():
        out["round"] = int(rnd)
    return out
