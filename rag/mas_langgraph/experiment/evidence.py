"""Strict evidence projections for Trace / Trace+Provenance / AXIS."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..artifacts.registry import ArtifactRegistry
from ..axis.models import ExplainedCase
from ..axis.slicing import CaseSlice


# Keys that must never appear in Trace-only evidence.
_TRACE_FORBIDDEN_KEYS = frozenset(
    {
        "provenance_relations",
        "run_provenance_relations",
        "produced_by",
        "artifact_bindings",
        "relevant_relations",
        "replay_comparisons",
        "counterfactual",
        "noop_branch",
        "intervention",
    }
)

# Trace uses the full factual execution occurrence log — not CaseSlice
# (which is provenance-informed). Relation payloads are never included.
TRACE_OCCURRENCE_SELECTION_USES_CASE_SLICE = False


def occurrence_label(occ: Any) -> str:
    labeled = getattr(occ, "labeled_id", None)
    if labeled:
        return str(labeled)
    node_id = getattr(occ, "node_id", "?")
    idx = getattr(occ, "occurrence_index", None)
    if idx is not None:
        return f"{node_id}#{idx}"
    return str(node_id)


def _factual_trace_occurrences(
    *,
    explained_case: ExplainedCase,
    factual_occurrences: Optional[Any],
) -> list:
    """Provenance-independent factual execution order for Trace baseline."""
    run_id = getattr(explained_case, "run_id", None)
    branch_id = getattr(explained_case, "branch_id", "factual") or "factual"
    occs = list(factual_occurrences or ())
    filtered = []
    for o in occs:
        if getattr(o, "branch_id", None) and getattr(o, "branch_id") != branch_id:
            continue
        if run_id and getattr(o, "run_id", None) and getattr(o, "run_id") != run_id:
            continue
        filtered.append(o)
    filtered.sort(key=lambda o: int(getattr(o, "execution_index", 0) or 0))
    return filtered


def build_trace_evidence(
    *,
    explained_case: ExplainedCase,
    case_slice: Optional[CaseSlice] = None,
    factual_occurrences: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execution-order projection only — no provenance relations or replays.

    Occurrence selection uses the factual branch occurrence log only. ``case_slice``
    is accepted for API compatibility but must not drive Trace membership.
    """
    del case_slice  # provenance slice must not influence Trace selection
    occs = _factual_trace_occurrences(
        explained_case=explained_case,
        factual_occurrences=factual_occurrences,
    )
    labels = [occurrence_label(o) for o in occs]
    node_ids = [
        str(getattr(o, "node_id", "") or "")
        for o in occs
        if getattr(o, "node_id", None)
    ]
    evidence = {
        "kind": "trace",
        "run_id": explained_case.run_id,
        "branch_id": explained_case.branch_id,
        "candidate_ref": str(explained_case.artifact_ref),
        "decision_ref": str(explained_case.decision_ref),
        "assessment_ref": str(explained_case.assessment_ref)
        if explained_case.assessment_ref
        else None,
        "factual_decision": explained_case.outcome,
        "execution_trace": labels,
        "executed_nodes": list(dict.fromkeys(node_ids)),
        "provenance_included": False,
        "counterfactual_replay_included": False,
        "occurrence_selection": "factual_branch_occurrence_log",
        "occurrence_selection_uses_provenance_slice": False,
    }
    for key in _TRACE_FORBIDDEN_KEYS:
        assert key not in evidence or evidence.get(key) in (None, False, (), [])
    return evidence


def _preview_text(text: str, *, max_chars: int = 220) -> str:
    """Short participant-safe excerpt; collapse whitespace."""
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[: max_chars - 1].rsplit(" ", 1)[0]
    return (cut or cleaned[: max_chars - 1]).rstrip(".,;:") + "…"


def build_trace_provenance_evidence(
    *,
    explained_case: ExplainedCase,
    case_slice: CaseSlice,
    registry: Optional[ArtifactRegistry] = None,
) -> Dict[str, Any]:
    """Factual provenance slice + artifacts + assessment — no replays."""
    occs = tuple(getattr(case_slice, "relevant_occurrences", ()) or ())
    relations = tuple(getattr(case_slice, "relevant_relations", ()) or ())
    refs = tuple(getattr(case_slice, "relevant_artifact_refs", ()) or ())

    rel_rows: List[Dict[str, Any]] = []
    for rel in relations:
        rel_rows.append(
            {
                "relation": getattr(rel, "relation", None),
                "artifact_ref": str(getattr(rel, "artifact_ref", "") or ""),
                "producer_occurrence_id": getattr(rel, "producer_occurrence_id", None),
                "consumer_occurrence_id": getattr(rel, "consumer_occurrence_id", None),
            }
        )

    artifact_summaries: List[Dict[str, Any]] = []
    for ref in refs:
        row: Dict[str, Any] = {"ref": str(ref)}
        if registry is not None:
            env = registry.get_by_ref(ref, explained_case.branch_id) or registry.get_by_ref(
                ref
            )
            if env is not None:
                row["artifact_type"] = getattr(env, "artifact_type", None)
                payload = getattr(env, "payload", None)
                for attr in (
                    "section_id",
                    "lo_id",
                    "blueprint_id",
                    "title",
                    "objective_text",
                    "stem",
                    "decision",
                    "supported_by_pdf",
                    "passed",
                ):
                    if hasattr(payload, attr):
                        row[attr] = getattr(payload, attr)
                # Source text for Trace+Provenance / AXIS verbalization only.
                if hasattr(payload, "content"):
                    content = getattr(payload, "content", None)
                    if isinstance(content, str) and content.strip():
                        row["content"] = content
                        row["content_preview"] = _preview_text(content)
        artifact_summaries.append(row)

    criteria = []
    for cv in getattr(case_slice, "criterion_values", ()) or ():
        criteria.append(
            {
                "name": getattr(cv, "name", None),
                "value": getattr(cv, "value", None),
                "decision_relevant": bool(getattr(cv, "decision_relevant", False)),
            }
        )

    return {
        "kind": "trace_provenance",
        "run_id": explained_case.run_id,
        "branch_id": explained_case.branch_id,
        "candidate_ref": str(explained_case.artifact_ref),
        "decision_ref": str(explained_case.decision_ref),
        "assessment_ref": str(explained_case.assessment_ref)
        if explained_case.assessment_ref
        else None,
        "factual_decision": explained_case.outcome
        or getattr(case_slice, "decision", "")
        or "",
        "execution_trace": [occurrence_label(o) for o in occs],
        "provenance_relations": rel_rows,
        "relevant_artifacts": artifact_summaries,
        "decision_reasons": list(getattr(case_slice, "decision_reasons", ()) or ()),
        "required_fixes": list(getattr(case_slice, "required_fixes", ()) or ()),
        "criteria": criteria,
        "provenance_included": True,
        "counterfactual_replay_included": False,
        "replay_comparisons": [],
    }


def _comparison_research_dict(cmp: Any) -> Dict[str, Any]:
    """Serialize one ReplayComparison for research JSON + verbalizer input."""
    from ..axis.export import _comparison_dict  # reuse existing serializer

    try:
        return _comparison_dict(cmp)
    except Exception:
        return {
            "evidence_status": getattr(cmp, "evidence_status", None),
            "notes": list(getattr(cmp, "notes", ()) or ()),
        }


def build_axis_evidence_summary(
    *,
    explanation_result: Any,
    case_slice: Any = None,
    explained_case: Any = None,
    registry: Optional[ArtifactRegistry] = None,
) -> Dict[str, Any]:
    """Researcher-facing AXIS evidence (includes replays + optional factual slice).

    ``final_explanation_raw`` is retained for researcher/debug export only.
    Experiment synthesis must not feed it to the shared final synthesizer.
    """
    comparisons = list(getattr(explanation_result, "replay_comparisons", None) or [])
    interventions = list(getattr(explanation_result, "intervention_records", None) or [])
    successful = [
        c
        for c in comparisons
        if str(getattr(c, "evidence_status", "") or "")
        not in {"noop_error", "intervention_error", "outcome_unavailable", "replay_unstable"}
        and getattr(c, "evidence_status", None) is not None
    ]
    statuses = [str(getattr(c, "evidence_status", "") or "") for c in comparisons]
    rounds = list(getattr(explanation_result, "round_records", None) or [])

    # Reuse factual provenance projection so AXIS verbalization has process+deps.
    factual_part: Dict[str, Any] = {}
    if explained_case is not None and case_slice is not None:
        factual_part = build_trace_provenance_evidence(
            explained_case=explained_case,
            case_slice=case_slice,
            registry=registry,
        )

    payload = {
        "kind": "axis",
        "run_id": getattr(explained_case, "run_id", None),
        "branch_id": getattr(explained_case, "branch_id", "factual"),
        "candidate_ref": str(getattr(explained_case, "artifact_ref", "") or ""),
        "factual_decision": getattr(explained_case, "outcome", None)
        or factual_part.get("factual_decision"),
        "execution_trace": factual_part.get("execution_trace", []),
        "provenance_relations": factual_part.get("provenance_relations", []),
        "relevant_artifacts": factual_part.get("relevant_artifacts", []),
        "decision_reasons": factual_part.get("decision_reasons", []),
        "required_fixes": factual_part.get("required_fixes", []),
        "criteria": factual_part.get("criteria", []),
        "termination_reason": getattr(explanation_result, "termination_reason", ""),
        "rounds": int(getattr(explanation_result, "rounds", 0) or 0),
        "successful_round_count": int(
            getattr(
                explanation_result,
                "successful_round_count",
                getattr(explanation_result, "rounds", 0),
            )
            or 0
        ),
        "attempt_count": int(getattr(explanation_result, "attempt_count", 0) or 0),
        "valid_intervention_candidate_count": int(
            getattr(explanation_result, "valid_intervention_candidate_count", 0) or 0
        ),
        "axis_interrogation_mode": str(
            getattr(explanation_result, "axis_interrogation_mode", "") or "constrained"
        ),
        "axis_interrogation_policy": str(
            getattr(explanation_result, "axis_interrogation_policy", "")
            or "counterfactual_only"
        ),
        "interrogation_attempts": [
            a.to_dict() if hasattr(a, "to_dict") else dict(a)
            for a in (getattr(explanation_result, "interrogation_attempts", None) or [])
        ],
        "duplicate_attempt_count": sum(
            1
            for a in (getattr(explanation_result, "interrogation_attempts", None) or [])
            if str(getattr(a, "validation_status", "") or "") == "duplicate_error"
            or getattr(a, "duplicate_error", None)
        ),
        "interrogation_budget_exhausted": (
            str(getattr(explanation_result, "termination_reason", "") or "")
            == "attempt_budget_exhausted"
        ),
        "query_round_count": len(rounds),
        "counterfactual_count": len(interventions),
        "counterfactual_executions": len(interventions),
        "successful_replay_count": len(successful),
        "valid_replay_comparisons": len(successful),
        "replay_comparison_count": len(comparisons),
        "replay_comparisons": [_comparison_research_dict(c) for c in comparisons],
        "evidence_statuses": statuses,
        "axis_causal_evidence_available": len(successful) > 0
        and any(
            s
            in {
                "decision_effect",
                "criterion_effect",
                "downstream_effect",
                "no_demonstrated_effect",
            }
            for s in statuses
        ),
        "provenance_included": True,
        "counterfactual_replay_included": True,
        "warnings": list(getattr(explanation_result, "warnings", ()) or ()),
        # Researcher/debug only — excluded from shared experimental synthesis input.
        "final_explanation_raw": getattr(explanation_result, "final_explanation", "")
        or "",
        "noop_cache_stats": dict(
            getattr(explanation_result, "noop_cache_stats", None) or {}
        ),
    }
    diagnostics = getattr(explanation_result, "hybrid_add_diagnostics", None)
    if diagnostics:
        payload["hybrid_add_diagnostics"] = diagnostics
    return payload


def synthesis_evidence_without_axis_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Drop AXIS-only LLM prose before building the experiment evidence view."""
    out = dict(raw)
    out.pop("final_explanation_raw", None)
    out.pop("axis_report", None)
    out.pop("hybrid_add_diagnostics", None)
    out.pop("hybrid_add_source_artifacts", None)
    return out


def assert_no_leakage(trace_evidence: Dict[str, Any], *, allow_provenance: bool) -> None:
    """Raise if baseline evidence contains forbidden classes."""
    if trace_evidence.get("counterfactual_replay_included"):
        raise AssertionError("Baseline evidence must not include counterfactual replay")
    if "replay_comparisons" in trace_evidence and trace_evidence["replay_comparisons"]:
        raise AssertionError("Baseline evidence leaked replay_comparisons")
    if not allow_provenance:
        if trace_evidence.get("provenance_included"):
            raise AssertionError("Trace-only evidence must not include provenance")
        if trace_evidence.get("provenance_relations"):
            raise AssertionError("Trace-only evidence leaked provenance_relations")
