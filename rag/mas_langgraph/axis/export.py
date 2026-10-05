"""Serializable AXIS explanation reports (research artifacts)."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..artifacts.models import ArtifactRef
from .explainer import ExplanationResult


AXIS_EXPLANATION_SCHEMA_VERSION = 1


def _ref_dict(ref: Optional[ArtifactRef]) -> Optional[Dict[str, Any]]:
    if ref is None:
        return None
    return {
        "logical_id": ref.logical_id,
        "version": int(ref.version),
        "key": str(ref),
    }


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _observation_dict(obs: Any) -> Dict[str, Any]:
    if obs is None:
        return {}
    return {
        "run_id": getattr(obs, "run_id", None),
        "branch_id": getattr(obs, "branch_id", None),
        "subject_ref": _ref_dict(getattr(obs, "subject_ref", None)),
        "assessment_ref": _ref_dict(getattr(obs, "assessment_ref", None)),
        "decision_ref": _ref_dict(getattr(obs, "decision_ref", None)),
        "decision": getattr(obs, "decision", None),
        "subject_present": bool(getattr(obs, "subject_present", True)),
        "outcome_available": bool(getattr(obs, "outcome_available", False)),
        "error": getattr(obs, "error", None),
        "decision_reasons": list(getattr(obs, "decision_reasons", ()) or ()),
        "required_fixes": list(getattr(obs, "required_fixes", ()) or ()),
        "criterion_values": [
            {
                "name": getattr(cv, "name", None),
                "value": getattr(cv, "value", None),
                "decision_relevant": bool(getattr(cv, "decision_relevant", False)),
            }
            for cv in (getattr(obs, "criterion_values", ()) or ())
        ],
    }


def _criterion_delta_dict(d: Any) -> Dict[str, Any]:
    return {
        "name": getattr(d, "name", None),
        "factual": getattr(d, "factual_value", None),
        "noop": getattr(d, "noop_value", None),
        "counterfactual": getattr(d, "counterfactual_value", None),
        "decision_relevant": bool(getattr(d, "decision_relevant", False)),
        "noop_differs_from_factual": bool(
            getattr(d, "noop_differs_from_factual", False)
        ),
        "counterfactual_differs_from_noop": bool(
            getattr(d, "counterfactual_differs_from_noop", False)
        ),
    }


def _signature_dict(sig: Any) -> Optional[Dict[str, Any]]:
    if sig is None:
        return None
    return {
        "operation": getattr(sig, "operation", None),
        "target_ref": _ref_dict(getattr(sig, "target_ref", None)),
        "replacement_ref": _ref_dict(getattr(sig, "replacement_ref", None)),
        "checkpoint_id": getattr(sig, "checkpoint_id", None),
        "checkpoint_alias": getattr(sig, "checkpoint_alias", None),
        "boundary_raw": getattr(sig, "boundary_raw", None),
    }


def _resolved_dict(rq: Any) -> Optional[Dict[str, Any]]:
    if rq is None:
        return None
    return {
        "operation": getattr(rq, "operation", None),
        "raw_text": getattr(rq, "raw_text", None),
        "target_ref": _ref_dict(getattr(rq, "target_ref", None)),
        "replacement_ref": _ref_dict(getattr(rq, "replacement_ref", None)),
        "checkpoint_alias": getattr(rq, "checkpoint_alias", None),
        "checkpoint_id": getattr(rq, "checkpoint_id", None),
        "boundary_raw": getattr(rq, "boundary_raw", None),
        "target_artifact_type": getattr(rq, "target_artifact_type", None),
        "replacement_artifact_type": getattr(rq, "replacement_artifact_type", None),
    }


def _relation_summary(rel: Any) -> Dict[str, Any]:
    """Serialize a provenance relation or a normalized comparison tuple."""
    # Normalized tuple from normalize_provenance_relation:
    # (producer_label, artifact_ref_str, relation, consumer_label)
    if isinstance(rel, (tuple, list)) and len(rel) >= 4:
        return {
            "relation": rel[2],
            "artifact_ref": {"key": str(rel[1])} if rel[1] is not None else None,
            "producer_occurrence_id": rel[0],
            "consumer_occurrence_id": rel[3],
        }
    return {
        "relation": getattr(rel, "relation", None),
        "artifact_ref": _ref_dict(getattr(rel, "artifact_ref", None)),
        "producer_occurrence_id": getattr(rel, "producer_occurrence_id", None),
        "consumer_occurrence_id": getattr(rel, "consumer_occurrence_id", None),
    }


def _comparison_dict(cmp: Any) -> Dict[str, Any]:
    if cmp is None:
        return {}
    base = {
        "intervention_signature": _signature_dict(
            getattr(cmp, "intervention_signature", None)
        ),
        "evidence_status": getattr(cmp, "evidence_status", None),
        "factual": _observation_dict(getattr(cmp, "factual", None)),
        "noop": _observation_dict(getattr(cmp, "noop", None)),
        "counterfactual": _observation_dict(getattr(cmp, "counterfactual", None)),
        "criterion_deltas": [
            _criterion_delta_dict(d)
            for d in sorted(
                getattr(cmp, "criterion_deltas", ()) or (),
                key=lambda x: str(getattr(x, "name", "")),
            )
        ],
        "noop_path": list(getattr(cmp, "noop_path", ()) or ()),
        "counterfactual_path": list(getattr(cmp, "counterfactual_path", ()) or ()),
        "first_path_divergence": getattr(cmp, "first_path_divergence", None),
        "added_provenance_relations": [
            _relation_summary(r)
            for r in (getattr(cmp, "added_provenance_relations", ()) or ())
        ],
        "removed_provenance_relations": [
            _relation_summary(r)
            for r in (getattr(cmp, "removed_provenance_relations", ()) or ())
        ],
        "decision_control_stable": getattr(cmp, "decision_control_stable", None),
        "criteria_control_stable": getattr(cmp, "criteria_control_stable", None),
        "subject_content_control_stable": getattr(
            cmp, "subject_content_control_stable", None
        ),
        "intervention_before_subject_generation": getattr(
            cmp, "intervention_before_subject_generation", None
        ),
        "subject_generation_relation": getattr(
            cmp, "subject_generation_relation", None
        ),
        "question_content_comparison": _subject_content_dict(
            getattr(cmp, "subject_content_delta", None)
        ),
        "llm_replay_diagnostics": {
            "noop": dict(getattr(cmp, "noop_llm_replay_diagnostics", None) or {}),
            "counterfactual": dict(
                getattr(cmp, "counterfactual_llm_replay_diagnostics", None) or {}
            ),
        },
        "notes": list(getattr(cmp, "notes", ()) or ()),
    }
    # Researcher-facing information-role / semantic-validity metadata.
    try:
        from ..experiment.information_role import build_information_role_finding

        finding = build_information_role_finding(base)
        if finding is not None:
            base["information_role_finding"] = finding.diagnostics()
            base["semantic_intervention_validity"] = {
                "intervention_scope": finding.intervention_scope,
                "semantic_information_absent": finding.semantic_information_absent,
                "derived_information_may_persist": finding.derived_information_may_persist,
            }
    except Exception:
        pass
    return base


def _subject_content_dict(delta: Any) -> Optional[Dict[str, Any]]:
    if delta is None:
        return None
    if hasattr(delta, "to_dict"):
        return delta.to_dict()
    if isinstance(delta, dict):
        return dict(delta)
    return {
        "kind": getattr(delta, "kind", None),
        "control_stable": getattr(delta, "control_stable", None),
        "effect_status": getattr(delta, "effect_status", None),
        "details": dict(getattr(delta, "details", None) or {}),
    }


def _query_round_dict(rec: Any, kind: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "kind": kind,
        "generated_query": getattr(rec, "generated_query", None),
        "resolved": _resolved_dict(getattr(rec, "resolved", None)),
        "signature": _signature_dict(getattr(rec, "signature", None)),
        "error": getattr(rec, "error", None),
    }
    if kind == "factual":
        result = getattr(rec, "result", None)
        if result is not None:
            base["result"] = {
                "target_ref": _ref_dict(getattr(result, "target_ref", None)),
                "artifact_type": getattr(result, "artifact_type", None),
                "producer_occurrence_id": getattr(
                    result, "producer_occurrence_id", None
                ),
                "allowed_interventions": list(
                    getattr(result, "allowed_interventions", ()) or ()
                ),
            }
        else:
            base["result"] = None
    else:
        base["operation"] = getattr(getattr(rec, "resolved", None), "operation", None)
        base["target_ref"] = _ref_dict(
            getattr(getattr(rec, "resolved", None), "target_ref", None)
        )
        base["replacement_ref"] = _ref_dict(
            getattr(getattr(rec, "resolved", None), "replacement_ref", None)
        )
        base["checkpoint_alias"] = getattr(
            getattr(rec, "resolved", None), "checkpoint_alias", None
        )
        base["checkpoint_id"] = getattr(
            getattr(rec, "resolved", None), "checkpoint_id", None
        )
        base["noop_branch_id"] = getattr(rec, "noop_branch_id", None)
        base["counterfactual_branch_id"] = getattr(
            rec, "counterfactual_branch_id", None
        )
        base["noop_reused"] = bool(getattr(rec, "noop_reused", False))
        base["noop_cache_key"] = getattr(rec, "noop_cache_key", None)
        comparison = getattr(rec, "replay_comparison", None) or getattr(
            rec, "comparison", None
        )
        if hasattr(comparison, "evidence_status"):
            base["replay_comparison"] = _comparison_dict(comparison)
            base["evidence_status"] = getattr(comparison, "evidence_status", None)
        elif isinstance(comparison, Mapping):
            base["replay_comparison"] = dict(comparison)
            base["evidence_status"] = comparison.get("evidence_status")
        else:
            base["replay_comparison"] = None
            base["evidence_status"] = None
    return base


@dataclass(frozen=True)
class AxisExplanationReport:
    """Deterministic, serializable AXIS explanation research artifact."""

    schema_version: int
    case_id: str
    case_version: int
    run_id: str
    branch_id: str
    artifact_ref: ArtifactRef
    assessment_ref: Optional[ArtifactRef]
    decision_ref: ArtifactRef
    factual_decision: str
    question: str
    termination_reason: str
    rounds: int
    factual_artifacts: Tuple[str, ...]
    factual_occurrences: Tuple[str, ...]
    factual_relations: Tuple[Any, ...]
    query_rounds: Tuple[Any, ...]
    replay_comparisons: Tuple[Any, ...]
    final_explanation: str
    warnings: Tuple[str, ...]
    attempt_count: int = 0
    successful_round_count: int = 0
    valid_intervention_candidate_count: int = 0
    interrogation_attempts: Tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _case_meta(case_definition: Any) -> Tuple[str, int]:
    case_id = str(getattr(case_definition, "case_id", "") or "unknown")
    case_version = int(getattr(case_definition, "case_version", 1) or 1)
    if case_definition is not None and hasattr(case_definition, "load_case_configuration"):
        try:
            cfg = case_definition.load_case_configuration(require_graph=False)
            meta = getattr(cfg, "metadata", None)
            if meta is not None:
                case_id = str(getattr(meta, "case_id", case_id) or case_id)
                case_version = int(getattr(meta, "case_version", case_version) or case_version)
        except Exception:
            pass
    return case_id, case_version


def _framework_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import langgraph

        meta["langgraph_version"] = getattr(langgraph, "__version__", "unknown")
    except Exception:
        meta["langgraph_version"] = None
    return meta


def explanation_result_to_report(
    result: ExplanationResult,
    *,
    question: str = "",
    case_definition: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AxisExplanationReport:
    """Convert an in-memory ExplanationResult into a serializable report."""
    explained = result.explained_case
    case_slice = result.case_slice
    case_id, case_version = _case_meta(case_definition)

    artifacts = tuple(
        sorted(str(r) for r in (getattr(case_slice, "relevant_artifact_refs", ()) or ()))
    )
    occurrences = tuple(
        sorted(
            getattr(o, "labeled_id", None) or getattr(o, "occurrence_id", "") or str(o)
            for o in (getattr(case_slice, "relevant_occurrences", ()) or ())
        )
    )
    relations = tuple(
        _relation_summary(r)
        for r in sorted(
            getattr(case_slice, "relevant_relations", ()) or (),
            key=lambda x: (
                str(getattr(x, "relation", "")),
                str(getattr(x, "artifact_ref", "")),
                str(getattr(x, "consumer_occurrence_id", "")),
            ),
        )
    )

    query_rounds: List[Dict[str, Any]] = []
    for rec in result.factual_records or []:
        query_rounds.append(_query_round_dict(rec, "factual"))
    for rec in result.intervention_records or []:
        query_rounds.append(_query_round_dict(rec, "intervention"))

    attempts = tuple(
        a.to_dict() if hasattr(a, "to_dict") else dict(a)
        for a in (getattr(result, "interrogation_attempts", None) or [])
    )

    comparisons = tuple(
        _comparison_dict(c) for c in (result.replay_comparisons or [])
    )

    meta = dict(_framework_metadata())
    if metadata:
        meta.update(dict(metadata))

    return AxisExplanationReport(
        schema_version=AXIS_EXPLANATION_SCHEMA_VERSION,
        case_id=case_id,
        case_version=case_version,
        run_id=explained.run_id,
        branch_id=explained.branch_id,
        artifact_ref=explained.artifact_ref,
        assessment_ref=explained.assessment_ref,
        decision_ref=explained.decision_ref,
        factual_decision=str(explained.outcome or ""),
        question=question or "",
        termination_reason=str(result.termination_reason or ""),
        rounds=int(result.rounds or 0),
        attempt_count=int(getattr(result, "attempt_count", 0) or 0),
        successful_round_count=int(
            getattr(result, "successful_round_count", result.rounds) or 0
        ),
        valid_intervention_candidate_count=int(
            getattr(result, "valid_intervention_candidate_count", 0) or 0
        ),
        factual_artifacts=artifacts,
        factual_occurrences=occurrences,
        factual_relations=relations,
        query_rounds=tuple(query_rounds),
        interrogation_attempts=attempts,
        replay_comparisons=comparisons,
        final_explanation=str(result.final_explanation or ""),
        warnings=tuple(result.warnings or ()),
        metadata=meta,
    )


def axis_report_to_dict(report: AxisExplanationReport) -> Dict[str, Any]:
    """Deterministic dictionary representation for JSON export."""
    return {
        "schema_version": int(report.schema_version),
        "case_id": report.case_id,
        "case_version": int(report.case_version),
        "run_id": report.run_id,
        "branch_id": report.branch_id,
        "artifact_ref": _ref_dict(report.artifact_ref),
        "assessment_ref": _ref_dict(report.assessment_ref),
        "decision_ref": _ref_dict(report.decision_ref),
        "factual_decision": report.factual_decision,
        "question": report.question,
        "termination_reason": report.termination_reason,
        "rounds": int(report.rounds),
        "attempt_count": int(getattr(report, "attempt_count", 0) or 0),
        "successful_round_count": int(
            getattr(report, "successful_round_count", report.rounds) or 0
        ),
        "valid_intervention_candidate_count": int(
            getattr(report, "valid_intervention_candidate_count", 0) or 0
        ),
        "factual_artifacts": list(report.factual_artifacts),
        "factual_occurrences": list(report.factual_occurrences),
        "factual_relations": list(report.factual_relations),
        "query_rounds": list(report.query_rounds),
        "interrogation_attempts": list(
            getattr(report, "interrogation_attempts", ()) or ()
        ),
        "replay_comparisons": list(report.replay_comparisons),
        "final_explanation": report.final_explanation,
        "warnings": list(report.warnings),
        "metadata": dict(report.metadata or {}),
    }


def dump_axis_explanation(
    report_or_result: Union[AxisExplanationReport, ExplanationResult],
    path: str,
    *,
    question: str = "",
    case_definition: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Write an AXIS explanation report as UTF-8 pretty JSON."""
    if isinstance(report_or_result, ExplanationResult):
        report = explanation_result_to_report(
            report_or_result,
            question=question,
            case_definition=case_definition,
            metadata=metadata,
        )
    else:
        report = report_or_result
    payload = axis_report_to_dict(report)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(out.resolve())
