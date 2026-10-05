"""Exhaustive replay-reference sweep over the initial hybrid intervention vocabulary.

This module is evaluation-only. It does not change AXIS interrogation, query
selection, prompts, or replay semantics. The reference set is a replay-derived
enumeration of the finite hybrid counterfactual space, not external causal
ground truth.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from mas_langgraph.axis.comparison import NoOpReplayCache, ReplayComparator
from mas_langgraph.axis.executor import AxisQueryExecutor
from mas_langgraph.axis.interrogation_mode import is_hybrid_interrogation_mode
from mas_langgraph.axis.intervention_guidance import (
    HybridInterventionVocabulary,
    build_hybrid_intervention_vocabulary,
    build_intervention_candidates,
    hybrid_opportunity_counts,
    validate_resolved_against_hybrid_vocabulary,
)
from mas_langgraph.axis.models import AxisQueryContext, QuerySignature
from mas_langgraph.axis.queries import AxisQueryParser, AxisQueryResolver, QueryValidationError
from mas_langgraph.batch.io import atomic_write_csv
from mas_langgraph.evaluation.axis_metrics import (
    NA,
    _collect_comparisons,
    artifact_ref_key,
    ratio,
)

CF_OPERATIONS = {"REMOVE", "ADD", "WHATIF", "REPLACE"}
EVALUABLE_REFERENCE_STATUSES = frozenset({"valid_effect", "valid_no_effect"})
TERMINAL_RESUME_STATUSES = {
    "valid_effect",
    "valid_no_effect",
    "inconclusive_noop_unstable",
}

REFERENCE_INTERVENTION_FIELDS = [
    "lecturebank_id",
    "candidate_ref",
    "reference_intervention_signature",
    "operation",
    "target_ref",
    "replacement_ref",
    "checkpoint_id",
    "checkpoint_alias",
    "boundary_raw",
    "generated_query",
    "validation_status",
    "execution_status",
    "reference_status",
    "reference_evaluable",
    "noop_decision_stable",
    "noop_criteria_stable",
    "noop_content_stable",
    "full_noop_stable",
    "valid_comparison",
    "supported_effect",
    "decision_effect",
    "criterion_effect",
    "decision_relevant_criterion_effect",
    "content_effect",
    "stem_effect",
    "correct_answer_effect",
    "distractor_effect",
    "answer_explanation_effect",
    "question_not_reproduced",
    "evidence_status",
    "causal_classification",
    "information_role",
    "replay_completion_mode",
    "executed_node_count",
    "noop_live_provider_calls",
    "counterfactual_live_provider_calls",
    "selected_by_axis",
    "selected_within_k",
    "axis_successful_round",
    "axis_selected_signature_missing_from_reference",
    "axis_interrogation_mode",
]

EFFECT_DISCOVERY_QUESTION_FIELDS = [
    "lecturebank_id",
    "candidate_ref",
    "k",
    "reference_universe_count",
    "reference_remove_count",
    "reference_whatif_count",
    "reference_add_count",
    "reference_valid_comparison_count",
    "reference_inconclusive_count",
    "reference_replay_failed_count",
    "reference_effectful_count",
    "reference_no_effect_count",
    "reference_evaluable_count",
    "reference_evaluability_rate",
    "axis_counterfactual_selected_at_k",
    "axis_valid_counterfactual_comparisons_at_k",
    "axis_reference_evaluable_counterfactuals_at_k",
    "axis_reference_unevaluable_counterfactuals_at_k",
    "axis_selected_effectful_at_k",
    "axis_effectful_discoveries_at_k",
    "axis_valid_effectful_discoveries_at_k",
    "effect_discovery_recall_at_k",
    "effect_precision_at_k",
    "effect_discovery_end_to_end_yield_at_k",
    "causal_evidence_available_at_k",
    "reference_set_complete",
    "reference_opportunity_count_match",
    "initial_available_remove_count",
    "initial_available_add_count",
    "initial_available_whatif_count",
    "opportunity_mismatch_detail",
    "axis_selected_signature_missing_from_reference",
    "axis_interrogation_mode",
]


class HybridReferenceSweepError(ValueError):
    """Raised when the replay-reference sweep cannot run for the requested mode."""


@dataclass(frozen=True)
class ReferenceInterventionSpec:
    """One unique candidate from the initial hybrid counterfactual vocabulary."""

    operation: str
    target_ref: Any
    replacement_ref: Any = None
    boundary_raw: str = ""
    generated_query: str = ""


def canonical_operation(operation: Any) -> str:
    token = str(operation or "").strip().upper().replace("-", "_").replace(" ", "")
    if token in {"WHATIF", "WHAT_IF", "REPLACE"}:
        return "WHATIF"
    return token


def canonical_intervention_signature(
    *,
    operation: Any,
    target_ref: Any = None,
    replacement_ref: Any = None,
    checkpoint_id: Any = None,
) -> str:
    """Join key: operation|target_ref|replacement_ref|checkpoint_id.

    WHATIF and REPLACE share the WHATIF operation token. Checkpoint identity
    uses QuerySignature.checkpoint_id (UUID, falling back to alias).
    """
    op = canonical_operation(operation)
    target = artifact_ref_key(target_ref)
    replacement = artifact_ref_key(replacement_ref) if replacement_ref else ""
    checkpoint = str(checkpoint_id or "")
    return f"{op}|{target}|{replacement}|{checkpoint}"


def canonical_signature_from_query_signature(sig: Any) -> str:
    return canonical_intervention_signature(
        operation=getattr(sig, "operation", None),
        target_ref=getattr(sig, "target_ref", None),
        replacement_ref=getattr(sig, "replacement_ref", None),
        checkpoint_id=getattr(sig, "checkpoint_id", None),
    )


def canonical_signature_from_mapping(payload: Mapping[str, Any]) -> str:
    sig = payload.get("intervention_signature")
    if isinstance(sig, Mapping):
        return canonical_intervention_signature(
            operation=sig.get("operation"),
            target_ref=sig.get("target_ref"),
            replacement_ref=sig.get("replacement_ref"),
            checkpoint_id=sig.get("checkpoint_id"),
        )
    return canonical_intervention_signature(
        operation=payload.get("operation") or payload.get("operation_raw"),
        target_ref=payload.get("target_ref"),
        replacement_ref=payload.get("replacement_ref"),
        checkpoint_id=payload.get("checkpoint_id"),
    )


def format_hybrid_intervention_query(
    operation: str,
    target_ref: Any,
    boundary_raw: str,
    replacement_ref: Any = None,
) -> str:
    op = canonical_operation(operation)
    target = artifact_ref_key(target_ref) or str(target_ref)
    boundary = str(boundary_raw or "")
    if op == "REMOVE":
        return f"REMOVE({target}, {boundary})"
    if op == "ADD":
        return f"ADD({target}, {boundary})"
    if op == "WHATIF":
        alt = artifact_ref_key(replacement_ref) or str(replacement_ref)
        return f"WHATIF({target}, {alt}, {boundary})"
    raise ValueError(f"Not a counterfactual hybrid operation: {operation!r}")


def iter_hybrid_counterfactual_specs(
    vocabulary: HybridInterventionVocabulary,
) -> List[ReferenceInterventionSpec]:
    """Enumerate REMOVE / WHATIF / ADD from the initial hybrid vocabulary.

    WHAT and DONE are excluded. ADD uses ``add_boundaries_for(ref)``, never the
    compatibility-union ``add_boundaries`` Cartesian product.
    """
    specs: List[ReferenceInterventionSpec] = []
    for cand in vocabulary.intervention_candidates:
        if "REMOVE" in cand.allowed_operations:
            for boundary in cand.boundaries_for("REMOVE"):
                specs.append(
                    ReferenceInterventionSpec(
                        operation="REMOVE",
                        target_ref=cand.ref,
                        boundary_raw=str(boundary),
                        generated_query=format_hybrid_intervention_query(
                            "REMOVE", cand.ref, str(boundary)
                        ),
                    )
                )
        if "REPLACE" in cand.allowed_operations:
            for alt in vocabulary.replacements_for(cand.ref):
                for boundary in cand.boundaries_for("REPLACE"):
                    specs.append(
                        ReferenceInterventionSpec(
                            operation="WHATIF",
                            target_ref=cand.ref,
                            replacement_ref=alt,
                            boundary_raw=str(boundary),
                            generated_query=format_hybrid_intervention_query(
                                "WHATIF", cand.ref, str(boundary), alt
                            ),
                        )
                    )
    for ref in vocabulary.add_options:
        for boundary in vocabulary.add_boundaries_for(ref):
            specs.append(
                ReferenceInterventionSpec(
                    operation="ADD",
                    target_ref=ref,
                    boundary_raw=str(boundary),
                    generated_query=format_hybrid_intervention_query(
                        "ADD", ref, str(boundary)
                    ),
                )
            )
    return specs


def _hybrid_context_aliases(
    vocabulary: HybridInterventionVocabulary,
    intervention_alternatives: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    extra: Dict[str, Any] = dict(intervention_alternatives or {})
    for _old, news in (vocabulary.replacement_options or {}).items():
        for new_ref in news:
            extra.setdefault(str(new_ref), new_ref)
    for add_ref in vocabulary.add_options:
        extra.setdefault(str(add_ref), add_ref)
    return extra


def parse_resolve_validate_hybrid(
    query_text: str,
    *,
    context: AxisQueryContext,
    vocabulary: HybridInterventionVocabulary,
    parser: Optional[AxisQueryParser] = None,
    resolver: Optional[AxisQueryResolver] = None,
) -> Any:
    """Same parse → resolve(reserve=False) → hybrid-vocabulary path as AXIS."""
    parser = parser or AxisQueryParser()
    resolver = resolver or AxisQueryResolver()
    parsed = parser.parse(query_text)
    if parsed.operation in {"WHAT", "DONE"}:
        raise QueryValidationError(
            f"{parsed.operation} is excluded from the replay-reference intervention universe"
        )
    resolved = resolver.resolve(
        parsed,
        context,
        executed_signatures=None,
        reserve=False,
    )
    validate_resolved_against_hybrid_vocabulary(resolved, vocabulary)
    return resolved


def enumerate_initial_hybrid_universe(
    *,
    vocabulary: HybridInterventionVocabulary,
    context: AxisQueryContext,
    parser: Optional[AxisQueryParser] = None,
    resolver: Optional[AxisQueryResolver] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Unique validated counterfactual interventions from the INITIAL vocabulary."""
    parser = parser or AxisQueryParser()
    resolver = resolver or AxisQueryResolver()
    seen: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    for spec in iter_hybrid_counterfactual_specs(vocabulary):
        try:
            resolved = parse_resolve_validate_hybrid(
                spec.generated_query,
                context=context,
                vocabulary=vocabulary,
                parser=parser,
                resolver=resolver,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"excluded invalid hybrid candidate {spec.generated_query!r}: {exc}"
            )
            continue
        sig = QuerySignature.from_resolved(resolved)
        key = canonical_signature_from_query_signature(sig)
        if key in seen:
            continue
        op = canonical_operation(resolved.operation)
        seen[key] = {
            "reference_intervention_signature": key,
            "operation": op,
            "target_ref": artifact_ref_key(resolved.target_ref),
            "replacement_ref": artifact_ref_key(resolved.replacement_ref)
            if resolved.replacement_ref
            else "",
            "checkpoint_id": getattr(resolved, "checkpoint_id", None)
            or getattr(resolved, "checkpoint_alias", None),
            "checkpoint_alias": getattr(resolved, "checkpoint_alias", None),
            "boundary_raw": getattr(resolved, "boundary_raw", None) or spec.boundary_raw,
            "generated_query": spec.generated_query,
            "validation_status": "valid",
            "resolved": resolved,
            "signature": sig,
        }
    rows = [seen[k] for k in sorted(seen)]
    return rows, warnings


def build_initial_hybrid_vocabulary(
    *,
    case_slice: Any,
    registry: Any,
    contracts: Any,
    checkpoint_index: Any,
    explained_case: Any,
    intervention_alternatives: Optional[Mapping[str, Any]] = None,
    intervention_registry: Any = None,
    all_occurrences: Sequence[Any] = (),
    replay_executor: Any = None,
    run_id: str = "",
    priority_fn: Any = None,
) -> Tuple[HybridInterventionVocabulary, Tuple[Any, ...]]:
    """Rebuild the pre-interrogation hybrid vocabulary (empty query history)."""
    candidates = build_intervention_candidates(
        case_slice=case_slice,
        registry=registry,
        contracts=contracts,
        checkpoint_index=checkpoint_index,
        branch_id=getattr(explained_case, "branch_id", "factual") or "factual",
        all_occurrences=all_occurrences,
        priority_fn=priority_fn,
        intervention_alternatives=intervention_alternatives,
    )
    vocab = build_hybrid_intervention_vocabulary(
        case_slice=case_slice,
        registry=registry,
        contracts=contracts,
        checkpoint_index=checkpoint_index,
        branch_id=getattr(explained_case, "branch_id", "factual") or "factual",
        explained_case=explained_case,
        intervention_candidates=candidates,
        intervention_alternatives=intervention_alternatives,
        intervention_registry=intervention_registry,
        all_occurrences=all_occurrences,
        replay_executor=replay_executor,
        run_id=run_id or str(getattr(explained_case, "run_id", "") or ""),
    )
    return vocab, tuple(candidates)


def supported_effect_from_comparison_row(row: Mapping[str, Any]) -> Optional[bool]:
    """Same any-supported-effect rule as effect_yield (full-control-stable valid)."""
    if row.get("valid_comparison") is not True:
        return None
    if row.get("full_control_stable") is not True:
        return None
    if (
        row.get("content_effect") is True
        or row.get("criterion_effect_any_supported") is True
        or row.get("decision_effect") is True
    ):
        return True
    if (
        row.get("content_effect") is False
        or row.get("criterion_effect_any_supported") is False
        or row.get("decision_effect") is False
    ):
        return False
    return False


def classify_reference_status(
    *,
    validation_status: Any = "valid",
    execution_status: Any = None,
    comparison_row: Optional[Mapping[str, Any]] = None,
    execution_error: Any = None,
) -> str:
    """Map a reference execution onto replay-reference statuses.

    NOOP instability and replay failure are not demonstrated no-effect.
    """
    if str(validation_status or "") != "valid":
        return "invalid"
    if execution_error or str(execution_status or "") in {"failed", "error"}:
        return "replay_failed"
    if not comparison_row:
        if str(execution_status or "") == "executed":
            return "replay_failed"
        return "replay_failed"
    evidence = str(comparison_row.get("evidence_status") or "")
    if evidence in {"noop_error", "intervention_error", "outcome_unavailable"}:
        return "replay_failed"
    if evidence == "replay_unstable" or comparison_row.get("full_control_stable") is False:
        return "inconclusive_noop_unstable"
    if comparison_row.get("valid_comparison") is not True:
        return "replay_failed"
    if comparison_row.get("full_control_stable") is not True:
        return "inconclusive_noop_unstable"
    if supported_effect_from_comparison_row(comparison_row) is True:
        return "valid_effect"
    return "valid_no_effect"


def reference_status_is_evaluable(status: Any) -> Optional[bool]:
    """True iff the reference sweep produced a valid, NOOP-stable effect/no-effect label.

    Missing/unknown status is None, not False, and is not treated as no-effect.
    """
    token = str(status or "").strip()
    if not token:
        return None
    return token in EVALUABLE_REFERENCE_STATUSES


def _comparison_to_metric_row(
    comparison: Any,
    *,
    lecturebank_id: str,
    candidate_ref: str,
    checkpoint_index: Optional[Mapping[str, Any]] = None,
    raw: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    from mas_langgraph.axis.export import _comparison_dict

    from collections import Counter, defaultdict

    payload = _comparison_dict(comparison)
    rows: List[Dict[str, Any]] = []
    _collect_comparisons(
        lecturebank_id,
        candidate_ref,
        [payload],
        {},
        checkpoint_index,
        dict(raw or {}),
        rows,
        defaultdict(Counter),
        Counter(),
    )
    return rows[0] if rows else {}


def _snapshot_llm_ledger(ledger: Any) -> Optional[Dict[str, Any]]:
    if ledger is None:
        return None
    snap_fn = getattr(ledger, "_counter_snapshot", None)
    counters = dict(snap_fn()) if callable(snap_fn) else {}
    responses = getattr(ledger, "_responses", None)
    call_log = getattr(ledger, "call_log", None)
    return {
        "counters": counters,
        "responses": dict(responses) if isinstance(responses, dict) else None,
        "call_log": list(call_log) if isinstance(call_log, list) else None,
    }


def _restore_llm_ledger(ledger: Any, snapshot: Optional[Mapping[str, Any]]) -> None:
    if ledger is None or not snapshot:
        return
    for key, value in dict(snapshot.get("counters") or {}).items():
        if hasattr(ledger, key):
            setattr(ledger, key, value)
    if snapshot.get("responses") is not None and hasattr(ledger, "_responses"):
        ledger._responses.clear()
        ledger._responses.update(snapshot["responses"])
    if snapshot.get("call_log") is not None and hasattr(ledger, "call_log"):
        ledger.call_log[:] = list(snapshot["call_log"])


def _copy_noop_cache(source: Optional[NoOpReplayCache]) -> NoOpReplayCache:
    """Reuse cached NOOP branches without mutating AXIS cache counters/entries."""
    dest = NoOpReplayCache(log=False)
    if source is None:
        return dest
    dest._cache = dict(getattr(source, "_cache", None) or {})
    return dest


def _bind_observation_adapter(
    adapter: Any,
    *,
    case_slice: Any,
    checkpoint_index: Any,
    occurrences: Sequence[Any],
    factual_run_result: Any,
) -> Any:
    if adapter is None:
        return None
    if hasattr(adapter, "case_slice"):
        adapter.case_slice = case_slice
    if hasattr(adapter, "checkpoint_index"):
        adapter.checkpoint_index = checkpoint_index
    if hasattr(adapter, "occurrences"):
        adapter.occurrences = occurrences
    if hasattr(adapter, "produced_by"):
        pb = ()
        if factual_run_result is not None:
            pb = tuple(getattr(factual_run_result, "produced_by", None) or ())
            if not pb:
                fs = getattr(factual_run_result, "final_state", None) or {}
                if isinstance(fs, dict):
                    pb = tuple(fs.get("produced_by") or ())
        adapter.produced_by = pb
    return adapter


def _live_provider_delta(ledger: Any, before: Mapping[str, int], field: str) -> Optional[int]:
    if ledger is None:
        return None
    after = getattr(ledger, field, None)
    if after is None:
        return None
    return int(after or 0) - int(before.get(field) or 0)


def first_attempt_opportunity_counts(
    attempts: Sequence[Any],
) -> Dict[str, Optional[int]]:
    for att in attempts or []:
        mapping = att.to_dict() if hasattr(att, "to_dict") else att
        if not isinstance(mapping, Mapping):
            continue
        return {
            "available_remove_count": mapping.get("available_remove_count"),
            "available_add_count": mapping.get("available_add_count"),
            "available_whatif_count": mapping.get("available_whatif_count"),
        }
    return {
        "available_remove_count": None,
        "available_add_count": None,
        "available_whatif_count": None,
    }


def axis_successful_rounds_at_k(
    attempts: Sequence[Any],
    *,
    k: int,
) -> List[Dict[str, Any]]:
    """First K successful interrogation rounds; DONE is not a successful round."""
    picked: List[Dict[str, Any]] = []
    for att in attempts or []:
        mapping = att.to_dict() if hasattr(att, "to_dict") else dict(att or {})
        if not mapping.get("counted_as_successful_round"):
            continue
        picked.append(mapping)
        if len(picked) >= int(k):
            break
    return picked


def _signature_for_axis_attempt(
    att: Mapping[str, Any],
    *,
    rec_by_query: Mapping[str, Any],
    cmp_by_query: Mapping[str, Any],
    universe_by_query: Mapping[str, str],
) -> Tuple[str, Any]:
    query = str(att.get("generated_query") or "")
    rec = rec_by_query.get(query)
    if rec is not None:
        sig_obj = getattr(rec, "signature", None)
        if sig_obj is not None:
            return canonical_signature_from_query_signature(sig_obj), getattr(
                rec, "replay_comparison", None
            )
    cmp = cmp_by_query.get(query)
    if cmp is not None:
        if hasattr(cmp, "intervention_signature"):
            return canonical_signature_from_query_signature(cmp.intervention_signature), cmp
        if isinstance(cmp, Mapping):
            return canonical_signature_from_mapping(cmp), cmp
    if query in universe_by_query:
        return universe_by_query[query], cmp
    if att.get("reference_intervention_signature"):
        return str(att.get("reference_intervention_signature")), cmp
    if att.get("checkpoint_id") or att.get("target_ref"):
        return (
            canonical_intervention_signature(
                operation=att.get("parsed_operation"),
                target_ref=att.get("target_ref"),
                replacement_ref=att.get("replacement_ref"),
                checkpoint_id=att.get("checkpoint_id"),
            ),
            cmp,
        )
    return "", None


def axis_counterfactuals_at_k(
    attempts: Sequence[Any],
    comparisons: Sequence[Any] = (),
    *,
    k: int,
    intervention_records: Sequence[Any] = (),
    universe_by_query: Optional[Mapping[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Unique AXIS counterfactuals within the first K successful rounds."""
    rounds = axis_successful_rounds_at_k(attempts, k=k)
    rec_by_query = {
        str(getattr(rec, "generated_query", "") or ""): rec
        for rec in intervention_records or []
    }
    cmp_by_query: Dict[str, Any] = {}
    for rec in intervention_records or []:
        cmp = getattr(rec, "replay_comparison", None)
        query = str(getattr(rec, "generated_query", "") or "")
        if cmp is not None and query:
            cmp_by_query[query] = cmp
    for cmp in comparisons or []:
        if isinstance(cmp, Mapping) and cmp.get("generated_query"):
            cmp_by_query.setdefault(str(cmp.get("generated_query")), cmp)
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    round_by_sig: Dict[str, int] = {}
    successful_round = 0
    for att in rounds:
        successful_round += 1
        op = canonical_operation(att.get("parsed_operation"))
        if op not in {"REMOVE", "ADD", "WHATIF"}:
            continue
        sig, cmp = _signature_for_axis_attempt(
            att,
            rec_by_query=rec_by_query,
            cmp_by_query=cmp_by_query,
            universe_by_query=universe_by_query or {},
        )
        if not sig:
            continue
        if sig in seen:
            round_by_sig.setdefault(sig, successful_round)
            continue
        seen.add(sig)
        round_by_sig[sig] = successful_round
        selected.append(
            {
                "signature": sig,
                "operation": op,
                "generated_query": str(att.get("generated_query") or ""),
                "axis_successful_round": successful_round,
                "comparison": cmp,
            }
        )
    return selected, round_by_sig


def _csv_cell(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def reference_row_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("lecturebank_id") or ""),
        str(row.get("candidate_ref") or ""),
        str(row.get("reference_intervention_signature") or ""),
    )


def load_reference_intervention_rows(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if value in (None, "", "NA"):
                    row[key] = None
                elif str(value).lower() == "true":
                    row[key] = True
                elif str(value).lower() == "false":
                    row[key] = False
                else:
                    row[key] = value
            rows.append(row)
        return rows


def upsert_reference_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] = REFERENCE_INTERVENTION_FIELDS,
) -> None:
    existing = load_reference_intervention_rows(path)
    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in existing:
        merged[reference_row_key(row)] = dict(row)
    for row in rows:
        merged[reference_row_key(row)] = dict(row)
    ordered = [merged[k] for k in sorted(merged)]
    atomic_write_csv(
        Path(path),
        [{k: _csv_cell(row.get(k)) for k in fieldnames} for row in ordered],
        fieldnames=fieldnames,
    )


def resume_row_is_complete(row: Mapping[str, Any]) -> bool:
    status = str(row.get("reference_status") or "")
    if status in TERMINAL_RESUME_STATUSES:
        return True
    return False


def print_reference_sweep_plan(
    *,
    lecturebank_id: str,
    candidate_ref: str,
    universe: Sequence[Mapping[str, Any]],
    log: Any = None,
) -> None:
    remove_n = sum(1 for r in universe if r.get("operation") == "REMOVE")
    whatif_n = sum(1 for r in universe if r.get("operation") == "WHATIF")
    add_n = sum(1 for r in universe if r.get("operation") == "ADD")
    total = len(universe)
    stream = log
    print("Reference interventions to execute:", file=stream)
    print(f"  lecturebank_id={lecturebank_id} candidate_ref={candidate_ref}", file=stream)
    print(f"  - REMOVE: {remove_n}", file=stream)
    print(f"  - WHATIF: {whatif_n}", file=stream)
    print(f"  - ADD: {add_n}", file=stream)
    print(f"  - TOTAL: {total}", file=stream)


def execute_reference_intervention(
    *,
    candidate: Mapping[str, Any],
    executor: AxisQueryExecutor,
    comparator: ReplayComparator,
    factual_obs: Any,
    observation_adapter: Any,
    explained_case: Any,
    lecturebank_id: str,
    candidate_ref: str,
    checkpoint_index: Optional[Mapping[str, Any]] = None,
    ledger: Any = None,
    raw: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved = candidate["resolved"]
    sig = candidate["signature"]
    row = {
        "lecturebank_id": lecturebank_id,
        "candidate_ref": candidate_ref,
        "reference_intervention_signature": candidate["reference_intervention_signature"],
        "operation": candidate["operation"],
        "target_ref": candidate["target_ref"],
        "replacement_ref": candidate.get("replacement_ref") or "",
        "checkpoint_id": candidate.get("checkpoint_id"),
        "checkpoint_alias": candidate.get("checkpoint_alias"),
        "boundary_raw": candidate.get("boundary_raw"),
        "generated_query": candidate.get("generated_query"),
        "validation_status": "valid",
        "execution_status": None,
        "reference_status": None,
        "axis_interrogation_mode": "hybrid",
    }
    before = {}
    snap_fn = getattr(ledger, "_counter_snapshot", None)
    if callable(snap_fn):
        before = dict(snap_fn())
    try:
        rec = executor.execute(resolved)
    except Exception as exc:  # noqa: BLE001
        row["execution_status"] = "failed"
        row["reference_status"] = "replay_failed"
        row["execution_error"] = str(exc)
        return row
    if getattr(rec, "error", None):
        row["execution_status"] = "failed"
        row["reference_status"] = "replay_failed"
        row["execution_error"] = str(rec.error)
        row["noop_live_provider_calls"] = _live_provider_delta(
            ledger, before, "noop_live_provider_calls"
        )
        row["counterfactual_live_provider_calls"] = _live_provider_delta(
            ledger, before, "counterfactual_live_provider_calls"
        )
        return row
    row["execution_status"] = "executed"
    noop = getattr(rec, "noop_result", None)
    cf = getattr(rec, "counterfactual_result", None)
    metrics = {}
    if cf is not None:
        from mas_langgraph.axis.executor import _branch_replay_metrics

        metrics = _branch_replay_metrics(cf)
        row["replay_completion_mode"] = metrics.get("completion_mode")
        row["executed_node_count"] = metrics.get("executed_node_count")
    row["noop_live_provider_calls"] = _live_provider_delta(
        ledger, before, "noop_live_provider_calls"
    )
    row["counterfactual_live_provider_calls"] = _live_provider_delta(
        ledger, before, "counterfactual_live_provider_calls"
    )
    if not noop or not cf:
        row["reference_status"] = "replay_failed"
        return row
    try:
        comparison = comparator.compare(
            signature=sig,
            factual_observation=factual_obs,
            noop_result=noop,
            counterfactual_result=cf,
            observation_adapter=observation_adapter,
            explained_case=explained_case,
        )
    except Exception as exc:  # noqa: BLE001
        row["reference_status"] = "replay_failed"
        row["execution_error"] = f"comparison failed: {exc}"
        return row
    cmp_row = _comparison_to_metric_row(
        comparison,
        lecturebank_id=lecturebank_id,
        candidate_ref=candidate_ref,
        checkpoint_index=checkpoint_index,
        raw=raw,
    )
    row["noop_decision_stable"] = cmp_row.get("decision_control_stable")
    row["noop_criteria_stable"] = cmp_row.get("criteria_control_stable")
    row["noop_content_stable"] = cmp_row.get("subject_content_control_stable")
    row["full_noop_stable"] = cmp_row.get("full_control_stable")
    row["valid_comparison"] = cmp_row.get("valid_comparison")
    row["decision_effect"] = cmp_row.get("decision_effect")
    row["criterion_effect"] = cmp_row.get("criterion_effect_any_supported")
    row["decision_relevant_criterion_effect"] = cmp_row.get(
        "decision_relevant_criterion_effect_supported"
    )
    row["content_effect"] = cmp_row.get("content_effect")
    row["stem_effect"] = cmp_row.get("stem_effect")
    row["correct_answer_effect"] = cmp_row.get("correct_answer_effect")
    row["distractor_effect"] = cmp_row.get("distractor_effect")
    row["answer_explanation_effect"] = cmp_row.get("answer_explanation_effect")
    row["question_not_reproduced"] = cmp_row.get("question_not_reproduced")
    row["evidence_status"] = cmp_row.get("evidence_status")
    row["causal_classification"] = cmp_row.get("causal_classification")
    row["information_role"] = cmp_row.get("information_role")
    if not row.get("replay_completion_mode"):
        row["replay_completion_mode"] = cmp_row.get("replay_completion_mode")
    if row.get("executed_node_count") in (None, ""):
        row["executed_node_count"] = cmp_row.get("executed_node_count")
    row["_comparison_row"] = cmp_row
    row["reference_status"] = classify_reference_status(
        validation_status="valid",
        execution_status="executed",
        comparison_row=cmp_row,
    )
    effect = supported_effect_from_comparison_row(cmp_row)
    row["supported_effect"] = (
        True if row["reference_status"] == "valid_effect" else (
            False if row["reference_status"] == "valid_no_effect" else None
        )
    )
    del effect
    return row


def _public_reference_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = {k: row.get(k) for k in REFERENCE_INTERVENTION_FIELDS}
    out["reference_evaluable"] = reference_status_is_evaluable(row.get("reference_status"))
    return out


def compute_effect_discovery_for_question(
    *,
    lecturebank_id: str,
    candidate_ref: str,
    k: int,
    reference_rows: Sequence[Mapping[str, Any]],
    axis_attempts: Sequence[Any],
    axis_comparisons: Sequence[Any] = (),
    reference_set_complete: bool,
    initial_opportunity: Optional[Mapping[str, Any]] = None,
    axis_interrogation_mode: str = "hybrid",
    intervention_records: Sequence[Any] = (),
) -> Dict[str, Any]:
    universe = list(reference_rows)
    e_q = {
        str(r.get("reference_intervention_signature"))
        for r in universe
        if r.get("reference_status") == "valid_effect"
    }
    r_eval = {
        str(r.get("reference_intervention_signature"))
        for r in universe
        if reference_status_is_evaluable(r.get("reference_status")) is True
    }
    universe_by_query = {
        str(r.get("generated_query") or ""): str(r.get("reference_intervention_signature") or "")
        for r in universe
        if r.get("generated_query")
    }
    selected, round_by_sig = axis_counterfactuals_at_k(
        axis_attempts,
        axis_comparisons,
        k=k,
        intervention_records=intervention_records,
        universe_by_query=universe_by_query,
    )
    s_q = {str(item["signature"]) for item in selected}
    v_q: set[str] = set()
    for item in selected:
        cmp = item.get("comparison")
        cmp_row = None
        if isinstance(cmp, Mapping) and "valid_comparison" in cmp:
            cmp_row = cmp
        elif cmp is not None:
            try:
                cmp_row = _comparison_to_metric_row(
                    cmp,
                    lecturebank_id=lecturebank_id,
                    candidate_ref=candidate_ref,
                )
            except Exception:
                cmp_row = None
        if (
            cmp_row
            and cmp_row.get("valid_comparison") is True
            and cmp_row.get("full_control_stable") is True
        ):
            v_q.add(item["signature"])
        elif isinstance(cmp, Mapping) and cmp.get("valid_comparison") is True and cmp.get(
            "full_control_stable"
        ) is True:
            v_q.add(item["signature"])
    discovered = s_q & e_q
    v_eval = v_q & r_eval
    uneval = v_q - r_eval
    precision_hits = v_eval & e_q
    recall = ratio(len(discovered), len(e_q)) if e_q else NA
    if not reference_set_complete:
        recall = NA
    precision = ratio(len(precision_hits), len(v_eval))
    e2e = ratio(len(discovered), len(s_q))
    evaluable_count = sum(
        1 for r in universe if reference_status_is_evaluable(r.get("reference_status")) is True
    )
    evaluability_rate = ratio(evaluable_count, len(universe))
    causal_evidence = any(
        True
        for item in selected
        if item["signature"] in v_q
    )
    remove_n = sum(1 for r in universe if r.get("operation") == "REMOVE")
    whatif_n = sum(1 for r in universe if r.get("operation") == "WHATIF")
    add_n = sum(1 for r in universe if r.get("operation") == "ADD")
    opp = dict(initial_opportunity or {})
    match = None
    detail = ""
    if any(v is not None for v in opp.values()):
        want = (
            int(opp.get("available_remove_count") or 0),
            int(opp.get("available_add_count") or 0),
            int(opp.get("available_whatif_count") or 0),
        )
        got = (remove_n, add_n, whatif_n)
        match = want == got
        if not match:
            detail = (
                "initial advertised opportunities "
                f"REMOVE/ADD/WHATIF={want} vs reference universe {got}"
            )
    universe_sigs = {str(r.get("reference_intervention_signature")) for r in universe}
    missing = [
        item["signature"]
        for item in selected
        if item["signature"] not in universe_sigs
    ]
    annotated = []
    for row in universe:
        sig = str(row.get("reference_intervention_signature"))
        updated = dict(row)
        updated["selected_by_axis"] = sig in s_q
        updated["selected_within_k"] = sig in s_q
        updated["axis_successful_round"] = round_by_sig.get(sig)
        updated["axis_selected_signature_missing_from_reference"] = False
        updated["reference_evaluable"] = reference_status_is_evaluable(
            updated.get("reference_status")
        )
        annotated.append(updated)
    return {
        "question_row": {
            "lecturebank_id": lecturebank_id,
            "candidate_ref": candidate_ref,
            "k": int(k),
            "reference_universe_count": len(universe),
            "reference_remove_count": remove_n,
            "reference_whatif_count": whatif_n,
            "reference_add_count": add_n,
            "reference_valid_comparison_count": sum(
                1 for r in universe if r.get("reference_status") in {"valid_effect", "valid_no_effect"}
            ),
            "reference_inconclusive_count": sum(
                1 for r in universe if r.get("reference_status") == "inconclusive_noop_unstable"
            ),
            "reference_replay_failed_count": sum(
                1 for r in universe if r.get("reference_status") == "replay_failed"
            ),
            "reference_effectful_count": len(e_q),
            "reference_no_effect_count": sum(
                1 for r in universe if r.get("reference_status") == "valid_no_effect"
            ),
            "reference_evaluable_count": evaluable_count,
            "reference_evaluability_rate": evaluability_rate,
            "axis_counterfactual_selected_at_k": len(s_q),
            "axis_valid_counterfactual_comparisons_at_k": len(v_q),
            "axis_reference_evaluable_counterfactuals_at_k": len(v_eval),
            "axis_reference_unevaluable_counterfactuals_at_k": len(uneval),
            "axis_selected_effectful_at_k": len(discovered),
            "axis_effectful_discoveries_at_k": len(precision_hits),
            "axis_valid_effectful_discoveries_at_k": len(precision_hits),
            "effect_discovery_recall_at_k": recall,
            "effect_precision_at_k": precision,
            "effect_discovery_end_to_end_yield_at_k": e2e,
            "causal_evidence_available_at_k": causal_evidence,
            "reference_set_complete": bool(reference_set_complete),
            "reference_opportunity_count_match": match,
            "initial_available_remove_count": opp.get("available_remove_count"),
            "initial_available_add_count": opp.get("available_add_count"),
            "initial_available_whatif_count": opp.get("available_whatif_count"),
            "opportunity_mismatch_detail": detail,
            "axis_selected_signature_missing_from_reference": bool(missing),
            "axis_interrogation_mode": axis_interrogation_mode,
        },
        "reference_rows": annotated,
        "missing_axis_signatures": missing,
        "s_q": s_q,
        "e_q": e_q,
        "r_eval": r_eval,
        "v_q": v_q,
        "v_eval": v_eval,
        "discovered": discovered,
    }


def _axis_selected_effectful_count(row: Mapping[str, Any]) -> int:
    """Recall/e2e numerator: unique AXIS-selected signatures in E_q.

    Newer rows store this as axis_selected_effectful_at_k. Older rows stored
    |S_q ∩ E_q| in axis_effectful_discoveries_at_k.
    """
    if "axis_selected_effectful_at_k" in row and row.get("axis_selected_effectful_at_k") not in (
        None,
        "",
        "NA",
    ):
        return int(row.get("axis_selected_effectful_at_k") or 0)
    return int(row.get("axis_effectful_discoveries_at_k") or 0)


def aggregate_effect_discovery(
    question_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = [dict(r) for r in question_rows if r]
    qn = len(rows)
    precision_defined = [
        r for r in rows if r.get("effect_precision_at_k") is not None
    ]
    complete = [r for r in rows if r.get("reference_set_complete") is True]
    recall_eligible = [
        r
        for r in complete
        if int(r.get("reference_effectful_count") or 0) > 0
        and r.get("effect_discovery_recall_at_k") is not None
    ]
    micro_disc = sum(_axis_selected_effectful_count(r) for r in recall_eligible)
    micro_eq = sum(int(r.get("reference_effectful_count") or 0) for r in recall_eligible)
    micro_v_eval = sum(
        int(r.get("axis_reference_evaluable_counterfactuals_at_k") or 0) for r in rows
    )
    micro_prec_hits = sum(int(r.get("axis_effectful_discoveries_at_k") or 0) for r in rows)
    coverage_n = sum(1 for r in rows if r.get("causal_evidence_available_at_k") is True)
    match_n = sum(1 for r in rows if r.get("reference_opportunity_count_match") is True)
    match_d = sum(1 for r in rows if r.get("reference_opportunity_count_match") is not None)
    e2e_n = sum(_axis_selected_effectful_count(r) for r in rows)
    e2e_d = sum(int(r.get("axis_counterfactual_selected_at_k") or 0) for r in rows)
    na_recall = sum(
        1
        for r in complete
        if int(r.get("reference_effectful_count") or 0) == 0
    )
    incomplete = sum(1 for r in rows if r.get("reference_set_complete") is not True)
    macro_recall = (
        sum(float(r["effect_discovery_recall_at_k"]) for r in recall_eligible) / len(recall_eligible)
        if recall_eligible
        else NA
    )
    macro_precision = (
        sum(float(r["effect_precision_at_k"]) for r in precision_defined) / len(precision_defined)
        if precision_defined
        else NA
    )
    eval_rows = [r for r in rows if int(r.get("reference_universe_count") or 0) > 0]
    evaluable_count = sum(
        int(
            r.get("reference_evaluable_count")
            if r.get("reference_evaluable_count") not in (None, "", "NA")
            else r.get("reference_valid_comparison_count")
            or 0
        )
        for r in rows
    )
    universe_count = sum(int(r.get("reference_universe_count") or 0) for r in rows)
    eval_micro_n = sum(
        int(
            r.get("reference_evaluable_count")
            if r.get("reference_evaluable_count") not in (None, "", "NA")
            else r.get("reference_valid_comparison_count")
            or 0
        )
        for r in eval_rows
    )
    eval_micro_d = sum(int(r.get("reference_universe_count") or 0) for r in eval_rows)
    eval_rates = []
    for r in eval_rows:
        rate = r.get("reference_evaluability_rate")
        if rate is None or rate == "" or rate == "NA":
            num = r.get("reference_evaluable_count")
            if num in (None, "", "NA"):
                num = r.get("reference_valid_comparison_count")
            rate = ratio(num, r.get("reference_universe_count"))
        if rate is not None:
            eval_rates.append(float(rate))
    macro_eval = (sum(eval_rates) / len(eval_rates)) if eval_rates else NA
    return {
        "reference_benchmark_question_count": qn,
        "reference_intervention_count": universe_count,
        "reference_valid_comparison_count": sum(
            int(r.get("reference_valid_comparison_count") or 0) for r in rows
        ),
        "reference_evaluable_count": evaluable_count,
        "reference_evaluability_rate_micro": ratio(eval_micro_n, eval_micro_d),
        "reference_evaluability_rate_macro": macro_eval,
        "reference_evaluability_micro_num": eval_micro_n,
        "reference_evaluability_micro_den": eval_micro_d,
        "reference_evaluability_macro_num": len(eval_rates),
        "axis_reference_evaluable_counterfactuals_at_k": micro_v_eval,
        "axis_reference_unevaluable_counterfactuals_at_k": sum(
            int(r.get("axis_reference_unevaluable_counterfactuals_at_k") or 0) for r in rows
        ),
        "reference_inconclusive_count": sum(
            int(r.get("reference_inconclusive_count") or 0) for r in rows
        ),
        "reference_replay_failed_count": sum(
            int(r.get("reference_replay_failed_count") or 0) for r in rows
        ),
        "reference_effectful_count": sum(int(r.get("reference_effectful_count") or 0) for r in rows),
        "reference_no_effect_count": sum(int(r.get("reference_no_effect_count") or 0) for r in rows),
        "reference_remove_count": sum(int(r.get("reference_remove_count") or 0) for r in rows),
        "reference_whatif_count": sum(int(r.get("reference_whatif_count") or 0) for r in rows),
        "reference_add_count": sum(int(r.get("reference_add_count") or 0) for r in rows),
        "causal_evidence_coverage_at_k": ratio(coverage_n, qn),
        "causal_evidence_coverage_num": coverage_n,
        "causal_evidence_coverage_den": qn,
        "effect_discovery_recall_at_k_macro": macro_recall,
        "effect_discovery_recall_at_k_micro": ratio(micro_disc, micro_eq),
        "effect_discovery_recall_macro_num": len(recall_eligible),
        "effect_discovery_recall_micro_num": micro_disc,
        "effect_discovery_recall_micro_den": micro_eq,
        "effect_precision_at_k_macro": macro_precision,
        "effect_precision_at_k_micro": ratio(micro_prec_hits, micro_v_eval),
        "effect_precision_macro_num": len(precision_defined),
        "effect_precision_micro_num": micro_prec_hits,
        "effect_precision_micro_den": micro_v_eval,
        "effect_discovery_end_to_end_yield_at_k": ratio(e2e_n, e2e_d),
        "effect_discovery_end_to_end_num": e2e_n,
        "effect_discovery_end_to_end_den": e2e_d,
        "reference_opportunity_count_match_rate": ratio(match_n, match_d),
        "reference_opportunity_match_num": match_n,
        "reference_opportunity_match_den": match_d,
        "questions_with_empty_effectful_reference": na_recall,
        "questions_with_incomplete_reference_set": incomplete,
        "k": rows[0].get("k") if rows else None,
        "reference_set_complete": incomplete == 0 and qn > 0,
    }


def require_hybrid_reference_mode(axis_interrogation_mode: Any) -> None:
    if is_hybrid_interrogation_mode(axis_interrogation_mode):
        return
    raise HybridReferenceSweepError(
        "--build-effect-reference-set currently supports only "
        "--axis-interrogation-mode hybrid. Constrained mode is not supported "
        "because the exhaustive replay-reference universe must be the hybrid "
        "intervention vocabulary, not the constrained exact-choice list."
    )


def enumerate_effect_reference_universe_for_case(
    *,
    axis_result: Any,
    replay_executor: Any,
    factual_run_result: Any,
    case_definition: Any,
    checkpoint_index: Any,
    contracts: Any,
    intervention_registry: Any = None,
    intervention_alternatives: Optional[Mapping[str, Any]] = None,
    registry: Any = None,
    occurrences: Sequence[Any] = (),
    axis_interrogation_mode: str = "hybrid",
) -> Tuple[List[Dict[str, Any]], List[str], Any]:
    """Build the frozen hybrid replay-reference universe for one AXIS case."""
    require_hybrid_reference_mode(axis_interrogation_mode)
    explained = getattr(axis_result, "explained_case", None)
    case_slice = getattr(axis_result, "case_slice", None)
    if explained is None or case_slice is None:
        raise HybridReferenceSweepError("AXIS result missing explained_case or case_slice")
    registry = registry or getattr(factual_run_result, "artifact_registry", None)
    occurrences = tuple(occurrences or getattr(factual_run_result, "occurrences", ()) or ())
    priority_fn = None
    if case_definition is not None and hasattr(case_definition, "intervention_artifact_priority"):
        priority_fn = case_definition.intervention_artifact_priority
    vocab, _cands = build_initial_hybrid_vocabulary(
        case_slice=case_slice,
        registry=registry,
        contracts=contracts,
        checkpoint_index=checkpoint_index,
        explained_case=explained,
        intervention_alternatives=intervention_alternatives,
        intervention_registry=intervention_registry,
        all_occurrences=occurrences,
        replay_executor=replay_executor,
        run_id=str(getattr(explained, "run_id", "") or ""),
        priority_fn=priority_fn,
    )
    extra_aliases = _hybrid_context_aliases(vocab, intervention_alternatives)
    parser = AxisQueryParser()
    context = AxisQueryContext(
        explained_case=explained,
        case_slice=case_slice,
        registry=registry,
        checkpoint_index=checkpoint_index,
        contracts=contracts,
        intervention_registry=intervention_registry,
        alias_map=parser.alias_map,
        intervention_alternatives=extra_aliases,
    )
    universe, enum_warnings = enumerate_initial_hybrid_universe(
        vocabulary=vocab,
        context=context,
        parser=parser,
    )
    return list(universe), list(enum_warnings), vocab


def run_effect_reference_sweep_for_case(
    *,
    lecturebank_id: str,
    candidate_ref: str,
    axis_result: Any,
    replay_executor: Any,
    factual_run_result: Any,
    case_definition: Any,
    checkpoint_index: Any,
    contracts: Any,
    intervention_registry: Any = None,
    intervention_alternatives: Optional[Mapping[str, Any]] = None,
    registry: Any = None,
    occurrences: Sequence[Any] = (),
    max_rounds: int = 5,
    axis_interrogation_mode: str = "hybrid",
    reference_max_interventions_per_question: Optional[int] = None,
    resume_rows: Sequence[Mapping[str, Any]] = (),
    persist_rows_callback: Optional[Callable[[Sequence[Mapping[str, Any]]], None]] = None,
    observation_adapter: Any = None,
    axis_noop_cache: Optional[NoOpReplayCache] = None,
    log: Any = None,
    checkpoint_index_mapping: Optional[Mapping[str, Any]] = None,
    frozen_universe: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the exhaustive replay-reference sweep AFTER normal AXIS explanation."""
    require_hybrid_reference_mode(axis_interrogation_mode)
    explained = getattr(axis_result, "explained_case", None)
    case_slice = getattr(axis_result, "case_slice", None)
    if explained is None or case_slice is None:
        raise HybridReferenceSweepError("AXIS result missing explained_case or case_slice")
    registry = registry or getattr(factual_run_result, "artifact_registry", None)
    occurrences = tuple(occurrences or getattr(factual_run_result, "occurrences", ()) or ())
    vocab = None
    if frozen_universe is not None:
        universe = list(frozen_universe)
        enum_warnings: List[str] = []
    else:
        universe, enum_warnings, vocab = enumerate_effect_reference_universe_for_case(
            axis_result=axis_result,
            replay_executor=replay_executor,
            factual_run_result=factual_run_result,
            case_definition=case_definition,
            checkpoint_index=checkpoint_index,
            contracts=contracts,
            intervention_registry=intervention_registry,
            intervention_alternatives=intervention_alternatives,
            registry=registry,
            occurrences=occurrences,
            axis_interrogation_mode=axis_interrogation_mode,
        )
    extra_aliases = (
        _hybrid_context_aliases(vocab, intervention_alternatives)
        if vocab is not None
        else dict(intervention_alternatives or {})
    )
    parser = AxisQueryParser()
    context = AxisQueryContext(
        explained_case=explained,
        case_slice=case_slice,
        registry=registry,
        checkpoint_index=checkpoint_index,
        contracts=contracts,
        intervention_registry=intervention_registry,
        alias_map=parser.alias_map,
        intervention_alternatives=extra_aliases,
    )
    print_reference_sweep_plan(
        lecturebank_id=lecturebank_id,
        candidate_ref=candidate_ref,
        universe=universe,
        log=log,
    )
    complete = True
    work = list(universe)
    if reference_max_interventions_per_question is not None:
        limit = int(reference_max_interventions_per_question)
        if len(work) > limit:
            work = work[:limit]
            complete = False
    resume_by_sig = {
        str(r.get("reference_intervention_signature")): dict(r)
        for r in resume_rows
        if str(r.get("lecturebank_id") or "") == str(lecturebank_id)
        and str(r.get("candidate_ref") or "") == str(candidate_ref)
    }
    if observation_adapter is None and case_definition is not None:
        observation_adapter = getattr(case_definition, "observation_adapter", None)
        if observation_adapter is None:
            loader = getattr(case_definition, "load_run_observation_adapter", None)
            if callable(loader):
                try:
                    observation_adapter = loader()
                except Exception:
                    observation_adapter = None
    observation_adapter = _bind_observation_adapter(
        observation_adapter,
        case_slice=case_slice,
        checkpoint_index=checkpoint_index,
        occurrences=occurrences,
        factual_run_result=factual_run_result,
    )
    if observation_adapter is None:
        raise HybridReferenceSweepError(
            "observation adapter is required to reuse AXIS replay comparison"
        )
    factual_obs = observation_adapter.observe_factual(
        explained_case=explained,
        case_slice=case_slice,
        registry=registry,
    )
    sweep_cache = _copy_noop_cache(axis_noop_cache)
    executor = AxisQueryExecutor(
        context=context,
        replay_executor=replay_executor,
        noop_cache=sweep_cache,
        stop_when_observable=True,
    )
    comparator = ReplayComparator()
    run_id = str(getattr(explained, "run_id", "") or "")
    ledger = None
    getter = getattr(replay_executor, "get_or_create_llm_ledger", None)
    if callable(getter) and run_id:
        try:
            ledger = getter(run_id)
        except Exception:
            ledger = None
    ledger_snap = _snapshot_llm_ledger(ledger)
    from mas_langgraph.experiment.timing import ExperimentTimingCollector, timing_scope

    throwaway = ExperimentTimingCollector()
    executed_rows: List[Dict[str, Any]] = []
    try:
        with timing_scope(throwaway):
            for cand in work:
                sig = str(cand["reference_intervention_signature"])
                prior = resume_by_sig.get(sig)
                if prior is not None and resume_row_is_complete(prior):
                    executed_rows.append(prior)
                    continue
                row = execute_reference_intervention(
                    candidate=cand,
                    executor=executor,
                    comparator=comparator,
                    factual_obs=factual_obs,
                    observation_adapter=observation_adapter,
                    explained_case=explained,
                    lecturebank_id=lecturebank_id,
                    candidate_ref=candidate_ref,
                    checkpoint_index=checkpoint_index_mapping,
                    ledger=ledger,
                    raw={"relevant_artifacts": list(getattr(case_slice, "relevant_artifact_refs", ()) or ())},
                )
                executed_rows.append(row)
                if persist_rows_callback is not None:
                    persist_rows_callback([_public_reference_row(r) for r in executed_rows])
    finally:
        _restore_llm_ledger(ledger, ledger_snap)

    initial_opp = first_attempt_opportunity_counts(
        getattr(axis_result, "interrogation_attempts", ()) or ()
    )
    discovery = compute_effect_discovery_for_question(
        lecturebank_id=lecturebank_id,
        candidate_ref=candidate_ref,
        k=int(max_rounds),
        reference_rows=executed_rows,
        axis_attempts=getattr(axis_result, "interrogation_attempts", ()) or (),
        axis_comparisons=getattr(axis_result, "replay_comparisons", ()) or (),
        reference_set_complete=complete,
        intervention_records=getattr(axis_result, "intervention_records", ()) or (),
        initial_opportunity=initial_opp,
        axis_interrogation_mode="hybrid",
    )
    # Opportunity match and AXIS-join use the FULL initial U_q even if a
    # smoke-test limiter truncated execution.
    qrow = discovery["question_row"]
    full_remove = sum(1 for r in universe if r.get("operation") == "REMOVE")
    full_whatif = sum(1 for r in universe if r.get("operation") == "WHATIF")
    full_add = sum(1 for r in universe if r.get("operation") == "ADD")
    qrow["reference_universe_count"] = len(universe)
    qrow["reference_evaluability_rate"] = ratio(
        qrow.get("reference_evaluable_count"),
        qrow.get("reference_universe_count"),
    )
    qrow["reference_remove_count"] = full_remove
    qrow["reference_whatif_count"] = full_whatif
    qrow["reference_add_count"] = full_add
    if any(v is not None for v in initial_opp.values()):
        want = (
            int(initial_opp.get("available_remove_count") or 0),
            int(initial_opp.get("available_add_count") or 0),
            int(initial_opp.get("available_whatif_count") or 0),
        )
        got = (full_remove, full_add, full_whatif)
        qrow["reference_opportunity_count_match"] = want == got
        qrow["opportunity_mismatch_detail"] = (
            ""
            if want == got
            else (
                "initial advertised opportunities "
                f"REMOVE/ADD/WHATIF={want} vs reference universe {got}"
            )
        )
    full_sigs = {str(r.get("reference_intervention_signature")) for r in universe}
    missing = [sig for sig in discovery["s_q"] if sig not in full_sigs]
    discovery["missing_axis_signatures"] = missing
    qrow["axis_selected_signature_missing_from_reference"] = bool(missing)
    if discovery["question_row"].get("reference_opportunity_count_match") is False:
        enum_warnings.append(str(discovery["question_row"].get("opportunity_mismatch_detail") or ""))
    if discovery["missing_axis_signatures"]:
        enum_warnings.append(
            "AXIS-selected hybrid intervention(s) missing from the initial "
            "replay-reference universe: "
            + ", ".join(discovery["missing_axis_signatures"])
        )
    public_rows = [_public_reference_row(r) for r in discovery["reference_rows"]]
    if persist_rows_callback is not None:
        persist_rows_callback(public_rows)
    return {
        "intervention_rows": public_rows,
        "question_row": discovery["question_row"],
        "warnings": enum_warnings,
        "reference_set_complete": complete,
        "vocabulary_counts": (
            hybrid_opportunity_counts(vocab)
            if vocab is not None
            else {
                "available_remove_count": sum(
                    1 for r in universe if r.get("operation") == "REMOVE"
                ),
                "available_whatif_count": sum(
                    1 for r in universe if r.get("operation") == "WHATIF"
                ),
                "available_add_count": sum(
                    1 for r in universe if r.get("operation") == "ADD"
                ),
            }
        ),
        "executor": executor,
    }
