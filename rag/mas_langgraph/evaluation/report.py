"""Aggregate CSV / Markdown writers for LectureBank AXIS evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from mas_langgraph.batch.io import atomic_write_csv, atomic_write_text
from mas_langgraph.evaluation.axis_metrics import NA, describe, pct, ratio

DOCUMENT_FIELDS = [
    "lecturebank_id",
    "title",
    "topic_id",
    "topic_name",
    "instructor",
    "venue",
    "year",
    "url",
    "document_id",
    "pdf_source",
    "pdf_sha256",
    "failure_stage",
    "error_message",
    "rag_status",
    "mas_status",
    "axis_status",
    "eval_status",
    "accepted_question_count",
    "explained_question_count",
    "valid_replay_comparisons",
    "counterfactual_executions",
    "successful_replay_count",
    "intervention_validity_rate",
    "valid_comparison_rate",
    "full_noop_stability_rate",
    "content_effect_rate",
    "criterion_effect_rate",
    "decision_effect_rate",
    "noop_factual_ledger_hit_rate",
    "effect_yield_any",
    "checkpoint_position_pct_mean",
    "checkpoint_progress_pct_mean",
    "provenance_relation_completeness_rate",
    "intervention_target_in_slice_rate",
    "factual_workflow_s",
    "axis_total_s",
    "experiment_total_s",
    "axis_interrogation_mode",
    "axis_interrogation_policy",
]

ATTEMPT_FIELDS = [
    "lecturebank_id",
    "candidate_ref",
    "attempt_index",
    "generated_query",
    "parsed_operation",
    "parsed_operation_normalized",
    "axis_figure4_operation",
    "validation_status",
    "executed",
    "produced_counterfactual",
    "counted_as_successful_round",
    "parser_error",
    "resolution_error",
    "duplicate_error",
    "contract_error",
    "execution_error",
    "vocabulary_error",
    "available_what_count",
    "available_remove_count",
    "available_add_count",
    "available_whatif_count",
    "axis_interrogation_mode",
    "axis_interrogation_policy",
]

COMPARISON_FIELDS = [
    "lecturebank_id",
    "candidate_ref",
    "round",
    "operation",
    "operation_raw",
    "target_ref",
    "target_artifact_type",
    "target_artifact_type_source",
    "replacement_ref",
    "boundary_raw",
    "checkpoint_alias",
    "checkpoint_id",
    "checkpoint_alias_index",
    "checkpoint_ordinal_1based",
    "checkpoint_total",
    "checkpoint_position_pct",
    "checkpoint_progress_pct",
    "evidence_status",
    "decision_control_stable",
    "criteria_control_stable",
    "subject_content_control_stable",
    "full_control_stable",
    "valid_comparison",
    "criterion_effect_any",
    "decision_relevant_criterion_effect",
    "content_effect",
    "content_effect_raw",
    "decision_effect",
    "decision_effect_raw",
    "question_content_effect_status",
    "stem_effect",
    "correct_answer_effect",
    "distractor_effect",
    "answer_explanation_effect",
    "concept_change_label",
    "concept_change_reason",
    "concept_change_eligibility_reason",
    "noop_tested_concept",
    "counterfactual_tested_concept",
    "concept_change_evaluator_model",
    "concept_change_rubric_version",
    "information_role",
    "information_role_supported",
    "causal_classification",
    "semantic_information_absent",
    "derived_information_may_persist",
    "intervention_scope",
    "replay_completion_mode",
    "executed_node_count",
]

PROVENANCE_FIELDS = [
    "lecturebank_id",
    "candidate_ref",
    "relation",
    "artifact_ref",
    "artifact_type",
    "producer_occurrence_id",
    "consumer_occurrence_id",
    "producer_node",
    "consumer_node",
]

DIST_FIELDS = [
    "scope",
    "lecturebank_id",
    "dimension",
    "value",
    "count",
    "denominator",
    "percentage",
]

ROUND_FIELDS = [
    "lecturebank_id",
    "round",
    "comparison_count",
    "valid_comparison_count",
    "valid_comparison_rate",
    "operation_distribution",
    "target_artifact_type_distribution",
    "mean_checkpoint_position_pct",
    "full_noop_stability_rate",
    "content_effect_rate",
    "criterion_effect_rate",
    "decision_effect_rate",
    "causal_classification_distribution",
    "information_role_distribution",
]

OVERALL_FIELDS = ["metric", "value", "numerator", "denominator", "unit", "notes"]

EXTERNAL_NA_METRICS = (
    ("subjective_preference", "requires external human evaluation"),
    ("perceived_correctness_against_expert_causal_explanation", "requires expert annotation"),
    ("human_completeness_satisfaction_trust_detail", "requires user study"),
    ("goal_prediction_accuracy", "requires independent reference labels"),
    ("action_prediction_accuracy", "requires independent reference labels"),
    ("causal_precision", "requires independent ground-truth causal labels"),
    ("causal_recall", "requires independent exhaustive/reference intervention set"),
    ("macro_f1", "requires independent ground-truth labels"),
    ("causal_classification_accuracy", "no independent ground truth in this run"),
)


def csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _rows(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append({k: csv_value(row.get(k)) for k in fields})
    return out


def write_metrics_outputs(
    metrics_dir: Path,
    *,
    document_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
    round_rows: Sequence[Mapping[str, Any]],
    overall_rows: Sequence[Mapping[str, Any]],
    report_markdown: str,
    reference_intervention_rows: Sequence[Mapping[str, Any]] = (),
    effect_discovery_question_rows: Sequence[Mapping[str, Any]] = (),
    concept_change_assessment_rows: Sequence[Mapping[str, Any]] = (),
    concept_change_question_rows: Sequence[Mapping[str, Any]] = (),
    outcome_criteria_rows: Sequence[Mapping[str, Any]] = (),
    saquet_mas_rows: Sequence[Mapping[str, Any]] = (),
    hybrid_add_candidate_rows: Sequence[Mapping[str, Any]] = (),
    hybrid_add_boundary_rows: Sequence[Mapping[str, Any]] = (),
    hybrid_add_source_event_rows: Sequence[Mapping[str, Any]] = (),
    hybrid_add_source_summaries: Sequence[Mapping[str, Any]] = (),
) -> None:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(metrics_dir / "document_metrics.csv", _rows(document_rows, DOCUMENT_FIELDS), fieldnames=DOCUMENT_FIELDS)
    atomic_write_csv(metrics_dir / "interrogation_attempts.csv", _rows(attempt_rows, ATTEMPT_FIELDS), fieldnames=ATTEMPT_FIELDS)
    atomic_write_csv(metrics_dir / "replay_comparisons.csv", _rows(comparison_rows, COMPARISON_FIELDS), fieldnames=COMPARISON_FIELDS)
    atomic_write_csv(metrics_dir / "provenance_edges.csv", _rows(provenance_rows, PROVENANCE_FIELDS), fieldnames=PROVENANCE_FIELDS)
    atomic_write_csv(metrics_dir / "distributions.csv", _rows(distribution_rows, DIST_FIELDS), fieldnames=DIST_FIELDS)
    atomic_write_csv(metrics_dir / "round_metrics.csv", _rows(round_rows, ROUND_FIELDS), fieldnames=ROUND_FIELDS)
    atomic_write_csv(metrics_dir / "overall_metrics.csv", _rows(overall_rows, OVERALL_FIELDS), fieldnames=OVERALL_FIELDS)
    if reference_intervention_rows:
        from mas_langgraph.evaluation.effect_reference import (
            EFFECT_DISCOVERY_QUESTION_FIELDS,
            REFERENCE_INTERVENTION_FIELDS,
        )

        atomic_write_csv(
            metrics_dir / "reference_interventions.csv",
            _rows(reference_intervention_rows, REFERENCE_INTERVENTION_FIELDS),
            fieldnames=REFERENCE_INTERVENTION_FIELDS,
        )
        atomic_write_csv(
            metrics_dir / "effect_discovery_by_question.csv",
            _rows(effect_discovery_question_rows, EFFECT_DISCOVERY_QUESTION_FIELDS),
            fieldnames=EFFECT_DISCOVERY_QUESTION_FIELDS,
        )
    if concept_change_question_rows or concept_change_assessment_rows:
        from mas_langgraph.evaluation.concept_change import (
            CONCEPT_CHANGE_ASSESSMENT_FIELDS,
            CONCEPT_CHANGE_QUESTION_FIELDS,
        )

        if concept_change_question_rows or concept_change_assessment_rows:
            atomic_write_csv(
                metrics_dir / "concept_change_assessments.csv",
                _rows(concept_change_assessment_rows, CONCEPT_CHANGE_ASSESSMENT_FIELDS),
                fieldnames=CONCEPT_CHANGE_ASSESSMENT_FIELDS,
            )
            atomic_write_csv(
                metrics_dir / "concept_change_by_question.csv",
                _rows(concept_change_question_rows, CONCEPT_CHANGE_QUESTION_FIELDS),
                fieldnames=CONCEPT_CHANGE_QUESTION_FIELDS,
            )
    from mas_langgraph.evaluation.outcome_criteria import (
        OUTCOME_CRITERIA_FIELDS,
        filter_outcome_criteria_changed,
    )

    atomic_write_csv(
        metrics_dir / "outcome_criteria_changes.csv",
        _rows(outcome_criteria_rows, OUTCOME_CRITERIA_FIELDS),
        fieldnames=OUTCOME_CRITERIA_FIELDS,
    )
    atomic_write_csv(
        metrics_dir / "outcome_criteria_changed.csv",
        _rows(filter_outcome_criteria_changed(outcome_criteria_rows), OUTCOME_CRITERIA_FIELDS),
        fieldnames=OUTCOME_CRITERIA_FIELDS,
    )
    if saquet_mas_rows:
        from mas_langgraph.evaluation.saquet_eval import SAQUET_MAS_FIELDS

        atomic_write_csv(
            metrics_dir / "saquet_by_mas.csv",
            _rows(saquet_mas_rows, SAQUET_MAS_FIELDS),
            fieldnames=SAQUET_MAS_FIELDS,
        )
    if hybrid_add_candidate_rows or hybrid_add_boundary_rows:
        from mas_langgraph.axis.hybrid_add_diagnostics import (
            HYBRID_ADD_BOUNDARY_FIELDS,
            HYBRID_ADD_CANDIDATE_FIELDS,
        )

        atomic_write_csv(
            metrics_dir / "hybrid_add_candidates.csv",
            _rows(hybrid_add_candidate_rows, HYBRID_ADD_CANDIDATE_FIELDS),
            fieldnames=HYBRID_ADD_CANDIDATE_FIELDS,
        )
        atomic_write_csv(
            metrics_dir / "hybrid_add_boundaries.csv",
            _rows(hybrid_add_boundary_rows, HYBRID_ADD_BOUNDARY_FIELDS),
            fieldnames=HYBRID_ADD_BOUNDARY_FIELDS,
        )
    if hybrid_add_source_event_rows:
        from mas_langgraph.axis.hybrid_add_source import HYBRID_ADD_SOURCE_EVENT_FIELDS

        atomic_write_csv(
            metrics_dir / "hybrid_add_source_events.csv",
            _rows(hybrid_add_source_event_rows, HYBRID_ADD_SOURCE_EVENT_FIELDS),
            fieldnames=HYBRID_ADD_SOURCE_EVENT_FIELDS,
        )
    if hybrid_add_source_summaries:
        from mas_langgraph.batch.io import atomic_write_json

        atomic_write_json(
            metrics_dir / "hybrid_add_source_pools.json",
            list(hybrid_add_source_summaries),
        )
    atomic_write_text(metrics_dir / "report.md", report_markdown if report_markdown.endswith("\n") else report_markdown + "\n")


def overall_metric_rows(
    document_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    pooled: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(metric: str, value: Any, num: Any = None, den: Any = None, unit: str = "", notes: str = "") -> None:
        rows.append(
            {
                "metric": metric,
                "value": value,
                "numerator": num,
                "denominator": den,
                "unit": unit,
                "notes": notes,
            }
        )

    n_docs = len(document_rows)
    add("selected_document_count", n_docs, n_docs, n_docs, "count", "selected LectureBank IDs; failures remain in the denominator")
    for metric, notes in EXTERNAL_NA_METRICS:
        add(metric, NA, NA, NA, "NA", notes)

    if pooled.get("reference_sweep_ran"):
        k = pooled.get("k") or 5
        complete = bool(pooled.get("reference_set_complete"))
        recall_notes = (
            "replay-reference Effect Discovery Recall@K (macro); not causal classification ground truth"
            if complete
            else "replay reference set incomplete (Recall@K requires an exhaustive sweep)"
        )
        add(
            "effect_discovery_recall_at_5" if int(k) == 5 else f"effect_discovery_recall_at_{k}",
            pooled.get("effect_discovery_recall_at_k_macro") if complete else NA,
            pooled.get("effect_discovery_recall_micro_num") if complete else NA,
            pooled.get("effect_discovery_recall_micro_den") if complete else NA,
            "rate" if complete else "NA",
            recall_notes,
        )
        add(
            "reference_benchmark_question_count",
            pooled.get("reference_benchmark_question_count"),
            pooled.get("reference_benchmark_question_count"),
            pooled.get("reference_benchmark_question_count"),
            "count",
            "explained MCQs with a replay-reference sweep",
        )
        add(
            "reference_intervention_count",
            pooled.get("reference_intervention_count"),
            pooled.get("reference_intervention_count"),
            None,
            "count",
            "unique initial hybrid counterfactual interventions",
        )
        add(
            "reference_valid_comparison_count",
            pooled.get("reference_valid_comparison_count"),
            pooled.get("reference_valid_comparison_count"),
            pooled.get("reference_intervention_count"),
            "count",
        )
        add(
            "reference_evaluable_count",
            pooled.get("reference_evaluable_count"),
            pooled.get("reference_evaluable_count"),
            pooled.get("reference_intervention_count"),
            "count",
            "reference interventions with valid_effect or valid_no_effect",
        )
        add(
            "reference_evaluability_rate_micro",
            pooled.get("reference_evaluability_rate_micro"),
            pooled.get("reference_evaluability_micro_num"),
            pooled.get("reference_evaluability_micro_den"),
            "rate",
            "sum(evaluable) / sum(universe) over questions with universe>0; NA not coerced to 0",
        )
        add(
            "reference_evaluability_rate_macro",
            pooled.get("reference_evaluability_rate_macro"),
            pooled.get("reference_evaluability_macro_num"),
            pooled.get("reference_benchmark_question_count"),
            "rate",
            "mean per-question evaluable/universe; questions with universe=0 excluded",
        )
        add(
            "reference_effectful_count",
            pooled.get("reference_effectful_count"),
            pooled.get("reference_effectful_count"),
            pooled.get("reference_intervention_count"),
            "count",
            "replay-supported effectful reference interventions",
        )
        add(
            "causal_evidence_coverage_at_k",
            pooled.get("causal_evidence_coverage_at_k"),
            pooled.get("causal_evidence_coverage_num"),
            pooled.get("causal_evidence_coverage_den"),
            "rate",
            "explained MCQs with ≥1 valid NOOP-stable AXIS counterfactual within K",
        )
        add(
            "effect_discovery_recall_at_k_macro",
            pooled.get("effect_discovery_recall_at_k_macro") if complete else NA,
            pooled.get("effect_discovery_recall_macro_num") if complete else NA,
            pooled.get("reference_benchmark_question_count"),
            "rate" if complete else "NA",
            "mean per-question Recall@K over questions with |E_q|>0; undefined values excluded"
            if complete
            else "incomplete replay-reference sweep; Recall@K is NA",
        )
        add(
            "effect_discovery_recall_at_k_micro",
            pooled.get("effect_discovery_recall_at_k_micro") if complete else NA,
            pooled.get("effect_discovery_recall_micro_num") if complete else NA,
            pooled.get("effect_discovery_recall_micro_den") if complete else NA,
            "rate" if complete else "NA",
        )
        add(
            "effect_precision_at_k_macro",
            pooled.get("effect_precision_at_k_macro"),
            pooled.get("effect_precision_macro_num"),
            pooled.get("reference_benchmark_question_count"),
            "rate",
            "mean defined per-question Precision@K over AXIS valid CFs with evaluable reference labels; undefined values excluded",
        )
        add(
            "effect_precision_at_k_micro",
            pooled.get("effect_precision_at_k_micro"),
            pooled.get("effect_precision_micro_num"),
            pooled.get("effect_precision_micro_den"),
            "rate",
            "sum(V_eval ∩ E) / sum(V_eval); reference replay failures and NOOP-unstable labels excluded",
        )
        add(
            "axis_reference_evaluable_counterfactuals_at_k",
            pooled.get("axis_reference_evaluable_counterfactuals_at_k"),
            pooled.get("axis_reference_evaluable_counterfactuals_at_k"),
            None,
            "count",
            "AXIS valid+NOOP-stable CFs within K that have an evaluable reference label",
        )
        add(
            "axis_reference_unevaluable_counterfactuals_at_k",
            pooled.get("axis_reference_unevaluable_counterfactuals_at_k"),
            pooled.get("axis_reference_unevaluable_counterfactuals_at_k"),
            None,
            "count",
            "AXIS valid+NOOP-stable CFs within K whose reference sweep label is not evaluable",
        )
        add(
            "effect_discovery_end_to_end_yield_at_k",
            pooled.get("effect_discovery_end_to_end_yield_at_k"),
            pooled.get("effect_discovery_end_to_end_num"),
            pooled.get("effect_discovery_end_to_end_den"),
            "rate",
            "secondary; includes failed/inconclusive AXIS executions in the denominator; not Precision",
        )
        add(
            "reference_opportunity_count_match_rate",
            pooled.get("reference_opportunity_count_match_rate"),
            pooled.get("reference_opportunity_match_num"),
            pooled.get("reference_opportunity_match_den"),
            "rate",
            "fraction of MCQs whose reference universe counts match first-attempt advertised opportunities",
        )
    else:
        add(
            "effect_discovery_recall_at_5",
            NA,
            NA,
            NA,
            "NA",
            "no replay reference set"
            if not pooled.get("reference_sweep_ran")
            else "replay reference set incomplete (Recall@K requires an exhaustive sweep)",
        )

    if pooled.get("concept_change_ran"):
        add(
            "concept_change_rate_macro",
            pooled.get("concept_change_rate_macro"),
            pooled.get("concept_change_rate_macro_num"),
            pooled.get("concept_change_question_count"),
            "rate",
            "LLM-assessed concept change; mean of defined per-question rates; not causal Precision/Recall",
        )
        add(
            "concept_change_rate_micro",
            pooled.get("concept_change_rate_micro"),
            pooled.get("concept_change_rate_micro_num"),
            pooled.get("concept_change_rate_micro_den"),
            "rate",
            "changed / assessed; uncertain, missing, failed, and not_evaluable excluded from both",
        )
        add(
            "concept_assessed_count",
            pooled.get("concept_assessed_count"),
            pooled.get("concept_assessed_count"),
            None,
            "count",
            "same_concept + changed_concept among unique AXIS CFs within K",
        )
        add(
            "concept_changed_count",
            pooled.get("concept_changed_count"),
            pooled.get("concept_changed_count"),
            pooled.get("concept_assessed_count"),
            "count",
        )
        add(
            "concept_same_count",
            pooled.get("concept_same_count"),
            pooled.get("concept_same_count"),
            None,
            "count",
        )
        add(
            "concept_uncertain_count",
            pooled.get("concept_uncertain_count"),
            pooled.get("concept_uncertain_count"),
            None,
            "count",
        )
        add(
            "concept_not_evaluable_count",
            pooled.get("concept_not_evaluable_count"),
            pooled.get("concept_not_evaluable_count"),
            None,
            "count",
        )
        add(
            "concept_evaluation_error_count",
            pooled.get("concept_evaluation_error_count"),
            pooled.get("concept_evaluation_error_count"),
            None,
            "count",
        )
    else:
        add(
            "concept_change_rate_macro",
            NA,
            NA,
            NA,
            "NA",
            "not assessed; enable --evaluate-concept-change",
        )
        add(
            "concept_change_rate_micro",
            NA,
            NA,
            NA,
            "NA",
            "not assessed; enable --evaluate-concept-change",
        )

    for key, notes in (
        ("pdf_acquisition_success_rate", "documents whose PDF validated"),
        ("rag_success_rate", "documents whose isolated RAG index succeeded"),
        ("factual_mas_success_rate", "documents whose factual LangGraph MAS ran"),
        ("documents_with_accepted_mcq_rate", "documents producing >=1 accepted MCQ"),
        ("documents_with_valid_axis_replay_rate", "documents producing >=1 valid AXIS replay comparison"),
    ):
        num = pooled.get(f"{key}_num")
        den = pooled.get(f"{key}_den")
        add(key, ratio(num, den), num, den, "rate", notes)

    add("total_accepted_mcqs", pooled.get("total_accepted_mcqs"), pooled.get("total_accepted_mcqs"), n_docs, "count")
    add("total_explained_mcqs", pooled.get("total_explained_mcqs"), pooled.get("total_explained_mcqs"), n_docs, "count")
    add("total_valid_replay_comparisons", pooled.get("total_valid_replay_comparisons"), pooled.get("total_valid_replay_comparisons"), None, "count")

    for name, num_k, den_k, notes in (
        ("intervention_validity_rate", "executed_counterfactual_interventions", "counterfactual_intervention_proposals", "executed CF interventions / REMOVE+ADD+REPLACE proposals; WHAT/DONE excluded"),
        ("valid_comparison_rate", "valid_replay_comparisons", "counterfactual_executions", "valid replay comparisons / counterfactual executions"),
        ("completed_replay_rate", "successful_replay_count", "counterfactual_executions", "successful_replay_count / counterfactual_executions"),
        ("full_noop_stability_rate", "full_noop_stability_true", "full_noop_stability_known", "True / (True+False); missing excluded"),
        ("noop_decision_stability_rate", "noop_decision_stability_true", "noop_decision_stability_known", "True / (True+False); missing excluded"),
        ("noop_criteria_stability_rate", "noop_criteria_stability_true", "noop_criteria_stability_known", "True / (True+False); missing excluded"),
        ("noop_content_stability_rate", "noop_content_stability_true", "noop_content_stability_known", "True / (True+False); missing excluded"),
        ("content_effect_rate", "content_effect_count", "content_effect_known", "control-stable supported content effects"),
        ("criterion_effect_rate", "criterion_effect_count", "criterion_effect_known", "control-stable supported criterion effects"),
        ("decision_effect_rate", "decision_effect_count", "decision_effect_known", "NOOP vs counterfactual; requires decision_control_stable"),
        ("effect_yield_any", "effect_yield_any_count", "effect_yield_any_denominator", "not causal recall; replay-supported any-effect among full-control-stable valid comparisons"),
        ("noop_factual_ledger_hit_rate", "noop_factual_response_hits", "noop_llm_request_count", "pooled counts then rate"),
        ("noop_miss_rate", "noop_llm_request_misses", "noop_llm_request_count", "pooled"),
        ("noop_live_provider_rate", "noop_live_provider_calls", "noop_llm_request_count", "pooled"),
        ("cf_factual_reuse_rate", "counterfactual_factual_response_hits", "counterfactual_llm_request_count", "pooled"),
        ("cf_novel_request_rate", "counterfactual_novel_request_count", "counterfactual_llm_request_count", "pooled"),
        ("cf_memo_hit_rate", "counterfactual_memo_hits", "counterfactual_llm_request_count", "pooled"),
        ("cf_live_provider_rate", "counterfactual_live_provider_calls", "counterfactual_llm_request_count", "pooled"),
        ("provenance_relation_completeness_rate", "provenance_complete_relation_count", "provenance_relation_count", "nonempty relation+artifact_ref+producer+consumer"),
        ("intervention_target_in_slice_rate", "intervention_target_in_slice_count", "intervention_target_in_slice_denominator", "REMOVE/REPLACE only; ADD excluded"),
        ("intervention_target_has_recorded_use_rate", "intervention_target_has_recorded_use_count", "intervention_target_has_recorded_use_denominator", "REMOVE/REPLACE only"),
        ("initial_synthesis_valid_rate", "initial_synthesis_valid_count", "synthesis_case_count", "verbalizer QC; not AXIS causal quality"),
        ("presentation_valid_rate", "presentation_valid_count", "synthesis_case_count", "verbalizer QC"),
    ):
        add(name, ratio(pooled.get(num_k), pooled.get(den_k)), pooled.get(num_k), pooled.get(den_k), "rate", notes)

    for field, label in (
        ("intervention_validity_rate", "cross_document_intervention_validity_rate"),
        ("valid_comparison_rate", "cross_document_valid_comparison_rate"),
        ("full_noop_stability_rate", "cross_document_full_noop_stability_rate"),
        ("content_effect_rate", "cross_document_content_effect_rate"),
        ("criterion_effect_rate", "cross_document_criterion_effect_rate"),
        ("decision_effect_rate", "cross_document_decision_effect_rate"),
        ("noop_factual_ledger_hit_rate", "cross_document_noop_factual_ledger_hit_rate"),
    ):
        stats = describe([r.get(field) for r in document_rows])
        add(f"{label}_mean", stats["mean"], None, stats["n"], "rate", "cross-document execution robustness (descriptive)")
        add(f"{label}_stdev", stats["stdev"], None, stats["n"], "rate", "sample standard deviation")
        add(f"{label}_min", stats["min"], None, stats["n"], "rate", "")
        add(f"{label}_max", stats["max"], None, stats["n"], "rate", "")

    add("interrogation_attempt_count", len(attempt_rows), len(attempt_rows), None, "count")
    add("replay_comparison_count", len(comparison_rows), len(comparison_rows), None, "count")
    add("provenance_edge_count", len(provenance_rows), len(provenance_rows), None, "count")
    return rows


def build_report(
    *,
    catalog_rows: Sequence[Mapping[str, Any]],
    document_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
    round_rows: Sequence[Mapping[str, Any]],
    overall_rows: Sequence[Mapping[str, Any]],
    pooled: Mapping[str, Any],
    quiz_export_enabled: bool = False,
    quiz_row_count: int = 0,
    quiz_unavailable_ids: Sequence[str] = (),
    streamlit_export_enabled: bool = False,
    streamlit_json_count: int = 0,
    streamlit_unavailable_ids: Sequence[str] = (),
    reference_question_rows: Sequence[Mapping[str, Any]] = (),
    reference_intervention_rows: Sequence[Mapping[str, Any]] = (),
    reference_aggregate: Optional[Mapping[str, Any]] = None,
    concept_change_question_rows: Sequence[Mapping[str, Any]] = (),
    outcome_criteria_rows: Sequence[Mapping[str, Any]] = (),
    outcome_criteria_diagnostics: Sequence[Mapping[str, Any]] = (),
    saquet_mas_rows: Sequence[Mapping[str, Any]] = (),
    hybrid_add_candidate_rows: Sequence[Mapping[str, Any]] = (),
    hybrid_add_boundary_rows: Sequence[Mapping[str, Any]] = (),
    hybrid_add_source_event_rows: Sequence[Mapping[str, Any]] = (),
    hybrid_add_source_summaries: Sequence[Mapping[str, Any]] = (),
) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def rate_cell(num: Any, den: Any) -> str:
        return f"{fmt(ratio(num, den))} ({fmt(num)}/{fmt(den)})"

    modes = sorted(
        {
            str(row.get("axis_interrogation_mode") or "").strip()
            for row in document_rows
            if str(row.get("axis_interrogation_mode") or "").strip()
        }
    )
    mode_line = ", ".join(modes) if modes else "constrained (default / not recorded)"

    lines: List[str] = [
        "# LectureBank AXIS evaluation report",
        "",
        "This report aggregates **mechanism-level** metrics from the current AXIS implementation.",
        "It is not a human preference study and does not estimate causal classification accuracy.",
        "",
        f"AXIS interrogation mode: `{mode_line}`.",
    ]
    if quiz_export_enabled:
        lines.extend(
            [
                "",
                "Quiz export:",
                f"- questions.csv: {quiz_row_count} questions",
                "- quizzes.md: generated",
            ]
        )
        if quiz_unavailable_ids:
            lines.append(
                "- unavailable for previously completed documents without persisted "
                "quiz text: " + ", ".join(str(x) for x in quiz_unavailable_ids)
            )
    if streamlit_export_enabled:
        lines.extend(
            [
                "",
                "Streamlit experiment JSON:",
                f"- streamlit/quiz_explanation_experiment_*.json: {streamlit_json_count} file(s)",
            ]
        )
        if streamlit_unavailable_ids:
            lines.append(
                "- unavailable for previously completed documents; rerun with "
                "`--overwrite`: " + ", ".join(str(x) for x in streamlit_unavailable_ids)
            )
    lines.extend(
        [
        "",
        "## Metric interpretation",
        "",
        "- Missing values are **not** treated as False.",
        "- Unknown is **not** treated as no effect.",
        "- NOOP instability makes causal interpretation **inconclusive**.",
        "- Recorded provenance use is **not** causal influence.",
        "- `content_effect` is distinct from `decision_effect`.",
        "- `criterion_effect` is not necessarily an acceptance effect.",
        "- `influential_non_decisive` is a recorded class, not a correctness score.",
        "- Operation distribution is not causal accuracy.",
        "- `effect_yield` is not causal recall.",
        "- Provenance integrity/use is not external provenance correctness.",
        "",
        "## 1. Selected LectureBank documents",
        "",
        "| ID | Title | Topic | Instructor | Venue | Year | PDF source | Status |",
        "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in document_rows:
        lines.append(
            "| {lecturebank_id} | {title} | {topic} | {instructor} | {venue} | {year} | {pdf_source} | {status} |".format(
                lecturebank_id=row.get("lecturebank_id") or "",
                title=_md(row.get("title")),
                topic=_md(row.get("topic_name") or row.get("topic_id")),
                instructor=_md(row.get("instructor")),
                venue=_md(row.get("venue")),
                year=_md(row.get("year")),
                pdf_source=_md(row.get("pdf_source")),
                status=_md(row.get("eval_status") or row.get("failure_stage")),
            )
        )
    lines.extend(
        [
            "",
            "## 2. RAG/MAS execution summary",
            "",
            f"- Selected documents: {len(document_rows)}",
            f"- PDF acquisition success: {rate_cell(pooled.get('pdf_acquisition_success_rate_num'), pooled.get('pdf_acquisition_success_rate_den'))}",
            f"- RAG success: {rate_cell(pooled.get('rag_success_rate_num'), pooled.get('rag_success_rate_den'))}",
            f"- Factual MAS success: {rate_cell(pooled.get('factual_mas_success_rate_num'), pooled.get('factual_mas_success_rate_den'))}",
            f"- Documents with ≥1 accepted MCQ: {rate_cell(pooled.get('documents_with_accepted_mcq_rate_num'), pooled.get('documents_with_accepted_mcq_rate_den'))}",
            f"- Documents with ≥1 valid AXIS replay: {rate_cell(pooled.get('documents_with_valid_axis_replay_rate_num'), pooled.get('documents_with_valid_axis_replay_rate_den'))}",
            f"- Total accepted MCQs: {fmt(pooled.get('total_accepted_mcqs'))}",
            f"- Total explained MCQs: {fmt(pooled.get('total_explained_mcqs'))}",
            f"- Total valid replay comparisons: {fmt(pooled.get('total_valid_replay_comparisons'))}",
            "",
            "Failed selected IDs remain in every denominator. A failed ID is never replaced.",
            "",
            "## 3. Interrogation query distribution",
            "",
            "WHAT is disabled in the generalized AXIS configuration because factual "
            "workflow information is supplied directly by the execution trace and "
            "provenance model. Interrogation rounds therefore correspond to "
            "counterfactual interventions only.",
            "",
            "Active operations: REMOVE, ADD, REPLACE/WHATIF. DONE is reported separately "
            "and **excluded** from add/remove/whatif percentages.",
            "Canonical mapping: `WHATIF` → `REPLACE`. Figure-4 analogue: REPLACE → `whatif`.",
            "Opportunity counts (`available_remove_count` / `available_add_count` / "
            "`available_whatif_count`) are researcher-only and describe the advertised "
            "query space, not causal relevance. `available_what_count` is retained as 0 "
            "for schema compatibility; WHAT is not an active opportunity.",
            "",
            "| Operation | Proposed count | Proposed % of recognized non-DONE | Executed count | Executed % of recognized non-DONE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for op, pcount, ecount in (
        ("REMOVE", "proposed_remove_count", "executed_remove_count"),
        ("ADD", "proposed_add_count", "executed_add_count"),
        ("REPLACE/WHATIF", "proposed_replace_count", "executed_replace_count"),
    ):
        lines.append(
            f"| {op} | {fmt(pooled.get(pcount))} | {fmt(pct(pooled.get(pcount), pooled.get('proposed_recognized_non_done_count')))} "
            f"({fmt(pooled.get(pcount))}/{fmt(pooled.get('proposed_recognized_non_done_count'))}) | "
            f"{fmt(pooled.get(ecount))} | {fmt(pct(pooled.get(ecount), pooled.get('executed_recognized_non_done_count')))} "
            f"({fmt(pooled.get(ecount))}/{fmt(pooled.get('executed_recognized_non_done_count'))}) |"
        )
    lines.extend(
        [
            "",
            f"- DONE proposed: {fmt(pooled.get('proposed_done_count'))}; executed: {fmt(pooled.get('executed_done_count'))}",
            f"- UNPARSED proposed: {fmt(pooled.get('proposed_unparsed_count'))}",
            f"- Counterfactual intervention proposals (REMOVE/ADD/REPLACE): {fmt(pooled.get('counterfactual_intervention_proposals'))}",
            f"- Executed counterfactual interventions: {fmt(pooled.get('executed_counterfactual_interventions'))}",
            f"- Intervention validity rate: {rate_cell(pooled.get('executed_counterfactual_interventions'), pooled.get('counterfactual_intervention_proposals'))}",
            "",
        ]
    )
    hist_what = pooled.get("proposed_what_count")
    if hist_what not in (None, 0, "NA", "0"):
        lines.extend(
            [
                "Historical WHAT rows (not an active operation in this configuration): "
                f"proposed={fmt(pooled.get('proposed_what_count'))}, "
                f"executed={fmt(pooled.get('executed_what_count'))}.",
                "",
            ]
        )
    lines.extend(
        [
            "Validation failure distribution (not merged):",
            "",
            f"- parser_error: {fmt(pooled.get('validation_parser_error_count'))}",
            f"- resolution_error: {fmt(pooled.get('validation_resolution_error_count'))}",
            f"- duplicate_error: {fmt(pooled.get('validation_duplicate_error_count'))}",
            f"- contract_error: {fmt(pooled.get('validation_contract_error_count'))}",
            f"- execution_error: {fmt(pooled.get('validation_execution_error_count'))}",
            f"- not_currently_executable: {fmt(pooled.get('validation_not_currently_executable_count'))}",
            f"- other validation statuses: {fmt(pooled.get('validation_other_status_count'))}",
            "",
            "## 4. Checkpoint position",
            "",
            "`checkpoint_position_pct = 100 * checkpoint_ordinal_1based / checkpoint_total`",
            "(8th of 10 = 80%). `c_0` is the first checkpoint.",
            "",
            "`checkpoint_progress_pct = 100 * zero_based_position / (checkpoint_total - 1)`",
            "(first = 0%, last = 100%; single checkpoint = NA).",
            "",
            f"- Mean position %: {fmt(pooled.get('checkpoint_position_pct_mean'))} (n={fmt(pooled.get('checkpoint_position_n'))})",
            f"- Median / min / max position %: {fmt(pooled.get('checkpoint_position_pct_median'))} / {fmt(pooled.get('checkpoint_position_pct_min'))} / {fmt(pooled.get('checkpoint_position_pct_max'))}",
            f"- Mean progress %: {fmt(pooled.get('checkpoint_progress_pct_mean'))} (n={fmt(pooled.get('checkpoint_progress_n'))})",
            f"- Median / min / max progress %: {fmt(pooled.get('checkpoint_progress_pct_median'))} / {fmt(pooled.get('checkpoint_progress_pct_min'))} / {fmt(pooled.get('checkpoint_progress_pct_max'))}",
            "",
            "See `replay_comparisons.csv` for per-intervention values and per-operation / per-artifact / per-document splits.",
            "",
            "## 5. Replay validity and NOOP stability",
            "",
            f"- Counterfactual executions: {fmt(pooled.get('counterfactual_executions'))}",
            f"- Successful replay count: {fmt(pooled.get('successful_replay_count'))}",
            f"- Valid replay comparisons: {fmt(pooled.get('valid_replay_comparisons'))}",
            f"- Completed replay rate: {rate_cell(pooled.get('successful_replay_count'), pooled.get('counterfactual_executions'))}",
            f"- Valid comparison rate: {rate_cell(pooled.get('valid_replay_comparisons'), pooled.get('counterfactual_executions'))}",
            f"- NOOP decision stability: {rate_cell(pooled.get('noop_decision_stability_true'), pooled.get('noop_decision_stability_known'))} (unavailable={fmt(pooled.get('noop_decision_stability_unavailable'))})",
            f"- NOOP criteria stability: {rate_cell(pooled.get('noop_criteria_stability_true'), pooled.get('noop_criteria_stability_known'))} (unavailable={fmt(pooled.get('noop_criteria_stability_unavailable'))})",
            f"- NOOP content stability: {rate_cell(pooled.get('noop_content_stability_true'), pooled.get('noop_content_stability_known'))} (unavailable={fmt(pooled.get('noop_content_stability_unavailable'))})",
            f"- Full NOOP stability: {rate_cell(pooled.get('full_noop_stability_true'), pooled.get('full_noop_stability_known'))} (unavailable={fmt(pooled.get('full_noop_stability_unavailable'))})",
            f"- Replay completion modes: target_observable={fmt(pooled.get('replay_completion_mode_target_observable'))}, graph_end={fmt(pooled.get('replay_completion_mode_graph_end'))}, fallback/full={fmt(pooled.get('replay_completion_mode_fallback_full'))}, error/unknown={fmt(pooled.get('replay_completion_mode_error_unknown'))}",
            f"- Executed node count mean/median: {fmt(pooled.get('executed_node_count_mean'))} / {fmt(pooled.get('executed_node_count_median'))}",
            "",
            "## 6. Intervention effects",
            "",
            "Intervention-effect comparisons use **NOOP → counterfactual**. Factual → counterfactual is not treated as an intervention effect when NOOP is unstable.",
            "",
            f"- Decision effect (supported): {rate_cell(pooled.get('decision_effect_count'), pooled.get('decision_effect_known'))}; raw observed {rate_cell(pooled.get('decision_effect_count_raw'), pooled.get('decision_effect_raw_known'))}",
            f"- Criterion effect (supported): {rate_cell(pooled.get('criterion_effect_count'), pooled.get('criterion_effect_known'))}; raw {rate_cell(pooled.get('criterion_effect_count_raw'), pooled.get('criterion_effect_raw_known'))}",
            f"- Decision-relevant criterion effect (supported): {rate_cell(pooled.get('decision_relevant_criterion_effect_count'), pooled.get('decision_relevant_criterion_effect_known'))}",
            f"- Content effect (supported): {rate_cell(pooled.get('content_effect_count'), pooled.get('content_effect_known'))}; raw {rate_cell(pooled.get('content_effect_count_raw'), pooled.get('content_effect_raw_known'))}",
            f"- Stem / correct / distractor / explanation change rates: {fmt(ratio(pooled.get('stem_change_count'), pooled.get('stem_change_known')))} / {fmt(ratio(pooled.get('correct_answer_change_count'), pooled.get('correct_answer_change_known')))} / {fmt(ratio(pooled.get('distractor_change_count'), pooled.get('distractor_change_known')))} / {fmt(ratio(pooled.get('explanation_change_count'), pooled.get('explanation_change_known')))}",
            f"- Question not reproduced: {rate_cell(pooled.get('question_not_reproduced_count'), pooled.get('question_not_reproduced_den'))}",
            f"- Effect yield (any supported effect / full-control-stable valid comparisons): {rate_cell(pooled.get('effect_yield_any_count'), pooled.get('effect_yield_any_denominator'))}",
            "",
            (
                "`effect_yield` is **not** causal recall. Effect Discovery Recall@K below uses the exhaustive replay-reference set."
                if reference_aggregate
                else "`effect_yield` is **not** causal recall. Effect Discovery Recall@K is NA without a replay-reference intervention set (`--build-effect-reference-set`)."
            ),
            "",
            "Existing `content_effect` and stem/correct/distractor/explanation change rates detect **normalized text differences**. "
            "The broad content-effect flag can include question non-reproduction. "
            "They do not measure whether the tested knowledge changed.",
            "",
            "## Outcome-criteria changes",
            "",
            "`outcome_criteria_changes.csv` is the complete export: one stored outcome criterion on one AXIS replay comparison, including unchanged and unevaluable criteria. "
            "Values come from recorded `criterion_deltas`, not from reconstructing the question text. "
            "`outcome_criteria_changed.csv` is a derived subset of that file: supported NOOP→counterfactual changes whose counterfactual value is available. "
            "It does not necessarily indicate a change in the final acceptance decision.",
            "",
            "- **Factual → NOOP** checks control stability (`noop_differs_from_factual`).",
            "- **NOOP → counterfactual** records the intervention difference (`counterfactual_differs_from_noop`).",
            "- A criterion change does not necessarily change the final acceptance decision.",
            "- `decision_relevant` means the criterion participates in decision logic, not that its observed change necessarily changed the decision.",
            "",
            "`criterion_change_supported` is the stored NOOP→counterfactual difference only when the comparison is valid and the full NOOP control is stable; otherwise it is NA. "
            "Raw differences are still exported. Missing historical fields are not treated as unchanged.",
            "",
            "Rates below are **per criterion**. They must not be summed across criteria and read as a count of distinct interventions. "
            "The denominator is valid, full-control-stable comparisons with a known change indicator for that criterion.",
            "",
            "| Criterion | Supported changes | Evaluable comparisons | Supported change rate |",
            "|---|---:|---:|---:|",
        ]
    )
    from mas_langgraph.evaluation.outcome_criteria import aggregate_outcome_criteria

    for row in aggregate_outcome_criteria(outcome_criteria_rows):
        den = row["evaluable_comparisons"]
        rate = "NA" if not den else rate_cell(row["supported_changes"], den)
        lines.append(
            f"| {row['criterion_name']} | {fmt(row['supported_changes'])} | "
            f"{fmt(den)} | {rate} |"
        )
    if not outcome_criteria_rows:
        lines.append("| — | NA | NA | NA |")
    if outcome_criteria_diagnostics:
        lines.extend(
            [
                "",
                f"- Mapping/value conflicts: {len(outcome_criteria_diagnostics)} "
                "(see diagnostics; conflicting cells are NA rather than silently chosen).",
            ]
        )
    lines.extend(
        [
            "",
            "## SAQUET item-writing flaws",
            "",
            "Optional SAQUET IWF evaluation of **final accepted factual MCQs**. "
            "Criterion functions, thresholds, prompts, and the original True→0 / False→1 conversion are unchanged. "
            "`saquet_by_mas.csv` is one aggregated row per document/MAS run. "
            "Question-level flags are stored only in the internal `saquet_question_cache.json` and are not a CSV deliverable.",
            "",
            "Overall IWF totals and zero-IWF rates use **completely evaluated** MCQs only, "
            "and only when the factual quiz is complete. "
            "Each criterion’s flag rate uses questions with a valid 0/1 result for that criterion "
            "when the quiz is complete; otherwise whole-MAS rates are NA. "
            "Failures and missing results are not treated as “no flaw.” Zero denominators are NA. "
            "SAQUET evaluates the complete final accepted factual quiz, not the AXIS explanation subset.",
            "",
        ]
    )
    if saquet_mas_rows:
        from mas_langgraph.evaluation.saquet_eval import SAQUET_CRITERIA, saquet_criterion_fields

        def _as_int(value: Any) -> Any:
            if value in (None, "", "NA"):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        accepted_vals = [_as_int(r.get("final_accepted_mcq_count")) for r in saquet_mas_rows]
        accepted = sum(v for v in accepted_vals if v is not None)
        fully = sum(int(r.get("fully_evaluated_mcq_count") or 0) for r in saquet_mas_rows)
        incomplete_n = sum(
            1
            for r in saquet_mas_rows
            if str(r.get("quiz_completeness") or "") in {"incomplete", "unknown"}
        )
        lines.extend(
            [
                f"- Documents: {len(saquet_mas_rows)}",
                f"- Final accepted MCQs: {fmt(accepted) if all(v is not None for v in accepted_vals) else 'NA'}",
                f"- Fully evaluated MCQs: {fmt(fully)}",
                f"- Evaluation completion rate: {rate_cell(fully, accepted) if accepted else 'NA'}",
                f"- Incomplete/unknown factual quizzes: {incomplete_n}",
                "",
                "| Criterion | Flagged | Evaluated | Flag rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for name in SAQUET_CRITERIA:
            flagged_f, eval_f, _rate_f = saquet_criterion_fields(name)
            flagged = sum(int(r.get(flagged_f) or 0) for r in saquet_mas_rows)
            evaluated = sum(int(r.get(eval_f) or 0) for r in saquet_mas_rows)
            rate = "NA" if (not evaluated or incomplete_n) else rate_cell(flagged, evaluated)
            lines.append(f"| {name} | {fmt(flagged)} | {fmt(evaluated)} | {rate} |")
    else:
        lines.append("SAQUET was not run in this evaluation (`--evaluate-saquet` / `--saquet-from-saved`).")
    lines.extend(
        [
            "",
            "## LLM-assessed tested-concept change",
            "",
            "This is an optional **LLM-assessed** comparison of the tested knowledge in the NOOP MCQ versus the corresponding counterfactual MCQ. "
            "It is not causal ground truth, not validated human judgment, and not semantic Precision@K or Recall@K.",
            "",
            "A comparison is assessed only when the replay is valid, NOOP-stable, and both corresponding questions are available with sufficient content. "
            "Missing or non-reproduced questions are `not_evaluable`, never `changed_concept` or `same_concept`. "
            "`same_concept` does not imply that all other semantic aspects of the question are unchanged.",
            "",
        ]
    )
    if pooled.get("concept_change_ran"):
        k = pooled.get("k") or 5
        lines.extend(
            [
                f"- Questions: {fmt(pooled.get('concept_change_question_count') or len(concept_change_question_rows or []))}",
                f"- AXIS-selected unique CFs within K={k}: {fmt(pooled.get('axis_counterfactual_selected_at_k'))}",
            ]
        )
        lines.extend(
            [
                f"- Assessed (changed + same): {fmt(pooled.get('concept_assessed_count'))}",
                f"- Changed concept: {fmt(pooled.get('concept_changed_count'))}",
                f"- Same concept: {fmt(pooled.get('concept_same_count'))}",
                f"- Uncertain: {fmt(pooled.get('concept_uncertain_count'))}",
                f"- Not evaluable: {fmt(pooled.get('concept_not_evaluable_count'))}",
                f"- Evaluation errors: {fmt(pooled.get('concept_evaluation_error_count'))}",
                f"- Macro concept-change rate: {fmt(pooled.get('concept_change_rate_macro'))}",
                f"- Micro concept-change rate: {rate_cell(pooled.get('concept_change_rate_micro_num'), pooled.get('concept_change_rate_micro_den'))}",
                "",
                "The rate is **conditional on assessable comparisons**. Uncertain, missing, failed, and not-evaluable cases are excluded rather than treated as unchanged. "
                "A high conditional rate can coexist with frequent non-reproduction; inspect the excluded counts.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Not assessed in this run (`--evaluate-concept-change` was off, or no assessments were persisted). Historical files without this field are NA, not zero.",
                "",
            ]
        )
    lines.extend(
        [
            "## 7. Causal classification distribution",
            "",
            "This is a **distribution of recorded labels**, not accuracy.",
            "",
        ]
    )
    lines.extend(_dist_table(distribution_rows, "causal_classification"))
    lines.extend(
        [
            "",
            "## 8. Information-role distribution",
            "",
        ]
    )
    lines.extend(_dist_table(distribution_rows, "information_role"))
    lines.extend(
        [
            "",
            "## 9. Intervention target artifact distribution",
            "",
        ]
    )
    lines.extend(_dist_table(distribution_rows, "target_artifact_type"))
    lines.extend(
        [
            "",
            f"- EvidenceChunk: {fmt(pooled.get('target_EvidenceChunk_count'))}",
            f"- KnowledgeAtom: {fmt(pooled.get('target_KnowledgeAtom_count'))}",
            f"- MCQBlueprint: {fmt(pooled.get('target_MCQBlueprint_count'))}",
            f"- CandidateMCQ: {fmt(pooled.get('target_CandidateMCQ_count'))}",
            f"- other: {fmt(pooled.get('target_other_count'))}",
            "",
            "## 10. Provenance integrity/use",
            "",
            "Not called provenance correctness: there is no external ground-truth graph here.",
            "",
            f"- Provenance relations: {fmt(pooled.get('provenance_relation_count'))}",
            f"- Unique relevant artifacts: {fmt(pooled.get('unique_relevant_artifact_count'))}",
            f"- Completeness rate: {rate_cell(pooled.get('provenance_complete_relation_count'), pooled.get('provenance_relation_count'))}",
            f"- REMOVE/REPLACE target in factual slice: {rate_cell(pooled.get('intervention_target_in_slice_count'), pooled.get('intervention_target_in_slice_denominator'))}",
            f"- REMOVE/REPLACE target has recorded use: {rate_cell(pooled.get('intervention_target_has_recorded_use_count'), pooled.get('intervention_target_has_recorded_use_denominator'))}",
            f"- Provenance slicing errors: {fmt(pooled.get('provenance_slicing_error_count'))}",
            "",
            "ADD targets are not required to already exist in the factual slice.",
            "",
            "## 11. Replay determinism",
            "",
            "Rates are **micro-averaged**: counts are summed across documents first.",
            "",
            f"- NOOP factual ledger hit rate: {rate_cell(pooled.get('noop_factual_response_hits'), pooled.get('noop_llm_request_count'))}",
            f"- NOOP miss rate: {rate_cell(pooled.get('noop_llm_request_misses'), pooled.get('noop_llm_request_count'))}",
            f"- NOOP live provider rate: {rate_cell(pooled.get('noop_live_provider_calls'), pooled.get('noop_llm_request_count'))}",
            f"- CF factual reuse rate: {rate_cell(pooled.get('counterfactual_factual_response_hits'), pooled.get('counterfactual_llm_request_count'))}",
            f"- CF novel request rate: {rate_cell(pooled.get('counterfactual_novel_request_count'), pooled.get('counterfactual_llm_request_count'))}",
            f"- CF memo hit rate: {rate_cell(pooled.get('counterfactual_memo_hits'), pooled.get('counterfactual_llm_request_count'))}",
            f"- CF live provider rate: {rate_cell(pooled.get('counterfactual_live_provider_calls'), pooled.get('counterfactual_llm_request_count'))}",
            "",
            "## 12. Performance over rounds",
            "",
            "Process-level analogue of original AXIS round analysis. This does **not** claim preference or correctness over rounds.",
            "",
            "| Round | Comparisons | Valid rate | Full NOOP stability | Content effect | Criterion effect | Decision effect | Mean checkpoint position % |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    overall_rounds: Dict[int, List[Mapping[str, Any]]] = {}
    for row in round_rows:
        try:
            rnd = int(row.get("round"))
        except (TypeError, ValueError):
            continue
        overall_rounds.setdefault(rnd, []).append(row)
    # Prefer already-aggregated overall rows (lecturebank_id empty or ALL)
    shown = [r for r in round_rows if str(r.get("lecturebank_id") or "") in {"", "ALL", "overall"}]
    if not shown:
        shown = []
        for rnd, group in sorted(overall_rounds.items()):
            shown.append(
                {
                    "round": rnd,
                    "comparison_count": sum(int(g.get("comparison_count") or 0) for g in group),
                    "valid_comparison_count": sum(int(g.get("valid_comparison_count") or 0) for g in group),
                    "valid_comparison_rate": ratio(
                        sum(int(g.get("valid_comparison_count") or 0) for g in group),
                        sum(int(g.get("comparison_count") or 0) for g in group),
                    ),
                    "full_noop_stability_rate": _mean_optional([g.get("full_noop_stability_rate") for g in group]),
                    "content_effect_rate": _mean_optional([g.get("content_effect_rate") for g in group]),
                    "criterion_effect_rate": _mean_optional([g.get("criterion_effect_rate") for g in group]),
                    "decision_effect_rate": _mean_optional([g.get("decision_effect_rate") for g in group]),
                    "mean_checkpoint_position_pct": _mean_optional([g.get("mean_checkpoint_position_pct") for g in group]),
                }
            )
    for row in shown:
        lines.append(
            "| {round} | {cmp} | {valid} | {noop} | {content} | {crit} | {dec} | {ckpt} |".format(
                round=row.get("round"),
                cmp=fmt(row.get("comparison_count")),
                valid=fmt(row.get("valid_comparison_rate")),
                noop=fmt(row.get("full_noop_stability_rate")),
                content=fmt(row.get("content_effect_rate")),
                crit=fmt(row.get("criterion_effect_rate")),
                dec=fmt(row.get("decision_effect_rate")),
                ckpt=fmt(row.get("mean_checkpoint_position_pct")),
            )
        )
    lines.extend(
        [
            "",
            "## 13. Cross-document execution robustness",
            "",
            "Descriptive robustness across the selected documents, **not** general model/hyperparameter robustness.",
            "",
        ]
    )
    for field, label in (
        ("intervention_validity_rate", "intervention validity"),
        ("valid_comparison_rate", "valid comparison"),
        ("full_noop_stability_rate", "full NOOP stability"),
        ("content_effect_rate", "content effect"),
        ("criterion_effect_rate", "criterion effect"),
        ("decision_effect_rate", "decision effect"),
        ("noop_factual_ledger_hit_rate", "NOOP factual ledger hit"),
    ):
        stats = describe([r.get(field) for r in document_rows])
        lines.append(
            f"- {label}: mean={fmt(stats['mean'])}, sd={fmt(stats['stdev'])}, "
            f"min={fmt(stats['min'])}, max={fmt(stats['max'])} (n={stats['n']})"
        )
    lines.extend(
        [
            "",
            "## 14. Runtime",
            "",
            "Secondary implementation metrics. Wall-clock sums are not mixed with percentages.",
            "",
            f"- Factual workflow seconds (sum): {fmt(pooled.get('factual_workflow_s_sum'))}; mean {fmt(pooled.get('factual_workflow_s_mean'))}",
            f"- Experiment total seconds (sum): {fmt(pooled.get('experiment_total_s_sum'))}",
            f"- AXIS total seconds (sum): {fmt(pooled.get('axis_total_s_sum'))}",
            f"- Interrogation / NOOP / CF replay / comparison seconds (sum): {fmt(pooled.get('interrogation_total_s_sum'))} / {fmt(pooled.get('noop_total_s_sum'))} / {fmt(pooled.get('counterfactual_replay_total_s_sum'))} / {fmt(pooled.get('comparison_total_s_sum'))}",
            "",
            "AXIS replay vs explanation-LLM percentages are taken from recorded timing totals and are not re-derived from summed wall clocks.",
            "",
            "## 15. Synthesis/verbalizer quality control",
            "",
            "These are **not** AXIS causal-mechanism quality.",
            "",
            f"- Initial synthesis valid rate: {rate_cell(pooled.get('initial_synthesis_valid_count'), pooled.get('synthesis_case_count'))}",
            f"- Repair attempt rate: {rate_cell(pooled.get('repair_attempt_count'), pooled.get('synthesis_case_count'))}",
            f"- Repair success rate: {rate_cell(pooled.get('repair_success_count'), pooled.get('repair_attempt_count'))}",
            f"- Fallback rate: {rate_cell(pooled.get('fallback_count'), pooled.get('synthesis_case_count'))}",
            f"- Presentation valid rate: {rate_cell(pooled.get('presentation_valid_count'), pooled.get('synthesis_case_count'))}",
            f"- Cases with faithfulness warning: {rate_cell(pooled.get('cases_with_faithfulness_warning_count'), pooled.get('synthesis_case_count'))}",
            f"- Participant word count mean/median: {fmt(pooled.get('participant_word_count_mean'))} / {fmt(pooled.get('participant_word_count_median'))}",
            f"- Participant sentence count mean/median: {fmt(pooled.get('participant_sentence_count_mean'))} / {fmt(pooled.get('participant_sentence_count_median'))}",
            "",
        ]
    )
    if reference_aggregate:
        k = reference_aggregate.get("k") or 5
        mismatch_rows = [
            r
            for r in reference_question_rows
            if r.get("reference_opportunity_count_match") is False
        ]
        missing_axis = [
            r
            for r in reference_question_rows
            if r.get("axis_selected_signature_missing_from_reference") is True
        ]
        lines.extend(
            [
                "## Effect discovery reference benchmark",
                "",
                "The reference set exhaustively evaluates the finite counterfactual "
                "intervention opportunities exposed by the initial hybrid vocabulary. "
                "It is a replay-derived reference set, not external causal ground truth.",
                "",
                f"- MCQs benchmarked: {fmt(reference_aggregate.get('reference_benchmark_question_count'))}",
                f"- Total reference intervention opportunities: {fmt(reference_aggregate.get('reference_intervention_count'))}",
                f"- REMOVE / WHATIF / ADD: {fmt(reference_aggregate.get('reference_remove_count'))} / {fmt(reference_aggregate.get('reference_whatif_count'))} / {fmt(reference_aggregate.get('reference_add_count'))}",
                f"- Valid comparison count: {fmt(reference_aggregate.get('reference_valid_comparison_count'))}",
                f"- Inconclusive (NOOP-unstable) count: {fmt(reference_aggregate.get('reference_inconclusive_count'))}",
                f"- Replay failure count: {fmt(reference_aggregate.get('reference_replay_failed_count'))}",
                f"- Effectful reference interventions: {fmt(reference_aggregate.get('reference_effectful_count'))}",
                f"- Valid no-effect interventions: {fmt(reference_aggregate.get('reference_no_effect_count'))}",
                f"- Reference evaluable interventions: {fmt(reference_aggregate.get('reference_evaluable_count'))} / {fmt(reference_aggregate.get('reference_intervention_count'))}",
                f"- Reference Evaluability Rate: {rate_cell(reference_aggregate.get('reference_evaluability_micro_num'), reference_aggregate.get('reference_evaluability_micro_den'))}",
                f"- Macro Reference Evaluability Rate: {fmt(reference_aggregate.get('reference_evaluability_rate_macro'))}",
                f"- Micro Reference Evaluability Rate: {rate_cell(reference_aggregate.get('reference_evaluability_micro_num'), reference_aggregate.get('reference_evaluability_micro_den'))}",
                "",
                "Reference evaluability is the proportion of the enumerated intervention "
                "universe for which the reference sweep produced a valid, NOOP-stable "
                "effect/no-effect label. A complete sweep can still have replay failures "
                "or NOOP-instability; reference_set_complete does not imply full evaluability.",
                "",
                f"- Causal Evidence Coverage@{k}: {rate_cell(reference_aggregate.get('causal_evidence_coverage_num'), reference_aggregate.get('causal_evidence_coverage_den'))}",
                f"- Macro Effect Discovery Recall@{k}: {fmt(reference_aggregate.get('effect_discovery_recall_at_k_macro'))}",
                f"- Micro Effect Discovery Recall@{k}: {rate_cell(reference_aggregate.get('effect_discovery_recall_micro_num'), reference_aggregate.get('effect_discovery_recall_micro_den'))}",
                f"- Macro Effect Precision@{k}: {fmt(reference_aggregate.get('effect_precision_at_k_macro'))}",
                f"- Micro Effect Precision@{k}: {rate_cell(reference_aggregate.get('effect_precision_micro_num'), reference_aggregate.get('effect_precision_micro_den'))}",
                f"- AXIS-selected valid CFs with evaluable reference labels: {fmt(reference_aggregate.get('axis_reference_evaluable_counterfactuals_at_k'))}",
                f"- AXIS-selected valid CFs without evaluable reference labels: {fmt(reference_aggregate.get('axis_reference_unevaluable_counterfactuals_at_k'))}",
                f"- End-to-end discovery yield@{k}: {rate_cell(reference_aggregate.get('effect_discovery_end_to_end_num'), reference_aggregate.get('effect_discovery_end_to_end_den'))} (not Precision)",
                "",
                "Effect Precision@K is calculated only over AXIS-selected interventions that "
                "have an evaluable reference label. Reference replay failures and NOOP-unstable "
                "reference comparisons are excluded rather than treated as no effect.",
                "",
                f"- MCQs where Recall@{k} is NA because reference_effectful_count = 0: {fmt(reference_aggregate.get('questions_with_empty_effectful_reference'))}",
                f"- Incomplete reference sets: {fmt(reference_aggregate.get('questions_with_incomplete_reference_set'))}",
                f"- Opportunity-count match rate vs first hybrid attempt: {rate_cell(reference_aggregate.get('reference_opportunity_match_num'), reference_aggregate.get('reference_opportunity_match_den'))}",
                "",
            ]
        )
        if mismatch_rows:
            lines.append("Opportunity-count mismatches (not silently equated):")
            for row in mismatch_rows:
                lines.append(
                    f"- {row.get('lecturebank_id')} {row.get('candidate_ref')}: "
                    f"{row.get('opportunity_mismatch_detail') or 'counts differ'}"
                )
            lines.append("")
        if missing_axis:
            lines.append(
                "**AXIS-selected hybrid interventions missing from the initial "
                "replay-reference universe:**"
            )
            for row in missing_axis:
                lines.append(f"- {row.get('lecturebank_id')} {row.get('candidate_ref')}")
            lines.append("")
        _ = reference_intervention_rows
    if hybrid_add_candidate_rows or hybrid_add_boundary_rows:
        from mas_langgraph.axis.hybrid_add_diagnostics import (
            format_hybrid_add_report_section,
            summarize_hybrid_add_diagnostics,
        )

        summary = summarize_hybrid_add_diagnostics(
            hybrid_add_candidate_rows,
            hybrid_add_boundary_rows,
        )
        lines.extend(format_hybrid_add_report_section(summary))
        lines.append("")
    if hybrid_add_source_summaries:
        from mas_langgraph.axis.hybrid_add_source import (
            format_hybrid_add_source_report_section,
        )

        for src in hybrid_add_source_summaries:
            if isinstance(src, Mapping) and src:
                lines.extend(format_hybrid_add_source_report_section(src))
        if hybrid_add_source_event_rows:
            lines.append(
                f"- Source-prep event rows: {len(hybrid_add_source_event_rows)}"
            )
            lines.append("")
    lines.extend(
        [
            "## 16. Metrics requiring external human/reference evaluation",
            "",
            "The following **cannot** be obtained automatically from these LectureBank runs alone:",
            "",
            "- subjective preference",
            "- perceived correctness against an expert causal explanation",
            "- human completeness / satisfaction / trust / detail scores",
            "- goal prediction accuracy",
            "- action prediction accuracy",
            "- causal precision",
            "- causal recall",
            "- Macro-F1",
            "- causal classification accuracy",
            "",
            "Causal precision / recall / Macro-F1 / causal classification accuracy still require independent causal labels and remain NA.",
            (
                "Effect Discovery Recall@K is reported from the exhaustive replay-reference set above."
                if reference_aggregate and reference_aggregate.get("reference_set_complete")
                else "Effect Discovery Recall@K is NA without a complete exhaustive replay-reference set (`--build-effect-reference-set`)."
            ),
            "",
            "They are recorded as NA / `requires external reference set` in `overall_metrics.csv` except Effect Discovery Recall@K when a complete replay-reference sweep exists.",
            "",
        ]
    )
    _ = catalog_rows, overall_rows
    return "\n".join(lines) + "\n"


def _dist_table(rows: Sequence[Mapping[str, Any]], dimension: str) -> List[str]:
    overall = [
        r
        for r in rows
        if r.get("dimension") == dimension and str(r.get("scope") or "") in {"overall", "ALL", ""}
        or (r.get("dimension") == dimension and str(r.get("lecturebank_id") or "") in {"", "ALL", "overall"} and r.get("scope") == "overall")
    ]
    if not overall:
        overall = [r for r in rows if r.get("dimension") == dimension and str(r.get("scope")) == "overall"]
    if not overall:
        # aggregate document-scope rows
        from collections import Counter

        counts: Counter = Counter()
        for r in rows:
            if r.get("dimension") != dimension:
                continue
            if str(r.get("scope")) not in {"document", "overall"}:
                continue
            if str(r.get("scope")) == "document":
                counts[str(r.get("value"))] += int(r.get("count") or 0)
        denom = sum(counts.values())
        overall = [
            {"value": k, "count": v, "denominator": denom, "percentage": pct(v, denom)}
            for k, v in counts.items()
        ]
    lines = [
        "| Value | Count | Denominator | Percentage |",
        "|---|---:|---:|---:|",
    ]
    for r in sorted(overall, key=lambda x: str(x.get("value") or "")):
        lines.append(
            f"| {r.get('value')} | {r.get('count')} | {r.get('denominator')} | {r.get('percentage') if r.get('percentage') is not None else 'NA'} |"
        )
    if len(lines) == 2:
        lines.append("| *(none recorded)* | 0 | 0 | NA |")
    return lines


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|")


def _mean_optional(values: Sequence[Any]) -> Optional[float]:
    nums = []
    for v in values:
        if v is None or v == "NA":
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return None
    return sum(nums) / len(nums)
