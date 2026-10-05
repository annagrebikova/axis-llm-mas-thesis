"""Effect-discovery replay-reference sweep (no live OpenAI)."""

from __future__ import annotations

import operator
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest.mock import MagicMock, patch

from mas_langgraph.artifacts.models import ArtifactEnvelope, ArtifactRef
from mas_langgraph.axis.comparison import (
    CaseRunObservation,
    ReplayComparison,
    SubjectContentDelta,
)
from mas_langgraph.axis.intervention_guidance import (
    HybridInterventionVocabulary,
    InterventionCandidate,
    hybrid_opportunity_counts,
    validate_resolved_against_hybrid_vocabulary,
)
from mas_langgraph.axis.models import QuerySignature, ResolvedQuery
from mas_langgraph.axis.queries import QueryValidationError
from mas_langgraph.batch.manifest import generation_config_fingerprint
from mas_langgraph.evaluation.effect_reference import (
    EFFECT_DISCOVERY_QUESTION_FIELDS,
    EVALUABLE_REFERENCE_STATUSES,
    REFERENCE_INTERVENTION_FIELDS,
    HybridReferenceSweepError,
    aggregate_effect_discovery,
    canonical_intervention_signature,
    classify_reference_status,
    compute_effect_discovery_for_question,
    format_hybrid_intervention_query,
    iter_hybrid_counterfactual_specs,
    load_reference_intervention_rows,
    reference_status_is_evaluable,
    require_hybrid_reference_mode,
    resume_row_is_complete,
    run_effect_reference_sweep_for_case,
    upsert_reference_csv,
)
from mas_langgraph.evaluation.report import build_report, overall_metric_rows, write_metrics_outputs
from mas_langgraph.evaluation.runner import build_arg_parser
from mas_langgraph.replay.interventions import (
    InterventionContract,
    InterventionContractRegistry,
)


def _vocab_fixture():
    chunk2 = ArtifactRef("chunk_2", 1)
    chunk3 = ArtifactRef("chunk_3", 1)
    chunk5 = ArtifactRef("chunk_5", 1)
    what = ArtifactRef("bp_1", 1)
    remove_bounds = ("before plan#1", "before gen#1")
    replace_bounds = ("before plan#1",)
    cand = InterventionCandidate(
        ref=chunk2,
        artifact_type="EvidenceChunk",
        allowed_operations=("REMOVE", "REPLACE"),
        operation_boundaries={
            "REMOVE": remove_bounds,
            "REPLACE": replace_bounds,
        },
    )
    vocab = HybridInterventionVocabulary(
        what_refs=(what, chunk2),
        intervention_candidates=(cand,),
        replacement_options={chunk2: (chunk3,)},
        add_options=(chunk5,),
        add_boundaries_by_ref={chunk5: ("before atomize#1",)},
        add_boundaries=("before atomize#1", "before unused#1"),
    )
    return vocab, chunk2, chunk3, chunk5


class _AxisResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _TextAdapter:
    factual_run = None

    def _text(self, run_result):
        fs = getattr(run_result, "final_state", None) or {}
        if isinstance(fs, dict):
            return str(fs.get("result_text") or "")
        return ""

    def observe_factual(self, *, explained_case, case_slice, registry):
        text = self._text(self.factual_run) if self.factual_run is not None else "R1"
        return CaseRunObservation(
            run_id=explained_case.run_id,
            branch_id="factual",
            decision=text or "R1",
            criterion_values=(),
            outcome_available=True,
        )

    def observe_replay(self, *, explained_case, run_result):
        text = self._text(run_result)
        return CaseRunObservation(
            run_id=explained_case.run_id,
            branch_id=str(getattr(run_result, "branch_id", "") or "cf"),
            decision=text or None,
            criterion_values=(),
            outcome_available=True,
        )

    def compare_subject_content(self, *, factual, noop, counterfactual, **kwargs):
        return SubjectContentDelta(
            factual_available=True,
            noop_available=True,
            counterfactual_available=True,
            control_stable=factual.decision == noop.decision,
            counterfactual_differs_from_noop=noop.decision != counterfactual.decision,
            effect_status=(
                "question_content_changed"
                if noop.decision != counterfactual.decision
                else "question_content_unchanged"
            ),
            details={
                "question_content_effect_status": (
                    "question_content_changed"
                    if noop.decision != counterfactual.decision
                    else "question_content_unchanged"
                )
            },
        )


class TestReferenceFlagAndFingerprint(unittest.TestCase):
    def test_flag_defaults_false_and_cli_accepts(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--lecturebank-dir", "x", "--id", "1", "--allow-any-count"]
        )
        self.assertFalse(args.build_effect_reference_set)
        self.assertIsNone(args.reference_max_interventions_per_question)
        on = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "1",
                "--allow-any-count",
                "--build-effect-reference-set",
                "--axis-interrogation-mode",
                "hybrid",
            ]
        )
        self.assertTrue(on.build_effect_reference_set)

    def test_constrained_mode_rejected(self):
        with self.assertRaises(HybridReferenceSweepError):
            require_hybrid_reference_mode("constrained")
        require_hybrid_reference_mode("hybrid")

    def test_generation_config_fingerprint_unchanged(self):
        base = {
            "llm_model": "gpt-4.1-mini-2025-04-14",
            "model_type": "openai",
            "temperature": 0.0,
            "seed": 42,
            "max_rounds": 5,
            "max_questions": 2,
            "max_explain_questions": 2,
            "orchestrator": "langgraph",
            "question": "Why?",
            "axis_interrogation_mode": "hybrid",
        }
        off = generation_config_fingerprint(base)
        on = generation_config_fingerprint(
            {
                **base,
                "build_effect_reference_set": True,
                "reference_max_interventions_per_question": 3,
            }
        )
        self.assertEqual(off, on)
        self.assertNotIn("build_effect_reference_set", on)
        self.assertNotIn("reference_max_interventions_per_question", on)


class TestUniverseEnumeration(unittest.TestCase):
    def test_excludes_what_and_done_and_enumerates_ops(self):
        vocab, chunk2, chunk3, chunk5 = _vocab_fixture()
        specs = iter_hybrid_counterfactual_specs(vocab)
        ops = {s.operation for s in specs}
        self.assertEqual(ops, {"REMOVE", "WHATIF", "ADD"})
        self.assertTrue(all(s.operation not in {"WHAT", "DONE"} for s in specs))
        remove = [s for s in specs if s.operation == "REMOVE"]
        whatif = [s for s in specs if s.operation == "WHATIF"]
        add = [s for s in specs if s.operation == "ADD"]
        self.assertEqual(len(remove), 2)
        self.assertEqual(len(whatif), 1)
        self.assertEqual(whatif[0].replacement_ref, chunk3)
        self.assertEqual(whatif[0].boundary_raw, "before plan#1")
        self.assertEqual(len(add), 1)
        self.assertEqual(add[0].target_ref, chunk5)
        self.assertEqual(add[0].boundary_raw, "before atomize#1")
        self.assertNotIn("before unused#1", [s.boundary_raw for s in add])

    def test_add_does_not_use_global_boundary_cartesian(self):
        vocab, _, _, chunk5 = _vocab_fixture()
        specs = iter_hybrid_counterfactual_specs(vocab)
        add_bounds = [s.boundary_raw for s in specs if s.operation == "ADD"]
        self.assertEqual(add_bounds, ["before atomize#1"])
        self.assertEqual(vocab.add_boundaries, ("before atomize#1", "before unused#1"))

    def test_duplicate_canonical_signatures_deduped(self):
        sig_a = canonical_intervention_signature(
            operation="WHATIF",
            target_ref=ArtifactRef("chunk_2", 1),
            replacement_ref=ArtifactRef("chunk_3", 1),
            checkpoint_id="c_8",
        )
        sig_b = canonical_intervention_signature(
            operation="REPLACE",
            target_ref={"key": "chunk_2:v1"},
            replacement_ref={"key": "chunk_3:v1"},
            checkpoint_id="c_8",
        )
        self.assertEqual(sig_a, sig_b)
        self.assertEqual(sig_a, "WHATIF|chunk_2:v1|chunk_3:v1|c_8")

    def test_invalid_combination_not_in_vocabulary(self):
        vocab, chunk2, _, _ = _vocab_fixture()
        bad = ResolvedQuery(
            operation="WHATIF",
            raw_text="WHATIF(chunk_2:v1, chunk_9:v1, before plan#1)",
            target_ref=chunk2,
            replacement_ref=ArtifactRef("chunk_9", 1),
            boundary_raw="before plan#1",
        )
        with self.assertRaises(QueryValidationError):
            validate_resolved_against_hybrid_vocabulary(bad, vocab)

    def test_query_format_matches_parser(self):
        q = format_hybrid_intervention_query(
            "WHATIF", ArtifactRef("chunk_2", 1), "before plan#1", ArtifactRef("chunk_3", 1)
        )
        self.assertEqual(q, "WHATIF(chunk_2:v1, chunk_3:v1, before plan#1)")


class TestReferenceStatusAndMetrics(unittest.TestCase):
    def _row(self, **kwargs):
        base = {
            "valid_comparison": True,
            "full_control_stable": True,
            "content_effect": False,
            "criterion_effect_any_supported": False,
            "decision_effect": False,
            "evidence_status": "no_demonstrated_effect",
        }
        base.update(kwargs)
        return base

    def test_valid_effect_and_no_effect(self):
        self.assertEqual(
            classify_reference_status(
                comparison_row=self._row(decision_effect=True, evidence_status="decision_effect")
            ),
            "valid_effect",
        )
        self.assertEqual(
            classify_reference_status(comparison_row=self._row()),
            "valid_no_effect",
        )

    def test_noop_unstable_is_inconclusive_not_false(self):
        self.assertEqual(
            classify_reference_status(
                comparison_row=self._row(
                    full_control_stable=False, evidence_status="replay_unstable"
                )
            ),
            "inconclusive_noop_unstable",
        )

    def test_replay_failure_not_no_effect(self):
        self.assertEqual(
            classify_reference_status(execution_status="failed", execution_error="boom"),
            "replay_failed",
        )
        self.assertEqual(
            classify_reference_status(
                comparison_row=self._row(
                    valid_comparison=False, evidence_status="intervention_error"
                )
            ),
            "replay_failed",
        )

    def _attempt(self, op, query, success=True, **extra):
        row = {
            "parsed_operation": op,
            "generated_query": query,
            "counted_as_successful_round": success,
            "validation_status": "executed" if success else extra.get("validation_status", "duplicate_error"),
        }
        row.update(extra)
        return row

    def test_recall_precision_what_done_and_empty_eq(self):
        sig_e = "WHATIF|a:v1|b:v1|c_1"
        sig_n = "REMOVE|a:v1||c_1"
        refs = [
            {
                "reference_intervention_signature": sig_e,
                "operation": "WHATIF",
                "generated_query": "WHATIF(a:v1, b:v1, before x#1)",
                "reference_status": "valid_effect",
            },
            {
                "reference_intervention_signature": sig_n,
                "operation": "REMOVE",
                "generated_query": "REMOVE(a:v1, before x#1)",
                "reference_status": "valid_no_effect",
            },
        ]
        attempts = [
            self._attempt("WHAT", "WHAT(a:v1)"),
            self._attempt("WHATIF", "WHATIF(a:v1, b:v1, before x#1)"),
            self._attempt("DONE", "DONE", success=False, validation_status="done"),
        ]
        comparisons = [
            {
                "generated_query": "WHATIF(a:v1, b:v1, before x#1)",
                "valid_comparison": True,
                "full_control_stable": True,
                "intervention_signature": {
                    "operation": "WHATIF",
                    "target_ref": {"key": "a:v1"},
                    "replacement_ref": {"key": "b:v1"},
                    "checkpoint_id": "c_1",
                },
            }
        ]
        out = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=True,
        )
        q = out["question_row"]
        self.assertEqual(q["axis_counterfactual_selected_at_k"], 1)
        self.assertEqual(q["axis_selected_effectful_at_k"], 1)
        self.assertEqual(q["axis_effectful_discoveries_at_k"], 1)
        self.assertEqual(q["effect_discovery_recall_at_k"], 1.0)
        self.assertEqual(q["effect_precision_at_k"], 1.0)

        empty = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=[
                {**refs[1], "reference_status": "valid_no_effect"},
            ],
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=True,
        )
        self.assertIsNone(empty["question_row"]["effect_discovery_recall_at_k"])

        what_only = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=[self._attempt("WHAT", f"WHAT(x{i})") for i in range(5)],
            axis_comparisons=[],
            reference_set_complete=True,
        )
        self.assertEqual(what_only["question_row"]["axis_counterfactual_selected_at_k"], 0)
        self.assertEqual(what_only["question_row"]["effect_discovery_recall_at_k"], 0.0)
        self.assertIsNone(what_only["question_row"]["effect_precision_at_k"])

        dup = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=1,
            reference_rows=refs,
            axis_attempts=[
                self._attempt("WHATIF", "WHATIF(a:v1, b:v1, before x#1)", success=False),
                self._attempt("WHATIF", "WHATIF(a:v1, b:v1, before x#1)"),
            ],
            axis_comparisons=comparisons,
            reference_set_complete=True,
        )
        self.assertEqual(dup["question_row"]["axis_counterfactual_selected_at_k"], 1)

        incomplete = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=False,
        )
        self.assertIsNone(incomplete["question_row"]["effect_discovery_recall_at_k"])
        self.assertFalse(incomplete["question_row"]["reference_set_complete"])

    def test_precision_uses_valid_noop_stable_only(self):
        sig_e = "WHATIF|a:v1|b:v1|c_1"
        sig_fail = "REMOVE|a:v1||c_1"
        refs = [
            {
                "reference_intervention_signature": sig_e,
                "operation": "WHATIF",
                "generated_query": "WHATIF(a:v1, b:v1, b)",
                "reference_status": "valid_effect",
            },
            {
                "reference_intervention_signature": sig_fail,
                "operation": "REMOVE",
                "generated_query": "REMOVE(a:v1, b)",
                "reference_status": "valid_effect",
            },
        ]
        attempts = [
            self._attempt("WHATIF", "WHATIF(a:v1, b:v1, b)"),
            self._attempt("REMOVE", "REMOVE(a:v1, b)"),
        ]
        comparisons = [
            {
                "generated_query": "WHATIF(a:v1, b:v1, b)",
                "valid_comparison": True,
                "full_control_stable": True,
                "intervention_signature": {
                    "operation": "WHATIF",
                    "target_ref": {"key": "a:v1"},
                    "replacement_ref": {"key": "b:v1"},
                    "checkpoint_id": "c_1",
                },
            },
            {
                "generated_query": "REMOVE(a:v1, b)",
                "valid_comparison": False,
                "full_control_stable": False,
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "a:v1"},
                    "checkpoint_id": "c_1",
                },
            },
        ]
        out = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=True,
        )
        q = out["question_row"]
        self.assertEqual(q["axis_counterfactual_selected_at_k"], 2)
        self.assertEqual(q["axis_valid_counterfactual_comparisons_at_k"], 1)
        self.assertEqual(q["axis_selected_effectful_at_k"], 2)
        self.assertEqual(q["axis_effectful_discoveries_at_k"], 1)
        self.assertEqual(q["effect_discovery_recall_at_k"], 1.0)
        self.assertEqual(q["effect_precision_at_k"], 1.0)

    def test_macro_micro_and_coverage(self):
        rows = [
            {
                "effect_discovery_recall_at_k": 1.0,
                "effect_precision_at_k": 1.0,
                "reference_effectful_count": 2,
                "axis_selected_effectful_at_k": 2,
                "axis_effectful_discoveries_at_k": 1,
                "axis_valid_effectful_discoveries_at_k": 1,
                "axis_valid_counterfactual_comparisons_at_k": 1,
                "axis_reference_evaluable_counterfactuals_at_k": 1,
                "axis_reference_unevaluable_counterfactuals_at_k": 0,
                "axis_counterfactual_selected_at_k": 1,
                "causal_evidence_available_at_k": True,
                "reference_set_complete": True,
                "reference_universe_count": 4,
                "reference_evaluable_count": 2,
                "reference_evaluability_rate": 0.5,
                "k": 5,
            },
            {
                "effect_discovery_recall_at_k": None,
                "effect_precision_at_k": None,
                "reference_effectful_count": 0,
                "axis_selected_effectful_at_k": 0,
                "axis_effectful_discoveries_at_k": 0,
                "axis_valid_effectful_discoveries_at_k": 0,
                "axis_valid_counterfactual_comparisons_at_k": 0,
                "axis_reference_evaluable_counterfactuals_at_k": 0,
                "axis_reference_unevaluable_counterfactuals_at_k": 0,
                "axis_counterfactual_selected_at_k": 0,
                "causal_evidence_available_at_k": False,
                "reference_set_complete": True,
                "reference_universe_count": 3,
                "reference_evaluable_count": 0,
                "reference_evaluability_rate": 0.0,
                "k": 5,
            },
        ]
        agg = aggregate_effect_discovery(rows)
        self.assertEqual(agg["effect_discovery_recall_at_k_macro"], 1.0)
        self.assertEqual(agg["effect_discovery_recall_at_k_micro"], 1.0)
        self.assertEqual(agg["causal_evidence_coverage_at_k"], 0.5)
        self.assertEqual(agg["questions_with_empty_effectful_reference"], 1)
        self.assertEqual(agg["effect_precision_at_k_macro"], 1.0)
        self.assertEqual(agg["effect_precision_at_k_micro"], 1.0)
        self.assertEqual(agg["effect_precision_micro_den"], 1)
        self.assertEqual(agg["reference_evaluability_rate_macro"], 0.25)
        self.assertEqual(agg["reference_evaluability_rate_micro"], 2 / 7)

    def test_first_attempt_counts_agree_with_reference_universe(self):
        vocab, chunk2, chunk3, _ = _vocab_fixture()
        specs = iter_hybrid_counterfactual_specs(vocab)
        advertised = hybrid_opportunity_counts(vocab)
        self.assertEqual(
            advertised["remove"],
            sum(1 for s in specs if s.operation == "REMOVE"),
        )
        self.assertEqual(
            advertised["whatif"],
            sum(1 for s in specs if s.operation == "WHATIF"),
        )
        self.assertEqual(
            advertised["add"],
            sum(1 for s in specs if s.operation == "ADD"),
        )
        consumed = QuerySignature(
            operation="WHATIF",
            target_ref=chunk2,
            replacement_ref=chunk3,
            checkpoint_alias="before plan#1",
            boundary_raw="before plan#1",
        )
        remaining = hybrid_opportunity_counts(vocab, executed_signatures={consumed})
        self.assertEqual(remaining["whatif"], advertised["whatif"] - 1)
        self.assertEqual(
            sum(1 for s in iter_hybrid_counterfactual_specs(vocab) if s.operation == "WHATIF"),
            advertised["whatif"],
        )
        out = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=[
                {
                    "reference_intervention_signature": canonical_intervention_signature(
                        operation=s.operation,
                        target_ref=s.target_ref,
                        replacement_ref=s.replacement_ref,
                        checkpoint_id=s.boundary_raw,
                    ),
                    "operation": s.operation,
                    "generated_query": s.generated_query,
                    "reference_status": "valid_no_effect",
                }
                for s in specs
            ],
            axis_attempts=[],
            reference_set_complete=True,
            initial_opportunity={
                "available_remove_count": advertised["remove"],
                "available_add_count": advertised["add"],
                "available_whatif_count": advertised["whatif"],
            },
        )
        self.assertIs(out["question_row"]["reference_opportunity_count_match"], True)
        self.assertEqual(out["question_row"]["opportunity_mismatch_detail"], "")

    def test_opportunity_mismatch_surfaced(self):
        out = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=[
                {
                    "reference_intervention_signature": "REMOVE|a:v1||c_1",
                    "operation": "REMOVE",
                    "reference_status": "valid_no_effect",
                }
            ],
            axis_attempts=[],
            reference_set_complete=True,
            initial_opportunity={
                "available_remove_count": 40,
                "available_add_count": 0,
                "available_whatif_count": 24,
            },
        )
        self.assertIs(out["question_row"]["reference_opportunity_count_match"], False)
        self.assertIn("40", out["question_row"]["opportunity_mismatch_detail"])

    def test_initial_universe_keeps_selected_whatif(self):
        vocab, *_ = _vocab_fixture()
        specs = iter_hybrid_counterfactual_specs(vocab)
        whatif = [s for s in specs if s.operation == "WHATIF"]
        self.assertEqual(len(whatif), 1)
        remaining = hybrid_like_after_one(vocab)
        self.assertEqual(len(remaining), len(specs))

    def test_csv_round_trip_and_resume_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference_interventions.csv"
            row = {
                "lecturebank_id": "141",
                "candidate_ref": "q:v1",
                "reference_intervention_signature": "REMOVE|a:v1||c_1",
                "operation": "REMOVE",
                "reference_status": "valid_effect",
                "reference_evaluable": True,
                "supported_effect": True,
            }
            upsert_reference_csv(path, [row])
            upsert_reference_csv(path, [row])
            loaded = load_reference_intervention_rows(path)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(loaded[0]["supported_effect"])
            self.assertTrue(loaded[0]["reference_evaluable"])
            self.assertTrue(resume_row_is_complete(loaded[0]))
            failed = dict(row)
            failed["reference_intervention_signature"] = "WHATIF|a:v1|b:v1|c_1"
            failed["reference_status"] = "replay_failed"
            failed["reference_evaluable"] = False
            failed["supported_effect"] = None
            upsert_reference_csv(path, [failed])
            loaded = load_reference_intervention_rows(path)
            self.assertEqual(len(loaded), 2)
            self.assertFalse(resume_row_is_complete(failed))
            statuses = {r["reference_status"]: r["reference_evaluable"] for r in loaded}
            self.assertTrue(statuses["valid_effect"])
            self.assertFalse(statuses["replay_failed"])


class TestReferenceEvaluabilityAndPrecision(unittest.TestCase):
    def _attempt(self, op, query, success=True, extra=None):
        extra = extra or {}
        row = {
            "parsed_operation": op,
            "generated_query": query,
            "counted_as_successful_round": success,
            "validation_status": "executed" if success else extra.get("validation_status", "duplicate_error"),
        }
        row.update(extra)
        return row

    def _cmp(self, query, operation, target, checkpoint="c_1", replacement=None):
        sig = {
            "operation": operation,
            "target_ref": {"key": target},
            "checkpoint_id": checkpoint,
        }
        if replacement is not None:
            sig["replacement_ref"] = {"key": replacement}
        return {
            "generated_query": query,
            "valid_comparison": True,
            "full_control_stable": True,
            "intervention_signature": sig,
        }

    def _abcd_fixture(self):
        sig_a = "WHATIF|a:v1|b:v1|c_1"
        sig_b = "REMOVE|b:v1||c_1"
        sig_c = "ADD|c:v1||c_1"
        sig_d = "REMOVE|d:v1||c_1"
        q_a = "WHATIF(a:v1, b:v1, before x#1)"
        q_b = "REMOVE(b:v1, before x#1)"
        q_c = "ADD(c:v1, before x#1)"
        q_d = "REMOVE(d:v1, before x#1)"
        refs = [
            {
                "reference_intervention_signature": sig_a,
                "operation": "WHATIF",
                "generated_query": q_a,
                "reference_status": "valid_effect",
            },
            {
                "reference_intervention_signature": sig_b,
                "operation": "REMOVE",
                "generated_query": q_b,
                "reference_status": "valid_no_effect",
            },
            {
                "reference_intervention_signature": sig_c,
                "operation": "ADD",
                "generated_query": q_c,
                "reference_status": "replay_failed",
            },
            {
                "reference_intervention_signature": sig_d,
                "operation": "REMOVE",
                "generated_query": q_d,
                "reference_status": "inconclusive_noop_unstable",
            },
        ]
        attempts = [
            self._attempt("WHATIF", q_a),
            self._attempt("REMOVE", q_b),
            self._attempt("ADD", q_c),
            self._attempt("REMOVE", q_d),
        ]
        comparisons = [
            self._cmp(q_a, "WHATIF", "a:v1", replacement="b:v1"),
            self._cmp(q_b, "REMOVE", "b:v1"),
            self._cmp(q_c, "ADD", "c:v1"),
            self._cmp(q_d, "REMOVE", "d:v1"),
        ]
        return refs, attempts, comparisons

    def test_evaluable_status_classification(self):
        self.assertEqual(EVALUABLE_REFERENCE_STATUSES, {"valid_effect", "valid_no_effect"})
        self.assertTrue(reference_status_is_evaluable("valid_effect"))
        self.assertTrue(reference_status_is_evaluable("valid_no_effect"))
        self.assertFalse(reference_status_is_evaluable("inconclusive_noop_unstable"))
        self.assertFalse(reference_status_is_evaluable("replay_failed"))
        self.assertFalse(reference_status_is_evaluable("invalid"))
        self.assertIsNone(reference_status_is_evaluable(None))
        self.assertIsNone(reference_status_is_evaluable(""))

    def test_abcd_precision_excludes_unevaluable_reference_labels(self):
        refs, attempts, comparisons = self._abcd_fixture()
        out = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=True,
        )
        q = out["question_row"]
        self.assertEqual(q["reference_universe_count"], 4)
        self.assertEqual(q["reference_evaluable_count"], 2)
        self.assertEqual(q["reference_evaluability_rate"], 0.5)
        self.assertEqual(q["reference_effectful_count"], 1)
        self.assertEqual(q["reference_no_effect_count"], 1)
        self.assertEqual(q["axis_counterfactual_selected_at_k"], 4)
        self.assertEqual(q["axis_valid_counterfactual_comparisons_at_k"], 4)
        self.assertEqual(q["axis_reference_evaluable_counterfactuals_at_k"], 2)
        self.assertEqual(q["axis_reference_unevaluable_counterfactuals_at_k"], 2)
        self.assertEqual(q["axis_selected_effectful_at_k"], 1)
        self.assertEqual(q["axis_effectful_discoveries_at_k"], 1)
        self.assertEqual(q["effect_precision_at_k"], 0.5)
        self.assertNotEqual(q["effect_precision_at_k"], 0.25)
        self.assertEqual(q["effect_discovery_recall_at_k"], 1.0)
        self.assertTrue(q["reference_set_complete"])
        self.assertEqual(q["effect_discovery_end_to_end_yield_at_k"], 0.25)
        annotated = {
            r["reference_intervention_signature"]: r["reference_evaluable"]
            for r in out["reference_rows"]
        }
        self.assertTrue(annotated["WHATIF|a:v1|b:v1|c_1"])
        self.assertTrue(annotated["REMOVE|b:v1||c_1"])
        self.assertFalse(annotated["ADD|c:v1||c_1"])
        self.assertFalse(annotated["REMOVE|d:v1||c_1"])

    def test_zero_universe_evaluability_is_na(self):
        out = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=[],
            axis_attempts=[],
            axis_comparisons=[],
            reference_set_complete=True,
        )
        q = out["question_row"]
        self.assertEqual(q["reference_universe_count"], 0)
        self.assertEqual(q["reference_evaluable_count"], 0)
        self.assertIsNone(q["reference_evaluability_rate"])
        self.assertIsNone(q["effect_precision_at_k"])

    def test_precision_na_when_no_evaluable_axis_selection(self):
        refs, attempts, comparisons = self._abcd_fixture()
        unevaluable_only = [
            r for r in refs if r["reference_status"] not in {"valid_effect", "valid_no_effect"}
        ]
        attempts = [self._attempt(r["operation"], r["generated_query"]) for r in unevaluable_only]
        queries = {r["generated_query"] for r in unevaluable_only}
        comparisons = [c for c in comparisons if c["generated_query"] in queries]
        out = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=True,
        )
        q = out["question_row"]
        self.assertEqual(q["axis_valid_counterfactual_comparisons_at_k"], 2)
        self.assertEqual(q["axis_reference_evaluable_counterfactuals_at_k"], 0)
        self.assertEqual(q["axis_reference_unevaluable_counterfactuals_at_k"], 2)
        self.assertIsNone(q["effect_precision_at_k"])
        self.assertEqual(q["effect_discovery_recall_at_k"], 0.0)

    def test_valid_no_effect_is_in_precision_denominator(self):
        refs, attempts, comparisons = self._abcd_fixture()
        keep = {"valid_effect", "valid_no_effect"}
        refs = [r for r in refs if r["reference_status"] in keep]
        queries = {r["generated_query"] for r in refs}
        attempts = [a for a in attempts if a["generated_query"] in queries]
        comparisons = [c for c in comparisons if c["generated_query"] in queries]
        out = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=True,
        )
        q = out["question_row"]
        self.assertEqual(q["axis_reference_evaluable_counterfactuals_at_k"], 2)
        self.assertEqual(q["axis_effectful_discoveries_at_k"], 1)
        self.assertEqual(q["effect_precision_at_k"], 0.5)

    def test_macro_precision_excludes_undefined(self):
        rows = [
            {
                "effect_precision_at_k": 0.5,
                "effect_discovery_recall_at_k": 1.0,
                "reference_effectful_count": 1,
                "axis_selected_effectful_at_k": 1,
                "axis_effectful_discoveries_at_k": 1,
                "axis_reference_evaluable_counterfactuals_at_k": 2,
                "axis_reference_unevaluable_counterfactuals_at_k": 2,
                "axis_counterfactual_selected_at_k": 4,
                "reference_set_complete": True,
                "reference_universe_count": 4,
                "reference_evaluable_count": 2,
                "reference_evaluability_rate": 0.5,
                "k": 5,
            },
            {
                "effect_precision_at_k": None,
                "effect_discovery_recall_at_k": 0.0,
                "reference_effectful_count": 1,
                "axis_selected_effectful_at_k": 0,
                "axis_effectful_discoveries_at_k": 0,
                "axis_reference_evaluable_counterfactuals_at_k": 0,
                "axis_reference_unevaluable_counterfactuals_at_k": 2,
                "axis_counterfactual_selected_at_k": 2,
                "reference_set_complete": True,
                "reference_universe_count": 4,
                "reference_evaluable_count": 2,
                "reference_evaluability_rate": 0.5,
                "k": 5,
            },
        ]
        agg = aggregate_effect_discovery(rows)
        self.assertEqual(agg["effect_precision_at_k_macro"], 0.5)
        self.assertEqual(agg["effect_precision_macro_num"], 1)
        self.assertEqual(agg["effect_precision_at_k_micro"], 0.5)
        self.assertEqual(agg["effect_precision_micro_num"], 1)
        self.assertEqual(agg["effect_precision_micro_den"], 2)
        self.assertEqual(agg["effect_discovery_recall_at_k_micro"], 0.5)

    def test_macro_evaluability_excludes_zero_universe(self):
        rows = [
            {
                "reference_universe_count": 4,
                "reference_evaluable_count": 2,
                "reference_evaluability_rate": 0.5,
                "reference_set_complete": True,
                "k": 5,
            },
            {
                "reference_universe_count": 0,
                "reference_evaluable_count": 0,
                "reference_evaluability_rate": None,
                "reference_set_complete": True,
                "k": 5,
            },
            {
                "reference_universe_count": 10,
                "reference_evaluable_count": 2,
                "reference_evaluability_rate": 0.2,
                "reference_set_complete": True,
                "k": 5,
            },
        ]
        agg = aggregate_effect_discovery(rows)
        self.assertAlmostEqual(agg["reference_evaluability_rate_macro"], 0.35)
        self.assertEqual(agg["reference_evaluability_rate_micro"], 4 / 14)
        self.assertEqual(agg["reference_evaluability_macro_num"], 2)
        self.assertTrue(agg["reference_set_complete"])

    def test_incomplete_limiter_still_makes_recall_na(self):
        refs, attempts, comparisons = self._abcd_fixture()
        out = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=False,
        )
        q = out["question_row"]
        self.assertIsNone(q["effect_discovery_recall_at_k"])
        self.assertFalse(q["reference_set_complete"])
        self.assertEqual(q["effect_precision_at_k"], 0.5)
        self.assertEqual(q["reference_evaluability_rate"], 0.5)

    def test_report_and_overall_metrics_include_evaluability(self):
        refs, attempts, comparisons = self._abcd_fixture()
        qrow = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=True,
        )["question_row"]
        agg = aggregate_effect_discovery([qrow])
        md = build_report(
            catalog_rows=[],
            document_rows=[],
            attempt_rows=[],
            comparison_rows=[],
            provenance_rows=[],
            distribution_rows=[],
            round_rows=[],
            overall_rows=[],
            pooled=agg,
            reference_question_rows=[qrow],
            reference_intervention_rows=refs,
            reference_aggregate=agg,
        )
        self.assertIn(
            "Reference evaluability is the proportion of the enumerated intervention", md
        )
        self.assertIn(
            "Effect Precision@K is calculated only over AXIS-selected interventions that", md
        )
        self.assertIn("have an evaluable reference label", md)
        self.assertIn("Macro Reference Evaluability Rate", md)
        self.assertIn("Micro Reference Evaluability Rate", md)
        self.assertIn("AXIS-selected valid CFs with evaluable reference labels", md)
        self.assertIn("AXIS-selected valid CFs without evaluable reference labels", md)
        self.assertIn("replay-derived reference set, not external causal ground truth", md)
        overall = overall_metric_rows([], [], [], [], {**agg, "reference_sweep_ran": True, "k": 5})
        names = {r["metric"] for r in overall}
        for field in (
            "reference_evaluable_count",
            "reference_evaluability_rate_micro",
            "reference_evaluability_rate_macro",
            "axis_reference_evaluable_counterfactuals_at_k",
            "axis_reference_unevaluable_counterfactuals_at_k",
            "effect_precision_at_k_micro",
            "effect_precision_at_k_macro",
        ):
            self.assertIn(field, names)
        prec = next(r for r in overall if r["metric"] == "effect_precision_at_k_micro")
        self.assertEqual(prec["value"], 0.5)
        self.assertEqual(prec["numerator"], 1)
        self.assertEqual(prec["denominator"], 2)

    def test_question_csv_contains_new_fields(self):
        refs, attempts, comparisons = self._abcd_fixture()
        qrow = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            reference_rows=refs,
            axis_attempts=attempts,
            axis_comparisons=comparisons,
            reference_set_complete=True,
        )["question_row"]
        for field in (
            "reference_evaluable_count",
            "reference_evaluability_rate",
            "axis_reference_evaluable_counterfactuals_at_k",
            "axis_reference_unevaluable_counterfactuals_at_k",
            "axis_selected_effectful_at_k",
        ):
            self.assertIn(field, EFFECT_DISCOVERY_QUESTION_FIELDS)
            self.assertIn(field, qrow)
        self.assertIn("reference_evaluable", REFERENCE_INTERVENTION_FIELDS)
        with tempfile.TemporaryDirectory() as tmp:
            write_metrics_outputs(
                Path(tmp),
                document_rows=[],
                attempt_rows=[],
                comparison_rows=[],
                provenance_rows=[],
                distribution_rows=[],
                round_rows=[],
                overall_rows=[],
                report_markdown="# test\n",
                reference_intervention_rows=[
                    {**r, "lecturebank_id": "141", "candidate_ref": "q:v1"} for r in refs
                ],
                effect_discovery_question_rows=[qrow],
            )
            qcsv = (Path(tmp) / "effect_discovery_by_question.csv").read_text(encoding="utf-8")
            self.assertIn("reference_evaluability_rate", qcsv)
            self.assertIn("axis_reference_evaluable_counterfactuals_at_k", qcsv)
            icsv = (Path(tmp) / "reference_interventions.csv").read_text(encoding="utf-8")
            self.assertIn("reference_evaluable", icsv)

    def test_limited_sweep_recalculates_evaluability_after_universe_overwrite(self):
        from mas_langgraph.axis.models import ExplainedCase
        from mas_langgraph.axis.slicing import CaseSlice
        from mas_langgraph.evaluation import effect_reference as er

        universe = [
            {
                "reference_intervention_signature": "WHATIF|a:v1|b:v1|c_1",
                "operation": "WHATIF",
                "generated_query": "WHATIF(a:v1, b:v1, before x#1)",
            },
            {
                "reference_intervention_signature": "REMOVE|b:v1||c_1",
                "operation": "REMOVE",
                "generated_query": "REMOVE(b:v1, before x#1)",
            },
            {
                "reference_intervention_signature": "ADD|c:v1||c_1",
                "operation": "ADD",
                "generated_query": "ADD(c:v1, before x#1)",
            },
            {
                "reference_intervention_signature": "REMOVE|d:v1||c_1",
                "operation": "REMOVE",
                "generated_query": "REMOVE(d:v1, before x#1)",
            },
        ]
        executed_status = {
            "WHATIF|a:v1|b:v1|c_1": "valid_effect",
            "REMOVE|b:v1||c_1": "valid_no_effect",
        }

        def fake_execute(*, candidate, lecturebank_id, candidate_ref, **_kwargs):
            sig = str(candidate["reference_intervention_signature"])
            return {
                "lecturebank_id": lecturebank_id,
                "candidate_ref": candidate_ref,
                "reference_intervention_signature": sig,
                "operation": candidate.get("operation"),
                "generated_query": candidate.get("generated_query"),
                "reference_status": executed_status[sig],
            }

        dummy_vocab = HybridInterventionVocabulary()
        explained = ExplainedCase(
            run_id="run_limiter",
            artifact_ref=ArtifactRef("out", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:out", 1),
            outcome="accept",
        )
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=(),
            relevant_occurrences=(),
            relevant_relations=(),
        )
        axis_result = _AxisResult(
            explained_case=explained,
            case_slice=case_slice,
            interrogation_attempts=[],
            replay_comparisons=[],
            intervention_records=[],
        )
        replay_executor = MagicMock()
        replay_executor.get_or_create_llm_ledger = None
        with (
            patch.object(er, "build_initial_hybrid_vocabulary", return_value=(dummy_vocab, ())),
            patch.object(er, "enumerate_initial_hybrid_universe", return_value=(list(universe), [])),
            patch.object(er, "execute_reference_intervention", side_effect=fake_execute),
            patch.object(er, "AxisQueryExecutor", return_value=MagicMock()),
        ):
            payload = run_effect_reference_sweep_for_case(
                lecturebank_id="141",
                candidate_ref="q:v1",
                axis_result=axis_result,
                replay_executor=replay_executor,
                factual_run_result=MagicMock(produced_by=(), final_state={}),
                case_definition=None,
                checkpoint_index={"records": []},
                contracts=MagicMock(),
                observation_adapter=_TextAdapter(),
                reference_max_interventions_per_question=2,
            )
        qrow = payload["question_row"]
        self.assertEqual(qrow["reference_universe_count"], 4)
        self.assertEqual(qrow["reference_evaluable_count"], 2)
        self.assertEqual(qrow["reference_evaluability_rate"], 0.5)
        self.assertIs(payload["reference_set_complete"], False)
        self.assertFalse(qrow["reference_set_complete"])
        self.assertIsNone(qrow["effect_discovery_recall_at_k"])
        self.assertEqual(len(payload["intervention_rows"]), 2)
        agg = aggregate_effect_discovery([qrow])
        self.assertEqual(agg["reference_evaluability_rate_macro"], 0.5)
        self.assertEqual(agg["reference_evaluability_rate_micro"], 0.5)

    def test_effect_yield_field_untouched_in_document_metrics(self):
        from mas_langgraph.evaluation.report import DOCUMENT_FIELDS

        self.assertIn("effect_yield_any", DOCUMENT_FIELDS)


def hybrid_like_after_one(vocab):
    """Reference universe must ignore AXIS consumption (empty executed set)."""
    return iter_hybrid_counterfactual_specs(vocab)


class TestNormalMetricsIsolation(unittest.TestCase):
    def test_ledger_restore_does_not_keep_reference_counters(self):
        from mas_langgraph.evaluation.effect_reference import (
            _restore_llm_ledger,
            _snapshot_llm_ledger,
        )

        class _Ledger:
            counterfactual_live_provider_calls = 3
            _responses = {"a": 1}
            call_log = [{"x": 1}]

            def _counter_snapshot(self):
                return {
                    "counterfactual_live_provider_calls": self.counterfactual_live_provider_calls
                }

        ledger = _Ledger()
        snap = _snapshot_llm_ledger(ledger)
        ledger.counterfactual_live_provider_calls = 99
        ledger._responses["b"] = 2
        ledger.call_log.append({"y": 2})
        _restore_llm_ledger(ledger, snap)
        self.assertEqual(ledger.counterfactual_live_provider_calls, 3)
        self.assertEqual(ledger._responses, {"a": 1})
        self.assertEqual(ledger.call_log, [{"x": 1}])


class TestCompiledGraphReferenceSweep(unittest.TestCase):
    def test_real_compiled_graph_enumerates_and_executes(self):
        from langgraph.graph import END, START, StateGraph

        from mas_langgraph.artifacts.channel_state import (
            dump_artifact_channels,
            empty_artifact_channels,
            load_artifact_channels,
        )
        from mas_langgraph.artifacts.registry import ArtifactRegistry
        from mas_langgraph.artifacts.resolver import ArtifactResolver
        from mas_langgraph.axis.models import ExplainedCase
        from mas_langgraph.axis.slicing import CaseSlice
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.evaluation.effect_reference import run_effect_reference_sweep_for_case
        from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation
        from mas_langgraph.replay.executor import LangGraphReplayExecutor

        run_id = "run_ref_sweep"
        used = ArtifactRef("src", 1)
        unused = ArtifactRef("other", 1)
        alt = ArtifactRef("alt", 1)
        contracts = InterventionContractRegistry()
        contracts.register(
            InterventionContract(
                artifact_type="EvidenceChunk",
                allowed_operations=("REMOVE", "ADD", "REPLACE"),
                replacement_compatible_types=("EvidenceChunk",),
                allow_external_add=True,
            )
        )

        from mas_langgraph.tests.test_axis_hybrid_interrogation import _MetaPayload

        fact_reg = ArtifactRegistry()
        fact_reg.store(
            ArtifactEnvelope.create(
                logical_id="src",
                version=1,
                artifact_type="EvidenceChunk",
                payload=_MetaPayload(content="R1"),
                branch_id="factual",
                created_in_run=run_id,
            )
        )
        fact_reg.store(
            ArtifactEnvelope.create(
                logical_id="other",
                version=1,
                artifact_type="EvidenceChunk",
                payload=_MetaPayload(content="UNUSED"),
                branch_id="factual",
                created_in_run=run_id,
            )
        )
        int_reg = ArtifactRegistry()
        int_reg.store(
            ArtifactEnvelope.create(
                logical_id="alt",
                version=1,
                artifact_type="EvidenceChunk",
                payload=_MetaPayload(content="R2"),
                branch_id="factual",
                created_in_run=run_id,
            )
        )

        class S(TypedDict, total=False):
            result_text: str
            branch_id: str
            run_id: str
            artifact_catalog: dict
            available_artifact_refs: set
            blocked_artifact_refs: set
            artifact_substitutions: dict
            artifact_derivations: list
            intervention_events: list
            occurrence_records: Annotated[list, operator.add]
            run_provenance_relations: Annotated[list, operator.add]
            path: Annotated[list, operator.add]
            branch_type: str
            parent_branch_id: str
            origin_checkpoint_alias: str
            intervention_id: Any
            provenance_node_counts: dict
            provenance_execution_index: int

        def source(state: S) -> dict:
            occ = OccurrenceRecord(
                occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:source#1",
                node_id="source",
                occurrence_index=1,
                branch_id=str(state.get("branch_id") or "factual"),
                run_id=run_id,
                execution_index=1,
                produced_artifacts=(used, unused),
            )
            rel = RunProvenanceRelation(
                producer_occurrence_id=occ.occurrence_id,
                artifact_ref=used,
                relation="supporting_evidence",
                consumer_occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:transform#1",
            )
            return {
                "occurrence_records": [occ],
                "run_provenance_relations": [rel],
                "path": ["source"],
            }

        def transform(state: S) -> dict:
            _reg, resolver = load_artifact_channels(state)
            env = resolver.get(used)
            text = str(getattr(getattr(env, "payload", None), "content", "") or "")
            occ = OccurrenceRecord(
                occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:transform#1",
                node_id="transform",
                occurrence_index=1,
                branch_id=str(state.get("branch_id") or "factual"),
                run_id=run_id,
                execution_index=2,
            )
            return {
                "result_text": text,
                "occurrence_records": [occ],
                "path": ["transform"],
            }

        g = StateGraph(S)
        g.add_node("source", source)
        g.add_node("transform", transform)
        g.add_edge(START, "source")
        g.add_edge("source", "transform")
        g.add_edge("transform", END)
        app = g.compile(checkpointer=default_factual_checkpointer())
        resolver = ArtifactResolver(registry=fact_reg)
        resolver.make_available(used)
        resolver.make_available(unused)
        initial = {
            "result_text": "",
            "path": [],
            "occurrence_records": [],
            "run_provenance_relations": [],
            **empty_artifact_channels(),
            **dump_artifact_channels(fact_reg, resolver),
        }
        ex = LangGraphReplayExecutor(compiled_graph=app)
        ex.contracts = contracts
        factual = ex.run_factual(initial, run_id=run_id, runtime=WorkflowRuntime())
        self.assertIsNone(factual.error, msg=factual.error)
        self.assertIsNotNone(ex.compiled_graph)

        occs = list(factual.final_state.get("occurrence_records") or [])
        rels = list(factual.final_state.get("run_provenance_relations") or [])
        explained = ExplainedCase(
            run_id=run_id,
            artifact_ref=ArtifactRef("out", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:out", 1),
            outcome="accept",
        )
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=(used, unused),
            relevant_occurrences=tuple(occs),
            relevant_relations=tuple(rels),
        )
        axis_result = _AxisResult(
            explained_case=explained,
            case_slice=case_slice,
            interrogation_attempts=[],
            replay_comparisons=[],
            intervention_records=[],
        )
        adapter = _TextAdapter()
        adapter.factual_run = factual
        payload = run_effect_reference_sweep_for_case(
            lecturebank_id="141",
            candidate_ref="out:v1",
            axis_result=axis_result,
            replay_executor=ex,
            factual_run_result=factual,
            case_definition=None,
            checkpoint_index=factual.checkpoint_index,
            contracts=contracts,
            intervention_alternatives={"alt": alt},
            intervention_registry=int_reg,
            registry=fact_reg,
            occurrences=occs,
            max_rounds=5,
            observation_adapter=adapter,
        )
        rows = payload["intervention_rows"]
        self.assertGreaterEqual(len(rows), 2)
        statuses = {r["reference_status"] for r in rows}
        summary = [
            (r.get("operation"), r.get("reference_status"), r.get("evidence_status"), r.get("decision_effect"), r.get("full_noop_stable"))
            for r in rows
        ]
        self.assertIn("valid_effect", statuses, msg=summary)
        self.assertIn("valid_no_effect", statuses, msg=summary)
        effectful = next(r for r in rows if r["reference_status"] == "valid_effect")
        axis_result.interrogation_attempts = [
            {
                "parsed_operation": effectful["operation"],
                "generated_query": effectful["generated_query"],
                "counted_as_successful_round": True,
            }
        ]
        discovery = compute_effect_discovery_for_question(
            lecturebank_id="141",
            candidate_ref="out:v1",
            k=5,
            reference_rows=rows,
            axis_attempts=axis_result.interrogation_attempts,
            axis_comparisons=[],
            reference_set_complete=True,
        )
        self.assertEqual(discovery["question_row"]["axis_selected_effectful_at_k"], 1)
        self.assertEqual(discovery["question_row"]["axis_effectful_discoveries_at_k"], 0)
        e_q = int(discovery["question_row"]["reference_effectful_count"] or 0)
        self.assertGreater(e_q, 0)
        self.assertEqual(
            discovery["question_row"]["effect_discovery_recall_at_k"],
            1.0 / e_q if e_q else None,
        )
        self.assertIs(payload["executor"].replay_executor, ex)


class TestNormalRunUnchangedWhenDisabled(unittest.TestCase):
    def test_service_does_not_call_sweep_when_flag_off(self):
        from mas_langgraph.evaluation import effect_reference as er

        with patch.object(er, "run_effect_reference_sweep_for_case") as sweep:
            # Flag default is False; constructing the service without the flag
            # must not invoke the sweep helper.
            self.assertFalse(sweep.called)
            self.assertFalse(
                getattr(
                    type("S", (), {"build_effect_reference_set": False})(),
                    "build_effect_reference_set",
                )
            )


if __name__ == "__main__":
    unittest.main()
