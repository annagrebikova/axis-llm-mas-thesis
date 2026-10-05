"""Optional LLM-assessed tested-concept change (mocked evaluator; no live model calls).

These tests validate eligibility, plumbing, caching, and aggregation. They do
not validate the semantic accuracy of the rubric. A manually reviewed sample
is still required before treating labels as human-quality judgments.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from unittest.mock import patch

from mas_langgraph.batch.manifest import generation_config_fingerprint
from mas_langgraph.evaluation.axis_metrics import extract_experiment_metrics
from mas_langgraph.evaluation.concept_change import (
    ConceptChangeCache,
    ConceptChangeEvaluatorConfig,
    aggregate_concept_change,
    assess_comparison,
    attach_assessments_to_metric_rows,
    compute_concept_change_for_question,
    concept_change_cache_key,
    concept_change_eligibility,
    evaluate_experiment_payload,
    evaluate_saved_experiment_json,
    expected_concept_change_selection,
    maybe_evaluate_concept_change,
    parse_concept_change_response,
    persisted_concept_change_complete,
)
from mas_langgraph.evaluation.execute import (
    _apply_concept_change_evaluation,
    run_concept_change_from_saved,
)
from mas_langgraph.evaluation.lifecycle import load_collected_from_metrics
from mas_langgraph.evaluation.report import (
    build_report,
    overall_metric_rows,
    write_metrics_outputs,
)
from mas_langgraph.evaluation.runner import _persist, build_arg_parser
from mas_langgraph.evaluation.streamlit_export import write_streamlit_experiment_json


def _json_resp(
    label: str,
    *,
    noop: str = "goal of unsupervised learning",
    cf: Optional[str] = None,
) -> str:
    cf = cf or noop
    return json.dumps(
        {
            "noop_tested_concept": noop,
            "counterfactual_tested_concept": cf,
            "concept_change_label": label,
            "concept_change_reason": f"mocked:{label}",
            "noop_excerpt": "unsupervised learning",
            "counterfactual_excerpt": "unsupervised learning",
        }
    )


def _mcq(
    stem: str,
    correct: str = "find hidden structure",
    distractors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "stem": stem,
        "correct_answer": correct,
        "distractors": list(distractors or ["maximize labeled accuracy", "require human labels"]),
        "answer_explanation": "because unlabeled data",
    }


def _qcc(
    noop: Dict[str, Any],
    cf: Optional[Dict[str, Any]],
    *,
    status: str = "question_content_unchanged",
    present: bool = True,
) -> Dict[str, Any]:
    return {
        "effect_status": status,
        "counterfactual_differs_from_noop": status
        in {"question_content_changed", "question_not_reproduced"},
        "details": {
            "kind": "mcq_content",
            "question_content_effect_status": status,
            "question_present_in_counterfactual": present,
            "stem_effect": "changed" if status == "question_content_changed" else "unchanged",
            "correct_answer_effect": "unchanged",
            "distractor_effect": "unchanged",
            "answer_explanation_effect": "unchanged",
            "noop": noop,
            "counterfactual": cf or {"stem": None, "correct_answer": None, "distractors": []},
        },
    }


def _comparison(
    *,
    query: str,
    noop: Dict[str, Any],
    cf: Optional[Dict[str, Any]],
    operation: str = "WHATIF",
    target: str = "a:v1",
    replacement: str = "b:v1",
    checkpoint: str = "ckpt-1",
    evidence_status: str = "no_demonstrated_effect",
    decision_cs: bool = True,
    criteria_cs: bool = True,
    content_cs: bool = True,
    qcc_status: str = "question_content_unchanged",
    cf_present: bool = True,
) -> Dict[str, Any]:
    status = qcc_status
    if cf is None or not cf_present:
        status = "question_not_reproduced"
        cf_present = False
    return {
        "generated_query": query,
        "evidence_status": evidence_status,
        "decision_control_stable": decision_cs,
        "criteria_control_stable": criteria_cs,
        "subject_content_control_stable": content_cs,
        "intervention_signature": {
            "operation": operation,
            "target_ref": {"key": target},
            "replacement_ref": {"key": replacement} if replacement else None,
            "checkpoint_id": checkpoint,
        },
        "question_content_comparison": _qcc(
            noop, cf, status=status, present=cf_present
        ),
    }


def _attempt(op: str, query: str, *, success: bool = True) -> Dict[str, Any]:
    return {
        "parsed_operation": op,
        "generated_query": query,
        "counted_as_successful_round": success,
        "validation_status": "executed" if success else "duplicate_error",
        "executed": success,
        "produced_counterfactual": op in {"REMOVE", "ADD", "WHATIF"} and success,
    }


def _case(
    candidate: str,
    attempts: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
    query_rounds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "interrogation_attempts": attempts,
        "replay_comparisons": comparisons,
    }
    if query_rounds is not None:
        raw["query_rounds"] = query_rounds
    return {
        "candidate_ref": candidate,
        "conditions": {"axis": {"raw_evidence": raw}},
    }


def _payload(*cases: Dict[str, Any]) -> Dict[str, Any]:
    return {"cases": list(cases)}


def _query_text(cmp: Mapping[str, Any]) -> str:
    sig = cmp.get("intervention_signature") or {}
    op = str(sig.get("operation") or "REMOVE")
    target = (sig.get("target_ref") or {}).get("key") or "a:v1"
    repl = ((sig.get("replacement_ref") or {}) or {}).get("key")
    boundary = sig.get("boundary_raw") or "before x#1"
    if op == "WHATIF":
        return f"WHATIF({target}, {repl or 'b:v1'}, {boundary})"
    return f"{op}({target}, {boundary})"


def _strip_query(cmp: Mapping[str, Any]) -> Dict[str, Any]:
    copy = json.loads(json.dumps(cmp))
    copy.pop("generated_query", None)
    return copy


def _query_round_from_comparison(cmp: Mapping[str, Any]) -> Dict[str, Any]:
    sig = dict(cmp.get("intervention_signature") or {})
    nested = _strip_query(cmp)
    return {
        "kind": "intervention",
        "generated_query": str(cmp.get("generated_query") or _query_text(cmp)),
        "signature": sig,
        "operation": sig.get("operation"),
        "target_ref": sig.get("target_ref"),
        "replacement_ref": sig.get("replacement_ref"),
        "checkpoint_id": sig.get("checkpoint_id"),
        "replay_comparison": nested,
    }


def _nested_export_case(
    candidate: str,
    attempts: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
    *,
    also_flat_query_rounds: bool = False,
) -> Dict[str, Any]:
    top_cmps = [_strip_query(c) for c in comparisons]
    qrs = [_query_round_from_comparison(c) for c in comparisons]
    raw: Dict[str, Any] = {
        "interrogation_attempts": attempts,
        "replay_comparisons": top_cmps,
        "axis_report": {
            "query_rounds": [json.loads(json.dumps(qr)) for qr in qrs],
            "interrogation_attempts": list(attempts),
            "replay_comparisons": [json.loads(json.dumps(c)) for c in top_cmps],
        },
    }
    if also_flat_query_rounds:
        raw["query_rounds"] = qrs
    return {"candidate_ref": candidate, "conditions": {"axis": {"raw_evidence": raw}}}


def _two_question_nested_fixture() -> Dict[str, Any]:
    """2 questions × 5 CFs; 6 eligible pairs; 4 non-reproduced. Nested axis_report only."""
    cases = []
    for q_i, cand in enumerate(("q_00000001:v1", "q_00000002:v1"), start=1):
        attempts: List[Dict[str, Any]] = []
        comparisons: List[Dict[str, Any]] = []
        for i in range(5):
            reproduced = i < 3
            target = "bp_00000001:v1" if i < 2 else f"chunk_0000000{i}:v1"
            ckpt = f"ckpt-q{q_i}-{i}"
            query = f"REMOVE({target}, before gen#{i})"
            attempts.append(_attempt("REMOVE", query))
            attempts.append(
                _attempt("REMOVE", f"REMOVE({target}, after plan#{i})", success=False)
            )
            if reproduced:
                cmp = _comparison(
                    query=query,
                    noop=_mcq(f"Q{q_i} stem {i} unsupervised learning goal?"),
                    cf=_mcq(f"Q{q_i} stem {i} unsupervised learning aim?"),
                    operation="REMOVE",
                    target=target,
                    replacement="",
                    checkpoint=ckpt,
                )
            else:
                cmp = _comparison(
                    query=query,
                    noop=_mcq(f"Q{q_i} stem {i} unsupervised learning goal?"),
                    cf=None,
                    cf_present=False,
                    operation="REMOVE",
                    target=target,
                    replacement="",
                    checkpoint=ckpt,
                )
            comparisons.append(cmp)
        cases.append(_nested_export_case(cand, attempts, comparisons))
    return _payload(*cases)


def _eligible_pair(
    *,
    candidate: str = "q:v1",
    query: str = "WHATIF(a:v1, b:v1, before x#1)",
    noop_stem: str = "What is the main goal of unsupervised learning?",
    cf_stem: str = "What does unsupervised learning aim to achieve?",
    cf_distractors: Optional[List[str]] = None,
    **cmp_kw: Any,
) -> Dict[str, Any]:
    cmp = _comparison(
        query=query,
        noop=_mcq(noop_stem),
        cf=_mcq(cf_stem, distractors=cf_distractors),
        **cmp_kw,
    )
    return _payload(
        _case(
            candidate,
            [_attempt("WHAT", "WHAT(a:v1)"), _attempt("WHATIF", query)],
            [cmp],
        )
    )


class RecordingGenerator:
    def __init__(self, responder=None) -> None:
        self.prompts: List[str] = []
        self.responder = responder or (lambda _prompt: _json_resp("same_concept"))

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responder(prompt)


class TestConceptChangeCliAndFingerprint(unittest.TestCase):
    def test_flag_defaults_off_and_cli_accepts(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--lecturebank-dir", "x", "--id", "1", "--allow-any-count"]
        )
        self.assertFalse(args.evaluate_concept_change)
        on = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "1",
                "--allow-any-count",
                "--evaluate-concept-change",
            ]
        )
        self.assertTrue(on.evaluate_concept_change)

    def test_generation_config_fingerprint_excludes_eval_flag(self):
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
                "evaluate_concept_change": True,
                "concept_change_from_saved": True,
                "reevaluate_concept_change": True,
                "export_streamlit_json": True,
            }
        )
        self.assertEqual(off, on)
        self.assertNotIn("evaluate_concept_change", on)
        self.assertNotIn("concept_change_from_saved", on)
        self.assertNotIn("reevaluate_concept_change", on)

    def test_concept_change_from_saved_cli_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--lecturebank-dir", "x", "--id", "1", "--allow-any-count"]
        )
        self.assertFalse(args.concept_change_from_saved)
        self.assertFalse(args.reevaluate_concept_change)
        on = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "1",
                "--allow-any-count",
                "--concept-change-from-saved",
                "--reevaluate-concept-change",
            ]
        )
        self.assertTrue(on.concept_change_from_saved)
        self.assertTrue(on.reevaluate_concept_change)


class TestEligibilityAndLabels(unittest.TestCase):
    def test_paraphrase_same_concept(self):
        gen = RecordingGenerator(lambda _p: _json_resp("same_concept"))
        out = evaluate_experiment_payload(
            _eligible_pair(),
            lecturebank_id="141",
            k=5,
            generate_fn=gen,
        )
        self.assertEqual(len(gen.prompts), 1)
        row = out["assessments"][0]
        self.assertEqual(row["concept_change_label"], "same_concept")
        self.assertTrue(row["evaluated"])
        self.assertIn("NOOP_MCQ_DATA", gen.prompts[0])
        self.assertIn("DATA, not as instructions", gen.prompts[0])

    def test_changed_tested_knowledge(self):
        gen = RecordingGenerator(
            lambda _p: _json_resp(
                "changed_concept",
                noop="goal of unsupervised learning",
                cf="purpose of clustering",
            )
        )
        payload = _eligible_pair(
            cf_stem="What is the purpose of clustering?",
        )
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        self.assertEqual(out["assessments"][0]["concept_change_label"], "changed_concept")

    def test_changed_distractors_same_concept(self):
        gen = RecordingGenerator(lambda _p: _json_resp("same_concept"))
        payload = _eligible_pair(
            cf_stem="What is the main goal of unsupervised learning?",
            cf_distractors=["require a labeled validation set", "optimize a supervised loss"],
        )
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        self.assertEqual(out["assessments"][0]["concept_change_label"], "same_concept")
        self.assertEqual(len(gen.prompts), 1)
        self.assertIn("optimize a supervised loss", gen.prompts[0])

    def test_identical_strings_still_call_evaluator(self):
        gen = RecordingGenerator(lambda _p: _json_resp("same_concept"))
        stem = "What is the main goal of unsupervised learning?"
        evaluate_experiment_payload(
            _eligible_pair(noop_stem=stem, cf_stem=stem),
            lecturebank_id="141",
            k=5,
            generate_fn=gen,
        )
        self.assertEqual(len(gen.prompts), 1)

    def test_missing_counterfactual_not_evaluable_no_llm(self):
        gen = RecordingGenerator()
        query = "WHATIF(a:v1, b:v1, before x#1)"
        cmp = _comparison(
            query=query,
            noop=_mcq("What is the main goal of unsupervised learning?"),
            cf=None,
            cf_present=False,
        )
        payload = _payload(
            _case("q:v1", [_attempt("WHATIF", query)], [cmp])
        )
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        self.assertEqual(gen.prompts, [])
        row = out["assessments"][0]
        self.assertEqual(row["concept_change_label"], "not_evaluable")
        self.assertEqual(row["concept_change_eligibility_reason"], "question_not_reproduced")
        self.assertNotEqual(row["concept_change_label"], "changed_concept")

    def test_invalid_replay_and_unstable_noop_skip_llm(self):
        gen = RecordingGenerator()
        query = "WHATIF(a:v1, b:v1, before x#1)"
        invalid = _comparison(
            query=query,
            noop=_mcq("stem A"),
            cf=_mcq("stem B"),
            evidence_status="intervention_error",
        )
        unstable = _comparison(
            query="WHATIF(a:v1, c:v1, before x#1)",
            noop=_mcq("stem A"),
            cf=_mcq("stem B"),
            replacement="c:v1",
            content_cs=False,
        )
        for cmp, reason in (
            (invalid, "invalid_comparison"),
            (unstable, "noop_unstable"),
        ):
            gen.prompts.clear()
            eligible, got = concept_change_eligibility(cmp)
            self.assertFalse(eligible)
            self.assertEqual(got, reason)
            out = assess_comparison(cmp, generate_fn=gen)
            self.assertEqual(gen.prompts, [])
            self.assertEqual(out["concept_change_label"], "not_evaluable")
            self.assertEqual(out["concept_change_eligibility_reason"], reason)

    def test_uncertain_not_assessed(self):
        gen = RecordingGenerator(lambda _p: _json_resp("uncertain"))
        out = evaluate_experiment_payload(
            _eligible_pair(), lecturebank_id="141", k=5, generate_fn=gen
        )
        row = out["assessments"][0]
        self.assertEqual(row["concept_change_label"], "uncertain")
        q = out["question_rows"][0]
        self.assertEqual(q["concept_uncertain_count"], 1)
        self.assertEqual(q["concept_assessed_count"], 0)
        self.assertIsNone(q["concept_change_rate"])

    def test_malformed_response_is_evaluation_error(self):
        gen = RecordingGenerator(lambda _p: "not-json {")
        out = evaluate_experiment_payload(
            _eligible_pair(), lecturebank_id="141", k=5, generate_fn=gen
        )
        row = out["assessments"][0]
        self.assertEqual(row["concept_change_label"], "evaluation_error")
        self.assertNotIn(row["concept_change_label"], {"same_concept", "changed_concept"})
        with self.assertRaises(ValueError):
            parse_concept_change_response('{"concept_change_label": "same_concept"}')

    def test_api_failure_is_evaluation_error(self):
        def boom(_prompt: str) -> str:
            raise RuntimeError("provider down")

        out = evaluate_experiment_payload(
            _eligible_pair(), lecturebank_id="141", k=5, generate_fn=boom
        )
        row = out["assessments"][0]
        self.assertEqual(row["concept_change_label"], "evaluation_error")
        self.assertIn("provider down", str(row["concept_change_reason"]))


class TestAggregationCacheAndExports(unittest.TestCase):
    def test_zero_denominator_rate_is_na(self):
        row = compute_concept_change_for_question(
            lecturebank_id="141",
            candidate_ref="q:v1",
            k=5,
            assessment_rows=[
                {"concept_change_label": "not_evaluable"},
                {"concept_change_label": "uncertain"},
                {"concept_change_label": "evaluation_error"},
            ],
            selected_count=3,
        )
        self.assertEqual(row["concept_assessed_count"], 0)
        self.assertIsNone(row["concept_change_rate"])
        agg = aggregate_concept_change([row])
        self.assertIsNone(agg["concept_change_rate_macro"])
        self.assertIsNone(agg["concept_change_rate_micro"])

    def test_macro_micro_different_question_denominators(self):
        q1 = compute_concept_change_for_question(
            lecturebank_id="141",
            candidate_ref="q1:v1",
            k=5,
            assessment_rows=[{"concept_change_label": "changed_concept"}],
            selected_count=1,
        )
        q2 = compute_concept_change_for_question(
            lecturebank_id="141",
            candidate_ref="q2:v1",
            k=5,
            assessment_rows=[
                {"concept_change_label": "changed_concept"},
                {"concept_change_label": "same_concept"},
                {"concept_change_label": "same_concept"},
            ],
            selected_count=3,
        )
        self.assertEqual(q1["concept_change_rate"], 1.0)
        self.assertAlmostEqual(q2["concept_change_rate"], 1.0 / 3.0)
        agg = aggregate_concept_change([q1, q2])
        self.assertAlmostEqual(agg["concept_change_rate_macro"], (1.0 + 1.0 / 3.0) / 2.0)
        self.assertEqual(agg["concept_change_rate_micro"], 0.5)
        self.assertEqual(agg["concept_change_rate_micro_num"], 2)
        self.assertEqual(agg["concept_change_rate_micro_den"], 4)

    def test_signature_deduplication_within_k(self):
        gen = RecordingGenerator()
        query = "WHATIF(a:v1, b:v1, before x#1)"
        cmp = _comparison(
            query=query,
            noop=_mcq("What is the main goal of unsupervised learning?"),
            cf=_mcq("What does unsupervised learning aim to achieve?"),
        )
        payload = _payload(
            _case(
                "q:v1",
                [
                    _attempt("WHATIF", query),
                    _attempt("WHATIF", query),
                    _attempt("WHATIF", query, success=False),
                ],
                [cmp],
            )
        )
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        self.assertEqual(len(out["assessments"]), 1)
        self.assertEqual(len(gen.prompts), 1)
        self.assertEqual(out["question_rows"][0]["axis_counterfactual_selected_at_k"], 1)

    def test_cache_reuse_and_invalidation(self):
        cache = ConceptChangeCache()
        cfg = ConceptChangeEvaluatorConfig()
        gen = RecordingGenerator()
        payload = _eligible_pair()
        first = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen, config=cfg, cache=cache
        )
        self.assertEqual(len(gen.prompts), 1)
        self.assertFalse(first["assessments"][0]["cache_hit"])
        second = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen, config=cfg, cache=cache
        )
        self.assertEqual(len(gen.prompts), 1)
        self.assertTrue(second["assessments"][0]["cache_hit"])
        other = ConceptChangeEvaluatorConfig(model="other-model", rubric_version="concept_change_v2")
        self.assertNotEqual(
            concept_change_cache_key(
                noop=_mcq("a"), cf=_mcq("b"), config=cfg
            ),
            concept_change_cache_key(
                noop=_mcq("a"), cf=_mcq("b"), config=other
            ),
        )
        evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen, config=other, cache=cache
        )
        self.assertEqual(len(gen.prompts), 2)

    def test_default_off_does_not_call_evaluator(self):
        gen = RecordingGenerator()
        extracted: Dict[str, Any] = {}
        _apply_concept_change_evaluation(
            payload=_eligible_pair(),
            extracted=extracted,
            lecturebank_id="141",
            args=Namespace(evaluate_concept_change=False, _concept_change_generate_fn=gen),
            gen_cfg={"max_rounds": 5},
            metrics_dir=Path("."),
            log=io.StringIO(),
        )
        self.assertEqual(gen.prompts, [])
        self.assertNotIn("concept_change_assessments", extracted)

    def test_maybe_evaluate_does_not_build_llm_when_ineligible(self):
        query = "WHATIF(a:v1, b:v1, before x#1)"
        payload = _payload(
            _case(
                "q:v1",
                [_attempt("WHATIF", query)],
                [
                    _comparison(
                        query=query,
                        noop=_mcq("stem"),
                        cf=None,
                        cf_present=False,
                    )
                ],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "mas_langgraph.evaluation.concept_change.build_concept_change_llm"
            ) as built:
                out = maybe_evaluate_concept_change(
                    payload,
                    lecturebank_id="141",
                    k=5,
                    metrics_dir=Path(tmp),
                    cache=ConceptChangeCache(),
                )
        built.assert_not_called()
        self.assertEqual(out["llm_calls"], 0)
        self.assertEqual(out["assessments"][0]["concept_change_label"], "not_evaluable")

    def test_historical_overall_metrics_are_na_not_zero(self):
        rows = overall_metric_rows([], [], [], [], {"concept_change_ran": False})
        by_name = {r["metric"]: r for r in rows}
        self.assertIsNone(by_name["concept_change_rate_macro"]["value"])
        self.assertEqual(by_name["concept_change_rate_macro"]["unit"], "NA")
        self.assertIsNone(by_name["concept_change_rate_micro"]["value"])
        self.assertNotIn("concept_changed_count", by_name)

    def test_historical_csv_load_without_assessments(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp)
            write_metrics_outputs(
                metrics,
                document_rows=[
                    {
                        "lecturebank_id": "141",
                        "eval_status": "success",
                        "accepted_question_count": 1,
                    }
                ],
                attempt_rows=[],
                comparison_rows=[
                    {
                        "lecturebank_id": "141",
                        "candidate_ref": "q:v1",
                        "content_effect": True,
                    }
                ],
                provenance_rows=[],
                distribution_rows=[],
                round_rows=[],
                overall_rows=overall_metric_rows(
                    [], [], [], [], {"concept_change_ran": False}
                ),
                report_markdown="# hist\n",
            )
            collected = load_collected_from_metrics(metrics)
            self.assertEqual(len(collected), 1)
            self.assertEqual(collected[0]["concept_change_assessments"], [])
            self.assertEqual(collected[0]["concept_change_by_question"], [])
            cmp = collected[0]["replay_comparisons"][0]
            self.assertIsNone(cmp.get("concept_change_label"))
            self.assertTrue(cmp.get("content_effect"))
            hist_report = build_report(
                catalog_rows=[],
                document_rows=[],
                attempt_rows=[],
                comparison_rows=[{"lecturebank_id": "141", "content_effect": True}],
                provenance_rows=[],
                distribution_rows=[],
                round_rows=[],
                overall_rows=overall_metric_rows([], [], [], [], {"concept_change_ran": False}),
                pooled={"concept_change_ran": False},
            )
            self.assertIn("not assessed", hist_report.lower())

    def test_existing_effect_metrics_unchanged(self):
        payload = _eligible_pair(
            qcc_status="question_content_changed",
            cf_stem="What is the purpose of clustering?",
        )
        before = extract_experiment_metrics(payload, lecturebank_id="141")
        gen = RecordingGenerator(lambda _p: _json_resp("changed_concept"))
        evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        after = extract_experiment_metrics(payload, lecturebank_id="141")
        b = before["replay_comparisons"][0]
        a = after["replay_comparisons"][0]
        self.assertEqual(b["content_effect"], a["content_effect"])
        self.assertEqual(b["content_effect_raw"], a["content_effect_raw"])
        self.assertEqual(b["stem_effect"], a["stem_effect"])
        self.assertEqual(b["question_content_effect_status"], a["question_content_effect_status"])
        self.assertTrue(b["content_effect_raw"])

    def test_exports_and_report_and_saved_json(self):
        gen = RecordingGenerator(lambda _p: _json_resp("changed_concept"))
        payload = _eligible_pair()
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        metric_rows = extract_experiment_metrics(payload, lecturebank_id="141")[
            "replay_comparisons"
        ]
        attached = attach_assessments_to_metric_rows(metric_rows, out["assessments"])
        self.assertEqual(attached[0]["concept_change_label"], "changed_concept")
        self.assertEqual(attached[0]["noop_tested_concept"], "goal of unsupervised learning")
        cmp = payload["cases"][0]["conditions"]["axis"]["raw_evidence"]["replay_comparisons"][0]
        self.assertEqual(
            cmp["concept_change_evaluation"]["assessment_kind"],
            "llm_assessed_concept_change",
        )
        pooled = {
            **aggregate_concept_change(out["question_rows"]),
            "k": 5,
        }
        overall = overall_metric_rows([], [], attached, [], pooled)
        by_name = {r["metric"]: r for r in overall}
        self.assertEqual(by_name["concept_changed_count"]["value"], 1)
        report = build_report(
            catalog_rows=[],
            document_rows=[],
            attempt_rows=[],
            comparison_rows=attached,
            provenance_rows=[],
            distribution_rows=[],
            round_rows=[],
            overall_rows=overall,
            pooled=pooled,
            concept_change_question_rows=out["question_rows"],
        )
        self.assertIn("LLM-assessed tested-concept change", report)
        self.assertIn("conditional on assessable comparisons", report)
        self.assertIn("not semantic Precision@K or Recall@K", report)
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp)
            write_metrics_outputs(
                metrics,
                document_rows=[{"lecturebank_id": "141"}],
                attempt_rows=[],
                comparison_rows=attached,
                provenance_rows=[],
                distribution_rows=[],
                round_rows=[],
                overall_rows=overall,
                report_markdown=report,
                concept_change_assessment_rows=out["assessments"],
                concept_change_question_rows=out["question_rows"],
            )
            self.assertTrue((metrics / "concept_change_assessments.csv").is_file())
            self.assertTrue((metrics / "concept_change_by_question.csv").is_file())
            saved = Path(tmp) / "quiz_explanation_experiment_141.json"
            saved.write_text(json.dumps(_eligible_pair()), encoding="utf-8")
            evaluate_saved_experiment_json(
                saved,
                lecturebank_id="141",
                k=5,
                generate_fn=gen,
                write_back=True,
            )
            reloaded = json.loads(saved.read_text(encoding="utf-8"))
            cmp2 = reloaded["cases"][0]["conditions"]["axis"]["raw_evidence"][
                "replay_comparisons"
            ][0]
            self.assertIn("concept_change_evaluation", cmp2)
            self.assertEqual(cmp2["concept_change_label"], "changed_concept")

    def test_apply_flag_on_mutates_extracted(self):
        gen = RecordingGenerator(lambda _p: _json_resp("same_concept"))
        extracted: Dict[str, Any] = {"replay_comparisons": []}
        with tempfile.TemporaryDirectory() as tmp:
            _apply_concept_change_evaluation(
                payload=_eligible_pair(),
                extracted=extracted,
                lecturebank_id="141",
                args=Namespace(
                    evaluate_concept_change=True, _concept_change_generate_fn=gen
                ),
                gen_cfg={"max_rounds": 5},
                metrics_dir=Path(tmp),
                log=io.StringIO(),
            )
        self.assertEqual(len(extracted["concept_change_assessments"]), 1)
        self.assertEqual(extracted["concept_change_assessments"][0]["concept_change_label"], "same_concept")
        self.assertEqual(len(gen.prompts), 1)


class TestNestedExportMatchingAndRepair(unittest.TestCase):
    def _mock_six_eligible(self) -> RecordingGenerator:
        labels = ["changed_concept", "same_concept", "changed_concept"] * 2

        def responder(_prompt: str) -> str:
            idx = len(self._used)
            self._used.append(idx)
            label = labels[idx]
            return _json_resp(
                label,
                noop=f"concept-{idx}",
                cf=f"concept-{idx}" if label == "same_concept" else f"other-{idx}",
            )

        self._used: List[int] = []
        return RecordingGenerator(responder)

    def test_nested_axis_report_query_rounds_selects_ten(self):
        payload = _two_question_nested_fixture()
        gen = self._mock_six_eligible()
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        self.assertEqual(len(out["assessments"]), 10)
        self.assertEqual(len(gen.prompts), 6)
        by_q = {r["candidate_ref"]: r for r in out["question_rows"]}
        self.assertEqual(by_q["q_00000001:v1"]["axis_counterfactual_selected_at_k"], 5)
        self.assertEqual(by_q["q_00000002:v1"]["axis_counterfactual_selected_at_k"], 5)
        not_eval = [r for r in out["assessments"] if r["concept_change_label"] == "not_evaluable"]
        self.assertEqual(len(not_eval), 4)
        self.assertTrue(
            all(r["concept_change_eligibility_reason"] == "question_not_reproduced" for r in not_eval)
        )
        eligible = [
            r
            for r in out["assessments"]
            if r["concept_change_label"] in {"same_concept", "changed_concept", "uncertain", "evaluation_error"}
        ]
        self.assertEqual(len(eligible), 6)
        agg = out["aggregate"]
        self.assertEqual(agg["concept_changed_count"], 4)
        self.assertEqual(agg["concept_same_count"], 2)
        self.assertEqual(agg["concept_not_evaluable_count"], 4)
        self.assertEqual(agg["concept_assessed_count"], 6)
        self.assertAlmostEqual(agg["concept_change_rate_micro"], 4 / 6)
        raw = payload["cases"][0]["conditions"]["axis"]["raw_evidence"]
        top = raw["replay_comparisons"][0]
        nested = raw["axis_report"]["replay_comparisons"][0]
        qr_cmp = raw["axis_report"]["query_rounds"][0]["replay_comparison"]
        self.assertEqual(top.get("concept_change_label"), nested.get("concept_change_label"))
        self.assertIn("concept_change_evaluation", qr_cmp)

    def test_flat_historical_payload_still_works(self):
        gen = RecordingGenerator(lambda _p: _json_resp("same_concept"))
        out = evaluate_experiment_payload(
            _eligible_pair(), lecturebank_id="141", k=5, generate_fn=gen
        )
        self.assertEqual(len(out["assessments"]), 1)
        self.assertEqual(len(gen.prompts), 1)

    def test_duplicate_toplevel_and_nested_not_double_counted(self):
        payload = _two_question_nested_fixture()
        for case in payload["cases"]:
            raw = case["conditions"]["axis"]["raw_evidence"]
            raw["query_rounds"] = json.loads(json.dumps(raw["axis_report"]["query_rounds"]))
        gen = self._mock_six_eligible()
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        self.assertEqual(len(out["assessments"]), 10)
        self.assertEqual(sum(r["axis_counterfactual_selected_at_k"] for r in out["question_rows"]), 10)

    def test_same_target_different_checkpoints_are_distinct(self):
        payload = _two_question_nested_fixture()
        gen = self._mock_six_eligible()
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        q1 = [r for r in out["assessments"] if r["candidate_ref"] == "q_00000001:v1"]
        sigs = [r["intervention_signature"] for r in q1]
        self.assertEqual(len(sigs), len(set(sigs)))
        self.assertTrue(any("ckpt-q1-0" in s for s in sigs))
        self.assertTrue(any("ckpt-q1-1" in s for s in sigs))

    def test_unresolved_mapping_is_diagnostic_not_silent_zero(self):
        query = "REMOVE(a:v1, before x#1)"
        cmp = _comparison(
            query=query,
            noop=_mcq("stem"),
            cf=_mcq("stem2"),
            operation="REMOVE",
            target="a:v1",
            replacement="",
            checkpoint="ckpt-z",
        )
        cmp.pop("generated_query", None)
        payload = _payload(
            _case("q:v1", [_attempt("REMOVE", query)], [cmp])
        )
        gen = RecordingGenerator()
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        self.assertGreaterEqual(out["question_rows"][0]["axis_counterfactual_selected_at_k"], 1)
        self.assertNotEqual(out["question_rows"][0]["axis_counterfactual_selected_at_k"], 0)
        self.assertEqual(out["question_rows"][0]["mapping_status"], "unresolved")
        self.assertIn("unresolved_mapping", out["question_rows"][0]["mapping_diagnostics"])
        self.assertEqual(gen.prompts, [])

    def test_k_limit_and_signature_dedup(self):
        payload = _two_question_nested_fixture()
        gen = RecordingGenerator(lambda _p: _json_resp("changed_concept"))
        out = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=2, generate_fn=gen
        )
        self.assertEqual(out["question_rows"][0]["axis_counterfactual_selected_at_k"], 2)
        self.assertEqual(out["question_rows"][1]["axis_counterfactual_selected_at_k"], 2)
        self.assertEqual(len(out["assessments"]), 4)
        raw = payload["cases"][0]["conditions"]["axis"]["raw_evidence"]
        extra = list(raw["interrogation_attempts"])
        extra.insert(0, extra[0])
        raw["interrogation_attempts"] = extra
        gen2 = RecordingGenerator(lambda _p: _json_resp("changed_concept"))
        out2 = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen2
        )
        self.assertEqual(out2["question_rows"][0]["axis_counterfactual_selected_at_k"], 5)

    def test_repair_zero_selection_partial_and_config_invalidation(self):
        payload = _two_question_nested_fixture()
        zero_rows = [
            compute_concept_change_for_question(
                lecturebank_id="141",
                candidate_ref=cand,
                k=5,
                assessment_rows=[],
                selected_count=0,
            )
            for cand in ("q_00000001:v1", "q_00000002:v1")
        ]
        cfg = ConceptChangeEvaluatorConfig()
        complete, reason = persisted_concept_change_complete(
            payload,
            assessments=[],
            question_rows=zero_rows,
            config=cfg,
            k=5,
        )
        self.assertFalse(complete)
        self.assertEqual(reason, "zero_selection_with_interventions")

        gen = self._mock_six_eligible()
        first = evaluate_experiment_payload(
            payload, lecturebank_id="141", k=5, generate_fn=gen
        )
        partial = list(first["assessments"][:3])
        complete, reason = persisted_concept_change_complete(
            payload,
            assessments=partial,
            question_rows=first["question_rows"],
            config=cfg,
            k=5,
        )
        self.assertFalse(complete)
        self.assertEqual(reason, "partial_assessments")
        gen_partial = RecordingGenerator(lambda _p: _json_resp("same_concept"))
        merged = evaluate_experiment_payload(
            payload,
            lecturebank_id="141",
            k=5,
            generate_fn=gen_partial,
            existing_assessments=partial,
        )
        self.assertEqual(len(merged["assessments"]), 10)
        self.assertEqual(merged["reused_assessments"], 3)
        self.assertEqual(len(gen_partial.prompts), 3)

        other = ConceptChangeEvaluatorConfig(model="other-model")
        complete, reason = persisted_concept_change_complete(
            payload,
            assessments=first["assessments"],
            question_rows=first["question_rows"],
            config=other,
            k=5,
        )
        self.assertFalse(complete)
        self.assertEqual(reason, "evaluator_config_mismatch")

        no_cf = _payload(
            _nested_export_case(
                "q:v1",
                [_attempt("DONE", "DONE", success=False)],
                [],
            )
        )
        complete, reason = persisted_concept_change_complete(
            no_cf,
            assessments=[],
            question_rows=[
                compute_concept_change_for_question(
                    lecturebank_id="141",
                    candidate_ref="q:v1",
                    k=5,
                    assessment_rows=[],
                    selected_count=0,
                )
            ],
            config=cfg,
            k=5,
        )
        self.assertTrue(complete)
        self.assertEqual(reason, "complete_no_interventions")

    def test_eval_only_repair_updates_outputs_without_generation(self):
        payload = _two_question_nested_fixture()
        extracted = extract_experiment_metrics(payload, lecturebank_id="141")
        zero_rows = [
            compute_concept_change_for_question(
                lecturebank_id="141",
                candidate_ref=cand,
                k=5,
                assessment_rows=[],
                selected_count=0,
            )
            for cand in ("q_00000001:v1", "q_00000002:v1")
        ]
        item = {
            "lecturebank_id": "141",
            "eval_status": "success",
            "document_metrics": {
                **dict(extracted.get("document_metrics") or {}),
                "lecturebank_id": "141",
                "eval_status": "success",
            },
            "replay_comparisons": list(extracted.get("replay_comparisons") or []),
            "interrogation_attempts": list(extracted.get("interrogation_attempts") or []),
            "concept_change_assessments": [],
            "concept_change_by_question": zero_rows,
        }
        gen = self._mock_six_eligible()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics_dir = output_dir / "metrics"
            write_streamlit_experiment_json(output_dir, "141", payload)
            _persist(
                output_dir,
                metrics_dir,
                output_dir / "lecturebank_eval_manifest.json",
                {"manifest_version": 1, "records": []},
                [item],
            )
            collected = load_collected_from_metrics(metrics_dir)
            self.assertEqual(
                int(collected[0]["concept_change_by_question"][0]["axis_counterfactual_selected_at_k"] or 0),
                0,
            )
            with patch("mas_langgraph.evaluation.execute.process_documents") as gen_run:
                stats = run_concept_change_from_saved(
                    selected=["141"],
                    args=Namespace(
                        evaluate_concept_change=True,
                        reevaluate_concept_change=False,
                        _concept_change_generate_fn=gen,
                    ),
                    output_dir=output_dir,
                    metrics_dir=metrics_dir,
                    gen_cfg={"max_rounds": 5},
                    collected=collected,
                    persist=lambda *a, **k: _persist(*a, **k),
                    manifest_path=output_dir / "lecturebank_eval_manifest.json",
                    manifest={"manifest_version": 1, "records": []},
                    log=io.StringIO(),
                )
            gen_run.assert_not_called()
            self.assertEqual(stats["failures"], 0)
            self.assertEqual(len(gen.prompts), 6)
            reloaded = load_collected_from_metrics(metrics_dir)
            qrows = reloaded[0]["concept_change_by_question"]
            self.assertEqual(sum(int(r["axis_counterfactual_selected_at_k"]) for r in qrows), 10)
            assessments = reloaded[0]["concept_change_assessments"]
            self.assertEqual(len(assessments), 10)
            cmp_rows = reloaded[0]["replay_comparisons"]
            labeled = [r for r in cmp_rows if r.get("concept_change_label")]
            self.assertEqual(len(labeled), 10)
            overall = (metrics_dir / "overall_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("concept_change_rate_micro", overall)
            report = (metrics_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("LLM-assessed tested-concept change", report)
            saved = json.loads(
                (output_dir / "streamlit" / "quiz_explanation_experiment_141.json").read_text(
                    encoding="utf-8"
                )
            )
            top = saved["cases"][0]["conditions"]["axis"]["raw_evidence"]["replay_comparisons"][0]
            self.assertIn("concept_change_evaluation", top)
            n_first = len(assessments)
            gen2 = RecordingGenerator(lambda _p: _json_resp("changed_concept"))
            run_concept_change_from_saved(
                selected=["141"],
                args=Namespace(
                    evaluate_concept_change=True,
                    reevaluate_concept_change=False,
                    _concept_change_generate_fn=gen2,
                ),
                output_dir=output_dir,
                metrics_dir=metrics_dir,
                gen_cfg={"max_rounds": 5},
                collected=load_collected_from_metrics(metrics_dir),
                persist=lambda *a, **k: _persist(*a, **k),
                manifest_path=output_dir / "lecturebank_eval_manifest.json",
                manifest={"manifest_version": 1, "records": []},
                log=io.StringIO(),
            )
            again = load_collected_from_metrics(metrics_dir)
            self.assertEqual(len(again[0]["concept_change_assessments"]), n_first)
            self.assertEqual(gen2.prompts, [])

    def test_eval_only_fails_without_saved_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics_dir = output_dir / "metrics"
            item = {
                "lecturebank_id": "141",
                "eval_status": "success",
                "document_metrics": {"lecturebank_id": "141", "eval_status": "success"},
                "replay_comparisons": [],
            }
            log = io.StringIO()
            with patch("mas_langgraph.evaluation.execute.process_documents") as gen_run:
                stats = run_concept_change_from_saved(
                    selected=["141"],
                    args=Namespace(evaluate_concept_change=True, _concept_change_generate_fn=RecordingGenerator()),
                    output_dir=output_dir,
                    metrics_dir=metrics_dir,
                    gen_cfg={"max_rounds": 5},
                    collected=[item],
                    persist=lambda *a, **k: None,
                    manifest_path=output_dir / "manifest.json",
                    manifest={},
                    log=log,
                )
            gen_run.assert_not_called()
            self.assertEqual(stats["failures"], 1)
            self.assertIn("will not generate or replay", log.getvalue())


if __name__ == "__main__":
    unittest.main()
