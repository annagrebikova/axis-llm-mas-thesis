"""SAQUET per-MAS aggregation. Mocked criteria only; no live SAQUET or LLM."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping
from unittest.mock import patch

from mas_langgraph.batch.manifest import generation_config_fingerprint
from mas_langgraph.evaluation.axis_metrics import NA
from mas_langgraph.evaluation.execute import run_saquet_from_saved
from mas_langgraph.evaluation.lifecycle import load_collected_from_metrics
from mas_langgraph.evaluation.report import build_report, write_metrics_outputs
from mas_langgraph.evaluation.runner import _persist, build_arg_parser
from mas_langgraph.evaluation.saquet_eval import (
    SAQUET_CONFIG_VERSION,
    SAQUET_CRITERIA,
    SAQUET_MAS_FIELDS,
    SaquetQuestionCache,
    _iwf_flag,
    aggregate_saquet_by_mas,
    default_apply_saquet_criteria,
    default_saquet_model,
    empty_saquet_mas_row,
    evaluate_document_saquet,
    evaluate_quiz_row,
    persist_factual_quiz,
    saquet_criterion_fields,
)
from mas_langgraph.evaluation.streamlit_export import write_streamlit_experiment_json


def _quiz_row(
    *,
    stem: str = "What is X?",
    options: List[str] | None = None,
    correct: str | None = None,
    ref: str = "q_00000001:v1",
    lecturebank_id: str = "141",
) -> Dict[str, Any]:
    opts = list(options or ["alpha", "beta", "gamma", "delta"])
    correct_text = correct if correct is not None else opts[0]
    return {
        "lecturebank_id": lecturebank_id,
        "candidate_ref": ref,
        "stem": stem,
        "option_A": opts[0] if len(opts) > 0 else "",
        "option_B": opts[1] if len(opts) > 1 else "",
        "option_C": opts[2] if len(opts) > 2 else "",
        "option_D": opts[3] if len(opts) > 3 else "",
        "correct_option": correct_text,
    }


def _flags(*, flaw: str | None = None, missing: str | None = None) -> Dict[str, Any]:
    flags = {name: 0 for name in SAQUET_CRITERIA}
    if flaw:
        flags[flaw] = 1
    if missing:
        flags[missing] = NA
    return flags


def _apply_fixed(flags: Mapping[str, Any]):
    def _fn(_row: Mapping[str, Any]) -> Dict[str, Any]:
        return {"flags": dict(flags), "errors": {}}

    return _fn


def _mcq_case(
    *,
    ref: str = "q_00000001:v1",
    stem: str = "What is X?",
    decision: str = "accept",
) -> Dict[str, Any]:
    logical, _, ver = ref.partition(":v")
    return {
        "candidate_ref": {
            "key": ref,
            "logical_id": logical,
            "version": int(ver or 1),
        },
        "factual_decision": decision,
        "question": {
            "stem": stem,
            "options": [
                {"option_id": "A", "text": "alpha", "is_correct": True},
                {"option_id": "B", "text": "beta", "is_correct": False},
                {"option_id": "C", "text": "gamma", "is_correct": False},
                {"option_id": "D", "text": "delta", "is_correct": False},
            ],
            "correct_option": "A",
        },
    }


def _payload(*cases: Mapping[str, Any], run_id: str = "run-saquet") -> Dict[str, Any]:
    return {"run_id": run_id, "cases": list(cases)}


def _counting_apply(sequence: List[Mapping[str, Any]]):
    calls = {"n": 0}

    def _fn(_row: Mapping[str, Any]) -> Dict[str, Any]:
        idx = calls["n"]
        calls["n"] += 1
        if idx >= len(sequence):
            return {"flags": dict(sequence[-1]), "errors": {}}
        item = sequence[idx]
        if item.get("_raise"):
            raise RuntimeError(str(item.get("_raise") or "saquet failed"))
        return {"flags": dict(item), "errors": dict(item.get("_errors") or {})}

    return _fn, calls


class TestSaquetConversionAndCriteria(unittest.TestCase):
    def test_original_true_false_conversion(self):
        self.assertEqual(_iwf_flag(True), 0)
        self.assertEqual(_iwf_flag(False), 1)
        self.assertIs(_iwf_flag(None), NA)
        self.assertIs(_iwf_flag("yes"), NA)

    def test_numpy_bool_conversion_and_unsupported_values(self):
        import numpy as np

        self.assertEqual(_iwf_flag(np.bool_(True)), 0)
        self.assertEqual(_iwf_flag(np.bool_(False)), 1)
        self.assertIsInstance(_iwf_flag(np.bool_(True)), int)
        self.assertIs(_iwf_flag(None), NA)
        self.assertIs(_iwf_flag("False"), NA)
        self.assertIs(_iwf_flag("True"), NA)
        self.assertIs(_iwf_flag(0), NA)
        self.assertIs(_iwf_flag(1), NA)
        self.assertIs(_iwf_flag(0.0), NA)
        self.assertIs(_iwf_flag(1.0), NA)

    def test_mocked_llm_ratings_export_python_bool_flags(self):
        import numpy as np

        fake = types.ModuleType("saquet.saquet_core")
        fake.all_criteria = list(SAQUET_CRITERIA)
        fake.MultipleChoiceQuestion = lambda **kwargs: types.SimpleNamespace(**kwargs)

        def _rating(values):
            rating = np.mean(values)
            result = bool(rating >= 5)
            self.assertIsInstance(result, bool)
            self.assertNotIsInstance(result, np.bool_)
            return result

        for name in SAQUET_CRITERIA:
            if name == "unfocused_stem":
                setattr(fake, name, lambda _q: _rating([9.0, 10.0]))
            elif name == "grammatical_cues_in_stem":
                setattr(fake, name, lambda _q: _rating([3.0, 4.0]))
            elif name == "absolute_terms":
                setattr(fake, name, lambda _q: _rating([9.0, 9.0]))
            else:
                setattr(fake, name, lambda _q: True)
        with patch.dict(sys.modules, {"saquet.saquet_core": fake}):
            out = default_apply_saquet_criteria(
                _quiz_row(),
                criteria=[
                    "unfocused_stem",
                    "grammatical_cues_in_stem",
                    "absolute_terms",
                ],
            )
        self.assertEqual(out["flags"]["unfocused_stem"], 0)
        self.assertEqual(out["flags"]["grammatical_cues_in_stem"], 1)
        self.assertEqual(out["flags"]["absolute_terms"], 0)
        self.assertFalse(out.get("errors"))

    def test_default_apply_accepts_numpy_bool_results(self):
        import numpy as np

        fake = types.ModuleType("saquet.saquet_core")
        fake.all_criteria = list(SAQUET_CRITERIA)
        fake.MultipleChoiceQuestion = lambda **kwargs: types.SimpleNamespace(**kwargs)
        for name in SAQUET_CRITERIA:
            setattr(
                fake,
                name,
                lambda _q, n=name: np.bool_(False) if n == "vague_terms" else np.bool_(True),
            )
        with patch.dict(sys.modules, {"saquet.saquet_core": fake}):
            out = default_apply_saquet_criteria(_quiz_row())
        self.assertEqual(out["flags"]["vague_terms"], 1)
        self.assertEqual(out["flags"]["true_or_false"], 0)
        self.assertTrue(all(v in (0, 1) for v in out["flags"].values()))

    def test_nineteen_criteria_match_required_names(self):
        self.assertEqual(len(SAQUET_CRITERIA), 19)
        self.assertEqual(len(SAQUET_MAS_FIELDS), 15 + 19 * 3 + 2)
        self.assertIn("evaluator_model", SAQUET_MAS_FIELDS)
        self.assertIn("evaluator_config_version", SAQUET_MAS_FIELDS)
        self.assertNotIn("saquet_by_question", "".join(SAQUET_MAS_FIELDS))

    def test_default_apply_uses_supplied_functions_and_conversion(self):
        fake = types.ModuleType("saquet.saquet_core")
        fake.all_criteria = list(SAQUET_CRITERIA)
        fake.MultipleChoiceQuestion = lambda **kwargs: types.SimpleNamespace(**kwargs)
        for name in SAQUET_CRITERIA:
            setattr(
                fake,
                name,
                (lambda _q, n=name: False if n == "none_of_the_above" else True),
            )
        with patch.dict(sys.modules, {"saquet.saquet_core": fake}):
            out = default_apply_saquet_criteria(_quiz_row())
        self.assertEqual(out["flags"]["none_of_the_above"], 1)
        self.assertEqual(out["flags"]["true_or_false"], 0)
        self.assertTrue(all(v in (0, 1) for v in out["flags"].values()))

    def test_default_model_does_not_import_saquet_core(self):
        with patch("importlib.import_module") as imp:
            model = default_saquet_model()
            imp.assert_not_called()
        self.assertEqual(model, "gpt-4o")


class TestSaquetAggregation(unittest.TestCase):
    def test_empty_accepted_summary_row(self):
        row = aggregate_saquet_by_mas(
            [],
            lecturebank_id="141",
            run_id="r1",
            accepted_count=0,
        )
        self.assertEqual(row["final_accepted_mcq_count"], 0)
        self.assertEqual(row["fully_evaluated_mcq_count"], 0)
        self.assertEqual(row["partial_mcq_count"], 0)
        self.assertEqual(row["failed_mcq_count"], 0)
        self.assertEqual(row["not_evaluable_mcq_count"], 0)
        self.assertIs(row["evaluation_completion_rate"], NA)
        self.assertIs(row["total_iwf_count"], NA)
        self.assertIs(row["mean_iwf_per_mcq"], NA)
        self.assertIs(row["zero_iwf_mcq_count"], NA)
        self.assertIs(row["zero_iwf_mcq_rate"], NA)
        for name in SAQUET_CRITERIA:
            flagged_f, eval_f, rate_f = saquet_criterion_fields(name)
            self.assertEqual(row[flagged_f], 0)
            self.assertEqual(row[eval_f], 0)
            self.assertIs(row[rate_f], NA)

    def test_iwf_totals_use_fully_evaluated_only(self):
        full_zero = {
            "status": "fully_evaluated",
            "flags": _flags(),
            "total_iwf_count": 0,
        }
        full_flaw = {
            "status": "fully_evaluated",
            "flags": _flags(flaw="none_of_the_above"),
            "total_iwf_count": 1,
        }
        partial = {
            "status": "partial",
            "flags": _flags(flaw="true_or_false", missing="vague_terms"),
            "total_iwf_count": NA,
        }
        failed = {
            "status": "failed",
            "flags": {name: NA for name in SAQUET_CRITERIA},
            "total_iwf_count": NA,
        }
        row = aggregate_saquet_by_mas(
            [full_zero, full_flaw, partial, failed],
            lecturebank_id="141",
            run_id="r1",
            accepted_count=4,
        )
        self.assertEqual(row["fully_evaluated_mcq_count"], 2)
        self.assertEqual(row["partial_mcq_count"], 1)
        self.assertEqual(row["failed_mcq_count"], 1)
        self.assertEqual(row["total_iwf_count"], 1)
        self.assertEqual(row["mean_iwf_per_mcq"], 0.5)
        self.assertEqual(row["zero_iwf_mcq_count"], 1)
        self.assertEqual(row["zero_iwf_mcq_rate"], 0.5)
        self.assertEqual(row["evaluation_completion_rate"], 0.5)
        self.assertEqual(row["none_of_the_above_flagged_count"], 1)
        self.assertEqual(row["none_of_the_above_evaluated_count"], 3)
        self.assertEqual(row["true_or_false_flagged_count"], 1)
        self.assertEqual(row["true_or_false_evaluated_count"], 3)
        self.assertEqual(row["vague_terms_evaluated_count"], 2)
        self.assertEqual(row["vague_terms_flagged_count"], 0)

    def test_failures_are_not_counted_as_no_flaw(self):
        failed = {
            "status": "failed",
            "flags": {name: NA for name in SAQUET_CRITERIA},
            "total_iwf_count": NA,
        }
        row = aggregate_saquet_by_mas(
            [failed],
            lecturebank_id="141",
            accepted_count=1,
        )
        self.assertIs(row["total_iwf_count"], NA)
        self.assertIs(row["zero_iwf_mcq_count"], NA)
        self.assertIs(row["zero_iwf_mcq_rate"], NA)
        self.assertEqual(row["none_of_the_above_flagged_count"], 0)
        self.assertEqual(row["none_of_the_above_evaluated_count"], 0)
        self.assertIs(row["none_of_the_above_flag_rate"], NA)

    def test_no_fully_evaluated_makes_overall_iwf_na(self):
        partial = {
            "status": "partial",
            "flags": _flags(flaw="all_of_the_above", missing="lost_sequence"),
            "total_iwf_count": NA,
        }
        row = aggregate_saquet_by_mas([partial], lecturebank_id="141", accepted_count=1)
        self.assertIs(row["total_iwf_count"], NA)
        self.assertIs(row["mean_iwf_per_mcq"], NA)
        self.assertEqual(row["all_of_the_above_flagged_count"], 1)
        self.assertEqual(row["all_of_the_above_evaluated_count"], 1)
        self.assertEqual(row["all_of_the_above_flag_rate"], 1.0)
        self.assertEqual(row["lost_sequence_evaluated_count"], 0)
        self.assertIs(row["lost_sequence_flag_rate"], NA)


class TestSaquetCacheAndEvaluate(unittest.TestCase):
    def test_not_evaluable_is_cached_and_does_not_call_apply(self):
        calls = {"n": 0}

        def apply(_row):
            calls["n"] += 1
            raise AssertionError("apply must not run for unmappable questions")

        with tempfile.TemporaryDirectory() as tmp:
            cache = SaquetQuestionCache(Path(tmp) / "saquet_question_cache.json")
            row = _quiz_row(options=["only-one"], correct="missing")
            first = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
            second = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
        self.assertEqual(first["status"], "not_evaluable")
        self.assertTrue(second["cache_hit"])
        self.assertEqual(calls["n"], 0)

    def test_failed_and_partial_are_persisted_and_retried(self):
        apply, calls = _counting_apply(
            [
                {"_raise": "boom"},
                _flags(flaw="true_or_false", missing="vague_terms"),
                _flags(),
            ]
        )
        row = _quiz_row()
        with tempfile.TemporaryDirectory() as tmp:
            cache = SaquetQuestionCache(Path(tmp) / "cache.json")
            failed = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
            partial = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
            full = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
            cached = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
        self.assertEqual(failed["status"], "failed")
        self.assertIsNotNone(cache.get(failed["cache_key"]))
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(full["status"], "fully_evaluated")
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(calls["n"], 3)

    def test_evaluate_document_does_not_import_saquet_core(self):
        payload = _payload(_mcq_case(), _mcq_case(ref="q_00000002:v1", stem="What is Y?"))
        with patch("importlib.import_module") as imp:
            out = evaluate_document_saquet(
                payload,
                lecturebank_id="141",
                run_id="run-saquet",
                apply_fn=_apply_fixed(_flags(flaw="none_of_the_above")),
                evaluator_model="gpt-4o",
                accepted_count_metadata=2,
            )
            imp.assert_not_called()
        mas = out["mas_row"]
        self.assertEqual(mas["final_accepted_mcq_count"], 2)
        self.assertEqual(mas["fully_evaluated_mcq_count"], 2)
        self.assertEqual(mas["total_iwf_count"], 2)
        self.assertEqual(mas["none_of_the_above_flagged_count"], 2)
        self.assertEqual(mas["evaluator_model"], "gpt-4o")
        self.assertEqual(mas["evaluator_config_version"], SAQUET_CONFIG_VERSION)

    def test_empty_payload_still_emits_mas_row(self):
        out = evaluate_document_saquet(
            {
                "run_id": "run-empty",
                "cases": [],
                "factual_quiz": {
                    "kind": "final_accepted_factual_quiz",
                    "accepted_count": 0,
                    "questions": [],
                },
            },
            lecturebank_id="141",
            run_id="run-empty",
            apply_fn=_apply_fixed(_flags()),
            evaluator_model="gpt-4o",
        )
        mas = out["mas_row"]
        self.assertEqual(mas["final_accepted_mcq_count"], 0)
        self.assertIs(mas["evaluation_completion_rate"], NA)
        self.assertIs(mas["total_iwf_count"], NA)


class TestSaquetCsvAndReport(unittest.TestCase):
    def test_writes_per_mas_csv_only(self):
        mas = empty_saquet_mas_row(lecturebank_id="141", run_id="r1")
        mas["final_accepted_mcq_count"] = 1
        mas["fully_evaluated_mcq_count"] = 1
        mas["evaluation_completion_rate"] = 1.0
        mas["total_iwf_count"] = 0
        mas["mean_iwf_per_mcq"] = 0.0
        mas["zero_iwf_mcq_count"] = 1
        mas["zero_iwf_mcq_rate"] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp)
            write_metrics_outputs(
                metrics,
                document_rows=[{"lecturebank_id": "141"}],
                attempt_rows=[],
                comparison_rows=[],
                provenance_rows=[],
                distribution_rows=[],
                round_rows=[],
                overall_rows=[],
                report_markdown="# t\n",
                saquet_mas_rows=[mas],
            )
            self.assertTrue((metrics / "saquet_by_mas.csv").is_file())
            self.assertFalse((metrics / "saquet_by_question.csv").exists())
            with (metrics / "saquet_by_mas.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0].keys()), list(SAQUET_MAS_FIELDS))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["lecturebank_id"], "141")
            self.assertEqual(rows[0]["evaluator_config_version"], SAQUET_CONFIG_VERSION)

    def test_report_describes_mas_csv_not_question_csv(self):
        mas = empty_saquet_mas_row(lecturebank_id="141", run_id="r1")
        report = build_report(
            catalog_rows=[{"lecturebank_id": "141"}],
            document_rows=[{"lecturebank_id": "141"}],
            attempt_rows=[],
            comparison_rows=[],
            provenance_rows=[],
            distribution_rows=[],
            round_rows=[],
            overall_rows=[],
            pooled={},
            saquet_mas_rows=[mas],
        )
        self.assertIn("saquet_by_mas.csv", report)
        self.assertNotIn("saquet_by_question.csv", report)
        self.assertIn("SAQUET item-writing flaws", report)


class TestSaquetCliFingerprintAndOffline(unittest.TestCase):
    def test_cli_flags_default_off(self):
        parser = build_arg_parser()
        off = parser.parse_args(
            ["--lecturebank-dir", "x", "--id", "1", "--allow-any-count"]
        )
        self.assertFalse(off.evaluate_saquet)
        self.assertFalse(off.saquet_from_saved)
        self.assertFalse(off.reevaluate_saquet)
        on = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "1",
                "--allow-any-count",
                "--evaluate-saquet",
                "--saquet-from-saved",
                "--reevaluate-saquet",
            ]
        )
        self.assertTrue(on.evaluate_saquet)
        self.assertTrue(on.saquet_from_saved)
        self.assertTrue(on.reevaluate_saquet)

    def test_generation_fingerprint_excludes_saquet_flags(self):
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
                "evaluate_saquet": True,
                "saquet_from_saved": True,
                "reevaluate_saquet": True,
            }
        )
        self.assertEqual(off, on)
        self.assertNotIn("evaluate_saquet", on)
        self.assertNotIn("saquet_from_saved", on)
        self.assertNotIn("reevaluate_saquet", on)

    def test_from_saved_writes_mas_csv_without_process_documents(self):
        payload = _payload(_mcq_case(), _mcq_case(ref="q_00000002:v1", stem="What is Y?"))
        item = {
            "lecturebank_id": "141",
            "run_id": "run-saquet",
            "eval_status": "success",
            "document_metrics": {
                "lecturebank_id": "141",
                "eval_status": "success",
                "accepted_question_count": 2,
            },
        }
        apply_fn = _apply_fixed(_flags())
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
            with patch(
                "mas_langgraph.evaluation.execute.process_documents",
                side_effect=AssertionError("RAG/MAS/AXIS must not run"),
            ) as gen_run, patch(
                "mas_langgraph.batch.generate.run_factual_and_explanation_in_memory",
                side_effect=AssertionError("MAS generation must not run"),
            ), patch(
                "mas_langgraph.batch.prepare.prepare_presentation",
                side_effect=AssertionError("RAG must not run"),
            ):
                stats = run_saquet_from_saved(
                    selected=["141"],
                    args=Namespace(
                        evaluate_saquet=True,
                        reevaluate_saquet=False,
                        _saquet_apply_fn=apply_fn,
                    ),
                    output_dir=output_dir,
                    metrics_dir=metrics_dir,
                    collected=collected,
                    persist=lambda *a, **k: _persist(*a, **k),
                    manifest_path=output_dir / "lecturebank_eval_manifest.json",
                    manifest={"manifest_version": 1, "records": []},
                    log=io.StringIO(),
                )
            gen_run.assert_not_called()
            self.assertEqual(stats["failures"], 0)
            self.assertTrue((metrics_dir / "saquet_by_mas.csv").is_file())
            self.assertFalse((metrics_dir / "saquet_by_question.csv").exists())
            self.assertTrue((metrics_dir / "saquet_question_cache.json").is_file())
            with (metrics_dir / "saquet_by_mas.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["final_accepted_mcq_count"], "2")
            self.assertEqual(rows[0]["fully_evaluated_mcq_count"], "2")
            cache = json.loads(
                (metrics_dir / "saquet_question_cache.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cache["kind"], "saquet_question_cache")
            self.assertEqual(len(cache["entries"]), 2)

    def test_from_saved_missing_json_fails_without_generation(self):
        item = {
            "lecturebank_id": "141",
            "eval_status": "success",
            "document_metrics": {"lecturebank_id": "141", "eval_status": "success"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics_dir = output_dir / "metrics"
            _persist(
                output_dir,
                metrics_dir,
                output_dir / "lecturebank_eval_manifest.json",
                {"manifest_version": 1, "records": []},
                [item],
            )
            collected = load_collected_from_metrics(metrics_dir)
            with patch(
                "mas_langgraph.evaluation.execute.process_documents",
                side_effect=AssertionError("RAG/MAS/AXIS must not run"),
            ) as gen_run, patch(
                "mas_langgraph.batch.generate.run_factual_and_explanation_in_memory",
                side_effect=AssertionError("MAS generation must not run"),
            ), patch(
                "mas_langgraph.batch.prepare.prepare_presentation",
                side_effect=AssertionError("RAG must not run"),
            ):
                stats = run_saquet_from_saved(
                    selected=["141"],
                    args=Namespace(evaluate_saquet=True, _saquet_apply_fn=_apply_fixed(_flags())),
                    output_dir=output_dir,
                    metrics_dir=metrics_dir,
                    collected=collected,
                    persist=lambda *a, **k: _persist(*a, **k),
                    manifest_path=output_dir / "lecturebank_eval_manifest.json",
                    manifest={"manifest_version": 1, "records": []},
                    log=io.StringIO(),
                )
            gen_run.assert_not_called()
            self.assertEqual(stats["failures"], 1)
            self.assertFalse((metrics_dir / "saquet_by_mas.csv").exists())

    def test_from_saved_reuses_persisted_factual_quiz_without_streamlit(self):
        item = {
            "lecturebank_id": "141",
            "eval_status": "success",
            "run_id": "run-saquet",
            "accepted_question_count": 2,
            "document_metrics": {
                "lecturebank_id": "141",
                "eval_status": "success",
                "accepted_question_count": 2,
            },
        }
        apply_fn = _apply_fixed(_flags())
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir()
            persist_factual_quiz(
                metrics_dir,
                "141",
                {
                    "kind": "final_accepted_factual_quiz",
                    "accepted_count": 2,
                    "questions": [_factual_question(1), _factual_question(2)],
                },
            )
            _persist(
                output_dir,
                metrics_dir,
                output_dir / "lecturebank_eval_manifest.json",
                {"manifest_version": 1, "records": []},
                [item],
            )
            collected = load_collected_from_metrics(metrics_dir)
            with patch(
                "mas_langgraph.evaluation.execute.process_documents",
                side_effect=AssertionError("RAG/MAS/AXIS must not run"),
            ) as gen_run:
                stats = run_saquet_from_saved(
                    selected=["141"],
                    args=Namespace(
                        evaluate_saquet=True,
                        reevaluate_saquet=False,
                        _saquet_apply_fn=apply_fn,
                    ),
                    output_dir=output_dir,
                    metrics_dir=metrics_dir,
                    collected=collected,
                    persist=lambda *a, **k: _persist(*a, **k),
                    manifest_path=output_dir / "lecturebank_eval_manifest.json",
                    manifest={"manifest_version": 1, "records": []},
                    log=io.StringIO(),
                )
            gen_run.assert_not_called()
            self.assertEqual(stats["failures"], 0)
            self.assertTrue((metrics_dir / "saquet_by_mas.csv").is_file())
            with (metrics_dir / "saquet_by_mas.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["final_accepted_mcq_count"], "2")
            self.assertEqual(rows[0]["fully_evaluated_mcq_count"], "2")
            self.assertFalse((output_dir / "streamlit").exists())


def _factual_question(index: int, *, stem: str | None = None, ref: str | None = None) -> Dict[str, Any]:
    qid = f"q_{index:08d}"
    return {
        "question_index": index,
        "q_id": qid,
        "candidate_ref": ref or f"{qid}:v1",
        "stem": stem or f"What is fact {index}?",
        "options": [
            {"option_id": "A", "text": "alpha", "is_correct": True},
            {"option_id": "B", "text": "beta", "is_correct": False},
            {"option_id": "C", "text": "gamma", "is_correct": False},
            {"option_id": "D", "text": "delta", "is_correct": False},
        ],
        "correct_option_id": "A",
        "correct_option": "alpha",
        "question_version": 1,
    }


class TestSaquetFactualQuizSource(unittest.TestCase):
    def test_evaluates_all_accepted_mcqs_not_just_explanation_cases(self):
        payload = _payload(
            _mcq_case(ref="q_00000001:v1", stem="Explained 1"),
            _mcq_case(ref="q_00000002:v1", stem="Explained 2"),
        )
        payload["factual_quiz"] = {
            "kind": "final_accepted_factual_quiz",
            "accepted_count": 10,
            "questions": [_factual_question(i) for i in range(1, 11)],
        }
        seen: List[str] = []

        def apply(row, criteria=None):
            ref = str(row.get("candidate_ref") or "")
            if ref not in seen:
                seen.append(ref)
            names = list(criteria or SAQUET_CRITERIA)
            return {"flags": {name: 0 for name in names}, "errors": {}}

        out = evaluate_document_saquet(
            payload,
            lecturebank_id="141",
            apply_fn=apply,
            evaluator_model="gpt-4o",
            accepted_count_metadata=10,
        )
        self.assertEqual(len(seen), 10)
        self.assertEqual(out["mas_row"]["final_accepted_mcq_count"], 10)
        self.assertEqual(out["mas_row"]["fully_evaluated_mcq_count"], 10)
        self.assertEqual(out["completeness"], "complete")
        self.assertEqual(out["quiz_source"], "factual_quiz")

    def test_historical_incomplete_export_does_not_treat_subset_as_full_quiz(self):
        payload = _payload(
            _mcq_case(ref="q_00000001:v1", stem="Explained 1"),
            _mcq_case(ref="q_00000002:v1", stem="Explained 2"),
        )
        out = evaluate_document_saquet(
            payload,
            lecturebank_id="141",
            apply_fn=_apply_fixed(_flags()),
            evaluator_model="gpt-4o",
            accepted_count_metadata=10,
        )
        mas = out["mas_row"]
        self.assertEqual(mas["quiz_completeness"], "incomplete")
        self.assertEqual(mas["final_accepted_mcq_count"], 10)
        self.assertEqual(mas["fully_evaluated_mcq_count"], 2)
        self.assertIs(mas["total_iwf_count"], NA)
        self.assertIs(mas["none_of_the_above_flag_rate"], NA)
        self.assertIn("were not regenerated", mas["quiz_completeness_reason"])

    def test_missing_accepted_count_is_unknown_not_zero(self):
        out = evaluate_document_saquet(
            {"run_id": "r", "cases": []},
            lecturebank_id="141",
            apply_fn=_apply_fixed(_flags()),
            evaluator_model="gpt-4o",
        )
        mas = out["mas_row"]
        self.assertEqual(mas["quiz_completeness"], "unknown")
        self.assertIs(mas["final_accepted_mcq_count"], NA)
        self.assertIs(mas["total_iwf_count"], NA)


class TestSaquetCacheCoverage(unittest.TestCase):
    def test_csv_summary_does_not_establish_completeness(self):
        row_a = _quiz_row(ref="q_00000001:v1", stem="One")
        row_b = _quiz_row(ref="q_00000002:v1", stem="Two")
        calls: List[str] = []

        def apply(row, criteria=None):
            ref = str(row.get("candidate_ref") or "")
            if not criteria or criteria == list(SAQUET_CRITERIA):
                calls.append(ref)
            elif ref not in calls:
                calls.append(ref)
            names = list(criteria or SAQUET_CRITERIA)
            return {"flags": {name: 0 for name in names}, "errors": {}}

        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp)
            evaluate_document_saquet(
                lecturebank_id="141",
                quiz_rows=[row_a],
                metrics_dir=metrics,
                apply_fn=apply,
                evaluator_model="gpt-4o",
            )
            looking_complete = empty_saquet_mas_row(lecturebank_id="141", run_id="r")
            looking_complete["final_accepted_mcq_count"] = 2
            looking_complete["fully_evaluated_mcq_count"] = 2
            looking_complete["evaluation_completion_rate"] = 1.0
            write_metrics_outputs(
                metrics,
                document_rows=[{"lecturebank_id": "141"}],
                attempt_rows=[],
                comparison_rows=[],
                provenance_rows=[],
                distribution_rows=[],
                round_rows=[],
                overall_rows=[],
                report_markdown="# t\n",
                saquet_mas_rows=[looking_complete],
            )
            out = evaluate_document_saquet(
                lecturebank_id="141",
                quiz_rows=[row_a, row_b],
                metrics_dir=metrics,
                apply_fn=apply,
                evaluator_model="gpt-4o",
            )
        self.assertIn("q_00000002:v1", calls)
        self.assertEqual(out["mas_row"]["final_accepted_mcq_count"], 2)
        self.assertEqual(out["mas_row"]["fully_evaluated_mcq_count"], 2)

    def test_content_change_same_count_is_reevaluated(self):
        calls = {"n": 0}

        def apply(row, criteria=None):
            if not criteria or len(list(criteria)) == len(SAQUET_CRITERIA):
                calls["n"] += 1
            elif calls.get("_seen") != row.get("stem"):
                calls["n"] += 1
                calls["_seen"] = row.get("stem")
            names = list(criteria or SAQUET_CRITERIA)
            return {"flags": {name: 0 for name in names}, "errors": {}}

        with tempfile.TemporaryDirectory() as tmp:
            cache = SaquetQuestionCache(Path(tmp) / "cache.json")
            evaluate_quiz_row(
                _quiz_row(stem="Old stem", ref="q1:v1"),
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
            n_after_first = calls["n"]
            evaluate_quiz_row(
                _quiz_row(stem="New stem", ref="q1:v1"),
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
        self.assertGreater(calls["n"], n_after_first)

    def test_removed_question_is_dropped_from_aggregate(self):
        apply = _apply_fixed(_flags())
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp)
            first = evaluate_document_saquet(
                lecturebank_id="141",
                quiz_rows=[
                    _quiz_row(ref="q_00000001:v1"),
                    _quiz_row(ref="q_00000002:v1", stem="Second"),
                ],
                metrics_dir=metrics,
                apply_fn=apply,
                evaluator_model="gpt-4o",
            )
            second = evaluate_document_saquet(
                lecturebank_id="141",
                quiz_rows=[_quiz_row(ref="q_00000001:v1")],
                metrics_dir=metrics,
                apply_fn=apply,
                evaluator_model="gpt-4o",
            )
        self.assertEqual(first["mas_row"]["final_accepted_mcq_count"], 2)
        self.assertEqual(second["mas_row"]["final_accepted_mcq_count"], 1)
        self.assertEqual(second["mas_row"]["fully_evaluated_mcq_count"], 1)

    def test_evaluator_config_change_invalidates_cache(self):
        calls = {"n": 0}

        def apply(row, criteria=None):
            calls["n"] += 1
            names = list(criteria or SAQUET_CRITERIA)
            return {"flags": {name: 0 for name in names}, "errors": {}}

        row = _quiz_row()
        with tempfile.TemporaryDirectory() as tmp:
            cache = SaquetQuestionCache(Path(tmp) / "cache.json")
            evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
            first = calls["n"]
            evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version="saquet_iwf_other",
                apply_fn=apply,
                cache=cache,
            )
        self.assertGreater(calls["n"], first)


class TestSaquetPartialCriterionRetry(unittest.TestCase):
    def test_retries_only_the_failed_criterion(self):
        failed_name = "negative_worded_stem"
        calls: List[str] = []

        def apply(row, criteria=None):
            names = list(criteria or SAQUET_CRITERIA)
            calls.extend(names)
            flags = {}
            for name in names:
                if name == failed_name and calls.count(failed_name) == 1:
                    raise RuntimeError("transient criterion failure")
                flags[name] = 0
            return {"flags": flags, "errors": {}}

        row = _quiz_row()
        with tempfile.TemporaryDirectory() as tmp:
            cache = SaquetQuestionCache(Path(tmp) / "cache.json")
            first = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
            self.assertEqual(first["status"], "partial")
            self.assertEqual(first["flags"][failed_name], NA)
            self.assertEqual(sum(1 for n in SAQUET_CRITERIA if first["flags"][n] in (0, 1)), 18)
            n_first = list(calls)
            second = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
        self.assertEqual(second["status"], "fully_evaluated")
        self.assertEqual(second["flags"][failed_name], 0)
        self.assertEqual(calls[len(n_first) :], [failed_name])

    def test_resume_preserves_valid_flags_and_clears_recovered_errors(self):
        missing_name = "unfocused_stem"
        preserved_name = "true_or_false"
        calls: List[str] = []

        def apply(row, criteria=None):
            names = list(criteria or SAQUET_CRITERIA)
            calls.extend(names)
            flags = {}
            errors = {}
            for name in names:
                if name == missing_name and calls.count(missing_name) == 1:
                    flags[name] = NA
                    errors[name] = "criterion_result_unavailable"
                elif name == preserved_name:
                    flags[name] = 1
                else:
                    flags[name] = 0
            return {"flags": flags, "errors": errors}

        row = _quiz_row()
        with tempfile.TemporaryDirectory() as tmp:
            cache = SaquetQuestionCache(Path(tmp) / "cache.json")
            first = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
            self.assertEqual(first["status"], "partial")
            self.assertEqual(first["flags"][missing_name], NA)
            self.assertEqual(first["flags"][preserved_name], 1)
            self.assertEqual(first["errors"].get(missing_name), "criterion_result_unavailable")
            n_first = list(calls)
            second = evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
        self.assertEqual(second["status"], "fully_evaluated")
        self.assertEqual(second["flags"][missing_name], 0)
        self.assertEqual(second["flags"][preserved_name], 1)
        self.assertNotIn(missing_name, second.get("errors") or {})
        self.assertEqual(calls[len(n_first) :], [missing_name])
        self.assertNotIn(preserved_name, calls[len(n_first) :])

    def test_reevaluate_flag_reruns_successful_criteria(self):
        calls: List[str] = []

        def apply(row, criteria=None):
            names = list(criteria or SAQUET_CRITERIA)
            calls.extend(names)
            return {"flags": {name: 0 for name in names}, "errors": {}}

        row = _quiz_row()
        with tempfile.TemporaryDirectory() as tmp:
            cache = SaquetQuestionCache(Path(tmp) / "cache.json")
            evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
            )
            n_first = len(calls)
            evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
                reevaluate=False,
            )
            self.assertEqual(len(calls), n_first)
            evaluate_quiz_row(
                row,
                evaluator_model="gpt-4o",
                evaluator_config_version=SAQUET_CONFIG_VERSION,
                apply_fn=apply,
                cache=cache,
                reevaluate=True,
            )
        self.assertEqual(len(calls), n_first * 2)


class TestSerializeFinalAcceptedQuiz(unittest.TestCase):
    def test_pack_questions_are_serialized_with_versions(self):
        from mas_langgraph.core.entities import QuizPack, ValidatedQuestion
        from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ
        from mas_langgraph.domains.quiz.packaging import serialize_final_accepted_quiz

        mcq = CandidateMCQ(
            q_id="q_00000001",
            blueprint_id="bp",
            stem="What is X?",
            options=[
                AnswerOption("A", "alpha", True),
                AnswerOption("B", "beta", False),
            ],
            correct_option_id="A",
            answer_explanation="because",
            revision_count=1,
        )
        vq = ValidatedQuestion(
            q_id="q_00000001",
            question_text="What is X?",
            question_type="multiple_choice",
            correct_answer="alpha",
            difficulty="intermediate",
            options=["alpha", "beta"],
        )
        state = types.SimpleNamespace(
            final_quiz_packs=[QuizPack("pack1", "Quiz", [vq])],
            candidate_mcqs={"q_00000001": mcq},
        )
        quiz = serialize_final_accepted_quiz(state)
        self.assertEqual(quiz["accepted_count"], 1)
        self.assertEqual(quiz["questions"][0]["candidate_ref"], "q_00000001:v2")
        self.assertEqual(quiz["questions"][0]["correct_option"], "alpha")


if __name__ == "__main__":
    unittest.main()
