"""Per-criterion AXIS replay export (stored criterion_deltas; no live model calls)."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from mas_langgraph.evaluation.axis_metrics import NA, extract_experiment_metrics
from mas_langgraph.evaluation.execute import run_outcome_criteria_from_saved
from mas_langgraph.evaluation.lifecycle import load_collected_from_metrics
from mas_langgraph.evaluation.outcome_criteria import (
    OUTCOME_CRITERIA_FIELDS,
    aggregate_outcome_criteria,
    extract_outcome_criteria_changes,
    filter_outcome_criteria_changed,
)
from mas_langgraph.evaluation.report import build_report, write_metrics_outputs
from mas_langgraph.evaluation.runner import _persist, build_arg_parser
from mas_langgraph.evaluation.streamlit_export import write_streamlit_experiment_json


_UNCHANGED_CRITERIA = [
    ("answer_supported", True),
    ("blueprint_available", True),
    ("candidate_status", "evidence_verified"),
    ("correct_option_count", 1),
    ("cross_cove_majority_present", True),
    ("cross_cove_passed", True),
    ("cross_cove_report_available", True),
    ("cross_cove_required", True),
    ("distractors_valid", True),
    ("evidence_report_available", True),
    ("explanation_supported", True),
    ("hard_blocking_flags", []),
    ("max_revisions", 2),
    ("max_revisions_reached_flag", False),
    ("pdf_supported", True),
    ("revision_count", 0),
    ("source_refs_present", True),
    ("stem_supported", True),
    ("structural_reject_flags", []),
    ("target_atom_count", 6),
]


def _delta(
    name: str,
    factual: Any,
    noop: Any = None,
    counterfactual: Any = None,
    *,
    decision_relevant: bool = False,
    noop_differs: bool = False,
    cf_differs: bool = False,
    omit: Optional[List[str]] = None,
) -> Dict[str, Any]:
    row = {
        "name": name,
        "factual": factual if noop is None else factual,
        "noop": factual if noop is None else noop,
        "counterfactual": factual if counterfactual is None else counterfactual,
        "decision_relevant": decision_relevant,
        "noop_differs_from_factual": noop_differs,
        "counterfactual_differs_from_noop": cf_differs,
    }
    for key in omit or []:
        row.pop(key, None)
    return row


def _regression_deltas() -> List[Dict[str, Any]]:
    rows = [
        _delta("blueprint_evidence_count", 2, 2, 1, decision_relevant=True, cf_differs=True),
        _delta("evidence_confidence", 1.0, 1.0, 0.0, decision_relevant=False, cf_differs=True),
    ]
    for name, value in _UNCHANGED_CRITERIA:
        rows.append(_delta(name, value))
    return rows


def _cmp(
    *,
    op: str = "WHATIF",
    target: str = "chunk_00000002:v1",
    replacement: str = "chunk_00000003:v1",
    alias: str = "c_8",
    ckpt_id: str = "ckpt-c8",
    evidence: str = "criterion_effect",
    decision_cs: Any = True,
    criteria_cs: Any = True,
    content_cs: Any = True,
    deltas: Optional[List[Dict[str, Any]]] = None,
    include_deltas: bool = True,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "intervention_signature": {
            "operation": op,
            "target_ref": {"key": target},
            "replacement_ref": {"key": replacement} if replacement else None,
            "checkpoint_alias": alias,
            "checkpoint_id": ckpt_id,
        },
        "operation": op,
        "target_ref": {"key": target},
        "replacement_ref": {"key": replacement} if replacement else None,
        "checkpoint_alias": alias,
        "checkpoint_id": ckpt_id,
        "evidence_status": evidence,
        "decision_control_stable": decision_cs,
        "criteria_control_stable": criteria_cs,
        "subject_content_control_stable": content_cs,
        "factual": {"run_id": "run_test"},
    }
    if include_deltas:
        payload["criterion_deltas"] = list(deltas if deltas is not None else _regression_deltas())
    return payload


def _case(
    comparisons: List[Dict[str, Any]],
    *,
    nested: bool = False,
    candidate: str = "q_00000001:v1",
    extra_copies: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    top = [json.loads(json.dumps(c)) for c in comparisons]
    raw: Dict[str, Any] = {"replay_comparisons": top}
    copies = extra_copies if extra_copies is not None else (comparisons if nested else None)
    if copies is not None:
        cloned = [json.loads(json.dumps(c)) for c in copies]
        raw["axis_report"] = {
            "replay_comparisons": cloned,
            "query_rounds": [{"replay_comparison": json.loads(json.dumps(c))} for c in cloned],
        }
    return {
        "candidate_ref": {"key": candidate, "logical_id": candidate.split(":")[0], "version": 1},
        "conditions": {"axis": {"raw_evidence": raw}},
    }


def _experiment(*cases: Dict[str, Any], run_id: str = "run_test") -> Dict[str, Any]:
    return {"run_id": run_id, "cases": list(cases)}


def _saved_experiment_path() -> Optional[Path]:
    rag = Path(__file__).resolve().parents[2]
    for path in (
        rag / "quiz_explanation_experiment_141(2).json",
        rag / "lecturebank_axis_eval_141_concept_check" / "streamlit" / "quiz_explanation_experiment_141.json",
    ):
        if path.is_file() and path.stat().st_size:
            return path
    return None


def _row_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["criterion_name"]): r for r in rows}


class TestOutcomeCriteriaExport(unittest.TestCase):
    def test_changed_and_unchanged_criteria(self):
        out = extract_outcome_criteria_changes(
            _experiment(_case([_cmp()])),
            lecturebank_id="141",
        )
        rows = out["rows"]
        self.assertEqual(len(rows), 22)
        by_name = _row_map(rows)
        changed = [r for r in rows if r["criterion_change_supported"] is True]
        self.assertEqual(len(changed), 2)
        ev = by_name["blueprint_evidence_count"]
        self.assertEqual(ev["factual_value"], 2)
        self.assertEqual(ev["noop_value"], 2)
        self.assertEqual(ev["counterfactual_value"], 1)
        self.assertIs(ev["decision_relevant"], True)
        self.assertIs(ev["criterion_change_supported"], True)
        conf = by_name["evidence_confidence"]
        self.assertEqual(conf["factual_value"], 1.0)
        self.assertEqual(conf["noop_value"], 1.0)
        self.assertEqual(conf["counterfactual_value"], 0.0)
        self.assertIs(conf["decision_relevant"], False)
        self.assertIs(conf["criterion_change_supported"], True)
        unchanged = by_name["answer_supported"]
        self.assertIs(unchanged["counterfactual_differs_from_noop"], False)
        self.assertIs(unchanged["criterion_change_supported"], False)
        self.assertEqual(unchanged["operation"], "REPLACE")
        self.assertIn("chunk_00000002:v1", unchanged["intervention_signature"])
        self.assertIn("chunk_00000003:v1", unchanged["intervention_signature"])
        self.assertIn("ckpt-c8", unchanged["intervention_signature"])

    def test_invalid_and_unstable_noop_keep_raw_diffs(self):
        invalid = _cmp(evidence="noop_error")
        unstable = _cmp(
            target="chunk_00000004:v1",
            replacement="chunk_00000005:v1",
            alias="c_7",
            ckpt_id="ckpt-c7",
            decision_cs=False,
        )
        rows = extract_outcome_criteria_changes(
            _experiment(_case([invalid, unstable])),
            lecturebank_id="141",
        )["rows"]
        inv = [r for r in rows if r["checkpoint_id"] == "ckpt-c8"]
        uns = [r for r in rows if r["checkpoint_id"] == "ckpt-c7"]
        self.assertTrue(inv)
        self.assertTrue(uns)
        self.assertTrue(all(r["criterion_change_supported"] is NA for r in inv + uns))
        self.assertTrue(all(r["valid_comparison"] is False for r in inv))
        self.assertTrue(all(r["full_control_stable"] is False for r in uns))
        self.assertTrue(
            any(
                r["criterion_name"] == "blueprint_evidence_count"
                and r["counterfactual_differs_from_noop"] is True
                for r in inv
            )
        )

    def test_missing_historical_values_are_na_not_unchanged(self):
        deltas = [
            _delta("present", 1, 1, 2, cf_differs=True),
            {"name": "legacy_missing_flags", "factual": 3},
        ]
        row_map = _row_map(
            extract_outcome_criteria_changes(
                _experiment(_case([_cmp(deltas=deltas)])),
                lecturebank_id="141",
            )["rows"]
        )
        missing = row_map["legacy_missing_flags"]
        self.assertIs(missing["noop_value"], NA)
        self.assertIs(missing["counterfactual_value"], NA)
        self.assertIs(missing["counterfactual_differs_from_noop"], NA)
        self.assertIs(missing["criterion_change_supported"], NA)
        self.assertIsNot(missing["counterfactual_differs_from_noop"], False)

    def test_list_valued_criteria_round_trip_json(self):
        deltas = [
            _delta("hard_blocking_flags", [], [], ["duplicate_atom"], cf_differs=True),
            _delta("structural_reject_flags", []),
        ]
        rows = extract_outcome_criteria_changes(
            _experiment(_case([_cmp(deltas=deltas)])),
            lecturebank_id="141",
        )["rows"]
        flags = _row_map(rows)["hard_blocking_flags"]
        self.assertEqual(flags["factual_value"], [])
        self.assertEqual(flags["counterfactual_value"], ["duplicate_atom"])
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
                outcome_criteria_rows=rows,
            )
            csv_path = metrics / "outcome_criteria_changes.csv"
            with csv_path.open(encoding="utf-8", newline="") as handle:
                loaded = list(csv.DictReader(handle))
            cell = next(r for r in loaded if r["criterion_name"] == "hard_blocking_flags")
            self.assertEqual(json.loads(cell["factual_value"]), [])
            self.assertEqual(json.loads(cell["counterfactual_value"]), ["duplicate_atom"])
            empty = next(r for r in loaded if r["criterion_name"] == "structural_reject_flags")
            self.assertEqual(json.loads(empty["factual_value"]), [])
            self.assertNotEqual(empty["factual_value"], "NA")

    def test_nested_and_flat_export_deduplicates(self):
        cmp = _cmp()
        nested = extract_outcome_criteria_changes(
            _experiment(_case([cmp], nested=True)),
            lecturebank_id="141",
        )["rows"]
        flat = extract_outcome_criteria_changes(
            _experiment(_case([cmp], nested=False)),
            lecturebank_id="141",
        )["rows"]
        self.assertEqual(len(nested), 22)
        self.assertEqual(len(flat), 22)
        keys = {
            (r["candidate_ref"], r["intervention_signature"], r["criterion_name"])
            for r in nested
        }
        self.assertEqual(len(keys), 22)

    def test_conflicting_duplicate_values_are_na(self):
        left = _cmp()
        right = json.loads(json.dumps(left))
        for delta in right["criterion_deltas"]:
            if delta["name"] == "blueprint_evidence_count":
                delta["counterfactual"] = 99
        case = _case([left])
        raw = case["conditions"]["axis"]["raw_evidence"]
        raw["axis_report"] = {"replay_comparisons": [right]}
        out = extract_outcome_criteria_changes(_experiment(case), lecturebank_id="141")
        row = _row_map(out["rows"])["blueprint_evidence_count"]
        self.assertIs(row["counterfactual_value"], NA)
        self.assertIs(row["criterion_change_supported"], NA)
        self.assertTrue(any(d["kind"] == "conflicting_criterion_values" for d in out["diagnostics"]))

    def _nested_conflict_payload(
        self,
        *,
        name: str,
        factual: Any,
        noop: Any,
        counterfactuals: tuple,
        cf_differs: bool,
    ) -> Dict[str, Any]:
        copies = []
        for cf in counterfactuals:
            copies.append(
                _cmp(
                    deltas=[
                        _delta(
                            name,
                            factual,
                            noop,
                            cf,
                            decision_relevant=True,
                            cf_differs=cf_differs,
                        )
                    ]
                )
            )
        return _experiment(_case([copies[0]], extra_copies=copies[1:]))

    def test_conflicting_true_diff_flags_supported_is_na(self):
        out = extract_outcome_criteria_changes(
            self._nested_conflict_payload(
                name="blueprint_evidence_count",
                factual=2,
                noop=2,
                counterfactuals=(1, 3),
                cf_differs=True,
            ),
            lecturebank_id="141",
        )
        row = _row_map(out["rows"])["blueprint_evidence_count"]
        self.assertEqual(row["noop_value"], 2)
        self.assertIs(row["counterfactual_value"], NA)
        self.assertIs(row["counterfactual_differs_from_noop"], True)
        self.assertIs(row["criterion_change_supported"], NA)
        self.assertTrue(any(d["kind"] == "conflicting_criterion_values" for d in out["diagnostics"]))
        agg = {r["criterion_name"]: r for r in aggregate_outcome_criteria(out["rows"])}
        self.assertEqual(agg["blueprint_evidence_count"]["supported_changes"], 0)
        self.assertEqual(agg["blueprint_evidence_count"]["evaluable_comparisons"], 0)
        self.assertIs(agg["blueprint_evidence_count"]["supported_change_rate"], NA)

    def test_conflicting_false_diff_flags_supported_is_na(self):
        out = extract_outcome_criteria_changes(
            self._nested_conflict_payload(
                name="answer_supported",
                factual=True,
                noop=True,
                counterfactuals=(True, False),
                cf_differs=False,
            ),
            lecturebank_id="141",
        )
        row = _row_map(out["rows"])["answer_supported"]
        self.assertIs(row["counterfactual_value"], NA)
        self.assertIs(row["counterfactual_differs_from_noop"], False)
        self.assertIs(row["criterion_change_supported"], NA)
        agg = {r["criterion_name"]: r for r in aggregate_outcome_criteria(out["rows"])}
        self.assertEqual(agg["answer_supported"]["supported_changes"], 0)
        self.assertEqual(agg["answer_supported"]["evaluable_comparisons"], 0)
        self.assertIs(agg["answer_supported"]["supported_change_rate"], NA)

    def test_identical_nested_duplicates_remain_evaluable(self):
        out = extract_outcome_criteria_changes(
            _experiment(_case([_cmp()], nested=True)),
            lecturebank_id="141",
        )
        rows = out["rows"]
        self.assertEqual(len(rows), 22)
        by_name = _row_map(rows)
        self.assertEqual(by_name["blueprint_evidence_count"]["counterfactual_value"], 1)
        self.assertIs(by_name["blueprint_evidence_count"]["criterion_change_supported"], True)
        self.assertIs(by_name["answer_supported"]["criterion_change_supported"], False)
        self.assertFalse(any(d["kind"] == "conflicting_criterion_values" for d in out["diagnostics"]))
        agg = {r["criterion_name"]: r for r in aggregate_outcome_criteria(rows)}
        self.assertEqual(agg["blueprint_evidence_count"]["supported_changes"], 1)
        self.assertEqual(agg["blueprint_evidence_count"]["evaluable_comparisons"], 1)
        self.assertEqual(agg["answer_supported"]["supported_changes"], 0)
        self.assertEqual(agg["answer_supported"]["evaluable_comparisons"], 1)

    def test_same_artifact_different_checkpoints_are_distinct(self):
        early = _cmp(alias="c_3", ckpt_id="ckpt-c3")
        late = _cmp(alias="c_8", ckpt_id="ckpt-c8")
        rows = extract_outcome_criteria_changes(
            _experiment(_case([early, late])),
            lecturebank_id="141",
        )["rows"]
        names = [r for r in rows if r["criterion_name"] == "blueprint_evidence_count"]
        self.assertEqual(len(names), 2)
        self.assertNotEqual(names[0]["intervention_signature"], names[1]["intervention_signature"])
        self.assertEqual({r["checkpoint_id"] for r in names}, {"ckpt-c3", "ckpt-c8"})

    def test_reference_sweep_rows_are_not_mixed_in(self):
        axis_cmp = _cmp()
        sweep = _cmp(
            target="chunk_00000099:v1",
            replacement="chunk_00000098:v1",
            alias="c_1",
            ckpt_id="ckpt-sweep",
        )
        case = _case([axis_cmp])
        case["conditions"]["axis"]["raw_evidence"]["reference_interventions"] = [sweep]
        rows = extract_outcome_criteria_changes(
            _experiment(case, run_id="run_test"),
            lecturebank_id="141",
        )["rows"]
        self.assertTrue(all("chunk_00000099:v1" not in str(r["target_ref"]) for r in rows))
        self.assertTrue(all(r["checkpoint_id"] != "ckpt-sweep" for r in rows))

    def test_per_criterion_denominators_and_zero_denominator_na(self):
        valid = _cmp()
        invalid = _cmp(
            target="chunk_00000010:v1",
            replacement="chunk_00000011:v1",
            alias="c_1",
            ckpt_id="ckpt-bad",
            evidence="intervention_error",
            deltas=[_delta("only_on_invalid", 0, 0, 1, cf_differs=True)],
        )
        rows = extract_outcome_criteria_changes(
            _experiment(_case([valid, invalid])),
            lecturebank_id="141",
        )["rows"]
        agg = {r["criterion_name"]: r for r in aggregate_outcome_criteria(rows)}
        self.assertEqual(agg["blueprint_evidence_count"]["supported_changes"], 1)
        self.assertEqual(agg["blueprint_evidence_count"]["evaluable_comparisons"], 1)
        self.assertEqual(agg["blueprint_evidence_count"]["supported_change_rate"], 1.0)
        self.assertEqual(agg["answer_supported"]["supported_changes"], 0)
        self.assertEqual(agg["answer_supported"]["evaluable_comparisons"], 1)
        self.assertEqual(agg["answer_supported"]["supported_change_rate"], 0.0)
        self.assertEqual(agg["only_on_invalid"]["evaluable_comparisons"], 0)
        self.assertIs(agg["only_on_invalid"]["supported_change_rate"], NA)
        md = build_report(
            catalog_rows=[{"lecturebank_id": "141"}],
            document_rows=[{"lecturebank_id": "141"}],
            attempt_rows=[],
            comparison_rows=[],
            provenance_rows=[],
            distribution_rows=[],
            round_rows=[],
            overall_rows=[],
            pooled={},
            outcome_criteria_rows=rows,
        )
        self.assertIn("## Outcome-criteria changes", md)
        self.assertIn("does not necessarily change the final acceptance decision", md)
        self.assertIn("`decision_relevant` means the criterion participates in decision logic", md)
        self.assertIn("must not be summed across criteria", md)
        self.assertIn("| only_on_invalid | 0 | 0 | NA |", md)

    def test_existing_criterion_effect_metrics_unchanged(self):
        exp = _experiment(_case([_cmp()]))
        extracted = extract_experiment_metrics(exp, lecturebank_id="141")
        dm = extracted["document_metrics"]
        self.assertIn("criterion_effect_rate", dm)
        self.assertIn("outcome_criteria_changes", extracted)
        self.assertEqual(len(extracted["outcome_criteria_changes"]), 22)
        self.assertEqual(len(extracted["replay_comparisons"]), 1)
        self.assertIs(extracted["replay_comparisons"][0]["criterion_effect_any"], True)

    def test_repeated_export_preserves_other_documents(self):
        rows_141 = extract_outcome_criteria_changes(
            _experiment(_case([_cmp()])),
            lecturebank_id="141",
        )["rows"]
        other = [
            {
                "lecturebank_id": "14",
                "run_id": "run_other",
                "candidate_ref": "q_other:v1",
                "intervention_signature": "REMOVE|chunk_x:v1||ckpt-z",
                "round": 1,
                "operation": "REMOVE",
                "target_ref": "chunk_x:v1",
                "replacement_ref": "",
                "checkpoint_id": "ckpt-z",
                "checkpoint_alias": "c_0",
                "criterion_name": "preexisting_other_doc",
                "factual_value": 9,
                "noop_value": 9,
                "counterfactual_value": 9,
                "decision_relevant": False,
                "noop_differs_from_factual": False,
                "counterfactual_differs_from_noop": False,
                "valid_comparison": True,
                "full_control_stable": True,
                "criterion_change_supported": False,
            }
        ]
        collected = [
            {
                "lecturebank_id": "141",
                "eval_status": "success",
                "document_metrics": {
                    "lecturebank_id": "141",
                    "eval_status": "success",
                    "criterion_effect_rate": 0.4,
                },
                "replay_comparisons": [
                    {"lecturebank_id": "141", "criterion_effect_any": True}
                ],
                "outcome_criteria_changes": rows_141,
            },
            {
                "lecturebank_id": "14",
                "eval_status": "success",
                "document_metrics": {
                    "lecturebank_id": "14",
                    "eval_status": "success",
                    "criterion_effect_rate": 0.1,
                },
                "replay_comparisons": [
                    {"lecturebank_id": "14", "criterion_effect_any": False}
                ],
                "outcome_criteria_changes": other,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics_dir = output_dir / "metrics"
            _persist(
                output_dir,
                metrics_dir,
                output_dir / "lecturebank_eval_manifest.json",
                {"manifest_version": 1, "records": []},
                collected,
            )
            first = load_collected_from_metrics(metrics_dir)
            self.assertEqual(len(first[0]["outcome_criteria_changes"]) + len(first[1]["outcome_criteria_changes"]), 23)
            _persist(
                output_dir,
                metrics_dir,
                output_dir / "lecturebank_eval_manifest.json",
                {"manifest_version": 1, "records": []},
                first,
            )
            again = load_collected_from_metrics(metrics_dir)
            by_id = {str(item["lecturebank_id"]): item for item in again}
            self.assertEqual(len(by_id["141"]["outcome_criteria_changes"]), 22)
            self.assertEqual(len(by_id["14"]["outcome_criteria_changes"]), 1)
            self.assertEqual(
                by_id["14"]["outcome_criteria_changes"][0]["criterion_name"],
                "preexisting_other_doc",
            )
            self.assertEqual(by_id["14"]["replay_comparisons"][0]["criterion_effect_any"], False)
            self.assertEqual(float(by_id["141"]["document_metrics"]["criterion_effect_rate"]), 0.4)
            with (metrics_dir / "outcome_criteria_changes.csv").open(encoding="utf-8") as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(len(exported), 23)
            self.assertEqual(list(exported[0].keys()), OUTCOME_CRITERIA_FIELDS)

    def test_offline_export_makes_no_generation_replay_or_llm_calls(self):
        payload = _experiment(_case([_cmp()], nested=True))
        item = {
            "lecturebank_id": "141",
            "eval_status": "success",
            "document_metrics": {"lecturebank_id": "141", "eval_status": "success"},
            "replay_comparisons": [{"lecturebank_id": "141", "criterion_effect_any": True}],
            "outcome_criteria_changes": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics_dir = output_dir / "metrics"
            write_streamlit_experiment_json(output_dir, "141", payload)
            with patch("mas_langgraph.evaluation.execute.process_documents") as gen_run, patch(
                "mas_langgraph.batch.generate.run_factual_and_explanation_in_memory"
            ) as mas_run, patch(
                "mas_langgraph.evaluation.concept_change.build_concept_change_llm"
            ) as llm_build:
                stats = run_outcome_criteria_from_saved(
                    selected=["141"],
                    output_dir=output_dir,
                    metrics_dir=metrics_dir,
                    collected=[item],
                    persist=lambda *a, **k: _persist(*a, **k),
                    manifest_path=output_dir / "lecturebank_eval_manifest.json",
                    manifest={"manifest_version": 1, "records": []},
                    log=io.StringIO(),
                )
            gen_run.assert_not_called()
            mas_run.assert_not_called()
            llm_build.assert_not_called()
            self.assertEqual(stats["failures"], 0)
            reloaded = load_collected_from_metrics(metrics_dir)
            self.assertEqual(len(reloaded[0]["outcome_criteria_changes"]), 22)
            self.assertIs(reloaded[0]["replay_comparisons"][0]["criterion_effect_any"], True)
            report = (metrics_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("Outcome-criteria changes", report)

    def test_offline_fails_without_saved_json(self):
        item = {
            "lecturebank_id": "141",
            "eval_status": "success",
            "document_metrics": {"lecturebank_id": "141", "eval_status": "success"},
            "replay_comparisons": [],
        }
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("mas_langgraph.evaluation.execute.process_documents") as gen_run:
                stats = run_outcome_criteria_from_saved(
                    selected=["141"],
                    output_dir=Path(tmp),
                    metrics_dir=Path(tmp) / "metrics",
                    collected=[item],
                    persist=lambda *a, **k: None,
                    manifest_path=Path(tmp) / "manifest.json",
                    manifest={},
                    log=log,
                )
        gen_run.assert_not_called()
        self.assertEqual(stats["failures"], 1)
        self.assertIn("will not generate or replay", log.getvalue())

    def test_cli_flag(self):
        parser = build_arg_parser()
        off = parser.parse_args(["--lecturebank-dir", "x", "--id", "1", "--allow-any-count"])
        self.assertFalse(off.outcome_criteria_from_saved)
        on = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "1",
                "--allow-any-count",
                "--outcome-criteria-from-saved",
            ]
        )
        self.assertTrue(on.outcome_criteria_from_saved)

    def test_saved_experiment_regression_if_available(self):
        path = _saved_experiment_path()
        if path is None:
            payload = _experiment(_case([_cmp()], nested=True))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        rows = extract_outcome_criteria_changes(payload, lecturebank_id="141")["rows"]
        first = [
            r
            for r in rows
            if r["candidate_ref"] == "q_00000001:v1"
            and r["target_ref"] == "chunk_00000002:v1"
            and r["replacement_ref"] == "chunk_00000003:v1"
            and r["checkpoint_alias"] == "c_8"
        ]
        self.assertEqual(len(first), 22)
        by_name = _row_map(first)
        ev = by_name["blueprint_evidence_count"]
        self.assertEqual(ev["factual_value"], 2)
        self.assertEqual(ev["noop_value"], 2)
        self.assertEqual(ev["counterfactual_value"], 1)
        self.assertIs(ev["decision_relevant"], True)
        self.assertIs(ev["criterion_change_supported"], True)
        conf = by_name["evidence_confidence"]
        self.assertEqual(conf["factual_value"], 1.0)
        self.assertEqual(conf["noop_value"], 1.0)
        self.assertEqual(conf["counterfactual_value"], 0.0)
        self.assertIs(conf["decision_relevant"], False)
        self.assertIs(conf["criterion_change_supported"], True)
        self.assertEqual(sum(1 for r in first if r["counterfactual_differs_from_noop"] is True), 2)


def _write_outcome_csvs(metrics: Path, rows: List[Dict[str, Any]]) -> None:
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
        outcome_criteria_rows=rows,
    )


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv_names(path: Path) -> List[str]:
    return [r["criterion_name"] for r in _read_csv_rows(path)]


class TestOutcomeCriteriaChangedCsv(unittest.TestCase):
    def test_supported_changes_with_available_values_are_included(self):
        rows = extract_outcome_criteria_changes(
            _experiment(_case([_cmp()])),
            lecturebank_id="141",
        )["rows"]
        complete = {r["criterion_name"] for r in rows}
        self.assertIn("correct_option_count", complete)
        changed = filter_outcome_criteria_changed(rows)
        names = [r["criterion_name"] for r in changed]
        self.assertEqual(names, ["blueprint_evidence_count", "evidence_confidence"])
        self.assertEqual(len(rows), 22)
        self.assertEqual(len(changed), 2)
        by_name = _row_map(changed)
        self.assertEqual(by_name["blueprint_evidence_count"]["counterfactual_value"], 1)
        self.assertEqual(by_name["evidence_confidence"]["counterfactual_value"], 0.0)
        self.assertIs(by_name["evidence_confidence"]["decision_relevant"], False)

    def test_unchanged_and_na_counterfactual_excluded(self):
        deltas = [
            _delta("blueprint_evidence_count", 2, 2, 1, decision_relevant=True, cf_differs=True),
            _delta("correct_option_count", 1),
            _delta("legacy_missing_cf", 4, 4, omit=["counterfactual"], cf_differs=True),
        ]
        rows = extract_outcome_criteria_changes(
            _experiment(_case([_cmp(deltas=deltas)])),
            lecturebank_id="141",
        )["rows"]
        by_name = _row_map(rows)
        self.assertIs(by_name["legacy_missing_cf"]["counterfactual_value"], NA)
        self.assertIs(by_name["legacy_missing_cf"]["criterion_change_supported"], True)
        names = [r["criterion_name"] for r in filter_outcome_criteria_changed(rows)]
        self.assertEqual(names, ["blueprint_evidence_count"])
        self.assertNotIn("correct_option_count", names)
        self.assertNotIn("legacy_missing_cf", names)

    def test_unsupported_and_conflicted_duplicates_excluded(self):
        invalid = _cmp(evidence="noop_error")
        conflicted = extract_outcome_criteria_changes(
            TestOutcomeCriteriaExport()._nested_conflict_payload(
                name="blueprint_evidence_count",
                factual=2,
                noop=2,
                counterfactuals=(1, 3),
                cf_differs=True,
            ),
            lecturebank_id="141",
        )["rows"]
        self.assertEqual(filter_outcome_criteria_changed(conflicted), [])
        invalid_rows = extract_outcome_criteria_changes(
            _experiment(_case([invalid])),
            lecturebank_id="141",
        )["rows"]
        self.assertEqual(filter_outcome_criteria_changed(invalid_rows), [])
        string_false = [
            {
                **invalid_rows[0],
                "criterion_name": "string_false_flag",
                "counterfactual_differs_from_noop": "false",
                "criterion_change_supported": "false",
                "counterfactual_value": 1,
            }
        ]
        self.assertEqual(filter_outcome_criteria_changed(string_false), [])

    def test_false_zero_and_empty_list_are_not_missing(self):
        deltas = [
            _delta("bool_false", True, True, False, cf_differs=True),
            _delta("zero_count", 1, 1, 0, cf_differs=True),
            _delta("empty_flags", ["x"], ["x"], [], cf_differs=True),
        ]
        rows = extract_outcome_criteria_changes(
            _experiment(_case([_cmp(deltas=deltas)])),
            lecturebank_id="141",
        )["rows"]
        changed = filter_outcome_criteria_changed(rows)
        by_name = _row_map(changed)
        self.assertEqual(set(by_name), {"bool_false", "zero_count", "empty_flags"})
        self.assertIs(by_name["bool_false"]["counterfactual_value"], False)
        self.assertEqual(by_name["zero_count"]["counterfactual_value"], 0)
        self.assertEqual(by_name["empty_flags"]["counterfactual_value"], [])

    def test_filter_after_csv_reload_and_empty_overwrite(self):
        rows = extract_outcome_criteria_changes(
            _experiment(_case([_cmp()])),
            lecturebank_id="141",
        )["rows"]
        before_agg = aggregate_outcome_criteria(rows)
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp)
            stale = metrics / "outcome_criteria_changed.csv"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("lecturebank_id,criterion_name\nstale,keep_me\n", encoding="utf-8")
            _write_outcome_csvs(metrics, rows)
            complete = _read_csv_rows(metrics / "outcome_criteria_changes.csv")
            filtered = _read_csv_rows(metrics / "outcome_criteria_changed.csv")
            self.assertEqual(len(complete), 22)
            self.assertEqual(
                _read_csv_names(metrics / "outcome_criteria_changed.csv"),
                ["blueprint_evidence_count", "evidence_confidence"],
            )
            self.assertEqual(list(filtered[0].keys()), OUTCOME_CRITERIA_FIELDS)
            self.assertNotIn(
                "keep_me",
                (metrics / "outcome_criteria_changed.csv").read_text(encoding="utf-8"),
            )
            collected = load_collected_from_metrics(metrics)
            reloaded = list(collected[0]["outcome_criteria_changes"])
            self.assertEqual(
                [r["criterion_name"] for r in filter_outcome_criteria_changed(reloaded)],
                ["blueprint_evidence_count", "evidence_confidence"],
            )
            unchanged = [r for r in rows if r["criterion_name"] == "correct_option_count"]
            _write_outcome_csvs(metrics, unchanged)
            empty = _read_csv_rows(metrics / "outcome_criteria_changed.csv")
            self.assertEqual(empty, [])
            header = (metrics / "outcome_criteria_changed.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header.split(","), OUTCOME_CRITERIA_FIELDS)
            self.assertNotIn(
                "blueprint_evidence_count",
                (metrics / "outcome_criteria_changed.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(len(_read_csv_rows(metrics / "outcome_criteria_changes.csv")), 1)
            self.assertEqual(aggregate_outcome_criteria(rows), before_agg)
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
                outcome_criteria_rows=rows,
            )
            self.assertIn("outcome_criteria_changes.csv", report)
            self.assertIn("outcome_criteria_changed.csv", report)
            self.assertIn(
                "does not necessarily indicate a change in the final acceptance decision",
                report,
            )


if __name__ == "__main__":
    unittest.main()
