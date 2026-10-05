"""Offline tests for LectureBank AXIS evaluation (no provider calls)."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

from mas_langgraph.evaluation.axis_metrics import (
    NA,
    RECOGNIZED_QUERY_OPS,
    checkpoint_position,
    extract_experiment_metrics,
    normalize_operation,
    pool_rates,
    ratio,
)
from mas_langgraph.evaluation.report import build_report
from mas_langgraph.evaluation.runner import _overall_distributions
from mas_langgraph.evaluation.lecturebank_meta import (
    SelectionError,
    load_lecturebank_catalog,
    resolve_selected_ids,
)
from mas_langgraph.evaluation.pdf_source import (
    PdfSourceError,
    download_pdf,
    filename_matches_lecturebank_id,
    find_local_pdfs,
    is_pdf_bytes,
    resolve_lecturebank_pdf,
)


@contextmanager
def _tempdir() -> Iterator[Path]:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        yield Path(tmp.name)
    finally:
        tmp.cleanup()


def _write_alldata(root: Path) -> Path:
    tsv = root / "alldata.tsv"
    tsv.write_text(
        "ID\tInstructor\tTitle\tTopic\tURL\tVenue\tYear\n"
        "14\tAda\tIntro ML\t1\thttp://example.test/14.pdf\tYale\t2016\n"
        "141\tGrace\tDeep Learning\t2\thttp://example.test/141.pdf\tYale\t2018\n"
        "205\tAlan\tMachine Learning Survey\t1\thttp://example.test/205.pdf\tMIT\t2017\n"
        "101\tIda\tIR Basics\t3\thttp://example.test/101.pdf\tCMU\t2015\n",
        encoding="utf-8",
    )
    (root / "taxonomy.csv").write_text("id,name\n1,Machine Learning\n2,Deep Learning\n3,IR\n", encoding="utf-8")
    return tsv


def _axis_case(
    *,
    attempts=None,
    comparisons=None,
    provenance=None,
    artifacts=None,
    summary=None,
    candidate="CandidateMCQ:q1:v1",
):
    return {
        "candidate_ref": {"key": candidate, "logical_id": "q1", "version": 1},
        "conditions": {
            "axis": {
                "evidence_summary": summary or {},
                "raw_evidence": {
                    "interrogation_attempts": attempts or [],
                    "replay_comparisons": comparisons or [],
                    "provenance_relations": provenance or [],
                    "relevant_artifacts": artifacts or [],
                    "termination_reason": "max_rounds",
                },
            }
        },
        "counterfactual_count": len(comparisons or []),
        "successful_replay_count": sum(
            1
            for c in (comparisons or [])
            if c.get("evidence_status") not in {
                None,
                "noop_error",
                "intervention_error",
                "outcome_unavailable",
                "replay_unstable",
            }
        ),
    }


def _cmp(
    *,
    op="REMOVE",
    target="EvidenceChunk:e1:v1",
    alias="c_7",
    ckpt_id="ckpt-7",
    evidence="no_demonstrated_effect",
    decision_cs=True,
    criteria_cs=True,
    content_cs=True,
    noop_decision="accept",
    cf_decision="accept",
    criterion_deltas=None,
    finding=None,
    qcc=None,
    replacement=None,
):
    return {
        "intervention_signature": {
            "operation": op,
            "target_ref": {"key": target},
            "replacement_ref": {"key": replacement} if replacement else None,
            "checkpoint_alias": alias,
            "checkpoint_id": ckpt_id,
            "boundary_raw": "before generate_candidate_mcqs#1",
        },
        "evidence_status": evidence,
        "decision_control_stable": decision_cs,
        "criteria_control_stable": criteria_cs,
        "subject_content_control_stable": content_cs,
        "noop": {"decision": noop_decision},
        "counterfactual": {"decision": cf_decision},
        "criterion_deltas": criterion_deltas or [],
        "information_role_finding": finding or {},
        "question_content_comparison": qcc or {},
    }


class LectureBankMetaTests(unittest.TestCase):
    def test_tsv_header_parse_and_exact_id(self):
        with _tempdir() as root:
            _write_alldata(root)
            catalog = load_lecturebank_catalog(root)
            rec = catalog.require("14")
            self.assertEqual(rec.title, "Intro ML")
            self.assertEqual(rec.instructor, "Ada")
            self.assertEqual(rec.topic_name, "Machine Learning")
            self.assertIsNone(catalog.get("999"))
            self.assertEqual(catalog.require("141").title, "Deep Learning")

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(SelectionError) as ctx:
            resolve_selected_ids(ids_csv="14,141,14", allow_any_count=True)
        self.assertIn("Duplicate", str(ctx.exception))

    def test_normal_mode_requires_exactly_10(self):
        with self.assertRaises(SelectionError) as ctx:
            resolve_selected_ids(ids_csv="1,2,3")
        self.assertIn("exactly 10", str(ctx.exception))
        ids = [str(i) for i in range(10)]
        self.assertEqual(resolve_selected_ids(id_values=ids), ids)

    def test_allow_any_count_smoke(self):
        got = resolve_selected_ids(id_values=["123"], allow_any_count=True)
        self.assertEqual(got, ["123"])
        got = resolve_selected_ids(ids_csv="1,2", allow_any_count=True)
        self.assertEqual(got, ["1", "2"])

    def test_expected_count_100_accepts_exactly_100(self):
        ids = [str(i) for i in range(100)]
        self.assertEqual(resolve_selected_ids(id_values=ids, expected_count=100), ids)

    def test_expected_count_100_rejects_99_or_101(self):
        with self.assertRaises(SelectionError) as ctx:
            resolve_selected_ids(id_values=[str(i) for i in range(99)], expected_count=100)
        self.assertIn("exactly 100", str(ctx.exception))
        with self.assertRaises(SelectionError) as ctx:
            resolve_selected_ids(id_values=[str(i) for i in range(101)], expected_count=100)
        self.assertIn("exactly 100", str(ctx.exception))

    def test_expected_count_duplicates_do_not_satisfy_count(self):
        with self.assertRaises(SelectionError) as ctx:
            resolve_selected_ids(ids_csv=",".join(["1"] * 100), expected_count=100)
        self.assertIn("Duplicate", str(ctx.exception))

    def test_expected_count_and_allow_any_count_conflict(self):
        with self.assertRaises(SelectionError) as ctx:
            resolve_selected_ids(id_values=["1"], allow_any_count=True, expected_count=100)
        self.assertIn("mutually exclusive", str(ctx.exception))


class PdfMatchAndDownloadTests(unittest.TestCase):
    def test_id_14_does_not_match_141(self):
        self.assertTrue(filename_matches_lecturebank_id("14.pdf", "14"))
        self.assertTrue(filename_matches_lecturebank_id("lecturebank_14.pdf", "14"))
        self.assertFalse(filename_matches_lecturebank_id("141.pdf", "14"))
        self.assertFalse(filename_matches_lecturebank_id("14.pdf", "141"))
        self.assertTrue(filename_matches_lecturebank_id("141.pdf", "141"))
        self.assertTrue(filename_matches_lecturebank_id("14_slides.pdf", "14"))
        self.assertFalse(filename_matches_lecturebank_id("141_slides.pdf", "14"))

    def test_find_local_pdfs_does_not_confuse_14_and_141(self):
        with _tempdir() as root:
            (root / "14.pdf").write_bytes(b"%PDF-1.4 dummy")
            (root / "141.pdf").write_bytes(b"%PDF-1.4 dummy")
            hits14 = find_local_pdfs([root], "14")
            hits141 = find_local_pdfs([root], "141")
            self.assertEqual([p.name for p in hits14], ["14.pdf"])
            self.assertEqual([p.name for p in hits141], ["141.pdf"])

    def test_pdf_dir_used_in_place_without_copy(self):
        with _tempdir() as root:
            pdf_dir = root / "pdfs"
            cache = root / "cache"
            pdf_dir.mkdir()
            src = pdf_dir / "14.pdf"
            src.write_bytes(b"%PDF-1.4 dummy")
            info = resolve_lecturebank_pdf(
                lecturebank_id="14",
                url="http://example.test/14.pdf",
                cache_dir=cache,
                pdf_dir=pdf_dir,
                no_download=True,
            )
            self.assertEqual(Path(info["path"]).resolve(), src.resolve())
            self.assertEqual(info["source"], "local")
            self.assertFalse(info["copied"])
            self.assertFalse((cache / "14.pdf").exists())

    def test_html_download_rejected(self):
        self.assertFalse(is_pdf_bytes(b"<!DOCTYPE html><html>not a pdf</html>"))
        self.assertTrue(is_pdf_bytes(b"%PDF-1.7\n..."))

        class _Resp:
            headers = {"Content-Type": "text/html"}

            def read(self):
                return b"<html>login page</html>"

            def geturl(self):
                return "http://example.test/login"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with _tempdir() as root:
            dest = root / "14.pdf"
            with patch(
                "mas_langgraph.evaluation.pdf_source.urlopen", return_value=_Resp()
            ):
                with self.assertRaises(PdfSourceError) as ctx:
                    download_pdf("http://example.test/14", dest)
            self.assertIn("did not return a PDF", str(ctx.exception))
            self.assertFalse(dest.exists())


class QueryDistributionTests(unittest.TestCase):
    def test_normalize_whatif_and_unparsed(self):
        self.assertEqual(normalize_operation("WHATIF"), ("REPLACE", "WHATIF"))
        self.assertEqual(normalize_operation("REPLACE"), ("REPLACE", "REPLACE"))
        self.assertEqual(normalize_operation("remove"), ("REMOVE", "remove"))
        self.assertEqual(normalize_operation(None), ("UNPARSED", ""))
        self.assertEqual(normalize_operation("DONE"), ("DONE", "DONE"))

    def test_query_distribution_and_validity_excludes_what_done(self):
        attempts = [
            {"attempt_index": 0, "generated_query": "REMOVE x", "parsed_operation": "REMOVE", "validation_status": "ok", "executed": True, "produced_counterfactual": True},
            {"attempt_index": 1, "generated_query": "ADD y", "parsed_operation": "ADD", "validation_status": "ok", "executed": True, "produced_counterfactual": True},
            {"attempt_index": 2, "generated_query": "WHATIF z", "parsed_operation": "WHATIF", "validation_status": "ok", "executed": True, "produced_counterfactual": True},
            {"attempt_index": 3, "generated_query": "REPLACE z", "parsed_operation": "REPLACE", "validation_status": "contract_error", "contract_error": "no replace", "executed": False, "produced_counterfactual": False},
            {"attempt_index": 4, "generated_query": "WHAT x", "parsed_operation": "WHAT", "validation_status": "ok", "executed": True, "produced_counterfactual": False},
            {"attempt_index": 5, "generated_query": "DONE", "parsed_operation": "DONE", "validation_status": "ok", "executed": False, "produced_counterfactual": False},
            {"attempt_index": 6, "generated_query": "???", "parsed_operation": None, "validation_status": "parser_error", "parser_error": "bad", "executed": False, "produced_counterfactual": False},
        ]
        exp = {"cases": [_axis_case(attempts=attempts)], "workflow_llm_diagnostics": {}, "timing": {}}
        out = extract_experiment_metrics(exp, {"records": []}, lecturebank_id="14")
        dm = out["document_metrics"]
        self.assertEqual(dm["proposed_remove_count"], 1)
        self.assertEqual(dm["proposed_add_count"], 1)
        self.assertEqual(dm["proposed_replace_count"], 2)  # WHATIF + REPLACE
        self.assertEqual(dm["proposed_what_count"], 1)
        self.assertEqual(dm["proposed_done_count"], 1)
        self.assertEqual(dm["proposed_unparsed_count"], 1)
        # recognized non-DONE = REMOVE+ADD+REPLACE+WHAT = 5
        self.assertEqual(dm["proposed_recognized_non_done_count"], 5)
        self.assertAlmostEqual(dm["proposed_remove_pct"], 20.0)
        self.assertEqual(dm["counterfactual_intervention_proposals"], 4)  # R+A+2 REPLACE, not WHAT/DONE
        self.assertEqual(dm["executed_counterfactual_interventions"], 3)
        self.assertEqual(dm["intervention_validity_rate"], 3 / 4)
        whatif_row = [r for r in out["interrogation_attempts"] if r["parsed_operation"] == "WHATIF"][0]
        self.assertEqual(whatif_row["parsed_operation_normalized"], "REPLACE")
        self.assertEqual(whatif_row["axis_figure4_operation"], "whatif")


def _executed_op_rows(distributions, *, scope="document"):
    return [
        r
        for r in distributions
        if r.get("dimension") == "executed_query_operation"
        and r.get("scope", "document") == scope
        and r.get("value") in RECOGNIZED_QUERY_OPS
    ]


class ExecutedOperationDistributionTests(unittest.TestCase):
    def test_single_executed_remove_is_not_double_counted_with_replay(self):
        attempts = [
            {
                "attempt_index": 0,
                "generated_query": "REMOVE x",
                "parsed_operation": "REMOVE",
                "validation_status": "ok",
                "executed": True,
                "produced_counterfactual": True,
            }
        ]
        exp = {
            "cases": [_axis_case(attempts=attempts, comparisons=[_cmp(op="REMOVE")])],
            "workflow_llm_diagnostics": {},
            "timing": {},
        }
        out = extract_experiment_metrics(exp, {"records": []}, lecturebank_id="14")
        rows = _executed_op_rows(out["distributions"])
        self.assertEqual(len(out["interrogation_attempts"]), 1)
        self.assertEqual(len(out["replay_comparisons"]), 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "REMOVE")
        self.assertEqual(rows[0]["count"], 1)
        self.assertEqual(rows[0]["denominator"], 1)
        self.assertEqual(rows[0]["percentage"], 100.0)
        self.assertEqual(sum(r["count"] for r in rows), rows[0]["denominator"])

        overall = _overall_distributions(out["distributions"])
        overall_rows = _executed_op_rows(overall, scope="overall")
        self.assertEqual(overall_rows[0]["count"], 1)
        self.assertEqual(overall_rows[0]["denominator"], 1)
        self.assertEqual(overall_rows[0]["percentage"], 100.0)

    def test_multiple_executed_operations_use_recognized_denominator(self):
        attempts = [
            {"parsed_operation": "REMOVE", "validation_status": "ok", "executed": True, "produced_counterfactual": True},
            {"parsed_operation": "REMOVE", "validation_status": "ok", "executed": True, "produced_counterfactual": True},
            {"parsed_operation": "ADD", "validation_status": "ok", "executed": True, "produced_counterfactual": True},
            {"parsed_operation": "REPLACE", "validation_status": "ok", "executed": True, "produced_counterfactual": True},
        ]
        comparisons = [
            _cmp(op="REMOVE"),
            _cmp(op="REMOVE", alias="c_1"),
            _cmp(op="ADD", alias="c_2"),
            _cmp(op="REPLACE", alias="c_3"),
        ]
        exp = {
            "cases": [_axis_case(attempts=attempts, comparisons=comparisons)],
            "workflow_llm_diagnostics": {},
            "timing": {},
        }
        out = extract_experiment_metrics(exp, {"records": []}, lecturebank_id="14")
        rows = {r["value"]: r for r in _executed_op_rows(out["distributions"])}
        self.assertEqual(rows["REMOVE"]["count"], 2)
        self.assertEqual(rows["ADD"]["count"], 1)
        self.assertEqual(rows["REPLACE"]["count"], 1)
        denom = 4
        self.assertEqual(rows["REMOVE"]["denominator"], denom)
        self.assertEqual(rows["ADD"]["denominator"], denom)
        self.assertEqual(rows["REPLACE"]["denominator"], denom)
        self.assertEqual(rows["REMOVE"]["percentage"], 50.0)
        self.assertEqual(rows["ADD"]["percentage"], 25.0)
        self.assertEqual(rows["REPLACE"]["percentage"], 25.0)
        self.assertEqual(sum(r["count"] for r in rows.values()), denom)
        self.assertAlmostEqual(sum(r["percentage"] for r in rows.values()), 100.0)
        for row in rows.values():
            self.assertLessEqual(row["percentage"], 100.0)

        overall = _overall_distributions(out["distributions"])
        overall_rows = {r["value"]: r for r in _executed_op_rows(overall, scope="overall")}
        self.assertEqual(sum(r["count"] for r in overall_rows.values()), 4)
        self.assertEqual(overall_rows["REMOVE"]["denominator"], 4)
        self.assertAlmostEqual(sum(r["percentage"] for r in overall_rows.values()), 100.0)


class ReportWordingTests(unittest.TestCase):
    def test_robustness_section_does_not_hardcode_ten_documents(self):
        md = build_report(
            catalog_rows=[{"lecturebank_id": "137"}],
            document_rows=[{"lecturebank_id": "137"}],
            attempt_rows=[],
            comparison_rows=[],
            provenance_rows=[],
            distribution_rows=[],
            round_rows=[],
            overall_rows=[],
            pooled={},
        )
        self.assertNotIn("10-document", md)
        self.assertIn("Descriptive robustness across the selected documents", md)


class CheckpointPositionTests(unittest.TestCase):
    def test_eighth_of_ten_is_80_percent(self):
        index = {"records": [{"alias": f"c_{i}", "checkpoint_id": f"id{i}"} for i in range(10)]}
        pos = checkpoint_position(alias="c_7", checkpoint_index=index)
        self.assertEqual(pos["checkpoint_alias_index"], 7)
        self.assertEqual(pos["checkpoint_ordinal_1based"], 8)
        self.assertEqual(pos["checkpoint_total"], 10)
        self.assertEqual(pos["checkpoint_position_pct"], 80.0)

    def test_progress_first_zero_last_hundred(self):
        index = {"records": [{"alias": f"c_{i}", "checkpoint_id": f"id{i}"} for i in range(10)]}
        first = checkpoint_position(alias="c_0", checkpoint_index=index)
        last = checkpoint_position(alias="c_9", checkpoint_index=index)
        self.assertEqual(first["checkpoint_progress_pct"], 0.0)
        self.assertEqual(last["checkpoint_progress_pct"], 100.0)
        single = checkpoint_position(
            alias="c_0", checkpoint_index={"records": [{"alias": "c_0", "checkpoint_id": "only"}]}
        )
        self.assertIsNone(single["checkpoint_progress_pct"])


class ReplayAndEffectTests(unittest.TestCase):
    def _index(self):
        return {"records": [{"alias": f"c_{i}", "checkpoint_id": f"id{i}"} for i in range(10)]}

    def test_noop_stability_and_missing_not_false(self):
        comparisons = [
            _cmp(decision_cs=True, criteria_cs=True, content_cs=True),
            _cmp(decision_cs=True, criteria_cs=False, content_cs=True, alias="c_3"),
            _cmp(decision_cs=None, criteria_cs=None, content_cs=None, alias="c_2"),
        ]
        exp = {"cases": [_axis_case(comparisons=comparisons)], "workflow_llm_diagnostics": {}, "timing": {}}
        dm = extract_experiment_metrics(exp, self._index(), lecturebank_id="14")["document_metrics"]
        self.assertEqual(dm["noop_decision_stability_true"], 2)
        self.assertEqual(dm["noop_decision_stability_false"], 0)
        self.assertEqual(dm["noop_decision_stability_unavailable"], 1)
        self.assertEqual(dm["noop_decision_stability_rate"], 1.0)
        self.assertEqual(dm["noop_criteria_stability_true"], 1)
        self.assertEqual(dm["noop_criteria_stability_false"], 1)
        self.assertEqual(dm["noop_criteria_stability_unavailable"], 1)
        self.assertEqual(dm["noop_criteria_stability_rate"], 0.5)
        self.assertEqual(dm["full_noop_stability_unavailable"], 1)
        self.assertEqual(dm["full_noop_stability_true"], 1)
        self.assertEqual(dm["full_noop_stability_false"], 1)

    def test_criterion_and_decision_relevant_effects(self):
        deltas = [
            {
                "name": "supported_by_pdf",
                "counterfactual_differs_from_noop": True,
                "decision_relevant": True,
                "noop_differs_from_factual": False,
            },
            {
                "name": "length_ok",
                "counterfactual_differs_from_noop": False,
                "decision_relevant": False,
            },
        ]
        comparisons = [_cmp(criterion_deltas=deltas)]
        rows = extract_experiment_metrics(
            {"cases": [_axis_case(comparisons=comparisons)], "workflow_llm_diagnostics": {}, "timing": {}},
            self._index(),
            lecturebank_id="14",
        )["replay_comparisons"]
        self.assertTrue(rows[0]["criterion_effect_any"])
        self.assertTrue(rows[0]["decision_relevant_criterion_effect"])
        self.assertTrue(rows[0]["criterion_effect_any_supported"])

    def test_decision_effect_uses_noop_vs_counterfactual(self):
        comparisons = [
            _cmp(noop_decision="accept", cf_decision="reject", decision_cs=True),
            _cmp(noop_decision="accept", cf_decision="reject", decision_cs=False, alias="c_1"),
        ]
        rows = extract_experiment_metrics(
            {"cases": [_axis_case(comparisons=comparisons)], "workflow_llm_diagnostics": {}, "timing": {}},
            self._index(),
            lecturebank_id="14",
        )["replay_comparisons"]
        self.assertTrue(rows[0]["decision_effect_raw"])
        self.assertTrue(rows[0]["decision_effect"])
        self.assertTrue(rows[1]["decision_effect_raw"])
        self.assertIsNone(rows[1]["decision_effect"])

    def test_content_effect_extraction(self):
        finding = {
            "information_role": "focus_shaping",
            "information_role_supported": True,
            "causal_classification": "influential_non_decisive",
            "content_effect": True,
            "stem_effect": "changed",
            "correct_answer_effect": "unchanged",
            "distractor_effect": "changed",
            "answer_explanation_effect": "unchanged",
        }
        qcc = {
            "effect_status": "question_content_changed",
            "details": {
                "question_content_effect_status": "question_content_changed",
                "stem_effect": "changed",
                "correct_answer_effect": "unchanged",
                "distractor_effect": "changed",
                "answer_explanation_effect": "unchanged",
                "question_present_in_counterfactual": True,
            },
        }
        rows = extract_experiment_metrics(
            {
                "cases": [_axis_case(comparisons=[_cmp(finding=finding, qcc=qcc)])],
                "workflow_llm_diagnostics": {},
                "timing": {},
            },
            self._index(),
            lecturebank_id="14",
        )["replay_comparisons"]
        self.assertTrue(rows[0]["content_effect_raw"])
        self.assertTrue(rows[0]["content_effect"])
        self.assertEqual(rows[0]["stem_effect"], "changed")
        self.assertEqual(rows[0]["question_content_effect_status"], "question_content_changed")


class DistributionAndProvenanceTests(unittest.TestCase):
    def test_unknown_labels_preserved(self):
        finding = {
            "information_role": "brand_new_role",
            "causal_classification": "future_class_x",
            "information_role_supported": True,
        }
        out = extract_experiment_metrics(
            {
                "cases": [_axis_case(comparisons=[_cmp(finding=finding)])],
                "workflow_llm_diagnostics": {},
                "timing": {},
            },
            {"records": [{"alias": "c_7", "checkpoint_id": "id7"}] * 10},
            lecturebank_id="14",
        )
        roles = {r["value"] for r in out["distributions"] if r["dimension"] == "information_role"}
        causal = {r["value"] for r in out["distributions"] if r["dimension"] == "causal_classification"}
        self.assertIn("brand_new_role", roles)
        self.assertIn("future_class_x", causal)
        self.assertEqual(out["replay_comparisons"][0]["information_role"], "brand_new_role")
        self.assertEqual(out["replay_comparisons"][0]["causal_classification"], "future_class_x")

    def test_provenance_completeness_and_remove_replace_vs_add(self):
        artifacts = [
            {"ref": "EvidenceChunk:e1:v1", "artifact_type": "EvidenceChunk"},
            {"ref": "KnowledgeAtom:a1:v1", "artifact_type": "KnowledgeAtom"},
        ]
        relations = [
            {
                "relation": "produced_by",
                "artifact_ref": {"key": "EvidenceChunk:e1:v1"},
                "producer_occurrence_id": "run:factual:atomize_evidence#1",
                "consumer_occurrence_id": "run:factual:plan_mcq_blueprints#1",
            },
            {
                "relation": "",
                "artifact_ref": {"key": "KnowledgeAtom:a1:v1"},
                "producer_occurrence_id": "run:factual:x#1",
                "consumer_occurrence_id": "",
            },
        ]
        comparisons = [
            _cmp(op="REMOVE", target="EvidenceChunk:e1:v1"),
            _cmp(op="REPLACE", target="KnowledgeAtom:a1:v1", alias="c_3"),
            _cmp(op="ADD", target="EvidenceChunk:new:v1", alias="c_4", replacement="EvidenceChunk:new:v1"),
        ]
        out = extract_experiment_metrics(
            {
                "cases": [
                    _axis_case(
                        comparisons=comparisons,
                        provenance=relations,
                        artifacts=artifacts,
                    )
                ],
                "workflow_llm_diagnostics": {},
                "timing": {},
            },
            {"records": [{"alias": f"c_{i}"} for i in range(10)]},
            lecturebank_id="14",
        )
        dm = out["document_metrics"]
        self.assertEqual(dm["provenance_relation_count"], 2)
        self.assertEqual(dm["provenance_complete_relation_count"], 1)
        self.assertEqual(dm["provenance_relation_completeness_rate"], 0.5)
        self.assertEqual(out["provenance_edges"][0]["producer_node"], "atomize_evidence")
        self.assertEqual(out["provenance_edges"][0]["consumer_node"], "plan_mcq_blueprints")
        # REMOVE + REPLACE are eligible; ADD is not required in factual slice
        self.assertEqual(dm["intervention_target_in_slice_denominator"], 2)
        self.assertEqual(dm["intervention_target_in_slice_count"], 2)
        self.assertEqual(dm["intervention_target_in_slice_rate"], 1.0)

    def test_add_not_required_in_factual_slice(self):
        comparisons = [_cmp(op="ADD", target="EvidenceChunk:brand_new:v1", replacement="EvidenceChunk:brand_new:v1")]
        dm = extract_experiment_metrics(
            {
                "cases": [
                    _axis_case(
                        comparisons=comparisons,
                        provenance=[],
                        artifacts=[{"ref": "EvidenceChunk:e1:v1", "artifact_type": "EvidenceChunk"}],
                    )
                ],
                "workflow_llm_diagnostics": {},
                "timing": {},
            },
            {"records": [{"alias": "c_0"}]},
            lecturebank_id="14",
        )["document_metrics"]
        self.assertEqual(dm["intervention_target_in_slice_denominator"], 0)
        self.assertIsNone(dm["intervention_target_in_slice_rate"])


class DeterminismAndPoolingTests(unittest.TestCase):
    def test_replay_determinism_and_zero_denominator_na(self):
        diag = {
            "factual_llm_request_count": 10,
            "factual_llm_unique_request_count": 8,
            "factual_llm_memo_hits": 2,
            "noop_llm_request_count": 4,
            "noop_factual_response_hits": 4,
            "noop_llm_request_misses": 0,
            "noop_live_provider_calls": 0,
            "counterfactual_llm_request_count": 5,
            "counterfactual_factual_response_hits": 2,
            "counterfactual_novel_request_count": 3,
            "counterfactual_memo_hits": 1,
            "counterfactual_live_provider_calls": 3,
        }
        dm = extract_experiment_metrics(
            {"cases": [], "workflow_llm_diagnostics": diag, "timing": {}},
            {"records": []},
            lecturebank_id="14",
        )["document_metrics"]
        self.assertEqual(dm["noop_factual_ledger_hit_rate"], 1.0)
        self.assertEqual(dm["cf_novel_request_rate"], 3 / 5)
        empty = extract_experiment_metrics(
            {
                "cases": [],
                "workflow_llm_diagnostics": {
                    "noop_llm_request_count": 0,
                    "noop_factual_response_hits": 0,
                    "counterfactual_llm_request_count": 0,
                    "counterfactual_live_provider_calls": 0,
                },
                "timing": {},
            },
            {"records": []},
            lecturebank_id="14",
        )["document_metrics"]
        self.assertIsNone(empty["noop_factual_ledger_hit_rate"])
        self.assertIsNone(empty["cf_live_provider_rate"])
        self.assertIsNone(ratio(1, 0))
        self.assertIsNone(ratio(0, 0))
        self.assertIs(NA, None)

    def test_pooled_rates_sum_counts_not_average_percentages(self):
        rows = [
            {"hits": 1, "reqs": 2},  # 50%
            {"hits": 8, "reqs": 8},  # 100%
        ]
        # naive mean of percentages = 75%; pooled = 9/10 = 90%
        self.assertEqual(pool_rates(rows, "hits", "reqs"), 0.9)
        self.assertNotAlmostEqual(pool_rates(rows, "hits", "reqs"), 0.75)


class CliHelpSmokeTests(unittest.TestCase):
    def test_cli_parser_exists(self):
        from mas_langgraph.evaluation.runner import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--list-ids",
                "--search",
                "machine learning",
                "--limit",
                "50",
            ]
        )
        self.assertTrue(args.list_ids)
        self.assertEqual(args.search, "machine learning")
        args2 = parser.parse_args(
            ["--lecturebank-dir", "x", "--id", "123", "--allow-any-count", "--max-questions", "1"]
        )
        self.assertTrue(args2.allow_any_count)
        self.assertEqual(args2.max_questions, 1)
        self.assertIsNone(args2.max_explain_questions)
        self.assertEqual(args2.axis_interrogation_mode, "constrained")
        self.assertFalse(args2.export_quiz)
        self.assertFalse(args2.export_streamlit_json)
        self.assertFalse(args2.build_effect_reference_set)
        self.assertFalse(args2.evaluate_concept_change)
        self.assertFalse(args2.diagnose_hybrid_add)
        self.assertFalse(args2.outcome_criteria_from_saved)
        self.assertFalse(args2.evaluate_saquet)
        self.assertFalse(args2.saquet_from_saved)
        self.assertFalse(args2.reevaluate_saquet)
        self.assertIsNone(args2.reference_max_interventions_per_question)
        args_export = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "123",
                "--allow-any-count",
                "--export-quiz",
            ]
        )
        self.assertTrue(args_export.export_quiz)
        self.assertFalse(args_export.export_streamlit_json)
        args_streamlit = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "123",
                "--allow-any-count",
                "--export-streamlit-json",
            ]
        )
        self.assertTrue(args_streamlit.export_streamlit_json)
        args_concept = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "123",
                "--allow-any-count",
                "--evaluate-concept-change",
            ]
        )
        self.assertTrue(args_concept.evaluate_concept_change)
        args_diag = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "123",
                "--allow-any-count",
                "--diagnose-hybrid-add",
            ]
        )
        self.assertTrue(args_diag.diagnose_hybrid_add)
        args_addsrc = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "123",
                "--allow-any-count",
                "--hybrid-add-from-source",
                "--hybrid-add-source-limit",
                "5",
            ]
        )
        self.assertTrue(args_addsrc.hybrid_add_from_source)
        self.assertEqual(args_addsrc.hybrid_add_source_limit, 5)
        args_default = parser.parse_args(
            ["--lecturebank-dir", "x", "--id", "123", "--allow-any-count"]
        )
        self.assertFalse(args_default.hybrid_add_from_source)
        args_oc = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "123",
                "--allow-any-count",
                "--outcome-criteria-from-saved",
            ]
        )
        self.assertTrue(args_oc.outcome_criteria_from_saved)
        args_hybrid = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "123",
                "--allow-any-count",
                "--axis-interrogation-mode",
                "hybrid",
            ]
        )
        self.assertEqual(args_hybrid.axis_interrogation_mode, "hybrid")
        args_ref = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "123",
                "--allow-any-count",
                "--axis-interrogation-mode",
                "hybrid",
                "--build-effect-reference-set",
                "--reference-max-interventions-per-question",
                "3",
            ]
        )
        self.assertTrue(args_ref.build_effect_reference_set)
        self.assertEqual(args_ref.reference_max_interventions_per_question, 3)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--lecturebank-dir",
                    "x",
                    "--id",
                    "123",
                    "--allow-any-count",
                    "--axis-interrogation-mode",
                    "freeform",
                ]
            )
        args3 = parser.parse_args(
            ["--lecturebank-dir", "x", "--ids-file", "lecturebank_100_ids.txt", "--expected-count", "100"]
        )
        self.assertEqual(args3.expected_count, 100)
        self.assertTrue(args3.temporary_vector_stores is False)
        args4 = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--extraction-jsonl",
                r"..\lecturebank\pdf_extraction_results.jsonl",
                "--id",
                "137",
                "--allow-any-count",
            ]
        )
        self.assertEqual(
            args4.extraction_jsonl,
            r"..\lecturebank\pdf_extraction_results.jsonl",
        )
        args5 = parser.parse_args(
            [
                "--extraction-jsonl",
                r"..\lecturebank\pdf_extraction_results.jsonl",
                "--jsonl-authoritative",
                "--first-jsonl",
                "10",
                "--allow-any-count",
            ]
        )
        self.assertTrue(args5.jsonl_authoritative)
        self.assertEqual(args5.first_jsonl, 10)
        self.assertIsNone(args5.lecturebank_dir)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--lecturebank-dir",
                    "x",
                    "--id",
                    "1",
                    "--expected-count",
                    "100",
                    "--allow-any-count",
                ]
            )


if __name__ == "__main__":
    unittest.main()
