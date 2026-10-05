"""Tests for Streamlit-independent explanation-graph UI helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mas_langgraph.tests.test_explanation_graph import make_exported_case
from mas_langgraph.visualization import build_explanation_graph
from mas_langgraph.visualization.ui_helpers import (
    ExperimentPayloadError,
    MISSING_QUESTION_COMPARISON,
    control_stability_explanation,
    download_filename,
    evidence_status_explanation,
    extract_replay_question_views,
    find_local_experiment_files,
    get_case_display_label,
    get_case_key,
    inspector_sections,
    list_case_options,
    mcq_views_equal,
    node_display_label,
    parse_experiment_json_text,
    question_change_summary,
    replay_meaning_sentences,
    replay_selection_explanation,
    resolve_replay_question_comparison,
    resolve_selected_node_id,
    role_explanation,
    role_glossary,
    validate_experiment_payload,
    visible_replay_units,
)


def _experiment_payload():
    case = make_exported_case()
    return {
        "schema_version": 1,
        "run_id": "run_viz",
        "branch_id": "factual",
        "experiment_seed": 42,
        "user_question": "Why was this question selected?",
        "question_count": 1,
        "models": {
            "workflow": {"model": "gpt-4.1-mini-2025-04-14"},
            "cross_cove_verifier": {"model": "gpt-5.4-mini-2026-03-17"},
        },
        "warnings": [],
        "cases": [case],
    }


class TestExperimentValidation(unittest.TestCase):
    def test_valid_payload_accepted(self):
        payload = validate_experiment_payload(_experiment_payload())
        self.assertIn("cases", payload)
        self.assertEqual(len(payload["cases"]), 1)

    def test_missing_cases_rejected(self):
        with self.assertRaises(ExperimentPayloadError) as ctx:
            validate_experiment_payload({"run_id": "x"})
        self.assertIn("explanation experiment", str(ctx.exception))

    def test_malformed_json_rejected(self):
        with self.assertRaises(ExperimentPayloadError):
            parse_experiment_json_text("{not json")


class TestCaseLabels(unittest.TestCase):
    def test_v2_version_visible_in_label_and_key(self):
        case = make_exported_case()
        self.assertEqual(get_case_key(case, 0), "q_00000010:v2")
        label = get_case_display_label(case, 10)
        self.assertIn("q_00000010:v2", label)
        self.assertIn("Q10", label)
        self.assertIn("k-means", label.lower())
        self.assertNotEqual(get_case_key(case, 0), "q_00000010")


class TestNodeSelectorLabels(unittest.TestCase):
    def test_same_human_label_distinguished_by_ref(self):
        graph = build_explanation_graph(
            make_exported_case(), condition="trace_provenance", detail_level="full"
        )
        labels = [node_display_label(n) for n in graph.nodes]
        self.assertEqual(len(labels), len(set(labels)))
        v2 = graph.node_map()["artifact::q_00000010:v2"]
        text = node_display_label(v2)
        self.assertIn("q_00000010:v2", text)


class TestDownloadFilenames(unittest.TestCase):
    def test_deterministic_and_filesystem_safe(self):
        name = download_filename("q_00000010:v2", "axis", "essential", "tex")
        self.assertEqual(name, "q_00000010_v2__axis__essential.tex")
        self.assertNotIn(":", name)
        self.assertNotIn(" ", name)
        self.assertEqual(
            download_filename("q_00000010:v2", "axis", "essential", "tex"),
            name,
        )


class TestLocalFileDiscovery(unittest.TestCase):
    def test_only_expected_pattern_newest_first(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "quiz_explanation_experiment_old.json").write_text("{}", encoding="utf-8")
            newer = root / "quiz_explanation_experiment_new.json"
            newer.write_text("{}", encoding="utf-8")
            older = root / "quiz_explanation_experiment_old.json"
            older.touch()
            import os

            os.utime(older, (1_000_000, 1_000_000))
            os.utime(newer, (2_000_000, 2_000_000))
            (root / "other.json").write_text("{}", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "quiz_explanation_experiment_nested.json").write_text(
                "{}", encoding="utf-8"
            )
            found = find_local_experiment_files([root])
            names = [p.name for p in found]
            self.assertEqual(
                names,
                [
                    "quiz_explanation_experiment_new.json",
                    "quiz_explanation_experiment_old.json",
                ],
            )
            self.assertNotIn("other.json", names)
            self.assertFalse(any("nested" in str(p) for p in found))


class TestSelectedNodeFallback(unittest.TestCase):
    def test_hidden_node_falls_back_after_full_to_essential(self):
        case = make_exported_case()
        full = build_explanation_graph(case, "trace", "full")
        ess = build_explanation_graph(case, "trace", "essential")
        hidden = [nid for nid in full.node_ids() if nid not in set(ess.node_ids())]
        self.assertTrue(hidden)
        fallback = f"artifact::{ess.target_ref}"
        chosen = resolve_selected_node_id(
            hidden[0], ess.node_ids(), fallback_id=fallback
        )
        self.assertIn(chosen, ess.node_ids())
        self.assertEqual(chosen, fallback)
        kept = resolve_selected_node_id(
            fallback, ess.node_ids(), fallback_id=fallback
        )
        self.assertEqual(kept, fallback)


class TestNoGraphSemanticMutation(unittest.TestCase):
    def test_helpers_do_not_alter_graph(self):
        case = make_exported_case()
        graph = build_explanation_graph(case, "axis", "essential")
        before_nodes = graph.node_ids()
        before_edges = graph.edge_ids()
        node = graph.node_map()[graph.node_ids()[0]]
        details_before = json.dumps(dict(node.details), sort_keys=True)
        inspector_sections(node)
        node_display_label(node)
        visible_replay_units(graph)
        list_case_options(_experiment_payload())
        self.assertEqual(graph.node_ids(), before_nodes)
        self.assertEqual(graph.edge_ids(), before_edges)
        self.assertEqual(json.dumps(dict(node.details), sort_keys=True), details_before)
        lo = graph.node_map().get("artifact::lo_1:v1")
        if lo is not None:
            text = json.dumps(inspector_sections(lo))
            self.assertIn("unsupervised learning", text)


_FULL_QCC = {
    "kind": "mcq_content",
    "details": {
        "stem_changed": True,
        "correct_answer_changed": False,
        "distractors_changed": True,
        "stem_effect": "changed",
        "correct_answer_effect": "unchanged",
        "distractor_effect": "changed",
        "factual": {
            "stem": "What is clustering?",
            "correct_answer": "Grouping similar observations",
            "distractors": ["Predicting labels", "Random splits"],
            "answer_explanation": "Factual explanation",
        },
        "noop": {
            "stem": "What is clustering?",
            "correct_answer": "Grouping similar observations",
            "distractors": ["Predicting labels", "Random splits"],
            "answer_explanation": "Factual explanation",
        },
        "counterfactual": {
            "stem": "What is k-means?",
            "correct_answer": "Grouping similar observations",
            "distractors": ["Predicting labels", "A neural network"],
            "answer_explanation": "CF explanation",
        },
    },
}


class TestReplayInterpretationHelpers(unittest.TestCase):
    def test_role_explanations(self):
        self.assertIn("conceptual focus", role_explanation("focus_shaping"))
        self.assertIn("conceptual focus", role_explanation("▲ Focus-shaping"))
        self.assertIn("correct answer", role_explanation("answer_anchoring"))
        self.assertIn("distractors", role_explanation("distractor_shaping"))
        self.assertIn("phrased", role_explanation("formulation_shaping"))
        self.assertIn("semantic information", role_explanation("artifact_level_only"))
        self.assertIn("reproduced", role_explanation("necessary_for_reproduction"))
        self.assertIn("specific supported", role_explanation("no_demonstrated_role"))
        self.assertIn("decision-level", role_explanation("acceptance_relevant"))
        labels = [row[0] for row in role_glossary()]
        self.assertIn("▲ Focus-shaping", labels)
        self.assertIn("Acceptance-relevant", labels)

    def test_evidence_status_explanations(self):
        self.assertIn("criterion", evidence_status_explanation("criterion_effect"))
        self.assertIn("downstream", evidence_status_explanation("downstream_effect"))
        self.assertIn("invalid", evidence_status_explanation("intervention_error"))
        self.assertIn("NOOP", evidence_status_explanation("noop_error"))
        self.assertIn("stable", evidence_status_explanation("replay_unstable"))
        self.assertEqual(evidence_status_explanation("not_a_status"), "")

    def test_control_stability_explanations(self):
        true_text = control_stability_explanation(True)
        self.assertIn("stable replay baseline", true_text)
        self.assertIn("does not by itself prove causality", true_text)
        self.assertIn("cautiously", control_stability_explanation(False))
        self.assertIn("not recorded", control_stability_explanation(None))

    def test_question_comparison_extraction(self):
        views = extract_replay_question_views(_FULL_QCC)
        self.assertIsNotNone(views)
        self.assertEqual(views["factual"]["stem"], "What is clustering?")
        self.assertEqual(views["counterfactual"]["correct_answer"], "Grouping similar observations")
        self.assertEqual(len(views["noop"]["distractors"]), 2)
        self.assertTrue(mcq_views_equal(views["factual"], views["noop"]))
        self.assertFalse(mcq_views_equal(views["factual"], views["counterfactual"]))

    def test_missing_question_comparison_fallback(self):
        self.assertIsNone(extract_replay_question_views({}))
        self.assertIsNone(extract_replay_question_views({"details": {"kind": "mcq_content"}}))
        self.assertEqual(
            MISSING_QUESTION_COMPARISON,
            "Structured question comparison is not available for this replay.",
        )
        unit = {"question_content_comparison": {}, "comparison_index": 0}
        self.assertEqual(resolve_replay_question_comparison(unit, {"conditions": {}}), {})

    def test_change_summary_uses_structured_flags(self):
        rows = dict(
            question_change_summary(
                _FULL_QCC,
                decision_changed=False,
                noop_decision="accept",
                counterfactual_decision="accept",
            )
        )
        self.assertEqual(rows["Stem"], "changed")
        self.assertEqual(rows["Correct answer"], "unchanged")
        self.assertEqual(rows["Distractors"], "changed")
        self.assertEqual(rows["Acceptance"], "unchanged")

    def test_change_summary_compares_when_flags_missing(self):
        qcc = {
            "details": {
                "factual": {
                    "stem": "A",
                    "correct_answer": "X",
                    "distractors": ["d1"],
                },
                "noop": {
                    "stem": "A",
                    "correct_answer": "X",
                    "distractors": ["d1"],
                },
                "counterfactual": {
                    "stem": "B",
                    "correct_answer": "X",
                    "distractors": ["d1"],
                },
            }
        }
        rows = dict(
            question_change_summary(
                qcc,
                decision_changed=None,
                noop_decision="accept",
                counterfactual_decision="reject",
            )
        )
        self.assertEqual(rows["Stem"], "changed")
        self.assertEqual(rows["Correct answer"], "unchanged")
        self.assertEqual(rows["Distractors"], "unchanged")
        self.assertEqual(rows["Acceptance"], "changed")

    def test_resolve_comparison_from_case_raw_evidence(self):
        case = {
            "conditions": {
                "axis": {
                    "raw_evidence": {
                        "replay_comparisons": [
                            {"question_content_comparison": _FULL_QCC},
                        ]
                    }
                }
            }
        }
        unit = {"question_content_comparison": {}, "comparison_index": 0}
        found = resolve_replay_question_comparison(unit, case)
        self.assertEqual(found, _FULL_QCC)

    def test_replay_selection_explanation(self):
        from mas_langgraph.visualization.models import ExplanationGraph

        def meta_graph(**metadata):
            return ExplanationGraph(
                schema_version=1,
                condition="axis",
                detail_level="essential",
                target_ref="q_00000010:v2",
                nodes=(),
                edges=(),
                lanes=(),
                metadata=metadata,
            )

        all_shown = replay_selection_explanation(
            meta_graph(
                selected_replay_indices=[0, 1, 2],
                hidden_replay_count=0,
                replay_count=3,
            )
        )
        self.assertIn("All available replay units are shown", all_shown)
        hidden = replay_selection_explanation(
            meta_graph(
                selected_replay_indices=[0, 1, 2],
                hidden_replay_count=4,
                replay_count=7,
                replay_ranking=[{"index": 0}],
            )
        )
        self.assertIn("presentation/relevance ranking", hidden)
        self.assertIn("not a causal-strength score", hidden)
        missing = replay_selection_explanation(
            meta_graph(selected_replay_indices=[], hidden_replay_count=0, replay_count=0)
        )
        self.assertIn("No ranking metadata", missing)

    def test_visible_units_carry_question_comparison(self):
        graph = build_explanation_graph(
            make_exported_case(), condition="axis", detail_level="essential"
        )
        units = visible_replay_units(graph)
        self.assertTrue(units)
        self.assertIn("question_content_comparison", units[0])
        self.assertIsNone(extract_replay_question_views(units[0]["question_content_comparison"]))
        meaning = replay_meaning_sentences(units[0])
        self.assertTrue(meaning)


class TestStreamlitAppSmoke(unittest.TestCase):
    def test_app_starts_without_workflow(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest is not available")
        app = Path(__file__).resolve().parents[2] / "streamlit_explanation_graphs.py"
        self.assertTrue(app.is_file())
        at = AppTest.from_file(str(app), default_timeout=90)
        at.run()
        self.assertFalse(at.exception, msg=str(at.exception))


if __name__ == "__main__":
    unittest.main()
