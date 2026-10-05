"""Presentation-layer artifact resolution and concept-change matching."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from mas_langgraph.tests.test_explanation_graph import make_exported_case
from mas_langgraph.visualization import build_explanation_graph
from mas_langgraph.visualization.case_inspect import (
    ARTIFACT_CONTENT_UNAVAILABLE,
    CONCEPT_CHANGE_UNASSESSED,
    USER_QUESTION_UNAVAILABLE,
    artifact_content_view,
    case_user_question,
    display_user_question,
    interrogation_attempt_views,
    intervention_signature_key,
    match_concept_change_for_query,
    match_concept_change_for_unit,
    replay_artifact_views,
    resolve_artifact_record,
)
from mas_langgraph.visualization.ui_helpers import visible_replay_units


def _ref(logical_id: str, version: int = 1) -> dict:
    return {
        "logical_id": logical_id,
        "version": version,
        "key": f"{logical_id}:v{version}",
    }


def _art(logical_id, artifact_type, content, version=1, branch="factual"):
    return {
        "ref": _ref(logical_id, version),
        "artifact_type": artifact_type,
        "branch_id": branch,
        "content": content,
    }


def _sig(operation, target, checkpoint, replacement=None):
    target_ref = _ref(*target) if isinstance(target, tuple) else {"key": target}
    replacement_ref = None
    if replacement is not None:
        replacement_ref = (
            _ref(*replacement) if isinstance(replacement, tuple) else {"key": replacement}
        )
    return {
        "operation": operation,
        "target_ref": target_ref,
        "replacement_ref": replacement_ref,
        "checkpoint_id": checkpoint,
        "checkpoint_alias": checkpoint,
    }


def _round(query, operation, target, checkpoint, replacement=None, comparison=None):
    sig = _sig(operation, target, checkpoint, replacement)
    return {
        "kind": "intervention",
        "generated_query": query,
        "resolved": {
            "operation": operation,
            "target_ref": sig["target_ref"],
            "replacement_ref": sig["replacement_ref"],
            "checkpoint_id": checkpoint,
        },
        "signature": sig,
        "operation": operation,
        "target_ref": sig["target_ref"],
        "replacement_ref": sig["replacement_ref"],
        "checkpoint_id": checkpoint,
        "replay_comparison": comparison,
    }


def _cmp(sig, *, nested=None, flat_label=None, flat_reason=None):
    row = {"intervention_signature": sig}
    if nested is not None:
        row["concept_change_evaluation"] = nested
    if flat_label is not None:
        row["concept_change_label"] = flat_label
        row["concept_change_reason"] = flat_reason
    return row


def _attempt(index, query, operation, *, executed=True, status="executed"):
    return {
        "attempt_index": index,
        "generated_query": query,
        "parsed_operation": operation,
        "validation_status": status,
        "executed": executed,
    }


def _nested_case():
    chunk_target = _art(
        "chunk_00000002",
        "EvidenceChunk",
        {
            "chunk_id": "chunk_00000002",
            "content": "Unsupervised learning tries to understand structure in data.",
            "metadata": {"source_file": "141.pdf", "page_number": 3},
        },
    )
    chunk_repl = _art(
        "chunk_00000003",
        "EvidenceChunk",
        {
            "chunk_id": "chunk_00000003",
            "content": "Clustering splits data into discrete clusters.",
            "metadata": {"source_file": "141.pdf", "page_number": 4},
        },
    )
    chunk_v2 = _art(
        "chunk_00000002",
        "EvidenceChunk",
        {"content": "WRONG VERSION TEXT that must not be substituted."},
        version=2,
    )
    atom = _art(
        "atom_00000001",
        "KnowledgeAtom",
        {
            "content": "Unsupervised learning does not use labeled outputs.",
            "citations": [{"source": "141.pdf", "page": 3, "chunk_id": "chunk_00000002"}],
        },
    )
    blueprint = _art(
        "bp_00000001",
        "MCQBlueprint",
        {
            "generation_goal": "Assess recall of unsupervised learning.",
            "question_type": "basic_definition",
            "difficulty": "beginner",
            "bloom_level": "Remembering",
            "target_atom_ids": ["atom_00000001"],
        },
    )
    candidate = _art(
        "q_00000001",
        "CandidateMCQ",
        {
            "stem": "What is the main goal of unsupervised learning?",
            "options": [
                {"option_id": "A", "text": "Find structure without labels", "is_correct": True},
                {"option_id": "B", "text": "Use labeled outputs", "is_correct": False},
            ],
            "correct_option_id": "A",
            "answer_explanation": "Definition from the lecture.",
        },
    )
    ckpt_a = "ckpt-plan"
    ckpt_b = "ckpt-verify"
    whatif_sig = _sig("WHATIF", ("chunk_00000002", 1), ckpt_a, ("chunk_00000003", 1))
    remove_plan = _sig("REMOVE", ("atom_00000001", 1), ckpt_a)
    remove_verify = _sig("REMOVE", ("atom_00000001", 1), ckpt_b)
    whatif_cmp = _cmp(
        whatif_sig,
        nested={
            "label": "changed_concept",
            "reason": "NOOP tests unsupervised structure; CF tests clustering.",
            "noop_tested_concept": "goal of unsupervised learning",
            "counterfactual_tested_concept": "definition of clustering",
        },
    )
    remove_plan_cmp = _cmp(
        remove_plan,
        nested={
            "label": "same_concept",
            "reason": "Both still test unsupervised learning.",
        },
    )
    remove_verify_cmp = _cmp(
        remove_verify,
        nested={
            "label": "not_evaluable",
            "reason": "question_not_reproduced",
            "eligibility_reason": "question_not_reproduced",
        },
    )
    query_whatif = "WHATIF(chunk_00000002:v1, chunk_00000003:v1, before plan)"
    query_plan = "REMOVE(atom_00000001:v1, before plan)"
    query_verify = "REMOVE(atom_00000001:v1, before verify)"
    return {
        "candidate_ref": _ref("q_00000001"),
        "branch_id": "factual",
        "user_question": "Why was this question selected?",
        "question": {"stem": "What is the main goal of unsupervised learning?"},
        "conditions": {
            "axis": {
                "raw_evidence": {
                    "interrogation_attempts": [
                        _attempt(1, query_whatif, "WHATIF"),
                        _attempt(2, query_plan, "REMOVE"),
                        _attempt(3, query_verify, "REMOVE"),
                        _attempt(4, "REMOVE(atom_00000001:v1, before plan)", "REMOVE", executed=False, status="duplicate"),
                        _attempt(5, "DONE", "DONE", executed=False, status="done"),
                    ],
                    "replay_comparisons": [whatif_cmp, remove_plan_cmp, remove_verify_cmp],
                    "axis_report": {
                        "query_rounds": [
                            _round(query_whatif, "WHATIF", ("chunk_00000002", 1), ckpt_a, ("chunk_00000003", 1), whatif_cmp),
                            _round(query_plan, "REMOVE", ("atom_00000001", 1), ckpt_a, comparison=remove_plan_cmp),
                            _round(query_verify, "REMOVE", ("atom_00000001", 1), ckpt_b, comparison=remove_verify_cmp),
                        ],
                        "replay_comparisons": [whatif_cmp, remove_plan_cmp, remove_verify_cmp],
                    },
                }
            }
        },
        "visualization_data": {
            "schema_version": 1,
            "artifacts": [chunk_target, chunk_repl, chunk_v2, atom, blueprint, candidate],
        },
    }


class TestUserQuestion(unittest.TestCase):
    def test_case_question_preferred_over_payload_and_stem(self):
        case = {
            "user_question": "Why was this question selected?",
            "question": {"stem": "What is clustering?"},
        }
        payload = {"user_question": "Fallback experiment question"}
        self.assertEqual(
            case_user_question(case, payload), "Why was this question selected?"
        )
        self.assertNotIn("clustering", case_user_question(case, payload).lower())

    def test_missing_question_message(self):
        self.assertEqual(display_user_question({}, {}), USER_QUESTION_UNAVAILABLE)


class TestArtifactResolution(unittest.TestCase):
    def test_type_aware_content_and_exact_version(self):
        case = _nested_case()
        chunk = resolve_artifact_record(case, "chunk_00000002:v1")
        self.assertIsNotNone(chunk)
        view = artifact_content_view(chunk, ref="chunk_00000002:v1", role="Target")
        self.assertIn("structure in data", view["body"])
        self.assertEqual(dict(view["fields"])["Source"], "141.pdf")
        self.assertEqual(dict(view["fields"])["Page"], "3")

        v2 = resolve_artifact_record(case, "chunk_00000002:v2")
        self.assertIn("WRONG VERSION", v2["content"]["content"])
        v1_again = resolve_artifact_record(case, "chunk_00000002:v1")
        self.assertNotIn("WRONG VERSION", v1_again["content"]["content"])

        atom = artifact_content_view(
            resolve_artifact_record(case, "atom_00000001:v1"),
            ref="atom_00000001:v1",
        )
        self.assertIn("labeled outputs", atom["body"])
        self.assertTrue(any("141.pdf" in line for line in atom["lines"]))

        blueprint = artifact_content_view(
            resolve_artifact_record(case, "bp_00000001:v1"),
            ref="bp_00000001:v1",
        )
        self.assertIn("Assess recall", blueprint["body"])
        self.assertTrue(any(k == "question_type" for k, _ in blueprint["fields"]))

        candidate = artifact_content_view(
            resolve_artifact_record(case, "q_00000001:v1"),
            ref="q_00000001:v1",
        )
        self.assertIn("main goal", candidate["body"])
        self.assertTrue(any("Correct answer" == k for k, _ in candidate["fields"]))
        self.assertTrue(any("Find structure" in line for line in candidate["lines"]))

    def test_missing_artifact_does_not_use_question_or_explanation(self):
        case = _nested_case()
        missing = resolve_artifact_record(case, "chunk_missing:v1")
        self.assertIsNone(missing)
        view = artifact_content_view(missing, ref="chunk_missing:v1", role="Replacement")
        self.assertFalse(view["available"])
        self.assertEqual(view["body"], ARTIFACT_CONTENT_UNAVAILABLE)
        self.assertNotIn("main goal", view["body"])

    def test_branch_specific_record_preferred(self):
        case = _nested_case()
        case["visualization_data"]["artifacts"].append(
            _art(
                "chunk_00000002",
                "EvidenceChunk",
                {"content": "Counterfactual-branch text"},
                branch="counterfactual",
            )
        )
        factual = resolve_artifact_record(case, "chunk_00000002:v1", branch_id="factual")
        self.assertIn("structure in data", factual["content"]["content"])
        cf = resolve_artifact_record(case, "chunk_00000002:v1", branch_id="counterfactual")
        self.assertEqual(cf["content"]["content"], "Counterfactual-branch text")

    def test_whatif_original_and_replacement_are_independent(self):
        case = _nested_case()
        unit = {
            "operation": "WHATIF",
            "target_ref": "chunk_00000002:v1",
            "replacement_ref": "chunk_00000003:v1",
            "intervention_signature": _sig(
                "WHATIF", ("chunk_00000002", 1), "ckpt-plan", ("chunk_00000003", 1)
            ),
        }
        views = replay_artifact_views(case, unit)
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0]["role"], "Original")
        self.assertEqual(views[0]["section_label"], "Original artifact")
        self.assertEqual(views[0]["ref"], "chunk_00000002:v1")
        self.assertEqual(views[1]["role"], "Replacement")
        self.assertEqual(views[1]["section_label"], "Replacement artifact")
        self.assertEqual(views[1]["ref"], "chunk_00000003:v1")
        self.assertIn("structure in data", views[0]["body"])
        self.assertIn("discrete clusters", views[1]["body"])
        self.assertNotEqual(views[0]["body"], views[1]["body"])

    def test_replace_alias_uses_same_whatif_layout(self):
        case = _nested_case()
        unit = {
            "operation": "REPLACE",
            "target_ref": "chunk_00000002:v1",
            "replacement_ref": "chunk_00000003:v1",
        }
        views = replay_artifact_views(case, unit)
        self.assertEqual([v["role"] for v in views], ["Original", "Replacement"])
        self.assertIn("discrete clusters", views[1]["body"])

    def test_whatif_missing_replacement_keeps_ref_and_unavailable(self):
        case = _nested_case()
        unit = {
            "operation": "WHATIF",
            "target_ref": "chunk_00000002:v1",
            "replacement_ref": "chunk_missing:v1",
        }
        views = replay_artifact_views(case, unit)
        self.assertEqual(len(views), 2)
        self.assertEqual(views[1]["ref"], "chunk_missing:v1")
        self.assertFalse(views[1]["available"])
        self.assertEqual(views[1]["body"], ARTIFACT_CONTENT_UNAVAILABLE)
        self.assertIn("structure in data", views[0]["body"])
        self.assertNotIn("structure in data", views[1]["body"])

    def test_remove_shows_only_target(self):
        case = _nested_case()
        views = replay_artifact_views(
            case,
            {
                "operation": "REMOVE",
                "target_ref": "atom_00000001:v1",
                "replacement_ref": None,
            },
        )
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["role"], "Target")
        self.assertEqual(views[0]["ref"], "atom_00000001:v1")
        self.assertIn("labeled outputs", views[0]["body"])

    def test_historical_export_without_visualization_data(self):
        case = {"branch_id": "factual", "conditions": {"axis": {"raw_evidence": {}}}}
        record = resolve_artifact_record(case, "chunk_00000002:v1")
        self.assertIsNone(record)
        views = replay_artifact_views(case, {"target_ref": "chunk_00000002:v1"})
        self.assertEqual(views[0]["body"], ARTIFACT_CONTENT_UNAVAILABLE)
        self.assertEqual(interrogation_attempt_views(case), [])


class TestConceptChangeMatching(unittest.TestCase):
    def test_nested_query_rounds_and_distinct_checkpoints(self):
        case = _nested_case()
        whatif = match_concept_change_for_query(
            case, "WHATIF(chunk_00000002:v1, chunk_00000003:v1, before plan)"
        )
        plan = match_concept_change_for_query(case, "REMOVE(atom_00000001:v1, before plan)")
        verify = match_concept_change_for_query(
            case, "REMOVE(atom_00000001:v1, before verify)"
        )
        self.assertEqual(whatif["label"], "changed_concept")
        self.assertEqual(plan["label"], "same_concept")
        self.assertEqual(verify["label"], "not_evaluable")
        self.assertEqual(verify["reason_display"], "The question was not reproduced.")
        self.assertNotEqual(plan["reason_display"], verify["reason_display"])
        sig_plan = intervention_signature_key(_sig("REMOVE", ("atom_00000001", 1), "ckpt-plan"))
        sig_verify = intervention_signature_key(
            _sig("REMOVE", ("atom_00000001", 1), "ckpt-verify")
        )
        self.assertNotEqual(sig_plan, sig_verify)

    def test_flat_query_rounds_and_flattened_assessment(self):
        sig = _sig("REMOVE", ("bp_00000001", 1), "ckpt-gen")
        case = {
            "conditions": {
                "axis": {
                    "raw_evidence": {
                        "query_rounds": [
                            _round(
                                "REMOVE(bp_00000001:v1, before gen)",
                                "REMOVE",
                                ("bp_00000001", 1),
                                "ckpt-gen",
                            )
                        ],
                        "replay_comparisons": [
                            _cmp(
                                sig,
                                flat_label="uncertain",
                                flat_reason="The two stems are too different to judge.",
                            )
                        ],
                    }
                }
            }
        }
        found = match_concept_change_for_query(case, "REMOVE(bp_00000001:v1, before gen)")
        self.assertEqual(found["label"], "uncertain")
        self.assertEqual(found["label_display"], "Uncertain")
        self.assertIn("too different", found["reason_display"])

    def test_duplicate_top_level_and_nested_copies_do_not_change_label(self):
        case = _nested_case()
        rows = interrogation_attempt_views(case)
        executed = [r for r in rows if r["attach_assessment"]]
        self.assertEqual(len(executed), 3)
        labels = [r["assessment"]["label"] for r in executed]
        self.assertEqual(labels, ["changed_concept", "same_concept", "not_evaluable"])

    def test_rejected_and_done_do_not_receive_assessments(self):
        case = _nested_case()
        rows = interrogation_attempt_views(case)
        rejected = [r for r in rows if r["parsed_operation"] == "REMOVE" and not r["attach_assessment"]]
        done = [r for r in rows if r["parsed_operation"] == "DONE"]
        self.assertTrue(rejected)
        self.assertTrue(done)
        self.assertIsNone(rejected[0]["assessment"])
        self.assertIsNone(done[0]["assessment"])
        self.assertFalse(rejected[0]["attach_assessment"])
        self.assertFalse(done[0]["attach_assessment"])

    def test_missing_assessment_is_explicit(self):
        case = _nested_case()
        case["conditions"]["axis"]["raw_evidence"]["replay_comparisons"] = []
        case["conditions"]["axis"]["raw_evidence"]["axis_report"]["replay_comparisons"] = []
        for qr in case["conditions"]["axis"]["raw_evidence"]["axis_report"]["query_rounds"]:
            qr["replay_comparison"] = None
        rows = interrogation_attempt_views(case)
        executed = [r for r in rows if r["attach_assessment"]]
        self.assertTrue(all(r["assessment"] is None for r in executed))
        self.assertEqual(
            executed[0]["concept_change_display"]["label_display"],
            CONCEPT_CHANGE_UNASSESSED,
        )

    def test_list_position_is_not_used(self):
        case = _nested_case()
        raw = case["conditions"]["axis"]["raw_evidence"]
        raw["replay_comparisons"] = list(reversed(raw["replay_comparisons"]))
        whatif = match_concept_change_for_query(
            case, "WHATIF(chunk_00000002:v1, chunk_00000003:v1, before plan)"
        )
        self.assertEqual(whatif["label"], "changed_concept")

    def test_replay_unit_signature_match(self):
        case = _nested_case()
        unit = {
            "operation": "REMOVE",
            "target_ref": "atom_00000001:v1",
            "replacement_ref": None,
            "checkpoint_id": "ckpt-verify",
            "intervention_signature": _sig("REMOVE", ("atom_00000001", 1), "ckpt-verify"),
        }
        found = match_concept_change_for_unit(case, unit)
        self.assertEqual(found["label"], "not_evaluable")
        other = {
            "operation": "REMOVE",
            "target_ref": "atom_00000001:v1",
            "checkpoint_id": "ckpt-plan",
            "intervention_signature": _sig("REMOVE", ("atom_00000001", 1), "ckpt-plan"),
        }
        self.assertEqual(match_concept_change_for_unit(case, other)["label"], "same_concept")


class TestNestedExportSample(unittest.TestCase):
    def _sample_path(self) -> Path:
        named = Path(
            r"c:\Users\anasa\Documents\Hochschule\Master Thesis\code\EMaES\rag"
        ) / "quiz_explanation_experiment_141(2).json"
        if named.is_file():
            return named
        return (
            Path(__file__).resolve().parents[2]
            / "lecturebank_axis_eval_141_concept_check"
            / "streamlit"
            / "quiz_explanation_experiment_141.json"
        )

    def test_sample_json_if_present(self):
        path = self._sample_path()
        if not path.is_file():
            self.skipTest("representative experiment JSON is not available")
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = [c for c in payload.get("cases") or [] if isinstance(c, dict)]
        self.assertGreaterEqual(len(cases), 2)
        questions = {display_user_question(case, payload) for case in cases}
        self.assertTrue(questions)
        self.assertNotIn(USER_QUESTION_UNAVAILABLE, questions)
        first = cases[0]
        chunk = resolve_artifact_record(first, "chunk_00000002:v1")
        self.assertIsNotNone(chunk)
        self.assertIn("Unsupervised Learning", chunk["content"]["content"])
        atom = resolve_artifact_record(first, "atom_00000001:v1")
        self.assertIsNotNone(atom)
        blueprint = resolve_artifact_record(first, "bp_00000001:v1")
        self.assertIsNotNone(blueprint)
        candidate = resolve_artifact_record(first, "q_00000001:v1")
        self.assertIsNotNone(candidate)
        second = cases[1]
        replacement = resolve_artifact_record(second, "chunk_00000003:v1")
        self.assertIsNotNone(replacement)
        views = interrogation_attempt_views(first)
        executed = [row for row in views if row["attach_assessment"]]
        self.assertEqual(len(executed), 5)
        labels = {row["assessment"]["label"] for row in executed if row["assessment"]}
        self.assertTrue(labels)
        graph = build_explanation_graph(first, condition="axis", detail_level="full")
        units = visible_replay_units(graph)
        self.assertTrue(units)
        whatif = next((u for u in units if str(u.get("operation")) in {"WHATIF", "REPLACE"}), None)
        if whatif is not None:
            artifact_views = replay_artifact_views(first, whatif, graph=graph)
            self.assertEqual(len(artifact_views), 2)
            self.assertEqual(artifact_views[0]["role"], "Original")
            self.assertEqual(artifact_views[1]["role"], "Replacement")
            self.assertEqual(artifact_views[0]["ref"], whatif["target_ref"])
            self.assertEqual(artifact_views[1]["ref"], whatif["replacement_ref"])
            if artifact_views[0]["available"] and artifact_views[1]["available"]:
                self.assertNotEqual(artifact_views[0]["body"], artifact_views[1]["body"])


class TestVisibleUnitsCarryReplacement(unittest.TestCase):
    def test_replacement_ref_copied_from_signature(self):
        case = make_exported_case()
        graph = build_explanation_graph(case, condition="axis", detail_level="full")
        units = visible_replay_units(graph)
        self.assertTrue(units)
        self.assertIn("replacement_ref", units[0])
        self.assertIn("intervention_signature", units[0])


if __name__ == "__main__":
    unittest.main()
