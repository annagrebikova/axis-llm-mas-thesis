"""Participant-safe explanation presentation: projection, blinding, leakage."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.experiment.blinding import blinding_mapping
from mas_langgraph.experiment.models import EVIDENCE_ACCESS
from mas_langgraph.tests.test_explanation_graph import _ref, make_exported_case
from mas_langgraph.visualization import build_explanation_graph
from mas_langgraph.visualization.participant import (
    BLINDED_LABELS,
    PARTICIPANT_DETAIL_LEVEL,
    PARTICIPANT_LANE_LABELS,
    PARTICIPANT_UNAVAILABLE_MESSAGE,
    ParticipantPresentationError,
    assert_projection_semantics,
    build_participant_condition_view,
    forbidden_structural_hits,
    graph_has_provenance,
    graph_has_replay,
    load_participant_experiment,
    participant_view_snapshot,
    preflight_participant_experiment,
    project_participant_graph,
    render_participant_dot,
    resolve_participant_experiment_path,
    structural_display_strings,
    validate_blinding,
)


TRACE_TEXT = (
    "The recorded workflow steps produced this question from the lesson outline."
)
TP_TEXT = "The question used the recorded teaching statements about grouping."
AXIS_TEXT = (
    "A controlled test changed the information and the wording shifted."
)

FIXED_BLINDING = {
    "A": "axis",
    "B": "trace",
    "C": "trace_provenance",
}


def make_participant_case(
    *,
    blinding=None,
    candidate_version: int = 2,
) -> dict:
    case = copy.deepcopy(make_exported_case())
    case["question"]["options"] = [
        {"option_id": "A", "text": "Nearest centroid", "is_correct": True},
        {"option_id": "B", "text": "Random cluster", "is_correct": False},
    ]
    case["question"]["correct_option"] = "A"
    if candidate_version == 1:
        case["candidate_ref"] = _ref("q_00000010", 1)
        case["assessment_ref"] = _ref("assessment:q_00000010", 1)
        case["decision_ref"] = _ref("decision:q_00000010", 1)
        case["question"]["stem"] = "Old v1 stem that must not replace v2."
        case["question"]["options"] = [
            {"option_id": "A", "text": "v1 option", "is_correct": True},
        ]
        case["visualization_data"]["target"]["candidate_ref"] = _ref("q_00000010", 1)
        case["visualization_data"]["target"]["assessment_ref"] = _ref(
            "assessment:q_00000010", 1
        )
        case["visualization_data"]["target"]["decision_ref"] = _ref(
            "decision:q_00000010", 1
        )
    case["blinding"] = dict(blinding or FIXED_BLINDING)
    for cond, text in (
        ("trace", TRACE_TEXT),
        ("trace_provenance", TP_TEXT),
        ("axis", AXIS_TEXT),
    ):
        block = case["conditions"].setdefault(cond, {})
        block["explanation"] = text
        block["status"] = "ok"
        block["condition"] = cond
    return case


def make_participant_experiment(cases=None) -> dict:
    rows = list(cases) if cases is not None else [make_participant_case()]
    return {
        "schema_version": 1,
        "run_id": "run_viz",
        "branch_id": "factual",
        "experiment_seed": 42,
        "user_question": "Why was this question selected?",
        "question_count": len(rows),
        "evidence_access": copy.deepcopy(EVIDENCE_ACCESS),
        "models": {
            "workflow": {"model": "gpt-4.1-mini-2025-04-14"},
        },
        "warnings": ["researcher-only warning"],
        "cases": rows,
    }


def _kinds(graph):
    return {n.kind for n in graph.nodes}


def _edge_kinds(graph):
    return {e.kind for e in graph.edges}


class TestBlindingValidation(unittest.TestCase):
    def test_valid_mapping_accepted(self):
        mapping = validate_blinding(FIXED_BLINDING)
        self.assertEqual(mapping["A"], "axis")
        self.assertEqual(set(mapping), set(BLINDED_LABELS))

    def test_missing_blinding_fails_closed(self):
        with self.assertRaises(ParticipantPresentationError):
            validate_blinding(None)
        with self.assertRaises(ParticipantPresentationError):
            build_participant_condition_view(
                make_participant_experiment(),
                make_exported_case(),
                "A",
            )

    def test_duplicate_condition_fails_closed(self):
        with self.assertRaises(ParticipantPresentationError):
            validate_blinding({"A": "axis", "B": "axis", "C": "trace"})

    def test_true_names_are_not_used_as_fallback(self):
        case = make_participant_case()
        case["blinding"] = {"trace": "A", "axis": "B", "trace_provenance": "C"}
        with self.assertRaises(ParticipantPresentationError):
            build_participant_condition_view(
                make_participant_experiment([case]), case, "A"
            )

    def test_existing_algorithm_is_used_when_exported(self):
        mapping = blinding_mapping(
            run_id="run_viz",
            candidate_ref=ArtifactRef(logical_id="q_00000010", version=2),
            experiment_seed=42,
        )
        case = make_participant_case(blinding=mapping)
        experiment = make_participant_experiment([case])
        for label, condition in mapping.items():
            view = build_participant_condition_view(
                experiment, case, label, ordinal=1, total=1
            )
            self.assertEqual(
                view.explanation, case["conditions"][condition]["explanation"]
            )
            self.assertEqual(view.heading, f"Explanation {label}")


class TestProjectionInvariant(unittest.TestCase):
    def test_node_and_edge_sets_preserved(self):
        case = make_participant_case()
        for condition in ("trace", "trace_provenance", "axis"):
            canonical = build_explanation_graph(
                case, condition=condition, detail_level=PARTICIPANT_DETAIL_LEVEL
            )
            participant = project_participant_graph(canonical, case)
            assert_projection_semantics(canonical, participant)
            self.assertEqual(set(canonical.node_ids()), set(participant.node_ids()))
            self.assertEqual(set(canonical.edge_ids()), set(participant.edge_ids()))
            self.assertEqual(canonical.detail_level, PARTICIPANT_DETAIL_LEVEL)
            self.assertEqual(participant.detail_level, PARTICIPANT_DETAIL_LEVEL)


class TestConditionLeakage(unittest.TestCase):
    def setUp(self):
        self.case = make_participant_case()

    def _pair(self, condition: str):
        canonical = build_explanation_graph(
            self.case, condition=condition, detail_level="essential"
        )
        participant = project_participant_graph(canonical, self.case)
        return canonical, participant

    def test_trace_has_no_provenance_or_replay(self):
        for graph in self._pair("trace"):
            self.assertFalse(graph_has_provenance(graph))
            self.assertFalse(graph_has_replay(graph))
            self.assertNotIn("produces", _edge_kinds(graph))
            self.assertNotIn("consumes", _edge_kinds(graph))
            self.assertNotIn("declared_reference", _edge_kinds(graph))
            self.assertNotIn("intervention", _kinds(graph))
            self.assertNotIn("comparison", _kinds(graph))
            self.assertFalse(any(n.subtype == "noop" for n in graph.nodes))
            self.assertFalse(any(n.subtype == "counterfactual" for n in graph.nodes))
            self.assertFalse(any(n.subtype == "EvidenceChunk" for n in graph.nodes))
            self.assertFalse(any(n.subtype == "KnowledgeAtom" for n in graph.nodes))

    def test_trace_provenance_has_recorded_links_but_no_replay(self):
        for graph in self._pair("trace_provenance"):
            self.assertTrue(graph_has_provenance(graph))
            self.assertFalse(graph_has_replay(graph))
            self.assertIn("produces", _edge_kinds(graph))
            self.assertNotIn("intervention", _kinds(graph))
            self.assertFalse(any(n.node_id.startswith("replay::") for n in graph.nodes))
            self.assertFalse(
                any(
                    "role" in (n.details or {})
                    and n.kind == "comparison"
                    for n in graph.nodes
                )
            )

    def test_axis_may_include_replay(self):
        canonical, participant = self._pair("axis")
        self.assertTrue(graph_has_replay(canonical))
        self.assertTrue(graph_has_replay(participant))
        self.assertIn("intervention", _kinds(participant))
        self.assertTrue(any(n.subtype == "noop" for n in participant.nodes))
        self.assertTrue(any(n.kind == "comparison" for n in participant.nodes))


class TestTraceParticipantSafety(unittest.TestCase):
    def test_target_mcq_omits_researcher_fields(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        view = build_participant_condition_view(
            experiment, case, "B", ordinal=1, total=1
        )
        self.assertEqual(FIXED_BLINDING["B"], "trace")
        mcq = view.graph.node_map()["artifact::q_00000010:v2"]
        self.assertEqual(mcq.label, "Question")
        details = dict(mcq.details)
        self.assertEqual(set(details), {"stem", "options", "correct_option_id"})
        self.assertNotIn("blueprint_id", details)
        self.assertNotIn("answer_explanation", details)
        self.assertNotIn("revision_count", details)
        self.assertNotIn("flags", details)
        for opt in details["options"]:
            self.assertEqual(set(opt), {"option_id", "text"})
        blob = json.dumps(details)
        self.assertNotIn("misconception", blob)
        self.assertNotIn("Random assignment", blob)


class TestParticipantWording(unittest.TestCase):
    def test_lane_and_node_labels(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        view = build_participant_condition_view(
            experiment, case, "A", ordinal=1, total=1
        )
        lane_map = {ln.lane_id: ln.label for ln in view.graph.lanes}
        for lane_id, label in PARTICIPANT_LANE_LABELS.items():
            if lane_id in lane_map:
                self.assertEqual(lane_map[lane_id], label)
        labels = {n.label for n in view.graph.nodes}
        self.assertIn("Tested change", labels)
        self.assertIn("Result without a change", labels)
        self.assertIn("Result after the change", labels)
        self.assertIn("Observed effect", labels)
        self.assertIn("Test starting point", labels)
        self.assertNotIn("Intervention", labels)
        self.assertNotIn("Checkpoint", labels)
        self.assertNotIn("Replay result", labels)

    def test_axis_terms_are_translated(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        view = build_participant_condition_view(
            experiment, case, "A", ordinal=1, total=1
        )
        previews = "\n".join(n.preview for n in view.graph.nodes)
        self.assertIn("Remove information", previews)
        self.assertNotIn("REMOVE(", previews)
        self.assertNotIn("NOOP", previews)
        self.assertNotIn("Counterfactual", previews)
        comparison = next(n for n in view.graph.nodes if n.kind == "comparison")
        self.assertIn("Question content:", comparison.preview)
        self.assertIn("Acceptance:", comparison.preview)
        self.assertNotIn("Outcome changed", comparison.preview)

    def test_researcher_details_are_not_carried(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        view = build_participant_condition_view(
            experiment, case, "A", ordinal=1, total=1
        )
        forbidden_keys = {
            "run_id",
            "branch_id",
            "occurrence_id",
            "intervention_signature",
            "evidence_status",
            "verifier_model",
            "comparison_index",
            "replay_id",
            "produced_by_occurrence_id",
            "input_bindings",
        }
        for node in view.graph.nodes:
            self.assertIsNone(node.source_ref)
            self.assertIsNone(node.occurrence_id)
            self.assertTrue(forbidden_keys.isdisjoint(node.details.keys()))
        self.assertEqual(view.graph.metadata, {"presentation": "participant"})


class TestBlindingAndText(unittest.TestCase):
    def test_text_follows_blinded_mapping(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        expected = {"A": AXIS_TEXT, "B": TRACE_TEXT, "C": TP_TEXT}
        for label, text in expected.items():
            view = build_participant_condition_view(
                experiment, case, label, ordinal=1, total=1
            )
            self.assertEqual(view.explanation, text)
            self.assertEqual(view.heading, f"Explanation {label}")
            payload = view.display_payload()
            self.assertEqual(payload["explanation"], text)
            blob = json.dumps(payload)
            self.assertNotIn("axis", blob)
            self.assertNotIn("trace", blob)
            self.assertNotIn("AXIS", blob)
            self.assertNotIn("Trace", blob)

    def test_common_context_identical(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        contexts = []
        for label in BLINDED_LABELS:
            view = build_participant_condition_view(
                experiment, case, label, ordinal=1, total=1
            )
            contexts.append(
                {
                    "stem": view.question["stem"],
                    "options": view.question["options"],
                    "correct_answer": view.question["correct_answer"],
                    "factual_decision": view.question["factual_decision"],
                }
            )
        self.assertEqual(contexts[0], contexts[1])
        self.assertEqual(contexts[1], contexts[2])

    def test_structural_strings_have_no_internal_jargon(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        for label in BLINDED_LABELS:
            view = build_participant_condition_view(
                experiment, case, label, ordinal=1, total=1
            )
            strings = structural_display_strings(view)
            hits = forbidden_structural_hits(strings)
            self.assertEqual(hits, [], msg=hits)
            payload = json.dumps(view.display_payload())
            self.assertNotIn("run_viz", payload)
            self.assertNotIn("q_00000010:v2", payload)
            self.assertNotIn("gpt-", payload)


class TestVersionAndFailClosed(unittest.TestCase):
    def test_v2_identity_preserved_without_showing_ref(self):
        v1 = make_participant_case(candidate_version=1)
        v2 = make_participant_case(candidate_version=2)
        experiment = make_participant_experiment([v1, v2])
        view = build_participant_condition_view(
            experiment, v2, "A", ordinal=2, total=2
        )
        self.assertEqual(view.internal_case_key, "q_00000010:v2")
        self.assertEqual(view.graph.target_ref, "q_00000010:v2")
        self.assertEqual(view.ordinal_label, "Question 2 of 2")
        self.assertIn("artifact::q_00000010:v2", view.graph.node_ids())
        payload = json.dumps(view.display_payload())
        self.assertNotIn("q_00000010:v2", payload)
        self.assertNotIn("q_00000010:v1", payload)
        self.assertIn("Question 2 of 2", payload)
        self.assertEqual(
            view.question["stem"], "What does k-means assign each point to?"
        )
        self.assertNotIn("Old v1 stem", view.question["stem"])

    def test_missing_visualization_data_fails_closed(self):
        case = make_participant_case()
        del case["visualization_data"]
        experiment = make_participant_experiment([case])
        with self.assertRaises(ParticipantPresentationError) as ctx:
            preflight_participant_experiment(experiment)
        self.assertIn("visualization_data", str(ctx.exception))
        with self.assertRaises(ParticipantPresentationError):
            build_participant_condition_view(experiment, case, "A")

    def test_evidence_access_mismatch_fails_closed(self):
        experiment = make_participant_experiment()
        experiment["evidence_access"]["trace"]["provenance"] = True
        with self.assertRaises(ParticipantPresentationError):
            preflight_participant_experiment(experiment)

    def test_detail_level_is_always_essential(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        with patch(
            "mas_langgraph.visualization.participant.build_explanation_graph",
            wraps=build_explanation_graph,
        ) as mocked:
            build_participant_condition_view(
                experiment, case, "A", ordinal=1, total=1
            )
        self.assertTrue(mocked.called)
        for call in mocked.call_args_list:
            kwargs = call.kwargs
            detail = kwargs.get("detail_level")
            if detail is None and len(call.args) >= 3:
                detail = call.args[2]
            self.assertEqual(detail, "essential")
            self.assertEqual(detail, PARTICIPANT_DETAIL_LEVEL)


class TestParticipantDotAndSnapshot(unittest.TestCase):
    def test_opaque_dot_hides_versioned_ids(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        view = build_participant_condition_view(
            experiment, case, "A", ordinal=1, total=1
        )
        dot = render_participant_dot(view.graph)
        self.assertIn("digraph", dot)
        self.assertNotIn("q_00000010", dot)
        self.assertIn("Tested change", dot)
        self.assertIn("How the question was created", dot)

    def test_snapshot_is_compact_and_blinded(self):
        case = make_participant_case()
        experiment = make_participant_experiment([case])
        snapshots = {}
        for label in BLINDED_LABELS:
            view = build_participant_condition_view(
                experiment, case, label, ordinal=1, total=1
            )
            snap = participant_view_snapshot(view)
            snapshots[label] = snap
            self.assertEqual(snap["blinded_label"], label)
            self.assertEqual(snap["heading"], f"Explanation {label}")
            self.assertIn("explanation", snap)
            self.assertIn("nodes", snap)
            self.assertIn("edges", snap)
            self.assertIn("legend", snap)
            blob = json.dumps(
                {
                    "blinded_label": snap["blinded_label"],
                    "heading": snap["heading"],
                    "legend": snap["legend"],
                    "labels": [n["label"] for n in snap["nodes"]],
                    "lane_labels": snap["lane_labels"],
                }
            )
            self.assertNotIn("AXIS", blob)
            self.assertNotIn("Trace", blob)
        self.assertEqual(snapshots["A"]["explanation"], AXIS_TEXT)
        self.assertEqual(snapshots["B"]["explanation"], TRACE_TEXT)
        self.assertEqual(snapshots["C"]["explanation"], TP_TEXT)
        self.assertNotEqual(
            [n["label"] for n in snapshots["A"]["nodes"]],
            [n["label"] for n in snapshots["B"]["nodes"]],
        )


class TestResolvePath(unittest.TestCase):
    def test_missing_env_fails_closed(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "EXPLANATION_EXPERIMENT_JSON",
                "PARTICIPANT_PREVIEW_ALLOW_LOCAL_DISCOVERY",
            }
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ParticipantPresentationError):
                resolve_participant_experiment_path()

    def test_load_configured_file(self):
        experiment = make_participant_experiment()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(experiment, handle)
            path = handle.name
        try:
            loaded = load_participant_experiment(Path(path))
            self.assertEqual(len(loaded["cases"]), 1)
        finally:
            Path(path).unlink(missing_ok=True)


class TestParticipantApp(unittest.TestCase):
    def setUp(self):
        self._old_json = os.environ.get("EXPLANATION_EXPERIMENT_JSON")
        self._old_preview = os.environ.get("PARTICIPANT_PREVIEW_ALLOW_LOCAL_DISCOVERY")
        self._tmp_paths = []

    def tearDown(self):
        if self._old_json is None:
            os.environ.pop("EXPLANATION_EXPERIMENT_JSON", None)
        else:
            os.environ["EXPLANATION_EXPERIMENT_JSON"] = self._old_json
        if self._old_preview is None:
            os.environ.pop("PARTICIPANT_PREVIEW_ALLOW_LOCAL_DISCOVERY", None)
        else:
            os.environ["PARTICIPANT_PREVIEW_ALLOW_LOCAL_DISCOVERY"] = self._old_preview
        for path in self._tmp_paths:
            Path(path).unlink(missing_ok=True)

    def _write(self, payload: dict) -> str:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, handle)
        handle.close()
        self._tmp_paths.append(handle.name)
        return handle.name

    def _run_app(self, path: str):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest is not available")
        app = (
            Path(__file__).resolve().parents[2] / "streamlit_explanation_participant.py"
        )
        os.environ["EXPLANATION_EXPERIMENT_JSON"] = path
        os.environ.pop("PARTICIPANT_PREVIEW_ALLOW_LOCAL_DISCOVERY", None)
        at = AppTest.from_file(str(app), default_timeout=90)
        at.run()
        return at

    def _page_text(self, at) -> str:
        parts = []
        for attr in (
            "title",
            "header",
            "subheader",
            "markdown",
            "text",
            "error",
            "info",
            "warning",
            "caption",
            "success",
        ):
            block = getattr(at, attr, None)
            if not block:
                continue
            for item in block:
                value = getattr(item, "value", None)
                if value is not None:
                    parts.append(str(value))
        return "\n".join(parts)

    def test_configured_app_starts_without_researcher_widgets(self):
        path = self._write(make_participant_experiment())
        at = self._run_app(path)
        self.assertFalse(at.exception, msg=str(at.exception))
        page = self._page_text(at)
        self.assertIn("Explanation Study", page)
        self.assertIn(AXIS_TEXT, page)
        self.assertNotIn("Researcher view", page)
        self.assertEqual(len(at.get("file_uploader")), 0)
        self.assertTrue(at.radio)
        options = list(at.radio[0].options)
        self.assertEqual(options, ["A", "B", "C"])
        self.assertNotIn("Trace", options)
        self.assertNotIn("AXIS", options)
        for box in list(at.selectbox):
            opts = [str(o) for o in getattr(box, "options", [])]
            self.assertNotIn("Essential", opts)
            self.assertNotIn("Relevant", opts)
            self.assertNotIn("Full", opts)
        self.assertNotIn("Essential", page)
        self.assertNotIn("Relevant", page)
        self.assertNotIn("Full", page)
        self.assertNotIn("run_viz", page)
        self.assertNotIn("gpt-4.1", page)
        self.assertNotIn("q_00000010:v2", page)
        self.assertNotIn("Trace + Provenance", page)
        self.assertNotIn("AXIS", page)
        charts = at.get("graphviz_chart")
        self.assertGreaterEqual(len(charts), 1)
        app_src = (
            Path(__file__).resolve().parents[2]
            / "streamlit_explanation_participant.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("file_uploader", app_src)
        self.assertNotIn("detail_level", app_src)
        self.assertNotIn("CONDITION_LABELS", app_src)
        at.radio[0].set_value("B")
        at.run()
        page_b = self._page_text(at)
        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertIn(TRACE_TEXT, page_b)
        self.assertNotIn(AXIS_TEXT, page_b)

    def test_missing_visualization_data_app_fails_closed(self):
        case = make_participant_case()
        del case["visualization_data"]
        path = self._write(make_participant_experiment([case]))
        try:
            at = self._run_app(path)
            self.assertFalse(at.exception, msg=str(at.exception))
            page = self._page_text(at)
            self.assertIn(PARTICIPANT_UNAVAILABLE_MESSAGE, page)
            self.assertNotIn(TRACE_TEXT, page)
            self.assertNotIn(AXIS_TEXT, page)
            self.assertEqual(len(at.get("file_uploader")), 0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_blinding_app_fails_closed(self):
        case = make_participant_case()
        case["blinding"] = {"A": "axis", "B": "axis", "C": "trace"}
        path = self._write(make_participant_experiment([case]))
        try:
            at = self._run_app(path)
            self.assertFalse(at.exception, msg=str(at.exception))
            page = self._page_text(at)
            self.assertIn(PARTICIPANT_UNAVAILABLE_MESSAGE, page)
            self.assertNotIn(AXIS_TEXT, page)
        finally:
            Path(path).unlink(missing_ok=True)


class TestInitIsolation(unittest.TestCase):
    def test_visualization_init_does_not_export_participant(self):
        from mas_langgraph import visualization as viz

        self.assertFalse(hasattr(viz, "build_participant_condition_view"))
        self.assertNotIn("participant", viz.__all__)


if __name__ == "__main__":
    unittest.main()
