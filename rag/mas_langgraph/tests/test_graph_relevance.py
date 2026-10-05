"""Deterministic relevance filtering for explanation graphs."""

from __future__ import annotations

import unittest

from mas_langgraph.tests.test_explanation_graph import make_exported_case
from mas_langgraph.visualization import build_explanation_graph


def _non_control_edge_ids(graph):
    return {e.edge_id for e in graph.edges if e.kind != "control_flow"}


class TestGraphRelevance(unittest.TestCase):
    def setUp(self):
        self.case = make_exported_case()

    def test_filter_monotonicity_per_condition(self):
        for condition in ("trace", "trace_provenance", "axis"):
            ess = build_explanation_graph(self.case, condition, "essential")
            rel = build_explanation_graph(self.case, condition, "relevant")
            full = build_explanation_graph(self.case, condition, "full")
            self.assertTrue(
                set(ess.node_ids()) <= set(rel.node_ids()) <= set(full.node_ids()),
                msg=condition,
            )
            self.assertTrue(
                _non_control_edge_ids(ess)
                <= _non_control_edge_ids(rel)
                <= _non_control_edge_ids(full),
                msg=condition,
            )

    def test_relevant_drops_isolated_artifact(self):
        rel = build_explanation_graph(
            self.case, condition="trace_provenance", detail_level="relevant"
        )
        full = build_explanation_graph(
            self.case, condition="trace_provenance", detail_level="full"
        )
        self.assertIn("artifact::lo_unrelated:v1", full.node_ids())
        self.assertNotIn("artifact::lo_unrelated:v1", rel.node_ids())

    def test_relevant_keeps_target_chain(self):
        rel = build_explanation_graph(
            self.case, condition="trace_provenance", detail_level="relevant"
        )
        ids = set(rel.node_ids())
        self.assertIn("artifact::q_00000010:v2", ids)
        self.assertIn("artifact::bp_1:v1", ids)
        self.assertIn("artifact::lo_1:v1", ids)
        self.assertIn("artifact::A1:v1", ids)
        self.assertIn("artifact::E1:v1", ids)

    def test_essential_hides_non_key_occurrences_honestly(self):
        ess = build_explanation_graph(
            self.case, condition="trace", detail_level="essential"
        )
        occ_subtypes = {n.subtype for n in ess.nodes if n.kind == "occurrence"}
        self.assertNotIn("build_outline", occ_subtypes)
        self.assertNotIn("curate_evidence", occ_subtypes)
        self.assertIn("generate_learning_objectives", occ_subtypes)
        self.assertIn("apply_acceptance_gate", occ_subtypes)
        skipped = [
            e
            for e in ess.edges
            if e.kind == "control_flow"
            and int((e.details or {}).get("skipped_occurrence_count") or 0) > 0
        ]
        self.assertTrue(skipped)
        self.assertTrue(any("steps" in e.label for e in skipped))

    def test_essential_axis_selects_at_most_three_valid_replays(self):
        ess = build_explanation_graph(self.case, "axis", "essential")
        rel = build_explanation_graph(self.case, "axis", "relevant")
        full = build_explanation_graph(self.case, "axis", "full")
        ess_ids = {
            n.node_id.split("::")[1]
            for n in ess.nodes
            if n.node_id.startswith("replay::")
        }
        self.assertLessEqual(len(ess_ids), 3)
        selected = ess.metadata.get("selected_replay_indices") or []
        self.assertLessEqual(len(selected), 3)
        self.assertGreaterEqual(full.metadata.get("replay_count"), 8)
        self.assertGreater(ess.hidden_counts.get("hidden_replay_count"), 0)
        # Invalid statuses must not outrank valid evidence.
        ess_statuses = {
            n.details.get("evidence_status")
            for n in ess.nodes
            if n.kind == "comparison"
        }
        self.assertNotIn("noop_error", ess_statuses)
        self.assertNotIn("intervention_error", ess_statuses)
        self.assertNotIn("outcome_unavailable", ess_statuses)
        self.assertNotIn("replay_unstable", ess_statuses)
        ranking = ess.metadata.get("replay_ranking") or []
        self.assertTrue(ranking)
        first_selected = next(r for r in ranking if r.get("selected"))
        first_tuple = first_selected["rank_tuple"]
        self.assertEqual(first_tuple[0], 0)
        self.assertEqual(first_tuple[1], 0)  # decision_effect
        self.assertTrue(set(ess.node_ids()) <= set(rel.node_ids()))

    def test_relevant_hides_invalid_replays_when_valid_exist(self):
        rel = build_explanation_graph(self.case, "axis", "relevant")
        statuses = {
            n.details.get("evidence_status")
            for n in rel.nodes
            if n.kind == "comparison"
        }
        self.assertIn("criterion_effect", statuses)
        self.assertNotIn("noop_error", statuses)
        self.assertNotIn("intervention_error", statuses)

    def test_full_keeps_invalid_replays(self):
        full = build_explanation_graph(self.case, "axis", "full")
        statuses = {
            n.details.get("evidence_status")
            for n in full.nodes
            if n.kind == "comparison"
        }
        self.assertIn("noop_error", statuses)
        self.assertIn("replay_unstable", statuses)

    def test_control_flow_skip_does_not_invent_nodes(self):
        full = build_explanation_graph(self.case, "trace", "full")
        ess = build_explanation_graph(self.case, "trace", "essential")
        self.assertTrue(set(ess.node_ids()) <= set(full.node_ids()))
        self.assertFalse(any("merged" in n.node_id for n in ess.nodes))


if __name__ == "__main__":
    unittest.main()
