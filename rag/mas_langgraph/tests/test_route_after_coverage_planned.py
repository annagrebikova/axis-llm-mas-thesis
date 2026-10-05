"""LangGraph routing: consume all planned blueprints before coverage repair."""

from __future__ import annotations

import unittest
from typing import List
from unittest.mock import patch

from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.domains.quiz.entities import MCQBlueprint
from mas_langgraph.input.constants import QUIZ_QUESTIONS_PER_ROUND
from mas_langgraph.orchestrator.langgraph_workflow import (
    node_generate_candidate_mcqs_from_blueprints,
    route_after_coverage,
)
from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context


def _planned_blueprint(i: int, *, status: str = "planned") -> MCQBlueprint:
    return MCQBlueprint(
        blueprint_id=f"bp_{i:08d}",
        section_id=f"sec_{i}",
        lo_id=f"lo_{i}",
        target_atom_ids=[f"atom_{i}"],
        evidence_chunk_ids=[f"chunk_{i}"],
        question_type="basic_definition",
        difficulty="intermediate",
        bloom_level="Remembering",
        generation_goal=f"Cover section {i}",
        status=status,
    )


def _state_with_planned(n: int = 10) -> CooperativeState:
    state = CooperativeState()
    for i in range(1, n + 1):
        state.add_section(
            DocSection(
                section_id=f"sec_{i}",
                title=f"Section {i}",
                slide_range=(i, i),
                importance_weight=1.0,
                content_summary=f"Summary {i}",
            )
        )
        state.add_learning_objective(
            LearningObjective(
                lo_id=f"lo_{i}",
                section_id=f"sec_{i}",
                objective_text=f"Explain section {i}.",
                difficulty_level="intermediate",
            )
        )
        state.add_evidence_chunk(
            EvidenceChunk(
                chunk_id=f"chunk_{i}",
                content=f"Evidence for section {i}.",
                metadata={"section_id": f"sec_{i}", "page_number": i},
                quality_score=0.9,
            )
        )
        state.add_knowledge_atom(
            KnowledgeAtom(
                atom_id=f"atom_{i}",
                content=f"Atom for section {i}.",
                atom_type="definition",
                citations=[{"chunk_id": f"chunk_{i}", "page": i}],
                section_id=f"sec_{i}",
                lo_id=f"lo_{i}",
            )
        )
        state.link_chunk_to_atom(f"chunk_{i}", f"atom_{i}")
    state.store_mcq_blueprints([_planned_blueprint(i) for i in range(1, n + 1)])
    state.update_section_evidence_status()
    return state


def _graph_state(
    cs: CooperativeState,
    *,
    gaps: List[dict] | None = None,
    evidence_iters: int = 0,
    blueprint_iters: int = 0,
    max_evidence: int = 2,
    max_blueprint: int = 2,
) -> dict:
    return {
        "cooperative_state": cs,
        "coverage_gaps": gaps if gaps is not None else [{"section_id": "sec_9"}],
        "evidence_repair_iterations": evidence_iters,
        "max_evidence_repair_iterations": max_evidence,
        "blueprint_repair_iterations": blueprint_iters,
        "max_blueprint_repair_iterations": max_blueprint,
        "step_logs": [],
    }


class TestRouteAfterCoveragePlannedPriority(unittest.TestCase):
    def test_planned_takes_precedence_over_blueprint_repair(self):
        cs = _state_with_planned(3)
        self.assertGreater(len(cs.planned_blueprints()), 0)
        route = route_after_coverage(
            _graph_state(cs, gaps=[{"section_id": "sec_1"}], blueprint_iters=0),
            workflow_runtime_context(),
        )
        self.assertEqual(route, "generate_candidate_mcqs_from_blueprints")

    def test_planned_takes_precedence_over_evidence_repair(self):
        cs = _state_with_planned(3)
        # Remove evidence for one section so missing_evidence is non-empty.
        cs.evidence_pool.clear()
        cs.knowledge_atoms.clear()
        cs.update_section_evidence_status()
        self.assertTrue(cs.get_sections_without_evidence())
        self.assertTrue(cs.planned_blueprints())
        route = route_after_coverage(
            _graph_state(cs, evidence_iters=0, gaps=[{"section_id": "sec_1"}]),
            workflow_runtime_context(),
        )
        self.assertEqual(route, "generate_candidate_mcqs_from_blueprints")

    def test_evidence_repair_when_no_planned(self):
        cs = _state_with_planned(2)
        for bp in cs.mcq_blueprints.values():
            bp.status = "generated"
        cs.evidence_pool.clear()
        cs.knowledge_atoms.clear()
        cs.update_section_evidence_status()
        self.assertEqual(len(cs.planned_blueprints()), 0)
        self.assertTrue(cs.get_sections_without_evidence())
        route = route_after_coverage(
            _graph_state(cs, evidence_iters=0),
            workflow_runtime_context(),
        )
        self.assertEqual(route, "repair_missing_evidence")

    def test_blueprint_repair_when_no_planned_and_no_evidence_need(self):
        cs = _state_with_planned(2)
        for bp in cs.mcq_blueprints.values():
            bp.status = "generated"
        # Sufficient evidence already present for all sections in helper state.
        cs.update_section_evidence_status()
        self.assertEqual(len(cs.planned_blueprints()), 0)
        self.assertFalse(cs.get_sections_without_evidence())
        route = route_after_coverage(
            _graph_state(
                cs,
                gaps=[{"section_id": "sec_1", "reason": "no accepted MCQ"}],
                evidence_iters=2,
                max_evidence=2,
                blueprint_iters=0,
            ),
            workflow_runtime_context(),
        )
        self.assertEqual(route, "generate_repair_blueprints")

    def test_package_when_no_planned_and_no_repair(self):
        cs = _state_with_planned(2)
        for bp in cs.mcq_blueprints.values():
            bp.status = "generated"
        cs.update_section_evidence_status()
        route = route_after_coverage(
            _graph_state(
                cs,
                gaps=[],
                evidence_iters=2,
                blueprint_iters=2,
            ),
            workflow_runtime_context(),
        )
        self.assertEqual(route, "package_final_quiz")

    def test_package_unreachable_while_planned_remain(self):
        cs = _state_with_planned(1)
        # Even with exhausted repair budgets and empty gaps, planned wins.
        route = route_after_coverage(
            _graph_state(
                cs,
                gaps=[],
                evidence_iters=99,
                blueprint_iters=99,
                max_evidence=2,
                max_blueprint=2,
            ),
            workflow_runtime_context(),
        )
        self.assertEqual(route, "generate_candidate_mcqs_from_blueprints")
        self.assertNotEqual(route, "package_final_quiz")

    def test_batching_consumes_all_ten_planned_before_package(self):
        self.assertEqual(QUIZ_QUESTIONS_PER_ROUND, 2)
        cs = _state_with_planned(10)
        blueprint_iters = 0
        planned_counts = [len(cs.planned_blueprints())]
        routes: List[str] = []

        # Simulate five generation batches marking blueprints generated.
        while cs.planned_blueprints():
            route = route_after_coverage(
                _graph_state(
                    cs,
                    gaps=[{"section_id": "sec_x"}],
                    blueprint_iters=blueprint_iters,
                ),
                workflow_runtime_context(),
            )
            routes.append(route)
            self.assertEqual(route, "generate_candidate_mcqs_from_blueprints")
            # Blueprint-repair budget must not be consumed while draining planned.
            self.assertEqual(blueprint_iters, 0)
            batch = cs.planned_blueprints()[:QUIZ_QUESTIONS_PER_ROUND]
            for bp in batch:
                bp.status = "generated"
            planned_counts.append(len(cs.planned_blueprints()))

        self.assertEqual(planned_counts, [10, 8, 6, 4, 2, 0])
        self.assertEqual(len(routes), 5)
        self.assertTrue(all(r == "generate_candidate_mcqs_from_blueprints" for r in routes))

        # Only now may packaging / repair be considered.
        final = route_after_coverage(
            _graph_state(cs, gaps=[], blueprint_iters=2, evidence_iters=2),
            workflow_runtime_context(),
        )
        self.assertEqual(final, "package_final_quiz")

    def test_repair_blueprints_also_consumed_before_next_repair(self):
        cs = _state_with_planned(0)
        # No original planned; create two repair blueprints as planned.
        cs.store_mcq_blueprints([
            _planned_blueprint(101),
            _planned_blueprint(102),
        ])
        cs.update_section_evidence_status()
        # First coverage decision after repair creation must generate, not re-repair.
        route = route_after_coverage(
            _graph_state(
                cs,
                gaps=[{"section_id": "sec_101"}],
                evidence_iters=2,
                blueprint_iters=1,
                max_blueprint=2,
            ),
            workflow_runtime_context(),
        )
        self.assertEqual(route, "generate_candidate_mcqs_from_blueprints")

        for bp in cs.planned_blueprints():
            bp.status = "generated"
        route2 = route_after_coverage(
            _graph_state(
                cs,
                gaps=[{"section_id": "sec_101"}],
                evidence_iters=2,
                blueprint_iters=1,
                max_blueprint=2,
            ),
            workflow_runtime_context(),
        )
        self.assertEqual(route2, "generate_repair_blueprints")

    def test_failed_blueprint_does_not_infinite_loop(self):
        cs = CooperativeState()
        bp = _planned_blueprint(1)
        cs.store_mcq_blueprints([bp])
        graph_state = {
            "cooperative_state": cs,
            "step_logs": [],
            "coverage_gaps": [],
            "evidence_repair_iterations": 0,
            "max_evidence_repair_iterations": 0,
            "blueprint_repair_iterations": 0,
            "max_blueprint_repair_iterations": 0,
        }
        runtime = workflow_runtime_context()

        with patch(
            "mas_langgraph.orchestrator.langgraph_workflow.generate_candidate_from_blueprint",
            side_effect=RuntimeError("boom"),
        ):
            update = node_generate_candidate_mcqs_from_blueprints(graph_state, runtime)

        updated = update["cooperative_state"]
        self.assertEqual(updated.mcq_blueprints[bp.blueprint_id].status, "failed")
        self.assertEqual(len(updated.planned_blueprints()), 0)
        route = route_after_coverage(
            {
                **graph_state,
                "cooperative_state": updated,
            },
            workflow_runtime_context(),
        )
        self.assertEqual(route, "package_final_quiz")

    def test_graph_wires_planned_generation_from_coverage(self):
        graph = build_mcq_workflow_graph()
        nodes = set(graph.get_graph().nodes)
        self.assertIn("generate_candidate_mcqs_from_blueprints", nodes)
        self.assertIn("check_coverage", nodes)
        import inspect
        from mas_langgraph.core import graph_builder as gb

        src = inspect.getsource(gb.build_mcq_workflow_graph)
        self.assertIn("route_after_coverage", src)
        # Conditional map from check_coverage includes generation return path.
        self.assertRegex(
            src,
            r'"generate_candidate_mcqs_from_blueprints"\s*:\s*'
            r'\n?\s*"generate_candidate_mcqs_from_blueprints"',
        )


if __name__ == "__main__":
    unittest.main()
