"""Regression: RAG outline/LO writes must land on the LangGraph working CooperativeState."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, List, Tuple

from mas_langgraph.artifacts.channel_state import empty_artifact_channels
from mas_langgraph.cases.mcq import MCQCaseDefinition
from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
from mas_langgraph.core.checkpointer import default_factual_checkpointer
from mas_langgraph.core.entities import DocSection, LearningObjective
from mas_langgraph.core.runtime import WorkflowRuntime
from mas_langgraph.core.workflow_executor import WorkflowExecutor
from mas_langgraph.orchestrator.langgraph_workflow import (
    WorkflowGraphState,
    node_build_outline,
    node_generate_learning_objectives,
    node_initial_retrieve_evidence_for_all_sections,
)
from mas_langgraph.orchestrator.needs import NeedsEvaluator
from mas_langgraph.orchestrator.state_binding import bind_rag_agents
from mas_langgraph.provenance.channel_state import empty_provenance_channels
from mas_langgraph.provenance.instrumentation import instrument_node
from mas_langgraph.artifacts.resolution import enforce_artifact_resolution


class _FakeDoc:
    def __init__(self, text: str, page: int = 1) -> None:
        self.page_content = text
        self.metadata = {"page_number": page, "pdf_id": "141"}


class _FakePipeline:
    def search(self, query: str, k: int = 5) -> List[Tuple[_FakeDoc, float]]:
        return [(_FakeDoc(f"chunk for {query}", page=1), 0.9)]


class FakeRAGAgents:
    """Mirrors RAGAgents: methods mutate ``self.state``, not a wrapper attribute."""

    def __init__(self) -> None:
        self.state = CooperativeState()
        self.rag_pipeline = _FakePipeline()
        self.writes_on_original = 0

    def build_outline_from_document(self) -> str:
        self.writes_on_original += 1 if self.state is getattr(self, "_original", None) else 0
        sections = [
            DocSection(f"s{i}", f"Section {i} Clustering", (i + 1, i + 2), 0.5)
            for i in range(10)
        ]
        self.state.set_doc_outline(sections)
        return f"Built {len(self.state.doc_outline)} sections"

    def generate_learning_objectives_for_outline(self) -> str:
        n = 0
        for i, sec in enumerate(self.state.doc_outline):
            for j in range(3):
                self.state.add_learning_objective(
                    LearningObjective(
                        f"lo_{i}_{j}",
                        sec.section_id,
                        f"Explain {sec.title} aspect {j}",
                        "intermediate",
                    )
                )
                n += 1
        return f"Built {n} LOs"

    def log_step(self, *args: Any, **kwargs: Any) -> None:
        return None

    def detect_duplicate_chunks(self) -> str:
        return ""


def _wrap(node_id, fn):
    return enforce_artifact_resolution(node_id, instrument_node(node_id, fn))


def _initial_state(cs: CooperativeState, run_id: str) -> dict:
    return {
        "cooperative_state": cs,
        "run_id": run_id,
        "branch_id": "factual",
        "branch_type": "factual",
        "parent_branch_id": "",
        "origin_checkpoint_alias": "",
        "intervention_id": None,
        "searched_queries": set(),
        "last_agent": None,
        "step_logs": [],
        "error": None,
        "retrieve_attempts": 0,
        "quiz_verify_loops": 0,
        "repair_iterations": 0,
        "coverage_gaps": [],
        "revision_iterations": 0,
        "max_revision_iterations": 2,
        "pending_evidence_repair_atomize": False,
        "return_to_coverage_check": False,
        "evidence_repair_iterations": 0,
        "max_evidence_repair_iterations": 2,
        "blueprint_repair_iterations": 0,
        "max_blueprint_repair_iterations": 2,
        **empty_artifact_channels(),
        **empty_provenance_channels(),
    }


class TestStateBoundRAGAgents(unittest.TestCase):
    def test_bound_methods_write_working_state_not_service_state(self):
        rag = FakeRAGAgents()
        original = rag.state
        working = CooperativeState()
        bound = bind_rag_agents(rag, working)
        bound.build_outline_from_document()
        self.assertEqual(len(working.doc_outline), 10)
        self.assertEqual(len(original.doc_outline), 0)
        self.assertIs(rag.state, original)


class TestOutlineLoRetrievalPropagation(unittest.TestCase):
    """Real node wrappers + LangGraph transitions (no OpenAI / Chroma)."""

    def test_retrieval_sees_outline_and_los_from_prior_nodes(self):
        from langgraph.graph import END, START, StateGraph

        rag = FakeRAGAgents()
        cs = CooperativeState()
        runner = type(
            "R",
            (),
            {
                "coverage_plan": None,
                "pdf_id": "141",
                "invoke": lambda self, *a, **k: {"role": "x", "result": "stub"},
                "retrieval_exhausted": lambda self, **k: True,
            },
        )()
        rt = WorkflowRuntime(
            agent_runner=runner,
            rag_agents=rag,
            needs_evaluator=NeedsEvaluator(cs, None, agent_runner=runner),
            case_definition=MCQCaseDefinition(),
        )
        graph = StateGraph(WorkflowGraphState, context_schema=WorkflowRuntime)
        graph.add_node("build_outline", _wrap("build_outline", node_build_outline))
        graph.add_node(
            "generate_learning_objectives",
            _wrap("generate_learning_objectives", node_generate_learning_objectives),
        )
        graph.add_node(
            "initial_retrieve_evidence_for_all_sections",
            _wrap(
                "initial_retrieve_evidence_for_all_sections",
                node_initial_retrieve_evidence_for_all_sections,
            ),
        )
        graph.add_edge(START, "build_outline")
        graph.add_edge("build_outline", "generate_learning_objectives")
        graph.add_edge(
            "generate_learning_objectives",
            "initial_retrieve_evidence_for_all_sections",
        )
        graph.add_edge("initial_retrieve_evidence_for_all_sections", END)

        cp = default_factual_checkpointer()
        compiled = graph.compile(checkpointer=cp)
        ex = WorkflowExecutor(runtime=rt, checkpointer=cp, compiled_graph=compiled)
        run_id = "run_outline_lo_prop"
        result = ex.run(_initial_state(cs, run_id), runtime=rt, run_id=run_id)
        self.assertIsNone(result.error)

        history = list(ex.get_state_history(run_id))
        by_next = {}
        for snap in history:
            nxt = tuple(getattr(snap, "next", None) or ())
            vals = getattr(snap, "values", None) or {}
            coop = vals.get("cooperative_state")
            by_next[nxt] = coop

        after_outline = None
        after_los = None
        after_retrieve = result.final_state["cooperative_state"]
        for snap in history:
            nxt = list(getattr(snap, "next", None) or [])
            vals = getattr(snap, "values", None) or {}
            coop = vals.get("cooperative_state")
            if coop is None:
                continue
            if nxt == ["generate_learning_objectives"]:
                after_outline = coop
            if nxt == ["initial_retrieve_evidence_for_all_sections"]:
                after_los = coop

        self.assertIsNotNone(after_outline)
        self.assertEqual(len(after_outline.doc_outline), 10)
        self.assertEqual(len(after_outline.learning_objectives), 0)

        self.assertIsNotNone(after_los)
        self.assertEqual(len(after_los.doc_outline), 10)
        self.assertEqual(len(after_los.learning_objectives), 30)

        self.assertEqual(len(after_retrieve.doc_outline), 10)
        self.assertEqual(len(after_retrieve.learning_objectives), 30)
        self.assertGreater(len(after_retrieve.evidence_pool), 0)
        # Original RAG blackboard must not be the only place data landed.
        self.assertEqual(len(rag.state.doc_outline), 0)


if __name__ == "__main__":
    unittest.main()
