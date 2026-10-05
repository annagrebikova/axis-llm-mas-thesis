"""Per-node LLM attribution follows LangGraph post-node ``updates`` events."""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from mas_langgraph.artifacts.channel_state import empty_artifact_channels
from mas_langgraph.core.runtime import WorkflowRuntime
from mas_langgraph.experiment.timing import (
    assemble_replay_profile,
    replay_profile_scope,
)
from mas_langgraph.replay.branches import BranchMetadata
from mas_langgraph.replay.executor import LangGraphReplayExecutor
from mas_langgraph.replay.llm_determinism import (
    ReplayAwareWorkflowLLM,
    workflow_llm_execution_context,
)


@dataclass
class AIMessage:
    content: str


class IncrementingLLM:
    def __init__(self, name: str = "fake") -> None:
        self.model_name = name
        self.temperature = 0
        self.calls = 0

    def invoke(self, prompt, config=None, **kwargs):
        self.calls += 1
        return AIMessage(content=f"{prompt}:{self.calls}")


class _FakeSnap:
    def __init__(
        self,
        values: Dict[str, Any],
        checkpoint_id: str = "ckpt",
        parent_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.values = values
        self.config = {"configurable": {"checkpoint_id": checkpoint_id, "thread_id": "run_attr"}}
        self.parent_config = parent_config


class _PostNodeGraph:
    """Emit updates AFTER optional LLM work, matching LangGraph post-node semantics."""

    def __init__(
        self,
        steps: List[Tuple[str, Optional[Callable[[], None]]]],
        *,
        history_size: int = 3,
        history_sleep_s: float = 0.0,
        superstep: bool = False,
    ) -> None:
        self.steps = steps
        self.history_size = history_size
        self.history_sleep_s = history_sleep_s
        self.superstep = superstep
        self.invoke_calls = 0
        self.stream_calls = 0
        self.get_state_history_calls = 0
        self._values = {
            "branch_id": "cf_attr",
            "cooperative_state": None,
            **empty_artifact_channels(),
        }

    def stream(self, input, config=None, context=None, stream_mode=None, **kwargs):
        self.stream_calls += 1
        yield ("values", dict(self._values))
        if self.superstep:
            for _name, cb in self.steps:
                if cb is not None:
                    cb()
            payload = {name: {} for name, _cb in self.steps}
            yield ("updates", payload)
            yield ("values", dict(self._values))
            return
        for name, cb in self.steps:
            if cb is not None:
                cb()
            yield ("updates", {name: {}})
            yield ("values", dict(self._values))

    def invoke(self, input, config=None, context=None, **kwargs):
        self.invoke_calls += 1
        return dict(self._values)

    def get_state(self, config=None):
        return _FakeSnap(dict(self._values))

    def get_state_history(self, config=None):
        self.get_state_history_calls += 1
        if self.history_sleep_s:
            time.sleep(self.history_sleep_s)
        snaps = [
            _FakeSnap({"branch_id": "other"}, checkpoint_id=f"h{i}")
            for i in range(max(0, self.history_size - 1))
        ]
        snaps.append(_FakeSnap(dict(self._values), checkpoint_id="h_branch"))
        return list(reversed(snaps))

    def update_state(self, config, values):
        return {"configurable": {"thread_id": "run_attr", "checkpoint_id": "fork"}}


def _always_continue(*_a, **_k):
    from mas_langgraph.cases.mcq.replay_observable import ReplayObservableStatus

    return ReplayObservableStatus.CONTINUE


def _explained():
    from mas_langgraph.artifacts.models import ArtifactRef
    from mas_langgraph.axis.models import ExplainedCase

    return ExplainedCase(
        run_id="run_attr",
        artifact_ref=ArtifactRef("q_1", 1),
        branch_id="factual",
        decision_ref=ArtifactRef("decision:q_1", 1),
        outcome="accept",
    )


def _executor_with_graph(graph: _PostNodeGraph) -> Tuple[LangGraphReplayExecutor, str, str]:
    run_id = "run_attr"
    ex = LangGraphReplayExecutor(compiled_graph=graph)
    ex._runtimes[run_id] = WorkflowRuntime()
    ex.branches.register(BranchMetadata.factual(run_id))
    meta = ex.branches.fork(
        parent=ex.branches.get(run_id, "factual"),
        branch_type="counterfactual",
        origin_checkpoint_alias="c_8",
        origin_checkpoint_id="ckpt8",
    )
    ex._branch_fork_configs[(run_id, meta.branch_id)] = {
        "configurable": {"thread_id": run_id, "checkpoint_id": "fork"}
    }
    graph._values["branch_id"] = meta.branch_id
    return ex, run_id, meta.branch_id


def _by_node(timings: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["node"]): row for row in timings}


class TestPostNodeAttribution(unittest.TestCase):
    def test_provider_calls_belong_to_completed_update_node(self):
        base = IncrementingLLM()
        wrapped = ReplayAwareWorkflowLLM(base)

        def call_b():
            wrapped.invoke("prompt-B")

        def call_c():
            wrapped.invoke("prompt-C0")
            wrapped.invoke("prompt-C1")
            wrapped.invoke("prompt-C2")

        graph = _PostNodeGraph(
            [
                ("node_A", None),
                ("node_B", call_b),
                ("node_C", call_c),
            ]
        )
        ex, run_id, branch_id = _executor_with_graph(graph)
        result = ex.replay_downstream(
            run_id,
            branch_id,
            explained_case=_explained(),
            stop_when_observable=True,
            stop_predicate=_always_continue,
        )
        by_node = _by_node(result.node_timings)
        self.assertEqual(by_node["node_A"]["provider_calls"], 0)
        self.assertEqual(by_node["node_B"]["provider_calls"], 1)
        self.assertEqual(by_node["node_C"]["provider_calls"], 3)
        self.assertEqual(by_node["node_A"]["logical_llm_requests"], 0)
        self.assertEqual(by_node["node_B"]["logical_llm_requests"], 1)
        self.assertEqual(by_node["node_C"]["logical_llm_requests"], 3)
        self.assertNotEqual(by_node["node_A"]["provider_calls"], 1)
        self.assertEqual(base.calls, 4)

    def test_cross_cove_node_receives_cross_cove_calls(self):
        base = IncrementingLLM()
        wrapped = ReplayAwareWorkflowLLM(base)

        graph = _PostNodeGraph(
            [
                ("generate_candidate_mcqs_from_blueprints", lambda: wrapped.invoke("gen")),
                (
                    "verify_candidate_mcqs_against_evidence",
                    lambda: wrapped.invoke("evid"),
                ),
                (
                    "cross_cove_verify_candidate_mcqs",
                    lambda: [wrapped.invoke(f"cove{i}") for i in range(3)],
                ),
                ("revise_candidate_mcqs", None),
            ]
        )
        ex, run_id, branch_id = _executor_with_graph(graph)
        result = ex.replay_downstream(
            run_id,
            branch_id,
            explained_case=_explained(),
            stop_when_observable=True,
            stop_predicate=_always_continue,
        )
        by_node = _by_node(result.node_timings)
        self.assertEqual(by_node["generate_candidate_mcqs_from_blueprints"]["provider_calls"], 1)
        self.assertEqual(by_node["verify_candidate_mcqs_against_evidence"]["provider_calls"], 1)
        self.assertEqual(by_node["cross_cove_verify_candidate_mcqs"]["provider_calls"], 3)
        self.assertEqual(by_node["revise_candidate_mcqs"]["provider_calls"], 0)
        self.assertEqual(
            by_node["verify_candidate_mcqs_against_evidence"]["provider_calls"],
            1,
        )
        self.assertNotEqual(
            by_node["verify_candidate_mcqs_against_evidence"]["provider_calls"],
            3,
        )

    def test_ledger_hit_stays_on_the_node_that_requested_it(self):
        base = IncrementingLLM()
        wrapped = ReplayAwareWorkflowLLM(base)
        graph = _PostNodeGraph(
            [
                ("node_A", None),
                ("node_B", lambda: wrapped.invoke("shared")),
                ("node_C", lambda: wrapped.invoke("novel")),
            ]
        )
        ex, run_id, branch_id = _executor_with_graph(graph)
        ledger = ex.get_or_create_llm_ledger(run_id)
        with workflow_llm_execution_context(
            run_id=run_id, branch_id="factual", branch_type="factual", ledger=ledger
        ):
            wrapped.invoke("shared")
        self.assertEqual(base.calls, 1)
        result = ex.replay_downstream(
            run_id,
            branch_id,
            explained_case=_explained(),
            stop_when_observable=True,
            stop_predicate=_always_continue,
        )
        by_node = _by_node(result.node_timings)
        self.assertEqual(by_node["node_B"]["factual_ledger_hits"], 1)
        self.assertEqual(by_node["node_B"]["provider_calls"], 0)
        self.assertEqual(by_node["node_C"]["provider_calls"], 1)
        self.assertEqual(by_node["node_C"]["factual_ledger_hits"], 0)
        self.assertEqual(base.calls, 2)

    def test_sequential_reconciliation(self):
        base = IncrementingLLM()
        wrapped = ReplayAwareWorkflowLLM(base)
        graph = _PostNodeGraph(
            [
                ("node_A", None),
                ("node_B", lambda: wrapped.invoke("B")),
                ("node_C", lambda: [wrapped.invoke("C0"), wrapped.invoke("C1"), wrapped.invoke("C2")]),
            ]
        )
        ex, run_id, branch_id = _executor_with_graph(graph)
        result = ex.replay_downstream(
            run_id,
            branch_id,
            explained_case=_explained(),
            stop_when_observable=True,
            stop_predicate=_always_continue,
        )
        diag = result.llm_replay_diagnostics
        timings = result.node_timings
        unattr = result.node_profile_unattributed or {}

        def _sum(key: str) -> int:
            return sum(int(row.get(key) or 0) for row in timings) + int(unattr.get(key) or 0)

        self.assertEqual(_sum("logical_llm_requests"), int(diag.get("request_count") or 0))
        self.assertEqual(_sum("provider_calls"), int(diag.get("live_provider_calls") or 0))
        self.assertEqual(_sum("factual_ledger_hits"), int(diag.get("factual_response_hits") or 0))
        self.assertEqual(_sum("cf_memo_hits"), int(diag.get("memo_hits") or 0))

    def test_superstep_is_marked_ambiguous(self):
        base = IncrementingLLM()
        wrapped = ReplayAwareWorkflowLLM(base)
        graph = _PostNodeGraph(
            [
                ("node_P", lambda: wrapped.invoke("P")),
                ("node_Q", lambda: wrapped.invoke("Q")),
            ],
            superstep=True,
        )
        ex, run_id, branch_id = _executor_with_graph(graph)
        result = ex.replay_downstream(
            run_id,
            branch_id,
            explained_case=_explained(),
            stop_when_observable=True,
            stop_predicate=_always_continue,
        )
        self.assertEqual(len(result.node_timings), 1)
        row = result.node_timings[0]
        self.assertEqual(row["attribution_scope"], "superstep")
        self.assertEqual(row["provider_calls"], 2)
        self.assertIn("node_P", row["nodes"])
        self.assertIn("node_Q", row["nodes"])


class TestReplayProfileScopes(unittest.TestCase):
    def test_nested_spans_are_excluded_from_top_level_sum(self):
        profile = assemble_replay_profile(
            spans={
                "graph_execution": 10.0,
                "provider_wait": 7.0,
                "ledger_lookup": 1.0,
                "observability_check": 0.5,
                "branch_history_fetch": 4.0,
            },
            total_s=15.0,
        )
        top = profile["top_level"]
        nested = profile["nested"]
        self.assertGreaterEqual(float(top["graph_execution_s"]), 0.0)
        self.assertGreaterEqual(float(nested["provider_wait_s"]), 0.0)
        exclusive = sum(float(v) for k, v in top.items() if k != "other_s")
        self.assertAlmostEqual(exclusive + float(top["other_s"]), 15.0, places=6)
        self.assertLess(float(top["other_s"]), 1.1)
        self.assertEqual(float(nested["provider_wait_s"]), 7.0)
        self.assertNotEqual(
            exclusive + float(nested["provider_wait_s"]),
            15.0,
        )

    def test_provider_free_post_graph_work_is_named(self):
        graph = _PostNodeGraph(
            [("node_A", None)],
            history_size=40,
            history_sleep_s=0.03,
        )
        ex, run_id, branch_id = _executor_with_graph(graph)
        t0 = time.perf_counter()
        with replay_profile_scope() as spans:
            result = ex.replay_downstream(
                run_id,
                branch_id,
                explained_case=_explained(),
                stop_when_observable=True,
                stop_predicate=_always_continue,
            )
        total = time.perf_counter() - t0
        profile = assemble_replay_profile(
            spans=spans,
            total_s=total,
            extra=result.replay_span_profile,
        )
        diag = result.llm_replay_diagnostics
        self.assertEqual(int(diag.get("live_provider_calls") or 0), 0)
        self.assertEqual(graph.get_state_history_calls, 0)
        self.assertEqual(profile.get("history_collection_mode"), "parent_lineage")
        self.assertLessEqual(int(profile["history_snapshots_scanned"]), 2)
        self.assertGreaterEqual(int(profile["branch_snapshots_retained"]), 1)
        other = float(profile["top_level"]["other_s"])
        exclusive = sum(
            float(v) for k, v in profile["top_level"].items() if k != "other_s"
        )
        self.assertAlmostEqual(exclusive + other, total, places=2)
        self.assertLess(other, 0.05)


if __name__ == "__main__":
    unittest.main()
