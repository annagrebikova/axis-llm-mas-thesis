"""Branch-history collection must be O(current branch), not O(thread history)."""

from __future__ import annotations

import operator
import unittest
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from mas_langgraph.artifacts.channel_state import empty_artifact_channels
from mas_langgraph.core.runtime import WorkflowRuntime
from mas_langgraph.experiment.timing import replay_profile_scope
from mas_langgraph.replay.branches import BranchMetadata
from mas_langgraph.replay.checkpoints import BoundarySelector, extract_checkpoint_id
from mas_langgraph.replay.executor import LangGraphReplayExecutor


@dataclass
class _Snap:
    values: Dict[str, Any]
    checkpoint_id: str
    parent: Optional["_Snap"] = None
    thread_id: str = "run_hist"

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "configurable": {
                "thread_id": self.thread_id,
                "checkpoint_id": self.checkpoint_id,
            }
        }

    @property
    def parent_config(self) -> Optional[Dict[str, Any]]:
        if self.parent is None:
            return None
        return self.parent.config


class _LineageGraph:
    """Sibling snapshots exist in get_state_history but not in parent lineage."""

    def __init__(
        self,
        *,
        branch_id: str,
        branch_len: int = 4,
        sibling_snapshots: int = 80,
    ) -> None:
        self.branch_id = branch_id
        self.get_state_history_calls = 0
        self.get_state_calls = 0
        factual = _Snap({"branch_id": "factual"}, "factual_root")
        chain: List[_Snap] = []
        parent = factual
        for i in range(branch_len):
            snap = _Snap({"branch_id": branch_id, "step": i}, f"branch_{i}", parent=parent)
            chain.append(snap)
            parent = snap
        self.chain = chain
        self.latest = chain[-1]
        self.by_id: Dict[str, _Snap] = {factual.checkpoint_id: factual}
        for snap in chain:
            self.by_id[snap.checkpoint_id] = snap
        self.siblings = [
            _Snap({"branch_id": f"sibling_{i}"}, f"sib_{i}", parent=factual)
            for i in range(sibling_snapshots)
        ]
        self._values = {"branch_id": branch_id, **empty_artifact_channels()}

    def stream(self, input, config=None, context=None, stream_mode=None, **kwargs):
        yield ("values", dict(self._values))
        yield ("updates", {"node_A": {}})
        yield ("values", dict(self._values))

    def invoke(self, input, config=None, context=None, **kwargs):
        return dict(self._values)

    def get_state(self, config=None):
        self.get_state_calls += 1
        configurable = (config or {}).get("configurable") or {}
        cid = configurable.get("checkpoint_id")
        if cid:
            return self.by_id[cid]
        return self.latest

    def get_state_history(self, config=None):
        self.get_state_history_calls += 1
        # Newest-first like LangGraph: siblings + current branch + factual.
        return list(reversed(self.siblings + self.chain + [self.by_id["factual_root"]]))

    def update_state(self, config, values):
        return {"configurable": {"thread_id": "run_hist", "checkpoint_id": "fork"}}


def _explained():
    from mas_langgraph.artifacts.models import ArtifactRef
    from mas_langgraph.axis.models import ExplainedCase

    return ExplainedCase(
        run_id="run_hist",
        artifact_ref=ArtifactRef("q_1", 1),
        branch_id="factual",
        decision_ref=ArtifactRef("decision:q_1", 1),
        outcome="accept",
    )


def _always_continue(*_a, **_k):
    from mas_langgraph.cases.mcq.replay_observable import ReplayObservableStatus

    return ReplayObservableStatus.CONTINUE


class TestParentLineageHistoryCollection(unittest.TestCase):
    def test_does_not_scan_sibling_thread_history(self):
        branch_len = 4
        sibling_snapshots = 80
        graph = _LineageGraph(
            branch_id="cf_hist",
            branch_len=branch_len,
            sibling_snapshots=sibling_snapshots,
        )
        ex = LangGraphReplayExecutor(compiled_graph=graph)
        run_id = "run_hist"
        ex._runtimes[run_id] = WorkflowRuntime()
        ex.branches.register(BranchMetadata.factual(run_id))
        meta = ex.branches.fork(
            parent=ex.branches.get(run_id, "factual"),
            branch_type="counterfactual",
            origin_checkpoint_alias="c_1",
            origin_checkpoint_id="factual_root",
        )
        graph.branch_id = meta.branch_id
        graph._values["branch_id"] = meta.branch_id
        for snap in graph.chain:
            snap.values["branch_id"] = meta.branch_id
        ex._branch_fork_configs[(run_id, meta.branch_id)] = {
            "configurable": {"thread_id": run_id, "checkpoint_id": "fork"}
        }

        with replay_profile_scope():
            result = ex.replay_downstream(
                run_id,
                meta.branch_id,
                explained_case=_explained(),
                stop_when_observable=True,
                stop_predicate=_always_continue,
            )

        self.assertEqual(graph.get_state_history_calls, 0)
        self.assertTrue(result.branch_history)
        self.assertEqual(len(result.branch_history), branch_len)
        scanned = int((result.replay_span_profile or {}).get("history_snapshots_scanned") or 0)
        retained = int((result.replay_span_profile or {}).get("branch_snapshots_retained") or 0)
        self.assertEqual(retained, branch_len)
        self.assertLessEqual(scanned, branch_len + 1)
        self.assertLess(scanned, sibling_snapshots)
        ids = [extract_checkpoint_id(s) for s in result.branch_history]
        self.assertEqual(ids, [f"branch_{i}" for i in range(branch_len)])
        for snap in result.branch_history:
            self.assertEqual(snap.values.get("branch_id"), meta.branch_id)
            self.assertFalse(str(extract_checkpoint_id(snap)).startswith("synth_"))
        self.assertEqual(result.branch_metadata.final_checkpoint_id, ids[-1])
        self.assertEqual(
            (result.replay_span_profile or {}).get("history_collection_mode"),
            "parent_lineage",
        )
        self.assertEqual(result.final_state.get("branch_id"), meta.branch_id)

    def test_pinned_fork_config_is_not_used_as_latest_state(self):
        graph = _LineageGraph(branch_id="cf_hist", branch_len=3, sibling_snapshots=2)
        stale = _Snap({"branch_id": "cf_hist", "stale": True}, "fork", parent=graph.by_id["factual_root"])
        graph.by_id["fork"] = stale
        ex = LangGraphReplayExecutor(compiled_graph=graph)
        run_id = "run_hist"
        ex._runtimes[run_id] = WorkflowRuntime()
        ex.branches.register(BranchMetadata.factual(run_id))
        meta = ex.branches.fork(
            parent=ex.branches.get(run_id, "factual"),
            branch_type="counterfactual",
            origin_checkpoint_alias="c_1",
            origin_checkpoint_id="factual_root",
        )
        for snap in graph.chain:
            snap.values["branch_id"] = meta.branch_id
        graph._values["branch_id"] = meta.branch_id
        ex._branch_fork_configs[(run_id, meta.branch_id)] = {
            "configurable": {"thread_id": run_id, "checkpoint_id": "fork"}
        }
        result = ex.replay_downstream(run_id, meta.branch_id)
        last_vals = result.branch_history[-1].values
        self.assertNotIn("stale", last_vals)
        self.assertEqual(extract_checkpoint_id(result.branch_history[-1]), "branch_2")


class TestRealGraphBranchHistory(unittest.TestCase):
    def _compile(self):
        from langgraph.graph import END, START, StateGraph

        from mas_langgraph.core.checkpointer import default_factual_checkpointer

        class S(TypedDict, total=False):
            n: int
            branch_id: str
            path: Annotated[list, operator.add]
            artifact_catalog: dict
            available_artifact_refs: set
            blocked_artifact_refs: set
            artifact_substitutions: dict
            artifact_derivations: list
            intervention_events: list
            run_id: str
            branch_type: str
            parent_branch_id: str
            origin_checkpoint_alias: str
            intervention_id: Any
            provenance_node_counts: dict
            provenance_execution_index: int
            occurrence_records: list
            produced_by: list
            run_provenance_relations: list
            provenance_violations: list
            provenance_unresolved_producers: list

        def a(s: S) -> dict:
            return {"n": int(s.get("n") or 0) + 1, "path": ["a"]}

        def b(s: S) -> dict:
            return {"n": int(s.get("n") or 0) + 10, "path": ["b"]}

        def c(s: S) -> dict:
            return {"n": int(s.get("n") or 0) + 100, "path": ["c"]}

        g = StateGraph(S)
        g.add_node("a", a)
        g.add_node("b", b)
        g.add_node("c", c)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", END)
        app = g.compile(checkpointer=default_factual_checkpointer())
        return app

    def test_sibling_replays_do_not_grow_history_scan(self):
        app = self._compile()
        orig_history = app.get_state_history
        history_calls = {"n": 0}

        def counting_history(*args, **kwargs):
            history_calls["n"] += 1
            return orig_history(*args, **kwargs)

        app.get_state_history = counting_history  # type: ignore[method-assign]

        ex = LangGraphReplayExecutor(compiled_graph=app)
        run_id = "run_hist_real"
        initial = {
            "n": 0,
            "path": [],
            **empty_artifact_channels(),
        }
        factual = ex.run_factual(
            initial,
            run_id=run_id,
            runtime=WorkflowRuntime(),
        )
        self.assertIsNone(factual.error, msg=factual.error)
        factual_index_len = len(factual.checkpoint_index.records)
        origin = factual.checkpoint_index.resolve(BoundarySelector.before_next_node("b"))
        after_factual_calls = history_calls["n"]

        sibling_count = 8
        for _ in range(sibling_count):
            prev = ex.fork_noop(run_id, BoundarySelector.by_alias(origin.alias))
            self.assertIsNone(prev.error, msg=prev.error)
            self.assertTrue(prev.branch_history)
            self.assertEqual(prev.final_state.get("n"), 111)

        self.assertEqual(history_calls["n"], after_factual_calls)

        last = ex.fork_noop(run_id, BoundarySelector.by_alias(origin.alias))
        self.assertIsNone(last.error, msg=last.error)
        self.assertEqual(history_calls["n"], after_factual_calls)
        self.assertTrue(last.branch_history)
        for snap in last.branch_history:
            self.assertEqual(snap.values.get("branch_id"), last.branch_id)
            self.assertFalse(str(extract_checkpoint_id(snap)).startswith("synth_"))
        self.assertEqual(
            last.branch_metadata.final_checkpoint_id,
            extract_checkpoint_id(last.branch_history[-1]),
        )
        scanned = int((last.replay_span_profile or {}).get("history_snapshots_scanned") or 0)
        retained = len(last.branch_history)
        self.assertEqual(retained, int((last.replay_span_profile or {}).get("branch_snapshots_retained") or 0))
        self.assertLessEqual(scanned, retained + 2)
        self.assertLess(scanned, sibling_count * retained)
        self.assertEqual(len(factual.checkpoint_index.records), factual_index_len)

        full = list(orig_history({"configurable": {"thread_id": run_id}}))
        full_chrono = list(reversed(full))
        filtered = [
            s
            for s in full_chrono
            if (getattr(s, "values", None) or {}).get("branch_id") == last.branch_id
        ]
        self.assertEqual(
            [extract_checkpoint_id(s) for s in last.branch_history],
            [extract_checkpoint_id(s) for s in filtered],
        )
        self.assertEqual(last.final_state.get("n"), 111)
        self.assertIn("c", last.final_state.get("path") or [])


if __name__ == "__main__":
    unittest.main()
