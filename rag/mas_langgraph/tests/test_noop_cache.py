"""NOOP replay cache: one control per factual run + checkpoint + config."""

from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from mas_langgraph.artifacts.models import ArtifactEnvelope, ArtifactRef
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.artifacts.resolver import ArtifactResolver
from mas_langgraph.axis.comparison import (
    CaseRunObservation,
    NoOpReplayCache,
    NoopReplayCacheKey,
    ReplayComparator,
    REPLAY_UNSTABLE,
)
from mas_langgraph.axis.executor import AxisQueryExecutor
from mas_langgraph.axis.explainer import AxisExplanationController
from mas_langgraph.axis.export import _query_round_dict
from mas_langgraph.axis.models import (
    AxisQueryContext,
    ExplainedCase,
    InterventionQueryRecord,
    ResolvedQuery,
)
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.core.outcomes import CriterionValue
from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation
from mas_langgraph.replay.checkpoints import CheckpointIndex, CheckpointRecord
from mas_langgraph.replay.executor import LangGraphReplayExecutor, WorkflowRunResult
from mas_langgraph.replay.interventions import (
    InterventionContract,
    InterventionContractRegistry,
)


@dataclass
class _FakePayload:
    content: str = ""


def _env(lid: str, atype: str = "EvidenceChunk") -> ArtifactEnvelope:
    return ArtifactEnvelope.create(
        logical_id=lid,
        version=1,
        artifact_type=atype,
        payload=_FakePayload(content=lid),
        branch_id="factual",
        created_in_run="run1",
    )


def _build_multi_target_fixture():
    reg = ArtifactRegistry()
    for lid in ("E17", "E18", "E19", "Q12"):
        atype = "CandidateMCQ" if lid == "Q12" else "EvidenceChunk"
        reg.store(_env(lid, atype))
    for lid in ("decision:Q12", "assessment:Q12"):
        atype = "SupervisorDecision" if lid.startswith("decision") else "OutcomeAssessment"
        reg.store(_env(lid, atype))

    explained = ExplainedCase(
        run_id="run1",
        artifact_ref=ArtifactRef("Q12", 1),
        branch_id="factual",
        decision_ref=ArtifactRef("decision:Q12", 1),
        outcome="accept",
        assessment_ref=ArtifactRef("assessment:Q12", 1),
    )
    occ = OccurrenceRecord(
        occurrence_id="run1:factual:gen#1",
        node_id="gen",
        occurrence_index=1,
        branch_id="factual",
        run_id="run1",
        execution_index=3,
        produced_artifacts=(ArtifactRef("Q12", 1),),
    )
    rels = [
        RunProvenanceRelation(
            producer_occurrence_id="run1:factual:gen#1",
            artifact_ref=ArtifactRef(lid, 1),
            relation="supporting_evidence",
            consumer_occurrence_id="run1:factual:verify#1",
        )
        for lid in ("E17", "E18", "E19")
    ]
    case_slice = CaseSlice(
        explained_case=explained,
        relevant_artifact_refs=tuple(ArtifactRef(x, 1) for x in ("E17", "E18", "E19", "Q12")),
        relevant_occurrences=(occ,),
        relevant_relations=tuple(rels),
        decision="accept",
        criterion_values=(
            CriterionValue(name="supported_by_pdf", value=True, decision_relevant=True),
        ),
    )
    idx = CheckpointIndex(thread_id="run1")
    idx.add(CheckpointRecord(alias="c_0", thread_id="run1", checkpoint_id="ckpt_0", superstep=0))
    idx.add(
        CheckpointRecord(
            alias="c_8",
            thread_id="run1",
            checkpoint_id="ckpt_8",
            superstep=8,
            after_occurrences=("run1:factual:gen#1",),
            parent_checkpoint_id="ckpt_0",
        )
    )
    idx.add(
        CheckpointRecord(
            alias="c_10",
            thread_id="run1",
            checkpoint_id="ckpt_10",
            superstep=10,
            after_occurrences=("run1:factual:gen#1",),
            parent_checkpoint_id="ckpt_8",
        )
    )
    contracts = InterventionContractRegistry()
    contracts.register(
        InterventionContract(
            artifact_type="EvidenceChunk",
            allowed_operations=("REMOVE", "ADD", "REPLACE"),
            replacement_compatible_types=("EvidenceChunk",),
        )
    )
    ex = LangGraphReplayExecutor(compiled_graph=None)
    ex.run_factual(
        {"step_logs": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]},
        run_id="run1",
        allow_synthetic_history=True,
    )
    resolver = ArtifactResolver(registry=reg)
    for ref in reg.all_refs():
        resolver.make_available(ref)
    ex._resolvers[("run1", "factual")] = resolver
    return explained, reg, case_slice, idx, contracts, ex


class TestNoopCacheKeyIdentity(unittest.TestCase):
    def test_key_excludes_intervention(self):
        k1 = NoopReplayCacheKey("run1", "ckpt_8", "fixture")
        k2 = NoopReplayCacheKey("run1", "ckpt_8", "fixture")
        self.assertEqual(k1, k2)
        self.assertNotEqual(
            NoopReplayCacheKey("run1", "ckpt_8", "fixture"),
            NoopReplayCacheKey("run2", "ckpt_8", "fixture"),
        )
        self.assertNotEqual(
            NoopReplayCacheKey("run1", "ckpt_8", "fixture"),
            NoopReplayCacheKey("run1", "ckpt_10", "fixture"),
        )


class TestSameCheckpointOneNoop(unittest.TestCase):
    def test_three_interventions_one_noop(self):
        explained, reg, case_slice, idx, contracts, ex = _build_multi_target_fixture()
        cache = NoOpReplayCache(log=False)
        ctx = AxisQueryContext(
            explained_case=explained,
            case_slice=case_slice,
            registry=reg,
            checkpoint_index=idx,
            contracts=contracts,
        )
        executor = AxisQueryExecutor(
            context=ctx, replay_executor=ex, noop_cache=cache
        )
        call_count = [0]
        original = ex.fork_noop

        def counting(*a, **k):
            call_count[0] += 1
            return original(*a, **k)

        ex.fork_noop = counting  # type: ignore[method-assign]

        records: List[InterventionQueryRecord] = []
        for lid in ("E17", "E18", "E19"):
            rq = ResolvedQuery(
                operation="REMOVE",
                raw_text=f"REMOVE({lid}:v1, c_8)",
                target_ref=ArtifactRef(lid, 1),
                checkpoint_alias="c_8",
                checkpoint_id="ckpt_8",
            )
            rec = executor.execute(rq)
            self.assertIsInstance(rec, InterventionQueryRecord)
            self.assertIsNone(rec.error)
            records.append(rec)

        self.assertEqual(call_count[0], 1)
        self.assertEqual(cache.noop_replay_requests, 3)
        self.assertEqual(cache.noop_replay_executions, 1)
        self.assertEqual(cache.noop_cache_hits, 2)
        self.assertEqual(cache.counterfactual_replay_executions, 3)
        self.assertFalse(records[0].noop_reused)
        self.assertTrue(records[1].noop_reused)
        self.assertTrue(records[2].noop_reused)
        # Same control branch identity
        self.assertEqual(records[0].noop_branch_id, records[1].noop_branch_id)
        self.assertEqual(records[1].noop_branch_id, records[2].noop_branch_id)
        # Distinct CF branches
        cf_ids = {r.counterfactual_branch_id for r in records}
        self.assertEqual(len(cf_ids), 3)


class TestDifferentCheckpoints(unittest.TestCase):
    def test_two_checkpoints_two_noops(self):
        explained, reg, case_slice, idx, contracts, ex = _build_multi_target_fixture()
        cache = NoOpReplayCache(log=False)
        ctx = AxisQueryContext(
            explained_case=explained,
            case_slice=case_slice,
            registry=reg,
            checkpoint_index=idx,
            contracts=contracts,
        )
        executor = AxisQueryExecutor(
            context=ctx, replay_executor=ex, noop_cache=cache
        )
        call_count = [0]
        original = ex.fork_noop

        def counting(*a, **k):
            call_count[0] += 1
            return original(*a, **k)

        ex.fork_noop = counting  # type: ignore[method-assign]

        for alias, lid in (("c_8", "E17"), ("c_10", "E18")):
            rq = ResolvedQuery(
                operation="REMOVE",
                raw_text=f"REMOVE({lid}:v1, {alias})",
                target_ref=ArtifactRef(lid, 1),
                checkpoint_alias=alias,
            )
            rec = executor.execute(rq)
            self.assertIsNone(rec.error)

        self.assertEqual(call_count[0], 2)
        self.assertEqual(cache.noop_replay_executions, 2)
        self.assertEqual(cache.noop_cache_hits, 0)


class TestDifferentRuns(unittest.TestCase):
    def test_no_cross_run_reuse(self):
        cache = NoOpReplayCache(log=False)
        calls = {"n": 0}

        def factory_a():
            calls["n"] += 1
            return WorkflowRunResult(
                run_id="runA", branch_id="noop_a", final_state={"marker": "A"}
            )

        def factory_b():
            calls["n"] += 1
            return WorkflowRunResult(
                run_id="runB", branch_id="noop_b", final_state={"marker": "B"}
            )

        r1, reused1, _ = cache.get_or_run(
            run_id="runA", checkpoint_id="ckpt_8", factory=factory_a
        )
        r2, reused2, _ = cache.get_or_run(
            run_id="runB", checkpoint_id="ckpt_8", factory=factory_b
        )
        self.assertFalse(reused1)
        self.assertFalse(reused2)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(r1.final_state["marker"], "A")
        self.assertEqual(r2.final_state["marker"], "B")
        self.assertEqual(cache.noop_replay_executions, 2)


class TestSharedBranchDifferentCases(unittest.TestCase):
    def test_observe_twice_from_one_noop(self):
        cache = NoOpReplayCache(log=False)
        calls = [0]

        def factory():
            calls[0] += 1
            return WorkflowRunResult(
                run_id="run1",
                branch_id="noop_shared",
                final_state={
                    "branch_id": "noop_shared",
                    "cooperative_state": MagicMock(
                        candidate_mcqs={
                            "q1": MagicMock(blueprint_id="bp1"),
                            "q2": MagicMock(blueprint_id="bp2"),
                        }
                    ),
                    "artifact_catalog": {},
                    "occurrence_records": [],
                    "run_provenance_relations": [],
                },
            )

        noop1, reused1, _ = cache.get_or_run(
            run_id="run1", checkpoint_id="ckpt_8", factory=factory
        )
        noop2, reused2, _ = cache.get_or_run(
            run_id="run1", checkpoint_id="ckpt_8", factory=factory
        )
        self.assertEqual(calls[0], 1)
        self.assertFalse(reused1)
        self.assertTrue(reused2)
        self.assertEqual(noop1.branch_id, noop2.branch_id)

        # Case-specific observations remain separate objects.
        obs_q1 = CaseRunObservation(
            run_id="run1", branch_id=noop1.branch_id, decision="accept", subject_present=True
        )
        obs_q2 = CaseRunObservation(
            run_id="run1", branch_id=noop2.branch_id, decision="accept", subject_present=True
        )
        self.assertEqual(obs_q1.decision, obs_q2.decision)
        self.assertIsNot(obs_q1, obs_q2)


class TestStableAndUnstableReuse(unittest.TestCase):
    def test_stable_control_reuse(self):
        factual = CaseRunObservation(
            run_id="r",
            branch_id="factual",
            decision="accept",
            criterion_values=(
                CriterionValue(name="c1", value=True, decision_relevant=True),
            ),
        )
        noop_res = WorkflowRunResult(
            run_id="r", branch_id="noop1", final_state={"branch_id": "noop1"}
        )
        cf1 = WorkflowRunResult(
            run_id="r", branch_id="cf1", final_state={"branch_id": "cf1"}
        )
        cf2 = WorkflowRunResult(
            run_id="r", branch_id="cf2", final_state={"branch_id": "cf2"}
        )

        class _Adapter:
            def observe_replay(self, *, explained_case, run_result):
                return CaseRunObservation(
                    run_id=run_result.run_id,
                    branch_id=run_result.branch_id,
                    decision="accept",
                    criterion_values=factual.criterion_values,
                    subject_present=True,
                    outcome_available=True,
                )

        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q1", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("d", 1),
            outcome="accept",
        )
        cmp = ReplayComparator()
        from mas_langgraph.axis.models import QuerySignature

        sig = QuerySignature(operation="REMOVE", target_ref=ArtifactRef("E1", 1))
        c1 = cmp.compare(
            signature=sig,
            factual_observation=factual,
            noop_result=noop_res,
            counterfactual_result=cf1,
            observation_adapter=_Adapter(),
            explained_case=explained,
        )
        c2 = cmp.compare(
            signature=sig,
            factual_observation=factual,
            noop_result=noop_res,
            counterfactual_result=cf2,
            observation_adapter=_Adapter(),
            explained_case=explained,
        )
        self.assertTrue(c1.decision_control_stable)
        self.assertTrue(c1.criteria_control_stable)
        self.assertTrue(c2.decision_control_stable)
        self.assertTrue(c2.criteria_control_stable)

    def test_unstable_cached_control_not_rerun(self):
        cache = NoOpReplayCache(log=False)
        calls = [0]

        def factory():
            calls[0] += 1
            return WorkflowRunResult(
                run_id="r",
                branch_id="noop_unstable",
                final_state={"branch_id": "noop_unstable", "decision": "reject"},
            )

        noop, _, _ = cache.get_or_run(
            run_id="r", checkpoint_id="ckpt_8", factory=factory
        )
        noop2, reused, _ = cache.get_or_run(
            run_id="r", checkpoint_id="ckpt_8", factory=factory
        )
        self.assertEqual(calls[0], 1)
        self.assertTrue(reused)

        factual = CaseRunObservation(
            run_id="r", branch_id="factual", decision="accept", subject_present=True
        )

        class _Adapter:
            def observe_replay(self, *, explained_case, run_result):
                # Cached NOOP remains REJECT for every observation.
                return CaseRunObservation(
                    run_id=run_result.run_id,
                    branch_id=run_result.branch_id,
                    decision="reject",
                    subject_present=True,
                    outcome_available=True,
                )

        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q1", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("d", 1),
            outcome="accept",
        )
        from mas_langgraph.axis.models import QuerySignature

        sig = QuerySignature(operation="REMOVE", target_ref=ArtifactRef("E1", 1))
        cmp = ReplayComparator()
        for cf_bid in ("cf1", "cf2"):
            cf = WorkflowRunResult(
                run_id="r", branch_id=cf_bid, final_state={"branch_id": cf_bid}
            )
            out = cmp.compare(
                signature=sig,
                factual_observation=factual,
                noop_result=noop if cf_bid == "cf1" else noop2,
                counterfactual_result=cf,
                observation_adapter=_Adapter(),
                explained_case=explained,
            )
            self.assertEqual(out.evidence_status, REPLAY_UNSTABLE)
        self.assertEqual(calls[0], 1)


class TestCachedObjectImmutability(unittest.TestCase):
    def test_caller_mutation_does_not_corrupt_cache(self):
        cache = NoOpReplayCache(log=False)

        def factory():
            return WorkflowRunResult(
                run_id="r",
                branch_id="noop1",
                final_state={"branch_id": "noop1", "marker": "clean"},
            )

        first, _, _ = cache.get_or_run(
            run_id="r", checkpoint_id="ckpt_8", factory=factory
        )
        first.final_state["marker"] = "mutated"
        second, reused, _ = cache.get_or_run(
            run_id="r", checkpoint_id="ckpt_8", factory=factory
        )
        self.assertTrue(reused)
        self.assertEqual(second.final_state["marker"], "clean")
        # Underlying cache entry remains clean.
        cached = cache.get("r", "ckpt_8")
        self.assertEqual(cached.final_state["marker"], "clean")

    def test_nested_state_mutation_does_not_corrupt_cache(self):
        cache = NoOpReplayCache(log=False)

        def factory():
            return WorkflowRunResult(
                run_id="r",
                branch_id="noop1",
                final_state={"branch_id": "noop1", "nested": {"k": "v"}},
                branch_history=["snap-a"],
            )

        first, _, _ = cache.get_or_run(
            run_id="r", checkpoint_id="ckpt_8", factory=factory
        )
        first.final_state["nested"]["k"] = "mutated"
        first.branch_history.append("snap-b")
        second, reused, _ = cache.get_or_run(
            run_id="r", checkpoint_id="ckpt_8", factory=factory
        )
        self.assertTrue(reused)
        self.assertEqual(second.final_state["nested"]["k"], "v")
        self.assertEqual(list(second.branch_history), ["snap-a"])
        cached = cache.get("r", "ckpt_8")
        self.assertEqual(cached.final_state["nested"]["k"], "v")
        self.assertEqual(list(cached.branch_history), ["snap-a"])

    def test_history_snapshots_are_shared_not_deepcopied(self):
        cache = NoOpReplayCache(log=False)
        snap = {"id": "heavy-snapshot"}

        def factory():
            return WorkflowRunResult(
                run_id="r",
                branch_id="noop1",
                final_state={"branch_id": "noop1"},
                branch_history=[snap],
            )

        first, _, _ = cache.get_or_run(
            run_id="r", checkpoint_id="ckpt_8", factory=factory
        )
        second, reused, _ = cache.get_or_run(
            run_id="r", checkpoint_id="ckpt_8", factory=factory
        )
        self.assertTrue(reused)
        self.assertIs(first.branch_history[0], second.branch_history[0])
        cached = cache.get("r", "ckpt_8")
        self.assertIs(second.branch_history[0], cached.branch_history[0])


class TestExportAuditability(unittest.TestCase):
    def test_same_noop_branch_different_cf(self):
        recs = [
            InterventionQueryRecord(
                generated_query="REMOVE(E17:v1, c_8)",
                resolved=ResolvedQuery(
                    operation="REMOVE",
                    raw_text="REMOVE(E17:v1, c_8)",
                    target_ref=ArtifactRef("E17", 1),
                    checkpoint_alias="c_8",
                ),
                noop_branch_id="noop_shared",
                counterfactual_branch_id="cf_1",
                noop_reused=False,
                noop_cache_key="run1|ckpt_8|fixture",
                comparison={"noop_branch": "noop_shared", "counterfactual_branch": "cf_1"},
            ),
            InterventionQueryRecord(
                generated_query="REMOVE(E18:v1, c_8)",
                resolved=ResolvedQuery(
                    operation="REMOVE",
                    raw_text="REMOVE(E18:v1, c_8)",
                    target_ref=ArtifactRef("E18", 1),
                    checkpoint_alias="c_8",
                ),
                noop_branch_id="noop_shared",
                counterfactual_branch_id="cf_2",
                noop_reused=True,
                noop_cache_key="run1|ckpt_8|fixture",
                comparison={"noop_branch": "noop_shared", "counterfactual_branch": "cf_2"},
            ),
        ]
        exported = [_query_round_dict(r, "intervention") for r in recs]
        self.assertEqual(exported[0]["noop_branch_id"], exported[1]["noop_branch_id"])
        self.assertNotEqual(
            exported[0]["counterfactual_branch_id"],
            exported[1]["counterfactual_branch_id"],
        )
        self.assertFalse(exported[0]["noop_reused"])
        self.assertTrue(exported[1]["noop_reused"])


class TestControllerUsesCache(unittest.TestCase):
    def test_scripted_three_removes_one_noop(self):
        explained, reg, case_slice, idx, contracts, ex = _build_multi_target_fixture()
        call_count = [0]
        original = ex.fork_noop

        def counting(*a, **k):
            call_count[0] += 1
            return original(*a, **k)

        ex.fork_noop = counting  # type: ignore[method-assign]
        cache = NoOpReplayCache(log=False)
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=list(case_slice.relevant_occurrences),
            provenance_relations=list(case_slice.relevant_relations),
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={
                "E17": ArtifactRef("E17", 1),
                "E18": ArtifactRef("E18", 1),
                "E19": ArtifactRef("E19", 1),
            },
            compatibility_mode=True,
            noop_cache=cache,
        )
        result = controller.run(
            "Why?",
            scripted_queries=[
                "REMOVE(E17:v1, c_8)",
                "REMOVE(E18:v1, c_8)",
                "REMOVE(E19:v1, c_8)",
                "DONE",
            ],
        )
        self.assertEqual(call_count[0], 1)
        self.assertEqual(result.noop_cache_stats.get("noop_replay_executions"), 1)
        self.assertEqual(result.noop_cache_stats.get("noop_cache_hits"), 2)
        self.assertEqual(
            result.noop_cache_stats.get("counterfactual_replay_executions"), 3
        )


class TestCacheHitDoesNotExecuteGraph(unittest.TestCase):
    def test_second_get_or_run_skips_factory(self):
        cache = NoOpReplayCache(log=False)
        calls = [0]

        def factory():
            calls[0] += 1
            return WorkflowRunResult(
                run_id="run1",
                branch_id="noop_1",
                final_state={"marker": "cached"},
            )

        first, reused1, _ = cache.get_or_run(
            run_id="run1", checkpoint_id="ckpt_8", factory=factory, checkpoint_alias="c_8"
        )
        self.assertFalse(reused1)
        self.assertEqual(calls[0], 1)
        self.assertTrue(cache.last_profile.get("graph_executed"))

        second, reused2, _ = cache.get_or_run(
            run_id="run1", checkpoint_id="ckpt_8", factory=factory, checkpoint_alias="c_8"
        )
        self.assertTrue(reused2)
        self.assertEqual(calls[0], 1)
        self.assertFalse(cache.last_profile.get("graph_executed"))
        self.assertTrue(cache.last_profile.get("cache_hit"))
        self.assertEqual(second.final_state["marker"], "cached")
        self.assertIsNot(first, second)
        for key in (
            "cache_lookup_s",
            "cache_factory_s",
            "cached_result_clone_s",
            "cached_result_conversion_s",
            "other_s",
            "total_noop_s",
        ):
            self.assertIn(key, cache.last_profile)
            self.assertGreaterEqual(float(cache.last_profile[key]), 0.0)
        self.assertEqual(float(cache.last_profile["cache_factory_s"]), 0.0)


if __name__ == "__main__":
    unittest.main()
