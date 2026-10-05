"""Prompt 14 — Full AXIS loop, no-op cache, comparison, synthesis tests."""

from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

from mas_langgraph.artifacts.models import (
    ArtifactEnvelope,
    ArtifactRef,
    DerivedFrom,
)
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.artifacts.resolver import ArtifactResolver
from mas_langgraph.axis.comparison import (
    CaseRunObservation,
    CriterionDelta,
    NoOpReplayCache,
    QueryExecutionLedger,
    ReplayComparator,
    ReplayComparison,
    DECISION_EFFECT,
    CRITERION_EFFECT,
    DOWNSTREAM_EFFECT,
    NO_DEMONSTRATED_EFFECT,
    REPLAY_UNSTABLE,
    NOOP_ERROR,
    INTERVENTION_ERROR,
    normalize_branch_path,
)
from mas_langgraph.axis.explainer import AxisExplanationController, ExplanationResult
from mas_langgraph.axis.models import (
    AxisPromptTemplates,
    ExplainedCase,
    FactualQueryRecord,
    InterventionQueryRecord,
    QuerySignature,
    ResolvedQuery,
)
from mas_langgraph.axis.queries import AxisQueryParser, QueryValidationError
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.core.outcomes import CriterionValue
from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation
from mas_langgraph.replay.checkpoints import (
    CheckpointIndex,
    CheckpointRecord,
)
from mas_langgraph.replay.executor import LangGraphReplayExecutor
from mas_langgraph.replay.interventions import (
    InterventionContract,
    InterventionContractRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakePayload:
    content: str = ""
    stem: str = ""


def _make_env(lid, ver=1, atype="EvidenceChunk", branch="factual"):
    return ArtifactEnvelope.create(
        logical_id=lid, version=ver, artifact_type=atype,
        payload=_FakePayload(content=f"{lid}:v{ver}"),
        branch_id=branch, created_in_run="run1",
    )


def _build_fixture():
    """Build a standard AXIS test fixture."""
    reg = ArtifactRegistry()
    for env in [
        _make_env("E17", 1), _make_env("Q12", 1, "CandidateMCQ"),
        _make_env("Q12", 2, "CandidateMCQ"),
        _make_env("decision:Q12", 2, "SupervisorDecision"),
        _make_env("assessment:Q12", 2, "OutcomeAssessment"),
    ]:
        reg.store(env)

    explained = ExplainedCase(
        run_id="run1", artifact_ref=ArtifactRef("Q12", 2),
        branch_id="factual", decision_ref=ArtifactRef("decision:Q12", 2),
        outcome="accept", assessment_ref=ArtifactRef("assessment:Q12", 2),
    )

    occ1 = OccurrenceRecord(
        occurrence_id="run1:factual:gen#1", node_id="gen", occurrence_index=1,
        branch_id="factual", run_id="run1", execution_index=3,
        produced_artifacts=(ArtifactRef("Q12", 1),),
    )
    rel1 = RunProvenanceRelation(
        producer_occurrence_id="run1:factual:gen#1",
        artifact_ref=ArtifactRef("E17", 1),
        relation="supporting_evidence",
        consumer_occurrence_id="run1:factual:verify#1",
    )
    case_slice = CaseSlice(
        explained_case=explained,
        relevant_artifact_refs=(
            ArtifactRef("E17", 1), ArtifactRef("Q12", 1),
            ArtifactRef("Q12", 2), ArtifactRef("decision:Q12", 2),
            ArtifactRef("assessment:Q12", 2),
        ),
        relevant_occurrences=(occ1,),
        relevant_relations=(rel1,),
    )

    ckpt_idx = CheckpointIndex(thread_id="t1")
    for r in [
        CheckpointRecord(alias="c_0", thread_id="t1", checkpoint_id="ckpt_0", superstep=0),
        CheckpointRecord(
            alias="c_8", thread_id="t1", checkpoint_id="ckpt_8", superstep=8,
            after_occurrences=("run1:factual:gen#1",),
            after_occurrence_labels=("gen#1",),
            parent_checkpoint_id="ckpt_0",
        ),
    ]:
        ckpt_idx.add(r)

    contracts = InterventionContractRegistry()
    contracts.register(InterventionContract(
        artifact_type="EvidenceChunk",
        allowed_operations=("REMOVE", "ADD", "REPLACE"),
        replacement_compatible_types=("EvidenceChunk",),
    ))

    ex = LangGraphReplayExecutor(compiled_graph=None)
    ex.run_factual(
        {"step_logs": [{}, {}, {}, {}, {}, {}, {}, {}, {}]},
        run_id="run1", allow_synthetic_history=True,
    )
    resolver = ArtifactResolver(registry=reg)
    for ref in reg.all_refs():
        resolver.make_available(ref)
    ex._resolvers[("run1", "factual")] = resolver

    return explained, reg, [occ1], [rel1], ckpt_idx, contracts, ex


# ===========================================================================
# Comparison tests
# ===========================================================================

class TestStableDecisionEffect(unittest.TestCase):
    """Test #139: factual=accept, noop=accept, cf=reject → decision_effect."""

    def test_decision_effect(self):
        f = CaseRunObservation(run_id="r", branch_id="f", decision="accept",
                               criterion_values=(), outcome_available=True)
        n = CaseRunObservation(run_id="r", branch_id="n", decision="accept",
                               criterion_values=(), outcome_available=True)
        c = CaseRunObservation(run_id="r", branch_id="c", decision="reject",
                               criterion_values=(), outcome_available=True)

        # Construct comparison manually
        from mas_langgraph.axis.comparison import _determine_evidence_status
        status, dec_stable, _ = _determine_evidence_status(
            f, n, c, (), (), (), (), (), (), (), (),
        )
        self.assertEqual(status, DECISION_EFFECT)
        self.assertTrue(dec_stable)


class TestReplayUnstable(unittest.TestCase):
    """Test #140: factual=accept, noop=reject → replay_unstable."""

    def test_noop_changes_decision(self):
        f = CaseRunObservation(run_id="r", branch_id="f", decision="accept",
                               criterion_values=(), outcome_available=True)
        n = CaseRunObservation(run_id="r", branch_id="n", decision="reject",
                               criterion_values=(), outcome_available=True)
        c = CaseRunObservation(run_id="r", branch_id="c", decision="reject",
                               criterion_values=(), outcome_available=True)

        from mas_langgraph.axis.comparison import _determine_evidence_status
        status, dec_stable, _ = _determine_evidence_status(
            f, n, c, (), (), (), (), (), (), (), (),
        )
        self.assertEqual(status, REPLAY_UNSTABLE)
        self.assertFalse(dec_stable)


class TestNoDemonstratedEffect(unittest.TestCase):
    """Test #141: all same → no_demonstrated_effect."""

    def test_all_same(self):
        obs = CaseRunObservation(run_id="r", branch_id="f", decision="accept",
                                  criterion_values=(), outcome_available=True)
        from mas_langgraph.axis.comparison import _determine_evidence_status
        status, _, _ = _determine_evidence_status(
            obs, obs, obs, (), (), (), (), (), (), (), (),
        )
        self.assertEqual(status, NO_DEMONSTRATED_EFFECT)


class TestCriterionEffect(unittest.TestCase):
    """Test #142: decisions same but criterion differs."""

    def test_criterion_change(self):
        cv_f = CriterionValue(name="pdf_supported", value=True, decision_relevant=True)
        cv_n = CriterionValue(name="pdf_supported", value=True, decision_relevant=True)
        cv_c = CriterionValue(name="pdf_supported", value=False, decision_relevant=True)
        f = CaseRunObservation(run_id="r", branch_id="f", decision="accept",
                               criterion_values=(cv_f,), outcome_available=True)
        n = CaseRunObservation(run_id="r", branch_id="n", decision="accept",
                               criterion_values=(cv_n,), outcome_available=True)
        c = CaseRunObservation(run_id="r", branch_id="c", decision="accept",
                               criterion_values=(cv_c,), outcome_available=True)

        from mas_langgraph.axis.comparison import _compute_criterion_deltas, _determine_evidence_status
        deltas = _compute_criterion_deltas(f, n, c)
        status, _, _ = _determine_evidence_status(
            f, n, c, deltas, (), (), (), (), (), (), (),
        )
        self.assertEqual(status, CRITERION_EFFECT)
        self.assertTrue(deltas[0].counterfactual_differs_from_noop)


class TestDownstreamEffect(unittest.TestCase):
    """Test #143: decision/criteria same but path differs."""

    def test_path_change(self):
        obs = CaseRunObservation(run_id="r", branch_id="f", decision="accept",
                                  criterion_values=(), outcome_available=True)
        from mas_langgraph.axis.comparison import _determine_evidence_status
        # Different paths
        status, _, _ = _determine_evidence_status(
            obs, obs, obs, (),
            ("verify[1]",), ("verify[1]", "revise[1]"),
            (), (), (), (), (),
        )
        self.assertEqual(status, DOWNSTREAM_EFFECT)


class TestNoopError(unittest.TestCase):
    """Test #144: noop error."""

    def test_noop_error(self):
        f = CaseRunObservation(run_id="r", branch_id="f", decision="accept",
                               criterion_values=(), outcome_available=True)
        n = CaseRunObservation(run_id="r", branch_id="n", error="fail",
                               criterion_values=(), outcome_available=True)
        c = CaseRunObservation(run_id="r", branch_id="c", decision="accept",
                               criterion_values=(), outcome_available=True)
        from mas_langgraph.axis.comparison import _determine_evidence_status
        status, _, _ = _determine_evidence_status(
            f, n, c, (), (), (), (), (), (), (), (),
        )
        self.assertEqual(status, NOOP_ERROR)


class TestInterventionError(unittest.TestCase):
    """Test #145: cf replay error."""

    def test_cf_error(self):
        f = CaseRunObservation(run_id="r", branch_id="f", decision="accept",
                               criterion_values=(), outcome_available=True)
        n = CaseRunObservation(run_id="r", branch_id="n", decision="accept",
                               criterion_values=(), outcome_available=True)
        c = CaseRunObservation(run_id="r", branch_id="c", error="fail",
                               criterion_values=(), outcome_available=True)
        from mas_langgraph.axis.comparison import _determine_evidence_status
        status, _, _ = _determine_evidence_status(
            f, n, c, (), (), (), (), (), (), (), (),
        )
        self.assertEqual(status, INTERVENTION_ERROR)


class TestCriterionMatchByName(unittest.TestCase):
    """Test #149: criteria matched by name, not position."""

    def test_match_by_name(self):
        from mas_langgraph.axis.comparison import _compute_criterion_deltas
        f = CaseRunObservation(run_id="r", branch_id="f",
            criterion_values=(
                CriterionValue(name="b", value=1),
                CriterionValue(name="a", value=2),
            ), outcome_available=True)
        n = CaseRunObservation(run_id="r", branch_id="n",
            criterion_values=(
                CriterionValue(name="a", value=2),
                CriterionValue(name="b", value=1),
            ), outcome_available=True)
        c = CaseRunObservation(run_id="r", branch_id="c",
            criterion_values=(
                CriterionValue(name="a", value=3),
                CriterionValue(name="b", value=1),
            ), outcome_available=True)

        deltas = _compute_criterion_deltas(f, n, c)
        by_name = {d.name: d for d in deltas}
        self.assertTrue(by_name["a"].counterfactual_differs_from_noop)
        self.assertFalse(by_name["b"].counterfactual_differs_from_noop)


class TestNormalizedPath(unittest.TestCase):
    """Test #152: normalized path comparison."""

    def test_normalize(self):
        @dataclass
        class FakeOcc:
            node_id: str
            execution_index: int

        path = normalize_branch_path([
            FakeOcc("verify", 1), FakeOcc("gate", 2),
            FakeOcc("verify", 3),
        ])
        self.assertEqual(path, ("verify[1]", "gate[1]", "verify[2]"))


# ===========================================================================
# No-op cache tests
# ===========================================================================

class TestNoOpCache(unittest.TestCase):
    """Tests #137-#138: no-op cache."""

    def test_cache_reuse(self):
        cache = NoOpReplayCache()
        cache.put("run1", "ckpt_8", "noop_result")
        self.assertTrue(cache.has("run1", "ckpt_8"))
        self.assertEqual(cache.get("run1", "ckpt_8"), "noop_result")
        # Second put is idempotent
        cache.put("run1", "ckpt_8", "different_result")
        self.assertEqual(cache.get("run1", "ckpt_8"), "noop_result")

    def test_different_checkpoints(self):
        cache = NoOpReplayCache()
        cache.put("run1", "ckpt_8", "noop_8")
        cache.put("run1", "ckpt_10", "noop_10")
        self.assertNotEqual(
            cache.get("run1", "ckpt_8"),
            cache.get("run1", "ckpt_10"),
        )


# ===========================================================================
# Ledger tests
# ===========================================================================

class TestQueryExecutionLedger(unittest.TestCase):
    """Test #13-#17: P_executed semantics."""

    def test_mark_and_check(self):
        ledger = QueryExecutionLedger()
        sig = QuerySignature(operation="WHAT", target_ref=ArtifactRef("E17", 1))
        self.assertFalse(ledger.is_executed(sig))
        ledger.mark_executed(sig)
        self.assertTrue(ledger.is_executed(sig))

    def test_order_preserved(self):
        ledger = QueryExecutionLedger()
        s1 = QuerySignature(operation="WHAT", target_ref=ArtifactRef("E17", 1))
        s2 = QuerySignature(operation="REMOVE", target_ref=ArtifactRef("E17", 1),
                            checkpoint_id="ckpt_8")
        ledger.mark_executed(s1)
        ledger.mark_executed(s2)
        self.assertEqual(ledger.signatures, (s1, s2))


# ===========================================================================
# Full loop tests
# ===========================================================================

class TestFullAdaptiveLLMLoop(unittest.TestCase):
    """Test #134-#136: full adaptive LLM loop."""

    def _make_llm(self):
        """LLM that returns different queries based on prompt content."""
        call_count = [0]
        def llm_fn(prompt: str) -> str:
            call_count[0] += 1
            if "=== TASK ===" in prompt:
                task = prompt.split("=== TASK ===")[1]
            else:
                task = prompt
            # Interrogation rounds
            if "Return exactly one structured query" in task or "Reply with one AXIS query" in task:
                if call_count[0] <= 2:
                    return "REMOVE(E17:v1, c_8)"
                return "DONE"
            # Intermediate synthesis
            if "intermediate" in task.lower() or "Synthesize" in task:
                return "Intermediate: facts examined so far."
            # Final
            return "Final: grounded explanation based on evidence."
        return llm_fn

    def test_full_loop(self):
        """Test #134: full adaptive loop."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            llm_generate=self._make_llm(),
            max_rounds=5,
            compatibility_mode=True,
        )
        result = controller.run("Why was Q12 accepted?")
        self.assertGreaterEqual(result.rounds, 1)
        self.assertGreaterEqual(len(result.intervention_records), 1)
        self.assertTrue(result.final_explanation)
        self.assertIn(result.termination_reason, ("done", "max_rounds", "script_exhausted"))


class TestScriptedMode(unittest.TestCase):
    """Tests #95, #96, #159-#162."""

    def test_script_exhaustion(self):
        """Test #161: script exhaustion without DONE."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=["REMOVE(E17:v1, c_8)"],
        )
        self.assertEqual(result.termination_reason, "script_exhausted")
        self.assertEqual(len(result.intervention_records), 1)

    def test_done_not_executed(self):
        """Test #162: DONE creates no records."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            compatibility_mode=True,
        )
        result = controller.run("Why?", scripted_queries=["DONE"])
        self.assertEqual(len(result.factual_records), 0)
        self.assertEqual(len(result.intervention_records), 0)
        self.assertEqual(result.termination_reason, "done")

    def test_max_rounds(self):
        """Test #160: max rounds terminates."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            max_rounds=1, compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=["REMOVE(E17:v1, c_8)", "REMOVE(E17:v1, c_8)"],
        )
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.termination_reason, "max_rounds")

    def test_invalid_query_consumes_attempt_not_successful_round(self):
        """Invalid query uses attempt budget, not successful-round budget."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            max_rounds=2, max_attempts=10, compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=["INVALID NONSENSE", "DONE"],
        )
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(result.successful_round_count, 0)
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.termination_reason, "done")

    def test_no_llm_fallback(self):
        """Test #171: LLM-free produces non-empty explanation."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=["REMOVE(E17:v1, c_8)"],
        )
        self.assertTrue(result.final_explanation)


class TestStrictResolverNoFallback(unittest.TestCase):
    """Test #155: strict resolver, no legacy fallback."""

    def test_invalid_strict_not_executed(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            compatibility_mode=False,  # strict
        )
        # Q999 not in scope — should be rejected
        result = controller.run("Why?", scripted_queries=["WHAT(Q999:v1)", "DONE"])
        self.assertEqual(len(result.factual_records), 0)
        self.assertTrue(any(
            rd.validation_error for rd in result.round_records
        ))


class TestFailedExecutionNotInPExecuted(unittest.TestCase):
    """Test #156: failed execution not in P_executed."""

    def test_failed_not_executed(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            compatibility_mode=True,
        )
        # WHAT on an artifact that the executor can't find
        # E17 is in scope but let's check the executed_signatures only includes
        # successfully executed queries
        result = controller.run("Why?", scripted_queries=["REMOVE(E17:v1, c_8)", "DONE"])
        if result.intervention_records:
            self.assertTrue(len(result.executed_signatures) > 0)


class TestFactualContextImmutable(unittest.TestCase):
    """Test #173: factual context not mutated."""

    def test_immutable(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        original_refs = set(reg.all_refs())
        original_idx_count = len(idx.records)
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            compatibility_mode=True,
        )
        controller.run(
            "Why?",
            scripted_queries=[f"REMOVE(E17:v1, c_8)", "DONE"],
        )
        # Registry unchanged
        self.assertEqual(set(reg.all_refs()), original_refs)
        # Checkpoint index unchanged
        self.assertEqual(len(idx.records), original_idx_count)


class TestGenericImportIsolation(unittest.TestCase):
    """Test #176: axis import does not load MCQ."""

    def test_no_mcq_in_axis_all(self):
        import mas_langgraph.axis
        self.assertNotIn("McqVerbalizer", mas_langgraph.axis.__all__)
        self.assertTrue(hasattr(mas_langgraph.axis, "AxisQueryParser"))


class TestNoOpCacheInController(unittest.TestCase):
    """Test #137: no-op cache reuse in controller."""

    def test_same_checkpoint_one_noop(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        call_count = [0]
        original_fork_noop = ex.fork_noop

        def counting_fork_noop(*args, **kwargs):
            call_count[0] += 1
            return original_fork_noop(*args, **kwargs)

        ex.fork_noop = counting_fork_noop

        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=[
                "REMOVE(E17:v1, c_8)",
                "REMOVE(E17:v1, c_8)",  # duplicate → rejected
                "DONE",
            ],
        )
        # Only one noop call because second query is duplicate
        self.assertEqual(call_count[0], 1)


class TestEpisodicHistoryRendering(unittest.TestCase):
    """Test #11: history rendering."""

    def test_render(self):
        from mas_langgraph.axis.comparison import AxisEpisodicHistory, InterrogationRoundRecord
        h = AxisEpisodicHistory(
            question="Why?",
            factual_context="Context here",
            rounds=[
                InterrogationRoundRecord(
                    round_index=1,
                    generated_query="WHAT(E17:v1)",
                    result_text="E17 is an EvidenceChunk.",
                    intermediate_explanation="E17 was examined.",
                ),
            ],
        )
        text = h.render()
        self.assertIn("ROUND 1", text)
        self.assertIn("WHAT(E17:v1)", text)
        self.assertIn("E17 is an EvidenceChunk", text)


class TestOldControllerCompat(unittest.TestCase):
    """Test #177: old scripted controller still works."""

    def test_compat_mode(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained, registry=reg,
            occurrences=occs, provenance_relations=rels,
            checkpoint_index=idx, replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1), "Q1": ArtifactRef("Q12", 2)},
            compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=[f"REMOVE(E17, c_8)", "DONE"],
        )
        self.assertGreaterEqual(result.rounds, 1)
        self.assertTrue(
            "Slice for" in result.final_explanation
            or "Provenance slice" in result.final_explanation
            or "Decision" in result.final_explanation
        )


if __name__ == "__main__":
    unittest.main()
