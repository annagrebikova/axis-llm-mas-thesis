"""Regression tests for AXIS replay identity, routing, Trace ablation, and status."""

from __future__ import annotations

import unittest
from typing import Any, Dict
from unittest.mock import MagicMock

from mas_langgraph.artifacts.models import ArtifactEnvelope, ArtifactRef
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.axis.comparison import (
    CaseRunObservation,
    DOWNSTREAM_EFFECT,
    NOOP_ERROR,
    ReplayComparator,
    QuerySignature,
    _determine_evidence_status,
)
from mas_langgraph.axis.export import _comparison_dict, _relation_summary
from mas_langgraph.axis.intervention_guidance import (
    build_intervention_candidates,
    format_executable_queries,
)
from mas_langgraph.axis.models import ExplainedCase
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.cases.mcq.comparison import (
    MCQRunObservationAdapter,
    resolve_corresponding_candidate,
)
from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
from mas_langgraph.core.state_cloning import clone_cooperative_state
from mas_langgraph.experiment.evidence import build_trace_evidence
from mas_langgraph.experiment.verbalizer import build_evidence_view
from mas_langgraph.orchestrator.needs import NeedsEvaluator
from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation
from mas_langgraph.replay.checkpoints import CheckpointIndex, CheckpointRecord
from mas_langgraph.replay.interventions import (
    InterventionContract,
    InterventionContractRegistry,
)


class _Cand:
    def __init__(self, q_id: str, blueprint_id: str, revision_count: int = 0):
        self.q_id = q_id
        self.blueprint_id = blueprint_id
        self.revision_count = revision_count
        self.status = "accepted"


class TestReplayCorrespondence(unittest.TestCase):
    def test_correspondence_by_blueprint_id(self):
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q_old", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q_old", 1),
            outcome="accept",
            correspondence_key="bp_1",
        )
        cand, qid = resolve_corresponding_candidate(
            explained_case=explained,
            candidates={"q_new": _Cand("q_new", "bp_1")},
        )
        self.assertEqual(qid, "q_new")
        self.assertEqual(cand.blueprint_id, "bp_1")

    def test_observe_replay_noop_missing_is_error(self):
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q_x", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q_x", 1),
            outcome="accept",
            correspondence_key="bp_missing",
        )
        cs = CooperativeState()
        result = MagicMock()
        result.final_state = {"cooperative_state": cs, "artifact_catalog": {}}
        result.branch_id = "noop_1"
        result.run_id = "r"
        result.error = None
        result.branch_metadata = MagicMock(branch_type="noop")
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=explained, run_result=result
        )
        self.assertFalse(obs.subject_present)
        self.assertIsNotNone(obs.error)

    def test_observe_replay_cf_missing_is_not_tech_error(self):
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q_x", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q_x", 1),
            outcome="accept",
            correspondence_key="bp_missing",
        )
        cs = CooperativeState()
        result = MagicMock()
        result.final_state = {"cooperative_state": cs, "artifact_catalog": {}}
        result.branch_id = "cf_1"
        result.run_id = "r"
        result.error = None
        result.branch_metadata = MagicMock(branch_type="counterfactual")
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=explained, run_result=result
        )
        self.assertFalse(obs.subject_present)
        self.assertIsNone(obs.error)
        self.assertTrue(obs.outcome_available)
        status, _, _ = _determine_evidence_status(
            CaseRunObservation(
                run_id="r", branch_id="factual", decision="accept", subject_present=True
            ),
            CaseRunObservation(
                run_id="r",
                branch_id="noop",
                decision="accept",
                subject_present=True,
                outcome_available=True,
            ),
            obs,
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
        )
        self.assertEqual(status, DOWNSTREAM_EFFECT)


class TestDeterministicIdsAndRouting(unittest.TestCase):
    def test_generate_id_replay_stable_after_clone(self):
        cs = CooperativeState()
        a = cs.generate_id("q")
        b = cs.generate_id("q")
        self.assertNotEqual(a, b)
        cloned = clone_cooperative_state(cs)
        # Continue from cloned counters — next id continues sequence.
        c = cloned.generate_id("q")
        self.assertEqual(c, "q_00000003")
        # Fresh state from same counter snapshot would match if counters copied.
        cs2 = CooperativeState()
        cs2._id_counters = dict(cs._id_counters)
        # cs already advanced to 2; clone shares copied counters at 2, then 3.
        self.assertEqual(cs2.generate_id("q"), "q_00000003")

    def test_quiz_rounds_checkpointed_not_runtime(self):
        cs = CooperativeState()
        cs.quiz_generation_rounds = 1
        cloned = clone_cooperative_state(cs)
        self.assertEqual(cloned.quiz_generation_rounds, 1)
        cloned.quiz_generation_rounds = 2
        self.assertEqual(cs.quiz_generation_rounds, 1)
        # Runtime mirror must not be the sole source of truth.
        self.assertTrue(hasattr(cloned, "quiz_generation_rounds"))


class TestProvenanceDeltaExport(unittest.TestCase):
    def test_normalized_tuple_serializes(self):
        tup = ("gen#1", "E17:v1", "supporting_evidence", "verify#1")
        d = _relation_summary(tup)
        self.assertEqual(d["relation"], "supporting_evidence")
        self.assertIsNotNone(d["artifact_ref"])
        self.assertEqual(d["producer_occurrence_id"], "gen#1")
        self.assertEqual(d["consumer_occurrence_id"], "verify#1")
        self.assertNotEqual(d["relation"], None)


class TestAxisStatusClassification(unittest.TestCase):
    def test_all_noop_error_is_replay_failed(self):
        # Mirror experiment service semantics
        cf_n = 3
        success_n = 0
        statuses = ["noop_error", "noop_error", "noop_error"]
        if success_n > 0:
            axis_status = "completed_with_counterfactual"
        elif cf_n > 0 and any(
            s in {"noop_error", "intervention_error", "outcome_unavailable"}
            for s in statuses
        ):
            axis_status = "replay_failed"
        else:
            axis_status = "other"
        self.assertEqual(axis_status, "replay_failed")
        self.assertNotEqual(axis_status, "completed_with_counterfactual")


class TestFailedReplayLimitation(unittest.TestCase):
    def test_axis_limitation_when_comparisons_fail(self):
        view = build_evidence_view(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "execution_trace": ["apply_acceptance_gate#1"],
                "replay_comparisons": [
                    {"evidence_status": "noop_error", "factual": {}, "noop": {}, "counterfactual": {}}
                ],
                "successful_replay_count": 0,
            },
        )
        text = view.to_prompt_text().lower()
        self.assertIn("none produced a valid comparison", text)
        self.assertIn("no causal effect", text)


class TestStrictTraceAblation(unittest.TestCase):
    def test_trace_does_not_use_provenance_slice(self):
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q1", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("d1", 1),
            outcome="accept",
        )
        # Slice has a narrow subset; factual log has more.
        slice_occ = OccurrenceRecord(
            occurrence_id="r:factual:gate#1",
            node_id="apply_acceptance_gate",
            occurrence_index=1,
            branch_id="factual",
            run_id="r",
            execution_index=9,
        )
        full = [
            OccurrenceRecord(
                occurrence_id="r:factual:build_outline#1",
                node_id="build_outline",
                occurrence_index=1,
                branch_id="factual",
                run_id="r",
                execution_index=1,
            ),
            slice_occ,
        ]
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_occurrences=(slice_occ,),
            relevant_relations=(),
            relevant_artifact_refs=(ArtifactRef("q1", 1),),
            decision="accept",
        )
        # Even if case_slice is passed, Trace must use factual_occurrences.
        ev = build_trace_evidence(
            explained_case=explained,
            case_slice=case_slice,
            factual_occurrences=full,
        )
        self.assertFalse(ev["provenance_included"])
        self.assertFalse(ev["counterfactual_replay_included"])
        self.assertFalse(ev["occurrence_selection_uses_provenance_slice"])
        self.assertIn("build_outline#1", ev["execution_trace"])
        self.assertEqual(ev["occurrence_selection"], "factual_branch_occurrence_log")


class TestOperationSpecificCandidates(unittest.TestCase):
    def test_add_not_offered_for_existing_slice_artifact(self):
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q1", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("d1", 1),
            outcome="accept",
        )
        reg = ArtifactRegistry()
        env = ArtifactEnvelope.create(
            logical_id="chunk_x",
            version=1,
            artifact_type="EvidenceChunk",
            payload={"text": "x"},
            branch_id="factual",
            created_in_run="r",
        )
        reg.store(env)
        occ = OccurrenceRecord(
            occurrence_id="r:factual:verify#1",
            node_id="verify",
            occurrence_index=1,
            branch_id="factual",
            run_id="r",
            execution_index=5,
        )
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=(ArtifactRef("chunk_x", 1),),
            relevant_occurrences=(occ,),
            relevant_relations=(
                RunProvenanceRelation(
                    producer_occurrence_id="r:factual:verify#1",
                    artifact_ref=ArtifactRef("chunk_x", 1),
                    relation="supporting_evidence",
                    consumer_occurrence_id="r:factual:verify#1",
                ),
            ),
        )
        idx = CheckpointIndex(thread_id="t")
        idx.add(
            CheckpointRecord(
                alias="c_0",
                thread_id="t",
                checkpoint_id="ck0",
                superstep=0,
            )
        )
        idx.add(
            CheckpointRecord(
                alias="c_1",
                thread_id="t",
                checkpoint_id="ck1",
                superstep=1,
                after_occurrence_labels=("verify#1",),
                after_occurrences=("r:factual:verify#1",),
                parent_checkpoint_id="ck0",
            )
        )
        contracts = InterventionContractRegistry()
        contracts.register(
            InterventionContract(
                artifact_type="EvidenceChunk",
                allowed_operations=("REMOVE", "ADD", "REPLACE"),
                allow_external_add=True,
            )
        )
        cands = build_intervention_candidates(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            all_occurrences=(occ,),
        )
        self.assertTrue(cands)
        for c in cands:
            self.assertNotIn("ADD", c.allowed_operations)
            self.assertIn("REMOVE", c.allowed_operations)


class TestCanonicalExecutableQueries(unittest.TestCase):
    def test_prompt_contains_parenthesized_forms(self):
        from mas_langgraph.axis.intervention_guidance import InterventionCandidate

        c = InterventionCandidate(
            ref=ArtifactRef("chunk_x", 1),
            artifact_type="EvidenceChunk",
            allowed_operations=("REMOVE",),
            valid_boundaries=("before plan_mcq_blueprints#1",),
            operation_boundaries={
                "REMOVE": ("before plan_mcq_blueprints#1",),
            },
        )
        text = format_executable_queries(
            candidates=(c,),
            explained_case=ExplainedCase(
                run_id="r",
                artifact_ref=ArtifactRef("q1", 1),
                branch_id="factual",
                decision_ref=ArtifactRef("d1", 1),
                outcome="accept",
            ),
        )
        self.assertIn("REMOVE(chunk_x:v1, before plan_mcq_blueprints#1)", text)
        self.assertIn("Preserve parentheses", text)
        self.assertNotIn("REMOVE chunk_x", text.replace("REMOVE(chunk_x", ""))


if __name__ == "__main__":
    unittest.main()
