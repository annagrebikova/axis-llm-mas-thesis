"""Focused tests for autonomous AXIS interrogation reaching counterfactual replays."""

from __future__ import annotations

import unittest
from typing import List
from unittest.mock import MagicMock, patch

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.explainer import AxisExplanationController
from mas_langgraph.axis.intervention_guidance import (
    InterventionCandidate,
    build_intervention_candidates,
    format_intervenable_handoff,
)
from mas_langgraph.axis.comparison import (
    CaseRunObservation,
    NO_DEMONSTRATED_EFFECT,
    QuerySignature,
    ReplayComparison,
)
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.cases.mcq.case_definition import MCQCaseDefinition
from mas_langgraph.replay.interventions import (
    InterventionContract,
    InterventionContractRegistry,
)
from mas_langgraph.tests.test_prompt14_loop import _build_fixture, _make_env


def _e17_candidate() -> InterventionCandidate:
    return InterventionCandidate(
        ref=ArtifactRef("E17", 1),
        artifact_type="EvidenceChunk",
        allowed_operations=("REMOVE", "ADD", "REPLACE"),
        used_by=("verify#1",),
        valid_boundaries=("c_8", "before gen#1"),
        priority=1,
    )


class TestAttemptVsSuccessfulBudget(unittest.TestCase):
    def test_invalid_does_not_consume_successful_rounds(self):
        """Test A: invalid / contract-invalid leave room for valid REMOVE."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        queries = [
            "INVALID NONSENSE",
            "REMOVE(decision:Q12:v2, c_8)",  # no contract for SupervisorDecision
            "WHAT(E17:v1)",  # WHAT is disabled; does not consume a successful round
            "REMOVE(E17:v1, c_8)",
            "DONE",
        ]
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            max_rounds=3,
            max_attempts=10,
            compatibility_mode=True,
        )
        with patch(
            "mas_langgraph.axis.explainer.build_intervention_candidates",
            return_value=(_e17_candidate(),),
        ):
            result = controller.run("Why?", scripted_queries=queries)
        self.assertGreaterEqual(result.successful_round_count, 1)
        self.assertEqual(len(result.factual_records), 0)
        self.assertGreaterEqual(len(result.intervention_records), 1)
        self.assertGreaterEqual(result.attempt_count, 4)
        self.assertLessEqual(result.successful_round_count, 3)

    def test_attempt_budget_bounded(self):
        """Test B: repeated invalid queries terminate at max_attempts."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            max_rounds=5,
            max_attempts=3,
            compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=["BAD1", "BAD2", "BAD3", "WHAT(E17:v1)", "DONE"],
        )
        self.assertEqual(result.termination_reason, "attempt_budget_exhausted")
        self.assertEqual(result.attempt_count, 3)
        self.assertEqual(result.successful_round_count, 0)
        self.assertEqual(len(result.factual_records), 0)


class TestFailedQueryFeedback(unittest.TestCase):
    def test_feedback_in_history(self):
        """Test C: controlled validation reason reaches interrogation history."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            max_rounds=3,
            compatibility_mode=False,
        )
        with patch(
            "mas_langgraph.axis.explainer.build_intervention_candidates",
            return_value=(_e17_candidate(),),
        ):
            result = controller.run(
                "Why?",
                scripted_queries=[
                    "REMOVE(decision:Q12:v2, c_8)",
                    "DONE",
                ],
            )
        joined = "\n".join(result.intermediate_texts).lower()
        self.assertTrue(
            "could not be executed" in joined
            or "contract" in joined
            or "previous query" in joined
        )
        statuses = [a.validation_status for a in result.interrogation_attempts]
        self.assertTrue(
            "contract_error" in statuses or "resolution_error" in statuses
        )


class TestDynamicCandidates(unittest.TestCase):
    def test_excludes_non_intervenable(self):
        """Test D: CrossCoVeReport excluded when no contract."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        reg.store(_make_env("ccr1", 1, "CrossCoVeReport"))
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=(
                ArtifactRef("E17", 1),
                ArtifactRef("Q12", 2),
                ArtifactRef("ccr1", 1),
                ArtifactRef("decision:Q12", 2),
            ),
            relevant_occurrences=tuple(occs),
            relevant_relations=tuple(rels),
        )
        contracts.register(
            InterventionContract(
                artifact_type="CandidateMCQ",
                allowed_operations=("REMOVE",),
            )
        )
        cands = build_intervention_candidates(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            all_occurrences=occs,
            priority_fn=MCQCaseDefinition().intervention_artifact_priority,
        )
        types = {c.artifact_type for c in cands}
        self.assertIn("EvidenceChunk", types)
        self.assertIn("CandidateMCQ", types)
        self.assertNotIn("CrossCoVeReport", types)
        self.assertNotIn("SupervisorDecision", types)


class TestWhatHandoff(unittest.TestCase):
    def test_handoff_text(self):
        """Test E: WHAT intervenable artifact produces explicit handoff text."""
        text = format_intervenable_handoff(
            target_ref=ArtifactRef("chunk_x", 1),
            allowed_operations=("REMOVE", "ADD", "REPLACE"),
            valid_boundaries=("before verify#1",),
            has_successful_intervention=False,
        )
        self.assertIn("intervenable", text.lower())
        self.assertIn("chunk_x", text)
        self.assertIn("REMOVE", text)
        self.assertIn("before verify#1", text)
        self.assertIn("No successful counterfactual", text)

    def test_handoff_in_controller_after_what(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        seen: List[str] = []
        inter_calls = {"n": 0}

        def llm2(prompt: str) -> str:
            if "Return exactly one structured query" in prompt or "Reply with one AXIS query" in prompt or "structured query" in prompt:
                inter_calls["n"] += 1
                seen.append(prompt)
                if inter_calls["n"] == 1:
                    return "WHAT(E17:v1)"
                if inter_calls["n"] == 2:
                    return "REMOVE(E17:v1, c_8)"
                return "DONE"
            if "Synthesize" in prompt or "intermediate" in prompt.lower():
                return "ok"
            return "Final explanation."

        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            llm_generate=llm2,
            max_rounds=5,
            max_attempts=10,
            compatibility_mode=True,
        )
        with patch(
            "mas_langgraph.axis.explainer.build_intervention_candidates",
            return_value=(_e17_candidate(),),
        ):
            result = controller.run("Why?")
        self.assertTrue(
            any(a.parsed_operation == "WHAT" and a.validation_status == "vocabulary_error"
                for a in result.interrogation_attempts)
        )
        self.assertGreaterEqual(len(result.intervention_records), 1)


class TestDoneAndLastChanceNudges(unittest.TestCase):
    def test_done_terminates_without_nudge(self):
        """DONE with remaining candidates terminates; no forced intervention."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        calls = {"n": 0}

        def llm(prompt: str) -> str:
            if "structured query" in prompt or "Return exactly one" in prompt:
                calls["n"] += 1
                if calls["n"] == 1:
                    return "DONE"
                return "REMOVE(E17:v1, c_8)"
            if "Synthesize" in prompt:
                return "ok"
            return "Final."

        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            llm_generate=llm,
            max_rounds=5,
            compatibility_mode=True,
        )
        with patch(
            "mas_langgraph.axis.explainer.build_intervention_candidates",
            return_value=(_e17_candidate(),),
        ):
            result = controller.run("Why?")
        self.assertNotIn("counterfactual_nudge_before_done", result.warnings)
        self.assertNotIn("last_chance_intervention_nudge", result.warnings)
        self.assertEqual(result.termination_reason, "done")
        self.assertEqual(len(result.intervention_records), 0)

    def test_max_rounds_no_last_chance_what(self):
        """Successful rounds are interventions; WHAT cannot consume the budget."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        calls = {"n": 0}

        def llm(prompt: str) -> str:
            if "structured query" in prompt or "Return exactly one" in prompt:
                calls["n"] += 1
                if calls["n"] == 1:
                    return "WHAT(E17:v1)"
                return "REMOVE(E17:v1, c_8)"
            if "Synthesize" in prompt:
                return "ok"
            return "Final."

        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            llm_generate=llm,
            max_rounds=1,
            max_attempts=5,
            compatibility_mode=True,
        )
        with patch(
            "mas_langgraph.axis.explainer.build_intervention_candidates",
            return_value=(_e17_candidate(),),
        ):
            result = controller.run("Why?")
        self.assertNotIn("last_chance_intervention_nudge", result.warnings)
        self.assertGreaterEqual(len(result.intervention_records), 1)
        self.assertEqual(result.successful_round_count, 1)

    def test_no_forced_when_empty_candidates(self):
        """Test H: no candidates → factual-only termination allowed."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        empty = InterventionContractRegistry()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=empty,
            compatibility_mode=True,
        )
        result = controller.run("Why?", scripted_queries=["DONE"])
        self.assertEqual(result.valid_intervention_candidate_count, 0)
        self.assertIn(
            result.termination_reason,
            {"done", "no_valid_intervention_available"},
        )


class TestSuccessfulRemoveReplay(unittest.TestCase):
    def test_remove_produces_comparison(self):
        """Test I: REMOVE executes NOOP + CF and yields ReplayComparison."""
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=["REMOVE(E17:v1, c_8)", "DONE"],
        )
        self.assertEqual(len(result.intervention_records), 1)
        self.assertEqual(len(result.replay_comparisons), 1)
        rec = result.intervention_records[0]
        self.assertIsNotNone(rec.noop_result)
        self.assertIsNotNone(rec.counterfactual_result)
        # Fixture uses default observation adapter → noop_error is expected;
        # the important invariant is that both replays actually executed.
        self.assertIsNone(rec.error)

    def test_no_demonstrated_effect_counts(self):
        """Test J: equal decisions still count as causal evidence available."""
        from mas_langgraph.experiment.evidence import build_axis_evidence_summary

        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        obs = CaseRunObservation(
            run_id="run1",
            branch_id="factual",
            decision="accept",
            outcome_available=True,
        )
        cmp = ReplayComparison(
            intervention_signature=QuerySignature(
                operation="REMOVE",
                target_ref=ArtifactRef("E17", 1),
                checkpoint_id="c_8",
            ),
            factual=obs,
            noop=obs,
            counterfactual=obs,
            evidence_status=NO_DEMONSTRATED_EFFECT,
            decision_control_stable=True,
        )

        class _Fake:
            replay_comparisons = [cmp]
            intervention_records = [MagicMock()]
            round_records = []
            rounds = 1
            successful_round_count = 1
            attempt_count = 1
            interrogation_attempts = []
            termination_reason = "done"
            warnings = ()
            final_explanation = ""

        summary = build_axis_evidence_summary(
            explanation_result=_Fake(),
            explained_case=explained,
        )
        self.assertEqual(summary["successful_replay_count"], 1)
        self.assertTrue(summary["axis_causal_evidence_available"])


class TestLegacyInvalidRoundSemantics(unittest.TestCase):
    def test_invalid_increments_attempts_not_rounds(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            max_rounds=2,
            max_attempts=10,
            compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=["INVALID NONSENSE", "DONE"],
        )
        self.assertEqual(result.successful_round_count, 0)
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(result.termination_reason, "done")


if __name__ == "__main__":
    unittest.main()
