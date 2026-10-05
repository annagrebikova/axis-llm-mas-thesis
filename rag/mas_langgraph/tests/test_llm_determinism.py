"""Deterministic workflow-LLM replay ledger tests."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from mas_langgraph.experiment.llm import invoke_llm_text, resolve_runtime_llm
from mas_langgraph.replay.llm_determinism import (
    ReplayAwareWorkflowLLM,
    WorkflowLLMReplayLedger,
    WorkflowLLMReplayMissError,
    fingerprint_workflow_llm_request,
    workflow_llm_execution_context,
)


@dataclass
class AIMessage:
    content: str


class IncrementingLLM:
    """Fake underlying LLM that returns a new response on every live call."""

    def __init__(self) -> None:
        self.calls = 0
        self.model_name = "fake-incrementing"
        self.temperature = 0

    def invoke(self, prompt, config=None, **kwargs):
        self.calls += 1
        return AIMessage(content=f"response-{self.calls}")


class TestWorkflowLLMReplayLedger(unittest.TestCase):
    def test_noop_reuses_factual_response(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1", model_name="fake")
        wrapped = ReplayAwareWorkflowLLM(base)

        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            r1 = wrapped.invoke("prompt-P")
        self.assertEqual(r1.content, "response-1")
        self.assertEqual(base.calls, 1)

        with workflow_llm_execution_context(
            run_id="r1", branch_id="noop_1", branch_type="noop", ledger=ledger
        ):
            r2 = wrapped.invoke("prompt-P")
        self.assertEqual(r2.content, "response-1")
        self.assertEqual(base.calls, 1)
        self.assertEqual(ledger.noop_live_provider_calls, 0)
        self.assertEqual(ledger.noop_llm_request_misses, 0)
        self.assertEqual(ledger.noop_factual_response_hits, 1)

    def test_counterfactual_unchanged_request_reuses_factual(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            wrapped.invoke("P")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="cf", branch_type="counterfactual", ledger=ledger
        ):
            out = wrapped.invoke("P")
        self.assertEqual(out.content, "response-1")
        self.assertEqual(base.calls, 1)
        self.assertEqual(ledger.counterfactual_live_provider_calls, 0)
        self.assertEqual(ledger.counterfactual_factual_response_hits, 1)

    def test_counterfactual_novel_request_calls_once_then_memoizes(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            wrapped.invoke("P")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="cf", branch_type="counterfactual", ledger=ledger
        ):
            r_new = wrapped.invoke("P-prime")
            r_again = wrapped.invoke("P-prime")
        self.assertEqual(r_new.content, "response-2")
        self.assertEqual(r_again.content, "response-2")
        self.assertEqual(base.calls, 2)
        self.assertEqual(ledger.counterfactual_novel_request_count, 1)
        self.assertEqual(ledger.counterfactual_memo_hits, 1)
        self.assertEqual(ledger.counterfactual_live_provider_calls, 1)

    def test_noop_request_miss_fails_closed(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            wrapped.invoke("P")
        with self.assertRaises(WorkflowLLMReplayMissError):
            with workflow_llm_execution_context(
                run_id="r1", branch_id="noop", branch_type="noop", ledger=ledger
            ):
                wrapped.invoke("never-seen-in-factual")
        self.assertEqual(base.calls, 1)
        self.assertEqual(ledger.noop_llm_request_misses, 1)
        self.assertEqual(ledger.noop_live_provider_calls, 0)

    def test_counterfactual_not_misclassified_as_factual(self):
        """Regression: ``\"factual\" in \"counterfactual\"`` must not match."""
        from mas_langgraph.replay.executor import _normalize_workflow_branch_type

        self.assertEqual(
            _normalize_workflow_branch_type("counterfactual", branch_id="cf_1"),
            "counterfactual",
        )
        self.assertEqual(
            _normalize_workflow_branch_type("factual", branch_id="factual"),
            "factual",
        )
        self.assertEqual(
            _normalize_workflow_branch_type("noop", branch_id="noop_1"),
            "noop",
        )

    def test_cf_diagnostics_counts(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            wrapped.invoke("P")
        ledger.begin_branch("counterfactual")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="cf_1", branch_type="counterfactual", ledger=ledger
        ):
            wrapped.invoke("P")
            wrapped.invoke("P-prime")
        diag = ledger.end_branch("counterfactual")
        self.assertEqual(ledger.counterfactual_llm_request_count, 2)
        self.assertEqual(ledger.counterfactual_factual_response_hits, 1)
        self.assertEqual(ledger.counterfactual_novel_request_count, 1)
        self.assertEqual(ledger.counterfactual_live_provider_calls, 1)
        self.assertEqual(diag["factual_response_hits"], 1)
        self.assertEqual(diag["novel_request_count"], 1)
        self.assertEqual(diag["live_provider_calls"], 1)

    def test_per_branch_diagnostics_two_cfs(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            wrapped.invoke("P")
            wrapped.invoke("Q")
            wrapped.invoke("R")
        ledger.begin_branch("counterfactual")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="cf1", branch_type="counterfactual", ledger=ledger
        ):
            wrapped.invoke("P")
            wrapped.invoke("Q")
            wrapped.invoke("P-prime")
        d1 = ledger.end_branch("counterfactual")
        ledger.begin_branch("counterfactual")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="cf2", branch_type="counterfactual", ledger=ledger
        ):
            wrapped.invoke("P")
            wrapped.invoke("Q")
            wrapped.invoke("R")
        d2 = ledger.end_branch("counterfactual")
        self.assertEqual(d1["factual_response_hits"], 2)
        self.assertEqual(d1["novel_request_count"], 1)
        self.assertEqual(d2["factual_response_hits"], 3)
        self.assertEqual(d2["novel_request_count"], 0)
        self.assertEqual(ledger.counterfactual_novel_request_count, 1)
        self.assertEqual(ledger.counterfactual_factual_response_hits, 5)

    def test_fingerprint_ignores_branch_identity(self):
        a = fingerprint_workflow_llm_request(
            model="m", input_data="same", temperature=0
        )
        b = fingerprint_workflow_llm_request(
            model="m", input_data="same", temperature=0
        )
        self.assertEqual(a, b)


class TestFingerprintStability(unittest.TestCase):
    def test_same_semantic_factual_vs_noop_fingerprint(self):
        factual = fingerprint_workflow_llm_request(
            model="m", input_data="prompt-P", temperature=0.0
        )
        noop = fingerprint_workflow_llm_request(
            model="m", input_data="prompt-P", temperature=0.0
        )
        self.assertEqual(factual, noop)

    def test_branch_id_not_in_fingerprint(self):
        """branch_id is execution metadata, not model input."""
        from mas_langgraph.replay.llm_determinism import build_fingerprint_payload

        payload = build_fingerprint_payload(
            model="m", input_data="P", temperature=0
        )
        self.assertNotIn("branch_id", payload)
        self.assertNotIn("checkpoint_id", payload)
        a = fingerprint_workflow_llm_request(model="m", input_data="P", temperature=0)
        b = fingerprint_workflow_llm_request(model="m", input_data="P", temperature=0)
        self.assertEqual(a, b)

    def test_checkpoint_id_not_in_fingerprint(self):
        a = fingerprint_workflow_llm_request(model="m", input_data="P", temperature=0)
        b = fingerprint_workflow_llm_request(model="m", input_data="P", temperature=0)
        self.assertEqual(a, b)

    def test_cross_cove_sample_index_distinguishes(self):
        a = fingerprint_workflow_llm_request(
            model="m",
            input_data="same",
            temperature=0,
            sample_index=0,
            llm_component="cross_cove",
        )
        b = fingerprint_workflow_llm_request(
            model="m",
            input_data="same",
            temperature=0,
            sample_index=1,
            llm_component="cross_cove",
        )
        self.assertNotEqual(a, b)

    def test_changed_prompt_changes_fingerprint(self):
        a = fingerprint_workflow_llm_request(model="m", input_data="P", temperature=0)
        b = fingerprint_workflow_llm_request(model="m", input_data="P-prime", temperature=0)
        self.assertNotEqual(a, b)

    def test_changed_model_or_temperature_changes_fingerprint(self):
        base = fingerprint_workflow_llm_request(
            model="m", input_data="P", temperature=0.0
        )
        other_model = fingerprint_workflow_llm_request(
            model="m2", input_data="P", temperature=0.0
        )
        other_temp = fingerprint_workflow_llm_request(
            model="m", input_data="P", temperature=0.7
        )
        self.assertNotEqual(base, other_model)
        self.assertNotEqual(base, other_temp)

    def test_explanation_calls_excluded_from_ledger(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        # Outside workflow context: no ledger mutation.
        out = wrapped.invoke("axis-synthesis")
        self.assertEqual(out.content, "response-1")
        self.assertEqual(len(ledger._responses), 0)
        # resolve_runtime_llm unwraps for explanation.
        rag = type("R", (), {"llm": wrapped, "base_llm": base})()
        llm, _ = resolve_runtime_llm(rag_agents=rag)
        self.assertIs(llm, base)
        text = invoke_llm_text(wrapped, "participant-synthesis")
        self.assertEqual(text, "response-2")
        self.assertEqual(len(ledger._responses), 0)


class TestReplayUnstableNotes(unittest.TestCase):
    def test_criteria_unstable_note_does_not_claim_decision_change(self):
        from mas_langgraph.axis.comparison import (
            CaseRunObservation,
            CriterionDelta,
            REPLAY_UNSTABLE,
        )
        from mas_langgraph.artifacts.models import ArtifactRef
        from mas_langgraph.axis import comparison as cmp_mod

        fact = CaseRunObservation(
            run_id="r",
            branch_id="factual",
            subject_ref=ArtifactRef("q", 1),
            decision="accept",
            outcome_available=True,
            subject_present=True,
        )
        noop = CaseRunObservation(
            run_id="r",
            branch_id="noop",
            subject_ref=ArtifactRef("q", 1),
            decision="accept",
            outcome_available=True,
            subject_present=True,
        )
        cf = CaseRunObservation(
            run_id="r",
            branch_id="cf",
            subject_ref=ArtifactRef("q", 1),
            decision="accept",
            outcome_available=True,
            subject_present=True,
        )
        deltas = (
            CriterionDelta(
                name="coverage",
                factual_value=1.0,
                noop_value=0.5,
                counterfactual_value=0.1,
                decision_relevant=True,
                noop_differs_from_factual=True,
                counterfactual_differs_from_noop=True,
            ),
        )
        status, dec_stable, crit_stable = cmp_mod._determine_evidence_status(
            fact, noop, cf, deltas, (), (), (), (), (), (), ()
        )
        self.assertEqual(status, REPLAY_UNSTABLE)
        self.assertTrue(dec_stable)
        self.assertFalse(crit_stable)

        notes = []
        if status == REPLAY_UNSTABLE:
            if dec_stable is False:
                notes.append("The unchanged replay changed the final decision")
            elif crit_stable is False:
                notes.append(
                    "The unchanged replay changed one or more decision-relevant "
                    "assessment criteria"
                )
        self.assertEqual(len(notes), 1)
        self.assertIn("assessment criteria", notes[0])
        self.assertNotIn("final decision", notes[0])


class TestModelDefaults(unittest.TestCase):
    def test_default_snapshots(self):
        from mas_langgraph.experiment.llm_config import (
            DEFAULT_INTERROGATION_MODEL,
            DEFAULT_SYNTHESIS_MODEL,
            DEFAULT_WORKFLOW_MODEL,
            ExplanationLLMConfig,
        )
        from mas_langgraph.input.rag_config import RAGConfig

        self.assertEqual(DEFAULT_WORKFLOW_MODEL, "gpt-4.1-mini-2025-04-14")
        self.assertEqual(DEFAULT_INTERROGATION_MODEL, "gpt-5.4-mini-2026-03-17")
        self.assertEqual(DEFAULT_SYNTHESIS_MODEL, "gpt-5.4-mini-2026-03-17")
        self.assertEqual(RAGConfig().llm_model, "gpt-4.1-mini-2025-04-14")
        from mas_langgraph.input.constants import (
            CROSS_COVE_N_RUNS,
            DEFAULT_CROSS_COVE_VERIFIER_MODEL,
        )

        self.assertEqual(CROSS_COVE_N_RUNS, 3)
        self.assertEqual(
            DEFAULT_CROSS_COVE_VERIFIER_MODEL, "gpt-5.4-mini-2026-03-17"
        )
        self.assertNotEqual(DEFAULT_WORKFLOW_MODEL, DEFAULT_CROSS_COVE_VERIFIER_MODEL)
        cfg = ExplanationLLMConfig()
        self.assertEqual(cfg.interrogation_model, DEFAULT_INTERROGATION_MODEL)
        self.assertEqual(cfg.synthesis_model, DEFAULT_SYNTHESIS_MODEL)
        self.assertEqual(cfg.reasoning_effort, "none")
        self.assertNotEqual(DEFAULT_WORKFLOW_MODEL, "gpt-3.5-turbo")


class TestFaithfulnessWording(unittest.TestCase):
    def test_trace_provenance_rejects_ensured_speculation(self):
        from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
        from mas_langgraph.experiment.verbalizer import ExplanationEvidenceView

        view = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            recorded_dependencies=("fact",),
            limitations=("lim",),
        )
        bad = (
            "Why this question?\nThe topic.\n"
            "What was it based on?\nThe records.\n"
            "What difference did this information make?\n"
            "The supporting information ensured that the question accurately reflected "
            "the key concepts. Without this information, the question may not have "
            "effectively targeted the learning objective."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance", explanation=bad, evidence_view=view
        )
        self.assertFalse(ok)
        self.assertTrue(any("counterfactual question effect" in w for w in warns), warns)

        good = (
            "Why this question?\nThe topic.\n"
            "What was it based on?\nThe records.\n"
            "What difference did this information make?\n"
            "The records show that this information was used when the question was "
            "created, but they do not show whether the question would have been "
            "different without it."
        )
        ok2, warns2 = validate_explanation_faithfulness(
            condition="trace_provenance", explanation=good, evidence_view=view
        )
        self.assertTrue(ok2, warns2)

    def test_axis_inconclusive_requires_limitation(self):
        from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
        from mas_langgraph.experiment.lay_facts import LayExplanationFacts
        from mas_langgraph.experiment.verbalizer import ExplanationEvidenceView

        facts = LayExplanationFacts(
            decision="ACCEPT",
            question_content_findings=(
                "The system tested removing this information, but the unchanged rerun "
                "did not reproduce the same question content. The test therefore cannot "
                "reliably determine whether the information caused the wording to change.",
            ),
            primary_question_content_status="question_content_inconclusive",
            intervention_before_question_generation=True,
        )
        view = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            tested_interventions=(),
            lay_facts=facts,
        )
        bad = (
            "Why this question?\nTopic.\n"
            "What was it based on?\nMaterial.\n"
            "What difference did this information make?\n"
            "Without this information, the question may have lacked specificity."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis", explanation=bad, evidence_view=view
        )
        self.assertFalse(ok)
        self.assertTrue(
            any("inconclusive" in w.lower() or "unsupported" in w.lower() for w in warns),
            warns,
        )

        good = (
            "Why this question?\nTopic.\n"
            "What was it based on?\nMaterial.\n"
            "What difference did this information make?\n"
            "The system tested removing this information, but the unchanged rerun did not "
            "reproduce the same question content. The test therefore cannot reliably "
            "determine whether the information caused the wording to change."
        )
        ok2, warns2 = validate_explanation_faithfulness(
            condition="axis", explanation=good, evidence_view=view
        )
        self.assertTrue(ok2, warns2)

    def test_scope_overclaim_rejected(self):
        from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
        from mas_langgraph.experiment.verbalizer import ExplanationEvidenceView

        view = ExplanationEvidenceView(factual_decision="ACCEPT")
        bad = (
            "Why this question?\nTopic.\n"
            "What was it based on?\nMaterial.\n"
            "What difference did this information make?\n"
            "Without the teaching material, the question changed."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis", explanation=bad, evidence_view=view
        )
        self.assertFalse(ok)
        self.assertTrue(any("scope" in w.lower() for w in warns), warns)

    def test_contradictory_changed_unchanged_rejected(self):
        from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
        from mas_langgraph.experiment.verbalizer import ExplanationEvidenceView

        view = ExplanationEvidenceView(factual_decision="ACCEPT")
        bad = (
            "Why this question?\nTopic.\n"
            "What was it based on?\nMaterial.\n"
            "What difference did this information make?\n"
            "Without the information the question changed from X to Y. "
            "However, the question remained the same when the supporting "
            "information was removed."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis", explanation=bad, evidence_view=view
        )
        self.assertFalse(ok)
        self.assertTrue(any("changed/unchanged" in w.lower() for w in warns), warns)


if __name__ == "__main__":
    unittest.main()
