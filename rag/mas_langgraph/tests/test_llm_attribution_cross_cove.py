"""LLM call attribution and Cross-CoVe sample-identity replay tests."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import List

from mas_langgraph.domains.quiz.cross_cove import (
    majority_vote,
    run_cross_cove_for_candidate,
)
from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ
from mas_langgraph.experiment.llm import invoke_llm_text, resolve_runtime_llm
from mas_langgraph.replay.llm_determinism import (
    ReplayAwareWorkflowLLM,
    WorkflowLLMReplayLedger,
    fingerprint_workflow_llm_request,
    summarize_workflow_llm_calls,
    workflow_llm_context,
    workflow_llm_execution_context,
)


@dataclass
class AIMessage:
    content: str


class IncrementingLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.model_name = "fake-incrementing"
        self.temperature = 0

    def invoke(self, prompt, config=None, **kwargs):
        self.calls += 1
        return AIMessage(content=f"response-{self.calls}")


class SequenceLLM:
    """Return successive scripted contents; each live call advances."""

    def __init__(self, contents: List[str]) -> None:
        self.contents = list(contents)
        self.calls = 0
        self.model_name = "fake-sequence"
        self.temperature = 0

    def invoke(self, prompt, config=None, **kwargs):
        self.calls += 1
        idx = min(self.calls - 1, len(self.contents) - 1)
        return AIMessage(content=self.contents[idx])


def _mcq() -> CandidateMCQ:
    return CandidateMCQ(
        q_id="q_00000001",
        blueprint_id="bp_1",
        stem="What is 2+2?",
        options=[
            AnswerOption(option_id="A", text="3", is_correct=False),
            AnswerOption(option_id="B", text="4", is_correct=True),
            AnswerOption(option_id="C", text="5", is_correct=False),
            AnswerOption(option_id="D", text="22", is_correct=False),
        ],
        correct_option_id="B",
        answer_explanation="Basic arithmetic.",
    )


class TestLLMAttribution(unittest.TestCase):
    def test_native_mcq_generation_attribution(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            with workflow_llm_context(layer="native_mcq", component="mcq_generation"):
                wrapped.invoke("generate-mcq")
        entry = ledger.call_log[0]
        self.assertEqual(entry["llm_layer"], "native_mcq")
        self.assertEqual(entry["llm_component"], "mcq_generation")
        self.assertEqual(entry["branch_type"], "factual")
        self.assertTrue(entry["provider_called"])

    def test_rag_atomization_attribution(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            with workflow_llm_context(layer="rag_backend", component="atomization"):
                wrapped.invoke("extract-atoms")
        entry = ledger.call_log[0]
        self.assertEqual(entry["llm_layer"], "rag_backend")
        self.assertEqual(entry["llm_component"], "atomization")

    def test_branch_attribution_separates_counts(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            with workflow_llm_context(layer="native_mcq", component="cross_cove", sample_index=0):
                wrapped.invoke("same-prompt")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="noop", branch_type="noop", ledger=ledger
        ):
            with workflow_llm_context(layer="native_mcq", component="cross_cove", sample_index=0):
                wrapped.invoke("same-prompt")
        summary = summarize_workflow_llm_calls(ledger)
        self.assertEqual(
            summary["factual"]["native_mcq"]["cross_cove"],
            {"logical": 1, "provider": 1, "memo_hits": 0},
        )
        self.assertEqual(
            summary["noop"]["native_mcq"]["cross_cove"],
            {"logical": 1, "provider": 0, "memo_hits": 1},
        )


class TestCrossCoVeSampleIdentity(unittest.TestCase):
    def test_factual_independence_three_provider_calls(self):
        answers = [
            json.dumps({"selected_option_id": "A", "reason": "r1"}),
            json.dumps({"selected_option_id": "B", "reason": "r2"}),
            json.dumps({"selected_option_id": "A", "reason": "r3"}),
        ]
        base = SequenceLLM(answers)
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            report = run_cross_cove_for_candidate(_mcq(), verifier_llm=wrapped, n_runs=3)

        self.assertEqual(base.calls, 3)
        self.assertEqual(report.verifier_answers, ["A", "B", "A"])
        self.assertEqual(report.majority_answer, "A")
        cove = [
            e
            for e in ledger.call_log
            if e["llm_component"] == "cross_cove" and e["branch_type"] == "factual"
        ]
        self.assertEqual(len(cove), 3)
        self.assertEqual(sum(1 for e in cove if e["provider_called"]), 3)
        self.assertEqual(sum(1 for e in cove if e["memo_hit"]), 0)
        samples = [e["sample_index"] for e in cove]
        self.assertEqual(samples, [0, 1, 2])
        fps = {e["request_fingerprint"] for e in cove}
        self.assertEqual(len(fps), 3)

    def test_majority_not_memo_collapse(self):
        maj, conf = majority_vote(["A", "B", "A"])
        self.assertEqual(maj, "A")
        self.assertAlmostEqual(conf, 2 / 3)

    def test_noop_replays_each_sample(self):
        answers = [
            json.dumps({"selected_option_id": "A", "reason": "r1"}),
            json.dumps({"selected_option_id": "B", "reason": "r2"}),
            json.dumps({"selected_option_id": "A", "reason": "r3"}),
        ]
        base = SequenceLLM(answers)
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            factual = run_cross_cove_for_candidate(_mcq(), verifier_llm=wrapped, n_runs=3)
        self.assertEqual(factual.verifier_answers, ["A", "B", "A"])
        self.assertEqual(base.calls, 3)

        with workflow_llm_execution_context(
            run_id="r1", branch_id="noop_1", branch_type="noop", ledger=ledger
        ):
            noop = run_cross_cove_for_candidate(_mcq(), verifier_llm=wrapped, n_runs=3)

        self.assertEqual(base.calls, 3)  # no new provider calls
        self.assertEqual(noop.verifier_answers, ["A", "B", "A"])
        self.assertEqual(noop.majority_answer, "A")
        cove = [
            e
            for e in ledger.call_log
            if e["llm_component"] == "cross_cove" and e["branch_type"] == "noop"
        ]
        self.assertEqual(len(cove), 3)
        self.assertEqual(sum(1 for e in cove if e["provider_called"]), 0)
        self.assertEqual(sum(1 for e in cove if e["memo_hit"]), 3)

    def test_counterfactual_unchanged_reuses_samples(self):
        answers = [
            json.dumps({"selected_option_id": "A", "reason": "r1"}),
            json.dumps({"selected_option_id": "B", "reason": "r2"}),
            json.dumps({"selected_option_id": "A", "reason": "r3"}),
        ]
        base = SequenceLLM(answers)
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            run_cross_cove_for_candidate(_mcq(), verifier_llm=wrapped, n_runs=3)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="cf", branch_type="counterfactual", ledger=ledger
        ):
            cf = run_cross_cove_for_candidate(_mcq(), verifier_llm=wrapped, n_runs=3)
        self.assertEqual(base.calls, 3)
        self.assertEqual(cf.verifier_answers, ["A", "B", "A"])
        self.assertEqual(ledger.counterfactual_live_provider_calls, 0)
        self.assertEqual(ledger.counterfactual_factual_response_hits, 3)

    def test_counterfactual_changed_prompt_calls_provider(self):
        answers = [
            json.dumps({"selected_option_id": "A", "reason": "r1"}),
            json.dumps({"selected_option_id": "B", "reason": "r2"}),
            json.dumps({"selected_option_id": "A", "reason": "r3"}),
            json.dumps({"selected_option_id": "C", "reason": "cf0"}),
            json.dumps({"selected_option_id": "C", "reason": "cf1"}),
            json.dumps({"selected_option_id": "C", "reason": "cf2"}),
        ]
        base = SequenceLLM(answers)
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            run_cross_cove_for_candidate(_mcq(), verifier_llm=wrapped, n_runs=3)

        changed = _mcq()
        changed.stem = "What is 3+3?"
        with workflow_llm_execution_context(
            run_id="r1", branch_id="cf", branch_type="counterfactual", ledger=ledger
        ):
            cf = run_cross_cove_for_candidate(changed, verifier_llm=wrapped, n_runs=3)
        self.assertEqual(base.calls, 6)
        self.assertEqual(cf.verifier_answers, ["C", "C", "C"])
        self.assertEqual(ledger.counterfactual_live_provider_calls, 3)

    def test_ordinary_memoization_unchanged(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            with workflow_llm_context(layer="native_mcq", component="mcq_generation"):
                a = wrapped.invoke("identical-prompt")
                b = wrapped.invoke("identical-prompt")
        self.assertEqual(a.content, b.content)
        self.assertEqual(base.calls, 1)
        self.assertEqual(ledger.factual_llm_memo_hits, 1)

    def test_sample_fingerprint_differs_for_same_prompt(self):
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
        c = fingerprint_workflow_llm_request(
            model="m", input_data="same", temperature=0
        )
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)

    def test_explanation_layer_excluded_from_ledger(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        # Outside workflow context: no ledger mutation.
        out = wrapped.invoke("axis-synthesis")
        self.assertEqual(out.content, "response-1")
        self.assertEqual(len(ledger.call_log), 0)
        rag = type("R", (), {"llm": wrapped, "base_llm": base})()
        llm, _ = resolve_runtime_llm(rag_agents=rag)
        self.assertIs(llm, base)
        text = invoke_llm_text(wrapped, "participant-synthesis")
        self.assertEqual(text, "response-2")
        self.assertEqual(len(ledger.call_log), 0)
        self.assertEqual(summarize_workflow_llm_calls(ledger), {})

    def test_diagnostics_include_workflow_llm_calls(self):
        base = IncrementingLLM()
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        wrapped = ReplayAwareWorkflowLLM(base)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            with workflow_llm_context(layer="rag_backend", component="outline"):
                wrapped.invoke("outline")
        diag = ledger.diagnostics()
        self.assertIn("workflow_llm_calls", diag)
        self.assertEqual(
            diag["workflow_llm_calls"]["factual"]["rag_backend"]["outline"]["provider"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
