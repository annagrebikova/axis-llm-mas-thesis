"""Cross-CoVe uses an independent verifier LLM, including during replay."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import List

from mas_langgraph.domains.quiz.cross_cove import (
    build_cross_cove_verifier_llm,
    llm_model_identity,
    run_cross_cove_for_candidate,
)
from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ
from mas_langgraph.input.orchestrator_config import OrchestratorConfig
from mas_langgraph.orchestrator.langgraph_workflow import node_cross_cove_verify_candidate_mcqs
from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context
from mas_langgraph.replay.llm_determinism import (
    ReplayAwareWorkflowLLM,
    WorkflowLLMReplayLedger,
    fingerprint_workflow_llm_request,
    workflow_llm_execution_context,
)


@dataclass
class AIMessage:
    content: str


class NamedLLM:
    def __init__(self, name: str, contents: List[str]) -> None:
        self.model_name = name
        self.model = name
        self.temperature = 0
        self.contents = list(contents)
        self.calls = 0
        self.prompts: List[str] = []

    def invoke(self, prompt, config=None, **kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        idx = min(self.calls - 1, len(self.contents) - 1)
        return AIMessage(content=self.contents[idx])


def _answers(*labels: str) -> List[str]:
    return [json.dumps({"selected_option_id": lab, "reason": lab}) for lab in labels]


def _mcq(stem: str = "What is 2+2?") -> CandidateMCQ:
    return CandidateMCQ(
        q_id="q_00000001",
        blueprint_id="bp_1",
        stem=stem,
        options=[
            AnswerOption(option_id="A", text="3", is_correct=False),
            AnswerOption(option_id="B", text="4", is_correct=True),
            AnswerOption(option_id="C", text="5", is_correct=False),
            AnswerOption(option_id="D", text="22", is_correct=False),
        ],
        correct_option_id="B",
        answer_explanation="Basic arithmetic.",
    )


class TestSeparateCrossCoVeVerifier(unittest.TestCase):
    def test_generator_and_verifier_are_different_objects(self):
        gen = NamedLLM("gpt-4.1-mini-2025-04-14", _answers("B", "B", "B"))
        ver = NamedLLM("gpt-5.4-mini-2026-03-17", _answers("B", "B", "B"))
        wrapped_gen = ReplayAwareWorkflowLLM(gen)
        wrapped_ver = ReplayAwareWorkflowLLM(ver)
        self.assertIsNot(wrapped_gen, wrapped_ver)
        self.assertNotEqual(llm_model_identity(wrapped_gen), llm_model_identity(wrapped_ver))
        self.assertEqual(llm_model_identity(wrapped_ver), "gpt-5.4-mini-2026-03-17")

    def test_build_reuses_generator_when_models_match(self):
        gen = NamedLLM("gpt-4.1-mini-2025-04-14", _answers("B"))
        llm, model, separate = build_cross_cove_verifier_llm(
            verifier_model="gpt-4.1-mini-2025-04-14",
            generator_llm=gen,
            generator_model="gpt-4.1-mini-2025-04-14",
        )
        self.assertIs(llm, gen)
        self.assertFalse(separate)
        self.assertEqual(model, "gpt-4.1-mini-2025-04-14")

    def test_build_reuses_stub_llm_instead_of_constructing_provider(self):
        gen = NamedLLM("scripted", _answers("B"))
        llm, model, separate = build_cross_cove_verifier_llm(
            verifier_model="gpt-5.4-mini-2026-03-17",
            generator_llm=gen,
            generator_model="scripted",
        )
        self.assertIs(llm, gen)
        self.assertFalse(separate)
        self.assertEqual(model, "scripted")

    def test_report_verifier_model_is_actual_llm_identity(self):
        ver = NamedLLM("gpt-5.4-mini-2026-03-17", _answers("B", "B", "B"))
        wrapped = ReplayAwareWorkflowLLM(ver)
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            report = run_cross_cove_for_candidate(
                _mcq(),
                verifier_llm=wrapped,
                verifier_model_name="should-be-overridden",
                n_runs=3,
            )
        self.assertEqual(report.verifier_model, "gpt-5.4-mini-2026-03-17")
        self.assertEqual(ver.calls, 3)

    def test_sample_index_0_1_2_remain_distinct(self):
        ver = NamedLLM("gpt-5.4-mini-2026-03-17", _answers("A", "B", "A"))
        wrapped = ReplayAwareWorkflowLLM(ver)
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            run_cross_cove_for_candidate(_mcq(), verifier_llm=wrapped, n_runs=3)
        cove = [e for e in ledger.call_log if e["llm_component"] == "cross_cove"]
        self.assertEqual([e["sample_index"] for e in cove], [0, 1, 2])
        self.assertEqual(len({e["request_fingerprint"] for e in cove}), 3)

    def test_fingerprints_include_verifier_model(self):
        same_prompt = "same"
        a = fingerprint_workflow_llm_request(
            model="gpt-4.1-mini-2025-04-14",
            input_data=same_prompt,
            temperature=0,
            sample_index=0,
            llm_component="cross_cove",
        )
        b = fingerprint_workflow_llm_request(
            model="gpt-5.4-mini-2026-03-17",
            input_data=same_prompt,
            temperature=0,
            sample_index=0,
            llm_component="cross_cove",
        )
        self.assertNotEqual(a, b)

    def test_noop_hits_factual_ledger_on_verifier_model(self):
        ver = NamedLLM("gpt-5.4-mini-2026-03-17", _answers("B", "A", "B"))
        gen = NamedLLM("gpt-4.1-mini-2025-04-14", _answers("A", "A", "A"))
        wrapped_ver = ReplayAwareWorkflowLLM(ver)
        wrapped_gen = ReplayAwareWorkflowLLM(gen)
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            factual = run_cross_cove_for_candidate(
                _mcq(), verifier_llm=wrapped_ver, n_runs=3
            )
        self.assertEqual(ver.calls, 3)
        self.assertEqual(gen.calls, 0)
        with workflow_llm_execution_context(
            run_id="r1", branch_id="noop", branch_type="noop", ledger=ledger
        ):
            noop = run_cross_cove_for_candidate(
                _mcq(), verifier_llm=wrapped_ver, n_runs=3
            )
        self.assertEqual(ver.calls, 3)
        self.assertEqual(gen.calls, 0)
        self.assertEqual(noop.verifier_answers, factual.verifier_answers)
        self.assertEqual(ledger.noop_live_provider_calls, 0)
        self.assertEqual(ledger.noop_llm_request_misses, 0)
        self.assertEqual(ledger.noop_factual_response_hits, 3)

    def test_changed_cf_calls_verifier_not_generator(self):
        ver = NamedLLM(
            "gpt-5.4-mini-2026-03-17",
            _answers("B", "B", "B", "C", "C", "C"),
        )
        gen = NamedLLM("gpt-4.1-mini-2025-04-14", _answers("A", "A", "A", "A", "A", "A"))
        wrapped_ver = ReplayAwareWorkflowLLM(ver)
        wrapped_gen = ReplayAwareWorkflowLLM(gen)
        ledger = WorkflowLLMReplayLedger(run_id="r1")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="factual", branch_type="factual", ledger=ledger
        ):
            run_cross_cove_for_candidate(_mcq(), verifier_llm=wrapped_ver, n_runs=3)
        changed = _mcq("What is 3+3?")
        with workflow_llm_execution_context(
            run_id="r1", branch_id="cf", branch_type="counterfactual", ledger=ledger
        ):
            cf = run_cross_cove_for_candidate(changed, verifier_llm=wrapped_ver, n_runs=3)
        self.assertEqual(ver.calls, 6)
        self.assertEqual(gen.calls, 0)
        self.assertEqual(cf.verifier_answers, ["C", "C", "C"])
        self.assertEqual(cf.verifier_model, "gpt-5.4-mini-2026-03-17")
        self.assertEqual(ledger.counterfactual_live_provider_calls, 3)
        self.assertIsNot(wrapped_ver, wrapped_gen)

    def test_node_uses_runtime_verifier_not_generator(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.entities import (
            DocSection,
            EvidenceChunk,
            KnowledgeAtom,
            LearningObjective,
        )
        from mas_langgraph.domains.quiz.blueprint_planner import (
            create_mcq_blueprints_from_coverage,
        )
        from mas_langgraph.domains.quiz.entities import EvidenceCheckReport, SourceRef

        state = CooperativeState()
        state.add_section(
            DocSection(
                section_id="sec_kmeans",
                title="k-Means",
                slide_range=(6, 10),
                importance_weight=0.9,
                content_summary="k-Means clustering.",
            )
        )
        state.add_learning_objective(
            LearningObjective(
                lo_id="lo_kmeans_steps",
                section_id="sec_kmeans",
                objective_text="Explain k-means steps.",
                difficulty_level="intermediate",
            )
        )
        chunk = EvidenceChunk(
            chunk_id="chunk_kmeans",
            content="k-means assigns each point to the nearest centroid.",
            metadata={"page_number": 8, "pdf_id": "141"},
            quality_score=0.9,
        )
        state.add_evidence_chunk(chunk)
        atom = KnowledgeAtom(
            atom_id="atom_kmeans",
            content="k-means alternates assignment and centroid recomputation.",
            atom_type="procedure",
            citations=[{"chunk_id": "chunk_kmeans", "page": 8}],
            section_id="sec_kmeans",
            lo_id="lo_kmeans_steps",
        )
        state.add_knowledge_atom(atom)
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = _mcq("What does k-means assign each point to?")
        mcq.q_id = "q_cross"
        mcq.blueprint_id = blueprint.blueprint_id
        mcq.source_refs = [
            SourceRef(doc_id="141", page=8, chunk_id="chunk_kmeans", atom_id="atom_kmeans")
        ]
        state.store_mcq_blueprints([blueprint])
        state.store_candidate_mcq(mcq)
        state.store_evidence_report(
            EvidenceCheckReport(
                q_id=mcq.q_id,
                supported_by_pdf=True,
                unsupported_claims=[],
                source_refs=mcq.source_refs,
                confidence=0.9,
                stem_supported=True,
                answer_supported=True,
                distractors_valid=True,
                explanation_supported=True,
            )
        )

        gen = NamedLLM("gpt-4.1-mini-2025-04-14", _answers("A", "A", "A"))
        ver = NamedLLM("gpt-5.4-mini-2026-03-17", _answers("B", "B", "B"))
        wrapped_gen = ReplayAwareWorkflowLLM(gen)
        wrapped_ver = ReplayAwareWorkflowLLM(ver)
        rag = type(
            "R",
            (),
            {
                "llm": wrapped_gen,
                "model_name": "gpt-4.1-mini-2025-04-14",
                "llm_model": "gpt-4.1-mini-2025-04-14",
            },
        )()
        cfg = OrchestratorConfig(
            cross_cove_enabled=True,
            cross_cove_n_runs=3,
            cross_cove_verifier_model="gpt-5.4-mini-2026-03-17",
        )
        runtime = workflow_runtime_context(
            orchestrator_config=cfg,
            rag_agents=rag,
            llm=wrapped_gen,
            cross_cove_verifier_llm=wrapped_ver,
        )
        self.assertIsNot(
            runtime.context.llm, runtime.context.cross_cove_verifier_llm
        )
        graph_state = {"cooperative_state": state, "step_logs": [], "error": None}
        update = node_cross_cove_verify_candidate_mcqs(graph_state, runtime)
        report = update["cooperative_state"].cross_cove_reports[mcq.q_id]
        self.assertEqual(ver.calls, 3)
        self.assertEqual(gen.calls, 0)
        self.assertEqual(report.verifier_model, "gpt-5.4-mini-2026-03-17")
        self.assertEqual(report.verifier_answers, ["B", "B", "B"])

    def test_node_refuses_silent_generator_fallback(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState

        cfg = OrchestratorConfig(
            cross_cove_enabled=True,
            cross_cove_n_runs=3,
            cross_cove_verifier_model="gpt-5.4-mini-2026-03-17",
        )
        gen = NamedLLM("gpt-4.1-mini-2025-04-14", _answers("B", "B", "B"))
        rag = type(
            "R",
            (),
            {"llm": ReplayAwareWorkflowLLM(gen), "llm_model": gen.model_name},
        )()
        runtime = workflow_runtime_context(
            orchestrator_config=cfg,
            rag_agents=rag,
            llm=rag.llm,
            cross_cove_verifier_llm=None,
        )
        with self.assertRaises(RuntimeError):
            node_cross_cove_verify_candidate_mcqs(
                {"cooperative_state": CooperativeState(), "step_logs": [], "error": None},
                runtime,
            )


if __name__ == "__main__":
    unittest.main()
