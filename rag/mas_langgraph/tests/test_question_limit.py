"""Offline tests for the per-run final-MCQ cap (generation, repair, replay)."""

from __future__ import annotations

import json
import unittest
from typing import Any

from mas_langgraph.backbone import CooperativeBackbone
from mas_langgraph.batch.generate import profile_with_question_limit
from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.domains.quiz.coverage import select_final_mcqs_for_quiz
from mas_langgraph.domains.quiz.entities import CandidateMCQ, MCQBlueprint, SupervisorDecision
from mas_langgraph.domains.quiz.generation import generate_candidate_from_blueprint
from mas_langgraph.domains.quiz.packaging import package_final_quiz
from mas_langgraph.domains.quiz.question_limit import (
    QUESTION_LIMIT_SEMANTICS,
    accepted_final_question_count,
    accepted_final_question_ids,
    build_generation_limit_report,
    first_generation_payload,
    generation_batch_size,
    generation_payloads_from_parsed,
    remaining_final_question_capacity,
    select_final_mcqs_within_limit,
)
from mas_langgraph.input.constants import QUIZ_QUESTIONS_PER_ROUND
from mas_langgraph.input.orchestrator_config import OrchestratorConfig
from mas_langgraph.input.profile import DEFAULT_PROFILE
from mas_langgraph.orchestrator.langgraph_workflow import (
    node_generate_candidate_mcqs_from_blueprints,
    node_generate_repair_blueprints,
    node_package_final_quiz,
    route_after_coverage,
)
from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context


def _planned_blueprint(i: int, *, status: str = "planned") -> MCQBlueprint:
    return MCQBlueprint(
        blueprint_id=f"bp_{i:08d}",
        section_id=f"sec_{i}",
        lo_id=f"lo_{i}",
        target_atom_ids=[f"atom_{i}"],
        evidence_chunk_ids=[f"chunk_{i}"],
        question_type="basic_definition",
        difficulty="intermediate",
        bloom_level="Remembering",
        generation_goal=f"Cover section {i}",
        status=status,
    )


def _state_with_planned(n: int = 6) -> CooperativeState:
    state = CooperativeState()
    for i in range(1, n + 1):
        state.add_section(
            DocSection(
                section_id=f"sec_{i}",
                title=f"Section {i}",
                slide_range=(i, i),
                importance_weight=1.0,
                content_summary=f"Summary {i}",
            )
        )
        state.add_learning_objective(
            LearningObjective(
                lo_id=f"lo_{i}",
                section_id=f"sec_{i}",
                objective_text=f"Explain section {i}.",
                difficulty_level="intermediate",
            )
        )
        state.add_evidence_chunk(
            EvidenceChunk(
                chunk_id=f"chunk_{i}",
                content=f"Evidence for section {i}.",
                metadata={"section_id": f"sec_{i}", "page_number": i},
                quality_score=0.9,
            )
        )
        state.add_knowledge_atom(
            KnowledgeAtom(
                atom_id=f"atom_{i}",
                content=f"Atom for section {i}.",
                atom_type="definition",
                citations=[{"chunk_id": f"chunk_{i}", "page": i}],
                section_id=f"sec_{i}",
                lo_id=f"lo_{i}",
            )
        )
        state.link_chunk_to_atom(f"chunk_{i}", f"atom_{i}")
    state.store_mcq_blueprints([_planned_blueprint(i) for i in range(1, n + 1)])
    state.update_section_evidence_status()
    return state


def _accept(state: CooperativeState, q_id: str, blueprint_id: str = "bp_00000001") -> None:
    mcq = CandidateMCQ(
        q_id=q_id,
        blueprint_id=blueprint_id,
        stem=f"Stem {q_id}",
        options=[],
        correct_option_id="A",
        answer_explanation="because",
        source_refs=[],
        flags=[],
        status="accepted",
    )
    state.candidate_mcqs[q_id] = mcq
    state.store_supervisor_decision(
        SupervisorDecision(q_id=q_id, decision="accept", reasons=["ok"], candidate_version=1)
    )


def _reject(state: CooperativeState, q_id: str, blueprint_id: str) -> None:
    mcq = CandidateMCQ(
        q_id=q_id,
        blueprint_id=blueprint_id,
        stem=f"Rejected {q_id}",
        options=[],
        correct_option_id="A",
        answer_explanation="because",
        source_refs=[],
        flags=["cross_cove_failed"],
        status="rejected",
    )
    state.candidate_mcqs[q_id] = mcq
    state.store_supervisor_decision(
        SupervisorDecision(q_id=q_id, decision="reject", reasons=["quality"], candidate_version=1)
    )


def _limit_runtime(limit: int) -> Any:
    cfg = OrchestratorConfig(max_final_questions=limit)
    return workflow_runtime_context(orchestrator_config=cfg)


def _graph_state(cs: CooperativeState, **extra: Any) -> dict:
    state = {
        "cooperative_state": cs,
        "coverage_gaps": extra.get("gaps", [{"section_id": "sec_9"}]),
        "evidence_repair_iterations": extra.get("evidence_iters", 0),
        "max_evidence_repair_iterations": extra.get("max_evidence", 2),
        "blueprint_repair_iterations": extra.get("blueprint_iters", 0),
        "max_blueprint_repair_iterations": extra.get("max_blueprint", 2),
        "step_logs": [],
    }
    return state


class OverproducingLLM:
    def invoke(self, prompt: str) -> str:
        q1 = {
            "stem": "First overproduced question?",
            "correct_option_id": "A",
            "answer_explanation": "A is supported.",
            "options": [
                {"option_id": "A", "text": "Correct", "is_correct": True},
                {"option_id": "B", "text": "Wrong 1", "is_correct": False},
                {"option_id": "C", "text": "Wrong 2", "is_correct": False},
                {"option_id": "D", "text": "Wrong 3", "is_correct": False},
            ],
        }
        q2 = dict(q1)
        q2["stem"] = "Second overproduced question?"
        return json.dumps({"questions": [q1, q2]})


class TestQuestionLimitHelpers(unittest.TestCase):
    def test_default_batch_and_explicit_smaller_cap(self):
        self.assertEqual(generation_batch_size(None), QUIZ_QUESTIONS_PER_ROUND)
        self.assertEqual(generation_batch_size(10), QUIZ_QUESTIONS_PER_ROUND)
        self.assertEqual(generation_batch_size(1), 1)
        self.assertEqual(generation_batch_size(0), 0)

    def test_llm_overproduction_keeps_first_payload_only(self):
        parsed = {"questions": [{"stem": "A"}, {"stem": "B"}, {"stem": "C"}]}
        self.assertEqual(generation_payloads_from_parsed(parsed, limit=1)[0]["stem"], "A")
        self.assertEqual(first_generation_payload([{"stem": "X"}, {"stem": "Y"}])["stem"], "X")


class TestWorkflowEnforcesLimit(unittest.TestCase):
    def test_generation_rounds_shrink_to_remaining_capacity(self):
        cs = _state_with_planned(6)
        _accept(cs, "q_pre_1", "bp_00000001")
        _accept(cs, "q_pre_2", "bp_00000002")
        cs.mcq_blueprints["bp_00000001"].status = "generated"
        cs.mcq_blueprints["bp_00000002"].status = "generated"
        self.assertEqual(remaining_final_question_capacity(cs, {"max_final_questions": 3}), 1)
        before = set(cs.candidate_mcqs)
        update = node_generate_candidate_mcqs_from_blueprints(
            _graph_state(cs),
            _limit_runtime(3),
        )
        created = [
            qid for qid in update["cooperative_state"].candidate_mcqs if qid not in before
        ]
        self.assertEqual(len(created), 1)

    def test_second_round_stops_when_cap_reached(self):
        cs = _state_with_planned(6)
        for i in range(1, 4):
            _accept(cs, f"q_cap_{i}", f"bp_{i:08d}")
        update = node_generate_candidate_mcqs_from_blueprints(
            _graph_state(cs),
            _limit_runtime(3),
        )
        generated_ids = [
            qid
            for qid, mcq in update["cooperative_state"].candidate_mcqs.items()
            if mcq.status == "draft"
        ]
        self.assertEqual(generated_ids, [])

    def test_repair_does_not_loop_past_cap(self):
        cs = _state_with_planned(3)
        for bp in cs.mcq_blueprints.values():
            bp.status = "generated"
        for i in range(1, 3):
            _accept(cs, f"q_acc_{i}", f"bp_{i:08d}")
        route = route_after_coverage(
            _graph_state(
                cs,
                gaps=[{"section_id": "sec_3", "reason": "no accepted MCQ"}],
                evidence_iters=2,
                blueprint_iters=0,
            ),
            _limit_runtime(2),
        )
        self.assertEqual(route, "package_final_quiz")
        before = len(cs.mcq_blueprints)
        update = node_generate_repair_blueprints(
            _graph_state(cs, gaps=[{"section_id": "sec_3", "lo_id": "lo_3"}]),
            _limit_runtime(2),
        )
        self.assertEqual(len(update["cooperative_state"].mcq_blueprints), before)

    def test_packaging_and_acceptance_honor_limit(self):
        cs = _state_with_planned(4)
        for i in range(1, 5):
            qid = f"q_pack_{i:08d}"
            _accept(cs, qid, f"bp_{i:08d}")
        selected = select_final_mcqs_for_quiz(cs, limit=2)
        self.assertEqual(len(selected), 2)
        pack = package_final_quiz(cs, title="T", max_final_questions=2)
        self.assertEqual(len(pack.questions), 2)

    def test_llm_overproduction_registers_one_mcq(self):
        cs = _state_with_planned(1)
        bp = next(iter(cs.mcq_blueprints.values()))
        mcq = generate_candidate_from_blueprint(bp, cs, llm=OverproducingLLM())
        self.assertIn("First overproduced", mcq.stem)
        self.assertNotIn("Second overproduced", mcq.stem)

    def test_fewer_valid_questions_allowed(self):
        cs = _state_with_planned(1)
        _accept(cs, "q_only", "bp_00000001")
        for bp in cs.mcq_blueprints.values():
            bp.status = "generated"
        route = route_after_coverage(
            _graph_state(cs, gaps=[], evidence_iters=2, blueprint_iters=2),
            _limit_runtime(10),
        )
        self.assertEqual(route, "package_final_quiz")
        pack = package_final_quiz(cs, title="T", max_final_questions=10)
        self.assertEqual(len(pack.questions), 1)


class TestReplayReceivesSameLimit(unittest.TestCase):
    def test_profile_limit_is_on_orchestrator_runtime_not_shared_defaults(self):
        self.assertIsNone(DEFAULT_PROFILE.orchestrator_config.max_final_questions)
        profile = profile_with_question_limit(7)
        self.assertEqual(profile.orchestrator_config.max_final_questions, 7)
        self.assertIsNone(DEFAULT_PROFILE.orchestrator_config.max_final_questions)
        bb = CooperativeBackbone(profile=profile, orchestrator_mode="langgraph")
        self.assertEqual(bb.orchestrator.config.max_final_questions, 7)
        runtime_cfg = bb.orchestrator._workflow_runtime.orchestrator_config
        self.assertEqual(runtime_cfg.max_final_questions, 7)
        self.assertEqual(bb.orchestrator.needs.max_final_questions, 7)
        self.assertEqual(QUESTION_LIMIT_SEMANTICS, "final_accepted_mcqs")


class TestAcceptedCountSemantics(unittest.TestCase):
    def test_revision_chain_counts_as_one_logical_question(self):
        cs = _state_with_planned(1)
        mcq = CandidateMCQ(
            q_id="q_00000004",
            blueprint_id="bp_00000001",
            stem="Revised stem",
            options=[],
            correct_option_id="A",
            answer_explanation="because",
            source_refs=[],
            flags=[],
            status="accepted",
            revision_count=2,
        )
        cs.candidate_mcqs["q_00000004"] = mcq
        cs.store_supervisor_decision(
            SupervisorDecision(
                q_id="q_00000004",
                decision="accept",
                reasons=["ok"],
                candidate_version=3,
            )
        )
        self.assertEqual(accepted_final_question_count(cs), 1)
        self.assertEqual(accepted_final_question_ids(cs), ["q_00000004"])

    def test_rejections_do_not_stop_while_planned_remain(self):
        cs = _state_with_planned(14)
        for i in range(1, 11):
            cs.mcq_blueprints[f"bp_{i:08d}"].status = "generated"
            if i <= 6:
                _accept(cs, f"q_{i:08d}", f"bp_{i:08d}")
            else:
                _reject(cs, f"q_{i:08d}", f"bp_{i:08d}")
        self.assertEqual(accepted_final_question_count(cs), 6)
        self.assertEqual(len(cs.planned_blueprints()), 4)
        route = route_after_coverage(
            _graph_state(cs, gaps=[], evidence_iters=2, blueprint_iters=2),
            _limit_runtime(10),
        )
        self.assertEqual(route, "generate_candidate_mcqs_from_blueprints")

    def test_rejected_candidate_stays_rejected_below_target(self):
        cs = _state_with_planned(2)
        for bp in cs.mcq_blueprints.values():
            bp.status = "generated"
        _accept(cs, "q_00000001", "bp_00000001")
        _reject(cs, "q_00000002", "bp_00000002")
        pack = package_final_quiz(cs, title="T", max_final_questions=10)
        self.assertEqual(len(pack.questions), 1)
        self.assertEqual(cs.candidate_mcqs["q_00000002"].status, "rejected")

    def test_packaging_caps_surplus_deterministically(self):
        cs = _state_with_planned(11)
        for i in range(1, 12):
            _accept(cs, f"q_{i:08d}", f"bp_{i:08d}")
        selected = select_final_mcqs_within_limit(cs.candidate_mcqs.values(), 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual([m.q_id for m in selected], [f"q_{i:08d}" for i in range(1, 11)])
        pack = package_final_quiz(cs, title="T", max_final_questions=10)
        self.assertEqual(len(pack.questions), 10)
        self.assertEqual(len(cs.candidate_mcqs), 11)

    def test_question_cap_reached_only_when_accepted_meets_target(self):
        cs = _state_with_planned(6)
        for i in range(1, 7):
            _accept(cs, f"q_{i:08d}", f"bp_{i:08d}")
            cs.mcq_blueprints[f"bp_{i:08d}"].status = "generated"
        hit = build_generation_limit_report(
            cs, config={"max_final_questions": 6}, packaged_count=6
        )
        miss = build_generation_limit_report(
            cs, config={"max_final_questions": 10}, packaged_count=6
        )
        self.assertTrue(hit["question_cap_reached"])
        self.assertFalse(hit["generation_exhausted"])
        self.assertFalse(miss["question_cap_reached"])
        self.assertTrue(miss["generation_exhausted"])
        self.assertTrue(miss["generation_exhaustion_reasons"])

    def test_wald_like_six_opportunities_exhaust_below_ten(self):
        cs = _state_with_planned(6)
        for i in range(1, 7):
            qid = f"q_{i:08d}"
            _accept(cs, qid, f"bp_{i:08d}")
            cs.mcq_blueprints[f"bp_{i:08d}"].status = "generated"
        cs.candidate_mcqs["q_00000004"].revision_count = 2
        cs.store_supervisor_decision(
            SupervisorDecision(
                q_id="q_00000004",
                decision="accept",
                reasons=["ok"],
                candidate_version=3,
            )
        )
        self.assertEqual(accepted_final_question_count(cs), 6)
        route = route_after_coverage(
            _graph_state(cs, gaps=[], evidence_iters=2, blueprint_iters=2),
            _limit_runtime(10),
        )
        self.assertEqual(route, "package_final_quiz")
        update = node_package_final_quiz(
            _graph_state(cs, gaps=[], evidence_iters=2, blueprint_iters=2),
            _limit_runtime(10),
        )
        pack = update["cooperative_state"].final_quiz_packs[-1]
        self.assertEqual(len(pack.questions), 6)
        diag = update["cooperative_state"].mcq_coverage_summary["generation_limit"]
        self.assertEqual(diag["final_accepted_questions"], 6)
        self.assertFalse(diag["question_cap_reached"])
        self.assertTrue(diag["generation_exhausted"])
        self.assertIn("no_remaining_blueprints", diag["generation_exhaustion_reasons"])

    def test_below_target_with_live_gaps_does_not_package_while_repair_budget_remains(self):
        cs = _state_with_planned(3)
        for bp in cs.mcq_blueprints.values():
            bp.status = "failed"
        _accept(cs, "q_00000001", "bp_00000001")
        route = route_after_coverage(
            _graph_state(cs, gaps=[], evidence_iters=2, blueprint_iters=0),
            _limit_runtime(10),
        )
        self.assertEqual(route, "generate_repair_blueprints")

    def test_cap_reached_packages_even_if_planned_remain(self):
        cs = _state_with_planned(12)
        for i in range(1, 11):
            _accept(cs, f"q_{i:08d}", f"bp_{i:08d}")
            cs.mcq_blueprints[f"bp_{i:08d}"].status = "generated"
        self.assertTrue(cs.planned_blueprints())
        route = route_after_coverage(
            _graph_state(cs),
            _limit_runtime(10),
        )
        self.assertEqual(route, "package_final_quiz")


if __name__ == "__main__":
    unittest.main()
