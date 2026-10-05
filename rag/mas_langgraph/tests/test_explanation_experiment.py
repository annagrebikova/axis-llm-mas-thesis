"""Deterministic tests for Trace / Trace+Provenance / AXIS experiment ablation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import patch

from mas_langgraph.artifacts.channel_state import (
    dump_artifact_channels,
    empty_artifact_channels,
    load_artifact_channels,
)
from mas_langgraph.artifacts.models import ArtifactEnvelope, ArtifactRef
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.axis.models import ExplainedCase
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.cases.mcq import MCQCaseDefinition
from mas_langgraph.cases.mcq.artifact_registration import register_mcq_artifacts_for_stage
from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
from mas_langgraph.cases.mcq.explained_case import MCQExplainedCaseResolver
from mas_langgraph.core.checkpointer import default_factual_checkpointer
from mas_langgraph.core.entities import (
    DocSection,
    EvidenceChunk,
    KnowledgeAtom,
    LearningObjective,
    QuizPack,
    ValidatedQuestion,
)
from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
from mas_langgraph.core.outcomes import OutcomeAssessment
from mas_langgraph.core.runtime import WorkflowRuntime
from mas_langgraph.core.workflow_executor import WorkflowExecutor
from mas_langgraph.domains.quiz.entities import (
    AnswerOption,
    CandidateMCQ,
    SupervisorDecision,
)
from mas_langgraph.experiment.blinding import blinding_mapping
from mas_langgraph.experiment.evidence import (
    assert_no_leakage,
    build_trace_evidence,
    build_trace_provenance_evidence,
)
from mas_langgraph.experiment.export import (
    dump_blinded_experiment_markdown,
    dump_condition_key,
    dump_explanation_experiment,
)
from mas_langgraph.experiment.models import ExplanationConditionResult, MCQExplanationComparison
from mas_langgraph.experiment.selection import select_experiment_refs
from mas_langgraph.experiment.service import QuizExplanationExperimentService
from mas_langgraph.orchestrator.needs import NeedsEvaluator
from mas_langgraph.provenance.channel_state import (
    dump_provenance_channels,
    empty_provenance_channels,
    load_provenance_recorder,
)
from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation
from mas_langgraph.replay.executor import WorkflowRunResult
from mas_langgraph.tests.test_prompt16_end_to_end import _build_seeded_factual_run


def _manual_explainable_fixture() -> Tuple[WorkflowRunResult, ArtifactRef]:
    """Minimal complete candidate+assessment+decision registry (no OpenAI)."""
    reg = ArtifactRegistry()
    cref = ArtifactRef("q_exp1", 1)
    aref = ArtifactRef("assessment:q_exp1", 1)
    dref = ArtifactRef("decision:q_exp1", 1)
    cand = CandidateMCQ(
        q_id="q_exp1",
        blueprint_id="bp1",
        stem="What does k-means assign points to?",
        options=[
            AnswerOption("A", "Nearest centroid", True),
            AnswerOption("B", "Random cluster", False),
            AnswerOption("C", "Previous label", False),
            AnswerOption("D", "Outlier bucket", False),
        ],
        correct_option_id="A",
        answer_explanation="Nearest centroid.",
        status="accepted",
    )
    assessment = OutcomeAssessment(
        assessment_ref=aref, target_ref=cref, criteria=()
    )
    decision = SupervisorDecision(
        q_id="q_exp1",
        decision="accept",
        reasons=["ok"],
        candidate_version=1,
        assessment_ref=aref,
    )
    for logical, ver, atype, payload in (
        ("q_exp1", 1, "CandidateMCQ", cand),
        ("assessment:q_exp1", 1, "OutcomeAssessment", assessment),
        ("decision:q_exp1", 1, "SupervisorDecision", decision),
    ):
        reg.store(
            ArtifactEnvelope.create(
                logical_id=logical,
                version=ver,
                artifact_type=atype,
                payload=payload,
                branch_id="factual",
                created_in_run="run_exp",
            )
        )
    # Historical incomplete version for selection tests
    reg.store(
        ArtifactEnvelope.create(
            logical_id="q_old",
            version=1,
            artifact_type="CandidateMCQ",
            payload=CandidateMCQ(
                q_id="q_old",
                blueprint_id="bp0",
                stem="old",
                options=[],
                correct_option_id="A",
                answer_explanation="",
                status="accepted",
            ),
            branch_id="factual",
            created_in_run="run_exp",
        )
    )
    occ = OccurrenceRecord(
        occurrence_id="run_exp:factual:apply_acceptance_gate#1",
        node_id="apply_acceptance_gate",
        occurrence_index=1,
        branch_id="factual",
        run_id="run_exp",
        execution_index=5,
    )
    rel = RunProvenanceRelation(
        relation="candidate_question",
        artifact_ref=cref,
        producer_occurrence_id="run_exp:factual:generate_candidate_mcqs_from_blueprints#1",
        consumer_occurrence_id=occ.occurrence_id,
    )
    pack = QuizPack(
        pack_id="pack_exp",
        title="exp",
        questions=[
            ValidatedQuestion(
                q_id="q_exp1",
                question_text=cand.stem,
                question_type="multiple_choice",
                correct_answer="Nearest centroid",
                difficulty="intermediate",
                options=[o.text for o in cand.options],
            )
        ],
    )
    cs = CooperativeState()
    cs.final_quiz_packs = [pack]
    fr = WorkflowRunResult(
        run_id="run_exp",
        branch_id="factual",
        final_state={
            "artifact_catalog": dict(reg.catalog),
            "occurrence_records": [occ],
            "run_provenance_relations": [rel],
            "produced_by": [],
            "cooperative_state": cs,
            "branch_id": "factual",
            "run_id": "run_exp",
        },
    )
    return fr, cref


class TestEvidenceLeakage(unittest.TestCase):
    def test_trace_excludes_provenance_and_replay(self):
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("Q", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:Q", 1),
            outcome="accept",
            assessment_ref=ArtifactRef("assessment:Q", 1),
        )
        occ = OccurrenceRecord(
            occurrence_id="r:factual:verify#1",
            node_id="verify_candidate_mcqs_against_evidence",
            occurrence_index=1,
            branch_id="factual",
            run_id="r",
            execution_index=1,
        )
        rel = RunProvenanceRelation(
            relation="supporting_evidence",
            artifact_ref=ArtifactRef("E1", 1),
            producer_occurrence_id="r:factual:retrieve#1",
            consumer_occurrence_id=occ.occurrence_id,
        )
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_occurrences=(occ,),
            relevant_relations=(rel,),
            relevant_artifact_refs=(ArtifactRef("Q", 1), ArtifactRef("E1", 1)),
            decision="accept",
        )
        trace = build_trace_evidence(
            explained_case=explained,
            factual_occurrences=case_slice.relevant_occurrences,
        )
        assert_no_leakage(trace, allow_provenance=False)
        self.assertFalse(trace["provenance_included"])
        self.assertNotIn("provenance_relations", trace)
        self.assertFalse(trace.get("replay_comparisons"))

        tp = build_trace_provenance_evidence(
            explained_case=explained, case_slice=case_slice
        )
        assert_no_leakage(tp, allow_provenance=True)
        self.assertTrue(tp["provenance_included"])
        self.assertTrue(tp["provenance_relations"])
        self.assertEqual(tp["replay_comparisons"], [])
        self.assertFalse(tp["counterfactual_replay_included"])


class TestBlindingReproducibility(unittest.TestCase):
    def test_same_seed_same_mapping(self):
        ref = ArtifactRef("q_x", 1)
        a = blinding_mapping(run_id="run1", candidate_ref=ref, experiment_seed=42)
        b = blinding_mapping(run_id="run1", candidate_ref=ref, experiment_seed=42)
        self.assertEqual(a, b)
        self.assertEqual(set(a.keys()), {"A", "B", "C"})
        self.assertEqual(set(a.values()), {"trace", "trace_provenance", "axis"})
        c = blinding_mapping(run_id="run1", candidate_ref=ref, experiment_seed=99)
        # Different seed may differ; still a valid permutation.
        self.assertEqual(set(c.values()), {"trace", "trace_provenance", "axis"})


class TestHistoricalExclusion(unittest.TestCase):
    def test_default_excludes_non_pack_refs(self):
        fr, cref = _manual_explainable_fixture()
        resolver = MCQExplainedCaseResolver()
        all_refs = resolver.list_explainable_artifacts(factual_run_result=fr)
        # Only q_exp1 is fully explainable; q_old incomplete.
        self.assertIn(cref, all_refs)
        selected = select_experiment_refs(
            explainable_refs=all_refs,
            factual_run_result=fr,
            resolver=resolver,
            include_historical_versions=False,
        )
        self.assertEqual(selected, (cref,))
        # Empty pack → empty selection
        fr2 = WorkflowRunResult(
            run_id="run_exp",
            branch_id="factual",
            final_state={
                **fr.final_state,
                "cooperative_state": CooperativeState(),
            },
        )
        empty = select_experiment_refs(
            explainable_refs=all_refs,
            factual_run_result=fr2,
            resolver=resolver,
            include_historical_versions=False,
        )
        self.assertEqual(empty, ())


class TestThreeConditionsSameIdentity(unittest.TestCase):
    def test_seeded_graph_three_conditions(self):
        case_def, ex, factual = _build_seeded_factual_run("run_exp_three")
        self.assertIsNone(factual.error)
        # Ensure a quiz pack exists for selection.
        cs = factual.final_state.get("cooperative_state")
        refs = MCQExplainedCaseResolver().list_explainable_artifacts(
            factual_run_result=factual
        )
        if refs and cs is not None and not getattr(cs, "final_quiz_packs", None):
            # Package should normally exist; if not, attach synthetic pack from refs.
            q = ValidatedQuestion(
                q_id=refs[-1].logical_id,
                question_text="t",
                question_type="multiple_choice",
                correct_answer="A",
                difficulty="intermediate",
            )
            cs.final_quiz_packs = [QuizPack("pack_t", "t", [q])]

        service = QuizExplanationExperimentService(
            case_definition=case_def,
            factual_run_result=factual,
            replay_executor=ex.replay_executor,
            checkpoint_index=factual.checkpoint_index,
            llm_generate=None,
            synthesis_llm_generate=None,
            cooperative_state=cs,
            # Deterministic AXIS (no OpenAI): DONE is a valid first query
            axis_scripted_queries=["DONE"],
            max_rounds=3,
        )
        before_keys = sorted(
            (factual.artifact_registry.catalog if factual.artifact_registry else {}).keys()
        )
        before_rels = list(factual.provenance_relations)
        experiment = service.generate(
            question="Why did the system produce this decision?",
            experiment_seed=42,
            include_historical_versions=False,
        )
        self.assertGreaterEqual(len(experiment.cases), 1)
        cmp_case = experiment.cases[0]
        self.assertEqual(cmp_case.trace.condition, "trace")
        self.assertEqual(cmp_case.trace_provenance.condition, "trace_provenance")
        self.assertEqual(cmp_case.axis.condition, "axis")
        self.assertEqual(cmp_case.run_id, factual.run_id)
        self.assertEqual(cmp_case.trace.raw_evidence["candidate_ref"], str(cmp_case.candidate_ref))
        self.assertEqual(
            cmp_case.trace_provenance.raw_evidence["candidate_ref"],
            str(cmp_case.candidate_ref),
        )
        self.assertEqual(cmp_case.assessment_ref.version, cmp_case.candidate_ref.version)
        self.assertEqual(cmp_case.decision_ref.version, cmp_case.candidate_ref.version)
        # No leakage
        assert_no_leakage(cmp_case.trace.raw_evidence, allow_provenance=False)
        assert_no_leakage(cmp_case.trace_provenance.raw_evidence, allow_provenance=True)
        # Factual immutability
        after_keys = sorted(
            (factual.artifact_registry.catalog if factual.artifact_registry else {}).keys()
        )
        self.assertEqual(before_keys, after_keys)
        self.assertEqual(before_rels, list(factual.provenance_relations))
        # Export works
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            dump_explanation_experiment(experiment, str(p / "exp.json"))
            dump_condition_key(experiment, str(p / "key.json"))
            dump_blinded_experiment_markdown(experiment, str(p / "eval.md"))
            blinded = (p / "eval.md").read_text(encoding="utf-8")
            self.assertIn("Explanation A", blinded)
            self.assertNotIn("## Trace", blinded)
            self.assertNotIn("AXIS", blinded.split("Explanation")[0])
            key = json.loads((p / "key.json").read_text(encoding="utf-8"))
            self.assertIn(str(cmp_case.candidate_ref), key["mapping"])


class TestAxisFailureKeepsBaselines(unittest.TestCase):
    def test_axis_exception_preserves_baselines(self):
        case_def, ex, factual = _build_seeded_factual_run("run_exp_fail")
        cs = factual.final_state.get("cooperative_state")
        refs = MCQExplainedCaseResolver().list_explainable_artifacts(
            factual_run_result=factual
        )
        if not refs:
            self.skipTest("fixture produced no explainable refs")
        if cs is not None and not getattr(cs, "final_quiz_packs", None):
            cs.final_quiz_packs = [
                QuizPack(
                    "pack_t",
                    "t",
                    [
                        ValidatedQuestion(
                            refs[0].logical_id,
                            "t",
                            "multiple_choice",
                            "A",
                            "intermediate",
                        )
                    ],
                )
            ]
        service = QuizExplanationExperimentService(
            case_definition=case_def,
            factual_run_result=factual,
            replay_executor=ex.replay_executor,
            checkpoint_index=factual.checkpoint_index,
            llm_generate=None,
            cooperative_state=cs,
            axis_scripted_queries=["DONE"],
        )

        def boom(*a, **k):
            raise RuntimeError("axis boom")

        with patch.object(service._axis_service, "explain", side_effect=boom):
            experiment = service.generate(
                experiment_seed=1,
                include_historical_versions=False,
            )
        self.assertGreaterEqual(len(experiment.cases), 1)
        c = experiment.cases[0]
        self.assertTrue(c.trace.explanation)
        self.assertTrue(c.trace_provenance.explanation)
        self.assertEqual(c.axis_status, "failed")
        self.assertFalse(c.axis_causal_evidence_available)
        self.assertTrue(c.axis.warnings)


class TestExportBlindingNotInEvaluatorMd(unittest.TestCase):
    def test_condition_names_absent_from_blinded_md(self):
        fr, cref = _manual_explainable_fixture()
        explained = MCQExplainedCaseResolver().resolve(
            factual_run_result=fr, artifact_ref=cref
        )
        case = MCQExplanationComparison(
            run_id=fr.run_id,
            branch_id="factual",
            candidate_ref=cref,
            assessment_ref=explained.assessment_ref,
            decision_ref=explained.decision_ref,
            question_stem="stem",
            answer_options=(),
            correct_option_id="A",
            factual_decision="accept",
            user_question="why?",
            trace=ExplanationConditionResult("trace", "T"),
            trace_provenance=ExplanationConditionResult("trace_provenance", "P"),
            axis=ExplanationConditionResult("axis", "X"),
            blinding={"A": "axis", "B": "trace", "C": "trace_provenance"},
        )
        from mas_langgraph.experiment.models import QuizExplanationExperiment

        exp = QuizExplanationExperiment(
            schema_version=1,
            run_id=fr.run_id,
            branch_id="factual",
            experiment_seed=42,
            user_question="why?",
            cases=(case,),
        )
        with tempfile.TemporaryDirectory() as td:
            path = dump_blinded_experiment_markdown(exp, str(Path(td) / "e.md"))
            text = Path(path).read_text(encoding="utf-8")
        self.assertIn("Explanation A", text)
        self.assertIn("Explanation B", text)
        self.assertIn("Explanation C", text)
        self.assertNotIn("## Trace", text)
        self.assertNotIn("Trace + Provenance", text)
        self.assertNotIn("## AXIS", text)


class TestAxisReplayEvidenceRecorded(unittest.TestCase):
    def test_empty_replay_marked_not_causal(self):
        from mas_langgraph.experiment.evidence import build_axis_evidence_summary

        class _Fake:
            replay_comparisons = []
            intervention_records = []
            round_records = []
            rounds = 2
            termination_reason = "done"
            warnings = ()
            final_explanation = "x"

        summary = build_axis_evidence_summary(explanation_result=_Fake())
        self.assertFalse(summary["axis_causal_evidence_available"])
        self.assertEqual(summary["replay_comparison_count"], 0)

    def test_decision_effect_marked_causal(self):
        from mas_langgraph.experiment.evidence import build_axis_evidence_summary

        class _Cmp:
            evidence_status = "decision_effect"

        class _Fake:
            replay_comparisons = [_Cmp()]
            intervention_records = [object()]
            round_records = [object()]
            rounds = 3
            termination_reason = "done"
            warnings = ()
            final_explanation = "x"

        summary = build_axis_evidence_summary(explanation_result=_Fake())
        self.assertTrue(summary["axis_causal_evidence_available"])
        self.assertEqual(summary["successful_replay_count"], 1)


if __name__ == "__main__":
    unittest.main()
