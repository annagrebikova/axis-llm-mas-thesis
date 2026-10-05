"""Regression tests: single-case experiment API + human-readable explanations."""

from __future__ import annotations

import unittest
from typing import Any, List, Optional
from unittest.mock import MagicMock

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.models import ExplainedCase
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.core.entities import EvidenceChunk, KnowledgeAtom
from mas_langgraph.experiment.evidence import (
    build_trace_evidence,
    build_trace_provenance_evidence,
)
from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
from mas_langgraph.experiment.models import (
    ExplanationConditionResult,
    MCQExplanationComparison,
    QuizExplanationExperiment,
)
from mas_langgraph.experiment.service import QuizExplanationExperimentService
from mas_langgraph.experiment.lay_facts import LayExplanationFacts
from mas_langgraph.experiment.synthesis import (
    mechanical_synthesize_from_view,
    synthesize_explanation,
)
from mas_langgraph.experiment.verbalizer import (
    ExplanationEvidenceView,
    build_evidence_view,
    participant_section_headings,
    verbalize_replay_comparison,
)
from mas_langgraph.provenance.models import OccurrenceRecord


_HEADINGS = participant_section_headings()
_JARGON = (
    "EvidenceChunk",
    "KnowledgeAtom",
    "MCQBlueprint",
    "CrossCoVeReport",
    "OutcomeAssessment",
    "SupervisorDecision",
    "ArtifactRef",
    "noop",
    "NOOP",
    "Cross-CoVe",
)


def _fake_service_with_refs(refs: List[ArtifactRef]) -> QuizExplanationExperimentService:
    """Minimal service stub that only exercises generate() selection logic."""
    svc = object.__new__(QuizExplanationExperimentService)
    cs = MagicMock()
    cs.final_quiz_packs = []
    ckpt = MagicMock()
    ckpt.records = []
    svc.factual_run_result = MagicMock(
        run_id="run_sel",
        branch_id="factual",
        artifact_registry=MagicMock(catalog={}),
        produced_by=[],
        provenance_relations=[],
        occurrences=[],
        final_state={"cooperative_state": cs},
        checkpoint_index=ckpt,
    )
    svc.synthesis_model = ""
    svc.model_bundle = None

    def list_refs(*, include_historical_versions: bool = False):
        del include_historical_versions
        return tuple(refs)

    def generate_one(ref, *, question, experiment_seed, progress_callback=None):
        del experiment_seed, progress_callback
        empty_t = ExplanationConditionResult(
            condition="trace",
            explanation="x",
            evidence_summary={},
            raw_evidence={},
        )
        empty_p = ExplanationConditionResult(
            condition="trace_provenance",
            explanation="x",
            evidence_summary={},
            raw_evidence={},
        )
        empty_a = ExplanationConditionResult(
            condition="axis",
            explanation="x",
            evidence_summary={},
            raw_evidence={},
        )
        return MCQExplanationComparison(
            candidate_ref=ref,
            run_id="run_sel",
            branch_id="factual",
            question_stem="stem",
            answer_options=(),
            correct_option_id=None,
            factual_decision="accept",
            user_question=question,
            assessment_ref=ArtifactRef(f"assessment:{ref.logical_id}", ref.version),
            decision_ref=ArtifactRef(f"decision:{ref.logical_id}", ref.version),
            trace=empty_t,
            trace_provenance=empty_p,
            axis=empty_a,
            blinding={"A": "trace", "B": "trace_provenance", "C": "axis"},
        )

    svc.list_experiment_refs = list_refs  # type: ignore[method-assign]
    svc._generate_one = generate_one  # type: ignore[method-assign]
    return svc


class TestSingleCaseSelectionAPI(unittest.TestCase):
    def test_candidate_refs_filters_and_validates(self):
        r1 = ArtifactRef("q_00000001", 1)
        r2 = ArtifactRef("q_00000002", 1)
        r3 = ArtifactRef("q_00000003", 1)
        svc = _fake_service_with_refs([r1, r2, r3])

        all_exp = svc.generate(question="Why?", experiment_seed=1)
        self.assertEqual(len(all_exp.cases), 3)

        one = svc.generate(
            question="Why?",
            experiment_seed=1,
            candidate_refs=[r2],
        )
        self.assertEqual(len(one.cases), 1)
        self.assertEqual(one.cases[0].candidate_ref, r2)

        with self.assertRaises(ValueError):
            svc.generate(
                question="Why?",
                experiment_seed=1,
                candidate_refs=[ArtifactRef("q_missing", 1)],
            )

        # Exact version required — no silent substitution.
        with self.assertRaises(ValueError):
            svc.generate(
                question="Why?",
                experiment_seed=1,
                candidate_refs=[ArtifactRef("q_00000001", 99)],
            )


class TestTraceLeakageParticipant(unittest.TestCase):
    def test_trace_has_no_source_content_or_replay(self):
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q1", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q1", 1),
            outcome="accept",
            assessment_ref=ArtifactRef("assessment:q1", 1),
        )
        occs = (
            OccurrenceRecord(
                occurrence_id="r:factual:generate_candidate_mcqs_from_blueprints#1",
                node_id="generate_candidate_mcqs_from_blueprints",
                occurrence_index=1,
                branch_id="factual",
                run_id="r",
                execution_index=1,
            ),
            OccurrenceRecord(
                occurrence_id="r:factual:apply_acceptance_gate#1",
                node_id="apply_acceptance_gate",
                occurrence_index=1,
                branch_id="factual",
                run_id="r",
                execution_index=2,
            ),
        )
        raw = build_trace_evidence(
            explained_case=explained, factual_occurrences=occs
        )
        view = build_evidence_view(condition="trace", raw_evidence=raw)
        text = synthesize_explanation(
            user_question="Why?",
            factual_decision="accept",
            question_stem="stem",
            evidence_view=view,
            llm_generate=None,
        )
        self.assertFalse(raw["occurrence_selection_uses_provenance_slice"])
        self.assertFalse(raw["provenance_included"])
        self.assertFalse(raw["counterfactual_replay_included"])
        joined = (view.to_prompt_text() + "\n" + text).lower()
        self.assertNotIn("source passage explaining", joined)
        self.assertNotIn("counterfactual", joined)
        self.assertNotIn("tested change", joined)
        self.assertNotIn("q1:v1", text)
        self.assertNotIn("Why this question?", text)
        self.assertTrue(
            "execution record" in text.lower()
            or "trace shows" in text.lower()
            or "workflow" in text.lower()
            or "generated" in text.lower(),
            text,
        )


class TestProvenanceSourceContent(unittest.TestCase):
    def test_chunk_content_reaches_human_evidence(self):
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q1", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q1", 1),
            outcome="accept",
        )
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_occurrences=(),
            relevant_relations=(),
            relevant_artifact_refs=(ArtifactRef("chunk_x", 1),),
            decision="accept",
        )

        class _Env:
            artifact_type = "EvidenceChunk"
            payload = EvidenceChunk(
                chunk_id="chunk_x",
                content="The first step is to place k means at random.",
                metadata={},
            )

        class _Reg:
            def get_by_ref(self, ref, branch=None):
                del branch
                if str(ref) == "chunk_x:v1":
                    return _Env()
                return None

        raw = build_trace_provenance_evidence(
            explained_case=explained,
            case_slice=case_slice,
            registry=_Reg(),
        )
        row = raw["relevant_artifacts"][0]
        self.assertIn("content", row)
        self.assertIn("place k means at random", row["content"])
        view = build_evidence_view(condition="trace_provenance", raw_evidence=raw)
        joined = " ".join(view.recorded_dependencies)
        self.assertIn("place k means at random", joined)
        self.assertNotIn("chunk_x", joined)


class TestSharedStructureAndJargon(unittest.TestCase):
    def test_concise_prose_and_no_jargon(self):
        views = {
            "trace": ExplanationEvidenceView(
                factual_decision="ACCEPT",
                observed_process=("The candidate question was generated.",),
                limitations=(
                    "The execution record shows which steps occurred, but it does not "
                    "show which specific source items were used for this question or "
                    "whether an individual input changed the decision.",
                ),
                lay_facts=LayExplanationFacts(
                    decision="ACCEPT",
                    educational_purpose=("The candidate question was generated.",),
                    limitations=(
                        "The execution record alone does not identify which specific "
                        "teaching information explains why this particular question "
                        "was selected.",
                    ),
                    source_basis_unavailable=True,
                ),
            ),
            "trace_provenance": ExplanationEvidenceView(
                factual_decision="ACCEPT",
                observed_process=("The candidate question was generated.",),
                recorded_dependencies=(
                    'The question was based on a source passage explaining that '
                    "GMM places k means at random.",
                ),
                decision_evidence=(
                    "The recorded decision for the candidate question was ACCEPT.",
                ),
                limitations=(
                    "The records show which information was used, but not whether "
                    "an individual item was necessary for the decision.",
                ),
                lay_facts=LayExplanationFacts(
                    decision="ACCEPT",
                    educational_purpose=("The candidate question was generated.",),
                    source_basis=(
                        "GMM places k means at random.",
                    ),
                    limitations=(
                        "The records show that this information was used for the "
                        "question, but they do not establish whether it was decisive "
                        "for selecting this particular MCQ.",
                    ),
                    specific_source_known=True,
                    recorded_use_known=True,
                ),
            ),
            "axis": ExplanationEvidenceView(
                factual_decision="ACCEPT",
                observed_process=("The candidate question was generated.",),
                recorded_dependencies=("A source passage was used.",),
                decision_evidence=(
                    "The recorded decision for the candidate question was ACCEPT.",
                ),
                tested_interventions=(
                    "Tested change 1:",
                    "The system tested what would happen without a source passage.",
                    "A control replay without any change reproduced the original ACCEPT decision.",
                    "After the tested change, the final decision became REJECT (was ACCEPT). "
                    "The tested change altered the final decision.",
                ),
                limitations=(
                    "Causal claims apply only to the tested changes below; they do not "
                    "generalize beyond those tests.",
                ),
            ),
        }
        for cond, view in views.items():
            text = synthesize_explanation(
                user_question="Why?",
                factual_decision="accept",
                question_stem="What is GMM?",
                evidence_view=view,
                llm_generate=None,
            )
            self.assertNotIn("Why this question?", text)
            self.assertNotIn("What was it based on?", text)
            for term in _JARGON:
                self.assertNotIn(term, text)
            self.assertNotRegex(text, r"\bq_\d")
            self.assertNotRegex(text, r"\bchunk_")
            self.assertNotIn(":v1", text)
            ok, warns = validate_explanation_faithfulness(
                condition=cond,  # type: ignore[arg-type]
                explanation=text,
                evidence_view=view,
            )
            self.assertTrue(ok, msg=f"{cond}: {warns}")


class TestAxisReplayVisibility(unittest.TestCase):
    def test_decision_effect_appears_in_final_text(self):
        cmp = {
            "evidence_status": "decision_effect",
            "intervention_signature": {
                "operation": "REMOVE",
                "target_ref": {"key": "chunk_1:v1"},
            },
            "factual": {"decision": "accept", "subject_present": True},
            "noop": {"decision": "accept", "subject_present": True},
            "counterfactual": {"decision": "reject", "subject_present": True},
            "criterion_deltas": [],
            "notes": [],
        }
        arts = [
            {
                "ref": "chunk_1:v1",
                "artifact_type": "EvidenceChunk",
                "content_preview": "GMM places k means at random",
            }
        ]
        interventions = verbalize_replay_comparison(cmp, 1, relevant_artifacts=arts)
        view = build_evidence_view(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": arts,
                "replay_comparisons": [cmp],
            },
        )
        text = mechanical_synthesize_from_view(view=view).lower()
        self.assertIn("tested", text)
        self.assertIn("accept", text)
        self.assertIn("final decision changed", text)
        self.assertNotIn("chunk_1", text)
        self.assertNotIn("noop", text)

    def test_no_demonstrated_effect_not_irrelevance(self):
        cmp = {
            "evidence_status": "no_demonstrated_effect",
            "intervention_signature": {
                "operation": "REMOVE",
                "target_ref": {"key": "chunk_1:v1"},
            },
            "factual": {"decision": "accept"},
            "noop": {"decision": "accept"},
            "counterfactual": {"decision": "accept"},
            "notes": [],
        }
        view = build_evidence_view(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "replay_comparisons": [cmp],
                "relevant_artifacts": [],
            },
        )
        text = mechanical_synthesize_from_view(view=view).lower()
        self.assertIn("remained accept", text)
        self.assertNotIn("the source was irrelevant", text)
        self.assertNotIn("does not matter", text)

    def test_replay_unstable_no_causal_claim(self):
        cmp = {
            "evidence_status": "replay_unstable",
            "intervention_signature": {
                "operation": "REMOVE",
                "target_ref": {"key": "chunk_1:v1"},
            },
            "factual": {"decision": "accept"},
            "noop": {"decision": "reject"},
            "counterfactual": {"decision": "reject"},
            "notes": [],
        }
        view = build_evidence_view(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "replay_comparisons": [cmp],
                "relevant_artifacts": [],
            },
        )
        text = mechanical_synthesize_from_view(view=view).lower()
        self.assertTrue(
            "no reliable" in text
            or "no causal" in text
            or "could be drawn" in text
            or "not produce" in text
            or "inconclusive" in text
            or "limitation" in " ".join(view.limitations).lower()
        )
        self.assertNotIn("final decision changed", text)


class TestAxisOmissionTriggersRepair(unittest.TestCase):
    def test_generic_axis_text_without_intervention_is_flagged(self):
        view = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            observed_process=("The candidate question was generated.",),
            tested_interventions=(
                "Tested change 1:",
                "The system tested what would happen without a source passage.",
                "A control replay without any change reproduced the original ACCEPT decision.",
                "The tested change altered the final decision.",
            ),
        )
        generic = (
            "Why this question?\n"
            "The question was generated from teaching material.\n\n"
            "What was it based on?\n"
            "Source material was retrieved.\n\n"
            "What difference did this information make?\n"
            "The candidate passed structural, evidence, and additional answer checks."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis", explanation=generic, evidence_view=view
        )
        self.assertFalse(ok)
        self.assertTrue(any("omits valid tested-intervention" in w for w in warns))


if __name__ == "__main__":
    unittest.main()
