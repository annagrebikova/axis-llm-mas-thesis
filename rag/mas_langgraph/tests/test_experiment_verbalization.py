"""Verbalization refactor tests — shared prompt, evidence views, faithfulness."""

from __future__ import annotations

import re
import unittest
from typing import Any

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.models import ExplainedCase
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.experiment.evidence import (
    assert_no_leakage,
    build_axis_evidence_summary,
    build_trace_evidence,
    build_trace_provenance_evidence,
    synthesis_evidence_without_axis_raw,
)
from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
from mas_langgraph.experiment.lay_facts import LayExplanationFacts
from mas_langgraph.experiment.synthesis import (
    SHARED_FINAL_SYNTHESIS_PROMPT,
    TARGET_WORD_RANGE,
    build_shared_final_prompt,
    synthesize_explanation,
)
from mas_langgraph.experiment.verbalizer import (
    ExplanationEvidenceView,
    build_evidence_view,
    verbalize_dependencies,
    verbalize_process_steps,
    verbalize_replay_comparison,
)
from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation


def _slice_with_trace() -> CaseSlice:
    explained = ExplainedCase(
        run_id="r",
        artifact_ref=ArtifactRef("q1", 1),
        branch_id="factual",
        decision_ref=ArtifactRef("decision:q1", 1),
        outcome="accept",
        assessment_ref=ArtifactRef("assessment:q1", 1),
    )
    labels = [
        ("build_outline", 1),
        ("generate_learning_objectives", 1),
        ("initial_retrieve_evidence_for_all_sections", 1),
        ("atomize_evidence", 1),
        ("plan_mcq_blueprints", 1),
        ("generate_candidate_mcqs_from_blueprints", 1),
        ("verify_candidate_mcqs_against_evidence", 1),
        ("cross_cove_verify_candidate_mcqs", 1),
        ("apply_acceptance_gate", 1),
    ]
    occs = tuple(
        OccurrenceRecord(
            occurrence_id=f"r:factual:{node}#{idx}",
            node_id=node,
            occurrence_index=idx,
            branch_id="factual",
            run_id="r",
            execution_index=i + 1,
        )
        for i, (node, idx) in enumerate(labels)
    )
    rel = RunProvenanceRelation(
        relation="supporting_evidence",
        artifact_ref=ArtifactRef("chunk_1", 1),
        producer_occurrence_id=occs[2].occurrence_id,
        consumer_occurrence_id=occs[5].occurrence_id,
    )
    return CaseSlice(
        explained_case=explained,
        relevant_occurrences=occs,
        relevant_relations=(rel,),
        relevant_artifact_refs=(
            ArtifactRef("q1", 1),
            ArtifactRef("chunk_1", 1),
            ArtifactRef("atom_1", 1),
            ArtifactRef("atom_2", 1),
            ArtifactRef("atom_3", 1),
            ArtifactRef("atom_4", 1),
            ArtifactRef("atom_5", 1),
            ArtifactRef("atom_6", 1),
            ArtifactRef("sec_kmeans", 1),
            ArtifactRef("lo_1", 1),
        ),
        decision="accept",
        decision_reasons=("ok",),
        criterion_values=(),
    )


class TestSharedFinalPrompt(unittest.TestCase):
    def test_identical_prompt_for_all_conditions(self):
        view_t = ExplanationEvidenceView(
            factual_decision="ACCEPT", observed_process=("Step one.",)
        )
        view_p = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            observed_process=("Step one.",),
            recorded_dependencies=("Used section X.",),
        )
        view_a = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            observed_process=("Step one.",),
            tested_interventions=(
                "Tested change 1:",
                "A control replay without any change reproduced the original ACCEPT decision.",
            ),
        )
        prompts = []
        for view in (view_t, view_p, view_a):
            prompts.append(
                build_shared_final_prompt(
                    user_question="Why did the system produce this decision?",
                    factual_decision="accept",
                    question_stem="What is k-means?",
                    verbalized_evidence=view.to_prompt_text(),
                )
            )
        for p in prompts:
            self.assertIn("Verified explanation facts:", p)
            self.assertIn("Do not create fixed sections", p)
            self.assertNotIn("USE EXACTLY THESE SECTIONS", p)
            self.assertNotIn("Condition: trace", p)
            self.assertNotIn("Condition: trace_provenance", p)
            self.assertNotIn("Condition: axis", p)
            self.assertNotIn("Condition rules:", p)
            self.assertIn("Never mention", p)
            self.assertIn("AXIS", SHARED_FINAL_SYNTHESIS_PROMPT)  # banned list only
        self.assertEqual(TARGET_WORD_RANGE, (1, 200))
        from mas_langgraph.experiment.synthesis import MAX_PARTICIPANT_WORDS
        self.assertEqual(MAX_PARTICIPANT_WORDS, 200)
        self.assertTrue(all("You are writing a user-facing explanation" in p for p in prompts))


class TestTraceVerbalization(unittest.TestCase):
    def test_human_readable_process_no_provenance(self):
        case_slice = _slice_with_trace()
        explained = case_slice.explained_case
        raw = build_trace_evidence(
            explained_case=explained,
            factual_occurrences=case_slice.relevant_occurrences,
        )
        assert_no_leakage(raw, allow_provenance=False)
        view = build_evidence_view(condition="trace", raw_evidence=raw)
        text = view.to_prompt_text()
        self.assertTrue(
            "trace" in text.lower()
            or "workflow" in text.lower()
            or "generated" in text.lower()
            or "execution record" in text.lower()
            or "does not identify" in text.lower()
            or "specific_source_known = false" in text.lower(),
            text,
        )
        self.assertNotIn("organized the teaching material into sections", text.lower())
        self.assertNotIn("unlabeled data", text.lower())
        self.assertIn("ACCEPT", text)
        self.assertTrue(view.limitations)
        self.assertNotIn("RunProvenanceRelation", text)
        self.assertNotIn("ArtifactBinding", text)
        self.assertNotIn("counterfactual", text.lower())
        # Trace may mention the word only in banned senses; ensure no CF findings.
        self.assertFalse(view.tested_interventions)
        self.assertFalse(raw.get("occurrence_selection_uses_provenance_slice"))
        self.assertNotIn("unlabeled data", text.lower())


class TestProvenanceAggregation(unittest.TestCase):
    def test_aggregate_atoms(self):
        arts = [
            {"ref": "sec_kmeans:v1", "artifact_type": "DocSection", "title": "k-Means"},
            {
                "ref": "lo_1:v1",
                "artifact_type": "LearningObjective",
                "objective_text": "Explain k-means",
            },
            {
                "ref": "chunk_1:v1",
                "artifact_type": "EvidenceChunk",
                "content_preview": "Place k means at random.",
            },
            {"ref": "chunk_2:v1", "artifact_type": "EvidenceChunk"},
        ] + [
            {
                "ref": f"atom_{i}:v1",
                "artifact_type": "KnowledgeAtom",
                "content_preview": "Clustering groups similar points.",
            }
            for i in range(6)
        ]
        lines = verbalize_dependencies(
            relevant_artifacts=arts,
            provenance_relations=[{"relation": "grounding_atom"}] * 6,
        )
        joined = " ".join(lines)
        self.assertTrue(
            "k-Means" in joined
            or "Explain k-means" in joined
            or "Place k means" in joined
            or "Clustering groups" in joined
        )
        # Prefer clean KnowledgeAtom content when available.
        self.assertTrue(
            "Place k means at random" in joined or "Clustering groups" in joined,
            joined,
        )
        self.assertNotIn("knowledge atom", joined.lower())
        self.assertNotIn("EvidenceChunk", joined)
        self.assertNotIn("chunk_1", joined)


class TestProvenanceCausalDiscipline(unittest.TestCase):
    def test_no_causal_claim_in_mechanical_tp(self):
        case_slice = _slice_with_trace()
        raw = build_trace_provenance_evidence(
            explained_case=case_slice.explained_case, case_slice=case_slice
        )
        raw["relevant_artifacts"] = [
            {"ref": "sec_kmeans:v1", "artifact_type": "DocSection", "title": "k-Means"},
            {
                "ref": "chunk_1:v1",
                "artifact_type": "EvidenceChunk",
                "content": "The first step is to place k means at random.",
                "content_preview": "The first step is to place k means at random.",
            },
        ]
        view = build_evidence_view(condition="trace_provenance", raw_evidence=raw)
        text = synthesize_explanation(
            user_question="Why?",
            factual_decision="accept",
            question_stem="stem",
            evidence_view=view,
            llm_generate=None,
        )
        self.assertNotRegex(text.lower(), r"\bcaused\b")
        if re.search(r"\bdecisive\b", text.lower()):
            self.assertRegex(
                text.lower(),
                r"(not\s+decisive|whether\s+it\s+was\s+decisive|do\s+not\s+establish)",
            )
        self.assertIn("place k means at random", " ".join(view.recorded_dependencies).lower())
        joined_lim = (
            " ".join(view.limitations)
            + " "
            + " ".join(getattr(view.lay_facts, "relevant_provenance_facts", ()) or ())
            + " "
            + text
        ).lower()
        self.assertTrue(
            "linked" in joined_lim
            or "stating that" in joined_lim
            or "place k means" in joined_lim,
            joined_lim,
        )
        self.assertNotIn("available during planning, generation, and checking", joined_lim)
        self.assertNotIn("would have been different", joined_lim)
        self.assertNotIn("Why this question?", text)
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance", explanation=text, evidence_view=view
        )
        self.assertTrue(ok, msg=str(warns))


class TestAxisReplayVerbalization(unittest.TestCase):
    def test_decision_effect_block(self):
        cmp = {
            "evidence_status": "decision_effect",
            "intervention_signature": {
                "operation": "REMOVE",
                "target_ref": {"key": "chunk_abc:v1", "logical_id": "chunk_abc", "version": 1},
            },
            "factual": {"decision": "accept", "outcome_available": True},
            "noop": {"decision": "accept", "outcome_available": True},
            "counterfactual": {"decision": "reject", "outcome_available": True},
            "criterion_deltas": [],
            "decision_control_stable": True,
            "notes": [],
        }
        arts = [
            {
                "ref": "chunk_abc:v1",
                "artifact_type": "EvidenceChunk",
                "content_preview": "initial placement of k means",
            }
        ]
        lines = verbalize_replay_comparison(cmp, 1, relevant_artifacts=arts)
        joined = "\n".join(lines)
        self.assertIn("tested", joined.lower())
        self.assertIn("final decision changed", joined.lower())
        self.assertNotIn("NOOP", joined)
        self.assertNotIn("chunk_abc", joined)
        self.assertNotIn("decision_effect", joined)
        self.assertIn("initial placement of k means", joined.lower())

    def test_no_demonstrated_effect_wording(self):
        cmp = {
            "evidence_status": "no_demonstrated_effect",
            "intervention_signature": {
                "operation": "REMOVE",
                "target_ref": {"key": "E1:v1"},
            },
            "factual": {"decision": "accept"},
            "noop": {"decision": "accept"},
            "counterfactual": {"decision": "accept"},
            "criterion_deltas": [],
            "notes": [],
        }
        joined = "\n".join(verbalize_replay_comparison(cmp, 1))
        self.assertIn("remained accept", joined.lower())
        self.assertNotIn("the source was irrelevant", joined.lower())
        self.assertNotIn("does not matter", joined.lower())
        self.assertNotIn("NOOP", joined)

    def test_replay_unstable(self):
        cmp = {
            "evidence_status": "replay_unstable",
            "intervention_signature": {
                "operation": "REMOVE",
                "target_ref": {"key": "E1:v1"},
            },
            "factual": {"decision": "accept"},
            "noop": {"decision": "reject"},
            "counterfactual": {"decision": "reject"},
            "criterion_deltas": [],
            "notes": [],
        }
        # Unstable comparisons are limitations, not causal findings.
        lines = verbalize_replay_comparison(cmp, 1)
        self.assertEqual(lines, ())


class TestNoAxisRawLeakage(unittest.TestCase):
    def test_synthesis_strips_final_explanation_raw(self):
        class _Fake:
            replay_comparisons = []
            intervention_records = []
            round_records = []
            rounds = 2
            termination_reason = "done"
            warnings = ()
            final_explanation = "SECRET AXIS PROSE THAT MUST NOT ENTER SYNTHESIS"

        raw = build_axis_evidence_summary(explanation_result=_Fake())
        self.assertIn("final_explanation_raw", raw)
        cleaned = synthesis_evidence_without_axis_raw(raw)
        self.assertNotIn("final_explanation_raw", cleaned)
        view = build_evidence_view(condition="axis", raw_evidence=cleaned)
        prompt = build_shared_final_prompt(
            user_question="Why?",
            factual_decision="accept",
            question_stem="stem",
            verbalized_evidence=view.to_prompt_text(),
        )
        self.assertNotIn("SECRET AXIS PROSE", prompt)


class TestAcceptSemantics(unittest.TestCase):
    def test_accept_is_about_candidate(self):
        steps = verbalize_process_steps(
            ["apply_acceptance_gate#1"], factual_decision="accept"
        )
        joined = " ".join(steps).lower()
        # Trace purpose is educational/process, not the ACCEPT label itself.
        self.assertIn("question", joined)
        self.assertTrue(
            "criteria" in joined or "learning" in joined or "evaluated" in joined,
            joined,
        )


class TestNoConditionNamesInSynthesis(unittest.TestCase):
    def test_mechanical_output_has_no_condition_labels(self):
        view = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            observed_process=("The candidate question was generated.",),
            limitations=(
                "The execution record shows which steps occurred, but it does not "
                "show which specific source items were used for this question or "
                "whether an individual input changed the decision.",
            ),
        )
        text = synthesize_explanation(
            user_question="Why?",
            factual_decision="accept",
            question_stem="stem",
            evidence_view=view,
            llm_generate=None,
        )
        self.assertNotIn("Trace", text)
        self.assertNotIn("AXIS", text)
        self.assertNotIn("experimental condition", text.lower())
        self.assertNotIn("Why this question?", text)


class TestMinimalTraceNoFiller(unittest.TestCase):
    def test_short_grounded_trace(self):
        view = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            observed_process=(
                "The candidate question was generated.",
                "The system recorded the final decision for the candidate question: ACCEPT.",
            ),
            limitations=(
                "The execution record shows which steps occurred, but it does not "
                "show which specific source items were used for this question or "
                "whether an individual input changed the decision.",
            ),
            lay_facts=LayExplanationFacts(
                decision="ACCEPT",
                educational_purpose=(
                    "The candidate question was generated.",
                ),
                limitations=(
                    "The execution record alone does not identify which specific "
                    "teaching information explains why this particular question "
                    "was selected.",
                ),
                source_basis_unavailable=True,
            ),
        )
        text = synthesize_explanation(
            user_question="Why?",
            factual_decision="accept",
            question_stem="stem",
            evidence_view=view,
            llm_generate=None,
        )
        self.assertLess(len(text.split()), 160)
        self.assertNotIn("structured process", text.lower())
        self.assertNotIn("external factors", text.lower())
        ok, warns = validate_explanation_faithfulness(
            condition="trace", explanation=text, evidence_view=view
        )
        self.assertTrue(ok, msg=str(warns))


class TestFaithfulnessRejectsTraceCausal(unittest.TestCase):
    def test_trace_causal_flagged(self):
        view = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            observed_process=("The candidate question was generated.",),
        )
        bad = (
            "Why this question?\n"
            "The outline step caused the acceptance and was decisive.\n\n"
            "What was it based on?\n"
            "Unknown.\n\n"
            "What difference did this information make?\n"
            "ACCEPT."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="trace", explanation=bad, evidence_view=view
        )
        self.assertFalse(ok)
        self.assertTrue(any("causal" in w for w in warns))


class TestAxisViewIncludesInterventions(unittest.TestCase):
    def test_axis_view_populated(self):
        raw = {
            "kind": "axis",
            "factual_decision": "accept",
            "execution_trace": ["apply_acceptance_gate#1"],
            "relevant_artifacts": [],
            "provenance_relations": [],
            "replay_comparisons": [
                {
                    "evidence_status": "decision_effect",
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "cross-cove-report:q1:v1"},
                    },
                    "factual": {"decision": "accept"},
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "reject"},
                    "criterion_deltas": [],
                    "decision_control_stable": True,
                    "notes": [],
                }
            ],
            "final_explanation_raw": "should be stripped before build_evidence_view by caller",
        }
        cleaned = synthesis_evidence_without_axis_raw(raw)
        view = build_evidence_view(condition="axis", raw_evidence=cleaned)
        self.assertTrue(view.tested_interventions)
        joined = "\n".join(view.tested_interventions)
        self.assertIn("final decision changed", joined.lower())
        self.assertIn("tested", joined.lower())
        self.assertNotIn("decision_effect", joined)
        self.assertNotIn("NOOP", joined)

if __name__ == "__main__":
    unittest.main()
