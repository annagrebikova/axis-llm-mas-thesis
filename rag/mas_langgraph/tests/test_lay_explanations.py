"""Human-centered lay explanation selection / aggregation tests."""

from __future__ import annotations

import unittest

from mas_langgraph.axis.comparison import (
    CaseRunObservation,
    CriterionDelta,
    ReplayComparator,
    REPLAY_UNSTABLE,
    DOWNSTREAM_EFFECT,
)
from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.models import ExplainedCase, QuerySignature
from mas_langgraph.core.outcomes import CriterionValue
from mas_langgraph.experiment.lay_facts import (
    build_lay_explanation_facts,
    group_interventions,
    is_metadata_only_snippet,
    select_source_facts,
)
from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
from mas_langgraph.experiment.synthesis import (
    TARGET_WORD_RANGE,
    mechanical_synthesize_from_view,
    synthesize_explanation,
)
from mas_langgraph.experiment.verbalizer import (
    build_evidence_view,
    participant_section_headings,
)
from mas_langgraph.replay.executor import WorkflowRunResult


_META = (
    "Unsupervised Learning\n"
    "George Konidaris\n"
    "gdk@cs.duke.edu\n"
    "Spring 2016"
)
_ATOM1 = (
    "Unsupervised learning is a type of machine learning where the model is "
    "trained on unlabeled data without any specific output to learn from."
)
_ATOM2 = (
    "Unsupervised learning algorithms try to learn the patterns and structure "
    "within the data without any guidance."
)


def _arts_with_meta_and_atoms():
    return [
        {
            "ref": "sec_1:v1",
            "artifact_type": "DocSection",
            "title": "Unsupervised Learning",
        },
        {
            "ref": "lo_1:v1",
            "artifact_type": "LearningObjective",
            "objective_text": "Identify the key concepts of unsupervised learning",
        },
        {
            "ref": "chunk_meta:v1",
            "artifact_type": "EvidenceChunk",
            "content": _META,
            "content_preview": " ".join(_META.split()),
        },
        {
            "ref": "atom_1:v1",
            "artifact_type": "KnowledgeAtom",
            "content": _ATOM1,
            "content_preview": _ATOM1,
        },
        {
            "ref": "atom_2:v1",
            "artifact_type": "KnowledgeAtom",
            "content": _ATOM2,
            "content_preview": _ATOM2,
        },
    ]


class TestMetadataFiltering(unittest.TestCase):
    def test_title_card_is_metadata(self):
        self.assertTrue(is_metadata_only_snippet(_META))
        self.assertFalse(is_metadata_only_snippet(_ATOM1))

    def test_prefers_atoms_over_metadata_chunk(self):
        facts, discarded = select_source_facts(_arts_with_meta_and_atoms(), max_facts=2)
        joined = " ".join(facts).lower()
        self.assertIn("unlabeled data", joined)
        self.assertNotIn("gdk@cs.duke.edu", joined)
        self.assertTrue(any("gdk@" in d or "Spring 2016" in d for d in discarded))


class TestMaxTwoSourceFacts(unittest.TestCase):
    def test_caps_at_two(self):
        arts = []
        for i in range(8):
            text = (
                f"Unsupervised learning uses unlabeled data in setting number {i} "
                f"and learns structure without guidance from labels."
            )
            arts.append(
                {
                    "ref": f"atom_{i}:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content": text,
                    "content_preview": text,
                }
            )
        facts, _ = select_source_facts(arts, max_facts=2)
        self.assertLessEqual(len(facts), 2)


class TestInterventionGrouping(unittest.TestCase):
    def test_four_removes_one_group(self):
        cmps = []
        for i, status in enumerate(
            [
                "no_demonstrated_effect",
                "no_demonstrated_effect",
                "downstream_effect",
                "no_demonstrated_effect",
            ]
        ):
            cmps.append(
                {
                    "evidence_status": status,
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "chunk_meta:v1"},
                    },
                }
            )
        groups = group_interventions(cmps, artifacts=_arts_with_meta_and_atoms())
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].valid_test_count, 4)
        self.assertEqual(groups[0].strongest_effect(), "downstream_effect")

        lay = build_lay_explanation_facts(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": _arts_with_meta_and_atoms(),
                "replay_comparisons": cmps,
                "execution_trace": [],
            },
        )
        self.assertEqual(lay.raw_intervention_count, 4)
        self.assertEqual(lay.grouped_intervention_count, 1)
        text = " ".join(lay.controlled_change_findings).lower()
        self.assertIn("4 points", text)
        self.assertNotIn("gdk@cs.duke.edu", text)
        # Should not repeat four nearly identical tested sentences.
        self.assertEqual(text.count("the system tested removing"), 1)


class TestEffectRanking(unittest.TestCase):
    def test_decision_effect_preferred(self):
        cmps = [
            {
                "evidence_status": "no_demonstrated_effect",
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "chunk_a:v1"},
                },
            },
            {
                "evidence_status": "downstream_effect",
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "chunk_b:v1"},
                },
            },
            {
                "evidence_status": "decision_effect",
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "chunk_c:v1"},
                },
            },
        ]
        lay = build_lay_explanation_facts(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": [],
                "replay_comparisons": cmps,
            },
        )
        joined = " ".join(lay.controlled_change_findings).lower()
        self.assertIn("final decision changed", joined)


class TestUnstableControl(unittest.TestCase):
    def test_criteria_mismatch_is_replay_unstable(self):
        factual = CaseRunObservation(
            run_id="r",
            branch_id="factual",
            decision="accept",
            criterion_values=(
                CriterionValue(name="supported_by_pdf", value=True, decision_relevant=True),
            ),
            subject_present=True,
        )

        class _Adapter:
            def observe_replay(self, *, explained_case, run_result):
                if run_result.branch_id == "noop":
                    return CaseRunObservation(
                        run_id="r",
                        branch_id="noop",
                        decision="accept",
                        criterion_values=(
                            CriterionValue(
                                name="supported_by_pdf",
                                value=False,
                                decision_relevant=True,
                            ),
                        ),
                        subject_present=True,
                    )
                return CaseRunObservation(
                    run_id="r",
                    branch_id="cf",
                    decision="accept",
                    criterion_values=(
                        CriterionValue(
                            name="supported_by_pdf",
                            value=False,
                            decision_relevant=True,
                        ),
                    ),
                    subject_present=True,
                )

        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q1", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("d", 1),
            outcome="accept",
        )
        cmp = ReplayComparator().compare(
            signature=QuerySignature(
                operation="REMOVE", target_ref=ArtifactRef("E1", 1)
            ),
            factual_observation=factual,
            noop_result=WorkflowRunResult(
                run_id="r", branch_id="noop", final_state={"branch_id": "noop"}
            ),
            counterfactual_result=WorkflowRunResult(
                run_id="r", branch_id="cf", final_state={"branch_id": "cf"}
            ),
            observation_adapter=_Adapter(),
            explained_case=explained,
        )
        self.assertEqual(cmp.evidence_status, REPLAY_UNSTABLE)
        self.assertFalse(cmp.criteria_control_stable)

        lay = build_lay_explanation_facts(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "replay_comparisons": [
                    {
                        "evidence_status": "replay_unstable",
                        "intervention_signature": {
                            "operation": "REMOVE",
                            "target_ref": {"key": "E1:v1"},
                        },
                    }
                ],
                "relevant_artifacts": [],
            },
        )
        self.assertEqual(lay.controlled_change_findings, ())
        self.assertTrue(any("no reliable" in x.lower() or "no causal" in x.lower() or "not produce" in x.lower() or "could be drawn" in x.lower() for x in lay.limitations))


class TestConditionAdvantages(unittest.TestCase):
    def test_trace_no_source_content_or_cf(self):
        view = build_evidence_view(
            condition="trace",
            raw_evidence={
                "factual_decision": "accept",
                "execution_trace": [
                    "build_outline#1",
                    "generate_candidate_mcqs_from_blueprints#1",
                    "apply_acceptance_gate#1",
                ],
                "relevant_artifacts": _arts_with_meta_and_atoms(),
                "replay_comparisons": [
                    {"evidence_status": "decision_effect", "intervention_signature": {}}
                ],
            },
        )
        text = view.to_prompt_text().lower()
        self.assertNotIn("unlabeled data", text)
        self.assertFalse(view.tested_interventions)
        self.assertEqual(view.lay_facts.selected_source_fact_count, 0)

    def test_provenance_richer_than_trace(self):
        trace = build_lay_explanation_facts(
            condition="trace",
            raw_evidence={"factual_decision": "accept", "execution_trace": ["gen#1"]},
        )
        tp = build_lay_explanation_facts(
            condition="trace_provenance",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": _arts_with_meta_and_atoms(),
                "execution_trace": ["gen#1"],
            },
        )
        self.assertGreater(len(tp.source_basis), 0)
        self.assertTrue(
            any(
                tok in s.lower()
                for s in tp.source_basis
                for tok in ("unlabeled", "patterns", "structure", "guidance")
            ),
            tp.source_basis,
        )
        self.assertFalse(any("unlabeled" in s.lower() for s in trace.source_basis))

    def test_axis_has_controlled_finding(self):
        tp = build_lay_explanation_facts(
            condition="trace_provenance",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": _arts_with_meta_and_atoms(),
            },
        )
        axis = build_lay_explanation_facts(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": _arts_with_meta_and_atoms(),
                "replay_comparisons": [
                    {
                        "evidence_status": "downstream_effect",
                        "intervention_signature": {
                            "operation": "REMOVE",
                            "target_ref": {"key": "chunk_meta:v1"},
                        },
                    }
                ]
                * 4,
            },
        )
        self.assertFalse(tp.controlled_change_findings)
        self.assertTrue(axis.controlled_change_findings)


class TestSharedHeadingsAndFallback(unittest.TestCase):
    def test_headings(self):
        # Legacy helper retained; participant output no longer requires headings.
        self.assertEqual(len(participant_section_headings()), 3)
        self.assertEqual(TARGET_WORD_RANGE, (1, 200))
        from mas_langgraph.experiment.synthesis import MAX_PARTICIPANT_WORDS
        self.assertEqual(MAX_PARTICIPANT_WORDS, 200)

    def test_no_duplicate_fallback(self):
        view = build_evidence_view(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": _arts_with_meta_and_atoms(),
                "replay_comparisons": [
                    {
                        "evidence_status": "downstream_effect",
                        "intervention_signature": {
                            "operation": "REMOVE",
                            "target_ref": {"key": "chunk_meta:v1"},
                        },
                    }
                ]
                * 4,
            },
        )
        text = mechanical_synthesize_from_view(view=view)
        self.assertNotIn("Why this question?", text)
        self.assertEqual(text.lower().count("the system tested removing"), 1)
        self.assertNotIn("gdk@cs.duke.edu", text)
        self.assertNotIn("EvidenceChunk", text)
        self.assertNotIn("NOOP", text)
        ok, warns = validate_explanation_faithfulness(
            condition="axis", explanation=text, evidence_view=view
        )
        self.assertTrue(ok, msg=str(warns))

    def test_no_jargon_in_all_conditions(self):
        for cond in ("trace", "trace_provenance", "axis"):
            raw = {
                "factual_decision": "accept",
                "execution_trace": ["generate_candidate_mcqs_from_blueprints#1"],
                "relevant_artifacts": _arts_with_meta_and_atoms(),
                "replay_comparisons": [
                    {
                        "evidence_status": "no_demonstrated_effect",
                        "intervention_signature": {
                            "operation": "REMOVE",
                            "target_ref": {"key": "atom_1:v1"},
                        },
                    }
                ]
                if cond == "axis"
                else [],
            }
            view = build_evidence_view(condition=cond, raw_evidence=raw)
            text = synthesize_explanation(
                user_question="Why?",
                factual_decision="accept",
                question_stem="What is unsupervised learning?",
                evidence_view=view,
                llm_generate=None,
            )
            for bad in (
                "EvidenceChunk",
                "KnowledgeAtom",
                "MCQBlueprint",
                "Cross-CoVe",
                "NOOP",
                "ArtifactRef",
                "downstream_effect",
            ):
                self.assertNotIn(bad, text)


if __name__ == "__main__":
    unittest.main()
