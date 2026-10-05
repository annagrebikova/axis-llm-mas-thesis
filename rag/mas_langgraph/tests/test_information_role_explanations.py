"""Information-role explanations, semantic validity, interrogation ordering."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.explainer import _sanitize_raw_axis_summary
from mas_langgraph.axis.intervention_guidance import (
    InterventionCandidate,
    format_executable_queries,
)
from mas_langgraph.cases.mcq.content_comparison import (
    compare_mcq_subject_content,
    prefer_pre_generation_boundaries,
)
from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
from mas_langgraph.experiment.information_role import (
    ROLE_DISTRACTOR_SHAPING,
    ROLE_FOCUS_SHAPING,
    ROLE_INFLUENTIAL_NON_DECISIVE,
    ROLE_NECESSARY_FOR_REPRODUCTION,
    build_information_role_finding,
    changed_distractor_count_from_lists,
    infer_semantic_intervention_validity,
    resolve_interrogation_goal,
)
from mas_langgraph.experiment.lay_facts import LayExplanationFacts, build_lay_explanation_facts
from mas_langgraph.experiment.question_content_facts import (
    verbalize_question_content_comparison,
)
from mas_langgraph.experiment.verbalizer import ExplanationEvidenceView
from mas_langgraph.tests.test_mcq_content_effects import _cand, _obs


class TestDistractorCardinality(unittest.TestCase):
    def test_changed_distractor_count_two(self):
        n = changed_distractor_count_from_lists(
            removed=["B", "C"],
            added=["D", "E"],
        )
        self.assertEqual(n, 2)

        base = _cand(
            stem="Q?",
            correct="A",
            distractors=["A1", "B", "C"],
        )
        # Multiset: A1 kept; B,C replaced by D,E → count 2
        cf = _cand(
            stem="Q?",
            correct="A",
            distractors=["A1", "D", "E"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(base),
            noop=_obs(base),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        self.assertEqual(delta.details["changed_distractor_count"], 2)
        self.assertEqual(delta.details["distractor_effect"], "changed")

        cmp = {
            "intervention_before_subject_generation": True,
            "question_content_comparison": delta.to_dict(),
            "noop": {"decision": "accept"},
            "counterfactual": {"decision": "accept"},
            "intervention_signature": {
                "operation": "REMOVE",
                "target_ref": {"key": "atom_1:v1"},
                "checkpoint_id": "before plan_mcq_blueprints#1",
            },
        }
        lines, meta = verbalize_question_content_comparison(cmp)
        joined = " ".join(lines).lower()
        self.assertEqual(meta["changed_distractor_count"], 2)
        self.assertNotIn("one incorrect", joined)
        self.assertIn("incorrect alternatives", joined)

        facts = LayExplanationFacts(
            decision="ACCEPT",
            question_content_findings=lines,
            primary_question_content_status="question_content_changed",
            stem_changed=False,
            correct_answer_changed=False,
            distractors_changed=True,
            changed_distractor_count=2,
            distractor_effect="changed",
        )
        view = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            lay_facts=facts,
            tested_interventions=("x",),
        )
        ok, warnings = validate_explanation_faithfulness(
            condition="axis",
            explanation=(
                "Why this question?\n...\nWhat was it based on?\n...\n"
                "What difference did this information make?\n"
                "One incorrect option changed."
            ),
            evidence_view=view,
        )
        self.assertFalse(ok)
        self.assertTrue(any("cardinality" in w for w in warnings), warnings)

        ok2, w2 = validate_explanation_faithfulness(
            condition="axis",
            explanation=(
                "Why this question?\n...\nWhat was it based on?\n...\n"
                "What difference did this information make?\n"
                "Some incorrect alternatives were influenced by this information."
            ),
            evidence_view=view,
        )
        # May still warn on other coverage — cardinality itself must not fire.
        self.assertFalse(any("cardinality" in w for w in w2), w2)


class TestSameStemNotSameMcq(unittest.TestCase):
    def test_reject_same_question_claim(self):
        facts = LayExplanationFacts(
            decision="ACCEPT",
            question_content_findings=(
                "This information helps shape the incorrect alternatives.",
            ),
            primary_question_content_status="question_content_changed",
            stem_changed=False,
            correct_answer_changed=False,
            distractors_changed=True,
            changed_distractor_count=2,
        )
        view = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            lay_facts=facts,
            tested_interventions=("x",),
        )
        ok, warnings = validate_explanation_faithfulness(
            condition="axis",
            explanation=(
                "Why this question?\n...\nWhat was it based on?\n...\n"
                "What difference did this information make?\n"
                "The question stayed the same."
            ),
            evidence_view=view,
        )
        self.assertFalse(ok)
        self.assertTrue(any("same stem" in w.lower() or "same MCQ" in w or "same question" in w for w in warnings), warnings)

        ok2, w2 = validate_explanation_faithfulness(
            condition="axis",
            explanation=(
                "Why this question?\n...\nWhat was it based on?\n...\n"
                "What difference did this information make?\n"
                "The question continued to ask the same thing, but the incorrect "
                "alternatives were different. This information helps shape the "
                "incorrect alternatives."
            ),
            evidence_view=view,
        )
        self.assertFalse(any("same stem" in w.lower() or "same MCQ" in w for w in w2), w2)


class TestNoParticipantMechanics(unittest.TestCase):
    def test_reject_remove_and_noop(self):
        facts = LayExplanationFacts(
            decision="ACCEPT",
            question_content_findings=("role line",),
            primary_question_content_status="question_content_changed",
            distractors_changed=True,
            stem_changed=False,
            correct_answer_changed=False,
        )
        view = ExplanationEvidenceView(
            factual_decision="ACCEPT",
            lay_facts=facts,
            tested_interventions=("x",),
        )
        _, warnings = validate_explanation_faithfulness(
            condition="axis",
            explanation=(
                "Why this question?\n...\nWhat was it based on?\n...\n"
                "What difference did this information make?\n"
                "REMOVE(chunk_1:v1) on a counterfactual branch with NOOP checkpoint replay."
            ),
            evidence_view=view,
        )
        self.assertTrue(any("mechanics" in w for w in warnings), warnings)


class TestSemanticPersistence(unittest.TestCase):
    def test_chunk_after_atomize_artifact_only(self):
        sem = infer_semantic_intervention_validity(
            artifact_type="EvidenceChunk",
            intervention_before_generation=True,
            boundary_label="before plan_mcq_blueprints#1",
            provenance_relations=[
                SimpleNamespace(
                    relation="evidence_to_atomize",
                    artifact_ref=ArtifactRef("chunk_00000002", 1),
                )
            ],
            target_ref="chunk_00000002:v1",
        )
        self.assertFalse(sem["semantic_information_absent"])
        self.assertTrue(sem["derived_information_may_persist"])
        self.assertEqual(sem["intervention_scope"], "artifact_only")

    def test_chunk_before_atomize_semantic(self):
        sem = infer_semantic_intervention_validity(
            artifact_type="EvidenceChunk",
            intervention_before_generation=True,
            boundary_label="before atomize_evidence#1",
            provenance_relations=[
                SimpleNamespace(
                    relation="evidence_to_atomize",
                    artifact_ref=ArtifactRef("chunk_00000002", 1),
                )
            ],
            target_ref="chunk_00000002:v1",
        )
        self.assertTrue(sem["semantic_information_absent"])
        self.assertFalse(sem["derived_information_may_persist"])
        self.assertEqual(sem["intervention_scope"], "semantic_information")


class TestKnowledgeAtomRoles(unittest.TestCase):
    def test_focus_shaping_and_distractor_shaping(self):
        base = _cand(stem="Purpose of unsupervised?", correct="a", distractors=["b", "c", "d"])
        cf_stem = _cand(stem="Advantage of clustering?", correct="a", distractors=["b", "c", "d"])
        delta_stem = compare_mcq_subject_content(
            factual=_obs(base), noop=_obs(base), counterfactual=_obs(cf_stem),
            intervention_before_subject_generation=True,
        )
        finding = build_information_role_finding(
            {
                "intervention_before_subject_generation": True,
                "question_content_comparison": delta_stem.to_dict(),
                "noop": {"decision": "accept"},
                "counterfactual": {"decision": "accept"},
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "atom_1:v1"},
                    "checkpoint_id": "before plan_mcq_blueprints#1",
                },
            }
        )
        assert finding is not None
        self.assertEqual(finding.role_type, ROLE_FOCUS_SHAPING)
        self.assertEqual(finding.causal_classification, ROLE_INFLUENTIAL_NON_DECISIVE)

        cf_dist = _cand(
            stem="Purpose of unsupervised?",
            correct="a",
            distractors=["b", "x", "y"],
        )
        delta_d = compare_mcq_subject_content(
            factual=_obs(base), noop=_obs(base), counterfactual=_obs(cf_dist),
            intervention_before_subject_generation=True,
        )
        finding2 = build_information_role_finding(
            {
                "intervention_before_subject_generation": True,
                "question_content_comparison": delta_d.to_dict(),
                "noop": {"decision": "accept"},
                "counterfactual": {"decision": "accept"},
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "atom_2:v1"},
                    "checkpoint_id": "before plan_mcq_blueprints#1",
                },
            }
        )
        assert finding2 is not None
        self.assertEqual(finding2.role_type, ROLE_DISTRACTOR_SHAPING)
        joined = " ".join(finding2.participant_lines).lower()
        self.assertNotIn("irrelevant", joined)


class TestNecessityAndInfluential(unittest.TestCase):
    def test_necessary_only_when_not_reproduced(self):
        c = _cand(stem="Q?", correct="a", distractors=["b", "c", "d"])
        delta = compare_mcq_subject_content(
            factual=_obs(c),
            noop=_obs(c),
            counterfactual=_obs(None, present=False),
            intervention_before_subject_generation=True,
        )
        finding = build_information_role_finding(
            {
                "intervention_before_subject_generation": True,
                "question_content_comparison": delta.to_dict(),
                "noop": {"decision": "accept"},
                "counterfactual": {"decision": None},
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "atom_1:v1"},
                    "checkpoint_id": "before plan_mcq_blueprints#1",
                },
            }
        )
        assert finding is not None
        self.assertEqual(finding.role_type, ROLE_NECESSARY_FOR_REPRODUCTION)

        # Stem change alone is not necessity.
        cf = _cand(stem="Other?", correct="a", distractors=["b", "c", "d"])
        delta2 = compare_mcq_subject_content(
            factual=_obs(c), noop=_obs(c), counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        finding2 = build_information_role_finding(
            {
                "intervention_before_subject_generation": True,
                "question_content_comparison": delta2.to_dict(),
                "noop": {"decision": "accept"},
                "counterfactual": {"decision": "accept"},
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "atom_1:v1"},
                    "checkpoint_id": "before plan_mcq_blueprints#1",
                },
            }
        )
        assert finding2 is not None
        self.assertNotEqual(finding2.role_type, ROLE_NECESSARY_FOR_REPRODUCTION)


class TestRawAxisConsistency(unittest.TestCase):
    def test_sanitize_no_question_effect_claim(self):
        scd = SimpleNamespace(
            effect_status="question_content_changed",
            details={
                "distractor_effect": "changed",
                "stem_effect": "unchanged",
                "correct_answer_effect": "unchanged",
            },
        )
        comp = SimpleNamespace(subject_content_delta=scd)
        out = _sanitize_raw_axis_summary(
            "There was no demonstrated effect on the question.",
            comparisons=[comp],
        )
        self.assertNotIn("no demonstrated effect on the question", out.lower())
        self.assertIn("decision effect", out.lower())


class TestInterrogationPolicy(unittest.TestCase):
    def test_question_content_goal_and_ordering(self):
        self.assertEqual(
            resolve_interrogation_goal(
                "Why did the system generate this question, and would it have been "
                "different without the supporting information?"
            ),
            "question_content",
        )
        explained = SimpleNamespace(artifact_ref=ArtifactRef("q_1", 1))
        cands = (
            InterventionCandidate(
                ref=ArtifactRef("q_1", 1),
                artifact_type="CandidateMCQ",
                allowed_operations=("REMOVE",),
                valid_boundaries=("before verify#1",),
                operation_boundaries={"REMOVE": ("before verify#1",)},
                priority=4,
            ),
            InterventionCandidate(
                ref=ArtifactRef("atom_1", 1),
                artifact_type="KnowledgeAtom",
                allowed_operations=("REMOVE",),
                valid_boundaries=("before plan_mcq_blueprints#1",),
                operation_boundaries={
                    "REMOVE": ("before plan_mcq_blueprints#1",)
                },
                priority=2,
            ),
            InterventionCandidate(
                ref=ArtifactRef("chunk_1", 1),
                artifact_type="EvidenceChunk",
                allowed_operations=("REMOVE",),
                valid_boundaries=(
                    "before atomize_evidence#1",
                    "before plan_mcq_blueprints#1",
                ),
                operation_boundaries={
                    "REMOVE": (
                        "before atomize_evidence#1",
                        "before plan_mcq_blueprints#1",
                    )
                },
                priority=1,
            ),
            InterventionCandidate(
                ref=ArtifactRef("bp_1", 1),
                artifact_type="MCQBlueprint",
                allowed_operations=("REMOVE",),
                valid_boundaries=("before generate_candidate_mcqs#1",),
                operation_boundaries={
                    "REMOVE": ("before generate_candidate_mcqs#1",)
                },
                priority=3,
            ),
        )
        text = format_executable_queries(
            candidates=cands,
            explained_case=explained,
            interrogation_goal="question_content",
            max_queries=8,
        )
        lines = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip()
            and not ln.startswith("Executable")
            and not ln.startswith("Return")
            and not ln.startswith("Preserve")
            and ln.strip() != "DONE"
        ]
        # First actionable should be upstream REMOVE, not CandidateMCQ / WHAT(candidate).
        self.assertTrue(lines[0].startswith("REMOVE("), lines)
        self.assertTrue(
            "chunk_1" in lines[0] or "atom_1" in lines[0],
            lines,
        )
        idx_cand_remove = next(
            (i for i, ln in enumerate(lines) if "REMOVE(q_1" in ln),
            len(lines),
        )
        idx_atom = next(i for i, ln in enumerate(lines) if "REMOVE(atom_1" in ln)
        self.assertLess(idx_atom, idx_cand_remove)

    def test_no_what_before_untested_atom(self):
        cands = (
            InterventionCandidate(
                ref=ArtifactRef("atom_1", 1),
                artifact_type="KnowledgeAtom",
                allowed_operations=("REMOVE",),
                valid_boundaries=("before plan_mcq_blueprints#1",),
                operation_boundaries={
                    "REMOVE": ("before plan_mcq_blueprints#1",)
                },
                priority=2,
            ),
        )
        text = format_executable_queries(
            candidates=cands,
            explained_case=SimpleNamespace(artifact_ref=ArtifactRef("q_1", 1)),
            interrogation_goal="question_content",
            max_queries=5,
        )
        lines = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip().startswith(("WHAT", "REMOVE", "WHATIF"))
        ]
        self.assertTrue(lines[0].startswith("REMOVE("), lines)
        self.assertFalse(lines[0].startswith("WHAT("), lines)

    def test_whatif_only_with_controlled_alternative(self):
        cands = (
            InterventionCandidate(
                ref=ArtifactRef("atom_1", 1),
                artifact_type="KnowledgeAtom",
                allowed_operations=("REMOVE", "REPLACE"),
                valid_boundaries=("before plan_mcq_blueprints#1",),
                operation_boundaries={
                    "REMOVE": ("before plan_mcq_blueprints#1",),
                    "REPLACE": ("before plan_mcq_blueprints#1",),
                },
                priority=2,
            ),
        )
        # No alternatives → no WHATIF.
        text_no = format_executable_queries(
            candidates=cands,
            interrogation_goal="question_content",
            intervention_alternatives={},
            registry=None,
        )
        self.assertNotIn("WHATIF(", text_no)

        reg = MagicMock()
        env = MagicMock()
        env.artifact_type = "KnowledgeAtom"
        reg.get_by_ref.return_value = env

        text_yes = format_executable_queries(
            candidates=cands,
            interrogation_goal="question_content",
            intervention_alternatives={"alt": ArtifactRef("atom_2", 1)},
            registry=reg,
        )
        self.assertIn("WHATIF(atom_1:v1, atom_2:v1", text_yes)

    def test_boundary_preference_artifact_aware(self):
        bounds = prefer_pre_generation_boundaries(
            (
                "before plan_mcq_blueprints#1",
                "before atomize_evidence#1",
                "before generate_candidate_mcqs#1",
            ),
            artifact_type="EvidenceChunk",
        )
        self.assertIn("atomize", bounds[0].lower())


class TestLayFactsRoleMetadata(unittest.TestCase):
    def test_role_diagnostics_exported(self):
        base = _cand(stem="Q?", correct="a", distractors=["b", "c", "d"])
        cf = _cand(stem="Q?", correct="a", distractors=["b", "x", "y"])
        delta = compare_mcq_subject_content(
            factual=_obs(base), noop=_obs(base), counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        facts = build_lay_explanation_facts(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "replay_comparisons": [
                    {
                        "evidence_status": "no_demonstrated_effect",
                        "intervention_before_subject_generation": True,
                        "question_content_comparison": delta.to_dict(),
                        "noop": {"decision": "accept"},
                        "counterfactual": {"decision": "accept"},
                        "intervention_signature": {
                            "operation": "REMOVE",
                            "target_ref": {"key": "atom_9:v1"},
                            "checkpoint_id": "before plan_mcq_blueprints#1",
                        },
                    }
                ],
            },
        )
        self.assertEqual(facts.information_role, ROLE_DISTRACTOR_SHAPING)
        self.assertTrue(facts.semantic_information_absent)
        self.assertEqual(facts.changed_distractor_count, 2)
        self.assertFalse(facts.decision_effect)
        self.assertTrue(facts.content_effect)


if __name__ == "__main__":
    unittest.main()
