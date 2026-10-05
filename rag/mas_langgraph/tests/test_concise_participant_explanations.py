"""Concise selection-focused participant explanations (no mandatory headings)."""

from __future__ import annotations

import unittest

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.export import _signature_dict
from mas_langgraph.axis.intervention_guidance import (
    InterventionCandidate,
    executable_query_forms,
    format_executable_queries,
)
from mas_langgraph.axis.models import QuerySignature, ResolvedQuery
from mas_langgraph.cases.mcq.content_comparison import compare_mcq_subject_content
from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
from mas_langgraph.experiment.information_role import (
    ROLE_FOCUS_SHAPING,
    ROLE_FORMULATION_SHAPING,
    build_information_role_finding,
    classify_stem_semantic_relation,
    infer_semantic_intervention_validity,
    resolve_interrogation_goal,
)
from mas_langgraph.experiment.lay_facts import (
    LayExplanationFacts,
    build_lay_explanation_facts,
    select_source_facts,
)
from mas_langgraph.experiment.synthesis import (
    SHARED_FINAL_SYNTHESIS_PROMPT,
    mechanical_synthesize_from_lay_facts,
)
from mas_langgraph.experiment.verbalizer import ExplanationEvidenceView
from mas_langgraph.tests.test_mcq_content_effects import _cand, _obs


class TestSelectionQuestion(unittest.TestCase):
    def test_goal_question_selection(self):
        self.assertEqual(
            resolve_interrogation_goal("Why was this question selected?"),
            "question_selection",
        )
        self.assertEqual(
            resolve_interrogation_goal("Why was this question chosen?"),
            "question_selection",
        )

    def test_prompt_selection_guidance(self):
        self.assertIn("Do not create fixed sections", SHARED_FINAL_SYNTHESIS_PROMPT)
        self.assertIn("Do NOT enumerate workflow stages", SHARED_FINAL_SYNTHESIS_PROMPT)
        self.assertIn("Do not invent missing levels of evidence", SHARED_FINAL_SYNTHESIS_PROMPT)
        self.assertIn("Write 1 sentence", SHARED_FINAL_SYNTHESIS_PROMPT)
        self.assertIn("2–3 compact sentences", SHARED_FINAL_SYNTHESIS_PROMPT)
        self.assertNotIn("condition ==", SHARED_FINAL_SYNTHESIS_PROMPT)


class TestNoMandatoryHeadings(unittest.TestCase):
    def test_mechanical_trace_no_headings(self):
        facts = build_lay_explanation_facts(
            condition="trace",
            raw_evidence={
                "factual_decision": "accept",
                "execution_trace": ["generate_candidate_mcqs#1"],
            },
        )
        text = mechanical_synthesize_from_lay_facts(facts=facts)
        self.assertNotIn("Why this question?", text)
        self.assertNotIn("What was it based on?", text)
        self.assertNotIn("would have been different", text.lower())
        from mas_langgraph.experiment.synthesis import count_participant_sentences
        self.assertEqual(count_participant_sentences(text), 1)
        view = ExplanationEvidenceView(factual_decision="ACCEPT", lay_facts=facts)
        ok, warns = validate_explanation_faithfulness(
            condition="trace",
            explanation=text,
            evidence_view=view,
        )
        self.assertTrue(ok, warns)


class TestTraceConcise(unittest.TestCase):
    def test_trace_selection_limitation_rejects_role(self):
        facts = build_lay_explanation_facts(
            condition="trace",
            raw_evidence={
                "factual_decision": "accept",
                "execution_trace": ["generate_candidate_mcqs#1", "acceptance_gate#1"],
            },
        )
        self.assertTrue(facts.source_basis_unavailable)
        self.assertTrue(
            any(
                "responsible" in lim.lower() or "specific teaching" in lim.lower()
                for lim in facts.limitations
            ),
            facts.limitations,
        )
        self.assertNotIn(
            "would have been different", " ".join(facts.limitations).lower()
        )
        good = (
            "The system organized the teaching material and generated the question. "
            "The trace shows this sequence of events, but not which specific teaching "
            "facts were responsible for the exact content of this question."
        )
        view = ExplanationEvidenceView(factual_decision="ACCEPT", lay_facts=facts)
        ok, warns = validate_explanation_faithfulness(
            condition="trace", explanation=good, evidence_view=view
        )
        self.assertTrue(ok, warns)

        bad_role = (
            "The supporting information shaped the question's educational focus "
            "and the correct answer is anchored to that concept."
        )
        ok2, warns2 = validate_explanation_faithfulness(
            condition="trace", explanation=bad_role, evidence_view=view
        )
        self.assertFalse(ok2)

        bad_cf = (
            "The question was accepted. It would have been different without the "
            "supporting information."
        )
        ok3, warns3 = validate_explanation_faithfulness(
            condition="trace", explanation=bad_cf, evidence_view=view
        )
        self.assertFalse(ok3)
        self.assertTrue(any("unasked counterfactual" in w for w in warns3), warns3)


class TestTraceProvenanceConcise(unittest.TestCase):
    def test_tp_mechanical_mentions_source_selection_limit(self):
        facts = build_lay_explanation_facts(
            condition="trace_provenance",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": [
                    {
                        "ref": "lo_1:v1",
                        "artifact_type": "LearningObjective",
                        "objective_text": "Define unsupervised learning and its purpose",
                    },
                    {
                        "ref": "atom_1:v1",
                        "artifact_type": "KnowledgeAtom",
                        "content_preview": (
                            "Unsupervised learning tries to understand the structure "
                            "of data without labeled outputs."
                        ),
                    },
                ],
            },
        )
        text = mechanical_synthesize_from_lay_facts(facts=facts)
        self.assertNotIn("Why this question?", text)
        self.assertIn("structure", text.lower())
        from mas_langgraph.experiment.synthesis import count_participant_sentences
        self.assertLessEqual(count_participant_sentences(text), 2)
        self.assertTrue(
            "linked" in text.lower() or "stating that" in text.lower(),
            text,
        )
        self.assertNotIn("available during planning, generation, and checking", text.lower())
        self.assertNotIn("would have been different", text.lower())
        self.assertNotRegex(text.lower(), r"\binfluenced\b")
        self.assertNotRegex(text.lower(), r"\bshaped\b")

    def test_prefers_clean_atom_over_noisy_chunk(self):
        selected, discarded = select_source_facts(
            [
                {
                    "ref": "chunk_noisy:v1",
                    "artifact_type": "EvidenceChunk",
                    "content": (
                        "Unsupervised Learning Input: inputs X = {x , …, x } 1 n "
                        "! Try to understand the structure"
                    ),
                },
                {
                    "ref": "atom_1:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": (
                        "Unsupervised learning tries to understand the structure "
                        "of data without labeled outputs."
                    ),
                },
            ],
            question_stem="What is the primary goal of unsupervised learning?",
            correct_answer="Understand structure without labels",
            learning_objective="Define unsupervised learning",
            max_facts=1,
        )
        self.assertEqual(len(selected), 1)
        self.assertIn("structure of data", selected[0].lower())
        self.assertNotIn("inputs x", selected[0].lower())
        self.assertTrue(any("inputs x" in d.lower() for d in discarded) or discarded)


class TestAxisConciseRole(unittest.TestCase):
    def test_distractor_role_with_target_description(self):
        base = _cand(stem="Q?", correct="a", distractors=["b", "c", "d"])
        cf = _cand(stem="Q?", correct="a", distractors=["b", "c", "x"])
        delta = compare_mcq_subject_content(
            factual=_obs(base),
            noop=_obs(base),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        arts = [
            {
                "ref": "atom_00000002:v1",
                "artifact_type": "KnowledgeAtom",
                "content_preview": "Identifying different types of items in data.",
            }
        ]
        facts = build_lay_explanation_facts(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": arts,
                "replay_comparisons": [
                    {
                        "evidence_status": "no_demonstrated_effect",
                        "intervention_before_subject_generation": True,
                        "question_content_comparison": delta.to_dict(),
                        "noop": {"decision": "accept"},
                        "counterfactual": {"decision": "accept"},
                        "intervention_signature": {
                            "operation": "REMOVE",
                            "target_ref": {"key": "atom_00000002:v1"},
                            "checkpoint_id": "uuid-ignored",
                            "checkpoint_alias": "c_8",
                            "boundary_raw": "before plan_mcq_blueprints#1",
                        },
                    }
                ],
            },
        )
        self.assertEqual(facts.information_role, "distractor_shaping")
        self.assertTrue(facts.decision_effect_known)
        self.assertIs(facts.decision_effect, False)
        text = mechanical_synthesize_from_lay_facts(facts=facts)
        self.assertNotIn("Why this question?", text)
        self.assertIn("types of items", text.lower())
        self.assertNotIn("REMOVE", text)
        # No repeated "influenced" role dumps.
        self.assertLessEqual(text.lower().count("influenced"), 2)
        from mas_langgraph.experiment.synthesis import count_participant_sentences
        n = count_participant_sentences(text)
        self.assertGreaterEqual(n, 2)
        self.assertLessEqual(n, 4)

    def test_formulation_selection_wording(self):
        base = _cand(
            stem="What is the primary goal of unsupervised learning?",
            correct="Understand structure",
            distractors=["b", "c", "d"],
        )
        cf = _cand(
            stem="What is the primary purpose of unsupervised learning?",
            correct="Understand structure",
            distractors=["b", "c", "d"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(base),
            noop=_obs(base),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        facts = build_lay_explanation_facts(
            condition="axis",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": [
                    {
                        "ref": "atom_1:v1",
                        "artifact_type": "KnowledgeAtom",
                        "content_preview": (
                            "Unsupervised learning tries to understand the structure "
                            "of the data without labeled outputs."
                        ),
                    }
                ],
                "replay_comparisons": [
                    {
                        "evidence_status": "no_demonstrated_effect",
                        "intervention_before_subject_generation": True,
                        "question_content_comparison": delta.to_dict(),
                        "noop": {"decision": "accept"},
                        "counterfactual": {"decision": "accept"},
                        "intervention_signature": {
                            "operation": "REMOVE",
                            "target_ref": {"key": "atom_1:v1"},
                            "checkpoint_alias": "c_8",
                            "boundary_raw": "before plan_mcq_blueprints#1",
                        },
                    }
                ],
            },
        )
        self.assertEqual(facts.information_role, ROLE_FORMULATION_SHAPING)
        self.assertTrue(facts.decision_effect_known)
        self.assertIs(facts.decision_effect, False)
        text = mechanical_synthesize_from_lay_facts(facts=facts).lower()
        self.assertIn("phrased", text)
        self.assertTrue(
            "acceptable" in text or "produced" in text or "checks" in text,
            text,
        )
        self.assertNotIn("probably", text)
        self.assertNotIn("would have been", text)
        self.assertNotIn(
            "not decisive for whether the question was selected",
            text,
        )
        from mas_langgraph.experiment.synthesis import count_participant_sentences
        n = count_participant_sentences(text)
        self.assertGreaterEqual(n, 2)
        self.assertLessEqual(n, 4)


class TestPrimaryInformationAlignment(unittest.TestCase):
    def test_description_is_target_not_stem(self):
        base = _cand(
            stem="What is the primary goal of unsupervised learning?",
            correct="a",
            distractors=["b", "c", "d"],
        )
        cf = _cand(
            stem="What is the primary purpose of unsupervised learning?",
            correct="a",
            distractors=["b", "c", "d"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(base),
            noop=_obs(base),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        finding = build_information_role_finding(
            {
                "intervention_before_subject_generation": True,
                "question_content_comparison": delta.to_dict(),
                "noop": {"decision": "accept"},
                "counterfactual": {"decision": "accept"},
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "atom_1:v1"},
                    "checkpoint_alias": "c_8",
                    "boundary_raw": "before plan_mcq_blueprints#1",
                },
            },
            relevant_artifacts=[
                {
                    "ref": "atom_1:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": (
                        "Unsupervised learning tries to understand the structure "
                        "of the data without labeled outputs."
                    ),
                },
                {
                    "ref": "atom_2:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": "Cars can be grouped by type.",
                },
            ],
        )
        assert finding is not None
        self.assertIn("structure of the data", finding.information_description)
        self.assertNotIn("primary goal", finding.information_description.lower())
        self.assertNotIn("cars", finding.information_description.lower())


class TestFormulationVsFocus(unittest.TestCase):
    def test_goal_vs_purpose_is_formulation(self):
        rel = classify_stem_semantic_relation(
            "What is the primary goal of unsupervised learning?",
            "What is the primary purpose of unsupervised learning?",
        )
        self.assertEqual(rel, "equivalent_rephrasing")
        base = _cand(
            stem="What is the primary goal of unsupervised learning?",
            correct="a",
            distractors=["b", "c", "d"],
        )
        cf = _cand(
            stem="What is the primary purpose of unsupervised learning?",
            correct="a",
            distractors=["b", "c", "d"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(base),
            noop=_obs(base),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        finding = build_information_role_finding(
            {
                "intervention_before_subject_generation": True,
                "question_content_comparison": delta.to_dict(),
                "noop": {"decision": "accept"},
                "counterfactual": {"decision": "accept"},
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "atom_1:v1"},
                    "boundary_raw": "before plan_mcq_blueprints#1",
                    "checkpoint_alias": "c_8",
                },
            },
            relevant_artifacts=[
                {
                    "ref": "atom_1:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": "Definition of unsupervised learning.",
                }
            ],
        )
        assert finding is not None
        self.assertEqual(finding.role_type, ROLE_FORMULATION_SHAPING)

    def test_clustering_shift_may_be_focus(self):
        rel = classify_stem_semantic_relation(
            "What is the primary goal of unsupervised learning?",
            "What is an advantage of clustering?",
        )
        self.assertIn(rel, {"different_focus", "same_concept_different_aspect"})
        base = _cand(
            stem="What is the primary goal of unsupervised learning?",
            correct="a",
            distractors=["b", "c", "d"],
        )
        cf = _cand(
            stem="What is an advantage of clustering?",
            correct="a",
            distractors=["b", "c", "d"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(base),
            noop=_obs(base),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        finding = build_information_role_finding(
            {
                "intervention_before_subject_generation": True,
                "question_content_comparison": delta.to_dict(),
                "noop": {"decision": "accept"},
                "counterfactual": {"decision": "accept"},
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "atom_1:v1"},
                    "boundary_raw": "before plan_mcq_blueprints#1",
                    "checkpoint_alias": "c_8",
                },
            },
            relevant_artifacts=[
                {
                    "ref": "atom_1:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": "Definition of unsupervised learning.",
                }
            ],
        )
        assert finding is not None
        self.assertEqual(finding.role_type, ROLE_FOCUS_SHAPING)


class TestLiveBoundaryMetadata(unittest.TestCase):
    def test_boundary_raw_survives_signature_export(self):
        rq = ResolvedQuery(
            operation="REMOVE",
            raw_text="REMOVE(chunk_00000002:v1, before atomize_evidence#1)",
            target_ref=ArtifactRef("chunk_00000002", 1),
            checkpoint_alias="c_6",
            checkpoint_id="1f1a9451-8093-6cfa-8005-ee4e8273bcab",
            boundary_raw="before atomize_evidence#1",
        )
        sig = QuerySignature.from_resolved(rq)
        exported = _signature_dict(sig)
        assert exported is not None
        self.assertEqual(exported["checkpoint_alias"], "c_6")
        self.assertEqual(exported["boundary_raw"], "before atomize_evidence#1")
        self.assertEqual(exported["checkpoint_id"], rq.checkpoint_id)

    def test_live_like_before_atomize_uses_boundary_raw(self):
        finding = build_information_role_finding(
            {
                "intervention_before_subject_generation": True,
                "question_content_comparison": compare_mcq_subject_content(
                    factual=_obs(_cand(stem="Q?", correct="a", distractors=["b", "c", "d"])),
                    noop=_obs(_cand(stem="Q?", correct="a", distractors=["b", "c", "d"])),
                    counterfactual=_obs(
                        _cand(stem="Q2?", correct="a", distractors=["b", "c", "d"])
                    ),
                    intervention_before_subject_generation=True,
                ).to_dict(),
                "noop": {"decision": "accept"},
                "counterfactual": {"decision": "accept"},
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "chunk_00000002:v1"},
                    "checkpoint_id": "1f1a9451-8093-6cfa-8005-ee4e8273bcab",
                    "checkpoint_alias": "c_6",
                    "boundary_raw": "before atomize_evidence#1",
                },
            },
            provenance_relations=[
                {
                    "relation": "evidence_to_atomize",
                    "artifact_ref": {"key": "chunk_00000002:v1"},
                }
            ],
        )
        assert finding is not None
        self.assertTrue(finding.semantic_information_absent)
        self.assertFalse(finding.derived_information_may_persist)
        self.assertEqual(finding.intervention_scope, "semantic_information")

    def test_c6_alias_alone_is_not_before_atomize(self):
        sem = infer_semantic_intervention_validity(
            artifact_type="EvidenceChunk",
            intervention_before_generation=True,
            boundary_label="c_6",
            provenance_relations=[
                {
                    "relation": "evidence_to_atomize",
                    "artifact_ref": {"key": "chunk_00000002:v1"},
                }
            ],
            target_ref="chunk_00000002:v1",
        )
        self.assertFalse(sem["semantic_information_absent"])
        self.assertTrue(sem["derived_information_may_persist"])

    def test_after_atomize_plan_boundary(self):
        finding = build_information_role_finding(
            {
                "intervention_before_subject_generation": True,
                "question_content_comparison": compare_mcq_subject_content(
                    factual=_obs(_cand(stem="Q?", correct="a", distractors=["b", "c", "d"])),
                    noop=_obs(_cand(stem="Q?", correct="a", distractors=["b", "c", "d"])),
                    counterfactual=_obs(
                        _cand(stem="Q?", correct="a", distractors=["b", "c", "d"])
                    ),
                    intervention_before_subject_generation=True,
                ).to_dict(),
                "noop": {"decision": "accept"},
                "counterfactual": {"decision": "accept"},
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"key": "chunk_00000002:v1"},
                    "checkpoint_alias": "c_8",
                    "boundary_raw": "before plan_mcq_blueprints#1",
                },
            },
            provenance_relations=[
                {
                    "relation": "evidence_to_atomize",
                    "artifact_ref": {"key": "chunk_00000002:v1"},
                }
            ],
        )
        assert finding is not None
        self.assertFalse(finding.semantic_information_absent)
        self.assertTrue(finding.derived_information_may_persist)
        self.assertEqual(finding.intervention_scope, "artifact_only")


class TestDuplicateExecutableFilter(unittest.TestCase):
    def test_executed_remove_not_listed_again(self):
        cands = (
            InterventionCandidate(
                ref=ArtifactRef("chunk_2", 1),
                artifact_type="EvidenceChunk",
                allowed_operations=("REMOVE",),
                valid_boundaries=("before atomize_evidence#1", "before plan_mcq_blueprints#1"),
                operation_boundaries={
                    "REMOVE": (
                        "before atomize_evidence#1",
                        "before plan_mcq_blueprints#1",
                    )
                },
                priority=1,
            ),
        )
        text = format_executable_queries(
            candidates=cands,
            interrogation_goal="question_selection",
            exclude_query_texts={"REMOVE(chunk_2:v1, before atomize_evidence#1)"},
            max_queries=8,
        )
        self.assertNotIn("REMOVE(chunk_2:v1, before atomize_evidence#1)", text)
        self.assertIn("REMOVE(chunk_2:v1, before plan_mcq_blueprints#1)", text)

    def test_question_selection_prioritizes_upstream(self):
        cands = (
            InterventionCandidate(
                ref=ArtifactRef("cand_1", 1),
                artifact_type="CandidateMCQ",
                allowed_operations=("REMOVE",),
                valid_boundaries=("after generate_candidate_mcqs#1",),
                operation_boundaries={"REMOVE": ("after generate_candidate_mcqs#1",)},
                priority=1,
            ),
            InterventionCandidate(
                ref=ArtifactRef("atom_1", 1),
                artifact_type="KnowledgeAtom",
                allowed_operations=("REMOVE",),
                valid_boundaries=("before plan_mcq_blueprints#1",),
                operation_boundaries={"REMOVE": ("before plan_mcq_blueprints#1",)},
                priority=5,
            ),
            InterventionCandidate(
                ref=ArtifactRef("chunk_2", 1),
                artifact_type="EvidenceChunk",
                allowed_operations=("REMOVE",),
                valid_boundaries=("before atomize_evidence#1",),
                operation_boundaries={"REMOVE": ("before atomize_evidence#1",)},
                priority=5,
            ),
        )
        text = format_executable_queries(
            candidates=cands,
            interrogation_goal="question_selection",
            max_queries=6,
        )
        forms = [ln for ln in text.splitlines() if ln.startswith("REMOVE(")]
        self.assertTrue(forms)
        self.assertTrue(forms[0].startswith("REMOVE(chunk_2:v1"))
        self.assertTrue(any("atom_1" in f for f in forms))
        # CandidateMCQ post-generation appears later or not in first slots.
        if any("cand_1" in f for f in forms):
            self.assertGreater(
                next(i for i, f in enumerate(forms) if "cand_1" in f),
                next(i for i, f in enumerate(forms) if "chunk_2" in f),
            )

    def test_executable_query_forms_helper(self):
        text = format_executable_queries(
            candidates=(
                InterventionCandidate(
                    ref=ArtifactRef("chunk_2", 1),
                    artifact_type="EvidenceChunk",
                    allowed_operations=("REMOVE",),
                    valid_boundaries=("before atomize_evidence#1",),
                    operation_boundaries={"REMOVE": ("before atomize_evidence#1",)},
                ),
            ),
            interrogation_goal="question_selection",
            max_queries=4,
        )
        forms = executable_query_forms(text)
        self.assertIn("remove(chunk_2:v1, before atomize_evidence#1)", forms)
        self.assertNotIn("done", forms)


class TestTraceVerbalizationRich(unittest.TestCase):
    def test_trace_events_no_raw_nodes(self):
        from mas_langgraph.experiment.trace_verbalizer import verbalize_trace_events
        from mas_langgraph.experiment.synthesis import (
            MAX_PARTICIPANT_WORDS,
            count_words,
            mechanical_synthesize_from_lay_facts,
        )

        lines = verbalize_trace_events(
            [
                "build_outline#1",
                "generate_learning_objectives#1",
                "initial_retrieve_evidence_for_all_sections#1",
                "plan_mcq_blueprints#1",
                "generate_candidate_mcqs_from_blueprints#1",
                "verify_candidate_mcqs_against_evidence#1",
                "apply_acceptance_gate#1",
                "package_final_quiz#1",
            ]
        )
        joined = " ".join(lines).lower()
        self.assertIn("learning objectives", joined)
        self.assertIn("generated", joined)
        self.assertNotIn("build_outline", joined)
        self.assertNotIn("package_final_quiz", joined)
        facts = build_lay_explanation_facts(
            condition="trace",
            raw_evidence={
                "factual_decision": "accept",
                "execution_trace": [
                    "build_outline#1",
                    "generate_learning_objectives#1",
                    "generate_candidate_mcqs_from_blueprints#1",
                    "verify_candidate_mcqs_against_evidence#1",
                    "apply_acceptance_gate#1",
                    "package_final_quiz#1",
                ],
            },
        )
        text = mechanical_synthesize_from_lay_facts(facts=facts)
        self.assertLessEqual(count_words(text), MAX_PARTICIPANT_WORDS)
        self.assertNotIn("build_outline", text)

    def test_repeated_cycles_compressed(self):
        from mas_langgraph.experiment.trace_verbalizer import verbalize_trace_events

        lines = verbalize_trace_events(
            [
                "generate_candidate_mcqs_from_blueprints#1",
                "verify_candidate_mcqs_against_evidence#1",
                "repair_missing_evidence#1",
                "generate_candidate_mcqs_from_blueprints#2",
                "verify_candidate_mcqs_against_evidence#2",
                "repair_missing_evidence#2",
                "generate_candidate_mcqs_from_blueprints#3",
                "verify_candidate_mcqs_against_evidence#3",
            ]
        )
        joined = " ".join(lines).lower()
        self.assertIn("repeated", joined)
        self.assertLessEqual(len(lines), 3)


class TestWordLimit(unittest.TestCase):
    def test_enforce_word_limit(self):
        from mas_langgraph.experiment.synthesis import (
            MAX_PARTICIPANT_WORDS,
            count_words,
            enforce_participant_word_limit,
        )

        long = " ".join(["word"] * 250) + "."
        out, used = enforce_participant_word_limit(
            long, fallback="Short faithful fallback under the limit."
        )
        self.assertTrue(used)
        self.assertLessEqual(count_words(out), MAX_PARTICIPANT_WORDS)


class TestComponentSelectionNoDoubleCount(unittest.TestCase):
    def test_broad_focus_does_not_also_fill_answer_distractor(self):
        from mas_langgraph.experiment.question_content_facts import (
            select_component_role_findings,
        )

        base = _cand(
            stem="What is the primary goal of unsupervised learning?",
            correct="structure",
            distractors=["a", "b", "c"],
        )
        cf = _cand(
            stem="What is an advantage of clustering?",
            correct="groups",
            distractors=["x", "y", "z"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(base),
            noop=_obs(base),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        lines, meta, _ = select_component_role_findings(
            [
                {
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": delta.to_dict(),
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "atom_1:v1"},
                        "boundary_raw": "before plan_mcq_blueprints#1",
                    },
                }
            ],
            relevant_artifacts=[
                {
                    "ref": "atom_1:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": "Definition of unsupervised learning.",
                }
            ],
        )
        roles = meta.get("selected_component_roles") or []
        self.assertLessEqual(int(meta.get("selected_component_finding_count") or 0), 3)
        self.assertNotIn("answer_anchoring", roles)
        self.assertNotIn("distractor_shaping", roles)
        self.assertTrue(lines)


class TestEvidenceChunkParticipantRendering(unittest.TestCase):
    def test_noisy_chunk_uses_clean_descendant_atoms(self):
        from mas_langgraph.experiment.information_role import (
            _resolve_information_description,
        )

        desc = _resolve_information_description(
            target_key="chunk_00000002:v1",
            relevant_artifacts=[
                {
                    "ref": "chunk_00000002:v1",
                    "artifact_type": "EvidenceChunk",
                    "content_preview": (
                        "Input X = {x...} ! ! ! Unsupervised Learning noisy OCR"
                    ),
                },
                {
                    "ref": "atom_00000001:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": (
                        "Unsupervised learning tries to understand the structure "
                        "of data without labeled outputs."
                    ),
                },
                {
                    "ref": "atom_00000002:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": "Identify how many types of items exist.",
                },
                {
                    "ref": "atom_00000003:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": "Understand variation in the data.",
                },
            ],
        )
        low = desc.lower()
        self.assertIn("structure", low)
        self.assertNotIn("{x", low)
        self.assertNotIn("! !", low)


class TestIdentityVsAcceptanceWording(unittest.TestCase):
    def test_identity_true_acceptance_false_fields(self):
        from mas_langgraph.experiment.information_role import verbalize_information_role
        from mas_langgraph.experiment.information_role import InformationRoleFinding

        finding = InformationRoleFinding(
            role_type="focus_shaping",
            target_ref="atom_1:v1",
            information_description="Definition of unsupervised learning.",
            stem_effect="changed",
            correct_answer_effect="changed",
            distractor_effect="changed",
            decision_effect="unchanged",
            decision_before="ACCEPT",
            decision_after="ACCEPT",
            content_effect=True,
            decision_effect_bool=False,
            causal_classification="influential_non_decisive",
        )
        lines = " ".join(verbalize_information_role(finding)).lower()
        self.assertNotIn("not decisive for whether this question was selected", lines)
        self.assertTrue(
            "influenced which question was produced" in lines
            or "acceptable" in lines
            or "focused" in lines,
            lines,
        )


if __name__ == "__main__":
    unittest.main()
