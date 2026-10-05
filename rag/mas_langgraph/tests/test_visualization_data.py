"""Researcher-only visualization_data contract tests."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy

from mas_langgraph.artifacts.models import ArtifactEnvelope, ArtifactRef, ProducedBy
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.axis.models import ExplainedCase
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.core.entities import EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.outcomes import CriterionValue, OutcomeAssessment
from mas_langgraph.domains.quiz.entities import (
    AnswerOption,
    CandidateMCQ,
    CrossCoVeReport,
    EvidenceCheckReport,
    MCQBlueprint,
    SourceRef,
    SupervisorDecision,
)
from mas_langgraph.experiment.evidence import (
    build_trace_evidence,
    build_trace_provenance_evidence,
)
from mas_langgraph.experiment.export import case_to_dict
from mas_langgraph.experiment.models import (
    ExplanationConditionResult,
    MCQExplanationComparison,
)
from mas_langgraph.experiment.visualization_data import (
    build_case_visualization_data,
    serialize_ref,
)
from mas_langgraph.provenance.models import ArtifactBinding, OccurrenceRecord, RunProvenanceRelation


def _env(logical_id, version, artifact_type, payload, branch="factual"):
    return ArtifactEnvelope.create(
        logical_id=logical_id,
        version=version,
        artifact_type=artifact_type,
        payload=payload,
        branch_id=branch,
        created_in_run="run_viz",
    )


def _explained(q_id="q_00000010", version=2):
    return ExplainedCase(
        run_id="run_viz",
        artifact_ref=ArtifactRef(q_id, version),
        branch_id="factual",
        decision_ref=ArtifactRef(f"decision:{q_id}", version),
        outcome="accept",
        correspondence_key="bp_1",
        assessment_ref=ArtifactRef(f"assessment:{q_id}", version),
    )


def _occ(*, oid, node, idx, exec_i, produced=(), bindings=(), superstep=3):
    return OccurrenceRecord(
        occurrence_id=oid,
        node_id=node,
        occurrence_index=idx,
        branch_id="factual",
        run_id="run_viz",
        execution_index=exec_i,
        checkpoint_before=f"before-{exec_i}",
        checkpoint_after=f"after-{exec_i}",
        superstep=superstep,
        input_bindings=tuple(bindings),
        produced_artifacts=tuple(produced),
        state_delta={"skip": True} if exec_i == 12 else {},
        error=None,
    )


def _long_text(prefix: str, n: int = 400) -> str:
    return prefix + " " + ("content-token " * n)


class TestVisualizationDataContract(unittest.TestCase):
    def setUp(self):
        self.lo = LearningObjective(
            lo_id="lo_1",
            section_id="sec_1",
            objective_text="Explain unsupervised learning",
            difficulty_level="intermediate",
        )
        self.chunk_text = _long_text("Evidence chunk full text about clustering.", 80)
        self.atom_text = _long_text("Knowledge atom full text about k-means assignment.", 80)
        self.chunk = EvidenceChunk(
            chunk_id="E1",
            content=self.chunk_text,
            metadata={"page_number": 4, "section_id": "sec_1", "pdf_id": "141"},
            relevance_score=0.8,
            quality_score=0.9,
            relevance_tags=["clustering"],
            used_in_atoms=["A1"],
            duplicate_of=None,
            flags=[],
        )
        self.atom = KnowledgeAtom(
            atom_id="A1",
            content=self.atom_text,
            atom_type="procedure",
            citations=[{"chunk_id": "E1", "page": 4}],
            section_id="sec_1",
            lo_id="lo_1",
        )
        self.bp = MCQBlueprint(
            blueprint_id="bp_1",
            section_id="sec_1",
            lo_id="lo_1",
            target_atom_ids=["A1"],
            evidence_chunk_ids=["E1"],
            question_type="key_characteristic",
            difficulty="easy",
            bloom_level="Understanding",
            generation_goal="Ask which statement best describes k-means assignment.",
            status="generated",
        )
        self.mcq = CandidateMCQ(
            q_id="q_00000010",
            blueprint_id="bp_1",
            stem="What does k-means assign each point to?",
            options=[
                AnswerOption(
                    option_id="A",
                    text="Nearest centroid",
                    is_correct=True,
                    explanation="Assignment uses distance.",
                    misconception=None,
                    source_refs=[SourceRef(doc_id="141", page=4, chunk_id="E1")],
                ),
                AnswerOption(
                    option_id="B",
                    text="Random cluster",
                    is_correct=False,
                    explanation="Not supported.",
                    misconception="Random assignment",
                    source_refs=[],
                ),
            ],
            correct_option_id="A",
            answer_explanation="Each point is assigned to the nearest centroid.",
            source_refs=[SourceRef(doc_id="141", page=4, chunk_id="E1")],
            status="accepted",
            revision_count=1,
            flags=[],
        )
        self.ev_report = EvidenceCheckReport(
            q_id="q_00000010",
            supported_by_pdf=True,
            unsupported_claims=[],
            source_refs=[SourceRef(doc_id="141", page=4, chunk_id="E1")],
            confidence=0.91,
            candidate_version=2,
            answer_supported=True,
            stem_supported=True,
            distractors_valid=True,
            explanation_supported=True,
            required_fixes=[],
            notes=["ok"],
        )
        self.cove = CrossCoVeReport(
            q_id="q_00000010",
            generator_answer="A",
            verifier_answers=["A", "A", "A"],
            majority_answer="A",
            passed=True,
            candidate_version=2,
            confidence=1.0,
            verifier_model="gpt-5.4-mini-2026-03-17",
            notes=["Stub Cross-CoVe verifier (no LLM)."],
        )
        self.decision = SupervisorDecision(
            q_id="q_00000010",
            decision="accept",
            reasons=["passed"],
            candidate_version=2,
            required_fixes=[],
            assessment_ref=ArtifactRef("assessment:q_00000010", 2),
        )
        self.assessment = OutcomeAssessment(
            assessment_ref=ArtifactRef("assessment:q_00000010", 2),
            target_ref=ArtifactRef("q_00000010", 2),
            criteria=(
                CriterionValue(name="pdf_supported", value=True, value_type="bool"),
            ),
            notes=("gate",),
        )
        self.unrelated = LearningObjective(
            lo_id="lo_unrelated",
            section_id="sec_other",
            objective_text="This must not be exported",
            difficulty_level="beginner",
        )

        self.reg = ArtifactRegistry()
        self.envs = [
            _env("lo_1", 1, "LearningObjective", self.lo),
            _env("E1", 1, "EvidenceChunk", self.chunk),
            _env("A1", 1, "KnowledgeAtom", self.atom),
            _env("bp_1", 1, "MCQBlueprint", self.bp),
            _env("q_00000010", 2, "CandidateMCQ", self.mcq),
            _env("evidence-report:q_00000010", 2, "EvidenceCheckReport", self.ev_report),
            _env("cross-cove-report:q_00000010", 2, "CrossCoVeReport", self.cove),
            _env("decision:q_00000010", 2, "SupervisorDecision", self.decision),
            _env("assessment:q_00000010", 2, "OutcomeAssessment", self.assessment),
            _env("lo_unrelated", 1, "LearningObjective", self.unrelated),
        ]
        for env in self.envs:
            self.reg.store(env)

        gen_occ = _occ(
            oid="run_viz:factual:generate_candidate_mcqs_from_blueprints#1",
            node="generate_candidate_mcqs_from_blueprints",
            idx=1,
            exec_i=12,
            produced=(ArtifactRef("q_00000010", 2),),
            bindings=(
                ArtifactBinding(artifact_ref=ArtifactRef("bp_1", 1), relation="generation_blueprint"),
            ),
            superstep=9,
        )
        lo_occ = _occ(
            oid="run_viz:factual:generate_learning_objectives#1",
            node="generate_learning_objectives",
            idx=1,
            exec_i=2,
            produced=(ArtifactRef("lo_1", 1),),
        )
        self.rel = RunProvenanceRelation(
            producer_occurrence_id=lo_occ.occurrence_id,
            artifact_ref=ArtifactRef("lo_1", 1),
            relation="used_learning_objective",
            consumer_occurrence_id=gen_occ.occurrence_id,
            artifact_type="LearningObjective",
        )
        self.slice = CaseSlice(
            explained_case=_explained(),
            relevant_artifact_refs=(
                ArtifactRef("lo_1", 1),
                ArtifactRef("E1", 1),
                ArtifactRef("A1", 1),
                ArtifactRef("bp_1", 1),
                ArtifactRef("q_00000010", 2),
                ArtifactRef("evidence-report:q_00000010", 2),
                ArtifactRef("cross-cove-report:q_00000010", 2),
                ArtifactRef("decision:q_00000010", 2),
                ArtifactRef("assessment:q_00000010", 2),
            ),
            relevant_occurrences=(gen_occ, lo_occ),
            relevant_relations=(self.rel,),
            criterion_values=self.assessment.criteria,
            decision="accept",
        )
        self.explained = _explained()

    def _viz(self):
        return build_case_visualization_data(
            explained_case=self.explained,
            case_slice=self.slice,
            registry=self.reg,
            produced_by=(
                ProducedBy(
                    artifact_ref=ArtifactRef("q_00000010", 2),
                    producer_occurrence_id="run_viz:factual:generate_candidate_mcqs_from_blueprints#1",
                ),
            ),
            factual_decision="accept",
        )

    def _by_type(self, viz, artifact_type):
        return [a for a in viz["artifacts"] if a["artifact_type"] == artifact_type]

    def test_learning_objective_full_text(self):
        viz = self._viz()
        los = self._by_type(viz, "LearningObjective")
        self.assertEqual(len(los), 1)
        self.assertEqual(
            los[0]["content"]["objective_text"],
            "Explain unsupervised learning",
        )

    def test_evidence_and_atom_content_not_truncated(self):
        viz = self._viz()
        chunk = self._by_type(viz, "EvidenceChunk")[0]["content"]
        atom = self._by_type(viz, "KnowledgeAtom")[0]["content"]
        self.assertEqual(chunk["content"], self.chunk_text)
        self.assertGreater(len(chunk["content"]), 200)
        self.assertEqual(atom["content"], self.atom_text)
        self.assertGreater(len(atom["content"]), 200)
        self.assertEqual(atom["citations"], [{"chunk_id": "E1", "page": 4}])

    def test_blueprint_details(self):
        viz = self._viz()
        bp = self._by_type(viz, "MCQBlueprint")[0]["content"]
        self.assertEqual(
            bp["generation_goal"],
            "Ask which statement best describes k-means assignment.",
        )
        self.assertEqual(bp["question_type"], "key_characteristic")
        self.assertEqual(bp["difficulty"], "easy")
        self.assertEqual(bp["bloom_level"], "Understanding")
        self.assertEqual(bp["target_atom_ids"], ["A1"])
        self.assertEqual(bp["evidence_chunk_ids"], ["E1"])

    def test_candidate_mcq_preserves_v2(self):
        viz = self._viz()
        cand = self._by_type(viz, "CandidateMCQ")[0]
        self.assertEqual(cand["ref"], serialize_ref(ArtifactRef("q_00000010", 2)))
        self.assertNotEqual(cand["ref"]["key"], "q_00000010:v1")
        self.assertEqual(cand["ref"]["version"], 2)
        content = cand["content"]
        self.assertEqual(content["stem"], self.mcq.stem)
        self.assertEqual(len(content["options"]), 2)
        self.assertEqual(content["options"][0]["text"], "Nearest centroid")
        self.assertEqual(content["correct_option_id"], "A")
        self.assertEqual(content["answer_explanation"], self.mcq.answer_explanation)
        self.assertEqual(content["revision_count"], 1)

    def test_verification_cross_cove_decision_fields(self):
        viz = self._viz()
        ev = self._by_type(viz, "EvidenceCheckReport")[0]["content"]
        cove = self._by_type(viz, "CrossCoVeReport")[0]["content"]
        dec = self._by_type(viz, "SupervisorDecision")[0]["content"]
        assess = self._by_type(viz, "OutcomeAssessment")[0]["content"]
        self.assertEqual(ev["candidate_version"], 2)
        self.assertTrue(ev["supported_by_pdf"])
        self.assertTrue(ev["stem_supported"])
        self.assertEqual(cove["candidate_version"], 2)
        self.assertEqual(cove["generator_answer"], "A")
        self.assertEqual(cove["verifier_answers"], ["A", "A", "A"])
        self.assertTrue(cove["passed"])
        self.assertEqual(cove["verifier_model"], "gpt-5.4-mini-2026-03-17")
        self.assertEqual(dec["decision"], "accept")
        self.assertEqual(dec["candidate_version"], 2)
        self.assertEqual(dec["assessment_ref"]["key"], "assessment:q_00000010:v2")
        self.assertEqual(assess["target_ref"]["key"], "q_00000010:v2")

    def test_structured_occurrences(self):
        viz = self._viz()
        occs = viz["occurrences"]
        self.assertEqual(len(occs), 2)
        self.assertEqual(occs[0]["execution_index"], 2)
        gen = occs[1]
        self.assertEqual(gen["occurrence_id"], "run_viz:factual:generate_candidate_mcqs_from_blueprints#1")
        self.assertEqual(gen["node_id"], "generate_candidate_mcqs_from_blueprints")
        self.assertEqual(gen["execution_index"], 12)
        self.assertEqual(gen["superstep"], 9)
        self.assertEqual(gen["checkpoint_before"], "before-12")
        self.assertEqual(gen["checkpoint_after"], "after-12")
        self.assertEqual(gen["produced_artifacts"][0]["key"], "q_00000010:v2")
        self.assertEqual(gen["input_bindings"][0]["relation"], "generation_blueprint")
        self.assertEqual(gen["input_bindings"][0]["artifact_ref"]["key"], "bp_1:v1")
        self.assertTrue(gen["has_state_delta"])
        self.assertNotIn("state_delta", gen)

    def test_provenance_connectivity(self):
        viz = self._viz()
        rel = viz["provenance_relations"][0]
        self.assertEqual(rel["producer_occurrence_id"], "run_viz:factual:generate_learning_objectives#1")
        self.assertEqual(rel["consumer_occurrence_id"], "run_viz:factual:generate_candidate_mcqs_from_blueprints#1")
        self.assertEqual(rel["artifact_ref"]["key"], "lo_1:v1")
        self.assertEqual(rel["relation"], "used_learning_objective")
        self.assertNotIn("caused", rel)
        self.assertNotIn("causal", rel)

    def test_unrelated_registry_artifact_not_exported(self):
        viz = self._viz()
        keys = {(a["ref"]["logical_id"], a["ref"]["version"]) for a in viz["artifacts"]}
        self.assertNotIn(("lo_unrelated", 1), keys)
        texts = [a["content"].get("objective_text") for a in viz["artifacts"]]
        self.assertNotIn("This must not be exported", texts)

    def test_determinism(self):
        a = self._viz()
        b = self._viz()
        self.assertEqual(a, b)
        dumped = json.dumps(a, sort_keys=True)
        self.assertEqual(dumped, json.dumps(b, sort_keys=True))

    def test_json_safe_without_default_str(self):
        viz = self._viz()
        json.dumps(viz)

    def test_participant_evidence_unchanged(self):
        tp_before = deepcopy(
            build_trace_provenance_evidence(
                explained_case=self.explained,
                case_slice=self.slice,
                registry=self.reg,
            )
        )
        trace_before = deepcopy(
            build_trace_evidence(
                explained_case=self.explained,
                factual_occurrences=self.slice.relevant_occurrences,
            )
        )
        viz = self._viz()
        tp_after = build_trace_provenance_evidence(
            explained_case=self.explained,
            case_slice=self.slice,
            registry=self.reg,
        )
        trace_after = build_trace_evidence(
            explained_case=self.explained,
            factual_occurrences=self.slice.relevant_occurrences,
        )
        self.assertEqual(tp_before, tp_after)
        self.assertEqual(trace_before, trace_after)
        self.assertNotIn("visualization_data", tp_after)
        self.assertNotIn("visualization_data", trace_after)
        self.assertIn("content_preview", str(tp_after.get("relevant_artifacts")))
        lo_rows = [r for r in tp_after["relevant_artifacts"] if "lo_1" in str(r.get("ref"))]
        self.assertTrue(lo_rows)
        self.assertEqual(lo_rows[0].get("objective_text"), "Explain unsupervised learning")
        self.assertNotIn("generation_goal", lo_rows[0])
        self.assertNotIn("options", "".join(str(r) for r in tp_after["relevant_artifacts"] if r.get("stem")))

        empty = ExplanationConditionResult(condition="trace", explanation="t")
        case = MCQExplanationComparison(
            run_id="run_viz",
            branch_id="factual",
            candidate_ref=ArtifactRef("q_00000010", 2),
            assessment_ref=ArtifactRef("assessment:q_00000010", 2),
            decision_ref=ArtifactRef("decision:q_00000010", 2),
            question_stem="stem",
            answer_options=(),
            correct_option_id="A",
            factual_decision="accept",
            user_question="Why?",
            trace=ExplanationConditionResult(
                condition="trace",
                explanation="t",
                raw_evidence=trace_after,
            ),
            trace_provenance=ExplanationConditionResult(
                condition="trace_provenance",
                explanation="tp",
                raw_evidence=tp_after,
            ),
            axis=empty,
            visualization_data=viz,
        )
        exported = case_to_dict(case)
        self.assertIn("visualization_data", exported)
        self.assertNotIn("visualization_data", exported["conditions"]["trace"])
        self.assertNotIn("visualization_data", exported["conditions"]["trace_provenance"])
        self.assertNotIn("visualization_data", exported["conditions"]["trace"]["raw_evidence"])
        cand_rows = [
            a for a in exported["visualization_data"]["artifacts"]
            if a["artifact_type"] == "CandidateMCQ"
        ]
        self.assertEqual(cand_rows[0]["ref"]["key"], "q_00000010:v2")

    def test_target_uses_explained_version(self):
        viz = self._viz()
        self.assertEqual(viz["target"]["candidate_ref"]["key"], "q_00000010:v2")
        self.assertEqual(viz["target"]["factual_decision"], "accept")
        self.assertEqual(viz["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
