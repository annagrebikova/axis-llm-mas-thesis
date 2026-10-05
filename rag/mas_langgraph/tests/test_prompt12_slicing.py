import unittest


from mas_langgraph.axis.slicing import ProvenanceSlicer
from mas_langgraph.axis.models import ExplainedCase
from mas_langgraph.axis.slicing import SliceValidationError
from mas_langgraph.cases.mcq.slicing import (
    MCQProvenanceSliceAdapter,
    resolve_mcq_explained_case,
)
from mas_langgraph.cases.mcq.artifacts import (
    wrap_blueprint,
    wrap_candidate_mcq,
    wrap_cross_cove_report,
    wrap_evidence_chunk,
    wrap_evidence_report,
    wrap_knowledge_atom,
    wrap_outcome_assessment,
    wrap_supervisor_decision,
)
from mas_langgraph.artifacts.models import ArtifactRef, ProducedBy
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.core.entities import EvidenceChunk, KnowledgeAtom, DocSection, LearningObjective
from mas_langgraph.domains.quiz.entities import (
    AnswerOption,
    MCQBlueprint,
    CandidateMCQ,
    EvidenceCheckReport,
    CrossCoVeReport,
    SupervisorDecision,
)
from mas_langgraph.core.outcomes import CriterionValue, OutcomeAssessment
from mas_langgraph.provenance.models import (
    ArtifactBinding,
    OccurrenceRecord,
    RunProvenanceRelation,
)
from mas_langgraph.replay.events import InterventionEvent


def _opt_tuple(xs):
    return tuple(xs) if xs is not None else ()


class TestPrompt12ProvenanceSlicing(unittest.TestCase):
    def _build_q12_q13_factual_fixture(
        self,
        *,
        run_id: str = "run_p12_factual",
        branch_id: str = "factual",
        q12_decision: str = "accept",
    ):
        reg = ArtifactRegistry()

        # Doc / LO
        doc_s2 = DocSection(section_id="S2", title="T2", slide_range=(1, 2), importance_weight=1.0)
        doc_s3 = DocSection(section_id="S3", title="T3", slide_range=(1, 2), importance_weight=1.0)
        lo3 = LearningObjective(lo_id="LO3", section_id="S2", objective_text="O3", difficulty_level="easy")
        lo4 = LearningObjective(lo_id="LO4", section_id="S3", objective_text="O4", difficulty_level="easy")

        # Evidence chunks and atoms
        e17 = EvidenceChunk(chunk_id="E17", content="e17", metadata={}, relevance_score=0.1, quality_score=0.9)
        e23 = EvidenceChunk(chunk_id="E23", content="e23", metadata={}, relevance_score=0.1, quality_score=0.9)
        a3 = KnowledgeAtom(
            atom_id="A3",
            content="a3",
            atom_type="procedure",
            citations=[{"chunk_id": "E17"}],
            section_id="S2",
            lo_id="LO3",
        )
        a4 = KnowledgeAtom(
            atom_id="A4",
            content="a4",
            atom_type="procedure",
            citations=[{"chunk_id": "E23"}],
            section_id="S3",
            lo_id="LO4",
        )

        # Blueprints
        bp12 = MCQBlueprint(
            blueprint_id="BP12",
            section_id="S2",
            lo_id="LO3",
            target_atom_ids=["A3"],
            evidence_chunk_ids=["E17"],
            question_type="basic_definition",
            difficulty="easy",
            bloom_level="remembering",
            generation_goal="g",
            status="planned",
        )
        bp13 = MCQBlueprint(
            blueprint_id="BP13",
            section_id="S3",
            lo_id="LO4",
            target_atom_ids=["A4"],
            evidence_chunk_ids=["E23"],
            question_type="basic_definition",
            difficulty="easy",
            bloom_level="remembering",
            generation_goal="g",
            status="planned",
        )

        # Candidates v1
        opts = [AnswerOption("A", "a", True), AnswerOption("B", "b", False)]
        cand12 = CandidateMCQ(
            q_id="Q12",
            blueprint_id="BP12",
            stem="stem12",
            options=opts,
            correct_option_id="A",
            answer_explanation="x",
            status="accepted",
            revision_count=0,
            flags=[],
        )
        cand13 = CandidateMCQ(
            q_id="Q13",
            blueprint_id="BP13",
            stem="stem13",
            options=opts,
            correct_option_id="A",
            answer_explanation="x",
            status="accepted",
            revision_count=0,
            flags=[],
        )

        # Persist doc / LO / evidence / atoms / blueprints / candidates
        for env in [
            wrap_blueprint(bp12, branch_id=branch_id, registry=reg),
            wrap_blueprint(bp13, branch_id=branch_id, registry=reg),
            wrap_candidate_mcq(cand12, version=1, branch_id=branch_id, registry=reg),
            wrap_candidate_mcq(cand13, version=1, branch_id=branch_id, registry=reg),
            wrap_evidence_chunk(e17, branch_id=branch_id, registry=reg),
            wrap_evidence_chunk(e23, branch_id=branch_id, registry=reg),
            wrap_knowledge_atom(a3, branch_id=branch_id, registry=reg),
            wrap_knowledge_atom(a4, branch_id=branch_id, registry=reg),
        ]:
            pass

        # Doc/LO wrappers: reuse envelope wrappers via direct wrapping
        from mas_langgraph.cases.mcq.artifacts import wrap_doc_section, wrap_learning_objective

        wrap_doc_section(doc_s2, branch_id=branch_id, registry=reg)
        wrap_doc_section(doc_s3, branch_id=branch_id, registry=reg)
        wrap_learning_objective(lo3, branch_id=branch_id, registry=reg)
        wrap_learning_objective(lo4, branch_id=branch_id, registry=reg)

        q12_ref = ArtifactRef("Q12", 1)
        q13_ref = ArtifactRef("Q13", 1)
        bp12_ref = ArtifactRef("BP12", 1)
        bp13_ref = ArtifactRef("BP13", 1)
        e17_ref = ArtifactRef("E17", 1)
        e23_ref = ArtifactRef("E23", 1)
        a3_ref = ArtifactRef("A3", 1)
        a4_ref = ArtifactRef("A4", 1)

        # Evidence reports and cross-cove reports
        er12 = EvidenceCheckReport(
            q_id="Q12",
            supported_by_pdf=True,
            unsupported_claims=[],
            source_refs=[],
            confidence=0.9,
            candidate_version=1,
        )
        er13 = EvidenceCheckReport(
            q_id="Q13",
            supported_by_pdf=True,
            unsupported_claims=[],
            source_refs=[],
            confidence=0.9,
            candidate_version=1,
        )
        cr12 = CrossCoVeReport(
            q_id="Q12",
            generator_answer="ga12",
            verifier_answers=["va"],
            majority_answer="ga12",
            passed=True,
            candidate_version=1,
            confidence=0.8,
        )
        cr13 = CrossCoVeReport(
            q_id="Q13",
            generator_answer="ga13",
            verifier_answers=["va"],
            majority_answer="ga13",
            passed=True,
            candidate_version=1,
            confidence=0.8,
        )
        wrap_evidence_report(er12, version=1, branch_id=branch_id, registry=reg)
        wrap_evidence_report(er13, version=1, branch_id=branch_id, registry=reg)
        wrap_cross_cove_report(cr12, version=1, branch_id=branch_id, registry=reg)
        wrap_cross_cove_report(cr13, version=1, branch_id=branch_id, registry=reg)

        er12_ref = ArtifactRef("evidence-report:Q12", 1)
        er13_ref = ArtifactRef("evidence-report:Q13", 1)
        cr12_ref = ArtifactRef("cross-cove-report:Q12", 1)
        cr13_ref = ArtifactRef("cross-cove-report:Q13", 1)

        # Outcome assessment + decision
        cv = (
            CriterionValue("criterion_a", True),
            CriterionValue("criterion_b", 0.25, value_type="float"),
        )
        assessment12_ref = ArtifactRef("assessment:Q12", 1)
        assessment13_ref = ArtifactRef("assessment:Q13", 1)
        assessment12 = OutcomeAssessment(
            assessment_ref=assessment12_ref, target_ref=q12_ref, criteria=cv
        )
        assessment13 = OutcomeAssessment(
            assessment_ref=assessment13_ref, target_ref=q13_ref, criteria=cv
        )
        wrap_outcome_assessment(assessment12, version=1, branch_id=branch_id, registry=reg)
        wrap_outcome_assessment(assessment13, version=1, branch_id=branch_id, registry=reg)

        decision12_ref = ArtifactRef("decision:Q12", 1)
        decision13_ref = ArtifactRef("decision:Q13", 1)
        decision12 = SupervisorDecision(
            q_id="Q12",
            decision=q12_decision,
            reasons=["r1"],
            candidate_version=1,
            required_fixes=("fix1",),
            assessment_ref=assessment12_ref,
        )
        decision13 = SupervisorDecision(
            q_id="Q13",
            decision="accept",
            reasons=["r1"],
            candidate_version=1,
            required_fixes=("fix1",),
            assessment_ref=assessment13_ref,
        )
        wrap_supervisor_decision(decision12, version=1, branch_id=branch_id, registry=reg)
        wrap_supervisor_decision(decision13, version=1, branch_id=branch_id, registry=reg)

        # Occurrence records + provenance relations
        # Doc/LO/evidence/atoms producers
        occ_doc_s2 = OccurrenceRecord(
            occurrence_id="doc_s2",
            node_id="build_outline",
            occurrence_index=1,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=0,
            superstep=0,
            checkpoint_before="c0",
            checkpoint_after="c1",
            produced_artifacts=(ArtifactRef("S2", 1),),
        )
        occ_doc_s3 = OccurrenceRecord(
            occurrence_id="doc_s3",
            node_id="build_outline",
            occurrence_index=2,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=0,
            superstep=0,
            checkpoint_before="c0",
            checkpoint_after="c1",
            produced_artifacts=(ArtifactRef("S3", 1),),
        )
        occ_lo3 = OccurrenceRecord(
            occurrence_id="lo3",
            node_id="generate_learning_objectives",
            occurrence_index=1,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=0,
            superstep=0,
            checkpoint_before="c0",
            checkpoint_after="c1",
            produced_artifacts=(ArtifactRef("LO3", 1),),
        )
        occ_lo4 = OccurrenceRecord(
            occurrence_id="lo4",
            node_id="generate_learning_objectives",
            occurrence_index=2,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=0,
            superstep=0,
            checkpoint_before="c0",
            checkpoint_after="c1",
            produced_artifacts=(ArtifactRef("LO4", 1),),
        )
        occ_e17 = OccurrenceRecord(
            occurrence_id="e17",
            node_id="curate_evidence",
            occurrence_index=1,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=1,
            superstep=1,
            checkpoint_before="c1",
            checkpoint_after="c2",
            produced_artifacts=(e17_ref,),
        )
        occ_e23 = OccurrenceRecord(
            occurrence_id="e23",
            node_id="curate_evidence",
            occurrence_index=2,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=1,
            superstep=1,
            checkpoint_before="c1",
            checkpoint_after="c2",
            produced_artifacts=(e23_ref,),
        )
        occ_a3 = OccurrenceRecord(
            occurrence_id="a3",
            node_id="atomize_evidence",
            occurrence_index=1,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=2,
            superstep=1,
            checkpoint_before="c1",
            checkpoint_after="c2",
            produced_artifacts=(a3_ref,),
        )
        occ_a4 = OccurrenceRecord(
            occurrence_id="a4",
            node_id="atomize_evidence",
            occurrence_index=2,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=2,
            superstep=1,
            checkpoint_before="c1",
            checkpoint_after="c2",
            produced_artifacts=(a4_ref,),
        )

        # plan_mcq_blueprints consumes doc/lo/atoms/evidence for both blueprints.
        occ_plan = OccurrenceRecord(
            occurrence_id="plan#1",
            node_id="plan_mcq_blueprints",
            occurrence_index=1,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=3,
            superstep=2,
            checkpoint_before="c2",
            checkpoint_after="c3",
            input_bindings=(
                ArtifactBinding(ArtifactRef("S2", 1), "document_section"),
                ArtifactBinding(ArtifactRef("LO3", 1), "learning_objective"),
                ArtifactBinding(a3_ref, "planning_atom"),
                ArtifactBinding(e17_ref, "planning_evidence"),
                ArtifactBinding(ArtifactRef("S3", 1), "document_section"),
                ArtifactBinding(ArtifactRef("LO4", 1), "learning_objective"),
                ArtifactBinding(a4_ref, "planning_atom"),
                ArtifactBinding(e23_ref, "planning_evidence"),
            ),
            produced_artifacts=(bp12_ref, bp13_ref),
        )

        occ_generate = OccurrenceRecord(
            occurrence_id="gen#1",
            node_id="generate_candidate_mcqs_from_blueprints",
            occurrence_index=1,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=4,
            superstep=3,
            checkpoint_before="c3",
            checkpoint_after="c4",
            input_bindings=(
                ArtifactBinding(bp12_ref, "generation_blueprint"),
                ArtifactBinding(a3_ref, "knowledge_atom"),
                ArtifactBinding(e17_ref, "supporting_evidence"),
                ArtifactBinding(bp13_ref, "generation_blueprint"),
                ArtifactBinding(a4_ref, "knowledge_atom"),
                ArtifactBinding(e23_ref, "supporting_evidence"),
            ),
            produced_artifacts=(q12_ref, q13_ref),
        )

        occ_verify = OccurrenceRecord(
            occurrence_id="verify#1",
            node_id="verify_candidate_mcqs_against_evidence",
            occurrence_index=1,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=5,
            superstep=4,
            checkpoint_before="c4",
            checkpoint_after="c5",
            input_bindings=(
                ArtifactBinding(q12_ref, "candidate_question"),
                ArtifactBinding(bp12_ref, "verification_blueprint"),
                ArtifactBinding(a3_ref, "grounding_atom"),
                ArtifactBinding(e17_ref, "supporting_evidence"),
                ArtifactBinding(q13_ref, "candidate_question"),
                ArtifactBinding(bp13_ref, "verification_blueprint"),
                ArtifactBinding(a4_ref, "grounding_atom"),
                ArtifactBinding(e23_ref, "supporting_evidence"),
            ),
            produced_artifacts=(er12_ref, er13_ref),
        )

        occ_cross = OccurrenceRecord(
            occurrence_id="cross#1",
            node_id="cross_cove_verify_candidate_mcqs",
            occurrence_index=1,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=6,
            superstep=5,
            checkpoint_before="c5",
            checkpoint_after="c6",
            input_bindings=(
                ArtifactBinding(q12_ref, "candidate_question"),
                ArtifactBinding(er12_ref, "evidence_verification_gate"),
                ArtifactBinding(q13_ref, "candidate_question"),
                ArtifactBinding(er13_ref, "evidence_verification_gate"),
            ),
            produced_artifacts=(cr12_ref, cr13_ref),
        )

        occ_apply = OccurrenceRecord(
            occurrence_id="apply#1",
            node_id="apply_acceptance_gate",
            occurrence_index=1,
            branch_id=branch_id,
            run_id=run_id,
            execution_index=7,
            superstep=6,
            checkpoint_before="c6",
            checkpoint_after="c7",
            input_bindings=(
                ArtifactBinding(q12_ref, "candidate_question"),
                ArtifactBinding(bp12_ref, "blueprint_constraints"),
                ArtifactBinding(er12_ref, "evidence_report"),
                ArtifactBinding(cr12_ref, "cross_cove_report"),
                ArtifactBinding(q13_ref, "candidate_question"),
                ArtifactBinding(bp13_ref, "blueprint_constraints"),
                ArtifactBinding(er13_ref, "evidence_report"),
                ArtifactBinding(cr13_ref, "cross_cove_report"),
            ),
            produced_artifacts=(decision12_ref, assessment12_ref, decision13_ref, assessment13_ref),
        )

        occurrences = [
            occ_doc_s2,
            occ_doc_s3,
            occ_lo3,
            occ_lo4,
            occ_e17,
            occ_e23,
            occ_a3,
            occ_a4,
            occ_plan,
            occ_generate,
            occ_verify,
            occ_cross,
            occ_apply,
        ]

        # ProducedBy mapping (authoritative)
        produced_by = []
        for artifact, producer in [
            (ArtifactRef("S2", 1), occ_doc_s2.occurrence_id),
            (ArtifactRef("S3", 1), occ_doc_s3.occurrence_id),
            (ArtifactRef("LO3", 1), occ_lo3.occurrence_id),
            (ArtifactRef("LO4", 1), occ_lo4.occurrence_id),
            (e17_ref, occ_e17.occurrence_id),
            (e23_ref, occ_e23.occurrence_id),
            (a3_ref, occ_a3.occurrence_id),
            (a4_ref, occ_a4.occurrence_id),
            (bp12_ref, occ_plan.occurrence_id),
            (bp13_ref, occ_plan.occurrence_id),
            (q12_ref, occ_generate.occurrence_id),
            (q13_ref, occ_generate.occurrence_id),
            (er12_ref, occ_verify.occurrence_id),
            (er13_ref, occ_verify.occurrence_id),
            (cr12_ref, occ_cross.occurrence_id),
            (cr13_ref, occ_cross.occurrence_id),
            (decision12_ref, occ_apply.occurrence_id),
            (assessment12_ref, occ_apply.occurrence_id),
            (decision13_ref, occ_apply.occurrence_id),
            (assessment13_ref, occ_apply.occurrence_id),
        ]:
            produced_by.append(ProducedBy(artifact, producer))

        # D_run relations
        rels = []

        def rel(producer_id, artifact_ref, relation, consumer_id):
            rels.append(
                RunProvenanceRelation(
                    producer_occurrence_id=producer_id,
                    artifact_ref=artifact_ref,
                    relation=relation,
                    consumer_occurrence_id=consumer_id,
                )
            )

        # plan inputs
        rel(occ_doc_s2.occurrence_id, ArtifactRef("S2", 1), "document_section", occ_plan.occurrence_id)
        rel(occ_lo3.occurrence_id, ArtifactRef("LO3", 1), "learning_objective", occ_plan.occurrence_id)
        rel(occ_a3.occurrence_id, a3_ref, "planning_atom", occ_plan.occurrence_id)
        rel(occ_e17.occurrence_id, e17_ref, "planning_evidence", occ_plan.occurrence_id)

        rel(occ_doc_s3.occurrence_id, ArtifactRef("S3", 1), "document_section", occ_plan.occurrence_id)
        rel(occ_lo4.occurrence_id, ArtifactRef("LO4", 1), "learning_objective", occ_plan.occurrence_id)
        rel(occ_a4.occurrence_id, a4_ref, "planning_atom", occ_plan.occurrence_id)
        rel(occ_e23.occurrence_id, e23_ref, "planning_evidence", occ_plan.occurrence_id)

        # generation inputs
        rel(occ_plan.occurrence_id, bp12_ref, "generation_blueprint", occ_generate.occurrence_id)
        rel(occ_a3.occurrence_id, a3_ref, "knowledge_atom", occ_generate.occurrence_id)
        rel(occ_e17.occurrence_id, e17_ref, "supporting_evidence", occ_generate.occurrence_id)

        rel(occ_plan.occurrence_id, bp13_ref, "generation_blueprint", occ_generate.occurrence_id)
        rel(occ_a4.occurrence_id, a4_ref, "knowledge_atom", occ_generate.occurrence_id)
        rel(occ_e23.occurrence_id, e23_ref, "supporting_evidence", occ_generate.occurrence_id)

        # verify inputs
        rel(occ_generate.occurrence_id, q12_ref, "candidate_question", occ_verify.occurrence_id)
        rel(occ_plan.occurrence_id, bp12_ref, "verification_blueprint", occ_verify.occurrence_id)
        rel(occ_a3.occurrence_id, a3_ref, "grounding_atom", occ_verify.occurrence_id)
        rel(occ_e17.occurrence_id, e17_ref, "supporting_evidence", occ_verify.occurrence_id)

        rel(occ_generate.occurrence_id, q13_ref, "candidate_question", occ_verify.occurrence_id)
        rel(occ_plan.occurrence_id, bp13_ref, "verification_blueprint", occ_verify.occurrence_id)
        rel(occ_a4.occurrence_id, a4_ref, "grounding_atom", occ_verify.occurrence_id)
        rel(occ_e23.occurrence_id, e23_ref, "supporting_evidence", occ_verify.occurrence_id)

        # cross inputs
        rel(occ_generate.occurrence_id, q12_ref, "candidate_question", occ_cross.occurrence_id)
        rel(occ_verify.occurrence_id, er12_ref, "evidence_verification_gate", occ_cross.occurrence_id)

        rel(occ_generate.occurrence_id, q13_ref, "candidate_question", occ_cross.occurrence_id)
        rel(occ_verify.occurrence_id, er13_ref, "evidence_verification_gate", occ_cross.occurrence_id)

        # apply inputs
        rel(occ_generate.occurrence_id, q12_ref, "candidate_question", occ_apply.occurrence_id)
        rel(occ_plan.occurrence_id, bp12_ref, "blueprint_constraints", occ_apply.occurrence_id)
        rel(occ_verify.occurrence_id, er12_ref, "evidence_report", occ_apply.occurrence_id)
        rel(occ_cross.occurrence_id, cr12_ref, "cross_cove_report", occ_apply.occurrence_id)

        rel(occ_generate.occurrence_id, q13_ref, "candidate_question", occ_apply.occurrence_id)
        rel(occ_plan.occurrence_id, bp13_ref, "blueprint_constraints", occ_apply.occurrence_id)
        rel(occ_verify.occurrence_id, er13_ref, "evidence_report", occ_apply.occurrence_id)
        rel(occ_cross.occurrence_id, cr13_ref, "cross_cove_report", occ_apply.occurrence_id)

        return {
            "registry": reg,
            "occurrences": occurrences,
            "provenance_relations": rels,
            "produced_by": produced_by,
            "artifact_derivations": [],
            "intervention_events": [],
            "q12_ref": q12_ref,
            "q13_ref": q13_ref,
            "bp12_ref": bp12_ref,
            "bp13_ref": bp13_ref,
            "er12_ref": er12_ref,
            "er13_ref": er13_ref,
            "cr12_ref": cr12_ref,
            "cr13_ref": cr13_ref,
            "decision12_ref": decision12_ref,
            "assessment12_ref": assessment12_ref,
            "criterion_values": cv,
        }

    def test_decision_root_traversal_and_q12_scoping(self):
        fixture = self._build_q12_q13_factual_fixture(q12_decision="accept")
        reg = fixture["registry"]

        explained = resolve_mcq_explained_case(
            run_id="run_p12_factual",
            branch_id="factual",
            artifact_ref=fixture["q12_ref"],
            registry=reg,
        )
        slicer = ProvenanceSlicer()
        adapter = MCQProvenanceSliceAdapter()

        case_slice = slicer.backward_slice(
            explained,
            fixture["occurrences"],
            fixture["provenance_relations"],
            registry=reg,
            produced_by=fixture["produced_by"],
            artifact_derivations=fixture["artifact_derivations"],
            intervention_events=fixture["intervention_events"],
            adapter=adapter,
            strict=True,
        )

        node_ids = {o.node_id for o in case_slice.relevant_occurrences}
        self.assertIn("apply_acceptance_gate", node_ids)
        self.assertIn("generate_candidate_mcqs_from_blueprints", node_ids)
        self.assertIn("verify_candidate_mcqs_against_evidence", node_ids)
        self.assertIn("cross_cove_verify_candidate_mcqs", node_ids)
        self.assertIn("plan_mcq_blueprints", node_ids)

        refs = set(case_slice.relevant_artifact_refs)
        self.assertIn(fixture["decision12_ref"], refs)
        self.assertIn(fixture["assessment12_ref"], refs)
        self.assertIn(fixture["q12_ref"], refs)
        self.assertIn(fixture["er12_ref"], refs)
        self.assertIn(fixture["cr12_ref"], refs)
        self.assertIn(fixture["bp12_ref"], refs)

        # Must exclude unrelated Q13 artifacts processed in same occurrences.
        self.assertNotIn(fixture["q13_ref"], refs)
        self.assertNotIn(fixture["er13_ref"], refs)
        self.assertNotIn(fixture["cr13_ref"], refs)
        self.assertNotIn(fixture["bp13_ref"], refs)

    def test_exact_assessment_criteria_and_decision_details(self):
        fixture = self._build_q12_q13_factual_fixture(q12_decision="accept")
        reg = fixture["registry"]

        explained = resolve_mcq_explained_case(
            run_id="run_p12_factual",
            branch_id="factual",
            artifact_ref=fixture["q12_ref"],
            registry=reg,
        )

        slicer = ProvenanceSlicer()
        case_slice = slicer.backward_slice(
            explained,
            fixture["occurrences"],
            fixture["provenance_relations"],
            registry=reg,
            produced_by=fixture["produced_by"],
            adapter=MCQProvenanceSliceAdapter(),
            strict=True,
        )

        self.assertEqual(case_slice.assessment_ref, fixture["assessment12_ref"])
        self.assertEqual(case_slice.criterion_values, fixture["criterion_values"])
        self.assertEqual(case_slice.decision, "accept")
        self.assertEqual(case_slice.decision_reasons, ("r1",))
        self.assertEqual(case_slice.required_fixes, ("fix1",))

    def test_unrelated_candidate_exclusion_in_generation_occurrence(self):
        fixture = self._build_q12_q13_factual_fixture(q12_decision="accept")
        reg = fixture["registry"]

        explained = resolve_mcq_explained_case(
            run_id="run_p12_factual",
            branch_id="factual",
            artifact_ref=fixture["q12_ref"],
            registry=reg,
        )
        case_slice = ProvenanceSlicer().backward_slice(
            explained,
            fixture["occurrences"],
            fixture["provenance_relations"],
            registry=reg,
            produced_by=fixture["produced_by"],
            adapter=MCQProvenanceSliceAdapter(),
            strict=True,
        )
        refs = set(case_slice.relevant_artifact_refs)
        # Q12 blueprint evidence / atoms should be present;
        # Q13 blueprint evidence / atoms must be absent.
        self.assertIn(ArtifactRef("E17", 1), refs)
        self.assertIn(ArtifactRef("A3", 1), refs)
        self.assertNotIn(ArtifactRef("E23", 1), refs)
        self.assertNotIn(ArtifactRef("A4", 1), refs)

    def test_intervention_supplied_boundary(self):
        slicer = ProvenanceSlicer()
        alpha = ArtifactRef("E99", 1)

        ev = InterventionEvent(
            intervention_id="int1",
            run_id="run_p12_factual",
            branch_id="factual",
            operation="ADD",
            origin_checkpoint_alias="cX",
            origin_checkpoint_id="idX",
            artifact_refs=(alpha,),
            supplied_refs=(alpha,),
        )

        explained = ExplainedCase(
            run_id="run_p12_factual",
            artifact_ref=alpha,
            branch_id="factual",
            decision_ref=alpha,
            outcome="accept",
        )

        case_slice = slicer.backward_slice(
            explained,
            occurrence_records=[],
            provenance_relations=[],
            intervention_events=[ev],
            strict=False,
        )

        refs = set(case_slice.relevant_artifact_refs)
        self.assertIn(alpha, refs)
        self.assertTrue(
            any(b.kind == "intervention_supplied" and b.artifact_ref == alpha for b in case_slice.source_boundaries)
        )

    def test_invalid_decision_version_strict_raises(self):
        fixture = self._build_q12_q13_factual_fixture(q12_decision="accept")
        reg = fixture["registry"]

        # Candidate claims v2, but decision payload is for v1.
        explained = ExplainedCase(
            run_id="run_p12_factual",
            artifact_ref=ArtifactRef("Q12", 2),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:Q12", 1),
            outcome="accept",
        )

        with self.assertRaises(SliceValidationError):
            ProvenanceSlicer().backward_slice(
                explained,
                fixture["occurrences"],
                fixture["provenance_relations"],
                registry=reg,
                produced_by=fixture["produced_by"],
                adapter=MCQProvenanceSliceAdapter(),
                strict=True,
            )

    def test_ambiguous_producer_strict_raises(self):
        slicer = ProvenanceSlicer()
        cand = ArtifactRef("Q12", 1)
        decision = ArtifactRef("decision:Q12", 1)

        occ_a = OccurrenceRecord(
            occurrence_id="pA",
            node_id="x",
            occurrence_index=1,
            branch_id="factual",
            run_id="run",
            execution_index=1,
            produced_artifacts=(cand, decision),
            input_bindings=(),
        )
        occ_b = OccurrenceRecord(
            occurrence_id="pB",
            node_id="y",
            occurrence_index=1,
            branch_id="factual",
            run_id="run",
            execution_index=2,
            produced_artifacts=(cand,),
            input_bindings=(),
        )

        explained = ExplainedCase(
            run_id="run",
            artifact_ref=cand,
            branch_id="factual",
            decision_ref=decision,
            outcome="accept",
        )

        produced_by = [
            ProducedBy(artifact_ref=cand, producer_occurrence_id="pA"),
            ProducedBy(artifact_ref=cand, producer_occurrence_id="pB"),  # ambiguity
            ProducedBy(artifact_ref=decision, producer_occurrence_id="pA"),
        ]

        with self.assertRaises(SliceValidationError):
            slicer.backward_slice(
                explained,
                occurrence_records=[occ_a, occ_b],
                provenance_relations=[],
                produced_by=produced_by,
                strict=True,
            )

    def test_missing_runtime_relation_strict_raises(self):
        slicer = ProvenanceSlicer()
        out = ArtifactRef("OUT", 1)
        beta = ArtifactRef("IN", 1)

        consumer = OccurrenceRecord(
            occurrence_id="cons",
            node_id="apply_acceptance_gate",
            occurrence_index=1,
            branch_id="factual",
            run_id="run",
            execution_index=1,
            produced_artifacts=(out,),
            input_bindings=(ArtifactBinding(beta, "r1"),),
        )

        explained = ExplainedCase(
            run_id="run",
            artifact_ref=out,
            branch_id="factual",
            decision_ref=out,
            outcome="accept",
        )

        def adapter_fn(*, output_ref, producer_occurrence, candidate_bindings, registry, branch_id):
            return candidate_bindings  # force traversal requiring D_run attribution

        produced_by = [ProducedBy(artifact_ref=out, producer_occurrence_id="cons")]

        with self.assertRaises(SliceValidationError):
            slicer.backward_slice(
                explained,
                occurrence_records=[consumer],
                provenance_relations=[],
                produced_by=produced_by,
                adapter=adapter_fn,
                strict=True,
            )

    def test_deterministic_repeatability(self):
        fixture = self._build_q12_q13_factual_fixture(q12_decision="accept")
        reg = fixture["registry"]
        explained = resolve_mcq_explained_case(
            run_id="run_p12_factual",
            branch_id="factual",
            artifact_ref=fixture["q12_ref"],
            registry=reg,
        )
        slicer = ProvenanceSlicer()
        adapter = MCQProvenanceSliceAdapter()

        a = slicer.backward_slice(
            explained,
            fixture["occurrences"],
            fixture["provenance_relations"],
            registry=reg,
            produced_by=fixture["produced_by"],
            adapter=adapter,
            strict=True,
        )
        b = slicer.backward_slice(
            explained,
            fixture["occurrences"],
            fixture["provenance_relations"],
            registry=reg,
            produced_by=fixture["produced_by"],
            adapter=adapter,
            strict=True,
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

