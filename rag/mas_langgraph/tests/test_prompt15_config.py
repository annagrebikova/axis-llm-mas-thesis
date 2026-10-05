"""Prompt 15 — case configuration, AXIS capabilities, generic isolation."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

from mas_langgraph.artifacts.models import ArtifactEnvelope, ArtifactRef
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.axis.explainer import AxisExplanationController
from mas_langgraph.axis.models import AxisPromptTemplates, ExplainedCase
from mas_langgraph.axis.slicing import CaseSlice, ProvenanceSlicer
from mas_langgraph.cases.example_case import ExampleCaseDefinition
from mas_langgraph.cases.mcq import MCQCaseDefinition
from mas_langgraph.cases.mcq.comparison import MCQRunObservationAdapter
from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer
from mas_langgraph.core.case_config import (
    CaseConfigurationError,
    extract_graph_node_ids,
    load_axis_case_capabilities,
    load_case_configuration,
)
from mas_langgraph.core.outcomes import CriterionValue
from mas_langgraph.core.runtime import WorkflowRuntime
from mas_langgraph.core.workflow_executor import WorkflowExecutor
from mas_langgraph.replay.checkpoints import CheckpointIndex
from mas_langgraph.replay.executor import LangGraphReplayExecutor, WorkflowRunResult
from mas_langgraph.replay.interventions import InterventionContractRegistry


_MCQ_INPUTS = Path(__file__).resolve().parents[1] / "cases" / "mcq" / "inputs"


def _copy_mcq_inputs(tmp: Path) -> Path:
    dest = tmp / "inputs"
    shutil.copytree(_MCQ_INPUTS, dest)
    return dest


class TestCaseMetadataLoad(unittest.TestCase):
    def test_complete_mcq_configuration_loads(self):
        cfg = MCQCaseDefinition().load_case_configuration()
        self.assertEqual(cfg.metadata.case_id, "mcq_generation")
        self.assertEqual(cfg.metadata.case_version, 1)
        self.assertTrue(cfg.workflow.nodes)
        self.assertIn("CandidateMCQ", cfg.artifact_types)
        self.assertTrue(cfg.provenance_schema.relations)
        self.assertIsNotNone(cfg.outcome_evaluator)
        self.assertIsNotNone(cfg.intervention_contracts.get("EvidenceChunk"))

    def test_case_yaml_identity_match(self):
        meta = MCQCaseDefinition().load_case_metadata()
        self.assertEqual(meta.case_id, MCQCaseDefinition.case_id)
        self.assertEqual(meta.case_version, MCQCaseDefinition.case_version)

    def test_case_id_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            text = (dest / "case.yaml").read_text(encoding="utf-8")
            (dest / "case.yaml").write_text(
                text.replace("case_id: mcq_generation", "case_id: wrong_case"),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_bad_python_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            text = (dest / "case.yaml").read_text(encoding="utf-8")
            (dest / "case.yaml").write_text(
                text.replace(
                    "mas_langgraph.cases.mcq.case_definition.MCQCaseDefinition",
                    "foo.bar.DoesNotExist",
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()


class TestWorkflowMetadata(unittest.TestCase):
    def test_all_real_graph_nodes_declared(self):
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph

        graph = build_mcq_workflow_graph()
        actual = extract_graph_node_ids(graph)
        self.assertEqual(len(actual), 22)
        cfg = MCQCaseDefinition().load_case_configuration(
            compiled_graph=graph, require_graph=True
        )
        self.assertEqual(cfg.workflow.node_ids(), actual)

    def test_missing_workflow_node_metadata(self):
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph

        graph = build_mcq_workflow_graph()
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            yaml_text = (dest / "workflow.yaml").read_text(encoding="utf-8")
            yaml_text = yaml_text.replace("apply_acceptance_gate", "apply_acceptance_gate_removed")
            (dest / "workflow.yaml").write_text(yaml_text, encoding="utf-8")
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration(
                    compiled_graph=graph, require_graph=True
                )

    def test_unknown_yaml_node(self):
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph

        graph = build_mcq_workflow_graph()
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            extra = (
                "  - id: fake_llm_supervisor\n"
                "    label: Fake\n"
                "    type: agent\n"
                "    purpose: fake\n"
                "    inputs: []\n"
                "    outputs: []\n"
                "    status: active\n"
            )
            path = dest / "workflow.yaml"
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("routing:", extra + "\nrouting:"), encoding="utf-8")
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration(
                    compiled_graph=graph, require_graph=True
                )

    def test_unwired_compatibility_nodes_preserved(self):
        wf = MCQCaseDefinition().load_workflow_metadata()
        unwired = {n.node_id for n in wf.nodes if n.status == "compatibility_unwired"}
        self.assertEqual(
            unwired,
            {
                "retrieve_evidence",
                "verify_candidates",
                "package_or_end",
                "end_with_error",
            },
        )


class TestArtifactAndSchemaConfig(unittest.TestCase):
    def test_docsection_and_learning_objective_declared(self):
        types = MCQCaseDefinition().load_artifact_types()
        self.assertIn("DocSection", types)
        self.assertIn("LearningObjective", types)

    def test_every_provenance_type_declared(self):
        cfg = MCQCaseDefinition().load_case_configuration()
        for rel in cfg.provenance_schema.relations:
            self.assertIn(rel.artifact_type, cfg.artifact_types)

    def test_unknown_provenance_node_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "provenance_schema.yaml"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(
                    "consumer: generate_learning_objectives",
                    "consumer: nonexistent_node",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_unknown_provenance_artifact_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "provenance_schema.yaml"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace("artifact_type: DocSection", "artifact_type: HallucinatedArtifact", 1),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_invalid_serialization_adapter_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "artifact_types.yaml"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(
                    "mas_langgraph.cases.mcq.artifacts.wrap_evidence_chunk",
                    "mas_langgraph.foo.not_a_function",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()


class TestContractAndCriteriaConfig(unittest.TestCase):
    def test_contract_unknown_artifact_type_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "intervention_contracts.yaml"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace("artifact_type: EvidenceChunk", "artifact_type: ImaginaryArtifact", 1),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_non_intervenable_contracted_type_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "artifact_types.yaml"
            text = path.read_text(encoding="utf-8")
            # First EvidenceChunk block: intervenable true -> false
            path.write_text(
                text.replace(
                    "  - name: EvidenceChunk\n    description: Retrieved PDF evidence chunk\n"
                    "    explanation_target: true\n    intervenable: true",
                    "  - name: EvidenceChunk\n    description: Retrieved PDF evidence chunk\n"
                    "    explanation_target: true\n    intervenable: false",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_unknown_replacement_type_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "intervention_contracts.yaml"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(
                    "replacement_compatible_types: [EvidenceChunk]",
                    "replacement_compatible_types: [FooArtifact]",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_inconsistent_external_add_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "intervention_contracts.yaml"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(
                    "allowed_operations: [REMOVE, ADD, REPLACE]",
                    "allowed_operations: [REMOVE, REPLACE]",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_duplicate_outcome_criterion_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "outcome_criteria.yaml"
            text = path.read_text(encoding="utf-8")
            dup = (
                "  - name: pdf_supported\n"
                "    label: Dup\n"
                "    value_type: bool\n"
                "    decision_relevant: true\n"
                "    explanation: dup\n"
            )
            path.write_text(text + "\n" + dup, encoding="utf-8")
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_unsupported_value_type_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "outcome_criteria.yaml"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace("value_type: int", "value_type: magical_boolean", 1),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_unimplemented_decision_criterion_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            path = dest / "outcome_criteria.yaml"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(
                    "    implemented: false\n    explanation:",
                    "    implemented: false\n    decision_relevant: true\n    explanation:",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()

    def test_missing_strict_yaml_fails(self):
        with tempfile.TemporaryDirectory() as td:
            dest = _copy_mcq_inputs(Path(td))
            (dest / "intervention_contracts.yaml").unlink()
            with self.assertRaises(CaseConfigurationError):
                MCQCaseDefinition(dest).load_case_configuration()


class TestAxisCapabilities(unittest.TestCase):
    def test_full_axis_capabilities_load(self):
        caps = load_axis_case_capabilities(MCQCaseDefinition(), strict=True)
        self.assertIsNotNone(caps.case_configuration)
        self.assertIsNotNone(caps.verbalizer)
        self.assertTrue(caps.prompt_templates.system)
        self.assertTrue(caps.prompt_templates.factual_context)
        self.assertTrue(caps.prompt_templates.interrogation)
        self.assertTrue(caps.prompt_templates.intermediate_explanation)
        self.assertTrue(caps.prompt_templates.final_explanation)
        self.assertIsNotNone(caps.slicing_adapter)
        self.assertIsNotNone(caps.observation_adapter)
        self.assertIsNotNone(caps.provenance_instrumentation_adapter)
        self.assertIsNotNone(caps.explained_case_resolver)


class TestFactualSlicePath(unittest.TestCase):
    def _explained(self):
        return ExplainedCase(
            run_id="runA",
            artifact_ref=ArtifactRef("Q12", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:Q12", 1),
            outcome="accept",
        )

    def test_controller_uses_slice_run_result(self):
        explained = self._explained()
        fr = WorkflowRunResult(
            run_id="runA",
            branch_id="factual",
            final_state={"occurrence_records": [], "run_provenance_relations": []},
        )
        called = {"n": 0, "strict": None}

        def fake_slice(self, explained_case, run_result, **kwargs):
            called["n"] += 1
            called["strict"] = kwargs.get("strict")
            return CaseSlice(
                explained_case=explained_case,
                relevant_artifact_refs=(explained_case.artifact_ref,),
            )

        ex = LangGraphReplayExecutor(compiled_graph=None)
        ex.run_factual({"step_logs": [{}]}, run_id="runA", allow_synthetic_history=True)
        idx = ex.get_checkpoint_index("runA")
        with patch.object(ProvenanceSlicer, "slice_run_result", fake_slice):
            controller = AxisExplanationController(
                explained_case=explained,
                registry=ArtifactRegistry(),
                occurrences=[],
                provenance_relations=[],
                checkpoint_index=idx,
                replay_executor=ex,
                factual_run_result=fr,
                compatibility_mode=True,
            )
            controller.run("Why?", scripted_queries=["DONE"])
        self.assertEqual(called["n"], 1)
        self.assertTrue(called["strict"])

    def test_factual_run_id_mismatch_fails(self):
        explained = self._explained()
        fr = WorkflowRunResult(run_id="runB", branch_id="factual", final_state={})
        ex = LangGraphReplayExecutor(compiled_graph=None)
        idx = CheckpointIndex(thread_id="runA")
        controller = AxisExplanationController(
            explained_case=explained,
            registry=ArtifactRegistry(),
            occurrences=[],
            provenance_relations=[],
            checkpoint_index=idx,
            replay_executor=ex,
            factual_run_result=fr,
            compatibility_mode=True,
        )
        with self.assertRaises(CaseConfigurationError):
            controller.run("Why?", scripted_queries=["DONE"])

    def test_factual_branch_mismatch_fails(self):
        explained = self._explained()
        fr = WorkflowRunResult(run_id="runA", branch_id="cf_1", final_state={})
        ex = LangGraphReplayExecutor(compiled_graph=None)
        idx = CheckpointIndex(thread_id="runA")
        controller = AxisExplanationController(
            explained_case=explained,
            registry=ArtifactRegistry(),
            occurrences=[],
            provenance_relations=[],
            checkpoint_index=idx,
            replay_executor=ex,
            factual_run_result=fr,
            compatibility_mode=False,
        )
        with self.assertRaises(CaseConfigurationError):
            controller.run("Why?", scripted_queries=["DONE"])

    def test_conflicting_factual_registry_fails(self):
        explained = self._explained()
        env_a = ArtifactEnvelope.create(
            logical_id="Q12", version=1, artifact_type="CandidateMCQ",
            payload={"stem": "A"}, branch_id="factual", created_in_run="runA",
        )
        env_b = ArtifactEnvelope.create(
            logical_id="Q12", version=1, artifact_type="CandidateMCQ",
            payload={"stem": "B"}, branch_id="factual", created_in_run="runA",
        )
        supplied = ArtifactRegistry()
        supplied.store(env_a)
        fact_reg = ArtifactRegistry()
        fact_reg.store(env_b)
        fr = WorkflowRunResult(
            run_id="runA",
            branch_id="factual",
            final_state={"artifact_catalog": fact_reg.catalog},
        )
        ex = LangGraphReplayExecutor(compiled_graph=None)
        idx = CheckpointIndex(thread_id="runA")
        controller = AxisExplanationController(
            explained_case=explained,
            registry=supplied,
            occurrences=[],
            provenance_relations=[],
            checkpoint_index=idx,
            replay_executor=ex,
            factual_run_result=fr,
            compatibility_mode=True,
        )
        with self.assertRaises(CaseConfigurationError):
            controller.run("Why?", scripted_queries=["DONE"])


class TestFactualContextSections(unittest.TestCase):
    def test_sections_populated(self):
        explained = ExplainedCase(
            run_id="run1",
            artifact_ref=ArtifactRef("Q12", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:Q12", 1),
            outcome="reject",
            assessment_ref=ArtifactRef("assessment:Q12", 1),
        )
        cv = CriterionValue(name="pdf_supported", value=False, decision_relevant=True)
        from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation

        occ = OccurrenceRecord(
            occurrence_id="run1:factual:gen#1",
            node_id="gen",
            occurrence_index=1,
            branch_id="factual",
            run_id="run1",
        )
        rel = RunProvenanceRelation(
            producer_occurrence_id="run1:factual:gen#1",
            artifact_ref=ArtifactRef("E17", 1),
            relation="supporting_evidence",
            consumer_occurrence_id="run1:factual:verify#1",
        )
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=(ArtifactRef("E17", 1), ArtifactRef("Q12", 1)),
            relevant_occurrences=(occ,),
            relevant_relations=(rel,),
            criterion_values=(cv,),
            decision="reject",
            decision_reasons=("unsupported",),
        )
        reg = ArtifactRegistry()
        e17 = ArtifactEnvelope.create(
            logical_id="E17", version=1, artifact_type="EvidenceChunk",
            payload=_Chunk("E17", "The PDF states the limit is 3."),
            branch_id="factual", created_in_run="run1",
        )
        q12 = ArtifactEnvelope.create(
            logical_id="Q12", version=1, artifact_type="CandidateMCQ",
            payload=_Cand("Q12", 0, "What is the limit?"),
            branch_id="factual", created_in_run="run1",
        )
        bp = ArtifactEnvelope.create(
            logical_id="BP1", version=1, artifact_type="MCQBlueprint",
            payload=_Blueprint("BP1", ["E17"]),
            branch_id="factual", created_in_run="run1",
        )
        alt = ArtifactEnvelope.create(
            logical_id="E99", version=1, artifact_type="EvidenceChunk",
            payload=_Chunk("E99", "Alternative evidence about limits."),
            branch_id="factual", created_in_run="run1",
        )
        for env in (e17, q12, bp, alt):
            reg.store(env)
        v = McqVerbalizer()
        sections = v.factual_context_sections(
            explained_case=explained,
            case_slice=case_slice,
            registry=reg,
            checkpoint_index=CheckpointIndex(thread_id="t"),
            contracts=InterventionContractRegistry(),
            intervention_alternatives={"E99": ArtifactRef("E99", 1)},
        )
        for key in (
            "criteria",
            "decision_details",
            "factual_artifacts",
            "occurrences",
            "provenance_relations",
            "intervention_space",
            "valid_boundaries",
        ):
            self.assertTrue(str(sections[key]).strip(), key)
        self.assertIn("The PDF states the limit is 3.", sections["factual_artifacts"])
        self.assertIn("controlled alternative", sections["controlled_alternatives"])
        self.assertIn("Alternative evidence", sections["controlled_alternatives"])
        self.assertIn("evidence=['E17']", v.verbalize_artifact(bp))

    def test_strict_template_render_error(self):
        templates = AxisPromptTemplates(
            factual_context="Need {missing_placeholder} and {explained_case}"
        )
        v = McqVerbalizer()
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("Q12", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:Q12", 1),
            outcome="accept",
        )
        case_slice = CaseSlice(explained_case=explained)
        from mas_langgraph.axis.explainer import _build_factual_context

        with self.assertRaises((ValueError, CaseConfigurationError, KeyError)):
            _build_factual_context(
                question="q",
                explained_case=explained,
                case_slice=case_slice,
                registry=ArtifactRegistry(),
                checkpoint_index=CheckpointIndex(thread_id="t"),
                contracts=InterventionContractRegistry(),
                verbalizer=v,
                prompt_templates=templates,
                strict=True,
            )


@dataclass
class _Chunk:
    chunk_id: str
    content: str
    section: str = ""
    page: str = ""


@dataclass
class _Cand:
    q_id: str
    revision_count: int = 0
    stem: str = "stem"


@dataclass
class _Blueprint:
    blueprint_id: str
    evidence_chunk_ids: list
    target_atom_ids: list = field(default_factory=list)
    learning_objective: str = "LO"


@dataclass
class _Decision:
    q_id: str
    decision: str
    reasons: list
    required_fixes: list
    candidate_version: int
    assessment_ref: Optional[ArtifactRef] = None


@dataclass
class _Assessment:
    assessment_ref: ArtifactRef
    target_ref: ArtifactRef
    criteria: tuple


@dataclass
class _CS:
    candidate_mcqs: dict
    supervisor_decisions: dict = field(default_factory=dict)
    outcome_assessments: dict = field(default_factory=dict)


class TestReplayObservation(unittest.TestCase):
    def _run(self, cs, catalog, branch="cf"):
        return WorkflowRunResult(
            run_id="run1",
            branch_id=branch,
            final_state={
                "cooperative_state": cs,
                "artifact_catalog": catalog,
                "branch_id": branch,
            },
        )

    def test_revised_candidate_version(self):
        cand = _Cand("Q12", revision_count=1)
        env = ArtifactEnvelope.create(
            logical_id="Q12", version=2, artifact_type="CandidateMCQ",
            payload=cand, branch_id="cf", created_in_run="run1",
        )
        catalog = {env.instance_id: env}
        explained = ExplainedCase(
            run_id="run1", artifact_ref=ArtifactRef("Q12", 1),
            branch_id="factual", decision_ref=ArtifactRef("decision:Q12", 1),
            outcome="reject",
        )
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=explained,
            run_result=self._run(_CS(candidate_mcqs={"Q12": cand}), catalog),
        )
        self.assertEqual(obs.subject_ref, ArtifactRef("Q12", 2))

    def test_assessment_criteria_and_decision_fields(self):
        cand = _Cand("Q12", 1)
        cv = CriterionValue(name="pdf_supported", value=True)
        aref = ArtifactRef("assessment:Q12", 2)
        dref = ArtifactRef("decision:Q12", 2)
        assess = _Assessment(aref, ArtifactRef("Q12", 2), (cv,))
        dec = _Decision(
            "Q12", "accept", ["ok"], ["none"], 2, assessment_ref=aref
        )
        cenv = ArtifactEnvelope.create(
            logical_id="Q12", version=2, artifact_type="CandidateMCQ",
            payload=cand, branch_id="cf", created_in_run="run1",
        )
        aenv = ArtifactEnvelope.create(
            logical_id="assessment:Q12", version=2, artifact_type="OutcomeAssessment",
            payload=assess, branch_id="cf", created_in_run="run1",
        )
        denv = ArtifactEnvelope.create(
            logical_id="decision:Q12", version=2, artifact_type="SupervisorDecision",
            payload=dec, branch_id="cf", created_in_run="run1",
        )
        catalog = {e.instance_id: e for e in (cenv, aenv, denv)}
        explained = ExplainedCase(
            run_id="run1", artifact_ref=ArtifactRef("Q12", 1),
            branch_id="factual", decision_ref=ArtifactRef("decision:Q12", 1),
            outcome="reject",
        )
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=explained,
            run_result=self._run(
                _CS(
                    candidate_mcqs={"Q12": cand},
                    supervisor_decisions={"Q12": dec},
                    outcome_assessments={"Q12": assess},
                ),
                catalog,
            ),
        )
        self.assertEqual(obs.criterion_values, (cv,))
        self.assertEqual(obs.decision_reasons, ("ok",))
        self.assertEqual(obs.required_fixes, ("none",))
        self.assertEqual(obs.decision_ref, dref)
        self.assertEqual(obs.assessment_ref, aref)

    def test_stale_decision_rejected(self):
        cand = _Cand("Q12", 1)
        cenv = ArtifactEnvelope.create(
            logical_id="Q12", version=2, artifact_type="CandidateMCQ",
            payload=cand, branch_id="cf", created_in_run="run1",
        )
        dec = _Decision("Q12", "accept", [], [], 1)
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=ExplainedCase(
                run_id="run1", artifact_ref=ArtifactRef("Q12", 1),
                branch_id="factual", decision_ref=ArtifactRef("decision:Q12", 1),
                outcome="reject",
            ),
            run_result=self._run(
                _CS(candidate_mcqs={"Q12": cand}, supervisor_decisions={"Q12": dec}),
                {cenv.instance_id: cenv},
            ),
        )
        self.assertFalse(obs.outcome_available)
        self.assertIsNone(obs.decision)
        self.assertIn("stale decision", obs.error or "")

    def test_stale_assessment_rejected(self):
        cand = _Cand("Q12", 1)
        cenv = ArtifactEnvelope.create(
            logical_id="Q12", version=2, artifact_type="CandidateMCQ",
            payload=cand, branch_id="cf", created_in_run="run1",
        )
        aref = ArtifactRef("assessment:Q12", 2)
        assess = _Assessment(aref, ArtifactRef("Q12", 1), ())
        aenv = ArtifactEnvelope.create(
            logical_id="assessment:Q12", version=2, artifact_type="OutcomeAssessment",
            payload=assess, branch_id="cf", created_in_run="run1",
        )
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=ExplainedCase(
                run_id="run1", artifact_ref=ArtifactRef("Q12", 1),
                branch_id="factual", decision_ref=ArtifactRef("decision:Q12", 1),
                outcome="reject",
            ),
            run_result=self._run(
                _CS(candidate_mcqs={"Q12": cand}, outcome_assessments={"Q12": assess}),
                {cenv.instance_id: cenv, aenv.instance_id: aenv},
            ),
        )
        self.assertFalse(obs.outcome_available)
        self.assertIn("stale assessment", obs.error or "")

    def test_missing_immutable_decision_envelope(self):
        cand = _Cand("Q12", 0)
        cenv = ArtifactEnvelope.create(
            logical_id="Q12", version=1, artifact_type="CandidateMCQ",
            payload=cand, branch_id="cf", created_in_run="run1",
        )
        dec = _Decision("Q12", "reject", ["r"], [], 1)
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=ExplainedCase(
                run_id="run1", artifact_ref=ArtifactRef("Q12", 1),
                branch_id="factual", decision_ref=ArtifactRef("decision:Q12", 1),
                outcome="reject",
            ),
            run_result=self._run(
                _CS(candidate_mcqs={"Q12": cand}, supervisor_decisions={"Q12": dec}),
                {cenv.instance_id: cenv},
            ),
        )
        self.assertFalse(obs.outcome_available)
        self.assertIn("decision envelope", obs.error or "")

    def test_missing_immutable_assessment_envelope(self):
        cand = _Cand("Q12", 0)
        cenv = ArtifactEnvelope.create(
            logical_id="Q12", version=1, artifact_type="CandidateMCQ",
            payload=cand, branch_id="cf", created_in_run="run1",
        )
        aref = ArtifactRef("assessment:Q12", 1)
        assess = _Assessment(aref, ArtifactRef("Q12", 1), ())
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=ExplainedCase(
                run_id="run1", artifact_ref=ArtifactRef("Q12", 1),
                branch_id="factual", decision_ref=ArtifactRef("decision:Q12", 1),
                outcome="reject",
            ),
            run_result=self._run(
                _CS(candidate_mcqs={"Q12": cand}, outcome_assessments={"Q12": assess}),
                {cenv.instance_id: cenv},
            ),
        )
        self.assertFalse(obs.outcome_available)
        self.assertIn("assessment envelope", obs.error or "")


class TestGenericIsolation(unittest.TestCase):
    def test_provenance_instrumentation_import_isolation(self):
        rag_dir = Path(__file__).resolve().parents[2]
        code = (
            "import sys\n"
            "import mas_langgraph.provenance.instrumentation as m\n"
            "assert 'mas_langgraph.cases.mcq' not in sys.modules\n"
            "assert 'mas_langgraph.domains.quiz' not in sys.modules\n"
        )
        subprocess.check_call([sys.executable, "-c", code], cwd=str(rag_dir))

    def test_state_cloning_import_isolation(self):
        rag_dir = Path(__file__).resolve().parents[2]
        code = (
            "import sys\n"
            "import mas_langgraph.core.state_cloning as m\n"
            "assert 'mas_langgraph.cases.mcq' not in sys.modules\n"
        )
        subprocess.check_call([sys.executable, "-c", code], cwd=str(rag_dir))

    def test_instrumentation_uses_case_adapter(self):
        from mas_langgraph.provenance.instrumentation import instrument_node
        from mas_langgraph.provenance.models import ArtifactBinding

        class FakeAdapter:
            def input_bindings_for_node(self, node_id, state, registry, **kwargs):
                return (ArtifactBinding(ArtifactRef("X", 1), "known"),)

            def snapshot_state_summary(self, state):
                return {"n": 1}

            def compact_state_delta(self, before, after, **kwargs):
                return {"used": "fake"}

        class FakeCase:
            def load_provenance_schema(self):
                from mas_langgraph.provenance.models import ProvenanceSchema, SchemaRelation

                return ProvenanceSchema(
                    relations=[
                        SchemaRelation("n", "ExampleSourceArtifact", "known", "n")
                    ]
                )

            def load_provenance_instrumentation_adapter(self):
                return FakeAdapter()

        class _RT:
            context = WorkflowRuntime(case_definition=FakeCase())

        def node_fn(state, runtime):
            return {}

        wrapped = instrument_node("n", node_fn)
        state = {
            "run_id": "r",
            "branch_id": "factual",
            "artifact_catalog": {},
            "occurrence_records": [],
            "produced_by": [],
            "run_provenance_relations": [],
        }
        update = wrapped(state, _RT())
        occs = update.get("occurrence_records") or []
        self.assertTrue(occs)
        self.assertEqual(occs[-1].input_bindings[0].relation, "known")
        self.assertEqual(occs[-1].state_delta.get("used"), "fake")

    def test_no_provenance_case_configured(self):
        from mas_langgraph.provenance.instrumentation import instrument_node

        class _RT:
            context = WorkflowRuntime(case_definition=None)

        wrapped = instrument_node("n", lambda s, r: {})
        with self.assertRaises(RuntimeError):
            wrapped({"run_id": "r", "branch_id": "factual"}, _RT())

    def test_workflow_executor_no_implicit_mcq(self):
        with self.assertRaises(CaseConfigurationError):
            WorkflowExecutor(runtime=None, compiled_graph=None)

    def test_workflow_executor_uses_case_graph(self):
        class FakeGraph:
            nodes = {"n": object()}

            def invoke(self, *a, **k):
                return {"ok": True}

        class FakeCase:
            case_id = "x"

            def build_graph(self, checkpointer=None):
                return FakeGraph()

            def load_intervention_contracts(self):
                return InterventionContractRegistry()

        rt = WorkflowRuntime(case_definition=FakeCase())
        ex = WorkflowExecutor(runtime=rt, compiled_graph=None)
        self.assertIsInstance(ex.compiled_graph, FakeGraph)

    def test_graph_builder_errors_propagate(self):
        class BadCase:
            def build_graph(self, checkpointer=None):
                raise RuntimeError("bad graph")

        with self.assertRaises(RuntimeError):
            WorkflowExecutor(
                runtime=WorkflowRuntime(case_definition=BadCase()),
                compiled_graph=None,
            )

    def test_case_contract_errors_propagate(self):
        class FakeGraph:
            def invoke(self, *a, **k):
                return {}

        class BadContracts:
            def build_graph(self, checkpointer=None):
                return FakeGraph()

            def load_intervention_contracts(self):
                raise CaseConfigurationError("bad contracts")

        ex = WorkflowExecutor(
            runtime=WorkflowRuntime(case_definition=BadContracts()),
            compiled_graph=FakeGraph(),
        )
        with self.assertRaises(CaseConfigurationError):
            ex.run({}, runtime=WorkflowRuntime(case_definition=BadContracts()), run_id="r")


class TestExampleCase(unittest.TestCase):
    def test_example_imports_without_mcq(self):
        rag_dir = Path(__file__).resolve().parents[2]
        code = (
            "import sys\n"
            "import mas_langgraph.cases.example_case as m\n"
            "assert 'mas_langgraph.cases.mcq' not in sys.modules\n"
            "assert 'mas_langgraph.domains.quiz' not in sys.modules\n"
        )
        subprocess.check_call([sys.executable, "-c", code], cwd=str(rag_dir))

    def test_example_declarative_files_load(self):
        case = ExampleCaseDefinition()
        cfg = case.validate_configuration(compiled_graph=None, require_graph=False)
        self.assertEqual(cfg.metadata.case_id, "example_case")
        self.assertTrue(cfg.workflow.nodes)
        self.assertIn("ExampleSourceArtifact", cfg.artifact_types)
        self.assertTrue(cfg.provenance_schema.relations)
        self.assertTrue(cfg.outcome_criteria)
        self.assertIsNotNone(cfg.intervention_contracts.get("ExampleSourceArtifact"))

    def test_example_metadata_has_no_mcq_terms(self):
        root = Path(__file__).resolve().parents[1] / "cases" / "example_case" / "inputs"
        blob = ""
        for p in root.rglob("*"):
            if p.is_file():
                blob += p.read_text(encoding="utf-8")
        for term in ("MCQ", "CandidateMCQ", "EvidenceChunk", "CrossCoVe", "quiz"):
            self.assertNotIn(term, blob)


class TestMcqProvenanceEquivalent(unittest.TestCase):
    def test_adapter_delegates_bindings(self):
        from mas_langgraph.cases.mcq.provenance import input_bindings_for_node
        from mas_langgraph.cases.mcq.provenance_instrumentation import (
            MCQProvenanceInstrumentationAdapter,
        )

        adapter = MCQProvenanceInstrumentationAdapter()
        self.assertEqual(adapter.input_bindings_for_node.__func__.__name__, "input_bindings_for_node")
        self.assertTrue(callable(input_bindings_for_node))


if __name__ == "__main__":
    unittest.main()
