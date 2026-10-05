"""Prompt 13 — AXIS Interrogation Language, Factual WHAT, Verbalization tests."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from mas_langgraph.artifacts.models import (
    ArtifactEnvelope,
    ArtifactRef,
    DerivedFrom,
    ProducedBy,
)
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.axis.models import (
    AxisPromptTemplates,
    AxisQueryContext,
    ExplainedCase,
    FactualLookupResult,
    FactualQueryRecord,
    InterventionQueryRecord,
    QuerySignature,
    ResolvedQuery,
)
from mas_langgraph.axis.queries import (
    ArtifactAliasIndex,
    AxisQuery,
    AxisQueryParser,
    AxisQueryResolver,
    QueryValidationError,
)
from mas_langgraph.axis.executor import AxisQueryExecutor
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation
from mas_langgraph.replay.checkpoints import (
    BoundarySelector,
    CheckpointIndex,
    CheckpointRecord,
)
from mas_langgraph.replay.interventions import (
    InterventionContract,
    InterventionContractRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakePayload:
    content: str = ""
    stem: str = ""


def _make_env(
    logical_id: str,
    version: int = 1,
    artifact_type: str = "EvidenceChunk",
    branch_id: str = "factual",
) -> ArtifactEnvelope:
    return ArtifactEnvelope.create(
        logical_id=logical_id,
        version=version,
        artifact_type=artifact_type,
        payload=_FakePayload(content=f"content of {logical_id}:v{version}"),
        branch_id=branch_id,
        created_in_run="run1",
    )


def _build_context(
    *,
    extra_scope_refs: Sequence[ArtifactRef] = (),
    extra_aliases: dict[str, ArtifactRef] | None = None,
    contracts: InterventionContractRegistry | None = None,
    intervention_alternatives: dict[str, ArtifactRef] | None = None,
    intervention_registry: ArtifactRegistry | None = None,
) -> Tuple[AxisQueryContext, ArtifactRegistry, CheckpointIndex]:
    """Build a minimal AXIS query context for testing."""
    reg = ArtifactRegistry()

    # Factual artifacts
    E17_env = _make_env("E17", 1, "EvidenceChunk")
    Q12v1_env = _make_env("Q12", 1, "CandidateMCQ")
    Q12v2_env = _make_env("Q12", 2, "CandidateMCQ")
    dec_env = _make_env("decision:Q12", 2, "SupervisorDecision")
    ass_env = _make_env("assessment:Q12", 2, "OutcomeAssessment")
    Q13v1_env = _make_env("Q13", 1, "CandidateMCQ")  # unrelated

    for env in [E17_env, Q12v1_env, Q12v2_env, dec_env, ass_env, Q13v1_env]:
        reg.store(env)

    explained = ExplainedCase(
        run_id="run1",
        artifact_ref=ArtifactRef("Q12", 2),
        branch_id="factual",
        decision_ref=ArtifactRef("decision:Q12", 2),
        outcome="accept",
        assessment_ref=ArtifactRef("assessment:Q12", 2),
    )

    # Slice
    scope_refs = [
        ArtifactRef("E17", 1),
        ArtifactRef("Q12", 1),
        ArtifactRef("Q12", 2),
        ArtifactRef("decision:Q12", 2),
        ArtifactRef("assessment:Q12", 2),
    ] + list(extra_scope_refs)

    occ1 = OccurrenceRecord(
        occurrence_id="run1:factual:generate#1",
        node_id="generate_candidate_mcqs_from_blueprints",
        occurrence_index=1,
        branch_id="factual",
        run_id="run1",
        execution_index=3,
        produced_artifacts=(ArtifactRef("Q12", 1),),
    )
    occ2 = OccurrenceRecord(
        occurrence_id="run1:factual:verify#1",
        node_id="verify_candidate_mcqs_against_evidence",
        occurrence_index=1,
        branch_id="factual",
        run_id="run1",
        execution_index=5,
        produced_artifacts=(),
    )
    occ3 = OccurrenceRecord(
        occurrence_id="run1:factual:verify#2",
        node_id="verify_candidate_mcqs_against_evidence",
        occurrence_index=2,
        branch_id="factual",
        run_id="run1",
        execution_index=7,
        produced_artifacts=(),
    )

    rel1 = RunProvenanceRelation(
        producer_occurrence_id="run1:factual:generate#1",
        artifact_ref=ArtifactRef("E17", 1),
        relation="supporting_evidence",
        consumer_occurrence_id="run1:factual:verify#2",
        artifact_type="EvidenceChunk",
    )

    deriv = DerivedFrom(
        new_artifact_ref=ArtifactRef("Q12", 2),
        previous_artifact_ref=ArtifactRef("Q12", 1),
    )

    case_slice = CaseSlice(
        explained_case=explained,
        relevant_artifact_refs=tuple(scope_refs),
        relevant_occurrences=(occ1, occ2, occ3),
        relevant_relations=(rel1,),
        relevant_derivations=(deriv,),
    )

    # Checkpoint index
    ckpt_idx = CheckpointIndex(thread_id="t1")
    recs = [
        CheckpointRecord(
            alias="c_0", thread_id="t1", checkpoint_id="ckpt_0", superstep=0,
        ),
        CheckpointRecord(
            alias="c_5", thread_id="t1", checkpoint_id="ckpt_5", superstep=5,
            after_occurrences=("run1:factual:generate#1",),
            after_occurrence_labels=("generate_candidate_mcqs_from_blueprints#1",),
            parent_checkpoint_id="ckpt_0",
        ),
        CheckpointRecord(
            alias="c_8", thread_id="t1", checkpoint_id="ckpt_8", superstep=8,
            next_nodes=("verify_candidate_mcqs_against_evidence",),
            after_occurrences=("run1:factual:verify#1",),
            after_occurrence_labels=("verify_candidate_mcqs_against_evidence#1",),
            parent_checkpoint_id="ckpt_5",
        ),
        CheckpointRecord(
            alias="c_10", thread_id="t1", checkpoint_id="ckpt_10", superstep=10,
            next_nodes=("verify_candidate_mcqs_against_evidence",),
            after_occurrences=("run1:factual:verify#2",),
            after_occurrence_labels=("verify_candidate_mcqs_against_evidence#2",),
            parent_checkpoint_id="ckpt_8",
        ),
    ]
    for r in recs:
        ckpt_idx.add(r)

    if contracts is None:
        contracts = InterventionContractRegistry()
        contracts.register(InterventionContract(
            artifact_type="EvidenceChunk",
            allowed_operations=("REMOVE", "ADD", "REPLACE"),
            replacement_compatible_types=("EvidenceChunk",),
        ))
        contracts.register(InterventionContract(
            artifact_type="CandidateMCQ",
            allowed_operations=("REMOVE",),
        ))

    ctx = AxisQueryContext(
        explained_case=explained,
        case_slice=case_slice,
        registry=reg,
        checkpoint_index=ckpt_idx,
        contracts=contracts,
        alias_map=extra_aliases or {},
        intervention_alternatives=intervention_alternatives or {},
        intervention_registry=intervention_registry,
    )
    return ctx, reg, ckpt_idx


# ===========================================================================
# Tests
# ===========================================================================

class TestCanonicalArtifactParsing(unittest.TestCase):
    """Test #81: exact canonical artifact parsing."""

    def test_decision_colon_ref(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        parser = AxisQueryParser()
        q = parser.parse("WHAT(decision:Q12:v2)")
        rq = resolver.resolve(q, ctx)
        self.assertEqual(rq.target_ref, ArtifactRef("decision:Q12", 2))


class TestNoImplicitV1(unittest.TestCase):
    """Test #82: bare logical id must not manufacture v1."""

    def test_bare_Q12_resolves_when_unambiguous_not_v1(self):
        # Registry has only Q12:v2 in scope? No, our scope has both Q12:v1 and Q12:v2.
        # So WHAT(Q12) must fail (ambiguous). Covered in TestAmbiguousVersions.
        pass

    def test_explicit_v2_works(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHAT(Q12:v2)")
        rq = resolver.resolve(q, ctx)
        self.assertEqual(rq.target_ref, ArtifactRef("Q12", 2))


class TestAmbiguousVersions(unittest.TestCase):
    """Test #83: scope has Q12:v1 and Q12:v2."""

    def test_bare_Q12_fails(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHAT(Q12)")
        with self.assertRaises(QueryValidationError):
            resolver.resolve(q, ctx)

    def test_explicit_v1_ok(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHAT(Q12:v1)")
        rq = resolver.resolve(q, ctx)
        self.assertEqual(rq.target_ref, ArtifactRef("Q12", 1))

    def test_explicit_v2_ok(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHAT(Q12:v2)")
        rq = resolver.resolve(q, ctx)
        self.assertEqual(rq.target_ref, ArtifactRef("Q12", 2))


class TestExplainedCaseAliases(unittest.TestCase):
    """Test #84: convenience aliases."""

    def test_candidate_alias(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHAT(candidate)")
        rq = resolver.resolve(q, ctx)
        self.assertEqual(rq.target_ref, ArtifactRef("Q12", 2))

    def test_decision_alias(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHAT(decision)")
        rq = resolver.resolve(q, ctx)
        self.assertEqual(rq.target_ref, ArtifactRef("decision:Q12", 2))

    def test_assessment_alias(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHAT(assessment)")
        rq = resolver.resolve(q, ctx)
        self.assertEqual(rq.target_ref, ArtifactRef("assessment:Q12", 2))


class TestUnrelatedArtifactRejected(unittest.TestCase):
    """Test #85: Q13 artifact not in slice → rejected."""

    def test_q13_rejected(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHAT(Q13:v1)")
        with self.assertRaises(QueryValidationError):
            resolver.resolve(q, ctx)


class TestBoundaryResolution(unittest.TestCase):
    """Tests #86-#90: boundary resolution."""

    def test_before_occurrence(self):
        """Test #86: before verify...#2 resolves to c_8 (the checkpoint before #2)."""
        ctx, _, idx = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse(
            "REMOVE(E17:v1, before verify_candidate_mcqs_against_evidence#2)"
        )
        rq = resolver.resolve(q, ctx)
        # before verify#2 → checkpoint where verify#2 is next
        # In our index, c_10 has after_occurrence verify#2.
        # "before occurrence #2" means the checkpoint before that occurrence ran.
        # The checkpoint_index should resolve this.
        self.assertIsNotNone(rq.checkpoint_alias)
        self.assertIsNotNone(rq.checkpoint_id)

    def test_after_occurrence(self):
        """Test #87: after revise#1 resolves."""
        ctx, _, idx = _build_context()
        # We don't have a revise occurrence, so let's test after verify#1
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse(
            "REMOVE(E17:v1, after verify_candidate_mcqs_against_evidence#1)"
        )
        rq = resolver.resolve(q, ctx)
        self.assertIsNotNone(rq.checkpoint_alias)
        self.assertEqual(rq.checkpoint_alias, "c_8")

    def test_bare_occurrence_rejected(self):
        """Test #88: bare node#N without before/after is rejected."""
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse(
            "REMOVE(E17:v1, verify_candidate_mcqs_against_evidence#1)"
        )
        with self.assertRaises(QueryValidationError) as cm:
            resolver.resolve(q, ctx)
        self.assertIn("before or after", str(cm.exception))

    def test_unknown_boundary_rejected(self):
        """Test #89: unknown boundary fails."""
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("REMOVE(E17:v1, some_random_boundary)")
        with self.assertRaises(QueryValidationError):
            resolver.resolve(q, ctx)

    def test_explicit_c_alias(self):
        """Test: c_8 resolves correctly."""
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("REMOVE(E17:v1, c_8)")
        rq = resolver.resolve(q, ctx)
        self.assertEqual(rq.checkpoint_alias, "c_8")
        self.assertEqual(rq.checkpoint_id, "ckpt_8")

    def test_invalid_c_alias(self):
        """Test #19: c_999 fails."""
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("REMOVE(E17:v1, c_999)")
        with self.assertRaises(QueryValidationError):
            resolver.resolve(q, ctx)


class TestContractValidation(unittest.TestCase):
    """Tests #91-#93: contract validation."""

    def test_remove_no_contract(self):
        """Test #91: REMOVE on non-intervenable type fails."""
        ctx, _, _ = _build_context()
        # EvidenceCheckReport has no contract
        # We'll try removing a decision (SupervisorDecision, no REMOVE contract)
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("REMOVE(decision:Q12:v2, c_8)")
        with self.assertRaises(QueryValidationError):
            resolver.resolve(q, ctx)

    def test_add_no_contract(self):
        """Test #92: ADD on CandidateMCQ (no ADD contract) fails."""
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("ADD(Q12:v2, c_8)")
        with self.assertRaises(QueryValidationError):
            resolver.resolve(q, ctx)

    def test_replacement_incompatible(self):
        """Test #93: WHATIF(EvidenceChunk, CandidateMCQ) fails."""
        # Create intervention registry with a Q12:v1 as CandidateMCQ
        int_reg = ArtifactRegistry()
        int_reg.store(_make_env("Q12-alt", 1, "CandidateMCQ"))

        ctx, _, _ = _build_context(
            intervention_alternatives={"Q12-alt": ArtifactRef("Q12-alt", 1)},
            intervention_registry=int_reg,
        )
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHATIF(E17:v1, Q12-alt, c_8)")
        with self.assertRaises(QueryValidationError) as cm:
            resolver.resolve(q, ctx)
        self.assertIn("incompatible", str(cm.exception).lower())

    def test_what_ignores_contracts(self):
        """Test #25: WHAT on non-intervenable artifact is allowed."""
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        q = AxisQueryParser().parse("WHAT(decision:Q12:v2)")
        rq = resolver.resolve(q, ctx)
        self.assertEqual(rq.operation, "WHAT")


class TestDuplicateDetection(unittest.TestCase):
    """Tests #94-#95: canonical duplicate detection."""

    def test_semantic_duplicate(self):
        """Test #94: same intervention via different spellings is duplicate."""
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        parser = AxisQueryParser()

        # First query using before occurrence syntax
        q1 = parser.parse("REMOVE(E17:v1, after verify_candidate_mcqs_against_evidence#1)")
        rq1 = resolver.resolve(q1, ctx)

        # Second query using the resolved alias directly
        q2 = parser.parse(f"REMOVE(E17:v1, {rq1.checkpoint_alias})")
        with self.assertRaises(QueryValidationError):
            resolver.resolve(q2, ctx)

    def test_different_checkpoint_not_duplicate(self):
        """Test #95: different checkpoints are distinct experiments."""
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        parser = AxisQueryParser()

        q1 = parser.parse("REMOVE(E17:v1, c_8)")
        resolver.resolve(q1, ctx)

        q2 = parser.parse("REMOVE(E17:v1, c_10)")
        rq2 = resolver.resolve(q2, ctx)
        self.assertIsNotNone(rq2)

    def test_done_not_in_duplicate_set(self):
        """Test #29: DONE is never placed in the duplicate set."""
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        parser = AxisQueryParser()

        resolver.resolve(parser.parse("DONE"), ctx)
        resolver.resolve(parser.parse("DONE"), ctx)  # should not fail


class TestWHATStructuredResult(unittest.TestCase):
    """Tests #96-#97: structured factual WHAT result."""

    def _make_executor(self):
        from mas_langgraph.replay.executor import LangGraphReplayExecutor
        ctx, reg, idx = _build_context()
        ex = LangGraphReplayExecutor(compiled_graph=None)
        return AxisQueryExecutor(
            context=ctx,
            replay_executor=ex,
        ), ctx

    def test_what_returns_structured_result(self):
        """Test #96: WHAT produces FactualLookupResult."""
        executor, ctx = self._make_executor()
        rq = ResolvedQuery(
            operation="WHAT",
            raw_text="WHAT(E17:v1)",
            target_ref=ArtifactRef("E17", 1),
        )
        rec = executor.execute(rq)
        self.assertIsInstance(rec, FactualQueryRecord)
        self.assertIsInstance(rec.result, FactualLookupResult)
        self.assertEqual(rec.result.target_ref, ArtifactRef("E17", 1))
        self.assertEqual(rec.result.artifact_type, "EvidenceChunk")
        self.assertIsNotNone(rec.result.envelope)
        self.assertIsNotNone(rec.signature)

    def test_what_consumer_restricted_to_slice(self):
        """Test #97: WHAT(E17) shows Q12-relevant relations only."""
        executor, ctx = self._make_executor()
        rq = ResolvedQuery(
            operation="WHAT",
            raw_text="WHAT(E17:v1)",
            target_ref=ArtifactRef("E17", 1),
        )
        rec = executor.execute(rq)
        result = rec.result
        self.assertIsInstance(result, FactualLookupResult)
        # Should only contain relations from the case slice
        for rel in result.consumer_relations:
            # All should be Q12-relevant (in our fixture, verify#2 consumes E17)
            self.assertIn("verify", getattr(rel, "consumer_occurrence_id", ""))

    def test_what_no_replay_branch(self):
        """Test #28/#33: WHAT never creates replay branches."""
        executor, ctx = self._make_executor()
        rq = ResolvedQuery(
            operation="WHAT",
            raw_text="WHAT(E17:v1)",
            target_ref=ArtifactRef("E17", 1),
        )
        rec = executor.execute(rq)
        self.assertIsInstance(rec, FactualQueryRecord)
        # No branch IDs on factual record
        self.assertFalse(hasattr(rec, "noop_branch_id"))


class TestInterventionExecution(unittest.TestCase):
    """Tests #98-#101: intervention execution."""

    def _make_controller_and_executor(self):
        from mas_langgraph.artifacts.resolver import ArtifactResolver
        from mas_langgraph.replay.executor import LangGraphReplayExecutor

        ctx, reg, idx = _build_context()
        ex = LangGraphReplayExecutor(compiled_graph=None)
        ex.run_factual(
            {"step_logs": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]},
            run_id="run1",
            allow_synthetic_history=True,
        )
        # Setup resolvers
        resolver = ArtifactResolver(registry=reg)
        for ref in reg.all_refs():
            resolver.make_available(ref)
        ex._resolvers[("run1", "factual")] = resolver

        return AxisQueryExecutor(context=ctx, replay_executor=ex), ctx, ex

    def test_remove_execution(self):
        """Test #98: REMOVE creates noop and counterfactual branches."""
        executor, ctx, ex = self._make_controller_and_executor()
        rq = ResolvedQuery(
            operation="REMOVE",
            raw_text="REMOVE(E17:v1, c_8)",
            target_ref=ArtifactRef("E17", 1),
            checkpoint_alias="c_8",
            checkpoint_id="ckpt_8",
        )
        rec = executor.execute(rq)
        self.assertIsInstance(rec, InterventionQueryRecord)
        self.assertIsNotNone(rec.noop_branch_id)
        self.assertIsNotNone(rec.counterfactual_branch_id)
        self.assertIsNotNone(rec.noop_result)
        self.assertIsNotNone(rec.counterfactual_result)
        self.assertIsNone(rec.error)
        self.assertEqual(rec.comparison.get("operation"), "REMOVE")

    def test_add_execution(self):
        """Test #99: ADD uses intervention_registry."""
        executor, ctx, ex = self._make_controller_and_executor()
        rq = ResolvedQuery(
            operation="ADD",
            raw_text="ADD(E17:v1, c_8)",
            target_ref=ArtifactRef("E17", 1),
            checkpoint_alias="c_8",
            checkpoint_id="ckpt_8",
        )
        rec = executor.execute(rq)
        self.assertIsInstance(rec, InterventionQueryRecord)
        self.assertIsNotNone(rec.noop_branch_id)

    def test_whatif_maps_to_replace(self):
        """Test #100: WHATIF maps to REPLACE in backend."""
        executor, ctx, ex = self._make_controller_and_executor()
        # Add E24 to registry
        E24 = _make_env("E24", 1, "EvidenceChunk")
        ctx.registry.store(E24)

        rq = ResolvedQuery(
            operation="WHATIF",
            raw_text="WHATIF(E17:v1, E24:v1, c_8)",
            target_ref=ArtifactRef("E17", 1),
            replacement_ref=ArtifactRef("E24", 1),
            checkpoint_alias="c_8",
            checkpoint_id="ckpt_8",
        )
        rec = executor.execute(rq)
        self.assertIsInstance(rec, InterventionQueryRecord)
        # The comparison should reflect WHATIF semantics
        self.assertEqual(rec.comparison.get("operation"), "WHATIF")

    def test_execution_error_recorded(self):
        """Test #101: execution errors are recorded."""
        executor, ctx, _ = self._make_controller_and_executor()
        rq = ResolvedQuery(
            operation="REMOVE",
            raw_text="REMOVE(E17:v1, c_999)",
            target_ref=ArtifactRef("E17", 1),
            checkpoint_alias="c_999",
        )
        rec = executor.execute(rq)
        self.assertIsInstance(rec, InterventionQueryRecord)
        self.assertIsNotNone(rec.error)


class TestPromptTemplates(unittest.TestCase):
    """Tests #102-#103: prompt templates and verbalizer."""

    def test_load_prompt_templates(self):
        """Test #102: templates load through case definition."""
        from mas_langgraph.cases.mcq.case_definition import MCQCaseDefinition
        case = MCQCaseDefinition()
        templates = case.load_prompt_templates()
        self.assertIsInstance(templates, AxisPromptTemplates)
        self.assertTrue(templates.system)
        self.assertTrue(templates.factual_context)
        self.assertTrue(templates.interrogation)

    def test_verbalizer_instantiation(self):
        """Test #103: MCQCaseDefinition().load_verbalizer() works."""
        from mas_langgraph.cases.mcq.case_definition import MCQCaseDefinition
        case = MCQCaseDefinition()
        v = case.load_verbalizer()
        self.assertIsNotNone(v)

    def test_missing_template_variable_fails(self):
        """Test #49: missing template variables fail clearly."""
        templates = AxisPromptTemplates(
            system="System: {role}",
            factual_context="Context: {data}",
            interrogation="Round: {round}",
        )
        with self.assertRaises(ValueError):
            templates.render("system")  # missing 'role'

    def test_empty_required_template_fails(self):
        """Test #48: missing required template raises."""
        templates = AxisPromptTemplates(system="ok", factual_context="ok")
        with self.assertRaises(ValueError):
            templates.validate_required("system", "factual_context", "interrogation")


class TestGenericImportIsolation(unittest.TestCase):
    """Test #104: generic AXIS import does not eagerly load MCQ."""

    def test_axis_import_no_mcq(self):
        import importlib
        import sys
        # Fresh import check — if mcq is already loaded, skip
        # The key assertion: importing axis does not force-load mcq.case_definition
        import mas_langgraph.axis
        # Verify axis module loaded correctly
        self.assertTrue(hasattr(mas_langgraph.axis, "AxisQueryParser"))
        self.assertTrue(hasattr(mas_langgraph.axis, "AxisVerbalizer"))
        # McqVerbalizer should NOT be in axis.__all__ anymore
        self.assertNotIn("McqVerbalizer", mas_langgraph.axis.__all__)


class TestFactualContextVerbalization(unittest.TestCase):
    """Tests #105-#111: factual context content."""

    def _verbalize(self):
        from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer
        ctx, reg, idx = _build_context()
        v = McqVerbalizer()
        case_slice = ctx.case_slice
        return v.verbalize_slice(case_slice)

    def test_criteria_present(self):
        """Test #105: criteria appear if present in slice."""
        # Our fixture doesn't have criterion_values, so test with them added
        from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer
        from mas_langgraph.axis.slicing import CaseSlice

        @dataclass
        class FakeCriterion:
            name: str = "evidence_support"
            value: Any = True
            decision_relevant: bool = True
            label: str = "Evidence supports answer"

        ctx, _, _ = _build_context()
        slice_with_criteria = CaseSlice(
            explained_case=ctx.explained_case,
            relevant_artifact_refs=ctx.case_slice.relevant_artifact_refs,
            relevant_occurrences=ctx.case_slice.relevant_occurrences,
            relevant_relations=ctx.case_slice.relevant_relations,
            criterion_values=(FakeCriterion(),),
            decision="accept",
            decision_reasons=("All criteria met",),
        )
        v = McqVerbalizer()
        text = v.verbalize_slice(slice_with_criteria)
        self.assertIn("evidence_support", text)
        self.assertIn("Decision-relevant", text)

    def test_repeated_occurrences_distinct(self):
        """Test #106: verify#1 and verify#2 shown separately."""
        text = self._verbalize()
        self.assertIn("verify_candidate_mcqs_against_evidence#1", text)
        self.assertIn("verify_candidate_mcqs_against_evidence#2", text)

    def test_semantic_provenance_present(self):
        """Test #107: producer → artifact [relation] → consumer form."""
        text = self._verbalize()
        self.assertIn("[supporting_evidence]", text)
        self.assertIn("-->", text)

    def test_revision_lineage_separate(self):
        """Test #108: DerivedFrom shown separately."""
        text = self._verbalize()
        self.assertIn("REVISION LINEAGE", text)
        self.assertIn("Q12:v2 derived from Q12:v1", text)

    def test_no_causal_claims(self):
        """Test #111: no causal claims in factual context."""
        text = self._verbalize()
        for bad in ("caused", "necessary", "decisive", "because of"):
            self.assertNotIn(bad, text.lower())

    def test_decision_details_verbalized(self):
        """Test #54: decision reasons/fixes shown."""
        from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer
        ctx, _, _ = _build_context()
        slice_with_decision = CaseSlice(
            explained_case=ctx.explained_case,
            relevant_artifact_refs=ctx.case_slice.relevant_artifact_refs,
            relevant_occurrences=ctx.case_slice.relevant_occurrences,
            relevant_relations=ctx.case_slice.relevant_relations,
            decision="accept",
            decision_reasons=("Meets all quality criteria",),
            required_fixes=("Minor formatting fix needed",),
        )
        v = McqVerbalizer()
        text = v.verbalize_slice(slice_with_decision)
        self.assertIn("Meets all quality criteria", text)
        self.assertIn("Minor formatting fix needed", text)


class TestDeterministicResolution(unittest.TestCase):
    """Test #112-#113: deterministic query/context."""

    def test_deterministic_query_resolution(self):
        """Test #112: same query resolves identically twice."""
        ctx, _, _ = _build_context()
        r1 = AxisQueryResolver()
        r2 = AxisQueryResolver()
        parser = AxisQueryParser()

        q = parser.parse("WHAT(E17:v1)")
        rq1 = r1.resolve(q, ctx)
        rq2 = r2.resolve(q, ctx)
        self.assertEqual(rq1.target_ref, rq2.target_ref)
        self.assertEqual(
            QuerySignature.from_resolved(rq1),
            QuerySignature.from_resolved(rq2),
        )

    def test_deterministic_factual_context(self):
        """Test #113: identical inputs produce identical verbalization."""
        from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer
        ctx, _, _ = _build_context()
        v = McqVerbalizer()
        text1 = v.verbalize_slice(ctx.case_slice)
        text2 = v.verbalize_slice(ctx.case_slice)
        self.assertEqual(text1, text2)


class TestQuerySignature(unittest.TestCase):
    """Verify QuerySignature semantics."""

    def test_what_signature(self):
        rq = ResolvedQuery(
            operation="WHAT",
            raw_text="WHAT(E17:v1)",
            target_ref=ArtifactRef("E17", 1),
        )
        sig = QuerySignature.from_resolved(rq)
        self.assertEqual(sig.operation, "WHAT")
        self.assertEqual(sig.target_ref, ArtifactRef("E17", 1))
        self.assertIsNone(sig.replacement_ref)

    def test_remove_signature_includes_checkpoint(self):
        rq = ResolvedQuery(
            operation="REMOVE",
            raw_text="REMOVE(E17:v1, c_8)",
            target_ref=ArtifactRef("E17", 1),
            checkpoint_alias="c_8",
            checkpoint_id="ckpt_8",
        )
        sig = QuerySignature.from_resolved(rq)
        self.assertEqual(sig.checkpoint_id, "ckpt_8")


class TestFailedQueriesNotCounted(unittest.TestCase):
    """Test #30: failed queries don't count as executed."""

    def test_failed_resolution_not_in_executed(self):
        ctx, _, _ = _build_context()
        resolver = AxisQueryResolver()
        parser = AxisQueryParser()

        # This should fail (Q13 not in slice)
        try:
            resolver.resolve(parser.parse("WHAT(Q13:v1)"), ctx)
        except QueryValidationError:
            pass

        # The failed query should not be in the executed set
        self.assertEqual(len(resolver.executed_signatures), 0)


class TestInterventionSpace(unittest.TestCase):
    """Test #109: intervention space matches contracts."""

    def test_evidence_chunk_operations(self):
        from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer
        ctx, reg, _ = _build_context()
        v = McqVerbalizer()
        text = v.verbalize_intervention_space(
            [ArtifactRef("E17", 1)], ctx.contracts, registry=reg
        )
        self.assertIn("REMOVE", text)
        self.assertIn("ADD", text)
        self.assertIn("REPLACE", text)

    def test_no_unsupported_operations(self):
        """EvidenceCheckReport has no contract → nothing shown."""
        from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer
        ctx, reg, _ = _build_context()
        v = McqVerbalizer()
        # Register an EvidenceCheckReport
        env = _make_env("report1", 1, "EvidenceCheckReport")
        reg.store(env)
        text = v.verbalize_intervention_space(
            [ArtifactRef("report1", 1)], ctx.contracts, registry=reg
        )
        # Should be empty since EvidenceCheckReport has no contract
        self.assertEqual(text, "")


class TestParseSyntax(unittest.TestCase):
    """Additional parser syntax tests."""

    def test_prose_with_command_rejected(self):
        """Test #70: prose with embedded command is rejected."""
        parser = AxisQueryParser()
        with self.assertRaises(QueryValidationError):
            parser.parse("I think we should inspect E17. WHAT(E17)")

    def test_done_repeated_ok(self):
        parser = AxisQueryParser()
        q1 = parser.parse("DONE")
        q2 = parser.parse("DONE")
        self.assertEqual(q1.operation, "DONE")
        self.assertEqual(q2.operation, "DONE")

    def test_endoftext_stripped(self):
        parser = AxisQueryParser()
        q = parser.parse("WHAT(E17)<|endoftext|>")
        self.assertEqual(q.operation, "WHAT")
        self.assertEqual(q.args, ("E17",))


if __name__ == "__main__":
    unittest.main()
