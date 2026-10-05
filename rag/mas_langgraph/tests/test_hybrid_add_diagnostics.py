"""Controlled tests for hybrid ADD vocabulary filtering and opt-in diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.hybrid_add_diagnostics import (
    ADD_PRESENCE_PRESENT,
    ADD_PRESENCE_UNKNOWN,
    HybridAddDiagnostics,
    extract_hybrid_add_diagnostics,
    inspect_saved_experiment_hybrid_add_replay,
)
from mas_langgraph.axis.intervention_guidance import (
    build_hybrid_intervention_vocabulary,
    format_hybrid_vocabulary,
    hybrid_opportunity_counts,
)
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.evaluation.report import write_metrics_outputs
from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation
from mas_langgraph.tests.test_axis_hybrid_interrogation import (
    _availability_fixture,
    _hybrid_vocab,
    _store_meta,
)

SAVED_141 = (
    Path(__file__).resolve().parents[2]
    / "lecturebank_axis_eval_141_concept_check"
    / "streamlit"
    / "quiz_explanation_experiment_141.json"
)


def _build_vocab(
    explained,
    reg,
    occs,
    idx,
    contracts,
    case_slice,
    ex,
    *,
    diagnostics=None,
    max_alternatives_per_target=8,
    intervention_alternatives=None,
    intervention_registry=None,
    replay_executor="default",
):
    executor = ex if replay_executor == "default" else replay_executor
    return build_hybrid_intervention_vocabulary(
        case_slice=case_slice,
        registry=reg,
        contracts=contracts,
        checkpoint_index=idx,
        branch_id="factual",
        explained_case=explained,
        intervention_alternatives=intervention_alternatives or {},
        intervention_registry=intervention_registry,
        all_occurrences=occs,
        replay_executor=executor,
        run_id=getattr(explained, "run_id", "") or "run1",
        max_alternatives_per_target=max_alternatives_per_target,
        diagnostics=diagnostics,
    )


class TestHybridAddPositiveOpportunity(unittest.TestCase):
    def test_sourceable_out_of_slice_chunk_absent_at_boundary_is_advertised(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        self.assertIn(ArtifactRef("chunk_out", 1), vocab.add_options)
        bounds = vocab.add_boundaries_for(ArtifactRef("chunk_out", 1))
        self.assertTrue(bounds)
        counts = hybrid_opportunity_counts(vocab)
        self.assertEqual(
            counts["add"],
            sum(len(vocab.add_boundaries_for(ref)) for ref in vocab.add_options),
        )
        self.assertGreater(counts["add"], 0)


class TestHybridAddNegativeCases(unittest.TestCase):
    def test_already_in_relevant_slice(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        vocab = _build_vocab(
            explained, reg, occs, idx, contracts, case_slice, ex, diagnostics=diag
        )
        self.assertNotIn(ArtifactRef("chunk_1", 1), vocab.add_options)
        row = next(
            r
            for r in diag.candidate_rows
            if r["artifact_ref"] == "chunk_1:v1"
            and r["candidate_source"] == "factual_registry"
        )
        self.assertEqual(row["eligibility_result"], "already_in_relevant_slice")
        self.assertEqual(row["in_relevant_slice"], "true")

    def test_contract_disallows_add_for_knowledge_atom(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        vocab = _build_vocab(
            explained, reg, occs, idx, contracts, case_slice, ex, diagnostics=diag
        )
        self.assertNotIn(ArtifactRef("atom_out", 1), vocab.add_options)
        row = next(
            r
            for r in diag.candidate_rows
            if r["artifact_ref"] == "atom_out:v1"
        )
        self.assertEqual(row["eligibility_result"], "contract_disallows_add")
        self.assertEqual(row["contract_permits_add"], "false")

    def test_payload_unavailable_superseded_explicit_alternative(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        hist = ArtifactRef("chunk_hist", 1)
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        vocab = _build_vocab(
            explained,
            reg,
            occs,
            idx,
            contracts,
            case_slice,
            ex,
            diagnostics=diag,
            intervention_alternatives={"hist": hist},
        )
        self.assertNotIn(hist, vocab.add_options)
        row = next(r for r in diag.candidate_rows if r["artifact_ref"] == "chunk_hist:v1")
        self.assertEqual(row["eligibility_result"], "payload_unavailable")
        self.assertEqual(row["explicit_exception_applied"], "true")

    def test_contextually_unrelated_automatic_candidate(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        vocab = _build_vocab(
            explained, reg, occs, idx, contracts, case_slice, ex, diagnostics=diag
        )
        self.assertNotIn(ArtifactRef("chunk_other", 1), vocab.add_options)
        row = next(
            r
            for r in diag.candidate_rows
            if r["artifact_ref"] == "chunk_other:v1"
            and r["candidate_source"] == "factual_registry"
        )
        self.assertEqual(row["eligibility_result"], "contextually_unrelated")
        self.assertEqual(row["explicit_exception_applied"], "false")

    def test_explicit_alternative_skips_automatic_contextual_filter(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        other = ArtifactRef("chunk_other", 1)
        vocab = _build_vocab(
            explained,
            reg,
            occs,
            idx,
            contracts,
            case_slice,
            ex,
            intervention_alternatives={"other": other},
        )
        self.assertIn(other, vocab.add_options)
        auto = _build_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        self.assertNotIn(other, auto.add_options)

    def test_no_suitable_consumer_boundary(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        empty_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=case_slice.relevant_artifact_refs,
            relevant_occurrences=case_slice.relevant_occurrences,
            relevant_relations=tuple(
                rel
                for rel in case_slice.relevant_relations
                if getattr(rel, "artifact_ref", None)
                not in {
                    ArtifactRef("chunk_1", 1),
                    ArtifactRef("chunk_2", 1),
                    ArtifactRef("E17", 1),
                }
            ),
        )
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        vocab = _build_vocab(
            explained, reg, occs, idx, contracts, empty_slice, ex, diagnostics=diag
        )
        self.assertNotIn(ArtifactRef("chunk_out", 1), vocab.add_options)
        row = next(
            r
            for r in diag.candidate_rows
            if r["artifact_ref"] == "chunk_out:v1"
            and r["eligibility_result"] == "reached_boundary_checks"
        )
        self.assertEqual(row["final_status"], "no_suitable_consumer_boundary")

    def test_unresolvable_checkpoint(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        missing_occ = OccurrenceRecord(
            occurrence_id="run1:factual:ghost_node#1",
            node_id="ghost_node",
            occurrence_index=1,
            branch_id="factual",
            run_id="run1",
            execution_index=99,
        )
        occs = list(occs) + [missing_occ]
        ghost_rel = RunProvenanceRelation(
            producer_occurrence_id="run1:factual:gen#1",
            artifact_ref=ArtifactRef("chunk_1", 1),
            relation="supporting_evidence",
            consumer_occurrence_id="run1:factual:ghost_node#1",
        )
        ghost_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=case_slice.relevant_artifact_refs,
            relevant_occurrences=tuple(occs),
            relevant_relations=(ghost_rel,),
        )
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        vocab = _build_vocab(
            explained, reg, occs, idx, contracts, ghost_slice, ex, diagnostics=diag
        )
        self.assertEqual(vocab.add_options, ())
        unresolved = [
            r
            for r in diag.boundary_rows
            if r["presence_detail"] == "checkpoint_unresolvable"
        ]
        self.assertTrue(unresolved)
        self.assertTrue(all(r["presence"] == ADD_PRESENCE_UNKNOWN for r in unresolved))
        self.assertTrue(all(r["checkpoint_resolved"] == "false" for r in unresolved))

    def test_confirmed_present_at_checkpoint(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        existing = ArtifactRef("chunk_out", 1)
        existing_env = reg.get_by_ref(existing, "factual")
        catalog = {}
        for ref in case_slice.relevant_artifact_refs:
            env = reg.get_by_ref(ref, "factual") or reg.get_by_ref(ref)
            if env is not None:
                catalog[env.instance_id] = env
        catalog[existing_env.instance_id] = existing_env
        from mas_langgraph.tests.test_axis_hybrid_interrogation import (
            _stamp_snapshot_catalogs,
        )

        _stamp_snapshot_catalogs(ex, "run1", catalog)
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        vocab = _build_vocab(
            explained, reg, occs, idx, contracts, case_slice, ex, diagnostics=diag
        )
        self.assertNotIn(existing, vocab.add_options)
        present = [
            r
            for r in diag.boundary_rows
            if r["artifact_ref"] == "chunk_out:v1"
            and r["presence"] == ADD_PRESENCE_PRESENT
        ]
        self.assertTrue(present)
        self.assertTrue(all(r["inclusion_reason"] == "excluded_present" for r in present))

    def test_checkpoint_state_unavailable(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        vocab = _build_vocab(
            explained,
            reg,
            occs,
            idx,
            contracts,
            case_slice,
            ex,
            diagnostics=diag,
            replay_executor=None,
        )
        self.assertEqual(vocab.add_options, ())
        unknown = [
            r
            for r in diag.boundary_rows
            if r["artifact_ref"] == "chunk_out:v1"
        ]
        self.assertTrue(unknown)
        self.assertTrue(all(r["presence"] == ADD_PRESENCE_UNKNOWN for r in unknown))
        self.assertTrue(
            all(r["presence_detail"] == "state_unavailable" for r in unknown)
        )
        self.assertTrue(all(r["advertised"] == "false" for r in unknown))

    def test_eligible_candidate_excluded_by_cap(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        _store_meta(
            reg,
            "chunk_out2",
            atype="EvidenceChunk",
            content="second outside-slice chunk same section",
            section_id="sec_a",
            tags=("kmeans",),
        )
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        vocab = _build_vocab(
            explained,
            reg,
            occs,
            idx,
            contracts,
            case_slice,
            ex,
            diagnostics=diag,
            max_alternatives_per_target=1,
        )
        self.assertEqual(len(vocab.add_options), 1)
        omitted = [r for r in diag.candidate_rows if r["omitted_by_cap"] == "true"]
        self.assertTrue(omitted)
        self.assertTrue(all(r["final_status"] == "omitted_by_cap" for r in omitted))
        advertised = [r for r in diag.candidate_rows if r["advertised"] == "true"]
        self.assertEqual(len({r["artifact_ref"] for r in advertised}), 1)


class TestDiagnosticsDoNotChangeVocabulary(unittest.TestCase):
    def test_diagnose_on_off_identical_vocabulary_and_prompt_text(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        off = _build_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        diag = HybridAddDiagnostics(run_id="run1", explained_ref="Q12:v2")
        on = _build_vocab(
            explained, reg, occs, idx, contracts, case_slice, ex, diagnostics=diag
        )
        self.assertEqual(off.add_options, on.add_options)
        self.assertEqual(off.add_boundaries, on.add_boundaries)
        self.assertEqual(off.add_boundaries_by_ref, on.add_boundaries_by_ref)
        self.assertEqual(off.replacement_options, on.replacement_options)
        self.assertEqual(
            format_hybrid_vocabulary(vocabulary=off, registry=reg, branch_id="factual"),
            format_hybrid_vocabulary(vocabulary=on, registry=reg, branch_id="factual"),
        )
        self.assertTrue(diag.candidate_rows)
        advertised = [r for r in diag.boundary_rows if r["advertised"] == "true"]
        self.assertEqual(
            hybrid_opportunity_counts(on)["add"],
            len(advertised),
        )
        self.assertEqual(
            diag.summary()["executable_add_opportunities"],
            hybrid_opportunity_counts(on)["add"],
        )


class TestDiagnosticExport(unittest.TestCase):
    def test_extract_and_write_csvs(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        diag = HybridAddDiagnostics(
            run_id="run1", explained_ref="q_00000001:v1", lecturebank_id="141"
        )
        _build_vocab(
            explained, reg, occs, idx, contracts, case_slice, ex, diagnostics=diag
        )
        payload = {
            "run_id": "run1",
            "cases": [
                {
                    "candidate_ref": "q_00000001:v1",
                    "run_id": "run1",
                    "conditions": {
                        "axis": {"raw_evidence": {"hybrid_add_diagnostics": diag.to_payload()}}
                    },
                }
            ],
        }
        extracted = extract_hybrid_add_diagnostics(payload, lecturebank_id="141")
        self.assertTrue(extracted["hybrid_add_candidates"])
        self.assertTrue(extracted["hybrid_add_boundaries"])
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics"
            write_metrics_outputs(
                metrics,
                document_rows=[],
                attempt_rows=[],
                comparison_rows=[],
                provenance_rows=[],
                distribution_rows=[],
                round_rows=[],
                overall_rows=[],
                report_markdown="# test\n",
                hybrid_add_candidate_rows=extracted["hybrid_add_candidates"],
                hybrid_add_boundary_rows=extracted["hybrid_add_boundaries"],
            )
            cand_path = metrics / "hybrid_add_candidates.csv"
            bound_path = metrics / "hybrid_add_boundaries.csv"
            self.assertTrue(cand_path.is_file())
            self.assertTrue(bound_path.is_file())
            cand_text = cand_path.read_text(encoding="utf-8")
            self.assertIn("chunk_out:v1", cand_text)
            self.assertNotIn("Unsupervised Learning", cand_text)


class TestSavedRunReplayability(unittest.TestCase):
    def test_concept_check_run_lacks_checkpoint_state_for_builder_replay(self):
        if not SAVED_141.is_file():
            self.skipTest(f"saved run missing: {SAVED_141}")
        payload = json.loads(SAVED_141.read_text(encoding="utf-8"))
        report = inspect_saved_experiment_hybrid_add_replay(payload)
        self.assertFalse(report["can_replay_real_vocabulary_builder"])
        missing = set(report["missing_for_replay"])
        self.assertIn("full_factual_artifact_registry", missing)
        self.assertTrue(
            "checkpoint_artifact_catalog" in missing or "checkpoint_snapshots" in missing
        )
        observed = report["observed"]
        self.assertTrue(observed["available_add_count_all_zero"])
        self.assertGreater(observed["available_whatif_count_max"] or 0, 0)
        self.assertGreater(observed["available_remove_count_max"] or 0, 0)
        self.assertIsNone(report["exclusion_stage"])
        self.assertFalse(observed["hybrid_add_diagnostics_present"])
        self.assertGreater(len(observed["whatif_replacement_refs"]), 0)
        # Observed only: a WHATIF replacement can exist outside the explained
        # slice. That does not identify the ADD exclusion stage.


if __name__ == "__main__":
    unittest.main()
