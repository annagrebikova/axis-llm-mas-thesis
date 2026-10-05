"""Mocked tests for optional same-document hybrid ADD source (no live APIs)."""

from __future__ import annotations

import json
import math
import operator
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from mas_langgraph.artifacts.channel_state import (
    dump_artifact_channels,
    empty_artifact_channels,
)
from mas_langgraph.artifacts.models import ArtifactEnvelope, ArtifactRef
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.artifacts.resolver import ArtifactResolver
from mas_langgraph.axis.hybrid_add_diagnostics import (
    ADD_PRESENCE_UNKNOWN,
    HybridAddDiagnostics,
    extract_hybrid_add_diagnostics,
)
from mas_langgraph.axis.hybrid_add_source import (
    CANDIDATE_SOURCE_HYBRID_ADD_FROM_SOURCE,
    DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY,
    DEFAULT_HYBRID_ADD_SOURCE_LIMIT,
    DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS,
    DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
    EMPTY_BOUNDED_SEARCH_REASON,
    HYBRID_ADD_FROM_SOURCE_POLICY,
    INTERVENTION_PREP_BRANCH,
    POOL_FILENAME,
    SELECTION_RANK_INDEXING,
    corrupt_frozen_pool_report,
    export_selected_add_artifacts,
    frozen_pool_compatible,
    load_hybrid_add_source_pool_result,
    mmr_selection_score,
    persist_hybrid_add_source_pool,
    prepare_hybrid_add_source_registry,
    query_relevance_from_distance,
    registry_from_manifest,
    select_diverse_relevant,
    should_persist_hybrid_add_source_pool,
)
from mas_langgraph.axis.intervention_guidance import (
    build_hybrid_intervention_vocabulary,
    hybrid_opportunity_counts,
)
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.batch.manifest import (
    generation_config_fingerprint,
    generation_fingerprints_compatible,
)
from mas_langgraph.batch.generate import run_factual_and_explanation_in_memory
from mas_langgraph.cases.mcq.artifacts import wrap_evidence_chunk
from mas_langgraph.cases.mcq.intervention_view import project_cooperative_state_for_resolution
from mas_langgraph.experiment.models import (
    AnswerOptionView,
    ExplanationConditionResult,
    MCQExplanationComparison,
    QuizExplanationExperiment,
)
from mas_langgraph.core.entities import DocSection, EvidenceChunk, LearningObjective
from mas_langgraph.core.runtime import WorkflowRuntime
from mas_langgraph.evaluation.effect_reference import build_initial_hybrid_vocabulary
from mas_langgraph.evaluation.lifecycle import metrics_resume_ok
from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation
from mas_langgraph.core.checkpointer import default_factual_checkpointer
from mas_langgraph.replay.checkpoints import BoundarySelector
from mas_langgraph.replay.executor import LangGraphReplayExecutor
from mas_langgraph.replay.interventions import ArtifactIntervention
from mas_langgraph.tests.test_axis_hybrid_interrogation import (
    _availability_fixture,
    _hybrid_vocab,
    _mcq_contracts,
    _stamp_snapshot_catalogs,
    _store_meta,
)
from mas_langgraph.tests.test_hybrid_add_diagnostics import _build_vocab


@dataclass
class _Doc:
    page_content: str
    metadata: Dict[str, Any]


@dataclass
class _State:
    evidence_pool: Dict[str, EvidenceChunk] = field(default_factory=dict)
    doc_outline: List[DocSection] = field(default_factory=list)
    learning_objectives: List[LearningObjective] = field(default_factory=list)


def _factual_state() -> _State:
    return _State(
        evidence_pool={
            "chunk_00000001": EvidenceChunk(
                chunk_id="chunk_00000001",
                content="k-means assigns points to nearest centroids",
                metadata={
                    "pdf_id": "lecturebank_141",
                    "page_number": 1,
                    "chunk_index": 0,
                    "source_file": "141.pdf",
                    "section_id": "sec_a",
                },
                relevance_tags=["k-means clustering"],
            )
        },
        doc_outline=[
            DocSection(
                section_id="sec_a",
                title="K-means clustering",
                slide_range=(1, 4),
                importance_weight=1.0,
                content_summary="centroid assignment",
            )
        ],
        learning_objectives=[
            LearningObjective(
                lo_id="lo1",
                section_id="sec_a",
                objective_text="Explain k-means assignment",
                difficulty_level="easy",
            )
        ],
    )


def _hits_for(*rows):
    return [(_Doc(text, dict(meta)), score) for text, meta, score in rows]


def _same_doc_meta(page=2, chunk_index=1, pdf_id="lecturebank_141", **extra):
    meta = {
        "pdf_id": pdf_id,
        "page_number": page,
        "chunk_index": chunk_index,
        "source_file": "141.pdf",
        "title": "Deep Learning",
    }
    meta.update(extra)
    return meta


_RAG_FP = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "chunk_by_page": True,
}
_SOURCE_HASH = "a" * 64


def _prepare(**kwargs):
    defaults = dict(
        state=_factual_state(),
        document_id="lecturebank_141",
        lecturebank_id="141",
        run_id="run_src",
        source_hash=_SOURCE_HASH,
        rag_config_fingerprint=_RAG_FP,
        embedding_model=_RAG_FP["embedding_model"],
        limit=8,
    )
    defaults.update(kwargs)
    return prepare_hybrid_add_source_registry(**defaults)


def _prepared_add_env(
    reg: ArtifactRegistry,
    logical_id: str,
    *,
    selection_rank: int,
    section_id: str = "sec_a",
    tags: tuple = ("kmeans",),
    content: str = "",
) -> ArtifactEnvelope:
    chunk = EvidenceChunk(
        chunk_id=logical_id,
        content=content or f"prepared {logical_id}",
        metadata={"section_id": section_id, "pdf_id": "lecturebank_141"},
        relevance_tags=list(tags),
        flags=["hybrid_add_from_source"],
    )
    env = wrap_evidence_chunk(
        chunk,
        branch_id=INTERVENTION_PREP_BRANCH,
        created_in_run="run1:hybrid_add_prep",
    )
    stamped = ArtifactEnvelope(
        instance_id=env.instance_id,
        ref=env.ref,
        artifact_type=env.artifact_type,
        payload=env.payload,
        branch_id=env.branch_id,
        created_in_run=env.created_in_run,
        superseded=False,
        metadata={
            "hybrid_add_from_source": True,
            "retrieved_for_intervention_prep": True,
            "selection_rank": selection_rank,
            "selection_rank_indexing": SELECTION_RANK_INDEXING,
        },
    )
    return reg.store(stamped)


def _vocab_from_intervention(int_reg: ArtifactRegistry, *, cap: int = 8):
    explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
    catalog = {}
    for ref in case_slice.relevant_artifact_refs:
        item = reg.get_by_ref(ref, "factual") or reg.get_by_ref(ref)
        if item is not None:
            catalog[item.instance_id] = item
    _stamp_snapshot_catalogs(ex, "run1", catalog)
    vocab = _build_vocab(
        explained,
        reg,
        occs,
        idx,
        contracts,
        case_slice,
        ex,
        max_alternatives_per_target=cap,
        intervention_registry=int_reg,
    )
    return vocab


def _mini_batch_experiment(doc_id: str) -> QuizExplanationExperiment:
    case = MCQExplanationComparison(
        run_id=f"run_{doc_id}",
        branch_id="factual",
        candidate_ref=ArtifactRef("q1", 1),
        assessment_ref=ArtifactRef("a1", 1),
        decision_ref=ArtifactRef("d1", 1),
        question_stem=f"Question for {doc_id}?",
        answer_options=(
            AnswerOptionView("A", "First option", True),
            AnswerOptionView("B", "Second option", False),
        ),
        correct_option_id="A",
        factual_decision="accept",
        user_question="Why was this question selected?",
        trace=ExplanationConditionResult("trace", "Trace text."),
        trace_provenance=ExplanationConditionResult("trace_provenance", "Provenance text."),
        axis=ExplanationConditionResult("axis", "AXIS text."),
        blinding={"A": "trace", "B": "trace_provenance", "C": "axis"},
        axis_status="completed_with_counterfactual",
        axis_causal_evidence_available=True,
        successful_replay_count=1,
    )
    return QuizExplanationExperiment(
        schema_version=1,
        run_id=f"run_{doc_id}",
        branch_id="factual",
        experiment_seed=42,
        user_question="Why was this question selected?",
        cases=(case,),
    )


class _BatchHybridBackbone:
    def __init__(self, cfg, profile=None, *, bag: Dict[str, Any]):
        self.cfg = cfg
        self.profile = profile
        self.state = _factual_state()
        self._bag = bag

        class _Pipe:
            def search(inner_self, query, k=8, **kwargs):
                bag["search_calls"].append({"query": query, "k": k})
                impl = bag.get("search_impl")
                if callable(impl):
                    return impl(query, k=k, **kwargs)
                extra = "batch extra silhouette paragraph"
                return _hits_for((extra, _same_doc_meta(), 0.2))

        class _Agents:
            rag_pipeline = _Pipe()

        self.rag_agents = _Agents()

    def run_full_workflow(self) -> Dict[str, Any]:
        return {"run_id": f"run_{self.cfg.pdf_id}"}

    def list_explanation_experiment_refs(self, include_historical_versions: bool = False):
        return (ArtifactRef("q1", 1),)

    def generate_explanation_experiment(self, **kwargs: Any) -> QuizExplanationExperiment:
        self._bag["generate_kwargs"] = kwargs
        return _mini_batch_experiment(self.cfg.pdf_id)


def _batch_layout(tmp: Path):
    input_dir = tmp / "in"
    output_dir = tmp / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    jsonl = input_dir / "141.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")
    index_rel = "index_lecturebank_141"
    index_dir = output_dir / index_rel
    index_dir.mkdir()
    (index_dir / "dummy.bin").write_text("index", encoding="utf-8")
    record = {
        "document_id": "lecturebank_141",
        "lecturebank_id": "141",
        "rag_status": "success",
        "input_source_type": "extraction_jsonl",
        "jsonl_path": str(jsonl),
        "source_hash": _SOURCE_HASH,
        "embedding_model": _RAG_FP["embedding_model"],
        "chunk_size": _RAG_FP["chunk_size"],
        "chunk_overlap": _RAG_FP["chunk_overlap"],
        "chunk_by_page": _RAG_FP["chunk_by_page"],
        "index_dir": index_rel,
    }
    gen_cfg = {
        "llm_model": "gpt-4.1-mini-2025-04-14",
        "model_type": "openai",
        "temperature": 0.0,
        "seed": 42,
        "max_rounds": 5,
        "max_questions": 1,
        "question": "Why was this question selected?",
        "axis_interrogation_mode": "hybrid",
        "hybrid_add_from_source": True,
    }
    frozen_path = output_dir / "presentation_lecturebank_141" / POOL_FILENAME
    return input_dir, output_dir, record, gen_cfg, frozen_path


def _run_hybrid_batch(
    tmp: Path,
    *,
    reuse: bool = True,
    bag: Optional[Dict[str, Any]] = None,
    frozen_bytes: Optional[bytes] = None,
    frozen_text: Optional[str] = None,
):
    input_dir, output_dir, record, gen_cfg, frozen_path = _batch_layout(tmp)
    if frozen_bytes is not None or frozen_text is not None:
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        if frozen_bytes is not None:
            frozen_path.write_bytes(frozen_bytes)
        else:
            frozen_path.write_text(frozen_text or "", encoding="utf-8")
    capture = bag if bag is not None else {"search_calls": []}
    capture.setdefault("search_calls", [])

    def factory(cfg, profile=None):
        return _BatchHybridBackbone(cfg, profile, bag=capture)

    result = run_factual_and_explanation_in_memory(
        record=record,
        input_dir=input_dir,
        output_dir=output_dir,
        rag_config_values=dict(_RAG_FP),
        generation_config=gen_cfg,
        backbone_factory=factory,
        lecturebank_id="141",
        reuse_hybrid_add_source_pool=reuse,
    )
    return result, frozen_path, capture


class TestPrepareSameDocumentPool(unittest.TestCase):
    def test_relevant_same_document_chunk_is_registered(self):
        extra = "silhouette score chooses k"
        search = lambda query, k=8: _hits_for(
            (extra, _same_doc_meta(), 0.9),
        )
        state = _factual_state()
        registry, report = prepare_hybrid_add_source_registry(
            state=state,
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
            limit=8,
        )
        self.assertEqual(report.empty_reason, "")
        self.assertEqual(report.candidates_registered, 1)
        self.assertEqual(len(registry.catalog), 1)
        env = next(iter(registry.catalog.values()))
        self.assertEqual(env.payload.content, extra)
        self.assertEqual(env.payload.content, extra)
        self.assertTrue(str(env.ref.logical_id).startswith("addsrc_"))
        self.assertNotEqual(env.ref.logical_id, "chunk_00000001")
        self.assertEqual(env.branch_id, INTERVENTION_PREP_BRANCH)
        self.assertTrue(env.metadata.get("not_factually_produced"))
        self.assertTrue(env.metadata.get("retrieved_for_intervention_prep"))
        self.assertEqual(env.payload.metadata.get("section_id"), "sec_a")
        self.assertIn("chunk_00000001", state.evidence_pool)
        self.assertNotIn(env.ref.logical_id, state.evidence_pool)

    def test_wrong_document_rejected(self):
        search = lambda query, k=8: _hits_for(
            ("foreign text", _same_doc_meta(pdf_id="lecturebank_999"), 0.99),
        )
        _, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
        )
        self.assertEqual(report.wrong_document_rejected, 1)
        self.assertEqual(report.candidates_registered, 0)
        self.assertEqual(report.empty_reason, EMPTY_BOUNDED_SEARCH_REASON)

    def test_factual_chunk_under_different_id_rejected(self):
        factual = "k-means assigns points to nearest centroids"
        search = lambda query, k=8: _hits_for(
            (factual, _same_doc_meta(page=9, chunk_index=99), 0.95),
        )
        _, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
        )
        self.assertEqual(report.factual_duplicates_rejected, 1)
        self.assertEqual(report.candidates_registered, 0)

    def test_duplicate_external_candidates_deduplicated(self):
        extra = "elbow method for choosing k"
        search = lambda query, k=8: _hits_for(
            (extra, _same_doc_meta(page=2, chunk_index=1), 0.8),
            (extra, _same_doc_meta(page=2, chunk_index=1), 0.7),
            ("  ELBOW METHOD for choosing k  ", _same_doc_meta(page=3, chunk_index=2), 0.6),
        )
        _, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
        )
        self.assertEqual(report.candidates_registered, 1)
        self.assertGreaterEqual(report.pool_duplicates_rejected, 1)

    def test_no_suitable_chunks_empty_pool(self):
        search = lambda query, k=8: []
        _, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
        )
        self.assertEqual(report.candidates_registered, 0)
        self.assertEqual(report.empty_reason, EMPTY_BOUNDED_SEARCH_REASON)
        self.assertEqual(report.stopping_reason, "queries_complete")
        self.assertNotEqual(report.stopping_reason, "index_exhausted")
        self.assertFalse(report.retrieval_error)

    def test_search_failure_is_not_no_chunks_found(self):
        def boom(query, k=8):
            raise RuntimeError("chroma down")

        _, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=boom,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
        )
        self.assertEqual(report.empty_reason, "retrieval_failure")
        self.assertIn("chroma down", report.retrieval_error)
        self.assertNotIn("no relevant chunks found", (report.retrieval_error or "").lower())

    def test_title_is_not_document_identity(self):
        search = lambda query, k=8: _hits_for(
            (
                "looks similar",
                {
                    "pdf_id": "lecturebank_205",
                    "page_number": 1,
                    "chunk_index": 0,
                    "title": "Deep Learning",
                    "source_file": "141.pdf",
                },
                0.99,
            ),
        )
        _, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
        )
        self.assertEqual(report.wrong_document_rejected, 1)
        self.assertEqual(report.candidates_registered, 0)

    def test_does_not_mutate_factual_state(self):
        state = _factual_state()
        pool_before = dict(state.evidence_pool)
        extra = "new same-doc paragraph"
        search = lambda query, k=8: _hits_for((extra, _same_doc_meta(), 0.9))
        prepare_hybrid_add_source_registry(
            state=state,
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
        )
        self.assertEqual(set(state.evidence_pool), set(pool_before))
        self.assertEqual(
            state.evidence_pool["chunk_00000001"].content,
            pool_before["chunk_00000001"].content,
        )

    def test_missing_search_backend_is_explicit(self):
        _, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=None,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
        )
        self.assertEqual(report.empty_reason, "no_search_backend")
        self.assertEqual(report.retrieval_error, "no_search_backend")


class TestHybridAddVocabularyFromSource(unittest.TestCase):
    def _register_extra(self):
        extra = "silhouette score chooses k"
        search = lambda query, k=8: _hits_for((extra, _same_doc_meta(), 0.9))
        registry, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run1",
        )
        env = next(iter(registry.catalog.values()))
        return registry, report, env

    def test_absent_same_document_chunk_becomes_add_candidate(self):
        int_reg, _report, env = self._register_extra()
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        catalog = {}
        for ref in case_slice.relevant_artifact_refs:
            item = reg.get_by_ref(ref, "factual") or reg.get_by_ref(ref)
            if item is not None:
                catalog[item.instance_id] = item
        _stamp_snapshot_catalogs(ex, "run1", catalog)
        vocab = _build_vocab(
            explained,
            reg,
            occs,
            idx,
            contracts,
            case_slice,
            ex,
            intervention_registry=int_reg,
        )
        self.assertIn(env.ref, vocab.add_options)
        self.assertTrue(vocab.add_boundaries_for(env.ref))

    def test_missing_checkpoint_state_does_not_authorize_add(self):
        int_reg, _report, env = self._register_extra()
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        diag = HybridAddDiagnostics(
            run_id="run1",
            explained_ref="Q12:v2",
            lecturebank_id="141",
            document_id="lecturebank_141",
        )
        vocab = _build_vocab(
            explained,
            reg,
            occs,
            idx,
            contracts,
            case_slice,
            ex,
            diagnostics=diag,
            intervention_registry=int_reg,
            replay_executor=None,
        )
        self.assertNotIn(env.ref, vocab.add_options)
        unknown = [r for r in diag.boundary_rows if r["artifact_ref"] == str(env.ref)]
        self.assertTrue(unknown)
        self.assertTrue(all(r["presence"] == ADD_PRESENCE_UNKNOWN for r in unknown))
        self.assertTrue(all(r["advertised"] == "false" for r in unknown))
        self.assertEqual(diag.candidate_rows[0]["lecturebank_id"], "141")
        self.assertEqual(diag.boundary_rows[0]["document_id"], "lecturebank_141")

    def test_whatif_and_remove_unchanged(self):
        int_reg, _report, env = self._register_extra()
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        catalog = {}
        for ref in list(case_slice.relevant_artifact_refs) + [ArtifactRef("chunk_out", 1)]:
            item = reg.get_by_ref(ref, "factual") or reg.get_by_ref(ref)
            if item is not None:
                catalog[item.instance_id] = item
        _stamp_snapshot_catalogs(ex, "run1", catalog)
        without_ext, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        with_ext = _build_vocab(
            explained,
            reg,
            occs,
            idx,
            contracts,
            case_slice,
            ex,
            intervention_registry=int_reg,
        )
        self.assertEqual(without_ext.replacement_options, with_ext.replacement_options)
        self.assertEqual(
            hybrid_opportunity_counts(without_ext)["whatif"],
            hybrid_opportunity_counts(with_ext)["whatif"],
        )
        self.assertEqual(
            hybrid_opportunity_counts(without_ext)["remove"],
            hybrid_opportunity_counts(with_ext)["remove"],
        )
        self.assertNotIn(env.ref, [r for refs in with_ext.replacement_options.values() for r in refs])

    def test_default_off_preserves_vocabulary(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        catalog = {}
        for ref in case_slice.relevant_artifact_refs:
            item = reg.get_by_ref(ref, "factual") or reg.get_by_ref(ref)
            if item is not None:
                catalog[item.instance_id] = item
        _stamp_snapshot_catalogs(ex, "run1", catalog)
        off = _build_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        self.assertFalse(any(str(r).startswith("addsrc_") for r in off.add_options))


class TestReplayInsertsExternalPayload(unittest.TestCase):
    def test_external_payload_inserted_and_branches_isolated(self):
        extra = "silhouette score chooses k"
        search = lambda query, k=8: _hits_for((extra, _same_doc_meta(), 0.9))
        int_reg, _report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_addsrc",
        )
        env = next(iter(int_reg.catalog.values()))
        ext_ref = env.ref

        from langgraph.graph import END, START, StateGraph

        fact_reg = ArtifactRegistry()
        a_env = _store_meta(
            fact_reg,
            "chunk_A",
            atype="EvidenceChunk",
            content="factual A",
            section_id="sec_a",
            tags=("kmeans",),
        )
        a_ref = a_env.ref
        run_id = "run_addsrc"

        class S(TypedDict, total=False):
            chunk_count: int
            cooperative_state: Any
            branch_id: str
            run_id: str
            artifact_catalog: dict
            available_artifact_refs: set
            blocked_artifact_refs: set
            artifact_substitutions: dict
            artifact_derivations: list
            intervention_events: list
            occurrence_records: Annotated[list, operator.add]
            run_provenance_relations: Annotated[list, operator.add]
            path: Annotated[list, operator.add]
            branch_type: str
            parent_branch_id: str
            origin_checkpoint_alias: str
            intervention_id: Any

        def source(state: S) -> dict:
            return {
                "occurrence_records": [
                    OccurrenceRecord(
                        occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:source#1",
                        node_id="source",
                        occurrence_index=1,
                        branch_id=str(state.get("branch_id") or "factual"),
                        run_id=run_id,
                        execution_index=1,
                        produced_artifacts=(a_ref,),
                    )
                ],
                "run_provenance_relations": [
                    RunProvenanceRelation(
                        producer_occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:source#1",
                        artifact_ref=a_ref,
                        relation="supporting_evidence",
                        consumer_occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:consume#1",
                    )
                ],
                "path": ["source"],
            }

        def consume(state: S) -> dict:
            projected = project_cooperative_state_for_resolution(state)
            pool = getattr(projected, "evidence_pool", None) or {}
            n = len(pool)
            out = {
                "chunk_count": n,
                "occurrence_records": [
                    OccurrenceRecord(
                        occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:consume#1",
                        node_id="consume",
                        occurrence_index=1,
                        branch_id=str(state.get("branch_id") or "factual"),
                        run_id=run_id,
                        execution_index=2,
                    )
                ],
                "path": ["consume"],
            }
            if projected is not None:
                out["cooperative_state"] = projected
            return out

        g = StateGraph(S)
        g.add_node("source", source)
        g.add_node("consume", consume)
        g.add_edge(START, "source")
        g.add_edge("source", "consume")
        g.add_edge("consume", END)
        app = g.compile(checkpointer=default_factual_checkpointer())

        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState

        cs = CooperativeState()
        cs.evidence_pool = {
            "chunk_A": EvidenceChunk(
                chunk_id="chunk_A",
                content="factual A",
                metadata={"section_id": "sec_a", "pdf_id": "lecturebank_141"},
            )
        }
        resolver = ArtifactResolver(registry=fact_reg)
        resolver.make_available(a_ref)
        initial = {
            "chunk_count": 0,
            "path": [],
            "occurrence_records": [],
            "run_provenance_relations": [],
            "cooperative_state": cs,
            **empty_artifact_channels(),
            **dump_artifact_channels(fact_reg, resolver),
        }
        ex = LangGraphReplayExecutor(compiled_graph=app)
        ex.contracts = _mcq_contracts()
        factual = ex.run_factual(initial, run_id=run_id, runtime=WorkflowRuntime())
        self.assertIsNone(factual.error, msg=factual.error)
        factual_count = factual.final_state.get("chunk_count")
        origin = factual.checkpoint_index.resolve(
            BoundarySelector.before_occurrence("consume", 1)
        )
        noop = ex.fork_noop(run_id, BoundarySelector.by_alias(origin.alias))
        self.assertIsNone(noop.error, msg=noop.error)
        self.assertEqual(noop.final_state.get("chunk_count"), factual_count)

        cf1 = ex.fork_with_intervention(
            run_id,
            ArtifactIntervention(
                operation="ADD",
                artifact=ext_ref,
                checkpoint=origin,
            ),
            registry=int_reg,
        )
        self.assertIsNone(cf1.error, msg=cf1.error)
        projected = project_cooperative_state_for_resolution(cf1.final_state)
        self.assertIn(ext_ref.logical_id, projected.evidence_pool)
        self.assertEqual(projected.evidence_pool[ext_ref.logical_id].content, extra)
        self.assertGreater(cf1.final_state.get("chunk_count"), factual_count)

        cf2 = ex.fork_with_intervention(
            run_id,
            ArtifactIntervention(
                operation="ADD",
                artifact=ext_ref,
                checkpoint=origin,
            ),
            registry=int_reg,
        )
        self.assertIsNone(cf2.error, msg=cf2.error)
        self.assertNotEqual(cf1.branch_id, cf2.branch_id)
        self.assertEqual(factual.final_state.get("chunk_count"), factual_count)
        self.assertEqual(noop.final_state.get("chunk_count"), factual_count)
        self.assertNotIn(
            ext_ref.logical_id,
            getattr(project_cooperative_state_for_resolution(factual.final_state), "evidence_pool", {}),
        )


class TestResumeFingerprintAndReferenceSweep(unittest.TestCase):
    def test_fingerprint_includes_feature_and_saquet_independent(self):
        base = {
            "llm_model": "gpt-4.1-mini-2025-04-14",
            "model_type": "openai",
            "temperature": 0.0,
            "seed": 42,
            "max_rounds": 5,
            "max_questions": 10,
            "question": "Why was this question selected?",
            "orchestrator": "langgraph",
            "axis_interrogation_mode": "hybrid",
        }
        off = generation_config_fingerprint(base)
        on = generation_config_fingerprint({**base, "hybrid_add_from_source": True})
        saquet = generation_config_fingerprint({**base, "evaluate_saquet": True})
        self.assertFalse(off["hybrid_add_from_source"])
        self.assertTrue(on["hybrid_add_from_source"])
        self.assertEqual(on["hybrid_add_from_source_policy"], HYBRID_ADD_FROM_SOURCE_POLICY)
        self.assertEqual(on["hybrid_add_source_limit"], DEFAULT_HYBRID_ADD_SOURCE_LIMIT)
        self.assertEqual(
            on["hybrid_add_source_k_per_query"], DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY
        )
        self.assertEqual(
            on["hybrid_add_source_max_search_hits"],
            DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS,
        )
        self.assertEqual(
            on["hybrid_add_source_mmr_lambda"], DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA
        )
        self.assertEqual(
            on["hybrid_add_source_selection_rank_indexing"], SELECTION_RANK_INDEXING
        )
        self.assertIsNone(off["hybrid_add_source_k_per_query"])
        self.assertIsNone(off["hybrid_add_source_selection_rank_indexing"])
        self.assertIsNone(saquet["hybrid_add_source_mmr_lambda"])
        self.assertNotEqual(off, on)
        self.assertEqual(off, saquet)
        self.assertNotIn("diagnose_hybrid_add", off)
        self.assertTrue(generation_fingerprints_compatible(off, off))
        self.assertFalse(generation_fingerprints_compatible(off, on))
        old = {k: v for k, v in off.items() if not k.startswith("hybrid_add")}
        self.assertTrue(generation_fingerprints_compatible(old, off))
        self.assertFalse(generation_fingerprints_compatible(old, on))
        v2_on = {
            k: v
            for k, v in on.items()
            if k != "hybrid_add_source_selection_rank_indexing"
        }
        v2_on["hybrid_add_from_source_policy"] = "hybrid_add_from_source.v2"
        self.assertFalse(generation_fingerprints_compatible(v2_on, on))

    def test_resume_reuses_frozen_pool(self):
        extra = "frozen extra chunk text"
        calls = {"n": 0}

        def search(query, k=8, **_kwargs):
            calls["n"] += 1
            return _hits_for((extra, _same_doc_meta(), 0.2))

        registry, report = _prepare(search_fn=search, run_id="run_a")
        self.assertEqual(calls["n"], len(report.retrieval_config["queries"]))
        frozen = report.to_dict()
        self.assertTrue(
            frozen_pool_compatible(
                frozen,
                document_id="lecturebank_141",
                lecturebank_id="141",
                limit=8,
                embedding_model=_RAG_FP["embedding_model"],
                run_id="run_a",
                source_hash=_SOURCE_HASH,
                rag_config_fingerprint=_RAG_FP,
                factual_evidence_fingerprint=report.factual_evidence_fingerprint,
                queries=report.retrieval_config["queries"],
                k_per_query=DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY,
                max_search_hits=DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS,
                mmr_lambda=DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
                state=_factual_state(),
            )
        )

        def boom(query, k=8, **_kwargs):
            raise AssertionError("resume must not search again")

        reused_reg, reused = _prepare(
            search_fn=boom,
            run_id="run_a",
            frozen_manifest=frozen,
        )
        self.assertTrue(reused.reused_frozen_pool)
        orig = next(iter(registry.catalog.values()))
        again = next(iter(reused_reg.catalog.values()))
        self.assertEqual(orig.ref, again.ref)
        self.assertEqual(orig.payload.content, extra)
        self.assertEqual(again.payload.content, extra)
        self.assertEqual(again.payload.quality_score, 0.0)

        incompatible = dict(frozen)
        incompatible["retrieval_config"] = {**frozen["retrieval_config"], "limit": 3}
        self.assertFalse(
            frozen_pool_compatible(
                incompatible,
                document_id="lecturebank_141",
                lecturebank_id="141",
                limit=8,
                embedding_model=_RAG_FP["embedding_model"],
                run_id="run_a",
                source_hash=_SOURCE_HASH,
                rag_config_fingerprint=_RAG_FP,
                queries=frozen["retrieval_config"]["queries"],
                state=_factual_state(),
            )
        )

    def test_incompatible_resume_detected(self):
        off = generation_config_fingerprint(
            {"axis_interrogation_mode": "hybrid", "hybrid_add_from_source": False}
        )
        on = generation_config_fingerprint(
            {"axis_interrogation_mode": "hybrid", "hybrid_add_from_source": True}
        )
        stored = {
            "lecturebank_id": "141",
            "document_id": "lecturebank_141",
            "eval_status": "success",
            "source_hash": "abc",
            "rag_config": {"embedding_model": "x"},
            "generation_config": off,
            "metrics_persisted": True,
            "document_metrics": {"lecturebank_id": "141"},
        }
        collected = [{"lecturebank_id": "141", "document_metrics": {"lecturebank_id": "141"}}]
        self.assertTrue(
            metrics_resume_ok(
                lecturebank_id="141",
                source_hash="abc",
                rag_fp={"embedding_model": "x"},
                gen_fp=off,
                stored=stored,
                collected=collected,
            )
        )
        self.assertFalse(
            metrics_resume_ok(
                lecturebank_id="141",
                source_hash="abc",
                rag_fp={"embedding_model": "x"},
                gen_fp=on,
                stored=stored,
                collected=collected,
            )
        )

    def test_reference_sweep_uses_same_external_candidates(self):
        extra = "silhouette score chooses k"
        search = lambda query, k=8: _hits_for((extra, _same_doc_meta(), 0.9))
        int_reg, _report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run1",
        )
        env = next(iter(int_reg.catalog.values()))
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        catalog = {}
        for ref in case_slice.relevant_artifact_refs:
            item = reg.get_by_ref(ref, "factual") or reg.get_by_ref(ref)
            if item is not None:
                catalog[item.instance_id] = item
        _stamp_snapshot_catalogs(ex, "run1", catalog)
        vocab, _ = build_initial_hybrid_vocabulary(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            intervention_registry=int_reg,
            all_occurrences=occs,
            replay_executor=ex,
            run_id="run1",
        )
        self.assertIn(env.ref, vocab.add_options)

    def test_diagnostics_and_export_include_document_and_content(self):
        extra = "inspectable extra evidence"
        search = lambda query, k=8: _hits_for((extra, _same_doc_meta(), 0.9))
        int_reg, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_export",
        )
        env = next(iter(int_reg.catalog.values()))
        self.assertEqual(report.document_id, "lecturebank_141")
        self.assertEqual(report.lecturebank_id, "141")
        self.assertEqual(report.candidates[0]["content"], extra)

        class _Resolved:
            operation = "ADD"
            target_ref = env.ref

        class _Rec:
            resolved = _Resolved()

        class _Result:
            intervention_records = [_Rec()]

        exported = export_selected_add_artifacts(_Result(), int_reg)
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["content"], extra)
        self.assertEqual(exported[0]["candidate_source"], CANDIDATE_SOURCE_HYBRID_ADD_FROM_SOURCE)

        payload = {
            "lecturebank_id": "141",
            "hybrid_add_source": report.to_dict(),
            "cases": [
                {
                    "candidate_ref": "Q:v1",
                    "conditions": {
                        "axis": {
                            "raw_evidence": {
                                "hybrid_add_diagnostics": {
                                    "enabled": True,
                                    "run_id": "run_export",
                                    "explained_ref": "Q:v1",
                                    "candidate_rows": [
                                        {
                                            "lecturebank_id": "",
                                            "artifact_ref": str(env.ref),
                                            "candidate_source": "intervention_registry",
                                        }
                                    ],
                                    "boundary_rows": [
                                        {
                                            "lecturebank_id": "",
                                            "artifact_ref": str(env.ref),
                                            "boundary": "before:consume#1",
                                            "presence": "absent",
                                        }
                                    ],
                                },
                                "hybrid_add_source_artifacts": exported,
                            }
                        }
                    },
                }
            ],
        }
        extracted = extract_hybrid_add_diagnostics(payload, lecturebank_id="141")
        cand = extracted["hybrid_add_candidates"][0]
        bound = extracted["hybrid_add_boundaries"][0]
        self.assertEqual(cand["lecturebank_id"], "141")
        self.assertEqual(bound["lecturebank_id"], "141")
        self.assertEqual(cand["document_id"], "lecturebank_141")
        self.assertEqual(exported[0]["content"], extra)


class TestDistanceRankingAndBudgets(unittest.TestCase):
    def test_lower_chroma_distance_ranks_as_more_relevant(self):
        near = "near-query complementary silhouette analysis"
        far = "far-query leftover appendix citation"
        search = lambda query, k=32, **_k: _hits_for(
            (far, _same_doc_meta(page=4, chunk_index=4), 1.7),
            (near, _same_doc_meta(page=3, chunk_index=3), 0.05),
        )
        _, report = _prepare(search_fn=search, limit=1, embed_fn=None)
        self.assertEqual(report.candidates_registered, 1)
        self.assertEqual(report.candidates[0]["content"], near)
        self.assertEqual(report.candidates[0]["retrieval_distance"], 0.05)
        self.assertEqual(report.selection_mode, "relevance_only_embedding_unavailable")
        self.assertTrue(report.embedding_unavailable)
        env_quality = 0.0
        # quality_score must not copy raw distance
        _, report2 = _prepare(search_fn=search, limit=2, embed_fn=None)
        self.assertEqual([c["content"] for c in report2.candidates], [near, far])
        self.assertLess(
            report2.candidates[0]["retrieval_distance"],
            report2.candidates[1]["retrieval_distance"],
        )
        self.assertEqual(env_quality, 0.0)

    def test_multi_query_hit_keeps_best_distance(self):
        extra = "shared chunk seen under several queries"
        distances = {"first": 0.9, "later": 0.15, "other": 0.4}

        def search(query, k=32, **_k):
            q = str(query).lower()
            if "k-means clustering" in q:
                d = distances["first"]
            elif "centroid" in q:
                d = distances["later"]
            else:
                d = distances["other"]
            return _hits_for((extra, _same_doc_meta(page=5, chunk_index=5), d))

        _, report = _prepare(search_fn=search, limit=1, embed_fn=None)
        self.assertEqual(report.candidates_registered, 1)
        self.assertEqual(report.candidates[0]["retrieval_distance"], 0.15)
        self.assertGreaterEqual(len(report.candidates[0]["supporting_queries"]), 2)
        self.assertIn("centroid", report.candidates[0]["query"].lower())

    def test_invalid_and_non_finite_distances_are_explicit(self):
        good = "valid distance extra paragraph"
        search = lambda query, k=32, **_k: [
            (_Doc("nan chunk", _same_doc_meta(page=6, chunk_index=1)), math.nan),
            (_Doc("inf chunk", _same_doc_meta(page=6, chunk_index=2)), math.inf),
            (_Doc("missing score", _same_doc_meta(page=6, chunk_index=3)),),
            (_Doc(good, _same_doc_meta(page=6, chunk_index=4)), 0.3),
        ]
        _, report = _prepare(search_fn=search, limit=8, embed_fn=None)
        self.assertGreaterEqual(report.invalid_distance_rejected, 1)
        self.assertEqual(report.candidates_registered, 1)
        self.assertEqual(report.candidates[0]["content"], good)
        self.assertNotIn("no_additional_same_document_chunks", report.empty_reason)
        events = [e["event_type"] for e in report.events]
        self.assertIn("invalid_distance", events)

    def test_deeper_than_eight_eligible_hit_is_considered(self):
        factual = "k-means assigns points to nearest centroids"
        extra = "ninth hit is a distinct same-document paragraph"
        ks = []

        def search(query, k=8, **_k):
            ks.append(k)
            rows = [
                (factual, _same_doc_meta(page=1, chunk_index=i), 0.01 + i * 0.001)
                for i in range(8)
            ]
            rows.append((extra, _same_doc_meta(page=8, chunk_index=8), 0.2))
            return _hits_for(*rows[:k])

        _, hidden = _prepare(search_fn=search, limit=8, k_per_query=8, embed_fn=None)
        self.assertEqual(hidden.candidates_registered, 0)
        self.assertEqual(hidden.factual_duplicates_rejected, 8)
        _, found = _prepare(search_fn=search, limit=8, embed_fn=None)
        self.assertEqual(found.candidates_registered, 1)
        self.assertEqual(found.candidates[0]["content"], extra)
        self.assertGreaterEqual(min(ks), 8)
        self.assertIn(DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY, ks)

    def test_retrieval_remains_bounded(self):
        calls = {"n": 0, "k": []}

        def search(query, k=8, **_k):
            calls["n"] += 1
            calls["k"].append(k)
            rows = [
                (f"extra chunk number {i}", _same_doc_meta(page=10, chunk_index=i), 0.1 + i * 0.01)
                for i in range(20)
            ]
            return _hits_for(*rows[:k])

        _, report = _prepare(
            search_fn=search,
            limit=3,
            k_per_query=5,
            max_search_hits=4,
            embed_fn=None,
        )
        self.assertLessEqual(report.unique_hits, 4)
        self.assertEqual(report.stopping_reason, "search_budget_exhausted")
        self.assertEqual(set(calls["k"]), {5})
        self.assertLess(calls["n"], len(report.retrieval_config["queries"]))
        self.assertLessEqual(report.candidates_registered, 3)

    def test_empty_bounded_search_is_not_exhaustive_absence(self):
        _, report = _prepare(search_fn=lambda query, k=8, **_k: [], embed_fn=None)
        self.assertEqual(report.empty_reason, EMPTY_BOUNDED_SEARCH_REASON)
        self.assertEqual(report.stopping_reason, "queries_complete")
        self.assertNotEqual(report.stopping_reason, "index_exhausted")
        text = " ".join(
            str(e.get("detail") or "") for e in report.events if e["event_type"] == "empty_pool"
        ).lower()
        self.assertIn("does not establish", text)
        self.assertNotEqual(report.stopping_reason, "index_exhausted")

    def test_index_exhausted_only_when_established(self):
        extra = "only extra in tiny index"
        search = lambda query, k=32, **_k: _hits_for(
            ("k-means assigns points to nearest centroids", _same_doc_meta(page=1, chunk_index=0), 0.01),
            (extra, _same_doc_meta(page=2, chunk_index=1), 0.2),
        )
        _, report = _prepare(
            search_fn=search,
            embed_fn=None,
            index_chunk_count=2,
        )
        self.assertEqual(report.stopping_reason, "index_exhausted")
        self.assertEqual(report.candidates_registered, 1)


class TestMmrDiversityAndDeterminism(unittest.TestCase):
    def _vectors(self):
        factual = "k-means assigns points to nearest centroids"
        paraphrase = "k-means assigns each point to the nearest centroid"
        complementary = "silhouette score chooses k independently of assignment"
        unrelated = "parking permit refund policy for campus visitors"
        vectors = {
            factual: [1.0, 0.0, 0.0],
            paraphrase: [0.99, 0.01, 0.0],
            complementary: [0.0, 1.0, 0.0],
            unrelated: [0.0, 0.0, 1.0],
        }

        def embed_fn(texts):
            return [list(vectors.get(t, [0.0, 0.0, 0.0])) for t in texts]

        return factual, paraphrase, complementary, unrelated, embed_fn

    def test_complementary_can_outrank_redundant_paraphrase(self):
        _fact, paraphrase, complementary, _unrel, embed_fn = self._vectors()
        search = lambda query, k=32, **_k: _hits_for(
            (paraphrase, _same_doc_meta(page=3, chunk_index=1), 0.05),
            (complementary, _same_doc_meta(page=3, chunk_index=2), 0.20),
        )
        _, report = _prepare(
            search_fn=search,
            embed_fn=embed_fn,
            limit=1,
            mmr_lambda=DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
        )
        self.assertEqual(report.selection_mode, "mmr_distance_relevance_v1")
        self.assertFalse(report.embedding_unavailable)
        self.assertEqual(report.candidates[0]["content"], complementary)
        considered = {row["selection_reason"]: row for row in report.considered}
        self.assertTrue(any(row["content_hash"] and row["selected"] for row in report.considered))
        para = [row for row in report.considered if not row["selected"]][0]
        self.assertIsNotNone(para["redundancy_component"])
        self.assertGreater(para["redundancy_component"], 0.5)

    def test_mmr_synthetic_fixture_far_orthogonal_loses_when_limit_is_small(self):
        """Synthetic embeddings only: not a guarantee of semantic relevance or novelty."""
        _fact, paraphrase, complementary, unrelated, embed_fn = self._vectors()
        search = lambda query, k=32, **_k: _hits_for(
            (paraphrase, _same_doc_meta(page=3, chunk_index=1), 0.05),
            (complementary, _same_doc_meta(page=3, chunk_index=2), 0.20),
            (unrelated, _same_doc_meta(page=9, chunk_index=9), 1.80),
        )
        _, report = _prepare(
            search_fn=search,
            embed_fn=embed_fn,
            limit=1,
            mmr_lambda=DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
        )
        self.assertEqual(report.candidates[0]["content"], complementary)
        self.assertNotEqual(report.candidates[0]["content"], unrelated)
        _, report2 = _prepare(
            search_fn=search,
            embed_fn=embed_fn,
            limit=2,
            mmr_lambda=DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
        )
        selected = [c["content"] for c in report2.candidates]
        self.assertNotIn(unrelated, selected)

    def test_mmr_lambda_07_numerical_tradeoff(self):
        """Formula trade-off at λ=0.7; not proof that either candidate is semantically relevant."""
        close_redundant = mmr_selection_score(
            relevance=query_relevance_from_distance(0.5),
            redundancy=1.0,
            mmr_lambda=DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
        )
        far_novel = mmr_selection_score(
            relevance=query_relevance_from_distance(1.5),
            redundancy=0.0,
            mmr_lambda=DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
        )
        self.assertAlmostEqual(close_redundant, 0.1667, places=4)
        self.assertAlmostEqual(far_novel, 0.28, places=4)
        self.assertGreater(far_novel, close_redundant)

    def test_identical_inputs_produce_identical_pools(self):
        extra_a = "first extra chunk about elbow method"
        extra_b = "second extra chunk about gap statistic"
        search = lambda query, k=32, **_k: _hits_for(
            (extra_a, _same_doc_meta(page=4, chunk_index=1), 0.2),
            (extra_b, _same_doc_meta(page=4, chunk_index=2), 0.3),
        )
        vectors = {
            "k-means assigns points to nearest centroids": [1.0, 0.0],
            extra_a: [0.0, 1.0],
            extra_b: [0.1, 0.9],
        }

        def embed_fn(texts):
            return [list(vectors.get(t, [0.0, 0.0])) for t in texts]

        first = _prepare(search_fn=search, embed_fn=embed_fn, limit=2)
        second = _prepare(search_fn=search, embed_fn=embed_fn, limit=2)
        self.assertEqual(
            [c["logical_id"] for c in first[1].candidates],
            [c["logical_id"] for c in second[1].candidates],
        )
        self.assertEqual(
            [c["content"] for c in first[1].candidates],
            [c["content"] for c in second[1].candidates],
        )
        self.assertEqual(
            first[1].retrieval_config["queries"],
            second[1].retrieval_config["queries"],
        )


class TestFrozenPoolIdentity(unittest.TestCase):
    def _frozen(self):
        extra = "frozen extra chunk text for identity tests"
        search = lambda query, k=32, **_k: _hits_for((extra, _same_doc_meta(), 0.2))
        registry, report = _prepare(search_fn=search, run_id="run_freeze", embed_fn=None)
        return extra, registry, report, report.to_dict()

    def test_compatible_resume_does_not_search(self):
        extra, registry, report, frozen = self._frozen()

        def boom(query, k=32, **_k):
            raise AssertionError("compatible resume must not search")

        reused_reg, reused = _prepare(
            search_fn=boom,
            run_id="run_freeze",
            frozen_manifest=frozen,
            embed_fn=None,
        )
        self.assertTrue(reused.reused_frozen_pool)
        self.assertEqual(reused.frozen_compatible, True)
        orig = next(iter(registry.catalog.values()))
        again = next(iter(reused_reg.catalog.values()))
        self.assertEqual(orig.ref, again.ref)
        self.assertEqual(again.payload.content, extra)
        self.assertEqual(orig.metadata.get("selection_rank"), 0)
        self.assertEqual(again.metadata.get("selection_rank"), 0)

    def test_identity_changes_invalidate_reuse(self):
        _extra, _registry, report, frozen = self._frozen()

        def boom(query, k=32, **_k):
            raise AssertionError("incompatible resume must not search")

        cases = [
            {"run_id": "run_other"},
            {"source_hash": "b" * 64},
            {"document_id": "lecturebank_999"},
            {"rag_config_fingerprint": {**_RAG_FP, "chunk_size": 999}},
            {"k_per_query": 16},
            {"mmr_lambda": 0.1},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                prep_kwargs = dict(
                    search_fn=boom,
                    frozen_manifest=frozen,
                    embed_fn=None,
                    run_id="run_freeze",
                )
                prep_kwargs.update(kwargs)
                reg, reused = _prepare(**prep_kwargs)
                self.assertFalse(reused.reused_frozen_pool)
                self.assertEqual(reused.empty_reason, "frozen_pool_incompatible")
                self.assertEqual(len(reg.catalog), 0)
                self.assertTrue(reused.frozen_mismatch_reasons)

        changed_state = _factual_state()
        changed_state.evidence_pool["chunk_00000002"] = EvidenceChunk(
            chunk_id="chunk_00000002",
            content="new factual evidence body",
            metadata={"pdf_id": "lecturebank_141", "page_number": 2, "chunk_index": 2},
        )
        reg, reused = _prepare(
            state=changed_state,
            search_fn=boom,
            run_id="run_freeze",
            frozen_manifest=frozen,
            embed_fn=None,
        )
        self.assertEqual(reused.empty_reason, "frozen_pool_incompatible")
        self.assertIn("factual_evidence_fingerprint", reused.frozen_mismatch_reasons)

        query_state = _factual_state()
        query_state.doc_outline[0].title = "Completely different outline title"
        reg, reused = _prepare(
            state=query_state,
            search_fn=boom,
            run_id="run_freeze",
            frozen_manifest=frozen,
            embed_fn=None,
        )
        self.assertEqual(reused.empty_reason, "frozen_pool_incompatible")
        self.assertIn("queries", reused.frozen_mismatch_reasons)

    def test_corrupted_hash_and_new_factual_duplicate_rejected(self):
        extra, _registry, _report, frozen = self._frozen()

        def boom(query, k=32, **_k):
            raise AssertionError("corrupt pool must not search")

        corrupt = json.loads(json.dumps(frozen))
        corrupt["candidates"][0]["content_hash"] = "deadbeef"
        reg, reused = _prepare(
            search_fn=boom,
            run_id="run_freeze",
            frozen_manifest=corrupt,
            embed_fn=None,
        )
        self.assertEqual(reused.empty_reason, "frozen_pool_incompatible")
        self.assertIn("content_hash_mismatch", reused.frozen_mismatch_reasons)
        self.assertEqual(len(reg.catalog), 0)

        dup_state = _factual_state()
        dup_state.evidence_pool["chunk_dup"] = EvidenceChunk(
            chunk_id="chunk_dup",
            content=extra,
            metadata={"pdf_id": "lecturebank_141", "page_number": 9, "chunk_index": 9},
        )
        reg, reused = _prepare(
            state=dup_state,
            search_fn=boom,
            run_id="run_freeze",
            frozen_manifest=frozen,
            embed_fn=None,
        )
        self.assertEqual(reused.empty_reason, "frozen_pool_incompatible")
        self.assertTrue(
            "factual_duplicate_on_resume" in reused.frozen_mismatch_reasons
            or "factual_evidence_fingerprint" in reused.frozen_mismatch_reasons
        )

    def test_failed_prep_is_not_a_valid_empty_pool(self):
        failed = {
            "policy": HYBRID_ADD_FROM_SOURCE_POLICY,
            "document_id": "lecturebank_141",
            "lecturebank_id": "141",
            "run_id": "run_freeze",
            "source_hash": _SOURCE_HASH,
            "empty_reason": "retrieval_failure",
            "retrieval_error": "chroma down",
            "candidates": [],
            "retrieval_config": {"limit": 8},
        }
        calls = {"n": 0}

        def search(query, k=32, **_k):
            calls["n"] += 1
            return _hits_for(("retry extra after failure", _same_doc_meta(), 0.2))

        _reg, report = _prepare(
            search_fn=search,
            run_id="run_freeze",
            frozen_manifest=failed,
            embed_fn=None,
        )
        self.assertGreater(calls["n"], 0)
        self.assertNotEqual(report.empty_reason, "retrieval_failure")
        self.assertEqual(report.candidates_registered, 1)

    def test_missing_identity_is_not_a_wildcard(self):
        extra = "needs identity"
        search = lambda query, k=32, **_k: _hits_for((extra, _same_doc_meta(), 0.2))
        _reg, report = prepare_hybrid_add_source_registry(
            state=_factual_state(),
            search_fn=search,
            document_id="lecturebank_141",
            lecturebank_id="141",
            run_id="run_src",
            limit=8,
        )
        frozen = report.to_dict()
        self.assertFalse(
            frozen_pool_compatible(
                frozen,
                document_id="lecturebank_141",
                lecturebank_id="141",
                limit=8,
            )
        )


class TestQueryRelevanceHelper(unittest.TestCase):
    def test_relevance_is_decreasing_in_distance(self):
        self.assertGreater(
            query_relevance_from_distance(0.1),
            query_relevance_from_distance(0.5),
        )


class TestSelectionRankVocabularyOrder(unittest.TestCase):
    def test_vocab_preserves_selection_rank_over_reference_order(self):
        int_reg = ArtifactRegistry()
        preferred = _prepared_add_env(int_reg, "zzz_pref", selection_rank=0)
        later = _prepared_add_env(int_reg, "aaa_pref", selection_rank=1)
        self.assertGreater(str(preferred.ref), str(later.ref))
        vocab = _vocab_from_intervention(int_reg, cap=8)
        prepared = [ref for ref in vocab.add_options if str(ref.logical_id).startswith("zzz_pref") or str(ref.logical_id).startswith("aaa_pref")]
        self.assertEqual(prepared, [preferred.ref, later.ref])

    def test_vocabulary_cap_keeps_higher_ranked_eligible_candidate(self):
        int_reg = ArtifactRegistry()
        preferred = _prepared_add_env(int_reg, "zzz_pref", selection_rank=0)
        _prepared_add_env(int_reg, "aaa_pref", selection_rank=1)
        vocab = _vocab_from_intervention(int_reg, cap=1)
        self.assertIn(preferred.ref, vocab.add_options)
        self.assertEqual(vocab.add_options[0], preferred.ref)
        self.assertFalse(any(str(r.logical_id).startswith("aaa_pref") for r in vocab.add_options))

    def test_section_match_priority_still_applies(self):
        int_reg = ArtifactRegistry()
        other_section = _prepared_add_env(
            int_reg, "zzz_other", selection_rank=0, section_id="sec_z", tags=("kmeans",)
        )
        matching = _prepared_add_env(
            int_reg, "aaa_match", selection_rank=1, section_id="sec_a", tags=("kmeans",)
        )
        vocab = _vocab_from_intervention(int_reg, cap=1)
        self.assertEqual(vocab.add_options[0], matching.ref)
        self.assertNotIn(other_section.ref, vocab.add_options)

    def test_ineligible_higher_rank_does_not_block_lower_rank(self):
        int_reg = ArtifactRegistry()
        ineligible = _prepared_add_env(
            int_reg, "zzz_ineligible", selection_rank=0, section_id="sec_none", tags=()
        )
        eligible = _prepared_add_env(
            int_reg, "aaa_eligible", selection_rank=1, section_id="sec_a", tags=("kmeans",)
        )
        vocab = _vocab_from_intervention(int_reg, cap=8)
        self.assertNotIn(ineligible.ref, vocab.add_options)
        self.assertIn(eligible.ref, vocab.add_options)

    def test_frozen_pool_round_trip_preserves_rank_and_vocab_order(self):
        near = "preferred extra silhouette analysis"
        far = "later extra appendix citation"
        search = lambda query, k=32, **_k: _hits_for(
            (far, _same_doc_meta(page=4, chunk_index=4), 0.9),
            (near, _same_doc_meta(page=3, chunk_index=3), 0.1),
        )
        registry, report = _prepare(search_fn=search, embed_fn=None, limit=2)
        ranks = [c["selection_rank"] for c in report.candidates]
        self.assertEqual(ranks, [0, 1])
        self.assertEqual(report.candidates[0]["content"], near)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / POOL_FILENAME
            persist_hybrid_add_source_pool(path, report)
            loaded = load_hybrid_add_source_pool_result(path)
            self.assertEqual(loaded.status, "loaded")
            reused_reg = registry_from_manifest(loaded.manifest, run_id="run_src")
        reconstructed = sorted(
            reused_reg.catalog.values(),
            key=lambda env: env.metadata.get("selection_rank"),
        )
        self.assertEqual(
            [env.metadata.get("selection_rank") for env in reconstructed],
            [0, 1],
        )
        orig_order = [c["artifact_ref"] for c in report.candidates]
        self.assertEqual([str(env.ref) for env in reconstructed], orig_order)
        vocab = _vocab_from_intervention(reused_reg, cap=8)
        prepared = [ref for ref in vocab.add_options if str(ref).startswith("addsrc_")]
        self.assertEqual([str(ref) for ref in prepared[:2]], orig_order)

    def test_relevance_only_fallback_preserves_preparation_order(self):
        rows = [
            {
                "text": "later far hit",
                "logical_id": "addsrc_aaaa",
                "retrieval_distance": 0.9,
                "content_hash": "h-far",
            },
            {
                "text": "preferred near hit",
                "logical_id": "addsrc_zzzz",
                "retrieval_distance": 0.1,
                "content_hash": "h-near",
            },
        ]
        annotated, mode, unavailable, _err = select_diverse_relevant(
            rows,
            factual_texts=["k-means assigns points to nearest centroids"],
            limit=2,
            mmr_lambda=DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
            embed_fn=None,
        )
        self.assertEqual(mode, "relevance_only_embedding_unavailable")
        self.assertTrue(unavailable)
        selected = [row for row in annotated if row.get("selected")]
        self.assertEqual([row["text"] for row in selected], ["preferred near hit", "later far hit"])
        self.assertEqual([row["selection_rank"] for row in selected], [0, 1])
        int_reg = ArtifactRegistry()
        for row in selected:
            _prepared_add_env(
                int_reg,
                row["logical_id"],
                selection_rank=int(row["selection_rank"]),
            )
        vocab = _vocab_from_intervention(int_reg, cap=8)
        prepared = [
            ref
            for ref in vocab.add_options
            if str(ref.logical_id) in {"addsrc_zzzz", "addsrc_aaaa"}
        ]
        self.assertEqual(
            [ref.logical_id for ref in prepared],
            ["addsrc_zzzz", "addsrc_aaaa"],
        )

    def test_historical_manifest_without_new_policy_is_incompatible(self):
        extra = "legacy extra chunk"
        search = lambda query, k=32, **_k: _hits_for((extra, _same_doc_meta(), 0.2))
        _reg, report = _prepare(search_fn=search, embed_fn=None, run_id="run_freeze")
        frozen = report.to_dict()
        frozen["policy"] = "hybrid_add_from_source.v2"
        cfg = dict(frozen.get("retrieval_config") or {})
        cfg.pop("selection_rank_indexing", None)
        frozen["retrieval_config"] = cfg
        for item in frozen.get("candidates") or []:
            item.pop("selection_rank", None)
            item.pop("selection_rank_indexing", None)

        def boom(query, k=32, **_k):
            raise AssertionError("historical pool must not search or invent ranks")

        reg, reused = _prepare(
            search_fn=boom,
            run_id="run_freeze",
            frozen_manifest=frozen,
            embed_fn=None,
        )
        self.assertEqual(reused.empty_reason, "frozen_pool_incompatible")
        self.assertIn("policy", reused.frozen_mismatch_reasons)
        self.assertIn("selection_rank_indexing", reused.frozen_mismatch_reasons)
        self.assertEqual(len(reg.catalog), 0)
        self.assertFalse(reused.external_add_pool_available)
        self.assertEqual(reused.source_preparation_status, "failed")

        v3_missing_ranks = report.to_dict()
        for item in v3_missing_ranks.get("candidates") or []:
            item.pop("selection_rank", None)
        reg2, reused2 = _prepare(
            search_fn=boom,
            run_id="run_freeze",
            frozen_manifest=v3_missing_ranks,
            embed_fn=None,
        )
        self.assertEqual(reused2.empty_reason, "frozen_pool_incompatible")
        self.assertIn("missing_selection_rank", reused2.frozen_mismatch_reasons)
        self.assertEqual(len(reg2.catalog), 0)


class TestCorruptFrozenPoolBatchPath(unittest.TestCase):
    def _assert_factual_exportable(self, result):
        self.assertIsNotNone(result.experiment)
        self.assertTrue(result.experiment.cases)
        payload = result.experiment_payload or {}
        self.assertIn("factual_quiz", payload)
        self.assertEqual(payload["factual_quiz"]["kind"], "final_accepted_factual_quiz")

    def test_malformed_json_does_not_search_or_overwrite(self):
        garbage = b"{not valid json"
        with tempfile.TemporaryDirectory() as td:
            result, frozen_path, bag = _run_hybrid_batch(
                Path(td), frozen_bytes=garbage
            )
            after = frozen_path.read_bytes()
            src = (result.experiment_payload or {}).get("hybrid_add_source") or {}
            self.assertEqual(after, garbage)
            self.assertEqual(bag["search_calls"], [])
            self.assertIsNone((bag.get("generate_kwargs") or {}).get("intervention_registry"))
            self.assertEqual(src.get("empty_reason"), "frozen_pool_corrupt")
            self.assertIn("malformed_json", str(src.get("corrupt_reason") or ""))
            self.assertEqual(src.get("source_preparation_status"), "failed")
            self.assertFalse(src.get("external_add_pool_available"))
            self.assertTrue(src.get("requested_hybrid_add_from_source"))
            self.assertFalse(should_persist_hybrid_add_source_pool(src))
            warnings = result.record.get("warnings") or []
            self.assertTrue(any("hybrid_add_from_source_corrupt_pool" in str(w) for w in warnings))
            self.assertTrue(any(str(frozen_path) in str(w) for w in warnings))
            self._assert_factual_exportable(result)

    def test_non_object_json_is_corrupt_not_missing(self):
        text = "[1, 2]"
        with tempfile.TemporaryDirectory() as td:
            result, frozen_path, bag = _run_hybrid_batch(
                Path(td), frozen_text=text
            )
            self.assertEqual(frozen_path.read_text(encoding="utf-8"), text)
            self.assertEqual(bag["search_calls"], [])
            src = (result.experiment_payload or {}).get("hybrid_add_source") or {}
            self.assertEqual(src.get("empty_reason"), "frozen_pool_corrupt")
            self.assertIn("invalid_top_level_type", str(src.get("corrupt_reason") or ""))
            self.assertEqual(src.get("source_preparation_status"), "failed")
            self._assert_factual_exportable(result)

    def test_missing_file_still_permits_normal_preparation(self):
        with tempfile.TemporaryDirectory() as td:
            result, frozen_path, bag = _run_hybrid_batch(Path(td))
            self.assertTrue(bag["search_calls"])
            self.assertTrue(frozen_path.is_file())
            loaded = load_hybrid_add_source_pool_result(frozen_path)
            self.assertEqual(loaded.status, "loaded")
            src = (result.experiment_payload or {}).get("hybrid_add_source") or {}
            self.assertNotEqual(src.get("empty_reason"), "frozen_pool_corrupt")
            self.assertGreaterEqual(int(src.get("candidates_registered") or 0), 1)
            self.assertTrue(src.get("external_add_pool_available"))
            self._assert_factual_exportable(result)

    def test_compatible_valid_pool_resumes_without_search(self):
        extra = "frozen extra chunk text for identity tests"
        search = lambda query, k=32, **_k: _hits_for((extra, _same_doc_meta(), 0.2))
        _reg, report = _prepare(
            search_fn=search,
            run_id="run_lecturebank_141",
            embed_fn=None,
        )
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            input_dir, output_dir, record, gen_cfg, frozen_path = _batch_layout(tmp)
            persist_hybrid_add_source_pool(frozen_path, report)
            before = frozen_path.read_bytes()
            bag = {"search_calls": []}

            def factory(cfg, profile=None):
                return _BatchHybridBackbone(cfg, profile, bag=bag)

            result = run_factual_and_explanation_in_memory(
                record=record,
                input_dir=input_dir,
                output_dir=output_dir,
                rag_config_values=dict(_RAG_FP),
                generation_config=gen_cfg,
                backbone_factory=factory,
                lecturebank_id="141",
                reuse_hybrid_add_source_pool=True,
            )
            self.assertEqual(bag["search_calls"], [])
            self.assertEqual(frozen_path.read_bytes(), before)
            src = (result.experiment_payload or {}).get("hybrid_add_source") or {}
            self.assertTrue(src.get("reused_frozen_pool"))
            self.assertEqual((bag.get("generate_kwargs") or {}).get("intervention_registry") is not None, True)
            self._assert_factual_exportable(result)

    def test_persisted_retrieval_failure_is_retryable(self):
        failed = {
            "policy": HYBRID_ADD_FROM_SOURCE_POLICY,
            "document_id": "lecturebank_141",
            "lecturebank_id": "141",
            "run_id": "run_lecturebank_141",
            "source_hash": _SOURCE_HASH,
            "empty_reason": "retrieval_failure",
            "retrieval_error": "chroma down",
            "candidates": [],
            "retrieval_config": {"limit": 8},
        }
        with tempfile.TemporaryDirectory() as td:
            result, frozen_path, bag = _run_hybrid_batch(
                Path(td), frozen_text=json.dumps(failed)
            )
            self.assertTrue(bag["search_calls"])
            src = (result.experiment_payload or {}).get("hybrid_add_source") or {}
            self.assertNotEqual(src.get("empty_reason"), "retrieval_failure")
            self.assertGreaterEqual(int(src.get("candidates_registered") or 0), 1)
            self._assert_factual_exportable(result)

    def test_overwrite_rebuilds_corrupt_pool(self):
        garbage = b"{not valid json"
        with tempfile.TemporaryDirectory() as td:
            result, frozen_path, bag = _run_hybrid_batch(
                Path(td),
                reuse=False,
                frozen_bytes=garbage,
            )
            self.assertTrue(bag["search_calls"])
            after = frozen_path.read_bytes()
            self.assertNotEqual(after, garbage)
            loaded = json.loads(after.decode("utf-8"))
            self.assertIsInstance(loaded, dict)
            self.assertGreaterEqual(len(loaded.get("candidates") or []), 1)
            src = (result.experiment_payload or {}).get("hybrid_add_source") or {}
            self.assertNotEqual(src.get("empty_reason"), "frozen_pool_corrupt")
            self._assert_factual_exportable(result)


if __name__ == "__main__":
    unittest.main()
