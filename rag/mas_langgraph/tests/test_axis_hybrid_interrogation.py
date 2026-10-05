"""Constrained vs hybrid AXIS interrogation (no live OpenAI)."""

from __future__ import annotations

import operator
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest.mock import patch

from mas_langgraph.artifacts.models import ArtifactEnvelope, ArtifactRef
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.artifacts.resolver import ArtifactResolver
from mas_langgraph.axis.explainer import (
    AxisExplanationController,
    _build_factual_context,
    _render_system,
    hybridize_system_prompt,
)
from mas_langgraph.axis.interrogation_mode import (
    DEFAULT_AXIS_INTERROGATION_MODE,
    normalize_axis_interrogation_mode,
)
from mas_langgraph.axis.intervention_guidance import (
    HYBRID_SEMANTIC_CONTEXT_FIELDS,
    _add_boundaries_for_type,
    build_hybrid_intervention_vocabulary,
    build_intervention_candidates,
    format_executable_queries,
    format_hybrid_vocabulary,
    hybrid_opportunity_counts,
    hybrid_replacement_eligible,
    validate_resolved_against_hybrid_vocabulary,
)
from mas_langgraph.axis.models import AxisPromptTemplates, ResolvedQuery
from mas_langgraph.axis.queries import QueryValidationError
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.batch.manifest import generation_config_fingerprint
from mas_langgraph.evaluation.lifecycle import metrics_resume_ok
from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation
from mas_langgraph.replay.checkpoints import BoundarySelector, CheckpointRecord
from mas_langgraph.replay.interventions import (
    ArtifactIntervention,
    InterventionContract,
    InterventionContractRegistry,
)
from mas_langgraph.tests.test_prompt14_loop import _build_fixture


PROMPT_DIR = (
    Path(__file__).resolve().parents[1] / "cases" / "mcq" / "inputs" / "prompts"
)


@dataclass
class _MetaPayload:
    content: str = ""
    lo_id: str = ""
    section_id: str = ""
    tags: tuple = ()
    relevance_tags: tuple = ()
    keywords: tuple = ()
    context_ids: tuple = ()
    source_chunk_ids: tuple = ()
    used_in_atoms: tuple = ()


@dataclass
class _ChunkLikePayload:
    """Mirrors EvidenceChunk: section_id lives in nested metadata."""

    content: str = ""
    metadata: dict = None
    relevance_tags: tuple = ()


class _PromptRecorder:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.replies:
            return self.replies.pop(0)
        return "DONE"


def _mcq_contracts() -> InterventionContractRegistry:
    contracts = InterventionContractRegistry()
    contracts.register(
        InterventionContract(
            artifact_type="EvidenceChunk",
            allowed_operations=("REMOVE", "ADD", "REPLACE"),
            replacement_compatible_types=("EvidenceChunk",),
            allow_external_add=True,
        )
    )
    contracts.register(
        InterventionContract(
            artifact_type="KnowledgeAtom",
            allowed_operations=("REMOVE", "REPLACE"),
            replacement_compatible_types=("KnowledgeAtom",),
            allow_external_add=False,
        )
    )
    contracts.register(
        InterventionContract(
            artifact_type="CandidateMCQ",
            allowed_operations=("REMOVE",),
            allow_external_add=False,
        )
    )
    contracts.register(
        InterventionContract(
            artifact_type="MCQBlueprint",
            allowed_operations=("REMOVE",),
            allow_external_add=False,
        )
    )
    return contracts


def _store_meta(
    reg: ArtifactRegistry,
    logical_id: str,
    *,
    atype: str,
    version: int = 1,
    content: str = "",
    lo_id: str = "",
    section_id: str = "",
    tags: tuple = (),
    relevance_tags: tuple = (),
    keywords: tuple = (),
    superseded: bool = False,
) -> ArtifactEnvelope:
    env = ArtifactEnvelope.create(
        logical_id=logical_id,
        version=version,
        artifact_type=atype,
        payload=_MetaPayload(
            content=content or f"{logical_id}:v{version}",
            lo_id=lo_id,
            section_id=section_id,
            tags=tags,
            relevance_tags=relevance_tags,
            keywords=keywords,
        ),
        branch_id="factual",
        created_in_run="run1",
    )
    if superseded:
        env = ArtifactEnvelope(
            instance_id=env.instance_id,
            ref=env.ref,
            artifact_type=env.artifact_type,
            payload=env.payload,
            branch_id=env.branch_id,
            created_in_run=env.created_in_run,
            superseded=True,
            metadata=dict(env.metadata),
        )
    return reg.store(env)


def _availability_fixture():
    """Several slice chunks/atoms plus compatible extras in the factual registry."""
    explained, reg, occs, rels, idx, _old_contracts, ex = _build_fixture()
    contracts = _mcq_contracts()
    _store_meta(
        reg,
        "chunk_1",
        atype="EvidenceChunk",
        content="slice chunk about k-means",
        section_id="sec_a",
        tags=("kmeans",),
    )
    _store_meta(
        reg,
        "chunk_2",
        atype="EvidenceChunk",
        content="slice chunk about pca",
        section_id="sec_b",
        tags=("pca",),
    )
    _store_meta(
        reg,
        "atom_1",
        atype="KnowledgeAtom",
        content="atom on k-means steps",
        lo_id="lo_kmeans",
        section_id="sec_a",
    )
    _store_meta(
        reg,
        "atom_2",
        atype="KnowledgeAtom",
        content="atom on k-means eval",
        lo_id="lo_kmeans",
        section_id="sec_a",
    )
    _store_meta(
        reg,
        "chunk_out",
        atype="EvidenceChunk",
        content="outside-slice chunk same section",
        section_id="sec_a",
        tags=("kmeans",),
    )
    _store_meta(
        reg,
        "chunk_other",
        atype="EvidenceChunk",
        content="outside-slice other section",
        section_id="sec_z",
        tags=("other",),
    )
    _store_meta(
        reg,
        "atom_out",
        atype="KnowledgeAtom",
        content="outside atom same lo",
        lo_id="lo_kmeans",
        section_id="sec_a",
    )
    _store_meta(
        reg,
        "atom_other",
        atype="KnowledgeAtom",
        content="outside atom other lo",
        lo_id="lo_pca",
        section_id="sec_z",
    )
    old = _store_meta(
        reg,
        "chunk_hist",
        atype="EvidenceChunk",
        version=1,
        content="superseded historical chunk",
        section_id="sec_a",
    )
    _store_meta(
        reg,
        "chunk_hist",
        atype="EvidenceChunk",
        version=2,
        content="latest live historical id",
        section_id="sec_a",
    )
    reg.mark_superseded(old.ref, "factual")
    _store_meta(
        reg,
        "Q_unrelated",
        atype="CandidateMCQ",
        content="unrelated MCQ",
    )

    extra_refs = [
        ArtifactRef("chunk_1", 1),
        ArtifactRef("chunk_2", 1),
        ArtifactRef("atom_1", 1),
        ArtifactRef("atom_2", 1),
    ]
    occ_atom = OccurrenceRecord(
        occurrence_id="run1:factual:atomize#1",
        node_id="atomize",
        occurrence_index=1,
        branch_id="factual",
        run_id="run1",
        execution_index=1,
    )
    occs = list(occs) + [occ_atom]
    idx.add(
        CheckpointRecord(
            alias="c_4",
            thread_id="t1",
            checkpoint_id="ckpt_4",
            superstep=4,
            after_occurrences=("run1:factual:atomize#1",),
            after_occurrence_labels=("atomize#1",),
            parent_checkpoint_id="ckpt_0",
        )
    )
    extra_rels = [
        RunProvenanceRelation(
            producer_occurrence_id="run1:factual:gen#1",
            artifact_ref=ref,
            relation="supporting_evidence",
            consumer_occurrence_id="run1:factual:atomize#1",
        )
        for ref in extra_refs
    ]
    rels = list(rels) + extra_rels

    slice_refs = (
        ArtifactRef("E17", 1),
        ArtifactRef("chunk_1", 1),
        ArtifactRef("chunk_2", 1),
        ArtifactRef("atom_1", 1),
        ArtifactRef("atom_2", 1),
        ArtifactRef("Q12", 1),
        ArtifactRef("Q12", 2),
        ArtifactRef("decision:Q12", 2),
        ArtifactRef("assessment:Q12", 2),
    )
    case_slice = CaseSlice(
        explained_case=explained,
        relevant_artifact_refs=slice_refs,
        relevant_occurrences=tuple(occs),
        relevant_relations=tuple(rels),
    )
    resolver = ArtifactResolver(registry=reg)
    for ref in reg.all_refs():
        resolver.make_available(ref)
    ex._resolvers[("run1", "factual")] = resolver
    slice_catalog = {}
    for ref in slice_refs:
        env = reg.get_by_ref(ref, "factual") or reg.get_by_ref(ref)
        if env is not None:
            slice_catalog[env.instance_id] = env
    _stamp_snapshot_catalogs(ex, "run1", slice_catalog)
    return explained, reg, occs, rels, idx, contracts, ex, case_slice


def _case_slice(explained, occs, rels, extra_refs=()):
    refs = [
        ArtifactRef("E17", 1),
        ArtifactRef("Q12", 1),
        ArtifactRef("Q12", 2),
        ArtifactRef("decision:Q12", 2),
        ArtifactRef("assessment:Q12", 2),
        *list(extra_refs),
    ]
    seen = set()
    unique = []
    for ref in refs:
        if str(ref) in seen:
            continue
        seen.add(str(ref))
        unique.append(ref)
    return CaseSlice(
        explained_case=explained,
        relevant_artifact_refs=tuple(unique),
        relevant_occurrences=tuple(occs),
        relevant_relations=tuple(rels),
    )


def _controller(explained, reg, occs, rels, idx, contracts, ex, **kwargs):
    templates = kwargs.pop("prompt_templates", None)
    if templates is None:
        templates = AxisPromptTemplates(
            system=(PROMPT_DIR / "system.txt").read_text(encoding="utf-8"),
            system_hybrid=(PROMPT_DIR / "system_hybrid.txt").read_text(encoding="utf-8"),
            interrogation=(PROMPT_DIR / "interrogation.txt").read_text(encoding="utf-8"),
            interrogation_hybrid=(PROMPT_DIR / "interrogation_hybrid.txt").read_text(
                encoding="utf-8"
            ),
        )
    kwargs.setdefault("compatibility_mode", False)
    kwargs.setdefault("intervention_alternatives", {})
    controller = AxisExplanationController(
        explained_case=explained,
        registry=reg,
        occurrences=occs,
        provenance_relations=rels,
        checkpoint_index=idx,
        replay_executor=ex,
        contracts=contracts,
        prompt_templates=templates,
        **kwargs,
    )
    return controller


def _stamp_snapshot_catalogs(ex, run_id, catalog, *, by_alias=None):
    """Write artifact_catalog onto synthetic factual snapshots."""
    for snap in getattr(ex, "_factual_histories", {}).get(run_id, []) or []:
        values = dict(getattr(snap, "values", None) or {})
        alias_catalog = None
        if by_alias:
            from mas_langgraph.replay.checkpoints import extract_checkpoint_id

            cid = extract_checkpoint_id(snap)
            index = ex.get_checkpoint_index(run_id)
            rec = next(
                (r for r in index.records if r.checkpoint_id == cid),
                None,
            )
            if rec is not None:
                alias_catalog = by_alias.get(rec.alias)
        values["artifact_catalog"] = dict(
            alias_catalog if alias_catalog is not None else catalog
        )
        snap.values = values


def _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex=None):
    cands = build_intervention_candidates(
        case_slice=case_slice,
        registry=reg,
        contracts=contracts,
        checkpoint_index=idx,
        branch_id="factual",
        all_occurrences=occs,
        intervention_alternatives={},
    )
    return build_hybrid_intervention_vocabulary(
        case_slice=case_slice,
        registry=reg,
        contracts=contracts,
        checkpoint_index=idx,
        branch_id="factual",
        explained_case=explained,
        intervention_candidates=cands,
        intervention_alternatives={},
        all_occurrences=occs,
        replay_executor=ex,
        run_id=getattr(explained, "run_id", "") or "run1",
    ), cands


def _whatif_query(vocab, old, new):
    cand = vocab.candidate_for(old)
    self_bounds = cand.boundaries_for("REPLACE") if cand is not None else ()
    if not self_bounds:
        raise AssertionError(f"no WHATIF boundaries for {old}")
    allowed = {str(r) for r in vocab.replacements_for(old)}
    if str(new) not in allowed:
        raise AssertionError(f"{new} not advertised for {old}: {sorted(allowed)}")
    return f"WHATIF({old}, {new}, {self_bounds[0]})"


def _add_query(vocab, ref):
    allowed = {str(r) for r in vocab.add_options}
    if str(ref) not in allowed:
        raise AssertionError(f"{ref} not advertised as ADD: {sorted(allowed)}")
    bounds = vocab.add_boundaries_for(ref)
    if not bounds:
        raise AssertionError(f"no ADD boundaries advertised for {ref}")
    return f"ADD({ref}, {bounds[0]})"


def _run(controller, question, case_slice, **kwargs):
    controller.slicer.backward_slice = lambda *args, **kw: case_slice
    return controller.run(question, **kwargs)


class TestConstrainedModePreserved(unittest.TestCase):
    def test_default_mode_is_constrained(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            compatibility_mode=True,
        )
        self.assertEqual(controller.axis_interrogation_mode, "constrained")
        self.assertEqual(
            normalize_axis_interrogation_mode(None),
            DEFAULT_AXIS_INTERROGATION_MODE,
        )

    def test_constrained_still_uses_format_executable_queries_and_exact_form_prompt(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        recorder = _PromptRecorder(["DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            max_rounds=1,
            max_attempts=2,
        )
        with patch(
            "mas_langgraph.axis.explainer.format_executable_queries",
            wraps=format_executable_queries,
        ) as mocked:
            _run(controller, "Why was this question selected?", _case_slice(explained, occs, rels))
            self.assertTrue(mocked.called)
        prompt = recorder.prompts[0]
        self.assertIn("Return exactly one of the executable query forms shown below.", prompt)
        self.assertIn("REMOVE(E17:v1,", prompt)
        self.assertNotIn("WHAT(target)", prompt)
        self.assertNotRegex(prompt, r"(?m)^WHAT\(")
        self.assertNotIn("Compatible replacements:", prompt)

    def test_constrained_ordering_and_empty_alternatives_unchanged(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        cands = build_intervention_candidates(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            branch_id="factual",
            all_occurrences=occs,
            intervention_alternatives={},
        )
        text = format_executable_queries(
            candidates=cands,
            explained_case=explained,
            intervention_alternatives={},
            registry=reg,
            branch_id="factual",
        )
        self.assertNotIn("WHATIF(", text)
        lines = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.startswith("Executable") and not ln.startswith("Return")
            and not ln.startswith("Preserve") and ln.strip() != "DONE"
        ]
        self.assertFalse(any(ln.startswith("WHAT(") for ln in lines))
        text2 = format_executable_queries(
            candidates=cands,
            explained_case=explained,
            intervention_alternatives={},
            registry=reg,
            branch_id="factual",
        )
        self.assertEqual(text, text2)
        self.assertGreaterEqual(len(lines), 1)

    def test_explicit_alternatives_still_create_whatif_in_constrained(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        alt = ArtifactRef("chunk_out", 1)
        cands = build_intervention_candidates(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            branch_id="factual",
            all_occurrences=occs,
            intervention_alternatives={"alt": alt},
        )
        text = format_executable_queries(
            candidates=cands,
            explained_case=explained,
            intervention_alternatives={"alt": alt},
            registry=reg,
            branch_id="factual",
        )
        self.assertIn("WHATIF(", text)
        self.assertIn("chunk_out:v1", text)


class TestHybridVocabulary(unittest.TestCase):
    def test_hybrid_discovers_replacements_excludes_self_incompatible_and_superseded(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        cands = build_intervention_candidates(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            branch_id="factual",
            all_occurrences=occs,
            intervention_alternatives={},
        )
        vocab = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            branch_id="factual",
            explained_case=explained,
            intervention_candidates=cands,
            intervention_alternatives={},
            replay_executor=ex,
            run_id="run1",
        )
        chunk1 = ArtifactRef("chunk_1", 1)
        alts = vocab.replacements_for(chunk1)
        self.assertNotIn(chunk1, alts)
        self.assertIn(ArtifactRef("chunk_out", 1), alts)
        self.assertNotIn(ArtifactRef("Q_unrelated", 1), alts)
        self.assertNotIn(ArtifactRef("chunk_hist", 1), alts)
        self.assertIn(ArtifactRef("chunk_hist", 2), alts)
        self.assertNotIn(ArtifactRef("chunk_2", 1), alts)
        self.assertNotIn(ArtifactRef("chunk_other", 1), alts)
        atom_alts = vocab.replacements_for(ArtifactRef("atom_1", 1))
        self.assertNotIn(ArtifactRef("atom_2", 1), atom_alts)
        self.assertIn(ArtifactRef("atom_out", 1), atom_alts)
        self.assertNotIn(ArtifactRef("atom_other", 1), atom_alts)
        self.assertEqual(atom_alts[0], ArtifactRef("atom_out", 1))
        self.assertTrue(vocab.add_options)
        self.assertIn(ArtifactRef("chunk_out", 1), vocab.add_options)
        self.assertNotIn(ArtifactRef("chunk_other", 1), vocab.add_options)
        self.assertNotIn(ArtifactRef("atom_out", 1), vocab.add_options)
        self.assertFalse(vocab.what_refs)

    def test_ranking_is_deterministic(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        cands = build_intervention_candidates(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            branch_id="factual",
            all_occurrences=occs,
        )
        a = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            intervention_candidates=cands,
            replay_executor=ex,
            run_id="run1",
        )
        b = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            intervention_candidates=cands,
            replay_executor=ex,
            run_id="run1",
        )
        self.assertEqual(a.replacements_for(ArtifactRef("chunk_1", 1)), b.replacements_for(ArtifactRef("chunk_1", 1)))
        self.assertEqual(a.add_options, b.add_options)

    def test_add_unavailable_when_no_safe_source(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        contracts = _mcq_contracts()
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=tuple(reg.all_refs("factual")),
            relevant_occurrences=tuple(occs),
            relevant_relations=tuple(rels),
        )
        vocab = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            intervention_alternatives={},
            replay_executor=ex,
            run_id="run1",
        )
        self.assertEqual(vocab.add_options, ())

    def test_hybrid_never_synthesizes_payloads(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        before = {eid: env.payload for eid, env in reg.catalog.items()}
        vocab = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            replay_executor=ex,
            run_id="run1",
        )
        after = {eid: env.payload for eid, env in reg.catalog.items()}
        self.assertEqual(before, after)
        self.assertTrue(vocab.replacement_options)


class TestHybridController(unittest.TestCase):
    def test_hybrid_mode_selectable_and_prompt_is_vocabulary_not_exact_list(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        recorder = _PromptRecorder(["DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            max_rounds=1,
            max_attempts=3,
        )
        with patch(
            "mas_langgraph.axis.explainer.format_executable_queries",
            wraps=format_executable_queries,
        ) as mocked:
            result = _run(
                controller,
                "Why was this question selected?",
                case_slice,
            )
            self.assertFalse(mocked.called)
        self.assertEqual(result.axis_interrogation_mode, "hybrid")
        prompt = recorder.prompts[0]
        self.assertIn("You may compose exactly one query", prompt)
        self.assertNotIn("WHAT(target)", prompt)
        self.assertNotIn("FACTUAL QUERY TARGETS", prompt)
        self.assertNotIn("Return exactly one of the executable query forms shown below.", prompt)
        self.assertNotIn("Copy one executable form exactly", prompt)
        self.assertIn("chunk_1:v1", prompt)
        self.assertIn("REMOVE boundaries:", prompt)
        self.assertIn("WHATIF boundaries:", prompt)
        self.assertIn("Compatible replacements:", prompt)
        self.assertNotRegex(prompt, r"(?m)^REMOVE\(chunk_1:v1,")
        att = result.interrogation_attempts[0]
        self.assertEqual(att.available_what_count, 0)
        self.assertGreater(att.available_remove_count, 0)
        self.assertGreater(att.available_whatif_count, 0)
        self.assertGreater(att.available_add_count, 0)

    def test_availability_fixture_constrained_vs_hybrid(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        cands = build_intervention_candidates(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            branch_id="factual",
            all_occurrences=occs,
            intervention_alternatives={},
        )
        constrained = format_executable_queries(
            candidates=cands,
            explained_case=explained,
            intervention_alternatives={},
            registry=reg,
            branch_id="factual",
        )
        vocab = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            intervention_candidates=cands,
            replay_executor=ex,
            run_id="run1",
        )
        hybrid = format_hybrid_vocabulary(vocabulary=vocab, registry=reg)
        self.assertNotIn("WHATIF(", constrained)
        self.assertIn("WHATIF", hybrid)
        self.assertIn("REMOVE", hybrid)
        self.assertIn("ADDABLE ARTIFACTS", hybrid)
        self.assertFalse(vocab.what_refs)
        self.assertTrue(any("REMOVE" in c.allowed_operations for c in vocab.intervention_candidates))
        self.assertTrue(vocab.add_options)

    def test_valid_remove_accepted_what_rejected(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        recorder = _PromptRecorder(["WHAT(E17:v1)", "REMOVE(E17:v1, before gen#1)", "DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            alias_map={"E17": ArtifactRef("E17", 1)},
            max_rounds=5,
            max_attempts=8,
        )
        result = _run(controller, "Why?", _case_slice(explained, occs, rels))
        ops_executed = [a.parsed_operation for a in result.interrogation_attempts if a.executed]
        self.assertNotIn("WHAT", ops_executed)
        self.assertIn("REMOVE", ops_executed)
        self.assertTrue(
            any(
                a.parsed_operation == "WHAT" and a.validation_status == "vocabulary_error"
                for a in result.interrogation_attempts
            )
        )
        self.assertFalse(result.factual_records)
        self.assertTrue(result.intervention_records)

    def test_valid_whatif_accepted_invalid_replacement_and_boundary_retry(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _cands = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        valid = _whatif_query(vocab, ArtifactRef("chunk_1", 1), ArtifactRef("chunk_out", 1))
        recorder = _PromptRecorder(
            [
                "WHATIF(chunk_1:v1, chunk_hist:v1, before gen#1)",
                "WHATIF(chunk_1:v1, chunk_out:v1, before does_not_exist#9)",
                valid,
                "DONE",
            ]
        )
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            max_rounds=5,
            max_attempts=10,
        )
        result = _run(controller, "Why was this question selected?", case_slice)
        statuses = [a.validation_status for a in result.interrogation_attempts]
        self.assertIn("vocabulary_error", statuses)
        executed = [a for a in result.interrogation_attempts if a.executed]
        self.assertTrue(any(a.parsed_operation == "WHATIF" for a in executed))
        feedback = "\n".join(result.intermediate_texts)
        self.assertIn("replacement", feedback.lower())
        self.assertNotIn("Copy one executable form exactly", feedback)
        self.assertNotIn("Copy one executable query exactly", feedback)

    def test_duplicate_rejected_done_and_budgets(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        recorder = _PromptRecorder(
            ["REMOVE(E17:v1, before gen#1)", "REMOVE(E17:v1, before gen#1)", "DONE"]
        )
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            alias_map={"E17": ArtifactRef("E17", 1)},
            max_rounds=5,
            max_attempts=8,
        )
        result = _run(controller, "Why?", _case_slice(explained, occs, rels))
        statuses = [a.validation_status for a in result.interrogation_attempts]
        self.assertTrue(
            any(s in {"duplicate_error", "not_currently_executable"} for s in statuses)
        )
        self.assertIn(result.termination_reason, {"done", "max_rounds"})
        recorder2 = _PromptRecorder(["INVALID"] * 12)
        controller2 = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder2,
            axis_interrogation_mode="hybrid",
            max_rounds=2,
            max_attempts=3,
        )
        result2 = _run(controller2, "Why?", _case_slice(explained, occs, rels))
        self.assertEqual(result2.termination_reason, "attempt_budget_exhausted")
        self.assertLessEqual(result2.successful_round_count, 2)

    def test_valid_add_when_source_available(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _cands = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        recorder = _PromptRecorder(
            [_add_query(vocab, ArtifactRef("chunk_out", 1)), "DONE"]
        )
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            max_rounds=3,
            max_attempts=6,
        )
        result = _run(controller, "Why?", case_slice)
        executed = [a for a in result.interrogation_attempts if a.executed]
        self.assertTrue(any(a.parsed_operation == "ADD" for a in executed))

    def test_hybrid_uses_same_executor_replay_path(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        recorder = _PromptRecorder(["REMOVE(E17:v1, before gen#1)", "DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            alias_map={"E17": ArtifactRef("E17", 1)},
            max_rounds=3,
            max_attempts=6,
        )
        with patch.object(
            ex, "fork_with_intervention", wraps=ex.fork_with_intervention
        ) as mocked:
            result = _run(controller, "Why?", _case_slice(explained, occs, rels))
            self.assertTrue(mocked.called)
        self.assertTrue(result.intervention_records)

    def test_scripted_hybrid_still_vocabulary_validates(self):
        explained, reg, occs, rels, idx, contracts, ex, _slice = _availability_fixture()
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            axis_interrogation_mode="hybrid",
            max_rounds=3,
            max_attempts=6,
        )
        result = _run(
            controller,
            "Why?",
            _slice,
            scripted_queries=["WHATIF(chunk_1:v1, chunk_hist:v1, before gen#1)", "DONE"],
        )
        statuses = [a.validation_status for a in result.interrogation_attempts]
        self.assertIn("vocabulary_error", statuses)

    def test_vocabulary_validation_rejects_unadvertised_registry_artifact(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            replay_executor=ex,
            run_id="run1",
        )
        rq = ResolvedQuery(
            operation="WHATIF",
            raw_text="WHATIF(chunk_1:v1, Q_unrelated:v1, c_8)",
            target_ref=ArtifactRef("chunk_1", 1),
            replacement_ref=ArtifactRef("Q_unrelated", 1),
            boundary_raw="before gen#1",
            checkpoint_alias="c_8",
        )
        with self.assertRaises(QueryValidationError):
            validate_resolved_against_hybrid_vocabulary(rq, vocab)


class TestHybridPromptConsistency(unittest.TestCase):
    def test_hybrid_system_never_contradicts_discovered_replacements(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer

        templates = AxisPromptTemplates(
            system=(PROMPT_DIR / "system.txt").read_text(encoding="utf-8"),
            system_hybrid=(PROMPT_DIR / "system_hybrid.txt").read_text(encoding="utf-8"),
            factual_context=(PROMPT_DIR / "factual_context.txt").read_text(
                encoding="utf-8"
            ),
            interrogation_hybrid=(PROMPT_DIR / "interrogation_hybrid.txt").read_text(
                encoding="utf-8"
            ),
        )
        recorder = _PromptRecorder(["DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            prompt_templates=templates,
            verbalizer=McqVerbalizer(),
            max_rounds=1,
            max_attempts=2,
        )
        _run(controller, "Why was this question selected?", case_slice)
        prompt = recorder.prompts[0]
        self.assertIn("Compatible replacements:", prompt)
        self.assertIn("chunk_out:v1", prompt)
        self.assertIn("HYBRID INTERVENTION VOCABULARY", prompt)
        self.assertNotIn("CONTROLLED INTERVENTION ALTERNATIVES", prompt)
        self.assertNotIn("(no controlled alternatives)", prompt)
        self.assertNotRegex(prompt, r"(?i)no alternatives exist")
        self.assertNotIn("Return exactly one of the executable query forms shown below.", prompt)

    def test_hybrid_system_fallback_rewrites_constrained_alternatives_rule(self):
        constrained = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
        templates = AxisPromptTemplates(system=constrained)
        hybrid = _render_system(templates, axis_interrogation_mode="hybrid")
        self.assertIn("HYBRID INTERVENTION VOCABULARY", hybrid)
        self.assertNotIn("CONTROLLED INTERVENTION ALTERNATIVES", hybrid)
        rewritten = hybridize_system_prompt(constrained)
        self.assertNotIn("CONTROLLED INTERVENTION ALTERNATIVES", rewritten)
        self.assertIn("CONTROLLED INTERVENTION ALTERNATIVES", constrained)

    def test_hybrid_factual_context_does_not_say_no_alternatives(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer

        templates = AxisPromptTemplates(
            factual_context=(PROMPT_DIR / "factual_context.txt").read_text(
                encoding="utf-8"
            )
        )
        text = _build_factual_context(
            question="Why?",
            explained_case=explained,
            case_slice=case_slice,
            registry=reg,
            checkpoint_index=idx,
            contracts=contracts,
            verbalizer=McqVerbalizer(),
            prompt_templates=templates,
            intervention_alternatives={},
            strict=True,
            axis_interrogation_mode="hybrid",
        )
        self.assertNotIn("(no controlled alternatives)", text)
        self.assertIn("HYBRID INTERVENTION VOCABULARY", text)
        self.assertIn("CONTRACT-LEVEL INTERVENTION CAPABILITIES", text)
        self.assertIn("structural contract permissions only", text.lower())
        constrained = _build_factual_context(
            question="Why?",
            explained_case=explained,
            case_slice=case_slice,
            registry=reg,
            checkpoint_index=idx,
            contracts=contracts,
            verbalizer=McqVerbalizer(),
            prompt_templates=templates,
            intervention_alternatives={},
            strict=True,
            axis_interrogation_mode="constrained",
        )
        self.assertIn("(no controlled alternatives)", constrained)
        self.assertNotIn("CONTRACT-LEVEL INTERVENTION CAPABILITIES", constrained)


class TestHybridSemanticFilter(unittest.TestCase):
    def test_whatif_not_advertised_without_compatible_replacement(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _cands = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        text = format_hybrid_vocabulary(vocabulary=vocab, registry=reg)
        self.assertEqual(vocab.replacements_for(ArtifactRef("chunk_2", 1)), ())
        self.assertEqual(vocab.replacements_for(ArtifactRef("E17", 1)), ())
        if "EvidenceChunk chunk_2:v1" in text:
            rest = text.split("EvidenceChunk chunk_2:v1", 1)[1]
            until_next = rest.split("\n\nKnowledgeAtom", 1)[0]
            until_next = until_next.split("\n\nEvidenceChunk", 1)[0]
            until_next = until_next.split("\n\nADDABLE", 1)[0]
            self.assertNotIn("WHATIF", until_next)
        self.assertNotRegex(
            text,
            r"EvidenceChunk chunk_2:v1\n\nAllowed operations:\n(?:- REMOVE\n)?- WHATIF",
        )

    def test_knowledge_atom_eligibility_and_no_same_type_fallback(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        old = reg.get_by_ref(ArtifactRef("atom_1", 1), "factual")
        same_lo = reg.get_by_ref(ArtifactRef("atom_out", 1), "factual")
        other_lo = reg.get_by_ref(ArtifactRef("atom_other", 1), "factual")
        in_slice = reg.get_by_ref(ArtifactRef("atom_2", 1), "factual")
        self.assertTrue(hybrid_replacement_eligible(old, same_lo))
        self.assertFalse(hybrid_replacement_eligible(old, other_lo))
        self.assertTrue(hybrid_replacement_eligible(old, in_slice))
        vocab, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        alts = vocab.replacements_for(ArtifactRef("atom_1", 1))
        self.assertNotIn(ArtifactRef("atom_2", 1), alts)
        self.assertNotIn(ArtifactRef("atom_other", 1), alts)

    def test_evidence_chunk_nested_metadata_and_tag_overlap(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        nested = ArtifactEnvelope.create(
            logical_id="chunk_nested",
            version=1,
            artifact_type="EvidenceChunk",
            payload=_ChunkLikePayload(
                content="nested section match",
                metadata={"section_id": "sec_a"},
                relevance_tags=("kmeans",),
            ),
            branch_id="factual",
            created_in_run="run1",
        )
        unrelated = ArtifactEnvelope.create(
            logical_id="chunk_unrelated_meta",
            version=1,
            artifact_type="EvidenceChunk",
            payload=_ChunkLikePayload(
                content="other section",
                metadata={"section_id": "sec_z"},
                relevance_tags=("other",),
            ),
            branch_id="factual",
            created_in_run="run1",
        )
        old = reg.get_by_ref(ArtifactRef("chunk_1", 1), "factual")
        self.assertTrue(hybrid_replacement_eligible(old, nested))
        self.assertFalse(hybrid_replacement_eligible(old, unrelated))
        self.assertIn("relevance_tags", HYBRID_SEMANTIC_CONTEXT_FIELDS)
        self.assertIn("tags", HYBRID_SEMANTIC_CONTEXT_FIELDS)
        self.assertIn("keywords", HYBRID_SEMANTIC_CONTEXT_FIELDS)
        self.assertNotIn("relevance", HYBRID_SEMANTIC_CONTEXT_FIELDS)
        self.assertNotIn("relevance_score", HYBRID_SEMANTIC_CONTEXT_FIELDS)

    def test_automatic_in_slice_replacements_excluded_explicit_exception(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        self.assertNotIn(
            ArtifactRef("atom_2", 1), vocab.replacements_for(ArtifactRef("atom_1", 1))
        )
        self.assertNotIn(
            ArtifactRef("chunk_2", 1), vocab.replacements_for(ArtifactRef("chunk_1", 1))
        )
        explicit = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            intervention_alternatives={"keep": ArtifactRef("atom_2", 1)},
            all_occurrences=occs,
            replay_executor=ex,
            run_id="run1",
        )
        self.assertIn(
            ArtifactRef("atom_2", 1),
            explicit.replacements_for(ArtifactRef("atom_1", 1)),
        )

    def test_add_sources_are_contextually_related(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        self.assertIn(ArtifactRef("chunk_out", 1), vocab.add_options)
        self.assertNotIn(ArtifactRef("chunk_other", 1), vocab.add_options)
        slice_keys = {str(r) for r in case_slice.relevant_artifact_refs}
        for ref in vocab.add_options:
            self.assertNotIn(str(ref), slice_keys)

    def test_add_boundaries_are_type_appropriate_not_first_slice_artifact(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        occ_verify = OccurrenceRecord(
            occurrence_id="run1:factual:verify#1",
            node_id="verify",
            occurrence_index=1,
            branch_id="factual",
            run_id="run1",
            execution_index=9,
        )
        idx.add(
            CheckpointRecord(
                alias="c_9",
                thread_id="t1",
                checkpoint_id="ckpt_9",
                superstep=9,
                after_occurrences=("run1:factual:verify#1",),
                after_occurrence_labels=("verify#1",),
                parent_checkpoint_id="ckpt_8",
            )
        )
        q_rel = RunProvenanceRelation(
            producer_occurrence_id="run1:factual:gen#1",
            artifact_ref=ArtifactRef("Q12", 2),
            relation="subject",
            consumer_occurrence_id="run1:factual:verify#1",
        )
        chunk_consumer_rels = [
            r
            for r in rels
            if getattr(r, "artifact_ref", None)
            in {
                ArtifactRef("chunk_1", 1),
                ArtifactRef("chunk_2", 1),
                ArtifactRef("atom_1", 1),
                ArtifactRef("atom_2", 1),
            }
        ]
        first_q12 = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=(
                ArtifactRef("Q12", 2),
                ArtifactRef("chunk_1", 1),
                ArtifactRef("E17", 1),
            ),
            relevant_occurrences=tuple(list(occs) + [occ_verify]),
            relevant_relations=tuple(list(chunk_consumer_rels) + [q_rel]),
        )
        vocab = build_hybrid_intervention_vocabulary(
            case_slice=first_q12,
            registry=reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            all_occurrences=tuple(list(occs) + [occ_verify]),
            replay_executor=ex,
            run_id="run1",
        )
        add_text = " ".join(vocab.add_boundaries).lower()
        self.assertTrue(vocab.add_boundaries)
        self.assertIn("atomize", add_text)
        self.assertNotIn("verify", add_text)
        self.assertNotIn("outline", add_text)
        self.assertNotIn("retrieve", add_text)
        self.assertNotIn("curate", add_text)
        q12_bounds = _add_boundaries_for_type(
            artifact_type="CandidateMCQ",
            case_slice=first_q12,
            checkpoint_index=idx,
            all_occurrences=tuple(list(occs) + [occ_verify]),
            registry=reg,
            branch_id="factual",
        )
        chunk_bounds = _add_boundaries_for_type(
            artifact_type="EvidenceChunk",
            case_slice=first_q12,
            checkpoint_index=idx,
            all_occurrences=tuple(list(occs) + [occ_verify]),
            registry=reg,
            branch_id="factual",
        )
        self.assertTrue(chunk_bounds)
        self.assertTrue(any("atomize" in b for b in chunk_bounds))
        self.assertFalse(any("verify" in b for b in chunk_bounds))
        self.assertTrue(any("verify" in b for b in q12_bounds))
        self.assertFalse(any("atomize" in b for b in q12_bounds))
        self.assertNotEqual(tuple(chunk_bounds), tuple(q12_bounds))

    def test_opportunity_counts_match_conservative_vocabulary(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        counts = hybrid_opportunity_counts(vocab)
        expected_whatif = 0
        for cand in vocab.intervention_candidates:
            if "REPLACE" not in cand.allowed_operations:
                continue
            alts = vocab.replacements_for(cand.ref)
            bounds = cand.boundaries_for("REPLACE")
            if not alts or not bounds:
                continue
            expected_whatif += len(alts) * len(bounds)
        expected_add = sum(
            len(vocab.add_boundaries_for(ref)) for ref in vocab.add_options
        )
        self.assertEqual(counts["whatif"], expected_whatif)
        self.assertEqual(counts["add"], expected_add)
        self.assertGreater(counts["whatif"], 0)
        self.assertGreater(counts["add"], 0)
        self.assertEqual(vocab.replacements_for(ArtifactRef("chunk_2", 1)), ())

    def test_filter_audit_example(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        accepted = vocab.replacements_for(ArtifactRef("chunk_1", 1))
        self.assertIn(ArtifactRef("chunk_out", 1), accepted)
        self.assertIn(ArtifactRef("chunk_hist", 2), accepted)
        self.assertNotIn(ArtifactRef("chunk_other", 1), accepted)
        self.assertNotIn(ArtifactRef("chunk_2", 1), accepted)


class TestHybridResolverReplay(unittest.TestCase):
    """Unmocked intervention/resolver replay tests (compiled_graph=None fixture path)."""

    def test_unmocked_hybrid_whatif_resolver_replay(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        old = ArtifactRef("chunk_1", 1)
        new = ArtifactRef("chunk_out", 1)
        query = _whatif_query(vocab, old, new)
        recorder = _PromptRecorder([query, "DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            max_rounds=3,
            max_attempts=6,
        )
        with patch.object(
            ex, "fork_with_intervention", wraps=ex.fork_with_intervention
        ) as mocked:
            result = _run(controller, "Why was this question selected?", case_slice)
            self.assertTrue(mocked.called)
            args, kwargs = mocked.call_args
            intervention = kwargs.get("intervention")
            if intervention is None:
                intervention = args[1] if len(args) > 1 else args[0]
        self.assertIsInstance(intervention, ArtifactIntervention)
        self.assertEqual(intervention.operation, "REPLACE")
        replaced_old, replaced_new = intervention.artifact
        self.assertEqual(str(replaced_old), str(old))
        self.assertEqual(str(replaced_new), str(new))
        recs = [r for r in result.intervention_records if r.resolved.operation == "WHATIF"]
        self.assertTrue(recs)
        rec = recs[0]
        self.assertFalse(rec.error)
        self.assertIsNotNone(rec.counterfactual_result)
        resolver = ex.get_resolver(rec.counterfactual_result.branch_id)
        self.assertEqual(resolver.resolve_ref(old), new)
        self.assertTrue(resolver.is_available(new))
        self.assertTrue(rec.comparison)
        self.assertTrue(rec.comparison.get("resolver_resolves_old_to_new"))
        self.assertTrue(result.replay_comparisons)

    def test_unmocked_hybrid_add_resolver_replay(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        added = ArtifactRef("chunk_out", 1)
        self.assertNotIn(str(added), {str(r) for r in case_slice.relevant_artifact_refs})
        self.assertIsNotNone(reg.get_by_ref(added, "factual"))
        query = _add_query(vocab, added)
        self.assertTrue(any("atomize" in b for b in vocab.add_boundaries_for(added)))
        recorder = _PromptRecorder([query, "DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            max_rounds=3,
            max_attempts=6,
        )
        with patch.object(
            ex, "fork_with_intervention", wraps=ex.fork_with_intervention
        ) as mocked:
            result = _run(controller, "Why was this question selected?", case_slice)
            self.assertTrue(mocked.called)
            args, kwargs = mocked.call_args
            intervention = kwargs.get("intervention")
            if intervention is None:
                intervention = args[1] if len(args) > 1 else args[0]
        self.assertIsInstance(intervention, ArtifactIntervention)
        self.assertEqual(intervention.operation, "ADD")
        self.assertEqual(str(intervention.artifact), str(added))
        recs = [r for r in result.intervention_records if r.resolved.operation == "ADD"]
        self.assertTrue(recs)
        rec = recs[0]
        self.assertFalse(rec.error)
        self.assertIsNotNone(rec.counterfactual_result)
        resolver = ex.get_resolver(rec.counterfactual_result.branch_id)
        self.assertTrue(resolver.is_available(added))
        self.assertTrue(rec.comparison)
        self.assertTrue(rec.comparison.get("target_available_in_counterfactual"))
        self.assertTrue(result.replay_comparisons)


class TestHybridAddCheckpointValidity(unittest.TestCase):
    def test_outside_slice_but_present_at_checkpoint_rejected(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        existing = ArtifactRef("chunk_out", 1)
        existing_env = reg.get_by_ref(existing, "factual")
        self.assertIsNotNone(existing_env)
        self.assertNotIn(str(existing), {str(r) for r in case_slice.relevant_artifact_refs})
        catalog = {}
        for ref in case_slice.relevant_artifact_refs:
            env = reg.get_by_ref(ref, "factual") or reg.get_by_ref(ref)
            if env is not None:
                catalog[env.instance_id] = env
        catalog[existing_env.instance_id] = existing_env
        _stamp_snapshot_catalogs(ex, "run1", catalog)
        _store_meta(
            reg,
            "chunk_absent",
            atype="EvidenceChunk",
            content="absent extra same section",
            section_id="sec_a",
            tags=("kmeans",),
        )
        vocab, _ = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        add_ids = {str(r) for r in vocab.add_options}
        self.assertNotIn(str(existing), add_ids)
        self.assertIn("chunk_absent:v1", add_ids)
        self.assertTrue(vocab.add_boundaries_for(ArtifactRef("chunk_absent", 1)))
        self.assertEqual(vocab.add_boundaries_for(existing), ())
        counts = hybrid_opportunity_counts(vocab)
        expected_add = sum(
            len(vocab.add_boundaries_for(ref)) for ref in vocab.add_options
        )
        self.assertEqual(counts["add"], expected_add)
        self.assertGreater(counts["add"], 0)
        add_section = format_hybrid_vocabulary(vocabulary=vocab, registry=reg)
        add_section = add_section.split("ADDABLE ARTIFACTS", 1)[-1]
        self.assertIn("chunk_absent:v1", add_section)
        self.assertIn("Valid ADD boundaries:", add_section)
        self.assertNotIn("chunk_out:v1", add_section)

    def test_add_boundaries_differ_by_checkpoint_presence(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        x_ref = ArtifactRef("chunk_out", 1)
        x_env = reg.get_by_ref(x_ref, "factual")
        occ_plan = OccurrenceRecord(
            occurrence_id="run1:factual:plan_mcq#1",
            node_id="plan_mcq",
            occurrence_index=1,
            branch_id="factual",
            run_id="run1",
            execution_index=2,
        )
        idx.add(
            CheckpointRecord(
                alias="c_7",
                thread_id="t1",
                checkpoint_id="ckpt_7",
                superstep=6,
                parent_checkpoint_id="ckpt_0",
            )
        )
        idx.add(
            CheckpointRecord(
                alias="c_plan",
                thread_id="t1",
                checkpoint_id="ckpt_plan",
                superstep=7,
                after_occurrences=("run1:factual:plan_mcq#1",),
                after_occurrence_labels=("plan_mcq#1",),
                parent_checkpoint_id="ckpt_7",
            )
        )
        rel_plan = RunProvenanceRelation(
            producer_occurrence_id="run1:factual:gen#1",
            artifact_ref=ArtifactRef("chunk_1", 1),
            relation="supporting_evidence",
            consumer_occurrence_id="run1:factual:plan_mcq#1",
        )
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=case_slice.relevant_artifact_refs,
            relevant_occurrences=tuple(list(occs) + [occ_plan]),
            relevant_relations=tuple(list(rels) + [rel_plan]),
        )
        slice_catalog = {}
        for ref in case_slice.relevant_artifact_refs:
            env = reg.get_by_ref(ref, "factual") or reg.get_by_ref(ref)
            if env is not None:
                slice_catalog[env.instance_id] = env
        catalog_with_x = dict(slice_catalog)
        catalog_with_x[x_env.instance_id] = x_env
        _stamp_snapshot_catalogs(
            ex, "run1", slice_catalog, by_alias={"c_7": catalog_with_x}
        )
        idx.enrich_from_occurrences(list(occs) + [occ_plan])
        vocab, _ = _hybrid_vocab(
            explained, reg, list(occs) + [occ_plan], idx, contracts, case_slice, ex
        )
        self.assertIn(x_ref, vocab.add_options)
        type_bounds = _add_boundaries_for_type(
            artifact_type="EvidenceChunk",
            case_slice=case_slice,
            checkpoint_index=idx,
            all_occurrences=list(occs) + [occ_plan],
            registry=reg,
            branch_id="factual",
        )
        self.assertIn("before atomize#1", type_bounds)
        self.assertIn("before plan_mcq#1", type_bounds)
        bounds = vocab.add_boundaries_for(x_ref)
        self.assertIn("before atomize#1", bounds)
        self.assertNotIn("before plan_mcq#1", bounds)
        counts = hybrid_opportunity_counts(vocab)
        self.assertEqual(
            counts["add"],
            sum(len(vocab.add_boundaries_for(ref)) for ref in vocab.add_options),
        )
        rq = ResolvedQuery(
            operation="ADD",
            raw_text="ADD(chunk_out:v1, before plan_mcq#1)",
            target_ref=x_ref,
            boundary_raw="before plan_mcq#1",
            checkpoint_alias="c_7",
        )
        with self.assertRaises(QueryValidationError):
            validate_resolved_against_hybrid_vocabulary(rq, vocab)


class TestHybridCompiledGraphReplay(unittest.TestCase):
    """Compiled LangGraph integration tests (compiled_graph is not None)."""

    def _contracts(self):
        return _mcq_contracts()

    def _explained(self, run_id, ref):
        from mas_langgraph.axis.models import ExplainedCase

        return ExplainedCase(
            run_id=run_id,
            artifact_ref=ref,
            branch_id="factual",
            decision_ref=ArtifactRef("decision:out", 1),
            outcome="accept",
        )

    def test_compiled_langgraph_whatif_changes_downstream(self):
        from typing import Annotated, Any, TypedDict
        import operator

        from langgraph.graph import END, START, StateGraph

        from mas_langgraph.artifacts.channel_state import (
            dump_artifact_channels,
            empty_artifact_channels,
            load_artifact_channels,
        )
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.replay.executor import LangGraphReplayExecutor

        run_id = "run_hyb_whatif"
        old = ArtifactRef("src", 1)
        new = ArtifactRef("alt", 1)
        contracts = self._contracts()

        def _payload(text):
            return _MetaPayload(content=text, section_id="sec_a", tags=("kmeans",))

        fact_reg = ArtifactRegistry()
        old_env = ArtifactEnvelope.create(
            logical_id="src", version=1, artifact_type="EvidenceChunk",
            payload=_payload("R1"), branch_id="factual", created_in_run=run_id,
        )
        new_env = ArtifactEnvelope.create(
            logical_id="alt", version=1, artifact_type="EvidenceChunk",
            payload=_payload("R2"), branch_id="factual", created_in_run=run_id,
        )
        fact_reg.store(old_env)
        int_reg = ArtifactRegistry()
        int_reg.store(new_env)

        class S(TypedDict, total=False):
            result_text: str
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
            provenance_node_counts: dict
            provenance_execution_index: int

        def source(state: S) -> dict:
            occ = OccurrenceRecord(
                occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:source#1",
                node_id="source",
                occurrence_index=1,
                branch_id=str(state.get("branch_id") or "factual"),
                run_id=run_id,
                execution_index=1,
                produced_artifacts=(old,),
            )
            rel = RunProvenanceRelation(
                producer_occurrence_id=occ.occurrence_id,
                artifact_ref=old,
                relation="supporting_evidence",
                consumer_occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:transform#1",
            )
            return {
                "occurrence_records": [occ],
                "run_provenance_relations": [rel],
                "path": ["source"],
            }

        def transform(state: S) -> dict:
            _reg, resolver = load_artifact_channels(state)
            env = resolver.get(old)
            text = str(getattr(getattr(env, "payload", None), "content", "") or "")
            occ = OccurrenceRecord(
                occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:transform#1",
                node_id="transform",
                occurrence_index=1,
                branch_id=str(state.get("branch_id") or "factual"),
                run_id=run_id,
                execution_index=2,
            )
            return {
                "result_text": text,
                "occurrence_records": [occ],
                "path": ["transform"],
            }

        g = StateGraph(S)
        g.add_node("source", source)
        g.add_node("transform", transform)
        g.add_edge(START, "source")
        g.add_edge("source", "transform")
        g.add_edge("transform", END)
        app = g.compile(checkpointer=default_factual_checkpointer())
        self.assertIsNotNone(app)

        resolver = ArtifactResolver(registry=fact_reg)
        resolver.make_available(old)
        initial = {
            "result_text": "",
            "path": [],
            "occurrence_records": [],
            "run_provenance_relations": [],
            **empty_artifact_channels(),
            **dump_artifact_channels(fact_reg, resolver),
        }
        ex = LangGraphReplayExecutor(compiled_graph=app)
        ex.contracts = contracts
        factual = ex.run_factual(initial, run_id=run_id, runtime=WorkflowRuntime())
        self.assertIsNone(factual.error, msg=factual.error)
        self.assertEqual(factual.final_state.get("result_text"), "R1")
        self.assertIsNotNone(ex.compiled_graph)

        occs = list(factual.final_state.get("occurrence_records") or [])
        rels = list(factual.final_state.get("run_provenance_relations") or [])
        explained = self._explained(run_id, ArtifactRef("out", 1))
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=(old,),
            relevant_occurrences=tuple(occs),
            relevant_relations=tuple(rels),
        )
        idx = factual.checkpoint_index
        vocab = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=fact_reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            intervention_alternatives={"alt": new},
            intervention_registry=int_reg,
            all_occurrences=occs,
            replay_executor=ex,
            run_id=run_id,
        )
        self.assertTrue(vocab.replacements_for(old), msg=str(vocab.replacement_options))
        cand = vocab.candidate_for(old)
        self.assertIsNotNone(cand)
        before_bounds = [
            b
            for b in cand.boundaries_for("REPLACE")
            if str(b).lower().startswith("before ")
        ]
        self.assertTrue(before_bounds, msg=cand.boundaries_for("REPLACE"))
        query = f"WHATIF({old}, {new}, {before_bounds[0]})"
        recorder = _PromptRecorder([query, "DONE"])
        controller = _controller(
            explained, fact_reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            intervention_alternatives={"alt": new},
            intervention_registry=int_reg,
            factual_run_result=factual,
            max_rounds=3,
            max_attempts=6,
        )
        self.assertIsNotNone(ex.compiled_graph)
        with patch.object(
            ex, "_fixture_fork_intervention", wraps=ex._fixture_fork_intervention
        ) as fixture_path:
            result = _run(controller, "Why?", case_slice)
            self.assertFalse(fixture_path.called)
        self.assertIsNotNone(ex.compiled_graph)
        recs = [r for r in result.intervention_records if r.resolved.operation == "WHATIF"]
        self.assertTrue(recs)
        rec = recs[0]
        self.assertFalse(rec.error, msg=rec.error)
        cf_state = rec.counterfactual_result.final_state
        self.assertEqual(cf_state.get("result_text"), "R2")
        self.assertNotEqual(factual.final_state.get("result_text"), cf_state.get("result_text"))
        self.assertTrue(result.replay_comparisons)
        self.assertTrue("transform" in (cf_state.get("path") or []))

    def test_compiled_langgraph_add_changes_downstream(self):
        from typing import Annotated, Any, TypedDict
        import operator

        from langgraph.graph import END, START, StateGraph

        from mas_langgraph.artifacts.channel_state import (
            dump_artifact_channels,
            empty_artifact_channels,
            load_artifact_channels,
        )
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.replay.branch_artifacts import artifact_exists_at_checkpoint
        from mas_langgraph.replay.executor import LangGraphReplayExecutor

        run_id = "run_hyb_add"
        a_ref = ArtifactRef("chunk_A", 1)
        b_ref = ArtifactRef("chunk_B", 1)
        contracts = self._contracts()

        def _payload(text, section="sec_a"):
            return _MetaPayload(content=text, section_id=section, tags=("kmeans",))

        fact_reg = ArtifactRegistry()
        a_env = ArtifactEnvelope.create(
            logical_id="chunk_A", version=1, artifact_type="EvidenceChunk",
            payload=_payload("A"), branch_id="factual", created_in_run=run_id,
        )
        b_env = ArtifactEnvelope.create(
            logical_id="chunk_B", version=1, artifact_type="EvidenceChunk",
            payload=_payload("B"), branch_id="factual", created_in_run=run_id,
        )
        fact_reg.store(a_env)
        int_reg = ArtifactRegistry()
        int_reg.store(b_env)

        class S(TypedDict, total=False):
            chunk_count: int
            result_text: str
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
            provenance_node_counts: dict
            provenance_execution_index: int

        def source(state: S) -> dict:
            occ = OccurrenceRecord(
                occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:source#1",
                node_id="source",
                occurrence_index=1,
                branch_id=str(state.get("branch_id") or "factual"),
                run_id=run_id,
                execution_index=1,
                produced_artifacts=(a_ref,),
            )
            rel = RunProvenanceRelation(
                producer_occurrence_id=occ.occurrence_id,
                artifact_ref=a_ref,
                relation="supporting_evidence",
                consumer_occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:consume#1",
            )
            return {
                "occurrence_records": [occ],
                "run_provenance_relations": [rel],
                "path": ["source"],
            }

        def consume(state: S) -> dict:
            _reg, resolver = load_artifact_channels(state)
            n = 0
            seen = set()
            for env in (_reg.catalog or {}).values():
                ref = getattr(env, "ref", None)
                if ref is None or ref in seen:
                    continue
                if str(getattr(env, "artifact_type", "")) != "EvidenceChunk":
                    continue
                if resolver.is_available(ref):
                    seen.add(ref)
                    n += 1
            occ = OccurrenceRecord(
                occurrence_id=f"{run_id}:{state.get('branch_id') or 'factual'}:consume#1",
                node_id="consume",
                occurrence_index=1,
                branch_id=str(state.get("branch_id") or "factual"),
                run_id=run_id,
                execution_index=2,
            )
            return {"chunk_count": n, "occurrence_records": [occ], "path": ["consume"]}

        g = StateGraph(S)
        g.add_node("source", source)
        g.add_node("consume", consume)
        g.add_edge(START, "source")
        g.add_edge("source", "consume")
        g.add_edge("consume", END)
        app = g.compile(checkpointer=default_factual_checkpointer())
        self.assertIsNotNone(app)

        resolver = ArtifactResolver(registry=fact_reg)
        resolver.make_available(a_ref)
        initial = {
            "chunk_count": 0,
            "path": [],
            "occurrence_records": [],
            "run_provenance_relations": [],
            **empty_artifact_channels(),
            **dump_artifact_channels(fact_reg, resolver),
        }
        ex = LangGraphReplayExecutor(compiled_graph=app)
        ex.contracts = contracts
        factual = ex.run_factual(initial, run_id=run_id, runtime=WorkflowRuntime())
        self.assertIsNone(factual.error, msg=factual.error)
        self.assertEqual(factual.final_state.get("chunk_count"), 1)
        self.assertIsNotNone(ex.compiled_graph)

        occs = list(factual.final_state.get("occurrence_records") or [])
        rels = list(factual.final_state.get("run_provenance_relations") or [])
        explained = self._explained(run_id, ArtifactRef("out", 1))
        case_slice = CaseSlice(
            explained_case=explained,
            relevant_artifact_refs=(a_ref,),
            relevant_occurrences=tuple(occs),
            relevant_relations=tuple(rels),
        )
        idx = factual.checkpoint_index
        origin = idx.resolve(BoundarySelector.before_occurrence("consume", 1))
        origin_snap = ex.get_checkpoint_snapshot(
            run_id,
            BoundarySelector.by_alias(origin.alias),
        )
        self.assertFalse(
            artifact_exists_at_checkpoint(origin_snap.values, b_ref)
        )
        vocab = build_hybrid_intervention_vocabulary(
            case_slice=case_slice,
            registry=fact_reg,
            contracts=contracts,
            checkpoint_index=idx,
            explained_case=explained,
            intervention_alternatives={"b": b_ref},
            intervention_registry=int_reg,
            all_occurrences=occs,
            replay_executor=ex,
            run_id=run_id,
        )
        self.assertIn(b_ref, vocab.add_options)
        query = _add_query(vocab, b_ref)
        recorder = _PromptRecorder([query, "DONE"])
        controller = _controller(
            explained, fact_reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            intervention_alternatives={"b": b_ref},
            intervention_registry=int_reg,
            factual_run_result=factual,
            max_rounds=3,
            max_attempts=6,
        )
        self.assertIsNotNone(ex.compiled_graph)
        with patch.object(
            ex, "_fixture_fork_intervention", wraps=ex._fixture_fork_intervention
        ) as fixture_path:
            result = _run(controller, "Why?", case_slice)
            self.assertFalse(fixture_path.called)
        self.assertIsNotNone(ex.compiled_graph)
        recs = [r for r in result.intervention_records if r.resolved.operation == "ADD"]
        self.assertTrue(recs)
        rec = recs[0]
        self.assertFalse(rec.error, msg=rec.error)
        cf_state = rec.counterfactual_result.final_state
        self.assertEqual(cf_state.get("chunk_count"), 2)
        self.assertNotEqual(factual.final_state.get("chunk_count"), cf_state.get("chunk_count"))
        self.assertTrue(result.replay_comparisons)
        self.assertIn("consume", cf_state.get("path") or [])


class TestModePersistenceAndCli(unittest.TestCase):
    def test_fingerprints_differ_and_resume_does_not_mix_modes(self):
        base = {
            "llm_model": "gpt-4.1-mini-2025-04-14",
            "model_type": "openai",
            "temperature": 0.0,
            "seed": 42,
            "max_rounds": 5,
            "max_questions": 10,
            "question": "Why was this question selected?",
            "orchestrator": "langgraph",
        }
        constrained = generation_config_fingerprint({**base, "axis_interrogation_mode": "constrained"})
        hybrid = generation_config_fingerprint({**base, "axis_interrogation_mode": "hybrid"})
        self.assertEqual(constrained["axis_interrogation_mode"], "constrained")
        self.assertEqual(hybrid["axis_interrogation_mode"], "hybrid")
        self.assertEqual(constrained["axis_interrogation_policy"], "counterfactual_only")
        self.assertEqual(hybrid["axis_interrogation_policy"], "counterfactual_only")
        self.assertNotEqual(constrained, hybrid)
        diag = generation_config_fingerprint(
            {**base, "axis_interrogation_mode": "hybrid", "diagnose_hybrid_add": True}
        )
        self.assertNotIn("diagnose_hybrid_add", diag)
        self.assertEqual(diag, hybrid)
        old_what_enabled = {k: v for k, v in hybrid.items() if k != "axis_interrogation_policy"}
        self.assertNotEqual(hybrid, old_what_enabled)
        stored = {
            "lecturebank_id": "1",
            "document_id": "lecturebank_1",
            "eval_status": "success",
            "source_hash": "abc",
            "rag_config": {"embedding_model": "x"},
            "generation_config": constrained,
            "metrics_persisted": True,
            "document_metrics": {"lecturebank_id": "1"},
        }
        collected = [{"lecturebank_id": "1", "document_metrics": {"lecturebank_id": "1"}}]
        self.assertTrue(
            metrics_resume_ok(
                lecturebank_id="1",
                stored=stored,
                collected=collected,
                rag_fp={"embedding_model": "x"},
                gen_fp=constrained,
                source_hash="abc",
            )
        )
        self.assertFalse(
            metrics_resume_ok(
                lecturebank_id="1",
                stored=stored,
                collected=collected,
                rag_fp={"embedding_model": "x"},
                gen_fp=hybrid,
                source_hash="abc",
            )
        )
        old_stored = dict(stored)
        old_stored["generation_config"] = old_what_enabled
        self.assertFalse(
            metrics_resume_ok(
                lecturebank_id="1",
                stored=old_stored,
                collected=collected,
                rag_fp={"embedding_model": "x"},
                gen_fp=hybrid,
                source_hash="abc",
            )
        )

    def test_batch_cli_accepts_hybrid(self):
        from mas_langgraph.batch.runner import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            ["--input-dir", "x", "--axis-interrogation-mode", "hybrid"]
        )
        self.assertEqual(args.axis_interrogation_mode, "hybrid")
        args2 = parser.parse_args(["--input-dir", "x"])
        self.assertEqual(args2.axis_interrogation_mode, "constrained")

    def test_mode_recorded_on_result(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            axis_interrogation_mode="hybrid",
        )
        result = controller.run("Why?", scripted_queries=["DONE"])
        self.assertEqual(result.axis_interrogation_mode, "hybrid")
        self.assertEqual(result.interrogation_attempts[0].axis_interrogation_mode, "hybrid")


if __name__ == "__main__":
    unittest.main()
