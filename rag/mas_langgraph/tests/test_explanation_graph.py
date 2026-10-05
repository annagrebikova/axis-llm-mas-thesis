"""Explanation-graph construction from exported case JSON."""

from __future__ import annotations

import json
import unittest

from mas_langgraph.visualization import (
    VisualizationDataError,
    build_explanation_graph,
    render_dot,
    render_mermaid,
    render_tikz,
)


SPECIAL = 'quotes "brackets" [braces] {pipes} | underscores_ % & # \nnewline'
LONG_CHUNK = (
    SPECIAL + " Evidence chunk full text about clustering. " + ("content-token " * 80)
)
LONG_ATOM = (
    "Knowledge atom full text about k-means assignment. " + ("content-token " * 80)
)


def _ref(logical_id: str, version: int) -> dict:
    return {
        "logical_id": logical_id,
        "version": version,
        "key": f"{logical_id}:v{version}",
    }


def _art(logical_id, version, artifact_type, content, produced_by=None):
    row = {
        "ref": _ref(logical_id, version),
        "artifact_type": artifact_type,
        "branch_id": "factual",
        "content": content,
    }
    if produced_by:
        row["produced_by_occurrence_id"] = produced_by
    return row


def _occ(oid, node, exec_i, produced=(), bindings=(), idx=None):
    return {
        "occurrence_id": oid,
        "node_id": node,
        "occurrence_index": idx if idx is not None else exec_i,
        "execution_index": exec_i,
        "superstep": 3,
        "branch_id": "factual",
        "run_id": "run_viz",
        "checkpoint_before": f"before-{exec_i}",
        "checkpoint_after": f"after-{exec_i}",
        "error": None,
        "produced_artifacts": [_ref(*p) if not isinstance(p, dict) else p for p in produced],
        "input_bindings": list(bindings),
        "has_state_delta": False,
    }


def _obs(decision, **kwargs):
    row = {
        "run_id": "run_viz",
        "branch_id": kwargs.get("branch_id", "factual"),
        "subject_ref": _ref("q_00000010", 2),
        "assessment_ref": _ref("assessment:q_00000010", 2),
        "decision_ref": _ref("decision:q_00000010", 2),
        "decision": decision,
        "subject_present": True,
        "outcome_available": True,
        "error": None,
        "decision_reasons": [],
        "required_fixes": [],
        "criterion_values": [],
    }
    row.update(kwargs)
    return row


def _replay(
    *,
    operation,
    target,
    evidence_status,
    noop_decision="accept",
    cf_decision="accept",
    factual_decision="accept",
    content_changed=False,
    control_stable=True,
    role=None,
    role_supported=False,
    criterion_deltas=None,
    error=None,
    replacement=None,
    checkpoint_id="ckpt-plan",
):
    qcc = {
        "kind": "subject_content",
        "factual_available": True,
        "noop_available": True,
        "counterfactual_available": True,
        "control_stable": control_stable,
        "counterfactual_differs_from_noop": content_changed,
        "effect_status": (
            "question_content_changed" if content_changed else "question_content_unchanged"
        ),
        "details": {
            "kind": "mcq_content",
            "content_changed": content_changed,
            "question_content_effect_status": (
                "question_content_changed"
                if content_changed
                else "question_content_unchanged"
            ),
        },
    }
    finding = {}
    if role is not None:
        finding = {
            "information_role": role,
            "information_role_supported": bool(role_supported),
            "causal_classification": role,
            "content_effect": content_changed,
            "decision_effect": evidence_status == "decision_effect",
        }
    return {
        "intervention_signature": {
            "operation": operation,
            "target_ref": _ref(*target) if isinstance(target, tuple) else target,
            "replacement_ref": (
                _ref(*replacement) if isinstance(replacement, tuple) else replacement
            ),
            "checkpoint_id": checkpoint_id,
            "checkpoint_alias": "after_plan",
            "boundary_raw": "after planning",
        },
        "evidence_status": evidence_status,
        "factual": _obs(factual_decision, branch_id="factual"),
        "noop": _obs(
            noop_decision,
            branch_id="noop",
            error=error if evidence_status == "noop_error" else None,
            outcome_available=evidence_status != "outcome_unavailable",
        ),
        "counterfactual": _obs(
            cf_decision,
            branch_id="counterfactual",
            error=error if evidence_status == "intervention_error" else None,
        ),
        "criterion_deltas": list(criterion_deltas or []),
        "noop_path": ["plan_mcq_blueprints", "generate_candidate_mcqs_from_blueprints"],
        "counterfactual_path": [
            "plan_mcq_blueprints",
            "generate_candidate_mcqs_from_blueprints",
        ],
        "first_path_divergence": None,
        "decision_control_stable": control_stable,
        "criteria_control_stable": control_stable,
        "subject_content_control_stable": control_stable,
        "question_content_comparison": qcc,
        "information_role_finding": finding,
        "semantic_intervention_validity": {"status": "valid"},
        "notes": [],
    }


def make_exported_case() -> dict:
    """Synthetic exported case JSON (no ArtifactRegistry / workflow objects)."""
    lo = _art(
        "lo_1",
        1,
        "LearningObjective",
        {
            "lo_id": "lo_1",
            "section_id": "sec_1",
            "objective_text": "Explain the purpose of unsupervised learning",
            "difficulty_level": "intermediate",
        },
        produced_by="occ_lo",
    )
    chunk = _art(
        "E1",
        1,
        "EvidenceChunk",
        {
            "chunk_id": "E1",
            "content": LONG_CHUNK,
            "metadata": {"page_number": 4},
            "used_in_atoms": ["A1"],
            "flags": [],
        },
        produced_by="occ_retrieve",
    )
    atom = _art(
        "A1",
        1,
        "KnowledgeAtom",
        {
            "atom_id": "A1",
            "content": LONG_ATOM,
            "atom_type": "procedure",
            "citations": [{"chunk_id": "E1", "page": 4}],
            "section_id": "sec_1",
            "lo_id": "lo_1",
        },
        produced_by="occ_atom",
    )
    atom2 = _art(
        "A2",
        1,
        "KnowledgeAtom",
        {
            "atom_id": "A2",
            "content": "Secondary knowledge statement about centroids.",
            "atom_type": "concept",
            "citations": [{"chunk_id": "E1", "page": 4}],
            "section_id": "sec_1",
            "lo_id": "lo_1",
        },
        produced_by="occ_atom",
    )
    bp = _art(
        "bp_1",
        1,
        "MCQBlueprint",
        {
            "blueprint_id": "bp_1",
            "section_id": "sec_1",
            "lo_id": "lo_1",
            "target_atom_ids": ["A1", "A2"],
            "evidence_chunk_ids": ["E1"],
            "question_type": "key_characteristic",
            "difficulty": "easy",
            "bloom_level": "Understanding",
            "generation_goal": "Ask which statement best describes k-means assignment.",
            "status": "generated",
        },
        produced_by="occ_plan",
    )
    mcq_v1 = _art(
        "q_00000010",
        1,
        "CandidateMCQ",
        {
            "q_id": "q_00000010",
            "blueprint_id": "bp_1",
            "stem": "Old v1 stem that must not replace v2.",
            "options": [
                {
                    "option_id": "A",
                    "text": "v1 option",
                    "is_correct": True,
                    "explanation": "v1",
                    "misconception": None,
                    "source_refs": [],
                }
            ],
            "correct_option_id": "A",
            "answer_explanation": "v1 explanation",
            "source_refs": [{"doc_id": "141", "page": 4, "chunk_id": "E1"}],
            "status": "revised",
            "revision_count": 0,
            "flags": [],
        },
        produced_by="occ_gen",
    )
    mcq_v2 = _art(
        "q_00000010",
        2,
        "CandidateMCQ",
        {
            "q_id": "q_00000010",
            "blueprint_id": "bp_1",
            "stem": "What does k-means assign each point to?",
            "options": [
                {
                    "option_id": "A",
                    "text": "Nearest centroid",
                    "is_correct": True,
                    "explanation": "Assignment uses distance.",
                    "misconception": None,
                    "source_refs": [{"doc_id": "141", "page": 4, "chunk_id": "E1"}],
                },
                {
                    "option_id": "B",
                    "text": "Random cluster",
                    "is_correct": False,
                    "explanation": "Not supported.",
                    "misconception": "Random assignment",
                    "source_refs": [],
                },
            ],
            "correct_option_id": "A",
            "answer_explanation": "Each point is assigned to the nearest centroid.",
            "source_refs": [{"doc_id": "141", "page": 4, "chunk_id": "E1"}],
            "status": "accepted",
            "revision_count": 1,
            "flags": [],
        },
        produced_by="occ_revise",
    )
    ev = _art(
        "evidence:q_00000010",
        2,
        "EvidenceCheckReport",
        {
            "q_id": "q_00000010",
            "candidate_version": 2,
            "supported_by_pdf": True,
            "confidence": 0.91,
            "stem_supported": True,
            "answer_supported": True,
            "distractors_valid": True,
            "explanation_supported": True,
            "unsupported_claims": [],
            "required_fixes": [],
            "notes": ["ok"],
            "source_refs": [],
        },
        produced_by="occ_verify",
    )
    cove = _art(
        "crosscove:q_00000010",
        2,
        "CrossCoVeReport",
        {
            "q_id": "q_00000010",
            "candidate_version": 2,
            "generator_answer": "A",
            "verifier_answers": ["A", "A", "A"],
            "majority_answer": "A",
            "passed": True,
            "confidence": 1.0,
            "verifier_model": "gpt-5.4-mini-2026-03-17",
            "notes": ["Stub Cross-CoVe verifier (no LLM)."],
        },
        produced_by="occ_cove",
    )
    decision = _art(
        "decision:q_00000010",
        2,
        "SupervisorDecision",
        {
            "q_id": "q_00000010",
            "candidate_version": 2,
            "decision": "accept",
            "reasons": ["passed"],
            "required_fixes": [],
            "assessment_ref": _ref("assessment:q_00000010", 2),
        },
        produced_by="occ_gate",
    )
    assessment = _art(
        "assessment:q_00000010",
        2,
        "OutcomeAssessment",
        {
            "assessment_ref": _ref("assessment:q_00000010", 2),
            "target_ref": _ref("q_00000010", 2),
            "criteria": [
                {
                    "name": "pdf_supported",
                    "value": True,
                    "value_type": "bool",
                    "label": "",
                    "explanation": "",
                    "decision_relevant": True,
                },
                {
                    "name": "cross_cove_passed",
                    "value": True,
                    "value_type": "bool",
                    "label": "",
                    "explanation": "",
                    "decision_relevant": True,
                },
            ],
            "notes": ["gate"],
        },
        produced_by="occ_gate",
    )
    unrelated = _art(
        "lo_unrelated",
        1,
        "LearningObjective",
        {
            "lo_id": "lo_unrelated",
            "section_id": "sec_other",
            "objective_text": "This isolated objective should not stay in relevant.",
            "difficulty_level": "beginner",
        },
    )

    occurrences = [
        _occ("occ_outline", "build_outline", 1),
        _occ("occ_lo", "generate_learning_objectives", 2, produced=[("lo_1", 1)]),
        _occ("occ_retrieve", "initial_retrieve_evidence_for_all_sections", 3, produced=[("E1", 1)]),
        _occ("occ_curate", "curate_evidence", 4),
        _occ(
            "occ_atom",
            "atomize_evidence",
            5,
            produced=[("A1", 1)],
            bindings=[
                {"relation": "used_as_input", "artifact_ref": _ref("E1", 1)},
            ],
        ),
        _occ("occ_plan", "plan_mcq_blueprints", 6, produced=[("bp_1", 1)]),
        _occ(
            "occ_gen",
            "generate_candidate_mcqs_from_blueprints",
            7,
            produced=[("q_00000010", 1)],
        ),
        _occ("occ_revise", "revise_candidate_mcqs", 8, produced=[("q_00000010", 2)]),
        _occ(
            "occ_verify",
            "verify_candidate_mcqs_against_evidence",
            9,
            produced=[("evidence:q_00000010", 2)],
        ),
        _occ(
            "occ_cove",
            "cross_cove_verify_candidate_mcqs",
            10,
            produced=[("crosscove:q_00000010", 2)],
        ),
        _occ(
            "occ_gate",
            "apply_acceptance_gate",
            11,
            produced=[("decision:q_00000010", 2), ("assessment:q_00000010", 2)],
        ),
    ]

    relations = [
        {
            "relation": "produced",
            "artifact_ref": _ref("lo_1", 1),
            "producer_occurrence_id": "occ_lo",
            "consumer_occurrence_id": "",
            "artifact_type": "LearningObjective",
        },
        {
            "relation": "produced",
            "artifact_ref": _ref("E1", 1),
            "producer_occurrence_id": "occ_retrieve",
            "consumer_occurrence_id": "",
            "artifact_type": "EvidenceChunk",
        },
        {
            "relation": "produced",
            "artifact_ref": _ref("A1", 1),
            "producer_occurrence_id": "occ_atom",
            "consumer_occurrence_id": "",
            "artifact_type": "KnowledgeAtom",
        },
        {
            "relation": "used_as_input",
            "artifact_ref": _ref("E1", 1),
            "producer_occurrence_id": "occ_retrieve",
            "consumer_occurrence_id": "occ_atom",
            "artifact_type": "EvidenceChunk",
        },
        {
            "relation": "produced",
            "artifact_ref": _ref("bp_1", 1),
            "producer_occurrence_id": "occ_plan",
            "consumer_occurrence_id": "",
            "artifact_type": "MCQBlueprint",
        },
        {
            "relation": "produced",
            "artifact_ref": _ref("q_00000010", 2),
            "producer_occurrence_id": "occ_revise",
            "consumer_occurrence_id": "",
            "artifact_type": "CandidateMCQ",
        },
        {
            "relation": "produced",
            "artifact_ref": _ref("evidence:q_00000010", 2),
            "producer_occurrence_id": "occ_verify",
            "consumer_occurrence_id": "",
            "artifact_type": "EvidenceCheckReport",
        },
        {
            "relation": "produced",
            "artifact_ref": _ref("crosscove:q_00000010", 2),
            "producer_occurrence_id": "occ_cove",
            "consumer_occurrence_id": "",
            "artifact_type": "CrossCoVeReport",
        },
        {
            "relation": "produced",
            "artifact_ref": _ref("decision:q_00000010", 2),
            "producer_occurrence_id": "occ_gate",
            "consumer_occurrence_id": "",
            "artifact_type": "SupervisorDecision",
        },
    ]

    replays = [
        _replay(
            operation="REMOVE",
            target=("A2", 1),
            evidence_status="decision_effect",
            noop_decision="accept",
            cf_decision="reject",
            content_changed=True,
            role="necessary_for_reproduction",
            role_supported=True,
        ),
        _replay(
            operation="REMOVE",
            target=("A1", 1),
            evidence_status="criterion_effect",
            noop_decision="accept",
            cf_decision="accept",
            content_changed=True,
            role="influential_non_decisive",
            role_supported=True,
            criterion_deltas=[
                {
                    "name": "pdf_supported",
                    "factual": True,
                    "noop": True,
                    "counterfactual": False,
                    "decision_relevant": True,
                    "noop_differs_from_factual": False,
                    "counterfactual_differs_from_noop": True,
                }
            ],
        ),
        _replay(
            operation="REMOVE",
            target=("E1", 1),
            evidence_status="downstream_effect",
            content_changed=False,
            role="artifact_level_only",
            role_supported=False,
        ),
        _replay(
            operation="REMOVE",
            target=("bp_1", 1),
            evidence_status="no_demonstrated_effect",
            content_changed=False,
            role="no_demonstrated_role",
            role_supported=False,
        ),
        _replay(
            operation="REMOVE",
            target=("A1", 1),
            evidence_status="replay_unstable",
            noop_decision="reject",
            cf_decision="reject",
            factual_decision="accept",
            control_stable=False,
            role="influential_non_decisive",
            role_supported=True,
            checkpoint_id="ckpt-unstable",
        ),
        _replay(
            operation="REMOVE",
            target=("A1", 1),
            evidence_status="noop_error",
            error="noop failed",
            checkpoint_id="ckpt-noop-error",
        ),
        _replay(
            operation="REMOVE",
            target=("A1", 1),
            evidence_status="intervention_error",
            error="cf failed",
            checkpoint_id="ckpt-int-error",
        ),
        _replay(
            operation="REMOVE",
            target=("A1", 1),
            evidence_status="outcome_unavailable",
            checkpoint_id="ckpt-outcome",
        ),
    ]
    # A2 is referenced by the first replay but not present as an exported
    # artifact; distance ranking should still work for A1.

    return {
        "candidate_ref": _ref("q_00000010", 2),
        "assessment_ref": _ref("assessment:q_00000010", 2),
        "decision_ref": _ref("decision:q_00000010", 2),
        "run_id": "run_viz",
        "branch_id": "factual",
        "factual_decision": "accept",
        "question": {
            "stem": "What does k-means assign each point to?",
            "options": [],
            "correct_option": "A",
        },
        "conditions": {
            "trace": {"condition": "trace", "raw_evidence": {}},
            "trace_provenance": {"condition": "trace_provenance", "raw_evidence": {}},
            "axis": {
                "condition": "axis",
                "raw_evidence": {"replay_comparisons": replays},
            },
        },
        "visualization_data": {
            "schema_version": 1,
            "target": {
                "candidate_ref": _ref("q_00000010", 2),
                "assessment_ref": _ref("assessment:q_00000010", 2),
                "decision_ref": _ref("decision:q_00000010", 2),
                "factual_decision": "accept",
            },
            "occurrences": occurrences,
            "artifacts": [
                lo,
                chunk,
                atom,
                atom2,
                bp,
                mcq_v1,
                mcq_v2,
                ev,
                cove,
                decision,
                assessment,
                unrelated,
            ],
            "provenance_relations": relations,
            "criteria": [{"name": "pdf_supported", "value": True}],
        },
    }


def reverse_dict_keys(obj):
    if isinstance(obj, dict):
        items = list(obj.items())[::-1]
        return {k: reverse_dict_keys(v) for k, v in items}
    if isinstance(obj, list):
        return [reverse_dict_keys(x) for x in obj]
    return obj


def _kinds(graph):
    return {n.kind for n in graph.nodes}


def _edge_kinds(graph):
    return {e.kind for e in graph.edges}


class TestExplanationGraphContract(unittest.TestCase):
    def setUp(self):
        self.case = make_exported_case()

    def test_missing_visualization_data_raises_clear_error(self):
        with self.assertRaises(VisualizationDataError) as ctx:
            build_explanation_graph({"question": {}}, condition="trace")
        self.assertIn("visualization_data", str(ctx.exception).lower())

    def test_invalid_condition_and_level(self):
        with self.assertRaises(VisualizationDataError):
            build_explanation_graph(self.case, condition="nope")
        with self.assertRaises(VisualizationDataError):
            build_explanation_graph(self.case, condition="trace", detail_level="tiny")

    def test_selection_scope_is_case_slice(self):
        g = build_explanation_graph(self.case, condition="trace", detail_level="full")
        self.assertEqual(g.metadata.get("selection_scope"), "target_case_slice")

    def test_trace_excludes_provenance_and_replay(self):
        g = build_explanation_graph(self.case, condition="trace", detail_level="full")
        self.assertNotIn("produces", _edge_kinds(g))
        self.assertNotIn("consumes", _edge_kinds(g))
        self.assertNotIn("declared_reference", _edge_kinds(g))
        self.assertNotIn("intervention", _kinds(g))
        self.assertNotIn("comparison", _kinds(g))
        self.assertFalse(any(n.subtype == "noop" for n in g.nodes))
        self.assertFalse(any(n.subtype == "counterfactual" for n in g.nodes))
        self.assertFalse(any(n.node_id.startswith("replay::") for n in g.nodes))
        self.assertFalse(any(n.subtype == "EvidenceChunk" for n in g.nodes))
        self.assertFalse(any(n.subtype == "KnowledgeAtom" for n in g.nodes))
        blob = json.dumps(g.to_dict())
        self.assertNotIn("used_as_input", blob)
        self.assertNotIn("produced_artifacts", blob)
        self.assertNotIn("input_bindings", blob)

    def test_trace_allows_target_question_and_decision(self):
        g = build_explanation_graph(self.case, condition="trace", detail_level="full")
        ids = set(g.node_ids())
        self.assertIn("artifact::q_00000010:v2", ids)
        self.assertTrue(
            any(n.style_role == "decision" for n in g.nodes)
            or "artifact::decision:q_00000010:v2" in ids
        )
        mcq = g.node_map()["artifact::q_00000010:v2"]
        self.assertEqual(mcq.preview, "What does k-means assign each point to?")
        content = mcq.details["content"]
        self.assertEqual(len(content["options"]), 2)
        self.assertNotIn("source_refs", content)

    def test_trace_provenance_excludes_replay(self):
        g = build_explanation_graph(
            self.case, condition="trace_provenance", detail_level="full"
        )
        self.assertIn("produces", _edge_kinds(g))
        self.assertIn("consumes", _edge_kinds(g))
        self.assertNotIn("intervention", _kinds(g))
        self.assertFalse(any(n.node_id.startswith("replay::") for n in g.nodes))
        self.assertNotIn("replay_branch", _edge_kinds(g))
        self.assertNotIn("intervention_target", _edge_kinds(g))

    def test_axis_includes_replay_unit(self):
        g = build_explanation_graph(self.case, condition="axis", detail_level="full")
        self.assertIn("intervention", _kinds(g))
        self.assertTrue(any(n.subtype == "noop" for n in g.nodes))
        self.assertTrue(any(n.subtype == "counterfactual" for n in g.nodes))
        self.assertTrue(any(n.kind == "comparison" for n in g.nodes))
        self.assertIn("Ø", "".join(n.preview for n in g.nodes if n.subtype == "noop"))

    def test_condition_information_class_subset(self):
        tr = build_explanation_graph(self.case, "trace", "full")
        tp = build_explanation_graph(self.case, "trace_provenance", "full")
        ax = build_explanation_graph(self.case, "axis", "full")
        self.assertTrue(set(tr.node_ids()) <= set(tp.node_ids()) <= set(ax.node_ids()))

    def test_declared_reference_is_not_consumes(self):
        g = build_explanation_graph(
            self.case, condition="trace_provenance", detail_level="full"
        )
        lo_plan_consumes = [
            e
            for e in g.edges
            if e.kind == "consumes"
            and e.source == "artifact::lo_1:v1"
            and e.target == "occurrence::occ_plan"
        ]
        self.assertEqual(lo_plan_consumes, [])
        declared = [
            e
            for e in g.edges
            if e.kind == "declared_reference"
            and e.source == "artifact::bp_1:v1"
            and e.target == "artifact::lo_1:v1"
        ]
        self.assertTrue(declared)
        self.assertEqual(declared[0].relation, "lo_id")
        self.assertFalse(
            any(
                word in (e.label + e.relation + e.kind).lower()
                for e in g.edges
                for word in ("cause", "influenc", "determin")
            )
        )

    def test_target_version_v2_is_preserved(self):
        g = build_explanation_graph(self.case, condition="axis", detail_level="full")
        self.assertEqual(g.target_ref, "q_00000010:v2")
        self.assertIn("artifact::q_00000010:v2", g.node_ids())
        payload = json.dumps(g.to_dict())
        self.assertIn("q_00000010:v2", payload)
        target_node = g.node_map()["artifact::q_00000010:v2"]
        self.assertEqual(target_node.source_ref, "q_00000010:v2")
        self.assertEqual(target_node.details["ref"]["version"], 2)
        self.assertEqual(target_node.details["ref"]["key"], "q_00000010:v2")
        # v1 may exist as a separate node but must not replace the target.
        if "artifact::q_00000010:v1" in g.node_map():
            self.assertNotEqual(g.target_ref, "q_00000010:v1")
            self.assertNotEqual(g.target_ref, "q_00000010")
        ess = build_explanation_graph(self.case, condition="axis", detail_level="essential")
        self.assertEqual(ess.target_ref, "q_00000010:v2")
        self.assertIn("artifact::q_00000010:v2", ess.node_ids())
        self.assertNotIn("artifact::q_00000010:v1", ess.node_ids())

    def test_content_previews_and_full_details(self):
        g = build_explanation_graph(
            self.case, condition="trace_provenance", detail_level="full"
        )
        lo = g.node_map()["artifact::lo_1:v1"]
        self.assertEqual(lo.label, "Learning objective")
        self.assertIn("unsupervised learning", lo.preview)
        self.assertEqual(
            lo.details["content"]["objective_text"],
            "Explain the purpose of unsupervised learning",
        )
        mcq = g.node_map()["artifact::q_00000010:v2"]
        self.assertEqual(mcq.preview, "What does k-means assign each point to?")
        self.assertEqual(len(mcq.details["content"]["options"]), 2)
        chunk = g.node_map()["artifact::E1:v1"]
        full = chunk.details["content"]["content"]
        self.assertEqual(full, LONG_CHUNK)
        self.assertLess(len(chunk.preview), len(full))
        self.assertTrue(chunk.preview.endswith("..."))
        atom = g.node_map()["artifact::A1:v1"]
        self.assertEqual(atom.details["content"]["content"], LONG_ATOM)
        bp = g.node_map()["artifact::bp_1:v1"]
        self.assertIn("k-means assignment", bp.details["content"]["generation_goal"])
        cove = g.node_map()["artifact::crosscove:q_00000010:v2"]
        self.assertEqual(cove.details["content"]["verifier_answers"], ["A", "A", "A"])
        self.assertEqual(
            cove.details["content"]["verifier_model"], "gpt-5.4-mini-2026-03-17"
        )

    def test_axis_replay_content_vs_decision(self):
        g = build_explanation_graph(self.case, condition="axis", detail_level="full")
        results = [n for n in g.nodes if n.kind == "comparison"]
        target = [
            n
            for n in results
            if n.details.get("evidence_status") == "criterion_effect"
            and (n.source_ref or "").startswith("A1")
        ]
        self.assertTrue(target)
        node = target[0]
        self.assertTrue(
            any(x.node_id.endswith("::intervention") for x in g.nodes if node.source_ref == x.source_ref or x.node_id.split("::")[1] == node.node_id.split("::")[1])
        )
        rid = node.node_id.split("::")[1]
        ids = set(g.node_ids())
        self.assertIn(f"replay::{rid}::intervention", ids)
        self.assertIn(f"replay::{rid}::noop", ids)
        self.assertIn(f"replay::{rid}::counterfactual", ids)
        self.assertIn(f"replay::{rid}::result", ids)
        self.assertEqual(node.details["decision_changed"], False)
        self.assertEqual(node.details["question_content_changed"], True)
        self.assertNotEqual(node.details.get("information_role"), "decisive")
        self.assertNotIn("▲ Decisive", node.preview)
        self.assertIn("influential_non_decisive", str(node.details.get("information_role")))

    def test_unstable_control_is_not_supported_causal_evidence(self):
        g = build_explanation_graph(self.case, condition="axis", detail_level="full")
        unstable = [
            n
            for n in g.nodes
            if n.kind == "comparison"
            and (
                n.details.get("unstable")
                or n.details.get("evidence_status") == "replay_unstable"
            )
        ]
        self.assertTrue(unstable)
        for node in unstable:
            self.assertEqual(node.style_role, "warning")
            self.assertNotIn("▲ Influential, non-decisive", node.preview)
            self.assertIn("Inconclusive", node.preview)

    def test_invalid_replays_use_warning_style(self):
        g = build_explanation_graph(self.case, condition="axis", detail_level="full")
        statuses = {
            n.details.get("evidence_status")
            for n in g.nodes
            if n.kind == "comparison"
        }
        for status in (
            "noop_error",
            "replay_unstable",
            "intervention_error",
            "outcome_unavailable",
        ):
            self.assertIn(status, statuses)
        invalid_nodes = [
            n
            for n in g.nodes
            if n.kind == "comparison"
            and n.details.get("evidence_status")
            in {
                "noop_error",
                "replay_unstable",
                "intervention_error",
                "outcome_unavailable",
            }
        ]
        self.assertTrue(all(n.style_role == "warning" for n in invalid_nodes))

    def test_determinism_against_reversed_dict_keys(self):
        a = build_explanation_graph(self.case, "axis", "essential")
        b = build_explanation_graph(reverse_dict_keys(self.case), "axis", "essential")
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(render_dot(a), render_dot(b))
        self.assertEqual(render_mermaid(a), render_mermaid(b))
        self.assertEqual(render_tikz(a), render_tikz(b))

    def test_to_dict_is_json_serializable(self):
        g = build_explanation_graph(self.case, "axis", "relevant")
        raw = json.dumps(g.to_dict())
        self.assertIn("q_00000010:v2", raw)


if __name__ == "__main__":
    unittest.main()
