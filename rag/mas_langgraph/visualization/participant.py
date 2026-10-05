"""Participant-safe presentation projection for blinded explanation evaluation.

This module does not change ExplanationGraph evidence semantics, relevance,
replay ranking, or causality. It always builds the canonical graph first:

    canonical = build_explanation_graph(case, condition, PARTICIPANT_DETAIL_LEVEL)
    participant = project_participant_graph(canonical, case)

Only presentation fields may change: labels, previews, lane display labels,
and a whitelist of participant-safe details.

``PARTICIPANT_DETAIL_LEVEL`` is fixed to ``essential`` so graph detail is not
an experimental variable. Do not expose a participant UI control for it.

Participant modules are imported only by the participant app and its tests.
They are not re-exported from ``mas_langgraph.visualization``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mas_langgraph.experiment.models import EVIDENCE_ACCESS

from .builder import build_explanation_graph
from .labels import (
    artifact_preview,
    effect_symbol,
    effect_word,
    make_preview,
    occurrence_label,
)
from .models import (
    CONDITION_AXIS,
    CONDITION_TRACE,
    CONDITION_TRACE_PROVENANCE,
    EDGE_COMPARISON,
    EDGE_CONSUMES,
    EDGE_CONTROL_FLOW,
    EDGE_DECLARED_REFERENCE,
    EDGE_INTERVENTION_TARGET,
    EDGE_PRODUCES,
    EDGE_REPLAY_BRANCH,
    KIND_ARTIFACT,
    KIND_CHECKPOINT,
    KIND_COMPARISON,
    KIND_INTERVENTION,
    KIND_OCCURRENCE,
    KIND_OUTCOME,
    ExplanationGraph,
    GraphEdge,
    GraphLane,
    GraphNode,
    VisualizationDataError,
)
from .render_dot import render_dot


logger = logging.getLogger(__name__)

PARTICIPANT_DETAIL_LEVEL = "essential"

ENV_EXPERIMENT_JSON = "EXPLANATION_EXPERIMENT_JSON"
ENV_PREVIEW_DISCOVERY = "PARTICIPANT_PREVIEW_ALLOW_LOCAL_DISCOVERY"

PARTICIPANT_UNAVAILABLE_MESSAGE = "This explanation set is not available."

BLINDED_LABELS: Tuple[str, ...] = ("A", "B", "C")
REQUIRED_CONDITIONS = frozenset(
    {CONDITION_TRACE, CONDITION_TRACE_PROVENANCE, CONDITION_AXIS}
)

PARTICIPANT_LANE_LABELS: Dict[str, str] = {
    "artifacts": "Information",
    "factual": "How the question was created",
    "noop": "Test without a change",
    "counterfactual": "Test with one change",
    "comparison": "Observed result",
}

PARTICIPANT_ARTIFACT_LABELS: Dict[str, str] = {
    "DocSection": "Document section",
    "LearningObjective": "Learning objective",
    "EvidenceChunk": "Teaching evidence",
    "KnowledgeAtom": "Knowledge statement",
    "MCQBlueprint": "Question plan",
    "CandidateMCQ": "Question",
    "EvidenceCheckReport": "Evidence check",
    "CrossCoVeReport": "Independent answer check",
    "SupervisorDecision": "Acceptance decision",
    "OutcomeAssessment": "Acceptance criteria",
}

PARTICIPANT_NODE_KIND_LABELS: Dict[str, str] = {
    KIND_INTERVENTION: "Tested change",
    KIND_CHECKPOINT: "Test starting point",
    KIND_COMPARISON: "Observed effect",
}

PARTICIPANT_OUTCOME_LABELS: Dict[str, str] = {
    "noop": "Result without a change",
    "counterfactual": "Result after the change",
}

PARTICIPANT_OPERATION_LABELS: Dict[str, str] = {
    "REMOVE": "Remove information",
    "ADD": "Add information",
    "REPLACE": "Replace information",
    "WHATIF": "Replace information",
}

PARTICIPANT_ROLE_LABELS: Dict[str, str] = {
    "focus_shaping": "Changed the focus of the question",
    "answer_anchoring": "Helped anchor the answer",
    "distractor_shaping": "Changed the distractors",
    "formulation_shaping": "Changed the wording of the question",
    "necessary_for_reproduction": "Needed to reproduce the same question",
    "influential_non_decisive": (
        "Influenced the question; acceptance stayed the same"
    ),
    "decisive": "Changed the acceptance decision",
    "no_demonstrated_effect": "No demonstrated effect in this test",
    "no_demonstrated_role": "No demonstrated effect in this test",
    "robust_to_tested_information": "Result stayed stable in this test",
    "artifact_level_downstream": "A downstream change was observed",
    "artifact_level_only": "Only an intermediate change was observed",
    "inconclusive": "Inconclusive",
}

SUPPORTED_CAUSAL_ROLES = frozenset(
    {
        "focus_shaping",
        "answer_anchoring",
        "distractor_shaping",
        "formulation_shaping",
        "necessary_for_reproduction",
        "influential_non_decisive",
        "decisive",
    }
)

CONTENT_ARTIFACT_SUBTYPES = frozenset(
    {
        "LearningObjective",
        "EvidenceChunk",
        "KnowledgeAtom",
        "MCQBlueprint",
        "CandidateMCQ",
        "DocSection",
    }
)

PARTICIPANT_LEGEND: Tuple[str, ...] = (
    "Blue boxes/arrows: How the question was created",
    "Light-blue cards: Information recorded by the system",
    "Purple dashed arrow: Recorded use of information",
    "Gray dotted arrow: Recorded link between information items",
    "Orange: Tested change",
    "Gray test lane: Test without a change",
    "Δ: Changed",
    "=: Unchanged",
    "?: Inconclusive / unavailable",
    (
        "A recorded information link shows what the system recorded as being "
        "used or connected. By itself, it does not show that the information "
        "caused the final result."
    ),
)

PARTICIPANT_UI_HEADINGS: Tuple[str, ...] = (
    "Explanation Study",
    "Correct answer:",
    "Acceptance:",
    "Explanation:",
    "Text explanation",
    "Graphical explanation",
    "How to read the graph",
    "Previous question",
    "Next question",
)

# Structural UI jargon only. Do not run this list over teaching-source previews.
FORBIDDEN_STRUCTURAL_TOKENS: Tuple[str, ...] = (
    "AXIS",
    "Trace + Provenance",
    "Trace",
    "provenance",
    "NOOP",
    "counterfactual",
    "checkpoint",
    "REMOVE",
    "ADD",
    "REPLACE",
    "WHATIF",
    "ArtifactRef",
    "CandidateMCQ",
    "KnowledgeAtom",
    "EvidenceChunk",
    "CrossCoVeReport",
    "SupervisorDecision",
    "OutcomeAssessment",
    "D_run",
    "input_bindings",
    "occurrence_id",
    "branch_id",
    "run_id",
    "visualization_data",
    "intervention_signature",
    "evidence_status",
    "verifier_model",
    "gpt-",
)

_CONTENT_PREVIEW_SUBTYPES = CONTENT_ARTIFACT_SUBTYPES


class ParticipantPresentationError(ValueError):
    """Raised when a participant presentation cannot be built safely."""


@dataclass(frozen=True)
class ParticipantConditionView:
    """Everything the participant app may render for one blinded condition."""

    blinded_label: str
    heading: str
    ordinal: int
    total: int
    ordinal_label: str
    question: Mapping[str, Any]
    explanation: str
    graph: ExplanationGraph
    legend: Tuple[str, ...] = field(default=PARTICIPANT_LEGEND)
    availability: str = "ok"
    internal_case_key: str = ""

    def display_payload(self) -> Dict[str, Any]:
        """UI objects that may be sent to participant display.

        Omits the graph object, internal case identity, and true condition.
        """
        return {
            "blinded_label": self.blinded_label,
            "heading": self.heading,
            "ordinal_label": self.ordinal_label,
            "question": {
                "stem": self.question.get("stem") or "",
                "options": list(self.question.get("options") or []),
                "correct_answer": self.question.get("correct_answer"),
                "factual_decision": self.question.get("factual_decision"),
            },
            "explanation": self.explanation,
            "legend": list(self.legend),
            "availability": self.availability,
            "headings": list(PARTICIPANT_UI_HEADINGS),
        }


def blinded_heading(label: str) -> str:
    return f"Explanation {label}"


def validate_blinding(mapping: Any) -> Dict[str, str]:
    """Require exactly A/B/C mapping onto the three conditions once each."""
    if not isinstance(mapping, Mapping):
        raise ParticipantPresentationError(
            "case blinding is missing or is not a mapping"
        )
    keys = set(str(k) for k in mapping.keys())
    if keys != set(BLINDED_LABELS):
        raise ParticipantPresentationError(
            "case blinding must contain exactly labels A, B, and C"
        )
    values = [str(mapping[label]) for label in BLINDED_LABELS]
    if set(values) != REQUIRED_CONDITIONS:
        raise ParticipantPresentationError(
            "case blinding must map A, B, and C onto trace, "
            "trace_provenance, and axis"
        )
    if len(values) != len(set(values)):
        raise ParticipantPresentationError(
            "case blinding must contain each condition exactly once"
        )
    return {label: str(mapping[label]) for label in BLINDED_LABELS}


def validate_evidence_access(payload: Mapping[str, Any]) -> None:
    """Safety assertion only. The canonical builder remains authoritative."""
    actual = payload.get("evidence_access")
    if not isinstance(actual, Mapping):
        raise ParticipantPresentationError(
            "experiment evidence_access is missing or invalid"
        )
    for condition in (CONDITION_TRACE, CONDITION_TRACE_PROVENANCE, CONDITION_AXIS):
        expected_block = EVIDENCE_ACCESS[condition]
        block = actual.get(condition)
        if not isinstance(block, Mapping):
            raise ParticipantPresentationError(
                f"evidence_access.{condition} is missing or invalid"
            )
        for flag in ("execution_trace", "provenance", "counterfactual_replay"):
            if bool(block.get(flag)) != bool(expected_block[flag]):
                raise ParticipantPresentationError(
                    "evidence_access contradicts the expected access policy"
                )


def _case_identity_key(case: Mapping[str, Any], index: int = 0) -> str:
    key = _ref_key(case.get("candidate_ref"))
    if key:
        return key
    viz = case.get("visualization_data")
    viz = viz if isinstance(viz, Mapping) else {}
    target = viz.get("target") if isinstance(viz.get("target"), Mapping) else {}
    key = _ref_key(target.get("candidate_ref"))
    if key:
        return key
    return f"case_{index}"


def _ref_key(ref: Any) -> Optional[str]:
    if isinstance(ref, Mapping) and ref.get("key"):
        return str(ref.get("key"))
    if isinstance(ref, str) and ref.strip():
        return ref.strip()
    if isinstance(ref, Mapping):
        logical = str(ref.get("logical_id") or "")
        version = ref.get("version")
        if logical and version is not None:
            return f"{logical}:v{int(version)}"
        if logical:
            return logical
    return None


def _require_visualization_data(case: Mapping[str, Any]) -> None:
    viz = case.get("visualization_data")
    if not isinstance(viz, Mapping) or not viz:
        raise ParticipantPresentationError(
            "visualization_data is missing; participant mode cannot render "
            "a text-only fallback"
        )
    if viz.get("schema_version") != 1:
        raise ParticipantPresentationError(
            "visualization_data.schema_version is unsupported"
        )


def _require_condition_explanations(case: Mapping[str, Any]) -> None:
    conditions = case.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ParticipantPresentationError("case conditions are missing")
    for name in (CONDITION_TRACE, CONDITION_TRACE_PROVENANCE, CONDITION_AXIS):
        block = conditions.get(name)
        if not isinstance(block, Mapping):
            raise ParticipantPresentationError(
                f"condition {name} explanation block is missing"
            )
        if "explanation" not in block:
            raise ParticipantPresentationError(
                f"condition {name} explanation is missing"
            )
        if block.get("explanation") is None:
            raise ParticipantPresentationError(
                f"condition {name} explanation is missing"
            )
        if not isinstance(block.get("explanation"), str):
            raise ParticipantPresentationError(
                f"condition {name} explanation must be a string"
            )


def _require_canonical_graphs(case: Mapping[str, Any]) -> None:
    for condition in (CONDITION_TRACE, CONDITION_TRACE_PROVENANCE, CONDITION_AXIS):
        try:
            build_explanation_graph(
                case,
                condition=condition,
                detail_level=PARTICIPANT_DETAIL_LEVEL,
            )
        except VisualizationDataError as exc:
            raise ParticipantPresentationError(
                f"canonical graph build failed for {condition}: {exc}"
            ) from exc


def preflight_participant_experiment(payload: Any) -> Mapping[str, Any]:
    """Fail closed before any participant case is shown."""
    if not isinstance(payload, Mapping):
        raise ParticipantPresentationError("experiment JSON is not an object")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ParticipantPresentationError("experiment JSON has no cases")
    validate_evidence_access(payload)
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ParticipantPresentationError(f"case {index} is not an object")
        _require_visualization_data(case)
        validate_blinding(case.get("blinding"))
        _require_condition_explanations(case)
        _require_canonical_graphs(case)
    return payload


def resolve_participant_experiment_path() -> Path:
    """Resolve the configured experiment file. Default is fail-closed."""
    raw = str(os.environ.get(ENV_EXPERIMENT_JSON) or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_file():
            raise ParticipantPresentationError(
                "EXPLANATION_EXPERIMENT_JSON does not point to a file"
            )
        return path
    preview = str(os.environ.get(ENV_PREVIEW_DISCOVERY) or "").strip()
    if preview == "1":
        from .ui_helpers import find_local_experiment_files

        files = find_local_experiment_files()
        if files:
            logger.info("Participant preview discovery selected a local experiment file")
            return files[0]
        raise ParticipantPresentationError(
            "participant preview discovery is enabled but no experiment file was found"
        )
    raise ParticipantPresentationError(
        "EXPLANATION_EXPERIMENT_JSON is not set; participant mode is fail-closed"
    )


def load_participant_experiment(path: Path) -> Mapping[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParticipantPresentationError(
            "experiment JSON could not be read"
        ) from exc
    return preflight_participant_experiment(payload)


def common_question_context(
    case: Mapping[str, Any],
    *,
    ordinal: int,
    total: int,
) -> Dict[str, Any]:
    question = case.get("question") if isinstance(case.get("question"), Mapping) else {}
    stem = str(question.get("stem") or "")
    options = _safe_options(question.get("options"))
    if not options:
        options = _options_from_visualization(case)
    correct = question.get("correct_option") or question.get("correct_option_id")
    if correct in (None, ""):
        correct = _correct_from_options(question.get("options"))
    viz = case.get("visualization_data") if isinstance(case.get("visualization_data"), Mapping) else {}
    target = viz.get("target") if isinstance(viz.get("target"), Mapping) else {}
    decision = case.get("factual_decision")
    if decision in (None, ""):
        decision = target.get("factual_decision")
    return {
        "stem": stem,
        "options": options,
        "correct_answer": correct,
        "factual_decision": decision,
        "ordinal": int(ordinal),
        "total": int(total),
        "ordinal_label": f"Question {int(ordinal)} of {int(total)}",
    }


def _safe_options(raw: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        oid = str(item.get("option_id") or "")
        text = str(item.get("text") or "")
        if not oid and not text:
            continue
        out.append({"option_id": oid, "text": text})
    return out


def _correct_from_options(raw: Any) -> Optional[str]:
    for item in raw or []:
        if isinstance(item, Mapping) and item.get("is_correct"):
            return item.get("option_id")
    return None


def _options_from_visualization(case: Mapping[str, Any]) -> List[Dict[str, str]]:
    viz = case.get("visualization_data")
    if not isinstance(viz, Mapping):
        return []
    target_key = _ref_key((viz.get("target") or {}).get("candidate_ref")) if isinstance(viz.get("target"), Mapping) else None
    if not target_key:
        target_key = _ref_key(case.get("candidate_ref"))
    for art in viz.get("artifacts") or []:
        if not isinstance(art, Mapping):
            continue
        if _ref_key(art.get("ref")) != target_key:
            continue
        content = art.get("content") if isinstance(art.get("content"), Mapping) else {}
        return _safe_options(content.get("options"))
    return []


def condition_explanation(case: Mapping[str, Any], true_condition: str) -> str:
    conditions = case.get("conditions") if isinstance(case.get("conditions"), Mapping) else {}
    block = conditions.get(true_condition)
    if not isinstance(block, Mapping) or "explanation" not in block:
        raise ParticipantPresentationError(
            f"explanation for blinded condition is missing"
        )
    text = block.get("explanation")
    if not isinstance(text, str):
        raise ParticipantPresentationError("explanation is not a string")
    return text


def participant_node_label(node: GraphNode) -> str:
    kind = str(node.kind or "")
    subtype = str(node.subtype or "")
    if kind == KIND_OCCURRENCE:
        return occurrence_label(subtype)
    if kind == KIND_INTERVENTION:
        return PARTICIPANT_NODE_KIND_LABELS[KIND_INTERVENTION]
    if kind == KIND_CHECKPOINT:
        return PARTICIPANT_NODE_KIND_LABELS[KIND_CHECKPOINT]
    if kind == KIND_COMPARISON:
        return PARTICIPANT_NODE_KIND_LABELS[KIND_COMPARISON]
    if kind == KIND_OUTCOME and subtype in PARTICIPANT_OUTCOME_LABELS:
        return PARTICIPANT_OUTCOME_LABELS[subtype]
    if subtype in PARTICIPANT_ARTIFACT_LABELS:
        return PARTICIPANT_ARTIFACT_LABELS[subtype]
    return "Information"


def _node_content(node: GraphNode) -> Mapping[str, Any]:
    details = node.details if isinstance(node.details, Mapping) else {}
    content = details.get("content")
    if isinstance(content, Mapping):
        return content
    return details


def _participant_artifact_preview(node: GraphNode) -> str:
    subtype = str(node.subtype or "")
    content = _node_content(node)
    if subtype == "MCQBlueprint":
        return make_preview(content.get("generation_goal"))
    if subtype == "CandidateMCQ":
        return make_preview(content.get("stem") or node.preview)
    if subtype == "LearningObjective":
        return make_preview(content.get("objective_text"))
    if subtype in {"EvidenceChunk", "KnowledgeAtom"}:
        return make_preview(content.get("content"))
    return artifact_preview(subtype, content) or make_preview(node.preview)


def _lookup_artifact_preview(graph: ExplanationGraph, ref_key: Optional[str]) -> str:
    if not ref_key:
        return ""
    node = graph.node_map().get(f"artifact::{ref_key}")
    if node is None:
        for candidate in graph.nodes:
            if candidate.source_ref == ref_key and candidate.kind == KIND_ARTIFACT:
                node = candidate
                break
    if node is None:
        return ""
    return _participant_artifact_preview(node)


def _operation_label(subtype: str) -> str:
    op = str(subtype or "").upper()
    return PARTICIPANT_OPERATION_LABELS.get(op, "Change information")


def _unavailable_change_preview(subtype: str) -> str:
    op = str(subtype or "").upper()
    if op == "REMOVE":
        return "Remove selected information"
    if op == "ADD":
        return "Add selected information"
    if op in {"REPLACE", "WHATIF"}:
        return "Replace selected information"
    return "Change selected information"


def _intervention_preview(node: GraphNode, canonical: ExplanationGraph) -> str:
    op_label = _operation_label(node.subtype)
    excerpt = _lookup_artifact_preview(canonical, node.source_ref)
    if excerpt:
        return f"{op_label}:\n\"{excerpt}\""
    return _unavailable_change_preview(node.subtype)


def _participant_role_label(details: Mapping[str, Any]) -> str:
    invalid = bool(details.get("invalid"))
    unstable = bool(details.get("unstable"))
    if invalid or unstable:
        return PARTICIPANT_ROLE_LABELS["inconclusive"]
    role = details.get("information_role") or details.get("causal_classification")
    if not role:
        return ""
    key = str(role)
    supported = details.get("information_role_supported")
    if key in SUPPORTED_CAUSAL_ROLES and supported is False:
        # Canonical stored the role but did not treat it as supported evidence.
        fallback = details.get("causal_classification")
        if fallback and str(fallback) in PARTICIPANT_ROLE_LABELS:
            return PARTICIPANT_ROLE_LABELS[str(fallback)]
        return PARTICIPANT_ROLE_LABELS.get(key, PARTICIPANT_ROLE_LABELS["inconclusive"])
    return PARTICIPANT_ROLE_LABELS.get(key, PARTICIPANT_ROLE_LABELS["inconclusive"])


def _effect_preview(details: Mapping[str, Any]) -> str:
    q_changed = details.get("question_content_changed")
    d_changed = details.get("decision_changed")
    line = (
        f"Question content: {effect_symbol(q_changed)} {effect_word(q_changed)}; "
        f"Acceptance: {effect_symbol(d_changed)} {effect_word(d_changed)}"
    )
    role = _participant_role_label(details)
    if role:
        return f"{line}\n{role}"
    return line


def _safe_details_for_node(
    node: GraphNode,
    canonical: ExplanationGraph,
) -> Dict[str, Any]:
    kind = str(node.kind or "")
    subtype = str(node.subtype or "")
    details = node.details if isinstance(node.details, Mapping) else {}
    content = _node_content(node)

    if kind == KIND_OCCURRENCE:
        return {"step": occurrence_label(subtype)}

    if kind == KIND_INTERVENTION:
        excerpt = _lookup_artifact_preview(canonical, node.source_ref)
        return {
            "change": _operation_label(subtype),
            "target_excerpt": excerpt,
        }

    if kind == KIND_CHECKPOINT:
        return {}

    if kind == KIND_COMPARISON:
        return {
            "question_content": (
                f"{effect_symbol(details.get('question_content_changed'))} "
                f"{effect_word(details.get('question_content_changed'))}"
            ),
            "acceptance": (
                f"{effect_symbol(details.get('decision_changed'))} "
                f"{effect_word(details.get('decision_changed'))}"
            ),
            "role": _participant_role_label(details),
        }

    if kind == KIND_OUTCOME and subtype in {"noop", "counterfactual"}:
        decision = details.get("decision")
        return {"decision": decision} if decision not in (None, "") else {}

    if subtype == "CandidateMCQ":
        options = _safe_options(content.get("options"))
        correct = content.get("correct_option_id") or content.get("correct_option")
        return {
            "stem": content.get("stem") or "",
            "options": options,
            "correct_option_id": correct,
        }

    if subtype == "SupervisorDecision":
        decision = content.get("decision") or details.get("decision")
        return {"decision": decision} if decision not in (None, "") else {}

    if subtype in PARTICIPANT_ARTIFACT_LABELS:
        return {"kind_label": PARTICIPANT_ARTIFACT_LABELS[subtype]}

    return {}


def _participant_preview(node: GraphNode, canonical: ExplanationGraph) -> str:
    kind = str(node.kind or "")
    subtype = str(node.subtype or "")
    details = node.details if isinstance(node.details, Mapping) else {}

    if kind == KIND_OCCURRENCE:
        return occurrence_label(subtype)
    if kind == KIND_INTERVENTION:
        return _intervention_preview(node, canonical)
    if kind == KIND_CHECKPOINT:
        return ""
    if kind == KIND_COMPARISON:
        return _effect_preview(details)
    if kind == KIND_OUTCOME and subtype in {"noop", "counterfactual"}:
        decision = details.get("decision")
        return str(decision) if decision not in (None, "") else ""
    if kind in {KIND_ARTIFACT, KIND_OUTCOME}:
        return _participant_artifact_preview(node)
    return make_preview(node.preview)


def _participant_edge_label(edge: GraphEdge) -> str:
    if edge.kind == EDGE_CONTROL_FLOW:
        skipped = 0
        if isinstance(edge.details, Mapping):
            skipped = int(edge.details.get("skipped_occurrence_count") or 0)
        if skipped > 0:
            return f"... {skipped} steps ..."
        return ""
    return ""


def _participant_edge_details(edge: GraphEdge) -> Dict[str, Any]:
    if edge.kind == EDGE_CONTROL_FLOW and isinstance(edge.details, Mapping):
        skipped = edge.details.get("skipped_occurrence_count")
        if skipped not in (None, ""):
            return {"skipped_occurrence_count": int(skipped)}
    return {}


def project_participant_graph(
    canonical_graph: ExplanationGraph,
    case_dict: Mapping[str, Any],
) -> ExplanationGraph:
    """Rewrite presentation fields only. Semantic membership is unchanged."""
    del case_dict  # reserved for future case-level display context
    nodes = []
    for node in canonical_graph.nodes:
        nodes.append(
            GraphNode(
                node_id=node.node_id,
                kind=node.kind,
                subtype=node.subtype,
                lane=node.lane,
                label=participant_node_label(node),
                preview=_participant_preview(node, canonical_graph),
                details=_safe_details_for_node(node, canonical_graph),
                source_ref=None,
                occurrence_id=None,
                style_role=node.style_role,
                sort_key=node.sort_key,
            )
        )
    edges = []
    for edge in canonical_graph.edges:
        edges.append(
            GraphEdge(
                edge_id=edge.edge_id,
                source=edge.source,
                target=edge.target,
                kind=edge.kind,
                relation=edge.relation,
                label=_participant_edge_label(edge),
                style_role=edge.style_role,
                sort_key=edge.sort_key,
                details=_participant_edge_details(edge),
            )
        )
    lanes = tuple(
        GraphLane(
            lane_id=lane.lane_id,
            label=PARTICIPANT_LANE_LABELS.get(lane.lane_id, "Information"),
            order=lane.order,
        )
        for lane in canonical_graph.lanes
    )
    return canonical_graph.replace(
        nodes=nodes,
        edges=edges,
        lanes=lanes,
        metadata={"presentation": "participant"},
        hidden_counts={
            "hidden_node_count": 0,
            "hidden_edge_count": 0,
            "hidden_replay_count": 0,
        },
        warnings=(),
    )


def assert_projection_semantics(
    canonical: ExplanationGraph,
    participant: ExplanationGraph,
) -> None:
    if set(canonical.node_ids()) != set(participant.node_ids()):
        raise AssertionError("participant projection changed node membership")
    if set(canonical.edge_ids()) != set(participant.edge_ids()):
        raise AssertionError("participant projection changed edge membership")
    c_nodes = canonical.node_map()
    p_nodes = participant.node_map()
    for nid, c_node in c_nodes.items():
        p_node = p_nodes[nid]
        if p_node.kind != c_node.kind or p_node.subtype != c_node.subtype:
            raise AssertionError(
                f"participant projection changed semantic identity of {nid}"
            )
        if p_node.lane != c_node.lane:
            raise AssertionError(
                f"participant projection changed lane membership of {nid}"
            )
    c_edges = {e.edge_id: e for e in canonical.edges}
    for edge in participant.edges:
        original = c_edges[edge.edge_id]
        if (
            edge.kind != original.kind
            or edge.source != original.source
            or edge.target != original.target
        ):
            raise AssertionError(
                f"participant projection changed edge {edge.edge_id}"
            )


def build_participant_condition_view(
    experiment_dict: Mapping[str, Any],
    case_dict: Mapping[str, Any],
    blinded_label: str = "A",
    *,
    ordinal: int = 1,
    total: int = 1,
) -> ParticipantConditionView:
    del experiment_dict  # used by callers for preflight; case is self-contained
    label = str(blinded_label or "").strip()
    if label not in BLINDED_LABELS:
        raise ParticipantPresentationError("blinded label must be A, B, or C")
    _require_visualization_data(case_dict)
    blinding = validate_blinding(case_dict.get("blinding"))
    _require_condition_explanations(case_dict)
    true_condition = blinding[label]
    explanation = condition_explanation(case_dict, true_condition)
    try:
        canonical = build_explanation_graph(
            case_dict,
            condition=true_condition,
            detail_level=PARTICIPANT_DETAIL_LEVEL,
        )
    except VisualizationDataError as exc:
        raise ParticipantPresentationError(
            f"canonical graph build failed: {exc}"
        ) from exc
    participant_graph = project_participant_graph(canonical, case_dict)
    try:
        assert_projection_semantics(canonical, participant_graph)
    except AssertionError as exc:
        raise ParticipantPresentationError(str(exc)) from exc
    question = common_question_context(case_dict, ordinal=ordinal, total=total)
    return ParticipantConditionView(
        blinded_label=label,
        heading=blinded_heading(label),
        ordinal=int(ordinal),
        total=int(total),
        ordinal_label=str(question["ordinal_label"]),
        question=question,
        explanation=explanation,
        graph=participant_graph,
        legend=PARTICIPANT_LEGEND,
        availability="ok",
        internal_case_key=_case_identity_key(case_dict),
    )


def render_participant_dot(graph: ExplanationGraph) -> str:
    """Graphviz DOT with opaque identifiers. Canonical ``render_dot`` is unchanged.

    Visible labels come from the already-projected participant graph. Internal
    node ids are remapped to n1, n2, ... only for the DOT identifier namespace
    so SVG title attributes cannot leak versioned artifact refs. Structure,
    labels, and previews are otherwise identical to ``render_dot``.
    """
    nodes_sorted = sorted(graph.nodes, key=lambda n: (n.sort_key, n.node_id))
    id_map = {node.node_id: f"n{index}" for index, node in enumerate(nodes_sorted, start=1)}
    remapped_nodes = [
        GraphNode(
            node_id=id_map[node.node_id],
            kind=node.kind,
            subtype=node.subtype,
            lane=node.lane,
            label=node.label,
            preview=node.preview,
            details=node.details,
            source_ref=None,
            occurrence_id=None,
            style_role=node.style_role,
            sort_key=node.sort_key,
        )
        for node in graph.nodes
    ]
    remapped_edges = [
        GraphEdge(
            edge_id=edge.edge_id,
            source=id_map[edge.source],
            target=id_map[edge.target],
            kind=edge.kind,
            relation=edge.relation,
            label=edge.label,
            style_role=edge.style_role,
            sort_key=edge.sort_key,
            details=edge.details,
        )
        for edge in graph.edges
        if edge.source in id_map and edge.target in id_map
    ]
    render_graph = graph.replace(nodes=remapped_nodes, edges=remapped_edges)
    return render_dot(render_graph)


def structural_display_strings(view: ParticipantConditionView) -> List[str]:
    """Participant STRUCTURAL display strings for jargon audits.

    Excludes teaching/source content previews (learning objectives, evidence,
    knowledge statements, question plans, and question stems).
    """
    strings: List[str] = [
        view.heading,
        view.ordinal_label,
        *PARTICIPANT_UI_HEADINGS,
        *view.legend,
    ]
    for lane in view.graph.lanes:
        strings.append(lane.label)
    for edge in view.graph.edges:
        if edge.label:
            strings.append(edge.label)
    for node in view.graph.nodes:
        strings.append(node.label)
        if node.subtype in _CONTENT_PREVIEW_SUBTYPES:
            continue
        if node.preview:
            strings.append(node.preview)
        for value in (node.details or {}).values():
            if isinstance(value, str) and value:
                strings.append(value)
    payload = view.display_payload()
    strings.append(str(payload.get("heading") or ""))
    strings.append(str(payload.get("ordinal_label") or ""))
    strings.extend(str(item) for item in payload.get("legend") or [])
    strings.extend(str(item) for item in payload.get("headings") or [])
    return [s for s in strings if s]


def forbidden_structural_hits(strings: Sequence[str]) -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    for raw in strings:
        text = str(raw)
        for token in FORBIDDEN_STRUCTURAL_TOKENS:
            if _token_in_structural(text, token):
                hits.append((token, text))
    return hits


def _token_in_structural(text: str, token: str) -> bool:
    if token.endswith("-"):
        return token.lower() in text.lower()
    if token == "Trace":
        return re.search(r"(?<![A-Za-z])Trace(?![A-Za-z])", text) is not None
    if token.isupper() or token[0].isupper() and any(c.islower() for c in token[1:]):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
        return re.search(pattern, text) is not None
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def participant_view_snapshot(view: ParticipantConditionView) -> Dict[str, Any]:
    """Compact developer/test snapshot. Not shown in the participant UI."""
    nodes = []
    for node in sorted(view.graph.nodes, key=lambda n: (n.sort_key, n.node_id)):
        nodes.append(
            {
                "id": node.node_id,
                "kind": node.kind,
                "subtype": node.subtype,
                "lane": node.lane,
                "label": node.label,
                "preview": node.preview,
            }
        )
    edges = []
    for edge in sorted(view.graph.edges, key=lambda e: (e.sort_key, e.edge_id)):
        edges.append(
            {
                "id": edge.edge_id,
                "kind": edge.kind,
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
            }
        )
    return {
        "blinded_label": view.blinded_label,
        "heading": view.heading,
        "ordinal_label": view.ordinal_label,
        "question": {
            "stem": view.question.get("stem"),
            "options": list(view.question.get("options") or []),
            "correct_answer": view.question.get("correct_answer"),
            "factual_decision": view.question.get("factual_decision"),
        },
        "explanation": view.explanation,
        "nodes": nodes,
        "edges": edges,
        "legend": list(view.legend),
        "lane_labels": [
            {"id": ln.lane_id, "label": ln.label}
            for ln in sorted(view.graph.lanes, key=lambda x: (x.order, x.lane_id))
        ],
    }


def graph_has_replay(graph: ExplanationGraph) -> bool:
    if any(n.kind in {KIND_INTERVENTION, KIND_CHECKPOINT, KIND_COMPARISON} for n in graph.nodes):
        return True
    if any(n.subtype in {"noop", "counterfactual"} for n in graph.nodes):
        return True
    if any(n.node_id.startswith("replay::") for n in graph.nodes):
        return True
    kinds = {e.kind for e in graph.edges}
    return bool(
        kinds
        & {
            EDGE_REPLAY_BRANCH,
            EDGE_INTERVENTION_TARGET,
            EDGE_COMPARISON,
        }
    )


def graph_has_provenance(graph: ExplanationGraph) -> bool:
    kinds = {e.kind for e in graph.edges}
    return bool(kinds & {EDGE_PRODUCES, EDGE_CONSUMES, EDGE_DECLARED_REFERENCE})


def log_developer_error(exc: BaseException) -> None:
    logger.error("Participant presentation failed: %s", exc)
