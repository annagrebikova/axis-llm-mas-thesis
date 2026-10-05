"""Streamlit-independent helpers for the researcher explanation-graph explorer.

These helpers must not decide graph semantics, relevance, ranking, or causality.
They only format, validate, and select among values already present in exported
JSON or an ExplanationGraph.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .labels import (
    INFORMATION_ROLE_LABELS,
    SYMBOL_CHANGED,
    SYMBOL_INCONCLUSIVE,
    SYMBOL_NOOP,
    SYMBOL_UNCHANGED,
    effect_symbol,
    effect_word,
    make_preview,
)
from .models import ExplanationGraph, GraphEdge, GraphNode


class ExperimentPayloadError(ValueError):
    """Raised when a JSON file is not an explanation-experiment payload."""


CONDITION_LABELS = {
    "trace": "Trace",
    "trace_provenance": "Trace + Provenance",
    "axis": "AXIS",
}
CONDITION_VALUES = ("trace", "trace_provenance", "axis")
DETAIL_LABELS = {
    "essential": "Essential",
    "relevant": "Relevant",
    "full": "Full",
}
DETAIL_VALUES = ("essential", "relevant", "full")

DETAIL_HELP = {
    "essential": (
        "Compact target-focused explanation. "
        "For AXIS, at most the highest-ranked replay comparisons."
    ),
    "relevant": "Connected target-relevant explanation subgraph.",
    "full": (
        "All information permitted by the selected condition within the "
        "target-specific case slice."
    ),
}

ARTIFACT_BROWSER_SUBTYPES = (
    "LearningObjective",
    "EvidenceChunk",
    "KnowledgeAtom",
    "MCQBlueprint",
    "CandidateMCQ",
    "EvidenceCheckReport",
    "CrossCoVeReport",
    "SupervisorDecision",
    "OutcomeAssessment",
    "DocSection",
)

EXPERIMENT_GLOB = "quiz_explanation_experiment_*.json"
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")

PROVENANCE_NOTE = (
    "Recorded provenance shows documented use/dependency. It does not by "
    "itself imply causal influence."
)
RELATION_NOTE = (
    "declared_reference != recorded provenance; provenance != causal influence."
)

LEGEND_NODES = (
    ("Workflow step", "rounded rectangle"),
    ("Artifact / information", "rectangle / card"),
    ("Target question", "emphasized / double-border card"),
    ("Decision", "diamond"),
    ("Intervention", "octagon"),
    ("NOOP outcome", f"pill with {SYMBOL_NOOP}"),
    ("Counterfactual outcome", "pill"),
    ("Replay comparison", "hexagon / result box"),
)
LEGEND_EDGES = (
    ("solid", "execution / production"),
    ("dashed purple", "recorded provenance / consumption"),
    ("dotted neutral", "declared structural reference"),
    ("dotted orange", "intervention target"),
)
LEGEND_SYMBOLS = (
    (SYMBOL_NOOP, "no-intervention control"),
    (SYMBOL_UNCHANGED, "unchanged"),
    (SYMBOL_CHANGED, "changed"),
    (SYMBOL_INCONCLUSIVE, "inconclusive / unavailable"),
)


def default_search_dirs() -> List[Path]:
    """Shallow search: CWD and the repository `rag/` directory."""
    dirs: List[Path] = []
    cwd = Path.cwd().resolve()
    dirs.append(cwd)
    rag_dir = Path(__file__).resolve().parents[2]
    if rag_dir not in dirs:
        dirs.append(rag_dir)
    return dirs


def find_local_experiment_files(
    search_dirs: Optional[Sequence[Path]] = None,
) -> List[Path]:
    """Return matching experiment JSON files. Non-recursive. Newest first."""
    found: Dict[str, Path] = {}
    for raw in search_dirs if search_dirs is not None else default_search_dirs():
        directory = Path(raw)
        if not directory.is_dir():
            continue
        for path in directory.glob(EXPERIMENT_GLOB):
            if path.is_file():
                found[str(path.resolve())] = path.resolve()
    files = list(found.values())
    files.sort(key=lambda p: (-p.stat().st_mtime, str(p).lower()))
    return files


def parse_experiment_json_text(text: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExperimentPayloadError(f"Malformed JSON: {exc.msg}.") from exc
    return validate_experiment_payload(payload)


def parse_experiment_json_bytes(data: bytes) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExperimentPayloadError("This file is not valid UTF-8 JSON.") from exc
    return parse_experiment_json_text(text)


def validate_experiment_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ExperimentPayloadError(
            "This JSON does not contain an explanation experiment."
        )
    if "cases" not in payload:
        raise ExperimentPayloadError(
            "This JSON does not contain an explanation experiment."
        )
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ExperimentPayloadError(
            "This JSON does not contain an explanation experiment."
        )
    return payload


def case_visualization_issue(case: Mapping[str, Any], index: int = 0) -> Optional[str]:
    key = get_case_key(case, index)
    viz = case.get("visualization_data")
    if not isinstance(viz, Mapping) or not viz:
        return (
            f"Case {key} has no visualization_data. Re-export it with the current "
            "visualization contract."
        )
    schema = viz.get("schema_version")
    if schema != 1:
        return (
            f"Case {key} has unsupported visualization_data.schema_version "
            f"{schema!r}."
        )
    return None


def get_case_key(case: Mapping[str, Any], index: int = 0) -> str:
    ref = case.get("candidate_ref")
    key = _ref_key(ref)
    if key:
        return key
    viz = case.get("visualization_data") if isinstance(case.get("visualization_data"), Mapping) else {}
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


def get_case_display_label(case: Mapping[str, Any], ordinal: int) -> str:
    key = get_case_key(case, max(0, ordinal - 1))
    stem = ""
    question = case.get("question")
    if isinstance(question, Mapping):
        stem = str(question.get("stem") or "")
    preview = make_preview(stem, 60)
    if preview:
        return f"Q{ordinal} · {key} · {preview}"
    return f"Q{ordinal} · {key}"


def list_case_options(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, case in enumerate(payload.get("cases") or []):
        if not isinstance(case, Mapping):
            continue
        rows.append(
            {
                "index": i,
                "key": get_case_key(case, i),
                "label": get_case_display_label(case, i + 1),
                "issue": case_visualization_issue(case, i),
            }
        )
    return rows


def find_case(
    payload: Mapping[str, Any],
    *,
    key: Optional[str] = None,
    index: Optional[int] = None,
) -> Optional[Mapping[str, Any]]:
    cases = [c for c in (payload.get("cases") or []) if isinstance(c, Mapping)]
    if index is not None and 0 <= index < len(cases):
        return cases[index]
    if key:
        for i, case in enumerate(cases):
            if get_case_key(case, i) == key:
                return case
    return None


def resolve_selected_key(
    previous: Optional[str], visible_keys: Sequence[str]
) -> Optional[str]:
    keys = [str(k) for k in visible_keys]
    if previous and previous in keys:
        return previous
    return keys[0] if keys else None


def resolve_selected_node_id(
    previous: Optional[str],
    visible_ids: Sequence[str],
    *,
    fallback_id: Optional[str] = None,
) -> Optional[str]:
    ids = [str(i) for i in visible_ids]
    if previous and previous in ids:
        return previous
    if fallback_id and fallback_id in ids:
        return fallback_id
    return ids[0] if ids else None


def node_display_label(node: GraphNode) -> str:
    parts = [str(node.label or node.kind or "node")]
    preview = make_preview(node.preview or "", 70)
    if preview and preview != parts[0]:
        parts.append(preview)
    ident = node.source_ref or node.node_id
    if ident:
        parts.append(str(ident))
    return " · ".join(parts)


def edge_display_label(edge: GraphEdge, nodes: Mapping[str, GraphNode]) -> str:
    src = nodes.get(edge.source)
    tgt = nodes.get(edge.target)
    src_name = src.label if src is not None else edge.source
    tgt_name = tgt.label if tgt is not None else edge.target
    kind = edge.kind or edge.relation or "edge"
    return f"{src_name} → {tgt_name} ({kind})"


def summarize_graph(graph: ExplanationGraph) -> Dict[str, Any]:
    hidden = dict(graph.hidden_counts or {})
    replay_ids = {
        _replay_id(n.node_id) for n in graph.nodes if _replay_id(n.node_id)
    }
    return {
        "visible_nodes": len(graph.nodes),
        "visible_edges": len(graph.edges),
        "hidden_nodes": int(hidden.get("hidden_node_count") or 0),
        "hidden_edges": int(hidden.get("hidden_edge_count") or 0),
        "hidden_replays": int(
            hidden.get("hidden_replay_count")
            or (graph.metadata or {}).get("hidden_replay_count")
            or 0
        ),
        "replay_units": len(replay_ids),
        "warnings": list(graph.warnings or ()),
        "target_ref": graph.target_ref,
        "selection_scope": (graph.metadata or {}).get("selection_scope"),
        "selected_replay_indices": list(
            (graph.metadata or {}).get("selected_replay_indices") or []
        ),
        "replay_count": (graph.metadata or {}).get("replay_count"),
    }


def download_stem(candidate_key: str, condition: str, detail_level: str) -> str:
    raw = f"{candidate_key}__{condition}__{detail_level}"
    raw = raw.replace(":v", "_v")
    safe = _UNSAFE_FILENAME.sub("_", raw).strip("._-")
    return safe or "explanation_graph"


def download_filename(
    candidate_key: str, condition: str, detail_level: str, extension: str
) -> str:
    ext = str(extension).lstrip(".")
    return f"{download_stem(candidate_key, condition, detail_level)}.{ext}"


def _replay_id(node_id: str) -> Optional[str]:
    if not str(node_id).startswith("replay::"):
        return None
    parts = str(node_id).split("::")
    if len(parts) < 3:
        return None
    return parts[1]


def _replay_role(node_id: str) -> Optional[str]:
    if not str(node_id).startswith("replay::"):
        return None
    parts = str(node_id).split("::")
    if len(parts) < 3:
        return None
    return parts[2]


def visible_replay_units(graph: ExplanationGraph) -> List[Dict[str, Any]]:
    """Group already-selected AXIS replay nodes. Does not re-rank raw JSON."""
    grouped: Dict[str, Dict[str, GraphNode]] = {}
    order: List[str] = []
    for node in graph.nodes:
        rid = _replay_id(node.node_id)
        role = _replay_role(node.node_id)
        if not rid or not role:
            continue
        if rid not in grouped:
            grouped[rid] = {}
            order.append(rid)
        grouped[rid][role] = node

    units: List[Dict[str, Any]] = []
    for rid in order:
        roles = grouped[rid]
        result = roles.get("result")
        details = dict(result.details) if result is not None else {}
        intervention = roles.get("intervention")
        sig = {}
        if intervention is not None and isinstance(
            intervention.details.get("intervention_signature"), Mapping
        ):
            sig = dict(intervention.details.get("intervention_signature") or {})
        elif isinstance(details.get("intervention_signature"), Mapping):
            sig = dict(details.get("intervention_signature") or {})
        index = details.get("comparison_index")
        if index is None and intervention is not None:
            index = intervention.details.get("comparison_index")
        units.append(
            {
                "replay_id": rid,
                "comparison_index": index,
                "roles": {k: v.node_id for k, v in roles.items()},
                "operation": sig.get("operation")
                or (intervention.subtype if intervention is not None else None),
                "target_ref": _ref_key(sig.get("target_ref"))
                or (intervention.source_ref if intervention is not None else None),
                "replacement_ref": _ref_key(sig.get("replacement_ref")),
                "intervention_signature": dict(sig),
                "generated_query": details.get("generated_query"),
                "checkpoint_id": sig.get("checkpoint_id"),
                "checkpoint_alias": sig.get("checkpoint_alias"),
                "evidence_status": details.get("evidence_status"),
                "decision_changed": details.get("decision_changed"),
                "question_content_changed": details.get("question_content_changed"),
                "display_role_label": details.get("display_role_label"),
                "information_role": details.get("information_role"),
                "decision_control_stable": details.get("decision_control_stable"),
                "unstable": details.get("unstable"),
                "invalid": details.get("invalid"),
                "noop_decision": details.get("noop_decision"),
                "counterfactual_decision": details.get("counterfactual_decision"),
                "factual_decision": details.get("factual_decision"),
                "causal_classification": details.get("causal_classification"),
                "information_role_supported": details.get("information_role_supported"),
                "criteria_control_stable": details.get("criteria_control_stable"),
                "subject_content_control_stable": details.get(
                    "subject_content_control_stable"
                ),
                "question_content_comparison": details.get(
                    "question_content_comparison"
                )
                or {},
                "information_role_finding": details.get("information_role_finding")
                or {},
                "effect_line": (
                    f"Question content: {effect_symbol(details.get('question_content_changed'))} "
                    f"{effect_word(details.get('question_content_changed'))}; "
                    f"Acceptance: {effect_symbol(details.get('decision_changed'))} "
                    f"{effect_word(details.get('decision_changed'))}"
                ),
            }
        )
    units.sort(
        key=lambda u: (
            u["comparison_index"] is None,
            int(u["comparison_index"]) if isinstance(u["comparison_index"], int) else 10**6,
            str(u["replay_id"]),
        )
    )
    return units


def artifact_rows(graph: ExplanationGraph) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.kind not in {"artifact", "outcome"}:
            continue
        if node.subtype == "noop" or node.subtype == "counterfactual":
            continue
        rows.append(
            {
                "node_id": node.node_id,
                "type": node.subtype,
                "ref": node.source_ref or "",
                "label": node.label,
                "preview": node.preview,
            }
        )
    rows.sort(key=lambda r: (str(r["type"]), str(r["ref"]), str(r["node_id"])))
    return rows


def edge_rows(graph: ExplanationGraph) -> List[Dict[str, Any]]:
    nodes = graph.node_map()
    rows: List[Dict[str, Any]] = []
    for edge in graph.edges:
        src = nodes.get(edge.source)
        tgt = nodes.get(edge.target)
        rows.append(
            {
                "edge_id": edge.edge_id,
                "kind": edge.kind,
                "relation": edge.relation,
                "label": edge.label,
                "source_id": edge.source,
                "target_id": edge.target,
                "source": node_display_label(src) if src is not None else edge.source,
                "target": node_display_label(tgt) if tgt is not None else edge.target,
                "details": dict(edge.details or {}),
            }
        )
    rows.sort(key=lambda r: (str(r["kind"]), str(r["edge_id"])))
    return rows


def question_view(case: Mapping[str, Any]) -> Dict[str, Any]:
    question = case.get("question") if isinstance(case.get("question"), Mapping) else {}
    options = list(question.get("options") or [])
    return {
        "stem": str(question.get("stem") or ""),
        "options": options,
        "correct_option": question.get("correct_option")
        or question.get("correct_option_id"),
        "factual_decision": case.get("factual_decision"),
        "candidate_key": get_case_key(case, 0),
    }


def experiment_overview(payload: Mapping[str, Any]) -> Dict[str, Any]:
    models = payload.get("models") if isinstance(payload.get("models"), Mapping) else {}

    def _model(name: str) -> Optional[str]:
        block = models.get(name)
        if isinstance(block, Mapping) and block.get("model"):
            return str(block.get("model"))
        return None

    return {
        "run_id": payload.get("run_id"),
        "branch_id": payload.get("branch_id"),
        "question_count": payload.get("question_count"),
        "user_question": payload.get("user_question"),
        "experiment_seed": payload.get("experiment_seed"),
        "warnings": list(payload.get("warnings") or []),
        "models": {
            "workflow": _model("workflow"),
            "cross_cove_verifier": _model("cross_cove_verifier"),
            "axis_interrogation": _model("axis_interrogation"),
            "synthesis": _model("synthesis"),
        },
    }


def condition_text(case: Mapping[str, Any], condition: str) -> Dict[str, Any]:
    conditions = case.get("conditions") if isinstance(case.get("conditions"), Mapping) else {}
    block = conditions.get(condition) if isinstance(conditions.get(condition), Mapping) else {}
    return {
        "condition": condition,
        "label": CONDITION_LABELS.get(condition, condition),
        "explanation": block.get("explanation"),
        "status": block.get("status"),
        "warnings": list(block.get("warnings") or []),
    }


def node_content(node: GraphNode) -> Mapping[str, Any]:
    details = node.details if isinstance(node.details, Mapping) else {}
    content = details.get("content")
    if isinstance(content, Mapping):
        return content
    return details


def _display(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        if not value:
            return "—"
        return ", ".join(_display(v) for v in value)
    if isinstance(value, Mapping):
        key = value.get("key")
        if key:
            return str(key)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def inspector_sections(node: GraphNode) -> List[Dict[str, Any]]:
    """Readable sections derived from GraphNode.details. Never mutates details."""
    subtype = str(node.subtype or "")
    kind = str(node.kind or "")
    content = node_content(node)
    sections: List[Dict[str, Any]] = [
        {
            "title": "Identity",
            "pairs": [
                ("node_id", node.node_id),
                ("kind", kind),
                ("subtype", subtype or "—"),
                ("source_ref", node.source_ref or "—"),
                ("lane", node.lane),
            ],
        }
    ]
    if subtype == "LearningObjective":
        sections.append({"title": "Learning objective", "text": content.get("objective_text") or "", "full": True})
        sections.append(
            {
                "title": "Fields",
                "pairs": [
                    ("lo_id", _display(content.get("lo_id"))),
                    ("section_id", _display(content.get("section_id"))),
                    ("difficulty_level", _display(content.get("difficulty_level"))),
                ],
            }
        )
    elif subtype == "DocSection":
        sections.append({"title": "Title", "text": content.get("title") or "", "full": True})
        sections.append({"title": "Content summary", "text": content.get("content_summary") or "", "full": True})
        sections.append(
            {
                "title": "Fields",
                "pairs": [
                    ("section_id", _display(content.get("section_id"))),
                    ("slide_range", _display(content.get("slide_range"))),
                    ("importance_weight", _display(content.get("importance_weight"))),
                ],
            }
        )
    elif subtype == "EvidenceChunk":
        sections.append({"title": "Content", "text": content.get("content") or "", "full": True})
        sections.append(
            {
                "title": "Metadata",
                "pairs": [
                    ("chunk_id", _display(content.get("chunk_id"))),
                    ("relevance_score", _display(content.get("relevance_score"))),
                    ("quality_score", _display(content.get("quality_score"))),
                    ("relevance_tags", _display(content.get("relevance_tags"))),
                    ("used_in_atoms", _display(content.get("used_in_atoms"))),
                    ("duplicate_of", _display(content.get("duplicate_of"))),
                    ("flags", _display(content.get("flags"))),
                    ("metadata", _display(content.get("metadata"))),
                ],
            }
        )
    elif subtype == "KnowledgeAtom":
        sections.append({"title": "Content", "text": content.get("content") or "", "full": True})
        sections.append(
            {
                "title": "Fields",
                "pairs": [
                    ("atom_id", _display(content.get("atom_id"))),
                    ("atom_type", _display(content.get("atom_type"))),
                    ("section_id", _display(content.get("section_id"))),
                    ("lo_id", _display(content.get("lo_id"))),
                    ("citations", _display(content.get("citations"))),
                ],
            }
        )
    elif subtype == "MCQBlueprint":
        sections.append({"title": "Generation goal", "text": content.get("generation_goal") or "", "full": True})
        sections.append(
            {
                "title": "Fields",
                "pairs": [
                    ("blueprint_id", _display(content.get("blueprint_id"))),
                    ("question_type", _display(content.get("question_type"))),
                    ("difficulty", _display(content.get("difficulty"))),
                    ("bloom_level", _display(content.get("bloom_level"))),
                    ("section_id", _display(content.get("section_id"))),
                    ("lo_id", _display(content.get("lo_id"))),
                    ("target_atom_ids", _display(content.get("target_atom_ids"))),
                    ("evidence_chunk_ids", _display(content.get("evidence_chunk_ids"))),
                    ("status", _display(content.get("status"))),
                ],
            }
        )
    elif subtype == "CandidateMCQ":
        sections.append({"title": "Stem", "text": content.get("stem") or "", "full": True})
        sections.append({"title": "Options", "lines": _option_lines(content), "full": True})
        sections.append({"title": "Answer explanation", "text": content.get("answer_explanation") or "", "full": True})
        ref = node.details.get("ref") if isinstance(node.details.get("ref"), Mapping) else {}
        sections.append(
            {
                "title": "Fields",
                "pairs": [
                    ("correct_option_id", _display(content.get("correct_option_id"))),
                    ("q_id", _display(content.get("q_id"))),
                    ("exact ref", _display(ref.get("key") or node.source_ref)),
                    ("version", _display(ref.get("version"))),
                    ("blueprint_id", _display(content.get("blueprint_id"))),
                    ("status", _display(content.get("status"))),
                    ("revision_count", _display(content.get("revision_count"))),
                    ("flags", _display(content.get("flags"))),
                    ("source_refs", _display(content.get("source_refs"))),
                ],
            }
        )
    elif subtype == "EvidenceCheckReport":
        sections.append(
            {
                "title": "Verification",
                "pairs": [
                    ("supported_by_pdf", _display(content.get("supported_by_pdf"))),
                    ("confidence", _display(content.get("confidence"))),
                    ("stem_supported", _display(content.get("stem_supported"))),
                    ("answer_supported", _display(content.get("answer_supported"))),
                    ("distractors_valid", _display(content.get("distractors_valid"))),
                    ("explanation_supported", _display(content.get("explanation_supported"))),
                    ("candidate_version", _display(content.get("candidate_version"))),
                    ("q_id", _display(content.get("q_id"))),
                    ("unsupported_claims", _display(content.get("unsupported_claims"))),
                    ("required_fixes", _display(content.get("required_fixes"))),
                    ("notes", _display(content.get("notes"))),
                    ("source_refs", _display(content.get("source_refs"))),
                ],
            }
        )
    elif subtype == "CrossCoVeReport":
        sections.append(
            {
                "title": "Independent answer check",
                "pairs": [
                    ("verifier_model", _display(content.get("verifier_model"))),
                    ("generator_answer", _display(content.get("generator_answer"))),
                    ("verifier_answers", _display(content.get("verifier_answers"))),
                    ("majority_answer", _display(content.get("majority_answer"))),
                    ("passed", _display(content.get("passed"))),
                    ("confidence", _display(content.get("confidence"))),
                    ("candidate_version", _display(content.get("candidate_version"))),
                    ("q_id", _display(content.get("q_id"))),
                    ("notes", _display(content.get("notes"))),
                ],
            }
        )
    elif subtype == "SupervisorDecision":
        sections.append(
            {
                "title": "Acceptance decision",
                "pairs": [
                    ("decision", _display(content.get("decision"))),
                    ("reasons", _display(content.get("reasons"))),
                    ("required_fixes", _display(content.get("required_fixes"))),
                    ("candidate_version", _display(content.get("candidate_version"))),
                    ("q_id", _display(content.get("q_id"))),
                    ("assessment_ref", _display(content.get("assessment_ref"))),
                ],
            }
        )
    elif subtype == "OutcomeAssessment":
        criteria = content.get("criteria") or node.details.get("criteria") or []
        lines = []
        if isinstance(criteria, list):
            for item in criteria:
                if isinstance(item, Mapping):
                    lines.append(
                        f"{item.get('name')}: {_display(item.get('value'))}"
                        f"{' (decision-relevant)' if item.get('decision_relevant') else ''}"
                    )
        sections.append({"title": "Criteria", "lines": lines or ["—"], "full": True})
        sections.append(
            {
                "title": "Fields",
                "pairs": [
                    ("target_ref", _display(content.get("target_ref"))),
                    ("notes", _display(content.get("notes"))),
                ],
            }
        )
    elif kind == "occurrence":
        details = node.details if isinstance(node.details, Mapping) else {}
        sections.append(
            {
                "title": "Occurrence",
                "pairs": [
                    ("label", node.label),
                    ("node_id (workflow)", _display(details.get("node_id") or node.subtype)),
                    ("occurrence_id", _display(details.get("occurrence_id") or node.occurrence_id)),
                    ("execution_index", _display(details.get("execution_index"))),
                    ("superstep", _display(details.get("superstep"))),
                    ("checkpoint_before", _display(details.get("checkpoint_before"))),
                    ("checkpoint_after", _display(details.get("checkpoint_after"))),
                    ("error", _display(details.get("error"))),
                ],
            }
        )
    elif kind in {"intervention", "comparison", "checkpoint"} or subtype in {
        "noop",
        "counterfactual",
    }:
        details = node.details if isinstance(node.details, Mapping) else {}
        pairs = [
            ("evidence_status", _display(details.get("evidence_status"))),
            ("decision_changed", _display(details.get("decision_changed"))),
            ("question_content_changed", _display(details.get("question_content_changed"))),
            ("information_role", _display(details.get("information_role"))),
            ("display_role_label", _display(details.get("display_role_label"))),
            ("decision_control_stable", _display(details.get("decision_control_stable"))),
            ("unstable", _display(details.get("unstable"))),
            ("invalid", _display(details.get("invalid"))),
            ("noop_decision", _display(details.get("noop_decision"))),
            ("counterfactual_decision", _display(details.get("counterfactual_decision"))),
            ("factual_decision", _display(details.get("factual_decision"))),
        ]
        sig = details.get("intervention_signature")
        if isinstance(sig, Mapping):
            pairs.extend(
                [
                    ("operation", _display(sig.get("operation"))),
                    ("target_ref", _display(sig.get("target_ref"))),
                    ("replacement_ref", _display(sig.get("replacement_ref"))),
                    ("checkpoint_id", _display(sig.get("checkpoint_id"))),
                    ("checkpoint_alias", _display(sig.get("checkpoint_alias"))),
                ]
            )
        sections.append({"title": "Replay details", "pairs": pairs})
        if details.get("noop_path") or details.get("counterfactual_path"):
            sections.append(
                {
                    "title": "Replay paths (metadata only; not occurrence nodes)",
                    "pairs": [
                        ("noop_path", _display(details.get("noop_path"))),
                        ("counterfactual_path", _display(details.get("counterfactual_path"))),
                    ],
                }
            )
    else:
        sections.append({"title": "Preview", "text": node.preview or "", "full": True})
    return sections


def _option_lines(content: Mapping[str, Any]) -> List[str]:
    correct = str(content.get("correct_option_id") or "")
    lines: List[str] = []
    for opt in content.get("options") or []:
        if not isinstance(opt, Mapping):
            continue
        oid = str(opt.get("option_id") or "")
        text = str(opt.get("text") or "")
        mark = "  ← correct" if oid and oid == correct else ""
        lines.append(f"{oid}. {text}{mark}".strip())
    if not lines:
        lines.append("—")
    return lines


def replay_selection_caption(graph: ExplanationGraph) -> Optional[str]:
    meta = graph.metadata or {}
    selected = list(meta.get("selected_replay_indices") or [])
    hidden = int(meta.get("hidden_replay_count") or 0)
    total = meta.get("replay_count")
    if total is None:
        total = len(selected) + hidden
    try:
        total_i = int(total)
    except (TypeError, ValueError):
        total_i = len(selected)
    if graph.detail_level != "essential":
        return None
    if not selected and total_i == 0:
        return None
    return (
        f"Showing {len(selected)} of {total_i} replay comparisons according to the "
        "deterministic Essential ranking."
    )

MISSING_QUESTION_COMPARISON = (
    "The question components (stem, correct answer, distractors, explanation) "
    "could not be compared for this replay, for example because the question "
    "was not produced in one of the runs. This does not mean the questions "
    "were identical."
)

CHECKPOINT_DEPENDENCE_NOTE = (
    "An intervention is defined by what is changed and at which checkpoint the "
    "replay starts. REMOVE(chunk_X, c6) and REMOVE(chunk_X, c8) are therefore "
    "different tests: if the chunk is removed after its content has already "
    "been turned into knowledge atoms, that content can still reach the "
    "question through the atoms."
)

CONTENT_VS_ACCEPTANCE_NOTES = (
    (
        "Question-content effect",
        "The question was produced again, but its stem, correct answer, "
        "distractors, explanation, or tested concept changed.",
    ),
    (
        "Acceptance effect",
        "The question was produced again but received a different final "
        "decision (e.g. ACCEPT → REVISE or REJECT), or it was not produced "
        "at all. These two cases are shown separately in the comparison.",
    ),
)

CONTENT_VS_ACCEPTANCE_FOOTNOTE = (
    "A question can change and still be accepted: a different version can "
    "pass the same checks. An unchanged decision does not mean an unchanged "
    "question."
)

ROLE_EXPLANATIONS: Dict[str, str] = {
    "focus_shaping": (
        "The intervention changed which concept the question tests "
        "(e.g. from choosing k to updating centroids). The question may still "
        "be accepted, but it now tests something else."
    ),
    "answer_anchoring": (
        "The intervention changed the correct answer. This shows that the "
        "answer depends on the tested information; it does not show which "
        "answer is factually right or that the information was needed for "
        "acceptance."
    ),
    "distractor_shaping": (
        "The intervention changed the wrong answer options (distractors), "
        "while the stem and correct answer stayed largely the same. This may "
        "affect difficulty, which was not assessed here."
    ),
    "formulation_shaping": (
        "The intervention changed how the question or answer is worded. "
        "Whether the meaning stayed the same was not checked."
    ),
    "artifact_level_only": (
        "Changing this artifact changed the replay result, but the role of its "
        "content cannot be isolated: the content may already have been copied "
        "or summarized into later artifacts before this checkpoint. So even "
        "after a REMOVE, the same information can still appear in the final "
        "question. The result describes this artifact at this checkpoint only."
    ),
    "necessary_for_reproduction": (
        "After this removal, the question was no longer produced, although the "
        "control run produced it. This holds for this checkpoint and this "
        "replay. If the removed artifact was the question itself or its "
        "blueprint, the result is expected and says little about which "
        "information the question was based on."
    ),
    "no_demonstrated_role": (
        "An effect may have occurred, but the replay does not show a specific "
        "role of the tested information (focus, answer, distractors, or "
        "wording). This does not mean the information was irrelevant."
    ),
    "no_demonstrated_effect": (
        "No tracked outcome changed: the decision, criteria, and question "
        "content stayed the same in this replay. This does not prove that the "
        "information can never matter."
    ),
    "acceptance_relevant": (
        "The intervention affected the final outcome of the question: either "
        "the decision changed, or the question was not produced at all. Check "
        "the comparison to see which of the two happened."
    ),
    "influential_non_decisive": (
        "The intervention changed the question content, but the final decision "
        "stayed the same. 'Non-decisive' refers only to the decision; the "
        "change can still matter for teaching."
    ),
    "robust_to_tested_information": (
        "With a stable control, removing the tested information before "
        "generation did not change the question. The information was either "
        "not needed or available from another source. This holds for this "
        "removal only."
    ),
    "inconclusive": (
        "The replay does not allow a claim about the role of the tested "
        "information, for example because the control run already differed "
        "from the original run. Any difference is not attributed to the "
        "intervention."
    ),
    "artifact_level_downstream": (
        "Changing this artifact changed the question produced later in the "
        "workflow, but it is unclear whether it affected the concept, answer, "
        "distractors, or wording."
    ),
    "decisive": (
        "The intervention changed the final decision. This label is taken from "
        "the recorded result; the interface does not compute it."
    ),
}

EVIDENCE_STATUS_EXPLANATIONS: Dict[str, str] = {
    "criterion_effect": (
        "At least one tracked criterion value changed (e.g. the number of "
        "evidence items), even if the final decision stayed the same."
    ),
    "decision_effect": (
        "The question received a different final decision than in the control "
        "run (e.g. ACCEPT → REJECT or REVISE). This is a change in the "
        "system's decision, not a judgement of educational quality."
    ),
    "downstream_effect": (
        "The intervention changed later artifacts or whether the question was "
        "produced, but the change cannot be linked to a specific role of the "
        "information."
    ),
    "no_demonstrated_effect": (
        "The comparison was valid, but no change in the decision, criteria, or "
        "later artifacts was observed for this intervention at this checkpoint."
    ),
    "intervention_error": (
        "The intervention or the counterfactual replay failed. This is not a "
        "'no effect' result: the test produced no usable evidence."
    ),
    "noop_error": (
        "The control replay (NOOP) failed, so there is no baseline to compare "
        "the counterfactual result with."
    ),
    "replay_unstable": (
        "The control replay did not reproduce the original run, so the replay "
        "itself already changed the result. Differences in the counterfactual "
        "run are therefore not treated as effects of the intervention."
    ),
    "outcome_unavailable": (
        "An observation needed for the comparison is missing (e.g. the replay "
        "was interrupted), so the comparison cannot be evaluated. This differs "
        "from a completed replay in which the question was not produced."
    ),
}

CONTROL_STABILITY_TRUE = (
    "Stable control: the control replay (NOOP) reproduced the original "
    "decision, criteria, and question content, so differences in the "
    "counterfactual run can be attributed to the intervention. This does not "
    "guarantee that the same effect would occur in a repeated run."
)
CONTROL_STABILITY_FALSE = (
    "Unstable control: the control replay (NOOP) did not reproduce the "
    "original run. Differences in the counterfactual run are not reported as "
    "effects of the intervention."
)
CONTROL_STABILITY_UNKNOWN = (
    "Control stability was not recorded for this replay, so it is unknown "
    "whether differences can be attributed to the intervention."
)
_ROLE_KEY_ALIASES = {
    "no_demonstrated": "no_demonstrated_role",
    "no_demonstrated_effect": "no_demonstrated_effect",
    "artifact_level_only": "artifact_level_only",
    "artifact_level_downstream": "artifact_level_downstream",
    "focus_shaping": "focus_shaping",
    "answer_anchoring": "answer_anchoring",
    "distractor_shaping": "distractor_shaping",
    "formulation_shaping": "formulation_shaping",
    "necessary_for_reproduction": "necessary_for_reproduction",
    "acceptance_relevant": "acceptance_relevant",
    "influential_non_decisive": "influential_non_decisive",
    "robust_to_tested_information": "robust_to_tested_information",
    "inconclusive": "inconclusive",
    "decisive": "decisive",
}


def _norm_compare_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def normalize_role_key(raw: Any) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"^[▲○?\s]+", "", text)
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return _ROLE_KEY_ALIASES.get(slug, slug)


def role_explanation(raw: Any) -> str:
    key = normalize_role_key(raw)
    if not key:
        return ""
    if key in ROLE_EXPLANATIONS:
        return ROLE_EXPLANATIONS[key]
    return ""


def role_glossary() -> List[Tuple[str, str]]:
    """Stable researcher glossary: display label + explanation."""
    order = (
        "focus_shaping",
        "answer_anchoring",
        "distractor_shaping",
        "formulation_shaping",
        "artifact_level_only",
        "necessary_for_reproduction",
        "no_demonstrated_role",
        "acceptance_relevant",
        "influential_non_decisive",
        "robust_to_tested_information",
        "inconclusive",
        "artifact_level_downstream",
        "decisive",
    )
    rows: List[Tuple[str, str]] = []
    for key in order:
        label = INFORMATION_ROLE_LABELS.get(key) or key.replace("_", " ")
        if key == "acceptance_relevant":
            label = "Acceptance-relevant"
        rows.append((label, ROLE_EXPLANATIONS[key]))
    return rows


def evidence_status_explanation(status: Any) -> str:
    key = str(status or "").strip()
    if not key:
        return ""
    return EVIDENCE_STATUS_EXPLANATIONS.get(key, "")


def control_stability_explanation(value: Any) -> str:
    if value is True:
        return CONTROL_STABILITY_TRUE
    if value is False:
        return CONTROL_STABILITY_FALSE
    return CONTROL_STABILITY_UNKNOWN


def _qcc_details(qcc: Any) -> Dict[str, Any]:
    if not isinstance(qcc, Mapping):
        return {}
    details = qcc.get("details")
    if isinstance(details, Mapping) and (
        "factual" in details
        or "noop" in details
        or "counterfactual" in details
        or "stem_changed" in details
        or "stem_effect" in details
    ):
        return dict(details)
    return dict(qcc)


def extract_mcq_question_view(block: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(block, Mapping):
        return None
    stem = block.get("stem")
    correct = block.get("correct_answer")
    distractors = [
        str(item)
        for item in (block.get("distractors") or [])
        if item not in (None, "")
    ]
    explanation = block.get("answer_explanation")
    if stem in (None, "") and correct in (None, "") and not distractors:
        return None
    return {
        "stem": "" if stem is None else str(stem),
        "correct_answer": "" if correct is None else str(correct),
        "distractors": distractors,
        "answer_explanation": "" if explanation is None else str(explanation),
    }


def mcq_views_equal(left: Optional[Mapping[str, Any]], right: Optional[Mapping[str, Any]]) -> bool:
    if left is None or right is None:
        return left is right
    if _norm_compare_text(left.get("stem")) != _norm_compare_text(right.get("stem")):
        return False
    if _norm_compare_text(left.get("correct_answer")) != _norm_compare_text(
        right.get("correct_answer")
    ):
        return False
    left_d = sorted(_norm_compare_text(x) for x in (left.get("distractors") or []))
    right_d = sorted(_norm_compare_text(x) for x in (right.get("distractors") or []))
    return left_d == right_d


def extract_replay_question_views(
    qcc: Any,
) -> Optional[Dict[str, Optional[Dict[str, Any]]]]:
    details = _qcc_details(qcc)
    views = {
        "factual": extract_mcq_question_view(details.get("factual")),
        "noop": extract_mcq_question_view(details.get("noop")),
        "counterfactual": extract_mcq_question_view(details.get("counterfactual")),
    }
    if views["factual"] is None and views["noop"] is None and views["counterfactual"] is None:
        return None
    return views


def axis_raw_replay_comparisons(case: Optional[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if not isinstance(case, Mapping):
        return []
    conditions = case.get("conditions") if isinstance(case.get("conditions"), Mapping) else {}
    axis = conditions.get("axis") if isinstance(conditions.get("axis"), Mapping) else {}
    raw = axis.get("raw_evidence") if isinstance(axis.get("raw_evidence"), Mapping) else {}
    rows = raw.get("replay_comparisons") or []
    return [row for row in rows if isinstance(row, Mapping)]


def resolve_replay_question_comparison(
    unit: Mapping[str, Any],
    case: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    qcc = unit.get("question_content_comparison")
    if extract_replay_question_views(qcc) is not None:
        return qcc if isinstance(qcc, Mapping) else {}
    rows = axis_raw_replay_comparisons(case)
    index = unit.get("comparison_index")
    if isinstance(index, int) and 0 <= index < len(rows):
        found = rows[index].get("question_content_comparison") or {}
        if isinstance(found, Mapping):
            return found
    return qcc if isinstance(qcc, Mapping) else {}


def _tri_state_change_word(flag: Any, effect: Any) -> str:
    text = str(effect or "").strip().lower()
    if text in {"changed", "unchanged", "inconclusive", "unavailable"}:
        return text
    if flag is True:
        return "changed"
    if flag is False:
        return "unchanged"
    return "unavailable"


def _compare_field_change(
    left: Optional[Mapping[str, Any]],
    right: Optional[Mapping[str, Any]],
    field: str,
) -> str:
    if left is None or right is None:
        return "unavailable"
    if field == "distractors":
        left_d = sorted(_norm_compare_text(x) for x in (left.get("distractors") or []))
        right_d = sorted(_norm_compare_text(x) for x in (right.get("distractors") or []))
        return "changed" if left_d != right_d else "unchanged"
    left_v = _norm_compare_text(left.get(field))
    right_v = _norm_compare_text(right.get(field))
    if not left_v and not right_v:
        return "unavailable"
    return "changed" if left_v != right_v else "unchanged"


def question_change_summary(
    qcc: Any,
    *,
    decision_changed: Any = None,
    noop_decision: Any = None,
    counterfactual_decision: Any = None,
) -> List[Tuple[str, str]]:
    """Deterministic stem/answer/distractor/acceptance change rows."""
    details = _qcc_details(qcc)
    views = extract_replay_question_views(qcc) or {}
    control = views.get("noop") or views.get("factual")
    counterfactual = views.get("counterfactual")

    stem = _tri_state_change_word(
        details.get("stem_changed"), details.get("stem_effect")
    )
    correct = _tri_state_change_word(
        details.get("correct_answer_changed"), details.get("correct_answer_effect")
    )
    distractors = _tri_state_change_word(
        details.get("distractors_changed"), details.get("distractor_effect")
    )
    if stem == "unavailable":
        stem = _compare_field_change(control, counterfactual, "stem")
    if correct == "unavailable":
        correct = _compare_field_change(control, counterfactual, "correct_answer")
    if distractors == "unavailable":
        distractors = _compare_field_change(control, counterfactual, "distractors")

    if decision_changed is True:
        acceptance = "changed"
    elif decision_changed is False:
        acceptance = "unchanged"
    elif noop_decision not in (None, "") and counterfactual_decision not in (None, ""):
        acceptance = (
            "changed"
            if str(noop_decision) != str(counterfactual_decision)
            else "unchanged"
        )
    else:
        acceptance = "unavailable"
    return [
        ("Stem", stem),
        ("Correct answer", correct),
        ("Distractors", distractors),
        ("Acceptance", acceptance),
    ]


def replay_meaning_sentences(unit: Mapping[str, Any]) -> List[str]:
    """2–4 static sentences from exported fields. No new causal inference."""
    lines: List[str] = []
    role_text = role_explanation(
        unit.get("information_role") or unit.get("display_role_label")
    )
    if role_text:
        lines.append(role_text)
    causal = str(unit.get("causal_classification") or "").strip()
    role_key = normalize_role_key(unit.get("information_role"))
    if causal and normalize_role_key(causal) != role_key:
        extra = role_explanation(causal)
        if extra and extra not in lines:
            lines.append(extra)
    status_text = evidence_status_explanation(unit.get("evidence_status"))
    if status_text and status_text not in lines:
        lines.append(status_text)
    if (
        unit.get("question_content_changed") is True
        and unit.get("decision_changed") is False
    ):
        lines.append(CONTENT_VS_ACCEPTANCE_FOOTNOTE)
    if not lines:
        lines.append(control_stability_explanation(unit.get("decision_control_stable")))
    return lines[:4]


def replay_selection_explanation(graph: ExplanationGraph) -> str:
    meta = graph.metadata or {}
    selected = list(meta.get("selected_replay_indices") or [])
    hidden = int(meta.get("hidden_replay_count") or 0)
    total = meta.get("replay_count")
    ranking = list(meta.get("replay_ranking") or [])
    try:
        total_i = int(total) if total is not None else len(selected) + hidden
    except (TypeError, ValueError):
        total_i = len(selected) + hidden
    if total_i == len(selected) and hidden == 0 and total_i > 0:
        return "All available replay units are shown; no replay was hidden."
    if hidden > 0 or (total_i > 0 and len(selected) < total_i):
        return (
            "The graph builder deterministically selected the most relevant "
            "replay units for the current detail level. This ranking is a "
            "presentation/relevance ranking, not a causal-strength score."
        )
    if not ranking and not selected:
        return (
            "No ranking metadata is present on this graph. No causal ranking "
            "is implied."
        )
    return (
        "Replay selection follows the graph builder's deterministic presentation "
        "ranking. It is not a causal-strength score."
    )
