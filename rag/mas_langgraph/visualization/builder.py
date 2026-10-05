"""Build an ExplanationGraph from exported experiment/case JSON only."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .labels import (
    SYMBOL_INCONCLUSIVE,
    SYMBOL_NOOP,
    artifact_label,
    artifact_preview,
    effect_symbol,
    format_decision_content_line,
    information_role_label,
    lane_label,
    occurrence_label,
    strip_trace_mcq_content,
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
    GRAPH_SCHEMA_VERSION,
    KIND_ARTIFACT,
    KIND_CHECKPOINT,
    KIND_COMPARISON,
    KIND_INTERVENTION,
    KIND_OCCURRENCE,
    KIND_OUTCOME,
    LANE_ARTIFACTS,
    LANE_COMPARISON,
    LANE_COUNTERFACTUAL,
    LANE_FACTUAL,
    LANE_NOOP,
    LANE_ORDER,
    VALID_CONDITIONS,
    VALID_DETAIL_LEVELS,
    ExplanationGraph,
    GraphEdge,
    GraphLane,
    GraphNode,
    VisualizationDataError,
    json_clone,
)
from .relevance import (
    apply_detail_level,
    artifact_node_id,
    occurrence_node_id,
    unique_replay_ids,
)
from .styles import (
    EDGE_KIND_TO_ROLE,
    ROLE_ARTIFACT,
    ROLE_CHECKPOINT,
    ROLE_COMPARISON_CHANGED,
    ROLE_COMPARISON_INCONCLUSIVE,
    ROLE_COMPARISON_SAME,
    ROLE_COUNTERFACTUAL,
    ROLE_DECISION,
    ROLE_FACTUAL_OCCURRENCE,
    ROLE_INTERVENTION,
    ROLE_NOOP,
    ROLE_TARGET_QUESTION,
    ROLE_WARNING,
    lane_order_index,
)
INVALID_EVIDENCE_STATUSES = frozenset(
    {
        "noop_error",
        "replay_unstable",
        "intervention_error",
        "outcome_unavailable",
    }
)

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

TRACE_OCCURRENCE_DETAIL_KEYS = (
    "occurrence_id",
    "node_id",
    "occurrence_index",
    "execution_index",
    "superstep",
    "branch_id",
    "checkpoint_before",
    "checkpoint_after",
    "error",
    "has_state_delta",
)


def build_explanation_graph(
    case_dict: Mapping[str, Any],
    condition: str = CONDITION_TRACE,
    detail_level: str = "full",
) -> ExplanationGraph:
    """Construct a canonical explanation graph from exported case JSON."""
    if not isinstance(case_dict, Mapping):
        raise VisualizationDataError("case_dict must be a mapping of exported case JSON")
    condition_key = str(condition or "").strip()
    if condition_key not in VALID_CONDITIONS:
        raise VisualizationDataError(
            f"Unsupported condition {condition!r}. "
            f"Expected one of: {', '.join(sorted(VALID_CONDITIONS))}"
        )
    level_key = str(detail_level or "").strip()
    if level_key not in VALID_DETAIL_LEVELS:
        raise VisualizationDataError(
            f"Unsupported detail_level {detail_level!r}. "
            f"Expected one of: {', '.join(sorted(VALID_DETAIL_LEVELS))}"
        )

    viz = case_dict.get("visualization_data")
    if not isinstance(viz, Mapping) or not viz:
        raise VisualizationDataError(
            "visualization_data is absent. The graph builder requires the "
            "exported researcher visualization payload."
        )
    schema = viz.get("schema_version")
    if schema != 1:
        raise VisualizationDataError(
            f"Unsupported visualization_data.schema_version {schema!r}. Expected 1."
        )

    full = _build_full_graph(case_dict, viz, condition_key)
    return apply_detail_level(full, level_key, case_dict=case_dict)


def _build_full_graph(
    case_dict: Mapping[str, Any],
    viz: Mapping[str, Any],
    condition: str,
) -> ExplanationGraph:
    target = viz.get("target") if isinstance(viz.get("target"), Mapping) else {}
    target_ref = _ref_key(target.get("candidate_ref")) or _ref_key(
        case_dict.get("candidate_ref")
    )
    occurrences = _sorted_occurrences(viz.get("occurrences") or [])
    artifacts = list(viz.get("artifacts") or [])
    artifacts.sort(key=_artifact_sort)
    relations = list(viz.get("provenance_relations") or [])
    warnings: List[str] = []

    nodes: Dict[str, GraphNode] = {}
    edges: Dict[str, GraphEdge] = {}

    include_artifacts = condition in {CONDITION_TRACE_PROVENANCE, CONDITION_AXIS}
    include_provenance = include_artifacts
    include_axis = condition == CONDITION_AXIS

    _add_occurrence_nodes(nodes, occurrences, include_bindings=include_provenance)
    _add_control_flow_edges(edges, occurrences)

    target_artifact = _find_artifact_by_key(artifacts, target_ref)
    decision_key = _ref_key(target.get("decision_ref")) or _ref_key(
        case_dict.get("decision_ref")
    )
    decision_artifact = _find_artifact_by_key(artifacts, decision_key)

    if include_artifacts:
        for art in artifacts:
            node = _artifact_node(
                art,
                is_target=_artifact_key(art) == target_ref,
                is_decision=_artifact_key(art) == decision_key,
                lane=LANE_ARTIFACTS,
            )
            if node is not None:
                nodes[node.node_id] = node
        _add_provenance_edges(nodes, edges, relations, occurrences, warnings)
        _add_declared_reference_edges(nodes, edges, artifacts)
    else:
        if target_artifact is not None:
            node = _artifact_node(
                target_artifact,
                is_target=True,
                is_decision=False,
                lane=LANE_FACTUAL,
                trace_safe=True,
            )
            if node is not None:
                nodes[node.node_id] = node
        if decision_artifact is not None:
            node = _artifact_node(
                decision_artifact,
                is_target=False,
                is_decision=True,
                lane=LANE_FACTUAL,
                trace_safe=True,
            )
            if node is not None:
                nodes[node.node_id] = node
        elif target.get("factual_decision") not in (None, ""):
            node = _synthetic_decision_node(
                str(target.get("factual_decision")),
                decision_key or "factual_decision",
            )
            nodes[node.node_id] = node

    comparisons = _axis_comparisons(case_dict) if include_axis else []
    replay_ids: List[str] = []
    if include_axis:
        replay_ids = _add_replay_units(
            nodes,
            edges,
            comparisons,
            artifacts,
            target_ref=target_ref,
            warnings=warnings,
        )

    used_lanes = {n.lane for n in nodes.values()}
    lanes = tuple(
        GraphLane(lane_id=lid, label=lane_label(lid), order=lane_order_index(lid))
        for lid in LANE_ORDER
        if lid in used_lanes
    )

    metadata = {
        "selection_scope": "target_case_slice",
        "schema_source": "visualization_data",
        "condition": condition,
        "target_ref": target_ref,
        "target": json_clone(dict(target)),
        "occurrence_order": [str(o.get("occurrence_id") or "") for o in occurrences],
        "replay_ids": replay_ids,
        "replay_count": len(comparisons),
        "selected_replay_indices": list(range(len(comparisons))) if include_axis else [],
        "hidden_replay_count": 0,
        "replay_ranking": [],
    }

    graph = ExplanationGraph(
        schema_version=GRAPH_SCHEMA_VERSION,
        condition=condition,
        detail_level="full",
        target_ref=target_ref,
        nodes=tuple(sorted(nodes.values(), key=lambda n: (n.sort_key, n.node_id))),
        edges=tuple(sorted(edges.values(), key=lambda e: (e.sort_key, e.edge_id))),
        lanes=lanes,
        metadata=metadata,
        hidden_counts={
            "hidden_node_count": 0,
            "hidden_edge_count": 0,
            "hidden_replay_count": 0,
        },
        warnings=tuple(warnings),
    )
    return graph


def _ref_key(ref: Any) -> Optional[str]:
    if ref is None:
        return None
    if isinstance(ref, str):
        text = ref.strip()
        return text or None
    if isinstance(ref, Mapping):
        key = ref.get("key")
        if key:
            return str(key)
        logical = str(ref.get("logical_id") or "")
        version = ref.get("version")
        if logical and version is not None:
            return f"{logical}:v{int(version)}"
        if logical:
            return logical
    return None


def _artifact_key(art: Mapping[str, Any]) -> str:
    return _ref_key((art or {}).get("ref")) or ""


def _artifact_sort(art: Mapping[str, Any]) -> Tuple[Any, ...]:
    ref = art.get("ref") if isinstance(art.get("ref"), Mapping) else {}
    return (
        str(art.get("artifact_type") or ""),
        str(ref.get("logical_id") or ""),
        int(ref.get("version") or 0),
        _artifact_key(art),
    )


def _sorted_occurrences(raw: Sequence[Any]) -> List[Dict[str, Any]]:
    occs: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            occs.append(dict(item))
    occs.sort(
        key=lambda o: (
            int(o.get("execution_index") or 0),
            str(o.get("occurrence_id") or ""),
        )
    )
    return occs


def _add_occurrence_nodes(
    nodes: Dict[str, GraphNode],
    occurrences: Sequence[Mapping[str, Any]],
    *,
    include_bindings: bool,
) -> None:
    for occ in occurrences:
        oid = str(occ.get("occurrence_id") or "")
        if not oid:
            continue
        node_id = occ.get("node_id") or ""
        exec_i = int(occ.get("execution_index") or 0)
        if include_bindings:
            details = json_clone(dict(occ))
        else:
            details = {
                k: json_clone(occ.get(k))
                for k in TRACE_OCCURRENCE_DETAIL_KEYS
                if k in occ
            }
        nodes[occurrence_node_id(oid)] = GraphNode(
            node_id=occurrence_node_id(oid),
            kind=KIND_OCCURRENCE,
            subtype=str(node_id),
            lane=LANE_FACTUAL,
            label=occurrence_label(str(node_id)),
            preview=occurrence_label(str(node_id)),
            details=details,
            source_ref=oid,
            occurrence_id=oid,
            style_role=ROLE_FACTUAL_OCCURRENCE,
            sort_key=(lane_order_index(LANE_FACTUAL), 0, exec_i, oid),
        )


def _add_control_flow_edges(
    edges: Dict[str, GraphEdge],
    occurrences: Sequence[Mapping[str, Any]],
    *,
    skipped: Optional[Sequence[int]] = None,
) -> None:
    ids = [str(o.get("occurrence_id") or "") for o in occurrences]
    ids = [i for i in ids if i]
    skip_vals = list(skipped or [0] * max(0, len(ids) - 1))
    for i in range(len(ids) - 1):
        src = occurrence_node_id(ids[i])
        tgt = occurrence_node_id(ids[i + 1])
        n_skip = int(skip_vals[i]) if i < len(skip_vals) else 0
        label = f"... {n_skip} steps ..." if n_skip > 0 else ""
        edge_id = f"control_flow::{ids[i]}::{ids[i + 1]}"
        edges[edge_id] = GraphEdge(
            edge_id=edge_id,
            source=src,
            target=tgt,
            kind=EDGE_CONTROL_FLOW,
            relation=EDGE_CONTROL_FLOW,
            label=label,
            style_role=EDGE_KIND_TO_ROLE[EDGE_CONTROL_FLOW],
            sort_key=(0, ids[i], ids[i + 1]),
            details={"skipped_occurrence_count": n_skip},
        )


def _artifact_node(
    art: Mapping[str, Any],
    *,
    is_target: bool,
    is_decision: bool,
    lane: str,
    trace_safe: bool = False,
) -> Optional[GraphNode]:
    key = _artifact_key(art)
    if not key:
        return None
    atype = str(art.get("artifact_type") or "")
    content = art.get("content") if isinstance(art.get("content"), Mapping) else {}
    content = json_clone(dict(content))
    if trace_safe and atype == "CandidateMCQ":
        content = dict(strip_trace_mcq_content(content))
    elif trace_safe and atype == "SupervisorDecision":
        content = {
            "q_id": content.get("q_id"),
            "candidate_version": content.get("candidate_version"),
            "decision": content.get("decision"),
            "reasons": list(content.get("reasons") or []),
            "required_fixes": list(content.get("required_fixes") or []),
        }
    details = {
        "ref": json_clone(art.get("ref") or {}),
        "artifact_type": atype,
        "branch_id": art.get("branch_id"),
        "content": content,
    }
    if not trace_safe and art.get("produced_by_occurrence_id"):
        details["produced_by_occurrence_id"] = art.get("produced_by_occurrence_id")

    if is_target and atype == "CandidateMCQ":
        style = ROLE_TARGET_QUESTION
        kind = KIND_ARTIFACT
    elif is_decision or atype == "SupervisorDecision":
        style = ROLE_DECISION
        kind = KIND_OUTCOME
    else:
        style = ROLE_ARTIFACT
        kind = KIND_ARTIFACT

    logical = ""
    version = 0
    ref = art.get("ref") if isinstance(art.get("ref"), Mapping) else {}
    logical = str(ref.get("logical_id") or "")
    version = int(ref.get("version") or 0)
    return GraphNode(
        node_id=artifact_node_id(key),
        kind=kind,
        subtype=atype,
        lane=lane,
        label=artifact_label(atype),
        preview=artifact_preview(atype, content),
        details=details,
        source_ref=key,
        occurrence_id=None if trace_safe else art.get("produced_by_occurrence_id"),
        style_role=style,
        sort_key=(lane_order_index(lane), 1, atype, logical, version, key),
    )


def _synthetic_decision_node(decision: str, key: str) -> GraphNode:
    nid = f"outcome::{key}"
    return GraphNode(
        node_id=nid,
        kind=KIND_OUTCOME,
        subtype="SupervisorDecision",
        lane=LANE_FACTUAL,
        label=artifact_label("SupervisorDecision"),
        preview=str(decision),
        details={"decision": decision, "source": "visualization_data.target"},
        source_ref=key,
        occurrence_id=None,
        style_role=ROLE_DECISION,
        sort_key=(lane_order_index(LANE_FACTUAL), 2, key),
    )


def _find_artifact_by_key(
    artifacts: Sequence[Mapping[str, Any]], key: Optional[str]
) -> Optional[Mapping[str, Any]]:
    if not key:
        return None
    for art in artifacts:
        if _artifact_key(art) == key:
            return art
    return None


def _index_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Map logical ids / domain ids to artifact node ids."""
    index: Dict[str, List[str]] = {}

    def add(token: Any, node_id: str) -> None:
        if token in (None, ""):
            return
        index.setdefault(str(token), [])
        if node_id not in index[str(token)]:
            index[str(token)].append(node_id)

    for art in artifacts:
        key = _artifact_key(art)
        if not key:
            continue
        nid = artifact_node_id(key)
        ref = art.get("ref") if isinstance(art.get("ref"), Mapping) else {}
        content = art.get("content") if isinstance(art.get("content"), Mapping) else {}
        atype = str(art.get("artifact_type") or "")
        add(key, nid)
        add(ref.get("logical_id"), nid)
        add(content.get("lo_id"), nid)
        add(content.get("chunk_id"), nid)
        add(content.get("atom_id"), nid)
        add(content.get("blueprint_id"), nid)
        add(content.get("q_id"), nid)
        if atype == "LearningObjective":
            add(content.get("lo_id") or ref.get("logical_id"), nid)
        if atype == "EvidenceChunk":
            add(content.get("chunk_id") or ref.get("logical_id"), nid)
        if atype == "KnowledgeAtom":
            add(content.get("atom_id") or ref.get("logical_id"), nid)
        if atype == "MCQBlueprint":
            add(content.get("blueprint_id") or ref.get("logical_id"), nid)
        if atype == "CandidateMCQ":
            add(content.get("q_id") or ref.get("logical_id"), nid)
    return index


def _lookup_ids(index: Mapping[str, List[str]], token: Any) -> List[str]:
    if token in (None, ""):
        return []
    return list(index.get(str(token), []))


def _add_declared_reference_edges(
    nodes: Dict[str, GraphNode],
    edges: Dict[str, GraphEdge],
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    index = _index_artifacts(artifacts)
    for art in artifacts:
        src_key = _artifact_key(art)
        if not src_key:
            continue
        src = artifact_node_id(src_key)
        if src not in nodes:
            continue
        content = art.get("content") if isinstance(art.get("content"), Mapping) else {}
        atype = str(art.get("artifact_type") or "")
        refs: List[Tuple[str, Any]] = []
        if atype == "CandidateMCQ":
            refs.append(("blueprint_id", content.get("blueprint_id")))
        elif atype == "MCQBlueprint":
            refs.append(("lo_id", content.get("lo_id")))
            for atom_id in content.get("target_atom_ids") or []:
                refs.append(("target_atom_ids", atom_id))
            for chunk_id in content.get("evidence_chunk_ids") or []:
                refs.append(("evidence_chunk_ids", chunk_id))
        elif atype == "KnowledgeAtom":
            refs.append(("lo_id", content.get("lo_id")))
            for cit in content.get("citations") or []:
                if isinstance(cit, Mapping):
                    refs.append(("citations", cit.get("chunk_id")))
        elif atype == "EvidenceChunk":
            for atom_id in content.get("used_in_atoms") or []:
                refs.append(("used_in_atoms", atom_id))
        elif atype == "SupervisorDecision":
            q_id = content.get("q_id")
            version = content.get("candidate_version")
            if q_id not in (None, "") and version is not None:
                refs.append(("candidate_version", f"{q_id}:v{int(version)}"))
            aref = _ref_key(content.get("assessment_ref"))
            if aref:
                refs.append(("assessment_ref", aref))
        elif atype in {"EvidenceCheckReport", "CrossCoVeReport"}:
            q_id = content.get("q_id")
            version = content.get("candidate_version")
            if q_id not in (None, "") and version is not None:
                refs.append(("candidate_version", f"{q_id}:v{int(version)}"))
        elif atype == "OutcomeAssessment":
            tref = _ref_key(content.get("target_ref"))
            if tref:
                refs.append(("target_ref", tref))

        for field, token in refs:
            for tgt in _lookup_ids(index, token):
                if tgt == src or tgt not in nodes:
                    continue
                _put_edge(
                    edges,
                    edge_id=f"declared_reference::{src_key}::{tgt}::{field}",
                    source=src,
                    target=tgt,
                    kind=EDGE_DECLARED_REFERENCE,
                    relation=field,
                    label="",
                    sort_key=(4, src, tgt, field),
                    details={"field": field, "token": str(token)},
                )


def _add_provenance_edges(
    nodes: Dict[str, GraphNode],
    edges: Dict[str, GraphEdge],
    relations: Sequence[Any],
    occurrences: Sequence[Mapping[str, Any]],
    warnings: List[str],
) -> None:
    recorded: Set[Tuple[str, str, str]] = set()
    for rel in relations:
        if not isinstance(rel, Mapping):
            continue
        art_key = _ref_key(rel.get("artifact_ref"))
        producer = str(rel.get("producer_occurrence_id") or "")
        consumer = str(rel.get("consumer_occurrence_id") or "")
        relation = str(rel.get("relation") or "")
        if art_key and producer:
            src = occurrence_node_id(producer)
            tgt = artifact_node_id(art_key)
            if src in nodes and tgt in nodes:
                _put_edge(
                    edges,
                    edge_id=f"produces::{producer}::{art_key}",
                    source=src,
                    target=tgt,
                    kind=EDGE_PRODUCES,
                    relation=relation or EDGE_PRODUCES,
                    label="",
                    sort_key=(2, producer, art_key),
                    details={"source": "provenance_relations"},
                )
                recorded.add(("produces", producer, art_key))
        if art_key and consumer:
            src = artifact_node_id(art_key)
            tgt = occurrence_node_id(consumer)
            if src in nodes and tgt in nodes:
                _put_edge(
                    edges,
                    edge_id=f"consumes::{art_key}::{consumer}",
                    source=src,
                    target=tgt,
                    kind=EDGE_CONSUMES,
                    relation=relation or EDGE_CONSUMES,
                    label="",
                    sort_key=(3, art_key, consumer),
                    details={"source": "provenance_relations"},
                )
                recorded.add(("consumes", art_key, consumer))

    for occ in occurrences:
        oid = str(occ.get("occurrence_id") or "")
        if not oid or occurrence_node_id(oid) not in nodes:
            continue
        for pref in occ.get("produced_artifacts") or []:
            key = _ref_key(pref)
            if not key or artifact_node_id(key) not in nodes:
                continue
            if ("produces", oid, key) in recorded:
                continue
            _put_edge(
                edges,
                edge_id=f"produces::{oid}::{key}",
                source=occurrence_node_id(oid),
                target=artifact_node_id(key),
                kind=EDGE_PRODUCES,
                relation=EDGE_PRODUCES,
                label="",
                sort_key=(2, oid, key),
                details={"source": "occurrence.produced_artifacts"},
            )
            recorded.add(("produces", oid, key))
        for binding in occ.get("input_bindings") or []:
            if not isinstance(binding, Mapping):
                continue
            key = _ref_key(binding.get("artifact_ref"))
            relation = str(binding.get("relation") or EDGE_CONSUMES)
            if not key or artifact_node_id(key) not in nodes:
                continue
            if ("consumes", key, oid) in recorded:
                continue
            _put_edge(
                edges,
                edge_id=f"consumes::{key}::{oid}",
                source=artifact_node_id(key),
                target=occurrence_node_id(oid),
                kind=EDGE_CONSUMES,
                relation=relation,
                label="",
                sort_key=(3, key, oid),
                details={"source": "occurrence.input_bindings"},
            )
            recorded.add(("consumes", key, oid))


def _put_edge(
    edges: Dict[str, GraphEdge],
    *,
    edge_id: str,
    source: str,
    target: str,
    kind: str,
    relation: str,
    label: str,
    sort_key: Tuple[Any, ...],
    details: Optional[Mapping[str, Any]] = None,
    style_role: Optional[str] = None,
) -> None:
    if source == target:
        return
    if any(token in str(relation).lower() for token in ("cause", "influenc", "determin")):
        relation = kind
        label = ""
    edges[edge_id] = GraphEdge(
        edge_id=edge_id,
        source=source,
        target=target,
        kind=kind,
        relation=relation,
        label=label,
        style_role=style_role or EDGE_KIND_TO_ROLE.get(kind, ROLE_ARTIFACT),
        sort_key=sort_key,
        details=dict(details or {}),
    )


def _axis_comparisons(case_dict: Mapping[str, Any]) -> List[Dict[str, Any]]:
    conditions = case_dict.get("conditions")
    if not isinstance(conditions, Mapping):
        return []
    axis = conditions.get("axis")
    if not isinstance(axis, Mapping):
        return []
    raw = axis.get("raw_evidence")
    if not isinstance(raw, Mapping):
        return []
    rows = raw.get("replay_comparisons") or []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def replay_node_id(replay_id: str, role: str) -> str:
    return f"replay::{replay_id}::{role}"


def _add_replay_units(
    nodes: Dict[str, GraphNode],
    edges: Dict[str, GraphEdge],
    comparisons: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
    *,
    target_ref: Optional[str],
    warnings: List[str],
) -> List[str]:
    ids = unique_replay_ids(comparisons)
    artifact_keys = {_artifact_key(a) for a in artifacts}
    for index, (replay_id, cmp) in enumerate(zip(ids, comparisons)):
        facts = replay_display_facts(cmp)
        sig = cmp.get("intervention_signature") if isinstance(cmp.get("intervention_signature"), Mapping) else {}
        target_key = _ref_key(sig.get("target_ref"))
        ckpt_id = sig.get("checkpoint_id") or sig.get("checkpoint_alias")
        op = str(sig.get("operation") or "INTERVENTION")

        intervention = GraphNode(
            node_id=replay_node_id(replay_id, "intervention"),
            kind=KIND_INTERVENTION,
            subtype=str(op),
            lane=LANE_COUNTERFACTUAL,
            label="Intervention",
            preview=f"{op} {target_key or ''}".strip(),
            details={
                "comparison_index": index,
                "replay_id": replay_id,
                "intervention_signature": json_clone(dict(sig)),
                "noop_path": list(cmp.get("noop_path") or []),
                "counterfactual_path": list(cmp.get("counterfactual_path") or []),
            },
            source_ref=target_key,
            occurrence_id=None,
            style_role=ROLE_INTERVENTION,
            sort_key=(lane_order_index(LANE_COUNTERFACTUAL), 0, index, replay_id),
        )
        nodes[intervention.node_id] = intervention

        noop_decision = facts["noop_decision"]
        cf_decision = facts["cf_decision"]
        noop = GraphNode(
            node_id=replay_node_id(replay_id, "noop"),
            kind=KIND_OUTCOME,
            subtype="noop",
            lane=LANE_NOOP,
            label=f"{SYMBOL_NOOP} NOOP",
            preview=f"{SYMBOL_NOOP} NOOP: {noop_decision or SYMBOL_INCONCLUSIVE}",
            details={
                "comparison_index": index,
                "observation": json_clone(cmp.get("noop") or {}),
                "decision": noop_decision,
                "control_stable": facts["decision_control_stable"],
            },
            source_ref=None,
            occurrence_id=None,
            style_role=ROLE_NOOP,
            sort_key=(lane_order_index(LANE_NOOP), 0, index, replay_id),
        )
        nodes[noop.node_id] = noop

        cf = GraphNode(
            node_id=replay_node_id(replay_id, "counterfactual"),
            kind=KIND_OUTCOME,
            subtype="counterfactual",
            lane=LANE_COUNTERFACTUAL,
            label="Counterfactual",
            preview=(
                f"CF: {cf_decision or '?'} · "
                f"Acceptance {effect_symbol(facts['decision_changed'])}"
            ),
            details={
                "comparison_index": index,
                "observation": json_clone(cmp.get("counterfactual") or {}),
                "decision": cf_decision,
                "paths": {
                    "noop_path": list(cmp.get("noop_path") or []),
                    "counterfactual_path": list(cmp.get("counterfactual_path") or []),
                    "first_path_divergence": cmp.get("first_path_divergence"),
                },
            },
            source_ref=None,
            occurrence_id=None,
            style_role=ROLE_COUNTERFACTUAL,
            sort_key=(lane_order_index(LANE_COUNTERFACTUAL), 1, index, replay_id),
        )
        nodes[cf.node_id] = cf

        result_style = _comparison_style_role(facts)
        role_text = facts["display_role_label"]
        preview = format_decision_content_line(
            decision_changed=facts["decision_changed"],
            question_content_changed=facts["question_content_changed"],
        )
        if role_text:
            preview = f"{preview}; {role_text}"
        result_details = dict(facts["result_details"])
        result_details["comparison_index"] = index
        result_details["replay_id"] = replay_id
        result = GraphNode(
            node_id=replay_node_id(replay_id, "result"),
            kind=KIND_COMPARISON,
            subtype=str(facts["evidence_status"] or "comparison"),
            lane=LANE_COMPARISON,
            label="Replay result",
            preview=preview,
            details=result_details,
            source_ref=target_key,
            occurrence_id=None,
            style_role=result_style,
            sort_key=(lane_order_index(LANE_COMPARISON), 0, index, replay_id),
        )
        nodes[result.node_id] = result

        if ckpt_id:
            ckpt_node = GraphNode(
                node_id=replay_node_id(replay_id, "checkpoint"),
                kind=KIND_CHECKPOINT,
                subtype="checkpoint",
                lane=LANE_FACTUAL,
                label="Checkpoint",
                preview=str(sig.get("checkpoint_alias") or ckpt_id),
                details={
                    "checkpoint_id": sig.get("checkpoint_id"),
                    "checkpoint_alias": sig.get("checkpoint_alias"),
                    "boundary_raw": sig.get("boundary_raw"),
                    "comparison_index": index,
                },
                source_ref=str(ckpt_id),
                occurrence_id=None,
                style_role=ROLE_CHECKPOINT,
                sort_key=(lane_order_index(LANE_FACTUAL), 9, index, replay_id),
            )
            nodes[ckpt_node.node_id] = ckpt_node
            _put_edge(
                edges,
                edge_id=f"replay_branch::{replay_id}::checkpoint::noop",
                source=ckpt_node.node_id,
                target=noop.node_id,
                kind=EDGE_REPLAY_BRANCH,
                relation="noop",
                label=f"{SYMBOL_NOOP} NOOP",
                sort_key=(10, index, "checkpoint", "noop"),
                style_role=ROLE_NOOP,
            )
            _put_edge(
                edges,
                edge_id=f"replay_branch::{replay_id}::checkpoint::intervention",
                source=ckpt_node.node_id,
                target=intervention.node_id,
                kind=EDGE_REPLAY_BRANCH,
                relation="counterfactual",
                label="",
                sort_key=(10, index, "checkpoint", "intervention"),
            )

        _put_edge(
            edges,
            edge_id=f"replay_branch::{replay_id}::intervention::cf",
            source=intervention.node_id,
            target=cf.node_id,
            kind=EDGE_REPLAY_BRANCH,
            relation="counterfactual",
            label="",
            sort_key=(11, index, "intervention", "cf"),
        )
        if target_key and target_key in artifact_keys:
            _put_edge(
                edges,
                edge_id=f"intervention_target::{replay_id}::{target_key}",
                source=intervention.node_id,
                target=artifact_node_id(target_key),
                kind=EDGE_INTERVENTION_TARGET,
                relation=EDGE_INTERVENTION_TARGET,
                label="",
                sort_key=(12, index, target_key),
            )
        _put_edge(
            edges,
            edge_id=f"comparison::{replay_id}::noop::result",
            source=noop.node_id,
            target=result.node_id,
            kind=EDGE_COMPARISON,
            relation=EDGE_COMPARISON,
            label="",
            sort_key=(13, index, "noop"),
        )
        _put_edge(
            edges,
            edge_id=f"comparison::{replay_id}::cf::result",
            source=cf.node_id,
            target=result.node_id,
            kind=EDGE_COMPARISON,
            relation=EDGE_COMPARISON,
            label="",
            sort_key=(13, index, "cf"),
        )
        if facts["invalid"] or facts["unstable"]:
            warnings.append(
                f"Replay {index} marked {facts['evidence_status'] or 'unstable'}; "
                "not treated as supported causal evidence."
            )
    return ids


def _comparison_style_role(facts: Mapping[str, Any]) -> str:
    if facts.get("invalid") or facts.get("unstable"):
        return ROLE_WARNING
    if facts.get("decision_changed") is True or facts.get("question_content_changed") is True:
        return ROLE_COMPARISON_CHANGED
    if (
        facts.get("decision_changed") is False
        and facts.get("question_content_changed") is False
    ):
        return ROLE_COMPARISON_SAME
    if facts.get("evidence_status") == "no_demonstrated_effect":
        return ROLE_COMPARISON_SAME
    return ROLE_COMPARISON_INCONCLUSIVE


def replay_display_facts(cmp: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic display facts from raw replay comparison JSON."""
    status = str(cmp.get("evidence_status") or "")
    invalid = status in INVALID_EVIDENCE_STATUSES
    noop = cmp.get("noop") if isinstance(cmp.get("noop"), Mapping) else {}
    cf = cmp.get("counterfactual") if isinstance(cmp.get("counterfactual"), Mapping) else {}
    factual = cmp.get("factual") if isinstance(cmp.get("factual"), Mapping) else {}
    noop_decision = noop.get("decision")
    cf_decision = cf.get("decision")
    factual_decision = factual.get("decision")
    control_stable = cmp.get("decision_control_stable")
    unstable = control_stable is False or (
        factual_decision not in (None, "")
        and noop_decision not in (None, "")
        and str(factual_decision) != str(noop_decision)
    )

    decision_changed: Optional[bool]
    if invalid or unstable:
        decision_changed = None
    elif status == "decision_effect":
        decision_changed = True
    elif noop_decision not in (None, "") and cf_decision not in (None, ""):
        decision_changed = str(noop_decision) != str(cf_decision)
    else:
        decision_changed = None

    q_changed = _question_content_changed(cmp)
    # Keep structured content/decision facts even when the replay is invalid;
    # display_role_label is what must not claim supported causality.
    q_changed_display = None if (invalid or unstable) and q_changed is None else q_changed
    if invalid or unstable:
        # Do not advertise a supported content effect in the preview/role.
        q_changed_display = q_changed

    criterion_changed = False
    for delta in cmp.get("criterion_deltas") or []:
        if isinstance(delta, Mapping) and delta.get("counterfactual_differs_from_noop"):
            criterion_changed = True
            break
    if status == "criterion_effect":
        criterion_changed = True

    finding = cmp.get("information_role_finding")
    finding = dict(finding) if isinstance(finding, Mapping) else {}
    raw_role = finding.get("information_role")
    raw_supported = finding.get("information_role_supported")
    causal = finding.get("causal_classification")

    display_role = ""
    if not invalid and not unstable:
        if raw_supported and raw_role:
            display_role = information_role_label(str(raw_role))
        elif raw_role in {"inconclusive", "no_demonstrated_role", "artifact_level_only"}:
            display_role = information_role_label(str(raw_role))
        elif causal:
            display_role = information_role_label(str(causal))
    else:
        display_role = information_role_label("inconclusive")

    result_details = {
        "comparison_index": None,
        "evidence_status": status,
        "decision_changed": decision_changed,
        "question_content_changed": q_changed_display,
        "criterion_changed": bool(criterion_changed) if not invalid else None,
        "downstream_effect": status == "downstream_effect",
        "no_demonstrated_effect": status == "no_demonstrated_effect",
        "invalid": invalid,
        "unstable": bool(unstable),
        "decision_control_stable": control_stable,
        "criteria_control_stable": cmp.get("criteria_control_stable"),
        "subject_content_control_stable": cmp.get("subject_content_control_stable"),
        "noop_decision": noop_decision,
        "counterfactual_decision": cf_decision,
        "factual_decision": factual_decision,
        "question_content_comparison": json_clone(
            cmp.get("question_content_comparison") or {}
        ),
        "information_role_finding": json_clone(finding),
        "information_role": raw_role,
        "information_role_supported": raw_supported,
        "causal_classification": causal,
        "display_role_label": display_role,
        "semantic_intervention_validity": json_clone(
            cmp.get("semantic_intervention_validity") or {}
        ),
        "notes": list(cmp.get("notes") or []),
    }

    return {
        "evidence_status": status,
        "invalid": invalid,
        "unstable": bool(unstable),
        "decision_changed": decision_changed,
        "question_content_changed": q_changed_display,
        "criterion_changed": criterion_changed,
        "decision_control_stable": control_stable,
        "noop_decision": noop_decision,
        "cf_decision": cf_decision,
        "factual_decision": factual_decision,
        "raw_role": raw_role,
        "display_role_label": display_role,
        "result_details": result_details,
    }


def _question_content_changed(cmp: Mapping[str, Any]) -> Optional[bool]:
    qcc = cmp.get("question_content_comparison")
    if not isinstance(qcc, Mapping):
        return None
    status = str(qcc.get("effect_status") or "")
    details = qcc.get("details") if isinstance(qcc.get("details"), Mapping) else {}
    if status in {"question_content_changed", "changed"}:
        return True
    if status in {"question_content_unchanged", "unchanged"}:
        return False
    if status in {"question_content_inconclusive", "inconclusive", "unavailable"}:
        return None
    if details.get("content_changed") is True:
        return True
    if details.get("content_changed") is False:
        return False
    flag = qcc.get("counterfactual_differs_from_noop")
    if isinstance(flag, bool):
        return flag
    return None


def is_replay_valid(cmp: Mapping[str, Any]) -> bool:
    facts = replay_display_facts(cmp)
    return not facts["invalid"] and not facts["unstable"]
