"""Deterministic relevance filtering over a canonical ExplanationGraph.

The filter only hides existing nodes and edges. It never invents relations
or changes causal classification.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .labels import ESSENTIAL_OCCURRENCE_NODE_IDS
from .models import (
    CONDITION_AXIS,
    CONDITION_TRACE,
    DETAIL_ESSENTIAL,
    DETAIL_FULL,
    DETAIL_RELEVANT,
    EDGE_CONSUMES,
    EDGE_CONTROL_FLOW,
    EDGE_DECLARED_REFERENCE,
    EDGE_PRODUCES,
    KIND_ARTIFACT,
    KIND_CHECKPOINT,
    KIND_COMPARISON,
    KIND_INTERVENTION,
    KIND_OCCURRENCE,
    KIND_OUTCOME,
    LANE_ARTIFACTS,
    LANE_COMPARISON,
    LANE_COUNTERFACTUAL,
    LANE_NOOP,
    LANE_ORDER,
    ExplanationGraph,
    GraphEdge,
    GraphLane,
    GraphNode,
)
from .styles import EDGE_KIND_TO_ROLE, lane_order_index
from .labels import lane_label


DISTANCE_EDGE_KINDS = frozenset(
    {
        EDGE_CONTROL_FLOW,
        EDGE_PRODUCES,
        EDGE_CONSUMES,
        EDGE_DECLARED_REFERENCE,
    }
)

INVALID_EVIDENCE_STATUSES = frozenset(
    {
        "noop_error",
        "replay_unstable",
        "intervention_error",
        "outcome_unavailable",
    }
)

ESSENTIAL_ARTIFACT_TYPES = frozenset(
    {
        "CandidateMCQ",
        "SupervisorDecision",
        "MCQBlueprint",
        "LearningObjective",
        "KnowledgeAtom",
        "EvidenceChunk",
        "EvidenceCheckReport",
        "CrossCoVeReport",
        "OutcomeAssessment",
    }
)

MAX_ESSENTIAL_REPLAYS = 3
MISSING_DISTANCE = 10**6


def apply_detail_level(
    graph: ExplanationGraph,
    detail_level: str,
    *,
    case_dict: Mapping[str, Any],
) -> ExplanationGraph:
    level = str(detail_level or DETAIL_FULL)
    if level == DETAIL_FULL:
        return graph.replace(detail_level=DETAIL_FULL)

    relevant = _filter_relevant(graph, case_dict)
    if level == DETAIL_RELEVANT:
        return relevant
    if level == DETAIL_ESSENTIAL:
        return _filter_essential(relevant, graph, case_dict)
    return graph.replace(detail_level=level)


def _filter_relevant(
    graph: ExplanationGraph,
    case_dict: Mapping[str, Any],
) -> ExplanationGraph:
    node_map = graph.node_map()
    anchors = _anchor_node_ids(graph)
    connected = _connected_component(graph, anchors) if anchors else set(node_map)
    # Trace has no provenance links from the question to occurrences.
    # Keep the factual occurrence backbone; it is already CaseSlice-scoped.
    if graph.condition == CONDITION_TRACE:
        connected |= {
            n.node_id for n in graph.nodes if n.kind == KIND_OCCURRENCE
        }
        connected |= {
            n.node_id
            for n in graph.nodes
            if n.style_role in {"target_question", "decision"}
        }

    replay_keep = _relevant_replay_ids(graph, connected, case_dict)
    keep = set(connected)
    keep |= _replay_nodes_for_ids(graph, replay_keep)

    return _project_graph(
        graph,
        keep,
        detail_level=DETAIL_RELEVANT,
        case_dict=case_dict,
        selected_replay_ids=replay_keep,
    )


def _filter_essential(
    relevant: ExplanationGraph,
    full: ExplanationGraph,
    case_dict: Mapping[str, Any],
) -> ExplanationGraph:
    node_map = relevant.node_map()
    keep: Set[str] = set()

    target_id = artifact_node_id(full.target_ref or "")
    if target_id in node_map:
        keep.add(target_id)

    for node in relevant.nodes:
        if node.kind == KIND_OCCURRENCE and node.subtype in ESSENTIAL_OCCURRENCE_NODE_IDS:
            keep.add(node.node_id)
        if node.style_role in {"target_question", "decision"}:
            keep.add(node.node_id)
        if node.subtype in {
            "SupervisorDecision",
            "EvidenceCheckReport",
            "CrossCoVeReport",
            "OutcomeAssessment",
        } and node.node_id in _anchor_node_ids(relevant):
            keep.add(node.node_id)

    # Compact target chain: question → blueprint → LO / atoms / chunks.
    chain_types = {
        "MCQBlueprint",
        "LearningObjective",
        "KnowledgeAtom",
        "EvidenceChunk",
        "CandidateMCQ",
        "SupervisorDecision",
        "EvidenceCheckReport",
        "CrossCoVeReport",
        "OutcomeAssessment",
    }
    frontier = set(keep)
    seen = set(keep)
    for _ in range(3):
        extra: Set[str] = set()
        for nid in frontier:
            extra |= _direct_typed_neighbors(relevant, nid, chain_types)
        extra -= seen
        extra = {
            eid
            for eid in extra
            if not _is_non_target_candidate(node_map.get(eid), full.target_ref)
        }
        keep |= extra
        seen |= extra
        frontier = extra

    selected_replay: List[str] = []
    ranking: List[Tuple[str, Dict[str, Any]]] = []
    if relevant.condition == CONDITION_AXIS:
        ranking = rank_replays(full, case_dict)
        present = {
            _replay_id_from_node(n)
            for n in relevant.nodes
            if _replay_id_from_node(n)
        }
        ranked_present = [rid for rid, _meta in ranking if rid in present]
        selected_replay = ranked_present[:MAX_ESSENTIAL_REPLAYS]
        keep |= _replay_nodes_for_ids(relevant, selected_replay)

    return _project_graph(
        relevant,
        keep,
        detail_level=DETAIL_ESSENTIAL,
        case_dict=case_dict,
        selected_replay_ids=selected_replay,
        ranking=ranking,
        full_graph=full,
    )


def _project_graph(
    graph: ExplanationGraph,
    keep: Set[str],
    *,
    detail_level: str,
    case_dict: Mapping[str, Any],
    selected_replay_ids: Sequence[str],
    ranking: Optional[Sequence[Tuple[str, Mapping[str, Any]]]] = None,
    full_graph: Optional[ExplanationGraph] = None,
) -> ExplanationGraph:
    base = full_graph or graph
    nodes = [n for n in graph.nodes if n.node_id in keep]
    kept_ids = {n.node_id for n in nodes}

    retained_edges: List[GraphEdge] = []
    for edge in graph.edges:
        if edge.kind == EDGE_CONTROL_FLOW:
            continue
        if edge.source in kept_ids and edge.target in kept_ids:
            retained_edges.append(edge)
    retained_edges.extend(rebuild_control_flow(base, kept_ids))

    used_lanes = {n.lane for n in nodes}
    lanes = tuple(
        GraphLane(lane_id=lid, label=lane_label(lid), order=lane_order_index(lid))
        for lid in LANE_ORDER
        if lid in used_lanes
    )

    full_replay_ids = list((base.metadata or {}).get("replay_ids") or [])
    hidden_replay = 0
    if graph.condition == CONDITION_AXIS:
        hidden_replay = max(0, len(full_replay_ids) - len(set(selected_replay_ids)))

    hidden_nodes = max(0, len(base.nodes) - len(nodes))
    # Edge hide count is measured against the source graph of this projection.
    source_edge_ids = {e.edge_id for e in graph.edges}
    new_edge_ids = {e.edge_id for e in retained_edges}
    hidden_edges = max(0, len(source_edge_ids) - len(source_edge_ids & new_edge_ids))

    metadata = dict(graph.metadata or {})
    metadata["selected_replay_indices"] = [
        _replay_index(rid, full_replay_ids) for rid in selected_replay_ids
    ]
    metadata["selected_replay_ids"] = list(selected_replay_ids)
    metadata["hidden_replay_count"] = hidden_replay
    if ranking is not None:
        metadata["replay_ranking"] = [
            {
                "replay_id": rid,
                "rank_tuple": list(meta.get("rank_tuple") or []),
                "comparison_index": meta.get("comparison_index"),
                "selected": rid in set(selected_replay_ids),
            }
            for rid, meta in ranking
        ]

    warnings = list(graph.warnings)
    return graph.replace(
        nodes=tuple(sorted(nodes, key=lambda n: (n.sort_key, n.node_id))),
        edges=tuple(sorted(retained_edges, key=lambda e: (e.sort_key, e.edge_id))),
        lanes=lanes,
        detail_level=detail_level,
        metadata=metadata,
        hidden_counts={
            "hidden_node_count": hidden_nodes,
            "hidden_edge_count": hidden_edges,
            "hidden_replay_count": hidden_replay,
        },
        warnings=tuple(warnings),
    )


def rebuild_control_flow(
    graph: ExplanationGraph,
    kept_node_ids: Set[str],
) -> List[GraphEdge]:
    order = [str(x) for x in (graph.metadata or {}).get("occurrence_order") or []]
    retained: List[str] = []
    skips: List[int] = []
    pending_skip = 0
    for oid in order:
        nid = occurrence_node_id(oid)
        if nid in kept_node_ids:
            if retained:
                skips.append(pending_skip)
            retained.append(oid)
            pending_skip = 0
        elif retained:
            pending_skip += 1

    edges: List[GraphEdge] = []
    for i in range(len(retained) - 1):
        src_oid = retained[i]
        tgt_oid = retained[i + 1]
        n_skip = int(skips[i]) if i < len(skips) else 0
        label = f"... {n_skip} steps ..." if n_skip > 0 else ""
        edge_id = f"control_flow::{src_oid}::{tgt_oid}"
        edges.append(
            GraphEdge(
                edge_id=edge_id,
                source=occurrence_node_id(src_oid),
                target=occurrence_node_id(tgt_oid),
                kind=EDGE_CONTROL_FLOW,
                relation=EDGE_CONTROL_FLOW,
                label=label,
                style_role=EDGE_KIND_TO_ROLE[EDGE_CONTROL_FLOW],
                sort_key=(0, src_oid, tgt_oid),
                details={"skipped_occurrence_count": n_skip},
            )
        )
    return edges


def _is_non_target_candidate(node: Optional[GraphNode], target_ref: Optional[str]) -> bool:
    if node is None or node.subtype != "CandidateMCQ":
        return False
    if not target_ref:
        return False
    return str(node.source_ref or "") != str(target_ref)


def occurrence_node_id(occurrence_id: str) -> str:
    return f"occurrence::{occurrence_id}"


def artifact_node_id(artifact_key: str) -> str:
    return f"artifact::{artifact_key}"


def _anchor_node_ids(graph: ExplanationGraph) -> Set[str]:
    anchors: Set[str] = set()
    target = graph.target_ref
    if target:
        anchors.add(artifact_node_id(target))
    logical, version = _split_ref(target)
    for node in graph.nodes:
        if node.source_ref == target:
            anchors.add(node.node_id)
        content = {}
        if isinstance(node.details, Mapping):
            inner = node.details.get("content")
            content = inner if isinstance(inner, Mapping) else {}
            ref = node.details.get("ref")
            if isinstance(ref, Mapping) and _ref_key(ref) == target:
                anchors.add(node.node_id)
        if node.subtype in {
            "SupervisorDecision",
            "OutcomeAssessment",
            "EvidenceCheckReport",
            "CrossCoVeReport",
        }:
            if _matches_target_version(content, node.details, logical, version):
                anchors.add(node.node_id)
        if node.style_role in {"target_question", "decision"}:
            anchors.add(node.node_id)
    return {a for a in anchors if a in graph.node_map()}


def _matches_target_version(
    content: Mapping[str, Any],
    details: Mapping[str, Any],
    logical: Optional[str],
    version: Optional[int],
) -> bool:
    if not logical or version is None:
        return False
    q_id = content.get("q_id") or content.get("logical_id")
    cand_ver = content.get("candidate_version")
    if q_id == logical and cand_ver is not None and int(cand_ver) == int(version):
        return True
    tref = content.get("target_ref") if isinstance(content.get("target_ref"), Mapping) else None
    if tref and _ref_key(tref) == f"{logical}:v{int(version)}":
        return True
    ref = details.get("ref") if isinstance(details.get("ref"), Mapping) else {}
    if _ref_key(ref) and str(ref.get("logical_id")) == logical and int(ref.get("version") or 0) == int(version):
        # Decision/assessment refs use different logical ids; only treat as
        # match when candidate_version already matched above.
        return False
    return False


def _split_ref(ref: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if not ref:
        return None, None
    if ":v" in ref:
        logical, _, ver = ref.rpartition(":v")
        try:
            return logical, int(ver)
        except ValueError:
            return ref, None
    return ref, None


def _ref_key(ref: Any) -> Optional[str]:
    if ref is None:
        return None
    if isinstance(ref, str):
        return ref.strip() or None
    if isinstance(ref, Mapping):
        if ref.get("key"):
            return str(ref.get("key"))
        logical = str(ref.get("logical_id") or "")
        version = ref.get("version")
        if logical and version is not None:
            return f"{logical}:v{int(version)}"
        return logical or None
    return None


def _undirected_adj(graph: ExplanationGraph, kinds: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
    allowed = kinds or DISTANCE_EDGE_KINDS
    adj: Dict[str, Set[str]] = {n.node_id: set() for n in graph.nodes}
    for edge in graph.edges:
        if edge.kind not in allowed:
            continue
        if edge.source in adj and edge.target in adj:
            adj[edge.source].add(edge.target)
            adj[edge.target].add(edge.source)
    return adj


def _connected_component(graph: ExplanationGraph, seeds: Iterable[str]) -> Set[str]:
    adj = _undirected_adj(graph)
    seen: Set[str] = set()
    q: deque[str] = deque()
    for seed in seeds:
        if seed in adj:
            seen.add(seed)
            q.append(seed)
    while q:
        cur = q.popleft()
        for nxt in sorted(adj.get(cur, ())):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _reaches(graph: ExplanationGraph, start: str, goal: str) -> bool:
    if start == goal:
        return True
    adj = _undirected_adj(graph)
    seen = {start}
    q: deque[str] = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def graph_distance(graph: ExplanationGraph, start: str, goal: str) -> Optional[int]:
    if start not in graph.node_map() or goal not in graph.node_map():
        return None
    adj = _undirected_adj(graph)
    seen = {start: 0}
    q: deque[str] = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            return seen[cur]
        for nxt in sorted(adj.get(cur, ())):
            if nxt not in seen:
                seen[nxt] = seen[cur] + 1
                q.append(nxt)
    return None


def _direct_typed_neighbors(
    graph: ExplanationGraph, node_id: str, subtypes: Set[str]
) -> Set[str]:
    out: Set[str] = set()
    node_map = graph.node_map()
    for edge in graph.edges:
        if edge.kind not in DISTANCE_EDGE_KINDS:
            continue
        other = None
        if edge.source == node_id:
            other = edge.target
        elif edge.target == node_id:
            other = edge.source
        if other is None:
            continue
        node = node_map.get(other)
        if node is not None and node.subtype in subtypes:
            out.add(other)
    return out


def _replay_id_from_node(node: GraphNode) -> Optional[str]:
    if not node.node_id.startswith("replay::"):
        return None
    parts = node.node_id.split("::")
    if len(parts) < 3:
        return None
    return parts[1]


def _replay_nodes_for_ids(graph: ExplanationGraph, replay_ids: Iterable[str]) -> Set[str]:
    wanted = set(replay_ids)
    keep: Set[str] = set()
    for node in graph.nodes:
        rid = _replay_id_from_node(node)
        if rid and rid in wanted:
            keep.add(node.node_id)
    for edge in graph.edges:
        if edge.source in keep and edge.target in keep:
            keep.add(edge.source)
            keep.add(edge.target)
    return keep


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
    out: List[Dict[str, Any]] = []
    for item in raw.get("replay_comparisons") or []:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def unique_replay_ids(comparisons: Sequence[Mapping[str, Any]]) -> List[str]:
    used: Set[str] = set()
    ids: List[str] = []
    for i, cmp in enumerate(comparisons):
        sig = cmp.get("intervention_signature")
        sig = sig if isinstance(sig, Mapping) else {}
        op = str(sig.get("operation") or "")
        target = _ref_key(sig.get("target_ref")) or ""
        replacement = _ref_key(sig.get("replacement_ref")) or ""
        ckpt = str(sig.get("checkpoint_id") or sig.get("checkpoint_alias") or "")
        base = f"{op}|{target}|{replacement}|{ckpt}" or f"index:{i}"
        token = base if base not in used else f"{base}#{i}"
        used.add(token)
        ids.append(token)
    return ids


def _replay_index(replay_id: str, all_ids: Sequence[str]) -> int:
    try:
        return list(all_ids).index(replay_id)
    except ValueError:
        return -1


def _relevant_replay_ids(
    graph: ExplanationGraph,
    connected: Set[str],
    case_dict: Mapping[str, Any],
) -> List[str]:
    if graph.condition != CONDITION_AXIS:
        return []
    comparisons = _axis_comparisons(case_dict)
    replay_ids = unique_replay_ids(comparisons)
    valid_ids: List[str] = []
    invalid_ids: List[str] = []
    for i, (rid, cmp) in enumerate(zip(replay_ids, comparisons)):
        sig = cmp.get("intervention_signature") if isinstance(cmp.get("intervention_signature"), Mapping) else {}
        target_key = _ref_key(sig.get("target_ref"))
        target_nid = artifact_node_id(target_key) if target_key else None
        connected_target = bool(target_nid and target_nid in connected)
        # Also keep if any replay node already sits in the connected set.
        unit_nodes = {n.node_id for n in graph.nodes if _replay_id_from_node(n) == rid}
        if not connected_target and not (unit_nodes & connected):
            # Intervention target not in retained factual/provenance subgraph.
            continue
        if _is_invalid_replay(cmp):
            invalid_ids.append(rid)
        else:
            valid_ids.append(rid)
    if valid_ids:
        return valid_ids
    return invalid_ids


def _is_invalid_replay(cmp: Mapping[str, Any]) -> bool:
    status = str(cmp.get("evidence_status") or "")
    if status in INVALID_EVIDENCE_STATUSES:
        return True
    if cmp.get("decision_control_stable") is False:
        return True
    factual = cmp.get("factual") if isinstance(cmp.get("factual"), Mapping) else {}
    noop = cmp.get("noop") if isinstance(cmp.get("noop"), Mapping) else {}
    fd = factual.get("decision")
    nd = noop.get("decision")
    if fd not in (None, "") and nd not in (None, "") and str(fd) != str(nd):
        return True
    return False


def rank_replays(
    graph: ExplanationGraph,
    case_dict: Mapping[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    comparisons = _axis_comparisons(case_dict)
    replay_ids = unique_replay_ids(comparisons)
    target_nid = artifact_node_id(graph.target_ref or "")
    ranked: List[Tuple[Tuple[Any, ...], str, Dict[str, Any]]] = []
    for i, (rid, cmp) in enumerate(zip(replay_ids, comparisons)):
        sig = cmp.get("intervention_signature") if isinstance(cmp.get("intervention_signature"), Mapping) else {}
        target_key = _ref_key(sig.get("target_ref"))
        dist = None
        if target_key and target_nid:
            dist = graph_distance(graph, artifact_node_id(target_key), target_nid)
        rank = _rank_tuple(cmp, i, dist, rid)
        meta = {
            "rank_tuple": rank,
            "comparison_index": i,
            "distance": dist,
            "replay_id": rid,
        }
        ranked.append((rank, rid, meta))
    ranked.sort(key=lambda row: row[0])
    return [(rid, meta) for _rank, rid, meta in ranked]


def _rank_tuple(
    cmp: Mapping[str, Any],
    index: int,
    distance: Optional[int],
    replay_id: str,
) -> Tuple[Any, ...]:
    status = str(cmp.get("evidence_status") or "")
    invalid = _is_invalid_replay(cmp)
    k1 = 0 if not invalid else 1
    k2 = 0 if status == "decision_effect" else 1
    criterion = status == "criterion_effect"
    for delta in cmp.get("criterion_deltas") or []:
        if (
            isinstance(delta, Mapping)
            and delta.get("decision_relevant")
            and delta.get("counterfactual_differs_from_noop")
        ):
            criterion = True
    k3 = 0 if criterion else 1
    q_changed = _question_content_changed(cmp)
    k4 = 0 if q_changed is True else 1
    k5 = 0 if status == "downstream_effect" else 1
    k6 = 0 if status == "no_demonstrated_effect" else 1
    k7 = int(distance) if distance is not None else MISSING_DISTANCE
    k8 = int(index)
    k9 = str(replay_id)
    return (k1, k2, k3, k4, k5, k6, k7, k8, k9)


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
    if details.get("content_changed") is True:
        return True
    if details.get("content_changed") is False:
        return False
    flag = qcc.get("counterfactual_differs_from_noop")
    if isinstance(flag, bool):
        return flag
    return None
