"""Renderer-independent explanation-graph models.

These types are JSON-safe and must not depend on ArtifactRegistry, LangGraph,
CaseSlice objects, replay executors, or LLM clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


GRAPH_SCHEMA_VERSION = 1

CONDITION_TRACE = "trace"
CONDITION_TRACE_PROVENANCE = "trace_provenance"
CONDITION_AXIS = "axis"
VALID_CONDITIONS = frozenset(
    {CONDITION_TRACE, CONDITION_TRACE_PROVENANCE, CONDITION_AXIS}
)

DETAIL_ESSENTIAL = "essential"
DETAIL_RELEVANT = "relevant"
DETAIL_FULL = "full"
VALID_DETAIL_LEVELS = frozenset(
    {DETAIL_ESSENTIAL, DETAIL_RELEVANT, DETAIL_FULL}
)

LANE_ARTIFACTS = "artifacts"
LANE_FACTUAL = "factual"
LANE_NOOP = "noop"
LANE_COUNTERFACTUAL = "counterfactual"
LANE_COMPARISON = "comparison"
LANE_ORDER: Tuple[str, ...] = (
    LANE_ARTIFACTS,
    LANE_FACTUAL,
    LANE_NOOP,
    LANE_COUNTERFACTUAL,
    LANE_COMPARISON,
)

KIND_OCCURRENCE = "occurrence"
KIND_ARTIFACT = "artifact"
KIND_INTERVENTION = "intervention"
KIND_OUTCOME = "outcome"
KIND_COMPARISON = "comparison"
KIND_CHECKPOINT = "checkpoint"

EDGE_CONTROL_FLOW = "control_flow"
EDGE_PRODUCES = "produces"
EDGE_CONSUMES = "consumes"
EDGE_DECLARED_REFERENCE = "declared_reference"
EDGE_INTERVENTION_TARGET = "intervention_target"
EDGE_REPLAY_BRANCH = "replay_branch"
EDGE_COMPARISON = "comparison"

PROVENANCE_EDGE_KINDS = frozenset({EDGE_PRODUCES, EDGE_CONSUMES})
REPLAY_NODE_KINDS = frozenset(
    {KIND_INTERVENTION, KIND_CHECKPOINT, KIND_COMPARISON}
)
REPLAY_LANES = frozenset({LANE_NOOP, LANE_COUNTERFACTUAL, LANE_COMPARISON})


class VisualizationDataError(ValueError):
    """Raised when exported case JSON cannot be turned into a graph."""


def json_clone(value: Any) -> Any:
    """Deep-copy a JSON-safe value, sorting mapping keys for determinism."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): json_clone(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [json_clone(v) for v in value]
    raise TypeError(
        f"ExplanationGraph values must be JSON-safe, got {type(value).__name__}"
    )


def _tuple_to_list(value: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    for item in value:
        if isinstance(item, tuple):
            out.append(_tuple_to_list(item))
        else:
            out.append(json_clone(item) if isinstance(item, (dict, list)) else item)
    return out


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: str
    subtype: str
    lane: str
    label: str
    preview: str
    details: Mapping[str, Any] = field(default_factory=dict)
    source_ref: Optional[str] = None
    occurrence_id: Optional[str] = None
    style_role: str = "artifact"
    sort_key: Tuple[Any, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "subtype": self.subtype,
            "lane": self.lane,
            "label": self.label,
            "preview": self.preview,
            "details": json_clone(dict(self.details or {})),
            "source_ref": self.source_ref,
            "occurrence_id": self.occurrence_id,
            "style_role": self.style_role,
            "sort_key": _tuple_to_list(self.sort_key),
        }


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    source: str
    target: str
    kind: str
    relation: str
    label: str = ""
    style_role: str = "provenance"
    sort_key: Tuple[Any, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "relation": self.relation,
            "label": self.label,
            "style_role": self.style_role,
            "sort_key": _tuple_to_list(self.sort_key),
            "details": json_clone(dict(self.details or {})),
        }


@dataclass(frozen=True)
class GraphLane:
    lane_id: str
    label: str
    order: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "label": self.label,
            "order": int(self.order),
        }


@dataclass(frozen=True)
class ExplanationGraph:
    schema_version: int
    condition: str
    detail_level: str
    target_ref: Optional[str]
    nodes: Tuple[GraphNode, ...]
    edges: Tuple[GraphEdge, ...]
    lanes: Tuple[GraphLane, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hidden_counts: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        nodes = tuple(sorted(self.nodes, key=_node_sort))
        edges = tuple(sorted(self.edges, key=_edge_sort))
        lanes = tuple(sorted(self.lanes, key=lambda ln: (ln.order, ln.lane_id)))
        return {
            "schema_version": int(self.schema_version),
            "condition": self.condition,
            "detail_level": self.detail_level,
            "target_ref": self.target_ref,
            "nodes": [n.to_dict() for n in nodes],
            "edges": [e.to_dict() for e in edges],
            "lanes": [ln.to_dict() for ln in lanes],
            "metadata": json_clone(dict(self.metadata or {})),
            "hidden_counts": json_clone(dict(self.hidden_counts or {})),
            "warnings": list(self.warnings),
        }

    def node_ids(self) -> Tuple[str, ...]:
        return tuple(n.node_id for n in sorted(self.nodes, key=_node_sort))

    def edge_ids(self) -> Tuple[str, ...]:
        return tuple(e.edge_id for e in sorted(self.edges, key=_edge_sort))

    def node_map(self) -> Dict[str, GraphNode]:
        return {n.node_id: n for n in self.nodes}

    def replace(
        self,
        *,
        nodes: Optional[Sequence[GraphNode]] = None,
        edges: Optional[Sequence[GraphEdge]] = None,
        lanes: Optional[Sequence[GraphLane]] = None,
        detail_level: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        hidden_counts: Optional[Mapping[str, Any]] = None,
        warnings: Optional[Sequence[str]] = None,
    ) -> "ExplanationGraph":
        return ExplanationGraph(
            schema_version=self.schema_version,
            condition=self.condition,
            detail_level=self.detail_level if detail_level is None else detail_level,
            target_ref=self.target_ref,
            nodes=tuple(nodes) if nodes is not None else self.nodes,
            edges=tuple(edges) if edges is not None else self.edges,
            lanes=tuple(lanes) if lanes is not None else self.lanes,
            metadata=dict(metadata) if metadata is not None else dict(self.metadata),
            hidden_counts=(
                dict(hidden_counts)
                if hidden_counts is not None
                else dict(self.hidden_counts)
            ),
            warnings=tuple(warnings) if warnings is not None else self.warnings,
        )


def _node_sort(node: GraphNode) -> Tuple[Any, ...]:
    sk = node.sort_key if node.sort_key else ()
    return (sk, node.node_id)


def _edge_sort(edge: GraphEdge) -> Tuple[Any, ...]:
    sk = edge.sort_key if edge.sort_key else ()
    return (sk, edge.edge_id, edge.source, edge.target)
