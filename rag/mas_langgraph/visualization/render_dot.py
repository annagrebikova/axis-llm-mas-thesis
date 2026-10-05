"""Deterministic Graphviz DOT renderer."""

from __future__ import annotations

from typing import Dict, List

from .models import ExplanationGraph, GraphEdge, GraphNode
from .styles import LINE_DASHED, LINE_DOTTED, LINE_SOLID, edge_style, node_style


def render_dot(graph: ExplanationGraph) -> str:
    nodes = sorted(graph.nodes, key=lambda n: (n.sort_key, n.node_id))
    edges = sorted(graph.edges, key=lambda e: (e.sort_key, e.edge_id))
    lanes = sorted(graph.lanes, key=lambda ln: (ln.order, ln.lane_id))
    lines: List[str] = [
        "digraph ExplanationGraph {",
        "  graph [rankdir=LR, bgcolor=white];",
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]
    used_ids = {n.node_id for n in nodes}
    for lane in lanes:
        lane_nodes = [n for n in nodes if n.lane == lane.lane_id]
        if not lane_nodes:
            continue
        cluster = _dot_id(f"cluster::{lane.lane_id}")
        lines.append(f"  subgraph {cluster} {{")
        lines.append(f"    label={_dot_quote(lane.label)};")
        lines.append("    color=lightgrey;")
        lines.append("    style=rounded;")
        for node in lane_nodes:
            lines.append("    " + _dot_node(node))
        lines.append("  }")
    # Nodes whose lane is not in the lane list (should not happen).
    known_lanes = {ln.lane_id for ln in lanes}
    for node in nodes:
        if node.lane not in known_lanes:
            lines.append("  " + _dot_node(node))
    for edge in edges:
        if edge.source in used_ids and edge.target in used_ids:
            lines.append("  " + _dot_edge(edge))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dot_node(node: GraphNode) -> str:
    style = node_style(node.style_role)
    shape = _dot_shape(style.get("shape", "rectangle"))
    label = _node_label(node)
    extras = [
        f"shape={shape}",
        f"style={_dot_node_style(style.get('shape', ''))}",
        f"fillcolor={_dot_quote(style.get('fill', '#FFFFFF'))}",
        f"color={_dot_quote(style.get('stroke', '#000000'))}",
        f"fontcolor={_dot_quote(style.get('color', '#000000'))}",
        f"label={_dot_quote(label)}",
    ]
    if style.get("shape") == "double_rectangle":
        extras.append("peripheries=2")
    return f"{_dot_id(node.node_id)} [{', '.join(extras)}];"


def _dot_edge(edge: GraphEdge) -> str:
    style = edge_style(edge.kind)
    line = style.get("line", LINE_SOLID)
    attrs = [
        f"color={_dot_quote(style.get('color', '#000000'))}",
        f"style={_dot_line(line)}",
        f"arrowhead=normal",
    ]
    if style.get("width") == "thin":
        attrs.append("penwidth=1.0")
    else:
        attrs.append("penwidth=1.6")
    if edge.label:
        attrs.append(f"label={_dot_quote(edge.label)}")
    return (
        f"{_dot_id(edge.source)} -> {_dot_id(edge.target)} [{', '.join(attrs)}];"
    )


def _node_label(node: GraphNode) -> str:
    parts = [node.label]
    if node.preview and node.preview != node.label:
        parts.append(node.preview)
    return "\n".join(parts)


def _dot_shape(semantic: str) -> str:
    return {
        "rounded_rectangle": "box",
        "rectangle": "box",
        "double_rectangle": "box",
        "diamond": "diamond",
        "octagon": "octagon",
        "pill": "oval",
        "hexagon": "hexagon",
    }.get(semantic, "box")


def _dot_node_style(semantic: str) -> str:
    if semantic in {"rounded_rectangle", "pill"}:
        return "filled,rounded"
    return "filled"


def _dot_line(line: str) -> str:
    if line == LINE_DASHED:
        return "dashed"
    if line == LINE_DOTTED:
        return "dotted"
    return "solid"


def _dot_id(node_id: str) -> str:
    """Stable Graphviz identifier derived from the canonical node/edge id."""
    safe = []
    for ch in node_id:
        if ch.isalnum() or ch == "_":
            safe.append(ch)
        else:
            safe.append(f"_{ord(ch):x}_")
    token = "n_" + "".join(safe)
    if token[0].isdigit():
        token = "n_" + token
    return token


def _dot_quote(text: str) -> str:
    escaped = (
        str(text)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )
    return f'"{escaped}"'
