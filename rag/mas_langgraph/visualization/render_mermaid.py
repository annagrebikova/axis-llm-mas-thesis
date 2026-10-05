"""Deterministic Mermaid flowchart renderer."""

from __future__ import annotations

import hashlib
from typing import Dict, List

from .models import ExplanationGraph, GraphEdge, GraphNode
from .styles import LINE_DASHED, LINE_DOTTED, edge_style, node_style


def render_mermaid(graph: ExplanationGraph) -> str:
    nodes = sorted(graph.nodes, key=lambda n: (n.sort_key, n.node_id))
    edges = sorted(graph.edges, key=lambda e: (e.sort_key, e.edge_id))
    lanes = sorted(graph.lanes, key=lambda ln: (ln.order, ln.lane_id))
    id_map: Dict[str, str] = {n.node_id: _mermaid_id(n.node_id) for n in nodes}
    lines: List[str] = ["flowchart LR"]
    used = set(id_map)
    for lane in lanes:
        lane_nodes = [n for n in nodes if n.lane == lane.lane_id]
        if not lane_nodes:
            continue
        sub_id = _mermaid_id(f"lane::{lane.lane_id}")
        lines.append(f"  subgraph {sub_id}[{_mermaid_label(lane.label)}]")
        for node in lane_nodes:
            lines.append("    " + _mermaid_node(node, id_map[node.node_id]))
        lines.append("  end")
    known_lanes = {ln.lane_id for ln in lanes}
    for node in nodes:
        if node.lane not in known_lanes:
            lines.append("  " + _mermaid_node(node, id_map[node.node_id]))
    for edge in edges:
        if edge.source in used and edge.target in used:
            lines.append("  " + _mermaid_edge(edge, id_map))
    return "\n".join(lines) + "\n"


def _mermaid_node(node: GraphNode, mid: str) -> str:
    style = node_style(node.style_role)
    shape = style.get("shape", "rectangle")
    text = _node_text(node)
    body = _mermaid_label(text)
    if shape == "diamond":
        return f"{mid}{{{body}}}"
    if shape == "hexagon":
        return f"{mid}{{{{{body}}}}}"
    if shape == "octagon":
        return f"{mid}{{{{{body}}}}}"
    if shape in {"pill", "rounded_rectangle"}:
        return f"{mid}({body})"
    if shape == "double_rectangle":
        return f"{mid}[[{body}]]"
    return f"{mid}[{body}]"


def _mermaid_edge(edge: GraphEdge, id_map: Dict[str, str]) -> str:
    style = edge_style(edge.kind)
    line = style.get("line", "solid")
    src = id_map[edge.source]
    tgt = id_map[edge.target]
    label = _mermaid_label(edge.label) if edge.label else ""
    if line == LINE_DASHED:
        arrow = f"-.->|{label}|" if label else "-.->"
    elif line == LINE_DOTTED:
        arrow = f"-.->|{label}|" if label else "-.->"
    else:
        arrow = f"-->|{label}|" if label else "-->"
    return f"{src} {arrow} {tgt}"


def _node_text(node: GraphNode) -> str:
    if node.preview and node.preview != node.label:
        return f"{node.label}: {node.preview}"
    return node.label


def _mermaid_id(canonical: str) -> str:
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    return f"n_{digest}"


def _mermaid_label(text: str) -> str:
    """Escape Mermaid control characters without deleting content."""
    out = []
    replacements = {
        '"': "#quot;",
        "'": "#apos;",
        "[": "#91;",
        "]": "#93;",
        "{": "#123;",
        "}": "#125;",
        "(": "#40;",
        ")": "#41;",
        "<": "#lt;",
        ">": "#gt;",
        "|": "#124;",
        "#": "#35;",
        "&": "#amp;",
        "`": "#96;",
        ";": "#59;",
    }
    for ch in str(text):
        if ch in {"\n", "\r"}:
            out.append("<br/>")
        elif ch in replacements:
            out.append(replacements[ch])
        else:
            out.append(ch)
    return "".join(out)
