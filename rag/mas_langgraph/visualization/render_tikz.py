"""Deterministic TikZ renderer with fixed lane coordinates."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from .models import ExplanationGraph, GraphEdge, GraphNode
from .styles import (
    LINE_DASHED,
    LINE_DOTTED,
    edge_style,
    lane_y,
    node_style,
)

X_SPACING = 4.2
NODE_WIDTH = "3.4cm"


def render_tikz(graph: ExplanationGraph) -> str:
    nodes = sorted(graph.nodes, key=lambda n: (n.sort_key, n.node_id))
    edges = sorted(graph.edges, key=lambda e: (e.sort_key, e.edge_id))
    positions = _layout(nodes)
    id_map = {n.node_id: _tikz_id(n.node_id) for n in nodes}
    lines: List[str] = [
        r"\begin{tikzpicture}[",
        "  x=1cm, y=1cm,",
        "  every node/.style={font=\\small\\sffamily, align=center}",
        "]",
    ]
    used_lanes = []
    for lane in sorted(graph.lanes, key=lambda ln: (ln.order, ln.lane_id)):
        used_lanes.append(lane)
        y = lane_y(lane.lane_id)
        lines.append(
            rf"  \node[anchor=east, font=\footnotesize\sffamily\itshape] "
            rf"at ({-1.6:.2f},{y:.2f}) {{{_escape_latex(lane.label)}}};"
        )
    for node in nodes:
        x, y = positions[node.node_id]
        lines.append(_tikz_node(node, id_map[node.node_id], x, y))
    used = set(id_map)
    for edge in edges:
        if edge.source in used and edge.target in used:
            lines.append(
                _tikz_edge(
                    edge,
                    id_map[edge.source],
                    id_map[edge.target],
                    positions[edge.source],
                    positions[edge.target],
                )
            )
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def _layout(nodes: List[GraphNode]) -> Dict[str, Tuple[float, float]]:
    by_lane: Dict[str, List[GraphNode]] = {}
    for node in nodes:
        by_lane.setdefault(node.lane, []).append(node)
    positions: Dict[str, Tuple[float, float]] = {}
    for lane_id, grouped in by_lane.items():
        grouped = sorted(grouped, key=lambda n: (n.sort_key, n.node_id))
        y = lane_y(lane_id)
        for i, node in enumerate(grouped):
            positions[node.node_id] = (i * X_SPACING, y)
    return positions


def _tikz_node(node: GraphNode, tid: str, x: float, y: float) -> str:
    style = node_style(node.style_role)
    shape = style.get("shape", "rectangle")
    fill = style.get("fill", "#FFFFFF")
    stroke = style.get("stroke", "#000000")
    text = node.label
    if node.preview and node.preview != node.label:
        text = f"{node.label}\\\\{_escape_latex(node.preview)}"
    else:
        text = _escape_latex(text)
    opts = [
        _tikz_shape(shape),
        f"draw={_tikz_color(stroke)}",
        f"fill={_tikz_color(fill)}",
        "inner sep=4pt",
        f"text width={NODE_WIDTH}",
        "align=center",
    ]
    if shape == "double_rectangle":
        opts.append("double")
        opts.append("double distance=1.4pt")
    return (
        f"  \\node[{', '.join(opts)}] ({tid}) at ({x:.2f},{y:.2f}) {{{text}}};"
    )


def _tikz_edge(
    edge: GraphEdge,
    src: str,
    tgt: str,
    src_xy: Tuple[float, float],
    tgt_xy: Tuple[float, float],
) -> str:
    style = edge_style(edge.kind)
    line = style.get("line", "solid")
    color = _tikz_color(style.get("color", "#000000"))
    opts = ["->", f"draw={color}", "thick" if style.get("width") != "thin" else "thin"]
    if line == LINE_DASHED:
        opts.append("dashed")
    elif line == LINE_DOTTED:
        opts.append("dotted")
    src_anchor, tgt_anchor = _anchors(src_xy, tgt_xy)
    label = ""
    if edge.label:
        label = (
            f" node[midway, sloped, above, font=\\scriptsize, inner sep=1pt] "
            f"{{{_escape_latex(edge.label)}}}"
        )
    return (
        f"  \\draw[{', '.join(opts)}] ({src}.{src_anchor}) -- "
        f"({tgt}.{tgt_anchor}){label};"
    )


def _anchors(
    src_xy: Tuple[float, float], tgt_xy: Tuple[float, float]
) -> Tuple[str, str]:
    sx, sy = src_xy
    tx, ty = tgt_xy
    if abs(sy - ty) < 0.2:
        if tx >= sx:
            return "east", "west"
        return "west", "east"
    if ty < sy:
        return "south", "north"
    return "north", "south"


def _tikz_shape(semantic: str) -> str:
    return {
        "rounded_rectangle": "rounded corners=3pt, rectangle",
        "rectangle": "rectangle",
        "double_rectangle": "rectangle",
        "diamond": "diamond",
        "octagon": "regular polygon, regular polygon sides=8",
        "pill": "rounded corners=8pt, rectangle",
        "hexagon": "regular polygon, regular polygon sides=6",
    }.get(semantic, "rectangle")


def _tikz_id(canonical: str) -> str:
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    return f"n{digest}"


def _tikz_color(hex_color: str) -> str:
    raw = str(hex_color or "#000000").lstrip("#")
    if len(raw) != 6:
        raw = "000000"
    r = int(raw[0:2], 16)
    g = int(raw[2:4], 16)
    b = int(raw[4:6], 16)
    return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"


def _escape_latex(text: str) -> str:
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("$", r"\$"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    out = str(text)
    # Backslash first.
    out = out.replace("\\", "\u0000")
    for src, dst in replacements:
        if src == "\\":
            continue
        out = out.replace(src, dst)
    out = out.replace("\u0000", r"\textbackslash{}")
    out = out.replace("\r\n", r"\\").replace("\n", r"\\").replace("\r", r"\\")
    return out
