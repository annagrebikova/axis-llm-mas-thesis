"""Renderer-independent visual grammar.

Color is never the only carrier of meaning. Renderers map these roles onto
the closest available shapes, line styles, and symbols.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from .models import (
    EDGE_COMPARISON,
    EDGE_CONSUMES,
    EDGE_CONTROL_FLOW,
    EDGE_DECLARED_REFERENCE,
    EDGE_INTERVENTION_TARGET,
    EDGE_PRODUCES,
    EDGE_REPLAY_BRANCH,
    LANE_ORDER,
)


# Wong / color-blind-aware palette.
COLOR_FACTUAL = "#0072B2"
COLOR_ARTIFACT_FILL = "#D6EAF8"
COLOR_ARTIFACT_BORDER = "#0072B2"
COLOR_PROVENANCE = "#CC79A7"
COLOR_INTERVENTION = "#E69F00"
COLOR_CHANGED = "#D55E00"
COLOR_UNCHANGED = "#009E73"
COLOR_NOOP = "#7F7F7F"
COLOR_WARNING = "#D55E00"
COLOR_NEUTRAL = "#4D4D4D"

ROLE_FACTUAL_OCCURRENCE = "factual_occurrence"
ROLE_ARTIFACT = "artifact"
ROLE_TARGET_QUESTION = "target_question"
ROLE_DECISION = "decision"
ROLE_PROVENANCE = "provenance"
ROLE_DECLARED_REFERENCE = "declared_reference"
ROLE_INTERVENTION = "intervention"
ROLE_NOOP = "noop"
ROLE_COUNTERFACTUAL = "counterfactual"
ROLE_COMPARISON_CHANGED = "comparison_changed"
ROLE_COMPARISON_SAME = "comparison_same"
ROLE_COMPARISON_INCONCLUSIVE = "comparison_inconclusive"
ROLE_WARNING = "warning"
ROLE_CHECKPOINT = "checkpoint"

SHAPE_ROUNDED_RECTANGLE = "rounded_rectangle"
SHAPE_RECTANGLE = "rectangle"
SHAPE_DOUBLE_RECTANGLE = "double_rectangle"
SHAPE_DIAMOND = "diamond"
SHAPE_OCTAGON = "octagon"
SHAPE_PILL = "pill"
SHAPE_HEXAGON = "hexagon"

LINE_SOLID = "solid"
LINE_DASHED = "dashed"
LINE_DOTTED = "dotted"

LANE_Y = {
    "artifacts": 3.0,
    "factual": 1.0,
    "noop": -1.0,
    "counterfactual": -3.0,
    "comparison": -5.0,
}

NODE_STYLE: Dict[str, Dict[str, str]] = {
    ROLE_FACTUAL_OCCURRENCE: {
        "shape": SHAPE_ROUNDED_RECTANGLE,
        "fill": "#E8F4FC",
        "stroke": COLOR_FACTUAL,
        "color": COLOR_FACTUAL,
    },
    ROLE_ARTIFACT: {
        "shape": SHAPE_RECTANGLE,
        "fill": COLOR_ARTIFACT_FILL,
        "stroke": COLOR_ARTIFACT_BORDER,
        "color": COLOR_FACTUAL,
    },
    ROLE_TARGET_QUESTION: {
        "shape": SHAPE_DOUBLE_RECTANGLE,
        "fill": "#FFFFFF",
        "stroke": COLOR_FACTUAL,
        "color": COLOR_FACTUAL,
    },
    ROLE_DECISION: {
        "shape": SHAPE_DIAMOND,
        "fill": "#E8F4FC",
        "stroke": COLOR_FACTUAL,
        "color": COLOR_FACTUAL,
    },
    ROLE_INTERVENTION: {
        "shape": SHAPE_OCTAGON,
        "fill": "#FFF4D6",
        "stroke": COLOR_INTERVENTION,
        "color": COLOR_INTERVENTION,
    },
    ROLE_NOOP: {
        "shape": SHAPE_PILL,
        "fill": "#F0F0F0",
        "stroke": COLOR_NOOP,
        "color": COLOR_NOOP,
    },
    ROLE_COUNTERFACTUAL: {
        "shape": SHAPE_PILL,
        "fill": "#FDE8D0",
        "stroke": COLOR_CHANGED,
        "color": COLOR_CHANGED,
    },
    ROLE_COMPARISON_CHANGED: {
        "shape": SHAPE_HEXAGON,
        "fill": "#FDE8D0",
        "stroke": COLOR_CHANGED,
        "color": COLOR_CHANGED,
    },
    ROLE_COMPARISON_SAME: {
        "shape": SHAPE_HEXAGON,
        "fill": "#D9F3EA",
        "stroke": COLOR_UNCHANGED,
        "color": COLOR_UNCHANGED,
    },
    ROLE_COMPARISON_INCONCLUSIVE: {
        "shape": SHAPE_HEXAGON,
        "fill": "#F5F5F5",
        "stroke": COLOR_NOOP,
        "color": COLOR_NOOP,
    },
    ROLE_WARNING: {
        "shape": SHAPE_HEXAGON,
        "fill": "#FDE8D0",
        "stroke": COLOR_WARNING,
        "color": COLOR_WARNING,
    },
    ROLE_CHECKPOINT: {
        "shape": SHAPE_ROUNDED_RECTANGLE,
        "fill": "#F5F5F5",
        "stroke": COLOR_NEUTRAL,
        "color": COLOR_NEUTRAL,
    },
    ROLE_PROVENANCE: {
        "shape": SHAPE_RECTANGLE,
        "fill": COLOR_ARTIFACT_FILL,
        "stroke": COLOR_PROVENANCE,
        "color": COLOR_PROVENANCE,
    },
    ROLE_DECLARED_REFERENCE: {
        "shape": SHAPE_RECTANGLE,
        "fill": COLOR_ARTIFACT_FILL,
        "stroke": COLOR_NEUTRAL,
        "color": COLOR_NEUTRAL,
    },
}

EDGE_STYLE: Dict[str, Dict[str, str]] = {
    EDGE_CONTROL_FLOW: {
        "line": LINE_SOLID,
        "color": COLOR_FACTUAL,
        "width": "normal",
        "arrow": "normal",
        "style_role": ROLE_FACTUAL_OCCURRENCE,
    },
    EDGE_PRODUCES: {
        "line": LINE_SOLID,
        "color": COLOR_FACTUAL,
        "width": "thin",
        "arrow": "normal",
        "style_role": ROLE_PROVENANCE,
    },
    EDGE_CONSUMES: {
        "line": LINE_DASHED,
        "color": COLOR_PROVENANCE,
        "width": "normal",
        "arrow": "normal",
        "style_role": ROLE_PROVENANCE,
    },
    EDGE_DECLARED_REFERENCE: {
        "line": LINE_DOTTED,
        "color": COLOR_NEUTRAL,
        "width": "thin",
        "arrow": "normal",
        "style_role": ROLE_DECLARED_REFERENCE,
    },
    EDGE_INTERVENTION_TARGET: {
        "line": LINE_DOTTED,
        "color": COLOR_INTERVENTION,
        "width": "normal",
        "arrow": "normal",
        "style_role": ROLE_INTERVENTION,
    },
    EDGE_REPLAY_BRANCH: {
        "line": LINE_SOLID,
        "color": COLOR_INTERVENTION,
        "width": "normal",
        "arrow": "normal",
        "style_role": ROLE_INTERVENTION,
    },
    EDGE_COMPARISON: {
        "line": LINE_DOTTED,
        "color": COLOR_NEUTRAL,
        "width": "normal",
        "arrow": "normal",
        "style_role": ROLE_COMPARISON_SAME,
    },
}

EDGE_KIND_TO_ROLE = {
    EDGE_CONTROL_FLOW: ROLE_FACTUAL_OCCURRENCE,
    EDGE_PRODUCES: ROLE_PROVENANCE,
    EDGE_CONSUMES: ROLE_PROVENANCE,
    EDGE_DECLARED_REFERENCE: ROLE_DECLARED_REFERENCE,
    EDGE_INTERVENTION_TARGET: ROLE_INTERVENTION,
    EDGE_REPLAY_BRANCH: ROLE_INTERVENTION,
    EDGE_COMPARISON: ROLE_COMPARISON_SAME,
}


def node_style(style_role: str) -> Mapping[str, str]:
    return NODE_STYLE.get(style_role, NODE_STYLE[ROLE_ARTIFACT])


def edge_style(kind: str) -> Mapping[str, str]:
    return EDGE_STYLE.get(kind, EDGE_STYLE[EDGE_DECLARED_REFERENCE])


def lane_y(lane_id: str) -> float:
    return float(LANE_Y.get(lane_id, 0.0))


def lane_order_index(lane_id: str) -> int:
    try:
        return LANE_ORDER.index(lane_id)
    except ValueError:
        return 99
