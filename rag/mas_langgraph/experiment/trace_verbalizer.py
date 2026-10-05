"""Deterministic participant-facing verbalization of factual execution traces."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple


_NODE_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("build_outline", "organized the teaching material into sections"),
    ("generate_learning_objectives", "created learning objectives"),
    ("initial_retrieve_evidence", "retrieved candidate supporting material"),
    ("retrieve_evidence", "retrieved supporting material"),
    ("atomize_evidence", "extracted concise factual statements from the material"),
    ("atomize_new_evidence", "extracted additional factual statements from new material"),
    ("plan_mcq_blueprints", "created a plan for the question"),
    ("generate_candidate_mcqs", "generated the MCQ and its answer options"),
    ("verify_candidate_mcqs", "checked the question against the teaching material"),
    ("cross_cove_verify", "performed an additional answer check"),
    ("apply_acceptance_gate", "evaluated whether the question satisfied the required criteria"),
    ("check_coverage", "checked coverage of the learning objectives"),
    ("generate_repair_blueprints", "performed coverage repair where needed"),
    ("repair_missing_evidence", "performed coverage repair by gathering more support"),
    ("package_final_quiz", "retained the question for the final quiz"),
)


def _normalize_node(label: str) -> str:
    text = str(label or "").strip()
    # Drop occurrence suffixes: node#1, node[1], node_1 trailing indices in labels.
    text = re.sub(r"[#\[]\d+[\]]?$", "", text)
    text = re.sub(r"\[\d+\]$", "", text)
    return text.strip().lower()


def _phrase_for(node: str) -> str:
    for key, phrase in _NODE_PHRASES:
        if key in node:
            return phrase
    # Unknown nodes: omit rather than expose internal names.
    return ""


def verbalize_trace_events(execution_trace: Sequence[str]) -> Tuple[str, ...]:
    """Translate raw occurrence labels into chronological participant prose facts.

    Repeated generate/verify/repair cycles are compressed into one natural sentence.
    Does not invent provenance or causal roles.
    """
    stages: List[str] = []
    seen_keys: List[str] = []
    for raw in execution_trace or ():
        node = _normalize_node(str(raw))
        if not node:
            continue
        phrase = _phrase_for(node)
        if not phrase:
            continue
        # Collapse exact stage repeats while preserving first chronological appearance.
        key = next((k for k, _ in _NODE_PHRASES if k in node), phrase)
        if key in seen_keys:
            continue
        seen_keys.append(key)
        stages.append(phrase)

    if not stages:
        return (
            "The execution record shows that the question was generated and "
            "moved through the workflow's recorded checks.",
        )

    # Detect generate/verify/repair cycle pattern for compression note.
    has_gen = any("generated the MCQ" in s for s in stages)
    has_verify = any("checked the question" in s for s in stages)
    has_repair = any("coverage repair" in s for s in stages)
    has_accept = any("satisfied the required criteria" in s for s in stages)
    has_pack = any("final quiz" in s for s in stages)

    # Build one or two narrative sentences from stage list.
    early = [
        s
        for s in stages
        if any(
            tok in s
            for tok in (
                "organized the teaching material",
                "created learning objectives",
                "retrieved candidate supporting",
                "retrieved supporting material",
                "extracted concise factual",
                "extracted additional factual",
                "plan for the question",
            )
        )
    ]
    mid = [
        s
        for s in stages
        if any(
            tok in s
            for tok in (
                "generated the MCQ",
                "checked the question",
                "additional answer check",
                "coverage repair",
                "checked coverage of the learning",
            )
        )
    ]
    late = [
        s
        for s in stages
        if any(tok in s for tok in ("satisfied the required", "final quiz"))
    ]

    lines: List[str] = []
    if early:
        if len(early) == 1:
            lines.append(f"The system {early[0]}.")
        elif len(early) == 2:
            lines.append(f"The system {early[0]} and {early[1]}.")
        else:
            lines.append(
                "The system "
                + ", ".join(early[:-1])
                + f", and {early[-1]}."
            )

    if has_gen and has_verify and has_repair:
        lines.append(
            "It generated questions, checked them against the teaching material, "
            "performed an additional answer check where available, and repeated "
            "generation and checking where coverage repair was needed."
        )
    elif mid:
        if len(mid) == 1:
            lines.append(f"It then {mid[0]}.")
        elif len(mid) == 2:
            lines.append(f"It then {mid[0]} and {mid[1]}.")
        else:
            lines.append(
                "It then " + ", ".join(mid[:-1]) + f", and {mid[-1]}."
            )

    if late:
        if has_accept and has_pack:
            lines.append(
                "It evaluated the required criteria and retained the question "
                "for the final quiz."
            )
        elif len(late) == 1:
            lines.append(f"Finally, it {late[0]}.")
        else:
            lines.append(
                "Finally, it " + " and ".join(late) + "."
            )
    elif has_accept or has_pack:
        lines.append(
            "The question was retained after the workflow's recorded checks."
        )

    # Deduplicate accidental repeats.
    out: List[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return tuple(out) if out else (
        "The execution record shows that the question moved through the "
        "workflow and was retained.",
    )
