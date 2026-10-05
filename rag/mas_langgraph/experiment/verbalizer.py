"""Deterministic experiment verbalizer — shared evidence view for all conditions.

Builds compact participant-facing evidence from ``LayExplanationFacts``.
Does not call an LLM. Does not invent causality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .lay_facts import LayExplanationFacts, build_lay_explanation_facts
from .models import ConditionName

_PARTICIPANT_HEADINGS = (
    "Why this question?",
    "What was it based on?",
    "What difference did this information make?",
)


@dataclass(frozen=True)
class ExplanationEvidenceView:
    """Condition-neutral evidence schema shared by Trace / Trace+Prov / AXIS."""

    factual_decision: str
    observed_process: Tuple[str, ...] = ()
    recorded_dependencies: Tuple[str, ...] = ()
    decision_evidence: Tuple[str, ...] = ()
    tested_interventions: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    lay_facts: Optional[LayExplanationFacts] = None

    def to_prompt_text(self, *, omit_empty_sections: bool = True) -> str:
        """Format compact verified facts for the shared final synthesizer."""
        if self.lay_facts is not None:
            return self.lay_facts.to_prompt_text()
        sections = [
            ("Educational purpose", self.observed_process),
            ("Teaching-material basis", self.recorded_dependencies),
            ("Decision evidence", self.decision_evidence),
            ("Controlled-change findings", self.tested_interventions),
            ("Limitation", self.limitations),
        ]
        parts = [f"Recorded decision:\n{self.factual_decision.upper()}"]
        for title, lines in sections:
            if not lines and omit_empty_sections:
                continue
            parts.append(f"{title}:")
            for line in lines:
                parts.append(f"- {line}")
        return "\n".join(parts)


def participant_section_headings() -> Tuple[str, ...]:
    """Deprecated legacy headings — participant output no longer requires them."""
    return _PARTICIPANT_HEADINGS


def build_evidence_view(
    *,
    condition: ConditionName,
    raw_evidence: Mapping[str, Any],
    case_slice: Any = None,
    registry: Any = None,
) -> ExplanationEvidenceView:
    """Populate the shared schema from condition-specific raw evidence."""
    del case_slice, registry
    facts = build_lay_explanation_facts(condition=condition, raw_evidence=raw_evidence)
    if condition == "trace":
        diff: Tuple[str, ...] = ()
    else:
        diff = facts.question_content_findings or facts.controlled_change_findings
    return ExplanationEvidenceView(
        factual_decision=facts.decision,
        observed_process=facts.educational_purpose,
        recorded_dependencies=facts.source_basis,
        decision_evidence=facts.decision_basis,
        tested_interventions=diff,
        limitations=facts.limitations,
        lay_facts=facts,
    )


def verbalize_process_steps(
    execution_trace,
    *,
    factual_decision: str = "",
) -> Tuple[str, ...]:
    """Legacy helper — Trace educational-purpose style lines."""
    facts = build_lay_explanation_facts(
        condition="trace",
        raw_evidence={
            "factual_decision": factual_decision,
            "execution_trace": list(execution_trace or []),
        },
    )
    return facts.educational_purpose


def verbalize_dependencies(
    *,
    relevant_artifacts,
    provenance_relations,
) -> Tuple[str, ...]:
    del provenance_relations
    facts = build_lay_explanation_facts(
        condition="trace_provenance",
        raw_evidence={
            "factual_decision": "accept",
            "relevant_artifacts": list(relevant_artifacts or []),
            "provenance_relations": [],
            "execution_trace": [],
        },
    )
    return facts.source_basis or facts.educational_purpose


def verbalize_decision_evidence(
    *,
    factual_decision: str,
    decision_reasons=(),
    criteria=(),
    required_fixes=(),
) -> Tuple[str, ...]:
    del required_fixes
    facts = build_lay_explanation_facts(
        condition="trace_provenance",
        raw_evidence={
            "factual_decision": factual_decision,
            "decision_reasons": list(decision_reasons or []),
            "criteria": list(criteria or []),
            "relevant_artifacts": [],
            "execution_trace": [],
        },
    )
    return facts.decision_basis


def verbalize_replay_comparison(
    cmp: Any,
    index: int,
    *,
    relevant_artifacts=(),
) -> Tuple[str, ...]:
    """Compatibility wrapper — prefer grouped lay findings in production."""
    del index
    facts = build_lay_explanation_facts(
        condition="axis",
        raw_evidence={
            "factual_decision": "accept",
            "relevant_artifacts": list(relevant_artifacts or []),
            "replay_comparisons": [cmp] if isinstance(cmp, Mapping) else [],
            "execution_trace": [],
        },
    )
    return facts.controlled_change_findings


def verbalize_tested_interventions(
    replay_comparisons,
    *,
    relevant_artifacts=(),
) -> Tuple[str, ...]:
    facts = build_lay_explanation_facts(
        condition="axis",
        raw_evidence={
            "factual_decision": "accept",
            "relevant_artifacts": list(relevant_artifacts or []),
            "replay_comparisons": list(replay_comparisons or []),
            "execution_trace": [],
        },
    )
    return facts.controlled_change_findings
