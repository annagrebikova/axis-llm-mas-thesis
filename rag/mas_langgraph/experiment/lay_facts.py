"""Deterministic lay-user explanation fact selection and aggregation.

Raw researcher evidence remains intact. This module projects a compact
``LayExplanationFacts`` object for participant-facing synthesis only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .models import ConditionName

_VALID_EFFECTS = (
    "decision_effect",
    "criterion_effect",
    "downstream_effect",
    "no_demonstrated_effect",
)
_INVALID_STATUSES = frozenset(
    {
        "replay_unstable",
        "noop_error",
        "intervention_error",
        "outcome_unavailable",
    }
)
_EFFECT_RANK = {
    "decision_effect": 0,
    "criterion_effect": 1,
    "downstream_effect": 2,
    "no_demonstrated_effect": 3,
}

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SEMESTER_RE = re.compile(
    r"\b(spring|fall|summer|winter|autumn)\s+(?:19|20)\d{2}\b", re.I
)
_COPYRIGHT_RE = re.compile(r"(©|\bcopyright\b|\ball rights reserved\b)", re.I)


# Deterministic lay labels for known criterion names (never expose raw names).
_CRITERION_LAY: Dict[str, str] = {
    "target_atom_count": "the amount of supporting knowledge linked to the question",
    "blueprint_evidence_count": "the amount of source evidence linked to the question plan",
    "pdf_supported": "whether the question remained supported by the teaching material",
    "supported_by_pdf": "whether the question remained supported by the teaching material",
    "cross_cove_passed": "whether the additional answer check passed",
    "correct_option_count": "the number of answer options marked as correct",
    "hard_blocking_flags": "the presence of blocking quality issues",
    "structural_reject_flags": "the presence of structural rejection issues",
    "source_refs_present": "whether source references remained available",
    "candidate_status": "the recorded question-assessment status",
    "answer_supported": "whether the correct answer remained supported",
    "stem_supported": "whether the question stem remained supported",
    "blueprint_available": "whether a question plan remained available",
}


@dataclass(frozen=True)
class LayExplanationFacts:
    """Compact participant-facing facts for shared synthesis."""

    decision: str
    educational_purpose: Tuple[str, ...] = ()
    source_basis: Tuple[str, ...] = ()
    decision_basis: Tuple[str, ...] = ()
    controlled_change_findings: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    # Evidence-type projection for the shared synthesizer.
    trace_narrative: Tuple[str, ...] = ()
    relevant_provenance_facts: Tuple[str, ...] = ()
    counterfactual_component_findings: Tuple[str, ...] = ()
    # Researcher diagnostics (not shown to participants).
    selected_source_fact_count: int = 0
    discarded_low_value_snippets: Tuple[str, ...] = ()
    raw_intervention_count: int = 0
    grouped_intervention_count: int = 0
    selected_intervention_finding_count: int = 0
    selected_component_finding_count: int = 0
    primary_effect_status: Optional[str] = None
    primary_effect_target: Optional[str] = None
    primary_effect_details: Tuple[str, ...] = ()
    primary_decision_changed: Optional[bool] = None
    selected_criterion_change_count: int = 0
    selected_downstream_change_type: Optional[str] = None
    question_content_findings: Tuple[str, ...] = ()
    primary_question_content_status: Optional[str] = None
    question_content_control_stable: Optional[bool] = None
    question_changed_parts: Tuple[str, ...] = ()
    factual_question_stem: Optional[str] = None
    counterfactual_question_stem: Optional[str] = None
    intervention_before_question_generation: Optional[bool] = None
    stem_changed: Optional[bool] = None
    correct_answer_changed: Optional[bool] = None
    distractors_changed: Optional[bool] = None
    answer_explanation_changed: Optional[bool] = None
    stem_control_stable: Optional[bool] = None
    correct_answer_control_stable: Optional[bool] = None
    distractors_control_stable: Optional[bool] = None
    answer_explanation_control_stable: Optional[bool] = None
    stem_effect: Optional[str] = None
    correct_answer_effect: Optional[str] = None
    distractor_effect: Optional[str] = None
    answer_explanation_effect: Optional[str] = None
    primary_question_content_intervention: Optional[Dict[str, Any]] = None
    changed_distractor_count: Optional[int] = None
    information_role: Optional[str] = None
    information_role_supported: Optional[bool] = None
    semantic_information_absent: Optional[bool] = None
    derived_information_may_persist: Optional[bool] = None
    intervention_scope: Optional[str] = None
    causal_classification: Optional[str] = None
    content_effect: Optional[bool] = None
    decision_effect: Optional[bool] = None
    question_identity_effect: Optional[bool] = None
    question_identity_effect_known: bool = False
    acceptance_effect: Optional[bool] = None
    acceptance_effect_known: bool = False
    # Evidence-capability flags (condition-blind; derived from facts).
    specific_source_known: bool = False
    recorded_use_known: bool = False
    information_role_known: bool = False
    counterfactual_effect_known: bool = False
    decision_effect_known: bool = False
    primary_information_role_target: Optional[str] = None
    primary_information_description: Optional[str] = None
    primary_workflow_effect_target: Optional[str] = None
    source_basis_unavailable: bool = False

    def to_prompt_text(self) -> str:
        parts: List[str] = [f"Recorded decision:\n{self.decision}"]

        def _sec(title: str, lines: Tuple[str, ...]) -> None:
            if not lines:
                return
            parts.append(f"{title}:")
            for line in lines:
                parts.append(f"- {line}")

        parts.append("Evidence capabilities:")
        parts.append(
            f"- specific_source_known = {str(bool(self.specific_source_known)).lower()}"
        )
        parts.append(
            f"- recorded_use_known = {str(bool(self.recorded_use_known)).lower()}"
        )
        parts.append(
            f"- information_role_known = {str(bool(self.information_role_known)).lower()}"
        )
        # Prompt-facing alias: avoid the word "counterfactual" in Trace evidence text.
        parts.append(
            f"- role_effect_known = "
            f"{str(bool(self.counterfactual_effect_known)).lower()}"
        )
        parts.append(
            f"- decision_effect_known = {str(bool(self.decision_effect_known)).lower()}"
        )
        parts.append(
            "- participant_presentation = do not enumerate workflow stages; "
            "do not use generic process chronology as explanation content"
        )

        # Stage-by-stage execution narrative remains in trace_narrative for
        # researcher diagnostics; it is not participant synthesis content.
        _sec(
            "Relevant provenance facts",
            self.relevant_provenance_facts
            or (
                self.source_basis
                if self.specific_source_known and not self.information_role_known
                else ()
            ),
        )
        _sec(
            "Supported counterfactual component findings",
            self.counterfactual_component_findings
            or (
                self.question_content_findings if self.information_role_known else ()
            ),
        )
        if self.information_role_known:
            parts.append("Tested teaching information:")
            if self.primary_information_description:
                parts.append(f"- description = {self.primary_information_description}")
            if self.information_role:
                parts.append(f"- role = {self.information_role}")
            if self.information_role_supported is not None:
                parts.append(
                    "- role_supported = "
                    f"{str(bool(self.information_role_supported)).lower()}"
                )
            parts.append("Observed question-content effect:")
            if self.stem_changed is not None:
                parts.append(f"- stem_changed = {str(bool(self.stem_changed)).lower()}")
            if self.correct_answer_changed is not None:
                parts.append(
                    "- correct_answer_changed = "
                    f"{str(bool(self.correct_answer_changed)).lower()}"
                )
            if self.distractors_changed is not None:
                parts.append(
                    "- distractors_changed = "
                    f"{str(bool(self.distractors_changed)).lower()}"
                )
            if self.factual_question_stem:
                parts.append(f'- factual_question = "{self.factual_question_stem}"')
            if self.counterfactual_question_stem:
                parts.append(
                    f'- alternative_question = "{self.counterfactual_question_stem}"'
                )
        if self.source_basis_unavailable and not self.information_role_known:
            _sec("Limitation", self.limitations[:1] if self.limitations else ())
        elif (
            self.limitations
            and not (
                self.counterfactual_component_findings or self.information_role_known
            )
            and not (
                self.specific_source_known and self.recorded_use_known
            )
        ):
            _sec("Limitation", self.limitations[:1])
        return "\n".join(parts)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "selected_source_fact_count": self.selected_source_fact_count,
            "discarded_low_value_snippets": list(self.discarded_low_value_snippets),
            "raw_intervention_count": self.raw_intervention_count,
            "grouped_intervention_count": self.grouped_intervention_count,
            "selected_intervention_finding_count": (
                self.selected_intervention_finding_count
            ),
            "selected_component_finding_count": self.selected_component_finding_count,
            "primary_effect_status": self.primary_effect_status,
            "primary_effect_target": self.primary_effect_target,
            "primary_effect_details": list(self.primary_effect_details),
            "primary_decision_changed": self.primary_decision_changed,
            "selected_criterion_change_count": self.selected_criterion_change_count,
            "selected_downstream_change_type": self.selected_downstream_change_type,
            "primary_question_content_status": self.primary_question_content_status,
            "question_content_control_stable": self.question_content_control_stable,
            "question_changed_parts": list(self.question_changed_parts),
            "factual_question_stem": self.factual_question_stem,
            "counterfactual_question_stem": self.counterfactual_question_stem,
            "intervention_before_question_generation": (
                self.intervention_before_question_generation
            ),
            "stem_changed": self.stem_changed,
            "correct_answer_changed": self.correct_answer_changed,
            "distractors_changed": self.distractors_changed,
            "answer_explanation_changed": self.answer_explanation_changed,
            "stem_control_stable": self.stem_control_stable,
            "correct_answer_control_stable": self.correct_answer_control_stable,
            "distractors_control_stable": self.distractors_control_stable,
            "answer_explanation_control_stable": self.answer_explanation_control_stable,
            "stem_effect": self.stem_effect,
            "correct_answer_effect": self.correct_answer_effect,
            "distractor_effect": self.distractor_effect,
            "answer_explanation_effect": self.answer_explanation_effect,
            "changed_distractor_count": self.changed_distractor_count,
            "information_role": self.information_role,
            "information_role_supported": self.information_role_supported,
            "semantic_information_absent": self.semantic_information_absent,
            "derived_information_may_persist": self.derived_information_may_persist,
            "intervention_scope": self.intervention_scope,
            "causal_classification": self.causal_classification,
            "content_effect": self.content_effect,
            "decision_effect": self.decision_effect,
            "question_identity_effect": self.question_identity_effect,
            "question_identity_effect_known": self.question_identity_effect_known,
            "acceptance_effect": self.acceptance_effect,
            "acceptance_effect_known": self.acceptance_effect_known,
            "specific_source_known": self.specific_source_known,
            "recorded_use_known": self.recorded_use_known,
            "information_role_known": self.information_role_known,
            "counterfactual_effect_known": self.counterfactual_effect_known,
            "decision_effect_known": self.decision_effect_known,
            "primary_information_role_target": self.primary_information_role_target,
            "primary_information_description": self.primary_information_description,
            "primary_workflow_effect_target": self.primary_workflow_effect_target
            or self.primary_effect_target,
            "source_basis_unavailable": self.source_basis_unavailable,
            "primary_question_content_intervention": dict(
                self.primary_question_content_intervention or {}
            ),
        }


@dataclass(frozen=True)
class CriterionChangeSummary:
    name: str
    lay_label: str
    noop_value: Any
    counterfactual_value: Any
    decision_relevant: bool
    detail_sentence: str


@dataclass(frozen=True)
class DownstreamChangeSummary:
    change_type: str  # path_repair | path_change | recorded_use | artifact_change | generic
    detail_sentence: str


@dataclass
class _GroupedIntervention:
    target_key: str
    target_description: str
    operation: str
    statuses: List[str] = field(default_factory=list)
    comparisons: List[Mapping[str, Any]] = field(default_factory=list)

    @property
    def valid_test_count(self) -> int:
        return sum(1 for s in self.statuses if s in _VALID_EFFECTS)

    def counts(self) -> Dict[str, int]:
        out = {k: 0 for k in _VALID_EFFECTS}
        for s in self.statuses:
            if s in out:
                out[s] += 1
        return out

    def strongest_effect(self) -> Optional[str]:
        valid = [s for s in self.statuses if s in _EFFECT_RANK]
        if not valid:
            return None
        return sorted(valid, key=lambda s: _EFFECT_RANK[s])[0]


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def is_metadata_only_snippet(text: str) -> bool:
    """True when a snippet is mostly title/author/email/date metadata.

    Conservative: only reject clear title-slide / header-footer cards.
    Short teaching phrases (e.g. ``Place k means at random.``) stay usable.
    """
    raw = str(text or "")
    cleaned = _clean_text(raw)
    if not cleaned:
        return True

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        lines = [cleaned]

    has_email = bool(_EMAIL_RE.search(cleaned))
    has_semester = bool(_SEMESTER_RE.search(cleaned))
    has_copyright = bool(_COPYRIGHT_RE.search(cleaned))
    strong_meta = sum([has_email, has_semester, has_copyright])

    # Explicit title-card pattern (common slide opening).
    if strong_meta >= 1 and len(lines) >= 2:
        return True
    if has_email and has_semester:
        return True
    if has_copyright and len(cleaned) < 160:
        return True

    # Single short author / semester / year-only lines.
    if len(lines) == 1:
        ln = lines[0]
        words = ln.split()
        if has_email or has_semester or has_copyright:
            return True
        if _YEAR_RE.fullmatch(ln.strip()):
            return True
        # Bare proper name (2–4 Title Case tokens, no sentence punctuation).
        if (
            1 <= len(words) <= 4
            and not any(ch in ln for ch in ".!?;:")
            and all(w[:1].isupper() for w in words if w.isalpha())
            and not any(
                tok in ln.lower()
                for tok in (
                    "learn",
                    "data",
                    "model",
                    "algorithm",
                    "cluster",
                    "means",
                    "pattern",
                    "unlabeled",
                    "labeled",
                    "uses",
                    "without",
                    "train",
                    "place",
                    "group",
                    "identify",
                    "define",
                )
            )
        ):
            return True
        return False

    # Multi-line: metadata if every line is short label-like and ≥2 meta cues.
    meta_lines = 0
    for ln in lines:
        words = ln.split()
        if (
            _EMAIL_RE.search(ln)
            or _SEMESTER_RE.search(ln)
            or _COPYRIGHT_RE.search(ln)
            or _YEAR_RE.fullmatch(ln.strip())
            or (
                len(words) <= 4
                and not any(ch in ln for ch in ".!?;:")
                and (ln.istitle() or ln.isupper())
            )
        ):
            meta_lines += 1
    if meta_lines == len(lines) and (strong_meta >= 1 or meta_lines >= 3):
        return True
    return False


def _snippet(row: Mapping[str, Any], *, max_chars: int = 180) -> str:
    preview = row.get("content_preview") or row.get("content") or ""
    text = _clean_text(str(preview))
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0]
    return (cut or text[: max_chars - 1]).rstrip(".,;:") + "…"


def _decision_phrase(decision: str) -> str:
    d = str(decision or "").strip().lower()
    mapping = {
        "accept": "ACCEPT",
        "accepted": "ACCEPT",
        "revise": "REVISE",
        "needs_revision": "REVISE",
        "reject": "REJECT",
        "rejected": "REJECT",
    }
    return mapping.get(d, d.upper() or "UNKNOWN")


def _target_key(cmp: Mapping[str, Any]) -> str:
    sig = cmp.get("intervention_signature") or {}
    if not isinstance(sig, Mapping):
        return "unknown"
    tr = sig.get("target_ref")
    if isinstance(tr, Mapping):
        return str(tr.get("key") or tr.get("logical_id") or "unknown")
    return str(tr or "unknown")


def _operation(cmp: Mapping[str, Any]) -> str:
    sig = cmp.get("intervention_signature") or {}
    if isinstance(sig, Mapping):
        return str(sig.get("operation") or "REMOVE").upper()
    return "REMOVE"


def _lookup_artifact(
    artifacts: Sequence[Mapping[str, Any]], key: str
) -> Optional[Mapping[str, Any]]:
    for row in artifacts:
        if str(row.get("ref") or "") == key:
            return row
    return None


def _describe_target(
    *,
    key: str,
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    row = _lookup_artifact(artifacts, key)
    if row is None:
        return "supporting teaching-material information"
    atype = str(row.get("artifact_type") or "")
    snippet = _snippet(row, max_chars=100)
    if atype == "EvidenceChunk":
        if snippet and not is_metadata_only_snippet(snippet):
            return f'supporting source information stating "{snippet}"'
        return "supporting source information from the teaching material"
    if atype == "KnowledgeAtom":
        if snippet and not is_metadata_only_snippet(snippet):
            return f'a derived teaching fact stating "{snippet}"'
        return "a derived teaching fact"
    if atype == "CrossCoVeReport":
        return "the additional answer-check result"
    if atype == "EvidenceCheckReport":
        return "the source-support check result"
    return "supporting recorded information"


def _is_noisy_extraction_snippet(text: str) -> bool:
    """Detect raw PDF extraction that is poor participant-facing evidence."""
    t = str(text or "").strip()
    if not t:
        return True
    low = t.lower()
    # Math/input notation fragments common in slide OCR.
    noisy_markers = (
        "input: inputs",
        "inputs x =",
        "x = {x",
        "{x ,",
        "{x,",
        "! try to",
        "…, x }",
        "..., x}",
    )
    if any(m in low for m in noisy_markers):
        return True
    # Heavy symbol / fragment density.
    symbol_n = sum(1 for ch in t if ch in "{}=![]<>|\\")
    if symbol_n >= 4 and len(t) < 220:
        return True
    return False


def _token_overlap_score(text: str, *contexts: str) -> int:
    stop = {
        "the", "a", "an", "of", "to", "in", "and", "or", "for", "is", "are",
        "that", "this", "with", "on", "as", "by", "from", "be", "was", "were",
        "it", "its", "at", "into", "without", "which", "what",
    }
    ctx = " ".join(str(c or "").lower() for c in contexts)
    ctx_tokens = {
        w for w in re.findall(r"[a-z]{3,}", ctx) if w not in stop
    }
    if not ctx_tokens:
        return 0
    text_tokens = set(re.findall(r"[a-z]{3,}", str(text or "").lower()))
    return len(ctx_tokens & text_tokens)


def _weighted_relevance_score(
    text: str,
    *,
    question_stem: str = "",
    correct_answer: str = "",
    learning_objective: str = "",
) -> int:
    """Prefer correct-answer overlap, then stem, then learning objective."""
    return (
        4 * _token_overlap_score(text, correct_answer)
        + 3 * _token_overlap_score(text, question_stem)
        + 1 * _token_overlap_score(text, learning_objective)
    )


def select_source_facts(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    max_facts: int = 3,
    question_stem: str = "",
    correct_answer: str = "",
    learning_objective: str = "",
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return (selected_facts, discarded_low_value_previews).

    Participant ranking prefers clean KnowledgeAtoms over noisy EvidenceChunks.
    Weighted overlap favors correct-answer and stem relevance.
    """
    discarded: List[str] = []
    ranked: List[Tuple[int, int, str]] = []  # (priority, -score, snippet)

    section_line = ""
    lo_line = ""
    lo_text = learning_objective
    for row in artifacts:
        atype = str(row.get("artifact_type") or "")
        if atype == "DocSection" and not section_line:
            title = row.get("title") or row.get("section_id")
            if title:
                section_line = f'Topic: {title}.'
        if atype == "LearningObjective" and not lo_line:
            text = row.get("objective_text") or row.get("title") or row.get("lo_id")
            if text:
                lo_line = f'Learning objective: {text}.'
                if not lo_text:
                    lo_text = str(text)

    for row in artifacts:
        atype = str(row.get("artifact_type") or "")
        if atype != "KnowledgeAtom":
            continue
        snip = _snippet(row, max_chars=160)
        if not snip or is_metadata_only_snippet(snip) or _is_noisy_extraction_snippet(snip):
            if snip:
                discarded.append(snip)
            continue
        score = _weighted_relevance_score(
            snip,
            question_stem=question_stem,
            correct_answer=correct_answer,
            learning_objective=lo_text,
        )
        ranked.append((0, -score, snip))

    for row in artifacts:
        atype = str(row.get("artifact_type") or "")
        if atype != "EvidenceChunk":
            continue
        snip = _snippet(row)
        if not snip:
            continue
        if is_metadata_only_snippet(row.get("content") or snip):
            discarded.append(snip)
            continue
        if _is_noisy_extraction_snippet(snip):
            discarded.append(snip)
            score = _weighted_relevance_score(
                snip,
                question_stem=question_stem,
                correct_answer=correct_answer,
                learning_objective=lo_text,
            )
            ranked.append((9, -score, snip))
            continue
        score = _weighted_relevance_score(
            snip,
            question_stem=question_stem,
            correct_answer=correct_answer,
            learning_objective=lo_text,
        )
        ranked.append((1, -score, snip))

    selected: List[str] = []
    seen_norm: set[str] = set()
    for _prio, _ov, snip in sorted(ranked, key=lambda x: (x[0], x[1], x[2])):
        norm = snip.lower()
        if norm in seen_norm:
            continue
        if any(norm in s.lower() or s.lower() in norm for s in selected):
            continue
        seen_norm.add(norm)
        selected.append(snip)
        if len(selected) >= max_facts:
            break

    if not selected:
        if lo_line:
            selected.append(lo_line)
        elif section_line:
            selected.append(section_line)
        elif discarded:
            selected.append(
                "Source material from the teaching document was recorded as an "
                "input, but the available excerpts were mostly document metadata."
            )

    cleaned_facts: List[str] = []
    for s in selected:
        if s.startswith(("Topic:", "Learning objective:", "Source material")):
            cleaned_facts.append(s)
        else:
            cleaned_facts.append(s)
    return tuple(cleaned_facts[:max_facts]), tuple(dict.fromkeys(discarded))


def verbalize_provenance_links(
    *,
    provenance_relations: Sequence[Any],
    source_facts: Sequence[str],
) -> Tuple[str, ...]:
    """Natural observational provenance statements (no causal verbs)."""
    _ = provenance_relations  # relations remain available to callers; unused here.
    if not source_facts:
        return ()
    lines: List[str] = []
    primary = source_facts[0]
    lines.append(
        "The records link the question to the teaching fact that "
        f"{primary.rstrip('.')}."
    )
    extras = [f for f in source_facts[1:3] if f and f != primary]
    if extras:
        if len(extras) == 1:
            lines.append(f"Other linked material described {extras[0].rstrip('.')}.")
        else:
            lines.append(
                "Other linked material described "
                + "; ".join(e.rstrip(".") for e in extras)
                + "."
            )
    return tuple(lines)


def _educational_purpose(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    condition: ConditionName,
    execution_trace: Sequence[str],
) -> Tuple[str, ...]:
    if condition == "trace":
        # Trace may only speak from execution record.
        lines = [
            "The execution record shows that the question was generated after the "
            "teaching material and learning objectives were processed and that it "
            "passed the workflow's checks."
        ]
        return tuple(lines)

    section = None
    lo = None
    for row in artifacts:
        atype = str(row.get("artifact_type") or "")
        if atype == "DocSection" and section is None:
            section = row.get("title") or row.get("section_id")
        if atype == "LearningObjective" and lo is None:
            lo = row.get("objective_text") or row.get("title") or row.get("lo_id")
    out: List[str] = []
    if section and lo:
        out.append(
            f'The question targeted the topic "{section}" and the learning '
            f'objective "{lo}".'
        )
    elif lo:
        out.append(f'The question targeted the learning objective "{lo}".')
    elif section:
        out.append(f'The question targeted the topic "{section}".')
    else:
        out.append(
            "The question was generated from the teaching material after "
            "learning objectives were defined."
        )
    return tuple(out)


def _decision_basis(
    *,
    condition: ConditionName,
    decision: str,
    artifacts: Sequence[Mapping[str, Any]],
    decision_reasons: Sequence[str],
    criteria: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    dec = _decision_phrase(decision)
    if condition == "trace":
        return (
            f"The record shows that the question underwent a source-support check "
            f"and an additional answer check before {dec} was recorded.",
            "The execution record does not allow us to test whether removing a "
            "particular piece of information would have changed the question or "
            "its final decision.",
        )

    lines: List[str] = []
    by_type = {str(r.get("artifact_type") or "") for r in artifacts}
    if "EvidenceCheckReport" in by_type or any(
        str(c.get("name")) == "supported_by_pdf" and c.get("value") for c in criteria
    ):
        lines.append("The question was supported by the teaching material.")
    if "CrossCoVeReport" in by_type or any(
        str(c.get("name")) == "cross_cove_passed" and c.get("value") for c in criteria
    ):
        lines.append("The additional answer check passed.")
    hard = next(
        (c for c in criteria if str(c.get("name")) == "hard_blocking_flags"),
        None,
    )
    if hard is not None and not hard.get("value"):
        lines.append("No blocking quality issue was recorded.")
    # Soften reasons only when they add something beyond structured checks.
    existing_l = " ".join(lines).lower()
    for r in decision_reasons[:2]:
        text = str(r or "").strip()
        if not text:
            continue
        text = text.replace("Cross-CoVe", "additional answer check")
        text = text.replace("CrossCoVe", "additional answer check")
        low = text.lower()
        if "candidate passed" in low and (
            "supported by the teaching material" in existing_l
            or "additional answer check passed" in existing_l
        ):
            continue
        if text not in lines:
            lines.append(text)
    if not lines:
        lines.append(f"The recorded decision for this question was {dec}.")
    return tuple(dict.fromkeys(lines))


def group_interventions(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    artifacts: Sequence[Mapping[str, Any]],
) -> List[_GroupedIntervention]:
    groups: Dict[Tuple[str, str], _GroupedIntervention] = {}
    for cmp in comparisons:
        if not isinstance(cmp, Mapping):
            continue
        status = str(cmp.get("evidence_status") or "")
        key = _target_key(cmp)
        op = _operation(cmp)
        gkey = (op, key)
        if gkey not in groups:
            groups[gkey] = _GroupedIntervention(
                target_key=key,
                target_description=_describe_target(key=key, artifacts=artifacts),
                operation=op,
            )
        groups[gkey].statuses.append(status)
        groups[gkey].comparisons.append(cmp)

    def sort_key(g: _GroupedIntervention) -> Tuple[int, str]:
        strong = g.strongest_effect()
        rank = _EFFECT_RANK.get(strong or "", 99)
        return (rank, g.target_key)

    return sorted(groups.values(), key=sort_key)


def _delta_values(delta: Mapping[str, Any]) -> Tuple[Any, Any]:
    """NOOP / CF values from research or in-memory delta dicts."""
    noop = delta.get("noop", delta.get("noop_value"))
    cf = delta.get("counterfactual", delta.get("counterfactual_value"))
    return noop, cf


def _format_simple_value(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        words = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        return words.get(value, str(value))
    if isinstance(value, str) and value.strip() and len(value) < 40:
        return value.strip()
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) <= 3:
        return f"{len(value)} items" if value else "none"
    return None


def _criterion_detail_sentence(
    *,
    lay_label: str,
    noop_value: Any,
    cf_value: Any,
) -> str:
    n_s = _format_simple_value(noop_value)
    c_s = _format_simple_value(cf_value)
    if isinstance(noop_value, bool) and isinstance(cf_value, bool):
        if lay_label.startswith("whether "):
            prop = lay_label[len("whether ") :]
            if noop_value and not cf_value:
                return f"{prop} changed from yes to no"
            if (not noop_value) and cf_value:
                return f"{prop} changed from no to yes"
        return f"{lay_label} changed"
    if n_s is not None and c_s is not None and n_s != c_s:
        return f"{lay_label} changed from {n_s} to {c_s}"
    return f"{lay_label} changed"


def extract_criterion_changes(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    max_changes: int = 2,
) -> List[CriterionChangeSummary]:
    """Select up to ``max_changes`` human-readable CF-vs-NOOP criterion changes."""
    ranked: List[Tuple[int, CriterionChangeSummary]] = []
    seen_names: set[str] = set()
    for cmp in comparisons:
        if str(cmp.get("evidence_status") or "") != "criterion_effect":
            continue
        for delta in cmp.get("criterion_deltas") or []:
            if not isinstance(delta, Mapping):
                continue
            if not delta.get("counterfactual_differs_from_noop"):
                continue
            name = str(delta.get("name") or "")
            if not name or name in seen_names:
                continue
            noop_v, cf_v = _delta_values(delta)
            lay = _CRITERION_LAY.get(name)
            if lay is None:
                # Unknown names: generic label, still useful without exposing the name.
                lay = "a tracked assessment property"
                prio = 2
            else:
                prio = 0 if delta.get("decision_relevant") else 1
            detail = _criterion_detail_sentence(
                lay_label=lay, noop_value=noop_v, cf_value=cf_v
            )
            # Capitalize first letter for standalone sentence fragments.
            detail_cap = detail[:1].upper() + detail[1:] if detail else detail
            ranked.append(
                (
                    prio,
                    CriterionChangeSummary(
                        name=name,
                        lay_label=lay,
                        noop_value=noop_v,
                        counterfactual_value=cf_v,
                        decision_relevant=bool(delta.get("decision_relevant")),
                        detail_sentence=detail_cap,
                    ),
                )
            )
            seen_names.add(name)
    ranked.sort(key=lambda x: (x[0], x[1].name))
    return [item for _, item in ranked[:max_changes]]


def _path_nodes(path: Sequence[Any]) -> List[str]:
    return [str(x) for x in (path or [])]


def _has_repair_step(only_cf: Sequence[str]) -> bool:
    return any("repair_missing_evidence" in n for n in only_cf)


def extract_downstream_change(
    comparisons: Sequence[Mapping[str, Any]],
) -> Optional[DownstreamChangeSummary]:
    """Derive a specific downstream description from comparison fields."""
    candidates: List[DownstreamChangeSummary] = []
    rank = {
        "path_repair": 0,
        "path_change": 1,
        "artifact_change": 2,
        "recorded_use": 3,
        "generic": 4,
    }
    for cmp in comparisons:
        status = str(cmp.get("evidence_status") or "")
        if status not in {
            "downstream_effect",
            "criterion_effect",
            "decision_effect",
        }:
            continue
        noop_path = _path_nodes(cmp.get("noop_path") or [])
        cf_path = _path_nodes(cmp.get("counterfactual_path") or [])
        path_diff = (
            noop_path != cf_path or cmp.get("first_path_divergence") is not None
        )
        only_cf = [n for n in cf_path if n not in set(noop_path)]
        added_prov = list(cmp.get("added_provenance_relations") or [])
        removed_prov = list(cmp.get("removed_provenance_relations") or [])
        added_art = list(cmp.get("added_artifacts") or [])
        removed_art = list(cmp.get("removed_artifacts") or [])
        payload = list(cmp.get("payload_changed_artifacts") or [])

        if path_diff and _has_repair_step(only_cf):
            candidates.append(
                DownstreamChangeSummary(
                    change_type="path_repair",
                    detail_sentence=(
                        "The tested change triggered additional evidence-repair "
                        "processing."
                    ),
                )
            )
            continue
        if path_diff:
            candidates.append(
                DownstreamChangeSummary(
                    change_type="path_change",
                    detail_sentence=(
                        "The tested change altered the later sequence of "
                        "processing steps."
                    ),
                )
            )
            continue
        if (added_art or removed_art or payload) and not path_diff:
            candidates.append(
                DownstreamChangeSummary(
                    change_type="artifact_change",
                    detail_sentence=(
                        "The tested change altered information produced later in "
                        "the workflow, while the sequence of steps stayed the same."
                    ),
                )
            )
            continue
        if (added_prov or removed_prov) and not path_diff:
            candidates.append(
                DownstreamChangeSummary(
                    change_type="recorded_use",
                    detail_sentence=(
                        "The sequence of processing steps stayed the same, but the "
                        "information recorded as being used later in the workflow "
                        "changed."
                    ),
                )
            )
            continue
        if status == "downstream_effect":
            candidates.append(
                DownstreamChangeSummary(
                    change_type="generic",
                    detail_sentence=(
                        "Later workflow behaviour changed in a tracked way, "
                        "although the final decision did not."
                    ),
                )
            )
    if not candidates:
        return None
    candidates.sort(key=lambda c: rank.get(c.change_type, 99))
    return candidates[0]


def _decision_from_group(group: _GroupedIntervention) -> Tuple[str, bool]:
    """Return (decision phrase, decision_changed) from strongest comparison."""
    strong = group.strongest_effect()
    for cmp in group.comparisons:
        if str(cmp.get("evidence_status") or "") != strong:
            continue
        fact = cmp.get("factual") or {}
        noop = cmp.get("noop") or {}
        cf = cmp.get("counterfactual") or {}
        if not isinstance(fact, Mapping):
            fact = {}
        if not isinstance(noop, Mapping):
            noop = {}
        if not isinstance(cf, Mapping):
            cf = {}
        dec = _decision_phrase(
            str(noop.get("decision") or fact.get("decision") or "ACCEPT")
        )
        changed = bool(
            cf.get("decision") is not None
            and noop.get("decision") is not None
            and cf.get("decision") != noop.get("decision")
        )
        return dec, changed
    return "ACCEPT", strong == "decision_effect"


def _finding_lines(group: _GroupedIntervention) -> Tuple[List[str], Dict[str, Any]]:
    """Return participant lines and structured metadata for one group."""
    counts = group.counts()
    valid_n = group.valid_test_count
    meta: Dict[str, Any] = {
        "effect_status": group.strongest_effect(),
        "effect_target": group.target_key,
        "criterion_changes": [],
        "downstream_type": None,
        "decision_changed": None,
        "effect_details": [],
    }
    if valid_n <= 0:
        return [], meta

    op = group.operation
    target = group.target_description
    lines: List[str] = []
    if valid_n == 1:
        if op == "REMOVE":
            lines.append(f"The system tested what would happen without {target}.")
        elif op == "ADD":
            lines.append(
                f"The system tested what would happen if {target} were added."
            )
        else:
            lines.append(
                f"The system tested a controlled change involving {target}."
            )
    else:
        if op == "REMOVE":
            lines.append(
                f"The system tested removing the same supporting information "
                f"at {valid_n} points in the workflow."
            )
        else:
            lines.append(
                f"The system tested the same controlled change involving {target} "
                f"in {valid_n} valid tests."
            )

    strong = group.strongest_effect()
    decision, decision_changed = _decision_from_group(group)
    meta["decision_changed"] = decision_changed

    if strong == "decision_effect":
        detail = (
            f"In the strongest valid test, the final decision changed "
            f"(it was no longer {decision})."
            if decision
            else "In the strongest valid test, the final decision changed."
        )
        lines.append(detail)
        meta["effect_details"].append(detail)

    elif strong == "criterion_effect":
        changes = extract_criterion_changes(group.comparisons, max_changes=2)
        meta["criterion_changes"] = changes
        if changes:
            ch0 = changes[0]
            n_s = _format_simple_value(ch0.noop_value)
            c_s = _format_simple_value(ch0.counterfactual_value)
            if len(changes) == 1 and n_s and c_s and n_s != c_s:
                detail = (
                    f"In a valid test, {ch0.lay_label} changed from {n_s} to {c_s}, "
                    f"while the final decision remained {decision}."
                )
            elif len(changes) == 1:
                detail = (
                    f"In a valid test, {ch0.lay_label} changed, while the final "
                    f"decision remained {decision}."
                )
            else:
                joined = "; ".join(c.detail_sentence for c in changes)
                detail = (
                    f"In valid tests, tracked assessment properties changed "
                    f"({joined.lower()}), while the final decision remained "
                    f"{decision}."
                )
            lines.append(detail)
            meta["effect_details"].append(detail)
            lines.append(
                "This shows that the tested information affected the assessment, "
                "although it was not shown to be decisive for the final decision."
            )
        else:
            detail = (
                "A tracked assessment property changed in at least one valid test, "
                f"while the final decision remained {decision}."
            )
            lines.append(detail)
            meta["effect_details"].append(detail)

    elif strong == "downstream_effect":
        down = extract_downstream_change(group.comparisons)
        if down is not None:
            meta["downstream_type"] = down.change_type
            lines.append(down.detail_sentence)
            meta["effect_details"].append(down.detail_sentence)
        lines.append(
            f"The final decision remained {decision} in the valid tests."
        )
        lines.append(
            "This demonstrates an effect on the workflow, but these tests did not "
            "show that the information was decisive for the final decision."
        )

    elif strong == "no_demonstrated_effect":
        lines.append(
            f"The final decision remained {decision} in all valid tests of this "
            "information, and no tracked assessment or workflow change was "
            "demonstrated."
        )
        lines.append(
            "This applies only to the tested changes and does not mean the "
            "information can be ignored in general."
        )
        meta["effect_details"].append(lines[-2])

    if (
        strong in {"downstream_effect", "criterion_effect", "decision_effect"}
        and counts.get("no_demonstrated_effect", 0) > 0
        and strong != "no_demonstrated_effect"
    ):
        lines.append(
            f"In {counts['no_demonstrated_effect']} of the valid tests, no change "
            "to the tracked result was observed."
        )
    return lines, meta


def select_controlled_findings(
    groups: Sequence[_GroupedIntervention],
) -> Tuple[Tuple[str, ...], int, Dict[str, Any]]:
    """Pick strongest finding (+ optional complementary); return lines, count, meta."""
    empty_meta: Dict[str, Any] = {
        "primary_effect_status": None,
        "primary_effect_target": None,
        "primary_effect_details": (),
        "primary_decision_changed": None,
        "selected_criterion_change_count": 0,
        "selected_downstream_change_type": None,
    }
    usable = [g for g in groups if g.valid_test_count > 0]
    if not usable:
        return (), 0, empty_meta

    def priority(g: _GroupedIntervention) -> Tuple[int, int]:
        strong = g.strongest_effect() or "no_demonstrated_effect"
        return (_EFFECT_RANK.get(strong, 99), -g.valid_test_count)

    ordered = sorted(usable, key=priority)
    primary = ordered[0]
    lines, pmeta = _finding_lines(primary)
    selected = 1
    out_meta: Dict[str, Any] = {
        "primary_effect_status": pmeta.get("effect_status"),
        "primary_effect_target": pmeta.get("effect_target"),
        "primary_effect_details": tuple(pmeta.get("effect_details") or ()),
        "primary_decision_changed": pmeta.get("decision_changed"),
        "selected_criterion_change_count": len(pmeta.get("criterion_changes") or []),
        "selected_downstream_change_type": pmeta.get("downstream_type"),
    }

    primary_status = primary.strongest_effect()

    # Complementary finding on a different explanatory dimension / target.
    if primary_status == "criterion_effect":
        # Prefer a different group with a concrete downstream effect.
        for g in ordered[1:]:
            if g.strongest_effect() != "downstream_effect":
                continue
            extra, emeta = _finding_lines(g)
            if not extra:
                continue
            # Keep compact: tested + specific downstream + decision unchanged.
            compact = [extra[0]]
            if emeta.get("downstream_type"):
                down = extract_downstream_change(g.comparisons)
                if down is not None:
                    compact.append(down.detail_sentence)
                    out_meta["selected_downstream_change_type"] = down.change_type
            decision, _ = _decision_from_group(g)
            compact.append(f"The final decision remained {decision}.")
            lines.extend(compact)
            selected += 1
            break
        else:
            # Same-group downstream enrichment when primary group also has path/use diffs.
            down = extract_downstream_change(primary.comparisons)
            if down is not None and down.detail_sentence not in lines:
                lines.append(down.detail_sentence)
                out_meta["selected_downstream_change_type"] = down.change_type

    elif primary_status != "no_demonstrated_effect":
        for g in ordered[1:]:
            if g.strongest_effect() == "no_demonstrated_effect":
                extra, _ = _finding_lines(g)
                if extra:
                    lines.append(extra[0])
                    if len(extra) > 1:
                        lines.append(extra[1])
                    selected += 1
                break

    return tuple(dict.fromkeys(lines)), selected, out_meta


def build_lay_explanation_facts(
    *,
    condition: ConditionName,
    raw_evidence: Mapping[str, Any],
) -> LayExplanationFacts:
    """Deterministic projection from raw condition evidence to lay facts."""
    decision = _decision_phrase(
        str(raw_evidence.get("factual_decision") or "")
    )
    arts = list(raw_evidence.get("relevant_artifacts") or [])
    comparisons = [
        c
        for c in (raw_evidence.get("replay_comparisons") or [])
        if isinstance(c, Mapping)
    ]

    if condition == "trace":
        from .trace_verbalizer import verbalize_trace_events

        trace_lines = verbalize_trace_events(
            list(raw_evidence.get("execution_trace") or [])
        )
        limitations = (
            "The trace shows this sequence of events, but not which specific "
            "teaching facts were responsible for the exact content of this question.",
        )
        return LayExplanationFacts(
            decision=decision,
            educational_purpose=trace_lines,
            trace_narrative=trace_lines,
            source_basis=(),
            decision_basis=(),
            controlled_change_findings=(),
            limitations=limitations,
            question_content_findings=(),
            selected_source_fact_count=0,
            discarded_low_value_snippets=(),
            raw_intervention_count=0,
            grouped_intervention_count=0,
            selected_intervention_finding_count=0,
            specific_source_known=False,
            recorded_use_known=False,
            information_role_known=False,
            counterfactual_effect_known=False,
            decision_effect_known=False,
            source_basis_unavailable=True,
        )

    from .trace_verbalizer import verbalize_trace_events

    trace_lines = verbalize_trace_events(
        list(raw_evidence.get("execution_trace") or [])
    )
    purpose = _educational_purpose(
        arts,
        condition=condition,
        execution_trace=list(raw_evidence.get("execution_trace") or []),
    )
    # Prefer richer chronological trace when available.
    if trace_lines:
        purpose = trace_lines
    stem = str(
        raw_evidence.get("question_stem")
        or raw_evidence.get("factual_question_stem")
        or ""
    )
    correct = str(
        raw_evidence.get("correct_answer")
        or raw_evidence.get("factual_correct_answer")
        or ""
    )
    lo_hint = ""
    for row in arts:
        if str(row.get("artifact_type") or "") == "LearningObjective":
            lo_hint = str(
                row.get("objective_text") or row.get("title") or row.get("lo_id") or ""
            )
            break
    source_facts, discarded = select_source_facts(
        arts,
        max_facts=3,
        question_stem=stem,
        correct_answer=correct,
        learning_objective=lo_hint,
    )
    decision_basis = _decision_basis(
        condition=condition,
        decision=decision,
        artifacts=arts,
        decision_reasons=list(raw_evidence.get("decision_reasons") or []),
        criteria=list(raw_evidence.get("criteria") or []),
    )

    findings: Tuple[str, ...] = ()
    grouped_n = 0
    selected_n = 0
    component_n = 0
    raw_n = len(comparisons)
    limitations: List[str] = []
    effect_meta: Dict[str, Any] = {
        "primary_effect_status": None,
        "primary_effect_target": None,
        "primary_effect_details": (),
        "primary_decision_changed": None,
        "selected_criterion_change_count": 0,
        "selected_downstream_change_type": None,
    }

    if condition == "axis":
        groups = group_interventions(comparisons, artifacts=arts)
        grouped_n = len(groups)
        findings, selected_n, effect_meta = select_controlled_findings(groups)
        invalid = sum(
            1
            for c in comparisons
            if str(c.get("evidence_status") or "") in _INVALID_STATUSES
        )
        if not findings:
            if raw_n == 0:
                limitations.append(
                    "No controlled change was successfully tested for this "
                    "question, so causal effects were not demonstrated."
                )
            elif invalid == raw_n:
                limitations.append(
                    "Controlled-change tests were attempted, but none produced a "
                    "valid comparison. No causal effect was therefore established."
                )
            else:
                limitations.append(
                    "No reliable causal conclusion could be drawn from the "
                    "available controlled-change tests."
                )
    elif condition == "trace_provenance":
        # Observational limitation lives inside provenance verbalization when facts exist.
        pass

    q_findings: Tuple[str, ...] = ()
    q_meta: Dict[str, Any] = {}
    prov_rels = list(raw_evidence.get("provenance_relations") or [])
    provenance_lines: Tuple[str, ...] = ()
    if condition == "axis":
        from .question_content_facts import select_component_role_findings

        q_findings, q_meta, _primary_cmp = select_component_role_findings(
            comparisons,
            factual_decision=decision,
            provenance_relations=prov_rels,
            relevant_artifacts=arts,
        )
        component_n = int(q_meta.get("selected_component_finding_count") or (
            1 if q_findings else 0
        ))
        if q_findings:
            findings = q_findings
            selected_n = max(selected_n, component_n)
            limitations = [
                lim
                for lim in limitations
                if "generalize beyond" not in lim.lower()
            ][:1]
        # AXIS also gets observational provenance context.
        if source_facts:
            provenance_lines = verbalize_provenance_links(
                provenance_relations=prov_rels,
                source_facts=source_facts[:3],
            )
            # Drop the trailing causal-limitation sentence for AXIS (roles supply that).
            provenance_lines = tuple(
                p
                for p in provenance_lines
                if "which parts of the MCQ depended" not in p.lower()
            )
    elif condition == "trace_provenance":
        q_findings = ()
        if source_facts:
            provenance_lines = verbalize_provenance_links(
                provenance_relations=prov_rels,
                source_facts=source_facts[:3],
            )
        else:
            limitations.append(
                "The records do not identify concrete teaching facts linked to "
                "this question."
            )

    # AXIS: align source basis; keep multi-fact provenance for T+P/AXIS context.
    primary_info_desc = q_meta.get("primary_information_description")
    primary_info_target = q_meta.get("primary_information_role_target")
    role_known = bool(
        condition == "axis"
        and q_meta.get("information_role_supported")
        and q_findings
    )
    if role_known and primary_info_desc:
        source_out: Tuple[str, ...] = (str(primary_info_desc),)
    elif condition in {"trace_provenance", "axis"}:
        source_out = source_facts[:3]
    else:
        source_out = source_facts[:1] if source_facts else ()

    specific_source = bool(source_out)
    recorded_use = condition in {"trace_provenance", "axis"} and specific_source
    counterfactual_known = bool(q_findings) and condition == "axis"
    decision_effect_known = (
        "decision_effect" in q_meta and q_meta.get("decision_effect") is not None
    )
    identity_known = bool(q_meta.get("question_identity_effect_known"))
    identity_effect = q_meta.get("question_identity_effect")
    acceptance_known = bool(q_meta.get("acceptance_effect_known"))
    acceptance_effect = q_meta.get("acceptance_effect")

    return LayExplanationFacts(
        decision=decision,
        educational_purpose=purpose,
        trace_narrative=trace_lines or purpose,
        relevant_provenance_facts=provenance_lines,
        counterfactual_component_findings=q_findings if role_known else (),
        source_basis=source_out,
        decision_basis=decision_basis if condition == "trace_provenance" else (),
        controlled_change_findings=() if q_findings else findings,
        limitations=tuple(dict.fromkeys(limitations)),
        question_content_findings=q_findings,
        selected_source_fact_count=len(source_out),
        discarded_low_value_snippets=discarded,
        raw_intervention_count=raw_n,
        grouped_intervention_count=grouped_n,
        selected_intervention_finding_count=selected_n,
        selected_component_finding_count=component_n,
        primary_effect_status=effect_meta.get("primary_effect_status"),
        primary_effect_target=effect_meta.get("primary_effect_target"),
        primary_workflow_effect_target=effect_meta.get("primary_effect_target"),
        primary_effect_details=tuple(effect_meta.get("primary_effect_details") or ()),
        primary_decision_changed=effect_meta.get("primary_decision_changed"),
        selected_criterion_change_count=int(
            effect_meta.get("selected_criterion_change_count") or 0
        ),
        selected_downstream_change_type=effect_meta.get(
            "selected_downstream_change_type"
        ),
        primary_question_content_status=q_meta.get("primary_question_content_status"),
        question_content_control_stable=q_meta.get("question_content_control_stable"),
        question_changed_parts=tuple(q_meta.get("question_changed_parts") or ()),
        factual_question_stem=q_meta.get("factual_question_stem"),
        counterfactual_question_stem=q_meta.get("counterfactual_question_stem"),
        intervention_before_question_generation=q_meta.get(
            "intervention_before_question_generation"
        ),
        stem_changed=q_meta.get("stem_changed"),
        correct_answer_changed=q_meta.get("correct_answer_changed"),
        distractors_changed=q_meta.get("distractors_changed"),
        answer_explanation_changed=q_meta.get("answer_explanation_changed"),
        stem_control_stable=q_meta.get("stem_control_stable"),
        correct_answer_control_stable=q_meta.get("correct_answer_control_stable"),
        distractors_control_stable=q_meta.get("distractors_control_stable"),
        answer_explanation_control_stable=q_meta.get(
            "answer_explanation_control_stable"
        ),
        stem_effect=q_meta.get("stem_effect"),
        correct_answer_effect=q_meta.get("correct_answer_effect"),
        distractor_effect=q_meta.get("distractor_effect"),
        answer_explanation_effect=q_meta.get("answer_explanation_effect"),
        primary_question_content_intervention=dict(
            q_meta.get("primary_question_content_intervention") or {}
        )
        or None,
        changed_distractor_count=q_meta.get("changed_distractor_count"),
        information_role=q_meta.get("information_role"),
        information_role_supported=q_meta.get("information_role_supported"),
        semantic_information_absent=q_meta.get("semantic_information_absent"),
        derived_information_may_persist=q_meta.get("derived_information_may_persist"),
        intervention_scope=q_meta.get("intervention_scope"),
        causal_classification=q_meta.get("causal_classification"),
        content_effect=q_meta.get("content_effect"),
        decision_effect=q_meta.get("decision_effect"),
        question_identity_effect=identity_effect if identity_known else None,
        question_identity_effect_known=identity_known,
        acceptance_effect=acceptance_effect if acceptance_known else None,
        acceptance_effect_known=acceptance_known,
        specific_source_known=specific_source,
        recorded_use_known=recorded_use,
        information_role_known=role_known,
        counterfactual_effect_known=counterfactual_known,
        decision_effect_known=decision_effect_known,
        primary_information_role_target=str(primary_info_target)
        if primary_info_target
        else None,
        primary_information_description=str(primary_info_desc)
        if primary_info_desc
        else None,
        source_basis_unavailable=False,
    )
