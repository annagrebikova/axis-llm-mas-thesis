"""Lightweight faithfulness checks for experiment explanations."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple

from .lay_facts import LayExplanationFacts
from .models import ConditionName
from .verbalizer import ExplanationEvidenceView

_CAUSAL_WORDS = re.compile(
    r"\b(caused|decisive|necessary|influenced|contributed|led to|was responsible)\b",
    re.IGNORECASE,
)
_BECAUSE = re.compile(r"\bbecause\b", re.IGNORECASE)
_ENSURED = re.compile(r"\bensured correctness\b", re.IGNORECASE)
_TRACE_FORBIDDEN = re.compile(
    r"\b(counterfactual|replay|provenance|artifact dependency|AXIS|Trace\+Provenance)\b",
    re.IGNORECASE,
)
_CONDITION_NAMES = re.compile(
    r"\b(experimental condition|baseline condition|\bAXIS\b|Trace\+Provenance)\b",
    re.IGNORECASE,
)
_SPECULATIVE = re.compile(
    r"\b(likely|apparently|presumably|probably)\b",
    re.IGNORECASE,
)
_ANSWER_ACCEPTED = re.compile(
    r"\b(the answer was accepted|answer option was accepted)\b",
    re.IGNORECASE,
)
_IRRELEVANT = re.compile(
    r"\b(irrelevant|universally irrelevant|does not matter|not matter at all)\b",
    re.IGNORECASE,
)
_PARTICIPANT_JARGON = re.compile(
    r"\b(EvidenceChunk|KnowledgeAtom|MCQBlueprint|CrossCoVeReport|"
    r"OutcomeAssessment|SupervisorDecision|ArtifactRef|NOOP|Cross-CoVe|"
    r"checkpoint|criterion vector|replay configuration|downstream_effect|"
    r"criterion_effect|decision_effect|provenance relation|target_atom_count)\b",
    re.IGNORECASE,
)
_ARTIFACT_IDISH = re.compile(r"\b(?:q_|chunk_|atom_|bp_)[0-9a-fA-F]+\b|:v\d+\b")
_EMAILISH = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_INFO_STOPWORDS = {
    "learning",
    "information",
    "stating",
    "teaching",
    "without",
    "labeled",
    "outputs",
    "structure",
}


def tested_information_is_named(description: str, explanation: str) -> bool:
    """True when the explanation names the exact tested-information description.

    Short labels/acronyms such as PCA, GMM, SVM count via exact word-boundary
    match. Longer descriptions still use distinctive-content tokens. Generic
    phrases such as "the supporting material" are never sufficient by themselves.
    """
    desc = str(description or "").strip()
    text = str(explanation or "")
    if not desc:
        return True
    desc_l = desc.lower()
    text_l = text.lower()
    if desc_l in text_l:
        return True
    compact = re.sub(r"[^a-z0-9]+", "", desc_l)
    if 2 <= len(compact) <= 6:
        if re.search(rf"\b{re.escape(desc_l)}\b", text_l):
            return True
        if compact != desc_l and re.search(rf"\b{re.escape(compact)}\b", text_l):
            return True
    tokens = [
        w
        for w in re.findall(r"[a-z]{5,}", desc_l)
        if w not in _INFO_STOPWORDS
    ][:5]
    tokens2 = [w for w in re.findall(r"[a-z]{5,}", desc_l)][:6]
    distinctive = tokens or tokens2
    return bool(distinctive) and any(t in text_l for t in distinctive[:4])

# Trace: decision-attribution from temporal checks (not mere "based on teaching material").
_TRACE_DECISION_ATTRIBUTION = re.compile(
    r"("
    r"decision\s+to\s+accept\b.{0,80}\bbased\s+on\b|"
    r"\b(?:the\s+)?decision\s+was\s+based\s+on\b|"
    r"\bbased\s+on\s+(?:thorough\s+)?checks?\b|"
    r"\bchecks?\s+(?:led\s+to|resulted\s+in)\s+accept|"
    r"\bleading\s+to\s+(?:the\s+)?(?:decision|accept)|"
    r"\b(?:therefore|thus)\s+(?:the\s+)?(?:question\s+was\s+)?accepted\b|"
    r"\bchecks?\s+confirmed\s+(?:the\s+)?question'?s?\s+validity\b|"
    r"\bchecks?\s+justified\s+accept|"
    r"\bconfirmed\s+(?:its\s+)?(?:alignment|validity).{0,40}\baccept"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_TP_CAUSAL_ATTRIBUTION = re.compile(
    r"("
    r"\bleading\s+to\s+(?:the\s+)?(?:decision\s+to\s+)?accept|"
    r"\btherefore\s+accepted\b|"
    r"\bcaused\s+accept|"
    r"\bwas\s+necessary\s+for\s+accept|"
    r"\bnecessary\s+for\s+acceptance\b"
    r")",
    re.IGNORECASE,
)

# Positive causal-role language forbidden when provenance-only (no AXIS roles).
TP_CAUSAL_ROLE_PATTERNS = (
    r"\binfluenc(?:e|ed|es|ing)\b",
    r"\bshap(?:e|ed|es|ing)\b",
    r"\bdetermin(?:e|ed|es|ing)\b",
    r"\banchor(?:ed|s|ing)?\b",
    r"\bcaus(?:e|ed|es|ing)\b",
    r"\baffect(?:ed|s|ing)?\b",
    r"\bsteer(?:ed|s|ing)?\b",
    r"\bdriv(?:e|en|ing)\b",
    r"\bdrove\b",
    r"\bled to\b",
    r"\bleads? to\b",
    r"\bresulted in\b",
    r"\bresponsible for\b",
    r"\bhelped determine\b",
    r"\bhelped shape\b",
    r"\bhelped keep\b",
    r"\bcontributed\b",
)
_TP_CAUSAL_ROLE_RES = tuple(
    re.compile(p, re.IGNORECASE) for p in TP_CAUSAL_ROLE_PATTERNS
)

TP_NONCAUSAL_LIMITATION_MARKERS = (
    "does not show",
    "do not show",
    "does not establish",
    "do not establish",
    "cannot establish",
    "cannot determine",
    "cannot infer",
    "cannot tell",
    "does not demonstrate",
    "do not demonstrate",
    "does not indicate whether",
    "do not indicate whether",
    "not by itself",
    "not by themselves",
    "do not identify",
    "does not identify",
    "cannot identify",
    "whether it influenced",
    "whether this information influenced",
    "whether that information influenced",
)

# Trace+Provenance must not speculate about counterfactual question outcomes.
_TP_COUNTERFACTUAL_SPECULATION = re.compile(
    r"("
    r"\bensured\s+that\s+the\s+question\b|"
    r"\bensured\s+the\s+question\b|"
    r"\bmade\s+the\s+question\b|"
    r"\ballowed\s+the\s+question\s+to\b|"
    r"\benabled\s+the\s+question\s+to\b|"
    r"\bwithout\s+this\s+information,?\s+the\s+question\s+may\b|"
    r"\bwithout\s+this\s+information,?\s+the\s+question\s+might\b|"
    r"\bwithout\s+this\s+information,?\s+the\s+question\s+would\b|"
    r"\bif\s+this\s+information\s+were\s+absent\b|"
    r"\bmay\s+not\s+have\s+effectively\s+targeted\b|"
    r"\bmay\s+have\s+lacked\b"
    r")",
    re.IGNORECASE,
)

_AXIS_INCONCLUSIVE_LIMITATION = re.compile(
    r"("
    r"cannot\s+reliably\s+determine|"
    r"could\s+not\s+determine|"
    r"unchanged\s+rerun\s+did\s+not\s+reproduce|"
    r"test\s+(?:was|is)\s+inconclusive|"
    r"cannot\s+reliably\s+show"
    r")",
    re.IGNORECASE,
)

_AXIS_UNSUPPORTED_CONTENT_CAUSAL = re.compile(
    r"("
    r"the\s+information\s+ensured\b|"
    r"supporting\s+information\s+ensured\b|"
    r"without\s+this\s+(?:information|content),?\s+the\s+question\s+(?:may|might|would)\b|"
    r"this\s+information\s+made\s+the\s+question\b|"
    r"without\s+this\s+content,?\s+the\s+question\s+may\s+have\s+lacked\b"
    r")",
    re.IGNORECASE,
)

_TRACE_INFORMATION_ROLE = re.compile(
    r"("
    r"\b(?:supporting\s+)?(?:information|material|content|source)\b.{0,40}\b"
    r"(?:shaped|helps?\s+shape|influenced|determined|anchored|helps?\s+anchor|"
    r"steered|led\s+to|made\s+the\s+question|was\s+responsible|was\s+important\s+for|"
    r"was\s+necessary\s+for)\b|"
    r"\b(?:shaped|helps?\s+shape|influenced|determined|anchored|helps?\s+anchor|"
    r"steered)\b.{0,40}\b(?:question|answer|distractor|alternative|focus|wording)\b|"
    r"\bcorrect\s+answer\s+(?:is\s+)?anchored\b|"
    r"\bhelped\s+shape\s+the\s+question"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_TRACE_ROLE_DENIAL = re.compile(
    r"("
    r"does\s+not\s+(?:identify|show|establish)|"
    r"cannot\s+(?:identify|show|determine|establish)|"
    r"do\s+not\s+show\s+whether|"
    r"execution\s+record\s+(?:alone\s+)?does\s+not"
    r")",
    re.IGNORECASE,
)

_AXIS_SCOPE_OVERCLAIM = re.compile(
    r"("
    r"without\s+the\s+teaching\s+material|"
    r"without\s+the\s+source\s+document|"
    r"without\s+any\s+supporting\s+information|"
    r"without\s+the\s+entire\s+(?:pdf|document|material)|"
    r"without\s+the\s+supporting\s+material"
    r")",
    re.IGNORECASE,
)

_AXIS_CHANGED_UNCHANGED_CONTRADICTION = re.compile(
    r"("
    r"(?:question\s+changed|changed\s+from).{0,220}"
    r"(?:question\s+remained\s+the\s+same|remained\s+the\s+same\s+when)|"
    r"(?:question\s+remained\s+the\s+same|remained\s+the\s+same\s+when).{0,220}"
    r"(?:question\s+changed|changed\s+from)"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_AXIS_MECHANICS = re.compile(
    r"("
    r"\bREMOVE\s*\(|"
    r"\bREPLACE\s*\(|"
    r"\bWHATIF\s*\(|"
    r"\bcounterfactual\s+branch\b|"
    r"\bNOOP\b|"
    r"\bcheckpoint\b|"
    r"\breplay\s+branch\b|"
    r"\bfork\b|"
    r"\bcontrolled\s+test\s+removed\b|"
    r"\bregenerated\s+after\s+removing\b|"
    r"\bwe\s+removed\s+the\s+information\b|"
    r"\bremoved\s+the\s+information\b|"
    r"\bdeleted\s+the\s+information\b"
    r")",
    re.IGNORECASE,
)

_AXIS_SAME_QUESTION_CLAIM = re.compile(
    r"("
    r"\bthe\s+question\s+stayed\s+the\s+same\b|"
    r"\bthe\s+mcq\s+remained\s+(?:the\s+)?(?:same|unchanged)\b|"
    r"\bthe\s+same\s+question\s+was\s+generated\b|"
    r"\bthe\s+question\s+remained\s+unchanged\b|"
    r"\bthe\s+mcq\s+remained\s+unchanged\b"
    r")",
    re.IGNORECASE,
)

_AXIS_STEM_SCOPED_SAMENESS = re.compile(
    r"("
    r"continued\s+to\s+ask\s+the\s+same\s+thing|"
    r"wording\s+of\s+the\s+question\s+itself|"
    r"question\s+continues\s+to\s+assess\s+the\s+same\s+idea|"
    r"main\s+question\s+remains\s+focused|"
    r"helps\s+shape\s+the\s+incorrect\s+alternatives"
    r")",
    re.IGNORECASE,
)

_AXIS_ONE_DISTRACTOR = re.compile(
    r"\b(?:one|1)\s+incorrect\s+(?:answer\s+)?(?:option|alternative)s?\s+changed\b",
    re.IGNORECASE,
)
_AXIS_TWO_DISTRACTOR = re.compile(
    r"\b(?:two|2)\s+incorrect\s+(?:answer\s+)?(?:option|alternative)s?\s+changed\b",
    re.IGNORECASE,
)
_AXIS_THREE_DISTRACTOR = re.compile(
    r"\b(?:three|3)\s+incorrect\s+(?:answer\s+)?(?:option|alternative)s?\s+changed\b",
    re.IGNORECASE,
)
_AXIS_ALL_DISTRACTORS = re.compile(
    r"\b(?:all|every)\s+(?:incorrect\s+)?(?:answer\s+)?(?:options?|alternatives?|distractors?)\s+changed\b",
    re.IGNORECASE,
)

_SEPARATE_TEST_QUALIFIER = re.compile(
    r"\b(in\s+one\s+test|in\s+a\s+separate\s+test|in\s+another\s+test|"
    r"in\s+the\s+strongest\s+test)\b",
    re.IGNORECASE,
)


def _lay(view: ExplanationEvidenceView) -> Optional[LayExplanationFacts]:
    return getattr(view, "lay_facts", None)


def provenance_only_role_boundary(facts: Optional[LayExplanationFacts]) -> bool:
    """True when recorded use is known but causal information roles are not."""
    if facts is None:
        return True
    return bool(
        facts.recorded_use_known
        and not facts.information_role_known
        and not facts.counterfactual_effect_known
    )


def _split_epistemic_clauses(text: str) -> List[str]:
    """Split explanation into clauses for capability-scoped causality checks."""
    clauses: List[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    for sent in sentences:
        if not sent.strip():
            continue
        parts = re.split(
            r"\s*\b(?:but|however|although|yet|while)\b\s*|;",
            sent,
            flags=re.IGNORECASE,
        )
        for part in parts:
            cleaned = part.strip(" ,;")
            if cleaned:
                clauses.append(cleaned)
    return clauses


def _tp_clause_has_unsupported_causality(clause: str) -> bool:
    """Positive causal-role claim without an explicit epistemic limitation."""
    lower = str(clause or "").lower()
    if not lower:
        return False
    has_causal_role_term = any(rx.search(lower) for rx in _TP_CAUSAL_ROLE_RES)
    if not has_causal_role_term:
        return False
    explicitly_limited = any(
        marker in lower for marker in TP_NONCAUSAL_LIMITATION_MARKERS
    )
    return not explicitly_limited


def tp_has_unsupported_causal_role_claim(explanation: str) -> bool:
    """True if any clause asserts an unsupported causal information role."""
    return any(
        _tp_clause_has_unsupported_causality(c)
        for c in _split_epistemic_clauses(explanation)
    )


def _has_valid_intervention_block(view: ExplanationEvidenceView) -> bool:
    facts = _lay(view)
    if facts is not None:
        return bool(facts.question_content_findings or facts.controlled_change_findings)
    text = " ".join(view.tested_interventions).lower()
    if not text:
        return False
    markers = (
        "tested",
        "removing",
        "final decision",
        "valid test",
        "workflow",
        "did not show",
        "remained unchanged",
        "remained accept",
        "different question",
        "same question",
    )
    return any(m in text for m in markers)


def _explanation_mentions_intervention(explanation: str) -> bool:
    # Ignore required section headings so "this information" in the heading
    # alone does not count as covering a tested intervention.
    body_lines = []
    for ln in (explanation or "").splitlines():
        s = ln.strip()
        if not s or s.endswith("?"):
            continue
        body_lines.append(s.lower())
    low = " ".join(body_lines)
    markers = (
        "tested",
        "removing",
        "without",
        "remained unchanged",
        "remained accept",
        "final decision",
        "workflow",
        "did not show",
        "valid test",
        "assessment",
        "supporting knowledge",
        "helps shape",
        "helps determine",
        "helps anchor",
        "helped keep",
        "helped determine",
        "helped shape",
        "incorrect alternatives",
        "conceptual focus",
        "focused on",
        "not decisive",
        "robust to this",
        "control replay",
        "tested change",
        "altered the final decision",
        "influenced which question",
        "acceptable mcq",
    )
    return any(m in low for m in markers)


def _duplicate_intervention_statements(explanation: str) -> bool:
    lines = [
        re.sub(r"\s+", " ", ln.strip().lower())
        for ln in (explanation or "").splitlines()
        if ln.strip() and not ln.strip().endswith("?")
    ]
    interventionish = [
        ln
        for ln in lines
        if "tested" in ln and ("remov" in ln or "without" in ln or "change" in ln)
    ]
    counts = Counter(interventionish)
    return any(v >= 2 for v in counts.values())


def _axis_effect_coverage_ok(
    *,
    explanation: str,
    facts: LayExplanationFacts,
) -> List[str]:
    """Effect-specific coverage of the strongest selected finding."""
    warnings: List[str] = []
    text = (explanation or "").lower()
    status = facts.primary_effect_status
    if not status:
        return warnings

    decision_unchanged = (
        (
            "remained" in text
            and ("decision" in text or "accept" in text)
            and ("unchanged" in text or "same" in text or "accept" in text)
        )
        or ("final decision remained" in text)
        or ("remained accept" in text)
        or ("decision remained" in text)
        or ("final decision did not" in text)
        or ("decision did not" in text and "change" in text)
        or ("still accepted" in text)
        or ("was still accepted" in text)
        or ("still accept" in text)
    )

    if status == "decision_effect":
        if "final decision changed" not in text and "decision changed" not in text:
            if "no longer" not in text and "changed from" not in text:
                warnings.append(
                    "AXIS omits decision_effect: final decision change not stated"
                )

    elif status == "criterion_effect":
        property_markers = (
            "supporting knowledge",
            "source evidence linked",
            "answer check",
            "blocking quality",
            "source reference",
            "question-assessment",
            "question plan",
            "tracked assessment",
            "amount of supporting",
            "assessment property",
            "assessment,",  # "affected the assessment,"
            "affected the assessment",
        )
        detail_hit = False
        for d in facts.primary_effect_details:
            # Prefer distinctive multi-word fragments from structured detail.
            low_d = d.lower()
            for frag in (
                "supporting knowledge",
                "source evidence",
                "answer check",
                "blocking quality",
                "from two to",
                "from one to",
                "from three to",
                "tracked assessment",
                "lay_label",  # noop
            ):
                if frag in low_d and frag in text:
                    detail_hit = True
                    break
            # Numeric change pattern from structured sentence.
            if re.search(r"changed from \w+ to \w+", low_d):
                m = re.search(r"changed from (\w+) to (\w+)", low_d)
                if m and m.group(0) in text:
                    detail_hit = True
            if detail_hit:
                break
            # Fall back: require the lay property phrase stem present in both.
            for stem in (
                "supporting knowledge linked to the question",
                "source evidence linked to the question plan",
                "additional answer check",
                "blocking quality issues",
            ):
                if stem in low_d and stem in text:
                    detail_hit = True
                    break
        if not detail_hit and not any(m in text for m in property_markers):
            warnings.append(
                "AXIS omits criterion_effect: assessment-property change not stated"
            )
        if not decision_unchanged and "decision changed" not in text:
            if "remained" not in text and "unchanged" not in text and "same" not in text:
                warnings.append(
                    "AXIS omits criterion_effect: unchanged final decision not stated"
                )
        only_decision = (
            decision_unchanged
            and not detail_hit
            and not any(m in text for m in property_markers)
            and ("tested" in text or "removal" in text or "removing" in text)
        )
        if only_decision:
            warnings.append(
                "AXIS reduces criterion_effect to unchanged-decision only"
            )

    elif status == "downstream_effect":
        down_markers = (
            "workflow",
            "processing",
            "repair",
            "sequence of",
            "later",
            "recorded as being used",
            "information produced later",
        )
        dtype = facts.selected_downstream_change_type
        if dtype == "path_repair" and "repair" not in text:
            warnings.append(
                "AXIS omits downstream path-repair detail present in selected facts"
            )
        elif dtype == "recorded_use" and (
            "repair" in text or "sequence of processing steps" in text and "altered the later sequence" in text
        ):
            # If selected type is recorded_use, inventing repair/path is wrong;
            # only flag missing recorded-use wording.
            if "recorded as being used" not in text and "information recorded" not in text:
                warnings.append(
                    "AXIS omits recorded-use downstream detail"
                )
        elif not any(m in text for m in down_markers):
            warnings.append(
                "AXIS omits downstream_effect: later-workflow change not stated"
            )
        if not decision_unchanged and "decision changed" not in text:
            if "remained" not in text and "unchanged" not in text:
                warnings.append(
                    "AXIS omits downstream_effect: unchanged final decision not stated"
                )

    elif status == "no_demonstrated_effect":
        if "no tracked" not in text and "no change" not in text and "unchanged" not in text:
            if "remained" not in text:
                warnings.append(
                    "AXIS omits no_demonstrated_effect wording"
                )

    return warnings


def validate_explanation_faithfulness(
    *,
    condition: ConditionName,
    explanation: str,
    evidence_view: ExplanationEvidenceView,
) -> Tuple[bool, Tuple[str, ...]]:
    """Return (ok, warnings). Does not call an LLM."""
    text = explanation or ""
    warnings: List[str] = []
    facts = _lay(evidence_view)

    if _SPECULATIVE.search(text):
        warnings.append("speculative wording (likely/apparently/presumably/probably)")
    if _ANSWER_ACCEPTED.search(text):
        warnings.append("treats ACCEPT as answer-option acceptance")
    if _CONDITION_NAMES.search(text):
        warnings.append("mentions experimental condition labels")
    if _PARTICIPANT_JARGON.search(text):
        warnings.append("participant-facing jargon (internal type/method names)")
    if _ARTIFACT_IDISH.search(text):
        warnings.append("exposes internal artifact identifiers")
    if _ENSURED.search(text):
        warnings.append('uses unsupported phrase "ensured correctness"')
    if _duplicate_intervention_statements(text):
        warnings.append("duplicate intervention statements")

    # Participant format is free prose — no mandatory headings.

    # Metadata-only source shown while better content was selected.
    if facts is not None and facts.discarded_low_value_snippets and facts.source_basis:
        for bad in facts.discarded_low_value_snippets:
            if _EMAILISH.search(bad) and any(
                tok in text for tok in bad.split() if "@" in tok
            ):
                warnings.append(
                    "metadata-only source snippet shown despite better content"
                )
                break

    if condition == "trace":
        # Temporal "after" / chronological narration is allowed; only flag
        # strong causal attribution verbs when not denied.
        if _CAUSAL_WORDS.search(text):
            low = text.lower()
            if not any(
                d in low
                for d in ("does not", "do not", "cannot", "not which", "not identify")
            ):
                warnings.append("unsupported causal language in Trace explanation")
        elif _BECAUSE.search(text) and not re.search(
            r"\bnot\b.*\bbecause\b", text, re.I
        ):
            warnings.append("unsupported causal language in Trace explanation")
        if _TRACE_DECISION_ATTRIBUTION.search(text):
            warnings.append(
                "Trace attributes ACCEPT to checks without causal evidence"
            )
        if _TRACE_FORBIDDEN.search(text):
            warnings.append("Trace explanation mentions unavailable analyses")
        if evidence_view.tested_interventions or (
            facts is not None
            and facts.controlled_change_findings
            and not facts.question_content_findings
        ):
            warnings.append("Trace evidence view unexpectedly includes interventions")
        if re.search(r'Teaching material fact: "', text):
            warnings.append("Trace explanation leaks provenance-selected source facts")
        if _TRACE_INFORMATION_ROLE.search(text) and not _TRACE_ROLE_DENIAL.search(text):
            warnings.append(
                "Trace makes unsupported information-role / causal shaping claim"
            )
        # When user asks about supporting information and Trace lacks sources,
        # require a short natural limitation.
        if facts is not None and facts.source_basis_unavailable:
            low = text.lower()
            if not any(
                p in low
                for p in (
                    "does not identify",
                    "cannot determine",
                    "cannot show",
                    "does not show which",
                    "execution record",
                    "trace shows",
                    "not which specific teaching",
                    "responsible for the exact",
                    "explains why this particular question was selected",
                    "why this particular question was selected",
                )
            ):
                warnings.append(
                    "Trace omits natural limitation that specific information is unknown"
                )
            if "would have been different" in low or (
                "without a particular piece of information" in low
                and "selected" not in low
            ):
                warnings.append(
                    "Trace answers an unasked counterfactual about missing information"
                )

    elif condition == "trace_provenance":
        # Capability-based boundary: recorded use without causal role evidence.
        role_boundary = provenance_only_role_boundary(facts) or (
            facts is None
            or (
                not bool(getattr(facts, "information_role_known", False))
                and not bool(getattr(facts, "counterfactual_effect_known", False))
            )
        )
        if role_boundary and tp_has_unsupported_causal_role_claim(text):
            warnings.append(
                "unsupported causal information-role claim under provenance-only evidence"
            )
        if _TP_CAUSAL_ATTRIBUTION.search(text):
            warnings.append(
                "Trace+Provenance implies checks/inputs caused ACCEPT"
            )
        if re.search(r"\bcounterfactual\b|\breplay\b", text, re.IGNORECASE):
            warnings.append("Trace+Provenance mentions counterfactual/replay")
        if _TP_COUNTERFACTUAL_SPECULATION.search(text):
            # Allow only when the same sentence denies the claim.
            low = text.lower()
            if not (
                "do not show whether" in low
                or "does not show whether" in low
                or "cannot show whether" in low
                or "records alone do not" in low
                or "do not establish whether" in low
                or "cannot establish whether" in low
                or "not establish whether it was decisive" in low
                or any(m in low for m in TP_NONCAUSAL_LIMITATION_MARKERS)
            ):
                warnings.append(
                    "Trace+Provenance asserts unsupported counterfactual question effect"
                )
        if facts is not None and facts.specific_source_known and facts.source_basis:
            # Require at least one contentful token from the selected source fact.
            src = " ".join(facts.source_basis).lower()
            tokens = [w for w in re.findall(r"[a-z]{5,}", src) if w not in {
                "learning", "information", "stating", "teaching", "derived", "source"
            }][:4]
            low = text.lower()
            if tokens and not any(t in low for t in tokens):
                if (
                    "teaching information" not in low
                    and "teaching material" not in low
                    and "used" not in low
                    and "checked against" not in low
                ):
                    warnings.append(
                        "Trace+Provenance omits available concrete source information"
                    )

    else:  # axis
        has_decision_effect = (
            facts is not None and facts.primary_effect_status == "decision_effect"
        ) or any(
            "final decision changed" in ln.lower()
            for ln in (
                facts.controlled_change_findings
                if facts is not None
                else evidence_view.tested_interventions
            )
        )
        if re.search(r"\bdecisive\b", text, re.IGNORECASE) and not has_decision_effect:
            if not re.search(r"\bnot\b.{0,40}\bdecisive\b", text, re.I):
                warnings.append("AXIS uses 'decisive' without decision_effect evidence")
        if _IRRELEVANT.search(text):
            warnings.append("AXIS overstates no_demonstrated_effect as irrelevance")
        if _has_valid_intervention_block(evidence_view) and not _explanation_mentions_intervention(
            text
        ):
            warnings.append(
                "AXIS explanation omits valid tested-intervention evidence"
            )
        if facts is not None and facts.primary_effect_status and not facts.question_content_findings:
            warnings.extend(_axis_effect_coverage_ok(explanation=text, facts=facts))
        if facts is not None and facts.question_content_findings:
            warnings.extend(_axis_question_content_coverage_ok(explanation=text, facts=facts))
        if _AXIS_MECHANICS.search(text):
            warnings.append(
                "AXIS participant text exposes intervention/replay mechanics"
            )
        if facts is not None:
            warnings.extend(_axis_distractor_cardinality_ok(explanation=text, facts=facts))
            warnings.extend(_axis_same_mcq_wording_ok(explanation=text, facts=facts))
            if facts.information_role_known and facts.primary_information_description:
                desc = facts.primary_information_description
                named = tested_information_is_named(desc, text)
                if not named:
                    warnings.append(
                        "AXIS omits primary tested information description"
                    )
                if re.search(
                    r"\b(?:the\s+)?(?:supporting|teaching)\s+material\b",
                    text,
                    re.I,
                ) and not named:
                    warnings.append(
                        "AXIS refers to supporting material without the tested information"
                    )
        if _BECAUSE.search(text) and not has_decision_effect:
            low = text.lower()
            # Allow content-causal "because" when supported role findings exist.
            content_because_ok = bool(
                facts is not None
                and facts.information_role_known
                and (
                    "focus" in low
                    or "answer" in low
                    or "incorrect" in low
                    or "option" in low
                    or "phrased" in low
                    or "wording" in low
                )
            )
            if "not because" not in low and not content_because_ok:
                warnings.append('unsupported "because" without decision-effect evidence')
        if facts is not None and facts.question_identity_effect is True:
            if re.search(
                r"not\s+decisive\s+for\s+whether\s+(?:this\s+)?(?:the\s+)?question\s+was\s+selected",
                text,
                re.I,
            ):
                warnings.append(
                    "AXIS misstates identity change as non-decisiveness for selection"
                )
        if _AXIS_SCOPE_OVERCLAIM.search(text):
            warnings.append(
                "AXIS overclaims intervention scope as entire teaching material"
            )
        if _AXIS_CHANGED_UNCHANGED_CONTRADICTION.search(text) and not _SEPARATE_TEST_QUALIFIER.search(
            text
        ):
            warnings.append(
                "AXIS combines changed/unchanged question claims without separate-test labels"
            )

    return (len(warnings) == 0, tuple(warnings))


def _axis_distractor_cardinality_ok(
    *,
    explanation: str,
    facts: LayExplanationFacts,
) -> List[str]:
    warnings: List[str] = []
    n = facts.changed_distractor_count
    if n is None and facts.distractors_changed:
        # Fall back: unknown exact count — still reject "all" without support.
        n = None
    if _AXIS_ALL_DISTRACTORS.search(explanation or ""):
        # Only allow if we somehow know all changed; we do not without total size.
        warnings.append("AXIS claims all distractors changed without exact evidence")
    if n == 2 and _AXIS_ONE_DISTRACTOR.search(explanation or ""):
        warnings.append("AXIS incorrect distractor cardinality: says one, evidence is 2")
    if n == 1 and _AXIS_TWO_DISTRACTOR.search(explanation or ""):
        warnings.append("AXIS incorrect distractor cardinality: says two, evidence is 1")
    if n == 2 and _AXIS_THREE_DISTRACTOR.search(explanation or ""):
        warnings.append("AXIS incorrect distractor cardinality: says three, evidence is 2")
    if n == 1 and _AXIS_THREE_DISTRACTOR.search(explanation or ""):
        warnings.append("AXIS incorrect distractor cardinality: says three, evidence is 1")
    if n == 3 and _AXIS_ONE_DISTRACTOR.search(explanation or ""):
        warnings.append("AXIS incorrect distractor cardinality: says one, evidence is 3")
    return warnings


def _axis_same_mcq_wording_ok(
    *,
    explanation: str,
    facts: LayExplanationFacts,
) -> List[str]:
    warnings: List[str] = []
    if not (
        facts.stem_changed is False
        and facts.correct_answer_changed is False
        and facts.distractors_changed is True
    ):
        return warnings
    if _AXIS_SAME_QUESTION_CLAIM.search(explanation or ""):
        if not _AXIS_STEM_SCOPED_SAMENESS.search(explanation or ""):
            warnings.append(
                "AXIS claims same question/MCQ despite distractor changes "
                "(same stem ≠ same MCQ)"
            )
    return warnings


def _axis_question_content_coverage_ok(
    *,
    explanation: str,
    facts: LayExplanationFacts,
) -> List[str]:
    warnings: List[str] = []
    text = (explanation or "").lower()
    status = facts.primary_question_content_status
    before = facts.intervention_before_question_generation

    if before is False and (
        "would have been generated" in text
        or "same question was regenerated" in text
        or "generated a different question" in text
    ):
        warnings.append(
            "AXIS claims generation effect for post-generation intervention"
        )

    if status == "question_content_changed":
        if facts.stem_changed:
            if "same question" in text and "different" not in text and "changed" not in text:
                warnings.append("AXIS omits stem change: claims same question")
            if not any(
                m in text
                for m in (
                    "different question",
                    "instead of asking",
                    "changed from",
                    "regenerated question",
                    "question changed",
                    "conceptual focus",
                    "helps determine",
                    "focus of the question",
                    "steers the question",
                )
            ):
                # Require at least one content-change cue.
                if facts.counterfactual_question_stem:
                    stem_l = facts.counterfactual_question_stem.lower()
                    # Accept if substantial stem fragment appears.
                    tokens = [w for w in stem_l.split() if len(w) > 4][:3]
                    if tokens and not any(t in text for t in tokens):
                        warnings.append(
                            "AXIS omits question-content effect: stem change not stated"
                        )
                else:
                    warnings.append(
                        "AXIS omits question-content effect: stem change not stated"
                    )
        elif facts.correct_answer_changed:
            if "correct answer" not in text:
                warnings.append(
                    "AXIS omits question-content effect: correct-answer change not stated"
                )
            if "completely" in text and "question itself changed" in text:
                warnings.append(
                    "AXIS overstates distractor/answer change as whole-question change"
                )
        elif facts.distractors_changed:
            if (
                "incorrect" not in text
                and "distractor" not in text
                and "option" not in text
                and "alternative" not in text
                and "shape" not in text
            ):
                warnings.append(
                    "AXIS omits question-content effect: distractor change not stated"
                )
            if "question itself changed completely" in text or (
                "different question" in text
                and "option" not in text
                and "alternative" not in text
                and "shape" not in text
            ):
                warnings.append(
                    "AXIS overstates distractor-only change as whole-question change"
                )
        if (
            not facts.stem_changed
            and (facts.correct_answer_changed or facts.distractors_changed)
            and re.search(r"\bthe question remained the same\b", text)
            and "correct" not in text
            and "incorrect" not in text
            and "option" not in text
            and "alternative" not in text
            and not _AXIS_STEM_SCOPED_SAMENESS.search(explanation or "")
        ):
            warnings.append(
                "AXIS says question remained the same despite answer-option changes"
            )
    elif status == "question_content_partially_supported":
        if facts.stem_changed:
            if not any(
                m in text
                for m in (
                    "changed from",
                    "instead of asking",
                    "different question",
                    "question changed",
                    "wording",
                )
            ):
                warnings.append(
                    "AXIS omits supported stem change in partial question-content finding"
                )
        if facts.distractors_changed is False and det_claims_distractors(text):
            warnings.append(
                "AXIS claims distractor change without distractor control stability"
            )
    elif status == "question_content_unchanged":
        if "different question" in text and "same question" not in text:
            warnings.append(
                "AXIS claims question changed despite unchanged content finding"
            )
    elif status == "question_not_reproduced":
        if "corresponding question" not in text and "did not generate" not in text:
            if "same question" in text:
                warnings.append(
                    "AXIS claims same question when CF question was not reproduced"
                )
    elif status == "question_content_inconclusive":
        if not _AXIS_INCONCLUSIVE_LIMITATION.search(explanation or ""):
            warnings.append(
                "AXIS omits inconclusive limitation for question-content control failure"
            )
        if _AXIS_UNSUPPORTED_CONTENT_CAUSAL.search(explanation or ""):
            # Allow only when a supported stem/correct/distractor claim is present.
            if not (
                facts.stem_changed
                or facts.correct_answer_changed
                or facts.distractors_changed
            ):
                warnings.append(
                    "AXIS asserts unsupported causal question-content claim while inconclusive"
                )

    # Must not replace content effect with decision-only.
    if status in {
        "question_content_changed",
        "question_content_partially_supported",
    } and facts.question_content_findings:
        if (
            ("remained" in text or "unchanged" in text or "still accept" in text)
            and not any(
                m in text
                for m in (
                    "different",
                    "changed",
                    "instead of",
                    "incorrect",
                    "correct answer",
                    "option",
                    "wording",
                )
            )
        ):
            warnings.append(
                "AXIS reduces question-content effect to unchanged-decision only"
            )
    return warnings


def det_claims_distractors(text: str) -> bool:
    return bool(
        re.search(
            r"incorrect\s+(?:answer\s+)?options?\s+changed|distractors?\s+changed|"
            r"changed\s+the\s+incorrect\s+options",
            text,
            re.I,
        )
    )
