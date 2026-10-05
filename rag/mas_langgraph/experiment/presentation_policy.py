"""Deterministic participant presentation-policy lint (not epistemic faithfulness).

Flags workflow chronology, redundant provenance boilerplate, and length/sentence
budgets derived from evidence *capabilities*. Does not judge factual correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple

# Soft preferred maxima (not the hard 200-word ceiling).
LENGTH_BUDGETS = {
    "topic_limitation": {
        "preferred_words": 45,
        "violation_words": 60,
        "preferred_sentences": 1,
        "max_sentences": 2,
    },
    "provenance_only": {
        "preferred_words": 65,
        "violation_words": 85,
        "preferred_sentences": 1,
        "max_sentences": 2,
    },
    "role_evidence": {
        "preferred_words": 110,
        "violation_words": 140,
        "preferred_sentences": 3,
        "max_sentences": 4,
    },
}

TRACE_LIMITATION_SENTENCE = (
    "The record shows that the question was generated from the teaching "
    "material, but it does not identify which specific teaching fact determined "
    "the wording, correct answer, or distractors."
)

# Upstream preparation stages vs compact generation/check/retain summaries.
_UPSTREAM_CATEGORIES = frozenset(
    {"organization", "objectives", "retrieval", "extraction", "planning"}
)
_DOWNSTREAM_CATEGORIES = frozenset({"generation", "checking", "acceptance"})

_CHRONOLOGY_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "organization",
        (
            r"organized the teaching material",
            r"organized the material",
            r"organized into sections",
            r"material was organized",
            r"the workflow organized",
            r"teaching material into sections",
        ),
    ),
    (
        "objectives",
        (
            r"created learning objectives",
            r"learning objectives were created",
            r"objectives were created",
            r"created objectives",
        ),
    ),
    (
        "retrieval",
        (
            r"retrieved candidate supporting",
            r"retrieved supporting material",
            r"retrieved candidate",
            r"supporting facts were retrieved",
            r"retrieved and extracted",
            r"retrieved candidate supporting material",
        ),
    ),
    (
        "extraction",
        (
            r"extracted concise factual",
            r"extracted factual statements",
            r"factual statements were extracted",
            r"extracted from the material",
            r"supporting facts were retrieved and extracted",
        ),
    ),
    (
        "planning",
        (
            r"created a plan for the question",
            r"plan for the question was created",
            r"created a plan for",
            r"a plan for the question",
        ),
    ),
    (
        "generation",
        (
            r"generated the mcq",
            r"generated the question",
            r"generated answer options",
            r"mcq and (?:its )?answer options were then generated",
            r"mcq and options were generated",
            r"generated the mcq and answer options",
            r"the question was generated",
            r"created the question",
        ),
    ),
    (
        "checking",
        (
            r"checked against the teaching material",
            r"checked for coverage",
            r"checked coverage of the learning",
            r"checked it against",
            r"checked the question",
            r"checked against teaching material",
        ),
    ),
    (
        "acceptance",
        (
            r"retained for the final quiz",
            r"evaluated whether the question satisfied",
            r"evaluated whether it satisfied",
            r"evaluated the required criteria",
            r"accepted after the recorded checks",
            r"accepted after those checks",
            r"the question was accepted",
            r"retained after the recorded checks",
            r"retained after those checks",
        ),
    ),
)

_AVAIL_MULTI_STAGE = re.compile(
    r"available during\s+planning.{0,60}generation.{0,60}checking",
    re.IGNORECASE | re.DOTALL,
)
_DISCLAIMER_PATTERNS = (
    re.compile(r"which parts of the (?:question|mcq) depended", re.I),
    re.compile(r"not by themselves which", re.I),
    re.compile(r"records show what information was used, but", re.I),
    re.compile(r"do not establish which parts", re.I),
    re.compile(
        r"records show that this information was used, but (?:they|the records) "
        r"do not establish",
        re.I,
    ),
)
_CONCRETE_SOURCE_PATTERNS = (
    re.compile(r"teaching (?:fact|information) (?:that|stating)", re.I),
    re.compile(r"linked to teaching", re.I),
    re.compile(r"records link", re.I),
    re.compile(r"recorded as used", re.I),
    re.compile(r"associated with teaching", re.I),
)
_UNINFORMATIVE_ACCEPTANCE = (
    re.compile(r"accepted after the recorded checks", re.I),
    re.compile(r"accepted after those checks", re.I),
    re.compile(r"selected and retained after the recorded checks", re.I),
    re.compile(r"still produced an acceptable mcq", re.I),
    re.compile(r"workflow still produced an acceptable", re.I),
    re.compile(r"the question was accepted\.", re.I),
    re.compile(r"and the question was accepted\b", re.I),
)
_IDENTITY_VS_ACCEPTANCE = re.compile(
    r"influenced which question was produced",
    re.I,
)
_LIMITATION_CUES = (
    "does not identify",
    "do not identify",
    "not which specific",
    "specific teaching fact determined",
)
_ROLE_CUES = (
    "helped keep",
    "helped determine",
    "influenced the incorrect",
    "influenced which question",
    "influenced how",
    "focus on",
    "necessary for reproducing",
    "required to reproduce",
    "instead favored",
    "instead produced",
    "alternative question",
    "factual question",
)


@dataclass(frozen=True)
class PresentationPolicyResult:
    """Lint result for participant-facing prose."""

    valid: bool
    violations: Tuple[str, ...] = ()
    length_class: str = "topic_limitation"
    word_count: int = 0
    sentence_count: int = 0
    chronology_categories: Tuple[str, ...] = ()
    details: Tuple[str, ...] = field(default=())

    def diagnostics(self) -> dict[str, Any]:
        return {
            "participant_word_count": self.word_count,
            "participant_sentence_count": self.sentence_count,
            "participant_length_policy": self.length_class,
            "participant_presentation_valid": self.valid,
            "participant_presentation_violations": list(self.violations),
        }


def count_presentation_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text or "")))


def _split_quote_aware_segments(text: str) -> list[str]:
    s = str(text or "").strip()
    if not s:
        return []
    segments: list[str] = []
    buf: list[str] = []
    quote: Optional[str] = None
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if quote is None:
            if ch == '"':
                quote = '"'
            elif ch == "“":
                quote = "“"
            elif ch in ".!?" and (i + 1 >= n or s[i + 1].isspace()):
                buf.append(ch)
                i += 1
                while i < n and s[i].isspace():
                    i += 1
                piece = "".join(buf).strip()
                if piece:
                    segments.append(piece)
                buf = []
                continue
        else:
            if quote == '"' and ch == '"':
                quote = None
            elif quote == "“" and ch == "”":
                quote = None
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments


def count_presentation_sentences(text: str) -> int:
    return len(_split_quote_aware_segments(text))


def presentation_length_class(facts: Any | None) -> str:
    """Map capability flags to a presentation budget class (condition-blind)."""
    if facts is None:
        return "topic_limitation"
    role_known = bool(getattr(facts, "information_role_known", False))
    role_effect = bool(getattr(facts, "counterfactual_effect_known", False))
    if role_known and role_effect:
        return "role_evidence"
    if getattr(facts, "counterfactual_component_findings", None) or getattr(
        facts, "controlled_change_findings", None
    ):
        if role_known or role_effect:
            return "role_evidence"
    if (
        bool(getattr(facts, "specific_source_known", False))
        and bool(getattr(facts, "recorded_use_known", False))
        and not role_known
    ):
        return "provenance_only"
    if role_known:
        return "role_evidence"
    if bool(getattr(facts, "specific_source_known", False)):
        return "provenance_only"
    return "topic_limitation"


def detect_workflow_chronology_categories(text: str) -> Tuple[str, ...]:
    """Return matched workflow-stage categories (not isolated process verbs)."""
    low = str(text or "")
    found: list[str] = []
    for name, patterns in _CHRONOLOGY_CATEGORIES:
        if any(re.search(pat, low, re.IGNORECASE) for pat in patterns):
            found.append(name)
    return tuple(found)


def has_detailed_workflow_chronology(text: str) -> bool:
    """True for multi-stage enumerations; False for compact generated/checked/retained."""
    cats = set(detect_workflow_chronology_categories(text))
    if not cats:
        return False
    upstream = cats & _UPSTREAM_CATEGORIES
    downstream = cats & _DOWNSTREAM_CATEGORIES
    if len(upstream) >= 2:
        return True
    if len(upstream) >= 1 and len(downstream) >= 2:
        return True
    if len(cats) >= 4:
        return True
    return False


def _has_concrete_source(text: str) -> bool:
    return any(p.search(text or "") for p in _CONCRETE_SOURCE_PATTERNS)


def _has_generic_disclaimer(text: str) -> bool:
    return any(p.search(text or "") for p in _DISCLAIMER_PATTERNS)


def has_redundant_provenance_boilerplate(text: str) -> bool:
    """Concrete observational facts plus both availability-triad and disclaimer."""
    body = str(text or "")
    if not _has_concrete_source(body):
        return False
    if not _AVAIL_MULTI_STAGE.search(body):
        return False
    return _has_generic_disclaimer(body)


def _visible_mcq_unchanged(facts: Any | None) -> bool:
    if facts is None:
        return False
    return (
        getattr(facts, "stem_changed", None) is False
        and getattr(facts, "correct_answer_changed", None) is False
        and getattr(facts, "distractors_changed", None) is False
    )


def _decision_effect_true(facts: Any | None) -> bool:
    if facts is None:
        return False
    if getattr(facts, "decision_effect", None) is True:
        return True
    if getattr(facts, "primary_decision_changed", None) is True:
        return True
    return False


def _has_uninformative_acceptance(text: str, facts: Any | None, length_class: str) -> bool:
    if length_class != "role_evidence":
        return False
    if _decision_effect_true(facts):
        return False
    if _IDENTITY_VS_ACCEPTANCE.search(text or ""):
        # Identity-vs-acceptance contrast is informative even if ACCEPT is stable.
        # Still flag a trailing generic "accepted after recorded checks" clause.
        if not any(p.search(text or "") for p in _UNINFORMATIVE_ACCEPTANCE):
            return False
        # The identity sentence itself uses "acceptable MCQ" — allow that pair.
        if _IDENTITY_VS_ACCEPTANCE.search(text or "") and re.search(
            r"could still produce an acceptable", text or "", re.I
        ):
            if not re.search(r"accepted after the recorded checks", text or "", re.I):
                return False
    return any(p.search(text or "") for p in _UNINFORMATIVE_ACCEPTANCE)


def validate_participant_presentation_policy(
    explanation: str,
    *,
    facts: Any | None = None,
) -> PresentationPolicyResult:
    """Lint participant prose. Does not judge faithfulness or evidence selection."""
    text = str(explanation or "").strip()
    length_class = presentation_length_class(facts)
    budget = LENGTH_BUDGETS[length_class]
    words = count_presentation_words(text)
    sentences = count_presentation_sentences(text)
    cats = detect_workflow_chronology_categories(text)
    violations: list[str] = []
    details: list[str] = []

    if has_detailed_workflow_chronology(text):
        violations.append("workflow_chronology")
        details.append("categories=" + ",".join(cats))
    if has_redundant_provenance_boilerplate(text):
        violations.append("redundant_provenance_boilerplate")
    if words > int(budget["violation_words"]):
        violations.append("excessive_length")
    if sentences > int(budget["max_sentences"]):
        violations.append("excessive_sentences")
    if _has_uninformative_acceptance(text, facts, length_class):
        violations.append("uninformative_acceptance")

    return PresentationPolicyResult(
        valid=not violations,
        violations=tuple(violations),
        length_class=length_class,
        word_count=words,
        sentence_count=sentences,
        chronology_categories=cats,
        details=tuple(details),
    )


def drop_chronology_only_sentences(text: str) -> str:
    """Drop sentences that are only multi-stage workflow narration."""
    kept: list[str] = []
    for seg in _split_quote_aware_segments(text):
        if has_detailed_workflow_chronology(seg) and not any(
            cue in seg.lower() for cue in _ROLE_CUES + _LIMITATION_CUES
        ):
            continue
        if has_detailed_workflow_chronology(seg) and not _has_concrete_source(seg):
            # Mixed sentence: try to keep the non-chronology prefix before a
            # generic workflow tail after a period-level join with "The workflow"
            # / "The material was organized".
            trimmed = _strip_trailing_chronology_clause(seg)
            if trimmed and not has_detailed_workflow_chronology(trimmed):
                kept.append(trimmed if trimmed.endswith((".", "?", "!")) else trimmed + ".")
                continue
            if not any(cue in seg.lower() for cue in _ROLE_CUES + _LIMITATION_CUES):
                continue
        kept.append(seg)
    return " ".join(kept).strip()


def _strip_trailing_chronology_clause(sentence: str) -> str:
    low = sentence.lower()
    markers = (
        " the workflow organized",
        " the material was organized",
        " the system organized",
        " organized the material",
        " organized the teaching material",
        " objectives were created",
        " supporting facts were retrieved",
    )
    cut = len(sentence)
    for m in markers:
        idx = low.find(m)
        if idx > 20:
            cut = min(cut, idx)
    out = sentence[:cut].rstrip(" ;,")
    return out.strip()


def drop_redundant_provenance_boilerplate(text: str) -> str:
    """Remove availability-triad and generic disclaimer sentences when sources remain."""
    if not has_redundant_provenance_boilerplate(text) and not (
        _AVAIL_MULTI_STAGE.search(text or "") or _has_generic_disclaimer(text or "")
    ):
        return str(text or "").strip()
    kept: list[str] = []
    for seg in _split_quote_aware_segments(text):
        if _AVAIL_MULTI_STAGE.search(seg) and not any(
            cue in seg.lower() for cue in _ROLE_CUES
        ):
            # Drop availability-only tails; keep a leading source clause.
            cleaned = _AVAIL_MULTI_STAGE.sub("", seg)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,")
            if _has_concrete_source(cleaned) and count_presentation_words(cleaned) >= 8:
                if not cleaned.endswith("."):
                    cleaned += "."
                kept.append(cleaned)
            continue
        if _has_generic_disclaimer(seg) and not any(cue in seg.lower() for cue in _ROLE_CUES):
            continue
        kept.append(seg)
    return " ".join(kept).strip()


def drop_uninformative_acceptance_sentences(text: str, *, facts: Any | None = None) -> str:
    if _decision_effect_true(facts):
        return str(text or "").strip()
    kept: list[str] = []
    for seg in _split_quote_aware_segments(text):
        if _IDENTITY_VS_ACCEPTANCE.search(seg) and re.search(
            r"could still produce an acceptable", seg, re.I
        ):
            kept.append(seg)
            continue
        if any(p.search(seg) for p in _UNINFORMATIVE_ACCEPTANCE) and not any(
            cue in seg.lower() for cue in _ROLE_CUES + _LIMITATION_CUES
        ):
            continue
        kept.append(seg)
    return " ".join(kept).strip()


def _is_limitation_or_availability_line(item: str) -> bool:
    low = str(item or "").lower()
    if any(cue in low for cue in (
        "depended on each fact",
        "not by themselves",
        "do not establish which parts",
        "records show what information was used",
        "available during planning",
        "available during generation",
        "available during checking",
    )):
        return True
    return has_detailed_workflow_chronology(item)


_LINK_PREFIXES = (
    "the records link the question to the teaching fact that ",
    "the records link this mcq to the teaching fact that ",
    "the records link this question to the teaching fact that ",
    "the records link this mcq to teaching facts about ",
    "the records link this question to teaching information stating that ",
    "the question was linked to teaching information stating that ",
    "other linked material described ",
    "topic: ",
    "learning objective: ",
    "source material from the teaching document ",
)


def source_content_clauses(facts: Any, *, max_facts: int = 3) -> Tuple[str, ...]:
    """Extract compact teaching-fact clauses from recorded-use evidence."""
    items: list[str] = []
    for pool in (
        getattr(facts, "relevant_provenance_facts", ()) or (),
        getattr(facts, "source_basis", ()) or (),
    ):
        for item in pool:
            if _is_limitation_or_availability_line(str(item)):
                continue
            clause = str(item or "").strip()
            if not clause:
                continue
            low = clause.lower()
            for prefix in _LINK_PREFIXES:
                if low.startswith(prefix):
                    clause = clause[len(prefix) :].strip()
                    low = clause.lower()
                    break
            clause = clause.rstrip(".;: ")
            # Split "fact; this fact was available during..."
            if "; this fact was available" in low:
                clause = clause.split(";", 1)[0].rstrip()
            if clause:
                items.append(clause)
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_facts:
            break
    return tuple(out)


def mechanical_trace_explanation(facts: Any | None = None) -> str:
    for lim in getattr(facts, "limitations", ()) or ():
        low = str(lim or "").lower()
        if any(
            cue in low
            for cue in (
                "causal",
                "controlled-change",
                "controlled change",
                "no reliable",
                "inconclusive",
                "valid comparison",
            )
        ):
            segs = _split_quote_aware_segments(str(lim))
            if segs:
                out = " ".join(segs[:2]).strip()
                if out and not out.endswith("."):
                    out += "."
                return out
    return TRACE_LIMITATION_SENTENCE


def mechanical_provenance_explanation(facts: Any) -> str:
    clauses = source_content_clauses(facts, max_facts=3)
    if not clauses:
        return (
            "The question was linked to recorded teaching information, using "
            "observational records of what was associated with the question."
        )
    if len(clauses) == 1:
        return (
            "The question was linked to teaching information stating that "
            f"{clauses[0]}."
        )
    if len(clauses) == 2:
        return (
            "The question was linked to teaching information stating that "
            f"{clauses[0]}, and that {clauses[1]}."
        )
    return (
        "The question was linked to teaching information stating that "
        f"{clauses[0]}, that {clauses[1]}, and that {clauses[2]}."
    )


def _visible_content_unchanged_from_finding_line_context(facts: Any) -> bool:
    return _visible_mcq_unchanged(facts)


def rewrite_necessary_for_reproduction_line(line: str, *, facts: Any | None) -> str:
    """Avoid implying stem/answer/distractor change when those were unchanged."""
    if facts is None:
        return line
    role = str(getattr(facts, "information_role", "") or "")
    if role != "necessary_for_reproduction":
        return line
    if not _visible_content_unchanged_from_finding_line_context(facts):
        return line
    low = str(line or "").lower()
    if not any(
        cue in low
        for cue in (
            "reproduc",
            "necessary for",
            "same educational focus",
            "corresponding question",
        )
    ):
        return line
    desc = str(getattr(facts, "primary_information_description", "") or "").strip()
    if desc:
        info = f'The information stating "{desc}"'
    else:
        info = "The tested information"
    return (
        f"{info} was required to reproduce the recorded result, although the "
        "visible stem, correct answer, and distractors remained unchanged."
    )


def filter_role_sentences(lines: Iterable[str], *, facts: Any | None) -> list[str]:
    """Keep evidence/effect sentences; drop chronology and uninformative acceptance."""
    out: list[str] = []
    for raw in lines:
        line = rewrite_necessary_for_reproduction_line(str(raw or "").strip(), facts=facts)
        if not line:
            continue
        if has_detailed_workflow_chronology(line) and not any(
            cue in line.lower() for cue in _ROLE_CUES
        ):
            continue
        if has_detailed_workflow_chronology(line):
            line = _strip_trailing_chronology_clause(line)
            if not line.endswith("."):
                line += "."
        if _has_uninformative_acceptance(line, facts, "role_evidence") and not any(
            cue in line.lower() for cue in _ROLE_CUES
        ):
            continue
        if any(p.search(line) for p in _UNINFORMATIVE_ACCEPTANCE) and not _decision_effect_true(
            facts
        ):
            if not _IDENTITY_VS_ACCEPTANCE.search(line):
                # Strip a trailing acceptance clause from an otherwise useful sentence.
                line = re.split(
                    r"\s+(?:The question was accepted|and the question was accepted|"
                    r"The material was organized|The workflow organized)",
                    line,
                    maxsplit=1,
                    flags=re.I,
                )[0].rstrip(" ;,")
                if line and not line.endswith("."):
                    line += "."
        if line:
            out.append(line)
    return out


def apply_presentation_cleanup(text: str, *, facts: Any | None = None) -> str:
    """Deterministic cleanup used by mechanical fallback (not an extra LLM round)."""
    cleaned = drop_chronology_only_sentences(text)
    cleaned = drop_redundant_provenance_boilerplate(cleaned)
    cleaned = drop_uninformative_acceptance_sentences(cleaned, facts=facts)
    return cleaned.strip()
