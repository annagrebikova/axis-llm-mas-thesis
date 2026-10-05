"""Shared final synthesis prompt for Trace / Trace+Provenance / AXIS.

The final synthesizer must not know which experimental condition produced the
evidence. Only the verbalized evidence content differs.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Tuple

from .lay_facts import LayExplanationFacts
from .presentation_policy import (
    apply_presentation_cleanup,
    filter_role_sentences,
    mechanical_provenance_explanation,
    mechanical_trace_explanation,
    rewrite_necessary_for_reproduction_line,
)
from .verbalizer import ExplanationEvidenceView

TARGET_WORD_RANGE = (1, 200)  # hard cap is MAX_PARTICIPANT_WORDS; length follows evidence flags
MAX_PARTICIPANT_WORDS = 200
PARTICIPANT_WORD_LIMIT = MAX_PARTICIPANT_WORDS


SHARED_FINAL_SYNTHESIS_PROMPT = """\
You are writing a user-facing explanation of verified evidence about an
automated quiz-generation question.

AUDIENCE
The reader is a university teacher who understands the teaching subject,
but has no knowledge of AI architecture, workflow graphs, provenance,
replay systems, internal artifact types, or AXIS.

GOAL
Write a clear explanation of the verified evidence available for this question.
Describe only what the evidence supports.

Do NOT enumerate workflow stages. Do not narrate organizing material, creating
learning objectives, retrieving supporting material, extracting facts, creating
a plan, generating the MCQ, checking coverage, or accepting the question as
the explanation. Those process steps are not participant content.

If specific teaching facts are recorded, name those facts.
If a supported information role is recorded, state that role and its content
effect.
If neither specific sources nor a role are known, write a limitation-centered
sentence: the question was generated from the teaching material, but the
record does not identify which specific teaching fact determined the wording,
correct answer, or distractors. A safely available broad topic may be named;
do not invent a topic.

Use observational verbs only when information_role_known = false:
linked to, recorded as used, associated with, available to the workflow.
Do not add a separate provenance disclaimer when the wording is already
observational.

If it contains supported role/effect findings, explain what role the tested
information played in:
- what the question asked / its focus,
- the correct answer,
- the incorrect answer options.

Use concrete examples from the verified evidence.
Do not invent missing levels of evidence.
Do not create fixed sections or headings.

LENGTH — match the verified evidence capabilities. Do not pad weaker evidence.

If specific_source_known = false and information_role_known = false:
Write 1 sentence (maximum 2). Prefer:
"The record shows that the question was generated from the teaching material,
but it does not identify which specific teaching fact determined the wording,
correct answer, or distractors."
Do not write a process chronology.

If specific_source_known = true and recorded_use_known = true and
information_role_known = false:
Write 1 sentence (maximum 2). Name 1–3 recorded teaching facts with
observational verbs only, e.g. "The question was linked to teaching
information stating that [fact]."
Do NOT add "available during planning, generation, and checking".
Do NOT add a generic sentence that the records do not show which part of the
question depended on each fact. Observational verbs already carry that limit.

If information_role_known = true and role_effect_known = true:
Write 2–3 compact sentences (maximum 4), in this order:
1. tested / linked teaching information and the supported role;
2. concrete content effect / factual-vs-alternative question when given;
3. answer, distractor, or formulation effect when supported;
4. decision/acceptance consequence only when the decision changed or
   acceptance is part of the supported finding.
Do not add a workflow summary. Do not end with "the question was accepted
after the recorded checks" when acceptance is unchanged and uninformative.

If information_role is necessary_for_reproduction and stem_changed,
correct_answer_changed, and distractors_changed are all false: do not imply
that the visible wording, correct answer, or distractors changed. Say the
information was required to reproduce the recorded result although the
visible stem, correct answer, and distractors remained unchanged.

Never exceed 200 words. Weaker evidence must stay shorter than richer evidence.
Do not repeat the same conclusion in different wording.

User question:
{user_question}

Question:
{question_stem}

Recorded decision:
{decision}

Verified explanation facts:
{verbalized_evidence}

EPISTEMIC RULES FOR INFORMATION USE
Use the capability information in the verified facts to determine how strongly
you may describe the role of teaching information.

If recorded_use_known = true but information_role_known = false:
- You may describe which teaching information was recorded as used, linked,
  associated with, or available to the workflow.
- Treat provenance as evidence of recorded use, not evidence of causal
  influence.
- Do NOT state or imply that a teaching fact influenced, shaped, determined,
  anchored, caused, drove, steered, changed, led to, or was responsible for:
    - the question,
    - the question focus,
    - the correct answer,
    - the incorrect answer options,
    - the final selection.
- Do NOT infer that a particular part of the MCQ depended on a fact merely
  because that fact was recorded as an input.
- Prefer observational wording such as:
    - "The records link the question to..."
    - "The question was linked to teaching information stating that..."
    - "This information was recorded as used..."
    - "This fact was associated with the question..."
    - "This information was available to the workflow..."
- Do not append generic workflow chronology.
- Do not append "available during planning, generation, and checking" when
  the sentence is already observational.
- Do not append a generic disclaimer that the records do not show which part
  depended on each fact when observational verbs are already used.

Only when information_role_known = true and verified facts contain a
supported information-role finding (role_effect_known = true) may you use
causal-role language such as:
- influenced
- shaped
- helped determine
- helped keep the question focused
- anchored

Do not hedge an unsupported causal claim with "probably", "likely", "may have",
or similar wording. If the evidence cannot establish the role, state the
limitation instead.

When provenance is available but counterfactual role evidence is not:

GOOD:
"The question was linked to teaching information stating that [recorded
teaching fact]."

GOOD:
"The records link this MCQ to the teaching fact that [recorded teaching fact]."

GOOD:
"The question was associated with teaching information describing [recorded
teaching fact]."

BAD:
"This information influenced which question was produced."

BAD:
"This fact shaped the incorrect answer options."

BAD:
"This information helped determine the correct answer."

BAD:
"This fact caused the question to be selected."

BAD:
"The supporting material anchored the expected answer."

RULES FOR COUNTERFACTUAL FINDINGS
When information_role_known = true and findings are marked as supported, state
the local role directly (influenced / helped keep focused / helped determine /
shaped incorrect options).
Do not hedge with probably / likely / apparently / presumably.
Refer to specific teaching content, not "the supporting material".
When question wording changed but acceptance stayed the same, prefer:
"This information influenced which question was produced."
Do not add that the workflow still produced an acceptable MCQ unless
decision_effect_known = true and the decision actually changed, or the user
question specifically asks about acceptance.
Do not say "not decisive for whether this question was selected" when a
different question was produced.

Same stem does NOT mean the same MCQ: if incorrect alternatives change, say so.

When a supported information-role finding includes a tested teaching-information
description, name that exact information. Do not replace it with vague phrases
such as "the supporting material", "the recorded reference", "the available
support", or "the remaining support" when a concrete description is provided.

If stem_changed = true and both factual_question and alternative_question are
given, state the content difference explicitly. Preferred pattern:
"With this information, the question asked '...'. When this information was
absent, the available content instead produced / favored '...'."

Never mention REMOVE(, REPLACE(, WHATIF(, NOOP, checkpoint, counterfactual branch,
replay branch, fork, EvidenceChunk, KnowledgeAtom, ArtifactRef, Trace, AXIS,
or experimental condition names.

Do not invent causal claims from provenance alone.
Do not expose IDs or versions.
"""


FAITHFULNESS_REPAIR_PROMPT = """\
Rewrite the explanation using the same verified facts.

The previous version failed verification checks for these reasons:
{warnings}

Previous explanation:
{previous_explanation}

Verified explanation facts (unchanged; do not invent missing levels):
{verbalized_evidence}

Rewrite rules:
- Keep the same verified facts. Do not add or remove factual information.
- Keep only claims supported by the verified facts above.
- Use the shortest explanation supported by the evidence. Do not pad.
- Remove generic workflow chronology. Do not add process stages such as
  organizing material, creating objectives, retrieving evidence, extracting
  facts, creating a plan, generating the MCQ, checking coverage, or accepting
  the question.
- Preserve concrete source facts with observational wording (linked to,
  recorded as used, associated with, available to the workflow).
- Do not append "available during planning, generation, and checking" or a
  generic disclaimer that the records do not show which part depended on each
  fact when the wording is already observational.
- Preserve supported role/effect wording. Do not replace a supported role
  with observational provenance verbs.
- If recorded_use_known = true but information_role_known = false, replace
  causal-role wording such as "influenced", "shaped", "determined",
  "anchored", "caused", or "helped determine" with observational wording such
  as "was used", "was linked", "was associated with", "was recorded as input",
  or "was available to the workflow".
- Do not infer which MCQ component depended on a fact from provenance alone.
- Only use causal-role language when information_role_known = true and
  role_effect_known = true with a supported information-role finding.
- If information_role_known = true and role_effect_known = true with a
  supported information-role finding, KEEP the supported role wording from the
  verified facts. Use:
    focus_shaping -> "helped keep the question focused on..."
    answer_anchoring -> "helped determine the correct answer..."
    distractor_shaping -> "influenced the incorrect alternatives..."
    formulation_shaping -> "influenced how the question/answer was phrased..."
  Do not replace a supported role with observational provenance verbs such as
  "associated with", "linked to", or "available during".
- Name the exact tested-information description as given in the verified facts.
  If the description is a short label, say that exact label. Do not
  expand it using another related source fact.
- Do not add or remove factual information.
- LENGTH: follow the same evidence-capability budget as the original task.
  specific_source_known = false and information_role_known = false -> 1 sentence
  (maximum 2).
  specific_source_known = true, recorded_use_known = true,
  information_role_known = false -> 1 sentence (maximum 2).
  information_role_known = true and role_effect_known = true -> 2–3 compact
  sentences (maximum 4).
  HARD: never exceed 200 words. Do not pad weaker evidence.
- If space is limited, keep in this order: exact tested information; supported
  information-role findings; required factual-vs-alternative question-content
  effect; acceptance consequence only when the decision changed. Never keep
  generic workflow chronology.
- Omit uninformative acceptance sentences such as "the question was accepted
  after the recorded checks" when the decision did not change.
- If necessary_for_reproduction is supported but stem, correct answer, and
  distractors are unchanged, say the information was required to reproduce the
  recorded result although the visible MCQ content remained unchanged.
- No headings. No internal IDs, REMOVE/REPLACE/WHATIF/NOOP, or condition names.
{warning_specific_rules}

Return only the rewritten explanation.
"""


def build_shared_final_prompt(
    *,
    user_question: str,
    factual_decision: str,
    question_stem: str,
    verbalized_evidence: str,
) -> str:
    """Identical final prompt for all three conditions (no condition labels)."""
    decision = str(factual_decision or "").strip().upper() or "UNKNOWN"
    if decision in {"ACCEPTED", "ACCEPT"}:
        decision = "ACCEPT"
    elif decision in {"NEEDS_REVISION", "REVISE"}:
        decision = "REVISE"
    elif decision in {"REJECTED", "REJECT"}:
        decision = "REJECT"
    return SHARED_FINAL_SYNTHESIS_PROMPT.format(
        user_question=user_question,
        question_stem=question_stem,
        decision=decision,
        verbalized_evidence=verbalized_evidence,
    )


def _warning_specific_repair_rules(warnings: Tuple[str, ...] | list[str]) -> str:
    """Extra repair instructions driven by deterministic validator warnings."""
    joined = "\n".join(str(w) for w in (warnings or ()) if str(w).strip())
    extras: list[str] = []
    if "supporting material without the tested information" in joined:
        extras.append(
            "- Name the concrete tested teaching information from the verified "
            "facts using the supplied tested-information description exactly as "
            "written. Do not replace it with a generic source/support phrase, "
            "and do not expand a short label into another related source fact."
        )
    if "omits valid tested-intervention evidence" in joined:
        extras.append(
            "- Preserve at least one supported information-role finding from "
            "the verified facts."
        )
    if "stem change not stated" in joined:
        extras.append(
            "- State the verified factual-vs-alternative question-stem "
            "difference using the supplied stems."
        )
    if "workflow_chronology" in joined:
        extras.append(
            "- Remove generic workflow chronology. Do not list organizing, "
            "objectives, retrieval, extraction, planning, generation, checking, "
            "or acceptance as explanation content."
        )
    if "redundant_provenance_boilerplate" in joined:
        extras.append(
            "- Keep the concrete teaching facts. Remove 'available during "
            "planning, generation, and checking' and any generic sentence "
            "that the records do not show which part depended on each fact."
        )
    if "uninformative_acceptance" in joined:
        extras.append(
            "- Remove uninformative acceptance sentences. Mention acceptance "
            "only if the decision changed or the user asked about it."
        )
    if "excessive_length" in joined or "excessive_sentences" in joined:
        extras.append(
            "- Shorten to the evidence-capability budget. Do not pad. Prefer "
            "one sentence when sources/role are not both known."
        )
    if not extras:
        return ""
    return "Warning-specific repair instructions:\n" + "\n".join(extras)


def build_faithfulness_repair_prompt(
    *,
    previous_explanation: str,
    warnings: Tuple[str, ...] | list[str],
    verbalized_evidence: str,
) -> str:
    """One-shot same-model repair prompt (condition-blind)."""
    warn_lines = "\n".join(f"- {w}" for w in (warnings or ()) if str(w).strip())
    if not warn_lines:
        warn_lines = "- unsupported claim relative to verified facts"
    extra = _warning_specific_repair_rules(warnings)
    if extra:
        extra = "\n" + extra
    return FAITHFULNESS_REPAIR_PROMPT.format(
        warnings=warn_lines,
        previous_explanation=str(previous_explanation or "").strip(),
        verbalized_evidence=str(verbalized_evidence or "").strip(),
        warning_specific_rules=extra,
    )


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text or "")))


def participant_quotes_are_balanced(text: str) -> bool:
    """True when straight/curly quotation marks are balanced."""
    s = str(text or "")
    if s.count('"') % 2 != 0:
        return False
    if s.count("“") != s.count("”"):
        return False
    return True


def _split_quote_aware_segments(text: str) -> list[str]:
    """Split on sentence boundaries that are outside quoted spans.

    Quoted ``"..."`` / ``“...”`` spans are atomic, including internal periods
    inside an MCQ stem.
    """
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


_HIGH_PRIORITY_CUES = (
    "helped keep",
    "helped determine",
    "influenced the incorrect",
    "influenced which question",
    "influenced how",
    "with this information, the question asked",
    "when this information was absent",
    "instead produced",
    "instead favored",
    "factual-vs-alternative",
)
_LOW_PRIORITY_CUES = (
    "organized the teaching material",
    "created learning objectives",
    "retrieved candidate",
    "extracted concise factual",
    "created a plan for the question",
)
_PROVENANCE_CUES = (
    "records link",
    "available during",
    "was used",
    "was linked",
    "was recorded as",
)


def _segment_priority(segment: str) -> int:
    """Lower numbers are dropped first under word pressure."""
    low = segment.lower()
    if ('"' in segment or "“" in segment) and (
        "?" in segment or "which of the following" in low
    ):
        return 3
    if any(cue in low for cue in _HIGH_PRIORITY_CUES):
        return 3
    if any(cue in low for cue in _LOW_PRIORITY_CUES):
        return 0
    if any(cue in low for cue in _PROVENANCE_CUES):
        return 1
    return 2


def compress_participant_explanation(
    text: str,
    *,
    max_words: int = MAX_PARTICIPANT_WORDS,
    fallback: str = "",
) -> Tuple[str, str]:
    """Quote-safe compression. Returns ``(text, strategy)``.

    strategy: none | safe_compression | mechanical_fallback
    Never cuts inside a quoted span. If a quoted stem cannot fit, the whole
    quoted sentence is omitted rather than truncated.
    """
    original = str(text or "").strip()
    if count_words(original) <= max_words and participant_quotes_are_balanced(original):
        return original, "none"

    segments = _split_quote_aware_segments(original)
    kept = list(segments)
    while kept and count_words(" ".join(kept)) > max_words:
        priorities = [_segment_priority(seg) for seg in kept]
        drop_rank = min(priorities)
        # Drop the first lowest-priority segment (generic workflow before roles).
        idx = next(i for i, p in enumerate(priorities) if p == drop_rank)
        kept.pop(idx)
    compressed = " ".join(seg.strip() for seg in kept if seg and str(seg).strip()).strip()
    if (
        compressed
        and count_words(compressed) <= max_words
        and participant_quotes_are_balanced(compressed)
    ):
        return compressed, "safe_compression"

    fb = str(fallback or "").strip()
    if fb:
        fb_text, fb_strategy = compress_participant_explanation(
            fb, max_words=max_words, fallback=""
        )
        if count_words(fb_text) <= max_words and participant_quotes_are_balanced(fb_text):
            return fb_text, "mechanical_fallback"
    if compressed and participant_quotes_are_balanced(compressed):
        return compressed, "safe_compression"
    return (fb or compressed or original), (
        "mechanical_fallback" if fb else "safe_compression"
    )


def enforce_participant_word_limit(
    text: str,
    *,
    max_words: int = MAX_PARTICIPANT_WORDS,
    fallback: str = "",
) -> Tuple[str, bool]:
    """Ensure participant text is within the hard word maximum.

    Returns (text, used_compression_or_fallback).
    Quoted MCQ stems are treated as atomic; never truncated mid-quote.
    """
    out, strategy = compress_participant_explanation(
        text, max_words=max_words, fallback=fallback
    )
    return out, strategy != "none"


def _unique_keep_order(lines: Tuple[str, ...] | list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def count_participant_sentences(text: str) -> int:
    return len(_split_quote_aware_segments(text))


def _first_sentence(text: str) -> str:
    segs = _split_quote_aware_segments(text)
    if segs:
        return segs[0].strip()
    return str(text or "").strip()


def _limit_sentences(text: str, max_n: int) -> str:
    segs = _split_quote_aware_segments(text)
    if max_n <= 0:
        return ""
    return " ".join(segs[:max_n]).strip()


def _as_one_sentence(*parts: str) -> str:
    clauses: list[str] = []
    for part in parts:
        for seg in _split_quote_aware_segments(str(part or "").strip()) or (
            [str(part or "").strip()] if str(part or "").strip() else []
        ):
            clause = seg.strip()
            if clause.endswith("."):
                clause = clause[:-1].rstrip()
            if clause:
                clauses.append(clause)
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0] + "."
    return "; ".join(clauses) + "."


def evidence_length_profile(facts: LayExplanationFacts) -> str:
    """Return trace | recorded_use | role from capability flags (condition-blind)."""
    role_known = bool(facts.information_role_known)
    role_effect = bool(facts.counterfactual_effect_known)
    if role_known and role_effect:
        return "role"
    if facts.counterfactual_component_findings or facts.controlled_change_findings:
        return "role"
    if (
        bool(facts.specific_source_known)
        and bool(facts.recorded_use_known)
        and not role_known
    ):
        return "recorded_use"
    if not bool(facts.specific_source_known) and not role_known:
        return "trace"
    if role_known:
        return "role"
    if bool(facts.specific_source_known):
        return "recorded_use"
    return "trace"


def _mechanical_trace_sentence(facts: LayExplanationFacts) -> str:
    return mechanical_trace_explanation(facts)


def _mechanical_recorded_use_sentence(facts: LayExplanationFacts) -> str:
    return mechanical_provenance_explanation(facts)


def _mechanical_role_sentences(facts: LayExplanationFacts) -> str:
    role_lines = _unique_keep_order(
        list(facts.counterfactual_component_findings or ())
        + list(facts.question_content_findings or ())
        + list(facts.controlled_change_findings or ())
    )
    deduped: list[str] = []
    for line in filter_role_sentences(role_lines, facts=facts):
        low = line.lower()
        if deduped and "influenced" in low and "influenced" in deduped[-1].lower():
            if "decisive" in low or "acceptable" in low or "produced" in low:
                deduped.append(line)
            continue
        if any(
            cue in low
            for cue in (
                "which parts of the MCQ depended",
                "not by themselves",
                "available during planning, generation, and checking",
            )
        ):
            continue
        deduped.append(line)
    if not deduped and facts.primary_information_description:
        desc = str(facts.primary_information_description).rstrip(".")
        role = str(facts.information_role or "").replace("_", " ")
        opener = (
            f"The question was linked to teaching information stating that {desc}."
        )
        if str(facts.information_role or "") == "necessary_for_reproduction":
            opener = rewrite_necessary_for_reproduction_line(
                (
                    f'The information stating "{desc}" is important for '
                    "reproducing this particular selected question."
                ),
                facts=facts,
            )
        elif role:
            opener = opener.rstrip(".") + (
                f" A controlled check supported a {role} role for that information."
            )
        deduped.append(opener)
    text = " ".join(deduped[:4]).strip()
    if count_participant_sentences(text) < 2 and facts.primary_information_description:
        opener = (
            "The question was linked to teaching information stating that "
            f"{str(facts.primary_information_description).rstrip('.')}."
        )
        if opener.lower() not in text.lower():
            text = (opener + " " + text).strip()
    text = apply_presentation_cleanup(text, facts=facts)
    text = _limit_sentences(text, 4)
    if not text:
        text = (
            f"The recorded decision for this question was {facts.decision}."
            if facts.decision
            else "Insufficient verified facts were available for an explanation."
        )
    return text


def mechanical_synthesize_from_lay_facts(*, facts: LayExplanationFacts) -> str:
    """Deterministic concise fallback (no mandatory headings).

    Length follows evidence capabilities:
    trace / recorded-use → about 1 sentence; supported role → 2–4 sentences.
    """
    profile = evidence_length_profile(facts)
    if profile == "trace":
        text = _mechanical_trace_sentence(facts)
    elif profile == "recorded_use":
        text = _mechanical_recorded_use_sentence(facts)
    else:
        text = _mechanical_role_sentences(facts)

    if not text:
        text = (
            f"The recorded decision for this question was {facts.decision}."
            if facts.decision
            else "Insufficient verified facts were available for an explanation."
        )

    text, _ = enforce_participant_word_limit(text)
    if not participant_quotes_are_balanced(text):
        text, _ = enforce_participant_word_limit(
            text.replace("“", '"').replace("”", '"'),
            fallback=(
                f"The recorded decision for this question was {facts.decision}."
                if facts.decision
                else "Insufficient verified facts were available for an explanation."
            ),
        )
    return text


def mechanical_synthesize_from_view(
    *,
    view: ExplanationEvidenceView,
) -> str:
    """Deterministic concise fallback when no LLM is available / repair."""
    if view.lay_facts is not None:
        return mechanical_synthesize_from_lay_facts(facts=view.lay_facts)
    facts = LayExplanationFacts(
        decision=view.factual_decision,
        educational_purpose=view.observed_process,
        trace_narrative=view.observed_process,
        source_basis=view.recorded_dependencies,
        decision_basis=view.decision_evidence,
        controlled_change_findings=view.tested_interventions,
        question_content_findings=view.tested_interventions,
        limitations=view.limitations,
    )
    return mechanical_synthesize_from_lay_facts(facts=facts)


def synthesize_explanation(
    *,
    user_question: str,
    factual_decision: str,
    question_stem: str,
    evidence_view: ExplanationEvidenceView,
    llm_generate: Optional[Callable[..., Any]] = None,
) -> str:
    """Shared synthesis path for all conditions."""
    prompt = build_shared_final_prompt(
        user_question=user_question,
        factual_decision=factual_decision,
        question_stem=question_stem,
        verbalized_evidence=evidence_view.to_prompt_text(),
    )
    text = ""
    if llm_generate is not None:
        try:
            text = str(llm_generate(prompt) or "").strip()
        except Exception:
            text = ""
        return text
    return mechanical_synthesize_from_view(view=evidence_view)
