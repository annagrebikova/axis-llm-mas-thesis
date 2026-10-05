"""Blueprint-driven MCQ generation — Step 3 (no topic strings, PDF-grounded only)."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .adapters import candidate_mcq_to_candidate_question
from .entities import AnswerOption, CandidateMCQ, MCQBlueprint, SourceRef
from .option_positioning import (
    OPTION_LABELS,
    assign_deterministic_option_positions,
)
from .question_limit import first_generation_payload

if TYPE_CHECKING:
    from ...core.state import CooperativeState

logger = logging.getLogger(__name__)

STEM_TEMPLATES = {
    "basic_definition": "Which statement best defines the concept described in the evidence?",
    "key_characteristic": "Which is a key characteristic supported by the evidence?",
    "difference_identification": "Which statement correctly distinguishes the concepts in the evidence?",
    "advantage_disadvantage": "Which advantage is supported by the evidence?",
    "example_scenario": "In a scenario based on the evidence, which action is most appropriate?",
}


def log_ok(message: str) -> None:
    logger.info(message)
    print(f"[OK] {message}")


def log_warn(message: str) -> None:
    logger.warning(message)
    print(f"[WARN] {message}")


def _normalize_title(title: str, section_id: str) -> str:
    cleaned = (title or "").strip()
    if not cleaned or cleaned.lower() == "general":
        return section_id
    return cleaned


def _source_refs_from_context(
    blueprint: MCQBlueprint,
    state: "CooperativeState",
) -> List[SourceRef]:
    refs: List[SourceRef] = []
    seen: set = set()

    for chunk_id in blueprint.evidence_chunk_ids:
        chunk = state.get_chunk(chunk_id)
        page = None
        doc_id = blueprint.section_id
        quote = None
        if chunk is not None:
            meta = getattr(chunk, "metadata", {}) or {}
            page = meta.get("page_number")
            doc_id = str(meta.get("pdf_id") or meta.get("source_file") or blueprint.section_id)
            quote = (chunk.content or "")[:240]
        key = ("chunk", chunk_id)
        if key not in seen:
            refs.append(SourceRef(doc_id=doc_id, page=page, chunk_id=chunk_id, quote=quote))
            seen.add(key)

    for atom_id in blueprint.target_atom_ids:
        atom = state.get_atom(atom_id)
        key = ("atom", atom_id)
        if key in seen:
            continue
        quote = atom.content[:240] if atom else None
        refs.append(
            SourceRef(
                doc_id=blueprint.section_id,
                page=None,
                chunk_id=None,
                atom_id=atom_id,
                quote=quote,
            )
        )
        seen.add(key)
    return refs


def build_generation_context_from_blueprint(
    blueprint: MCQBlueprint,
    state: "CooperativeState",
) -> Dict[str, Any]:
    """Collect PDF-grounded context for one blueprint — never uses topic 'general'."""
    section = state.get_section(blueprint.section_id)
    raw_title = section.title if section else blueprint.section_id
    section_title = _normalize_title(raw_title, blueprint.section_id)

    learning_objective = ""
    for lo in state.learning_objectives:
        if lo.lo_id == blueprint.lo_id:
            learning_objective = lo.objective_text
            break

    atoms: List[Dict[str, Any]] = []
    for atom_id in blueprint.target_atom_ids:
        atom = state.get_atom(atom_id)
        if atom is None:
            continue
        atoms.append({
            "atom_id": atom.atom_id,
            "content": atom.content,
            "atom_type": atom.atom_type,
            "section_id": atom.section_id,
            "lo_id": atom.lo_id,
        })

    evidence_chunks: List[Dict[str, Any]] = []
    for chunk_id in blueprint.evidence_chunk_ids:
        chunk = state.get_chunk(chunk_id)
        if chunk is None:
            continue
        meta = getattr(chunk, "metadata", {}) or {}
        evidence_chunks.append({
            "chunk_id": chunk_id,
            "content": chunk.content,
            "page_number": meta.get("page_number"),
            "source_file": meta.get("source_file"),
            "pdf_id": meta.get("pdf_id"),
        })

    source_refs = _source_refs_from_context(blueprint, state)
    context_flags: List[str] = []
    if section is None:
        context_flags.append("missing_section_title")
    if not evidence_chunks:
        context_flags.append("insufficient_evidence")
    if not atoms:
        context_flags.append("insufficient_atoms")

    return {
        "blueprint_id": blueprint.blueprint_id,
        "section_id": blueprint.section_id,
        "lo_id": blueprint.lo_id,
        "section_title": section_title,
        "learning_objective": learning_objective,
        "question_type": blueprint.question_type,
        "difficulty": blueprint.difficulty,
        "bloom_level": blueprint.bloom_level,
        "generation_goal": blueprint.generation_goal,
        "atoms": atoms,
        "evidence_chunks": evidence_chunks,
        "source_refs": source_refs,
        "flags": context_flags,
    }


def _build_llm_prompt(context: Dict[str, Any]) -> str:
    atoms_block = json.dumps(context.get("atoms") or [], indent=2, ensure_ascii=False)
    chunks_block = json.dumps(context.get("evidence_chunks") or [], indent=2, ensure_ascii=False)
    return f"""You are generating a multiple-choice question only from the provided PDF evidence.
Do not use outside knowledge.
If the evidence is insufficient, return INSUFFICIENT_EVIDENCE.

Section title: {context.get("section_title")}
Learning objective: {context.get("learning_objective")}
Question type: {context.get("question_type")}
Difficulty: {context.get("difficulty")}
Bloom level: {context.get("bloom_level")}

Knowledge atoms:
{atoms_block}

Evidence chunks:
{chunks_block}

Generate exactly one multiple-choice question with four answer options.
Exactly one option must be correct.
All options must be plausible.
Distractors should reflect likely misconceptions.
The correct answer must be supported by the provided evidence.
Explain why the correct answer is correct.
For each distractor, explain why it is wrong.
Do not create trivia questions.
Do not use general world knowledge.
Do not ask about information that is not present in the evidence.

Return JSON only with this schema:
{{
  "stem": "...",
  "options": [
    {{
      "option_id": "A",
      "text": "...",
      "is_correct": false,
      "explanation": "...",
      "misconception": "..."
    }},
    {{
      "option_id": "B",
      "text": "...",
      "is_correct": false,
      "explanation": "...",
      "misconception": "..."
    }},
    {{
      "option_id": "C",
      "text": "...",
      "is_correct": false,
      "explanation": "...",
      "misconception": "..."
    }},
    {{
      "option_id": "D",
      "text": "...",
      "is_correct": false,
      "explanation": "...",
      "misconception": "..."
    }}
  ],
  "correct_option_id": "<letter of the single correct option>",
  "answer_explanation": "...",
  "used_atom_ids": [],
  "used_chunk_ids": []
}}

Hard rules for correctness labels:
- exactly one option must have is_correct=true
- correct_option_id must match that option
- option display position will be normalized by the workflow after generation
"""


def _extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned == "INSUFFICIENT_EVIDENCE":
        return {"insufficient_evidence": True}
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")
    if start_arr >= 0 and (start_obj < 0 or start_arr < start_obj):
        end_arr = cleaned.rfind("]")
        if end_arr > start_arr:
            try:
                return json.loads(cleaned[start_arr : end_arr + 1])
            except json.JSONDecodeError:
                return None
        return None
    if start_obj >= 0:
        end_obj = cleaned.rfind("}")
        if end_obj > start_obj:
            try:
                return json.loads(cleaned[start_obj : end_obj + 1])
            except json.JSONDecodeError:
                return None
    return None


def _invoke_llm(llm: Any, prompt: str) -> str:
    from ...replay.llm_determinism import workflow_llm_context

    with workflow_llm_context(layer="native_mcq", component="mcq_generation"):
        response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def generate_stem_and_key_from_blueprint(context: Dict[str, Any]) -> Dict[str, Any]:
    """Derive stem and keyed correct answer text from blueprint context (stub path)."""
    section_title = context.get("section_title") or context.get("section_id")
    qtype = context.get("question_type") or "basic_definition"
    stem = STEM_TEMPLATES.get(qtype, STEM_TEMPLATES["basic_definition"])
    stem = f"{stem} (Section: {section_title})"

    atoms = context.get("atoms") or []
    chunks = context.get("evidence_chunks") or []
    if atoms:
        key_text = atoms[0]["content"][:120]
    elif chunks:
        key_text = chunks[0]["content"][:120]
    else:
        key_text = f"Concept from section {section_title}"

    return {"stem": stem, "correct_text": key_text}


def generate_distractors_from_blueprint(
    context: Dict[str, Any],
    correct_text: str,
) -> List[Dict[str, Any]]:
    """Build three plausible distractors grounded in section context."""
    section_title = context.get("section_title") or context.get("section_id")
    base_wrong = [
        f"A statement not supported by the evidence about {section_title}.",
        f"A plausible but incorrect interpretation of {section_title}.",
        f"A common misconception unrelated to the cited PDF evidence.",
    ]
    distractors: List[Dict[str, Any]] = []
    for idx, text in enumerate(base_wrong, start=1):
        if text.strip().lower() == correct_text.strip().lower():
            text = f"{text} (variant {idx})"
        distractors.append({
            "option_id": OPTION_LABELS[idx],
            "text": text,
            "is_correct": False,
            "explanation": "This option is not supported by the provided evidence.",
            "misconception": "Likely misconception or unsupported claim.",
        })
    return distractors


def generate_explanations_from_blueprint(
    context: Dict[str, Any],
    *,
    correct_text: str,
) -> Dict[str, str]:
    lo = context.get("learning_objective") or context.get("section_title")
    return {
        "answer_explanation": (
            f"The correct answer is supported by the PDF evidence for learning objective: {lo}."
        ),
        "correct_option_explanation": "Directly supported by cited atoms and evidence chunks.",
    }


def assemble_candidate_mcq(
    *,
    blueprint: MCQBlueprint,
    state: "CooperativeState",
    context: Dict[str, Any],
    parsed: Optional[Dict[str, Any]] = None,
    stub_parts: Optional[Dict[str, Any]] = None,
) -> CandidateMCQ:
    """Assemble CandidateMCQ from LLM JSON or deterministic stub parts."""
    q_id = state.generate_id("q")
    source_refs: List[SourceRef] = list(context.get("source_refs") or [])
    flags: List[str] = list(context.get("flags") or [])

    if parsed and parsed.get("insufficient_evidence"):
        if "insufficient_evidence" not in flags:
            flags.append("insufficient_evidence")
        return CandidateMCQ(
            q_id=q_id,
            blueprint_id=blueprint.blueprint_id,
            stem="",
            options=[],
            correct_option_id="",
            answer_explanation="",
            source_refs=source_refs,
            flags=flags,
            status="draft",
        )

    if parsed:
        stem = str(parsed.get("stem") or "").strip()
        answer_explanation = str(parsed.get("answer_explanation") or "").strip()
        parsed_correct_id = str(parsed.get("correct_option_id") or "").strip()
        options: List[AnswerOption] = []
        for raw in parsed.get("options") or []:
            opt_id = str(raw.get("option_id") or "")
            options.append(
                AnswerOption(
                    option_id=opt_id,
                    text=str(raw.get("text") or "").strip(),
                    is_correct=bool(raw.get("is_correct")),
                    explanation=raw.get("explanation"),
                    misconception=raw.get("misconception"),
                    source_refs=source_refs[:1] if raw.get("is_correct") else [],
                )
            )
        while len(options) < 4:
            label = OPTION_LABELS[len(options)]
            options.append(
                AnswerOption(
                    option_id=label,
                    text=f"{label}) Incomplete option",
                    is_correct=False,
                    misconception="Placeholder distractor.",
                )
            )
        correct_option_id = _resolve_correct_option_id(options, parsed_correct_id)
        options, correct_option_id = assign_deterministic_option_positions(
            options,
            stable_key=q_id,
            correct_option_id=correct_option_id or None,
        )
        mcq = CandidateMCQ(
            q_id=q_id,
            blueprint_id=blueprint.blueprint_id,
            stem=stem,
            options=options,
            correct_option_id=correct_option_id,
            answer_explanation=answer_explanation,
            source_refs=source_refs,
            flags=flags,
            status="draft",
        )
        return validate_candidate_mcq_structure(mcq)

    assert stub_parts is not None
    stem = stub_parts["stem"]
    correct_text = stub_parts["correct_text"]
    explanations = stub_parts["explanations"]
    distractors = stub_parts["distractors"]

    # Temporary local labels; display position is normalized by q_id below.
    correct_option = AnswerOption(
        option_id="TMP_CORRECT",
        text=correct_text,
        is_correct=True,
        explanation=explanations.get("correct_option_explanation"),
        misconception=None,
        source_refs=source_refs[:1],
    )
    wrong_options = [
        AnswerOption(
            option_id=d.get("option_id") or f"TMP_D{i}",
            text=d["text"],
            is_correct=False,
            explanation=d.get("explanation"),
            misconception=d.get("misconception"),
        )
        for i, d in enumerate(distractors)
    ]
    options = [correct_option] + wrong_options
    options, correct_option_id = assign_deterministic_option_positions(
        options,
        stable_key=q_id,
        correct_option_id="TMP_CORRECT",
    )
    mcq = CandidateMCQ(
        q_id=q_id,
        blueprint_id=blueprint.blueprint_id,
        stem=stem,
        options=options,
        correct_option_id=correct_option_id,
        answer_explanation=explanations.get("answer_explanation", ""),
        source_refs=source_refs,
        flags=flags,
        status="draft",
    )
    return validate_candidate_mcq_structure(mcq)


def _resolve_correct_option_id(
    options: List[AnswerOption],
    parsed_correct_id: str,
) -> str:
    """Derive correct_option_id from is_correct; never invent label A."""
    marked = [o for o in options if o.is_correct]
    if len(marked) == 1:
        return marked[0].option_id
    if len(marked) == 0 and parsed_correct_id:
        matches = [o for o in options if o.option_id == parsed_correct_id]
        if len(matches) == 1:
            matches[0].is_correct = True
            return parsed_correct_id
    if len(marked) > 1:
        return parsed_correct_id if parsed_correct_id else ""
    return parsed_correct_id or ""


def validate_candidate_mcq_structure(candidate: CandidateMCQ) -> CandidateMCQ:
    """Minimal in-MAS structural validation — not IWF checking."""
    flags = list(candidate.flags)

    if not (candidate.stem or "").strip():
        flags.append("empty_stem")
    if len(candidate.options) < 4:
        flags.append("too_few_options")

    option_ids = [o.option_id for o in candidate.options]
    if len(option_ids) != len(set(option_ids)):
        flags.append("duplicate_option_ids")

    texts = [(o.text or "").strip().lower() for o in candidate.options if (o.text or "").strip()]
    if len(texts) != len(set(texts)):
        flags.append("duplicate_options")

    if any(not (o.text or "").strip() for o in candidate.options):
        flags.append("empty_option_text")

    correct = [o for o in candidate.options if o.is_correct]
    if not correct:
        flags.append("no_correct_option")
    elif len(correct) > 1:
        flags.append("multiple_correct_options")

    if candidate.correct_option_id not in option_ids:
        flags.append("correct_option_id_missing")

    if not (candidate.answer_explanation or "").strip():
        flags.append("missing_answer_explanation")

    if not candidate.source_refs:
        flags.append("missing_source_refs")

    chunk_used = any(ref.chunk_id for ref in candidate.source_refs)
    if not chunk_used:
        flags.append("insufficient_evidence")

    candidate.flags = sorted(set(flags))
    return candidate


def generate_candidate_from_blueprint(
    blueprint: MCQBlueprint,
    state: "CooperativeState",
    llm: Optional[Any] = None,
) -> CandidateMCQ:
    """Generate one CandidateMCQ from blueprint context — never uses topic 'general'."""
    context = build_generation_context_from_blueprint(blueprint, state)
    section_title = context.get("section_title") or blueprint.section_id
    log_ok(
        f"Generating MCQ from blueprint {blueprint.blueprint_id} "
        f"(section={section_title!r}, lo={blueprint.lo_id})"
    )

    if "general" in str(context.get("section_title", "")).lower():
        log_warn(f"Blueprint {blueprint.blueprint_id} context contains forbidden topic 'general'")
    if "general" in str(context.get("learning_objective", "")).lower():
        log_warn(f"Blueprint {blueprint.blueprint_id} LO text contains forbidden topic 'general'")

    if "insufficient_evidence" in context.get("flags", []):
        mcq = assemble_candidate_mcq(
            blueprint=blueprint,
            state=state,
            context=context,
            parsed={"insufficient_evidence": True},
        )
        log_warn(f"CandidateMCQ {mcq.q_id} has flags: {mcq.flags}")
        return mcq

    if llm is not None:
        prompt = _build_llm_prompt(context)
        raw = _invoke_llm(llm, prompt)
        extracted = _extract_json(raw)
        parsed = first_generation_payload(extracted)
        if parsed is None:
            mcq = assemble_candidate_mcq(
                blueprint=blueprint,
                state=state,
                context={**context, "flags": context.get("flags", []) + ["llm_parse_error"]},
                parsed={"insufficient_evidence": True},
            )
            log_warn(f"CandidateMCQ {mcq.q_id} LLM parse failed; flags={mcq.flags}")
            return mcq
        mcq = assemble_candidate_mcq(
            blueprint=blueprint,
            state=state,
            context=context,
            parsed=parsed,
        )
    else:
        stem_key = generate_stem_and_key_from_blueprint(context)
        distractors = generate_distractors_from_blueprint(context, stem_key["correct_text"])
        explanations = generate_explanations_from_blueprint(context, correct_text=stem_key["correct_text"])
        mcq = assemble_candidate_mcq(
            blueprint=blueprint,
            state=state,
            context=context,
            stub_parts={
                "stem": stem_key["stem"],
                "correct_text": stem_key["correct_text"],
                "distractors": distractors,
                "explanations": explanations,
            },
        )

    if mcq.flags:
        log_warn(f"CandidateMCQ {mcq.q_id} has flags: {mcq.flags}")
    else:
        log_ok(f"Generated CandidateMCQ {mcq.q_id} from section {section_title!r}")
    return mcq


def register_candidate_mcq(
    state: "CooperativeState",
    mcq: CandidateMCQ,
    blueprint: MCQBlueprint,
) -> None:
    """Store CandidateMCQ, mirror to legacy candidate_questions, mark blueprint generated."""
    state.mcq_blueprints[blueprint.blueprint_id] = blueprint
    state.store_candidate_mcq(mcq)
    legacy = candidate_mcq_to_candidate_question(mcq, blueprint=blueprint)
    if mcq.q_id not in state.candidate_questions:
        state.add_candidate_question(legacy)
    else:
        state.candidate_questions[mcq.q_id] = legacy
    blueprint.status = "generated"
