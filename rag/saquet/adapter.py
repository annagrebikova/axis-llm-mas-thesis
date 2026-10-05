"""Adapt MAS quiz_141.json records into SAQUET MultipleChoiceQuestion objects.

Does not modify question wording, options, or answers — only structural mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

from .saquet_core import MultipleChoiceQuestion

PathLike = Union[str, Path]


def load_quiz_141_pack(path: PathLike) -> Dict[str, Any]:
    """Load the first quiz pack from quiz_141.json (or a pack dict)."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError(f"No quiz packs in {p}")
        pack = data[0]
    elif isinstance(data, dict):
        pack = data
    else:
        raise TypeError(f"Unexpected quiz JSON type: {type(data)}")
    if "questions" not in pack:
        raise ValueError(f"Quiz pack missing 'questions': {p}")
    return pack


def extract_quiz_records(path: PathLike) -> List[Dict[str, Any]]:
    """Return lightweight records for each final MCQ (no SAQUET evaluation)."""
    pack = load_quiz_141_pack(path)
    records: List[Dict[str, Any]] = []
    for q in pack.get("questions") or []:
        if not isinstance(q, Mapping):
            continue
        q_id = str(q.get("q_id") or q.get("id") or "").strip()
        stem = str(q.get("question_text") or q.get("text") or q.get("stem") or "")
        correct = str(q.get("correct_answer") or q.get("answer") or "")
        options = list(q.get("options") or [])
        options = [str(o) for o in options]
        records.append(
            {
                "q_id": q_id,
                "question_text": stem,
                "correct_answer": correct,
                "options": options,
            }
        )
    return records


def record_to_multiple_choice_question(record: Mapping[str, Any]) -> MultipleChoiceQuestion:
    """Map one MAS quiz record to SAQUET MultipleChoiceQuestion."""
    q_id = str(record.get("q_id") or record.get("id") or "").strip()
    stem = str(record.get("question_text") or record.get("text") or record.get("stem") or "")
    correct = str(record.get("correct_answer") or record.get("answer") or "").strip()
    options = [str(o) for o in list(record.get("options") or [])]
    if not stem:
        raise ValueError(f"Empty stem for q_id={q_id!r}")
    if not options:
        raise ValueError(f"No options for q_id={q_id!r}")
    if correct not in options:
        raise ValueError(
            f"correct_answer not found in options for q_id={q_id!r}: "
            f"{correct!r} not in {options!r}"
        )
    return MultipleChoiceQuestion(
        stem=stem,
        options=options,
        correct_option=correct,
        qid=q_id,
        quality=0,
    )


def quiz_path_to_saquet_questions(path: PathLike) -> List[MultipleChoiceQuestion]:
    """Load quiz_141.json and convert all questions to SAQUET objects."""
    return [record_to_multiple_choice_question(r) for r in extract_quiz_records(path)]


def validate_records(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return human-readable validation errors (empty if OK)."""
    errors: List[str] = []
    for r in records:
        q_id = r.get("q_id")
        opts = list(r.get("options") or [])
        correct = str(r.get("correct_answer") or "")
        if not q_id:
            errors.append("missing q_id")
        if not r.get("question_text"):
            errors.append(f"{q_id}: missing question_text")
        if not opts:
            errors.append(f"{q_id}: missing options")
        if correct not in opts:
            errors.append(f"{q_id}: correct_answer not in options")
    return errors
