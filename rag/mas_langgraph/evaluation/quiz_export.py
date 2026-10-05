"""Optional quiz-text export for LectureBank AXIS evaluation.

Reads final explained MCQs from the existing in-memory experiment payload.
Does not regenerate questions or change AXIS/MAS behavior.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mas_langgraph.batch.io import atomic_write_csv, atomic_write_text
from mas_langgraph.evaluation.axis_metrics import artifact_ref_key
from mas_langgraph.evaluation.pdf_source import document_id_for

OPTION_LABELS = ("A", "B", "C", "D")

QUESTION_FIELDS = [
    "lecturebank_id",
    "document_id",
    "question_index",
    "candidate_ref",
    "stem",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "correct_option",
    "axis_interrogation_mode",
    "question_version",
]

_REJECT_DECISIONS = frozenset(
    {"reject", "rejected", "needs_revision", "revise", "draft", "failed"}
)


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _ref_parts(ref: Any) -> Tuple[str, str, Optional[int]]:
    """Return (candidate_ref_key, logical_id, version)."""
    key = artifact_ref_key(ref)
    logical = ""
    version: Optional[int] = None
    if isinstance(ref, Mapping):
        logical = str(ref.get("logical_id") or "")
        raw_ver = ref.get("version")
        if raw_ver not in (None, ""):
            try:
                version = int(raw_ver)
            except (TypeError, ValueError):
                version = None
    if not logical and key:
        if ":v" in str(key):
            logical, _, ver_s = str(key).rpartition(":v")
            try:
                version = int(ver_s) if version is None else version
            except (TypeError, ValueError):
                pass
        else:
            logical = str(key)
    return str(key or ""), str(logical or key or ""), version


def _parse_options(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        if isinstance(item, Mapping):
            out.append(
                {
                    "option_id": str(item.get("option_id") or "").strip(),
                    "text": str(item.get("text") or ""),
                    "is_correct": bool(item.get("is_correct")),
                }
            )
        else:
            text = str(getattr(item, "text", None) or item or "")
            oid = str(getattr(item, "option_id", "") or "").strip()
            is_correct = bool(getattr(item, "is_correct", False))
            out.append({"option_id": oid, "text": text, "is_correct": is_correct})
    return out


def option_slots(options: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map display labels A–D onto option records without inventing text."""
    slots: Dict[str, Dict[str, Any]] = {}
    unlabeled: List[Dict[str, Any]] = []
    for opt in options:
        oid = str(opt.get("option_id") or "").strip().upper()
        if oid in OPTION_LABELS and oid not in slots:
            slots[oid] = dict(opt)
        else:
            unlabeled.append(dict(opt))
    for opt in unlabeled:
        for label in OPTION_LABELS:
            if label not in slots:
                slots[label] = opt
                break
    return slots


def normalize_correct_option(
    options: Sequence[Mapping[str, Any]],
    correct_value: Any = None,
) -> str:
    """Map stored correct-answer identity onto a display label A–D.

    Uses ``is_correct`` / ``correct_option_id`` / 0-based index when that is
    the actual representation. Does not invent a letter if none can be derived.
    """
    slots = option_slots(options)
    label_for_opt: Dict[int, str] = {}
    for label, opt in slots.items():
        for i, original in enumerate(options):
            if original is opt or (
                original.get("text") == opt.get("text")
                and original.get("option_id") == opt.get("option_id")
                and i not in label_for_opt
            ):
                label_for_opt[i] = label
                break

    marked = [i for i, opt in enumerate(options) if opt.get("is_correct")]
    if len(marked) == 1:
        return label_for_opt.get(marked[0], "")

    if correct_value in (None, ""):
        return ""

    if isinstance(correct_value, bool):
        return ""
    if isinstance(correct_value, int) and not isinstance(correct_value, bool):
        if 0 <= correct_value < len(OPTION_LABELS):
            return OPTION_LABELS[correct_value]
        return ""

    text = str(correct_value).strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in OPTION_LABELS:
        return upper
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(OPTION_LABELS):
            return OPTION_LABELS[idx]
        return ""

    for i, opt in enumerate(options):
        oid = str(opt.get("option_id") or "").strip()
        if oid and (oid == text or oid.upper() == upper):
            if oid.upper() in OPTION_LABELS:
                return oid.upper()
            return label_for_opt.get(i, "")
        if str(opt.get("text") or "") == text:
            return label_for_opt.get(i, "")
    return ""


def _is_exportable_case(case: Mapping[str, Any]) -> bool:
    decision = str(case.get("factual_decision") or "").strip().lower()
    if decision in _REJECT_DECISIONS:
        return False
    question = case.get("question")
    if isinstance(question, Mapping):
        stem = str(question.get("stem") or question.get("question_text") or "")
        options = question.get("options")
        if stem.strip() and options:
            return True
    stem = str(case.get("question_stem") or "")
    options = case.get("answer_options") or case.get("options")
    return bool(stem.strip() and options)


def extract_quiz_questions(
    experiment_payload: Optional[Mapping[str, Any]],
    *,
    lecturebank_id: str,
    axis_interrogation_mode: str = "constrained",
    document_id: str = "",
) -> List[Dict[str, Any]]:
    """Extract final-quiz MCQ rows from an in-memory experiment payload.

    ``experiment.cases`` are already the explained final-QuizPack versions
    (historical / rejected candidates are not included by generation). Extra
    rejected or superseded versions in a payload are filtered here.
    """
    payload = _as_mapping(experiment_payload)
    cases = list(payload.get("cases") or [])
    mode = str(
        axis_interrogation_mode
        or payload.get("axis_interrogation_mode")
        or "constrained"
    )
    doc_id = str(document_id or document_id_for(str(lecturebank_id)))
    best_by_logical: "OrderedDict[str, Tuple[int, Dict[str, Any]]]" = OrderedDict()
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        if not _is_exportable_case(case):
            continue
        key, logical, version = _ref_parts(case.get("candidate_ref"))
        if not key:
            continue
        question = case.get("question") if isinstance(case.get("question"), Mapping) else {}
        stem = str(
            (question or {}).get("stem")
            or (question or {}).get("question_text")
            or case.get("question_stem")
            or ""
        )
        options = _parse_options(
            (question or {}).get("options")
            or case.get("answer_options")
            or case.get("options")
        )
        if not stem.strip() or not options:
            continue
        slots = option_slots(options)
        correct_raw = (
            (question or {}).get("correct_option")
            if "correct_option" in (question or {})
            else (question or {}).get("correct_option_id")
        )
        if correct_raw in (None, "") and "correct_option_id" in case:
            correct_raw = case.get("correct_option_id")
        correct = normalize_correct_option(options, correct_raw)
        ver = version
        if ver is None and ":v" in key:
            try:
                ver = int(key.rsplit(":v", 1)[-1])
            except ValueError:
                ver = None
        row = {
            "lecturebank_id": str(lecturebank_id),
            "document_id": doc_id,
            "candidate_ref": key,
            "stem": stem,
            "option_A": str((slots.get("A") or {}).get("text") or ""),
            "option_B": str((slots.get("B") or {}).get("text") or ""),
            "option_C": str((slots.get("C") or {}).get("text") or ""),
            "option_D": str((slots.get("D") or {}).get("text") or ""),
            "correct_option": correct,
            "axis_interrogation_mode": mode,
            "question_version": ver if ver is not None else "",
            "_logical_id": logical,
        }
        rank = int(ver) if ver is not None else -1
        prev = best_by_logical.get(logical)
        if prev is None or rank >= prev[0]:
            best_by_logical[logical] = (rank, row)

    rows: List[Dict[str, Any]] = []
    for index, (_logical, (_rank, row)) in enumerate(best_by_logical.items(), start=1):
        out = dict(row)
        out.pop("_logical_id", None)
        out["question_index"] = index
        rows.append(out)
    return rows


def dedupe_quiz_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one row per (lecturebank_id, candidate_ref), preserving order."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        lid = str(row.get("lecturebank_id") or "")
        ref = str(row.get("candidate_ref") or "")
        key = (lid, ref)
        if not lid or not ref or key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def collect_quiz_rows(collected: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in collected:
        rows.extend(list(item.get("quiz_questions") or []))
    # Re-number per document after merge/dedupe.
    deduped = dedupe_quiz_rows(rows)
    by_doc: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for row in deduped:
        by_doc.setdefault(str(row.get("lecturebank_id") or ""), []).append(row)
    numbered: List[Dict[str, Any]] = []
    for lid, group in by_doc.items():
        for i, row in enumerate(group, start=1):
            item = dict(row)
            item["lecturebank_id"] = lid
            item["question_index"] = i
            numbered.append(item)
    return numbered


def _format_document_quiz_markdown(
    lecturebank_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    heading: str,
) -> str:
    mode = ""
    for row in rows:
        mode = str(row.get("axis_interrogation_mode") or "").strip()
        if mode:
            break
    lines: List[str] = [heading, ""]
    if mode:
        lines.append(f"AXIS interrogation mode: {mode}")
        lines.append("")
    for row in rows:
        idx = row.get("question_index") or ""
        lines.append(f"### Question {idx}")
        lines.append("")
        lines.append(str(row.get("stem") or ""))
        lines.append("")
        for label in OPTION_LABELS:
            text = str(row.get(f"option_{label}") or "")
            lines.append(f"{label}. {text}")
        lines.append("")
        correct = str(row.get("correct_option") or "").strip()
        lines.append(f"Correct answer: {correct}" if correct else "Correct answer:")
        lines.append("")
        lines.append(f"Candidate ref: {row.get('candidate_ref') or ''}")
        lines.append("")
        lines.append("---")
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    if lines and lines[-1] == "---":
        lines.pop()
        if lines and lines[-1] == "":
            lines.pop()
    lines.append("")
    return "\n".join(lines)


def render_quizzes_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    unavailable_ids: Sequence[str] = (),
) -> str:
    grouped: "OrderedDict[str, List[Mapping[str, Any]]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row.get("lecturebank_id") or ""), []).append(row)
    parts = ["# LectureBank Quiz Export", ""]
    missing = [str(x) for x in unavailable_ids if str(x)]
    if missing:
        parts.append(
            "Quiz text was not persisted for previously completed documents: "
            + ", ".join(missing)
            + ". Re-run those documents with `--overwrite` (or without `--resume`) "
            "to export questions. Text is not reconstructed from aggregate metrics."
        )
        parts.append("")
    if not grouped:
        if not missing:
            parts.append("No final quiz questions were persisted.")
            parts.append("")
        return "\n".join(parts)
    for lid, group in grouped.items():
        parts.append(_format_document_quiz_markdown(lid, group, heading=f"## Document {lid}"))
        parts.append("")
    text = "\n".join(parts).rstrip() + "\n"
    return text


def write_quiz_outputs(
    output_dir: Path,
    metrics_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    unavailable_ids: Sequence[str] = (),
) -> None:
    """Atomically write questions.csv, quizzes.md, and per-document Markdown."""
    metrics_dir = Path(metrics_dir)
    output_dir = Path(output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = [{k: row.get(k, "") for k in QUESTION_FIELDS} for row in rows]
    atomic_write_csv(metrics_dir / "questions.csv", csv_rows, fieldnames=QUESTION_FIELDS)
    atomic_write_text(
        output_dir / "quizzes.md",
        render_quizzes_markdown(rows, unavailable_ids=unavailable_ids),
    )
    quizzes_dir = output_dir / "quizzes"
    quizzes_dir.mkdir(parents=True, exist_ok=True)
    grouped: "OrderedDict[str, List[Mapping[str, Any]]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row.get("lecturebank_id") or ""), []).append(row)
    wanted = set()
    for lid, group in grouped.items():
        if not lid:
            continue
        wanted.add(f"{lid}.md")
        body = _format_document_quiz_markdown(
            lid, group, heading=f"# LectureBank Quiz Export — Document {lid}"
        )
        atomic_write_text(quizzes_dir / f"{lid}.md", body if body.endswith("\n") else body + "\n")
    for stale in quizzes_dir.glob("*.md"):
        if stale.name not in wanted:
            try:
                stale.unlink()
            except OSError:
                pass
