"""Vector-store cleanup and metric-row lifecycle for LectureBank evaluation."""

from __future__ import annotations

import csv
import gc
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from mas_langgraph.batch.manifest import generation_fingerprints_compatible
from mas_langgraph.evaluation.pdf_source import cached_pdf_path, document_id_for
from mas_langgraph.evaluation.report import (
    ATTEMPT_FIELDS,
    COMPARISON_FIELDS,
    DIST_FIELDS,
    DOCUMENT_FIELDS,
    PROVENANCE_FIELDS,
    ROUND_FIELDS,
)

COMPLETED_EVAL_STATUSES = {"success", "axis_no_valid_replay"}


def presentation_dir(output_dir: Path, document_id: str) -> Path:
    return Path(output_dir) / f"presentation_{document_id}"


def vector_store_dir(output_dir: Path, document_id: str) -> Path:
    return presentation_dir(output_dir, document_id) / "vector_store"


def is_safe_vector_store_path(
    path: Path,
    *,
    document_id: str,
    allowed_roots: Sequence[Path],
) -> bool:
    """True only for ``.../presentation_<document_id>/vector_store`` under an allowed root."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    if resolved.name != "vector_store":
        return False
    parent = resolved.parent
    if parent.name != f"presentation_{document_id}":
        return False
    for root in allowed_roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return True
        except ValueError:
            continue
    return False


def delete_vector_store(
    path: Path,
    *,
    document_id: str,
    allowed_roots: Sequence[Path],
) -> bool:
    """Delete one document's vector store. Returns True if it was removed."""
    gc.collect()
    resolved = Path(path)
    if not is_safe_vector_store_path(resolved, document_id=document_id, allowed_roots=allowed_roots):
        raise ValueError(
            f"Refusing to delete {path}: not this document's vector store."
        )
    if not resolved.exists():
        return False
    shutil.rmtree(resolved, ignore_errors=True)
    return not resolved.exists() or not any(resolved.iterdir()) if resolved.exists() else True


def promote_vector_store(*, src_output_dir: Path, dest_output_dir: Path, document_id: str) -> Optional[Path]:
    src = vector_store_dir(src_output_dir, document_id)
    dest = vector_store_dir(dest_output_dir, document_id)
    if not src.exists() or not any(src.iterdir()):
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest)
    return dest


def drop_document_rows(
    collected: List[Dict[str, Any]], lecturebank_id: str
) -> List[Dict[str, Any]]:
    want = str(lecturebank_id)
    return [row for row in collected if str(row.get("lecturebank_id") or "") != want]


def _csv_cell(value: str) -> Any:
    if value in {"", "NA"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            rows.append({k: _csv_cell(v) for k, v in raw.items()})
        return rows


def _read_quiz_csv(path: Path) -> List[Dict[str, Any]]:
    """Read questions.csv without coercing stems/options (true/false/NA)."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_collected_from_metrics(metrics_dir: Path) -> List[Dict[str, Any]]:
    """Rebuild in-memory per-document metric bundles from aggregate CSVs."""
    metrics_dir = Path(metrics_dir)
    docs = _read_csv(metrics_dir / "document_metrics.csv")
    attempts = _read_csv(metrics_dir / "interrogation_attempts.csv")
    comparisons = _read_csv(metrics_dir / "replay_comparisons.csv")
    provenance = _read_csv(metrics_dir / "provenance_edges.csv")
    distributions = _read_csv(metrics_dir / "distributions.csv")
    rounds = _read_csv(metrics_dir / "round_metrics.csv")
    by_id: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        lid = str(doc.get("lecturebank_id") or "")
        if not lid:
            continue
        by_id[lid] = {
            "lecturebank_id": lid,
            "document_id": doc.get("document_id") or document_id_for(lid),
            "title": doc.get("title"),
            "topic_id": doc.get("topic_id"),
            "topic_name": doc.get("topic_name"),
            "instructor": doc.get("instructor"),
            "venue": doc.get("venue"),
            "year": doc.get("year"),
            "url": doc.get("url"),
            "pdf_source": doc.get("pdf_source"),
            "pdf_sha256": doc.get("pdf_sha256"),
            "source_hash": doc.get("pdf_sha256"),
            "failure_stage": doc.get("failure_stage") or "",
            "error_message": doc.get("error_message") or "",
            "rag_status": doc.get("rag_status"),
            "mas_status": doc.get("mas_status"),
            "axis_status": doc.get("axis_status"),
            "eval_status": doc.get("eval_status"),
            "accepted_question_count": doc.get("accepted_question_count") or 0,
            "explained_question_count": doc.get("explained_question_count") or 0,
            "valid_replay_comparisons": doc.get("valid_replay_comparisons") or 0,
            "document_metrics": dict(doc),
            "interrogation_attempts": [],
            "replay_comparisons": [],
            "provenance_edges": [],
            "distributions": [],
            "round_metrics": [],
            "quiz_questions": [],
            "reference_interventions": [],
            "effect_discovery_by_question": [],
            "concept_change_assessments": [],
            "concept_change_by_question": [],
            "outcome_criteria_changes": [],
            "saquet_by_mas_rows": [],
            "hybrid_add_candidates": [],
            "hybrid_add_boundaries": [],
            "hybrid_add_source_events": [],
            "metrics_persisted": True,
        }

    def _add(rows: Iterable[Mapping[str, Any]], key: str) -> None:
        for row in rows:
            lid = str(row.get("lecturebank_id") or "")
            if lid in {"", "ALL", "overall"}:
                continue
            if str(row.get("scope") or "") == "overall":
                continue
            if lid not in by_id:
                continue
            by_id[lid][key].append(dict(row))

    questions = _read_quiz_csv(metrics_dir / "questions.csv")
    reference_interventions = _read_csv(metrics_dir / "reference_interventions.csv")
    effect_discovery = _read_csv(metrics_dir / "effect_discovery_by_question.csv")
    concept_change_assessments = _read_csv(metrics_dir / "concept_change_assessments.csv")
    concept_change_questions = _read_csv(metrics_dir / "concept_change_by_question.csv")
    outcome_criteria_changes = _read_csv(metrics_dir / "outcome_criteria_changes.csv")
    if outcome_criteria_changes:
        from mas_langgraph.evaluation.outcome_criteria import hydrate_outcome_criteria_row

        outcome_criteria_changes = [
            hydrate_outcome_criteria_row(row) for row in outcome_criteria_changes
        ]
    _add(attempts, "interrogation_attempts")
    _add(comparisons, "replay_comparisons")
    _add(provenance, "provenance_edges")
    _add(distributions, "distributions")
    _add(rounds, "round_metrics")
    _add(questions, "quiz_questions")
    _add(reference_interventions, "reference_interventions")
    _add(effect_discovery, "effect_discovery_by_question")
    _add(concept_change_assessments, "concept_change_assessments")
    _add(concept_change_questions, "concept_change_by_question")
    _add(outcome_criteria_changes, "outcome_criteria_changes")
    _add(_read_csv(metrics_dir / "saquet_by_mas.csv"), "saquet_by_mas_rows")
    _add(_read_csv(metrics_dir / "hybrid_add_candidates.csv"), "hybrid_add_candidates")
    _add(_read_csv(metrics_dir / "hybrid_add_boundaries.csv"), "hybrid_add_boundaries")
    _add(_read_csv(metrics_dir / "hybrid_add_source_events.csv"), "hybrid_add_source_events")
    for item in by_id.values():
        qrows = item.get("effect_discovery_by_question") or []
        if qrows and all(r.get("reference_set_complete") is True for r in qrows):
            item["reference_sweep_status"] = "complete"
        elif qrows or item.get("reference_interventions"):
            item["reference_sweep_status"] = "incomplete"
    _ = (DOCUMENT_FIELDS, ATTEMPT_FIELDS, COMPARISON_FIELDS, DIST_FIELDS, PROVENANCE_FIELDS, ROUND_FIELDS)
    return list(by_id.values())


def document_has_metric_rows(collected: Sequence[Mapping[str, Any]], lecturebank_id: str) -> bool:
    lid = str(lecturebank_id)
    for row in collected:
        if str(row.get("lecturebank_id")) != lid:
            continue
        if row.get("document_metrics") or row.get("eval_status"):
            return True
    return False


def eval_completed(status: Any) -> bool:
    return str(status or "") in COMPLETED_EVAL_STATUSES


def metrics_resume_ok(
    *,
    lecturebank_id: str,
    source_hash: str,
    rag_fp: Mapping[str, Any],
    gen_fp: Mapping[str, Any],
    stored: Optional[Mapping[str, Any]],
    collected: Sequence[Mapping[str, Any]],
    require_vector_store: bool = False,
    index_dir: Optional[Path] = None,
) -> bool:
    """Skip a completed document even if its vector store was deleted."""
    if not stored:
        return False
    ident = str(stored.get("lecturebank_id") or "")
    doc_id = str(stored.get("document_id") or "")
    if ident and ident != str(lecturebank_id) and doc_id != document_id_for(lecturebank_id):
        return False
    if not eval_completed(stored.get("eval_status")):
        return False
    if stored.get("completed") is False:
        return False
    stored_hash = str(stored.get("source_hash") or stored.get("pdf_sha256") or "")
    if stored_hash and stored_hash != str(source_hash):
        return False
    stored_rag = dict(stored.get("rag_config") or {})
    stored_gen = dict(stored.get("generation_config") or {})
    if stored_rag and stored_rag != dict(rag_fp):
        return False
    if stored_gen and not generation_fingerprints_compatible(stored_gen, dict(gen_fp)):
        return False
    if not stored.get("metrics_persisted", True):
        return False
    if not document_has_metric_rows(collected, lecturebank_id):
        return False
    if require_vector_store:
        if index_dir is None or not Path(index_dir).exists() or not any(Path(index_dir).iterdir()):
            return False
    return True


def maybe_delete_downloaded_pdf(
    *,
    pdf_info: Mapping[str, Any],
    cache_dir: Path,
    pdf_dir: Optional[Path],
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    if str(pdf_info.get("source") or "") not in {"download", "cache"}:
        return False
    path = Path(pdf_info.get("path") or "")
    if not path.is_file():
        return False
    try:
        path.resolve().relative_to(Path(cache_dir).resolve())
    except ValueError:
        return False
    if pdf_dir is not None:
        try:
            path.resolve().relative_to(Path(pdf_dir).resolve())
            return False
        except ValueError:
            pass
    expected = cached_pdf_path(cache_dir, str(pdf_info.get("lecturebank_id") or path.stem))
    if path.resolve() != expected.resolve() and path.name != expected.name:
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False
