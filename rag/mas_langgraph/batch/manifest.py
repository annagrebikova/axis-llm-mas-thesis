"""Batch manifest: portable mapping from source PDFs to isolated indexes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ids import presentation_dir_name
from .io import atomic_write_csv, atomic_write_json, load_json
from mas_langgraph.axis.interrogation_mode import (
    DEFAULT_AXIS_INTERROGATION_MODE,
    DEFAULT_AXIS_INTERROGATION_POLICY,
    normalize_axis_interrogation_mode,
    normalize_axis_interrogation_policy,
)
from mas_langgraph.domains.quiz.question_limit import QUESTION_LIMIT_SEMANTICS

MANIFEST_NAME = "manifest.json"
SUMMARY_NAME = "batch_summary.csv"
MANIFEST_VERSION = 1

SUMMARY_FIELDS = (
    "document_id",
    "relative_path",
    "rag_status",
    "generate_status",
    "num_pages",
    "nonempty_pages",
    "chunk_count",
    "requires_ocr",
    "rag_duration_s",
    "generate_duration_s",
    "max_questions",
    "requested_max_questions",
    "final_accepted_questions",
    "question_cap_reached",
    "generation_exhausted",
    "max_explain_questions",
    "explained_questions",
    "explained_question_count",
    "warnings",
    "errors",
    "results_md",
    "run_id",
)


def empty_manifest(*, input_dir: str, output_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "config": dict(config or {}),
        "presentations": [],
        "unsupported": [],
    }


def presentation_output_dir(output_dir: Path, document_id: str) -> Path:
    return Path(output_dir) / presentation_dir_name(document_id)


def new_presentation_record(
    *,
    relative_path: str,
    source_hash: str,
    document_id: str,
    collection_name: str,
    index_dir: str,
    source_md: str,
    rag_metadata: str,
) -> Dict[str, Any]:
    return {
        "relative_path": relative_path,
        "source_hash": source_hash,
        "document_id": document_id,
        "collection_name": collection_name,
        "index_dir": index_dir,
        "source_md": source_md,
        "rag_metadata": rag_metadata,
        "results_md": "",
        "experiment_json": "",
        "timing_csv": "",
        "embedding_model": "",
        "chunk_size": None,
        "chunk_overlap": None,
        "chunk_by_page": True,
        "num_pages": 0,
        "nonempty_pages": 0,
        "empty_pages": [],
        "chunk_count": 0,
        "requires_ocr": False,
        "rag_status": "pending",
        "generate_status": "not_started",
        "rag_duration_s": None,
        "extract_duration_s": None,
        "chunk_duration_s": None,
        "index_duration_s": None,
        "factual_workflow_s": None,
        "explanation_duration_s": None,
        "generate_duration_s": None,
        "run_id": "",
        "generation_config": {},
        "warnings": [],
        "errors": [],
        "updated_at": _now(),
    }


def find_record(manifest: Dict[str, Any], document_id: str) -> Optional[Dict[str, Any]]:
    for row in manifest.get("presentations") or []:
        if str(row.get("document_id")) == str(document_id):
            return row
    return None


def upsert_record(manifest: Dict[str, Any], record: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = list(manifest.get("presentations") or [])
    doc_id = str(record.get("document_id"))
    replaced = False
    for i, row in enumerate(rows):
        if str(row.get("document_id")) == doc_id:
            rows[i] = record
            replaced = True
            break
    if not replaced:
        rows.append(record)
    rows.sort(key=lambda r: str(r.get("relative_path") or "").lower())
    manifest["presentations"] = rows
    manifest["updated_at"] = _now()


def save_manifest(path: Path, manifest: Dict[str, Any]) -> Path:
    manifest = dict(manifest)
    manifest["updated_at"] = _now()
    return atomic_write_json(path, manifest)


def load_manifest(path: Path) -> Dict[str, Any]:
    return load_json(path)


def write_summary_csv(path: Path, manifest: Dict[str, Any]) -> Path:
    rows = []
    for rec in manifest.get("presentations") or []:
        rows.append(
            {
                "document_id": rec.get("document_id"),
                "relative_path": rec.get("relative_path"),
                "rag_status": rec.get("rag_status"),
                "generate_status": rec.get("generate_status"),
                "num_pages": rec.get("num_pages"),
                "nonempty_pages": rec.get("nonempty_pages"),
                "chunk_count": rec.get("chunk_count"),
                "requires_ocr": rec.get("requires_ocr"),
                "rag_duration_s": rec.get("rag_duration_s"),
                "generate_duration_s": rec.get("generate_duration_s"),
                "max_questions": rec.get("max_questions"),
                "requested_max_questions": rec.get(
                    "requested_max_questions", rec.get("max_questions")
                ),
                "final_accepted_questions": rec.get("final_accepted_questions"),
                "question_cap_reached": rec.get("question_cap_reached"),
                "generation_exhausted": rec.get("generation_exhausted"),
                "max_explain_questions": rec.get("max_explain_questions"),
                "explained_questions": rec.get("explained_questions"),
                "explained_question_count": rec.get(
                    "explained_question_count", rec.get("explained_questions")
                ),
                "warnings": "; ".join(rec.get("warnings") or []),
                "errors": "; ".join(rec.get("errors") or []),
                "results_md": rec.get("results_md"),
                "run_id": rec.get("run_id"),
            }
        )
    return atomic_write_csv(path, rows, fieldnames=SUMMARY_FIELDS)


def rag_config_fingerprint(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "embedding_model": config.get("embedding_model"),
        "chunk_size": config.get("chunk_size"),
        "chunk_overlap": config.get("chunk_overlap"),
        "chunk_by_page": config.get("chunk_by_page", True),
    }


def _hybrid_add_from_source_fingerprint(config: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool(config.get("hybrid_add_from_source"))
    if not enabled:
        return {
            "hybrid_add_from_source": False,
            "hybrid_add_from_source_policy": "",
            "hybrid_add_source_limit": None,
            "hybrid_add_source_k_per_query": None,
            "hybrid_add_source_max_search_hits": None,
            "hybrid_add_source_mmr_lambda": None,
            "hybrid_add_source_selection_rank_indexing": None,
        }
    from mas_langgraph.axis.hybrid_add_source import (
        DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY,
        DEFAULT_HYBRID_ADD_SOURCE_LIMIT,
        DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS,
        DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
        HYBRID_ADD_FROM_SOURCE_POLICY,
        SELECTION_RANK_INDEXING,
    )

    raw_limit = config.get("hybrid_add_source_limit")
    limit = (
        int(raw_limit)
        if raw_limit not in (None, "")
        else int(DEFAULT_HYBRID_ADD_SOURCE_LIMIT)
    )
    raw_k = config.get("hybrid_add_source_k_per_query")
    k_per_query = (
        int(raw_k)
        if raw_k not in (None, "")
        else int(DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY)
    )
    raw_hits = config.get("hybrid_add_source_max_search_hits")
    max_hits = (
        int(raw_hits)
        if raw_hits not in (None, "")
        else int(DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS)
    )
    raw_lambda = config.get("hybrid_add_source_mmr_lambda")
    mmr_lambda = (
        float(raw_lambda)
        if raw_lambda not in (None, "")
        else float(DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA)
    )
    return {
        "hybrid_add_from_source": True,
        "hybrid_add_from_source_policy": HYBRID_ADD_FROM_SOURCE_POLICY,
        "hybrid_add_source_limit": limit,
        "hybrid_add_source_k_per_query": k_per_query,
        "hybrid_add_source_max_search_hits": max_hits,
        "hybrid_add_source_mmr_lambda": mmr_lambda,
        "hybrid_add_source_selection_rank_indexing": SELECTION_RANK_INDEXING,
    }


def generation_config_fingerprint(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "llm_model": config.get("llm_model"),
        "model_type": config.get("model_type"),
        "temperature": config.get("temperature"),
        "seed": config.get("seed"),
        "max_rounds": config.get("max_rounds"),
        "max_questions": config.get("max_questions"),
        "question_limit_semantics": QUESTION_LIMIT_SEMANTICS,
        "max_explain_questions": config.get("max_explain_questions"),
        "orchestrator": config.get("orchestrator", "langgraph"),
        "question": config.get("question"),
        "axis_interrogation_mode": normalize_axis_interrogation_mode(
            config.get("axis_interrogation_mode", DEFAULT_AXIS_INTERROGATION_MODE)
        ),
        "axis_interrogation_policy": normalize_axis_interrogation_policy(
            config.get("axis_interrogation_policy", DEFAULT_AXIS_INTERROGATION_POLICY)
        ),
        **_hybrid_add_from_source_fingerprint(config),
    }


def generation_fingerprints_compatible(
    stored: Optional[Dict[str, Any]],
    current: Optional[Dict[str, Any]],
) -> bool:
    """Treat missing hybrid-ADD-from-source keys as the default-off setting."""

    def _norm(fp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(fp or {})
        if "hybrid_add_from_source" not in out:
            out["hybrid_add_from_source"] = False
            out["hybrid_add_from_source_policy"] = ""
            out["hybrid_add_source_limit"] = None
            out["hybrid_add_source_k_per_query"] = None
            out["hybrid_add_source_max_search_hits"] = None
            out["hybrid_add_source_mmr_lambda"] = None
            out["hybrid_add_source_selection_rank_indexing"] = None
        elif not out.get("hybrid_add_from_source"):
            out["hybrid_add_from_source"] = False
            out["hybrid_add_from_source_policy"] = ""
            out["hybrid_add_source_limit"] = None
            out["hybrid_add_source_k_per_query"] = None
            out["hybrid_add_source_max_search_hits"] = None
            out["hybrid_add_source_mmr_lambda"] = None
            out["hybrid_add_source_selection_rank_indexing"] = None
        return out

    return _norm(stored) == _norm(current)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
