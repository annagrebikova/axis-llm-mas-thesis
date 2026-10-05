"""RAG-only preparation: extract, chunk, embed, index one presentation."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rag_pipeline import RAGPipeline

from .ids import file_sha256, posix_relative
from .io import atomic_write_json, atomic_write_text
from .manifest import rag_config_fingerprint


class RequiresOCRError(RuntimeError):
    """Extraction produced no usable text; generation must not proceed."""


def write_source_markdown(
    *,
    path: Path,
    source_file: str,
    document_id: str,
    pages_text: list,
    empty_pages: list,
    requires_ocr: bool,
) -> Path:
    """Inspectable page-organized extraction. Not an LLM summary."""
    nonempty = sum(1 for p in pages_text if str(p.get("text") or "").strip())
    lines = [
        f"# Source extraction: {source_file}",
        "",
        f"- document_id: `{document_id}`",
        f"- pages: {len(pages_text)}",
        f"- nonempty_pages: {nonempty}",
        f"- empty_pages: {empty_pages}",
        f"- requires_ocr: {str(bool(requires_ocr)).lower()}",
        "",
        "This file is the extracted PDF text by original page number.",
        "Headings below are wrappers only and are not indexed as evidence.",
        "",
    ]
    for page in pages_text:
        num = page.get("page")
        text = str(page.get("text") or "")
        lines.append(f"## Page {num}")
        lines.append("")
        if text.strip():
            lines.append(text.rstrip())
        else:
            lines.append("_empty_")
        lines.append("")
    return atomic_write_text(path, "\n".join(lines) + "\n")


def prepare_presentation(
    *,
    pdf_path: Optional[Path] = None,
    input_dir: Path,
    output_dir: Path,
    document_id: str,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_by_page: bool = True,
    embeddings: Optional[object] = None,
    pipeline: Optional[RAGPipeline] = None,
    overwrite: bool = False,
    resume: bool = False,
    existing_record: Optional[Dict[str, Any]] = None,
    pdf_data: Optional[Dict[str, Any]] = None,
    source_hash: Optional[str] = None,
    input_source_type: str = "",
    jsonl_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Prepare one presentation into an isolated vector index.

    Provide either ``pdf_path`` (extract with pdfplumber) or preloaded
    ``pdf_data`` from ``load_selected_extractions`` / ``load_pdf_from_jsonl``.
    Does not run MCQ generation or AXIS.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    using_jsonl = pdf_data is not None
    if using_jsonl:
        if not source_hash:
            raise ValueError("source_hash is required when pdf_data is provided")
        jsonl_resolved = Path(jsonl_path).resolve() if jsonl_path is not None else None
        rel = jsonl_resolved.name if jsonl_resolved is not None else "extraction.jsonl"
        source_hash_value = str(source_hash)
        source_file_name = str(pdf_data.get("source_file") or rel)
        input_source_type = input_source_type or "extraction_jsonl"
    else:
        if pdf_path is None:
            raise ValueError("prepare_presentation requires pdf_path or pdf_data")
        pdf_path = Path(pdf_path).resolve()
        rel = posix_relative(pdf_path, input_dir)
        source_hash_value = file_sha256(pdf_path)
        source_file_name = pdf_path.name
        jsonl_resolved = None
        input_source_type = input_source_type or "pdf"

    pres_dir = output_dir / f"presentation_{document_id}"
    index_dir = pres_dir / "vector_store"
    source_md = pres_dir / "source.md"
    rag_meta_path = pres_dir / "rag_metadata.json"
    cfg_fp = rag_config_fingerprint(
        {
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_by_page": chunk_by_page,
        }
    )

    if (
        resume
        and not overwrite
        and existing_record
        and _rag_resume_ok(
            existing_record,
            source_hash=source_hash_value,
            config=cfg_fp,
            index_dir=index_dir,
            source_md=source_md,
            rag_meta_path=rag_meta_path,
        )
    ):
        rec = dict(existing_record)
        rec["warnings"] = list(rec.get("warnings") or []) + ["rag_resumed"]
        rec["rag_status"] = rec.get("rag_status") or "success"
        return rec

    t0 = time.perf_counter()
    pipe = pipeline or RAGPipeline(
        vector_store_path=str(index_dir),
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings=embeddings,
    )
    t_extract = time.perf_counter()
    if using_jsonl:
        loaded = dict(pdf_data or {})
        loaded["id"] = str(document_id)
        loaded["source_file"] = source_file_name
        extract_s = time.perf_counter() - t_extract
    else:
        loaded = pipe.load_pdf_from_file(str(pdf_path), document_id=document_id)
        extract_s = time.perf_counter() - t_extract
    stats = dict(loaded.get("statistics") or {})
    pages_text = list((loaded.get("text") or {}).get("pages_text") or [])
    empty_pages = list(stats.get("empty_pages") or [])
    requires_ocr = bool(stats.get("requires_ocr"))
    pres_dir.mkdir(parents=True, exist_ok=True)
    write_source_markdown(
        path=source_md,
        source_file=source_file_name,
        document_id=document_id,
        pages_text=pages_text,
        empty_pages=empty_pages,
        requires_ocr=requires_ocr,
    )

    record: Dict[str, Any] = {
        "relative_path": rel,
        "source_hash": source_hash_value,
        "document_id": document_id,
        "collection_name": collection_name,
        "index_dir": _portable(index_dir, output_dir),
        "source_md": _portable(source_md, output_dir),
        "rag_metadata": _portable(rag_meta_path, output_dir),
        "results_md": (existing_record or {}).get("results_md") or "",
        "experiment_json": (existing_record or {}).get("experiment_json") or "",
        "timing_csv": (existing_record or {}).get("timing_csv") or "",
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_by_page": chunk_by_page,
        "num_pages": int(stats.get("num_pages") or 0),
        "nonempty_pages": int(stats.get("nonempty_pages") or 0),
        "empty_pages": empty_pages,
        "chunk_count": 0,
        "requires_ocr": requires_ocr,
        "extract_duration_s": round(extract_s, 6),
        "chunk_duration_s": None,
        "index_duration_s": None,
        "rag_duration_s": None,
        "rag_status": "pending",
        "generate_status": (existing_record or {}).get("generate_status") or "not_started",
        "warnings": [],
        "errors": [],
        "run_id": (existing_record or {}).get("run_id") or "",
        "generation_config": (existing_record or {}).get("generation_config") or {},
        "input_source_type": input_source_type,
        "input_source_path": str(jsonl_resolved) if jsonl_resolved is not None else (str(pdf_path) if pdf_path is not None else ""),
        "jsonl_path": str(jsonl_resolved) if jsonl_resolved is not None else "",
    }

    if requires_ocr:
        record["rag_status"] = "requires_ocr"
        record["errors"] = [
            "PDF extraction produced no usable text (empty or image-only). "
            "OCR is required before MCQ generation."
        ]
        record["rag_duration_s"] = round(time.perf_counter() - t0, 6)
        atomic_write_json(rag_meta_path, {**record, "config": cfg_fp})
        return record

    t_chunk = time.perf_counter()
    documents = pipe.create_chunks_with_metadata(loaded, chunk_by_page=chunk_by_page)
    chunk_s = time.perf_counter() - t_chunk
    record["chunk_duration_s"] = round(chunk_s, 6)
    record["chunk_count"] = len(documents)
    if not documents:
        record["rag_status"] = "requires_ocr"
        record["requires_ocr"] = True
        record["errors"] = ["No indexable chunks after extraction."]
        record["rag_duration_s"] = round(time.perf_counter() - t0, 6)
        atomic_write_json(rag_meta_path, {**record, "config": cfg_fp})
        return record

    t_index = time.perf_counter()
    if pipeline is not None:
        pipe.vector_store = None
        pipe._active_collection = None
    elif pipe.vector_store is not None:
        pipe.close()
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    pipe._bind_collection(
        persist_directory=index_dir,
        collection_name=collection_name,
        documents=documents,
        replace_collection=True,
    )
    index_s = time.perf_counter() - t_index
    pipe.assert_exclusive_document(document_id)

    verify = RAGPipeline(
        vector_store_path=str(index_dir),
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings=pipe.embeddings,
    )
    verify.load_existing_index(collection_name=collection_name)
    verify.assert_exclusive_document(document_id)
    vstats = verify.get_index_stats()
    if int(vstats.get("total_chunks") or 0) != len(documents):
        verify.close()
        raise RuntimeError(
            f"Reopened index chunk count {vstats.get('total_chunks')} "
            f"!= written {len(documents)} for {document_id}"
        )
    verify.close()
    if pipeline is None:
        pipe.close()
    else:
        pipe.vector_store = None
        pipe._active_collection = None

    record["index_duration_s"] = round(index_s, 6)
    record["rag_duration_s"] = round(time.perf_counter() - t0, 6)
    record["rag_status"] = "success"
    atomic_write_json(
        rag_meta_path,
        {
            **record,
            "config": cfg_fp,
            "index_stats": vstats,
        },
    )
    return record


def _index_exists(index_dir: Path) -> bool:
    return Path(index_dir).exists() and any(Path(index_dir).iterdir())


def _portable(path: Path, output_dir: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(output_dir).resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def _rag_resume_ok(
    record: Dict[str, Any],
    *,
    source_hash: str,
    config: Dict[str, Any],
    index_dir: Path,
    source_md: Path,
    rag_meta_path: Path,
) -> bool:
    if str(record.get("source_hash")) != str(source_hash):
        return False
    if record.get("rag_status") != "success":
        return False
    stored = rag_config_fingerprint(
        {
            "embedding_model": record.get("embedding_model"),
            "chunk_size": record.get("chunk_size"),
            "chunk_overlap": record.get("chunk_overlap"),
            "chunk_by_page": record.get("chunk_by_page", True),
        }
    )
    if stored != config:
        return False
    if not source_md.is_file() or not rag_meta_path.is_file():
        return False
    if not _index_exists(index_dir):
        return False
    return True
