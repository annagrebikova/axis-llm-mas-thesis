"""Folder-based presentation batch: RAG prepare, then optional MCQ/AXIS generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from mas_langgraph.input.rag_config import DEFAULT_WORKFLOW_MODEL

from .discover import discover_presentations, format_unsupported_report
from .generate import generate_presentation
from .ids import collection_safe_id, file_sha256, posix_relative
from .manifest import (
    MANIFEST_NAME,
    SUMMARY_NAME,
    empty_manifest,
    find_record,
    load_manifest,
    save_manifest,
    upsert_record,
    write_summary_csv,
)
from .prepare import prepare_presentation
from .select import SelectionError, catalog_presentations, format_pdf_list, resolve_pdf_selection


DEFAULTS = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "chunk_by_page": True,
    "llm_model": DEFAULT_WORKFLOW_MODEL,
    "model_type": "openai",
    "temperature": 0.0,
    "seed": 42,
    "max_rounds": 5,
    "max_questions": 10,
    "max_explain_questions": None,
    "question": "Why was this question selected?",
    "orchestrator": "langgraph",
    "axis_interrogation_mode": "constrained",
}

DEFAULT_MAX_QUESTIONS = 10


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process a folder of PDF presentations independently: "
            "index each one, then optionally generate MCQs and AXIS explanations. "
            "PowerPoint files are not read as PDFs; export them to PDF first. "
            "With --recursive, PDFs in subfolders are included for listing, "
            "selection, and processing."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Folder of presentation PDFs.")
    parser.add_argument(
        "--output-dir",
        default="presentation_outputs",
        help="Batch output folder (manifest, per-presentation indexes and results).",
    )
    parser.add_argument(
        "--stage",
        choices=["rag", "generate", "all"],
        default="rag",
        help="rag = index only (default); generate = MCQ+AXIS from prepared indexes; all = both.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include PDFs in subfolders for listing, selection, and processing.",
    )
    parser.add_argument(
        "--list-pdfs",
        action="store_true",
        help=(
            "List selectable PDFs (relative path, document ID, RAG/generation status) "
            "and exit. Does not load models, index, generate, or write outputs."
        ),
    )
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        dest="pdf_selectors",
        help=(
            "Select a PDF by exact relative path, or by filename if unique. "
            "Repeatable. Example: --pdf \"lecture_01.pdf\""
        ),
    )
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        dest="document_ids",
        help="Select a PDF by stable document ID from --list-pdfs. Repeatable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip presentations whose hash, config, and artifacts still match.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this batch's outputs for the selected stage/presentation only.",
    )
    parser.add_argument("--embedding-model", default=DEFAULTS["embedding_model"])
    parser.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"])
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULTS["chunk_overlap"])
    parser.add_argument(
        "--no-page-chunking",
        action="store_true",
        help="Chunk full text instead of page-based chunks.",
    )
    parser.add_argument("--llm-model", default=DEFAULTS["llm_model"])
    parser.add_argument("--model-type", choices=["openai", "huggingface"], default="openai")
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--max-rounds", type=int, default=DEFAULTS["max_rounds"])
    parser.add_argument(
        "--max-questions",
        type=_positive_int,
        default=DEFAULT_MAX_QUESTIONS,
        help=(
            "Maximum target number of final accepted MCQs generated per presentation. "
            "The workflow may return fewer if valid evidence-supported questions are "
            "exhausted."
        ),
    )
    parser.add_argument(
        "--max-explain-questions",
        type=_positive_int,
        default=None,
        help=(
            "Optional cap on how many accepted MCQs receive Trace, "
            "Trace+Provenance, and AXIS explanations. Defaults to all accepted "
            "questions."
        ),
    )
    parser.add_argument("--question", default=DEFAULTS["question"])
    parser.add_argument(
        "--axis-interrogation-mode",
        choices=["constrained", "hybrid"],
        default=DEFAULTS["axis_interrogation_mode"],
        help=(
            "AXIS interrogation architecture. constrained (default) enumerates "
            "exact executable queries; hybrid lets the LLM compose a query from "
            "a controlled vocabulary. Replay semantics are unchanged."
        ),
    )
    return parser


def run_batch(
    args: argparse.Namespace,
    *,
    embeddings: Optional[object] = None,
    pipeline: Any = None,
    backbone_factory: Optional[Callable[..., Any]] = None,
    stdout: Any = None,
) -> int:
    """Run the requested stage(s). Returns a process exit code."""
    log = stdout or sys.stdout
    try:
        max_questions = _require_positive_limit(getattr(args, "max_questions", None), "--max-questions")
        max_explain = getattr(args, "max_explain_questions", None)
        if max_explain is not None:
            max_explain = _require_positive_limit(max_explain, "--max-explain-questions")
    except ValueError as exc:
        print(str(exc), file=log)
        return 2

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    try:
        discovered = discover_presentations(input_dir, recursive=bool(args.recursive))
    except FileNotFoundError as exc:
        print(str(exc), file=log)
        return 1

    if getattr(args, "list_pdfs", False):
        print(_list_pdfs_text(discovered, input_dir, output_dir), file=log)
        return 0 if discovered.pdfs else 1

    try:
        selected = resolve_pdf_selection(
            discovered,
            input_dir,
            pdf_selectors=list(getattr(args, "pdf_selectors", None) or []),
            document_ids=list(getattr(args, "document_ids", None) or []),
        )
    except SelectionError as exc:
        print(str(exc), file=log)
        return 1

    if discovered.unsupported and not (getattr(args, "pdf_selectors", None) or getattr(args, "document_ids", None)):
        print(format_unsupported_report(discovered.unsupported), file=log)

    rag_cfg = {
        "embedding_model": args.embedding_model,
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "chunk_by_page": not bool(args.no_page_chunking),
    }
    gen_cfg = {
        "llm_model": args.llm_model,
        "model_type": args.model_type,
        "temperature": float(args.temperature),
        "seed": int(args.seed),
        "max_rounds": int(args.max_rounds),
        "max_questions": max_questions,
        "max_explain_questions": max_explain,
        "question": args.question,
        "orchestrator": "langgraph",
        "axis_interrogation_mode": args.axis_interrogation_mode,
        "axis_interrogation_policy": "counterfactual_only",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        manifest["input_dir"] = str(input_dir)
        manifest["output_dir"] = str(output_dir)
        manifest["config"] = {**rag_cfg, **gen_cfg}
    else:
        manifest = empty_manifest(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            config={**rag_cfg, **gen_cfg},
        )
    if not (getattr(args, "pdf_selectors", None) or getattr(args, "document_ids", None)):
        manifest["unsupported"] = [
            {"path": str(p), "reason": reason} for p, reason in discovered.unsupported
        ]

    failures = 0
    if not selected:
        print("No PDF presentations found.", file=log)
        failures += 1

    shared_pipeline = pipeline
    stage = args.stage
    do_rag = stage in {"rag", "all"}
    do_gen = stage in {"generate", "all"}
    if do_rag and shared_pipeline is None:
        from rag_pipeline import RAGPipeline

        shared_pipeline = RAGPipeline(
            vector_store_path=str(output_dir / "_embedding_cache"),
            embedding_model=str(rag_cfg["embedding_model"]),
            chunk_size=int(rag_cfg["chunk_size"]),
            chunk_overlap=int(rag_cfg["chunk_overlap"]),
            embeddings=embeddings,
        )
    if do_gen and not do_rag:
        print("Generation stage only: using prepared indexes from the manifest.", file=log)

    for pdf_path in selected:
        rel = posix_relative(pdf_path, input_dir)
        doc_id = collection_safe_id(rel)
        print(f"\n=== {rel}  id={doc_id} ===", file=log)
        existing = find_record(manifest, doc_id)
        try:
            if do_rag:
                rec = prepare_presentation(
                    pdf_path=pdf_path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    document_id=doc_id,
                    collection_name=doc_id,
                    embedding_model=str(rag_cfg["embedding_model"]),
                    chunk_size=int(rag_cfg["chunk_size"]),
                    chunk_overlap=int(rag_cfg["chunk_overlap"]),
                    chunk_by_page=bool(rag_cfg["chunk_by_page"]),
                    embeddings=embeddings,
                    pipeline=shared_pipeline,
                    overwrite=bool(args.overwrite),
                    resume=bool(args.resume),
                    existing_record=existing,
                )
                existing = rec
                upsert_record(manifest, rec)
                save_manifest(manifest_path, manifest)
                write_summary_csv(output_dir / SUMMARY_NAME, manifest)
                if rec.get("rag_status") != "success":
                    print(f"  RAG {rec.get('rag_status')}: {rec.get('errors')}", file=log)
                    failures += 1
                    if do_gen:
                        print("  Skipping generation for this presentation.", file=log)
                    continue
                print(
                    f"  RAG ok: pages={rec.get('num_pages')} "
                    f"chunks={rec.get('chunk_count')}",
                    file=log,
                )
            elif existing is None:
                print(
                    "  No manifest record; run --stage rag before --stage generate.",
                    file=log,
                )
                failures += 1
                continue

            if do_gen:
                rec = generate_presentation(
                    record=existing or find_record(manifest, doc_id) or {},
                    input_dir=input_dir,
                    output_dir=output_dir,
                    rag_config_values=rag_cfg,
                    generation_config=gen_cfg,
                    overwrite=bool(args.overwrite),
                    resume=bool(args.resume),
                    backbone_factory=backbone_factory,
                )
                upsert_record(manifest, rec)
                save_manifest(manifest_path, manifest)
                write_summary_csv(output_dir / SUMMARY_NAME, manifest)
                status = rec.get("generate_status")
                print(f"  generate {status} run_id={rec.get('run_id')}", file=log)
                if status not in {"success", "partial"} and "generate_resumed" not in (
                    rec.get("warnings") or []
                ):
                    failures += 1
        except Exception as exc:
            failures += 1
            rec = existing or {
                "document_id": doc_id,
                "relative_path": rel,
                "source_hash": file_sha256(pdf_path) if pdf_path.is_file() else "",
            }
            rec = dict(rec)
            rec["errors"] = list(rec.get("errors") or []) + [str(exc)]
            if do_rag and rec.get("rag_status") not in {"success", "requires_ocr"}:
                rec["rag_status"] = "error"
            if do_gen:
                rec["generate_status"] = "error"
            upsert_record(manifest, rec)
            save_manifest(manifest_path, manifest)
            write_summary_csv(output_dir / SUMMARY_NAME, manifest)
            print(f"  ERROR: {exc}", file=log)
            continue

    if shared_pipeline is not None and hasattr(shared_pipeline, "close"):
        shared_pipeline.close()
    save_manifest(manifest_path, manifest)
    write_summary_csv(output_dir / SUMMARY_NAME, manifest)
    print(f"\nWrote {manifest_path}", file=log)
    print(f"Wrote {output_dir / SUMMARY_NAME}", file=log)
    if failures:
        print(f"Completed with {failures} failure(s).", file=log)
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_batch(args)


def _require_positive_limit(value: Any, flag: str) -> int:
    if value is None:
        raise ValueError(f"{flag} is required and must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag} must be a positive integer.") from exc
    if parsed < 1:
        raise ValueError(f"{flag} must be a positive integer.")
    return parsed


def _list_pdfs_text(discovered: Any, input_dir: Path, output_dir: Path) -> str:
    rows = catalog_presentations(discovered, input_dir)
    statuses: Dict[str, Dict[str, str]] = {}
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = load_manifest(manifest_path)
        except Exception:
            manifest = {}
        for rec in manifest.get("presentations") or []:
            doc_id = str(rec.get("document_id") or "")
            if doc_id:
                statuses[doc_id] = {
                    "rag_status": str(rec.get("rag_status") or "not_started"),
                    "generate_status": str(rec.get("generate_status") or "not_started"),
                }
    if discovered.unsupported:
        extra = "\n\n" + format_unsupported_report(discovered.unsupported)
    else:
        extra = ""
    return format_pdf_list(rows, statuses=statuses) + extra
