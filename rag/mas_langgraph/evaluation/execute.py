"""Per-document LectureBank eval loop (storage/lifecycle around existing RAG/MAS/AXIS)."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TextIO

from mas_langgraph.batch.generate import run_factual_and_explanation_in_memory
from mas_langgraph.batch.io import atomic_write_json, load_json
from mas_langgraph.batch.prepare import prepare_presentation
from mas_langgraph.evaluation.axis_metrics import extract_experiment_metrics
from mas_langgraph.evaluation.lifecycle import (
    delete_vector_store,
    drop_document_rows,
    eval_completed,
    maybe_delete_downloaded_pdf,
    metrics_resume_ok,
    presentation_dir,
    promote_vector_store,
    vector_store_dir,
)
from mas_langgraph.evaluation.reference_sweep_snapshot import (
    EVAL_STATUS_SWEEP_IN_PROGRESS,
    MissingReferenceSweepSnapshotError,
    ReferenceSweepIdentityMismatchError,
    ReferenceSweepSnapshotError,
    build_identity,
    resolve_reference_sweep_action,
)
from mas_langgraph.evaluation.quiz_export import extract_quiz_questions
from mas_langgraph.evaluation.streamlit_export import (
    UNAVAILABLE_RESUME_MESSAGE,
    payload_is_experiment,
    streamlit_json_path as streamlit_experiment_json_path,
    visualization_export_issues,
    write_streamlit_experiment_json,
)
from mas_langgraph.evaluation.lecturebank_meta import lecturebank_record_from_extraction
from mas_langgraph.evaluation.pdf_source import (
    PdfSourceError,
    classify_lecturebank_source,
    document_id_for,
    resolve_lecturebank_source,
)

_JSONL_STRATEGY_SHORT = {
    "normalized_url": "url",
    "metadata_composite": "metadata",
}


def jsonl_lookup_for_selected(
    jsonl_path: Path,
    selected: Sequence[str],
    catalog: Any,
    *,
    authoritative: bool = False,
) -> Dict[str, Any]:
    """Look up JSONL records for selected IDs.

    Default: metadata-confirmed match against LectureBank records.
    Authoritative: exact JSONL extraction ID lookup, no TSV matching.
    """
    if authoritative:
        from rag_pipeline import load_selected_extractions

        lookups = load_selected_extractions(jsonl_path, selected)
        for ident, item in lookups.items():
            if item.status == "usable":
                item.match_strategy = "jsonl_id"
                item.match_status = "unique_match"
                item.extraction_id = item.extraction_id or str(ident)
        return lookups
    from rag_pipeline import match_lecturebank_extractions

    records = []
    for lb_id in selected:
        rec = catalog.get(lb_id) if catalog is not None else None
        if rec is not None:
            records.append(rec)
    return match_lecturebank_extractions(jsonl_path, records)


def run_preflight(
    selected: Sequence[str],
    *,
    catalog: Any,
    cache_dir: Path,
    pdf_dir: Optional[Path],
    lecturebank_dir: Path,
    no_download: bool,
    log: TextIO,
    jsonl_path: Optional[Path] = None,
    jsonl_authoritative: bool = False,
) -> int:
    print("ID\tPDF status\tsource", file=log)
    available = 0
    unavailable = 0
    jsonl_n = 0
    local_n = 0
    download_n = 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    jsonl_lookup = None
    if jsonl_path is not None:
        jsonl_lookup = jsonl_lookup_for_selected(
            jsonl_path,
            selected,
            catalog,
            authoritative=jsonl_authoritative,
        )
    for lb_id in selected:
        rec = catalog.get(lb_id) if catalog is not None else None
        if rec is None and not jsonl_authoritative:
            print(f"{lb_id}\tunavailable\tmissing_metadata", file=log)
            unavailable += 1
            continue
        info = classify_lecturebank_source(
            lecturebank_id=lb_id,
            url=(rec.url if rec is not None else ""),
            cache_dir=cache_dir,
            pdf_dir=pdf_dir,
            search_roots=(lecturebank_dir,) if lecturebank_dir is not None else (),
            no_download=no_download,
            jsonl_lookup=jsonl_lookup,
            require_jsonl=jsonl_authoritative,
        )
        status = info.get("status") or "unavailable"
        source = info.get("source") or ""
        extra = ""
        if jsonl_authoritative:
            title = str(info.get("title") or (rec.title if rec is not None else "") or "")
            reason = str(info.get("reason") or "")
            extra = "\t" + (title if status == "available" else reason)
        elif source == "extraction_jsonl":
            ext_id = str(info.get("extraction_jsonl_id") or "")
            strategy = str(info.get("jsonl_match_strategy") or "")
            short = _JSONL_STRATEGY_SHORT.get(strategy, strategy)
            bits = []
            if ext_id:
                bits.append(f"jsonl_id={ext_id}")
            if short:
                bits.append(f"match={short}")
            if bits:
                extra = "\t" + "\t".join(bits)
        print(f"{lb_id}\t{status}\t{source}{extra}", file=log)
        if status == "available":
            available += 1
            if source == "extraction_jsonl":
                jsonl_n += 1
            elif source in {"local_pdf", "cached_pdf"}:
                local_n += 1
            elif source == "downloadable_pdf":
                download_n += 1
        else:
            unavailable += 1
    print("", file=log)
    print(f"Selected: {len(selected)}", file=log)
    if jsonl_lookup is not None:
        print(f"JSONL available: {jsonl_n}", file=log)
        print(f"Local/cache PDF: {local_n}", file=log)
        print(f"Downloadable PDF: {download_n}", file=log)
    print(f"{available} available", file=log)
    print(f"{unavailable} unavailable", file=log)
    return 0 if unavailable == 0 else 1


def lifecycle_record(
    lecturebank_id: str,
    meta: Mapping[str, Any],
    *,
    pdf_info: Optional[Mapping[str, Any]] = None,
    rag_fp: Optional[Mapping[str, Any]] = None,
    gen_fp: Optional[Mapping[str, Any]] = None,
    rag_record: Optional[Mapping[str, Any]] = None,
    eval_status: str = "",
    source_resolved: bool = False,
    rag_completed: bool = False,
    factual_completed: bool = False,
    axis_completed: bool = False,
    metrics_persisted: bool = False,
    completed: bool = False,
    vector_store_retained: Optional[bool] = None,
    run_id: Any = "",
    accepted: Any = 0,
    explained: Any = 0,
    failure_stage: str = "",
    error: str = "",
    streamlit_json_path: str = "",
) -> Dict[str, Any]:
    pdf = dict(pdf_info or {})
    rec = dict(rag_record or {})
    source = str(pdf.get("source") or "")
    acquisition = {
        "local": "local",
        "cache": "cached",
        "download": "downloaded",
        "extraction_jsonl": "extraction_jsonl",
    }.get(source, source)
    out = {
        **dict(meta),
        "lecturebank_id": lecturebank_id,
        "document_id": document_id_for(lecturebank_id),
        "title": meta.get("title") or rec.get("title") or ((pdf.get("pdf_data") or {}).get("metadata") or {}).get("title"),
        "url": meta.get("url") or pdf.get("url"),
        "source_url": pdf.get("url") or meta.get("url"),
        "source_path": pdf.get("source_path") or str(pdf.get("path") or pdf.get("input_source_path") or ""),
        "pdf_source": source,
        "pdf_acquisition_type": acquisition,
        "input_source_type": pdf.get("input_source_type") or source,
        "input_source_path": pdf.get("input_source_path") or str(pdf.get("path") or ""),
        "jsonl_status": pdf.get("jsonl_status") or "",
        "extraction_jsonl_id": pdf.get("extraction_jsonl_id") or "",
        "jsonl_match_strategy": pdf.get("jsonl_match_strategy") or "",
        "jsonl_match_status": pdf.get("jsonl_match_status") or "",
        "jsonl_title": pdf.get("jsonl_title") or "",
        "jsonl_url": pdf.get("jsonl_url") or "",
        "lecturebank_title": meta.get("title") or "",
        "lecturebank_url": meta.get("url") or "",
        "num_pages": pdf.get("num_pages") or rec.get("num_pages"),
        "pdf_sha256": pdf.get("sha256") or rec.get("source_hash"),
        "source_hash": pdf.get("sha256") or rec.get("source_hash"),
        "rag_config": dict(rag_fp or {}),
        "generation_config": dict(gen_fp or {}),
        "embedding_model": (rag_fp or {}).get("embedding_model"),
        "chunk_size": (rag_fp or {}).get("chunk_size"),
        "chunk_overlap": (rag_fp or {}).get("chunk_overlap"),
        "chunk_by_page": (rag_fp or {}).get("chunk_by_page"),
        "llm_model": (gen_fp or {}).get("llm_model"),
        "temperature": (gen_fp or {}).get("temperature"),
        "seed": (gen_fp or {}).get("seed"),
        "max_questions": (gen_fp or {}).get("max_questions"),
        "max_explain_questions": (gen_fp or {}).get("max_explain_questions"),
        "max_rounds": (gen_fp or {}).get("max_rounds"),
        "question": (gen_fp or {}).get("question"),
        "run_id": run_id or rec.get("run_id") or "",
        "accepted_question_count": accepted,
        "explained_question_count": explained,
        "eval_status": eval_status,
        "rag_status": rec.get("rag_status") or "",
        "source_resolved": source_resolved,
        "rag_completed": rag_completed,
        "factual_completed": factual_completed,
        "axis_completed": axis_completed,
        "metrics_persisted": metrics_persisted,
        "completed": completed,
        "vector_store_retained": vector_store_retained,
        "failure_stage": failure_stage,
        "error_message": error,
    }
    if streamlit_json_path:
        out["streamlit_json_path"] = streamlit_json_path
    if rec:
        for key in ("document_id", "collection_name", "index_dir", "relative_path", "num_pages", "chunk_count"):
            if rec.get(key) is not None:
                out[key] = rec.get(key)
    return out


def _apply_concept_change_evaluation(
    *,
    payload: Mapping[str, Any],
    extracted: Dict[str, Any],
    lecturebank_id: str,
    args: Any,
    gen_cfg: Mapping[str, Any],
    metrics_dir: Path,
    log: TextIO,
) -> None:
    if not bool(getattr(args, "evaluate_concept_change", False)):
        return
    if not payload_is_experiment(payload):
        print("  concept-change: experiment payload missing question content; not evaluable", file=log)
        return
    from mas_langgraph.evaluation.concept_change import (
        attach_assessments_to_metric_rows,
        default_concept_change_config,
        maybe_evaluate_concept_change,
        persisted_concept_change_complete,
    )

    k = int(gen_cfg.get("max_rounds") or 5)
    generate_fn = getattr(args, "_concept_change_generate_fn", None)
    reevaluate = bool(getattr(args, "reevaluate_concept_change", False))
    cfg = default_concept_change_config()
    if not reevaluate and persisted_concept_change_complete(
        payload,
        assessments=list(extracted.get("concept_change_assessments") or []),
        question_rows=list(extracted.get("concept_change_by_question") or []),
        config=cfg,
        k=k,
        reevaluate=False,
    )[0]:
        print("  concept-change: persisted assessments cover the expected selection", file=log)
        return
    try:
        out = maybe_evaluate_concept_change(
            payload,
            lecturebank_id=lecturebank_id,
            k=k,
            metrics_dir=metrics_dir,
            generate_fn=generate_fn,
            config=cfg,
            existing_assessments=list(extracted.get("concept_change_assessments") or []),
            reevaluate=reevaluate,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  concept-change evaluation failed: {exc}", file=log)
        return
    extracted["concept_change_assessments"] = list(out.get("assessments") or [])
    extracted["concept_change_by_question"] = list(out.get("question_rows") or [])
    extracted["concept_change_aggregate"] = dict(out.get("aggregate") or {})
    extracted["replay_comparisons"] = attach_assessments_to_metric_rows(
        list(extracted.get("replay_comparisons") or []),
        extracted["concept_change_assessments"],
    )
    print(
        f"  concept-change: assessed {len(extracted['concept_change_assessments'])} "
        f"AXIS CFs within K (llm_calls={out.get('llm_calls')} cache_hits={out.get('cache_hits')})",
        file=log,
    )


def _apply_saquet_evaluation(
    *,
    payload: Mapping[str, Any],
    extracted: Dict[str, Any],
    lecturebank_id: str,
    args: Any,
    metrics_dir: Path,
    log: TextIO,
) -> None:
    if not bool(getattr(args, "evaluate_saquet", False)):
        return
    from mas_langgraph.evaluation.saquet_eval import evaluate_document_saquet

    reevaluate = bool(getattr(args, "reevaluate_saquet", False))
    apply_fn = getattr(args, "_saquet_apply_fn", None)
    accepted_meta = extracted.get("accepted_question_count")
    if accepted_meta in (None, ""):
        accepted_meta = (extracted.get("document_metrics") or {}).get("accepted_question_count")
    try:
        out = evaluate_document_saquet(
            payload,
            lecturebank_id=str(lecturebank_id),
            run_id=str(extracted.get("run_id") or (payload or {}).get("run_id") or ""),
            metrics_dir=metrics_dir,
            apply_fn=apply_fn,
            reevaluate=reevaluate,
            accepted_count_metadata=accepted_meta,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  saquet evaluation failed: {exc}", file=log)
        return
    extracted["saquet_by_mas_rows"] = [out["mas_row"]]
    print(
        f"  saquet: accepted={out.get('accepted_count')} "
        f"completeness={out.get('completeness')} "
        f"fully={out['mas_row'].get('fully_evaluated_mcq_count')} "
        f"cache_hits={out.get('cache_hits')}",
        file=log,
    )
    reason = str((out.get("mas_row") or {}).get("quiz_completeness_reason") or "")
    if str(out.get("completeness") or "") in {"incomplete", "unknown"} and reason:
        print(f"  saquet: {reason}", file=log)


def log_doc_status(
    log: TextIO,
    result: Mapping[str, Any],
    *,
    vector_cleaned: bool,
    comparisons: Optional[int] = None,
) -> None:
    cmp_n = comparisons
    if cmp_n is None:
        cmp_n = len(result.get("replay_comparisons") or [])
    print(f"  accepted MCQs: {result.get('accepted_question_count') or 0}", file=log)
    print(f"  explained MCQs: {result.get('explained_question_count') or 0}", file=log)
    print(f"  AXIS comparisons: {cmp_n}", file=log)
    print("  metrics persisted: yes", file=log)
    print(f"  vector store cleaned: {'yes' if vector_cleaned else 'no'}", file=log)


def process_documents(
    *,
    selected: Sequence[str],
    catalog: Any,
    args: Any,
    lecturebank_dir: Path,
    output_dir: Path,
    pdf_dir: Optional[Path],
    cache_dir: Path,
    metrics_dir: Path,
    manifest: Dict[str, Any],
    manifest_path: Path,
    rag_cfg: Mapping[str, Any],
    gen_cfg: Mapping[str, Any],
    rag_fp: Mapping[str, Any],
    gen_fp: Mapping[str, Any],
    collected: List[Dict[str, Any]],
    persist: Callable[..., None],
    failed_document: Callable[..., Dict[str, Any]],
    find_record: Callable[..., Optional[Dict[str, Any]]],
    upsert: Callable[..., None],
    embeddings: Optional[object] = None,
    pipeline: Any = None,
    backbone_factory: Optional[Callable[..., Any]] = None,
    log: TextIO,
) -> Dict[str, int]:
    cleanup_store = bool(getattr(args, "cleanup_vector_store", False)) or bool(
        getattr(args, "temporary_vector_stores", False)
    )
    use_temp = bool(getattr(args, "temporary_vector_stores", False))
    cleanup_pdfs = bool(getattr(args, "cleanup_downloaded_pdfs", False))
    stage = str(getattr(args, "stage", "all") or "all")
    do_rag = stage in {"rag", "all"}
    do_eval = stage in {"evaluate", "all"}
    failures = 0
    skipped_resume = 0
    stores_cleaned = 0
    stores_retained = 0
    total = len(selected)

    shared_pipeline = pipeline
    if do_rag and shared_pipeline is None:
        from rag_pipeline import RAGPipeline

        shared_pipeline = RAGPipeline(
            vector_store_path=str(output_dir / "_embedding_cache"),
            embedding_model=str(rag_cfg["embedding_model"]),
            chunk_size=int(rag_cfg["chunk_size"]),
            chunk_overlap=int(rag_cfg["chunk_overlap"]),
            embeddings=embeddings,
        )
    if do_eval and not do_rag:
        print("Evaluation stage only: using prepared indexes when present.", file=log)

    jsonl_lookup = None
    jsonl_path = getattr(args, "extraction_jsonl", None)
    jsonl_authoritative = bool(getattr(args, "jsonl_authoritative", False))
    if getattr(args, "_jsonl_lookup", None) is not None:
        jsonl_lookup = args._jsonl_lookup
    elif jsonl_path:
        jsonl_lookup = jsonl_lookup_for_selected(
            Path(jsonl_path),
            selected,
            catalog,
            authoritative=jsonl_authoritative,
        )
    if jsonl_path:
        usable = sum(
            1
            for item in (jsonl_lookup or {}).values()
            if getattr(item, "status", None) == "usable"
        )
        if jsonl_authoritative:
            print(
                f"Extraction JSONL: {usable}/{len(selected)} selected JSONL IDs usable from {jsonl_path}",
                file=log,
            )
        else:
            print(
                f"Extraction JSONL: {usable}/{len(selected)} selected LectureBank IDs uniquely matched from {jsonl_path}",
                file=log,
            )

    for index, lb_id in enumerate(selected, start=1):
        print(f"\n[{index}/{total}] LectureBank {lb_id}", file=log)
        rec_meta = catalog.get(lb_id) if catalog is not None else None
        if jsonl_authoritative:
            item = (jsonl_lookup or {}).get(lb_id)
            if item is not None and getattr(item, "status", "") != "missing":
                rec_meta = lecturebank_record_from_extraction(lb_id, item)
            else:
                rec_meta = None
        if rec_meta is None:
            collected[:] = drop_document_rows(collected, lb_id)
            missing_src = jsonl_path if jsonl_authoritative else getattr(catalog, "tsv_path", "catalog")
            stage = "selection"
            error = (
                f"JSONL extraction ID {lb_id} is not present in {missing_src}"
                if jsonl_authoritative
                else f"LectureBank ID {lb_id} is not in {missing_src}"
            )
            result = failed_document(
                lb_id,
                {},
                stage=stage,
                error=error,
            )
            collected.append(result)
            failures += 1
            stores_retained += 1
            persist(output_dir, metrics_dir, manifest_path, manifest, collected)
            log_doc_status(log, result, vector_cleaned=False)
            continue

        try:
            pdf_info = resolve_lecturebank_source(
                lecturebank_id=lb_id,
                url=rec_meta.url,
                cache_dir=cache_dir,
                pdf_dir=pdf_dir,
                search_roots=(lecturebank_dir,) if lecturebank_dir is not None else (),
                no_download=bool(getattr(args, "no_download", False)),
                jsonl_lookup=jsonl_lookup,
                require_jsonl=jsonl_authoritative,
            )
        except PdfSourceError as exc:
            collected[:] = drop_document_rows(collected, lb_id)
            result = failed_document(
                lb_id,
                rec_meta.to_meta_dict(),
                stage="pdf",
                error=str(exc),
            )
            collected.append(result)
            failures += 1
            stores_retained += 1
            upsert(
                manifest,
                lifecycle_record(
                    lb_id,
                    rec_meta.to_meta_dict(),
                    rag_fp=rag_fp,
                    gen_fp=gen_fp,
                    eval_status="failed",
                    source_resolved=False,
                    error=str(exc),
                    failure_stage="pdf",
                ),
            )
            persist(output_dir, metrics_dir, manifest_path, manifest, collected)
            log_doc_status(log, result, vector_cleaned=False)
            continue

        doc_id = document_id_for(lb_id)
        existing = find_record(manifest, lb_id)
        using_jsonl = str(pdf_info.get("source") or "") == "extraction_jsonl"
        if using_jsonl:
            pdf_input_dir = Path(pdf_info["input_source_path"]).resolve().parent
        else:
            pdf_input_dir = Path(pdf_info["path"]).resolve().parent
        if (
            bool(args.resume)
            and not bool(args.overwrite)
            and do_eval
            and metrics_resume_ok(
                lecturebank_id=lb_id,
                source_hash=str(pdf_info["sha256"]),
                rag_fp=rag_fp,
                gen_fp=gen_fp,
                stored=existing,
                collected=collected,
            )
        ):
            ref_complete = True
            if bool(getattr(args, "build_effect_reference_set", False)):
                resumed_item = next(
                    (
                        row
                        for row in collected
                        if str(row.get("lecturebank_id") or "") == str(lb_id)
                    ),
                    None,
                )
                qrows = list((resumed_item or {}).get("effect_discovery_by_question") or [])
                status = str((resumed_item or {}).get("reference_sweep_status") or "")
                ref_complete = status == "complete" or (
                    bool(qrows)
                    and all(r.get("reference_set_complete") is True for r in qrows)
                )
                if not ref_complete:
                    print(
                        "  resume: AXIS metrics complete but replay-reference sweep "
                        "incomplete; restoring the saved factual snapshot (not regenerating MCQs)",
                        file=log,
                    )
            if ref_complete:
                skipped_resume += 1
                print("  skipped by resume (metrics already persisted)", file=log)
                if bool(getattr(args, "export_quiz", False)):
                    resumed_item = next(
                        (
                            row
                            for row in collected
                            if str(row.get("lecturebank_id") or "") == str(lb_id)
                        ),
                        None,
                    )
                    has_quiz = bool(resumed_item and list(resumed_item.get("quiz_questions") or []))
                    if not has_quiz:
                        print(
                            "  quiz export unavailable: previously completed document "
                            "has no persisted quiz text; not reconstructing from metrics",
                            file=log,
                        )
                        if resumed_item is not None:
                            resumed_item["quiz_export_unavailable"] = True
                if bool(getattr(args, "export_streamlit_json", False)):
                    json_path = streamlit_experiment_json_path(output_dir, lb_id)
                    if not json_path.is_file() or json_path.stat().st_size == 0:
                        print(f"  {UNAVAILABLE_RESUME_MESSAGE}", file=log)
                        resumed_item = next(
                            (
                                row
                                for row in collected
                                if str(row.get("lecturebank_id") or "") == str(lb_id)
                            ),
                            None,
                        )
                        if resumed_item is not None:
                            resumed_item["streamlit_export_unavailable"] = True
                if bool(getattr(args, "evaluate_concept_change", False)):
                    resumed_item = next(
                        (
                            row
                            for row in collected
                            if str(row.get("lecturebank_id") or "") == str(lb_id)
                        ),
                        None,
                    )
                    json_path = streamlit_experiment_json_path(output_dir, lb_id)
                    if json_path.is_file() and json_path.stat().st_size > 0:
                        try:
                            saved = load_json(json_path)
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"  concept-change: could not read saved experiment JSON ({exc})",
                                file=log,
                            )
                            saved = None
                        if resumed_item is None:
                            print(
                                "  concept-change: no collected metrics row to attach assessments",
                                file=log,
                            )
                        elif not payload_is_experiment(saved):
                            print(
                                "  concept-change: saved experiment JSON missing question content; not evaluable",
                                file=log,
                            )
                            resumed_item["concept_change_unavailable"] = True
                        else:
                            _apply_concept_change_evaluation(
                                payload=saved,
                                extracted=resumed_item,
                                lecturebank_id=str(lb_id),
                                args=args,
                                gen_cfg=gen_cfg,
                                metrics_dir=metrics_dir,
                                log=log,
                            )
                            try:
                                atomic_write_json(json_path, saved)
                            except Exception as exc:  # noqa: BLE001
                                print(
                                    f"  concept-change: could not write assessments back to JSON ({exc})",
                                    file=log,
                                )
                    else:
                        print(
                            "  concept-change: saved experiment JSON missing; "
                            "not evaluable without rerunning MAS",
                            file=log,
                        )
                        if resumed_item is not None:
                            resumed_item["concept_change_unavailable"] = True
                if bool(getattr(args, "evaluate_saquet", False)):
                    resumed_item = next(
                        (
                            row
                            for row in collected
                            if str(row.get("lecturebank_id") or "") == str(lb_id)
                        ),
                        None,
                    )
                    json_path = streamlit_experiment_json_path(output_dir, lb_id)
                    if json_path.is_file() and json_path.stat().st_size > 0:
                        try:
                            saved = load_json(json_path)
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"  saquet: could not read saved experiment JSON ({exc})",
                                file=log,
                            )
                            saved = None
                        if resumed_item is None:
                            print(
                                "  saquet: no collected metrics row to attach assessments",
                                file=log,
                            )
                        elif saved is None:
                            pass
                        else:
                            _apply_saquet_evaluation(
                                payload=saved,
                                extracted=resumed_item,
                                lecturebank_id=str(lb_id),
                                args=args,
                                metrics_dir=metrics_dir,
                                log=log,
                            )
                    else:
                        print(
                            "  saquet: saved experiment JSON missing; "
                            "not evaluable without rerunning MAS",
                            file=log,
                        )
                cleaned = existing is not None and existing.get("vector_store_retained") is False
                log_doc_status(
                    log,
                    existing or {"accepted_question_count": 0, "explained_question_count": 0},
                    vector_cleaned=bool(cleaned),
                    comparisons=int((existing or {}).get("explained_question_count") or 0),
                )
                continue

        collected[:] = drop_document_rows(collected, lb_id)
        persistent_index = vector_store_dir(output_dir, doc_id)
        can_reuse_index = (
            bool(args.resume)
            and not bool(args.overwrite)
            and persistent_index.exists()
            and any(persistent_index.iterdir())
            and existing is not None
            and existing.get("rag_status") == "success"
            and str(existing.get("source_hash") or existing.get("pdf_sha256") or "")
            == str(pdf_info["sha256"])
        )
        need_rag = do_rag or (do_eval and not can_reuse_index)
        work_out = output_dir
        tmp_obj = None
        build_ref = bool(getattr(args, "build_effect_reference_set", False))
        if use_temp and need_rag and not can_reuse_index and not build_ref:
            tmp_obj = tempfile.TemporaryDirectory(
                prefix=f"emaes_lb_{lb_id}_",
                ignore_cleanup_errors=True,
            )
            work_out = Path(tmp_obj.name)

        persisted_ok = False
        vector_cleaned = False
        rag_record: Optional[Dict[str, Any]] = existing if can_reuse_index else None
        try:
            if need_rag:
                try:
                    if using_jsonl:
                        rag_record = prepare_presentation(
                            pdf_data=dict(pdf_info.get("pdf_data") or {}),
                            source_hash=str(pdf_info["sha256"]),
                            input_source_type="extraction_jsonl",
                            jsonl_path=Path(pdf_info["input_source_path"]),
                            input_dir=pdf_input_dir,
                            output_dir=work_out,
                            document_id=doc_id,
                            collection_name=doc_id,
                            embedding_model=str(rag_cfg["embedding_model"]),
                            chunk_size=int(rag_cfg["chunk_size"]),
                            chunk_overlap=int(rag_cfg["chunk_overlap"]),
                            chunk_by_page=bool(rag_cfg["chunk_by_page"]),
                            embeddings=embeddings,
                            pipeline=shared_pipeline,
                            overwrite=bool(args.overwrite),
                            resume=bool(args.resume) and work_out == output_dir,
                            existing_record=existing if work_out == output_dir else None,
                        )
                        pdf_info.pop("pdf_data", None)
                        if jsonl_lookup is not None and str(lb_id) in jsonl_lookup:
                            jsonl_lookup[str(lb_id)].record = None
                    else:
                        rag_record = prepare_presentation(
                            pdf_path=Path(pdf_info["path"]),
                            input_dir=pdf_input_dir,
                            output_dir=work_out,
                            document_id=doc_id,
                            collection_name=doc_id,
                            embedding_model=str(rag_cfg["embedding_model"]),
                            chunk_size=int(rag_cfg["chunk_size"]),
                            chunk_overlap=int(rag_cfg["chunk_overlap"]),
                            chunk_by_page=bool(rag_cfg["chunk_by_page"]),
                            embeddings=embeddings,
                            pipeline=shared_pipeline,
                            overwrite=bool(args.overwrite),
                            resume=bool(args.resume) and work_out == output_dir,
                            existing_record=existing if work_out == output_dir else None,
                        )
                except Exception as exc:
                    result = failed_document(
                        lb_id,
                        rec_meta.to_meta_dict(),
                        stage="rag",
                        error=str(exc),
                        extra={
                            "pdf_source": pdf_info.get("source"),
                            "pdf_sha256": pdf_info.get("sha256"),
                        },
                    )
                    collected.append(result)
                    failures += 1
                    upsert(
                        manifest,
                        lifecycle_record(
                            lb_id,
                            rec_meta.to_meta_dict(),
                            pdf_info=pdf_info,
                            rag_fp=rag_fp,
                            gen_fp=gen_fp,
                            eval_status="failed",
                            source_resolved=True,
                            error=str(exc),
                            failure_stage="rag",
                        ),
                    )
                    persist(output_dir, metrics_dir, manifest_path, manifest, collected)
                    persisted_ok = True
                    stores_retained += 1
                    log_doc_status(log, result, vector_cleaned=False)
                    continue
                upsert(
                    manifest,
                    lifecycle_record(
                        lb_id,
                        rec_meta.to_meta_dict(),
                        pdf_info=pdf_info,
                        rag_fp=rag_fp,
                        gen_fp=gen_fp,
                        rag_record=rag_record,
                        source_resolved=True,
                        rag_completed=rag_record.get("rag_status") == "success",
                        eval_status="rag_only" if not do_eval else "pending",
                    ),
                )
                atomic_write_json(manifest_path, manifest)
                if rag_record.get("rag_status") != "success":
                    result = failed_document(
                        lb_id,
                        rec_meta.to_meta_dict(),
                        stage="rag",
                        error="; ".join(rag_record.get("errors") or ["RAG failed"]),
                        extra={
                            "pdf_source": pdf_info.get("source"),
                            "pdf_sha256": pdf_info.get("sha256"),
                            "rag_status": rag_record.get("rag_status"),
                        },
                    )
                    collected.append(result)
                    failures += 1
                    persist(output_dir, metrics_dir, manifest_path, manifest, collected)
                    persisted_ok = True
                    stores_retained += 1
                    log_doc_status(log, result, vector_cleaned=False)
                    continue
                print(
                    f"  RAG ok: pages={rag_record.get('num_pages')} chunks={rag_record.get('chunk_count')}",
                    file=log,
                )
            elif rag_record is None or rag_record.get("rag_status") != "success":
                result = failed_document(
                    lb_id,
                    rec_meta.to_meta_dict(),
                    stage="rag",
                    error="No successful prepared index; run --stage rag or --stage all.",
                )
                collected.append(result)
                failures += 1
                persist(output_dir, metrics_dir, manifest_path, manifest, collected)
                persisted_ok = True
                stores_retained += 1
                log_doc_status(log, result, vector_cleaned=False)
                continue

            if not do_eval:
                result = {
                    **rec_meta.to_meta_dict(),
                    "document_id": doc_id,
                    "pdf_source": pdf_info.get("source"),
                    "pdf_sha256": pdf_info.get("sha256"),
                    "failure_stage": "",
                    "error_message": "",
                    "rag_status": (rag_record or {}).get("rag_status") or "success",
                    "mas_status": "not_started",
                    "axis_status": "not_started",
                    "eval_status": "rag_only",
                    "accepted_question_count": 0,
                    "explained_question_count": 0,
                    "interrogation_attempts": [],
                    "replay_comparisons": [],
                    "provenance_edges": [],
                    "distributions": [],
                    "round_metrics": [],
                    "document_metrics": {},
                }
                collected.append(result)
                upsert(
                    manifest,
                    lifecycle_record(
                        lb_id,
                        rec_meta.to_meta_dict(),
                        pdf_info=pdf_info,
                        rag_fp=rag_fp,
                        gen_fp=gen_fp,
                        rag_record=rag_record,
                        eval_status="rag_only",
                        source_resolved=True,
                        rag_completed=True,
                        metrics_persisted=True,
                        vector_store_retained=True,
                    ),
                )
                persist(output_dir, metrics_dir, manifest_path, manifest, collected)
                persisted_ok = True
                stores_retained += 1
                log_doc_status(log, result, vector_cleaned=False)
                continue

            try:
                from mas_langgraph.evaluation.effect_reference import (
                    load_reference_intervention_rows,
                    upsert_reference_csv,
                )
                from mas_langgraph.experiment.export import dump_explanation_experiment
                from mas_langgraph.batch.generate import serialize_checkpoint_index

                build_ref = bool(getattr(args, "build_effect_reference_set", False))
                ref_csv = metrics_dir / "reference_interventions.csv"
                resume_rows = load_reference_intervention_rows(ref_csv) if build_ref else []
                persist_cb = None
                if build_ref:
                    def persist_cb(rows, _path=ref_csv):
                        upsert_reference_csv(_path, rows)

                resume_sweep = False
                if build_ref:
                    current_identity = build_identity(
                        run_id="",
                        lecturebank_id=str(lb_id),
                        document_id=str(doc_id),
                        source_hash=str(pdf_info.get("sha256") or ""),
                        rag_config=dict(rag_fp),
                        generation_config=dict(gen_fp),
                        axis_interrogation_mode=str(
                            gen_cfg.get("axis_interrogation_mode") or "hybrid"
                        ),
                        reference_max_interventions_per_question=getattr(
                            args, "reference_max_interventions_per_question", None
                        ),
                    )
                    action = resolve_reference_sweep_action(
                        output_dir=output_dir,
                        document_id=str(doc_id),
                        lecturebank_id=str(lb_id),
                        csv_path=ref_csv,
                        current_identity=current_identity,
                    )
                    if action == "skip_complete":
                        print(
                            "  resume: reference sweep snapshot is complete; "
                            "restoring saved factual execution without repeating interventions",
                            file=log,
                        )
                    resume_sweep = action in {"resume", "skip_complete"}
                    if action == "resume":
                        print(
                            "  resume: restoring saved factual execution and skipping "
                            "completed reference interventions",
                            file=log,
                        )

                def after_axis_metrics(experiment, service):
                    from mas_langgraph.experiment.export import experiment_to_dict

                    payload = experiment_to_dict(
                        experiment, include_blinding=False, include_raw=True
                    )
                    ckpt = serialize_checkpoint_index(
                        getattr(service, "checkpoint_index", None)
                        or getattr(service, "factual_run_result", None)
                    )
                    extracted = extract_experiment_metrics(
                        payload, ckpt, lecturebank_id=lb_id
                    )
                    valid_n = int(
                        (extracted.get("document_metrics") or {}).get(
                            "valid_replay_comparisons"
                        )
                        or 0
                    )
                    axis_status = "success" if valid_n else "no_valid_replay"
                    partial = {
                        **rec_meta.to_meta_dict(),
                        "document_id": doc_id,
                        "pdf_source": pdf_info.get("source"),
                        "pdf_sha256": pdf_info.get("sha256"),
                        "failure_stage": "",
                        "error_message": "",
                        "rag_status": "success",
                        "mas_status": "success",
                        "axis_status": axis_status,
                        "eval_status": EVAL_STATUS_SWEEP_IN_PROGRESS,
                        "accepted_question_count": (
                            extracted.get("document_metrics") or {}
                        ).get("accepted_question_count")
                        or 0,
                        "explained_question_count": (
                            extracted.get("document_metrics") or {}
                        ).get("explained_question_count")
                        or 0,
                        "run_id": getattr(experiment, "run_id", "") or "",
                        "source_hash": pdf_info.get("sha256"),
                        "rag_config": dict(rag_fp),
                        "generation_config": dict(gen_fp),
                        "reference_sweep_status": "in_progress",
                        **extracted,
                    }
                    collected[:] = drop_document_rows(collected, lb_id)
                    collected.append(partial)
                    upsert(
                        manifest,
                        lifecycle_record(
                            lb_id,
                            rec_meta.to_meta_dict(),
                            pdf_info=pdf_info,
                            rag_fp=rag_fp,
                            gen_fp=gen_fp,
                            rag_record=rag_record,
                            eval_status=EVAL_STATUS_SWEEP_IN_PROGRESS,
                            source_resolved=True,
                            rag_completed=True,
                            factual_completed=True,
                            axis_completed=True,
                            metrics_persisted=True,
                            completed=False,
                            vector_store_retained=True,
                            run_id=partial.get("run_id"),
                            accepted=partial.get("accepted_question_count"),
                            explained=partial.get("explained_question_count"),
                        ),
                    )
                    persist(output_dir, metrics_dir, manifest_path, manifest, collected)
                    dump_explanation_experiment(
                        experiment,
                        str(presentation_dir(output_dir, doc_id) / "experiment.json"),
                        include_blinding=False,
                        include_raw=True,
                    )
                    print(
                        "  saved factual/AXIS metrics before the reference sweep "
                        f"(run_id={partial.get('run_id')})",
                        file=log,
                    )

                gen_record = dict(rag_record or {})
                gen_record.setdefault("source_hash", pdf_info.get("sha256"))
                gen_record.setdefault("pdf_sha256", pdf_info.get("sha256"))
                gen_record.setdefault("lecturebank_id", str(lb_id))
                gen = run_factual_and_explanation_in_memory(
                    record=gen_record,
                    input_dir=pdf_input_dir,
                    output_dir=output_dir if build_ref else work_out,
                    rag_config_values=dict(rag_cfg),
                    generation_config=dict(gen_cfg),
                    backbone_factory=backbone_factory,
                    build_effect_reference_set=build_ref,
                    reference_max_interventions_per_question=getattr(
                        args, "reference_max_interventions_per_question", None
                    ),
                    reference_resume_rows=resume_rows,
                    reference_persist_callback=persist_cb,
                    lecturebank_id=str(lb_id),
                    reuse_hybrid_add_source_pool=not bool(args.overwrite),
                    resume_reference_sweep=resume_sweep,
                    after_axis_metrics_callback=after_axis_metrics if build_ref else None,
                    reference_progress_csv=ref_csv if build_ref else None,
                )
            except (MissingReferenceSweepSnapshotError, ReferenceSweepIdentityMismatchError) as exc:
                result = failed_document(
                    lb_id,
                    rec_meta.to_meta_dict(),
                    stage="reference_sweep_resume",
                    error=str(exc),
                    extra={
                        "pdf_source": pdf_info.get("source"),
                        "pdf_sha256": pdf_info.get("sha256"),
                        "rag_status": "success",
                    },
                )
                collected.append(result)
                failures += 1
                upsert(
                    manifest,
                    lifecycle_record(
                        lb_id,
                        rec_meta.to_meta_dict(),
                        pdf_info=pdf_info,
                        rag_fp=rag_fp,
                        gen_fp=gen_fp,
                        rag_record=rag_record,
                        eval_status="failed",
                        source_resolved=True,
                        rag_completed=True,
                        error=str(exc),
                        failure_stage="reference_sweep_resume",
                    ),
                )
                persist(output_dir, metrics_dir, manifest_path, manifest, collected)
                persisted_ok = True
                stores_retained += 1
                log_doc_status(log, result, vector_cleaned=False)
                continue
            except ReferenceSweepSnapshotError as exc:
                result = failed_document(
                    lb_id,
                    rec_meta.to_meta_dict(),
                    stage="reference_sweep_snapshot",
                    error=str(exc),
                    extra={
                        "pdf_source": pdf_info.get("source"),
                        "pdf_sha256": pdf_info.get("sha256"),
                        "rag_status": "success",
                    },
                )
                collected.append(result)
                failures += 1
                upsert(
                    manifest,
                    lifecycle_record(
                        lb_id,
                        rec_meta.to_meta_dict(),
                        pdf_info=pdf_info,
                        rag_fp=rag_fp,
                        gen_fp=gen_fp,
                        rag_record=rag_record,
                        eval_status="failed",
                        source_resolved=True,
                        rag_completed=True,
                        error=str(exc),
                        failure_stage="reference_sweep_snapshot",
                    ),
                )
                persist(output_dir, metrics_dir, manifest_path, manifest, collected)
                persisted_ok = True
                stores_retained += 1
                log_doc_status(log, result, vector_cleaned=False)
                continue
            except Exception as exc:
                result = failed_document(
                    lb_id,
                    rec_meta.to_meta_dict(),
                    stage="mas",
                    error=str(exc),
                    extra={
                        "pdf_source": pdf_info.get("source"),
                        "pdf_sha256": pdf_info.get("sha256"),
                        "rag_status": "success",
                    },
                )
                collected.append(result)
                failures += 1
                upsert(
                    manifest,
                    lifecycle_record(
                        lb_id,
                        rec_meta.to_meta_dict(),
                        pdf_info=pdf_info,
                        rag_fp=rag_fp,
                        gen_fp=gen_fp,
                        rag_record=rag_record,
                        eval_status="failed",
                        source_resolved=True,
                        rag_completed=True,
                        error=str(exc),
                        failure_stage="mas",
                    ),
                )
                persist(output_dir, metrics_dir, manifest_path, manifest, collected)
                persisted_ok = True
                stores_retained += 1
                log_doc_status(log, result, vector_cleaned=False)
                continue

            gen_status = str((gen.record or {}).get("generate_status") or "")
            if gen.experiment_payload is None:
                stage_name = (
                    "mcq"
                    if gen_status == "mcq_failed"
                    else ("axis" if gen_status == "explanation_failed" else "mas")
                )
                result = failed_document(
                    lb_id,
                    rec_meta.to_meta_dict(),
                    stage=stage_name,
                    error="; ".join(
                        (gen.record or {}).get("errors") or [gen_status or "generation failed"]
                    ),
                    extra={
                        "pdf_source": pdf_info.get("source"),
                        "pdf_sha256": pdf_info.get("sha256"),
                        "rag_status": "success",
                        "mas_status": "success" if gen_status == "mcq_failed" else gen_status,
                        "accepted_question_count": gen.accepted_count,
                    },
                )
                collected.append(result)
                failures += 1
                gen.experiment = None
                gc.collect()
                upsert(
                    manifest,
                    lifecycle_record(
                        lb_id,
                        rec_meta.to_meta_dict(),
                        pdf_info=pdf_info,
                        rag_fp=rag_fp,
                        gen_fp=gen_fp,
                        rag_record=gen.record or rag_record,
                        eval_status="failed",
                        source_resolved=True,
                        rag_completed=True,
                        factual_completed=stage_name != "mas",
                        error=result["error_message"],
                        failure_stage=stage_name,
                    ),
                )
                persist(output_dir, metrics_dir, manifest_path, manifest, collected)
                persisted_ok = True
                stores_retained += 1
                log_doc_status(log, result, vector_cleaned=False)
                continue

            extracted = extract_experiment_metrics(
                gen.experiment_payload,
                gen.checkpoint_index,
                lecturebank_id=lb_id,
            )
            valid_n = int(
                (extracted.get("document_metrics") or {}).get("valid_replay_comparisons") or 0
            )
            axis_status = "success" if valid_n else "no_valid_replay"
            result = {
                **rec_meta.to_meta_dict(),
                "document_id": doc_id,
                "pdf_source": pdf_info.get("source"),
                "pdf_sha256": pdf_info.get("sha256"),
                "failure_stage": "" if valid_n else "axis",
                "error_message": "" if valid_n else "AXIS produced no valid replay comparison",
                "rag_status": "success",
                "mas_status": "success",
                "axis_status": axis_status,
                "eval_status": "success" if valid_n else "axis_no_valid_replay",
                "accepted_question_count": gen.accepted_count
                or (extracted.get("document_metrics") or {}).get("accepted_question_count")
                or 0,
                "explained_question_count": (extracted.get("document_metrics") or {}).get(
                    "explained_question_count"
                )
                or 0,
                "run_id": (gen.record or {}).get("run_id"),
                "source_hash": pdf_info.get("sha256"),
                "rag_config": dict(rag_fp),
                "generation_config": dict(gen_fp),
                **extracted,
            }
            doc_metrics = dict(extracted.get("document_metrics") or {})
            doc_metrics.update(
                {
                    "lecturebank_id": lb_id,
                    "title": rec_meta.title,
                    "topic_id": rec_meta.topic_id,
                    "topic_name": rec_meta.topic_name,
                    "instructor": rec_meta.instructor,
                    "venue": rec_meta.venue,
                    "year": rec_meta.year,
                    "url": rec_meta.url,
                    "document_id": doc_id,
                    "pdf_source": pdf_info.get("source"),
                    "pdf_sha256": pdf_info.get("sha256"),
                    "failure_stage": result["failure_stage"],
                    "error_message": result["error_message"],
                    "rag_status": "success",
                    "mas_status": "success",
                    "axis_status": axis_status,
                    "eval_status": result["eval_status"],
                    "accepted_question_count": result["accepted_question_count"],
                }
            )
            result["document_metrics"] = doc_metrics
            from mas_langgraph.axis.hybrid_add_diagnostics import (
                extract_hybrid_add_diagnostics,
            )

            add_diag = extract_hybrid_add_diagnostics(
                gen.experiment_payload, lecturebank_id=str(lb_id)
            )
            result["hybrid_add_candidates"] = list(
                add_diag.get("hybrid_add_candidates") or []
            )
            result["hybrid_add_boundaries"] = list(
                add_diag.get("hybrid_add_boundaries") or []
            )
            result["hybrid_add_summary"] = dict(add_diag.get("hybrid_add_summary") or {})
            from mas_langgraph.axis.hybrid_add_source import (
                extract_hybrid_add_source_reports,
            )

            src = extract_hybrid_add_source_reports(
                gen.experiment_payload or {}, lecturebank_id=str(lb_id)
            )
            result["hybrid_add_source_events"] = list(
                src.get("hybrid_add_source_events") or []
            )
            result["hybrid_add_source_summary"] = dict(
                src.get("hybrid_add_source_summary") or {}
            )
            ref_payload = (gen.extras or {}).get("effect_reference") or {}
            if ref_payload:
                result["reference_interventions"] = list(ref_payload.get("intervention_rows") or [])
                result["effect_discovery_by_question"] = list(ref_payload.get("question_rows") or [])
                result["reference_benchmark"] = dict(ref_payload.get("aggregate") or {})
                result["reference_sweep_status"] = str(ref_payload.get("status") or "incomplete")
                result["reference_sweep_warnings"] = list(ref_payload.get("warnings") or [])
            if bool(getattr(args, "build_effect_reference_set", False)):
                sweep_status = str(result.get("reference_sweep_status") or "")
                if sweep_status != "complete":
                    result["eval_status"] = EVAL_STATUS_SWEEP_IN_PROGRESS
                    result["failure_stage"] = ""
                    result["error_message"] = ""
            _apply_concept_change_evaluation(
                payload=gen.experiment_payload,
                extracted=result,
                lecturebank_id=str(lb_id),
                args=args,
                gen_cfg=gen_cfg,
                metrics_dir=metrics_dir,
                log=log,
            )
            _apply_saquet_evaluation(
                payload=gen.experiment_payload,
                extracted=result,
                lecturebank_id=str(lb_id),
                args=args,
                metrics_dir=metrics_dir,
                log=log,
            )
            if bool(getattr(args, "export_quiz", False)):
                result["quiz_questions"] = extract_quiz_questions(
                    gen.experiment_payload,
                    lecturebank_id=lb_id,
                    axis_interrogation_mode=str(
                        gen_cfg.get("axis_interrogation_mode")
                        or getattr(args, "axis_interrogation_mode", None)
                        or "constrained"
                    ),
                    document_id=doc_id,
                )
            if bool(getattr(args, "export_streamlit_json", False)) and payload_is_experiment(
                gen.experiment_payload
            ):
                issues = visualization_export_issues(gen.experiment_payload)
                for issue in issues:
                    print(f"  streamlit export: {issue}", file=log)
                result["streamlit_json_path"] = write_streamlit_experiment_json(
                    output_dir,
                    lb_id,
                    gen.experiment_payload,
                )
            if not valid_n:
                failures += 1
            collected[:] = drop_document_rows(collected, lb_id)
            collected.append(result)
            gen.experiment = None
            gen.experiment_payload = None
            gc.collect()

            sweep_incomplete = bool(
                getattr(args, "build_effect_reference_set", False)
            ) and str(result.get("reference_sweep_status") or "") != "complete"
            should_cleanup = bool(cleanup_store and do_eval) and not sweep_incomplete
            done = eval_completed(result["eval_status"]) and not sweep_incomplete
            upsert(
                manifest,
                lifecycle_record(
                    lb_id,
                    rec_meta.to_meta_dict(),
                    pdf_info=pdf_info,
                    rag_fp=rag_fp,
                    gen_fp=gen_fp,
                    rag_record=gen.record or rag_record,
                    eval_status=result["eval_status"],
                    source_resolved=True,
                    rag_completed=True,
                    factual_completed=True,
                    axis_completed=True,
                    metrics_persisted=True,
                    completed=done,
                    vector_store_retained=True,
                    run_id=result.get("run_id"),
                    accepted=result.get("accepted_question_count"),
                    explained=result.get("explained_question_count"),
                    failure_stage=result.get("failure_stage") or "",
                    error=result.get("error_message") or "",
                    streamlit_json_path=str(result.get("streamlit_json_path") or ""),
                ),
            )
            try:
                persist(output_dir, metrics_dir, manifest_path, manifest, collected)
            except Exception:
                persisted_ok = False
                upsert(
                    manifest,
                    {
                        "lecturebank_id": lb_id,
                        "document_id": doc_id,
                        "metrics_persisted": False,
                        "completed": False,
                        "vector_store_retained": True,
                    },
                )
                raise
            persisted_ok = True

            if should_cleanup:
                allowed = [work_out, output_dir]
                if delete_vector_store(
                    vector_store_dir(work_out, doc_id),
                    document_id=doc_id,
                    allowed_roots=allowed,
                ):
                    vector_cleaned = True
                    stores_cleaned += 1
                leftover = vector_store_dir(output_dir, doc_id)
                if leftover.exists() and leftover.resolve() != vector_store_dir(work_out, doc_id).resolve():
                    delete_vector_store(
                        leftover, document_id=doc_id, allowed_roots=[output_dir]
                    )
                upsert(
                    manifest,
                    {
                        "lecturebank_id": lb_id,
                        "document_id": doc_id,
                        "vector_store_retained": not vector_cleaned,
                        "metrics_persisted": True,
                        "completed": done,
                    },
                )
                atomic_write_json(manifest_path, manifest)
            else:
                stores_retained += 1
            if cleanup_pdfs:
                maybe_delete_downloaded_pdf(
                    pdf_info={**pdf_info, "lecturebank_id": lb_id},
                    cache_dir=cache_dir,
                    pdf_dir=pdf_dir,
                    enabled=True,
                )
            log_doc_status(
                log,
                result,
                vector_cleaned=vector_cleaned,
                comparisons=len(result.get("replay_comparisons") or []),
            )
        finally:
            if tmp_obj is not None:
                if not persisted_ok:
                    promote_vector_store(
                        src_output_dir=work_out,
                        dest_output_dir=output_dir,
                        document_id=doc_id,
                    )
                    stores_retained += 1
                tmp_obj.cleanup()
            gc.collect()

    persist(output_dir, metrics_dir, manifest_path, manifest, collected)
    return {
        "failures": failures,
        "skipped_resume": skipped_resume,
        "stores_cleaned": stores_cleaned,
        "stores_retained": stores_retained,
        "total": total,
    }


def run_concept_change_from_saved(
    *,
    selected: Sequence[str],
    args: Any,
    output_dir: Path,
    metrics_dir: Path,
    gen_cfg: Mapping[str, Any],
    collected: List[Dict[str, Any]],
    persist: Callable[..., None],
    manifest_path: Path,
    manifest: Dict[str, Any],
    log: TextIO,
) -> Dict[str, int]:
    """Repair/evaluate concept-change from saved experiment JSON only.

    Never runs RAG, MAS, AXIS interrogation, or replay.
    """
    args.evaluate_concept_change = True
    if not collected:
        print(
            "concept-change-from-saved: no persisted document metrics in the output folder; "
            "not running MAS/AXIS. Re-run evaluation first or point --output-dir at an existing run.",
            file=log,
        )
        return {
            "failures": len(selected),
            "skipped_resume": 0,
            "stores_cleaned": 0,
            "stores_retained": 0,
            "total": len(selected),
        }
    failures = 0
    for lb_id in selected:
        json_path = streamlit_experiment_json_path(output_dir, lb_id)
        item = next(
            (row for row in collected if str(row.get("lecturebank_id") or "") == str(lb_id)),
            None,
        )
        if item is None:
            print(
                f"  {lb_id}: no collected metrics row; not running MAS/AXIS",
                file=log,
            )
            failures += 1
            continue
        if not json_path.is_file() or json_path.stat().st_size == 0:
            print(
                f"  {lb_id}: saved experiment JSON missing at {json_path}; "
                "concept-change-from-saved will not generate or replay",
                file=log,
            )
            item["concept_change_unavailable"] = True
            failures += 1
            continue
        try:
            saved = load_json(json_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  {lb_id}: could not read saved experiment JSON ({exc})", file=log)
            failures += 1
            continue
        if not payload_is_experiment(saved):
            print(
                f"  {lb_id}: saved JSON is not an experiment payload; not evaluable",
                file=log,
            )
            item["concept_change_unavailable"] = True
            failures += 1
            continue
        _apply_concept_change_evaluation(
            payload=saved,
            extracted=item,
            lecturebank_id=str(lb_id),
            args=args,
            gen_cfg=gen_cfg,
            metrics_dir=metrics_dir,
            log=log,
        )
        try:
            atomic_write_json(json_path, saved)
        except Exception as exc:  # noqa: BLE001
            print(f"  {lb_id}: could not write assessments back to JSON ({exc})", file=log)
            failures += 1
    persist(output_dir, metrics_dir, manifest_path, manifest, collected)
    return {
        "failures": failures,
        "skipped_resume": 0,
        "stores_cleaned": 0,
        "stores_retained": 0,
        "total": len(selected),
    }


def run_outcome_criteria_from_saved(
    *,
    selected: Sequence[str],
    output_dir: Path,
    metrics_dir: Path,
    collected: List[Dict[str, Any]],
    persist: Callable[..., None],
    manifest_path: Path,
    manifest: Dict[str, Any],
    log: TextIO,
) -> Dict[str, int]:
    """Export stored criterion deltas from saved experiment JSON only.

    Never runs RAG, MAS, AXIS interrogation, replay, or an LLM evaluator.
    """
    from mas_langgraph.evaluation.outcome_criteria import extract_outcome_criteria_changes

    if not collected:
        print(
            "outcome-criteria-from-saved: no persisted document metrics in the output folder; "
            "not running MAS/AXIS. Re-run evaluation first or point --output-dir at an existing run.",
            file=log,
        )
        return {
            "failures": len(selected),
            "skipped_resume": 0,
            "stores_cleaned": 0,
            "stores_retained": 0,
            "total": len(selected),
        }
    failures = 0
    for lb_id in selected:
        json_path = streamlit_experiment_json_path(output_dir, lb_id)
        item = next(
            (row for row in collected if str(row.get("lecturebank_id") or "") == str(lb_id)),
            None,
        )
        if item is None:
            print(
                f"  {lb_id}: no collected metrics row; not running MAS/AXIS",
                file=log,
            )
            failures += 1
            continue
        if not json_path.is_file() or json_path.stat().st_size == 0:
            print(
                f"  {lb_id}: saved experiment JSON missing at {json_path}; "
                "outcome-criteria-from-saved will not generate or replay",
                file=log,
            )
            failures += 1
            continue
        try:
            saved = load_json(json_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  {lb_id}: could not read saved experiment JSON ({exc})", file=log)
            failures += 1
            continue
        if not payload_is_experiment(saved):
            print(
                f"  {lb_id}: saved JSON is not an experiment payload; not evaluable",
                file=log,
            )
            failures += 1
            continue
        extracted = extract_outcome_criteria_changes(saved, lecturebank_id=str(lb_id))
        item["outcome_criteria_changes"] = list(extracted.get("rows") or [])
        item["outcome_criteria_diagnostics"] = list(extracted.get("diagnostics") or [])
        print(
            f"  {lb_id}: wrote {len(item['outcome_criteria_changes'])} outcome-criterion rows",
            file=log,
        )
    persist(output_dir, metrics_dir, manifest_path, manifest, collected)
    return {
        "failures": failures,
        "skipped_resume": 0,
        "stores_cleaned": 0,
        "stores_retained": 0,
        "total": len(selected),
    }


def run_saquet_from_saved(
    *,
    selected: Sequence[str],
    args: Any,
    output_dir: Path,
    metrics_dir: Path,
    collected: List[Dict[str, Any]],
    persist: Callable[..., None],
    manifest_path: Path,
    manifest: Dict[str, Any],
    log: TextIO,
) -> Dict[str, int]:
    """Evaluate SAQUET from saved experiment JSON only.

    Never runs RAG, MAS, AXIS interrogation, or replay.
    """
    from mas_langgraph.evaluation.saquet_eval import load_persisted_factual_quiz

    args.evaluate_saquet = True
    if not collected:
        print(
            "saquet-from-saved: no persisted document metrics in the output folder; "
            "not running MAS/AXIS. Re-run evaluation first or point --output-dir at an existing run.",
            file=log,
        )
        return {
            "failures": len(selected),
            "skipped_resume": 0,
            "stores_cleaned": 0,
            "stores_retained": 0,
            "total": len(selected),
        }
    failures = 0
    for lb_id in selected:
        item = next(
            (row for row in collected if str(row.get("lecturebank_id") or "") == str(lb_id)),
            None,
        )
        if item is None:
            print(
                f"  {lb_id}: no collected metrics row; not running MAS/AXIS",
                file=log,
            )
            failures += 1
            continue
        quiz = load_persisted_factual_quiz(metrics_dir, str(lb_id))
        payload: Any = {}
        if quiz is None:
            json_path = streamlit_experiment_json_path(output_dir, lb_id)
            if not json_path.is_file() or json_path.stat().st_size == 0:
                print(
                    f"  {lb_id}: saved factual quiz and experiment JSON missing "
                    f"(looked for metrics/saquet_factual_quiz.json and {json_path}); "
                    "saquet-from-saved will not generate or replay",
                    file=log,
                )
                failures += 1
                continue
            try:
                saved = load_json(json_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  {lb_id}: could not read saved experiment JSON ({exc})", file=log)
                failures += 1
                continue
            if not payload_is_experiment(saved):
                print(
                    f"  {lb_id}: saved JSON is not an experiment payload; not evaluable",
                    file=log,
                )
                failures += 1
                continue
            payload = saved
        else:
            print(
                f"  {lb_id}: reusing metrics/saquet_factual_quiz.json "
                "(not loading experiment JSON)",
                file=log,
            )
        _apply_saquet_evaluation(
            payload=payload,
            extracted=item,
            lecturebank_id=str(lb_id),
            args=args,
            metrics_dir=metrics_dir,
            log=log,
        )
    persist(output_dir, metrics_dir, manifest_path, manifest, collected)
    return {
        "failures": failures,
        "skipped_resume": 0,
        "stores_cleaned": 0,
        "stores_retained": 0,
        "total": len(selected),
    }
