"""Generate MCQs and AXIS explanations from a prepared presentation index."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from mas_langgraph.domains.quiz.question_limit import (
    QUESTION_LIMIT_SEMANTICS,
    accepted_final_question_count,
    build_generation_limit_report,
)
from mas_langgraph.experiment.export import (
    dump_explanation_experiment,
    dump_explanation_experiment_markdown,
    experiment_to_dict,
)
from mas_langgraph.experiment.timing import ExperimentTimingCollector, timing_scope
from mas_langgraph.input.orchestrator_config import OrchestratorConfig
from mas_langgraph.input.profile import DEFAULT_PROFILE
from mas_langgraph.input.rag_config import RAGConfig

from .ids import file_sha256
from .manifest import (
    generation_config_fingerprint,
    generation_fingerprints_compatible,
    rag_config_fingerprint,
)


class StaleIndexError(RuntimeError):
    """Prepared index is missing, OCR-blocked, or does not match the source/config."""


def resolve_output_path(output_dir: Path, stored: str) -> Path:
    stored_path = Path(stored)
    if stored_path.is_absolute():
        return stored_path
    return (Path(output_dir) / stored_path).resolve()


def profile_with_question_limit(max_final_questions: int):
    """Per-run MAS profile; does not mutate DEFAULT_PROFILE or module constants."""
    orch = replace(
        DEFAULT_PROFILE.orchestrator_config or OrchestratorConfig(),
        max_final_questions=int(max_final_questions),
    )
    return replace(DEFAULT_PROFILE, orchestrator_config=orch)


def validate_prepared_record(
    record: Dict[str, Any],
    *,
    input_dir: Path,
    output_dir: Path,
    expected_config: Dict[str, Any],
) -> Path:
    """Return the isolated index directory or raise ``StaleIndexError``."""
    if record.get("rag_status") == "requires_ocr":
        raise StaleIndexError(
            f"{record.get('document_id')}: extraction requires OCR; generation is blocked."
        )
    if record.get("rag_status") != "success":
        raise StaleIndexError(
            f"{record.get('document_id')}: RAG status is {record.get('rag_status')!r}; "
            "run --stage rag first."
        )
    if str(record.get("input_source_type") or "") == "extraction_jsonl":
        stored_jsonl = str(record.get("jsonl_path") or record.get("input_source_path") or "")
        if stored_jsonl:
            jsonl = Path(stored_jsonl)
            if not jsonl.is_file():
                alt = Path(input_dir) / jsonl.name
                if not alt.is_file():
                    raise StaleIndexError(
                        f"{record.get('document_id')}: extraction JSONL missing at {jsonl}"
                    )
    else:
        src = Path(input_dir) / str(record.get("relative_path") or "")
        if not src.is_file():
            raise StaleIndexError(
                f"{record.get('document_id')}: source PDF missing at {src}"
            )
        current_hash = file_sha256(src)
        if current_hash != str(record.get("source_hash") or ""):
            raise StaleIndexError(
                f"{record.get('document_id')}: source hash changed; re-run --stage rag "
                f"(stored {record.get('source_hash')}, current {current_hash})."
            )
    stored_cfg = rag_config_fingerprint(
        {
            "embedding_model": record.get("embedding_model"),
            "chunk_size": record.get("chunk_size"),
            "chunk_overlap": record.get("chunk_overlap"),
            "chunk_by_page": record.get("chunk_by_page", True),
        }
    )
    want_cfg = rag_config_fingerprint(expected_config)
    if stored_cfg != want_cfg:
        raise StaleIndexError(
            f"{record.get('document_id')}: RAG config mismatch; re-run --stage rag. "
            f"stored={stored_cfg} expected={want_cfg}"
        )
    index_dir = resolve_output_path(output_dir, str(record.get("index_dir") or ""))
    if not index_dir.exists() or not any(index_dir.iterdir()):
        raise StaleIndexError(
            f"{record.get('document_id')}: index missing at {index_dir}"
        )
    return index_dir


@dataclass
class InMemoryGenerationResult:
    """Factual workflow + explanation experiment without requiring disk export."""

    record: Dict[str, Any]
    experiment: Any = None
    experiment_payload: Optional[Dict[str, Any]] = None
    checkpoint_index: Optional[Dict[str, Any]] = None
    generation_diag: Optional[Dict[str, Any]] = None
    collector: Any = None
    accepted_count: int = 0
    cap_reached: bool = False
    wf_s: Optional[float] = None
    explanation_s: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def serialize_checkpoint_index(source: Any) -> Dict[str, Any]:
    """Chronological factual checkpoint aliases (``c_0``, ``c_1``, …)."""
    idx = source
    if source is None:
        return {"records": [], "count": 0}
    if hasattr(source, "orchestrator") or hasattr(source, "get_checkpoint_index"):
        idx = _checkpoint_index_from_backbone(source)
    records_src = getattr(idx, "records", None)
    if records_src is None and isinstance(idx, dict):
        records_src = idx.get("records")
    out_records = []
    for i, rec in enumerate(list(records_src or [])):
        if isinstance(rec, dict):
            alias = rec.get("alias") or f"c_{i}"
            out_records.append(
                {
                    "alias": alias,
                    "checkpoint_id": rec.get("checkpoint_id"),
                    "index": i,
                    "superstep": rec.get("superstep"),
                }
            )
        else:
            out_records.append(
                {
                    "alias": getattr(rec, "alias", None) or f"c_{i}",
                    "checkpoint_id": getattr(rec, "checkpoint_id", None),
                    "index": i,
                    "superstep": getattr(rec, "superstep", None),
                }
            )
    return {"records": out_records, "count": len(out_records)}


def _checkpoint_index_from_backbone(bb: Any) -> Any:
    orch = getattr(bb, "orchestrator", None)
    for obj in (orch, bb):
        if obj is None:
            continue
        getter = getattr(obj, "get_checkpoint_index", None)
        if not callable(getter):
            continue
        try:
            return getter()
        except TypeError:
            run_id = ""
            summary = getattr(bb, "last_summary", None)
            if isinstance(summary, dict):
                run_id = str(summary.get("run_id") or "")
            factual = getattr(bb, "factual_run_result", None) or getattr(
                obj, "_last_run_id", None
            )
            if not run_id and factual is not None:
                run_id = str(getattr(factual, "run_id", factual) or "")
            try:
                return getter(run_id)
            except Exception:
                continue
        except Exception:
            continue
    factual = getattr(bb, "factual_run_result", None)
    if factual is not None:
        return getattr(factual, "checkpoint_index", None)
    return None


def run_factual_and_explanation_in_memory(
    *,
    record: Dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    rag_config_values: Dict[str, Any],
    generation_config: Dict[str, Any],
    backbone_factory: Optional[Callable[..., Any]] = None,
    build_effect_reference_set: bool = False,
    reference_max_interventions_per_question: Optional[int] = None,
    reference_resume_rows: Optional[Sequence[Any]] = None,
    reference_persist_callback: Optional[Callable[..., None]] = None,
    lecturebank_id: str = "",
    reuse_hybrid_add_source_pool: bool = True,
    resume_reference_sweep: bool = False,
    after_axis_metrics_callback: Optional[Callable[..., None]] = None,
    reference_progress_csv: Optional[Path] = None,
) -> InMemoryGenerationResult:
    """Run factual LangGraph MAS + AXIS experiment and keep the result in memory.

    Does not write ``experiment.json``, participant markdown, or timing CSV.
    """
    out = dict(record)
    gen_fp = generation_config_fingerprint(generation_config)
    max_final = int(generation_config["max_questions"])
    max_explain = generation_config.get("max_explain_questions")
    profile = profile_with_question_limit(max_final)

    try:
        index_dir = validate_prepared_record(
            out,
            input_dir=input_dir,
            output_dir=output_dir,
            expected_config=rag_config_values,
        )
    except StaleIndexError as exc:
        out["generate_status"] = "blocked"
        out["errors"] = list(out.get("errors") or []) + [str(exc)]
        return InMemoryGenerationResult(record=out)

    t0 = time.perf_counter()
    cfg = RAGConfig(
        pdf_id=str(record["document_id"]),
        vector_store_path=str(index_dir.resolve()),
        embedding_model=str(rag_config_values.get("embedding_model") or record.get("embedding_model")),
        llm_model=str(generation_config.get("llm_model") or "gpt-4.1-mini-2025-04-14"),
        model_type=str(generation_config.get("model_type") or "openai"),
        temperature=float(generation_config.get("temperature") or 0.0),
    )
    factory = backbone_factory or _default_backbone
    if bool(resume_reference_sweep) and bool(build_effect_reference_set):
        return _resume_effect_reference_sweep(
            record=out,
            input_dir=input_dir,
            output_dir=output_dir,
            rag_config_values=rag_config_values,
            generation_config=generation_config,
            cfg=cfg,
            factory=factory,
            profile=profile,
            build_effect_reference_set=True,
            reference_max_interventions_per_question=reference_max_interventions_per_question,
            reference_resume_rows=reference_resume_rows,
            reference_persist_callback=reference_persist_callback,
            lecturebank_id=str(lecturebank_id or record.get("lecturebank_id") or ""),
            t0=t0,
            gen_fp=gen_fp,
            after_axis_metrics_callback=after_axis_metrics_callback,
            reference_progress_csv=reference_progress_csv,
        )
    collector = ExperimentTimingCollector()
    bb = None
    hybrid_add_source_report = None
    with timing_scope(collector):
        collector.start_experiment()
        bb = _construct_backbone(factory, cfg, profile)
        t_wf = time.perf_counter()
        summary = bb.run_full_workflow()
        wf_s = time.perf_counter() - t_wf
        collector.record_experiment_phase("factual_workflow_s", wf_s)
        run_id = str(summary.get("run_id") or "")
        collector.run_id = run_id
        generation_diag = _generation_diagnostics(bb, summary, max_final)
        accepted_count = int(generation_diag["final_accepted_questions"])
        cap_reached = bool(generation_diag["question_cap_reached"])
        eligible = list(
            bb.list_explanation_experiment_refs(include_historical_versions=False)
        )
        # Explanation cap is independent of the factual accepted-MCQ target.
        if max_explain is not None:
            eligible = eligible[: int(max_explain)]
        if not eligible:
            out["generate_status"] = "mcq_failed"
            out["errors"] = list(out.get("errors") or []) + [
                "No eligible final MCQs; refusing to overwrite prior results."
            ]
            out["factual_workflow_s"] = round(time.perf_counter() - t_wf, 6)
            out["generate_duration_s"] = round(time.perf_counter() - t0, 6)
            out["run_id"] = run_id
            out.update(_manifest_limit_fields(generation_diag, explained=0, max_explain=max_explain))
            out["generation_config"] = gen_fp
            ckpt = serialize_checkpoint_index(bb)
            del bb
            return InMemoryGenerationResult(
                record=out,
                checkpoint_index=ckpt,
                generation_diag=generation_diag,
                collector=collector,
                accepted_count=accepted_count,
                cap_reached=cap_reached,
                wf_s=wf_s,
            )

        intervention_registry = None
        hybrid_add_source_report = None
        if bool(generation_config.get("hybrid_add_from_source")):
            from mas_langgraph.axis.hybrid_add_source import (
                DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY,
                DEFAULT_HYBRID_ADD_SOURCE_LIMIT,
                DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS,
                DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA,
                POOL_FILENAME,
                corrupt_frozen_pool_report,
                embed_fn_from_backbone,
                index_chunk_count_from_backbone,
                is_failed_prep_manifest,
                load_hybrid_add_source_pool_result,
                persist_hybrid_add_source_pool,
                prepare_hybrid_add_source_registry,
                search_fn_from_backbone,
                should_persist_hybrid_add_source_pool,
            )

            lid = str(lecturebank_id or record.get("lecturebank_id") or "")
            doc_id = str(record.get("document_id") or getattr(bb, "pdf_id", "") or "")
            raw_limit = generation_config.get("hybrid_add_source_limit")
            src_limit = (
                int(raw_limit)
                if raw_limit not in (None, "")
                else int(DEFAULT_HYBRID_ADD_SOURCE_LIMIT)
            )
            raw_k = generation_config.get("hybrid_add_source_k_per_query")
            src_k = (
                int(raw_k)
                if raw_k not in (None, "")
                else int(DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY)
            )
            raw_hits = generation_config.get("hybrid_add_source_max_search_hits")
            src_hits = (
                int(raw_hits)
                if raw_hits not in (None, "")
                else int(DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS)
            )
            raw_lambda = generation_config.get("hybrid_add_source_mmr_lambda")
            src_lambda = (
                float(raw_lambda)
                if raw_lambda not in (None, "")
                else float(DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA)
            )
            frozen_path = (
                Path(output_dir) / f"presentation_{doc_id}" / POOL_FILENAME
            )
            frozen = None
            skip_source_prep = False
            if reuse_hybrid_add_source_pool:
                loaded = load_hybrid_add_source_pool_result(frozen_path)
                if loaded.status == "corrupt":
                    skip_source_prep = True
                    intervention_registry = None
                    hybrid_add_source_report = corrupt_frozen_pool_report(
                        path=loaded.path,
                        reason=loaded.reason,
                        document_id=doc_id,
                        lecturebank_id=lid,
                        run_id=run_id,
                        source_hash=str(record.get("source_hash") or ""),
                    )
                    warn = list(out.get("warnings") or [])
                    warn.append(
                        "hybrid_add_from_source_corrupt_pool: "
                        f"{loaded.path}: {loaded.reason}; "
                        "factual MAS result preserved; explanation continues without "
                        "external ADD; rebuild with --overwrite or a fresh output directory"
                    )
                    out["warnings"] = warn
                elif loaded.status == "loaded":
                    if is_failed_prep_manifest(loaded.manifest):
                        frozen = None
                    else:
                        frozen = loaded.manifest
            if not skip_source_prep:
                try:
                    intervention_registry, hybrid_add_source_report = (
                        prepare_hybrid_add_source_registry(
                            state=getattr(bb, "state", None),
                            search_fn=search_fn_from_backbone(bb),
                            embed_fn=embed_fn_from_backbone(bb),
                            document_id=doc_id,
                            lecturebank_id=lid,
                            run_id=run_id,
                            limit=src_limit,
                            embedding_model=str(
                                rag_config_values.get("embedding_model") or ""
                            ),
                            k_per_query=src_k,
                            max_search_hits=src_hits,
                            mmr_lambda=src_lambda,
                            source_hash=str(record.get("source_hash") or ""),
                            rag_config_fingerprint=rag_config_fingerprint(rag_config_values),
                            index_chunk_count=index_chunk_count_from_backbone(bb),
                            frozen_manifest=frozen,
                        )
                    )
                    if should_persist_hybrid_add_source_pool(hybrid_add_source_report):
                        persist_hybrid_add_source_pool(
                            frozen_path, hybrid_add_source_report
                        )
                except Exception as exc:  # noqa: BLE001
                    warn = list(out.get("warnings") or [])
                    warn.append(f"hybrid_add_from_source_failed: {exc}")
                    out["warnings"] = warn
                    intervention_registry = None
                    hybrid_add_source_report = {
                        "empty_reason": "preparation_failed",
                        "retrieval_error": f"{type(exc).__name__}: {exc}",
                        "source_preparation_status": "failed",
                        "external_add_pool_available": False,
                        "requested_hybrid_add_from_source": True,
                        "lecturebank_id": lid,
                        "document_id": doc_id,
                        "run_id": run_id,
                        "candidates": [],
                        "events": [
                            {
                                "lecturebank_id": lid,
                                "document_id": doc_id,
                                "run_id": run_id,
                                "event_type": "retrieval_failure",
                                "detail": f"{type(exc).__name__}: {exc}",
                            }
                        ],
                        "candidates_registered": 0,
                    }

        t_exp = time.perf_counter()
        after_axis = None
        if bool(build_effect_reference_set):
            after_axis = _make_after_axis_snapshot_callback(
                backbone_holder={"bb": bb},
                output_dir=Path(output_dir),
                record=out,
                rag_config_values=rag_config_values,
                generation_config=generation_config,
                gen_fp=gen_fp,
                lecturebank_id=str(lecturebank_id or record.get("lecturebank_id") or ""),
                run_id=run_id,
                persist_rows_callback=reference_persist_callback,
                after_axis_metrics_callback=after_axis_metrics_callback,
                csv_path=reference_progress_csv,
            )
        experiment = bb.generate_explanation_experiment(
            question=str(generation_config.get("question") or "Why was this question selected?"),
            experiment_seed=int(generation_config.get("seed") or 42),
            include_historical_versions=False,
            candidate_refs=eligible,
            max_rounds=int(generation_config.get("max_rounds") or 5),
            axis_interrogation_mode=str(
                generation_config.get("axis_interrogation_mode") or "constrained"
            ),
            build_effect_reference_set=bool(build_effect_reference_set),
            reference_max_interventions_per_question=reference_max_interventions_per_question,
            reference_resume_rows=reference_resume_rows,
            reference_persist_callback=reference_persist_callback,
            lecturebank_id=str(lecturebank_id or record.get("lecturebank_id") or ""),
            diagnose_hybrid_add=bool(generation_config.get("diagnose_hybrid_add")),
            intervention_registry=intervention_registry,
            after_axis_callback=after_axis,
        )
        collector.end_experiment()
        object.__setattr__(experiment, "timing", collector.to_dict())
        explanation_s = time.perf_counter() - t_exp

    if not experiment.cases:
        out["generate_status"] = "explanation_failed"
        out["errors"] = list(out.get("errors") or []) + [
            "Explanation experiment produced no cases; prior results kept."
        ]
        out["run_id"] = run_id
        out["generate_duration_s"] = round(time.perf_counter() - t0, 6)
        out.update(_manifest_limit_fields(generation_diag, explained=0, max_explain=max_explain))
        out["generation_config"] = gen_fp
        ckpt = serialize_checkpoint_index(bb)
        del bb
        return InMemoryGenerationResult(
            record=out,
            checkpoint_index=ckpt,
            generation_diag=generation_diag,
            collector=collector,
            accepted_count=accepted_count,
            cap_reached=cap_reached,
            wf_s=wf_s,
            explanation_s=explanation_s,
        )

    ckpt = serialize_checkpoint_index(bb)
    payload = experiment_to_dict(
        experiment,
        include_blinding=False,
        include_raw=True,
    )
    from mas_langgraph.domains.quiz.packaging import serialize_final_accepted_quiz

    factual_quiz = serialize_final_accepted_quiz(getattr(bb, "state", None))
    factual_quiz["run_id"] = run_id
    payload["factual_quiz"] = factual_quiz
    source_payload = None
    if hybrid_add_source_report is not None:
        source_payload = (
            hybrid_add_source_report.to_dict()
            if hasattr(hybrid_add_source_report, "to_dict")
            else dict(hybrid_add_source_report)
        )
        payload["hybrid_add_source"] = source_payload
    extras: Dict[str, Any] = {
        "t0": t0,
        "gen_fp": gen_fp,
        "max_explain": max_explain,
        "run_id": run_id,
        "factual_quiz": factual_quiz,
    }
    if source_payload is not None:
        extras["hybrid_add_source"] = source_payload
    ref = getattr(experiment, "_effect_reference", None)
    if ref:
        extras["effect_reference"] = ref
    del bb
    return InMemoryGenerationResult(
        record=out,
        experiment=experiment,
        experiment_payload=payload,
        checkpoint_index=ckpt,
        generation_diag=generation_diag,
        collector=collector,
        accepted_count=accepted_count,
        cap_reached=cap_reached,
        wf_s=wf_s,
        explanation_s=explanation_s,
        extras=extras,
    )


def generate_presentation(
    *,
    record: Dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    rag_config_values: Dict[str, Any],
    generation_config: Dict[str, Any],
    overwrite: bool = False,
    resume: bool = False,
    backbone_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run factual workflow + explanation experiment for one prepared index."""
    out = dict(record)
    gen_fp = generation_config_fingerprint(generation_config)
    pres_dir = Path(output_dir) / f"presentation_{record['document_id']}"
    results_md = pres_dir / "results.md"
    experiment_json = pres_dir / "experiment.json"
    timing_csv = pres_dir / "timing.csv"
    max_explain = generation_config.get("max_explain_questions")

    try:
        validate_prepared_record(
            out,
            input_dir=input_dir,
            output_dir=output_dir,
            expected_config=rag_config_values,
        )
    except StaleIndexError as exc:
        out["generate_status"] = "blocked"
        out["errors"] = list(out.get("errors") or []) + [str(exc)]
        return out

    if (
        resume
        and not overwrite
        and _generate_resume_ok(out, gen_fp, results_md, experiment_json)
    ):
        warnings = list(out.get("warnings") or [])
        if "generate_resumed" not in warnings:
            warnings.append("generate_resumed")
        out["warnings"] = warnings
        return out

    result = run_factual_and_explanation_in_memory(
        record=out,
        input_dir=input_dir,
        output_dir=output_dir,
        rag_config_values=rag_config_values,
        generation_config=generation_config,
        backbone_factory=backbone_factory,
        reuse_hybrid_add_source_pool=not overwrite,
    )
    out = result.record
    experiment = result.experiment
    if experiment is None:
        return out

    dump_explanation_experiment(experiment, str(experiment_json))
    dump_explanation_experiment_markdown(
        experiment,
        str(results_md),
        title=str(record.get("relative_path") or record.get("document_id")),
        document_id=str(record.get("document_id")),
    )
    collector = result.collector
    if collector is not None:
        collector.write_csv(str(timing_csv))

    extras = result.extras or {}
    t0 = float(extras.get("t0") or time.perf_counter())
    run_id = str(extras.get("run_id") or getattr(experiment, "run_id", "") or "")
    generation_diag = result.generation_diag or {}
    cap_reached = bool(result.cap_reached)
    accepted_count = int(result.accepted_count)
    wf_s = float(result.wf_s or 0.0)
    explanation_s = float(result.explanation_s or 0.0)

    axis_failed = [
        c.axis_status
        for c in experiment.cases
        if str(c.axis_status or "").lower() not in {"ok", "success", "completed_with_counterfactual"}
        or (
            not c.axis_causal_evidence_available
            and str(c.axis_status or "").lower()
            in {"failed", "inconclusive", "completed_without_counterfactual"}
        )
    ]
    generate_status = "success"
    warnings = list(out.get("warnings") or [])
    if cap_reached:
        warnings.append("final_question_cap_reached")
    if any(not c.axis_causal_evidence_available for c in experiment.cases):
        warnings.append(
            "One or more AXIS cases have no supported counterfactual effects; "
            "that is not treated as successful causal evidence."
        )
        if all(not c.axis_causal_evidence_available for c in experiment.cases):
            generate_status = "partial"
    if axis_failed and generate_status == "success":
        generate_status = "partial"

    out.update(
        {
            "generate_status": generate_status,
            "results_md": results_md.relative_to(output_dir).as_posix(),
            "experiment_json": experiment_json.relative_to(output_dir).as_posix(),
            "timing_csv": timing_csv.relative_to(output_dir).as_posix(),
            "run_id": run_id,
            "generation_config": gen_fp,
            "factual_workflow_s": round(wf_s, 6),
            "explanation_duration_s": round(explanation_s, 6),
            "generate_duration_s": round(time.perf_counter() - t0, 6),
            "warnings": warnings,
            "question_count": accepted_count,
            "explained_question_count": len(experiment.cases),
            **_manifest_limit_fields(
                generation_diag,
                explained=len(experiment.cases),
                max_explain=max_explain,
            ),
        }
    )
    return out


def _generate_resume_ok(
    record: Dict[str, Any],
    gen_fp: Dict[str, Any],
    results_md: Path,
    experiment_json: Path,
) -> bool:
    if record.get("generate_status") not in {"success", "partial"}:
        return False
    stored = dict(record.get("generation_config") or {})
    if stored.get("question_limit_semantics") != QUESTION_LIMIT_SEMANTICS:
        return False
    if not generation_fingerprints_compatible(stored, gen_fp):
        return False
    if not results_md.is_file() or not experiment_json.is_file():
        return False
    if results_md.stat().st_size == 0 or experiment_json.stat().st_size == 0:
        return False
    return True


def _hybrid_pool_fingerprint(output_dir: Path, document_id: str) -> str:
    from mas_langgraph.axis.hybrid_add_source import POOL_FILENAME

    path = Path(output_dir) / f"presentation_{document_id}" / POOL_FILENAME
    if not path.is_file() or path.stat().st_size <= 0:
        return ""
    try:
        return file_sha256(path)
    except OSError:
        return ""


def _make_after_axis_snapshot_callback(
    *,
    backbone_holder: Dict[str, Any],
    output_dir: Path,
    record: Mapping[str, Any],
    rag_config_values: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    gen_fp: Mapping[str, Any],
    lecturebank_id: str,
    run_id: str,
    persist_rows_callback: Optional[Callable[..., None]],
    after_axis_metrics_callback: Optional[Callable[..., None]],
    csv_path: Optional[Path] = None,
) -> Callable[..., None]:
    from mas_langgraph.evaluation.reference_sweep_snapshot import (
        build_identity,
        save_reference_sweep_snapshot,
        snapshot_dir,
        wrap_persist_with_progress,
    )

    def _after_axis(experiment: Any, service: Any) -> None:
        bb = backbone_holder.get("bb")
        if bb is None:
            raise RuntimeError("Factual backbone missing while snapshotting AXIS results")
        orch = bb._require_langgraph_orchestrator()
        universes = dict(getattr(service, "_frozen_universes", None) or {})
        planned = sum(len(list(v or [])) for v in universes.values())
        doc_id = str(record.get("document_id") or "")
        identity = build_identity(
            run_id=str(run_id or getattr(experiment, "run_id", "") or ""),
            lecturebank_id=str(lecturebank_id or ""),
            document_id=doc_id,
            source_hash=str(record.get("source_hash") or record.get("pdf_sha256") or ""),
            rag_config=rag_config_fingerprint(rag_config_values),
            generation_config=dict(gen_fp or generation_config_fingerprint(generation_config)),
            axis_interrogation_mode=str(
                generation_config.get("axis_interrogation_mode")
                or getattr(experiment, "axis_interrogation_mode", "")
                or "hybrid"
            ),
            reference_max_interventions_per_question=getattr(
                service, "reference_max_interventions_per_question", None
            ),
            planned_intervention_count=planned,
            planned_question_count=len(universes),
            candidate_refs=list(universes.keys()),
            hybrid_add_pool_fingerprint=_hybrid_pool_fingerprint(output_dir, doc_id),
        )
        payload = experiment_to_dict(
            experiment, include_blinding=False, include_raw=True
        )
        directory = snapshot_dir(output_dir, doc_id)
        save_reference_sweep_snapshot(
            directory,
            identity=identity,
            checkpointer=orch.checkpointer,
            replay_executor=orch.replay_executor,
            factual_run_result=orch.get_factual_run_result(),
            axis_results=dict(getattr(service, "_axis_results_by_ref", None) or {}),
            frozen_universes=universes,
            intervention_registry=getattr(service, "intervention_registry", None),
            intervention_alternatives=getattr(service, "intervention_alternatives", None),
            axis_noop_cache=getattr(
                getattr(service, "_axis_service", None), "noop_cache", None
            ),
            experiment=experiment,
            cooperative_state=getattr(bb, "state", None),
            experiment_payload=payload,
        )
        service.reference_persist_callback = wrap_persist_with_progress(
            directory=directory,
            identity=identity,
            frozen_universes=universes,
            persist_rows_callback=persist_rows_callback,
            csv_path=csv_path,
        )
        if callable(after_axis_metrics_callback):
            after_axis_metrics_callback(experiment, service)

    return _after_axis


def _resume_effect_reference_sweep(
    *,
    record: Dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    rag_config_values: Dict[str, Any],
    generation_config: Dict[str, Any],
    cfg: RAGConfig,
    factory: Callable[..., Any],
    profile: Any,
    build_effect_reference_set: bool,
    reference_max_interventions_per_question: Optional[int],
    reference_resume_rows: Optional[Sequence[Any]],
    reference_persist_callback: Optional[Callable[..., None]],
    lecturebank_id: str,
    t0: float,
    gen_fp: Dict[str, Any],
    after_axis_metrics_callback: Optional[Callable[..., None]],
    reference_progress_csv: Optional[Path] = None,
) -> InMemoryGenerationResult:
    from mas_langgraph.experiment.service import build_experiment_service_from_backbone
    from mas_langgraph.evaluation.reference_sweep_snapshot import (
        build_identity,
        load_reference_sweep_snapshot,
        snapshot_dir,
        validate_identity,
        wrap_persist_with_progress,
    )

    del input_dir
    del build_effect_reference_set
    collector = ExperimentTimingCollector()
    doc_id = str(record.get("document_id") or "")
    directory = snapshot_dir(output_dir, doc_id)
    loaded = load_reference_sweep_snapshot(directory)
    identity = loaded["identity"]
    runtime = loaded["runtime"]
    current = build_identity(
        run_id=str(identity.run_id),
        lecturebank_id=str(lecturebank_id or identity.lecturebank_id),
        document_id=doc_id or identity.document_id,
        source_hash=str(record.get("source_hash") or record.get("pdf_sha256") or ""),
        rag_config=rag_config_fingerprint(rag_config_values),
        generation_config=dict(gen_fp),
        axis_interrogation_mode=str(
            generation_config.get("axis_interrogation_mode") or identity.axis_interrogation_mode
        ),
        reference_max_interventions_per_question=reference_max_interventions_per_question,
        planned_intervention_count=identity.planned_intervention_count,
        planned_question_count=identity.planned_question_count,
        candidate_refs=identity.candidate_refs,
        hybrid_add_pool_fingerprint=_hybrid_pool_fingerprint(output_dir, doc_id),
    )
    validate_identity(identity, current)
    run_id = str(identity.run_id)
    bb = _construct_backbone(factory, cfg, profile)
    orch = bb._require_langgraph_orchestrator()
    restored = orch.restore_factual_runtime(
        run_id=run_id,
        factual_run_result=runtime.get("factual_run_result"),
        checkpointer_state=runtime.get("checkpointer_state"),
        replay_executor_state=runtime.get("replay_executor_state"),
        cooperative_state=runtime.get("cooperative_state"),
    )
    if restored is None:
        from mas_langgraph.evaluation.reference_sweep_snapshot import (
            MissingReferenceSweepSnapshotError,
        )

        raise MissingReferenceSweepSnapshotError(
            "Reference-sweep snapshot is missing the factual WorkflowRunResult."
        )
    if runtime.get("cooperative_state") is not None:
        bb.state = runtime["cooperative_state"]
    experiment = runtime.get("experiment")
    if experiment is None:
        from mas_langgraph.evaluation.reference_sweep_snapshot import (
            MissingReferenceSweepSnapshotError,
        )

        raise MissingReferenceSweepSnapshotError(
            "Reference-sweep snapshot is missing the AXIS experiment; "
            "refusing to regenerate MCQs or rerun AXIS selection."
        )
    service = build_experiment_service_from_backbone(
        bb,
        max_rounds=int(generation_config.get("max_rounds") or 5),
        axis_interrogation_mode=str(
            generation_config.get("axis_interrogation_mode") or "hybrid"
        ),
        build_effect_reference_set=True,
        reference_max_interventions_per_question=reference_max_interventions_per_question,
        reference_resume_rows=reference_resume_rows,
        reference_persist_callback=wrap_persist_with_progress(
            directory=directory,
            identity=identity,
            frozen_universes=runtime.get("frozen_universes") or {},
            persist_rows_callback=reference_persist_callback,
            csv_path=reference_progress_csv,
        ),
        lecturebank_id=str(lecturebank_id or identity.lecturebank_id),
        intervention_registry=runtime.get("intervention_registry"),
    )
    service.apply_resume_axis_state(
        axis_results=runtime.get("axis_results"),
        frozen_universes=runtime.get("frozen_universes"),
        intervention_registry=runtime.get("intervention_registry"),
        intervention_alternatives=runtime.get("intervention_alternatives"),
        axis_noop_cache=runtime.get("axis_noop_cache"),
        experiment=experiment,
    )
    service.run_deferred_reference_sweeps()
    service.attach_effect_reference(experiment)
    collector.end_experiment()
    object.__setattr__(experiment, "timing", collector.to_dict())
    generation_diag = {
        "requested_max_questions": int(generation_config.get("max_questions") or 0),
        "final_accepted_questions": len(getattr(experiment, "cases", ()) or ()),
        "question_cap_reached": False,
        "generation_exhausted": False,
        "generation_exhaustion_reasons": [],
        "generation_exhausted_reason": None,
    }
    accepted_count = len(getattr(experiment, "cases", ()) or ())
    ckpt = serialize_checkpoint_index(bb)
    payload = experiment_to_dict(experiment, include_blinding=False, include_raw=True)
    from mas_langgraph.domains.quiz.packaging import serialize_final_accepted_quiz

    factual_quiz = serialize_final_accepted_quiz(getattr(bb, "state", None))
    factual_quiz["run_id"] = run_id
    payload["factual_quiz"] = factual_quiz
    extras: Dict[str, Any] = {
        "t0": t0,
        "gen_fp": gen_fp,
        "max_explain": generation_config.get("max_explain_questions"),
        "run_id": run_id,
        "factual_quiz": factual_quiz,
        "resumed_reference_sweep": True,
    }
    ref = getattr(experiment, "_effect_reference", None)
    if ref:
        extras["effect_reference"] = ref
    out = dict(record)
    out["run_id"] = run_id
    out["generate_status"] = "success"
    del bb
    return InMemoryGenerationResult(
        record=out,
        experiment=experiment,
        experiment_payload=payload,
        checkpoint_index=ckpt,
        generation_diag=generation_diag,
        collector=collector,
        accepted_count=accepted_count,
        cap_reached=False,
        wf_s=0.0,
        explanation_s=0.0,
        extras=extras,
    )


def _construct_backbone(factory: Callable[..., Any], cfg: RAGConfig, profile: Any) -> Any:
    try:
        return factory(cfg, profile=profile)
    except TypeError:
        return factory(cfg)


def _generation_diagnostics(bb: Any, summary: Dict[str, Any], max_final: int) -> Dict[str, Any]:
    """Prefer packaged-quiz diagnostics from the factual workflow."""
    state = getattr(bb, "state", None)
    stored = {}
    if state is not None:
        stored = dict(
            (getattr(state, "mcq_coverage_summary", None) or {}).get("generation_limit")
            or {}
        )
    if not stored:
        stored = {
            key: summary.get(key)
            for key in (
                "requested_max_questions",
                "final_accepted_questions",
                "question_cap_reached",
                "generation_exhausted",
                "generation_exhaustion_reasons",
                "generation_exhausted_reason",
            )
            if key in summary
        }
    accepted = stored.get("final_accepted_questions")
    if accepted is None:
        accepted = _accepted_count_from_backbone(bb)
    cap_reached = bool(stored.get("question_cap_reached"))
    if "question_cap_reached" not in stored:
        cap_reached = int(accepted) >= int(max_final)
    exhausted = stored.get("generation_exhausted")
    if exhausted is None:
        exhausted = int(accepted) < int(max_final)
    reasons = list(stored.get("generation_exhaustion_reasons") or [])
    if exhausted and not reasons:
        if state is not None:
            try:
                reasons = list(
                    build_generation_limit_report(
                        state,
                        config={"max_final_questions": max_final},
                        packaged_count=int(accepted),
                    ).get("generation_exhaustion_reasons")
                    or []
                )
            except Exception:
                reasons = ["no_supported_generation_opportunities"]
        else:
            reasons = ["no_supported_generation_opportunities"]
    if not exhausted:
        reasons = []
    single = reasons[0] if len(reasons) == 1 else None
    return {
        "requested_max_questions": int(max_final),
        "final_accepted_questions": int(accepted),
        "question_cap_reached": bool(cap_reached),
        "generation_exhausted": bool(exhausted),
        "generation_exhaustion_reasons": reasons,
        "generation_exhausted_reason": stored.get("generation_exhausted_reason", single),
    }


def _manifest_limit_fields(
    generation_diag: Dict[str, Any],
    *,
    explained: int,
    max_explain: Optional[int],
) -> Dict[str, Any]:
    explained_n = int(explained)
    accepted = int(generation_diag["final_accepted_questions"])
    return {
        "max_questions": generation_diag["requested_max_questions"],
        "requested_max_questions": generation_diag["requested_max_questions"],
        "final_accepted_questions": accepted,
        "question_cap_reached": generation_diag["question_cap_reached"],
        "generation_exhausted": generation_diag["generation_exhausted"],
        "generation_exhaustion_reasons": list(
            generation_diag.get("generation_exhaustion_reasons") or []
        ),
        "generation_exhausted_reason": generation_diag.get("generation_exhausted_reason"),
        "max_explain_questions": max_explain,
        "explained_questions": explained_n,
        "explained_question_count": explained_n,
        "generation": {
            "requested_max_questions": generation_diag["requested_max_questions"],
            "final_accepted_questions": accepted,
            "question_cap_reached": generation_diag["question_cap_reached"],
            "generation_exhausted": generation_diag["generation_exhausted"],
            "generation_exhaustion_reasons": list(
                generation_diag.get("generation_exhaustion_reasons") or []
            ),
        },
        "explanation": {
            "max_explain_questions": max_explain,
            "explained_question_count": explained_n,
        },
    }


def _accepted_count_from_backbone(bb: Any) -> int:
    state = getattr(bb, "state", None)
    if state is not None:
        packs = list(getattr(state, "final_quiz_packs", None) or [])
        if packs:
            questions = getattr(packs[-1], "questions", None) or []
            return len(questions)
        try:
            return accepted_final_question_count(state)
        except Exception:
            pass
    refs = list(bb.list_explanation_experiment_refs(include_historical_versions=False))
    return len(refs)


def _default_backbone(cfg: RAGConfig, profile: Any = None) -> Any:
    from mas_langgraph import CooperativeBackbone

    kwargs: Dict[str, Any] = {"orchestrator_mode": "langgraph", "rag_config": cfg}
    if profile is not None:
        kwargs["profile"] = profile
    return CooperativeBackbone(**kwargs)
