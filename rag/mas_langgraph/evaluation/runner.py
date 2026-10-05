"""LectureBank AXIS evaluation runner (batch/metrics layer only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from mas_langgraph.batch.io import atomic_write_json, load_json
from mas_langgraph.batch.manifest import generation_config_fingerprint, rag_config_fingerprint
from mas_langgraph.batch.runner import DEFAULTS, DEFAULT_MAX_QUESTIONS, _positive_int
from mas_langgraph.evaluation.axis_metrics import NA, RECOGNIZED_QUERY_OPS, describe, pct
from mas_langgraph.evaluation.execute import (
    process_documents,
    run_concept_change_from_saved,
    run_outcome_criteria_from_saved,
    run_preflight,
    run_saquet_from_saved,
)
from mas_langgraph.evaluation.lecturebank_meta import (
    REQUIRED_EVAL_ID_COUNT,
    SelectionError,
    catalog_from_extraction_lookups,
    format_id_table,
    load_lecturebank_catalog,
    resolve_selected_ids,
)
from mas_langgraph.evaluation.lifecycle import eval_completed, load_collected_from_metrics
from mas_langgraph.evaluation.quiz_export import collect_quiz_rows, write_quiz_outputs
from mas_langgraph.evaluation.streamlit_export import (
    collect_streamlit_unavailable_ids,
    count_streamlit_json_files,
)
from mas_langgraph.evaluation.pdf_source import document_id_for, local_pdf_available
from mas_langgraph.evaluation.report import (
    build_report,
    overall_metric_rows,
    write_metrics_outputs,
)
from mas_langgraph.input.rag_config import DEFAULT_WORKFLOW_MODEL

MANIFEST_NAME = "lecturebank_eval_manifest.json"
EVAL_RESULT_NAME = "eval_result.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select LectureBank documents by ID, run isolated RAG → factual "
            "LangGraph MAS → AXIS, and write aggregate mechanism-level metrics. "
            "Does not export per-document experiment.json."
        )
    )
    parser.add_argument(
        "--lecturebank-dir",
        default=None,
        help=(
            "Local LectureBank clone containing alldata.tsv (or lecturebank.tsv). "
            "Required unless --jsonl-authoritative."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="lecturebank_axis_eval",
        help="Output folder for PDF cache, indexes, manifest, and metrics/.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Optional local directory of PDFs named by LectureBank ID (e.g. 101.pdf).",
    )
    parser.add_argument(
        "--extraction-jsonl",
        default=None,
        help=(
            "Optional pre-extracted LectureBank JSONL (pdf_extraction_results.jsonl). "
            "Default: preferred over PDF when a unique URL/metadata match has usable "
            "page text. With --jsonl-authoritative, selected IDs are JSONL extraction IDs."
        ),
    )
    parser.add_argument(
        "--jsonl-authoritative",
        action="store_true",
        help=(
            "Use JSONL records as the document identity (id, metadata, page text). "
            "Does not match against lecturebank.tsv. Requires --extraction-jsonl."
        ),
    )
    parser.add_argument(
        "--first-jsonl",
        type=_positive_int,
        default=None,
        help=(
            "With --jsonl-authoritative, select the first N unique JSONL records "
            "that have usable page text, in file order."
        ),
    )
    parser.add_argument(
        "--id",
        action="append",
        default=[],
        dest="id_values",
        help="LectureBank ID, or JSONL extraction ID with --jsonl-authoritative. Repeatable.",
    )
    parser.add_argument(
        "--ids",
        default="",
        help='Comma-separated LectureBank IDs, e.g. "101,205,333".',
    )
    parser.add_argument("--ids-file", default=None, help="Text file of IDs (LectureBank, or JSONL IDs with --jsonl-authoritative).")
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument(
        "--allow-any-count",
        action="store_true",
        help=f"Allow fewer or more than {REQUIRED_EVAL_ID_COUNT} IDs (smoke tests).",
    )
    count_group.add_argument(
        "--expected-count",
        type=_positive_int,
        default=None,
        help=(
            "Require exactly N unique selected IDs. Default without this flag "
            f"or --allow-any-count is {REQUIRED_EVAL_ID_COUNT}."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even if some selected documents failed.",
    )
    parser.add_argument("--list-ids", action="store_true", help="List metadata and exit.")
    parser.add_argument("--search", default="", help="Filter --list-ids by substring.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows for --list-ids.")
    parser.add_argument("--no-download", action="store_true", help="Do not download PDFs from URLs.")
    parser.add_argument(
        "--stage",
        choices=["rag", "evaluate", "all"],
        default="all",
        help="rag = index only; evaluate = MAS+AXIS+metrics from indexes; all = both.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULTS["embedding_model"])
    parser.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"])
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULTS["chunk_overlap"])
    parser.add_argument("--no-page-chunking", action="store_true")
    parser.add_argument("--llm-model", default=DEFAULT_WORKFLOW_MODEL)
    parser.add_argument("--model-type", choices=["openai", "huggingface"], default="openai")
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--max-rounds", type=int, default=DEFAULTS["max_rounds"])
    parser.add_argument(
        "--max-questions",
        type=_positive_int,
        default=DEFAULT_MAX_QUESTIONS,
        help="Factual accepted-MCQ target/cap per document (default 10).",
    )
    parser.add_argument(
        "--max-explain-questions",
        type=_positive_int,
        default=None,
        help="Optional AXIS explanation cap. Default: explain all accepted questions.",
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
    parser.add_argument(
        "--export-quiz",
        action="store_true",
        default=False,
        help=(
            "Optionally write metrics/questions.csv and quizzes.md from the "
            "already-generated experiment payload. Does not regenerate MCQs "
            "or change AXIS interrogation."
        ),
    )
    parser.add_argument(
        "--export-streamlit-json",
        action="store_true",
        default=False,
        help=(
            "Optionally write streamlit/quiz_explanation_experiment_<id>.json "
            "from the existing explanation-experiment payload for the current "
            "Streamlit graph app. Does not regenerate experiments or change AXIS."
        ),
    )
    parser.add_argument(
        "--diagnose-hybrid-add",
        action="store_true",
        default=False,
        help=(
            "Opt-in hybrid ADD vocabulary diagnostics. Writes "
            "metrics/hybrid_add_candidates.csv and "
            "metrics/hybrid_add_boundaries.csv and a report.md summary. "
            "Does not change advertised interventions, prompts, ranking, or "
            "generation fingerprints. Default off."
        ),
    )
    parser.add_argument(
        "--hybrid-add-from-source",
        action="store_true",
        default=False,
        help=(
            "Opt-in: retrieve additional same-document EvidenceChunks from the "
            "PDF index and register them in a separate intervention catalog for "
            "hybrid ADD. Default off. Does not change factual retrieval, force "
            "ADD selection, or expand WHATIF replacements. Included in the AXIS "
            "generation fingerprint."
        ),
    )
    parser.add_argument(
        "--hybrid-add-source-limit",
        type=_positive_int,
        default=None,
        help=(
            "Max extra same-document chunks registered for hybrid ADD "
            "(default 8, matching the hybrid vocabulary cap). Used only with "
            "--hybrid-add-from-source."
        ),
    )
    parser.add_argument(
        "--hybrid-add-source-k-per-query",
        type=_positive_int,
        default=None,
        help=(
            "Retrieval depth per query for optional hybrid ADD source prep "
            "(default 32). Independent of the final pool limit. Used only with "
            "--hybrid-add-from-source."
        ),
    )
    parser.add_argument(
        "--hybrid-add-source-max-search-hits",
        type=_positive_int,
        default=None,
        help=(
            "Max unique retrieval hits considered across queries for hybrid ADD "
            "source prep (default 128). Used only with --hybrid-add-from-source."
        ),
    )
    parser.add_argument(
        "--hybrid-add-source-mmr-lambda",
        type=float,
        default=None,
        help=(
            "MMR relevance weight in [0, 1] for hybrid ADD source selection "
            "(default 0.7). Higher prefers query relevance (inverse distance); "
            "lower prefers low embedding-cosine redundancy vs factual evidence "
            "and already-selected candidates. This balances relevance with "
            "redundancy; it does not guarantee semantic novelty or relevance. "
            "Used only with --hybrid-add-from-source."
        ),
    )
    parser.add_argument(
        "--evaluate-concept-change",
        action="store_true",
        default=False,
        help=(
            "Optionally run an LLM-assessed tested-concept change comparison "
            "on AXIS-selected NOOP vs counterfactual MCQs within K. "
            "Evaluation-only; default off; not causal ground truth. "
            "Does not change AXIS interrogation, replay, or MAS."
        ),
    )
    parser.add_argument(
        "--concept-change-from-saved",
        action="store_true",
        default=False,
        help=(
            "Evaluate tested-concept change from saved experiment JSON and "
            "rewrite concept metrics only. Never runs RAG, MAS, AXIS "
            "interrogation, NOOP/counterfactual replay, or reference sweeps. "
            "Fails if streamlit/quiz_explanation_experiment_<id>.json is missing."
        ),
    )
    parser.add_argument(
        "--outcome-criteria-from-saved",
        action="store_true",
        default=False,
        help=(
            "Export per-criterion AXIS replay values from saved experiment JSON "
            "into metrics/outcome_criteria_changes.csv and the derived "
            "metrics/outcome_criteria_changed.csv. Never runs RAG, MAS, "
            "AXIS interrogation, NOOP/counterfactual replay, or an LLM evaluator. "
            "Fails if streamlit/quiz_explanation_experiment_<id>.json is missing."
        ),
    )
    parser.add_argument(
        "--evaluate-saquet",
        action="store_true",
        default=False,
        help=(
            "Optionally run the supplied SAQUET IWF criteria on final accepted "
            "factual MCQs and write metrics/saquet_by_mas.csv. Default off. "
            "Does not change MAS, AXIS interrogation, or replay. "
            "Does not write a per-question SAQUET CSV."
        ),
    )
    parser.add_argument(
        "--saquet-from-saved",
        action="store_true",
        default=False,
        help=(
            "Evaluate SAQUET from saved factual MCQs and rewrite "
            "metrics/saquet_by_mas.csv only. Prefers metrics/saquet_factual_quiz.json, "
            "falls back to the saved experiment JSON, and reuses valid flags in "
            "metrics/saquet_question_cache.json (retries missing criteria only). "
            "Never runs RAG, MAS, AXIS interrogation, or replay. Fails if neither "
            "the factual-quiz file nor the saved experiment JSON is available."
        ),
    )
    parser.add_argument(
        "--reevaluate-saquet",
        action="store_true",
        default=False,
        help=(
            "Force SAQUET to rerun already successful criterion assessments, "
            "which incurs new API calls. Ordinary continuation retries only "
            "failed or missing criteria. Does not change generation fingerprints."
        ),
    )
    parser.add_argument(
        "--reevaluate-concept-change",
        action="store_true",
        default=False,
        help=(
            "Force concept-change evaluation even when persisted assessments "
            "exist, including retry of previous evaluation_error rows. "
            "Does not change generation fingerprints."
        ),
    )
    parser.add_argument(
        "--build-effect-reference-set",
        action="store_true",
        default=False,
        help=(
            "After normal hybrid AXIS interrogation/explanation, exhaustively "
            "replay the initial hybrid counterfactual vocabulary as a "
            "replay-derived reference set. Evaluation-only; does not change AXIS."
        ),
    )
    parser.add_argument(
        "--reference-max-interventions-per-question",
        type=int,
        default=None,
        help=(
            "Optional smoke-test cap on deterministic reference interventions "
            "per explained MCQ. Incomplete sweeps make Effect Discovery Recall@K NA."
        ),
    )
    parser.add_argument(
        "--cleanup-vector-store",
        action="store_true",
        help=(
            "After metrics and the manifest are persisted, delete this document's "
            "vector store only."
        ),
    )
    parser.add_argument(
        "--temporary-vector-stores",
        action="store_true",
        help=(
            "Build each document's isolated Chroma store in a temporary directory "
            "and delete it after successful metric persistence. Preferred low-disk mode."
        ),
    )
    parser.add_argument(
        "--cleanup-downloaded-pdfs",
        action="store_true",
        help=(
            "After successful completion, delete PDFs this run downloaded into "
            "pdf_cache/. Never deletes --pdf-dir files."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Resolve all selected PDFs and exit without RAG, MAS, or AXIS.",
    )
    return parser


def _resolve_eval_ids(
    args: Any,
    *,
    extraction_jsonl: Optional[Path],
    jsonl_authoritative: bool,
    first_jsonl: Any,
) -> List[str]:
    id_values = list(getattr(args, "id_values", None) or [])
    ids_csv = str(getattr(args, "ids", "") or "")
    ids_file = Path(args.ids_file) if getattr(args, "ids_file", None) else None
    allow_any_count = bool(getattr(args, "allow_any_count", False))
    expected_count = getattr(args, "expected_count", None)
    if first_jsonl:
        if not jsonl_authoritative:
            raise SelectionError("--first-jsonl requires --jsonl-authoritative.")
        if id_values or ids_csv.strip() or ids_file is not None:
            raise SelectionError(
                "--first-jsonl cannot be combined with --id, --ids, or --ids-file."
            )
        from rag_pipeline import select_first_usable_jsonl_ids

        want = int(first_jsonl)
        selected = select_first_usable_jsonl_ids(extraction_jsonl, want)
        if not selected:
            raise SelectionError("No usable JSONL records found for --first-jsonl.")
        if expected_count is not None and len(selected) != int(expected_count):
            raise SelectionError(
                f"--first-jsonl selected {len(selected)} IDs but --expected-count "
                f"is {int(expected_count)}."
            )
        if not allow_any_count and expected_count is None and len(selected) != want:
            raise SelectionError(
                f"--first-jsonl {want} found only {len(selected)} usable unique JSONL records. "
                "Use --allow-any-count to accept fewer."
            )
        return selected
    return resolve_selected_ids(
        id_values=id_values,
        ids_csv=ids_csv,
        ids_file=ids_file,
        allow_any_count=allow_any_count,
        expected_count=expected_count,
    )


def run_eval(
    args: argparse.Namespace,
    *,
    embeddings: Optional[object] = None,
    pipeline: Any = None,
    backbone_factory: Optional[Callable[..., Any]] = None,
    stdout: Any = None,
) -> int:
    log = stdout or sys.stdout
    jsonl_authoritative = bool(getattr(args, "jsonl_authoritative", False))
    first_jsonl = getattr(args, "first_jsonl", None)
    output_dir = Path(args.output_dir).resolve()
    pdf_dir = Path(args.pdf_dir).resolve() if getattr(args, "pdf_dir", None) else None
    extraction_jsonl = (
        Path(args.extraction_jsonl).resolve()
        if getattr(args, "extraction_jsonl", None)
        else None
    )
    cache_dir = output_dir / "pdf_cache"
    metrics_dir = output_dir / "metrics"

    if jsonl_authoritative and extraction_jsonl is None:
        print("--jsonl-authoritative requires --extraction-jsonl", file=log)
        return 1
    if extraction_jsonl is not None and not extraction_jsonl.is_file():
        print(f"Extraction JSONL not found: {extraction_jsonl}", file=log)
        return 1
    if extraction_jsonl is not None:
        args.extraction_jsonl = extraction_jsonl

    lecturebank_dir: Optional[Path] = None
    catalog = None
    if getattr(args, "lecturebank_dir", None):
        lecturebank_dir = Path(args.lecturebank_dir).resolve()
        if not jsonl_authoritative or bool(getattr(args, "list_ids", False)):
            try:
                catalog = load_lecturebank_catalog(lecturebank_dir)
            except (FileNotFoundError, ValueError) as exc:
                print(str(exc), file=log)
                return 1
    elif not jsonl_authoritative:
        print("--lecturebank-dir is required unless --jsonl-authoritative", file=log)
        return 1
    if jsonl_authoritative and lecturebank_dir is None:
        lecturebank_dir = extraction_jsonl.parent if extraction_jsonl is not None else output_dir

    if getattr(args, "list_ids", False):
        if catalog is None:
            print("--list-ids requires --lecturebank-dir with LectureBank metadata.", file=log)
            return 1
        records = (
            catalog.search(args.search, limit=args.limit)
            if str(getattr(args, "search", "") or "").strip()
            else catalog.all_records(limit=args.limit)
        )
        avail = {
            rec.lecturebank_id: local_pdf_available(
                rec.lecturebank_id,
                cache_dir=cache_dir if cache_dir.exists() else None,
                pdf_dir=pdf_dir,
                search_roots=(lecturebank_dir,),
            )
            for rec in records
        }
        print(format_id_table(records, pdf_available=avail), file=log, end="")
        return 0 if records else 1

    try:
        selected = _resolve_eval_ids(
            args,
            extraction_jsonl=extraction_jsonl,
            jsonl_authoritative=jsonl_authoritative,
            first_jsonl=first_jsonl,
        )
    except SelectionError as exc:
        print(str(exc), file=log)
        return 2

    if jsonl_authoritative:
        from rag_pipeline import load_selected_extractions

        lookups = load_selected_extractions(extraction_jsonl, selected)
        for ident, item in lookups.items():
            if item.status == "usable":
                item.match_strategy = "jsonl_id"
                item.match_status = "unique_match"
                item.extraction_id = item.extraction_id or str(ident)
        catalog = catalog_from_extraction_lookups(extraction_jsonl, lookups)
        args._jsonl_lookup = lookups

    expected = (
        "any"
        if bool(getattr(args, "allow_any_count", False))
        else int(
            getattr(args, "expected_count", None)
            or (first_jsonl if first_jsonl else REQUIRED_EVAL_ID_COUNT)
        )
    )
    print(
        f"Selected unique {'JSONL' if jsonl_authoritative else 'LectureBank'} IDs: {len(selected)}",
        file=log,
    )
    print(f"Expected: {expected}", file=log)

    rag_cfg = {
        "embedding_model": args.embedding_model,
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "chunk_by_page": not bool(getattr(args, "no_page_chunking", False)),
    }
    max_questions = int(args.max_questions)
    max_explain = getattr(args, "max_explain_questions", None)
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
        "axis_interrogation_mode": getattr(
            args, "axis_interrogation_mode", DEFAULTS["axis_interrogation_mode"]
        ),
        "axis_interrogation_policy": "counterfactual_only",
        "diagnose_hybrid_add": bool(getattr(args, "diagnose_hybrid_add", False)),
        "hybrid_add_from_source": bool(getattr(args, "hybrid_add_from_source", False)),
        "hybrid_add_source_limit": getattr(args, "hybrid_add_source_limit", None),
        "hybrid_add_source_k_per_query": getattr(
            args, "hybrid_add_source_k_per_query", None
        ),
        "hybrid_add_source_max_search_hits": getattr(
            args, "hybrid_add_source_max_search_hits", None
        ),
        "hybrid_add_source_mmr_lambda": getattr(
            args, "hybrid_add_source_mmr_lambda", None
        ),
    }
    rag_fp = rag_config_fingerprint(rag_cfg)
    gen_fp = generation_config_fingerprint(gen_cfg)

    if bool(getattr(args, "build_effect_reference_set", False)):
        from mas_langgraph.evaluation.effect_reference import require_hybrid_reference_mode

        try:
            require_hybrid_reference_mode(gen_cfg.get("axis_interrogation_mode"))
        except Exception as exc:
            print(str(exc), file=log)
            return 2

    if getattr(args, "preflight", False):
        return run_preflight(
            selected,
            catalog=catalog,
            cache_dir=cache_dir,
            pdf_dir=pdf_dir,
            lecturebank_dir=lecturebank_dir,
            no_download=bool(getattr(args, "no_download", False)),
            log=log,
            jsonl_path=extraction_jsonl,
            jsonl_authoritative=jsonl_authoritative,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {
            "manifest_version": 1,
            "lecturebank_dir": str(lecturebank_dir),
            "output_dir": str(output_dir),
            "selected_ids": selected,
            "config": {**rag_cfg, **gen_cfg},
            "records": [],
        }
    manifest["selected_ids"] = selected
    manifest["config"] = {**rag_cfg, **gen_cfg}
    manifest["lecturebank_dir"] = str(lecturebank_dir)
    manifest["output_dir"] = str(output_dir)

    collected: List[Dict[str, Any]] = []
    if (
        bool(args.resume)
        or bool(getattr(args, "concept_change_from_saved", False))
        or bool(getattr(args, "outcome_criteria_from_saved", False))
        or bool(getattr(args, "saquet_from_saved", False))
    ) and (metrics_dir / "document_metrics.csv").is_file():
        collected = load_collected_from_metrics(metrics_dir)

    export_quiz = bool(getattr(args, "export_quiz", False))
    export_streamlit_json = bool(getattr(args, "export_streamlit_json", False))

    def persist(
        output_dir: Path,
        metrics_dir: Path,
        manifest_path: Path,
        manifest: Dict[str, Any],
        collected: Sequence[Mapping[str, Any]],
    ) -> None:
        _persist(
            output_dir,
            metrics_dir,
            manifest_path,
            manifest,
            collected,
            export_quiz=export_quiz,
            export_streamlit_json=export_streamlit_json,
        )

    ran_offline = False
    stats: Dict[str, int] = {
        "failures": 0,
        "skipped_resume": 0,
        "stores_cleaned": 0,
        "stores_retained": 0,
        "total": len(selected),
    }
    if bool(getattr(args, "concept_change_from_saved", False)):
        ran_offline = True
        args.evaluate_concept_change = True
        print(
            "Concept-change-from-saved: evaluating saved experiment JSON only "
            "(no RAG, MAS, AXIS interrogation, or replay).",
            file=log,
        )
        stats = run_concept_change_from_saved(
            selected=selected,
            args=args,
            output_dir=output_dir,
            metrics_dir=metrics_dir,
            gen_cfg=gen_cfg,
            collected=collected,
            persist=persist,
            manifest_path=manifest_path,
            manifest=manifest,
            log=log,
        )
    if bool(getattr(args, "outcome_criteria_from_saved", False)):
        ran_offline = True
        print(
            "Outcome-criteria-from-saved: exporting stored criterion deltas only "
            "(no RAG, MAS, AXIS interrogation, replay, or LLM evaluator).",
            file=log,
        )
        oc_stats = run_outcome_criteria_from_saved(
            selected=selected,
            output_dir=output_dir,
            metrics_dir=metrics_dir,
            collected=collected,
            persist=persist,
            manifest_path=manifest_path,
            manifest=manifest,
            log=log,
        )
        stats = {
            "failures": max(int(stats.get("failures") or 0), int(oc_stats.get("failures") or 0)),
            "skipped_resume": int(stats.get("skipped_resume") or 0),
            "stores_cleaned": int(stats.get("stores_cleaned") or 0),
            "stores_retained": int(stats.get("stores_retained") or 0),
            "total": len(selected),
        }
    if bool(getattr(args, "saquet_from_saved", False)):
        ran_offline = True
        args.evaluate_saquet = True
        print(
            "SAQUET-from-saved: evaluating saved factual MCQs only "
            "(no RAG, MAS, AXIS interrogation, or replay). "
            "Reuses metrics/saquet_factual_quiz.json and valid cache flags.",
            file=log,
        )
        sq_stats = run_saquet_from_saved(
            selected=selected,
            args=args,
            output_dir=output_dir,
            metrics_dir=metrics_dir,
            collected=collected,
            persist=persist,
            manifest_path=manifest_path,
            manifest=manifest,
            log=log,
        )
        stats = {
            "failures": max(int(stats.get("failures") or 0), int(sq_stats.get("failures") or 0)),
            "skipped_resume": int(stats.get("skipped_resume") or 0),
            "stores_cleaned": int(stats.get("stores_cleaned") or 0),
            "stores_retained": int(stats.get("stores_retained") or 0),
            "total": len(selected),
        }
    if not ran_offline:
        stats = process_documents(
        selected=selected,
        catalog=catalog,
        args=args,
        lecturebank_dir=lecturebank_dir,
        output_dir=output_dir,
        pdf_dir=pdf_dir,
        cache_dir=cache_dir,
        metrics_dir=metrics_dir,
        manifest=manifest,
        manifest_path=manifest_path,
        rag_cfg=rag_cfg,
        gen_cfg=gen_cfg,
        rag_fp=rag_fp,
        gen_fp=gen_fp,
        collected=collected,
        persist=persist,
        failed_document=_failed_document,
        find_record=_find_manifest_record,
        upsert=_upsert_manifest,
        embeddings=embeddings,
        pipeline=pipeline,
        backbone_factory=backbone_factory,
        log=log,
    )
    completed_n = sum(1 for r in collected if eval_completed(r.get("eval_status")))
    failed_n = sum(1 for r in collected if str(r.get("eval_status") or "") == "failed")
    print("", file=log)
    print(f"Documents selected: {stats['total']}", file=log)
    print(f"Completed: {completed_n}", file=log)
    print(f"Failed: {failed_n}", file=log)
    print(f"Skipped by resume: {stats['skipped_resume']}", file=log)
    print(f"Vector stores retained: {stats['stores_retained']}", file=log)
    print(f"Vector stores cleaned: {stats['stores_cleaned']}", file=log)
    print(f"Metrics directory: {metrics_dir}", file=log)
    if stats["failures"] and not bool(getattr(args, "allow_partial", False)):
        print(f"{stats['failures']} selected document(s) failed.", file=log)
        return 1
    return 0


def _eval_result_path(output_dir: Path, document_id: str) -> Path:
    return Path(output_dir) / f"presentation_{document_id}" / EVAL_RESULT_NAME


def _find_manifest_record(manifest: Mapping[str, Any], lecturebank_id: str) -> Optional[Dict[str, Any]]:
    want = document_id_for(lecturebank_id)
    for rec in manifest.get("records") or []:
        if str(rec.get("lecturebank_id")) == str(lecturebank_id) or str(rec.get("document_id")) == want:
            return dict(rec)
    return None


def _upsert_manifest(manifest: Dict[str, Any], record: Mapping[str, Any]) -> None:
    recs = list(manifest.get("records") or [])
    ident = str(record.get("lecturebank_id") or record.get("document_id") or "")
    out = []
    replaced = False
    for rec in recs:
        key = str(rec.get("lecturebank_id") or rec.get("document_id") or "")
        if ident and key == ident:
            out.append({**rec, **record})
            replaced = True
        elif record.get("document_id") and rec.get("document_id") == record.get("document_id"):
            out.append({**rec, **record})
            replaced = True
        else:
            out.append(rec)
    if not replaced:
        out.append(dict(record))
    manifest["records"] = out


def _eval_resume_ok(
    *,
    output_dir: Path,
    lecturebank_id: str,
    document_id: str,
    source_hash: str,
    rag_fp: Mapping[str, Any],
    gen_fp: Mapping[str, Any],
    existing: Optional[Mapping[str, Any]],
) -> bool:
    path = _eval_result_path(output_dir, document_id)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        stored = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if str(stored.get("lecturebank_id")) != str(lecturebank_id):
        return False
    if str(stored.get("source_hash") or stored.get("pdf_sha256") or "") != str(source_hash):
        return False
    if dict(stored.get("rag_config") or {}) != dict(rag_fp):
        return False
    if dict(stored.get("generation_config") or {}) != dict(gen_fp):
        return False
    if stored.get("eval_status") not in {"success", "axis_no_valid_replay", "rag_only"}:
        return False
    index_dir = Path(output_dir) / f"presentation_{document_id}" / "vector_store"
    if not index_dir.exists() or not any(index_dir.iterdir()):
        return False
    if not stored.get("document_metrics") and stored.get("eval_status") == "success":
        return False
    _ = existing
    return True


def _resume_payload(
    result: Mapping[str, Any],
    rag_fp: Mapping[str, Any],
    gen_fp: Mapping[str, Any],
    pdf_info: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(result),
        "source_hash": pdf_info.get("sha256"),
        "pdf_sha256": pdf_info.get("sha256"),
        "rag_config": dict(rag_fp),
        "generation_config": dict(gen_fp),
    }


def _failed_document(
    lecturebank_id: str,
    meta: Mapping[str, Any],
    *,
    stage: str,
    error: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        **dict(meta),
        "lecturebank_id": lecturebank_id,
        "document_id": document_id_for(lecturebank_id),
        "failure_stage": stage,
        "error_message": error,
        "rag_status": (
            "failed"
            if stage in {"rag", "pdf", "selection"}
            else ((extra or {}).get("rag_status") or "")
        ),
        "mas_status": (
            "failed" if stage == "mas" else ((extra or {}).get("mas_status") or "")
        ),
        "axis_status": "failed" if stage == "axis" else "",
        "eval_status": "failed",
        "accepted_question_count": (extra or {}).get("accepted_question_count") or 0,
        "explained_question_count": (extra or {}).get("explained_question_count") or 0,
        "valid_replay_comparisons": 0,
        "interrogation_attempts": [],
        "replay_comparisons": [],
        "provenance_edges": [],
        "distributions": [],
        "round_metrics": [],
        "document_metrics": {
            **dict(meta),
            "lecturebank_id": lecturebank_id,
            "document_id": document_id_for(lecturebank_id),
            "failure_stage": stage,
            "error_message": error,
            "eval_status": "failed",
            "rag_status": "failed" if stage in {"rag", "pdf", "selection"} else (extra or {}).get("rag_status") or "",
            "mas_status": "failed" if stage == "mas" else (extra or {}).get("mas_status") or "",
            "axis_status": "failed" if stage in {"axis", "mcq"} else "",
            "accepted_question_count": (extra or {}).get("accepted_question_count") or 0,
            "explained_question_count": 0,
            "valid_replay_comparisons": 0,
        },
    }
    if extra:
        for key in ("pdf_source", "pdf_sha256", "rag_status", "mas_status"):
            if key in extra:
                row[key] = extra[key]
                row["document_metrics"][key] = extra[key]
    return row


def _persist(
    output_dir: Path,
    metrics_dir: Path,
    manifest_path: Path,
    manifest: Dict[str, Any],
    collected: Sequence[Mapping[str, Any]],
    export_quiz: bool = False,
    export_streamlit_json: bool = False,
) -> None:
    document_rows = []
    attempts = []
    comparisons = []
    provenance = []
    distributions = []
    rounds = []
    reference_interventions = []
    effect_discovery_questions = []
    concept_change_assessments = []
    concept_change_questions = []
    outcome_criteria_rows = []
    outcome_criteria_diagnostics = []
    saquet_mas_rows = []
    hybrid_add_candidates = []
    hybrid_add_boundaries = []
    hybrid_add_source_events = []
    hybrid_add_source_summaries = []
    for item in collected:
        doc = dict(item.get("document_metrics") or {})
        for key in (
            "lecturebank_id",
            "title",
            "topic_id",
            "topic_name",
            "instructor",
            "venue",
            "year",
            "url",
            "document_id",
            "pdf_source",
            "pdf_sha256",
            "failure_stage",
            "error_message",
            "rag_status",
            "mas_status",
            "axis_status",
            "eval_status",
            "accepted_question_count",
            "explained_question_count",
        ):
            if not doc.get(key) and item.get(key) not in (None, ""):
                doc[key] = item.get(key)
        document_rows.append(doc)
        attempts.extend(list(item.get("interrogation_attempts") or []))
        comparisons.extend(list(item.get("replay_comparisons") or []))
        provenance.extend(list(item.get("provenance_edges") or []))
        distributions.extend(list(item.get("distributions") or []))
        reference_interventions.extend(list(item.get("reference_interventions") or []))
        effect_discovery_questions.extend(list(item.get("effect_discovery_by_question") or []))
        concept_change_assessments.extend(list(item.get("concept_change_assessments") or []))
        concept_change_questions.extend(list(item.get("concept_change_by_question") or []))
        outcome_criteria_rows.extend(list(item.get("outcome_criteria_changes") or []))
        outcome_criteria_diagnostics.extend(list(item.get("outcome_criteria_diagnostics") or []))
        saquet_mas_rows.extend(list(item.get("saquet_by_mas_rows") or []))
        hybrid_add_candidates.extend(list(item.get("hybrid_add_candidates") or []))
        hybrid_add_boundaries.extend(list(item.get("hybrid_add_boundaries") or []))
        hybrid_add_source_events.extend(list(item.get("hybrid_add_source_events") or []))
        src_sum = item.get("hybrid_add_source_summary") or {}
        if src_sum:
            hybrid_add_source_summaries.append(dict(src_sum))
        for rnd in item.get("round_metrics") or []:
            row = dict(rnd)
            row.setdefault("lecturebank_id", item.get("lecturebank_id"))
            rounds.append(row)
    overall_dists = _overall_distributions(distributions)
    distributions = list(distributions) + overall_dists
    pooled = pool_collected(collected, document_rows)
    from mas_langgraph.evaluation.effect_reference import aggregate_effect_discovery

    if effect_discovery_questions:
        pooled.update(aggregate_effect_discovery(effect_discovery_questions))
        pooled["reference_sweep_ran"] = True
    else:
        pooled["reference_sweep_ran"] = False
    if concept_change_questions or concept_change_assessments:
        from mas_langgraph.evaluation.concept_change import aggregate_concept_change

        pooled.update(aggregate_concept_change(concept_change_questions))
        pooled["concept_change_ran"] = True
    else:
        pooled["concept_change_ran"] = False
    overall = overall_metric_rows(document_rows, attempts, comparisons, provenance, pooled)
    quiz_rows: List[Dict[str, Any]] = []
    quiz_unavailable: List[str] = []
    if export_quiz:
        quiz_rows = collect_quiz_rows(collected)
        seen_unavail = set()
        for item in collected:
            if not item.get("quiz_export_unavailable"):
                continue
            lid = str(item.get("lecturebank_id") or "")
            if lid and lid not in seen_unavail:
                seen_unavail.add(lid)
                quiz_unavailable.append(lid)
    report = build_report(
        catalog_rows=document_rows,
        document_rows=document_rows,
        attempt_rows=attempts,
        comparison_rows=comparisons,
        provenance_rows=provenance,
        distribution_rows=distributions,
        round_rows=rounds,
        overall_rows=overall,
        pooled=pooled,
        quiz_export_enabled=bool(export_quiz),
        quiz_row_count=len(quiz_rows),
        quiz_unavailable_ids=quiz_unavailable,
        streamlit_export_enabled=bool(export_streamlit_json),
        streamlit_json_count=count_streamlit_json_files(output_dir) if export_streamlit_json else 0,
        streamlit_unavailable_ids=collect_streamlit_unavailable_ids(collected)
        if export_streamlit_json
        else (),
        reference_question_rows=effect_discovery_questions,
        reference_intervention_rows=reference_interventions,
        reference_aggregate=pooled if pooled.get("reference_sweep_ran") else None,
        concept_change_question_rows=concept_change_questions,
        outcome_criteria_rows=outcome_criteria_rows,
        outcome_criteria_diagnostics=outcome_criteria_diagnostics,
        saquet_mas_rows=saquet_mas_rows,
        hybrid_add_candidate_rows=hybrid_add_candidates,
        hybrid_add_boundary_rows=hybrid_add_boundaries,
        hybrid_add_source_event_rows=hybrid_add_source_events,
        hybrid_add_source_summaries=hybrid_add_source_summaries,
    )
    write_metrics_outputs(
        metrics_dir,
        document_rows=document_rows,
        attempt_rows=attempts,
        comparison_rows=comparisons,
        provenance_rows=provenance,
        distribution_rows=distributions,
        round_rows=rounds,
        overall_rows=overall,
        report_markdown=report,
        reference_intervention_rows=reference_interventions,
        effect_discovery_question_rows=effect_discovery_questions,
        concept_change_assessment_rows=concept_change_assessments,
        concept_change_question_rows=concept_change_questions,
        outcome_criteria_rows=outcome_criteria_rows,
        saquet_mas_rows=saquet_mas_rows,
        hybrid_add_candidate_rows=hybrid_add_candidates,
        hybrid_add_boundary_rows=hybrid_add_boundaries,
        hybrid_add_source_event_rows=hybrid_add_source_events,
        hybrid_add_source_summaries=hybrid_add_source_summaries,
    )
    if export_quiz:
        write_quiz_outputs(
            output_dir,
            metrics_dir,
            quiz_rows,
            unavailable_ids=quiz_unavailable,
        )
    atomic_write_json(manifest_path, manifest)


def _overall_distributions(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict, Counter

    grouped: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if str(row.get("scope")) != "document":
            continue
        grouped[str(row.get("dimension"))].update({str(row.get("value")): int(row.get("count") or 0)})
    out = []
    for dim, counter in grouped.items():
        if dim in {"proposed_query_operation", "executed_query_operation"}:
            denom = sum(int(counter.get(op) or 0) for op in RECOGNIZED_QUERY_OPS)
        else:
            denom = sum(counter.values())
        for value, count in counter.items():
            use_den = denom
            if dim in {"proposed_query_operation", "executed_query_operation"} and value == "DONE":
                use_den = NA
            out.append(
                {
                    "scope": "overall",
                    "lecturebank_id": "ALL",
                    "dimension": dim,
                    "value": value,
                    "count": count,
                    "denominator": use_den,
                    "percentage": pct(count, use_den) if use_den not in (None, 0) else NA,
                }
            )
    return out


def pool_collected(
    collected: Sequence[Mapping[str, Any]],
    document_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    n = len(document_rows)
    pdf_ok = sum(
        1
        for r in document_rows
        if r.get("failure_stage") not in {"pdf", "selection"}
    )
    rag_ok = sum(1 for r in document_rows if r.get("rag_status") == "success")
    mas_ok = sum(1 for r in document_rows if r.get("mas_status") == "success")
    mcq_ok = sum(1 for r in document_rows if int(r.get("accepted_question_count") or 0) >= 1)
    axis_ok = sum(1 for r in document_rows if int(r.get("valid_replay_comparisons") or 0) >= 1)

    pooled: Dict[str, Any] = {
        "pdf_acquisition_success_rate_num": pdf_ok,
        "pdf_acquisition_success_rate_den": n,
        "rag_success_rate_num": rag_ok,
        "rag_success_rate_den": n,
        "factual_mas_success_rate_num": mas_ok,
        "factual_mas_success_rate_den": n,
        "documents_with_accepted_mcq_rate_num": mcq_ok,
        "documents_with_accepted_mcq_rate_den": n,
        "documents_with_valid_axis_replay_rate_num": axis_ok,
        "documents_with_valid_axis_replay_rate_den": n,
        "total_accepted_mcqs": sum(int(r.get("accepted_question_count") or 0) for r in document_rows),
        "total_explained_mcqs": sum(int(r.get("explained_question_count") or 0) for r in document_rows),
        "total_valid_replay_comparisons": sum(int(r.get("valid_replay_comparisons") or 0) for r in document_rows),
    }
    sum_keys = (
        "proposed_remove_count",
        "proposed_add_count",
        "proposed_replace_count",
        "proposed_what_count",
        "proposed_done_count",
        "proposed_unparsed_count",
        "proposed_recognized_non_done_count",
        "executed_remove_count",
        "executed_add_count",
        "executed_replace_count",
        "executed_what_count",
        "executed_done_count",
        "executed_unparsed_count",
        "executed_recognized_non_done_count",
        "counterfactual_intervention_proposals",
        "executed_counterfactual_interventions",
        "validation_parser_error_count",
        "validation_resolution_error_count",
        "validation_duplicate_error_count",
        "validation_contract_error_count",
        "validation_execution_error_count",
        "validation_not_currently_executable_count",
        "validation_other_status_count",
        "counterfactual_executions",
        "successful_replay_count",
        "valid_replay_comparisons",
        "noop_decision_stability_true",
        "noop_decision_stability_false",
        "noop_decision_stability_unavailable",
        "noop_criteria_stability_true",
        "noop_criteria_stability_false",
        "noop_criteria_stability_unavailable",
        "noop_content_stability_true",
        "noop_content_stability_false",
        "noop_content_stability_unavailable",
        "full_noop_stability_true",
        "full_noop_stability_false",
        "full_noop_stability_unavailable",
        "replay_completion_mode_target_observable",
        "replay_completion_mode_graph_end",
        "replay_completion_mode_fallback_full",
        "replay_completion_mode_error_unknown",
        "decision_effect_count",
        "decision_effect_count_raw",
        "decision_effect_denominator",
        "decision_effect_raw_denominator",
        "criterion_effect_denominator",
        "criterion_effect_raw_denominator",
        "decision_relevant_criterion_effect_denominator",
        "content_effect_denominator",
        "content_effect_raw_denominator",
        "stem_change_denominator",
        "correct_answer_change_denominator",
        "distractor_change_denominator",
        "explanation_change_denominator",
        "criterion_effect_count",
        "criterion_effect_count_raw",
        "decision_relevant_criterion_effect_count",
        "content_effect_count",
        "content_effect_count_raw",
        "stem_change_count",
        "correct_answer_change_count",
        "distractor_change_count",
        "explanation_change_count",
        "question_not_reproduced_count",
        "effect_yield_any_count",
        "effect_yield_any_denominator",
        "valid_control_stable_comparisons",
        "factual_llm_request_count",
        "factual_llm_unique_request_count",
        "factual_llm_memo_hits",
        "noop_llm_request_count",
        "noop_factual_response_hits",
        "noop_llm_request_misses",
        "noop_live_provider_calls",
        "counterfactual_llm_request_count",
        "counterfactual_factual_response_hits",
        "counterfactual_novel_request_count",
        "counterfactual_memo_hits",
        "counterfactual_live_provider_calls",
        "provenance_relation_count",
        "provenance_complete_relation_count",
        "unique_relevant_artifact_count",
        "intervention_target_in_slice_count",
        "intervention_target_in_slice_denominator",
        "intervention_target_has_recorded_use_count",
        "intervention_target_has_recorded_use_denominator",
        "provenance_slicing_error_count",
        "target_EvidenceChunk_count",
        "target_KnowledgeAtom_count",
        "target_MCQBlueprint_count",
        "target_CandidateMCQ_count",
        "target_other_count",
        "synthesis_case_count",
        "initial_synthesis_valid_count",
        "repair_attempt_count",
        "repair_success_count",
        "fallback_count",
        "presentation_valid_count",
        "cases_with_faithfulness_warning_count",
        "checkpoint_position_n",
        "checkpoint_progress_n",
    )
    for key in sum_keys:
        pooled[key] = _sum_num(document_rows, key)
    pooled["noop_decision_stability_known"] = (pooled["noop_decision_stability_true"] or 0) + (
        pooled["noop_decision_stability_false"] or 0
    )
    pooled["noop_criteria_stability_known"] = (pooled["noop_criteria_stability_true"] or 0) + (
        pooled["noop_criteria_stability_false"] or 0
    )
    pooled["noop_content_stability_known"] = (pooled["noop_content_stability_true"] or 0) + (
        pooled["noop_content_stability_false"] or 0
    )
    pooled["full_noop_stability_known"] = (pooled["full_noop_stability_true"] or 0) + (
        pooled["full_noop_stability_false"] or 0
    )
    pooled["decision_effect_known"] = pooled.get("decision_effect_denominator") or 0
    pooled["decision_effect_raw_known"] = pooled.get("decision_effect_raw_denominator") or 0
    pooled["criterion_effect_known"] = pooled.get("criterion_effect_denominator") or 0
    pooled["criterion_effect_raw_known"] = pooled.get("criterion_effect_raw_denominator") or 0
    pooled["decision_relevant_criterion_effect_known"] = pooled.get(
        "decision_relevant_criterion_effect_denominator"
    ) or 0
    pooled["content_effect_known"] = pooled.get("content_effect_denominator") or 0
    pooled["content_effect_raw_known"] = pooled.get("content_effect_raw_denominator") or 0
    pooled["stem_change_known"] = pooled.get("stem_change_denominator") or 0
    pooled["correct_answer_change_known"] = pooled.get("correct_answer_change_denominator") or 0
    pooled["distractor_change_known"] = pooled.get("distractor_change_denominator") or 0
    pooled["explanation_change_known"] = pooled.get("explanation_change_denominator") or 0
    pooled["question_not_reproduced_den"] = _sum_num(document_rows, "valid_replay_comparisons")
    pos_vals = [r.get("checkpoint_position_pct_mean") for r in document_rows]
    prog_vals = [r.get("checkpoint_progress_pct_mean") for r in document_rows]
    pos_stats = describe(pos_vals)
    prog_stats = describe(prog_vals)
    pooled["checkpoint_position_pct_mean"] = pos_stats["mean"]
    pooled["checkpoint_position_pct_median"] = _median([r.get("checkpoint_position_pct_median") for r in document_rows])
    pooled["checkpoint_position_pct_min"] = pos_stats["min"]
    pooled["checkpoint_position_pct_max"] = pos_stats["max"]
    pooled["checkpoint_progress_pct_mean"] = prog_stats["mean"]
    pooled["checkpoint_progress_pct_median"] = _median([r.get("checkpoint_progress_pct_median") for r in document_rows])
    pooled["checkpoint_progress_pct_min"] = prog_stats["min"]
    pooled["checkpoint_progress_pct_max"] = prog_stats["max"]
    for tkey in (
        "factual_workflow_s",
        "experiment_total_s",
        "axis_total_s",
        "interrogation_total_s",
        "noop_total_s",
        "counterfactual_replay_total_s",
        "comparison_total_s",
    ):
        vals = [r.get(tkey) for r in document_rows if r.get(tkey) not in (None, "", "NA")]
        nums = []
        for v in vals:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                pass
        pooled[f"{tkey}_sum"] = sum(nums) if nums else NA
        pooled[f"{tkey}_mean"] = (sum(nums) / len(nums)) if nums else NA
    word_means = [r.get("participant_word_count_mean") for r in document_rows]
    sent_means = [r.get("participant_sentence_count_mean") for r in document_rows]
    pooled["participant_word_count_mean"] = describe(word_means)["mean"]
    pooled["participant_word_count_median"] = _median([r.get("participant_word_count_median") for r in document_rows])
    pooled["participant_sentence_count_mean"] = describe(sent_means)["mean"]
    pooled["participant_sentence_count_median"] = _median(
        [r.get("participant_sentence_count_median") for r in document_rows]
    )
    pooled["executed_node_count_mean"] = describe(
        [r.get("executed_node_count_mean") for r in document_rows]
    )["mean"]
    pooled["executed_node_count_median"] = _median(
        [r.get("executed_node_count_median") for r in document_rows]
    )
    _ = collected
    return pooled


def _sum_num(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        val = row.get(key)
        if val in (None, "", "NA"):
            continue
        try:
            total += int(val)
        except (TypeError, ValueError):
            try:
                total += int(float(val))
            except (TypeError, ValueError):
                continue
    return total


def _sum_optional_pair(rows: Sequence[Mapping[str, Any]], num_key: str, rate_key: str) -> Optional[int]:
    known = 0
    saw = False
    for row in rows:
        n = row.get(num_key)
        r = row.get(rate_key)
        if n in (None, "", "NA"):
            continue
        try:
            n_i = int(n)
        except (TypeError, ValueError):
            continue
        if r in (None, "", "NA"):
            known += n_i
            saw = True
            continue
        try:
            rf = float(r)
            if rf == 0:
                # cannot recover denom from 0 rate without explicit denom
                den = row.get(num_key.replace("_count", "_denominator")) or row.get(
                    rate_key.replace("_rate", "_denominator")
                )
                if den not in (None, "", "NA"):
                    known += int(den)
                    saw = True
                else:
                    known += n_i
                    saw = True
            else:
                known += int(round(n_i / rf)) if rf else n_i
                saw = True
        except (TypeError, ValueError, ZeroDivisionError):
            known += n_i
            saw = True
    return known if saw else 0


def _median(values: Sequence[Any]) -> Any:
    nums = []
    for v in values:
        if v in (None, "", "NA"):
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return NA
    nums.sort()
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_eval(args)
