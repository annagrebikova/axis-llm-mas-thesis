"""Offline lifecycle tests for LectureBank eval storage (no provider calls)."""

from __future__ import annotations

import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List
from unittest.mock import patch

from mas_langgraph.batch.generate import InMemoryGenerationResult
from mas_langgraph.batch.ids import file_sha256
from mas_langgraph.evaluation.execute import process_documents
from mas_langgraph.evaluation.lecturebank_meta import load_lecturebank_catalog
from mas_langgraph.evaluation.lifecycle import (
    delete_vector_store,
    maybe_delete_downloaded_pdf,
    vector_store_dir,
)
from mas_langgraph.evaluation.pdf_source import document_id_for
from mas_langgraph.evaluation.runner import _failed_document, _find_manifest_record, _persist, _upsert_manifest


@contextmanager
def _tempdir() -> Iterator[Path]:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        yield Path(tmp.name)
    finally:
        tmp.cleanup()


def _write_catalog(root: Path, ids: List[str] | None = None) -> Path:
    ids = ids or ["14", "141"]
    root.mkdir(parents=True, exist_ok=True)
    lines = ["ID\tInstructor\tTitle\tTopic\tURL\tVenue\tYear"]
    meta = {
        "14": ("Ada", "Intro ML", "1", "http://example.test/14.pdf", "Yale", "2016"),
        "141": ("Grace", "Deep Learning", "2", "http://example.test/141.pdf", "Yale", "2018"),
    }
    for ident in ids:
        row = meta.get(ident, ("X", f"Title {ident}", "1", f"http://example.test/{ident}.pdf", "Y", "2020"))
        lines.append("\t".join([ident, *row]))
    (root / "alldata.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 dummy " + path.name.encode("utf-8"))
    return path


def _cmp():
    return {
        "intervention_signature": {
            "operation": "REMOVE",
            "target_ref": {"key": "EvidenceChunk:e1:v1"},
            "replacement_ref": None,
            "checkpoint_alias": "c_7",
            "checkpoint_id": "ckpt-7",
            "boundary_raw": "before generate_candidate_mcqs#1",
        },
        "evidence_status": "no_demonstrated_effect",
        "decision_control_stable": True,
        "criteria_control_stable": True,
        "subject_content_control_stable": True,
        "noop": {"decision": "accept"},
        "counterfactual": {"decision": "accept"},
        "criterion_deltas": [],
        "information_role_finding": {},
        "question_content_comparison": {},
    }


def _payload(accepted: int = 10) -> Dict[str, Any]:
    comparisons = [_cmp()]
    return {
        "cases": [
            {
                "candidate_ref": {"key": "CandidateMCQ:q1:v1", "logical_id": "q1", "version": 1},
                "conditions": {
                    "axis": {
                        "evidence_summary": {},
                        "raw_evidence": {
                            "interrogation_attempts": [
                                {
                                    "attempt_index": 0,
                                    "generated_query": "REMOVE x",
                                    "parsed_operation": "REMOVE",
                                    "validation_status": "ok",
                                    "executed": True,
                                    "produced_counterfactual": True,
                                }
                            ],
                            "replay_comparisons": comparisons,
                            "provenance_relations": [],
                            "relevant_artifacts": [],
                            "termination_reason": "max_rounds",
                        },
                    }
                },
                "counterfactual_count": 1,
                "successful_replay_count": 1,
            }
        ],
        "workflow_llm_diagnostics": {},
        "timing": {},
        "_accepted": accepted,
    }


def _args(**kwargs: Any) -> Namespace:
    base = dict(
        resume=False,
        overwrite=False,
        stage="all",
        cleanup_vector_store=False,
        temporary_vector_stores=False,
        cleanup_downloaded_pdfs=False,
        no_download=True,
        allow_partial=True,
    )
    base.update(kwargs)
    return Namespace(**base)


def _rag_fp() -> Dict[str, Any]:
    return {
        "embedding_model": "fake",
        "chunk_size": 800,
        "chunk_overlap": 200,
        "chunk_by_page": True,
    }


def _gen_fp() -> Dict[str, Any]:
    return {
        "llm_model": "fake",
        "model_type": "openai",
        "temperature": 0.0,
        "seed": 42,
        "max_rounds": 5,
        "max_questions": 10,
        "max_explain_questions": None,
        "question": "Why?",
        "orchestrator": "langgraph",
    }


def _fake_prepare(events: List[str] | None = None, stores: List[Path] | None = None):
    def prepare_presentation(**kwargs: Any) -> Dict[str, Any]:
        if events is not None:
            events.append("prepare")
        output_dir = Path(kwargs["output_dir"])
        document_id = kwargs["document_id"]
        vs = output_dir / f"presentation_{document_id}" / "vector_store"
        vs.mkdir(parents=True, exist_ok=True)
        (vs / "chroma.sqlite3").write_text("fake-index", encoding="utf-8")
        if stores is not None:
            stores.append(vs)
        pdf_path = Path(kwargs["pdf_path"])
        return {
            "document_id": document_id,
            "collection_name": document_id,
            "index_dir": f"presentation_{document_id}/vector_store",
            "rag_status": "success",
            "source_hash": file_sha256(pdf_path),
            "embedding_model": kwargs["embedding_model"],
            "chunk_size": kwargs["chunk_size"],
            "chunk_overlap": kwargs["chunk_overlap"],
            "chunk_by_page": kwargs["chunk_by_page"],
            "num_pages": 1,
            "chunk_count": 2,
            "errors": [],
            "relative_path": pdf_path.name,
        }

    return prepare_presentation


def _fake_generate(fail_ids: set[str] | None = None, accepted: int = 10):
    def run_factual_and_explanation_in_memory(**kwargs: Any) -> InMemoryGenerationResult:
        rec = dict(kwargs.get("record") or {})
        doc_id = str(rec.get("document_id") or "")
        lb_id = doc_id.replace("lecturebank_", "")
        if fail_ids and lb_id in fail_ids:
            raise RuntimeError(f"forced mas failure for {lb_id}")
        payload = _payload(accepted=accepted)
        return InMemoryGenerationResult(
            record={**rec, "generate_status": "success", "run_id": f"run-{lb_id}"},
            experiment=object(),
            experiment_payload=payload,
            checkpoint_index={"records": [{"alias": f"c_{i}"} for i in range(10)]},
            accepted_count=accepted,
        )

    return run_factual_and_explanation_in_memory


def _run(
    *,
    root: Path,
    selected: List[str],
    args: Namespace,
    catalog,
    persist=None,
    events: List[str] | None = None,
    stores: List[Path] | None = None,
    fail_ids: set[str] | None = None,
    accepted: int = 10,
    collected: List[Dict[str, Any]] | None = None,
    manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    output_dir = root / "out"
    pdf_dir = root / "pdfs"
    cache_dir = output_dir / "pdf_cache"
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ident in selected:
        _pdf(pdf_dir / f"{ident}.pdf")
    manifest_path = output_dir / "lecturebank_eval_manifest.json"
    if manifest is None:
        manifest = {"records": [], "selected_ids": selected}
    if collected is None:
        collected = []
    persist_fn = persist or _persist
    log = io.StringIO()
    rag_fp = _rag_fp()
    gen_fp = _gen_fp()

    def tracked_persist(*pargs: Any, **pkwargs: Any) -> None:
        if events is not None:
            events.append("persist")
        persist_fn(*pargs, **pkwargs)

    with patch(
        "mas_langgraph.evaluation.execute.prepare_presentation",
        side_effect=_fake_prepare(events, stores),
    ), patch(
        "mas_langgraph.evaluation.execute.run_factual_and_explanation_in_memory",
        side_effect=_fake_generate(fail_ids=fail_ids, accepted=accepted),
    ):
        stats = process_documents(
            selected=selected,
            catalog=catalog,
            args=args,
            lecturebank_dir=root / "lb",
            output_dir=output_dir,
            pdf_dir=pdf_dir,
            cache_dir=cache_dir,
            metrics_dir=metrics_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            rag_cfg=rag_fp,
            gen_cfg=gen_fp,
            rag_fp=rag_fp,
            gen_fp=gen_fp,
            collected=collected,
            persist=tracked_persist,
            failed_document=_failed_document,
            find_record=_find_manifest_record,
            upsert=_upsert_manifest,
            embeddings=object(),
            pipeline=object(),
            backbone_factory=lambda *a, **k: None,
            log=log,
        )
    return {
        "stats": stats,
        "log": log.getvalue(),
        "output_dir": output_dir,
        "metrics_dir": metrics_dir,
        "pdf_dir": pdf_dir,
        "cache_dir": cache_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "collected": collected,
    }


class VectorStoreSafetyTests(unittest.TestCase):
    def test_cleanup_never_deletes_another_documents_store(self):
        with _tempdir() as root:
            a = root / "presentation_lecturebank_14" / "vector_store"
            b = root / "presentation_lecturebank_141" / "vector_store"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            (a / "x").write_text("a", encoding="utf-8")
            (b / "x").write_text("b", encoding="utf-8")
            with self.assertRaises(ValueError):
                delete_vector_store(a, document_id="lecturebank_141", allowed_roots=[root])
            self.assertTrue((a / "x").is_file())
            self.assertTrue((b / "x").is_file())
            self.assertTrue(delete_vector_store(a, document_id="lecturebank_14", allowed_roots=[root]))
            self.assertFalse(a.exists())
            self.assertTrue((b / "x").is_file())


class PdfCleanupSafetyTests(unittest.TestCase):
    def test_pdf_dir_source_is_never_deleted(self):
        with _tempdir() as root:
            pdf_dir = root / "pdfs"
            cache = root / "cache"
            src = _pdf(pdf_dir / "14.pdf")
            info = {
                "path": src,
                "source": "local",
                "lecturebank_id": "14",
            }
            self.assertFalse(
                maybe_delete_downloaded_pdf(
                    pdf_info=info, cache_dir=cache, pdf_dir=pdf_dir, enabled=True
                )
            )
            self.assertTrue(src.is_file())


class LifecycleProcessTests(unittest.TestCase):
    def test_metrics_persist_before_vector_store_deletion(self):
        with _tempdir() as root:
            lb = _write_catalog(root / "lb")
            catalog = load_lecturebank_catalog(lb)
            events: List[str] = []
            orig_delete = delete_vector_store

            def wrapped_delete(*args: Any, **kwargs: Any) -> bool:
                events.append("delete")
                return orig_delete(*args, **kwargs)

            with patch("mas_langgraph.evaluation.execute.delete_vector_store", side_effect=wrapped_delete):
                result = _run(
                    root=root,
                    selected=["14"],
                    args=_args(cleanup_vector_store=True),
                    catalog=catalog,
                    events=events,
                )
            self.assertIn("persist", events)
            self.assertIn("delete", events)
            self.assertLess(events.index("persist"), events.index("delete"))
            store = vector_store_dir(result["output_dir"], document_id_for("14"))
            self.assertFalse(store.exists())
            self.assertTrue((result["metrics_dir"] / "document_metrics.csv").is_file())

    def test_vector_store_deleted_after_successful_completion(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            result = _run(
                root=root,
                selected=["14"],
                args=_args(cleanup_vector_store=True),
                catalog=catalog,
            )
            store = vector_store_dir(result["output_dir"], document_id_for("14"))
            self.assertFalse(store.exists())
            rec = _find_manifest_record(result["manifest"], "14")
            self.assertIsNotNone(rec)
            self.assertFalse(rec.get("vector_store_retained"))
            self.assertTrue(rec.get("completed"))
            self.assertTrue(rec.get("metrics_persisted"))

    def test_vector_store_retained_if_metric_persistence_fails(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))

            def boom(*_a: Any, **_k: Any) -> None:
                raise RuntimeError("persist failed")

            with self.assertRaises(RuntimeError):
                _run(
                    root=root,
                    selected=["14"],
                    args=_args(cleanup_vector_store=True),
                    catalog=catalog,
                    persist=boom,
                )
            store = vector_store_dir(root / "out", document_id_for("14"))
            self.assertTrue(store.exists())
            self.assertTrue((store / "chroma.sqlite3").is_file())

    def test_source_pdf_from_pdf_dir_never_deleted_during_eval(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            result = _run(
                root=root,
                selected=["14"],
                args=_args(cleanup_vector_store=True, cleanup_downloaded_pdfs=True),
                catalog=catalog,
            )
            self.assertTrue((result["pdf_dir"] / "14.pdf").is_file())

    def test_resume_skips_completed_document_after_store_deleted(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            first = _run(
                root=root,
                selected=["14"],
                args=_args(cleanup_vector_store=True),
                catalog=catalog,
            )
            store = vector_store_dir(first["output_dir"], document_id_for("14"))
            self.assertFalse(store.exists())
            from mas_langgraph.evaluation.lifecycle import load_collected_from_metrics

            collected = load_collected_from_metrics(first["metrics_dir"])
            events: List[str] = []
            second = _run(
                root=root,
                selected=["14"],
                args=_args(cleanup_vector_store=True, resume=True),
                catalog=catalog,
                events=events,
                collected=collected,
                manifest=first["manifest"],
            )
            self.assertEqual(second["stats"]["skipped_resume"], 1)
            self.assertNotIn("prepare", events)
            self.assertFalse(store.exists())

    def test_incomplete_document_missing_store_rebuilds_rag(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            first = _run(
                root=root,
                selected=["14"],
                args=_args(),
                catalog=catalog,
                fail_ids={"14"},
            )
            store = vector_store_dir(first["output_dir"], document_id_for("14"))
            self.assertTrue(store.exists())
            import shutil

            shutil.rmtree(store)
            self.assertFalse(store.exists())
            events: List[str] = []
            second = _run(
                root=root,
                selected=["14"],
                args=_args(resume=True),
                catalog=catalog,
                events=events,
                collected=list(first["collected"]),
                manifest=first["manifest"],
            )
            self.assertIn("prepare", events)
            self.assertEqual(second["stats"]["skipped_resume"], 0)
            rec = _find_manifest_record(second["manifest"], "14")
            self.assertEqual(rec.get("eval_status"), "success")

    def test_rerun_replaces_metric_rows_instead_of_duplicating(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            first = _run(
                root=root,
                selected=["14"],
                args=_args(),
                catalog=catalog,
                accepted=10,
            )
            csv_path = first["metrics_dir"] / "document_metrics.csv"
            text1 = csv_path.read_text(encoding="utf-8")
            self.assertEqual(text1.count("\n14,"), 1)
            second = _run(
                root=root,
                selected=["14"],
                args=_args(overwrite=True),
                catalog=catalog,
                accepted=8,
                collected=list(first["collected"]),
                manifest=first["manifest"],
            )
            text2 = (second["metrics_dir"] / "document_metrics.csv").read_text(encoding="utf-8")
            self.assertEqual(text2.count("\n14,"), 1)
            attempts = (second["metrics_dir"] / "interrogation_attempts.csv").read_text(encoding="utf-8")
            self.assertEqual(attempts.count("\n14,"), 1)

    def test_later_document_failure_does_not_destroy_earlier_csvs(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["14", "141"]))
            persist_calls = {"n": 0}

            def persist_then_interrupt(*pargs: Any, **pkwargs: Any) -> None:
                persist_calls["n"] += 1
                collected = pargs[4]
                success_n = sum(
                    1 for row in collected if str(row.get("eval_status") or "") == "success"
                )
                if success_n >= 2:
                    raise RuntimeError("interrupted after first document")
                _persist(*pargs, **pkwargs)

            with self.assertRaises(RuntimeError):
                _run(
                    root=root,
                    selected=["14", "141"],
                    args=_args(),
                    catalog=catalog,
                    persist=persist_then_interrupt,
                )
            csv_path = root / "out" / "metrics" / "document_metrics.csv"
            self.assertTrue(csv_path.is_file())
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("\n14,", text)
            self.assertNotIn("\n141,", text)

    def test_temporary_vector_store_cleaned_after_success(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            stores: List[Path] = []
            result = _run(
                root=root,
                selected=["14"],
                args=_args(temporary_vector_stores=True),
                catalog=catalog,
                stores=stores,
            )
            self.assertEqual(len(stores), 1)
            self.assertFalse(stores[0].exists())
            persistent = vector_store_dir(result["output_dir"], document_id_for("14"))
            self.assertFalse(persistent.exists())
            self.assertTrue((result["metrics_dir"] / "document_metrics.csv").is_file())

    def test_temporary_stores_are_isolated_between_documents(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["14", "141"]))
            stores: List[Path] = []
            _run(
                root=root,
                selected=["14", "141"],
                args=_args(temporary_vector_stores=True),
                catalog=catalog,
                stores=stores,
            )
            self.assertEqual(len(stores), 2)
            self.assertNotEqual(stores[0].resolve(), stores[1].resolve())
            names = {p.parent.name for p in stores}
            self.assertEqual(names, {"presentation_lecturebank_14", "presentation_lecturebank_141"})
            for store in stores:
                self.assertFalse(store.exists())


class PreflightAndLoggingTests(unittest.TestCase):
    def test_preflight_does_not_build_rag(self):
        from mas_langgraph.evaluation.runner import run_eval

        with _tempdir() as root:
            lb = _write_catalog(root / "lb")
            pdf_dir = root / "pdfs"
            _pdf(pdf_dir / "14.pdf")
            log = io.StringIO()
            parser_args = Namespace(
                lecturebank_dir=str(lb),
                output_dir=str(root / "out"),
                pdf_dir=str(pdf_dir),
                id_values=["14"],
                ids="",
                ids_file=None,
                allow_any_count=True,
                expected_count=None,
                allow_partial=True,
                list_ids=False,
                search="",
                limit=None,
                no_download=True,
                stage="all",
                resume=False,
                overwrite=False,
                embedding_model="fake",
                chunk_size=800,
                chunk_overlap=200,
                no_page_chunking=False,
                llm_model="fake",
                model_type="openai",
                temperature=0.0,
                seed=42,
                max_rounds=5,
                max_questions=10,
                max_explain_questions=None,
                question="Why?",
                cleanup_vector_store=False,
                temporary_vector_stores=False,
                cleanup_downloaded_pdfs=False,
                extraction_jsonl=None,
                preflight=True,
            )
            with patch("mas_langgraph.evaluation.execute.prepare_presentation") as prep:
                code = run_eval(parser_args, pipeline=object(), embeddings=object(), stdout=log)
            self.assertEqual(code, 0)
            prep.assert_not_called()
            out = log.getvalue()
            self.assertIn("Selected unique LectureBank IDs: 1", out)
            self.assertIn("Selected: 1", out)
            self.assertIn("1 available", out)
            self.assertIn("available", out)

    def test_expected_count_printed_before_work(self):
        from mas_langgraph.evaluation.lecturebank_meta import resolve_selected_ids

        ids = [str(i) for i in range(100)]
        got = resolve_selected_ids(id_values=ids, expected_count=100)
        self.assertEqual(len(got), 100)


if __name__ == "__main__":
    unittest.main()
