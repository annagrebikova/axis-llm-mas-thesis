"""Offline tests for LectureBank JSONL extraction source (no provider calls)."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List
from unittest.mock import patch

from mas_langgraph.batch.generate import InMemoryGenerationResult, validate_prepared_record
from mas_langgraph.batch.ids import file_sha256
from mas_langgraph.batch.prepare import prepare_presentation
from mas_langgraph.evaluation.execute import jsonl_lookup_for_selected, process_documents, run_preflight
from mas_langgraph.evaluation.lecturebank_meta import LectureBankRecord, load_lecturebank_catalog
from mas_langgraph.evaluation.lifecycle import load_collected_from_metrics, vector_store_dir
from mas_langgraph.evaluation.pdf_source import (
    classify_lecturebank_source,
    document_id_for,
    resolve_lecturebank_source,
)
from mas_langgraph.evaluation.runner import _failed_document, _find_manifest_record, _persist, _upsert_manifest
from rag_pipeline import (
    extraction_has_usable_pages,
    load_selected_extractions,
    match_lecturebank_extractions,
    metadata_composite_key,
    normalize_extraction_pdf_data,
    normalize_source_url,
    peek_jsonl_id,
    select_first_usable_jsonl_ids,
)


@contextmanager
def _tempdir() -> Iterator[Path]:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        yield Path(tmp.name)
    finally:
        tmp.cleanup()


class FakeEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        vec = [0.0] * 8
        data = (text or "").encode("utf-8")
        if not data:
            return vec
        for i, byte in enumerate(data):
            vec[i % 8] += (byte % 17) / 17.0
        return vec


def _write_catalog(root: Path, ids: List[str] | None = None) -> Path:
    ids = ids or ["14", "137", "141"]
    root.mkdir(parents=True, exist_ok=True)
    lines = ["ID\tInstructor\tTitle\tTopic\tURL\tVenue\tYear"]
    for ident in ids:
        lines.append(
            f"{ident}\tAda\tTitle {ident}\t1\thttp://example.test/{ident}.pdf\tYale\t2016"
        )
    (root / "alldata.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 dummy " + path.name.encode("utf-8"))
    return path


def _jsonl_record(
    ident: str,
    *,
    empty: bool = False,
    pages: List[Dict[str, Any]] | None = None,
    title: str | None = None,
    url: str | None = None,
    instructor: str = "Ada",
    venue: str = "Yale",
    year: str = "2016",
    include_url: bool = True,
) -> Dict[str, Any]:
    if pages is None:
        pages = (
            [{"page": 1, "text": ""}]
            if empty
            else [
                {"page": 1, "text": f"Planning page one for {ident}."},
                {"page": 2, "text": f"Second page unique to {ident}."},
            ]
        )
    metadata: Dict[str, Any] = {
        "title": f"Title {ident}" if title is None else title,
        "instructor": instructor,
        "venue": venue,
        "year": year,
    }
    if include_url:
        metadata["url"] = f"http://example.test/{ident}.pdf" if url is None else url
    return {
        "id": ident,
        "metadata": metadata,
        "text": {
            "full_text": "SHOULD_NOT_REPLACE_PAGES",
            "pages_text": pages,
            "num_pages": len(pages),
        },
        "statistics": {"has_text": not empty, "num_pages": len(pages)},
        "images": ["drop-me"],
        "tables": ["drop-me"],
        "ocr": {"text": "drop-me"},
    }


def _write_jsonl(path: Path, rows: List[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row.rstrip("\n") + "\n")
            else:
                handle.write(json.dumps(row) + "\n")
    return path


def _payload() -> Dict[str, Any]:
    return {
        "cases": [
            {
                "candidate_ref": {"key": "CandidateMCQ:q1:v1", "logical_id": "q1", "version": 1},
                "conditions": {
                    "axis": {
                        "evidence_summary": {},
                        "raw_evidence": {
                            "interrogation_attempts": [],
                            "replay_comparisons": [
                                {
                                    "intervention_signature": {
                                        "operation": "REMOVE",
                                        "target_ref": {"key": "EvidenceChunk:e1:v1"},
                                        "checkpoint_alias": "c_0",
                                        "checkpoint_id": "ckpt-0",
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
                            ],
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
    }


def _fake_generate(**kwargs: Any) -> InMemoryGenerationResult:
    rec = dict(kwargs.get("record") or {})
    return InMemoryGenerationResult(
        record={**rec, "generate_status": "success", "run_id": "run-jsonl"},
        experiment=object(),
        experiment_payload=_payload(),
        checkpoint_index={"records": [{"alias": f"c_{i}"} for i in range(10)]},
        accepted_count=1,
    )


def _fake_prepare(events: List[str] | None = None, captured: List[Dict[str, Any]] | None = None):
    def prepare_presentation(**kwargs: Any) -> Dict[str, Any]:
        if events is not None:
            events.append("prepare")
        if captured is not None:
            captured.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        document_id = kwargs["document_id"]
        vs = output_dir / f"presentation_{document_id}" / "vector_store"
        vs.mkdir(parents=True, exist_ok=True)
        (vs / "chroma.sqlite3").write_text("fake-index", encoding="utf-8")
        pdf_data = kwargs.get("pdf_data")
        source_hash = kwargs.get("source_hash")
        if not source_hash and kwargs.get("pdf_path"):
            source_hash = file_sha256(Path(kwargs["pdf_path"]))
        return {
            "document_id": document_id,
            "collection_name": document_id,
            "index_dir": f"presentation_{document_id}/vector_store",
            "rag_status": "success",
            "source_hash": source_hash,
            "embedding_model": kwargs["embedding_model"],
            "chunk_size": kwargs["chunk_size"],
            "chunk_overlap": kwargs["chunk_overlap"],
            "chunk_by_page": kwargs["chunk_by_page"],
            "num_pages": 2 if pdf_data else 1,
            "chunk_count": 2,
            "errors": [],
            "relative_path": Path(kwargs.get("jsonl_path") or kwargs.get("pdf_path") or "x").name,
            "input_source_type": kwargs.get("input_source_type") or "",
            "jsonl_path": str(kwargs.get("jsonl_path") or ""),
        }

    return prepare_presentation


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
        extraction_jsonl=None,
        jsonl_authoritative=False,
    )
    base.update(kwargs)
    return Namespace(**base)


def _run_eval(
    *,
    root: Path,
    selected: List[str],
    jsonl_path: Path,
    args: Namespace,
    catalog,
    events: List[str] | None = None,
    captured: List[Dict[str, Any]] | None = None,
    collected: List[Dict[str, Any]] | None = None,
    manifest: Dict[str, Any] | None = None,
    create_pdfs: bool = False,
) -> Dict[str, Any]:
    output_dir = root / "out"
    pdf_dir = root / "pdfs"
    cache_dir = output_dir / "pdf_cache"
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if create_pdfs:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for ident in selected:
            _pdf(pdf_dir / f"{ident}.pdf")
    args.extraction_jsonl = jsonl_path
    manifest_path = output_dir / "lecturebank_eval_manifest.json"
    if manifest is None:
        manifest = {"records": [], "selected_ids": selected}
    if collected is None:
        collected = []
    rag_fp = {
        "embedding_model": "fake",
        "chunk_size": 800,
        "chunk_overlap": 200,
        "chunk_by_page": True,
    }
    gen_fp = {
        "llm_model": "fake",
        "model_type": "openai",
        "temperature": 0.0,
        "seed": 42,
        "max_rounds": 1,
        "max_questions": 1,
        "max_explain_questions": 1,
        "question": "Why?",
        "orchestrator": "langgraph",
    }
    log = io.StringIO()

    def tracked_persist(*pargs: Any, **pkwargs: Any) -> None:
        if events is not None:
            events.append("persist")
        _persist(*pargs, **pkwargs)

    with patch(
        "mas_langgraph.evaluation.execute.prepare_presentation",
        side_effect=_fake_prepare(events, captured),
    ), patch(
        "mas_langgraph.evaluation.execute.run_factual_and_explanation_in_memory",
        side_effect=_fake_generate,
    ):
        stats = process_documents(
            selected=selected,
            catalog=catalog,
            args=args,
            lecturebank_dir=root / "lb",
            output_dir=output_dir,
            pdf_dir=pdf_dir if create_pdfs else None,
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
        "collected": collected,
    }


class JsonlLoaderTests(unittest.TestCase):
    def test_exact_id_found_and_14_does_not_match_141(self):
        with _tempdir() as root:
            path = _write_jsonl(
                root / "ext.jsonl",
                [_jsonl_record("14"), _jsonl_record("141"), "not-json {", _jsonl_record("137")],
            )
            found = load_selected_extractions(path, ["14", "141", "999"])
            self.assertEqual(found["14"].status, "usable")
            self.assertEqual(found["141"].status, "usable")
            self.assertEqual(found["999"].status, "missing")
            self.assertNotEqual(found["14"].record["id"], found["141"].record["id"])
            self.assertTrue(all("Second page unique to 14" in p["text"] or p["page"] == 1 for p in found["14"].record["text"]["pages_text"]))
            self.assertFalse(any("unique to 141" in p["text"] for p in found["14"].record["text"]["pages_text"]))

    def test_peek_id_is_exact(self):
        self.assertEqual(peek_jsonl_id('{"id": "14", "text": {}}'), "14")
        self.assertEqual(peek_jsonl_id('{"id": "141"}'), "141")
        self.assertNotEqual(peek_jsonl_id('{"id": "141"}'), "14")

    def test_page_text_preserved_and_heavy_fields_dropped(self):
        raw = _jsonl_record("137")
        normalized = normalize_extraction_pdf_data(raw)
        pages = normalized["text"]["pages_text"]
        self.assertEqual(pages[0]["page"], 1)
        self.assertIn("Planning page one", pages[0]["text"])
        self.assertEqual(pages[1]["page"], 2)
        self.assertNotIn("images", normalized)
        self.assertNotIn("ocr", normalized)
        self.assertTrue(extraction_has_usable_pages(normalized))
        self.assertIn("Planning page one", normalized["text"]["full_text"])

    def test_empty_and_duplicate_and_malformed(self):
        with _tempdir() as root:
            path = _write_jsonl(
                root / "ext.jsonl",
                [
                    "{{{{",
                    _jsonl_record("14", empty=True),
                    _jsonl_record("137"),
                    _jsonl_record("137"),
                    _jsonl_record("141"),
                ],
            )
            found = load_selected_extractions(path, ["14", "137", "141"])
            self.assertEqual(found["14"].status, "empty")
            self.assertEqual(found["137"].status, "duplicate")
            self.assertIsNone(found["137"].record)
            self.assertEqual(found["141"].status, "usable")


class JsonlPrepareTests(unittest.TestCase):
    def test_jsonl_reaches_chunking_and_creates_vector_store(self):
        with _tempdir() as root:
            jsonl = _write_jsonl(root / "ext.jsonl", [_jsonl_record("137")])
            lookup = load_selected_extractions(jsonl, ["137"])
            item = lookup["137"]
            out = root / "out"
            rec = prepare_presentation(
                pdf_data=item.record,
                source_hash=item.source_hash,
                input_source_type="extraction_jsonl",
                jsonl_path=jsonl,
                input_dir=jsonl.parent,
                output_dir=out,
                document_id="lecturebank_137",
                collection_name="lecturebank_137",
                embedding_model="fake",
                chunk_size=800,
                chunk_overlap=200,
                chunk_by_page=True,
                embeddings=FakeEmbeddings(),
            )
            self.assertEqual(rec["rag_status"], "success")
            self.assertGreaterEqual(int(rec["chunk_count"]), 2)
            self.assertEqual(rec["document_id"], "lecturebank_137")
            self.assertEqual(rec["collection_name"], "lecturebank_137")
            store = out / "presentation_lecturebank_137" / "vector_store"
            self.assertTrue(store.exists())
            self.assertTrue(any(store.iterdir()))
            self.assertFalse(list(root.glob("**/*.pdf")))
            index = validate_prepared_record(
                rec,
                input_dir=jsonl.parent,
                output_dir=out,
                expected_config={
                    "embedding_model": "fake",
                    "chunk_size": 800,
                    "chunk_overlap": 200,
                    "chunk_by_page": True,
                },
            )
            self.assertEqual(index, store.resolve())
            from rag_pipeline import RAGPipeline

            pipe = RAGPipeline(
                vector_store_path=str(store),
                embedding_model="fake",
                chunk_size=800,
                chunk_overlap=200,
                embeddings=FakeEmbeddings(),
            )
            pipe.load_existing_index(collection_name="lecturebank_137")
            metas = pipe.chunk_metadatas()
            self.assertTrue(metas)
            self.assertTrue(all(m.get("pdf_id") == "lecturebank_137" for m in metas))
            self.assertTrue(any(int(m.get("page_number") or 0) == 1 for m in metas))
            self.assertTrue(any(int(m.get("page_number") or 0) == 2 for m in metas))
            pipe.close()


class JsonlSourcePriorityTests(unittest.TestCase):
    def test_jsonl_preferred_when_usable(self):
        with _tempdir() as root:
            jsonl = _write_jsonl(root / "ext.jsonl", [_jsonl_record("137")])
            pdf_dir = root / "pdfs"
            _pdf(pdf_dir / "137.pdf")
            lookup = load_selected_extractions(jsonl, ["137"])
            info = resolve_lecturebank_source(
                lecturebank_id="137",
                url="http://example.test/137.pdf",
                cache_dir=root / "cache",
                pdf_dir=pdf_dir,
                no_download=True,
                jsonl_lookup=lookup,
            )
            self.assertEqual(info["source"], "extraction_jsonl")
            self.assertEqual(info["input_source_type"], "extraction_jsonl")
            self.assertIsNone(info["path"])
            self.assertIn("pages_text", info["pdf_data"]["text"])

    def test_missing_jsonl_falls_back_to_pdf(self):
        with _tempdir() as root:
            jsonl = _write_jsonl(root / "ext.jsonl", [_jsonl_record("141")])
            pdf_dir = root / "pdfs"
            src = _pdf(pdf_dir / "137.pdf")
            lookup = load_selected_extractions(jsonl, ["137"])
            info = resolve_lecturebank_source(
                lecturebank_id="137",
                cache_dir=root / "cache",
                pdf_dir=pdf_dir,
                no_download=True,
                jsonl_lookup=lookup,
            )
            self.assertEqual(info["source"], "local")
            self.assertEqual(info["jsonl_status"], "jsonl_miss")
            self.assertEqual(Path(info["path"]).resolve(), src.resolve())

    def test_empty_jsonl_falls_back_to_pdf(self):
        with _tempdir() as root:
            jsonl = _write_jsonl(root / "ext.jsonl", [_jsonl_record("137", empty=True)])
            pdf_dir = root / "pdfs"
            _pdf(pdf_dir / "137.pdf")
            lookup = load_selected_extractions(jsonl, ["137"])
            info = resolve_lecturebank_source(
                lecturebank_id="137",
                cache_dir=root / "cache",
                pdf_dir=pdf_dir,
                no_download=True,
                jsonl_lookup=lookup,
            )
            self.assertEqual(info["source"], "local")
            self.assertEqual(info["jsonl_status"], "empty")

    def test_duplicate_jsonl_is_not_used_silently(self):
        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [_jsonl_record("137"), _jsonl_record("137")],
            )
            pdf_dir = root / "pdfs"
            _pdf(pdf_dir / "137.pdf")
            lookup = load_selected_extractions(jsonl, ["137"])
            self.assertEqual(lookup["137"].status, "duplicate")
            info = resolve_lecturebank_source(
                lecturebank_id="137",
                cache_dir=root / "cache",
                pdf_dir=pdf_dir,
                no_download=True,
                jsonl_lookup=lookup,
            )
            self.assertEqual(info["source"], "local")
            self.assertEqual(info["jsonl_status"], "duplicate")


class JsonlEvalLoopTests(unittest.TestCase):
    def test_preflight_reports_extraction_jsonl(self):
        with _tempdir() as root:
            lb = _write_catalog(root / "lb")
            catalog = load_lecturebank_catalog(lb)
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [_jsonl_record("137"), _jsonl_record("14", empty=True)],
            )
            pdf_dir = root / "pdfs"
            _pdf(pdf_dir / "14.pdf")
            log = io.StringIO()
            lookup = load_selected_extractions(jsonl, ["137", "14", "141"])
            code = run_preflight(
                ["137", "14", "141"],
                catalog=catalog,
                cache_dir=root / "cache",
                pdf_dir=pdf_dir,
                lecturebank_dir=lb,
                no_download=True,
                log=log,
                jsonl_path=jsonl,
            )
            out = log.getvalue()
            self.assertIn("137\tavailable\textraction_jsonl", out)
            self.assertIn("jsonl_id=137", out)
            self.assertIn("match=url", out)
            self.assertIn("14\tavailable\tlocal_pdf", out)
            self.assertIn("141\tunavailable", out)
            self.assertIn("JSONL available: 1", out)
            self.assertIn("Local/cache PDF: 1", out)
            self.assertEqual(lookup["137"].status, "usable")
            self.assertEqual(code, 1)

    def test_temporary_jsonl_creates_no_pdf_and_cleans_store(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            jsonl = _write_jsonl(root / "ext.jsonl", [_jsonl_record("137")])
            captured: List[Dict[str, Any]] = []
            events: List[str] = []
            result = _run_eval(
                root=root,
                selected=["137"],
                jsonl_path=jsonl,
                args=_args(temporary_vector_stores=True),
                catalog=catalog,
                events=events,
                captured=captured,
                create_pdfs=False,
            )
            self.assertEqual(len(captured), 1)
            self.assertIsNone(captured[0].get("pdf_path"))
            self.assertTrue(captured[0].get("pdf_data"))
            self.assertEqual(captured[0]["document_id"], "lecturebank_137")
            self.assertEqual(captured[0]["collection_name"], "lecturebank_137")
            self.assertFalse(list(result["cache_dir"].glob("*.pdf")))
            self.assertFalse(result["pdf_dir"].exists() and any(result["pdf_dir"].glob("*.pdf")))
            rec = _find_manifest_record(result["manifest"], "137")
            self.assertEqual(rec.get("input_source_type"), "extraction_jsonl")
            self.assertEqual(rec.get("extraction_jsonl_id"), "137")
            self.assertEqual(rec.get("jsonl_match_strategy"), "normalized_url")
            self.assertEqual(rec.get("jsonl_match_status"), "unique_match")
            self.assertEqual(rec.get("lecturebank_title"), "Title 137")
            self.assertEqual(rec.get("jsonl_title"), "Title 137")
            self.assertFalse(vector_store_dir(result["output_dir"], document_id_for("137")).exists())
            self.assertIn("persist", events)

    def test_resume_skips_completed_jsonl_document(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            jsonl = _write_jsonl(root / "ext.jsonl", [_jsonl_record("137")])
            first = _run_eval(
                root=root,
                selected=["137"],
                jsonl_path=jsonl,
                args=_args(temporary_vector_stores=True),
                catalog=catalog,
            )
            events: List[str] = []
            collected = load_collected_from_metrics(first["metrics_dir"])
            second = _run_eval(
                root=root,
                selected=["137"],
                jsonl_path=jsonl,
                args=_args(temporary_vector_stores=True, resume=True),
                catalog=catalog,
                events=events,
                collected=collected,
                manifest=first["manifest"],
            )
            self.assertEqual(second["stats"]["skipped_resume"], 1)
            self.assertNotIn("prepare", events)

    def test_classify_preflight_jsonl(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            jsonl = _write_jsonl(root / "ext.jsonl", [_jsonl_record("137")])
            lookup = jsonl_lookup_for_selected(jsonl, ["137"], catalog)
            info = classify_lecturebank_source(
                lecturebank_id="137",
                url=catalog.require("137").url,
                jsonl_lookup=lookup,
                no_download=True,
            )
            self.assertEqual(info["status"], "available")
            self.assertEqual(info["source"], "extraction_jsonl")
            self.assertEqual(info["extraction_jsonl_id"], "137")
            self.assertEqual(info["jsonl_match_strategy"], "normalized_url")
            self.assertEqual(info["jsonl_match_status"], "unique_match")


class JsonlIdentityMatchTests(unittest.TestCase):
    def _pos_record(self, ident: str = "137") -> LectureBankRecord:
        return LectureBankRecord(
            lecturebank_id=ident,
            title="Statistical Part of Speech Tagging",
            instructor="Ada",
            venue="Yale",
            year="2016",
            url="http://example.test/pos.pdf",
            topic_id="1",
        )

    def test_same_numeric_id_conflicting_metadata_is_not_matched(self):
        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record(
                        "137",
                        title="Planning",
                        url="http://example.test/planning.pdf",
                    )
                ],
            )
            lookup = match_lecturebank_extractions(jsonl, [self._pos_record()])
            item = lookup["137"]
            self.assertNotEqual(item.status, "usable")
            self.assertEqual(item.match_status, "metadata_conflict")
            self.assertIsNone(item.record)
            pdf_dir = root / "pdfs"
            src = _pdf(pdf_dir / "137.pdf")
            info = resolve_lecturebank_source(
                lecturebank_id="137",
                url="http://example.test/pos.pdf",
                cache_dir=root / "cache",
                pdf_dir=pdf_dir,
                no_download=True,
                jsonl_lookup=lookup,
            )
            self.assertEqual(info["source"], "local")
            self.assertEqual(info["jsonl_match_status"], "metadata_conflict")
            self.assertEqual(Path(info["path"]).resolve(), src.resolve())

    def test_conflicting_id_does_not_block_unique_url_match_with_other_jsonl_id(self):
        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record(
                        "137",
                        title="Planning",
                        url="http://example.test/planning.pdf",
                        pages=[
                            {"page": 1, "text": "Planning"},
                            {"page": 2, "text": "STRIPS Planning"},
                        ],
                    ),
                    _jsonl_record(
                        "999",
                        title="Statistical Part of Speech Tagging",
                        url="HTTP://EXAMPLE.TEST/pos.pdf#page=1/",
                        pages=[
                            {"page": 1, "text": "POS tagging lecture."},
                            {"page": 2, "text": "HMM and Viterbi."},
                        ],
                    ),
                ],
            )
            lookup = match_lecturebank_extractions(jsonl, [self._pos_record()])
            item = lookup["137"]
            self.assertEqual(item.status, "usable")
            self.assertEqual(item.match_status, "unique_match")
            self.assertEqual(item.match_strategy, "normalized_url")
            self.assertEqual(item.extraction_id, "999")
            self.assertEqual(item.record["id"], "999")
            self.assertIn("POS tagging lecture", item.record["text"]["pages_text"][0]["text"])
            self.assertFalse(any("STRIPS" in p["text"] for p in item.record["text"]["pages_text"]))

    def test_unique_url_match_with_different_numeric_ids(self):
        rec = LectureBankRecord(
            lecturebank_id="137",
            title="Statistical Part of Speech Tagging",
            instructor="Ada",
            venue="Yale",
            year="2016",
            url="http://example.test/pos.pdf",
        )
        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [_jsonl_record("42", title="Statistical Part of Speech Tagging", url="http://example.test/pos.pdf")],
            )
            lookup = match_lecturebank_extractions(jsonl, [rec])
            self.assertEqual(lookup["137"].extraction_id, "42")
            self.assertEqual(lookup["137"].match_strategy, "normalized_url")
            self.assertEqual(normalize_source_url("HTTP://EXAMPLE.TEST/pos.pdf/"), "http://example.test/pos.pdf")

    def test_unique_composite_metadata_match(self):
        rec = LectureBankRecord(
            lecturebank_id="137",
            title="Statistical Part of Speech Tagging",
            instructor="Ada",
            venue="Yale",
            year="2016",
            url="",
        )
        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record(
                        "777",
                        title="  Statistical   Part of Speech Tagging ",
                        instructor="ada",
                        venue="Yale",
                        year="2016",
                        include_url=False,
                    )
                ],
            )
            lookup = match_lecturebank_extractions(jsonl, [rec])
            self.assertEqual(lookup["137"].status, "usable")
            self.assertEqual(lookup["137"].extraction_id, "777")
            self.assertEqual(lookup["137"].match_strategy, "metadata_composite")
            self.assertTrue(
                metadata_composite_key("Statistical Part of Speech Tagging", "Ada", "Yale", "2016")
            )

    def test_ambiguous_metadata_match_falls_back_to_pdf(self):
        rec = LectureBankRecord(
            lecturebank_id="137",
            title="Statistical Part of Speech Tagging",
            instructor="Ada",
            venue="Yale",
            year="2016",
            url="",
        )
        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record("10", title="Statistical Part of Speech Tagging", include_url=False),
                    _jsonl_record("11", title="Statistical Part of Speech Tagging", include_url=False),
                ],
            )
            pdf_dir = root / "pdfs"
            _pdf(pdf_dir / "137.pdf")
            lookup = match_lecturebank_extractions(jsonl, [rec])
            self.assertEqual(lookup["137"].match_status, "ambiguous_match")
            self.assertIsNone(lookup["137"].record)
            info = resolve_lecturebank_source(
                lecturebank_id="137",
                cache_dir=root / "cache",
                pdf_dir=pdf_dir,
                no_download=True,
                jsonl_lookup=lookup,
            )
            self.assertEqual(info["source"], "local")
            self.assertEqual(info["jsonl_match_status"], "ambiguous_match")

    def test_no_match_falls_back_to_pdf(self):
        rec = LectureBankRecord(
            lecturebank_id="137",
            title="Statistical Part of Speech Tagging",
            instructor="Ada",
            venue="Yale",
            year="2016",
            url="http://example.test/pos.pdf",
        )
        with _tempdir() as root:
            jsonl = _write_jsonl(root / "ext.jsonl", [_jsonl_record("141")])
            pdf_dir = root / "pdfs"
            src = _pdf(pdf_dir / "137.pdf")
            lookup = match_lecturebank_extractions(jsonl, [rec])
            self.assertEqual(lookup["137"].match_status, "no_match")
            info = resolve_lecturebank_source(
                lecturebank_id="137",
                url=rec.url,
                cache_dir=root / "cache",
                pdf_dir=pdf_dir,
                no_download=True,
                jsonl_lookup=lookup,
            )
            self.assertEqual(info["source"], "local")
            self.assertEqual(Path(info["path"]).resolve(), src.resolve())

    def test_duplicate_extraction_records_are_not_used(self):
        rec = self._pos_record()
        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record("999", title="Statistical Part of Speech Tagging", url="http://example.test/pos.pdf"),
                    _jsonl_record("999", title="Statistical Part of Speech Tagging", url="http://example.test/pos.pdf"),
                ],
            )
            pdf_dir = root / "pdfs"
            _pdf(pdf_dir / "137.pdf")
            lookup = match_lecturebank_extractions(jsonl, [rec])
            self.assertEqual(lookup["137"].match_status, "duplicate_jsonl_id")
            self.assertIsNone(lookup["137"].record)
            info = resolve_lecturebank_source(
                lecturebank_id="137",
                url=rec.url,
                cache_dir=root / "cache",
                pdf_dir=pdf_dir,
                no_download=True,
                jsonl_lookup=lookup,
            )
            self.assertEqual(info["source"], "local")

    def test_preflight_and_manifest_preserve_both_ids(self):
        with _tempdir() as root:
            lb = root / "lb"
            lb.mkdir(parents=True, exist_ok=True)
            (lb / "alldata.tsv").write_text(
                "ID\tInstructor\tTitle\tTopic\tURL\tVenue\tYear\n"
                "137\tAda\tStatistical Part of Speech Tagging\t1\thttp://example.test/pos.pdf\tYale\t2016\n",
                encoding="utf-8",
            )
            catalog = load_lecturebank_catalog(lb)
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record("137", title="Planning", url="http://example.test/planning.pdf"),
                    _jsonl_record(
                        "999",
                        title="Statistical Part of Speech Tagging",
                        url="http://example.test/pos.pdf",
                    ),
                ],
            )
            log = io.StringIO()
            code = run_preflight(
                ["137"],
                catalog=catalog,
                cache_dir=root / "cache",
                pdf_dir=None,
                lecturebank_dir=lb,
                no_download=True,
                log=log,
                jsonl_path=jsonl,
            )
            out = log.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("137\tavailable\textraction_jsonl", out)
            self.assertIn("jsonl_id=999", out)
            self.assertIn("match=url", out)
            self.assertNotIn("jsonl_id=137", out)

            captured: List[Dict[str, Any]] = []
            result = _run_eval(
                root=root,
                selected=["137"],
                jsonl_path=jsonl,
                args=_args(temporary_vector_stores=True),
                catalog=catalog,
                captured=captured,
            )
            self.assertEqual(captured[0]["document_id"], "lecturebank_137")
            self.assertEqual(captured[0]["collection_name"], "lecturebank_137")
            self.assertEqual(captured[0]["pdf_data"]["id"], "999")
            rec = _find_manifest_record(result["manifest"], "137")
            self.assertEqual(rec.get("lecturebank_id"), "137")
            self.assertEqual(rec.get("extraction_jsonl_id"), "999")
            self.assertEqual(rec.get("jsonl_match_strategy"), "normalized_url")
            self.assertEqual(rec.get("jsonl_match_status"), "unique_match")
            self.assertEqual(rec.get("lecturebank_title"), "Statistical Part of Speech Tagging")
            self.assertEqual(rec.get("jsonl_title"), "Statistical Part of Speech Tagging")
            self.assertEqual(document_id_for("137"), "lecturebank_137")

    def test_preflight_does_not_treat_conflicting_same_id_as_available_jsonl(self):
        with _tempdir() as root:
            lb = root / "lb"
            lb.mkdir(parents=True, exist_ok=True)
            (lb / "alldata.tsv").write_text(
                "ID\tInstructor\tTitle\tTopic\tURL\tVenue\tYear\n"
                "137\tAda\tStatistical Part of Speech Tagging\t1\thttp://example.test/pos.pdf\tYale\t2016\n",
                encoding="utf-8",
            )
            catalog = load_lecturebank_catalog(lb)
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [_jsonl_record("137", title="Planning", url="http://example.test/planning.pdf")],
            )
            log = io.StringIO()
            run_preflight(
                ["137"],
                catalog=catalog,
                cache_dir=root / "cache",
                pdf_dir=None,
                lecturebank_dir=lb,
                no_download=True,
                log=log,
                jsonl_path=jsonl,
            )
            out = log.getvalue()
            self.assertNotIn("137\tavailable\textraction_jsonl", out)
            self.assertIn("metadata_conflict", out)


class JsonlAuthoritativeTests(unittest.TestCase):
    def test_authoritative_loads_jsonl_137_even_when_tsv_conflicts(self):
        with _tempdir() as root:
            lb = _write_catalog(root / "lb")
            catalog = load_lecturebank_catalog(lb)
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record(
                        "137",
                        title="Planning",
                        instructor="George Konidaris",
                        venue="Duke",
                        year="2016",
                        url="http://example.test/planning.pdf",
                        pages=[
                            {"page": 1, "text": "Planning"},
                            {"page": 2, "text": "STRIPS Planning"},
                        ],
                    )
                ],
            )
            captured: List[Dict[str, Any]] = []
            result = _run_eval(
                root=root,
                selected=["137"],
                jsonl_path=jsonl,
                args=_args(jsonl_authoritative=True, temporary_vector_stores=True),
                catalog=catalog,
                captured=captured,
            )
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["pdf_data"]["id"], "137")
            self.assertEqual(captured[0]["pdf_data"]["metadata"]["title"], "Planning")
            self.assertEqual(captured[0]["document_id"], "lecturebank_137")
            self.assertEqual(captured[0]["collection_name"], "lecturebank_137")
            rec = _find_manifest_record(result["manifest"], "137")
            self.assertEqual(rec.get("title"), "Planning")
            self.assertEqual(rec.get("instructor"), "George Konidaris")
            self.assertEqual(rec.get("venue"), "Duke")
            self.assertEqual(rec.get("input_source_type"), "extraction_jsonl")
            self.assertEqual(rec.get("extraction_jsonl_id"), "137")
            self.assertEqual(rec.get("jsonl_match_strategy"), "jsonl_id")
            dm = result["collected"][0]["document_metrics"]
            self.assertEqual(dm["title"], "Planning")
            self.assertEqual(dm["instructor"], "George Konidaris")
            self.assertEqual(dm["pdf_source"], "extraction_jsonl")
            self.assertNotEqual(catalog.require("137").title, "Planning")

    def test_default_mode_still_rejects_conflicting_same_id(self):
        with _tempdir() as root:
            lb = root / "lb"
            lb.mkdir(parents=True, exist_ok=True)
            (lb / "alldata.tsv").write_text(
                "ID\tInstructor\tTitle\tTopic\tURL\tVenue\tYear\n"
                "137\tAda\tStatistical Part of Speech Tagging\t1\thttp://example.test/pos.pdf\tYale\t2016\n",
                encoding="utf-8",
            )
            catalog = load_lecturebank_catalog(lb)
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [_jsonl_record("137", title="Planning", url="http://example.test/planning.pdf")],
            )
            lookup = jsonl_lookup_for_selected(jsonl, ["137"], catalog, authoritative=False)
            self.assertNotEqual(lookup["137"].status, "usable")
            lookup_auth = jsonl_lookup_for_selected(jsonl, ["137"], catalog, authoritative=True)
            self.assertEqual(lookup_auth["137"].status, "usable")
            self.assertEqual(lookup_auth["137"].record["metadata"]["title"], "Planning")

    def test_exact_id_does_not_match_prefix(self):
        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [_jsonl_record("1370"), _jsonl_record("137")],
            )
            found = load_selected_extractions(jsonl, ["137"])
            self.assertEqual(found["137"].status, "usable")
            self.assertEqual(found["137"].record["id"], "137")
            self.assertNotIn("1370", found)

    def test_missing_id_fails_cleanly(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            jsonl = _write_jsonl(root / "ext.jsonl", [_jsonl_record("141")])
            result = _run_eval(
                root=root,
                selected=["137"],
                jsonl_path=jsonl,
                args=_args(jsonl_authoritative=True, temporary_vector_stores=True),
                catalog=catalog,
            )
            rec = result["collected"][0]
            self.assertEqual(rec.get("eval_status"), "failed")
            self.assertIn("not present", rec.get("error_message") or rec.get("error") or "")

    def test_duplicate_id_is_rejected(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [_jsonl_record("137", title="Planning"), _jsonl_record("137", title="Planning")],
            )
            result = _run_eval(
                root=root,
                selected=["137"],
                jsonl_path=jsonl,
                args=_args(jsonl_authoritative=True, temporary_vector_stores=True),
                catalog=catalog,
            )
            rec = result["collected"][0]
            self.assertEqual(rec.get("eval_status"), "failed")
            self.assertIn("duplicate", (rec.get("error_message") or rec.get("error") or "").lower())

    def test_empty_pages_are_rejected(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb"))
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [_jsonl_record("137", title="Planning", empty=True)],
            )
            result = _run_eval(
                root=root,
                selected=["137"],
                jsonl_path=jsonl,
                args=_args(jsonl_authoritative=True, temporary_vector_stores=True),
                catalog=catalog,
            )
            rec = result["collected"][0]
            self.assertEqual(rec.get("eval_status"), "failed")
            msg = (rec.get("error_message") or rec.get("error") or "").lower()
            self.assertTrue("no usable" in msg or "empty" in msg)

    def test_preflight_authoritative_has_no_rag_and_shows_jsonl_title(self):
        from mas_langgraph.evaluation.runner import run_eval

        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record("137", title="Planning"),
                    _jsonl_record("138", title="HMMs"),
                    _jsonl_record("139", title="Empty", empty=True),
                ],
            )
            log = io.StringIO()
            args = Namespace(
                lecturebank_dir=None,
                output_dir=str(root / "out"),
                pdf_dir=None,
                extraction_jsonl=str(jsonl),
                jsonl_authoritative=True,
                first_jsonl=None,
                id_values=["137", "138", "139"],
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
                preflight=True,
            )
            with patch("mas_langgraph.evaluation.execute.prepare_presentation") as prep, patch(
                "mas_langgraph.evaluation.execute.run_factual_and_explanation_in_memory"
            ) as gen:
                code = run_eval(args, pipeline=object(), embeddings=object(), stdout=log)
            prep.assert_not_called()
            gen.assert_not_called()
            out = log.getvalue()
            self.assertEqual(code, 1)
            self.assertIn("137\tavailable\textraction_jsonl\tPlanning", out)
            self.assertIn("138\tavailable\textraction_jsonl\tHMMs", out)
            self.assertIn("139\tunavailable\textraction_jsonl\tno usable text", out)
            self.assertIn("Selected unique JSONL IDs: 3", out)

    def test_first_jsonl_selects_usable_in_file_order(self):
        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record("1370"),
                    _jsonl_record("10", empty=True),
                    _jsonl_record("137"),
                    _jsonl_record("137"),
                    _jsonl_record("14"),
                    _jsonl_record("141"),
                ],
            )
            ids = select_first_usable_jsonl_ids(jsonl, 3)
            self.assertEqual(ids, ["1370", "137", "14"])
            self.assertNotIn("10", ids)
            self.assertEqual(ids.count("137"), 1)

    def test_first_jsonl_via_run_eval_preflight(self):
        from mas_langgraph.evaluation.runner import run_eval

        with _tempdir() as root:
            jsonl = _write_jsonl(
                root / "ext.jsonl",
                [
                    _jsonl_record("a", title="One"),
                    _jsonl_record("b", title="Two", empty=True),
                    _jsonl_record("c", title="Three"),
                ],
            )
            log = io.StringIO()
            args = Namespace(
                lecturebank_dir=None,
                output_dir=str(root / "out"),
                pdf_dir=None,
                extraction_jsonl=str(jsonl),
                jsonl_authoritative=True,
                first_jsonl=2,
                id_values=[],
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
                preflight=True,
            )
            code = run_eval(args, stdout=log)
            out = log.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("a\tavailable\textraction_jsonl\tOne", out)
            self.assertIn("c\tavailable\textraction_jsonl\tThree", out)
            self.assertNotIn("b\tavailable", out)
            self.assertIn("Selected unique JSONL IDs: 2", out)


if __name__ == "__main__":
    unittest.main()
