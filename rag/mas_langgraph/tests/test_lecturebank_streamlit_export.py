"""Offline tests for optional Streamlit experiment JSON export (no provider calls)."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional
from unittest.mock import patch

from mas_langgraph.batch.generate import InMemoryGenerationResult
from mas_langgraph.batch.ids import file_sha256
from mas_langgraph.batch.manifest import generation_config_fingerprint
from mas_langgraph.evaluation.execute import process_documents
from mas_langgraph.evaluation.lecturebank_meta import load_lecturebank_catalog
from mas_langgraph.evaluation.lifecycle import load_collected_from_metrics
from mas_langgraph.evaluation.runner import (
    _failed_document,
    _find_manifest_record,
    _persist,
    _upsert_manifest,
    build_arg_parser,
)
from mas_langgraph.evaluation.streamlit_export import (
    UNAVAILABLE_RESUME_MESSAGE,
    streamlit_experiment_filename,
    streamlit_json_path,
    write_streamlit_experiment_json,
)
from mas_langgraph.tests.test_explanation_graph import make_exported_case
from mas_langgraph.visualization import build_explanation_graph
from mas_langgraph.visualization.ui_helpers import (
    case_visualization_issue,
    parse_experiment_json_text,
    validate_experiment_payload,
)


@contextmanager
def _tempdir() -> Iterator[Path]:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        yield Path(tmp.name)
    finally:
        tmp.cleanup()


def _write_catalog(root: Path, ids: Optional[List[str]] = None) -> Path:
    ids = ids or ["141"]
    root.mkdir(parents=True, exist_ok=True)
    lines = ["ID\tInstructor\tTitle\tTopic\tURL\tVenue\tYear"]
    meta = {
        "141": ("Grace", "Deep Learning", "2", "http://example.test/141.pdf", "Yale", "2018"),
        "142": ("Alan", "Survey", "1", "http://example.test/142.pdf", "MIT", "2017"),
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


def _streamlit_payload(*, mode: str = "hybrid", n_cases: int = 2, run_id: str = "run_viz") -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for i in range(n_cases):
        case = copy.deepcopy(make_exported_case())
        if i:
            ref = {"logical_id": f"q_{i:08d}", "version": 1, "key": f"q_{i:08d}:v1"}
            case["candidate_ref"] = ref
            case["visualization_data"]["target"]["candidate_ref"] = ref
        cases.append(case)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "branch_id": "factual",
        "experiment_seed": 42,
        "user_question": "Why was this question selected?",
        "question_count": n_cases,
        "models": {"workflow": {"model": "fake"}},
        "evidence_access": {"mode": "researcher"},
        "warnings": [],
        "axis_interrogation_mode": mode,
        "cases": cases,
    }


def _round_trip(payload: Mapping[str, Any]) -> Any:
    return json.loads(json.dumps(payload, default=str))


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
        export_quiz=False,
        export_streamlit_json=False,
        axis_interrogation_mode="constrained",
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


def _gen_fp(mode: str = "constrained") -> Dict[str, Any]:
    return {
        "llm_model": "fake",
        "model_type": "openai",
        "temperature": 0.0,
        "seed": 42,
        "max_rounds": 5,
        "max_questions": 2,
        "max_explain_questions": 2,
        "question": "Why?",
        "orchestrator": "langgraph",
        "axis_interrogation_mode": mode,
    }


def _fake_prepare():
    def prepare_presentation(**kwargs: Any) -> Dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        document_id = kwargs["document_id"]
        vs = output_dir / f"presentation_{document_id}" / "vector_store"
        vs.mkdir(parents=True, exist_ok=True)
        (vs / "chroma.sqlite3").write_text("fake-index", encoding="utf-8")
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


def _fake_generate(
    *,
    payload: Optional[Dict[str, Any]] = None,
    gens: Optional[List[InMemoryGenerationResult]] = None,
    fail: bool = False,
):
    def run_factual_and_explanation_in_memory(**kwargs: Any) -> InMemoryGenerationResult:
        rec = dict(kwargs.get("record") or {})
        doc_id = str(rec.get("document_id") or "")
        lb_id = doc_id.replace("lecturebank_", "")
        if fail:
            gen = InMemoryGenerationResult(
                record={
                    **rec,
                    "generate_status": "explanation_failed",
                    "errors": ["forced explanation failure"],
                    "run_id": f"run-{lb_id}",
                },
                experiment=None,
                experiment_payload=None,
                accepted_count=0,
            )
        else:
            chosen = payload or _streamlit_payload()
            gen = InMemoryGenerationResult(
                record={**rec, "generate_status": "success", "run_id": f"run-{lb_id}"},
                experiment=object(),
                experiment_payload=chosen,
                checkpoint_index={"records": [{"alias": f"c_{i}"} for i in range(10)]},
                accepted_count=2,
            )
        if gens is not None:
            gens.append(gen)
        return gen

    return run_factual_and_explanation_in_memory


def _run(
    *,
    root: Path,
    selected: List[str],
    args: Namespace,
    catalog,
    persist=None,
    collected: Optional[List[Dict[str, Any]]] = None,
    manifest: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    gens: Optional[List[InMemoryGenerationResult]] = None,
    fail: bool = False,
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
    used_gen_fp = _gen_fp(str(getattr(args, "axis_interrogation_mode", "constrained")))

    def tracked_persist(*pargs: Any, **pkwargs: Any) -> None:
        persist_fn(*pargs, **pkwargs)

    with patch(
        "mas_langgraph.evaluation.execute.prepare_presentation",
        side_effect=_fake_prepare(),
    ), patch(
        "mas_langgraph.evaluation.execute.run_factual_and_explanation_in_memory",
        side_effect=_fake_generate(payload=payload, gens=gens, fail=fail),
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
            gen_cfg=used_gen_fp,
            rag_fp=rag_fp,
            gen_fp=used_gen_fp,
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
        "collected": collected,
        "manifest": manifest,
    }


def _persist_streamlit(*pargs: Any, **pkwargs: Any) -> None:
    _persist(*pargs, **pkwargs, export_streamlit_json=True)


class CliStreamlitExportTests(unittest.TestCase):
    def test_export_streamlit_json_defaults_false_and_cli_accepts_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--lecturebank-dir", "x", "--id", "141", "--allow-any-count"])
        self.assertFalse(args.export_streamlit_json)
        args_on = parser.parse_args(
            [
                "--lecturebank-dir",
                "x",
                "--id",
                "141",
                "--allow-any-count",
                "--export-streamlit-json",
            ]
        )
        self.assertTrue(args_on.export_streamlit_json)
        self.assertFalse(args_on.export_quiz)


class DirectPayloadExportTests(unittest.TestCase):
    def test_round_trip_preserves_schema_viz_refs_conditions_and_graph(self):
        payload = _streamlit_payload(mode="hybrid", n_cases=2)
        with _tempdir() as root:
            rel = write_streamlit_experiment_json(root, "141", payload)
            self.assertEqual(rel, "streamlit/quiz_explanation_experiment_141.json")
            path = streamlit_json_path(root, "141")
            self.assertEqual(path.name, "quiz_explanation_experiment_141.json")
            text = path.read_text(encoding="utf-8")
            loaded = parse_experiment_json_text(text)
            accepted = validate_experiment_payload(loaded)
            self.assertEqual(accepted, _round_trip(payload))
            self.assertEqual(loaded["schema_version"], 1)
            self.assertEqual(loaded["axis_interrogation_mode"], "hybrid")
            self.assertEqual(len(loaded["cases"]), 2)
            case = loaded["cases"][0]
            self.assertEqual(case["candidate_ref"], payload["cases"][0]["candidate_ref"])
            self.assertEqual(case["visualization_data"], payload["cases"][0]["visualization_data"])
            self.assertEqual(case["visualization_data"]["schema_version"], 1)
            self.assertEqual(case["conditions"], payload["cases"][0]["conditions"])
            self.assertIsNone(case_visualization_issue(case))
            graph = build_explanation_graph(case, condition="axis", detail_level="essential")
            self.assertTrue(graph.node_ids())

    def test_constrained_mode_field_is_preserved(self):
        payload = _streamlit_payload(mode="constrained", n_cases=1)
        with _tempdir() as root:
            write_streamlit_experiment_json(root, "141", payload)
            loaded = parse_experiment_json_text(
                streamlit_json_path(root, "141").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded["axis_interrogation_mode"], "constrained")
            graph = build_explanation_graph(
                loaded["cases"][0], condition="axis", detail_level="essential"
            )
            self.assertTrue(graph.node_ids())


class EvalWorkflowStreamlitExportTests(unittest.TestCase):
    def test_disabled_export_does_not_write_json(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            result = _run(
                root=root,
                selected=["141"],
                args=_args(export_streamlit_json=False),
                catalog=catalog,
                payload=_streamlit_payload(),
            )
            self.assertFalse(streamlit_json_path(result["output_dir"], "141").exists())
            self.assertFalse((result["output_dir"] / "streamlit").exists())
            report = (result["metrics_dir"] / "report.md").read_text(encoding="utf-8")
            self.assertNotIn("Streamlit experiment JSON:", report)

    def test_enabled_export_writes_named_file_and_clears_payload(self):
        gens: List[InMemoryGenerationResult] = []
        payload = _streamlit_payload(mode="hybrid", n_cases=2)
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            result = _run(
                root=root,
                selected=["141"],
                args=_args(export_streamlit_json=True, axis_interrogation_mode="hybrid"),
                catalog=catalog,
                persist=_persist_streamlit,
                payload=payload,
                gens=gens,
            )
            path = streamlit_json_path(result["output_dir"], "141")
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, streamlit_experiment_filename("141"))
            self.assertEqual(
                list((result["output_dir"] / "streamlit").glob("*.json")),
                [path],
            )
            loaded = parse_experiment_json_text(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, _round_trip(payload))
            self.assertEqual(len(loaded["cases"]), 2)
            self.assertEqual(loaded["axis_interrogation_mode"], "hybrid")
            case = loaded["cases"][0]
            self.assertIsNone(case_visualization_issue(case))
            graph = build_explanation_graph(case, condition="axis", detail_level="essential")
            self.assertTrue(graph.node_ids())
            rec = _find_manifest_record(result["manifest"], "141")
            self.assertEqual(rec.get("streamlit_json_path"), "streamlit/quiz_explanation_experiment_141.json")
            report = (result["metrics_dir"] / "report.md").read_text(encoding="utf-8")
            self.assertIn("Streamlit experiment JSON:", report)
            self.assertTrue(gens)
            self.assertIsNone(gens[0].experiment_payload)

    def test_constrained_workflow_preserves_mode(self):
        payload = _streamlit_payload(mode="constrained", n_cases=1)
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            result = _run(
                root=root,
                selected=["141"],
                args=_args(export_streamlit_json=True, axis_interrogation_mode="constrained"),
                catalog=catalog,
                persist=_persist_streamlit,
                payload=payload,
            )
            loaded = parse_experiment_json_text(
                streamlit_json_path(result["output_dir"], "141").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded["axis_interrogation_mode"], "constrained")

    def test_missing_payload_does_not_create_file(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            result = _run(
                root=root,
                selected=["141"],
                args=_args(export_streamlit_json=True),
                catalog=catalog,
                persist=_persist_streamlit,
                fail=True,
            )
            self.assertFalse(streamlit_json_path(result["output_dir"], "141").exists())

    def test_resume_does_not_create_duplicates(self):
        payload = _streamlit_payload()
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            first = _run(
                root=root,
                selected=["141"],
                args=_args(export_streamlit_json=True, axis_interrogation_mode="hybrid"),
                catalog=catalog,
                persist=_persist_streamlit,
                payload=payload,
            )
            collected = load_collected_from_metrics(first["metrics_dir"])
            second = _run(
                root=root,
                selected=["141"],
                args=_args(
                    export_streamlit_json=True,
                    axis_interrogation_mode="hybrid",
                    resume=True,
                ),
                catalog=catalog,
                persist=_persist_streamlit,
                collected=collected,
                manifest=first["manifest"],
                payload=_streamlit_payload(run_id="should-not-be-written"),
            )
            self.assertEqual(second["stats"]["skipped_resume"], 1)
            files = list((second["output_dir"] / "streamlit").glob("*.json"))
            self.assertEqual(len(files), 1)
            loaded = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(loaded["run_id"], "run_viz")

    def test_overwrite_atomically_replaces_same_filename(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            first = _run(
                root=root,
                selected=["141"],
                args=_args(export_streamlit_json=True),
                catalog=catalog,
                persist=_persist_streamlit,
                payload=_streamlit_payload(run_id="run_old"),
            )
            path = streamlit_json_path(first["output_dir"], "141")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["run_id"], "run_old")
            second = _run(
                root=root,
                selected=["141"],
                args=_args(export_streamlit_json=True, overwrite=True),
                catalog=catalog,
                persist=_persist_streamlit,
                collected=list(first["collected"]),
                manifest=first["manifest"],
                payload=_streamlit_payload(run_id="run_new"),
            )
            files = list((second["output_dir"] / "streamlit").glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "quiz_explanation_experiment_141.json")
            loaded = parse_experiment_json_text(files[0].read_text(encoding="utf-8"))
            self.assertEqual(loaded["run_id"], "run_new")
            graph = build_explanation_graph(
                loaded["cases"][0], condition="axis", detail_level="essential"
            )
            self.assertTrue(graph.node_ids())

    def test_old_completed_run_is_not_reconstructed_from_metrics(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            first = _run(
                root=root,
                selected=["141"],
                args=_args(export_streamlit_json=False),
                catalog=catalog,
                payload=_streamlit_payload(),
            )
            self.assertFalse(streamlit_json_path(first["output_dir"], "141").exists())
            collected = load_collected_from_metrics(first["metrics_dir"])
            second = _run(
                root=root,
                selected=["141"],
                args=_args(export_streamlit_json=True, resume=True),
                catalog=catalog,
                persist=_persist_streamlit,
                collected=collected,
                manifest=first["manifest"],
                payload=_streamlit_payload(run_id="invented"),
            )
            self.assertEqual(second["stats"]["skipped_resume"], 1)
            self.assertIn(UNAVAILABLE_RESUME_MESSAGE, second["log"])
            self.assertFalse(streamlit_json_path(second["output_dir"], "141").exists())
            report = (second["metrics_dir"] / "report.md").read_text(encoding="utf-8")
            self.assertIn("rerun with `--overwrite`", report)

    def test_export_flag_does_not_change_generation_config_fingerprint(self):
        base = {
            "llm_model": "fake",
            "model_type": "openai",
            "temperature": 0.0,
            "seed": 42,
            "max_rounds": 5,
            "max_questions": 2,
            "max_explain_questions": 2,
            "orchestrator": "langgraph",
            "question": "Why?",
            "axis_interrogation_mode": "hybrid",
        }
        off = generation_config_fingerprint(base)
        on = generation_config_fingerprint(
            {**base, "export_streamlit_json": True, "export_quiz": True}
        )
        self.assertEqual(off, on)
        self.assertNotIn("export_streamlit_json", on)
        self.assertNotIn("export_quiz", on)


if __name__ == "__main__":
    unittest.main()
