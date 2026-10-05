"""Offline tests for optional LectureBank quiz export (no provider calls)."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch

from mas_langgraph.batch.generate import InMemoryGenerationResult
from mas_langgraph.batch.ids import file_sha256
from mas_langgraph.batch.manifest import generation_config_fingerprint
from mas_langgraph.evaluation.execute import process_documents
from mas_langgraph.evaluation.lecturebank_meta import load_lecturebank_catalog
from mas_langgraph.evaluation.lifecycle import load_collected_from_metrics
from mas_langgraph.evaluation.quiz_export import (
    extract_quiz_questions,
    normalize_correct_option,
    render_quizzes_markdown,
    write_quiz_outputs,
)
from mas_langgraph.evaluation.runner import (
    _failed_document,
    _find_manifest_record,
    _persist,
    _upsert_manifest,
    build_arg_parser,
)


@contextmanager
def _tempdir() -> Iterator[Path]:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        yield Path(tmp.name)
    finally:
        tmp.cleanup()


def _write_catalog(root: Path, ids: Optional[List[str]] = None) -> Path:
    ids = ids or ["141", "142"]
    root.mkdir(parents=True, exist_ok=True)
    lines = ["ID\tInstructor\tTitle\tTopic\tURL\tVenue\tYear"]
    meta = {
        "141": ("Grace", "Deep Learning", "2", "http://example.test/141.pdf", "Yale", "2018"),
        "142": ("Alan", "Survey", "1", "http://example.test/142.pdf", "MIT", "2017"),
        "14": ("Ada", "Intro ML", "1", "http://example.test/14.pdf", "Yale", "2016"),
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


def _option(oid: str, text: str, correct: bool = False) -> Dict[str, Any]:
    return {"option_id": oid, "text": text, "is_correct": correct}


def _case(
    *,
    key: str,
    logical_id: str,
    version: int,
    stem: str,
    options: List[Dict[str, Any]],
    correct_option: Any,
    factual_decision: str = "accept",
) -> Dict[str, Any]:
    return {
        "candidate_ref": {"key": key, "logical_id": logical_id, "version": version},
        "factual_decision": factual_decision,
        "question": {
            "stem": stem,
            "options": options,
            "correct_option": correct_option,
        },
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
                    "replay_comparisons": [_cmp()],
                    "provenance_relations": [],
                    "relevant_artifacts": [],
                    "termination_reason": "max_rounds",
                },
            }
        },
        "counterfactual_count": 1,
        "successful_replay_count": 1,
    }


def _quiz_payload(
    *,
    mode: str = "hybrid",
    extra_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    cases = [
        _case(
            key="CandidateMCQ:q_00000001:v2",
            logical_id="q_00000001",
            version=2,
            stem="What is backpropagation?",
            options=[
                _option("A", "A search heuristic"),
                _option("B", "Gradient-based training", True),
                _option("C", "A clustering method"),
                _option("D", "An activation function"),
            ],
            correct_option="B",
        ),
        _case(
            key="CandidateMCQ:q_00000002:v1",
            logical_id="q_00000002",
            version=1,
            stem='Which loss, if "quoted", uses commas?',
            options=[
                _option("A", "Mean squared error, L2"),
                _option("B", "Hinge\nloss"),
                _option("C", "Perplexité"),
                _option("D", "Accuracy"),
            ],
            correct_option="A",
        ),
    ]
    if extra_cases:
        cases.extend(extra_cases)
    return {
        "cases": cases,
        "axis_interrogation_mode": mode,
        "workflow_llm_diagnostics": {},
        "timing": {},
        "question_count": 2,
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
        export_quiz=False,
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
    by_id: Optional[Dict[str, Dict[str, Any]]] = None,
):
    def run_factual_and_explanation_in_memory(**kwargs: Any) -> InMemoryGenerationResult:
        rec = dict(kwargs.get("record") or {})
        doc_id = str(rec.get("document_id") or "")
        lb_id = doc_id.replace("lecturebank_", "")
        chosen = (by_id or {}).get(lb_id) or payload or _quiz_payload()
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
    by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    gens: Optional[List[InMemoryGenerationResult]] = None,
    gen_fp: Optional[Dict[str, Any]] = None,
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
    used_gen_fp = gen_fp or _gen_fp(str(getattr(args, "axis_interrogation_mode", "constrained")))

    def tracked_persist(*pargs: Any, **pkwargs: Any) -> None:
        persist_fn(*pargs, **pkwargs)

    with patch(
        "mas_langgraph.evaluation.execute.prepare_presentation",
        side_effect=_fake_prepare(),
    ), patch(
        "mas_langgraph.evaluation.execute.run_factual_and_explanation_in_memory",
        side_effect=_fake_generate(payload=payload, gens=gens, by_id=by_id),
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


def _read_questions(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _persist_quiz(*pargs: Any, **pkwargs: Any) -> None:
    _persist(*pargs, **pkwargs, export_quiz=True)


class CliExportQuizTests(unittest.TestCase):
    def test_export_quiz_defaults_false_and_cli_accepts_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--lecturebank-dir", "x", "--id", "141", "--allow-any-count"])
        self.assertFalse(args.export_quiz)
        args_on = parser.parse_args(
            ["--lecturebank-dir", "x", "--id", "141", "--allow-any-count", "--export-quiz"]
        )
        self.assertTrue(args_on.export_quiz)


class CorrectOptionMappingTests(unittest.TestCase):
    def test_letter_index_and_is_correct_map_to_display_label(self):
        options = [
            _option("A", "one"),
            _option("B", "two", True),
            _option("C", "three"),
            _option("D", "four"),
        ]
        self.assertEqual(normalize_correct_option(options, "B"), "B")
        self.assertEqual(normalize_correct_option(options, "b"), "B")
        unlabeled = [
            {"option_id": "", "text": "one", "is_correct": False},
            {"option_id": "", "text": "two", "is_correct": False},
            {"option_id": "", "text": "three", "is_correct": False},
            {"option_id": "", "text": "four", "is_correct": False},
        ]
        self.assertEqual(normalize_correct_option(unlabeled, 0), "A")
        self.assertEqual(normalize_correct_option(unlabeled, 1), "B")
        self.assertEqual(normalize_correct_option(unlabeled, 2), "C")
        self.assertEqual(normalize_correct_option(unlabeled, 3), "D")
        self.assertEqual(normalize_correct_option(unlabeled, "2"), "C")
        marked = [
            _option("A", "one"),
            _option("B", "two"),
            _option("C", "three", True),
            _option("D", "four"),
        ]
        self.assertEqual(normalize_correct_option(marked, None), "C")


class ExtractFinalQuizTests(unittest.TestCase):
    def test_rejected_and_superseded_versions_are_excluded(self):
        payload = _quiz_payload(
            extra_cases=[
                _case(
                    key="CandidateMCQ:q_rejected:v1",
                    logical_id="q_rejected",
                    version=1,
                    stem="Rejected stem?",
                    options=[
                        _option("A", "x", True),
                        _option("B", "y"),
                        _option("C", "z"),
                        _option("D", "w"),
                    ],
                    correct_option="A",
                    factual_decision="reject",
                ),
                _case(
                    key="CandidateMCQ:q_00000001:v1",
                    logical_id="q_00000001",
                    version=1,
                    stem="Old superseded stem?",
                    options=[
                        _option("A", "old A", True),
                        _option("B", "old B"),
                        _option("C", "old C"),
                        _option("D", "old D"),
                    ],
                    correct_option="A",
                ),
            ]
        )
        rows = extract_quiz_questions(
            payload, lecturebank_id="141", axis_interrogation_mode="hybrid"
        )
        self.assertEqual(len(rows), 2)
        refs = [row["candidate_ref"] for row in rows]
        self.assertIn("CandidateMCQ:q_00000001:v2", refs)
        self.assertIn("CandidateMCQ:q_00000002:v1", refs)
        self.assertNotIn("CandidateMCQ:q_rejected:v1", refs)
        self.assertNotIn("CandidateMCQ:q_00000001:v1", refs)
        revised = next(row for row in rows if row["candidate_ref"].endswith(":v2"))
        self.assertEqual(revised["stem"], "What is backpropagation?")
        self.assertEqual(revised["correct_option"], "B")
        self.assertEqual(revised["question_version"], 2)

    def test_max_questions_two_yields_at_most_two_final_rows(self):
        payload = _quiz_payload()
        self.assertEqual(len(payload["cases"]), 2)
        rows = extract_quiz_questions(
            payload, lecturebank_id="141", axis_interrogation_mode="constrained"
        )
        self.assertLessEqual(len(rows), 2)
        self.assertEqual(len(rows), 2)

    def test_hybrid_and_constrained_mode_are_recorded(self):
        hybrid = extract_quiz_questions(
            _quiz_payload(mode="hybrid"),
            lecturebank_id="141",
            axis_interrogation_mode="hybrid",
        )
        constrained = extract_quiz_questions(
            _quiz_payload(mode="constrained"),
            lecturebank_id="141",
            axis_interrogation_mode="constrained",
        )
        self.assertEqual({row["axis_interrogation_mode"] for row in hybrid}, {"hybrid"})
        self.assertEqual({row["axis_interrogation_mode"] for row in constrained}, {"constrained"})


class QuizMarkdownAndCsvTests(unittest.TestCase):
    def test_csv_round_trip_preserves_commas_quotes_newlines_and_unicode(self):
        rows = extract_quiz_questions(
            _quiz_payload(), lecturebank_id="141", axis_interrogation_mode="hybrid"
        )
        with _tempdir() as root:
            metrics = root / "metrics"
            write_quiz_outputs(root, metrics, rows)
            loaded = _read_questions(metrics / "questions.csv")
        self.assertEqual(len(loaded), 2)
        stem = loaded[1]["stem"]
        self.assertIn(",", loaded[1]["option_A"])
        self.assertIn('"', stem)
        self.assertIn("\n", loaded[1]["option_B"])
        self.assertIn("é", loaded[1]["option_C"])
        self.assertEqual(loaded[0]["candidate_ref"], "CandidateMCQ:q_00000001:v2")

    def test_two_documents_are_grouped_in_quizzes_md(self):
        rows = extract_quiz_questions(
            _quiz_payload(), lecturebank_id="141", axis_interrogation_mode="hybrid"
        ) + extract_quiz_questions(
            _quiz_payload(), lecturebank_id="142", axis_interrogation_mode="hybrid"
        )
        text = render_quizzes_markdown(rows)
        self.assertIn("## Document 141", text)
        self.assertIn("## Document 142", text)
        self.assertLess(text.index("## Document 141"), text.index("## Document 142"))
        self.assertIn("Candidate ref: CandidateMCQ:q_00000001:v2", text)
        self.assertIn("Correct answer: B", text)


class FingerprintAndClearPayloadTests(unittest.TestCase):
    def test_export_quiz_does_not_change_generation_config_fingerprint(self):
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
        on = generation_config_fingerprint({**base, "export_quiz": True})
        self.assertEqual(off, on)
        self.assertNotIn("export_quiz", on)
        self.assertEqual(on["axis_interrogation_mode"], "hybrid")

    def test_experiment_payload_can_be_cleared_after_export(self):
        payload = _quiz_payload()
        rows = extract_quiz_questions(
            payload, lecturebank_id="141", axis_interrogation_mode="hybrid"
        )
        payload = None
        self.assertIsNone(payload)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["candidate_ref"], "CandidateMCQ:q_00000001:v2")


class EvalWorkflowExportTests(unittest.TestCase):
    def test_disabled_export_does_not_write_quiz_files(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            result = _run(
                root=root,
                selected=["141"],
                args=_args(export_quiz=False),
                catalog=catalog,
            )
            self.assertFalse((result["metrics_dir"] / "questions.csv").exists())
            self.assertFalse((result["output_dir"] / "quizzes.md").exists())
            self.assertFalse((result["output_dir"] / "quizzes").exists())
            report = (result["metrics_dir"] / "report.md").read_text(encoding="utf-8")
            self.assertNotIn("Quiz export:", report)

    def test_enabled_export_writes_csv_markdown_and_report_line(self):
        gens: List[InMemoryGenerationResult] = []
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            result = _run(
                root=root,
                selected=["141"],
                args=_args(export_quiz=True, axis_interrogation_mode="hybrid"),
                catalog=catalog,
                persist=_persist_quiz,
                gens=gens,
            )
            questions = result["metrics_dir"] / "questions.csv"
            quizzes = result["output_dir"] / "quizzes.md"
            per_doc = result["output_dir"] / "quizzes" / "141.md"
            self.assertTrue(questions.is_file())
            self.assertTrue(quizzes.is_file())
            self.assertTrue(per_doc.is_file())
            rows = _read_questions(questions)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["candidate_ref"], "CandidateMCQ:q_00000001:v2")
            self.assertEqual(rows[0]["axis_interrogation_mode"], "hybrid")
            self.assertEqual([row["document_id"] for row in rows], ["lecturebank_141"] * 2)
            md = quizzes.read_text(encoding="utf-8")
            self.assertIn("# LectureBank Quiz Export", md)
            self.assertIn("## Document 141", md)
            self.assertIn("AXIS interrogation mode: hybrid", md)
            report = (result["metrics_dir"] / "report.md").read_text(encoding="utf-8")
            self.assertIn("Quiz export:", report)
            self.assertIn("questions.csv: 2 questions", report)
            self.assertIn("quizzes.md: generated", report)
            self.assertTrue(gens)
            self.assertIsNone(gens[0].experiment_payload)

    def test_constrained_mode_recorded_in_workflow_export(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            result = _run(
                root=root,
                selected=["141"],
                args=_args(export_quiz=True, axis_interrogation_mode="constrained"),
                catalog=catalog,
                persist=_persist_quiz,
                payload=_quiz_payload(mode="constrained"),
            )
            rows = _read_questions(result["metrics_dir"] / "questions.csv")
            self.assertEqual({row["axis_interrogation_mode"] for row in rows}, {"constrained"})

    def test_resume_does_not_duplicate_quiz_rows(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            first = _run(
                root=root,
                selected=["141"],
                args=_args(export_quiz=True, axis_interrogation_mode="hybrid"),
                catalog=catalog,
                persist=_persist_quiz,
            )
            collected = load_collected_from_metrics(first["metrics_dir"])
            self.assertEqual(len(collected[0].get("quiz_questions") or []), 2)
            second = _run(
                root=root,
                selected=["141"],
                args=_args(export_quiz=True, axis_interrogation_mode="hybrid", resume=True),
                catalog=catalog,
                persist=_persist_quiz,
                collected=collected,
                manifest=first["manifest"],
            )
            self.assertEqual(second["stats"]["skipped_resume"], 1)
            rows = _read_questions(second["metrics_dir"] / "questions.csv")
            self.assertEqual(len(rows), 2)
            refs = [row["candidate_ref"] for row in rows]
            self.assertEqual(len(refs), len(set(refs)))

    def test_export_after_old_completed_run_does_not_invent_text(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141"]))
            first = _run(
                root=root,
                selected=["141"],
                args=_args(export_quiz=False),
                catalog=catalog,
            )
            self.assertFalse((first["metrics_dir"] / "questions.csv").exists())
            collected = load_collected_from_metrics(first["metrics_dir"])
            second = _run(
                root=root,
                selected=["141"],
                args=_args(export_quiz=True, resume=True),
                catalog=catalog,
                persist=_persist_quiz,
                collected=collected,
                manifest=first["manifest"],
            )
            self.assertEqual(second["stats"]["skipped_resume"], 1)
            self.assertIn("quiz export unavailable", second["log"])
            rows = _read_questions(second["metrics_dir"] / "questions.csv")
            self.assertEqual(rows, [])
            md = (second["output_dir"] / "quizzes.md").read_text(encoding="utf-8")
            self.assertIn("was not persisted", md)
            self.assertNotIn("What is backpropagation?", md)
            report = (second["metrics_dir"] / "report.md").read_text(encoding="utf-8")
            self.assertIn("unavailable for previously completed documents", report)

    def test_two_documents_export_and_per_document_files(self):
        with _tempdir() as root:
            catalog = load_lecturebank_catalog(_write_catalog(root / "lb", ["141", "142"]))
            result = _run(
                root=root,
                selected=["141", "142"],
                args=_args(export_quiz=True, axis_interrogation_mode="hybrid"),
                catalog=catalog,
                persist=_persist_quiz,
            )
            rows = _read_questions(result["metrics_dir"] / "questions.csv")
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["lecturebank_id"] for row in rows}, {"141", "142"})
            md = (result["output_dir"] / "quizzes.md").read_text(encoding="utf-8")
            self.assertIn("## Document 141", md)
            self.assertIn("## Document 142", md)
            self.assertTrue((result["output_dir"] / "quizzes" / "141.md").is_file())
            self.assertTrue((result["output_dir"] / "quizzes" / "142.md").is_file())
            report = (result["metrics_dir"] / "report.md").read_text(encoding="utf-8")
            self.assertIn("questions.csv: 4 questions", report)


if __name__ == "__main__":
    unittest.main()
