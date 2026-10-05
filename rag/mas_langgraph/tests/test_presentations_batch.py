"""Offline tests for folder-based presentation batch processing."""

from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List
from unittest.mock import patch

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.batch.discover import discover_presentations
from mas_langgraph.batch.generate import validate_prepared_record
from mas_langgraph.batch.ids import collection_safe_id
from mas_langgraph.batch.manifest import load_manifest
from mas_langgraph.batch.prepare import prepare_presentation
from mas_langgraph.batch.runner import build_arg_parser, run_batch
from mas_langgraph.experiment.models import (
    AnswerOptionView,
    ExplanationConditionResult,
    MCQExplanationComparison,
    QuizExplanationExperiment,
)
from mas_langgraph.input.rag_config import RAGConfig


@contextmanager
def _tempdir() -> Iterator[str]:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        yield tmp.name
    finally:
        tmp.cleanup()


class FakeEmbeddings:
    """Deterministic embeddings; no network and no MiniLM download."""

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


def _pages(text: str, n: int = 1) -> Dict[str, Any]:
    pages = [{"page": i + 1, "text": text if i == 0 else "", "length": len(text if i == 0 else "")} for i in range(n)]
    nonempty = sum(1 for p in pages if str(p["text"]).strip())
    empty = [p["page"] for p in pages if not str(p["text"]).strip()]
    return {
        "pages_text": pages,
        "nonempty_pages": nonempty,
        "empty_pages": empty,
        "text_length": len(text.strip()),
        "requires_ocr": nonempty == 0,
    }


def _fake_loader(mapping: Dict[str, Dict[str, Any]]):
    def load(*args: Any, document_id=None, pdf_path: str | None = None, **kwargs: Any):
        positional = [a for a in args if not hasattr(a, "load_pdf_from_file")]
        path_arg = pdf_path or (positional[0] if positional else None)
        if path_arg is None:
            raise TypeError("pdf_path is required")
        if document_id is None and len(positional) > 1:
            document_id = positional[1]
        path = Path(path_arg).resolve()
        spec = mapping[str(path)]
        pages = spec["pages_text"]
        doc_id = document_id or spec.get("id") or "doc"
        return {
            "id": doc_id,
            "source_file": path.name,
            "text": {
                "full_text": "\n".join(p["text"] for p in pages).strip(),
                "pages_text": pages,
            },
            "statistics": {
                "num_pages": len(pages),
                "nonempty_pages": spec["nonempty_pages"],
                "empty_pages": spec["empty_pages"],
                "text_length": spec["text_length"],
                "requires_ocr": spec["requires_ocr"],
            },
        }

    return load


def _write_dummy_pdf(path: Path, marker: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-DUMMY\n" + marker + b"\n")
    return path.resolve()


def _mini_experiment(doc_id: str, stem: str) -> QuizExplanationExperiment:
    case = MCQExplanationComparison(
        run_id=f"run_{doc_id}",
        branch_id="factual",
        candidate_ref=ArtifactRef("q1", 1),
        assessment_ref=ArtifactRef("a1", 1),
        decision_ref=ArtifactRef("d1", 1),
        question_stem=stem,
        answer_options=(
            AnswerOptionView("A", "First option", True),
            AnswerOptionView("B", "Second option", False),
        ),
        correct_option_id="A",
        factual_decision="accept",
        user_question="Why was this question selected?",
        trace=ExplanationConditionResult("trace", "Trace text."),
        trace_provenance=ExplanationConditionResult("trace_provenance", "Provenance text."),
        axis=ExplanationConditionResult("axis", "AXIS text."),
        blinding={"A": "trace", "B": "trace_provenance", "C": "axis"},
        axis_status="completed_with_counterfactual",
        axis_causal_evidence_available=True,
        successful_replay_count=1,
    )
    return QuizExplanationExperiment(
        schema_version=1,
        run_id=f"run_{doc_id}",
        branch_id="factual",
        experiment_seed=42,
        user_question="Why was this question selected?",
        cases=(case,),
    )


class FakeBackbone:
    def __init__(self, cfg: RAGConfig, profile=None):
        self.cfg = cfg
        self.profile = profile

    def run_full_workflow(self) -> Dict[str, Any]:
        return {"run_id": f"run_{self.cfg.pdf_id}"}

    def list_explanation_experiment_refs(self, include_historical_versions: bool = False):
        return (ArtifactRef("q1", 1),)

    def generate_explanation_experiment(self, **kwargs: Any) -> QuizExplanationExperiment:
        return _mini_experiment(self.cfg.pdf_id, f"Question for {self.cfg.pdf_id}?")


class EmptyQuizBackbone(FakeBackbone):
    def list_explanation_experiment_refs(self, include_historical_versions: bool = False):
        return ()


def _args(**kwargs: Any) -> argparse.Namespace:
    parser = build_arg_parser()
    base = [
        "--input-dir",
        str(kwargs.pop("input_dir")),
        "--output-dir",
        str(kwargs.pop("output_dir")),
    ]
    stage = kwargs.pop("stage", "rag")
    base.extend(["--stage", stage])
    if kwargs.pop("recursive", False):
        base.append("--recursive")
    if kwargs.pop("resume", False):
        base.append("--resume")
    if kwargs.pop("overwrite", False):
        base.append("--overwrite")
    if kwargs.pop("list_pdfs", False):
        base.append("--list-pdfs")
    for pdf in kwargs.pop("pdfs", []) or []:
        base.extend(["--pdf", str(pdf)])
    for doc_id in kwargs.pop("document_ids", []) or []:
        base.extend(["--document-id", str(doc_id)])
    if "max_questions" in kwargs:
        base.extend(["--max-questions", str(kwargs.pop("max_questions"))])
    if "max_explain_questions" in kwargs:
        base.extend(["--max-explain-questions", str(kwargs.pop("max_explain_questions"))])
    if "chunk_size" in kwargs:
        base.extend(["--chunk-size", str(kwargs.pop("chunk_size"))])
    if "chunk_overlap" in kwargs:
        base.extend(["--chunk-overlap", str(kwargs.pop("chunk_overlap"))])
    return parser.parse_args(base)


class TestStableIds(unittest.TestCase):
    def test_prefix_collision_and_nested_same_name(self):
        a = collection_safe_id("141_A.pdf")
        b = collection_safe_id("141_B.pdf")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, "141")
        nested_a = collection_safe_id("sub1/foo.pdf")
        nested_b = collection_safe_id("sub2/foo.pdf")
        self.assertNotEqual(nested_a, nested_b)
        self.assertEqual(collection_safe_id("sub1/foo.pdf"), nested_a)


class TestDiscoverUnsupported(unittest.TestCase):
    def test_pptx_is_reported_not_treated_as_pdf(self):
        with _tempdir() as td:
            root = Path(td)
            _write_dummy_pdf(root / "ok.pdf", b"ok")
            (root / "slides.pptx").write_bytes(b"not a pdf")
            found = discover_presentations(root)
            self.assertEqual(len(found.pdfs), 1)
            self.assertEqual(len(found.unsupported), 1)
            self.assertIn("PowerPoint", found.unsupported[0][1])


class TestRagIsolation(unittest.TestCase):
    def test_two_presentations_cannot_retrieve_each_other(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            p1 = _write_dummy_pdf(inp / "alpha.pdf", b"alpha")
            p2 = _write_dummy_pdf(inp / "beta.pdf", b"beta")
            mapping = {
                str(p1): _pages("UNIQUE_TOKEN_ALPHA k-means clustering"),
                str(p2): _pages("UNIQUE_TOKEN_BETA gaussian mixture models"),
            }
            embeddings = FakeEmbeddings()
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                rec1 = prepare_presentation(
                    pdf_path=p1,
                    input_dir=inp,
                    output_dir=out,
                    document_id=collection_safe_id("alpha.pdf"),
                    collection_name=collection_safe_id("alpha.pdf"),
                    embedding_model="fake",
                    chunk_size=1000,
                    chunk_overlap=0,
                    embeddings=embeddings,
                )
                rec2 = prepare_presentation(
                    pdf_path=p2,
                    input_dir=inp,
                    output_dir=out,
                    document_id=collection_safe_id("beta.pdf"),
                    collection_name=collection_safe_id("beta.pdf"),
                    embedding_model="fake",
                    chunk_size=1000,
                    chunk_overlap=0,
                    embeddings=embeddings,
                )
            self.assertEqual(rec1["rag_status"], "success")
            self.assertEqual(rec2["rag_status"], "success")
            from rag_pipeline import RAGPipeline

            id1 = rec1["document_id"]
            id2 = rec2["document_id"]
            pipe1 = RAGPipeline(
                vector_store_path=str(out / rec1["index_dir"]),
                embeddings=embeddings,
            )
            pipe1.load_existing_index(collection_name=id1)
            pipe1.assert_exclusive_document(id1)
            hits1 = pipe1.search("UNIQUE_TOKEN_ALPHA", k=4)
            for doc, _score in hits1:
                self.assertEqual(doc.metadata.get("pdf_id"), id1)
                self.assertNotIn("UNIQUE_TOKEN_BETA", doc.page_content)
            pipe1.close()

            pipe2 = RAGPipeline(
                vector_store_path=str(out / rec2["index_dir"]),
                embeddings=embeddings,
            )
            pipe2.load_existing_index(collection_name=id2)
            pipe2.assert_exclusive_document(id2)
            hits2 = pipe2.search("UNIQUE_TOKEN_BETA", k=4)
            for doc, _score in hits2:
                self.assertEqual(doc.metadata.get("pdf_id"), id2)
                self.assertNotIn("UNIQUE_TOKEN_ALPHA", doc.page_content)
            pipe2.close()

    def test_same_pipeline_does_not_append_across_collections(self):
        with _tempdir() as td:
            root = Path(td)
            from rag_pipeline import RAGPipeline

            embeddings = FakeEmbeddings()
            pipe = RAGPipeline(vector_store_path=str(root / "unused"), embeddings=embeddings)
            mapping = {
                str(root / "one.pdf"): _pages("DOC_ONE_ONLY"),
                str(root / "two.pdf"): _pages("DOC_TWO_ONLY"),
            }
            _write_dummy_pdf(root / "one.pdf", b"1")
            _write_dummy_pdf(root / "two.pdf", b"2")
            with patch.object(pipe, "load_pdf_from_file", _fake_loader(mapping)):
                pipe.add_pdf_to_index(
                    pdf_path=str(root / "one.pdf"),
                    document_id="doc_one",
                    collection_name="doc_one",
                    persist_directory=root / "store_one",
                )
                pipe.add_pdf_to_index(
                    pdf_path=str(root / "two.pdf"),
                    document_id="doc_two",
                    collection_name="doc_two",
                    persist_directory=root / "store_two",
                )
            pipe.assert_exclusive_document("doc_two")
            reopen = RAGPipeline(
                vector_store_path=str(root / "store_one"), embeddings=embeddings
            )
            reopen.load_existing_index("doc_one")
            reopen.assert_exclusive_document("doc_one")
            joined = " ".join(m.get("pdf_id", "") for m in reopen.chunk_metadatas())
            self.assertNotIn("doc_two", joined)
            pipe.close()
            reopen.close()


class TestResumeAndEmpty(unittest.TestCase):
    def test_resume_does_not_duplicate_chunks(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            pdf = _write_dummy_pdf(inp / "same.pdf", b"same-v1")
            mapping = {str(pdf): _pages("resume token clustering")}
            embeddings = FakeEmbeddings()
            ns = _args(input_dir=inp, output_dir=out, stage="rag")
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                code1 = run_batch(ns, embeddings=embeddings, stdout=io.StringIO())
                self.assertEqual(code1, 0)
                manifest = load_manifest(out / "manifest.json")
                rec = manifest["presentations"][0]
                first_count = rec["chunk_count"]
                ns_resume = _args(input_dir=inp, output_dir=out, stage="rag", resume=True)
                code2 = run_batch(ns_resume, embeddings=embeddings, stdout=io.StringIO())
            self.assertEqual(code2, 0)
            rec2 = load_manifest(out / "manifest.json")["presentations"][0]
            self.assertEqual(rec2["chunk_count"], first_count)
            self.assertIn("rag_resumed", rec2.get("warnings") or [])
            from rag_pipeline import RAGPipeline

            pipe = RAGPipeline(
                vector_store_path=str(out / rec2["index_dir"]),
                embeddings=embeddings,
            )
            pipe.load_existing_index(rec2["collection_name"])
            self.assertEqual(len(pipe.chunk_metadatas()), first_count)
            pipe.close()

    def test_changed_hash_or_config_invalidates_resume(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            pdf = _write_dummy_pdf(inp / "chg.pdf", b"v1")
            mapping = {str(pdf): _pages("config token")}
            embeddings = FakeEmbeddings()
            ns = _args(input_dir=inp, output_dir=out, stage="rag")
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                self.assertEqual(run_batch(ns, embeddings=embeddings, stdout=io.StringIO()), 0)
                rec = load_manifest(out / "manifest.json")["presentations"][0]
                pdf.write_bytes(b"%PDF-DUMMY\nv2\n")
                ns_resume = _args(input_dir=inp, output_dir=out, stage="rag", resume=True)
                run_batch(ns_resume, embeddings=embeddings, stdout=io.StringIO())
                rec_hash = load_manifest(out / "manifest.json")["presentations"][0]
                self.assertNotEqual(rec_hash["source_hash"], rec["source_hash"])
                self.assertNotIn("rag_resumed", rec_hash.get("warnings") or [])

                ns_cfg = _args(
                    input_dir=inp,
                    output_dir=out,
                    stage="rag",
                    resume=True,
                    chunk_size=400,
                    chunk_overlap=20,
                )
                run_batch(ns_cfg, embeddings=embeddings, stdout=io.StringIO())
                rec_cfg = load_manifest(out / "manifest.json")["presentations"][0]
                self.assertEqual(rec_cfg["chunk_size"], 400)
                self.assertNotIn("rag_resumed", rec_cfg.get("warnings") or [])

    def test_empty_extraction_cannot_generate(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            pdf = _write_dummy_pdf(inp / "blank.pdf", b"blank")
            mapping = {str(pdf): _pages("", n=2)}
            embeddings = FakeEmbeddings()
            ns = _args(input_dir=inp, output_dir=out, stage="all")
            called = {"n": 0}

            def boom(cfg: RAGConfig):
                called["n"] += 1
                raise AssertionError("generation must not start for OCR-blocked PDFs")

            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                code = run_batch(
                    ns,
                    embeddings=embeddings,
                    backbone_factory=boom,
                    stdout=io.StringIO(),
                )
            self.assertEqual(code, 1)
            self.assertEqual(called["n"], 0)
            rec = load_manifest(out / "manifest.json")["presentations"][0]
            self.assertEqual(rec["rag_status"], "requires_ocr")
            source_md = (out / rec["source_md"]).read_text(encoding="utf-8")
            self.assertIn("_empty_", source_md)
            with self.assertRaises(Exception):
                validate_prepared_record(
                    rec,
                    input_dir=inp,
                    output_dir=out,
                    expected_config={
                        "embedding_model": rec["embedding_model"],
                        "chunk_size": rec["chunk_size"],
                        "chunk_overlap": rec["chunk_overlap"],
                        "chunk_by_page": True,
                    },
                )


class TestGenerateIsolation(unittest.TestCase):
    def test_separate_markdown_and_failure_isolation(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            good = _write_dummy_pdf(inp / "good.pdf", b"good")
            bad = _write_dummy_pdf(inp / "bad.pdf", b"bad")
            mapping = {
                str(good): _pages("GOOD_PRESENTATION_TEXT linear projections"),
                str(bad): _pages(""),
            }
            embeddings = FakeEmbeddings()
            ns = _args(input_dir=inp, output_dir=out, stage="all")
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                code = run_batch(
                    ns,
                    embeddings=embeddings,
                    backbone_factory=FakeBackbone,
                    stdout=io.StringIO(),
                )
            self.assertEqual(code, 1)
            manifest = load_manifest(out / "manifest.json")
            by_rel = {r["relative_path"]: r for r in manifest["presentations"]}
            self.assertEqual(by_rel["good.pdf"]["generate_status"], "success")
            self.assertEqual(by_rel["bad.pdf"]["rag_status"], "requires_ocr")
            results = out / by_rel["good.pdf"]["results_md"]
            text = results.read_text(encoding="utf-8")
            self.assertIn("Question for", text)
            self.assertIn("Correct answer: A. First option", text)
            self.assertIn("## Trace", text)
            self.assertIn("## AXIS", text)
            self.assertTrue((out / by_rel["good.pdf"]["experiment_json"]).is_file())
            self.assertFalse(by_rel["bad.pdf"].get("results_md"))

    def test_stage_rag_does_not_construct_backbone(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            pdf = _write_dummy_pdf(inp / "only.pdf", b"only")
            mapping = {str(pdf): _pages("only text")}
            called = {"n": 0}

            def boom(cfg: RAGConfig):
                called["n"] += 1
                return FakeBackbone(cfg)

            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                code = run_batch(
                    _args(input_dir=inp, output_dir=out, stage="rag"),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=boom,
                    stdout=io.StringIO(),
                )
            self.assertEqual(code, 0)
            self.assertEqual(called["n"], 0)

    def test_empty_quiz_does_not_overwrite_prior_results(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            pdf = _write_dummy_pdf(inp / "keep.pdf", b"keep")
            mapping = {str(pdf): _pages("keep text")}
            embeddings = FakeEmbeddings()
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="all"),
                    embeddings=embeddings,
                    backbone_factory=FakeBackbone,
                    stdout=io.StringIO(),
                )
                rec = load_manifest(out / "manifest.json")["presentations"][0]
                results = out / rec["results_md"]
                prior = results.read_text(encoding="utf-8")
                code = run_batch(
                    _args(input_dir=inp, output_dir=out, stage="generate", overwrite=True),
                    embeddings=embeddings,
                    backbone_factory=EmptyQuizBackbone,
                    stdout=io.StringIO(),
                )
            self.assertEqual(code, 1)
            self.assertEqual(results.read_text(encoding="utf-8"), prior)


class TestQuestionLimitCli(unittest.TestCase):
    def test_default_max_questions_is_ten(self):
        parser = build_arg_parser()
        ns = parser.parse_args(["--input-dir", "presentations", "--output-dir", "out"])
        self.assertEqual(ns.max_questions, 10)

    def test_workflow_profile_receives_explicit_cap(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            pdf = _write_dummy_pdf(inp / "cap.pdf", b"cap")
            mapping = {str(pdf): _pages("cap text")}
            captured = {}

            class CaptureBackbone(FakeBackbone):
                def __init__(self, cfg, profile=None):
                    super().__init__(cfg, profile)
                    captured["limit"] = getattr(
                        getattr(profile, "orchestrator_config", None),
                        "max_final_questions",
                        None,
                    )

            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="all", max_questions=5),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=CaptureBackbone,
                    stdout=io.StringIO(),
                )
            self.assertEqual(captured.get("limit"), 5)
            rec = load_manifest(out / "manifest.json")["presentations"][0]
            self.assertEqual(rec["max_questions"], 5)
            self.assertEqual(rec["generation_config"]["question_limit_semantics"], "final_accepted_mcqs")
            self.assertEqual(rec["generation"]["requested_max_questions"], 5)
            self.assertEqual(rec["explanation"]["explained_question_count"], rec["explained_question_count"])

    def test_invalid_limit_fails_before_work(self):
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--input-dir", "presentations", "--max-questions", "0"])
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            _write_dummy_pdf(inp / "a.pdf", b"a")
            ns = argparse.Namespace(**vars(_args(input_dir=inp, output_dir=out, stage="rag")))
            ns.max_questions = 0
            called = {"n": 0}

            def boom(*_a, **_k):
                called["n"] += 1

            code = run_batch(ns, embeddings=FakeEmbeddings(), pipeline=boom, stdout=io.StringIO())
            self.assertEqual(code, 2)
            self.assertEqual(called["n"], 0)
            self.assertFalse(out.exists())


class TestPdfSelection(unittest.TestCase):
    def test_list_pdfs_is_read_only(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            _write_dummy_pdf(inp / "alpha.pdf", b"a")
            _write_dummy_pdf(inp / "beta.pdf", b"b")
            buf = io.StringIO()
            called = {"n": 0}

            def boom(*_a, **_k):
                called["n"] += 1
                raise AssertionError("models must not load for --list-pdfs")

            code = run_batch(
                _args(input_dir=inp, output_dir=out, list_pdfs=True),
                embeddings=FakeEmbeddings(),
                pipeline=boom,
                backbone_factory=boom,
                stdout=buf,
            )
            text = buf.getvalue()
            self.assertEqual(code, 0)
            self.assertEqual(called["n"], 0)
            self.assertFalse(out.exists())
            self.assertIn("alpha.pdf", text)
            self.assertIn("beta.pdf", text)
            self.assertIn("document_id", text)

    def test_select_one_pdf_and_dedupe(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            a = _write_dummy_pdf(inp / "alpha.pdf", b"a")
            b = _write_dummy_pdf(inp / "beta.pdf", b"b")
            mapping = {
                str(a): _pages("alpha text clustering"),
                str(b): _pages("beta text clustering"),
            }
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                code = run_batch(
                    _args(
                        input_dir=inp,
                        output_dir=out,
                        stage="rag",
                        pdfs=["alpha.pdf", "alpha.pdf"],
                    ),
                    embeddings=FakeEmbeddings(),
                    stdout=io.StringIO(),
                )
            self.assertEqual(code, 0)
            recs = load_manifest(out / "manifest.json")["presentations"]
            self.assertEqual([r["relative_path"] for r in recs], ["alpha.pdf"])

    def test_unknown_and_ambiguous_fail_without_processing(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            (inp / "sub1").mkdir(parents=True)
            (inp / "sub2").mkdir(parents=True)
            _write_dummy_pdf(inp / "sub1" / "foo.pdf", b"1")
            _write_dummy_pdf(inp / "sub2" / "foo.pdf", b"2")
            buf = io.StringIO()
            code = run_batch(
                _args(input_dir=inp, output_dir=out, stage="rag", recursive=True, pdfs=["missing.pdf"]),
                embeddings=FakeEmbeddings(),
                stdout=buf,
            )
            self.assertEqual(code, 1)
            self.assertFalse(out.exists())
            self.assertIn("Unknown --pdf", buf.getvalue())

            buf2 = io.StringIO()
            code2 = run_batch(
                _args(input_dir=inp, output_dir=out, stage="rag", recursive=True, pdfs=["foo.pdf"]),
                embeddings=FakeEmbeddings(),
                stdout=buf2,
            )
            self.assertEqual(code2, 1)
            self.assertFalse(out.exists())
            self.assertIn("Ambiguous filename", buf2.getvalue())
            self.assertIn("sub1/foo.pdf", buf2.getvalue().replace("\\", "/"))

    def test_unselected_records_are_preserved(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            a = _write_dummy_pdf(inp / "alpha.pdf", b"a")
            b = _write_dummy_pdf(inp / "beta.pdf", b"b")
            mapping = {
                str(a): _pages("alpha text"),
                str(b): _pages("beta text"),
            }
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="all"),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=FakeBackbone,
                    stdout=io.StringIO(),
                )
                before = load_manifest(out / "manifest.json")
                beta = next(r for r in before["presentations"] if r["relative_path"] == "beta.pdf")
                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="generate", pdfs=["alpha.pdf"], overwrite=True),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=FakeBackbone,
                    stdout=io.StringIO(),
                )
            after = load_manifest(out / "manifest.json")
            beta_after = next(r for r in after["presentations"] if r["relative_path"] == "beta.pdf")
            self.assertEqual(beta["run_id"], beta_after["run_id"])
            self.assertEqual(beta["generate_status"], beta_after["generate_status"])

    def test_document_id_selection(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            a = _write_dummy_pdf(inp / "alpha.pdf", b"a")
            b = _write_dummy_pdf(inp / "beta.pdf", b"b")
            mapping = {str(a): _pages("alpha"), str(b): _pages("beta")}
            doc_id = collection_safe_id("alpha.pdf")
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                code = run_batch(
                    _args(input_dir=inp, output_dir=out, stage="rag", document_ids=[doc_id]),
                    embeddings=FakeEmbeddings(),
                    stdout=io.StringIO(),
                )
            self.assertEqual(code, 0)
            recs = load_manifest(out / "manifest.json")["presentations"]
            self.assertEqual([r["relative_path"] for r in recs], ["alpha.pdf"])

    def test_resume_rejects_legacy_explanation_only_limit_and_changed_limit(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            pdf = _write_dummy_pdf(inp / "keep.pdf", b"keep")
            mapping = {str(pdf): _pages("keep text")}
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="all", max_questions=10),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=FakeBackbone,
                    stdout=io.StringIO(),
                )
                rec = load_manifest(out / "manifest.json")["presentations"][0]
                rec["generation_config"] = {
                    "llm_model": rec["generation_config"]["llm_model"],
                    "model_type": "openai",
                    "temperature": 0.0,
                    "seed": 42,
                    "max_rounds": 5,
                    "max_questions": 10,
                    "orchestrator": "langgraph",
                    "question": rec["generation_config"]["question"],
                }
                from mas_langgraph.batch.manifest import save_manifest, MANIFEST_NAME

                manifest = load_manifest(out / "manifest.json")
                manifest["presentations"][0] = rec
                save_manifest(out / MANIFEST_NAME, manifest)
                called = {"n": 0}

                class CountingBackbone(FakeBackbone):
                    def run_full_workflow(self):
                        called["n"] += 1
                        return super().run_full_workflow()

                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="generate", resume=True, max_questions=10),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=CountingBackbone,
                    stdout=io.StringIO(),
                )
                self.assertEqual(called["n"], 1)

                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="all", max_questions=10),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=FakeBackbone,
                    stdout=io.StringIO(),
                )
                called["n"] = 0
                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="generate", resume=True, max_questions=5),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=CountingBackbone,
                    stdout=io.StringIO(),
                )
                self.assertEqual(called["n"], 1)

    def test_resume_rejects_stale_source_even_if_generation_config_matches(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            pdf = _write_dummy_pdf(inp / "chg.pdf", b"v1")
            mapping = {str(pdf): _pages("v1 text")}
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="all"),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=FakeBackbone,
                    stdout=io.StringIO(),
                )
                pdf.write_bytes(b"%PDF-DUMMY\nv2\n")
                rec = run_batch(
                    _args(input_dir=inp, output_dir=out, stage="generate", resume=True),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=FakeBackbone,
                    stdout=io.StringIO(),
                )
            self.assertEqual(rec, 1)
            stored = load_manifest(out / "manifest.json")["presentations"][0]
            self.assertEqual(stored["generate_status"], "blocked")
            self.assertNotIn("generate_resumed", stored.get("warnings") or [])


class OpportunityBackbone:
    """Factual workflow double: min(profile cap, legitimate opportunities) accepted MCQs."""

    def __init__(self, cfg: RAGConfig, profile=None, *, opportunities: int = 6):
        self.cfg = cfg
        self.profile = profile
        self.opportunities = int(opportunities)
        limit = getattr(
            getattr(profile, "orchestrator_config", None),
            "max_final_questions",
            None,
        )
        accepted = self.opportunities
        if limit is not None:
            accepted = min(int(limit), self.opportunities)
        self.accepted_n = accepted
        self._refs = tuple(
            ArtifactRef(f"q_{i:08d}", 1) for i in range(1, accepted + 1)
        )

    def run_full_workflow(self) -> Dict[str, Any]:
        limit = getattr(
            getattr(self.profile, "orchestrator_config", None),
            "max_final_questions",
            None,
        )
        cap = limit is not None and self.accepted_n >= int(limit)
        exhausted = limit is not None and self.accepted_n < int(limit)
        return {
            "run_id": f"run_{self.cfg.pdf_id}",
            "requested_max_questions": limit,
            "final_accepted_questions": self.accepted_n,
            "question_cap_reached": cap,
            "generation_exhausted": exhausted,
            "generation_exhaustion_reasons": (
                ["no_supported_generation_opportunities"] if exhausted else []
            ),
        }

    def list_explanation_experiment_refs(self, include_historical_versions: bool = False):
        del include_historical_versions
        return self._refs

    def generate_explanation_experiment(self, **kwargs: Any) -> QuizExplanationExperiment:
        refs = list(kwargs.get("candidate_refs") or self._refs)
        cases = []
        for ref in refs:
            exp = _mini_experiment(self.cfg.pdf_id, f"Question {ref.logical_id}?")
            case = exp.cases[0]
            object.__setattr__(case, "candidate_ref", ref)
            object.__setattr__(case, "question_stem", f"Question {ref.logical_id}?")
            cases.append(case)
        return QuizExplanationExperiment(
            schema_version=1,
            run_id=f"run_{self.cfg.pdf_id}",
            branch_id="factual",
            experiment_seed=int(kwargs.get("experiment_seed") or 42),
            user_question=str(kwargs.get("question") or "Why was this question selected?"),
            cases=tuple(cases),
        )


class AlwaysSixAcceptedBackbone(FakeBackbone):
    """Ignores the factual cap; used to prove batch does not slice explanations by max_questions."""

    def list_explanation_experiment_refs(self, include_historical_versions: bool = False):
        del include_historical_versions
        return tuple(ArtifactRef(f"q_{i:08d}", 1) for i in range(1, 7))

    def generate_explanation_experiment(self, **kwargs: Any) -> QuizExplanationExperiment:
        refs = list(kwargs.get("candidate_refs") or self.list_explanation_experiment_refs())
        cases = []
        for ref in refs:
            exp = _mini_experiment(self.cfg.pdf_id, f"Question {ref.logical_id}?")
            case = exp.cases[0]
            object.__setattr__(case, "candidate_ref", ref)
            cases.append(case)
        return QuizExplanationExperiment(
            schema_version=1,
            run_id=f"run_{self.cfg.pdf_id}",
            branch_id="factual",
            experiment_seed=42,
            user_question="Why was this question selected?",
            cases=tuple(cases),
        )


def _run_opportunity_batch(*, max_questions: int, max_explain=None, opportunities: int = 6):
    with _tempdir() as td:
        root = Path(td)
        inp = root / "in"
        out = root / "out"
        pdf = _write_dummy_pdf(inp / "demo.pdf", b"demo")
        mapping = {str(pdf): _pages("demo text clustering")}

        def factory(cfg, profile=None):
            return OpportunityBackbone(cfg, profile, opportunities=opportunities)

        kwargs = dict(
            input_dir=inp,
            output_dir=out,
            stage="all",
            max_questions=max_questions,
        )
        if max_explain is not None:
            kwargs["max_explain_questions"] = max_explain
        with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
            code = run_batch(
                _args(**kwargs),
                embeddings=FakeEmbeddings(),
                backbone_factory=factory,
                stdout=io.StringIO(),
            )
        rec = load_manifest(out / "manifest.json")["presentations"][0]
        return code, rec


class TestAcceptedQuestionTargetSemantics(unittest.TestCase):
    def test_cli_help_describes_target_not_explanation_slice(self):
        parser = build_arg_parser()
        help_text = " ".join(parser.format_help().split())
        self.assertIn("Maximum target number of final accepted MCQs generated per presentation", help_text)
        self.assertIn("may return fewer if valid evidence-supported questions are exhausted", help_text)
        self.assertIn("Trace, Trace+Provenance, and AXIS explanations", help_text)
        self.assertIn("Defaults to all accepted questions", help_text)

    def test_max_questions_one_is_factual_quiz_size(self):
        code, rec = _run_opportunity_batch(max_questions=1, opportunities=6)
        self.assertEqual(code, 0)
        self.assertEqual(rec["final_accepted_questions"], 1)
        self.assertEqual(rec["question_count"], 1)
        self.assertEqual(rec["explained_question_count"], 1)
        self.assertTrue(rec["question_cap_reached"])
        self.assertFalse(rec["generation_exhausted"])

    def test_enough_opportunities_reach_ten(self):
        code, rec = _run_opportunity_batch(max_questions=10, opportunities=12)
        self.assertEqual(code, 0)
        self.assertEqual(rec["generation"]["final_accepted_questions"], 10)
        self.assertTrue(rec["question_cap_reached"])
        self.assertFalse(rec["generation_exhausted"])
        self.assertEqual(rec["explained_question_count"], 10)

    def test_only_six_opportunities_exhausts_below_ten(self):
        code, rec = _run_opportunity_batch(max_questions=10, opportunities=6)
        self.assertEqual(code, 0)
        self.assertEqual(rec["final_accepted_questions"], 6)
        self.assertFalse(rec["question_cap_reached"])
        self.assertTrue(rec["generation_exhausted"])
        self.assertTrue(rec["generation_exhaustion_reasons"])
        self.assertEqual(rec["explained_question_count"], 6)

    def test_explain_cap_is_independent_of_quiz_size(self):
        code, rec = _run_opportunity_batch(
            max_questions=10, max_explain=2, opportunities=12
        )
        self.assertEqual(code, 0)
        self.assertEqual(rec["final_accepted_questions"], 10)
        self.assertEqual(rec["explained_question_count"], 2)
        self.assertEqual(rec["explanation"]["max_explain_questions"], 2)
        self.assertEqual(rec["generation"]["final_accepted_questions"], 10)

    def test_default_explains_all_accepted(self):
        code, rec = _run_opportunity_batch(max_questions=10, opportunities=7)
        self.assertEqual(code, 0)
        self.assertEqual(rec["final_accepted_questions"], 7)
        self.assertIsNone(rec["max_explain_questions"])
        self.assertEqual(rec["explained_question_count"], 7)

    def test_batch_does_not_slice_explanations_by_max_questions(self):
        with _tempdir() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            pdf = _write_dummy_pdf(inp / "six.pdf", b"six")
            mapping = {str(pdf): _pages("six text")}
            with patch("rag_pipeline.RAGPipeline.load_pdf_from_file", _fake_loader(mapping)):
                run_batch(
                    _args(input_dir=inp, output_dir=out, stage="all", max_questions=1),
                    embeddings=FakeEmbeddings(),
                    backbone_factory=AlwaysSixAcceptedBackbone,
                    stdout=io.StringIO(),
                )
            rec = load_manifest(out / "manifest.json")["presentations"][0]
            self.assertEqual(rec["explained_question_count"], 6)
            self.assertEqual(rec["explanation"]["explained_question_count"], 6)

    def test_wald_like_six_accepted_ten_target_explain_cap_independent(self):
        code, rec = _run_opportunity_batch(
            max_questions=10, max_explain=1, opportunities=6
        )
        self.assertEqual(code, 0)
        self.assertEqual(rec["final_accepted_questions"], 6)
        self.assertFalse(rec["question_cap_reached"])
        self.assertTrue(rec["generation_exhausted"])
        self.assertEqual(rec["explained_question_count"], 1)
        self.assertNotEqual(rec["final_accepted_questions"], rec["explained_question_count"])


if __name__ == "__main__":
    unittest.main()
