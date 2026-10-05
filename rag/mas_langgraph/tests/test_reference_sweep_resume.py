"""Offline resume of exhaustive AXIS reference sweeps after process death."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping
from unittest.mock import MagicMock, patch

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.models import ExplainedCase
from mas_langgraph.axis.slicing import CaseSlice
from mas_langgraph.core.checkpointer import default_factual_checkpointer
from mas_langgraph.evaluation.effect_reference import (
    load_reference_intervention_rows,
    resume_row_is_complete,
    run_effect_reference_sweep_for_case,
    upsert_reference_csv,
)
from mas_langgraph.evaluation.reference_sweep_snapshot import (
    MissingReferenceSweepSnapshotError,
    ReferenceSweepIdentityMismatchError,
    ReferenceSweepSnapshotError,
    build_identity,
    extract_checkpointer_state,
    load_reference_sweep_progress,
    load_reference_sweep_snapshot,
    resolve_reference_sweep_action,
    restore_checkpointer_state,
    save_reference_sweep_snapshot,
    snapshot_dir,
    validate_identity,
    wrap_persist_with_progress,
)
from mas_langgraph.replay.executor import WorkflowRunResult


class _AxisResult:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Adapter:
    def observe_factual(self, **_kwargs: Any) -> Any:
        return None

    def observe_replay(self, **_kwargs: Any) -> Any:
        return None

    def compare_subject_content(self, **_kwargs: Any) -> Any:
        return None


class _InterruptSweep(RuntimeError):
    """Raised in tests after a bounded number of live executions."""


def _universe() -> List[Dict[str, Any]]:
    rows = []
    for op, sig in (
        ("REMOVE", "REMOVE|a:v1||c_1"),
        ("REMOVE", "REMOVE|b:v1||c_1"),
        ("WHATIF", "WHATIF|a:v1|b:v1|c_1"),
        ("ADD", "ADD|c:v1||c_1"),
    ):
        rows.append(
            {
                "reference_intervention_signature": sig,
                "operation": op,
                "generated_query": f"{op}({sig})",
                "resolved": object(),
                "signature": object(),
            }
        )
    return rows


def _axis_result(run_id: str) -> _AxisResult:
    explained = ExplainedCase(
        run_id=run_id,
        artifact_ref=ArtifactRef("q", 1),
        branch_id="factual",
        decision_ref=ArtifactRef("decision:q", 1),
        outcome="accept",
    )
    case_slice = CaseSlice(
        explained_case=explained,
        relevant_artifact_refs=(),
        relevant_occurrences=(),
        relevant_relations=(),
    )
    return _AxisResult(
        explained_case=explained,
        case_slice=case_slice,
        interrogation_attempts=[],
        replay_comparisons=[],
        intervention_records=[],
    )


def _execute_logged(log_path: Path):
    def fake_execute(*, candidate, lecturebank_id, candidate_ref, **_kwargs):
        sig = str(candidate["reference_intervention_signature"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(sig + "\n")
        return {
            "lecturebank_id": lecturebank_id,
            "candidate_ref": candidate_ref,
            "reference_intervention_signature": sig,
            "operation": candidate.get("operation"),
            "generated_query": candidate.get("generated_query"),
            "reference_status": "valid_no_effect",
            "reference_evaluable": True,
            "supported_effect": False,
        }

    return fake_execute


def _run_mocked_sweep(
    *,
    work: Path,
    interrupt_after: int | None,
    resume_rows: List[Mapping[str, Any]],
) -> None:
    from mas_langgraph.evaluation import effect_reference as er

    run_id = "run_resume_test"
    document_id = "lb_141"
    lecturebank_id = "141"
    candidate_ref = "q:v1"
    universe = _universe()
    log_path = work / "executed_signatures.txt"
    csv_path = work / "metrics" / "reference_interventions.csv"
    snap = snapshot_dir(work, document_id)
    loaded = load_reference_sweep_snapshot(snap)
    identity = loaded["identity"]
    runtime = loaded["runtime"]
    live = {"n": 0}

    def persist(rows):
        upsert_reference_csv(csv_path, rows)
        wrapped = wrap_persist_with_progress(
            directory=snap,
            identity=identity,
            frozen_universes=runtime.get("frozen_universes") or {},
            persist_rows_callback=None,
            csv_path=csv_path,
        )
        wrapped(rows)
        if interrupt_after is not None and live["n"] >= int(interrupt_after):
            raise _InterruptSweep(f"interrupt after {live['n']} live executions")

    def fake_execute(**kwargs):
        live["n"] += 1
        return _execute_logged(log_path)(**kwargs)

    replay_executor = MagicMock()
    replay_executor.get_or_create_llm_ledger = None
    with (
        patch.object(er, "execute_reference_intervention", side_effect=fake_execute),
        patch.object(er, "AxisQueryExecutor", return_value=MagicMock()),
    ):
        run_effect_reference_sweep_for_case(
            lecturebank_id=lecturebank_id,
            candidate_ref=candidate_ref,
            axis_result=runtime["axis_results"][candidate_ref],
            replay_executor=replay_executor,
            factual_run_result=runtime["factual_run_result"],
            case_definition=None,
            checkpoint_index={"records": []},
            contracts=MagicMock(),
            observation_adapter=_Adapter(),
            resume_rows=resume_rows,
            persist_rows_callback=persist,
            frozen_universe=runtime["frozen_universes"][candidate_ref],
        )
    _ = universe
    _ = run_id


def child_resume(work_dir: str) -> None:
    work = Path(work_dir)
    csv_path = work / "metrics" / "reference_interventions.csv"
    resume_rows = load_reference_intervention_rows(csv_path)
    _run_mocked_sweep(work=work, interrupt_after=None, resume_rows=resume_rows)


def _real_rag_step2_state():
    """Production blackboard path: RAG CooperativeState + ensure_step2_mcq_state."""
    rag_dir = Path(__file__).resolve().parents[2]
    if str(rag_dir) not in sys.path:
        sys.path.insert(0, str(rag_dir))
    from rag_agents_cooperative import (
        CooperativeState,
        DocSection,
        EvidenceChunk,
        KnowledgeAtom,
        LearningObjective,
    )
    from mas_langgraph.integrations.rag_agents_bridge import attach_snapshot_counts

    cs = CooperativeState()
    attach_snapshot_counts(cs)
    cs.add_section(
        DocSection(
            "sec_kmeans",
            "k-Means",
            (6, 10),
            0.9,
            content_summary="k-Means clustering",
        )
    )
    cs.add_learning_objective(
        LearningObjective("lo_kmeans", "sec_kmeans", "Explain k-means.", "intermediate")
    )
    cs.add_evidence_chunk(
        EvidenceChunk(
            "E17",
            "k-means assigns points to nearest centroids then updates them.",
            {"page_number": 8, "section_id": "sec_kmeans"},
            quality_score=0.9,
        )
    )
    cs.add_knowledge_atom(
        KnowledgeAtom(
            "A3",
            "k-means alternates assignment.",
            "procedure",
            [{"chunk_id": "E17"}],
            section_id="sec_kmeans",
            lo_id="lo_kmeans",
        )
    )
    cs.link_chunk_to_atom("E17", "A3")
    cs.update_section_evidence_status()
    return cs


def child_verify_rag_snapshot(work_dir: str) -> None:
    work = Path(work_dir)
    loaded = load_reference_sweep_snapshot(snapshot_dir(work, "lb_141"))
    cs = loaded["runtime"]["cooperative_state"]
    if cs is None:
        raise AssertionError("restored cooperative_state is missing")
    helper = getattr(cs, "update_section_evidence_status", None)
    if not callable(helper):
        raise AssertionError("restored update_section_evidence_status is missing")
    fn = getattr(helper, "__func__", helper)
    if "<locals>" in str(getattr(fn, "__qualname__", "")):
        raise AssertionError(
            f"restored helper is still a nested local: {fn.__qualname__}"
        )
    status = cs.update_section_evidence_status()
    if "sec_kmeans" not in dict(cs.section_evidence_status or {}):
        raise AssertionError(
            f"restored helper did not refresh section evidence: {status!r}"
        )
    counts = cs.snapshot_counts()
    if int(counts.get("doc_outline_len") or 0) < 1:
        raise AssertionError(f"restored snapshot_counts lost outline: {counts!r}")
    csv_path = work / "metrics" / "reference_interventions.csv"
    resume_rows = load_reference_intervention_rows(csv_path)
    _run_mocked_sweep(work=work, interrupt_after=None, resume_rows=resume_rows)


class TestReferenceSweepResume(unittest.TestCase):
    def test_csv_without_snapshot_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            metrics = work / "metrics"
            metrics.mkdir()
            csv_path = metrics / "reference_interventions.csv"
            upsert_reference_csv(
                csv_path,
                [
                    {
                        "lecturebank_id": "141",
                        "candidate_ref": "q:v1",
                        "reference_intervention_signature": "REMOVE|a:v1||c_1",
                        "operation": "REMOVE",
                        "reference_status": "valid_effect",
                    }
                ],
            )
            with self.assertRaises(MissingReferenceSweepSnapshotError) as ctx:
                resolve_reference_sweep_action(
                    output_dir=work,
                    document_id="lb_141",
                    lecturebank_id="141",
                    csv_path=csv_path,
                )
            message = str(ctx.exception)
            self.assertIn("MemorySaver", message)
            self.assertIn("SqliteSaver", message)
            self.assertIn("must not be merged", message)

    def test_identity_mismatch_refuses_merge(self):
        saved = build_identity(
            run_id="run_a",
            lecturebank_id="141",
            document_id="lb_141",
            source_hash="aaa",
            rag_config={"embedding_model": "x"},
            generation_config={"llm_model": "m"},
            axis_interrogation_mode="hybrid",
            planned_intervention_count=4,
            planned_question_count=1,
            candidate_refs=("q:v1",),
        )
        current = build_identity(
            run_id="run_b",
            lecturebank_id="141",
            document_id="lb_141",
            source_hash="aaa",
            rag_config={"embedding_model": "x"},
            generation_config={"llm_model": "m"},
            axis_interrogation_mode="hybrid",
            planned_intervention_count=4,
            planned_question_count=1,
            candidate_refs=("q:v1",),
        )
        with self.assertRaises(ReferenceSweepIdentityMismatchError) as ctx:
            validate_identity(saved, current)
        self.assertIn("run_id", str(ctx.exception))

    def test_memorysaver_roundtrip_preserves_storage(self):
        cp = default_factual_checkpointer()
        state = extract_checkpointer_state(cp)
        state["storage"] = {"thread-run_resume_test": {"marker": "factual-v1"}}
        restored = default_factual_checkpointer()
        restore_checkpointer_state(restored, state)
        self.assertEqual(
            getattr(restored, "storage", {}).get("thread-run_resume_test"),
            {"marker": "factual-v1"},
        )

    def test_interrupt_then_resume_in_fresh_process(self):
        run_id = "run_resume_test"
        document_id = "lb_141"
        lecturebank_id = "141"
        candidate_ref = "q:v1"
        universe = _universe()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "metrics").mkdir()
            csv_path = work / "metrics" / "reference_interventions.csv"
            log_path = work / "executed_signatures.txt"
            cp = default_factual_checkpointer()
            replay = MagicMock()
            replay._factual_histories = {}
            replay._factual_results = {run_id: {"result_text": "factual-v1"}}
            replay._factual_indices = {}
            replay._llm_ledgers = {}
            replay.branches = None
            replay.contracts = None
            replay.compiled_graph = None
            factual = WorkflowRunResult(
                run_id=run_id,
                branch_id="factual",
                final_state={"result_text": "factual-v1", "occurrence_records": []},
            )
            axis = _axis_result(run_id)
            identity = build_identity(
                run_id=run_id,
                lecturebank_id=lecturebank_id,
                document_id=document_id,
                source_hash="abc123",
                rag_config={"embedding_model": "test"},
                generation_config={"llm_model": "offline", "hybrid_add_from_source": False},
                axis_interrogation_mode="hybrid",
                planned_intervention_count=len(universe),
                planned_question_count=1,
                candidate_refs=(candidate_ref,),
            )
            save_reference_sweep_snapshot(
                snapshot_dir(work, document_id),
                identity=identity,
                checkpointer=cp,
                replay_executor=replay,
                factual_run_result=factual,
                axis_results={candidate_ref: axis},
                frozen_universes={candidate_ref: universe},
                experiment=None,
                cooperative_state=None,
                experiment_payload={"run_id": run_id, "cases": []},
            )
            with self.assertRaises(_InterruptSweep):
                _run_mocked_sweep(work=work, interrupt_after=2, resume_rows=[])
            first = [
                line.strip()
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(first), 2)
            rows = load_reference_intervention_rows(csv_path)
            self.assertEqual(sum(1 for row in rows if resume_row_is_complete(row)), 2)
            progress = load_reference_sweep_progress(snapshot_dir(work, document_id))
            self.assertIsNotNone(progress)
            self.assertEqual(progress.run_id, run_id)
            self.assertEqual(progress.planned_intervention_count, 4)
            self.assertEqual(progress.completed_intervention_count, 2)

            rag_dir = Path(__file__).resolve().parents[2]
            work_arg = json_dumps_path(work)
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    "from mas_langgraph.tests.test_reference_sweep_resume import child_resume; "
                    f"child_resume({work_arg})",
                ],
                cwd=str(rag_dir),
            )
            executed = [
                line.strip()
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(executed), 4)
            self.assertEqual(len(set(executed)), 4)
            self.assertEqual(executed[:2], first)
            loaded = load_reference_sweep_snapshot(snapshot_dir(work, document_id))
            restored_state = loaded["runtime"]["factual_run_result"].final_state
            self.assertEqual(restored_state.get("result_text"), "factual-v1")
            self.assertEqual(loaded["identity"].run_id, run_id)
            progress = load_reference_sweep_progress(snapshot_dir(work, document_id))
            self.assertEqual(progress.status, "complete")
            self.assertEqual(progress.completed_intervention_count, 4)
            final_rows = load_reference_intervention_rows(csv_path)
            self.assertEqual(len(final_rows), 4)


class TestRagEnsureStep2Snapshot(unittest.TestCase):
    def test_legacy_nested_helper_is_replaced_with_module_level(self):
        from mas_langgraph.domains.quiz.state_integration import (
            ensure_step2_mcq_state,
        )

        cs = _real_rag_step2_state()

        def update_section_evidence_status(*, min_chunks: int = 2):
            return {"nested": True}

        cs.update_section_evidence_status = update_section_evidence_status
        self.assertIn(
            "<locals>",
            getattr(cs.update_section_evidence_status, "__qualname__", ""),
        )
        ensure_step2_mcq_state(cs)
        fn = getattr(
            cs.update_section_evidence_status, "__func__", cs.update_section_evidence_status
        )
        self.assertNotIn("<locals>", str(getattr(fn, "__qualname__", "")))
        status = cs.update_section_evidence_status()
        self.assertIn("sec_kmeans", dict(cs.section_evidence_status or {}))
        self.assertTrue(status)

    def test_unpickleable_checkpointer_storage_is_reported(self):
        cp = default_factual_checkpointer()
        cp.storage = {"thread-run": lambda: None}
        with self.assertRaises(ReferenceSweepSnapshotError) as ctx:
            extract_checkpointer_state(cp)
        message = str(ctx.exception)
        self.assertIn("storage", message)
        self.assertIn("snapshot", message.lower())

    def test_unpickleable_extra_callable_is_not_silently_dropped(self):
        run_id = "run_pickle_fail"
        cs = _real_rag_step2_state()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            cp = default_factual_checkpointer()
            replay = MagicMock()
            replay._factual_histories = {}
            replay._factual_results = {run_id: {"result_text": "factual-v1"}}
            replay._factual_indices = {}
            replay._llm_ledgers = {}
            replay.branches = None
            replay.contracts = None
            replay.compiled_graph = None
            factual = WorkflowRunResult(
                run_id=run_id,
                branch_id="factual",
                final_state={
                    "cooperative_state": cs,
                    "occurrence_records": [],
                    "unexpected_runtime_hook": lambda: "live-client",
                },
            )
            identity = build_identity(
                run_id=run_id,
                lecturebank_id="141",
                document_id="lb_141",
                source_hash="abc123",
                planned_intervention_count=4,
                planned_question_count=1,
                candidate_refs=("q:v1",),
            )
            with self.assertRaises(ReferenceSweepSnapshotError) as ctx:
                save_reference_sweep_snapshot(
                    snapshot_dir(work, "lb_141"),
                    identity=identity,
                    checkpointer=cp,
                    replay_executor=replay,
                    factual_run_result=factual,
                    axis_results={"q:v1": _axis_result(run_id)},
                    frozen_universes={"q:v1": _universe()},
                    experiment=None,
                    cooperative_state=cs,
                )
            self.assertIn("pickle", str(ctx.exception).lower())

    def test_ensure_step2_state_roundtrip_fresh_process_and_mocked_replay(self):
        run_id = "run_rag_step2"
        document_id = "lb_141"
        lecturebank_id = "141"
        candidate_ref = "q:v1"
        universe = _universe()
        cs = _real_rag_step2_state()
        fn = getattr(
            cs.update_section_evidence_status, "__func__", cs.update_section_evidence_status
        )
        self.assertNotIn("<locals>", str(getattr(fn, "__qualname__", "")))
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "metrics").mkdir()
            csv_path = work / "metrics" / "reference_interventions.csv"
            log_path = work / "executed_signatures.txt"
            cp = default_factual_checkpointer()
            replay = MagicMock()
            replay._factual_histories = {}
            replay._factual_results = {
                run_id: WorkflowRunResult(
                    run_id=run_id,
                    branch_id="factual",
                    final_state={"cooperative_state": cs, "occurrence_records": []},
                )
            }
            replay._factual_indices = {}
            replay._llm_ledgers = {}
            replay.branches = None
            replay.contracts = None
            replay.compiled_graph = None
            factual = WorkflowRunResult(
                run_id=run_id,
                branch_id="factual",
                final_state={"cooperative_state": cs, "occurrence_records": []},
            )
            identity = build_identity(
                run_id=run_id,
                lecturebank_id=lecturebank_id,
                document_id=document_id,
                source_hash="abc123",
                rag_config={"embedding_model": "test"},
                generation_config={"llm_model": "offline", "hybrid_add_from_source": False},
                axis_interrogation_mode="hybrid",
                planned_intervention_count=len(universe),
                planned_question_count=1,
                candidate_refs=(candidate_ref,),
            )
            save_reference_sweep_snapshot(
                snapshot_dir(work, document_id),
                identity=identity,
                checkpointer=cp,
                replay_executor=replay,
                factual_run_result=factual,
                axis_results={candidate_ref: _axis_result(run_id)},
                frozen_universes={candidate_ref: universe},
                experiment=None,
                cooperative_state=cs,
                experiment_payload={"run_id": run_id, "cases": []},
            )
            with self.assertRaises(_InterruptSweep):
                _run_mocked_sweep(work=work, interrupt_after=2, resume_rows=[])
            first = [
                line.strip()
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(first), 2)
            rows = load_reference_intervention_rows(csv_path)
            self.assertEqual(sum(1 for row in rows if resume_row_is_complete(row)), 2)
            progress = load_reference_sweep_progress(snapshot_dir(work, document_id))
            self.assertEqual(progress.run_id, run_id)
            self.assertEqual(progress.completed_intervention_count, 2)

            rag_dir = Path(__file__).resolve().parents[2]
            work_arg = json_dumps_path(work)
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    "from mas_langgraph.tests.test_reference_sweep_resume import "
                    "child_verify_rag_snapshot; "
                    f"child_verify_rag_snapshot({work_arg})",
                ],
                cwd=str(rag_dir),
            )
            executed = [
                line.strip()
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(executed), 4)
            self.assertEqual(len(set(executed)), 4)
            self.assertEqual(executed[:2], first)
            loaded = load_reference_sweep_snapshot(snapshot_dir(work, document_id))
            restored_cs = loaded["runtime"]["cooperative_state"]
            restored_fn = getattr(
                restored_cs.update_section_evidence_status,
                "__func__",
                restored_cs.update_section_evidence_status,
            )
            self.assertNotIn("<locals>", str(getattr(restored_fn, "__qualname__", "")))
            self.assertEqual(loaded["identity"].run_id, run_id)
            restored_state = loaded["runtime"]["factual_run_result"].final_state
            self.assertEqual(
                restored_state["cooperative_state"].doc_outline[0].section_id,
                "sec_kmeans",
            )
            self.assertEqual(restored_cs.doc_outline[0].section_id, "sec_kmeans")
            progress = load_reference_sweep_progress(snapshot_dir(work, document_id))
            self.assertEqual(progress.status, "complete")
            self.assertEqual(progress.completed_intervention_count, 4)


def json_dumps_path(path: Path) -> str:
    import json

    return json.dumps(str(path))


if __name__ == "__main__":
    unittest.main()
