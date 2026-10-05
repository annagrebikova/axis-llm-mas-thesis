"""Durable resume state for exhaustive AXIS reference sweeps.

LangGraph factual checkpoints in this evaluation use an in-process
``MemorySaver``. They are discarded when the Python process exits. This
module writes the factual execution, AXIS selections, frozen intervention
universe, and checkpointer storage to disk *before* the sweep starts, then
records per-intervention progress so a later process can continue the same
run without regenerating MCQs or rerunning AXIS.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from mas_langgraph.batch.io import atomic_write_json, load_json
from mas_langgraph.batch.manifest import generation_fingerprints_compatible
from mas_langgraph.evaluation.effect_reference import (
    load_reference_intervention_rows,
    resume_row_is_complete,
)
from mas_langgraph.evaluation.lifecycle import presentation_dir

SNAPSHOT_DIRNAME = "reference_sweep"
IDENTITY_FILENAME = "identity.json"
PROGRESS_FILENAME = "progress.json"
RUNTIME_FILENAME = "runtime.pkl"
EXPERIMENT_FILENAME = "experiment.json"
UNIVERSE_FILENAME = "universe.json"
SCHEMA_VERSION = 1

REQUIRED_SNAPSHOT_FILES = (IDENTITY_FILENAME, PROGRESS_FILENAME, RUNTIME_FILENAME)

STATUS_IN_PROGRESS = "axis_complete_sweep_in_progress"
STATUS_COMPLETE = "complete"

EVAL_STATUS_SWEEP_IN_PROGRESS = "axis_complete_sweep_in_progress"


class MissingReferenceSweepSnapshotError(RuntimeError):
    """CSV/progress exists but the durable factual snapshot cannot be restored."""


class ReferenceSweepIdentityMismatchError(RuntimeError):
    """Saved sweep identity does not match the current configuration or run."""


class ReferenceSweepSnapshotError(RuntimeError):
    """The durable factual snapshot could not be serialized or restored."""


@dataclass(frozen=True)
class ReferenceSweepIdentity:
    schema_version: int
    run_id: str
    lecturebank_id: str
    document_id: str
    source_hash: str
    rag_config: Dict[str, Any]
    generation_config: Dict[str, Any]
    axis_interrogation_mode: str
    reference_max_interventions_per_question: Optional[int]
    planned_intervention_count: int
    planned_question_count: int
    candidate_refs: Tuple[str, ...]
    hybrid_add_pool_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "run_id": str(self.run_id),
            "lecturebank_id": str(self.lecturebank_id),
            "document_id": str(self.document_id),
            "source_hash": str(self.source_hash),
            "rag_config": dict(self.rag_config or {}),
            "generation_config": dict(self.generation_config or {}),
            "axis_interrogation_mode": str(self.axis_interrogation_mode),
            "reference_max_interventions_per_question": (
                self.reference_max_interventions_per_question
            ),
            "planned_intervention_count": int(self.planned_intervention_count),
            "planned_question_count": int(self.planned_question_count),
            "candidate_refs": list(self.candidate_refs),
            "hybrid_add_pool_fingerprint": str(self.hybrid_add_pool_fingerprint or ""),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ReferenceSweepIdentity":
        refs = payload.get("candidate_refs") or []
        return ReferenceSweepIdentity(
            schema_version=int(payload.get("schema_version") or SCHEMA_VERSION),
            run_id=str(payload.get("run_id") or ""),
            lecturebank_id=str(payload.get("lecturebank_id") or ""),
            document_id=str(payload.get("document_id") or ""),
            source_hash=str(payload.get("source_hash") or ""),
            rag_config=dict(payload.get("rag_config") or {}),
            generation_config=dict(payload.get("generation_config") or {}),
            axis_interrogation_mode=str(payload.get("axis_interrogation_mode") or ""),
            reference_max_interventions_per_question=payload.get(
                "reference_max_interventions_per_question"
            ),
            planned_intervention_count=int(payload.get("planned_intervention_count") or 0),
            planned_question_count=int(payload.get("planned_question_count") or 0),
            candidate_refs=tuple(str(r) for r in refs),
            hybrid_add_pool_fingerprint=str(
                payload.get("hybrid_add_pool_fingerprint") or ""
            ),
        )


@dataclass
class ReferenceSweepProgress:
    run_id: str
    status: str
    planned_intervention_count: int
    completed_intervention_count: int
    planned_question_count: int
    completed_question_count: int
    completed_keys: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": str(self.run_id),
            "status": str(self.status),
            "planned_intervention_count": int(self.planned_intervention_count),
            "completed_intervention_count": int(self.completed_intervention_count),
            "planned_question_count": int(self.planned_question_count),
            "completed_question_count": int(self.completed_question_count),
            "completed_keys": list(self.completed_keys),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ReferenceSweepProgress":
        keys = []
        for item in payload.get("completed_keys") or []:
            if isinstance(item, Mapping):
                keys.append(
                    {
                        "lecturebank_id": str(item.get("lecturebank_id") or ""),
                        "candidate_ref": str(item.get("candidate_ref") or ""),
                        "reference_intervention_signature": str(
                            item.get("reference_intervention_signature") or ""
                        ),
                    }
                )
        return ReferenceSweepProgress(
            run_id=str(payload.get("run_id") or ""),
            status=str(payload.get("status") or STATUS_IN_PROGRESS),
            planned_intervention_count=int(payload.get("planned_intervention_count") or 0),
            completed_intervention_count=int(
                payload.get("completed_intervention_count") or 0
            ),
            planned_question_count=int(payload.get("planned_question_count") or 0),
            completed_question_count=int(payload.get("completed_question_count") or 0),
            completed_keys=keys,
        )


def snapshot_dir(output_dir: Path, document_id: str) -> Path:
    return presentation_dir(output_dir, document_id) / SNAPSHOT_DIRNAME


def missing_required_files(directory: Path) -> List[str]:
    missing = []
    root = Path(directory)
    for name in REQUIRED_SNAPSHOT_FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(name)
    return missing


def atomic_write_pickle(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


_ATOMIC_PICKLE_TYPES = (str, bytes, bytearray, int, float, complex, bool, type(None))
_CHECKPOINTER_SKIP_ATTRS = {"serde", "_serde", "stack"}
_RUNTIME_RECONSTRUCT_ATTRS = {
    "llm",
    "_llm",
    "client",
    "_client",
    "openai_client",
    "embeddings",
    "vector_store",
    "vectorstore",
    "rag_pipeline",
    "rag_agents",
    "_agent_executors",
    "_lock",
    "lock",
    "serde",
    "_serde",
}
_MEMORYSAVER_STORAGE_KIND = "__memorysaver_storage__"


def _require_pickleable(value: Any, *, label: str) -> None:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise ReferenceSweepSnapshotError(
            "Cannot serialize "
            f"{label} for the durable reference-sweep snapshot: {exc}"
        ) from exc


def _is_mock(obj: Any) -> bool:
    try:
        from unittest.mock import Mock

        return isinstance(obj, Mock)
    except Exception:
        return False


def memorysaver_namespace_factory() -> Any:
    """Module-level factory matching ``InMemorySaver.storage`` inner maps."""
    return defaultdict(dict)


def _plain_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain_mapping(item) for key, item in value.items()}
    return value


def _rebuild_memorysaver_storage(items: Mapping[str, Any]) -> Any:
    storage: Any = defaultdict(memorysaver_namespace_factory)
    for thread_id, namespace_map in dict(items or {}).items():
        inner: Any = defaultdict(dict)
        if isinstance(namespace_map, dict):
            inner.update(namespace_map)
        else:
            inner = namespace_map
        storage[thread_id] = inner
    return storage


def prepare_object_for_pickle(obj: Any, *, _seen: Optional[set] = None) -> None:
    """Detach instance helpers so pickle stores dataclass fields only.

    Runtime-only handles (LLM clients, vector stores, serde, locks, mocks) are
    not walked. Helpers are reattached after save/load via
    ``restore_runtime_helpers``.
    """
    import types as pytypes

    from mas_langgraph.domains.quiz.state_integration import (
        _detach_instance_helpers,
        ensure_step2_mcq_state,
        looks_like_cooperative_state,
    )

    if obj is None or isinstance(obj, _ATOMIC_PICKLE_TYPES) or _is_mock(obj):
        return
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)
    if isinstance(
        obj,
        (
            pytypes.FunctionType,
            pytypes.MethodType,
            pytypes.BuiltinFunctionType,
            pytypes.ModuleType,
            type,
        ),
    ):
        return
    if looks_like_cooperative_state(obj):
        ensure_step2_mcq_state(obj)
        _detach_instance_helpers(obj)
        inst = getattr(obj, "__dict__", None)
        if isinstance(inst, dict):
            for name, value in inst.items():
                if name in _RUNTIME_RECONSTRUCT_ATTRS or callable(value):
                    continue
                prepare_object_for_pickle(value, _seen=_seen)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            prepare_object_for_pickle(value, _seen=_seen)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            prepare_object_for_pickle(value, _seen=_seen)
        return
    module = str(getattr(type(obj), "__module__", "") or "")
    if module.startswith(("unittest", "threading", "_thread", "socket", "http", "ssl")):
        return
    inst = getattr(obj, "__dict__", None)
    if isinstance(inst, dict):
        for name, value in inst.items():
            if name in _RUNTIME_RECONSTRUCT_ATTRS or "serde" in str(name):
                continue
            prepare_object_for_pickle(value, _seen=_seen)


def restore_runtime_helpers(obj: Any, *, _seen: Optional[set] = None) -> None:
    """Reattach module-level helpers after unpickle or after a snapshot save."""
    import types as pytypes

    from mas_langgraph.domains.quiz.state_integration import (
        looks_like_cooperative_state,
        rebind_pickleable_state_helpers,
    )

    if obj is None or isinstance(obj, _ATOMIC_PICKLE_TYPES) or _is_mock(obj):
        return
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)
    if isinstance(
        obj,
        (
            pytypes.FunctionType,
            pytypes.MethodType,
            pytypes.BuiltinFunctionType,
            pytypes.ModuleType,
            type,
        ),
    ):
        return
    if looks_like_cooperative_state(obj):
        rebind_pickleable_state_helpers(obj)
        inst = getattr(obj, "__dict__", None)
        if isinstance(inst, dict):
            for name, value in inst.items():
                if name in _RUNTIME_RECONSTRUCT_ATTRS or callable(value):
                    continue
                restore_runtime_helpers(value, _seen=_seen)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            restore_runtime_helpers(value, _seen=_seen)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            restore_runtime_helpers(value, _seen=_seen)
        return
    module = str(getattr(type(obj), "__module__", "") or "")
    if module.startswith(("unittest", "threading", "_thread", "socket", "http", "ssl")):
        return
    inst = getattr(obj, "__dict__", None)
    if isinstance(inst, dict):
        for name, value in inst.items():
            if name in _RUNTIME_RECONSTRUCT_ATTRS or "serde" in str(name):
                continue
            restore_runtime_helpers(value, _seen=_seen)


def extract_checkpointer_state(checkpointer: Any) -> Dict[str, Any]:
    """Copy MemorySaver storage. Serde and ExitStack are reconstructed on restore."""
    if checkpointer is None:
        raise MissingReferenceSweepSnapshotError(
            "Cannot snapshot the factual run: the LangGraph checkpointer is missing."
        )
    out: Dict[str, Any] = {}
    for name, value in vars(checkpointer).items():
        if name in _CHECKPOINTER_SKIP_ATTRS or "serde" in name:
            continue
        if name == "storage":
            items = _plain_mapping(value)
            prepare_object_for_pickle(items)
            _require_pickleable(
                items,
                label="LangGraph checkpointer storage",
            )
            out[name] = {_MEMORYSAVER_STORAGE_KIND: True, "items": items}
            continue
        _require_pickleable(
            value,
            label=f"LangGraph checkpointer attribute {name!r}",
        )
        out[name] = value
    if "storage" not in out and not out:
        raise MissingReferenceSweepSnapshotError(
            "Cannot snapshot the factual run: the in-memory checkpointer has no "
            "pickleable storage. This evaluation does not use an on-disk "
            "SqliteSaver or other persistent checkpoint store."
        )
    if "storage" not in out:
        raise ReferenceSweepSnapshotError(
            "Cannot snapshot the factual run: LangGraph checkpointer storage "
            "could not be copied into a durable mapping."
        )
    return out


def restore_checkpointer_state(checkpointer: Any, state: Mapping[str, Any]) -> None:
    if checkpointer is None:
        raise MissingReferenceSweepSnapshotError(
            "Cannot restore the factual run: the compiled graph has no checkpointer."
        )
    for name, value in dict(state or {}).items():
        if (
            name == "storage"
            and isinstance(value, dict)
            and value.get(_MEMORYSAVER_STORAGE_KIND)
        ):
            setattr(
                checkpointer,
                name,
                _rebuild_memorysaver_storage(value.get("items") or {}),
            )
        else:
            setattr(checkpointer, name, value)


def extract_replay_executor_state(executor: Any, run_id: str) -> Dict[str, Any]:
    run_id = str(run_id or "")
    histories = getattr(executor, "_factual_histories", {}) or {}
    results = getattr(executor, "_factual_results", {}) or {}
    indices = getattr(executor, "_factual_indices", {}) or {}
    ledgers = getattr(executor, "_llm_ledgers", {}) or {}
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "factual_results": {run_id: results.get(run_id)} if run_id else {},
        "factual_indices": {run_id: indices.get(run_id)} if run_id else {},
        "llm_ledgers": {run_id: ledgers.get(run_id)} if run_id else {},
        "branches": getattr(executor, "branches", None),
        "contracts": getattr(executor, "contracts", None),
    }
    hist = histories.get(run_id)
    if hist is not None and not _is_mock(hist):
        prepare_object_for_pickle(hist)
        payload["factual_histories"] = {run_id: hist}
    else:
        payload["factual_histories"] = {}
    if _is_mock(payload.get("branches")):
        payload["branches"] = None
    if _is_mock(payload.get("contracts")):
        payload["contracts"] = None
    prepare_object_for_pickle(payload)
    _require_pickleable(payload, label="replay executor factual state")
    return payload


def restore_replay_executor_state(executor: Any, state: Mapping[str, Any]) -> None:
    run_id = str(state.get("run_id") or "")
    results = dict(state.get("factual_results") or {})
    indices = dict(state.get("factual_indices") or {})
    histories = dict(state.get("factual_histories") or {})
    ledgers = dict(state.get("llm_ledgers") or {})
    if results:
        executor._factual_results.update(results)
    if indices:
        executor._factual_indices.update(indices)
    if histories:
        executor._factual_histories.update(histories)
    if ledgers:
        executor._llm_ledgers.update(ledgers)
    if state.get("branches") is not None:
        executor.branches = state["branches"]
    if state.get("contracts") is not None:
        executor.contracts = state["contracts"]
    executor._histories = executor._factual_histories
    executor._final_states = executor._factual_results
    executor._indices = executor._factual_indices
    if run_id and run_id not in executor._factual_histories and executor.compiled_graph:
        try:
            hist = list(
                executor.compiled_graph.get_state_history(
                    {"configurable": {"thread_id": run_id}}
                )
            )
            if hist:
                executor._factual_histories[run_id] = hist
        except Exception:
            pass


def _pickleable_factual_run_result(result: Any) -> Any:
    if result is None:
        return None
    try:
        return replace(result, raw_result=None)
    except Exception:
        try:
            result.raw_result = None
        except Exception:
            pass
        return result


def build_identity(
    *,
    run_id: str,
    lecturebank_id: str,
    document_id: str,
    source_hash: str,
    rag_config: Optional[Mapping[str, Any]] = None,
    generation_config: Optional[Mapping[str, Any]] = None,
    axis_interrogation_mode: str = "hybrid",
    reference_max_interventions_per_question: Optional[int] = None,
    planned_intervention_count: int = 0,
    planned_question_count: int = 0,
    candidate_refs: Sequence[str] = (),
    hybrid_add_pool_fingerprint: str = "",
) -> ReferenceSweepIdentity:
    return ReferenceSweepIdentity(
        schema_version=SCHEMA_VERSION,
        run_id=str(run_id or ""),
        lecturebank_id=str(lecturebank_id or ""),
        document_id=str(document_id or ""),
        source_hash=str(source_hash or ""),
        rag_config=dict(rag_config or {}),
        generation_config=dict(generation_config or {}),
        axis_interrogation_mode=str(axis_interrogation_mode or ""),
        reference_max_interventions_per_question=reference_max_interventions_per_question,
        planned_intervention_count=int(planned_intervention_count),
        planned_question_count=int(planned_question_count),
        candidate_refs=tuple(str(r) for r in candidate_refs),
        hybrid_add_pool_fingerprint=str(hybrid_add_pool_fingerprint or ""),
    )


def validate_identity(
    saved: ReferenceSweepIdentity,
    current: ReferenceSweepIdentity,
) -> None:
    if not saved.run_id:
        raise ReferenceSweepIdentityMismatchError(
            "Saved reference-sweep identity has an empty run_id; refusing to "
            "merge interventions from an unknown factual execution."
        )
    if str(saved.run_id) != str(current.run_id):
        raise ReferenceSweepIdentityMismatchError(
            "Saved reference-sweep run_id "
            f"{saved.run_id!r} does not match the restored factual run "
            f"{current.run_id!r}. Interventions from different factual "
            "executions must not be merged."
        )
    if str(saved.lecturebank_id) != str(current.lecturebank_id):
        raise ReferenceSweepIdentityMismatchError(
            "Saved lecturebank_id "
            f"{saved.lecturebank_id!r} does not match {current.lecturebank_id!r}."
        )
    if str(saved.document_id) != str(current.document_id):
        raise ReferenceSweepIdentityMismatchError(
            "Saved document_id "
            f"{saved.document_id!r} does not match {current.document_id!r}."
        )
    if str(saved.source_hash) and str(current.source_hash):
        if str(saved.source_hash) != str(current.source_hash):
            raise ReferenceSweepIdentityMismatchError(
                "Saved source_hash does not match the current PDF/JSONL hash. "
                "Refusing to continue this sweep against a different document."
            )
    if saved.rag_config and current.rag_config:
        if dict(saved.rag_config) != dict(current.rag_config):
            raise ReferenceSweepIdentityMismatchError(
                "Saved RAG config fingerprint does not match the current run."
            )
    if saved.generation_config and current.generation_config:
        if not generation_fingerprints_compatible(
            dict(saved.generation_config), dict(current.generation_config)
        ):
            raise ReferenceSweepIdentityMismatchError(
                "Saved generation config fingerprint does not match the current run."
            )
    if saved.axis_interrogation_mode and current.axis_interrogation_mode:
        if str(saved.axis_interrogation_mode) != str(current.axis_interrogation_mode):
            raise ReferenceSweepIdentityMismatchError(
                "Saved AXIS interrogation mode "
                f"{saved.axis_interrogation_mode!r} does not match "
                f"{current.axis_interrogation_mode!r}."
            )
    if (
        saved.hybrid_add_pool_fingerprint
        and current.hybrid_add_pool_fingerprint
        and str(saved.hybrid_add_pool_fingerprint)
        != str(current.hybrid_add_pool_fingerprint)
    ):
        raise ReferenceSweepIdentityMismatchError(
            "Saved hybrid ADD source-pool fingerprint does not match the current pool."
        )


def _universe_summary(frozen_universes: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    questions = []
    total = 0
    for ref, rows in frozen_universes.items():
        items = list(rows or [])
        total += len(items)
        questions.append(
            {
                "candidate_ref": str(ref),
                "planned_count": len(items),
                "signatures": [
                    str(r.get("reference_intervention_signature") or "")
                    for r in items
                ],
            }
        )
    return {"planned_intervention_count": total, "questions": questions}


def save_reference_sweep_snapshot(
    directory: Path,
    *,
    identity: ReferenceSweepIdentity,
    checkpointer: Any,
    replay_executor: Any,
    factual_run_result: Any,
    axis_results: Mapping[str, Any],
    frozen_universes: Mapping[str, Sequence[Any]],
    intervention_registry: Any = None,
    intervention_alternatives: Optional[Mapping[str, Any]] = None,
    axis_noop_cache: Any = None,
    experiment: Any = None,
    cooperative_state: Any = None,
    experiment_payload: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist everything needed to resume the same factual sweep after restart."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = str(identity.run_id or getattr(factual_run_result, "run_id", "") or "")
    if not run_id:
        raise MissingReferenceSweepSnapshotError(
            "Cannot snapshot the reference sweep: factual run_id is empty."
        )
    to_restore = [
        factual_run_result,
        axis_results,
        frozen_universes,
        intervention_registry,
        intervention_alternatives,
        axis_noop_cache,
        experiment,
        cooperative_state,
    ]
    runtime: Optional[Dict[str, Any]] = None
    try:
        for obj in to_restore:
            prepare_object_for_pickle(obj)
        runtime = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "checkpointer_state": extract_checkpointer_state(checkpointer),
            "replay_executor_state": extract_replay_executor_state(replay_executor, run_id),
            "factual_run_result": _pickleable_factual_run_result(factual_run_result),
            "axis_results": dict(axis_results or {}),
            "frozen_universes": {
                str(k): list(v or []) for k, v in dict(frozen_universes or {}).items()
            },
            "intervention_registry": intervention_registry,
            "intervention_alternatives": dict(intervention_alternatives or {}),
            "axis_noop_cache": axis_noop_cache,
            "experiment": experiment,
            "cooperative_state": cooperative_state,
        }
        prepare_object_for_pickle(runtime)
        _require_pickleable(runtime, label="reference-sweep runtime payload")
        atomic_write_pickle(directory / RUNTIME_FILENAME, runtime)
    except ReferenceSweepSnapshotError:
        raise
    except Exception as exc:
        raise ReferenceSweepSnapshotError(
            "Cannot write the durable reference-sweep runtime snapshot: "
            f"{exc}"
        ) from exc
    finally:
        for obj in to_restore:
            restore_runtime_helpers(obj)
        if runtime is not None:
            restore_runtime_helpers(runtime)
    atomic_write_json(directory / IDENTITY_FILENAME, identity.to_dict())
    progress = ReferenceSweepProgress(
        run_id=run_id,
        status=STATUS_IN_PROGRESS,
        planned_intervention_count=int(identity.planned_intervention_count),
        completed_intervention_count=0,
        planned_question_count=int(identity.planned_question_count),
        completed_question_count=0,
        completed_keys=[],
    )
    atomic_write_json(directory / PROGRESS_FILENAME, progress.to_dict())
    atomic_write_json(directory / UNIVERSE_FILENAME, _universe_summary(runtime["frozen_universes"]))
    if experiment_payload is not None:
        atomic_write_json(directory / EXPERIMENT_FILENAME, dict(experiment_payload))
    return directory


def load_reference_sweep_snapshot(directory: Path) -> Dict[str, Any]:
    directory = Path(directory)
    missing = missing_required_files(directory)
    if missing:
        raise MissingReferenceSweepSnapshotError(
            missing_snapshot_message(directory, missing_files=missing)
        )
    identity = ReferenceSweepIdentity.from_dict(load_json(directory / IDENTITY_FILENAME))
    progress = ReferenceSweepProgress.from_dict(load_json(directory / PROGRESS_FILENAME))
    runtime = load_pickle(directory / RUNTIME_FILENAME)
    if not isinstance(runtime, dict):
        raise MissingReferenceSweepSnapshotError(
            f"Reference-sweep runtime pickle in {directory} is not a mapping."
        )
    restore_runtime_helpers(runtime)
    saved_run = str(runtime.get("run_id") or "")
    if not saved_run or saved_run != str(identity.run_id):
        raise ReferenceSweepIdentityMismatchError(
            "Runtime pickle run_id "
            f"{saved_run!r} does not match identity.json run_id {identity.run_id!r}."
        )
    experiment_payload = None
    exp_path = directory / EXPERIMENT_FILENAME
    if exp_path.is_file() and exp_path.stat().st_size > 0:
        try:
            experiment_payload = load_json(exp_path)
        except Exception:
            experiment_payload = None
    return {
        "identity": identity,
        "progress": progress,
        "runtime": runtime,
        "experiment_payload": experiment_payload,
        "directory": directory,
    }


def save_reference_sweep_progress(
    directory: Path,
    progress: ReferenceSweepProgress,
) -> Path:
    return atomic_write_json(Path(directory) / PROGRESS_FILENAME, progress.to_dict())


def load_reference_sweep_progress(directory: Path) -> Optional[ReferenceSweepProgress]:
    path = Path(directory) / PROGRESS_FILENAME
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    return ReferenceSweepProgress.from_dict(load_json(path))


def progress_from_rows(
    *,
    identity: ReferenceSweepIdentity,
    rows: Sequence[Mapping[str, Any]],
    frozen_universes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> ReferenceSweepProgress:
    complete_keys = []
    for row in rows:
        if not resume_row_is_complete(row):
            continue
        if str(row.get("lecturebank_id") or "") != str(identity.lecturebank_id):
            continue
        complete_keys.append(
            {
                "lecturebank_id": str(row.get("lecturebank_id") or ""),
                "candidate_ref": str(row.get("candidate_ref") or ""),
                "reference_intervention_signature": str(
                    row.get("reference_intervention_signature") or ""
                ),
            }
        )
    by_ref: Dict[str, int] = {}
    for key in complete_keys:
        by_ref[key["candidate_ref"]] = by_ref.get(key["candidate_ref"], 0) + 1
    planned_by_ref = {
        str(ref): len(list(items or []))
        for ref, items in dict(frozen_universes or {}).items()
    }
    completed_questions = 0
    for ref, planned in planned_by_ref.items():
        if planned > 0 and by_ref.get(ref, 0) >= planned:
            completed_questions += 1
    planned = int(identity.planned_intervention_count)
    completed = len(complete_keys)
    status = STATUS_COMPLETE if planned > 0 and completed >= planned else STATUS_IN_PROGRESS
    return ReferenceSweepProgress(
        run_id=str(identity.run_id),
        status=status,
        planned_intervention_count=planned,
        completed_intervention_count=completed,
        planned_question_count=int(identity.planned_question_count),
        completed_question_count=completed_questions,
        completed_keys=complete_keys,
    )


def wrap_persist_with_progress(
    *,
    directory: Path,
    identity: ReferenceSweepIdentity,
    frozen_universes: Mapping[str, Sequence[Mapping[str, Any]]],
    persist_rows_callback: Optional[Callable[[Sequence[Mapping[str, Any]]], None]],
    csv_path: Optional[Path] = None,
) -> Callable[[Sequence[Mapping[str, Any]]], None]:
    def _persist(rows: Sequence[Mapping[str, Any]]) -> None:
        if persist_rows_callback is not None:
            persist_rows_callback(rows)
        source: Sequence[Mapping[str, Any]] = rows
        if csv_path is not None:
            source = csv_rows_for_lecturebank(csv_path, identity.lecturebank_id)
        progress = progress_from_rows(
            identity=identity,
            rows=source,
            frozen_universes=frozen_universes,
        )
        save_reference_sweep_progress(directory, progress)

    return _persist


def restore_replay_runtime(
    *,
    checkpointer: Any,
    replay_executor: Any,
    runtime: Mapping[str, Any],
    workflow_runtime: Any = None,
) -> Any:
    """Restore MemorySaver storage, factual indexes, and the LLM replay ledger."""
    restore_checkpointer_state(checkpointer, runtime.get("checkpointer_state") or {})
    restore_replay_executor_state(replay_executor, runtime.get("replay_executor_state") or {})
    restore_runtime_helpers(runtime.get("factual_run_result"))
    restore_runtime_helpers(runtime.get("cooperative_state"))
    run_id = str(runtime.get("run_id") or "")
    if workflow_runtime is not None and run_id:
        replay_executor._runtimes[run_id] = workflow_runtime
        ensure = getattr(replay_executor, "_ensure_workflow_llm_wrapped", None)
        if callable(ensure):
            ensure(workflow_runtime)
        wrap_cc = getattr(replay_executor, "_ensure_cross_cove_verifier_llm", None)
        if callable(wrap_cc):
            try:
                workflow_runtime = wrap_cc(workflow_runtime)
                replay_executor._runtimes[run_id] = workflow_runtime
            except Exception:
                pass
    return runtime.get("factual_run_result")


def missing_snapshot_message(
    directory: Path,
    *,
    csv_path: Optional[Path] = None,
    csv_rows: int = 0,
    missing_files: Optional[Sequence[str]] = None,
) -> str:
    missing = list(missing_files or missing_required_files(directory))
    missing_txt = ", ".join(missing) if missing else "identity.json, progress.json, runtime.pkl"
    csv_note = ""
    if csv_path is not None:
        csv_note = (
            f" Found {int(csv_rows)} row(s) in {csv_path}. Those rows record "
            "completed interventions but not the factual LangGraph checkpoints, "
            "artifact registry, replay ledger, AXIS selections, or frozen "
            "intervention universe."
        )
    return (
        "Cannot resume the exhaustive AXIS reference sweep in "
        f"{directory.parent if directory.name == SNAPSHOT_DIRNAME else directory}. "
        f"Required durable snapshot files are missing ({missing_txt})."
        f"{csv_note} "
        "This evaluation stores factual checkpoints only in an in-process "
        "LangGraph MemorySaver. There is no SqliteSaver or other on-disk "
        "checkpoint store in this repository, so a Python exit or computer "
        "restart discards the original factual execution. Re-running MAS or "
        "AXIS in this folder would create a new factual run_id and must not "
        "be merged with the existing reference_interventions.csv. Start a new "
        "sweep in a fresh --output-dir, or resume only from a folder that "
        "contains presentation_<document_id>/reference_sweep/ produced by a "
        "resumable run."
    )


def csv_rows_for_lecturebank(csv_path: Path, lecturebank_id: str) -> List[Dict[str, Any]]:
    if csv_path is None:
        return []
    path = Path(csv_path)
    if not path.is_file() or path.stat().st_size <= 0:
        return []
    return [
        row
        for row in load_reference_intervention_rows(path)
        if str(row.get("lecturebank_id") or "") == str(lecturebank_id)
    ]


def resolve_reference_sweep_action(
    *,
    output_dir: Path,
    document_id: str,
    lecturebank_id: str,
    csv_path: Optional[Path],
    current_identity: Optional[ReferenceSweepIdentity] = None,
) -> str:
    """Return ``fresh``, ``resume``, or ``skip_complete``.

    Raises when leftover CSV/progress exists without a matching snapshot.
    Never returns a path that would start a new factual execution on top of
    another run's interventions.
    """
    directory = snapshot_dir(output_dir, document_id)
    rows = csv_rows_for_lecturebank(csv_path, lecturebank_id) if csv_path else []
    has_csv = bool(rows)
    missing = missing_required_files(directory) if directory.exists() else list(REQUIRED_SNAPSHOT_FILES)
    snapshot_ok = not missing
    if not snapshot_ok:
        if has_csv or (directory / PROGRESS_FILENAME).is_file() or (directory / IDENTITY_FILENAME).is_file():
            raise MissingReferenceSweepSnapshotError(
                missing_snapshot_message(
                    directory,
                    csv_path=csv_path,
                    csv_rows=len(rows),
                    missing_files=missing,
                )
            )
        return "fresh"
    loaded = load_reference_sweep_snapshot(directory)
    identity: ReferenceSweepIdentity = loaded["identity"]
    if current_identity is not None:
        current = current_identity
        if not current.run_id:
            current = ReferenceSweepIdentity(
                schema_version=current.schema_version,
                run_id=identity.run_id,
                lecturebank_id=current.lecturebank_id or identity.lecturebank_id,
                document_id=current.document_id or identity.document_id,
                source_hash=current.source_hash or identity.source_hash,
                rag_config=current.rag_config or identity.rag_config,
                generation_config=current.generation_config or identity.generation_config,
                axis_interrogation_mode=(
                    current.axis_interrogation_mode or identity.axis_interrogation_mode
                ),
                reference_max_interventions_per_question=(
                    current.reference_max_interventions_per_question
                    if current.reference_max_interventions_per_question is not None
                    else identity.reference_max_interventions_per_question
                ),
                planned_intervention_count=current.planned_intervention_count
                or identity.planned_intervention_count,
                planned_question_count=current.planned_question_count
                or identity.planned_question_count,
                candidate_refs=current.candidate_refs or identity.candidate_refs,
                hybrid_add_pool_fingerprint=(
                    current.hybrid_add_pool_fingerprint
                    or identity.hybrid_add_pool_fingerprint
                ),
            )
        validate_identity(identity, current)
    progress: ReferenceSweepProgress = loaded["progress"]
    planned = int(identity.planned_intervention_count or progress.planned_intervention_count or 0)
    completed = sum(1 for row in rows if resume_row_is_complete(row))
    if progress.status == STATUS_COMPLETE and planned > 0 and completed >= planned:
        return "skip_complete"
    return "resume"
