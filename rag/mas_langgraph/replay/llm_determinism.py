"""Deterministic workflow-LLM replay ledger.

Within one factual run and its NOOP/counterfactual replays:

    same exact semantic LLM request → same response

Explanation-layer LLM calls must use the unwrapped base model and must
not read or write this ledger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class WorkflowLLMReplayMissError(RuntimeError):
    """NOOP requested a workflow LLM fingerprint absent from the factual ledger."""

    def __init__(
        self,
        fingerprint: str,
        *,
        detail: str = "",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fingerprint = fingerprint
        self.diagnostics = dict(diagnostics or {})
        msg = (
            f"noop_llm_request_miss: fingerprint={fingerprint[:16]}…"
            + (f" ({detail})" if detail else "")
        )
        super().__init__(msg)


def _hash_text(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:16]


def describe_fingerprint_inputs(
    *,
    model: str,
    input_data: Any,
    temperature: Any = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    sample_index: Optional[int] = None,
    llm_component: str = "",
    llm_layer: str = "",
) -> Dict[str, Any]:
    """Field-level fingerprint inputs for mismatch diagnostics (hashes, not full prompts)."""
    serialized = serialize_llm_input(input_data)
    visible = serialized
    if isinstance(serialized, dict) and "content" in serialized:
        visible = serialized.get("content")
    prompt_text = visible if isinstance(visible, str) else _canonical_json(visible)
    return {
        "model": str(model or ""),
        "temperature": temperature,
        "model_kwargs": dict(model_kwargs or {}),
        "llm_layer": str(llm_layer or ""),
        "llm_component": str(llm_component or ""),
        "sample_index": sample_index,
        "input_kind": type(input_data).__name__,
        "visible_prompt_hash": _hash_text(prompt_text),
        "visible_prompt_chars": len(prompt_text),
        "serialized_input_hash": _hash_text(serialized),
        "has_logical_sample_identity": sample_index is not None,
    }


def build_fingerprint_payload(
    *,
    model: str,
    input_data: Any,
    temperature: Any = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    sample_index: Optional[int] = None,
    llm_component: str = "",
) -> Dict[str, Any]:
    """Canonical payload hashed by ``fingerprint_workflow_llm_request``."""
    payload: Dict[str, Any] = {
        "model": str(model or ""),
        "input": serialize_llm_input(input_data),
        "temperature": temperature,
        "other_relevant_model_kwargs": dict(model_kwargs or {}),
    }
    if sample_index is not None:
        payload["logical_sample_identity"] = {
            "llm_component": str(llm_component or ""),
            "sample_index": int(sample_index),
        }
    return payload


def diff_fingerprint_input_fields(
    factual: Dict[str, Any],
    other: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return fields whose diagnostic values differ."""
    keys = sorted(set(factual) | set(other))
    changed: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        left = factual.get(key)
        right = other.get(key)
        if left != right:
            changed[key] = {"factual": left, "noop": right}
    return changed


def format_fingerprint_mismatch_report(
    *,
    operation: str,
    factual_fingerprint: str,
    noop_fingerprint: str,
    changed_fields: Dict[str, Dict[str, Any]],
) -> str:
    lines = [
        "[LLM REPLAY FINGERPRINT MISMATCH]",
        f"operation={operation or 'unknown'}",
        f"factual_fingerprint={factual_fingerprint}",
        f"noop_fingerprint={noop_fingerprint}",
        "",
        "changed_fields:",
    ]
    if not changed_fields:
        lines.append("    (no describe_fingerprint_inputs fields differed;")
        lines.append("     prompts may still differ at full serialized fidelity)")
    else:
        for key, pair in changed_fields.items():
            lines.append(f"    {key}:")
            lines.append(f"        factual={pair.get('factual')!r}")
            lines.append(f"        noop={pair.get('noop')!r}")
    return "\n".join(lines)


_branch_type: ContextVar[str] = ContextVar("workflow_llm_branch_type", default="")
_run_id: ContextVar[str] = ContextVar("workflow_llm_run_id", default="")
_ledger: ContextVar[Optional["WorkflowLLMReplayLedger"]] = ContextVar(
    "workflow_llm_ledger", default=None
)
_enabled: ContextVar[bool] = ContextVar("workflow_llm_enabled", default=False)
_llm_layer: ContextVar[str] = ContextVar("workflow_llm_layer", default="")
_llm_component: ContextVar[str] = ContextVar("workflow_llm_component", default="")
_sample_index: ContextVar[Optional[int]] = ContextVar(
    "workflow_llm_sample_index", default=None
)


def current_branch_type() -> str:
    return str(_branch_type.get() or "")


def current_ledger() -> Optional["WorkflowLLMReplayLedger"]:
    return _ledger.get()


def current_llm_layer() -> str:
    return str(_llm_layer.get() or "")


def current_llm_component() -> str:
    return str(_llm_component.get() or "")


def current_sample_index() -> Optional[int]:
    return _sample_index.get()


def workflow_llm_context_active() -> bool:
    return bool(_enabled.get()) and current_ledger() is not None


@contextmanager
def workflow_llm_context(
    *,
    layer: str = "",
    component: str = "",
    sample_index: Optional[int] = None,
) -> Iterator[None]:
    """Attribute nested workflow LLM call(s) to a stable layer/component.

    ``sample_index`` distinguishes intentional repeated samples (Cross-CoVe)
    that share identical prompt text.
    """
    t_layer = _llm_layer.set(str(layer or ""))
    t_comp = _llm_component.set(str(component or ""))
    t_sample = _sample_index.set(sample_index)
    try:
        yield
    finally:
        _llm_layer.reset(t_layer)
        _llm_component.reset(t_comp)
        _sample_index.reset(t_sample)


@contextmanager
def workflow_llm_execution_context(
    *,
    run_id: str,
    branch_id: str,
    branch_type: str,
    ledger: "WorkflowLLMReplayLedger",
) -> Iterator[None]:
    """Scope ledger + branch identity around compiled_graph.invoke(...)."""
    del branch_id  # reserved for diagnostics; fingerprint must ignore branch
    t_bt = _branch_type.set(str(branch_type or ""))
    t_rid = _run_id.set(str(run_id or ""))
    t_led = _ledger.set(ledger)
    t_en = _enabled.set(True)
    try:
        yield
    finally:
        _branch_type.reset(t_bt)
        _run_id.reset(t_rid)
        _ledger.reset(t_led)
        _enabled.reset(t_en)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def serialize_llm_input(input_data: Any) -> Any:
    """Best-effort structural serialization of LangChain-style inputs."""
    if input_data is None:
        return None
    if isinstance(input_data, (str, int, float, bool)):
        return input_data
    if isinstance(input_data, dict):
        return {str(k): serialize_llm_input(v) for k, v in input_data.items()}
    if isinstance(input_data, (list, tuple)):
        return [serialize_llm_input(x) for x in input_data]
    role = getattr(input_data, "type", None) or getattr(input_data, "role", None)
    content = getattr(input_data, "content", None)
    if role is not None or content is not None:
        return {
            "role": str(role) if role is not None else None,
            "content": serialize_llm_input(content),
        }
    return str(input_data)


def fingerprint_workflow_llm_request(
    *,
    model: str,
    input_data: Any,
    temperature: Any = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    sample_index: Optional[int] = None,
    llm_component: str = "",
) -> str:
    # Intentional repeated samples (Cross-CoVe) share prompt text but must be
    # independent logical calls. Ordinary calls omit logical_sample_identity so
    # fingerprints remain unchanged.
    payload = build_fingerprint_payload(
        model=model,
        input_data=input_data,
        temperature=temperature,
        model_kwargs=model_kwargs,
        sample_index=sample_index,
        llm_component=llm_component,
    )
    raw = _canonical_json(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class WorkflowLLMRecord:
    request_fingerprint: str
    response: Any
    origin_branch_type: str
    provider_called: bool
    llm_layer: str = ""
    llm_component: str = ""
    sample_index: Optional[int] = None
    memo_hit: bool = False
    logical_call_index: int = 0
    fingerprint_inputs: Dict[str, Any] = field(default_factory=dict)


def summarize_workflow_llm_calls(ledger: "WorkflowLLMReplayLedger") -> Dict[str, Any]:
    """Group logical/provider/memo counts by branch → layer → component."""
    summary: Dict[str, Any] = {}
    for entry in ledger.call_log:
        branch = str(entry.get("branch_type") or "factual")
        layer = str(entry.get("llm_layer") or "unattributed")
        component = str(entry.get("llm_component") or "unattributed")
        branch_bucket = summary.setdefault(branch, {})
        layer_bucket = branch_bucket.setdefault(layer, {})
        comp = layer_bucket.setdefault(
            component,
            {"logical": 0, "provider": 0, "memo_hits": 0},
        )
        comp["logical"] += 1
        if entry.get("provider_called"):
            comp["provider"] += 1
        if entry.get("memo_hit"):
            comp["memo_hits"] += 1
    return summary


@dataclass
class WorkflowLLMReplayLedger:
    """Run-scoped deterministic map: request fingerprint → response."""

    run_id: str = ""
    model_name: str = ""
    temperature: Any = None

    _responses: Dict[str, WorkflowLLMRecord] = field(default_factory=dict)
    call_log: List[Dict[str, Any]] = field(default_factory=list)

    factual_llm_request_count: int = 0
    factual_llm_memo_hits: int = 0
    noop_llm_request_count: int = 0
    noop_factual_response_hits: int = 0
    noop_llm_request_misses: int = 0
    noop_live_provider_calls: int = 0
    counterfactual_llm_request_count: int = 0
    counterfactual_factual_response_hits: int = 0
    counterfactual_novel_request_count: int = 0
    counterfactual_memo_hits: int = 0
    counterfactual_live_provider_calls: int = 0

    _branch_snap: Optional[Dict[str, int]] = field(default=None, repr=False)
    last_branch_diagnostics: Dict[str, Any] = field(default_factory=dict)
    last_resolve_meta: Dict[str, Any] = field(default_factory=dict)
    cross_cove_verifier_model: str = ""

    def _counter_snapshot(self) -> Dict[str, int]:
        return {
            "factual_llm_request_count": self.factual_llm_request_count,
            "factual_llm_memo_hits": self.factual_llm_memo_hits,
            "noop_llm_request_count": self.noop_llm_request_count,
            "noop_factual_response_hits": self.noop_factual_response_hits,
            "noop_llm_request_misses": self.noop_llm_request_misses,
            "noop_live_provider_calls": self.noop_live_provider_calls,
            "counterfactual_llm_request_count": self.counterfactual_llm_request_count,
            "counterfactual_factual_response_hits": (
                self.counterfactual_factual_response_hits
            ),
            "counterfactual_novel_request_count": (
                self.counterfactual_novel_request_count
            ),
            "counterfactual_memo_hits": self.counterfactual_memo_hits,
            "counterfactual_live_provider_calls": (
                self.counterfactual_live_provider_calls
            ),
        }

    def begin_branch(self, branch_type: str) -> None:
        del branch_type
        self._branch_snap = self._counter_snapshot()

    def end_branch(self, branch_type: str) -> Dict[str, Any]:
        before = self._branch_snap or self._counter_snapshot()
        after = self._counter_snapshot()
        bt = str(branch_type or "").lower()
        if bt == "noop":
            diag = {
                "factual_response_hits": after["noop_factual_response_hits"]
                - before["noop_factual_response_hits"],
                "novel_request_count": 0,
                "memo_hits": 0,
                "live_provider_calls": after["noop_live_provider_calls"]
                - before["noop_live_provider_calls"],
                "request_misses": after["noop_llm_request_misses"]
                - before["noop_llm_request_misses"],
                "request_count": after["noop_llm_request_count"]
                - before["noop_llm_request_count"],
            }
        elif bt == "counterfactual":
            diag = {
                "factual_response_hits": after["counterfactual_factual_response_hits"]
                - before["counterfactual_factual_response_hits"],
                "novel_request_count": after["counterfactual_novel_request_count"]
                - before["counterfactual_novel_request_count"],
                "memo_hits": after["counterfactual_memo_hits"]
                - before["counterfactual_memo_hits"],
                "live_provider_calls": after["counterfactual_live_provider_calls"]
                - before["counterfactual_live_provider_calls"],
                "request_count": after["counterfactual_llm_request_count"]
                - before["counterfactual_llm_request_count"],
            }
        else:
            diag = {
                "factual_response_hits": 0,
                "novel_request_count": after["factual_llm_request_count"]
                - before["factual_llm_request_count"]
                - (after["factual_llm_memo_hits"] - before["factual_llm_memo_hits"]),
                "memo_hits": after["factual_llm_memo_hits"]
                - before["factual_llm_memo_hits"],
                "live_provider_calls": (
                    after["factual_llm_request_count"]
                    - before["factual_llm_request_count"]
                    - (after["factual_llm_memo_hits"] - before["factual_llm_memo_hits"])
                ),
                "request_count": after["factual_llm_request_count"]
                - before["factual_llm_request_count"],
            }
        self.last_branch_diagnostics = diag
        self._branch_snap = None
        return diag

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "workflow_llm_determinism_enabled": True,
            "workflow_llm_model": self.model_name,
            "workflow_llm_temperature": self.temperature,
            "cross_cove_verifier_model": self.cross_cove_verifier_model,
            "factual_llm_request_count": self.factual_llm_request_count,
            "factual_llm_unique_request_count": sum(
                1
                for r in self._responses.values()
                if r.origin_branch_type == "factual"
            ),
            "factual_llm_memo_hits": self.factual_llm_memo_hits,
            "noop_llm_request_count": self.noop_llm_request_count,
            "noop_factual_response_hits": self.noop_factual_response_hits,
            "noop_llm_request_misses": self.noop_llm_request_misses,
            "noop_live_provider_calls": self.noop_live_provider_calls,
            "counterfactual_llm_request_count": self.counterfactual_llm_request_count,
            "counterfactual_factual_response_hits": (
                self.counterfactual_factual_response_hits
            ),
            "counterfactual_novel_request_count": (
                self.counterfactual_novel_request_count
            ),
            "counterfactual_memo_hits": self.counterfactual_memo_hits,
            "counterfactual_live_provider_calls": (
                self.counterfactual_live_provider_calls
            ),
            "workflow_llm_calls": summarize_workflow_llm_calls(self),
        }

    def _log_call(
        self,
        *,
        fingerprint: str,
        branch_type: str,
        provider_called: bool,
        memo_hit: bool,
        llm_layer: str,
        llm_component: str,
        sample_index: Optional[int],
    ) -> int:
        idx = len(self.call_log)
        self.call_log.append(
            {
                "logical_call_index": idx,
                "branch_type": branch_type,
                "llm_layer": llm_layer,
                "llm_component": llm_component,
                "sample_index": sample_index,
                "request_fingerprint": fingerprint,
                "provider_called": provider_called,
                "memo_hit": memo_hit,
            }
        )
        return idx

    def _nearest_factual_record(
        self,
        *,
        llm_component: str,
        fingerprint_inputs: Optional[Dict[str, Any]],
    ) -> Optional[WorkflowLLMRecord]:
        factual = [
            r
            for r in self._responses.values()
            if r.origin_branch_type == "factual"
        ]
        if not factual:
            return None
        same_comp = [
            r for r in factual if (r.llm_component or "") == (llm_component or "")
        ]
        pool = same_comp or factual
        if not fingerprint_inputs:
            return pool[0]
        best = None
        best_score = -1
        for rec in pool:
            rec_inputs = rec.fingerprint_inputs or {}
            score = sum(
                1
                for key in (
                    "visible_prompt_hash",
                    "serialized_input_hash",
                    "model",
                    "temperature",
                    "sample_index",
                    "llm_component",
                )
                if rec_inputs.get(key) == fingerprint_inputs.get(key)
            )
            if score > best_score:
                best_score = score
                best = rec
        return best

    def _diagnose_noop_miss(
        self,
        *,
        fingerprint: str,
        llm_layer: str,
        llm_component: str,
        fingerprint_inputs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        nearest = self._nearest_factual_record(
            llm_component=llm_component,
            fingerprint_inputs=fingerprint_inputs,
        )
        diag: Dict[str, Any] = {
            "operation": llm_component or llm_layer or "unknown",
            "noop_fingerprint": fingerprint,
            "noop_inputs": dict(fingerprint_inputs or {}),
            "factual_fingerprint": None,
            "changed_fields": {},
        }
        if nearest is not None:
            diag["factual_fingerprint"] = nearest.request_fingerprint
            diag["changed_fields"] = diff_fingerprint_input_fields(
                nearest.fingerprint_inputs or {},
                fingerprint_inputs or {},
            )
            report = format_fingerprint_mismatch_report(
                operation=str(diag["operation"]),
                factual_fingerprint=str(nearest.request_fingerprint),
                noop_fingerprint=fingerprint,
                changed_fields=diag["changed_fields"],
            )
            diag["report"] = report
            logger.warning(report)
        else:
            logger.warning(
                "[LLM REPLAY FINGERPRINT MISMATCH]\n"
                "operation=%s\n"
                "factual_fingerprint=(none in ledger)\n"
                "noop_fingerprint=%s\n"
                "noop_inputs=%s",
                diag["operation"],
                fingerprint,
                diag["noop_inputs"],
            )
        return diag

    def _record_lookup_timing(self, duration_s: float) -> None:
        try:
            from ..experiment.timing import add_replay_span

            add_replay_span("ledger_lookup", float(max(0.0, duration_s)))
        except Exception:
            return

    def _set_resolve_meta(
        self,
        *,
        provider_called: bool,
        memo_hit: bool,
        factual_hit: bool = False,
        model: str = "",
        component: str = "",
    ) -> None:
        self.last_resolve_meta = {
            "provider_called": bool(provider_called),
            "memo_hit": bool(memo_hit),
            "factual_hit": bool(factual_hit),
            "model": str(model or ""),
            "component": str(component or ""),
        }

    def resolve(
        self,
        *,
        fingerprint: str,
        branch_type: str,
        call_provider,
        llm_layer: str = "",
        llm_component: str = "",
        sample_index: Optional[int] = None,
        fingerprint_inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        bt = str(branch_type or "factual").lower()
        layer = str(llm_layer or "")
        component = str(llm_component or "")
        inputs = dict(fingerprint_inputs or {})
        t_lookup = time.perf_counter()

        if bt in {"noop", "no_op", "control"}:
            self.noop_llm_request_count += 1
            rec = self._responses.get(fingerprint)
            lookup_s = time.perf_counter() - t_lookup
            self._record_lookup_timing(lookup_s)
            if rec is None:
                self.noop_llm_request_misses += 1
                self._log_call(
                    fingerprint=fingerprint,
                    branch_type="noop",
                    provider_called=False,
                    memo_hit=False,
                    llm_layer=layer,
                    llm_component=component,
                    sample_index=sample_index,
                )
                diag = self._diagnose_noop_miss(
                    fingerprint=fingerprint,
                    llm_layer=layer,
                    llm_component=component,
                    fingerprint_inputs=inputs,
                )
                self._set_resolve_meta(
                    provider_called=False,
                    memo_hit=False,
                    factual_hit=False,
                    model=str(inputs.get("model") or ""),
                    component=component,
                )
                raise WorkflowLLMReplayMissError(fingerprint, diagnostics=diag)
            self.noop_factual_response_hits += 1
            self._log_call(
                fingerprint=fingerprint,
                branch_type="noop",
                provider_called=False,
                memo_hit=True,
                llm_layer=layer or rec.llm_layer,
                llm_component=component or rec.llm_component,
                sample_index=sample_index
                if sample_index is not None
                else rec.sample_index,
            )
            self._set_resolve_meta(
                provider_called=False,
                memo_hit=True,
                factual_hit=True,
                model=str(inputs.get("model") or ""),
                component=component,
            )
            return deepcopy(rec.response)

        if bt in {"counterfactual", "cf"}:
            self.counterfactual_llm_request_count += 1
            rec = self._responses.get(fingerprint)
            lookup_s = time.perf_counter() - t_lookup
            self._record_lookup_timing(lookup_s)
            if rec is not None:
                factual_hit = rec.origin_branch_type == "factual"
                if factual_hit:
                    self.counterfactual_factual_response_hits += 1
                else:
                    self.counterfactual_memo_hits += 1
                self._log_call(
                    fingerprint=fingerprint,
                    branch_type="counterfactual",
                    provider_called=False,
                    memo_hit=True,
                    llm_layer=layer or rec.llm_layer,
                    llm_component=component or rec.llm_component,
                    sample_index=sample_index
                    if sample_index is not None
                    else rec.sample_index,
                )
                self._set_resolve_meta(
                    provider_called=False,
                    memo_hit=True,
                    factual_hit=factual_hit,
                    model=str(inputs.get("model") or ""),
                    component=component,
                )
                return deepcopy(rec.response)
            self.counterfactual_novel_request_count += 1
            self.counterfactual_live_provider_calls += 1
            response = call_provider()
            call_idx = self._log_call(
                fingerprint=fingerprint,
                branch_type="counterfactual",
                provider_called=True,
                memo_hit=False,
                llm_layer=layer,
                llm_component=component,
                sample_index=sample_index,
            )
            self._responses[fingerprint] = WorkflowLLMRecord(
                request_fingerprint=fingerprint,
                response=deepcopy(response),
                origin_branch_type="counterfactual",
                provider_called=True,
                llm_layer=layer,
                llm_component=component,
                sample_index=sample_index,
                memo_hit=False,
                logical_call_index=call_idx,
                fingerprint_inputs=inputs,
            )
            self._set_resolve_meta(
                provider_called=True,
                memo_hit=False,
                factual_hit=False,
                model=str(inputs.get("model") or ""),
                component=component,
            )
            return response

        # Factual (default)
        self.factual_llm_request_count += 1
        rec = self._responses.get(fingerprint)
        lookup_s = time.perf_counter() - t_lookup
        self._record_lookup_timing(lookup_s)
        if rec is not None:
            self.factual_llm_memo_hits += 1
            self._log_call(
                fingerprint=fingerprint,
                branch_type="factual",
                provider_called=False,
                memo_hit=True,
                llm_layer=layer or rec.llm_layer,
                llm_component=component or rec.llm_component,
                sample_index=sample_index
                if sample_index is not None
                else rec.sample_index,
            )
            self._set_resolve_meta(
                provider_called=False,
                memo_hit=True,
                factual_hit=True,
                model=str(inputs.get("model") or ""),
                component=component,
            )
            return deepcopy(rec.response)
        response = call_provider()
        call_idx = self._log_call(
            fingerprint=fingerprint,
            branch_type="factual",
            provider_called=True,
            memo_hit=False,
            llm_layer=layer,
            llm_component=component,
            sample_index=sample_index,
        )
        self._responses[fingerprint] = WorkflowLLMRecord(
            request_fingerprint=fingerprint,
            response=deepcopy(response),
            origin_branch_type="factual",
            provider_called=True,
            llm_layer=layer,
            llm_component=component,
            sample_index=sample_index,
            memo_hit=False,
            logical_call_index=call_idx,
            fingerprint_inputs=inputs,
        )
        self._set_resolve_meta(
            provider_called=True,
            memo_hit=False,
            factual_hit=False,
            model=str(inputs.get("model") or ""),
            component=component,
        )
        return response


class ReplayAwareWorkflowLLM:
    """LangChain-compatible wrapper that memoizes workflow LLM responses."""

    def __init__(self, base_llm: Any, *, ledger: Optional[WorkflowLLMReplayLedger] = None):
        self.base_llm = base_llm
        self._static_ledger = ledger
        self.model_name = (
            getattr(base_llm, "model_name", None)
            or getattr(base_llm, "model", None)
            or type(base_llm).__name__
        )
        self.temperature = getattr(base_llm, "temperature", None)

    def __getattr__(self, item: str) -> Any:
        if item in {"base_llm", "_static_ledger"}:
            raise AttributeError(item)
        return getattr(self.base_llm, item)

    def _active_ledger(self) -> Optional[WorkflowLLMReplayLedger]:
        if workflow_llm_context_active():
            return current_ledger()
        return self._static_ledger

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        ledger = self._active_ledger()
        if ledger is None or not workflow_llm_context_active():
            if config is None:
                return self.base_llm.invoke(input, **kwargs)
            return self.base_llm.invoke(input, config=config, **kwargs)

        model = str(
            getattr(self.base_llm, "model_name", None)
            or getattr(self.base_llm, "model", None)
            or self.model_name
            or ""
        )
        temperature = getattr(self.base_llm, "temperature", self.temperature)
        layer = current_llm_layer()
        component = current_llm_component()
        sample_index = current_sample_index()
        model_kwargs = {k: kwargs[k] for k in sorted(kwargs.keys())}
        fp = fingerprint_workflow_llm_request(
            model=model,
            input_data=input,
            temperature=temperature,
            model_kwargs=model_kwargs,
            sample_index=sample_index,
            llm_component=component,
        )
        fp_inputs = describe_fingerprint_inputs(
            model=model,
            input_data=input,
            temperature=temperature,
            model_kwargs=model_kwargs,
            sample_index=sample_index,
            llm_component=component,
            llm_layer=layer,
        )
        branch = current_branch_type() or "factual"

        try:
            from ..experiment.timing import current_timing as _current_timing
        except Exception:  # pragma: no cover - optional instrumentation
            _current_timing = lambda: None  # type: ignore[assignment]

        timing = _current_timing()
        if timing is not None:
            timing.record_workflow_logical_call(
                component=component or "unknown",
                model=model,
            )

        def _call():
            t0 = time.perf_counter()
            try:
                if config is None:
                    return self.base_llm.invoke(input, **kwargs)
                return self.base_llm.invoke(input, config=config, **kwargs)
            finally:
                if timing is not None:
                    timing.record_workflow_provider_call(
                        component=component or "unknown",
                        duration_s=time.perf_counter() - t0,
                        model=model,
                    )

        response = ledger.resolve(
            fingerprint=fp,
            branch_type=branch,
            call_provider=_call,
            llm_layer=layer,
            llm_component=component,
            sample_index=sample_index,
            fingerprint_inputs=fp_inputs,
        )
        if timing is not None:
            meta = dict(getattr(ledger, "last_resolve_meta", None) or {})
            if meta.get("memo_hit") or meta.get("factual_hit"):
                timing.record_workflow_ledger_hit(
                    component=component or "unknown",
                    model=model or str(meta.get("model") or ""),
                    factual_hit=bool(meta.get("factual_hit")),
                    cf_memo_hit=bool(meta.get("memo_hit")) and not bool(meta.get("factual_hit")),
                )
        return response


def unwrap_base_llm(llm: Any) -> Any:
    """Return the underlying provider LLM if ``llm`` is replay-aware."""
    if isinstance(llm, ReplayAwareWorkflowLLM):
        return llm.base_llm
    base = getattr(llm, "base_llm", None)
    if base is not None and llm is not base:
        return base
    return llm


def install_replay_aware_llm(rag_agents: Any, *, temperature: Any = 0) -> ReplayAwareWorkflowLLM:
    """Wrap ``rag_agents.llm`` once; preserve ``base_llm`` for explanation use.

    ``temperature`` is recorded on the ledger for diagnostics. Prefer configuring
    experimental workflow temperature via ``RAGConfig`` at construction time so
    the explanation base model is not silently mutated after the fact.
    """
    current = getattr(rag_agents, "llm", None)
    if current is None:
        raise ValueError("rag_agents.llm is None")
    if isinstance(current, ReplayAwareWorkflowLLM):
        return current
    del temperature
    wrapped = ReplayAwareWorkflowLLM(current)
    rag_agents.llm = wrapped
    rag_agents.base_llm = current
    rag_agents.explanation_llm = current
    rag_agents.workflow_llm = wrapped
    return wrapped
