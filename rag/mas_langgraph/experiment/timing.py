"""High-resolution wall-clock timing for explanation experiments (instrumentation only).

Uses ``time.perf_counter()``. Does not change workflow or AXIS methodology.
"""

from __future__ import annotations

import csv
import statistics
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

__all__ = [
    "AXIS_EXPLANATION_LLM_PHASES",
    "REPLAY_NESTED_SPANS",
    "REPLAY_TOP_LEVEL_SPANS",
    "ExperimentTimingCollector",
    "add_replay_span",
    "assemble_replay_profile",
    "current_replay_profile",
    "current_timing",
    "explanation_llm_phase",
    "replay_profile_scope",
    "set_replay_profile_meta",
    "set_timing_target",
    "set_timing_round",
    "timed_replay_span",
    "timing_scope",
]


_collector: ContextVar[Optional["ExperimentTimingCollector"]] = ContextVar(
    "emaes_timing_collector", default=None
)
_phase: ContextVar[str] = ContextVar("emaes_timing_phase", default="")
_target: ContextVar[str] = ContextVar("emaes_timing_target", default="")
_round: ContextVar[Optional[int]] = ContextVar("emaes_timing_round", default=None)
_replay_profile: ContextVar[Optional[Dict[str, float]]] = ContextVar(
    "emaes_replay_profile", default=None
)

# AXIS-condition explanation LLM phases (inside axis_total_s).
# Trace / Trace+Provenance participant synthesis is recorded separately and
# must not be counted as AXIS explanation LLM time.
AXIS_EXPLANATION_LLM_PHASES = frozenset(
    {
        "interrogation",
        "intermediate_synthesis",
        "final_synthesis",
        "axis_participant_synthesis",
    }
)

# Mutually exclusive wall-clock phases along one counterfactual replay.
# Nested diagnostic spans (provider wait, ledger lookup, ...) live inside these
# and must not be added into the same total.
REPLAY_TOP_LEVEL_SPANS = (
    "replay_setup",
    "checkpoint_restore",
    "intervention_prepare",
    "intervention_application",
    "fork_creation",
    "fork_resolver",
    "graph_execution",
    # branch_history_fetch: time obtaining branch-local snapshots (parent walk).
    # history_snapshots_scanned: snapshots inspected for the current branch.
    "branch_history_fetch",
    "branch_history_filter",
    "branch_metadata_update",
    "final_state_store_copy",
    "artifact_resolver_rebuild",
    "origin_checkpoint_lookup",
    "workflow_result_materialization",
)

REPLAY_NESTED_SPANS = (
    "provider_wait",
    "ledger_lookup",
    "observability_check",
    "artifact_registration",
    "provenance_recording",
    "final_observation",
)


def _span_seconds(spans: Dict[str, Any], name: str) -> float:
    raw = spans.get(name)
    if raw is None:
        raw = spans.get(f"{name}_s")
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def assemble_replay_profile(
    *,
    spans: Optional[Dict[str, Any]] = None,
    total_s: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a replay profile with non-overlapping top-level phases.

    Nested spans are diagnostic only and are excluded from
    ``sum(top_level) ≈ total_replay_s``.
    """
    merged: Dict[str, Any] = {}
    merged.update(dict(spans or {}))
    merged.update(dict(extra or {}))
    top_level: Dict[str, float] = {}
    for name in REPLAY_TOP_LEVEL_SPANS:
        top_level[f"{name}_s"] = max(0.0, _span_seconds(merged, name))
    nested: Dict[str, float] = {}
    for name in REPLAY_NESTED_SPANS:
        nested[f"{name}_s"] = max(0.0, _span_seconds(merged, name))
    top_sum = sum(top_level.values())
    other_s = max(0.0, float(total_s) - top_sum)
    top_level["other_s"] = other_s

    def _meta_int(key: str) -> int:
        try:
            return int(merged.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    scanned = _meta_int("history_snapshots_scanned")
    retained = _meta_int("branch_snapshots_retained")
    # 0 means unknown / not scanned. Do not substitute scanned snapshots:
    # parent-lineage collection no longer enumerates the full thread.
    thread_size = _meta_int("total_thread_history_size")
    mode = str(merged.get("history_collection_mode") or "").strip()
    profile: Dict[str, Any] = {
        "total_replay_s": float(total_s),
        "top_level": dict(top_level),
        "nested": dict(nested),
        "other_s": other_s,
        "history_snapshots_scanned": scanned,
        "branch_snapshots_retained": retained,
        "total_thread_history_size": thread_size,
        "history_collection_mode": mode,
        "top_level_sum_s": top_sum + other_s,
    }
    # Backward-compatible flat fields (nested values are inclusive/diagnostic).
    for key, value in top_level.items():
        profile[key] = value
    for key, value in nested.items():
        profile[key] = value
    return profile


def current_replay_profile() -> Optional[Dict[str, float]]:
    return _replay_profile.get()


def add_replay_span(name: str, duration_s: float) -> None:
    profile = _replay_profile.get()
    if profile is None:
        return
    key = str(name or "")
    if not key:
        return
    profile[key] = float(profile.get(key) or 0.0) + float(max(0.0, duration_s))


def set_replay_profile_meta(key: str, value: Any) -> None:
    """Record non-duration replay diagnostics on the active profile."""
    profile = _replay_profile.get()
    if profile is None:
        return
    name = str(key or "")
    if not name:
        return
    profile[name] = value


@contextmanager
def replay_profile_scope() -> Iterator[Dict[str, float]]:
    profile: Dict[str, float] = {}
    token = _replay_profile.set(profile)
    try:
        yield profile
    finally:
        _replay_profile.reset(token)


@contextmanager
def timed_replay_span(name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        add_replay_span(name, time.perf_counter() - t0)


def current_timing() -> Optional["ExperimentTimingCollector"]:
    return _collector.get()


def set_timing_target(target: str) -> None:
    _target.set(str(target or ""))


def set_timing_round(round_index: Optional[int]) -> None:
    _round.set(round_index)


@contextmanager
def explanation_llm_phase(phase: str) -> Iterator[None]:
    token = _phase.set(str(phase or ""))
    try:
        yield
    finally:
        _phase.reset(token)


@contextmanager
def timing_scope(collector: "ExperimentTimingCollector") -> Iterator["ExperimentTimingCollector"]:
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)


def _now() -> float:
    return time.perf_counter()


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(part) / float(total)


def _extract_usage(response: Any) -> Dict[str, Optional[int]]:
    """Best-effort token usage from an already-returned provider response."""
    meta = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", None)
    if not isinstance(meta, dict):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    usage = meta.get("token_usage") if "token_usage" in meta else meta
    if not isinstance(usage, dict):
        usage = meta
    inp = usage.get("input_tokens", usage.get("prompt_tokens"))
    out = usage.get("output_tokens", usage.get("completion_tokens"))
    tot = usage.get("total_tokens")
    if tot is None and inp is not None and out is not None:
        try:
            tot = int(inp) + int(out)
        except Exception:
            tot = None
    def _i(v: Any) -> Optional[int]:
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    return {
        "input_tokens": _i(inp),
        "output_tokens": _i(out),
        "total_tokens": _i(tot),
    }


@dataclass
class ExperimentTimingCollector:
    """Accumulates experiment / AXIS / replay / explanation-LLM timings."""

    run_id: str = ""
    experiment: Dict[str, float] = field(default_factory=dict)
    conditions: Dict[str, float] = field(default_factory=dict)
    axis_totals: Dict[str, float] = field(default_factory=dict)
    workflow_llm: Dict[str, Any] = field(default_factory=dict)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    explanation_llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    replay_events: List[Dict[str, Any]] = field(default_factory=list)
    provider_by_component: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    noop_cache_profiles: List[Dict[str, Any]] = field(default_factory=list)
    replay_profiles: List[Dict[str, Any]] = field(default_factory=list)
    node_profiles: List[Dict[str, Any]] = field(default_factory=list)

    _case: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _round: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _exp_t0: Optional[float] = field(default=None, repr=False)
    _marks: Dict[str, float] = field(default_factory=dict, repr=False)

    # -- lifecycle ---------------------------------------------------------

    def start_experiment(self) -> None:
        self._exp_t0 = _now()

    def end_experiment(self) -> None:
        if self._exp_t0 is not None:
            self.experiment["experiment_total_s"] = _now() - self._exp_t0
        # Directly measured aggregates only (no estimates).
        expl = sum(float(c.get("duration_s") or 0.0) for c in self.explanation_llm_calls)
        replay = sum(float(e.get("duration_s") or 0.0) for e in self.replay_events)
        provider = float(self.workflow_llm.get("provider_time_s") or 0.0)
        self.experiment["explanation_layer_llm_time_s"] = expl
        self.experiment["replay_execution_time_s"] = replay
        self.experiment["workflow_provider_time_s"] = provider
        axis_total = float(self.axis_totals.get("axis_total_s") or 0.0)
        axis_expl = float(self.axis_totals.get("axis_explanation_llm_s") or 0.0)
        if axis_expl <= 0:
            axis_expl = float(self.axis_totals.get("explanation_llm_s") or 0.0)
        axis_replay = float(self.axis_totals.get("axis_replay_s") or 0.0)
        if axis_replay <= 0:
            axis_replay = float(self.axis_totals.get("workflow_replay_s") or 0.0)
        axis_other = float(self.axis_totals.get("axis_non_llm_non_replay_s") or 0.0)
        if axis_other <= 0:
            axis_other = max(0.0, axis_total - axis_expl - axis_replay)
        # Residual of measured AXIS wall-clock vs measured AXIS LLM+replay.
        self.experiment["non_llm_processing_time_s"] = axis_other
        self.experiment["axis_explanation_llm_s"] = axis_expl
        self.experiment["axis_replay_s"] = axis_replay
        self.experiment["axis_non_llm_non_replay_s"] = axis_other

    def mark(self, name: str) -> None:
        self._marks[name] = _now()

    def elapsed_since(self, name: str) -> float:
        t0 = self._marks.get(name)
        if t0 is None:
            return 0.0
        return _now() - t0

    def record_experiment_phase(self, name: str, duration_s: float) -> None:
        self.experiment[str(name)] = float(max(0.0, duration_s))

    def add_condition(self, name: str, duration_s: float) -> None:
        key = f"{name}_total_s" if not str(name).endswith("_s") else str(name)
        self.conditions[key] = self.conditions.get(key, 0.0) + float(max(0.0, duration_s))

    # -- case / round ------------------------------------------------------

    def begin_case(
        self,
        *,
        target: str,
        candidate_version: Any = None,
        factual_decision: str = "",
    ) -> None:
        set_timing_target(target)
        set_timing_round(None)
        self._case = {
            "target": str(target),
            "candidate_version": candidate_version,
            "factual_decision": str(factual_decision or ""),
            "trace_s": 0.0,
            "trace_provenance_s": 0.0,
            "axis_total_s": 0.0,
            "phases": {
                "interrogation_total_s": 0.0,
                "intervention_resolution_total_s": 0.0,
                "noop_total_s": 0.0,
                "counterfactual_replay_total_s": 0.0,
                "comparison_total_s": 0.0,
                "intermediate_synthesis_total_s": 0.0,
                "final_synthesis_s": 0.0,
            },
            "rounds": [],
            "meta": {
                "rounds": 0,
                "counterfactual_executions": 0,
                "valid_replay_comparisons": 0,
                "noop_cache_hits": 0,
                "noop_executions": 0,
                "counterfactual_provider_calls": 0,
                "counterfactual_logical_requests": 0,
                "target_observable_count": 0,
                "graph_end_count": 0,
                "fallback_full_count": 0,
                "counterfactual_target_observable_count": 0,
                "counterfactual_graph_end_count": 0,
                "counterfactual_fallback_full_count": 0,
                "noop_execution_graph_end_count": 0,
                "noop_cache_hit_count": 0,
            },
            "explanation_llm_s": 0.0,
            "axis_explanation_llm_s": 0.0,
            "non_axis_explanation_llm_s": 0.0,
            "workflow_replay_s": 0.0,
            "_t0": _now(),
        }
        self._round = None

    def record_case_condition(self, name: str, duration_s: float) -> None:
        if self._case is None:
            return
        d = float(max(0.0, duration_s))
        if name == "trace":
            self._case["trace_s"] = d
        elif name in {"trace_provenance", "trace+provenance"}:
            self._case["trace_provenance_s"] = d
        elif name == "axis":
            self._case["axis_total_s"] = d

    def begin_round(
        self,
        *,
        round_index: int,
        intervention: str = "",
        checkpoint: str = "",
        checkpoint_alias: str = "",
    ) -> None:
        set_timing_round(int(round_index))
        self._round = {
            "target": _target.get() or (self._case or {}).get("target", ""),
            "round": int(round_index),
            "intervention": str(intervention or ""),
            "checkpoint": str(checkpoint or ""),
            "checkpoint_alias": str(checkpoint_alias or ""),
            "interrogation_s": 0.0,
            "intervention_resolution_s": 0.0,
            "noop_s": 0.0,
            "counterfactual_replay_s": 0.0,
            "comparison_s": 0.0,
            "intermediate_synthesis_s": 0.0,
            "total_s": 0.0,
            "evidence_status": "",
            "causal_classification": "",
            "noop_cache_hit": False,
            "replay_completion_mode": "",
            "stopped_after_node": None,
            "executed_node_count": None,
            "replay_logical_llm_requests": 0,
            "replay_factual_response_hits": 0,
            "replay_counterfactual_memo_hits": 0,
            "replay_live_provider_calls": 0,
            "full_remaining_nodes_avoided": None,
            "_t0": _now(),
        }

    def add_round_phase(self, name: str, duration_s: float) -> None:
        if self._round is None:
            return
        key = f"{name}_s" if not str(name).endswith("_s") else str(name)
        if key not in self._round:
            return
        self._round[key] = float(self._round.get(key) or 0.0) + float(max(0.0, duration_s))
        if self._case is not None:
            totals = self._case["phases"]
            map_key = {
                "interrogation_s": "interrogation_total_s",
                "intervention_resolution_s": "intervention_resolution_total_s",
                "noop_s": "noop_total_s",
                "counterfactual_replay_s": "counterfactual_replay_total_s",
                "comparison_s": "comparison_total_s",
                "intermediate_synthesis_s": "intermediate_synthesis_total_s",
            }.get(key)
            if map_key:
                totals[map_key] = float(totals.get(map_key) or 0.0) + float(
                    max(0.0, duration_s)
                )

    def update_round_meta(self, **kwargs: Any) -> None:
        if self._round is None:
            return
        for k, v in kwargs.items():
            if k in self._round or k.startswith("replay_") or k in {
                "evidence_status",
                "causal_classification",
                "noop_cache_hit",
                "intervention",
                "checkpoint",
                "checkpoint_alias",
                "full_remaining_nodes_avoided",
            }:
                self._round[k] = v

    def abandon_round(self) -> None:
        """Drop an in-progress round without printing (failed attempt)."""
        self._round = None
        set_timing_round(None)

    def end_round(self) -> None:
        if self._round is None:
            return
        t0 = self._round.pop("_t0", None)
        if t0 is not None:
            self._round["total_s"] = float(_now() - t0)
        else:
            parts = (
                float(self._round.get("interrogation_s") or 0.0)
                + float(self._round.get("intervention_resolution_s") or 0.0)
                + float(self._round.get("noop_s") or 0.0)
                + float(self._round.get("counterfactual_replay_s") or 0.0)
                + float(self._round.get("comparison_s") or 0.0)
                + float(self._round.get("intermediate_synthesis_s") or 0.0)
            )
            self._round["total_s"] = parts
        self.print_round_summary(self._round)
        if self._case is not None:
            self._case["rounds"].append(dict(self._round))
            self._case["meta"]["rounds"] = len(self._case["rounds"])
        self._round = None
        set_timing_round(None)

    def record_final_synthesis(self, duration_s: float) -> None:
        d = float(max(0.0, duration_s))
        if self._case is not None:
            self._case["phases"]["final_synthesis_s"] = d

    def end_case(self, *, meta: Optional[Dict[str, Any]] = None) -> None:
        if self._case is None:
            return
        if meta:
            self._case["meta"].update(dict(meta))
        t0 = self._case.pop("_t0", None)
        if t0 is not None and not self._case.get("axis_total_s"):
            # Prefer explicitly recorded axis duration when present.
            pass
        phases = self._case["phases"]
        explanation_llm = float(self._case.get("axis_explanation_llm_s") or 0.0)
        # Keep explanation_llm_s as AXIS-scoped (non-overlapping with axis_total_s).
        all_expl = float(self._case.get("explanation_llm_s") or 0.0)
        if explanation_llm <= 0:
            explanation_llm = all_expl
        workflow_replay = float(self._case.get("workflow_replay_s") or 0.0)
        axis_total = float(self._case.get("axis_total_s") or 0.0)
        if axis_total <= 0:
            axis_total = (
                float(phases.get("interrogation_total_s") or 0.0)
                + float(phases.get("intervention_resolution_total_s") or 0.0)
                + float(phases.get("noop_total_s") or 0.0)
                + float(phases.get("counterfactual_replay_total_s") or 0.0)
                + float(phases.get("comparison_total_s") or 0.0)
                + float(phases.get("intermediate_synthesis_total_s") or 0.0)
                + float(phases.get("final_synthesis_s") or 0.0)
            )
            self._case["axis_total_s"] = axis_total
        other = max(0.0, axis_total - explanation_llm - workflow_replay)
        self._case["axis_explanation_llm_s"] = explanation_llm
        self._case["axis_replay_s"] = workflow_replay
        self._case["axis_non_llm_non_replay_s"] = other
        self._case["other_processing_s"] = other
        self._case["explanation_llm_pct"] = _pct(explanation_llm, axis_total)
        self._case["workflow_replay_pct"] = _pct(workflow_replay, axis_total)
        self._case["other_processing_pct"] = _pct(other, axis_total)
        self._case["axis_explanation_llm_pct"] = _pct(explanation_llm, axis_total)
        self._case["axis_replay_pct"] = _pct(workflow_replay, axis_total)
        self._case["axis_non_llm_non_replay_pct"] = _pct(other, axis_total)
        self.print_case_summary(self._case)
        self.cases.append(dict(self._case))
        # Aggregate AXIS totals
        for k, v in phases.items():
            self.axis_totals[k] = float(self.axis_totals.get(k) or 0.0) + float(v or 0.0)
        self.axis_totals["axis_total_s"] = float(
            self.axis_totals.get("axis_total_s") or 0.0
        ) + float(self._case.get("axis_total_s") or 0.0)
        self.axis_totals["explanation_llm_s"] = float(
            self.axis_totals.get("explanation_llm_s") or 0.0
        ) + explanation_llm
        self.axis_totals["axis_explanation_llm_s"] = float(
            self.axis_totals.get("axis_explanation_llm_s") or 0.0
        ) + explanation_llm
        self.axis_totals["workflow_replay_s"] = float(
            self.axis_totals.get("workflow_replay_s") or 0.0
        ) + workflow_replay
        self.axis_totals["axis_replay_s"] = float(
            self.axis_totals.get("axis_replay_s") or 0.0
        ) + workflow_replay
        self.axis_totals["axis_non_llm_non_replay_s"] = float(
            self.axis_totals.get("axis_non_llm_non_replay_s") or 0.0
        ) + other
        self.axis_totals["non_axis_explanation_llm_s"] = float(
            self.axis_totals.get("non_axis_explanation_llm_s") or 0.0
        ) + float(self._case.get("non_axis_explanation_llm_s") or 0.0)
        self._case = None
        set_timing_target("")

    # -- explanation LLM / replay / provider -------------------------------

    def record_explanation_llm(
        self,
        *,
        phase: str,
        duration_s: float,
        model: str = "",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        d = float(max(0.0, duration_s))
        entry = {
            "target": _target.get() or (self._case or {}).get("target", ""),
            "round": _round.get(),
            "phase": str(phase or ""),
            "model": str(model or ""),
            "duration_s": d,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        self.explanation_llm_calls.append(entry)
        if self._case is not None:
            phase_name = str(phase or "")
            if phase_name in AXIS_EXPLANATION_LLM_PHASES:
                self._case["axis_explanation_llm_s"] = float(
                    self._case.get("axis_explanation_llm_s") or 0.0
                ) + d
                # explanation_llm_s is AXIS-scoped so percentages stay non-overlapping.
                self._case["explanation_llm_s"] = float(
                    self._case.get("explanation_llm_s") or 0.0
                ) + d
            else:
                self._case["non_axis_explanation_llm_s"] = float(
                    self._case.get("non_axis_explanation_llm_s") or 0.0
                ) + d

    def record_replay(
        self,
        *,
        branch: str,
        duration_s: float,
        checkpoint: str = "",
        checkpoint_alias: str = "",
        bounded_requested: bool = False,
        cache_hit: Optional[bool] = None,
        completion_mode: str = "",
        stopped_after_node: Any = None,
        executed_node_count: Any = None,
        logical_llm_requests: int = 0,
        factual_response_hits: int = 0,
        counterfactual_memo_hits: int = 0,
        live_provider_calls: int = 0,
        full_remaining_nodes_avoided: Any = None,
    ) -> None:
        d = float(max(0.0, duration_s))
        entry = {
            "target": _target.get() or (self._case or {}).get("target", ""),
            "round": _round.get(),
            "branch": str(branch),
            "checkpoint": str(checkpoint or checkpoint_alias or ""),
            "checkpoint_alias": str(checkpoint_alias or ""),
            "bounded_requested": bool(bounded_requested),
            "duration_s": d,
            "cache_hit": cache_hit,
            "completion_mode": str(completion_mode or ""),
            "stopped_after_node": stopped_after_node,
            "executed_node_count": executed_node_count,
            "logical_llm_requests": int(logical_llm_requests or 0),
            "factual_response_hits": int(factual_response_hits or 0),
            "counterfactual_memo_hits": int(counterfactual_memo_hits or 0),
            "live_provider_calls": int(live_provider_calls or 0),
            "full_remaining_nodes_avoided": full_remaining_nodes_avoided,
        }
        self.replay_events.append(entry)
        if self._case is not None:
            self._case["workflow_replay_s"] = float(
                self._case.get("workflow_replay_s") or 0.0
            ) + d
            meta = self._case["meta"]
            mode = str(completion_mode or "")
            if branch == "counterfactual":
                if mode == "target_observable":
                    meta["counterfactual_target_observable_count"] = int(
                        meta.get("counterfactual_target_observable_count") or 0
                    ) + 1
                    meta["target_observable_count"] = int(
                        meta.get("target_observable_count") or 0
                    ) + 1
                elif mode == "fallback_full":
                    meta["counterfactual_fallback_full_count"] = int(
                        meta.get("counterfactual_fallback_full_count") or 0
                    ) + 1
                    meta["fallback_full_count"] = int(meta.get("fallback_full_count") or 0) + 1
                elif mode == "graph_end":
                    meta["counterfactual_graph_end_count"] = int(
                        meta.get("counterfactual_graph_end_count") or 0
                    ) + 1
            elif branch == "noop":
                if cache_hit:
                    meta["noop_cache_hit_count"] = int(
                        meta.get("noop_cache_hit_count") or 0
                    ) + 1
                else:
                    meta["noop_execution_graph_end_count"] = int(
                        meta.get("noop_execution_graph_end_count") or 0
                    ) + 1
                    if mode == "graph_end":
                        meta["graph_end_count"] = int(meta.get("graph_end_count") or 0) + 1
            if branch == "noop":
                if cache_hit:
                    meta["noop_cache_hits"] = int(meta.get("noop_cache_hits") or 0) + 1
                else:
                    meta["noop_executions"] = int(meta.get("noop_executions") or 0) + 1
            elif branch == "counterfactual":
                meta["counterfactual_executions"] = int(
                    meta.get("counterfactual_executions") or 0
                ) + 1
                meta["counterfactual_provider_calls"] = int(
                    meta.get("counterfactual_provider_calls") or 0
                ) + int(live_provider_calls or 0)
                meta["counterfactual_logical_requests"] = int(
                    meta.get("counterfactual_logical_requests") or 0
                ) + int(logical_llm_requests or 0)
        if self._round is not None:
            if branch == "noop":
                self.add_round_phase("noop", d)
                self.update_round_meta(noop_cache_hit=bool(cache_hit))
            elif branch == "counterfactual":
                self.add_round_phase("counterfactual_replay", d)
                self.update_round_meta(
                    replay_completion_mode=completion_mode,
                    stopped_after_node=stopped_after_node,
                    executed_node_count=executed_node_count,
                    replay_logical_llm_requests=logical_llm_requests,
                    replay_factual_response_hits=factual_response_hits,
                    replay_counterfactual_memo_hits=counterfactual_memo_hits,
                    replay_live_provider_calls=live_provider_calls,
                    full_remaining_nodes_avoided=full_remaining_nodes_avoided,
                )

    def _component_bucket(self, component: str, *, model: str = "") -> Dict[str, Any]:
        key = str(component or "unattributed")
        bucket = self.provider_by_component.setdefault(
            key,
            {
                "model": "",
                "logical_calls": 0.0,
                "factual_ledger_hits": 0.0,
                "cf_memo_hits": 0.0,
                "provider_calls": 0.0,
                "provider_time_s": 0.0,
            },
        )
        if model and not bucket.get("model"):
            bucket["model"] = str(model)
        return bucket

    def record_workflow_provider_call(
        self, *, component: str, duration_s: float, model: str = ""
    ) -> None:
        d = float(max(0.0, duration_s))
        bucket = self._component_bucket(component, model=model)
        bucket["provider_calls"] = float(bucket["provider_calls"]) + 1.0
        bucket["provider_time_s"] = float(bucket["provider_time_s"]) + d
        self.workflow_llm.setdefault("provider_time_s", 0.0)
        self.workflow_llm["provider_time_s"] = float(
            self.workflow_llm["provider_time_s"]
        ) + d
        add_replay_span("provider_wait", d)

    def record_workflow_logical_call(self, *, component: str, model: str = "") -> None:
        bucket = self._component_bucket(component, model=model)
        bucket["logical_calls"] = float(bucket["logical_calls"]) + 1.0

    def record_workflow_ledger_hit(
        self,
        *,
        component: str,
        model: str = "",
        factual_hit: bool = False,
        cf_memo_hit: bool = False,
    ) -> None:
        bucket = self._component_bucket(component, model=model)
        if factual_hit:
            bucket["factual_ledger_hits"] = float(bucket.get("factual_ledger_hits") or 0.0) + 1.0
        if cf_memo_hit:
            bucket["cf_memo_hits"] = float(bucket.get("cf_memo_hits") or 0.0) + 1.0

    def record_noop_cache_profile(self, profile: Dict[str, Any]) -> None:
        entry = dict(profile or {})
        entry.setdefault("target", _target.get() or (self._case or {}).get("target", ""))
        entry.setdefault("round", _round.get())
        self.noop_cache_profiles.append(entry)
        if self._round is not None:
            self._round["noop_cache_profile"] = dict(entry)
        self.print_noop_cache_profile(entry)

    def update_noop_cache_profile(self, **spans: Any) -> None:
        if self._round is None:
            return
        profile = dict(self._round.get("noop_cache_profile") or {})
        if not profile:
            return
        for k, v in spans.items():
            try:
                profile[k] = float(profile.get(k) or 0.0) + float(max(0.0, v))
            except (TypeError, ValueError):
                profile[k] = v
        known = (
            float(profile.get("cache_lookup_s") or 0.0)
            + float(profile.get("cache_factory_s") or 0.0)
            + float(profile.get("cached_result_clone_s") or 0.0)
            + float(profile.get("cached_result_observation_s") or 0.0)
            + float(profile.get("cached_result_conversion_s") or 0.0)
            + float(profile.get("provenance_processing_s") or 0.0)
        )
        total = float(profile.get("total_noop_s") or 0.0)
        if total > 0:
            profile["other_s"] = max(0.0, total - known)
        self._round["noop_cache_profile"] = profile
        if self.noop_cache_profiles:
            self.noop_cache_profiles[-1] = dict(profile)

    def record_replay_profile(self, profile: Dict[str, Any]) -> None:
        entry = dict(profile or {})
        entry.setdefault("target", _target.get() or (self._case or {}).get("target", ""))
        entry.setdefault("round", _round.get())
        self.replay_profiles.append(entry)
        if self._round is not None:
            self._round["replay_profile"] = dict(entry)
        self.print_replay_profile(entry)

    def record_node_profile(self, nodes: List[Dict[str, Any]]) -> None:
        target = _target.get() or (self._case or {}).get("target", "")
        round_idx = _round.get()
        wrapped = []
        for node in list(nodes or []):
            item = dict(node)
            item.setdefault("target", target)
            item.setdefault("round", round_idx)
            wrapped.append(item)
            self.node_profiles.append(item)
        if self._round is not None:
            self._round["node_profile"] = list(wrapped)
        if wrapped:
            self.print_node_profile(target=str(target), round_idx=round_idx, nodes=wrapped)

    # -- printing ----------------------------------------------------------

    def print_noop_cache_profile(self, profile: Dict[str, Any]) -> None:
        print(
            "\n[NOOP CACHE PROFILE]\n"
            f"target={profile.get('target')}\n"
            f"checkpoint={profile.get('checkpoint_alias') or profile.get('checkpoint')}\n"
            f"cache_hit={str(bool(profile.get('cache_hit'))).lower()}\n"
            f"graph_executed={str(bool(profile.get('graph_executed'))).lower()}\n\n"
            f"lookup_s={float(profile.get('cache_lookup_s') or 0):.4f}\n"
            f"factory_s={float(profile.get('cache_factory_s') or 0):.4f}\n"
            f"clone_s={float(profile.get('cached_result_clone_s') or 0):.4f}\n"
            f"conversion_s={float(profile.get('cached_result_conversion_s') or 0):.4f}\n"
            f"observation_s={float(profile.get('cached_result_observation_s') or 0):.4f}\n"
            f"provenance_s={float(profile.get('provenance_processing_s') or 0):.4f}\n"
            f"other_s={float(profile.get('other_s') or 0):.4f}\n"
            f"total_s={float(profile.get('total_noop_s') or 0):.4f}\n"
        )

    def print_replay_profile(self, profile: Dict[str, Any]) -> None:
        top = dict(profile.get("top_level") or {})
        nested = dict(profile.get("nested") or {})
        if not top and not nested:
            top = {
                k: float(profile.get(k) or 0.0)
                for k in (
                    "checkpoint_restore_s",
                    "intervention_prepare_s",
                    "fork_creation_s",
                    "graph_execution_s",
                    "branch_history_fetch_s",
                    "branch_history_filter_s",
                    "final_state_store_copy_s",
                    "artifact_resolver_rebuild_s",
                    "workflow_result_materialization_s",
                    "other_s",
                )
            }
            nested = {
                k: float(profile.get(k) or 0.0)
                for k in (
                    "provider_wait_s",
                    "ledger_lookup_s",
                    "observability_check_s",
                )
            }
        print(
            "\n[AXIS REPLAY COST BREAKDOWN]\n"
            f"target={profile.get('target')}\n"
            f"round={profile.get('round')}\n\n"
            f"total_s={float(profile.get('total_replay_s') or 0):.4f}\n\n"
            "TOP LEVEL\n"
            f"checkpoint_restore_s={float(top.get('checkpoint_restore_s') or 0):.4f}\n"
            f"intervention_prepare_s={float(top.get('intervention_prepare_s') or 0):.4f}\n"
            f"intervention_application_s="
            f"{float(top.get('intervention_application_s') or 0):.4f}\n"
            f"fork_creation_s={float(top.get('fork_creation_s') or 0):.4f}\n"
            f"graph_execution_s={float(top.get('graph_execution_s') or 0):.4f}\n"
            f"branch_history_fetch_s={float(top.get('branch_history_fetch_s') or 0):.4f}\n"
            f"branch_history_filter_s={float(top.get('branch_history_filter_s') or 0):.4f}\n"
            f"final_state_store_copy_s={float(top.get('final_state_store_copy_s') or 0):.4f}\n"
            f"resolver_rebuild_s="
            f"{float(top.get('artifact_resolver_rebuild_s') or 0):.4f}\n"
            f"result_materialization_s="
            f"{float(top.get('workflow_result_materialization_s') or 0):.4f}\n"
            f"other_s={float(top.get('other_s') or profile.get('other_s') or 0):.4f}\n\n"
            "NESTED\n"
            f"provider_wait_s={float(nested.get('provider_wait_s') or 0):.4f}\n"
            f"ledger_lookup_s={float(nested.get('ledger_lookup_s') or 0):.4f}\n"
            f"observability_check_s={float(nested.get('observability_check_s') or 0):.4f}\n\n"
            f"history_collection_mode="
            f"{profile.get('history_collection_mode') or ''}\n"
            f"history_snapshots_scanned="
            f"{int(profile.get('history_snapshots_scanned') or 0)}\n"
            f"branch_snapshots_retained="
            f"{int(profile.get('branch_snapshots_retained') or 0)}\n"
        )

    def print_node_profile(
        self,
        *,
        target: str,
        round_idx: Any,
        nodes: List[Dict[str, Any]],
    ) -> None:
        print(
            f"\n[AXIS NODE TIMING]\n"
            f"target={target}\n"
            f"round={round_idx if round_idx is not None else ''}\n"
        )
        for node in nodes:
            name = str(node.get("node") or node.get("name") or "")
            elapsed = float(node.get("elapsed_s") or 0.0)
            print(
                f"{name:<40} {elapsed:8.4f} s  "
                f"logical={int(node.get('logical_llm_requests') or 0)} "
                f"factual_hits={int(node.get('factual_ledger_hits') or 0)} "
                f"cf_memo={int(node.get('cf_memo_hits') or 0)} "
                f"provider={int(node.get('provider_calls') or 0)} "
                f"provider_s={float(node.get('provider_time_s') or 0):.4f}"
            )

    def print_round_summary(self, rd: Dict[str, Any]) -> None:
        causal = str(rd.get("causal_classification") or "")
        evidence = str(rd.get("evidence_status") or "")
        # Do not print replay evidence_status as if it were AXIS causal role.
        causal_line = (
            f"causal_classification={causal}\n"
            if causal and causal != evidence
            else "causal_classification=(not computed at round time)\n"
        )
        print(
            "\n[AXIS ROUND TIMING]\n"
            f"target={rd.get('target')}\n"
            f"round={rd.get('round')}\n\n"
            f"interrogation_s={rd.get('interrogation_s', 0):.4f}\n"
            f"intervention_resolution_s={rd.get('intervention_resolution_s', 0):.4f}\n"
            f"noop_s={rd.get('noop_s', 0):.4f}\n"
            f"counterfactual_replay_s={rd.get('counterfactual_replay_s', 0):.4f}\n"
            f"comparison_s={rd.get('comparison_s', 0):.4f}\n"
            f"intermediate_synthesis_s={rd.get('intermediate_synthesis_s', 0):.4f}\n"
            f"total_s={rd.get('total_s', 0):.4f}\n\n"
            f"intervention={rd.get('intervention')}\n"
            f"evidence_status={evidence}\n"
            f"{causal_line}\n"
            f"noop_cache_hit={str(bool(rd.get('noop_cache_hit'))).lower()}\n\n"
            f"replay_completion_mode={rd.get('replay_completion_mode')}\n"
            f"replay_nodes={rd.get('executed_node_count')}\n"
            f"replay_provider_calls={rd.get('replay_live_provider_calls')}\n"
        )

    def print_case_summary(self, case: Dict[str, Any]) -> None:
        phases = case.get("phases") or {}
        meta = case.get("meta") or {}
        print(
            "\n[AXIS CASE TIMING]\n"
            f"target={case.get('target')}\n"
            f"rounds={meta.get('rounds')}\n\n"
            f"total_s={float(case.get('axis_total_s') or 0):.4f}\n\n"
            f"interrogation_s={float(phases.get('interrogation_total_s') or 0):.4f}\n"
            f"noop_s={float(phases.get('noop_total_s') or 0):.4f}\n"
            f"counterfactual_replay_s="
            f"{float(phases.get('counterfactual_replay_total_s') or 0):.4f}\n"
            f"comparison_s={float(phases.get('comparison_total_s') or 0):.4f}\n"
            f"intermediate_synthesis_s="
            f"{float(phases.get('intermediate_synthesis_total_s') or 0):.4f}\n"
            f"final_synthesis_s={float(phases.get('final_synthesis_s') or 0):.4f}\n\n"
            f"axis_explanation_llm_s={float(case.get('axis_explanation_llm_s') or case.get('explanation_llm_s') or 0):.4f}\n"
            f"axis_replay_s={float(case.get('axis_replay_s') or case.get('workflow_replay_s') or 0):.4f}\n"
            f"axis_non_llm_non_replay_s={float(case.get('axis_non_llm_non_replay_s') or case.get('other_processing_s') or 0):.4f}\n\n"
            f"explanation_llm_s={float(case.get('explanation_llm_s') or 0):.4f}\n"
            f"workflow_replay_s={float(case.get('workflow_replay_s') or 0):.4f}\n"
            f"other_processing_s={float(case.get('other_processing_s') or 0):.4f}\n\n"
            f"axis_explanation_llm_pct={float(case.get('axis_explanation_llm_pct') or case.get('explanation_llm_pct') or 0):.1f}\n"
            f"axis_replay_pct={float(case.get('axis_replay_pct') or case.get('workflow_replay_pct') or 0):.1f}\n"
            f"axis_non_llm_non_replay_pct={float(case.get('axis_non_llm_non_replay_pct') or case.get('other_processing_pct') or 0):.1f}\n\n"
            f"counterfactual_executions={meta.get('counterfactual_executions')}\n"
            f"valid_replays={meta.get('valid_replay_comparisons')}\n\n"
            f"noop_cache_hits={meta.get('noop_cache_hits')}\n"
            f"counterfactual_provider_calls={meta.get('counterfactual_provider_calls')}\n"
        )

    def print_experiment_summary(self) -> None:
        axis_total = float(self.axis_totals.get("axis_total_s") or 0.0)
        expl = float(self.axis_totals.get("axis_explanation_llm_s") or 0.0)
        if expl <= 0:
            expl = float(self.axis_totals.get("explanation_llm_s") or 0.0)
        replay = float(self.axis_totals.get("axis_replay_s") or 0.0)
        if replay <= 0:
            replay = float(self.axis_totals.get("workflow_replay_s") or 0.0)
        other = float(self.axis_totals.get("axis_non_llm_non_replay_s") or 0.0)
        if other <= 0:
            other = max(0.0, axis_total - expl - replay)
        cf_exec = sum(
            int((c.get("meta") or {}).get("counterfactual_executions") or 0)
            for c in self.cases
        )
        valid = sum(
            int((c.get("meta") or {}).get("valid_replay_comparisons") or 0)
            for c in self.cases
        )
        tob = sum(
            int((c.get("meta") or {}).get("counterfactual_target_observable_count") or 0)
            or int((c.get("meta") or {}).get("target_observable_count") or 0)
            for c in self.cases
        )
        cf_gend = sum(
            int((c.get("meta") or {}).get("counterfactual_graph_end_count") or 0)
            for c in self.cases
        )
        fb = sum(
            int((c.get("meta") or {}).get("counterfactual_fallback_full_count") or 0)
            or int((c.get("meta") or {}).get("fallback_full_count") or 0)
            for c in self.cases
        )
        noop_ex = sum(
            int((c.get("meta") or {}).get("noop_executions") or 0) for c in self.cases
        )
        noop_hit = sum(
            int((c.get("meta") or {}).get("noop_cache_hits") or 0) for c in self.cases
        )
        noop_end = sum(
            int((c.get("meta") or {}).get("noop_execution_graph_end_count") or 0)
            for c in self.cases
        )
        pct_expl = _pct(expl, axis_total)
        pct_replay = _pct(replay, axis_total)
        pct_other = _pct(other, axis_total)
        print(
            "\n========================================\n"
            "AXIS FULL EXPERIMENT TIMING\n"
            "========================================\n\n"
            f"questions_explained={len(self.cases)}\n\n"
            f"factual_workflow_s={float(self.experiment.get('factual_workflow_s') or 0):.4f}\n"
            f"trace_total_s={float(self.conditions.get('trace_total_s') or 0):.4f}\n"
            f"trace_provenance_total_s="
            f"{float(self.conditions.get('trace_provenance_total_s') or 0):.4f}\n"
            f"axis_total_s={axis_total:.4f}\n"
            f"experiment_total_s={float(self.experiment.get('experiment_total_s') or 0):.4f}\n\n"
            "AXIS PHASES\n"
            f"interrogation_s="
            f"{float(self.axis_totals.get('interrogation_total_s') or 0):.4f}\n"
            f"noop_s={float(self.axis_totals.get('noop_total_s') or 0):.4f}\n"
            f"counterfactual_replay_s="
            f"{float(self.axis_totals.get('counterfactual_replay_total_s') or 0):.4f}\n"
            f"comparison_s={float(self.axis_totals.get('comparison_total_s') or 0):.4f}\n"
            f"intermediate_synthesis_s="
            f"{float(self.axis_totals.get('intermediate_synthesis_total_s') or 0):.4f}\n"
            f"final_synthesis_s="
            f"{float(self.axis_totals.get('final_synthesis_s') or 0):.4f}\n\n"
            "AXIS SCOPES (non-overlapping; participant AXIS synthesis is inside axis_total_s)\n"
            f"axis_explanation_llm_s={expl:.4f}\n"
            f"axis_replay_s={replay:.4f}\n"
            f"axis_non_llm_non_replay_s={other:.4f}\n\n"
            f"axis_explanation_llm_pct={pct_expl:.1f}\n"
            f"axis_replay_pct={pct_replay:.1f}\n"
            f"axis_non_llm_non_replay_pct={pct_other:.1f}\n"
            f"axis_scope_pct_sum={pct_expl + pct_replay + pct_other:.1f}\n\n"
            f"explanation_layer_llm_s={expl:.4f}\n"
            f"workflow_replay_s={replay:.4f}\n"
            f"other_axis_processing_s={other:.4f}\n\n"
            f"explanation_layer_pct={pct_expl:.1f}\n"
            f"workflow_replay_pct={pct_replay:.1f}\n"
            f"other_processing_pct={pct_other:.1f}\n\n"
            "REPLAY\n"
            f"counterfactual_executions={cf_exec}\n"
            f"valid_replay_comparisons={valid}\n\n"
            f"counterfactual_target_observable_count={tob}\n"
            f"counterfactual_graph_end_count={cf_gend}\n"
            f"counterfactual_fallback_full_count={fb}\n"
            f"target_observable_count={tob}\n"
            f"graph_end_count={noop_end}\n"
            f"fallback_full_count={fb}\n\n"
            f"noop_executions={noop_ex}\n"
            f"noop_cache_hits={noop_hit}\n"
            f"noop_execution_graph_end_count={noop_end}\n"
            f"noop_cache_hit_count={noop_hit}\n\n"
            "WORKFLOW LLM\n"
            f"factual_provider_calls="
            f"{self.workflow_llm.get('factual_provider_calls', '')}\n"
            f"noop_provider_calls={self.workflow_llm.get('noop_provider_calls', '')}\n"
            f"counterfactual_provider_calls="
            f"{self.workflow_llm.get('counterfactual_provider_calls', '')}\n\n"
            f"factual_response_hits="
            f"{self.workflow_llm.get('factual_response_hits', '')}\n"
            f"counterfactual_memo_hits="
            f"{self.workflow_llm.get('counterfactual_memo_hits', '')}\n"
        )
        cove = self.provider_by_component.get("cross_cove") or {}
        if cove:
            print(
                "CROSS-COVE\n"
                f"model={cove.get('model', '')}\n"
                f"logical_calls={int(cove.get('logical_calls') or 0)}\n"
                f"factual_ledger_hits={int(cove.get('factual_ledger_hits') or 0)}\n"
                f"cf_memo_hits={int(cove.get('cf_memo_hits') or 0)}\n"
                f"provider_calls={int(cove.get('provider_calls') or 0)}\n"
                f"provider_time_s={float(cove.get('provider_time_s') or 0):.4f}\n"
            )
        self.print_per_question_table()

    def print_per_question_table(self) -> None:
        print("\ntarget | rounds | axis_s | explanation_llm_s | replay_s | provider_calls")
        axis_vals: List[float] = []
        for case in self.cases:
            meta = case.get("meta") or {}
            axis_s = float(case.get("axis_total_s") or 0.0)
            axis_vals.append(axis_s)
            print(
                f"{case.get('target')} | {meta.get('rounds')} | "
                f"{axis_s:.2f} | {float(case.get('explanation_llm_s') or 0):.2f} | "
                f"{float(case.get('workflow_replay_s') or 0):.2f} | "
                f"{meta.get('counterfactual_provider_calls')}"
            )
        if axis_vals:
            print(
                f"\navg_axis_s_per_question={statistics.mean(axis_vals):.4f}\n"
                f"median_axis_s_per_question={statistics.median(axis_vals):.4f}\n"
                f"min_axis_s={min(axis_vals):.4f}\n"
                f"max_axis_s={max(axis_vals):.4f}\n"
            )

    # -- export ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        axis_total = float(self.axis_totals.get("axis_total_s") or 0.0)
        expl = float(self.axis_totals.get("axis_explanation_llm_s") or 0.0)
        if expl <= 0:
            expl = float(self.axis_totals.get("explanation_llm_s") or 0.0)
        replay = float(self.axis_totals.get("axis_replay_s") or 0.0)
        if replay <= 0:
            replay = float(self.axis_totals.get("workflow_replay_s") or 0.0)
        other = float(self.axis_totals.get("axis_non_llm_non_replay_s") or 0.0)
        if other <= 0:
            other = max(0.0, axis_total - expl - replay)
        pct_expl = _pct(expl, axis_total)
        pct_replay = _pct(replay, axis_total)
        pct_other = _pct(other, axis_total)
        return {
            "experiment": dict(self.experiment),
            "conditions": dict(self.conditions),
            "axis": {
                "totals": {
                    **dict(self.axis_totals),
                    "axis_explanation_llm_s": expl,
                    "axis_replay_s": replay,
                    "axis_non_llm_non_replay_s": other,
                    "other_axis_processing_s": other,
                    "explanation_layer_pct": pct_expl,
                    "workflow_replay_pct": pct_replay,
                    "other_processing_pct": pct_other,
                    "axis_explanation_llm_pct": pct_expl,
                    "axis_replay_pct": pct_replay,
                    "axis_non_llm_non_replay_pct": pct_other,
                    "axis_scope_pct_sum": pct_expl + pct_replay + pct_other,
                },
                "cases": list(self.cases),
                "replay_profile": list(self.replay_profiles),
                "noop_cache_profile": list(self.noop_cache_profiles),
                "node_profile": list(self.node_profiles),
            },
            "workflow_llm": {
                **dict(self.workflow_llm),
                "by_component": dict(self.provider_by_component),
                "cross_cove": dict(self.provider_by_component.get("cross_cove") or {}),
            },
            "explanation_llm_calls": list(self.explanation_llm_calls),
            "replay_events": list(self.replay_events),
        }

    def write_csv(self, path: str) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "run_id",
            "target",
            "rounds",
            "axis_total_s",
            "interrogation_s",
            "noop_s",
            "replay_s",
            "comparison_s",
            "intermediate_synthesis_s",
            "final_synthesis_s",
            "explanation_llm_s",
            "counterfactual_provider_calls",
            "target_observable_replays",
            "graph_end_replays",
            "fallback_full_replays",
            "valid_replay_comparisons",
        ]
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for case in self.cases:
                phases = case.get("phases") or {}
                meta = case.get("meta") or {}
                w.writerow(
                    {
                        "run_id": self.run_id,
                        "target": case.get("target"),
                        "rounds": meta.get("rounds"),
                        "axis_total_s": case.get("axis_total_s"),
                        "interrogation_s": phases.get("interrogation_total_s"),
                        "noop_s": phases.get("noop_total_s"),
                        "replay_s": phases.get("counterfactual_replay_total_s"),
                        "comparison_s": phases.get("comparison_total_s"),
                        "intermediate_synthesis_s": phases.get(
                            "intermediate_synthesis_total_s"
                        ),
                        "final_synthesis_s": phases.get("final_synthesis_s"),
                        "explanation_llm_s": case.get("explanation_llm_s"),
                        "counterfactual_provider_calls": meta.get(
                            "counterfactual_provider_calls"
                        ),
                        "target_observable_replays": meta.get(
                            "target_observable_count"
                        ),
                        "graph_end_replays": meta.get("graph_end_count"),
                        "fallback_full_replays": meta.get("fallback_full_count"),
                        "valid_replay_comparisons": meta.get(
                            "valid_replay_comparisons"
                        ),
                    }
                )
        return str(out.resolve())


@contextmanager
def timed_span() -> Iterator[List[float]]:
    """Yield a one-element list filled with duration on exit."""
    box: List[float] = [0.0]
    t0 = _now()
    try:
        yield box
    finally:
        box[0] = _now() - t0
