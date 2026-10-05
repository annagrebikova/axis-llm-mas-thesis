"""AXIS query executor: factual WHAT and intervention dispatch."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from ..artifacts.registry import ArtifactRegistry
from ..replay.checkpoints import BoundarySelector
from ..replay.executor import LangGraphReplayExecutor
from ..replay.interventions import ArtifactIntervention
from .models import (
    AxisQueryContext,
    FactualLookupResult,
    FactualQueryRecord,
    InterventionQueryRecord,
    QuerySignature,
    ResolvedQuery,
)
from .queries import QueryValidationError


def _replay_configuration_id(replay_executor: Any, *, mode: str = "full") -> str:
    """Stable-enough config fingerprint for the NOOP cache key.

    NOOP controls are always full-to-END (see AxisQueryExecutor), so the cache
    key uses ``mode="full"``. Target-bounded replay is for counterfactuals only
    and must not share a cached NOOP branch across explained targets.
    """
    explicit = getattr(replay_executor, "replay_configuration_id", None)
    if explicit:
        return str(explicit)
    graph = getattr(replay_executor, "compiled_graph", None)
    tag = str(mode or "full")
    if graph is None:
        return f"fixture:{tag}"
    return f"graph:{id(graph)}:{tag}"


def _timing_context() -> Tuple[Any, str, Any]:
    try:
        from ..experiment.timing import _round, _target, current_timing

        return current_timing(), str(_target.get() or ""), _round.get()
    except Exception:
        return None, "", None


def _branch_replay_metrics(result: Any) -> Dict[str, Any]:
    """Extract replay diagnostics already attached to a WorkflowRunResult."""
    nodes = list(getattr(result, "executed_nodes", None) or [])
    llm = dict(getattr(result, "llm_replay_diagnostics", None) or {})
    executed_count: Any
    if nodes:
        executed_count = len(nodes)
    else:
        executed_count = llm.get("executed_node_count")
    return {
        "completion_mode": str(
            getattr(result, "replay_completion_mode", None)
            or llm.get("replay_completion_mode")
            or ""
        ),
        "stopped_after_node": getattr(result, "stopped_after_node", None)
        or llm.get("stopped_after_node"),
        "executed_node_count": executed_count,
        "logical_llm_requests": int(llm.get("request_count") or 0),
        "factual_response_hits": int(llm.get("factual_response_hits") or 0),
        "counterfactual_memo_hits": int(llm.get("memo_hits") or 0),
        "live_provider_calls": int(llm.get("live_provider_calls") or 0),
        "full_remaining_nodes_avoided": getattr(
            result, "skipped_downstream_steps", None
        ),
    }


def _print_replay_start(
    *,
    target: str,
    round_idx: Any,
    branch: str,
    checkpoint: str,
    bounded_requested: bool,
) -> None:
    print(
        "[AXIS REPLAY START]\n"
        f"target={target or '?'}\n"
        f"round={round_idx if round_idx is not None else ''}\n"
        f"branch={branch}\n"
        f"checkpoint={checkpoint}\n"
        f"bounded_requested={str(bool(bounded_requested)).lower()}"
    )


def _print_replay_end(
    *,
    target: str,
    round_idx: Any,
    branch: str,
    duration_s: float,
    metrics: Dict[str, Any],
    cache_hit: Optional[bool] = None,
) -> None:
    lines = [
        "[AXIS REPLAY END]",
        f"target={target or '?'}",
        f"round={round_idx if round_idx is not None else ''}",
        f"branch={branch}",
        f"duration_s={duration_s:.4f}",
        "",
        f"completion_mode={metrics.get('completion_mode')}",
        f"stopped_after_node={metrics.get('stopped_after_node')}",
        f"executed_node_count={metrics.get('executed_node_count')}",
        "",
        f"logical_llm_requests={metrics.get('logical_llm_requests')}",
        f"factual_response_hits={metrics.get('factual_response_hits')}",
        f"counterfactual_memo_hits={metrics.get('counterfactual_memo_hits')}",
        f"live_provider_calls={metrics.get('live_provider_calls')}",
    ]
    if cache_hit is not None:
        lines.extend(["", f"cache_hit={str(bool(cache_hit)).lower()}"])
    print("\n".join(lines))


class AxisQueryExecutor:
    """Execute resolved AXIS queries against the factual/replay backend."""

    def __init__(
        self,
        *,
        context: AxisQueryContext,
        replay_executor: LangGraphReplayExecutor,
        noop_cache: Optional[Any] = None,  # NoOpReplayCache
        stop_when_observable: bool = True,
    ) -> None:
        self.context = context
        self.replay_executor = replay_executor
        self.noop_cache = noop_cache
        self.stop_when_observable = bool(stop_when_observable)
        if self.stop_when_observable:
            setattr(self.replay_executor, "target_bounded_replay", True)

    def execute(
        self, resolved: ResolvedQuery
    ) -> "FactualQueryRecord | InterventionQueryRecord":
        if resolved.operation == "WHAT":
            return self._exec_what(resolved)
        if resolved.operation in {"REMOVE", "ADD", "WHATIF"}:
            return self._exec_intervention(resolved)
        if resolved.operation == "DONE":
            return FactualQueryRecord(
                generated_query=resolved.raw_text,
                resolved=resolved,
                signature=QuerySignature.from_resolved(resolved),
            )
        raise QueryValidationError(f"Cannot execute {resolved.operation}")

    # -- WHAT (factual only, no replay) ---------------------------------

    def _exec_what(self, resolved: ResolvedQuery) -> FactualQueryRecord:
        assert resolved.target_ref is not None
        ref = resolved.target_ref
        registry = self.context.registry
        branch_id = self.context.explained_case.branch_id

        env = registry.get_by_ref(ref, branch_id=branch_id) or registry.get_by_ref(ref)
        if env is None:
            return FactualQueryRecord(
                generated_query=resolved.raw_text,
                resolved=resolved,
                error=f"Unknown artifact {ref}",
                signature=QuerySignature.from_resolved(resolved),
            )

        # Producer occurrence
        producer_occ_id = None
        case_slice = self.context.case_slice
        slice_occs = getattr(case_slice, "relevant_occurrences", ())
        for occ in slice_occs:
            if ref in getattr(occ, "produced_artifacts", ()):
                producer_occ_id = getattr(occ, "labeled_id", None) or getattr(
                    occ, "occurrence_id", None
                )
                break

        # Consumer relations (restricted to case slice)
        slice_rels = getattr(case_slice, "relevant_relations", ())
        consumer_rels = tuple(
            r for r in slice_rels if getattr(r, "artifact_ref", None) == ref
        )

        # Derivations
        slice_derivs = getattr(case_slice, "relevant_derivations", ())
        derivations = tuple(
            d
            for d in slice_derivs
            if getattr(d, "previous_artifact_ref", None) == ref
            or getattr(d, "new_artifact_ref", None) == ref
        )

        # Allowed interventions
        a_type = env.artifact_type
        allowed: list[str] = []
        contract = self.context.contracts.get(a_type)
        if contract is not None:
            for op in ("REMOVE", "ADD", "REPLACE"):
                if contract.allows(op):
                    allowed.append(op)

        result = FactualLookupResult(
            target_ref=ref,
            artifact_type=a_type,
            producer_occurrence_id=producer_occ_id,
            consumer_relations=consumer_rels,
            derivations=derivations,
            allowed_interventions=tuple(allowed),
            envelope=env,
        )
        return FactualQueryRecord(
            generated_query=resolved.raw_text,
            resolved=resolved,
            result=result,
            signature=QuerySignature.from_resolved(resolved),
        )

    # -- Intervention (REMOVE/ADD/WHATIF → real replay) -----------------

    def _get_or_run_noop(
        self, *, run_id: str, ckpt: Any
    ) -> Tuple[Any, bool, Optional[str]]:
        """Return ``(noop_branch, reused, cache_key_str)``.

        NOOP is always full-to-END so one cached control branch can be observed
        for every explained target from the same checkpoint. Target-bounded
        early-stop applies only to counterfactual forks.
        """
        ckpt_id = str(getattr(ckpt, "checkpoint_id", "") or "")
        alias = str(getattr(ckpt, "alias", "") or ckpt_id)
        # Always full — never cache a target-bounded NOOP under this key.
        cfg = _replay_configuration_id(self.replay_executor, mode="full")
        timing, target, round_idx = _timing_context()

        def _factory() -> Any:
            return self.replay_executor.fork_noop(
                run_id,
                BoundarySelector.by_alias(ckpt.alias),
                explained_case=self.context.explained_case,
                stop_when_observable=False,
            )

        if timing is not None:
            _print_replay_start(
                target=target,
                round_idx=round_idx,
                branch="noop",
                checkpoint=alias,
                bounded_requested=False,
            )
        t0 = time.perf_counter()
        noop: Any = None
        reused = False
        key_s: Optional[str] = None
        try:
            if self.noop_cache is not None and hasattr(self.noop_cache, "get_or_run"):
                noop, reused, key = self.noop_cache.get_or_run(
                    run_id=run_id,
                    checkpoint_id=ckpt_id,
                    factory=_factory,
                    replay_configuration=cfg,
                    checkpoint_alias=alias,
                )
                key_s = str(key)
            else:
                # Fallback without cache (legacy / tests).
                noop = _factory()
                reused, key_s = False, None
        finally:
            duration = time.perf_counter() - t0
        if noop is not None and timing is not None:
            metrics = _branch_replay_metrics(noop)
            _print_replay_end(
                target=target,
                round_idx=round_idx,
                branch="noop",
                duration_s=duration,
                metrics=metrics,
                cache_hit=bool(reused),
            )
            timing.record_replay(
                branch="noop",
                duration_s=duration,
                checkpoint=ckpt_id,
                checkpoint_alias=alias,
                bounded_requested=False,
                cache_hit=bool(reused),
                completion_mode=metrics["completion_mode"],
                stopped_after_node=metrics["stopped_after_node"],
                executed_node_count=metrics["executed_node_count"],
                logical_llm_requests=metrics["logical_llm_requests"],
                factual_response_hits=metrics["factual_response_hits"],
                counterfactual_memo_hits=metrics["counterfactual_memo_hits"],
                live_provider_calls=metrics["live_provider_calls"],
                full_remaining_nodes_avoided=metrics["full_remaining_nodes_avoided"],
            )
            cache_profile = {}
            if self.noop_cache is not None:
                cache_profile = dict(getattr(self.noop_cache, "last_profile", None) or {})
            cache_profile.setdefault("cache_hit", bool(reused))
            cache_profile.setdefault("graph_executed", not bool(reused))
            cache_profile.setdefault("checkpoint", ckpt_id)
            cache_profile.setdefault("checkpoint_alias", alias)
            cache_profile["total_noop_s"] = float(
                cache_profile.get("total_noop_s") or duration
            )
            timing.record_noop_cache_profile(cache_profile)
        return noop, reused, key_s

    def _exec_intervention(
        self, resolved: ResolvedQuery
    ) -> InterventionQueryRecord:
        assert resolved.target_ref is not None
        alias = resolved.checkpoint_alias
        assert alias is not None

        try:
            ckpt = self.context.checkpoint_index.get_alias(alias)
        except Exception as exc:
            return InterventionQueryRecord(
                generated_query=resolved.raw_text,
                resolved=resolved,
                error=str(exc),
                signature=QuerySignature.from_resolved(resolved),
            )

        run_id = self.context.explained_case.run_id

        # Shared NOOP control for this run + checkpoint (+ replay config).
        noop, noop_reused, noop_cache_key = self._get_or_run_noop(
            run_id=run_id, ckpt=ckpt
        )

        # Build intervention
        if resolved.operation == "REMOVE":
            intervention = ArtifactIntervention(
                operation="REMOVE",
                artifact=resolved.target_ref,
                checkpoint=ckpt,
            )
        elif resolved.operation == "ADD":
            intervention = ArtifactIntervention(
                operation="ADD",
                artifact=resolved.target_ref,
                checkpoint=ckpt,
            )
        elif resolved.operation == "WHATIF":
            assert resolved.replacement_ref is not None
            intervention = ArtifactIntervention(
                operation="REPLACE",
                artifact=(resolved.target_ref, resolved.replacement_ref),
                checkpoint=ckpt,
            )
        else:
            return InterventionQueryRecord(
                generated_query=resolved.raw_text,
                resolved=resolved,
                error=f"Unsupported intervention {resolved.operation}",
                signature=QuerySignature.from_resolved(resolved),
                noop_branch_id=getattr(noop, "branch_id", None),
                noop_result=noop,
                noop_reused=noop_reused,
                noop_cache_key=noop_cache_key,
            )

        # Use intervention_registry (not factual) for ADD/REPLACE
        int_reg = self.context.intervention_registry
        fact_reg = self.context.registry

        try:
            timing, target, round_idx = _timing_context()
            alias_s = str(getattr(ckpt, "alias", "") or "")
            if timing is not None:
                _print_replay_start(
                    target=target,
                    round_idx=round_idx,
                    branch="counterfactual",
                    checkpoint=alias_s,
                    bounded_requested=bool(self.stop_when_observable),
                )
            t0 = time.perf_counter()
            cf: Any = None
            span_profile: Dict[str, float] = {}
            try:
                from ..experiment.timing import replay_profile_scope

                with replay_profile_scope() as spans:
                    cf = self.replay_executor.fork_with_intervention(
                        run_id,
                        intervention,
                        registry=fact_reg,
                        intervention_registry=int_reg,
                        explained_case=self.context.explained_case,
                        stop_when_observable=self.stop_when_observable,
                    )
                    span_profile = dict(spans)
            finally:
                cf_duration = time.perf_counter() - t0
            if cf is not None and timing is not None:
                metrics = _branch_replay_metrics(cf)
                _print_replay_end(
                    target=target,
                    round_idx=round_idx,
                    branch="counterfactual",
                    duration_s=cf_duration,
                    metrics=metrics,
                )
                timing.record_replay(
                    branch="counterfactual",
                    duration_s=cf_duration,
                    checkpoint=str(getattr(ckpt, "checkpoint_id", "") or ""),
                    checkpoint_alias=alias_s,
                    bounded_requested=bool(self.stop_when_observable),
                    cache_hit=None,
                    completion_mode=metrics["completion_mode"],
                    stopped_after_node=metrics["stopped_after_node"],
                    executed_node_count=metrics["executed_node_count"],
                    logical_llm_requests=metrics["logical_llm_requests"],
                    factual_response_hits=metrics["factual_response_hits"],
                    counterfactual_memo_hits=metrics["counterfactual_memo_hits"],
                    live_provider_calls=metrics["live_provider_calls"],
                    full_remaining_nodes_avoided=metrics[
                        "full_remaining_nodes_avoided"
                    ],
                )
                try:
                    from ..experiment.timing import assemble_replay_profile

                    attached = dict(getattr(cf, "replay_span_profile", None) or {})
                    attached.update(span_profile)
                    profile = assemble_replay_profile(
                        spans=attached,
                        total_s=cf_duration,
                        extra=attached,
                    )
                    timing.record_replay_profile(profile)
                    nodes = list(getattr(cf, "node_timings", None) or [])
                    if nodes:
                        timing.record_node_profile(nodes)
                except Exception:
                    pass
            if cf is None:
                raise RuntimeError("counterfactual replay returned no result")
            if self.noop_cache is not None and hasattr(
                self.noop_cache, "record_counterfactual_execution"
            ):
                self.noop_cache.record_counterfactual_execution()
            resolver = self.replay_executor.get_resolver(cf.branch_id)

            # Mechanical metadata (no causal interpretation)
            comparison = {
                "operation": resolved.operation,
                "target": str(resolved.target_ref),
                "checkpoint": ckpt.alias,
                "noop_branch": noop.branch_id,
                "counterfactual_branch": cf.branch_id,
                "noop_reused": noop_reused,
                "noop_cache_key": noop_cache_key,
            }
            if resolved.operation == "REMOVE":
                comparison["target_available_in_counterfactual"] = resolver.is_available(
                    resolved.target_ref
                )
            elif resolved.operation == "ADD":
                comparison["target_available_in_counterfactual"] = resolver.is_available(
                    resolved.target_ref
                )
            elif resolved.operation == "WHATIF":
                assert resolved.replacement_ref is not None
                comparison["resolver_resolves_old_to_new"] = (
                    resolver.resolve_ref(resolved.target_ref) == resolved.replacement_ref
                )

            return InterventionQueryRecord(
                generated_query=resolved.raw_text,
                resolved=resolved,
                noop_branch_id=noop.branch_id,
                counterfactual_branch_id=cf.branch_id,
                comparison=comparison,
                signature=QuerySignature.from_resolved(resolved),
                noop_result=noop,
                counterfactual_result=cf,
                noop_reused=noop_reused,
                noop_cache_key=noop_cache_key,
            )
        except Exception as exc:
            return InterventionQueryRecord(
                generated_query=resolved.raw_text,
                resolved=resolved,
                noop_branch_id=getattr(noop, "branch_id", None),
                error=str(exc),
                signature=QuerySignature.from_resolved(resolved),
                noop_result=noop,
                noop_reused=noop_reused,
                noop_cache_key=noop_cache_key,
            )
