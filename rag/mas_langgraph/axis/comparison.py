"""Generic AXIS replay comparison layer.

Compares factual / no-op / counterfactual observations to produce
structured evidence without causal claims.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..artifacts.models import ArtifactRef
from ..core.outcomes import CriterionValue
from .models import ExplainedCase, QuerySignature


# ---------------------------------------------------------------------------
# Evidence statuses
# ---------------------------------------------------------------------------

DECISION_EFFECT = "decision_effect"
CRITERION_EFFECT = "criterion_effect"
DOWNSTREAM_EFFECT = "downstream_effect"
NO_DEMONSTRATED_EFFECT = "no_demonstrated_effect"
REPLAY_UNSTABLE = "replay_unstable"
NOOP_ERROR = "noop_error"
INTERVENTION_ERROR = "intervention_error"
OUTCOME_UNAVAILABLE = "outcome_unavailable"


# ---------------------------------------------------------------------------
# Run observation (generic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseRunObservation:
    """Observed outcome of one workflow branch for the explained case."""

    run_id: str
    branch_id: str

    subject_ref: Optional[ArtifactRef] = None

    assessment_ref: Optional[ArtifactRef] = None
    decision_ref: Optional[ArtifactRef] = None

    decision: Optional[str] = None
    criterion_values: Tuple[CriterionValue, ...] = ()

    decision_reasons: Tuple[str, ...] = ()
    required_fixes: Tuple[str, ...] = ()

    subject_payload: Any = None

    # Explicit subject/output presence in this branch (distinct from errors).
    subject_present: bool = True

    outcome_available: bool = True
    error: Optional[str] = None


@runtime_checkable
class CaseRunObservationAdapter(Protocol):
    """Case-specific adapter for extracting observations from workflow results."""

    def observe_factual(
        self,
        *,
        explained_case: ExplainedCase,
        case_slice: Any,
        registry: Any,
    ) -> CaseRunObservation: ...

    def observe_replay(
        self,
        *,
        explained_case: ExplainedCase,
        run_result: Any,  # WorkflowRunResult
    ) -> CaseRunObservation: ...


# ---------------------------------------------------------------------------
# Criterion delta
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubjectContentDelta:
    """Generic observable-output comparison (case-specific details in ``details``)."""

    kind: str = "subject_content"

    factual_available: bool = False
    noop_available: bool = False
    counterfactual_available: bool = False

    control_stable: Optional[bool] = None
    counterfactual_differs_from_noop: Optional[bool] = None

    effect_status: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "factual_available": self.factual_available,
            "noop_available": self.noop_available,
            "counterfactual_available": self.counterfactual_available,
            "control_stable": self.control_stable,
            "counterfactual_differs_from_noop": self.counterfactual_differs_from_noop,
            "effect_status": self.effect_status,
            "details": dict(self.details or {}),
        }


@dataclass(frozen=True)
class CriterionDelta:
    name: str

    factual_value: Any = None
    noop_value: Any = None
    counterfactual_value: Any = None

    decision_relevant: bool = False

    noop_differs_from_factual: bool = False
    counterfactual_differs_from_noop: bool = False


# ---------------------------------------------------------------------------
# Replay comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayComparison:
    """Structured evidence from one intervention experiment."""

    intervention_signature: QuerySignature

    factual: CaseRunObservation
    noop: CaseRunObservation
    counterfactual: CaseRunObservation

    criterion_deltas: Tuple[CriterionDelta, ...] = ()

    added_artifacts: Tuple[ArtifactRef, ...] = ()
    removed_artifacts: Tuple[ArtifactRef, ...] = ()
    payload_changed_artifacts: Tuple[ArtifactRef, ...] = ()

    noop_path: Tuple[str, ...] = ()
    counterfactual_path: Tuple[str, ...] = ()
    first_path_divergence: Optional[int] = None

    added_provenance_relations: Tuple[Any, ...] = ()
    removed_provenance_relations: Tuple[Any, ...] = ()

    decision_control_stable: Optional[bool] = None
    criteria_control_stable: Optional[bool] = None

    evidence_status: str = NO_DEMONSTRATED_EFFECT

    # Orthogonal observable-output comparison (case-specific payload in details).
    subject_content_delta: Optional[SubjectContentDelta] = None
    subject_content_control_stable: Optional[bool] = None
    intervention_before_subject_generation: Optional[bool] = None
    subject_generation_relation: Optional[str] = None

    notes: Tuple[str, ...] = ()
    counterfactual_llm_replay_diagnostics: Dict[str, Any] = field(default_factory=dict)
    noop_llm_replay_diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episodic history
# ---------------------------------------------------------------------------

@dataclass
class InterrogationRoundRecord:
    """One interrogation round in the AXIS loop."""

    round_index: int

    generated_query: str = ""
    resolved_query: Optional[Any] = None  # ResolvedQuery
    signature: Optional[QuerySignature] = None

    factual_record: Optional[Any] = None  # FactualQueryRecord
    intervention_record: Optional[Any] = None  # InterventionQueryRecord
    comparison: Optional[ReplayComparison] = None

    result_text: str = ""
    intermediate_explanation: str = ""

    validation_error: Optional[str] = None
    execution_error: Optional[str] = None


@dataclass
class AxisEpisodicHistory:
    """Complete episodic history H_N."""

    question: str = ""
    factual_context: str = ""
    rounds: List[InterrogationRoundRecord] = field(default_factory=list)

    def render(self, verbalizer: Any = None) -> str:
        """Render episodic history as human-readable text."""
        parts: list[str] = []
        parts.append(f"Question: {self.question}")
        parts.append("")
        for rd in self.rounds:
            parts.append(f"=== ROUND {rd.round_index} ===")
            if rd.generated_query:
                parts.append(f"Query: {rd.generated_query}")
            if rd.resolved_query:
                parts.append(f"Resolved: {rd.resolved_query}")
            if rd.validation_error:
                parts.append(f"Validation error: {rd.validation_error}")
            elif rd.execution_error:
                parts.append(f"Execution error: {rd.execution_error}")
            elif rd.result_text:
                parts.append(f"Result: {rd.result_text}")
            if rd.comparison:
                parts.append(f"Evidence status: {rd.comparison.evidence_status}")
                if rd.comparison.decision_control_stable is not None:
                    parts.append(
                        f"Control stable: {rd.comparison.decision_control_stable}"
                    )
            if rd.intermediate_explanation:
                parts.append(f"Intermediate synthesis: {rd.intermediate_explanation}")
            parts.append("")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# No-op replay cache (shared control per factual run + checkpoint)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoopReplayCacheKey:
    """Stable identity for a reusable NOOP control branch.

    Does not include intervention target/operation — multiple interventions from
    the same checkpoint share one control realization.
    """

    run_id: str
    checkpoint_id: str
    replay_configuration: str = ""

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.run_id, self.checkpoint_id, self.replay_configuration)

    def __str__(self) -> str:
        cfg = self.replay_configuration or "-"
        return f"{self.run_id}|{self.checkpoint_id}|{cfg}"


@dataclass
class NoopCacheEntry:
    """Cached NOOP branch plus researcher metadata."""

    key: NoopReplayCacheKey
    result: Any  # WorkflowRunResult
    checkpoint_alias: str = ""


@dataclass
class NoOpReplayCache:
    """Run-scoped cache: one NOOP branch per (run, checkpoint, replay config).

    Stores the full ``WorkflowRunResult`` (not only the decision). Observation of
    a specific explained case is performed separately via ``observe_replay``.
    """

    _cache: Dict[Tuple[str, str, str], NoopCacheEntry] = field(default_factory=dict)
    noop_replay_requests: int = 0
    noop_replay_executions: int = 0
    noop_cache_hits: int = 0
    counterfactual_replay_executions: int = 0
    last_profile: Dict[str, Any] = field(default_factory=dict)
    log: bool = True

    def clear(self) -> None:
        self._cache.clear()
        self.noop_replay_requests = 0
        self.noop_replay_executions = 0
        self.noop_cache_hits = 0
        self.counterfactual_replay_executions = 0

    def stats(self) -> Dict[str, int]:
        return {
            "noop_replay_requests": int(self.noop_replay_requests),
            "noop_replay_executions": int(self.noop_replay_executions),
            "noop_cache_hits": int(self.noop_cache_hits),
            "counterfactual_replay_executions": int(
                self.counterfactual_replay_executions
            ),
            "cached_entries": len(self._cache),
        }

    def make_key(
        self,
        *,
        run_id: str,
        checkpoint_id: str,
        replay_configuration: str = "",
    ) -> NoopReplayCacheKey:
        return NoopReplayCacheKey(
            run_id=str(run_id),
            checkpoint_id=str(checkpoint_id),
            replay_configuration=str(replay_configuration or ""),
        )

    def has_key(self, key: NoopReplayCacheKey) -> bool:
        return key.as_tuple() in self._cache

    def has(self, run_id: str, checkpoint_id: str, replay_configuration: str = "") -> bool:
        return self.has_key(
            self.make_key(
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                replay_configuration=replay_configuration,
            )
        )

    def get(
        self,
        run_id: str,
        checkpoint_id: str,
        replay_configuration: str = "",
    ) -> Optional[Any]:
        entry = self._cache.get(
            self.make_key(
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                replay_configuration=replay_configuration,
            ).as_tuple()
        )
        return entry.result if entry is not None else None

    def put(
        self,
        run_id: str,
        checkpoint_id: str,
        result: Any,
        *,
        replay_configuration: str = "",
        checkpoint_alias: str = "",
    ) -> NoopCacheEntry:
        """Store first-writer-wins. Deep-copies mutable final_state for safety."""
        key = self.make_key(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            replay_configuration=replay_configuration,
        )
        tup = key.as_tuple()
        if tup in self._cache:
            return self._cache[tup]
        frozen = _freeze_noop_result(result)
        entry = NoopCacheEntry(
            key=key, result=frozen, checkpoint_alias=str(checkpoint_alias or "")
        )
        self._cache[tup] = entry
        return entry

    def get_or_run(
        self,
        *,
        run_id: str,
        checkpoint_id: str,
        factory: Any,
        replay_configuration: str = "",
        checkpoint_alias: str = "",
    ) -> Tuple[Any, bool, NoopReplayCacheKey]:
        """Return ``(noop_result, reused, key)``.

        ``factory`` is called at most once per key and must return a full
        ``WorkflowRunResult`` from a real NOOP fork+replay.
        """
        import time as _time

        self.noop_replay_requests += 1
        t_all = _time.perf_counter()
        t0 = _time.perf_counter()
        key = self.make_key(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            replay_configuration=replay_configuration,
        )
        tup = key.as_tuple()
        lookup_s = _time.perf_counter() - t0
        alias = checkpoint_alias or checkpoint_id
        factory_s = 0.0
        clone_s = 0.0
        conversion_s = 0.0
        graph_executed = False
        if tup in self._cache:
            self.noop_cache_hits += 1
            if self.log:
                print(
                    f"[AXIS] Reusing cached NOOP replay for checkpoint {alias} "
                    f"(id={checkpoint_id})"
                )
            t_clone = _time.perf_counter()
            cloned = _clone_cached_noop_result(self._cache[tup].result)
            clone_s = _time.perf_counter() - t_clone
            self.last_profile = {
                "cache_hit": True,
                "graph_executed": False,
                "checkpoint": checkpoint_id,
                "checkpoint_alias": alias,
                "cache_lookup_s": lookup_s,
                "cache_factory_s": 0.0,
                "cached_result_clone_s": clone_s,
                "cached_result_conversion_s": 0.0,
                "cached_result_observation_s": 0.0,
                "provenance_processing_s": 0.0,
                "other_s": 0.0,
                "total_noop_s": _time.perf_counter() - t_all,
            }
            known = lookup_s + clone_s
            self.last_profile["other_s"] = max(
                0.0, float(self.last_profile["total_noop_s"]) - known
            )
            return cloned, True, key

        if self.log:
            print(
                f"[AXIS] Running NOOP replay for checkpoint {alias} "
                f"(id={checkpoint_id})"
            )
        t_factory = _time.perf_counter()
        result = factory()
        factory_s = _time.perf_counter() - t_factory
        graph_executed = True
        self.noop_replay_executions += 1
        t_conv = _time.perf_counter()
        self.put(
            run_id,
            checkpoint_id,
            result,
            replay_configuration=replay_configuration,
            checkpoint_alias=alias,
        )
        conversion_s = _time.perf_counter() - t_conv
        t_clone = _time.perf_counter()
        cloned = _clone_cached_noop_result(self._cache[tup].result)
        clone_s = _time.perf_counter() - t_clone
        self.last_profile = {
            "cache_hit": False,
            "graph_executed": graph_executed,
            "checkpoint": checkpoint_id,
            "checkpoint_alias": alias,
            "cache_lookup_s": lookup_s,
            "cache_factory_s": factory_s,
            "cached_result_clone_s": clone_s,
            "cached_result_conversion_s": conversion_s,
            "cached_result_observation_s": 0.0,
            "provenance_processing_s": 0.0,
            "other_s": 0.0,
            "total_noop_s": _time.perf_counter() - t_all,
        }
        known = lookup_s + factory_s + clone_s + conversion_s
        self.last_profile["other_s"] = max(
            0.0, float(self.last_profile["total_noop_s"]) - known
        )
        return cloned, False, key

    def record_counterfactual_execution(self) -> None:
        self.counterfactual_replay_executions += 1


def _clone_cached_noop_result(result: Any) -> Any:
    """Return an isolated NOOP result without cloning LangGraph histories.

    Downstream observation/comparison only reads ``final_state``. Isolation is
    preserved by deep-copying that mapping (and list/dict containers callers may
    mutate) while sharing large immutable checkpoint snapshots.
    """
    from dataclasses import fields, is_dataclass, replace

    if result is None:
        return None
    final_state = getattr(result, "final_state", None)
    frozen_state = deepcopy(final_state) if isinstance(final_state, dict) else deepcopy(final_state)
    if is_dataclass(result) and not isinstance(result, type):
        try:
            return replace(
                result,
                final_state=frozen_state,
                step_logs=list(getattr(result, "step_logs", None) or []),
                executed_nodes=list(getattr(result, "executed_nodes", None) or []),
                node_timings=[dict(x) for x in (getattr(result, "node_timings", None) or [])],
                llm_replay_diagnostics=dict(
                    getattr(result, "llm_replay_diagnostics", None) or {}
                ),
                node_profile_unattributed=dict(
                    getattr(result, "node_profile_unattributed", None) or {}
                ),
                replay_span_profile=dict(
                    getattr(result, "replay_span_profile", None) or {}
                ),
                branch_history=list(getattr(result, "branch_history", None) or []),
            )
        except Exception:
            pass
        try:
            cloned = object.__new__(type(result))
            for f in fields(result):
                value = getattr(result, f.name)
                if f.name == "final_state":
                    value = frozen_state
                elif f.name in {
                    "step_logs",
                    "executed_nodes",
                    "branch_history",
                }:
                    value = list(value or [])
                elif f.name in {
                    "node_timings",
                }:
                    value = [dict(x) for x in (value or [])]
                elif f.name in {
                    "llm_replay_diagnostics",
                    "node_profile_unattributed",
                    "replay_span_profile",
                }:
                    value = dict(value or {})
                setattr(cloned, f.name, value)
            return cloned
        except Exception:
            pass
    return deepcopy(result)


def _freeze_noop_result(result: Any) -> Any:
    """Defensive copy of mutable final_state so later CF/observe cannot mutate cache."""
    from dataclasses import fields, is_dataclass, replace

    if result is None:
        return None
    final_state = getattr(result, "final_state", None)
    if not isinstance(final_state, dict):
        return deepcopy(result)
    frozen_state = deepcopy(final_state)
    if is_dataclass(result) and not isinstance(result, type):
        try:
            return replace(result, final_state=frozen_state)
        except Exception:
            pass
        try:
            cloned = object.__new__(type(result))
            for f in fields(result):
                setattr(
                    cloned,
                    f.name,
                    frozen_state if f.name == "final_state" else getattr(result, f.name),
                )
            return cloned
        except Exception:
            pass
    try:
        cloned = object.__new__(type(result))
        cloned.__dict__.update(getattr(result, "__dict__", {}))
        cloned.final_state = frozen_state
        return cloned
    except Exception:
        return deepcopy(result)


# ---------------------------------------------------------------------------
# Execution ledger
# ---------------------------------------------------------------------------

@dataclass
class QueryExecutionLedger:
    """Tracks P_executed (successfully executed query signatures)."""

    _executed: List[QuerySignature] = field(default_factory=list)
    _executed_set: set = field(default_factory=set)

    def is_executed(self, sig: QuerySignature) -> bool:
        return sig in self._executed_set

    def mark_executed(self, sig: QuerySignature) -> None:
        if sig not in self._executed_set:
            self._executed_set.add(sig)
            self._executed.append(sig)

    @property
    def signatures(self) -> Tuple[QuerySignature, ...]:
        return tuple(self._executed)


# ---------------------------------------------------------------------------
# Evidence ledger
# ---------------------------------------------------------------------------

@dataclass
class ExplanationEvidenceLedger:
    """Structured evidence source for final synthesis."""

    factual_context: str = ""
    factual_records: List[Any] = field(default_factory=list)
    intervention_records: List[Any] = field(default_factory=list)
    comparisons: List[ReplayComparison] = field(default_factory=list)
    intermediate_explanations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers: normalize branch paths
# ---------------------------------------------------------------------------

def normalize_branch_path(occurrences: Sequence[Any]) -> Tuple[str, ...]:
    """Normalize branch occurrences to (node_id[local_count], ...) sequence."""
    counts: Dict[str, int] = {}
    path: list[str] = []
    for occ in sorted(occurrences, key=lambda o: getattr(o, "execution_index", 0)):
        node_id = getattr(occ, "node_id", "?")
        counts[node_id] = counts.get(node_id, 0) + 1
        path.append(f"{node_id}[{counts[node_id]}]")
    return tuple(path)


def normalize_provenance_relation(rel: Any) -> Tuple[str, ...]:
    """Normalize a provenance relation to branch-independent label tuple."""
    prod = getattr(rel, "producer_occurrence_id", "?")
    art = getattr(rel, "artifact_ref", "?")
    sem = getattr(rel, "relation", "?")
    cons = getattr(rel, "consumer_occurrence_id", "?")

    def _label(occ_id: str) -> str:
        # Extract node#N from run:branch:node#N
        parts = str(occ_id).split(":")
        return parts[-1] if parts else occ_id

    return (_label(prod), str(art), sem, _label(cons))


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

def _compute_criterion_deltas(
    factual: CaseRunObservation,
    noop: CaseRunObservation,
    counterfactual: CaseRunObservation,
) -> Tuple[CriterionDelta, ...]:
    """Match criteria by name and compute deltas."""
    fact_map = {c.name: c for c in factual.criterion_values}
    noop_map = {c.name: c for c in noop.criterion_values}
    cf_map = {c.name: c for c in counterfactual.criterion_values}
    all_names = sorted(set(fact_map) | set(noop_map) | set(cf_map))
    deltas: list[CriterionDelta] = []
    for name in all_names:
        fv = fact_map.get(name)
        nv = noop_map.get(name)
        cv = cf_map.get(name)
        f_val = fv.value if fv else None
        n_val = nv.value if nv else None
        c_val = cv.value if cv else None
        dr = (fv.decision_relevant if fv else False) or (nv.decision_relevant if nv else False)
        deltas.append(CriterionDelta(
            name=name,
            factual_value=f_val,
            noop_value=n_val,
            counterfactual_value=c_val,
            decision_relevant=dr,
            noop_differs_from_factual=(f_val != n_val),
            counterfactual_differs_from_noop=(n_val != c_val),
        ))
    return tuple(deltas)


def _compute_artifact_deltas(
    noop_result: Any,
    cf_result: Any,
) -> Tuple[Tuple[ArtifactRef, ...], Tuple[ArtifactRef, ...], Tuple[ArtifactRef, ...]]:
    """Compare artifact catalogs between noop and cf branches."""

    def _catalog_map(result: Any) -> Dict[ArtifactRef, Any]:
        catalog: Dict[ArtifactRef, Any] = {}
        state = getattr(result, "final_state", {}) or {}
        for env in (state.get("artifact_catalog") or {}).values():
            ref = getattr(env, "ref", None)
            if ref:
                catalog[ref] = getattr(env, "payload", None)
        return catalog

    noop_map = _catalog_map(noop_result)
    cf_map = _catalog_map(cf_result)

    added = tuple(sorted(set(cf_map) - set(noop_map), key=str))
    removed = tuple(sorted(set(noop_map) - set(cf_map), key=str))
    changed: list[ArtifactRef] = []
    for ref in sorted(set(noop_map) & set(cf_map), key=str):
        try:
            if noop_map[ref] != cf_map[ref]:
                changed.append(ref)
        except Exception:
            changed.append(ref)
    return added, removed, tuple(changed)


def _compute_path_delta(
    noop_result: Any,
    cf_result: Any,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Optional[int]]:
    noop_occs = getattr(noop_result, "branch_occurrences", [])
    cf_occs = getattr(cf_result, "branch_occurrences", [])
    noop_path = normalize_branch_path(noop_occs)
    cf_path = normalize_branch_path(cf_occs)
    divergence = None
    for i, (a, b) in enumerate(zip(noop_path, cf_path)):
        if a != b:
            divergence = i
            break
    else:
        if len(noop_path) != len(cf_path):
            divergence = min(len(noop_path), len(cf_path))
    return noop_path, cf_path, divergence


def _compute_provenance_delta(
    noop_result: Any,
    cf_result: Any,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    noop_rels = set()
    for r in getattr(noop_result, "branch_provenance_relations", []):
        noop_rels.add(normalize_provenance_relation(r))
    cf_rels = set()
    for r in getattr(cf_result, "branch_provenance_relations", []):
        cf_rels.add(normalize_provenance_relation(r))
    added = tuple(sorted(cf_rels - noop_rels))
    removed = tuple(sorted(noop_rels - cf_rels))
    return added, removed


def _determine_evidence_status(
    factual: CaseRunObservation,
    noop: CaseRunObservation,
    counterfactual: CaseRunObservation,
    criterion_deltas: Tuple[CriterionDelta, ...],
    noop_path: Tuple[str, ...],
    cf_path: Tuple[str, ...],
    added_artifacts: Tuple[ArtifactRef, ...],
    removed_artifacts: Tuple[ArtifactRef, ...],
    payload_changed: Tuple[ArtifactRef, ...],
    added_prov: Tuple[Any, ...],
    removed_prov: Tuple[Any, ...],
) -> Tuple[str, Optional[bool], Optional[bool]]:
    """Determine evidence status and control stability."""

    # Error cases
    if noop.error:
        return NOOP_ERROR, None, None
    if not getattr(noop, "subject_present", True):
        # NOOP must reproduce the corresponding explained output.
        return NOOP_ERROR, None, None
    # Observation inconsistency / execution error is not subject disappearance.
    if counterfactual.error:
        return INTERVENTION_ERROR, None, None
    if not noop.outcome_available:
        return OUTCOME_UNAVAILABLE, None, None

    # Counterfactual subject disappearance is a valid downstream/output effect.
    if not getattr(counterfactual, "subject_present", True):
        decision_stable = factual.decision == noop.decision
        criteria_stable = all(
            not d.noop_differs_from_factual
            for d in criterion_deltas
            if d.decision_relevant
        )
        if not criteria_stable or (
            factual.decision is not None
            and noop.decision is not None
            and not decision_stable
        ):
            return REPLAY_UNSTABLE, decision_stable, False
        return DOWNSTREAM_EFFECT, decision_stable, True

    if not counterfactual.outcome_available:
        return OUTCOME_UNAVAILABLE, None, None

    # Control stability
    decision_stable = (factual.decision == noop.decision)
    criteria_stable = all(
        not d.noop_differs_from_factual
        for d in criterion_deltas
        if d.decision_relevant
    )

    # If noop changed the decision from factual → replay_unstable
    if factual.decision is not None and noop.decision is not None and not decision_stable:
        return REPLAY_UNSTABLE, False, criteria_stable

    # Decision-relevant criteria must also match between factual and NOOP.
    # An unstable control must not support ordinary causal claims.
    if not criteria_stable:
        return REPLAY_UNSTABLE, decision_stable, False

    # Decision effect: factual==noop but cf differs
    if (
        factual.decision is not None
        and noop.decision is not None
        and counterfactual.decision is not None
        and decision_stable
        and counterfactual.decision != noop.decision
    ):
        return DECISION_EFFECT, True, True

    # Criterion effect: decisions match but criteria differ (CF vs NOOP)
    any_criterion_changed = any(d.counterfactual_differs_from_noop for d in criterion_deltas)
    if any_criterion_changed:
        return CRITERION_EFFECT, decision_stable, True

    # Downstream effect: path/artifacts/provenance differ
    has_downstream = bool(
        noop_path != cf_path
        or added_artifacts
        or removed_artifacts
        or payload_changed
        or added_prov
        or removed_prov
    )
    if has_downstream:
        return DOWNSTREAM_EFFECT, decision_stable, True

    return NO_DEMONSTRATED_EFFECT, decision_stable, True


class ReplayComparator:
    """Deterministic comparison of factual/noop/counterfactual observations.

    This class performs NO replay and NO LLM calls.
    """

    def compare(
        self,
        *,
        signature: QuerySignature,
        factual_observation: CaseRunObservation,
        noop_result: Any,  # WorkflowRunResult
        counterfactual_result: Any,  # WorkflowRunResult
        observation_adapter: CaseRunObservationAdapter,
        explained_case: ExplainedCase,
    ) -> ReplayComparison:
        import time as _time

        t_noop_obs = _time.perf_counter()
        noop_obs = observation_adapter.observe_replay(
            explained_case=explained_case,
            run_result=noop_result,
        )
        noop_obs_s = _time.perf_counter() - t_noop_obs
        t_cf_obs = _time.perf_counter()
        cf_obs = observation_adapter.observe_replay(
            explained_case=explained_case,
            run_result=counterfactual_result,
        )
        cf_obs_s = _time.perf_counter() - t_cf_obs

        criterion_deltas = _compute_criterion_deltas(factual_observation, noop_obs, cf_obs)
        added_arts, removed_arts, changed_arts = _compute_artifact_deltas(
            noop_result, counterfactual_result
        )
        noop_path, cf_path, divergence = _compute_path_delta(noop_result, counterfactual_result)
        t_prov = _time.perf_counter()
        added_prov, removed_prov = _compute_provenance_delta(noop_result, counterfactual_result)
        prov_s = _time.perf_counter() - t_prov
        try:
            from ..experiment.timing import add_replay_span, current_timing

            add_replay_span("final_observation", cf_obs_s)
            timing = current_timing()
            if timing is not None:
                timing.update_noop_cache_profile(
                    cached_result_observation_s=noop_obs_s,
                    provenance_processing_s=prov_s,
                )
        except Exception:
            pass

        status, dec_stable, crit_stable = _determine_evidence_status(
            factual_observation, noop_obs, cf_obs,
            criterion_deltas,
            noop_path, cf_path,
            added_arts, removed_arts, changed_arts,
            added_prov, removed_prov,
        )

        notes: list[str] = []
        if status == REPLAY_UNSTABLE:
            if dec_stable is False:
                notes.append(
                    "The unchanged replay changed the final decision without any "
                    "semantic intervention. Counterfactual decision difference cannot "
                    "be reliably attributed to the tested artifact."
                )
            elif crit_stable is False:
                notes.append(
                    "The unchanged replay changed one or more decision-relevant "
                    "assessment criteria without any semantic intervention. "
                    "Criterion-level counterfactual differences cannot be reliably "
                    "attributed to the tested artifact."
                )
            else:
                notes.append(
                    "The unchanged replay was unstable relative to the factual run "
                    "without any semantic intervention. Counterfactual differences "
                    "cannot be reliably attributed to the tested artifact."
                )

        subject_delta: Optional[SubjectContentDelta] = None
        gen_before: Optional[bool] = None
        gen_rel: Optional[str] = None
        compare_fn = getattr(observation_adapter, "compare_subject_content", None)
        if callable(compare_fn):
            try:
                # Optional timing hint from adapter.
                timing_fn = getattr(
                    observation_adapter, "infer_subject_generation_relation", None
                )
                if callable(timing_fn):
                    try:
                        gen_before, gen_rel = timing_fn(
                            signature=signature,
                            explained_case=explained_case,
                        )
                    except Exception:
                        gen_before, gen_rel = None, "unknown"
                subject_delta = compare_fn(
                    factual=factual_observation,
                    noop=noop_obs,
                    counterfactual=cf_obs,
                    intervention_before_subject_generation=gen_before,
                    subject_generation_relation=gen_rel,
                )
            except Exception as exc:
                notes.append(f"subject content comparison unavailable: {exc}")

        content_stable = None
        if subject_delta is not None:
            content_stable = subject_delta.control_stable

        return ReplayComparison(
            intervention_signature=signature,
            factual=factual_observation,
            noop=noop_obs,
            counterfactual=cf_obs,
            criterion_deltas=criterion_deltas,
            added_artifacts=added_arts,
            removed_artifacts=removed_arts,
            payload_changed_artifacts=changed_arts,
            noop_path=noop_path,
            counterfactual_path=cf_path,
            first_path_divergence=divergence,
            added_provenance_relations=added_prov,
            removed_provenance_relations=removed_prov,
            decision_control_stable=dec_stable,
            criteria_control_stable=crit_stable,
            evidence_status=status,
            subject_content_delta=subject_delta,
            subject_content_control_stable=content_stable,
            intervention_before_subject_generation=gen_before,
            subject_generation_relation=gen_rel,
            notes=tuple(notes),
            counterfactual_llm_replay_diagnostics=dict(
                getattr(counterfactual_result, "llm_replay_diagnostics", None) or {}
            ),
            noop_llm_replay_diagnostics=dict(
                getattr(noop_result, "llm_replay_diagnostics", None) or {}
            ),
        )
