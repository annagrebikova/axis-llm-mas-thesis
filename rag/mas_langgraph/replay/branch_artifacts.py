"""Rebase factual artifact catalogs into a logical replay branch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from ..artifacts.channel_state import dump_artifact_channels, load_artifact_channels
from ..artifacts.models import ArtifactEnvelope, ArtifactRef
from ..artifacts.registry import ArtifactRegistry
from ..artifacts.resolver import ArtifactResolver
from ..artifacts.utils import payloads_equivalent

SOURCE_ORIGIN = "origin_checkpoint"
SOURCE_FACTUAL = "factual_catalog"
SOURCE_INTERVENTION = "intervention_catalog"


@dataclass(frozen=True)
class InterventionArtifactSource:
    """Resolved immutable envelope from an approved intervention source."""

    envelope: ArtifactEnvelope
    source_kind: str
    source_branch: str
    source_instance_id: str


def _envelope_for_ref(
    catalog: Mapping[str, Any],
    ref: ArtifactRef,
) -> Optional[ArtifactEnvelope]:
    for env in (catalog or {}).values():
        eref = getattr(env, "ref", None)
        if eref is not None and eref == ref:
            return env  # type: ignore[return-value]
    return None


def _collect_source_candidates(
    *,
    ref: ArtifactRef,
    origin_values: Mapping[str, Any],
    factual_final_state: Mapping[str, Any],
    intervention_registry: Optional[ArtifactRegistry],
) -> List[InterventionArtifactSource]:
    found: List[InterventionArtifactSource] = []
    origin_env = _envelope_for_ref(origin_values.get("artifact_catalog") or {}, ref)
    if origin_env is not None:
        found.append(
            InterventionArtifactSource(
                envelope=origin_env,
                source_kind=SOURCE_ORIGIN,
                source_branch=str(
                    getattr(origin_env, "branch_id", "factual") or "factual"
                ),
                source_instance_id=str(origin_env.instance_id),
            )
        )
    factual_env = _envelope_for_ref(
        factual_final_state.get("artifact_catalog") or {}, ref
    )
    if factual_env is not None:
        found.append(
            InterventionArtifactSource(
                envelope=factual_env,
                source_kind=SOURCE_FACTUAL,
                source_branch=str(
                    getattr(factual_env, "branch_id", "factual") or "factual"
                ),
                source_instance_id=str(factual_env.instance_id),
            )
        )
    if intervention_registry is not None:
        for env in intervention_registry.catalog.values():
            if getattr(env, "ref", None) == ref:
                found.append(
                    InterventionArtifactSource(
                        envelope=env,
                        source_kind=SOURCE_INTERVENTION,
                        source_branch=str(
                            getattr(env, "branch_id", "intervention") or "intervention"
                        ),
                        source_instance_id=str(env.instance_id),
                    )
                )
                break
    return found


def resolve_intervention_source_envelope(
    *,
    ref: ArtifactRef,
    origin_values: Mapping[str, Any],
    factual_final_state: Mapping[str, Any],
    intervention_registry: Optional[ArtifactRegistry] = None,
) -> InterventionArtifactSource:
    """Resolve ``ref`` from approved catalogs with deterministic priority.

    Priority: origin checkpoint → factual run catalog → intervention registry.

    Raises ``ValueError`` when missing or when the same ref has non-equivalent
    payloads across sources that contain it.
    """
    candidates = _collect_source_candidates(
        ref=ref,
        origin_values=origin_values,
        factual_final_state=factual_final_state,
        intervention_registry=intervention_registry,
    )
    if not candidates:
        raise ValueError(
            f"Artifact {ref} not found in origin checkpoint, factual catalog, "
            "or controlled intervention registry"
        )
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            if not payloads_equivalent(left.envelope.payload, right.envelope.payload):
                raise ValueError(
                    f"Conflicting payloads for {ref} across intervention sources "
                    f"({left.source_kind} vs {right.source_kind})"
                )
    priority = {SOURCE_ORIGIN: 0, SOURCE_FACTUAL: 1, SOURCE_INTERVENTION: 2}
    candidates.sort(key=lambda c: priority.get(c.source_kind, 99))
    return candidates[0]


def validate_no_substitution_cycle(
    substitutions: Mapping[ArtifactRef, ArtifactRef],
) -> None:
    """Reject substitution maps that contain cycles before replay."""
    for start in substitutions:
        seen: Set[ArtifactRef] = set()
        current = start
        while current in substitutions:
            if current in seen:
                raise ValueError(
                    f"Substitution cycle involving {start} "
                    f"(map={dict(substitutions)!r})"
                )
            seen.add(current)
            current = substitutions[current]


def clone_envelope_for_branch(
    env: ArtifactEnvelope,
    *,
    run_id: str,
    new_branch_id: str,
    metadata_extra: Optional[Mapping[str, Any]] = None,
) -> ArtifactEnvelope:
    """Create a branch-local envelope with the same ArtifactRef and payload."""
    meta = dict(env.metadata or {})
    meta.setdefault("inherited_from_branch", env.branch_id)
    meta.setdefault("origin_instance_id", env.instance_id)
    if metadata_extra:
        meta.update(dict(metadata_extra))
    rebased = ArtifactEnvelope.create(
        logical_id=env.ref.logical_id,
        artifact_type=env.artifact_type,
        payload=deepcopy(env.payload),
        branch_id=new_branch_id,
        created_in_run=run_id,
        version=env.ref.version,
        metadata=meta,
    )
    if env.superseded:
        rebased = dc_replace(rebased, superseded=True)
    return rebased


def rebase_artifact_channels_for_branch(
    state: Mapping[str, Any],
    *,
    run_id: str,
    new_branch_id: str,
    blocked: Optional[Set[ArtifactRef]] = None,
    additional_envelopes: Optional[Sequence[ArtifactEnvelope]] = None,
    additional_source_meta: Optional[Sequence[Mapping[str, Any]]] = None,
    explicit_available_refs: Optional[Set[ArtifactRef]] = None,
    substitutions: Optional[Mapping[ArtifactRef, ArtifactRef]] = None,
) -> Dict[str, Any]:
    """Clone envelopes present at the fork into ``new_branch_id``.

    Inherited artifacts keep the same ``ArtifactRef`` and payload. They are
    **not** newly produced. Optional ``blocked`` refs remain in the historical
    catalog. ``additional_envelopes`` are imported without becoming ProducedBy.
    """
    blocked = set(blocked or ())
    explicit_available_refs = set(explicit_available_refs or ())
    registry, resolver = load_artifact_channels(dict(state))
    source_branch = str(state.get("branch_id") or "factual")

    new_registry = ArtifactRegistry()
    new_registry.derived_from = list(registry.derived_from)
    new_registry.produced_by = list(getattr(registry, "produced_by", []) or [])

    available: Set[ArtifactRef] = set()
    blocked_out: Set[ArtifactRef] = set(blocked)
    new_substitutions: Dict[ArtifactRef, ArtifactRef] = dict(resolver.substitutions)
    if substitutions:
        new_substitutions.update(dict(substitutions))
    validate_no_substitution_cycle(new_substitutions)

    for env in list(registry.catalog.values()):
        if env.branch_id not in {source_branch, "factual", new_branch_id}:
            continue
        rebased = clone_envelope_for_branch(
            env, run_id=run_id, new_branch_id=new_branch_id
        )
        new_registry.store(rebased)
        if env.ref in blocked or rebased.ref in blocked:
            blocked_out.add(rebased.ref)
        else:
            available.add(rebased.ref)

    extras = list(additional_envelopes or ())
    metas = list(additional_source_meta or ())
    for idx, env in enumerate(extras):
        extra_meta = dict(metas[idx]) if idx < len(metas) else {}
        if new_registry.get_by_ref(env.ref, new_branch_id) is not None:
            available.add(env.ref)
            blocked_out.discard(env.ref)
            continue
        rebased = clone_envelope_for_branch(
            env,
            run_id=run_id,
            new_branch_id=new_branch_id,
            metadata_extra=extra_meta,
        )
        new_registry.store(rebased)
        available.add(rebased.ref)
        blocked_out.discard(rebased.ref)

    for ref in explicit_available_refs:
        available.add(ref)
        blocked_out.discard(ref)

    new_resolver = ArtifactResolver(
        registry=new_registry,
        branch_id=new_branch_id,
        available=set(available) - blocked_out,
        blocked=set(blocked_out),
        substitutions=dict(new_substitutions),
        _availability_enforced=True,
    )
    for ref in blocked_out:
        new_resolver.block(ref)
    for ref in explicit_available_refs:
        new_resolver.make_available(ref, override_block=True)
    for old, new in new_substitutions.items():
        new_resolver.substitutions[old] = new
        new_resolver.make_available(new, override_block=True)

    validate_no_substitution_cycle(new_resolver.substitutions)
    return dump_artifact_channels(new_registry, new_resolver)


def artifact_exists_at_checkpoint(
    snapshot_values: Mapping[str, Any],
    ref: ArtifactRef,
) -> bool:
    """Return True when ``ref`` is present in the checkpoint artifact catalog."""
    return _envelope_for_ref(snapshot_values.get("artifact_catalog") or {}, ref) is not None


def envelope_type_at_checkpoint(
    snapshot_values: Mapping[str, Any],
    ref: ArtifactRef,
) -> Optional[str]:
    env = _envelope_for_ref(snapshot_values.get("artifact_catalog") or {}, ref)
    if env is None:
        return None
    return str(getattr(env, "artifact_type", "") or "")


def envelope_at_checkpoint(
    snapshot_values: Mapping[str, Any],
    ref: ArtifactRef,
) -> Optional[ArtifactEnvelope]:
    return _envelope_for_ref(snapshot_values.get("artifact_catalog") or {}, ref)


def intervention_supplied_refs(state: Mapping[str, Any]) -> Set[ArtifactRef]:
    """Refs explicitly supplied by branch intervention events."""
    out: Set[ArtifactRef] = set()
    for ev in state.get("intervention_events") or []:
        for ref in getattr(ev, "supplied_refs", ()) or ():
            out.add(ref)
    catalog = state.get("artifact_catalog") or {}
    for env in catalog.values():
        meta = getattr(env, "metadata", None) or {}
        if meta.get("supplied_by_intervention"):
            ref = getattr(env, "ref", None)
            if ref is not None:
                out.add(ref)
    return out
