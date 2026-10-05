"""Checkpointed artifact channel helpers — branch-local, copy-safe."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, MutableMapping, Optional, Set, Tuple

from .models import ArtifactEnvelope, ArtifactRef
from .registry import ArtifactRegistry
from .resolver import ArtifactResolver


def empty_artifact_channels() -> Dict[str, Any]:
    """Default checkpointed artifact fields for a new workflow branch."""
    return {
        "artifact_catalog": {},
        "available_artifact_refs": set(),
        "blocked_artifact_refs": set(),
        "artifact_substitutions": {},
        "artifact_derivations": [],
        "intervention_events": [],
    }


def ensure_artifact_channels(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Ensure graph/checkpoint state contains artifact channel defaults."""
    merged = dict(state)
    defaults = empty_artifact_channels()
    for key, default in defaults.items():
        merged.setdefault(key, deepcopy(default))
    return merged


def clone_artifact_channels_dict(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Deep-copy artifact channels from graph state for a node transition."""
    catalog = state.get("artifact_catalog") or {}
    derivations = state.get("artifact_derivations") or []
    return {
        "artifact_catalog": deepcopy(catalog),
        "available_artifact_refs": set(state.get("available_artifact_refs") or set()),
        "blocked_artifact_refs": set(state.get("blocked_artifact_refs") or set()),
        "artifact_substitutions": dict(state.get("artifact_substitutions") or {}),
        "artifact_derivations": list(derivations),
        "intervention_events": list(state.get("intervention_events") or []),
    }


def load_artifact_channels(
    state: MutableMapping[str, Any],
) -> Tuple[ArtifactRegistry, ArtifactResolver]:
    """Build registry/resolver views over checkpointed artifact channels."""
    channels = clone_artifact_channels_dict(state)
    catalog = channels["artifact_catalog"]
    if isinstance(catalog, list):
        # Defensive: rebuild mapping from envelope instance ids
        catalog = {
            getattr(e, "instance_id", str(i)): e for i, e in enumerate(catalog)
        }
        channels["artifact_catalog"] = catalog
    registry = ArtifactRegistry.from_catalog(catalog if isinstance(catalog, dict) else {})
    registry.derived_from = list(channels["artifact_derivations"])
    branch_id = str(state.get("branch_id") or "factual")
    subs = channels["artifact_substitutions"]
    if isinstance(subs, list):
        subs = dict(subs) if subs and isinstance(subs[0], (tuple, list)) else {}
    elif not isinstance(subs, dict):
        subs = {}
    resolver = ArtifactResolver(
        registry=registry,
        branch_id=branch_id,
        available=set(channels["available_artifact_refs"]),
        blocked=set(channels["blocked_artifact_refs"]),
        substitutions=dict(subs),
    )
    return registry, resolver


def dump_artifact_channels(
    registry: ArtifactRegistry,
    resolver: ArtifactResolver,
    *,
    intervention_events: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Serialize registry/resolver back into checkpointed graph fields."""
    out = {
        "artifact_catalog": dict(registry.catalog),
        "available_artifact_refs": set(resolver.available),
        "blocked_artifact_refs": set(resolver.blocked),
        "artifact_substitutions": dict(resolver.substitutions),
        "artifact_derivations": list(registry.derived_from),
    }
    if intervention_events is not None:
        out["intervention_events"] = list(intervention_events)
    return out

