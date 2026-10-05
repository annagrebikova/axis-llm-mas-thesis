"""MCQ provenance instrumentation adapter (case-specific)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ...artifacts.models import ArtifactRef
from ...artifacts.registry import ArtifactRegistry
from ...artifacts.resolver import ArtifactResolver
from .provenance import compact_state_delta, input_bindings_for_node


class MCQProvenanceInstrumentationAdapter:
    """Delegates to existing MCQ provenance binding helpers."""

    def input_bindings_for_node(
        self,
        node_id: str,
        state: Mapping[str, Any],
        registry: ArtifactRegistry,
        *,
        branch_id: str,
        resolver: ArtifactResolver,
    ) -> Sequence[Any]:
        return input_bindings_for_node(
            node_id, state, registry, branch_id=branch_id, resolver=resolver
        )

    def snapshot_state_summary(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        cs = state.get("cooperative_state")
        if cs is not None and hasattr(cs, "snapshot_counts"):
            try:
                return dict(cs.snapshot_counts())
            except Exception:
                return {}
        return {}

    def compact_state_delta(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        new_artifact_refs: Sequence[ArtifactRef],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        return compact_state_delta(
            dict(before),
            dict(after),
            new_artifact_refs=new_artifact_refs,
            extra=dict(extra) if extra else None,
        )
