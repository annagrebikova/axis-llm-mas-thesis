"""Artifact type declarations and immutable catalog storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from .models import ArtifactEnvelope, ArtifactRef
from .utils import payloads_equivalent


@dataclass
class ArtifactTypeDecl:
    """Declarative metadata for one artifact type."""

    name: str
    description: str = ""
    explanation_target: bool = False
    intervenable: bool = False
    serialization_adapter: Optional[str] = None


class ArtifactTypeRegistry:
    """Registry of known artifact type declarations."""

    def __init__(self, types: Optional[Iterable[ArtifactTypeDecl]] = None) -> None:
        self._types: Dict[str, ArtifactTypeDecl] = {}
        for t in types or ():
            self.register(t)

    def register(self, decl: ArtifactTypeDecl) -> None:
        """Register or replace a type declaration."""
        if not decl.name:
            raise ValueError("Artifact type name must be non-empty")
        self._types[decl.name] = decl

    def get(self, name: str) -> Optional[ArtifactTypeDecl]:
        return self._types.get(name)

    def require(self, name: str) -> ArtifactTypeDecl:
        decl = self.get(name)
        if decl is None:
            raise KeyError(f"Unknown artifact type: {name!r}")
        return decl

    def names(self) -> List[str]:
        return sorted(self._types)

    def __contains__(self, name: str) -> bool:
        return name in self._types

    def __len__(self) -> int:
        return len(self._types)


@dataclass
class ArtifactRegistry:
    """Immutable historical catalog of artifact envelopes.

    Blocking an artifact removes it from *availability*, not from this catalog.
    """

    catalog: Dict[str, ArtifactEnvelope] = field(default_factory=dict)
    # key: f"{branch_id}:{logical_id}:v{version}" -> instance_id
    _ref_index: Dict[str, str] = field(default_factory=dict)
    derived_from: List[Any] = field(default_factory=list)
    produced_by: List[Any] = field(default_factory=list)

    @staticmethod
    def ref_key(ref: ArtifactRef, branch_id: str = "factual") -> str:
        return f"{branch_id}:{ref.logical_id}:v{ref.version}"

    @classmethod
    def from_catalog(cls, catalog: Dict[str, ArtifactEnvelope]) -> "ArtifactRegistry":
        """Rebuild ref index from a checkpointed catalog mapping."""
        reg = cls(catalog=dict(catalog))
        reg._ref_index = {}
        for envelope in catalog.values():
            key = cls.ref_key(envelope.ref, envelope.branch_id)
            reg._ref_index[key] = envelope.instance_id
        return reg

    def store(self, envelope: ArtifactEnvelope) -> ArtifactEnvelope:
        """Store an immutable envelope.

        Idempotent when the same branch/ref already holds an equivalent payload.
        Raises when the same ref would acquire a different payload.
        """
        key = self.ref_key(envelope.ref, envelope.branch_id)
        if key in self._ref_index:
            existing = self.catalog[self._ref_index[key]]
            if payloads_equivalent(existing.payload, envelope.payload):
                return existing
            raise ValueError(
                f"Conflicting immutable artifact registration for {key}: "
                "payload differs from existing envelope"
            )
        self.catalog[envelope.instance_id] = envelope
        self._ref_index[key] = envelope.instance_id
        return envelope

    def get_by_ref(
        self,
        ref: ArtifactRef,
        branch_id: str = "factual",
    ) -> Optional[ArtifactEnvelope]:
        instance_id = self._ref_index.get(self.ref_key(ref, branch_id))
        if instance_id is None:
            return None
        return self.catalog.get(instance_id)

    def exists(self, ref: ArtifactRef, branch_id: str = "factual") -> bool:
        return self.get_by_ref(ref, branch_id) is not None

    def get_by_instance(self, instance_id: str) -> Optional[ArtifactEnvelope]:
        return self.catalog.get(instance_id)

    def versions_of(
        self,
        logical_id: str,
        branch_id: str = "factual",
    ) -> List[ArtifactEnvelope]:
        """Return all versions of a logical artifact in one branch, sorted by version."""
        found = [
            e
            for e in self.catalog.values()
            if e.ref.logical_id == logical_id and e.branch_id == branch_id
        ]
        return sorted(found, key=lambda e: e.ref.version)

    def latest(
        self,
        logical_id: str,
        branch_id: str = "factual",
    ) -> Optional[ArtifactEnvelope]:
        versions = self.versions_of(logical_id, branch_id)
        return versions[-1] if versions else None

    def mark_superseded(self, ref: ArtifactRef, branch_id: str = "factual") -> Optional[ArtifactEnvelope]:
        """Return a new superseded envelope; replace catalog entry."""
        old = self.get_by_ref(ref, branch_id)
        if old is None:
            return None
        marked = ArtifactEnvelope(
            instance_id=old.instance_id,
            ref=old.ref,
            artifact_type=old.artifact_type,
            payload=old.payload,
            branch_id=old.branch_id,
            created_in_run=old.created_in_run,
            superseded=True,
            metadata=dict(old.metadata),
        )
        self.catalog[old.instance_id] = marked
        return marked

    def all_refs(self, branch_id: Optional[str] = None) -> Set[ArtifactRef]:
        if branch_id is None:
            return {e.ref for e in self.catalog.values()}
        return {e.ref for e in self.catalog.values() if e.branch_id == branch_id}
