"""Resolve artifact refs considering availability, blocks, and substitutions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .models import ArtifactEnvelope, ArtifactRef
from .registry import ArtifactRegistry


@dataclass
class ArtifactResolver:
    """Resolves artifact refs with branch-local availability semantics.

    - REMOVE blocks an artifact from all downstream resolution.
    - REPLACE maps old -> new for every downstream consumer.
    - Historical catalog entries remain queryable via the registry directly.
    """

    registry: ArtifactRegistry
    branch_id: str = "factual"
    available: Set[ArtifactRef] = field(default_factory=set)
    blocked: Set[ArtifactRef] = field(default_factory=set)
    substitutions: Dict[ArtifactRef, ArtifactRef] = field(default_factory=dict)
    _availability_enforced: bool = False

    def make_available(self, ref: ArtifactRef, *, override_block: bool = False) -> None:
        """Mark an artifact as available downstream.

        An explicit REMOVE/block wins over normal registration unless
        ``override_block=True`` (reserved for future ADD/unblock).
        """
        self._availability_enforced = True
        if ref in self.blocked and not override_block:
            return
        if override_block:
            self.blocked.discard(ref)
        self.available.add(ref)

    def unblock(self, ref: ArtifactRef) -> None:
        """Explicitly clear a REMOVE block (future ADD path)."""
        self._availability_enforced = True
        self.blocked.discard(ref)
        self.available.add(ref)

    def block(self, ref: ArtifactRef) -> None:
        """Block an artifact from downstream resolution (REMOVE)."""
        self._availability_enforced = True
        self.available.discard(ref)
        self.blocked.add(ref)

    def substitute(self, old: ArtifactRef, new: ArtifactRef) -> None:
        """REPLACE: every resolution of ``old`` returns ``new``."""
        self._availability_enforced = True
        self.substitutions[old] = new
        self.make_available(new, override_block=True)

    def _apply_substitutions(self, ref: ArtifactRef) -> ArtifactRef:
        current = ref
        seen: Set[ArtifactRef] = set()
        while current in self.substitutions:
            if current in seen:
                raise ValueError(f"Substitution cycle involving {current}")
            seen.add(current)
            current = self.substitutions[current]
        return current

    def resolve_ref(self, ref: ArtifactRef) -> Optional[ArtifactRef]:
        """Apply substitutions; return None if blocked or unknown."""
        current = self._apply_substitutions(ref)
        if current in self.blocked:
            return None
        if self.registry.get_by_ref(current, self.branch_id) is None:
            return None
        if self._availability_enforced and current not in self.available:
            return None
        return current

    def resolve(self, ref: ArtifactRef) -> Optional[ArtifactRef]:
        """Alias for ``resolve_ref``."""
        return self.resolve_ref(ref)

    def exists(self, ref: ArtifactRef) -> bool:
        return self.registry.exists(ref, self.branch_id)

    def get(self, ref: ArtifactRef) -> Optional[ArtifactEnvelope]:
        """Resolve and return the envelope, or None if unavailable."""
        resolved = self.resolve_ref(ref)
        if resolved is None:
            return None
        return self.registry.get_by_ref(resolved, self.branch_id)

    def is_available(self, ref: ArtifactRef) -> bool:
        return self.resolve_ref(ref) is not None

    def latest(self, logical_id: str) -> Optional[ArtifactEnvelope]:
        return self.registry.latest(logical_id, self.branch_id)

    def all_versions(self, logical_id: str) -> List[ArtifactEnvelope]:
        return self.registry.versions_of(logical_id, self.branch_id)

    def get_historical(self, ref: ArtifactRef) -> Optional[ArtifactEnvelope]:
        """Bypass availability — factual historical lookup only."""
        return self.registry.get_by_ref(ref, self.branch_id)
