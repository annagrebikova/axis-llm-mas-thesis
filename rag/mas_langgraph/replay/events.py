"""Intervention experimental records — not node occurrences, not D_run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ..artifacts.models import ArtifactRef


@dataclass(frozen=True)
class InterventionEvent:
    """Immutable record of the experimental manipulation that created a branch.

    Not a node occurrence and not a ``RunProvenanceRelation``. Records that an
    artifact was supplied or blocked by an AXIS intervention.
    """

    intervention_id: str
    run_id: str
    branch_id: str
    operation: str
    origin_checkpoint_alias: str
    origin_checkpoint_id: str
    artifact_refs: Tuple[ArtifactRef, ...]
    supplied_refs: Tuple[ArtifactRef, ...] = ()
    source_kinds: Tuple[str, ...] = ()
    description: str = ""
