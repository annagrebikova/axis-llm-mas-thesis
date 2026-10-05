"""Immutable artifact identity models — case-agnostic."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from .utils import make_instance_id


@dataclass(frozen=True)
class ArtifactRef:
    """Logical identity of an artifact across revisions.

    ``logical_id`` is stable across versions; ``version`` identifies an immutable revision.
    """

    logical_id: str
    version: int = 1

    def __str__(self) -> str:
        return f"{self.logical_id}:v{self.version}"

    def next_version(self) -> "ArtifactRef":
        """Return a new ref with version incremented by one."""
        return ArtifactRef(logical_id=self.logical_id, version=self.version + 1)


@dataclass(frozen=True)
class ArtifactEnvelope:
    """Immutable wrapper around a case-specific payload.

    Generic infrastructure must only handle payloads through this envelope.
    """

    instance_id: str
    ref: ArtifactRef
    artifact_type: str
    payload: Any
    branch_id: str
    created_in_run: str
    superseded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        *,
        logical_id: str,
        artifact_type: str,
        payload: Any,
        branch_id: str,
        created_in_run: str,
        version: int = 1,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ArtifactEnvelope":
        """Construct a new envelope with deterministic instance_id and deep-copied payload."""
        ref = ArtifactRef(logical_id=logical_id, version=version)
        instance_id = make_instance_id(
            created_in_run=created_in_run,
            branch_id=branch_id,
            artifact_type=artifact_type,
            logical_id=logical_id,
            version=version,
        )
        return ArtifactEnvelope(
            instance_id=instance_id,
            ref=ref,
            artifact_type=artifact_type,
            payload=deepcopy(payload),
            branch_id=branch_id,
            created_in_run=created_in_run,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class ProducedBy:
    """Links an artifact to the occurrence that produced it."""

    artifact_ref: ArtifactRef
    producer_occurrence_id: str


@dataclass(frozen=True)
class DerivedFrom:
    """Links a revised artifact to its predecessor version."""

    new_artifact_ref: ArtifactRef
    previous_artifact_ref: ArtifactRef
