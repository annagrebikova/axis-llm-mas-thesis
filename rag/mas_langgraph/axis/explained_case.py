"""Generic explained-case resolution protocol (case-neutral)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..artifacts.models import ArtifactRef
from .models import ExplainedCase


@runtime_checkable
class ExplainedCaseResolver(Protocol):
    """Resolve exact ExplainedCase identities from a factual WorkflowRunResult."""

    def list_explainable_artifacts(
        self,
        *,
        factual_run_result: Any,
    ) -> Sequence[ArtifactRef]:
        ...

    def resolve(
        self,
        *,
        factual_run_result: Any,
        artifact_ref: ArtifactRef,
    ) -> ExplainedCase:
        ...
