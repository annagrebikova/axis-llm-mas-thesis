"""MCQ explained-case resolution for AXIS."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from ...artifacts.models import ArtifactRef
from ...axis.models import ExplainedCase
from .slicing import resolve_mcq_explained_case


class MCQExplainedCaseResolver:
    """List and resolve exact CandidateMCQ versions that are fully explainable."""

    def list_explainable_artifacts(
        self,
        *,
        factual_run_result: Any,
    ) -> Tuple[ArtifactRef, ...]:
        registry = getattr(factual_run_result, "artifact_registry", None)
        if registry is None:
            return ()
        branch_id = getattr(factual_run_result, "branch_id", "factual") or "factual"
        run_id = getattr(factual_run_result, "run_id", "")
        candidates: List[ArtifactRef] = []
        for env in (getattr(registry, "catalog", {}) or {}).values():
            if getattr(env, "artifact_type", None) != "CandidateMCQ":
                continue
            if getattr(env, "branch_id", branch_id) != branch_id:
                continue
            ref = getattr(env, "ref", None)
            if ref is None:
                continue
            try:
                resolve_mcq_explained_case(
                    run_id=run_id,
                    branch_id=branch_id,
                    artifact_ref=ref,
                    registry=registry,
                )
            except Exception:
                continue
            candidates.append(ref)
        # Deterministic order: logical_id, version
        unique = {str(r): r for r in candidates}
        ordered = sorted(unique.values(), key=lambda r: (r.logical_id, r.version))
        return tuple(ordered)

    def resolve(
        self,
        *,
        factual_run_result: Any,
        artifact_ref: ArtifactRef,
    ) -> ExplainedCase:
        registry = getattr(factual_run_result, "artifact_registry", None)
        if registry is None:
            raise ValueError("factual_run_result has no artifact_registry")
        return resolve_mcq_explained_case(
            run_id=getattr(factual_run_result, "run_id", ""),
            branch_id=getattr(factual_run_result, "branch_id", "factual") or "factual",
            artifact_ref=artifact_ref,
            registry=registry,
        )

    def resolve_current(self, factual_run_result: Any, q_id: str) -> ExplainedCase:
        """Convenience: explain the highest version of ``q_id`` that is explainable."""
        refs = [
            r
            for r in self.list_explainable_artifacts(
                factual_run_result=factual_run_result
            )
            if r.logical_id == str(q_id)
        ]
        if not refs:
            raise ValueError(f"No explainable CandidateMCQ for q_id={q_id!r}")
        return self.resolve(factual_run_result=factual_run_result, artifact_ref=refs[-1])
