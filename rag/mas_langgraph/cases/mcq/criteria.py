"""MCQ outcome evaluator — executable Gamma / delta for candidate versions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ...artifacts.channel_state import load_artifact_channels
from ...artifacts.models import ArtifactRef
from ...core.outcomes import DecisionArtifact, OutcomeAssessment
from ...domains.quiz.acceptance_policy import (
    DEFAULT_CRITERION_ORDER,
    build_decision_artifact,
    build_gate_inputs,
    build_outcome_assessment,
    decide_acceptance,
    gate_inputs_from_criteria,
)


class McqOutcomeEvaluator:
    """Assess CandidateMCQ versions using the authoritative acceptance policy.

    Exact-version assessment loads immutable envelopes from ``artifact_catalog``.
    Decisions target an exact ``ArtifactRef``. ``candidate.status`` remains a
    compatibility projection updated by ``apply_acceptance_gate``.
    """

    def __init__(
        self,
        *,
        criteria_meta: Optional[Dict[str, Dict[str, Any]]] = None,
        criteria_order: Optional[Sequence[str]] = None,
        max_revisions: int = 2,
        require_cross_cove: bool = True,
    ) -> None:
        self.criteria_meta = criteria_meta or {}
        self.criteria_order = list(criteria_order) if criteria_order else list(
            DEFAULT_CRITERION_ORDER
        )
        # Drop unimplemented metadata-only entries from the active Gamma order.
        self.criteria_order = [
            name
            for name in self.criteria_order
            if (self.criteria_meta.get(name) or {}).get("implemented", True) is not False
        ]
        self.max_revisions = max_revisions
        self.require_cross_cove = require_cross_cove

    def assess(
        self,
        state: Dict[str, Any],
        artifact_ref: ArtifactRef,
        *,
        allow_operational_fallback: bool = False,
    ) -> OutcomeAssessment:
        """Build criterion values for one exact candidate version.

        When ``allow_operational_fallback`` is False (default), the candidate
        envelope must exist in the artifact catalog. Operational CooperativeState
        dictionaries are not used as a silent substitute for historical versions.
        """
        candidate, evidence, cove, blueprint = self._resolve_exact_inputs(
            state,
            artifact_ref,
            allow_operational_fallback=allow_operational_fallback,
        )
        correct_count = sum(
            1 for o in (getattr(candidate, "options", None) or []) if getattr(o, "is_correct", False)
        )
        inputs = build_gate_inputs(
            revision_count=int(getattr(candidate, "revision_count", 0) or 0),
            max_revisions=self.max_revisions,
            candidate_status=str(getattr(candidate, "status", "draft") or "draft"),
            flags=list(getattr(candidate, "flags", []) or []),
            evidence_report=evidence,
            cross_cove_report=cove,
            cross_cove_required=self.require_cross_cove,
            source_refs_present=bool(getattr(candidate, "source_refs", None)),
            correct_option_count=correct_count,
            blueprint=blueprint,
        )
        return build_outcome_assessment(
            artifact_ref,
            inputs,
            criteria_meta=self.criteria_meta,
            criteria_order=self.criteria_order,
        )

    def decide(self, assessment: OutcomeAssessment) -> DecisionArtifact:
        """Map assessment criteria to accept / revise / reject via decide_acceptance."""
        inputs = gate_inputs_from_criteria(assessment.criteria)
        gate = decide_acceptance(inputs)
        return build_decision_artifact(assessment, gate)

    def _resolve_exact_inputs(
        self,
        state: Dict[str, Any],
        artifact_ref: ArtifactRef,
        *,
        allow_operational_fallback: bool,
    ):
        registry = None
        try:
            registry, _ = load_artifact_channels(state)
        except Exception:
            registry = state.get("artifact_registry")

        candidate = None
        evidence = None
        cove = None
        blueprint = None
        branch_id = str(state.get("branch_id") or "factual")

        if registry is not None:
            env = registry.get_by_ref(artifact_ref, branch_id)
            if env is None:
                env = registry.get_by_ref(artifact_ref)
            if env is not None:
                candidate = env.payload
            evidence_env = registry.get_by_ref(
                ArtifactRef(f"evidence-report:{artifact_ref.logical_id}", artifact_ref.version),
                branch_id,
            ) or registry.get_by_ref(
                ArtifactRef(f"evidence-report:{artifact_ref.logical_id}", artifact_ref.version)
            )
            if evidence_env is not None:
                evidence = evidence_env.payload
            cove_env = registry.get_by_ref(
                ArtifactRef(f"cross-cove-report:{artifact_ref.logical_id}", artifact_ref.version),
                branch_id,
            ) or registry.get_by_ref(
                ArtifactRef(f"cross-cove-report:{artifact_ref.logical_id}", artifact_ref.version)
            )
            if cove_env is not None:
                cove = cove_env.payload

        cs = state.get("cooperative_state") or state.get("state")
        if candidate is None:
            if not allow_operational_fallback:
                raise LookupError(
                    f"Candidate envelope {artifact_ref} not found in artifact_catalog "
                    "(exact-version assessment requires catalog history)."
                )
            if cs is None:
                raise LookupError(f"No candidate available for {artifact_ref}")
            candidate = getattr(cs, "candidate_mcqs", {}).get(artifact_ref.logical_id)
            if candidate is None:
                raise LookupError(f"No candidate available for {artifact_ref}")

        blueprint_id = getattr(candidate, "blueprint_id", None)
        if blueprint_id and registry is not None:
            bp_env = registry.get_by_ref(ArtifactRef(str(blueprint_id), 1), branch_id) or registry.get_by_ref(
                ArtifactRef(str(blueprint_id), 1)
            )
            if bp_env is not None:
                blueprint = bp_env.payload
        if blueprint is None and cs is not None and blueprint_id:
            blueprint = getattr(cs, "mcq_blueprints", {}).get(blueprint_id)

        if allow_operational_fallback and cs is not None:
            if evidence is None:
                evidence = getattr(cs, "evidence_reports", {}).get(artifact_ref.logical_id)
            if cove is None:
                cove = getattr(cs, "cross_cove_reports", {}).get(artifact_ref.logical_id)

        return candidate, evidence, cove, blueprint
