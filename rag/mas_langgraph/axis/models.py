"""AXIS explanation case and query record models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..artifacts.models import ArtifactEnvelope, ArtifactRef, DerivedFrom, ProducedBy
from ..artifacts.registry import ArtifactRegistry


@dataclass(frozen=True)
class ExplainedCase:
    """The exact per-artifact decision case under explanation."""

    run_id: str
    artifact_ref: ArtifactRef
    branch_id: str
    decision_ref: ArtifactRef
    outcome: str
    assessment_ref: Optional[ArtifactRef] = None
    # Branch-independent correspondence key (e.g. MCQ blueprint_id).
    correspondence_key: Optional[str] = None

    def __post_init__(self) -> None:
        outcome = str(self.outcome or "").lower().strip()
        mapping = {
            "accepted": "accept",
            "accept": "accept",
            "needs_revision": "revise",
            "revise": "revise",
            "rejected": "reject",
            "reject": "reject",
        }
        canonical = mapping.get(outcome)
        if canonical is not None:
            object.__setattr__(self, "outcome", canonical)


# ---------------------------------------------------------------------------
# Resolved query and its canonical signature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedQuery:
    """Validated, resolved form of an LLM-generated AXIS query (p_n_resolved)."""

    operation: str
    raw_text: str
    target_ref: Optional[ArtifactRef] = None
    replacement_ref: Optional[ArtifactRef] = None
    checkpoint_alias: Optional[str] = None
    checkpoint_id: Optional[str] = None
    checkpoint_ns: Optional[str] = None
    boundary_raw: Optional[str] = None
    target_artifact_type: Optional[str] = None
    replacement_artifact_type: Optional[str] = None


@dataclass(frozen=True)
class QuerySignature:
    """Canonical deduplication key for a resolved query."""

    operation: str
    target_ref: Optional[ArtifactRef] = None
    replacement_ref: Optional[ArtifactRef] = None
    checkpoint_id: Optional[str] = None
    # Internal checkpoint alias (e.g. c_6); not part of dedup identity.
    checkpoint_alias: Optional[str] = field(default=None, compare=False, hash=False)
    # Human-readable semantic boundary (e.g. before atomize_evidence#1).
    boundary_raw: Optional[str] = field(default=None, compare=False, hash=False)

    @staticmethod
    def from_resolved(rq: ResolvedQuery) -> "QuerySignature":
        return QuerySignature(
            operation=rq.operation,
            target_ref=rq.target_ref,
            replacement_ref=rq.replacement_ref,
            # Prefer UUID for restoration / dedup; fall back to alias if UUID absent.
            checkpoint_id=rq.checkpoint_id or rq.checkpoint_alias,
            checkpoint_alias=rq.checkpoint_alias,
            boundary_raw=rq.boundary_raw,
        )


# ---------------------------------------------------------------------------
# Factual WHAT result (ρ_n)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactualLookupResult:
    """Structured factual lookup result for WHAT queries (ρ_n)."""

    target_ref: ArtifactRef
    artifact_type: str
    producer_occurrence_id: Optional[str] = None
    consumer_relations: Tuple[Any, ...] = ()
    derivations: Tuple[DerivedFrom, ...] = ()
    allowed_interventions: Tuple[str, ...] = ()
    envelope: Optional[ArtifactEnvelope] = None


# ---------------------------------------------------------------------------
# Query records (r_n)
# ---------------------------------------------------------------------------

@dataclass
class FactualQueryRecord:
    """Record of a factual WHAT lookup (r_n for WHAT)."""

    generated_query: str
    resolved: ResolvedQuery
    result: Any = None
    error: Optional[str] = None
    signature: Optional[QuerySignature] = None


@dataclass
class InterventionQueryRecord:
    """Record of REMOVE/ADD/WHATIF with factual, noop, and counterfactual traces."""

    generated_query: str
    resolved: ResolvedQuery
    factual_branch_id: str = "factual"
    noop_branch_id: Optional[str] = None
    counterfactual_branch_id: Optional[str] = None
    comparison: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    signature: Optional[QuerySignature] = None
    # In-memory trace handles (not serialized)
    noop_result: Any = None
    counterfactual_result: Any = None
    replay_comparison: Any = None  # Optional[ReplayComparison]
    # Researcher-only NOOP cache audit fields (not participant-facing).
    noop_reused: bool = False
    noop_cache_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Query context (in-memory, not checkpointed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisQueryContext:
    """In-memory explanation/query context for resolution and execution."""

    explained_case: ExplainedCase
    case_slice: Any  # CaseSlice (forward ref avoids circular)
    registry: ArtifactRegistry
    checkpoint_index: Any  # CheckpointIndex
    contracts: Any  # InterventionContractRegistry
    artifact_types: Any = None  # Optional[ArtifactTypeRegistry]
    intervention_registry: Optional[ArtifactRegistry] = None
    alias_map: Mapping[str, ArtifactRef] = field(default_factory=dict)
    intervention_alternatives: Mapping[str, ArtifactRef] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt template bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisPromptTemplates:
    """Case-specific prompt template bundle."""

    system: str = ""
    system_hybrid: str = ""
    factual_context: str = ""
    interrogation: str = ""
    interrogation_hybrid: str = ""
    intermediate_explanation: str = ""
    final_explanation: str = ""

    def render(self, name: str, **values: Any) -> str:
        """Render a named template with strict variable substitution."""
        template = getattr(self, name, None)
        if template is None:
            raise ValueError(f"Unknown template: {name!r}")
        if not template:
            raise ValueError(f"Empty required template: {name!r}")
        try:
            return template.format(**values)
        except KeyError as exc:
            raise ValueError(
                f"Missing required template variable {exc} in template {name!r}"
            ) from exc

    @staticmethod
    def load_from_dir(prompt_dir: Any) -> "AxisPromptTemplates":
        """Load templates from a directory of .txt files."""
        from pathlib import Path

        d = Path(prompt_dir)
        templates: Dict[str, str] = {}
        for name in (
            "system",
            "system_hybrid",
            "factual_context",
            "interrogation",
            "interrogation_hybrid",
            "intermediate_explanation",
            "final_explanation",
        ):
            path = d / f"{name}.txt"
            if path.exists():
                templates[name] = path.read_text(encoding="utf-8")
        return AxisPromptTemplates(**templates)

    def validate_required(self, *names: str) -> None:
        """Raise if any of the named templates are empty."""
        for name in names:
            val = getattr(self, name, None)
            if not val:
                raise ValueError(f"Required prompt template missing: {name!r}")
