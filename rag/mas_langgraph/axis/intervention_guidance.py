"""Deterministic intervention-candidate guidance for AXIS interrogation.

Builds valid, contract-backed candidates from CaseSlice + contracts.
Does not invent causality — candidates are testable hypotheses only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import AbstractSet, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..artifacts.models import ArtifactRef
from ..replay.branch_artifacts import artifact_exists_at_checkpoint
from ..replay.checkpoints import BoundarySelector
from .hybrid_add_diagnostics import (
    ADD_PRESENCE_ABSENT,
    ADD_PRESENCE_PRESENT,
    ADD_PRESENCE_UNKNOWN,
    CANDIDATE_SOURCE_EXPLICIT,
    CANDIDATE_SOURCE_FACTUAL,
    CANDIDATE_SOURCE_INTERVENTION,
    HybridAddDiagnostics,
)
from .interrogation_mode import (
    DEFAULT_HYBRID_MAX_ALTERNATIVES_PER_TARGET,
    WHAT_DISABLED_IN_INTERROGATION,
)


@dataclass(frozen=True)
class InterventionCandidate:
    """One contract-valid intervention target with operation-specific boundaries."""

    ref: ArtifactRef
    artifact_type: str
    allowed_operations: Tuple[str, ...]
    used_by: Tuple[str, ...] = ()
    valid_boundaries: Tuple[str, ...] = ()
    # operation -> resolvable boundary strings that are valid for that op
    operation_boundaries: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    priority: int = 100

    def boundaries_for(self, operation: str) -> Tuple[str, ...]:
        ob = self.operation_boundaries or {}
        if operation in ob:
            return tuple(ob[operation])
        if operation in self.allowed_operations:
            return self.valid_boundaries
        return ()


@dataclass(frozen=True)
class InterrogationAttempt:
    """Researcher-only record of one interrogation generation attempt."""

    attempt_index: int
    generated_query: str
    parsed_operation: Optional[str] = None
    validation_status: str = "unknown"
    parser_error: Optional[str] = None
    resolution_error: Optional[str] = None
    duplicate_error: Optional[str] = None
    contract_error: Optional[str] = None
    execution_error: Optional[str] = None
    executed: bool = False
    produced_counterfactual: bool = False
    counted_as_successful_round: bool = False
    vocabulary_error: Optional[str] = None
    available_what_count: Optional[int] = None
    available_remove_count: Optional[int] = None
    available_add_count: Optional[int] = None
    available_whatif_count: Optional[int] = None
    axis_interrogation_mode: Optional[str] = None
    axis_interrogation_policy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "generated_query": self.generated_query,
            "parsed_operation": self.parsed_operation,
            "validation_status": self.validation_status,
            "parser_error": self.parser_error,
            "resolution_error": self.resolution_error,
            "duplicate_error": self.duplicate_error,
            "contract_error": self.contract_error,
            "execution_error": self.execution_error,
            "executed": self.executed,
            "produced_counterfactual": self.produced_counterfactual,
            "counted_as_successful_round": self.counted_as_successful_round,
            "vocabulary_error": self.vocabulary_error,
            "available_what_count": self.available_what_count,
            "available_remove_count": self.available_remove_count,
            "available_add_count": self.available_add_count,
            "available_whatif_count": self.available_whatif_count,
            "axis_interrogation_mode": self.axis_interrogation_mode,
            "axis_interrogation_policy": self.axis_interrogation_policy,
        }


def classify_validation_error(message: str) -> str:
    """Map resolver/parser messages to controlled error categories."""
    text = str(message or "").lower()
    if "not currently executable" in text or "not_currently_executable" in text:
        return "not_currently_executable"
    if "vocabulary" in text or "advertised" in text:
        return "vocabulary_error"
    if "duplicate" in text:
        return "duplicate_error"
    if "contract" in text or "no remove" in text or "no add" in text or "no replace" in text:
        return "contract_error"
    if "boundary" in text or "cannot resolve" in text:
        return "resolution_error"
    if "not in" in text or "slice" in text or "scope" in text:
        return "resolution_error"
    return "resolution_error"


def controlled_validation_feedback(
    *,
    status: str,
    detail: str,
    executable_queries: str = "",
) -> str:
    """Human-readable feedback for the next interrogation prompt (no tracebacks)."""
    detail = str(detail or "").strip()
    if status == "parser_error":
        msg = (
            "The previous query could not be parsed.\n"
            f"{detail}\n"
            "Choose a different valid structured query."
        )
        if executable_queries:
            msg += (
                "\n\nCopy one executable form exactly:\n"
                + "\n".join(
                    line
                    for line in executable_queries.splitlines()
                    if line.strip()
                    and not line.startswith("Executable")
                    and not line.startswith("Return")
                    and not line.startswith("Preserve")
                )[:800]
            )
        return msg
    if status == "duplicate_error":
        msg = (
            "The previous query duplicated an already executed query.\n"
            "Do not repeat it."
        )
        if executable_queries:
            remaining = [
                line.strip()
                for line in executable_queries.splitlines()
                if line.strip()
                and not line.startswith("Executable")
                and not line.startswith("Return")
                and not line.startswith("Preserve")
                and line.strip() != "DONE"
            ][:8]
            if remaining:
                msg += (
                    "\n\nChoose one of these untested executable queries:\n"
                    + "\n".join(remaining)
                )
            else:
                msg += "\nChoose a different query, or DONE if no options remain."
        else:
            msg += "\nChoose a different query."
        return msg
    if status == "not_currently_executable":
        msg = (
            "The previous query is not in the current executable set "
            "(it may already have been tested or is no longer available).\n"
            "Return exactly one of the executable query forms shown below, or DONE."
        )
        if executable_queries:
            remaining = [
                line.strip()
                for line in executable_queries.splitlines()
                if line.strip()
                and not line.startswith("Executable")
                and not line.startswith("Return")
                and not line.startswith("Preserve")
            ][:10]
            if remaining:
                msg += "\n\n" + "\n".join(remaining)
        return msg
    if status == "contract_error":
        return (
            "The previous query could not be executed:\n"
            f"{detail}\n"
            "Choose another valid query from the intervention candidates."
        )
    if status == "resolution_error":
        return (
            "The previous query could not be resolved.\n"
            f"{detail}\n"
            "Use one of the available valid checkpoint boundaries shown below."
        )
    if status == "execution_error":
        return (
            "The previous query failed during execution.\n"
            f"{detail}\n"
            "Choose another valid query."
        )
    return f"The previous query could not be used.\n{detail}\nChoose a different query."


def controlled_hybrid_validation_feedback(
    *,
    status: str,
    detail: str,
) -> str:
    """Hybrid-mode retry feedback. Never asks the LLM to copy an exact query list."""
    detail = str(detail or "").strip()
    if status == "parser_error":
        return (
            "The previous query could not be parsed.\n"
            f"{detail}\n"
            "Compose one structured query using the grammar and the exact "
            "artifact refs, operations, replacements, and boundaries shown below."
        )
    if status == "duplicate_error":
        return (
            "The previous query duplicated an already executed query.\n"
            "Do not repeat it. Compose a different query from the remaining "
            "vocabulary, or return DONE."
        )
    if status == "not_currently_executable":
        return (
            "The previous query was already executed or is no longer available.\n"
            "Compose a different query from the remaining vocabulary, or return DONE."
        )
    if status == "vocabulary_error":
        text = detail.lower()
        if "replacement" in text:
            return (
                "The proposed replacement is not in the allowed replacement vocabulary.\n"
                "Choose an exact replacement ref from the alternatives listed for this target."
            )
        if "boundary" in text:
            return (
                "The boundary is invalid for this target.\n"
                "Choose one of the listed boundaries."
            )
        if "add" in text:
            return (
                "The proposed ADD artifact is not in the advertised ADD vocabulary.\n"
                "Choose an exact artifact ref from the addable artifacts listed below, "
                "or compose a different operation."
            )
        if "what" in text and "target" in text:
            return (
                "The proposed WHAT target is not in the advertised factual vocabulary.\n"
                "Choose an exact artifact ref from the factual query targets listed below."
            )
        if "operation" in text or "remove" in text or "whatif" in text:
            return (
                "That operation is not advertised for this target.\n"
                "Choose an allowed operation from the vocabulary listed for the target."
            )
        return (
            "The proposed query is not in the advertised hybrid vocabulary.\n"
            f"{detail}\n"
            "Compose one query using only the exact refs, operations, and "
            "boundaries shown below."
        )
    if status == "contract_error":
        return (
            "The previous query could not be executed:\n"
            f"{detail}\n"
            "Compose another valid query from the advertised vocabulary."
        )
    if status == "resolution_error":
        return (
            "The previous query could not be resolved.\n"
            f"{detail}\n"
            "Use an exact artifact ref and one of the listed checkpoint boundaries."
        )
    if status == "execution_error":
        return (
            "The previous query failed during execution.\n"
            f"{detail}\n"
            "Compose another valid query from the advertised vocabulary."
        )
    return (
        "The previous query could not be used.\n"
        f"{detail}\n"
        "Compose a different query from the advertised vocabulary."
    )


def _artifact_type(registry: Any, ref: ArtifactRef, branch_id: str) -> str:
    if registry is None:
        return ""
    env = None
    getter = getattr(registry, "get_by_ref", None)
    if callable(getter):
        env = getter(ref, branch_id) or getter(ref)
    return str(getattr(env, "artifact_type", "") or "") if env is not None else ""


def _occ_by_id(occurrences: Sequence[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for occ in occurrences or ():
        oid = getattr(occ, "occurrence_id", None)
        if oid:
            out[str(oid)] = occ
    return out


def _resolve_boundary_text(checkpoint_index: Any, text: str) -> Optional[str]:
    """Return the boundary text if it resolves; else None."""
    token = str(text or "").strip()
    if not token or checkpoint_index is None:
        return None
    try:
        if token.startswith("before "):
            rest = token[len("before ") :].strip()
            if "#" in rest:
                node, idx_s = rest.rsplit("#", 1)
                sel = BoundarySelector.before_occurrence(node.strip(), int(idx_s))
            else:
                return None
        elif token.startswith("after "):
            rest = token[len("after ") :].strip()
            if "#" in rest:
                node, idx_s = rest.rsplit("#", 1)
                sel = BoundarySelector.after_occurrence(node.strip(), int(idx_s))
            else:
                return None
        else:
            # Alias form c_s
            sel = BoundarySelector.by_alias(token)
        checkpoint_index.resolve(sel)
        return token
    except Exception:
        return None


def _resolve_boundary_record(checkpoint_index: Any, text: str) -> Any:
    """Resolve a boundary token to a CheckpointRecord, or None."""
    token = str(text or "").strip()
    if not token or checkpoint_index is None:
        return None
    try:
        if token.startswith("before "):
            rest = token[len("before ") :].strip()
            if "#" not in rest:
                return None
            node, idx_s = rest.rsplit("#", 1)
            sel = BoundarySelector.before_occurrence(node.strip(), int(idx_s))
        elif token.startswith("after "):
            rest = token[len("after ") :].strip()
            if "#" not in rest:
                return None
            node, idx_s = rest.rsplit("#", 1)
            sel = BoundarySelector.after_occurrence(node.strip(), int(idx_s))
        else:
            sel = BoundarySelector.by_alias(token)
        return checkpoint_index.resolve(sel)
    except Exception:
        return None


def _snapshot_values_for_boundary(
    *,
    boundary: str,
    checkpoint_index: Any,
    replay_executor: Any,
    run_id: str,
) -> Optional[Mapping[str, Any]]:
    """Load factual checkpoint values for a resolvable boundary. None if unknown."""
    if replay_executor is None or not run_id:
        return None
    record = _resolve_boundary_record(checkpoint_index, boundary)
    if record is None:
        return None
    getter = getattr(replay_executor, "get_checkpoint_snapshot", None)
    if not callable(getter):
        return None
    alias = str(getattr(record, "alias", "") or "")
    if not alias:
        return None
    try:
        snap = getter(run_id, BoundarySelector.by_alias(alias))
    except Exception:
        return None
    values = getattr(snap, "values", None)
    if not isinstance(values, Mapping):
        return None
    return values


@dataclass(frozen=True)
class AddPresenceClassification:
    """Checkpoint presence for one ADD candidate–boundary pair."""

    presence: str
    detail: str
    checkpoint_resolved: bool
    checkpoint_alias: str = ""
    checkpoint_id: str = ""
    state_inspectable: bool = False


def classify_artifact_presence_at_add_boundary(
    ref: ArtifactRef,
    boundary: str,
    *,
    checkpoint_index: Any,
    replay_executor: Any,
    run_id: str,
) -> AddPresenceClassification:
    """Classify presence without coercing unknown state to present or absent.

    ``present`` / ``absent`` require an inspectable ``artifact_catalog``.
    Missing snapshot, missing catalog, or an unresolvable checkpoint is
    ``unknown`` and must not be advertised as ADD.
    """
    record = _resolve_boundary_record(checkpoint_index, boundary)
    alias = str(getattr(record, "alias", "") or "") if record is not None else ""
    checkpoint_id = (
        str(getattr(record, "checkpoint_id", "") or "") if record is not None else ""
    )
    if record is None:
        return AddPresenceClassification(
            presence=ADD_PRESENCE_UNKNOWN,
            detail="checkpoint_unresolvable",
            checkpoint_resolved=False,
            checkpoint_alias=alias,
            checkpoint_id=checkpoint_id,
            state_inspectable=False,
        )
    values = _snapshot_values_for_boundary(
        boundary=boundary,
        checkpoint_index=checkpoint_index,
        replay_executor=replay_executor,
        run_id=run_id,
    )
    if values is None:
        return AddPresenceClassification(
            presence=ADD_PRESENCE_UNKNOWN,
            detail="state_unavailable",
            checkpoint_resolved=True,
            checkpoint_alias=alias,
            checkpoint_id=checkpoint_id,
            state_inspectable=False,
        )
    if "artifact_catalog" not in values:
        return AddPresenceClassification(
            presence=ADD_PRESENCE_UNKNOWN,
            detail="artifact_catalog_missing",
            checkpoint_resolved=True,
            checkpoint_alias=alias,
            checkpoint_id=checkpoint_id,
            state_inspectable=False,
        )
    if artifact_exists_at_checkpoint(values, ref):
        return AddPresenceClassification(
            presence=ADD_PRESENCE_PRESENT,
            detail="confirmed_present",
            checkpoint_resolved=True,
            checkpoint_alias=alias,
            checkpoint_id=checkpoint_id,
            state_inspectable=True,
        )
    return AddPresenceClassification(
        presence=ADD_PRESENCE_ABSENT,
        detail="confirmed_absent",
        checkpoint_resolved=True,
        checkpoint_alias=alias,
        checkpoint_id=checkpoint_id,
        state_inspectable=True,
    )


def artifact_absent_at_add_boundary(
    ref: ArtifactRef,
    boundary: str,
    *,
    checkpoint_index: Any,
    replay_executor: Any,
    run_id: str,
) -> bool:
    """True only when factual checkpoint state is known and ``ref`` is absent.

    Missing snapshot / missing ``artifact_catalog`` is treated as unverifiable
    and therefore not advertised as ADD.
    """
    classified = classify_artifact_presence_at_add_boundary(
        ref,
        boundary,
        checkpoint_index=checkpoint_index,
        replay_executor=replay_executor,
        run_id=run_id,
    )
    return classified.presence == ADD_PRESENCE_ABSENT


def _boundaries_for_artifact(
    *,
    ref: ArtifactRef,
    case_slice: Any,
    all_occurrences: Sequence[Any],
    checkpoint_index: Any,
    max_boundaries: int = 4,
) -> Tuple[str, Tuple[str, ...]]:
    """Return (used_by labels, resolvable boundary strings)."""
    occ_map = _occ_by_id(all_occurrences)
    # Prefer full occurrence list; fall back to slice.
    if not occ_map:
        occ_map = _occ_by_id(getattr(case_slice, "relevant_occurrences", ()) or ())

    used_by: List[str] = []
    boundaries: List[str] = []
    seen_b: set[str] = set()

    for rel in getattr(case_slice, "relevant_relations", ()) or ():
        aref = getattr(rel, "artifact_ref", None)
        if aref != ref:
            continue
        cons_id = getattr(rel, "consumer_occurrence_id", None)
        occ = occ_map.get(str(cons_id)) if cons_id else None
        if occ is None:
            continue
        labeled = getattr(occ, "labeled_id", None) or ""
        node = getattr(occ, "node_id", "") or ""
        if labeled:
            used_by.append(str(labeled))
        elif node:
            used_by.append(str(node))
        if labeled:
            for prefix in ("before", "after"):
                text = f"{prefix} {labeled}"
                if text in seen_b:
                    continue
                resolved = _resolve_boundary_text(checkpoint_index, text)
                if resolved:
                    seen_b.add(text)
                    boundaries.append(resolved)
                if len(boundaries) >= max_boundaries:
                    break
        if len(boundaries) >= max_boundaries:
            break

    # Fallback: any resolvable before/after from slice occurrences.
    if not boundaries:
        for occ in getattr(case_slice, "relevant_occurrences", ()) or ():
            labeled = getattr(occ, "labeled_id", None)
            if not labeled:
                continue
            text = f"before {labeled}"
            if text in seen_b:
                continue
            resolved = _resolve_boundary_text(checkpoint_index, text)
            if resolved:
                seen_b.add(text)
                boundaries.append(resolved)
            if len(boundaries) >= max_boundaries:
                break

    # Last resort: checkpoint aliases present in the index.
    if not boundaries and checkpoint_index is not None:
        records = list(getattr(checkpoint_index, "records", None) or [])
        for rec in records:
            alias = getattr(rec, "alias", None)
            if not alias:
                continue
            resolved = _resolve_boundary_text(checkpoint_index, str(alias))
            if resolved and resolved not in seen_b:
                seen_b.add(resolved)
                boundaries.append(resolved)
            if len(boundaries) >= max_boundaries:
                break

    # Dedupe used_by preserving order
    seen_u: set[str] = set()
    used_clean: List[str] = []
    for u in used_by:
        if u not in seen_u:
            seen_u.add(u)
            used_clean.append(u)
    return tuple(used_clean), tuple(boundaries)


def _boundary_preference_rank(
    boundary: str,
    artifact_type: str = "",
) -> Tuple[int, str]:
    """Lower rank = earlier / more type-appropriate insertion point."""
    a_type = str(artifact_type or "")
    low = str(boundary).lower()
    if a_type in {"EvidenceChunk", "EvidenceChunkPayload"}:
        if "before" in low and "atomize" in low:
            return (0, low)
        if "before" in low and "plan_mcq" in low:
            return (2, low)
        if "before" in low and "generate_candidate" in low:
            return (3, low)
    if a_type in {"KnowledgeAtom", "KnowledgeAtomPayload"}:
        if "before" in low and "plan_mcq" in low:
            return (0, low)
        if "before" in low and "generate_candidate" in low:
            return (1, low)
    if a_type in {"MCQBlueprint", "MCQBlueprintPayload"}:
        if "before" in low and "generate_candidate" in low:
            return (0, low)
    if a_type in {"CandidateMCQ", "CandidateMCQPayload"}:
        if "before" in low and any(
            x in low for x in ("verify", "acceptance", "cross_cove")
        ):
            return (8, low)
        if "after" in low and "generate" in low:
            return (9, low)

    if low.startswith("before ") and any(
        x in low
        for x in (
            "atomize",
            "plan_mcq",
            "curate",
            "retrieve",
            "outline",
            "learning_objective",
            "deduplicate",
            "generate_candidate",
        )
    ):
        return (4, low)
    if low.startswith("after ") and any(
        x in low
        for x in ("generate_candidate", "acceptance", "verify", "cross_cove")
    ):
        return (10, low)
    return (5, low)


def _prefer_early_boundaries(
    boundaries: Sequence[str],
    *,
    artifact_type: str = "",
) -> Tuple[str, ...]:
    """Prefer boundaries that typically precede subject generation.

    Artifact-aware when ``artifact_type`` is provided (MCQ question-content).
    """
    a_type = str(artifact_type or "")
    return tuple(
        sorted(
            (str(b) for b in boundaries),
            key=lambda b: _boundary_preference_rank(b, a_type),
        )
    )


def build_intervention_candidates(
    *,
    case_slice: Any,
    registry: Any,
    contracts: Any,
    checkpoint_index: Any,
    branch_id: str = "factual",
    all_occurrences: Optional[Sequence[Any]] = None,
    priority_fn: Optional[Callable[[str], int]] = None,
    intervention_alternatives: Optional[Mapping[str, ArtifactRef]] = None,
) -> Tuple[InterventionCandidate, ...]:
    """Contract-backed intervenable artifacts with operation-specific boundaries.

    Slice artifacts may receive REMOVE/REPLACE (they already exist).
    ADD is only offered for controlled alternatives that are not already in-slice.
    """
    refs = list(getattr(case_slice, "relevant_artifact_refs", ()) or ())
    slice_set = set(refs)
    occs = list(all_occurrences or getattr(case_slice, "relevant_occurrences", ()) or ())
    cands: List[InterventionCandidate] = []

    for ref in refs:
        a_type = _artifact_type(registry, ref, branch_id)
        if not a_type:
            continue
        contract = contracts.get(a_type) if contracts is not None else None
        if contract is None:
            continue
        used_by, boundaries = _boundaries_for_artifact(
            ref=ref,
            case_slice=case_slice,
            all_occurrences=occs,
            checkpoint_index=checkpoint_index,
        )
        if not boundaries:
            continue
        op_bounds: Dict[str, Tuple[str, ...]] = {}
        # Existing slice artifacts: REMOVE/REPLACE only (ADD of existing is a no-op).
        if contract.allows("REMOVE"):
            op_bounds["REMOVE"] = _prefer_early_boundaries(
                boundaries, artifact_type=a_type
            )
        if contract.allows("REPLACE"):
            op_bounds["REPLACE"] = _prefer_early_boundaries(
                boundaries, artifact_type=a_type
            )
        ops = tuple(op_bounds.keys())
        if not ops:
            continue
        pri = int(priority_fn(a_type)) if priority_fn is not None else 100
        cands.append(
            InterventionCandidate(
                ref=ref,
                artifact_type=a_type,
                allowed_operations=ops,
                used_by=used_by,
                valid_boundaries=boundaries,
                operation_boundaries=op_bounds,
                priority=pri,
            )
        )

    # Controlled ADD alternatives that are not already present in the slice.
    for _alias, alt_ref in (intervention_alternatives or {}).items():
        if alt_ref in slice_set:
            continue
        a_type = _artifact_type(registry, alt_ref, branch_id)
        if not a_type:
            continue
        contract = contracts.get(a_type) if contracts is not None else None
        if contract is None or not contract.allows("ADD"):
            continue
        # Use slice occurrence boundaries as generic ADD insertion points.
        _used, boundaries = _boundaries_for_artifact(
            ref=alt_ref,
            case_slice=case_slice,
            all_occurrences=occs,
            checkpoint_index=checkpoint_index,
        )
        if not boundaries:
            # Fall back to any alias/before from slice occurrences only.
            _, boundaries = _boundaries_for_artifact(
                ref=refs[0] if refs else alt_ref,
                case_slice=case_slice,
                all_occurrences=occs,
                checkpoint_index=checkpoint_index,
            )
        if not boundaries:
            continue
        pri = int(priority_fn(a_type)) if priority_fn is not None else 100
        cands.append(
            InterventionCandidate(
                ref=alt_ref,
                artifact_type=a_type,
                allowed_operations=("ADD",),
                used_by=(),
                valid_boundaries=boundaries,
                operation_boundaries={"ADD": boundaries},
                priority=pri,
            )
        )

    cands.sort(key=lambda c: (c.priority, str(c.ref), c.artifact_type))
    return tuple(cands)


def format_intervention_candidates(candidates: Sequence[InterventionCandidate]) -> str:
    if not candidates:
        return "(no valid intervention candidates with resolvable boundaries)"
    parts: List[str] = []
    for i, c in enumerate(candidates, start=1):
        parts.append(f"{i}. {c.artifact_type} {c.ref}")
        if c.used_by:
            parts.append("   Used by:")
            for u in c.used_by[:6]:
                parts.append(f"   - {u}")
        for op in c.allowed_operations:
            bounds = c.boundaries_for(op)
            if not bounds:
                continue
            parts.append(f"   {op} boundaries:")
            for b in bounds:
                parts.append(f"   - {b}")
    return "\n".join(parts)


def format_executable_queries(
    *,
    candidates: Sequence[InterventionCandidate],
    explained_case: Any = None,
    max_queries: int = 12,
    exclude_signatures: Optional[AbstractSet[Any]] = None,
    resolve_query: Optional[Callable[[str], Any]] = None,
    interrogation_goal: str = "general",
    intervention_alternatives: Optional[Mapping[str, ArtifactRef]] = None,
    registry: Any = None,
    branch_id: str = "factual",
    exclude_query_texts: Optional[AbstractSet[str]] = None,
) -> str:
    """Canonical executable query strings for the interrogation LLM.

    When ``exclude_signatures`` and ``resolve_query`` are provided, already
    executed intervention signatures are omitted from the recommended list.

    For ``interrogation_goal`` in ``question_content`` / ``question_selection``,
    prioritize pre-generation EvidenceChunk / KnowledgeAtom interventions over
    CandidateMCQ REMOVE. WHAT is not advertised: factual information is supplied
    by the execution trace and provenance model.
    """
    goal = str(interrogation_goal or "general").strip().lower()
    content_like = goal in {"question_content", "question_selection"}
    lines: List[str] = [
        "Executable queries:",
        "",
        "Return exactly one of the executable query forms shown below.",
        "Preserve parentheses, commas, artifact versions and boundary spelling exactly.",
        "",
    ]

    # Sort candidates by priority (already sorted) with goal-aware demotion of CandidateMCQ.
    ordered = list(candidates)
    if content_like:
        def _cand_rank(c: InterventionCandidate) -> Tuple[int, int, str]:
            a = c.artifact_type
            type_rank = {
                "EvidenceChunk": 0,
                "EvidenceChunkPayload": 0,
                "KnowledgeAtom": 1,
                "KnowledgeAtomPayload": 1,
                "MCQBlueprint": 3,
                "MCQBlueprintPayload": 3,
                "CandidateMCQ": 9,
                "CandidateMCQPayload": 9,
            }.get(a, 5)
            return (type_rank, c.priority, str(c.ref))

        ordered = sorted(ordered, key=_cand_rank)

    # Controlled WHATIF pairs: same-type alternatives for REPLACE-capable candidates.
    alt_refs: List[ArtifactRef] = list((intervention_alternatives or {}).values())
    alt_by_type: Dict[str, List[ArtifactRef]] = {}
    for alt in alt_refs:
        a_type = _artifact_type(registry, alt, branch_id) if registry is not None else ""
        if a_type:
            alt_by_type.setdefault(a_type, []).append(alt)

    queries: List[str] = []

    def _append_interventions_for(c: InterventionCandidate) -> None:
        for op in c.allowed_operations:
            bounds = c.boundaries_for(op)[:2]
            if op == "REPLACE":
                alts = alt_by_type.get(c.artifact_type) or []
                for alt in alts[:2]:
                    if alt == c.ref or str(alt) == str(c.ref):
                        continue
                    for b in bounds:
                        queries.append(f"WHATIF({c.ref}, {alt}, {b})")
                continue
            for b in bounds:
                queries.append(f"{op}({c.ref}, {b})")

    if content_like:
        for c in ordered:
            if c.artifact_type in {
                "CandidateMCQ",
                "CandidateMCQPayload",
            }:
                continue
            _append_interventions_for(c)
        for c in ordered:
            if c.artifact_type in {"CandidateMCQ", "CandidateMCQPayload"}:
                _append_interventions_for(c)
    else:
        for c in ordered:
            _append_interventions_for(c)
            if len(queries) >= max_queries * 3:
                break

    # Deduplicate preserving order; drop already-executed when resolvable.
    seen: set[str] = set()
    excluded = exclude_signatures or set()
    excluded_texts = {
        re.sub(r"\s+", " ", str(t or "").strip().lower())
        for t in (exclude_query_texts or set())
        if str(t or "").strip()
    }
    kept: List[str] = []
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        q_norm = re.sub(r"\s+", " ", str(q).strip().lower())
        if q_norm in excluded_texts:
            continue
        if excluded and resolve_query is not None:
            try:
                sig = resolve_query(q)
            except Exception:
                sig = None
            if sig is not None and sig in excluded:
                continue
        kept.append(q)
        if len(kept) >= max_queries:
            break
    for q in kept:
        lines.append(q)
    lines.append("")
    lines.append("DONE")
    return "\n".join(lines)


def executable_query_forms(executable_queries_text: str) -> set[str]:
    """Normalize executable query forms (excluding DONE / prose headers)."""
    out: set[str] = set()
    for line in str(executable_queries_text or "").splitlines():
        text = re.sub(r"\s+", " ", line.strip())
        if not text or text == "DONE":
            continue
        if text.startswith("Executable") or text.startswith("Return") or text.startswith("Preserve"):
            continue
        if text.startswith("("):
            continue
        out.add(text.lower())
    return out


def format_factual_only_artifacts(
    *,
    case_slice: Any,
    registry: Any,
    contracts: Any,
    branch_id: str = "factual",
) -> str:
    """Artifacts in the slice that have no applicable intervention contract."""
    lines: List[str] = []
    for ref in getattr(case_slice, "relevant_artifact_refs", ()) or ():
        a_type = _artifact_type(registry, ref, branch_id)
        contract = contracts.get(a_type) if (contracts is not None and a_type) else None
        if contract is not None and any(
            contract.allows(op) for op in ("REMOVE", "REPLACE")
        ):
            continue
        label = a_type or "Unknown"
        lines.append(f"- {label} {ref}")
    return "\n".join(lines) if lines else "(none)"


def format_intervenable_handoff(
    *,
    target_ref: Any,
    allowed_operations: Sequence[str],
    valid_boundaries: Sequence[str],
    has_successful_intervention: bool,
    executable_examples: Sequence[str] = (),
) -> str:
    if has_successful_intervention or not allowed_operations:
        return ""
    ops = ", ".join(str(o) for o in allowed_operations)
    parts = [
        "The artifact inspected in the previous round is intervenable.",
        "",
        f"Artifact: {target_ref}",
        "",
        f"Allowed operations: {ops}",
        "",
        "No successful counterfactual intervention has been executed yet.",
        "",
        "Prefer testing one valid intervention on this artifact before asking "
        "more factual questions, unless another intervention is clearly more informative.",
    ]
    if executable_examples:
        parts.append("")
        parts.append("Example executable queries:")
        for ex in executable_examples[:4]:
            parts.append(ex)
    elif valid_boundaries:
        parts.append("")
        parts.append("Valid boundaries for this artifact:")
        for b in valid_boundaries:
            parts.append(f"- {b}")
    return "\n".join(parts)


def format_done_nudge(candidates: Sequence[InterventionCandidate]) -> str:
    lines = [
        "No successful counterfactual intervention has been executed yet.",
        "At least one valid intervention candidate exists:",
        "",
    ]
    for c in list(candidates)[:5]:
        for op in c.allowed_operations:
            bounds = c.boundaries_for(op)
            if not bounds:
                continue
            lines.append(f"{op}({c.ref}, {bounds[0]})")
            break
    lines.append("")
    lines.append(
        "If a valid intervention could clarify the explanation, generate one "
        "before terminating. Otherwise return DONE."
    )
    return "\n".join(lines)


def format_last_chance_nudge(candidates: Sequence[InterventionCandidate]) -> str:
    lines = [
        "You have not yet executed a counterfactual intervention.",
        "",
        "Choose one valid intervention query now, or answer DONE only if no "
        "available intervention would help answer the user's question.",
        "",
        "Executable examples:",
    ]
    for c in list(candidates)[:5]:
        for op in c.allowed_operations:
            bounds = c.boundaries_for(op)
            if not bounds:
                continue
            lines.append(f"{op}({c.ref}, {bounds[0]})")
            break
    return "\n".join(lines)


def find_candidate_for_ref(
    candidates: Sequence[InterventionCandidate],
    ref: Any,
) -> Optional[InterventionCandidate]:
    for c in candidates:
        if c.ref == ref or str(c.ref) == str(ref):
            return c
    return None


# ---------------------------------------------------------------------------
# Hybrid interrogation vocabulary (query composition, not exact-form copy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridInterventionVocabulary:
    """Controlled vocabulary advertised to the LLM in hybrid interrogation mode.

    Researcher-auditable: every ref/operation/boundary shown here is derived
    from the factual registry, contracts, and checkpoint index. No synthetic
    artifact payloads are created.
    """

    # Retained for historical vocabulary objects. New builders leave this empty:
    # WHAT is disabled in generalized AXIS interrogation.
    what_refs: Tuple[ArtifactRef, ...] = ()
    intervention_candidates: Tuple[InterventionCandidate, ...] = ()
    replacement_options: Mapping[ArtifactRef, Tuple[ArtifactRef, ...]] = field(
        default_factory=dict
    )
    add_options: Tuple[ArtifactRef, ...] = ()
    add_boundaries_by_ref: Mapping[ArtifactRef, Tuple[str, ...]] = field(
        default_factory=dict
    )
    add_boundaries: Tuple[str, ...] = ()
    max_alternatives_per_target: int = DEFAULT_HYBRID_MAX_ALTERNATIVES_PER_TARGET

    def replacements_for(self, ref: Any) -> Tuple[ArtifactRef, ...]:
        for old, news in (self.replacement_options or {}).items():
            if old == ref or str(old) == str(ref):
                return tuple(news)
        return ()

    def add_boundaries_for(self, ref: Any) -> Tuple[str, ...]:
        for add_ref, bounds in (self.add_boundaries_by_ref or {}).items():
            if add_ref == ref or str(add_ref) == str(ref):
                return tuple(bounds)
        return ()

    def candidate_for(self, ref: Any) -> Optional[InterventionCandidate]:
        return find_candidate_for_ref(self.intervention_candidates, ref)


def _payload_get(obj: Any, *names: str) -> Any:
    if obj is None:
        return None
    for name in names:
        val = None
        if isinstance(obj, Mapping):
            val = obj.get(name)
        else:
            val = getattr(obj, name, None)
        if val not in (None, ""):
            return val
    return None


def _envelope_field(env: Any, *names: str) -> Any:
    payload = getattr(env, "payload", None)
    val = _payload_get(payload, *names)
    if val not in (None, ""):
        return val
    meta = getattr(env, "metadata", None)
    val = _payload_get(meta, *names)
    if val not in (None, ""):
        return val
    if isinstance(payload, Mapping):
        nested = payload.get("metadata")
        val = _payload_get(nested, *names)
        if val not in (None, ""):
            return val
    nested_meta = getattr(payload, "metadata", None)
    return _payload_get(nested_meta, *names)


def _as_token_set(value: Any) -> set[str]:
    if value is None or value == "":
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(x).strip().lower() for x in value if str(x).strip()}
    text = str(value).strip().lower()
    if not text:
        return set()
    if "," in text:
        return {p.strip() for p in text.split(",") if p.strip()}
    return {text}


def latest_live_envelopes(
    registry: Any,
    *,
    branch_id: str = "factual",
) -> Tuple[Any, ...]:
    """Latest non-superseded envelope per logical id on one branch."""
    if registry is None:
        return ()
    by_logical: Dict[str, Any] = {}
    catalog = getattr(registry, "catalog", None) or {}
    for env in catalog.values():
        if str(getattr(env, "branch_id", "") or "factual") != str(branch_id):
            continue
        if bool(getattr(env, "superseded", False)):
            continue
        ref = getattr(env, "ref", None)
        if ref is None:
            continue
        lid = str(getattr(ref, "logical_id", "") or "")
        if not lid:
            continue
        prev = by_logical.get(lid)
        if prev is None or int(getattr(ref, "version", 0) or 0) > int(
            getattr(prev.ref, "version", 0) or 0
        ):
            by_logical[lid] = env
    return tuple(sorted(by_logical.values(), key=lambda e: str(getattr(e, "ref", ""))))


def _artifact_description(env: Any, *, limit: int = 80) -> str:
    text = _envelope_field(
        env,
        "content",
        "text",
        "atom_text",
        "snippet",
        "stem",
        "objective_text",
        "description",
    )
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return ""
    if len(raw) > limit:
        return raw[: limit - 1] + "…"
    return raw


def _artifact_lo_id(env: Any) -> str:
    return str(_envelope_field(env, "lo_id") or "").strip()


def _artifact_section_id(env: Any) -> str:
    return str(_envelope_field(env, "section_id") or "").strip()


# Existing EvidenceChunk / KnowledgeAtom payload fields used as a semantic
# compatibility FILTER (not a ranking-only heuristic). Looked up on the
# payload, envelope.metadata, and nested payload.metadata. No new fields.
HYBRID_SEMANTIC_CONTEXT_FIELDS: Tuple[str, ...] = (
    "relevance_tags",
    "tags",
    "keywords",
    "context_ids",
    "source_chunk_ids",
    "used_in_atoms",
)


def _artifact_context_tokens(env: Any) -> set[str]:
    tokens: set[str] = set()
    for name in HYBRID_SEMANTIC_CONTEXT_FIELDS:
        tokens |= _as_token_set(_envelope_field(env, name))
    return {t for t in tokens if t}


def _is_knowledge_atom_type(artifact_type: str) -> bool:
    return str(artifact_type or "") in {"KnowledgeAtom", "KnowledgeAtomPayload"}


def _is_evidence_chunk_type(artifact_type: str) -> bool:
    return str(artifact_type or "") in {"EvidenceChunk", "EvidenceChunkPayload"}


def hybrid_replacement_eligible(old_env: Any, new_env: Any) -> bool:
    """Conservative semantic eligibility for automatic WHATIF replacements.

    KnowledgeAtom:
      both have lo_id and they are equal
      OR, when lo_id is unavailable on either side, both have equal section_id.
      Otherwise reject. No unrelated same-type fallback.

    EvidenceChunk:
      equal section_id (payload / nested metadata)
      OR overlap in existing relevance/tag/context fields listed in
      HYBRID_SEMANTIC_CONTEXT_FIELDS.
      Otherwise reject. No "any other EvidenceChunk" fallback.

    Other types are not auto-discovered as replacements.
    """
    if old_env is None or new_env is None:
        return False
    old_type = str(getattr(old_env, "artifact_type", "") or "")
    new_type = str(getattr(new_env, "artifact_type", "") or "")
    if not old_type or old_type != new_type:
        return False
    old_ref = getattr(old_env, "ref", None)
    new_ref = getattr(new_env, "ref", None)
    if old_ref is not None and new_ref is not None and (
        old_ref == new_ref or str(old_ref) == str(new_ref)
    ):
        return False
    if _is_knowledge_atom_type(old_type):
        old_lo = _artifact_lo_id(old_env)
        new_lo = _artifact_lo_id(new_env)
        if old_lo and new_lo:
            return old_lo == new_lo
        old_section = _artifact_section_id(old_env)
        new_section = _artifact_section_id(new_env)
        return bool(old_section and new_section and old_section == new_section)
    if _is_evidence_chunk_type(old_type):
        old_section = _artifact_section_id(old_env)
        new_section = _artifact_section_id(new_env)
        if old_section and new_section and old_section == new_section:
            return True
        old_tokens = _artifact_context_tokens(old_env)
        new_tokens = _artifact_context_tokens(new_env)
        return bool(old_tokens and new_tokens and (old_tokens & new_tokens))
    return False


def _hybrid_replacement_rank(old_env: Any, new_env: Any) -> int:
    """Deterministic rank among already-eligible replacements. Lower is better."""
    a_type = str(getattr(old_env, "artifact_type", "") or "")
    if _is_knowledge_atom_type(a_type):
        old_lo = _artifact_lo_id(old_env)
        new_lo = _artifact_lo_id(new_env)
        if old_lo and new_lo and old_lo == new_lo:
            return 0
        old_section = _artifact_section_id(old_env)
        new_section = _artifact_section_id(new_env)
        if old_section and new_section and old_section == new_section:
            return 1
        return 2
    if _is_evidence_chunk_type(a_type):
        old_section = _artifact_section_id(old_env)
        new_section = _artifact_section_id(new_env)
        if old_section and new_section and old_section == new_section:
            return 0
        if _artifact_context_tokens(old_env) & _artifact_context_tokens(new_env):
            return 1
        return 2
    return 9


def rank_hybrid_replacements(
    *,
    old_env: Any,
    candidates: Sequence[Any],
    max_n: int = DEFAULT_HYBRID_MAX_ALTERNATIVES_PER_TARGET,
) -> Tuple[ArtifactRef, ...]:
    """Filter then rank automatic replacements. Type-compatibility is not enough."""
    scored: List[Tuple[int, str, ArtifactRef]] = []
    for env in candidates:
        ref = getattr(env, "ref", None)
        if ref is None:
            continue
        if not hybrid_replacement_eligible(old_env, env):
            continue
        scored.append((_hybrid_replacement_rank(old_env, env), str(ref), ref))
    scored.sort(key=lambda t: (t[0], t[1]))
    n = max(0, int(max_n))
    return tuple(ref for _, __, ref in scored[:n])


def _slice_semantic_context(
    case_slice: Any,
    registry: Any,
    branch_id: str,
) -> Tuple[set[str], set[str]]:
    """Section ids and context tokens already present on slice artifacts."""
    sections: set[str] = set()
    tokens: set[str] = set()
    if registry is None:
        return sections, tokens
    getter = getattr(registry, "get_by_ref", None)
    for ref in getattr(case_slice, "relevant_artifact_refs", ()) or ():
        env = None
        if callable(getter):
            env = getter(ref, branch_id) or getter(ref)
        if env is None:
            continue
        section = _artifact_section_id(env)
        if section:
            sections.add(section)
        tokens |= _artifact_context_tokens(env)
    return sections, tokens


def hybrid_add_eligible(
    env: Any,
    *,
    slice_sections: AbstractSet[str],
    slice_tokens: AbstractSet[str],
) -> bool:
    """Automatic ADD sources must be contextually related to the explained slice.

    For MCQ, automatic ADD is EvidenceChunk-only. Eligible when:
      the chunk's section_id is represented in the case slice
      OR it shares existing relevance/tag/context metadata with slice artifacts.
    No arbitrary same-type fallback.
    """
    if env is None:
        return False
    a_type = str(getattr(env, "artifact_type", "") or "")
    if not _is_evidence_chunk_type(a_type):
        return False
    section = _artifact_section_id(env)
    if section and section in slice_sections:
        return True
    tokens = _artifact_context_tokens(env)
    return bool(tokens and slice_tokens and (tokens & set(slice_tokens)))


def _hybrid_add_sourceable(
    ref: ArtifactRef,
    *,
    registry: Any,
    branch_id: str,
    intervention_registry: Any,
    contract: Any,
) -> bool:
    env = None
    if registry is not None:
        getter = getattr(registry, "get_by_ref", None)
        if callable(getter):
            env = getter(ref, branch_id) or getter(ref)
    if env is not None and not bool(getattr(env, "superseded", False)):
        return True
    if (
        intervention_registry is not None
        and contract is not None
        and bool(getattr(contract, "allow_external_add", False))
    ):
        getter = getattr(intervention_registry, "get_by_ref", None)
        env2 = None
        if callable(getter):
            env2 = getter(ref, branch_id) or getter(ref)
        if env2 is None:
            catalog = getattr(intervention_registry, "catalog", None) or {}
            for item in catalog.values():
                if getattr(item, "ref", None) == ref:
                    env2 = item
                    break
        if env2 is not None:
            return True
    return False


def _add_boundaries_for_type(
    *,
    artifact_type: str,
    case_slice: Any,
    checkpoint_index: Any,
    all_occurrences: Sequence[Any],
    registry: Any = None,
    branch_id: str = "factual",
    unresolved_out: Optional[List[str]] = None,
) -> Tuple[str, ...]:
    """Insertion boundaries immediately before actual consumers of type T.

    Uses producer/consumer provenance of same-type artifacts. Does not invent
    generic node-name fallbacks (outline/retrieve/curate/…). If no resolvable
    consumer boundary exists, return empty so automatic ADD is not advertised.
    """
    a_type = str(artifact_type or "")
    if not a_type:
        return ()
    occs = list(all_occurrences or getattr(case_slice, "relevant_occurrences", ()) or ())
    occ_map = _occ_by_id(occs)
    if not occ_map:
        occ_map = _occ_by_id(getattr(case_slice, "relevant_occurrences", ()) or ())
    collected: List[str] = []
    seen: set[str] = set()
    for rel in getattr(case_slice, "relevant_relations", ()) or ():
        aref = getattr(rel, "artifact_ref", None)
        if aref is None:
            continue
        ref_type = (
            _artifact_type(registry, aref, branch_id) if registry is not None else ""
        )
        if ref_type != a_type:
            continue
        cons_id = getattr(rel, "consumer_occurrence_id", None)
        occ = occ_map.get(str(cons_id)) if cons_id else None
        if occ is None:
            continue
        labeled = getattr(occ, "labeled_id", None)
        if not labeled:
            node = getattr(occ, "node_id", "") or ""
            idx = getattr(occ, "occurrence_index", None)
            if node and idx is not None:
                labeled = f"{node}#{idx}"
        if not labeled:
            continue
        text = f"before {labeled}"
        if text in seen:
            continue
        seen.add(text)
        resolved = _resolve_boundary_text(checkpoint_index, text)
        if not resolved:
            if unresolved_out is not None:
                unresolved_out.append(text)
            continue
        collected.append(resolved)
    if not collected:
        return ()
    return tuple(_prefer_early_boundaries(collected, artifact_type=a_type))


def _is_prepared_external_add(env: Any) -> bool:
    if str(getattr(env, "branch_id", "") or "") == "intervention_prep":
        return True
    meta = getattr(env, "metadata", None) or {}
    if isinstance(meta, Mapping):
        return bool(
            meta.get("hybrid_add_from_source") or meta.get("retrieved_for_intervention_prep")
        )
    return False


def _hybrid_add_rank(env: Any, slice_sections: AbstractSet[str]) -> Tuple[int, int, int, str]:
    """Cap ranking: section, prepared vs factual, then preparation ``selection_rank``.

    ``selection_rank`` is the zero-based order assigned during source
    preparation. It is not a re-sort by MMR score. Artifact-ref strings are
    only a final deterministic tie-breaker. Eligibility and vocabulary size
    are unchanged.
    """
    from .hybrid_add_source import UNRANKED_SELECTION_RANK, parse_selection_rank

    section = _artifact_section_id(env)
    section_rank = 0 if section and section in slice_sections else 1
    prepared = _is_prepared_external_add(env)
    prep_rank = 0 if prepared else 1
    sel_rank = UNRANKED_SELECTION_RANK
    if prepared:
        meta = getattr(env, "metadata", None) or {}
        parsed = parse_selection_rank(
            meta.get("selection_rank") if isinstance(meta, Mapping) else None
        )
        if parsed is None:
            payload = getattr(env, "payload", None)
            pmeta = getattr(payload, "metadata", None) or {}
            parsed = parse_selection_rank(
                pmeta.get("selection_rank") if isinstance(pmeta, Mapping) else None
            )
        if parsed is not None:
            sel_rank = parsed
    ref = getattr(env, "ref", None)
    return (section_rank, prep_rank, sel_rank, str(ref) if ref is not None else "")


def build_hybrid_intervention_vocabulary(
    *,
    case_slice: Any,
    registry: Any,
    contracts: Any,
    checkpoint_index: Any,
    branch_id: str = "factual",
    explained_case: Any = None,
    intervention_candidates: Optional[Sequence[InterventionCandidate]] = None,
    intervention_alternatives: Optional[Mapping[str, ArtifactRef]] = None,
    intervention_registry: Any = None,
    all_occurrences: Sequence[Any] = (),
    max_alternatives_per_target: int = DEFAULT_HYBRID_MAX_ALTERNATIVES_PER_TARGET,
    replay_executor: Any = None,
    run_id: str = "",
    diagnostics: Optional[HybridAddDiagnostics] = None,
) -> HybridInterventionVocabulary:
    """Build the exact hybrid vocabulary shown to the LLM.

    Replacements and ADD options are existing ArtifactRefs only. No payloads
    are synthesized.
    """
    candidates = tuple(intervention_candidates or ())
    if not candidates:
        candidates = build_intervention_candidates(
            case_slice=case_slice,
            registry=registry,
            contracts=contracts,
            checkpoint_index=checkpoint_index,
            branch_id=branch_id,
            all_occurrences=all_occurrences,
            intervention_alternatives=intervention_alternatives,
        )

    slice_refs = tuple(getattr(case_slice, "relevant_artifact_refs", ()) or ())
    slice_set = set(slice_refs)
    # WHAT is disabled: do not advertise factual inspection targets.
    what_refs: List[ArtifactRef] = []

    live = latest_live_envelopes(registry, branch_id=branch_id)
    live_by_type: Dict[str, List[Any]] = {}
    for env in live:
        live_by_type.setdefault(str(env.artifact_type), []).append(env)

    explicit_alts = list((intervention_alternatives or {}).values())
    explicit_alt_keys = {str(r) for r in explicit_alts}
    slice_keys = {str(r) for r in slice_set}
    cap = max(0, int(max_alternatives_per_target))
    replacement_options: Dict[ArtifactRef, Tuple[ArtifactRef, ...]] = {}
    # Automatic WHATIF replacements come only from the factual live catalog and
    # explicit intervention_alternatives. The intervention_registry catalog is an
    # ADD payload source and is not pooled as WHATIF replacements unless a ref is
    # listed in intervention_alternatives.
    for cand in candidates:
        if "REPLACE" not in cand.allowed_operations:
            continue
        contract = contracts.get(cand.artifact_type) if contracts is not None else None
        if contract is None or not contract.allows("REPLACE"):
            continue
        old_env = None
        if registry is not None:
            getter = getattr(registry, "get_by_ref", None)
            if callable(getter):
                old_env = getter(cand.ref, branch_id) or getter(cand.ref)
        kept: List[ArtifactRef] = []
        seen: set[str] = set()

        def _accept(
            new_ref: ArtifactRef,
            new_type: str,
            *,
            from_explicit: bool,
            new_env: Any = None,
        ) -> None:
            if new_ref == cand.ref or str(new_ref) == str(cand.ref):
                return
            if str(new_ref) in seen:
                return
            if not new_type:
                return
            if contract is not None and not contract.replacement_ok(new_type):
                return
            if not from_explicit:
                if str(new_ref) in slice_keys:
                    return
                if old_env is None or not hybrid_replacement_eligible(old_env, new_env):
                    return
            seen.add(str(new_ref))
            kept.append(new_ref)

        for alt in explicit_alts:
            a_type = _artifact_type(registry, alt, branch_id)
            alt_env = None
            if registry is not None:
                getter = getattr(registry, "get_by_ref", None)
                if callable(getter):
                    alt_env = getter(alt, branch_id) or getter(alt)
            if not a_type and intervention_registry is not None:
                a_type = _artifact_type(intervention_registry, alt, branch_id)
            _accept(alt, a_type, from_explicit=True, new_env=alt_env)

        pool = list(live_by_type.get(cand.artifact_type) or [])
        if old_env is not None:
            ranked = rank_hybrid_replacements(
                old_env=old_env, candidates=pool, max_n=cap + 8
            )
            ranked_envs = []
            by_ref = {str(getattr(e, "ref", "")): e for e in pool}
            for r in ranked:
                env = by_ref.get(str(r))
                if env is not None:
                    ranked_envs.append(env)
            pool = ranked_envs
        else:
            pool = []
        for env in pool:
            _accept(
                env.ref,
                str(env.artifact_type),
                from_explicit=False,
                new_env=env,
            )
            if len(kept) >= cap:
                break
        if kept:
            replacement_options[cand.ref] = tuple(kept[:cap])

    slice_sections, slice_tokens = _slice_semantic_context(
        case_slice, registry, branch_id
    )
    add_options: List[ArtifactRef] = []
    add_seen: set[str] = set()
    add_pool_envs: List[Any] = []

    def _candidate_source(*, from_external: bool, from_explicit: bool) -> str:
        if from_explicit:
            return CANDIDATE_SOURCE_EXPLICIT
        if from_external:
            return CANDIDATE_SOURCE_INTERVENTION
        return CANDIDATE_SOURCE_FACTUAL

    def _consider_add(
        env: Any,
        a_type: str,
        *,
        from_external: bool,
        from_explicit: bool,
    ) -> None:
        ref = getattr(env, "ref", None) if env is not None else None
        if ref is None:
            return
        source = _candidate_source(
            from_external=from_external, from_explicit=from_explicit
        )
        in_slice = ref in slice_set
        already_seen = str(ref) in add_seen

        def _record(
            reason: str,
            *,
            reached: bool = False,
            contract: Any = None,
            sourceable: Optional[bool] = None,
            eligible: Optional[bool] = None,
        ) -> None:
            if diagnostics is None:
                return
            permits = (
                bool(contract is not None and contract.allows("ADD"))
                if contract is not None or (a_type and contracts is not None)
                else None
            )
            diagnostics.record_candidate(
                artifact_ref=ref,
                artifact_type=a_type,
                candidate_source=source,
                in_relevant_slice=in_slice,
                contract_permits_add=permits,
                payload_sourceable=sourceable,
                contextual_eligible=True if from_explicit else eligible,
                explicit_exception_applied=from_explicit,
                section_id=_artifact_section_id(env) if env is not None else "",
                context_tokens=_artifact_context_tokens(env) if env is not None else (),
                eligibility_result=reason,
                reached_boundary_checks=reached,
            )

        if in_slice or already_seen:
            _record("already_in_relevant_slice" if in_slice else "already_seen")
            return
        if not a_type or contracts is None:
            _record("missing_artifact_type" if not a_type else "missing_contracts")
            return
        contract = contracts.get(a_type)
        if contract is None or not contract.allows("ADD"):
            _record("contract_disallows_add", contract=contract)
            return
        if from_external and not bool(getattr(contract, "allow_external_add", False)):
            _record("external_add_not_allowed", contract=contract)
            return
        sourceable = _hybrid_add_sourceable(
            ref,
            registry=registry,
            branch_id=branch_id,
            intervention_registry=intervention_registry,
            contract=contract,
        )
        if not sourceable:
            _record(
                "payload_unavailable",
                contract=contract,
                sourceable=False,
            )
            return
        eligible = from_explicit or hybrid_add_eligible(
            env, slice_sections=slice_sections, slice_tokens=slice_tokens
        )
        if not from_explicit and not eligible:
            _record(
                "contextually_unrelated",
                contract=contract,
                sourceable=True,
                eligible=False,
            )
            return
        add_seen.add(str(ref))
        add_pool_envs.append(env)
        _record(
            "reached_boundary_checks",
            reached=True,
            contract=contract,
            sourceable=True,
            eligible=True,
        )

    for env in live:
        _consider_add(
            env, str(env.artifact_type), from_external=False, from_explicit=False
        )
    if intervention_registry is not None:
        catalog = getattr(intervention_registry, "catalog", None) or {}
        for env in catalog.values():
            _consider_add(
                env,
                str(getattr(env, "artifact_type", "")),
                from_external=True,
                from_explicit=str(getattr(env, "ref", "")) in explicit_alt_keys,
            )
    for alt in explicit_alts:
        alt_env = None
        if registry is not None:
            getter = getattr(registry, "get_by_ref", None)
            if callable(getter):
                alt_env = getter(alt, branch_id) or getter(alt)
        if alt_env is None and intervention_registry is not None:
            getter = getattr(intervention_registry, "get_by_ref", None)
            if callable(getter):
                alt_env = getter(alt, branch_id) or getter(alt)
        a_type = str(getattr(alt_env, "artifact_type", "") or "")
        if not a_type:
            a_type = _artifact_type(registry, alt, branch_id)
        if alt_env is None:
            continue
        _consider_add(alt_env, a_type, from_external=False, from_explicit=True)

    add_pool_envs.sort(key=lambda e: _hybrid_add_rank(e, slice_sections))
    add_run_id = str(run_id or getattr(explained_case, "run_id", "") or "")
    add_types = {
        str(getattr(env, "artifact_type", "") or "")
        for env in add_pool_envs
        if str(getattr(env, "artifact_type", "") or "")
    }
    add_boundaries_by_type: Dict[str, Tuple[str, ...]] = {}
    unresolved_by_type: Dict[str, Tuple[str, ...]] = {}
    for a_type in sorted(add_types):
        unresolved: List[str] = []
        bounds = _add_boundaries_for_type(
            artifact_type=a_type,
            case_slice=case_slice,
            checkpoint_index=checkpoint_index,
            all_occurrences=all_occurrences
            or getattr(case_slice, "relevant_occurrences", ()),
            registry=registry,
            branch_id=branch_id,
            unresolved_out=unresolved if diagnostics is not None else None,
        )
        if bounds:
            add_boundaries_by_type[a_type] = bounds
        if unresolved:
            unresolved_by_type[a_type] = tuple(unresolved)
    add_boundaries_by_ref: Dict[ArtifactRef, Tuple[str, ...]] = {}
    add_options: List[ArtifactRef] = []
    omitted_by_cap_refs: List[ArtifactRef] = []
    diagnose = diagnostics is not None
    for env in add_pool_envs:
        if (not diagnose) and len(add_options) >= cap:
            break
        a_type = str(getattr(env, "artifact_type", "") or "")
        candidate_bounds = add_boundaries_by_type.get(a_type, ())
        cap_full = len(add_options) >= cap
        valid: List[str] = []
        seen_b: set[str] = set()
        for b in candidate_bounds:
            if b in seen_b:
                continue
            seen_b.add(b)
            classified = classify_artifact_presence_at_add_boundary(
                env.ref,
                b,
                checkpoint_index=checkpoint_index,
                replay_executor=replay_executor,
                run_id=add_run_id,
            )
            absent = classified.presence == ADD_PRESENCE_ABSENT
            if diagnose:
                if classified.presence == ADD_PRESENCE_ABSENT:
                    inclusion = (
                        "omitted_by_cap" if cap_full else "advertised_absent"
                    )
                elif classified.presence == ADD_PRESENCE_PRESENT:
                    inclusion = "excluded_present"
                else:
                    inclusion = f"excluded_unknown_{classified.detail}"
                diagnostics.record_boundary(
                    artifact_ref=env.ref,
                    artifact_type=a_type,
                    boundary=b,
                    checkpoint_resolved=classified.checkpoint_resolved,
                    checkpoint_alias=classified.checkpoint_alias,
                    checkpoint_id=classified.checkpoint_id,
                    state_inspectable=classified.state_inspectable,
                    presence=classified.presence,
                    presence_detail=classified.detail,
                    inclusion_reason=inclusion,
                    advertised=bool(absent and not cap_full),
                    omitted_by_cap=bool(absent and cap_full),
                )
            if not absent:
                continue
            valid.append(b)
        if diagnose and not candidate_bounds:
            diagnostics.record_boundary(
                artifact_ref=env.ref,
                artifact_type=a_type,
                boundary="",
                checkpoint_resolved=False,
                state_inspectable=False,
                presence=ADD_PRESENCE_UNKNOWN,
                presence_detail="no_suitable_consumer_boundary",
                inclusion_reason="no_suitable_consumer_boundary",
            )
            for text in unresolved_by_type.get(a_type, ()):
                diagnostics.record_boundary(
                    artifact_ref=env.ref,
                    artifact_type=a_type,
                    boundary=text,
                    checkpoint_resolved=False,
                    state_inspectable=False,
                    presence=ADD_PRESENCE_UNKNOWN,
                    presence_detail="checkpoint_unresolvable",
                    inclusion_reason="excluded_unknown_checkpoint_unresolvable",
                )
        if not valid:
            continue
        if cap_full:
            omitted_by_cap_refs.append(env.ref)
            continue
        add_boundaries_by_ref[env.ref] = tuple(valid)
        add_options.append(env.ref)
    union_bounds: List[str] = []
    seen_union: set[str] = set()
    for ref in add_options:
        for b in add_boundaries_by_ref.get(ref, ()):
            if b in seen_union:
                continue
            seen_union.add(b)
            union_bounds.append(b)

    vocab = HybridInterventionVocabulary(
        what_refs=tuple(what_refs),
        intervention_candidates=candidates,
        replacement_options=replacement_options,
        add_options=tuple(add_options),
        add_boundaries_by_ref=add_boundaries_by_ref,
        add_boundaries=tuple(union_bounds),
        max_alternatives_per_target=cap,
    )
    if diagnostics is not None:
        diagnostics.finalize(
            advertised_refs=vocab.add_options,
            advertised_boundaries=vocab.add_boundaries_by_ref,
            executable_add_count=hybrid_opportunity_counts(vocab)["add"],
            omitted_by_cap_refs=omitted_by_cap_refs,
        )
    return vocab


def format_hybrid_vocabulary(
    *,
    vocabulary: HybridInterventionVocabulary,
    registry: Any = None,
    branch_id: str = "factual",
) -> str:
    """Structured hybrid vocabulary. Not an exact executable-query copy list."""
    parts: List[str] = [
        "INTERVENTION TARGETS",
    ]
    slice_like = [
        c
        for c in vocabulary.intervention_candidates
        if "ADD" not in c.allowed_operations or len(c.allowed_operations) > 1
    ]
    shown = False
    for c in vocabulary.intervention_candidates:
        remove_bounds = c.boundaries_for("REMOVE")
        replace_bounds = c.boundaries_for("REPLACE")
        replacements = vocabulary.replacements_for(c.ref)
        ops: List[str] = []
        if remove_bounds and "REMOVE" in c.allowed_operations:
            ops.append("REMOVE")
        if (
            "REPLACE" in c.allowed_operations
            and replace_bounds
            and replacements
        ):
            ops.append("WHATIF")
        if not ops:
            continue
        shown = True
        parts.append("")
        parts.append(f"{c.artifact_type} {c.ref}")
        parts.append("")
        parts.append("Allowed operations:")
        for op in ops:
            parts.append(f"- {op}")
        if "REMOVE" in ops:
            parts.append("")
            parts.append("REMOVE boundaries:")
            for b in remove_bounds:
                parts.append(f"- {b}")
        if "WHATIF" in ops:
            parts.append("")
            parts.append("WHATIF boundaries:")
            for b in replace_bounds:
                parts.append(f"- {b}")
            parts.append("")
            parts.append("Compatible replacements:")
            for alt in replacements:
                alt_env = None
                if registry is not None:
                    getter = getattr(registry, "get_by_ref", None)
                    if callable(getter):
                        alt_env = getter(alt, branch_id) or getter(alt)
                desc = _artifact_description(alt_env) if alt_env is not None else ""
                if desc:
                    parts.append(f'- {alt} — "{desc}"')
                else:
                    parts.append(f"- {alt}")
    if not shown:
        parts.append("")
        parts.append("(no intervenable targets)")

    parts.append("")
    parts.append("ADDABLE ARTIFACTS")
    if not vocabulary.add_options:
        parts.append("(none)")
    else:
        for ref in vocabulary.add_options:
            a_type = _artifact_type(registry, ref, branch_id) or "Unknown"
            env = None
            if registry is not None:
                getter = getattr(registry, "get_by_ref", None)
                if callable(getter):
                    env = getter(ref, branch_id) or getter(ref)
            desc = _artifact_description(env) if env is not None else ""
            suffix = f' — {a_type}: "{desc}"' if desc else f" — {a_type}"
            parts.append("")
            parts.append(f"{a_type} {ref}{suffix}")
            parts.append("Valid ADD boundaries:")
            bounds = vocabulary.add_boundaries_for(ref)
            if not bounds:
                parts.append("(none)")
            else:
                for b in bounds:
                    parts.append(f"- {b}")
    _ = slice_like
    return "\n".join(parts)


def validate_resolved_against_hybrid_vocabulary(
    resolved: Any,
    vocabulary: HybridInterventionVocabulary,
) -> None:
    """Raise QueryValidationError if the resolved query was not advertised."""
    from .queries import QueryValidationError

    op = str(getattr(resolved, "operation", "") or "").upper()
    if op in {"DONE", ""}:
        return
    target = getattr(resolved, "target_ref", None)
    boundary = str(getattr(resolved, "boundary_raw", "") or "")
    if op == "WHAT":
        raise QueryValidationError(WHAT_DISABLED_IN_INTERROGATION)
    if op == "REMOVE":
        cand = vocabulary.candidate_for(target)
        if cand is None or "REMOVE" not in cand.allowed_operations:
            raise QueryValidationError(
                f"REMOVE target {target} is not an advertised intervention candidate"
            )
        bounds = cand.boundaries_for("REMOVE")
        if boundary not in bounds:
            raise QueryValidationError(
                f"Boundary {boundary!r} is not in the advertised REMOVE vocabulary "
                f"for {target}"
            )
        return
    if op in {"WHATIF", "REPLACE"}:
        cand = vocabulary.candidate_for(target)
        if cand is None or "REPLACE" not in cand.allowed_operations:
            raise QueryValidationError(
                f"WHATIF target {target} is not an advertised REPLACE-capable candidate"
            )
        replacement = getattr(resolved, "replacement_ref", None)
        allowed = vocabulary.replacements_for(target)
        allowed_s = {str(r) for r in allowed}
        if replacement is None or str(replacement) not in allowed_s:
            raise QueryValidationError(
                f"Replacement {replacement} is not in the advertised replacement "
                f"vocabulary for {target}"
            )
        bounds = cand.boundaries_for("REPLACE")
        if boundary not in bounds:
            raise QueryValidationError(
                f"Boundary {boundary!r} is not in the advertised WHATIF vocabulary "
                f"for {target}"
            )
        return
    if op == "ADD":
        add_s = {str(r) for r in vocabulary.add_options}
        if target is None or str(target) not in add_s:
            raise QueryValidationError(
                f"ADD artifact {target} is not in the advertised ADD vocabulary"
            )
        allowed_bounds = vocabulary.add_boundaries_for(target)
        if boundary not in allowed_bounds:
            raise QueryValidationError(
                f"Boundary {boundary!r} is not in the advertised ADD vocabulary "
                f"for {target}"
            )
        return
    raise QueryValidationError(
        f"Operation {op} is not in the advertised hybrid vocabulary"
    )


def constrained_opportunity_counts(executable_queries: str) -> Dict[str, int]:
    """Count remaining exact executable forms by operation (constrained mode)."""
    counts = {"what": 0, "remove": 0, "add": 0, "whatif": 0}
    for form in executable_query_forms(executable_queries):
        low = form.lower()
        if low.startswith("whatif("):
            counts["whatif"] += 1
        elif low.startswith("what("):
            counts["what"] += 1
        elif low.startswith("remove("):
            counts["remove"] += 1
        elif low.startswith("add("):
            counts["add"] += 1
    return counts


def hybrid_opportunity_counts(
    vocabulary: HybridInterventionVocabulary,
    *,
    executed_signatures: Optional[AbstractSet[Any]] = None,
) -> Dict[str, int]:
    """Count advertised hybrid combinations, minus executed signatures when given."""
    excluded = executed_signatures or set()

    def _used(op: str, target: Any, replacement: Any = None, boundary: str = "") -> bool:
        if not excluded:
            return False
        for sig in excluded:
            if str(getattr(sig, "operation", "") or "").upper() != op:
                continue
            if getattr(sig, "target_ref", None) != target and str(
                getattr(sig, "target_ref", None)
            ) != str(target):
                continue
            if op == "WHATIF":
                if getattr(sig, "replacement_ref", None) != replacement and str(
                    getattr(sig, "replacement_ref", None)
                ) != str(replacement):
                    continue
            raw = str(getattr(sig, "boundary_raw", "") or "")
            alias = str(getattr(sig, "checkpoint_alias", "") or "")
            if boundary and boundary not in {raw, alias} and raw and raw != boundary:
                if alias != boundary:
                    continue
            return True
        return False

    what = 0
    for ref in vocabulary.what_refs:
        if not _used("WHAT", ref):
            what += 1
    remove = 0
    whatif = 0
    for cand in vocabulary.intervention_candidates:
        if "REMOVE" in cand.allowed_operations:
            for b in cand.boundaries_for("REMOVE"):
                if not _used("REMOVE", cand.ref, boundary=b):
                    remove += 1
        if "REPLACE" in cand.allowed_operations:
            for alt in vocabulary.replacements_for(cand.ref):
                for b in cand.boundaries_for("REPLACE"):
                    if not _used("WHATIF", cand.ref, replacement=alt, boundary=b):
                        whatif += 1
    add = 0
    for ref in vocabulary.add_options:
        for b in vocabulary.add_boundaries_for(ref):
            if not _used("ADD", ref, boundary=b):
                add += 1
    return {"what": what, "remove": remove, "add": add, "whatif": whatif}


def format_hybrid_intervenable_handoff(
    *,
    target_ref: Any,
    vocabulary: HybridInterventionVocabulary,
    has_successful_intervention: bool,
) -> str:
    if has_successful_intervention:
        return ""
    cand = vocabulary.candidate_for(target_ref)
    if cand is None:
        return ""
    ops: List[str] = []
    if "REMOVE" in cand.allowed_operations and cand.boundaries_for("REMOVE"):
        ops.append("REMOVE")
    replacements = vocabulary.replacements_for(target_ref)
    if (
        "REPLACE" in cand.allowed_operations
        and cand.boundaries_for("REPLACE")
        and replacements
    ):
        ops.append("WHATIF")
    if not ops:
        return ""
    parts = [
        "The inspected artifact is intervenable.",
        "",
        f"Artifact: {target_ref}",
        "",
        "Allowed operations:",
    ]
    for op in ops:
        parts.append(op)
    if "REMOVE" in ops:
        parts.append("")
        parts.append("Valid REMOVE boundaries:")
        for b in cand.boundaries_for("REMOVE"):
            parts.append(f"- {b}")
    if "WHATIF" in ops:
        parts.append("")
        parts.append("Valid WHATIF boundaries:")
        for b in cand.boundaries_for("REPLACE"):
            parts.append(f"- {b}")
        parts.append("")
        parts.append("Compatible replacements:")
        for alt in replacements:
            parts.append(f"- {alt}")
    parts.append("")
    parts.append(
        "Compose one intervention from this vocabulary if it would help answer "
        "the question. Do not copy a pre-written executable form."
    )
    return "\n".join(parts)


def format_hybrid_done_nudge(vocabulary: HybridInterventionVocabulary) -> str:
    has_cf = False
    for cand in vocabulary.intervention_candidates:
        if cand.boundaries_for("REMOVE") or (
            cand.boundaries_for("REPLACE") and vocabulary.replacements_for(cand.ref)
        ):
            has_cf = True
            break
    if any(vocabulary.add_boundaries_for(ref) for ref in vocabulary.add_options):
        has_cf = True
    if not has_cf:
        return ""
    return (
        "At least one valid counterfactual intervention remains.\n"
        "If another counterfactual would help answer the question, compose one "
        "from the available REMOVE / ADD / WHATIF vocabulary.\n"
        "Otherwise return DONE."
    )


def format_hybrid_last_chance_nudge(vocabulary: HybridInterventionVocabulary) -> str:
    return (
        "You have not yet executed a counterfactual intervention.\n"
        "If a counterfactual would help answer the question, compose one from "
        "the available REMOVE / ADD / WHATIF vocabulary.\n"
        "Otherwise return DONE."
    )
