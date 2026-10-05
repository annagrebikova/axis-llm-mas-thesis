"""Structured AXIS interrogation query parser, resolver, and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..artifacts.models import ArtifactRef
from ..artifacts.registry import ArtifactRegistry
from ..replay.checkpoints import BoundarySelector, CheckpointIndex
from ..replay.interventions import InterventionContractRegistry
from .models import (
    AxisQueryContext,
    QuerySignature,
    ResolvedQuery,
)


class QueryValidationError(ValueError):
    """Raised when an AXIS query fails syntax or contract validation."""


@dataclass(frozen=True)
class AxisQuery:
    """Parsed structured interrogation query (p_n before resolution)."""

    operation: str
    args: Tuple[str, ...]
    raw_text: str


_QUERY_RE = re.compile(
    r"^\s*(WHAT|REMOVE|ADD|WHATIF|DONE)\s*(?:\((.*)\))?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Artifact alias index
# ---------------------------------------------------------------------------

@dataclass
class ArtifactAliasIndex:
    """Deterministic alias → ArtifactRef resolution.

    Aliases come from three sources (highest priority first):
    1. Explicit caller-supplied aliases.
    2. Explained-case convenience aliases (candidate/decision/assessment).
    3. Unambiguous logical-id resolution from a scope set.
    """

    _explicit: Dict[str, ArtifactRef] = field(default_factory=dict)
    _scope_by_logical: Dict[str, List[ArtifactRef]] = field(default_factory=dict)

    @staticmethod
    def build(
        *,
        explicit_aliases: Mapping[str, ArtifactRef] | None = None,
        explained_case: object | None = None,
        scope_refs: Sequence[ArtifactRef] = (),
        intervention_alternatives: Mapping[str, ArtifactRef] | None = None,
    ) -> "ArtifactAliasIndex":
        idx = ArtifactAliasIndex()

        # Scope (lowest priority: unambiguous logical ids)
        for ref in scope_refs:
            idx._scope_by_logical.setdefault(ref.logical_id, []).append(ref)

        # Explained-case convenience aliases
        if explained_case is not None:
            case_ref = getattr(explained_case, "artifact_ref", None)
            dec_ref = getattr(explained_case, "decision_ref", None)
            ass_ref = getattr(explained_case, "assessment_ref", None)
            for alias, ref in [
                ("candidate", case_ref),
                ("decision", dec_ref),
                ("assessment", ass_ref),
            ]:
                if ref is None:
                    continue
                if alias in (explicit_aliases or {}):
                    existing = (explicit_aliases or {})[alias]
                    if existing != ref:
                        raise QueryValidationError(
                            f"Alias {alias!r} conflicts: explicit={existing} vs case={ref}"
                        )
                else:
                    idx._explicit[alias] = ref

        # Explicit caller aliases (highest priority)
        for alias, ref in (explicit_aliases or {}).items():
            idx._explicit[alias] = ref

        # Intervention alternatives
        for alias, ref in (intervention_alternatives or {}).items():
            if alias in idx._explicit:
                existing = idx._explicit[alias]
                if existing != ref:
                    raise QueryValidationError(
                        f"Alias {alias!r} conflicts: explicit={existing} vs alternative={ref}"
                    )
            idx._explicit[alias] = ref

        return idx

    def resolve(self, token: str) -> ArtifactRef:
        """Resolve a token to an exact ArtifactRef."""
        token = token.strip().strip("'\"")

        # 1. Explicit alias match
        if token in self._explicit:
            return self._explicit[token]

        # 2. Canonical :vN syntax
        if ":v" in token:
            logical, _, ver = token.rpartition(":v")
            try:
                return ArtifactRef(logical_id=logical, version=int(ver))
            except (ValueError, TypeError) as exc:
                raise QueryValidationError(
                    f"Invalid version in {token!r}: {exc}"
                ) from exc

        # 3. Unambiguous scope resolution (bare logical id)
        candidates = self._scope_by_logical.get(token) or []
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise QueryValidationError(
                f"Ambiguous artifact version for {token!r}: "
                f"{sorted(str(r) for r in candidates)}"
            )

        raise QueryValidationError(f"Unknown artifact: {token!r}")


# ---------------------------------------------------------------------------
# Parser (syntax only)
# ---------------------------------------------------------------------------

class AxisQueryParser:
    """Parse structured LLM output into typed AXIS queries.

    This class handles only syntax. Semantic resolution is in ``AxisQueryResolver``.
    """

    def __init__(
        self,
        *,
        alias_map: Optional[Dict[str, ArtifactRef]] = None,
        checkpoint_index: Optional[CheckpointIndex] = None,
        contracts: Optional[InterventionContractRegistry] = None,
        tested: Optional[set] = None,
    ) -> None:
        self.alias_map = dict(alias_map or {})
        self.checkpoint_index = checkpoint_index
        self.contracts = contracts or InterventionContractRegistry()
        self.tested = tested if tested is not None else set()

    def parse(self, text: str) -> AxisQuery:
        """Parse structured LLM output into an AxisQuery."""
        cleaned = (text or "").strip().strip("`")
        cleaned = cleaned.replace("<|endoftext|>", "").strip()
        m = _QUERY_RE.match(cleaned)
        if not m:
            raise QueryValidationError(f"Invalid query syntax: {text!r}")
        op = m.group(1).upper()
        arg_str = (m.group(2) or "").strip()
        if op == "DONE":
            if arg_str:
                raise QueryValidationError("DONE takes no arguments")
            return AxisQuery(operation="DONE", args=(), raw_text=cleaned)
        args = tuple(a.strip() for a in arg_str.split(",") if a.strip())
        if op == "WHAT" and len(args) != 1:
            raise QueryValidationError("WHAT(target) requires one argument")
        if op in {"REMOVE", "ADD"} and len(args) != 2:
            raise QueryValidationError(
                f"{op}(artifact, boundary) requires two arguments"
            )
        if op == "WHATIF" and len(args) != 3:
            raise QueryValidationError(
                "WHATIF(old, new, boundary) requires three arguments"
            )
        return AxisQuery(operation=op, args=args, raw_text=cleaned)

    # -- Compatibility delegates (used by old callers / tests) ----------------

    def resolve_artifact(self, token: str) -> ArtifactRef:
        """Resolve via explicit alias map or canonical :vN syntax."""
        token = token.strip().strip("'\"")
        if token in self.alias_map:
            return self.alias_map[token]
        if ":v" in token:
            logical, _, ver = token.rpartition(":v")
            return ArtifactRef(logical_id=logical, version=int(ver))
        # bare logical id → version 1 (legacy compat only)
        return ArtifactRef(logical_id=token, version=1)

    def resolve_boundary(self, token: str) -> str:
        """Resolve boundary token to a checkpoint alias (legacy compat)."""
        token = token.strip().strip("'\"")
        if self.checkpoint_index is None:
            return token
        if token.startswith("c_"):
            rec = self.checkpoint_index.resolve(BoundarySelector.by_alias(token))
            return rec.alias
        if "#" in token:
            node, _, idx = token.partition("#")
            before = False
            node_tok = node
            if node.lower().startswith("before "):
                before = True
                node_tok = node[7:].strip()
            elif node.lower().startswith("after "):
                node_tok = node[6:].strip()
            sel = (
                BoundarySelector.before_occurrence(node_tok, int(idx))
                if before
                else BoundarySelector.after_occurrence(node_tok, int(idx))
            )
            return self.checkpoint_index.resolve(sel).alias
        if token.lower().startswith("before "):
            rest = token[7:].strip()
            if "#" in rest:
                node, _, idx = rest.partition("#")
                sel = BoundarySelector.before_occurrence(node, int(idx))
                return self.checkpoint_index.resolve(sel).alias
            sel = BoundarySelector.before_next_node(rest)
            return self.checkpoint_index.resolve(sel).alias
        if token.lower().startswith("after "):
            rest = token[6:].strip()
            if "#" in rest:
                node, _, idx = rest.partition("#")
                sel = BoundarySelector.after_occurrence(node, int(idx))
                return self.checkpoint_index.resolve(sel).alias
        return token

    def resolve(self, query: AxisQuery) -> ResolvedQuery:
        """Legacy: validate and resolve a parsed query to p_n_resolved."""
        key = query.raw_text.lower()
        if key in self.tested and query.operation != "DONE":
            raise QueryValidationError(
                f"Duplicate tested query: {query.raw_text}"
            )
        if query.operation == "DONE":
            return ResolvedQuery(operation="DONE", raw_text=query.raw_text)
        if query.operation == "WHAT":
            ref = self.resolve_artifact(query.args[0])
            resolved = ResolvedQuery(
                operation="WHAT",
                raw_text=query.raw_text,
                target_ref=ref,
            )
            self.tested.add(key)
            return resolved
        if query.operation in {"REMOVE", "ADD"}:
            ref = self.resolve_artifact(query.args[0])
            alias = self.resolve_boundary(query.args[1])
            resolved = ResolvedQuery(
                operation=query.operation,
                raw_text=query.raw_text,
                target_ref=ref,
                checkpoint_alias=alias,
                boundary_raw=query.args[1],
            )
            self.tested.add(key)
            return resolved
        if query.operation == "WHATIF":
            old = self.resolve_artifact(query.args[0])
            new = self.resolve_artifact(query.args[1])
            alias = self.resolve_boundary(query.args[2])
            resolved = ResolvedQuery(
                operation="WHATIF",
                raw_text=query.raw_text,
                target_ref=old,
                replacement_ref=new,
                checkpoint_alias=alias,
                boundary_raw=query.args[2],
            )
            self.tested.add(key)
            return resolved
        raise QueryValidationError(f"Unsupported operation: {query.operation}")


# ---------------------------------------------------------------------------
# Resolver (semantic validation, AXIS-ready)
# ---------------------------------------------------------------------------

def _resolve_boundary_strict(
    token: str,
    checkpoint_index: object,
) -> Tuple[str, str, str]:
    """Resolve a boundary token to (alias, checkpoint_id, checkpoint_ns).

    Raises QueryValidationError on ambiguity or unknown boundaries.
    """
    token = token.strip().strip("'\"")

    if not checkpoint_index:
        raise QueryValidationError("No checkpoint index available for boundary resolution")

    idx = checkpoint_index  # CheckpointIndex

    # Explicit checkpoint alias (c_N)
    if token.startswith("c_"):
        try:
            rec = idx.resolve(BoundarySelector.by_alias(token))
        except (KeyError, ValueError) as exc:
            raise QueryValidationError(f"Unknown checkpoint alias: {token!r}") from exc
        return rec.alias, rec.checkpoint_id, rec.checkpoint_ns

    # "before node#N" or "after node#N"
    lower = token.lower()
    if lower.startswith("before ") or lower.startswith("after "):
        is_before = lower.startswith("before ")
        rest = token[7:].strip() if is_before else token[6:].strip()

        if "#" in rest:
            node, _, idx_str = rest.partition("#")
            try:
                occ_idx = int(idx_str)
            except ValueError:
                raise QueryValidationError(
                    f"Invalid occurrence index in boundary: {token!r}"
                )
            sel = (
                BoundarySelector.before_occurrence(node.strip(), occ_idx)
                if is_before
                else BoundarySelector.after_occurrence(node.strip(), occ_idx)
            )
        else:
            if is_before:
                sel = BoundarySelector.before_next_node(rest.strip())
            else:
                raise QueryValidationError(
                    f"'after' requires node#N syntax: {token!r}"
                )

        try:
            rec = idx.resolve(sel)
        except (KeyError, ValueError) as exc:
            raise QueryValidationError(
                f"Cannot resolve boundary {token!r}: {exc}"
            ) from exc
        return rec.alias, rec.checkpoint_id, rec.checkpoint_ns

    # Bare node#N without before/after → reject as ambiguous
    if "#" in token:
        raise QueryValidationError(
            f"Occurrence boundary must explicitly specify before or after: {token!r}"
        )

    # Unrecognized
    raise QueryValidationError(f"Unknown boundary: {token!r}")


def _artifact_type_from_registry(
    registry: ArtifactRegistry,
    ref: ArtifactRef,
    branch_id: str,
) -> str:
    env = registry.get_by_ref(ref, branch_id=branch_id) or registry.get_by_ref(ref)
    return getattr(env, "artifact_type", "") if env else ""


class AxisQueryResolver:
    """Semantic resolution of parsed AXIS queries (AXIS-ready path).

    By default, the resolver manages its own ``_executed`` set for duplicate
    detection (Prompt-13 compat). Prompt-14 callers may pass
    ``executed_signatures`` and ``reserve=False`` to decouple duplicate
    tracking from resolution.
    """

    def __init__(self) -> None:
        self._executed: Set[QuerySignature] = set()

    @property
    def executed_signatures(self) -> Set[QuerySignature]:
        return set(self._executed)

    def resolve(
        self,
        query: AxisQuery,
        context: AxisQueryContext,
        *,
        executed_signatures: Optional[Set[QuerySignature]] = None,
        reserve: bool = True,
    ) -> ResolvedQuery:
        """Resolve a parsed query into p_n_resolved with full validation.

        Args:
            executed_signatures: If given, used for duplicate detection instead
                of the resolver's internal set.
            reserve: If True (default/compat), add the signature to the internal
                ``_executed`` set on success. Prompt-14 sets False so the
                controller adds only after successful execution.
        """
        if query.operation == "DONE":
            return ResolvedQuery(operation="DONE", raw_text=query.raw_text)

        dup_set = executed_signatures if executed_signatures is not None else self._executed

        alias_idx = ArtifactAliasIndex.build(
            explicit_aliases=dict(context.alias_map),
            explained_case=context.explained_case,
            scope_refs=list(
                getattr(context.case_slice, "relevant_artifact_refs", ())
            ),
            intervention_alternatives=dict(context.intervention_alternatives),
        )

        branch_id = context.explained_case.branch_id

        if query.operation == "WHAT":
            ref = alias_idx.resolve(query.args[0])
            slice_refs = set(
                getattr(context.case_slice, "relevant_artifact_refs", ())
            )
            if ref not in slice_refs:
                raise QueryValidationError(
                    f"Artifact {ref} is not in the explained case slice"
                )
            a_type = _artifact_type_from_registry(
                context.registry, ref, branch_id
            )
            rq = ResolvedQuery(
                operation="WHAT",
                raw_text=query.raw_text,
                target_ref=ref,
                target_artifact_type=a_type or None,
            )
            sig = QuerySignature.from_resolved(rq)
            if sig in dup_set:
                raise QueryValidationError(
                    f"Duplicate query: {query.raw_text}"
                )
            if reserve:
                self._executed.add(sig)
            return rq

        if query.operation in {"REMOVE", "ADD"}:
            ref = alias_idx.resolve(query.args[0])
            a_type = _artifact_type_from_registry(
                context.registry, ref, branch_id
            )

            if query.operation == "REMOVE":
                slice_refs = set(
                    getattr(context.case_slice, "relevant_artifact_refs", ())
                )
                if ref not in slice_refs:
                    raise QueryValidationError(
                        f"REMOVE target {ref} not in case slice"
                    )
                if a_type:
                    contract = context.contracts.get(a_type)
                    if contract is None or not contract.allows("REMOVE"):
                        raise QueryValidationError(
                            f"No REMOVE contract for artifact type {a_type!r}"
                        )

            if query.operation == "ADD":
                if a_type:
                    contract = context.contracts.get(a_type)
                    if contract is None or not contract.allows("ADD"):
                        raise QueryValidationError(
                            f"No ADD contract for artifact type {a_type!r}"
                        )

            alias, ckpt_id, ckpt_ns = _resolve_boundary_strict(
                query.args[1], context.checkpoint_index
            )
            rq = ResolvedQuery(
                operation=query.operation,
                raw_text=query.raw_text,
                target_ref=ref,
                checkpoint_alias=alias,
                checkpoint_id=ckpt_id,
                checkpoint_ns=ckpt_ns,
                boundary_raw=query.args[1],
                target_artifact_type=a_type or None,
            )
            sig = QuerySignature.from_resolved(rq)
            if sig in dup_set:
                raise QueryValidationError(
                    f"Duplicate intervention: {query.raw_text}"
                )
            if reserve:
                self._executed.add(sig)
            return rq

        if query.operation == "WHATIF":
            old_ref = alias_idx.resolve(query.args[0])
            new_ref = alias_idx.resolve(query.args[1])

            slice_refs = set(
                getattr(context.case_slice, "relevant_artifact_refs", ())
            )
            if old_ref not in slice_refs:
                raise QueryValidationError(
                    f"WHATIF old target {old_ref} not in case slice"
                )

            old_type = _artifact_type_from_registry(
                context.registry, old_ref, branch_id
            )
            new_type = _artifact_type_from_registry(
                context.registry, new_ref, branch_id
            )
            if not new_type and context.intervention_registry:
                new_type = _artifact_type_from_registry(
                    context.intervention_registry, new_ref, branch_id
                )

            if old_type:
                contract = context.contracts.get(old_type)
                if contract is None or not contract.allows("REPLACE"):
                    raise QueryValidationError(
                        f"No REPLACE contract for {old_type!r}"
                    )
                if new_type and not contract.replacement_ok(new_type):
                    raise QueryValidationError(
                        f"Replacement type {new_type!r} incompatible with {old_type!r}"
                    )

            alias, ckpt_id, ckpt_ns = _resolve_boundary_strict(
                query.args[2], context.checkpoint_index
            )
            rq = ResolvedQuery(
                operation="WHATIF",
                raw_text=query.raw_text,
                target_ref=old_ref,
                replacement_ref=new_ref,
                checkpoint_alias=alias,
                checkpoint_id=ckpt_id,
                checkpoint_ns=ckpt_ns,
                boundary_raw=query.args[2],
                target_artifact_type=old_type or None,
                replacement_artifact_type=new_type or None,
            )
            sig = QuerySignature.from_resolved(rq)
            if sig in dup_set:
                raise QueryValidationError(
                    f"Duplicate intervention: {query.raw_text}"
                )
            if reserve:
                self._executed.add(sig)
            return rq

        raise QueryValidationError(f"Unsupported operation: {query.operation}")
