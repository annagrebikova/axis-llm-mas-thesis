"""Branch metadata for factual and counterfactual replays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import uuid4


@dataclass(frozen=True)
class BranchMetadata:
    """Independent replay branch forked from a factual checkpoint.

    ``thread_id`` is the LangGraph checkpoint thread (factual ``run_id``).
    ``branch_id`` is the logical AXIS execution-branch identity.
    """

    run_id: str
    branch_id: str
    branch_type: str  # factual | noop | counterfactual
    parent_branch_id: str
    origin_checkpoint_alias: str
    origin_checkpoint_id: str = ""
    thread_id: str = ""
    intervention_id: Optional[str] = None
    fork_checkpoint_id: Optional[str] = None
    final_checkpoint_id: Optional[str] = None

    @staticmethod
    def factual(run_id: str) -> "BranchMetadata":
        return BranchMetadata(
            run_id=run_id,
            branch_id="factual",
            branch_type="factual",
            parent_branch_id="",
            origin_checkpoint_alias="c_0",
            origin_checkpoint_id="",
            thread_id=run_id,
        )


@dataclass
class BranchStore:
    """Tracks sibling forks per factual run; never mutates the factual branch."""

    _by_run: Dict[str, Dict[str, BranchMetadata]] = field(default_factory=dict)

    def register(self, meta: BranchMetadata) -> BranchMetadata:
        bucket = self._by_run.setdefault(meta.run_id, {})
        if meta.branch_id in bucket:
            raise ValueError(
                f"Branch already exists: run={meta.run_id!r} branch={meta.branch_id!r}"
            )
        bucket[meta.branch_id] = meta
        return meta

    def get(self, run_id: str, branch_id: str) -> BranchMetadata:
        bucket = self._by_run.get(run_id) or {}
        if branch_id not in bucket:
            raise KeyError(f"Unknown branch: run={run_id!r} branch={branch_id!r}")
        return bucket[branch_id]

    def get_optional(self, run_id: str, branch_id: str) -> Optional[BranchMetadata]:
        return (self._by_run.get(run_id) or {}).get(branch_id)

    def update(self, meta: BranchMetadata) -> BranchMetadata:
        bucket = self._by_run.setdefault(meta.run_id, {})
        bucket[meta.branch_id] = meta
        return meta

    def fork(
        self,
        *,
        parent: BranchMetadata,
        branch_type: str,
        origin_checkpoint_alias: str,
        origin_checkpoint_id: str = "",
        intervention_id: Optional[str] = None,
        branch_id: Optional[str] = None,
    ) -> BranchMetadata:
        """Create an independent sibling fork (never stacks interventions)."""
        if parent.branch_type != "factual":
            raise ValueError(
                "Interventions must fork from the factual branch, not from "
                "another intervention result"
            )
        if branch_type not in {"noop", "counterfactual"}:
            raise ValueError(f"Unsupported branch_type: {branch_type}")
        prefix = "noop" if branch_type == "noop" else "cf"
        meta = BranchMetadata(
            run_id=parent.run_id,
            branch_id=branch_id or f"{prefix}_{uuid4().hex[:10]}",
            branch_type=branch_type,
            parent_branch_id=parent.branch_id,
            origin_checkpoint_alias=origin_checkpoint_alias,
            origin_checkpoint_id=origin_checkpoint_id or "",
            # Same LangGraph thread as the factual run (Prompt 10).
            thread_id=parent.thread_id or parent.run_id,
            intervention_id=intervention_id,
        )
        return self.register(meta)

    def list_siblings(
        self,
        parent_branch_id: str,
        *,
        run_id: Optional[str] = None,
    ) -> List[BranchMetadata]:
        if run_id is not None:
            bucket = self._by_run.get(run_id) or {}
            return [
                b for b in bucket.values() if b.parent_branch_id == parent_branch_id
            ]
        out: List[BranchMetadata] = []
        for bucket in self._by_run.values():
            out.extend(
                b for b in bucket.values() if b.parent_branch_id == parent_branch_id
            )
        return out

    def for_run(self, run_id: str) -> Dict[str, BranchMetadata]:
        return dict(self._by_run.get(run_id) or {})

    @property
    def branches(self) -> Dict[str, BranchMetadata]:
        """Compatibility flat view (may collide on ``factual`` across runs)."""
        flat: Dict[str, BranchMetadata] = {}
        for run_id, bucket in self._by_run.items():
            for bid, meta in bucket.items():
                key = bid if bid not in flat else f"{run_id}:{bid}"
                flat[key] = meta
        return flat
