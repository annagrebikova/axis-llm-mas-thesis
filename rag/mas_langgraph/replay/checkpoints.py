"""Checkpoint index over LangGraph state history.

Connects real LangGraph checkpoints with Prompt-7 occurrence history so that
boundaries such as ``before verify_candidate_mcqs_against_evidence#2`` resolve
to one concrete factual checkpoint.

``after_occurrences`` holds full deterministic occurrence IDs
(``run:branch:node#n``). ``after_occurrence_labels`` holds human-readable
``node#n`` labels. Resolve matches either form.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..provenance.models import OccurrenceRecord


@dataclass(frozen=True)
class CheckpointRecord:
    """Readable alias over a LangGraph checkpoint configuration.

    ``superstep`` is the real LangGraph ``metadata["step"]`` value (may be
    ``None``, ``-1``, ``0``, …). It is **not** the alias index.
    """

    alias: str
    thread_id: str
    checkpoint_id: str
    checkpoint_ns: str = ""
    superstep: Optional[int] = None
    next_nodes: Tuple[str, ...] = ()
    # Canonical deterministic occurrence IDs first visible at this checkpoint.
    after_occurrences: Tuple[str, ...] = ()
    # Human-readable labels such as ``verify_...#2``.
    after_occurrence_labels: Tuple[str, ...] = ()
    raw_config: Optional[dict] = None
    source: str = "langgraph"
    parent_checkpoint_id: Optional[str] = None


@dataclass(frozen=True)
class BoundarySelector:
    """Public query for resolving a factual checkpoint boundary."""

    kind: str  # alias | after_occurrence | before_occurrence | before_next_node
    value: str
    occurrence_index: Optional[int] = None

    @staticmethod
    def by_alias(alias: str) -> "BoundarySelector":
        return BoundarySelector(kind="alias", value=alias)

    @staticmethod
    def after_occurrence(node_id: str, index: int) -> "BoundarySelector":
        return BoundarySelector(
            kind="after_occurrence", value=node_id, occurrence_index=index
        )

    @staticmethod
    def before_occurrence(node_id: str, index: int) -> "BoundarySelector":
        return BoundarySelector(
            kind="before_occurrence", value=node_id, occurrence_index=index
        )

    @staticmethod
    def before_next_node(node_id: str) -> "BoundarySelector":
        return BoundarySelector(kind="before_next_node", value=node_id)


def _configurable(config: Any) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    cfg = config.get("configurable")
    return dict(cfg) if isinstance(cfg, dict) else {}


def extract_checkpoint_id(snapshot: Any) -> str:
    cfg = getattr(snapshot, "config", None) or {}
    ckpt = _configurable(cfg)
    cid = ckpt.get("checkpoint_id") or getattr(snapshot, "id", None)
    if cid is None:
        raise ValueError("Snapshot has no checkpoint_id")
    return str(cid)


def extract_thread_id(snapshot: Any, default: str = "") -> str:
    cfg = getattr(snapshot, "config", None) or {}
    ckpt = _configurable(cfg)
    return str(ckpt.get("thread_id") or default)


def extract_checkpoint_ns(snapshot: Any) -> str:
    cfg = getattr(snapshot, "config", None) or {}
    ckpt = _configurable(cfg)
    return str(ckpt.get("checkpoint_ns") or "")


def extract_parent_checkpoint_id(snapshot: Any) -> Optional[str]:
    parent = getattr(snapshot, "parent_config", None)
    if parent is None:
        return None
    ckpt = _configurable(parent)
    cid = ckpt.get("checkpoint_id")
    return str(cid) if cid is not None else None


def extract_superstep(snapshot: Any) -> Optional[int]:
    """Real LangGraph metadata step — never invent from list position."""
    meta = getattr(snapshot, "metadata", None) or {}
    if not isinstance(meta, Mapping):
        return None
    if "step" not in meta:
        return None
    step = meta.get("step")
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def extract_next_nodes(snapshot: Any) -> Tuple[str, ...]:
    next_attr = getattr(snapshot, "next", None)
    if not next_attr:
        return ()
    return tuple(str(n) for n in next_attr)


def extract_occurrence_records(snapshot: Any) -> List[OccurrenceRecord]:
    values = getattr(snapshot, "values", None) or {}
    if not isinstance(values, Mapping):
        return []
    raw = values.get("occurrence_records") or []
    out: List[OccurrenceRecord] = []
    for item in raw:
        if isinstance(item, OccurrenceRecord):
            out.append(item)
    return out


def occurrence_label(occurrence_id: str, node_id: str = "", occurrence_index: int = 0) -> str:
    if node_id and occurrence_index:
        return f"{node_id}#{occurrence_index}"
    if "#" in occurrence_id:
        # run:branch:node#n → node#n (node may contain no '#')
        _, _, tail = occurrence_id.rpartition(":")
        return tail if "#" in tail else occurrence_id
    return occurrence_id


def _record_contains_occurrence(
    record: CheckpointRecord,
    *,
    node_id: str,
    occurrence_index: int,
    occurrence_id: Optional[str] = None,
) -> bool:
    label = f"{node_id}#{occurrence_index}"
    if label in record.after_occurrence_labels:
        return True
    if label in record.after_occurrences:
        return True
    if occurrence_id and occurrence_id in record.after_occurrences:
        return True
    suffix = f"{node_id}#{occurrence_index}"
    for oid in record.after_occurrences:
        if oid == suffix or oid.endswith(f":{suffix}"):
            return True
    return False


@dataclass
class CheckpointIndex:
    """Maps readable aliases and occurrence boundaries to real checkpoints.

    One index belongs to exactly one factual ``thread_id`` / ``run_id``.
    """

    thread_id: str
    records: List[CheckpointRecord] = field(default_factory=list)
    resolved_occurrences: Tuple[OccurrenceRecord, ...] = ()
    # occurrence_id -> CheckpointRecord alias
    _after_alias_by_occurrence: Dict[str, str] = field(default_factory=dict, repr=False)
    _before_alias_by_occurrence: Dict[str, str] = field(default_factory=dict, repr=False)
    _occurrence_by_id: Dict[str, OccurrenceRecord] = field(default_factory=dict, repr=False)
    _occurrence_by_label: Dict[str, OccurrenceRecord] = field(default_factory=dict, repr=False)
    _linear_parent_fallback_ok: bool = field(default=False, repr=False)

    def add(self, record: CheckpointRecord) -> None:
        if record.thread_id != self.thread_id:
            raise ValueError(
                f"Checkpoint thread_id {record.thread_id!r} does not match "
                f"index thread_id {self.thread_id!r}"
            )
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()
        self.resolved_occurrences = ()
        self._after_alias_by_occurrence.clear()
        self._before_alias_by_occurrence.clear()
        self._occurrence_by_id.clear()
        self._occurrence_by_label.clear()
        self._linear_parent_fallback_ok = False

    def build_from_history(
        self,
        history: Sequence[Any],
        *,
        occurrence_labels: Optional[Sequence[str]] = None,
        allow_synthetic: bool = False,
        source: str = "langgraph",
    ) -> None:
        """Populate aliases ``c_0``, ``c_1``, … from chronological LangGraph history.

        ``history`` must be oldest-first. Occurrence membership is derived from
        cumulative ``values["occurrence_records"]`` deltas between consecutive
        checkpoints. ``occurrence_labels`` is ignored for AXIS-ready builds
        (kept only for signature compatibility; deltas take precedence).
        """
        del occurrence_labels  # deprecated positional wiring — deltas are authoritative
        self.clear()
        if not history:
            return

        chronological = list(history)
        self._validate_thread_and_synthetic(chronological, allow_synthetic=allow_synthetic)
        self._linear_parent_fallback_ok = self._validate_parent_chain(chronological)

        prev_ids: List[str] = []
        prev_id_set = set()
        by_checkpoint_id: Dict[str, CheckpointRecord] = {}

        for i, snap in enumerate(chronological):
            cfg = getattr(snap, "config", None)
            raw_config = dict(cfg) if isinstance(cfg, dict) else None
            if raw_config is None:
                # Ensure enough info for later retrieval
                raw_config = {
                    "configurable": {
                        "thread_id": extract_thread_id(snap, self.thread_id),
                        "checkpoint_id": extract_checkpoint_id(snap),
                        "checkpoint_ns": extract_checkpoint_ns(snap),
                    }
                }
            else:
                # Guarantee configurable keys are present
                conf = dict(_configurable(raw_config))
                conf.setdefault("thread_id", extract_thread_id(snap, self.thread_id))
                conf.setdefault("checkpoint_id", extract_checkpoint_id(snap))
                conf.setdefault("checkpoint_ns", extract_checkpoint_ns(snap))
                raw_config = {**raw_config, "configurable": conf}

            checkpoint_id = extract_checkpoint_id(snap)
            thread_id = extract_thread_id(snap, self.thread_id)
            if thread_id != self.thread_id:
                raise ValueError(
                    f"Snapshot thread_id {thread_id!r} != index thread_id {self.thread_id!r}"
                )

            current_records = extract_occurrence_records(snap)
            current_ids = [o.occurrence_id for o in current_records]
            self._validate_monotonic(prev_ids, current_ids, alias=f"c_{i}")

            new_records = [
                occ
                for occ in current_records
                if occ.occurrence_id not in prev_id_set
            ]
            new_records.sort(key=lambda o: (o.execution_index, o.occurrence_index, o.occurrence_id))
            after_ids = tuple(o.occurrence_id for o in new_records)
            after_labels = tuple(o.labeled_id for o in new_records)

            parent_id = extract_parent_checkpoint_id(snap)
            record = CheckpointRecord(
                alias=f"c_{i}",
                thread_id=self.thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=extract_checkpoint_ns(snap),
                superstep=extract_superstep(snap),
                next_nodes=extract_next_nodes(snap),
                after_occurrences=after_ids,
                after_occurrence_labels=after_labels,
                raw_config=raw_config,
                source=source if not str(checkpoint_id).startswith("synth_") else "synthetic",
                parent_checkpoint_id=parent_id,
            )
            self.add(record)
            by_checkpoint_id[checkpoint_id] = record

            for occ in new_records:
                self._after_alias_by_occurrence[occ.occurrence_id] = record.alias
                before_alias = self._resolve_before_alias(record, by_checkpoint_id, i)
                if before_alias is not None:
                    self._before_alias_by_occurrence[occ.occurrence_id] = before_alias

            prev_ids = current_ids
            prev_id_set = set(current_ids)

        # Enrich from the latest cumulative occurrence list when present
        if chronological:
            latest = extract_occurrence_records(chronological[-1])
            if latest:
                self.resolved_occurrences = self._enrich_occurrence_list(latest)

    def enrich_from_occurrences(
        self,
        occurrences: Sequence[OccurrenceRecord],
    ) -> Tuple[OccurrenceRecord, ...]:
        """Derive checkpoint_before/after/superstep without mutating history."""
        enriched = self._enrich_occurrence_list(occurrences)
        self.resolved_occurrences = enriched
        return enriched

    def _enrich_occurrence_list(
        self,
        occurrences: Sequence[OccurrenceRecord],
    ) -> Tuple[OccurrenceRecord, ...]:
        enriched: List[OccurrenceRecord] = []
        by_alias = {r.alias: r for r in self.records}
        for occ in occurrences:
            after_alias = self._after_alias_by_occurrence.get(occ.occurrence_id)
            before_alias = self._before_alias_by_occurrence.get(occ.occurrence_id)
            superstep = None
            if after_alias and after_alias in by_alias:
                superstep = by_alias[after_alias].superstep
            new_occ = replace(
                occ,
                checkpoint_before=before_alias,
                checkpoint_after=after_alias,
                superstep=superstep,
            )
            enriched.append(new_occ)
            self._occurrence_by_id[new_occ.occurrence_id] = new_occ
            self._occurrence_by_label[new_occ.labeled_id] = new_occ
        return tuple(enriched)

    def _resolve_before_alias(
        self,
        record: CheckpointRecord,
        by_checkpoint_id: Mapping[str, CheckpointRecord],
        index: int,
    ) -> Optional[str]:
        if record.parent_checkpoint_id:
            parent = by_checkpoint_id.get(record.parent_checkpoint_id)
            if parent is not None:
                return parent.alias
            raise ValueError(
                f"Parent checkpoint_id {record.parent_checkpoint_id!r} for "
                f"{record.alias} not found in factual index"
            )
        # parent_config unavailable: linear chronological predecessor only
        if index > 0:
            return self.records[index - 1].alias
        return None

    def _validate_thread_and_synthetic(
        self,
        chronological: Sequence[Any],
        *,
        allow_synthetic: bool,
    ) -> None:
        for snap in chronological:
            cid = extract_checkpoint_id(snap)
            if str(cid).startswith("synth_") and not allow_synthetic:
                raise ValueError(
                    f"Synthetic checkpoint {cid!r} rejected for AXIS-ready factual index. "
                    "Pass allow_synthetic=True only for explicit unit-test fixtures."
                )
            tid = extract_thread_id(snap, self.thread_id)
            if tid and tid != self.thread_id:
                raise ValueError(
                    f"Mixed thread_id in history: {tid!r} vs index {self.thread_id!r}"
                )

    def _validate_parent_chain(self, chronological: Sequence[Any]) -> bool:
        """Validate parent_config forms one factual chain. Return whether linear fallback is OK."""
        if not chronological:
            return True
        ids = [extract_checkpoint_id(s) for s in chronological]
        id_set = set(ids)
        if len(ids) != len(id_set):
            raise ValueError("Duplicate checkpoint_id in factual history")
        has_any_parent = False
        for i, snap in enumerate(chronological):
            parent_id = extract_parent_checkpoint_id(snap)
            if parent_id is None:
                continue
            has_any_parent = True
            if parent_id not in id_set:
                # Parent outside this filtered history — not a hard failure for truncated views
                continue
            parent_pos = ids.index(parent_id)
            if parent_pos >= i:
                raise ValueError(
                    f"Parent checkpoint {parent_id} is not earlier than child {ids[i]}"
                )
            if parent_pos != i - 1:
                raise ValueError(
                    f"Factual parent chain break at {ids[i]}: parent {parent_id} "
                    f"is not the immediate chronological predecessor {ids[i - 1]}"
                )
        # Linear fallback allowed when no parents contradict order (including no parents at all)
        return True

    @staticmethod
    def _validate_monotonic(
        prev_ids: Sequence[str],
        current_ids: Sequence[str],
        *,
        alias: str,
    ) -> None:
        prev_set = set(prev_ids)
        current_set = set(current_ids)
        missing = prev_set - current_set
        if missing:
            raise ValueError(
                f"Occurrence history not monotonic at {alias}: "
                f"missing {sorted(missing)}"
            )
        # Prefer prefix order when both are ordered lists
        if len(prev_ids) <= len(current_ids):
            if list(current_ids[: len(prev_ids)]) != list(prev_ids):
                # Allow set-superset if ordering differs but all previous present
                if not prev_set.issubset(current_set):
                    raise ValueError(
                        f"Occurrence history not a subset at {alias}"
                    )

    def resolve(self, selector: BoundarySelector) -> CheckpointRecord:
        """Resolve a boundary selector to exactly one checkpoint.

        Raises:
            KeyError: boundary does not exist.
            ValueError: selector is ambiguous.
        """
        matches: List[CheckpointRecord] = []
        if selector.kind == "alias":
            matches = [r for r in self.records if r.alias == selector.value]
        elif selector.kind == "after_occurrence":
            if selector.occurrence_index is None:
                raise KeyError(f"Boundary not found: {selector}")
            occ = self._lookup_occurrence(selector.value, selector.occurrence_index)
            after_alias = self._after_alias_by_occurrence.get(occ.occurrence_id)
            if after_alias:
                matches = [r for r in self.records if r.alias == after_alias]
            else:
                matches = [
                    r
                    for r in self.records
                    if _record_contains_occurrence(
                        r,
                        node_id=selector.value,
                        occurrence_index=selector.occurrence_index,
                        occurrence_id=occ.occurrence_id,
                    )
                ]
        elif selector.kind == "before_occurrence":
            if selector.occurrence_index is None:
                raise KeyError(f"Boundary not found: {selector}")
            occ = self._lookup_occurrence(selector.value, selector.occurrence_index)
            before_alias = self._before_alias_by_occurrence.get(occ.occurrence_id)
            if before_alias:
                matches = [r for r in self.records if r.alias == before_alias]
            else:
                # Derive from after record's parent without positional occurrence guessing
                after_matches = [
                    r
                    for r in self.records
                    if _record_contains_occurrence(
                        r,
                        node_id=selector.value,
                        occurrence_index=selector.occurrence_index,
                        occurrence_id=occ.occurrence_id,
                    )
                ]
                if len(after_matches) != 1:
                    raise KeyError(f"Boundary not found: {selector}")
                after = after_matches[0]
                parent = self._parent_record(after)
                if parent is None:
                    raise KeyError(
                        f"No before-checkpoint for occurrence "
                        f"{selector.value}#{selector.occurrence_index}"
                    )
                matches = [parent]
        elif selector.kind == "before_next_node":
            matches = [r for r in self.records if selector.value in r.next_nodes]
        else:
            raise ValueError(f"Unknown boundary selector kind: {selector.kind}")

        if not matches:
            raise KeyError(f"Boundary not found: {selector}")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous boundary {selector}: {[m.alias for m in matches]}"
            )
        return matches[0]

    def _parent_record(self, record: CheckpointRecord) -> Optional[CheckpointRecord]:
        if record.parent_checkpoint_id:
            for r in self.records:
                if r.checkpoint_id == record.parent_checkpoint_id:
                    return r
            return None
        # Linear fallback by alias index only when parent_config absent
        try:
            idx = int(record.alias.split("_", 1)[1])
        except (IndexError, ValueError):
            return None
        if idx <= 0:
            return None
        prev_alias = f"c_{idx - 1}"
        for r in self.records:
            if r.alias == prev_alias:
                return r
        return None

    def _lookup_occurrence(self, node_id: str, occurrence_index: int) -> OccurrenceRecord:
        label = f"{node_id}#{occurrence_index}"
        if label in self._occurrence_by_label:
            return self._occurrence_by_label[label]
        # Search after_occurrence maps / records
        for occ in self.resolved_occurrences:
            if occ.node_id == node_id and occ.occurrence_index == occurrence_index:
                return occ
        # Build a stub identity from records if enrichment not yet applied
        for r in self.records:
            if _record_contains_occurrence(
                r, node_id=node_id, occurrence_index=occurrence_index
            ):
                # Prefer full id from after_occurrences
                suffix = f"{node_id}#{occurrence_index}"
                for oid in r.after_occurrences:
                    if oid == suffix or oid.endswith(f":{suffix}"):
                        return OccurrenceRecord(
                            occurrence_id=oid,
                            node_id=node_id,
                            occurrence_index=occurrence_index,
                            branch_id="factual",
                            run_id=self.thread_id,
                        )
                return OccurrenceRecord(
                    occurrence_id=label,
                    node_id=node_id,
                    occurrence_index=occurrence_index,
                    branch_id="factual",
                    run_id=self.thread_id,
                )
        raise KeyError(
            f"Occurrence not found: {node_id}#{occurrence_index}"
        )

    def get_alias(self, alias: str) -> CheckpointRecord:
        return self.resolve(BoundarySelector.by_alias(alias))

    def occurrence(
        self,
        node_id: str,
        occurrence_index: int,
    ) -> OccurrenceRecord:
        """Return the exact enriched OccurrenceRecord for ``node_id#occurrence_index``."""
        return self._lookup_occurrence(node_id, occurrence_index)

    def occurrence_by_id(self, occurrence_id: str) -> OccurrenceRecord:
        if occurrence_id in self._occurrence_by_id:
            return self._occurrence_by_id[occurrence_id]
        for occ in self.resolved_occurrences:
            if occ.occurrence_id == occurrence_id:
                return occ
        raise KeyError(f"Occurrence not found: {occurrence_id}")

    def occurrences_by_superstep(self) -> Dict[Optional[int], Tuple[OccurrenceRecord, ...]]:
        """Group enriched occurrences by real LangGraph super-step (Ω_s)."""
        groups: Dict[Optional[int], List[OccurrenceRecord]] = {}
        for occ in self.resolved_occurrences:
            groups.setdefault(occ.superstep, []).append(occ)
        return {k: tuple(v) for k, v in groups.items()}
