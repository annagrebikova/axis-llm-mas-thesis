"""Opt-in diagnostics for hybrid ADD vocabulary construction.

Records eligibility and checkpoint-presence decisions without changing which
ADD interventions are advertised. Unknown checkpoint state is never reported
as confirmed presence or confirmed absence.

Vocabulary is built once per explained question, before interrogation rounds.
``attempt_index`` is therefore empty at construction; the same candidate can
still appear in later rounds only if a future builder rebuilds the vocabulary.
``available_add_count`` counts remaining executable ADD(ref, boundary) pairs
after subtracting executed signatures (see ``hybrid_opportunity_counts``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ADD_PRESENCE_PRESENT = "present"
ADD_PRESENCE_ABSENT = "absent"
ADD_PRESENCE_UNKNOWN = "unknown"

VOCABULARY_CONSTRUCTION_PHASE = "vocabulary"

CANDIDATE_SOURCE_FACTUAL = "factual_registry"
CANDIDATE_SOURCE_INTERVENTION = "intervention_registry"
CANDIDATE_SOURCE_EXPLICIT = "explicit_alternative"

HYBRID_ADD_CANDIDATE_FIELDS = [
    "lecturebank_id",
    "document_id",
    "run_id",
    "explained_ref",
    "attempt_index",
    "construction_phase",
    "artifact_ref",
    "artifact_type",
    "candidate_source",
    "in_relevant_slice",
    "contract_permits_add",
    "payload_sourceable",
    "contextual_eligible",
    "explicit_exception_applied",
    "section_id",
    "context_tokens",
    "eligibility_result",
    "reached_boundary_checks",
    "omitted_by_cap",
    "advertised",
    "advertised_add_boundary_count",
    "executable_add_opportunity_count",
    "final_status",
]

HYBRID_ADD_BOUNDARY_FIELDS = [
    "lecturebank_id",
    "document_id",
    "run_id",
    "explained_ref",
    "attempt_index",
    "construction_phase",
    "artifact_ref",
    "artifact_type",
    "boundary",
    "checkpoint_resolved",
    "checkpoint_alias",
    "checkpoint_id",
    "state_inspectable",
    "presence",
    "presence_detail",
    "inclusion_reason",
    "advertised",
    "omitted_by_cap",
]

_TOKEN_LIMIT = 200


def _bool_cell(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _join_tokens(tokens: Iterable[str]) -> str:
    parts = sorted({str(t).strip() for t in tokens if str(t).strip()})
    text = "|".join(parts)
    if len(text) > _TOKEN_LIMIT:
        return text[: _TOKEN_LIMIT - 1] + "…"
    return text


def _identity_key(row: Mapping[str, Any]) -> Tuple[str, ...]:
    return (
        str(row.get("lecturebank_id") or ""),
        str(row.get("run_id") or ""),
        str(row.get("explained_ref") or ""),
        str(row.get("attempt_index") or ""),
        str(row.get("construction_phase") or VOCABULARY_CONSTRUCTION_PHASE),
        str(row.get("artifact_ref") or ""),
        str(row.get("candidate_source") or ""),
        str(row.get("boundary") or ""),
    )


@dataclass
class HybridAddDiagnostics:
    """Collector for one vocabulary-construction pass (one explained question)."""

    run_id: str = ""
    explained_ref: str = ""
    lecturebank_id: str = ""
    document_id: str = ""
    attempt_index: str = ""
    construction_phase: str = VOCABULARY_CONSTRUCTION_PHASE
    candidate_rows: List[Dict[str, Any]] = field(default_factory=list)
    boundary_rows: List[Dict[str, Any]] = field(default_factory=list)
    _candidate_index: Dict[Tuple[str, str], int] = field(default_factory=dict, repr=False)

    def _base(self) -> Dict[str, Any]:
        return {
            "lecturebank_id": self.lecturebank_id,
            "document_id": self.document_id,
            "run_id": self.run_id,
            "explained_ref": self.explained_ref,
            "attempt_index": self.attempt_index,
            "construction_phase": self.construction_phase,
        }

    def record_candidate(
        self,
        *,
        artifact_ref: Any,
        artifact_type: str,
        candidate_source: str,
        in_relevant_slice: bool,
        contract_permits_add: Optional[bool],
        payload_sourceable: Optional[bool],
        contextual_eligible: Optional[bool],
        explicit_exception_applied: bool,
        section_id: str = "",
        context_tokens: Iterable[str] = (),
        eligibility_result: str,
        reached_boundary_checks: bool = False,
    ) -> None:
        key = (str(artifact_ref), str(candidate_source))
        row = {
            **self._base(),
            "artifact_ref": str(artifact_ref),
            "artifact_type": str(artifact_type or ""),
            "candidate_source": candidate_source,
            "in_relevant_slice": _bool_cell(in_relevant_slice),
            "contract_permits_add": _bool_cell(contract_permits_add),
            "payload_sourceable": _bool_cell(payload_sourceable),
            "contextual_eligible": _bool_cell(contextual_eligible),
            "explicit_exception_applied": _bool_cell(explicit_exception_applied),
            "section_id": str(section_id or ""),
            "context_tokens": _join_tokens(context_tokens),
            "eligibility_result": str(eligibility_result or ""),
            "reached_boundary_checks": _bool_cell(reached_boundary_checks),
            "omitted_by_cap": _bool_cell(False),
            "advertised": _bool_cell(False),
            "advertised_add_boundary_count": 0,
            "executable_add_opportunity_count": 0,
            "final_status": str(eligibility_result or ""),
        }
        self._candidate_index[key] = len(self.candidate_rows)
        self.candidate_rows.append(row)

    def record_boundary(
        self,
        *,
        artifact_ref: Any,
        artifact_type: str,
        boundary: str,
        checkpoint_resolved: bool,
        checkpoint_alias: str = "",
        checkpoint_id: str = "",
        state_inspectable: bool,
        presence: str,
        presence_detail: str,
        inclusion_reason: str,
        advertised: bool = False,
        omitted_by_cap: bool = False,
    ) -> None:
        self.boundary_rows.append(
            {
                **self._base(),
                "artifact_ref": str(artifact_ref),
                "artifact_type": str(artifact_type or ""),
                "boundary": str(boundary or ""),
                "checkpoint_resolved": _bool_cell(checkpoint_resolved),
                "checkpoint_alias": str(checkpoint_alias or ""),
                "checkpoint_id": str(checkpoint_id or ""),
                "state_inspectable": _bool_cell(state_inspectable),
                "presence": str(presence or ADD_PRESENCE_UNKNOWN),
                "presence_detail": str(presence_detail or ""),
                "inclusion_reason": str(inclusion_reason or ""),
                "advertised": _bool_cell(advertised),
                "omitted_by_cap": _bool_cell(omitted_by_cap),
            }
        )

    def _candidate_row_for(self, artifact_ref: Any) -> Optional[Dict[str, Any]]:
        want = str(artifact_ref)
        for row in self.candidate_rows:
            if str(row.get("artifact_ref") or "") != want:
                continue
            if str(row.get("eligibility_result") or "") == "reached_boundary_checks":
                return row
        for row in self.candidate_rows:
            if str(row.get("artifact_ref") or "") == want:
                return row
        return None

    def finalize(
        self,
        *,
        advertised_refs: Sequence[Any],
        advertised_boundaries: Mapping[Any, Sequence[str]],
        executable_add_count: int,
        omitted_by_cap_refs: Sequence[Any] = (),
    ) -> None:
        advertised_keys = {str(r) for r in advertised_refs}
        omitted_keys = {str(r) for r in omitted_by_cap_refs}
        bound_map = {str(k): tuple(v) for k, v in (advertised_boundaries or {}).items()}
        for row in self.candidate_rows:
            key = str(row.get("artifact_ref") or "")
            bounds = bound_map.get(key, ())
            advertised = key in advertised_keys
            omitted = key in omitted_keys
            row["advertised"] = _bool_cell(advertised)
            row["omitted_by_cap"] = _bool_cell(omitted)
            row["advertised_add_boundary_count"] = len(bounds) if advertised else 0
            row["executable_add_opportunity_count"] = len(bounds) if advertised else 0
            if advertised:
                row["final_status"] = "advertised"
            elif omitted:
                row["final_status"] = "omitted_by_cap"
                row["reached_boundary_checks"] = _bool_cell(True)
            elif str(row.get("eligibility_result") or "") == "reached_boundary_checks":
                pair_reasons = [
                    str(b.get("inclusion_reason") or "")
                    for b in self.boundary_rows
                    if str(b.get("artifact_ref") or "") == key
                ]
                if any(r == "no_suitable_consumer_boundary" for r in pair_reasons) and not any(
                    r.startswith("advertised") for r in pair_reasons
                ):
                    row["final_status"] = "no_suitable_consumer_boundary"
                elif not pair_reasons:
                    row["final_status"] = "no_suitable_consumer_boundary"
                else:
                    row["final_status"] = "no_confirmed_absent_boundary"
        advertised_bound_keys = {
            (str(ref), str(b))
            for ref, bounds in bound_map.items()
            if str(ref) in advertised_keys
            for b in bounds
        }
        for brow in self.boundary_rows:
            key = (str(brow.get("artifact_ref") or ""), str(brow.get("boundary") or ""))
            is_adv = key in advertised_bound_keys
            brow["advertised"] = _bool_cell(is_adv)
            if str(brow.get("artifact_ref") or "") in omitted_keys and str(
                brow.get("presence") or ""
            ) == ADD_PRESENCE_ABSENT:
                brow["omitted_by_cap"] = _bool_cell(True)
                if not is_adv:
                    brow["inclusion_reason"] = "omitted_by_cap"
        # executable_add_count is the vocabulary-level available_add_count
        # at construction (no executed signatures yet).
        self._executable_add_count = int(executable_add_count)

    def summary(self) -> Dict[str, Any]:
        return summarize_hybrid_add_diagnostics(
            self.candidate_rows,
            self.boundary_rows,
            executable_add_count=int(getattr(self, "_executable_add_count", 0) or 0),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "run_id": self.run_id,
            "explained_ref": self.explained_ref,
            "lecturebank_id": self.lecturebank_id,
            "document_id": self.document_id,
            "attempt_index": self.attempt_index,
            "construction_phase": self.construction_phase,
            "candidate_rows": list(self.candidate_rows),
            "boundary_rows": list(self.boundary_rows),
            "summary": self.summary(),
        }


def summarize_hybrid_add_diagnostics(
    candidate_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    *,
    executable_add_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate candidate vs candidate-boundary counts. Does not mix the two."""
    unique_artifacts = {str(r.get("artifact_ref") or "") for r in candidate_rows if r.get("artifact_ref")}
    first_by_artifact: Dict[str, Mapping[str, Any]] = {}
    for row in candidate_rows:
        key = str(row.get("artifact_ref") or "")
        if key and key not in first_by_artifact:
            first_by_artifact[key] = row
    eligibility_counts = Counter(
        str(r.get("eligibility_result") or "") for r in first_by_artifact.values()
    )
    final_counts = Counter(str(r.get("final_status") or "") for r in first_by_artifact.values())
    reaching = sum(
        1
        for r in first_by_artifact.values()
        if str(r.get("reached_boundary_checks") or "") == "true"
        or str(r.get("eligibility_result") or "") == "reached_boundary_checks"
    )
    omitted = sum(
        1 for r in first_by_artifact.values() if str(r.get("omitted_by_cap") or "") == "true"
    )
    advertised_artifacts = sum(
        1 for r in first_by_artifact.values() if str(r.get("advertised") or "") == "true"
    )
    presence_counts = Counter(str(r.get("presence") or "") for r in boundary_rows)
    advertised_pairs = sum(1 for r in boundary_rows if str(r.get("advertised") or "") == "true")
    if executable_add_count is None:
        executable_add_count = advertised_pairs
    return {
        "unique_candidate_artifacts": len(unique_artifacts),
        "candidate_consideration_rows": len(candidate_rows),
        "eligibility_exclusions": dict(eligibility_counts),
        "final_status_counts": dict(final_counts),
        "candidates_reaching_boundary_checks": reaching,
        "candidates_omitted_by_cap": omitted,
        "advertised_add_artifacts": advertised_artifacts,
        "candidate_boundary_pairs": len(boundary_rows),
        "pairs_confirmed_present": int(presence_counts.get(ADD_PRESENCE_PRESENT, 0)),
        "pairs_confirmed_absent": int(presence_counts.get(ADD_PRESENCE_ABSENT, 0)),
        "pairs_unknown_state": int(presence_counts.get(ADD_PRESENCE_UNKNOWN, 0)),
        "advertised_add_artifact_boundary_pairs": advertised_pairs,
        "executable_add_opportunities": int(executable_add_count),
        "available_add_count_definition": (
            "executable ADD(ref, boundary) pairs remaining in the advertised "
            "vocabulary after subtracting executed signatures "
            "(hybrid_opportunity_counts['add']). Construction-time value has "
            "no executed signatures yet."
        ),
    }


def extract_hybrid_add_diagnostics(
    experiment: Mapping[str, Any],
    *,
    lecturebank_id: str = "",
) -> Dict[str, Any]:
    """Collect diagnostic rows from one experiment payload (single copy)."""
    candidate_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    seen_c: set[Tuple[str, ...]] = set()
    seen_b: set[Tuple[str, ...]] = set()
    lid = str(lecturebank_id or experiment.get("lecturebank_id") or "")
    doc_id = str(experiment.get("document_id") or "")
    source = experiment.get("hybrid_add_source") if isinstance(experiment, Mapping) else None
    if isinstance(source, Mapping):
        doc_id = str(source.get("document_id") or doc_id)
        if not lid:
            lid = str(source.get("lecturebank_id") or "")
    if not doc_id and lid:
        doc_id = f"lecturebank_{lid}"
    run_id = str(experiment.get("run_id") or "")
    for case in list(experiment.get("cases") or []):
        conds = case.get("conditions") if isinstance(case.get("conditions"), Mapping) else {}
        axis = conds.get("axis") if isinstance(conds, Mapping) else {}
        if not isinstance(axis, Mapping):
            axis = case.get("axis") if isinstance(case.get("axis"), Mapping) else {}
        raw = axis.get("raw_evidence") if isinstance(axis, Mapping) else {}
        if not isinstance(raw, Mapping):
            raw = {}
        payload = raw.get("hybrid_add_diagnostics")
        if not isinstance(payload, Mapping) or not payload.get("enabled"):
            continue
        explained = str(
            payload.get("explained_ref")
            or case.get("candidate_ref")
            or ""
        )
        case_run = str(payload.get("run_id") or case.get("run_id") or run_id)
        for row in list(payload.get("candidate_rows") or []):
            if not isinstance(row, Mapping):
                continue
            out = dict(row)
            if not str(out.get("lecturebank_id") or "").strip():
                out["lecturebank_id"] = lid
            if not str(out.get("document_id") or "").strip():
                out["document_id"] = doc_id
            if not str(out.get("run_id") or "").strip():
                out["run_id"] = case_run
            if not str(out.get("explained_ref") or "").strip():
                out["explained_ref"] = explained
            key = _identity_key(out)
            if key in seen_c:
                continue
            seen_c.add(key)
            candidate_rows.append(out)
        for row in list(payload.get("boundary_rows") or []):
            if not isinstance(row, Mapping):
                continue
            out = dict(row)
            if not str(out.get("lecturebank_id") or "").strip():
                out["lecturebank_id"] = lid
            if not str(out.get("document_id") or "").strip():
                out["document_id"] = doc_id
            if not str(out.get("run_id") or "").strip():
                out["run_id"] = case_run
            if not str(out.get("explained_ref") or "").strip():
                out["explained_ref"] = explained
            key = _identity_key(out)
            if key in seen_b:
                continue
            seen_b.add(key)
            boundary_rows.append(out)
    summaries = [
        summarize_hybrid_add_diagnostics(
            [r for r in candidate_rows if str(r.get("explained_ref") or "") == expl],
            [r for r in boundary_rows if str(r.get("explained_ref") or "") == expl],
        )
        for expl in sorted({str(r.get("explained_ref") or "") for r in candidate_rows})
    ]
    executable = sum(int(s.get("executable_add_opportunities") or 0) for s in summaries)
    overall = summarize_hybrid_add_diagnostics(
        candidate_rows, boundary_rows, executable_add_count=executable
    )
    return {
        "hybrid_add_candidates": candidate_rows,
        "hybrid_add_boundaries": boundary_rows,
        "hybrid_add_summary": overall,
    }


def inspect_saved_experiment_hybrid_add_replay(
    experiment: Mapping[str, Any],
) -> Dict[str, Any]:
    """Report whether a saved experiment can replay the real ADD vocabulary builder.

    Missing checkpoint snapshots or a full artifact registry cannot be inferred
    from ``available_add_count = 0``.
    """
    cases = list(experiment.get("cases") or [])
    has_diagnostics = False
    slice_artifacts = 0
    provenance_rels = 0
    add_counts: List[int] = []
    whatif_counts: List[int] = []
    remove_counts: List[int] = []
    modes: set[str] = set()
    replacement_refs: set[str] = set()
    slice_refs: set[str] = set()
    replacements_outside_own_slice: set[str] = set()
    slice_types: Counter = Counter()
    catalog_hits = 0
    snapshot_hits = 0
    registry_hits = 0

    def _walk(obj: Any, *, path: str = "") -> None:
        nonlocal catalog_hits, snapshot_hits, registry_hits
        if isinstance(obj, Mapping):
            keys = {str(k) for k in obj.keys()}
            if "artifact_catalog" in keys:
                catalog_hits += 1
            if "checkpoint_snapshot" in keys or "checkpoint_snapshots" in keys:
                snapshot_hits += 1
            if "artifact_registry" in keys or "factual_registry" in keys:
                registry_hits += 1
            if path.count(".") < 6:
                for k, v in obj.items():
                    _walk(v, path=f"{path}.{k}" if path else str(k))
        elif isinstance(obj, list) and path.count(".") < 5:
            for i, item in enumerate(obj[:20]):
                _walk(item, path=f"{path}[{i}]")

    _walk(experiment)
    for case in cases:
        conds = case.get("conditions") if isinstance(case.get("conditions"), Mapping) else {}
        axis = conds.get("axis") if isinstance(conds, Mapping) else {}
        if not isinstance(axis, Mapping):
            continue
        raw = axis.get("raw_evidence") if isinstance(axis.get("raw_evidence"), Mapping) else {}
        if raw.get("hybrid_add_diagnostics"):
            has_diagnostics = True
        arts = list(raw.get("relevant_artifacts") or [])
        slice_artifacts += len(arts)
        case_slice_refs: set[str] = set()
        for art in arts:
            if isinstance(art, Mapping):
                ref = str(art.get("ref") or "")
                if ref:
                    slice_refs.add(ref)
                    case_slice_refs.add(ref)
                slice_types[str(art.get("artifact_type") or "")] += 1
        provenance_rels += len(list(raw.get("provenance_relations") or []))
        modes.add(str(raw.get("axis_interrogation_mode") or experiment.get("axis_interrogation_mode") or ""))
        for att in list(raw.get("interrogation_attempts") or []):
            if not isinstance(att, Mapping):
                continue
            modes.add(str(att.get("axis_interrogation_mode") or ""))
            try:
                add_counts.append(int(att.get("available_add_count") or 0))
            except (TypeError, ValueError):
                add_counts.append(0)
            try:
                whatif_counts.append(int(att.get("available_whatif_count") or 0))
            except (TypeError, ValueError):
                whatif_counts.append(0)
            try:
                remove_counts.append(int(att.get("available_remove_count") or 0))
            except (TypeError, ValueError):
                remove_counts.append(0)
            query = str(att.get("generated_query") or "")
            if query.upper().startswith("WHATIF("):
                inner = query[len("WHATIF(") :].rstrip(")")
                parts = [p.strip() for p in inner.split(",")]
                if len(parts) >= 2:
                    replacement_refs.add(parts[1])
                    if parts[1] and parts[1] not in case_slice_refs:
                        replacements_outside_own_slice.add(parts[1])

    missing: List[str] = []
    if registry_hits == 0:
        missing.append("full_factual_artifact_registry")
    if catalog_hits == 0:
        missing.append("checkpoint_artifact_catalog")
    if snapshot_hits == 0 and catalog_hits == 0:
        missing.append("checkpoint_snapshots")
    missing.append("replay_executor_or_live_checkpoint_getter")
    can_replay = registry_hits > 0 and catalog_hits > 0
    observed = {
        "case_count": len(cases),
        "axis_interrogation_modes": sorted(m for m in modes if m),
        "available_add_count_min": min(add_counts) if add_counts else None,
        "available_add_count_max": max(add_counts) if add_counts else None,
        "available_add_count_all_zero": bool(add_counts) and all(c == 0 for c in add_counts),
        "available_remove_count_max": max(remove_counts) if remove_counts else None,
        "available_whatif_count_max": max(whatif_counts) if whatif_counts else None,
        "slice_artifact_rows": slice_artifacts,
        "slice_evidencechunk_rows": int(slice_types.get("EvidenceChunk", 0)),
        "slice_provenance_relations": provenance_rels,
        "whatif_replacement_refs": sorted(replacement_refs),
        "whatif_replacements_outside_explained_slice": sorted(
            replacements_outside_own_slice
        ),
        "hybrid_add_diagnostics_present": has_diagnostics,
        "artifact_catalog_keys_found": catalog_hits,
        "checkpoint_snapshot_keys_found": snapshot_hits,
        "registry_keys_found": registry_hits,
    }
    return {
        "can_replay_real_vocabulary_builder": can_replay,
        "missing_for_replay": missing if not can_replay else [],
        "observed": observed,
        "exclusion_stage": None if not can_replay else "replay_required",
        "note": (
            "Do not infer the ADD exclusion stage from available_add_count=0. "
            "The real builder needs the full live artifact registry plus "
            "checkpoint snapshots containing artifact_catalog."
        ),
    }


def format_hybrid_add_report_section(summary: Mapping[str, Any]) -> List[str]:
    """Markdown lines for report.md when diagnostics ran."""
    elig = summary.get("eligibility_exclusions") or {}
    finals = summary.get("final_status_counts") or {}
    lines = [
        "## Hybrid ADD vocabulary diagnostics",
        "",
        "Opt-in trace of hybrid ADD filtering (`--diagnose-hybrid-add`). "
        "Does not change advertised interventions. "
        "`available_add_count` is the number of executable ADD(ref, boundary) "
        "pairs remaining after subtracting executed signatures, not the number "
        "of unique ADD artifacts.",
        "",
        "### Candidate artifacts (unique refs, first consideration)",
        "",
        f"- Unique candidate artifacts considered: {summary.get('unique_candidate_artifacts', 0)}",
        f"- Consideration rows (source × artifact, including duplicates): {summary.get('candidate_consideration_rows', 0)}",
        f"- Reaching boundary checks: {summary.get('candidates_reaching_boundary_checks', 0)}",
        f"- Omitted by vocabulary cap: {summary.get('candidates_omitted_by_cap', 0)}",
        f"- Final advertised ADD artifacts: {summary.get('advertised_add_artifacts', 0)}",
        "",
        "Eligibility result (first exclusion, or reached_boundary_checks):",
        "",
    ]
    if elig:
        for reason, count in sorted(elig.items(), key=lambda kv: (-int(kv[1]), kv[0])):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "Final status:",
            "",
        ]
    )
    if finals:
        for reason, count in sorted(finals.items(), key=lambda kv: (-int(kv[1]), kv[0])):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "### Candidate–boundary pairs",
            "",
            f"- Pairs inspected: {summary.get('candidate_boundary_pairs', 0)}",
            f"- Confirmed present: {summary.get('pairs_confirmed_present', 0)}",
            f"- Confirmed absent: {summary.get('pairs_confirmed_absent', 0)}",
            f"- Unknown state (not treated as present or absent): {summary.get('pairs_unknown_state', 0)}",
            f"- Advertised ADD artifact–boundary pairs: {summary.get('advertised_add_artifact_boundary_pairs', 0)}",
            f"- Executable ADD opportunities (construction-time `available_add_count`): {summary.get('executable_add_opportunities', 0)}",
            "",
        ]
    )
    return lines
