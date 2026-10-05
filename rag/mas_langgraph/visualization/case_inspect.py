"""Presentation helpers for exported explanation-experiment JSON.

These helpers only read already-exported records. They do not call models,
run AXIS, replay workflows, or infer new concept-change labels.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .labels import PREVIEW_MAX_CHARS, artifact_preview, make_preview
from .models import ExplanationGraph
from .ui_helpers import _ref_key, node_content

USER_QUESTION_UNAVAILABLE = "User question not available."
ARTIFACT_CONTENT_UNAVAILABLE = "Content not available in this export."
CONCEPT_CHANGE_UNASSESSED = "Concept change not assessed."
CONCEPT_CHANGE_NOTE = (
    "Concept change is assessed by an LLM by comparing the knowledge tested "
    "by the NOOP and counterfactual questions."
)

CONCEPT_CHANGE_LABELS = {
    "same_concept": "Same tested concept",
    "changed_concept": "Changed tested concept",
    "uncertain": "Uncertain",
    "not_evaluable": "Not evaluable",
    "evaluation_error": "Evaluation error",
}

CONCEPT_CHANGE_REASON_LABELS = {
    "question_not_reproduced": "The question was not reproduced.",
}

_CF_OPERATIONS = frozenset({"REMOVE", "ADD", "WHATIF"})
_LONG_CONTENT_CHARS = 420


def case_user_question(
    case: Optional[Mapping[str, Any]],
    payload: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the AXIS explanation request for the selected case."""
    for source in (case, payload):
        if not isinstance(source, Mapping):
            continue
        value = source.get("user_question")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def display_user_question(
    case: Optional[Mapping[str, Any]],
    payload: Optional[Mapping[str, Any]] = None,
) -> str:
    return case_user_question(case, payload) or USER_QUESTION_UNAVAILABLE


def _canonical_operation(operation: Any) -> str:
    token = str(operation or "").strip().upper().replace("-", "_").replace(" ", "")
    if token in {"WHATIF", "WHAT_IF", "REPLACE"}:
        return "WHATIF"
    return token


def _canonical_signature(
    *,
    operation: Any,
    target_ref: Any = None,
    replacement_ref: Any = None,
    checkpoint_id: Any = None,
) -> str:
    op = _canonical_operation(operation)
    target = _ref_key(target_ref) or ""
    replacement = _ref_key(replacement_ref) or ""
    checkpoint = str(checkpoint_id or "")
    return f"{op}|{target}|{replacement}|{checkpoint}"


def _signature_usable(sig: str) -> bool:
    parts = str(sig or "").split("|")
    if len(parts) != 4:
        return False
    op, target, _replacement, checkpoint = parts
    return bool(op) and bool(target) and bool(checkpoint)


def intervention_signature_key(payload: Optional[Mapping[str, Any]]) -> str:
    """Canonical operation|target|replacement|checkpoint identity."""
    if not isinstance(payload, Mapping):
        return ""
    if isinstance(payload.get("intervention_signature"), Mapping):
        src: Mapping[str, Any] = payload.get("intervention_signature")  # type: ignore[assignment]
    elif isinstance(payload.get("signature"), Mapping):
        src = payload.get("signature")  # type: ignore[assignment]
    elif isinstance(payload.get("resolved"), Mapping):
        src = payload.get("resolved")  # type: ignore[assignment]
    else:
        src = payload
    sig = _canonical_signature(
        operation=src.get("operation")
        or payload.get("operation")
        or payload.get("parsed_operation"),
        target_ref=src.get("target_ref") or payload.get("target_ref"),
        replacement_ref=(
            src.get("replacement_ref")
            if "replacement_ref" in src
            else payload.get("replacement_ref")
        ),
        checkpoint_id=(
            src.get("checkpoint_id")
            or src.get("checkpoint_alias")
            or payload.get("checkpoint_id")
            or payload.get("checkpoint_alias")
        ),
    )
    if _signature_usable(sig):
        return sig
    nested = payload.get("replay_comparison")
    if isinstance(nested, Mapping) and nested is not payload:
        return intervention_signature_key(nested)
    return ""


def _axis_raw(case: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(case, Mapping):
        return {}
    conds = case.get("conditions") if isinstance(case.get("conditions"), Mapping) else {}
    axis = conds.get("axis") if isinstance(conds, Mapping) else {}
    if not isinstance(axis, Mapping):
        return {}
    raw = axis.get("raw_evidence")
    if isinstance(raw, Mapping) and raw:
        return dict(raw)
    summary = axis.get("evidence_summary")
    if isinstance(summary, Mapping) and summary:
        return dict(summary)
    return {}


def _axis_report(raw: Mapping[str, Any]) -> Dict[str, Any]:
    report = raw.get("axis_report")
    return dict(report) if isinstance(report, Mapping) else {}


def _iter_mapping_rows(*groups: Any) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    seen: set[int] = set()
    for group in groups:
        for item in group or []:
            if isinstance(item, Mapping) and id(item) not in seen:
                seen.add(id(item))
                out.append(item)
    return out


def _payload_interrogation_attempts(raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    top = _iter_mapping_rows(raw.get("interrogation_attempts"))
    if top:
        return top
    return _iter_mapping_rows(_axis_report(raw).get("interrogation_attempts"))


def _iter_query_rounds(raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    report = _axis_report(raw)
    return _iter_mapping_rows(raw.get("query_rounds"), report.get("query_rounds"))


def _primary_comparisons(raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    top = _iter_mapping_rows(raw.get("replay_comparisons"))
    if top:
        return top
    return _iter_mapping_rows(_axis_report(raw).get("replay_comparisons"))


def _all_comparison_copies(raw: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    report = _axis_report(raw)
    copies = _iter_mapping_rows(
        raw.get("replay_comparisons"),
        report.get("replay_comparisons"),
    )
    nested = [qr.get("replay_comparison") for qr in _iter_query_rounds(raw)]
    return _iter_mapping_rows(copies, nested)


def _query_round_operation(qr: Mapping[str, Any]) -> str:
    sig = qr.get("signature") if isinstance(qr.get("signature"), Mapping) else {}
    resolved = qr.get("resolved") if isinstance(qr.get("resolved"), Mapping) else {}
    return _canonical_operation(
        sig.get("operation")
        or resolved.get("operation")
        or qr.get("operation")
        or qr.get("parsed_operation")
    )


def query_round_universe(raw: Mapping[str, Any]) -> Dict[str, str]:
    """Map generated_query -> canonical signature from flat or nested rounds."""
    universe: Dict[str, str] = {}
    for qr in _iter_query_rounds(raw):
        if _query_round_operation(qr) not in _CF_OPERATIONS:
            continue
        query = str(qr.get("generated_query") or "")
        if not query:
            continue
        sig = intervention_signature_key(qr)
        if not sig:
            continue
        prior = universe.get(query)
        if prior and prior != sig:
            continue
        universe[query] = sig
    return universe


def _comparisons_by_signature(raw: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    by_sig: Dict[str, Mapping[str, Any]] = {}
    for cmp in _all_comparison_copies(raw):
        sig = intervention_signature_key(cmp)
        if not sig:
            continue
        current = by_sig.get(sig)
        if current is None or (
            not _has_concept_change_fields(current) and _has_concept_change_fields(cmp)
        ):
            by_sig[sig] = cmp
    return by_sig


def _has_concept_change_fields(payload: Mapping[str, Any]) -> bool:
    if isinstance(payload.get("concept_change_evaluation"), Mapping):
        nested = payload.get("concept_change_evaluation")
        if nested.get("label") not in (None, ""):
            return True
    return payload.get("concept_change_label") not in (None, "")


def extract_concept_change_assessment(
    payload: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Read flattened or nested stored concept-change fields. Never infers labels."""
    if not isinstance(payload, Mapping):
        return None
    nested = payload.get("concept_change_evaluation")
    nested = nested if isinstance(nested, Mapping) else {}
    label = nested.get("label")
    if label in (None, ""):
        label = payload.get("concept_change_label")
    if label in (None, ""):
        return None
    reason = nested.get("reason")
    if reason in (None, ""):
        reason = payload.get("concept_change_reason")
    eligibility = nested.get("eligibility_reason")
    if eligibility in (None, ""):
        eligibility = payload.get("concept_change_eligibility_reason")
    return {
        "label": str(label),
        "label_display": CONCEPT_CHANGE_LABELS.get(str(label), str(label)),
        "reason": reason,
        "reason_display": format_concept_change_reason(reason, eligibility),
        "eligibility_reason": eligibility,
        "noop_tested_concept": nested.get("noop_tested_concept")
        or payload.get("concept_change_noop_tested_concept"),
        "counterfactual_tested_concept": nested.get("counterfactual_tested_concept")
        or payload.get("concept_change_counterfactual_tested_concept"),
        "evaluated": nested.get("evaluated"),
    }


def format_concept_change_reason(reason: Any, eligibility_reason: Any = None) -> str:
    token = reason if reason not in (None, "") else eligibility_reason
    if token in (None, ""):
        return ""
    text = str(token).strip()
    return CONCEPT_CHANGE_REASON_LABELS.get(text, text)


def _attempt_is_executed_counterfactual(attempt: Mapping[str, Any]) -> bool:
    op = _canonical_operation(
        attempt.get("parsed_operation") or attempt.get("operation")
    )
    if op not in _CF_OPERATIONS:
        return False
    if attempt.get("executed") is False:
        return False
    status = str(attempt.get("validation_status") or "").strip().lower()
    if status in {"done", "rejected", "invalid", "duplicate", "error", "parser_error"}:
        return False
    if attempt.get("executed") is True or status in {"executed", "ok", "success"}:
        return True
    return False


def match_concept_change_for_signature(
    case: Optional[Mapping[str, Any]],
    signature: str,
) -> Optional[Dict[str, Any]]:
    if not signature:
        return None
    raw = _axis_raw(case)
    found = _comparisons_by_signature(raw).get(signature)
    return extract_concept_change_assessment(found)


def match_concept_change_for_query(
    case: Optional[Mapping[str, Any]],
    generated_query: str,
) -> Optional[Dict[str, Any]]:
    raw = _axis_raw(case)
    universe = query_round_universe(raw)
    sig = universe.get(str(generated_query or ""))
    if not sig:
        return None
    return match_concept_change_for_signature(case, sig)


def match_concept_change_for_unit(
    case: Optional[Mapping[str, Any]],
    unit: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(unit, Mapping):
        return None
    raw = _axis_raw(case)
    universe = query_round_universe(raw)
    query = str(unit.get("generated_query") or "")
    sig = universe.get(query) if query else ""
    if not sig:
        sig = intervention_signature_key(unit)
    if not sig:
        sig = _canonical_signature(
            operation=unit.get("operation"),
            target_ref=unit.get("target_ref"),
            replacement_ref=unit.get("replacement_ref"),
            checkpoint_id=unit.get("checkpoint_id") or unit.get("checkpoint_alias"),
        )
        if not _signature_usable(sig):
            sig = ""
    if not sig:
        return None
    return match_concept_change_for_signature(case, sig)


def interrogation_attempt_views(case: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    raw = _axis_raw(case)
    universe = query_round_universe(raw)
    by_sig = _comparisons_by_signature(raw)
    rows: List[Dict[str, Any]] = []
    for att in _payload_interrogation_attempts(raw):
        query = str(att.get("generated_query") or "")
        attach = _attempt_is_executed_counterfactual(att)
        assessment = None
        assessed = False
        if attach:
            sig = universe.get(query) or intervention_signature_key(att)
            if sig:
                assessment = extract_concept_change_assessment(by_sig.get(sig))
                assessed = assessment is not None
        rows.append(
            {
                "attempt_index": att.get("attempt_index"),
                "generated_query": query,
                "parsed_operation": att.get("parsed_operation") or att.get("operation"),
                "validation_status": att.get("validation_status"),
                "executed": att.get("executed"),
                "attach_assessment": attach,
                "assessment": assessment,
                "assessed": assessed,
                "concept_change_display": (
                    assessment
                    if assessment is not None
                    else (
                        {"label_display": CONCEPT_CHANGE_UNASSESSED, "reason_display": ""}
                        if attach
                        else None
                    )
                ),
            }
        )
    return rows


def _artifact_records(case: Optional[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if not isinstance(case, Mapping):
        return []
    viz = case.get("visualization_data")
    if not isinstance(viz, Mapping):
        return []
    rows = []
    for item in viz.get("artifacts") or []:
        if isinstance(item, Mapping):
            rows.append(item)
    return rows


def _preferred_branch(case: Optional[Mapping[str, Any]], branch_id: Optional[str]) -> str:
    if branch_id:
        return str(branch_id)
    if isinstance(case, Mapping) and case.get("branch_id"):
        return str(case.get("branch_id"))
    return "factual"


def resolve_artifact_record(
    case: Optional[Mapping[str, Any]],
    ref: Any,
    *,
    branch_id: Optional[str] = None,
    graph: Optional[ExplanationGraph] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve an exact versioned artifact ref from the selected case export."""
    key = _ref_key(ref)
    if not key:
        return None
    preferred = _preferred_branch(case, branch_id)
    matches: List[Mapping[str, Any]] = []
    for art in _artifact_records(case):
        art_key = _ref_key(art.get("ref")) or _ref_key(art.get("key"))
        if art_key == key:
            matches.append(art)
    branched = [
        art
        for art in matches
        if str(art.get("branch_id") or "") == preferred
    ]
    chosen = None
    if branched:
        chosen = branched[0]
    elif len(matches) == 1:
        chosen = matches[0]
    elif not matches and graph is not None:
        node = graph.node_map().get(f"artifact::{key}")
        if node is None:
            node = graph.node_map().get(f"outcome::{key}")
        if node is not None and (node.source_ref or "") == key:
            return {
                "ref": {"key": key},
                "artifact_type": node.subtype,
                "branch_id": (node.details or {}).get("branch_id") if isinstance(node.details, Mapping) else None,
                "content": dict(node_content(node)),
                "source": "graph_node",
            }
    if chosen is None:
        return None
    content = chosen.get("content") if isinstance(chosen.get("content"), Mapping) else {}
    return {
        "ref": chosen.get("ref") or {"key": key},
        "artifact_type": chosen.get("artifact_type") or chosen.get("type"),
        "branch_id": chosen.get("branch_id"),
        "content": dict(content or {}),
        "source": "visualization_data",
    }


def _chunk_source_fields(content: Mapping[str, Any]) -> List[Tuple[str, str]]:
    meta = content.get("metadata") if isinstance(content.get("metadata"), Mapping) else {}
    pairs: List[Tuple[str, str]] = []
    source = (
        meta.get("source_file")
        or meta.get("source")
        or content.get("source_file")
        or content.get("source")
    )
    page = meta.get("page_number")
    if page is None:
        page = meta.get("page")
    if page is None:
        page = content.get("page")
    if source not in (None, ""):
        pairs.append(("Source", str(source)))
    if page not in (None, ""):
        pairs.append(("Page", str(page)))
    title = meta.get("title")
    if title not in (None, ""):
        pairs.append(("Title", str(title)))
    return pairs


def _citation_lines(citations: Any) -> List[str]:
    lines: List[str] = []
    for item in citations or []:
        if isinstance(item, Mapping):
            bits = []
            for key in ("source", "source_file", "chunk_id", "page"):
                value = item.get(key)
                if value not in (None, ""):
                    label = "page" if key == "page" else key
                    bits.append(f"{label} {value}" if key == "page" else str(value))
            if bits:
                lines.append(" · ".join(bits))
            else:
                lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        elif item not in (None, ""):
            lines.append(str(item))
    return lines


def _option_lines(content: Mapping[str, Any]) -> List[str]:
    correct = str(content.get("correct_option_id") or content.get("correct_option") or "")
    lines: List[str] = []
    for opt in content.get("options") or []:
        if not isinstance(opt, Mapping):
            continue
        oid = str(opt.get("option_id") or "")
        text = str(opt.get("text") or "")
        mark = "  ← correct" if oid and oid == correct else ""
        lines.append(f"{oid}. {text}{mark}".strip())
    return lines


def artifact_content_view(
    record: Optional[Mapping[str, Any]],
    *,
    ref: Any = None,
    role: str = "",
) -> Dict[str, Any]:
    key = _ref_key(ref) or _ref_key((record or {}).get("ref")) or ""
    role_label = str(role or "").strip()
    if record is None:
        return {
            "ref": key,
            "role": role_label,
            "available": False,
            "artifact_type": "",
            "preview": "",
            "body": ARTIFACT_CONTENT_UNAVAILABLE,
            "fields": [],
            "lines": [],
            "raw": None,
            "use_expander": False,
        }
    atype = str(record.get("artifact_type") or "")
    content = record.get("content") if isinstance(record.get("content"), Mapping) else {}
    fields: List[Tuple[str, str]] = []
    lines: List[str] = []
    body = ""
    if atype == "EvidenceChunk":
        body = str(content.get("content") or "")
        fields.extend(_chunk_source_fields(content))
    elif atype == "KnowledgeAtom":
        body = str(content.get("content") or "")
        lines.extend(_citation_lines(content.get("citations")))
        for key_name in ("atom_type", "section_id", "lo_id"):
            if content.get(key_name) not in (None, ""):
                fields.append((key_name, str(content.get(key_name))))
    elif atype == "MCQBlueprint":
        body = str(content.get("generation_goal") or "")
        for key_name in (
            "question_type",
            "difficulty",
            "bloom_level",
            "section_id",
            "lo_id",
            "target_atom_ids",
            "evidence_chunk_ids",
            "status",
        ):
            value = content.get(key_name)
            if value not in (None, "", []):
                if isinstance(value, (list, tuple)):
                    fields.append((key_name, ", ".join(str(v) for v in value)))
                else:
                    fields.append((key_name, str(value)))
    elif atype == "CandidateMCQ":
        body = str(content.get("stem") or "")
        lines.extend(_option_lines(content))
        correct = content.get("correct_option_id") or content.get("correct_option")
        if correct not in (None, ""):
            fields.append(("Correct answer", str(correct)))
        explanation = content.get("answer_explanation")
        if explanation not in (None, ""):
            fields.append(("Explanation", str(explanation)))
    else:
        body = str(
            content.get("content")
            or content.get("objective_text")
            or content.get("title")
            or content.get("decision")
            or ""
        )
        for key_name, value in content.items():
            if key_name in {"content", "objective_text", "title", "decision"}:
                continue
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (list, dict)):
                fields.append((str(key_name), json.dumps(value, ensure_ascii=False, sort_keys=True)))
            else:
                fields.append((str(key_name), str(value)))
    preview = artifact_preview(atype, content) or make_preview(body, PREVIEW_MAX_CHARS)
    use_expander = len(body) > _LONG_CONTENT_CHARS
    return {
        "ref": key,
        "role": role_label,
        "available": True,
        "artifact_type": atype,
        "preview": preview,
        "body": body,
        "fields": fields,
        "lines": lines,
        "raw": dict(content),
        "use_expander": use_expander,
    }


def _unit_operation(unit: Mapping[str, Any]) -> str:
    sig = unit.get("intervention_signature")
    sig = sig if isinstance(sig, Mapping) else {}
    return _canonical_operation(unit.get("operation") or sig.get("operation"))


def _unit_ref(unit: Mapping[str, Any], field: str) -> str:
    key = _ref_key(unit.get(field))
    if key:
        return key
    sig = unit.get("intervention_signature")
    if isinstance(sig, Mapping):
        return _ref_key(sig.get(field)) or ""
    return ""


def replay_artifact_views(
    case: Optional[Mapping[str, Any]],
    unit: Optional[Mapping[str, Any]],
    *,
    graph: Optional[ExplanationGraph] = None,
) -> List[Dict[str, Any]]:
    """Artifact contents for one replay.

    WHATIF/REPLACE always yield independent original and replacement views.
    REMOVE yields the target only. Missing records keep their exact ref.
    """
    if not isinstance(unit, Mapping):
        return []
    preferred = _preferred_branch(case, None)
    target_key = _unit_ref(unit, "target_ref")
    replacement_key = _unit_ref(unit, "replacement_ref")
    whatif = _unit_operation(unit) == "WHATIF"

    def _view(role: str, key: str, section_label: str) -> Dict[str, Any]:
        record = None
        if key:
            record = resolve_artifact_record(
                case, key, branch_id=preferred, graph=graph
            )
        view = artifact_content_view(record, ref=key, role=role)
        view["section_label"] = section_label
        return view

    if whatif:
        original = _view("Original", target_key, "Original artifact")
        replacement = _view("Replacement", replacement_key, "Replacement artifact")
        if (
            original.get("available")
            and replacement.get("available")
            and original.get("ref")
            and original.get("ref") == replacement.get("ref")
        ):
            replacement = artifact_content_view(
                None, ref=replacement_key, role="Replacement"
            )
            replacement["section_label"] = "Replacement artifact"
        return [original, replacement]

    views: List[Dict[str, Any]] = []
    if target_key:
        views.append(_view("Target", target_key, "Target artifact"))
    return views
