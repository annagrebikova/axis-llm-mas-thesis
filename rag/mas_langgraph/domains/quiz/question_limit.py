"""Per-run target/cap on distinct final accepted MCQs.

CLI ``--max-questions`` maps to ``OrchestratorConfig.max_final_questions``.
That field is the factual TARGET/CAP of accepted logical questions — not an
explanation-only slice. ``None`` means unlimited (non-batch default).

The workflow tries to reach the target while legitimate evidence-supported
blueprint opportunities remain. It must not relax quality gates to fill the
quota, and it may finish below target when opportunities are exhausted.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from ...input.constants import QUIZ_QUESTIONS_PER_ROUND

QUESTION_LIMIT_SEMANTICS = "final_accepted_mcqs"
COVERAGE_CAP_WARNING = "coverage_incomplete_due_to_question_cap"
CAP_REACHED_WARNING = "final_question_cap_reached"

EXHAUSTION_NO_REMAINING_BLUEPRINTS = "no_remaining_blueprints"
EXHAUSTION_NO_SUPPORTED_OPPORTUNITIES = "no_supported_generation_opportunities"
EXHAUSTION_EVIDENCE_BUDGET = "evidence_repair_budget_exhausted"
EXHAUSTION_BLUEPRINT_BUDGET = "blueprint_repair_budget_exhausted"
EXHAUSTION_CANDIDATES_REJECTED = "remaining_candidates_rejected"
EXHAUSTION_NO_COVERAGE_OPPORTUNITIES = "no_remaining_coverage_opportunities"


def effective_max_final_questions(config: Any) -> Optional[int]:
    if config is None:
        return None
    raw = getattr(config, "max_final_questions", None)
    if raw is None and isinstance(config, dict):
        raw = config.get("max_final_questions")
        if raw is None:
            raw = config.get("max_questions")
    if raw is None:
        return None
    limit = int(raw)
    if limit < 1:
        raise ValueError("max_final_questions must be a positive integer")
    return limit


def accepted_final_question_ids(state: Any) -> List[str]:
    """Distinct q_ids of accepted CandidateMCQs (revisions share one q_id)."""
    seen: List[str] = []
    found = set()
    for mcq in _accepted_mcqs(state):
        qid = str(getattr(mcq, "q_id", "") or "")
        if qid and qid not in found:
            found.add(qid)
            seen.append(qid)
    return seen


def accepted_final_question_count(state: Any) -> int:
    return len(accepted_final_question_ids(state))


def remaining_final_question_capacity(state: Any, config: Any) -> Optional[int]:
    """Slots left for newly accepted final questions. ``None`` means unlimited."""
    limit = effective_max_final_questions(config)
    if limit is None:
        return None
    return max(0, limit - accepted_final_question_count(state))


def final_question_cap_reached(state: Any, config: Any) -> bool:
    """True iff accepted logical questions already meet/exceed the target."""
    remaining = remaining_final_question_capacity(state, config)
    return remaining is not None and remaining <= 0


def generation_batch_size(
    remaining: Optional[int],
    *,
    default: int = QUIZ_QUESTIONS_PER_ROUND,
) -> int:
    if remaining is None:
        return max(0, int(default))
    return max(0, min(int(default), int(remaining)))


def slice_to_limit(items: Sequence[Any], remaining: Optional[int]) -> List[Any]:
    if remaining is None:
        return list(items)
    if remaining <= 0:
        return []
    return list(items)[: int(remaining)]


def generation_payloads_from_parsed(parsed: Any, *, limit: int = 1) -> List[Dict[str, Any]]:
    """Deterministically keep at most ``limit`` question objects from an LLM payload."""
    if parsed is None or limit <= 0:
        return []
    payloads: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        payloads = [p for p in parsed if isinstance(p, dict)]
    elif isinstance(parsed, dict):
        extra = parsed.get("questions")
        if isinstance(extra, list) and extra:
            payloads = [p for p in extra if isinstance(p, dict)]
        else:
            payloads = [parsed]
    return payloads[: int(limit)]


def first_generation_payload(parsed: Any) -> Optional[Dict[str, Any]]:
    payloads = generation_payloads_from_parsed(parsed, limit=1)
    return payloads[0] if payloads else None


def select_final_mcqs_within_limit(mcqs: Iterable[Any], limit: Optional[int]) -> List[Any]:
    """Deterministic cap: stable q_id order, at most ``limit`` logical questions."""
    by_qid: Dict[str, Any] = {}
    for mcq in mcqs:
        qid = str(getattr(mcq, "q_id", "") or "")
        if not qid or qid in by_qid:
            continue
        by_qid[qid] = mcq
    ordered = [by_qid[qid] for qid in sorted(by_qid)]
    if limit is None:
        return list(ordered)
    return ordered[: int(limit)]


def record_coverage_cap_warning(state: Any, *, gaps: Sequence[Any], config: Any) -> Optional[str]:
    if not final_question_cap_reached(state, config):
        return None
    if not gaps:
        return CAP_REACHED_WARNING
    warnings = list(getattr(state, "warnings", None) or [])
    if COVERAGE_CAP_WARNING not in warnings:
        warnings.append(COVERAGE_CAP_WARNING)
        try:
            state.warnings = warnings
        except Exception:
            pass
    return COVERAGE_CAP_WARNING


def planned_blueprint_list(state: Any) -> List[Any]:
    getter = getattr(state, "planned_blueprints", None)
    if callable(getter):
        try:
            return list(getter() or [])
        except Exception:
            return []
    blueprints = getattr(state, "mcq_blueprints", None) or {}
    return [
        bp for bp in (blueprints.values() if hasattr(blueprints, "values") else [])
        if getattr(bp, "status", "") == "planned"
    ]


def remaining_repairable_blueprints(state: Any, gaps: Optional[Sequence[Any]] = None) -> List[Any]:
    """New one-per-LO/section blueprints that current coverage logic can still plan."""
    from .coverage import create_repair_blueprints, find_coverage_gaps

    if gaps is None:
        try:
            gaps = find_coverage_gaps(state)
        except Exception:
            gaps = []
    try:
        return list(create_repair_blueprints(state, list(gaps)))
    except Exception:
        return []


def collect_generation_exhaustion_reasons(
    state: Any,
    *,
    graph_state: Optional[Dict[str, Any]] = None,
    config: Any = None,
) -> List[str]:
    """Deterministic stop reasons from workflow state. Never asks an LLM."""
    del config
    reasons: List[str] = []
    graph_state = graph_state or {}

    if not planned_blueprint_list(state):
        reasons.append(EXHAUSTION_NO_REMAINING_BLUEPRINTS)

    live_gaps: List[Any] = list(graph_state.get("coverage_gaps") or [])
    try:
        from .coverage import compute_mcq_coverage, find_coverage_gaps

        live_gaps = list(find_coverage_gaps(state) or live_gaps)
        coverage = compute_mcq_coverage(state)
    except Exception:
        coverage = {}

    repairable = remaining_repairable_blueprints(state, live_gaps)
    if not repairable:
        reasons.append(EXHAUSTION_NO_SUPPORTED_OPPORTUNITIES)

    missing: List[Any] = []
    getter = getattr(state, "get_sections_without_evidence", None)
    if callable(getter):
        try:
            missing = list(getter() or [])
        except Exception:
            missing = []
    evidence_iters = int(graph_state.get("evidence_repair_iterations", 0) or 0)
    max_evidence = int(graph_state.get("max_evidence_repair_iterations", 0) or 0)
    if missing and evidence_iters >= max_evidence:
        reasons.append(EXHAUSTION_EVIDENCE_BUDGET)

    blueprint_iters = int(
        graph_state.get(
            "blueprint_repair_iterations",
            graph_state.get("repair_iterations", 0),
        )
        or 0
    )
    max_blueprint = int(graph_state.get("max_blueprint_repair_iterations", 0) or 0)
    if live_gaps and blueprint_iters >= max_blueprint:
        reasons.append(EXHAUSTION_BLUEPRINT_BUDGET)

    rejected = [
        mcq
        for mcq in (getattr(state, "candidate_mcqs", None) or {}).values()
        if getattr(mcq, "status", "") == "rejected"
    ]
    if rejected:
        reasons.append(EXHAUSTION_CANDIDATES_REJECTED)

    uncovered_supported = list(coverage.get("uncovered_supported_sections") or [])
    if not uncovered_supported and not repairable:
        reasons.append(EXHAUSTION_NO_COVERAGE_OPPORTUNITIES)

    seen = set()
    ordered: List[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return ordered


def build_generation_limit_report(
    state: Any,
    *,
    config: Any,
    graph_state: Optional[Dict[str, Any]] = None,
    packaged_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Structured generation-target diagnostics for packaging and the batch manifest."""
    limit = effective_max_final_questions(config)
    if packaged_count is not None:
        accepted = int(packaged_count)
    else:
        accepted = accepted_final_question_count(state)
    cap_reached = limit is not None and accepted >= int(limit)
    exhausted = limit is not None and accepted < int(limit)
    reasons = (
        collect_generation_exhaustion_reasons(
            state, graph_state=graph_state, config=config
        )
        if exhausted
        else []
    )
    single = reasons[0] if len(reasons) == 1 else None
    return {
        "requested_max_questions": limit,
        "final_accepted_questions": accepted,
        "question_cap_reached": cap_reached,
        "generation_exhausted": exhausted,
        "generation_exhaustion_reasons": reasons,
        "generation_exhausted_reason": single,
    }


def _accepted_mcqs(state: Any) -> List[Any]:
    from .coverage import _accepted_mcqs as accepted

    return list(accepted(state))
