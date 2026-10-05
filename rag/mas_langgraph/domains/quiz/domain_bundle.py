"""Quiz domain wiring for MAS profile (kept separate to avoid import cycles)."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from ...input.coverage_plan import DEFAULT_COVERAGE_PLAN
from ...input.orchestrator_config import OrchestratorConfig
from ...input.runner_config import AGENT_INVOKE_METHODS, AGENT_PRIMARY_TOOL
from ...orchestrator.coverage import compute_coverage_gaps
from ...orchestrator.needs import NeedsEvaluator

if TYPE_CHECKING:
    from ...core.state import CooperativeState


def format_quiz_workflow_result(
    *,
    topic: str,
    steps: int,
    orchestration_log: List[Any],
    final_state: Dict[str, Any],
    state: "CooperativeState",
    **_: Any,
) -> Dict[str, Any]:
    from .question_limit import accepted_final_question_count

    diagnostics = dict(
        (getattr(state, "mcq_coverage_summary", None) or {}).get("generation_limit") or {}
    )
    packs = list(getattr(state, "final_quiz_packs", None) or [])
    packaged = len(packs[-1].questions) if packs else accepted_final_question_count(state)
    payload: Dict[str, Any] = {
        "packages_exported": final_state.get("final_quiz_packs_len", 0),
        "validated_questions": final_state.get("validated_questions_len", 0),
        "mcq_blueprints": final_state.get("mcq_blueprints_len", 0),
        "candidate_mcqs": final_state.get("candidate_mcqs_len", 0),
        "final_accepted_questions": diagnostics.get(
            "final_accepted_questions", packaged
        ),
        "question_cap_reached": bool(diagnostics.get("question_cap_reached", False)),
        "generation_exhausted": bool(diagnostics.get("generation_exhausted", False)),
        "generation_exhaustion_reasons": list(
            diagnostics.get("generation_exhaustion_reasons") or []
        ),
    }
    if "requested_max_questions" in diagnostics:
        payload["requested_max_questions"] = diagnostics.get("requested_max_questions")
    return payload


def default_orchestrator_config() -> OrchestratorConfig:
    from ...input.orchestrator_config import DEFAULT_ORCHESTRATOR_CONFIG

    return OrchestratorConfig(
        max_steps=DEFAULT_ORCHESTRATOR_CONFIG.max_steps,
        no_progress_iterations=DEFAULT_ORCHESTRATOR_CONFIG.no_progress_iterations,
        rules=list(DEFAULT_ORCHESTRATOR_CONFIG.rules),
        progress_metric_keys=DEFAULT_ORCHESTRATOR_CONFIG.progress_metric_keys,
        terminal_success_agent=DEFAULT_ORCHESTRATOR_CONFIG.terminal_success_agent,
        step_modulo_default=DEFAULT_ORCHESTRATOR_CONFIG.step_modulo_default,
        step_threshold_default=DEFAULT_ORCHESTRATOR_CONFIG.step_threshold_default,
        cross_cove_enabled=DEFAULT_ORCHESTRATOR_CONFIG.cross_cove_enabled,
        cross_cove_n_runs=DEFAULT_ORCHESTRATOR_CONFIG.cross_cove_n_runs,
        cross_cove_verifier_model=DEFAULT_ORCHESTRATOR_CONFIG.cross_cove_verifier_model,
        cross_cove_verifier_temperature=(
            DEFAULT_ORCHESTRATOR_CONFIG.cross_cove_verifier_temperature
        ),
        max_revision_iterations=DEFAULT_ORCHESTRATOR_CONFIG.max_revision_iterations,
        format_workflow_result=format_quiz_workflow_result,
        max_final_questions=DEFAULT_ORCHESTRATOR_CONFIG.max_final_questions,
    )


DOMAIN_NAME = "quiz_rag"
NEEDS_EVALUATOR_CLASS = NeedsEvaluator
COVERAGE_PLAN = DEFAULT_COVERAGE_PLAN
COVERAGE_FN = compute_coverage_gaps
AGENT_PRIMARY_TOOLS = AGENT_PRIMARY_TOOL
AGENT_INVOKE_METHODS_MAP = AGENT_INVOKE_METHODS
