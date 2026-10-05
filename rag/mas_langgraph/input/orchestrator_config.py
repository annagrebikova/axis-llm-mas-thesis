"""Bundled orchestrator settings — edit or replace via input/profile.py."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agents import TERMINAL_SUCCESS_AGENT
from .constants import (
    CROSS_COVE_ENABLED,
    CROSS_COVE_N_RUNS,
    CROSS_COVE_VERIFIER_MODEL,
    CROSS_COVE_VERIFIER_TEMPERATURE,
    LANGGRAPH_MAX_REVISION_ITERATIONS,
    MAX_ORCHESTRATION_STEPS,
    NO_PROGRESS_ITERATIONS,
    QUIZ_GENERATION_STEP_MODULO,
    QUIZ_GENERATION_STEP_THRESHOLD,
)
from .orchestration_rules import ORCHESTRATION_RULES, PROGRESS_METRIC_KEYS


@dataclass
class OrchestratorConfig:
    """Settings consumed by BaseWorkflowOrchestrator."""

    max_steps: int = MAX_ORCHESTRATION_STEPS
    no_progress_iterations: int = NO_PROGRESS_ITERATIONS
    rules: List[Dict[str, Any]] = field(default_factory=lambda: list(ORCHESTRATION_RULES))
    progress_metric_keys: Tuple[str, ...] = PROGRESS_METRIC_KEYS
    terminal_success_agent: str = TERMINAL_SUCCESS_AGENT
    step_modulo_default: int = QUIZ_GENERATION_STEP_MODULO
    step_threshold_default: int = QUIZ_GENERATION_STEP_THRESHOLD
    # Maps input_template name -> resolver on needs evaluator (see base_workflow)
    input_templates: Tuple[str, ...] = (
        "section_query", "material_query", "quiz_target", "coverage_topic", "pack_title",
    )
    cross_cove_enabled: bool = CROSS_COVE_ENABLED
    cross_cove_n_runs: int = CROSS_COVE_N_RUNS
    cross_cove_verifier_model: Optional[str] = CROSS_COVE_VERIFIER_MODEL
    cross_cove_verifier_temperature: float = CROSS_COVE_VERIFIER_TEMPERATURE
    max_revision_iterations: int = LANGGRAPH_MAX_REVISION_ITERATIONS
    format_workflow_result: Optional[Callable[..., Dict[str, Any]]] = None
    # Per-run cap on distinct final accepted MCQs. None = unlimited (non-batch default).
    max_final_questions: Optional[int] = None


DEFAULT_ORCHESTRATOR_CONFIG = OrchestratorConfig()
