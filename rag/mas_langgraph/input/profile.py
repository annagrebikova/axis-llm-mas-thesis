"""MAS profile — central registry for swappable core, orchestrator, and runner components.

Edit this file to point at your own classes in domains/ or custom modules.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

from ..cases.mcq.cooperative_state import CooperativeState
from ..core.state_base import BaseCooperativeState
from ..domains.quiz import domain_bundle as quiz_domain
from ..orchestrator.base_workflow import BaseWorkflowOrchestrator
from ..orchestrator.needs_base import BaseNeedsEvaluator
from .coverage_plan import DEFAULT_COVERAGE_PLAN
from .orchestrator_config import OrchestratorConfig


def _default_agent_runner_class() -> Type[Any]:
    from ..agents.runner import AgentRunner
    return AgentRunner


def _default_case_definition_class() -> Type[Any]:
    from ..cases.mcq.case_definition import MCQCaseDefinition

    return MCQCaseDefinition


@dataclass
class MASProfile:
    """Bundles every swappable component for a multi-agent system."""

    name: str = quiz_domain.DOMAIN_NAME

    # --- Case (generic backbone bridge) ---
    case_definition_class: Type[Any] = field(
        default_factory=_default_case_definition_class,
    )

    # --- Core (override for new blackboard / entities) ---
    state_class: Type[BaseCooperativeState] = CooperativeState

    # --- Orchestrator (override for new conditions / loop behavior) ---
    needs_evaluator_class: Type[BaseNeedsEvaluator] = quiz_domain.NEEDS_EVALUATOR_CLASS
    orchestrator_class: Type[BaseWorkflowOrchestrator] = BaseWorkflowOrchestrator
    orchestrator_config: OrchestratorConfig = field(
        default_factory=quiz_domain.default_orchestrator_config,
    )
    coverage_fn: Optional[Callable[..., Any]] = quiz_domain.COVERAGE_FN

    # --- Runner / tools ---
    agent_runner_class: Type[Any] = field(default_factory=_default_agent_runner_class)
    agent_primary_tools: Dict[str, str] = field(
        default_factory=lambda: dict(quiz_domain.AGENT_PRIMARY_TOOLS),
    )
    agent_invoke_methods: Dict[str, str] = field(
        default_factory=lambda: dict(quiz_domain.AGENT_INVOKE_METHODS_MAP),
    )

    # --- Shared config objects ---
    coverage_plan: Any = field(default_factory=lambda: DEFAULT_COVERAGE_PLAN)


DEFAULT_PROFILE = MASProfile()


def get_profile(name: Optional[str] = None) -> MASProfile:
    """Return profile by name; today only 'quiz_rag' / default is built-in."""
    if name is None or name == DEFAULT_PROFILE.name:
        return DEFAULT_PROFILE
    raise ValueError(
        f"Unknown MAS profile: {name!r}. "
        f"Register it in input/profile.py or pass a custom MASProfile instance."
    )
