"""Factory — build state, runner, and orchestrator from a MASProfile."""

from typing import Any, Dict, Optional, Tuple, Union

from .agents.runner import AgentRunner
from .core.state_base import BaseCooperativeState
from .input.profile import DEFAULT_PROFILE, MASProfile
from .input.rag_config import RAGConfig
from .orchestrator.base_workflow import BaseWorkflowOrchestrator
from .orchestrator.needs_base import BaseNeedsEvaluator

RunnerType = Union[AgentRunner, Any]


def create_needs_evaluator(
    profile: MASProfile,
    state: BaseCooperativeState,
    agent_runner: Any = None,
) -> BaseNeedsEvaluator:
    kwargs: Dict[str, Any] = {}
    if profile.coverage_fn is not None:
        kwargs["coverage_fn"] = profile.coverage_fn
    if agent_runner is not None:
        kwargs["agent_runner"] = agent_runner
    cfg = getattr(profile, "orchestrator_config", None)
    if cfg is not None:
        kwargs["max_final_questions"] = getattr(cfg, "max_final_questions", None)
    return profile.needs_evaluator_class(state, profile.coverage_plan, **kwargs)


def create_orchestrator(
    profile: MASProfile,
    state: BaseCooperativeState,
    agent_runner: RunnerType,
    needs_evaluator: Optional[BaseNeedsEvaluator] = None,
    orchestrator_mode: str = "classic",
    rag_agents: Any = None,
) -> BaseWorkflowOrchestrator:
    needs = needs_evaluator or create_needs_evaluator(profile, state, agent_runner=agent_runner)

    if orchestrator_mode == "langgraph":
        from .orchestrator.langgraph_workflow import LangGraphWorkflowOrchestrator

        case_def = profile.case_definition_class()
        return LangGraphWorkflowOrchestrator(
            state=state,
            agent_runner=agent_runner,
            needs_evaluator=needs,
            config=profile.orchestrator_config,
            rag_agents=rag_agents or getattr(agent_runner, "rag_agents", None),
            case_definition=case_def,
        )

    return profile.orchestrator_class(
        state=state,
        agent_runner=agent_runner,
        needs_evaluator=needs,
        config=profile.orchestrator_config,
    )


def create_agent_runner(
    profile: MASProfile,
    state: BaseCooperativeState,
    topic: str = "",
    rag_pipeline: Any = None,
    llm: Any = None,
    use_langchain: bool = False,
) -> AgentRunner:
    return profile.agent_runner_class(
        state=state,
        topic=topic,
        profile=profile,
        coverage_plan=profile.coverage_plan,
        rag_pipeline=rag_pipeline,
        llm=llm,
        use_langchain=use_langchain,
    )


def create_rag_components(
    rag_config: RAGConfig,
    profile: Optional[MASProfile] = None,
    orchestrator_mode: str = "classic",
) -> Tuple[BaseCooperativeState, RunnerType, BaseWorkflowOrchestrator, Any]:
    """Build MAS using rag_agents_cooperative.RAGAgents (real PDF + LLM)."""
    from .integrations.rag_agents_bridge import RAGAgentRunner, create_rag_agents

    prof = profile or DEFAULT_PROFILE
    rag_agents = create_rag_agents(rag_config)
    rag_agents.coverage_plan = prof.coverage_plan
    runner = RAGAgentRunner(
        rag_agents,
        profile=prof,
        coverage_plan=prof.coverage_plan,
    )
    orchestrator = create_orchestrator(
        prof, runner.state, runner,
        orchestrator_mode=orchestrator_mode,
        rag_agents=rag_agents,
    )
    return runner.state, runner, orchestrator, rag_agents


def create_components(
    profile: Optional[MASProfile] = None,
    rag_pipeline: Any = None,
    llm: Any = None,
    use_langchain: bool = False,
    rag_config: Optional[RAGConfig] = None,
    orchestrator_mode: str = "classic",
) -> Tuple[BaseCooperativeState, RunnerType, BaseWorkflowOrchestrator]:
    """Instantiate full MAS stack from profile (stub) or RAG config."""
    if rag_config is not None:
        state, runner, orchestrator, _ = create_rag_components(
            rag_config, profile=profile, orchestrator_mode=orchestrator_mode,
        )
        return state, runner, orchestrator

    prof = profile or DEFAULT_PROFILE
    state = prof.state_class()
    runner = create_agent_runner(
        prof, state,
        rag_pipeline=rag_pipeline, llm=llm, use_langchain=use_langchain,
    )
    orchestrator = create_orchestrator(
        prof, state, runner, orchestrator_mode=orchestrator_mode,
    )
    return state, runner, orchestrator
