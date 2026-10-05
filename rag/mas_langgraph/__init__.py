"""Cooperative multi-agent backbone — run via `python -m mas_langgraph` from EMaES/rag/."""

__all__ = [
    "AgentRunner",
    "BaseCooperativeState",
    "BaseNeedsEvaluator",
    "BaseWorkflowOrchestrator",
    "CaseDefinition",
    "CooperativeBackbone",
    "CooperativeState",
    "DEFAULT_PROFILE",
    "MASProfile",
    "NeedsEvaluator",
    "NeedsEvaluatorProtocol",
    "StateProtocol",
    "WorkflowExecutor",
    "WorkflowOrchestratorProtocol",
    "WorkflowRuntime",
    "WorkflowStepLog",
    "create_components",
    "get_profile",
]

_LAZY_EXPORTS = {
    "CooperativeBackbone": (".backbone", "CooperativeBackbone"),
    "create_components": (".factory", "create_components"),
    "AgentRunner": (".agents", "AgentRunner"),
    "DEFAULT_PROFILE": (".input.profile", "DEFAULT_PROFILE"),
    "MASProfile": (".input.profile", "MASProfile"),
    "get_profile": (".input.profile", "get_profile"),
    "BaseNeedsEvaluator": (".orchestrator", "BaseNeedsEvaluator"),
    "BaseWorkflowOrchestrator": (".orchestrator", "BaseWorkflowOrchestrator"),
    "NeedsEvaluator": (".orchestrator", "NeedsEvaluator"),
    "BaseCooperativeState": (".core", "BaseCooperativeState"),
    "CaseDefinition": (".core", "CaseDefinition"),
    "CooperativeState": (".cases.mcq.cooperative_state", "CooperativeState"),
    "NeedsEvaluatorProtocol": (".core", "NeedsEvaluatorProtocol"),
    "StateProtocol": (".core", "StateProtocol"),
    "WorkflowExecutor": (".core", "WorkflowExecutor"),
    "WorkflowOrchestratorProtocol": (".core", "WorkflowOrchestratorProtocol"),
    "WorkflowRuntime": (".core", "WorkflowRuntime"),
    "WorkflowStepLog": (".core.entities", "WorkflowStepLog"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        import importlib

        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
