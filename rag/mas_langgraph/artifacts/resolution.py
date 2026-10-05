"""Enforce artifact-resolution projection before node / provenance execution."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from langgraph.runtime import Runtime

from ..core.runtime import WorkflowRuntime


def enforce_artifact_resolution(
    node_id: str,
    node_fn: Callable[..., Dict[str, Any]],
) -> Callable[..., Dict[str, Any]]:
    """Outer wrapper: project availability, then call the (instrumented) node.

    Ordering::

        raw state → resolution projection → provenance instrumentation → business node
    """

    def wrapped(
        state: Dict[str, Any],
        runtime: Runtime[WorkflowRuntime],
    ) -> Dict[str, Any]:
        projected: Mapping[str, Any] = state
        case = getattr(runtime.context, "case_definition", None)
        if case is not None and hasattr(case, "project_state_for_artifact_resolution"):
            projected = case.project_state_for_artifact_resolution(
                state, node_id=node_id
            )
        return node_fn(projected, runtime) or {}

    wrapped.__name__ = getattr(node_fn, "__name__", node_id)
    wrapped.__qualname__ = getattr(node_fn, "__qualname__", node_id)
    return wrapped
