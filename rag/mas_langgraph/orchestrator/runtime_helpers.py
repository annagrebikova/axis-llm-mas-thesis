"""Helpers for LangGraph runtime context (live graph + unit tests)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langgraph.runtime import Runtime

from ..core.runtime import WorkflowRuntime


def workflow_runtime_context(**kwargs: Any) -> SimpleNamespace:
    """Build a minimal ``runtime`` object for direct node/route unit tests."""
    return SimpleNamespace(context=WorkflowRuntime(**kwargs))


def get_workflow_runtime(runtime: Runtime[WorkflowRuntime]) -> WorkflowRuntime:
    return runtime.context
