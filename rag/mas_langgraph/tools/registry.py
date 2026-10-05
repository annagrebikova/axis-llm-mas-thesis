"""Tool registry — builds handlers from input/tools_registry.py."""

from typing import Dict, List

from ..input.tools_registry import TOOLS_BY_AGENT
from .base import ToolHandler
from .stubs import PRIMARY_TOOLS, PassthroughTool


def build_tool_registry() -> Dict[str, ToolHandler]:
    """One handler instance per unique tool name."""
    registry: Dict[str, ToolHandler] = {}
    all_names = {name for names in TOOLS_BY_AGENT.values() for name in names}
    for name in sorted(all_names):
        if name in PRIMARY_TOOLS:
            registry[name] = PRIMARY_TOOLS[name]()
        else:
            registry[name] = PassthroughTool(name)
    return registry


def tools_for_agent(agent_id: str, registry: Dict[str, ToolHandler]) -> List[ToolHandler]:
    names = TOOLS_BY_AGENT.get(agent_id, ["get_state_summary"])
    return [registry[n] for n in names if n in registry]
