"""Tool base types and execution context."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.state import CooperativeState


@dataclass
class ToolContext:
    state: "CooperativeState"
    agent_id: str
    topic: str = ""
    rag_pipeline: Any = None
    llm: Any = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, tool: str, input_data: Any = None, output: Any = None) -> None:
        self.execution_log.append({
            "step": len(self.execution_log) + 1,
            "agent": self.agent_id,
            "tool": tool,
            "input": input_data,
            "output": str(output)[:300] if output is not None else "",
        })


class ToolHandler(ABC):
    name: str

    @abstractmethod
    def run(self, ctx: ToolContext, input_str: str) -> str:
        """Execute tool; return observation string for agent."""
