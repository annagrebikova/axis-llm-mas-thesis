"""Agent invocation — stub ReAct loop or direct tool dispatch (mirrors RAGAgents)."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..input.agents import AGENT_METADATA
from ..input.runner_config import AGENT_INVOKE_METHODS, AGENT_PRIMARY_TOOL
from ..core.state_base import BaseCooperativeState
from ..tools.base import ToolContext
from ..tools.registry import build_tool_registry, tools_for_agent

if TYPE_CHECKING:
    from ..input.profile import MASProfile


class AgentRunner:
    """Runs specialist agents via tools (stub) or optional LangChain agents later."""

    def __init__(
        self,
        state: BaseCooperativeState,
        topic: str = "",
        coverage_plan=None,
        profile: Optional["MASProfile"] = None,
        rag_pipeline: Any = None,
        llm: Any = None,
        use_langchain: bool = False,
        agent_primary_tools: Optional[Dict[str, str]] = None,
        agent_invoke_methods: Optional[Dict[str, str]] = None,
    ) -> None:
        from ..input.profile import DEFAULT_PROFILE

        self.state = state  # Deprecated for LangGraph — use cooperative_state= on invoke()
        self.topic = topic
        self.profile = profile or DEFAULT_PROFILE
        self.coverage_plan = coverage_plan if coverage_plan is not None else self.profile.coverage_plan
        self.rag_pipeline = rag_pipeline
        self.llm = llm
        self.use_langchain = use_langchain
        self.primary_tools = agent_primary_tools or self.profile.agent_primary_tools or AGENT_PRIMARY_TOOL
        self.invoke_methods = agent_invoke_methods or self.profile.agent_invoke_methods or AGENT_INVOKE_METHODS
        self.tool_registry = build_tool_registry()
        self.execution_log: List[Dict[str, Any]] = []
        self._searched_queries: set = set()

    def _context(self, agent_id: str, cooperative_state: Optional[BaseCooperativeState] = None) -> ToolContext:
        cs = cooperative_state if cooperative_state is not None else self.state
        return ToolContext(
            state=cs,
            agent_id=agent_id,
            topic=self.topic,
            rag_pipeline=self.rag_pipeline,
            llm=self.llm,
            execution_log=self.execution_log,
        )

    def run_tool(
        self,
        agent_id: str,
        tool_name: str,
        input_str: str = "",
        *,
        cooperative_state: Optional[BaseCooperativeState] = None,
    ) -> str:
        handler = self.tool_registry.get(tool_name)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return handler.run(self._context(agent_id, cooperative_state), input_str)

    def run_agent(
        self,
        agent_id: str,
        input_str: str = "",
        *,
        cooperative_state: Optional[BaseCooperativeState] = None,
    ) -> Dict[str, Any]:
        """Generic agent run — stub uses primary tool; full mode can wire LangChain ReAct."""
        if self.use_langchain and self.llm is not None:
            return self._run_langchain_agent(agent_id, input_str, cooperative_state=cooperative_state)
        return self._run_stub_agent(agent_id, input_str, cooperative_state=cooperative_state)

    def next_retrieval_query(
        self,
        searched_queries: Any = None,
        step: int = 0,
        *,
        cooperative_state: Optional[BaseCooperativeState] = None,
    ):
        # Classic orchestrator: next_retrieval_query(step_int)
        if isinstance(searched_queries, int) and step == 0:
            step = searched_queries
            searched_queries = None
        from ..orchestrator.material_context import MaterialContext
        cs = cooperative_state if cooperative_state is not None else self.state
        mat = MaterialContext(cs, self.coverage_plan)
        searched = (
            searched_queries
            if searched_queries is not None
            else getattr(self, "_searched_queries", set())
        )
        return mat.next_retrieval_query(searched, step)

    def retrieval_exhausted(
        self,
        *,
        cooperative_state: Optional[BaseCooperativeState] = None,
        searched_queries: Optional[set] = None,
    ) -> bool:
        return self.next_retrieval_query(
            searched_queries=searched_queries,
            step=0,
            cooperative_state=cooperative_state,
        ) is None

    def _run_stub_agent(
        self,
        agent_id: str,
        input_str: str,
        *,
        cooperative_state: Optional[BaseCooperativeState] = None,
    ) -> Dict[str, Any]:
        tool_name = self.primary_tools.get(agent_id)
        if not tool_name:
            return {
                "role": agent_id,
                "result": f"No primary tool for {agent_id}",
                "error": "missing_primary_tool",
            }
        try:
            meta = AGENT_METADATA.get(agent_id, {})
            out = self.run_tool(agent_id, tool_name, input_str or meta.get("default_input", ""), cooperative_state=cooperative_state)
            return {"role": agent_id, "result": out}
        except Exception as e:
            return {"role": agent_id, "result": str(e), "error": str(e)}

    def _run_langchain_agent(
        self,
        agent_id: str,
        input_str: str,
        *,
        cooperative_state: Optional[BaseCooperativeState] = None,
    ) -> Dict[str, Any]:
        return self._run_stub_agent(agent_id, input_str, cooperative_state=cooperative_state)

    def retrieve(
        self,
        query: str,
        *,
        cooperative_state: Optional[BaseCooperativeState] = None,
        searched_queries: Optional[set] = None,
    ) -> Dict[str, Any]:
        from ..integrations.rag_direct_tools import normalize_query

        norm = normalize_query(query)
        searched = set(searched_queries if searched_queries is not None else self._searched_queries)
        if norm in searched:
            return {
                "role": "retriever",
                "result": f"Query already searched; skipping: {query}",
                "skipped": True,
            }
        searched.add(norm)
        out = self.run_tool(
            "retriever",
            "retrieve_chunks",
            query,
            cooperative_state=cooperative_state,
        )
        return {"role": "retriever", "result": out, "searched_queries": searched}

    def generate_quiz(self, topic: str) -> Dict[str, Any]:
        out = self.run_tool("quiz_generator", "generate_quiz_draft", topic)
        return {"role": "quiz_generator", "result": out}

    def verify_all_candidates(self) -> Dict[str, Any]:
        out = self.run_tool("verifier", "verify_all_candidates", "Verify all candidate questions.")
        return {"role": "verifier", "result": out}

    def export_quiz_pack(
        self,
        title: str,
        *,
        cooperative_state: Optional[BaseCooperativeState] = None,
    ) -> Dict[str, Any]:
        out = self.run_tool(
            "packager_export",
            "export_pack",
            title,
            cooperative_state=cooperative_state,
        )
        cs = cooperative_state if cooperative_state is not None else self.state
        pack_id = None
        q_count = 0
        final_packs = getattr(cs, "final_quiz_packs", None)
        if final_packs:
            pack_id = final_packs[-1].pack_id
            q_count = len(final_packs[-1].questions)
        return {
            "role": "packager_export",
            "result": out,
            "pack_id": pack_id,
            "question_count": q_count,
        }

    def invoke(
        self,
        agent_id: str,
        suggested_input: Optional[str] = None,
        *,
        cooperative_state: Optional[BaseCooperativeState] = None,
    ) -> Dict[str, Any]:
        """Orchestrator entry — dispatch via profile invoke map or generic run_agent."""
        inp = suggested_input or self.topic
        cs = cooperative_state if cooperative_state is not None else self.state
        method_name = self.invoke_methods.get(agent_id)
        if method_name and hasattr(self, method_name):
            method = getattr(self, method_name)
            if method_name == "verify_all_candidates":
                return method()
            if method_name == "export_quiz_pack":
                return method(inp or "Generated Quiz", cooperative_state=cs)
            return method(inp)
        return self.run_agent(agent_id, inp, cooperative_state=cs)

    def list_tools(self, agent_id: str) -> List[str]:
        return [t.name for t in tools_for_agent(agent_id, self.tool_registry)]
