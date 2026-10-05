"""Bridge cooperative MAS to rag_agents_cooperative.RAGAgents (real PDF + vector store + LLM)."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

from ..input.constants import QUIZ_QUESTIONS_PER_ROUND, VERIFY_BATCH_SIZE
from ..input.rag_config import RAGConfig
from .rag_direct_tools import (
    direct_atomize_chunks,
    direct_build_outline,
    direct_curate_evidence,
    direct_curriculum_lo,
    direct_detect_duplicates,
    direct_generate_quiz,
    direct_verify_candidates,
    retrieve_with_dedupe,
)

if TYPE_CHECKING:
    from rag_agents_cooperative import RAGAgents

# Agents that call monolith methods directly (no LangChain ReAct).
DIRECT_DISPATCH_AGENTS = frozenset({
    "structure_outline",
    "curriculum_lo",
    "quality_duplicate_detector",
    "evidence_curator",
    "atomizer",
    "quiz_generator",
    "verifier",
})


def _ensure_rag_on_path() -> Path:
    """mas_langgraph/integrations -> EMaES/rag (sibling of rag_agents_cooperative.py)."""
    rag_dir = Path(__file__).resolve().parent.parent.parent
    if str(rag_dir) not in sys.path:
        sys.path.insert(0, str(rag_dir))
    return rag_dir


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    rag_dir = _ensure_rag_on_path()
    # EMaES/.env (one level above rag/)
    env_path = rag_dir.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    # Also try rag/.env if present
    local_env = rag_dir / ".env"
    if local_env.exists():
        load_dotenv(local_env)


def attach_snapshot_counts(state: Any) -> None:
    """Monolith CooperativeState lacks snapshot_counts(); cooperative orchestrator needs it.

    Bind module-level helpers via ``types.MethodType`` so a reference-sweep
    snapshot can be pickled and restored in a fresh process. Nested closures
    attached as instance attributes are not pickleable.
    """
    from ..domains.quiz.state_integration import rebind_pickleable_state_helpers

    rebind_pickleable_state_helpers(state)


def create_rag_agents(config: RAGConfig) -> "RAGAgents":
    """Initialize RAGAgents with vector store for pdf_id (same as monolith)."""
    _load_dotenv()
    _ensure_rag_on_path()
    from rag_agents_cooperative import RAGAgents

    kwargs: Dict[str, Any] = {
        "pdf_id": config.pdf_id,
        "vector_store_path": config.vector_store_path,
        "embedding_model": config.embedding_model,
        "llm_model": config.llm_model,
        "model_type": config.model_type,
        "temperature": config.temperature,
        "device": config.device,
    }
    if config.model_type == "huggingface":
        kwargs["huggingface_api_key"] = config.huggingface_api_key
    else:
        kwargs["openai_api_key"] = config.openai_api_key

    agents = RAGAgents(**kwargs)
    attach_snapshot_counts(agents.state)
    # Wrap workflow LLM for deterministic factual/NOOP/CF replay (ledger is
    # activated via workflow_llm_execution_context around graph.invoke).
    try:
        from ..replay.llm_determinism import install_replay_aware_llm

        install_replay_aware_llm(agents, temperature=config.temperature)
    except Exception:
        # Keep agents usable even if wrap fails in non-experiment contexts.
        pass
    return agents


class RAGAgentRunner:
    """Delegates agent execution to RAGAgents; deterministic tools bypass ReAct."""

    def __init__(
        self,
        rag_agents: "RAGAgents",
        profile: Any = None,
        coverage_plan: Any = None,
    ) -> None:
        from ..input.profile import DEFAULT_PROFILE

        self.rag_agents = rag_agents
        self.state = rag_agents.state
        self.profile = profile or DEFAULT_PROFILE
        self.coverage_plan = coverage_plan or self.profile.coverage_plan
        self.rag_pipeline = rag_agents.rag_pipeline
        self.llm = rag_agents.llm
        self.use_langchain = True
        self.pdf_id = rag_agents.pdf_id
        self.topic = ""
        self._searched_queries: Set[str] = set()
        self._quiz_generation_rounds = 0

        if coverage_plan is not None:
            self.rag_agents.coverage_plan = coverage_plan

    @property
    def execution_log(self):
        return self.rag_agents.execution_log

    def next_retrieval_query(self, step: int = 0) -> Optional[str]:
        from ..orchestrator.material_context import MaterialContext

        mat = MaterialContext(self.state, self.coverage_plan, pdf_id=self.pdf_id)
        return mat.next_retrieval_query(self._searched_queries, step)

    def retrieval_exhausted(self) -> bool:
        return self.next_retrieval_query(0) is None

    def retrieve(self, query: str) -> Dict[str, Any]:
        result, self._searched_queries = retrieve_with_dedupe(
            self.rag_agents,
            query,
            self._searched_queries,
        )
        return result

    def invoke(self, agent_id: str, suggested_input: Optional[str] = None) -> Dict[str, Any]:
        """Dispatch to RAGAgents; deterministic agents skip ReAct."""
        from ..orchestrator.material_context import MaterialContext

        mat = MaterialContext(self.state, self.coverage_plan, pdf_id=self.pdf_id)

        if agent_id == "retriever":
            query = suggested_input or self.next_retrieval_query(0) or mat.retrieval_query(0)
            return self.retrieve(query)

        if agent_id == "structure_outline":
            return direct_build_outline(self.rag_agents)

        if agent_id == "curriculum_lo":
            return direct_curriculum_lo(self.rag_agents)

        if agent_id == "quality_duplicate_detector":
            return direct_detect_duplicates(self.rag_agents)

        if agent_id == "evidence_curator":
            return direct_curate_evidence(self.rag_agents)

        if agent_id == "atomizer":
            return direct_atomize_chunks(self.rag_agents)

        if agent_id == "verifier":
            return direct_verify_candidates(
                self.rag_agents, batch_size=VERIFY_BATCH_SIZE,
            )

        if agent_id == "quiz_generator":
            spec = suggested_input or mat.quiz_target()
            self._quiz_generation_rounds += 1
            return direct_generate_quiz(
                self.rag_agents,
                spec,
                num_questions=QUIZ_QUESTIONS_PER_ROUND,
            )

        inp = suggested_input or mat.document_theme()

        if agent_id == "packager_export":
            return self.rag_agents.export_quiz_pack(inp or "Generated Quiz")

        return self.rag_agents.run_agent(agent_id, inp or "")

    def list_tools(self, agent_id: str) -> list:
        executors = getattr(self.rag_agents, "_agent_executors", {})
        executor = executors.get(agent_id)
        if executor and hasattr(executor, "tools"):
            return [t.name for t in executor.tools]
        return []
