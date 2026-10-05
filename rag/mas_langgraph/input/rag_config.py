"""RAG integration settings — mirror rag_agents_cooperative CLI defaults."""

from dataclasses import dataclass
from typing import Optional

# Fixed workflow snapshot for experimental factual/replay runs.
DEFAULT_WORKFLOW_MODEL = "gpt-4.1-mini-2025-04-14"


@dataclass
class RAGConfig:
    pdf_id: str = "141"
    vector_store_path: str = "vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = DEFAULT_WORKFLOW_MODEL
    model_type: str = "openai"  # openai | huggingface
    # Secondary reproducibility control for experimental workflow LLM calls.
    # Correctness for replay still relies on WorkflowLLMReplayLedger, not provider RNG.
    temperature: float = 0.0
    device: str = "cpu"
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
