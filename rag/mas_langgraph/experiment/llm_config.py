"""Explicit explanation-layer LLM configuration (separate from workflow LLM)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .llm import make_llm_generate, unwrap_base_llm


# Fixed snapshots for thesis reproducibility (no floating aliases).
DEFAULT_WORKFLOW_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_INTERROGATION_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_SYNTHESIS_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_REASONING_EFFORT = "none"


@dataclass(frozen=True)
class ExplanationLLMConfig:
    """Participant-facing / AXIS interrogation models (never workflow ledger)."""

    interrogation_model: str = DEFAULT_INTERROGATION_MODEL
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    temperature: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interrogation_model": self.interrogation_model,
            "synthesis_model": self.synthesis_model,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class ExperimentModelBundle:
    """Resolved model provenance for one experiment run."""

    workflow_model: str = DEFAULT_WORKFLOW_MODEL
    workflow_temperature: float = 0.0
    interrogation_model: str = DEFAULT_INTERROGATION_MODEL
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    cross_cove_verifier_model: str = ""
    cross_cove_verifier_temperature: float = 0.0

    def to_export_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "workflow": {
                "model": self.workflow_model,
                "temperature": self.workflow_temperature,
            },
            "axis_interrogation": {
                "model": self.interrogation_model,
                "reasoning_effort": self.reasoning_effort,
            },
            "synthesis": {
                "model": self.synthesis_model,
                "reasoning_effort": self.reasoning_effort,
            },
        }
        if self.cross_cove_verifier_model:
            payload["cross_cove_verifier"] = {
                "model": self.cross_cove_verifier_model,
                "temperature": self.cross_cove_verifier_temperature,
            }
        return payload

    def startup_lines(self) -> Tuple[str, ...]:
        lines = [
            f"Workflow model: {self.workflow_model}",
            f"AXIS interrogation model: {self.interrogation_model}",
            f"Synthesis model: {self.synthesis_model}",
            f"Workflow temperature: {self.workflow_temperature}",
            f"Synthesis reasoning effort: {self.reasoning_effort}",
        ]
        if self.cross_cove_verifier_model:
            lines.insert(
                1,
                f"Cross-CoVe verifier model: {self.cross_cove_verifier_model}",
            )
        return tuple(lines)


def default_explanation_llm_config() -> ExplanationLLMConfig:
    return ExplanationLLMConfig()


def _openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or None


def build_chat_openai(
    *,
    model: str,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
) -> Any:
    """Construct a ChatOpenAI (or compatible) instance for explanation roles."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI  # type: ignore

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    # GPT-5.x family: prefer explicit reasoning_effort when supported.
    if reasoning_effort is not None and str(model).startswith("gpt-5"):
        # langchain-openai may accept model_kwargs or reasoning_effort directly.
        try:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        except TypeError:
            kwargs["model_kwargs"] = {"reasoning_effort": reasoning_effort}
    return ChatOpenAI(**kwargs)


def build_explanation_generators(
    config: Optional[ExplanationLLMConfig] = None,
    *,
    openai_api_key: Optional[str] = None,
) -> Tuple[Callable[[str], str], Callable[[str], str], ExperimentModelBundle]:
    """Create interrogation + synthesis callables that never touch the workflow ledger."""
    cfg = config or default_explanation_llm_config()
    key = openai_api_key or _openai_api_key()
    if key:
        os.environ.setdefault("OPENAI_API_KEY", key)

    interrogation_llm = build_chat_openai(
        model=cfg.interrogation_model,
        temperature=cfg.temperature,
        reasoning_effort=cfg.reasoning_effort,
    )
    # Same config/snapshot for fairness across conditions; separate instance OK.
    if cfg.synthesis_model == cfg.interrogation_model:
        synthesis_llm = interrogation_llm
    else:
        synthesis_llm = build_chat_openai(
            model=cfg.synthesis_model,
            temperature=cfg.temperature,
            reasoning_effort=cfg.reasoning_effort,
        )

    # Defensive unwrap if a caller accidentally passes a wrapped LLM.
    interrogation_llm = unwrap_base_llm(interrogation_llm)
    synthesis_llm = unwrap_base_llm(synthesis_llm)

    bundle = ExperimentModelBundle(
        interrogation_model=cfg.interrogation_model,
        synthesis_model=cfg.synthesis_model,
        reasoning_effort=cfg.reasoning_effort,
    )
    return make_llm_generate(interrogation_llm), make_llm_generate(synthesis_llm), bundle


def resolve_workflow_model_from_backbone(backbone: Any) -> Tuple[str, float]:
    """Best-effort workflow model/temperature from a completed backbone."""
    rag = getattr(backbone, "rag_agents", None)
    if rag is None:
        rag = getattr(getattr(backbone, "agent_runner", None), "rag_agents", None)
    base = unwrap_base_llm(getattr(rag, "base_llm", None) or getattr(rag, "llm", None))
    model = (
        getattr(base, "model_name", None)
        or getattr(base, "model", None)
        or DEFAULT_WORKFLOW_MODEL
    )
    temp = getattr(base, "temperature", 0.0)
    try:
        temp_f = float(temp) if temp is not None else 0.0
    except (TypeError, ValueError):
        temp_f = 0.0
    return str(model), temp_f
