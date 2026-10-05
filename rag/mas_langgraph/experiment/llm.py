"""Runtime LLM adapters for experiment synthesis and AXIS interrogation."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple

from ..replay.llm_determinism import unwrap_base_llm
from .timing import (
    _extract_usage,
    _phase,
    _round,
    _target,
    current_timing,
)

__all__ = [
    "invoke_llm_text",
    "make_llm_generate",
    "resolve_runtime_llm",
    "default_llm_generate_from_backbone",
    "unwrap_base_llm",
]


def invoke_llm_text(llm: Any, prompt: str) -> str:
    """Best-effort text extraction from LangChain-style or callable LLMs."""
    if llm is None:
        raise ValueError("llm is None")
    # Explanation / synthesis must never use the workflow replay ledger.
    llm = unwrap_base_llm(llm)
    model = str(
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or type(llm).__name__
    )
    phase = str(_phase.get() or "explanation")
    target = str(_target.get() or "")
    round_idx = _round.get()
    timing = current_timing()
    if timing is not None:
        print(
            f"[EXPLANATION LLM START]\n"
            f"target={target or '?'}\n"
            f"round={round_idx if round_idx is not None else ''}\n"
            f"phase={phase}\n"
            f"model={model}"
        )
    t0 = time.perf_counter()
    out: Any = None
    try:
        if callable(llm) and not hasattr(llm, "invoke"):
            out = llm(prompt)
        elif hasattr(llm, "invoke"):
            out = llm.invoke(prompt)
        elif hasattr(llm, "predict"):
            out = llm.predict(prompt)
        else:
            raise TypeError(f"Unsupported LLM type: {type(llm)!r}")
        return getattr(out, "content", None) or str(out)
    finally:
        duration = time.perf_counter() - t0
        if timing is not None:
            usage = _extract_usage(out)
            print(
                f"[EXPLANATION LLM END]\n"
                f"target={target or '?'}\n"
                f"round={round_idx if round_idx is not None else ''}\n"
                f"phase={phase}\n"
                f"duration_s={duration:.4f}"
            )
            timing.record_explanation_llm(
                phase=phase,
                duration_s=duration,
                model=model,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
            )


def make_llm_generate(llm: Any) -> Callable[[str], str]:
    base = unwrap_base_llm(llm)

    def _gen(prompt: str) -> str:
        return invoke_llm_text(base, prompt)

    return _gen


def resolve_runtime_llm(
    *,
    backbone: Any = None,
    agent_runner: Any = None,
    rag_agents: Any = None,
    llm: Any = None,
) -> Tuple[Optional[Any], str]:
    """Locate the configured explanation LLM (unwrapped base model).

    Prefer ``base_llm`` / ``explanation_llm`` when the workflow LLM has been
    wrapped by ``ReplayAwareWorkflowLLM``, so synthesis never hits the ledger.
    """
    candidates = []
    if llm is not None:
        candidates.append(llm)
    for obj in (backbone, rag_agents, agent_runner):
        if obj is None:
            continue
        for attr in (
            "explanation_llm",
            "base_llm",
            "llm",
        ):
            val = getattr(obj, attr, None)
            if val is not None:
                candidates.append(val)
        nested = getattr(obj, "rag_agents", None)
        if nested is not None:
            for attr in ("explanation_llm", "base_llm", "llm"):
                val = getattr(nested, attr, None)
                if val is not None:
                    candidates.append(val)
        nested_ar = getattr(obj, "agent_runner", None)
        if nested_ar is not None:
            for attr in ("explanation_llm", "base_llm", "llm"):
                val = getattr(nested_ar, attr, None)
                if val is not None:
                    candidates.append(val)
            nested2 = getattr(nested_ar, "rag_agents", None)
            if nested2 is not None:
                for attr in ("explanation_llm", "base_llm", "llm"):
                    val = getattr(nested2, attr, None)
                    if val is not None:
                        candidates.append(val)

    for source in candidates:
        if source is None:
            continue
        unwrapped = unwrap_base_llm(source)
        name = (
            getattr(unwrapped, "model_name", None)
            or getattr(unwrapped, "model", None)
            or type(unwrapped).__name__
        )
        return unwrapped, str(name)
    return None, ""


def default_llm_generate_from_backbone(backbone: Any) -> Optional[Callable[[str], str]]:
    llm, _ = resolve_runtime_llm(backbone=backbone)
    if llm is None:
        return None
    return make_llm_generate(llm)
