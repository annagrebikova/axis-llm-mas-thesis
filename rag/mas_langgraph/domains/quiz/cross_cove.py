"""Cross-CoVe answer verification for CandidateMCQs (independent verifier majority vote)."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .entities import CandidateMCQ, CrossCoVeReport
from .generation import log_ok, log_warn
from .versioning import (
    candidate_version_number,
    get_operational_evidence_report,
    report_targets_candidate,
)

if TYPE_CHECKING:
    from ...core.state import CooperativeState

OPTION_LABELS = ("A", "B", "C", "D", "E", "F")


def _label_for_option_index(index: int) -> str:
    if index < len(OPTION_LABELS):
        return OPTION_LABELS[index]
    return f"opt_{index}"


def _option_id_to_label(candidate: CandidateMCQ, option_id: str) -> str:
    for idx, opt in enumerate(candidate.options):
        if opt.option_id == option_id:
            return _label_for_option_index(idx)
        if opt.option_id.upper() == option_id.upper():
            return _label_for_option_index(idx)
    if option_id.upper() in OPTION_LABELS:
        return option_id.upper()
    return option_id


def _label_to_option_id(candidate: CandidateMCQ, label: str) -> Optional[str]:
    if not label:
        return None
    normalized = label.strip().upper()
    if normalized in OPTION_LABELS:
        idx = OPTION_LABELS.index(normalized)
        if idx < len(candidate.options):
            return candidate.options[idx].option_id
    for opt in candidate.options:
        if opt.option_id.upper() == normalized:
            return opt.option_id
    return None


def build_cross_cove_prompt(candidate: CandidateMCQ) -> str:
    """Build verifier prompt — stem and options only, no correct answer or evidence."""
    option_lines = []
    for idx, opt in enumerate(candidate.options):
        label = _label_for_option_index(idx)
        option_lines.append(f"{label}) {opt.text}")

    options_block = "\n".join(option_lines)
    return f"""You are verifying a multiple-choice question.
Select the single best answer.

Return JSON only:
{{
  "selected_option_id": "A",
  "reason": "short explanation"
}}

MCQ:
{candidate.stem}

Options:
{options_block}

Important:
Return only one option ID.
Do not say that multiple answers are correct.
Do not use external explanations.
Choose the best answer based on the question and options.
"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def parse_verifier_answer(
    raw: str,
    candidate: Optional[CandidateMCQ] = None,
) -> Tuple[Optional[str], List[str]]:
    """Extract option label (A/B/C/D) from messy LLM output."""
    notes: List[str] = []
    if not raw:
        notes.append("Empty verifier response.")
        return None, notes

    parsed = _extract_json(raw)
    if parsed is not None:
        selected = parsed.get("selected_option_id") or parsed.get("option_id") or parsed.get("answer")
        if selected:
            label = str(selected).strip().upper()
            if re.fullmatch(r"[A-F]", label):
                return label, notes
            if candidate is not None:
                mapped = _option_id_to_label(candidate, str(selected))
                if mapped:
                    return mapped.upper() if mapped.upper() in OPTION_LABELS else mapped, notes

    upper = raw.upper()
    for label in OPTION_LABELS:
        if re.search(rf'\b{label}\b', upper):
            return label, notes

    match = re.search(r'"selected_option_id"\s*:\s*"([A-F])"', raw, re.IGNORECASE)
    if match:
        return match.group(1).upper(), notes

    notes.append("Could not parse verifier answer.")
    return None, notes


def majority_vote(answers: List[Optional[str]]) -> Tuple[Optional[str], float]:
    """Return majority option label and confidence; None on tie or no valid answers."""
    valid = [a.strip().upper() for a in answers if a]
    if not valid:
        return None, 0.0

    counts = Counter(valid)
    top_label, top_count = counts.most_common(1)[0]
    if len(counts) > 1 and counts.most_common(2)[1][1] == top_count:
        return None, 0.0

    confidence = top_count / len(valid)
    return top_label, confidence


def _invoke_llm(llm: Any, prompt: str, *, sample_index: Optional[int] = None) -> str:
    from ...replay.llm_determinism import workflow_llm_context

    with workflow_llm_context(
        layer="native_mcq",
        component="cross_cove",
        sample_index=sample_index,
    ):
        response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def _structural_blockers(candidate: CandidateMCQ) -> List[str]:
    blockers: List[str] = []
    if candidate.status == "rejected":
        blockers.append("candidate_rejected")
    if not candidate.options:
        blockers.append("no_options")
    if len(candidate.options) < 4:
        blockers.append("too_few_options")
    if not candidate.correct_option_id:
        blockers.append("missing_correct_option_id")
    option_ids = {o.option_id for o in candidate.options}
    if candidate.correct_option_id not in option_ids:
        blockers.append("correct_option_id_not_in_options")
    return blockers


def _stub_verifier_answer(candidate: CandidateMCQ) -> str:
    """Deterministic stub: pick label of marked correct option."""
    label = _option_id_to_label(candidate, candidate.correct_option_id)
    return json.dumps({"selected_option_id": label, "reason": "Stub verifier."})


def run_cross_cove_for_candidate(
    candidate: CandidateMCQ,
    verifier_llm: Optional[Any] = None,
    verifier_model_name: Optional[str] = None,
    n_runs: int = 3,
) -> CrossCoVeReport:
    """Run Cross-CoVe verification for one candidate."""
    notes: List[str] = []
    generator_label = _option_id_to_label(candidate, candidate.correct_option_id)

    actual_model = llm_model_identity(verifier_llm) or verifier_model_name
    blockers = _structural_blockers(candidate)
    if blockers:
        return CrossCoVeReport(
            q_id=candidate.q_id,
            generator_answer=generator_label,
            verifier_answers=[],
            majority_answer="",
            passed=False,
            candidate_version=candidate_version_number(candidate),
            confidence=0.0,
            verifier_model=actual_model,
            notes=[f"Structural blocker: {b}" for b in blockers],
        )

    prompt = build_cross_cove_prompt(candidate)
    verifier_answers: List[str] = []
    parsed_labels: List[Optional[str]] = []

    for run_idx in range(n_runs):
        if verifier_llm is not None:
            raw = _invoke_llm(verifier_llm, prompt, sample_index=run_idx)
        else:
            raw = _stub_verifier_answer(candidate)
            if run_idx == 0:
                notes.append("Stub Cross-CoVe verifier (no LLM).")

        label, parse_notes = parse_verifier_answer(raw, candidate)
        notes.extend(parse_notes)
        verifier_answers.append(label or "")
        parsed_labels.append(label)

    majority_label, confidence = majority_vote(parsed_labels)
    majority_option_id = (
        _label_to_option_id(candidate, majority_label) if majority_label else None
    )
    passed = (
        majority_option_id is not None
        and majority_option_id == candidate.correct_option_id
    )

    if majority_label is None:
        notes.append("No majority answer (tie or unparsed responses).")

    return CrossCoVeReport(
        q_id=candidate.q_id,
        generator_answer=generator_label,
        verifier_answers=verifier_answers,
        majority_answer=majority_label or "",
        passed=passed,
        candidate_version=candidate_version_number(candidate),
        confidence=confidence,
        verifier_model=actual_model,
        notes=notes,
    )


def _apply_cross_cove_flags(candidate: CandidateMCQ, report: CrossCoVeReport) -> None:
    if "cross_cove_failed" in candidate.flags:
        candidate.flags.remove("cross_cove_failed")
    if "cross_cove_no_majority" in candidate.flags:
        candidate.flags.remove("cross_cove_no_majority")

    if not report.passed:
        if "cross_cove_failed" not in candidate.flags:
            candidate.flags.append("cross_cove_failed")
    if not report.majority_answer:
        if "cross_cove_no_majority" not in candidate.flags:
            candidate.flags.append("cross_cove_no_majority")


def _should_verify_candidate(
    candidate: CandidateMCQ,
    state: "CooperativeState",
    *,
    evidence_supported_only: bool = True,
) -> bool:
    if candidate.status == "rejected":
        return False
    if not evidence_supported_only:
        return True
    evidence = get_operational_evidence_report(state, candidate)
    if evidence is None:
        return False
    return evidence.supported_by_pdf


def run_cross_cove_for_all_candidates(
    state: "CooperativeState",
    verifier_llm: Optional[Any] = None,
    max_candidates: Optional[int] = None,
    *,
    force: bool = False,
    n_runs: int = 3,
    verifier_model_name: Optional[str] = None,
    evidence_supported_only: bool = True,
) -> List[CrossCoVeReport]:
    """Run Cross-CoVe on candidates and store reports on state."""
    reports: List[CrossCoVeReport] = []
    candidates = list(state.candidate_mcqs.values())
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    for candidate in candidates:
        existing = state.cross_cove_reports.get(candidate.q_id)
        if not force and existing is not None and report_targets_candidate(existing, candidate):
            reports.append(existing)
            continue
        if not _should_verify_candidate(candidate, state, evidence_supported_only=evidence_supported_only):
            continue

        report = run_cross_cove_for_candidate(
            candidate,
            verifier_llm=verifier_llm,
            verifier_model_name=verifier_model_name,
            n_runs=n_runs,
        )
        _apply_cross_cove_flags(candidate, report)
        state.store_cross_cove_report(report)
        state.store_candidate_mcq(candidate)
        reports.append(report)

    return reports


def llm_model_identity(llm: Any) -> str:
    """Best-effort provider model name from a (possibly replay-wrapped) LLM."""
    if llm is None:
        return ""
    from ...replay.llm_determinism import unwrap_base_llm

    base = unwrap_base_llm(llm)
    return str(
        getattr(base, "model_name", None)
        or getattr(base, "model", None)
        or ""
    )


def is_live_provider_llm(llm: Any) -> bool:
    """True when ``llm`` is a real ChatOpenAI (possibly replay-wrapped)."""
    if llm is None:
        return False
    from ...replay.llm_determinism import unwrap_base_llm

    base = unwrap_base_llm(llm)
    module = str(getattr(type(base), "__module__", "") or "")
    name = type(base).__name__
    return name == "ChatOpenAI" or "langchain_openai" in module


def resolve_cross_cove_verifier_model(
    config: Any = None,
    *,
    generator_model: Optional[str] = None,
) -> Optional[str]:
    """Env override, then orchestrator config, then generator model."""
    env = os.getenv("CROSS_COVE_VERIFIER_MODEL")
    if env is not None and str(env).strip():
        return str(env).strip()
    if config is not None:
        cfg = getattr(config, "cross_cove_verifier_model", None)
        if cfg:
            return str(cfg)
    from ...input.constants import CROSS_COVE_VERIFIER_MODEL

    if CROSS_COVE_VERIFIER_MODEL:
        return str(CROSS_COVE_VERIFIER_MODEL)
    return generator_model


def resolve_cross_cove_verifier_temperature(config: Any = None) -> float:
    if config is not None:
        temp = getattr(config, "cross_cove_verifier_temperature", None)
        if temp is not None:
            try:
                return float(temp)
            except (TypeError, ValueError):
                pass
    from ...input.constants import CROSS_COVE_VERIFIER_TEMPERATURE

    return float(CROSS_COVE_VERIFIER_TEMPERATURE)


def build_cross_cove_verifier_llm(
    *,
    verifier_model: Optional[str],
    generator_llm: Any,
    generator_model: Optional[str],
    temperature: float = 0.0,
) -> Tuple[Any, str, bool]:
    """Construct the Cross-CoVe verifier once (never inside a graph node).

    Returns ``(llm, resolved_model_name, is_separate_object)``.
    """
    resolved = str(verifier_model or generator_model or "")
    separate = bool(
        verifier_model
        and generator_model
        and str(verifier_model) != str(generator_model)
    )
    if not separate:
        return generator_llm, resolved, False
    if generator_llm is None:
        # Stub / unit-test workflows: do not construct a provider client.
        return None, str(verifier_model), True
    if not is_live_provider_llm(generator_llm):
        # Scripted test LLMs must not spawn ChatOpenAI. Reuse the stub object;
        # production install happens only for a live generator client.
        return generator_llm, llm_model_identity(generator_llm) or resolved, False

    from ...experiment.llm_config import DEFAULT_REASONING_EFFORT, build_chat_openai
    from ...replay.llm_determinism import ReplayAwareWorkflowLLM

    base = build_chat_openai(
        model=str(verifier_model),
        temperature=float(temperature),
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    return ReplayAwareWorkflowLLM(base), str(verifier_model), True


def print_cross_cove_model_config(
    *,
    generator_model: Optional[str],
    verifier_model: Optional[str],
    n_runs: int,
) -> None:
    gen = str(generator_model or "")
    ver = str(verifier_model or "")
    print(
        "[MODEL CONFIG]\n"
        f"workflow_generator={gen}\n"
        f"cross_cove_verifier={ver}\n"
        f"cross_cove_n_runs={int(n_runs)}"
    )
    if gen and ver and gen == ver:
        print(
            "Cross-CoVe verifier uses the same model as generator; "
            "this behaves closer to Normal-CoVe than a cross-model verifier."
        )


def log_cross_cove_model_choice(
    verifier_model_name: Optional[str],
    generator_model_name: Optional[str],
) -> None:
    """Log whether Cross-CoVe uses a separate verifier model."""
    if verifier_model_name and generator_model_name and verifier_model_name != generator_model_name:
        log_ok(f"Cross-CoVe verifier model: {verifier_model_name}")
    elif verifier_model_name and generator_model_name and verifier_model_name == generator_model_name:
        log_warn(
            "Cross-CoVe verifier uses the same model as generator; "
            "this behaves closer to Normal-CoVe than a cross-model verifier."
        )
    elif verifier_model_name:
        log_ok(f"Cross-CoVe verifier model: {verifier_model_name}")
    else:
        log_warn(
            "Cross-CoVe verifier uses the same model as generator; "
            "this behaves closer to Normal-CoVe than a cross-model verifier."
        )
