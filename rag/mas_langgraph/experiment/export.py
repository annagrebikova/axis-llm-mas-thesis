"""JSON / Markdown exporters for the explanation experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..artifacts.models import ArtifactRef
from .models import (
    EXPERIMENT_SCHEMA_VERSION,
    ExplanationConditionResult,
    MCQExplanationComparison,
    QuizExplanationExperiment,
)


def _ref_dict(ref: ArtifactRef) -> Dict[str, Any]:
    return {
        "logical_id": ref.logical_id,
        "version": int(ref.version),
        "key": str(ref),
    }


def _condition_dict(cond: ExplanationConditionResult, *, include_raw: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "condition": cond.condition,
        "explanation": cond.explanation,
        "status": cond.status,
        "warnings": list(cond.warnings),
        "evidence_summary": dict(cond.evidence_summary),
    }
    if include_raw:
        out["raw_evidence"] = dict(cond.raw_evidence)
    return out


def case_to_dict(
    case: MCQExplanationComparison,
    *,
    include_blinding: bool = True,
    include_raw: bool = True,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "candidate_ref": _ref_dict(case.candidate_ref),
        "assessment_ref": _ref_dict(case.assessment_ref),
        "decision_ref": _ref_dict(case.decision_ref),
        "run_id": case.run_id,
        "branch_id": case.branch_id,
        "user_question": case.user_question,
        "question": {
            "stem": case.question_stem,
            "options": [
                {
                    "option_id": o.option_id,
                    "text": o.text,
                    "is_correct": o.is_correct,
                }
                for o in case.answer_options
            ],
            "correct_option": case.correct_option_id,
        },
        "factual_decision": case.factual_decision,
        "conditions": {
            "trace": _condition_dict(case.trace, include_raw=include_raw),
            "trace_provenance": _condition_dict(
                case.trace_provenance, include_raw=include_raw
            ),
            "axis": _condition_dict(case.axis, include_raw=include_raw),
        },
        "axis_status": case.axis_status,
        "axis_causal_evidence_available": case.axis_causal_evidence_available,
        "counterfactual_count": case.counterfactual_count,
        "successful_replay_count": case.successful_replay_count,
        "visualization_data": dict(getattr(case, "visualization_data", None) or {}),
    }
    if include_blinding:
        payload["blinding"] = dict(case.blinding)
    return payload


def experiment_to_dict(
    experiment: QuizExplanationExperiment,
    *,
    include_blinding: bool = True,
    include_raw: bool = True,
) -> Dict[str, Any]:
    return {
        "schema_version": experiment.schema_version,
        "run_id": experiment.run_id,
        "branch_id": experiment.branch_id,
        "experiment_seed": experiment.experiment_seed,
        "user_question": experiment.user_question,
        "question_count": len(experiment.cases),
        "include_historical_versions": experiment.include_historical_versions,
        "synthesis_model": experiment.synthesis_model,
        "models": dict(getattr(experiment, "models", None) or {}),
        "evidence_access": experiment.evidence_access,
        "warnings": list(experiment.warnings),
        "workflow_llm_diagnostics": dict(
            getattr(experiment, "workflow_llm_diagnostics", None) or {}
        ),
        "timing": dict(getattr(experiment, "timing", None) or {}),
        "axis_interrogation_mode": getattr(
            experiment, "axis_interrogation_mode", None
        )
        or "constrained",
        "axis_interrogation_policy": getattr(
            experiment, "axis_interrogation_policy", None
        )
        or "counterfactual_only",
        "cases": [
            case_to_dict(c, include_blinding=include_blinding, include_raw=include_raw)
            for c in experiment.cases
        ],
    }


def dump_explanation_experiment(
    experiment: QuizExplanationExperiment,
    path: str,
    *,
    include_blinding: bool = True,
    include_raw: bool = True,
) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = experiment_to_dict(
        experiment, include_blinding=include_blinding, include_raw=include_raw
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return str(out.resolve())


def _format_options(case: MCQExplanationComparison) -> str:
    lines = []
    for opt in case.answer_options:
        lines.append(f"{opt.option_id}. {opt.text}")
    return "\n".join(lines)


def dump_explanation_experiment_markdown(
    experiment: QuizExplanationExperiment,
    path: str,
    *,
    title: Optional[str] = None,
    document_id: Optional[str] = None,
) -> str:
    """Researcher-facing Markdown with true condition names."""
    heading = "# Quiz Explanation Experiment"
    if title:
        heading = f"# Quiz Explanation Experiment — {title}"
    parts = [
        heading,
        "",
        f"- run_id: `{experiment.run_id}`",
        f"- seed: `{experiment.experiment_seed}`",
        f"- questions: {len(experiment.cases)}",
        f"- user question: {experiment.user_question}",
    ]
    if document_id:
        parts.append(f"- document_id: `{document_id}`")
    parts.append("")
    for i, case in enumerate(experiment.cases, start=1):
        parts.extend(
            [
                f"# Question {i}",
                "",
                case.question_stem,
                "",
                _format_options(case),
                "",
                _format_correct_answer(case),
                f"System decision: {str(case.factual_decision).upper()}",
                f"Candidate: `{case.candidate_ref}`",
                "",
                "## Trace",
                "",
                case.trace.explanation,
                "",
                "## Trace + Provenance",
                "",
                case.trace_provenance.explanation,
                "",
                "## AXIS",
                "",
                case.axis.explanation,
                "",
                f"_axis_status={case.axis_status}; "
                f"causal_evidence={case.axis_causal_evidence_available}; "
                f"replays={case.successful_replay_count}_",
                "",
            ]
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return str(out.resolve())


def _format_correct_answer(case: MCQExplanationComparison) -> str:
    cid = case.correct_option_id or ""
    text = ""
    for opt in case.answer_options:
        if opt.is_correct or (cid and opt.option_id == cid):
            text = opt.text
            break
    if cid and text:
        return f"Correct answer: {cid}. {text}"
    if cid:
        return f"Correct answer: {cid}"
    if text:
        return f"Correct answer: {text}"
    return "Correct answer:"


def dump_blinded_experiment_markdown(
    experiment: QuizExplanationExperiment,
    path: str,
) -> str:
    """Evaluator-facing Markdown with Explanation A/B/C only."""
    parts = [
        "# Quiz Explanation Evaluation",
        "",
        "For each question, rate Explanation A, B, and C.",
        "Do not assume which method produced each explanation.",
        "",
    ]
    for i, case in enumerate(experiment.cases, start=1):
        by_cond = {
            "trace": case.trace.explanation,
            "trace_provenance": case.trace_provenance.explanation,
            "axis": case.axis.explanation,
        }
        parts.extend(
            [
                f"# Question {i}",
                "",
                case.question_stem,
                "",
                _format_options(case),
                "",
                f"System decision: {str(case.factual_decision).upper()}",
                "",
            ]
        )
        for label in ("A", "B", "C"):
            cond = case.blinding.get(label)
            text = by_cond.get(cond or "", "")
            parts.extend([f"## Explanation {label}", "", text, ""])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return str(out.resolve())


def dump_condition_key(
    experiment: QuizExplanationExperiment,
    path: str,
) -> str:
    """Answer key mapping Explanation A/B/C → true condition (researcher only)."""
    key: Dict[str, Any] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": experiment.run_id,
        "experiment_seed": experiment.experiment_seed,
        "mapping": {
            str(case.candidate_ref): dict(case.blinding) for case in experiment.cases
        },
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, ensure_ascii=False)
    return str(out.resolve())
