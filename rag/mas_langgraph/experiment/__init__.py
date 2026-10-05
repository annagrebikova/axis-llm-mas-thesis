"""Thesis experiment: Trace vs Trace+Provenance vs AXIS ablation."""

from .export import (
    dump_blinded_experiment_markdown,
    dump_condition_key,
    dump_explanation_experiment,
    dump_explanation_experiment_markdown,
)
from .models import (
    ExplanationConditionResult,
    MCQExplanationComparison,
    QuizExplanationExperiment,
)
from .service import QuizExplanationExperimentService
from .synthesis import SHARED_FINAL_SYNTHESIS_PROMPT, build_shared_final_prompt
from .verbalizer import ExplanationEvidenceView, build_evidence_view

__all__ = [
    "ExplanationConditionResult",
    "ExplanationEvidenceView",
    "MCQExplanationComparison",
    "QuizExplanationExperiment",
    "QuizExplanationExperimentService",
    "SHARED_FINAL_SYNTHESIS_PROMPT",
    "build_evidence_view",
    "build_shared_final_prompt",
    "dump_explanation_experiment",
    "dump_explanation_experiment_markdown",
    "dump_blinded_experiment_markdown",
    "dump_condition_key",
]
