"""MCQCaseDefinition — wires current MCQ behavior into the generic CaseDefinition API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ...core.case_config import (
    CaseConfiguration,
    CaseConfigurationError,
    load_axis_case_capabilities,
    load_case_configuration,
    load_case_metadata,
    load_outcome_criteria_file,
    load_workflow_metadata,
    require_case_files,
)
from ...core.input_loader import load_mapping
from .criteria import McqOutcomeEvaluator

_INPUTS = Path(__file__).resolve().parent / "inputs"


class MCQCaseDefinition:
    """Adapter pointing at current MCQ behavior without rewriting it.

    Declarative metadata in ``cases/mcq/inputs/`` is authoritative on the
    strict AXIS-ready path. Executable graph, nodes, and decision policy
    remain Python.
    """

    case_id: str = "mcq_generation"
    case_version: int = 1

    def __init__(
        self,
        inputs_dir: Optional[Path] = None,
        *,
        strict: bool = True,
    ) -> None:
        self.inputs_dir = Path(inputs_dir) if inputs_dir else _INPUTS
        self.strict = strict
        self._configuration: Optional[CaseConfiguration] = None
        self._axis_capabilities: Any = None

    def load_case_metadata(self) -> Any:
        return load_case_metadata(self.inputs_dir / "case.yaml")

    def load_workflow_metadata(self) -> Any:
        return load_workflow_metadata(self.inputs_dir / "workflow.yaml")

    def _outcome_evaluator_from_file(self) -> McqOutcomeEvaluator:
        path = self.inputs_dir / "outcome_criteria.yaml"
        if not path.exists():
            if self.strict:
                raise CaseConfigurationError(
                    f"Required case file missing: {path}"
                )
            return McqOutcomeEvaluator(criteria_meta={}, criteria_order=[])
        items = load_outcome_criteria_file(path)
        meta = {str(item["name"]): dict(item) for item in items}
        order = [str(item["name"]) for item in items]
        return McqOutcomeEvaluator(criteria_meta=meta, criteria_order=order)

    def load_case_configuration(
        self,
        *,
        compiled_graph: Any = None,
        require_graph: bool = False,
    ) -> CaseConfiguration:
        if self._configuration is not None and compiled_graph is None:
            return self._configuration
        if self.strict:
            require_case_files(self.inputs_dir, require_prompts=True)
        configuration = load_case_configuration(
            self,
            compiled_graph=compiled_graph,
            strict=self.strict,
            require_graph=require_graph,
            require_prompts=self.strict,
            outcome_evaluator=self._outcome_evaluator_from_file(),
        )
        if compiled_graph is None:
            self._configuration = configuration
        return configuration

    def validate_configuration(
        self,
        compiled_graph: Any = None,
        require_graph: bool = False,
    ) -> CaseConfiguration:
        self._configuration = None
        return self.load_case_configuration(
            compiled_graph=compiled_graph,
            require_graph=require_graph,
        )

    def load_artifact_types(self) -> Any:
        return self.load_case_configuration().artifact_types

    def load_provenance_schema(self) -> Any:
        return self.load_case_configuration().provenance_schema

    def load_outcome_evaluator(self) -> McqOutcomeEvaluator:
        return self.load_case_configuration().outcome_evaluator

    def load_intervention_contracts(self) -> Any:
        return self.load_case_configuration().intervention_contracts

    def intervention_artifact_priority(self, artifact_type: str) -> int:
        """MCQ-specific ordering among currently valid intervention contracts.

        Lower numbers are preferred. Provenance presence ≠ causal importance;
        this only ranks *testable* hypotheses for the interrogation LLM.
        """
        order = {
            "EvidenceChunk": 1,
            "KnowledgeAtom": 2,
            "MCQBlueprint": 3,
            "CandidateMCQ": 4,
        }
        return int(order.get(str(artifact_type or ""), 50))

    def project_state_for_artifact_resolution(
        self,
        state: Any,
        *,
        node_id: str = "",
    ) -> Any:
        from .intervention_view import project_state_for_artifact_resolution

        return project_state_for_artifact_resolution(state, node_id=node_id)

    def prompt_dir(self) -> Path:
        return self.inputs_dir / "prompts"

    def load_verbalizer(self) -> Any:
        from .verbalizer import McqVerbalizer

        templates = self.load_prompt_templates()
        return McqVerbalizer(
            prompt_dir=self.prompt_dir(),
            prompt_templates=templates,
        )

    def load_prompt_templates(self) -> Any:
        from ...axis.models import AxisPromptTemplates

        templates = AxisPromptTemplates.load_from_dir(self.prompt_dir())
        templates.validate_required(
            "system",
            "factual_context",
            "interrogation",
            "intermediate_explanation",
            "final_explanation",
        )
        return templates

    def load_run_observation_adapter(self) -> Any:
        from .comparison import MCQRunObservationAdapter

        return MCQRunObservationAdapter()

    def is_replay_observable(
        self,
        final_state: Any,
        explained_case: Any,
        *,
        branch_id: str = "",
    ) -> Any:
        """Case-level stop predicate for target-bounded AXIS replay."""
        from .replay_observable import evaluate_mcq_replay_observable

        return evaluate_mcq_replay_observable(
            final_state, explained_case, branch_id=branch_id
        )

    def load_provenance_slice_adapter(self) -> Any:
        from .slicing import MCQProvenanceSliceAdapter

        return MCQProvenanceSliceAdapter()

    def load_provenance_instrumentation_adapter(self) -> Any:
        from .provenance_instrumentation import MCQProvenanceInstrumentationAdapter

        return MCQProvenanceInstrumentationAdapter()

    def load_explained_case_resolver(self) -> Any:
        from .explained_case import MCQExplainedCaseResolver

        return MCQExplainedCaseResolver()

    def load_axis_capabilities(self, compiled_graph: Any = None) -> Any:
        if self._axis_capabilities is not None and compiled_graph is None:
            return self._axis_capabilities
        caps = load_axis_case_capabilities(
            self, compiled_graph=compiled_graph, strict=self.strict
        )
        if compiled_graph is None:
            self._axis_capabilities = caps
        return caps

    def state_class(self) -> type:
        from .cooperative_state import CooperativeState

        return CooperativeState

    def build_graph(self, checkpointer: Any = None) -> Any:
        from ...core.graph_builder import build_mcq_workflow_graph

        return build_mcq_workflow_graph(checkpointer=checkpointer)
