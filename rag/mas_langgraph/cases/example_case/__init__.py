"""Example case template — demonstrates what a new case must provide.

This package imports only the generic backbone (no MCQ / quiz-domain dependencies).
Copy and replace stubs to implement another LangGraph MAS case.
"""

from pathlib import Path
from typing import Any, Optional

from ...core.case_config import load_case_configuration
from ...core.state_base import BaseCooperativeState

CASE_ID = "example_case"
_INPUTS = Path(__file__).resolve().parent / "inputs"


class ExampleCaseDefinition:
    """Minimal CaseDefinition stub for documentation and import tests."""

    case_id: str = CASE_ID
    case_version: int = 1

    def __init__(self, inputs_dir: Optional[Path] = None) -> None:
        self.inputs_dir = Path(inputs_dir) if inputs_dir else _INPUTS
        self._configuration = None

    def load_case_metadata(self) -> Any:
        from ...core.case_config import load_case_metadata

        return load_case_metadata(self.inputs_dir / "case.yaml")

    def load_workflow_metadata(self) -> Any:
        from ...core.case_config import load_workflow_metadata

        return load_workflow_metadata(self.inputs_dir / "workflow.yaml")

    def load_case_configuration(self, compiled_graph: Any = None, require_graph: bool = False):
        if self._configuration is None:
            self._configuration = load_case_configuration(
                self,
                compiled_graph=compiled_graph,
                strict=True,
                require_graph=require_graph,
                require_prompts=False,
                outcome_evaluator=None,
            )
        return self._configuration

    def validate_configuration(self, compiled_graph: Any = None, require_graph: bool = False):
        self._configuration = None
        return self.load_case_configuration(
            compiled_graph=compiled_graph, require_graph=require_graph
        )

    def load_artifact_types(self) -> Any:
        return self.load_case_configuration().artifact_types

    def load_provenance_schema(self) -> Any:
        return self.load_case_configuration().provenance_schema

    def load_outcome_evaluator(self) -> Any:
        return None

    def load_intervention_contracts(self) -> Any:
        return self.load_case_configuration().intervention_contracts

    def load_verbalizer(self) -> Any:
        return None

    def load_prompt_templates(self) -> Any:
        from ...axis.models import AxisPromptTemplates

        return AxisPromptTemplates.load_from_dir(self.inputs_dir / "prompts")

    def state_class(self) -> type:
        return BaseCooperativeState

    def build_graph(self, checkpointer: Any = None) -> Any:
        raise NotImplementedError(
            "Example case has no executable graph. Implement build_graph() "
            "in Python; workflow.yaml does not replace LangGraph construction."
        )
