# Case Definition

A case implements `CaseDefinition` (factual LangGraph execution):

- `case_id` / `case_version`
- `build_graph(checkpointer=None)`
- `load_case_metadata()` / `load_workflow_metadata()` / `load_case_configuration()`
- `load_artifact_types()`
- `load_provenance_schema()`
- `load_outcome_evaluator()`
- `load_intervention_contracts()`
- `state_class()`
- optional `project_state_for_artifact_resolution(...)`

AXIS-capable cases additionally implement `AxisCaseDefinition`:

- `load_verbalizer()`
- `load_prompt_templates()` → `AxisPromptTemplates`
- `load_provenance_slice_adapter()`
- `load_run_observation_adapter()`
- `load_provenance_instrumentation_adapter()`

`workflow.yaml` is descriptive metadata validated against the compiled graph. It does **not** replace LangGraph graph construction.

MCQ: `mas_langgraph.cases.mcq.MCQCaseDefinition` loads YAML from `cases/mcq/inputs/`.

Example stub: `mas_langgraph.cases.example_case.ExampleCaseDefinition` (no MCQ imports).
