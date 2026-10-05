# Adding a New Case

To reuse the backbone without MCQ dependencies:

1. Copy `cases/example_case/` to `cases/my_case/`.
2. Provide **declarative configuration** under `inputs/`:
   - `case.yaml`
   - `workflow.yaml` (descriptive; does **not** replace LangGraph construction)
   - `artifact_types.yaml`
   - `provenance_schema.yaml`
   - `outcome_criteria.yaml` (criterion metadata only)
   - `intervention_contracts.yaml`
   - `prompts/` for AXIS, if the case is explanation-capable
3. Provide **Python implementation**:
   - graph builder (`build_graph`) and node functions
   - artifact payloads / wrap-unwrap adapters and registration
   - outcome evaluator and acceptance policy
   - artifact-resolution projection
   - provenance instrumentation adapter
   - slicing adapter and run-observation adapter
   - verbalizer
4. Register via `WorkflowRuntime(case_definition=MyCaseDefinition())`.

Declarative YAML describes identity, types, schema, and contracts. Executable decision, routing, and replay logic stay in Python adapters.

The framework is generic when `cases.example_case` imports cleanly without `domains.quiz`.
