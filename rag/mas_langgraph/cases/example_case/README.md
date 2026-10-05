# Example case template

This package shows the **declarative / Python** split for a new LangGraph case.

## Declarative (`inputs/`)

- `case.yaml` — identity and python adapter path
- `workflow.yaml` — **descriptive** node metadata (does not execute the graph)
- `artifact_types.yaml` — artifact type catalog
- `provenance_schema.yaml` — allowed provenance relations
- `outcome_criteria.yaml` — criterion *metadata* (not accept/revise/reject policy)
- `intervention_contracts.yaml` — allowed replay operations
- `prompts/` — AXIS templates if the case is explanation-capable

## Python (this package)

Must still implement graph construction, node functions, payload wrappers,
outcome evaluation, provenance bindings, slicing, observation, and verbalization.

`build_graph()` is intentionally unimplemented here so this template is not a
second production workflow.
