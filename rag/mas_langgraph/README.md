# MAS LangGraph — Multi-Agent Quiz Pipeline

Modular, **profile-driven** cooperative MAS living under `EMaES/rag/mas_langgraph/`.
Integrates with `rag_agents_cooperative.py` (vector store + LLM tools).

## Layout

```
rag/
├── rag_agents_cooperative.py
├── rag_pipeline.py
├── docs/                       # Architecture, AXIS, reproducibility, thesis mapping
├── mas_langgraph/              # ★ Main MAS package
│   ├── core/                   # Runtime, executor, graph builder, case config
│   ├── artifacts/              # Immutable versioned artifacts
│   ├── provenance/             # Occurrences + D_schema / D_run
│   ├── replay/                 # Checkpoints, interventions, fork-and-replay
│   ├── axis/                   # Explanation service, controller, export
│   ├── cases/mcq/              # MCQ case definition + adapters
│   ├── cases/example_case/     # Declarative template (no executable graph)
│   ├── orchestrator/           # Classic + LangGraph node implementations
│   ├── domains/quiz/           # Legacy MCQ domain logic (compatibility)
│   └── backbone.py             # Public CooperativeBackbone entry point
└── vector_store/
```

## Package roles

| Package | Role |
|---------|------|
| `core/` | Protocols, `WorkflowRuntime`, authoritative graph builder, case configuration |
| `artifacts/` | Immutable `ArtifactRef` / envelopes / registry / resolver |
| `provenance/` | Occurrence recording + schema + run relations |
| `replay/` | Checkpoints, branches, NOOP/REMOVE/ADD/REPLACE |
| `axis/` | Queries, slicing, comparison, `AxisExplanationService`, JSON export |
| `cases/mcq/` | MCQ case definition, CooperativeState, YAML inputs, adapters |
| `cases/example_case/` | Template case without MCQ deps |

Requires **langgraph ≥ 1.0** (`context_schema` + `invoke(..., context=)`).
Thesis development/tests used **langgraph 1.2.9**.

See `rag/docs/architecture.md`, `rag/docs/axis_explainer.md`,
`rag/docs/thesis_mapping.md`, `rag/docs/reproducibility.md`.

## Quick start

Run all commands from **`EMaES/rag`**:

**Stub mode** (no PDF / API keys):

```bash
cd EMaES/rag
python -m mas_langgraph --orchestrate --stub
python -m mas_langgraph --orchestrate --stub --orchestrator langgraph
```

**RAG mode**:

```bash
python rag_pipeline.py --pdf-id 141
python -m mas_langgraph --orchestrate --pdf-id 141 --dump-pack quiz.json
python -m mas_langgraph --orchestrate --pdf-id 141 --orchestrator langgraph
```

## Programmatic use + AXIS

```python
from mas_langgraph import CooperativeBackbone
from mas_langgraph.axis.export import dump_axis_explanation

bb = CooperativeBackbone(orchestrator_mode="langgraph")
summary = bb.run_full_workflow()

refs = bb.list_explainable_artifacts()
if refs:
    # Autonomous interrogation uses the configured runtime LLM when
    # llm_generate is omitted. Pass scripted_queries= for deterministic tests.
    explanation = bb.explain_artifact(
        refs[0],
        "Why did the system produce this decision?",
    )
    dump_axis_explanation(explanation, "axis_report.json")
```

## Thesis experiment: Trace vs Trace+Provenance vs AXIS

One factual run → three explanation conditions per final-quiz MCQ:

```bash
python run_explanation_experiment_141.py
```

Or programmatically:

```python
experiment = bb.generate_explanation_experiment(
    question="Why did the system produce this decision?",
    experiment_seed=42,
    include_historical_versions=False,
)
```

Exports: research JSON/Markdown, blinded evaluator Markdown, and condition key.
See `mas_langgraph/experiment/`.

Low-level APIs (`WorkflowExecutor`, `LangGraphReplayExecutor`,
`AxisExplanationController`) remain available for research use.

## Orchestrators

| Mode | Module | Notes |
|------|--------|-------|
| `classic` | `orchestrator/base_workflow.py` | Rule loop (legacy compatibility; not AXIS-ready default) |
| `langgraph` | `orchestrator/langgraph_workflow.py` | Deterministic StateGraph with checkpoints + AXIS |

AXIS checkpoint replay requires `orchestrator_mode="langgraph"` and a completed
factual run.

## Compatibility note

Legacy MCQ payload classes and live node bodies still reside in `domains/quiz/`
and `orchestrator/langgraph_workflow.py`. Generalization is at the case/framework
interface — see `docs/architecture.md`.
