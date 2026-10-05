# Architecture

Final architecture of the generalized LangGraph MAS framework used for thesis
experiments. This document describes the **implemented** system.

## Formal workflow

\[
\mathcal{W}_{LG} = (G, X, \mathcal{A}^{wf}, D, \delta)
\]

Factual execution produces a trace

\[
\tau = (x_0, \Omega_1, x_1, \ldots, \Omega_S, x_S)
\]

with immutable artifact history, occurrence-level provenance, outcome
assessments, decisions, and a frozen checkpoint index. AXIS explanations then
operate on an exact explained artifact \(\zeta\) without mutating \(\tau\).

## Authoritative layers

### 1. Generic workflow framework

| Package | Role |
|---------|------|
| `core/` | Protocols, `WorkflowRuntime`, `WorkflowExecutor`, graph builder, case configuration |
| `artifacts/` | Immutable `ArtifactRef` / envelopes / registry / resolver |
| `provenance/` | Occurrence recording, schema, run relations |
| `replay/` | Checkpoint index, branch store, interventions, `LangGraphReplayExecutor` |
| `axis/` | Slicing, queries, comparison, controller, **public** `AxisExplanationService`, export |

Generic packages must not import MCQ payload types (`CandidateMCQ`, etc.).

### 2. MCQ case

`cases/mcq/` provides declarative YAML metadata, adapters, verbalizer, explained-case
resolver, CooperativeState, and case definition. Executable node functions remain in
`orchestrator/langgraph_workflow.py` and are wired through `core/graph_builder.py`.

### 3. Factual execution

```text
CooperativeBackbone(orchestrator_mode="langgraph")
  → LangGraphWorkflowOrchestrator
  → WorkflowExecutor
  → compiled MCQ StateGraph + MemorySaver
  → WorkflowRunResult τ
```

Runtime services live in `WorkflowRuntime` (LangGraph `context=`), not in
checkpointed state. `thread_id == run_id`.

Public accessors after a factual run:

- `orchestrator.get_factual_run_result()`
- `orchestrator.replay_executor` / `get_checkpoint_index()`
- `backbone.factual_run_result()` / `list_explainable_artifacts()` / `explain_artifact(...)`

### 4. Artifact model

Immutable `ArtifactEnvelope` history in `artifact_catalog`. Exact versions such as
`Q12:v1` and `Q12:v2` remain distinguishable. Operational dicts hold latest pointers only.

### 5. Provenance

\(D_{\mathrm{schema}}\) → `ProvenanceSchema`  
\(D_{\mathrm{run}}^\tau\) → `RunProvenanceRelation`  
Occurrences use node-local `occurrence_index` and global `execution_index`.

### 6. Outcome assessment

\(\Gamma_\alpha\) → `OutcomeAssessment.criteria`  
\(\delta_\alpha\) → `decide_acceptance` / `SupervisorDecision`

### 7. Checkpoints / replay

Frozen `CheckpointIndex` over real LangGraph checkpoints. Interventions:

- `NOOP` — control replay from the same origin
- `REMOVE` — block availability
- `ADD` — controlled availability injection via `InterventionEvent`
- `REPLACE` — substitution old → new via `InterventionEvent`

### 8. AXIS interrogation

```text
WorkflowRunResult
  → ExplainedCaseResolver → ExplainedCase
  → CaseSlice / C_fact
  → AxisExplanationController
       WHAT | REMOVE | ADD | WHATIF | DONE
  → ReplayComparison(s)
  → ExplanationResult
  → AxisExplanationReport (JSON)
```

### 9. Case configuration

Strict AXIS cases load declarative `case.yaml`, `workflow.yaml`, artifact types,
contracts, criteria, provenance schema, and prompt templates. Required capability
bundle includes verbalizer, slice/observation/provenance adapters, and
`ExplainedCaseResolver`.

### 10. Compatibility layers

Historical MCQ implementation still exists in:

- `core/entities.py` (shared entity dataclasses)
- `orchestrator/langgraph_workflow.py` (live node bodies)
- `domains/quiz/` (generation, verification, acceptance, packaging)

These are called through case adapters. **Generalization is achieved at the
framework/case interface, not by physically relocating every legacy payload class.**

Compatibility-unwired graph nodes (`retrieve_evidence`, `verify_candidates`,
`package_or_end`, `end_with_error`) remain registered for topology synchronization
with `workflow.yaml` (22 nodes: 18 active/reachable + 4 compatibility_unwired).

## End-to-end path

```text
INPUT MATERIAL
      │
      ▼
MCQ LangGraph factual execution
      │
      ▼
WorkflowRunResult τ
      ├── checkpoint history Cτ
      ├── artifacts A^wf
      ├── provenance D_run
      ├── assessments Γ
      └── decisions y
      │
      ▼
select exact explained artifact ζ
      │
      ▼
AxisExplanationService
      │
      ▼
grounded ExplanationResult → AxisExplanationReport / JSON
```

## Conceptual invariants

\[
\boxed{\text{provenance relevance} \neq \text{causal relevance}}
\]

\[
\boxed{\text{factual dependency} \neq \text{demonstrated intervention effect}}
\]

A local statement such as “Removing E17 at checkpoint c_8 changed reject → accept
while the no-op remained reject” is supported as a **local intervention result**.
The framework does not infer universal causation.

## Historical migration notes

Prompts 2–15 incrementally introduced artifacts, provenance, checkpoints, replay,
slicing, queries, comparison, and case configuration. That migration is complete
for thesis purposes. Prefer this document over older prompt-numbered status text.

## See also

- `docs/axis_explainer.md` — operational AXIS loop
- `docs/checkpoint_replay.md` — NOOP / REMOVE / ADD / REPLACE
- `docs/provenance.md` — slicing and intervention boundaries
- `docs/thesis_mapping.md` — notation ↔ code
- `docs/reproducibility.md` — how to run tests and smokes
