# Thesis notation ↔ implementation mapping

This document maps formal thesis symbols to the code under `rag/mas_langgraph/`.

## Workflow

\[
\mathcal{W}_{LG} = (G, X, \mathcal{A}^{wf}, D, \delta)
\]

| Symbol | Implementation |
|--------|----------------|
| \(G\) | Compiled LangGraph from `core/graph_builder.py` / `CaseDefinition.build_graph` |
| \(X\) | `WorkflowGraphState` plus runtime-managed execution context. `WorkflowRuntime` is external services (runner, RAG, LLM), **not** the entire mathematical \(X\). |
| \(\mathcal{A}^{wf}\) | `ArtifactEnvelope` / `ArtifactRef` history in `artifact_catalog` |
| \(D\) | `ProvenanceSchema` + `RunProvenanceRelation` |
| \(\delta\) | MCQ acceptance decision policy (`decide_acceptance` / `SupervisorDecision`) |

## Factual trace

\[
\tau = (x_0, \Omega_1, x_1, \ldots, \Omega_S, x_S)
\]

| Symbol | Implementation |
|--------|----------------|
| \(x_s\) | Persisted LangGraph checkpoint state |
| \(\Omega_s\) | Occurrence records mapped to that super-step |
| \(c_s\) | `CheckpointRecord` |

Returned as `WorkflowRunResult` with frozen `CheckpointIndex`.

## Occurrence

\[
\mathrm{Occ}(\omega) = (v, B, O, \Delta)
\]

| Component | Field |
|-----------|-------|
| \(v\) | `OccurrenceRecord.node_id` |
| \(B\) | `input_bindings` |
| \(O\) | `produced_artifacts` |
| \(\Delta\) | `state_delta` |

## Provenance

| Symbol | Implementation |
|--------|----------------|
| \(D_{\mathrm{schema}}\) | `ProvenanceSchema` |
| \(D_{\mathrm{run}}^\tau\) | `RunProvenanceRelation` |

## Outcome

| Symbol | Implementation |
|--------|----------------|
| \(\Gamma_\alpha\) | `OutcomeAssessment.criteria` |
| \(\delta_\alpha\) | `decide_acceptance` |
| \(y_\alpha\) | `SupervisorDecision.decision` |

## Interventions and replay

\[
\iota = (op, \alpha, c)
\]

| Symbol | Implementation |
|--------|----------------|
| \(op\) | NOOP / REMOVE / ADD / REPLACE (`ArtifactIntervention`) |
| \(\alpha\) | Exact `ArtifactRef` (REPLACE uses `(old_ref, new_ref)`) |
| \(c\) | `CheckpointRecord` |
| \(\Pi_{LG}\) | `LangGraphReplayExecutor` (load checkpoint → `update_state` fork → `invoke(None)`) |
| \(\tau^\iota\) | Counterfactual / no-op `WorkflowRunResult` |

## Explanation

| Symbol | Implementation |
|--------|----------------|
| \(\zeta\) | `ExplainedCase` |
| \(C_{\mathrm{fact}}^\zeta\) | `CaseSlice` + criteria + decision |
| \(p_n\) | `AxisQuery` |
| \(p_n^{res}\) | `ResolvedQuery` |
| \(\rho_n\) (WHAT) | `FactualLookupResult` |
| \(r_n\) | `FactualQueryRecord` / `InterventionQueryRecord` |
| \(H_N\) | `AxisEpisodicHistory` |
| \(E_{LG}\) | `AxisExplanationController` (+ verbalizer / prompts / comparator) |

## Terminology

| Term | Meaning |
|------|---------|
| node | Graph vertex id (e.g. `apply_acceptance_gate`) |
| occurrence | One execution of a node (`node#n`) |
| checkpoint | Persisted LangGraph state snapshot (`c_k`) |
| artifact | Immutable versioned envelope (`logical_id:vN`) |
| artifact version | Integer version on `ArtifactRef` |
| run | Factual execution identified by `run_id` / `thread_id` |
| branch | Logical factual / noop / counterfactual lineage |
| decision | Terminal gate outcome for one candidate version |
| assessment | Criterion vector \(\Gamma\) for one candidate version |
| provenance relation | Logged producer/consumer dependency |
| intervention event | ADD/REPLACE supply metadata (not fake ProducedBy) |
| no-op | Control replay from the same origin checkpoint |
| counterfactual branch | Sibling replay under an intervention |

## Methodological limitations

1. Deterministic orchestration does not make LLM outputs deterministic.
2. One no-op sample per checkpoint is currently used.
3. Causal evidence is local to tested interventions/checkpoints.
4. Intervention universe is constrained by contracts.
5. Provenance only covers explicitly instrumented artifact relations.
6. Some legacy MCQ domain types remain outside `cases/mcq`.
7. `cases/example_case` demonstrates configuration but has no executable workflow.

These are scope limitations, not unintended defects.
