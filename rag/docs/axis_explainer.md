# AXIS Explainer

Operational procedure for grounded explanations over a completed factual
LangGraph run. Prefer the public service over manual wiring.

## Public API (recommended)

```python
from mas_langgraph import CooperativeBackbone
from mas_langgraph.artifacts import ArtifactRef
from mas_langgraph.axis.export import dump_axis_explanation

bb = CooperativeBackbone(orchestrator_mode="langgraph")
summary = bb.run_full_workflow()

refs = bb.list_explainable_artifacts()
explanation = bb.explain_artifact(
    refs[0],
    "Why did the system produce this decision?",
    scripted_queries=[
        "WHAT(candidate)",
        "DONE",
    ],
)

dump_axis_explanation(explanation, "axis_report.json")
```

Adaptive interrogation (explicit LLM callable only — never auto-wired):

```python
explanation = bb.explain_artifact(
    ref,
    question,
    llm_generate=my_llm_callable,
    max_rounds=5,
    axis_interrogation_mode="constrained",  # default; "hybrid" composes from vocabulary
)
```

Illustrative counterfactual (valid aliases/boundaries are case-specific):

```text
REMOVE(E17, before verify_candidate_mcqs_against_evidence#1)
```

Prefer boundaries exposed in the factual context rather than hard-coded aliases.

## Formal objects

Factual explanation context for exact case \(\zeta\):

\[
C_{\mathrm{fact}}^\zeta =
(
\tau_\zeta,
A_\zeta^{wf},
D_{\mathrm{run},\zeta},
\Gamma_\zeta,
y_\zeta
)
\]

Final explanation:

\[
e = E_{LG}
(
q_u,
\tau,
\{\tau^{\iota_1},\ldots,\tau^{\iota_m}\},
D
).
\]

`AxisExplanationController` implements the interrogation loop; it is not the
mathematical function itself.

## Loop components

| Concept | Implementation |
|---------|----------------|
| Explained artifact \(\zeta\) | `ExplainedCase` via `ExplainedCaseResolver` |
| Factual slice | `CaseSlice` / `ProvenanceSlicer.slice_run_result` |
| Factual context \(C_{\mathrm{fact}}\) | verbalizer + prompt templates |
| Query \(p_n\) | `AxisQuery` / parser |
| Resolved query \(p_n^{res}\) | `ResolvedQuery` |
| WHAT result \(\rho_n\) | `FactualLookupResult` |
| Query record \(r_n\) | `FactualQueryRecord` / `InterventionQueryRecord` |
| History \(H_N\) | `AxisEpisodicHistory` |
| No-op cache | `NoOpReplayCache` |
| Comparison | `ReplayComparison` |
| Export | `AxisExplanationReport` (`schema_version=1`) |

## Query semantics

| User query | Backend |
|------------|---------|
| `WHAT(target)` | Factual lookup only — no replay branch |
| `REMOVE(target, boundary)` | NOOP + REMOVE sibling forks |
| `ADD(target, boundary)` | NOOP + ADD |
| `WHATIF(old, new, boundary)` | User-level WHATIF; backend **REPLACE** |
| `DONE` | Terminate and synthesize |

## Checkpoints vs occurrences

\[
\omega \neq c
\]

Example: `before verify...#2` is a semantic boundary that resolves to a real
checkpoint; it is not itself a checkpoint. Boundary selectors use
`occurrence_index`.

## Provenance vs causality

\[
\boxed{\text{logged dependency} \neq \text{demonstrated causal effect}}
\]

\[
\boxed{\text{provenance relevance} \neq \text{causal relevance}}
\]

Evidence statuses (exported, conservative):

`decision_effect`, `criterion_effect`, `downstream_effect`,
`no_demonstrated_effect`, `replay_unstable`, `noop_error`,
`intervention_error`, `outcome_unavailable`.

No causal probability / strength / “true cause” claims are produced.

## Low-level APIs

Researchers may still use `WorkflowExecutor`, `LangGraphReplayExecutor`, and
`AxisExplanationController` directly. `AxisExplanationService` only coordinates
construction and validation (factual branch, run/index identity, case capabilities).

Classic orchestrator mode cannot run checkpoint-based AXIS explanations.

Prompts: `cases/mcq/inputs/prompts/`.
