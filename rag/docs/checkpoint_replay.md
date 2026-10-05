# Checkpoint Replay

`CheckpointIndex` builds run-local aliases `c_0`, `c_1`, … over **real** LangGraph
checkpoints (oldest-first). `CheckpointRecord.superstep` comes from
`snapshot.metadata["step"]` — never from the alias index.

Occurrence membership for each checkpoint is the **delta** of checkpointed
`occurrence_records` versus its parent. Canonical IDs live in
`after_occurrences`; readable `node#n` labels live in `after_occurrence_labels`.

## Boundaries vs checkpoints

Semantic intervention boundaries such as

```text
before verify_candidate_mcqs_against_evidence#2
```

are **not** checkpoints themselves. They resolve through `BoundarySelector` to a
concrete `CheckpointRecord` with a real LangGraph checkpoint UUID.

`BoundarySelector.before_occurrence` / `after_occurrence` take the **node-local**
`occurrence_index` (nth execution of that node), not the global `execution_index`.

- `execution_index` — global chronological occurrence position in the branch
- `occurrence_index` — nth execution of this node inside the branch

## Supported interventions

All of the following are real LangGraph fork-and-replay operations:

| Operation | Meaning |
|-----------|---------|
| `NOOP` | Replay from the origin checkpoint with no artifact change (control) |
| `REMOVE` | Block artifact availability from the origin onward |
| `ADD` | Controlled availability injection of a supplied artifact |
| `REPLACE` | Substitution of an available artifact old → new |

```text
factual checkpoint c_s  (thread_id == run_id)
        │
        ├── update_state(branch metadata)              → NOOP fork → invoke(None)
        ├── update_state(metadata + blocked refs)      → REMOVE CF
        ├── update_state(metadata + supplied envs)     → ADD CF
        └── update_state(metadata + substitutions)     → REPLACE CF
```

- Logical `branch_id` ≠ LangGraph `thread_id` (thread stays the factual `run_id`).
- `BranchStore` is scoped by `(run_id, branch_id)`.
- Authoritative REMOVE uses `blocked_artifact_refs` in checkpointed state.
- ADD/REPLACE supplied artifacts are represented via `InterventionEvent`, not fake
  `ProducedBy` edges.
- Live nodes see resolver-filtered `CooperativeState` via
  `cases/mcq/intervention_view.py` **before** provenance input bindings.
- Factual `CheckpointIndex` is frozen at end of factual execution and is not
  rebuilt after forks.

## No-op control

Every intervention comparison uses a sibling no-op from the **same** factual
origin checkpoint:

\[
\Delta_{\mathrm{noop}} = \mathrm{Diff}(O_f, O_{\mathrm{noop}})
\]

\[
\Delta_\iota = \mathrm{Diff}(O_{\mathrm{noop}}, O_{\mathrm{cf}})
\]

If the no-op decision drifts from the factual decision, evidence status is
`replay_unstable`: counterfactual differences cannot be attributed to the tested
artifact from a single control sample.

Deterministic graph routing does **not** guarantee deterministic external
LLM/API outputs; the no-op exists partly for that reason. The current
implementation uses one no-op sample per checkpoint.

Intervention contracts: `cases/mcq/inputs/intervention_contracts.yaml`.
