# Provenance

Occurrence records distinguish node identity from execution using **actual LangGraph vertex IDs**:

- `verify_candidate_mcqs_against_evidence#1`
- `verify_candidate_mcqs_against_evidence#2`

Occurrence IDs are deterministic: `{run_id}:{branch_id}:{node_id}#{occurrence_index}`.

- `execution_index` — global chronological position
- `occurrence_index` — nth execution of this node (used by `BoundarySelector`)

Live nodes are wrapped by `provenance.instrumentation.instrument_node` (wired in
`core/graph_builder.py`). The instrumentation adapter is case-provided; generic
code does not import MCQ. Checkpointed channels hold `occurrence_records`,
`produced_by`, `run_provenance_relations` (\(D_{\mathrm{run}}\)), and violations.

## Schema and run relations

- \(D_{\mathrm{schema}}\) → `ProvenanceSchema` (`cases/mcq/inputs/provenance_schema.yaml`)
- \(D_{\mathrm{run}}^\tau\) → `RunProvenanceRelation`

MCQ input bindings are resolved in `cases/mcq/provenance.py` from structured domain
fields (blueprint IDs, evidence chunk IDs, version-matched reports).

Runtime relations reconstruct \(D_{\mathrm{run}}\) from `ProducedBy` + consumer
bindings. Schema violations and unresolved producers are retained and reported.

## Provenance slicing

AXIS builds an output-specific `CaseSlice` for one exact explained artifact
(e.g. `CandidateMCQ` `Q12:v2`) including:

- relevant artifact refs and occurrences
- relevant \(D_{\mathrm{run}}\) relations
- criteria / decision when present
- revision lineage (`DerivedFrom`) when historical versions participate

MCQ scoping filters batch-node bindings so one candidate does not inherit
unrelated sibling outputs.

## Intervention-supplied artifacts

ADD/REPLACE inject envelopes through `InterventionEvent` metadata rather than
fabricating producer `ProducedBy` edges. DerivedFrom remains reserved for
revision lineage.

## Outcome criteria vs provenance

Provenance records *what was consumed/produced*. Outcome assessment records
*why the gate decided*:

\[
y_\alpha = \delta_\alpha(\Gamma_\alpha(x))
\]

`OutcomeAssessment` (`assessment:Q12:vN`) and `SupervisorDecision`
(`decision:Q12:vN`) are both produced in the same `apply_acceptance_gate#n`
occurrence. The decision links to its assessment via
`SupervisorDecision.assessment_ref` — there is no fabricated assessment→decision
\(D_{\mathrm{run}}\) edge.

## Causal distinction

\[
\boxed{\text{logged dependency} \neq \text{demonstrated causal effect}}
\]

\[
\boxed{\text{provenance relevance} \neq \text{causal relevance}}
\]

Provenance relevance selects candidates for interrogation. Demonstrated
intervention effects require controlled no-op / counterfactual comparison.
