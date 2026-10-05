# Artifacts and Versions

Immutable identity:

- `ArtifactRef(logical_id, version)` — stable across revisions (`Q12:v1` ≠ `Q12:v2`)
- `ArtifactEnvelope` — frozen wrapper around case payloads
- `ArtifactRegistry` — historical catalog (REMOVE does not delete)
- `ArtifactResolver` — availability, blocks, substitutions

MCQ adapters in `cases/mcq/artifacts.py` wrap EvidenceChunk, CandidateMCQ, reports, etc.

`revise_candidate_version` creates a new envelope + `DerivedFrom` link without mutating the old payload.
