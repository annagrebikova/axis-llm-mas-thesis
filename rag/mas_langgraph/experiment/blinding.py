"""Deterministic blinded condition labels for human evaluation."""

from __future__ import annotations

import hashlib
from typing import Dict, Mapping

from ..artifacts.models import ArtifactRef
from .models import ConditionName

_LABELS = ("A", "B", "C")
_CONDITIONS: tuple[ConditionName, ...] = ("trace", "trace_provenance", "axis")


def _seed_bytes(run_id: str, candidate_ref: ArtifactRef, experiment_seed: int) -> bytes:
    material = f"{run_id}|{candidate_ref}|{int(experiment_seed)}"
    return hashlib.sha256(material.encode("utf-8")).digest()


def blinding_mapping(
    *,
    run_id: str,
    candidate_ref: ArtifactRef,
    experiment_seed: int = 42,
) -> Dict[str, ConditionName]:
    """Return ``{\"A\": condition, \"B\": ..., \"C\": ...}`` reproducibly."""
    digest = _seed_bytes(run_id, candidate_ref, experiment_seed)
    # Fisher–Yates using digest bytes
    conditions = list(_CONDITIONS)
    for i in range(len(conditions) - 1, 0, -1):
        j = digest[i] % (i + 1)
        conditions[i], conditions[j] = conditions[j], conditions[i]
    return {label: conditions[idx] for idx, label in enumerate(_LABELS)}


def invert_blinding(mapping: Mapping[str, ConditionName]) -> Dict[ConditionName, str]:
    return {cond: label for label, cond in mapping.items()}
