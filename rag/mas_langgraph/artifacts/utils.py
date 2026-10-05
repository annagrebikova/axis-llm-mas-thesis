"""Generic artifact helpers — no case-specific imports."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def payloads_equivalent(left: Any, right: Any) -> bool:
    """Return True when two artifact payloads represent the same immutable value."""
    if is_dataclass(left) and is_dataclass(right):
        return asdict(left) == asdict(right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left == right
    return left == right


def make_instance_id(
    *,
    created_in_run: str,
    branch_id: str,
    artifact_type: str,
    logical_id: str,
    version: int,
) -> str:
    """Deterministic concrete instance identity within one execution branch."""
    return f"{created_in_run}:{branch_id}:{artifact_type}:{logical_id}:v{version}"
