"""Intervention contracts and typed intervention payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
from uuid import uuid4

from ..artifacts.models import ArtifactRef
from .checkpoints import CheckpointRecord


@dataclass(frozen=True)
class ArtifactIntervention:
    """Global intervention ι = (op, α, c) from a selected checkpoint onward."""

    operation: str  # REMOVE | ADD | REPLACE | NOOP
    artifact: Union[ArtifactRef, tuple]
    checkpoint: CheckpointRecord
    intervention_id: str = field(default_factory=lambda: f"iv_{uuid4().hex[:10]}")
    description: str = ""

    def __post_init__(self) -> None:
        op = self.operation.upper()
        object.__setattr__(self, "operation", op)
        if op not in {"REMOVE", "ADD", "REPLACE", "NOOP"}:
            raise ValueError(f"Unsupported operation: {self.operation}")
        if op == "REPLACE":
            if not isinstance(self.artifact, tuple) or len(self.artifact) != 2:
                raise ValueError("REPLACE requires (old_ref, new_ref)")
        elif op in {"REMOVE", "ADD"}:
            if not isinstance(self.artifact, ArtifactRef):
                raise ValueError(f"{op} requires a single ArtifactRef")


@dataclass(frozen=True)
class InterventionContract:
    """Declarative permissions for intervening on an artifact type."""

    artifact_type: str
    allowed_operations: tuple
    replacement_compatible_types: tuple = ()
    allow_external_add: bool = False
    required_state_channels: tuple = ()
    description: str = ""

    def allows(self, operation: str) -> bool:
        return operation.upper() in {o.upper() for o in self.allowed_operations}

    def replacement_ok(self, new_type: str) -> bool:
        if not self.replacement_compatible_types:
            return new_type == self.artifact_type
        return new_type in self.replacement_compatible_types


@dataclass
class InterventionContractRegistry:
    """Case-loaded intervention contracts."""

    contracts: Dict[str, InterventionContract] = field(default_factory=dict)

    def register(self, contract: InterventionContract) -> None:
        self.contracts[contract.artifact_type] = contract

    def get(self, artifact_type: str) -> Optional[InterventionContract]:
        return self.contracts.get(artifact_type)

    def require(self, artifact_type: str) -> InterventionContract:
        c = self.get(artifact_type)
        if c is None:
            raise KeyError(f"No intervention contract for {artifact_type!r}")
        return c

    def validate(self, intervention: ArtifactIntervention, artifact_type: str) -> None:
        """Raise ValueError if the intervention violates the contract."""
        if intervention.operation == "NOOP":
            return
        contract = self.require(artifact_type)
        if not contract.allows(intervention.operation):
            raise ValueError(
                f"Operation {intervention.operation} not allowed for {artifact_type}"
            )
        if intervention.operation == "REPLACE":
            # new type checked by caller with replacement_ok
            pass
        if intervention.operation == "ADD" and not contract.allow_external_add:
            # still allow adds from factual catalog; caller enforces source
            pass


def load_intervention_contracts(
    source: Union[str, Path, Mapping[str, Any], None] = None,
) -> InterventionContractRegistry:
    """Load contracts from YAML/JSON mapping or path."""
    registry = InterventionContractRegistry()
    if source is None:
        return registry
    if isinstance(source, Mapping):
        data = source
    else:
        from ..core.input_loader import load_mapping

        data = load_mapping(source)
    for item in data.get("contracts", []) or []:
        registry.register(
            InterventionContract(
                artifact_type=str(item["artifact_type"]),
                allowed_operations=tuple(item.get("allowed_operations", [])),
                replacement_compatible_types=tuple(
                    item.get("replacement_compatible_types", [])
                ),
                allow_external_add=bool(item.get("allow_external_add", False)),
                required_state_channels=tuple(
                    item.get("required_state_channels", [])
                ),
                description=str(item.get("description", "")),
            )
        )
    return registry
