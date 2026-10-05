"""Generic immutable artifact identity, registry, and resolution."""

from .channel_state import (
    clone_artifact_channels_dict,
    dump_artifact_channels,
    empty_artifact_channels,
    ensure_artifact_channels,
    load_artifact_channels,
)
from .models import ArtifactEnvelope, ArtifactRef, DerivedFrom, ProducedBy
from .registry import ArtifactRegistry, ArtifactTypeRegistry
from .resolver import ArtifactResolver
from .utils import make_instance_id, payloads_equivalent

__all__ = [
    "ArtifactEnvelope",
    "ArtifactRef",
    "ArtifactRegistry",
    "ArtifactResolver",
    "ArtifactTypeRegistry",
    "DerivedFrom",
    "ProducedBy",
    "clone_artifact_channels_dict",
    "dump_artifact_channels",
    "empty_artifact_channels",
    "ensure_artifact_channels",
    "load_artifact_channels",
    "make_instance_id",
    "payloads_equivalent",
]
