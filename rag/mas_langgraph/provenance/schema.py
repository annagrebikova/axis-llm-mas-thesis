"""Load provenance schema from declarative YAML/dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

from .models import ProvenanceSchema, SchemaRelation


def load_provenance_schema(
    source: Union[str, Path, Mapping[str, Any], None] = None,
) -> ProvenanceSchema:
    """Load a :class:`ProvenanceSchema` from a path, mapping, or empty default."""
    if source is None:
        return ProvenanceSchema()
    if isinstance(source, Mapping):
        return _from_mapping(source)
    from ..core.input_loader import load_mapping

    return _from_mapping(load_mapping(source))


def _from_mapping(data: Mapping[str, Any]) -> ProvenanceSchema:
    relations = []
    for item in data.get("relations", []) or []:
        relations.append(
            SchemaRelation(
                producer=str(item["producer"]),
                artifact_type=str(item["artifact_type"]),
                relation=str(item["relation"]),
                consumer=str(item["consumer"]),
            )
        )
    return ProvenanceSchema(relations=relations)
