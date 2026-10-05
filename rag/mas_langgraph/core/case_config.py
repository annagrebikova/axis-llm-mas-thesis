"""Declarative case configuration models, loading, and validation.

``workflow.yaml`` describes the workflow for validation, explanation,
documentation, and developer inspection. The executable graph topology
remains Python/LangGraph code and is not constructed from this YAML.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from ..artifacts.channel_state import empty_artifact_channels
from ..artifacts.registry import ArtifactTypeDecl, ArtifactTypeRegistry
from ..core.input_loader import load_mapping
from ..provenance.schema import load_provenance_schema
from ..replay.interventions import load_intervention_contracts


class CaseConfigurationError(ValueError):
    """Declarative case configuration is invalid."""


ALLOWED_NODE_TYPES = frozenset({"agent", "computational"})
ALLOWED_CRITERION_VALUE_TYPES = frozenset(
    {"bool", "int", "float", "str", "tuple[str, ...]"}
)
KNOWN_ARTIFACT_CHANNELS = frozenset(empty_artifact_channels().keys())

REQUIRED_CASE_FILES = (
    "case.yaml",
    "workflow.yaml",
    "artifact_types.yaml",
    "provenance_schema.yaml",
    "outcome_criteria.yaml",
    "intervention_contracts.yaml",
)

REQUIRED_AXIS_PROMPTS = (
    "system.txt",
    "factual_context.txt",
    "interrogation.txt",
    "intermediate_explanation.txt",
    "final_explanation.txt",
)


# ---------------------------------------------------------------------------
# Metadata models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseMetadata:
    case_id: str
    case_version: int
    python_adapter: str
    description: str = ""


@dataclass(frozen=True)
class WorkflowNodeMetadata:
    node_id: str
    label: str
    node_type: str
    purpose: str
    input_artifact_types: Tuple[str, ...] = ()
    output_artifact_types: Tuple[str, ...] = ()
    status: str = "active"


@dataclass(frozen=True)
class WorkflowRoutingMetadata:
    source_node: str
    description: str = ""


@dataclass(frozen=True)
class WorkflowMetadata:
    """Descriptive workflow metadata — not executable topology."""

    nodes: Tuple[WorkflowNodeMetadata, ...]
    routing: Tuple[WorkflowRoutingMetadata, ...] = ()

    def node_ids(self) -> Set[str]:
        return {n.node_id for n in self.nodes}

    def get(self, node_id: str) -> Optional[WorkflowNodeMetadata]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None


@dataclass(frozen=True)
class CaseConfiguration:
    metadata: CaseMetadata
    workflow: WorkflowMetadata
    artifact_types: ArtifactTypeRegistry
    provenance_schema: Any
    outcome_evaluator: Any
    intervention_contracts: Any
    outcome_criteria: Tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class AxisCaseCapabilities:
    case_configuration: CaseConfiguration
    verbalizer: Any
    prompt_templates: Any
    slicing_adapter: Any
    observation_adapter: Any
    provenance_instrumentation_adapter: Any = None
    explained_case_resolver: Any = None


# ---------------------------------------------------------------------------
# Dotted-path loading
# ---------------------------------------------------------------------------

def resolve_dotted_path(path: str, *, source: str = "") -> Any:
    """Import ``module.path.Name`` without evaluating YAML expressions."""
    text = str(path or "").strip()
    if not text or "." not in text:
        raise CaseConfigurationError(
            f"{source or 'dotted path'}: expected 'module.path.Name', got {path!r}"
        )
    module_name, _, attr = text.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise CaseConfigurationError(
            f"{source or 'dotted path'}: cannot import {module_name!r} ({exc})"
        ) from exc
    if not hasattr(module, attr):
        raise CaseConfigurationError(
            f"{source or 'dotted path'}: {module_name!r} has no attribute {attr!r}"
        )
    return getattr(module, attr)


def require_callable_path(path: str, *, source: str) -> Any:
    obj = resolve_dotted_path(path, source=source)
    if not callable(obj):
        raise CaseConfigurationError(
            f"{source}: {path!r} resolved but is not callable"
        )
    return obj


# ---------------------------------------------------------------------------
# Graph introspection
# ---------------------------------------------------------------------------

_TECHNICAL_NODES = frozenset(
    {"START", "END", "__start__", "__end__", "__START__", "__END__"}
)


def _node_id(node: Any) -> str:
    if isinstance(node, str):
        return node
    for attr in ("id", "name"):
        val = getattr(node, attr, None)
        if val:
            return str(val)
    return str(node)


def extract_graph_node_ids(compiled_graph: Any) -> Set[str]:
    """Registered workflow node IDs, excluding START/END."""
    raw: Iterable[Any]
    if compiled_graph is None:
        return set()
    if hasattr(compiled_graph, "nodes") and isinstance(compiled_graph.nodes, Mapping):
        raw = compiled_graph.nodes.keys()
    elif hasattr(compiled_graph, "get_graph"):
        g = compiled_graph.get_graph()
        nodes = getattr(g, "nodes", {})
        raw = nodes.keys() if isinstance(nodes, Mapping) else nodes
    else:
        raise CaseConfigurationError(
            "Cannot introspect compiled graph node IDs"
        )
    return {nid for nid in (_node_id(n) for n in raw) if nid not in _TECHNICAL_NODES}


def extract_reachable_node_ids(compiled_graph: Any) -> Set[str]:
    """Node IDs reachable from START via compiled-graph edges."""
    if compiled_graph is None:
        return set()
    if not hasattr(compiled_graph, "get_graph"):
        return extract_graph_node_ids(compiled_graph)
    g = compiled_graph.get_graph()
    nodes = getattr(g, "nodes", {})
    node_ids = {
        _node_id(n)
        for n in (nodes.keys() if isinstance(nodes, Mapping) else nodes)
    }
    edges = list(getattr(g, "edges", []) or [])
    adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
    for edge in edges:
        src = _node_id(getattr(edge, "source", None) or getattr(edge, "src", ""))
        dst = _node_id(getattr(edge, "target", None) or getattr(edge, "dst", ""))
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set())
    starts = [n for n in adj if n in _TECHNICAL_NODES and "start" in n.lower()]
    if not starts:
        starts = [n for n in adj if n in {"START", "__start__"}]
    seen: Set[str] = set()
    stack = list(starts)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                stack.append(nxt)
    return {n for n in seen if n not in _TECHNICAL_NODES}


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def case_inputs_dir(case_definition: Any) -> Path:
    if getattr(case_definition, "inputs_dir", None):
        return Path(case_definition.inputs_dir)
    if hasattr(case_definition, "prompt_dir"):
        return Path(case_definition.prompt_dir()).parent
    raise CaseConfigurationError(
        "Case definition has no inputs_dir / prompt_dir for declarative files"
    )


def require_case_files(
    inputs_dir: Path,
    *,
    require_prompts: bool = False,
) -> None:
    for name in REQUIRED_CASE_FILES:
        path = inputs_dir / name
        if not path.exists():
            raise CaseConfigurationError(
                f"Required case file missing: {path}"
            )
    if require_prompts:
        prompt_dir = inputs_dir / "prompts"
        for name in REQUIRED_AXIS_PROMPTS:
            path = prompt_dir / name
            if not path.exists():
                raise CaseConfigurationError(
                    f"Required AXIS prompt missing: {path}"
                )


def load_case_metadata(source: Any) -> CaseMetadata:
    data = load_mapping(source)
    try:
        case_id = str(data["case_id"])
        case_version = int(data["case_version"])
        python_adapter = str(data["python_adapter"])
    except KeyError as exc:
        raise CaseConfigurationError(
            f"case.yaml missing required field {exc}"
        ) from exc
    return CaseMetadata(
        case_id=case_id,
        case_version=case_version,
        python_adapter=python_adapter,
        description=str(data.get("description", "")),
    )


def load_workflow_metadata(source: Any) -> WorkflowMetadata:
    data = load_mapping(source)
    nodes: list[WorkflowNodeMetadata] = []
    seen: Set[str] = set()
    for item in data.get("nodes", []) or []:
        node_id = str(item.get("id") or item.get("node_id") or "")
        if not node_id:
            raise CaseConfigurationError("workflow.yaml: node missing id")
        if node_id in seen:
            raise CaseConfigurationError(
                f"workflow.yaml: duplicate node id {node_id!r}"
            )
        seen.add(node_id)
        node_type = str(item.get("type") or item.get("node_type") or "")
        if node_type not in ALLOWED_NODE_TYPES:
            raise CaseConfigurationError(
                f"workflow.yaml node {node_id!r}: unsupported node type {node_type!r} "
                f"(allowed: {sorted(ALLOWED_NODE_TYPES)})"
            )
        status = str(item.get("status") or "active")
        nodes.append(
            WorkflowNodeMetadata(
                node_id=node_id,
                label=str(item.get("label", node_id)),
                node_type=node_type,
                purpose=str(item.get("purpose", "")),
                input_artifact_types=tuple(item.get("inputs") or ()),
                output_artifact_types=tuple(item.get("outputs") or ()),
                status=status,
            )
        )
    routing: list[WorkflowRoutingMetadata] = []
    for item in data.get("routing", []) or []:
        routing.append(
            WorkflowRoutingMetadata(
                source_node=str(item.get("from") or item.get("source_node") or ""),
                description=str(item.get("description", "")),
            )
        )
    return WorkflowMetadata(nodes=tuple(nodes), routing=tuple(routing))


def load_artifact_types_file(source: Any) -> ArtifactTypeRegistry:
    data = load_mapping(source)
    registry = ArtifactTypeRegistry()
    for item in data.get("types", []) or []:
        name = str(item.get("name") or "")
        if not name:
            raise CaseConfigurationError("artifact_types.yaml: type missing name")
        registry.register(
            ArtifactTypeDecl(
                name=name,
                description=str(item.get("description", "")),
                explanation_target=bool(item.get("explanation_target", False)),
                intervenable=bool(item.get("intervenable", False)),
                serialization_adapter=item.get("serialization_adapter"),
            )
        )
    return registry


def load_outcome_criteria_file(source: Any) -> Tuple[Mapping[str, Any], ...]:
    data = load_mapping(source)
    items = tuple(data.get("criteria", []) or ())
    if not items:
        raise CaseConfigurationError(
            f"outcome_criteria.yaml has no criteria: {source}"
        )
    return tuple(items)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class CaseConfigurationValidator:
    def validate(
        self,
        *,
        case_definition: Any,
        configuration: CaseConfiguration,
        compiled_graph: Any = None,
        require_graph: bool = False,
        require_prompts: bool = False,
    ) -> None:
        self._validate_identity(case_definition, configuration.metadata)
        self._validate_python_adapter(case_definition, configuration.metadata)
        self._validate_artifact_types(configuration)
        self._validate_workflow_graph(configuration.workflow, compiled_graph, require_graph)
        self._validate_provenance(configuration)
        self._validate_contracts(configuration)
        self._validate_outcome_criteria(configuration)
        if require_prompts:
            require_case_files(case_inputs_dir(case_definition), require_prompts=True)

    def _validate_identity(self, case_definition: Any, metadata: CaseMetadata) -> None:
        case_id = getattr(case_definition, "case_id", None)
        case_version = getattr(case_definition, "case_version", None)
        if case_id is not None and str(case_id) != metadata.case_id:
            raise CaseConfigurationError(
                f"case.yaml case_id {metadata.case_id!r} does not match "
                f"Python case_id {case_id!r}"
            )
        if case_version is not None and int(case_version) != int(metadata.case_version):
            raise CaseConfigurationError(
                f"case.yaml case_version {metadata.case_version!r} does not match "
                f"Python case_version {case_version!r}"
            )

    def _validate_python_adapter(
        self, case_definition: Any, metadata: CaseMetadata
    ) -> None:
        adapter = resolve_dotted_path(
            metadata.python_adapter, source="case.yaml python_adapter"
        )
        if not isinstance(adapter, type) and not callable(adapter):
            raise CaseConfigurationError(
                f"case.yaml python_adapter {metadata.python_adapter!r} is not a class"
            )
        if isinstance(adapter, type) and not isinstance(case_definition, adapter):
            if type(case_definition) is not adapter:
                raise CaseConfigurationError(
                    f"case.yaml python_adapter {metadata.python_adapter!r} does not "
                    f"match active case {type(case_definition).__name__}"
                )

    def _validate_artifact_types(self, configuration: CaseConfiguration) -> None:
        types = configuration.artifact_types
        for decl_name in types.names():
            decl = types.require(decl_name)
            adapter = getattr(decl, "serialization_adapter", None)
            if adapter:
                require_callable_path(
                    str(adapter),
                    source=f"artifact_types.yaml type {decl_name!r} serialization_adapter",
                )
        referenced: Set[str] = set()
        for node in configuration.workflow.nodes:
            referenced.update(node.input_artifact_types)
            referenced.update(node.output_artifact_types)
        for rel in getattr(configuration.provenance_schema, "relations", ()) or ():
            referenced.add(rel.artifact_type)
        for name in sorted(referenced):
            if name not in types:
                raise CaseConfigurationError(
                    f"artifact type {name!r} is referenced by workflow/provenance "
                    "but missing from artifact_types.yaml"
                )

    def _validate_workflow_graph(
        self,
        workflow: WorkflowMetadata,
        compiled_graph: Any,
        require_graph: bool,
    ) -> None:
        if compiled_graph is None:
            if require_graph:
                raise CaseConfigurationError(
                    "workflow.yaml validation requires a compiled graph"
                )
            return
        actual = extract_graph_node_ids(compiled_graph)
        declared = workflow.node_ids()
        missing = actual - declared
        extra = declared - actual
        if missing:
            raise CaseConfigurationError(
                f"workflow.yaml missing graph node(s): {sorted(missing)}"
            )
        if extra:
            raise CaseConfigurationError(
                f"workflow.yaml declares node(s) not registered in graph: {sorted(extra)}"
            )
        reachable = extract_reachable_node_ids(compiled_graph)
        for node in workflow.nodes:
            if node.status == "active" and node.node_id not in reachable:
                raise CaseConfigurationError(
                    f"workflow.yaml node {node.node_id!r} is status=active but "
                    "not reachable from START"
                )
            if node.status == "compatibility_unwired" and node.node_id in reachable:
                raise CaseConfigurationError(
                    f"workflow.yaml node {node.node_id!r} is compatibility_unwired "
                    "but is reachable from START"
                )

    def _validate_provenance(self, configuration: CaseConfiguration) -> None:
        schema = configuration.provenance_schema
        relations = getattr(schema, "relations", None)
        if not relations:
            raise CaseConfigurationError("provenance_schema.yaml has no relations")
        node_ids = configuration.workflow.node_ids()
        types = configuration.artifact_types
        for i, rel in enumerate(relations):
            relation = getattr(rel, "relation", "")
            if not relation:
                raise CaseConfigurationError(
                    f"provenance_schema.yaml relation #{i} has empty relation name"
                )
            atype = getattr(rel, "artifact_type", "")
            if atype not in types:
                raise CaseConfigurationError(
                    f"provenance_schema.yaml relation #{i} references unknown "
                    f"artifact_type {atype!r}"
                )
            producer = getattr(rel, "producer", "")
            consumer = getattr(rel, "consumer", "")
            if producer not in node_ids:
                raise CaseConfigurationError(
                    f"provenance_schema.yaml relation #{i} producer {producer!r} "
                    "is not a workflow.yaml node"
                )
            if consumer not in node_ids:
                raise CaseConfigurationError(
                    f"provenance_schema.yaml relation #{i} consumer {consumer!r} "
                    "is not a workflow.yaml node"
                )

    def _validate_contracts(self, configuration: CaseConfiguration) -> None:
        contracts = configuration.intervention_contracts
        mapping = getattr(contracts, "contracts", {}) or {}
        types = configuration.artifact_types
        for artifact_type, contract in mapping.items():
            if artifact_type not in types:
                raise CaseConfigurationError(
                    f"intervention_contracts.yaml: unknown artifact_type {artifact_type!r}"
                )
            decl = types.require(artifact_type)
            ops = tuple(getattr(contract, "allowed_operations", ()) or ())
            if ops and not getattr(decl, "intervenable", False):
                raise CaseConfigurationError(
                    f"intervention_contracts.yaml: {artifact_type!r} has allowed "
                    f"operations {ops} but artifact_types.yaml intervenable=false"
                )
            if getattr(contract, "allow_external_add", False) and "ADD" not in {
                o.upper() for o in ops
            }:
                raise CaseConfigurationError(
                    f"intervention_contracts.yaml: {artifact_type!r} has "
                    "allow_external_add=true but ADD is not an allowed operation"
                )
            for repl in getattr(contract, "replacement_compatible_types", ()) or ():
                if repl not in types:
                    raise CaseConfigurationError(
                        f"intervention_contracts.yaml: {artifact_type!r} "
                        f"replacement_compatible_types includes unknown {repl!r}"
                    )
            for ch in getattr(contract, "required_state_channels", ()) or ():
                if ch not in KNOWN_ARTIFACT_CHANNELS:
                    raise CaseConfigurationError(
                        f"intervention_contracts.yaml: {artifact_type!r} "
                        f"required_state_channels includes unknown channel {ch!r}"
                    )

    def _validate_outcome_criteria(self, configuration: CaseConfiguration) -> None:
        seen: Set[str] = set()
        for item in configuration.outcome_criteria:
            name = str(item.get("name") or "")
            if not name:
                raise CaseConfigurationError("outcome_criteria.yaml: criterion missing name")
            if name in seen:
                raise CaseConfigurationError(
                    f"outcome_criteria.yaml: duplicate criterion name {name!r}"
                )
            seen.add(name)
            value_type = str(item.get("value_type") or "bool")
            if value_type not in ALLOWED_CRITERION_VALUE_TYPES:
                raise CaseConfigurationError(
                    f"outcome_criteria.yaml criterion {name!r}: unsupported "
                    f"value_type {value_type!r}"
                )
            implemented = item.get("implemented", True)
            decision_relevant = bool(item.get("decision_relevant", False))
            if implemented is False and decision_relevant:
                raise CaseConfigurationError(
                    f"outcome_criteria.yaml criterion {name!r}: implemented=false "
                    "cannot be decision_relevant=true"
                )


# ---------------------------------------------------------------------------
# Bundle loaders
# ---------------------------------------------------------------------------

def load_case_configuration(
    case_definition: Any,
    *,
    compiled_graph: Any = None,
    strict: bool = True,
    require_graph: bool = False,
    require_prompts: bool = False,
    outcome_evaluator: Any = None,
) -> CaseConfiguration:
    """Load and optionally validate declarative case configuration."""
    inputs = case_inputs_dir(case_definition)
    if strict:
        require_case_files(inputs, require_prompts=require_prompts)
    metadata = load_case_metadata(inputs / "case.yaml")
    workflow = load_workflow_metadata(inputs / "workflow.yaml")
    artifact_types = load_artifact_types_file(inputs / "artifact_types.yaml")
    provenance_schema = load_provenance_schema(load_mapping(inputs / "provenance_schema.yaml"))
    outcome_criteria = load_outcome_criteria_file(inputs / "outcome_criteria.yaml")
    contracts = load_intervention_contracts(load_mapping(inputs / "intervention_contracts.yaml"))
    if outcome_evaluator is None and hasattr(case_definition, "load_outcome_evaluator"):
        outcome_evaluator = case_definition.load_outcome_evaluator()
    configuration = CaseConfiguration(
        metadata=metadata,
        workflow=workflow,
        artifact_types=artifact_types,
        provenance_schema=provenance_schema,
        outcome_evaluator=outcome_evaluator,
        intervention_contracts=contracts,
        outcome_criteria=outcome_criteria,
    )
    if strict:
        CaseConfigurationValidator().validate(
            case_definition=case_definition,
            configuration=configuration,
            compiled_graph=compiled_graph,
            require_graph=require_graph,
            require_prompts=require_prompts,
        )
    return configuration


def load_axis_case_capabilities(
    case_definition: Any,
    *,
    compiled_graph: Any = None,
    strict: bool = True,
) -> AxisCaseCapabilities:
    """Load AXIS adapters and validate that required capabilities exist."""
    missing: list[str] = []
    for name in (
        "load_verbalizer",
        "load_prompt_templates",
        "load_provenance_slice_adapter",
        "load_run_observation_adapter",
        "load_provenance_instrumentation_adapter",
        "load_explained_case_resolver",
    ):
        if not hasattr(case_definition, name):
            missing.append(name)
    if missing:
        raise CaseConfigurationError(
            f"Case {getattr(case_definition, 'case_id', type(case_definition).__name__)} "
            f"is not AXIS-capable; missing {missing}"
        )
    configuration = load_case_configuration(
        case_definition,
        compiled_graph=compiled_graph,
        strict=strict,
        require_graph=compiled_graph is not None,
        require_prompts=True,
        outcome_evaluator=case_definition.load_outcome_evaluator(),
    )
    verbalizer = case_definition.load_verbalizer()
    templates = case_definition.load_prompt_templates()
    slicing = case_definition.load_provenance_slice_adapter()
    observation = case_definition.load_run_observation_adapter()
    provenance_adapter = case_definition.load_provenance_instrumentation_adapter()
    explained_resolver = case_definition.load_explained_case_resolver()
    if verbalizer is None or templates is None or slicing is None or observation is None:
        raise CaseConfigurationError(
            "AXIS case capabilities returned None for a required adapter"
        )
    if provenance_adapter is None:
        raise CaseConfigurationError(
            "AXIS case capabilities missing provenance instrumentation adapter"
        )
    if explained_resolver is None:
        raise CaseConfigurationError(
            "AXIS case capabilities missing explained-case resolver"
        )
    return AxisCaseCapabilities(
        case_configuration=configuration,
        verbalizer=verbalizer,
        prompt_templates=templates,
        slicing_adapter=slicing,
        observation_adapter=observation,
        provenance_instrumentation_adapter=provenance_adapter,
        explained_case_resolver=explained_resolver,
    )
