"""Node execution protocol and tracing wrapper."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from ..provenance.models import ArtifactBinding
from ..provenance.recorder import ProvenanceRecorder


@runtime_checkable
class NodeFn(Protocol):
    """LangGraph node: maps state -> partial state update."""

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]: ...


def wrap_node_with_provenance(
    node_id: str,
    node_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    *,
    get_recorder: Callable[[Dict[str, Any]], Optional[ProvenanceRecorder]],
    resolve_bindings: Optional[
        Callable[[Dict[str, Any]], tuple]
    ] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Instrument a node to record occurrence-level provenance.

    Captures state snapshot counts before domain logic when possible.
    Does not change the node's returned update keys.
    """

    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        recorder = get_recorder(state)
        bindings = ()
        if resolve_bindings is not None:
            bindings = tuple(resolve_bindings(state) or ())
        partial = None
        if recorder is not None:
            partial = recorder.begin_occurrence(node_id, input_bindings=bindings)
        error = None
        update: Dict[str, Any] = {}
        try:
            update = node_fn(state) or {}
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            if recorder is not None and partial is not None:
                produced = tuple(update.get("produced_artifact_refs") or ())
                delta = {
                    k: v
                    for k, v in update.items()
                    if k not in {"produced_artifact_refs", "cooperative_state"}
                }
                recorder.complete_occurrence(
                    partial,
                    produced_artifacts=produced,
                    state_delta=delta,
                    error=error,
                )
        return update

    wrapped.__name__ = getattr(node_fn, "__name__", node_id)
    wrapped.__doc__ = getattr(node_fn, "__doc__", None)
    return wrapped
