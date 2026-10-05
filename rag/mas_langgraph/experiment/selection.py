"""Select explainable CandidateMCQ refs that belong to the final QuizPack."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

from ..artifacts.models import ArtifactRef
from ..axis.explained_case import ExplainedCaseResolver


def final_quiz_q_ids(factual_run_result: Any, cooperative_state: Any = None) -> Set[str]:
    """Exact q_ids present in the final QuizPack (last pack if multiple)."""
    packs: List[Any] = []
    if cooperative_state is not None:
        packs = list(getattr(cooperative_state, "final_quiz_packs", []) or [])
    if not packs:
        cs = (getattr(factual_run_result, "final_state", {}) or {}).get(
            "cooperative_state"
        )
        packs = list(getattr(cs, "final_quiz_packs", []) or []) if cs is not None else []
    if not packs:
        return set()
    pack = packs[-1]
    return {str(getattr(q, "q_id", "")) for q in (getattr(pack, "questions", []) or [])}


def select_experiment_refs(
    *,
    explainable_refs: Sequence[ArtifactRef],
    factual_run_result: Any,
    resolver: ExplainedCaseResolver,
    include_historical_versions: bool = False,
    cooperative_state: Any = None,
) -> Tuple[ArtifactRef, ...]:
    """Default: final-quiz accepted versions only; no silent latest substitution.

    When ``include_historical_versions`` is False, keep only explainable refs whose
    ``logical_id`` appears in the final QuizPack. Prefer the highest version with
    outcome ``accept`` for each q_id; if none accept, keep the highest explainable
    version for that pack q_id.
    """
    refs = list(explainable_refs)
    if include_historical_versions:
        return tuple(sorted(refs, key=lambda r: (r.logical_id, r.version)))

    pack_ids = final_quiz_q_ids(factual_run_result, cooperative_state)
    if not pack_ids:
        # No pack → empty experimental set (do not silently use all history).
        return ()

    by_qid: dict[str, List[ArtifactRef]] = {}
    for ref in refs:
        if ref.logical_id not in pack_ids:
            continue
        by_qid.setdefault(ref.logical_id, []).append(ref)

    selected: List[ArtifactRef] = []
    for q_id in sorted(by_qid):
        versions = sorted(by_qid[q_id], key=lambda r: r.version)
        accepted: Optional[ArtifactRef] = None
        for ref in versions:
            try:
                case = resolver.resolve(
                    factual_run_result=factual_run_result,
                    artifact_ref=ref,
                )
            except Exception:
                continue
            if str(getattr(case, "outcome", "") or "") == "accept":
                accepted = ref
        selected.append(accepted if accepted is not None else versions[-1])
    return tuple(selected)
