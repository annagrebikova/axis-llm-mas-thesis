"""Repair retrieval for sections missing PDF evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from ...input.constants import MAX_CHUNKS_PER_SECTION, MIN_CHUNKS_PER_SECTION
from ...integrations.rag_direct_tools import retrieve_with_dedupe
from .generation import log_ok, log_warn
from .section_evidence import (
    get_sections_without_evidence,
    increment_section_retrieval_attempts,
    mark_section_unsupported,
    update_section_evidence_status,
    _chunks_for_section,
)

if TYPE_CHECKING:
    from ...core.state import CooperativeState

SECTION_QUERY_SUFFIXES = (
    "key concepts definitions",
    "algorithm steps",
    "formal definition",
    "examples",
    "advantages disadvantages",
)


def build_section_retrieval_queries(
    section_title: str,
    *,
    lo_texts: Optional[List[str]] = None,
) -> List[str]:
    """Build targeted retrieval queries for one outline section."""
    title = (section_title or "").strip()
    if not title or title.lower() == "general":
        return []
    queries = [f"{title} {suffix}" for suffix in SECTION_QUERY_SUFFIXES]
    for lo_text in lo_texts or []:
        snippet = (lo_text or "").strip()[:120]
        if snippet:
            queries.append(snippet)
    seen: Set[str] = set()
    unique: List[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def _tag_new_chunks_for_section(
    state: "CooperativeState",
    section_id: str,
    before_ids: Set[str],
) -> List[str]:
    new_ids: List[str] = []
    for cid, chunk in state.evidence_pool.items():
        if cid in before_ids:
            continue
        if getattr(chunk, "duplicate_of", None) is not None:
            continue
        meta = getattr(chunk, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
            chunk.metadata = meta
        meta["section_id"] = section_id
        chunk.metadata = meta
        new_ids.append(cid)
    return new_ids


def retrieve_missing_section_evidence(
    state: "CooperativeState",
    rag_agents: Any,
    searched_queries: Set[str],
    section_status,
    *,
    min_chunks: int = MIN_CHUNKS_PER_SECTION,
    max_chunks: int = MAX_CHUNKS_PER_SECTION,
) -> Tuple[Dict[str, Any], Set[str], int]:
    """Retrieve evidence for one section until min chunks or queries exhausted."""
    section_id = section_status.section_id
    title = section_status.section_title
    lo_texts = [
        lo.objective_text
        for lo in state.learning_objectives
        if lo.section_id == section_id
    ]
    queries = build_section_retrieval_queries(title, lo_texts=lo_texts)
    increment_section_retrieval_attempts(state, section_id)

    total_added = 0
    last_result: Dict[str, Any] = {"result": "No queries generated."}

    for query in queries:
        current = _chunks_for_section(state, section_id)
        if len(current) >= max_chunks:
            break
        if len(current) >= min_chunks and total_added > 0:
            break

        before = set(state.evidence_pool.keys())
        last_result, searched_queries = retrieve_with_dedupe(
            rag_agents,
            query,
            searched_queries,
            k=5,
            section_id=section_id,
        )
        added = int(last_result.get("added") or 0)
        if added:
            _tag_new_chunks_for_section(state, section_id, before)
            total_added += added

    return last_result, searched_queries, total_added


def repair_missing_evidence_for_sections(
    state: "CooperativeState",
    rag_agents: Optional[Any],
    searched_queries: Set[str],
    *,
    min_chunks: int = MIN_CHUNKS_PER_SECTION,
    max_chunks: int = MAX_CHUNKS_PER_SECTION,
    max_sections: Optional[int] = None,
) -> Dict[str, Any]:
    """Run repair retrieval for all sections still missing evidence."""
    update_section_evidence_status(state)
    missing = get_sections_without_evidence(state)
    if max_sections is not None:
        missing = missing[:max_sections]

    if not missing:
        return {
            "sections_targeted": 0,
            "chunks_added": 0,
            "sections_recovered": 0,
            "searched_queries": searched_queries,
        }

    if rag_agents is None:
        for section in missing:
            mark_section_unsupported(
                state,
                section.section_id,
                "unsupported_after_retrieval_attempts_no_rag",
            )
        return {
            "sections_targeted": len(missing),
            "chunks_added": 0,
            "sections_recovered": 0,
            "searched_queries": searched_queries,
            "warning": "no_rag_agents",
        }

    total_added = 0
    recovered = 0
    for section in missing:
        before_count = len(_chunks_for_section(state, section.section_id))
        _, searched_queries, added = retrieve_missing_section_evidence(
            state,
            rag_agents,
            searched_queries,
            section,
            min_chunks=min_chunks,
            max_chunks=max_chunks,
        )
        total_added += added
        after_count = len(_chunks_for_section(state, section.section_id))
        if after_count > before_count:
            recovered += 1
        update_section_evidence_status(state)
        refreshed = state.section_evidence_status.get(section.section_id)
        if refreshed and not refreshed.has_sufficient_evidence:
            if refreshed.retrieval_attempts >= 2:
                mark_section_unsupported(
                    state,
                    section.section_id,
                    "unsupported_after_retrieval_attempts",
                )

    update_section_evidence_status(state)
    if total_added:
        log_ok(f"Repair retrieval added {total_added} new chunk(s) for {recovered} section(s)")

    return {
        "sections_targeted": len(missing),
        "chunks_added": total_added,
        "sections_recovered": recovered,
        "searched_queries": searched_queries,
    }


def retrieve_evidence_for_all_sections(
    state: "CooperativeState",
    rag_agents: Optional[Any],
    searched_queries: Set[str],
    *,
    min_chunks: int = MIN_CHUNKS_PER_SECTION,
    max_chunks: int = MAX_CHUNKS_PER_SECTION,
) -> Dict[str, Any]:
    """Initial retrieval pass — attempt evidence for every outline section."""
    total_added = 0
    if rag_agents is None:
        log_warn("No RAG agents — skipping section retrieval (stub mode)")
        update_section_evidence_status(state)
        return {"chunks_added": 0, "searched_queries": searched_queries}

    for section in state.doc_outline:
        status = update_section_evidence_status(state).get(section.section_id)
        if status is None:
            continue
        _, searched_queries, added = retrieve_missing_section_evidence(
            state,
            rag_agents,
            searched_queries,
            status,
            min_chunks=min_chunks,
            max_chunks=max_chunks,
        )
        total_added += added

    update_section_evidence_status(state)
    return {"chunks_added": total_added, "searched_queries": searched_queries}
