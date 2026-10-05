"""Deterministic RAG tool helpers — bypass ReAct for reliable pipeline steps."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


def normalize_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def content_key(text: str) -> str:
    """Stable key for near-identical chunk bodies."""
    normalized = " ".join((text or "").strip().lower().split())
    return normalized[:500]


def existing_content_keys(state: Any) -> Set[str]:
    keys: Set[str] = set()
    for chunk in getattr(state, "evidence_pool", {}).values():
        keys.add(content_key(getattr(chunk, "content", "")))
    return keys


def canonical_chunk_ids(state: Any, limit: Optional[int] = None) -> List[str]:
    ids = [
        cid
        for cid, chunk in getattr(state, "evidence_pool", {}).items()
        if getattr(chunk, "duplicate_of", None) is None
    ]
    if limit is not None:
        return ids[:limit]
    return ids


def retrieve_with_dedupe(
    rag_agents: Any,
    query: str,
    searched_queries: Set[str],
    *,
    k: int = 5,
    section_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Set[str]]:
    """Search vector store, skip duplicate bodies and repeated queries, bulk-mark dupes."""
    norm_query = normalize_query(query)
    if norm_query in searched_queries:
        return {
            "role": "retriever",
            "query": query,
            "result": f"Query already searched; skipping retrieval for: {query}",
            "skipped": True,
            "added": 0,
        }, searched_queries

    searched_queries = set(searched_queries)
    searched_queries.add(norm_query)

    try:
        results = rag_agents.rag_pipeline.search(query, k=k)
    except Exception as e:
        return {
            "role": "retriever",
            "query": query,
            "result": f"Error retrieving chunks: {e}",
            "error": str(e),
            "added": 0,
        }, searched_queries

    if not results:
        return {
            "role": "retriever",
            "query": query,
            "result": "No relevant chunks found.",
            "added": 0,
        }, searched_queries

    from rag_agents_cooperative import EvidenceChunk

    known = existing_content_keys(rag_agents.state)
    chunks_info: List[str] = []
    added = 0
    skipped = 0

    for i, (doc, score) in enumerate(results, 1):
        body_key = content_key(doc.page_content)
        if body_key in known:
            skipped += 1
            continue

        metadata = doc.metadata
        if section_id:
            if not isinstance(metadata, dict):
                metadata = dict(metadata) if metadata else {}
            metadata = dict(metadata)
            metadata["section_id"] = section_id
        chunk_id = rag_agents.state.generate_id("chunk")
        evidence_chunk = EvidenceChunk(
            chunk_id=chunk_id,
            content=doc.page_content,
            metadata=metadata,
            relevance_score=float(score),
            quality_score=float(score),
            relevance_tags=[query.lower()],
        )
        rag_agents.state.add_evidence_chunk(evidence_chunk)
        known.add(body_key)
        added += 1
        chunks_info.append(
            f"\nChunk {i} (ID: {chunk_id}, Relevance Score: {score:.4f}):\n"
            f"- Source: {metadata.get('source_file', 'unknown')}\n"
            f"- Page: {metadata.get('page_number', 'unknown')}\n"
            f"- Title: {metadata.get('title', 'unknown')}\n"
            f"- Content: {doc.page_content[:500]}...\n"
        )

    dedup_report = ""
    if added > 0:
        dedup_report = rag_agents.detect_duplicate_chunks()

    summary_parts = []
    if added:
        summary_parts.append(f"Added {added} new chunk(s)")
    if skipped:
        summary_parts.append(f"skipped {skipped} duplicate body/ies")
    header = "; ".join(summary_parts) if summary_parts else "No new chunks added"
    body = "\n".join(chunks_info) if chunks_info else ""
    result_text = f"{header} for query: {query}"
    if body:
        result_text += "\n" + body
    if dedup_report:
        result_text += f"\n\nPost-retrieval dedup: {dedup_report}"

    rag_agents.log_step(
        "retriever",
        "retrieve_chunks_deduped",
        {"query": query, "k": k, "added": added, "skipped": skipped},
    )

    return {
        "role": "retriever",
        "query": query,
        "result": result_text,
        "added": added,
        "skipped_duplicates": skipped,
    }, searched_queries


def direct_detect_duplicates(rag_agents: Any) -> Dict[str, Any]:
    out = rag_agents.detect_duplicate_chunks()
    return {"role": "quality_duplicate_detector", "result": out}


def direct_curate_evidence(rag_agents: Any) -> Dict[str, Any]:
    out = rag_agents.curate_evidence()
    return {"role": "evidence_curator", "result": out}


def direct_build_outline(rag_agents: Any) -> Dict[str, Any]:
    out = rag_agents.build_outline_from_document()
    n_sections = len(getattr(rag_agents.state, "doc_outline", []) or [])
    return {
        "role": "structure_outline",
        "result": out,
        "sections": n_sections,
    }


def direct_curriculum_lo(rag_agents: Any) -> Dict[str, Any]:
    out = rag_agents.generate_learning_objectives_for_outline()
    return {
        "role": "curriculum_lo",
        "result": out,
        "los": len(getattr(rag_agents.state, "learning_objectives", []) or []),
    }


def parse_quiz_spec(spec: str) -> tuple:
    """Parse topic|question_type|difficulty from orchestrator input."""
    parts = [p.strip() for p in (spec or "").split("|")]
    topic = parts[0] if parts else "course material"
    qtype = parts[1] if len(parts) > 1 else "multiple_choice"
    difficulty = parts[2] if len(parts) > 2 else "intermediate"
    if qtype not in ("multiple_choice", "open_ended"):
        qtype = "multiple_choice"
    return topic, qtype, difficulty


def direct_generate_quiz(
    rag_agents: Any,
    spec: str,
    *,
    num_questions: int = 2,
) -> Dict[str, Any]:
    topic, qtype, difficulty = parse_quiz_spec(spec)
    if int(num_questions) <= 0:
        return {
            "role": "quiz_generator",
            "result": "Final question cap reached; skipping quiz generation.",
            "topic": topic,
            "question_type": qtype,
            "difficulty": difficulty,
            "warning": "final_question_cap_reached",
        }
    out = rag_agents._generate_quiz(
        topic,
        qtype,
        num_questions=num_questions,
        target_difficulty=difficulty,
    )
    return {
        "role": "quiz_generator",
        "result": out,
        "topic": topic,
        "question_type": qtype,
        "difficulty": difficulty,
    }


def direct_verify_candidates(
    rag_agents: Any,
    *,
    batch_size: int = 5,
) -> Dict[str, Any]:
    out = rag_agents.verify_all_candidates(limit=batch_size)
    return out if isinstance(out, dict) else {"role": "verifier", "result": str(out)}


def chunk_ids_without_atoms(state: Any) -> List[str]:
    """Canonical chunks not yet linked to any knowledge atom."""
    ids: List[str] = []
    for cid in canonical_chunk_ids(state):
        chunk = state.evidence_pool[cid]
        if not getattr(chunk, "used_in_atoms", []):
            ids.append(cid)
    return ids


def direct_atomize_chunks(rag_agents: Any, *, max_chunks: int = 12) -> Dict[str, Any]:
    chunk_ids = canonical_chunk_ids(rag_agents.state, limit=max_chunks)
    if not chunk_ids:
        return {
            "role": "atomizer",
            "result": "No canonical chunks to atomize.",
            "error": "no_chunks",
        }
    out = rag_agents.atomize_chunks(chunk_ids)
    return {"role": "atomizer", "result": out, "chunk_count": len(chunk_ids)}


def direct_atomize_new_chunks(rag_agents: Any, *, max_chunks: int = 50) -> Dict[str, Any]:
    """Atomize only chunks that have not yet been linked to atoms."""
    chunk_ids = chunk_ids_without_atoms(rag_agents.state)
    if not chunk_ids:
        return {
            "role": "atomizer",
            "result": "No new chunks to atomize.",
            "chunk_count": 0,
            "new_only": True,
        }
    batch = chunk_ids[:max_chunks]
    out = rag_agents.atomize_chunks(batch)
    return {
        "role": "atomizer",
        "result": out,
        "chunk_count": len(batch),
        "new_only": True,
    }
