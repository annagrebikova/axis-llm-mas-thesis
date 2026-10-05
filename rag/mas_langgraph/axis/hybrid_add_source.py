"""Optional same-document EvidenceChunk pool for hybrid AXIS ADD.

Retrieves existing indexed chunks from the same PDF/document as the factual
MAS run, converts eligible extras into a separate intervention registry, and
freezes that pool before interrogation. Does not generate or paraphrase text,
does not change factual retrieval, and does not register candidates as
explicit WHATIF alternatives.

Chroma ``similarity_search_with_score`` returns a **distance** (smaller means
more similar to the retrieval query). Raw distance is stored as
``retrieval_distance``. It is not confidence, quality, or evidence strength.

Selection (policy ``hybrid_add_from_source.v3``)
------------------------------------------------
1. Retrieve a bounded over-set of same-document hits (``k_per_query`` depth
   per query, ``max_search_hits`` unique-hit budget). Exact factual/pool
   duplicates are removed. Retrieval continues across queries until the
   budget or query list is exhausted; it does **not** stop at the first
   full final pool.
2. From that relevant retrieved set, select at most ``limit`` candidates
   with MMR-style scoring when embeddings are available:

   ``mmr = λ * relevance - (1-λ) * max_cosine_redundancy``

   * ``relevance = 1 / (1 + retrieval_distance)`` — a strictly decreasing
     transform of distance, not a calibrated quality score.
   * ``max_cosine_redundancy`` is the maximum cosine similarity of the
     candidate embedding to factual EvidenceChunks and already-selected
     external candidates. Embedding cosine is a **proxy** for redundancy,
     not proof of semantic novelty.
   * ``λ = mmr_lambda`` (default 0.7): higher values prefer query
     relevance; lower values prefer low redundancy within the retrieved
     set. Candidate selection balances retrieval relevance with
     embedding-based redundancy relative to factual evidence and
     previously selected candidates; it does **not** guarantee semantic
     novelty or relevance. Selected candidates must still pass the
     existing ADD eligibility checks.

   Selected candidates receive a zero-based integer ``selection_rank`` in
   the order they were chosen (first selected is 0). That rank is the
   preparation order used later by the hybrid vocabulary cap; it is not
   a re-sort by final MMR score. Legacy manifests without this policy and
   rank are incompatible and are not reconstructed by inventing ranks.

   If embeddings are unavailable, rank by ascending distance only and
   report ``selection_mode=relevance_only_embedding_unavailable``. Diversity
   selection is not claimed in that case.

Deduplication (exact, separate from graded redundancy)
------------------------------------------------------
A retrieved hit is an exact duplicate when any of the following match a
factual EvidenceChunk or an earlier external candidate:

* exact SHA-256 of the raw chunk text
* ``content_key`` from ``rag_direct_tools`` (whitespace-normalized, first 500
  characters)
* source span identity ``(pdf_id, page_number, chunk_index)``

Document isolation uses the index ``pdf_id`` field. Title and filename are
never used as identity. Hits with a missing ``pdf_id`` are rejected.

Frozen-pool resume
------------------
Reuse requires an exact match of policy, document id, lecturebank id,
factual run id, source PDF hash, index/RAG fingerprint, factual-evidence
fingerprint, queries, budgets, diversity settings, and pool limit. Missing
identity fields are mismatches, not wildcards. Compatible resume reconstructs
the selected artifacts without search. Incompatible or corrupted manifests
are not filtered or partially reused; rebuild requires the explicit
overwrite/fresh-run path. A persisted retrieval failure is not a valid
empty pool and may be retried.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..artifacts.models import ArtifactEnvelope
from ..artifacts.registry import ArtifactRegistry
from ..core.entities import EvidenceChunk
from ..integrations.rag_direct_tools import content_key
from .interrogation_mode import DEFAULT_HYBRID_MAX_ALTERNATIVES_PER_TARGET

HYBRID_ADD_FROM_SOURCE_POLICY = "hybrid_add_from_source.v3"
HYBRID_ADD_SOURCE_SELECTION_POLICY = "mmr_distance_relevance_v1"
SELECTION_RANK_INDEXING = "zero_based"
UNRANKED_SELECTION_RANK = 10 ** 9
INTERVENTION_PREP_BRANCH = "intervention_prep"
CANDIDATE_SOURCE_HYBRID_ADD_FROM_SOURCE = "hybrid_add_from_source"
DEFAULT_HYBRID_ADD_SOURCE_LIMIT = DEFAULT_HYBRID_MAX_ALTERNATIVES_PER_TARGET
DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY = 32
DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS = 128
DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA = 0.7
POOL_FILENAME = "hybrid_add_source_pool.json"
EMPTY_BOUNDED_SEARCH_REASON = "no_eligible_candidates_in_searched_results"
FAILED_PREP_EMPTY_REASONS = frozenset(
    {
        "retrieval_failure",
        "no_search_backend",
        "preparation_failed",
    }
)

HYBRID_ADD_SOURCE_EVENT_FIELDS = [
    "lecturebank_id",
    "document_id",
    "run_id",
    "event_type",
    "candidate_source",
    "logical_id",
    "artifact_ref",
    "pdf_id",
    "page_number",
    "chunk_index",
    "content_hash",
    "query",
    "detail",
    "retrieval_distance",
    "supporting_queries",
    "duplicate_rejection_reason",
    "relevance_component",
    "redundancy_component",
    "mmr_score",
    "selected",
    "selection_reason",
    "selection_policy",
    "selection_rank",
    "stopping_reason",
    "k_per_query",
    "max_search_hits",
    "frozen_compatible",
    "mismatch_reasons",
]


def allowed_document_ids(document_id: str, lecturebank_id: str = "") -> frozenset:
    """Index ``pdf_id`` values that belong to this document (not title/filename)."""
    allowed = set()
    doc = str(document_id or "").strip()
    lid = str(lecturebank_id or "").strip()
    if doc:
        allowed.add(doc)
        if doc.startswith("lecturebank_"):
            allowed.add(doc[len("lecturebank_") :])
    if lid:
        allowed.add(lid)
        allowed.add(f"lecturebank_{lid}")
    return frozenset(x for x in allowed if x)


def content_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def query_relevance_from_distance(distance: float) -> float:
    """Strictly decreasing transform of Chroma distance. Not a quality score."""
    return 1.0 / (1.0 + float(distance))


def mmr_selection_score(
    *,
    relevance: float,
    redundancy: float,
    mmr_lambda: float,
) -> float:
    """MMR trade-off used at each greedy selection step.

    Scores from different steps are not interchangeable with selection order.
    """
    lam = float(mmr_lambda)
    return lam * float(relevance) - (1.0 - lam) * float(redundancy)


def parse_selection_rank(value: Any) -> Optional[int]:
    """Parse a zero-based preparation ``selection_rank``. Invalid values are missing."""
    if value is None or value == "":
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    if rank < 0:
        return None
    return rank


def span_key(pdf_id: Any, page_number: Any, chunk_index: Any) -> Optional[Tuple[str, str, str]]:
    pdf = str(pdf_id or "").strip()
    if not pdf:
        return None
    page = "" if page_number in (None, "") else str(page_number)
    idx = "" if chunk_index in (None, "") else str(chunk_index)
    if page == "" and idx == "":
        return None
    return (pdf, page, idx)


def intervention_chunk_logical_id(
    *,
    document_id: str,
    page_number: Any,
    chunk_index: Any,
    content_hash: str,
) -> str:
    """Deterministic, collision-safe id distinct from factual ``chunk_*`` ids."""
    material = "|".join(
        (
            str(document_id or ""),
            "" if page_number in (None, "") else str(page_number),
            "" if chunk_index in (None, "") else str(chunk_index),
            str(content_hash or ""),
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"addsrc_{digest}"


def section_id_for_page(state: Any, page_number: Any) -> str:
    if page_number in (None, ""):
        return ""
    try:
        page = int(page_number)
    except (TypeError, ValueError):
        return ""
    for sec in list(getattr(state, "doc_outline", None) or []):
        sl = getattr(sec, "slide_range", None)
        if not sl or len(sl) < 2:
            continue
        try:
            lo, hi = int(sl[0]), int(sl[1])
        except (TypeError, ValueError):
            continue
        if lo <= page <= hi:
            return str(getattr(sec, "section_id", "") or "")
    return ""


def retrieval_queries_from_state(state: Any) -> Tuple[str, ...]:
    """Stable queries from factual outline/LOs and existing retrieval tags."""
    seen: set[str] = set()
    ordered: List[str] = []

    def _add(text: Any) -> None:
        raw = " ".join(str(text or "").split()).strip()
        if not raw:
            return
        key = raw.lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(raw)

    for sec in list(getattr(state, "doc_outline", None) or []):
        _add(getattr(sec, "title", None))
        _add(getattr(sec, "content_summary", None))
    for lo in list(getattr(state, "learning_objectives", None) or []):
        _add(getattr(lo, "objective_text", None))
    for chunk in list((getattr(state, "evidence_pool", None) or {}).values()):
        for tag in list(getattr(chunk, "relevance_tags", None) or []):
            _add(tag)
    return tuple(sorted(ordered, key=lambda t: t.lower()))


def _meta_of(chunk: Any) -> Dict[str, Any]:
    meta = getattr(chunk, "metadata", None)
    return dict(meta) if isinstance(meta, Mapping) else {}


def _canon_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def factual_identity_sets(state: Any) -> Tuple[set, set, set]:
    """Return (content_keys, span_keys, content_hashes) from factual evidence."""
    keys: set = set()
    spans: set = set()
    hashes: set = set()
    for chunk in list((getattr(state, "evidence_pool", None) or {}).values()):
        text = str(getattr(chunk, "content", "") or "")
        keys.add(content_key(text))
        hashes.add(content_sha256(text))
        meta = _meta_of(chunk)
        sk = span_key(meta.get("pdf_id"), meta.get("page_number"), meta.get("chunk_index"))
        if sk is not None:
            spans.add(sk)
    return keys, spans, hashes


def factual_evidence_fingerprint(state: Any) -> str:
    """Deterministic identity of factual evidence ids and content."""
    rows: List[Dict[str, Any]] = []
    pool = getattr(state, "evidence_pool", None) or {}
    items = []
    if isinstance(pool, Mapping):
        items = list(pool.items())
    else:
        items = [(getattr(c, "chunk_id", ""), c) for c in list(pool)]
    for logical_id, chunk in sorted(items, key=lambda kv: str(kv[0])):
        text = str(getattr(chunk, "content", "") or "")
        meta = _meta_of(chunk)
        cid = str(getattr(chunk, "chunk_id", "") or logical_id)
        rows.append(
            {
                "logical_id": str(logical_id),
                "chunk_id": cid,
                "content_hash": content_sha256(text),
                "content_key": content_key(text),
                "pdf_id": str(meta.get("pdf_id") or ""),
                "page_number": "" if meta.get("page_number") in (None, "") else str(meta.get("page_number")),
                "chunk_index": "" if meta.get("chunk_index") in (None, "") else str(meta.get("chunk_index")),
            }
        )
    return hashlib.sha256(_canon_json(rows).encode("utf-8")).hexdigest()


def canonicalize_rag_config_fingerprint(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "embedding_model": value.get("embedding_model"),
        "chunk_size": value.get("chunk_size"),
        "chunk_overlap": value.get("chunk_overlap"),
        "chunk_by_page": value.get("chunk_by_page", True),
    }


def search_fn_from_backbone(backbone: Any) -> Optional[Callable[..., Any]]:
    pipe = None
    agents = getattr(backbone, "rag_agents", None)
    if agents is not None:
        pipe = getattr(agents, "rag_pipeline", None)
    if pipe is None:
        runner = getattr(backbone, "agent_runner", None)
        pipe = getattr(runner, "rag_pipeline", None) if runner is not None else None
    search = getattr(pipe, "search", None)
    return search if callable(search) else None


def embed_fn_from_backbone(backbone: Any) -> Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]]:
    pipe = None
    agents = getattr(backbone, "rag_agents", None)
    if agents is not None:
        pipe = getattr(agents, "rag_pipeline", None)
    if pipe is None:
        runner = getattr(backbone, "agent_runner", None)
        pipe = getattr(runner, "rag_pipeline", None) if runner is not None else None
    emb = getattr(pipe, "embeddings", None) if pipe is not None else None
    if emb is None:
        return None
    embed_documents = getattr(emb, "embed_documents", None)
    embed_query = getattr(emb, "embed_query", None)
    if not callable(embed_documents) and not callable(embed_query):
        return None

    def _embed(texts: Sequence[str]) -> List[List[float]]:
        values = [str(t or "") for t in texts]
        if callable(embed_documents):
            vectors = embed_documents(values)
        else:
            vectors = [embed_query(t) for t in values]
        out: List[List[float]] = []
        for vec in list(vectors or []):
            out.append([float(x) for x in list(vec)])
        return out

    return _embed


def index_chunk_count_from_backbone(backbone: Any) -> Optional[int]:
    pipe = None
    agents = getattr(backbone, "rag_agents", None)
    if agents is not None:
        pipe = getattr(agents, "rag_pipeline", None)
    if pipe is None:
        runner = getattr(backbone, "agent_runner", None)
        pipe = getattr(runner, "rag_pipeline", None) if runner is not None else None
    stats_fn = getattr(pipe, "get_index_stats", None) if pipe is not None else None
    if not callable(stats_fn):
        return None
    try:
        stats = stats_fn() or {}
    except Exception:
        return None
    raw = stats.get("total_chunks") if isinstance(stats, Mapping) else None
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def parse_retrieval_distance(value: Any) -> Tuple[Optional[float], str]:
    """Parse a Chroma/backend distance. Missing or non-finite is invalid."""
    if value is None or value == "":
        return None, "missing_distance"
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return None, "non_numeric_distance"
    if not math.isfinite(distance):
        return None, "non_finite_distance"
    return distance, ""


def _hit_parts(hit: Any) -> Tuple[str, Dict[str, Any], Optional[float], str]:
    if isinstance(hit, (tuple, list)) and hit:
        doc = hit[0]
        raw = hit[1] if len(hit) > 1 else None
    else:
        doc = hit
        raw = getattr(hit, "score", None)
    text = str(getattr(doc, "page_content", None) or getattr(doc, "content", "") or "")
    meta = getattr(doc, "metadata", None)
    if not isinstance(meta, Mapping):
        meta = {}
    distance, err = parse_retrieval_distance(raw)
    return text, dict(meta), distance, err


def _hit_identity(pdf_id: str, page: Any, chunk_index: Any, content_hash: str) -> Tuple[str, str, str, str]:
    return (
        str(pdf_id or ""),
        "" if page in (None, "") else str(page),
        "" if chunk_index in (None, "") else str(chunk_index),
        str(content_hash or ""),
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for a, b in zip(left, right):
        fa = float(a)
        fb = float(b)
        dot += fa * fb
        na += fa * fa
        nb += fb * fb
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _sort_num(value: Any) -> Tuple[int, str]:
    try:
        return (0, f"{int(value):010d}")
    except (TypeError, ValueError):
        return (1, str(value or ""))


def retrieval_tie_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    distance = row.get("retrieval_distance")
    dist_key = float(distance) if isinstance(distance, (int, float)) and math.isfinite(float(distance)) else math.inf
    return (
        dist_key,
        _sort_num(row.get("page_number")),
        _sort_num(row.get("chunk_index")),
        str(row.get("content_hash") or ""),
        str(row.get("query") or ""),
    )


def _invoke_search(
    search_fn: Callable[..., Any],
    query: str,
    *,
    k: int,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> Any:
    if filter_dict:
        try:
            return search_fn(query, k=k, filter_dict=filter_dict)
        except TypeError:
            pass
    try:
        return search_fn(query, k=k)
    except TypeError:
        return search_fn(query)


def same_document_filter(document_id: str, lecturebank_id: str = "") -> Optional[Dict[str, Any]]:
    allowed = sorted(allowed_document_ids(document_id, lecturebank_id))
    if not allowed:
        return None
    if len(allowed) == 1:
        return {"pdf_id": allowed[0]}
    return {"pdf_id": {"$in": allowed}}


def is_failed_prep_manifest(manifest: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(manifest, Mapping):
        return False
    if str(manifest.get("retrieval_error") or "").strip():
        return True
    return str(manifest.get("empty_reason") or "") in FAILED_PREP_EMPTY_REASONS


@dataclass
class FrozenPoolCompatibility:
    compatible: bool
    mismatches: List[str] = field(default_factory=list)
    retry_failed_prep: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": self.compatible,
            "mismatches": list(self.mismatches),
            "retry_failed_prep": self.retry_failed_prep,
        }


def frozen_pool_compatibility_report(
    manifest: Optional[Mapping[str, Any]],
    *,
    document_id: str,
    lecturebank_id: str = "",
    limit: int,
    embedding_model: str = "",
    run_id: str = "",
    source_hash: str = "",
    rag_config_fingerprint: Optional[Mapping[str, Any]] = None,
    factual_fp: str = "",
    queries: Optional[Sequence[str]] = None,
    k_per_query: Optional[int] = None,
    max_search_hits: Optional[int] = None,
    mmr_lambda: Optional[float] = None,
    state: Any = None,
) -> FrozenPoolCompatibility:
    """Compare a frozen pool to the current factual experiment identity.

    Missing required identity fields are mismatches, not wildcards.
    """
    mismatches: List[str] = []
    if not isinstance(manifest, Mapping):
        return FrozenPoolCompatibility(False, ["missing_manifest"])
    if is_failed_prep_manifest(manifest):
        return FrozenPoolCompatibility(
            False,
            ["persisted_retrieval_failure"],
            retry_failed_prep=True,
        )

    if str(manifest.get("policy") or "") != HYBRID_ADD_FROM_SOURCE_POLICY:
        mismatches.append("policy")
    if str(manifest.get("document_id") or "") != str(document_id or ""):
        mismatches.append("document_id")
    if str(manifest.get("lecturebank_id") or "") != str(lecturebank_id or ""):
        mismatches.append("lecturebank_id")

    stored_run = str(manifest.get("run_id") or "")
    current_run = str(run_id or "")
    if not stored_run or not current_run:
        mismatches.append("missing_run_id")
    elif stored_run != current_run:
        mismatches.append("run_id")

    stored_hash = str(manifest.get("source_hash") or "")
    current_hash = str(source_hash or "")
    if not stored_hash or not current_hash:
        mismatches.append("missing_source_hash")
    elif stored_hash != current_hash:
        mismatches.append("source_hash")

    cfg = dict(manifest.get("retrieval_config") or {})
    stored_rag = canonicalize_rag_config_fingerprint(
        cfg.get("rag_config_fingerprint") or manifest.get("rag_config_fingerprint")
    )
    current_rag = canonicalize_rag_config_fingerprint(rag_config_fingerprint)
    if not stored_rag or not current_rag:
        mismatches.append("missing_rag_config_fingerprint")
    elif stored_rag != current_rag:
        mismatches.append("rag_config_fingerprint")

    stored_model = str(cfg.get("embedding_model") or "")
    current_model = str(embedding_model or current_rag.get("embedding_model") or "")
    if stored_model != current_model:
        mismatches.append("embedding_model")

    stored_factual = str(
        manifest.get("factual_evidence_fingerprint")
        or cfg.get("factual_evidence_fingerprint")
        or ""
    )
    current_factual = str(factual_fp or "")
    if not current_factual and state is not None:
        current_factual = factual_evidence_fingerprint(state)
    if not stored_factual or not current_factual:
        mismatches.append("missing_factual_evidence_fingerprint")
    elif stored_factual != current_factual:
        mismatches.append("factual_evidence_fingerprint")

    if int(cfg.get("limit") or 0) != int(limit):
        mismatches.append("limit")

    want_k = (
        int(k_per_query)
        if k_per_query is not None
        else int(DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY)
    )
    if int(cfg.get("k_per_query") or 0) != want_k:
        mismatches.append("k_per_query")

    want_hits = (
        int(max_search_hits)
        if max_search_hits is not None
        else int(DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS)
    )
    if int(cfg.get("max_search_hits") or 0) != want_hits:
        mismatches.append("max_search_hits")

    stored_lambda = cfg.get("mmr_lambda")
    want_lambda = (
        float(mmr_lambda)
        if mmr_lambda is not None
        else float(DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA)
    )
    try:
        stored_lambda_f = float(stored_lambda)
    except (TypeError, ValueError):
        stored_lambda_f = None
    if stored_lambda_f is None or not math.isclose(stored_lambda_f, want_lambda, rel_tol=0.0, abs_tol=1e-12):
        mismatches.append("mmr_lambda")

    stored_queries = [str(q) for q in list(cfg.get("queries") or [])]
    if queries is None:
        mismatches.append("missing_queries")
    else:
        current_queries = [str(q) for q in list(queries)]
        if stored_queries != current_queries:
            mismatches.append("queries")

    stored_selection = str(cfg.get("selection_policy") or manifest.get("selection_policy") or "")
    if stored_selection and stored_selection not in {
        HYBRID_ADD_SOURCE_SELECTION_POLICY,
        "relevance_only_embedding_unavailable",
    }:
        mismatches.append("selection_policy")

    stored_indexing = str(
        cfg.get("selection_rank_indexing")
        or manifest.get("selection_rank_indexing")
        or ""
    )
    if stored_indexing != SELECTION_RANK_INDEXING:
        mismatches.append("selection_rank_indexing")

    return FrozenPoolCompatibility(not mismatches, mismatches)


def frozen_pool_compatible(
    manifest: Optional[Mapping[str, Any]],
    **kwargs: Any,
) -> bool:
    if "factual_evidence_fingerprint" in kwargs and "factual_fp" not in kwargs:
        kwargs["factual_fp"] = kwargs.pop("factual_evidence_fingerprint")
    else:
        kwargs.pop("factual_evidence_fingerprint", None)
    return frozen_pool_compatibility_report(manifest, **kwargs).compatible


def verify_frozen_candidates(
    manifest: Mapping[str, Any],
    *,
    state: Any,
    document_id: str,
    lecturebank_id: str = "",
) -> List[str]:
    """Reject corrupted hashes, identity drift, and newly detected factual duplicates."""
    reasons: List[str] = []
    allowed = allowed_document_ids(document_id, lecturebank_id)
    factual_keys, factual_spans, factual_hashes = factual_identity_sets(state)
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for item in list(manifest.get("candidates") or []):
        if not isinstance(item, Mapping):
            reasons.append("malformed_candidate")
            continue
        text = str(item.get("content") or "")
        if not text:
            reasons.append("missing_candidate_content")
            continue
        digest = content_sha256(text)
        stored_hash = str(item.get("content_hash") or "")
        if not stored_hash or stored_hash != digest:
            reasons.append("content_hash_mismatch")
        pdf_id = str(item.get("pdf_id") or document_id or "").strip()
        if not pdf_id:
            reasons.append("missing_candidate_pdf_id")
        elif pdf_id not in allowed:
            reasons.append("candidate_wrong_document")
        page = item.get("page_number")
        cidx = item.get("chunk_index")
        expected_id = intervention_chunk_logical_id(
            document_id=str(document_id or ""),
            page_number=page,
            chunk_index=cidx,
            content_hash=digest,
        )
        logical_id = str(item.get("logical_id") or "")
        if logical_id != expected_id:
            reasons.append("logical_id_mismatch")
        if logical_id in seen_ids:
            reasons.append("duplicate_logical_id")
        seen_ids.add(logical_id)
        body_key = content_key(text)
        sk = span_key(pdf_id, page, cidx)
        if digest in factual_hashes or body_key in factual_keys or (
            sk is not None and sk in factual_spans
        ):
            reasons.append("factual_duplicate_on_resume")
        rank = parse_selection_rank(item.get("selection_rank"))
        if rank is None:
            reasons.append("missing_selection_rank")
        elif rank in seen_ranks:
            reasons.append("duplicate_selection_rank")
        else:
            seen_ranks.add(rank)
    return reasons


@dataclass
class HybridAddSourceReport:
    """Frozen retrieval/preparation record for one document."""

    policy: str = HYBRID_ADD_FROM_SOURCE_POLICY
    lecturebank_id: str = ""
    document_id: str = ""
    run_id: str = ""
    source_hash: str = ""
    factual_evidence_fingerprint: str = ""
    candidate_source: str = CANDIDATE_SOURCE_HYBRID_ADD_FROM_SOURCE
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    empty_reason: str = ""
    retrieval_error: str = ""
    reused_frozen_pool: bool = False
    source_preparation_status: str = "completed"
    external_add_pool_available: bool = False
    requested_hybrid_add_from_source: bool = True
    frozen_pool_path: str = ""
    corrupt_reason: str = ""
    stopping_reason: str = ""
    selection_mode: str = ""
    selection_policy: str = HYBRID_ADD_SOURCE_SELECTION_POLICY
    embedding_unavailable: bool = False
    source_chunks_retrieved: int = 0
    unique_hits: int = 0
    unique_same_document_hits: int = 0
    eligible_unique_chunks: int = 0
    wrong_document_rejected: int = 0
    missing_document_id_rejected: int = 0
    factual_duplicates_rejected: int = 0
    pool_duplicates_rejected: int = 0
    empty_text_rejected: int = 0
    invalid_distance_rejected: int = 0
    candidates_registered: int = 0
    frozen_compatible: Optional[bool] = None
    frozen_mismatch_reasons: List[str] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    considered: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "lecturebank_id": self.lecturebank_id,
            "document_id": self.document_id,
            "run_id": self.run_id,
            "source_hash": self.source_hash,
            "factual_evidence_fingerprint": self.factual_evidence_fingerprint,
            "candidate_source": self.candidate_source,
            "retrieval_config": dict(self.retrieval_config),
            "empty_reason": self.empty_reason,
            "retrieval_error": self.retrieval_error,
            "reused_frozen_pool": self.reused_frozen_pool,
            "source_preparation_status": self.source_preparation_status,
            "external_add_pool_available": self.external_add_pool_available,
            "requested_hybrid_add_from_source": self.requested_hybrid_add_from_source,
            "frozen_pool_path": self.frozen_pool_path,
            "corrupt_reason": self.corrupt_reason,
            "stopping_reason": self.stopping_reason,
            "selection_mode": self.selection_mode,
            "selection_policy": self.selection_policy,
            "embedding_unavailable": self.embedding_unavailable,
            "source_chunks_retrieved": self.source_chunks_retrieved,
            "unique_hits": self.unique_hits,
            "unique_same_document_hits": self.unique_same_document_hits,
            "eligible_unique_chunks": self.eligible_unique_chunks,
            "wrong_document_rejected": self.wrong_document_rejected,
            "missing_document_id_rejected": self.missing_document_id_rejected,
            "factual_duplicates_rejected": self.factual_duplicates_rejected,
            "pool_duplicates_rejected": self.pool_duplicates_rejected,
            "empty_text_rejected": self.empty_text_rejected,
            "invalid_distance_rejected": self.invalid_distance_rejected,
            "candidates_registered": self.candidates_registered,
            "frozen_compatible": self.frozen_compatible,
            "frozen_mismatch_reasons": list(self.frozen_mismatch_reasons),
            "events": list(self.events),
            "candidates": list(self.candidates),
            "considered": list(self.considered),
        }


def report_from_manifest(manifest: Mapping[str, Any], *, run_id: str = "") -> HybridAddSourceReport:
    return HybridAddSourceReport(
        policy=str(manifest.get("policy") or HYBRID_ADD_FROM_SOURCE_POLICY),
        lecturebank_id=str(manifest.get("lecturebank_id") or ""),
        document_id=str(manifest.get("document_id") or ""),
        run_id=str(run_id or manifest.get("run_id") or ""),
        source_hash=str(manifest.get("source_hash") or ""),
        factual_evidence_fingerprint=str(manifest.get("factual_evidence_fingerprint") or ""),
        retrieval_config=dict(manifest.get("retrieval_config") or {}),
        empty_reason=str(manifest.get("empty_reason") or ""),
        retrieval_error=str(manifest.get("retrieval_error") or ""),
        reused_frozen_pool=True,
        source_preparation_status="reused",
        external_add_pool_available=bool(list(manifest.get("candidates") or [])),
        requested_hybrid_add_from_source=True,
        frozen_pool_path=str(manifest.get("frozen_pool_path") or ""),
        stopping_reason=str(manifest.get("stopping_reason") or "reused_frozen_pool"),
        selection_mode=str(manifest.get("selection_mode") or ""),
        selection_policy=str(
            manifest.get("selection_policy")
            or (manifest.get("retrieval_config") or {}).get("selection_policy")
            or HYBRID_ADD_SOURCE_SELECTION_POLICY
        ),
        embedding_unavailable=bool(manifest.get("embedding_unavailable")),
        source_chunks_retrieved=int(manifest.get("source_chunks_retrieved") or 0),
        unique_hits=int(manifest.get("unique_hits") or 0),
        unique_same_document_hits=int(manifest.get("unique_same_document_hits") or 0),
        eligible_unique_chunks=int(manifest.get("eligible_unique_chunks") or 0),
        wrong_document_rejected=int(manifest.get("wrong_document_rejected") or 0),
        missing_document_id_rejected=int(manifest.get("missing_document_id_rejected") or 0),
        factual_duplicates_rejected=int(manifest.get("factual_duplicates_rejected") or 0),
        pool_duplicates_rejected=int(manifest.get("pool_duplicates_rejected") or 0),
        empty_text_rejected=int(manifest.get("empty_text_rejected") or 0),
        invalid_distance_rejected=int(manifest.get("invalid_distance_rejected") or 0),
        candidates_registered=len(list(manifest.get("candidates") or [])),
        frozen_compatible=True,
        frozen_mismatch_reasons=[],
        events=list(manifest.get("events") or []),
        candidates=list(manifest.get("candidates") or []),
        considered=list(manifest.get("considered") or []),
    )


def _event(
    report: HybridAddSourceReport,
    event_type: str,
    *,
    logical_id: str = "",
    artifact_ref: str = "",
    pdf_id: str = "",
    page_number: Any = "",
    chunk_index: Any = "",
    content_hash: str = "",
    query: str = "",
    detail: str = "",
    **extra: Any,
) -> None:
    row: Dict[str, Any] = {
        "lecturebank_id": report.lecturebank_id,
        "document_id": report.document_id,
        "run_id": report.run_id,
        "event_type": event_type,
        "candidate_source": CANDIDATE_SOURCE_HYBRID_ADD_FROM_SOURCE,
        "logical_id": logical_id,
        "artifact_ref": artifact_ref,
        "pdf_id": str(pdf_id or ""),
        "page_number": "" if page_number in (None, "") else page_number,
        "chunk_index": "" if chunk_index in (None, "") else chunk_index,
        "content_hash": content_hash,
        "query": query,
        "detail": detail,
        "selection_policy": report.selection_policy or report.selection_mode,
        "stopping_reason": report.stopping_reason,
        "k_per_query": (report.retrieval_config or {}).get("k_per_query"),
        "max_search_hits": (report.retrieval_config or {}).get("max_search_hits"),
        "frozen_compatible": report.frozen_compatible,
        "mismatch_reasons": ";".join(report.frozen_mismatch_reasons),
    }
    for key, value in extra.items():
        if value is not None:
            row[key] = value
    report.events.append(row)


def _wrap_candidate(
    *,
    logical_id: str,
    text: str,
    metadata: Mapping[str, Any],
    retrieval_distance: Optional[float],
    query: str,
    supporting_queries: Sequence[str],
    run_id: str,
    document_id: str,
    content_hash: str,
    registry: ArtifactRegistry,
    relevance_component: Optional[float] = None,
    redundancy_component: Optional[float] = None,
    mmr_score: Optional[float] = None,
    instance_id: str = "",
    selection_rank: Optional[int] = None,
) -> ArtifactEnvelope:
    from ..cases.mcq.artifacts import wrap_evidence_chunk

    meta = dict(metadata)
    meta.setdefault("pdf_id", document_id)
    if retrieval_distance is not None:
        meta["retrieval_distance"] = float(retrieval_distance)
    if selection_rank is not None:
        meta["selection_rank"] = int(selection_rank)
    derived_relevance = (
        float(relevance_component)
        if relevance_component is not None
        else (query_relevance_from_distance(retrieval_distance) if retrieval_distance is not None else 0.0)
    )
    tags = []
    for item in [query, *list(supporting_queries or ())]:
        raw = str(item or "").strip()
        if raw and raw.lower() not in tags:
            tags.append(raw.lower())
    chunk = EvidenceChunk(
        chunk_id=logical_id,
        content=text,
        metadata=meta,
        relevance_score=derived_relevance,
        quality_score=0.0,
        relevance_tags=tags,
        flags=["hybrid_add_from_source"],
    )
    env = wrap_evidence_chunk(
        chunk,
        branch_id=INTERVENTION_PREP_BRANCH,
        created_in_run=f"{run_id}:hybrid_add_prep" if run_id else "hybrid_add_prep",
        registry=None,
    )
    stamped = ArtifactEnvelope(
        instance_id=str(instance_id or env.instance_id),
        ref=env.ref,
        artifact_type=env.artifact_type,
        payload=env.payload,
        branch_id=env.branch_id,
        created_in_run=env.created_in_run,
        superseded=False,
        metadata={
            "hybrid_add_from_source": True,
            "hybrid_add_from_source_policy": HYBRID_ADD_FROM_SOURCE_POLICY,
            "retrieved_for_intervention_prep": True,
            "not_factually_produced": True,
            "not_factually_consumed": True,
            "content_hash": content_hash,
            "document_id": document_id,
            "retrieval_query": query,
            "supporting_queries": list(supporting_queries or []),
            "retrieval_distance": retrieval_distance,
            "query_relevance": derived_relevance,
            "redundancy_component": redundancy_component,
            "mmr_score": mmr_score,
            "selection_rank": selection_rank,
            "selection_rank_indexing": SELECTION_RANK_INDEXING,
        },
    )
    return registry.store(stamped)


def _candidate_record(
    *,
    env: ArtifactEnvelope,
    text: str,
    meta: Mapping[str, Any],
    retrieval_distance: Optional[float],
    query: str,
    supporting_queries: Sequence[str],
    content_hash: str,
    document_id: str,
    relevance_component: Optional[float] = None,
    redundancy_component: Optional[float] = None,
    mmr_score: Optional[float] = None,
    selected: bool = True,
    selection_reason: str = "selected",
    selection_policy: str = "",
    selection_rank: Optional[int] = None,
) -> Dict[str, Any]:
    payload = env.payload
    pmeta = _meta_of(payload)
    return {
        "logical_id": env.ref.logical_id,
        "artifact_ref": str(env.ref),
        "instance_id": env.instance_id,
        "content": text,
        "content_hash": content_hash,
        "pdf_id": str(pmeta.get("pdf_id") or meta.get("pdf_id") or document_id),
        "page_number": pmeta.get("page_number", meta.get("page_number")),
        "chunk_index": pmeta.get("chunk_index", meta.get("chunk_index")),
        "source_file": pmeta.get("source_file", meta.get("source_file")),
        "section_id": pmeta.get("section_id", meta.get("section_id") or ""),
        "title": pmeta.get("title", meta.get("title")),
        "retrieval_distance": retrieval_distance,
        "query_relevance": relevance_component,
        "relevance_component": relevance_component,
        "redundancy_component": redundancy_component,
        "mmr_score": mmr_score,
        "query": query,
        "supporting_queries": list(supporting_queries or []),
        "selected": selected,
        "selection_reason": selection_reason,
        "selection_policy": selection_policy,
        "selection_rank": selection_rank,
        "selection_rank_indexing": SELECTION_RANK_INDEXING,
        "candidate_source": CANDIDATE_SOURCE_HYBRID_ADD_FROM_SOURCE,
    }


def registry_from_manifest(
    manifest: Mapping[str, Any],
    *,
    run_id: str = "",
) -> ArtifactRegistry:
    """Rebuild the intervention registry from a frozen pool (no search)."""
    registry = ArtifactRegistry()
    document_id = str(manifest.get("document_id") or "")
    rid = str(run_id or manifest.get("run_id") or "")
    for item in list(manifest.get("candidates") or []):
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("content") or "")
        if not text:
            continue
        logical_id = str(item.get("logical_id") or "")
        if not logical_id:
            continue
        meta = {
            "pdf_id": item.get("pdf_id") or document_id,
            "page_number": item.get("page_number"),
            "chunk_index": item.get("chunk_index"),
            "source_file": item.get("source_file"),
            "section_id": item.get("section_id") or "",
            "title": item.get("title"),
        }
        distance = item.get("retrieval_distance")
        parsed, _err = (
            parse_retrieval_distance(distance)
            if distance is not None
            else (None, "missing_distance")
        )
        _wrap_candidate(
            logical_id=logical_id,
            text=text,
            metadata=meta,
            retrieval_distance=parsed,
            query=str(item.get("query") or ""),
            supporting_queries=list(item.get("supporting_queries") or []),
            run_id=rid,
            document_id=document_id,
            content_hash=str(item.get("content_hash") or content_sha256(text)),
            registry=registry,
            relevance_component=item.get("relevance_component", item.get("query_relevance")),
            redundancy_component=item.get("redundancy_component"),
            mmr_score=item.get("mmr_score"),
            instance_id=str(item.get("instance_id") or ""),
            selection_rank=parse_selection_rank(item.get("selection_rank")),
        )
    return registry


class FrozenPoolCorruptError(ValueError):
    """Present frozen-pool file is unreadable or structurally invalid."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass
class FrozenPoolLoadResult:
    """Outcome of reading a frozen hybrid ADD source pool from disk."""

    status: str
    path: str = ""
    reason: str = ""
    manifest: Optional[Dict[str, Any]] = None


def load_hybrid_add_source_pool_result(path: Path) -> FrozenPoolLoadResult:
    """Distinguish a missing pool file from a present but corrupt manifest.

    ``status`` is ``missing``, ``loaded``, or ``corrupt``. A corrupt file is
    never returned as a usable empty pool.
    """
    p = Path(path)
    location = str(p)
    if not p.is_file():
        return FrozenPoolLoadResult(status="missing", path=location, reason="missing_file")
    try:
        size = p.stat().st_size
    except OSError as exc:
        return FrozenPoolLoadResult(
            status="corrupt",
            path=location,
            reason=f"unreadable:{type(exc).__name__}: {exc}",
        )
    if size == 0:
        return FrozenPoolLoadResult(status="corrupt", path=location, reason="empty_file")
    try:
        with p.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        return FrozenPoolLoadResult(
            status="corrupt",
            path=location,
            reason=f"malformed_json:{exc.msg} (line {exc.lineno})",
        )
    except OSError as exc:
        return FrozenPoolLoadResult(
            status="corrupt",
            path=location,
            reason=f"unreadable:{type(exc).__name__}: {exc}",
        )
    if not isinstance(data, dict):
        return FrozenPoolLoadResult(
            status="corrupt",
            path=location,
            reason=f"invalid_top_level_type:{type(data).__name__}",
        )
    if "policy" not in data:
        return FrozenPoolLoadResult(
            status="corrupt",
            path=location,
            reason="missing_required_field:policy",
            manifest=data,
        )
    if "candidates" not in data:
        return FrozenPoolLoadResult(
            status="corrupt",
            path=location,
            reason="missing_required_field:candidates",
            manifest=data,
        )
    if not isinstance(data.get("candidates"), list):
        return FrozenPoolLoadResult(
            status="corrupt",
            path=location,
            reason="invalid_candidates_type",
            manifest=data,
        )
    return FrozenPoolLoadResult(status="loaded", path=location, manifest=data)


def load_hybrid_add_source_pool(path: Path) -> Optional[Dict[str, Any]]:
    result = load_hybrid_add_source_pool_result(path)
    if result.status == "loaded":
        return result.manifest
    if result.status == "missing":
        return None
    raise FrozenPoolCorruptError(result.path, result.reason)


def should_persist_hybrid_add_source_pool(report: Any) -> bool:
    """Do not replace an incompatible, corrupt, or already-frozen on-disk pool."""
    payload = report.to_dict() if hasattr(report, "to_dict") else dict(report or {})
    reason = str(payload.get("empty_reason") or "")
    if reason in {"frozen_pool_incompatible", "frozen_pool_corrupt"}:
        return False
    if bool(payload.get("reused_frozen_pool")):
        return False
    return True


def corrupt_frozen_pool_report(
    *,
    path: str,
    reason: str,
    document_id: str = "",
    lecturebank_id: str = "",
    run_id: str = "",
    source_hash: str = "",
) -> HybridAddSourceReport:
    report = HybridAddSourceReport(
        lecturebank_id=str(lecturebank_id or ""),
        document_id=str(document_id or ""),
        run_id=str(run_id or ""),
        source_hash=str(source_hash or ""),
        empty_reason="frozen_pool_corrupt",
        stopping_reason="frozen_pool_corrupt",
        source_preparation_status="failed",
        external_add_pool_available=False,
        requested_hybrid_add_from_source=True,
        frozen_pool_path=str(path or ""),
        corrupt_reason=str(reason or ""),
        frozen_compatible=False,
        frozen_mismatch_reasons=["frozen_pool_corrupt"],
        candidates_registered=0,
    )
    _event(
        report,
        "frozen_pool_corrupt",
        detail=(
            f"Corrupt hybrid ADD source pool at {path}: {reason}. "
            "Factual MAS output is preserved. External ADD was requested but is "
            "unavailable. Rebuild with --overwrite or a fresh output directory. "
            "The corrupt file was not overwritten."
        ),
        mismatch_reasons="frozen_pool_corrupt",
    )
    return report


def persist_hybrid_add_source_pool(path: Path, report: Any) -> Path:
    from mas_langgraph.batch.io import atomic_write_json

    payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    return atomic_write_json(Path(path), payload)


def _embed_texts(
    embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    texts: Sequence[str],
) -> List[List[float]]:
    vectors = list(embed_fn(list(texts)) or [])
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"embed_fn returned {len(vectors)} vectors for {len(texts)} texts"
        )
    return [[float(x) for x in list(vec)] for vec in vectors]


def select_diverse_relevant(
    eligible: Sequence[Mapping[str, Any]],
    *,
    factual_texts: Sequence[str],
    limit: int,
    mmr_lambda: float,
    embed_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]],
) -> Tuple[List[Dict[str, Any]], str, bool, str]:
    """MMR-style selection inside the already-filtered relevant set.

    Returns (annotated rows in original eligible order of selection+rejects,
    selection_mode, embedding_unavailable, error).
    """
    cap = max(0, int(limit))
    rows = [dict(row) for row in eligible]
    for row in rows:
        distance = row.get("retrieval_distance")
        row["relevance_component"] = (
            query_relevance_from_distance(float(distance))
            if isinstance(distance, (int, float)) and math.isfinite(float(distance))
            else None
        )
        row["redundancy_component"] = None
        row["mmr_score"] = None
        row["selected"] = False
        row["selection_reason"] = ""
        row["selection_rank"] = None

    if cap <= 0 or not rows:
        return rows, HYBRID_ADD_SOURCE_SELECTION_POLICY, False, ""

    if embed_fn is None:
        ordered = sorted(rows, key=retrieval_tie_key)
        for idx, row in enumerate(ordered):
            if idx < cap:
                row["selected"] = True
                row["selection_reason"] = "relevance_only_embedding_unavailable"
                row["mmr_score"] = row.get("relevance_component")
                row["selection_rank"] = idx
            else:
                row["selected"] = False
                row["selection_reason"] = "not_selected_relevance_only_cap"
                row["selection_rank"] = None
        return (
            ordered,
            "relevance_only_embedding_unavailable",
            True,
            "",
        )

    try:
        candidate_texts = [str(row.get("text") or "") for row in rows]
        fact_list = [str(t or "") for t in factual_texts]
        all_texts = list(fact_list) + candidate_texts
        vectors = _embed_texts(embed_fn, all_texts)
    except Exception as exc:
        ordered = sorted(rows, key=retrieval_tie_key)
        for idx, row in enumerate(ordered):
            if idx < cap:
                row["selected"] = True
                row["selection_reason"] = "relevance_only_embedding_failed"
                row["mmr_score"] = row.get("relevance_component")
                row["selection_rank"] = idx
            else:
                row["selected"] = False
                row["selection_reason"] = "not_selected_relevance_only_cap"
                row["selection_rank"] = None
        return (
            ordered,
            "relevance_only_embedding_unavailable",
            True,
            f"{type(exc).__name__}: {exc}",
        )

    factual_vecs = vectors[: len(fact_list)]
    cand_vecs = vectors[len(fact_list) :]
    remaining = list(range(len(rows)))
    selected_idx: List[int] = []
    selected_vecs: List[List[float]] = list(factual_vecs)
    lam = float(mmr_lambda)

    def _redundancy(vec: Sequence[float]) -> float:
        if not selected_vecs:
            return 0.0
        return max(cosine_similarity(vec, other) for other in selected_vecs)

    while remaining and len(selected_idx) < cap:
        best_i = None
        best_key = None
        for i in remaining:
            row = rows[i]
            rel = row.get("relevance_component")
            if rel is None:
                continue
            red = _redundancy(cand_vecs[i])
            mmr = mmr_selection_score(
                relevance=float(rel),
                redundancy=float(red),
                mmr_lambda=lam,
            )
            key = (
                -mmr,
                *retrieval_tie_key(row),
                i,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
                rows[i]["_pending_redundancy"] = red
                rows[i]["_pending_mmr"] = mmr
        if best_i is None:
            break
        row = rows[best_i]
        row["redundancy_component"] = float(row.pop("_pending_redundancy", 0.0))
        row["mmr_score"] = float(row.pop("_pending_mmr", 0.0))
        row["selected"] = True
        row["selection_reason"] = "mmr_selected"
        row["selection_rank"] = len(selected_idx)
        selected_idx.append(best_i)
        selected_vecs.append(cand_vecs[best_i])
        remaining.remove(best_i)

    for i in remaining:
        row = rows[i]
        row.pop("_pending_redundancy", None)
        row.pop("_pending_mmr", None)
        rel = row.get("relevance_component")
        red = _redundancy(cand_vecs[i]) if rel is not None else None
        row["redundancy_component"] = red
        row["mmr_score"] = (
            mmr_selection_score(
                relevance=float(rel), redundancy=float(red), mmr_lambda=lam
            )
            if rel is not None and red is not None
            else None
        )
        row["selected"] = False
        row["selection_rank"] = None
        if rel is None:
            row["selection_reason"] = "invalid_distance"
        else:
            row["selection_reason"] = "not_selected_mmr_cap"

    selected_set = set(selected_idx)
    ordered = [rows[i] for i in selected_idx] + [rows[i] for i in range(len(rows)) if i not in selected_set]
    return ordered, HYBRID_ADD_SOURCE_SELECTION_POLICY, False, ""


def prepare_hybrid_add_source_registry(
    *,
    state: Any,
    search_fn: Optional[Callable[..., Any]],
    document_id: str,
    lecturebank_id: str = "",
    run_id: str = "",
    limit: int = DEFAULT_HYBRID_ADD_SOURCE_LIMIT,
    embedding_model: str = "",
    k_per_query: Optional[int] = None,
    max_search_hits: Optional[int] = None,
    mmr_lambda: Optional[float] = None,
    frozen_manifest: Optional[Mapping[str, Any]] = None,
    source_hash: str = "",
    rag_config_fingerprint: Optional[Mapping[str, Any]] = None,
    embed_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
    index_chunk_count: Optional[int] = None,
) -> Tuple[ArtifactRegistry, HybridAddSourceReport]:
    """Build a separate intervention registry of same-document extra chunks.

    Does not mutate ``state``. Does not choose candidates from replay effects.
    """
    cap = max(0, int(limit))
    k = int(k_per_query) if k_per_query is not None else int(DEFAULT_HYBRID_ADD_SOURCE_K_PER_QUERY)
    k = max(1, k)
    budget = (
        int(max_search_hits)
        if max_search_hits is not None
        else int(DEFAULT_HYBRID_ADD_SOURCE_MAX_SEARCH_HITS)
    )
    budget = max(1, budget)
    lam = (
        float(mmr_lambda)
        if mmr_lambda is not None
        else float(DEFAULT_HYBRID_ADD_SOURCE_MMR_LAMBDA)
    )
    queries = retrieval_queries_from_state(state)
    rag_fp = canonicalize_rag_config_fingerprint(rag_config_fingerprint)
    factual_fp = factual_evidence_fingerprint(state)
    model = str(embedding_model or rag_fp.get("embedding_model") or "")
    report = HybridAddSourceReport(
        lecturebank_id=str(lecturebank_id or ""),
        document_id=str(document_id or ""),
        run_id=str(run_id or ""),
        source_hash=str(source_hash or ""),
        factual_evidence_fingerprint=factual_fp,
        retrieval_config={
            "policy": HYBRID_ADD_FROM_SOURCE_POLICY,
            "selection_policy": HYBRID_ADD_SOURCE_SELECTION_POLICY,
            "limit": cap,
            "k_per_query": k,
            "max_search_hits": budget,
            "mmr_lambda": lam,
            "embedding_model": model,
            "rag_config_fingerprint": rag_fp,
            "queries": list(queries),
            "distance_semantics": "chroma_distance_smaller_more_similar",
            "query_relevance": "1/(1+retrieval_distance)",
            "diversity": "mmr_embedding_cosine_proxy_vs_factual_and_selected",
            "selection_rank_indexing": SELECTION_RANK_INDEXING,
            "same_document_filter": True,
        },
    )

    if frozen_manifest is not None:
        compat = frozen_pool_compatibility_report(
            frozen_manifest,
            document_id=str(document_id or ""),
            lecturebank_id=str(lecturebank_id or ""),
            limit=cap,
            embedding_model=model,
            run_id=str(run_id or ""),
            source_hash=str(source_hash or ""),
            rag_config_fingerprint=rag_fp,
            factual_fp=factual_fp,
            queries=queries,
            k_per_query=k,
            max_search_hits=budget,
            mmr_lambda=lam,
            state=state,
        )
        report.frozen_compatible = compat.compatible
        report.frozen_mismatch_reasons = list(compat.mismatches)
        if compat.retry_failed_prep:
            report.frozen_compatible = False
            report.frozen_mismatch_reasons = list(compat.mismatches)
            _event(
                report,
                "frozen_pool_retry_failed_prep",
                detail=(
                    "Persisted retrieval failure is not a completed empty pool; "
                    "retrying search"
                ),
                mismatch_reasons=";".join(compat.mismatches),
            )
        elif compat.compatible:
            verify_reasons = verify_frozen_candidates(
                frozen_manifest,
                state=state,
                document_id=str(document_id or ""),
                lecturebank_id=str(lecturebank_id or ""),
            )
            if verify_reasons:
                report.frozen_compatible = False
                report.frozen_mismatch_reasons = verify_reasons
                report.empty_reason = "frozen_pool_incompatible"
                report.stopping_reason = "frozen_pool_incompatible"
                _event(
                    report,
                    "frozen_pool_incompatible",
                    detail="Frozen pool failed content/identity recheck; refusing reuse",
                    mismatch_reasons=";".join(verify_reasons),
                )
                report.source_preparation_status = "failed"
                report.external_add_pool_available = False
                return ArtifactRegistry(), report
            registry = registry_from_manifest(frozen_manifest, run_id=str(run_id or ""))
            reused = report_from_manifest(frozen_manifest, run_id=str(run_id or ""))
            reused.frozen_compatible = True
            reused.stopping_reason = "reused_frozen_pool"
            _event(
                reused,
                "reused_frozen_pool",
                detail="resume reconstructed frozen hybrid ADD source pool without search",
                frozen_compatible=True,
            )
            return registry, reused
        else:
            report.empty_reason = "frozen_pool_incompatible"
            report.stopping_reason = "frozen_pool_incompatible"
            _event(
                report,
                "frozen_pool_incompatible",
                detail=(
                    "Frozen pool is not compatible with this factual run; "
                    "rebuild requires --overwrite / a fresh run. "
                    "Not searching and not partially reusing the stored pool."
                ),
                mismatch_reasons=";".join(compat.mismatches),
            )
            report.source_preparation_status = "failed"
            report.external_add_pool_available = False
            return ArtifactRegistry(), report

    registry = ArtifactRegistry()
    if cap <= 0:
        report.empty_reason = "limit_zero"
        report.stopping_reason = "limit_zero"
        _event(report, "empty_pool", detail="hybrid_add_source_limit is 0")
        return registry, report
    if not queries:
        report.empty_reason = "no_retrieval_queries"
        report.stopping_reason = "no_retrieval_queries"
        _event(
            report,
            "empty_pool",
            detail="No section titles, learning objectives, or factual retrieval tags to query",
        )
        return registry, report
    if search_fn is None:
        report.empty_reason = "no_search_backend"
        report.retrieval_error = "no_search_backend"
        report.stopping_reason = "retrieval_backend_failure"
        _event(
            report,
            "retrieval_failure",
            detail="No vector-store search function is available",
        )
        report.source_preparation_status = "failed"
        report.external_add_pool_available = False
        return registry, report

    allowed = allowed_document_ids(document_id, lecturebank_id)
    factual_keys, factual_spans, factual_hashes = factual_identity_sets(state)
    filter_dict = same_document_filter(document_id, lecturebank_id)
    merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    unique_order: List[Tuple[str, str, str, str]] = []
    budget_exhausted = False
    queries_issued = 0

    try:
        for query in queries:
            if len(unique_order) >= budget:
                budget_exhausted = True
                break
            queries_issued += 1
            hits = _invoke_search(search_fn, query, k=k, filter_dict=filter_dict)
            if not hits:
                continue
            for hit in hits:
                text, meta, distance, dist_err = _hit_parts(hit)
                pdf_id = str(meta.get("pdf_id") or "").strip()
                page = meta.get("page_number")
                cidx = meta.get("chunk_index")
                digest = content_sha256(text)
                hit_id = _hit_identity(pdf_id, page, cidx, digest)
                report.source_chunks_retrieved += 1
                if hit_id not in merged:
                    if len(unique_order) >= budget:
                        budget_exhausted = True
                        break
                    unique_order.append(hit_id)
                    merged[hit_id] = {
                        "text": text,
                        "metadata": dict(meta),
                        "pdf_id": pdf_id,
                        "page_number": page,
                        "chunk_index": cidx,
                        "content_hash": digest,
                        "retrieval_distance": None,
                        "query": query if not dist_err else "",
                        "supporting_queries": [query],
                    }
                else:
                    prev = merged[hit_id]
                    if query not in prev["supporting_queries"]:
                        prev["supporting_queries"].append(query)
                    if not prev.get("text") and text:
                        prev["text"] = text
                        prev["metadata"] = dict(meta)
                if dist_err:
                    _event(
                        report,
                        "invalid_distance",
                        pdf_id=pdf_id,
                        page_number=page,
                        chunk_index=cidx,
                        content_hash=digest,
                        query=query,
                        detail=dist_err,
                        duplicate_rejection_reason=dist_err,
                    )
                    continue
                prev = merged[hit_id]
                prev_dist = prev.get("retrieval_distance")
                if prev_dist is None or float(distance) < float(prev_dist):
                    prev["retrieval_distance"] = float(distance)
                    prev["query"] = query
                    prev["text"] = text
                    prev["metadata"] = dict(meta)
                    prev["pdf_id"] = pdf_id
                    prev["page_number"] = page
                    prev["chunk_index"] = cidx
                    prev["content_hash"] = digest
            if budget_exhausted:
                break
    except Exception as exc:
        report.retrieval_error = f"{type(exc).__name__}: {exc}"
        report.empty_reason = "retrieval_failure"
        report.stopping_reason = "retrieval_backend_failure"
        _event(report, "retrieval_failure", detail=report.retrieval_error)
        report.source_preparation_status = "failed"
        report.external_add_pool_available = False
        return ArtifactRegistry(), report

    report.unique_hits = len(unique_order)
    unique_same_doc: set = set()
    pool_keys: set = set()
    pool_spans: set = set()
    pool_hashes: set = set()
    for hit_id in unique_order:
        row = merged[hit_id]
        text = str(row.get("text") or "")
        meta = dict(row.get("metadata") or {})
        pdf_id = str(row.get("pdf_id") or "").strip()
        page = row.get("page_number")
        cidx = row.get("chunk_index")
        digest = str(row.get("content_hash") or content_sha256(text))
        query = str(row.get("query") or "")
        distance = row.get("retrieval_distance")
        if distance is None:
            report.invalid_distance_rejected += 1
            row["status"] = "invalid_distance"
            continue
        if not text.strip():
            report.empty_text_rejected += 1
            row["status"] = "empty_text"
            _event(
                report,
                "empty_text_rejected",
                pdf_id=pdf_id,
                page_number=page,
                chunk_index=cidx,
                query=query,
                retrieval_distance=float(distance),
            )
            continue
        if not pdf_id:
            report.missing_document_id_rejected += 1
            row["status"] = "missing_pdf_id"
            _event(
                report,
                "missing_document_id_rejected",
                page_number=page,
                chunk_index=cidx,
                query=query,
                retrieval_distance=float(distance),
                detail="index hit has no pdf_id; title/filename are not used as identity",
                duplicate_rejection_reason="missing_pdf_id",
            )
            continue
        if pdf_id not in allowed:
            report.wrong_document_rejected += 1
            row["status"] = "wrong_document"
            _event(
                report,
                "wrong_document_rejected",
                pdf_id=pdf_id,
                page_number=page,
                chunk_index=cidx,
                query=query,
                retrieval_distance=float(distance),
                detail=f"pdf_id {pdf_id!r} not in {sorted(allowed)}",
                duplicate_rejection_reason="wrong_document",
            )
            continue
        unique_same_doc.add(hit_id)
        body_key = content_key(text)
        sk = span_key(pdf_id, page, cidx)
        if digest in factual_hashes or body_key in factual_keys or (
            sk is not None and sk in factual_spans
        ):
            report.factual_duplicates_rejected += 1
            row["status"] = "factual_duplicate"
            _event(
                report,
                "factual_duplicate_rejected",
                pdf_id=pdf_id,
                page_number=page,
                chunk_index=cidx,
                content_hash=digest,
                query=query,
                retrieval_distance=float(distance),
                detail="matches factual evidence by hash, content_key, or source span",
                duplicate_rejection_reason="factual_duplicate",
            )
            continue
        if digest in pool_hashes or body_key in pool_keys or (
            sk is not None and sk in pool_spans
        ):
            report.pool_duplicates_rejected += 1
            row["status"] = "pool_duplicate"
            _event(
                report,
                "pool_duplicate_rejected",
                pdf_id=pdf_id,
                page_number=page,
                chunk_index=cidx,
                content_hash=digest,
                query=query,
                retrieval_distance=float(distance),
                duplicate_rejection_reason="pool_duplicate",
            )
            continue
        pool_hashes.add(digest)
        pool_keys.add(body_key)
        if sk is not None:
            pool_spans.add(sk)
        section_id = str(meta.get("section_id") or "") or section_id_for_page(state, page)
        if section_id:
            meta = dict(meta)
            meta["section_id"] = section_id
            row["metadata"] = meta
        row["status"] = "eligible"

    report.unique_same_document_hits = len(unique_same_doc)
    index_n: Optional[int] = None
    try:
        index_n = int(index_chunk_count) if index_chunk_count is not None else None
    except (TypeError, ValueError):
        index_n = None
    index_exhausted = bool(index_n is not None and index_n > 0 and len(unique_same_doc) >= index_n)
    if index_exhausted:
        report.stopping_reason = "index_exhausted"
    elif budget_exhausted:
        report.stopping_reason = "search_budget_exhausted"
    else:
        report.stopping_reason = "queries_complete"

    eligible = [
        row
        for row in (merged[hid] for hid in unique_order if hid in merged)
        if row.get("status") == "eligible" and row.get("retrieval_distance") is not None
    ]
    report.eligible_unique_chunks = len(eligible)

    factual_texts = [
        str(getattr(chunk, "content", "") or "")
        for chunk in list((getattr(state, "evidence_pool", None) or {}).values())
    ]
    annotated, selection_mode, emb_unavailable, embed_err = select_diverse_relevant(
        eligible,
        factual_texts=factual_texts,
        limit=cap,
        mmr_lambda=lam,
        embed_fn=embed_fn,
    )
    report.selection_mode = selection_mode
    report.selection_policy = selection_mode
    report.embedding_unavailable = bool(emb_unavailable)
    report.retrieval_config["selection_policy"] = selection_mode
    report.retrieval_config["embedding_unavailable"] = bool(emb_unavailable)
    report.retrieval_config["queries_issued"] = queries_issued
    report.retrieval_config["index_chunk_count"] = index_n
    if embed_err:
        report.retrieval_config["embedding_error"] = embed_err
        _event(
            report,
            "embedding_unavailable",
            detail=embed_err,
        )
    elif emb_unavailable:
        _event(
            report,
            "embedding_unavailable",
            detail=(
                "No embedding function; selected by ascending retrieval distance only. "
                "Diversity selection was not performed."
            ),
        )

    selected_rows = [row for row in annotated if row.get("selected")]
    report.considered = []
    for row in annotated:
        rec = {
            "logical_id": intervention_chunk_logical_id(
                document_id=str(document_id or ""),
                page_number=row.get("page_number"),
                chunk_index=row.get("chunk_index"),
                content_hash=str(row.get("content_hash") or ""),
            ),
            "content_hash": row.get("content_hash"),
            "pdf_id": row.get("pdf_id"),
            "page_number": row.get("page_number"),
            "chunk_index": row.get("chunk_index"),
            "retrieval_distance": row.get("retrieval_distance"),
            "query": row.get("query"),
            "supporting_queries": list(row.get("supporting_queries") or []),
            "relevance_component": row.get("relevance_component"),
            "redundancy_component": row.get("redundancy_component"),
            "mmr_score": row.get("mmr_score"),
            "selected": bool(row.get("selected")),
            "selection_reason": row.get("selection_reason"),
            "selection_policy": selection_mode,
            "selection_rank": row.get("selection_rank"),
        }
        report.considered.append(rec)
        _event(
            report,
            "selected" if row.get("selected") else "not_selected",
            logical_id=str(rec["logical_id"]),
            pdf_id=row.get("pdf_id"),
            page_number=row.get("page_number"),
            chunk_index=row.get("chunk_index"),
            content_hash=str(row.get("content_hash") or ""),
            query=str(row.get("query") or ""),
            retrieval_distance=row.get("retrieval_distance"),
            supporting_queries=";".join(list(row.get("supporting_queries") or [])),
            relevance_component=row.get("relevance_component"),
            redundancy_component=row.get("redundancy_component"),
            mmr_score=row.get("mmr_score"),
            selected=bool(row.get("selected")),
            selection_reason=row.get("selection_reason"),
            selection_rank=row.get("selection_rank"),
        )

    for row in selected_rows:
        logical_id = intervention_chunk_logical_id(
            document_id=str(document_id or ""),
            page_number=row.get("page_number"),
            chunk_index=row.get("chunk_index"),
            content_hash=str(row["content_hash"]),
        )
        env = _wrap_candidate(
            logical_id=logical_id,
            text=row["text"],
            metadata=row["metadata"],
            retrieval_distance=row.get("retrieval_distance"),
            query=str(row.get("query") or ""),
            supporting_queries=list(row.get("supporting_queries") or []),
            run_id=str(run_id or ""),
            document_id=str(document_id or ""),
            content_hash=str(row["content_hash"]),
            registry=registry,
            relevance_component=row.get("relevance_component"),
            redundancy_component=row.get("redundancy_component"),
            mmr_score=row.get("mmr_score"),
            selection_rank=parse_selection_rank(row.get("selection_rank")),
        )
        rec = _candidate_record(
            env=env,
            text=row["text"],
            meta=row["metadata"],
            retrieval_distance=row.get("retrieval_distance"),
            query=str(row.get("query") or ""),
            supporting_queries=list(row.get("supporting_queries") or []),
            content_hash=str(row["content_hash"]),
            document_id=str(document_id or ""),
            relevance_component=row.get("relevance_component"),
            redundancy_component=row.get("redundancy_component"),
            mmr_score=row.get("mmr_score"),
            selected=True,
            selection_reason=str(row.get("selection_reason") or "selected"),
            selection_policy=selection_mode,
            selection_rank=parse_selection_rank(row.get("selection_rank")),
        )
        report.candidates.append(rec)
        report.candidates_registered += 1
        _event(
            report,
            "registered",
            logical_id=logical_id,
            artifact_ref=str(env.ref),
            pdf_id=row.get("pdf_id"),
            page_number=row.get("page_number"),
            chunk_index=row.get("chunk_index"),
            content_hash=str(row["content_hash"]),
            query=str(row.get("query") or ""),
            retrieval_distance=row.get("retrieval_distance"),
            supporting_queries=";".join(list(row.get("supporting_queries") or [])),
            relevance_component=row.get("relevance_component"),
            redundancy_component=row.get("redundancy_component"),
            mmr_score=row.get("mmr_score"),
            selected=True,
            selection_reason=str(row.get("selection_reason") or "selected"),
            selection_rank=parse_selection_rank(row.get("selection_rank")),
        )

    if report.candidates_registered == 0 and not report.empty_reason:
        report.empty_reason = EMPTY_BOUNDED_SEARCH_REASON
        detail = (
            "No eligible same-document chunks in the bounded search results "
            f"(stopping_reason={report.stopping_reason}). This does not establish "
            "that the PDF contains no additional evidence."
        )
        if index_exhausted:
            detail = (
                "No eligible extra chunks after an established same-document index "
                "scan. Bounded search is not an absence proof unless index_exhausted."
            )
        _event(report, "empty_pool", detail=detail)
    report.external_add_pool_available = report.candidates_registered > 0
    if report.empty_reason in {
        "retrieval_failure",
        "no_search_backend",
        "frozen_pool_incompatible",
        "frozen_pool_corrupt",
        "preparation_failed",
    }:
        report.source_preparation_status = "failed"
    elif not report.source_preparation_status:
        report.source_preparation_status = "completed"
    return registry, report


def export_selected_add_artifacts(
    explanation_result: Any,
    intervention_registry: Any,
) -> List[Dict[str, Any]]:
    """Exact payloads for ADD artifacts selected during interrogation."""
    if intervention_registry is None:
        return []
    catalog = getattr(intervention_registry, "catalog", None) or {}
    by_ref = {
        str(getattr(env, "ref", "") or ""): env
        for env in catalog.values()
        if getattr(env, "ref", None) is not None
    }
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for rec in list(getattr(explanation_result, "intervention_records", None) or []):
        resolved = getattr(rec, "resolved", None)
        if str(getattr(resolved, "operation", "") or "").upper() != "ADD":
            continue
        ref = getattr(resolved, "target_ref", None)
        key = str(ref or "")
        if not key or key in seen:
            continue
        seen.add(key)
        env = by_ref.get(key)
        if env is None:
            continue
        payload = getattr(env, "payload", None)
        meta = _meta_of(payload)
        text = str(getattr(payload, "content", "") or "")
        env_meta = dict(getattr(env, "metadata", None) or {})
        out.append(
            {
                "artifact_ref": key,
                "logical_id": getattr(ref, "logical_id", ""),
                "artifact_type": str(getattr(env, "artifact_type", "") or "EvidenceChunk"),
                "content": text,
                "content_hash": str(env_meta.get("content_hash") or content_sha256(text)),
                "metadata": meta,
                "retrieval_distance": env_meta.get("retrieval_distance", meta.get("retrieval_distance")),
                "query": env_meta.get("retrieval_query"),
                "candidate_source": CANDIDATE_SOURCE_HYBRID_ADD_FROM_SOURCE,
                "branch_id": str(getattr(env, "branch_id", "") or ""),
                "instance_id": str(getattr(env, "instance_id", "") or ""),
            }
        )
    return out


def format_hybrid_add_source_report_section(report: Mapping[str, Any]) -> List[str]:
    if not report:
        return []
    empty = str(report.get("empty_reason") or "")
    err = str(report.get("retrieval_error") or "")
    cfg = dict(report.get("retrieval_config") or {})
    mismatches = report.get("frozen_mismatch_reasons") or []
    lines = [
        "## Hybrid ADD from source",
        "",
        "Opt-in (`--hybrid-add-from-source`): extra EvidenceChunks retrieved from the "
        "same indexed PDF and registered in a separate intervention catalog for ADD. "
        "Chroma scores are distances (smaller is more similar). Selection balances "
        "query relevance with embedding-cosine redundancy as a proxy; it does not "
        "guarantee semantic novelty or relevance. Candidates still pass contract, "
        "contextual eligibility, sourceability, valid boundary, and confirmed-absence "
        "checks. They are not explicit WHATIF alternatives.",
        "",
        f"- Policy: `{report.get('policy') or HYBRID_ADD_FROM_SOURCE_POLICY}`",
        f"- Selection: `{report.get('selection_mode') or report.get('selection_policy') or ''}`",
        f"- Document ID: `{report.get('document_id') or ''}`",
        f"- LectureBank ID: `{report.get('lecturebank_id') or ''}`",
        f"- Factual run ID: `{report.get('run_id') or ''}`",
        f"- Reused frozen pool: {'yes' if report.get('reused_frozen_pool') else 'no'}",
        f"- Frozen compatible: {report.get('frozen_compatible')}",
        f"- Source preparation status: `{report.get('source_preparation_status') or ''}`",
        f"- External ADD pool available: "
        f"{'yes' if report.get('external_add_pool_available') else 'no'}",
        f"- Stopping reason: `{report.get('stopping_reason') or ''}`",
        f"- Retrieval depth per query: {cfg.get('k_per_query')}",
        f"- Max unique search hits: {cfg.get('max_search_hits')}",
        f"- Final pool limit: {cfg.get('limit')}",
        f"- MMR lambda (relevance vs redundancy): {cfg.get('mmr_lambda')}",
        f"- Unique hits: {report.get('unique_hits', 0)}",
        f"- Unique same-document hits: {report.get('unique_same_document_hits', 0)}",
        f"- Eligible unique chunks: {report.get('eligible_unique_chunks', 0)}",
        f"- Source chunks retrieved: {report.get('source_chunks_retrieved', 0)}",
        f"- Wrong-document rejected: {report.get('wrong_document_rejected', 0)}",
        f"- Missing document id rejected: {report.get('missing_document_id_rejected', 0)}",
        f"- Invalid distance rejected: {report.get('invalid_distance_rejected', 0)}",
        f"- Factual duplicates rejected: {report.get('factual_duplicates_rejected', 0)}",
        f"- Pool duplicates rejected: {report.get('pool_duplicates_rejected', 0)}",
        f"- Candidates registered: {report.get('candidates_registered', 0)}",
        f"- Embedding unavailable: {'yes' if report.get('embedding_unavailable') else 'no'}",
    ]
    if mismatches:
        lines.append(f"- Frozen mismatch reasons: {', '.join(str(x) for x in mismatches)}")
    if err:
        lines.append(f"- Retrieval/preparation error: {err}")
    elif empty:
        lines.append(f"- Empty pool reason: `{empty}`")
        if empty == EMPTY_BOUNDED_SEARCH_REASON:
            lines.append(
                "- Bounded retrieval found no eligible extras; this is not an "
                "exhaustive claim that the PDF contains no additional evidence."
            )
        if empty == "frozen_pool_incompatible":
            lines.append(
                "- Rebuild the pool with `--overwrite` / a fresh run. A failed or "
                "incompatible manifest is not a completed empty pool."
            )
        if empty == "frozen_pool_corrupt":
            lines.append(
                f"- Corrupt frozen pool at `{report.get('frozen_pool_path') or ''}`: "
                f"{report.get('corrupt_reason') or ''}. Factual MAS output was kept. "
                "External ADD was requested but is unavailable. The corrupt file was "
                "not overwritten; rebuild with `--overwrite` or a new output directory."
            )
    if (
        report.get("requested_hybrid_add_from_source")
        and not report.get("external_add_pool_available")
        and str(report.get("source_preparation_status") or "") == "failed"
    ):
        lines.append(
            "- Requested hybrid ADD from source was unavailable; this is not a "
            "successful source-enabled experiment."
        )
    lines.append("")
    return lines


def extract_hybrid_add_source_reports(
    experiment: Mapping[str, Any],
    *,
    lecturebank_id: str = "",
) -> Dict[str, Any]:
    payload = experiment.get("hybrid_add_source") if isinstance(experiment, Mapping) else None
    if not isinstance(payload, Mapping):
        return {
            "hybrid_add_source_events": [],
            "hybrid_add_source_candidates": [],
            "hybrid_add_source_summary": {},
        }
    out = dict(payload)
    lid = str(lecturebank_id or out.get("lecturebank_id") or "")
    if lid and not str(out.get("lecturebank_id") or "").strip():
        out["lecturebank_id"] = lid
    events = []
    for row in list(out.get("events") or []):
        if not isinstance(row, Mapping):
            continue
        item = dict(row)
        if lid and not str(item.get("lecturebank_id") or "").strip():
            item["lecturebank_id"] = lid
        if out.get("document_id") and not str(item.get("document_id") or "").strip():
            item["document_id"] = out.get("document_id")
        events.append(item)
    candidates = []
    for row in list(out.get("candidates") or []):
        if not isinstance(row, Mapping):
            continue
        item = dict(row)
        item.setdefault("lecturebank_id", lid)
        item.setdefault("document_id", out.get("document_id") or "")
        item.setdefault("candidate_source", CANDIDATE_SOURCE_HYBRID_ADD_FROM_SOURCE)
        candidates.append(item)
    return {
        "hybrid_add_source_events": events,
        "hybrid_add_source_candidates": candidates,
        "hybrid_add_source_summary": out,
    }
