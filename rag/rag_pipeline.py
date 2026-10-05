"""
RAG Pipeline for Single PDF
Extracts, chunks, embeds, and stores PDF content in a vector index
Tags chunks with source file name and page number for traceability
"""

import hashlib
import json
import re
import pdfplumber
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
import logging
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

# Vector store and embeddings
LANGCHAIN_AVAILABLE = False
Document = None
RecursiveCharacterTextSplitter = None
Chroma = None
HuggingFaceEmbeddings = None

try:
    # Text splitter moved to separate package in newer LangChain versions
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback to older langchain import
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Vector stores and embeddings
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        # Fallback to older langchain imports
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings
    
    # Document schema
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain_core.documents import Document
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    # Create a dummy Document class for type hints
    class Document:
        def __init__(self, page_content: str, metadata: Dict):
            self.page_content = page_content
            self.metadata = metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

_JSONL_ID_RE = re.compile(r'"id"\s*:\s*(?:"([^"]*)"|(-?\d+))')


@dataclass
class ExtractionLookup:
    """Result of matching one LectureBank evaluation ID to a JSONL extraction record."""

    lecturebank_id: str
    status: str
    record: Optional[Dict] = None
    source_hash: str = ""
    jsonl_path: str = ""
    num_pages: int = 0
    nonempty_pages: int = 0
    reason: str = ""
    line_number: Optional[int] = None
    extraction_id: str = ""
    match_strategy: str = ""
    match_status: str = ""
    jsonl_title: str = ""
    jsonl_url: str = ""
    jsonl_instructor: str = ""
    jsonl_venue: str = ""
    jsonl_year: str = ""


@dataclass
class _ExtractionIndexEntry:
    extraction_id: str
    url_norm: str = ""
    url_raw: str = ""
    composite: str = ""
    title: str = ""
    instructor: str = ""
    venue: str = ""
    year: str = ""
    line_number: int = 0
    source_hash: str = ""
    usable: bool = False
    duplicate_id: bool = False


def peek_jsonl_id(line: str) -> Optional[str]:
    """Read the record ``id`` without ``json.loads`` of a possibly huge line."""
    match = _JSONL_ID_RE.search(line)
    if not match:
        return None
    if match.group(1) is not None:
        return str(match.group(1)).strip()
    return str(match.group(2)).strip()


def extraction_has_usable_pages(data: Dict) -> bool:
    """True when ``text.pages_text`` contains at least one non-empty page."""
    text = data.get("text") if isinstance(data.get("text"), dict) else {}
    pages = text.get("pages_text") or []
    if not isinstance(pages, list):
        return False
    for page in pages:
        if isinstance(page, dict) and str(page.get("text") or "").strip():
            return True
    return False


def normalize_extraction_pdf_data(data: Dict, *, source_file: Optional[str] = None) -> Dict:
    """Authoritative JSONL → ``pdf_data`` shape used by ``create_chunks_with_metadata``.

    Keeps page-level ``pages_text``. Drops bulky fields (images/tables/ocr).
    Reconstructs ``full_text`` from pages so ``chunk_by_page=False`` stays page-aligned.
    """
    ident = str(data.get("id") or "").strip()
    text = data.get("text") if isinstance(data.get("text"), dict) else {}
    pages_raw = text.get("pages_text") or []
    pages: List[Dict] = []
    empty_pages: List = []
    parts: List[str] = []
    if isinstance(pages_raw, list):
        for page in pages_raw:
            if not isinstance(page, dict):
                continue
            page_num = page.get("page")
            page_text = str(page.get("text") or "")
            pages.append({"page": page_num, "text": page_text, "length": len(page_text)})
            parts.append(page_text)
            if not page_text.strip():
                empty_pages.append(page_num)
    full_text = "\n".join(parts).strip()
    nonempty = len(pages) - len(empty_pages)
    stats_in = data.get("statistics") if isinstance(data.get("statistics"), dict) else {}
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return {
        "id": ident,
        "source_file": source_file or data.get("source_file") or (f"{ident}.pdf" if ident else "unknown.pdf"),
        "metadata": {
            "title": meta.get("title"),
            "instructor": meta.get("instructor"),
            "year": meta.get("year"),
            "url": meta.get("url"),
            "venue": meta.get("venue"),
            "topic": meta.get("topic"),
        },
        "text": {
            "full_text": full_text,
            "pages_text": pages,
            "num_pages": len(pages),
        },
        "statistics": {
            "has_text": nonempty > 0,
            "num_pages": len(pages) or int(stats_in.get("num_pages") or text.get("num_pages") or 0),
            "nonempty_pages": nonempty,
            "empty_pages": empty_pages,
            "text_length": len(full_text),
            "requires_ocr": nonempty == 0 or len(full_text) == 0,
        },
    }


def load_selected_extractions(
    jsonl_path: Union[str, Path],
    wanted_ids: Iterable[str],
) -> Dict[str, ExtractionLookup]:
    """Single-pass lookup of selected JSONL records. Does not parse unselected rows.

    Duplicate IDs are marked ``duplicate`` and are not treated as usable.
    Malformed unselected (or unidentifiable) rows are skipped.
    """
    path = Path(jsonl_path)
    wanted = {str(ident).strip() for ident in wanted_ids if str(ident).strip()}
    results: Dict[str, ExtractionLookup] = {
        ident: ExtractionLookup(
            lecturebank_id=ident,
            status="missing",
            jsonl_path=str(path),
            reason="ID not present in extraction JSONL",
        )
        for ident in wanted
    }
    if not wanted:
        return results
    if not path.is_file():
        for ident in wanted:
            results[ident].reason = f"JSONL file not found: {path}"
        return results

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            peek_id = peek_jsonl_id(line)
            if peek_id is None or peek_id not in wanted:
                continue
            item = results[peek_id]
            if item.status != "missing":
                item.status = "duplicate"
                item.match_status = "duplicate_jsonl_id"
                item.extraction_id = peek_id
                item.reason = (
                    f"Duplicate JSONL records for ID {peek_id} "
                    f"(first at line {item.line_number}, also line {line_no})"
                )
                item.record = None
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                results[peek_id] = ExtractionLookup(
                    lecturebank_id=peek_id,
                    status="malformed",
                    jsonl_path=str(path),
                    reason=f"Malformed JSONL at line {line_no}: {exc}",
                    line_number=line_no,
                    extraction_id=peek_id,
                    match_status="invalid_jsonl_record",
                )
                continue
            record_id = str(data.get("id") or "").strip()
            if record_id != peek_id:
                results[peek_id] = ExtractionLookup(
                    lecturebank_id=peek_id,
                    status="malformed",
                    jsonl_path=str(path),
                    reason=(
                        f"JSONL line {line_no} peek id {peek_id!r} != parsed id {record_id!r}"
                    ),
                    line_number=line_no,
                    extraction_id=peek_id,
                    match_status="invalid_jsonl_record",
                )
                continue
            source_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
            normalized = normalize_extraction_pdf_data(data)
            if not extraction_has_usable_pages(normalized):
                stats = normalized.get("statistics") or {}
                results[peek_id] = ExtractionLookup(
                    lecturebank_id=peek_id,
                    status="empty",
                    record=None,
                    source_hash=source_hash,
                    jsonl_path=str(path),
                    num_pages=int(stats.get("num_pages") or 0),
                    nonempty_pages=0,
                    reason="JSONL record has no usable page text",
                    line_number=line_no,
                    extraction_id=peek_id,
                    match_status="invalid_jsonl_record",
                    **_extraction_meta_fields(normalized),
                )
                continue
            stats = normalized.get("statistics") or {}
            results[peek_id] = ExtractionLookup(
                lecturebank_id=peek_id,
                status="usable",
                record=normalized,
                source_hash=source_hash,
                jsonl_path=str(path),
                num_pages=int(stats.get("num_pages") or 0),
                nonempty_pages=int(stats.get("nonempty_pages") or 0),
                line_number=line_no,
                extraction_id=peek_id,
                match_status="unique_match",
                **_extraction_meta_fields(normalized),
            )
    return results


def _extraction_meta_fields(data: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    meta = data.get("metadata") if isinstance(data, Mapping) and isinstance(data.get("metadata"), dict) else {}
    return {
        "jsonl_title": str(meta.get("title") or ""),
        "jsonl_url": str(meta.get("url") or ""),
        "jsonl_instructor": str(meta.get("instructor") or ""),
        "jsonl_venue": str(meta.get("venue") or ""),
        "jsonl_year": str(meta.get("year") or ""),
    }


def select_first_usable_jsonl_ids(
    jsonl_path: Union[str, Path],
    count: int,
) -> List[str]:
    """Return the first ``count`` unique JSONL IDs with usable page text, in file order.

    Duplicate IDs after the first occurrence are skipped, not silently substituted.
    """
    n = int(count)
    if n < 1:
        raise ValueError("--first-jsonl must be a positive integer.")
    path = Path(jsonl_path)
    selected: List[str] = []
    seen: set[str] = set()
    if not path.is_file():
        return selected
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            peek_id = peek_jsonl_id(line)
            if peek_id is None or peek_id in seen:
                continue
            seen.add(peek_id)
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(data.get("id") or "").strip() != peek_id:
                continue
            if extraction_has_usable_pages(data):
                selected.append(peek_id)
                if len(selected) >= n:
                    break
    return selected


def normalize_source_url(url: Any) -> str:
    """Conservative URL normalization for exact document matching."""
    text = str(url or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()
    path = parts.path or ""
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def normalize_meta_token(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def metadata_composite_key(
    title: Any = "",
    instructor: Any = "",
    venue: Any = "",
    year: Any = "",
) -> str:
    parts = [normalize_meta_token(v) for v in (title, instructor, venue, year)]
    if any(not part for part in parts):
        return ""
    return "\x1f".join(parts)


def _lb_field(record: Any, name: str) -> str:
    if record is None:
        return ""
    if isinstance(record, Mapping):
        return str(record.get(name) or "")
    return str(getattr(record, name, "") or "")


def _build_extraction_index(jsonl_path: Path) -> Dict[str, _ExtractionIndexEntry]:
    """Compact metadata for every JSONL record. Does not retain page text."""
    by_id: Dict[str, _ExtractionIndexEntry] = {}
    if not jsonl_path.is_file():
        return by_id
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            peek_id = peek_jsonl_id(line)
            if peek_id is None:
                continue
            if peek_id in by_id:
                by_id[peek_id].duplicate_id = True
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                by_id[peek_id] = _ExtractionIndexEntry(
                    extraction_id=peek_id,
                    line_number=line_no,
                    usable=False,
                )
                continue
            record_id = str(data.get("id") or "").strip()
            if record_id != peek_id:
                continue
            usable = extraction_has_usable_pages(data)
            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            title = str(meta.get("title") or "")
            instructor = str(meta.get("instructor") or "")
            venue = str(meta.get("venue") or "")
            year = str(meta.get("year") or "")
            url_raw = str(meta.get("url") or "")
            del data
            by_id[peek_id] = _ExtractionIndexEntry(
                extraction_id=peek_id,
                url_norm=normalize_source_url(url_raw),
                url_raw=url_raw,
                composite=metadata_composite_key(title, instructor, venue, year),
                title=title,
                instructor=instructor,
                venue=venue,
                year=year,
                line_number=line_no,
                source_hash=hashlib.sha256(line.encode("utf-8")).hexdigest(),
                usable=usable,
            )
    return by_id


def _lookup_from_entry(
    lecturebank_id: str,
    entry: _ExtractionIndexEntry,
    *,
    jsonl_path: str,
    status: str,
    match_status: str,
    match_strategy: str = "",
    reason: str = "",
) -> ExtractionLookup:
    return ExtractionLookup(
        lecturebank_id=lecturebank_id,
        status=status,
        jsonl_path=jsonl_path,
        reason=reason,
        line_number=entry.line_number,
        extraction_id=entry.extraction_id,
        match_strategy=match_strategy,
        match_status=match_status,
        jsonl_title=entry.title,
        jsonl_url=entry.url_raw,
        jsonl_instructor=entry.instructor,
        jsonl_venue=entry.venue,
        jsonl_year=entry.year,
        source_hash=entry.source_hash,
    )


def match_lecturebank_extractions(
    jsonl_path: Union[str, Path],
    lecturebank_records: Sequence[Any],
) -> Dict[str, ExtractionLookup]:
    """Match selected LectureBank records to JSONL text by URL/metadata, not numeric ID.

    Numeric JSONL ``id`` equality is never sufficient on its own.
    """
    path = Path(jsonl_path)
    records = [rec for rec in lecturebank_records if rec is not None]
    results: Dict[str, ExtractionLookup] = {}
    if not records:
        return results
    if not path.is_file():
        for rec in records:
            lb_id = _lb_field(rec, "lecturebank_id") or _lb_field(rec, "id")
            results[lb_id] = ExtractionLookup(
                lecturebank_id=lb_id,
                status="missing",
                match_status="no_match",
                jsonl_path=str(path),
                reason=f"JSONL file not found: {path}",
            )
        return results

    by_id = _build_extraction_index(path)
    by_url: Dict[str, List[_ExtractionIndexEntry]] = {}
    by_composite: Dict[str, List[_ExtractionIndexEntry]] = {}
    for entry in by_id.values():
        if entry.url_norm:
            by_url.setdefault(entry.url_norm, []).append(entry)
        if entry.composite:
            by_composite.setdefault(entry.composite, []).append(entry)

    pending_load: Dict[str, str] = {}
    for rec in records:
        lb_id = _lb_field(rec, "lecturebank_id") or _lb_field(rec, "id")
        results[lb_id] = _match_one_lecturebank(
            lb_id,
            rec,
            by_id=by_id,
            by_url=by_url,
            by_composite=by_composite,
            jsonl_path=str(path),
        )
        item = results[lb_id]
        if item.status == "usable" and item.extraction_id:
            pending_load[lb_id] = item.extraction_id

    if pending_load:
        loaded = load_selected_extractions(path, set(pending_load.values()))
        for lb_id, ext_id in pending_load.items():
            item = results[lb_id]
            full = loaded.get(ext_id)
            if full is None or full.status == "missing":
                item.status = "invalid_jsonl_record"
                item.match_status = "invalid_jsonl_record"
                item.reason = "Matched JSONL record could not be loaded"
                item.record = None
                continue
            if full.status == "duplicate":
                item.status = "duplicate"
                item.match_status = "duplicate_jsonl_id"
                item.reason = full.reason or "Duplicate JSONL records for this extraction ID"
                item.record = None
                continue
            if full.status != "usable" or full.record is None:
                item.status = "invalid_jsonl_record"
                item.match_status = full.status or "invalid_jsonl_record"
                item.reason = full.reason or "Matched JSONL record has no usable page text"
                item.record = None
                continue
            item.record = full.record
            item.source_hash = full.source_hash
            item.num_pages = full.num_pages
            item.nonempty_pages = full.nonempty_pages
            item.line_number = full.line_number
    return results


def _match_one_lecturebank(
    lb_id: str,
    rec: Any,
    *,
    by_id: Mapping[str, _ExtractionIndexEntry],
    by_url: Mapping[str, List[_ExtractionIndexEntry]],
    by_composite: Mapping[str, List[_ExtractionIndexEntry]],
    jsonl_path: str,
) -> ExtractionLookup:
    same_id = by_id.get(str(lb_id))
    url_n = normalize_source_url(_lb_field(rec, "url"))
    if url_n:
        hits = list(by_url.get(url_n) or [])
        return _finish_unique_hits(
            lb_id,
            hits,
            jsonl_path=jsonl_path,
            strategy="normalized_url",
            same_id=same_id,
        )
    composite = metadata_composite_key(
        _lb_field(rec, "title"),
        _lb_field(rec, "instructor"),
        _lb_field(rec, "venue"),
        _lb_field(rec, "year"),
    )
    if composite:
        hits = list(by_composite.get(composite) or [])
        return _finish_unique_hits(
            lb_id,
            hits,
            jsonl_path=jsonl_path,
            strategy="metadata_composite",
            same_id=same_id,
        )
    if same_id is not None:
        return _lookup_from_entry(
            lb_id,
            same_id,
            jsonl_path=jsonl_path,
            status="metadata_conflict",
            match_status="metadata_conflict",
            reason=(
                "JSONL extraction id equals LectureBank id but URL/metadata "
                "do not confirm the same document"
            ),
        )
    return ExtractionLookup(
        lecturebank_id=lb_id,
        status="missing",
        match_status="no_match",
        jsonl_path=jsonl_path,
        reason="No JSONL record matched LectureBank URL or metadata",
    )


def _finish_unique_hits(
    lb_id: str,
    hits: Sequence[_ExtractionIndexEntry],
    *,
    jsonl_path: str,
    strategy: str,
    same_id: Optional[_ExtractionIndexEntry],
) -> ExtractionLookup:
    if len(hits) > 1:
        return ExtractionLookup(
            lecturebank_id=lb_id,
            status="ambiguous_match",
            match_status="ambiguous_match",
            match_strategy=strategy,
            jsonl_path=jsonl_path,
            reason=f"Ambiguous JSONL {strategy} match ({len(hits)} records)",
            extraction_id=",".join(h.extraction_id for h in hits),
        )
    if len(hits) == 1:
        entry = hits[0]
        if entry.duplicate_id:
            return _lookup_from_entry(
                lb_id,
                entry,
                jsonl_path=jsonl_path,
                status="duplicate",
                match_status="duplicate_jsonl_id",
                match_strategy=strategy,
                reason="Duplicate JSONL records share this extraction ID",
            )
        if not entry.usable:
            return _lookup_from_entry(
                lb_id,
                entry,
                jsonl_path=jsonl_path,
                status="empty",
                match_status="invalid_jsonl_record",
                match_strategy=strategy,
                reason="Matched JSONL record has no usable page text",
            )
        return _lookup_from_entry(
            lb_id,
            entry,
            jsonl_path=jsonl_path,
            status="usable",
            match_status="unique_match",
            match_strategy=strategy,
        )
    if same_id is not None:
        return _lookup_from_entry(
            lb_id,
            same_id,
            jsonl_path=jsonl_path,
            status="metadata_conflict",
            match_status="metadata_conflict",
            reason=(
                "JSONL extraction id equals LectureBank id but URL/metadata "
                "do not confirm the same document"
            ),
        )
    return ExtractionLookup(
        lecturebank_id=lb_id,
        status="missing",
        match_status="no_match",
        jsonl_path=jsonl_path,
        reason="No JSONL record matched LectureBank URL or metadata",
    )


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk"""
    source_file: str
    page_number: int
    chunk_index: int
    pdf_id: Optional[str] = None
    title: Optional[str] = None


class RAGPipeline:
    """RAG Pipeline for processing PDFs"""
    
    def __init__(
        self,
        vector_store_path: str = "vector_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings: Optional[object] = None,
    ):
        """
        Initialize RAG Pipeline
        
        Args:
            vector_store_path: Path to store vector database
            embedding_model: HuggingFace model name for embeddings
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            embeddings: Optional pre-built embedding function (tests / reuse)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain and dependencies not installed. Run: pip install langchain chromadb sentence-transformers")
        
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self._active_collection: Optional[str] = None
        
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            # Initialize embeddings
            logging.info(f"Loading embedding model: {embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'}  # Use 'cuda' if GPU available
            )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Vector store (will be initialized when adding documents)
        self.vector_store = None
    
    def load_pdf_from_file(
        self,
        pdf_path: str,
        document_id: Optional[str] = None,
    ) -> Dict:
        """
        Load and extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            document_id: Optional explicit collection/document ID. When omitted,
                keep the historical filename-prefix ID (e.g. ``141_*.pdf`` → ``141``)
                so existing single-PDF commands remain compatible.
            
        Returns:
            Dictionary with extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logging.info(f"Loading PDF: {pdf_path}")
        
        full_text = ""
        pages_text = []
        empty_pages: List[int] = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"
                pages_text.append({
                    'page': page_num,
                    'text': page_text,
                    'length': len(page_text)
                })
                if not page_text.strip():
                    empty_pages.append(page_num)
        
        if document_id:
            pdf_id = str(document_id)
        else:
            # Extract PDF ID from filename (e.g., "141_Unsupervised Learning.pdf" -> "141")
            pdf_id = pdf_path.stem.split('_')[0] if '_' in pdf_path.stem else pdf_path.stem
        
        nonempty_pages = len(pages_text) - len(empty_pages)
        return {
            'id': pdf_id,
            'source_file': pdf_path.name,
            'text': {
                'full_text': full_text.strip(),
                'pages_text': pages_text
            },
            'statistics': {
                'num_pages': len(pages_text),
                'nonempty_pages': nonempty_pages,
                'empty_pages': empty_pages,
                'text_length': len(full_text.strip()),
                'requires_ocr': nonempty_pages == 0 or len(full_text.strip()) == 0,
            }
        }
    
    def load_pdf_from_jsonl(self, jsonl_path: str, pdf_id: str) -> Optional[Dict]:
        """
        Load PDF data from jsonl extraction results
        
        Args:
            jsonl_path: Path to jsonl file
            pdf_id: PDF ID to load
            
        Returns:
            Dictionary with extracted data or None if not found
        """
        lookups = load_selected_extractions(jsonl_path, [str(pdf_id)])
        item = lookups.get(str(pdf_id))
        if item is None or item.status == "missing":
            logging.warning(f"PDF {pdf_id} not found in {jsonl_path}")
            return None
        if item.status == "malformed":
            logging.warning(item.reason or f"Malformed JSONL record for {pdf_id}")
            return None
        if item.record is None:
            logging.warning(item.reason or f"PDF {pdf_id} has no usable JSONL text")
            return None
        if item.status == "duplicate":
            logging.warning(item.reason or f"Duplicate JSONL records for {pdf_id}")
        if "source_file" not in item.record:
            item.record["source_file"] = f"{pdf_id}.pdf"
        return item.record
    
    def create_chunks_with_metadata(
        self,
        pdf_data: Dict,
        chunk_by_page: bool = True
    ) -> List[Document]:
        """
        Create text chunks with metadata
        
        Args:
            pdf_data: Dictionary with PDF text data
            chunk_by_page: If True, chunk each page separately. If False, chunk full text.
            
        Returns:
            List of LangChain Document objects with metadata
        """
        source_file = pdf_data.get('source_file', 'unknown.pdf')
        pdf_id = pdf_data.get('id', 'unknown')
        metadata_dict = pdf_data.get('metadata', {})
        title = metadata_dict.get('title', source_file)
        
        documents = []
        
        if chunk_by_page:
            # Chunk each page separately to preserve page numbers
            pages_text = pdf_data['text'].get('pages_text', [])
            
            for page_data in pages_text:
                page_num = page_data.get('page', 0)
                page_text = page_data.get('text', '').strip()
                
                if not page_text:
                    continue
                
                # Split page text into chunks if it's too long
                if len(page_text) > self.chunk_size:
                    page_chunks = self.text_splitter.split_text(page_text)
                    for chunk_idx, chunk_text in enumerate(page_chunks):
                        metadata = {
                            'source_file': source_file,
                            'page_number': page_num,
                            'chunk_index': chunk_idx,
                            'pdf_id': pdf_id,
                            'title': title
                        }
                        documents.append(Document(
                            page_content=chunk_text,
                            metadata=metadata
                        ))
                else:
                    # Use entire page as single chunk
                    metadata = {
                        'source_file': source_file,
                        'page_number': page_num,
                        'chunk_index': 0,
                        'pdf_id': pdf_id,
                        'title': title
                    }
                    documents.append(Document(
                        page_content=page_text,
                        metadata=metadata
                    ))
        else:
            # Chunk full text across pages
            full_text = pdf_data['text'].get('full_text', '').strip()
            if not full_text:
                return documents
            
            chunks = self.text_splitter.split_text(full_text)
            
            # Try to assign page numbers based on text position
            pages_text = pdf_data['text'].get('pages_text', [])
            page_boundaries = []
            cumulative_length = 0
            for page_data in pages_text:
                page_boundaries.append({
                    'page': page_data.get('page', 0),
                    'start': cumulative_length,
                    'end': cumulative_length + len(page_data.get('text', ''))
                })
                cumulative_length += len(page_data.get('text', '')) + 1  # +1 for newline
            
            for chunk_idx, chunk_text in enumerate(chunks):
                # Find which page this chunk belongs to
                chunk_start = full_text.find(chunk_text)
                page_num = 1
                for boundary in page_boundaries:
                    if boundary['start'] <= chunk_start < boundary['end']:
                        page_num = boundary['page']
                        break
                
                metadata = {
                    'source_file': source_file,
                    'page_number': page_num,
                    'chunk_index': chunk_idx,
                    'pdf_id': pdf_id,
                    'title': title
                }
                documents.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))
        
        logging.info(f"Created {len(documents)} chunks from PDF {pdf_id}")
        return documents
    
    def add_pdf_to_index(
        self,
        pdf_path: Optional[str] = None,
        pdf_id: Optional[str] = None,
        jsonl_path: str = "../data/pdf_extraction_results.jsonl",
        chunk_by_page: bool = True,
        collection_name: Optional[str] = None,
        document_id: Optional[str] = None,
        persist_directory: Optional[Union[str, Path]] = None,
        replace_collection: bool = False,
    ):
        """
        Add PDF to vector index
        
        Args:
            pdf_path: Path to PDF file (if loading from file)
            pdf_id: PDF ID (if loading from jsonl)
            jsonl_path: Path to jsonl file with extracted data
            chunk_by_page: Whether to chunk by page or full text
            collection_name: Name for Chroma collection (default: pdf_id or 'default')
            document_id: Explicit document/collection ID when loading from a file
            persist_directory: Isolated store path. Defaults to ``self.vector_store_path``.
            replace_collection: Drop the target collection before writing.
        """
        # Load PDF data
        if pdf_path:
            pdf_data = self.load_pdf_from_file(pdf_path, document_id=document_id)
        elif pdf_id:
            pdf_data = self.load_pdf_from_jsonl(jsonl_path, pdf_id)
            if pdf_data is None:
                raise ValueError(f"Could not load PDF {pdf_id} from {jsonl_path}")
        else:
            raise ValueError("Must provide either pdf_path or pdf_id")

        if document_id:
            pdf_data['id'] = str(document_id)
        
        # Create chunks
        documents = self.create_chunks_with_metadata(pdf_data, chunk_by_page=chunk_by_page)
        
        if not documents:
            raise ValueError("No documents created from PDF")
        
        # Determine collection name
        if collection_name is None:
            collection_name = str(pdf_data.get('id', 'default'))
        target_path = Path(persist_directory) if persist_directory else self.vector_store_path
        target_path.mkdir(parents=True, exist_ok=True)

        self._bind_collection(
            persist_directory=target_path,
            collection_name=collection_name,
            documents=documents,
            replace_collection=replace_collection,
        )
        
        logging.info(f"✓ Added {len(documents)} chunks to vector index")
        logging.info(f"  Source: {pdf_data.get('source_file', 'unknown')}")
        logging.info(f"  Pages: {pdf_data['statistics'].get('num_pages', 0)}")
        logging.info(f"  Collection: {collection_name}")
        logging.info(f"  Persist: {target_path}")
        
        return pdf_data

    def _bind_collection(
        self,
        *,
        persist_directory: Path,
        collection_name: str,
        documents: Optional[List] = None,
        replace_collection: bool = False,
    ) -> None:
        """Open or create exactly one collection at ``persist_directory``.

        Never appends into a previously bound collection when the path or
        collection name changes (that was the cross-presentation leak).
        """
        persist_directory = Path(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)
        same_store = (
            self.vector_store is not None
            and Path(self.vector_store_path).resolve() == persist_directory.resolve()
            and self._active_collection == collection_name
        )
        if replace_collection or not same_store:
            if replace_collection:
                self._delete_collection(persist_directory, collection_name)
            self.vector_store_path = persist_directory
            if documents:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(persist_directory),
                    collection_name=collection_name,
                )
            else:
                self.vector_store = Chroma(
                    persist_directory=str(persist_directory),
                    embedding_function=self.embeddings,
                    collection_name=collection_name,
                )
            self._active_collection = collection_name
            return
        if documents:
            logging.info(
                "Adding documents to bound collection %s at %s",
                collection_name,
                persist_directory,
            )
            self.vector_store.add_documents(documents)

    def _delete_collection(self, persist_directory: Path, collection_name: str) -> None:
        persist_directory = Path(persist_directory)
        if self.vector_store is not None:
            self.close()
        if not persist_directory.exists():
            return
        client = None
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(persist_directory))
            client.delete_collection(collection_name)
        except Exception:
            logging.debug(
                "Could not delete collection %s at %s; recreating.",
                collection_name,
                persist_directory,
                exc_info=True,
            )
        finally:
            if client is not None:
                for closer in ("stop", "reset"):
                    try:
                        method = getattr(client, closer, None)
                        if callable(method) and closer == "stop":
                            method()
                    except Exception:
                        pass
            self.vector_store = None
            self._active_collection = None
            import gc

            gc.collect()
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search the vector index
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters (e.g., {'page_number': 5})
            
        Returns:
            List of relevant documents with metadata
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add PDFs first.")
        
        # Perform similarity search
        if filter_dict:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return results
    
    def load_existing_index(self, collection_name: str = "default"):
        """
        Load existing vector index
        
        Args:
            collection_name: Name of the collection to load
        """
        if not self.vector_store_path.exists():
            raise FileNotFoundError(f"Vector store not found at: {self.vector_store_path}")
        
        logging.info(f"Loading existing vector store from: {self.vector_store_path}")
        self.vector_store = Chroma(
            persist_directory=str(self.vector_store_path),
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        self._active_collection = collection_name
        
        # Get collection info
        collection = self.vector_store._collection
        count = collection.count()
        logging.info(f"Loaded {count} documents from collection: {collection_name}")
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the vector index"""
        if self.vector_store is None:
            return {"status": "No index loaded"}
        
        collection = self.vector_store._collection
        count = collection.count()
        
        # Get unique source files and pages
        results = collection.get()
        if results and 'metadatas' in results:
            metadatas = results['metadatas']
            source_files = set(m.get('source_file', 'unknown') for m in metadatas)
            pages = set(m.get('page_number', 0) for m in metadatas)
            
            return {
                "total_chunks": count,
                "source_files": list(source_files),
                "num_source_files": len(source_files),
                "pages_covered": sorted(list(pages)),
                "num_pages": len(pages)
            }
        
        return {"total_chunks": count}

    def chunk_metadatas(self) -> List[Dict]:
        """Return all chunk metadatas from the bound collection."""
        if self.vector_store is None:
            return []
        results = self.vector_store._collection.get()
        return list(results.get("metadatas") or []) if results else []

    def assert_exclusive_document(self, document_id: str) -> None:
        """Fail if the bound collection contains chunks from another document."""
        foreign = []
        for meta in self.chunk_metadatas():
            owner = str(meta.get("pdf_id") or "")
            if owner and owner != str(document_id):
                foreign.append(owner)
        if foreign:
            raise ValueError(
                f"Collection {self._active_collection!r} contains chunks from "
                f"other documents: {sorted(set(foreign))}"
            )

    def close(self) -> None:
        """Release the Chroma client so persist directories can be deleted on Windows."""
        store = self.vector_store
        self.vector_store = None
        self._active_collection = None
        if store is None:
            return
        try:
            persist = getattr(store, "persist", None)
            if callable(persist):
                persist()
        except Exception:
            pass
        for attr in ("_client", "_collection"):
            try:
                setattr(store, attr, None)
            except Exception:
                pass
        try:
            del store
        except Exception:
            pass
        import gc

        gc.collect()


def main():
    """Test the RAG pipeline with PDF 141"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Pipeline for PDF Processing')
    parser.add_argument('--pdf-path', type=str, help='Path to PDF file')
    parser.add_argument('--pdf-id', type=str, default='141', help='PDF ID from jsonl (default: 141)')
    parser.add_argument('--jsonl', type=str, default='../data/pdf_extraction_results.jsonl', help='Path to jsonl file')
    parser.add_argument('--query', type=str, help='Query to search')
    parser.add_argument('--k', type=int, default=5, help='Number of search results')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size in characters')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap in characters')
    parser.add_argument('--no-page-chunking', action='store_true', help='Chunk full text instead of by page')
    parser.add_argument('--load-index', action='store_true', help='Load existing index instead of creating new')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Load or create index
    if args.load_index:
        pipeline.load_existing_index(collection_name=args.pdf_id)
    else:
        # Add PDF to index
        if args.pdf_path:
            pdf_data = pipeline.add_pdf_to_index(
                pdf_path=args.pdf_path,
                chunk_by_page=not args.no_page_chunking
            )
        else:
            pdf_data = pipeline.add_pdf_to_index(
                pdf_id=args.pdf_id,
                jsonl_path=args.jsonl,
                chunk_by_page=not args.no_page_chunking
            )
    
    # Print stats
    stats = pipeline.get_index_stats()
    print("\n" + "="*60)
    print("INDEX STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Search if query provided
    if args.query:
        print("\n" + "="*60)
        print(f"SEARCH RESULTS for: '{args.query}'")
        print("="*60)
        
        results = pipeline.search(args.query, k=args.k)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n--- Result {i} (Score: {score:.4f}) ---")
            print(f"Source: {doc.metadata.get('source_file', 'unknown')}")
            print(f"Page: {doc.metadata.get('page_number', 'unknown')}")
            print(f"Chunk: {doc.metadata.get('chunk_index', 'unknown')}")
            print(f"Title: {doc.metadata.get('title', 'unknown')}")
            print(f"\nContent:\n{doc.page_content[:300]}...")
    
    print("\n" + "="*60)
    print("Pipeline ready for queries!")
    print("="*60)


if __name__ == '__main__':
    main()

