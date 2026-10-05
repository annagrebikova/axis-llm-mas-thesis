"""Resolve and optionally download LectureBank PDFs without adding HTTP deps."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mas_langgraph.batch.ids import file_sha256
from mas_langgraph.batch.io import atomic_write_json

PDF_MAGIC = b"%PDF-"
USER_AGENT = "EMaES-LectureBankEval/1.0"
DOWNLOAD_TIMEOUT_S = 60

_KNOWN_PDF_SUBDIRS = (
    "",
    "lecturebank_files",
    "pdfs",
    "pdf",
    "data",
    "files",
    "resources",
)


class PdfSourceError(RuntimeError):
    """A selected LectureBank PDF could not be acquired or validated."""


def document_id_for(lecturebank_id: str) -> str:
    return f"lecturebank_{lecturebank_id}"


def cached_pdf_path(cache_dir: Path, lecturebank_id: str) -> Path:
    return Path(cache_dir) / f"{lecturebank_id}.pdf"


def cached_meta_path(cache_dir: Path, lecturebank_id: str) -> Path:
    return Path(cache_dir) / f"{lecturebank_id}.meta.json"


def is_pdf_bytes(data: bytes) -> bool:
    if not data:
        return False
    return data.startswith(PDF_MAGIC) or data.lstrip(b"\x00 \t\r\n").startswith(PDF_MAGIC)


def is_pdf_file(path: Path) -> bool:
    try:
        with Path(path).open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return False
    return is_pdf_bytes(head)


def filename_matches_lecturebank_id(filename: str, lecturebank_id: str) -> bool:
    """Exact-ID match: ``14`` must not match ``141.pdf``."""
    stem = Path(filename).stem
    ident = str(lecturebank_id)
    if Path(filename).suffix.lower() != ".pdf":
        return False
    pattern = rf"^(?:lecturebank[_-])?{re.escape(ident)}(?:[_-].*)?$"
    return bool(re.fullmatch(pattern, stem, flags=re.IGNORECASE))


def find_local_pdfs(directories: Sequence[Path], lecturebank_id: str) -> List[Path]:
    matches: List[Path] = []
    seen = set()
    for directory in directories:
        root = Path(directory)
        if not root.exists():
            continue
        for path in _iter_pdf_candidates(root):
            key = str(path.resolve())
            if key in seen:
                continue
            if filename_matches_lecturebank_id(path.name, lecturebank_id):
                seen.add(key)
                matches.append(path)
    return matches


def _iter_pdf_candidates(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".pdf":
            yield root
        return
    for sub in _KNOWN_PDF_SUBDIRS:
        folder = root / sub if sub else root
        if not folder.is_dir():
            continue
        try:
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() == ".pdf":
                    yield path
        except OSError:
            continue


def write_source_meta(path: Path, payload: Dict[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def load_source_meta(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def resolve_lecturebank_pdf(
    *,
    lecturebank_id: str,
    url: str = "",
    cache_dir: Path,
    pdf_dir: Optional[Path] = None,
    search_roots: Sequence[Path] = (),
    no_download: bool = False,
    timeout_s: int = DOWNLOAD_TIMEOUT_S,
    copy_local_to_cache: bool = False,
) -> Dict[str, Any]:
    """Return a validated PDF path and source metadata.

    Preference: ``pdf_dir`` in place → existing cache → other local roots in
    place → optional download into ``pdf_cache``. Local ``--pdf-dir`` files are
    not copied unless ``copy_local_to_cache`` is True.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_pdf = cached_pdf_path(cache_dir, lecturebank_id)
    cache_meta = cached_meta_path(cache_dir, lecturebank_id)

    if pdf_dir is not None:
        local_hits = find_local_pdfs([Path(pdf_dir)], lecturebank_id)
        if local_hits:
            return _use_local_pdf(
                local_hits[0],
                lecturebank_id=lecturebank_id,
                url=url,
                cache_pdf=cache_pdf,
                cache_meta=cache_meta,
                copy_to_cache=copy_local_to_cache,
            )

    if cache_pdf.is_file():
        if is_pdf_file(cache_pdf):
            meta = load_source_meta(cache_meta)
            sha = meta.get("sha256") or file_sha256(cache_pdf)
            if not meta.get("sha256"):
                meta = {
                    **meta,
                    "lecturebank_id": lecturebank_id,
                    "path": str(cache_pdf),
                    "sha256": sha,
                    "source": meta.get("source") or "cache",
                }
                write_source_meta(cache_meta, meta)
            return {
                "path": cache_pdf,
                "sha256": sha,
                "source": meta.get("source") or "cache",
                "url": meta.get("url") or url,
                "meta_path": cache_meta,
                "copied": False,
                "source_path": meta.get("source_path") or str(cache_pdf),
                "cached": True,
            }
        raise PdfSourceError(
            f"Cached file for LectureBank ID {lecturebank_id} is not a valid PDF: {cache_pdf}"
        )

    search = [Path(p) for p in search_roots]
    local_hits = find_local_pdfs(search, lecturebank_id)
    if local_hits:
        return _use_local_pdf(
            local_hits[0],
            lecturebank_id=lecturebank_id,
            url=url,
            cache_pdf=cache_pdf,
            cache_meta=cache_meta,
            copy_to_cache=copy_local_to_cache,
        )

    if no_download:
        raise PdfSourceError(
            f"No local PDF for LectureBank ID {lecturebank_id} and --no-download was set. "
            f"Place the file in --pdf-dir as {lecturebank_id}.pdf."
        )
    if not str(url or "").strip():
        raise PdfSourceError(
            f"No local PDF and no URL in LectureBank metadata for ID {lecturebank_id}. "
            f"Place the PDF in --pdf-dir as {lecturebank_id}.pdf."
        )
    download_pdf(url, cache_pdf, timeout_s=timeout_s)
    sha = file_sha256(cache_pdf)
    meta = {
        "lecturebank_id": lecturebank_id,
        "path": str(cache_pdf),
        "sha256": sha,
        "source": "download",
        "url": url,
    }
    write_source_meta(cache_meta, meta)
    return {
        "path": cache_pdf,
        "sha256": sha,
        "source": "download",
        "url": url,
        "meta_path": cache_meta,
        "copied": False,
        "source_path": str(cache_pdf),
        "cached": True,
    }


def _use_local_pdf(
    src: Path,
    *,
    lecturebank_id: str,
    url: str,
    cache_pdf: Path,
    cache_meta: Path,
    copy_to_cache: bool,
) -> Dict[str, Any]:
    src = Path(src).resolve()
    if not is_pdf_file(src):
        raise PdfSourceError(
            f"Local file for LectureBank ID {lecturebank_id} is not a valid PDF: {src}"
        )
    if copy_to_cache:
        _copy_pdf_atomic(src, cache_pdf)
        used = cache_pdf
        copied = True
    else:
        used = src
        copied = False
    sha = file_sha256(used)
    meta = {
        "lecturebank_id": lecturebank_id,
        "path": str(used),
        "source_path": str(src),
        "sha256": sha,
        "source": "local",
        "url": url,
        "copied_to_cache": copied,
    }
    write_source_meta(cache_meta, meta)
    return {
        "path": used,
        "sha256": sha,
        "source": "local",
        "url": url,
        "meta_path": cache_meta,
        "copied": copied,
        "source_path": str(src),
        "cached": copied,
    }


def download_pdf(url: str, dest: Path, *, timeout_s: int = DOWNLOAD_TIMEOUT_S) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            content_type = str(response.headers.get("Content-Type") or "")
            data = response.read()
            final_url = str(getattr(response, "geturl", lambda: url)())
    except HTTPError as exc:
        raise PdfSourceError(
            f"HTTP {exc.code} downloading {url}. Place the PDF manually in --pdf-dir."
        ) from exc
    except URLError as exc:
        raise PdfSourceError(
            f"Could not download {url}: {exc.reason}. Place the PDF manually in --pdf-dir."
        ) from exc
    except TimeoutError as exc:
        raise PdfSourceError(
            f"Timed out downloading {url}. Place the PDF manually in --pdf-dir."
        ) from exc

    if not is_pdf_bytes(data):
        hint = content_type or "unknown content-type"
        raise PdfSourceError(
            f"URL did not return a PDF (got {hint}; magic header missing). "
            f"Final URL: {final_url}. Place the PDF manually in --pdf-dir as "
            f"{dest.name}."
        )
    _write_bytes_atomic(dest, data)
    return dest


def _copy_pdf_atomic(src: Path, dest: Path) -> None:
    data = Path(src).read_bytes()
    if not is_pdf_bytes(data):
        raise PdfSourceError(f"Refusing to copy non-PDF bytes from {src}")
    _write_bytes_atomic(dest, data)


def _write_bytes_atomic(dest: Path, data: bytes) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def local_pdf_available(
    lecturebank_id: str,
    *,
    cache_dir: Optional[Path] = None,
    pdf_dir: Optional[Path] = None,
    search_roots: Sequence[Path] = (),
) -> bool:
    if cache_dir is not None:
        cached = cached_pdf_path(cache_dir, lecturebank_id)
        if cached.is_file() and is_pdf_file(cached):
            return True
    roots: List[Path] = []
    if pdf_dir is not None:
        roots.append(Path(pdf_dir))
    roots.extend(Path(p) for p in search_roots)
    for path in find_local_pdfs(roots, lecturebank_id):
        if is_pdf_file(path):
            return True
    return False


_PDF_SOURCE_TYPES = {
    "local": "local_pdf",
    "cache": "cached_pdf",
    "download": "downloaded_pdf",
}


def _jsonl_match_fields(item: Any) -> Dict[str, Any]:
    return {
        "extraction_jsonl_id": str(getattr(item, "extraction_id", "") or ""),
        "jsonl_match_strategy": str(getattr(item, "match_strategy", "") or ""),
        "jsonl_match_status": str(getattr(item, "match_status", "") or ""),
        "jsonl_title": str(getattr(item, "jsonl_title", "") or ""),
        "jsonl_url": str(getattr(item, "jsonl_url", "") or ""),
        "jsonl_instructor": str(getattr(item, "jsonl_instructor", "") or ""),
        "jsonl_venue": str(getattr(item, "jsonl_venue", "") or ""),
        "jsonl_year": str(getattr(item, "jsonl_year", "") or ""),
    }


def _jsonl_unusable_reason(item: Any) -> str:
    status = str(getattr(item, "status", "") or "missing") if item is not None else "missing"
    if status in {"empty", "invalid_jsonl_record"}:
        return "no usable text"
    if status in {"duplicate", "duplicate_jsonl_id"}:
        return "duplicate id"
    if status in {"malformed"}:
        return "invalid record"
    if status in {"missing", "no_match", ""}:
        return "not found"
    return status.replace("_", " ")


def _jsonl_unusable_message(lecturebank_id: str, item: Any) -> str:
    reason = _jsonl_unusable_reason(item)
    detail = str(getattr(item, "reason", "") or "") if item is not None else ""
    if detail:
        return f"JSONL extraction ID {lecturebank_id} is unavailable ({reason}): {detail}"
    return f"JSONL extraction ID {lecturebank_id} is unavailable ({reason})"


def resolve_lecturebank_source(
    *,
    lecturebank_id: str,
    url: str = "",
    cache_dir: Path,
    pdf_dir: Optional[Path] = None,
    search_roots: Sequence[Path] = (),
    no_download: bool = False,
    jsonl_lookup: Optional[Mapping[str, Any]] = None,
    require_jsonl: bool = False,
) -> Dict[str, Any]:
    """Resolve one LectureBank document: JSONL (if usable) then existing PDF logic."""
    jsonl_note = ""
    item = None
    if jsonl_lookup is not None:
        item = jsonl_lookup.get(str(lecturebank_id))
        if item is not None and getattr(item, "status", None) == "usable" and getattr(item, "record", None):
            meta = (item.record or {}).get("metadata") or {}
            return {
                "source": "extraction_jsonl",
                "input_source_type": "extraction_jsonl",
                "input_source_path": str(item.jsonl_path),
                "jsonl_path": str(item.jsonl_path),
                "path": None,
                "pdf_data": item.record,
                "sha256": item.source_hash,
                "url": url or meta.get("url") or "",
                "source_path": str(item.jsonl_path),
                "num_pages": item.num_pages,
                "nonempty_pages": item.nonempty_pages,
                "jsonl_status": "usable",
                "copied": False,
                "cached": False,
                **_jsonl_match_fields(item),
            }
        if require_jsonl:
            raise PdfSourceError(_jsonl_unusable_message(lecturebank_id, item))
        if item is None or getattr(item, "status", "missing") in {"missing", "no_match"}:
            jsonl_note = "jsonl_miss"
        else:
            jsonl_note = str(item.status)
    elif require_jsonl:
        raise PdfSourceError(_jsonl_unusable_message(lecturebank_id, None))

    info = resolve_lecturebank_pdf(
        lecturebank_id=lecturebank_id,
        url=url,
        cache_dir=cache_dir,
        pdf_dir=pdf_dir,
        search_roots=search_roots,
        no_download=no_download,
    )
    info["input_source_type"] = _PDF_SOURCE_TYPES.get(str(info.get("source") or ""), str(info.get("source") or ""))
    info["input_source_path"] = str(info.get("path") or "")
    info["jsonl_status"] = jsonl_note
    if item is not None:
        info.update(_jsonl_match_fields(item))
    if jsonl_note and jsonl_note != "jsonl_miss":
        info["jsonl_fallback_reason"] = getattr(item, "reason", None) or jsonl_note
    return info


def classify_lecturebank_source(
    *,
    lecturebank_id: str,
    url: str = "",
    cache_dir: Optional[Path] = None,
    pdf_dir: Optional[Path] = None,
    search_roots: Sequence[Path] = (),
    no_download: bool = False,
    jsonl_lookup: Optional[Mapping[str, Any]] = None,
    require_jsonl: bool = False,
) -> Dict[str, str]:
    """Preflight classification without RAG/LLM. Does not download PDFs."""
    if require_jsonl:
        item = jsonl_lookup.get(str(lecturebank_id)) if jsonl_lookup is not None else None
        if (
            item is not None
            and getattr(item, "status", None) == "usable"
            and getattr(item, "record", None)
        ):
            return {
                "status": "available",
                "source": "extraction_jsonl",
                "title": str(getattr(item, "jsonl_title", "") or ""),
                **_jsonl_match_fields(item),
            }
        reason = _jsonl_unusable_reason(item)
        out = {
            "status": "unavailable",
            "source": "extraction_jsonl",
            "reason": reason,
            "jsonl_status": str(getattr(item, "status", "") or "missing") if item is not None else "missing",
        }
        if item is not None:
            out.update(_jsonl_match_fields(item))
            if getattr(item, "jsonl_title", ""):
                out["title"] = str(item.jsonl_title)
        return out
    if jsonl_lookup is not None:
        item = jsonl_lookup.get(str(lecturebank_id))
        if (
            item is not None
            and getattr(item, "status", None) == "usable"
            and (getattr(item, "match_status", "") or "unique_match") == "unique_match"
        ):
            return {
                "status": "available",
                "source": "extraction_jsonl",
                **_jsonl_match_fields(item),
            }
        if item is None or getattr(item, "status", "missing") in {"missing", "no_match"}:
            jsonl_note = "jsonl_miss"
        else:
            jsonl_note = str(item.status)
    else:
        jsonl_note = ""

    if pdf_dir is not None and local_pdf_available(lecturebank_id, pdf_dir=pdf_dir):
        return {"status": "available", "source": "local_pdf", "jsonl_status": jsonl_note}
    if cache_dir is not None:
        cached = cached_pdf_path(cache_dir, lecturebank_id)
        if cached.is_file() and is_pdf_file(cached):
            return {"status": "available", "source": "cached_pdf", "jsonl_status": jsonl_note}
    if search_roots and local_pdf_available(lecturebank_id, search_roots=search_roots):
        return {"status": "available", "source": "local_pdf", "jsonl_status": jsonl_note}
    if str(url or "").strip() and not no_download:
        return {"status": "available", "source": "downloadable_pdf", "jsonl_status": jsonl_note}
    reason = jsonl_note or "no usable source"
    if jsonl_note in {
        "empty",
        "duplicate",
        "malformed",
        "ambiguous_match",
        "metadata_conflict",
        "duplicate_jsonl_id",
        "invalid_jsonl_record",
    }:
        reason = f"no usable source ({jsonl_note})"
    out = {"status": "unavailable", "source": reason, "jsonl_status": jsonl_note}
    if jsonl_lookup is not None:
        item = jsonl_lookup.get(str(lecturebank_id))
        if item is not None:
            out.update(_jsonl_match_fields(item))
    return out
