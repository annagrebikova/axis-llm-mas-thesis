"""Local LectureBank metadata: TSV parsing and exact ID selection."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REQUIRED_EVAL_ID_COUNT = 10

_HEADER_ALIASES = {
    "id": "id",
    "lecturebank_id": "id",
    "resource_id": "id",
    "title": "title",
    "instructor": "instructor",
    "author": "instructor",
    "topic": "topic_id",
    "topic_id": "topic_id",
    "topicid": "topic_id",
    "url": "url",
    "link": "url",
    "venue": "venue",
    "year": "year",
    "school": "venue",
}

_HEADERLESS_FIELDS = ("id", "title", "url", "topic_id", "year", "instructor", "topic_code", "venue")


class SelectionError(ValueError):
    """Invalid LectureBank ID selection."""


@dataclass(frozen=True)
class LectureBankRecord:
    lecturebank_id: str
    title: str = ""
    topic_id: str = ""
    topic_name: str = ""
    instructor: str = ""
    venue: str = ""
    year: str = ""
    url: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)

    def search_blob(self) -> str:
        parts = [
            self.lecturebank_id,
            self.title,
            self.topic_id,
            self.topic_name,
            self.instructor,
            self.venue,
            self.year,
            self.url,
        ]
        return " ".join(str(p or "") for p in parts).lower()

    def to_meta_dict(self) -> Dict[str, str]:
        return {
            "lecturebank_id": self.lecturebank_id,
            "title": self.title,
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "instructor": self.instructor,
            "venue": self.venue,
            "year": self.year,
            "url": self.url,
        }


def lecturebank_record_from_extraction(ident: str, item: Any) -> LectureBankRecord:
    """Build evaluation metadata from a JSONL extraction record (authoritative mode)."""
    meta: Mapping[str, Any] = {}
    record = getattr(item, "record", None)
    if isinstance(record, Mapping) and isinstance(record.get("metadata"), dict):
        meta = record.get("metadata") or {}

    def _field(jsonl_attr: str, key: str) -> str:
        return str(getattr(item, jsonl_attr, "") or meta.get(key) or "")

    return LectureBankRecord(
        lecturebank_id=str(ident),
        title=_field("jsonl_title", "title"),
        instructor=_field("jsonl_instructor", "instructor"),
        venue=_field("jsonl_venue", "venue"),
        year=_field("jsonl_year", "year"),
        url=_field("jsonl_url", "url"),
    )


def catalog_from_extraction_lookups(
    jsonl_path: Path,
    lookups: Mapping[str, Any],
) -> LectureBankCatalog:
    """Synthetic catalog keyed by JSONL extraction ID. Does not read lecturebank.tsv."""
    records: Dict[str, LectureBankRecord] = {}
    for ident, item in (lookups or {}).items():
        if item is None or getattr(item, "status", "") == "missing":
            continue
        records[str(ident)] = lecturebank_record_from_extraction(str(ident), item)
    path = Path(jsonl_path)
    return LectureBankCatalog(root=path.parent, tsv_path=path, records=records)


@dataclass
class LectureBankCatalog:
    root: Path
    tsv_path: Path
    records: Dict[str, LectureBankRecord]
    taxonomy: Dict[str, str] = field(default_factory=dict)

    def get(self, lecturebank_id: str) -> Optional[LectureBankRecord]:
        return self.records.get(normalize_lecturebank_id(lecturebank_id))

    def require(self, lecturebank_id: str) -> LectureBankRecord:
        rec = self.get(lecturebank_id)
        if rec is None:
            raise SelectionError(
                f"LectureBank ID {lecturebank_id!r} is not present in {self.tsv_path}."
            )
        return rec

    def search(self, text: str, *, limit: Optional[int] = None) -> List[LectureBankRecord]:
        needle = str(text or "").strip().lower()
        rows = list(self.records.values())
        if needle:
            rows = [r for r in rows if needle in r.search_blob()]
        rows.sort(key=_id_sort_key)
        if limit is not None:
            rows = rows[: int(limit)]
        return rows

    def all_records(self, *, limit: Optional[int] = None) -> List[LectureBankRecord]:
        rows = sorted(self.records.values(), key=_id_sort_key)
        if limit is not None:
            rows = rows[: int(limit)]
        return rows


def normalize_lecturebank_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise SelectionError("LectureBank ID must be non-empty.")
    return text


def parse_id_list(text: str) -> List[str]:
    parts: List[str] = []
    for chunk in re.split(r"[,;\s]+", str(text or "").strip()):
        if chunk:
            parts.append(normalize_lecturebank_id(chunk))
    return parts


def load_ids_file(path: Path) -> List[str]:
    ids: List[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = re.split(r"[\s,;]+", line, maxsplit=1)[0]
        ids.append(normalize_lecturebank_id(token))
    return ids


def resolve_selected_ids(
    *,
    id_values: Sequence[str] = (),
    ids_csv: str = "",
    ids_file: Optional[Path] = None,
    allow_any_count: bool = False,
    expected_count: Optional[int] = None,
) -> List[str]:
    if allow_any_count and expected_count is not None:
        raise SelectionError("--expected-count and --allow-any-count are mutually exclusive.")
    collected: List[str] = []
    for value in id_values or ():
        collected.extend(parse_id_list(str(value)))
    if ids_csv:
        collected.extend(parse_id_list(ids_csv))
    if ids_file is not None:
        collected.extend(load_ids_file(Path(ids_file)))
    if not collected:
        raise SelectionError(
            "No LectureBank IDs selected. Use --id, --ids, and/or --ids-file."
        )
    seen: Dict[str, int] = {}
    dupes: List[str] = []
    unique: List[str] = []
    for ident in collected:
        if ident in seen:
            if ident not in dupes:
                dupes.append(ident)
            continue
        seen[ident] = 1
        unique.append(ident)
    if dupes:
        raise SelectionError(
            "Duplicate LectureBank IDs are not allowed: " + ", ".join(dupes)
        )
    if allow_any_count:
        return unique
    want = REQUIRED_EVAL_ID_COUNT if expected_count is None else int(expected_count)
    if want < 1:
        raise SelectionError("--expected-count must be a positive integer.")
    if len(unique) != want:
        hint = (
            " Use --allow-any-count for smoke tests."
            if expected_count is None
            else ""
        )
        raise SelectionError(
            f"Evaluation requires exactly {want} unique LectureBank IDs "
            f"(got {len(unique)}).{hint}"
        )
    return unique


def load_lecturebank_catalog(root: Path) -> LectureBankCatalog:
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"LectureBank directory does not exist: {root}")
    tsv_path = _find_metadata_tsv(root)
    taxonomy = _load_taxonomy(root)
    records = _parse_tsv(tsv_path, taxonomy=taxonomy)
    if not records:
        raise ValueError(f"No LectureBank rows parsed from {tsv_path}")
    return LectureBankCatalog(root=root, tsv_path=tsv_path, records=records, taxonomy=taxonomy)


def _find_metadata_tsv(root: Path) -> Path:
    candidates = [
        root / "alldata.tsv",
        root / "data" / "alldata.tsv",
        root / "lecturebank.tsv",
        root / "data" / "lecturebank.tsv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(root.rglob("alldata.tsv"))
    if matches:
        return matches[0]
    matches = sorted(p for p in root.rglob("lecturebank.tsv") if p.is_file())
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find alldata.tsv or lecturebank.tsv under {root}"
    )


def _load_taxonomy(root: Path) -> Dict[str, str]:
    names = ("taxonomy.csv", "taxonomy.tsv", "data/taxonomy.csv", "data/taxonomy.tsv")
    paths: List[Path] = []
    for name in names:
        path = root / name
        if path.is_file():
            paths.append(path)
    paths.extend(sorted(root.rglob("taxonomy.csv"))[:3])
    paths.extend(sorted(root.rglob("taxonomy.tsv"))[:3])
    seen = set()
    mapping: Dict[str, str] = {}
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        mapping.update(_parse_taxonomy(path))
    return mapping


def _parse_taxonomy(path: Path) -> Dict[str, str]:
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    mapping: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delim)
        rows = list(reader)
    if not rows:
        return mapping
    header = [c.strip().lower() for c in rows[0]]
    id_idx = _first_index(header, ("id", "topic", "topic_id", "topicid"))
    name_idx = _first_index(header, ("name", "topic_name", "label", "title"))
    start = 1
    if id_idx is None or name_idx is None:
        start = 0
        id_idx, name_idx = 0, 1 if len(rows[0]) > 1 else 0
    for row in rows[start:]:
        if len(row) <= max(id_idx, name_idx):
            continue
        tid = str(row[id_idx]).strip()
        name = str(row[name_idx]).strip()
        if tid:
            mapping[tid] = name
    return mapping


def _first_index(header: Sequence[str], names: Sequence[str]) -> Optional[int]:
    for name in names:
        if name in header:
            return header.index(name)
    return None


def _parse_tsv(path: Path, *, taxonomy: Mapping[str, str]) -> Dict[str, LectureBankRecord]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.excel_tab
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        except csv.Error:
            dialect = csv.excel_tab
        reader = csv.reader(handle, dialect)
        rows = [row for row in reader if any(str(c).strip() for c in row)]
    if not rows:
        return {}
    header_map = _header_map(rows[0])
    body = rows[1:] if header_map else rows
    if not header_map:
        header_map = {name: i for i, name in enumerate(_HEADERLESS_FIELDS)}
    records: Dict[str, LectureBankRecord] = {}
    for row in body:
        rec = _row_to_record(row, header_map, taxonomy)
        if rec is None:
            continue
        records[rec.lecturebank_id] = rec
    return records


def _header_map(row: Sequence[str]) -> Dict[str, int]:
    mapped: Dict[str, int] = {}
    looks_like_header = False
    for i, raw in enumerate(row):
        key = str(raw or "").strip().lower().replace(" ", "_")
        canon = _HEADER_ALIASES.get(key)
        if canon is not None:
            mapped[canon] = i
            if key in {"id", "title", "url", "instructor"}:
                looks_like_header = True
    if not looks_like_header or "id" not in mapped:
        return {}
    return mapped


def _row_to_record(
    row: Sequence[str],
    header_map: Mapping[str, int],
    taxonomy: Mapping[str, str],
) -> Optional[LectureBankRecord]:
    def cell(name: str) -> str:
        idx = header_map.get(name)
        if idx is None or idx >= len(row):
            return ""
        return str(row[idx]).strip()

    ident = cell("id")
    if not ident or ident.lower() == "id":
        return None
    topic_id = cell("topic_id")
    topic_name = taxonomy.get(topic_id, "")
    extra = {str(i): str(v) for i, v in enumerate(row)}
    return LectureBankRecord(
        lecturebank_id=ident,
        title=cell("title"),
        topic_id=topic_id,
        topic_name=topic_name,
        instructor=cell("instructor"),
        venue=cell("venue"),
        year=cell("year"),
        url=cell("url"),
        extra=extra,
    )


def _id_sort_key(record: LectureBankRecord) -> Tuple[Any, ...]:
    ident = record.lecturebank_id
    if re.fullmatch(r"-?\d+", ident):
        return (0, int(ident), ident)
    return (1, ident.lower(), ident)


def format_id_table(
    records: Iterable[LectureBankRecord],
    *,
    pdf_available: Optional[Mapping[str, bool]] = None,
) -> str:
    lines = [
        "lecturebank_id\ttitle\ttopic_id\ttopic_name\tinstructor\tvenue\tyear\turl\tlocal_pdf"
    ]
    avail = pdf_available or {}
    for rec in records:
        local = avail.get(rec.lecturebank_id)
        local_s = "" if local is None else ("yes" if local else "no")
        lines.append(
            "\t".join(
                [
                    rec.lecturebank_id,
                    rec.title,
                    rec.topic_id,
                    rec.topic_name,
                    rec.instructor,
                    rec.venue,
                    rec.year,
                    rec.url,
                    local_s,
                ]
            )
        )
    return "\n".join(lines) + "\n"
