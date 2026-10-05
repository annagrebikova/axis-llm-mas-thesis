"""Select PDFs from a discovered input folder without moving files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .discover import DiscoveryResult
from .ids import collection_safe_id, posix_relative


class SelectionError(ValueError):
    """Unknown or ambiguous PDF selector; nothing should be processed."""


def catalog_presentations(
    discovered: DiscoveryResult,
    input_dir: Path,
) -> List[Dict[str, str]]:
    """Relative path, document ID, and resolved path for each discovered PDF."""
    root = Path(input_dir).resolve()
    rows: List[Dict[str, str]] = []
    for path in discovered.pdfs:
        rel = posix_relative(path, root)
        rows.append(
            {
                "relative_path": rel,
                "document_id": collection_safe_id(rel),
                "path": str(path),
            }
        )
    return rows


def resolve_pdf_selection(
    discovered: DiscoveryResult,
    input_dir: Path,
    *,
    pdf_selectors: Sequence[str] = (),
    document_ids: Sequence[str] = (),
) -> Tuple[Path, ...]:
    """Deduplicated union of selectors, in discovery order.

    With no selectors, return every discovered PDF. Never silently ignore a
    bad selector by falling back to the full set.
    """
    catalog = catalog_presentations(discovered, input_dir)
    if not pdf_selectors and not document_ids:
        return tuple(discovered.pdfs)

    by_rel = {row["relative_path"]: Path(row["path"]) for row in catalog}
    by_rel_folded = _fold_map(by_rel)
    by_id = {row["document_id"]: Path(row["path"]) for row in catalog}
    by_name: Dict[str, List[Tuple[str, Path]]] = {}
    for row in catalog:
        name = Path(row["relative_path"]).name
        by_name.setdefault(name.casefold(), []).append(
            (row["relative_path"], Path(row["path"]))
        )

    selected: List[Path] = []
    for raw in pdf_selectors:
        selected.append(
            _resolve_pdf_selector(
                str(raw),
                by_rel=by_rel,
                by_rel_folded=by_rel_folded,
                by_name=by_name,
            )
        )
    for raw in document_ids:
        ident = str(raw).strip()
        path = by_id.get(ident)
        if path is None:
            raise SelectionError(
                f"Unknown --document-id {ident!r}. "
                "Use --list-pdfs to see selectable document IDs."
            )
        selected.append(path)

    wanted = {p.resolve() for p in selected}
    ordered = tuple(p for p in discovered.pdfs if p.resolve() in wanted)
    return ordered


def format_pdf_list(
    rows: Sequence[Mapping[str, str]],
    *,
    statuses: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> str:
    statuses = statuses or {}
    lines = [
        "relative_path\tdocument_id\trag_status\tgenerate_status",
    ]
    if not rows:
        lines.append("(no PDF presentations found)")
        return "\n".join(lines)
    for row in rows:
        doc_id = row["document_id"]
        st = statuses.get(doc_id) or {}
        rag = st.get("rag_status") or "not_started"
        gen = st.get("generate_status") or "not_started"
        lines.append(f"{row['relative_path']}\t{doc_id}\t{rag}\t{gen}")
    return "\n".join(lines)


def _resolve_pdf_selector(
    raw: str,
    *,
    by_rel: Mapping[str, Path],
    by_rel_folded: Mapping[str, List[Tuple[str, Path]]],
    by_name: Mapping[str, List[Tuple[str, Path]]],
) -> Path:
    selector = str(raw).replace("\\", "/").strip()
    if not selector:
        raise SelectionError("Empty --pdf selector is not allowed.")
    if selector in by_rel:
        return by_rel[selector]
    folded = selector.casefold()
    rel_hits = by_rel_folded.get(folded) or []
    if len(rel_hits) == 1:
        return rel_hits[0][1]
    if len(rel_hits) > 1:
        paths = ", ".join(item[0] for item in rel_hits)
        raise SelectionError(
            f"Ambiguous --pdf {raw!r} matches multiple relative paths: {paths}. "
            "Pass an exact relative path."
        )
    name_hits = by_name.get(Path(selector).name.casefold()) or []
    if len(name_hits) == 1:
        return name_hits[0][1]
    if len(name_hits) > 1:
        listed = "\n".join(f"  - {item[0]}" for item in name_hits)
        raise SelectionError(
            f"Ambiguous filename {Path(selector).name!r}. Matching relative paths:\n"
            f"{listed}\nPass an exact --pdf relative path."
        )
    raise SelectionError(
        f"Unknown --pdf {raw!r}. Use --list-pdfs to see selectable files."
    )


def _fold_map(by_rel: Mapping[str, Path]) -> Dict[str, List[Tuple[str, Path]]]:
    out: Dict[str, List[Tuple[str, Path]]] = {}
    for rel, path in by_rel.items():
        out.setdefault(rel.casefold(), []).append((rel, path))
    return out
