"""Deterministic, collection-safe presentation IDs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def posix_relative(path: Path, root: Path) -> str:
    """Portable relative path from ``root`` (forward slashes)."""
    resolved = Path(path).resolve()
    base = Path(root).resolve()
    return resolved.relative_to(base).as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collection_safe_id(relative_posix_path: str) -> str:
    """Stable Chroma collection / document ID from a relative source path.

    Existing single-PDF commands keep using filename-prefix IDs such as ``141``.
    Batch processing always derives IDs from the relative path plus a content-
    independent path hash so ``a/foo.pdf`` and ``b/foo.pdf`` never collide, and
    ``141_A.pdf`` vs ``141_B.pdf`` no longer share the prefix ``141``.
    """
    rel = str(relative_posix_path or "").replace("\\", "/").strip("/")
    if not rel:
        raise ValueError("relative path is required for a collection ID")
    path_digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:10]
    stem = Path(rel).with_suffix("").as_posix().lower()
    slug = _UNSAFE.sub("_", stem).strip("._-")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = "doc"
    # Chroma collection names: 3–63 chars, start/end alphanumeric.
    budget = 63 - 1 - len(path_digest)
    slug = slug[: max(1, budget)].rstrip("._-") or "doc"
    ident = f"{slug}_{path_digest}"
    if ident[0].isdigit() or ident[0] in "._-":
        ident = f"p{ident}"[:63]
    if ident[-1] in "._-":
        ident = ident.rstrip("._-") + "x"
    if len(ident) < 3:
        ident = f"doc_{path_digest}"
    return ident[:63]


def presentation_dir_name(document_id: str) -> str:
    return f"presentation_{document_id}"
