"""Discover presentation files for batch processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

PDF_SUFFIXES = {".pdf"}
POWERPOINT_SUFFIXES = {".ppt", ".pptx"}


@dataclass(frozen=True)
class DiscoveryResult:
    pdfs: Tuple[Path, ...]
    unsupported: Tuple[Tuple[Path, str], ...]


def discover_presentations(
    input_dir: Path,
    *,
    recursive: bool = False,
) -> DiscoveryResult:
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    pdfs: List[Path] = []
    unsupported: List[Tuple[Path, str]] = []
    for path in iterator:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            pdfs.append(path.resolve())
        elif suffix in POWERPOINT_SUFFIXES:
            unsupported.append(
                (
                    path.resolve(),
                    "PowerPoint files are not supported; export the presentation as PDF first.",
                )
            )
        elif suffix:
            unsupported.append(
                (
                    path.resolve(),
                    f"Unsupported file type {suffix}; only PDF presentations are processed.",
                )
            )
    pdfs.sort(key=lambda p: str(p.relative_to(root)).replace("\\", "/").lower())
    unsupported.sort(key=lambda item: str(item[0]))
    return DiscoveryResult(pdfs=tuple(pdfs), unsupported=tuple(unsupported))


def format_unsupported_report(items: Sequence[Tuple[Path, str]]) -> str:
    if not items:
        return ""
    lines = ["Unsupported input files (not processed as PDF):"]
    for path, reason in items:
        lines.append(f"  - {path}: {reason}")
    return "\n".join(lines)
