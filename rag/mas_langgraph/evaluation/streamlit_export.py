"""Optional Streamlit explanation-experiment JSON export.

Persists the existing ``gen.experiment_payload`` unchanged. Does not
reconstruct visualization data from LectureBank metrics CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Sequence

from mas_langgraph.batch.io import atomic_write_json
from mas_langgraph.visualization.ui_helpers import case_visualization_issue

STREAMLIT_SUBDIR = "streamlit"
UNAVAILABLE_RESUME_MESSAGE = (
    "Streamlit experiment JSON unavailable for previously completed document; "
    "rerun with --overwrite."
)


def streamlit_experiment_filename(lecturebank_id: str) -> str:
    return f"quiz_explanation_experiment_{lecturebank_id}.json"


def streamlit_json_relpath(lecturebank_id: str) -> str:
    return f"{STREAMLIT_SUBDIR}/{streamlit_experiment_filename(lecturebank_id)}"


def streamlit_json_path(output_dir: Path, lecturebank_id: str) -> Path:
    return Path(output_dir) / STREAMLIT_SUBDIR / streamlit_experiment_filename(lecturebank_id)


def payload_is_experiment(payload: Any) -> bool:
    return isinstance(payload, Mapping) and isinstance(payload.get("cases"), list)


def visualization_export_issues(payload: Mapping[str, Any]) -> List[str]:
    """Report missing/invalid visualization_data; do not invent replacements."""
    issues: List[str] = []
    for i, case in enumerate(payload.get("cases") or []):
        if not isinstance(case, Mapping):
            continue
        issue = case_visualization_issue(case, i)
        if issue:
            issues.append(issue)
    return issues


def write_streamlit_experiment_json(
    output_dir: Path,
    lecturebank_id: str,
    payload: Mapping[str, Any],
) -> str:
    """Atomically write the existing experiment payload. Returns the relative path."""
    if not payload_is_experiment(payload):
        raise ValueError("explanation experiment payload is missing a cases list")
    rel = streamlit_json_relpath(str(lecturebank_id))
    atomic_write_json(Path(output_dir) / rel, payload)
    return rel.replace("\\", "/")


def count_streamlit_json_files(output_dir: Path) -> int:
    folder = Path(output_dir) / STREAMLIT_SUBDIR
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.glob("quiz_explanation_experiment_*.json") if path.is_file())


def collect_streamlit_unavailable_ids(collected: Sequence[Mapping[str, Any]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in collected:
        if not item.get("streamlit_export_unavailable"):
            continue
        lid = str(item.get("lecturebank_id") or "")
        if lid and lid not in seen:
            seen.add(lid)
            out.append(lid)
    return out
