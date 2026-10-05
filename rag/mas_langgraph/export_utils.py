"""Serialize QuizPack objects to JSON."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, List


def packs_to_dicts(packs: List[Any]) -> List[dict]:
    return [asdict(p) for p in packs]


def dump_quiz_packs(packs: List[Any], path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(packs_to_dicts(packs), f, indent=2, default=str)
    return str(out.resolve())
