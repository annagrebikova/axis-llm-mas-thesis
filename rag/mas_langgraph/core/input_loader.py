"""Load simple YAML/JSON case input files without requiring PyYAML."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Union


def load_mapping(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load a mapping from path or dict. Prefers PyYAML, then JSON, then a minimal YAML subset."""
    if isinstance(source, dict):
        return source
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Try sibling .json
    if path.suffix in {".yaml", ".yml"}:
        sibling = path.with_suffix(".json")
        if sibling.exists():
            return json.loads(sibling.read_text(encoding="utf-8"))
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return _minimal_yaml(text)


def _minimal_yaml(text: str) -> Dict[str, Any]:
    """Parse a restricted YAML subset used by our case input files."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    root: Dict[str, Any] = {}
    stack: list = [(0, root)]
    pending_key: str | None = None

    def current_container():
        return stack[-1][1]

    i = 0
    while i < len(lines):
        ln = lines[i]
        indent = len(ln) - len(ln.lstrip(" "))
        content = ln.strip()

        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()

        if content.startswith("- "):
            item_text = content[2:].strip()
            container = current_container()
            if not isinstance(container, list):
                # convert parent key's value to list
                raise ValueError(f"List item without list parent: {ln}")
            if ":" in item_text and not item_text.startswith("["):
                # inline mapping start: "- key: val" possibly multi-line
                key, _, val = item_text.partition(":")
                obj: Dict[str, Any] = {key.strip(): _parse_scalar(val.strip())}
                container.append(obj)
                stack.append((indent + 2, obj))
            else:
                container.append(_parse_scalar(item_text))
            i += 1
            continue

        if ":" in content:
            key, _, val = content.partition(":")
            key = key.strip()
            val = val.strip()
            container = current_container()
            if not isinstance(container, dict):
                # nested key inside list item object — stack should have dict
                raise ValueError(f"Expected mapping container for key {key}")
            if val == "":
                # look ahead
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("- "):
                    lst: list = []
                    container[key] = lst
                    stack.append((indent + 2, lst))
                else:
                    nested: Dict[str, Any] = {}
                    container[key] = nested
                    stack.append((indent + 2, nested))
            else:
                container[key] = _parse_scalar(val)
            i += 1
            continue
        i += 1
    return root


def _parse_scalar(val: str) -> Any:
    if val == "" or val is None:
        return ""
    if val.lower() in {"true", "yes"}:
        return True
    if val.lower() in {"false", "no"}:
        return False
    if val.lower() in {"null", "~"}:
        return None
    if re.fullmatch(r"-?\d+", val):
        return int(val)
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x.strip()) for x in inner.split(",")]
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]
    return val
