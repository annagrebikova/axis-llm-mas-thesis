"""Coverage gap computation — targets are global; topics come from outline sections."""

from typing import Any, Dict, List, TYPE_CHECKING

from ..core.entities import CoveragePlan

if TYPE_CHECKING:
    from ..core.state import CooperativeState


def outline_section_titles(state: "CooperativeState") -> List[str]:
    outline = getattr(state, "doc_outline", None) or []
    titles = [sec.title for sec in outline if getattr(sec, "title", None)]
    return titles or ["course material"]


def _topic_match(tag: str, section_title: str) -> bool:
    t = (tag or "").lower()
    s = (section_title or "").lower()
    if not t or not s:
        return False
    return t == s or t in s or s in t


def _next_section_topic_for_gap(
    state: "CooperativeState",
    difficulty: str,
    question_type: str,
) -> str:
    """Pick outline section with fewest validated questions for this difficulty/type."""
    titles = outline_section_titles(state)
    counts = {title: 0 for title in titles}
    for q in state.validated_questions.values():
        if q.difficulty != difficulty or q.question_type != question_type:
            continue
        for title in titles:
            tags = q.topic_tags or []
            if any(_topic_match(tag, title) for tag in tags):
                counts[title] += 1
                break
    return min(counts, key=lambda t: (counts[t], t))


def compute_coverage_gaps(
    state: "CooperativeState",
    plan: CoveragePlan,
) -> List[Dict[str, Any]]:
    """Gaps for plan targets (global counts); each gap names an outline section to generate for."""
    validated_per_key: Dict[str, int] = {}
    for q in state.validated_questions.values():
        key = f"{q.difficulty}:{q.question_type}"
        validated_per_key[key] = validated_per_key.get(key, 0) + 1

    gaps: List[Dict[str, Any]] = []
    for (diff, qtype), target in plan.default_per_difficulty_type.items():
        if target <= 0:
            continue
        key = f"{diff}:{qtype}"
        current = validated_per_key.get(key, 0)
        gap = target - current
        if gap > 0:
            topic = _next_section_topic_for_gap(state, diff, qtype)
            gaps.append({
                "topic": topic,
                "difficulty": diff,
                "question_type": qtype,
                "target": target,
                "current": current,
                "gap": gap,
            })
    gaps.sort(key=lambda g: (-g["gap"], g["topic"]))
    return gaps
