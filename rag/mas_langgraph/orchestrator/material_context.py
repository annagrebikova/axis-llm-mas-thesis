"""Derive retrieval and quiz targets from document state — no user-supplied topic."""

from typing import Any, List, Optional, Set, TYPE_CHECKING

from .coverage import compute_coverage_gaps

if TYPE_CHECKING:
    from ..core.state_base import BaseCooperativeState


def _normalize_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


# Broad queries used when section-based retrieval is exhausted but more evidence is needed.
FALLBACK_RETRIEVAL_QUERIES = (
    "key concepts definitions overview summary",
    "main algorithms methods techniques examples",
    "important formulas procedures steps",
    "exercises problems applications case study",
)


class MaterialContext:
    """Build queries and labels from outline, LOs, and coverage gaps on the blackboard."""

    def __init__(
        self,
        state: "BaseCooperativeState",
        coverage_plan: Any = None,
        pdf_id: Optional[str] = None,
    ) -> None:
        self.state = state
        self.coverage_plan = coverage_plan
        self.pdf_id = pdf_id

    def section_queries(self) -> List[str]:
        """Distinct retrieval queries — one per outline section, then LOs, then fallbacks."""
        queries: List[str] = []
        outline = getattr(self.state, "doc_outline", None) or []
        for sec in outline:
            queries.append(f"{sec.title} key concepts and definitions")
        los = getattr(self.state, "learning_objectives", None) or []
        for lo in los:
            text = (getattr(lo, "objective_text", "") or "").strip()
            if text:
                queries.append(text[:120])
        if not queries and self.pdf_id:
            queries.append(f"main topics in document {self.pdf_id}")
        if not queries:
            queries.append("key concepts from the document")
        # Preserve order, drop exact duplicates
        seen: set = set()
        unique: List[str] = []
        for q in queries:
            key = _normalize_query(q)
            if key not in seen:
                seen.add(key)
                unique.append(q)
        return unique

    def next_retrieval_query(
        self,
        searched: Set[str],
        step: int = 0,
    ) -> Optional[str]:
        """First section/LO query not yet searched; then broad fallbacks."""
        normalized = {_normalize_query(q) for q in searched}
        for q in self.section_queries():
            if _normalize_query(q) not in normalized:
                return q
        for q in FALLBACK_RETRIEVAL_QUERIES:
            if _normalize_query(q) not in normalized:
                return q
        return None

    def document_theme(self) -> str:
        """Short label for the whole material (logging / result summary)."""
        outline = getattr(self.state, "doc_outline", None) or []
        if outline:
            if len(outline) == 1:
                return outline[0].title
            return outline[0].title + " and related sections"
        if self.pdf_id:
            return f"PDF {self.pdf_id}"
        return "course material"

    def retrieval_query(self, step: int = 0) -> str:
        """Rotate section titles so retrieval covers the document, not one user keyword."""
        queries = self.section_queries()
        if queries:
            return queries[step % len(queries)]
        return "key concepts from the document"

    def quiz_target(self) -> str:
        """Next quiz focus: topic|question_type|difficulty from first coverage gap."""
        if self.coverage_plan is not None:
            gaps = compute_coverage_gaps(self.state, self.coverage_plan)
            if gaps:
                g = gaps[0]
                topic = str(g.get("topic") or self.document_theme())
                qtype = str(g.get("question_type") or "multiple_choice")
                diff = str(g.get("difficulty") or "intermediate")
                return f"{topic}|{qtype}|{diff}"
        outline = getattr(self.state, "doc_outline", None) or []
        if outline:
            n_val = len(getattr(self.state, "validated_questions", {}) or {})
            sec = outline[n_val % len(outline)]
            return f"{sec.title}|multiple_choice|intermediate"
        return f"{self.document_theme()}|multiple_choice|intermediate"

    def pack_title(self) -> str:
        theme = self.document_theme()
        if theme.startswith("PDF "):
            return "Generated Quiz"
        return f"Quiz: {theme}"
