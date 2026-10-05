"""Orchestrator condition functions — used by input/orchestration_rules.py."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from .needs_base import BaseNeedsEvaluator
from ..input.constants import (
    LOW_QUALITY_CHUNK_RATIO,
    MAX_DUPLICATION_SCORE,
    MAX_PENDING_CANDIDATES,
    MAX_QUIZ_GENERATION_ROUNDS,
    MAX_REJECTION_RATE,
    MIN_ATOMS,
    MIN_EVIDENCE_CHUNKS,
    MIN_LO_COUNT,
    MIN_VALIDATED_QUESTIONS,
    PACKAGING_MIN_VALIDATED_WITH_GAPS,
)
from ..input.coverage_plan import DEFAULT_COVERAGE_PLAN
from .coverage import compute_coverage_gaps

if TYPE_CHECKING:
    from ..core.state import CooperativeState


class NeedsEvaluator(BaseNeedsEvaluator):
    """Quiz/RAG domain — evaluates pipeline needs against CooperativeState."""

    def __init__(
        self,
        state: "CooperativeState",
        coverage_plan=None,
        agent_runner: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(state, coverage_plan, **kwargs)
        if self.coverage_plan is None:
            self.coverage_plan = DEFAULT_COVERAGE_PLAN
        self.agent_runner = agent_runner
        self.max_final_questions = kwargs.get("max_final_questions")

    def _cs(self, cooperative_state: Optional["CooperativeState"] = None) -> "CooperativeState":
        """Resolve workflow state — LangGraph passes explicit ``cooperative_state``."""
        return cooperative_state if cooperative_state is not None else self.state

    def canonical_chunk_count(self, cooperative_state: Optional["CooperativeState"] = None) -> int:
        cs = self._cs(cooperative_state)
        return sum(
            1 for c in cs.evidence_pool.values() if c.duplicate_of is None
        )

    def needs_outline(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        cs = self._cs(cooperative_state)
        return len(cs.doc_outline) == 0

    def needs_learning_objectives(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        cs = self._cs(cooperative_state)
        outline = getattr(cs, "doc_outline", None) or []
        if outline and len(cs.learning_objectives) < len(outline):
            return True
        return len(cs.learning_objectives) < MIN_LO_COUNT

    def needs_more_evidence(
        self,
        topic: str = "",
        cooperative_state: Optional["CooperativeState"] = None,
        searched_queries: Optional[set] = None,
    ) -> bool:
        if self.canonical_chunk_count(cooperative_state) >= MIN_EVIDENCE_CHUNKS:
            return False
        runner = self.agent_runner
        if runner is not None:
            exhausted = runner.retrieval_exhausted
            if cooperative_state is not None or searched_queries is not None:
                if exhausted(
                    cooperative_state=cooperative_state,
                    searched_queries=searched_queries,
                ):
                    return False
            elif exhausted():
                return False
        return True

    def has_min_evidence(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        return self.canonical_chunk_count(cooperative_state) >= MIN_EVIDENCE_CHUNKS

    def needs_deduplication(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        cs = self._cs(cooperative_state)
        canonical = self.canonical_chunk_count(cooperative_state)
        if canonical < MIN_EVIDENCE_CHUNKS:
            return False

        total = len(cs.evidence_pool)
        marked_duplicates = sum(
            1 for c in cs.evidence_pool.values()
            if getattr(c, "duplicate_of", None) is not None
        )
        low_q = sum(
            1 for c in cs.evidence_pool.values() if "low_quality" in c.flags
        )

        # Duplicates already flagged — only re-run for low-quality ratio, not pool size.
        if marked_duplicates > 0 and canonical >= MIN_EVIDENCE_CHUNKS:
            return total > 0 and low_q / total > LOW_QUALITY_CHUNK_RATIO

        dup = cs.quality_metrics.duplication_score
        return dup > MAX_DUPLICATION_SCORE or (
            total > 0 and low_q / total > LOW_QUALITY_CHUNK_RATIO
        )

    def needs_atomization(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        cs = self._cs(cooperative_state)
        return (
            len(cs.knowledge_atoms) < MIN_ATOMS
            and self.canonical_chunk_count(cooperative_state) >= MIN_EVIDENCE_CHUNKS
        )

    def needs_more_questions(
        self,
        topic: str = "",
        cooperative_state: Optional["CooperativeState"] = None,
    ) -> bool:
        cs = self._cs(cooperative_state)
        if self._question_cap_reached(cs):
            return False
        if len(cs.knowledge_atoms) < MIN_ATOMS:
            return False
        if len(cs.candidate_questions) >= MAX_PENDING_CANDIDATES:
            return False
        runner = self.agent_runner
        if int(getattr(cs, "quiz_generation_rounds", 0) or 0) >= MAX_QUIZ_GENERATION_ROUNDS:
            return False
        # Legacy fallback only when checkpointed field is absent (pre-migration states).
        if (
            not hasattr(cs, "quiz_generation_rounds")
            and runner is not None
            and getattr(runner, "_quiz_generation_rounds", 0) >= MAX_QUIZ_GENERATION_ROUNDS
        ):
            return False
        gaps = compute_coverage_gaps(cs, self.coverage_plan)
        return bool(gaps) and any(g.get("gap", 0) > 0 for g in gaps)

    def needs_verification(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        cs = self._cs(cooperative_state)
        return len(cs.candidate_questions) > 0

    def needs_coverage_repair(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        cs = self._cs(cooperative_state)
        if len(cs.knowledge_atoms) < MIN_ATOMS:
            return False
        gaps = compute_coverage_gaps(cs, self.coverage_plan)
        return bool(gaps) and any(g.get("gap", 0) > 0 for g in gaps)

    def rejection_rate_too_high(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        cs = self._cs(cooperative_state)
        qm = cs.quality_metrics
        return qm.total_questions > 0 and qm.rejection_rate > MAX_REJECTION_RATE

    def ready_for_packaging(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        cs = self._cs(cooperative_state)
        if self._question_cap_reached(cs):
            return True
        val = len(cs.validated_questions)
        gaps = compute_coverage_gaps(cs, self.coverage_plan)
        critical = [g for g in gaps if g.get("gap", 0) > 0]
        return val >= MIN_VALIDATED_QUESTIONS and (
            len(critical) == 0 or val >= PACKAGING_MIN_VALIDATED_WITH_GAPS
        )

    def has_final_quiz_packs(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        cs = self._cs(cooperative_state)
        return len(cs.final_quiz_packs) > 0

    def coverage_topic(self, topic: str = "", cooperative_state: Optional["CooperativeState"] = None) -> str:
        from .material_context import MaterialContext

        cs = self._cs(cooperative_state)
        mat = MaterialContext(cs, self.coverage_plan)
        return mat.quiz_target()

    def _question_cap_reached(self, cooperative_state: Optional["CooperativeState"] = None) -> bool:
        from ..domains.quiz.question_limit import final_question_cap_reached

        return final_question_cap_reached(
            self._cs(cooperative_state),
            {"max_final_questions": self.max_final_questions},
        )

    def as_dict(self) -> Dict[str, Callable[..., bool]]:
        return self.register({
            "needs_outline": lambda **_: self.needs_outline(),
            "needs_learning_objectives": lambda **_: self.needs_learning_objectives(),
            "needs_more_evidence": lambda **_: self.needs_more_evidence(),
            "has_min_evidence": lambda **_: self.has_min_evidence(),
            "needs_deduplication": lambda **_: self.needs_deduplication(),
            "needs_atomization": lambda **_: self.needs_atomization(),
            "needs_more_questions": lambda **_: self.needs_more_questions(),
            "needs_verification": lambda **_: self.needs_verification(),
            "needs_coverage_repair": lambda **_: self.needs_coverage_repair(),
            "rejection_rate_too_high": lambda **_: self.rejection_rate_too_high(),
            "ready_for_packaging": lambda **_: self.ready_for_packaging(),
            "has_final_quiz_packs": lambda **_: self.has_final_quiz_packs(),
        })
