"""MCQ cooperative blackboard — same API as rag_agents_cooperative.CooperativeState."""

from typing import Any, Dict, List, Optional

from ...core.state_base import BaseCooperativeState
from ...core.entities import (
    CandidateQuestion,
    CoverageStats,
    DocSection,
    EvidenceChunk,
    KnowledgeAtom,
    LearningObjective,
    QualityMetrics,
    QuizPack,
    RejectedQuestion,
    ValidatedQuestion,
)
from ...domains.quiz.entities import (
    CandidateMCQ,
    CrossCoVeReport,
    EvidenceCheckReport,
    LanguageCheckReport,
    MCQBlueprint,
    RevisionRecord,
    SectionEvidenceStatus,
    SupervisorDecision,
)


class CooperativeState(BaseCooperativeState):
    """Shared state for cooperative agents (quiz/RAG domain default)."""

    def __init__(self) -> None:
        super().__init__()
        self.doc_outline: List[DocSection] = []
        self.learning_objectives: List[LearningObjective] = []
        self.knowledge_atoms: Dict[str, KnowledgeAtom] = {}
        self.evidence_pool: Dict[str, EvidenceChunk] = {}
        self.candidate_questions: Dict[str, CandidateQuestion] = {}
        self.validated_questions: Dict[str, ValidatedQuestion] = {}
        self.rejected_questions: Dict[str, RejectedQuestion] = {}
        self.coverage_matrix: Dict[str, CoverageStats] = {}
        self.quality_metrics: QualityMetrics = QualityMetrics()
        self.final_quiz_packs: List[QuizPack] = []
        self.draft_quiz_packs: Dict[str, QuizPack] = {}
        self.mcq_blueprints: Dict[str, MCQBlueprint] = {}
        self.candidate_mcqs: Dict[str, CandidateMCQ] = {}
        self.language_reports: Dict[str, LanguageCheckReport] = {}
        self.evidence_reports: Dict[str, EvidenceCheckReport] = {}
        self.cross_cove_reports: Dict[str, CrossCoVeReport] = {}
        self.supervisor_decisions: Dict[str, SupervisorDecision] = {}
        self.outcome_assessments: Dict[str, Any] = {}
        self.revision_history: List[RevisionRecord] = []
        self.mcq_coverage_summary: Dict[str, Any] = {}
        self.section_evidence_status: Dict[str, SectionEvidenceStatus] = {}
        # Checkpointed routing counter (must not live on shared runtime).
        self.quiz_generation_rounds: int = 0

    def add_section(self, section: DocSection) -> None:
        self.doc_outline.append(section)

    def add_learning_objective(self, lo: LearningObjective) -> None:
        self.learning_objectives.append(lo)

    def add_knowledge_atom(self, atom: KnowledgeAtom) -> None:
        self.knowledge_atoms[atom.atom_id] = atom

    def add_evidence_chunk(self, chunk: EvidenceChunk) -> None:
        self.evidence_pool[chunk.chunk_id] = chunk

    def add_candidate_question(self, question: CandidateQuestion) -> None:
        self.candidate_questions[question.q_id] = question
        self.quality_metrics.total_questions += 1
        self._update_coverage(question, action="add")

    def validate_question(self, q_id: str, validated: ValidatedQuestion) -> None:
        if q_id in self.candidate_questions:
            del self.candidate_questions[q_id]
        self.validated_questions[q_id] = validated
        self.quality_metrics.validated_questions += 1
        self._update_coverage(validated, action="validate")
        self._update_quality_metrics()

    def reject_question(self, q_id: str, rejection: RejectedQuestion) -> None:
        question = self.candidate_questions.get(q_id)
        if question is not None:
            self._update_coverage(question, action="reject")
        if q_id in self.candidate_questions:
            del self.candidate_questions[q_id]
        self.rejected_questions[q_id] = rejection
        self.quality_metrics.rejected_questions += 1
        self._update_quality_metrics()

    def _update_coverage(self, question: Any, action: str) -> None:
        topic_tags = getattr(question, "topic_tags", []) or []
        difficulty = getattr(question, "difficulty", "intermediate")
        question_type = getattr(question, "question_type", "multiple_choice")
        for topic in topic_tags:
            key = f"{topic}:{difficulty}:{question_type}"
            if key not in self.coverage_matrix:
                self.coverage_matrix[key] = CoverageStats(
                    topic=topic,
                    difficulty=difficulty,
                    question_type=question_type,
                )
            stats = self.coverage_matrix[key]
            if action == "add":
                stats.count += 1
            elif action == "validate":
                stats.validated_count += 1
            elif action == "reject":
                stats.rejected_count += 1

    def _update_quality_metrics(self) -> None:
        total = self.quality_metrics.total_questions
        if total > 0:
            self.quality_metrics.rejection_rate = (
                self.quality_metrics.rejected_questions / total
            )
        used_chunk_ids = set()
        for q in self.validated_questions.values():
            used_chunk_ids.update(q.evidence_chunks)
        if used_chunk_ids:
            scores = [
                self.evidence_pool[cid].quality_score
                for cid in used_chunk_ids
                if cid in self.evidence_pool
            ]
            if scores:
                self.quality_metrics.avg_evidence_strength = sum(scores) / len(scores)

    def get_atom(self, atom_id: str) -> Optional[KnowledgeAtom]:
        return self.knowledge_atoms.get(atom_id)

    def get_chunk(self, chunk_id: str) -> Optional[EvidenceChunk]:
        return self.evidence_pool.get(chunk_id)

    def get_question(self, q_id: str) -> Optional[CandidateQuestion]:
        return self.candidate_questions.get(q_id)

    def generate_id(self, prefix: str) -> str:
        return super().generate_id(prefix)

    def set_doc_outline(self, outline: List[DocSection]) -> None:
        self.doc_outline.clear()
        self.doc_outline.extend(outline)

    def get_section(self, section_id: str) -> Optional[DocSection]:
        for s in self.doc_outline:
            if s.section_id == section_id:
                return s
        return None

    def get_outline_summary(self) -> str:
        if not self.doc_outline:
            return "No outline yet."
        return "\n".join(
            f"- {s.section_id}: {s.title} slides{s.slide_range} weight={s.importance_weight}"
            for s in self.doc_outline
        )

    def get_learning_objectives_summary(self) -> str:
        if not self.learning_objectives:
            return "No learning objectives yet."
        return "\n".join(
            f"- {lo.lo_id} (section={lo.section_id}): {lo.objective_text[:80]}... "
            f"difficulty={lo.difficulty_level}"
            for lo in self.learning_objectives
        )

    def update_evidence_chunk(
        self,
        chunk_id: str,
        relevance_score: Optional[float] = None,
        quality_score: Optional[float] = None,
        relevance_tags: Optional[List[str]] = None,
        used_in_atoms: Optional[List[str]] = None,
        duplicate_of: Optional[str] = None,
    ) -> bool:
        if chunk_id not in self.evidence_pool:
            return False
        c = self.evidence_pool[chunk_id]
        if relevance_score is not None:
            c.relevance_score = relevance_score
        if quality_score is not None:
            c.quality_score = quality_score
        if relevance_tags is not None:
            c.relevance_tags = relevance_tags
        if used_in_atoms is not None:
            c.used_in_atoms = list(used_in_atoms)
        if duplicate_of is not None:
            c.duplicate_of = duplicate_of
        return True

    def set_chunk_flags(self, chunk_id: str, flags: List[str], append: bool = False) -> bool:
        if chunk_id not in self.evidence_pool:
            return False
        c = self.evidence_pool[chunk_id]
        current = list(getattr(c, "flags", None) or [])
        if append:
            for f in flags:
                if f not in current:
                    current.append(f)
        else:
            current = list(flags)
        c.flags = current
        return True

    def link_chunk_to_question(
        self, q_id: str, chunk_id: str, validated: bool = False,
    ) -> bool:
        if chunk_id not in self.evidence_pool:
            return False
        q = (
            self.validated_questions.get(q_id)
            if validated
            else self.candidate_questions.get(q_id)
        )
        if not q:
            return False
        if chunk_id not in q.evidence_chunks:
            q.evidence_chunks.append(chunk_id)
        return True

    def link_chunk_to_atom(self, chunk_id: str, atom_id: str) -> bool:
        if chunk_id not in self.evidence_pool or atom_id not in self.knowledge_atoms:
            return False
        if atom_id not in self.evidence_pool[chunk_id].used_in_atoms:
            self.evidence_pool[chunk_id].used_in_atoms.append(atom_id)
        return True

    def update_atom_lo(self, atom_id: str, lo_id: str) -> bool:
        if atom_id not in self.knowledge_atoms:
            return False
        self.knowledge_atoms[atom_id].lo_id = lo_id
        return True

    def update_atom_section(self, atom_id: str, section_id: str) -> bool:
        if atom_id not in self.knowledge_atoms:
            return False
        self.knowledge_atoms[atom_id].section_id = section_id
        return True

    def get_coverage_stats(self) -> Dict[str, CoverageStats]:
        return dict(self.coverage_matrix)

    def update_coverage_stats(
        self,
        key: str,
        count: int = 0,
        validated_count: int = 0,
        rejected_count: int = 0,
    ) -> None:
        parts = key.split(":")
        if len(parts) != 3:
            return
        topic, difficulty, qtype = parts[0], parts[1], parts[2]
        if key not in self.coverage_matrix:
            self.coverage_matrix[key] = CoverageStats(
                topic=topic, difficulty=difficulty, question_type=qtype,
            )
        s = self.coverage_matrix[key]
        if count != 0:
            s.count = count
        if validated_count != 0:
            s.validated_count = validated_count
        if rejected_count != 0:
            s.rejected_count = rejected_count

    def update_quality_metrics_fields(
        self,
        rejection_rate: Optional[float] = None,
        duplication_score: Optional[float] = None,
        avg_evidence_strength: Optional[float] = None,
    ) -> None:
        if rejection_rate is not None:
            self.quality_metrics.rejection_rate = rejection_rate
        if duplication_score is not None:
            self.quality_metrics.duplication_score = duplication_score
        if avg_evidence_strength is not None:
            self.quality_metrics.avg_evidence_strength = avg_evidence_strength

    def append_quiz_pack(self, pack: QuizPack) -> None:
        self.final_quiz_packs.append(pack)

    def store_mcq_blueprints(self, blueprints: List[MCQBlueprint]) -> None:
        for bp in blueprints:
            self.mcq_blueprints[bp.blueprint_id] = bp

    def store_candidate_mcq(self, candidate: CandidateMCQ) -> None:
        self.candidate_mcqs[candidate.q_id] = candidate

    def store_candidate_mcqs(self, candidates: List[CandidateMCQ]) -> None:
        for candidate in candidates:
            self.store_candidate_mcq(candidate)

    def store_evidence_report(self, report: EvidenceCheckReport) -> None:
        self.evidence_reports[report.q_id] = report

    def store_evidence_reports(self, reports: List[EvidenceCheckReport]) -> None:
        for report in reports:
            self.store_evidence_report(report)

    def store_supervisor_decision(self, decision: SupervisorDecision) -> None:
        self.supervisor_decisions[decision.q_id] = decision

    def store_outcome_assessment(self, assessment: Any) -> None:
        q_id = getattr(getattr(assessment, "target_ref", None), "logical_id", None)
        if q_id is None:
            return
        self.outcome_assessments[str(q_id)] = assessment

    def store_mcq_coverage_summary(self, summary: Dict[str, Any]) -> None:
        self.mcq_coverage_summary = summary

    def store_cross_cove_report(self, report: CrossCoVeReport) -> None:
        self.cross_cove_reports[report.q_id] = report

    def store_cross_cove_reports(self, reports: List[CrossCoVeReport]) -> None:
        for report in reports:
            self.store_cross_cove_report(report)

    def add_revision_record(self, record: RevisionRecord) -> None:
        self.revision_history.append(record)

    def get_revision_count(self, q_id: str) -> int:
        candidate = self.candidate_mcqs.get(q_id)
        if candidate is None:
            return 0
        return candidate.revision_count

    def planned_blueprints(self) -> List[MCQBlueprint]:
        return [
            bp for bp in self.mcq_blueprints.values()
            if bp.status == "planned"
        ]

    def update_section_evidence_status(self, *, min_chunks: int = 2) -> Dict[str, SectionEvidenceStatus]:
        from ...domains.quiz.section_evidence import update_section_evidence_status

        return update_section_evidence_status(self, min_chunks=min_chunks)

    def get_sections_without_evidence(self) -> List[SectionEvidenceStatus]:
        from ...domains.quiz.section_evidence import get_sections_without_evidence

        return get_sections_without_evidence(self)

    def get_sections_with_sufficient_evidence(self) -> List[SectionEvidenceStatus]:
        from ...domains.quiz.section_evidence import get_sections_with_sufficient_evidence

        return get_sections_with_sufficient_evidence(self)

    def snapshot_counts(self) -> Dict[str, Any]:
        """Lightweight snapshot for orchestration progress detection."""
        canonical_chunks = sum(
            1 for c in self.evidence_pool.values() if c.duplicate_of is None
        )
        low_quality_chunks = sum(
            1 for c in self.evidence_pool.values() if "low_quality" in c.flags
        )
        return {
            "doc_outline_len": len(self.doc_outline),
            "learning_objectives_len": len(self.learning_objectives),
            "evidence_pool_len": len(self.evidence_pool),
            "canonical_chunks": canonical_chunks,
            "low_quality_chunks": low_quality_chunks,
            "knowledge_atoms_len": len(self.knowledge_atoms),
            "candidate_questions_len": len(self.candidate_questions),
            "validated_questions_len": len(self.validated_questions),
            "rejected_questions_len": len(self.rejected_questions),
            "mcq_blueprints_len": len(self.mcq_blueprints),
            "candidate_mcqs_len": len(self.candidate_mcqs),
            "evidence_reports_len": len(self.evidence_reports),
            "supervisor_decisions_len": len(self.supervisor_decisions),
            "accepted_mcqs_len": sum(
                1 for mcq in self.candidate_mcqs.values() if mcq.status == "accepted"
            ),
            "cross_cove_reports_len": len(self.cross_cove_reports),
            "revision_history_len": len(self.revision_history),
            "section_evidence_status_len": len(self.section_evidence_status),
            "sections_with_evidence": len(self.get_sections_with_sufficient_evidence()),
            "duplication_score": self.quality_metrics.duplication_score,
            "rejection_rate": self.quality_metrics.rejection_rate,
            "avg_evidence_strength": self.quality_metrics.avg_evidence_strength,
            "final_quiz_packs_len": len(self.final_quiz_packs),
        }
