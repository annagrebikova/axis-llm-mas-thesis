"""Stub tool implementations — replace with real logic or delegate to rag_agents_cooperative."""

from typing import Dict, Type

from ..core.entities import (
    CandidateQuestion,
    DocSection,
    EvidenceChunk,
    QuizPack,
    ValidatedQuestion,
)
from ..core.privileges import agent_can
from .base import ToolContext, ToolHandler


class GetStateSummaryTool(ToolHandler):
    name = "get_state_summary"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        s = ctx.state.snapshot_counts()
        ctx.log(self.name, input_str, s)
        return (
            f"Sections: {s['doc_outline_len']}, LOs: {s['learning_objectives_len']}, "
            f"Chunks: {s['evidence_pool_len']}, Atoms: {s['knowledge_atoms_len']}, "
            f"Candidates: {s['candidate_questions_len']}, Validated: {s['validated_questions_len']}"
        )


class BuildOutlineFromDocumentTool(ToolHandler):
    name = "build_outline_from_document"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        if not agent_can(ctx.agent_id, "DocSection", "U"):
            return "Permission denied."
        ctx.state.set_doc_outline([
            DocSection("sec_1", "Introduction", (1, 5), 0.5),
            DocSection("sec_2", "Main content", (6, 15), 0.7),
        ])
        ctx.log(self.name, input_str, "2 sections")
        return f"Built outline with {len(ctx.state.doc_outline)} sections (stub)."


class GenerateLOsTool(ToolHandler):
    name = "generate_learning_objectives_for_outline"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        from ..core.entities import LearningObjective

        if not agent_can(ctx.agent_id, "LearningObjective", "C"):
            return "Permission denied."
        added = 0
        for sec in ctx.state.doc_outline:
            lo_id = f"{sec.section_id}_lo_1"
            ctx.state.add_learning_objective(LearningObjective(
                lo_id=lo_id,
                section_id=sec.section_id,
                objective_text=f"Understand {sec.title}",
                difficulty_level="intermediate",
            ))
            added += 1
        ctx.log(self.name, input_str, added)
        return f"Generated {added} learning objectives (stub)."


class RetrieveChunksTool(ToolHandler):
    name = "retrieve_chunks"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        if not agent_can(ctx.agent_id, "EvidenceChunk", "C"):
            return "Permission denied."
        query = input_str
        if not (query or "").strip():
            from ..orchestrator.material_context import MaterialContext
            query = MaterialContext(ctx.state).retrieval_query(0)
        known = {
            " ".join((c.content or "").strip().lower().split())[:500]
            for c in ctx.state.evidence_pool.values()
        }
        added = 0
        for i in range(3):
            body = f"Stub evidence for query: {query} (#{i+1})"
            key = " ".join(body.strip().lower().split())[:500]
            if key in known:
                continue
            cid = ctx.state.generate_id("chunk")
            ctx.state.add_evidence_chunk(EvidenceChunk(
                chunk_id=cid,
                content=body,
                metadata={"source_file": "stub.pdf", "page_number": i + 1},
                relevance_score=0.8,
                quality_score=0.8,
                relevance_tags=[query.lower()[:40]],
            ))
            known.add(key)
            added += 1
        ctx.log(self.name, query, added)
        return f"Retrieved {added} chunks for: {query} (stub)."


class AtomizeChunksTool(ToolHandler):
    name = "atomize_chunks"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        from ..core.entities import KnowledgeAtom

        if not agent_can(ctx.agent_id, "KnowledgeAtom", "C"):
            return "Permission denied."
        created = 0
        for cid, chunk in list(ctx.state.evidence_pool.items()):
            if chunk.duplicate_of:
                continue
            aid = ctx.state.generate_id("atom")
            ctx.state.add_knowledge_atom(KnowledgeAtom(
                atom_id=aid,
                content=chunk.content[:200],
                atom_type="fact",
                citations=[{"chunk_id": cid, "page": chunk.metadata.get("page_number")}],
            ))
            ctx.state.link_chunk_to_atom(cid, aid)
            created += 1
        ctx.log(self.name, input_str, created)
        return f"Created {created} atoms (stub)."


class GenerateQuizDraftTool(ToolHandler):
    name = "generate_quiz_draft"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        from ..orchestrator.coverage import compute_coverage_gaps
        from ..input.coverage_plan import DEFAULT_COVERAGE_PLAN

        if not agent_can(ctx.agent_id, "CandidateQuestion", "C"):
            return "Permission denied."
        from ..integrations.rag_direct_tools import parse_quiz_spec
        if input_str and "|" in input_str:
            topic, qtype, difficulty = parse_quiz_spec(input_str)
        else:
            gaps = compute_coverage_gaps(ctx.state, DEFAULT_COVERAGE_PLAN)
            if gaps:
                g = gaps[0]
                topic = str(g.get("topic") or "general")
                difficulty = g.get("difficulty", "intermediate")
                qtype = g.get("question_type", "multiple_choice")
            elif input_str:
                topic, qtype, difficulty = input_str, "multiple_choice", "intermediate"
            elif ctx.state.doc_outline:
                sec = ctx.state.doc_outline[len(ctx.state.candidate_questions) % len(ctx.state.doc_outline)]
                topic, qtype, difficulty = sec.title, "multiple_choice", "intermediate"
            else:
                topic, qtype, difficulty = "course material", "multiple_choice", "intermediate"
        qid = ctx.state.generate_id("q")
        ctx.state.add_candidate_question(CandidateQuestion(
            q_id=qid,
            question_text=f"What is a key idea about {topic}?",
            question_type=qtype,
            options=["A) option 1", "B) option 2", "C) option 3", "D) option 4"],
            correct_answer="A) option 1",
            difficulty=difficulty,
            topic_tags=[topic],
            related_atoms=list(ctx.state.knowledge_atoms.keys())[:2],
            evidence_chunks=list(ctx.state.evidence_pool.keys())[:2],
        ))
        ctx.log(self.name, topic, qid)
        return f"Added candidate question {qid} for {topic}/{difficulty}/{qtype} (stub)."


class VerifyAllCandidatesTool(ToolHandler):
    name = "verify_all_candidates"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        from ..core.entities import RejectedQuestion

        if not agent_can(ctx.agent_id, "CandidateQuestion", "R"):
            return "Permission denied."
        q_ids = list(ctx.state.candidate_questions.keys())[:5]
        if not q_ids:
            return "No candidates to verify."
        validated_n = 0
        for q_id in q_ids:
            q = ctx.state.candidate_questions[q_id]
            if q.evidence_chunks:
                ctx.state.validate_question(q_id, ValidatedQuestion(
                    q_id=q.q_id,
                    question_text=q.question_text,
                    question_type=q.question_type,
                    correct_answer=q.correct_answer or "A",
                    difficulty=q.difficulty,
                    options=q.options,
                    topic_tags=q.topic_tags,
                    related_atoms=q.related_atoms,
                    evidence_chunks=q.evidence_chunks,
                    validation_notes="Stub auto-validated",
                ))
                validated_n += 1
            else:
                ctx.state.reject_question(q_id, RejectedQuestion(
                    q_id=q_id,
                    question_text=q.question_text,
                    rejection_reason="No evidence chunks linked.",
                    original_question={"q_id": q_id},
                ))
        ctx.log(self.name, input_str, validated_n)
        return f"Verified {len(q_ids)}: {validated_n} validated (stub)."


class CurateEvidenceTool(ToolHandler):
    name = "curate_evidence"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        if not agent_can(ctx.agent_id, "EvidenceChunk", "U"):
            return "Permission denied."
        for c in ctx.state.evidence_pool.values():
            if c.quality_score < 0.5:
                c.quality_score = 0.75
        ctx.log(self.name, input_str, len(ctx.state.evidence_pool))
        return f"Curated {len(ctx.state.evidence_pool)} chunks (stub)."


class DetectDuplicateChunksTool(ToolHandler):
    name = "detect_duplicate_chunks"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        if not agent_can(ctx.agent_id, "EvidenceChunk", "U"):
            return "Permission denied."
        ctx.state.quality_metrics.duplication_score = 0.0
        ctx.log(self.name, input_str, "ok")
        return "Duplicate detection complete (stub)."


class GetNextTargetsTool(ToolHandler):
    name = "get_next_targets"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        from ..orchestrator.coverage import compute_coverage_gaps
        from ..input.coverage_plan import DEFAULT_COVERAGE_PLAN

        gaps = compute_coverage_gaps(ctx.state, DEFAULT_COVERAGE_PLAN)
        top = gaps[:3]
        ctx.log(self.name, input_str, top)
        if not top:
            return "No coverage gaps; targets satisfied (stub)."
        lines = [f"- {g['topic']} ({g['difficulty']}, {g['question_type']}): gap={g['gap']}" for g in top]
        return "Next targets:\n" + "\n".join(lines)


class ExportPackTool(ToolHandler):
    name = "export_pack"

    def run(self, ctx: ToolContext, input_str: str) -> str:
        if not agent_can(ctx.agent_id, "QuizPack", "C"):
            return "Permission denied."
        title = input_str or "Generated Quiz"
        pack_id = ctx.state.generate_id("pack")
        pack = QuizPack(
            pack_id=pack_id,
            title=title,
            questions=list(ctx.state.validated_questions.values()),
            metadata={"stub": True},
            coverage_summary={"coverage_matrix": ctx.state.get_coverage_stats()},
        )
        ctx.state.append_quiz_pack(pack)
        ctx.log(self.name, title, pack_id)
        return f"Exported pack {pack_id} with {len(pack.questions)} questions (stub)."


class PassthroughTool(ToolHandler):
    """Generic stub for tools not yet implemented."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, ctx: ToolContext, input_str: str) -> str:
        ctx.log(self.name, input_str, "ok")
        return f"{self.name}: stub OK (no-op)."


# Primary tools used by orchestrator direct paths
PRIMARY_TOOLS: Dict[str, Type[ToolHandler]] = {
    "get_state_summary": GetStateSummaryTool,
    "build_outline_from_document": BuildOutlineFromDocumentTool,
    "generate_learning_objectives_for_outline": GenerateLOsTool,
    "retrieve_chunks": RetrieveChunksTool,
    "atomize_chunks": AtomizeChunksTool,
    "generate_quiz_draft": GenerateQuizDraftTool,
    "verify_all_candidates": VerifyAllCandidatesTool,
    "curate_evidence": CurateEvidenceTool,
    "detect_duplicate_chunks": DetectDuplicateChunksTool,
    "get_next_targets": GetNextTargetsTool,
    "export_pack": ExportPackTool,
}
