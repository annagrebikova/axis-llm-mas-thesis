"""

New vers. LangChain Cooperative Agents with State Management
Implements specialized agents with explicit privileges per entity type:
- Structure/Outline Agent: build outline, slide ranges, weights; store in CooperativeState
- Curriculum/LO Agent: create learning objectives, set difficulty; bind atoms to LOs
- Retriever Agent: section/LO-targeted retrieval; add chunks to evidence_pool
- Quality/Duplicate Detector Agent: flag duplicate chunks (run right after retriever so duplicates do not contaminate later steps)
- Evidence Curator/Ranker Agent: dedupe, score, tag chunks; link chunks to atoms/questions
- Atomizer Agent: extract knowledge atoms from chunks; link to section/LO
- Quiz Generator Agent: generate candidate questions; use atoms/chunks; update coverage
- Verifier/Fact-check Agent: validate/reject questions; update coverage and quality metrics
- Coverage Controller Agent: analyze gaps; steer generation targets
- Packager/Export Agent: create QuizPack; append to final_quiz_packs

Recommended evidence-to-quiz workflow: retriever → duplicate_detector → curator → atomizer → quiz_generation → verifier → packager.

Central orchestrator (--orchestrate): Rule-based controller runs the default sequence with
conditional loops (e.g. more evidence, dedup, more questions, verification, coverage repair)
and stop conditions (packaging done, max steps, no progress). See run_full_workflow() / orchestrate().

Privileges: R=Read, C=Create, U=Update, L=Link, M=Move. CooperativeState is shared. OPENAI_API_KEY or HUGGINGFACE_API_KEY.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, List, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith tracing (optional)
# ---------------------------------------------------------------------------
# Install: pip install langsmith
# Enable via environment variables (no hardcoded keys):
#   LANGSMITH_TRACING=true
#   LANGSMITH_API_KEY=<your-key>
#   LANGSMITH_PROJECT=<project-name>   (optional; default from LangSmith)
#   LANGSMITH_ENDPOINT=<url>           (optional; for self-hosted)
#   LANGSMITH_WORKSPACE_ID=<id>        (optional; if relevant)
#
# Windows PowerShell (run before starting the script):
#   $env:LANGSMITH_TRACING="true"
#   $env:LANGSMITH_API_KEY="your-api-key"
#   $env:LANGSMITH_PROJECT="rag-agents"
#
# Run with tracing (demo mode):
#   python rag_agents_cooperative.py --demo
# Run with tracing (all agents, role all):
#   python rag_agents_cooperative.py --role all
# If LANGSMITH_TRACING is not set or false, the app runs normally without tracing.
# ---------------------------------------------------------------------------

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        """No-op decorator when langsmith is not installed."""
        def _decorator(f):
            return f
        if args and len(args) == 1 and callable(args[0]):
            return args[0]
        return _decorator

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


def _setup_langsmith_tracing() -> None:
    """
    Enable LangSmith tracing via environment variables. Fail-safe: if misconfigured
    or unavailable, log a warning and leave tracing disabled (app continues normally).
    Uses LANGSMITH_* vars and sets LANGCHAIN_* so LangChain/LangSmith SDK capture traces.
    """
    try:
        enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("true", "1", "yes")
        if not enabled:
            return
        api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
        if not api_key:
            logger.warning("LangSmith: LANGSMITH_TRACING is true but LANGSMITH_API_KEY is not set; tracing disabled.")
            return
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        project = os.getenv("LANGSMITH_PROJECT", "").strip()
        if project:
            os.environ["LANGCHAIN_PROJECT"] = project
        endpoint = os.getenv("LANGSMITH_ENDPOINT", "").strip()
        if endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        # LANGSMITH_WORKSPACE_ID is used by LangSmith UI; no LANGCHAIN_ equivalent needed for SDK
        logger.info("LangSmith tracing enabled (project=%s).", project or "default")
    except Exception as e:
        logger.warning("LangSmith tracing setup failed: %s; continuing without tracing.", e)


# LangChain imports (0.3.x: langchain.*; 1.x: langchain_classic + langchain_core)
try:
    try:
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.tools import Tool
        from langchain import hub
    except ImportError:
        from langchain_classic.agents import AgentExecutor, create_react_agent
        from langchain_classic.tools import Tool
        from langchain_classic import hub
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    # HuggingFace models (for Mistral via API)
    try:
        from langchain_community.llms import HuggingFaceEndpoint
        HUGGINGFACE_AVAILABLE = True
    except ImportError:
        HUGGINGFACE_AVAILABLE = False
    
    # OpenAI models (fallback)
    try:
        from langchain_openai import ChatOpenAI
        OPENAI_AVAILABLE = True
    except ImportError:
        try:
            from langchain.chat_models import ChatOpenAI
            OPENAI_AVAILABLE = True
        except ImportError:
            OPENAI_AVAILABLE = False
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    raise ImportError("LangChain not available. Install with: pip install langchain langchain-community")

from rag_pipeline import RAGPipeline


@dataclass
class DocSection:
    """Document section with metadata"""
    section_id: str
    title: str
    slide_range: tuple  # (start_slide, end_slide)
    importance_weight: float  # 0.0 to 1.0
    content_summary: Optional[str] = None


@dataclass
class LearningObjective:
    """Learning objective for a section"""
    lo_id: str
    section_id: str
    objective_text: str
    difficulty_level: str  # "beginner", "intermediate", "advanced"


@dataclass
class KnowledgeAtom:
    """Atomic fact/definition/procedure with citations"""
    atom_id: str
    content: str
    atom_type: str  # "fact", "definition", "procedure"
    citations: List[Dict[str, Any]]  # [{"page": int, "source": str, "chunk_id": str}]
    section_id: Optional[str] = None
    lo_id: Optional[str] = None


@dataclass
class EvidenceChunk:
    """Curated chunk with separate relevance (from retrieval) and quality (from curator)."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float = 0.0   # from vector DB (similarity/relevance to query)
    quality_score: float = 0.0     # set by curator: noise, duplication, source trust (0.0–1.0)
    relevance_tags: List[str] = field(default_factory=list)
    used_in_atoms: List[str] = field(default_factory=list)  # atom_ids that use this chunk
    duplicate_of: Optional[str] = None  # if set, this chunk is duplicate of canonical chunk_id (curator/quality dedupe)
    flags: List[str] = field(default_factory=list)  # e.g. "duplicate", "low_quality"


@dataclass
class CandidateQuestion:
    """Raw generated question"""
    q_id: str
    question_text: str
    question_type: str  # "multiple_choice", "open_ended", "true_false"
    options: Optional[List[str]] = None  # For multiple choice
    correct_answer: Optional[str] = None
    difficulty: str = "intermediate"  # "beginner", "intermediate", "advanced"
    topic_tags: List[str] = field(default_factory=list)
    related_atoms: List[str] = field(default_factory=list)  # atom_ids
    evidence_chunks: List[str] = field(default_factory=list)  # chunk_ids
    flags: List[str] = field(default_factory=list)  # e.g. "duplicate", "low_quality"
    duplicate_of: Optional[str] = None  # canonical q_id if this is a duplicate


@dataclass
class ValidatedQuestion:
    """Question that passed verification"""
    q_id: str
    question_text: str
    question_type: str
    correct_answer: str
    difficulty: str
    options: Optional[List[str]] = None
    topic_tags: List[str] = field(default_factory=list)
    related_atoms: List[str] = field(default_factory=list)
    evidence_chunks: List[str] = field(default_factory=list)
    validation_notes: Optional[str] = None
    flags: List[str] = field(default_factory=list)  # e.g. "duplicate", "low_quality"
    duplicate_of: Optional[str] = None  # canonical q_id if this is a duplicate


@dataclass
class RejectedQuestion:
    """Question that failed verification"""
    q_id: str
    question_text: str
    rejection_reason: str
    fix_hints: List[str] = field(default_factory=list)
    original_question: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoverageStats:
    """Coverage statistics for a topic-difficulty-type combination"""
    topic: str
    difficulty: str
    question_type: str
    count: int = 0
    validated_count: int = 0
    rejected_count: int = 0


@dataclass
class CoveragePlan:
    """Target coverage: min validated questions per (difficulty, type). Optional per-topic or per-LO overrides."""
    # Default min count per (difficulty, question_type)
    default_per_difficulty_type: Dict[Tuple[str, str], int] = field(default_factory=dict)
    # Optional: per topic override -> (difficulty, type) -> min
    per_topic: Optional[Dict[str, Dict[Tuple[str, str], int]]] = None
    # Optional: per lo_id override (counts via related_atoms) -> (difficulty, type) -> min
    per_lo: Optional[Dict[str, Dict[Tuple[str, str], int]]] = None


# Default: want at least 2 intermediate MCQ, 1 advanced MCQ, 1 beginner MCQ, 1 open_ended per topic
DEFAULT_COVERAGE_PLAN = CoveragePlan(
    default_per_difficulty_type={
        ("beginner", "multiple_choice"): 1,
        ("intermediate", "multiple_choice"): 2,
        ("advanced", "multiple_choice"): 1,
        ("beginner", "open_ended"): 0,
        ("intermediate", "open_ended"): 1,
        ("advanced", "open_ended"): 0,
        ("beginner", "true_false"): 0,
        ("intermediate", "true_false"): 0,
        ("advanced", "true_false"): 0,
    }
)

# ---------------------------------------------------------------------------
# Orchestrator constants (rule-based decision thresholds)
# ---------------------------------------------------------------------------
MAX_ORCHESTRATION_STEPS = 80
MIN_EVIDENCE_CHUNKS = 3
MIN_ATOMS = 2
MAX_DUPLICATION_SCORE = 0.35
MAX_REJECTION_RATE = 0.75
MIN_VALIDATED_QUESTIONS = 1
MIN_AVG_QUALITY_SCORE = 0.3
NO_PROGRESS_ITERATIONS = 4
MIN_LO_COUNT = 1


@dataclass
class QualityMetrics:
    """Overall quality metrics"""
    rejection_rate: float = 0.0  # rejected / total
    duplication_score: float = 0.0  # 0.0 = no duplicates, 1.0 = all duplicates
    avg_evidence_strength: float = 0.0  # average quality_score of used chunks
    total_questions: int = 0
    validated_questions: int = 0
    rejected_questions: int = 0


@dataclass
class QuizPack:
    """Export-ready quiz structure"""
    pack_id: str
    title: str
    questions: List[ValidatedQuestion]
    metadata: Dict[str, Any] = field(default_factory=dict)
    coverage_summary: Dict[str, Any] = field(default_factory=dict)


# Criteria for curator when evaluating chunk quality (noise, duplication, source trust)
CHUNK_QUALITY_CRITERIA = [
    "Low noise: content is substantive, not boilerplate or off-topic.",
    "No duplication: content does not largely repeat another chunk in the pool.",
    "Source trust: from a credible source (e.g. document section, page) and traceable.",
]

# Curator dedupe: similarity above this threshold => same cluster (duplicate)
CURATE_DEDUPE_THRESHOLD = 0.92

# Atomizer: rules for extracting knowledge atoms from chunks
ATOM_EXTRACTION_RULES = [
    "One fact per atom: no multi-fact paragraphs; split into separate atoms.",
    "Concrete and testable: avoid vague statements or non-testable claims.",
    "Types: 'fact' (single factual claim), 'definition' (concept definition), 'procedure' (step or method).",
    "Content: one or two clear sentences; no bullet lists inside one atom.",
]
ATOM_MIN_LENGTH = 10
ATOM_MAX_LENGTH = 500

# Verification rules used by the Verifier when checking candidate questions against evidence
VERIFICATION_RULES = [
    "Answer supported by evidence: The correct answer must be explicitly or clearly implied by the evidence chunks. For multiple choice, the chosen correct option must match the evidence.",
    "Options consistent: For multiple choice, all options must be distinct and plausible; exactly one must be correct according to evidence; incorrect options must not be supported by evidence.",
    "No hallucinations: The question text and options must not introduce facts, numbers, or claims that are not present in the evidence chunks.",
    "Question clarity: The question must be unambiguous and answerable from the evidence alone.",
]


@dataclass
class CooperativeState:
    """Shared state for cooperative agents"""
    # Document structure
    doc_outline: List[DocSection] = field(default_factory=list)
    
    # Learning objectives
    learning_objectives: List[LearningObjective] = field(default_factory=list)
    
    # Knowledge base
    knowledge_atoms: Dict[str, KnowledgeAtom] = field(default_factory=dict)  # atom_id -> atom
    
    # Evidence pool
    evidence_pool: Dict[str, EvidenceChunk] = field(default_factory=dict)  # chunk_id -> chunk
    
    # Questions
    candidate_questions: Dict[str, CandidateQuestion] = field(default_factory=dict)  # q_id -> question
    validated_questions: Dict[str, ValidatedQuestion] = field(default_factory=dict)  # q_id -> question
    rejected_questions: Dict[str, RejectedQuestion] = field(default_factory=dict)  # q_id -> rejection
    
    # Coverage tracking
    coverage_matrix: Dict[str, CoverageStats] = field(default_factory=dict)  # "topic:difficulty:type" -> stats
    
    # Quality metrics
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    
    # Final output
    final_quiz_packs: List[QuizPack] = field(default_factory=list)
    # Work-in-progress packs: create_quiz_pack → add_questions_to_pack → export_pack
    draft_quiz_packs: Dict[str, QuizPack] = field(default_factory=dict)
    # Checkpointed ID / routing counters (replay-stable; not runtime-only).
    _id_counters: Dict[str, int] = field(default_factory=dict)
    quiz_generation_rounds: int = 0
    # Step-2 MCQ containers must be dataclass fields so LangGraph checkpoint
    # serde preserves them. Monkey-patched attributes are dropped on deserialize,
    # which breaks NOOP/CF resume from boundaries such as
    # before generate_candidate_mcqs_from_blueprints#1 (empty planned blueprints
    # → novel repair/generation LLM requests → ledger misses).
    mcq_blueprints: Dict[str, Any] = field(default_factory=dict)
    candidate_mcqs: Dict[str, Any] = field(default_factory=dict)
    language_reports: Dict[str, Any] = field(default_factory=dict)
    evidence_reports: Dict[str, Any] = field(default_factory=dict)
    cross_cove_reports: Dict[str, Any] = field(default_factory=dict)
    supervisor_decisions: Dict[str, Any] = field(default_factory=dict)
    outcome_assessments: Dict[str, Any] = field(default_factory=dict)
    revision_history: List[Any] = field(default_factory=list)
    mcq_coverage_summary: Dict[str, Any] = field(default_factory=dict)
    section_evidence_status: Dict[str, Any] = field(default_factory=dict)

    def add_section(self, section: DocSection):
        """Add a document section"""
        self.doc_outline.append(section)
    
    def add_learning_objective(self, lo: LearningObjective):
        """Add a learning objective"""
        self.learning_objectives.append(lo)
    
    def add_knowledge_atom(self, atom: KnowledgeAtom):
        """Add a knowledge atom"""
        self.knowledge_atoms[atom.atom_id] = atom
    
    def add_evidence_chunk(self, chunk: EvidenceChunk):
        """Add an evidence chunk"""
        self.evidence_pool[chunk.chunk_id] = chunk
    
    def add_candidate_question(self, question: CandidateQuestion):
        """Add a candidate question"""
        self.candidate_questions[question.q_id] = question
        self.quality_metrics.total_questions += 1
        self._update_coverage(question, action="add")
    
    def validate_question(self, q_id: str, validated: ValidatedQuestion):
        """Move question from candidate to validated"""
        if q_id in self.candidate_questions:
            del self.candidate_questions[q_id]
        self.validated_questions[q_id] = validated
        self.quality_metrics.validated_questions += 1
        self._update_coverage(validated, action="validate")
        self._update_quality_metrics()
    
    def reject_question(self, q_id: str, rejection: RejectedQuestion):
        """Move question from candidate to rejected"""
        # Update coverage for this question's topic/difficulty/type before removing it
        question = self.candidate_questions.get(q_id)
        if question is not None:
            self._update_coverage(question, action="reject")
        if q_id in self.candidate_questions:
            del self.candidate_questions[q_id]
        self.rejected_questions[q_id] = rejection
        self.quality_metrics.rejected_questions += 1
        self._update_quality_metrics()
    
    def _update_coverage(
        self,
        question: Any,
        action: str,
    ) -> None:
        """Update coverage matrix.
        - add: when a candidate is added → increment count only.
        - validate: when a candidate is validated → increment validated_count only.
        - reject: when a candidate is rejected → increment rejected_count only.
        """
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
    
    def _update_quality_metrics(self):
        """Update overall quality metrics"""
        total = self.quality_metrics.total_questions
        if total > 0:
            self.quality_metrics.rejection_rate = self.quality_metrics.rejected_questions / total
        
        # Calculate average evidence strength
        used_chunk_ids = set()
        for q in self.validated_questions.values():
            used_chunk_ids.update(q.evidence_chunks)
        
        if used_chunk_ids:
            scores = [self.evidence_pool[cid].quality_score for cid in used_chunk_ids if cid in self.evidence_pool]
            if scores:
                self.quality_metrics.avg_evidence_strength = sum(scores) / len(scores)
    
    def get_atom(self, atom_id: str) -> Optional[KnowledgeAtom]:
        """Get knowledge atom by ID"""
        return self.knowledge_atoms.get(atom_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[EvidenceChunk]:
        """Get evidence chunk by ID"""
        return self.evidence_pool.get(chunk_id)
    
    def get_question(self, q_id: str) -> Optional[CandidateQuestion]:
        """Get candidate question by ID"""
        return self.candidate_questions.get(q_id)
    
    def generate_id(self, prefix: str) -> str:
        """Deterministic ID from checkpointed counters (replay-stable)."""
        counters = self._id_counters
        if not isinstance(counters, dict):
            counters = {}
            self._id_counters = counters
        n = int(counters.get(prefix, 0)) + 1
        counters[prefix] = n
        return f"{prefix}_{n:08d}"

    def set_doc_outline(self, outline: List["DocSection"]) -> None:
        """Replace doc outline (Structure agent)."""
        self.doc_outline.clear()
        self.doc_outline.extend(outline)

    def get_section(self, section_id: str) -> Optional["DocSection"]:
        """Get section by id."""
        for s in self.doc_outline:
            if s.section_id == section_id:
                return s
        return None

    def get_outline_summary(self) -> str:
        """Summary of doc outline for read-only agents."""
        if not self.doc_outline:
            return "No outline yet."
        return "\n".join(
            f"- {s.section_id}: {s.title} slides{s.slide_range} weight={s.importance_weight}"
            for s in self.doc_outline
        )

    def get_learning_objectives_summary(self) -> str:
        """Summary of learning objectives."""
        if not self.learning_objectives:
            return "No learning objectives yet."
        return "\n".join(
            f"- {lo.lo_id} (section={lo.section_id}): {lo.objective_text[:80]}... difficulty={lo.difficulty_level}"
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
        """Update chunk fields (Curator/Quality agents). Returns True if chunk exists."""
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
        """Set or append flags on a chunk (e.g. 'duplicate', 'low_quality')."""
        if chunk_id not in self.evidence_pool:
            return False
        c = self.evidence_pool[chunk_id]
        current = getattr(c, "flags", None) or []
        if not isinstance(current, list):
            current = []
        if append:
            for f in flags:
                if f not in current:
                    current.append(f)
        else:
            current = list(flags)
        setattr(c, "flags", current)
        return True

    def set_question_flags_and_duplicate(
        self, q_id: str, flags: Optional[List[str]] = None, duplicate_of: Optional[str] = None, validated: bool = False
    ) -> bool:
        """Set flags and/or duplicate_of on a candidate or validated question."""
        if validated:
            q = self.validated_questions.get(q_id)
        else:
            q = self.candidate_questions.get(q_id)
        if not q:
            return False
        if flags is not None:
            setattr(q, "flags", list(flags))
        if duplicate_of is not None:
            setattr(q, "duplicate_of", duplicate_of)
        return True

    def link_chunk_to_question(self, q_id: str, chunk_id: str, validated: bool = False) -> bool:
        """Link chunk to question (curator: this chunk supports question Q). Adds chunk_id to evidence_chunks if not present."""
        if chunk_id not in self.evidence_pool:
            return False
        if validated:
            if q_id not in self.validated_questions:
                return False
            q = self.validated_questions[q_id]
        else:
            if q_id not in self.candidate_questions:
                return False
            q = self.candidate_questions[q_id]
        if chunk_id not in q.evidence_chunks:
            q.evidence_chunks.append(chunk_id)
        return True

    def link_chunk_to_atom(self, chunk_id: str, atom_id: str) -> bool:
        """Record that chunk is used by atom (Curator/Atomizer)."""
        if chunk_id not in self.evidence_pool or atom_id not in self.knowledge_atoms:
            return False
        if atom_id not in self.evidence_pool[chunk_id].used_in_atoms:
            self.evidence_pool[chunk_id].used_in_atoms.append(atom_id)
        return True

    def update_atom_lo(self, atom_id: str, lo_id: str) -> bool:
        """Bind knowledge atom to learning objective (Curriculum agent)."""
        if atom_id not in self.knowledge_atoms:
            return False
        self.knowledge_atoms[atom_id].lo_id = lo_id
        return True

    def update_atom_section(self, atom_id: str, section_id: str) -> bool:
        """Set atom's section (Atomizer)."""
        if atom_id not in self.knowledge_atoms:
            return False
        self.knowledge_atoms[atom_id].section_id = section_id
        return True

    def get_coverage_stats(self) -> Dict[str, CoverageStats]:
        """Return coverage matrix for read/update."""
        return dict(self.coverage_matrix)

    def update_coverage_stats(self, key: str, count: int = 0, validated_count: int = 0, rejected_count: int = 0) -> None:
        """Update or create a coverage stat entry (Coverage Controller)."""
        parts = key.split(":")
        if len(parts) != 3:
            return
        topic, difficulty, qtype = parts[0], parts[1], parts[2]
        if key not in self.coverage_matrix:
            self.coverage_matrix[key] = CoverageStats(topic=topic, difficulty=difficulty, question_type=qtype)
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
        """Update quality metrics (Verifier, Quality agent, etc.)."""
        if rejection_rate is not None:
            self.quality_metrics.rejection_rate = rejection_rate
        if duplication_score is not None:
            self.quality_metrics.duplication_score = duplication_score
        if avg_evidence_strength is not None:
            self.quality_metrics.avg_evidence_strength = avg_evidence_strength

    def append_quiz_pack(self, pack: "QuizPack") -> None:
        """Append to final_quiz_packs (Packager)."""
        self.final_quiz_packs.append(pack)


# --- Privilege system: R=Read, C=Create, U=Update, L=Link, M=Move
class Privilege(str, Enum):
    R = "R"
    C = "C"
    U = "U"
    L = "L"
    M = "M"


ENTITY_TYPES = (
    "DocSection",
    "LearningObjective",
    "EvidenceChunk",
    "KnowledgeAtom",
    "CandidateQuestion",
    "ValidatedQuestion",
    "RejectedQuestion",
    "CoverageStats",
    "QualityMetrics",
    "QuizPack",
    "CooperativeState",
)

# Agent role id -> entity -> set of privileges (from table)
AGENT_PRIVILEGES: Dict[str, Dict[str, Set[str]]] = {
    "structure_outline": {
        "DocSection": {"C", "U"},
        "LearningObjective": {"L"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"L"},
        "CooperativeState": {"U"},
    },
    "curriculum_lo": {
        "DocSection": {"R"},
        "LearningObjective": {"C", "U"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"L"},
        "CooperativeState": {"U"},  # write learning_objectives
    },
    "retriever": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"C"},
        "CooperativeState": {"U"},  # writes evidence_pool
    },
    "evidence_curator": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R", "U"},
        "KnowledgeAtom": {"L"},
        "CandidateQuestion": {"L"},
        "ValidatedQuestion": {"L"},
        "RejectedQuestion": {"L"},
        "QualityMetrics": {"U"},
        "CooperativeState": {"U"},
    },
    "atomizer": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"C", "U", "L"},
        "CooperativeState": {"U"},
    },
    "quiz_generator": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R", "L"},
        "KnowledgeAtom": {"R", "L"},
        "CandidateQuestion": {"C", "U"},
        "CoverageStats": {"U"},
        "QualityMetrics": {"U"},
        "CooperativeState": {"U"},
    },
    "verifier": {
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"R"},
        "CandidateQuestion": {"R", "M"},
        "ValidatedQuestion": {"C"},
        "RejectedQuestion": {"C"},
        "CoverageStats": {"U"},
        "QualityMetrics": {"U"},
        "CooperativeState": {"U"},
    },
    "coverage_controller": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"R"},
        "CandidateQuestion": {"R", "U"},
        "ValidatedQuestion": {"R"},
        "RejectedQuestion": {"R", "U"},
        "CoverageStats": {"R", "U"},
        "QualityMetrics": {"R", "U"},
        "CooperativeState": {"U"},
    },
    "quality_duplicate_detector": {
        "EvidenceChunk": {"R", "U"},
        "KnowledgeAtom": {"R"},
        "CandidateQuestion": {"R", "U"},
        "ValidatedQuestion": {"R", "U"},
        "RejectedQuestion": {"R"},
        "CoverageStats": {"R"},
        "QualityMetrics": {"R", "U"},
        "CooperativeState": {"U"},
    },
    "packager_export": {
        "DocSection": {"R"},
        "LearningObjective": {"R"},
        "EvidenceChunk": {"R"},
        "KnowledgeAtom": {"R"},
        "ValidatedQuestion": {"R"},
        "RejectedQuestion": {"R"},
        "CoverageStats": {"R"},
        "QualityMetrics": {"R"},
        "QuizPack": {"C"},
        "CooperativeState": {"U"},
    },
}


def _agent_can(agent_id: str, entity: str, privilege: str) -> bool:
    """Check if agent has privilege on entity."""
    if agent_id not in AGENT_PRIVILEGES:
        return False
    return privilege in AGENT_PRIVILEGES[agent_id].get(entity, set())


class RAGAgents:
    """LangChain Agents with specialized roles for RAG tasks"""
    
    def __init__(
        self,
        pdf_id: str = "141",
        vector_store_path: str = "vector_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-4.1-mini-2025-04-14",
        model_type: str = "openai",
        temperature: float = 0.0,
        device: str = "cpu",
        openai_api_key: Optional[str] = None,
        huggingface_api_key: Optional[str] = None
    ):
        """
        Initialize RAG Agents (API only, no local model loading)
        
        Args:
            pdf_id: PDF ID to work with
            vector_store_path: Path to vector store
            embedding_model: Embedding model name
            llm_model: LLM model name (HuggingFace model ID or OpenAI model)
            model_type: "huggingface" for Mistral via API or "openai" for GPT models
            temperature: LLM temperature
            device: Not used (kept for backward compatibility, models run via API)
            openai_api_key: OpenAI API key (required if using OpenAI models)
            huggingface_api_key: HuggingFace API key (required if using HuggingFace Inference API)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available. Install required packages.")
        
        _setup_langsmith_tracing()
        
        self.pdf_id = pdf_id
        self.model_type = model_type
        self.llm_model = llm_model
        
        # Initialize cooperative state
        self.state = CooperativeState()
        self.coverage_plan: Optional[CoveragePlan] = DEFAULT_COVERAGE_PLAN
        self.execution_log: List[Dict[str, Any]] = []
        self.orchestration_log: List[Dict[str, Any]] = []

        self.llm = self._initialize_llm(
            model_type, llm_model, temperature, device,
            openai_api_key, huggingface_api_key
        )
        
        # Initialize RAG pipeline
        if not Path(vector_store_path).is_absolute():
            vector_store_path = str(Path(__file__).parent / vector_store_path)
        
        self.rag_pipeline = RAGPipeline(
            vector_store_path=vector_store_path,
            embedding_model=embedding_model
        )
        
        # Load vector store
        try:
            self.rag_pipeline.load_existing_index(collection_name=pdf_id)
            print(f"[OK] Loaded vector store for PDF {pdf_id}")
        except Exception as e:
            print(f"[WARN] Vector store not found. Please run RAG pipeline first.")
            print(f"  Run: python rag/rag_pipeline.py --pdf-id {pdf_id}")
            raise
        
        # Create agents
        self._create_agents()

    def log_step(self, agent: str, tool: str, input_data: Any = None, output: Any = None) -> None:
        """Append one step to execution log for tracing agent/tool runs."""
        self.execution_log.append({
            "step": len(self.execution_log) + 1,
            "agent": agent,
            "tool": tool,
            "input": input_data,
            "output": str(output)[:300] if output is not None else "",
        })
    
    def _langsmith_config(
        self,
        agent_id: Optional[str] = None,
        query: Optional[str] = None,
        topic: Optional[str] = None,
        statement: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Build run config with metadata and tags for LangSmith traces."""
        metadata: Dict[str, Any] = {
            "pdf_id": self.pdf_id,
            "model_type": self.model_type,
            "llm_model": getattr(self, "llm_model", ""),
        }
        if agent_id is not None:
            metadata["agent_id"] = agent_id
        if query is not None:
            metadata["query"] = query[:500] if isinstance(query, str) else str(query)[:500]
        if topic is not None:
            metadata["topic"] = topic[:500] if isinstance(topic, str) else str(topic)[:500]
        if statement is not None:
            metadata["statement"] = statement[:500] if isinstance(statement, str) else str(statement)[:500]
        metadata.update(extra)
        tags = [f"agent:{agent_id}"] if agent_id else []
        return {"metadata": metadata, "tags": tags}
    
    def _initialize_llm(self, model_type: str, llm_model: str, temperature: float,
                        device: str, openai_key: Optional[str], hf_key: Optional[str]):
        """Initialize LLM based on model type (API only, no local loading)"""
        if model_type == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("HuggingFace not available. Install: pip install langchain-community")
            
            # Use HuggingFace Inference API
            hf_token = hf_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not hf_token:
                raise ValueError(
                    "HuggingFace API key required for Inference API. "
                    "Set HUGGINGFACE_API_KEY env var or pass huggingface_api_key parameter."
                )
            
            # Set API key as environment variables (HuggingFace libraries check multiple names)
            # HUGGINGFACEHUB_API_TOKEN is the standard name used by huggingface_hub
            os.environ["HUGGINGFACE_API_KEY"] = hf_token
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
            
            print(f"Using HuggingFace Inference API for: {llm_model}")
            # Use repo_id - it automatically routes to the correct endpoint
            # The API key should be read from HUGGINGFACEHUB_API_TOKEN env var
            try:
                llm = HuggingFaceEndpoint(
                    repo_id=llm_model,
                    temperature=temperature,
                    max_new_tokens=512
                )
                print(f"[OK] Connected to HuggingFace Inference API: {llm_model}")
                return llm
            except (ValueError, RuntimeError, StopIteration) as e:
                error_str = str(e)
                if ("not supported" in error_str or "featherless-ai" in error_str.lower() or 
                    "StopIteration" in str(type(e).__name__) or "provider" in error_str.lower()):
                    error_msg = (
                        f"\n[ERROR] Model {llm_model} is not available or supported via HuggingFace Inference API.\n"
                        f"  Error: {error_str}\n\n"
                        f"  [WARN] HuggingFace Inference API has limited model support.\n\n"
                        f"  [OK] RECOMMENDED: Use OpenAI instead (more reliable for agents):\n"
                        f"     python rag_agents_example.py --model-type openai --api-key YOUR_OPENAI_KEY\n\n"
                        f"  Alternative HuggingFace models (may work):\n"
                        f"     --model google/flan-t5-base\n"
                        f"     --model microsoft/DialoGPT-medium\n"
                    )
                    raise ValueError(error_msg) from e
                raise
        
        elif model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install: pip install langchain-openai")
            
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            elif not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "OpenAI API key required. "
                    "Set OPENAI_API_KEY env var or pass openai_api_key parameter."
                )
            
            return ChatOpenAI(model=llm_model, temperature=temperature)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'huggingface' or 'openai'")
    
    def _get_llm_response(
        self,
        prompt: str,
        *,
        llm_layer: str = "rag_backend",
        llm_component: str = "",
    ) -> str:
        """Get response from LLM and handle different response types."""
        try:
            from mas_langgraph.replay.llm_determinism import workflow_llm_context
        except Exception:
            workflow_llm_context = None  # type: ignore[assignment]

        def _call() -> str:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        if workflow_llm_context is None:
            return _call()
        with workflow_llm_context(layer=llm_layer, component=llm_component):
            return _call()
    
    @traceable(name="retrieve_chunks")
    def _retrieve_chunks(self, query: str, k: int = 5) -> str:
        """Retrieve relevant chunks from vector DB and add to evidence pool"""
        self.log_step("retriever", "retrieve_chunks", {"query": query, "k": k})
        try:
            results = self.rag_pipeline.search(query, k=k)
            self.log_step("retriever", "retrieve_chunks_result", {"chunks": len(results) if results else 0})
            if not results:
                return "No relevant chunks found."
            
            chunks_info = []
            for i, (doc, score) in enumerate(results, 1):
                metadata = doc.metadata
                
                # Create chunk ID and add to evidence pool
                chunk_id = self.state.generate_id("chunk")
                evidence_chunk = EvidenceChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    metadata=metadata,
                    relevance_score=float(score),
                    quality_score=float(score),  # default until curator evaluates
                    relevance_tags=[query.lower()],
                )
                self.state.add_evidence_chunk(evidence_chunk)
                
                chunk_info = (
                    f"\nChunk {i} (ID: {chunk_id}, Relevance Score: {score:.4f}):\n"
                    f"- Source: {metadata.get('source_file', 'unknown')}\n"
                    f"- Page: {metadata.get('page_number', 'unknown')}\n"
                    f"- Title: {metadata.get('title', 'unknown')}\n"
                    f"- Content: {doc.page_content[:500]}...\n"
                )
                chunks_info.append(chunk_info)
            
            return "\n".join(chunks_info)
        except Exception as e:
            return f"Error retrieving chunks: {str(e)}"
    
    def _summarize(self, query: str) -> str:
        """Generate summary of document or specific sections"""
        try:
            results = self.rag_pipeline.search(query, k=10)
            if not results:
                return "No content found to summarize."
            
            combined_text = "\n\n".join([
                f"[Page {doc.metadata.get('page_number', '?')}]: {doc.page_content}"
                for doc, _ in results
            ])
            
            prompt = (
                "Based on the following content from the document, create a comprehensive summary.\n\n"
                f"Content:\n{combined_text[:3000]}\n\n"
                "Create a well-structured summary that covers the main points."
            )
            
            return self._get_llm_response(prompt)
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    # --- Quiz generation: structured JSON + robust parsing
    QUIZ_JSON_SCHEMA_MC = (
        'Return a single JSON array of objects. Each object must have: '
        '"question_text" (string), "options" (array of 4 strings, e.g. "A) ..."), '
        '"correct_answer" (string: either the letter "A"/"B"/"C"/"D" or the exact option text), '
        '"difficulty" (string: "beginner"|"intermediate"|"advanced"). '
        'No other text outside the JSON.'
    )
    QUIZ_JSON_SCHEMA_OPEN = (
        'Return a single JSON array of objects. Each object must have: '
        '"question_text" (string), "expected_answer" (string, key points), '
        '"difficulty" (string: "beginner"|"intermediate"|"advanced"). '
        'No other text outside the JSON.'
    )

    def _parse_quiz_json(self, response: str, question_type: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response; strip markdown code blocks if present."""
        text = response.strip()
        # Strip ```json ... ``` or ``` ... ```
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            text = m.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find first '[' and last ']'
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                data = json.loads(text[start : end + 1])
            else:
                return []
        if not isinstance(data, list):
            return []
        return data

    def _normalize_correct_answer(self, options: List[str], correct_answer: Optional[str]) -> Optional[str]:
        """Ensure correct_answer matches one of the options (letter or full text). Returns option text or None."""
        if not options:
            return correct_answer
        if not correct_answer or not str(correct_answer).strip():
            return options[0]
        raw = str(correct_answer).strip().upper()
        # Letter only: A, B, C, D -> index 0,1,2,3
        if len(raw) == 1 and raw in "ABCD":
            idx = ord(raw) - ord("A")
            if 0 <= idx < len(options):
                return options[idx]
        # Full option text: match exactly
        for opt in options:
            if opt.strip() == correct_answer.strip():
                return opt
            # Match "A) text" by letter
            if opt.strip().upper().startswith(raw[0] + ")"):
                return opt
        return options[0]

    @traceable(name="generate_quiz")
    def _generate_quiz(
        self,
        topic: str,
        question_type: str = "multiple_choice",
        num_questions: int = 3,
        target_difficulty: Optional[str] = None,
    ) -> str:
        """Generate quiz questions via structured JSON; uses section-focused RAG retrieval."""
        self.log_step("quiz_generator", "generate_quiz", {
            "topic": topic, "question_type": question_type, "num_questions": num_questions,
        })
        try:
            topic_l = (topic or "course material").lower()
            search_query = f"{topic} key concepts definitions examples"
            chunk_ids: List[str] = []
            combined_text: List[str] = []

            for cid, chunk in list(self.state.evidence_pool.items()):
                if getattr(chunk, "duplicate_of", None) is not None:
                    continue
                tags = [t.lower() for t in (chunk.relevance_tags or [])]
                if topic_l in (chunk.content or "").lower()[:300] or any(
                    topic_l in t or t in topic_l for t in tags
                ):
                    chunk_ids.append(cid)
                    combined_text.append(
                        f"[Page {chunk.metadata.get('page_number', '?')}]: {chunk.content[:600]}"
                    )
                if len(chunk_ids) >= 8:
                    break

            if len(chunk_ids) < 3:
                results = self.rag_pipeline.search(search_query, k=8)
                if not results:
                    return f"No content found for quiz generation on topic: {topic}"

                for doc, score in results:
                    body_key = " ".join(doc.page_content.strip().lower().split())[:500]
                    existing_id = None
                    for ecid, c in list(self.state.evidence_pool.items()):
                        if " ".join(c.content.strip().lower().split())[:500] == body_key:
                            existing_id = ecid
                            break
                    if existing_id:
                        if existing_id not in chunk_ids:
                            chunk_ids.append(existing_id)
                            c = self.state.get_chunk(existing_id)
                            if c:
                                combined_text.append(
                                    f"[Page {c.metadata.get('page_number', '?')}]: {c.content[:600]}"
                                )
                        continue
                    chunk_id = self.state.generate_id("chunk")
                    evidence_chunk = EvidenceChunk(
                        chunk_id=chunk_id,
                        content=doc.page_content,
                        metadata=doc.metadata,
                        relevance_score=float(score),
                        quality_score=float(score),
                        relevance_tags=[topic_l],
                    )
                    self.state.add_evidence_chunk(evidence_chunk)
                    chunk_ids.append(chunk_id)
                    combined_text.append(
                        f"[Page {doc.metadata.get('page_number', '?')}]: {doc.page_content[:600]}"
                    )
                    if len(chunk_ids) >= 8:
                        break

            if not chunk_ids:
                return f"No evidence chunks available for topic: {topic}"

            combined_text_str = "\n\n".join(combined_text)[:3000]
            n = max(1, min(int(num_questions), 5))
            diff_hint = (target_difficulty or "intermediate").strip().lower()
            if diff_hint not in ("beginner", "intermediate", "advanced"):
                diff_hint = "intermediate"

            if question_type == "multiple_choice":
                prompt = (
                    f"Based ONLY on the course material below, generate exactly {n} multiple-choice "
                    f"questions about \"{topic}\" with difficulty \"{diff_hint}\". "
                    "Questions must be answerable from the content — no general knowledge or trivia. "
                    f"{self.QUIZ_JSON_SCHEMA_MC}\n\n"
                    f"Content:\n{combined_text_str}\n\n"
                    "Example: [{\"question_text\": \"What is X?\", \"options\": [\"A) One\", \"B) Two\", \"C) Three\", \"D) Four\"], \"correct_answer\": \"B\", \"difficulty\": \"intermediate\"}]"
                )
            else:
                prompt = (
                    f"Based ONLY on the course material below, generate exactly {n} open-ended "
                    f"questions about \"{topic}\" with difficulty \"{diff_hint}\". "
                    "Questions must be answerable from the content — no general knowledge or trivia. "
                    f"{self.QUIZ_JSON_SCHEMA_OPEN}\n\n"
                    f"Content:\n{combined_text_str}\n\n"
                    "Example: [{\"question_text\": \"Explain X.\", \"expected_answer\": \"Key points...\", \"difficulty\": \"intermediate\"}]"
                )

            response = self._get_llm_response(prompt)
            items = self._parse_quiz_json(response, question_type)
            questions_created = []
            topic_tag = topic_l

            for item in items:
                if not isinstance(item, dict):
                    continue
                question_text = (item.get("question_text") or "").strip()
                if not question_text:
                    continue
                difficulty = (item.get("difficulty") or diff_hint).strip().lower()
                if difficulty not in ("beginner", "intermediate", "advanced"):
                    difficulty = diff_hint

                if question_type == "multiple_choice":
                    options_raw = item.get("options")
                    options = [str(o).strip() for o in (options_raw if isinstance(options_raw, list) else []) if str(o).strip()]
                    if len(options) < 2:
                        continue
                    correct_answer = self._normalize_correct_answer(options, item.get("correct_answer"))
                    q = CandidateQuestion(
                        q_id=self.state.generate_id("q"),
                        question_text=question_text,
                        question_type="multiple_choice",
                        options=options,
                        correct_answer=correct_answer,
                        difficulty=difficulty,
                        topic_tags=[topic_tag],
                        evidence_chunks=chunk_ids[:3],
                    )
                else:
                    expected = (item.get("expected_answer") or "").strip()
                    q = CandidateQuestion(
                        q_id=self.state.generate_id("q"),
                        question_text=question_text,
                        question_type="open_ended",
                        options=None,
                        correct_answer=expected or None,
                        difficulty=difficulty,
                        topic_tags=[topic_tag],
                        evidence_chunks=chunk_ids[:3],
                    )
                self.state.add_candidate_question(q)
                questions_created.append(q.q_id)
                if len(questions_created) >= n:
                    break

            return f"Generated {len(questions_created)} questions on '{topic}'. IDs: {', '.join(questions_created)}"
        except Exception as e:
            return f"Error generating quiz: {str(e)}"
    
    def _explain(self, question: str) -> str:
        """Answer questions with traceable references"""
        try:
            results = self.rag_pipeline.search(question, k=5)
            if not results:
                return "No relevant information found to answer the question."
            
            context_parts = []
            for doc, score in results:
                metadata = doc.metadata
                context_parts.append(
                    f"[Source: {metadata.get('source_file', 'unknown')}, "
                    f"Page {metadata.get('page_number', '?')}, "
                    f"Relevance: {score:.4f}]\n{doc.page_content}"
                )
            
            combined_context = "\n\n---\n\n".join(context_parts)
            
            prompt = (
                "Answer the following question based on the provided context. "
                "Include specific page references and citations.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{combined_context[:4000]}\n\n"
                "Provide a clear, comprehensive answer with:\n"
                "1. Direct answer to the question\n"
                "2. Supporting details from the document\n"
                "3. Page references for each claim (format: [Page X])\n"
                "4. Source file name"
            )
            
            return self._get_llm_response(prompt)
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    @traceable(name="verify")
    def _verify(self, statement: str) -> str:
        """Verify factuality against source"""
        try:
            results = self.rag_pipeline.search(statement, k=5)
            if not results:
                return "⚠️ Cannot verify: No relevant content found in source."
            
            combined_text = "\n\n".join([
                f"[Page {doc.metadata.get('page_number', '?')}]: {doc.page_content}"
                for doc, _ in results
            ])
            
            prompt = (
                "Verify if the following statement is factually correct based on the source document.\n"
                "Provide:\n"
                "1. Verification result (TRUE/FALSE/PARTIALLY TRUE)\n"
                "2. Evidence from source\n"
                "3. Page references\n"
                "4. Any discrepancies or hallucinations\n\n"
                f"Statement to verify: {statement}\n\n"
                f"Source content:\n{combined_text[:3000]}"
            )
            
            return self._get_llm_response(prompt)
        except Exception as e:
            return f"Error verifying statement: {str(e)}"
    
    def _add_knowledge_atom(self, content: str, atom_type: str = "fact", 
                           chunk_ids: Optional[List[str]] = None, 
                           section_id: Optional[str] = None, 
                           lo_id: Optional[str] = None) -> str:
        """Add a knowledge atom to the state"""
        atom_id = self.state.generate_id("atom")
        citations = []
        if chunk_ids is None:
            chunk_ids = []
        for chunk_id in chunk_ids:
            chunk = self.state.get_chunk(chunk_id)
            if chunk:
                citations.append({
                    "page": chunk.metadata.get("page_number", "unknown"),
                    "source": chunk.metadata.get("source_file", "unknown"),
                    "chunk_id": chunk_id
                })
        
        atom = KnowledgeAtom(
            atom_id=atom_id,
            content=content,
            atom_type=atom_type,
            citations=citations,
            section_id=section_id,
            lo_id=lo_id
        )
        self.state.add_knowledge_atom(atom)
        return f"Added knowledge atom {atom_id}: {content[:100]}..."
    
    def _get_state_summary(self) -> str:
        """Get summary of current state"""
        return (
            f"State Summary:\n"
            f"- Sections: {len(self.state.doc_outline)}\n"
            f"- Learning Objectives: {len(self.state.learning_objectives)}\n"
            f"- Knowledge Atoms: {len(self.state.knowledge_atoms)}\n"
            f"- Evidence Chunks: {len(self.state.evidence_pool)}\n"
            f"- Candidate Questions: {len(self.state.candidate_questions)}\n"
            f"- Validated Questions: {len(self.state.validated_questions)}\n"
            f"- Rejected Questions: {len(self.state.rejected_questions)}\n"
            f"- Quality Metrics: Rejection Rate: {self.state.quality_metrics.rejection_rate:.2%}, "
            f"Avg Evidence Strength: {self.state.quality_metrics.avg_evidence_strength:.2f}"
        )
    
    def _get_atom(self, atom_id: str) -> str:
        """Get knowledge atom by ID"""
        atom = self.state.get_atom(atom_id)
        if atom:
            return f"Atom {atom_id}: {atom.content}\nType: {atom.atom_type}\nCitations: {len(atom.citations)}"
        return f"Atom {atom_id} not found"
    
    def _get_chunk(self, chunk_id: str) -> str:
        """Get evidence chunk by ID"""
        chunk = self.state.get_chunk(chunk_id)
        if chunk:
            dup = getattr(chunk, "duplicate_of", None)
            dup_str = f" duplicate_of={dup}" if dup else ""
            return (
                f"Chunk {chunk_id}: {chunk.content[:200]}...\n"
                f"Relevance (vector): {chunk.relevance_score:.2f} | Quality (curator): {chunk.quality_score:.2f}{dup_str}"
            )
        return f"Chunk {chunk_id} not found"
    
    def _get_coverage(self) -> str:
        """Get coverage matrix summary"""
        if not self.state.coverage_matrix:
            return "No coverage data yet"
        
        lines = ["Coverage Matrix:"]
        for key, stats in self.state.coverage_matrix.items():
            lines.append(
                f"  {stats.topic} ({stats.difficulty}, {stats.question_type}): "
                f"{stats.validated_count} validated, {stats.rejected_count} rejected"
            )
        return "\n".join(lines)

    # --- Coverage Controller: plan, gaps, next targets
    def _get_validated_counts_by_lo(self) -> Dict[Tuple[str, str, str], int]:
        """Count validated questions per (lo_id, difficulty, type) via related_atoms."""
        counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        for q in self.state.validated_questions.values():
            lo_ids = set()
            for atom_id in getattr(q, "related_atoms", []) or []:
                atom = self.state.get_atom(atom_id)
                if atom and getattr(atom, "lo_id", None):
                    lo_ids.add(atom.lo_id)
            if not lo_ids:
                lo_ids.add("_no_lo")
            key = (q.difficulty, q.question_type)
            for lo_id in lo_ids:
                counts[(lo_id, key[0], key[1])] += 1
        return dict(counts)

    def _get_candidate_counts_by_lo(self) -> Dict[Tuple[str, str, str], int]:
        """Count candidate questions per (lo_id, difficulty, type) via related_atoms."""
        counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        for q in self.state.candidate_questions.values():
            lo_ids = set()
            for atom_id in getattr(q, "related_atoms", []) or []:
                atom = self.state.get_atom(atom_id)
                if atom and getattr(atom, "lo_id", None):
                    lo_ids.add(atom.lo_id)
            if not lo_ids:
                lo_ids.add("_no_lo")
            key = (q.difficulty, q.question_type)
            for lo_id in lo_ids:
                counts[(lo_id, key[0], key[1])] += 1
        return dict(counts)

    def _get_current_coverage_per_key(self) -> Dict[str, Tuple[int, int]]:
        """For each topic:difficulty:type key, return (validated_count, candidate_count). Coverage = validated + candidate."""
        validated: Dict[str, int] = defaultdict(int)
        candidate: Dict[str, int] = defaultdict(int)
        for q in self.state.validated_questions.values():
            topics = getattr(q, "topic_tags", []) or []
            if not topics:
                topics = ["general"]
            diff = getattr(q, "difficulty", "intermediate")
            qtype = getattr(q, "question_type", "multiple_choice")
            for topic in topics:
                key = f"{topic}:{diff}:{qtype}"
                validated[key] += 1
        for q in self.state.candidate_questions.values():
            topics = getattr(q, "topic_tags", []) or []
            if not topics:
                topics = ["general"]
            diff = getattr(q, "difficulty", "intermediate")
            qtype = getattr(q, "question_type", "multiple_choice")
            for topic in topics:
                key = f"{topic}:{diff}:{qtype}"
                candidate[key] += 1
        return {k: (validated[k], candidate[k]) for k in set(validated) | set(candidate)}

    def compute_coverage_gaps(
        self,
        plan: Optional[CoveragePlan] = None,
    ) -> List[Dict[str, Any]]:
        """Compare current coverage (validated + candidate) to plan; return ranked list of gaps. Controller can then decide: validate more or generate new."""
        plan = plan or getattr(self, "coverage_plan", None) or DEFAULT_COVERAGE_PLAN
        gaps: List[Dict[str, Any]] = []
        current_per_key = self._get_current_coverage_per_key()
        topics_seen = set(stats.topic for stats in self.state.coverage_matrix.values()) or {"general"}
        for k, (v, c) in current_per_key.items():
            if ":" in k:
                topics_seen.add(k.split(":")[0])

        # Topic-based: for each topic and each (difficulty, type) in plan, compare current to target
        for (diff, qtype), target in plan.default_per_difficulty_type.items():
            if target <= 0:
                continue
            for topic in topics_seen:
                key = f"{topic}:{diff}:{qtype}"
                val_count, cand_count = current_per_key.get(key, (0, 0))
                current = val_count + cand_count
                if plan.per_topic and topic in (plan.per_topic or {}):
                    target_here = plan.per_topic[topic].get((diff, qtype), target)
                else:
                    target_here = target
                if target_here <= 0:
                    continue
                if current < target_here:
                    gaps.append({
                        "kind": "topic",
                        "topic": topic,
                        "lo_id": None,
                        "difficulty": diff,
                        "question_type": qtype,
                        "validated_count": val_count,
                        "candidate_count": cand_count,
                        "current": current,
                        "target": target_here,
                        "gap": target_here - current,
                    })

        # LO-based gaps if plan has per_lo (coverage = validated + candidate)
        if plan.per_lo:
            val_lo = self._get_validated_counts_by_lo()
            cand_lo = self._get_candidate_counts_by_lo()
            for lo_id, dt_targets in plan.per_lo.items():
                for (diff, qtype), target in dt_targets.items():
                    if target <= 0:
                        continue
                    v = val_lo.get((lo_id, diff, qtype), 0)
                    c = cand_lo.get((lo_id, diff, qtype), 0)
                    current = v + c
                    if current < target:
                        gaps.append({
                            "kind": "lo",
                            "topic": None,
                            "lo_id": lo_id,
                            "difficulty": diff,
                            "question_type": qtype,
                            "validated_count": v,
                            "candidate_count": c,
                            "current": current,
                            "target": target,
                            "gap": target - current,
                        })

        # Rank by gap (largest first), then by difficulty order, then topic/lo
        diff_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
        def rank(g: Dict[str, Any]) -> tuple:
            return (-g["gap"], diff_order.get(g["difficulty"], 1), str(g.get("topic") or g.get("lo_id") or ""))
        gaps.sort(key=rank)
        return gaps

    def get_next_targets(self, n: int = 5, plan: Optional[CoveragePlan] = None) -> List[Dict[str, Any]]:
        """Return top n targets from coverage gaps. Coverage = validated + candidate; controller can validate more or generate new."""
        gaps = self.compute_coverage_gaps(plan)
        out = []
        for g in gaps[:n]:
            val_count = g.get("validated_count", 0)
            cand_count = g.get("candidate_count", 0)
            hint = f" (current: {val_count} validated, {cand_count} candidate)"
            action = (
                f"Need {g['gap']} more {g['question_type']} ({g['difficulty']}) for '{g.get('topic') or g.get('lo_id')}'"
                f"{hint}. Option: validate candidates or generate new."
            )
            if g.get("lo_id") and g["kind"] == "lo":
                action += f" [LO: {g['lo_id']}]"
            out.append({
                "topic": g.get("topic") or g.get("lo_id"),
                "difficulty": g["difficulty"],
                "question_type": g["question_type"],
                "validated_count": val_count,
                "candidate_count": cand_count,
                "current": g["current"],
                "target": g["target"],
                "gap": g["gap"],
                "action": action,
            })
        return out

    def _tool_get_next_targets(self, agent_id: str, n_str: str = "5") -> str:
        if not _agent_can(agent_id, "CoverageStats", "R"):
            return "Permission denied."
        try:
            n = int(n_str.strip() or "5")
            n = max(1, min(20, n))
        except ValueError:
            n = 5
        targets = self.get_next_targets(n=n)
        if not targets:
            return "No coverage gaps; targets are met or no plan configured."
        lines = [t["action"] for t in targets]
        return "Next targets:\n" + "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))

    # --- Quality / Duplicate Detector: detect duplicates, flag, compute duplication_score
    DEFAULT_DUPLICATE_CHUNK_THRESHOLD = 0.92
    DEFAULT_DUPLICATE_QUESTION_THRESHOLD = 0.90

    @traceable(name="detect_duplicate_chunks")
    def detect_duplicate_chunks(self, threshold: float = 0.92) -> str:
        """Cluster chunks by embedding similarity; mark non-canonical as duplicate_of and add 'duplicate' flag. Returns report."""
        chunk_ids = [
            cid for cid, c in self.state.evidence_pool.items()
            if getattr(c, "duplicate_of", None) is None
        ]
        if len(chunk_ids) <= 1:
            self._update_duplication_score()
            return "No duplicate chunks detected (pool too small or already deduped)."
        clusters = self._cluster_duplicates(chunk_ids, threshold)
        duplicate_pairs = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            canonical = self._pick_canonical(cluster)
            for cid in cluster:
                if cid != canonical:
                    self.state.update_evidence_chunk(cid, duplicate_of=canonical)
                    self.state.set_chunk_flags(cid, ["duplicate"], append=True)
                    duplicate_pairs.append((cid, canonical))
        self._update_duplication_score()
        if not duplicate_pairs:
            return "No duplicate chunks detected."
        lines = ["Duplicate chunks (id -> canonical):"] + [f"  {dup} -> {can}" for dup, can in duplicate_pairs]
        lines.append("Suggest: keep canonical chunks; remove or merge duplicates in evidence_pool.")
        return "\n".join(lines)

    def _cluster_questions(
        self, q_ids: List[str], texts: List[str], threshold: float
    ) -> List[List[str]]:
        """Cluster question IDs by embedding similarity of their texts. Returns list of clusters."""
        if len(q_ids) <= 1 or len(texts) != len(q_ids):
            return [[q] for q in q_ids] if q_ids else []
        vectors = self._embed_texts(texts)
        if len(vectors) != len(q_ids):
            return [[q] for q in q_ids]
        parent = {q: q for q in q_ids}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(q_ids)):
            for j in range(i + 1, len(q_ids)):
                if self._cosine_sim(vectors[i], vectors[j]) >= threshold:
                    union(q_ids[i], q_ids[j])
        clusters: Dict[str, List[str]] = defaultdict(list)
        for q in q_ids:
            clusters[find(q)].append(q)
        return list(clusters.values())

    def _pick_canonical_question(self, q_ids: List[str], prefer_validated: bool = True) -> str:
        """Pick canonical question in cluster: prefer validated, then by evidence strength."""
        best = q_ids[0]
        best_score = -1.0
        for qid in q_ids:
            q = self.state.validated_questions.get(qid) or self.state.candidate_questions.get(qid)
            if not q:
                continue
            score = 1.0 if (prefer_validated and qid in self.state.validated_questions) else 0.5
            chunks = getattr(q, "evidence_chunks", []) or []
            for cid in chunks[:3]:
                c = self.state.get_chunk(cid)
                if c:
                    score += getattr(c, "quality_score", 0.0) * 0.1
            if score > best_score:
                best_score = score
                best = qid
        return best

    @traceable(name="detect_duplicate_questions")
    def detect_duplicate_questions(self, threshold: float = 0.90) -> str:
        """Detect duplicate questions (candidate + validated) by stem similarity; set duplicate_of and flags. Returns report."""
        all_q = list(self.state.candidate_questions.items()) + list(self.state.validated_questions.items())
        if not all_q:
            self._update_duplication_score()
            return "No questions to check."
        q_ids = [qid for qid, _ in all_q]
        texts = []
        for qid in q_ids:
            q = self.state.candidate_questions.get(qid) or self.state.validated_questions.get(qid)
            stem = (getattr(q, "question_text", "") or "")[:500]
            texts.append(stem)
        clusters = self._cluster_questions(q_ids, texts, threshold)
        duplicate_pairs = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            canonical = self._pick_canonical_question(cluster)
            for qid in cluster:
                if qid != canonical:
                    self.state.set_question_flags_and_duplicate(
                        qid, flags=["duplicate"], duplicate_of=canonical,
                        validated=(qid in self.state.validated_questions),
                    )
                    duplicate_pairs.append((qid, canonical))
        self._update_duplication_score()
        if not duplicate_pairs:
            return "No duplicate questions detected."
        lines = ["Duplicate questions (id -> canonical):"] + [f"  {dup} -> {can}" for dup, can in duplicate_pairs]
        lines.append("Suggest: keep canonical; remove duplicates or reject duplicates (move to rejected) / regenerate.")
        return "\n".join(lines)

    def _update_duplication_score(self) -> None:
        """Set quality_metrics.duplication_score = (# items with duplicate_of set) / (total items)."""
        total = 0
        dup_count = 0
        for c in self.state.evidence_pool.values():
            total += 1
            if getattr(c, "duplicate_of", None) is not None:
                dup_count += 1
        for q in self.state.candidate_questions.values():
            total += 1
            if getattr(q, "duplicate_of", None) is not None:
                dup_count += 1
        for q in self.state.validated_questions.values():
            total += 1
            if getattr(q, "duplicate_of", None) is not None:
                dup_count += 1
        if total > 0:
            self.state.update_quality_metrics_fields(duplication_score=dup_count / total)
        else:
            self.state.update_quality_metrics_fields(duplication_score=0.0)

    def get_duplicate_report(self) -> str:
        """Return a summary of duplicate chunks and questions and suggestions."""
        chunk_dups = [(cid, getattr(c, "duplicate_of", None)) for cid, c in self.state.evidence_pool.items() if getattr(c, "duplicate_of", None)]
        cand_dups = [(qid, getattr(q, "duplicate_of", None)) for qid, q in self.state.candidate_questions.items() if getattr(q, "duplicate_of", None)]
        val_dups = [(qid, getattr(q, "duplicate_of", None)) for qid, q in self.state.validated_questions.items() if getattr(q, "duplicate_of", None)]
        lines = [
            f"Duplication score: {self.state.quality_metrics.duplication_score:.2%}",
            f"Chunk duplicates: {len(chunk_dups)} (duplicate_of set)",
            f"Candidate question duplicates: {len(cand_dups)}",
            f"Validated question duplicates: {len(val_dups)}",
        ]
        if chunk_dups:
            lines.append("Chunk IDs that are duplicates: " + ", ".join(cid for cid, _ in chunk_dups))
        if cand_dups or val_dups:
            lines.append("Question IDs that are duplicates: " + ", ".join(qid for qid, _ in cand_dups + val_dups))
        lines.append("Suggest: remove/merge duplicate chunks; reject duplicate questions or regenerate.")
        return "\n".join(lines)

    def _tool_detect_duplicate_chunks(self, agent_id: str, threshold_str: str = "") -> str:
        if not _agent_can(agent_id, "EvidenceChunk", "U") or not _agent_can(agent_id, "QualityMetrics", "U"):
            return "Permission denied."
        try:
            t = float(threshold_str.strip() or str(self.DEFAULT_DUPLICATE_CHUNK_THRESHOLD))
        except ValueError:
            t = self.DEFAULT_DUPLICATE_CHUNK_THRESHOLD
        return self.detect_duplicate_chunks(threshold=t)

    def _tool_detect_duplicate_questions(self, agent_id: str, threshold_str: str = "") -> str:
        if not _agent_can(agent_id, "CandidateQuestion", "U") or not _agent_can(agent_id, "QualityMetrics", "U"):
            return "Permission denied."
        try:
            t = float(threshold_str.strip() or str(self.DEFAULT_DUPLICATE_QUESTION_THRESHOLD))
        except ValueError:
            t = self.DEFAULT_DUPLICATE_QUESTION_THRESHOLD
        return self.detect_duplicate_questions(threshold=t)

    def _tool_get_duplicate_report(self, agent_id: str, _tool_input: str = "") -> str:
        if not _agent_can(agent_id, "QualityMetrics", "R"):
            return "Permission denied."
        return self.get_duplicate_report()
    
    def _validate_question(self, q_id: str, validation_notes: str = "") -> str:
        """Validate a candidate question and move it to validated"""
        question = self.state.get_question(q_id)
        if not question:
            return f"Question {q_id} not found in candidate questions"
        
        validated = ValidatedQuestion(
            q_id=q_id,
            question_text=question.question_text,
            question_type=question.question_type,
            correct_answer=question.correct_answer or "N/A",
            difficulty=question.difficulty,
            options=question.options,
            topic_tags=question.topic_tags,
            related_atoms=question.related_atoms,
            evidence_chunks=question.evidence_chunks,
            validation_notes=validation_notes,
            flags=getattr(question, "flags", []) or [],
            duplicate_of=getattr(question, "duplicate_of", None),
        )
        self.state.validate_question(q_id, validated)
        return f"Question {q_id} validated and moved to validated_questions"
    
    def _reject_question(self, q_id: str, rejection_reason: str, fix_hints: str = "") -> str:
        """Reject a candidate question and move it to rejected"""
        question = self.state.get_question(q_id)
        if not question:
            return f"Question {q_id} not found in candidate questions"
        
        # Parse fix_hints if provided as string
        hints_list = fix_hints.split(',') if fix_hints else []
        hints_list = [h.strip() for h in hints_list if h.strip()]
        
        rejection = RejectedQuestion(
            q_id=q_id,
            question_text=question.question_text,
            rejection_reason=rejection_reason,
            fix_hints=hints_list,
            original_question={
                "question_type": question.question_type,
                "options": question.options,
                "correct_answer": question.correct_answer,
                "difficulty": question.difficulty
            }
        )
        self.state.reject_question(q_id, rejection)
        return f"Question {q_id} rejected: {rejection_reason}"

    @traceable(name="verify_candidate_question")
    def _verify_candidate_question(self, q_id: str) -> str:
        """Verify one candidate question against its evidence using VERIFICATION_RULES; move to validated or rejected."""
        self.log_step("verifier", "verify_question", {"q_id": q_id})
        question = self.state.get_question(q_id)
        if not question:
            return f"Question {q_id} not found in candidate questions."

        # Gather evidence text from linked chunks
        evidence_parts = []
        for cid in question.evidence_chunks:
            ch = self.state.get_chunk(cid)
            if ch:
                evidence_parts.append(f"[Chunk {cid} relevance={ch.relevance_score:.2f} quality={ch.quality_score:.2f}]:\n{ch.content}")
        if not evidence_parts:
            self._reject_question(q_id, "No evidence chunks linked to this question.", "Link evidence chunks to the question first.")
            return f"Question {q_id} rejected: no evidence chunks linked."

        evidence_text = "\n\n---\n\n".join(evidence_parts)[:4000]
        options_str = ""
        if question.options:
            options_str = "Options: " + " | ".join(question.options) + "\n"
        correct_str = f"Correct answer: {question.correct_answer or 'N/A'}\n" if question.correct_answer else ""

        rules_text = "\n".join(f"- {r}" for r in VERIFICATION_RULES)
        prompt = (
            "You are a fact-checker. Check the following quiz question against the evidence using these rules:\n\n"
            f"RULES:\n{rules_text}\n\n"
            "EVIDENCE (from source document):\n"
            f"{evidence_text}\n\n"
            "QUESTION TO VERIFY:\n"
            f"{question.question_text}\n"
            f"{options_str}{correct_str}\n"
            "Respond with exactly one of:\n"
            "VERDICT: PASS\n"
            "REASON: <short reason>\n"
            "--- or ---\n"
            "VERDICT: FAIL\n"
            "REASON: <short reason>\n"
            "FIX_HINTS: <comma-separated hints>\n"
        )
        try:
            response = self._get_llm_response(prompt)
        except Exception as e:
            return f"Verification failed for {q_id}: {e}"

        response_upper = response.upper()
        if "VERDICT: PASS" in response_upper:
            reason = "Verification passed against evidence."
            if "REASON:" in response:
                idx = response.upper().find("REASON:")
                reason = response[idx + 7 :].split("\n")[0].strip()[:200]
            self._validate_question(q_id, reason)
            return f"Question {q_id} validated. {reason}"
        else:
            reason = "Verification failed."
            hints = ""
            if "REASON:" in response:
                idx = response.upper().find("REASON:")
                reason = response[idx + 7 :].split("\n")[0].strip()[:200]
            if "FIX_HINTS:" in response.upper():
                idx = response.upper().find("FIX_HINTS:")
                hints = response[idx + 10 :].split("\n")[0].strip()[:200]
            self._reject_question(q_id, reason, hints)
            return f"Question {q_id} rejected. {reason}"

    def _verify_all_candidate_questions(self, limit: Optional[int] = None) -> str:
        """Verify candidate questions against evidence; move each to validated or rejected."""
        q_ids = list(self.state.candidate_questions.keys())
        if limit is not None and limit > 0:
            q_ids = q_ids[:limit]
        if not q_ids:
            return "No candidate questions to verify."
        validated_n = 0
        rejected_n = 0
        results = []
        for q_id in q_ids:
            out = self._verify_candidate_question(q_id)
            results.append(out)
            if "validated" in out.lower():
                validated_n += 1
            else:
                rejected_n += 1
        return (
            f"Verified {len(q_ids)} candidate(s): {validated_n} validated, {rejected_n} rejected.\n"
            + "\n".join(results)
        )

    # --- Structure/Outline: validation and placeholder blocking
    STRUCTURE_PLACEHOLDER_BLOCKLIST = frozenset({
        "section_id", "title", "start_slide", "end_slide", "weight",
        "importance_weight", "slide_range", "section", "placeholder", "n/a", "tbd", "tba", ""
    })
    SECTION_ID_PATTERN = re.compile(r"^sec_\d+$")
    WEIGHT_MIN, WEIGHT_MAX = 0.1, 1.0

    def _is_placeholder_value(self, value: str) -> bool:
        """True if value is a forbidden placeholder (case-insensitive, stripped)."""
        if not value or not isinstance(value, str):
            return True
        v = value.strip().lower()
        return v in self.STRUCTURE_PLACEHOLDER_BLOCKLIST

    def _is_valid_section_id(self, section_id: str) -> bool:
        """Section ID must be unique and match sec_N pattern (N integer)."""
        if not section_id or not isinstance(section_id, str):
            return False
        s = section_id.strip()
        if self._is_placeholder_value(s):
            return False
        return bool(self.SECTION_ID_PATTERN.match(s))

    def _validate_section_for_add(
        self,
        section_id: str,
        title: str,
        start_slide: int,
        end_slide: int,
        weight: float,
    ) -> Tuple[bool, str]:
        """
        Validate one section before adding. Returns (ok, error_message).
        Rules: no placeholders; unique section_id (sec_N); non-empty title; valid numbers; no overlap with existing.
        """
        if self._is_placeholder_value(section_id):
            return False, "section_id must be a real ID (e.g. sec_1, sec_2), not a placeholder like 'section_id'."
        if not self.SECTION_ID_PATTERN.match(section_id.strip()):
            return False, "section_id must match pattern sec_1, sec_2, sec_3, ... (sec_ then a number)."
        existing_ids = {s.section_id for s in self.state.doc_outline}
        if section_id.strip() in existing_ids:
            return False, f"section_id '{section_id}' already exists in the outline. Use a unique ID."
        if self._is_placeholder_value(title):
            return False, "title must be a real section title (e.g. 'Introduction'), not a placeholder like 'title'."
        if not (title and title.strip()):
            return False, "title cannot be empty."
        try:
            start, end = int(start_slide), int(end_slide)
        except (TypeError, ValueError):
            return False, "start_slide and end_slide must be integers."
        if start < 1 or end < 1:
            return False, "start_slide and end_slide must be >= 1."
        if start > end:
            return False, "start_slide must be <= end_slide."
        try:
            w = float(weight)
        except (TypeError, ValueError):
            return False, "weight must be a number."
        if not (self.WEIGHT_MIN <= w <= self.WEIGHT_MAX):
            return False, f"weight must be between {self.WEIGHT_MIN} and {self.WEIGHT_MAX}."
        # No overlap with existing sections
        for sec in self.state.doc_outline:
            a, b = sec.slide_range[0], sec.slide_range[1]
            if not (end < a or start > b):
                return False, f"Slide range [{start},{end}] overlaps with existing section '{sec.section_id}' ({a}-{b})."
        return True, ""

    def _validate_outline(self, outline: List["DocSection"]) -> Tuple[bool, str]:
        """
        Validate full outline: unique section IDs, non-empty titles, valid weights, no overlapping ranges.
        Returns (ok, error_message).
        """
        if not outline:
            return True, ""
        seen_ids: Set[str] = set()
        for sec in outline:
            if not sec.section_id or sec.section_id.strip() in seen_ids:
                return False, f"Duplicate or empty section_id: {sec.section_id!r}."
            seen_ids.add(sec.section_id.strip())
            if self._is_placeholder_value(sec.section_id):
                return False, f"section_id cannot be a placeholder: {sec.section_id!r}."
            if not self.SECTION_ID_PATTERN.match(sec.section_id.strip()):
                return False, f"section_id must match sec_N: {sec.section_id!r}."
            if self._is_placeholder_value(sec.title) or not (sec.title and sec.title.strip()):
                return False, f"Section {sec.section_id}: title cannot be empty or a placeholder."
            w = sec.importance_weight
            if not (self.WEIGHT_MIN <= w <= self.WEIGHT_MAX):
                return False, f"Section {sec.section_id}: weight must be in [{self.WEIGHT_MIN}, {self.WEIGHT_MAX}]."
            a, b = sec.slide_range[0], sec.slide_range[1]
            if a > b or a < 1:
                return False, f"Section {sec.section_id}: invalid slide_range {sec.slide_range}."
        # Check overlaps
        ranges = [(sec.slide_range[0], sec.slide_range[1]) for sec in outline]
        ranges.sort()
        for i in range(1, len(ranges)):
            if ranges[i][0] <= ranges[i - 1][1]:
                return False, f"Overlapping slide ranges: {ranges[i - 1]} and {ranges[i]}."
        return True, ""

    # --- Privilege-checking tool helpers (agent_id is set when building per-agent tools)
    def _tool_get_outline(self, agent_id: str, _tool_input: str = "") -> str:
        if not _agent_can(agent_id, "DocSection", "R"):
            return "Permission denied: no Read on DocSection."
        return self.state.get_outline_summary()

    def _tool_build_outline(self, agent_id: str, section_id: str, title: str, start_slide: int, end_slide: int, weight: float) -> str:
        if not _agent_can(agent_id, "DocSection", "C") or not _agent_can(agent_id, "CooperativeState", "U"):
            return "Permission denied."
        ok, err = self._validate_section_for_add(section_id, title, start_slide, end_slide, weight)
        if not ok:
            return f"Validation failed: {err}"
        section = DocSection(section_id=section_id.strip(), title=title.strip(), slide_range=(start_slide, end_slide), importance_weight=round(float(weight), 2))
        self.state.add_section(section)
        return f"Added section {section_id}: {title}"

    def _tool_build_outline_section(self, agent_id: str, input_str: str) -> str:
        """
        Parse JSON or pipe-delimited input for a single section; validate (no placeholders, unique ID, one section); add via _tool_build_outline.
        """
        if not _agent_can(agent_id, "DocSection", "C") or not _agent_can(agent_id, "CooperativeState", "U"):
            return "Permission denied."
        raw = (input_str or "").strip()
        if not raw:
            return "Input required: JSON {\"section_id\":\"sec_1\",\"title\":\"...\",\"start_slide\":1,\"end_slide\":5,\"importance_weight\":0.5} or pipe: sec_1|Title|1|5|0.5"
        section_id = title = None
        start_slide = end_slide = 1
        weight = 0.5
        # Try JSON first
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
                if isinstance(obj, list):
                    return "Add one section per call. Pass a single JSON object, not an array."
                section_id = (obj.get("section_id") or "").strip()
                title = (obj.get("title") or "").strip()
                start_slide = int(obj.get("start_slide", 1))
                end_slide = int(obj.get("end_slide", 1))
                weight = float(obj.get("importance_weight", obj.get("weight", 0.5)))
            except json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"
            except (TypeError, ValueError) as e:
                return f"Invalid numbers in JSON: {e}"
        else:
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) < 5:
                return "Pipe format needs 5 fields: section_id|title|start_slide|end_slide|weight (e.g. sec_1|Introduction|1|5|0.5)"
            section_id, title = parts[0], parts[1]
            try:
                start_slide, end_slide, weight = int(parts[2]), int(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                return "start_slide, end_slide, and weight must be numbers."
        if not section_id or not title:
            return "section_id and title are required."
        return self._tool_build_outline(agent_id, section_id, title, start_slide, end_slide, weight)

    def _tool_set_full_outline(self, agent_id: str, outline_json: str) -> str:
        if not _agent_can(agent_id, "DocSection", "U") or not _agent_can(agent_id, "CooperativeState", "U"):
            return "Permission denied."
        try:
            data = json.loads(outline_json)
            if not isinstance(data, list):
                return "Outline must be a JSON array of section objects."
            outline = []
            for s in data:
                outline.append(DocSection(
                    section_id=str(s.get("section_id", "")).strip(),
                    title=str(s.get("title", "")).strip(),
                    slide_range=(int(s["slide_range"][0]), int(s["slide_range"][1])),
                    importance_weight=round(float(s.get("importance_weight", 0.5)), 2)
                ))
            ok, err = self._validate_outline(outline)
            if not ok:
                return f"Outline validation failed: {err}. Fix and retry."
            self.state.set_doc_outline(outline)
            return f"Set outline with {len(outline)} sections (validated)."
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return f"Invalid outline JSON: {e}"

    # --- Structure agent: build outline from document (retrieve → detect sections → ranges → weights → validate → store)
    OUTLINE_EXTRACTION_PROMPT = (
        "From the following document snippets, extract the document outline.\n"
        "Each snippet is the BEGINNING of a page/slide (where titles and headings usually appear).\n"
        "Infer section titles from those leading lines and assign contiguous page ranges.\n"
        "Use the actual title text from the page lead when possible (not generic names like 'Section 1').\n"
        "Return only a JSON array of objects: "
        "{\"section_id\": \"sec_1\", \"title\": \"...\", \"start_page\": N, \"end_page\": M}. "
        "Use section_id sec_1, sec_2, ... in document order. No other text."
    )
    OUTLINE_PAGE_LEAD_CHARS = 400
    OUTLINE_MAX_PAGES_SAMPLE = 80

    @staticmethod
    def _page_lead_text(text: str, max_chars: int = 400) -> str:
        """Keep the top of a page/slide — titles and headings usually appear first."""
        text = (text or "").strip()
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            lead = "\n".join(lines[:4])
            if len(lead) <= max_chars:
                return lead
            return lead[:max_chars].rsplit(" ", 1)[0] if len(lead) > max_chars else lead
        return text[:max_chars]

    def _collect_page_leads_from_index(
        self,
        max_pages: int = 80,
        lead_chars: int = 400,
    ) -> Dict[int, str]:
        """
        One lead snippet per page from the vector index — prefer chunk_index=0 (start of page).
        """
        by_page: Dict[int, Tuple[int, str]] = {}
        try:
            collection = self.rag_pipeline.vector_store._collection
            batch = collection.get(include=["documents", "metadatas"])
            docs = batch.get("documents") or []
            metas = batch.get("metadatas") or []
        except Exception:
            return {}

        for doc_text, meta in zip(docs, metas):
            if not meta:
                continue
            try:
                page = int(meta.get("page_number", 0))
            except (TypeError, ValueError):
                continue
            if page < 1:
                continue
            try:
                chunk_idx = int(meta.get("chunk_index", 0))
            except (TypeError, ValueError):
                chunk_idx = 0
            lead = self._page_lead_text(doc_text, lead_chars)
            if not lead:
                continue
            meta_title = (meta.get("title") or "").strip()
            if meta_title and meta_title.lower() not in lead.lower()[:80]:
                lead = f"{meta_title}\n{lead}"
            prev = by_page.get(page)
            if prev is None or chunk_idx < prev[0]:
                by_page[page] = (chunk_idx, lead)

        if not by_page:
            return {}

        pages_sorted = sorted(by_page.keys())
        if len(pages_sorted) > max_pages:
            step = len(pages_sorted) / max_pages
            pages_sorted = [pages_sorted[int(i * step)] for i in range(max_pages)]

        return {p: by_page[p][1] for p in pages_sorted}

    def build_outline_from_document(
        self,
        toc_queries: Optional[List[str]] = None,
        max_chunks: int = 50,
        default_weight: float = 0.5,
        weight_per_pages: float = 0.05,
        pages_for_weight: int = 3,
    ) -> str:
        """
        Build doc outline from document: sample page-leading text, supplement with TOC search,
        extract sections via LLM, validate ranges, assign weights, store via set_doc_outline.
        """
        lead_chars = self.OUTLINE_PAGE_LEAD_CHARS
        page_leads = self._collect_page_leads_from_index(
            max_pages=self.OUTLINE_MAX_PAGES_SAMPLE,
            lead_chars=lead_chars,
        )

        if toc_queries is None:
            toc_queries = [
                "table of contents agenda outline",
                "chapter section module heading",
                "slide title introduction overview",
            ]
        seen_pages = set(page_leads.keys())
        for query in toc_queries[:5]:
            try:
                results = self.rag_pipeline.search(query, k=min(20, max_chunks))
            except Exception:
                continue
            for doc, _ in results:
                page = doc.metadata.get("page_number")
                try:
                    page = int(page)
                except (TypeError, ValueError):
                    page = 1
                if page in seen_pages:
                    continue
                lead = self._page_lead_text(doc.page_content, lead_chars)
                if lead:
                    page_leads[page] = lead
                    seen_pages.add(page)

        if not page_leads:
            return "No document chunks retrieved; cannot build outline. Run retrieval first or check vector store."

        sorted_pages = sorted(page_leads.keys())
        lines = []
        for p in sorted_pages:
            lines.append(f"Page {p} (leading lines):\n{page_leads[p]}")
        combined = "\n\n".join(lines)[:12000]
        prompt = f"{self.OUTLINE_EXTRACTION_PROMPT}\n\nDocument chunks:\n{combined}"
        try:
            response = self._get_llm_response(prompt, llm_component="outline").strip()
        except Exception as e:
            return f"Outline extraction failed: {e}"
        if "```" in response:
            m = re.search(r"\[[\s\S]*?\]", response)
            if m:
                response = m.group(0)
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            start, end = response.find("["), response.rfind("]")
            if start != -1 and end > start:
                try:
                    data = json.loads(response[start : end + 1])
                except json.JSONDecodeError:
                    return "Could not parse outline JSON from model."
            else:
                return "Could not parse outline JSON from model."
        if not isinstance(data, list) or not data:
            return "No sections extracted."
        sections = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            sid = (item.get("section_id") or "").strip()
            if not sid or not self.SECTION_ID_PATTERN.match(sid):
                sid = f"sec_{i+1}"
            title = (item.get("title") or "").strip() or f"Section {i+1}"
            try:
                start = int(item.get("start_page", 1))
                end = int(item.get("end_page", start))
            except (TypeError, ValueError):
                start = end = 1
            if end < start:
                end = start
            sections.append({"section_id": sid, "title": title, "start": start, "end": end})
        sections.sort(key=lambda x: (x["start"], x["end"]))
        # Fix overlaps and gaps: contiguous non-overlapping ranges
        fixed = []
        for i, s in enumerate(sections):
            start, end = s["start"], s["end"]
            if fixed and start <= fixed[-1]["end"]:
                start = fixed[-1]["end"] + 1
            end = max(end, start)
            fixed.append({"section_id": s["section_id"], "title": s["title"], "start": start, "end": end})
        # Fill small gaps with a placeholder section if needed (optional; here we leave gaps as-is and just assign weights)
        outline = []
        for s in fixed:
            num_pages = max(1, s["end"] - s["start"] + 1)
            w = default_weight + (num_pages // pages_for_weight) * weight_per_pages
            w = min(0.9, max(0.1, w))
            outline.append(DocSection(
                section_id=s["section_id"],
                title=s["title"],
                slide_range=(s["start"], s["end"]),
                importance_weight=round(w, 2),
            ))
        ok, err = self._validate_outline(outline)
        if not ok:
            return f"Outline validation failed: {err}. No outline stored."
        self.state.set_doc_outline(outline)
        return f"Built outline with {len(outline)} sections (validated, no overlaps)."
    
    def _tool_build_outline_from_document(self, agent_id: str, _dummy: str = "") -> str:
        if not _agent_can(agent_id, "DocSection", "U") or not _agent_can(agent_id, "CooperativeState", "U"):
            return "Permission denied."
        return self.build_outline_from_document()

    def _tool_get_learning_objectives(self, agent_id: str, _tool_input: str = "") -> str:
        if not _agent_can(agent_id, "LearningObjective", "R"):
            return "Permission denied: no Read on LearningObjective."
        return self.state.get_learning_objectives_summary()

    def _tool_add_learning_objective(self, agent_id: str, lo_id: str, section_id: str, objective_text: str, difficulty_level: str) -> str:
        if not _agent_can(agent_id, "LearningObjective", "C") or not _agent_can(agent_id, "CooperativeState", "U"):
            return "Permission denied."
        lo = LearningObjective(lo_id=lo_id, section_id=section_id, objective_text=objective_text, difficulty_level=difficulty_level)
        self.state.add_learning_objective(lo)
        return f"Added LO {lo_id} for section {section_id}."

    # --- Curriculum/LO agent: generate LOs from outline (sections + retrieved chunks + optional atoms)
    ALLOWED_LO_DIFFICULTIES = ("beginner", "intermediate", "advanced")
    LO_GENERATION_PROMPT = (
        "Generate learning objectives for the following section. "
        "Output only a JSON array of objects: {\"objective_text\": \"...\", \"difficulty_level\": \"beginner\"|\"intermediate\"|\"advanced\"}. "
        "Use clear, actionable phrasing (verb + outcome). Balance difficulties across the list. "
        "No other text."
    )

    def _normalize_lo_text(self, text: str) -> str:
        """Canonical form for dedupe: lowercase, strip, collapse whitespace."""
        if not text or not isinstance(text, str):
            return ""
        return " ".join(text.lower().strip().split())

    def generate_learning_objectives_for_outline(
        self,
        los_per_section: int = 3,
        balance_difficulty: bool = True,
        max_chunks_per_section: int = 8,
        use_existing_atoms: bool = True,
    ) -> str:
        """
        Deterministic pipeline: for each section, gather title + page range + representative chunks
        (and optional atoms), prompt LLM for structured LOs, validate, dedupe, balance difficulty, store.
        """
        if not self.state.doc_outline:
            return "No document outline. Run structure agent to build outline first."
        existing_lo_ids = {lo.lo_id for lo in self.state.learning_objectives}
        existing_normalized = {self._normalize_lo_text(lo.objective_text) for lo in self.state.learning_objectives}
        added = 0
        for section in self.state.doc_outline:
            title = section.title or section.section_id
            start_p, end_p = section.slide_range if isinstance(section.slide_range, (list, tuple)) else (1, 1)
            try:
                start_p = int(start_p)
                end_p = int(end_p)
            except (TypeError, ValueError):
                start_p, end_p = 1, 1
            if end_p < start_p:
                end_p = start_p
            page_range = set(range(start_p, end_p + 1))
            # Representative chunks: search by section title, filter by page in range
            try:
                results = self.rag_pipeline.search(title, k=max(15, max_chunks_per_section * 2))
            except Exception:
                results = []
            chunks_in_range = []
            for doc, score in (results if results else []):
                p = doc.metadata.get("page_number")
                try:
                    p = int(p)
                except (TypeError, ValueError):
                    p = 1
                if p in page_range:
                    chunks_in_range.append((doc.page_content[:600], p))
                    if len(chunks_in_range) >= max_chunks_per_section:
                        break
            if not chunks_in_range:
                for doc, score in (results if results else []):
                    chunks_in_range.append((doc.page_content[:600], doc.metadata.get("page_number", 1)))
                    if len(chunks_in_range) >= max_chunks_per_section:
                        break
            section_context = f"Section: {title} (pages {start_p}-{end_p})\n\n"
            section_context += "Representative content:\n" + "\n\n".join(
                f"[Page {p}]: {c}" for c, p in chunks_in_range[:max_chunks_per_section]
            )
            if use_existing_atoms and self.state.knowledge_atoms:
                atoms_for_section = [
                    a.content[:200] for a in self.state.knowledge_atoms.values()
                    if getattr(a, "section_id", None) == section.section_id
                ]
                if atoms_for_section:
                    section_context += "\n\nExisting knowledge atoms (use to align LOs):\n" + "\n".join(f"- {a}" for a in atoms_for_section[:15])
            prompt = f"{self.LO_GENERATION_PROMPT}\n\n{section_context}\n\nGenerate exactly {los_per_section} learning objectives."
            try:
                response = self._get_llm_response(
                    prompt, llm_component="learning_objectives"
                ).strip()
            except Exception as e:
                continue
            if "```" in response:
                m = re.search(r"\[[\s\S]*?\]", response)
                if m:
                    response = m.group(0)
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                start, end = response.find("["), response.rfind("]")
                if start != -1 and end > start:
                    try:
                        data = json.loads(response[start : end + 1])
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            if not isinstance(data, list):
                continue
            difficulties = list(self.ALLOWED_LO_DIFFICULTIES) if balance_difficulty else [self.ALLOWED_LO_DIFFICULTIES[1]]
            idx = 0
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                obj_text = (item.get("objective_text") or "").strip()
                if not obj_text:
                    continue
                norm = self._normalize_lo_text(obj_text)
                if norm in existing_normalized:
                    continue
                diff = (item.get("difficulty_level") or "").strip().lower()
                if diff not in self.ALLOWED_LO_DIFFICULTIES:
                    diff = difficulties[idx % len(difficulties)]
                if balance_difficulty:
                    diff = difficulties[idx % len(difficulties)]
                    idx += 1
                lo_id = f"{section.section_id}_lo_{len(existing_lo_ids) + 1}"
                if lo_id in existing_lo_ids:
                    lo_id = self.state.generate_id("lo")
                existing_lo_ids.add(lo_id)
                existing_normalized.add(norm)
                lo = LearningObjective(lo_id=lo_id, section_id=section.section_id, objective_text=obj_text, difficulty_level=diff)
                self.state.add_learning_objective(lo)
                added += 1
        if added == 0:
            return "No new learning objectives added (outline empty, parse failed, or all duplicates)."
        return f"Generated and stored {added} learning objectives across {len(self.state.doc_outline)} sections (validated, deduped, difficulty balanced)."

    def _tool_generate_learning_objectives_for_outline(self, agent_id: str, _dummy: str = "") -> str:
        if not _agent_can(agent_id, "LearningObjective", "C") or not _agent_can(agent_id, "CooperativeState", "U"):
            return "Permission denied."
        return self.generate_learning_objectives_for_outline()

    def _tool_retrieve_chunks(self, agent_id: str, query: str, k: int = 5) -> str:
        if not _agent_can(agent_id, "EvidenceChunk", "C"):
            return "Permission denied: no Create on EvidenceChunk."
        return self._retrieve_chunks(query, k=k)

    def _tool_get_chunk(self, agent_id: str, chunk_id: str) -> str:
        if not _agent_can(agent_id, "EvidenceChunk", "R"):
            return "Permission denied."
        return self._get_chunk(chunk_id)

    def _tool_update_chunk(
        self,
        agent_id: str,
        chunk_id: str,
        relevance_score: Optional[float] = None,
        quality_score: Optional[float] = None,
        relevance_tags: Optional[str] = None,
    ) -> str:
        if not _agent_can(agent_id, "EvidenceChunk", "U"):
            return "Permission denied."
        tags = [t.strip() for t in relevance_tags.split(",")] if relevance_tags else None
        ok = self.state.update_evidence_chunk(
            chunk_id,
            relevance_score=relevance_score,
            quality_score=quality_score,
            relevance_tags=tags,
        )
        return f"Updated chunk {chunk_id}." if ok else f"Chunk {chunk_id} not found."

    def _evaluate_chunk_quality(self, chunk_id: str) -> str:
        """Evaluate one chunk's quality (noise, duplication, source trust) via LLM and set quality_score."""
        chunk = self.state.get_chunk(chunk_id)
        if not chunk:
            return f"Chunk {chunk_id} not found."
        # Optional: include snippets of other chunks for duplication check (limit size)
        other_snippets = []
        for cid, c in list(self.state.evidence_pool.items())[:15]:
            if cid == chunk_id:
                continue
            other_snippets.append(f"[{cid}]: {c.content[:150]}...")
        context = ""
        if other_snippets:
            context = "\nOther chunks (for duplication check):\n" + "\n".join(other_snippets[:5])
        criteria_text = "\n".join(f"- {c}" for c in CHUNK_QUALITY_CRITERIA)
        prompt = (
            "Rate the quality of this evidence chunk from 0.0 (poor) to 1.0 (excellent) using:\n\n"
            f"{criteria_text}\n\n"
            f"Chunk to rate:\nSource: {chunk.metadata.get('source_file', '?')} p.{chunk.metadata.get('page_number', '?')}\n"
            f"{chunk.content[:1500]}\n"
            f"{context}\n\n"
            "Respond with only a single number between 0.0 and 1.0 (e.g. 0.85)."
        )
        try:
            response = self._get_llm_response(
                prompt, llm_component="evidence_quality"
            ).strip()
            # Parse first float in response
            m = re.search(r"0?\.\d+|\d+\.\d*", response)
            if m:
                score = float(m.group(0))
                score = max(0.0, min(1.0, score))
                self.state.update_evidence_chunk(chunk_id, quality_score=score)
                return f"Chunk {chunk_id} quality_score set to {score:.2f}."
        except Exception as e:
            return f"Evaluation failed for {chunk_id}: {e}"
        return f"Could not parse quality score from response for {chunk_id}."

    def _evaluate_all_chunks_quality(self) -> str:
        """Evaluate quality (noise, duplication, source trust) for all chunks and set quality_score."""
        chunk_ids = list(self.state.evidence_pool.keys())
        if not chunk_ids:
            return "No chunks in evidence pool."
        results = []
        for cid in chunk_ids:
            results.append(self._evaluate_chunk_quality(cid))
        return "\n".join(results)

    def _tool_evaluate_chunk_quality(self, agent_id: str, chunk_id: str) -> str:
        if not _agent_can(agent_id, "EvidenceChunk", "U"):
            return "Permission denied."
        return self._evaluate_chunk_quality(chunk_id.strip())

    def _tool_evaluate_all_chunks_quality(self, agent_id: str, _dummy: str = "") -> str:
        if not _agent_can(agent_id, "EvidenceChunk", "U"):
            return "Permission denied."
        return self._evaluate_all_chunks_quality()

    # --- Evidence Curator: dedupe, quality re-score, tagging, link chunk→question
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using pipeline embeddings. Returns list of vectors."""
        if not texts:
            return []
        try:
            emb = getattr(self.rag_pipeline, "embeddings", None)
            if emb is None:
                return []
            return emb.embed_documents(texts)
        except Exception:
            return []

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _cluster_duplicates(
        self, chunk_ids: List[str], threshold: float = CURATE_DEDUPE_THRESHOLD
    ) -> List[List[str]]:
        """Cluster chunk_ids by embedding similarity (union-find). Returns list of clusters (each cluster = list of ids)."""
        if len(chunk_ids) <= 1:
            return [[c] for c in chunk_ids] if chunk_ids else []
        chunks = [self.state.get_chunk(cid) for cid in chunk_ids]
        texts = [c.content[:2000] if c else "" for c in chunks]
        vectors = self._embed_texts(texts)
        if len(vectors) != len(chunk_ids):
            return [[c] for c in chunk_ids]
        # Union-find
        parent = {cid: cid for cid in chunk_ids}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(chunk_ids)):
            for j in range(i + 1, len(chunk_ids)):
                if self._cosine_sim(vectors[i], vectors[j]) >= threshold:
                    union(chunk_ids[i], chunk_ids[j])
        # Group by root
        clusters: Dict[str, List[str]] = defaultdict(list)
        for cid in chunk_ids:
            clusters[find(cid)].append(cid)
        return list(clusters.values())

    def _pick_canonical(self, cluster: List[str]) -> str:
        """Pick canonical chunk in cluster: highest quality_score, then earliest page."""
        best_id = cluster[0]
        best_score = -1.0
        best_page = float("inf")
        for cid in cluster:
            c = self.state.get_chunk(cid)
            if not c:
                continue
            page = c.metadata.get("page_number") or 0
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = 9999
            if c.quality_score > best_score or (c.quality_score == best_score and page < best_page):
                best_score = c.quality_score
                best_page = page
                best_id = cid
        return best_id

    def _curate_dedupe(self, chunk_ids: Optional[List[str]] = None, threshold: float = CURATE_DEDUPE_THRESHOLD) -> str:
        """Deduplicate chunks: cluster by similarity, mark non-canonical as duplicate_of, merge tags into canonical."""
        if chunk_ids is None:
            chunk_ids = [cid for cid, c in self.state.evidence_pool.items() if getattr(c, "duplicate_of", None) is None]
        if len(chunk_ids) <= 1:
            return "No duplicates to merge."
        clusters = self._cluster_duplicates(chunk_ids, threshold)
        merged = 0
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            canonical = self._pick_canonical(cluster)
            all_tags = set()
            for cid in cluster:
                c = self.state.get_chunk(cid)
                if c:
                    all_tags.update(c.relevance_tags or [])
            self.state.update_evidence_chunk(canonical, relevance_tags=sorted(all_tags))
            for cid in cluster:
                if cid != canonical:
                    self.state.update_evidence_chunk(cid, duplicate_of=canonical)
                    merged += 1
        return f"Dedupe: {merged} chunks marked as duplicates of canonicals."

    def _curate_tags_for_chunks(self, chunk_ids: List[str]) -> str:
        """Generate topic tags via LLM (structured JSON), normalize (lowercase, dedupe), update relevance_tags."""
        if not chunk_ids:
            return "No chunks."
        results = []
        for cid in chunk_ids[:20]:  # limit batch
            c = self.state.get_chunk(cid)
            if not c:
                continue
            prompt = (
                "From this text, extract 3–8 short topic/concept tags (e.g. k-means, initialization, convergence). "
                "Return only a JSON array of strings, e.g. [\"tag1\", \"tag2\"]. No other text.\n\n"
                f"{c.content[:800]}"
            )
            try:
                resp = self._get_llm_response(
                    prompt, llm_component="evidence_tags"
                ).strip()
                # Strip markdown code block if present
                if "```" in resp:
                    m = re.search(r"\[[\s\S]*?\]", resp)
                    if m:
                        resp = m.group(0)
                arr = json.loads(resp)
                if isinstance(arr, list):
                    tags = [str(t).strip().lower() for t in arr if str(t).strip()]
                    tags = list(dict.fromkeys(tags))[:15]  # dedupe, cap
                    self.state.update_evidence_chunk(cid, relevance_tags=tags)
                    results.append(f"{cid}: {tags}")
            except Exception as e:
                results.append(f"{cid}: error {e}")
        return "\n".join(results) if results else "No tags updated."

    @traceable(name="curate_evidence")
    def curate_evidence(
        self,
        section_id: Optional[str] = None,
        lo_id: Optional[str] = None,
        query: Optional[str] = None,
        dedupe_threshold: float = CURATE_DEDUPE_THRESHOLD,
        re_score_quality: bool = True,
        generate_tags: bool = True,
    ) -> str:
        """
        Curate evidence pool: dedupe_chunks(state.evidence_pool), score_chunks (quality), tag_chunks.
        By default operates on the full state.evidence_pool; no parameters required.
        Optional filters section_id, lo_id, query restrict which chunks are curated (for advanced use).
        """
        # Resolve chunk set: all non-duplicate in pool (ignore None/empty/placeholder filters)
        all_ids = [
            cid for cid, c in self.state.evidence_pool.items()
            if getattr(c, "duplicate_of", None) is None
        ]
        # Optional filters only when real values are provided (not placeholder strings)
        if section_id and section_id.strip().lower() not in ("", "all", "section_id"):
            all_ids = [cid for cid in all_ids if self.state.get_chunk(cid) and self.state.get_chunk(cid).metadata.get("section_id") == section_id]
        if lo_id and lo_id.strip().lower() not in ("", "lo_id"):
            all_ids = [cid for cid in all_ids if self.state.get_chunk(cid) and self.state.get_chunk(cid).metadata.get("lo_id") == lo_id]
        if query and query.strip().lower() not in ("", "all", "query"):
            all_ids = [cid for cid in all_ids if self.state.get_chunk(cid) and query.strip().lower() in (self.state.get_chunk(cid).relevance_tags or [])]
        if not all_ids:
            return "No chunks to curate (evidence_pool empty or filter matched nothing)."
        out = []
        # 1) Dedupe
        out.append(self._curate_dedupe(all_ids, dedupe_threshold))
        canonical_ids = [cid for cid in all_ids if self.state.get_chunk(cid) and getattr(self.state.get_chunk(cid), "duplicate_of", None) is None]
        # 2) Re-score quality for canonicals
        if re_score_quality and canonical_ids:
            for cid in canonical_ids[:15]:
                out.append(self._evaluate_chunk_quality(cid))
        # 3) Generate tags
        if generate_tags and canonical_ids:
            out.append("Tags: " + self._curate_tags_for_chunks(canonical_ids[:15]))
        return "\n".join(out)

    def _tool_link_chunk_to_question(self, agent_id: str, q_id: str, chunk_id: str, validated: bool = False) -> str:
        if not _agent_can(agent_id, "CandidateQuestion", "L") and not (validated and _agent_can(agent_id, "ValidatedQuestion", "L")):
            return "Permission denied."
        ok = self.state.link_chunk_to_question(q_id.strip(), chunk_id.strip(), validated=validated)
        return f"Linked chunk {chunk_id} to question {q_id}." if ok else "Chunk or question not found."

    def _tool_mark_duplicate(self, agent_id: str, chunk_id: str, canonical_chunk_id: str) -> str:
        if not _agent_can(agent_id, "EvidenceChunk", "U"):
            return "Permission denied."
        if chunk_id not in self.state.evidence_pool or canonical_chunk_id not in self.state.evidence_pool:
            return "Chunk or canonical not found."
        self.state.update_evidence_chunk(chunk_id.strip(), duplicate_of=canonical_chunk_id.strip())
        return f"Marked {chunk_id} as duplicate of {canonical_chunk_id}."

    def _tool_curate_evidence(self, agent_id: str, _input: str = "") -> str:
        """Curate all evidence in state.evidence_pool: dedupe, score quality, generate tags. No parameters required."""
        if not _agent_can(agent_id, "EvidenceChunk", "U"):
            return "Permission denied."
        # Always operate on full evidence_pool; ignore input (avoids placeholder section_id|lo_id|query)
        return self.curate_evidence()

    def _tool_link_chunk_to_atom(self, agent_id: str, chunk_id: str, atom_id: str) -> str:
        if not _agent_can(agent_id, "EvidenceChunk", "L") or not _agent_can(agent_id, "KnowledgeAtom", "L"):
            return "Permission denied."
        ok = self.state.link_chunk_to_atom(chunk_id, atom_id)
        return f"Linked chunk {chunk_id} to atom {atom_id}." if ok else "Chunk or atom not found."

    def _tool_add_knowledge_atom(self, agent_id: str, content: str, atom_type: str = "fact", chunk_ids: Optional[str] = None, section_id: Optional[str] = None, lo_id: Optional[str] = None) -> str:
        if not _agent_can(agent_id, "KnowledgeAtom", "C"):
            return "Permission denied."
        cids = [x.strip() for x in (chunk_ids or "").split(",") if x.strip()]
        atom_id = self.state.generate_id("atom")
        citations = []
        for chunk_id in cids:
            ch = self.state.get_chunk(chunk_id)
            if ch:
                citations.append({"page": ch.metadata.get("page_number", "?"), "source": ch.metadata.get("source_file", "?"), "chunk_id": chunk_id})
        atom = KnowledgeAtom(atom_id=atom_id, content=content, atom_type=atom_type, citations=citations, section_id=section_id or None, lo_id=lo_id or None)
        self.state.add_knowledge_atom(atom)
        for cid in cids:
            self.state.link_chunk_to_atom(cid, atom_id)
        return f"Added atom {atom_id}: {content[:80]}..."

    def _tool_get_atom(self, agent_id: str, atom_id: str) -> str:
        if not _agent_can(agent_id, "KnowledgeAtom", "R"):
            return "Permission denied."
        return self._get_atom(atom_id)

    def _tool_bind_atom_to_lo(self, agent_id: str, atom_id: str, lo_id: str) -> str:
        if not _agent_can(agent_id, "KnowledgeAtom", "L"):
            return "Permission denied."
        ok = self.state.update_atom_lo(atom_id, lo_id)
        return f"Bound atom {atom_id} to LO {lo_id}." if ok else "Atom not found."

    # --- Atomizer: extract atoms from chunks (LLM, validate, dedupe, citations, section/LO)
    def _normalize_atom_content(self, content: str) -> str:
        """Normalize for dedupe: lowercase, strip, collapse whitespace."""
        if not content:
            return ""
        return " ".join(str(content).strip().lower().split())

    def _atom_exists_similar(self, content: str) -> bool:
        """True if an atom with same or very similar normalized content already exists."""
        key = self._normalize_atom_content(content)
        if not key or len(key) < ATOM_MIN_LENGTH:
            return False
        for atom in list(self.state.knowledge_atoms.values()):
            if self._normalize_atom_content(atom.content) == key:
                return True
            # Optional: high overlap (e.g. one contains the other)
            an = self._normalize_atom_content(atom.content)
            if key in an or an in key:
                return True
        return False

    def _validate_atom_candidate(self, item: Dict[str, Any]) -> bool:
        """Check length, single-fact heuristic, type."""
        if not isinstance(item, dict):
            return False
        content = (item.get("content") or "").strip()
        atype = (item.get("type") or "fact").strip().lower()
        if atype not in ("fact", "definition", "procedure"):
            atype = "fact"
        if len(content) < ATOM_MIN_LENGTH or len(content) > ATOM_MAX_LENGTH:
            return False
        # Reject if looks like multiple sentences (more than 2)
        if content.count(".") > 2:
            return False
        return True

    def _extract_atoms_from_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Call LLM to extract 3–7 atoms from chunk text; return list of {type, content, confidence}."""
        chunk = self.state.get_chunk(chunk_id)
        if not chunk:
            return []
        text = (chunk.content or "")[:2500]
        if not text.strip():
            return []
        rules = "\n".join(f"- {r}" for r in ATOM_EXTRACTION_RULES)
        prompt = (
            "Extract 3–7 knowledge atoms from the following text. "
            "Each atom must be a single fact, definition, or procedure step.\n\n"
            f"Rules:\n{rules}\n\n"
            "Return only a JSON array, no other text. Each element: {\"type\": \"fact\"|\"definition\"|\"procedure\", \"content\": \"...\", \"confidence\": 0.0-1.0}.\n\n"
            f"Text:\n{text}"
        )
        try:
            response = self._get_llm_response(
                prompt, llm_component="atomization"
            ).strip()
        except Exception:
            return []
        # Strip markdown and parse
        if "```" in response:
            m = re.search(r"\[[\s\S]*?\]", response)
            if m:
                response = m.group(0)
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            start, end = response.find("["), response.rfind("]")
            if start != -1 and end > start:
                try:
                    data = json.loads(response[start : end + 1])
                except json.JSONDecodeError:
                    return []
            else:
                return []
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if not self._validate_atom_candidate(item):
                continue
            content = (item.get("content") or "").strip()
            atype = (item.get("type") or "fact").strip().lower()
            if atype not in ("fact", "definition", "procedure"):
                atype = "fact"
            out.append({"type": atype, "content": content, "confidence": float(item.get("confidence", 0.8))})
        return out

    def _infer_section_lo_for_chunk(self, chunk_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Infer section_id and (optionally) first lo_id for chunk from outline/LO by page or metadata."""
        chunk = self.state.get_chunk(chunk_id)
        if not chunk or not self.state.doc_outline:
            return None, None
        page = chunk.metadata.get("page_number")
        try:
            page = int(page)
        except (TypeError, ValueError):
            return None, None
        section_id = None
        for sec in self.state.doc_outline:
            start, end = sec.slide_range[0], sec.slide_range[1]
            if start <= page <= end:
                section_id = sec.section_id
                break
        lo_id = None
        if section_id and self.state.learning_objectives:
            for lo in self.state.learning_objectives:
                if lo.section_id == section_id:
                    lo_id = lo.lo_id
                    break
        return section_id, lo_id

    @traceable(name="atomize_chunks")
    def atomize_chunks(
        self,
        chunk_ids: List[str],
        section_id: Optional[str] = None,
        lo_id: Optional[str] = None,
    ) -> str:
        """Extract atoms from chunks via LLM; validate, dedupe, create with citations, link chunk→atom, set section/LO."""
        created = []
        skipped_dup = 0
        skipped_invalid = 0
        for chunk_id in chunk_ids:
            chunk = self.state.get_chunk(chunk_id)
            if not chunk:
                continue
            items = self._extract_atoms_from_chunk(chunk_id)
            sec_id, l_id = section_id, lo_id
            if sec_id is None and l_id is None:
                sec_id, l_id = self._infer_section_lo_for_chunk(chunk_id)
            for item in items:
                content = item["content"]
                if self._atom_exists_similar(content):
                    skipped_dup += 1
                    continue
                atom_type = item["type"]
                citations = [{
                    "page": chunk.metadata.get("page_number", "?"),
                    "source": chunk.metadata.get("source_file", "?"),
                    "chunk_id": chunk_id,
                }]
                atom_id = self.state.generate_id("atom")
                atom = KnowledgeAtom(
                    atom_id=atom_id,
                    content=content,
                    atom_type=atom_type,
                    citations=citations,
                    section_id=sec_id,
                    lo_id=l_id,
                )
                self.state.add_knowledge_atom(atom)
                self.state.link_chunk_to_atom(chunk_id, atom_id)
                created.append(atom_id)
        return (
            f"Atomized {len(chunk_ids)} chunk(s): created {len(created)} atoms, "
            f"skipped {skipped_dup} duplicate(s), {skipped_invalid} invalid."
        )

    def _tool_atomize_chunks(self, agent_id: str, input_str: str = "") -> str:
        """Run LLM on chunks → extract atoms → add atoms automatically. No params = all evidence_pool."""
        if not _agent_can(agent_id, "KnowledgeAtom", "C"):
            return "Permission denied."
        raw = (input_str or "").strip().lower()
        # No input or placeholder = operate on full evidence_pool
        if raw in ("", "all", "yes", "all chunks", "evidence_pool"):
            chunk_ids = list(self.state.evidence_pool.keys())
        else:
            parts = [p.strip() for p in (input_str or "").split("|")]
            chunk_ids = [x.strip() for x in (parts[0] or "").split(",") if x.strip()] if parts else []
            if not chunk_ids:
                chunk_ids = list(self.state.evidence_pool.keys())
        section_id = None
        lo_id = None
        if "|" in (input_str or ""):
            parts = [p.strip() for p in (input_str or "").split("|")]
            if len(parts) > 1 and parts[1]:
                section_id = parts[1]
            if len(parts) > 2 and parts[2]:
                lo_id = parts[2]
        if not chunk_ids:
            return "No chunks in evidence pool. Run retriever then curate_evidence first."
        return self.atomize_chunks(chunk_ids, section_id=section_id, lo_id=lo_id)

    def _tool_add_candidate_question(
        self, agent_id: str, question_text: str, question_type: str = "multiple_choice",
        correct_answer: Optional[str] = None, difficulty: str = "intermediate", topic_tags: Optional[str] = None,
        related_atoms: Optional[str] = None, evidence_chunks: Optional[str] = None, options: Optional[str] = None
    ) -> str:
        if not _agent_can(agent_id, "CandidateQuestion", "C"):
            return "Permission denied."
        q_id = self.state.generate_id("q")
        tags = [x.strip() for x in (topic_tags or "").split(",") if x.strip()]
        atoms = [x.strip() for x in (related_atoms or "").split(",") if x.strip()]
        chunks = [x.strip() for x in (evidence_chunks or "").split(",") if x.strip()]
        opt_list = [x.strip() for x in (options or "").split("|")] if options else None
        q = CandidateQuestion(
            q_id=q_id, question_text=question_text, question_type=question_type,
            correct_answer=correct_answer, difficulty=difficulty, topic_tags=tags,
            related_atoms=atoms, evidence_chunks=chunks, options=opt_list
        )
        self.state.add_candidate_question(q)
        return f"Added candidate question {q_id}. Use validate_question or reject_question to move it."

    def _tool_get_candidate_questions(self, agent_id: str, _tool_input: str = "") -> str:
        if not _agent_can(agent_id, "CandidateQuestion", "R"):
            return "Permission denied."
        if not self.state.candidate_questions:
            return "No candidate questions."
        return "\n".join(f"- {q_id}: {q.question_text[:80]}..." for q_id, q in self.state.candidate_questions.items())

    def _tool_validate_question(self, agent_id: str, q_id: str, validation_notes: str = "") -> str:
        if not _agent_can(agent_id, "ValidatedQuestion", "C") or not _agent_can(agent_id, "CandidateQuestion", "M"):
            return "Permission denied."
        return self._validate_question(q_id, validation_notes)

    def _tool_reject_question(self, agent_id: str, q_id: str, rejection_reason: str, fix_hints: str = "") -> str:
        if not _agent_can(agent_id, "RejectedQuestion", "C") or not _agent_can(agent_id, "CandidateQuestion", "M"):
            return "Permission denied."
        return self._reject_question(q_id, rejection_reason, fix_hints)

    def _tool_get_coverage(self, agent_id: str, _tool_input: str = "") -> str:
        if not _agent_can(agent_id, "CoverageStats", "R"):
            return "Permission denied."
        return self._get_coverage()

    def _tool_get_quality_metrics(self, agent_id: str, _tool_input: str = "") -> str:
        if not _agent_can(agent_id, "QualityMetrics", "R"):
            return "Permission denied."
        m = self.state.quality_metrics
        return f"Rejection rate: {m.rejection_rate:.2%}, Duplication: {m.duplication_score:.2f}, Avg evidence strength: {m.avg_evidence_strength:.2f}, Total: {m.total_questions}, Validated: {m.validated_questions}, Rejected: {m.rejected_questions}"

    def _tool_update_quality_metrics(self, agent_id: str, duplication_score: Optional[float] = None, avg_evidence_strength: Optional[float] = None) -> str:
        if not _agent_can(agent_id, "QualityMetrics", "U"):
            return "Permission denied."
        self.state.update_quality_metrics_fields(duplication_score=duplication_score, avg_evidence_strength=avg_evidence_strength)
        return "Quality metrics updated."

    def _tool_create_quiz_pack(self, agent_id: str, title: str) -> str:
        """Create an empty quiz pack (draft). Next: add_questions_to_pack then export_pack."""
        if not _agent_can(agent_id, "QuizPack", "C") or not _agent_can(agent_id, "CooperativeState", "U"):
            return "Permission denied."
        pack_id = self.state.generate_id("pack")
        quiz_pack = QuizPack(
            pack_id=pack_id,
            title=title.strip() or "Quiz",
            questions=[],
            metadata={},
            coverage_summary={},
        )
        self.state.draft_quiz_packs[pack_id] = quiz_pack
        return f"Created draft pack {pack_id} (title: {quiz_pack.title}). Use add_questions_to_pack({pack_id}, question_ids) then export_pack({pack_id})."

    def _tool_add_questions_to_pack(self, agent_id: str, input_str: str) -> str:
        """Add validated questions to a draft pack. Input: pack_id|q_id1,q_id2,q_id3 (or pack_id only to add all validated)."""
        if not _agent_can(agent_id, "QuizPack", "C") or not _agent_can(agent_id, "ValidatedQuestion", "R"):
            return "Permission denied."
        raw = (input_str or "").strip()
        if "|" in raw:
            pack_id, rest = raw.split("|", 1)
            q_ids = [x.strip() for x in rest.split(",") if x.strip()]
        else:
            pack_id = raw
            q_ids = []
        pack_id = pack_id.strip()
        if not pack_id:
            return "Input: pack_id|q_id1,q_id2,q_id3 or pack_id (to add all validated questions)."
        pack = self.state.draft_quiz_packs.get(pack_id)
        if not pack:
            return f"Pack {pack_id} not found. Create one with create_quiz_pack(title) first."
        if not q_ids:
            q_ids = list(self.state.validated_questions.keys())
        added = 0
        existing_ids = {q.q_id for q in pack.questions}
        for q_id in q_ids:
            if q_id in existing_ids:
                continue
            q = self.state.validated_questions.get(q_id)
            if q:
                pack.questions.append(q)
                existing_ids.add(q_id)
                added += 1
        return f"Added {added} validated question(s) to pack {pack_id}. Total: {len(pack.questions)}. Use export_pack({pack_id}) to finalize."

    def _tool_export_pack(self, agent_id: str, pack_id: str) -> str:
        """Finalize a draft pack and append to final_quiz_packs."""
        if not _agent_can(agent_id, "QuizPack", "C") or not _agent_can(agent_id, "CooperativeState", "U"):
            return "Permission denied."
        pack = self.state.draft_quiz_packs.pop(pack_id.strip(), None)
        if not pack:
            return f"Pack {pack_id} not found in drafts. Create with create_quiz_pack then add_questions_to_pack."
        pack.metadata = {
            "pdf_id": self.pdf_id,
            "total_atoms": len(self.state.knowledge_atoms),
            "total_chunks": len(self.state.evidence_pool),
            "generation_date": str(Path(__file__).stat().st_mtime),
        }
        pack.coverage_summary = {
            "coverage_matrix": {k: {"topic": v.topic, "difficulty": v.difficulty, "type": v.question_type, "validated": v.validated_count, "rejected": v.rejected_count} for k, v in self.state.coverage_matrix.items()},
            "quality_metrics": {
                "rejection_rate": self.state.quality_metrics.rejection_rate,
                "avg_evidence_strength": self.state.quality_metrics.avg_evidence_strength,
                "total_questions": self.state.quality_metrics.total_questions,
                "validated": self.state.quality_metrics.validated_questions,
                "rejected": self.state.quality_metrics.rejected_questions,
            },
        }
        self.state.final_quiz_packs.append(pack)
        return f"Exported pack {pack.pack_id} with {len(pack.questions)} questions to final_quiz_packs."

    def _tool_get_state_summary(self, agent_id: str, _tool_input: str = "") -> str:
        return self._get_state_summary()

    def _tool_verify_statement(self, agent_id: str, statement: str) -> str:
        return self._verify(statement)

    def _tool_verify_candidate_question(self, agent_id: str, q_id: str) -> str:
        """Verifier-only: verify one candidate question against evidence and move to validated/rejected."""
        if not _agent_can(agent_id, "CandidateQuestion", "R") or not _agent_can(agent_id, "CandidateQuestion", "M"):
            return "Permission denied."
        return self._verify_candidate_question(q_id.strip())

    def _tool_verify_all_candidates(self, agent_id: str, _dummy: str = "") -> str:
        """Verifier-only: verify all candidate questions against evidence; move each to validated or rejected."""
        if not _agent_can(agent_id, "CandidateQuestion", "R") or not _agent_can(agent_id, "CandidateQuestion", "M"):
            return "Permission denied."
        return self._verify_all_candidate_questions()

    def _tool_generate_quiz_draft(self, agent_id: str, topic: str, question_type: str = "multiple_choice") -> str:
        if not _agent_can(agent_id, "CandidateQuestion", "C"):
            return "Permission denied."
        return self._generate_quiz(topic, question_type)

    def _create_tools_for_agent(self, agent_id: str) -> List[Tool]:
        """Build tool list for an agent based on its privilege matrix."""
        from functools import partial
        tools = []
        can = lambda e, p: _agent_can(agent_id, e, p)

        if can("DocSection", "R"):
            tools.append(Tool(name="get_outline", description="Get document outline (sections, titles, slide ranges, weights).", func=partial(self._tool_get_outline, agent_id)))
        if can("DocSection", "C") or can("DocSection", "U"):
            def _build_sec(x: str):
                return self._tool_build_outline_section(agent_id, x or "")
            tools.append(Tool(
                name="build_outline_section",
                description=(
                    "Add exactly ONE document section. Input: JSON object {\"section_id\": \"sec_1\", \"title\": \"Introduction\", \"start_slide\": 1, \"end_slide\": 5, \"importance_weight\": 0.5} "
                    "OR pipe format: sec_1|Introduction|1|5|0.5. Use unique section_id (sec_1, sec_2, ...), real titles and numbers. Never use placeholders like section_id or title."
                ),
                func=_build_sec,
            ))
            tools.append(Tool(name="set_full_outline", description="Set full outline from JSON: list of {section_id, title, slide_range: [start,end], importance_weight}.", func=partial(self._tool_set_full_outline, agent_id)))
            if agent_id == "structure_outline":
                tools.append(Tool(name="build_outline_from_document", description="Build outline from document: retrieve TOC/structure chunks, extract sections, assign page ranges and weights, validate no overlaps, store outline. Input: any (e.g. 'yes') to run.", func=lambda x: self._tool_build_outline_from_document(agent_id, x or "")))

        if can("LearningObjective", "R"):
            tools.append(Tool(name="get_learning_objectives", description="Get all learning objectives with section and difficulty.", func=partial(self._tool_get_learning_objectives, agent_id)))
        if can("LearningObjective", "C"):
            def _add_lo(x: str):
                parts = x.split("|")
                if len(parts) < 4:
                    return "Use: lo_id|section_id|objective_text|difficulty_level"
                return self._tool_add_learning_objective(agent_id, parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip())
            tools.append(Tool(name="add_learning_objective", description="Add LO: lo_id|section_id|objective_text|difficulty_level (beginner/intermediate/advanced).", func=_add_lo))
            if agent_id == "curriculum_lo":
                tools.append(Tool(name="generate_learning_objectives_for_outline", description="Create LOs from outline: for each section retrieve chunks, prompt LLM for objectives, validate/dedupe/balance difficulty, store. Input: any (e.g. 'yes') to run.", func=lambda x: self._tool_generate_learning_objectives_for_outline(agent_id, x or "")))

        if can("EvidenceChunk", "C"):
            tools.append(Tool(name="retrieve_chunks", description="Retrieve chunks from vector DB by query and add to evidence_pool. Input: query (optional: query, k).", func=lambda q: self._tool_retrieve_chunks(agent_id, q, 5)))
        if can("EvidenceChunk", "R"):
            tools.append(Tool(name="get_evidence_chunk", description="Get evidence chunk by chunk_id.", func=lambda cid: self._tool_get_chunk(agent_id, cid)))
        if can("EvidenceChunk", "U"):
            def _upd_chunk(x: str):
                parts = x.split("|")
                if len(parts) < 1:
                    return "Use: chunk_id|relevance_score|quality_score|relevance_tags"
                cid = parts[0].strip()
                rel_s, qual_s, tags = None, None, None
                if len(parts) > 1 and parts[1].strip():
                    try:
                        rel_s = float(parts[1].strip())
                    except ValueError:
                        pass
                if len(parts) > 2 and parts[2].strip():
                    try:
                        qual_s = float(parts[2].strip())
                    except ValueError:
                        pass
                if len(parts) > 3:
                    tags = parts[3].strip()
                return self._tool_update_chunk(agent_id, cid, relevance_score=rel_s, quality_score=qual_s, relevance_tags=tags)
            tools.append(Tool(name="update_evidence_chunk", description="Update chunk. Input: chunk_id|relevance_score|quality_score|relevance_tags (comma-separated). Quality = curator score (noise, duplication, trust).", func=_upd_chunk))
        if can("EvidenceChunk", "U"):
            tools.append(Tool(name="evaluate_chunk_quality", description="Evaluate one chunk's quality (noise, duplication, source trust) via LLM and set quality_score. Input: chunk_id.", func=lambda x: self._tool_evaluate_chunk_quality(agent_id, x.strip())))
            tools.append(Tool(name="evaluate_all_chunks_quality", description="Evaluate quality for all chunks in evidence_pool and set each quality_score. Input: any (e.g. 'yes').", func=lambda x: self._tool_evaluate_all_chunks_quality(agent_id, x)))
        if can("EvidenceChunk", "L") and can("KnowledgeAtom", "L"):
            tools.append(Tool(name="link_chunk_to_atom", description="Link evidence chunk to knowledge atom. Input: chunk_id|atom_id.", func=lambda x: self._tool_link_chunk_to_atom(agent_id, *x.split("|")[:2]) if "|" in x else "chunk_id|atom_id"))
        if can("CandidateQuestion", "L") or can("ValidatedQuestion", "L"):
            def _link_chunk_q(x: str):
                parts = x.split("|")
                if len(parts) < 2:
                    return "Use: q_id|chunk_id or q_id|chunk_id|validated"
                validated = len(parts) > 2 and parts[2].strip().lower() in ("true", "1", "yes")
                return self._tool_link_chunk_to_question(agent_id, parts[0].strip(), parts[1].strip(), validated=validated)
            tools.append(Tool(name="link_chunk_to_question", description="Link chunk to question (this chunk supports question Q). Input: q_id|chunk_id or q_id|chunk_id|validated.", func=_link_chunk_q))
        if can("EvidenceChunk", "U"):
            tools.append(Tool(name="mark_duplicate", description="Mark chunk as duplicate of another. Input: chunk_id|canonical_chunk_id.", func=lambda x: self._tool_mark_duplicate(agent_id, *x.split("|")[:2]) if "|" in x else "chunk_id|canonical_chunk_id"))
            tools.append(Tool(name="curate_evidence", description="Curate all evidence in the pool: dedupe chunks, score quality, generate tags. Operates on state.evidence_pool automatically. No parameters required; input any (e.g. 'yes') to run.", func=lambda x: self._tool_curate_evidence(agent_id, x or "")))

        if can("KnowledgeAtom", "C"):
            def _add_atom(x: str):
                parts = [p.strip() for p in x.split("|")]
                if len(parts) < 1:
                    return "Use: content|atom_type|chunk_ids|section_id|lo_id (at least content)"
                return self._tool_add_knowledge_atom(
                    agent_id,
                    parts[0],
                    parts[1] if len(parts) > 1 else "fact",
                    parts[2] if len(parts) > 2 else None,
                    parts[3] if len(parts) > 3 else None,
                    parts[4] if len(parts) > 4 else None,
                )
            tools.append(Tool(name="add_knowledge_atom", description="Add knowledge atom. Input: content|atom_type|chunk_ids(comma)|section_id|lo_id.", func=_add_atom))
            tools.append(Tool(name="atomize_chunks", description="Extract knowledge atoms from chunks via LLM and add them automatically (validate, dedupe, citations, link chunk→atom). Operates on full state.evidence_pool when no input or input 'yes'/'all'. Optional: chunk_id1,chunk_id2 or chunk_ids|section_id|lo_id for specific chunks.", func=lambda x: self._tool_atomize_chunks(agent_id, x or "")))
        if can("KnowledgeAtom", "R"):
            tools.append(Tool(name="get_knowledge_atom", description="Get knowledge atom by atom_id.", func=lambda aid: self._tool_get_atom(agent_id, aid)))
        if can("KnowledgeAtom", "L") and agent_id == "curriculum_lo":
            tools.append(Tool(name="bind_atom_to_lo", description="Bind knowledge atom to learning objective. Input: atom_id|lo_id.", func=lambda x: self._tool_bind_atom_to_lo(agent_id, *x.split("|")[:2]) if "|" in x else "atom_id|lo_id"))

        if can("CandidateQuestion", "C"):
            def _add_q(x: str):
                parts = x.split("|")
                if len(parts) < 1:
                    return "Use: question_text|type|correct_answer|difficulty|topic_tags|atoms|chunks|options"
                return self._tool_add_candidate_question(
                    agent_id,
                    parts[0].strip(),
                    parts[1].strip() if len(parts) > 1 else "multiple_choice",
                    parts[2].strip() if len(parts) > 2 else None,
                    parts[3].strip() if len(parts) > 3 else "intermediate",
                    parts[4] if len(parts) > 4 else None,
                    parts[5] if len(parts) > 5 else None,
                    parts[6] if len(parts) > 6 else None,
                    parts[7] if len(parts) > 7 else None,
                )
            tools.append(Tool(name="add_candidate_question", description="Add one candidate question. Input: question_text|type|correct_answer|difficulty|topic_tags|atoms|chunks|options.", func=_add_q))
            tools.append(Tool(name="generate_quiz_draft", description="Generate quiz questions on a topic (multiple_choice or open_ended); adds to candidate_questions.", func=lambda t: self._tool_generate_quiz_draft(agent_id, t, "multiple_choice")))
        if can("CandidateQuestion", "R"):
            tools.append(Tool(name="get_candidate_questions", description="List all candidate question IDs and previews.", func=partial(self._tool_get_candidate_questions, agent_id)))

        if can("ValidatedQuestion", "C") and can("CandidateQuestion", "M"):
            def _val_q(x: str):
                parts = x.split("|", 1)
                return self._tool_validate_question(agent_id, parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
            tools.append(Tool(name="validate_question", description="Validate candidate question by q_id; moves to validated_questions. Input: q_id or q_id|validation_notes.", func=_val_q))
        if can("RejectedQuestion", "C") and can("CandidateQuestion", "M"):
            def _rej_q(x: str):
                parts = x.split("|", 2)
                return self._tool_reject_question(agent_id, parts[0].strip(), parts[1].strip() if len(parts) > 1 else "Failed verification", parts[2].strip() if len(parts) > 2 else "")
            tools.append(Tool(name="reject_question", description="Reject candidate question. Input: q_id|rejection_reason|fix_hints.", func=_rej_q))

        if can("CoverageStats", "R"):
            tools.append(Tool(name="get_coverage_matrix", description="Get coverage matrix (topic, difficulty, type counts).", func=partial(self._tool_get_coverage, agent_id)))
            tools.append(Tool(name="get_next_targets", description="Get next N generation targets from coverage gaps (for quiz/retrieval). Input: N (default 5).", func=lambda x: self._tool_get_next_targets(agent_id, (x or "5").strip())))
        if can("QualityMetrics", "R"):
            tools.append(Tool(name="get_quality_metrics", description="Get quality metrics (rejection rate, duplication, evidence strength).", func=partial(self._tool_get_quality_metrics, agent_id)))
            if agent_id == "quality_duplicate_detector":
                tools.append(Tool(name="get_duplicate_report", description="Report duplicate chunks and questions and suggest remove/merge/regenerate.", func=partial(self._tool_get_duplicate_report, agent_id)))
        if can("QualityMetrics", "U"):
            def _upd_quality(x: str):
                parts = x.split("|")[:2]
                dup, avg = None, None
                if len(parts) > 0 and parts[0].strip():
                    try:
                        dup = float(parts[0].strip())
                    except ValueError:
                        pass
                if len(parts) > 1 and parts[1].strip():
                    try:
                        avg = float(parts[1].strip())
                    except ValueError:
                        pass
                return self._tool_update_quality_metrics(agent_id, duplication_score=dup, avg_evidence_strength=avg)
            tools.append(Tool(name="update_quality_metrics", description="Update quality metrics. Input: duplication_score|avg_evidence_strength.", func=_upd_quality))
            if agent_id == "quality_duplicate_detector":
                tools.append(Tool(name="detect_duplicate_chunks", description="Detect duplicate chunks by similarity (threshold default 0.92). Input: optional threshold.", func=lambda x: self._tool_detect_duplicate_chunks(agent_id, (x or "").strip())))
                tools.append(Tool(name="detect_duplicate_questions", description="Detect duplicate questions by stem similarity (threshold default 0.90). Input: optional threshold.", func=lambda x: self._tool_detect_duplicate_questions(agent_id, (x or "").strip())))

        if can("QuizPack", "C"):
            tools.append(Tool(name="create_quiz_pack", description="Create an empty draft quiz pack. Input: title. Workflow: create_quiz_pack(title) then add_questions_to_pack(pack_id|q_ids) then export_pack(pack_id).", func=lambda x: self._tool_create_quiz_pack(agent_id, (x or "").strip() or "Quiz")))
            if _agent_can(agent_id, "ValidatedQuestion", "R"):
                tools.append(Tool(name="add_questions_to_pack", description="Add validated questions to a draft pack. Input: pack_id|q_id1,q_id2,q_id3 or pack_id alone to add all validated questions.", func=lambda x: self._tool_add_questions_to_pack(agent_id, (x or "").strip())))
            tools.append(Tool(name="export_pack", description="Finalize a draft pack and append to final_quiz_packs. Input: pack_id.", func=lambda x: self._tool_export_pack(agent_id, (x or "").strip())))

        if agent_id == "verifier":
            if can("EvidenceChunk", "R"):
                tools.append(Tool(name="verify_statement", description="Verify an arbitrary statement against the source document (fact-check). Input: the statement.", func=partial(self._tool_verify_statement, agent_id)))
            if can("CandidateQuestion", "R") and can("CandidateQuestion", "M"):
                tools.append(Tool(name="verify_candidate_question", description="Verify one candidate question against its evidence (rules: answer supported, options consistent, no hallucinations). Input: q_id.", func=lambda x: self._tool_verify_candidate_question(agent_id, x.strip())))
                tools.append(Tool(name="verify_all_candidates", description="Verify all candidate questions; move each to validated or rejected. Input: any (e.g. 'yes') to run.", func=lambda x: self._tool_verify_all_candidates(agent_id, x)))

        tools.append(Tool(name="get_state_summary", description="Get summary of cooperative state (counts, metrics).", func=partial(self._tool_get_state_summary, agent_id)))
        return tools

    STRUCTURE_OUTLINE_SYSTEM_PROMPT = """You are the Structure/Outline Agent. You build and maintain the document outline (sections with titles, slide ranges, importance weights).

CRITICAL RULES:
- Never copy tool parameter names literally. Never use placeholder strings as values: not "section_id", "title", "start_slide", "end_slide", "weight", "importance_weight", "n/a", "tbd".
- For build_outline_section: provide REAL values only. section_id MUST be a unique identifier like sec_1, sec_2, sec_3 (pattern: sec_ then a number). Title MUST be a real section name (e.g. "Introduction", "Methodology"). start_slide and end_slide MUST be integers (e.g. 1 and 5). importance_weight MUST be a number between 0.1 and 1.0 (e.g. 0.5).
- Call build_outline_section ONCE per section. One section per tool call. Do not pass multiple sections in one input.
- Prefer build_outline_from_document when asked to "build" or "create" the outline from the document: it retrieves content, infers sections, validates ranges, and stores the outline. Use build_outline_section only when adding a single section manually.
- If you cannot infer a valid outline from the document, say so clearly. Do not invent placeholder data.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (real values only; never placeholders)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

    VERIFIER_SYSTEM_PROMPT = """You are the Verifier Agent. Your PRIMARY job is to verify CANDIDATE QUESTIONS against evidence and move them to validated_questions or rejected_questions.

WORKFLOW:
1. Use get_candidate_questions to see which candidate questions exist.
2. Use verify_all_candidates (input: yes) to check EVERY candidate against its evidence and move each to validated or rejected. This is the main action you should take when asked to "verify", "validate questions", or "check questions".
3. Optionally use verify_candidate_question(q_id) to verify a single question by ID.

RULES:
- When the user asks to verify, validate, or check questions, ALWAYS run verify_all_candidates (or verify_candidate_question for specific IDs). Do not only verify random statements.
- Each candidate is checked against VERIFICATION_RULES (answer supported by evidence, options consistent, no hallucinations). You move it to validated_questions or rejected_questions accordingly.
- Use verify_statement only for ad-hoc fact-checking of a single claim (e.g. "Is X true?"), not for the candidate-question workflow.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

    def _create_agents(self):
        """Create specialized agents with privilege-based tools"""
        from langchain_core.prompts import ChatPromptTemplate
        # Use string-based agent_scratchpad (create_react_agent supplies Thought/Action/Observation as string).
        # Do not use MessagesPlaceholder for agent_scratchpad or hub.pull("hwchase17/react") here, as that
        # can expect a list of messages and raise: "agent_scratchpad should be a list of base messages, got ... str"
        react_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that can use tools to answer questions.
You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""),
            ("user", "{input}\n\n{agent_scratchpad}"),
        ])

        from langchain_core.prompts import ChatPromptTemplate
        # agent_scratchpad must be a string (same as hub ReAct prompt), not MessagesPlaceholder
        structure_react_prompt = ChatPromptTemplate.from_messages([
            ("system", self.STRUCTURE_OUTLINE_SYSTEM_PROMPT),
            ("user", "{input}\n\n{agent_scratchpad}"),
        ])
        verifier_react_prompt = ChatPromptTemplate.from_messages([
            ("system", self.VERIFIER_SYSTEM_PROMPT),
            ("user", "{input}\n\n{agent_scratchpad}"),
        ])

        agent_ids = [
            "structure_outline",
            "curriculum_lo",
            "retriever",
            "evidence_curator",
            "atomizer",
            "quiz_generator",
            "verifier",
            "coverage_controller",
            "quality_duplicate_detector",
            "packager_export",
        ]
        self._agent_executors: Dict[str, AgentExecutor] = {}
        for aid in agent_ids:
            tool_list = self._create_tools_for_agent(aid)
            if not tool_list:
                tool_list = [Tool(name="get_state_summary", description="Get state summary.", func=self._get_state_summary)]
            prompt = (structure_react_prompt if aid == "structure_outline" else
                      verifier_react_prompt if aid == "verifier" else react_prompt)
            self._agent_executors[aid] = AgentExecutor(
                agent=create_react_agent(self.llm, tool_list, prompt),
                tools=tool_list,
                verbose=True,
                max_iterations=8,
                handle_parsing_errors=True,
            )

        # Backward-compatible aliases
        self.retriever_agent = self._agent_executors["retriever"]
        self.quiz_generator_agent = self._agent_executors["quiz_generator"]
        self.verifier_agent = self._agent_executors["verifier"]
    
    def run_agent(self, agent_id: str, input_text: str) -> Dict[str, Any]:
        """Run a privileged agent by id. agent_id one of: structure_outline, curriculum_lo, retriever, evidence_curator, atomizer, quiz_generator, verifier, coverage_controller, quality_duplicate_detector, packager_export."""
        if agent_id not in self._agent_executors:
            return {"role": agent_id, "error": f"Unknown agent: {agent_id}", "result": ""}
        config = self._langsmith_config(agent_id=agent_id)
        config["run_name"] = f"agent_{agent_id}"
        result = self._agent_executors[agent_id].invoke({"input": input_text}, config=config)
        return {"role": agent_id, "result": result["output"]}

    def structure_outline(self, input_text: str) -> Dict:
        """Structure/Outline Agent: build outline, slide ranges, weights; store in state."""
        return self.run_agent("structure_outline", input_text)

    def curriculum_lo(self, input_text: str) -> Dict:
        """Curriculum/LO Agent: create learning objectives, set difficulty; bind atoms to LOs."""
        return self.run_agent("curriculum_lo", input_text)

    def retrieve(self, query: str, k: int = 5) -> Dict:
        """Retriever Agent: get relevant chunks and add to evidence_pool."""
        config = self._langsmith_config(agent_id="retriever", query=query)
        config["run_name"] = "agent_retriever"
        result = self.retriever_agent.invoke({"input": f"Retrieve relevant chunks for: {query}"}, config=config)
        return {"role": "retriever", "query": query, "result": result["output"]}

    def evidence_curator(self, input_text: str) -> Dict:
        """Evidence Curator/Ranker Agent: dedupe, score, tag chunks; link to atoms/questions."""
        return self.run_agent("evidence_curator", input_text)

    def atomizer(self, input_text: str) -> Dict:
        """Atomizer Agent: extract knowledge atoms from chunks; link to section/LO."""
        return self.run_agent("atomizer", input_text)

    def generate_quiz(self, topic: str, question_type: str = "multiple_choice") -> Dict:
        """Quiz Generator Agent: create candidate questions; use atoms/chunks."""
        config = self._langsmith_config(agent_id="quiz_generator", topic=topic)
        config["run_name"] = "agent_quiz_generator"
        result = self.quiz_generator_agent.invoke({"input": f"Generate {question_type} quiz questions on: {topic}"}, config=config)
        return {"role": "quiz_generator", "topic": topic, "question_type": question_type, "result": result["output"]}

    def verify(self, statement: str) -> Dict:
        """Verify an arbitrary statement against the document (ad-hoc fact-check). For verifying candidate questions, use verify_candidate_question or verify_all_candidates."""
        config = self._langsmith_config(agent_id="verifier", statement=statement)
        config["run_name"] = "agent_verifier"
        result = self.verifier_agent.invoke({"input": f"Verify statement: {statement}"}, config=config)
        return {"role": "verifier", "statement": statement, "result": result["output"]}

    def verify_candidate_question(self, q_id: str) -> Dict:
        """Verify one candidate question against its evidence (rules: answer supported, options consistent, no hallucinations); move to validated or rejected."""
        out = self._verify_candidate_question(q_id)
        return {"role": "verifier", "q_id": q_id, "result": out}

    def verify_all_candidates(self, limit: Optional[int] = None) -> Dict:
        """Verify candidate questions against evidence; optional batch limit."""
        out = self._verify_all_candidate_questions(limit=limit)
        return {"role": "verifier", "result": out}

    def coverage_controller(self, input_text: str) -> Dict:
        """Coverage Controller Agent: analyze gaps; steer generation targets."""
        return self.run_agent("coverage_controller", input_text)

    def quality_duplicate_detector(self, input_text: str) -> Dict:
        """Quality/Duplicate Detector Agent: flag low-quality/duplicate chunks and questions."""
        return self.run_agent("quality_duplicate_detector", input_text)

    def packager_export(self, input_text: str) -> Dict:
        """Packager/Export Agent: create QuizPack; append to final_quiz_packs."""
        return self.run_agent("packager_export", input_text)

    # -------------------------------------------------------------------------
    # Central orchestrator: rule-based workflow with default sequence, loops,
    # and stop conditions. Does not replace specialist agents; it calls them.
    # -------------------------------------------------------------------------

    def _orchestration_state_snapshot(self) -> Dict[str, Any]:
        """Capture a minimal state snapshot for progress detection and logging."""
        canonical_chunks = sum(
            1 for c in self.state.evidence_pool.values()
            if getattr(c, "duplicate_of", None) is None
        )
        low_quality_chunks = sum(
            1 for c in self.state.evidence_pool.values()
            if "low_quality" in getattr(c, "flags", [])
        )
        return {
            "doc_outline_len": len(self.state.doc_outline),
            "learning_objectives_len": len(self.state.learning_objectives),
            "evidence_pool_len": len(self.state.evidence_pool),
            "canonical_chunks": canonical_chunks,
            "low_quality_chunks": low_quality_chunks,
            "knowledge_atoms_len": len(self.state.knowledge_atoms),
            "candidate_questions_len": len(self.state.candidate_questions),
            "validated_questions_len": len(self.state.validated_questions),
            "rejected_questions_len": len(self.state.rejected_questions),
            "duplication_score": self.state.quality_metrics.duplication_score,
            "rejection_rate": self.state.quality_metrics.rejection_rate,
            "avg_evidence_strength": self.state.quality_metrics.avg_evidence_strength,
            "final_quiz_packs_len": len(self.state.final_quiz_packs),
        }

    def needs_outline(self) -> bool:
        """True if no valid document outline exists."""
        return len(self.state.doc_outline) == 0

    def needs_learning_objectives(self) -> bool:
        """True if no or too few learning objectives."""
        return len(self.state.learning_objectives) < MIN_LO_COUNT

    def needs_more_evidence(self, topic: str) -> bool:
        """True if evidence pool is empty or too small (canonical chunks)."""
        canonical = sum(
            1 for c in self.state.evidence_pool.values()
            if getattr(c, "duplicate_of", None) is None
        )
        return canonical < MIN_EVIDENCE_CHUNKS

    def needs_deduplication(self) -> bool:
        """True if duplication score is too high or many chunks are low_quality."""
        dup = self.state.quality_metrics.duplication_score
        low_q = sum(1 for c in self.state.evidence_pool.values() if "low_quality" in getattr(c, "flags", []))
        total = len(self.state.evidence_pool)
        return dup > MAX_DUPLICATION_SCORE or (total > 0 and low_q / total > 0.4)

    def needs_atomization(self) -> bool:
        """True if atom count is too low and we have curated chunks to atomize."""
        canonical = sum(1 for c in self.state.evidence_pool.values() if getattr(c, "duplicate_of", None) is None)
        return len(self.state.knowledge_atoms) < MIN_ATOMS and canonical >= MIN_EVIDENCE_CHUNKS

    def needs_more_questions(self, topic: str) -> bool:
        """True if we have atoms but no or few candidates, or coverage gaps say we need more."""
        gaps = self.compute_coverage_gaps(self.coverage_plan)
        has_gaps = len(gaps) > 0 and any(g.get("gap", 0) > 0 for g in gaps)
        have_atoms = len(self.state.knowledge_atoms) >= MIN_ATOMS
        few_candidates = len(self.state.candidate_questions) + len(self.state.validated_questions) < 3
        return have_atoms and (few_candidates or has_gaps)

    def needs_verification(self) -> bool:
        """True if there are candidate questions not yet verified."""
        return len(self.state.candidate_questions) > 0

    def needs_coverage_repair(self) -> bool:
        """True if coverage controller reports gaps (need more questions or targeted generation)."""
        gaps = self.compute_coverage_gaps(self.coverage_plan)
        return len(gaps) > 0 and any(g.get("gap", 0) > 0 for g in gaps)

    def ready_for_packaging(self) -> bool:
        """True if we have enough validated questions and no critical gaps (or gaps acceptable)."""
        val = len(self.state.validated_questions)
        gaps = self.compute_coverage_gaps(self.coverage_plan)
        critical_gaps = [g for g in gaps if g.get("gap", 0) > 0]
        return val >= MIN_VALIDATED_QUESTIONS and (len(critical_gaps) == 0 or val >= 3)

    def get_next_orchestration_action(
        self,
        topic: str,
        last_agent_id: Optional[str] = None,
        step: int = 0,
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Rule-based selection of next agent. Returns (agent_id, reason, suggested_input).
        Returns (None, reason, None) when workflow should stop (done or cannot proceed).
        Order of checks defines default sequence and loops.
        """
        # A. Outline missing
        if self.needs_outline():
            return (
                "structure_outline",
                "No document outline; build outline first.",
                "Build document outline: section_id, title, slide range, importance weight per section.",
            )
        # B. Learning objectives missing
        if self.needs_learning_objectives():
            return (
                "curriculum_lo",
                "No or insufficient learning objectives; create LOs.",
                "Create learning objectives for each section with difficulty levels.",
            )
        # C. Not enough evidence
        if self.needs_more_evidence(topic):
            return (
                "retriever",
                f"Evidence pool too small (need at least {MIN_EVIDENCE_CHUNKS} canonical chunks); retrieve more.",
                topic or "key concepts from the document",
            )
        # C'. After retriever: run duplicate detector then curator (default sequence)
        if last_agent_id == "retriever" and len(self.state.evidence_pool) >= MIN_EVIDENCE_CHUNKS:
            return (
                "quality_duplicate_detector",
                "Evidence just retrieved; run duplicate detector before curator.",
                "yes",
            )
        if last_agent_id == "quality_duplicate_detector":
            return (
                "evidence_curator",
                "Run curator after duplicate detection to score and tag chunks.",
                "Curate all evidence in the pool.",
            )
        # D. Duplicates or low quality evidence (loop)
        if self.needs_deduplication():
            if last_agent_id != "quality_duplicate_detector":
                return (
                    "quality_duplicate_detector",
                    f"Duplication score or low-quality chunk ratio too high (dup={self.state.quality_metrics.duplication_score:.2f}); run duplicate detector.",
                    "yes",
                )
            return (
                "evidence_curator",
                "Run curator after duplicate detection to score and tag chunks.",
                "Curate all evidence in the pool.",
            )
        # E. Not enough atoms
        if self.needs_atomization():
            return (
                "atomizer",
                f"Too few knowledge atoms (have {len(self.state.knowledge_atoms)}, need at least {MIN_ATOMS}); atomize chunks.",
                "yes",
            )
        # F. Need more questions (generate)
        if self.needs_more_questions(topic):
            targets = self.get_next_targets(n=1)
            gen_topic = topic
            if targets:
                t = targets[0]
                gen_topic = str(t.get("topic") or topic)
            if last_agent_id == "coverage_controller":
                return (
                    "quiz_generator",
                    "Coverage gaps require more questions; generating for target.",
                    gen_topic,
                )
            if last_agent_id != "quiz_generator" or step % 6 > 2:
                return (
                    "quiz_generator",
                    "Need more candidate questions for coverage.",
                    gen_topic,
                )
        # G. Unverified candidates
        if self.needs_verification():
            return (
                "verifier",
                "Candidate questions pending; verify against evidence.",
                "Verify all candidate questions. Use get_candidate_questions then verify_all_candidates.",
            )
        # H. Rejection rate too high: do not package yet; run coverage to decide next target
        if (
            self.state.quality_metrics.total_questions > 0
            and self.state.quality_metrics.rejection_rate > MAX_REJECTION_RATE
        ):
            return (
                "coverage_controller",
                f"Rejection rate {self.state.quality_metrics.rejection_rate:.2%} > {MAX_REJECTION_RATE}; check coverage and consider more generation.",
                "Analyze coverage gaps and suggest next targets.",
            )
        # I. Coverage gaps
        if self.needs_coverage_repair():
            return (
                "coverage_controller",
                "Coverage gaps exist; analyze and steer next targets.",
                "Get next targets and suggest focus for retrieval or quiz generation.",
            )
        # J. Ready to package
        if self.ready_for_packaging():
            return (
                "packager_export",
                "Enough validated questions and no critical gaps; export quiz pack.",
                "Generated Quiz",
            )
        # No action (e.g. already packaged, or stuck)
        if len(self.state.final_quiz_packs) > 0:
            return (None, "Workflow complete: quiz pack(s) exported.", None)
        return (None, "No further action (state may be incomplete or at max iterations).", None)

    def run_full_workflow(self, topic: str) -> Dict[str, Any]:
        """
        Run the full orchestrated workflow: default sequence with conditional loops
        and stop conditions. Uses rule-based orchestration (no LLM for routing).
        Stops when: packaging succeeded, max steps, or no progress for N iterations.
        Returns final result, orchestration_log, and state summary.
        """
        self.orchestration_log = []
        last_agent_id: Optional[str] = None
        last_snapshot: Optional[Dict[str, Any]] = None
        no_progress_count = 0
        step = 0

        while step < MAX_ORCHESTRATION_STEPS:
            snapshot_before = self._orchestration_state_snapshot()
            next_agent, reason, suggested_input = self.get_next_orchestration_action(
                topic, last_agent_id=last_agent_id, step=step
            )
            if next_agent is None:
                self.orchestration_log.append({
                    "step": step + 1,
                    "agent": None,
                    "reason": reason,
                    "state_before": snapshot_before,
                    "state_after": self._orchestration_state_snapshot(),
                    "input": None,
                    "result_summary": "stop",
                })
                break

            # Invoke specialist agent
            try:
                if next_agent == "retriever":
                    result = self.retrieve(suggested_input or topic)
                    result_summary = str(result.get("result", ""))[:300]
                elif next_agent == "quiz_generator":
                    result = self.generate_quiz(suggested_input or topic)
                    result_summary = str(result.get("result", ""))[:300]
                elif next_agent == "verifier":
                    result = self.verify_all_candidates()
                    result_summary = str(result.get("result", ""))[:300]
                elif next_agent == "packager_export":
                    result = self.export_quiz_pack(suggested_input or "Generated Quiz")
                    result_summary = f"pack_id={result.get('pack_id')} questions={result.get('question_count')}"
                else:
                    inp = suggested_input or ""
                    result = self.run_agent(next_agent, inp)
                    result_summary = str(result.get("result", ""))[:300]
            except Exception as e:
                result_summary = f"error: {e}"
                result = {"role": next_agent, "error": str(e), "result": result_summary}

            snapshot_after = self._orchestration_state_snapshot()
            self.orchestration_log.append({
                "step": step + 1,
                "agent": next_agent,
                "reason": reason,
                "state_before": snapshot_before,
                "state_after": snapshot_after,
                "input": suggested_input,
                "result_summary": result_summary,
            })
            last_agent_id = next_agent

            # Stop if we just packaged
            if next_agent == "packager_export" and not result.get("error"):
                break

            # Progress detection
            if last_snapshot is not None:
                v_before = last_snapshot.get("validated_questions_len", 0)
                v_after = snapshot_after.get("validated_questions_len", 0)
                a_before = last_snapshot.get("knowledge_atoms_len", 0)
                a_after = snapshot_after.get("knowledge_atoms_len", 0)
                c_before = last_snapshot.get("canonical_chunks", 0)
                c_after = snapshot_after.get("canonical_chunks", 0)
                if (v_after == v_before and a_after == a_before and c_after == c_before):
                    no_progress_count += 1
                    if no_progress_count >= NO_PROGRESS_ITERATIONS:
                        self.orchestration_log.append({
                            "step": step + 2,
                            "agent": None,
                            "reason": f"No progress for {NO_PROGRESS_ITERATIONS} steps; stopping.",
                            "state_before": snapshot_after,
                            "state_after": snapshot_after,
                            "input": None,
                            "result_summary": "no_progress_stop",
                        })
                        break
                else:
                    no_progress_count = 0
            last_snapshot = snapshot_after
            step += 1

        final_snapshot = self._orchestration_state_snapshot()
        return {
            "topic": topic,
            "steps": step + 1,
            "orchestration_log": self.orchestration_log,
            "final_state": final_snapshot,
            "packages_exported": final_snapshot.get("final_quiz_packs_len", 0),
            "validated_questions": final_snapshot.get("validated_questions_len", 0),
        }

    def orchestrate(self, topic: str) -> Dict[str, Any]:
        """Alias for run_full_workflow(topic). Entry point for fully orchestrated mode."""
        return self.run_full_workflow(topic)
    
    @traceable(name="export_quiz_pack")
    def export_quiz_pack(self, title: str = "Generated Quiz") -> Dict:
        """Export final quiz pack from validated questions"""
        pack_id = self.state.generate_id("pack")
        quiz_pack = QuizPack(
            pack_id=pack_id,
            title=title,
            questions=list(self.state.validated_questions.values()),
            metadata={
                "pdf_id": self.pdf_id,
                "total_atoms": len(self.state.knowledge_atoms),
                "total_chunks": len(self.state.evidence_pool),
                "generation_date": str(Path(__file__).stat().st_mtime)
            },
            coverage_summary={
                "coverage_matrix": {k: {
                    "topic": v.topic,
                    "difficulty": v.difficulty,
                    "type": v.question_type,
                    "validated": v.validated_count,
                    "rejected": v.rejected_count
                } for k, v in self.state.coverage_matrix.items()},
                "quality_metrics": {
                    "rejection_rate": self.state.quality_metrics.rejection_rate,
                    "avg_evidence_strength": self.state.quality_metrics.avg_evidence_strength,
                    "total_questions": self.state.quality_metrics.total_questions,
                    "validated": self.state.quality_metrics.validated_questions,
                    "rejected": self.state.quality_metrics.rejected_questions
                }
            }
        )
        self.state.final_quiz_packs.append(quiz_pack)
        
        return {
            "pack_id": pack_id,
            "title": title,
            "question_count": len(quiz_pack.questions),
            "coverage": quiz_pack.coverage_summary,
            "questions": [
                {
                    "q_id": q.q_id,
                    "question": q.question_text,
                    "type": q.question_type,
                    "difficulty": q.difficulty,
                    "answer": q.correct_answer
                }
                for q in quiz_pack.questions
            ]
        }
    
    def get_state_dict(self) -> Dict:
        """Get complete state as dictionary for serialization"""
        return {
            "doc_outline": [
                {
                    "section_id": s.section_id,
                    "title": s.title,
                    "slide_range": s.slide_range,
                    "importance_weight": s.importance_weight
                }
                for s in self.state.doc_outline
            ],
            "learning_objectives": [
                {
                    "lo_id": lo.lo_id,
                    "section_id": lo.section_id,
                    "objective_text": lo.objective_text,
                    "difficulty_level": lo.difficulty_level
                }
                for lo in self.state.learning_objectives
            ],
            "knowledge_atoms": {
                k: {
                    "atom_id": v.atom_id,
                    "content": v.content,
                    "atom_type": v.atom_type,
                    "citations": v.citations
                }
                for k, v in self.state.knowledge_atoms.items()
            },
            "evidence_pool": {
                k: {
                    "chunk_id": v.chunk_id,
                    "content": v.content[:200],
                    "relevance_score": v.relevance_score,
                    "quality_score": v.quality_score,
                    "duplicate_of": getattr(v, "duplicate_of", None),
                    "metadata": v.metadata,
                }
                for k, v in self.state.evidence_pool.items()
            },
            "validated_questions": {
                k: {
                    "q_id": v.q_id,
                    "question_text": v.question_text,
                    "question_type": v.question_type,
                    "difficulty": v.difficulty
                }
                for k, v in self.state.validated_questions.items()
            },
            "quality_metrics": {
                "rejection_rate": self.state.quality_metrics.rejection_rate,
                "avg_evidence_strength": self.state.quality_metrics.avg_evidence_strength,
                "total_questions": self.state.quality_metrics.total_questions,
                "validated_questions": self.state.quality_metrics.validated_questions,
                "rejected_questions": self.state.quality_metrics.rejected_questions
            }
        }


def _run_orchestrated(args):
    """Run full orchestrated workflow: central orchestrator drives specialist agents."""
    print("=" * 70)
    print("🎯 ORCHESTRATED WORKFLOW (Central Orchestrator)")
    print("=" * 70)
    topic = args.topic or "k-means algorithm"
    print(f"\nTopic: {topic}")
    print("Orchestrator is rule-based: outline → LOs → retrieve → dedup → curator → atomize")
    print("→ quiz → verify → coverage → package (with loops when needed).\n")
    try:
        kwargs = {
            "pdf_id": args.pdf_id or "141",
            "vector_store_path": "vector_store",
            "llm_model": args.model,
            "model_type": args.model_type,
        }
        if args.model_type == "huggingface":
            kwargs.update({"device": args.device, "huggingface_api_key": args.api_key})
        else:
            kwargs.update({"openai_api_key": args.api_key})
        agents = RAGAgents(**kwargs)
        print(f"[OK] Agents initialized for PDF {args.pdf_id or '141'}\n")
    except Exception as e:
        print(f"[ERROR] Error initializing agents: {e}")
        return
    result = agents.orchestrate(topic)
    print("\n" + "=" * 70)
    print("ORCHESTRATION RESULT")
    print("=" * 70)
    print(f"Steps: {result.get('steps', 0)}")
    print(f"Packages exported: {result.get('packages_exported', 0)}")
    print(f"Validated questions: {result.get('validated_questions', 0)}")
    print("\nOrchestration log (agent order):")
    for entry in result.get("orchestration_log", [])[:40]:
        agent = entry.get("agent") or "STOP"
        reason = (entry.get("reason") or "")[:60]
        print(f"  Step {entry.get('step')}: {agent} — {reason}")
    if len(result.get("orchestration_log", [])) > 40:
        print("  ...")
    print("\n" + "=" * 70)
    print("Orchestrated run complete.")
    print("=" * 70)


def _run_demo(args):
    """Run a demo showing how the cooperative system works"""
    print("=" * 70)
    print("🎯 COOPERATIVE AGENTS DEMO (Privilege-based)")
    print("=" * 70)
    print("\nAgents and privileges: Structure/Outline, Curriculum/LO, Retriever,")
    print("Evidence Curator, Atomizer, Quiz Generator, Verifier, Coverage Controller,")
    print("Quality/Duplicate Detector, Packager/Export.")
    print("Workflow: retriever → duplicate_detector → curator → atomizer → quiz → verifier → packager")
    print("1. Retriever finds chunks → adds to evidence_pool")
    print("2. Duplicate detector marks duplicate chunks (before curator)")
    print("3. Curator dedupes, scores, tags chunks")
    print("4. Atomizer extracts knowledge atoms; Quiz generator creates questions")
    print("5. Verifier validates/rejects → moves to validated/rejected")
    print("6. State tracks coverage and quality metrics\n")
    
    # Initialize agents
    try:
        kwargs = {
            "pdf_id": args.pdf_id or "141",
            "vector_store_path": "vector_store",
            "llm_model": args.model,
            "model_type": args.model_type,
        }
        
        if args.model_type == "huggingface":
            kwargs.update({"device": args.device, "huggingface_api_key": args.api_key})
        else:
            kwargs.update({"openai_api_key": args.api_key})
        
        agents = RAGAgents(**kwargs)
        print(f"[OK] Agents initialized for PDF {args.pdf_id or '141'}\n")
    except Exception as e:
        print(f"[ERROR] Error initializing agents: {e}")
        return
    
    # Step 1: Retrieve chunks
    print("\n" + "=" * 70)
    print("STEP 1: Retriever Agent - Finding relevant chunks")
    print("=" * 70)
    query = args.query or "k-means clustering"
    result = agents.retrieve(query)
    print(f"\nQuery: {query}")
    print(f"\nState after retrieval:")
    print(f"  Evidence Chunks: {len(agents.state.evidence_pool)}")
    if agents.state.evidence_pool:
        chunk_ids = list(agents.state.evidence_pool.keys())[:3]
        print(f"  Sample Chunk IDs: {', '.join(chunk_ids)}")

    # Step 2: Duplicate detection (before curator so duplicates do not contaminate later steps)
    print("\n" + "=" * 70)
    print("STEP 2: Quality/Duplicate Detector - Detect duplicate chunks")
    print("=" * 70)
    result = agents.quality_duplicate_detector("yes")
    print(f"\nResult: {result.get('result', result)[:200]}...")
    print(f"  Evidence Chunks: {len(agents.state.evidence_pool)}")

    # Step 3: Curator
    print("\n" + "=" * 70)
    print("STEP 3: Evidence Curator - Dedupe, score, tag chunks")
    print("=" * 70)
    result = agents.evidence_curator("Curate all evidence in the pool.")
    print(f"\nResult: {str(result.get('result', result))[:300]}...")

    # Step 4: Atomizer
    print("\n" + "=" * 70)
    print("STEP 4: Atomizer - Extract knowledge atoms from chunks")
    print("=" * 70)
    result = agents.atomizer("yes")
    print(f"\nResult: {str(result.get('result', result))[:200]}...")
    print(f"  Knowledge Atoms: {len(agents.state.knowledge_atoms)}")

    # Step 5: Generate questions
    print("\n" + "=" * 70)
    print("STEP 5: Quiz Generator Agent - Creating questions")
    print("=" * 70)
    topic = args.topic or "k-means algorithm"
    result = agents.generate_quiz(topic)
    print(f"\nTopic: {topic}")
    print(f"\nState after question generation:")
    print(f"  Candidate Questions: {len(agents.state.candidate_questions)}")
    print(f"  Evidence Chunks: {len(agents.state.evidence_pool)}")
    if agents.state.candidate_questions:
        q_ids = list(agents.state.candidate_questions.keys())[:3]
        print(f"  Sample Question IDs: {', '.join(q_ids)}")
        for q_id in q_ids[:2]:
            q = agents.state.candidate_questions[q_id]
            print(f"    {q_id}: {q.question_text[:60]}...")
    
    # Step 6: Show state summary
    print("\n" + "=" * 70)
    print("STEP 6: State Summary")
    print("=" * 70)
    print(agents._get_state_summary())
    print("\n" + agents._get_coverage())

    # Step 7: Verifier - validate candidate questions
    if agents.state.candidate_questions:
        print("\n" + "=" * 70)
        print("STEP 7: Verifier Agent - Validating questions")
        print("=" * 70)
        # Validate first question as example
        q_id = list(agents.state.candidate_questions.keys())[0]
        q = agents.state.candidate_questions[q_id]
        print(f"\nValidating question {q_id}:")
        print(f"  Question: {q.question_text[:80]}...")
        
        # In a real scenario, the verifier would check the question
        # For demo, we'll just show the state change
        validated = ValidatedQuestion(
            q_id=q_id,
            question_text=q.question_text,
            question_type=q.question_type,
            correct_answer=q.correct_answer or "N/A",
            difficulty=q.difficulty,
            options=q.options,
            topic_tags=q.topic_tags,
            related_atoms=q.related_atoms,
            evidence_chunks=q.evidence_chunks,
            validation_notes="Demo validation"
        )
        agents.state.validate_question(q_id, validated)
        print(f"  [OK] Validated and moved to validated_questions")
        
        print(f"\nState after validation:")
        print(f"  Candidate Questions: {len(agents.state.candidate_questions)}")
        print(f"  Validated Questions: {len(agents.state.validated_questions)}")
        print(f"  Quality Metrics: Rejection Rate = {agents.state.quality_metrics.rejection_rate:.2%}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 FINAL STATE SUMMARY")
    print("=" * 70)
    print(agents._get_state_summary())
    
    if agents.state.validated_questions:
        print("\nExport Quiz Pack:")
        quiz_pack = agents.export_quiz_pack("Demo Quiz")
        print(f"  Pack ID: {quiz_pack['pack_id']}")
        print(f"  Questions: {quiz_pack['question_count']}")
        print(f"  Coverage: {len(quiz_pack['coverage']['coverage_matrix'])} topic-difficulty-type combinations")
    
    print("\n" + "=" * 70)
    print("Demo completed! 🎉")
    print("=" * 70)


def main():
    """Example usage of RAG Agents"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LangChain RAG Agents with Mistral')
    parser.add_argument('--pdf-id', type=str, default='141', help='PDF ID (default: 141)')
    parser.add_argument('--role', type=str,
                       choices=[
                           'structure_outline', 'curriculum_lo', 'retriever', 'evidence_curator',
                           'atomizer', 'quiz_generator', 'verifier', 'coverage_controller',
                           'quality_duplicate_detector', 'packager_export', 'all'
                       ],
                       default='all', help='Which agent to run')
    parser.add_argument('--query', type=str, help='Query for retriever/explainer')
    parser.add_argument('--topic', type=str, help='Topic for summarizer/quiz')
    parser.add_argument('--statement', type=str, help='Statement to verify')
    parser.add_argument('--question', type=str, help='Question for explainer')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini-2025-04-14',
                       help='Model name (default: gpt-4.1-mini-2025-04-14 for OpenAI workflow)')
    parser.add_argument('--model-type', type=str, choices=['huggingface', 'openai'],
                       default='openai', help='Model type: openai (GPT, recommended) or huggingface (limited support)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='Not used (kept for backward compatibility, models run via API)')
    parser.add_argument('--api-key', type=str,
                       help='API key (OpenAI or HuggingFace depending on model-type)')
    parser.add_argument('--demo', action='store_true',
                       help='Run a demo showing cooperative state progression')
    parser.add_argument('--orchestrate', '--full-workflow', '--run-orchestrator',
                       dest='orchestrate', action='store_true',
                       help='Run full orchestrated workflow (central orchestrator drives all agents)')
    
    args = parser.parse_args()
    
    # Orchestrated mode: run_full_workflow(topic)
    if args.orchestrate:
        _run_orchestrated(args)
        return
    
    # Demo mode
    if args.demo:
        _run_demo(args)
        return
    
    print("=" * 70)
    print("LangChain RAG Agents - Example for PDF 141")
    print("=" * 70)
    print()
    
    # Initialize agents
    try:
        kwargs = {
            "pdf_id": args.pdf_id,
            "vector_store_path": "vector_store",
            "llm_model": args.model,
            "model_type": args.model_type,
        }
        
        if args.model_type == "huggingface":
            kwargs.update({"device": args.device, "huggingface_api_key": args.api_key})
        else:
            kwargs.update({"openai_api_key": args.api_key})
        
        agents = RAGAgents(**kwargs)
        print(f"[OK] Agents initialized for PDF {args.pdf_id} with {args.model_type} model: {args.model}\n")
    except Exception as e:
        print(f"[ERROR] Error initializing agents: {e}")
        return
    
    # Run agents based on role
    # Order: retriever → duplicate_detector → curator → atomizer → quiz → verifier → coverage → packager (structure/curriculum first)
    agents_to_run = [
        ('structure_outline', agents.structure_outline, "Build document outline (section_id|title|start_slide|end_slide|weight per section).", "📐 STRUCTURE/OUTLINE AGENT"),
        ('curriculum_lo', agents.curriculum_lo, "Create learning objectives for sections.", "📋 CURRICULUM/LO AGENT"),
        ('retriever', lambda x: agents.retrieve(x), args.query or "k-means clustering", "📚 RETRIEVER AGENT"),
        ('quality_duplicate_detector', agents.quality_duplicate_detector, "Detect duplicate chunks in evidence pool (run after retriever, before curator). Input: optional threshold or 'yes'.", "🔍 QUALITY/DUPLICATE DETECTOR AGENT"),
        ('evidence_curator', agents.evidence_curator, "Dedupe, score, and tag evidence chunks.", "📎 EVIDENCE CURATOR AGENT"),
        ('atomizer', agents.atomizer, "Extract knowledge atoms from chunks.", "🧠 ATOMIZER AGENT"),
        ('quiz_generator', lambda x: agents.generate_quiz(x), args.topic or "k-means algorithm", "❓ QUIZ GENERATOR AGENT"),
        ('verifier', lambda x: agents.run_agent("verifier", x), f"Verify statement: {args.statement}" if args.statement else "Verify all candidate questions. Use get_candidate_questions to list them, then verify_all_candidates to check each against evidence and move to validated or rejected.", "✅ VERIFIER AGENT"),
        ('coverage_controller', agents.coverage_controller, "Analyze coverage gaps and steer targets.", "🎯 COVERAGE CONTROLLER AGENT"),
        ('packager_export', agents.packager_export, "Create and append quiz pack. Input: title.", "📦 PACKAGER/EXPORT AGENT"),
    ]
    for role, agent_func, default_input, title in agents_to_run:
        if args.role != role and args.role != 'all':
            continue
        print(f"\n{'=' * 70}")
        print(title)
        print(f"{'=' * 70}")
        inp = default_input if isinstance(default_input, str) else str(default_input)
        result = agent_func(inp)
        print(f"\nInput: {inp[:80]}...")
        print(f"\nResult:\n{result.get('result', result.get('output', result))}\n")
    
    # Show state summary after all agents run
    print("\n" + "=" * 70)
    print("📊 COOPERATIVE STATE SUMMARY")
    print("=" * 70)
    print(agents._get_state_summary())
    print("\n" + agents._get_coverage())
    
    # Show some examples from state
    if agents.state.evidence_pool:
        print(f"\n📦 Evidence Pool Sample (showing first 2 chunks):")
        for i, (chunk_id, chunk) in enumerate(list(agents.state.evidence_pool.items())[:2]):
            print(f"  {chunk_id}: relevance={chunk.relevance_score:.2f} quality={chunk.quality_score:.2f}, Content preview: {chunk.content[:100]}...")
    
    if agents.state.knowledge_atoms:
        print(f"\n🧠 Knowledge Atoms Sample (showing first 2 atoms):")
        for i, (atom_id, atom) in enumerate(list(agents.state.knowledge_atoms.items())[:2]):
            print(f"  {atom_id} ({atom.atom_type}): {atom.content[:100]}...")
    
    if agents.state.candidate_questions:
        print(f"\n❓ Candidate Questions ({len(agents.state.candidate_questions)}):")
        for q_id, q in list(agents.state.candidate_questions.items())[:3]:
            print(f"  {q_id}: {q.question_text[:80]}...")
    
    if agents.state.validated_questions:
        print(f"\nValidated Questions ({len(agents.state.validated_questions)}):")
        for q_id, q in list(agents.state.validated_questions.items())[:3]:
            print(f"  {q_id}: {q.question_text[:80]}...")
    
    if agents.state.rejected_questions:
        print(f"\n❌ Rejected Questions ({len(agents.state.rejected_questions)}):")
        for q_id, r in list(agents.state.rejected_questions.items())[:2]:
            print(f"  {q_id}: {r.rejection_reason}")
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)
    
    # Option to export quiz pack
    if agents.state.validated_questions:
        print("\n💾 To export quiz pack, use: agents.export_quiz_pack('My Quiz')")


if __name__ == '__main__':
    main()
