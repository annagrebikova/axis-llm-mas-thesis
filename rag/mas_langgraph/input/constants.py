"""Orchestrator thresholds and loop limits — edit these to tune workflow behavior."""

import os
from typing import Optional

# Main orchestration while-loop
MAX_ORCHESTRATION_STEPS = 80
NO_PROGRESS_ITERATIONS = 4

# Pipeline minimums
MIN_EVIDENCE_CHUNKS = 3
MIN_ATOMS = 2
MIN_LO_COUNT = 1
MIN_VALIDATED_QUESTIONS = 1
MIN_CANDIDATE_PLUS_VALIDATED = 3  # used by needs_more_questions

# Quality gates
MAX_DUPLICATION_SCORE = 0.35
MAX_REJECTION_RATE = 0.75
MIN_AVG_QUALITY_SCORE = 0.3
LOW_QUALITY_CHUNK_RATIO = 0.4

# Curator / atomizer
CURATE_DEDUPE_THRESHOLD = 0.92
ATOM_MIN_LENGTH = 10
ATOM_MAX_LENGTH = 500

# LangChain ReAct (when wiring full agents)
REACT_MAX_ITERATIONS = 8

# Packaging: allow export when validated >= this and gaps acceptable
PACKAGING_MIN_VALIDATED_WITH_GAPS = 3

# Quiz generation alternation in orchestrator (step % N)
QUIZ_GENERATION_STEP_MODULO = 6
QUIZ_GENERATION_STEP_THRESHOLD = 2

# Quiz / verify caps (RAG direct dispatch)
QUIZ_QUESTIONS_PER_ROUND = 2
VERIFY_BATCH_SIZE = 5
MAX_PENDING_CANDIDATES = 8
MAX_QUIZ_GENERATION_ROUNDS = 12

# LangGraph Step-1 loop limits
LANGGRAPH_MAX_RETRIEVE_ATTEMPTS = 10
LANGGRAPH_MAX_QUIZ_VERIFY_LOOPS = 8
LANGGRAPH_MAX_REPAIR_ITERATIONS = 2
LANGGRAPH_MAX_REVISION_ITERATIONS = 2

# Cross-CoVe verification (Step 6)
# Independent of the MCQ generator (gpt-4.1-mini-...). Override with env
# CROSS_COVE_VERIFIER_MODEL. Empty env falls back to this thesis default.
CROSS_COVE_ENABLED = True
CROSS_COVE_N_RUNS = 3
DEFAULT_CROSS_COVE_VERIFIER_MODEL = "gpt-5.4-mini-2026-03-17"
CROSS_COVE_VERIFIER_TEMPERATURE = 0.0


def _env_cross_cove_verifier_model() -> Optional[str]:
    raw = os.getenv("CROSS_COVE_VERIFIER_MODEL")
    if raw is None:
        return None
    stripped = str(raw).strip()
    return stripped or None


CROSS_COVE_VERIFIER_MODEL: Optional[str] = (
    _env_cross_cove_verifier_model() or DEFAULT_CROSS_COVE_VERIFIER_MODEL
)

# Step 8: section-level evidence retrieval
MIN_CHUNKS_PER_SECTION = 2
MAX_CHUNKS_PER_SECTION = 5
LANGGRAPH_MAX_EVIDENCE_REPAIR_ITERATIONS = 2
LANGGRAPH_MAX_BLUEPRINT_REPAIR_ITERATIONS = 2
