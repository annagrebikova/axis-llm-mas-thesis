"""Agent registry — edit roles, display names, and default orchestration inputs."""

from typing import Any, Dict, List

# Ordered list used by CLI --role all and documentation
AGENT_IDS: List[str] = [
    "structure_outline",
    "curriculum_lo",
    "retriever",
    "quality_duplicate_detector",
    "evidence_curator",
    "atomizer",
    "quiz_generator",
    "verifier",
    "coverage_controller",
    "packager_export",
]

AGENT_METADATA: Dict[str, Dict[str, Any]] = {
    "structure_outline": {
        "title": "Structure/Outline Agent",
        "description": "Build document outline (sections, slide ranges, weights).",
        "default_input": "Build document outline from the document.",
    },
    "curriculum_lo": {
        "title": "Curriculum/LO Agent",
        "description": "Create learning objectives per section.",
        "default_input": "Create learning objectives for each section with difficulty levels.",
    },
    "retriever": {
        "title": "Retriever Agent",
        "description": "Retrieve evidence chunks into the pool.",
        "default_input": "key concepts from the document",
    },
    "quality_duplicate_detector": {
        "title": "Quality/Duplicate Detector",
        "description": "Flag duplicate chunks and questions.",
        "default_input": "yes",
    },
    "evidence_curator": {
        "title": "Evidence Curator",
        "description": "Dedupe, score, and tag evidence chunks.",
        "default_input": "Curate all evidence in the pool.",
    },
    "atomizer": {
        "title": "Atomizer Agent",
        "description": "Extract knowledge atoms from chunks.",
        "default_input": "yes",
    },
    "quiz_generator": {
        "title": "Quiz Generator",
        "description": "Generate candidate questions from atoms and evidence.",
        "default_input": "topic from workflow",
    },
    "verifier": {
        "title": "Verifier Agent",
        "description": "Validate or reject candidate questions against evidence.",
        "default_input": "Verify all candidate questions.",
    },
    "coverage_controller": {
        "title": "Coverage Controller",
        "description": "Analyze coverage gaps and steer generation.",
        "default_input": "Analyze coverage gaps and suggest next targets.",
    },
    "packager_export": {
        "title": "Packager/Export Agent",
        "description": "Create and export QuizPack from validated questions.",
        "default_input": "Generated Quiz",
    },
}

# Agents invoked via dedicated methods (not generic run_agent) in the orchestrator loop
DIRECT_INVOKE_AGENTS = frozenset({
    "retriever",
    "quiz_generator",
    "verifier",
    "packager_export",
})

# Stop workflow successfully after this agent completes without error
TERMINAL_SUCCESS_AGENT = "packager_export"
