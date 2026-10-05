"""Graph builder — single authoritative MCQ LangGraph topology."""

from __future__ import annotations

from typing import Any, Optional

from .checkpointer import default_factual_checkpointer
from .runtime import WorkflowRuntime


def compile_with_checkpointer(
    graph: Any,
    checkpointer: Any = None,
    *,
    require_checkpointer: bool = False,
) -> Any:
    """Compile a LangGraph StateGraph with an optional checkpointer.

    When ``checkpointer`` is None and ``require_checkpointer`` is False, compiles
    without persistence (low-level compatibility only — not AXIS-ready).

    Factual execution via ``WorkflowExecutor`` always supplies a checkpointer.
    """
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    if require_checkpointer:
        return graph.compile(checkpointer=default_factual_checkpointer())
    return graph.compile()


def build_mcq_workflow_graph(
    checkpointer: Any = None,
    *,
    require_checkpointer: bool = False,
) -> Any:
    """Build and compile the live MCQ LangGraph workflow.

    Node implementations remain in ``orchestrator.langgraph_workflow``; this module
    owns the authoritative wiring (nodes, edges, conditional routes).

    ``context_schema`` is ``WorkflowRuntime`` — runtime services are not checkpointed.
    """
    from langgraph.graph import END, START, StateGraph

    from ..orchestrator import langgraph_workflow as lg
    from ..provenance.instrumentation import instrument_node
    from ..artifacts.resolution import enforce_artifact_resolution

    graph = StateGraph(lg.WorkflowGraphState, context_schema=WorkflowRuntime)
    for name, fn in [
        ("build_outline", lg.node_build_outline),
        ("generate_learning_objectives", lg.node_generate_learning_objectives),
        (
            "initial_retrieve_evidence_for_all_sections",
            lg.node_initial_retrieve_evidence_for_all_sections,
        ),
        ("retrieve_evidence", lg.node_retrieve_evidence),
        ("deduplicate_evidence", lg.node_deduplicate_evidence),
        ("curate_evidence", lg.node_curate_evidence),
        ("atomize_evidence", lg.node_atomize_evidence),
        ("atomize_new_evidence", lg.node_atomize_new_evidence),
        ("update_section_evidence_status", lg.node_update_section_evidence_status),
        ("repair_missing_evidence", lg.node_repair_missing_evidence),
        ("plan_mcq_blueprints", lg.node_plan_mcq_blueprints),
        (
            "generate_candidate_mcqs_from_blueprints",
            lg.node_generate_candidate_mcqs_from_blueprints,
        ),
        (
            "verify_candidate_mcqs_against_evidence",
            lg.node_verify_candidate_mcqs_against_evidence,
        ),
        (
            "cross_cove_verify_candidate_mcqs",
            lg.node_cross_cove_verify_candidate_mcqs,
        ),
        ("apply_acceptance_gate", lg.node_apply_acceptance_gate),
        ("revise_candidate_mcqs", lg.node_revise_candidate_mcqs),
        ("check_coverage", lg.node_check_coverage),
        ("generate_repair_blueprints", lg.node_generate_repair_blueprints),
        ("package_final_quiz", lg.node_package_final_quiz),
        ("verify_candidates", lg.node_verify_candidates),
        ("package_or_end", lg.node_package_or_end),
        ("end_with_error", lg.node_end_with_error),
    ]:
        # Outer resolution → inner provenance → business node
        graph.add_node(
            name,
            enforce_artifact_resolution(name, instrument_node(name, fn)),
        )

    graph.add_edge(START, "build_outline")
    graph.add_edge("build_outline", "generate_learning_objectives")
    graph.add_edge(
        "generate_learning_objectives",
        "initial_retrieve_evidence_for_all_sections",
    )
    graph.add_edge(
        "initial_retrieve_evidence_for_all_sections", "deduplicate_evidence"
    )
    graph.add_edge("deduplicate_evidence", "curate_evidence")
    graph.add_conditional_edges(
        "curate_evidence",
        lg.route_after_curate,
        {
            "atomize_evidence": "atomize_evidence",
            "atomize_new_evidence": "atomize_new_evidence",
        },
    )
    graph.add_edge("atomize_evidence", "update_section_evidence_status")
    graph.add_edge("atomize_new_evidence", "update_section_evidence_status")
    graph.add_conditional_edges(
        "update_section_evidence_status",
        lg.route_after_update_section_status,
        {
            "plan_mcq_blueprints": "plan_mcq_blueprints",
            "check_coverage": "check_coverage",
        },
    )
    graph.add_edge("plan_mcq_blueprints", "generate_candidate_mcqs_from_blueprints")
    graph.add_edge(
        "generate_candidate_mcqs_from_blueprints",
        "verify_candidate_mcqs_against_evidence",
    )
    graph.add_edge(
        "verify_candidate_mcqs_against_evidence",
        "cross_cove_verify_candidate_mcqs",
    )
    graph.add_edge(
        "cross_cove_verify_candidate_mcqs", "apply_acceptance_gate"
    )
    graph.add_conditional_edges(
        "apply_acceptance_gate",
        lg.route_after_acceptance_gate,
        {
            "revise_candidate_mcqs": "revise_candidate_mcqs",
            "check_coverage": "check_coverage",
        },
    )
    graph.add_edge(
        "revise_candidate_mcqs", "verify_candidate_mcqs_against_evidence"
    )
    graph.add_conditional_edges(
        "check_coverage",
        lg.route_after_coverage,
        {
            "generate_candidate_mcqs_from_blueprints":
                "generate_candidate_mcqs_from_blueprints",
            "repair_missing_evidence": "repair_missing_evidence",
            "generate_repair_blueprints": "generate_repair_blueprints",
            "package_final_quiz": "package_final_quiz",
        },
    )
    graph.add_edge("repair_missing_evidence", "deduplicate_evidence")
    graph.add_edge(
        "generate_repair_blueprints", "generate_candidate_mcqs_from_blueprints"
    )
    graph.add_edge("package_final_quiz", END)
    graph.add_edge("end_with_error", END)

    return compile_with_checkpointer(
        graph,
        checkpointer=checkpointer,
        require_checkpointer=require_checkpointer,
    )


# Backward-compatible alias used by earlier migration steps
build_existing_mcq_graph = build_mcq_workflow_graph
