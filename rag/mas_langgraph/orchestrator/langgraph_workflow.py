"""LangGraph orchestration — Step 8: full-outline evidence coverage and repair retrieval.

Reuses CooperativeState, AgentRunner, NeedsEvaluator, and direct-dispatch RAG helpers.

Included:
  - Section-level evidence retrieval for all outline sections
  - Evidence repair loop before blueprint repair
  - MCQBlueprint planning and blueprint-driven CandidateMCQ generation
  - PDF evidence grounding verification (not IWF)
  - Cross-CoVe independent verifier majority vote
  - Acceptance gating, revision loop, and coverage repair

Not yet included:
  - IWF checker (external to MAS)
  - Parallel revision, advanced scoring/ranking
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Set, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from ..agents.runner import AgentRunner
from ..artifacts.channel_state import empty_artifact_channels
from ..provenance.channel_state import empty_provenance_channels
from ..core.entities import WorkflowStepLog
from ..core.runtime import WorkflowRuntime
from ..cases.mcq.cooperative_state import CooperativeState
from ..input.constants import (
    LANGGRAPH_MAX_BLUEPRINT_REPAIR_ITERATIONS,
    LANGGRAPH_MAX_EVIDENCE_REPAIR_ITERATIONS,
    LANGGRAPH_MAX_QUIZ_VERIFY_LOOPS,
    LANGGRAPH_MAX_REPAIR_ITERATIONS,
    LANGGRAPH_MAX_REVISION_ITERATIONS,
    LANGGRAPH_MAX_RETRIEVE_ATTEMPTS,
    VERIFY_BATCH_SIZE,
)
from ..input.orchestrator_config import DEFAULT_ORCHESTRATOR_CONFIG, OrchestratorConfig
from ..domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from ..domains.quiz.generation import (
    generate_candidate_from_blueprint,
    log_ok,
    log_warn,
    register_candidate_mcq,
)
from ..domains.quiz.cross_cove import (
    build_cross_cove_verifier_llm,
    llm_model_identity,
    log_cross_cove_model_choice,
    print_cross_cove_model_config,
    resolve_cross_cove_verifier_model,
    resolve_cross_cove_verifier_temperature,
    run_cross_cove_for_all_candidates,
)
from ..domains.quiz.coverage import (
    apply_acceptance_gate_to_all,
    compute_mcq_coverage,
    create_repair_blueprints,
    find_coverage_gaps,
    store_coverage_summary_on_state,
    summarize_acceptance_decisions,
)
from ..domains.quiz.packaging import package_final_quiz
from ..domains.quiz.question_limit import (
    COVERAGE_CAP_WARNING,
    build_generation_limit_report,
    effective_max_final_questions,
    final_question_cap_reached,
    generation_batch_size,
    record_coverage_cap_warning,
    remaining_final_question_capacity,
    remaining_repairable_blueprints,
    slice_to_limit,
)
from ..domains.quiz.evidence_repair import (
    repair_missing_evidence_for_sections,
    retrieve_evidence_for_all_sections,
)
from ..domains.quiz.section_evidence import log_section_evidence_summary
from ..domains.quiz.revision import (
    count_candidates_needing_revision,
    revise_all_needed_candidates,
)
from ..domains.quiz.state_integration import ensure_step2_mcq_state
from ..domains.quiz.verification import verify_all_candidate_mcqs
from ..integrations.rag_direct_tools import (
    canonical_chunk_ids,
    direct_atomize_chunks,
    direct_atomize_new_chunks,
    direct_build_outline,
    direct_curate_evidence,
    direct_curriculum_lo,
    direct_detect_duplicates,
    direct_generate_quiz,
    direct_verify_candidates,
    retrieve_with_dedupe,
)
from .material_context import MaterialContext
from .needs_base import BaseNeedsEvaluator
from ..core.state_cloning import prepare_state_transition
from .artifact_wiring import commit_artifact_channels
from .state_binding import bind_rag_agents

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class WorkflowGraphState(TypedDict):
    """Checkpointed workflow data — no runtime service handles."""

    cooperative_state: CooperativeState
    run_id: str
    branch_id: str
    branch_type: str
    parent_branch_id: str
    origin_checkpoint_alias: str
    intervention_id: Optional[str]
    artifact_catalog: Dict[str, Any]
    available_artifact_refs: Set[Any]
    blocked_artifact_refs: Set[Any]
    artifact_substitutions: Dict[Any, Any]
    artifact_derivations: List[Any]
    intervention_events: List[Any]
    occurrence_records: List[Any]
    produced_by: List[Any]
    run_provenance_relations: List[Any]
    provenance_violations: List[str]
    provenance_unresolved_producers: List[str]
    provenance_node_counts: Dict[str, int]
    provenance_execution_index: int
    searched_queries: Set[str]
    last_agent: Optional[str]
    step_logs: Annotated[List[Dict[str, Any]], operator.add]
    error: Optional[str]
    retrieve_attempts: int
    quiz_verify_loops: int
    repair_iterations: int
    evidence_repair_iterations: int
    max_evidence_repair_iterations: int
    blueprint_repair_iterations: int
    max_blueprint_repair_iterations: int
    coverage_gaps: List[Dict[str, Any]]
    revision_iterations: int
    max_revision_iterations: int
    pending_evidence_repair_atomize: bool
    return_to_coverage_check: bool


RouteAfterRetrieve = Literal["deduplicate_evidence", "retrieve_evidence", "end_with_error"]
RouteAfterVerify = Literal[
    "generate_candidate_mcqs_from_blueprints", "verify_candidates", "package_or_end"
]
RouteAfterCoverage = Literal[
    "generate_candidate_mcqs_from_blueprints",
    "repair_missing_evidence",
    "generate_repair_blueprints",
    "package_final_quiz",
]
RouteAfterCurate = Literal["atomize_evidence", "atomize_new_evidence"]
RouteAfterUpdateSectionStatus = Literal["plan_mcq_blueprints", "check_coverage"]
RouteAfterAcceptance = Literal["revise_candidate_mcqs", "check_coverage"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(state: WorkflowGraphState) -> Dict[str, Any]:
    return state["cooperative_state"].snapshot_counts()


def _step_entry(
    state: WorkflowGraphState,
    agent_id: str,
    result: Dict[str, Any],
    *,
    cooperative_state: CooperativeState,
    input_text: Optional[str] = None,
    reason: str = "",
    state_before: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build partial state update with one log entry."""
    from copy import deepcopy

    summary = str(result.get("result", result))[:300]
    after = deepcopy(cooperative_state.snapshot_counts())
    before = deepcopy(state_before) if state_before is not None else deepcopy(after)
    log = {
        "step": len(state.get("step_logs", [])) + 1,
        "agent_id": agent_id,
        "reason": reason or f"LangGraph node: {agent_id}",
        "input_text": input_text,
        "result_summary": summary,
        "state_before": before,
        "state_after": after,
        "error": result.get("error"),
    }
    return {
        "last_agent": agent_id,
        "step_logs": [log],
        "error": result.get("error") or state.get("error"),
    }


def _orchestrator_config(runtime: Runtime[WorkflowRuntime]) -> OrchestratorConfig:
    return runtime.context.orchestrator_config or DEFAULT_ORCHESTRATOR_CONFIG


def _dispatch(
    state: WorkflowGraphState,
    runtime: Runtime[WorkflowRuntime],
    agent_id: str,
    *,
    cooperative_state: CooperativeState,
    suggested_input: Optional[str] = None,
    use_direct: bool = True,
) -> Dict[str, Any]:
    """Call direct RAG helper when rag_agents is set, else AgentRunner.invoke."""
    ctx = runtime.context
    runner = ctx.agent_runner
    bound_rag = bind_rag_agents(ctx.rag_agents, cooperative_state)

    if use_direct and bound_rag is not None:
        if agent_id == "structure_outline":
            return direct_build_outline(bound_rag)
        if agent_id == "curriculum_lo":
            return direct_curriculum_lo(bound_rag)
        if agent_id == "quality_duplicate_detector":
            return direct_detect_duplicates(bound_rag)
        if agent_id == "evidence_curator":
            return direct_curate_evidence(bound_rag)
        if agent_id == "atomizer":
            return direct_atomize_chunks(bound_rag)
        if agent_id == "quiz_generator":
            mat = MaterialContext(
                cooperative_state,
                getattr(runner, "coverage_plan", None),
                pdf_id=getattr(runner, "pdf_id", None),
            )
            spec = suggested_input or mat.quiz_target()
            remaining = remaining_final_question_capacity(
                cooperative_state, _orchestrator_config(runtime)
            )
            n_questions = generation_batch_size(remaining)
            return direct_generate_quiz(
                bound_rag, spec, num_questions=n_questions,
            )
        if agent_id == "verifier":
            return direct_verify_candidates(bound_rag, batch_size=VERIFY_BATCH_SIZE)

    try:
        return runner.invoke(
            agent_id,
            suggested_input or "",
            cooperative_state=cooperative_state,
        )
    except TypeError:
        return runner.invoke(agent_id, suggested_input or "")


def _material(
    cooperative_state: CooperativeState,
    runtime: Runtime[WorkflowRuntime],
) -> MaterialContext:
    runner = runtime.context.agent_runner
    return MaterialContext(
        cooperative_state,
        getattr(runner, "coverage_plan", None),
        pdf_id=getattr(runner, "pdf_id", None),
    )


def _graph_searched_queries(state: WorkflowGraphState) -> Set[str]:
    return set(state.get("searched_queries") or set())


def _with_cooperative_state(
    update: Dict[str, Any],
    working_cs: CooperativeState,
    *,
    input_state: WorkflowGraphState,
    artifact_stage: Optional[str] = None,
) -> Dict[str, Any]:
    update["cooperative_state"] = working_cs
    update.update(commit_artifact_channels(input_state, working_cs, artifact_stage))
    return update


# ---------------------------------------------------------------------------
# Graph nodes (deterministic — no ReAct routing)
# ---------------------------------------------------------------------------

def node_build_outline(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    result = _dispatch(state, runtime, "structure_outline", cooperative_state=working_cs)
    update = _step_entry(
        state, "structure_outline", result,
        cooperative_state=working_cs, state_before=state_before,
    )
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="outline",
    )


def node_generate_learning_objectives(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    result = _dispatch(state, runtime, "curriculum_lo", cooperative_state=working_cs)
    update = _step_entry(
        state, "curriculum_lo", result,
        cooperative_state=working_cs, state_before=state_before,
    )
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="learning_objectives",
    )


def node_initial_retrieve_evidence_for_all_sections(
    state: WorkflowGraphState,
    runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    bound_rag = bind_rag_agents(runtime.context.rag_agents, working_cs)
    searched = _graph_searched_queries(state)

    result_data = retrieve_evidence_for_all_sections(working_cs, bound_rag, searched)
    searched = set(result_data.get("searched_queries", searched))

    working_cs.update_section_evidence_status()
    log_section_evidence_summary(working_cs)

    chunks_added = result_data.get("chunks_added", 0)
    with_evidence = len(working_cs.get_sections_with_sufficient_evidence())
    total = len(working_cs.doc_outline)
    result = {
        "role": "retriever",
        "result": (
            f"Initial section retrieval added {chunks_added} chunk(s); "
            f"covered {with_evidence}/{total} sections."
        ),
        "chunks_added": chunks_added,
        "sections_covered": with_evidence,
        "sections_total": total,
    }
    update = _step_entry(
        state,
        "retriever",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Retrieve PDF evidence for every outline section",
    )
    update["searched_queries"] = searched
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="evidence",
    )


def node_update_section_evidence_status(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    working_cs.update_section_evidence_status()
    with_evidence = len(working_cs.get_sections_with_sufficient_evidence())
    total = len(working_cs.doc_outline)
    missing = working_cs.get_sections_without_evidence()
    result = {
        "role": "section_evidence_tracker",
        "result": f"Evidence status: {with_evidence}/{total} sections sufficient.",
        "sections_with_evidence": with_evidence,
        "sections_missing": len(missing),
    }
    update = _step_entry(
        state,
        "section_evidence_tracker",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Update per-section evidence and atom coverage status",
    )
    return _with_cooperative_state(update, working_cs, input_state=state)


def node_repair_missing_evidence(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    bound_rag = bind_rag_agents(runtime.context.rag_agents, working_cs)
    searched = _graph_searched_queries(state)
    evidence_repair_iterations = state.get("evidence_repair_iterations", 0) + 1
    max_evidence = state.get(
        "max_evidence_repair_iterations", LANGGRAPH_MAX_EVIDENCE_REPAIR_ITERATIONS,
    )

    log_ok(f"Evidence repair iteration {evidence_repair_iterations}/{max_evidence}")
    repair_result = repair_missing_evidence_for_sections(working_cs, bound_rag, searched)
    searched = set(repair_result.get("searched_queries", searched))

    working_cs.update_section_evidence_status()
    with_evidence = len(working_cs.get_sections_with_sufficient_evidence())
    total = len(working_cs.doc_outline)
    recovered = repair_result.get("sections_recovered", 0)
    chunks_added = repair_result.get("chunks_added", 0)
    if chunks_added:
        log_ok(f"Added evidence for {recovered} additional section(s)")
    log_ok(f"Evidence coverage: {with_evidence}/{total} sections")

    result = {
        "role": "evidence_repair",
        "result": (
            f"Repair iteration {evidence_repair_iterations}: "
            f"added {chunks_added} chunk(s) for {recovered} section(s)."
        ),
        **repair_result,
        "evidence_repair_iterations": evidence_repair_iterations,
    }
    update = _step_entry(
        state,
        "evidence_repair",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Repair retrieval for sections still missing PDF evidence",
    )
    update["searched_queries"] = searched
    update["evidence_repair_iterations"] = evidence_repair_iterations
    update["pending_evidence_repair_atomize"] = True
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="evidence",
    )


def node_atomize_new_evidence(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    bound_rag = bind_rag_agents(runtime.context.rag_agents, working_cs)
    if bound_rag is not None:
        atoms_before = len(working_cs.knowledge_atoms)
        result = direct_atomize_new_chunks(bound_rag)
        count = result.get("chunk_count", 0)
        if count:
            atoms_after = len(working_cs.knowledge_atoms)
            log_ok(f"Atomized {count} new chunk(s) into {atoms_after - atoms_before} atom(s)")
    else:
        result = {"role": "atomizer", "result": "No RAG agents for atomization.", "chunk_count": 0}
    update = _step_entry(
        state,
        "atomizer",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Atomize newly retrieved evidence chunks only",
    )
    update["return_to_coverage_check"] = True
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="atoms",
    )


def node_retrieve_evidence(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    runner = runtime.context.agent_runner
    bound_rag = bind_rag_agents(runtime.context.rag_agents, working_cs)
    mat = _material(working_cs, runtime)
    searched = _graph_searched_queries(state)
    query = mat.next_retrieval_query(searched, state.get("retrieve_attempts", 0))
    if not query:
        query = mat.retrieval_query(state.get("retrieve_attempts", 0))

    if bound_rag is not None:
        result, new_searched = retrieve_with_dedupe(bound_rag, query, searched)
        searched = set(new_searched)
    else:
        result = runner.retrieve(
            query,
            cooperative_state=working_cs,
            searched_queries=searched,
        )
        if "searched_queries" in result:
            searched = set(result.pop("searched_queries"))

    update = _step_entry(
        state, "retriever", result,
        cooperative_state=working_cs,
        state_before=state_before,
        input_text=query,
    )
    update["searched_queries"] = searched
    update["retrieve_attempts"] = state.get("retrieve_attempts", 0) + 1
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="evidence",
    )


def node_deduplicate_evidence(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    result = _dispatch(
        state, runtime, "quality_duplicate_detector", cooperative_state=working_cs,
    )
    update = _step_entry(
        state, "quality_duplicate_detector", result,
        cooperative_state=working_cs, state_before=state_before,
    )
    return _with_cooperative_state(update, working_cs, input_state=state)


def node_curate_evidence(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    result = _dispatch(state, runtime, "evidence_curator", cooperative_state=working_cs)
    update = _step_entry(
        state, "evidence_curator", result,
        cooperative_state=working_cs, state_before=state_before,
    )
    return _with_cooperative_state(update, working_cs, input_state=state)


def node_atomize_evidence(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    bound_rag = bind_rag_agents(runtime.context.rag_agents, working_cs)
    if bound_rag is not None:
        chunk_ids = canonical_chunk_ids(working_cs)
        if not chunk_ids:
            result = {
                "role": "atomizer",
                "result": "No canonical chunks to atomize.",
                "error": "no_chunks",
            }
        else:
            out = bound_rag.atomize_chunks(chunk_ids)
            result = {"role": "atomizer", "result": out, "chunk_count": len(chunk_ids)}
            log_ok(
                f"Atomized {len(chunk_ids)} chunk(s) into "
                f"{len(working_cs.knowledge_atoms)} atom(s)"
            )
    else:
        result = _dispatch(state, runtime, "atomizer", cooperative_state=working_cs)
    update = _step_entry(
        state, "atomizer", result,
        cooperative_state=working_cs, state_before=state_before,
    )
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="atoms",
    )


def node_plan_mcq_blueprints(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    ensure_step2_mcq_state(working_cs)

    try:
        blueprints = create_mcq_blueprints_from_coverage(working_cs)
    except Exception as exc:
        blueprints = []
        result = {
            "role": "mcq_blueprint_planner",
            "result": f"Blueprint planning failed: {exc}",
            "blueprints_created": 0,
            "warning": str(exc),
            "error": str(exc),
        }
        update = _step_entry(
            state,
            "mcq_blueprint_planner",
            result,
            cooperative_state=working_cs,
            state_before=state_before,
            reason="Plan literature-grounded MCQ blueprints from sections, LOs, atoms, and evidence",
        )
        return _with_cooperative_state(
            update, working_cs, input_state=state, artifact_stage="blueprints",
        )

    working_cs.store_mcq_blueprints(blueprints)
    with_evidence = len(working_cs.get_sections_with_sufficient_evidence())
    log_ok(f"Created {len(blueprints)} MCQ blueprint(s) with sufficient evidence")
    log_ok(f"Sections with evidence: {with_evidence}/{len(working_cs.doc_outline)}")
    result = {
        "role": "mcq_blueprint_planner",
        "result": f"Planned {len(blueprints)} MCQ blueprint(s).",
        "blueprints_created": len(blueprints),
    }
    if not blueprints:
        result["warning"] = "No MCQ blueprints could be planned from current coverage."
    update = _step_entry(
        state,
        "mcq_blueprint_planner",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Plan literature-grounded MCQ blueprints from sections, LOs, atoms, and evidence",
    )
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="blueprints",
    )


def node_generate_candidate_mcqs_from_blueprints(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    ensure_step2_mcq_state(working_cs)
    rag = runtime.context.rag_agents
    llm = getattr(rag, "llm", None) if rag is not None else None

    planned = working_cs.planned_blueprints()
    config = _orchestrator_config(runtime)
    remaining = remaining_final_question_capacity(working_cs, config)
    batch_size = generation_batch_size(remaining)
    if remaining is not None and remaining <= 0:
        result = {
            "role": "quiz_generator",
            "result": "Final question cap reached; skipping further generation.",
            "generated_q_ids": [],
            "warning": "final_question_cap_reached",
        }
        update = _step_entry(
            state, "quiz_generator", result,
            cooperative_state=working_cs, state_before=state_before,
        )
        return _with_cooperative_state(
            update, working_cs, input_state=state, artifact_stage="candidates",
        )
    if not planned:
        result = {
            "role": "quiz_generator",
            "result": "No planned MCQ blueprints available for generation.",
            "generated_q_ids": [],
            "warning": "no_planned_blueprints",
        }
        update = _step_entry(
            state, "quiz_generator", result,
            cooperative_state=working_cs, state_before=state_before,
        )
        return _with_cooperative_state(
            update, working_cs, input_state=state, artifact_stage="candidates",
        )

    batch = planned[:batch_size]
    generated_ids: List[str] = []
    flagged_ids: List[str] = []
    errors: List[str] = []

    for blueprint in batch:
        try:
            mcq = generate_candidate_from_blueprint(blueprint, working_cs, llm=llm)
            register_candidate_mcq(working_cs, mcq, blueprint)
            generated_ids.append(mcq.q_id)
            if mcq.flags:
                flagged_ids.append(mcq.q_id)
        except Exception as exc:
            # NOOP ledger misses must fail closed — do not mark blueprints failed
            # and continue into a divergent repair/generation path.
            from ..replay.llm_determinism import WorkflowLLMReplayMissError

            if isinstance(exc, WorkflowLLMReplayMissError):
                raise
            # Leave planned forever would infinite-loop via route_after_coverage.
            # Do not mark as successfully generated; use an explicit failure status.
            blueprint.status = "failed"
            working_cs.mcq_blueprints[blueprint.blueprint_id] = blueprint
            errors.append(f"{blueprint.blueprint_id}: {exc}")
            log_warn(f"Failed blueprint {blueprint.blueprint_id}: {exc}")

    log_ok(f"Stored {len(generated_ids)} CandidateMCQ(s)")
    if flagged_ids:
        log_warn(f"{len(flagged_ids)} CandidateMCQ(s) have structural flags")

    result = {
        "role": "quiz_generator",
        "result": (
            f"Generated {len(generated_ids)} CandidateMCQ(s) from {len(batch)} blueprint(s)."
        ),
        "generated_q_ids": generated_ids,
        "flagged_q_ids": flagged_ids,
        "blueprints_used": [bp.blueprint_id for bp in batch],
    }
    if errors:
        result["errors"] = errors

    runner = runtime.context.agent_runner
    rounds = int(getattr(working_cs, "quiz_generation_rounds", 0) or 0) + 1
    working_cs.quiz_generation_rounds = rounds
    # Keep legacy runtime mirror for older NeedsEvaluator paths; checkpointed
    # CooperativeState.quiz_generation_rounds is authoritative for replay.
    if runner is not None:
        runner._quiz_generation_rounds = rounds  # type: ignore[attr-defined]
    update = _step_entry(
        state,
        "quiz_generator",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        input_text=f"blueprints:{','.join(bp.blueprint_id for bp in batch)}",
        reason="Generate CandidateMCQs from MCQBlueprint context (PDF-grounded)",
    )
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="candidates",
    )


def node_verify_candidate_mcqs_against_evidence(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    ensure_step2_mcq_state(working_cs)
    rag = runtime.context.rag_agents
    llm = getattr(rag, "llm", None) if rag is not None else None

    reports = verify_all_candidate_mcqs(working_cs, llm=llm)
    supported = [r for r in reports if r.supported_by_pdf]
    unsupported = [r for r in reports if not r.supported_by_pdf]
    missing_evidence = [
        r for r in reports if "missing_evidence" in r.required_fixes
    ]

    log_ok(f"Evidence verification checked {len(reports)} candidate(s)")
    log_ok(f"Evidence-supported candidates: {len(supported)}")
    if unsupported:
        log_warn(f"Evidence-unsupported candidates: {len(unsupported)}")
    for report in unsupported:
        if report.unsupported_claims:
            log_warn(
                f"Candidate {report.q_id} unsupported claims: {report.unsupported_claims[:3]}"
            )
    if missing_evidence:
        log_warn(f"Candidates with missing evidence: {len(missing_evidence)}")
    log_ok(f"Stored {len(reports)} EvidenceCheckReport(s)")

    result = {
        "role": "evidence_verifier",
        "result": (
            f"Checked {len(reports)} candidate(s): "
            f"{len(supported)} supported, {len(unsupported)} unsupported."
        ),
        "checked": len(reports),
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "missing_evidence_count": len(missing_evidence),
    }
    update = _step_entry(
        state,
        "evidence_verifier",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Verify CandidateMCQs against attached PDF evidence (grounding only)",
    )
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="evidence_reports",
    )


def node_cross_cove_verify_candidate_mcqs(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    ensure_step2_mcq_state(working_cs)
    config = _orchestrator_config(runtime)

    if not getattr(config, "cross_cove_enabled", True):
        log_warn("Cross-CoVe verification disabled by config")
        result = {
            "role": "cross_cove_verifier",
            "result": "Cross-CoVe skipped (disabled).",
            "checked": 0,
            "passed_count": 0,
            "failed_count": 0,
            "no_majority_count": 0,
        }
        update = _step_entry(
            state,
            "cross_cove_verifier",
            result,
            cooperative_state=working_cs,
            state_before=state_before,
            reason="Cross-CoVe verification disabled",
        )
        return _with_cooperative_state(update, working_cs, input_state=state)

    rag = runtime.context.rag_agents
    generator_llm = getattr(rag, "llm", None) if rag is not None else None
    if generator_llm is None:
        generator_llm = getattr(runtime.context, "llm", None)
    generator_model = llm_model_identity(generator_llm)
    if not generator_model and rag is not None:
        generator_model = (
            getattr(rag, "model_name", None)
            or getattr(rag, "llm_model", None)
            or getattr(getattr(rag, "config", None), "llm_model", None)
        )
        generator_model = str(generator_model) if generator_model else None

    verifier_model = resolve_cross_cove_verifier_model(
        config, generator_model=generator_model
    )
    verifier_llm = getattr(runtime.context, "cross_cove_verifier_llm", None)
    if verifier_llm is None:
        if (
            verifier_model
            and generator_model
            and str(verifier_model) != str(generator_model)
            and generator_llm is not None
        ):
            raise RuntimeError(
                "cross_cove_verifier_model is set to a different model than the "
                "MCQ generator, but WorkflowRuntime.cross_cove_verifier_llm was "
                "not installed. Refusing to silently reuse the generator LLM."
            )
        verifier_llm = generator_llm

    log_cross_cove_model_choice(verifier_model, generator_model)
    n_runs = getattr(config, "cross_cove_n_runs", 3)
    actual_verifier_model = llm_model_identity(verifier_llm) or verifier_model or generator_model

    reports = run_cross_cove_for_all_candidates(
        working_cs,
        verifier_llm=verifier_llm,
        n_runs=n_runs,
        verifier_model_name=actual_verifier_model,
    )
    passed = [r for r in reports if r.passed]
    failed = [r for r in reports if not r.passed]
    no_majority = [r for r in reports if not r.majority_answer]
    avg_confidence = (
        sum(r.confidence for r in reports) / len(reports) if reports else 0.0
    )

    log_ok(f"Cross-CoVe checked {len(reports)} candidate(s)")
    log_ok(f"Cross-CoVe passed: {len(passed)}")
    if failed:
        log_warn(f"Cross-CoVe failed: {len(failed)}")
    if no_majority:
        log_warn(f"Cross-CoVe no majority: {len(no_majority)}")
    if reports:
        log_ok(f"Average Cross-CoVe confidence: {avg_confidence:.2f}")

    result = {
        "role": "cross_cove_verifier",
        "result": (
            f"Cross-CoVe checked {len(reports)}: "
            f"{len(passed)} passed, {len(failed)} failed."
        ),
        "checked": len(reports),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "no_majority_count": len(no_majority),
        "avg_confidence": avg_confidence,
    }
    update = _step_entry(
        state,
        "cross_cove_verifier",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Cross-CoVe independent verifier majority vote on MCQ answers",
    )
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="cross_cove_reports",
    )


def node_apply_acceptance_gate(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    ensure_step2_mcq_state(working_cs)
    config = _orchestrator_config(runtime)
    cross_cove_required = getattr(config, "cross_cove_enabled", True)
    max_revisions = getattr(config, "max_revision_iterations", LANGGRAPH_MAX_REVISION_ITERATIONS)

    criteria_meta = None
    criteria_order = None
    ctx = getattr(runtime, "context", None)
    case_def = getattr(ctx, "case_definition", None) if ctx is not None else None
    if case_def is not None and hasattr(case_def, "load_outcome_evaluator"):
        evaluator = case_def.load_outcome_evaluator()
        criteria_meta = getattr(evaluator, "criteria_meta", None)
        criteria_order = getattr(evaluator, "criteria_order", None)

    decisions = apply_acceptance_gate_to_all(
        working_cs,
        cross_cove_required=cross_cove_required,
        max_revisions=max_revisions,
        criteria_meta=criteria_meta,
        criteria_order=criteria_order,
        max_final_questions=effective_max_final_questions(config),
    )
    summary = summarize_acceptance_decisions(decisions)

    log_ok(f"Acceptance gate checked {len(decisions)} candidate(s)")
    log_ok(f"Accepted: {summary['accepted']}")
    if summary["needs_revision"]:
        log_warn(f"Needs revision: {summary['needs_revision']}")
    if summary["rejected"]:
        log_warn(f"Rejected: {summary['rejected']}")
    if summary["common_issue"]:
        log_warn(f"Common issue: {summary['common_issue']}")

    result = {
        "role": "acceptance_gate",
        "result": (
            f"Gate checked {len(decisions)}: "
            f"{summary['accepted']} accepted, "
            f"{summary['needs_revision']} needs revision, "
            f"{summary['rejected']} rejected."
        ),
        **summary,
    }
    update = _step_entry(
        state,
        "acceptance_gate",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Acceptance gate via OutcomeAssessment Gamma/delta (SupervisorDecision)",
    )
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="supervisor_decisions",
    )


def node_revise_candidate_mcqs(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    ensure_step2_mcq_state(working_cs)
    config = _orchestrator_config(runtime)
    rag = runtime.context.rag_agents
    llm = getattr(rag, "llm", None) if rag is not None else None
    max_revisions = getattr(config, "max_revision_iterations", LANGGRAPH_MAX_REVISION_ITERATIONS)
    revision_iterations = state.get("revision_iterations", 0) + 1

    before_rejected = sum(
        1 for c in working_cs.candidate_mcqs.values() if c.status == "rejected"
    )
    revised = revise_all_needed_candidates(
        working_cs, llm=llm, max_revisions=max_revisions,
    )
    after_rejected = sum(
        1 for c in working_cs.candidate_mcqs.values() if c.status == "rejected"
    )
    max_reached = after_rejected - before_rejected

    log_ok(f"Revision loop iteration {revision_iterations}/{max_revisions}")
    log_ok(f"Revised {len(revised)} CandidateMCQ(s)")
    if max_reached > 0:
        log_warn(f"{max_reached} candidate(s) reached max revisions and were rejected")

    result = {
        "role": "mcq_reviser",
        "result": f"Revised {len(revised)} candidate(s) on iteration {revision_iterations}.",
        "revised_count": len(revised),
        "max_revisions_reached": max_reached,
        "revision_iterations": revision_iterations,
    }
    update = _step_entry(
        state,
        "mcq_reviser",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Revise CandidateMCQs failing evidence, Cross-CoVe, or acceptance checks",
    )
    update["revision_iterations"] = revision_iterations
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="candidates",
    )


def node_check_coverage(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    ensure_step2_mcq_state(working_cs)

    coverage = compute_mcq_coverage(working_cs)
    gaps = find_coverage_gaps(working_cs)
    store_coverage_summary_on_state(working_cs, coverage, gaps)
    config = _orchestrator_config(runtime)
    cap_warning = record_coverage_cap_warning(working_cs, gaps=gaps, config=config)

    sections = {s.section_id: s.title for s in working_cs.doc_outline}
    uncovered_labels = [
        f"{sid} {sections.get(sid, sid)}"
        for sid in coverage["uncovered_sections"][:6]
    ]

    log_ok("Coverage check complete")
    log_ok(
        f"Covered sections: {coverage['coverage_sections_covered']} / "
        f"{coverage['coverage_sections_total']}"
    )
    supported = coverage.get("coverage_sections_supported_total", 0)
    if supported:
        log_ok(
            f"Final coverage: {coverage['coverage_sections_covered']}/"
            f"{supported} supported sections"
        )
    if uncovered_labels:
        log_warn(f"Uncovered sections: {', '.join(uncovered_labels)}")
    if coverage["uncovered_learning_objectives"]:
        log_warn(
            f"Uncovered learning objectives: {len(coverage['uncovered_learning_objectives'])}"
        )
    if cap_warning == COVERAGE_CAP_WARNING:
        log_warn(
            "Coverage remains incomplete because the final-question cap was reached."
        )

    result = {
        "role": "coverage_controller",
        "result": (
            f"Coverage {coverage['coverage_sections_covered']}/"
            f"{coverage['coverage_sections_total']} sections; "
            f"{len(gaps)} gap(s)."
        ),
        "coverage": {
            k: v for k, v in coverage.items() if k != "requirements"
        },
        "gaps": gaps,
        "gap_count": len(gaps),
        "question_cap_reached": bool(final_question_cap_reached(working_cs, config)),
        "coverage_cap_warning": cap_warning,
    }
    update = _step_entry(
        state,
        "coverage_controller",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Compute accepted MCQ coverage and detect section/LO gaps",
    )
    update["coverage_gaps"] = gaps
    update["return_to_coverage_check"] = False
    update["pending_evidence_repair_atomize"] = False
    return _with_cooperative_state(update, working_cs, input_state=state)


def node_generate_repair_blueprints(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    ensure_step2_mcq_state(working_cs)

    gaps = state.get("coverage_gaps") or []
    config = _orchestrator_config(runtime)
    remaining = remaining_final_question_capacity(working_cs, config)
    if remaining is not None and remaining <= 0:
        log_warn("Final question cap reached; skipping coverage-repair blueprints.")
        blueprints = []
    else:
        blueprints = create_repair_blueprints(working_cs, gaps)
        blueprints = slice_to_limit(blueprints, remaining)
    if blueprints:
        working_cs.store_mcq_blueprints(blueprints)
        log_ok(f"Created {len(blueprints)} repair blueprint(s)")
    else:
        log_warn("No repair blueprints created from current gaps")

    repair_iterations = state.get("blueprint_repair_iterations", state.get("repair_iterations", 0)) + 1
    if gaps and not blueprints:
        log_warn(
            f"No repair blueprints created — "
            f"{len(gaps)} gap(s) may lack evidence/atoms or already have planned blueprints"
        )
    result = {
        "role": "repair_blueprint_planner",
        "result": f"Created {len(blueprints)} repair blueprint(s).",
        "repair_blueprints_created": len(blueprints),
        "blueprint_repair_iterations": repair_iterations,
        "repair_iterations": repair_iterations,
    }
    update = _step_entry(
        state,
        "repair_blueprint_planner",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        reason="Plan MCQBlueprints for uncovered sections/LOs with evidence",
    )
    update["blueprint_repair_iterations"] = repair_iterations
    update["repair_iterations"] = repair_iterations
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="blueprints",
    )


def node_package_final_quiz(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    ensure_step2_mcq_state(working_cs)
    mat = _material(working_cs, runtime)
    title = mat.pack_title()
    config = _orchestrator_config(runtime)
    pack = package_final_quiz(
        working_cs,
        title=title,
        max_final_questions=effective_max_final_questions(config),
    )
    diagnostics = build_generation_limit_report(
        working_cs,
        config=config,
        graph_state=state,
        packaged_count=len(pack.questions),
    )
    summary = dict(getattr(working_cs, "mcq_coverage_summary", None) or {})
    summary["generation_limit"] = diagnostics
    working_cs.mcq_coverage_summary = summary
    metadata = getattr(pack, "metadata", None)
    if isinstance(metadata, dict):
        metadata["generation_limit"] = diagnostics
    else:
        pack.metadata = {"generation_limit": diagnostics}
    result = {
        "role": "packager_export",
        "result": f"Packaged {len(pack.questions)} accepted MCQ(s) into {pack.pack_id}.",
        "pack_id": pack.pack_id,
        "accepted_count": len(pack.questions),
        **diagnostics,
    }
    update = _step_entry(
        state,
        "packager_export",
        result,
        cooperative_state=working_cs,
        state_before=state_before,
        input_text=title,
        reason="Package only accepted CandidateMCQs into final QuizPack",
    )
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="quiz_packs",
    )


def node_verify_candidates(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    _, working_cs, state_before = prepare_state_transition(state)
    result = _dispatch(state, runtime, "verifier", cooperative_state=working_cs)
    update = _step_entry(
        state, "verifier", result,
        cooperative_state=working_cs, state_before=state_before,
    )
    update["quiz_verify_loops"] = state.get("quiz_verify_loops", 0) + 1
    return _with_cooperative_state(
        update, working_cs, input_state=state, artifact_stage="evidence_reports",
    )


def node_package_or_end(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    needs = runtime.context.needs_evaluator
    cs = state["cooperative_state"]
    if needs.ready_for_packaging(cooperative_state=cs) and not cs.final_quiz_packs:
        _, working_cs, state_before = prepare_state_transition(state)
        mat = _material(working_cs, runtime)
        title = mat.pack_title()
        result = _dispatch(
            state,
            runtime,
            "packager_export",
            cooperative_state=working_cs,
            suggested_input=title,
            use_direct=False,
        )
        update = _step_entry(
            state, "packager_export", result,
            cooperative_state=working_cs,
            state_before=state_before,
            input_text=title,
        )
        return _with_cooperative_state(
            update, working_cs, input_state=state, artifact_stage="quiz_packs",
        )
    return {"last_agent": "package_or_end"}


def node_end_with_error(
    state: WorkflowGraphState, runtime: Runtime[WorkflowRuntime],
) -> Dict[str, Any]:
    msg = state.get("error") or "Insufficient evidence after max retrieval attempts."
    return {
        "error": msg,
        "step_logs": [{
            "step": len(state.get("step_logs", [])) + 1,
            "agent_id": None,
            "reason": msg,
            "input_text": None,
            "result_summary": "end_with_error",
            "state_before": _snapshot(state),
            "state_after": _snapshot(state),
        }],
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def _resolve_route_runtime(explicit: Any = None) -> Any:
    """Resolve workflow Runtime for routers without a ``runtime`` param name.

    LangGraph 1.2 treats a parameter literally named ``runtime`` as a required
    injection (even with a default), which breaks ``update_state`` forks.
    Routers therefore accept an optional positional for unit tests and otherwise
    use ``get_runtime()`` when available during live ``invoke``.
    """
    if explicit is not None and getattr(explicit, "context", None) is not None:
        return explicit
    try:
        from langgraph.runtime import get_runtime

        return get_runtime()
    except Exception:
        return explicit


def route_after_curate(
    state: WorkflowGraphState,
    _runtime: Any = None,
) -> RouteAfterCurate:
    del _runtime
    if state.get("pending_evidence_repair_atomize"):
        return "atomize_new_evidence"
    return "atomize_evidence"


def route_after_update_section_status(
    state: WorkflowGraphState,
    _runtime: Any = None,
) -> RouteAfterUpdateSectionStatus:
    del _runtime
    if state.get("return_to_coverage_check"):
        return "check_coverage"
    return "plan_mcq_blueprints"


def route_after_retrieve(
    state: WorkflowGraphState,
    _runtime: Any = None,
) -> RouteAfterRetrieve:
    cs = state["cooperative_state"]
    searched = _graph_searched_queries(state)
    runtime = _resolve_route_runtime(_runtime)
    context = getattr(runtime, "context", None) if runtime is not None else None
    if context is None:
        # update_state forks may lack Runtime; prefer continuing when possible
        if state.get("retrieve_attempts", 0) >= LANGGRAPH_MAX_RETRIEVE_ATTEMPTS:
            return "end_with_error"
        return "deduplicate_evidence"
    needs = context.needs_evaluator
    if needs.has_min_evidence(cooperative_state=cs):
        return "deduplicate_evidence"
    attempts = state.get("retrieve_attempts", 0)
    if attempts >= LANGGRAPH_MAX_RETRIEVE_ATTEMPTS:
        return "end_with_error"
    runner = context.agent_runner
    if runner.retrieval_exhausted(cooperative_state=cs, searched_queries=searched):
        return "end_with_error"
    if needs.needs_more_evidence(cooperative_state=cs, searched_queries=searched):
        return "retrieve_evidence"
    return "end_with_error"


def route_after_acceptance_gate(
    state: WorkflowGraphState,
    _runtime: Any = None,
) -> RouteAfterAcceptance:
    del _runtime
    revision_iterations = state.get("revision_iterations", 0)
    max_revision_iterations = state.get(
        "max_revision_iterations", LANGGRAPH_MAX_REVISION_ITERATIONS,
    )
    cs = state["cooperative_state"]
    ensure_step2_mcq_state(cs)
    max_revision_iterations = state.get(
        "max_revision_iterations", LANGGRAPH_MAX_REVISION_ITERATIONS,
    )
    if (
        count_candidates_needing_revision(cs, max_revisions=max_revision_iterations) > 0
        and revision_iterations < max_revision_iterations
    ):
        return "revise_candidate_mcqs"
    return "check_coverage"


def route_after_coverage(
    state: WorkflowGraphState,
    _runtime: Any = None,
) -> RouteAfterCoverage:
    runtime = _resolve_route_runtime(_runtime)
    cs = state["cooperative_state"]
    # Checkpoint restore of RAG CooperativeState drops monkey-patched helpers.
    ensure_step2_mcq_state(cs)
    config = None
    if runtime is not None:
        config = getattr(getattr(runtime, "context", None), "orchestrator_config", None)

    # Target/cap is accepted logical questions, not generated candidates.
    if final_question_cap_reached(cs, config):
        return "package_final_quiz"

    evidence_repair_iterations = state.get("evidence_repair_iterations", 0)
    max_evidence = state.get(
        "max_evidence_repair_iterations", LANGGRAPH_MAX_EVIDENCE_REPAIR_ITERATIONS,
    )
    blueprint_repair_iterations = state.get(
        "blueprint_repair_iterations", state.get("repair_iterations", 0),
    )
    max_blueprint = state.get(
        "max_blueprint_repair_iterations", LANGGRAPH_MAX_BLUEPRINT_REPAIR_ITERATIONS,
    )

    # Consume every already-planned blueprint before any coverage repair.
    # A section may look uncovered only because its planned MCQ is still pending.
    if cs.planned_blueprints():
        return "generate_candidate_mcqs_from_blueprints"

    missing_evidence = cs.get_sections_without_evidence()
    if missing_evidence and evidence_repair_iterations < max_evidence:
        return "repair_missing_evidence"

    gaps = list(state.get("coverage_gaps") or [])
    remaining = remaining_final_question_capacity(cs, config)
    below_target = remaining is None or remaining > 0
    if below_target:
        # Coverage-complete graph state must not hide remaining supported
        # one-per-LO/section opportunities while the accepted target is open.
        try:
            live_gaps = find_coverage_gaps(cs)
        except Exception:
            live_gaps = []
        if live_gaps:
            gaps = live_gaps

    if (
        below_target
        and gaps
        and blueprint_repair_iterations < max_blueprint
    ):
        return "generate_repair_blueprints"

    # Below target with no remaining planned/repairable opportunities, or
    # repair budgets exhausted: package the quality-accepted quiz as-is.
    if (
        below_target
        and blueprint_repair_iterations < max_blueprint
        and remaining_repairable_blueprints(cs, gaps)
    ):
        return "generate_repair_blueprints"
    return "package_final_quiz"


def route_after_verify(
    state: WorkflowGraphState,
    _runtime: Any = None,
) -> RouteAfterVerify:
    cs = state["cooperative_state"]
    loops = state.get("quiz_verify_loops", 0)
    runtime = _resolve_route_runtime(_runtime)
    context = getattr(runtime, "context", None) if runtime is not None else None
    if context is None:
        if loops >= LANGGRAPH_MAX_QUIZ_VERIFY_LOOPS:
            return "package_or_end"
        return "package_or_end"
    needs = context.needs_evaluator

    if needs.ready_for_packaging(cooperative_state=cs):
        return "package_or_end"
    if loops >= LANGGRAPH_MAX_QUIZ_VERIFY_LOOPS:
        return "package_or_end"
    if needs.needs_verification(cooperative_state=cs):
        return "verify_candidates"
    if needs.needs_more_questions(cooperative_state=cs):
        return "generate_candidate_mcqs_from_blueprints"
    return "package_or_end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_workflow_graph(checkpointer: Any = None) -> Any:
    """Compile and return the MCQ LangGraph workflow.

    Delegates to ``core.graph_builder.build_mcq_workflow_graph`` — the single
    authoritative graph wiring for factual MCQ execution.
    """
    from ..core.graph_builder import build_mcq_workflow_graph

    return build_mcq_workflow_graph(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Orchestrator wrapper (API-compatible with BaseWorkflowOrchestrator)
# ---------------------------------------------------------------------------

class LangGraphWorkflowOrchestrator:
    """LangGraph-based orchestrator — runs alongside the classic rule loop."""

    def __init__(
        self,
        state: CooperativeState,
        agent_runner: AgentRunner,
        needs_evaluator: Optional[BaseNeedsEvaluator] = None,
        config: Optional[OrchestratorConfig] = None,
        rag_agents: Any = None,
        checkpointer: Any = None,
        run_id: Optional[str] = None,
        case_definition: Any = None,
    ) -> None:
        from ..core.checkpointer import default_factual_checkpointer
        from ..core.workflow_executor import WorkflowExecutor

        self.state = state
        self.agent_runner = agent_runner
        self.config = config or DEFAULT_ORCHESTRATOR_CONFIG
        self.needs = needs_evaluator or self._default_needs()
        self.rag_agents = rag_agents or getattr(agent_runner, "rag_agents", None)
        self.orchestration_log: List[WorkflowStepLog] = []
        self.checkpointer = checkpointer or default_factual_checkpointer()
        self.run_id = run_id
        if case_definition is None:
            from ..cases.mcq.case_definition import MCQCaseDefinition

            case_definition = MCQCaseDefinition()
        self.case_definition = case_definition
        generator_llm = getattr(self.rag_agents, "llm", None) if self.rag_agents else None
        generator_model = llm_model_identity(generator_llm)
        if not generator_model and self.rag_agents is not None:
            generator_model = (
                getattr(self.rag_agents, "llm_model", None)
                or getattr(self.rag_agents, "model_name", None)
                or getattr(getattr(self.rag_agents, "config", None), "llm_model", None)
            )
            generator_model = str(generator_model) if generator_model else None
        verifier_model = resolve_cross_cove_verifier_model(
            self.config, generator_model=generator_model
        )
        verifier_temp = resolve_cross_cove_verifier_temperature(self.config)
        verifier_llm, verifier_model, _separate = build_cross_cove_verifier_llm(
            verifier_model=verifier_model,
            generator_llm=generator_llm,
            generator_model=generator_model,
            temperature=verifier_temp,
        )
        self._workflow_runtime = WorkflowRuntime(
            case_definition=self.case_definition,
            agent_runner=self.agent_runner,
            rag_agents=self.rag_agents,
            needs_evaluator=self.needs,
            orchestrator_config=self.config,
            llm=generator_llm,
            cross_cove_verifier_llm=verifier_llm,
        )
        if self.rag_agents is not None:
            self.rag_agents.cross_cove_verifier_llm = verifier_llm
        self._cross_cove_verifier_model = verifier_model
        self._workflow_generator_model = generator_model
        self._graph = self.case_definition.build_graph(checkpointer=self.checkpointer)
        self._executor = WorkflowExecutor(
            runtime=self._workflow_runtime,
            checkpointer=self.checkpointer,
            compiled_graph=self._graph,
        )
        self._last_run_id: Optional[str] = None
        self._last_factual_result: Optional[Any] = None

    def _default_needs(self) -> BaseNeedsEvaluator:
        from .needs import NeedsEvaluator

        return NeedsEvaluator(
            self.state,
            getattr(self.agent_runner, "coverage_plan", None),
            agent_runner=self.agent_runner,
        )

    def get_factual_run_result(self) -> Any:
        """Return the most recent factual ``WorkflowRunResult``."""
        if self._last_factual_result is None:
            raise RuntimeError(
                "No factual LangGraph run has completed yet. "
                "Call run_full_workflow() first."
            )
        return self._last_factual_result

    def restore_factual_runtime(
        self,
        *,
        run_id: str,
        factual_run_result: Any,
        checkpointer_state: Any,
        replay_executor_state: Any,
        cooperative_state: Any = None,
        workflow_runtime: Any = None,
    ) -> Any:
        """Reinstall a previously snapshotted factual execution into this process."""
        from mas_langgraph.evaluation.reference_sweep_snapshot import (
            restore_checkpointer_state,
            restore_replay_executor_state,
            restore_runtime_helpers,
        )

        restore_checkpointer_state(self.checkpointer, checkpointer_state or {})
        replay = self.replay_executor
        restore_replay_executor_state(replay, replay_executor_state or {})
        restore_runtime_helpers(factual_run_result)
        restore_runtime_helpers(cooperative_state)
        rt = workflow_runtime if workflow_runtime is not None else self._workflow_runtime
        if run_id:
            replay._runtimes[str(run_id)] = rt
            ensure = getattr(replay, "_ensure_workflow_llm_wrapped", None)
            if callable(ensure):
                ensure(rt)
        self._last_factual_result = factual_run_result
        self._last_run_id = str(run_id or "")
        self.run_id = str(run_id or self.run_id or "")
        if cooperative_state is not None:
            self.state = cooperative_state
        return factual_run_result

    @property
    def replay_executor(self) -> Any:
        """Replay executor sharing the factual checkpoint tree."""
        return self._executor.replay_executor

    def get_replay_executor(self) -> Any:
        return self.replay_executor

    def get_checkpoint_index(self) -> Any:
        """Frozen factual checkpoint index for the most recent run."""
        result = self.get_factual_run_result()
        idx = getattr(result, "checkpoint_index", None)
        if idx is not None:
            return idx
        return self._executor.get_checkpoint_index(result.run_id)

    def get_next_orchestration_action(
        self,
        last_agent_id: Optional[str] = None,
        step: int = 0,
    ):
        """Not used by LangGraph path — kept for protocol compatibility."""
        return None, "LangGraph orchestrator uses compiled graph, not rule loop.", None

    def get_state_history(self) -> List[Any]:
        """Return LangGraph checkpoint history for the last factual run."""
        if self._last_run_id is None:
            return []
        try:
            return self._executor.get_state_history(self._last_run_id)
        except Exception:
            return []

    def run_full_workflow(self) -> Dict[str, Any]:
        from uuid import uuid4

        mat = MaterialContext(
            self.state,
            getattr(self.agent_runner, "coverage_plan", None),
            pdf_id=getattr(self.agent_runner, "pdf_id", None),
        )
        material_theme = mat.document_theme()
        run_id = self.run_id or f"run_{uuid4().hex[:12]}"
        self.run_id = run_id
        self._last_run_id = run_id

        print_cross_cove_model_config(
            generator_model=getattr(self, "_workflow_generator_model", None),
            verifier_model=getattr(self, "_cross_cove_verifier_model", None),
            n_runs=int(getattr(self.config, "cross_cove_n_runs", 3) or 3),
        )

        initial: WorkflowGraphState = {
            "cooperative_state": self.state,
            "run_id": run_id,
            "branch_id": "factual",
            "branch_type": "factual",
            "parent_branch_id": "",
            "origin_checkpoint_alias": "",
            "intervention_id": None,
            **empty_artifact_channels(),
            **empty_provenance_channels(),
            "searched_queries": set(),
            "last_agent": None,
            "step_logs": [],
            "error": None,
            "retrieve_attempts": 0,
            "quiz_verify_loops": 0,
            "repair_iterations": 0,
            "coverage_gaps": [],
            "revision_iterations": 0,
            "max_revision_iterations": getattr(
                self.config, "max_revision_iterations", LANGGRAPH_MAX_REVISION_ITERATIONS,
            ),
            "pending_evidence_repair_atomize": False,
            "return_to_coverage_check": False,
            "evidence_repair_iterations": 0,
            "max_evidence_repair_iterations": LANGGRAPH_MAX_EVIDENCE_REPAIR_ITERATIONS,
            "blueprint_repair_iterations": 0,
            "max_blueprint_repair_iterations": LANGGRAPH_MAX_BLUEPRINT_REPAIR_ITERATIONS,
        }

        wf_result = self._executor.run(
            initial,
            runtime=self._workflow_runtime,
            run_id=run_id,
        )
        self._last_factual_result = wf_result
        final = wf_result.final_state
        if "cooperative_state" in final:
            self.state = final["cooperative_state"]
        raw_logs = final.get("step_logs") or []
        self.orchestration_log = [
            WorkflowStepLog(
                step=entry.get("step", i + 1),
                agent_id=entry.get("agent_id"),
                reason=entry.get("reason", ""),
                input_text=entry.get("input_text"),
                result_summary=entry.get("result_summary", ""),
                state_before=entry.get("state_before", {}),
                state_after=entry.get("state_after", {}),
            )
            for i, entry in enumerate(raw_logs)
        ]

        result = self._format_result(
            material_theme,
            len(self.orchestration_log),
            error=final.get("error") or wf_result.error,
        )
        result["run_id"] = wf_result.run_id
        return result

    def _format_result(
        self,
        topic: str,
        steps: int,
        *,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        final = self.state.snapshot_counts()
        base: Dict[str, Any] = {
            "material_theme": topic,
            "topic": topic,
            "steps": steps,
            "orchestrator": "langgraph",
            "orchestration_log": self.orchestration_log,
            "final_state": final,
            "error": error,
        }
        if self.config.format_workflow_result is not None:
            base.update(self.config.format_workflow_result(
                topic=topic,
                steps=steps,
                orchestration_log=self.orchestration_log,
                final_state=final,
                state=self.state,
            ))
        return base
