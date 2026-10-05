"""Cooperative multi-agent backbone — modular, profile-driven MAS framework."""

from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .core.entities import CoveragePlan, WorkflowStepLog
from .factory import create_components
from .input.profile import DEFAULT_PROFILE, MASProfile
from .input.rag_config import RAGConfig
from .orchestrator.material_context import MaterialContext


class CooperativeBackbone:
    """Main entry point — quiz scope is derived from indexed material, not a user topic."""

    def __init__(
        self,
        profile: Optional[MASProfile] = None,
        coverage_plan: Optional[CoveragePlan] = None,
        rag_config: Optional[RAGConfig] = None,
        rag_pipeline: Any = None,
        llm: Any = None,
        use_langchain: bool = False,
        orchestrator_mode: str = "classic",
    ) -> None:
        self.profile = profile or DEFAULT_PROFILE
        if coverage_plan is not None:
            self.profile = replace(self.profile, coverage_plan=coverage_plan)

        self.rag_config = rag_config
        self.rag_mode = rag_config is not None
        self.orchestrator_mode = orchestrator_mode
        self.state, self.agent_runner, self.orchestrator = create_components(
            profile=self.profile,
            rag_pipeline=rag_pipeline,
            llm=llm,
            use_langchain=use_langchain,
            rag_config=rag_config,
            orchestrator_mode=orchestrator_mode,
        )
        self.rag_agents = getattr(self.agent_runner, "rag_agents", None)
        self.orchestration_log: List[WorkflowStepLog] = []

    @property
    def execution_log(self) -> List[Dict[str, Any]]:
        return self.agent_runner.execution_log

    @property
    def pdf_id(self) -> Optional[str]:
        if self.rag_config is not None:
            return self.rag_config.pdf_id
        return getattr(self.agent_runner, "pdf_id", None)

    def material_context(self, step: int = 0) -> MaterialContext:
        return MaterialContext(
            self.state,
            self.profile.coverage_plan,
            pdf_id=self.pdf_id,
        )

    def get_next_orchestration_action(
        self,
        last_agent_id: Optional[str] = None,
        step: int = 0,
    ):
        return self.orchestrator.get_next_orchestration_action(
            last_agent_id=last_agent_id, step=step,
        )

    def run_full_workflow(self) -> Dict[str, Any]:
        self.agent_runner.topic = self.material_context().document_theme()
        result = self.orchestrator.run_full_workflow()
        self.orchestration_log = result.get("orchestration_log", [])
        # LangGraph commits a cloned CooperativeState; the orchestrator rebinds
        # its own .state. Keep the public backbone view on that committed object
        # so --dump-pack / quiz_packs() see the packaged quiz.
        orch_state = getattr(self.orchestrator, "state", None)
        if orch_state is not None:
            self.state = orch_state
        return result

    def run_agent(self, agent_id: str, input_str: str = "") -> Dict[str, Any]:
        if not input_str:
            mat = self.material_context()
            if agent_id == "retriever":
                input_str = mat.retrieval_query(0)
            elif agent_id == "quiz_generator":
                input_str = mat.quiz_target()
            elif agent_id == "packager_export":
                input_str = mat.pack_title()
            else:
                input_str = mat.document_theme()
        return self.agent_runner.invoke(agent_id, input_str)

    def state_summary(self) -> Dict[str, Any]:
        return self.state.snapshot_counts()

    def quiz_packs(self) -> List[Any]:
        return list(getattr(self.state, "final_quiz_packs", []))

    def _require_langgraph_orchestrator(self) -> Any:
        if self.orchestrator_mode != "langgraph":
            raise RuntimeError(
                "AXIS replay explanations require a completed LangGraph factual run."
            )
        return self.orchestrator

    def factual_run_result(self) -> Any:
        """Most recent factual ``WorkflowRunResult`` (LangGraph mode only)."""
        orch = self._require_langgraph_orchestrator()
        if not hasattr(orch, "get_factual_run_result"):
            raise RuntimeError(
                "AXIS replay explanations require a completed LangGraph factual run."
            )
        return orch.get_factual_run_result()

    def list_explainable_artifacts(self) -> Tuple[Any, ...]:
        """Exact explainable artifact refs from the last factual LangGraph run."""
        from .axis.service import AxisExplanationService

        orch = self._require_langgraph_orchestrator()
        factual = self.factual_run_result()
        service = AxisExplanationService(
            case_definition=getattr(orch, "case_definition", None),
            factual_run_result=factual,
            replay_executor=orch.get_replay_executor(),
            checkpoint_index=orch.get_checkpoint_index(),
        )
        return service.list_explainable_artifacts()

    def default_llm_generate(self) -> Optional[Callable[[str], str]]:
        """Reuse the configured runtime LLM for AXIS / experiment synthesis."""
        from .experiment.llm import default_llm_generate_from_backbone

        return default_llm_generate_from_backbone(self)

    def explain_artifact(
        self,
        artifact_ref: Any,
        question: str,
        *,
        scripted_queries: Optional[Sequence[str]] = None,
        llm_generate: Optional[Callable[[str], str]] = None,
        intervention_registry: Any = None,
        intervention_alternatives: Optional[Mapping[str, Any]] = None,
        max_rounds: int = 5,
        axis_interrogation_mode: str = "constrained",
    ) -> Any:
        """Run AXIS explanation for one exact artifact from the factual run.

        When ``llm_generate`` is omitted, the configured runtime LLM is used so
        autonomous interrogation does not terminate with ``no_llm``.
        Pass an explicit callable to override; pass ``scripted_queries`` for
        deterministic scripts (tests / offline).
        """
        from .axis.service import AxisExplanationService

        orch = self._require_langgraph_orchestrator()
        factual = self.factual_run_result()
        gen = llm_generate if llm_generate is not None else self.default_llm_generate()
        service = AxisExplanationService(
            case_definition=getattr(orch, "case_definition", None),
            factual_run_result=factual,
            replay_executor=orch.get_replay_executor(),
            checkpoint_index=orch.get_checkpoint_index(),
            llm_generate=gen,
            intervention_registry=intervention_registry,
            intervention_alternatives=intervention_alternatives,
            max_rounds=max_rounds,
            axis_interrogation_mode=axis_interrogation_mode,
        )
        return service.explain(
            artifact_ref=artifact_ref,
            question=question,
            scripted_queries=scripted_queries,
        )

    def list_explanation_experiment_refs(
        self,
        *,
        include_historical_versions: bool = False,
    ) -> Tuple[Any, ...]:
        """Final-quiz explainable CandidateMCQ refs eligible for the experiment."""
        from .experiment.service import build_experiment_service_from_backbone

        service = build_experiment_service_from_backbone(
            self, configure_explanation_llms=False
        )
        return service.list_experiment_refs(
            include_historical_versions=include_historical_versions
        )

    def generate_explanation_experiment(
        self,
        question: str = "Why did the system produce this decision?",
        *,
        experiment_seed: int = 42,
        include_historical_versions: bool = False,
        candidate_refs: Optional[Sequence[Any]] = None,
        max_rounds: int = 5,
        llm_generate: Optional[Callable[[str], str]] = None,
        axis_scripted_queries: Optional[Sequence[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        axis_interrogation_mode: str = "constrained",
        build_effect_reference_set: bool = False,
        reference_max_interventions_per_question: Optional[int] = None,
        reference_resume_rows: Optional[Sequence[Any]] = None,
        reference_persist_callback: Optional[Callable[..., None]] = None,
        lecturebank_id: str = "",
        diagnose_hybrid_add: bool = False,
        intervention_registry: Optional[Any] = None,
        after_axis_callback: Optional[Callable[..., None]] = None,
    ) -> Any:
        """Trace / Trace+Provenance / AXIS ablation for final-quiz MCQs.

        All three conditions share one factual run and the same synthesis LLM.
        When ``candidate_refs`` is omitted, all normally eligible final MCQs are
        explained. When supplied, only those exact refs are explained (must be
        members of the eligible set; no silent version substitution).
        """
        from .experiment.service import build_experiment_service_from_backbone

        service = build_experiment_service_from_backbone(
            self,
            llm_generate=llm_generate,
            max_rounds=max_rounds,
            axis_scripted_queries=axis_scripted_queries,
            axis_interrogation_mode=axis_interrogation_mode,
            build_effect_reference_set=build_effect_reference_set,
            reference_max_interventions_per_question=reference_max_interventions_per_question,
            reference_resume_rows=reference_resume_rows,
            reference_persist_callback=reference_persist_callback,
            lecturebank_id=lecturebank_id,
            diagnose_hybrid_add=bool(diagnose_hybrid_add),
            intervention_registry=intervention_registry,
        )
        return service.generate(
            question=question,
            experiment_seed=experiment_seed,
            include_historical_versions=include_historical_versions,
            candidate_refs=candidate_refs,
            progress_callback=progress_callback,
            after_axis_callback=after_axis_callback,
        )
