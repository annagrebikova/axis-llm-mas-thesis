"""Generic rule-based orchestration loop — domain-agnostic engine."""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..core.entities import WorkflowStepLog
from ..input.orchestrator_config import DEFAULT_ORCHESTRATOR_CONFIG, OrchestratorConfig
from .material_context import MaterialContext
from .needs_base import BaseNeedsEvaluator

if TYPE_CHECKING:
    from ..agents.runner import AgentRunner
    from ..core.state_base import BaseCooperativeState


class BaseWorkflowOrchestrator:
    """Evaluates ordered rules and runs agents until stop conditions.

    Domain-specific behavior comes from:
      - OrchestratorConfig (rules, limits, terminal agent, result formatter)
      - BaseNeedsEvaluator subclass (condition functions)
      - BaseCooperativeState subclass (snapshot_counts)
    """

    def __init__(
        self,
        state: "BaseCooperativeState",
        agent_runner: "AgentRunner",
        needs_evaluator: Optional[BaseNeedsEvaluator] = None,
        config: Optional[OrchestratorConfig] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.state = state
        self.agent_runner = agent_runner
        self.config = config or DEFAULT_ORCHESTRATOR_CONFIG
        self.rules = rules if rules is not None else self.config.rules
        self.needs = needs_evaluator or self._default_needs_evaluator()
        self.orchestration_log: List[WorkflowStepLog] = []

    def _default_needs_evaluator(self) -> BaseNeedsEvaluator:
        from .needs import NeedsEvaluator

        needs = NeedsEvaluator(
            self.state,
            getattr(self.agent_runner, "coverage_plan", None),
            agent_runner=self.agent_runner,
        )
        return needs

    def _material(self, step: int = 0) -> MaterialContext:
        pdf_id = getattr(self.agent_runner, "pdf_id", None)
        plan = getattr(self.agent_runner, "coverage_plan", None)
        return MaterialContext(self.state, plan, pdf_id=pdf_id)

    def get_next_orchestration_action(
        self,
        last_agent_id: Optional[str] = None,
        step: int = 0,
    ) -> Tuple[Optional[str], str, Optional[str]]:
        conditions = self.needs.as_dict()

        for rule in self.rules:
            if not self._rule_applies(rule, last_agent_id, step):
                continue

            cond_name = rule.get("condition", "always")
            cond_fn = conditions.get(cond_name, lambda **_: False)
            if cond_name == "always":
                if rule.get("requires"):
                    req = rule["requires"]
                    if req not in conditions or not conditions[req]():
                        continue
            elif not cond_fn():
                continue

            skip_set = rule.get("skip_if_last_agent") or []
            if last_agent_id in skip_set:
                alt = rule.get("alternate")
                if alt:
                    alt_inp = alt.get("input")
                    if alt_inp is None and alt.get("input_template"):
                        alt_inp = self._resolve_input(alt, step)
                    return (
                        alt["agent"],
                        alt.get("reason", ""),
                        alt_inp,
                    )
                continue

            agent = rule.get("agent")
            reason = rule.get("reason", "")
            inp = self._resolve_input(rule, step)
            return agent, reason, inp

        return None, "No matching orchestration rule.", None

    def _rule_applies(
        self,
        rule: Dict[str, Any],
        last_agent_id: Optional[str],
        step: int,
    ) -> bool:
        after = rule.get("after_agent")
        if after is not None and last_agent_id != after:
            return False

        step_mod = rule.get("step_modulo")
        if step_mod and rule.get("skip_if_last_agent_unless_step"):
            last_is = rule["skip_if_last_agent_unless_step"]
            mod = step_mod.get("mod", self.config.step_modulo_default)
            threshold = step_mod.get("threshold", self.config.step_threshold_default)
            if last_agent_id == last_is and step % mod <= threshold:
                return False
        return True

    def _resolve_input(self, rule: Dict[str, Any], step: int) -> Optional[str]:
        if "input" in rule:
            return rule["input"]
        template = rule.get("input_template")
        mat = self._material(step)
        if template == "section_query":
            runner = self.agent_runner
            if hasattr(runner, "next_retrieval_query"):
                nxt = runner.next_retrieval_query(step)
                if nxt:
                    return nxt
            return mat.retrieval_query(step)
        if template == "material_query":
            return mat.document_theme()
        if template in ("coverage_topic", "quiz_target"):
            return mat.quiz_target()
        if template == "pack_title":
            return mat.pack_title()
        if template == "topic":
            return mat.document_theme()
        return None

    def run_full_workflow(self) -> Dict[str, Any]:
        cfg = self.config
        self.orchestration_log = []
        last_agent_id: Optional[str] = None
        last_snapshot: Optional[Dict[str, Any]] = None
        no_progress_count = 0
        step = 0
        material_theme = self._material(0).document_theme()

        while step < cfg.max_steps:
            snapshot_before = self.state.snapshot_counts()
            next_agent, reason, suggested_input = self.get_next_orchestration_action(
                last_agent_id=last_agent_id, step=step,
            )

            if next_agent is None:
                self.orchestration_log.append(WorkflowStepLog(
                    step=step + 1,
                    agent_id=None,
                    reason=reason,
                    input_text=None,
                    result_summary="stop",
                    state_before=snapshot_before,
                    state_after=self.state.snapshot_counts(),
                ))
                break

            try:
                fallback = self._material(step).quiz_target()
                result = self.agent_runner.invoke(
                    next_agent, suggested_input or fallback,
                )
                result_summary = str(result.get("result", ""))[:300]
                error = result.get("error")
            except Exception as e:
                result_summary = f"error: {e}"
                error = str(e)

            snapshot_after = self.state.snapshot_counts()
            self.orchestration_log.append(WorkflowStepLog(
                step=step + 1,
                agent_id=next_agent,
                reason=reason,
                input_text=suggested_input,
                result_summary=result_summary,
                state_before=snapshot_before,
                state_after=snapshot_after,
            ))
            last_agent_id = next_agent

            if next_agent == cfg.terminal_success_agent and not error:
                break

            if last_snapshot is not None:
                if all(
                    snapshot_after.get(k) == last_snapshot.get(k)
                    for k in cfg.progress_metric_keys
                ):
                    no_progress_count += 1
                    if no_progress_count >= cfg.no_progress_iterations:
                        self.orchestration_log.append(WorkflowStepLog(
                            step=step + 2,
                            agent_id=None,
                            reason=(
                                f"No progress for {cfg.no_progress_iterations} steps; stopping."
                            ),
                            input_text=None,
                            result_summary="no_progress_stop",
                            state_before=snapshot_after,
                            state_after=snapshot_after,
                        ))
                        break
                else:
                    no_progress_count = 0
            last_snapshot = snapshot_after
            step += 1

        return self._format_result(material_theme, step + 1)

    def _format_result(self, topic: str, steps: int) -> Dict[str, Any]:
        final = self.state.snapshot_counts()
        base = {
            "material_theme": topic,
            "topic": topic,
            "steps": steps,
            "orchestration_log": self.orchestration_log,
            "final_state": final,
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


# Backwards-compatible alias
WorkflowOrchestrator = BaseWorkflowOrchestrator
