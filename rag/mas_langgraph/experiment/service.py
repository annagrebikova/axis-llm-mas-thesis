"""Public experiment service: Trace / Trace+Provenance / AXIS ablation."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..artifacts.models import ArtifactRef
from ..artifacts.registry import ArtifactRegistry
from ..axis.export import explanation_result_to_report, axis_report_to_dict
from ..axis.models import ExplainedCase
from ..axis.service import AxisExplanationService
from ..axis.slicing import ProvenanceSlicer
from ..axis.interrogation_mode import (
    DEFAULT_AXIS_INTERROGATION_MODE,
    DEFAULT_AXIS_INTERROGATION_POLICY,
    normalize_axis_interrogation_mode,
)
from ..core.case_config import load_axis_case_capabilities
from ..replay.executor import WorkflowRunResult
from .blinding import blinding_mapping
from .evidence import (
    assert_no_leakage,
    build_axis_evidence_summary,
    build_trace_evidence,
    build_trace_provenance_evidence,
    synthesis_evidence_without_axis_raw,
)
from .faithfulness import validate_explanation_faithfulness
from .llm import make_llm_generate, resolve_runtime_llm
from .models import (
    EVIDENCE_ACCESS,
    EXPERIMENT_SCHEMA_VERSION,
    AnswerOptionView,
    ExplanationConditionResult,
    MCQExplanationComparison,
    QuizExplanationExperiment,
)
from .selection import select_experiment_refs
from .timing import current_timing, explanation_llm_phase
from .verbalizer import build_evidence_view
from .visualization_data import build_case_visualization_data


def _synthesize_condition(
    *,
    condition: str,
    raw_evidence: Dict[str, Any],
    question: str,
    factual_decision: str,
    question_stem: str,
    llm_generate: Optional[Callable[[str], str]],
    case_slice: Any = None,
    evidence_view: Any = None,
) -> tuple[str, Any, Tuple[str, ...], Dict[str, Any]]:
    """Verbalize → shared final prompt → validate → optional repair → word limit.

    Returns ``(text, view, warnings, diagnostics)``.
    """
    from .synthesis import (
        MAX_PARTICIPANT_WORDS,
        build_faithfulness_repair_prompt,
        compress_participant_explanation,
        count_words,
        mechanical_synthesize_from_view,
        participant_quotes_are_balanced,
        synthesize_explanation,
    )
    from .presentation_policy import (
        count_presentation_sentences,
        validate_participant_presentation_policy,
    )

    synthesis_raw = synthesis_evidence_without_axis_raw(raw_evidence)
    assert "final_explanation_raw" not in synthesis_raw
    if "question_stem" not in synthesis_raw and question_stem:
        synthesis_raw = {**synthesis_raw, "question_stem": question_stem}
    view = evidence_view if evidence_view is not None else build_evidence_view(
        condition=condition,  # type: ignore[arg-type]
        raw_evidence=synthesis_raw,
        case_slice=case_slice,
    )
    llm_raw = ""
    if llm_generate is not None:
        llm_raw = synthesize_explanation(
            user_question=question,
            factual_decision=factual_decision,
            question_stem=question_stem,
            evidence_view=view,
            llm_generate=llm_generate,
        )
    raw_before = llm_raw
    mechanical = mechanical_synthesize_from_view(view=view)
    text = llm_raw
    used_mechanical_fallback = False
    word_limit_enforced = False
    word_limit_strategy = "none"

    if llm_generate is None or not text:
        text = mechanical
        used_mechanical_fallback = True
        word_limit_strategy = "mechanical_fallback"

    def _check(candidate: str):
        ok_f, warns_f = validate_explanation_faithfulness(
            condition=condition,  # type: ignore[arg-type]
            explanation=candidate,
            evidence_view=view,
        )
        pres = validate_participant_presentation_policy(
            candidate, facts=getattr(view, "lay_facts", None)
        )
        return bool(ok_f), tuple(warns_f), pres

    ok, warns, pres = _check(text)
    warns_before = tuple(warns)
    synthesis_initial_valid = bool(ok)
    presentation_initial_valid = bool(pres.valid)
    repair_attempted = False
    repair_valid = False
    repaired_text = ""
    warns_after_repair: Tuple[str, ...] = ()
    needs_repair = (not ok) or (not pres.valid)

    if needs_repair and llm_generate is not None and not used_mechanical_fallback:
        repair_attempted = True
        combined_warnings = tuple(
            list(warns_before) + [f"presentation:{v}" for v in pres.violations]
        )
        repair_prompt = build_faithfulness_repair_prompt(
            previous_explanation=text,
            warnings=combined_warnings,
            verbalized_evidence=view.to_prompt_text(),
        )
        try:
            repaired_text = str(llm_generate(repair_prompt) or "").strip()
        except Exception:
            repaired_text = ""
        if repaired_text:
            ok_repair, warns_repair, pres_repair = _check(repaired_text)
            warns_after_repair = tuple(warns_repair)
            if ok_repair and pres_repair.valid:
                text = repaired_text
                ok = True
                pres = pres_repair
                repair_valid = True
                warns = warns_repair
            else:
                text = mechanical
                used_mechanical_fallback = True
                word_limit_strategy = "mechanical_fallback"
                ok, warns, pres = _check(text)
                warns = tuple(
                    list(warns_before)
                    + list(warns_repair)
                    + ["faithfulness_repair_used_mechanical_fallback"]
                )
        else:
            text = mechanical
            used_mechanical_fallback = True
            word_limit_strategy = "mechanical_fallback"
            ok, warns, pres = _check(text)
            warns = tuple(
                list(warns_before) + ["faithfulness_repair_used_mechanical_fallback"]
            )
    elif needs_repair and not used_mechanical_fallback:
        text = mechanical
        used_mechanical_fallback = True
        word_limit_strategy = "mechanical_fallback"
        ok, warns, pres = _check(text)
        warns = tuple(list(warns) + ["faithfulness_repair_used_mechanical_fallback"])

    def _apply_mechanical_last_resort(current_warns: Tuple[str, ...]) -> Tuple[str, Tuple[str, ...]]:
        fb, _ = compress_participant_explanation(mechanical, max_words=MAX_PARTICIPANT_WORDS)
        extra = []
        if "faithfulness_repair_used_mechanical_fallback" not in current_warns:
            extra.append("faithfulness_repair_used_mechanical_fallback")
        extra.append("participant_word_limit_enforced")
        return fb, tuple(list(current_warns) + extra)

    if not used_mechanical_fallback and (
        count_words(text) > MAX_PARTICIPANT_WORDS
        or not participant_quotes_are_balanced(text)
    ):
        compressed, strategy = compress_participant_explanation(
            text,
            max_words=MAX_PARTICIPANT_WORDS,
            fallback="",
        )
        if (
            compressed
            and count_words(compressed) <= MAX_PARTICIPANT_WORDS
            and participant_quotes_are_balanced(compressed)
        ):
            ok_c, warns_c, pres_c = _check(compressed)
            if ok_c and pres_c.valid:
                text = compressed
                ok = True
                pres = pres_c
                word_limit_enforced = True
                word_limit_strategy = "safe_compression"
                warns = tuple(warns_c)
            else:
                text, warns = _apply_mechanical_last_resort(tuple(warns_c))
                used_mechanical_fallback = True
                word_limit_enforced = True
                word_limit_strategy = "mechanical_fallback"
        else:
            text, warns = _apply_mechanical_last_resort(tuple(warns))
            used_mechanical_fallback = True
            word_limit_enforced = True
            word_limit_strategy = "mechanical_fallback"
    elif used_mechanical_fallback:
        text, _ = compress_participant_explanation(
            text, max_words=MAX_PARTICIPANT_WORDS, fallback=""
        )
        if count_words(text) > MAX_PARTICIPANT_WORDS:
            word_limit_enforced = True

    lay = getattr(view, "lay_facts", None)
    if not used_mechanical_fallback:
        # Final presentation check; mechanical last resort if lint still fails.
        final_pres = validate_participant_presentation_policy(text, facts=lay)
        if not final_pres.valid:
            text = mechanical
            used_mechanical_fallback = True
            word_limit_strategy = "mechanical_fallback"
            ok, warns, pres = _check(text)
            warns = tuple(
                list(warns) + ["faithfulness_repair_used_mechanical_fallback"]
            )
    pres = validate_participant_presentation_policy(text, facts=lay)
    diagnostics: Dict[str, Any] = {
        "synthesis_initial_valid": bool(synthesis_initial_valid),
        "synthesis_repair_attempted": bool(repair_attempted),
        "synthesis_repair_valid": bool(repair_valid),
        "synthesis_used_mechanical_fallback": bool(used_mechanical_fallback),
        "synthesis_used_fallback": bool(used_mechanical_fallback),
        "synthesis_word_limit_enforced": bool(word_limit_enforced),
        "synthesis_word_limit_strategy": word_limit_strategy,
        "synthesis_faithfulness_warning_count": len(
            [
                w
                for w in warns
                if w
                not in {
                    "faithfulness_repair_used_mechanical_fallback",
                    "participant_word_limit_enforced",
                }
            ]
        ),
        "participant_synthesis_raw_before_faithfulness": raw_before,
        "participant_synthesis_repaired": repaired_text if repair_attempted else "",
        "faithfulness_warnings_before_repair": list(warns_before),
        "faithfulness_warnings_after_repair": list(warns_after_repair),
        "participant_word_count": count_words(text),
        "participant_word_limit": MAX_PARTICIPANT_WORDS,
        "participant_sentence_count": count_presentation_sentences(text),
        "participant_length_policy": pres.length_class,
        "participant_presentation_valid": bool(pres.valid),
        "participant_presentation_violations": list(pres.violations),
        "participant_presentation_initial_valid": bool(presentation_initial_valid),
    }
    if lay is not None and hasattr(lay, "diagnostics"):
        diagnostics.update(lay.diagnostics())
    return text, view, warns, diagnostics

def _snapshot_factual_immutability(factual: WorkflowRunResult) -> Dict[str, Any]:
    reg = factual.artifact_registry
    catalog_keys = sorted((reg.catalog if reg else {}).keys())
    return {
        "run_id": factual.run_id,
        "branch_id": factual.branch_id,
        "catalog_keys": catalog_keys,
        "produced_by": len(list(factual.produced_by or [])),
        "relations": len(list(factual.provenance_relations or [])),
        "occurrences": len(list(factual.occurrences or [])),
        "packs": [
            getattr(p, "pack_id", None)
            for p in (
                getattr(
                    (factual.final_state or {}).get("cooperative_state"),
                    "final_quiz_packs",
                    [],
                )
                or []
            )
        ],
        "ckpt_ids": [
            getattr(r, "checkpoint_id", None)
            for r in (
                getattr(factual.checkpoint_index, "records", None) or []
            )
        ],
    }


def _mcq_presentation(
    *,
    registry: ArtifactRegistry,
    candidate_ref: ArtifactRef,
    branch_id: str,
) -> Tuple[str, Tuple[AnswerOptionView, ...], Optional[str]]:
    env = registry.get_by_ref(candidate_ref, branch_id) or registry.get_by_ref(
        candidate_ref
    )
    payload = getattr(env, "payload", None) if env is not None else None
    stem = str(getattr(payload, "stem", "") or "")
    options_raw = list(getattr(payload, "options", None) or [])
    correct = getattr(payload, "correct_option_id", None)
    opts: List[AnswerOptionView] = []
    for o in options_raw:
        oid = str(getattr(o, "option_id", "") or "")
        text = str(getattr(o, "text", "") or "")
        is_correct = bool(getattr(o, "is_correct", False)) or (
            correct is not None and oid == str(correct)
        )
        opts.append(AnswerOptionView(option_id=oid, text=text, is_correct=is_correct))
    if not opts and payload is not None:
        # ValidatedQuestion-style fallback
        stem = stem or str(getattr(payload, "question_text", "") or "")
        raw_opts = list(getattr(payload, "options", None) or [])
        for i, text in enumerate(raw_opts):
            label = chr(ord("A") + i)
            opts.append(
                AnswerOptionView(
                    option_id=label,
                    text=str(text),
                    is_correct=str(text) == str(getattr(payload, "correct_answer", "")),
                )
            )
            if opts[-1].is_correct:
                correct = label
    return stem, tuple(opts), str(correct) if correct is not None else None


class QuizExplanationExperimentService:
    """Generate Trace / Trace+Provenance / AXIS explanations for one factual run."""

    def __init__(
        self,
        *,
        case_definition: Any,
        factual_run_result: WorkflowRunResult,
        replay_executor: Any,
        checkpoint_index: Any = None,
        llm_generate: Optional[Callable[[str], str]] = None,
        synthesis_llm_generate: Optional[Callable[[str], str]] = None,
        synthesis_model: str = "",
        model_bundle: Any = None,
        cooperative_state: Any = None,
        intervention_registry: Any = None,
        intervention_alternatives: Optional[Mapping[str, Any]] = None,
        max_rounds: int = 5,
        axis_scripted_queries: Optional[Sequence[str]] = None,
        axis_interrogation_mode: str = DEFAULT_AXIS_INTERROGATION_MODE,
        build_effect_reference_set: bool = False,
        reference_max_interventions_per_question: Optional[int] = None,
        reference_resume_rows: Optional[Sequence[Mapping[str, Any]]] = None,
        reference_persist_callback: Optional[Callable[[Sequence[Mapping[str, Any]]], None]] = None,
        lecturebank_id: str = "",
        diagnose_hybrid_add: bool = False,
    ) -> None:
        if factual_run_result is None:
            raise ValueError("factual_run_result is required")
        if factual_run_result.branch_id != "factual":
            raise ValueError("experiment requires factual branch")
        caps = load_axis_case_capabilities(case_definition, strict=True)
        resolver = caps.explained_case_resolver
        if resolver is None:
            raise ValueError("case missing explained_case_resolver")
        self.case_definition = case_definition
        self.factual_run_result = factual_run_result
        self.replay_executor = replay_executor
        self.checkpoint_index = checkpoint_index or factual_run_result.checkpoint_index
        self.capabilities = caps
        self._resolver = resolver
        self.llm_generate = llm_generate
        # Shared final synthesis for all three conditions (fairness).
        self.synthesis_llm_generate = (
            synthesis_llm_generate
            if synthesis_llm_generate is not None
            else llm_generate
        )
        self.synthesis_model = synthesis_model
        self.model_bundle = model_bundle
        self.cooperative_state = cooperative_state
        self.intervention_registry = intervention_registry
        self.intervention_alternatives = dict(intervention_alternatives or {})
        self.max_rounds = int(max_rounds)
        self.max_attempts = int(max(10, self.max_rounds * 2))
        # Production path: autonomous interrogation. Tests may pass a script.
        self.axis_scripted_queries = (
            list(axis_scripted_queries) if axis_scripted_queries is not None else None
        )
        self.axis_interrogation_mode = normalize_axis_interrogation_mode(
            axis_interrogation_mode
        )
        self.axis_interrogation_policy = DEFAULT_AXIS_INTERROGATION_POLICY
        self.build_effect_reference_set = bool(build_effect_reference_set)
        self.reference_max_interventions_per_question = (
            reference_max_interventions_per_question
        )
        self.reference_resume_rows = list(reference_resume_rows or ())
        self.reference_persist_callback = reference_persist_callback
        self.lecturebank_id = str(lecturebank_id or "")
        self._effect_reference_interventions: List[Dict[str, Any]] = []
        self._effect_reference_questions: List[Dict[str, Any]] = []
        self._axis_results_by_ref: Dict[str, Any] = {}
        self._frozen_universes: Dict[str, List[Any]] = {}
        self.after_axis_callback: Optional[
            Callable[["QuizExplanationExperiment", "QuizExplanationExperimentService"], None]
        ] = None
        self._effect_reference_warnings: List[str] = []
        self._axis_service = AxisExplanationService(
            case_definition=case_definition,
            factual_run_result=factual_run_result,
            replay_executor=replay_executor,
            checkpoint_index=self.checkpoint_index,
            llm_generate=llm_generate,
            intervention_registry=intervention_registry,
            intervention_alternatives=self.intervention_alternatives,
            max_rounds=self.max_rounds,
            max_attempts=self.max_attempts,
            axis_interrogation_mode=self.axis_interrogation_mode,
            diagnose_hybrid_add=bool(diagnose_hybrid_add),
            lecturebank_id=str(lecturebank_id or ""),
        )

    def list_experiment_refs(
        self,
        *,
        include_historical_versions: bool = False,
    ) -> Tuple[ArtifactRef, ...]:
        refs = self._axis_service.list_explainable_artifacts()
        return select_experiment_refs(
            explainable_refs=refs,
            factual_run_result=self.factual_run_result,
            resolver=self._resolver,
            include_historical_versions=include_historical_versions,
            cooperative_state=self.cooperative_state,
        )

    def generate(
        self,
        *,
        question: str = "Why did the system produce this decision?",
        experiment_seed: int = 42,
        include_historical_versions: bool = False,
        candidate_refs: Optional[Sequence[ArtifactRef]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        after_axis_callback: Optional[
            Callable[["QuizExplanationExperiment", "QuizExplanationExperimentService"], None]
        ] = None,
        skip_axis_generation: bool = False,
    ) -> QuizExplanationExperiment:
        timing = current_timing()
        before = _snapshot_factual_immutability(self.factual_run_result)
        t_discover = time.perf_counter()
        eligible = self.list_experiment_refs(
            include_historical_versions=include_historical_versions
        )
        if timing is not None:
            timing.record_experiment_phase(
                "case_discovery_s", time.perf_counter() - t_discover
            )
        if candidate_refs is None:
            refs = eligible
        else:
            eligible_set = set(eligible)
            selected: List[ArtifactRef] = []
            for ref in candidate_refs:
                if not isinstance(ref, ArtifactRef):
                    raise TypeError(
                        "candidate_refs entries must be ArtifactRef instances "
                        f"(got {type(ref)!r})"
                    )
                if ref not in eligible_set:
                    raise ValueError(
                        f"candidate_ref {ref} is not in the eligible experiment set. "
                        "Eligible refs: "
                        + ", ".join(str(r) for r in eligible)
                        + (" (empty)" if not eligible else "")
                    )
                selected.append(ref)
            refs = tuple(selected)
        cases: List[MCQExplanationComparison] = []
        warnings: List[str] = []
        if not skip_axis_generation:
            for i, ref in enumerate(refs, start=1):
                if progress_callback:
                    progress_callback(f"[{i}/{len(refs)}] {ref}")
                case = self._generate_one(
                    ref,
                    question=question,
                    experiment_seed=experiment_seed,
                    progress_callback=progress_callback,
                )
                cases.append(case)
        else:
            cases = list(getattr(self, "_resumed_cases", ()) or [])
        after = _snapshot_factual_immutability(self.factual_run_result)
        if before != after:
            raise RuntimeError(
                "Factual run mutated during explanation experiment generation"
            )
        llm_diag: Dict[str, Any] = {}
        executor = getattr(self, "replay_executor", None)
        getter = (
            getattr(executor, "get_workflow_llm_diagnostics", None)
            if executor is not None
            else None
        )
        if callable(getter):
            try:
                llm_diag = dict(getter(self.factual_run_result.run_id) or {})
            except Exception:
                llm_diag = {}
        models_export: Dict[str, Any] = {}
        bundle = getattr(self, "model_bundle", None)
        if bundle is not None and hasattr(bundle, "to_export_dict"):
            models_export = dict(bundle.to_export_dict())
        timing_payload: Dict[str, Any] = {}
        if timing is not None:
            timing.run_id = str(self.factual_run_result.run_id)
            # Merge ledger diagnostics into timing without inventing values.
            for key in (
                "factual_llm_request_count",
                "noop_live_provider_calls",
                "counterfactual_live_provider_calls",
                "noop_factual_response_hits",
                "counterfactual_factual_response_hits",
                "counterfactual_memo_hits",
                "noop_llm_request_misses",
            ):
                if key in llm_diag:
                    timing.workflow_llm[key] = llm_diag[key]
            # Friendly aliases used by the terminal summary (derived only from
            # existing ledger counters — no estimated values).
            if (
                "factual_llm_request_count" in llm_diag
                and "factual_llm_memo_hits" in llm_diag
            ):
                timing.workflow_llm["factual_provider_calls"] = int(
                    llm_diag["factual_llm_request_count"] or 0
                ) - int(llm_diag["factual_llm_memo_hits"] or 0)
            if "noop_live_provider_calls" in llm_diag:
                timing.workflow_llm["noop_provider_calls"] = llm_diag[
                    "noop_live_provider_calls"
                ]
            if "counterfactual_live_provider_calls" in llm_diag:
                timing.workflow_llm["counterfactual_provider_calls"] = llm_diag[
                    "counterfactual_live_provider_calls"
                ]
            if "noop_factual_response_hits" in llm_diag or (
                "counterfactual_factual_response_hits" in llm_diag
            ):
                timing.workflow_llm["factual_response_hits"] = int(
                    llm_diag.get("noop_factual_response_hits") or 0
                ) + int(llm_diag.get("counterfactual_factual_response_hits") or 0)
            if "counterfactual_memo_hits" in llm_diag:
                timing.workflow_llm["counterfactual_memo_hits"] = llm_diag[
                    "counterfactual_memo_hits"
                ]
            timing_payload = timing.to_dict()
        experiment = QuizExplanationExperiment(
            schema_version=EXPERIMENT_SCHEMA_VERSION,
            run_id=self.factual_run_result.run_id,
            branch_id=self.factual_run_result.branch_id,
            experiment_seed=int(experiment_seed),
            user_question=question,
            cases=tuple(cases),
            evidence_access=dict(EVIDENCE_ACCESS),
            synthesis_model=self.synthesis_model,
            models=models_export,
            include_historical_versions=include_historical_versions,
            warnings=tuple(warnings),
            workflow_llm_diagnostics=llm_diag,
            timing=timing_payload,
            axis_interrogation_mode=getattr(
                self,
                "axis_interrogation_mode",
                DEFAULT_AXIS_INTERROGATION_MODE,
            ),
            axis_interrogation_policy=getattr(
                self,
                "axis_interrogation_policy",
                "counterfactual_only",
            ),
        )
        hook = after_axis_callback or getattr(self, "after_axis_callback", None)
        if getattr(self, "build_effect_reference_set", False) and not skip_axis_generation:
            self.prepare_reference_universes()
        if callable(hook):
            hook(experiment, self)
        if getattr(self, "build_effect_reference_set", False):
            self.run_deferred_reference_sweeps()
            from mas_langgraph.evaluation.effect_reference import aggregate_effect_discovery

            payload = {
                "intervention_rows": list(self._effect_reference_interventions),
                "question_rows": list(self._effect_reference_questions),
                "warnings": list(self._effect_reference_warnings),
                "aggregate": aggregate_effect_discovery(self._effect_reference_questions),
                "status": (
                    "complete"
                    if self._effect_reference_questions
                    and all(
                        r.get("reference_set_complete") is True
                        for r in self._effect_reference_questions
                    )
                    else "incomplete"
                ),
            }
            object.__setattr__(experiment, "_effect_reference", payload)
        return experiment

    def _reference_sweep_common_kwargs(self) -> Dict[str, Any]:
        caps = getattr(self, "capabilities", None)
        contracts = None
        obs_adapter = None
        if caps is not None:
            cfg = getattr(caps, "case_configuration", None)
            if cfg is not None:
                contracts = getattr(cfg, "intervention_contracts", None)
            obs_adapter = getattr(caps, "observation_adapter", None)
        idx = self.checkpoint_index
        ckpt_map = {
            "records": [
                {
                    "alias": getattr(rec, "alias", None),
                    "checkpoint_id": getattr(rec, "checkpoint_id", None),
                }
                for rec in (getattr(idx, "records", None) or [])
            ]
        }
        registry = getattr(self.factual_run_result, "artifact_registry", None)
        return {
            "replay_executor": self.replay_executor,
            "factual_run_result": self.factual_run_result,
            "case_definition": self.case_definition,
            "checkpoint_index": idx,
            "contracts": contracts,
            "intervention_registry": self.intervention_registry,
            "intervention_alternatives": self.intervention_alternatives,
            "registry": registry,
            "occurrences": getattr(self.factual_run_result, "occurrences", ()) or (),
            "observation_adapter": obs_adapter,
            "checkpoint_index_mapping": ckpt_map,
        }

    def prepare_reference_universes(self) -> Dict[str, List[Any]]:
        """Enumerate and freeze the replay-reference universe for every AXIS case."""
        from mas_langgraph.evaluation.effect_reference import (
            enumerate_effect_reference_universe_for_case,
        )

        if self._frozen_universes:
            return self._frozen_universes
        common = self._reference_sweep_common_kwargs()
        for ref, axis_result in dict(self._axis_results_by_ref).items():
            universe, warnings, _vocab = enumerate_effect_reference_universe_for_case(
                axis_result=axis_result,
                axis_interrogation_mode=self.axis_interrogation_mode,
                **{
                    k: common[k]
                    for k in (
                        "replay_executor",
                        "factual_run_result",
                        "case_definition",
                        "checkpoint_index",
                        "contracts",
                        "intervention_registry",
                        "intervention_alternatives",
                        "registry",
                        "occurrences",
                    )
                },
            )
            self._frozen_universes[str(ref)] = list(universe)
            self._effect_reference_warnings.extend(warnings)
        return self._frozen_universes

    def apply_resume_axis_state(
        self,
        *,
        axis_results: Optional[Mapping[str, Any]] = None,
        frozen_universes: Optional[Mapping[str, Sequence[Any]]] = None,
        intervention_registry: Any = None,
        intervention_alternatives: Optional[Mapping[str, Any]] = None,
        axis_noop_cache: Any = None,
        experiment: Any = None,
    ) -> None:
        if axis_results:
            self._axis_results_by_ref = {str(k): v for k, v in dict(axis_results).items()}
        if frozen_universes:
            self._frozen_universes = {
                str(k): list(v or []) for k, v in dict(frozen_universes).items()
            }
        if intervention_registry is not None:
            self.intervention_registry = intervention_registry
        if intervention_alternatives is not None:
            self.intervention_alternatives = dict(intervention_alternatives)
        if axis_noop_cache is not None:
            self._axis_service.noop_cache = axis_noop_cache
        if experiment is not None:
            self._resumed_cases = list(getattr(experiment, "cases", ()) or [])

    def run_deferred_reference_sweeps(self) -> None:
        """Execute remaining frozen-universe interventions without rerunning AXIS."""
        from mas_langgraph.evaluation.effect_reference import (
            run_effect_reference_sweep_for_case,
        )

        if not getattr(self, "build_effect_reference_set", False):
            return
        if not self._frozen_universes:
            self.prepare_reference_universes()
        common = self._reference_sweep_common_kwargs()
        self._effect_reference_interventions = []
        self._effect_reference_questions = []
        for ref, axis_result in dict(self._axis_results_by_ref).items():
            try:
                ref_payload = run_effect_reference_sweep_for_case(
                    lecturebank_id=str(getattr(self, "lecturebank_id", "") or ""),
                    candidate_ref=str(ref),
                    axis_result=axis_result,
                    max_rounds=self.max_rounds,
                    axis_interrogation_mode=self.axis_interrogation_mode,
                    reference_max_interventions_per_question=self.reference_max_interventions_per_question,
                    resume_rows=self.reference_resume_rows,
                    persist_rows_callback=self.reference_persist_callback,
                    axis_noop_cache=getattr(self._axis_service, "noop_cache", None),
                    frozen_universe=self._frozen_universes.get(str(ref)),
                    **common,
                )
                self._effect_reference_interventions.extend(
                    ref_payload.get("intervention_rows") or []
                )
                if ref_payload.get("question_row"):
                    self._effect_reference_questions.append(ref_payload["question_row"])
                self._effect_reference_warnings.extend(ref_payload.get("warnings") or [])
            except Exception as exc:  # noqa: BLE001
                self._effect_reference_warnings.append(str(exc))
                raise

    def attach_effect_reference(self, experiment: QuizExplanationExperiment) -> None:
        from mas_langgraph.evaluation.effect_reference import aggregate_effect_discovery

        payload = {
            "intervention_rows": list(self._effect_reference_interventions),
            "question_rows": list(self._effect_reference_questions),
            "warnings": list(self._effect_reference_warnings),
            "aggregate": aggregate_effect_discovery(self._effect_reference_questions),
            "status": (
                "complete"
                if self._effect_reference_questions
                and all(
                    r.get("reference_set_complete") is True
                    for r in self._effect_reference_questions
                )
                else "incomplete"
            ),
        }
        object.__setattr__(experiment, "_effect_reference", payload)

    def _generate_one(
        self,
        candidate_ref: ArtifactRef,
        *,
        question: str,
        experiment_seed: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> MCQExplanationComparison:
        timing = current_timing()
        explained = self._resolver.resolve(
            factual_run_result=self.factual_run_result,
            artifact_ref=candidate_ref,
        )
        registry = self.factual_run_result.artifact_registry
        if registry is None:
            raise ValueError("factual_run_result has no artifact_registry")

        if timing is not None:
            timing.begin_case(
                target=str(candidate_ref),
                candidate_version=getattr(candidate_ref, "version", None),
                factual_decision=str(explained.outcome or ""),
            )

        # Shared CaseSlice for Trace / Trace+Prov (no AXIS interrogation yet).
        slicer = ProvenanceSlicer()
        case_slice = slicer.slice_run_result(
            explained,
            self.factual_run_result,
            registry=registry,
            adapter=getattr(self.capabilities, "slicing_adapter", None),
        )

        stem, options, correct = _mcq_presentation(
            registry=registry,
            candidate_ref=candidate_ref,
            branch_id=explained.branch_id,
        )

        # --- Trace ---
        t_trace = time.perf_counter()
        trace_ev = build_trace_evidence(
            explained_case=explained,
            factual_occurrences=getattr(self.factual_run_result, "occurrences", None)
            or (),
        )
        assert_no_leakage(trace_ev, allow_provenance=False)
        with explanation_llm_phase("trace_participant_synthesis"):
            trace_text, _trace_view, trace_warns, trace_diag = _synthesize_condition(
                condition="trace",
                raw_evidence=trace_ev,
                question=question,
                factual_decision=explained.outcome,
                question_stem=stem,
                llm_generate=self.synthesis_llm_generate,
                case_slice=case_slice,
            )
        trace_s = time.perf_counter() - t_trace
        if timing is not None:
            timing.record_case_condition("trace", trace_s)
            timing.add_condition("trace", trace_s)
        if progress_callback:
            progress_callback("  trace: done")
        trace = ExplanationConditionResult(
            condition="trace",
            explanation=trace_text,
            evidence_summary={
                "execution_trace_len": len(trace_ev.get("execution_trace") or []),
                "provenance_included": False,
                "counterfactual_replay_included": False,
                "verbalized_process_steps": len(_trace_view.observed_process),
                **trace_diag,
            },
            raw_evidence=trace_ev,
            warnings=trace_warns,
        )

        # --- Trace + Provenance ---
        t_tp = time.perf_counter()
        tp_ev = build_trace_provenance_evidence(
            explained_case=explained,
            case_slice=case_slice,
            registry=registry,
        )
        assert_no_leakage(tp_ev, allow_provenance=True)
        with explanation_llm_phase("trace_provenance_participant_synthesis"):
            tp_text, _tp_view, tp_warns, tp_diag = _synthesize_condition(
                condition="trace_provenance",
                raw_evidence=tp_ev,
                question=question,
                factual_decision=explained.outcome,
                question_stem=stem,
                llm_generate=self.synthesis_llm_generate,
                case_slice=case_slice,
            )
        tp_s = time.perf_counter() - t_tp
        if timing is not None:
            timing.record_case_condition("trace_provenance", tp_s)
            timing.add_condition("trace_provenance", tp_s)
        if progress_callback:
            progress_callback("  trace+provenance: done")
        trace_prov = ExplanationConditionResult(
            condition="trace_provenance",
            explanation=tp_text,
            evidence_summary={
                "execution_trace_len": len(tp_ev.get("execution_trace") or []),
                "provenance_relations": len(tp_ev.get("provenance_relations") or []),
                "relevant_artifacts": len(tp_ev.get("relevant_artifacts") or []),
                "provenance_included": True,
                "counterfactual_replay_included": False,
                "verbalized_dependencies": len(_tp_view.recorded_dependencies),
                **tp_diag,
            },
            raw_evidence=tp_ev,
            warnings=tp_warns,
        )

        # --- AXIS (real controller / replay) ---
        t_axis = time.perf_counter()
        axis_warnings: List[str] = []
        axis_status = "ok"
        axis_result = None
        try:
            axis_result = self._axis_service.explain(
                artifact_ref=candidate_ref,
                question=question,
                scripted_queries=self.axis_scripted_queries,
            )
        except Exception as exc:  # noqa: BLE001
            axis_status = "failed"
            axis_warnings.append(str(exc))
            axis_result = None

        if axis_result is not None:
            axis_ev = build_axis_evidence_summary(
                explanation_result=axis_result,
                case_slice=case_slice,
                explained_case=explained,
                registry=registry,
            )
            if self.intervention_registry is not None:
                from mas_langgraph.axis.hybrid_add_source import (
                    export_selected_add_artifacts,
                )

                axis_ev["hybrid_add_source_artifacts"] = export_selected_add_artifacts(
                    axis_result, self.intervention_registry
                )
            term = str(getattr(axis_result, "termination_reason", "") or "")
            cf_n = int(axis_ev.get("replay_comparison_count") or 0)
            success_n = int(axis_ev.get("successful_replay_count") or 0)
            cand_n = int(axis_ev.get("valid_intervention_candidate_count") or 0)
            statuses = list(axis_ev.get("evidence_statuses") or [])
            if term == "no_valid_intervention_available" or (
                cf_n == 0
                and cand_n == 0
                and "no_valid_intervention_available"
                in (axis_ev.get("warnings") or [])
            ):
                axis_status = "no_valid_intervention_available"
            elif success_n > 0:
                # Evidence outcome is orthogonal to why interrogation stopped.
                axis_status = "completed_with_counterfactual"
            elif term in {"noop_error", "no_llm"}:
                axis_status = f"completed_with_{term}"
            elif term == "attempt_budget_exhausted":
                axis_status = "attempt_budget_exhausted"
            elif cf_n > 0 and any(
                s in {"noop_error", "intervention_error", "outcome_unavailable"}
                for s in statuses
            ):
                axis_status = "replay_failed"
            elif cf_n > 0:
                axis_status = "replay_failed"
            else:
                axis_status = "completed_without_counterfactual"
            with explanation_llm_phase("axis_participant_synthesis"):
                axis_text, _axis_view, axis_faith_warns, axis_diag = _synthesize_condition(
                    condition="axis",
                    raw_evidence=axis_ev,
                    question=question,
                    factual_decision=explained.outcome,
                    question_stem=stem,
                    llm_generate=self.synthesis_llm_generate,
                    case_slice=case_slice,
                )
            try:
                report = explanation_result_to_report(
                    axis_result,
                    question=question,
                    case_definition=self.case_definition,
                )
                axis_ev["axis_report"] = axis_report_to_dict(report)
            except Exception as exc:  # noqa: BLE001
                axis_warnings.append(f"axis report export: {exc}")
            axis_warnings.extend(list(axis_faith_warns))
            axis_ev["warnings"] = list(axis_ev.get("warnings") or []) + axis_warnings
            if progress_callback:
                progress_callback(
                    "  axis: done "
                    f"successful_rounds={axis_ev.get('successful_round_count') or axis_ev.get('rounds', 0)} "
                    f"counterfactual_executions={axis_ev.get('counterfactual_executions') or axis_ev.get('counterfactual_count', 0)} "
                    f"valid_replay_comparisons={axis_ev.get('valid_replay_comparisons') or axis_ev.get('successful_replay_count', 0)}"
                )
            axis_cond = ExplanationConditionResult(
                condition="axis",
                explanation=axis_text,
                evidence_summary={
                    "rounds": axis_ev.get("rounds"),
                    "successful_round_count": axis_ev.get("successful_round_count"),
                    "attempt_count": axis_ev.get("attempt_count"),
                    "valid_intervention_candidate_count": axis_ev.get(
                        "valid_intervention_candidate_count"
                    ),
                    "counterfactual_count": axis_ev.get("counterfactual_count"),
                    "counterfactual_executions": axis_ev.get(
                        "counterfactual_executions"
                    ),
                    "successful_replay_count": axis_ev.get("successful_replay_count"),
                    "valid_replay_comparisons": axis_ev.get("valid_replay_comparisons"),
                    "axis_causal_evidence_available": axis_ev.get(
                        "axis_causal_evidence_available"
                    ),
                    "termination_reason": axis_ev.get("termination_reason"),
                    "interrogation_budget_exhausted": axis_ev.get(
                        "interrogation_budget_exhausted"
                    ),
                    "duplicate_attempt_count": axis_ev.get("duplicate_attempt_count"),
                    "evidence_statuses": axis_ev.get("evidence_statuses"),
                    "verbalized_interventions": len(_axis_view.tested_interventions),
                    **axis_diag,
                },
                raw_evidence=axis_ev,
                warnings=tuple(axis_warnings),
                status=axis_status,
            )
            causal = bool(axis_ev.get("axis_causal_evidence_available"))
            cf_count = int(axis_ev.get("counterfactual_count") or 0)
            ok_count = int(axis_ev.get("successful_replay_count") or 0)
        else:
            empty_axis = {
                "kind": "axis",
                "factual_decision": explained.outcome,
                "execution_trace": trace_ev.get("execution_trace", []),
                "provenance_relations": tp_ev.get("provenance_relations", []),
                "relevant_artifacts": tp_ev.get("relevant_artifacts", []),
                "decision_reasons": tp_ev.get("decision_reasons", []),
                "criteria": tp_ev.get("criteria", []),
                "axis_causal_evidence_available": False,
                "evidence_statuses": [],
                "rounds": 0,
                "replay_comparison_count": 0,
                "replay_comparisons": [],
                "warnings": axis_warnings,
            }
            with explanation_llm_phase("axis_participant_synthesis"):
                axis_text, _axis_view, axis_faith_warns, axis_diag = _synthesize_condition(
                    condition="axis",
                    raw_evidence=empty_axis,
                    question=question,
                    factual_decision=explained.outcome,
                    question_stem=stem,
                    llm_generate=self.synthesis_llm_generate,
                    case_slice=case_slice,
                )
            axis_cond = ExplanationConditionResult(
                condition="axis",
                explanation=axis_text,
                evidence_summary={
                    "axis_causal_evidence_available": False,
                    **axis_diag,
                },
                raw_evidence={
                    **empty_axis,
                    "error": axis_warnings,
                    "final_explanation_raw": "",
                },
                warnings=tuple(list(axis_warnings) + list(axis_faith_warns)),
                status=axis_status,
            )
            causal = False
            cf_count = 0
            ok_count = 0

        axis_s = time.perf_counter() - t_axis
        if timing is not None:
            timing.record_case_condition("axis", axis_s)
            timing.add_condition("axis", axis_s)
            timing.end_case(
                meta={
                    "valid_replay_comparisons": ok_count,
                }
            )

        if axis_result is not None:
            self._axis_results_by_ref[str(candidate_ref)] = axis_result

        mapping = blinding_mapping(
            run_id=explained.run_id,
            candidate_ref=candidate_ref,
            experiment_seed=experiment_seed,
        )
        visualization_data = build_case_visualization_data(
            explained_case=explained,
            case_slice=case_slice,
            registry=registry,
            produced_by=getattr(self.factual_run_result, "produced_by", None),
            factual_decision=str(explained.outcome or ""),
        )
        return MCQExplanationComparison(
            run_id=explained.run_id,
            branch_id=explained.branch_id,
            candidate_ref=candidate_ref,
            assessment_ref=explained.assessment_ref
            or ArtifactRef(f"assessment:{candidate_ref.logical_id}", candidate_ref.version),
            decision_ref=explained.decision_ref,
            question_stem=stem,
            answer_options=options,
            correct_option_id=correct,
            factual_decision=str(explained.outcome or ""),
            user_question=question,
            trace=trace,
            trace_provenance=trace_prov,
            axis=axis_cond,
            blinding=mapping,
            axis_status=axis_status,
            axis_causal_evidence_available=causal,
            counterfactual_count=cf_count,
            successful_replay_count=ok_count,
            visualization_data=visualization_data,
        )


def build_experiment_service_from_backbone(
    backbone: Any,
    *,
    llm_generate: Optional[Callable[[str], str]] = None,
    synthesis_llm_generate: Optional[Callable[[str], str]] = None,
    max_rounds: int = 5,
    axis_scripted_queries: Optional[Sequence[str]] = None,
    explanation_llm_config: Any = None,
    configure_explanation_llms: bool = True,
    axis_interrogation_mode: str = DEFAULT_AXIS_INTERROGATION_MODE,
    build_effect_reference_set: bool = False,
    reference_max_interventions_per_question: Optional[int] = None,
    reference_resume_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    reference_persist_callback: Optional[Callable[[Sequence[Mapping[str, Any]]], None]] = None,
    lecturebank_id: str = "",
    diagnose_hybrid_add: bool = False,
    intervention_registry: Any = None,
) -> QuizExplanationExperimentService:
    """Wire experiment service from a completed LangGraph ``CooperativeBackbone``.

    Explanation-layer LLMs are constructed independently of the workflow
    ``ReplayAwareWorkflowLLM`` / ledger unless the caller overrides generators.
    """
    from .llm_config import (
        ExplanationLLMConfig,
        ExperimentModelBundle,
        build_explanation_generators,
        default_explanation_llm_config,
        resolve_workflow_model_from_backbone,
    )

    orch = backbone._require_langgraph_orchestrator()
    factual = backbone.factual_run_result()

    cfg = explanation_llm_config or default_explanation_llm_config()
    if not isinstance(cfg, ExplanationLLMConfig):
        cfg = default_explanation_llm_config()

    inter_gen = llm_generate
    synth_gen = synthesis_llm_generate
    wf_model, wf_temp = resolve_workflow_model_from_backbone(backbone)
    cc_model = ""
    cc_temp = 0.0
    try:
        orch_rt = getattr(orch, "_workflow_runtime", None)
        from ..domains.quiz.cross_cove import (
            llm_model_identity,
            resolve_cross_cove_verifier_model,
            resolve_cross_cove_verifier_temperature,
        )

        cc_llm = getattr(orch_rt, "cross_cove_verifier_llm", None) if orch_rt else None
        cc_model = llm_model_identity(cc_llm)
        if not cc_model:
            cc_model = str(
                resolve_cross_cove_verifier_model(
                    getattr(orch, "config", None),
                    generator_model=wf_model,
                )
                or ""
            )
        cc_temp = resolve_cross_cove_verifier_temperature(getattr(orch, "config", None))
    except Exception:
        cc_model = cc_model or ""
    model_bundle = ExperimentModelBundle(
        workflow_model=wf_model,
        workflow_temperature=wf_temp,
        interrogation_model=cfg.interrogation_model,
        synthesis_model=cfg.synthesis_model,
        reasoning_effort=cfg.reasoning_effort,
        cross_cove_verifier_model=cc_model,
        cross_cove_verifier_temperature=cc_temp,
    )

    if configure_explanation_llms and (inter_gen is None or synth_gen is None):
        built_inter, built_synth, _ = build_explanation_generators(cfg)
        if inter_gen is None:
            inter_gen = built_inter
        if synth_gen is None:
            synth_gen = built_synth
        for line in model_bundle.startup_lines():
            print(line)

    return QuizExplanationExperimentService(
        case_definition=getattr(orch, "case_definition", None),
        factual_run_result=factual,
        replay_executor=orch.get_replay_executor(),
        checkpoint_index=orch.get_checkpoint_index(),
        llm_generate=inter_gen,
        synthesis_llm_generate=synth_gen,
        synthesis_model=model_bundle.synthesis_model,
        model_bundle=model_bundle,
        cooperative_state=getattr(backbone, "state", None),
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
