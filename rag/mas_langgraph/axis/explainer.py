"""AXIS explanation loop controller (Prompt 14 — full iterative loop)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..artifacts.models import ArtifactRef
from ..artifacts.registry import ArtifactRegistry
from ..artifacts.utils import payloads_equivalent
from ..core.case_config import CaseConfigurationError, load_axis_case_capabilities
from ..provenance.models import OccurrenceRecord, RunProvenanceRelation
from ..replay.checkpoints import BoundarySelector, CheckpointIndex
from ..replay.executor import LangGraphReplayExecutor
from ..replay.interventions import ArtifactIntervention, InterventionContractRegistry
from .models import (
    AxisPromptTemplates,
    AxisQueryContext,
    ExplainedCase,
    FactualLookupResult,
    FactualQueryRecord,
    InterventionQueryRecord,
    QuerySignature,
    ResolvedQuery,
)
from .slicing import ProvenanceSliceAdapter
from .queries import AxisQueryParser, AxisQueryResolver, QueryValidationError
from .slicing import CaseSlice, ProvenanceSlicer
from .executor import AxisQueryExecutor
from .comparison import (
    AxisEpisodicHistory,
    CaseRunObservation,
    CaseRunObservationAdapter,
    ExplanationEvidenceLedger,
    InterrogationRoundRecord,
    NoOpReplayCache,
    QueryExecutionLedger,
    ReplayComparator,
    ReplayComparison,
)
from .intervention_guidance import (
    InterrogationAttempt,
    build_hybrid_intervention_vocabulary,
    build_intervention_candidates,
    classify_validation_error,
    constrained_opportunity_counts,
    controlled_hybrid_validation_feedback,
    controlled_validation_feedback,
    find_candidate_for_ref,
    format_executable_queries,
    format_factual_only_artifacts,
    format_hybrid_intervenable_handoff,
    format_hybrid_vocabulary,
    format_intervenable_handoff,
    format_intervention_candidates,
    hybrid_opportunity_counts,
    validate_resolved_against_hybrid_vocabulary,
)
from .interrogation_mode import (
    DEFAULT_AXIS_INTERROGATION_MODE,
    DEFAULT_AXIS_INTERROGATION_POLICY,
    WHAT_DISABLED_IN_INTERROGATION,
    is_hybrid_interrogation_mode,
    normalize_axis_interrogation_mode,
)
from .verbalizer import AxisVerbalizer, _GenericVerbalizer


def _question_content_changed(comp: Any) -> bool:
    scd = getattr(comp, "subject_content_delta", None)
    if scd is None:
        return False
    det = dict(getattr(scd, "details", None) or {})
    status = str(
        getattr(scd, "effect_status", None)
        or det.get("question_content_effect_status")
        or ""
    )
    if status in {
        "question_content_changed",
        "question_content_partially_supported",
    }:
        return True
    return any(
        det.get(k) == "changed"
        for k in ("stem_effect", "correct_answer_effect", "distractor_effect")
    )


def _sanitize_raw_axis_summary(
    text: str,
    *,
    comparisons: Sequence[Any] = (),
) -> str:
    """Prevent unqualified 'no question effect' when content comparison disagrees."""
    out = str(text or "")
    if not any(_question_content_changed(c) for c in comparisons or ()):
        return out
    import re

    patterns = (
        r"no demonstrated effect on the question",
        r"no demonstrated effect on question content",
        r"question would not have been different",
        r"question content did not change",
        r"no demonstrated effect on the generated question",
        r"\bno demonstrated effect\b(?!\s+on\s+(?:the\s+)?(?:final\s+)?decision)",
    )
    changed = False
    for pat in patterns:
        new_out, n = re.subn(
            pat,
            "no demonstrated decision effect (question content did change)",
            out,
            flags=re.IGNORECASE,
        )
        if n:
            out = new_out
            changed = True
    if changed and "question content" not in out.lower():
        out = (
            out.rstrip()
            + "\n\nNote: structured question-content comparison shows a change in "
            "stem, correct answer, and/or distractors; do not treat decision-only "
            "'no demonstrated effect' as applying to question wording."
        )
    return out


@dataclass
class ExplanationResult:
    """Final grounded explanation package."""

    explained_case: ExplainedCase
    case_slice: CaseSlice
    factual_context: str = ""
    factual_records: List[FactualQueryRecord] = field(default_factory=list)
    intervention_records: List[InterventionQueryRecord] = field(default_factory=list)
    replay_comparisons: List[ReplayComparison] = field(default_factory=list)
    round_records: List[InterrogationRoundRecord] = field(default_factory=list)
    intermediate_texts: List[str] = field(default_factory=list)
    final_explanation: str = ""
    rounds: int = 0  # successful executed interrogation rounds
    successful_round_count: int = 0
    attempt_count: int = 0
    interrogation_attempts: List[InterrogationAttempt] = field(default_factory=list)
    termination_reason: str = ""
    executed_signatures: Tuple[QuerySignature, ...] = ()
    warnings: Tuple[str, ...] = ()
    valid_intervention_candidate_count: int = 0
    # Researcher-only NOOP cache instrumentation (not participant-facing).
    noop_cache_stats: Dict[str, int] = field(default_factory=dict)
    axis_interrogation_mode: str = "constrained"
    axis_interrogation_policy: str = DEFAULT_AXIS_INTERROGATION_POLICY
    hybrid_add_diagnostics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Factual context builder
# ---------------------------------------------------------------------------

_HYBRID_CONTROLLED_ALTERNATIVES_NOTICE = (
    "HYBRID MODE: the HYBRID INTERVENTION VOCABULARY of the current interrogation "
    "round is the authoritative list of replacement artifacts, ADD sources, "
    "operations, and boundaries. This section is not that list. Do not invent artifacts."
)
_CONTROLLED_ALTERNATIVES_LINE = re.compile(
    r"(?im)^[^\n]*CONTROLLED INTERVENTION ALTERNATIVES[^\n]*$"
)
_HYBRID_VOCABULARY_RULE = (
    "- Only use artifact refs, replacements, ADD sources, operation types, and "
    "boundaries listed in the HYBRID INTERVENTION VOCABULARY shown for the "
    "current interrogation round. Do not invent artifacts."
)


def hybridize_system_prompt(system: str) -> str:
    """Rewrite a constrained system prompt so hybrid never sees the alternatives contradiction."""
    text = str(system or "")
    text = re.sub(r"(?m)^\s*WHAT\(target\).*\n?", "", text)
    if not text.strip():
        return _HYBRID_VOCABULARY_RULE
    if (
        "HYBRID INTERVENTION VOCABULARY" in text
        and "CONTROLLED INTERVENTION ALTERNATIVES" not in text
    ):
        return text
    replaced, n = _CONTROLLED_ALTERNATIVES_LINE.subn(_HYBRID_VOCABULARY_RULE, text)
    if n:
        return replaced
    return text.rstrip() + "\n" + _HYBRID_VOCABULARY_RULE + "\n"


def _build_factual_context(
    *,
    question: str,
    explained_case: ExplainedCase,
    case_slice: CaseSlice,
    registry: ArtifactRegistry,
    checkpoint_index: CheckpointIndex,
    contracts: InterventionContractRegistry,
    verbalizer: Any,
    prompt_templates: Optional[AxisPromptTemplates] = None,
    intervention_alternatives: Optional[Dict[str, ArtifactRef]] = None,
    strict: bool = True,
    axis_interrogation_mode: Any = None,
) -> str:
    """Build factual context using the template if available."""
    hybrid_mode = is_hybrid_interrogation_mode(axis_interrogation_mode)
    if prompt_templates and prompt_templates.factual_context:
        if not hasattr(verbalizer, "factual_context_sections"):
            if strict:
                raise CaseConfigurationError(
                    "AXIS verbalizer must implement factual_context_sections()"
                )
        else:
            sections = dict(
                verbalizer.factual_context_sections(
                    explained_case=explained_case,
                    case_slice=case_slice,
                    registry=registry,
                    checkpoint_index=checkpoint_index,
                    contracts=contracts,
                    intervention_alternatives=intervention_alternatives or {},
                )
            )
            if hybrid_mode:
                sections["controlled_alternatives"] = (
                    _HYBRID_CONTROLLED_ALTERNATIVES_NOTICE
                )
                space = str(sections.get("intervention_space") or "").strip()
                sections["intervention_space"] = (
                    "CONTRACT-LEVEL INTERVENTION CAPABILITIES\n"
                    "These describe structural contract permissions only.\n"
                    "The current HYBRID INTERVENTION VOCABULARY defines which "
                    "queries are actually available in this round.\n"
                    + (f"\n{space}" if space else "")
                )
            if getattr(case_slice, "criterion_values", ()) and not str(
                sections.get("criteria", "")
            ).strip():
                raise CaseConfigurationError(
                    "factual_context criteria section is blank despite recorded criteria"
                )
            if strict:
                return prompt_templates.render("factual_context", **sections)
            try:
                return prompt_templates.render("factual_context", **sections)
            except (ValueError, KeyError):
                pass
        if strict:
            raise CaseConfigurationError(
                "Unable to render factual_context.txt in strict AXIS mode"
            )
    return verbalizer.verbalize_slice(case_slice)


# ---------------------------------------------------------------------------
# LLM prompt construction helpers
# ---------------------------------------------------------------------------

def _render_system(
    templates: Optional[AxisPromptTemplates],
    *,
    axis_interrogation_mode: Any = None,
) -> str:
    if is_hybrid_interrogation_mode(axis_interrogation_mode):
        if templates and getattr(templates, "system_hybrid", ""):
            return templates.system_hybrid
        if templates and templates.system:
            return hybridize_system_prompt(templates.system)
        return _HYBRID_VOCABULARY_RULE
    if templates and templates.system:
        return templates.system
    return ""


def _render_interrogation(
    templates: Optional[AxisPromptTemplates],
    **kwargs: Any,
) -> str:
    defaults: Dict[str, Any] = {
        "round": "?",
        "max_rounds": "?",
        "attempt": "?",
        "max_attempts": "?",
        "question": "",
        "factual_context": "",
        "episodic_history": "",
        "tested_queries": "(none)",
        "available_artifacts": "",
        "intervention_space": "",
        "valid_boundaries": "",
        "intervention_candidates": "(none)",
        "factual_only_artifacts": "(none)",
        "executable_queries": "",
        "hybrid_vocabulary": "",
        "intervenable_handoff": "",
        "validation_feedback": "",
    }
    defaults.update({k: ("" if v is None else v) for k, v in kwargs.items()})
    template_name = str(defaults.pop("template_name", "interrogation") or "interrogation")
    if templates is not None:
        body = getattr(templates, template_name, None)
        if body:
            try:
                return templates.render(template_name, **defaults)
            except (ValueError, KeyError):
                pass
    if template_name == "interrogation_hybrid":
        return (
            f"Interrogation round {defaults.get('round', '?')}/"
            f"{defaults.get('max_rounds', '?')}.\n\n"
            f"Question:\n{defaults.get('question', '')}\n\n"
            "You may compose exactly one query:\n\n"
            "REMOVE(target, boundary)\n\n"
            "ADD(artifact, boundary)\n\n"
            "WHATIF(old_artifact, replacement_artifact, boundary)\n\nDONE\n\n"
            f"{defaults.get('hybrid_vocabulary', '')}\n\n"
            f"{defaults.get('intervenable_handoff', '')}\n"
            f"{defaults.get('validation_feedback', '')}\n"
            "Reply with exactly one structured query or DONE."
        )
    # Fallback
    return (
        f"Round {defaults.get('round', '?')}/{defaults.get('max_rounds', '?')} "
        f"(attempt {defaults.get('attempt', '?')}/{defaults.get('max_attempts', '?')})\n"
        f"Question: {defaults.get('question', '')}\n"
        f"{defaults.get('validation_feedback', '')}\n"
        f"{defaults.get('intervenable_handoff', '')}\n"
        f"Valid intervention candidates:\n{defaults.get('intervention_candidates', '')}\n"
        "Return exactly one structured query."
    )


def _render_intermediate(
    templates: Optional[AxisPromptTemplates],
    **kwargs: Any,
) -> str:
    if templates and templates.intermediate_explanation:
        try:
            return templates.render("intermediate_explanation", **kwargs)
        except (ValueError, KeyError):
            pass
    return ""


def _render_final(
    templates: Optional[AxisPromptTemplates],
    **kwargs: Any,
) -> str:
    if templates and templates.final_explanation:
        try:
            return templates.render("final_explanation", **kwargs)
        except (ValueError, KeyError):
            pass
    return ""


def _llm_call(
    llm_generate: Optional[Callable[[str], str]],
    system: str,
    task: str,
) -> str:
    """Combine system + task into one string for single-string callables."""
    if llm_generate is None:
        return ""
    prompt = f"=== SYSTEM ===\n{system}\n\n=== TASK ===\n{task}" if system else task
    return llm_generate(prompt)


# ---------------------------------------------------------------------------
# Default observation adapter (generic fallback)
# ---------------------------------------------------------------------------

class _DefaultObservationAdapter:
    """Minimal observation adapter when no case-specific adapter is available."""

    def observe_factual(
        self, *, explained_case: ExplainedCase, case_slice: Any, registry: Any
    ) -> CaseRunObservation:
        return CaseRunObservation(
            run_id=explained_case.run_id,
            branch_id=explained_case.branch_id,
            subject_ref=explained_case.artifact_ref,
            decision_ref=explained_case.decision_ref,
            decision=getattr(case_slice, "decision", None) or explained_case.outcome,
            criterion_values=tuple(getattr(case_slice, "criterion_values", ())),
            outcome_available=True,
        )

    def observe_replay(
        self, *, explained_case: ExplainedCase, run_result: Any
    ) -> CaseRunObservation:
        return CaseRunObservation(
            run_id=getattr(run_result, "run_id", "?"),
            branch_id=getattr(run_result, "branch_id", "?"),
            outcome_available=False,
            error="No case-specific observation adapter configured",
        )


# ===========================================================================
# Controller
# ===========================================================================

class AxisExplanationController:
    """Full AXIS interrogation loop (Prompt 14).

    Case-neutral: verbalizer, observation adapter, and prompt templates
    are supplied externally (e.g. by ``case_definition``).

    ``axis_interrogation_mode``:
      - ``constrained`` (default): LLM copies one exact executable query form.
      - ``hybrid``: LLM composes one query from a controlled vocabulary; the
        proposal is parsed, resolved, vocabulary-checked, then executed on the
        same replay engine.

    New interrogation is counterfactual-only (``axis_interrogation_policy`` =
    ``counterfactual_only``): successful non-DONE rounds are REMOVE / ADD /
    WHATIF. WHAT is parseable for historical JSON but is rejected in new runs.
    ``max_rounds`` therefore bounds successful counterfactual interventions.

    Scripted queries still go through parse / resolve / execute. In hybrid mode
    they are additionally checked against the advertised vocabulary.
    """

    def __init__(
        self,
        *,
        explained_case: ExplainedCase,
        registry: ArtifactRegistry,
        occurrences: Sequence[OccurrenceRecord],
        provenance_relations: Sequence[RunProvenanceRelation],
        checkpoint_index: CheckpointIndex,
        replay_executor: LangGraphReplayExecutor,
        contracts: Optional[InterventionContractRegistry] = None,
        alias_map: Optional[Dict[str, ArtifactRef]] = None,
        verbalizer: Optional[Any] = None,
        llm_generate: Optional[Callable[[str], str]] = None,
        max_rounds: int = 5,
        max_attempts: Optional[int] = None,
        prompt_dir: Optional[str] = None,
        slicing_adapter: Optional[ProvenanceSliceAdapter] = None,
        prompt_templates: Optional[AxisPromptTemplates] = None,
        intervention_registry: Optional[ArtifactRegistry] = None,
        intervention_alternatives: Optional[Dict[str, ArtifactRef]] = None,
        case_definition: Optional[Any] = None,
        observation_adapter: Optional[Any] = None,
        compatibility_mode: bool = False,
        factual_run_result: Optional[Any] = None,
        noop_cache: Optional[NoOpReplayCache] = None,
        axis_interrogation_mode: str = DEFAULT_AXIS_INTERROGATION_MODE,
        diagnose_hybrid_add: bool = False,
        lecturebank_id: str = "",
    ) -> None:
        # Load from case_definition if provided (strict AXIS uses one capability bundle)
        if case_definition is not None and not compatibility_mode:
            caps = load_axis_case_capabilities(case_definition, strict=True)
            verbalizer = verbalizer or caps.verbalizer
            prompt_templates = prompt_templates or caps.prompt_templates
            if contracts is None:
                contracts = caps.case_configuration.intervention_contracts
            slicing_adapter = slicing_adapter or caps.slicing_adapter
            observation_adapter = observation_adapter or caps.observation_adapter
        elif case_definition is not None and compatibility_mode:
            if verbalizer is None and hasattr(case_definition, "load_verbalizer"):
                try:
                    verbalizer = case_definition.load_verbalizer()
                except Exception:
                    pass
            if prompt_templates is None and hasattr(case_definition, "load_prompt_templates"):
                try:
                    prompt_templates = case_definition.load_prompt_templates()
                except Exception:
                    pass
            if contracts is None and hasattr(case_definition, "load_intervention_contracts"):
                try:
                    contracts = case_definition.load_intervention_contracts()
                except Exception:
                    pass
            if slicing_adapter is None and hasattr(case_definition, "load_provenance_slice_adapter"):
                try:
                    slicing_adapter = case_definition.load_provenance_slice_adapter()
                except Exception:
                    pass
            if observation_adapter is None and hasattr(
                case_definition, "load_run_observation_adapter"
            ):
                try:
                    observation_adapter = case_definition.load_run_observation_adapter()
                except Exception:
                    pass

        self.explained_case = explained_case
        self.registry = registry
        self.occurrences = list(occurrences)
        self.provenance_relations = list(provenance_relations)
        self.checkpoint_index = checkpoint_index
        self.replay_executor = replay_executor
        self.contracts = contracts or InterventionContractRegistry()
        self.verbalizer = verbalizer or _GenericVerbalizer()
        self.llm_generate = llm_generate
        self.max_rounds = int(max_rounds)
        self.max_attempts = int(
            max_attempts if max_attempts is not None else max(10, self.max_rounds * 2)
        )
        self.prompt_dir = prompt_dir
        self.slicer = ProvenanceSlicer()
        self.parser = AxisQueryParser(
            alias_map=alias_map,
            checkpoint_index=checkpoint_index,
            contracts=self.contracts,
        )
        self.slicing_adapter = slicing_adapter
        self.prompt_templates = prompt_templates
        self.intervention_registry = intervention_registry
        self.intervention_alternatives = intervention_alternatives or {}
        self.observation_adapter = observation_adapter or _DefaultObservationAdapter()
        self.compatibility_mode = compatibility_mode
        self.factual_run_result = factual_run_result
        self.case_definition = case_definition
        # Shared across explained cases when provided by AxisExplanationService.
        self.noop_cache = noop_cache
        self.axis_interrogation_mode = normalize_axis_interrogation_mode(
            axis_interrogation_mode
        )
        self.axis_interrogation_policy = DEFAULT_AXIS_INTERROGATION_POLICY
        self.diagnose_hybrid_add = bool(diagnose_hybrid_add)
        self.lecturebank_id = str(lecturebank_id or "")

    def run(
        self,
        question: str,
        *,
        scripted_queries: Optional[Sequence[str]] = None,
    ) -> ExplanationResult:
        """Execute the full AXIS interrogation loop."""

        # -- Build slice ------------------------------------------------
        registry = self.registry
        if self.factual_run_result is not None:
            case_slice, registry = self._slice_from_factual_result()
        else:
            case_slice = self.slicer.backward_slice(
                self.explained_case,
                self.occurrences,
                self.provenance_relations,
                registry=self.registry,
                adapter=self.slicing_adapter,
            )

        # -- Build context ---------------------------------------------
        context = AxisQueryContext(
            explained_case=self.explained_case,
            case_slice=case_slice,
            registry=registry,
            checkpoint_index=self.checkpoint_index,
            contracts=self.contracts,
            intervention_registry=self.intervention_registry,
            alias_map=self.parser.alias_map,
            intervention_alternatives=self.intervention_alternatives,
        )

        noop_cache = self.noop_cache if self.noop_cache is not None else NoOpReplayCache()
        ledger = QueryExecutionLedger()
        resolver = AxisQueryResolver()
        executor = AxisQueryExecutor(
            context=context,
            replay_executor=self.replay_executor,
            noop_cache=noop_cache,
            stop_when_observable=True,
        )
        comparator = ReplayComparator()

        # -- Factual context -------------------------------------------
        factual_context = _build_factual_context(
            question=question,
            explained_case=self.explained_case,
            case_slice=case_slice,
            registry=registry,
            checkpoint_index=self.checkpoint_index,
            contracts=self.contracts,
            verbalizer=self.verbalizer,
            prompt_templates=self.prompt_templates,
            intervention_alternatives=self.intervention_alternatives,
            strict=not self.compatibility_mode,
            axis_interrogation_mode=self.axis_interrogation_mode,
        )

        # Factual observation (for comparison baseline)
        factual_obs = self.observation_adapter.observe_factual(
            explained_case=self.explained_case,
            case_slice=case_slice,
            registry=registry,
        )
        # Bind timing context for MCQ subject-content comparison when available.
        adapter = self.observation_adapter
        if hasattr(adapter, "case_slice"):
            adapter.case_slice = case_slice
        if hasattr(adapter, "checkpoint_index"):
            adapter.checkpoint_index = self.checkpoint_index
        if hasattr(adapter, "occurrences"):
            adapter.occurrences = self.occurrences
        if hasattr(adapter, "produced_by"):
            pb = ()
            fr = self.factual_run_result
            if fr is not None:
                pb = tuple(getattr(fr, "produced_by", None) or ())
                if not pb:
                    fs = getattr(fr, "final_state", None) or {}
                    if isinstance(fs, dict):
                        pb = tuple(fs.get("produced_by") or ())
            adapter.produced_by = pb

        # -- History ---------------------------------------------------
        history = AxisEpisodicHistory(
            question=question,
            factual_context=factual_context,
        )

        factual_records: List[FactualQueryRecord] = []
        intervention_records: List[InterventionQueryRecord] = []
        replay_comparisons: List[ReplayComparison] = []
        intermediate_texts: List[str] = [factual_context]
        warnings: list[str] = []
        interrogation_attempts: List[InterrogationAttempt] = []

        priority_fn = None
        if self.case_definition is not None and hasattr(
            self.case_definition, "intervention_artifact_priority"
        ):
            priority_fn = self.case_definition.intervention_artifact_priority

        intervention_candidates = build_intervention_candidates(
            case_slice=case_slice,
            registry=registry,
            contracts=self.contracts,
            checkpoint_index=self.checkpoint_index,
            branch_id=self.explained_case.branch_id,
            all_occurrences=self.occurrences,
            priority_fn=priority_fn,
            intervention_alternatives=self.intervention_alternatives,
        )
        candidates_text = format_intervention_candidates(intervention_candidates)
        factual_only_text = format_factual_only_artifacts(
            case_slice=case_slice,
            registry=registry,
            contracts=self.contracts,
            branch_id=self.explained_case.branch_id,
        )
        # Keep intervention_space / valid_boundaries filled for older templates.
        intervention_space_text = candidates_text
        valid_boundaries_text = "\n".join(
            f"- {c.ref}: {', '.join(c.valid_boundaries) or '(none)'}"
            for c in intervention_candidates
        ) or "(no intervenable boundaries)"

        hybrid_mode = is_hybrid_interrogation_mode(self.axis_interrogation_mode)
        hybrid_vocab = None
        hybrid_vocabulary_text = ""
        hybrid_add_diag = None
        if hybrid_mode:
            if self.diagnose_hybrid_add:
                from .hybrid_add_diagnostics import HybridAddDiagnostics

                hybrid_add_diag = HybridAddDiagnostics(
                    run_id=str(self.explained_case.run_id or ""),
                    explained_ref=str(self.explained_case.artifact_ref or ""),
                    lecturebank_id=str(getattr(self, "lecturebank_id", "") or ""),
                    document_id=(
                        f"lecturebank_{self.lecturebank_id}"
                        if getattr(self, "lecturebank_id", "")
                        else ""
                    ),
                )
            hybrid_vocab = build_hybrid_intervention_vocabulary(
                case_slice=case_slice,
                registry=registry,
                contracts=self.contracts,
                checkpoint_index=self.checkpoint_index,
                branch_id=self.explained_case.branch_id,
                explained_case=self.explained_case,
                intervention_candidates=intervention_candidates,
                intervention_alternatives=self.intervention_alternatives,
                intervention_registry=self.intervention_registry,
                all_occurrences=self.occurrences,
                replay_executor=self.replay_executor,
                run_id=self.explained_case.run_id,
                diagnostics=hybrid_add_diag,
            )
            extra_aliases: Dict[str, ArtifactRef] = dict(self.intervention_alternatives)
            for _old, news in (hybrid_vocab.replacement_options or {}).items():
                for new_ref in news:
                    extra_aliases.setdefault(str(new_ref), new_ref)
            for add_ref in hybrid_vocab.add_options:
                extra_aliases.setdefault(str(add_ref), add_ref)
            context = AxisQueryContext(
                explained_case=self.explained_case,
                case_slice=case_slice,
                registry=registry,
                checkpoint_index=self.checkpoint_index,
                contracts=self.contracts,
                intervention_registry=self.intervention_registry,
                alias_map=self.parser.alias_map,
                intervention_alternatives=extra_aliases,
            )
            hybrid_vocabulary_text = format_hybrid_vocabulary(
                vocabulary=hybrid_vocab,
                registry=registry,
                branch_id=self.explained_case.branch_id,
            )
            executor.context = context

        def _resolve_query_signature(query_text: str):
            """Best-effort canonical signature for executable-list filtering."""
            try:
                parsed_q = self.parser.parse(query_text)
                if parsed_q.operation == "DONE":
                    return None
                resolved_q = resolver.resolve(
                    parsed_q,
                    context,
                    executed_signatures=None,
                    reserve=False,
                )
                return QuerySignature.from_resolved(resolved_q)
            except Exception:
                return None

        # Lightweight case-specific interrogation goal (not an LLM causal classifier).
        try:
            from ..experiment.information_role import resolve_interrogation_goal

            interrogation_goal = resolve_interrogation_goal(question)
        except Exception:
            interrogation_goal = "general"

        def _remaining_executable_queries() -> str:
            executed_texts = set()
            for rec in factual_records:
                if getattr(rec, "generated_query", None):
                    executed_texts.add(str(rec.generated_query))
            for rec in intervention_records:
                if getattr(rec, "generated_query", None):
                    executed_texts.add(str(rec.generated_query))
                # Exclude successful resolved forms by semantic boundary and alias.
                resolved = getattr(rec, "resolved", None)
                if resolved is not None and getattr(resolved, "operation", None) in {
                    "REMOVE",
                    "ADD",
                    "WHATIF",
                }:
                    op = resolved.operation
                    tgt = resolved.target_ref
                    boundary = getattr(resolved, "boundary_raw", None)
                    alias = resolved.checkpoint_alias or resolved.checkpoint_id
                    labels = [x for x in (boundary, alias) if x]
                    if op == "WHATIF" and resolved.replacement_ref is not None:
                        for lab in labels:
                            executed_texts.add(
                                f"WHATIF({tgt}, {resolved.replacement_ref}, {lab})"
                            )
                    elif tgt is not None:
                        for lab in labels:
                            executed_texts.add(f"{op}({tgt}, {lab})")
            return format_executable_queries(
                candidates=intervention_candidates,
                explained_case=self.explained_case,
                exclude_signatures=ledger._executed_set,
                resolve_query=_resolve_query_signature,
                interrogation_goal=interrogation_goal,
                intervention_alternatives=self.intervention_alternatives,
                registry=registry,
                branch_id=self.explained_case.branch_id,
                exclude_query_texts=executed_texts,
            )

        scripted = list(scripted_queries or [])
        attempt_count = 0
        successful_round_count = 0
        termination_reason = ""
        system_prompt = _render_system(
            self.prompt_templates,
            axis_interrogation_mode=self.axis_interrogation_mode,
        )
        validation_feedback = ""
        last_intervenable_handoff = ""
        autonomous = scripted_queries is None

        try:
            from ..experiment.timing import current_timing, explanation_llm_phase
        except Exception:  # pragma: no cover
            current_timing = lambda: None  # type: ignore[assignment,misc]
            explanation_llm_phase = None  # type: ignore[assignment]
        timing = current_timing()

        def _abandon_timing_round() -> None:
            if timing is not None:
                timing.abandon_round()

        def _append_attempt(**kwargs: Any) -> InterrogationAttempt:
            counts = _opportunity_counts()
            payload = {
                # WHAT is disabled; field retained as 0 for CSV schema stability.
                "available_what_count": 0,
                "available_remove_count": counts["remove"],
                "available_add_count": counts["add"],
                "available_whatif_count": counts["whatif"],
                "axis_interrogation_mode": self.axis_interrogation_mode,
                "axis_interrogation_policy": self.axis_interrogation_policy,
            }
            payload.update(kwargs)
            att = InterrogationAttempt(attempt_index=attempt_count, **payload)
            interrogation_attempts.append(att)
            return att

        def _validation_feedback(*, status: str, detail: str, executable_queries: str = "") -> str:
            if hybrid_mode:
                return controlled_hybrid_validation_feedback(status=status, detail=detail)
            return controlled_validation_feedback(
                status=status,
                detail=detail,
                executable_queries=executable_queries,
            )

        def _opportunity_counts() -> Dict[str, int]:
            if hybrid_mode and hybrid_vocab is not None:
                return hybrid_opportunity_counts(
                    hybrid_vocab, executed_signatures=ledger._executed_set
                )
            return constrained_opportunity_counts(_remaining_executable_queries())

        while True:
            # --- Budget / last-chance checks ---------------------------------
            if attempt_count >= self.max_attempts:
                termination_reason = "attempt_budget_exhausted"
                break

            if successful_round_count >= self.max_rounds:
                termination_reason = "max_rounds"
                break

            # Script exhausted (no LLM fallback when scripted_queries was provided)
            if scripted_queries is not None and not scripted:
                termination_reason = "script_exhausted"
                break
            if scripted_queries is None and self.llm_generate is None and not scripted:
                termination_reason = "no_llm"
                break

            attempt_count += 1
            rd = InterrogationRoundRecord(round_index=successful_round_count + 1)
            interrogation_s = 0.0
            resolution_s = 0.0
            round_opened = False

            # -- Generate query ----------------------------------------
            if scripted:
                raw = scripted.pop(0)
            else:
                interrogation_prompt = _render_interrogation(
                    self.prompt_templates,
                    template_name=(
                        "interrogation_hybrid" if hybrid_mode else "interrogation"
                    ),
                    round=str(successful_round_count + 1),
                    max_rounds=str(self.max_rounds),
                    attempt=str(attempt_count),
                    max_attempts=str(self.max_attempts),
                    question=question,
                    factual_context=factual_context,
                    episodic_history=history.render(),
                    tested_queries=", ".join(
                        str(s) for s in ledger.signatures
                    )
                    or "(none)",
                    available_artifacts=", ".join(
                        str(r)
                        for r in getattr(case_slice, "relevant_artifact_refs", ())
                    ),
                    intervention_space=intervention_space_text,
                    valid_boundaries=valid_boundaries_text,
                    intervention_candidates=candidates_text,
                    factual_only_artifacts=factual_only_text,
                    executable_queries=(
                        "" if hybrid_mode else _remaining_executable_queries()
                    ),
                    hybrid_vocabulary=hybrid_vocabulary_text if hybrid_mode else "",
                    intervenable_handoff=last_intervenable_handoff,
                    validation_feedback=validation_feedback,
                )
                t_inter = time.perf_counter()
                if explanation_llm_phase is not None:
                    with explanation_llm_phase("interrogation"):
                        raw = _llm_call(
                            self.llm_generate, system_prompt, interrogation_prompt
                        )
                else:
                    raw = _llm_call(
                        self.llm_generate, system_prompt, interrogation_prompt
                    )
                interrogation_s = time.perf_counter() - t_inter

            rd.generated_query = raw
            # Clear one-shot feedback after it has been shown for this attempt.
            validation_feedback = ""

            # -- Parse -------------------------------------------------
            try:
                parsed = self.parser.parse(raw)
            except QueryValidationError as exc:
                err = str(exc)
                rd.validation_error = err
                validation_feedback = _validation_feedback(
                    status="parser_error",
                    detail=err,
                    executable_queries="" if hybrid_mode else _remaining_executable_queries(),
                )
                history.rounds.append(rd)
                intermediate_texts.append(validation_feedback)
                _append_attempt(
                    generated_query=raw,
                    validation_status="parser_error",
                    parser_error=err,
                )
                continue

            if parsed.operation == "WHAT":
                err = WHAT_DISABLED_IN_INTERROGATION
                rd.validation_error = err
                validation_feedback = _validation_feedback(
                    status="vocabulary_error",
                    detail=err,
                    executable_queries="" if hybrid_mode else _remaining_executable_queries(),
                )
                history.rounds.append(rd)
                intermediate_texts.append(validation_feedback)
                _append_attempt(
                    generated_query=raw,
                    parsed_operation="WHAT",
                    validation_status="vocabulary_error",
                    vocabulary_error=err,
                    resolution_error=err,
                )
                continue

            if parsed.operation == "DONE":
                if (
                    autonomous
                    and not intervention_records
                    and not intervention_candidates
                ):
                    warnings.append("no_valid_intervention_available")
                    termination_reason = "no_valid_intervention_available"
                else:
                    termination_reason = "done"
                history.rounds.append(rd)
                _append_attempt(
                    generated_query=raw,
                    parsed_operation="DONE",
                    validation_status="done",
                )
                break

            # Enforce: already-executed interventions must not re-enter replay.
            # Prefer filtering the prompt list; this is the final safety net.
            if parsed.operation in {"REMOVE", "ADD", "WHATIF", "REPLACE"} and autonomous:
                current_exec = "" if hybrid_mode else _remaining_executable_queries()
                try:
                    cand_sig = _resolve_query_signature(raw)
                except Exception:
                    cand_sig = None
                if cand_sig is not None and cand_sig in ledger._executed_set:
                    err = (
                        "Query is not_currently_executable: it was already executed "
                        "or is no longer in the current executable set."
                    )
                    rd.validation_error = err
                    validation_feedback = _validation_feedback(
                        status="not_currently_executable",
                        detail=err,
                        executable_queries=current_exec,
                    )
                    history.rounds.append(rd)
                    intermediate_texts.append(validation_feedback)
                    _append_attempt(
                        generated_query=raw,
                        parsed_operation=parsed.operation,
                        validation_status="not_currently_executable",
                        resolution_error=err,
                    )
                    continue

            # -- Resolve -----------------------------------------------
            t_res = time.perf_counter()
            try:
                if self.compatibility_mode:
                    try:
                        resolved = resolver.resolve(
                            parsed,
                            context,
                            executed_signatures=ledger._executed_set,
                            reserve=False,
                        )
                    except QueryValidationError:
                        resolved = self.parser.resolve(parsed)
                else:
                    resolved = resolver.resolve(
                        parsed,
                        context,
                        executed_signatures=ledger._executed_set,
                        reserve=False,
                    )
            except QueryValidationError as exc:
                err = str(exc)
                status = classify_validation_error(err)
                rd.validation_error = err
                validation_feedback = _validation_feedback(
                    status=status,
                    detail=err,
                    executable_queries="" if hybrid_mode else _remaining_executable_queries(),
                )
                history.rounds.append(rd)
                intermediate_texts.append(validation_feedback)
                att_kwargs: Dict[str, Any] = {
                    "generated_query": raw,
                    "parsed_operation": parsed.operation,
                    "validation_status": status,
                }
                if status == "duplicate_error":
                    att_kwargs["duplicate_error"] = err
                elif status == "contract_error":
                    att_kwargs["contract_error"] = err
                elif status == "vocabulary_error":
                    att_kwargs["vocabulary_error"] = err
                    att_kwargs["resolution_error"] = err
                else:
                    att_kwargs["resolution_error"] = err
                _append_attempt(**att_kwargs)
                continue
            finally:
                resolution_s = time.perf_counter() - t_res

            if hybrid_mode and hybrid_vocab is not None:
                try:
                    validate_resolved_against_hybrid_vocabulary(resolved, hybrid_vocab)
                except QueryValidationError as exc:
                    err = str(exc)
                    status = classify_validation_error(err)
                    rd.validation_error = err
                    validation_feedback = _validation_feedback(
                        status=status, detail=err
                    )
                    history.rounds.append(rd)
                    intermediate_texts.append(validation_feedback)
                    _append_attempt(
                        generated_query=raw,
                        parsed_operation=parsed.operation,
                        validation_status=status,
                        vocabulary_error=err,
                        resolution_error=err,
                    )
                    continue

            rd.resolved_query = resolved
            sig = QuerySignature.from_resolved(resolved)
            rd.signature = sig

            if timing is not None:
                timing.begin_round(
                    round_index=successful_round_count + 1,
                    intervention=str(getattr(resolved, "raw_text", None) or raw),
                    checkpoint=str(
                        getattr(resolved, "checkpoint_id", None)
                        or getattr(resolved, "checkpoint_alias", None)
                        or ""
                    ),
                    checkpoint_alias=str(
                        getattr(resolved, "checkpoint_alias", None) or ""
                    ),
                )
                timing.add_round_phase("interrogation", interrogation_s)
                timing.add_round_phase("intervention_resolution", resolution_s)
                round_opened = True

            produced_cf = False
            counted = False

            # -- Execute -----------------------------------------------
            if resolved.operation == "WHAT":
                rec = executor._exec_what(resolved)
                rd.factual_record = rec
                if rec.error:
                    rd.execution_error = rec.error
                    validation_feedback = _validation_feedback(
                        status="execution_error", detail=str(rec.error)
                    )
                    intermediate_texts.append(validation_feedback)
                    history.rounds.append(rd)
                    _append_attempt(
                        generated_query=raw,
                        parsed_operation="WHAT",
                        validation_status="execution_error",
                        execution_error=str(rec.error),
                    )
                    if round_opened:
                        _abandon_timing_round()
                    continue
                factual_records.append(rec)
                ledger.mark_executed(sig)
                successful_round_count += 1
                counted = True
                result_text = self.verbalizer.verbalize_factual_query(rec)
                rd.result_text = result_text
                intermediate_texts.append(result_text)

                # WHAT → intervention handoff when artifact is intervenable.
                allowed = ()
                if isinstance(getattr(rec, "result", None), FactualLookupResult):
                    allowed = tuple(rec.result.allowed_interventions or ())
                cand = find_candidate_for_ref(
                    intervention_candidates, resolved.target_ref
                )
                if allowed and cand is not None:
                    # For question-content goals, do not steer toward CandidateMCQ
                    # post-generation interventions after WHAT(candidate).
                    if (
                        interrogation_goal in {"question_content", "question_selection"}
                        and cand.artifact_type
                        in {"CandidateMCQ", "CandidateMCQPayload"}
                    ):
                        last_intervenable_handoff = ""
                    else:
                        if hybrid_mode and hybrid_vocab is not None:
                            last_intervenable_handoff = format_hybrid_intervenable_handoff(
                                target_ref=resolved.target_ref,
                                vocabulary=hybrid_vocab,
                                has_successful_intervention=bool(intervention_records),
                            )
                        else:
                            examples = []
                            for op in allowed:
                                for b in cand.boundaries_for(op)[:2]:
                                    if op == "REPLACE":
                                        continue
                                    examples.append(f"{op}({resolved.target_ref}, {b})")
                            last_intervenable_handoff = format_intervenable_handoff(
                                target_ref=resolved.target_ref,
                                allowed_operations=allowed,
                                valid_boundaries=cand.valid_boundaries,
                                has_successful_intervention=bool(intervention_records),
                                executable_examples=examples,
                            )
                else:
                    last_intervenable_handoff = ""
            else:
                rec = executor._exec_intervention(resolved)
                rd.intervention_record = rec

                if rec.error:
                    rd.execution_error = rec.error
                    validation_feedback = _validation_feedback(
                        status="execution_error", detail=str(rec.error)
                    )
                    intermediate_texts.append(validation_feedback)
                    history.rounds.append(rd)
                    _append_attempt(
                        generated_query=raw,
                        parsed_operation=resolved.operation,
                        validation_status="execution_error",
                        execution_error=str(rec.error),
                    )
                    if round_opened:
                        _abandon_timing_round()
                    continue

                intervention_records.append(rec)
                ledger.mark_executed(sig)
                successful_round_count += 1
                counted = True
                last_intervenable_handoff = ""

                if rec.noop_result and rec.counterfactual_result:
                    produced_cf = True
                    try:
                        t_cmp = time.perf_counter()
                        comparison = comparator.compare(
                            signature=sig,
                            factual_observation=factual_obs,
                            noop_result=rec.noop_result,
                            counterfactual_result=rec.counterfactual_result,
                            observation_adapter=self.observation_adapter,
                            explained_case=self.explained_case,
                        )
                        if timing is not None:
                            timing.add_round_phase(
                                "comparison", time.perf_counter() - t_cmp
                            )
                            timing.update_round_meta(
                                evidence_status=str(
                                    getattr(comparison, "evidence_status", "") or ""
                                ),
                            )
                        rd.comparison = comparison
                        rec.replay_comparison = comparison
                        replay_comparisons.append(comparison)
                    except Exception as exc:
                        warnings.append(f"Comparison failed: {exc}")

                result_text = self.verbalizer.verbalize_intervention(rec)
                if rd.comparison:
                    if hasattr(self.verbalizer, "verbalize_replay_comparison"):
                        result_text += "\n" + self.verbalizer.verbalize_replay_comparison(
                            rd.comparison
                        )
                    else:
                        result_text += f"\nEvidence: {rd.comparison.evidence_status}"
                rd.result_text = result_text
                intermediate_texts.append(result_text)

            # -- Intermediate synthesis --------------------------------
            inter_text = ""
            if self.llm_generate and not rd.execution_error and not rd.validation_error:
                inter_prompt = _render_intermediate(
                    self.prompt_templates,
                    question=question,
                    factual_decision=self.explained_case.outcome,
                    episodic_history=history.render(),
                    latest_query=rd.generated_query,
                    latest_result=rd.result_text,
                    latest_comparison=(
                        str(rd.comparison.evidence_status)
                        if rd.comparison
                        else "(none)"
                    ),
                )
                if inter_prompt:
                    try:
                        t_inter_syn = time.perf_counter()
                        if explanation_llm_phase is not None:
                            with explanation_llm_phase("intermediate_synthesis"):
                                inter_text = _llm_call(
                                    self.llm_generate, system_prompt, inter_prompt
                                )
                        else:
                            inter_text = _llm_call(
                                self.llm_generate, system_prompt, inter_prompt
                            )
                        if timing is not None:
                            timing.add_round_phase(
                                "intermediate_synthesis",
                                time.perf_counter() - t_inter_syn,
                            )
                    except Exception as exc:
                        inter_text = f"[Intermediate synthesis failed: {exc}]"
                        warnings.append(
                            f"Intermediate synthesis failed round "
                            f"{successful_round_count}: {exc}"
                        )
            if not inter_text and rd.result_text:
                inter_text = rd.result_text

            rd.intermediate_explanation = inter_text
            if inter_text:
                intermediate_texts.append(inter_text)

            history.rounds.append(rd)
            _append_attempt(
                generated_query=raw,
                parsed_operation=resolved.operation,
                validation_status="executed",
                executed=True,
                produced_counterfactual=produced_cf,
                counted_as_successful_round=counted,
            )
            if round_opened and timing is not None:
                timing.end_round()

        # -- Termination -----------------------------------------------
        if not termination_reason:
            if scripted_queries is not None and not scripted:
                termination_reason = "script_exhausted"
            elif attempt_count >= self.max_attempts:
                termination_reason = "attempt_budget_exhausted"
            elif successful_round_count >= self.max_rounds:
                termination_reason = "max_rounds"
            else:
                termination_reason = "no_llm"

        if (
            not intervention_records
            and not intervention_candidates
            and termination_reason
            in {"done", "max_rounds", "attempt_budget_exhausted", "script_exhausted"}
        ):
            if "no_valid_intervention_available" not in warnings:
                warnings.append("no_valid_intervention_available")

        # -- Final synthesis -------------------------------------------
        evidence_ledger = ExplanationEvidenceLedger(
            factual_context=factual_context,
            factual_records=factual_records,
            intervention_records=intervention_records,
            comparisons=replay_comparisons,
            intermediate_explanations=[
                rd.intermediate_explanation
                for rd in history.rounds
                if rd.intermediate_explanation
            ],
        )

        t_final = time.perf_counter()
        if explanation_llm_phase is not None:
            with explanation_llm_phase("final_synthesis"):
                final_explanation = self._final_synthesis(
                    question=question,
                    factual_context=factual_context,
                    history=history,
                    evidence_ledger=evidence_ledger,
                    case_slice=case_slice,
                    factual_records=factual_records,
                    intervention_records=intervention_records,
                    system_prompt=system_prompt,
                    warnings=warnings,
                )
        else:
            final_explanation = self._final_synthesis(
                question=question,
                factual_context=factual_context,
                history=history,
                evidence_ledger=evidence_ledger,
                case_slice=case_slice,
                factual_records=factual_records,
                intervention_records=intervention_records,
                system_prompt=system_prompt,
                warnings=warnings,
            )
        if timing is not None:
            timing.record_final_synthesis(time.perf_counter() - t_final)

        return ExplanationResult(
            explained_case=self.explained_case,
            case_slice=case_slice,
            factual_context=factual_context,
            factual_records=factual_records,
            intervention_records=intervention_records,
            replay_comparisons=replay_comparisons,
            round_records=list(history.rounds),
            intermediate_texts=intermediate_texts,
            final_explanation=final_explanation,
            rounds=successful_round_count,
            successful_round_count=successful_round_count,
            attempt_count=attempt_count,
            interrogation_attempts=interrogation_attempts,
            termination_reason=termination_reason,
            executed_signatures=ledger.signatures,
            warnings=tuple(warnings),
            valid_intervention_candidate_count=len(intervention_candidates),
            axis_interrogation_mode=self.axis_interrogation_mode,
            axis_interrogation_policy=self.axis_interrogation_policy,
            noop_cache_stats=dict(noop_cache.stats())
            if hasattr(noop_cache, "stats")
            else {},
            hybrid_add_diagnostics=(
                hybrid_add_diag.to_payload() if hybrid_add_diag is not None else None
            ),
        )

    def _slice_from_factual_result(self) -> Tuple[CaseSlice, ArtifactRegistry]:
        fr = self.factual_run_result
        if getattr(fr, "run_id", None) != self.explained_case.run_id:
            raise CaseConfigurationError(
                f"factual_run_result.run_id {getattr(fr, 'run_id', None)!r} "
                f"does not match explained_case.run_id {self.explained_case.run_id!r}"
            )
        branch_id = getattr(fr, "branch_id", "factual")
        if not self.compatibility_mode and branch_id != "factual":
            raise CaseConfigurationError(
                f"factual_run_result.branch_id must be 'factual', got {branch_id!r}"
            )
        fact_reg = getattr(fr, "artifact_registry", None)
        if fact_reg is not None and self.registry is not None:
            self._assert_registry_agreement(self.registry, fact_reg)
        registry = fact_reg or self.registry
        case_slice = self.slicer.slice_run_result(
            self.explained_case,
            fr,
            registry=registry,
            adapter=self.slicing_adapter,
            strict=True,
        )
        return case_slice, registry

    @staticmethod
    def _assert_registry_agreement(
        supplied: ArtifactRegistry, factual: ArtifactRegistry
    ) -> None:
        for env in (getattr(factual, "catalog", {}) or {}).values():
            ref = getattr(env, "ref", None)
            if ref is None:
                continue
            other = supplied.get_by_ref(ref, getattr(env, "branch_id", "factual"))
            if other is None:
                other = supplied.get_by_ref(ref)
            if other is None:
                continue
            if not payloads_equivalent(env.payload, other.payload):
                raise CaseConfigurationError(
                    f"Conflicting factual artifact payload for {ref}"
                )

    def _final_synthesis(
        self,
        *,
        question: str,
        factual_context: str,
        history: AxisEpisodicHistory,
        evidence_ledger: ExplanationEvidenceLedger,
        case_slice: CaseSlice,
        factual_records: List[FactualQueryRecord],
        intervention_records: List[InterventionQueryRecord],
        system_prompt: str,
        warnings: list[str],
    ) -> str:
        """Generate final explanation via LLM or mechanical fallback."""
        # Build evidence ledger text
        evidence_parts: list[str] = []
        for rec in factual_records:
            evidence_parts.append(f"WHAT: {self.verbalizer.verbalize_factual_query(rec)}")
        for comp in evidence_ledger.comparisons:
            if hasattr(self.verbalizer, "verbalize_replay_comparison"):
                evidence_parts.append(self.verbalizer.verbalize_replay_comparison(comp))
            else:
                evidence_parts.append(f"Intervention: {comp.evidence_status}")
        evidence_text = "\n".join(evidence_parts) if evidence_parts else "(no evidence collected)"

        final_prompt = _render_final(
            self.prompt_templates,
            question=question,
            factual_context=factual_context,
            episodic_history=history.render(),
            evidence_ledger=evidence_text,
        )

        if final_prompt and self.llm_generate:
            try:
                text = _llm_call(self.llm_generate, system_prompt, final_prompt)
                return _sanitize_raw_axis_summary(
                    text, comparisons=list(evidence_ledger.comparisons or ())
                )
            except Exception as exc:
                warnings.append(f"Final synthesis LLM failed: {exc}")

        # Mechanical fallback
        parts = [
            f"Question: {question}",
            self.verbalizer.verbalize_decision(self.explained_case),
            self.verbalizer.verbalize_slice(case_slice),
        ]
        for rec in factual_records:
            parts.append(self.verbalizer.verbalize_factual_query(rec))
        for rec in intervention_records:
            parts.append(self.verbalizer.verbalize_intervention(rec))
        parts.append(
            "Only facts supported by the provenance slice, factual query records, "
            "and executed replay records are included. No unexecuted counterfactuals."
        )
        return _sanitize_raw_axis_summary(
            "\n".join(parts),
            comparisons=list(evidence_ledger.comparisons or ()),
        )

    # -- Legacy compatibility methods ----------------------------------

    def _exec_what(self, resolved: ResolvedQuery) -> FactualQueryRecord:
        """Legacy compatibility."""
        assert resolved.target_ref is not None
        env = self.registry.get_by_ref(resolved.target_ref)
        if env is None:
            return FactualQueryRecord(
                generated_query=resolved.raw_text,
                resolved=resolved,
                error=f"Unknown artifact {resolved.target_ref}",
            )
        return FactualQueryRecord(
            generated_query=resolved.raw_text,
            resolved=resolved,
            result=self.verbalizer.verbalize_artifact(env),
        )

    def _exec_intervention(
        self, resolved: ResolvedQuery
    ) -> InterventionQueryRecord:
        """Legacy compatibility."""
        assert resolved.target_ref is not None
        assert resolved.checkpoint_alias is not None
        try:
            ckpt = self.checkpoint_index.get_alias(resolved.checkpoint_alias)
        except Exception as exc:
            return InterventionQueryRecord(
                generated_query=resolved.raw_text,
                resolved=resolved,
                error=str(exc),
            )
        noop = self.replay_executor.fork_noop(
            self.explained_case.run_id,
            BoundarySelector.by_alias(ckpt.alias),
        )
        if resolved.operation == "REMOVE":
            intervention = ArtifactIntervention(
                operation="REMOVE", artifact=resolved.target_ref, checkpoint=ckpt,
            )
        elif resolved.operation == "ADD":
            intervention = ArtifactIntervention(
                operation="ADD", artifact=resolved.target_ref, checkpoint=ckpt,
            )
        elif resolved.operation == "WHATIF":
            assert resolved.replacement_ref is not None
            intervention = ArtifactIntervention(
                operation="REPLACE",
                artifact=(resolved.target_ref, resolved.replacement_ref),
                checkpoint=ckpt,
            )
        else:
            return InterventionQueryRecord(
                generated_query=resolved.raw_text, resolved=resolved,
                error=f"Unsupported intervention {resolved.operation}",
            )
        try:
            cf = self.replay_executor.fork_with_intervention(
                self.explained_case.run_id, intervention,
                registry=self.registry, intervention_registry=self.intervention_registry,
            )
            resolver = self.replay_executor.get_resolver(cf.branch_id)
            comparison = {
                "operation": resolved.operation,
                "target": str(resolved.target_ref),
                "checkpoint": ckpt.alias,
                "noop_branch": noop.branch_id,
                "counterfactual_branch": cf.branch_id,
            }
            if resolved.operation == "REMOVE":
                comparison["target_available_in_counterfactual"] = resolver.is_available(
                    resolved.target_ref
                )
            return InterventionQueryRecord(
                generated_query=resolved.raw_text, resolved=resolved,
                noop_branch_id=noop.branch_id, counterfactual_branch_id=cf.branch_id,
                comparison=comparison, noop_result=noop, counterfactual_result=cf,
            )
        except Exception as exc:
            return InterventionQueryRecord(
                generated_query=resolved.raw_text, resolved=resolved,
                noop_branch_id=noop.branch_id, error=str(exc), noop_result=noop,
            )

    def _synthesize(
        self, question: str, case_slice: CaseSlice,
        factual: List[FactualQueryRecord],
        interventions: List[InterventionQueryRecord],
    ) -> str:
        """Legacy mechanical synthesis."""
        parts = [
            f"Question: {question}",
            self.verbalizer.verbalize_decision(self.explained_case),
            self.verbalizer.verbalize_slice(case_slice),
        ]
        for rec in factual:
            parts.append(self.verbalizer.verbalize_factual_query(rec))
        for rec in interventions:
            parts.append(self.verbalizer.verbalize_intervention(rec))
        parts.append(
            "Only facts supported by the provenance slice, factual query records, "
            "and executed replay records are included. No unexecuted counterfactuals."
        )
        return "\n".join(parts)
