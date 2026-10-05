"""MCQ-specific verbalization for AXIS explanation context."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ...artifacts.models import ArtifactEnvelope, ArtifactRef
from ...provenance.models import OccurrenceRecord
from ...replay.checkpoints import CheckpointIndex, CheckpointRecord
from ...replay.interventions import InterventionContractRegistry
from ...axis.models import (
    AxisPromptTemplates,
    ExplainedCase,
    FactualLookupResult,
    FactualQueryRecord,
    InterventionQueryRecord,
)
from ...axis.slicing import CaseSlice


class McqVerbalizer:
    """MCQ-oriented verbalizer for artifacts, provenance, and interventions."""

    def __init__(
        self,
        *,
        prompt_dir: Any = None,
        prompt_templates: Optional[AxisPromptTemplates] = None,
    ) -> None:
        self._prompt_dir = prompt_dir
        self._templates = prompt_templates

    def verbalize_artifact(self, envelope: ArtifactEnvelope) -> str:
        payload = envelope.payload
        a_type = envelope.artifact_type
        ref = envelope.ref

        if a_type == "EvidenceChunk":
            content = getattr(payload, "content", None) or ""
            section = getattr(payload, "section", "")
            page = getattr(payload, "page", "")
            meta_parts = []
            if section:
                meta_parts.append(f"section={section}")
            if page:
                meta_parts.append(f"page={page}")
            meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
            text = str(content)
            if len(text) > 500:
                text = text[:500] + " [truncated]"
            return f"EvidenceChunk {ref}{meta}: {text}"

        if a_type == "CandidateMCQ":
            stem = getattr(payload, "stem", "") or ""
            options = getattr(payload, "options", None)
            correct = getattr(payload, "correct_option", None)
            q_id = getattr(payload, "q_id", "")
            parts = [f"CandidateMCQ {ref}"]
            if q_id:
                parts.append(f"q_id={q_id}")
            if stem:
                parts.append(f"stem={stem[:200]}")
            if options:
                parts.append(f"options={options}")
            if correct:
                parts.append(f"correct={correct}")
            src_refs = getattr(payload, "source_refs", None)
            if src_refs:
                parts.append(f"sources={src_refs}")
            return "; ".join(parts)

        if a_type == "MCQBlueprint":
            lo = getattr(payload, "learning_objective", "") or getattr(payload, "lo", "")
            bp_id = getattr(payload, "blueprint_id", "")
            atoms = getattr(payload, "target_atom_ids", None)
            ev_ids = getattr(payload, "evidence_chunk_ids", None) or getattr(
                payload, "evidence_ids", None
            )
            parts = [f"MCQBlueprint {ref}"]
            if bp_id:
                parts.append(f"id={bp_id}")
            if lo:
                parts.append(f"LO={lo[:150]}")
            if atoms:
                parts.append(f"atoms={atoms}")
            if ev_ids:
                parts.append(f"evidence={ev_ids}")
            return "; ".join(parts)

        # Reports: EvidenceCheckReport, CrossCoVeReport, etc.
        if "Report" in a_type or "Decision" in a_type or "Assessment" in a_type:
            parts = [f"{a_type} {ref}"]
            for field_name in ("result", "outcome", "flags", "majority", "confidence", "decision", "reasons", "required_fixes"):
                val = getattr(payload, field_name, None)
                if val is not None:
                    parts.append(f"{field_name}={val}")
            return "; ".join(parts)

        # Fallback
        stem = getattr(payload, "stem", None)
        if stem:
            return f"{a_type} {ref}: {stem[:200]}"
        content = getattr(payload, "content", None)
        if content:
            return f"{a_type} {ref}: {str(content)[:200]}"
        return f"{a_type} {ref} on branch {envelope.branch_id}"

    def verbalize_occurrence(self, occurrence: OccurrenceRecord) -> str:
        return (
            f"Occurrence {occurrence.labeled_id} "
            f"(id={occurrence.occurrence_id}, branch={occurrence.branch_id}) "
            f"produced {len(occurrence.produced_artifacts)} artifact(s)"
        )

    def verbalize_checkpoint(self, checkpoint: CheckpointRecord) -> str:
        return (
            f"Checkpoint {checkpoint.alias} "
            f"(thread={checkpoint.thread_id}, id={checkpoint.checkpoint_id})"
        )

    def _render_explained_case(self, explained_case: ExplainedCase) -> str:
        parts = [
            f"Candidate: {explained_case.artifact_ref}",
            f"Assessment: {explained_case.assessment_ref}",
            f"Decision: {explained_case.decision_ref}",
            f"Outcome: {explained_case.outcome}",
            f"Run: {explained_case.run_id}",
            f"Branch: {explained_case.branch_id}",
        ]
        return "\n".join(parts)

    def _render_criteria(self, case_slice: CaseSlice) -> str:
        criterion_values = getattr(case_slice, "criterion_values", ())
        if not criterion_values:
            return "(no criteria recorded)"
        parts = ["Criteria (Γα):"]
        decision_relevant = [
            c for c in criterion_values if getattr(c, "decision_relevant", False)
        ]
        diagnostic = [
            c for c in criterion_values if not getattr(c, "decision_relevant", False)
        ]
        if decision_relevant:
            parts.append("Decision-relevant:")
            for cv in decision_relevant:
                parts.append(self._format_criterion(cv))
        if diagnostic:
            parts.append("Diagnostic:")
            for cv in diagnostic:
                parts.append(self._format_criterion(cv))
        return "\n".join(parts)

    @staticmethod
    def _format_criterion(cv: Any) -> str:
        name = getattr(cv, "name", "?")
        val = getattr(cv, "value", "?")
        label = getattr(cv, "label", "")
        line = f"  {name}: {val}"
        if label:
            line += f" ({label})"
        return line

    def _render_decision_details(self, case_slice: CaseSlice) -> str:
        parts = [f"Decision: {getattr(case_slice, 'decision', '')}"]
        reasons = getattr(case_slice, "decision_reasons", ())
        fixes = getattr(case_slice, "required_fixes", ())
        if reasons:
            parts.append("Reasons:")
            for r in reasons:
                parts.append(f"  - {r}")
        if fixes:
            parts.append("Required fixes:")
            for f in fixes:
                parts.append(f"  - {f}")
        return "\n".join(parts)

    def _render_artifacts(
        self,
        refs: Sequence[ArtifactRef],
        registry: Any = None,
        *,
        branch_id: str = "factual",
    ) -> str:
        if not refs:
            return "(no factual artifacts)"
        parts: list[str] = []
        for ref in refs:
            env = None
            if registry is not None:
                env = registry.get_by_ref(ref, branch_id=branch_id) or registry.get_by_ref(
                    ref
                )
            if env is not None:
                parts.append(self.verbalize_artifact(env))
            else:
                parts.append(f"{ref}")
        return "\n".join(parts)

    def _render_occurrences(self, case_slice: CaseSlice) -> str:
        occs = getattr(case_slice, "relevant_occurrences", ())
        if not occs:
            return "(no occurrences)"
        parts = []
        for occ in occs:
            labeled = getattr(occ, "labeled_id", "?")
            occ_id = getattr(occ, "occurrence_id", "?")
            node_id = getattr(occ, "node_id", "?")
            superstep = getattr(occ, "superstep", "?")
            ckpt_before = getattr(occ, "checkpoint_before", "?")
            ckpt_after = getattr(occ, "checkpoint_after", "?")
            parts.append(
                f"  {labeled} (id={occ_id}, node={node_id}, "
                f"superstep={superstep}, before={ckpt_before}, after={ckpt_after})"
            )
        return "\n".join(parts)

    def _render_relations(self, case_slice: CaseSlice) -> str:
        rels = getattr(case_slice, "relevant_relations", ())
        if not rels:
            return "(no provenance relations)"
        parts = []
        for rel in rels:
            prod = getattr(rel, "producer_occurrence_id", "?")
            art = getattr(rel, "artifact_ref", "?")
            sem = getattr(rel, "relation", getattr(rel, "semantic_relation", "?"))
            cons = getattr(rel, "consumer_occurrence_id", "?")
            parts.append(f"  {prod} -- {art} [{sem}] --> {cons}")
        return "\n".join(parts)

    def _render_revision_lineage(self, case_slice: CaseSlice) -> str:
        derivs = getattr(case_slice, "relevant_derivations", ())
        if not derivs:
            return "(no revision lineage)"
        parts = []
        for d in derivs:
            target = getattr(d, "new_artifact_ref", "?")
            source = getattr(d, "previous_artifact_ref", "?")
            parts.append(f"  {target} derived from {source}")
        return "\n".join(parts)

    def _render_source_boundaries(self, case_slice: CaseSlice) -> str:
        boundaries = getattr(case_slice, "source_boundaries", ())
        if not boundaries:
            return "(no source boundaries)"
        return "\n".join(f"  {b}" for b in boundaries)

    def _render_controlled_alternatives(
        self,
        alternatives: Mapping[str, ArtifactRef],
        registry: Any = None,
    ) -> str:
        if not alternatives:
            return "(no controlled alternatives)"
        parts = []
        for alias, ref in alternatives.items():
            env = None
            if registry is not None:
                env = registry.get_by_ref(ref) or registry.get_by_ref(ref, branch_id="factual")
            a_type = env.artifact_type if env is not None else "?"
            summary = self.verbalize_artifact(env) if env is not None else str(ref)
            parts.append(
                f"Alternative alias: {alias}\n"
                f"Exact ref: {ref}\n"
                f"Type: {a_type}\n"
                f"Source: controlled alternative\n"
                f"Content: {summary}"
            )
        return "\n\n".join(parts)

    def factual_context_sections(
        self,
        *,
        explained_case: ExplainedCase,
        case_slice: CaseSlice,
        registry: Any,
        checkpoint_index: CheckpointIndex,
        contracts: InterventionContractRegistry,
        intervention_alternatives: Mapping[str, ArtifactRef] = None,
    ) -> Mapping[str, str]:
        alternatives = intervention_alternatives or {}
        criteria = self._render_criteria(case_slice)
        if getattr(case_slice, "criterion_values", ()) and not criteria.strip():
            raise ValueError("criteria section could not be built from recorded values")
        artifacts = self._render_artifacts(
            getattr(case_slice, "relevant_artifact_refs", ()),
            registry,
            branch_id=explained_case.branch_id,
        )
        return {
            "explained_case": self._render_explained_case(explained_case),
            "criteria": criteria,
            "decision_details": self._render_decision_details(case_slice),
            "factual_artifacts": artifacts,
            "occurrences": self._render_occurrences(case_slice),
            "provenance_relations": self._render_relations(case_slice),
            "revision_lineage": self._render_revision_lineage(case_slice),
            "source_boundaries": self._render_source_boundaries(case_slice),
            "intervention_space": self.verbalize_intervention_space(
                getattr(case_slice, "relevant_artifact_refs", ()),
                contracts,
                registry=registry,
            )
            or "(no intervenable artifacts)",
            "controlled_alternatives": self._render_controlled_alternatives(
                alternatives, registry
            ),
            "valid_boundaries": self.verbalize_boundaries(
                case_slice, checkpoint_index, strict=True
            ),
        }

    def verbalize_slice(self, case_slice: CaseSlice) -> str:
        """Full structured factual context from CaseSlice."""
        parts: list[str] = []
        parts.append("=== EXPLAINED CASE ===")
        parts.append(self._render_explained_case(case_slice.explained_case))
        parts.append("")
        parts.append("=== CRITERIA (Γα) ===")
        parts.append(self._render_criteria(case_slice))
        parts.append("")
        parts.append("=== DECISION DETAILS ===")
        parts.append(self._render_decision_details(case_slice))
        parts.append("")
        parts.append("=== FACTUAL CASE ARTIFACTS ===")
        parts.append(
            self._render_artifacts(getattr(case_slice, "relevant_artifact_refs", ()))
        )
        parts.append("")
        parts.append("=== OCCURRENCES ===")
        parts.append(self._render_occurrences(case_slice))
        parts.append("")
        parts.append("=== PROVENANCE RELATIONS ===")
        parts.append(self._render_relations(case_slice))
        parts.append("")
        parts.append("=== REVISION LINEAGE ===")
        parts.append(self._render_revision_lineage(case_slice))
        warnings = getattr(case_slice, "warnings", ())
        if warnings:
            parts.append("")
            parts.append("=== WARNINGS ===")
            for w in warnings:
                parts.append(f"  {w}")
        parts.append("")
        parts.append("=== SOURCE BOUNDARIES ===")
        parts.append(self._render_source_boundaries(case_slice))
        return "\n".join(parts)

    def verbalize_decision(self, case: ExplainedCase) -> str:
        return (
            f"Decision {case.decision_ref} for candidate {case.artifact_ref}: "
            f"{case.outcome}"
        )

    def verbalize_factual_query(self, record: FactualQueryRecord) -> str:
        """Verbalize a structured FactualLookupResult."""
        if record.error:
            return f"Factual query failed: {record.error}"
        result = record.result
        if not isinstance(result, FactualLookupResult):
            return f"WHAT result for {record.resolved.target_ref}: {result!r}"

        parts = [f"Artifact {result.target_ref} is a {result.artifact_type}."]
        if result.producer_occurrence_id:
            parts.append(f"Produced by {result.producer_occurrence_id}.")
        if result.consumer_relations:
            for rel in result.consumer_relations:
                sem = getattr(rel, "relation", getattr(rel, "semantic_relation", "?"))
                cons = getattr(rel, "consumer_occurrence_id", "?")
                parts.append(f"Consumed by {cons} as {sem}.")
        if result.allowed_interventions:
            parts.append(
                f"Allowed interventions: {', '.join(result.allowed_interventions)}."
            )
        if result.derivations:
            for d in result.derivations:
                target = getattr(d, "new_artifact_ref", "?")
                source = getattr(d, "previous_artifact_ref", "?")
                parts.append(f"Lineage: {target} derived from {source}.")
        if result.envelope:
            parts.append(f"Content: {self.verbalize_artifact(result.envelope)}")
        return " ".join(parts)

    def verbalize_intervention(self, record: InterventionQueryRecord) -> str:
        """Verbalize intervention result without causal overclaiming."""
        if record.error:
            return f"Intervention failed: {record.error}"
        comp = record.comparison
        parts = [
            f"Operation: {comp.get('operation', record.resolved.operation)}",
            f"Target: {comp.get('target', record.resolved.target_ref)}",
            f"Checkpoint: {comp.get('checkpoint', record.resolved.checkpoint_alias)}",
            f"No-op branch: {comp.get('noop_branch', record.noop_branch_id)}",
            f"Counterfactual branch: {comp.get('counterfactual_branch', record.counterfactual_branch_id)}",
        ]
        if "target_available_in_counterfactual" in comp:
            parts.append(
                f"Target available in counterfactual: {comp['target_available_in_counterfactual']}"
            )
        if "resolver_resolves_old_to_new" in comp:
            parts.append(
                f"Old resolves to new: {comp['resolver_resolves_old_to_new']}"
            )
        if record.error:
            parts.append(f"Execution error: {record.error}")
        return ". ".join(parts) + "."

    def verbalize_intervention_space(
        self,
        refs: Sequence[ArtifactRef],
        contracts: InterventionContractRegistry,
        registry: Any = None,
    ) -> str:
        """Verbalize allowed intervention operations per artifact."""
        parts: list[str] = []
        for ref in refs:
            a_type = ""
            if registry:
                env = getattr(registry, "get_by_ref", lambda r, **kw: None)(ref)
                if env:
                    a_type = env.artifact_type
            contract = contracts.get(a_type) if a_type else None
            if contract is None:
                continue
            ops = [o for o in ("REMOVE", "ADD", "REPLACE") if contract.allows(o)]
            if not ops:
                continue
            line = f"{ref} [{a_type}] — Allowed: {', '.join(ops)}"
            compat = getattr(contract, "replacement_compatible_types", ())
            if compat:
                line += f" — Replacement types: {', '.join(compat)}"
            parts.append(line)
        return "\n".join(parts)

    def verbalize_boundaries(
        self,
        case_slice: CaseSlice,
        checkpoint_index: CheckpointIndex,
        *,
        strict: bool = False,
    ) -> str:
        """Verbalize valid factual checkpoint boundaries with readable selectors.

        ``BoundarySelector.before/after_occurrence`` expects the **node-local**
        ``occurrence_index`` (nth execution of this node), not the global
        ``execution_index``.
        """
        parts: list[str] = []
        occs = getattr(case_slice, "relevant_occurrences", ())
        seen_selectors: set[str] = set()
        for occ in occs:
            labeled = getattr(occ, "labeled_id", None)
            if not labeled:
                continue
            node_id = getattr(occ, "node_id", "") or ""
            occ_idx = getattr(occ, "occurrence_index", None)
            if occ_idx is None and isinstance(labeled, str) and "#" in labeled:
                try:
                    occ_idx = int(labeled.rsplit("#", 1)[-1])
                except ValueError:
                    occ_idx = None
            if occ_idx is None:
                if strict:
                    raise ValueError(
                        f"Cannot determine occurrence_index for boundary of {labeled!r}"
                    )
                continue
            for prefix in ("before", "after"):
                selector_text = f"{prefix} {labeled}"
                if selector_text in seen_selectors:
                    continue
                try:
                    from ...replay.checkpoints import BoundarySelector as BS

                    if prefix == "before":
                        sel = BS.before_occurrence(node_id, int(occ_idx))
                    else:
                        sel = BS.after_occurrence(node_id, int(occ_idx))
                    rec = checkpoint_index.resolve(sel)
                except Exception as exc:
                    # Optional: after last occurrence / missing mapping — skip explicitly.
                    # Never advertise an unresolved boundary.
                    if strict and not isinstance(
                        exc, (KeyError, ValueError, LookupError)
                    ):
                        raise ValueError(
                            f"Unexpected failure resolving boundary "
                            f"{selector_text!r}: {exc}"
                        ) from exc
                    continue
                seen_selectors.add(selector_text)
                parts.append(f'  "{selector_text}" → {rec.alias}')
        return "\n".join(parts) if parts else "(no boundaries available)"

    def verbalize_replay_comparison(self, comparison: Any) -> str:
        """Verbalize a ReplayComparison showing factual/noop/cf separately."""
        parts: list[str] = []
        f_obs = comparison.factual
        n_obs = comparison.noop
        c_obs = comparison.counterfactual

        parts.append("Factual:")
        parts.append(f"  Candidate: {f_obs.subject_ref}")
        parts.append(f"  Decision: {f_obs.decision}")
        parts.append("")
        parts.append("No-op control:")
        parts.append(f"  Candidate: {n_obs.subject_ref}")
        parts.append(f"  Decision: {n_obs.decision}")
        parts.append("")
        parts.append("Counterfactual:")
        parts.append(f"  Candidate: {c_obs.subject_ref}")
        parts.append(f"  Decision: {c_obs.decision}")
        parts.append("")

        if comparison.decision_control_stable is not None:
            if comparison.decision_control_stable:
                parts.append("Control stability: Factual and no-op decision matched.")
            else:
                parts.append(
                    "Control stability: The no-op replay changed the decision. "
                    "Counterfactual decision difference cannot be reliably attributed "
                    "to the tested artifact in this single replay."
                )

        # Criterion deltas (only changed ones)
        changed = [d for d in comparison.criterion_deltas if d.counterfactual_differs_from_noop]
        if changed:
            parts.append("")
            parts.append("Changed criteria (noop → counterfactual):")
            for d in changed:
                parts.append(f"  {d.name}: {d.noop_value} → {d.counterfactual_value}")

        # Authoritative question-content comparison (must not contradict decision status).
        scd = getattr(comparison, "subject_content_delta", None)
        if scd is not None:
            det = dict(getattr(scd, "details", None) or {})
            status = str(
                getattr(scd, "effect_status", None)
                or det.get("question_content_effect_status")
                or ""
            )
            parts.append("")
            parts.append("Question-content comparison (authoritative for MCQ wording):")
            parts.append(f"  status: {status}")
            for key in (
                "stem_effect",
                "correct_answer_effect",
                "distractor_effect",
                "answer_explanation_effect",
                "changed_distractor_count",
            ):
                if key in det:
                    parts.append(f"  {key}: {det.get(key)}")
            if status in {
                "question_content_changed",
                "question_content_partially_supported",
            }:
                parts.append(
                    "  Note: question content differs under the intervention; "
                    "do not summarize as 'no demonstrated effect on the question'. "
                    "You may still report no demonstrated *decision* effect when "
                    "the final ACCEPT/REVISE/REJECT is unchanged."
                )
            elif status == "question_content_unchanged":
                parts.append(
                    "  Note: core question content (stem/correct/distractors) unchanged."
                )

        parts.append("")
        # Scope evidence_status: decision/criteria/downstream, not MCQ wording.
        parts.append(
            f"Evidence status (decision/criteria/downstream): {comparison.evidence_status}"
        )
        for note in comparison.notes:
            parts.append(f"Note: {note}")
        return "\n".join(parts)

    def load_prompt_template(self, path: str, **kwargs: Any) -> str:
        """Load a prompt template file and format with kwargs (legacy compat)."""
        from pathlib import Path as P
        text = P(path).read_text(encoding="utf-8")
        try:
            return text.format(**kwargs)
        except KeyError:
            return text
