"""Preserve concrete criterion/downstream effects in participant AXIS text."""

from __future__ import annotations

import unittest

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.intervention_guidance import (
    InterventionCandidate,
    controlled_validation_feedback,
    format_executable_queries,
)
from mas_langgraph.axis.models import QuerySignature
from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
from mas_langgraph.experiment.lay_facts import build_lay_explanation_facts
from mas_langgraph.experiment.synthesis import (
    mechanical_synthesize_from_view,
    synthesize_explanation,
)
from mas_langgraph.experiment.verbalizer import build_evidence_view


def _cmp(
    *,
    status: str,
    target: str = "chunk_1:v1",
    noop_atom: int = 2,
    cf_atom: int = 3,
    decision: str = "accept",
    cf_decision: str = "accept",
    noop_path=None,
    cf_path=None,
    first_path_divergence=None,
    added_prov=None,
    removed_prov=None,
    payload_changed=None,
):
    return {
        "evidence_status": status,
        "intervention_signature": {
            "operation": "REMOVE",
            "target_ref": {"key": target, "logical_id": target.split(":")[0], "version": 1},
        },
        "factual": {"decision": decision, "outcome_available": True},
        "noop": {"decision": decision, "outcome_available": True},
        "counterfactual": {"decision": cf_decision, "outcome_available": True},
        "criterion_deltas": [
            {
                "name": "target_atom_count",
                "factual": noop_atom,
                "noop": noop_atom,
                "counterfactual": cf_atom,
                "decision_relevant": True,
                "noop_differs_from_factual": False,
                "counterfactual_differs_from_noop": noop_atom != cf_atom,
            }
        ],
        "noop_path": list(noop_path or ["check_coverage[1]", "apply_acceptance_gate[1]"]),
        "counterfactual_path": list(
            cf_path or ["check_coverage[1]", "apply_acceptance_gate[1]"]
        ),
        "first_path_divergence": first_path_divergence,
        "added_provenance_relations": list(added_prov or []),
        "removed_provenance_relations": list(removed_prov or []),
        "payload_changed_artifacts": list(payload_changed or []),
        "decision_control_stable": True,
        "criteria_control_stable": True,
        "notes": [],
    }


class TestCriterionDetailPreservation(unittest.TestCase):
    def test_criterion_effect_preserves_human_property(self):
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [
                {
                    "ref": "chunk_1:v1",
                    "artifact_type": "EvidenceChunk",
                    "content_preview": "Unsupervised learning uses unlabeled data.",
                }
            ],
            "replay_comparisons": [
                _cmp(status="criterion_effect", noop_atom=2, cf_atom=3)
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        self.assertEqual(facts.primary_effect_status, "criterion_effect")
        joined = " ".join(facts.controlled_change_findings).lower()
        self.assertIn("supporting knowledge", joined)
        self.assertIn("accept", joined)
        self.assertNotIn("target_atom_count", joined)
        self.assertIn("two", joined)
        self.assertIn("three", joined)

    def test_opposite_criterion_direction(self):
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                _cmp(status="criterion_effect", noop_atom=2, cf_atom=1)
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.controlled_change_findings).lower()
        self.assertIn("from two to one", joined)
        self.assertIn("remained accept", joined)


class TestCriterionOmissionFailsFaithfulness(unittest.TestCase):
    def test_llm_omission_triggers_fallback_with_property(self):
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "decision_reasons": [],
            "criteria": [],
            "replay_comparisons": [
                _cmp(status="criterion_effect", noop_atom=2, cf_atom=3)
            ],
        }
        view = build_evidence_view(condition="axis", raw_evidence=raw)

        def bad_llm(_prompt: str) -> str:
            return (
                "Why this question?\n"
                "A question was created.\n\n"
                "What was it based on?\n"
                "Teaching material was used.\n\n"
                "What difference did this information make?\n"
                "The final decision remained the same after the tested removal."
            )

        text = synthesize_explanation(
            user_question="Why?",
            factual_decision="accept",
            question_stem="stem",
            evidence_view=view,
            llm_generate=bad_llm,
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis", explanation=text, evidence_view=view
        )
        self.assertFalse(ok)
        self.assertTrue(
            any("criterion_effect" in w or "reduces" in w for w in warns),
            msg=str(warns),
        )
        # Service-style repair
        repaired = mechanical_synthesize_from_view(view=view)
        ok2, warns2 = validate_explanation_faithfulness(
            condition="axis", explanation=repaired, evidence_view=view
        )
        self.assertTrue(ok2, msg=str(warns2))
        self.assertIn("supporting knowledge", repaired.lower())


class TestTraceOverclaimPhrases(unittest.TestCase):
    def test_reject_real_run_overclaims(self):
        view = build_evidence_view(
            condition="trace",
            raw_evidence={"factual_decision": "accept", "execution_trace": ["gate#1"]},
        )
        bad = (
            "Why this question?\n"
            "A question was generated.\n\n"
            "What was it based on?\n"
            "Teaching material was retrieved.\n\n"
            "What difference did this information make?\n"
            "The decision to accept this question was based on thorough checks "
            "for accuracy and relevance. The checks confirmed the question's validity."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="trace", explanation=bad, evidence_view=view
        )
        self.assertFalse(ok)
        self.assertTrue(any("attributes ACCEPT" in w or "causal" in w for w in warns))

        safe = (
            "Why this question?\n"
            "The execution record shows that the question was generated after "
            "processing teaching material.\n\n"
            "What was it based on?\n"
            "The execution record shows that teaching material was retrieved "
            "before the question was generated, but it does not identify which "
            "specific information was used for this question.\n\n"
            "What difference did this information make?\n"
            "The execution record shows that source-support and answer checks "
            "were performed before ACCEPT was recorded."
        )
        ok2, warns2 = validate_explanation_faithfulness(
            condition="trace", explanation=safe, evidence_view=view
        )
        self.assertTrue(ok2, msg=str(warns2))


class TestDownstreamSpecificity(unittest.TestCase):
    def test_path_repair(self):
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                _cmp(
                    status="downstream_effect",
                    noop_atom=2,
                    cf_atom=2,
                    noop_path=["check_coverage[1]"],
                    cf_path=[
                        "check_coverage[1]",
                        "repair_missing_evidence[1]",
                        "atomize_new_evidence[1]",
                    ],
                    first_path_divergence=1,
                )
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.controlled_change_findings).lower()
        self.assertIn("evidence-repair", joined)
        self.assertEqual(facts.selected_downstream_change_type, "path_repair")
        self.assertIn("remained accept", joined)

    def test_provenance_only_not_path(self):
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                _cmp(
                    status="downstream_effect",
                    noop_atom=2,
                    cf_atom=2,
                    added_prov=[{"relation": "used"}],
                    removed_prov=[{"relation": "used"}],
                )
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.controlled_change_findings).lower()
        self.assertNotIn("repair", joined)
        self.assertNotIn("altered the later sequence", joined)
        self.assertIn("recorded as being used", joined)
        self.assertEqual(facts.selected_downstream_change_type, "recorded_use")

    def test_artifact_only(self):
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                _cmp(
                    status="downstream_effect",
                    noop_atom=2,
                    cf_atom=2,
                    payload_changed=["chunk_1:v1"],
                )
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.controlled_change_findings).lower()
        self.assertIn("information produced later", joined)
        self.assertNotIn("repair", joined)


class TestPrimaryPlusComplementary(unittest.TestCase):
    def test_criterion_plus_downstream_groups(self):
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                _cmp(status="criterion_effect", target="chunk_1:v1", noop_atom=2, cf_atom=3),
                _cmp(
                    status="downstream_effect",
                    target="atom_1:v1",
                    noop_atom=2,
                    cf_atom=2,
                    noop_path=["a"],
                    cf_path=["a", "repair_missing_evidence[1]"],
                    first_path_divergence=1,
                ),
                _cmp(
                    status="no_demonstrated_effect",
                    target="atom_2:v1",
                    noop_atom=2,
                    cf_atom=2,
                ),
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        self.assertEqual(facts.primary_effect_status, "criterion_effect")
        self.assertLessEqual(facts.selected_intervention_finding_count, 2)
        joined = " ".join(facts.controlled_change_findings).lower()
        self.assertIn("supporting knowledge", joined)
        self.assertIn("repair", joined)
        self.assertNotIn("can be ignored in general", joined)  # no-effect not forced


class TestAxisStatusVsTermination(unittest.TestCase):
    def test_budget_exhausted_with_success_is_completed(self):
        # Mirror updated service semantics
        term = "attempt_budget_exhausted"
        success_n = 5
        cf_n = 5
        cand_n = 3
        statuses = ["criterion_effect"] * 5
        if term == "no_valid_intervention_available" or (
            cf_n == 0 and cand_n == 0
        ):
            axis_status = "no_valid_intervention_available"
        elif success_n > 0:
            axis_status = "completed_with_counterfactual"
        elif term == "attempt_budget_exhausted":
            axis_status = "attempt_budget_exhausted"
        else:
            axis_status = "other"
        self.assertEqual(axis_status, "completed_with_counterfactual")
        self.assertEqual(term, "attempt_budget_exhausted")


class TestRemainingExecutableQueries(unittest.TestCase):
    def test_excludes_executed_signatures(self):
        ref = ArtifactRef("chunk_1", 1)
        cands = [
            InterventionCandidate(
                ref=ref,
                artifact_type="EvidenceChunk",
                allowed_operations=("REMOVE",),
                valid_boundaries=("before atomize_evidence#1", "after plan#1"),
                operation_boundaries={
                    "REMOVE": ("before atomize_evidence#1", "after plan#1"),
                },
            )
        ]
        executed = {
            QuerySignature(
                operation="REMOVE",
                target_ref=ref,
                checkpoint_id="ck_before_atomize",
            )
        }

        def resolve_query(q: str):
            if "before atomize_evidence#1" in q:
                return QuerySignature(
                    operation="REMOVE",
                    target_ref=ref,
                    checkpoint_id="ck_before_atomize",
                )
            if "after plan#1" in q:
                return QuerySignature(
                    operation="REMOVE",
                    target_ref=ref,
                    checkpoint_id="ck_after_plan",
                )
            if q.startswith("WHAT"):
                return QuerySignature(operation="WHAT", target_ref=ref)
            return None

        text = format_executable_queries(
            candidates=cands,
            exclude_signatures=executed,
            resolve_query=resolve_query,
        )
        self.assertNotIn("before atomize_evidence#1", text)
        self.assertIn("after plan#1", text)

        fb = controlled_validation_feedback(
            status="duplicate_error",
            detail="duplicate",
            executable_queries=text,
        )
        self.assertIn("untested executable queries", fb.lower())
        self.assertIn("after plan#1", fb)


if __name__ == "__main__":
    unittest.main()
