"""Focused tests for explanation-experiment timing instrumentation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.experiment.export import experiment_to_dict
from mas_langgraph.experiment.llm import invoke_llm_text
from mas_langgraph.experiment.models import (
    EXPERIMENT_SCHEMA_VERSION,
    ExplanationConditionResult,
    MCQExplanationComparison,
    QuizExplanationExperiment,
)
from mas_langgraph.experiment.timing import (
    ExperimentTimingCollector,
    assemble_replay_profile,
    explanation_llm_phase,
    timing_scope,
)


def _minimal_experiment(**kwargs):
    case = MCQExplanationComparison(
        run_id="run1",
        branch_id="factual",
        candidate_ref=ArtifactRef("candidate_mcq:q1", 1),
        assessment_ref=ArtifactRef("assessment:q1", 1),
        decision_ref=ArtifactRef("decision:q1", 1),
        question_stem="Stem?",
        answer_options=(),
        correct_option_id="A",
        factual_decision="accepted",
        user_question="Why?",
        trace=ExplanationConditionResult(condition="trace", explanation="t"),
        trace_provenance=ExplanationConditionResult(
            condition="trace_provenance", explanation="tp"
        ),
        axis=ExplanationConditionResult(condition="axis", explanation="a"),
    )
    defaults = dict(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        run_id="run1",
        branch_id="factual",
        experiment_seed=42,
        user_question="Why?",
        cases=(case,),
    )
    defaults.update(kwargs)
    return QuizExplanationExperiment(**defaults)


class ExperimentTimingCollectorTests(unittest.TestCase):
    def test_experiment_case_round_timing_structure(self):
        c = ExperimentTimingCollector(run_id="r1")
        c.start_experiment()
        c.record_experiment_phase("factual_workflow_s", 1.5)
        c.record_experiment_phase("case_discovery_s", 0.25)
        c.begin_case(target="q_1", candidate_version=1, factual_decision="accepted")
        c.record_case_condition("trace", 0.1)
        c.add_condition("trace", 0.1)
        c.record_case_condition("trace_provenance", 0.2)
        c.add_condition("trace_provenance", 0.2)
        c.begin_round(
            round_index=1,
            intervention="REMOVE(x, ckpt)",
            checkpoint="ckpt",
            checkpoint_alias="alias",
        )
        c.add_round_phase("interrogation", 0.3)
        c.add_round_phase("intervention_resolution", 0.05)
        c.record_replay(
            branch="noop",
            duration_s=0.4,
            checkpoint="ckpt",
            checkpoint_alias="alias",
            bounded_requested=False,
            cache_hit=True,
            completion_mode="graph_end",
            executed_node_count=10,
            logical_llm_requests=3,
            factual_response_hits=3,
            live_provider_calls=0,
        )
        c.record_replay(
            branch="counterfactual",
            duration_s=0.7,
            checkpoint="ckpt",
            checkpoint_alias="alias",
            bounded_requested=True,
            completion_mode="target_observable",
            stopped_after_node="apply_acceptance_gate",
            executed_node_count=5,
            logical_llm_requests=5,
            factual_response_hits=2,
            counterfactual_memo_hits=1,
            live_provider_calls=2,
            full_remaining_nodes_avoided=4,
        )
        c.add_round_phase("comparison", 0.02)
        c.add_round_phase("intermediate_synthesis", 0.15)
        c.update_round_meta(
            evidence_status="demonstrated_effect",
            causal_classification="demonstrated_effect",
        )
        c.end_round()
        c.record_final_synthesis(0.2)
        with explanation_llm_phase("trace_participant_synthesis"):
            c.record_explanation_llm(phase="trace_participant_synthesis", duration_s=0.11)
        c.record_case_condition("axis", 2.0)
        c.add_condition("axis", 2.0)
        c.end_case(meta={"valid_replay_comparisons": 1})
        c.end_experiment()

        payload = c.to_dict()
        self.assertIn("experiment", payload)
        self.assertIn("conditions", payload)
        self.assertIn("axis", payload)
        self.assertIn("workflow_llm", payload)
        self.assertGreaterEqual(payload["experiment"]["factual_workflow_s"], 0.0)
        self.assertGreaterEqual(payload["experiment"]["experiment_total_s"], 0.0)
        self.assertEqual(len(payload["axis"]["cases"]), 1)
        case = payload["axis"]["cases"][0]
        self.assertEqual(case["target"], "q_1")
        self.assertGreaterEqual(case["axis_total_s"], 0.0)
        self.assertEqual(len(case["rounds"]), 1)
        rd = case["rounds"][0]
        for key in (
            "interrogation_s",
            "intervention_resolution_s",
            "noop_s",
            "counterfactual_replay_s",
            "comparison_s",
            "intermediate_synthesis_s",
            "total_s",
        ):
            self.assertIn(key, rd)
            self.assertGreaterEqual(float(rd[key]), 0.0)
        self.assertEqual(rd["replay_completion_mode"], "target_observable")
        self.assertTrue(rd["noop_cache_hit"])
        self.assertGreaterEqual(float(case["explanation_llm_s"]), 0.0)
        self.assertGreaterEqual(float(case["workflow_replay_s"]), 0.0)
        self.assertEqual(case["meta"]["valid_replay_comparisons"], 1)

    def test_explanation_llm_recorded_separately(self):
        c = ExperimentTimingCollector()
        with timing_scope(c):
            c.begin_case(target="q_1")

            class _LLM:
                model_name = "test-model"

                def invoke(self, prompt):
                    return type(
                        "R",
                        (),
                        {
                            "content": "hello",
                            "usage_metadata": {
                                "input_tokens": 3,
                                "output_tokens": 2,
                                "total_tokens": 5,
                            },
                        },
                    )()

            with explanation_llm_phase("interrogation"):
                text = invoke_llm_text(_LLM(), "prompt")
            self.assertEqual(text, "hello")
            self.assertEqual(len(c.explanation_llm_calls), 1)
            call = c.explanation_llm_calls[0]
            self.assertEqual(call["phase"], "interrogation")
            self.assertGreaterEqual(float(call["duration_s"]), 0.0)
            self.assertEqual(call["input_tokens"], 3)
            c.end_case()

    def test_export_backward_compatible_with_timing(self):
        exp = _minimal_experiment(
            timing={
                "experiment": {"experiment_total_s": 1.0},
                "conditions": {},
                "axis": {"totals": {}, "cases": []},
                "workflow_llm": {},
            }
        )
        payload = experiment_to_dict(exp)
        self.assertIn("schema_version", payload)
        self.assertIn("cases", payload)
        self.assertIn("timing", payload)
        self.assertEqual(payload["timing"]["experiment"]["experiment_total_s"], 1.0)
        # Existing fields remain.
        for key in (
            "run_id",
            "branch_id",
            "experiment_seed",
            "user_question",
            "workflow_llm_diagnostics",
        ):
            self.assertIn(key, payload)

    def test_export_without_timing_still_works(self):
        exp = _minimal_experiment()
        payload = experiment_to_dict(exp)
        self.assertEqual(payload["timing"], {})
        self.assertIn("cases", payload)

    def test_csv_summary_written(self):
        c = ExperimentTimingCollector(run_id="r1")
        c.begin_case(target="q_1")
        c.record_case_condition("axis", 1.25)
        c.end_case(meta={"rounds": 2, "valid_replay_comparisons": 1})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis_timing_summary.csv"
            out = c.write_csv(str(path))
            text = Path(out).read_text(encoding="utf-8")
            self.assertIn("target", text)
            self.assertIn("q_1", text)
            self.assertIn("axis_total_s", text)


class ExperimentTimingScopeTests(unittest.TestCase):
    def test_axis_scopes_are_non_overlapping_and_non_negative(self):
        c = ExperimentTimingCollector(run_id="r1")
        c.begin_case(target="q_1")
        with explanation_llm_phase("trace_participant_synthesis"):
            c.record_explanation_llm(phase="trace_participant_synthesis", duration_s=5.0)
        with explanation_llm_phase("interrogation"):
            c.record_explanation_llm(phase="interrogation", duration_s=1.0)
        c.record_replay(
            branch="noop",
            duration_s=2.0,
            cache_hit=True,
            completion_mode="graph_end",
        )
        c.record_replay(
            branch="counterfactual",
            duration_s=4.0,
            completion_mode="target_observable",
            live_provider_calls=1,
        )
        c.record_case_condition("axis", 10.0)
        c.end_case()
        case = c.cases[0]
        expl = float(case["axis_explanation_llm_s"])
        replay = float(case["axis_replay_s"])
        other = float(case["axis_non_llm_non_replay_s"])
        total = float(case["axis_total_s"])
        self.assertEqual(expl, 1.0)
        self.assertEqual(replay, 6.0)
        self.assertGreaterEqual(expl, 0.0)
        self.assertGreaterEqual(replay, 0.0)
        self.assertGreaterEqual(other, 0.0)
        self.assertAlmostEqual(expl + replay + other, total, places=6)
        pct_sum = (
            float(case["axis_explanation_llm_pct"])
            + float(case["axis_replay_pct"])
            + float(case["axis_non_llm_non_replay_pct"])
        )
        self.assertLessEqual(pct_sum, 100.1)
        self.assertEqual(case["non_axis_explanation_llm_s"], 5.0)
        meta = case["meta"]
        self.assertEqual(meta["counterfactual_target_observable_count"], 1)
        self.assertEqual(meta["counterfactual_graph_end_count"], 0)
        self.assertEqual(meta["noop_cache_hit_count"], 1)
        self.assertEqual(meta["graph_end_count"], 0)
        payload = c.to_dict()
        self.assertIn("replay_profile", payload["axis"])
        self.assertIn("noop_cache_profile", payload["axis"])
        self.assertIn("node_profile", payload["axis"])
        self.assertLessEqual(
            float(payload["axis"]["totals"]["axis_scope_pct_sum"]), 100.1
        )

    def test_round_causal_classification_not_copied_from_evidence(self):
        c = ExperimentTimingCollector()
        c.begin_case(target="q_1")
        c.begin_round(round_index=1, intervention="REMOVE(x)", checkpoint="c_8")
        c.update_round_meta(evidence_status="criterion_effect")
        c.end_round()
        rd = c._case["rounds"][0]
        self.assertEqual(rd["evidence_status"], "criterion_effect")
        self.assertIn(rd.get("causal_classification"), ("", None))

    def test_replay_top_level_excludes_nested_provider_wait(self):
        profile = assemble_replay_profile(
            spans={
                "graph_execution": 8.0,
                "provider_wait": 5.0,
                "branch_history_fetch": 2.0,
            },
            total_s=10.0,
        )
        top = profile["top_level"]
        for value in top.values():
            self.assertGreaterEqual(float(value), 0.0)
        exclusive = sum(float(v) for k, v in top.items() if k != "other_s")
        self.assertAlmostEqual(exclusive + float(top["other_s"]), 10.0, places=6)
        self.assertEqual(float(profile["nested"]["provider_wait_s"]), 5.0)
        self.assertIn("top_level", profile)
        self.assertIn("nested", profile)


if __name__ == "__main__":
    unittest.main()
