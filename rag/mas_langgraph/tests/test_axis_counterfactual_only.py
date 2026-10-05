"""Counterfactual-only AXIS interrogation (WHAT disabled in new runs)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.explainer import AxisExplanationController, _render_system
from mas_langgraph.axis.intervention_guidance import (
    format_hybrid_vocabulary,
    hybrid_opportunity_counts,
    validate_resolved_against_hybrid_vocabulary,
)
from mas_langgraph.axis.interrogation_mode import (
    DEFAULT_AXIS_INTERROGATION_POLICY,
)
from mas_langgraph.axis.models import AxisPromptTemplates, ResolvedQuery
from mas_langgraph.axis.queries import AxisQueryParser, QueryValidationError
from mas_langgraph.batch.manifest import generation_config_fingerprint
from mas_langgraph.evaluation.lifecycle import metrics_resume_ok
from mas_langgraph.evaluation.effect_reference import iter_hybrid_counterfactual_specs
from mas_langgraph.evaluation.report import build_report
from mas_langgraph.tests.test_axis_hybrid_interrogation import (
    PROMPT_DIR,
    _PromptRecorder,
    _availability_fixture,
    _build_fixture,
    _case_slice,
    _controller,
    _hybrid_vocab,
    _run,
)
from mas_langgraph.tests.test_prompt14_loop import _build_fixture as _p14_fixture


class TestCounterfactualOnlyPromptsAndVocabulary(unittest.TestCase):
    def test_hybrid_prompt_and_system_omit_what(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        templates = AxisPromptTemplates(
            system_hybrid=(PROMPT_DIR / "system_hybrid.txt").read_text(encoding="utf-8"),
            interrogation_hybrid=(PROMPT_DIR / "interrogation_hybrid.txt").read_text(
                encoding="utf-8"
            ),
        )
        recorder = _PromptRecorder(["DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            prompt_templates=templates,
            max_rounds=1,
            max_attempts=2,
        )
        _run(controller, "Why was this question selected?", case_slice)
        prompt = recorder.prompts[0]
        self.assertNotIn("WHAT(target)", prompt)
        self.assertNotIn("FACTUAL QUERY TARGETS", prompt)
        self.assertIn("REMOVE(target, boundary)", prompt)
        self.assertIn("WHATIF(old_artifact, replacement_artifact, boundary)", prompt)
        self.assertIn("DONE", prompt)
        system = _render_system(templates, axis_interrogation_mode="hybrid")
        self.assertNotIn("WHAT(target)", system)
        self.assertIn("REMOVE(artifact, boundary)", system)

    def test_hybrid_vocabulary_has_no_what_opportunities(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _cands = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        self.assertFalse(vocab.what_refs)
        counts = hybrid_opportunity_counts(vocab)
        self.assertEqual(counts["what"], 0)
        text = format_hybrid_vocabulary(vocabulary=vocab, registry=reg)
        self.assertNotIn("FACTUAL QUERY TARGETS", text)
        self.assertNotRegex(text, r"(?m)^WHAT\(")
        ops = {s.operation for s in iter_hybrid_counterfactual_specs(vocab)}
        self.assertTrue(ops <= {"REMOVE", "WHATIF", "ADD"})
        self.assertNotIn("WHAT", ops)

    def test_llm_generated_what_is_rejected(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        recorder = _PromptRecorder(["WHAT(E17:v1)", "DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            alias_map={"E17": ArtifactRef("E17", 1)},
            max_rounds=3,
            max_attempts=6,
        )
        result = _run(controller, "Why?", _case_slice(explained, occs, rels))
        what_atts = [a for a in result.interrogation_attempts if a.parsed_operation == "WHAT"]
        self.assertTrue(what_atts)
        self.assertTrue(all(a.validation_status == "vocabulary_error" for a in what_atts))
        self.assertTrue(all(not a.executed for a in what_atts))
        self.assertTrue(all(not a.counted_as_successful_round for a in what_atts))
        self.assertFalse(result.factual_records)
        executed_ops = [a.parsed_operation for a in result.interrogation_attempts if a.executed]
        self.assertNotIn("WHAT", executed_ops)

    def test_hybrid_validator_rejects_what(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _cands = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        resolved = ResolvedQuery(
            operation="WHAT",
            raw_text="WHAT(chunk_1:v1)",
            target_ref=ArtifactRef("chunk_1", 1),
        )
        with self.assertRaises(QueryValidationError) as ctx:
            validate_resolved_against_hybrid_vocabulary(resolved, vocab)
        self.assertIn("not in the advertised interrogation vocabulary", str(ctx.exception))


class TestCounterfactualOnlyExecution(unittest.TestCase):
    def test_remove_whatif_add_and_done_still_work(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _cands = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        remove = next(
            s.generated_query for s in iter_hybrid_counterfactual_specs(vocab) if s.operation == "REMOVE"
        )
        whatif = next(
            s.generated_query for s in iter_hybrid_counterfactual_specs(vocab) if s.operation == "WHATIF"
        )
        add = next(
            (s.generated_query for s in iter_hybrid_counterfactual_specs(vocab) if s.operation == "ADD"),
            None,
        )
        queries = [remove, whatif]
        if add:
            queries.append(add)
        queries.append("DONE")
        recorder = _PromptRecorder(queries)
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            max_rounds=5,
            max_attempts=12,
        )
        result = _run(controller, "Why was this question selected?", case_slice)
        executed = [a.parsed_operation for a in result.interrogation_attempts if a.executed]
        self.assertIn("REMOVE", executed)
        self.assertIn("WHATIF", executed)
        if add:
            self.assertIn("ADD", executed)
        self.assertEqual(result.termination_reason, "done")
        self.assertNotIn("WHAT", executed)

    def test_constrained_does_not_offer_what(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        recorder = _PromptRecorder(["DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="constrained",
            max_rounds=1,
            max_attempts=2,
        )
        _run(controller, "Why?", _case_slice(explained, occs, rels))
        prompt = recorder.prompts[0]
        self.assertNotIn("WHAT(target)", prompt)
        self.assertNotRegex(prompt, r"(?m)^WHAT\(")

    def test_max_rounds_counts_counterfactuals_not_what(self):
        explained, reg, occs, rels, idx, contracts, ex = _p14_fixture()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            max_rounds=1,
            max_attempts=8,
            compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=[
                "WHAT(E17:v1)",
                "WHAT(E17:v1)",
                "REMOVE(E17:v1, c_8)",
                "DONE",
            ],
        )
        self.assertEqual(result.successful_round_count, 1)
        self.assertEqual(len(result.intervention_records), 1)
        self.assertFalse(result.factual_records)
        self.assertEqual(result.termination_reason, "max_rounds")
        self.assertGreaterEqual(result.attempt_count, 3)

    def test_duplicate_invalid_do_not_consume_round(self):
        explained, reg, occs, rels, idx, contracts, ex = _p14_fixture()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            max_rounds=2,
            max_attempts=10,
            compatibility_mode=True,
        )
        result = controller.run(
            "Why?",
            scripted_queries=[
                "INVALID",
                "REMOVE(E17:v1, c_8)",
                "REMOVE(E17:v1, c_8)",
                "DONE",
            ],
        )
        self.assertEqual(result.successful_round_count, 1)
        self.assertEqual(result.termination_reason, "done")

    def test_done_before_max_rounds(self):
        explained, reg, occs, rels, idx, contracts, ex = _p14_fixture()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            max_rounds=5,
            compatibility_mode=True,
        )
        result = controller.run("Why?", scripted_queries=["DONE"])
        self.assertEqual(result.termination_reason, "done")
        self.assertEqual(result.successful_round_count, 0)
        self.assertNotIn("last_chance_intervention_nudge", result.warnings)
        self.assertNotIn("counterfactual_nudge_before_done", result.warnings)

    def test_no_last_chance_nudge_remains(self):
        explained, reg, occs, rels, idx, contracts, ex = _p14_fixture()
        controller = AxisExplanationController(
            explained_case=explained,
            registry=reg,
            occurrences=occs,
            provenance_relations=rels,
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": ArtifactRef("E17", 1)},
            max_rounds=1,
            max_attempts=5,
            compatibility_mode=True,
        )
        result = controller.run("Why?", scripted_queries=["REMOVE(E17:v1, c_8)"])
        self.assertNotIn("last_chance_intervention_nudge", result.warnings)
        self.assertEqual(result.successful_round_count, 1)


class TestLegacyCompatibilityAndFingerprint(unittest.TestCase):
    def test_parser_still_reads_historical_what(self):
        q = AxisQueryParser().parse("WHAT(chunk_2:v1)")
        self.assertEqual(q.operation, "WHAT")

    def test_historical_json_with_what_loads(self):
        payload = {
            "schema_version": 1,
            "axis_interrogation_mode": "hybrid",
            "cases": [
                {
                    "axis": {
                        "raw_evidence": {
                            "interrogation_attempts": [
                                {
                                    "generated_query": "WHAT(bp_00000001:v1)",
                                    "parsed_operation": "WHAT",
                                    "executed": True,
                                }
                            ]
                        }
                    },
                    "visualization_data": {"schema_version": 1, "artifacts": []},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        att = loaded["cases"][0]["axis"]["raw_evidence"]["interrogation_attempts"][0]
        self.assertEqual(att["parsed_operation"], "WHAT")
        self.assertEqual(loaded["cases"][0]["visualization_data"]["schema_version"], 1)

    def test_fingerprint_distinguishes_old_what_enabled_runs(self):
        base = {
            "llm_model": "gpt-4.1-mini-2025-04-14",
            "model_type": "openai",
            "temperature": 0.0,
            "seed": 42,
            "max_rounds": 5,
            "max_questions": 2,
            "orchestrator": "langgraph",
            "question": "Why?",
            "axis_interrogation_mode": "hybrid",
        }
        new_fp = generation_config_fingerprint(base)
        self.assertEqual(new_fp["axis_interrogation_policy"], "counterfactual_only")
        old_fp = {k: v for k, v in new_fp.items() if k != "axis_interrogation_policy"}
        self.assertNotEqual(new_fp, old_fp)
        stored = {
            "lecturebank_id": "141",
            "document_id": "lecturebank_141",
            "eval_status": "success",
            "source_hash": "abc",
            "rag_config": {"embedding_model": "x"},
            "generation_config": old_fp,
            "metrics_persisted": True,
            "document_metrics": {"lecturebank_id": "141"},
        }
        collected = [{"lecturebank_id": "141", "document_metrics": {"lecturebank_id": "141"}}]
        self.assertFalse(
            metrics_resume_ok(
                lecturebank_id="141",
                stored=stored,
                collected=collected,
                rag_fp={"embedding_model": "x"},
                gen_fp=new_fp,
                source_hash="abc",
            )
        )

    def test_new_attempts_have_zero_what_count(self):
        explained, reg, occs, rels, idx, contracts, ex = _build_fixture()
        recorder = _PromptRecorder(["DONE"])
        controller = _controller(
            explained, reg, occs, rels, idx, contracts, ex,
            llm_generate=recorder,
            axis_interrogation_mode="hybrid",
            max_rounds=1,
            max_attempts=2,
        )
        result = _run(controller, "Why?", _case_slice(explained, occs, rels))
        self.assertEqual(result.axis_interrogation_policy, DEFAULT_AXIS_INTERROGATION_POLICY)
        for att in result.interrogation_attempts:
            self.assertEqual(att.available_what_count, 0)
            if att.executed:
                self.assertNotEqual(att.parsed_operation, "WHAT")

    def test_report_does_not_treat_what_as_active(self):
        md = build_report(
            document_rows=[],
            attempt_rows=[],
            comparison_rows=[],
            provenance_rows=[],
            distribution_rows=[],
            round_rows=[],
            catalog_rows=[],
            overall_rows=[],
            pooled={"proposed_what_count": 0, "executed_what_count": 0},
        )
        self.assertIn("WHAT is disabled in the generalized AXIS configuration", md)
        self.assertNotIn("| WHAT |", md)

    def test_effect_discovery_universe_excludes_what(self):
        explained, reg, occs, rels, idx, contracts, ex, case_slice = _availability_fixture()
        vocab, _cands = _hybrid_vocab(explained, reg, occs, idx, contracts, case_slice, ex)
        ops = {s.operation for s in iter_hybrid_counterfactual_specs(vocab)}
        self.assertNotIn("WHAT", ops)
        self.assertNotIn("DONE", ops)


if __name__ == "__main__":
    unittest.main()
