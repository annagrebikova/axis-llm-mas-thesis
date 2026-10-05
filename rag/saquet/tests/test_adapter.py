"""Integration tests around SAQUET PDF-141 adapter (no live OpenAI / heavy NLP)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_RAG = Path(__file__).resolve().parents[2]  # .../rag
_QUIZ = _RAG / "quiz_141.json"

if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))

_HEAVY = [
    "Levenshtein",
    "spacy",
    "transformers",
    "torch",
    "sentence_transformers",
    "lexicalrichness",
    "nltk",
    "nltk.corpus",
    "nltk.stem",
    "nltk.tokenize",
    "openai",
    "evaluate",
    "language_tool_python",
    "pandas",
]


def _install_stubs(saved: dict) -> None:
    for name in _HEAVY:
        if name in sys.modules:
            saved[name] = sys.modules[name]
        sys.modules[name] = MagicMock()
    spacy = sys.modules["spacy"]
    spacy.load = MagicMock(return_value=MagicMock())
    nltk = sys.modules["nltk"]
    nltk.download = MagicMock()
    stopwords = MagicMock()
    stopwords.words = MagicMock(return_value=[])
    sys.modules["nltk.corpus"].stopwords = stopwords
    sys.modules["nltk.corpus"].wordnet = MagicMock()
    sys.modules["nltk.stem"].WordNetLemmatizer = MagicMock
    sys.modules["nltk.tokenize"].word_tokenize = MagicMock(return_value=[])
    sys.modules["nltk.tokenize"].sent_tokenize = MagicMock(return_value=[])
    nltk.trigrams = MagicMock()
    nltk.ngrams = MagicMock()
    nltk.word_tokenize = MagicMock(return_value=[])
    nltk.pos_tag = MagicMock(return_value=[])
    sys.modules["transformers"].pipeline = MagicMock(return_value=MagicMock())
    sys.modules["transformers"].AutoTokenizer.from_pretrained = MagicMock()
    sys.modules["transformers"].AutoModelForSequenceClassification.from_pretrained = (
        MagicMock()
    )
    sys.modules["openai"].OpenAI = MagicMock()


def _restore_stubs(saved: dict) -> None:
    for name in _HEAVY:
        if name in saved:
            sys.modules[name] = saved[name]
        elif name in sys.modules:
            del sys.modules[name]
    for mod in list(sys.modules):
        if mod == "saquet" or mod.startswith("saquet."):
            # Keep package importable; drop loaded core so next import can be real.
            if mod in {"saquet.saquet_core", "saquet.adapter"}:
                del sys.modules[mod]


class TestQuiz141Load(unittest.TestCase):
    def test_quiz_141_exists_and_has_questions(self):
        self.assertTrue(_QUIZ.is_file(), f"missing {_QUIZ}")
        data = json.loads(_QUIZ.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data[0]["questions"]), 1)

    def test_extract_records_without_saquet_core(self):
        data = json.loads(_QUIZ.read_text(encoding="utf-8"))
        qs = data[0]["questions"]
        for q in qs:
            self.assertIn("q_id", q)
            self.assertIn("question_text", q)
            self.assertIn("correct_answer", q)
            self.assertIn("options", q)
            self.assertIn(q["correct_answer"], q["options"])


class TestSaquetAdapter(unittest.TestCase):
    saved = {}

    @classmethod
    def setUpClass(cls):
        cls.saved = {}
        _install_stubs(cls.saved)
        for mod in list(sys.modules):
            if mod == "saquet" or mod.startswith("saquet."):
                del sys.modules[mod]
        from saquet.adapter import (
            extract_quiz_records,
            record_to_multiple_choice_question,
            quiz_path_to_saquet_questions,
            validate_records,
        )
        from saquet.saquet_core import MultipleChoiceQuestion, all_criteria

        cls.extract_quiz_records = staticmethod(extract_quiz_records)
        cls.record_to_mcq = staticmethod(record_to_multiple_choice_question)
        cls.quiz_to_qs = staticmethod(quiz_path_to_saquet_questions)
        cls.validate_records = staticmethod(validate_records)
        cls.MultipleChoiceQuestion = MultipleChoiceQuestion
        cls.all_criteria = all_criteria

    @classmethod
    def tearDownClass(cls):
        _restore_stubs(cls.saved)

    def test_records_load(self):
        records = self.extract_quiz_records(_QUIZ)
        self.assertEqual(len(records), 6)
        self.assertTrue(all(r["q_id"] for r in records))

    def test_adapter_creates_mcq(self):
        records = self.extract_quiz_records(_QUIZ)
        q = self.record_to_mcq(records[0])
        self.assertIsInstance(q, self.MultipleChoiceQuestion)
        self.assertEqual(q.qid, records[0]["q_id"])
        self.assertEqual(q.stem, records[0]["question_text"])
        self.assertEqual(q.correct_option, records[0]["correct_answer"])
        self.assertEqual(list(q.options), list(records[0]["options"]))

    def test_correct_answer_in_options(self):
        for q in self.quiz_to_qs(_QUIZ):
            self.assertIn(q.correct_option, q.options)

    def test_validate_records_ok(self):
        self.assertEqual(self.validate_records(self.extract_quiz_records(_QUIZ)), [])

    def test_all_19_criteria_present(self):
        expected = [
            "ambiguous_unclear_information",
            "implausible_distractors",
            "none_of_the_above",
            "longest_answer_correct",
            "gratuitous_information_in_stem",
            "true_or_false",
            "avoid_convergence_cues",
            "avoid_logical_cues",
            "all_of_the_above",
            "fill_in_the_blank",
            "absolute_terms",
            "word_repeats_in_stem_and_correct_answer",
            "unfocused_stem",
            "complex_k_type",
            "grammatical_cues_in_stem",
            "lost_sequence",
            "vague_terms",
            "more_than_one_correct",
            "negative_worded_stem",
        ]
        self.assertEqual(list(self.all_criteria), expected)
        self.assertEqual(len(self.all_criteria), 19)


class TestSaquetAggregation(unittest.TestCase):
    def test_zero_one_semantics_and_totals(self):
        from saquet.run_saquet_pdf_141 import ALL_CRITERIA, build_rows, summarize

        qs = [
            SimpleNamespace(
                qid="q1",
                stem="Stem?",
                correct_option="A text",
                options=["A text", "B", "C", "D"],
            ),
            SimpleNamespace(
                qid="q2",
                stem="Stem2?",
                correct_option="X",
                options=["X", "Y", "Z", "W"],
            ),
        ]
        flags = {c: [0, 0] for c in ALL_CRITERIA}
        flags["vague_terms"] = [1, 0]
        flags["absolute_terms"] = [1, 1]
        rows = build_rows(qs, flags)
        self.assertEqual(rows[0]["total_iwf_count"], 2)
        self.assertEqual(rows[1]["total_iwf_count"], 1)
        summary = summarize(rows)
        self.assertEqual(summary["n_mcqs"], 2)
        self.assertEqual(summary["total_detected_iwfs"], 3)
        self.assertEqual(summary["n_criteria"], 19)

    def test_evaluate_questions_uses_original_conversion(self):
        from saquet import run_saquet_pdf_141 as runner
        import types
        import saquet as saquet_pkg

        q = SimpleNamespace(qid="q", stem="s", correct_option="a", options=["a", "b"])
        fake = types.ModuleType("saquet.saquet_core")
        fake.all_criteria = list(runner.ALL_CRITERIA)

        def _pass(_q):
            return True

        def _fail(_q):
            return False

        for name in runner.ALL_CRITERIA:
            setattr(fake, name, _fail if name == "vague_terms" else _pass)

        with patch.dict(sys.modules, {"saquet.saquet_core": fake}):
            saquet_pkg.saquet_core = fake
            flags = runner.evaluate_questions([q])
        self.assertEqual(len(flags), 19)
        self.assertEqual(flags["vague_terms"], [1])
        self.assertEqual(flags["none_of_the_above"], [0])


if __name__ == "__main__":
    unittest.main()
