"""Bounded SAQUET LLM retries and response validation. No network or sleep."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_RAG = Path(__file__).resolve().parents[2]
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
    "numpy",
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
    nltk.pos_tag = MagicMock()
    sys.modules["transformers"].pipeline = MagicMock(return_value=MagicMock())
    sys.modules["openai"].OpenAI = MagicMock()
    sys.modules["numpy"].mean = MagicMock(side_effect=lambda pair: sum(pair) / 2)


def _restore_stubs(saved: dict) -> None:
    for name in _HEAVY:
        if name in saved:
            sys.modules[name] = saved[name]
        elif name in sys.modules:
            del sys.modules[name]
    for mod in list(sys.modules):
        if mod in {"saquet.saquet_core", "saquet.adapter"}:
            del sys.modules[mod]


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class _NpBool:
    """numpy.bool_-like value: truthy/falsy, but not identical to Python bool."""

    __module__ = "numpy"

    def __init__(self, value: bool):
        self._value = bool(value)

    def __bool__(self) -> bool:
        return self._value


_NpBool.__name__ = "bool_"


class _NpFloat:
    def __init__(self, value: float):
        self._value = float(value)

    def __ge__(self, other):
        return _NpBool(self._value >= float(other))


class TestSaquetLlmBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._saved: dict = {}
        _install_stubs(cls._saved)
        sys.modules.pop("saquet.saquet_core", None)
        import saquet.saquet_core as core

        cls.core = core
        core.SAQUET_LLM_MAX_ATTEMPTS = 5
        core.SAQUET_LLM_RETRY_SLEEP_S = 0
        core.SAQUET_LLM_TIMEOUT_S = 60.0

    @classmethod
    def tearDownClass(cls) -> None:
        _restore_stubs(cls._saved)

    def setUp(self) -> None:
        self.core.SAQUET_LLM_MAX_ATTEMPTS = 5
        self.core.SAQUET_LLM_RETRY_SLEEP_S = 0
        self.sleep = MagicMock()
        self.sleep_patch = patch.object(self.core.time, "sleep", self.sleep)
        self.sleep_patch.start()

    def tearDown(self) -> None:
        self.sleep_patch.stop()

    def test_timeout_passed_to_create_and_early_stop_on_two_high_ratings(self):
        create = MagicMock(return_value=_message("8"))
        self.core.client.chat.completions.create = create
        result = self.core.llm_rating("sys", "prompt")
        self.assertIs(result, True)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args.kwargs.get("timeout"), 60.0)
        self.sleep.assert_not_called()

    def test_permanent_api_failure_raises_and_does_not_return_bool(self):
        create = MagicMock(side_effect=RuntimeError("api down"))
        self.core.client.chat.completions.create = create
        with self.assertRaises(self.core.SaquetEvaluationError):
            self.core.llm_rating("sys", "prompt")
        self.assertEqual(create.call_count, 5)
        self.assertEqual(self.sleep.call_count, 5)

    def test_repeated_malformed_ratings_count_toward_limit(self):
        create = MagicMock(return_value=_message("not-a-score"))
        self.core.client.chat.completions.create = create
        with self.assertRaises(self.core.SaquetEvaluationError):
            self.core.llm_rating("sys", "prompt")
        self.assertEqual(create.call_count, 5)
        self.sleep.assert_not_called()

    def test_out_of_range_rating_is_malformed(self):
        self.assertIsNone(self.core._parse_llm_rating_value("0"))
        self.assertIsNone(self.core._parse_llm_rating_value("11"))
        self.assertEqual(self.core._parse_llm_rating_value("10"), 10)
        self.assertEqual(self.core._parse_llm_rating_value("1"), 1)

    def test_eventual_success_after_failures_and_malformed(self):
        create = MagicMock(
            side_effect=[
                RuntimeError("once"),
                _message("nope"),
                _message("9"),
                _message("9"),
            ]
        )
        self.core.client.chat.completions.create = create
        result = self.core.llm_rating("sys", "prompt")
        self.assertIs(result, True)
        self.assertEqual(create.call_count, 4)
        self.assertEqual(self.sleep.call_count, 1)

    def test_mean_of_unequal_high_ratings_returns_python_true(self):
        create = MagicMock(side_effect=[_message("9"), _message("10")])
        self.core.client.chat.completions.create = create
        self.core.np.mean = MagicMock(side_effect=lambda pair: _NpFloat(sum(pair) / 2))
        result = self.core.llm_rating("sys", "prompt")
        self.assertIs(result, True)
        self.assertIsInstance(result, bool)
        self.assertIsNot(type(result), _NpBool)
        from mas_langgraph.evaluation.saquet_eval import _iwf_flag

        self.assertEqual(_iwf_flag(result), 0)
        self.assertEqual(create.call_count, 2)

    def test_mean_of_unequal_low_ratings_returns_python_false(self):
        create = MagicMock(side_effect=[_message("3"), _message("4")])
        self.core.client.chat.completions.create = create
        self.core.np.mean = MagicMock(side_effect=lambda pair: _NpFloat(sum(pair) / 2))
        result = self.core.llm_rating("sys", "prompt")
        self.assertIs(result, False)
        self.assertIsInstance(result, bool)
        from mas_langgraph.evaluation.saquet_eval import _iwf_flag

        self.assertEqual(_iwf_flag(result), 1)
        self.assertEqual(create.call_count, 2)

    def test_equal_high_ratings_still_return_python_true(self):
        create = MagicMock(side_effect=[_message("9"), _message("9")])
        self.core.client.chat.completions.create = create
        result = self.core.llm_rating("sys", "prompt")
        self.assertIs(result, True)
        self.assertIsInstance(result, bool)
        from mas_langgraph.evaluation.saquet_eval import _iwf_flag

        self.assertEqual(_iwf_flag(result), 0)
        self.assertEqual(create.call_count, 2)

    def test_more_than_one_correct_rejects_malformed_json_then_accepts_valid(self):
        question = SimpleNamespace(
            stem="Which?",
            options=["alpha", "beta", "gamma", "delta"],
            correct_option="alpha",
        )
        valid = json.dumps({"Stem": "Which?", "correct_answers": [{"A": "alpha"}]})
        create = MagicMock(
            side_effect=[
                RuntimeError("api"),
                _message("{not json"),
                _message(valid),
            ]
        )
        self.core.client.chat.completions.create = create
        result = self.core.more_than_one_correct(question)
        self.assertIs(result, True)
        self.assertEqual(create.call_count, 3)

    def test_more_than_one_correct_exhaustion_does_not_become_true_or_false(self):
        question = SimpleNamespace(
            stem="Which?",
            options=["alpha", "beta"],
            correct_option="alpha",
        )
        create = MagicMock(return_value=_message("{not json"))
        self.core.client.chat.completions.create = create
        with self.assertRaises(self.core.SaquetEvaluationError):
            self.core.more_than_one_correct(question)
        self.assertEqual(create.call_count, 5)


if __name__ == "__main__":
    unittest.main()
