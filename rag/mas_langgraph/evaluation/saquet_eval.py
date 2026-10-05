"""Optional SAQUET IWF evaluation of final accepted factual MCQs.

Evaluation-only. Does not change MAS generation, AXIS interrogation, replay,
or SAQUET criterion logic. Question-level results stay in an internal JSON
cache; the CSV deliverable is one aggregated row per document/MAS run.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from mas_langgraph.batch.io import atomic_write_json, load_json
from mas_langgraph.evaluation.axis_metrics import NA, ratio
from mas_langgraph.evaluation.quiz_export import extract_quiz_questions

# Exact 19 IWF criteria from the supplied SAQUET implementation.
SAQUET_CRITERIA = [
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

SAQUET_CONFIG_VERSION = "saquet_iwf_v2"
SAQUET_DEFAULT_MODEL = "gpt-4o"
SAQUET_CACHE_NAME = "saquet_question_cache.json"
SAQUET_FACTUAL_QUIZ_NAME = "saquet_factual_quiz.json"

QUIZ_COMPLETE = "complete"
QUIZ_EMPTY = "empty"
QUIZ_INCOMPLETE = "incomplete"
QUIZ_UNKNOWN = "unknown"

OPTION_LABELS = ("A", "B", "C", "D")

SAQUET_MAS_BASE_FIELDS = [
    "lecturebank_id",
    "run_id",
    "final_accepted_mcq_count",
    "fully_evaluated_mcq_count",
    "partial_mcq_count",
    "failed_mcq_count",
    "not_evaluable_mcq_count",
    "evaluation_completion_rate",
    "total_iwf_count",
    "mean_iwf_per_mcq",
    "zero_iwf_mcq_count",
    "zero_iwf_mcq_rate",
    "quiz_completeness",
    "quiz_source",
    "quiz_completeness_reason",
]

SAQUET_MAS_FIELDS = (
    list(SAQUET_MAS_BASE_FIELDS)
    + [
        f"{name}_{suffix}"
        for name in SAQUET_CRITERIA
        for suffix in ("flagged_count", "evaluated_count", "flag_rate")
    ]
    + ["evaluator_model", "evaluator_config_version"]
)


def saquet_criterion_fields(name: str) -> Tuple[str, str, str]:
    return (
        f"{name}_flagged_count",
        f"{name}_evaluated_count",
        f"{name}_flag_rate",
    )


class SaquetQuestionCache:
    """Internal per-question SAQUET results. Not a CSV deliverable."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._rows: Dict[str, Dict[str, Any]] = {}
        if self.path is not None and self.path.is_file():
            try:
                raw = load_json(self.path)
            except Exception:
                raw = {}
            stored = raw.get("entries") if isinstance(raw, Mapping) else None
            if isinstance(stored, Mapping):
                self._rows = {
                    str(k): dict(v) for k, v in stored.items() if isinstance(v, Mapping)
                }

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        hit = self._rows.get(str(key))
        return dict(hit) if hit is not None else None

    def put(self, key: str, payload: Mapping[str, Any]) -> None:
        self._rows[str(key)] = dict(payload)
        self._flush()

    def _flush(self) -> None:
        if self.path is None:
            return
        atomic_write_json(
            self.path,
            {
                "kind": "saquet_question_cache",
                "evaluator_config_version": SAQUET_CONFIG_VERSION,
                "entries": self._rows,
            },
        )


def default_saquet_model() -> str:
    """Return the SAQUET evaluator model without importing saquet_core.

    Importing ``saquet.saquet_core`` loads heavy NLP stacks. Use the already-
    imported module when present; otherwise the supplied default ``gpt-4o``.
    """
    import sys

    core = sys.modules.get("saquet.saquet_core")
    if core is not None:
        model = str(getattr(core, "model_engine", "") or "").strip()
        if model:
            return model
    return SAQUET_DEFAULT_MODEL


def quiz_row_options(row: Mapping[str, Any]) -> List[str]:
    opts: List[str] = []
    for label in OPTION_LABELS:
        text = str(row.get(f"option_{label}") or "").strip()
        if text:
            opts.append(text)
    extra = row.get("options")
    if not opts and isinstance(extra, (list, tuple)):
        opts = []
        for item in extra:
            if isinstance(item, Mapping):
                text = str(item.get("text") or "").strip()
            else:
                text = str(item).strip()
            if text:
                opts.append(text)
    return opts


def quiz_row_correct_text(row: Mapping[str, Any], options: Sequence[str]) -> str:
    raw = str(row.get("correct_option") or row.get("correct_answer") or "").strip()
    if raw in options:
        return raw
    upper = raw.upper()
    if upper in OPTION_LABELS:
        keyed = str(row.get(f"option_{upper}") or "").strip()
        if keyed:
            return keyed
        idx = OPTION_LABELS.index(upper)
        if idx < len(options):
            return options[idx]
    extra = row.get("options")
    if isinstance(extra, (list, tuple)):
        for item in extra:
            if isinstance(item, Mapping) and item.get("is_correct"):
                text = str(item.get("text") or "").strip()
                if text:
                    return text
    return raw


def saquet_question_mappable(row: Mapping[str, Any]) -> Tuple[bool, str]:
    stem = str(row.get("stem") or row.get("question_text") or "").strip()
    options = quiz_row_options(row)
    correct = quiz_row_correct_text(row, options)
    if not stem:
        return False, "missing_stem"
    if not options:
        return False, "missing_options"
    if not correct or correct not in options:
        return False, "correct_not_in_options"
    return True, ""


def saquet_input_fingerprint(
    row: Mapping[str, Any],
    *,
    evaluator_model: str,
    evaluator_config_version: str,
) -> Dict[str, Any]:
    options = quiz_row_options(row)
    return {
        "lecturebank_id": str(row.get("lecturebank_id") or ""),
        "candidate_ref": str(row.get("candidate_ref") or ""),
        "stem": str(row.get("stem") or row.get("question_text") or ""),
        "options": options,
        "correct": quiz_row_correct_text(row, options),
        "evaluator_model": evaluator_model,
        "evaluator_config_version": evaluator_config_version,
    }


def saquet_cache_key(
    row: Mapping[str, Any],
    *,
    evaluator_model: str,
    evaluator_config_version: str,
) -> str:
    payload = saquet_input_fingerprint(
        row,
        evaluator_model=evaluator_model,
        evaluator_config_version=evaluator_config_version,
    )
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_numpy_bool(value: Any) -> bool:
    """True for ``numpy.bool_`` / ``numpy.bool8`` without importing NumPy."""
    cls = type(value)
    if getattr(cls, "__module__", "") != "numpy":
        return False
    return getattr(cls, "__name__", "") in {"bool_", "bool8"}


def _iwf_flag(result: Any) -> Optional[int]:
    """Original SAQUET conversion: True → 0 (no flaw), False → 1 (flaw).

    Accepts Python ``bool`` and ``numpy.bool_``. Does not coerce strings or
    numbers: ``"False"`` and ``1`` stay unavailable.
    """
    if isinstance(result, bool):
        return 0 if result else 1
    if _is_numpy_bool(result):
        return 0 if bool(result) else 1
    return NA


def _apply_supports_criteria(apply_fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(apply_fn).parameters
    except (TypeError, ValueError):
        return False
    if "criteria" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def default_apply_saquet_criteria(
    row: Mapping[str, Any],
    *,
    criteria: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Call supplied SAQUET criterion functions. Lazy-imports saquet_core."""
    import importlib

    core = importlib.import_module("saquet.saquet_core")
    names = list(getattr(core, "all_criteria", None) or SAQUET_CRITERIA)
    if list(names) != list(SAQUET_CRITERIA):
        raise RuntimeError("saquet_core.all_criteria does not match the required 19 IWF criteria.")
    options = quiz_row_options(row)
    correct = quiz_row_correct_text(row, options)
    question = core.MultipleChoiceQuestion(
        stem=str(row.get("stem") or row.get("question_text") or ""),
        options=options,
        correct_option=correct,
        qid=str(row.get("candidate_ref") or row.get("q_id") or ""),
        quality=0,
    )
    wanted = [name for name in (criteria or SAQUET_CRITERIA) if name in SAQUET_CRITERIA]
    flags: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for name in wanted:
        fn = getattr(core, name)
        try:
            flags[name] = _iwf_flag(fn(question))
        except Exception as exc:  # noqa: BLE001
            flags[name] = NA
            errors[name] = str(exc) or type(exc).__name__
    return {"flags": flags, "errors": errors}


def _assessment_status(flags: Mapping[str, Any]) -> str:
    known = [flags.get(name) for name in SAQUET_CRITERIA if flags.get(name) in (0, 1)]
    if len(known) == len(SAQUET_CRITERIA):
        return "fully_evaluated"
    if known:
        return "partial"
    return "failed"


def _total_iwf(flags: Mapping[str, Any]) -> Any:
    if any(flags.get(name) not in (0, 1) for name in SAQUET_CRITERIA):
        return NA
    return sum(int(flags[name]) for name in SAQUET_CRITERIA)


def _normalize_flags(raw_flags: Mapping[str, Any]) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    for name in SAQUET_CRITERIA:
        value = raw_flags.get(name)
        flags[name] = value if value in (0, 1) else NA
    return flags


def _needed_criteria(flags: Mapping[str, Any]) -> List[str]:
    return [name for name in SAQUET_CRITERIA if flags.get(name) not in (0, 1)]


def _call_apply(
    apply_fn: Callable[..., Mapping[str, Any]],
    row: Mapping[str, Any],
    criteria: Sequence[str],
) -> Mapping[str, Any]:
    if _apply_supports_criteria(apply_fn):
        return apply_fn(row, criteria=list(criteria)) or {}
    return apply_fn(row) or {}


def _base_assessment(
    row: Mapping[str, Any],
    *,
    key: str,
    fingerprint: Mapping[str, Any],
    evaluator_model: str,
    evaluator_config_version: str,
    status: str,
    flags: Mapping[str, Any],
    errors: Optional[Mapping[str, Any]] = None,
    not_evaluable_reason: str = "",
    error: str = "",
    cache_hit: bool = False,
) -> Dict[str, Any]:
    assessment: Dict[str, Any] = {
        "lecturebank_id": str(row.get("lecturebank_id") or fingerprint.get("lecturebank_id") or ""),
        "candidate_ref": str(row.get("candidate_ref") or fingerprint.get("candidate_ref") or ""),
        "status": status,
        "flags": dict(flags),
        "total_iwf_count": _total_iwf(flags),
        "errors": dict(errors or {}),
        "cache_key": key,
        "cache_hit": cache_hit,
        "evaluator_model": evaluator_model,
        "evaluator_config_version": evaluator_config_version,
        "input_fingerprint": dict(fingerprint),
    }
    if not_evaluable_reason:
        assessment["not_evaluable_reason"] = not_evaluable_reason
    if error:
        assessment["error"] = error
    return assessment


def _store_assessment(cache: Optional[SaquetQuestionCache], key: str, assessment: Mapping[str, Any]) -> None:
    if cache is None:
        return
    stored = {k: v for k, v in assessment.items() if k != "cache_hit"}
    cache.put(key, stored)


def evaluate_quiz_row(
    row: Mapping[str, Any],
    *,
    evaluator_model: str,
    evaluator_config_version: str,
    apply_fn: Callable[..., Mapping[str, Any]],
    cache: Optional[SaquetQuestionCache] = None,
    reevaluate: bool = False,
) -> Dict[str, Any]:
    fingerprint = saquet_input_fingerprint(
        row,
        evaluator_model=evaluator_model,
        evaluator_config_version=evaluator_config_version,
    )
    key = saquet_cache_key(
        row,
        evaluator_model=evaluator_model,
        evaluator_config_version=evaluator_config_version,
    )
    cached = None if reevaluate else (cache.get(key) if cache is not None else None)
    if cached is not None and str(cached.get("status") or "") == "not_evaluable":
        out = dict(cached)
        out["cache_hit"] = True
        out["cache_key"] = key
        return out
    ok, reason = saquet_question_mappable(row)
    if not ok:
        assessment = _base_assessment(
            row,
            key=key,
            fingerprint=fingerprint,
            evaluator_model=evaluator_model,
            evaluator_config_version=evaluator_config_version,
            status="not_evaluable",
            flags={name: NA for name in SAQUET_CRITERIA},
            not_evaluable_reason=reason,
        )
        _store_assessment(cache, key, assessment)
        return assessment

    flags = {name: NA for name in SAQUET_CRITERIA}
    errors: Dict[str, str] = {}
    reused = False
    if cached is not None:
        flags = _normalize_flags(cached.get("flags") or {})
        errors = dict(cached.get("errors") or {})
        reused = any(flags.get(name) in (0, 1) for name in SAQUET_CRITERIA)
        if str(cached.get("status") or "") == "fully_evaluated" and not _needed_criteria(flags):
            out = dict(cached)
            out["cache_hit"] = True
            out["cache_key"] = key
            out["flags"] = flags
            out["total_iwf_count"] = _total_iwf(flags)
            return out

    needed = _needed_criteria(flags)
    if not needed:
        assessment = _base_assessment(
            row,
            key=key,
            fingerprint=fingerprint,
            evaluator_model=evaluator_model,
            evaluator_config_version=evaluator_config_version,
            status="fully_evaluated",
            flags=flags,
            errors=errors,
            cache_hit=reused,
        )
        _store_assessment(cache, key, assessment)
        return assessment

    if _apply_supports_criteria(apply_fn):
        for name in needed:
            try:
                raw = _call_apply(apply_fn, row, [name])
            except Exception as exc:  # noqa: BLE001
                flags[name] = NA
                errors[name] = str(exc) or type(exc).__name__
            else:
                returned = dict(raw.get("flags") or {})
                if name in returned and returned[name] in (0, 1):
                    flags[name] = returned[name]
                    errors.pop(name, None)
                else:
                    flags[name] = NA
                    err_map = dict(raw.get("errors") or {})
                    if err_map.get(name):
                        errors[name] = str(err_map[name])
                    elif returned.get(name) not in (0, 1):
                        errors[name] = errors.get(name) or "criterion_result_unavailable"
                errors.update(
                    {
                        k: str(v)
                        for k, v in dict(raw.get("errors") or {}).items()
                        if k == name and v
                    }
                )
            assessment = _base_assessment(
                row,
                key=key,
                fingerprint=fingerprint,
                evaluator_model=evaluator_model,
                evaluator_config_version=evaluator_config_version,
                status=_assessment_status(flags),
                flags=flags,
                errors=errors,
                cache_hit=reused,
            )
            _store_assessment(cache, key, assessment)
        return assessment

    try:
        raw = apply_fn(row) or {}
    except Exception as exc:  # noqa: BLE001
        assessment = _base_assessment(
            row,
            key=key,
            fingerprint=fingerprint,
            evaluator_model=evaluator_model,
            evaluator_config_version=evaluator_config_version,
            status=_assessment_status(flags),
            flags=flags,
            errors={**errors, "_apply": str(exc) or type(exc).__name__},
            error=str(exc) or type(exc).__name__,
            cache_hit=reused,
        )
        _store_assessment(cache, key, assessment)
        return assessment
    returned = dict(raw.get("flags") or {})
    for name in needed:
        if returned.get(name) in (0, 1):
            flags[name] = returned[name]
            errors.pop(name, None)
        else:
            flags[name] = NA
    errors.update({k: str(v) for k, v in dict(raw.get("errors") or {}).items() if v})
    assessment = _base_assessment(
        row,
        key=key,
        fingerprint=fingerprint,
        evaluator_model=evaluator_model,
        evaluator_config_version=evaluator_config_version,
        status=_assessment_status(flags),
        flags=flags,
        errors=errors,
        cache_hit=reused,
    )
    _store_assessment(cache, key, assessment)
    return assessment


def empty_saquet_mas_row(
    *,
    lecturebank_id: str,
    run_id: str = "",
    evaluator_model: str = SAQUET_DEFAULT_MODEL,
    evaluator_config_version: str = SAQUET_CONFIG_VERSION,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "lecturebank_id": lecturebank_id,
        "run_id": run_id,
        "final_accepted_mcq_count": 0,
        "fully_evaluated_mcq_count": 0,
        "partial_mcq_count": 0,
        "failed_mcq_count": 0,
        "not_evaluable_mcq_count": 0,
        "evaluation_completion_rate": NA,
        "total_iwf_count": NA,
        "mean_iwf_per_mcq": NA,
        "zero_iwf_mcq_count": NA,
        "zero_iwf_mcq_rate": NA,
        "quiz_completeness": QUIZ_EMPTY,
        "quiz_source": "factual_quiz",
        "quiz_completeness_reason": "authoritative empty factual quiz",
        "evaluator_model": evaluator_model,
        "evaluator_config_version": evaluator_config_version,
    }
    for name in SAQUET_CRITERIA:
        flagged_f, eval_f, rate_f = saquet_criterion_fields(name)
        row[flagged_f] = 0
        row[eval_f] = 0
        row[rate_f] = NA
    return row


def aggregate_saquet_by_mas(
    assessments: Sequence[Mapping[str, Any]],
    *,
    lecturebank_id: str,
    run_id: str = "",
    accepted_count: Optional[int] = None,
    evaluator_model: str = SAQUET_DEFAULT_MODEL,
    evaluator_config_version: str = SAQUET_CONFIG_VERSION,
    quiz_completeness: str = QUIZ_COMPLETE,
    quiz_source: str = "factual_quiz",
    quiz_completeness_reason: str = "",
) -> Dict[str, Any]:
    """One MAS-run row. IWF totals use fully evaluated MCQs only."""
    n_accepted = accepted_count if accepted_count is not None else len(list(assessments))
    row = empty_saquet_mas_row(
        lecturebank_id=lecturebank_id,
        run_id=run_id,
        evaluator_model=evaluator_model,
        evaluator_config_version=evaluator_config_version,
    )
    row["quiz_completeness"] = quiz_completeness
    row["quiz_source"] = quiz_source
    row["quiz_completeness_reason"] = quiz_completeness_reason
    row["final_accepted_mcq_count"] = n_accepted if n_accepted is not None else NA
    if n_accepted == 0 and quiz_completeness in {QUIZ_EMPTY, QUIZ_COMPLETE}:
        row["quiz_completeness"] = QUIZ_EMPTY
        row["quiz_completeness_reason"] = quiz_completeness_reason or "authoritative empty factual quiz"
        return row
    if n_accepted is None or n_accepted == 0:
        if quiz_completeness == QUIZ_UNKNOWN:
            row["final_accepted_mcq_count"] = NA
            row["fully_evaluated_mcq_count"] = 0
            row["partial_mcq_count"] = 0
            row["failed_mcq_count"] = 0
            row["not_evaluable_mcq_count"] = 0
        return row

    fully = [a for a in assessments if str(a.get("status") or "") == "fully_evaluated"]
    partial = [a for a in assessments if str(a.get("status") or "") == "partial"]
    failed = [a for a in assessments if str(a.get("status") or "") == "failed"]
    not_eval = [a for a in assessments if str(a.get("status") or "") == "not_evaluable"]
    row["fully_evaluated_mcq_count"] = len(fully)
    row["partial_mcq_count"] = len(partial)
    row["failed_mcq_count"] = len(failed)
    row["not_evaluable_mcq_count"] = len(not_eval)
    row["evaluation_completion_rate"] = ratio(len(fully), n_accepted)

    mas_supported = quiz_completeness == QUIZ_COMPLETE
    if mas_supported and fully:
        totals = [int(a["total_iwf_count"]) for a in fully]
        total_iwf = sum(totals)
        zero_n = sum(1 for t in totals if t == 0)
        row["total_iwf_count"] = total_iwf
        row["mean_iwf_per_mcq"] = ratio(total_iwf, len(fully))
        row["zero_iwf_mcq_count"] = zero_n
        row["zero_iwf_mcq_rate"] = ratio(zero_n, len(fully))
    else:
        row["total_iwf_count"] = NA
        row["mean_iwf_per_mcq"] = NA
        row["zero_iwf_mcq_count"] = NA
        row["zero_iwf_mcq_rate"] = NA

    for name in SAQUET_CRITERIA:
        flagged_f, eval_f, rate_f = saquet_criterion_fields(name)
        flagged = 0
        evaluated = 0
        for assessment in assessments:
            flags = assessment.get("flags") if isinstance(assessment.get("flags"), Mapping) else {}
            flag = flags.get(name)
            if flag not in (0, 1):
                continue
            evaluated += 1
            if flag == 1:
                flagged += 1
        row[flagged_f] = flagged
        row[eval_f] = evaluated
        row[rate_f] = ratio(flagged, evaluated) if mas_supported else NA
    return row


def factual_quiz_to_rows(
    quiz: Mapping[str, Any],
    *,
    lecturebank_id: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(list(quiz.get("questions") or []), start=1):
        if not isinstance(item, Mapping):
            continue
        options = item.get("options")
        option_texts = quiz_row_options({"options": options, **item})
        slots = {label: "" for label in OPTION_LABELS}
        for i, text in enumerate(option_texts[:4]):
            slots[OPTION_LABELS[i]] = text
        correct = quiz_row_correct_text(item, option_texts)
        rows.append(
            {
                "lecturebank_id": str(item.get("lecturebank_id") or lecturebank_id),
                "candidate_ref": str(item.get("candidate_ref") or item.get("q_id") or ""),
                "q_id": str(item.get("q_id") or ""),
                "stem": str(item.get("stem") or item.get("question_text") or ""),
                "option_A": slots["A"],
                "option_B": slots["B"],
                "option_C": slots["C"],
                "option_D": slots["D"],
                "options": option_texts,
                "correct_option": correct,
                "question_index": item.get("question_index") or index,
                "question_version": item.get("question_version") or "",
            }
        )
    return rows


def persist_factual_quiz(
    metrics_dir: Path,
    lecturebank_id: str,
    quiz: Mapping[str, Any],
) -> None:
    path = Path(metrics_dir) / SAQUET_FACTUAL_QUIZ_NAME
    entries: Dict[str, Any] = {}
    if path.is_file():
        try:
            raw = load_json(path)
            stored = raw.get("entries") if isinstance(raw, Mapping) else None
            if isinstance(stored, Mapping):
                entries = {str(k): dict(v) for k, v in stored.items() if isinstance(v, Mapping)}
        except Exception:
            entries = {}
    entries[str(lecturebank_id)] = dict(quiz)
    atomic_write_json(
        path,
        {"kind": "saquet_factual_quiz", "entries": entries},
    )


def load_persisted_factual_quiz(metrics_dir: Optional[Path], lecturebank_id: str) -> Optional[Dict[str, Any]]:
    if metrics_dir is None:
        return None
    path = Path(metrics_dir) / SAQUET_FACTUAL_QUIZ_NAME
    if not path.is_file():
        return None
    try:
        raw = load_json(path)
    except Exception:
        return None
    entries = raw.get("entries") if isinstance(raw, Mapping) else None
    if not isinstance(entries, Mapping):
        return None
    hit = entries.get(str(lecturebank_id))
    return dict(hit) if isinstance(hit, Mapping) else None


def _coerce_accepted_count(value: Any) -> Optional[int]:
    if value in (None, "", "NA"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_saquet_quiz(
    payload: Optional[Mapping[str, Any]] = None,
    *,
    lecturebank_id: str,
    metrics_dir: Optional[Path] = None,
    accepted_count_metadata: Any = None,
    quiz_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Resolve the authoritative final accepted factual quiz when possible."""
    if quiz_rows is not None:
        rows = [dict(r) for r in quiz_rows]
        n = len(rows)
        completeness = QUIZ_EMPTY if n == 0 else QUIZ_COMPLETE
        return {
            "rows": rows,
            "accepted_count": n,
            "completeness": completeness,
            "source": "explicit_rows",
            "reason": "caller supplied SAQUET quiz rows",
            "factual_quiz": None,
        }

    payload_map = dict(payload) if isinstance(payload, Mapping) else {}
    quiz = payload_map.get("factual_quiz")
    if not isinstance(quiz, Mapping):
        quiz = load_persisted_factual_quiz(metrics_dir, lecturebank_id)
    if isinstance(quiz, Mapping) and str(quiz.get("kind") or "") in {
        "final_accepted_factual_quiz",
        "saquet_factual_quiz",
        "",
    }:
        rows = factual_quiz_to_rows(quiz, lecturebank_id=lecturebank_id)
        n = int(quiz.get("accepted_count") if quiz.get("accepted_count") not in (None, "") else len(rows))
        if n == 0 and not rows:
            return {
                "rows": [],
                "accepted_count": 0,
                "completeness": QUIZ_EMPTY,
                "source": "factual_quiz",
                "reason": "authoritative empty factual quiz",
                "factual_quiz": dict(quiz),
            }
        return {
            "rows": rows,
            "accepted_count": len(rows),
            "completeness": QUIZ_COMPLETE,
            "source": "factual_quiz",
            "reason": "authoritative final accepted factual QuizPack",
            "factual_quiz": dict(quiz),
        }

    case_rows = extract_quiz_questions(payload, lecturebank_id=lecturebank_id)
    meta = _coerce_accepted_count(accepted_count_metadata)
    if meta is None:
        meta = _coerce_accepted_count(payload_map.get("final_accepted_questions"))
    if meta is None and isinstance(payload_map.get("document_metrics"), Mapping):
        meta = _coerce_accepted_count(payload_map["document_metrics"].get("accepted_question_count"))

    if meta == 0 and not case_rows:
        return {
            "rows": [],
            "accepted_count": 0,
            "completeness": QUIZ_EMPTY,
            "source": "accepted_metadata",
            "reason": "authoritative accepted-question count is 0",
            "factual_quiz": None,
        }
    if meta is None:
        return {
            "rows": list(case_rows),
            "accepted_count": None,
            "completeness": QUIZ_UNKNOWN,
            "source": "explanation_cases",
            "reason": (
                "historical export has explanation cases only; authoritative "
                "accepted-question count is missing, so completeness cannot be established"
            ),
            "factual_quiz": None,
        }
    if len(case_rows) == meta:
        return {
            "rows": list(case_rows),
            "accepted_count": meta,
            "completeness": QUIZ_COMPLETE,
            "source": "explanation_cases",
            "reason": (
                "explanation cases match the authoritative accepted-question count"
            ),
            "factual_quiz": None,
        }
    return {
        "rows": list(case_rows),
        "accepted_count": meta,
        "completeness": QUIZ_INCOMPLETE,
        "source": "explanation_cases",
        "reason": (
            f"historical export has {len(case_rows)} explanation case(s) but "
            f"authoritative accepted-question count is {meta}; whole-MAS SAQUET "
            "metrics are NA and missing questions were not regenerated"
        ),
        "factual_quiz": None,
    }


def evaluate_document_saquet(
    payload: Optional[Mapping[str, Any]] = None,
    *,
    lecturebank_id: str,
    run_id: str = "",
    quiz_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    metrics_dir: Optional[Path] = None,
    apply_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    cache: Optional[SaquetQuestionCache] = None,
    reevaluate: bool = False,
    evaluator_model: Optional[str] = None,
    evaluator_config_version: str = SAQUET_CONFIG_VERSION,
    accepted_count_metadata: Any = None,
) -> Dict[str, Any]:
    """Evaluate accepted MCQs and return the per-MAS aggregate plus cache records.

    Completeness is determined from the current factual-question set, not from a
    previously written MAS CSV summary. Unchanged complete cache records are reused.
    """
    model = evaluator_model or default_saquet_model()
    resolved = resolve_saquet_quiz(
        payload,
        lecturebank_id=lecturebank_id,
        metrics_dir=metrics_dir,
        accepted_count_metadata=accepted_count_metadata,
        quiz_rows=quiz_rows,
    )
    rows = list(resolved["rows"])
    if cache is None and metrics_dir is not None:
        cache = SaquetQuestionCache(Path(metrics_dir) / SAQUET_CACHE_NAME)
    fn = apply_fn or default_apply_saquet_criteria
    assessments: List[Dict[str, Any]] = []
    for row in rows:
        assessments.append(
            evaluate_quiz_row(
                row,
                evaluator_model=model,
                evaluator_config_version=evaluator_config_version,
                apply_fn=fn,
                cache=cache,
                reevaluate=reevaluate,
            )
        )
    mas_row = aggregate_saquet_by_mas(
        assessments,
        lecturebank_id=lecturebank_id,
        run_id=str(run_id or (payload or {}).get("run_id") or ""),
        accepted_count=resolved["accepted_count"],
        evaluator_model=model,
        evaluator_config_version=evaluator_config_version,
        quiz_completeness=str(resolved["completeness"]),
        quiz_source=str(resolved["source"]),
        quiz_completeness_reason=str(resolved["reason"]),
    )
    if metrics_dir is not None and isinstance(resolved.get("factual_quiz"), Mapping):
        persist_factual_quiz(Path(metrics_dir), lecturebank_id, resolved["factual_quiz"])
    return {
        "mas_row": mas_row,
        "assessments": assessments,
        "accepted_count": resolved["accepted_count"],
        "completeness": resolved["completeness"],
        "quiz_source": resolved["source"],
        "cache_hits": sum(1 for a in assessments if a.get("cache_hit") is True),
    }
