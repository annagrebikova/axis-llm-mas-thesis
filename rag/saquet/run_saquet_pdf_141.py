"""Run the supplied SAQUET IWF criteria on PDF-141 final MCQs.

Uses ``saquet.saquet_core`` directly. Does not modify MAS generation logic.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

# Ensure rag/ is importable when run as a script.
_RAG_DIR = Path(__file__).resolve().parent.parent
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))


ALL_CRITERIA = [
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


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=True)
    except Exception:
        pass


def _verify_gpt4o_access() -> None:
    """Stop before evaluation if gpt-4o cannot be called."""
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run SAQUET with gpt-4o.")
    client = OpenAI(api_key=key)
    try:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Reply with the single digit 1."}],
            max_tokens=5,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"GPT-4o access check failed; refusing to run SAQUET evaluation: {exc}"
        ) from exc


def evaluate_questions(questions: Sequence[Any]) -> Dict[str, List[int]]:
    """Apply all 19 SAQUET criteria. Returns criterion -> list of 0/1 IWF flags."""
    import importlib

    core = importlib.import_module("saquet.saquet_core")

    # Prefer the module's all_criteria (source of truth).
    criteria = list(getattr(core, "all_criteria", None) or ALL_CRITERIA)
    if list(criteria) != list(ALL_CRITERIA):
        raise RuntimeError(
            "saquet_core.all_criteria does not match the required 19 IWF criteria."
        )

    all_data: Dict[str, List[int]] = {}
    for name in criteria:
        print(f"Applying criteria: {name}", flush=True)
        fn = getattr(core, name)
        # Original SAQUET conversion: 0 if criterion returns True else 1
        auto_iwf_results = [0 if fn(q) else 1 for q in questions]
        all_data[name] = auto_iwf_results
    return all_data


def build_rows(
    questions: Sequence[Any],
    criterion_flags: Mapping[str, Sequence[int]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(questions):
        flags = {c: int(criterion_flags[c][i]) for c in ALL_CRITERIA}
        total = sum(flags.values())
        rows.append(
            {
                "q_id": q.qid,
                "question_text": q.stem,
                "correct_answer": q.correct_option,
                "options": list(q.options),
                **flags,
                "total_iwf_count": total,
            }
        )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    totals = [int(r["total_iwf_count"]) for r in rows]
    total_iwfs = sum(totals)
    zero_n = sum(1 for t in totals if t == 0)
    criterion_stats = []
    for c in ALL_CRITERIA:
        flagged = sum(int(r[c]) for r in rows)
        criterion_stats.append(
            {
                "criterion": c,
                "number_flagged": flagged,
                "percentage_flagged": (100.0 * flagged / n) if n else 0.0,
            }
        )
    min_t = min(totals) if totals else None
    max_t = max(totals) if totals else None
    fewest = [r["q_id"] for r in rows if r["total_iwf_count"] == min_t]
    most = [r["q_id"] for r in rows if r["total_iwf_count"] == max_t]
    return {
        "pdf_id": "141",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": "gpt-4o",
        "n_mcqs": n,
        "total_detected_iwfs": total_iwfs,
        "mean_iwfs_per_mcq": (total_iwfs / n) if n else 0.0,
        "median_iwfs_per_mcq": float(statistics.median(totals)) if totals else 0.0,
        "min_iwf_count": min_t,
        "max_iwf_count": max_t,
        "n_mcqs_zero_iwf": zero_n,
        "pct_mcqs_zero_iwf": (100.0 * zero_n / n) if n else 0.0,
        "fewest_iwf_questions": fewest,
        "most_iwf_questions": most,
        "criteria": criterion_stats,
        "semantics": {
            "0": "criterion passed / Item-Writing Flaw not detected",
            "1": "Item-Writing Flaw detected",
        },
        "implementation": "supplied SAQUET notebook converted to saquet.saquet_core",
        "criteria_executed": list(ALL_CRITERIA),
        "n_criteria": len(ALL_CRITERIA),
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import pandas as pd

    flat = []
    for r in rows:
        item = dict(r)
        item["options"] = " | ".join(item.get("options") or [])
        flat.append(item)
    cols = ["q_id", "question_text", "correct_answer", "options"] + ALL_CRITERIA + [
        "total_iwf_count"
    ]
    pd.DataFrame(flat)[cols].to_csv(path, index=False, encoding="utf-8")


def write_markdown(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    path: Path,
    *,
    deviations: Sequence[str],
) -> None:
    lines = [
        "# SAQUET IWF Evaluation — PDF 141",
        "",
        "## 1. Experiment scope",
        "",
        "- PDF: **141**",
        f"- Evaluated MCQs: **{summary['n_mcqs']}**",
        "- Source quiz: `quiz_141.json` (existing final accepted questions; not regenerated)",
        "",
        "## 2. Implementation",
        "",
        "The supplied SAQUET implementation was used directly "
        "(`rag/saquet/saquet_core.py`).",
        "",
        f"- Model: `{summary['model']}`",
        f"- Criteria executed: **{summary['n_criteria']}**",
        "",
        "## 3. Result semantics",
        "",
        "```text",
        "0 = IWF not detected (criterion function returned True)",
        "1 = IWF detected (criterion function returned False)",
        "```",
        "",
        "Original conversion preserved:",
        "",
        "```python",
        "auto_iwf_results = [0 if globals()[criteria](q) else 1 for q in questions]",
        "```",
        "",
        "## 4. Questions and total IWF counts",
        "",
        "| q_id | total_iwf_count |",
        "|---|---:|",
    ]
    for r in rows:
        lines.append(f"| `{r['q_id']}` | {r['total_iwf_count']} |")
    lines.extend(
        [
            "",
            "## 5. Criterion flag rates",
            "",
            "| criterion | number_flagged | percentage_flagged |",
            "|---|---:|---:|",
        ]
    )
    for c in summary["criteria"]:
        lines.append(
            f"| `{c['criterion']}` | {c['number_flagged']} | "
            f"{c['percentage_flagged']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 6. Aggregate statistics",
            "",
            f"- Total detected IWFs: **{summary['total_detected_iwfs']}**",
            f"- Mean IWFs per MCQ: **{summary['mean_iwfs_per_mcq']:.3f}**",
            f"- Median IWFs per MCQ: **{summary['median_iwfs_per_mcq']:.3f}**",
            f"- Min IWF count: **{summary['min_iwf_count']}**",
            f"- Max IWF count: **{summary['max_iwf_count']}**",
            f"- MCQs with zero IWFs: **{summary['n_mcqs_zero_iwf']}** "
            f"({summary['pct_mcqs_zero_iwf']:.1f}%)",
            f"- Fewest IWFs: {', '.join('`'+x+'`' for x in summary['fewest_iwf_questions'])}",
            f"- Most IWFs: {', '.join('`'+x+'`' for x in summary['most_iwf_questions'])}",
            "",
            "## 7. Technical deviations from supplied SAQUET",
            "",
        ]
    )
    if deviations:
        for d in deviations:
            lines.append(f"- {d}")
    else:
        lines.append("- None.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    _load_dotenv()

    quiz_path = _RAG_DIR / "quiz_141.json"
    out_dir = _RAG_DIR / "results" / "saquet" / "pdf_141"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Quiz input: {quiz_path}")
    if not quiz_path.is_file():
        print("ERROR: quiz_141.json not found")
        return 1

    print("Checking GPT-4o access...")
    try:
        _verify_gpt4o_access()
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKER: {exc}")
        return 2
    print("GPT-4o access OK.")

    print("Loading SAQUET core (NLP models may download on first run)...")
    try:
        from saquet.adapter import quiz_path_to_saquet_questions, validate_records, extract_quiz_records
        from saquet.saquet_core import all_criteria, model_engine
    except Exception as exc:  # noqa: BLE001
        print("BLOCKER while importing SAQUET / dependencies:")
        traceback.print_exc()
        print(f"Error: {exc}")
        return 3

    if model_engine != "gpt-4o":
        print(f"BLOCKER: expected model_engine='gpt-4o', got {model_engine!r}")
        return 4
    if list(all_criteria) != ALL_CRITERIA:
        print("BLOCKER: all_criteria mismatch")
        return 5

    records = extract_quiz_records(quiz_path)
    errs = validate_records(records)
    if errs:
        print("BLOCKER: quiz adapter validation failed:")
        for e in errs:
            print(f"  - {e}")
        return 6

    questions = quiz_path_to_saquet_questions(quiz_path)
    print(f"Evaluating {len(questions)} MCQs with {len(ALL_CRITERIA)} criteria...")

    try:
        flags = evaluate_questions(questions)
    except Exception as exc:  # noqa: BLE001
        print("BLOCKER during SAQUET evaluation:")
        traceback.print_exc()
        print(f"Error: {exc}")
        return 7

    rows = build_rows(questions, flags)
    summary = summarize(rows)
    deviations = []
    dev_path = Path(__file__).resolve().parent / "TECHNICAL_DEVIATIONS.md"
    if dev_path.is_file():
        deviations = [
            ln[2:].strip() if ln.startswith("- ") else ln.strip()
            for ln in dev_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    csv_path = out_dir / "saquet_141_results.csv"
    json_path = out_dir / "saquet_141_summary.json"
    md_path = out_dir / "saquet_141_report.md"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(rows, summary, md_path, deviations=deviations)

    print("\nDone.")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print("\nq_id | total_iwf_count")
    for r in rows:
        print(f"{r['q_id']} | {r['total_iwf_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
