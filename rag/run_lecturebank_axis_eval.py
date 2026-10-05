"""LectureBank AXIS evaluation: RAG → MAS → AXIS → aggregate metrics.

Does not write a full experiment.json per PDF. Keeps the presentation batch
workflow unchanged.

Examples (from EMaES/rag):

  python run_lecturebank_axis_eval.py --lecturebank-dir ../LectureBank --list-ids --search "machine learning" --limit 50
  python run_lecturebank_axis_eval.py --lecturebank-dir ../lecturebank --extraction-jsonl ../lecturebank/pdf_extraction_results.jsonl --id 137 --allow-any-count
"""

from __future__ import annotations

from mas_langgraph.evaluation.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
