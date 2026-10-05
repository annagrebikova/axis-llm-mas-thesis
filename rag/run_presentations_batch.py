"""Batch-process PDF presentations: index independently, then generate MCQs/AXIS.

Default stage is RAG-only and does not start MCQ generation or AXIS.
Default --max-questions 10 is a factual-generation cap (final accepted MCQs).

Examples (from EMaES/rag):

  python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --list-pdfs
  python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage rag --pdf "lecture_01.pdf"
  python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage generate --pdf "lecture_01.pdf" --max-questions 10
  python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage all --pdf "lecture_01.pdf" --pdf "lecture_02.pdf" --max-questions 5
"""

from __future__ import annotations

from mas_langgraph.batch.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
