# RAG + MAS Pipeline

| Component | Path | Purpose |
|-----------|------|---------|
| **MAS orchestrator** | `mas_langgraph/` | Main multi-agent quiz pipeline (classic + LangGraph) |
| RAG monolith | `rag_agents_cooperative.py` | PDF tools, vector store, LLM agents |
| Indexing | `rag_pipeline.py` | Extract PDF → vector store |
| Launcher | `run_mas.py` | Shortcut for `python -m mas_langgraph` |

## Quick start

```bash
cd EMaES/rag

# Stub (no API key)
python -m mas_langgraph --orchestrate --stub

# LangGraph orchestrator
python -m mas_langgraph --orchestrate --stub --orchestrator langgraph

# Real RAG (after indexing)
python rag_pipeline.py --pdf-id 141
python -m mas_langgraph --orchestrate --pdf-id 141 --dump-pack quiz.json
```

Set `OPENAI_API_KEY` in `EMaES/.env`. See:

- [`mas_langgraph/README.md`](mas_langgraph/README.md) — package overview and AXIS quick-start
- [`docs/axis_explainer.md`](docs/axis_explainer.md) — explanation loop
- [`docs/architecture.md`](docs/architecture.md) — final architecture
- [`docs/reproducibility.md`](docs/reproducibility.md) — tests and environment

Install deps: `pip install -r requirements_rag.txt`

## Folder of presentations (batch)

Put PDF slides in a folder. PowerPoint files (`.ppt` / `.pptx`) are **not** read as PDFs — export them to PDF first. Each presentation gets its own vector index; retrieval never mixes documents.

All commands below are from `EMaES/rag`. Default `--stage rag` indexes only and **does not** generate MCQs or AXIS explanations. Default `--max-questions 10` is the **target/cap of final accepted MCQs per PDF** during the factual workflow (the quiz may contain fewer if valid evidence-supported questions are exhausted). AXIS then explains that quiz unless `--max-explain-questions` is set. `--recursive` applies to listing, selection, and processing.

```bash
# 1. Prepare / index every PDF (RAG only)
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs

# 2. Generate MCQs and AXIS from prepared indexes (try up to 10 accepted MCQs; explain all)
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage generate

# 3. Index and generate in one pass (explicit)
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage all

# 4. Resume interrupted work (hash + config + artifacts must still match)
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage all --resume

# 5. Smoke run: try 1 accepted MCQ, fewer AXIS rounds
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage all --max-questions 1 --max-rounds 3

# List selectable PDFs and document IDs (read-only)
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --list-pdfs

# Index one PDF without moving the others
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage rag --pdf "lecture_01.pdf"

# Generate for one PDF
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage generate --pdf "lecture_01.pdf" --max-questions 10

# Two PDFs, smaller generation cap
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage all --pdf "lecture_01.pdf" --pdf "lecture_02.pdf" --max-questions 5

# Target 10 accepted MCQs; explain only 2 of them
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage all --max-questions 10 --max-explain-questions 2

# Select by document ID from --list-pdfs
python run_presentations_batch.py --input-dir presentations --output-dir presentation_outputs --stage generate --document-id "<ID shown by --list-pdfs>"
```

Useful flags: `--recursive` (include subfolders), `--overwrite` (this batch’s selected presentation/stage only), `--chunk-size`, `--chunk-overlap`, `--embedding-model`, `--llm-model`, `--seed`, `--max-explain-questions` (optional cap on how many accepted MCQs receive Trace / Trace+Provenance / AXIS; default is all accepted questions). `--max-questions` is a factual generation target, not an explanation-only smoke cap.

Outputs:

```
presentation_outputs/manifest.json
presentation_outputs/batch_summary.csv
presentation_outputs/presentation_<id>/
  source.md
  rag_metadata.json
  vector_store/
  results.md          # after generate
  experiment.json     # after generate (Streamlit-compatible)
  timing.csv          # after generate
```

Existing single-PDF commands still use collection id `141`:

```bash
python rag_pipeline.py --pdf-id 141
python run_explanation_experiment_141.py
```

## LectureBank AXIS evaluation

Select LectureBank documents by ID from a **local** clone (metadata from `alldata.tsv`, not the internet). Each PDF is indexed in an isolated vector store (`lecturebank_<ID>`), then run through the current factual LangGraph MAS and AXIS. The workflow does **not** write `experiment.json`, `results.md`, or `timing.csv` per PDF. Aggregate thesis metrics go to `<output-dir>/metrics/`.

Default selection is **exactly 10** unique IDs. Use `--expected-count 100` for the full evaluation, or `--allow-any-count` for smoke tests. `--expected-count` and `--allow-any-count` cannot be combined.

The presentation batch above is unchanged.

```bash
cd EMaES/rag

# Inspect IDs (no RAG / LLM)
python run_lecturebank_axis_eval.py \
  --lecturebank-dir ../LectureBank \
  --list-ids \
  --search "machine learning" \
  --limit 50

# Check that all selected PDFs can be obtained (no RAG / LLM)
python run_lecturebank_axis_eval.py \
  --lecturebank-dir ../LectureBank \
  --ids-file lecturebank_100_ids.txt \
  --expected-count 100 \
  --pdf-dir ../LectureBank_pdfs \
  --preflight

# Recommended 100-document low-disk evaluation from pre-extracted JSONL
python run_lecturebank_axis_eval.py `
  --lecturebank-dir ..\lecturebank `
  --extraction-jsonl ..\lecturebank\pdf_extraction_results.jsonl `
  --ids-file .\lecturebank_100_ids.txt `
  --expected-count 100 `
  --output-dir lecturebank_axis_eval_100 `
  --stage all `
  --max-questions 10 `
  --max-rounds 5 `
  --temporary-vector-stores `
  --allow-partial `
  --resume
```

`--extraction-jsonl` is optional. When supplied, a matching JSONL record with usable `text.pages_text` is preferred over PDFs. Missing/empty/duplicate JSONL IDs fall back to `--pdf-dir`, cache, then download. JSONL text is not treated as already vectorized: the existing RAG chunking, embeddings, and isolated Chroma store still run. No fake PDF is written.

`--temporary-vector-stores` builds each document's Chroma index in a Python temporary directory and deletes it **after** that document's aggregate metrics and manifest completion flags are written. Isolation is unchanged: one LectureBank document still has one exclusive store; retrieval never mixes documents.

```powershell
# JSONL preflight (no RAG / LLM)
python run_lecturebank_axis_eval.py `
  --lecturebank-dir ..\lecturebank `
  --extraction-jsonl ..\lecturebank\pdf_extraction_results.jsonl `
  --ids-file .\lecturebank_100_ids.txt `
  --expected-count 100 `
  --preflight
```

If you prefer persistent stores that are deleted in place after each completed document:

```powershell
python run_lecturebank_axis_eval.py `
  --lecturebank-dir ..\LectureBank `
  --ids-file lecturebank_100_ids.txt `
  --expected-count 100 `
  --pdf-dir ..\LectureBank_pdfs `
  --output-dir lecturebank_axis_eval_100 `
  --stage all `
  --max-questions 10 `
  --max-rounds 5 `
  --allow-partial `
  --cleanup-vector-store `
  --resume
```

Ten-ID default and smoke test:

```bash
# Full evaluation (exactly 10 unique IDs)
python run_lecturebank_axis_eval.py \
  --lecturebank-dir ../LectureBank \
  --ids "101,205,333,410,512,640,711,812,955,1102" \
  --output-dir lecturebank_axis_eval \
  --stage all \
  --max-questions 10 \
  --max-rounds 5

# Smoke test (any ID count)
python run_lecturebank_axis_eval.py \
  --lecturebank-dir ../LectureBank \
  --id 123 \
  --allow-any-count \
  --max-questions 1 \
  --max-explain-questions 1 \
  --max-rounds 1
```

Constrained vs hybrid interrogation (same JSONL experiment, **separate output directories**):

```powershell
python run_lecturebank_axis_eval.py `
  --extraction-jsonl ..\lecturebank\pdf_extraction_results.jsonl `
  --jsonl-authoritative `
  --first-jsonl 10 `
  --allow-any-count `
  --output-dir lecturebank_axis_eval_constrained `
  --stage all `
  --max-questions 10 `
  --max-rounds 5 `
  --axis-interrogation-mode constrained `
  --temporary-vector-stores `
  --resume

python run_lecturebank_axis_eval.py `
  --extraction-jsonl ..\lecturebank\pdf_extraction_results.jsonl `
  --jsonl-authoritative `
  --first-jsonl 10 `
  --allow-any-count `
  --output-dir lecturebank_axis_eval_hybrid `
  --stage all `
  --max-questions 10 `
  --max-rounds 5 `
  --axis-interrogation-mode hybrid `
  --temporary-vector-stores `
  --resume
```

`--resume` treats `constrained` and `hybrid` as different generation configs. Do not mix them in one output directory.

Useful flags: `--id` (repeatable), `--ids-file lecturebank_100_ids.txt`, `--extraction-jsonl`, `--pdf-dir` (local PDFs named `{ID}.pdf`, used in place; `14` will not match `141.pdf`), `--no-download`, `--resume`, `--overwrite`, `--allow-partial`, `--expected-count`, `--preflight`, `--temporary-vector-stores`, `--cleanup-vector-store`, `--cleanup-downloaded-pdfs`, `--embedding-model`, `--chunk-size`, `--chunk-overlap`, `--llm-model`, `--temperature`, `--seed`, `--question`, `--axis-interrogation-mode {constrained,hybrid}`. `--max-explain-questions` is independent of `--max-questions` and defaults to all accepted MCQs. `--axis-interrogation-mode` defaults to `constrained` (existing exact-query copy architecture). `hybrid` is a separate interrogation-generation mode and must use a different `--output-dir` / `--resume` fingerprint.

`--pdf-dir` files are never copied into `pdf_cache/` and are never deleted. `--cleanup-downloaded-pdfs` only removes PDFs this workflow downloaded into `pdf_cache/`, and only after that document's metrics are persisted.

If a LectureBank URL is not a direct PDF, that ID fails clearly; put the file in `--pdf-dir` yourself. Failed IDs stay in the denominators and are never replaced.

### What is retained vs deleted

After **one completed document** with `--temporary-vector-stores` (or `--cleanup-vector-store`):

```
lecturebank_axis_eval_100/lecturebank_eval_manifest.json
lecturebank_axis_eval_100/metrics/
  document_metrics.csv
  interrogation_attempts.csv
  replay_comparisons.csv
  provenance_edges.csv
  distributions.csv
  round_metrics.csv
  overall_metrics.csv
  report.md
```

Optional: `pdf_cache/<id>.pdf` for downloaded sources (not for `--pdf-dir` files). The document's Chroma vector store is deleted.

After **all 100 documents**, the same aggregate files remain (now containing every completed ID). There are not 100 permanent vector stores.

`--resume` skips a document when the manifest marks it completed **and** its metric rows are already in the CSVs, even if the vector store was deleted. An incomplete document reuses a valid leftover store if present, otherwise rebuilds RAG.

Deleting vector stores saves disk space. If you deliberately rerun a **completed** document with a changed experiment configuration or `--overwrite`, RAG must be rebuilt because the index is gone.

The global manifest stores reproducibility metadata (LectureBank ID, title, URL, PDF SHA256, acquisition type, embedding/chunk settings, LLM/temperature/seed, max questions/rounds, completion flags). It does not store the full experiment payload.

Outputs (aggregates only):

```
lecturebank_axis_eval/pdf_cache/<id>.pdf   # downloads only
lecturebank_axis_eval/lecturebank_eval_manifest.json
lecturebank_axis_eval/metrics/
  document_metrics.csv
  interrogation_attempts.csv
  replay_comparisons.csv
  provenance_edges.csv
  distributions.csv
  round_metrics.csv
  overall_metrics.csv
  report.md
```

## Explanation graph explorer (researcher)

JSON viewer only — no workflow, replay, or LLM calls.

```bash
cd EMaES/rag
streamlit run streamlit_explanation_graphs.py
```

From the repository root: `streamlit run rag/streamlit_explanation_graphs.py`.

Load an exported `quiz_explanation_experiment_*.json`. Cases need `visualization_data` from a current export.

## Explanation presentation (participant)

Blinded A/B/C presentation for human evaluation. JSON viewer only — no workflow, replay, LLM calls, or response collection. Graph detail is locked to **Essential**.

Requires a current export that contains `visualization_data` for every case. Older files without that payload fail closed rather than showing text-only explanations.

```bash
cd EMaES/rag
EXPLANATION_EXPERIMENT_JSON=/path/to/current.json \
  streamlit run streamlit_explanation_participant.py
```

On Windows PowerShell:

```powershell
cd EMaES/rag
$env:EXPLANATION_EXPERIMENT_JSON = "C:\path\to\current.json"
streamlit run streamlit_explanation_participant.py
```

From the repository root: `streamlit run rag/streamlit_explanation_participant.py` with the same environment variable.

Do not use the researcher explorer for blinded evaluation. The participant app never shows true condition names (Trace / Trace + Provenance / AXIS).
