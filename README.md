# Replay-grounded explanations for LLM-based multi-agent systems

Master's thesis implementation by **Anna Grebíková**, MSc Data Science, **FH Technikum Wien**.

This project investigates how AXIS can be generalized to explain outcomes of LLM-based multi-agent systems (MAS) with deterministic workflow orchestration. Its application system generates evidence-grounded multiple-choice questions (MCQs) from teaching materials.

The MAS extracts and retrieves evidence, derives knowledge atoms, plans and generates questions, verifies them, and applies an acceptance policy. AXIS investigates selected outcomes by changing workflow artifacts at saved checkpoints and replaying the downstream workflow. It compares a no-intervention replay (NOOP) with a counterfactual replay and synthesizes an explanation from the available evidence.

Deterministic orchestration means that routing rules are explicitly defined. It does not mean that fresh LLM calls always return identical answers. The workflow LLM ledger reuses recorded responses for matching requests; novel counterfactual requests can still require new model calls.

## Research scope

The thesis addresses:

1. Requirements and design principles for adapting AXIS to checkpointed LLM workflows.
2. Intervention selection, replay reliability, and effect discovery under a limited interrogation budget.
3. Perceived explanation quality: **Trace**, **Trace + Provenance**, and **AXIS**.
4. Design and quality of the MCQ-generation application.

The prototype provides evidence about the tested interventions and recorded executions. It does not establish exhaustive causal explanations, universal reproducibility, or classroom readiness of every accepted question.


## 1. What belongs in this repository?

Use the GitHub repository root as the equivalent of your local `EMaES` folder. Retain the `rag/` directory beneath it.

| Path to include | Purpose |
| --- | --- |
| `README.md` | This introduction and run guide. |
| `.gitignore` | Excludes credentials, environments, caches, and generated runs. |
| `.env.example` | Placeholder for the API key; no real credential. |
| `rag/mas_langgraph/` | Complete MAS, AXIS, replay, configuration, visualization, and tests package. Keep its YAML files and prompts. |
| `rag/saquet/` | SAQUET evaluation integration, dependency list, tests, and technical notes. |
| `rag/docs/` | Architecture and methodological documentation. |
| `rag/rag_pipeline.py` | PDF extraction, chunking, embeddings, and vector indexing. |
| `rag/rag_agents_cooperative.py` | Cooperative state, tools, and LLM-backed workflow functionality. |
| `rag/run_presentations_batch.py` | Main entry point for your own PDFs: RAG, MAS, and AXIS. |
| `rag/run_lecturebank_axis_eval.py` | LectureBank benchmark and metrics entry point. |
| `rag/streamlit_explanation_graphs.py` | Viewer for exported experiment JSON. |
| `rag/requirements_rag.txt` | Main dependency list. |
| `rag/README.md` | Existing component-specific documentation; retain alongside this root guide. |
| `examples/` (optional, create it) | A reviewed, shareable experiment JSON for the viewer and an authorized sample PDF. |
| `analysis/` (optional, create it) | Selected analysis notebooks and anonymized metrics used in the thesis. |

For the documented commands, old root-level `run_*141*.py` scripts, individual `quiz_explanation_*` exports, and old evaluation folders are not required. If retaining a legacy launcher, also retain and document its specific inputs.

Do not upload `.env`, API keys, `.venv`, model caches, vector stores, large `runtime.pkl` snapshots, or all historical run folders. Share teaching materials and participant data only when you have permission; experiment JSON can contain source excerpts. The example JSON is a separate deliverable: copying application code alone does not create a demo dataset.

Keep applicable third-party license and attribution files. The selected folder list is a source-code selection, not a license grant for imported material.

## 2. Where the implementation lives

All paths below are relative to `rag/`.

| Path | Responsibility |
| --- | --- |
| `mas_langgraph/core/` | Core runtime and workflow graph construction. |
| `mas_langgraph/orchestrator/` | Workflow nodes, routing, and orchestration. |
| `mas_langgraph/agents/` and `tools/` | Agent and tool abstractions. |
| `mas_langgraph/domains/quiz/` | Quiz-specific processing, verification, and acceptance logic. |
| `mas_langgraph/cases/mcq/` | MCQ case definition, artifact registration, and provenance bindings. |
| `mas_langgraph/cases/mcq/inputs/` | Artifact types, intervention contracts, schemas, and prompts. |
| `mas_langgraph/artifacts/` and `provenance/` | Artifact identities, versions, and recorded dependencies. |
| `mas_langgraph/axis/` | AXIS interrogation and explanation logic. |
| `mas_langgraph/replay/` | Checkpoint replay, intervention handling, and workflow LLM response ledger. |
| `mas_langgraph/experiment/` | Explanation experiments, synthesis, and explanation-model configuration. |
| `mas_langgraph/batch/` | Processing of user-supplied PDF folders. |
| `mas_langgraph/evaluation/` | LectureBank runs, comparisons, metrics, and optional evaluators. |
| `mas_langgraph/visualization/` | Graph building and Streamlit display helpers. |
| `mas_langgraph/tests/` | Regression tests. |

## 3. Set up Python

The following commands use **Windows PowerShell**. Start in the repository root, the folder containing `rag/`. Use a Python environment compatible with the supplied requirements; this guide does not claim a newly validated Python version.

Create an isolated environment and use its interpreter directly. Activation is not required:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\rag\requirements_rag.txt
.\.venv\Scripts\python.exe -m pip install streamlit PyYAML
```

`streamlit` is not listed in the inspected main requirements file. PyYAML is included explicitly for YAML configuration support.

Then move into `rag/`:

```powershell
Set-Location .\rag
```

**All remaining run commands assume this working directory.** The interpreter path is consequently `..\.venv\Scripts\python.exe`.

The embedding model is downloaded on first use unless cached. Live generation and optional LLM-based evaluation require network access and incur provider charges.

## 4. Configure the API key and models

For generation, copy the template to `.env` in the repository root and edit it locally:

```powershell
Copy-Item ..\.env.example ..\.env
notepad ..\.env
```

Fill in `OPENAI_API_KEY`. Keep `.env` out of Git. The MAS integration also checks `rag/.env`; use one intended configuration to avoid ambiguity.

The inspected code defines these model identifiers:

| Role | Configured model |
| --- | --- |
| MCQ workflow | `gpt-4.1-mini-2025-04-14` |
| AXIS interrogation | `gpt-5.4-mini-2026-03-17` |
| Explanation synthesis | `gpt-5.4-mini-2026-03-17` |
| SAQUET LLM criteria | `gpt-4o` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |

These are identifiers stored in the inspected source, not a guarantee of account access or continuing provider availability. Explanation settings are in `mas_langgraph/experiment/llm_config.py`; workflow defaults also occur in `mas_langgraph/input/rag_config.py`. Changing `--llm-model` in the batch runner changes the workflow model, not automatically every explanation model. Record any model changes with the run configuration.

## 5. View a saved experiment in Streamlit

This is the quickest demonstration. It does not generate questions or run AXIS and does not require an API key.

```powershell
..\.venv\Scripts\python.exe -m streamlit run .\streamlit_explanation_graphs.py
```

1. Keep the terminal open.
2. Open the local URL printed by Streamlit, usually `http://localhost:8501`.
3. Upload an exported explanation experiment JSON.
4. Select a question and inspect its explanation conditions, recorded artifacts, and available replay comparisons.
5. Press **Ctrl+C** in the terminal to stop the viewer.

Compatible inputs include `presentation_outputs/presentation_<id>/experiment.json` and LectureBank's optional `streamlit/quiz_explanation_experiment_<id>.json` export. A raw quiz JSON containing only questions is not a substitute for the experiment export.

For the previously completed document 141 sweep, the saved file was:

```text
lecturebank_axis_sweep_jsonl141_10mcqs_v2/presentation_lecturebank_141/experiment.json
```

That path identifies a saved local experiment; it is not automatically included with the source code. An absent field in an older export cannot be reconstructed merely by opening the viewer.

## 6. Run MAS and AXIS on your own PDF

### Step 1: provide a PDF

```powershell
New-Item -ItemType Directory -Force .\presentations
```

Place a text-based teaching-material PDF in this folder, for example `lecture_01.pdf`. Export PowerPoint files to PDF first. Image-only slides may require OCR before this workflow can extract usable text.

List available PDFs and generated document IDs:

```powershell
..\.venv\Scripts\python.exe run_presentations_batch.py --input-dir .\presentations --output-dir .\presentation_outputs --list-pdfs
```

### Step 2: start with one question

This performs PDF indexing, factual MAS generation, and AXIS explanation in one command:

```powershell
..\.venv\Scripts\python.exe run_presentations_batch.py `
  --input-dir .\presentations `
  --output-dir .\presentation_outputs_smoke `
  --pdf "lecture_01.pdf" `
  --stage all `
  --max-questions 1 `
  --max-explain-questions 1 `
  --max-rounds 2 `
  --axis-interrogation-mode hybrid `
  --question "Why was this question selected?"
```

Replace the example filename with your actual PDF name. `--max-questions` is a cap/target, not a guarantee that the system accepts that many questions.

### Step 3: generate a larger quiz

Use a new output folder for a differently configured run:

```powershell
..\.venv\Scripts\python.exe run_presentations_batch.py `
  --input-dir .\presentations `
  --output-dir .\presentation_outputs `
  --pdf "lecture_01.pdf" `
  --stage all `
  --max-questions 10 `
  --max-explain-questions 10 `
  --max-rounds 5 `
  --seed 42 `
  --temperature 0.0 `
  --axis-interrogation-mode hybrid `
  --question "Why was this question selected?"
```

Remove `--pdf` to process all PDFs in the input folder. Use `--recursive` to include subfolders.

The presentation runner's default stage is **RAG only**. Specify `--stage all` to obtain questions and explanations. Its `--stage generate` runs **MAS and AXIS together** using prepared indexes; it is not an AXIS-only rerun of a saved quiz.

### Optional: split indexing and generation

```powershell
..\.venv\Scripts\python.exe run_presentations_batch.py --input-dir .\presentations --output-dir .\presentation_outputs_split --stage rag

..\.venv\Scripts\python.exe run_presentations_batch.py --input-dir .\presentations --output-dir .\presentation_outputs_split --stage generate --max-questions 10 --max-explain-questions 10 --max-rounds 5 --axis-interrogation-mode hybrid
```

The input files and indexing settings must remain compatible between stages. The inspected presentation CLI does not expose every LectureBank evaluation flag; do not copy SAQUET or hybrid-source-pool options into it without checking `--help`.

## 7. Run the MAS without AXIS

A stub run exercises the orchestrator without real PDF processing or provider calls:

```powershell
..\.venv\Scripts\python.exe -m mas_langgraph --orchestrate --stub --orchestrator langgraph
```

For a real standalone MAS run, prepare a PDF index as above, then inspect its `rag_metadata.json` and batch manifest for the `document_id`, `collection_name`, and `index_dir`.

```powershell
..\.venv\Scripts\python.exe -m mas_langgraph `
  --orchestrate `
  --orchestrator langgraph `
  --pdf-id "YOUR_INDEXED_COLLECTION_NAME" `
  --vector-store ".\presentation_outputs_split\presentation_YOUR_DOCUMENT_ID\vector_store" `
  --dump-pack ".\quiz_mas_only.json"
```

Replace both placeholders with the actual indexed values. The `--pdf-id` argument is used to load the vector collection. This is a new MAS execution, not a continuation of an earlier explanation experiment. Its quiz export does not contain the three explanation conditions required by the viewer. This standalone CLI also has different options from the presentation runner; the `--max-questions` option above belongs to the batch runner.

## 8. Locate the outputs

| Output | Contents |
| --- | --- |
| `presentation_outputs/manifest.json` | Document IDs, configuration, status, and file paths. |
| `presentation_outputs/batch_summary.csv` | Batch-level summary. |
| `presentation_outputs/presentation_<id>/source.md` | Extracted source text. |
| `presentation_outputs/presentation_<id>/rag_metadata.json` | Indexing and document metadata. |
| `presentation_outputs/presentation_<id>/vector_store/` | Local retrieval index. |
| `presentation_outputs/presentation_<id>/results.md` | Human-readable generation/explanation results. |
| `presentation_outputs/presentation_<id>/experiment.json` | Structured experiment for Streamlit. |
| `presentation_outputs/presentation_<id>/timing.csv` | Recorded timings. |

The `<id>` is assigned by the batch runner; it is not necessarily the PDF filename or a LectureBank number. Inspect the manifest rather than guessing it.

## 9. Optional: LectureBank metrics, SAQUET, and concept assessment

LectureBank is a separate dataset, not an application-code dependency. Your own-PDF workflow and the saved-experiment viewer do not require it.

For the extraction-based benchmark, supply your existing `pdf_extraction_results.jsonl`. The examples below place it in `lecturebank/` beside `rag/`. With `--jsonl-authoritative`, selection uses the extraction file rather than requiring a metadata-catalogue match. Do not commit the dataset automatically with the code.

Install SAQUET dependencies in the same environment used to run the evaluator:

```powershell
..\.venv\Scripts\python.exe -m pip install -r .\saquet\requirements-saquet.txt
..\.venv\Scripts\python.exe -m spacy download en_core_web_lg
```

The inspected evaluator also requests NLTK resources. Its first use can require additional downloads. Some optional language-tool functionality has further runtime requirements; only enable it with its dependencies installed.

Start with a single available document:

```powershell
..\.venv\Scripts\python.exe run_lecturebank_axis_eval.py `
  --extraction-jsonl ..\lecturebank\pdf_extraction_results.jsonl `
  --jsonl-authoritative `
  --id 141 `
  --allow-any-count `
  --output-dir .\lecturebank_demo `
  --stage all `
  --max-questions 10 `
  --max-explain-questions 10 `
  --max-rounds 5 `
  --axis-interrogation-mode hybrid `
  --hybrid-add-from-source `
  --hybrid-add-source-limit 5 `
  --diagnose-hybrid-add `
  --evaluate-saquet `
  --evaluate-concept-change `
  --export-streamlit-json `
  --temporary-vector-stores
```

Use an ID present in your extraction file. Remove `--evaluate-saquet` if SAQUET is not installed. Remove `--export-streamlit-json` if you need metrics only.

For the first 100 eligible JSONL IDs, replace `--id 141` and `--allow-any-count` with `--first-jsonl 100 --expected-count 100`, and use a new output directory. Start small before committing to a large provider-backed run.

The evaluator writes a run manifest and a `metrics/` directory containing reports and CSVs, including replay comparisons where generated. With `--export-streamlit-json`, it additionally writes `streamlit/quiz_explanation_experiment_<id>.json`.

Check completion and unavailable-result counts before interpreting optional evaluator scores. In particular, the inspected archive predates fixes described for later SAQUET runs; do not assume that installing dependencies resolves those code-level issues.

### Reference sweep

`--build-effect-reference-set` requests evaluation of a predefined finite intervention set. This can be substantially more expensive than the ordinary AXIS budget. It is optional and is not needed to demonstrate the app.

Do not rely on this inspected version to resume an interrupted sweep from disk. Later snapshot/recovery changes need separate verification. General `--resume` behaviour should not be interpreted as a guarantee of recovering in-memory workflow checkpoints.

## 10. Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named streamlit` | Install Streamlit using the same `.venv` interpreter used to launch the app. |
| No MCQs or explanations after the presentation command | Use `--stage all` or run `--stage generate` after indexing. |
| API authentication/model error | Check the local key, account access, and configured model identifiers. |
| Fewer than ten questions | Ten is a cap/target; inspect evidence availability and acceptance results. |
| No usable extracted text | Check that the PDF contains text; perform OCR separately if needed. |
| Unknown CLI argument | Check the appropriate runner's `--help`; the two batch CLIs differ. |
| Missing content in Streamlit | Inspect the export. The viewer cannot reconstruct unexported data. |
| Slow first startup | Embedding, spaCy, or NLTK resources may be downloading/loading. |
| Interrupted run | Preserve the output folder and inspect its manifest. Do not mix new executions into a reference set without verified recovery support. |

## 11. Publishing a reproducible thesis release

- Commit application code, prompts, contracts, and configuration together.
- Include a small authorized example and exact reproduction commands.
- Record the code commit, environment versions, model configuration, and run IDs corresponding to reported results.
- Keep generated bulk outputs outside normal source control; publish selected anonymized metrics or a separate research-data archive when appropriate.
- Attribute AXIS and SAQUET and preserve upstream license notices. Select a project license only after checking ownership and third-party terms.

Original AXIS paper: Gyevnár, Lucas, Albrecht, and Cohen, *Integrating Counterfactual Simulations with Language Models for Explaining Multi-Agent Behaviour*, arXiv:2505.17801, https://arxiv.org/abs/2505.17801.
