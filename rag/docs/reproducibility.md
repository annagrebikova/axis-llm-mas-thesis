# Reproducibility

## Environment

- Development/tests exercised against **Python 3.12** (conda) and **LangGraph 1.2.9**.
- Requirement constraint remains `langgraph>=1.0.0,<2.0.0` (see `rag/requirements_rag.txt`).
- Do not assume Git is installed for exports; explanation reports include Python /
  LangGraph versions when importable.

## Dependencies

From `EMaES/rag`:

```bash
pip install -r requirements_rag.txt
```

## Test categories

| Category | Command / location |
|----------|--------------------|
| Unit / regression | `pytest mas_langgraph/tests` |
| Framework backbone | `test_framework_backbone.py` |
| Slicing / queries / loop / config | `test_prompt12_*.py` … `test_prompt15_config.py` |
| Final integration | `test_prompt16_end_to_end.py` |
| Local deterministic integration | real LangGraph + stub AgentRunner/RAG/LLM |
| External integration | real PDF / vector store / OpenAI — **not required** for Prompt 16 |

LLM/provider outputs may vary. The no-op control exists partly for this reason.

## Full suite

```bash
cd EMaES/rag
python -m pytest mas_langgraph/tests -q
```

## Stub factual smoke

```bash
python -m mas_langgraph --orchestrate --stub --orchestrator langgraph
python -m mas_langgraph --orchestrate --stub --orchestrator classic
```

Expected: exit code 0, `run_id` present, no crash.

## AXIS scripted smoke (programmatic)

```python
from mas_langgraph import CooperativeBackbone

bb = CooperativeBackbone(orchestrator_mode="langgraph")
summary = bb.run_full_workflow()
# Stub default run may yield zero explainable candidates.
# Use the seeded fixture in test_prompt16_end_to_end for a full AXIS path.
```

Seeded deterministic AXIS E2E (no API keys, no network, no Chroma DB) lives in
`mas_langgraph/tests/test_prompt16_end_to_end.py`.

## Real RAG prerequisites (optional)

```bash
python rag_pipeline.py --pdf-id 141
python -m mas_langgraph --orchestrate --pdf-id 141 --orchestrator langgraph
```

Requires indexed `vector_store/` and provider credentials. Not part of the
deterministic Prompt 16 acceptance gate.
