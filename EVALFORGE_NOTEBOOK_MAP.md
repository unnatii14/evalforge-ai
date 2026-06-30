# EvalForge Notebook-to-Architecture Map

This repository currently contains a notebook-first RAG evaluation demo. The goal of this map is to turn the notebook flow into a production-ready EvalForge architecture without losing the useful ideas already implemented.

## What the notebook already does

The notebook in [RAG_Langchain_Ragas.ipynb](RAG_Langchain_Ragas.ipynb) already covers the full evaluation loop:

- model configuration and Hugging Face LLM loading
- data ingestion from a web source
- text splitting and chunking
- embedding generation
- FAISS vector store creation
- retrieval QA generation
- similarity checks
- RAGAS synthetic testset generation
- RAGAS metric evaluation
- CSV export of testset and results

The generated artifacts are already stored in [evaluation/testset.csv](evaluation/testset.csv) and [evaluation/results.csv](evaluation/results.csv).

## Recommended module mapping

| Notebook section | Current behavior | EvalForge module |
| --- | --- | --- |
| Setup / imports | Installs and imports notebook dependencies | `config/`, `core/`, project dependency files |
| `Model` class + `get_model()` | Hard-coded model config for local Hugging Face models | `app/llm_providers/` and `app/models/` |
| Data ingestion | Loads web content with `WebBaseLoader` | `app/services/dataset_service.py` |
| Splitting | Uses `RecursiveCharacterTextSplitter` | `app/retrieval/chunking.py` |
| Embedding | Builds `HuggingFaceBgeEmbeddings` | `app/retrieval/embeddings.py` |
| Vector store | Builds FAISS index | `app/retrieval/vector_store.py` |
| QA chain retrieval | Runs `RetrievalQA` over retrieved chunks | `app/services/evaluation_service.py` and `app/services/benchmark_service.py` |
| Similarity checks | Uses FAISS similarity methods | `app/utils/metrics.py` or `app/services/benchmark_service.py` |
| RAGAS testset generation | Generates synthetic questions and ground truth | `app/evaluators/ragas_evaluator.py` |
| RAGAS metric evaluation | Computes faithfulness, relevancy, recall, precision, correctness | `app/evaluators/ragas_evaluator.py` |
| Result export | Writes CSV outputs | `app/reports/csv_report.py`, `app/reports/json_report.py`, `app/reports/pdf_report.py` |

## What should become persistent state

The notebook currently treats most things as temporary runtime variables. In EvalForge, these should become tracked entities:

- dataset version
- model/provider configuration
- chunk size and overlap
- top-k retrieval setting
- embedding model
- prompt template version
- run timestamp and latency
- metric outputs per question
- aggregated scores per experiment

## Production gaps to fix next

1. Replace notebook-only parameters with Pydantic settings and typed request models.
2. Separate retrieval, evaluation, and reporting into independent services.
3. Persist experiments and runs in SQLite instead of only exporting CSV files.
4. Wrap Ollama behind a provider interface so OpenAI and Gemini can be added later.
5. Add a Streamlit dashboard on top of the stored experiments and reports.

## Best next milestone

The safest first refactor is to extract these notebook concepts into a backend skeleton:

- `app/core/config.py`
- `app/models/schemas.py`
- `app/db/session.py`
- `app/db/repositories.py`
- `app/services/dataset_service.py`
- `app/services/evaluation_service.py`
- `app/retrieval/chunking.py`
- `app/retrieval/vector_store.py`

That gives you the foundation for experiment tracking and lets the dashboard read from real persisted runs instead of notebook outputs.