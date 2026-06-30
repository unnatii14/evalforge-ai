# EvalForge

**AI Evaluation & Optimization Platform for LLM and RAG Applications**

EvalForge is a production-oriented toolkit for evaluating, benchmarking, comparing, and reporting on LLM and RAG systems. It is designed for AI engineers who need a clean way to manage datasets, run experiments, compare configurations, and track performance over time.

Maintainer: Unnati

This project is intentionally **not** another chatbot. It is an engineering platform for testing model quality, retrieval quality, hallucination risk, latency, and experiment quality across multiple configurations.

## Why this project exists

Modern AI applications need more than generation. They need:

- repeatable dataset management
- structured experiment tracking
- RAG and LLM evaluation
- comparison across prompts, models, chunk sizes, and top-k values
- exportable reports for analysis and sharing
- a dashboard that makes results easy to inspect

EvalForge packages those needs into one modular platform.

## Key features

- Dataset upload, preview, validation, and persistence
- LLM provider abstraction with Ollama support first
- RAG evaluation with RAGAS metrics
- LLM evaluation with DeepEval metrics
- Benchmarking across prompts, models, chunk sizes, and top-k values
- SQLite-backed experiment tracking
- CSV, JSON, and PDF report generation
- Streamlit dashboard with comparison views and filters
- Modular code structure for future growth

## Current status

The project is structured as a modular monolith with two main surfaces:

- **FastAPI backend** for datasets, evaluations, benchmarks, and reports
- **Streamlit dashboard** for exploration, comparison, and experiment visibility

The current implementation is built to support local development and can be extended later with:

- LangSmith
- Langfuse
- Promptfoo
- MCP
- multi-agent evaluation workflows

## Tech stack

- Python
- FastAPI
- Streamlit
- LangChain
- FAISS
- Ollama
- RAGAS
- DeepEval
- Pandas
- Plotly
- SQLite
- Pydantic
- ReportLab

## Architecture

EvalForge is organized around clear service boundaries:

- `app/api/` - FastAPI routes
- `app/services/` - business logic and orchestration
- `app/evaluators/` - RAGAS and DeepEval integration
- `app/llm_providers/` - model provider abstraction
- `app/retrieval/` - retrieval and vector store utilities
- `app/db/` - SQLite session and repository layer
- `app/reports/` - CSV, JSON, and PDF export logic
- `app/dashboard/` - Streamlit UI shell
- `app/models/` - typed Pydantic contracts
- `app/core/` - config and application setup

### High-level flow

1. Upload or select a dataset.
2. Configure a model and evaluation scenario.
3. Run RAGAS or DeepEval-based evaluation.
4. Compare runs and benchmark scenarios.
5. Export results and inspect the dashboard.

## Folder structure

```text
evalforge/
├─ app/
│  ├─ api/
│  ├─ core/
│  ├─ dashboard/
│  ├─ db/
│  ├─ evaluators/
│  ├─ llm_providers/
│  ├─ models/
│  ├─ reports/
│  └─ services/
├─ data/
│  ├─ datasets/
│  ├─ experiments/
│  └─ reports/
├─ evaluation/
│  ├─ testset.csv
│  └─ results.csv
├─ RAG_Langchain_Ragas.ipynb
├─ pyproject.toml
└─ README.md
```

## Dashboard pages

The Streamlit app is organized into the following pages:

- Overview
- Benchmark Comparison
- Dataset Manager
- Evaluation Runs
- Reports & Export
- Settings

The dashboard includes global filters for dataset, model, metrics, and date range.

## Evaluation capabilities

### RAGAS

EvalForge supports the following RAGAS metrics:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Answer Correctness

### DeepEval

EvalForge also supports LLM evaluation metrics such as:

- Hallucination
- Toxicity
- Bias
- G-Eval

## Data format

Uploaded evaluation datasets should contain at least:

- `question`

Optional columns:

- `ground_truth`
- `context`
- `expected_answer`

## Local setup

### 1. Clone the repository

```powershell
git clone https://github.com/unnatii14/evalforge-ai.git
cd evalforge-ai
```

### 2. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\activate
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -e .
```

### 4. Run the FastAPI backend

```powershell
uvicorn app.api.main:app --reload
```

### 5. Run the Streamlit dashboard

```powershell
streamlit run app/dashboard/main.py
```

## Example API endpoints

- `GET /health`
- `POST /datasets/preview`
- `POST /datasets/persist`
- `POST /evaluations/run`
- `POST /benchmarks/run`
- `POST /reports/generate`

## Example project usage

1. Upload a CSV dataset.
2. Select a provider and model configuration.
3. Run an evaluation with selected metrics.
4. Compare benchmark scenarios.
5. Export the results as CSV, JSON, or PDF.

## Future scope

Planned extensions include:

- OpenAI and Gemini provider support
- LangSmith tracing
- Langfuse experiment tracking
- Promptfoo prompt testing
- MCP-based tool integration
- multi-agent evaluation workflows
- richer experiment lineage and artifact versioning

## License

No license has been added yet. Add one before public distribution if you want to make reuse explicit.