# EvalForge
**AI Evaluation & Optimization Platform for LLM and RAG Applications**

EvalForge is a modular platform for evaluating, benchmarking, and reporting on LLM and RAG systems. It supports dataset management, model comparison, experiment tracking, and exportable reports for AI engineers building production-oriented workflows.

## Core Features

- Dataset upload, preview, and validation
- Ollama-based LLM configuration
- RAGAS evaluation for retrieval quality
- DeepEval evaluation for LLM quality
- Benchmark comparison across prompts, models, and chunk settings
- SQLite-backed experiment tracking
- CSV, JSON, and PDF report generation
- Streamlit dashboard for results and analysis

## Tech Stack

Python, FastAPI, Streamlit, LangChain, FAISS, Ollama, RAGAS, DeepEval, Pandas, Plotly, SQLite, Pydantic, ReportLab

## Project Structure

- `app/api/` - FastAPI routes
- `app/services/` - business logic
- `app/evaluators/` - RAGAS and DeepEval integration
- `app/llm_providers/` - model providers
- `app/db/` - SQLite persistence
- `app/reports/` - export logic
- `app/dashboard/` - Streamlit UI
- `app/models/` - typed schemas

## Local Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -e .
uvicorn app.api.main:app --reload
streamlit run app/dashboard/main.py
```

## Future Scope

- OpenAI and Gemini provider support
- LangSmith and Langfuse integration
- Promptfoo-based testing
- MCP integration
- multi-agent evaluation workflows
- richer experiment lineage and reporting

## License

No license has been added yet.