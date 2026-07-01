"""Seed a couple of demo evaluation runs so a fresh deployment isn't empty.

Runs only when the database has no runs yet. Inserts realistic metric/sample
data directly (no LLM calls), so the dashboard shows populated charts on first
load. Visitors can then trigger their own live Groq-backed runs from the
"Evaluation Runs" page.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.repositories import ExperimentRepository
from app.db.session import initialize_database
from app.models.schemas import (
    EvaluationMetricResult,
    EvaluationRunCreate,
    EvaluationSampleRecord,
)

DEMO_RUNS = [
    {
        "run": EvaluationRunCreate(
            dataset_name="halueval_rag_test",
            model_name="llama-3.3-70b-versatile",
            metadata={"provider": "groq", "metrics": ["ragas", "deepeval"], "seed": True},
        ),
        "metrics": [
            ("faithfulness", 0.91), ("answer_relevancy", 0.88), ("context_recall", 0.79),
            ("context_precision", 0.83), ("answer_correctness", 0.86),
            ("hallucination", 0.90), ("toxicity", 1.0), ("bias", 0.95), ("g_eval", 0.88),
        ],
    },
    {
        "run": EvaluationRunCreate(
            dataset_name="halueval_rag_test",
            model_name="llama-3.1-8b-instant",
            metadata={"provider": "groq", "metrics": ["ragas", "deepeval"], "seed": True},
        ),
        "metrics": [
            ("faithfulness", 0.78), ("answer_relevancy", 0.81), ("context_recall", 0.72),
            ("context_precision", 0.75), ("answer_correctness", 0.74),
            ("hallucination", 0.80), ("toxicity", 1.0), ("bias", 0.9), ("g_eval", 0.80),
        ],
    },
]

DEMO_SAMPLES = [
    EvaluationSampleRecord(
        question="What is retrieval-augmented generation?",
        answer="RAG combines a retriever that fetches relevant documents with a generator that conditions its answer on them.",
        contexts=["Retrieval-augmented generation augments an LLM with an external document retriever."],
        ground_truth="RAG augments an LLM with retrieved documents to ground its answers.",
        answer_source="generated",
    ),
    EvaluationSampleRecord(
        question="Why evaluate RAG systems with faithfulness?",
        answer="Faithfulness measures whether the generated answer is supported by the retrieved context, catching hallucinations.",
        contexts=["Faithfulness checks that claims in the answer are grounded in the provided context."],
        ground_truth="It checks that the answer is grounded in the retrieved context.",
        answer_source="generated",
    ),
]


def main() -> None:
    initialize_database()
    repo = ExperimentRepository()
    if repo.list_runs():
        print("[seed] database already has runs; skipping demo seed.")
        return
    for entry in DEMO_RUNS:
        record = repo.create_run(entry["run"])
        repo.save_metrics(
            record.id,
            [EvaluationMetricResult(name=n, score=s, details={"seed": True}) for n, s in entry["metrics"]],
        )
        repo.save_samples(record.id, DEMO_SAMPLES)
    print(f"[seed] inserted {len(DEMO_RUNS)} demo runs.")


if __name__ == "__main__":
    main()
