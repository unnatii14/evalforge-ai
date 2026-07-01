from pathlib import Path

import pandas as pd

from app.services.experiment_service import ExperimentService

DATASETS_DIR = Path("data") / "datasets"

class DashboardDataLoader:
    def __init__(self):
        self.experiment_service = ExperimentService()
        self._ensure_dirs()

    def _ensure_dirs(self):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        # Check if we should copy notebook CSVs as default datasets
        default_dataset = DATASETS_DIR / "halueval_rag_test.csv"
        if not default_dataset.exists() and Path("evaluation/testset.csv").exists():
            try:
                df = pd.read_csv("evaluation/testset.csv")
                df.to_csv(default_dataset, index=False)
            except Exception:
                pass

    def get_datasets(self) -> list[str]:
        """Scans the data/datasets directory for CSV datasets."""
        if not DATASETS_DIR.exists():
            return ["halueval_rag_test"]
        
        csv_files = list(DATASETS_DIR.glob("*.csv"))
        if not csv_files:
            return ["halueval_rag_test"]
        return [f.stem for f in csv_files]

    def get_runs(self) -> list[dict]:
        summaries = self.experiment_service.list_run_summaries()
        runs_list: list[dict] = []
        for summary in summaries:
            run = summary.run
            metric_map = {metric.name: metric.score for metric in summary.metrics}
            metadata = run.metadata or {}
            runs_list.append(
                {
                    "id": run.id,
                    "dataset_name": run.dataset_name,
                    "model_name": run.model_name,
                    "provider": metadata.get("provider", "ollama"),
                    "chunk_size": run.chunk_size,
                    "chunk_overlap": run.chunk_overlap,
                    "top_k": run.top_k,
                    "faithfulness": metric_map.get("faithfulness", 0.0),
                    "answer_relevancy": metric_map.get("answer_relevancy", 0.0),
                    "context_recall": metric_map.get("context_recall", 0.0),
                    "context_precision": metric_map.get("context_precision", 0.0),
                    "answer_correctness": metric_map.get("answer_correctness", 0.0),
                    "hallucination": metric_map.get("hallucination", 0.0),
                    "toxicity": metric_map.get("toxicity", 0.0),
                    "bias": metric_map.get("bias", 0.0),
                    "latency_seconds": metadata.get("total_latency_seconds", 0.0),
                    "cost_usd": metadata.get("cost_usd", 0.0),
                    "status": metadata.get("status", "completed"),
                    "created_at": run.created_at,
                    "overall_score": summary.overall_score,
                }
            )
        return runs_list

    def get_run_details(self, run_id: int) -> pd.DataFrame:
        details = self.experiment_service.get_run_details(run_id)
        if details is None or not details.samples:
            return pd.DataFrame()

        metric_map = {metric.name: metric.score for metric in details.metrics}
        rows = []
        for sample in details.samples:
            rows.append(
                {
                    "question": sample.question,
                    "answer": sample.answer,
                    "ground_truth": sample.ground_truth,
                    "contexts": sample.contexts,
                    "answer_source": sample.answer_source,
                    "faithfulness": metric_map.get("faithfulness", 0.0),
                    "answer_relevancy": metric_map.get("answer_relevancy", 0.0),
                    "context_recall": metric_map.get("context_recall", 0.0),
                    "context_precision": metric_map.get("context_precision", 0.0),
                    "answer_correctness": metric_map.get("answer_correctness", 0.0),
                }
            )

        return pd.DataFrame(rows)

