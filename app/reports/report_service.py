from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from app.models.schemas import (
    BenchmarkResponse,
    EvaluationResponse,
    ReportFormat,
    ReportRequest,
    ReportResult,
)


class ReportService:
    def __init__(self, output_dir: str | Path = Path("data") / "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, request: ReportRequest) -> ReportResult:
        payload = self._build_payload(request.evaluation, request.benchmark)
        target_path = self.output_dir / f"{request.report_name}.{request.format.value}"

        if request.format == ReportFormat.csv:
            dataframe = pd.DataFrame(payload)
            dataframe.to_csv(target_path, index=False)
            return ReportResult(report_name=request.report_name, format=request.format, file_path=str(target_path), row_count=len(dataframe))

        if request.format == ReportFormat.json:
            target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            row_count = len(payload) if isinstance(payload, list) else 1
            return ReportResult(report_name=request.report_name, format=request.format, file_path=str(target_path), row_count=row_count)

        if request.format == ReportFormat.pdf:
            self._write_pdf(target_path, request.report_name, payload)
            row_count = len(payload) if isinstance(payload, list) else 1
            return ReportResult(report_name=request.report_name, format=request.format, file_path=str(target_path), row_count=row_count)

        raise ValueError(f"Unsupported report format: {request.format}")

    def _build_payload(
        self,
        evaluation: EvaluationResponse | None,
        benchmark: BenchmarkResponse | None,
    ) -> list[dict[str, object]]:
        if evaluation is not None:
            return [
                {
                    "type": "evaluation",
                    "run_id": evaluation.run.id,
                    "dataset_name": evaluation.run.dataset_name,
                    "model_name": evaluation.run.model_name,
                    "provider": evaluation.run.metadata.get("provider", ""),
                    "metric_name": metric.name,
                    "metric_score": metric.score,
                    "latency_seconds": evaluation.latency_seconds,
                    "cost_usd": evaluation.cost_usd,
                }
                for metric in evaluation.metrics
            ]

        if benchmark is not None:
            return [
                {
                    "type": "benchmark",
                    "benchmark_name": benchmark.benchmark_name,
                    "scenario_name": row.scenario_name,
                    "model_name": row.model_name,
                    "provider": row.provider.value,
                    "overall_score": row.overall_score,
                    "latency_seconds": row.latency_seconds,
                    "metric_scores": [metric.model_dump() for metric in row.metric_scores],
                    "best_scenario_name": benchmark.best_scenario_name,
                }
                for row in benchmark.rows
            ]

        return []

    def _write_pdf(self, target_path: Path, report_name: str, payload: list[dict[str, object]]) -> None:
        styles = getSampleStyleSheet()
        document = SimpleDocTemplate(str(target_path), pagesize=A4)
        elements = [Paragraph(report_name, styles["Title"]), Spacer(1, 12)]

        if not payload:
            elements.append(Paragraph("No data available.", styles["BodyText"]))
            document.build(elements)
            return

        headers = list(payload[0].keys())
        table_data = [headers] + [[str(row.get(header, "")) for header in headers] for row in payload]
        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        elements.append(table)
        document.build(elements)
