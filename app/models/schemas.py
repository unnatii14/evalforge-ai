from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ProviderName(str, Enum):
    ollama = "ollama"
    openai = "openai"
    gemini = "gemini"


class DatasetRow(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    context: Optional[str] = None
    expected_answer: Optional[str] = None


class DatasetPreview(BaseModel):
    row_count: int
    columns: list[str]
    sample_rows: list[DatasetRow] = Field(default_factory=list)


class DatasetUploadResult(BaseModel):
    dataset_name: str
    row_count: int
    columns: list[str]
    file_path: str


class LLMConfig(BaseModel):
    provider: ProviderName = ProviderName.ollama
    model_name: str = "llama3.2"
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 3


class MetricScore(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)


class EvaluationRunCreate(BaseModel):
    dataset_name: str
    model_name: str
    chunk_size: int = 512
    chunk_overlap: int = 16
    top_k: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunRecord(EvaluationRunCreate):
    id: int
    created_at: datetime


class LLMGenerationRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    config: LLMConfig = Field(default_factory=LLMConfig)


class LLMGenerationResult(BaseModel):
    provider: ProviderName
    model_name: str
    output: str
    latency_seconds: float = 0.0


class EvaluationMetricResult(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    dataset_name: str
    config: LLMConfig = Field(default_factory=LLMConfig)
    metrics: list[str] = Field(default_factory=list)
    samples: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationSample(BaseModel):
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str = ""


class EvaluationResponse(BaseModel):
    run: EvaluationRunRecord
    metrics: list[EvaluationMetricResult]
    latency_seconds: float
    cost_usd: float = 0.0


class BenchmarkScenario(BaseModel):
    name: str
    config: LLMConfig = Field(default_factory=LLMConfig)
    samples: list[EvaluationSample] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRequest(BaseModel):
    benchmark_name: str
    scenarios: list[BenchmarkScenario] = Field(default_factory=list)


class BenchmarkRow(BaseModel):
    scenario_name: str
    model_name: str
    provider: ProviderName
    overall_score: float = Field(ge=0.0, le=1.0)
    latency_seconds: float = 0.0
    metric_scores: list[EvaluationMetricResult] = Field(default_factory=list)


class BenchmarkResponse(BaseModel):
    benchmark_name: str
    rows: list[BenchmarkRow] = Field(default_factory=list)
    best_scenario_name: str = ""


class ReportFormat(str, Enum):
    csv = "csv"
    json = "json"
    pdf = "pdf"


class ReportRequest(BaseModel):
    report_name: str
    format: ReportFormat = ReportFormat.csv
    evaluation: EvaluationResponse | None = None
    benchmark: BenchmarkResponse | None = None


class ReportResult(BaseModel):
    report_name: str
    format: ReportFormat
    file_path: str
    row_count: int = 0
