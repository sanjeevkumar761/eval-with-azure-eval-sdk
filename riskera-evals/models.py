"""
Data Models for Financial Metrics Evaluation Framework
=======================================================

Pydantic models for financial metrics, evaluation results,
and data exchange between components.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


class MetricCategory(str, Enum):
    """Categories of financial metrics."""
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    RATIOS = "ratios"


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"
    OTHER = "OTHER"


# =============================================================================
# Financial Metrics Models
# =============================================================================

class FinancialMetric(BaseModel):
    """A single extracted financial metric."""
    
    name: str = Field(..., description="Metric name (e.g., 'total_revenue')")
    value: Optional[float] = Field(None, description="Extracted numerical value")
    currency: Optional[Currency] = Field(None, description="Currency if applicable")
    unit: Optional[str] = Field(None, description="Unit (e.g., 'thousands', 'millions')")
    raw_text: Optional[str] = Field(None, description="Original text from document")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")
    source_location: Optional[str] = Field(None, description="Location in source document")
    
    @property
    def normalized_value(self) -> Optional[float]:
        """Get value normalized to base units."""
        if self.value is None:
            return None
        
        multipliers = {
            "thousands": 1_000,
            "millions": 1_000_000,
            "billions": 1_000_000_000,
        }
        multiplier = multipliers.get(self.unit, 1) if self.unit else 1
        return self.value * multiplier


class FinancialMetricsSet(BaseModel):
    """Complete set of extracted financial metrics."""
    
    document_id: str = Field(..., description="Unique document identifier")
    client_id: Optional[str] = Field(None, description="Client identifier")
    fiscal_year: int = Field(..., description="Fiscal year of the statement")
    fiscal_period: Optional[str] = Field(None, description="Period (e.g., 'Q1', 'FY')")
    
    # Extracted metrics
    metrics: dict[str, FinancialMetric] = Field(
        default_factory=dict,
        description="Dictionary of metric name to FinancialMetric"
    )
    
    # Metadata
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_document_type: Optional[str] = Field(None, description="Type of source document")
    
    def get_metric_value(self, name: str, normalized: bool = True) -> Optional[float]:
        """Get a metric value by name."""
        metric = self.metrics.get(name)
        if metric is None:
            return None
        return metric.normalized_value if normalized else metric.value
    
    def get_missing_metrics(self, required: list[str]) -> list[str]:
        """Get list of missing required metrics."""
        return [m for m in required if m not in self.metrics or self.metrics[m].value is None]


class GroundTruthData(BaseModel):
    """Ground truth data for evaluation (SME-validated)."""
    
    document_id: str
    metrics: dict[str, float] = Field(
        ..., description="Dictionary of metric name to ground truth value"
    )
    validated_by: Optional[str] = Field(None, description="SME who validated")
    validation_date: Optional[datetime] = None
    notes: Optional[str] = None


class PriorYearMetrics(BaseModel):
    """Prior year metrics for YoY consistency checking."""
    
    document_id: str
    fiscal_year: int
    metrics: dict[str, float] = Field(
        ..., description="Dictionary of metric name to prior year value"
    )


# =============================================================================
# Evaluation Result Models
# =============================================================================

class EvaluationScore(BaseModel):
    """Score from a single evaluator."""
    
    evaluator_name: str
    score: float = Field(..., ge=0.0, le=5.0, description="Score from 0-5")
    threshold: float = Field(3.0, description="Pass/fail threshold")
    passed: bool = Field(..., description="Whether score meets threshold")
    reason: Optional[str] = Field(None, description="Explanation of score")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")


class MetricEvaluationResult(BaseModel):
    """Evaluation result for a single metric."""
    
    metric_name: str
    extracted_value: Optional[float]
    ground_truth_value: Optional[float]
    deviation: Optional[float] = Field(None, description="Percentage deviation")
    accuracy_score: float = Field(..., ge=0.0, le=5.0)
    passed: bool


class DocumentEvaluationResult(BaseModel):
    """Complete evaluation result for a document."""
    
    document_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Overall scores
    numerical_accuracy_score: EvaluationScore
    completeness_score: EvaluationScore
    math_consistency_score: Optional[EvaluationScore] = None
    consistency_score: Optional[EvaluationScore] = None
    groundedness_score: Optional[EvaluationScore] = None
    
    # Per-metric results
    metric_results: list[MetricEvaluationResult] = Field(default_factory=list)
    
    # Aggregates
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    flagged_for_review: bool = False
    
    # Metadata
    extraction_latency_ms: Optional[float] = None
    evaluation_latency_ms: Optional[float] = None
    
    @property
    def all_scores(self) -> list[EvaluationScore]:
        """Get all non-None evaluation scores."""
        scores = [self.numerical_accuracy_score, self.completeness_score]
        if self.math_consistency_score:
            scores.append(self.math_consistency_score)
        if self.consistency_score:
            scores.append(self.consistency_score)
        if self.groundedness_score:
            scores.append(self.groundedness_score)
        return scores
    
    @property
    def average_score(self) -> float:
        """Calculate average across all evaluators."""
        scores = self.all_scores
        return sum(s.score for s in scores) / len(scores) if scores else 0.0


class BatchEvaluationResult(BaseModel):
    """Result from evaluating a batch of documents (offline evaluation)."""
    
    evaluation_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dataset_name: Optional[str] = None
    dataset_size: int
    
    # Individual results
    document_results: list[DocumentEvaluationResult] = Field(default_factory=list)
    
    # Aggregate statistics
    overall_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Average scores per evaluator"
    )
    per_metric_accuracy: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Per-metric accuracy statistics (mean, std)"
    )
    
    # Failure analysis
    failed_extractions: list[dict[str, Any]] = Field(default_factory=list)
    low_confidence_count: int = 0
    
    # Timing
    total_duration_seconds: float = 0.0


# =============================================================================
# Golden Dataset Models
# =============================================================================

class GoldenDatasetEntry(BaseModel):
    """Single entry in the golden dataset."""
    
    document_id: str
    raw_text: str = Field(..., description="Raw text extracted from document")
    ground_truth: GroundTruthData
    prior_year: Optional[PriorYearMetrics] = None
    
    # Metadata
    client_industry: Optional[str] = None
    document_format: Optional[str] = Field(None, description="Format type for analysis")
    difficulty_level: Optional[str] = Field(None, description="easy/medium/hard")


class GoldenDataset(BaseModel):
    """Complete golden dataset for offline evaluation."""
    
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"
    
    entries: list[GoldenDatasetEntry] = Field(default_factory=list)
    
    @property
    def size(self) -> int:
        return len(self.entries)


# =============================================================================
# Online Evaluation Models
# =============================================================================

class ExtractionRequest(BaseModel):
    """Request for online extraction with evaluation."""
    
    document_id: str
    document_text: str
    prior_year_metrics: Optional[dict[str, float]] = None
    required_metrics: Optional[list[str]] = None
    enable_evaluation: bool = True


class ExtractionResponse(BaseModel):
    """Response from online extraction with evaluation."""
    
    document_id: str
    extracted_metrics: FinancialMetricsSet
    evaluation: Optional[DocumentEvaluationResult] = None
    
    # Routing decision
    flagged_for_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    
    # Performance
    total_latency_ms: float


class SMEReviewRequest(BaseModel):
    """Request sent to SME review queue."""
    
    request_id: str
    document_id: str
    extracted_metrics: FinancialMetricsSet
    evaluation_result: DocumentEvaluationResult
    review_reasons: list[str]
    priority: str = "normal"  # low, normal, high, critical
    created_at: datetime = Field(default_factory=datetime.utcnow)
