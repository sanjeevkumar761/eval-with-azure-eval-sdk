"""
Online Evaluation Middleware
=============================

Real-time evaluation of financial metrics extraction in production.

Features:
- Per-request evaluation with low latency
- Confidence scoring and SME routing
- Azure Application Insights telemetry
- Drift detection with rolling windows
- Alert triggers on accuracy degradation
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import logging

from config import get_config, REQUIRED_METRICS
from models import (
    FinancialMetricsSet,
    DocumentEvaluationResult,
    ExtractionRequest,
    ExtractionResponse,
    SMEReviewRequest,
    EvaluationScore,
    MetricEvaluationResult,
)
from custom_evaluators import (
    NumericalAccuracyEvaluator,
    MetricCompletenessEvaluator,
    YoYConsistencyEvaluator,
    AggregateEvaluator,
)
from extraction_agent import FinancialMetricsExtractionAgent

logger = logging.getLogger(__name__)


# =============================================================================
# Telemetry / Application Insights Integration
# =============================================================================

class TelemetryClient:
    """
    Azure Application Insights telemetry client for online evaluation.
    
    Sends custom metrics and events for monitoring and alerting.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and connection_string
        self._client = None
        self._meter = None
        
        if self.enabled:
            self._initialize(connection_string)
    
    def _initialize(self, connection_string: str) -> None:
        """Initialize OpenTelemetry with Azure Monitor."""
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            from opentelemetry import metrics
            
            configure_azure_monitor(
                connection_string=connection_string,
            )
            
            self._meter = metrics.get_meter("financial_extraction_evaluation")
            
            # Create metric instruments
            self._accuracy_gauge = self._meter.create_gauge(
                name="extraction_accuracy_score",
                description="Accuracy score for financial metrics extraction",
                unit="score",
            )
            self._completeness_gauge = self._meter.create_gauge(
                name="metric_completeness_rate",
                description="Percentage of required metrics extracted",
                unit="%",
            )
            self._deviation_histogram = self._meter.create_histogram(
                name="numerical_deviation_avg",
                description="Average numerical deviation from ground truth",
                unit="%",
            )
            self._latency_histogram = self._meter.create_histogram(
                name="extraction_latency_ms",
                description="Total latency for extraction and evaluation",
                unit="ms",
            )
            self._low_confidence_counter = self._meter.create_counter(
                name="low_confidence_extractions",
                description="Count of extractions flagged for SME review",
            )
            self._anomaly_counter = self._meter.create_counter(
                name="yoy_anomaly_count",
                description="Count of year-over-year anomalies detected",
            )
            
            logger.info("Application Insights telemetry initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize telemetry: {e}")
            self.enabled = False
    
    def record_evaluation(
        self,
        result: DocumentEvaluationResult,
        latency_ms: float,
        attributes: Optional[dict] = None,
    ) -> None:
        """Record evaluation metrics to Application Insights."""
        if not self.enabled:
            return
        
        attrs = attributes or {}
        
        try:
            # Record scores
            self._accuracy_gauge.set(
                result.numerical_accuracy_score.score,
                attributes={"document_id": result.document_id, **attrs},
            )
            
            # Completeness
            completeness_pct = result.completeness_score.details.get("completeness_pct", 0) * 100
            self._completeness_gauge.set(completeness_pct, attributes=attrs)
            
            # Latency
            self._latency_histogram.record(latency_ms, attributes=attrs)
            
            # Low confidence counter
            if result.flagged_for_review:
                self._low_confidence_counter.add(1, attributes=attrs)
            
            # Anomaly counter
            if result.consistency_score:
                anomaly_count = result.consistency_score.details.get("anomaly_count", 0)
                if anomaly_count > 0:
                    self._anomaly_counter.add(anomaly_count, attributes=attrs)
                    
        except Exception as e:
            logger.error(f"Failed to record telemetry: {e}")
    
    def record_event(
        self,
        name: str,
        properties: Optional[dict] = None,
    ) -> None:
        """Record a custom event."""
        if not self.enabled:
            return
        
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer("financial_extraction")
            
            with tracer.start_as_current_span(name) as span:
                if properties:
                    for key, value in properties.items():
                        span.set_attribute(key, str(value))
                        
        except Exception as e:
            logger.error(f"Failed to record event: {e}")


# =============================================================================
# SME Review Queue
# =============================================================================

class SMEReviewQueue:
    """
    Azure Service Bus queue for routing low-confidence extractions
    to SME review.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        queue_name: str = "sme-review-queue",
        enabled: bool = True,
    ):
        self.enabled = enabled and connection_string
        self._sender = None
        self.queue_name = queue_name
        
        if self.enabled:
            self._initialize(connection_string)
    
    def _initialize(self, connection_string: str) -> None:
        """Initialize Service Bus sender."""
        try:
            from azure.servicebus.aio import ServiceBusClient
            
            self._client = ServiceBusClient.from_connection_string(connection_string)
            self._sender = self._client.get_queue_sender(self.queue_name)
            
            logger.info(f"SME review queue initialized: {self.queue_name}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize SME queue: {e}")
            self.enabled = False
    
    async def send_for_review(
        self,
        request: SMEReviewRequest,
    ) -> bool:
        """Send extraction for SME review."""
        if not self.enabled:
            logger.info(f"SME queue disabled - would send {request.document_id} for review")
            return False
        
        try:
            from azure.servicebus import ServiceBusMessage
            
            message = ServiceBusMessage(
                body=request.model_dump_json(),
                content_type="application/json",
                subject=f"review_{request.priority}",
                application_properties={
                    "document_id": request.document_id,
                    "priority": request.priority,
                },
            )
            
            await self._sender.send_messages(message)
            logger.info(f"Sent {request.document_id} to SME review queue")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send to SME queue: {e}")
            return False
    
    async def close(self) -> None:
        """Close queue connections."""
        if self._sender:
            await self._sender.close()
        if hasattr(self, "_client"):
            await self._client.close()


# =============================================================================
# Drift Detection
# =============================================================================

@dataclass
class DriftDetector:
    """
    Detects accuracy drift using rolling windows.
    
    Monitors for degradation in extraction quality over time.
    """
    
    window_size: int = 100  # Number of recent evaluations to consider
    alert_threshold: float = 0.15  # 15% drop triggers alert
    
    # Rolling window of scores
    accuracy_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    baseline_accuracy: Optional[float] = None
    
    def record(self, accuracy_score: float) -> Optional[dict]:
        """
        Record a new accuracy score and check for drift.
        
        Returns:
            Alert dict if drift detected, None otherwise
        """
        self.accuracy_scores.append(accuracy_score)
        
        # Need enough data for comparison
        if len(self.accuracy_scores) < self.window_size // 2:
            return None
        
        # Calculate current average
        current_avg = sum(self.accuracy_scores) / len(self.accuracy_scores)
        
        # Set baseline on first full window
        if self.baseline_accuracy is None and len(self.accuracy_scores) >= self.window_size:
            self.baseline_accuracy = current_avg
            return None
        
        # Check for drift
        if self.baseline_accuracy is not None:
            drift = (self.baseline_accuracy - current_avg) / self.baseline_accuracy
            
            if drift > self.alert_threshold:
                return {
                    "type": "accuracy_drift",
                    "baseline": self.baseline_accuracy,
                    "current": current_avg,
                    "drift_pct": drift * 100,
                    "severity": "high" if drift > 0.25 else "medium",
                    "timestamp": datetime.utcnow().isoformat(),
                }
        
        return None
    
    def reset_baseline(self) -> None:
        """Reset baseline to current average."""
        if self.accuracy_scores:
            self.baseline_accuracy = sum(self.accuracy_scores) / len(self.accuracy_scores)


# =============================================================================
# Online Evaluation Middleware
# =============================================================================

class OnlineEvaluationMiddleware:
    """
    Production middleware for real-time extraction evaluation.
    
    Wraps extraction calls with evaluation, telemetry, and SME routing.
    
    Usage:
        middleware = OnlineEvaluationMiddleware()
        await middleware.initialize()
        
        result = await middleware.extract_with_evaluation(
            document_text=doc_text,
            document_id="doc_001",
            prior_year_metrics=prior_metrics,
        )
        
        if result.flagged_for_review:
            # Handle low-confidence extraction
            pass
    """
    
    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        enable_telemetry: bool = True,
        enable_sme_queue: bool = True,
        enable_drift_detection: bool = True,
        alert_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.config = get_config()
        self.confidence_threshold = confidence_threshold or self.config.thresholds.confidence_threshold
        
        # Components
        self.extraction_agent: Optional[FinancialMetricsExtractionAgent] = None
        self.aggregate_evaluator = AggregateEvaluator(thresholds=self.config.thresholds)
        
        # Telemetry
        self.telemetry = TelemetryClient(
            connection_string=self.config.app_insights.connection_string,
            enabled=enable_telemetry and self.config.app_insights.enabled,
        )
        
        # SME Queue
        self.sme_queue = SMEReviewQueue(
            connection_string=self.config.sme_queue.connection_string,
            queue_name=self.config.sme_queue.queue_name,
            enabled=enable_sme_queue and self.config.sme_queue.enabled,
        )
        
        # Drift detection
        self.drift_detector = DriftDetector() if enable_drift_detection else None
        self.alert_callback = alert_callback
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize extraction agent and connections."""
        if self._initialized:
            return
        
        self.extraction_agent = FinancialMetricsExtractionAgent()
        await self.extraction_agent.initialize()
        
        self._initialized = True
        logger.info("Online evaluation middleware initialized")
    
    async def extract_with_evaluation(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        fiscal_year: int = 2025,
        prior_year_metrics: Optional[dict[str, float]] = None,
        reference_metrics: Optional[dict[str, float]] = None,
        required_metrics: Optional[list[str]] = None,
    ) -> ExtractionResponse:
        """
        Extract financial metrics with real-time evaluation.
        
        Args:
            document_text: Raw text from financial statement
            document_id: Optional document identifier
            fiscal_year: Fiscal year of the statement
            prior_year_metrics: Prior year values for consistency checking
            reference_metrics: Reference values for accuracy checking (if available)
            required_metrics: Override list of required metrics
        
        Returns:
            ExtractionResponse with metrics, evaluation, and routing decision
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        document_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        
        # Extract metrics
        extraction_start = time.time()
        extracted = await self.extraction_agent.extract_metrics(
            document_text=document_text,
            document_id=document_id,
            fiscal_year=fiscal_year,
            prior_year_metrics=prior_year_metrics,
        )
        extraction_latency = (time.time() - extraction_start) * 1000
        
        # Convert to dict for evaluation
        extracted_dict = {
            name: metric.normalized_value
            for name, metric in extracted.metrics.items()
            if metric.value is not None
        }
        
        # Run evaluation
        eval_start = time.time()
        
        # If reference metrics provided, use for accuracy check
        # Otherwise, use self-consistency checks only
        eval_result = self.aggregate_evaluator(
            extracted_metrics=extracted_dict,
            ground_truth_metrics=reference_metrics or extracted_dict,  # Self-eval if no reference
            prior_year_metrics=prior_year_metrics,
        )
        
        evaluation_latency = (time.time() - eval_start) * 1000
        total_latency = (time.time() - start_time) * 1000
        
        # Build evaluation result
        scores = eval_result["scores"]
        
        doc_eval_result = self._build_evaluation_result(
            document_id=document_id,
            eval_result=eval_result,
            scores=scores,
            extracted_dict=extracted_dict,
            reference_metrics=reference_metrics,
            extraction_latency=extraction_latency,
            evaluation_latency=evaluation_latency,
        )
        
        # Determine review status
        flagged = eval_result["confidence"] < self.confidence_threshold
        review_reasons = []
        
        if flagged:
            review_reasons = self._get_review_reasons(doc_eval_result)
        
        # Record telemetry
        self.telemetry.record_evaluation(
            result=doc_eval_result,
            latency_ms=total_latency,
            attributes={
                "has_reference": reference_metrics is not None,
                "has_prior_year": prior_year_metrics is not None,
            },
        )
        
        # Check for drift
        if self.drift_detector:
            drift_alert = self.drift_detector.record(scores.get("numerical_accuracy", 0))
            if drift_alert and self.alert_callback:
                self.alert_callback(drift_alert)
        
        # Send to SME queue if flagged
        if flagged and self.sme_queue.enabled:
            priority = self._determine_priority(doc_eval_result)
            
            sme_request = SMEReviewRequest(
                request_id=f"review_{uuid.uuid4().hex[:8]}",
                document_id=document_id,
                extracted_metrics=extracted,
                evaluation_result=doc_eval_result,
                review_reasons=review_reasons,
                priority=priority,
            )
            
            await self.sme_queue.send_for_review(sme_request)
        
        return ExtractionResponse(
            document_id=document_id,
            extracted_metrics=extracted,
            evaluation=doc_eval_result,
            flagged_for_review=flagged,
            review_reasons=review_reasons,
            total_latency_ms=total_latency,
        )
    
    def _build_evaluation_result(
        self,
        document_id: str,
        eval_result: dict,
        scores: dict,
        extracted_dict: dict,
        reference_metrics: Optional[dict],
        extraction_latency: float,
        evaluation_latency: float,
    ) -> DocumentEvaluationResult:
        """Build structured evaluation result."""
        
        # Per-metric results (if reference available)
        metric_results = []
        if reference_metrics:
            accuracy_results = eval_result["evaluator_results"]["numerical_accuracy"]["per_metric_results"]
            for mr in accuracy_results:
                metric_results.append(MetricEvaluationResult(
                    metric_name=mr["metric_name"],
                    extracted_value=mr["extracted_value"],
                    ground_truth_value=mr["ground_truth_value"],
                    deviation=mr["deviation"],
                    accuracy_score=mr["score"],
                    passed=mr["score"] >= self.config.thresholds.numerical_accuracy_threshold,
                ))
        
        return DocumentEvaluationResult(
            document_id=document_id,
            numerical_accuracy_score=EvaluationScore(
                evaluator_name="numerical_accuracy",
                score=scores.get("numerical_accuracy", 0),
                threshold=self.config.thresholds.numerical_accuracy_threshold,
                passed=scores.get("numerical_accuracy", 0) >= self.config.thresholds.numerical_accuracy_threshold,
                reason=eval_result["evaluator_results"]["numerical_accuracy"]["numerical_accuracy_reason"],
            ),
            completeness_score=EvaluationScore(
                evaluator_name="completeness",
                score=scores.get("completeness", 0),
                threshold=self.config.thresholds.completeness_threshold,
                passed=scores.get("completeness", 0) >= self.config.thresholds.completeness_threshold,
                reason=eval_result["evaluator_results"]["completeness"]["completeness_reason"],
                details={
                    "missing": eval_result["evaluator_results"]["completeness"]["missing_metrics"],
                    "completeness_pct": eval_result["evaluator_results"]["completeness"]["completeness_pct"],
                },
            ),
            math_consistency_score=EvaluationScore(
                evaluator_name="math_consistency",
                score=scores.get("math_consistency", 0),
                threshold=3.5,
                passed=scores.get("math_consistency", 0) >= 3.5,
                reason=eval_result["evaluator_results"]["math_consistency"]["math_consistency_reason"],
                details={
                    "checks_passed": eval_result["evaluator_results"]["math_consistency"]["checks_passed"],
                    "checks_total": eval_result["evaluator_results"]["math_consistency"]["checks_total"],
                    "failed_checks": eval_result["evaluator_results"]["math_consistency"]["checks_failed"],
                },
            ) if "math_consistency" in scores else None,
            consistency_score=EvaluationScore(
                evaluator_name="consistency",
                score=scores.get("consistency", 5.0),
                threshold=self.config.thresholds.consistency_threshold,
                passed=scores.get("consistency", 5.0) >= self.config.thresholds.consistency_threshold,
                reason=eval_result["evaluator_results"].get("consistency", {}).get("consistency_reason", "N/A"),
                details={
                    "anomaly_count": eval_result["evaluator_results"].get("consistency", {}).get("anomaly_count", 0),
                },
            ) if "consistency" in scores else None,
            metric_results=metric_results,
            overall_confidence=eval_result["confidence"],
            flagged_for_review=eval_result["confidence"] < self.confidence_threshold,
            extraction_latency_ms=extraction_latency,
            evaluation_latency_ms=evaluation_latency,
        )
    
    def _get_review_reasons(
        self,
        result: DocumentEvaluationResult,
    ) -> list[str]:
        """Get list of reasons for SME review."""
        reasons = []
        
        if not result.numerical_accuracy_score.passed:
            reasons.append(f"Low accuracy: {result.numerical_accuracy_score.reason}")
        
        if not result.completeness_score.passed:
            missing = result.completeness_score.details.get("missing", [])
            reasons.append(f"Missing metrics: {', '.join(missing[:5])}")
        
        if result.math_consistency_score and not result.math_consistency_score.passed:
            failed = result.math_consistency_score.details.get("checks_passed", 0)
            total = result.math_consistency_score.details.get("checks_total", 0)
            reasons.append(f"Math inconsistencies: {total - failed}/{total} checks failed")
        
        if result.consistency_score and not result.consistency_score.passed:
            anomalies = result.consistency_score.details.get("anomaly_count", 0)
            reasons.append(f"YoY anomalies detected: {anomalies}")
        
        if result.overall_confidence < 0.5:
            reasons.append(f"Very low confidence: {result.overall_confidence:.2f}")
        
        return reasons
    
    def _determine_priority(
        self,
        result: DocumentEvaluationResult,
    ) -> str:
        """Determine priority for SME review."""
        confidence = result.overall_confidence
        
        if confidence < 0.3:
            return "critical"
        elif confidence < 0.5:
            return "high"
        elif confidence < 0.7:
            return "normal"
        else:
            return "low"
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self.extraction_agent:
            await self.extraction_agent.cleanup()
        await self.sme_queue.close()
        logger.info("Online evaluation middleware shutdown complete")


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_middleware(
    enable_telemetry: bool = True,
    enable_sme_queue: bool = False,  # Disabled by default for local testing
) -> OnlineEvaluationMiddleware:
    """
    Create and initialize online evaluation middleware.
    
    Returns:
        Initialized OnlineEvaluationMiddleware
    """
    middleware = OnlineEvaluationMiddleware(
        enable_telemetry=enable_telemetry,
        enable_sme_queue=enable_sme_queue,
    )
    await middleware.initialize()
    return middleware


# =============================================================================
# Example Usage
# =============================================================================

async def demo():
    """Demonstration of online evaluation."""
    
    # Alert callback
    def on_alert(alert: dict):
        print(f"⚠️ ALERT: {alert['type']}")
        print(f"   Drift: {alert['drift_pct']:.1f}%")
        print(f"   Severity: {alert['severity']}")
    
    # Create middleware
    middleware = OnlineEvaluationMiddleware(
        confidence_threshold=0.7,
        enable_telemetry=False,  # Disable for demo
        enable_sme_queue=False,
        alert_callback=on_alert,
    )
    await middleware.initialize()
    
    # Sample document
    doc_text = """
    Company ABC Annual Report 2025
    
    Revenue: CHF 10,000,000
    Cost of Goods Sold: CHF 6,000,000
    Gross Profit: CHF 4,000,000
    Operating Expenses: CHF 2,000,000
    Operating Income: CHF 2,000,000
    Net Income: CHF 1,500,000
    
    Total Assets: CHF 20,000,000
    Total Liabilities: CHF 8,000,000
    Shareholders' Equity: CHF 12,000,000
    """
    
    # Prior year for consistency check
    prior_year = {
        "total_revenue": 9_500_000,
        "net_income": 1_400_000,
        "total_assets": 18_500_000,
    }
    
    # Run extraction with evaluation
    result = await middleware.extract_with_evaluation(
        document_text=doc_text,
        document_id="demo_001",
        prior_year_metrics=prior_year,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("EXTRACTION WITH EVALUATION")
    print("=" * 50)
    
    print("\nExtracted Metrics:")
    for name, metric in result.extracted_metrics.metrics.items():
        if metric.value:
            print(f"  {name}: {metric.value:,.0f}")
    
    print(f"\nEvaluation:")
    print(f"  Accuracy Score: {result.evaluation.numerical_accuracy_score.score:.2f}")
    print(f"  Completeness Score: {result.evaluation.completeness_score.score:.2f}")
    print(f"  Overall Confidence: {result.evaluation.overall_confidence:.2f}")
    print(f"  Flagged for Review: {result.flagged_for_review}")
    
    if result.review_reasons:
        print(f"\nReview Reasons:")
        for reason in result.review_reasons:
            print(f"  - {reason}")
    
    print(f"\nLatency: {result.total_latency_ms:.0f}ms")
    
    await middleware.shutdown()


if __name__ == "__main__":
    asyncio.run(demo())
