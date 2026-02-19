"""
Offline Evaluation Pipeline
============================

Batch evaluation of financial metrics extraction against
a golden dataset with SME-validated ground truth.

Features:
- Parallel batch processing
- Per-metric accuracy breakdown
- A/B testing for prompt comparison
- Statistical analysis
- Export to JSON/CSV
- CI/CD integration support
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
import statistics

from tqdm import tqdm

from config import get_config, get_model_config, REQUIRED_METRICS, METRIC_CATEGORIES
from models import (
    GoldenDataset,
    GoldenDatasetEntry,
    FinancialMetricsSet,
    DocumentEvaluationResult,
    BatchEvaluationResult,
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
from golden_dataset import load_golden_dataset


@dataclass
class OfflineEvaluationConfig:
    """Configuration for offline evaluation run."""
    
    # Parallelism
    max_concurrent: int = 5
    
    # Evaluation settings
    include_groundedness: bool = True
    include_consistency: bool = True
    
    # Output settings
    output_format: str = "json"  # json, csv, both
    save_per_document: bool = False
    
    # CI/CD settings
    fail_threshold: float = 3.5  # Minimum average score to pass
    
    # Retry settings
    max_retries: int = 2
    retry_delay: float = 1.0


class OfflineEvaluator:
    """
    Runs batch evaluation against a golden dataset.
    
    Usage:
        evaluator = OfflineEvaluator()
        results = await evaluator.run(
            dataset_path="data/golden_dataset.json",
            output_path="results/",
        )
    """
    
    def __init__(
        self,
        config: Optional[OfflineEvaluationConfig] = None,
    ):
        self.config = config or OfflineEvaluationConfig()
        self.app_config = get_config()
        
        # Initialize evaluators
        self.aggregate_evaluator = AggregateEvaluator(
            thresholds=self.app_config.thresholds,
        )
        
        # Groundedness evaluator (Azure AI Evaluation SDK)
        self._groundedness_evaluator = None
    
    def _get_groundedness_evaluator(self):
        """Lazy initialization of groundedness evaluator."""
        if self._groundedness_evaluator is None and self.config.include_groundedness:
            try:
                from azure.ai.evaluation import GroundednessEvaluator
                model_config = get_model_config()
                self._groundedness_evaluator = GroundednessEvaluator(
                    model_config=model_config,
                    threshold=self.app_config.thresholds.groundedness_threshold,
                )
            except Exception as e:
                print(f"Warning: Could not initialize GroundednessEvaluator: {e}")
        return self._groundedness_evaluator
    
    async def run(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
        extraction_agent: Optional[FinancialMetricsExtractionAgent] = None,
    ) -> BatchEvaluationResult:
        """
        Run offline evaluation on a golden dataset.
        
        Args:
            dataset_path: Path to golden dataset JSON file
            output_path: Optional path to save results
            extraction_agent: Optional pre-configured extraction agent
        
        Returns:
            BatchEvaluationResult with all evaluation metrics
        """
        start_time = time.time()
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        dataset = load_golden_dataset(dataset_path)
        print(f"Loaded {dataset.size} entries")
        
        # Initialize extraction agent if not provided
        if extraction_agent is None:
            extraction_agent = FinancialMetricsExtractionAgent()
            await extraction_agent.initialize()
        
        # Process entries with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        tasks = []
        
        for entry in dataset.entries:
            task = self._evaluate_entry_with_semaphore(
                semaphore=semaphore,
                entry=entry,
                extraction_agent=extraction_agent,
            )
            tasks.append(task)
        
        # Run with progress bar
        print("Running evaluation...")
        document_results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await coro
            if result:
                document_results.append(result)
        
        # Calculate aggregate statistics
        batch_result = self._aggregate_results(
            document_results=document_results,
            dataset_name=dataset.name,
            total_duration=time.time() - start_time,
        )
        
        # Save results
        if output_path:
            self._save_results(batch_result, output_path)
        
        # Cleanup
        if extraction_agent:
            await extraction_agent.cleanup()
        
        return batch_result
    
    async def _evaluate_entry_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        entry: GoldenDatasetEntry,
        extraction_agent: FinancialMetricsExtractionAgent,
    ) -> Optional[DocumentEvaluationResult]:
        """Evaluate a single entry with concurrency control."""
        async with semaphore:
            return await self._evaluate_entry(entry, extraction_agent)
    
    async def _evaluate_entry(
        self,
        entry: GoldenDatasetEntry,
        extraction_agent: FinancialMetricsExtractionAgent,
    ) -> Optional[DocumentEvaluationResult]:
        """Evaluate a single golden dataset entry."""
        try:
            start_time = time.time()
            
            # Extract metrics
            prior_metrics = None
            if entry.prior_year and self.config.include_consistency:
                prior_metrics = entry.prior_year.metrics
            
            extraction_start = time.time()
            extracted = await extraction_agent.extract_metrics(
                document_text=entry.raw_text,
                document_id=entry.document_id,
                fiscal_year=2025,  # Default
                prior_year_metrics=prior_metrics,
            )
            extraction_latency = (time.time() - extraction_start) * 1000
            
            # Convert to dict for evaluation
            extracted_dict = {
                name: metric.normalized_value
                for name, metric in extracted.metrics.items()
                if metric.value is not None
            }
            
            ground_truth_dict = entry.ground_truth.metrics
            
            # Run aggregate evaluation
            eval_result = self.aggregate_evaluator(
                extracted_metrics=extracted_dict,
                ground_truth_metrics=ground_truth_dict,
                prior_year_metrics=prior_metrics if self.config.include_consistency else None,
            )
            
            # Run groundedness evaluation if enabled
            groundedness_score = None
            if self.config.include_groundedness:
                groundedness_eval = self._get_groundedness_evaluator()
                if groundedness_eval:
                    try:
                        gs_result = groundedness_eval(
                            query="Extract all financial metrics from this document",
                            context=entry.raw_text[:10000],
                            response=json.dumps(extracted_dict),
                        )
                        groundedness_score = gs_result.get("groundedness", 0)
                    except Exception as e:
                        print(f"Groundedness eval failed for {entry.document_id}: {e}")
            
            # Build per-metric results
            metric_results = []
            accuracy_results = eval_result["evaluator_results"]["numerical_accuracy"]["per_metric_results"]
            for mr in accuracy_results:
                metric_results.append(MetricEvaluationResult(
                    metric_name=mr["metric_name"],
                    extracted_value=mr["extracted_value"],
                    ground_truth_value=mr["ground_truth_value"],
                    deviation=mr["deviation"],
                    accuracy_score=mr["score"],
                    passed=mr["score"] >= self.app_config.thresholds.numerical_accuracy_threshold,
                ))
            
            evaluation_latency = (time.time() - start_time) * 1000 - extraction_latency
            
            # Build document result
            scores = eval_result["scores"]
            
            return DocumentEvaluationResult(
                document_id=entry.document_id,
                numerical_accuracy_score=EvaluationScore(
                    evaluator_name="numerical_accuracy",
                    score=scores.get("numerical_accuracy", 0),
                    threshold=self.app_config.thresholds.numerical_accuracy_threshold,
                    passed=scores.get("numerical_accuracy", 0) >= self.app_config.thresholds.numerical_accuracy_threshold,
                    reason=eval_result["evaluator_results"]["numerical_accuracy"]["numerical_accuracy_reason"],
                    details={"per_metric": accuracy_results},
                ),
                completeness_score=EvaluationScore(
                    evaluator_name="completeness",
                    score=scores.get("completeness", 0),
                    threshold=self.app_config.thresholds.completeness_threshold,
                    passed=scores.get("completeness", 0) >= self.app_config.thresholds.completeness_threshold,
                    reason=eval_result["evaluator_results"]["completeness"]["completeness_reason"],
                    details={
                        "missing": eval_result["evaluator_results"]["completeness"]["missing_metrics"],
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
                    },
                ) if "math_consistency" in scores else None,
                consistency_score=EvaluationScore(
                    evaluator_name="consistency",
                    score=scores.get("consistency", 5.0),
                    threshold=self.app_config.thresholds.consistency_threshold,
                    passed=scores.get("consistency", 5.0) >= self.app_config.thresholds.consistency_threshold,
                    reason=eval_result["evaluator_results"].get("consistency", {}).get("consistency_reason", "No prior year data"),
                ) if "consistency" in scores else None,
                groundedness_score=EvaluationScore(
                    evaluator_name="groundedness",
                    score=groundedness_score or 0,
                    threshold=self.app_config.thresholds.groundedness_threshold,
                    passed=(groundedness_score or 0) >= self.app_config.thresholds.groundedness_threshold,
                    reason="Groundedness evaluation",
                ) if groundedness_score is not None else None,
                metric_results=metric_results,
                overall_confidence=eval_result["confidence"],
                flagged_for_review=eval_result["flagged_for_review"],
                extraction_latency_ms=extraction_latency,
                evaluation_latency_ms=evaluation_latency,
            )
            
        except Exception as e:
            print(f"Error evaluating {entry.document_id}: {e}")
            return None
    
    def _aggregate_results(
        self,
        document_results: list[DocumentEvaluationResult],
        dataset_name: str,
        total_duration: float,
    ) -> BatchEvaluationResult:
        """Aggregate individual document results into batch statistics."""
        
        # Calculate overall scores
        overall_scores = {}
        score_lists = {
            "numerical_accuracy": [],
            "completeness": [],
            "math_consistency": [],
            "consistency": [],
            "groundedness": [],
        }
        
        for doc in document_results:
            score_lists["numerical_accuracy"].append(doc.numerical_accuracy_score.score)
            score_lists["completeness"].append(doc.completeness_score.score)
            if doc.math_consistency_score:
                score_lists["math_consistency"].append(doc.math_consistency_score.score)
            if doc.consistency_score:
                score_lists["consistency"].append(doc.consistency_score.score)
            if doc.groundedness_score:
                score_lists["groundedness"].append(doc.groundedness_score.score)
        
        for key, scores in score_lists.items():
            if scores:
                overall_scores[key] = statistics.mean(scores)
        
        # Calculate per-metric accuracy
        per_metric_accuracy = {}
        metric_scores = {}
        
        for doc in document_results:
            for mr in doc.metric_results:
                if mr.metric_name not in metric_scores:
                    metric_scores[mr.metric_name] = []
                metric_scores[mr.metric_name].append(mr.accuracy_score)
        
        for metric, scores in metric_scores.items():
            per_metric_accuracy[metric] = {
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
            }
        
        # Identify failed extractions
        failed_extractions = []
        for doc in document_results:
            if not doc.numerical_accuracy_score.passed:
                failed_extractions.append({
                    "document_id": doc.document_id,
                    "score": doc.numerical_accuracy_score.score,
                    "reason": doc.numerical_accuracy_score.reason,
                })
        
        # Count low confidence
        low_confidence_count = sum(1 for doc in document_results if doc.flagged_for_review)
        
        return BatchEvaluationResult(
            evaluation_id=f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            dataset_name=dataset_name,
            dataset_size=len(document_results),
            document_results=document_results,
            overall_scores=overall_scores,
            per_metric_accuracy=per_metric_accuracy,
            failed_extractions=failed_extractions,
            low_confidence_count=low_confidence_count,
            total_duration_seconds=total_duration,
        )
    
    def _save_results(
        self,
        results: BatchEvaluationResult,
        output_path: str,
    ) -> None:
        """Save evaluation results to file."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        if self.config.output_format in ("json", "both"):
            json_path = output_dir / f"{results.evaluation_id}.json"
            with open(json_path, "w") as f:
                json.dump(results.model_dump(mode="json"), f, indent=2, default=str)
            print(f"Saved results to {json_path}")
        
        # Save CSV summary
        if self.config.output_format in ("csv", "both"):
            csv_path = output_dir / f"{results.evaluation_id}_summary.csv"
            self._save_csv_summary(results, csv_path)
            print(f"Saved summary to {csv_path}")
    
    def _save_csv_summary(
        self,
        results: BatchEvaluationResult,
        csv_path: Path,
    ) -> None:
        """Save CSV summary of results."""
        import csv
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Overall scores
            writer.writerow(["Overall Scores"])
            writer.writerow(["Evaluator", "Score"])
            for evaluator, score in results.overall_scores.items():
                writer.writerow([evaluator, f"{score:.2f}"])
            
            writer.writerow([])
            
            # Per-metric accuracy
            writer.writerow(["Per-Metric Accuracy"])
            writer.writerow(["Metric", "Mean", "Std", "Min", "Max", "Count"])
            for metric, stats in results.per_metric_accuracy.items():
                writer.writerow([
                    metric,
                    f"{stats['mean']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    stats['count'],
                ])
    
    def check_ci_threshold(
        self,
        results: BatchEvaluationResult,
    ) -> tuple[bool, str]:
        """
        Check if results meet CI/CD threshold.
        
        Returns:
            Tuple of (passed, message)
        """
        avg_score = statistics.mean(results.overall_scores.values())
        passed = avg_score >= self.config.fail_threshold
        
        if passed:
            message = f"PASSED: Average score {avg_score:.2f} >= threshold {self.config.fail_threshold}"
        else:
            message = f"FAILED: Average score {avg_score:.2f} < threshold {self.config.fail_threshold}"
        
        return passed, message


# =============================================================================
# A/B Testing Support
# =============================================================================

class ABTestEvaluator:
    """
    Compare two extraction configurations (prompts, models, etc.)
    with statistical significance testing.
    """
    
    def __init__(self):
        self.offline_evaluator = OfflineEvaluator()
    
    async def compare(
        self,
        dataset_path: str,
        agent_a: FinancialMetricsExtractionAgent,
        agent_b: FinancialMetricsExtractionAgent,
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run A/B comparison between two agent configurations.
        
        Returns:
            Comparison results with statistical analysis
        """
        # Run evaluation for both configurations
        print("Running evaluation for Configuration A...")
        results_a = await self.offline_evaluator.run(
            dataset_path=dataset_path,
            extraction_agent=agent_a,
        )
        
        print("Running evaluation for Configuration B...")
        results_b = await self.offline_evaluator.run(
            dataset_path=dataset_path,
            extraction_agent=agent_b,
        )
        
        # Statistical comparison
        comparison = self._statistical_comparison(results_a, results_b)
        
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "ab_comparison.json", "w") as f:
                json.dump(comparison, f, indent=2)
        
        return comparison
    
    def _statistical_comparison(
        self,
        results_a: BatchEvaluationResult,
        results_b: BatchEvaluationResult,
    ) -> dict[str, Any]:
        """Perform statistical comparison between two result sets."""
        comparison = {
            "summary": {},
            "per_evaluator": {},
            "recommendation": "",
        }
        
        # Compare overall scores
        for evaluator in results_a.overall_scores.keys():
            score_a = results_a.overall_scores.get(evaluator, 0)
            score_b = results_b.overall_scores.get(evaluator, 0)
            
            diff = score_b - score_a
            pct_change = (diff / score_a * 100) if score_a != 0 else 0
            
            comparison["per_evaluator"][evaluator] = {
                "config_a_score": score_a,
                "config_b_score": score_b,
                "difference": diff,
                "pct_change": pct_change,
                "winner": "A" if score_a > score_b else "B" if score_b > score_a else "tie",
            }
        
        # Overall summary
        avg_a = statistics.mean(results_a.overall_scores.values())
        avg_b = statistics.mean(results_b.overall_scores.values())
        
        comparison["summary"] = {
            "config_a_average": avg_a,
            "config_b_average": avg_b,
            "difference": avg_b - avg_a,
            "winner": "A" if avg_a > avg_b else "B" if avg_b > avg_a else "tie",
        }
        
        # Recommendation
        if avg_b > avg_a + 0.1:
            comparison["recommendation"] = "Configuration B shows meaningful improvement. Consider adopting."
        elif avg_a > avg_b + 0.1:
            comparison["recommendation"] = "Configuration A performs better. Keep current configuration."
        else:
            comparison["recommendation"] = "No significant difference. Consider other factors."
        
        return comparison


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_offline_evaluation(
    dataset_path: str,
    output_path: str = "results/",
    ci_mode: bool = False,
    threshold: float = 3.5,
) -> int:
    """
    Main entry point for offline evaluation.
    
    Args:
        dataset_path: Path to golden dataset
        output_path: Directory for results
        ci_mode: If True, return exit code based on threshold
        threshold: Minimum score to pass in CI mode
    
    Returns:
        Exit code (0 = success, 1 = failed threshold)
    """
    config = OfflineEvaluationConfig(fail_threshold=threshold)
    evaluator = OfflineEvaluator(config=config)
    
    results = await evaluator.run(
        dataset_path=dataset_path,
        output_path=output_path,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Dataset: {results.dataset_name}")
    print(f"Documents evaluated: {results.dataset_size}")
    print(f"Duration: {results.total_duration_seconds:.1f}s")
    print()
    print("Overall Scores:")
    for evaluator_name, score in results.overall_scores.items():
        status = "✓" if score >= threshold else "✗"
        print(f"  {status} {evaluator_name}: {score:.2f}")
    print()
    print(f"Low confidence (flagged): {results.low_confidence_count}/{results.dataset_size}")
    print(f"Failed extractions: {len(results.failed_extractions)}")
    
    if ci_mode:
        passed, message = evaluator.check_ci_threshold(results)
        print()
        print(message)
        return 0 if passed else 1
    
    return 0


if __name__ == "__main__":
    import sys
    
    # Simple CLI usage
    dataset = sys.argv[1] if len(sys.argv) > 1 else "data/golden_dataset.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "results/"
    
    exit_code = asyncio.run(run_offline_evaluation(dataset, output))
    sys.exit(exit_code)
