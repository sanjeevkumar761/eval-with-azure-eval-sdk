"""
Custom Evaluators for Financial Metrics Extraction
===================================================

Implements custom evaluators following Azure AI Evaluation SDK patterns:
- NumericalAccuracyEvaluator: Measures deviation from ground truth
- MetricCompletenessEvaluator: Measures extraction completeness
- YoYConsistencyEvaluator: Checks year-over-year consistency
"""

import math
from typing import Optional, Any
from dataclasses import dataclass

from config import EvaluationThresholds, REQUIRED_METRICS


@dataclass
class EvaluatorResult:
    """Standard result format for custom evaluators."""
    score: float  # 0-5 scale
    passed: bool
    reason: str
    details: dict[str, Any]


class NumericalAccuracyEvaluator:
    """
    Evaluates numerical accuracy of extracted financial metrics.
    
    Compares extracted values against ground truth and scores based on
    percentage deviation.
    
    Scoring:
        - Score 5: Deviation < 0.1%
        - Score 4: Deviation < 1%
        - Score 3: Deviation < 5%
        - Score 2: Deviation < 10%
        - Score 1: Deviation < 25%
        - Score 0: Deviation >= 25%
    """
    
    def __init__(
        self,
        thresholds: Optional[EvaluationThresholds] = None,
        pass_threshold: float = 3.5,
    ):
        self.thresholds = thresholds or EvaluationThresholds()
        self.pass_threshold = pass_threshold
    
    def _calculate_deviation(
        self,
        extracted: float,
        ground_truth: float,
    ) -> float:
        """Calculate percentage deviation between values."""
        if ground_truth == 0:
            return 1.0 if extracted != 0 else 0.0
        return abs(extracted - ground_truth) / abs(ground_truth)
    
    def _deviation_to_score(self, deviation: float) -> float:
        """Convert deviation to 0-5 score."""
        if deviation < self.thresholds.deviation_excellent:
            return 5.0
        elif deviation < self.thresholds.deviation_good:
            return 4.0
        elif deviation < self.thresholds.deviation_acceptable:
            return 3.0
        elif deviation < self.thresholds.deviation_marginal:
            return 2.0
        elif deviation < self.thresholds.deviation_poor:
            return 1.0
        else:
            return 0.0
    
    def __call__(
        self,
        extracted_metrics: dict[str, float],
        ground_truth_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """
        Evaluate numerical accuracy of extracted metrics.
        
        Args:
            extracted_metrics: Dictionary of metric name to extracted value
            ground_truth_metrics: Dictionary of metric name to ground truth value
        
        Returns:
            Dictionary with evaluation results
        """
        per_metric_results = []
        total_deviation = 0.0
        evaluated_count = 0
        
        for metric_name, ground_truth in ground_truth_metrics.items():
            if metric_name not in extracted_metrics:
                continue
            
            extracted = extracted_metrics[metric_name]
            if extracted is None or ground_truth is None:
                continue
            
            deviation = self._calculate_deviation(extracted, ground_truth)
            score = self._deviation_to_score(deviation)
            
            per_metric_results.append({
                "metric_name": metric_name,
                "extracted_value": extracted,
                "ground_truth_value": ground_truth,
                "deviation": deviation,
                "deviation_pct": f"{deviation * 100:.2f}%",
                "score": score,
            })
            
            total_deviation += deviation
            evaluated_count += 1
        
        # Calculate aggregate score
        if evaluated_count == 0:
            avg_score = 0.0
            avg_deviation = 1.0
            reason = "No metrics could be evaluated"
        else:
            avg_deviation = total_deviation / evaluated_count
            avg_score = self._deviation_to_score(avg_deviation)
            reason = f"Average deviation: {avg_deviation * 100:.2f}% across {evaluated_count} metrics"
        
        passed = avg_score >= self.pass_threshold
        
        return {
            "numerical_accuracy": avg_score,
            "numerical_accuracy_result": "pass" if passed else "fail",
            "numerical_accuracy_reason": reason,
            "numerical_accuracy_threshold": self.pass_threshold,
            "average_deviation": avg_deviation,
            "evaluated_metrics_count": evaluated_count,
            "per_metric_results": per_metric_results,
        }


class MetricCompletenessEvaluator:
    """
    Evaluates completeness of metric extraction.
    
    Measures what percentage of required metrics were successfully extracted.
    
    Scoring:
        - Score 5: 100% complete
        - Score 4: >= 90% complete
        - Score 3: >= 75% complete
        - Score 2: >= 50% complete
        - Score 1: >= 25% complete
        - Score 0: < 25% complete
    """
    
    def __init__(
        self,
        required_metrics: Optional[list[str]] = None,
        pass_threshold: float = 4.0,
    ):
        self.required_metrics = required_metrics or REQUIRED_METRICS
        self.pass_threshold = pass_threshold
    
    def _completeness_to_score(self, completeness: float) -> float:
        """Convert completeness ratio to 0-5 score."""
        if completeness >= 1.0:
            return 5.0
        elif completeness >= 0.90:
            return 4.0
        elif completeness >= 0.75:
            return 3.0
        elif completeness >= 0.50:
            return 2.0
        elif completeness >= 0.25:
            return 1.0
        else:
            return 0.0
    
    def __call__(
        self,
        extracted_metrics: dict[str, Any],
        required_metrics: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Evaluate completeness of extraction.
        
        Args:
            extracted_metrics: Dictionary of extracted metrics
            required_metrics: Optional override for required metrics list
        
        Returns:
            Dictionary with evaluation results
        """
        required = required_metrics or self.required_metrics
        
        found_metrics = []
        missing_metrics = []
        
        for metric in required:
            if metric in extracted_metrics and extracted_metrics[metric] is not None:
                found_metrics.append(metric)
            else:
                missing_metrics.append(metric)
        
        completeness = len(found_metrics) / len(required) if required else 1.0
        score = self._completeness_to_score(completeness)
        passed = score >= self.pass_threshold
        
        reason = f"Extracted {len(found_metrics)}/{len(required)} required metrics ({completeness * 100:.1f}%)"
        
        return {
            "completeness": score,
            "completeness_result": "pass" if passed else "fail",
            "completeness_reason": reason,
            "completeness_threshold": self.pass_threshold,
            "completeness_pct": completeness,
            "found_metrics": found_metrics,
            "missing_metrics": missing_metrics,
            "required_count": len(required),
            "found_count": len(found_metrics),
        }


class YoYConsistencyEvaluator:
    """
    Evaluates year-over-year consistency of extracted metrics.
    
    Checks if changes from prior year are within reasonable bounds
    and flags potential anomalies.
    
    Anomaly Detection:
        - Revenue change > 50% YoY
        - Net income sign flip
        - Asset/Liability ratio swing > 30%
        - Any metric change > 100%
    """
    
    def __init__(
        self,
        thresholds: Optional[EvaluationThresholds] = None,
        pass_threshold: float = 3.0,
    ):
        self.thresholds = thresholds or EvaluationThresholds()
        self.pass_threshold = pass_threshold
        
        # Metric-specific thresholds
        self.metric_thresholds = {
            "total_revenue": self.thresholds.yoy_revenue_max_change,
            "gross_profit": self.thresholds.yoy_revenue_max_change,
            "net_income": self.thresholds.yoy_revenue_max_change,
            "total_assets": self.thresholds.yoy_asset_max_change,
            "total_liabilities": self.thresholds.yoy_asset_max_change,
            "current_ratio": self.thresholds.yoy_ratio_max_change,
            "debt_to_equity": self.thresholds.yoy_ratio_max_change,
        }
        self.default_threshold = 0.50  # 50% change for unlisted metrics
    
    def _calculate_yoy_change(
        self,
        current: float,
        prior: float,
    ) -> Optional[float]:
        """Calculate YoY percentage change."""
        if prior == 0:
            return None if current == 0 else 1.0
        return (current - prior) / abs(prior)
    
    def _check_sign_flip(self, current: float, prior: float) -> bool:
        """Check if sign has flipped (e.g., profit to loss)."""
        return (current > 0 and prior < 0) or (current < 0 and prior > 0)
    
    def __call__(
        self,
        current_metrics: dict[str, float],
        prior_year_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """
        Evaluate year-over-year consistency.
        
        Args:
            current_metrics: Dictionary of current year metrics
            prior_year_metrics: Dictionary of prior year metrics
        
        Returns:
            Dictionary with evaluation results
        """
        anomalies = []
        consistency_results = []
        total_checks = 0
        passed_checks = 0
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in prior_year_metrics:
                continue
            
            prior_value = prior_year_metrics[metric_name]
            if current_value is None or prior_value is None:
                continue
            
            total_checks += 1
            yoy_change = self._calculate_yoy_change(current_value, prior_value)
            threshold = self.metric_thresholds.get(metric_name, self.default_threshold)
            
            is_anomaly = False
            anomaly_reasons = []
            
            # Check for excessive change
            if yoy_change is not None and abs(yoy_change) > threshold:
                is_anomaly = True
                anomaly_reasons.append(
                    f"Change of {yoy_change * 100:.1f}% exceeds threshold of {threshold * 100:.0f}%"
                )
            
            # Check for sign flip on income-related metrics
            sign_sensitive_metrics = ["net_income", "operating_income", "gross_profit"]
            if metric_name in sign_sensitive_metrics and self._check_sign_flip(current_value, prior_value):
                is_anomaly = True
                anomaly_reasons.append("Sign flip detected (profit to loss or vice versa)")
            
            result = {
                "metric_name": metric_name,
                "current_value": current_value,
                "prior_value": prior_value,
                "yoy_change": yoy_change,
                "yoy_change_pct": f"{yoy_change * 100:.1f}%" if yoy_change else "N/A",
                "threshold": threshold,
                "is_anomaly": is_anomaly,
                "anomaly_reasons": anomaly_reasons,
            }
            consistency_results.append(result)
            
            if is_anomaly:
                anomalies.append(result)
            else:
                passed_checks += 1
        
        # Calculate score based on anomaly rate
        if total_checks == 0:
            score = 5.0
            reason = "No prior year data available for comparison"
        else:
            anomaly_rate = len(anomalies) / total_checks
            # Score decreases with more anomalies
            if anomaly_rate == 0:
                score = 5.0
            elif anomaly_rate <= 0.1:
                score = 4.0
            elif anomaly_rate <= 0.2:
                score = 3.0
            elif anomaly_rate <= 0.3:
                score = 2.0
            elif anomaly_rate <= 0.5:
                score = 1.0
            else:
                score = 0.0
            
            reason = f"{len(anomalies)} anomalies detected out of {total_checks} comparisons ({anomaly_rate * 100:.1f}% anomaly rate)"
        
        passed = score >= self.pass_threshold
        
        return {
            "consistency": score,
            "consistency_result": "pass" if passed else "fail",
            "consistency_reason": reason,
            "consistency_threshold": self.pass_threshold,
            "anomaly_count": len(anomalies),
            "total_comparisons": total_checks,
            "anomalies": anomalies,
            "all_results": consistency_results,
        }


class MathConsistencyEvaluator:
    """
    Evaluates internal mathematical consistency of extracted metrics.
    
    Unlike NumericalAccuracyEvaluator (which needs ground truth), this works
    in production without any reference data by checking that extracted values
    satisfy known financial relationships.
    
    Checks:
        - gross_profit = total_revenue - cost_of_goods_sold
        - operating_income = gross_profit - operating_expenses
        - total_assets = total_liabilities + shareholders_equity
        - current_ratio ≈ current_assets / current_liabilities
        - gross_margin ≈ gross_profit / total_revenue
        - net_margin ≈ net_income / total_revenue
        - debt_to_equity ≈ total_liabilities / shareholders_equity
        - return_on_equity ≈ net_income / shareholders_equity
    """
    
    def __init__(self, tolerance: float = 0.05, pass_threshold: float = 3.5):
        self.tolerance = tolerance  # 5% tolerance for rounding
        self.pass_threshold = pass_threshold
    
    def _check_relationship(
        self,
        name: str,
        metrics: dict[str, float],
        result_key: str,
        operand_keys: list[str],
        operation: str,
    ) -> Optional[dict[str, Any]]:
        """
        Check a single mathematical relationship.
        
        Returns None if required metrics are missing, or a dict with check results.
        """
        result_val = metrics.get(result_key)
        operand_vals = [metrics.get(k) for k in operand_keys]
        
        if result_val is None or any(v is None for v in operand_vals):
            return None
        
        if operation == "subtract":
            expected = operand_vals[0] - operand_vals[1]
        elif operation == "add":
            expected = sum(operand_vals)
        elif operation == "divide":
            if operand_vals[1] == 0:
                return None
            expected = operand_vals[0] / operand_vals[1]
        else:
            return None
        
        if expected == 0:
            deviation = 0.0 if result_val == 0 else 1.0
        else:
            deviation = abs(result_val - expected) / abs(expected)
        
        passed = deviation <= self.tolerance
        
        return {
            "check_name": name,
            "formula": f"{result_key} = {' '.join(f'{op} {k}' for op, k in zip([operation] + [''] * len(operand_keys), operand_keys))}".strip(),
            "expected": expected,
            "actual": result_val,
            "deviation": deviation,
            "deviation_pct": f"{deviation * 100:.2f}%",
            "passed": passed,
        }
    
    def __call__(self, extracted_metrics: dict[str, float]) -> dict[str, Any]:
        """
        Evaluate mathematical consistency of extracted metrics.
        
        Args:
            extracted_metrics: Dictionary of metric name to extracted value
            
        Returns:
            Dictionary with evaluation results
        """
        checks = []
        
        # Income statement relationships
        check = self._check_relationship(
            "Gross Profit = Revenue - COGS",
            extracted_metrics, "gross_profit",
            ["total_revenue", "cost_of_goods_sold"], "subtract",
        )
        if check:
            checks.append(check)
        
        check = self._check_relationship(
            "Operating Income = Gross Profit - OpEx",
            extracted_metrics, "operating_income",
            ["gross_profit", "operating_expenses"], "subtract",
        )
        if check:
            checks.append(check)
        
        # Balance sheet relationship
        check = self._check_relationship(
            "Total Assets = Liabilities + Equity",
            extracted_metrics, "total_assets",
            ["total_liabilities", "shareholders_equity"], "add",
        )
        if check:
            checks.append(check)
        
        # Ratio checks
        check = self._check_relationship(
            "Current Ratio = Current Assets / Current Liabilities",
            extracted_metrics, "current_ratio",
            ["current_assets", "current_liabilities"], "divide",
        )
        if check:
            checks.append(check)
        
        check = self._check_relationship(
            "Gross Margin = Gross Profit / Revenue",
            extracted_metrics, "gross_margin",
            ["gross_profit", "total_revenue"], "divide",
        )
        if check:
            checks.append(check)
        
        check = self._check_relationship(
            "Net Margin = Net Income / Revenue",
            extracted_metrics, "net_margin",
            ["net_income", "total_revenue"], "divide",
        )
        if check:
            checks.append(check)
        
        check = self._check_relationship(
            "D/E Ratio = Liabilities / Equity",
            extracted_metrics, "debt_to_equity",
            ["total_liabilities", "shareholders_equity"], "divide",
        )
        if check:
            checks.append(check)
        
        check = self._check_relationship(
            "ROE = Net Income / Equity",
            extracted_metrics, "return_on_equity",
            ["net_income", "shareholders_equity"], "divide",
        )
        if check:
            checks.append(check)
        
        # Net income should be less than operating income (interest + taxes)
        oi = extracted_metrics.get("operating_income")
        ni = extracted_metrics.get("net_income")
        if oi is not None and ni is not None and oi != 0:
            reasonable = ni <= oi * 1.05  # Allow 5% tolerance
            checks.append({
                "check_name": "Net Income ≤ Operating Income",
                "formula": "net_income <= operating_income (after interest/taxes)",
                "expected": f"<= {oi:,.0f}",
                "actual": ni,
                "deviation": 0.0 if reasonable else abs(ni - oi) / abs(oi),
                "deviation_pct": "0.00%" if reasonable else f"{abs(ni - oi) / abs(oi) * 100:.2f}%",
                "passed": reasonable,
            })
        
        # Calculate score
        if not checks:
            score = 3.0  # Neutral — not enough data to check
            reason = "Insufficient metrics for consistency checks"
        else:
            passed_count = sum(1 for c in checks if c["passed"])
            pass_rate = passed_count / len(checks)
            
            if pass_rate >= 1.0:
                score = 5.0
            elif pass_rate >= 0.85:
                score = 4.0
            elif pass_rate >= 0.70:
                score = 3.0
            elif pass_rate >= 0.50:
                score = 2.0
            elif pass_rate >= 0.25:
                score = 1.0
            else:
                score = 0.0
            
            failed = [c for c in checks if not c["passed"]]
            reason = f"{passed_count}/{len(checks)} checks passed"
            if failed:
                reason += f" — failed: {', '.join(c['check_name'] for c in failed)}"
        
        passed = score >= self.pass_threshold
        
        return {
            "math_consistency": score,
            "math_consistency_result": "pass" if passed else "fail",
            "math_consistency_reason": reason,
            "math_consistency_threshold": self.pass_threshold,
            "checks_total": len(checks),
            "checks_passed": sum(1 for c in checks if c["passed"]),
            "checks_failed": [c for c in checks if not c["passed"]],
            "all_checks": checks,
        }


class AggregateEvaluator:
    """
    Aggregates results from multiple evaluators into overall scores.
    
    Combines:
        - NumericalAccuracyEvaluator
        - MetricCompletenessEvaluator
        - YoYConsistencyEvaluator
        - GroundednessEvaluator (from Azure AI Evaluation SDK)
    """
    
    def __init__(
        self,
        thresholds: Optional[EvaluationThresholds] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self.thresholds = thresholds or EvaluationThresholds()
        
        # Default weights for aggregation
        self.weights = weights or {
            "numerical_accuracy": 0.30,
            "completeness": 0.20,
            "math_consistency": 0.20,
            "consistency": 0.15,
            "groundedness": 0.15,
        }
        
        # Initialize evaluators
        self.numerical_evaluator = NumericalAccuracyEvaluator(
            thresholds=self.thresholds,
            pass_threshold=self.thresholds.numerical_accuracy_threshold,
        )
        self.completeness_evaluator = MetricCompletenessEvaluator(
            pass_threshold=self.thresholds.completeness_threshold,
        )
        self.math_consistency_evaluator = MathConsistencyEvaluator()
        self.consistency_evaluator = YoYConsistencyEvaluator(
            thresholds=self.thresholds,
            pass_threshold=self.thresholds.consistency_threshold,
        )
    
    def calculate_confidence(
        self,
        scores: dict[str, float],
    ) -> float:
        """
        Calculate overall confidence score (0-1) from individual scores.
        
        Args:
            scores: Dictionary of evaluator name to score (0-5)
        
        Returns:
            Confidence score between 0 and 1
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for evaluator, weight in self.weights.items():
            if evaluator in scores:
                # Normalize to 0-1 range
                normalized_score = scores[evaluator] / 5.0
                weighted_sum += normalized_score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def __call__(
        self,
        extracted_metrics: dict[str, float],
        ground_truth_metrics: dict[str, float],
        prior_year_metrics: Optional[dict[str, float]] = None,
        groundedness_score: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Run all evaluators and aggregate results.
        
        Args:
            extracted_metrics: Dictionary of extracted metric values
            ground_truth_metrics: Dictionary of ground truth values
            prior_year_metrics: Optional prior year values for consistency check
            groundedness_score: Optional pre-computed groundedness score
        
        Returns:
            Comprehensive evaluation results
        """
        results = {}
        scores = {}
        
        # Numerical accuracy
        accuracy_result = self.numerical_evaluator(
            extracted_metrics=extracted_metrics,
            ground_truth_metrics=ground_truth_metrics,
        )
        results["numerical_accuracy"] = accuracy_result
        scores["numerical_accuracy"] = accuracy_result["numerical_accuracy"]
        
        # Completeness
        completeness_result = self.completeness_evaluator(
            extracted_metrics=extracted_metrics,
        )
        results["completeness"] = completeness_result
        scores["completeness"] = completeness_result["completeness"]
        
        # Math consistency (works without ground truth)
        math_result = self.math_consistency_evaluator(
            extracted_metrics=extracted_metrics,
        )
        results["math_consistency"] = math_result
        scores["math_consistency"] = math_result["math_consistency"]
        
        # YoY Consistency (if prior year data available)
        if prior_year_metrics:
            consistency_result = self.consistency_evaluator(
                current_metrics=extracted_metrics,
                prior_year_metrics=prior_year_metrics,
            )
            results["consistency"] = consistency_result
            scores["consistency"] = consistency_result["consistency"]
        
        # Groundedness (if provided)
        if groundedness_score is not None:
            scores["groundedness"] = groundedness_score
            results["groundedness"] = {
                "groundedness": groundedness_score,
                "groundedness_result": "pass" if groundedness_score >= self.thresholds.groundedness_threshold else "fail",
            }
        
        # Calculate overall confidence
        confidence = self.calculate_confidence(scores)
        
        # Determine if flagged for review
        flagged = confidence < self.thresholds.confidence_threshold
        
        return {
            "scores": scores,
            "confidence": confidence,
            "flagged_for_review": flagged,
            "evaluator_results": results,
            "summary": {
                "numerical_accuracy": scores.get("numerical_accuracy", 0),
                "completeness": scores.get("completeness", 0),
                "math_consistency": scores.get("math_consistency"),
                "consistency": scores.get("consistency"),
                "groundedness": scores.get("groundedness"),
                "overall_confidence": confidence,
            },
        }
