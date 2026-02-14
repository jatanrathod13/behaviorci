"""Threshold evaluation for determining pass/fail status."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from promptguard.bundle.models import ThresholdConfig
    from promptguard.runner.engine import CaseResult


@dataclass
class ThresholdResult:
    """Result of evaluating a single threshold.

    Attributes:
        threshold: The threshold configuration
        passed: Whether the threshold was met
        actual_value: Computed metric value
        expected_value: Threshold target value
        operator: Comparison operator used
    """

    metric: str
    passed: bool
    actual_value: float
    expected_value: float
    operator: str


@dataclass
class ThresholdEvaluation:
    """Complete threshold evaluation result.

    Attributes:
        passed: Whether all thresholds passed
        results: Individual threshold results
        metrics: All computed metrics
    """

    passed: bool
    results: list[ThresholdResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class ThresholdEvaluator:
    """Evaluates thresholds against run results.

    Computes metrics from case results and compares
    against configured threshold targets.
    """

    def __init__(self, thresholds: list[ThresholdConfig]) -> None:
        """Initialize with threshold configurations.

        Args:
            thresholds: List of threshold configs to evaluate
        """
        self.thresholds = thresholds

    def evaluate(self, case_results: list[CaseResult]) -> ThresholdEvaluation:
        """Evaluate all thresholds against case results.

        Args:
            case_results: Results from running all cases

        Returns:
            ThresholdEvaluation with pass/fail and details
        """
        if not case_results:
            return ThresholdEvaluation(passed=True, metrics={})

        # Compute all metrics
        metrics = self._compute_metrics(case_results)

        # Evaluate each threshold
        results: list[ThresholdResult] = []
        all_passed = True

        for threshold in self.thresholds:
            if threshold.metric not in metrics:
                # Unknown metric = fail
                result = ThresholdResult(
                    metric=threshold.metric,
                    passed=False,
                    actual_value=0.0,
                    expected_value=threshold.value,
                    operator=threshold.operator,
                )
                all_passed = False
            else:
                actual = metrics[threshold.metric]
                passed = self._compare(actual, threshold.operator, threshold.value)
                result = ThresholdResult(
                    metric=threshold.metric,
                    passed=passed,
                    actual_value=actual,
                    expected_value=threshold.value,
                    operator=threshold.operator,
                )
                if not passed:
                    all_passed = False

            results.append(result)

        return ThresholdEvaluation(passed=all_passed, results=results, metrics=metrics)

    def _compute_metrics(self, case_results: list[CaseResult]) -> dict[str, float]:
        """Compute all metrics from case results.

        Args:
            case_results: List of case results

        Returns:
            Dict of metric name to computed value
        """
        total = len(case_results)
        if total == 0:
            return {}

        passed = sum(1 for r in case_results if r.passed)
        schema_valid = sum(
            1 for r in case_results if r.evaluation and r.evaluation.schema_valid
        )
        errors = sum(1 for r in case_results if r.error is not None)

        # Latency metrics
        latencies = [r.latency_ms for r in case_results if r.latency_ms is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0

        return {
            "pass_rate": passed / total,
            "schema_valid_rate": schema_valid / total,
            "error_rate": errors / total,
            "total_cases": float(total),
            "passed_cases": float(passed),
            "failed_cases": float(total - passed),
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
        }

    def _compare(self, actual: float, operator: str, expected: float) -> bool:
        """Compare actual value against expected using operator.

        Args:
            actual: Computed metric value
            operator: Comparison operator
            expected: Threshold target value

        Returns:
            Whether comparison passes
        """
        ops = {
            ">=": lambda a, e: a >= e,
            ">": lambda a, e: a > e,
            "<=": lambda a, e: a <= e,
            "<": lambda a, e: a < e,
            "==": lambda a, e: abs(a - e) < 0.0001,  # Float equality
        }
        return ops.get(operator, lambda a, e: False)(actual, expected)
