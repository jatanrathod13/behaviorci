"""JSON reporter for machine-readable output."""

from __future__ import annotations

import json
from typing import Any

from promptguard.reporters.base import Reporter
from promptguard.reporters.registry import register_reporter
from promptguard.runner.engine import RunResult


@register_reporter("json")
class JSONReporter(Reporter):
    """JSON reporter for machine-readable output.

    Produces structured JSON suitable for:
    - CI/CD pipelines
    - Integration with other tools
    - Automated analysis
    """

    def emit(self, result: RunResult, verbose: bool = False) -> str:
        """Generate JSON report.

        Args:
            result: Run result to report
            verbose: Include full case details

        Returns:
            JSON string
        """
        data = self._build_report(result, verbose)
        return json.dumps(data, indent=2, default=str)

    def _build_report(self, result: RunResult, verbose: bool) -> dict[str, Any]:
        """Build report data structure.

        Args:
            result: Run result
            verbose: Include case details

        Returns:
            Report dictionary
        """
        report: dict[str, Any] = {
            "bundle": result.bundle_name,
            "passed": result.passed,
            "summary": result.summary,
            "provider": result.provider,
            "model": result.model,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "duration_ms": result.duration_ms,
        }

        # Threshold results
        if result.threshold_evaluation:
            report["thresholds"] = {
                "passed": result.threshold_evaluation.passed,
                "metrics": result.threshold_evaluation.metrics,
                "results": [
                    {
                        "metric": t.metric,
                        "passed": t.passed,
                        "actual": t.actual_value,
                        "expected": t.expected_value,
                        "operator": t.operator,
                    }
                    for t in result.threshold_evaluation.results
                ],
            }

        # Case results
        if verbose:
            report["cases"] = [
                {
                    "id": c.case_id,
                    "passed": c.passed,
                    "latency_ms": c.latency_ms,
                    "output": c.output,
                    "error": c.error,
                    "evaluation": self._format_evaluation(c.evaluation) if c.evaluation else None,
                }
                for c in result.case_results
            ]
        else:
            # Summary only
            report["cases"] = [
                {
                    "id": c.case_id,
                    "passed": c.passed,
                    "latency_ms": c.latency_ms,
                    "error": c.error,
                }
                for c in result.case_results
            ]

        return report

    def _format_evaluation(self, evaluation: Any) -> dict[str, Any]:
        """Format evaluation result for JSON.

        Args:
            evaluation: EvaluationResult object

        Returns:
            Dictionary representation
        """
        return {
            "passed": evaluation.passed,
            "schema_valid": evaluation.schema_valid,
            "schema_errors": evaluation.schema_errors,
            "invariants_passed": evaluation.invariants_passed,
            "invariant_errors": evaluation.invariant_errors,
        }
