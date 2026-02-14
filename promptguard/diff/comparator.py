"""Comparator for comparing runs against baselines."""

from __future__ import annotations

from promptguard.baseline.models import BaselineRun
from promptguard.diff.models import DiffEntry, DiffResult, DiffType, MetricChange
from promptguard.runner.engine import RunResult


class Comparator:
    """Compares current run results against a baseline.

    Detects:
    - NEW failures: passed in baseline, failed in current
    - FIXED: failed in baseline, passed in current
    - REGRESSION: passed in baseline, failed in current (same as new_failure)
    - IMPROVED: failed in baseline, passed in current (same as fixed)
    - METRIC_CHANGES: changes in pass_rate, avg_latency, etc.
    """

    def __init__(self, baseline: BaselineRun, current: RunResult) -> None:
        """Initialize comparator.

        Args:
            baseline: The baseline run to compare against
            current: The current run result
        """
        self.baseline = baseline
        self.current = current

    def compare(self) -> DiffResult:
        """Compare current run against baseline.

        Returns:
            DiffResult with all detected changes
        """
        entries = self._compare_cases()
        metric_changes = self._compare_metrics()
        summary = self._generate_summary(entries, metric_changes)

        return DiffResult(
            bundle_name=self.current.bundle_name,
            baseline_timestamp=self.baseline.timestamp.isoformat(),
            current_timestamp=self.current.completed_at.isoformat()
            if self.current.completed_at
            else "",
            entries=entries,
            metric_changes=metric_changes,
            summary=summary,
        )

    def _compare_cases(self) -> list[DiffEntry]:
        """Compare individual case results.

        Returns:
            List of DiffEntry for each case
        """
        # Build baseline lookup by case_id
        baseline_lookup = {cr.case_id: cr for cr in self.baseline.case_results}

        entries: list[DiffEntry] = []

        # Compare current cases against baseline
        for case_result in self.current.case_results:
            case_id = case_result.case_id
            current_passed = case_result.passed
            baseline_case = baseline_lookup.get(case_id)

            if baseline_case is None:
                # New case not in baseline - treat as passed (new test)
                continue

            baseline_passed = baseline_case.passed

            # Determine diff type
            diff_type = self._get_diff_type(baseline_passed, current_passed)

            entry = DiffEntry(
                case_id=case_id,
                diff_type=diff_type,
                baseline_passed=baseline_passed,
                current_passed=current_passed,
                baseline_output=baseline_case.output,
                current_output=case_result.output,
                error=case_result.error,
            )
            entries.append(entry)

        return entries

    def _get_diff_type(self, baseline_passed: bool, current_passed: bool) -> DiffType:
        """Determine the diff type based on pass states.

        Args:
            baseline_passed: Whether case passed in baseline
            current_passed: Whether case passed in current run

        Returns:
            Appropriate DiffType
        """
        if baseline_passed and not current_passed:
            return DiffType.NEW_FAILURE
        elif not baseline_passed and current_passed:
            return DiffType.FIXED
        else:
            return DiffType.UNCHANGED

    def _compare_metrics(self) -> list[MetricChange]:
        """Compare aggregate metrics between baseline and current.

        Returns:
            List of MetricChange for each metric
        """
        changes: list[MetricChange] = []

        # Get current metrics
        current_summary = self.current.summary
        baseline_metrics = self.baseline.metrics

        # Pass rate
        current_pass_rate = current_summary.get("pass_rate", 0.0)
        baseline_pass_rate = baseline_metrics.pass_rate

        if baseline_pass_rate != current_pass_rate:
            changes.append(
                MetricChange(
                    metric="pass_rate",
                    baseline_value=baseline_pass_rate,
                    current_value=current_pass_rate,
                    absolute_change=current_pass_rate - baseline_pass_rate,
                )
            )

        # Total cases
        current_total = current_summary.get("total_cases", 0)
        baseline_total = baseline_metrics.total_cases

        if baseline_total != current_total:
            changes.append(
                MetricChange(
                    metric="total_cases",
                    baseline_value=float(baseline_total),
                    current_value=float(current_total),
                    absolute_change=float(current_total - baseline_total),
                )
            )

        # Passed cases
        current_passed = current_summary.get("passed_cases", 0)
        baseline_passed = baseline_metrics.passed_cases

        if baseline_passed != current_passed:
            changes.append(
                MetricChange(
                    metric="passed_cases",
                    baseline_value=float(baseline_passed),
                    current_value=float(current_passed),
                    absolute_change=float(current_passed - baseline_passed),
                )
            )

        # Failed cases
        current_failed = current_summary.get("failed_cases", 0)
        baseline_failed = baseline_metrics.failed_cases

        if baseline_failed != current_failed:
            changes.append(
                MetricChange(
                    metric="failed_cases",
                    baseline_value=float(baseline_failed),
                    current_value=float(current_failed),
                    absolute_change=float(current_failed - baseline_failed),
                )
            )

        # Duration
        current_duration = current_summary.get("duration_ms", 0.0)
        baseline_duration = baseline_metrics.duration_ms

        if baseline_duration != current_duration:
            changes.append(
                MetricChange(
                    metric="duration_ms",
                    baseline_value=baseline_duration,
                    current_value=current_duration,
                    absolute_change=current_duration - baseline_duration,
                )
            )

        # Average latency (if available)
        baseline_avg_latency = self._calculate_avg_latency(self.baseline)
        current_avg_latency = self._calculate_avg_latency(self.current)

        if baseline_avg_latency is not None and current_avg_latency is not None:
            if baseline_avg_latency != current_avg_latency:
                changes.append(
                    MetricChange(
                        metric="avg_latency_ms",
                        baseline_value=baseline_avg_latency,
                        current_value=current_avg_latency,
                        absolute_change=current_avg_latency - baseline_avg_latency,
                    )
                )

        return changes

    def _calculate_avg_latency(self, run: BaselineRun | RunResult) -> float | None:
        """Calculate average latency from a run.

        Args:
            run: The run to calculate from

        Returns:
            Average latency in milliseconds, or None if not available
        """
        if isinstance(run, BaselineRun):
            latencies = [
                cr.latency_ms
                for cr in run.case_results
                if cr.latency_ms is not None
            ]
        else:
            latencies = [
                cr.latency_ms
                for cr in run.case_results
                if cr.latency_ms is not None
            ]

        if not latencies:
            return None

        return sum(latencies) / len(latencies)

    def _generate_summary(
        self,
        entries: list[DiffEntry],
        metric_changes: list[MetricChange],
    ) -> dict[str, Any]:
        """Generate summary statistics.

        Args:
            entries: Case-level differences
            metric_changes: Metric changes

        Returns:
            Summary dictionary
        """
        new_failures = sum(1 for e in entries if e.diff_type == DiffType.NEW_FAILURE)
        fixed = sum(1 for e in entries if e.diff_type == DiffType.FIXED)
        unchanged = sum(1 for e in entries if e.diff_type == DiffType.UNCHANGED)

        pass_rate_change = None
        for mc in metric_changes:
            if mc.metric == "pass_rate":
                pass_rate_change = mc.absolute_change
                break

        return {
            "total_cases_compared": len(entries),
            "new_failures": new_failures,
            "fixed_cases": fixed,
            "unchanged_cases": unchanged,
            "has_regressions": new_failures > 0,
            "has_improvements": fixed > 0,
            "pass_rate_change": pass_rate_change,
            "metric_changes_count": len(metric_changes),
        }
