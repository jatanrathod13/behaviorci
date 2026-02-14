"""Diff models for comparing runs against baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiffType(Enum):
    """Types of changes detected between baseline and current run."""

    NEW_FAILURE = "new_failure"
    FIXED = "fixed"
    REGRESSION = "regression"
    IMPROVED = "improved"
    METRIC_CHANGE = "metric_change"
    UNCHANGED = "unchanged"


@dataclass
class DiffEntry:
    """A single case-level difference between baseline and current.

    Attributes:
        case_id: Identifier for the test case
        diff_type: Type of change detected
        baseline_passed: Whether the case passed in baseline
        current_passed: Whether the case passed in current run
        baseline_output: Output from baseline (if applicable)
        current_output: Output from current run (if applicable)
    """

    case_id: str
    diff_type: DiffType
    baseline_passed: bool
    current_passed: bool
    baseline_output: str = ""
    current_output: str = ""
    error: str | None = None


@dataclass
class MetricChange:
    """Represents a change in a metric value.

    Attributes:
        metric: Metric name (e.g., 'pass_rate', 'avg_latency')
        baseline_value: Value in baseline
        current_value: Value in current run
        absolute_change: Absolute change (current - baseline)
        relative_change: Relative change as percentage
    """

    metric: str
    baseline_value: float
    current_value: float
    absolute_change: float
    relative_change: float | None = None

    def __post_init__(self) -> None:
        """Calculate relative change."""
        if self.baseline_value != 0:
            self.relative_change = (
                (self.current_value - self.baseline_value) / self.baseline_value
            ) * 100


@dataclass
class DiffResult:
    """Complete diff result comparing current run against baseline.

    Attributes:
        bundle_name: Name of the bundle
        baseline_timestamp: When baseline was created
        current_timestamp: When current run was created
        entries: Individual case-level differences
        metric_changes: Changes in aggregate metrics
        summary: Summary statistics
    """

    bundle_name: str
    baseline_timestamp: str
    current_timestamp: str
    entries: list[DiffEntry] = field(default_factory=list)
    metric_changes: list[MetricChange] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def new_failures(self) -> list[DiffEntry]:
        """Cases that passed in baseline but failed in current."""
        return [e for e in self.entries if e.diff_type == DiffType.NEW_FAILURE]

    @property
    def fixed_cases(self) -> list[DiffEntry]:
        """Cases that failed in baseline but passed in current."""
        return [e for e in self.entries if e.diff_type == DiffType.FIXED]

    @property
    def regressions(self) -> list[DiffEntry]:
        """Cases that passed in baseline but failed in current (alias for new_failures)."""
        return self.new_failures

    @property
    def improved_cases(self) -> list[DiffEntry]:
        """Cases that failed in baseline but passed in current (alias for fixed)."""
        return self.fixed_cases

    def has_regressions(self) -> bool:
        """Check if any regressions were detected."""
        return len(self.new_failures) > 0

    def has_improvements(self) -> bool:
        """Check if any improvements were detected."""
        return len(self.fixed_cases) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "bundle_name": self.bundle_name,
            "baseline_timestamp": self.baseline_timestamp,
            "current_timestamp": self.current_timestamp,
            "entries": [
                {
                    "case_id": e.case_id,
                    "diff_type": e.diff_type.value,
                    "baseline_passed": e.baseline_passed,
                    "current_passed": e.current_passed,
                    "baseline_output": e.baseline_output,
                    "current_output": e.current_output,
                    "error": e.error,
                }
                for e in self.entries
            ],
            "metric_changes": [
                {
                    "metric": m.metric,
                    "baseline_value": m.baseline_value,
                    "current_value": m.current_value,
                    "absolute_change": m.absolute_change,
                    "relative_change": m.relative_change,
                }
                for m in self.metric_changes
            ],
            "summary": self.summary,
        }
