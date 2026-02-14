"""Pydantic models for baseline storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaselineCaseResult(BaseModel):
    """Case-level result stored in baseline.

    Attributes:
        case_id: Identifier for the test case
        passed: Whether the case passed evaluation
        output: Raw output from LLM
        error: Error message if execution failed (optional)
        latency_ms: Request latency in milliseconds (optional)
    """

    case_id: str
    passed: bool
    output: str = ""
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class BaselineThresholdResult(BaseModel):
    """Threshold result stored in baseline.

    Attributes:
        metric: Metric name (e.g., 'pass_rate')
        passed: Whether threshold was met
        actual_value: Computed metric value
        expected_value: Threshold target value
        operator: Comparison operator used
    """

    metric: str
    passed: bool
    actual_value: float
    expected_value: float
    operator: str


class BaselineThresholdEvaluation(BaseModel):
    """Complete threshold evaluation stored in baseline.

    Attributes:
        passed: Whether all thresholds passed
        results: Individual threshold results
        metrics: All computed metrics
    """

    passed: bool
    results: List[BaselineThresholdResult] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class BaselineRunMetrics(BaseModel):
    """Summary metrics for a baseline run.

    Attributes:
        total_cases: Total number of test cases
        passed_cases: Number of passed cases
        failed_cases: Number of failed cases
        pass_rate: Pass rate as fraction (0-1)
        duration_ms: Total run duration in milliseconds
    """

    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    duration_ms: float


@dataclass
class BaselineRun:
    """A stored baseline run for comparison.

    Attributes:
        bundle_name: Name of the bundle
        timestamp: When the baseline was created
        metrics: Run-level summary metrics
        case_results: Individual case results
        threshold_evaluation: Threshold evaluation details
        provider: Provider name used
        model: Model name used
    """

    bundle_name: str
    timestamp: datetime
    metrics: BaselineRunMetrics
    case_results: List[BaselineCaseResult] = field(default_factory=list)
    threshold_evaluation: Optional[BaselineThresholdEvaluation] = None
    provider: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "bundle_name": self.bundle_name,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics.model_dump(),
            "case_results": [cr.model_dump() for cr in self.case_results],
            "threshold_evaluation": self.threshold_evaluation.model_dump()
            if self.threshold_evaluation
            else None,
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaselineRun:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            bundle_name=data["bundle_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=BaselineRunMetrics(**data["metrics"]),
            case_results=[
                BaselineCaseResult(**cr) for cr in data.get("case_results", [])
            ],
            threshold_evaluation=BaselineThresholdEvaluation(**data["threshold_evaluation"])
            if data.get("threshold_evaluation")
            else None,
            provider=data.get("provider", ""),
            model=data.get("model", ""),
        )


@dataclass
class BaselineStorage:
    """Container for storing baseline runs.

    Attributes:
        bundle_name: Name of the bundle this baseline belongs to
        runs: List of baseline runs (usually ordered by timestamp)
        current_index: Index of the currently active baseline (default: -1 for latest)
    """

    bundle_name: str
    runs: List[BaselineRun] = field(default_factory=list)
    current_index: int = -1

    @property
    def current(self) -> Optional[BaselineRun]:
        """Get the current baseline run."""
        if not self.runs:
            return None
        # If current_index is -1, return the latest run
        if self.current_index < 0:
            return self.runs[-1]
        if self.current_index >= len(self.runs):
            return None
        return self.runs[self.current_index]

    def add_run(self, run: BaselineRun) -> None:
        """Add a new baseline run."""
        self.runs.append(run)
        self.current_index = len(self.runs) - 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "bundle_name": self.bundle_name,
            "runs": [run.to_dict() for run in self.runs],
            "current_index": self.current_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaselineStorage:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            bundle_name=data["bundle_name"],
            runs=[BaselineRun.from_dict(r) for r in data.get("runs", [])],
            current_index=data.get("current_index", -1),
        )
