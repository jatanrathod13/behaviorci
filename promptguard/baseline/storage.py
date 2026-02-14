"""JSON file storage backend for baselines."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from promptguard.baseline.models import BaselineRun, BaselineStorage


class BaselineStorageBackend:
    """JSON file storage for baseline runs.

    Stores baselines in a JSON file with atomic writes for safety.
    """

    def __init__(self, storage_dir: str | Path) -> None:
        """Initialize storage backend.

        Args:
            storage_dir: Directory to store baseline files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_storage_path(self, bundle_name: str) -> Path:
        """Get the storage file path for a bundle.

        Args:
            bundle_name: Name of the bundle

        Returns:
            Path to the storage file
        """
        # Sanitize bundle name for file system
        safe_name = bundle_name.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_name}_baseline.json"

    def save(self, storage: BaselineStorage) -> None:
        """Save baseline storage to JSON file.

        Uses atomic write (write to temp then rename) for safety.

        Args:
            storage: BaselineStorage to save
        """
        path = self._get_storage_path(storage.bundle_name)
        temp_path = path.with_suffix(".tmp")

        # Write to temp file first
        with open(temp_path, "w") as f:
            json.dump(storage.to_dict(), f, indent=2)

        # Atomic rename
        shutil.move(temp_path, path)

    def load(self, bundle_name: str) -> BaselineStorage | None:
        """Load baseline storage for a bundle.

        Args:
            bundle_name: Name of the bundle

        Returns:
            BaselineStorage if exists, None otherwise
        """
        # First check new format: .promptguard/baselines/{bundle_name}/*.json
        new_format_dir = self.storage_dir / bundle_name
        if new_format_dir.exists() and new_format_dir.is_dir():
            json_files = sorted(new_format_dir.glob("*.json"), reverse=True)
            if json_files:
                with open(json_files[0]) as f:
                    data = json.load(f)
                return BaselineStorage.from_dict(data)

        # Fallback to old format: .promptguard/baselines/{name}_baseline.json
        path = self._get_storage_path(bundle_name)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return BaselineStorage.from_dict(data)

    def delete(self, bundle_name: str) -> bool:
        """Delete baseline storage for a bundle.

        Args:
            bundle_name: Name of the bundle

        Returns:
            True if deleted, False if not found
        """
        path = self._get_storage_path(bundle_name)
        if not path.exists():
            return False

        path.unlink()
        return True

    def exists(self, bundle_name: str) -> bool:
        """Check if baseline storage exists for a bundle.

        Args:
            bundle_name: Name of the bundle

        Returns:
            True if baseline exists
        """
        path = self._get_storage_path(bundle_name)
        return path.exists()

    def list_bundles(self) -> list[str]:
        """List all bundles with stored baselines.

        Returns:
            List of bundle names
        """
        bundles = []
        for path in self.storage_dir.glob("*_baseline.json"):
            # Remove suffix to get bundle name
            name = path.stem.replace("_baseline", "")
            bundles.append(name)
        return bundles


def create_baseline_from_run_result(
    bundle_name: str,
    run_result: Any,
) -> BaselineRun:
    """Create a BaselineRun from a RunResult.

    This is a convenience function to convert engine.RunResult
    to baseline.BaselineRun for storage.

    Args:
        bundle_name: Name of the bundle
        run_result: RunResult from the engine

    Returns:
        BaselineRun ready for storage
    """
    # Import here to avoid circular imports
    from promptguard.baseline.models import (
        BaselineCaseResult,
        BaselineRunMetrics,
        BaselineThresholdEvaluation,
        BaselineThresholdResult,
    )

    # Convert case results
    case_results = []
    for cr in run_result.case_results:
        case_results.append(
            BaselineCaseResult(
                case_id=cr.case_id,
                passed=cr.passed,
                output=cr.output,
                error=cr.error,
            )
        )

    # Convert threshold evaluation
    threshold_evaluation = None
    if run_result.threshold_evaluation:
        te = run_result.threshold_evaluation
        results = []
        for tr in te.results:
            results.append(
                BaselineThresholdResult(
                    metric=tr.metric,
                    passed=tr.passed,
                    actual_value=tr.actual_value,
                    expected_value=tr.expected_value,
                    operator=tr.operator,
                )
            )
        threshold_evaluation = BaselineThresholdEvaluation(
            passed=te.passed,
            results=results,
            metrics=te.metrics,
        )

    # Get summary from run result
    summary = run_result.summary

    return BaselineRun(
        bundle_name=bundle_name,
        timestamp=run_result.completed_at,
        metrics=BaselineRunMetrics(
            total_cases=summary["total_cases"],
            passed_cases=summary["passed_cases"],
            failed_cases=summary["failed_cases"],
            pass_rate=summary["pass_rate"],
            duration_ms=run_result.duration_ms,
        ),
        case_results=case_results,
        threshold_evaluation=threshold_evaluation,
        provider=run_result.provider,
        model=run_result.model,
    )
