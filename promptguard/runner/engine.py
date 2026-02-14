"""Main execution engine for running Behavior Bundles."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from promptguard.bundle.dataset import DatasetCase
from promptguard.bundle.loader import BundleLoader
from promptguard.providers import get_provider
from promptguard.providers.base import LLMProvider
from promptguard.runner.evaluator import Evaluator, EvaluationResult
from promptguard.runner.thresholds import ThresholdEvaluator, ThresholdEvaluation


@dataclass
class CaseResult:
    """Result of running a single test case.

    Attributes:
        case_id: Identifier for the test case
        passed: Whether the case passed evaluation
        prompt: Rendered prompt sent to LLM
        output: Raw output from LLM
        evaluation: Detailed evaluation result
        expected_output: Expected output (if provided)
        latency_ms: Request latency in milliseconds
        error: Error message if execution failed
    """

    case_id: str
    passed: bool
    prompt: str = ""
    output: str = ""
    evaluation: EvaluationResult | None = None
    expected_output: Any = None
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of running a complete bundle.

    Attributes:
        bundle_name: Name of the bundle
        passed: Whether all thresholds passed
        case_results: Individual case results
        threshold_evaluation: Threshold evaluation details
        started_at: Run start timestamp
        completed_at: Run completion timestamp
        duration_ms: Total run duration
        provider: Provider name used
        model: Model name used
    """

    bundle_name: str
    passed: bool
    case_results: list[CaseResult] = field(default_factory=list)
    threshold_evaluation: ThresholdEvaluation | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_ms: float = 0.0
    provider: str = ""
    model: str = ""

    @property
    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        total = len(self.case_results)
        passed = sum(1 for r in self.case_results if r.passed)
        failed = total - passed

        return {
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "duration_ms": self.duration_ms,
        }


class Runner:
    """Execution engine for Behavior Bundles.

    Orchestrates the full execution pipeline:
    1. Load bundle configuration
    2. Initialize provider
    3. Execute test cases
    4. Evaluate outputs
    5. Apply thresholds
    """

    def __init__(
        self,
        bundle: BundleLoader,
        provider: LLMProvider | None = None,
    ) -> None:
        """Initialize runner with bundle.

        Args:
            bundle: Loaded bundle configuration
            provider: Optional provider override
        """
        self.bundle = bundle
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        """Get or create provider instance."""
        if self._provider is None:
            config = self.bundle.config
            self._provider = get_provider(
                config.provider.name,
                model=config.provider.model,
                temperature=config.provider.temperature,
                max_tokens=config.provider.max_tokens,
            )
        return self._provider

    async def run(self) -> RunResult:
        """Execute the bundle and return results.

        Returns:
            RunResult with complete execution details
        """
        config = self.bundle.config
        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        # Initialize result
        result = RunResult(
            bundle_name=config.name,
            passed=False,
            started_at=started_at,
            provider=self.provider.name,
            model=self.provider.model,
        )

        # Create evaluator
        evaluator = Evaluator(config.output_contract, self.bundle.bundle_dir)

        # Execute cases
        dataset = self.bundle.dataset
        if config.parallel:
            case_results = await self._run_parallel(dataset.cases, evaluator)
        else:
            case_results = await self._run_sequential(dataset.cases, evaluator)

        result.case_results = case_results

        # Evaluate thresholds
        threshold_evaluator = ThresholdEvaluator(config.thresholds)
        threshold_eval = threshold_evaluator.evaluate(case_results)
        result.threshold_evaluation = threshold_eval
        result.passed = threshold_eval.passed

        # Finalize timing
        result.completed_at = datetime.now(timezone.utc)
        result.duration_ms = (time.perf_counter() - start_time) * 1000

        return result

    async def _run_sequential(
        self,
        cases: list[DatasetCase],
        evaluator: Evaluator,
    ) -> list[CaseResult]:
        """Run cases sequentially.

        Args:
            cases: Test cases to run
            evaluator: Output evaluator

        Returns:
            List of case results
        """
        results: list[CaseResult] = []
        for case in cases:
            result = await self._run_case(case, evaluator)
            results.append(result)
        return results

    async def _run_parallel(
        self,
        cases: list[DatasetCase],
        evaluator: Evaluator,
    ) -> list[CaseResult]:
        """Run cases in parallel.

        Args:
            cases: Test cases to run
            evaluator: Output evaluator

        Returns:
            List of case results
        """
        tasks = [self._run_case(case, evaluator) for case in cases]
        return await asyncio.gather(*tasks)

    async def _run_case(
        self,
        case: DatasetCase,
        evaluator: Evaluator,
    ) -> CaseResult:
        """Run a single test case.

        Args:
            case: Test case to run
            evaluator: Output evaluator

        Returns:
            CaseResult with execution details
        """
        config = self.bundle.config
        assert case.id is not None  # Should be set by DatasetCase.__post_init__

        # Render prompt with case input
        try:
            prompt = self.bundle.render_prompt(case.input)
        except Exception as e:
            return CaseResult(
                case_id=case.id,
                passed=False,
                error=f"Prompt rendering failed: {e}",
                expected_output=case.expected_output,
                metadata=case.metadata,
            )

        # Call provider with retries
        output = ""
        latency_ms: float | None = None
        error: str | None = None

        for attempt in range(config.retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.provider.generate(prompt),
                    timeout=config.timeout,
                )
                output = response.content
                latency_ms = response.latency_ms
                error = None
                break
            except asyncio.TimeoutError:
                error = f"Timeout after {config.timeout}s"
            except Exception as e:
                error = str(e)

            # Wait before retry
            if attempt < config.retries:
                await asyncio.sleep(1.0 * (attempt + 1))

        # If we have an error, return failed result
        if error:
            return CaseResult(
                case_id=case.id,
                passed=False,
                prompt=prompt,
                error=error,
                expected_output=case.expected_output,
                latency_ms=latency_ms,
                metadata=case.metadata,
            )

        # Evaluate output
        evaluation = evaluator.evaluate(output)

        return CaseResult(
            case_id=case.id,
            passed=evaluation.passed,
            prompt=prompt,
            output=output,
            evaluation=evaluation,
            expected_output=case.expected_output,
            latency_ms=latency_ms,
            metadata=case.metadata,
        )
