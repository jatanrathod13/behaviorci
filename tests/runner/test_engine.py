"""Tests for runner engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptguard.bundle.loader import load_bundle
from promptguard.providers.mock import MockProvider
from promptguard.runner import Runner


class TestRunner:
    """Tests for Runner class."""

    @pytest.mark.asyncio
    async def test_run_with_mock_provider(self, complete_bundle: Path) -> None:
        """Test running bundle with mock provider."""
        bundle = load_bundle(complete_bundle)
        provider = MockProvider(default_response='{"answer": "test"}')
        runner = Runner(bundle, provider=provider)

        result = await runner.run()

        assert result.bundle_name == "test-bundle"
        assert len(result.case_results) == 3
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_all_cases_pass(self, complete_bundle: Path) -> None:
        """Test when all cases pass thresholds."""
        bundle = load_bundle(complete_bundle)
        provider = MockProvider(default_response='{"answer": "valid"}')
        runner = Runner(bundle, provider=provider)

        result = await runner.run()

        # With mock provider returning valid schema, should pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_threshold_failure(self, complete_bundle: Path) -> None:
        """Test when thresholds fail."""
        bundle = load_bundle(complete_bundle)
        # Return invalid JSON to fail schema validation
        provider = MockProvider(default_response="not json")
        runner = Runner(bundle, provider=provider)

        result = await runner.run()

        assert result.passed is False
        assert result.threshold_evaluation is not None

    @pytest.mark.asyncio
    async def test_provider_error_handling(self, complete_bundle: Path) -> None:
        """Test handling of provider errors."""
        bundle = load_bundle(complete_bundle)
        provider = MockProvider(fail_on=["Answer"])
        runner = Runner(bundle, provider=provider)

        result = await runner.run()

        # Should complete but with errors
        assert len(result.case_results) == 3
        error_cases = [c for c in result.case_results if c.error is not None]
        assert len(error_cases) > 0

    @pytest.mark.asyncio
    async def test_result_summary(self, complete_bundle: Path) -> None:
        """Test result summary generation."""
        bundle = load_bundle(complete_bundle)
        provider = MockProvider(default_response='{"answer": "test"}')
        runner = Runner(bundle, provider=provider)

        result = await runner.run()
        summary = result.summary

        assert "total_cases" in summary
        assert "passed_cases" in summary
        assert "failed_cases" in summary
        assert "pass_rate" in summary
        assert summary["total_cases"] == 3
