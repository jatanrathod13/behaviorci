"""Diff reporter for comparing test results between runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from promptguard.reporters.base import Reporter
from promptguard.reporters.registry import register_reporter
from promptguard.runner.engine import RunResult, CaseResult


@dataclass
class CaseDiff:
    """Represents the diff for a single test case."""

    case_id: str
    baseline_passed: bool | None
    current_passed: bool | None
    baseline_latency_ms: float | None
    current_latency_ms: float | None
    baseline_error: str | None
    current_error: str | None

    @property
    def status(self) -> str:
        """Get the diff status."""
        if self.baseline_passed is None:
            return "new"
        if self.current_passed is None:
            return "removed"
        if self.baseline_passed != self.current_passed:
            return "changed"
        return "unchanged"


@dataclass
class MetricDiff:
    """Represents diff for a metric."""

    metric: str
    baseline_value: float | None
    current_value: float | None
    change: float | None = None
    change_percent: float | None = None

    @property
    def status(self) -> str:
        """Get the diff status."""
        if self.baseline_value is None:
            return "new"
        if self.current_value is None:
            return "removed"
        if abs(self.change or 0) < 0.001:
            return "unchanged"
        return "changed"


@dataclass
class DiffResult:
    """Result of comparing two RunResult objects."""

    baseline: RunResult | None
    current: RunResult
    case_diffs: list[CaseDiff] = field(default_factory=list)
    metric_diffs: list[MetricDiff] = field(default_factory=list)
    new_failures: int = 0
    fixed_failures: int = 0
    new_passes: int = 0
    lost_passes: int = 0
    total_changes: int = 0


class DiffReporter(Reporter):
    """Reporter for diffing test results between runs.

    Compares a baseline run with a current run and produces
    diff reports showing what changed.
    """

    name: str = "diff"

    def emit(self, result: RunResult, verbose: bool = False) -> str:
        """Generate diff report.

        Note: This base emit method requires a DiffResult, not RunResult.
        Use generate_console_report(), generate_json_report(), or
        generate_markdown_report() directly with DiffResult.

        Args:
            result: Not used for diff reporter
            verbose: Not used for diff reporter

        Returns:
            Empty string - use specific generate methods
        """
        return ""

    def compute_diff(
        self,
        baseline: RunResult | None,
        current: RunResult,
    ) -> DiffResult:
        """Compute diff between baseline and current run results.

        Args:
            baseline: Previous run result (can be None for first run)
            current: Current run result

        Returns:
            DiffResult with computed differences
        """
        diff_result = DiffResult(baseline=baseline, current=current)

        if baseline is None:
            # First run - all cases are new
            for case in current.case_results:
                diff_result.case_diffs.append(
                    CaseDiff(
                        case_id=case.case_id,
                        baseline_passed=None,
                        current_passed=case.passed,
                        baseline_latency_ms=None,
                        current_latency_ms=case.latency_ms,
                        baseline_error=None,
                        current_error=case.error,
                    )
                )
            diff_result.total_changes = len(current.case_results)
            diff_result.new_failures = sum(1 for c in current.case_results if not c.passed)
        else:
            # Compare with baseline
            baseline_cases = {c.case_id: c for c in baseline.case_results}
            current_cases = {c.case_id: c for c in current.case_results}

            all_ids = set(baseline_cases.keys()) | set(current_cases.keys())

            for case_id in sorted(all_ids):
                base = baseline_cases.get(case_id)
                curr = current_cases.get(case_id)

                case_diff = CaseDiff(
                    case_id=case_id,
                    baseline_passed=base.passed if base else None,
                    current_passed=curr.passed if curr else None,
                    baseline_latency_ms=base.latency_ms if base else None,
                    current_latency_ms=curr.latency_ms if curr else None,
                    baseline_error=base.error if base else None,
                    current_error=curr.error if curr else None,
                )
                diff_result.case_diffs.append(case_diff)

                # Track status changes
                if base is None:
                    diff_result.new_failures += 1 if not curr.passed else 0
                elif curr is None:
                    diff_result.fixed_failures += 1 if base.passed else 0
                elif base.passed != curr.passed:
                    if curr.passed:
                        diff_result.fixed_failures += 1
                        diff_result.new_passes += 1
                    else:
                        diff_result.new_failures += 1
                        diff_result.lost_passes += 1

            # Compute metric diffs
            base_summary = baseline.summary
            curr_summary = current.summary

            metrics = [
                ("total_cases", base_summary.get("total_cases"), curr_summary.get("total_cases")),
                ("passed_cases", base_summary.get("passed_cases"), curr_summary.get("passed_cases")),
                ("failed_cases", base_summary.get("failed_cases"), curr_summary.get("failed_cases")),
                ("pass_rate", base_summary.get("pass_rate"), curr_summary.get("pass_rate")),
            ]

            for metric, base_val, curr_val in metrics:
                change = None
                change_pct = None
                if base_val is not None and curr_val is not None:
                    change = curr_val - base_val
                    if base_val != 0:
                        change_pct = (change / base_val) * 100

                diff_result.metric_diffs.append(
                    MetricDiff(
                        metric=metric,
                        baseline_value=base_val,
                        current_value=curr_val,
                        change=change,
                        change_percent=change_pct,
                    )
                )

            diff_result.total_changes = (
                diff_result.new_failures
                + diff_result.fixed_failures
                + diff_result.new_passes
                + diff_result.lost_passes
            )

        return diff_result

    def generate_console_report(self, diff_result: DiffResult) -> str:
        """Generate Rich table console output.

        Args:
            diff_result: Computed diff result

        Returns:
            Formatted string with Rich markup
        """
        output = StringIO()
        console = Console(file=output, force_terminal=True)

        console.print()
        console.print(f"[bold]Diff Report: {diff_result.current.bundle_name}[/bold]")
        console.print()

        # Summary table
        summary_table = Table(show_header=True, header_style="bold")
        summary_table.add_column("Change Type", style="cyan")
        summary_table.add_column("Count", justify="right")

        summary_table.add_row(
            "[green]Fixed Failures[/green]",
            f"[green]{diff_result.fixed_failures}[/green]",
        )
        summary_table.add_row(
            "[red]New Failures[/red]",
            f"[red]{diff_result.new_failures}[/red]",
        )
        summary_table.add_row(
            "[green]New Passes[/green]",
            f"[green]{diff_result.new_passes}[/green]",
        )
        summary_table.add_row(
            "[red]Lost Passes[/red]",
            f"[red]{diff_result.lost_passes}[/red]",
        )
        summary_table.add_row(
            "[bold]Total Changes[/bold]",
            f"[bold]{diff_result.total_changes}[/bold]",
        )

        console.print("[bold]Summary[/bold]")
        console.print(summary_table)
        console.print()

        # Metric comparison table
        if diff_result.metric_diffs:
            metric_table = Table(show_header=True, header_style="bold")
            metric_table.add_column("Metric")
            metric_table.add_column("Baseline", justify="right")
            metric_table.add_column("Current", justify="right")
            metric_table.add_column("Change", justify="right")
            metric_table.add_column("Change %", justify="right")

            for m in diff_result.metric_diffs:
                baseline_str = str(m.baseline_value) if m.baseline_value is not None else "-"
                current_str = str(m.current_value) if m.current_value is not None else "-"

                change_str = f"{m.change:+.2f}" if m.change is not None else "-"
                pct_str = f"{m.change_percent:+.1f}%" if m.change_percent is not None else "-"

                # Color based on change
                if m.metric == "pass_rate":
                    style = "[green]" if (m.change or 0) > 0 else "[red]" if (m.change or 0) < 0 else ""
                elif m.metric in ("failed_cases", "lost_passes"):
                    style = "[green]" if (m.change or 0) < 0 else "[red]" if (m.change or 0) > 0 else ""
                else:
                    style = ""

                metric_table.add_row(
                    m.metric,
                    baseline_str,
                    current_str,
                    f"{style}{change_str}[/]" if style else change_str,
                    f"{style}{pct_str}[/]" if style else pct_str,
                )

            console.print("[bold]Metrics Comparison[/bold]")
            console.print(metric_table)
            console.print()

        # Detailed case diffs
        if diff_result.case_diffs:
            console.print("[bold]Case Changes[/bold]")

            changed_cases = [c for c in diff_result.case_diffs if c.status != "unchanged"]
            if changed_cases:
                case_table = Table(show_header=True, header_style="bold")
                case_table.add_column("Case ID")
                case_table.add_column("Baseline", justify="center")
                case_table.add_column("Current", justify="center")
                case_table.add_column("Status", justify="center")
                case_table.add_column("Latency Change")

                for c in changed_cases:
                    base_status = "PASS" if c.baseline_passed else ("FAIL" if c.baseline_passed is False else "-")
                    curr_status = "PASS" if c.current_passed else ("FAIL" if c.current_passed is False else "-")

                    # Status icon
                    if c.status == "new":
                        status_icon = "[cyan]NEW[/cyan]"
                    elif c.status == "removed":
                        status_icon = "[dim]REM[/dim]"
                    elif c.current_passed:
                        status_icon = "[green]FIXED[/green]"
                    else:
                        status_icon = "[red]BROKE[/red]"

                    # Latency change
                    if c.baseline_latency_ms and c.current_latency_ms:
                        lat_change = c.current_latency_ms - c.baseline_latency_ms
                        lat_str = f"{lat_change:+.0f}ms"
                        lat_style = "[green]" if lat_change < 0 else "[red]" if lat_change > 0 else ""
                        lat_change_str = f"{lat_style}{lat_str}[/]" if lat_style else lat_str
                    else:
                        lat_change_str = "-"

                    case_table.add_row(
                        c.case_id,
                        base_status,
                        curr_status,
                        status_icon,
                        lat_change_str,
                    )

                console.print(case_table)
            else:
                console.print("[dim]No case changes[/dim]")

        return output.getvalue()

    def generate_json_report(self, diff_result: DiffResult) -> str:
        """Generate JSON output.

        Args:
            diff_result: Computed diff result

        Returns:
            JSON string
        """
        data = self._build_report(diff_result)
        return json.dumps(data, indent=2, default=str)

    def generate_markdown_report(self, diff_result: DiffResult) -> str:
        """Generate Markdown output for PR comments.

        Args:
            diff_result: Computed diff result

        Returns:
            Markdown string
        """
        lines: list[str] = []

        # Header
        lines.append(f"# PromptGuard Diff Report: {diff_result.current.bundle_name}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")

        if diff_result.baseline is None:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| New Failures | {diff_result.new_failures} |")
            lines.append(f"| Total Cases | {len(diff_result.current.case_results)} |")
        else:
            lines.append("| Change Type | Count |")
            lines.append("|-------------|-------|")
            lines.append(f"| Fixed Failures | {diff_result.fixed_failures} |")
            lines.append(f"| New Failures | {diff_result.new_failures} |")
            lines.append(f"| New Passes | {diff_result.new_passes} |")
            lines.append(f"| Lost Passes | {diff_result.lost_passes} |")
            lines.append(f"| **Total Changes** | **{diff_result.total_changes}** |")

        lines.append("")

        # Metrics comparison
        if diff_result.metric_diffs:
            lines.append("## Metrics Comparison")
            lines.append("")
            lines.append("| Metric | Baseline | Current | Change | Change % |")
            lines.append("|--------|----------|---------|--------|----------|")

            for m in diff_result.metric_diffs:
                baseline_str = f"{m.baseline_value:.2f}" if m.baseline_value is not None else "-"
                current_str = f"{m.current_value:.2f}" if m.current_value is not None else "-"
                change_str = f"{m.change:+.2f}" if m.change is not None else "-"
                pct_str = f"{m.change_percent:+.1f}%" if m.change_percent is not None else "-"
                lines.append(f"| {m.metric} | {baseline_str} | {current_str} | {change_str} | {pct_str} |")

            lines.append("")

        # Case changes
        changed_cases = [c for c in diff_result.case_diffs if c.status != "unchanged"]
        if changed_cases:
            lines.append("## Case Changes")
            lines.append("")
            lines.append("| Case ID | Baseline | Current | Status |")
            lines.append("|---------|----------|---------|--------|")

            for c in changed_cases:
                base_status = "PASS" if c.baseline_passed else ("FAIL" if c.baseline_passed is False else "-")
                curr_status = "PASS" if c.current_passed else ("FAIL" if c.current_passed is False else "-")

                if c.status == "new":
                    status = "NEW"
                elif c.status == "removed":
                    status = "REM"
                elif c.current_passed:
                    status = "FIXED"
                else:
                    status = "BROKE"

                lines.append(f"| {c.case_id} | {base_status} | {curr_status} | {status} |")

            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated by PromptGuard at {diff_result.current.started_at.isoformat()}*")

        return "\n".join(lines)

    def _build_report(self, diff_result: DiffResult) -> dict[str, Any]:
        """Build JSON report data structure.

        Args:
            diff_result: Computed diff result

        Returns:
            Report dictionary
        """
        report: dict[str, Any] = {
            "bundle": diff_result.current.bundle_name,
            "summary": {
                "new_failures": diff_result.new_failures,
                "fixed_failures": diff_result.fixed_failures,
                "new_passes": diff_result.new_passes,
                "lost_passes": diff_result.lost_passes,
                "total_changes": diff_result.total_changes,
            },
            "baseline": None,
            "current": {
                "bundle": diff_result.current.bundle_name,
                "passed": diff_result.current.passed,
                "summary": diff_result.current.summary,
                "provider": diff_result.current.provider,
                "model": diff_result.current.model,
                "duration_ms": diff_result.current.duration_ms,
            },
            "metrics": [],
            "cases": [],
        }

        if diff_result.baseline:
            report["baseline"] = {
                "bundle": diff_result.baseline.bundle_name,
                "passed": diff_result.baseline.passed,
                "summary": diff_result.baseline.summary,
                "provider": diff_result.baseline.provider,
                "model": diff_result.baseline.model,
                "duration_ms": diff_result.baseline.duration_ms,
            }

        for m in diff_result.metric_diffs:
            report["metrics"].append({
                "metric": m.metric,
                "baseline": m.baseline_value,
                "current": m.current_value,
                "change": m.change,
                "change_percent": m.change_percent,
            })

        for c in diff_result.case_diffs:
            if c.status != "unchanged":
                report["cases"].append({
                    "case_id": c.case_id,
                    "status": c.status,
                    "baseline_passed": c.baseline_passed,
                    "current_passed": c.current_passed,
                    "baseline_latency_ms": c.baseline_latency_ms,
                    "current_latency_ms": c.current_latency_ms,
                    "baseline_error": c.baseline_error,
                    "current_error": c.current_error,
                })

        return report


# Register the reporter
register_reporter("diff")(DiffReporter)
