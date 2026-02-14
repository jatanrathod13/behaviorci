"""Console reporter for human-readable terminal output."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from promptguard.reporters.base import Reporter
from promptguard.reporters.registry import register_reporter
from promptguard.runner.engine import RunResult


@register_reporter("console")
class ConsoleReporter(Reporter):
    """Console reporter for human-readable terminal output.

    Uses Rich for colored, formatted output that's
    easy to read in the terminal.
    """

    def emit(self, result: RunResult, verbose: bool = False) -> str:
        """Generate console report.

        Args:
            result: Run result to report
            verbose: Include case-by-case details

        Returns:
            Formatted string (Rich markup)
        """
        # Capture Rich output to string
        output = StringIO()
        console = Console(file=output, force_terminal=True) 

        # Header
        status = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
        console.print()
        console.print(
            Panel(
                f"[bold]{result.bundle_name}[/bold] — {status}",
                subtitle=f"{result.duration_ms:.0f}ms",
            )
        )

        # Summary table
        console.print()
        summary_table = Table(show_header=True, header_style="bold")
        summary_table.add_column("Metric")
        summary_table.add_column("Value", justify="right")

        summary = result.summary
        summary_table.add_row("Total Cases", str(summary["total_cases"]))
        summary_table.add_row(
            "Passed",
            f"[green]{summary['passed_cases']}[/green]",
        )
        summary_table.add_row(
            "Failed",
            f"[red]{summary['failed_cases']}[/red]" if summary["failed_cases"] > 0 else "0",
        )
        summary_table.add_row(
            "Pass Rate",
            f"{summary['pass_rate']:.1%}",
        )

        console.print(summary_table)

        # Threshold results
        if result.threshold_evaluation and result.threshold_evaluation.results:
            console.print()
            console.print("[bold]Thresholds[/bold]")

            threshold_table = Table(show_header=True)
            threshold_table.add_column("Metric")
            threshold_table.add_column("Expected")
            threshold_table.add_column("Actual", justify="right")
            threshold_table.add_column("Status", justify="center")

            for t in result.threshold_evaluation.results:
                status_icon = "[green]✓[/green]" if t.passed else "[red]✗[/red]"
                threshold_table.add_row(
                    t.metric,
                    f"{t.operator} {t.expected_value:.2f}",
                    f"{t.actual_value:.2f}",
                    status_icon,
                )

            console.print(threshold_table)

        # Failed cases (always show if any)
        failed_cases = [c for c in result.case_results if not c.passed]
        if failed_cases:
            console.print()
            console.print(f"[bold red]Failed Cases ({len(failed_cases)})[/bold red]")

            for case in failed_cases[:5]:  # Show first 5 failures
                console.print()
                console.print(f"  [dim]Case:[/dim] {case.case_id}")

                if case.error:
                    console.print(f"  [red]Error:[/red] {case.error}")

                if case.evaluation:
                    if case.evaluation.schema_errors:
                        console.print("  [red]Schema errors:[/red]")
                        for err in case.evaluation.schema_errors[:3]:
                            console.print(f"    • {err}")

                    for inv, passed in case.evaluation.invariants_passed.items():
                        if not passed:
                            err = case.evaluation.invariant_errors.get(inv, "Failed")
                            console.print(f"  [red]Invariant failed:[/red] {inv}")
                            console.print(f"    {err}")

            if len(failed_cases) > 5:
                console.print()
                console.print(f"  [dim]... and {len(failed_cases) - 5} more failures[/dim]")

        # Verbose: all cases
        if verbose:
            console.print()
            console.print("[bold]All Cases[/bold]")

            cases_table = Table(show_header=True)
            cases_table.add_column("ID")
            cases_table.add_column("Status", justify="center")
            cases_table.add_column("Latency", justify="right")
            cases_table.add_column("Notes")

            for case in result.case_results:
                status_icon = "[green]✓[/green]" if case.passed else "[red]✗[/red]"
                latency = f"{case.latency_ms:.0f}ms" if case.latency_ms else "-"
                notes = case.error or ""
                if len(notes) > 40:
                    notes = notes[:40] + "..."

                cases_table.add_row(case.case_id, status_icon, latency, notes)

            console.print(cases_table)

        return output.getvalue()
