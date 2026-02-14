"""BehaviorCI CLI - Command-line interface for LLM behavior testing."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from behaviorci import __version__
from behaviorci.agent.loader import load_agent_bundle
from behaviorci.agent.models import AgentRunResult
from behaviorci.agent.runner import AgentRunner
from behaviorci.bundle.loader import load_bundle
from behaviorci.exceptions import BehaviorCIError
from behaviorci.providers import get_provider
from behaviorci.reporters import get_reporter
from behaviorci.runner import Runner

console = Console()
error_console = Console(stderr=True)


@click.group()
@click.version_option(version=__version__, prog_name="behaviorci")
def main() -> None:
    """BehaviorCI - CI/CD for LLM behavior.

    Prompts don't ship until behavior passes tests.
    """
    pass


@main.command()
@click.argument("path", type=click.Path(), default="bundles/example")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def init(path: str, force: bool) -> None:
    """Initialize a new Behavior Bundle.

    Creates example bundle files at the specified PATH.
    """
    bundle_dir = Path(path)

    if bundle_dir.exists() and not force:
        if any(bundle_dir.iterdir()):
            error_console.print(
                f"[red]Directory '{path}' is not empty. Use --force to overwrite.[/red]"
            )
            sys.exit(1)

    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Create bundle.yaml
    bundle_yaml = bundle_dir / "bundle.yaml"
    bundle_yaml.write_text(EXAMPLE_BUNDLE_YAML)

    # Create prompt.md
    prompt_md = bundle_dir / "prompt.md"
    prompt_md.write_text(EXAMPLE_PROMPT)

    # Create dataset.jsonl
    dataset_jsonl = bundle_dir / "dataset.jsonl"
    dataset_jsonl.write_text(EXAMPLE_DATASET)

    # Create schema.json
    schema_json = bundle_dir / "schema.json"
    schema_json.write_text(EXAMPLE_SCHEMA)

    console.print(f"[green]✓[/green] Created bundle at [bold]{path}/[/bold]")
    console.print()
    console.print("Files created:")
    console.print(f"  • {bundle_yaml}")
    console.print(f"  • {prompt_md}")
    console.print(f"  • {dataset_jsonl}")
    console.print(f"  • {schema_json}")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Review and customize the bundle")
    console.print(f"  2. Validate: [bold]behaviorci validate {path}/bundle.yaml[/bold]")
    console.print(f"  3. Run: [bold]behaviorci run {path}/bundle.yaml[/bold]")


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
def validate(bundle_path: str) -> None:
    """Validate a Behavior Bundle configuration.

    Checks that the bundle.yaml is valid and all referenced files exist.
    """
    try:
        bundle = load_bundle(bundle_path)
        config = bundle.config

        console.print(f"[green]✓[/green] Bundle is valid: [bold]{config.name}[/bold]")
        console.print()

        # Show summary
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value")

        table.add_row("Name", config.name)
        table.add_row("Version", config.version)
        table.add_row("Provider", config.provider.name)
        table.add_row("Model", config.provider.model or "(default)")
        table.add_row("Dataset", config.dataset_path)
        table.add_row("Cases", str(len(bundle.dataset)))
        table.add_row("Thresholds", str(len(config.thresholds)))

        console.print(table)

    except BehaviorCIError as e:
        error_console.print(f"[red]✗ Validation failed:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option(
    "--provider", "-p", help="Override provider (openai, anthropic, mock)"
)
@click.option("--model", "-m", help="Override model")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["console", "json", "markdown"]),
    default="console",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), help="Write report to file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
def run(
    bundle_path: str,
    provider: str | None,
    model: str | None,
    output_format: str,
    output: str | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """Run a Behavior Bundle and evaluate results.

    Executes all test cases, validates outputs against contracts,
    and checks thresholds. Exit code is non-zero if thresholds fail.
    """
    try:
        bundle = load_bundle(bundle_path)
        config = bundle.config

        if not quiet:
            console.print(
                f"Running bundle: [bold]{config.name}[/bold] ({len(bundle.dataset)} cases)"
            )

        # Create provider override if specified
        provider_instance = None
        if provider:
            provider_instance = get_provider(
                provider,
                model=model or config.provider.model,
                temperature=config.provider.temperature,
                max_tokens=config.provider.max_tokens,
            )

        # Run the bundle
        runner = Runner(bundle, provider=provider_instance)
        result = asyncio.run(runner.run())

        # Generate report
        reporter = get_reporter(output_format)
        report = reporter.emit(result, verbose=verbose)

        # Output report
        if output:
            Path(output).write_text(report)
            if not quiet:
                console.print(f"Report written to: {output}")
        else:
            if output_format == "console":
                # Console reporter already prints
                console.print(report)
            else:
                # JSON/Markdown go to stdout
                print(report)

        # Exit with appropriate code
        if result.passed:
            if not quiet:
                console.print()
                console.print("[green]✓ All thresholds passed[/green]")
            sys.exit(0)
        else:
            if not quiet:
                console.print()
                console.print("[red]✗ Thresholds failed[/red]")
            sys.exit(1)

    except BehaviorCIError as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(2)


# =============================================================================
# Baseline Commands
# =============================================================================


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option(
    "--provider", "-p", help="Override provider (openai, anthropic, mock)"
)
@click.option("--model", "-m", help="Override model")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
def promote(
    bundle_path: str,
    provider: str | None,
    model: str | None,
    quiet: bool,
) -> None:
    """Run a bundle and save results as a baseline.

    Executes all test cases and saves the results to .behaviorci/baselines/
    for future comparison with subsequent runs.
    """
    import time
    from datetime import datetime, timezone

    from behaviorci.bundle.loader import load_bundle
    from behaviorci.runner import Runner
    from behaviorci.providers import get_provider
    from behaviorci.baseline.models import (
        BaselineCaseResult,
        BaselineRun,
        BaselineRunMetrics,
        BaselineThresholdEvaluation,
        BaselineThresholdResult,
    )

    try:
        bundle = load_bundle(bundle_path)
        config = bundle.config
        bundle_name = config.name

        if not quiet:
            console.print(
                f"Running bundle: [bold]{bundle_name}[/bold] ({len(bundle.dataset)} cases)"
            )

        # Create provider override if specified
        provider_instance = None
        if provider:
            provider_instance = get_provider(
                provider,
                model=model or config.provider.model,
                temperature=config.provider.temperature,
                max_tokens=config.provider.max_tokens,
            )

        # Run the bundle
        runner = Runner(bundle, provider=provider_instance)
        result = asyncio.run(runner.run())

        # Convert RunResult to BaselineRun
        summary = result.summary

        # Convert threshold evaluation
        threshold_eval = None
        if result.threshold_evaluation:
            threshold_results = []
            metrics = {}
            for metric_name, metric_value in result.threshold_evaluation.metrics.items():
                metrics[metric_name] = metric_value
                # Find corresponding threshold
                for thresh in config.thresholds:
                    if thresh.metric == metric_name:
                        threshold_results.append(
                            BaselineThresholdResult(
                                metric=metric_name,
                                passed=metric_value >= thresh.value if thresh.operator == ">=" else True,
                                actual_value=metric_value,
                                expected_value=thresh.value,
                                operator=thresh.operator,
                            )
                        )
                        break
            threshold_eval = BaselineThresholdEvaluation(
                passed=result.threshold_evaluation.passed,
                results=threshold_results,
                metrics=metrics,
            )

        # Convert case results
        case_results = [
            BaselineCaseResult(
                case_id=cr.case_id,
                passed=cr.passed,
                output=cr.output,
                error=cr.error,
            )
            for cr in result.case_results
        ]

        # Create baseline run
        baseline_run = BaselineRun(
            bundle_name=bundle_name,
            timestamp=datetime.now(timezone.utc),
            metrics=BaselineRunMetrics(
                total_cases=summary["total_cases"],
                passed_cases=summary["passed_cases"],
                failed_cases=summary["failed_cases"],
                pass_rate=summary["pass_rate"],
                duration_ms=summary["duration_ms"],
            ),
            case_results=case_results,
            threshold_evaluation=threshold_eval,
            provider=result.provider,
            model=result.model,
        )

        # Save baseline to file
        from behaviorci.baseline.models import BaselineStorage

        timestamp_str = baseline_run.timestamp.strftime("%Y%m%d_%H%M%S")
        baseline_dir = Path(".behaviorci") / "baselines" / bundle_name
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baseline_dir / f"{timestamp_str}.json"

        # Store in BaselineStorage and save
        import json
        storage = BaselineStorage(bundle_name=bundle_name, runs=[baseline_run])
        baseline_path.write_text(json.dumps(storage.to_dict(), indent=2))

        if not quiet:
            console.print()
            console.print(f"[green]✓[/green] Baseline saved: [bold]{baseline_path}[/bold]")
            console.print(f"  Pass rate: {summary['pass_rate']:.1%} ({summary['passed_cases']}/{summary['total_cases']})")

    except BehaviorCIError as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(2)


# =============================================================================
# Agent Commands
# =============================================================================


@main.group()
def agent() -> None:
    """Agent bundle commands for AI agent testing."""
    pass


@agent.command("init")
@click.argument("path", type=click.Path(), default="agents/example")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def agent_init(path: str, force: bool) -> None:
    """Initialize a new Agent Bundle.

    Creates example agent bundle files at the specified PATH.
    """
    bundle_dir = Path(path)

    if bundle_dir.exists() and not force:
        if any(bundle_dir.iterdir()):
            error_console.print(
                f"[red]Directory '{path}' is not empty. Use --force to overwrite.[/red]"
            )
            sys.exit(1)

    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Create agent.yaml
    agent_yaml = bundle_dir / "agent.yaml"
    agent_yaml.write_text(EXAMPLE_AGENT_YAML)

    # Create system-prompt.md
    system_prompt_md = bundle_dir / "system-prompt.md"
    system_prompt_md.write_text(EXAMPLE_AGENT_PROMPT)

    # Create tasks.jsonl
    tasks_jsonl = bundle_dir / "tasks.jsonl"
    tasks_jsonl.write_text(EXAMPLE_AGENT_TASKS)

    # Create evaluation.yaml
    eval_yaml = bundle_dir / "evaluation.yaml"
    eval_yaml.write_text(EXAMPLE_AGENT_EVALUATION)

    # Create thresholds.yaml
    thresholds_yaml = bundle_dir / "thresholds.yaml"
    thresholds_yaml.write_text(EXAMPLE_AGENT_THRESHOLDS)

    console.print(f"[green]✓[/green] Created agent bundle at [bold]{path}/[/bold]")
    console.print()
    console.print("Files created:")
    console.print(f"  • {agent_yaml}")
    console.print(f"  • {system_prompt_md}")
    console.print(f"  • {tasks_jsonl}")
    console.print(f"  • {eval_yaml}")
    console.print(f"  • {thresholds_yaml}")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Review and customize the agent bundle")
    console.print(f"  2. Validate: [bold]behaviorci agent validate {path}/[/bold]")
    console.print(f"  3. Run: [bold]behaviorci agent run {path}/[/bold]")


@agent.command("validate")
@click.argument("bundle_path", type=click.Path(exists=True))
def agent_validate(bundle_path: str) -> None:
    """Validate an Agent Bundle configuration.

    Checks that the agent.yaml is valid and all referenced files exist.
    """
    try:
        bundle = load_agent_bundle(bundle_path)
        config = bundle.config

        console.print(f"[green]✓[/green] Agent bundle is valid: [bold]{config.name}[/bold]")
        console.print()

        # Show summary
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value")

        table.add_row("Name", config.name)
        table.add_row("Version", config.version)
        table.add_row("Agent Type", config.agent.type.value)
        table.add_row("Model", config.agent.model)
        table.add_row("Max Steps", str(config.agent.max_steps))
        table.add_row("Tools", str(len(config.tools)))
        table.add_row("Tasks", str(len(bundle.tasks)))
        table.add_row("Provider", config.provider.name)

        console.print(table)

    except BehaviorCIError as e:
        error_console.print(f"[red]✗ Validation failed:[/red] {e}")
        sys.exit(1)


@agent.command("run")
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option(
    "--provider", "-p", help="Override provider (openai, anthropic, mock)"
)
@click.option("--model", "-m", help="Override model")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["console", "json", "markdown"]),
    default="console",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), help="Write report to file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
@click.option(
    "--ci-gate",
    is_flag=True,
    help="Exit with non-zero code if thresholds fail (for CI/CD)",
)
def agent_run(
    bundle_path: str,
    provider: str | None,
    model: str | None,
    output_format: str,
    output: str | None,
    verbose: bool,
    quiet: bool,
    ci_gate: bool,
) -> None:
    """Run an Agent Bundle and evaluate results.

    Executes all tasks, evaluates agent behavior against evaluation rules,
    and checks thresholds. Exit code is non-zero if thresholds fail.
    """
    try:
        bundle = load_agent_bundle(bundle_path)
        config = bundle.config

        if not quiet:
            console.print(
                f"Running agent bundle: [bold]{config.name}[/bold] ({len(bundle.tasks)} tasks)"
            )

        # Create provider
        provider_name = provider or config.provider.name
        provider_instance = get_provider(
            provider_name,
            model=model or config.agent.model,
            temperature=config.agent.temperature,
            max_tokens=config.agent.max_tokens,
        )

        # Get system prompt
        system_prompt = bundle.system_prompt

        # Create agent runner
        runner = AgentRunner(
            provider=provider_instance,
            system_prompt=system_prompt,
            agent_config=config.agent,
            execution_config=config.execution,
            memory_config=config.memory,
        )

        # Run all tasks
        import time

        start_time = time.perf_counter()
        task_results = []

        progress = None
        task_progress = None
        if not quiet:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            )
            task_progress = progress.add_task(
                f"Running {len(bundle.tasks)} tasks...",
                total=len(bundle.tasks),
            )
            progress.start()

        for task in bundle.tasks:
            if not quiet and progress and task_progress:
                progress.update(
                    task_progress,
                    description=f"Task: {task.task_id}",
                    advance=1,
                )

            result = asyncio.run(runner.execute_task(task))
            task_results.append(result)

        if not quiet and progress:
            progress.stop()

        total_time = time.perf_counter() - start_time

        # Evaluate results
        from behaviorci.agent.evaluator import AgentEvaluator

        evaluator = AgentEvaluator(bundle.evaluation, Path(bundle_path).parent)

        for result in task_results:
            task = next((t for t in bundle.tasks if t.task_id == result.task_id), None)
            if task:
                eval_result = evaluator.evaluate(result, task)
                result.evaluation_passed = eval_result.overall_pass
                result.evaluation_details = {
                    "tool_usage_passed": eval_result.tool_usage_passed,
                    "output_passed": eval_result.output_passed,
                    "safety_passed": eval_result.safety_passed,
                }

        # Compute metrics
        total = len(task_results)
        passed = sum(1 for r in task_results if r.success and r.evaluation_passed)
        failed = total - passed
        avg_steps = sum(r.steps_used for r in task_results) / total if total else 0
        avg_latency = sum(r.latency_ms for r in task_results) / total if total else 0
        total_tokens = sum(r.token_usage for r in task_results)

        # Build result
        run_result = AgentRunResult(
            bundle_name=config.name,
            total_tasks=total,
            passed_tasks=passed,
            failed_tasks=failed,
            task_results=task_results,
            task_success_rate=passed / total if total else 0,
            avg_steps_used=avg_steps,
            avg_latency_ms=avg_latency,
            total_token_usage=total_tokens,
        )

        # Check thresholds
        if bundle.thresholds:
            threshold_results = []
            all_passed = True

            for threshold in bundle.thresholds:
                metric_value = getattr(run_result, threshold.metric, None)
                if metric_value is None:
                    continue

                # Apply operator
                if threshold.operator == ">=":
                    passed = metric_value >= threshold.value
                elif threshold.operator == ">":
                    passed = metric_value > threshold.value
                elif threshold.operator == "<=":
                    passed = metric_value <= threshold.value
                elif threshold.operator == "<":
                    passed = metric_value < threshold.value
                elif threshold.operator == "==":
                    passed = metric_value == threshold.value

                threshold_results.append({
                    "metric": threshold.metric,
                    "operator": threshold.operator,
                    "value": threshold.value,
                    "actual": metric_value,
                    "passed": passed,
                })

                if not passed:
                    all_passed = False

            run_result.threshold_results = threshold_results
            run_result.threshold_passed = all_passed
        else:
            # No thresholds = always pass
            run_result.threshold_passed = True

        # Generate report
        if output_format == "console":
            _print_agent_console_report(run_result, verbose, quiet)
        elif output_format == "json":
            report = json.dumps(run_result.to_dict(), indent=2)
            if output:
                Path(output).write_text(report)
            else:
                print(report)
        else:
            report = _generate_agent_markdown_report(run_result)
            if output:
                Path(output).write_text(report)
            else:
                print(report)

        # Exit with appropriate code
        if ci_gate or output_format != "console":
            if run_result.threshold_passed:
                sys.exit(0)
            else:
                sys.exit(1)
        elif run_result.threshold_passed:
            if not quiet:
                console.print()
                console.print("[green]✓ All thresholds passed[/green]")
            sys.exit(0)
        else:
            if not quiet:
                console.print()
                console.print("[red]✗ Thresholds failed[/red]")
            sys.exit(1)

    except BehaviorCIError as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(2)


# =============================================================================
# Diff Commands
# =============================================================================


@main.command("diff")
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option(
    "--baseline", "-b",
    help="Baseline run ID or 'latest' (default: latest)",
    default="latest"
)
@click.option(
    "--format", "-f",
    "output_format",
    type=click.Choice(["console", "json", "markdown"]),
    default="console",
    help="Output format"
)
@click.option("--output", "-o", type=click.Path(), help="Write report to file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--ci-gate", is_flag=True, help="Exit non-zero if regressions found")
def diff(
    bundle_path: str,
    baseline: str,
    output_format: str,
    output: str | None,
    verbose: bool,
    ci_gate: bool,
) -> None:
    """Compare current run against a baseline.

    Runs the bundle and compares results against the specified baseline.
    Shows new failures, fixed failures, and metric changes.
    """
    try:
        from behaviorci.bundle.loader import BundleLoader
        from behaviorci.diff import Comparator
        from behaviorci.baseline.storage import BaselineStorage
        from pathlib import Path

        # Load bundle
        bundle = BundleLoader(Path(bundle_path))
        bundle_name = bundle.config.name

        # Get baseline
        from behaviorci.baseline.storage import BaselineStorageBackend

        storage_backend = BaselineStorageBackend(".behaviorci/baselines")
        storage = storage_backend.load(bundle_name)

        if not storage or not storage.current:
            error_console.print(f"[red]No baseline found for '{bundle_name}'[/red]")
            sys.exit(1)

        baseline_run = storage.current

        if not verbose:
            console.print(f"Comparing against baseline: {baseline_run.timestamp}")

        # Run current bundle
        from behaviorci.runner.engine import Runner
        from behaviorci.providers import get_provider

        provider = get_provider(
            bundle.config.provider.name,
            model=bundle.config.provider.model,
            temperature=bundle.config.provider.temperature,
            max_tokens=bundle.config.provider.max_tokens,
        )

        runner = Runner(bundle, provider)

        # Run and get results
        import asyncio
        current_result = asyncio.run(runner.run())

        # Compare
        comparator = Comparator(baseline_run, current_result)
        diff_result = comparator.compare()

        # Generate report
        if output_format == "console":
            _print_diff_console_report(diff_result, verbose)
        elif output_format == "json":
            report = json.dumps(diff_result.to_dict(), indent=2)
            if output:
                Path(output).write_text(report)
            else:
                print(report)
        else:
            report = _generate_diff_markdown_report(diff_result)
            if output:
                Path(output).write_text(report)
            else:
                print(report)

        # Exit code
        has_regressions = diff_result.has_regressions()
        if ci_gate or output_format != "console":
            sys.exit(1 if has_regressions else 0)
        elif has_regressions:
            console.print()
            console.print("[red]✗ Regressions detected[/red]")
            sys.exit(1)
        else:
            console.print()
            console.print("[green]✓ No regressions[/green]")
            sys.exit(0)

    except BehaviorCIError as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(2)


def _print_diff_console_report(diff_result, verbose: bool) -> None:
    """Print diff results to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Summary
    table = Table(title="Diff Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="yellow")
    table.add_column("Current", style="green")
    table.add_column("Change", style="bold")

    for change in diff_result.metric_changes:
        metric_name = change.metric
        baseline_val = change.baseline_value
        current_val = change.current_value
        diff_pct = change.relative_change or 0

        change_str = f"{diff_pct:+.1f}%"
        change_style = "green" if diff_pct >= 0 else "red"

        table.add_row(
            metric_name,
            f"{baseline_val:.2f}",
            f"{current_val:.2f}",
            f"[{change_style}]{change_str}[/{change_style}]"
        )

    console.print(table)

    # Status summary
    console.print()
    new_failures = diff_result.new_failures
    fixed_failures = diff_result.fixed_cases
    if diff_result.has_regressions():
        console.print(f"[red]✗ {len(new_failures)} new failures, {len(diff_result.regressions)} regressions[/red]")
    else:
        console.print(f"[green]✓ No regressions ({len(fixed_failures)} fixed)[/green]")

    # Details
    if verbose:
        if new_failures:
            console.print()
            console.print("[bold]New Failures:[/bold]")
            for entry in new_failures:
                console.print(f"  • {entry.case_id}")

        if fixed_failures:
            console.print()
            console.print("[bold]Fixed Failures:[/bold]")
            for entry in fixed_failures:
                console.print(f"  • {entry.case_id}")


def _generate_diff_markdown_report(diff_result) -> str:
    """Generate markdown report for diff results."""
    new_failures = diff_result.new_failures
    fixed_failures = diff_result.fixed_cases

    lines = [
        f"# Diff Results: {diff_result.bundle_name}",
        "",
        "## Summary",
        "",
        f"- **Baseline**: {diff_result.baseline_timestamp}",
        f"- **Current**: {diff_result.current_timestamp}",
        "",
    ]

    if diff_result.metric_changes:
        lines.append("## Metric Changes")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Change |")
        lines.append("|--------|----------|---------|--------|")

        for change in diff_result.metric_changes:
            metric_name = change.metric
            baseline_val = change.baseline_value
            current_val = change.current_value
            diff_pct = change.relative_change or 0
            status = "✓" if diff_pct >= 0 else "✗"
            lines.append(f"| {metric_name} | {baseline_val:.2f} | {current_val:.2f} | {diff_pct:+.1f}% {status} |")

        lines.append("")

    if new_failures:
        lines.append("## New Failures")
        lines.append("")
        for entry in new_failures:
            lines.append(f"- {entry.case_id}")
        lines.append("")

    if fixed_failures:
        lines.append("## Fixed Failures")
        lines.append("")
        for entry in fixed_failures:
            lines.append(f"- {entry.case_id}")
        lines.append("")

    lines.append(f"**Regressions**: {len(diff_result.regressions)}")
    lines.append(f"**Status**: {'✗ FAILED' if diff_result.has_regressions() else '✓ PASSED'}")

    return "\n".join(lines)


def _print_agent_console_report(
    result: AgentRunResult, verbose: bool, quiet: bool
) -> None:
    """Print agent run results to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Summary table
    table = Table(title="Agent Bundle Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Tasks", str(result.total_tasks))
    table.add_row("Passed", str(result.passed_tasks))
    table.add_row("Failed", str(result.failed_tasks))
    table.add_row("Success Rate", f"{result.task_success_rate:.1%}")
    table.add_row("Avg Steps", f"{result.avg_steps_used:.1f}")
    table.add_row("Avg Latency", f"{result.avg_latency_ms:.0f}ms")

    console.print(table)

    # Threshold results
    if result.threshold_results:
        console.print()
        threshold_table = Table(title="Threshold Results")
        threshold_table.add_column("Metric", style="cyan")
        threshold_table.add_column("Expected", style="yellow")
        threshold_table.add_column("Actual", style="green")
        threshold_table.add_column("Status", style="bold")

        for tr in result.threshold_results:
            status = "[green]✓ PASS[/green]" if tr["passed"] else "[red]✗ FAIL[/red]"
            threshold_table.add_row(
                tr["metric"],
                f"{tr['operator']} {tr['value']}",
                f"{tr['actual']:.3f}",
                status,
            )

        console.print(threshold_table)

    # Task results
    if verbose:
        console.print()
        task_table = Table(title="Task Results")
        task_table.add_column("Task ID", style="cyan")
        task_table.add_column("Status", style="bold")
        task_table.add_column("Steps", style="dim")
        task_table.add_column("Latency", style="dim")

        for tr in result.task_results:
            status = "[green]✓[/green]" if tr.success else "[red]✗[/red]"
            task_table.add_row(
                tr.task_id,
                status,
                str(tr.steps_used),
                f"{tr.latency_ms:.0f}ms",
            )

        console.print(task_table)


def _generate_agent_markdown_report(result: AgentRunResult) -> str:
    """Generate markdown report for agent run."""
    lines = [
        f"# Agent Bundle Results: {result.bundle_name}",
        "",
        "## Summary",
        "",
        f"- **Total Tasks**: {result.total_tasks}",
        f"- **Passed**: {result.passed_tasks}",
        f"- **Failed**: {result.failed_tasks}",
        f"- **Success Rate**: {result.task_success_rate:.1%}",
        f"- **Avg Steps**: {result.avg_steps_used:.1f}",
        f"- **Avg Latency**: {result.avg_latency_ms:.0f}ms",
        "",
    ]

    if result.threshold_results:
        lines.append("## Thresholds")
        lines.append("")
        lines.append("| Metric | Expected | Actual | Status |")
        lines.append("|--------|----------|--------|--------|")

        for tr in result.threshold_results:
            status = "✓" if tr["passed"] else "✗"
            lines.append(
                f"| {tr['metric']} | {tr['operator']} {tr['value']} | "
                f"{tr['actual']:.3f} | {status} |"
            )

        lines.append("")

    lines.append("## Tasks")
    lines.append("")
    lines.append("| Task ID | Status | Steps | Latency |")
    lines.append("|---------|--------|-------|---------|")

    for tr in result.task_results:
        status = "✓" if tr.success else "✗"
        lines.append(f"| {tr.task_id} | {status} | {tr.steps_used} | {tr.latency_ms:.0f}ms |")

    lines.append("")

    return "\n".join(lines)


# Example bundle templates
EXAMPLE_BUNDLE_YAML = """\
name: example-bundle
version: "1.0"
description: An example Behavior Bundle for testing

prompt_path: prompt.md
dataset_path: dataset.jsonl

output_contract:
  schema_path: schema.json
  invariants:
    - "len(raw_output) < 1000"
    - "'error' not in raw_output.lower()"

thresholds:
  - metric: pass_rate
    operator: ">="
    value: 0.8

provider:
  name: mock  # Change to 'openai' or 'anthropic' for real testing
  model: null
  temperature: 0.0
"""

EXAMPLE_PROMPT = """\
You are a helpful assistant that answers questions concisely.

Question: {{ question }}

Respond with a JSON object containing:
- "answer": Your concise answer
- "confidence": A number from 0 to 1 indicating confidence

Respond only with valid JSON, no additional text.
"""

EXAMPLE_DATASET = """\
{"input": {"question": "What is 2 + 2?"}, "expected_output": {"answer": "4"}}
{"input": {"question": "What color is the sky?"}, "expected_output": {"answer": "blue"}}
{"input": {"question": "What is the capital of France?"}, "expected_output": {"answer": "Paris"}}
"""

EXAMPLE_SCHEMA = """\
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["answer", "confidence"],
  "properties": {
    "answer": {
      "type": "string",
      "minLength": 1
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    }
  },
  "additionalProperties": false
}
"""


# Example Agent Bundle templates

EXAMPLE_AGENT_YAML = """\
name: example-agent
version: "1.0"
description: An example Agent Bundle for testing AI agents

agent:
  type: tool-calling
  model: gpt-4o
  temperature: 0.1
  max_steps: 10
  max_tokens: 4096

tools:
  - type: function
    name: read_file
    description: "Read contents of a file"
    parameters:
      type: object
      properties:
        path:
          type: string
          description: "Path to the file"
  - type: function
    name: write_file
    description: "Write content to a file"
    parameters:
      type: object
      properties:
        path:
          type: string
          description: "Path to the file"
        content:
          type: string
          description: "Content to write"
  - type: function
    name: list_directory
    description: "List files in a directory"
    parameters:
      type: object
      properties:
        path:
          type: string
          description: "Path to directory"

memory:
  enabled: true
  type: conversation
  max_turns: 20

execution:
  timeout: 60
  retry_on_failure: true
  record_traces: true
  max_steps: 10

provider:
  name: mock  # Change to 'openai' for real testing
"""

EXAMPLE_AGENT_PROMPT = """\
You are a helpful coding assistant that helps users with file operations.

You have access to the following tools:
- read_file: Read a file's contents
- write_file: Create or update a file
- list_directory: List files in a directory

Always complete the user's request efficiently and safely.
"""

EXAMPLE_AGENT_TASKS = """\
{"task_id": "task-001", "input": "List the files in the current directory", "expected": {"tool_calls": ["list_directory"]}}
{"task_id": "task-002", "input": "Read the README.md file if it exists", "expected": {"tool_calls": ["read_file"]}}
{"task_id": "task-003", "input": "Create a file called hello.txt with content 'Hello, World!'", "expected": {"tool_calls": ["write_file"], "contains": "hello.txt"}}
"""

EXAMPLE_AGENT_EVALUATION = """\
evaluation:
  tool_usage:
    required_tools: []
    forbidden_tools: []
    order_matters: false

  output:
    type: contains
    contains: ""

  task_completion:
    type: exact-match

  safety:
    - name: no_shell_exec

  invariants: []
"""

EXAMPLE_AGENT_THRESHOLDS = """\
thresholds:
  - metric: task_success_rate
    operator: ">="
    value: 0.8
"""


if __name__ == "__main__":
    main()
