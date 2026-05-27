"""Command-line interface for the memoir project.

Provides a Click-based CLI with Rich-powered terminal output for analysing
git repositories, detecting patterns, forecasting crises, and generating
developer autobiographies.

Usage::

    memoir scan --repo /path/to/repo
    memoir patterns --repo /path/to/repo
    memoir forecast --repo /path/to/repo
    memoir health --repo /path/to/repo
    memoir export --repo /path/to/repo -f markdown -f html
    memoir status --repo /path/to/repo
    memoir config ai_api_key sk-xxx
    memoir reset --repo /path/to/repo
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule

from memoir.core.git_analyzer import GitAnalyzer
from memoir.core.pattern_detector import PatternDetector
from memoir.core.crisis_forecast import CrisisForecaster
from memoir.core.narrator import Narrator, _compute_overall_health_score
from memoir.core.storage import Storage
from memoir.models.pattern import Pattern, PatternType, Severity
from memoir.models.forecast import Forecast, RiskLevel, RiskType
from memoir.utils.config import MemoirConfig
from memoir.utils.formatter import (
    format_date,
    format_number,
    format_percent,
    truncate_text,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rich console (global for this module)
# ---------------------------------------------------------------------------

console = Console()

# ---------------------------------------------------------------------------
# Severity / risk colour mapping for Rich
# ---------------------------------------------------------------------------

_SEVERITY_RICH_STYLE: Dict[Severity, str] = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "yellow",
    Severity.MEDIUM: "blue",
    Severity.LOW: "green",
}

_RISK_RICH_STYLE: Dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: "bold red",
    RiskLevel.HIGH: "yellow",
    RiskLevel.MODERATE: "blue",
    RiskLevel.LOW: "green",
}

_PATTERN_TYPE_LABELS: Dict[PatternType, str] = {
    PatternType.RECURRING_FIX: "Recurring Fix",
    PatternType.TECHNICAL_DEBT: "Technical Debt",
    PatternType.LEARNING_CURVE: "Learning Curve",
    PatternType.BURNOUT_INDICATOR: "Burnout Indicator",
    PatternType.CODE_OWNERSHIP: "Code Ownership",
    PatternType.ANTI_PATTERN: "Anti-Pattern",
    PatternType.IRREGULAR_HOURS: "Irregular Hours",
    PatternType.HIGH_CHURN: "High Churn",
}

_RISK_TYPE_LABELS: Dict[RiskType, str] = {
    RiskType.BURNOUT: "Burnout",
    RiskType.TECHNICAL_DEBT_CRISIS: "Tech Debt Crisis",
    RiskType.BUS_FACTOR: "Bus Factor",
    RiskType.MAINTAINABILITY: "Maintainability",
    RiskType.STAGNATION: "Stagnation",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_repo_path(repo: str) -> str:
    """Resolve and validate a repository path.

    Returns the absolute path.  Exits with a friendly error if the path
    does not exist or is not a git repository.
    """
    repo_path = os.path.abspath(repo)
    if not os.path.isdir(repo_path):
        console.print(
            Panel(
                f"[red]Directory not found:[/red] {repo_path}\n\n"
                "Please provide a valid path to a git repository.",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        raise SystemExit(1)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        console.print(
            Panel(
                f"[red]Not a git repository:[/red] {repo_path}\n\n"
                "The directory exists but does not contain a .git folder.\n"
                "Please navigate to or specify a git repository.",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        raise SystemExit(1)
    return repo_path


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse a YYYY-MM-DD date string, or return None."""
    if date_str is None:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        console.print(
            f"[red]Invalid date format:[/red] {date_str!r}  "
            "(expected YYYY-MM-DD)"
        )
        raise SystemExit(1)


def _get_repo_name(repo_path: str) -> str:
    """Return a human-readable repository name from its path."""
    return os.path.basename(os.path.abspath(repo_path))


def _health_score_style(score: float) -> str:
    """Return a Rich style string for a health score."""
    if score >= 80:
        return "bold green"
    if score >= 60:
        return "green"
    if score >= 40:
        return "yellow"
    if score >= 20:
        return "red"
    return "bold red"


def _health_score_label(score: float) -> str:
    """Return a descriptive label for a health score."""
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Fair"
    if score >= 20:
        return "Poor"
    return "Critical"


def _run_analysis(
    repo_path: str,
    author: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Run the full analysis pipeline and cache results.

    Returns a dict with keys: commits, git_stats, work_pattern,
    message_patterns, code_churn, most_changed_files, patterns,
    forecasts, memoir, health_score.
    """
    storage = Storage(repo_path)
    repo_name = _get_repo_name(repo_path)

    # Try loading cached data unless refresh is requested
    cached_analysis = None if refresh else storage.load_analysis()
    cached_patterns = None if refresh else storage.load_patterns()
    cached_forecasts = None if refresh else storage.load_forecasts()
    cached_memoir = None if refresh else storage.load_memoir()

    if (
        cached_analysis is not None
        and cached_patterns is not None
        and cached_forecasts is not None
        and cached_memoir is not None
    ):
        return {
            "commits": cached_analysis["commits"],
            "git_stats": cached_analysis["git_stats"],
            "work_pattern": cached_analysis["work_pattern"],
            "message_patterns": cached_analysis["message_patterns"],
            "code_churn": cached_analysis["code_churn"],
            "most_changed_files": cached_analysis["most_changed_files"],
            "patterns": cached_patterns,
            "forecasts": cached_forecasts,
            "memoir": cached_memoir,
            "health_score": cached_memoir.overall_health_score,
            "from_cache": True,
        }

    # Step 1: Analyse git history
    analyzer = GitAnalyzer(
        repo_path=repo_path,
        author_filter=author,
        since=since,
        until=until,
    )

    try:
        commits, git_stats = analyzer.analyze()
    except FileNotFoundError:
        console.print(
            Panel(
                f"[red]Repository path does not exist:[/red] {repo_path}",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        raise SystemExit(1)
    except ValueError as exc:
        msg = str(exc)
        if "no commits" in msg.lower():
            console.print(
                Panel(
                    f"[yellow]This repository has no commits yet.[/yellow]\n\n"
                    f"Repository: {repo_path}\n\n"
                    "Make at least one commit and try again.",
                    title="[bold yellow]Empty Repository[/bold yellow]",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                Panel(
                    f"[red]Not a valid git repository:[/red] {repo_path}\n\n{msg}",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )
        raise SystemExit(1)

    work_pattern = analyzer.get_work_pattern(commits)
    message_patterns = analyzer.get_commit_message_patterns(commits)
    code_churn = analyzer.get_code_churn()
    most_changed_files = analyzer.get_most_changed_files(commits)

    # Save analysis
    storage.save_analysis(
        commits, git_stats, work_pattern, message_patterns, code_churn, most_changed_files
    )

    # Step 2: Detect patterns
    detector = PatternDetector(
        commits=commits,
        git_stats=git_stats,
        work_pattern=work_pattern,
        message_patterns=message_patterns,
        code_churn=code_churn,
        most_changed_files=most_changed_files,
    )
    patterns = detector.detect_all()
    storage.save_patterns(patterns)

    # Step 3: Forecast risks
    forecaster = CrisisForecaster(
        commits=commits,
        git_stats=git_stats,
        patterns=patterns,
        work_pattern=work_pattern,
        message_patterns=message_patterns,
        code_churn=code_churn,
    )
    forecasts = forecaster.forecast_all()
    storage.save_forecasts(forecasts)

    # Step 4: Generate memoir (template mode — AI is handled separately)
    ai_config = None
    config = MemoirConfig(repo_path)
    if config.ai_api_key:
        ai_config = {
            "api_key": config.ai_api_key,
            "model": config.ai_model,
            "base_url": config.ai_base_url,
        }

    narrator = Narrator(
        commits=commits,
        git_stats=git_stats,
        patterns=patterns,
        forecasts=forecasts,
        repo_name=repo_name,
        ai_config=ai_config,
    )
    memoir = narrator.generate_memoir()
    storage.save_memoir(memoir)

    health_score = memoir.overall_health_score

    return {
        "commits": commits,
        "git_stats": git_stats,
        "work_pattern": work_pattern,
        "message_patterns": message_patterns,
        "code_churn": code_churn,
        "most_changed_files": most_changed_files,
        "patterns": patterns,
        "forecasts": forecasts,
        "memoir": memoir,
        "health_score": health_score,
        "from_cache": False,
    }


def _load_or_analyze(
    repo_path: str,
    refresh: bool = False,
    author: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Load cached results or run analysis if no cache exists.

    Shows a friendly message if no cached data is found and runs
    the full analysis pipeline.
    """
    storage = Storage(repo_path)

    if not refresh:
        cached_analysis = storage.load_analysis()
        cached_patterns = storage.load_patterns()
        cached_forecasts = storage.load_forecasts()
        cached_memoir = storage.load_memoir()

        if (
            cached_analysis is not None
            and cached_patterns is not None
            and cached_forecasts is not None
            and cached_memoir is not None
        ):
            return {
                "commits": cached_analysis["commits"],
                "git_stats": cached_analysis["git_stats"],
                "work_pattern": cached_analysis["work_pattern"],
                "message_patterns": cached_analysis["message_patterns"],
                "code_churn": cached_analysis["code_churn"],
                "most_changed_files": cached_analysis["most_changed_files"],
                "patterns": cached_patterns,
                "forecasts": cached_forecasts,
                "memoir": cached_memoir,
                "health_score": cached_memoir.overall_health_score,
                "from_cache": True,
            }

    # No cached data — run full analysis
    if not refresh:
        console.print(
            "[dim]No cached analysis found. Running full analysis...[/dim]"
        )

    return _run_analysis(
        repo_path=repo_path,
        author=author,
        since=since,
        until=until,
        refresh=refresh,
    )


def _export_memoir(
    memoir_obj: Any,
    export_formats: Tuple[str, ...],
    output_dir: Optional[str],
    repo_path: str,
) -> List[str]:
    """Export the memoir to the specified format(s).

    Returns a list of output file paths.
    """
    if output_dir is None:
        output_dir = os.path.join(repo_path, ".memoir", "exports")

    os.makedirs(output_dir, exist_ok=True)

    output_paths: List[str] = []

    for fmt in export_formats:
        fmt_lower = fmt.strip().lower()

        if fmt_lower == "markdown":
            from memoir.exporters.markdown import MarkdownExporter

            exporter = MarkdownExporter()
            path = exporter.export(memoir_obj, os.path.join(output_dir, "memoir.md"))
            output_paths.append(path)

        elif fmt_lower == "json":
            from memoir.exporters.json_export import JsonExporter

            exporter = JsonExporter()
            path = exporter.export(memoir_obj, os.path.join(output_dir, "memoir.json"))
            output_paths.append(path)

        elif fmt_lower == "html":
            from memoir.exporters.html import HtmlExporter

            exporter = HtmlExporter()
            path = exporter.export(memoir_obj, os.path.join(output_dir, "memoir.html"))
            output_paths.append(path)

        else:
            console.print(
                f"[yellow]Warning:[/yellow] Unknown export format "
                f"[bold]{fmt!r}[/bold].  Supported: markdown, json, html"
            )

    return output_paths


# ========================================================================
# CLI group
# ========================================================================


@click.group()
@click.version_option(version="0.1.0", prog_name="memoir")
def main() -> None:
    """memoir — your coding autobiography, written by data

    Analyse git repositories to detect patterns, forecast risks,
    and generate a data-driven developer memoir.
    """


# ========================================================================
# scan command
# ========================================================================


@main.command()
@click.option("--repo", "-r", default=".", help="Path to git repository")
@click.option(
    "--author", "-a", default=None, help="Filter by author name/email"
)
@click.option(
    "--since", default=None, help="Analyze commits since date (YYYY-MM-DD)"
)
@click.option(
    "--until", default=None, help="Analyze commits until date (YYYY-MM-DD)"
)
@click.option(
    "--format",
    "-f",
    "export_format",
    multiple=True,
    default=["markdown"],
    help="Export format: markdown, json, html",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory (default: .memoir/exports/)",
)
@click.option("--ai", is_flag=True, help="Enable AI-enhanced narratives")
@click.option(
    "--refresh", is_flag=True, help="Force refresh analysis cache"
)
def scan(
    repo: str,
    author: Optional[str],
    since: Optional[str],
    until: Optional[str],
    export_format: Tuple[str, ...],
    output: Optional[str],
    ai: bool,
    refresh: bool,
) -> None:
    """Analyze git history and generate your coding memoir."""
    repo_path = _resolve_repo_path(repo)
    since_dt = _parse_date(since)
    until_dt = _parse_date(until)
    repo_name = _get_repo_name(repo_path)

    # Determine if AI should be enabled
    if ai:
        config = MemoirConfig(repo_path)
        if not config.ai_api_key:
            console.print(
                Panel(
                    "[yellow]AI enhancement requested but no API key configured.[/yellow]\n\n"
                    "Set your API key with:\n"
                    "  [bold]memoir config ai_api_key sk-xxx[/bold]\n\n"
                    "Or set the [bold]MEMOIR_AI_API_KEY[/bold] environment variable.\n\n"
                    "Continuing with template-mode narratives...",
                    title="[bold yellow]AI Not Available[/bold yellow]",
                    border_style="yellow",
                )
            )

    # Run analysis with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        # Step 1: Analyse git history
        task = progress.add_task("Analyzing git history...", total=100)

        analyzer = GitAnalyzer(
            repo_path=repo_path,
            author_filter=author,
            since=since_dt,
            until=until_dt,
        )

        try:
            commits, git_stats = analyzer.analyze()
        except FileNotFoundError:
            console.print(
                Panel(
                    f"[red]Repository path does not exist:[/red] {repo_path}",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )
            raise SystemExit(1)
        except ValueError as exc:
            msg = str(exc)
            if "no commits" in msg.lower():
                console.print(
                    Panel(
                        f"[yellow]This repository has no commits yet.[/yellow]\n\n"
                        f"Repository: {repo_path}\n\n"
                        "Make at least one commit and try again.",
                        title="[bold yellow]Empty Repository[/bold yellow]",
                        border_style="yellow",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"[red]Not a valid git repository:[/red] {repo_path}\n\n{msg}",
                        title="[bold red]Error[/bold red]",
                        border_style="red",
                    )
                )
            raise SystemExit(1)

        progress.update(task, completed=20)

        work_pattern = analyzer.get_work_pattern(commits)
        message_patterns = analyzer.get_commit_message_patterns(commits)
        code_churn = analyzer.get_code_churn()
        most_changed_files = analyzer.get_most_changed_files(commits)

        storage = Storage(repo_path)
        storage.save_analysis(
            commits, git_stats, work_pattern, message_patterns, code_churn, most_changed_files
        )
        progress.update(task, completed=40, description="Detecting patterns...")

        # Step 2: Detect patterns
        detector = PatternDetector(
            commits=commits,
            git_stats=git_stats,
            work_pattern=work_pattern,
            message_patterns=message_patterns,
            code_churn=code_churn,
            most_changed_files=most_changed_files,
        )
        patterns = detector.detect_all()
        storage.save_patterns(patterns)
        progress.update(task, completed=55, description="Forecasting risks...")

        # Step 3: Forecast risks
        forecaster = CrisisForecaster(
            commits=commits,
            git_stats=git_stats,
            patterns=patterns,
            work_pattern=work_pattern,
            message_patterns=message_patterns,
            code_churn=code_churn,
        )
        forecasts = forecaster.forecast_all()
        storage.save_forecasts(forecasts)
        progress.update(task, completed=70, description="Generating memoir...")

        # Step 4: Generate memoir
        ai_config = None
        config = MemoirConfig(repo_path)
        if ai and config.ai_api_key:
            ai_config = {
                "api_key": config.ai_api_key,
                "model": config.ai_model,
                "base_url": config.ai_base_url,
            }
        elif config.ai_api_key:
            # Auto-enable AI if key is present
            ai_config = {
                "api_key": config.ai_api_key,
                "model": config.ai_model,
                "base_url": config.ai_base_url,
            }

        narrator = Narrator(
            commits=commits,
            git_stats=git_stats,
            patterns=patterns,
            forecasts=forecasts,
            repo_name=repo_name,
            ai_config=ai_config,
        )
        memoir_obj = narrator.generate_memoir()
        storage.save_memoir(memoir_obj)
        progress.update(task, completed=85, description="Exporting...")

        # Step 5: Export
        export_paths = _export_memoir(
            memoir_obj, export_format, output, repo_path
        )
        progress.update(task, completed=100, description="Done!")

    # Health score
    health_score = memoir_obj.overall_health_score

    # Final summary panel
    summary_lines = [
        f"[bold]Repository:[/bold]  {repo_name}",
        f"[bold]Commits:[/bold]     {format_number(git_stats.total_commits)}",
        f"[bold]Authors:[/bold]     {git_stats.unique_authors}",
        f"[bold]Date range:[/bold]  {format_date(git_stats.first_commit_date)} — {format_date(git_stats.last_commit_date)}",
        "",
        f"[bold]Health Score:[/bold] [{_health_score_style(health_score)}]{health_score:.1f}/100[/]  {_health_score_label(health_score)}",
        f"[bold]Patterns:[/bold]     {len(patterns)} detected",
        f"[bold]Forecasts:[/bold]    {len(forecasts)} issued",
        f"[bold]Chapters:[/bold]     {len(memoir_obj.chapters)} written",
        "",
    ]

    if export_paths:
        summary_lines.append("[bold]Exports:[/bold]")
        for p in export_paths:
            summary_lines.append(f"  📄 {p}")

    console.print()
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="[bold]📖 Memoir Scan Complete[/bold]",
            border_style="green" if health_score >= 60 else ("yellow" if health_score >= 40 else "red"),
            padding=(1, 2),
        )
    )


# ========================================================================
# patterns command
# ========================================================================


@main.command()
@click.option("--repo", "-r", default=".", help="Path to git repository")
@click.option("--refresh", is_flag=True, help="Force refresh")
def patterns(repo: str, refresh: bool) -> None:
    """Show detected patterns in your codebase."""
    repo_path = _resolve_repo_path(repo)

    data = _load_or_analyze(repo_path, refresh=refresh)
    detected: List[Pattern] = data["patterns"]

    if not detected:
        console.print(
            Panel(
                "[green]No significant patterns detected.[/green]\n\n"
                "This is a good sign — the analysis did not find any "
                "recurring issues, technical debt markers, or concerning "
                "work patterns in the commit history.",
                title="[bold green]All Clear[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
        return

    # Build a Rich table
    table = Table(
        title="🔍 Detected Patterns",
        show_lines=True,
        border_style="bright_blue",
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Type", style="dim", width=16)
    table.add_column("Severity", width=14)
    table.add_column("Title", min_width=30)
    table.add_column("Occurrences", justify="right", width=11)
    table.add_column("Confidence", justify="right", width=10)

    for p in detected:
        type_label = _PATTERN_TYPE_LABELS.get(p.pattern_type, p.pattern_type.value)
        sev_style = _SEVERITY_RICH_STYLE.get(p.severity, "white")
        sev_text = f"{p.severity.value.upper()}"
        conf_pct = f"{p.confidence * 100:.0f}%"

        table.add_row(
            type_label,
            f"[{sev_style}]{sev_text}[/{sev_style}]",
            p.title,
            str(p.occurrence_count),
            conf_pct,
        )

    console.print()
    console.print(table)

    # Show descriptions in a tree view grouped by type
    console.print()
    tree = Tree("[bold]📋 Pattern Details[/bold]")

    by_type: Dict[PatternType, List[Pattern]] = {}
    for p in detected:
        by_type.setdefault(p.pattern_type, []).append(p)

    for pattern_type, group in by_type.items():
        type_label = _PATTERN_TYPE_LABELS.get(pattern_type, pattern_type.value)
        branch = tree.add(f"[bold]{type_label}[/bold] ({len(group)})")

        for p in group:
            sev_style = _SEVERITY_RICH_STYLE.get(p.severity, "white")
            node_text = (
                f"[{sev_style}]●[/{sev_style}] "
                f"{p.title}  "
                f"[dim](×{p.occurrence_count}, {p.confidence * 100:.0f}% confidence)[/dim]"
            )
            leaf = branch.add(node_text)
            # Truncate description for readability
            desc = truncate_text(p.description, 150)
            leaf.add(f"[dim]{desc}[/dim]")
            if p.recommendation:
                rec = truncate_text(p.recommendation, 120)
                leaf.add(f"[italic]💡 {rec}[/italic]")

    console.print(tree)


# ========================================================================
# forecast command
# ========================================================================


@main.command()
@click.option("--repo", "-r", default=".", help="Path to git repository")
@click.option("--refresh", is_flag=True, help="Force refresh")
def forecast(repo: str, refresh: bool) -> None:
    """Show crisis forecasts and risk assessment."""
    repo_path = _resolve_repo_path(repo)

    data = _load_or_analyze(repo_path, refresh=refresh)
    forecasts_list: List[Forecast] = data["forecasts"]
    health_score: float = data["health_score"]

    # Health score panel at the top
    score_style = _health_score_style(health_score)
    console.print()
    console.print(
        Panel(
            f"[{score_style}]{health_score:.1f}[/] / 100\n"
            f"{_health_score_label(health_score)}",
            title="[bold]🏥 Repository Health Score[/bold]",
            border_style="green" if health_score >= 60 else ("yellow" if health_score >= 40 else "red"),
            padding=(1, 2),
        )
    )

    if not forecasts_list:
        console.print(
            Panel(
                "[green]No significant risk forecasts.[/green]\n\n"
                "The analysis did not identify any trends that suggest "
                "an impending crisis.  Continue monitoring by running "
                "[bold]memoir scan[/bold] periodically.",
                title="[bold green]Low Risk[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
        return

    # Display each forecast as a panel
    console.print()
    console.print(Rule("[bold]🔮 Risk Forecasts[/bold]"))

    for fc in forecasts_list:
        risk_style = _RISK_RICH_STYLE.get(fc.risk_level, "white")
        risk_label = fc.risk_level.value.upper()
        prob_pct = f"{fc.probability * 100:.0f}%"

        # Header line
        header = (
            f"[{risk_style}]● {risk_label}[/{risk_style}]  "
            f"{fc.title}  "
            f"[dim]({prob_pct} probability, timeline: {fc.estimated_timeline})[/dim]"
        )

        # Body
        body_parts: List[str] = [fc.description]

        # Indicators table
        if fc.indicators:
            body_parts.append("")
            body_parts.append("[bold]Indicators:[/bold]")
            for ind in fc.indicators:
                trend_arrow = {"rising": "↑", "stable": "→", "declining": "↓"}.get(ind.trend, "?")
                trend_color = {"rising": "red", "stable": "yellow", "declining": "green"}.get(ind.trend, "white")
                body_parts.append(
                    f"  • [bold]{ind.name}[/bold]: "
                    f"{ind.current_value:.1f} (threshold {ind.threshold_value:.1f}) "
                    f"— [{trend_color}]{trend_arrow} {ind.trend}[/{trend_color}]"
                )

        # Historical precedent
        if fc.historical_precedent:
            body_parts.append("")
            body_parts.append(
                f"[dim]📜 Historical precedent: {fc.historical_precedent}[/dim]"
            )

        # Recommendation
        if fc.recommendation:
            body_parts.append("")
            body_parts.append(f"[italic]💡 {fc.recommendation}[/italic]")

        border = "red" if fc.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH) else (
            "yellow" if fc.risk_level == RiskLevel.MODERATE else "green"
        )

        console.print(
            Panel(
                "\n".join(body_parts),
                title=header,
                border_style=border,
                padding=(1, 2),
            )
        )


# ========================================================================
# health command
# ========================================================================


@main.command()
@click.option("--repo", "-r", default=".", help="Path to git repository")
def health(repo: str) -> None:
    """Quick health check of your codebase."""
    repo_path = _resolve_repo_path(repo)

    # Try to load cached data first; if none, run a quick analysis
    storage = Storage(repo_path)
    cached_analysis = storage.load_analysis()
    cached_patterns = storage.load_patterns()
    cached_forecasts = storage.load_forecasts()
    cached_memoir = storage.load_memoir()

    if (
        cached_analysis is not None
        and cached_patterns is not None
        and cached_forecasts is not None
        and cached_memoir is not None
    ):
        git_stats = cached_analysis["git_stats"]
        work_pattern = cached_analysis["work_pattern"]
        patterns_list = cached_patterns
        forecasts_list = cached_forecasts
        health_score = cached_memoir.overall_health_score
        commits = cached_analysis["commits"]
    else:
        # No cache — run a quick analysis with a spinner
        with console.status("[bold blue]Analyzing repository...[/bold blue]", spinner="dots"):
            data = _run_analysis(repo_path)
        git_stats = data["git_stats"]
        work_pattern = data["work_pattern"]
        patterns_list = data["patterns"]
        forecasts_list = data["forecasts"]
        health_score = data["health_score"]
        commits = data["commits"]

    # Big health score display
    score_style = _health_score_style(health_score)
    console.print()

    # Build a compact dashboard
    dashboard = Table.grid(padding=(0, 3))
    dashboard.add_column(justify="center")
    dashboard.add_column()

    dashboard.add_row(
        f"[{score_style}]{health_score:.1f}[/]",
        "[bold]Health Score[/bold]",
    )
    dashboard.add_row(
        f"[dim]{_health_score_label(health_score)}[/]",
        f"[dim]out of 100[/]",
    )

    console.print(
        Panel(
            dashboard,
            title="[bold]🏥 Codebase Health[/bold]",
            border_style="green" if health_score >= 60 else ("yellow" if health_score >= 40 else "red"),
            padding=(1, 2),
            width=60,
        )
    )

    # Key stats
    console.print()
    stats_table = Table(
        title="📊 Key Stats",
        border_style="bright_blue",
        show_header=True,
        header_style="bold",
    )
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", justify="right")

    after_hours_ratio = work_pattern.get("after_hours_ratio", 0.0)
    weekend_ratio = work_pattern.get("weekend_ratio", 0.0)

    stats_table.add_row("Total Commits", format_number(git_stats.total_commits))
    stats_table.add_row("Unique Authors", str(git_stats.unique_authors))
    stats_table.add_row("After-hours %", format_percent(after_hours_ratio))
    stats_table.add_row("Weekend %", format_percent(weekend_ratio))

    console.print(stats_table)

    # Top 3 risks
    if forecasts_list:
        console.print()
        risk_table = Table(
            title="⚠️  Top Risks",
            border_style="red",
            show_header=True,
            header_style="bold",
        )
        risk_table.add_column("Risk", min_width=20)
        risk_table.add_column("Level", width=12)
        risk_table.add_column("Probability", justify="right", width=12)
        risk_table.add_column("Timeline", width=14)

        for fc in forecasts_list[:3]:
            risk_style = _RISK_RICH_STYLE.get(fc.risk_level, "white")
            risk_table.add_row(
                fc.title,
                f"[{risk_style}]{fc.risk_level.value.upper()}[/{risk_style}]",
                f"{fc.probability * 100:.0f}%",
                fc.estimated_timeline,
            )

        console.print(risk_table)

    # Quick recommendations
    console.print()
    recs: List[str] = []

    if health_score < 60:
        recs.append("Health score is below 60 — consider addressing the detected patterns.")

    if after_hours_ratio > 0.30:
        recs.append(
            f"After-hours commit ratio is high ({format_percent(after_hours_ratio)}). "
            "Consider setting boundaries around work time."
        )

    if git_stats.unique_authors == 1:
        recs.append(
            "Single contributor project — bus factor is 1. "
            "Consider onboarding additional contributors."
        )

    if forecasts_list:
        for fc in forecasts_list[:2]:
            if fc.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
                recs.append(f"⚠️  {fc.title}: {fc.recommendation}")

    if not recs:
        recs.append("No immediate concerns. Keep up the good work!")

    console.print(
        Panel(
            "\n".join(f"• {r}" for r in recs),
            title="[bold]💡 Quick Recommendations[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


# ========================================================================
# export command
# ========================================================================


@main.command()
@click.option("--repo", "-r", default=".", help="Path to git repository")
@click.option(
    "--format",
    "-f",
    "export_format",
    multiple=True,
    default=["markdown"],
)
@click.option("--output", "-o", default=None)
def export(
    repo: str,
    export_format: Tuple[str, ...],
    output: Optional[str],
) -> None:
    """Export memoir in chosen format(s)."""
    repo_path = _resolve_repo_path(repo)
    storage = Storage(repo_path)

    # Try to load cached memoir
    memoir_obj = storage.load_memoir()

    if memoir_obj is None:
        console.print(
            Panel(
                "[yellow]No cached memoir found.[/yellow]\n\n"
                "Run [bold]memoir scan[/bold] first to generate a memoir, "
                "then export it.",
                title="[bold yellow]No Data[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        raise SystemExit(1)

    # Export
    with console.status("[bold blue]Exporting memoir...[/bold blue]", spinner="dots"):
        paths = _export_memoir(memoir_obj, export_format, output, repo_path)

    if not paths:
        console.print("[red]No files were exported.[/red]")
        raise SystemExit(1)

    console.print()
    console.print(
        Panel(
            "\n".join(f"📄 {p}" for p in paths),
            title="[bold green]✅ Export Complete[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


# ========================================================================
# status command
# ========================================================================


@main.command()
@click.option("--repo", "-r", default=".", help="Path to git repository")
def status(repo: str) -> None:
    """Show storage status and cached data info."""
    repo_path = _resolve_repo_path(repo)
    storage = Storage(repo_path)
    status_data = storage.get_status()
    config = MemoirConfig(repo_path)

    # Storage location
    console.print()
    console.print(
        Panel(
            f"[bold]Storage Directory:[/bold]  {status_data['repo_dir']}\n"
            f"[bold]Exists:[/bold]  {'✅ Yes' if status_data['repo_dir_exists'] else '❌ No'}",
            title="[bold]📁 Storage Status[/bold]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )

    # Cached files
    console.print()
    files_table = Table(
        title="📦 Cached Data",
        border_style="bright_blue",
        show_header=True,
        header_style="bold",
    )
    files_table.add_column("Data Type", style="dim")
    files_table.add_column("Exists", width=8, justify="center")
    files_table.add_column("Last Updated", width=22)
    files_table.add_column("Size", justify="right", width=12)

    for key in ("analysis", "patterns", "forecasts", "memoir", "config"):
        info = status_data.get(key, {})
        exists = info.get("exists", False)
        last_updated = info.get("last_updated")
        size_bytes = info.get("size_bytes")

        exists_str = "✅" if exists else "—"
        updated_str = last_updated if last_updated else "—"

        if size_bytes is not None:
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            size_str = "—"

        files_table.add_row(key, exists_str, updated_str, size_str)

    console.print(files_table)

    # Config summary
    console.print()
    all_config = config.list_all()

    config_table = Table(
        title="⚙️  Configuration",
        border_style="bright_blue",
        show_header=True,
        header_style="bold",
    )
    config_table.add_column("Key", style="dim")
    config_table.add_column("Value")

    for key, value in all_config.items():
        # Mask sensitive values
        if "api_key" in key.lower() and value:
            display_val = value[:6] + "…" + value[-4:] if len(str(value)) > 10 else "••••••"
        elif value is None:
            display_val = "[dim](not set)[/dim]"
        else:
            display_val = str(value)

        config_table.add_row(key, display_val)

    console.print(config_table)

    # Hint if no data
    if not status_data.get("analysis", {}).get("exists"):
        console.print()
        console.print(
            "[dim]💡 No analysis data found. Run [bold]memoir scan[/bold] to get started.[/dim]"
        )


# ========================================================================
# config command
# ========================================================================


@main.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.option(
    "--global", "scope", flag_value="global", help="Set globally"
)
@click.option("--list", "list_all", is_flag=True, help="List all config")
@click.option("--repo", "-r", default=".", help="Path to git repository for project-scoped config")
def config(
    key: Optional[str],
    value: Optional[str],
    scope: Optional[str],
    list_all: bool,
    repo: str,
) -> None:
    """View or change configuration.

    \b
    memoir config                    Show all config
    memoir config ai_api_key         Show specific value
    memoir config ai_api_key sk-xxx  Set value (project scope)
    memoir config --global ai_model gpt-4o  Set globally
    """
    repo_path = os.path.abspath(repo)
    cfg = MemoirConfig(repo_path)

    # memoir config --list
    if list_all:
        all_config = cfg.list_all()

        table = Table(
            title="⚙️  All Configuration",
            border_style="bright_blue",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Key", style="dim")
        table.add_column("Value")
        table.add_column("Scope", style="dim", width=10)

        for k, v in all_config.items():
            # Mask sensitive values
            if "api_key" in k.lower() and v:
                display_val = str(v)[:6] + "…" + str(v)[-4:] if len(str(v)) > 10 else "••••••"
            elif v is None:
                display_val = "[dim](not set)[/dim]"
            else:
                display_val = str(v)

            table.add_row(k, display_val, "resolved")

        console.print(table)
        return

    # memoir config (no args) — show all
    if key is None:
        all_config = cfg.list_all()

        table = Table(
            title="⚙️  Configuration",
            border_style="bright_blue",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Key", style="dim")
        table.add_column("Value")

        for k, v in all_config.items():
            if "api_key" in k.lower() and v:
                display_val = str(v)[:6] + "…" + str(v)[-4:] if len(str(v)) > 10 else "••••••"
            elif v is None:
                display_val = "[dim](not set)[/dim]"
            else:
                display_val = str(v)

            table.add_row(k, display_val)

        console.print(table)
        return

    # memoir config <key> — show specific value
    if value is None:
        resolved = cfg.get(key)
        if resolved is None:
            console.print(f"[dim]{key}[/dim] = [dim](not set)[/dim]")
        else:
            # Mask sensitive values
            if "api_key" in key.lower() and resolved:
                display_val = str(resolved)[:6] + "…" + str(resolved)[-4:] if len(str(resolved)) > 10 else "••••••"
            else:
                display_val = str(resolved)
            console.print(f"[bold]{key}[/bold] = {display_val}")
        return

    # memoir config <key> <value> — set value
    effective_scope = scope or "project"
    try:
        cfg.set(key, value, scope=effective_scope)
        cfg.save()
        scope_label = "global" if effective_scope == "global" else "project"
        console.print(
            f"[green]✅ Set[/green] [bold]{key}[/bold] = [dim]{value}[/dim] "
            f"[green]({scope_label} scope)[/green]"
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)
    except Exception as exc:
        console.print(f"[red]Failed to save config:[/red] {exc}")
        raise SystemExit(1)


# ========================================================================
# reset command
# ========================================================================


@main.command()
@click.option("--repo", "-r", default=".", help="Path to git repository")
@click.confirmation_option(prompt="Are you sure? This deletes all cached data.")
def reset(repo: str) -> None:
    """Clear all cached analysis data."""
    repo_path = _resolve_repo_path(repo)
    storage = Storage(repo_path)

    try:
        storage.clear(what="all")
        console.print(
            Panel(
                f"All cached data has been cleared.\n\n"
                f"Repository: {repo_path}\n\n"
                "Run [bold]memoir scan[/bold] to generate fresh analysis.",
                title="[bold green]🗑️  Cache Cleared[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
    except Exception as exc:
        console.print(f"[red]Failed to clear cache:[/red] {exc}")
        raise SystemExit(1)
