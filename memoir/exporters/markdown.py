"""Markdown exporter for the memoir project.

Generates a book-style Markdown document from a ``Memoir`` instance,
complete with a cover page, table of contents, narrative chapters,
pattern callouts, forecast warnings, commit timelines, and statistical
summaries.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict

from memoir.models.chapter import Chapter, ChapterType, Memoir
from memoir.models.commit_data import CommitData, GitStats
from memoir.models.forecast import Forecast, RiskLevel
from memoir.models.pattern import Pattern, PatternType, Severity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity / risk helpers (plain-text, no ANSI)
# ---------------------------------------------------------------------------

_SEVERITY_EMOJI: Dict[Severity, str] = {
    Severity.LOW: "🟢",
    Severity.MEDIUM: "🟡",
    Severity.HIGH: "🟠",
    Severity.CRITICAL: "🔴",
}

_RISK_EMOJI: Dict[RiskLevel, str] = {
    RiskLevel.LOW: "🟢",
    RiskLevel.MODERATE: "🟡",
    RiskLevel.HIGH: "🟠",
    RiskLevel.CRITICAL: "🔴",
}

_PATTERN_TYPE_LABEL: Dict[PatternType, str] = {
    PatternType.RECURRING_FIX: "Recurring Fix",
    PatternType.TECHNICAL_DEBT: "Technical Debt",
    PatternType.LEARNING_CURVE: "Learning Curve",
    PatternType.BURNOUT_INDICATOR: "Burnout Indicator",
    PatternType.CODE_OWNERSHIP: "Code Ownership",
    PatternType.ANTI_PATTERN: "Anti-Pattern",
    PatternType.IRREGULAR_HOURS: "Irregular Hours",
    PatternType.HIGH_CHURN: "High Churn",
}

_CHAPTER_TYPE_LABEL: Dict[ChapterType, str] = {
    ChapterType.PROLOGUE: "Prologue",
    ChapterType.PATTERN: "Patterns",
    ChapterType.MILESTONE: "Milestone",
    ChapterType.CRISIS: "Crisis",
    ChapterType.CURRENT_STATE: "Current State",
    ChapterType.FORECAST: "Forecast",
}


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


def _health_score_bar(score: float) -> str:
    """Return a simple ASCII bar gauge for a health score."""
    filled = int(round(score / 10))
    empty = 10 - filled
    return f"[{'█' * filled}{'░' * empty}] {score:.1f}/100"


def _slugify(text: str) -> str:
    """Convert text to a Markdown-compatible anchor slug."""
    slug = text.lower().strip()
    # Replace non-alphanumeric with hyphens
    result: list[str] = []
    for ch in slug:
        if ch.isalnum():
            result.append(ch)
        elif ch in (" ", "-", "_"):
            result.append("-")
    return "".join(result).strip("-")


class MarkdownExporter:
    """Export a ``Memoir`` to a book-style Markdown file.

    The generated document is structured like a book with:

    - A cover page containing the title, repository name, generation date,
      and an ASCII health-score gauge.
    - A clickable table of contents.
    - Narrative chapters with pattern callouts, commit timelines, and
      data sidebars.
    - Forecast warnings rendered as blockquote callouts.
    - Statistical summary tables.

    Example::

        exporter = MarkdownExporter()
        path = exporter.export(memoir, "output/memoir.md")
    """

    def export(self, memoir: Memoir, output_path: str) -> str:
        """Export memoir to a Markdown file.

        Args:
            memoir: The memoir document to export.
            output_path: Destination file path.

        Returns:
            The absolute path of the written file.

        Raises:
            OSError: If the file cannot be written.
        """
        logger.info("Exporting memoir to Markdown: %s", output_path)

        sections: list[str] = [
            self._render_header(memoir),
            self._render_toc(memoir),
        ]

        for chapter in memoir.chapters:
            sections.append(self._render_chapter(chapter))

        # Forecasts section
        if memoir.forecasts:
            sections.append(self._render_forecasts_section(memoir))

        # Stats appendix
        sections.append(self._render_stats_summary(memoir.git_stats))

        content = "\n\n".join(sections) + "\n"

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        abs_path = os.path.abspath(output_path)
        logger.info("Markdown export complete: %s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # Cover page / header
    # ------------------------------------------------------------------

    def _render_header(self, memoir: Memoir) -> str:
        """Render the cover page with title, repo, date, and health score."""
        now = memoir.generated_at.strftime("%d %B %Y")
        label = _health_score_label(memoir.overall_health_score)
        bar = _health_score_bar(memoir.overall_health_score)

        lines: list[str] = [
            "---",
            'title: "Developer Memoir"',
            f"subtitle: \"{memoir.repo_name}\"",
            f"date: \"{now}\"",
            "author: Memoir Engine",
            "---",
            "",
            "<br/>",
            "",
            "<h1 align=\"center\">📖 Developer Memoir</h1>",
            "",
            f"<h2 align=\"center\"><em>{memoir.repo_name}</em></h2>",
            "",
            f"<p align=\"center\"><strong>Generated:</strong> {now}</p>",
            "",
            "<br/>",
            "",
            "<p align=\"center\">",
            f"<strong>Repository Health Score</strong><br/>",
            f"<code>{bar}</code><br/>",
            f"<em>{label}</em>",
            "</p>",
            "",
            "<br/>",
            "",
            "---",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Table of contents
    # ------------------------------------------------------------------

    def _render_toc(self, memoir: Memoir) -> str:
        """Render a clickable table of contents with chapter links."""
        lines: list[str] = [
            "## 📑 Table of Contents",
            "",
        ]

        for chapter in memoir.chapters:
            slug = _slugify(chapter.title)
            type_label = _CHAPTER_TYPE_LABEL.get(
                chapter.chapter_type, chapter.chapter_type.value
            )
            lines.append(
                f"- **{type_label}**: [{chapter.title}](#{slug})"
            )

        if memoir.forecasts:
            lines.append("- **Forecasts**: [Risk Forecasts](#risk-forecasts)")

        lines.append("- **Appendix**: [Statistics Summary](#statistics-summary)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Chapter rendering
    # ------------------------------------------------------------------

    def _render_chapter(self, chapter: Chapter) -> str:
        """Render a single chapter with title, narrative, patterns, and stats."""
        parts: list[str] = []

        # Horizontal rule separator
        parts.append("---")
        parts.append("")

        # Chapter heading
        type_label = _CHAPTER_TYPE_LABEL.get(
            chapter.chapter_type, chapter.chapter_type.value
        )
        parts.append(f"## {chapter.title}")
        parts.append(f"*{chapter.subtitle}*")
        parts.append("")

        # Period line
        if chapter.period_start and chapter.period_end:
            start = chapter.period_start.strftime("%d %b %Y")
            end = chapter.period_end.strftime("%d %b %Y")
            parts.append(f"**Period:** {start} — {end}")
            parts.append("")

        # Chapter-type badge
        parts.append(f"> 📖 **{type_label}**")
        parts.append("")

        # Narrative body
        if chapter.narrative:
            parts.append(chapter.narrative)
            parts.append("")

        # Patterns within the chapter
        for pattern in chapter.patterns:
            parts.append(self._render_pattern(pattern))
            parts.append("")

        # Key commits timeline
        if chapter.key_commits:
            parts.append(self._render_commit_timeline(chapter.key_commits))
            parts.append("")

        # Chapter stats sidebar
        if chapter.stats:
            parts.append(self._render_chapter_stats(chapter.stats))
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Pattern callout
    # ------------------------------------------------------------------

    def _render_pattern(self, pattern: Pattern) -> str:
        """Render a pattern as a blockquote callout box."""
        emoji = _SEVERITY_EMOJI.get(pattern.severity, "⚠️")
        severity_label = pattern.severity.value.upper()
        type_label = _PATTERN_TYPE_LABEL.get(
            pattern.pattern_type, pattern.pattern_type.value
        )

        lines: list[str] = [
            f"> {emoji} **{severity_label} — {type_label}**: {pattern.title}",
            f">",
            f"> {pattern.description}",
        ]

        # Affected files
        if pattern.affected_files:
            file_list = ", ".join(f"`{f}`" for f in pattern.affected_files[:5])
            suffix = (
                f" (+{len(pattern.affected_files) - 5} more)"
                if len(pattern.affected_files) > 5
                else ""
            )
            lines.append(f">")
            lines.append(f"> **Affected files:** {file_list}{suffix}")

        # Occurrence count
        if pattern.occurrence_count > 1:
            lines.append(f">")
            lines.append(
                f"> Observed **{pattern.occurrence_count}** times "
                f"({pattern.first_seen.strftime('%d %b %Y')} — "
                f"{pattern.last_seen.strftime('%d %b %Y')})"
            )

        # Confidence
        confidence_pct = pattern.confidence * 100
        lines.append(f">")
        lines.append(f"> Confidence: {confidence_pct:.0f}%")

        # Recommendation
        if pattern.recommendation:
            lines.append(f">")
            lines.append(f"> 💡 *{pattern.recommendation}*")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Forecast warning
    # ------------------------------------------------------------------

    def _render_forecast(self, forecast: Forecast) -> str:
        """Render a forecast as a warning blockquote."""
        emoji = _RISK_EMOJI.get(forecast.risk_level, "⚠️")
        risk_label = forecast.risk_level.value.upper()
        probability_pct = forecast.probability * 100

        lines: list[str] = [
            f"> {emoji} **{risk_label} RISK**: {forecast.title}",
            f">",
            f"> {forecast.description}",
            f">",
            f"> **Probability:** {probability_pct:.0f}% &nbsp;|&nbsp; "
            f"**Timeline:** {forecast.estimated_timeline}",
        ]

        # Indicators
        if forecast.indicators:
            lines.append(">")
            lines.append("> **Indicators:**")
            for ind in forecast.indicators:
                trend_arrow = {"rising": "↑", "stable": "→", "declining": "↓"}.get(
                    ind.trend, "?"
                )
                lines.append(
                    f"> - {ind.name}: {ind.current_value:.1f} "
                    f"(threshold {ind.threshold_value:.1f}) {trend_arrow} {ind.trend}"
                )

        # Historical precedent
        if forecast.historical_precedent:
            lines.append(">")
            lines.append(f"> 📜 *Historical precedent: {forecast.historical_precedent}*")

        # Recommendation
        if forecast.recommendation:
            lines.append(">")
            lines.append(f"> 💡 *{forecast.recommendation}*")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Forecasts section (all forecasts)
    # ------------------------------------------------------------------

    def _render_forecasts_section(self, memoir: Memoir) -> str:
        """Render a dedicated forecasts section."""
        parts: list[str] = [
            "---",
            "",
            "## 🔮 Risk Forecasts",
            "",
        ]

        for forecast in memoir.forecasts:
            parts.append(self._render_forecast(forecast))
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Commit timeline
    # ------------------------------------------------------------------

    def _render_commit_timeline(self, commits: list[CommitData]) -> str:
        """Render a chronological timeline of key commits."""
        parts: list[str] = [
            "### 📝 Key Commits",
            "",
        ]

        # Sort commits chronologically
        sorted_commits = sorted(commits, key=lambda c: c.date)

        for commit in sorted_commits:
            date_str = commit.date.strftime("%Y-%m-%d")
            # First line of message as subject
            subject = commit.message.split("\n", 1)[0].strip()
            if len(subject) > 80:
                subject = subject[:77] + "..."

            # Net change indicator
            net = commit.insertions - commit.deletions
            if net > 0:
                change_str = f"+{net}"
            elif net < 0:
                change_str = str(net)
            else:
                change_str = "0"

            parts.append(
                f"- `{commit.short_hash}` **{date_str}** — "
                f"{subject} *({change_str})*"
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Chapter stats sidebar
    # ------------------------------------------------------------------

    def _render_chapter_stats(self, stats: Dict[str, object]) -> str:
        """Render chapter-level statistics as a compact table."""
        if not stats:
            return ""

        parts: list[str] = [
            "### 📊 Chapter Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        for key, value in stats.items():
            # Format the key nicely
            label = key.replace("_", " ").replace("-", " ").title()
            display_val = f"{value:,}" if isinstance(value, int) else str(value)
            parts.append(f"| {label} | {display_val} |")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Global stats summary appendix
    # ------------------------------------------------------------------

    def _render_stats_summary(self, stats: GitStats) -> str:
        """Render aggregate git statistics as a markdown table."""
        parts: list[str] = [
            "---",
            "",
            "## 📊 Statistics Summary",
            "",
            "### Repository Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Commits | {stats.total_commits:,} |",
            f"| Unique Authors | {stats.unique_authors:,} |",
            f"| Files Changed | {stats.total_files_changed:,} |",
            f"| Lines Added | {stats.total_insertions:,} |",
            f"| Lines Removed | {stats.total_deletions:,} |",
            f"| Avg. Message Length | {stats.avg_message_length:.0f} chars |",
            f"| First Commit | {stats.first_commit_date.strftime('%d %b %Y')} |",
            f"| Latest Commit | {stats.last_commit_date.strftime('%d %b %Y')} |",
        ]

        # Authors
        if stats.author_names:
            parts.append("")
            parts.append("### Authors")
            parts.append("")
            for name in stats.author_names[:20]:
                parts.append(f"- {name}")
            if len(stats.author_names) > 20:
                parts.append(f"- … and {len(stats.author_names) - 20} more")

        # Commit frequency by day
        if stats.commit_frequency_by_day:
            parts.append("")
            parts.append("### Commits by Day of Week")
            parts.append("")
            parts.append("| Day | Commits |")
            parts.append("|-----|---------|")
            day_order = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday",
            ]
            for day in day_order:
                count = stats.commit_frequency_by_day.get(day, 0)
                if count > 0:
                    parts.append(f"| {day} | {count:,} |")

        # Hourly distribution (peak hours)
        if stats.hourly_distribution:
            parts.append("")
            parts.append("### Peak Hours")
            parts.append("")
            sorted_hours = sorted(
                stats.hourly_distribution.items(), key=lambda x: x[1], reverse=True
            )
            parts.append("| Hour | Commits |")
            parts.append("|------|---------|")
            for hour, count in sorted_hours[:8]:
                parts.append(f"| {hour:02d}:00 | {count:,} |")

        # Most changed files
        if stats.most_changed_files:
            parts.append("")
            parts.append("### Most Changed Files")
            parts.append("")
            parts.append("| File | Changes |")
            parts.append("|------|---------|")
            for file_path, count in stats.most_changed_files[:15]:
                # Truncate long file paths
                display_path = (
                    file_path if len(file_path) <= 50
                    else "…" + file_path[-49:]
                )
                parts.append(f"| `{display_path}` | {count:,} |")

        return "\n".join(parts)
