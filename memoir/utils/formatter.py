"""Text formatting utilities for the memoir project.

Provides human-readable formatting for dates, durations, numbers,
severity levels, and other display-oriented transformations used
in CLI output and generated narratives.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from memoir.models.forecast import RiskLevel
from memoir.models.pattern import Severity

# ---------------------------------------------------------------------------
# ANSI escape codes for terminal colouring
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"

_BRIGHT_RED = "\033[91m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_CYAN = "\033[96m"


# ---------------------------------------------------------------------------
# Date / time formatters
# ---------------------------------------------------------------------------

def format_date(dt: datetime) -> str:
    """Format a datetime as a short date string.

    Args:
        dt: The datetime to format.

    Returns:
        A string like ``"15 Mar 2024"``.
    """
    return dt.strftime("%d %b %Y")


def format_date_full(dt: datetime) -> str:
    """Format a datetime as a full date-and-time string.

    Args:
        dt: The datetime to format.

    Returns:
        A string like ``"15 March 2024, 14:30"``.
    """
    return dt.strftime("%d %B %Y, %H:%M")


def relative_time(dt: datetime) -> str:
    """Express a datetime as a human-friendly relative time.

    Produces strings like ``"2 weeks ago"``, ``"3 months ago"``,
    ``"just now"``.

    Args:
        dt: The datetime to compare against the current time.

    Returns:
        A relative time description.
    """
    now = datetime.now(tz=dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt
    seconds = int(delta.total_seconds())

    if seconds < 0:
        # Future timestamp — flip the sign and prefix "in"
        seconds = abs(seconds)
        if seconds < 60:
            return "in a few seconds"
        return _relative_future(seconds)

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = seconds // 60
        return pluralize(minutes, "minute") + " ago"
    if seconds < 86400:
        hours = seconds // 3600
        return pluralize(hours, "hour") + " ago"
    if seconds < 604800:
        days = seconds // 86400
        return pluralize(days, "day") + " ago"
    if seconds < 2592000:  # ~30 days
        weeks = seconds // 604800
        return pluralize(weeks, "week") + " ago"
    if seconds < 31536000:  # ~365 days
        months = seconds // 2592000
        return pluralize(months, "month") + " ago"
    years = seconds // 31536000
    return pluralize(years, "year") + " ago"


def _relative_future(seconds: int) -> str:
    """Build a future-facing relative time string."""
    if seconds < 3600:
        return "in " + pluralize(seconds // 60, "minute")
    if seconds < 86400:
        return "in " + pluralize(seconds // 3600, "hour")
    if seconds < 604800:
        return "in " + pluralize(seconds // 86400, "day")
    if seconds < 2592000:
        return "in " + pluralize(seconds // 604800, "week")
    if seconds < 31536000:
        return "in " + pluralize(seconds // 2592000, "month")
    return "in " + pluralize(seconds // 31536000, "year")


# ---------------------------------------------------------------------------
# Number / quantity formatters
# ---------------------------------------------------------------------------

def format_duration(hours: float) -> str:
    """Format a duration in hours to a compact string.

    Args:
        hours: Duration in decimal hours.

    Returns:
        A string like ``"3h 24m"`` or ``"45m"``.
    """
    if hours < 0:
        hours = abs(hours)
    total_minutes = int(round(hours * 60))
    h = total_minutes // 60
    m = total_minutes % 60
    if h > 0 and m > 0:
        return f"{h}h {m}m"
    if h > 0:
        return f"{h}h"
    return f"{m}m"


def format_number(n: int) -> str:
    """Format an integer with thousands separators.

    Args:
        n: The integer to format.

    Returns:
        A string like ``"1,234"``.
    """
    return f"{n:,}"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string.

    Args:
        value: The value to format (e.g. 0.853 → "85.3%").
        decimals: Number of decimal places.

    Returns:
        A string like ``"85.3%"``.
    """
    return f"{value * 100:.{decimals}f}%"


def format_file_size(lines: int) -> str:
    """Format a line count as a human-readable size string.

    Args:
        lines: Number of lines.

    Returns:
        A string like ``"1.2K lines"`` or ``"34 lines"``.
    """
    if lines < 1000:
        return pluralize(lines, "line")
    if lines < 1_000_000:
        k = lines / 1000
        if k == int(k):
            return f"{int(k)}K lines"
        return f"{k:.1f}K lines"
    m = lines / 1_000_000
    if m == int(m):
        return f"{int(m)}M lines"
    return f"{m:.1f}M lines"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with an ellipsis if it exceeds *max_length*.

    The ellipsis is included in the length budget.

    Args:
        text: The input text.
        max_length: Maximum output length including the ellipsis.

    Returns:
        The truncated string, or the original if short enough.
    """
    if len(text) <= max_length:
        return text
    ellipsis = "..."
    return text[: max_length - len(ellipsis)] + ellipsis


def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return a count with the appropriately pluralised noun.

    Args:
        count: The count.
        singular: Singular form of the noun.
        plural: Explicit plural form.  Defaults to ``singular + "s"``.

    Returns:
        A string like ``"3 files"`` or ``"1 file"``.
    """
    if count == 1:
        return f"1 {singular}"
    plural_form = plural if plural is not None else singular + "s"
    return f"{count} {plural_form}"


# ---------------------------------------------------------------------------
# Terminal colour helpers for severity / risk / scores
# ---------------------------------------------------------------------------

def severity_icon(severity: Severity) -> str:
    """Return a coloured terminal icon for a pattern severity level.

    Args:
        severity: The severity enum value.

    Returns:
        A coloured string such as ``"🟢 LOW"`` or ``"🔴 CRITICAL"``.
    """
    mapping = {
        Severity.LOW: f"{_GREEN}🟢 LOW{_RESET}",
        Severity.MEDIUM: f"{_YELLOW}🟡 MEDIUM{_RESET}",
        Severity.HIGH: f"{_RED}🟠 HIGH{_RESET}",
        Severity.CRITICAL: f"{_BRIGHT_RED}{_BOLD}🔴 CRITICAL{_RESET}",
    }
    return mapping.get(severity, str(severity))


def risk_icon(level: RiskLevel) -> str:
    """Return a coloured terminal icon for a forecast risk level.

    Args:
        level: The risk-level enum value.

    Returns:
        A coloured string such as ``"🟢 LOW"`` or ``"🔴 CRITICAL"``.
    """
    mapping = {
        RiskLevel.LOW: f"{_GREEN}🟢 LOW{_RESET}",
        RiskLevel.MODERATE: f"{_YELLOW}🟡 MODERATE{_RESET}",
        RiskLevel.HIGH: f"{_RED}🟠 HIGH{_RESET}",
        RiskLevel.CRITICAL: f"{_BRIGHT_RED}{_BOLD}🔴 CRITICAL{_RESET}",
    }
    return mapping.get(level, str(level))


def format_health_score(score: float) -> str:
    """Format a health score (0-100) with colour for terminal output.

    Args:
        score: Health score from 0 to 100.

    Returns:
        A coloured string like ``"85.3"`` (green) or ``"23.1"`` (red).
    """
    score = max(0.0, min(100.0, score))

    if score >= 80:
        colour = _BRIGHT_GREEN
    elif score >= 60:
        colour = _GREEN
    elif score >= 40:
        colour = _YELLOW
    elif score >= 20:
        colour = _RED
    else:
        colour = _BRIGHT_RED + _BOLD

    return f"{colour}{score:.1f}{_RESET}"


def format_trend(trend: str) -> str:
    """Format a trend direction with a coloured arrow for terminal output.

    Args:
        trend: One of ``"rising"``, ``"stable"``, or ``"declining"``.

    Returns:
        A coloured string like ``"↑ rising"`` or ``"→ stable"``.
    """
    trend_lower = trend.lower().strip()
    if trend_lower == "rising":
        return f"{_BRIGHT_RED}↑ rising{_RESET}"
    if trend_lower == "stable":
        return f"{_BRIGHT_CYAN}→ stable{_RESET}"
    if trend_lower == "declining":
        return f"{_BRIGHT_GREEN}↓ declining{_RESET}"
    # Fallback for unknown trend values
    return f"{_DIM}{trend}{_RESET}"
