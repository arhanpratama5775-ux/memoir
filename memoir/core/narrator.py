"""Narrative generation engine for the memoir project.

This module provides the :class:`Narrator` class which transforms raw commit
data, detected patterns, and risk forecasts into **human-readable narrative
chapters**.  It operates in two modes:

1. **Template mode** (default, no API key needed): Uses Jinja2 templates
   populated with real data to produce engaging, data-driven prose.
2. **AI mode** (optional, requires an OpenAI-compatible API key): Takes the
   template-mode draft and asks an LLM to enhance readability while keeping
   every fact accurate.

Narrative Principles
~~~~~~~~~~~~~~~~~~~~
1.  **Every claim must be backed by data.**  If the data says "47%
    after-hours commits", the narrative says exactly that -- no rounding,
    no embellishment, no fabrication.
2.  **Readable and engaging.**  Narratives read like a book about the
    codebase, not a dry metrics report.  They have personality, voice,
    and narrative arc.
3.  **Each chapter targets 300--800 words** in template mode.
4.  **AI enhancement is optional.**  The tool must work perfectly without
    any API key.  AI just makes it read better.
"""

from __future__ import annotations

import logging
import re as _re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment, StrictUndefined

from memoir.models.chapter import Chapter, ChapterType, Memoir
from memoir.models.commit_data import CommitData, GitStats
from memoir.models.forecast import Forecast, RiskLevel, RiskType
from memoir.models.pattern import Pattern, PatternType, Severity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrors git_analyzer / pattern_detector)
# ---------------------------------------------------------------------------

_AFTER_HOURS_START = 20
_AFTER_HOURS_END = 7

_VAGUE_RE = _re.compile(
    r"^(wip|fix|update|changes?|misc|stuff|cleanups?|tidy|tweaks?"
    r"|adjust|minor|fixes|updates|tmp|temp|hack|x)$",
    _re.IGNORECASE,
)
_VAGUE_LENGTH_THRESHOLD = 10

_FIX_KEYWORD_RE = _re.compile(
    r"\b(fix|bug|patch|hotfix|issue|close|closes|closed|resolve|resolves|resolved)\b",
    _re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Utility helpers (must be defined before Jinja2 env that uses them)
# ---------------------------------------------------------------------------


def _format_number(value: int) -> str:
    """Format a number with comma separators (e.g. 1,234)."""
    return f"{value:,}"


def _pct(value: float) -> str:
    """Format a float as a percentage string (e.g. 47%)."""
    return f"{value * 100:.0f}%"


def _fmt(value: float, spec: str = ".0f") -> str:
    """Format a float using Python's format specification."""
    return format(value, spec)


def _is_vague_message(message: str) -> bool:
    """Return True if a commit message is considered vague."""
    subject = message.strip().split("\n", 1)[0].strip()
    normalised = subject.lower().rstrip(".!? ")
    return len(subject) < _VAGUE_LENGTH_THRESHOLD or bool(
        _VAGUE_RE.match(normalised)
    )


def _pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return singular or plural form based on count."""
    if plural is None:
        plural = singular + "s"
    return singular if count == 1 else plural


def _span_text(days: int) -> str:
    """Convert a day span into a human-readable description."""
    if days < 0:
        return "0 days"
    if days < 30:
        return f"{days} {_pluralize(days, 'day')}"
    if days < 365:
        months = days // 30
        return f"about {months} {_pluralize(months, 'month')}"
    years = days // 365
    remaining = days % 365
    if remaining > 30:
        months = remaining // 30
        return f"about {years} {_pluralize(years, 'year')} and {months} {_pluralize(months, 'month')}"
    return f"about {years} {_pluralize(years, 'year')}"


# ---------------------------------------------------------------------------
# Jinja2 environment (inline templates only)
# ---------------------------------------------------------------------------

_JINJA_ENV = Environment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)
_JINJA_ENV.filters["format_number"] = _format_number
_JINJA_ENV.filters["pct"] = _pct
_JINJA_ENV.filters["fmt"] = _fmt

# ---------------------------------------------------------------------------
# Inline Jinja2 template strings
# ---------------------------------------------------------------------------

_PROLOGUE_TEMPLATE = r"""This story begins on {{ first_commit_date }}, when the first line of code was committed to {{ repo_name }}. Since that inaugural commit, {{ total_commits }} commits have shaped this codebase across {{ total_files }} files, weaving a narrative of creation, revision, and evolution.

Over a span of {{ project_span_days }} days ({{ project_span_text }}), {{ unique_authors }} {{ "contributor has" if unique_authors == 1 else "contributors have" }} collectively added {{ total_insertions|format_number }} lines and removed {{ total_deletions|format_number }} lines. The net result: {{ net_lines|format_number }} lines of living code, each one placed there for a reason.

{% if top_day %}The codebase seems to breathe most on {{ top_day }}s, with {{ top_day_count }} commits landing on that day alone. {% endif %}{% if peak_hour is not none %}The most popular commit hour is {{ peak_hour }}:00 — {{ "a time when most people are asleep" if peak_hour < 7 or peak_hour >= 22 else "the heart of the workday" if 10 <= peak_hour <= 17 else "the quiet edges of the day" }}. {% endif %}{% if avg_message_length > 0 %}On average, commit messages run {{ avg_message_length|fmt }} characters — {{ "terse and to the point" if avg_message_length < 30 else "reasonably descriptive" if avg_message_length < 80 else "detailed and communicative" }}. {% endif %}

{% if unique_authors == 1 %}This is a solo project — every commit, every decision, every late-night fix came from a single mind. That brings clarity of vision but also the weight of full responsibility. {% elif unique_authors <= 3 %}This is a tight-knit effort. With only {{ unique_authors }} contributors, every person's stamp on the codebase is unmistakable. {% else %}This is a {{ "medium-sized" if unique_authors <= 10 else "large" }} collaborative effort with {{ unique_authors }} contributors, each bringing their own patterns and preferences to the code. {% endif %}

{% if most_changed_files %}The files that have seen the most action are {% for f, c in most_changed_files[:5] %}{{ f }} ({{ c }} changes){{ ", " if not loop.last else "" }}{% endfor %}. These are the battlegrounds — the places where the most thinking, debating, and fixing has happened. {% endif %}

{% if weekly_activity_summary %}Activity has {{ weekly_activity_summary }}. {% endif %}{% if weekend_ratio > 0.15 %}Notably, {{ weekend_ratio|pct }} of commits were made on weekends — a sign that this project doesn't always respect the calendar. {% endif %}{% if after_hours_ratio > 0.20 %}And {{ after_hours_ratio|pct }} of commits were made outside conventional hours (before 7 AM or after 8 PM), which tells its own story. {% endif %}

This memoir tells that story — not through opinions, but through the data the codebase left behind.
"""

_PATTERN_CHAPTER_TEMPLATE = r"""{% if pattern_type_label == "Recurring Fix" %}There's a pattern here that keeps coming back like a bad penny. {% for p in patterns[:3] %}{{ p.title }}: {{ p.description }} {% if p.affected_files %}The file{{ "s" if p.affected_files|length > 1 else "" }} involved {{ "are" if p.affected_files|length > 1 else "is" }} {{ p.affected_files|join(", ") }}. {% endif %}It has been observed {{ p.occurrence_count }} time{{ "s" if p.occurrence_count != 1 else "" }} between {{ p.first_seen.strftime("%B %d, %Y") }} and {{ p.last_seen.strftime("%B %d, %Y") }} with {{ p.severity.value }} severity and {{ p.confidence|pct }} confidence.{% if not loop.last %}

{% endif %}{% endfor %}

Each fix addressed a different symptom, but the underlying issue may remain. {% if any_critical %}Some of these recurring fixes touch critical paths — the cost of continued patching likely exceeds the cost of a proper solution. {% endif %}{% if max_occurrences >= 5 %}With some files fixed {{ max_occurrences }} times, the message is clear: the root cause is still at large. {% endif %}{% if recommendations %}The data suggests: {{ recommendations[0] }} {% endif %}

{% elif pattern_type_label == "Technical Debt" %}The codebase is carrying weight that was never meant to be permanent. {% for p in patterns[:3] %}{{ p.description }} {% if p.affected_files %}The area{{ "s" if p.affected_files|length > 1 else "" }} most affected: {{ p.affected_files[:5]|join(", ") }}. {% endif %}This has been building since {{ p.first_seen.strftime("%B %d, %Y") }}, with {{ p.occurrence_count }} indicator{{ "s" if p.occurrence_count != 1 else "" }} detected ({{ p.severity.value }} severity, {{ p.confidence|pct }} confidence).{% if not loop.last %}

{% endif %}{% endfor %}

{% if total_occurrences >= 10 %}With {{ total_occurrences }} signals accumulated, the debt is no longer a nuisance — it's becoming a structural concern. {% endif %}Technical debt is like financial debt: the longer it compounds, the more expensive it becomes to resolve. {% if recommendations %}The path forward: {{ recommendations[0] }} {% endif %}

{% elif pattern_type_label == "Learning Curve" %}Every developer who joins a project goes through a learning phase — and the commit log captures it with remarkable clarity. {% for p in patterns[:3] %}{{ p.description }} This pattern spans from {{ p.first_seen.strftime("%B %d, %Y") }} to {{ p.last_seen.strftime("%B %d, %Y") }}, with {{ p.occurrence_count }} occurrence{{ "s" if p.occurrence_count != 1 else "" }} ({{ p.severity.value }} severity, {{ p.confidence|pct }} confidence).{% if not loop.last %}

{% endif %}{% endfor %}

{% if recommendations %}What helps: {{ recommendations[0] }} {% endif %}The learning curve isn't a flaw — it's a feature of every growing project. The question is whether the project is making the climb easier over time.

{% elif pattern_type_label == "Burnout Indicator" %}The commit log is a diary, and sometimes it tells a story of exhaustion. {% for p in patterns[:3] %}{{ p.description }} First flagged on {{ p.first_seen.strftime("%B %d, %Y") }}, last seen on {{ p.last_seen.strftime("%B %d, %Y") }}, with {{ p.occurrence_count }} indicator{{ "s" if p.occurrence_count != 1 else "" }} ({{ p.severity.value }} severity, {{ p.confidence|pct }} confidence).{% if not loop.last %}

{% endif %}{% endfor %}

These are signals derived from commit patterns — not self-reporting — so they may reflect project pressures rather than personal state. But the data is what it is. {% if any_critical_or_high %}The severity here is concerning. {% endif %}{% if recommendations %}The recommendation: {{ recommendations[0] }} {% endif %}

{% elif pattern_type_label == "Code Ownership" %}Knowledge concentration is a silent risk. {% for p in patterns[:3] %}{{ p.description }} {% if p.affected_files %}Files under concentrated ownership: {{ p.affected_files[:5]|join(", ") }}. {% endif %}Observed between {{ p.first_seen.strftime("%B %d, %Y") }} and {{ p.last_seen.strftime("%B %d, %Y") }} ({{ p.occurrence_count }} occurrence{{ "s" if p.occurrence_count != 1 else "" }}, {{ p.severity.value }} severity, {{ p.confidence|pct }} confidence).{% if not loop.last %}

{% endif %}{% endfor %}

When one person holds the keys to a critical module, a single departure can paralyze the team. {% if recommendations %}Action: {{ recommendations[0] }} {% endif %}

{% elif pattern_type_label == "Anti-Pattern" %}Some patterns aren't just suboptimal — they're actively harmful. {% for p in patterns[:3] %}{{ p.description }} {% if p.affected_files %}Affected files: {{ p.affected_files[:5]|join(", ") }}. {% endif %}Detected between {{ p.first_seen.strftime("%B %d, %Y") }} and {{ p.last_seen.strftime("%B %d, %Y") }} with {{ p.occurrence_count }} occurrence{{ "s" if p.occurrence_count != 1 else "" }} ({{ p.severity.value }} severity, {{ p.confidence|pct }} confidence).{% if not loop.last %}

{% endif %}{% endfor %}

Anti-patterns are the codebase's way of asking for intervention. {% if recommendations %}The fix: {{ recommendations[0] }} {% endif %}

{% elif pattern_type_label == "Irregular Hours" %}The clock tells a story that timesheets never will. {% for p in patterns[:3] %}{{ p.description }} {% if p.affected_files %}Files most often touched during irregular hours: {{ p.affected_files[:5]|join(", ") }}. {% endif %}This pattern spans from {{ p.first_seen.strftime("%B %d, %Y") }} to {{ p.last_seen.strftime("%B %d, %Y") }} ({{ p.occurrence_count }} occurrence{{ "s" if p.occurrence_count != 1 else "" }}, {{ p.severity.value }} severity, {{ p.confidence|pct }} confidence).{% if not loop.last %}

{% endif %}{% endfor %}

Commits at odd hours aren't always a problem — some people just prefer to code at night. But when irregular hours become the norm rather than the exception, it's worth asking why. {% if recommendations %}Suggestion: {{ recommendations[0] }} {% endif %}

{% elif pattern_type_label == "High Churn" %}Some files just can't sit still. {% for p in patterns[:3] %}{{ p.description }} {% if p.affected_files %}The restless file{{ "s" if p.affected_files|length > 1 else "" }}: {{ p.affected_files[:5]|join(", ") }}. {% endif %}Churn observed between {{ p.first_seen.strftime("%B %d, %Y") }} and {{ p.last_seen.strftime("%B %d, %Y") }} with {{ p.occurrence_count }} occurrence{{ "s" if p.occurrence_count != 1 else "" }} ({{ p.severity.value }} severity, {{ p.confidence|pct }} confidence).{% if not loop.last %}

{% endif %}{% endfor %}

High churn without net growth is the codebase equivalent of treading water — a lot of effort, not much forward motion. {% if recommendations %}What to do: {{ recommendations[0] }} {% endif %}

{% else %}A pattern has emerged that deserves attention. {% for p in patterns[:3] %}{{ p.title }}: {{ p.description }} {% if p.affected_files %}Affected files: {{ p.affected_files[:5]|join(", ") }}. {% endif %}Observed {{ p.occurrence_count }} time{{ "s" if p.occurrence_count != 1 else "" }} between {{ p.first_seen.strftime("%B %d, %Y") }} and {{ p.last_seen.strftime("%B %d, %Y") }} ({{ p.severity.value }} severity, {{ p.confidence|pct }} confidence).{% if not loop.last %}

{% endif %}{% endfor %}

{% if recommendations %}Recommendation: {{ recommendations[0] }} {% endif %}
{% endif %}"""

_MILESTONE_TEMPLATE = r"""Every project has its landmarks — moments that define the trajectory of everything that follows. For {{ repo_name }}, the data points to several.

**The Beginning.** On {{ first_commit_date }}, the first commit landed: "{{ first_commit_message }}". {% if first_commit_author %}It was authored by {{ first_commit_author }}. {% endif %}That first commit touched {{ first_commit_files }} file{{ "s" if first_commit_files != 1 else "" }} with {{ first_commit_insertions }} addition{{ "s" if first_commit_insertions != 1 else "" }} and {{ first_commit_deletions }} deletion{{ "s" if first_commit_deletions != 1 else "" }}. Every line of code that exists today traces its ancestry back to this moment.

{% if biggest_commit %}**The Largest Change.** The single biggest commit by volume arrived on {{ biggest_commit_date }}: "{{ biggest_commit_message }}" ({{ biggest_commit_hash }}). It added {{ biggest_commit_insertions|format_number }} lines and removed {{ biggest_commit_deletions|format_number }} lines across {{ biggest_commit_files }} file{{ "s" if biggest_commit_files != 1 else "" }}. {% if biggest_commit_author %}This was the work of {{ biggest_commit_author }}. {% endif %}Whether it was a major feature, a sweeping refactor, or a framework migration, this commit reshaped the codebase in a single stroke. {% endif %}

{% if most_deletions_commit %}**The Great Pruning.** On {{ most_deletions_date }}, {{ most_deletions_author }} committed "{{ most_deletions_message }}" ({{ most_deletions_hash }}), which removed {{ most_deletions_count|format_number }} lines — the most deletions in any single commit. Sometimes the most important work is knowing what to remove. {% endif %}

{% if longest_streak %}**The Longest Streak.** The longest active streak ran for {{ longest_streak_days }} consecutive days, from {{ longest_streak_start }} to {{ longest_streak_end }}, producing {{ longest_streak_commits }} commits. Sustained daily commits indicate a period of focused momentum — the project was clearly the top priority during this window. {% endif %}

{% if tagged_commits %}**Version Markers.** {% for tag_info in tagged_commits[:5] %}Version {{ tag_info.tag }} was tagged on {{ tag_info.date }} in commit {{ tag_info.hash }} ("{{ tag_info.message }}").{% if not loop.last %} {% endif %}{% endfor %} Each tag represents a moment the team declared "this is ready." {% endif %}

{% if merge_commits_count > 0 %}**Merges.** There {{ "has been 1 merge" if merge_commits_count == 1 else "have been " ~ merge_commits_count ~ " merges" }} across the project's history, suggesting {% if merge_commits_count <= 5 %}a relatively linear development flow with few parallel branches{% elif merge_commits_count <= 30 %}moderate use of branching and integration{% else %}heavy use of parallel branches and frequent integration{% endif %}. {% endif %}

{% if silence_periods %}**The Silences.** Not all milestones are about activity. {% for sp in silence_periods[:3] %}There was a {{ sp.days }}-day gap between commits starting {{ sp.start }}{% if not loop.last %}; {% else %}.{% endif %}{% endfor %} Silence in a commit log can mean many things — a well-earned break, a strategic pause, or a period of planning before the next surge. {% endif %}

These milestones are the skeleton of the project's story. The commits between them fill in the flesh.
"""

_CRISIS_TEMPLATE = r"""{% if crisis_patterns or after_hours_evidence or vague_evidence or silence_evidence %}Not every chapter in a project's history is about progress. Some are about struggle — periods when the data tells a story of pressure, fatigue, or difficulty that no retrospective can erase.

{% if after_hours_evidence %}**The Late-Night Sessions.** {{ after_hours_evidence }} These aren't just numbers — behind each after-midnight commit is someone who chose code over sleep. Sometimes that's passion; sometimes it's pressure. The commit log doesn't distinguish, but the pattern does.

{% endif %}{% if vague_evidence %}**When Messages Went Quiet.** {{ vague_evidence }} Vague commit messages often correlate with rushing — the work gets done, but the documentation of intent is sacrificed. Over time, this erodes the project's institutional memory.

{% endif %}{% if silence_evidence %}**The Gaps.** {{ silence_evidence }} Long silences between commits can signal burnout, context-switching, or simply life happening outside the codebase. Whatever the cause, the gap is real and measurable.

{% endif %}{% if crisis_patterns %}**Detected Crisis Patterns.** {% for p in crisis_patterns %}{{ p.title }}: {{ p.description }} ({{ p.severity.value }} severity, {{ p.confidence|pct }} confidence, first seen {{ p.first_seen.strftime("%B %d, %Y") }}).{% if not loop.last %} {% endif %}{% endfor %}

{% endif %}{% if any_critical_crisis %}The severity of some of these patterns demands attention. A codebase in crisis leaves clues, and the commit log is a witness that never forgets. {% endif %}

{% if fix_storm_evidence %}**The Bug Storms.** {{ fix_storm_evidence }} When fix commits cluster together, it usually means a change was made that had wider consequences than expected — or that a long-deferred problem finally demanded attention.

{% endif %}{% if recommendations %}The data-driven path forward: {% for r in recommendations[:3] %}{{ r }}{% if not loop.last %} {% endif %}{% endfor %} {% endif %}
{% else %}This chapter could have been about crises — burnout, bug storms, and difficult periods — but the data doesn't show them. The commit log reveals no after-hours emergencies, no suspicious clusters of vague messages, no extended silences that suggest exhaustion. Either this project has been remarkably well-managed, or the pressure hasn't left its mark on the git history. Either way, the absence of crisis patterns is itself a data point worth noting.
{% endif %}"""

_CURRENT_STATE_TEMPLATE = r"""Where does {{ repo_name }} stand today? The most recent data paints a picture of {% if recent_activity_level == "high" %}a project in active development, with strong momentum and frequent changes{% elif recent_activity_level == "moderate" %}a project under steady maintenance — not racing, but not idle either{% elif recent_activity_level == "low" %}a project that has slowed down significantly from its peak{% else %}a project with minimal recent activity{% endif %}.

**Recent Activity.** The last {{ recent_commit_count }} commits span from {{ recent_first_date }} to {{ recent_last_date }}. {% if recent_commit_count > 0 %}During this period, {{ recent_insertions|format_number }} lines were added and {{ recent_deletions|format_number }} lines were removed, for a net change of {{ recent_net_lines|format_number }} lines. {% if recent_authors %}Active contributor{{ "s" if recent_authors|length > 1 else "" }}: {{ recent_authors|join(", ") }}. {% endif %}{% endif %}

**Active Patterns.** {% if active_patterns %}Currently, {{ active_patterns|length }} pattern{{ "s" if active_patterns|length > 1 else "" }} remain active: {% for p in active_patterns[:5] %}{{ p.title }} ({{ p.severity.value }} severity){% if not loop.last %}, {% else %}.{% endif %}{% endfor %} {% if any_severe_active %}Some of these are severe enough to warrant immediate attention. {% endif %}{% else %}No active patterns have been detected — a positive sign for the project's current health. {% endif %}

**Codebase Scale.** Across its lifetime, {{ repo_name }} has accumulated {{ total_insertions|format_number }} line insertions and {{ total_deletions|format_number }} line deletions across {{ total_files }} files. {% if net_lines > 0 %}The codebase has grown by a net {{ net_lines|format_number }} lines. {% elif net_lines < 0 %}The codebase has actually shrunk by {{ abs_net_lines|format_number }} lines — more code has been removed than added, which could indicate aggressive refactoring or pruning. {% else %}Additions and deletions are roughly balanced. {% endif %}

**Contributor Health.** {% if unique_authors == 1 %}This remains a single-contributor project. All knowledge resides with one person, which is the very definition of a bus-factor risk. {% elif unique_authors <= 3 %}With {{ unique_authors }} contributors, knowledge is concentrated but shared. The bus factor is improved but still fragile. {% else %}{{ unique_authors }} contributors have touched this codebase, distributing knowledge more broadly. {% endif %}{% if recent_author_count < unique_authors %}However, only {{ recent_author_count }} contributor{{ "s" if recent_author_count != 1 else "" }} have been active recently, meaning {{ quiet_count }} {{ "has" if quiet_count == 1 else "have" }} gone quiet. {% endif %}

{% if most_changed_files_recent %}**Hot Spots.** The files seeing the most recent changes are {% for f, c in most_changed_files_recent[:5] %}{{ f }} ({{ c }} changes){{ ", " if not loop.last else "" }}{% endfor %}. These are the areas where the project's current energy is focused. {% endif %}

The story so far has been written in commits. The next chapter is still being written.
"""

_FORECAST_TEMPLATE = r"""If history is any guide — and in version control, it usually is — the patterns of the past offer a window into the future. Based on the data in {{ repo_name }}'s commit history, here is what the trends suggest.

{% if forecasts %}{% for f in forecasts %}**{{ f.title }}** ({{ f.risk_level.value }} risk, {{ f.probability|pct }} probability). {{ f.description }} {% if f.estimated_timeline and f.estimated_timeline != "unknown" %}Estimated timeline: {{ f.estimated_timeline }}. {% endif %}{% if f.indicators %}Key indicators: {% for ind in f.indicators %}{{ ind.name }} is {{ ind.trend }} (current: {{ ind.current_value }}, threshold: {{ ind.threshold_value }}){% if not loop.last %}; {% else %}.{% endif %}{% endfor %} {% endif %}{% if f.historical_precedent %}Historical precedent: {{ f.historical_precedent }} {% endif %}Recommendation: {{ f.recommendation }}

{% endfor %}{% else %}No significant risk forecasts have been generated from the available data. This could mean the project is in good shape — or that the data is too sparse to detect emerging trends. Either way, continued monitoring is always wise.
{% endif %}

{% if health_score is not none %}**Overall Health Score: {{ health_score|fmt }}/100.** {% if health_score >= 80 %}This project is in excellent shape — the data shows a well-maintained codebase with few warning signs. {% elif health_score >= 60 %}The project is in reasonable health, but there are areas that deserve attention before they become problems. {% elif health_score >= 40 %}Significant concerns are present. The health score reflects real patterns in the data that should not be ignored. {% else %}The health score is concerning. Multiple risk factors are present, and the data suggests the project needs intervention to avoid further deterioration. {% endif %}{% endif %}

The forecast is not destiny — it's a probability derived from past behavior. The best way to change the forecast is to change the patterns that drive it.
"""

# ---------------------------------------------------------------------------
# Pattern-type display labels
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Health score computation
# ---------------------------------------------------------------------------


def _compute_overall_health_score(
    patterns: List[Pattern],
    forecasts: List[Forecast],
    commits: List[CommitData],
    git_stats: GitStats,
) -> float:
    """Compute an overall project health score from 0 to 100.

    Starts at 100 and deducts points based on:
    - Pattern severity and confidence (capped total deduction)
    - Forecast risk levels and probabilities (capped total deduction)
    - Work-life balance indicators
    - Message quality
    """
    score = 100.0

    # Deductions from patterns (capped to avoid over-penalising)
    # Use logarithmic scaling so more patterns = worse but not catastrophic
    severity_deduction: Dict[Severity, float] = {
        Severity.CRITICAL: 4.0,
        Severity.HIGH: 2.0,
        Severity.MEDIUM: 1.0,
        Severity.LOW: 0.3,
    }
    pattern_penalty = 0.0
    for pattern in patterns:
        deduction = severity_deduction.get(pattern.severity, 0.0) * pattern.confidence
        pattern_penalty += deduction
    # Cap pattern penalty with diminishing returns
    pattern_penalty = min(pattern_penalty, 25.0)
    score -= pattern_penalty

    # Deductions from forecasts (capped)
    risk_deduction: Dict[RiskLevel, float] = {
        RiskLevel.CRITICAL: 8.0,
        RiskLevel.HIGH: 5.0,
        RiskLevel.MODERATE: 3.0,
        RiskLevel.LOW: 1.0,
    }
    forecast_penalty = 0.0
    for forecast in forecasts:
        deduction = risk_deduction.get(forecast.risk_level, 0.0) * forecast.probability
        forecast_penalty += deduction
    # Cap forecast penalty
    forecast_penalty = min(forecast_penalty, 30.0)
    score -= forecast_penalty

    # After-hours penalty
    if commits:
        after_hours = sum(
            1
            for c in commits
            if c.date.hour < _AFTER_HOURS_END or c.date.hour >= _AFTER_HOURS_START
        )
        after_hours_ratio = after_hours / len(commits)
        if after_hours_ratio > 0.30:
            score -= (after_hours_ratio - 0.30) * 20

    # Weekend penalty
    if commits:
        weekend = sum(1 for c in commits if c.date.weekday() >= 5)
        weekend_ratio = weekend / len(commits)
        if weekend_ratio > 0.20:
            score -= (weekend_ratio - 0.20) * 10

    # Vague message penalty
    if commits:
        vague = sum(1 for c in commits if _is_vague_message(c.message))
        vague_ratio = vague / len(commits)
        if vague_ratio > 0.30:
            score -= (vague_ratio - 0.30) * 15

    # Bus factor penalty
    if git_stats.unique_authors == 1:
        score -= 5.0

    return max(0.0, min(100.0, round(score, 1)))


# ---------------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------------


class Narrator:
    """Generates human-readable narrative chapters from commit data, patterns,
    and forecasts.

    Works in **two modes**:

    1. *Template mode* (default): Uses Jinja2 templates with real data to
       produce readable, data-driven chapters.
    2. *AI mode* (optional): If ``ai_config`` is provided with an ``api_key``,
       the draft narrative from template mode is sent to an LLM for
       readability enhancement while preserving factual accuracy.

    Parameters
    ----------
    commits:
        Complete list of commits for the analysed range.
    git_stats:
        Aggregate repository statistics.
    patterns:
        Patterns detected by :class:`PatternDetector`.
    forecasts:
        Risk forecasts from :class:`CrisisForecaster`.
    repo_name:
        Human-readable name of the repository.
    ai_config:
        Optional dict with keys ``api_key``, ``model`` (default
        ``gpt-4o-mini``), and ``base_url`` for non-OpenAI providers.
    """

    def __init__(
        self,
        commits: List[CommitData],
        git_stats: GitStats,
        patterns: List[Pattern],
        forecasts: List[Forecast],
        repo_name: str,
        ai_config: Optional[Dict[str, str]] = None,
    ) -> None:
        self.commits = sorted(commits, key=lambda c: c.date)
        self.git_stats = git_stats
        self.patterns = patterns
        self.forecasts = forecasts
        self.repo_name = repo_name
        self.ai_config = ai_config or {}

        # Pre-computed indices (lazily built)
        self._sorted_commits: Optional[List[CommitData]] = None

        logger.info(
            "Narrator initialised with %d commits, %d patterns, "
            "%d forecasts for repo '%s'",
            len(self.commits),
            len(self.patterns) if self.patterns else 0,
            len(self.forecasts) if self.forecasts else 0,
            self.repo_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sorted_commits(self) -> List[CommitData]:
        """Commits sorted chronologically (oldest first)."""
        if self._sorted_commits is None:
            self._sorted_commits = sorted(self.commits, key=lambda c: c.date)
        return self._sorted_commits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_memoir(self) -> Memoir:
        """Generate a complete memoir with all chapters.

        Returns
        -------
        Memoir
            The full autobiography document with all narrative chapters,
            forecasts, and health score.
        """
        chapters: List[Chapter] = []
        order = 0

        # Prologue
        prologue = self.generate_prologue()
        prologue.order = order
        chapters.append(prologue)
        order += 1

        # Pattern chapters
        for pattern_chapter in self.generate_pattern_chapters():
            pattern_chapter.order = order
            chapters.append(pattern_chapter)
            order += 1

        # Milestone chapter
        milestone = self.generate_milestone_chapter()
        milestone.order = order
        chapters.append(milestone)
        order += 1

        # Crisis chapter
        crisis = self.generate_crisis_chapter()
        crisis.order = order
        chapters.append(crisis)
        order += 1

        # Current state chapter
        current = self.generate_current_state_chapter()
        current.order = order
        chapters.append(current)
        order += 1

        # Forecast chapter
        forecast_ch = self.generate_forecast_chapter()
        forecast_ch.order = order
        chapters.append(forecast_ch)
        order += 1

        health_score = _compute_overall_health_score(
            self.patterns, self.forecasts, self.commits, self.git_stats
        )

        memoir = Memoir(
            repo_name=self.repo_name,
            repo_path="",
            generated_at=datetime.now(),
            git_stats=self.git_stats,
            chapters=chapters,
            forecasts=self.forecasts,
            overall_health_score=health_score,
        )

        logger.info(
            "Generated memoir with %d chapters, health score %.1f",
            len(chapters),
            health_score,
        )
        return memoir

    def generate_prologue(self) -> Chapter:
        """Generate the opening chapter: project overview, timeline, key stats.

        Returns
        -------
        Chapter
            The prologue chapter.
        """
        commits = self.sorted_commits
        stats = self.git_stats

        span_days = (stats.last_commit_date - stats.first_commit_date).days

        # Top day of the week
        if stats.commit_frequency_by_day:
            top_day_name, top_day_count = max(
                stats.commit_frequency_by_day.items(), key=lambda x: x[1]
            )
        else:
            top_day_name, top_day_count = None, 0

        # Peak hour
        if stats.hourly_distribution:
            peak_hour, peak_hour_count = max(
                stats.hourly_distribution.items(), key=lambda x: x[1]
            )
        else:
            peak_hour, peak_hour_count = None, 0

        # Weekly activity summary
        weekly_activity_summary = self._compute_weekly_activity_summary()

        # Weekend and after-hours ratios
        weekend_ratio = 0.0
        after_hours_ratio = 0.0
        if commits:
            weekend_count = sum(1 for c in commits if c.date.weekday() >= 5)
            weekend_ratio = weekend_count / len(commits)
            after_hours_count = sum(
                1
                for c in commits
                if c.date.hour < _AFTER_HOURS_END
                or c.date.hour >= _AFTER_HOURS_START
            )
            after_hours_ratio = after_hours_count / len(commits)

        context: Dict[str, Any] = {
            "repo_name": self.repo_name,
            "first_commit_date": stats.first_commit_date.strftime("%B %d, %Y"),
            "total_commits": stats.total_commits,
            "total_files": stats.total_files_changed,
            "project_span_days": span_days,
            "project_span_text": _span_text(span_days),
            "unique_authors": stats.unique_authors,
            "total_insertions": stats.total_insertions,
            "total_deletions": stats.total_deletions,
            "net_lines": stats.total_insertions - stats.total_deletions,
            "top_day": top_day_name,
            "top_day_count": top_day_count,
            "peak_hour": peak_hour,
            "avg_message_length": stats.avg_message_length,
            "most_changed_files": stats.most_changed_files[:5],
            "weekly_activity_summary": weekly_activity_summary,
            "weekend_ratio": weekend_ratio,
            "after_hours_ratio": after_hours_ratio,
        }

        narrative = self._render_template(_PROLOGUE_TEMPLATE, context)
        narrative = self._ai_enhance(narrative, context)

        return Chapter(
            id=str(uuid.uuid4()),
            chapter_type=ChapterType.PROLOGUE,
            title="The Beginning",
            subtitle=f"A story of {self.repo_name}",
            period_start=stats.first_commit_date,
            period_end=stats.last_commit_date,
            narrative=narrative,
            patterns=[],
            key_commits=commits[:5] if commits else [],
            stats={
                "total_commits": stats.total_commits,
                "span_days": span_days,
                "unique_authors": stats.unique_authors,
            },
            order=0,
        )

    def generate_pattern_chapters(self) -> List[Chapter]:
        """Generate one chapter per significant pattern type.

        Patterns are grouped by type.  Only pattern types with at least
        one detected pattern produce a chapter.

        Returns
        -------
        list[Chapter]
            One chapter per pattern type that has detected patterns.
        """
        chapters: List[Chapter] = []

        # Group patterns by type
        by_type: Dict[PatternType, List[Pattern]] = defaultdict(list)
        for pattern in self.patterns:
            by_type[pattern.pattern_type].append(pattern)

        # Sort pattern types by worst severity first
        severity_rank = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        sorted_types = sorted(
            by_type.keys(),
            key=lambda pt: min(
                severity_rank.get(p.severity, 99) for p in by_type[pt]
            ),
        )

        for pattern_type in sorted_types:
            group = by_type[pattern_type]
            label = _PATTERN_TYPE_LABELS.get(pattern_type, pattern_type.value)

            any_critical = any(
                p.severity in (Severity.CRITICAL, Severity.HIGH) for p in group
            )
            any_critical_or_high = any(
                p.severity in (Severity.CRITICAL, Severity.HIGH) for p in group
            )
            max_occurrences = max(p.occurrence_count for p in group) if group else 0
            total_occurrences = sum(p.occurrence_count for p in group)
            recommendations = [
                p.recommendation for p in group if p.recommendation
            ]

            # Time span of the chapter
            all_dates = []
            for p in group:
                all_dates.extend([p.first_seen, p.last_seen])
            period_start = min(all_dates) if all_dates else None
            period_end = max(all_dates) if all_dates else None

            context: Dict[str, Any] = {
                "pattern_type_label": label,
                "patterns": group,
                "any_critical": any_critical,
                "any_critical_or_high": any_critical_or_high,
                "max_occurrences": max_occurrences,
                "total_occurrences": total_occurrences,
                "recommendations": recommendations,
            }

            narrative = self._render_template(_PATTERN_CHAPTER_TEMPLATE, context)
            narrative = self._ai_enhance(narrative, context)

            chapter = Chapter(
                id=str(uuid.uuid4()),
                chapter_type=ChapterType.PATTERN,
                title=f"The {label} Pattern",
                subtitle=f"{len(group)} {_pluralize(len(group), 'instance')} detected",
                period_start=period_start,
                period_end=period_end,
                narrative=narrative,
                patterns=group,
                key_commits=[],
                stats={
                    "pattern_type": pattern_type.value,
                    "count": len(group),
                    "total_occurrences": total_occurrences,
                },
                order=0,  # Will be set by generate_memoir
            )
            chapters.append(chapter)

        return chapters

    def generate_milestone_chapter(self) -> Chapter:
        """Generate a chapter about key milestones.

        Covers: first commit, biggest changes, longest streaks, tagged
        commits, merge events, and notable silence periods.

        Returns
        -------
        Chapter
            The milestone chapter.
        """
        commits = self.sorted_commits
        stats = self.git_stats

        # First commit
        first_commit = commits[0] if commits else None
        first_commit_date = (
            first_commit.date.strftime("%B %d, %Y") if first_commit else "unknown"
        )
        first_commit_message = (
            first_commit.message.split("\n")[0][:120] if first_commit else "unknown"
        )
        first_commit_author = first_commit.author_name if first_commit else None
        first_commit_files = len(first_commit.files_changed) if first_commit else 0
        first_commit_insertions = first_commit.insertions if first_commit else 0
        first_commit_deletions = first_commit.deletions if first_commit else 0

        # Biggest commit by total changes
        biggest_commit = (
            max(commits, key=lambda c: c.insertions + c.deletions)
            if commits
            else None
        )
        most_deletions_commit = (
            max(commits, key=lambda c: c.deletions) if commits else None
        )

        # Longest streak
        longest_streak = self._compute_longest_streak()

        # Tagged commits
        tagged_commits = []
        for c in commits:
            if c.tags:
                for tag in c.tags:
                    tagged_commits.append(
                        {
                            "tag": tag,
                            "date": c.date.strftime("%B %d, %Y"),
                            "hash": c.short_hash,
                            "message": c.message.split("\n")[0][:80],
                        }
                    )

        # Merge commits
        merge_commits_count = sum(1 for c in commits if c.is_merge)

        # Silence periods (7+ day gaps)
        silence_periods = self._find_silence_periods(min_days=7)

        context: Dict[str, Any] = {
            "repo_name": self.repo_name,
            "first_commit_date": first_commit_date,
            "first_commit_message": first_commit_message,
            "first_commit_author": first_commit_author,
            "first_commit_files": first_commit_files,
            "first_commit_insertions": first_commit_insertions,
            "first_commit_deletions": first_commit_deletions,
            "biggest_commit": biggest_commit,
            "biggest_commit_date": (
                biggest_commit.date.strftime("%B %d, %Y")
                if biggest_commit
                else None
            ),
            "biggest_commit_message": (
                biggest_commit.message.split("\n")[0][:120]
                if biggest_commit
                else None
            ),
            "biggest_commit_hash": (
                biggest_commit.short_hash if biggest_commit else None
            ),
            "biggest_commit_insertions": (
                biggest_commit.insertions if biggest_commit else 0
            ),
            "biggest_commit_deletions": (
                biggest_commit.deletions if biggest_commit else 0
            ),
            "biggest_commit_files": (
                len(biggest_commit.files_changed) if biggest_commit else 0
            ),
            "biggest_commit_author": (
                biggest_commit.author_name if biggest_commit else None
            ),
            "most_deletions_commit": most_deletions_commit,
            "most_deletions_date": (
                most_deletions_commit.date.strftime("%B %d, %Y")
                if most_deletions_commit
                else None
            ),
            "most_deletions_author": (
                most_deletions_commit.author_name
                if most_deletions_commit
                else None
            ),
            "most_deletions_message": (
                most_deletions_commit.message.split("\n")[0][:120]
                if most_deletions_commit
                else None
            ),
            "most_deletions_hash": (
                most_deletions_commit.short_hash
                if most_deletions_commit
                else None
            ),
            "most_deletions_count": (
                most_deletions_commit.deletions if most_deletions_commit else 0
            ),
            "longest_streak": longest_streak,
            "longest_streak_days": (
                longest_streak.get("days", 0) if longest_streak else 0
            ),
            "longest_streak_start": (
                longest_streak.get("start", "") if longest_streak else ""
            ),
            "longest_streak_end": (
                longest_streak.get("end", "") if longest_streak else ""
            ),
            "longest_streak_commits": (
                longest_streak.get("commits", 0) if longest_streak else 0
            ),
            "tagged_commits": tagged_commits,
            "merge_commits_count": merge_commits_count,
            "silence_periods": silence_periods,
        }

        narrative = self._render_template(_MILESTONE_TEMPLATE, context)
        narrative = self._ai_enhance(narrative, context)

        key_commits = []
        if first_commit:
            key_commits.append(first_commit)
        if biggest_commit and biggest_commit != first_commit:
            key_commits.append(biggest_commit)
        if most_deletions_commit and most_deletions_commit not in key_commits:
            key_commits.append(most_deletions_commit)

        return Chapter(
            id=str(uuid.uuid4()),
            chapter_type=ChapterType.MILESTONE,
            title="Landmark Moments",
            subtitle="First commits, biggest changes, and notable events",
            period_start=stats.first_commit_date,
            period_end=stats.last_commit_date,
            narrative=narrative,
            patterns=[],
            key_commits=key_commits,
            stats={
                "merge_commits": merge_commits_count,
                "tagged_commits": len(tagged_commits),
                "longest_streak_days": (
                    longest_streak.get("days", 0) if longest_streak else 0
                ),
            },
            order=0,
        )

    def generate_crisis_chapter(self) -> Chapter:
        """Generate a chapter about crises detected.

        Covers: burnout indicators, bug storms, after-hours patterns,
        vague-message clusters, and silence periods.

        Returns
        -------
        Chapter
            The crisis chapter.
        """
        commits = self.sorted_commits
        stats = self.git_stats

        # Crisis-related patterns
        crisis_pattern_types = {
            PatternType.BURNOUT_INDICATOR,
            PatternType.IRREGULAR_HOURS,
            PatternType.HIGH_CHURN,
            PatternType.ANTI_PATTERN,
        }
        crisis_patterns = [
            p for p in self.patterns if p.pattern_type in crisis_pattern_types
        ]
        any_critical_crisis = any(
            p.severity in (Severity.CRITICAL, Severity.HIGH) for p in crisis_patterns
        )

        # After-hours evidence
        after_hours_evidence = self._build_after_hours_evidence(commits)

        # Vague message evidence
        vague_evidence = self._build_vague_evidence(commits)

        # Silence evidence
        silence_periods = self._find_silence_periods(min_days=7)
        if silence_periods:
            longest = max(silence_periods, key=lambda p: p["days"])
            silence_evidence = (
                f"The longest gap was {longest['days']} days starting "
                f"{longest['start']}. In total, {len(silence_periods)} "
                f"silence {_pluralize(len(silence_periods), 'period')} "
                f"of 7+ days were detected."
            )
        else:
            silence_evidence = ""

        # Fix storm evidence
        fix_storm_evidence = self._build_fix_storm_evidence(commits)

        # Recommendations
        recommendations = list(
            dict.fromkeys(
                p.recommendation for p in crisis_patterns if p.recommendation
            )
        )

        # Period
        period_start = stats.first_commit_date
        period_end = stats.last_commit_date

        context: Dict[str, Any] = {
            "crisis_patterns": crisis_patterns,
            "any_critical_crisis": any_critical_crisis,
            "after_hours_evidence": after_hours_evidence,
            "vague_evidence": vague_evidence,
            "silence_evidence": silence_evidence,
            "fix_storm_evidence": fix_storm_evidence,
            "recommendations": recommendations,
        }

        narrative = self._render_template(_CRISIS_TEMPLATE, context)
        narrative = self._ai_enhance(narrative, context)

        return Chapter(
            id=str(uuid.uuid4()),
            chapter_type=ChapterType.CRISIS,
            title="When Things Got Hard",
            subtitle="Burnout, bug storms, and difficult periods",
            period_start=period_start,
            period_end=period_end,
            narrative=narrative,
            patterns=crisis_patterns,
            key_commits=[],
            stats={
                "crisis_pattern_count": len(crisis_patterns),
                "silence_periods": len(silence_periods),
            },
            order=0,
        )

    def generate_current_state_chapter(self) -> Chapter:
        """Generate a chapter about the current state of the project.

        Returns
        -------
        Chapter
            The current-state chapter.
        """
        commits = self.sorted_commits
        stats = self.git_stats

        # Recent activity (last 30 days or last 20% of commits, whichever is larger)
        recent_cutoff = stats.last_commit_date - timedelta(days=30)
        recent_commits = [c for c in commits if c.date >= recent_cutoff]
        if not recent_commits and commits:
            # Fall back to last 20% of commits
            cutoff_idx = max(0, len(commits) - max(1, len(commits) // 5))
            recent_commits = commits[cutoff_idx:]

        recent_commit_count = len(recent_commits)
        recent_insertions = sum(c.insertions for c in recent_commits)
        recent_deletions = sum(c.deletions for c in recent_commits)
        recent_net_lines = recent_insertions - recent_deletions
        recent_authors = sorted({c.author_name for c in recent_commits})
        recent_first_date = (
            recent_commits[0].date.strftime("%B %d, %Y")
            if recent_commits
            else "N/A"
        )
        recent_last_date = (
            recent_commits[-1].date.strftime("%B %d, %Y")
            if recent_commits
            else "N/A"
        )

        # Activity level
        commits_per_week = 0.0
        if recent_commits:
            span_days = max(
                (recent_commits[-1].date - recent_commits[0].date).days, 1
            )
            commits_per_week = recent_commit_count / span_days * 7
            if commits_per_week >= 10:
                recent_activity_level = "high"
            elif commits_per_week >= 3:
                recent_activity_level = "moderate"
            elif commits_per_week >= 1:
                recent_activity_level = "low"
            else:
                recent_activity_level = "minimal"
        else:
            recent_activity_level = "minimal"

        # Active patterns (those with last_seen in the last 30 days)
        active_cutoff = stats.last_commit_date - timedelta(days=30)
        active_patterns = [
            p for p in self.patterns if p.last_seen >= active_cutoff
        ]
        any_severe_active = any(
            p.severity in (Severity.CRITICAL, Severity.HIGH)
            for p in active_patterns
        )

        # Most changed files recently
        recent_file_counter: Counter = Counter()
        for c in recent_commits:
            for fp in c.files_changed:
                recent_file_counter[fp] += 1
        most_changed_files_recent = recent_file_counter.most_common(5)

        total_files = len({fp for c in commits for fp in c.files_changed})
        net_lines = stats.total_insertions - stats.total_deletions
        recent_author_count = len(recent_authors)
        quiet_count = stats.unique_authors - recent_author_count

        context: Dict[str, Any] = {
            "repo_name": self.repo_name,
            "recent_commit_count": recent_commit_count,
            "recent_first_date": recent_first_date,
            "recent_last_date": recent_last_date,
            "recent_insertions": recent_insertions,
            "recent_deletions": recent_deletions,
            "recent_net_lines": recent_net_lines,
            "recent_authors": recent_authors,
            "recent_activity_level": recent_activity_level,
            "active_patterns": active_patterns,
            "any_severe_active": any_severe_active,
            "total_insertions": stats.total_insertions,
            "total_deletions": stats.total_deletions,
            "total_files": total_files,
            "net_lines": net_lines,
            "abs_net_lines": abs(net_lines),
            "unique_authors": stats.unique_authors,
            "recent_author_count": recent_author_count,
            "quiet_count": quiet_count,
            "most_changed_files_recent": most_changed_files_recent,
        }

        narrative = self._render_template(_CURRENT_STATE_TEMPLATE, context)
        narrative = self._ai_enhance(narrative, context)

        return Chapter(
            id=str(uuid.uuid4()),
            chapter_type=ChapterType.CURRENT_STATE,
            title="Where Things Stand",
            subtitle="Current project health and recent activity",
            period_start=stats.first_commit_date,
            period_end=stats.last_commit_date,
            narrative=narrative,
            patterns=active_patterns,
            key_commits=recent_commits[:5],
            stats={
                "recent_commits": recent_commit_count,
                "commits_per_week": round(commits_per_week, 1),
                "active_patterns": len(active_patterns),
                "recent_authors": recent_author_count,
            },
            order=0,
        )

    def generate_forecast_chapter(self) -> Chapter:
        """Generate a chapter about future forecasts and recommendations.

        Returns
        -------
        Chapter
            The forecast chapter.
        """
        stats = self.git_stats

        health_score = _compute_overall_health_score(
            self.patterns, self.forecasts, self.commits, self.git_stats
        )

        context: Dict[str, Any] = {
            "repo_name": self.repo_name,
            "forecasts": self.forecasts,
            "health_score": health_score,
        }

        narrative = self._render_template(_FORECAST_TEMPLATE, context)
        narrative = self._ai_enhance(narrative, context)

        return Chapter(
            id=str(uuid.uuid4()),
            chapter_type=ChapterType.FORECAST,
            title="Looking Ahead",
            subtitle="Forecasts and recommendations",
            period_start=None,
            period_end=None,
            narrative=narrative,
            patterns=[],
            key_commits=[],
            stats={
                "forecast_count": len(self.forecasts) if self.forecasts else 0,
                "health_score": health_score,
            },
            order=0,
        )

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_template(
        self, template_str: str, context: Dict[str, Any]
    ) -> str:
        """Render a Jinja2 template with context data.

        Parameters
        ----------
        template_str:
            Jinja2 template as a string.
        context:
            Variables to pass to the template.

        Returns
        -------
        str
            The rendered narrative text.
        """
        try:
            template = _JINJA_ENV.from_string(template_str)
            return template.render(**context)
        except Exception:
            logger.error(
                "Template rendering failed, falling back to simple output",
                exc_info=True,
            )
            # Fallback: return a simple text summary
            return self._fallback_narrative(context)

    def _fallback_narrative(self, context: Dict[str, Any]) -> str:
        """Generate a simple fallback narrative when template rendering fails.

        Parameters
        ----------
        context:
            The context dict that would have been passed to the template.

        Returns
        -------
        str
            A simple text narrative.
        """
        parts = []
        for key, value in context.items():
            if callable(value):
                continue
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list) and len(value) <= 5:
                parts.append(f"{key}: {value}")
        return "Narrative data: " + "; ".join(parts)

    # ------------------------------------------------------------------
    # AI enhancement
    # ------------------------------------------------------------------

    def _ai_enhance(
        self, draft_narrative: str, context: Dict[str, Any]
    ) -> str:
        """Optionally enhance narrative with AI.

        If ``ai_config`` was provided with an ``api_key``, sends the draft
        narrative along with structured data as context to an OpenAI-
        compatible LLM.  The AI is asked to enhance readability while
        keeping all facts accurate.

        If AI is unavailable (no API key, network error, etc.), silently
        falls back to the template-mode draft.

        Parameters
        ----------
        draft_narrative:
            The template-mode narrative text.
        context:
            Structured data used to generate the narrative.  Included in
            the AI prompt so the model has access to the raw numbers.

        Returns
        -------
        str
            Either the AI-enhanced narrative or the original draft.
        """
        api_key = self.ai_config.get("api_key")
        if not api_key:
            return draft_narrative

        try:
            import openai
        except ImportError:
            logger.debug("openai package not installed, skipping AI enhancement")
            return draft_narrative

        model = self.ai_config.get("model", "gpt-4o-mini")
        base_url = self.ai_config.get("base_url")

        # Build a serialisable summary of the context for the AI
        context_summary = self._summarise_context(context)

        system_prompt = (
            "You are a technical writer who turns data-driven draft narratives "
            "into polished, engaging prose.  You MUST follow these rules:\n"
            "1. Keep every fact, number, date, and file name exactly as given.\n"
            "2. Do not add information that is not in the draft or context.\n"
            "3. Improve readability, flow, and narrative voice.\n"
            "4. Maintain the same approximate length (300-800 words).\n"
            "5. Do not add headers, bullet points, or formatting — just prose.\n"
            "6. Write in first-person plural or third-person, as the draft does.\n"
            "7. The tone should be engaging but factual — like a well-written "
            "book about the codebase."
        )

        user_prompt = (
            f"Here is a draft narrative with its supporting data context.\n\n"
            f"--- DRAFT ---\n{draft_narrative}\n\n"
            f"--- DATA CONTEXT ---\n{context_summary}\n\n"
            f"Please enhance the draft for readability while preserving all "
            f"factual accuracy."
        )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        try:
            client = openai.OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=1200,
            )
            enhanced = response.choices[0].message.content
            if enhanced and len(enhanced.strip()) > 50:
                logger.info("AI enhancement succeeded (%d chars)", len(enhanced))
                return enhanced.strip()
            logger.warning(
                "AI enhancement returned short/empty response, using draft"
            )
            return draft_narrative
        except Exception:
            logger.warning(
                "AI enhancement failed, using template draft", exc_info=True
            )
            return draft_narrative

    @staticmethod
    def _summarise_context(context: Dict[str, Any]) -> str:
        """Create a human-readable summary of the context for AI prompts.

        Parameters
        ----------
        context:
            The context dictionary.

        Returns
        -------
        str
            A formatted summary string.
        """
        lines: List[str] = []
        for key, value in context.items():
            if callable(value):
                continue
            if isinstance(value, (str, int, float, bool)):
                lines.append(f"{key}: {value}")
            elif isinstance(value, list):
                # Truncate long lists
                if len(value) <= 5:
                    lines.append(f"{key}: {value}")
                else:
                    lines.append(
                        f"{key}: [{len(value)} items] first 3: {value[:3]}"
                    )
            elif isinstance(value, dict):
                # Show dict keys and a few values
                items = list(value.items())[:5]
                lines.append(f"{key}: {dict(items)}")
            else:
                # Objects (Pattern, CommitData, etc.) — try str representation
                try:
                    lines.append(f"{key}: {str(value)[:200]}")
                except Exception:
                    lines.append(f"{key}: [complex object]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers — data computations
    # ------------------------------------------------------------------

    def _compute_weekly_activity_summary(self) -> Optional[str]:
        """Compute a summary of how weekly activity has changed over time.

        Returns
        -------
        str or None
            A summary phrase like "increased over time" or "remained steady",
            or None if insufficient data.
        """
        weekly = self.git_stats.weekly_commit_counts
        if len(weekly) < 4:
            return None

        # Compare first quarter to last quarter
        n = len(weekly)
        first_quarter_count = sum(c for _, c in weekly[: n // 4])
        last_quarter_count = sum(c for _, c in weekly[3 * n // 4 :])

        if first_quarter_count == 0:
            if last_quarter_count > 0:
                return "grown from a quiet start to active development"
            return None

        ratio = last_quarter_count / first_quarter_count
        if ratio >= 1.5:
            return "increased significantly over time"
        if ratio >= 1.1:
            return "gradually increased over time"
        if ratio <= 0.5:
            return "decreased significantly over time"
        if ratio <= 0.8:
            return "gradually decreased over time"
        return "remained relatively steady over time"

    def _compute_longest_streak(self) -> Optional[Dict[str, Any]]:
        """Compute the longest consecutive-day commit streak.

        Returns
        -------
        dict or None
            Dict with keys ``start``, ``end``, ``days``, ``commits``,
            or None if insufficient data.
        """
        commits = self.sorted_commits
        if len(commits) < 2:
            return None

        # Group commits by calendar date
        by_date: Dict[str, List[CommitData]] = defaultdict(list)
        for c in commits:
            date_str = c.date.strftime("%Y-%m-%d")
            by_date[date_str].append(c)

        # Find longest consecutive-date streak
        sorted_dates = sorted(by_date.keys())
        best_start = sorted_dates[0]
        best_end = sorted_dates[0]
        current_start = sorted_dates[0]
        current_end = sorted_dates[0]

        for i in range(1, len(sorted_dates)):
            prev = datetime.strptime(sorted_dates[i - 1], "%Y-%m-%d")
            curr = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
            if (curr - prev).days == 1:
                current_end = sorted_dates[i]
            else:
                current_start = sorted_dates[i]
                current_end = sorted_dates[i]

            prev_best_days = (
                datetime.strptime(best_end, "%Y-%m-%d")
                - datetime.strptime(best_start, "%Y-%m-%d")
            ).days + 1
            current_days = (
                datetime.strptime(current_end, "%Y-%m-%d")
                - datetime.strptime(current_start, "%Y-%m-%d")
            ).days + 1

            if current_days > prev_best_days:
                best_start = current_start
                best_end = current_end

        streak_days = (
            datetime.strptime(best_end, "%Y-%m-%d")
            - datetime.strptime(best_start, "%Y-%m-%d")
        ).days + 1

        if streak_days < 2:
            return None

        streak_commits = sum(
            len(by_date[d])
            for d in sorted_dates
            if best_start <= d <= best_end
        )

        return {
            "start": datetime.strptime(best_start, "%Y-%m-%d").strftime(
                "%B %d, %Y"
            ),
            "end": datetime.strptime(best_end, "%Y-%m-%d").strftime("%B %d, %Y"),
            "days": streak_days,
            "commits": streak_commits,
        }

    def _find_silence_periods(
        self, min_days: int = 7
    ) -> List[Dict[str, Any]]:
        """Find extended periods of commit silence.

        Parameters
        ----------
        min_days:
            Minimum gap in days to qualify as a silence period.

        Returns
        -------
        list[dict[str, Any]]
            Silence period descriptors with ``start``, ``end``, ``days``.
        """
        commits = self.sorted_commits
        if len(commits) < 2:
            return []

        periods: List[Dict[str, Any]] = []
        for i in range(1, len(commits)):
            gap_days = (commits[i].date - commits[i - 1].date).days
            if gap_days >= min_days:
                periods.append(
                    {
                        "start": commits[i - 1].date.strftime("%B %d, %Y"),
                        "end": commits[i].date.strftime("%B %d, %Y"),
                        "days": gap_days,
                    }
                )

        return periods

    def _build_after_hours_evidence(self, commits: List[CommitData]) -> str:
        """Build narrative evidence about after-hours commit patterns.

        Parameters
        ----------
        commits:
            Chronologically sorted commits.

        Returns
        -------
        str
            Narrative text about after-hours patterns.
        """
        if not commits:
            return ""

        after_hours_commits = [
            c
            for c in commits
            if c.date.hour < _AFTER_HOURS_END or c.date.hour >= _AFTER_HOURS_START
        ]
        if not after_hours_commits:
            return ""

        count = len(after_hours_commits)
        ratio = count / len(commits)

        # Find the latest after-hours commit
        latest = max(after_hours_commits, key=lambda c: c.date)
        latest_msg = latest.message.split("\n")[0][:80]

        parts = [
            f"{count} {_pluralize(count, 'commit')} ({ratio:.0%} of the total) "
            f"were made outside conventional hours — before 7 AM or after 8 PM."
        ]

        if count >= 10:
            # Find the most extreme after-hours cluster
            midnight_commits = [
                c for c in after_hours_commits if 0 <= c.date.hour < 5
            ]
            if midnight_commits:
                parts.append(
                    f"Of those, {len(midnight_commits)} were committed between "
                    f"midnight and 5 AM."
                )

        parts.append(
            f"The most recent after-hours commit was '{latest_msg}' "
            f"on {latest.date.strftime('%B %d, %Y')} at "
            f"{latest.date.strftime('%I:%M %p')}."
        )

        return " ".join(parts)

    def _build_vague_evidence(self, commits: List[CommitData]) -> str:
        """Build narrative evidence about vague commit messages.

        Parameters
        ----------
        commits:
            Chronologically sorted commits.

        Returns
        -------
        str
            Narrative text about vague messages.
        """
        if not commits:
            return ""

        vague_commits = [c for c in commits if _is_vague_message(c.message)]
        if not vague_commits:
            return ""

        count = len(vague_commits)
        ratio = count / len(commits)

        examples = [c.message.split("\n")[0] for c in vague_commits[:3]]

        parts = [
            f"{count} {_pluralize(count, 'commit')} ({ratio:.0%}) have vague "
            f"messages like"
        ]

        for i, ex in enumerate(examples):
            parts.append(f"'{ex}'")
            if i < len(examples) - 1:
                parts.append(",")

        parts.append(
            "— messages so short or generic they tell future readers "
            "nothing about the intent behind the change."
        )

        return " ".join(parts)

    def _build_fix_storm_evidence(self, commits: List[CommitData]) -> str:
        """Build narrative evidence about fix-commit clusters.

        A "fix storm" is a period where fix commits cluster together
        unusually tightly.

        Parameters
        ----------
        commits:
            Chronologically sorted commits.

        Returns
        -------
        str
            Narrative text about fix storms.
        """
        if not commits:
            return ""

        fix_commits = [
            c for c in commits if _FIX_KEYWORD_RE.search(c.message)
        ]
        if not fix_commits:
            return ""

        # Look for weeks with 5+ fix commits
        weekly_fixes: Dict[str, List[CommitData]] = defaultdict(list)
        for c in fix_commits:
            iso_year, iso_week, _ = c.date.isocalendar()
            key = f"{iso_year}-W{iso_week:02d}"
            weekly_fixes[key].append(c)

        storm_weeks = {k: v for k, v in weekly_fixes.items() if len(v) >= 5}
        if not storm_weeks:
            # Lower threshold for smaller projects
            storm_weeks = {k: v for k, v in weekly_fixes.items() if len(v) >= 3}
            if not storm_weeks:
                return ""

        total_fixes = len(fix_commits)
        storm_count = len(storm_weeks)

        if storm_count == 1:
            week_key = list(storm_weeks.keys())[0]
            week_fixes = storm_weeks[week_key]
            return (
                f"In {week_key}, {len(week_fixes)} fix "
                f"{_pluralize(len(week_fixes), 'commit')} landed in a single "
                f"week — a concentrated bug-hunting session or the aftermath "
                f"of a change with wider consequences than expected. "
                f"In total, {total_fixes} fix {_pluralize(total_fixes, 'commit')} "
                f"appear across the project's history."
            )

        return (
            f"{storm_count} weeks saw clusters of 5+ fix commits, "
            f"suggesting recurring periods of instability. "
            f"In total, {total_fixes} fix {_pluralize(total_fixes, 'commit')} "
            f"appear across the project's history."
        )
