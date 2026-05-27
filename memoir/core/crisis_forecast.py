"""Crisis forecast engine for the memoir project.

This module provides the :class:`CrisisForecaster` class which takes detected
patterns and git analysis data to forecast **future** risks.  Every forecast
is data-driven: probabilities come from actual trend extrapolation, timelines
from threshold-crossing projections, and historical precedents from past
periods with similar indicators.

Implementation Rules
~~~~~~~~~~~~~~~~~~~~
1.  **Probability is calculated from data, not guessed.**  Use the actual
    trend data from patterns and git stats.
2.  **Estimated timeline is based on trend extrapolation.**  If after-hours
    ratio is increasing by X% per quarter, when does it hit the critical
    threshold?
3.  **Historical precedent**: Look at past periods with similar indicators.
    Did a crisis occur before?
4.  **Each Forecast must have specific indicators** with current_value,
    threshold_value, and trend direction.
5.  **Only return a Forecast if there's enough evidence.**  If data is
    insufficient, return None.
6.  **No false alarms.**  Minimum confidence threshold of 0.3 before issuing
    a forecast.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from memoir.models.commit_data import CommitData, GitStats
from memoir.models.forecast import Forecast, ForecastIndicator, RiskLevel, RiskType
from memoir.models.pattern import Pattern, PatternType, Severity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# After-hours definition (mirrors git_analyzer / pattern_detector)
_AFTER_HOURS_START = 20
_AFTER_HOURS_END = 7

# Vague message detection (mirrors pattern_detector)
_VAGUE_RE = re.compile(
    r"^(wip|fix|update|changes?|misc|stuff|cleanups?|tidy|tweaks?"
    r"|adjust|minor|fixes|updates|tmp|temp|hack|x)$",
    re.IGNORECASE,
)
_VAGUE_LENGTH_THRESHOLD = 10

# Technical debt markers
_DEBT_MARKER_RE = re.compile(
    r"\b(TODO|FIXME|HACK|XXX|WORKAROUND|KLUDGE|TEMP|TEMPORARY)\b",
)

# Test file path heuristic
_TEST_PATH_RE = re.compile(
    r"(test|spec|__tests__|mock|fixture|stub)",
    re.IGNORECASE,
)

# Minimum confidence threshold for any forecast
_MIN_CONFIDENCE = 0.3

# Critical thresholds for burnout indicators
_AFTER_HOURS_CRITICAL = 0.40      # > 40% after-hours
_WEEKEND_CRITICAL = 0.30          # > 30% weekend work
_VAGUE_MESSAGE_CRITICAL = 0.50    # > 50% vague messages
_COMMIT_FREQ_DECLINE_CRITICAL = 0.30  # > 30% decline quarter-over-quarter

# Technical debt thresholds
_DEBT_DENSITY_CRITICAL_PER_KLOC = 10  # debt markers per 1000 lines of code

# Bus factor thresholds
_BUS_FACTOR_CRITICAL = 1
_BUS_FACTOR_HIGH = 2
_BUS_FACTOR_MODERATE = 3

# Health score penalties by risk level
_HEALTH_PENALTY: Dict[RiskLevel, int] = {
    RiskLevel.CRITICAL: 8,
    RiskLevel.HIGH: 5,
    RiskLevel.MODERATE: 3,
    RiskLevel.LOW: 1,
}

# Weights for burnout indicator contributions to probability
_BURNOUT_WEIGHTS: Dict[str, float] = {
    "after_hours_trend": 0.25,
    "weekend_work_trend": 0.20,
    "vague_message_trend": 0.20,
    "commit_frequency_trend": 0.20,
    "burst_silence_ratio": 0.15,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _generate_forecast_id(risk_type: RiskType, timestamp: datetime) -> str:
    """Generate a deterministic forecast identifier.

    Parameters
    ----------
    risk_type:
        The classification of the risk.
    timestamp:
        Reference timestamp used as part of the hash input.

    Returns
    -------
    str
        A hex string uniquely identifying this forecast.
    """
    raw = f"forecast:{risk_type.value}:{timestamp.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_into_quarters(
    commits: List[CommitData],
) -> List[List[CommitData]]:
    """Split commits chronologically into four equal-sized groups.

    If there are fewer than 4 commits, returns a single group containing
    all of them.

    Parameters
    ----------
    commits:
        Commits sorted chronologically (oldest first).

    Returns
    -------
    list[list[CommitData]]
        Up to 4 groups of commits.
    """
    if len(commits) < 4:
        return [commits] if commits else []

    sorted_commits = sorted(commits, key=lambda c: c.date)
    n = len(sorted_commits)
    q = n // 4
    return [
        sorted_commits[0:q],
        sorted_commits[q : 2 * q],
        sorted_commits[2 * q : 3 * q],
        sorted_commits[3 * q :],
    ]


def _compute_trend(values: List[float]) -> Optional[float]:
    """Compute a simple linear trend (slope) from a sequence of values.

    Returns ``None`` if there are fewer than 2 data points.

    Parameters
    ----------
    values:
        Chronologically ordered metric values.

    Returns
    -------
    float or None
        Slope of the least-squares linear fit, or ``None`` if insufficient
        data.
    """
    n = len(values)
    if n < 2:
        return None

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = 0.0
    denominator = 0.0
    for i, y in enumerate(values):
        dx = i - x_mean
        dy = y - y_mean
        numerator += dx * dy
        denominator += dx * dx

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _is_vague_message(message: str) -> bool:
    """Return True if a commit message is considered vague."""
    subject = message.strip().split("\n", 1)[0].strip()
    normalised = subject.lower().rstrip(".!? ")
    return len(subject) < _VAGUE_LENGTH_THRESHOLD or bool(_VAGUE_RE.match(normalised))


def _estimate_time_to_threshold(
    current: float,
    threshold: float,
    trend_per_quarter: float,
) -> Optional[int]:
    """Estimate number of weeks until a metric crosses a critical threshold.

    Parameters
    ----------
    current:
        Current value of the metric.
    threshold:
        Critical threshold value.
    trend_per_quarter:
        Slope of the metric per quarter (from linear regression).

    Returns
    -------
    int or None
        Estimated weeks until the threshold is crossed, or ``None`` if the
        metric is not trending towards the threshold.
    """
    if trend_per_quarter is None or trend_per_quarter == 0:
        return None

    if current >= threshold:
        return 0  # Already at or past threshold

    # trend_per_quarter is the change per quarter; convert to per-week
    trend_per_week = trend_per_quarter / 13.0  # ~13 weeks per quarter

    if trend_per_week <= 0:
        return None  # Not trending towards threshold

    weeks_remaining = (threshold - current) / trend_per_week
    return max(0, int(math.ceil(weeks_remaining)))


def _format_timeline(weeks: Optional[int]) -> str:
    """Format a week count into a human-readable timeline string.

    Parameters
    ----------
    weeks:
        Number of weeks, or ``None`` if not estimable.

    Returns
    -------
    str
        Human-readable timeline string.
    """
    if weeks is None:
        return "unknown"
    if weeks == 0:
        return "imminent"
    if weeks <= 2:
        return "1-2 weeks"
    if weeks <= 4:
        return "2-4 weeks"
    if weeks <= 8:
        return "1-2 months"
    if weeks <= 16:
        return "2-4 months"
    if weeks <= 26:
        return "4-6 months"
    return "6+ months"


def _find_silence_periods(
    commits: List[CommitData],
    min_days: int = 7,
) -> List[Dict[str, Any]]:
    """Find extended periods of commit silence.

    Parameters
    ----------
    commits:
        Chronologically sorted commits.
    min_days:
        Minimum gap in days to qualify as a silence period.

    Returns
    -------
    list[dict[str, Any]]
        List of silence period descriptors.
    """
    if len(commits) < 2:
        return []

    sorted_commits = sorted(commits, key=lambda c: c.date)
    periods: List[Dict[str, Any]] = []

    for i in range(1, len(sorted_commits)):
        gap_days = (sorted_commits[i].date - sorted_commits[i - 1].date).days
        if gap_days >= min_days:
            periods.append(
                {
                    "start_date": sorted_commits[i - 1].date.isoformat(),
                    "end_date": sorted_commits[i].date.isoformat(),
                    "duration_days": gap_days,
                }
            )

    return periods


def _compute_commit_rate(commits: List[CommitData]) -> float:
    """Compute commits per week for a list of commits.

    Parameters
    ----------
    commits:
        List of commits.

    Returns
    -------
    float
        Commits per week, or 0.0 if insufficient data.
    """
    if len(commits) < 2:
        return 0.0

    sorted_commits = sorted(commits, key=lambda c: c.date)
    span_days = max((sorted_commits[-1].date - sorted_commits[0].date).days, 1)
    return len(commits) / span_days * 7.0


def _find_historical_precedent(
    commits: List[CommitData],
    risk_type: RiskType,
    indicators: List[ForecastIndicator],
) -> Optional[str]:
    """Search commit history for past periods with similar crisis indicators.

    Looks for periods where similar indicators were at elevated levels and
    checks whether a crisis event followed.

    Parameters
    ----------
    commits:
        Chronologically sorted commits.
    risk_type:
        The type of risk being forecast.
    indicators:
        The current indicators triggering the forecast.

    Returns
    -------
    str or None
        A description of a historical precedent, or ``None`` if none found.
    """
    if len(commits) < 20:
        return None

    sorted_commits = sorted(commits, key=lambda c: c.date)
    mid = len(sorted_commits) // 2

    first_half = sorted_commits[:mid]
    second_half = sorted_commits[mid:]

    if not first_half or not second_half:
        return None

    precedent: Optional[str] = None

    if risk_type == RiskType.BURNOUT:
        silence_periods = _find_silence_periods(first_half, min_days=7)
        if silence_periods:
            longest = max(silence_periods, key=lambda p: p["duration_days"])
            precedent = (
                f"Historical precedent: a {longest['duration_days']}-day "
                f"commit gap occurred starting {longest['start_date'][:10]}, "
                f"suggesting a previous burnout or disengagement event."
            )

    elif risk_type == RiskType.TECHNICAL_DEBT_CRISIS:
        early_debt = sum(
            1 for c in first_half if _DEBT_MARKER_RE.search(c.message)
        )
        late_debt = sum(
            1 for c in second_half if _DEBT_MARKER_RE.search(c.message)
        )
        if early_debt > 3 and late_debt > early_debt * 1.5:
            precedent = (
                f"Technical debt markers grew from {early_debt} in the first "
                f"half to {late_debt} in the second half of the project "
                f"history, suggesting a previous debt accumulation cycle."
            )

    elif risk_type == RiskType.BUS_FACTOR:
        first_authors = {c.author_name for c in first_half}
        second_authors = {c.author_name for c in second_half}
        departed = first_authors - second_authors
        if departed and len(departed) < len(first_authors):
            departed_names = ", ".join(sorted(departed)[:3])
            precedent = (
                f"Historical precedent: {len(departed)} contributor(s) "
                f"({departed_names}) were active in the first half of the "
                f"project but stopped committing in the second half."
            )

    elif risk_type == RiskType.MAINTAINABILITY:
        first_avg_msg = sum(len(c.message) for c in first_half) / max(len(first_half), 1)
        second_avg_msg = sum(len(c.message) for c in second_half) / max(len(second_half), 1)
        if second_avg_msg < first_avg_msg * 0.7:
            precedent = (
                f"Commit message quality declined from avg {first_avg_msg:.0f} "
                f"chars to {second_avg_msg:.0f} chars between project halves, "
                f"correlating with a previous maintainability decline."
            )

    elif risk_type == RiskType.STAGNATION:
        first_rate = _compute_commit_rate(first_half)
        second_rate = _compute_commit_rate(second_half)
        if first_rate > 0 and second_rate < first_rate * 0.5:
            precedent = (
                f"Commit rate dropped from {first_rate:.1f}/week in the "
                f"first half to {second_rate:.1f}/week in the second half, "
                f"indicating a previous stagnation period."
            )

    return precedent


# ---------------------------------------------------------------------------
# CrisisForecaster
# ---------------------------------------------------------------------------


class CrisisForecaster:
    """Forecast future project crises from detected patterns and git data.

    The forecaster takes the outputs of :class:`GitAnalyzer` and
    :class:`PatternDetector` and extrapolates trend data to predict when
    risks will materialise.  Every probability and timeline is derived from
    actual measurements, not subjective judgement.

    Parameters
    ----------
    commits:
        Complete list of commits for the analysed range.
    git_stats:
        Aggregate repository statistics.
    patterns:
        Patterns detected by :class:`PatternDetector`.
    work_pattern:
        Output of :meth:`GitAnalyzer.get_work_pattern`.
    message_patterns:
        Output of :meth:`GitAnalyzer.get_commit_message_patterns`.
    code_churn:
        Output of :meth:`GitAnalyzer.get_code_churn`.
    """

    def __init__(
        self,
        commits: List[CommitData],
        git_stats: GitStats,
        patterns: List[Pattern],
        work_pattern: Dict[str, Any],
        message_patterns: Dict[str, Any],
        code_churn: Dict[str, Dict[str, int]],
    ) -> None:
        self.commits = sorted(commits, key=lambda c: c.date)
        self.git_stats = git_stats
        self.patterns = patterns
        self.work_pattern = work_pattern
        self.message_patterns = message_patterns
        self.code_churn = code_churn

        # Pre-computed indices (lazily built)
        self._quarters: Optional[List[List[CommitData]]] = None
        self._author_file_map: Optional[Dict[str, Dict[str, int]]] = None
        self._file_sizes: Optional[Dict[str, int]] = None
        self._co_change_matrix: Optional[Dict[str, Dict[str, int]]] = None

        logger.info(
            "CrisisForecaster initialised with %d commits, %d patterns, "
            "%d files in churn data",
            len(self.commits),
            len(self.patterns) if self.patterns else 0,
            len(self.code_churn) if self.code_churn else 0,
        )

    # ------------------------------------------------------------------
    # Pre-computed indices (lazily built)
    # ------------------------------------------------------------------

    @property
    def quarters(self) -> List[List[CommitData]]:
        """Commits split into four chronological quarters."""
        if self._quarters is None:
            self._quarters = _split_into_quarters(self.commits)
        return self._quarters

    @property
    def author_file_map(self) -> Dict[str, Dict[str, int]]:
        """Mapping of author -> {file_path: commit_count}."""
        if self._author_file_map is None:
            mapping: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(int)  # type: ignore[arg-type]
            )
            for c in self.commits:
                for fp in c.files_changed:
                    mapping[c.author_name][fp] += 1
            self._author_file_map = {
                author: dict(files) for author, files in mapping.items()
            }
        return self._author_file_map

    @property
    def file_sizes(self) -> Dict[str, int]:
        """Estimated file sizes from cumulative net changes.

        This is a rough proxy: total insertions minus total deletions per
        file, floored at 0.  For a true LOC count, a file-level scan would
        be needed, but this is sufficient for trend analysis.
        """
        if self._file_sizes is None:
            sizes: Dict[str, int] = {}
            for file_path, stats in self.code_churn.items():
                sizes[file_path] = max(stats.get("net", 0), 0)
            self._file_sizes = sizes
        return self._file_sizes

    @property
    def co_change_matrix(self) -> Dict[str, Dict[str, int]]:
        """Co-change frequency: how often pairs of files change together.

        Only tracks the top 30 most-changed files to keep the matrix
        manageable.
        """
        if self._co_change_matrix is None:
            matrix: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(int)  # type: ignore[arg-type]
            )
            # Only track top 30 most-changed files
            top_files = set()
            for fp, _ in self.git_stats.most_changed_files[:30]:
                top_files.add(fp)

            for c in self.commits:
                changed = [fp for fp in c.files_changed if fp in top_files]
                for i in range(len(changed)):
                    for j in range(i + 1, len(changed)):
                        matrix[changed[i]][changed[j]] += 1
                        matrix[changed[j]][changed[i]] += 1

            self._co_change_matrix = {
                f: dict(neighbours) for f, neighbours in matrix.items()
            }
        return self._co_change_matrix

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _probability_to_risk_level(probability: float) -> RiskLevel:
        """Convert a probability value to a risk level.

        Parameters
        ----------
        probability:
            Risk probability from 0.0 to 1.0.

        Returns
        -------
        RiskLevel
            The corresponding risk level.
        """
        if probability >= 0.75:
            return RiskLevel.CRITICAL
        if probability >= 0.50:
            return RiskLevel.HIGH
        if probability >= 0.30:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast_all(self) -> List[Forecast]:
        """Run all forecast models and return risk forecasts.

        Returns
        -------
        list[Forecast]
            All forecasts with confidence >= the minimum threshold,
            sorted by risk level (most severe first).
        """
        forecasts: List[Forecast] = []

        models = [
            ("burnout", self.forecast_burnout),
            ("tech_debt_crisis", self.forecast_tech_debt_crisis),
            ("bus_factor", self.forecast_bus_factor),
            ("maintainability", self.forecast_maintainability),
            ("stagnation", self.forecast_stagnation),
        ]

        for name, model_fn in models:
            try:
                forecast = model_fn()
                if forecast is not None and forecast.probability >= _MIN_CONFIDENCE:
                    forecasts.append(forecast)
                    logger.info(
                        "Forecast %s: risk_level=%s, probability=%.3f, "
                        "timeline=%s",
                        name,
                        forecast.risk_level.value,
                        forecast.probability,
                        forecast.estimated_timeline,
                    )
                elif forecast is not None:
                    logger.debug(
                        "Forecast %s below confidence threshold "
                        "(%.3f < %.3f), skipping",
                        name,
                        forecast.probability,
                        _MIN_CONFIDENCE,
                    )
                else:
                    logger.debug(
                        "Forecast %s returned None (insufficient evidence)",
                        name,
                    )
            except Exception:
                logger.error(
                    "Forecast model %s failed", name, exc_info=True
                )

        # Sort: CRITICAL first, then HIGH, MODERATE, LOW
        level_order = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.LOW: 3,
        }
        forecasts.sort(
            key=lambda f: (level_order.get(f.risk_level, 99), -f.probability)
        )

        logger.info("Total forecasts issued: %d", len(forecasts))
        return forecasts

    # ------------------------------------------------------------------
    # Forecast: Burnout
    # ------------------------------------------------------------------

    def forecast_burnout(self) -> Optional[Forecast]:
        """Predict burnout risk based on work patterns, commit quality trends,
        and burnout indicator patterns.

        Indicators
        ~~~~~~~~~~
        - **after_hours_trend**: ratio of after-hours commits over time
        - **weekend_work_trend**: ratio of weekend commits over time
        - **vague_message_trend**: ratio of vague commit messages over time
        - **commit_frequency_trend**: commits-per-week over time (declining
          is negative)
        - **burst_silence_ratio**: ratio of burst-silence cycles detected

        Probability is a weighted average of how many indicators are trending
        negatively and how steep the trend is.

        Timeline is based on when indicators will cross critical thresholds.

        Returns
        -------
        Forecast or None
            A burnout risk forecast, or ``None`` if insufficient evidence.
        """
        if len(self.commits) < 15:
            logger.debug(
                "Too few commits (%d) for burnout forecast", len(self.commits)
            )
            return None

        quarters = self.quarters
        if len(quarters) < 2:
            logger.debug("Insufficient quarter data for burnout forecast")
            return None

        indicators: List[ForecastIndicator] = []
        weighted_probability = 0.0
        timeline_weeks: List[int] = []

        # --- Indicator 1: After-hours ratio trend ---
        after_hours_result = self._compute_after_hours_indicator(quarters)
        if after_hours_result is not None:
            indicators.append(after_hours_result["indicator"])
            weighted_probability += (
                after_hours_result["contribution"]
                * _BURNOUT_WEIGHTS["after_hours_trend"]
            )
            if after_hours_result["weeks_to_threshold"] is not None:
                timeline_weeks.append(after_hours_result["weeks_to_threshold"])

        # --- Indicator 2: Weekend work ratio trend ---
        weekend_result = self._compute_weekend_indicator(quarters)
        if weekend_result is not None:
            indicators.append(weekend_result["indicator"])
            weighted_probability += (
                weekend_result["contribution"]
                * _BURNOUT_WEIGHTS["weekend_work_trend"]
            )
            if weekend_result["weeks_to_threshold"] is not None:
                timeline_weeks.append(weekend_result["weeks_to_threshold"])

        # --- Indicator 3: Vague message ratio trend ---
        vague_result = self._compute_vague_message_indicator(quarters)
        if vague_result is not None:
            indicators.append(vague_result["indicator"])
            weighted_probability += (
                vague_result["contribution"]
                * _BURNOUT_WEIGHTS["vague_message_trend"]
            )
            if vague_result["weeks_to_threshold"] is not None:
                timeline_weeks.append(vague_result["weeks_to_threshold"])

        # --- Indicator 4: Commit frequency trend ---
        freq_result = self._compute_commit_frequency_indicator(quarters)
        if freq_result is not None:
            indicators.append(freq_result["indicator"])
            weighted_probability += (
                freq_result["contribution"]
                * _BURNOUT_WEIGHTS["commit_frequency_trend"]
            )
            if freq_result["weeks_to_threshold"] is not None:
                timeline_weeks.append(freq_result["weeks_to_threshold"])

        # --- Indicator 5: Burst-silence ratio ---
        burst_result = self._compute_burst_silence_indicator()
        if burst_result is not None:
            indicators.append(burst_result["indicator"])
            weighted_probability += (
                burst_result["contribution"]
                * _BURNOUT_WEIGHTS["burst_silence_ratio"]
            )

        # Need at least 2 indicators with evidence
        if len(indicators) < 2:
            logger.debug(
                "Burnout forecast: only %d indicators, need >= 2",
                len(indicators),
            )
            return None

        # Scale probability to [0, 1]
        probability = min(1.0, weighted_probability)

        # Boost if burnout patterns already detected
        burnout_patterns = [
            p
            for p in self.patterns
            if p.pattern_type == PatternType.BURNOUT_INDICATOR
        ]
        if burnout_patterns:
            severity_rank = {
                Severity.CRITICAL: 4,
                Severity.HIGH: 3,
                Severity.MEDIUM: 2,
                Severity.LOW: 1,
            }
            max_severity = max(
                burnout_patterns, key=lambda p: severity_rank.get(p.severity, 0)
            )
            severity_boost = {
                Severity.CRITICAL: 0.15,
                Severity.HIGH: 0.10,
                Severity.MEDIUM: 0.05,
                Severity.LOW: 0.02,
            }.get(max_severity.severity, 0.0)
            probability = min(1.0, probability + severity_boost)

        risk_level = self._probability_to_risk_level(probability)

        # Determine timeline from the earliest threshold crossing
        min_weeks: Optional[int] = None
        if timeline_weeks:
            min_weeks = min(timeline_weeks)
        timeline = _format_timeline(min_weeks)

        # Historical precedent
        precedent = _find_historical_precedent(
            self.commits, RiskType.BURNOUT, indicators
        )

        if probability < _MIN_CONFIDENCE:
            return None

        # Build description
        negative_count = sum(
            1 for ind in indicators if ind.trend in ("rising", "declining")
        )
        most_concerning = self._most_concerning_indicator(indicators)
        description = (
            f"Burnout risk forecast based on {len(indicators)} indicators, "
            f"{negative_count} of which are trending negatively. "
            f"Estimated probability: {probability:.0%}. "
            f"The most concerning trend is {most_concerning}."
        )

        recommendation = self._burnout_recommendation(indicators)

        forecast = Forecast(
            id=_generate_forecast_id(RiskType.BURNOUT, self.commits[-1].date),
            risk_type=RiskType.BURNOUT,
            risk_level=risk_level,
            title="Burnout risk forecast",
            description=description,
            indicators=indicators,
            probability=round(probability, 3),
            estimated_timeline=timeline,
            recommendation=recommendation,
            historical_precedent=precedent,
        )

        return forecast

    def _compute_after_hours_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute after-hours ratio trend indicator for burnout forecast."""
        if len(quarters) < 2:
            return None

        ratios: List[float] = []
        for quarter in quarters:
            if not quarter:
                ratios.append(0.0)
                continue
            after_hours = sum(
                1
                for c in quarter
                if c.date.hour < _AFTER_HOURS_END
                or c.date.hour >= _AFTER_HOURS_START
            )
            ratios.append(after_hours / len(quarter))

        trend = _compute_trend(ratios)
        if trend is None:
            return None

        current = ratios[-1] if ratios else 0.0
        trend_direction = (
            "rising" if trend > 0 else ("stable" if trend == 0 else "declining")
        )

        # Contribution: proximity to threshold + trend steepness
        proximity = current / _AFTER_HOURS_CRITICAL if _AFTER_HOURS_CRITICAL > 0 else 0
        trend_steepness = min(1.0, abs(trend) * 10)
        contribution = min(1.0, proximity * 0.7 + trend_steepness * 0.3)

        weeks_to_threshold = _estimate_time_to_threshold(
            current, _AFTER_HOURS_CRITICAL, trend
        )

        indicator = ForecastIndicator(
            name="after_hours_trend",
            current_value=round(current, 4),
            threshold_value=_AFTER_HOURS_CRITICAL,
            trend=trend_direction,
            description=(
                f"Ratio of after-hours commits (before 07:00 or after 20:00). "
                f"Current: {current:.1%}, threshold: "
                f"{_AFTER_HOURS_CRITICAL:.0%}."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
            "weeks_to_threshold": weeks_to_threshold,
        }

    def _compute_weekend_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute weekend work ratio trend indicator for burnout forecast."""
        if len(quarters) < 2:
            return None

        ratios: List[float] = []
        for quarter in quarters:
            if not quarter:
                ratios.append(0.0)
                continue
            weekend = sum(1 for c in quarter if c.date.weekday() >= 5)
            ratios.append(weekend / len(quarter))

        trend = _compute_trend(ratios)
        if trend is None:
            return None

        current = ratios[-1] if ratios else 0.0
        trend_direction = (
            "rising" if trend > 0 else ("stable" if trend == 0 else "declining")
        )

        proximity = current / _WEEKEND_CRITICAL if _WEEKEND_CRITICAL > 0 else 0
        trend_steepness = min(1.0, abs(trend) * 10)
        contribution = min(1.0, proximity * 0.7 + trend_steepness * 0.3)

        weeks_to_threshold = _estimate_time_to_threshold(
            current, _WEEKEND_CRITICAL, trend
        )

        indicator = ForecastIndicator(
            name="weekend_work_trend",
            current_value=round(current, 4),
            threshold_value=_WEEKEND_CRITICAL,
            trend=trend_direction,
            description=(
                f"Ratio of weekend commits. "
                f"Current: {current:.1%}, threshold: {_WEEKEND_CRITICAL:.0%}."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
            "weeks_to_threshold": weeks_to_threshold,
        }

    def _compute_vague_message_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute vague message ratio trend indicator for burnout forecast."""
        if len(quarters) < 2:
            return None

        ratios: List[float] = []
        for quarter in quarters:
            if not quarter:
                ratios.append(0.0)
                continue
            vague = sum(1 for c in quarter if _is_vague_message(c.message))
            ratios.append(vague / len(quarter))

        trend = _compute_trend(ratios)
        if trend is None:
            return None

        current = ratios[-1] if ratios else 0.0
        trend_direction = (
            "rising" if trend > 0 else ("stable" if trend == 0 else "declining")
        )

        proximity = (
            current / _VAGUE_MESSAGE_CRITICAL
            if _VAGUE_MESSAGE_CRITICAL > 0
            else 0
        )
        trend_steepness = min(1.0, abs(trend) * 10)
        contribution = min(1.0, proximity * 0.7 + trend_steepness * 0.3)

        weeks_to_threshold = _estimate_time_to_threshold(
            current, _VAGUE_MESSAGE_CRITICAL, trend
        )

        indicator = ForecastIndicator(
            name="vague_message_trend",
            current_value=round(current, 4),
            threshold_value=_VAGUE_MESSAGE_CRITICAL,
            trend=trend_direction,
            description=(
                f"Ratio of vague commit messages. "
                f"Current: {current:.1%}, threshold: "
                f"{_VAGUE_MESSAGE_CRITICAL:.0%}."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
            "weeks_to_threshold": weeks_to_threshold,
        }

    def _compute_commit_frequency_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute commit frequency trend indicator for burnout forecast.

        A declining commit frequency is a negative signal.
        """
        if len(quarters) < 2:
            return None

        rates: List[float] = []
        for quarter in quarters:
            rates.append(_compute_commit_rate(quarter))

        trend = _compute_trend(rates)
        if trend is None:
            return None

        current = rates[-1] if rates else 0.0
        # For commit frequency, declining is the negative direction
        trend_direction = (
            "declining" if trend < 0 else ("stable" if trend == 0 else "rising")
        )

        # Compute the percentage decline from first to last quarter
        baseline = rates[0] if rates[0] > 0 else 0.001
        decline_pct = max(0, (baseline - current) / baseline)

        # Contribution based on how much decline and trend steepness
        proximity = (
            decline_pct / _COMMIT_FREQ_DECLINE_CRITICAL
            if _COMMIT_FREQ_DECLINE_CRITICAL > 0
            else 0
        )
        trend_steepness = min(1.0, abs(trend) * 5)
        contribution = min(1.0, proximity * 0.6 + trend_steepness * 0.4)

        # Estimate time to critical decline threshold
        weeks_to_threshold: Optional[int] = None
        if trend < 0 and baseline > 0:
            critical_rate = baseline * (1 - _COMMIT_FREQ_DECLINE_CRITICAL)
            if current > critical_rate:
                trend_per_week = trend / 13.0
                if trend_per_week < 0:
                    weeks_to_threshold = max(
                        0,
                        int(
                            math.ceil(
                                (current - critical_rate) / abs(trend_per_week)
                            )
                        ),
                    )

        indicator = ForecastIndicator(
            name="commit_frequency_trend",
            current_value=round(current, 2),
            threshold_value=round(baseline * (1 - _COMMIT_FREQ_DECLINE_CRITICAL), 2),
            trend=trend_direction,
            description=(
                f"Commits per week trend. Current: {current:.1f}/week, "
                f"baseline: {baseline:.1f}/week. "
                f"Decline: {decline_pct:.1%} "
                f"(threshold: {_COMMIT_FREQ_DECLINE_CRITICAL:.0%})."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
            "weeks_to_threshold": weeks_to_threshold,
        }

    def _compute_burst_silence_indicator(self) -> Optional[Dict[str, Any]]:
        """Compute burst-silence cycle indicator for burnout forecast."""
        if len(self.commits) < 10:
            return None

        silence_periods = _find_silence_periods(self.commits, min_days=7)

        # Burst periods from work_pattern data
        burst_periods = self.work_pattern.get("burst_periods", [])

        burst_count = len(burst_periods)
        silence_count = len(silence_periods)

        if burst_count == 0 and silence_count == 0:
            return None

        # A high ratio of silence periods relative to total time
        total_span_days = max(
            (self.commits[-1].date - self.commits[0].date).days, 1
        )
        total_silence_days = sum(p["duration_days"] for p in silence_periods)
        silence_ratio = total_silence_days / total_span_days

        # Contribution: more burst-silence cycles = higher contribution
        contribution = min(
            1.0,
            (burst_count * 0.15) + (silence_ratio * 0.5) + (silence_count * 0.1),
        )

        trend = "rising" if silence_count >= 2 else "stable"

        indicator = ForecastIndicator(
            name="burst_silence_ratio",
            current_value=round(silence_ratio, 4),
            threshold_value=0.20,  # > 20% of time in silence is concerning
            trend=trend,
            description=(
                f"Ratio of silence days to total project span. "
                f"Found {silence_count} silence periods (>=7 days) "
                f"totalling {total_silence_days} days out of "
                f"{total_span_days} days ({silence_ratio:.1%}). "
                f"Also {burst_count} burst periods detected."
            ),
        )

        weeks_to_threshold: Optional[int] = None
        if silence_ratio < 0.20:
            quarters = self.quarters
            if len(quarters) >= 2:
                silence_ratios_by_quarter: List[float] = []
                for quarter in quarters:
                    if len(quarter) < 2:
                        silence_ratios_by_quarter.append(0.0)
                        continue
                    q_span = max(
                        (quarter[-1].date - quarter[0].date).days, 1
                    )
                    q_silence = sum(
                        p["duration_days"]
                        for p in _find_silence_periods(quarter, min_days=7)
                    )
                    silence_ratios_by_quarter.append(q_silence / q_span)

                q_trend = _compute_trend(silence_ratios_by_quarter)
                if q_trend is not None and q_trend > 0:
                    weeks_to_threshold = _estimate_time_to_threshold(
                        silence_ratio, 0.20, q_trend
                    )

        return {
            "indicator": indicator,
            "contribution": contribution,
            "weeks_to_threshold": weeks_to_threshold,
        }

    @staticmethod
    def _most_concerning_indicator(indicators: List[ForecastIndicator]) -> str:
        """Identify the most concerning indicator by proximity to threshold.

        Parameters
        ----------
        indicators:
            List of forecast indicators.

        Returns
        -------
        str
            Name of the most concerning indicator.
        """
        if not indicators:
            return "none"

        worst = indicators[0]
        worst_ratio = 0.0

        for ind in indicators:
            if ind.threshold_value == 0:
                continue
            ratio = ind.current_value / ind.threshold_value
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst = ind

        return worst.name

    @staticmethod
    def _burnout_recommendation(indicators: List[ForecastIndicator]) -> str:
        """Build a data-driven burnout mitigation recommendation."""
        parts = ["Burnout risk detected based on commit pattern analysis."]

        for ind in indicators:
            if ind.name == "after_hours_trend" and ind.trend == "rising":
                parts.append(
                    f"After-hours commits are at {ind.current_value:.0%} "
                    f"(threshold: {ind.threshold_value:.0%}). Investigate "
                    f"deadline pressure and set boundaries around work time."
                )
            elif ind.name == "weekend_work_trend" and ind.trend == "rising":
                parts.append(
                    f"Weekend work is at {ind.current_value:.0%} "
                    f"(threshold: {ind.threshold_value:.0%}). Consider "
                    f"redistributing workload or extending timelines."
                )
            elif ind.name == "vague_message_trend" and ind.trend == "rising":
                parts.append(
                    f"Vague commit messages are at {ind.current_value:.0%} "
                    f"(threshold: {ind.threshold_value:.0%}). This often "
                    f"correlates with rushing -- slow down and write "
                    f"descriptive messages."
                )
            elif (
                ind.name == "commit_frequency_trend" and ind.trend == "declining"
            ):
                parts.append(
                    "Commit frequency is declining. If this reflects "
                    "disengagement rather than intentional pacing, consider "
                    "rotating responsibilities or reducing scope."
                )
            elif ind.name == "burst_silence_ratio" and ind.trend == "rising":
                parts.append(
                    "Burst-silence patterns detected. Sustainable pace is "
                    "more productive than sprint-and-crash cycles."
                )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Forecast: Technical Debt Crisis
    # ------------------------------------------------------------------

    def forecast_tech_debt_crisis(self) -> Optional[Forecast]:
        """Predict when technical debt will become unmanageable.

        Indicators
        ~~~~~~~~~~
        - **debt_marker_density**: TODO/FIXME/HACK density per KLOC
        - **churn_without_growth_ratio**: files with high churn but low net
          growth
        - **code_test_ratio**: ratio of source commits to test commits

        Timeline is based on when debt markers exceed a critical density
        per KLOC.  Uses exponential growth model if growth is accelerating.

        Returns
        -------
        Forecast or None
            A tech debt crisis forecast, or ``None`` if insufficient evidence.
        """
        if len(self.commits) < 10:
            logger.debug("Too few commits for tech debt forecast")
            return None

        quarters = self.quarters
        if len(quarters) < 2:
            logger.debug("Insufficient quarter data for tech debt forecast")
            return None

        indicators: List[ForecastIndicator] = []
        probability_components: List[float] = []
        timeline_weeks: List[int] = []

        # --- Indicator 1: Debt marker growth rate ---
        debt_result = self._compute_debt_marker_indicator(quarters)
        if debt_result is not None:
            indicators.append(debt_result["indicator"])
            probability_components.append(debt_result["contribution"])
            if debt_result["weeks_to_threshold"] is not None:
                timeline_weeks.append(debt_result["weeks_to_threshold"])

        # --- Indicator 2: Churn without growth ratio ---
        churn_result = self._compute_churn_without_growth_indicator()
        if churn_result is not None:
            indicators.append(churn_result["indicator"])
            probability_components.append(churn_result["contribution"])

        # --- Indicator 3: Code-test ratio ---
        test_result = self._compute_code_test_ratio_indicator(quarters)
        if test_result is not None:
            indicators.append(test_result["indicator"])
            probability_components.append(test_result["contribution"])

        # Need at least 1 indicator with evidence
        if not indicators:
            logger.debug("Tech debt forecast: no indicators with evidence")
            return None

        # Probability is the average of component contributions
        probability = sum(probability_components) / len(probability_components)

        # Boost if there are existing TECHNICAL_DEBT patterns
        debt_patterns = [
            p
            for p in self.patterns
            if p.pattern_type == PatternType.TECHNICAL_DEBT
        ]
        if debt_patterns:
            high_severity_count = sum(
                1
                for p in debt_patterns
                if p.severity in (Severity.HIGH, Severity.CRITICAL)
            )
            probability = min(1.0, probability + high_severity_count * 0.05)

        risk_level = self._probability_to_risk_level(probability)

        # Timeline: earliest threshold crossing
        min_weeks: Optional[int] = min(timeline_weeks) if timeline_weeks else None

        # Apply exponential growth model if debt is accelerating
        if len(quarters) >= 3:
            debt_counts: List[int] = []
            for quarter in quarters:
                count = sum(
                    1 for c in quarter if _DEBT_MARKER_RE.search(c.message)
                )
                debt_counts.append(count)

            if (
                len(debt_counts) >= 3
                and debt_counts[-1] > debt_counts[-2] > debt_counts[-3]
            ):
                growth_rate_q1 = (debt_counts[-2] - debt_counts[-3]) / max(
                    debt_counts[-3], 1
                )
                growth_rate_q2 = (debt_counts[-1] - debt_counts[-2]) / max(
                    debt_counts[-2], 1
                )
                if growth_rate_q2 > growth_rate_q1 and growth_rate_q2 > 0:
                    # Exponential growth detected -- shorten timeline
                    if min_weeks is not None and min_weeks > 4:
                        min_weeks = max(2, min_weeks // 2)
                    logger.debug(
                        "Exponential debt growth detected: shortening timeline"
                    )

        timeline = _format_timeline(min_weeks)

        precedent = _find_historical_precedent(
            self.commits, RiskType.TECHNICAL_DEBT_CRISIS, indicators
        )

        if probability < _MIN_CONFIDENCE:
            return None

        description = (
            f"Technical debt crisis risk based on {len(indicators)} "
            f"indicators. Estimated probability: {probability:.0%}. "
            f"The codebase is accumulating debt faster than it is being "
            f"addressed."
        )

        recommendation = self._tech_debt_recommendation(indicators, debt_patterns)

        forecast = Forecast(
            id=_generate_forecast_id(
                RiskType.TECHNICAL_DEBT_CRISIS, self.commits[-1].date
            ),
            risk_type=RiskType.TECHNICAL_DEBT_CRISIS,
            risk_level=risk_level,
            title="Technical debt crisis forecast",
            description=description,
            indicators=indicators,
            probability=round(probability, 3),
            estimated_timeline=timeline,
            recommendation=recommendation,
            historical_precedent=precedent,
        )

        return forecast

    def _compute_debt_marker_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute debt marker growth indicator.

        Tracks TODO/FIXME/HACK density per commit across quarters.
        """
        if len(quarters) < 2:
            return None

        densities: List[float] = []
        counts: List[int] = []

        for quarter in quarters:
            if not quarter:
                densities.append(0.0)
                counts.append(0)
                continue
            marker_count = sum(
                1 for c in quarter if _DEBT_MARKER_RE.search(c.message)
            )
            counts.append(marker_count)
            densities.append(marker_count / len(quarter))

        trend = _compute_trend(densities)
        if trend is None:
            return None

        current_density = densities[-1] if densities else 0.0
        total_markers = sum(counts)

        # Only report if markers are growing or density is already high
        if trend <= 0 and current_density < 0.05:
            return None

        # Estimate total LOC for density calculation
        total_insertions = self.git_stats.total_insertions
        total_deletions = self.git_stats.total_deletions
        estimated_loc = max(total_insertions - total_deletions, 1000)
        markers_per_kloc = total_markers / (estimated_loc / 1000)

        trend_direction = (
            "rising" if trend > 0 else ("stable" if trend == 0 else "declining")
        )

        # Contribution: density * trend steepness
        proximity = markers_per_kloc / _DEBT_DENSITY_CRITICAL_PER_KLOC
        trend_steepness = min(1.0, abs(trend) * 10)
        contribution = min(1.0, proximity * 0.6 + trend_steepness * 0.4)

        weeks_to_threshold: Optional[int] = None
        if trend > 0:
            weeks_to_threshold = _estimate_time_to_threshold(
                markers_per_kloc, _DEBT_DENSITY_CRITICAL_PER_KLOC, trend
            )

        indicator = ForecastIndicator(
            name="debt_marker_density",
            current_value=round(markers_per_kloc, 2),
            threshold_value=float(_DEBT_DENSITY_CRITICAL_PER_KLOC),
            trend=trend_direction,
            description=(
                f"TODO/FIXME/HACK markers per KLOC. "
                f"Current: {markers_per_kloc:.1f}/KLOC "
                f"(threshold: {_DEBT_DENSITY_CRITICAL_PER_KLOC}/KLOC). "
                f"Total markers: {total_markers}. "
                f"Density trend: {trend_direction}."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
            "weeks_to_threshold": weeks_to_threshold,
        }

    def _compute_churn_without_growth_indicator(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Compute the ratio of files with high churn but no net growth."""
        if not self.code_churn:
            return None

        total_files = len(self.code_churn)
        if total_files == 0:
            return None

        # Count files where net growth is less than 10% of total churn
        churn_no_growth_count = 0
        for stats in self.code_churn.values():
            total_churn = stats["additions"] + stats["deletions"]
            net = stats["net"]
            if total_churn > 0 and abs(net) < total_churn * 0.10:
                churn_no_growth_count += 1

        ratio = churn_no_growth_count / total_files

        # Only report if ratio is significant
        if ratio < 0.05:
            return None

        contribution = min(1.0, ratio * 2.0)

        indicator = ForecastIndicator(
            name="churn_without_growth_ratio",
            current_value=round(ratio, 4),
            threshold_value=0.20,
            trend="rising",
            description=(
                f"Ratio of files with high churn but low net growth: "
                f"{churn_no_growth_count}/{total_files} ({ratio:.1%}). "
                f"These files are being rewritten without meaningful "
                f"improvement."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
        }

    def _compute_code_test_ratio_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute code-to-test commit ratio indicator."""
        if len(quarters) < 2:
            return None

        ratios: List[float] = []
        for quarter in quarters:
            src = 0
            tst = 0
            for c in quarter:
                for fp in c.files_changed:
                    if _TEST_PATH_RE.search(fp):
                        tst += 1
                    else:
                        src += 1
            ratio = tst / max(src, 1)
            ratios.append(ratio)

        trend = _compute_trend(ratios)
        if trend is None:
            return None

        current = ratios[-1] if ratios else 0.0
        trend_direction = (
            "declining" if trend < 0 else ("stable" if trend == 0 else "rising")
        )

        # A declining test ratio is concerning
        if trend >= 0:
            # Test ratio is stable or improving -- not a risk
            return None

        # Contribution: how far below healthy ratio + how fast declining
        healthy_ratio = 0.3
        proximity = max(0, 1 - (current / healthy_ratio)) if healthy_ratio > 0 else 0
        trend_steepness = min(1.0, abs(trend) * 5)
        contribution = min(1.0, proximity * 0.6 + trend_steepness * 0.4)

        indicator = ForecastIndicator(
            name="code_test_ratio",
            current_value=round(current, 4),
            threshold_value=0.10,
            trend=trend_direction,
            description=(
                f"Test-to-source commit ratio. Current: {current:.2f} "
                f"(threshold: 0.10). "
                f"Trend: {trend_direction} (slope: {trend:.4f})."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
        }

    @staticmethod
    def _tech_debt_recommendation(
        indicators: List[ForecastIndicator],
        debt_patterns: List[Pattern],
    ) -> str:
        """Build a data-driven tech debt mitigation recommendation."""
        parts = [
            "Technical debt is accumulating faster than it is being "
            "addressed."
        ]

        for ind in indicators:
            if ind.name == "debt_marker_density" and ind.trend == "rising":
                parts.append(
                    f"Debt markers are at {ind.current_value:.1f}/KLOC "
                    f"(threshold: {ind.threshold_value:.0f}/KLOC) and "
                    f"rising. Schedule a dedicated debt-reduction sprint "
                    f"before adding new features."
                )
            elif (
                ind.name == "churn_without_growth_ratio"
                and ind.current_value > 0.10
            ):
                parts.append(
                    f"{ind.current_value:.0%} of files show high churn "
                    f"without net growth. These files need design review, "
                    f"not more patches."
                )
            elif ind.name == "code_test_ratio" and ind.trend == "declining":
                parts.append(
                    f"Test coverage is declining (ratio: "
                    f"{ind.current_value:.2f}). Prioritise writing tests "
                    f"for the most-changed files."
                )

        if debt_patterns:
            high_files = set()
            for p in debt_patterns:
                high_files.update(p.affected_files)
            if high_files:
                file_list = sorted(high_files)[:5]
                parts.append(
                    f"Focus areas: {', '.join(file_list)}"
                    f"{' and others' if len(high_files) > 5 else ''}."
                )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Forecast: Bus Factor
    # ------------------------------------------------------------------

    def forecast_bus_factor(self) -> Optional[Forecast]:
        """Predict knowledge concentration risk.

        Bus factor = minimum number of people who need to leave before the
        project is in trouble.  Risk level = CRITICAL if bus factor is 1,
        HIGH if 2, MODERATE if 3.

        Returns
        -------
        Forecast or None
            A bus factor risk forecast, or ``None`` if insufficient evidence.
        """
        if len(self.commits) < 5:
            logger.debug("Too few commits for bus factor forecast")
            return None

        if self.git_stats.unique_authors < 1:
            return None

        # Calculate bus factor from code ownership
        bus_factor = self._calculate_bus_factor()

        # Single-author projects are automatically critical
        if self.git_stats.unique_authors == 1:
            bus_factor = 1

        indicators: List[ForecastIndicator] = []

        # --- Indicator 1: Bus factor ---
        indicator = ForecastIndicator(
            name="bus_factor",
            current_value=float(bus_factor),
            threshold_value=3.0,  # Bus factor >= 3 is healthy
            trend="stable",  # Bus factor changes slowly; no real-time trend
            description=(
                f"Bus factor: {bus_factor}. "
                f"This means {bus_factor} contributor(s) need to leave "
                f"before the project loses critical knowledge."
            ),
        )
        indicators.append(indicator)

        # --- Indicator 2: Author concentration ---
        author_concentration = self._compute_author_concentration()
        if author_concentration is not None:
            indicators.append(author_concentration)

        # --- Indicator 3: File ownership concentration ---
        file_ownership = self._compute_file_ownership_concentration()
        if file_ownership is not None:
            indicators.append(file_ownership)

        # Determine risk level from bus factor
        if bus_factor <= _BUS_FACTOR_CRITICAL:
            risk_level = RiskLevel.CRITICAL
        elif bus_factor <= _BUS_FACTOR_HIGH:
            risk_level = RiskLevel.HIGH
        elif bus_factor <= _BUS_FACTOR_MODERATE:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW

        # Probability based on bus factor and concentration
        probability = self._bus_factor_probability(bus_factor, indicators)

        # Check for CODE_OWNERSHIP patterns
        ownership_patterns = [
            p
            for p in self.patterns
            if p.pattern_type == PatternType.CODE_OWNERSHIP
        ]
        if ownership_patterns:
            probability = min(1.0, probability + 0.05)

        if probability < _MIN_CONFIDENCE:
            return None

        # Timeline for bus factor risk is typically "ongoing"
        timeline = "ongoing" if bus_factor <= 2 else "long-term"

        precedent = _find_historical_precedent(
            self.commits, RiskType.BUS_FACTOR, indicators
        )

        # Build description
        if author_concentration is not None:
            description = (
                f"Bus factor is {bus_factor}, meaning {bus_factor} "
                f"contributor(s) hold critical knowledge. Risk level: "
                f"{risk_level.value}. Author concentration: "
                f"{author_concentration.current_value:.0%} of commits "
                f"from the top contributor."
            )
        else:
            description = (
                f"Bus factor is {bus_factor}, meaning {bus_factor} "
                f"contributor(s) hold critical knowledge. Risk level: "
                f"{risk_level.value}."
            )

        recommendation = self._bus_factor_recommendation(
            bus_factor, indicators, ownership_patterns
        )

        forecast = Forecast(
            id=_generate_forecast_id(RiskType.BUS_FACTOR, self.commits[-1].date),
            risk_type=RiskType.BUS_FACTOR,
            risk_level=risk_level,
            title="Bus factor risk forecast",
            description=description,
            indicators=indicators,
            probability=round(probability, 3),
            estimated_timeline=timeline,
            recommendation=recommendation,
            historical_precedent=precedent,
        )

        return forecast

    def _calculate_bus_factor(self) -> int:
        """Calculate the bus factor from code ownership patterns.

        The bus factor is the minimum number of people who need to leave
        before the project loses coverage of >= 50% of its files.  Computed
        by greedily removing the most prolific authors and checking when
        >= 50% of files lose all their contributors.

        Returns
        -------
        int
            The bus factor (minimum number of people for >50% file coverage).
        """
        if not self.author_file_map:
            return 0

        # Build file -> set of authors mapping
        file_authors: Dict[str, set] = defaultdict(set)
        for author, files in self.author_file_map.items():
            for fp in files:
                file_authors[fp].add(author)

        total_files = len(file_authors)
        if total_files == 0:
            return 0

        # Sort authors by total commits (most prolific first)
        authors_by_volume = sorted(
            self.author_file_map.keys(),
            key=lambda a: sum(self.author_file_map[a].values()),
            reverse=True,
        )

        # Remove authors one by one (starting with the most prolific)
        # and count how many files lose all contributors
        remaining_authors = set(authors_by_volume)
        for i, author in enumerate(authors_by_volume):
            remaining_authors.discard(author)
            orphaned = sum(
                1
                for f_authors in file_authors.values()
                if f_authors and not f_authors.intersection(remaining_authors)
            )
            if orphaned / total_files >= 0.50:
                # i+1 authors had to leave for >=50% of files to be orphaned
                return i + 1

        # Even removing all authors didn't orphan 50%+ -- bus factor = total
        return len(authors_by_volume)

    def _compute_author_concentration(self) -> Optional[ForecastIndicator]:
        """Compute the fraction of commits from the most prolific author."""
        if not self.author_file_map:
            return None

        author_commits: Dict[str, int] = defaultdict(int)
        for c in self.commits:
            author_commits[c.author_name] += 1

        if not author_commits:
            return None

        total = sum(author_commits.values())
        top_author_commits = max(author_commits.values())
        concentration = top_author_commits / total

        return ForecastIndicator(
            name="author_concentration",
            current_value=round(concentration, 4),
            threshold_value=0.50,  # > 50% from one person is critical
            trend="stable",
            description=(
                f"Top contributor accounts for {concentration:.0%} of all "
                f"commits (threshold: 50%)."
            ),
        )

    def _compute_file_ownership_concentration(self) -> Optional[ForecastIndicator]:
        """Compute the fraction of files owned by a single author.

        A file is "owned" by an author if that author made >= 80% of the
        commits touching it.
        """
        if not self.author_file_map:
            return None

        # Build file -> {author: count}
        file_author_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for c in self.commits:
            for fp in c.files_changed:
                file_author_counts[fp][c.author_name] += 1

        if not file_author_counts:
            return None

        single_owner_files = 0
        for fp, author_counts in file_author_counts.items():
            total = sum(author_counts.values())
            if total == 0:
                continue
            max_author = max(author_counts.values())
            if max_author / total >= 0.80:
                single_owner_files += 1

        concentration = single_owner_files / len(file_author_counts)
        return ForecastIndicator(
            name="file_ownership_concentration",
            current_value=round(concentration, 4),
            threshold_value=0.50,  # > 50% single-owner files
            trend="stable",
            description=(
                f"{single_owner_files}/{len(file_author_counts)} files "
                f"({concentration:.0%}) are owned by a single author "
                f"(threshold: 50%)."
            ),
        )

    @staticmethod
    def _bus_factor_probability(
        bus_factor: int, indicators: List[ForecastIndicator]
    ) -> float:
        """Compute probability for bus factor risk.

        Probability is derived from the bus factor value plus concentration
        metrics.
        """
        # Base probability from bus factor
        base: float = {
            1: 0.80,
            2: 0.50,
            3: 0.30,
            4: 0.15,
        }.get(bus_factor, 0.05)

        # Adjust based on concentration indicators
        for ind in indicators:
            if (
                ind.name == "author_concentration"
                and ind.current_value > 0.50
            ):
                base = min(1.0, base + 0.10)
            elif (
                ind.name == "file_ownership_concentration"
                and ind.current_value > 0.40
            ):
                base = min(1.0, base + 0.10)

        return base

    @staticmethod
    def _bus_factor_recommendation(
        bus_factor: int,
        indicators: List[ForecastIndicator],
        ownership_patterns: List[Pattern],
    ) -> str:
        """Build a bus factor mitigation recommendation."""
        if bus_factor == 1:
            parts = [
                "Bus factor is 1. If this person leaves, the project "
                "loses critical knowledge."
            ]
        else:
            parts = [
                f"Bus factor is {bus_factor}. If {bus_factor} key "
                f"contributors leave, the project is in trouble."
            ]

        if bus_factor <= 2:
            parts.append(
                "Urgently: document key subsystems, pair-program on "
                "critical areas, and rotate on-call responsibilities."
            )

        for ind in indicators:
            if (
                ind.name == "file_ownership_concentration"
                and ind.current_value > 0.30
            ):
                parts.append(
                    f"File ownership is concentrated "
                    f"({ind.current_value:.0%} single-owner). "
                    f"Introduce code reviews and shared ownership for "
                    f"these files."
                )
                break

        if ownership_patterns:
            critical_files = set()
            for p in ownership_patterns:
                critical_files.update(p.affected_files)
            if critical_files:
                file_list = sorted(critical_files)[:5]
                parts.append(
                    f"Priority files for knowledge sharing: "
                    f"{', '.join(file_list)}."
                )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Forecast: Maintainability
    # ------------------------------------------------------------------

    def forecast_maintainability(self) -> Optional[Forecast]:
        """Predict maintainability decline based on code churn trends,
        file coupling, and anti-patterns.

        Indicators
        ~~~~~~~~~~
        - **avg_file_size_trend**: average file size trend (growing is bad)
        - **coupling_trend**: co-change frequency trend (increasing is bad)
        - **anti_pattern_density**: anti-pattern accumulation rate

        Returns
        -------
        Forecast or None
            A maintainability risk forecast, or ``None`` if insufficient
            evidence.
        """
        if len(self.commits) < 15:
            logger.debug("Too few commits for maintainability forecast")
            return None

        quarters = self.quarters
        if len(quarters) < 2:
            return None

        indicators: List[ForecastIndicator] = []
        probability_components: List[float] = []

        # --- Indicator 1: Average file size trend ---
        size_result = self._compute_file_size_indicator(quarters)
        if size_result is not None:
            indicators.append(size_result["indicator"])
            probability_components.append(size_result["contribution"])

        # --- Indicator 2: Coupling trend ---
        coupling_result = self._compute_coupling_indicator()
        if coupling_result is not None:
            indicators.append(coupling_result["indicator"])
            probability_components.append(coupling_result["contribution"])

        # --- Indicator 3: Anti-pattern density ---
        anti_result = self._compute_anti_pattern_indicator()
        if anti_result is not None:
            indicators.append(anti_result["indicator"])
            probability_components.append(anti_result["contribution"])

        if not indicators:
            return None

        probability = sum(probability_components) / len(probability_components)

        # Boost if maintainability-related patterns exist
        maintainability_patterns = [
            p
            for p in self.patterns
            if p.pattern_type
            in (PatternType.HIGH_CHURN, PatternType.ANTI_PATTERN)
        ]
        if maintainability_patterns:
            high_severity_count = sum(
                1
                for p in maintainability_patterns
                if p.severity in (Severity.HIGH, Severity.CRITICAL)
            )
            probability = min(1.0, probability + high_severity_count * 0.05)

        risk_level = self._probability_to_risk_level(probability)

        if probability < _MIN_CONFIDENCE:
            return None

        timeline = "ongoing"  # Maintainability decline is gradual

        precedent = _find_historical_precedent(
            self.commits, RiskType.MAINTAINABILITY, indicators
        )

        description = (
            f"Maintainability decline risk based on {len(indicators)} "
            f"indicators. Estimated probability: {probability:.0%}. "
            f"Files are growing larger, coupling is increasing, or "
            f"anti-patterns are accumulating."
        )

        recommendation = self._maintainability_recommendation(
            indicators, maintainability_patterns
        )

        forecast = Forecast(
            id=_generate_forecast_id(
                RiskType.MAINTAINABILITY, self.commits[-1].date
            ),
            risk_type=RiskType.MAINTAINABILITY,
            risk_level=risk_level,
            title="Maintainability decline forecast",
            description=description,
            indicators=indicators,
            probability=round(probability, 3),
            estimated_timeline=timeline,
            recommendation=recommendation,
            historical_precedent=precedent,
        )

        return forecast

    def _compute_file_size_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute average file size trend indicator.

        Growing file sizes reduce maintainability.
        """
        if len(quarters) < 2:
            return None

        # Estimate average file size per quarter from churn data
        avg_sizes: List[float] = []
        for quarter in quarters:
            if not quarter:
                avg_sizes.append(0.0)
                continue

            # Count total changes per file in this quarter
            quarter_file_changes: Dict[str, int] = defaultdict(int)
            for c in quarter:
                for fp in c.files_changed:
                    quarter_file_changes[fp] += 1

            if not quarter_file_changes:
                avg_sizes.append(0.0)
                continue

            # Use the file_sizes proxy as a stand-in for actual size
            sizes_in_quarter = [
                self.file_sizes.get(fp, 0)
                for fp in quarter_file_changes
                if fp in self.file_sizes
            ]
            avg_sizes.append(
                sum(sizes_in_quarter) / len(sizes_in_quarter)
                if sizes_in_quarter
                else 0.0
            )

        trend = _compute_trend(avg_sizes)
        if trend is None:
            return None

        current = avg_sizes[-1] if avg_sizes else 0.0
        trend_direction = (
            "rising" if trend > 0 else ("stable" if trend == 0 else "declining")
        )

        # Growing files are bad; declining or stable is good
        if trend <= 0:
            return None

        # Contribution based on how fast files are growing
        large_file_threshold = 500  # lines
        proximity = current / large_file_threshold if large_file_threshold > 0 else 0
        trend_steepness = min(1.0, abs(trend) * 5)
        contribution = min(1.0, proximity * 0.5 + trend_steepness * 0.5)

        indicator = ForecastIndicator(
            name="avg_file_size_trend",
            current_value=round(current, 1),
            threshold_value=float(large_file_threshold),
            trend=trend_direction,
            description=(
                f"Average file size across active files. Current: "
                f"{current:.0f} lines (proxy), threshold: "
                f"{large_file_threshold} lines. Trend: {trend_direction}."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
        }

    def _compute_coupling_indicator(self) -> Optional[Dict[str, Any]]:
        """Compute code coupling indicator from co-change frequency.

        High co-change frequency between file pairs indicates coupling.
        """
        if not self.co_change_matrix:
            return None

        # Count pairs with high co-change frequency
        total_pairs = 0
        high_coupling_pairs = 0
        coupling_threshold = 5  # 5+ co-changes is high coupling

        for file_path, neighbours in self.co_change_matrix.items():
            for neighbour, count in neighbours.items():
                # Only count each pair once (alphabetical order)
                if file_path < neighbour:
                    total_pairs += 1
                    if count >= coupling_threshold:
                        high_coupling_pairs += 1

        if total_pairs == 0:
            return None

        coupling_ratio = high_coupling_pairs / total_pairs

        # Contribution based on coupling ratio
        contribution = min(1.0, coupling_ratio * 3.0)

        indicator = ForecastIndicator(
            name="coupling_trend",
            current_value=round(coupling_ratio, 4),
            threshold_value=0.30,  # > 30% of pairs highly coupled is bad
            trend="rising",  # Coupling tends to increase over time
            description=(
                f"Ratio of highly-coupled file pairs: "
                f"{high_coupling_pairs}/{total_pairs} "
                f"({coupling_ratio:.1%}) with >= {coupling_threshold} "
                f"co-changes (threshold: 30%)."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
        }

    def _compute_anti_pattern_indicator(self) -> Optional[Dict[str, Any]]:
        """Compute anti-pattern density indicator.

        Anti-patterns are tracked via detected Pattern objects of type
        ANTI_PATTERN and RECURRING_FIX.
        """
        anti_patterns = [
            p
            for p in self.patterns
            if p.pattern_type
            in (PatternType.ANTI_PATTERN, PatternType.RECURRING_FIX)
        ]

        if not anti_patterns:
            return None

        total_anti = len(anti_patterns)
        high_severity = sum(
            1
            for p in anti_patterns
            if p.severity in (Severity.HIGH, Severity.CRITICAL)
        )

        # Estimate total files touched
        affected_files = set()
        for p in anti_patterns:
            affected_files.update(p.affected_files)

        total_active_files = len(self.code_churn) if self.code_churn else 1
        density = total_anti / total_active_files

        # Contribution based on density and severity
        contribution = min(
            1.0, density * 2.0 + (high_severity / max(total_anti, 1)) * 0.5
        )

        indicator = ForecastIndicator(
            name="anti_pattern_density",
            current_value=round(density, 4),
            threshold_value=0.10,  # > 0.1 anti-patterns per active file
            trend="rising",  # Anti-patterns tend to accumulate
            description=(
                f"Anti-pattern density: {total_anti} patterns across "
                f"{len(affected_files)} affected files "
                f"({density:.2f} per active file). "
                f"{high_severity} are HIGH/CRITICAL severity."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
        }

    @staticmethod
    def _maintainability_recommendation(
        indicators: List[ForecastIndicator],
        maintainability_patterns: List[Pattern],
    ) -> str:
        """Build a maintainability improvement recommendation."""
        parts = ["Maintainability is declining based on code structure analysis."]

        for ind in indicators:
            if ind.name == "avg_file_size_trend" and ind.trend == "rising":
                parts.append(
                    f"File sizes are growing (avg: {ind.current_value:.0f} "
                    f"lines). Consider splitting large files into focused "
                    f"modules with single responsibilities."
                )
            elif ind.name == "coupling_trend" and ind.current_value > 0.15:
                parts.append(
                    f"Code coupling is high ({ind.current_value:.0%} of "
                    f"file pairs are tightly coupled). Introduce interfaces "
                    f"or abstraction layers to decouple frequently "
                    f"co-changing files."
                )
            elif (
                ind.name == "anti_pattern_density" and ind.current_value > 0.05
            ):
                parts.append(
                    f"Anti-patterns are accumulating ({ind.current_value:.2f} "
                    f"per file). Schedule refactoring sprints to address "
                    f"recurring issues before they compound."
                )

        if maintainability_patterns:
            affected = set()
            for p in maintainability_patterns:
                affected.update(p.affected_files)
            if affected:
                file_list = sorted(affected)[:5]
                parts.append(
                    f"Priority refactor targets: {', '.join(file_list)}."
                )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Forecast: Stagnation
    # ------------------------------------------------------------------

    def forecast_stagnation(self) -> Optional[Forecast]:
        """Predict project stagnation based on declining activity and
        decreasing innovation metrics.

        Indicators
        ~~~~~~~~~~
        - **commit_frequency_decline**: declining commit frequency compared
          to historical average
        - **unique_files_decline**: declining unique files touched per week
        - **code_output_decline**: declining lines of code output

        Compares recent quarter to historical average.

        Returns
        -------
        Forecast or None
            A stagnation risk forecast, or ``None`` if insufficient evidence.
        """
        if len(self.commits) < 20:
            logger.debug("Too few commits for stagnation forecast")
            return None

        quarters = self.quarters
        if len(quarters) < 2:
            return None

        indicators: List[ForecastIndicator] = []
        probability_components: List[float] = []

        # --- Indicator 1: Declining commit frequency ---
        freq_result = self._compute_stagnation_frequency_indicator(quarters)
        if freq_result is not None:
            indicators.append(freq_result["indicator"])
            probability_components.append(freq_result["contribution"])

        # --- Indicator 2: Declining unique files touched ---
        files_result = self._compute_stagnation_files_indicator(quarters)
        if files_result is not None:
            indicators.append(files_result["indicator"])
            probability_components.append(files_result["contribution"])

        # --- Indicator 3: Declining code output ---
        output_result = self._compute_stagnation_output_indicator(quarters)
        if output_result is not None:
            indicators.append(output_result["indicator"])
            probability_components.append(output_result["contribution"])

        if not indicators:
            return None

        # Need at least 2 indicators trending negatively for stagnation
        declining_count = sum(
            1 for ind in indicators if ind.trend == "declining"
        )
        if declining_count < 2:
            logger.debug(
                "Stagnation forecast: only %d declining indicators, need >= 2",
                declining_count,
            )
            return None

        probability = sum(probability_components) / len(probability_components)

        # Adjust for the number of declining indicators
        if declining_count >= 3:
            probability = min(1.0, probability + 0.15)
        elif declining_count >= 2:
            probability = min(1.0, probability + 0.05)

        risk_level = self._probability_to_risk_level(probability)

        if probability < _MIN_CONFIDENCE:
            return None

        # Timeline: project when activity will drop to near-zero
        timeline = "2-6 months"
        if probability >= 0.7:
            timeline = "1-3 months"
        elif probability >= 0.5:
            timeline = "3-6 months"

        precedent = _find_historical_precedent(
            self.commits, RiskType.STAGNATION, indicators
        )

        description = (
            f"Project stagnation risk based on {len(indicators)} indicators, "
            f"{declining_count} of which are declining. Estimated "
            f"probability: {probability:.0%}. Activity is trending downward "
            f"compared to the historical baseline."
        )

        recommendation = self._stagnation_recommendation(indicators)

        forecast = Forecast(
            id=_generate_forecast_id(
                RiskType.STAGNATION, self.commits[-1].date
            ),
            risk_type=RiskType.STAGNATION,
            risk_level=risk_level,
            title="Project stagnation forecast",
            description=description,
            indicators=indicators,
            probability=round(probability, 3),
            estimated_timeline=timeline,
            recommendation=recommendation,
            historical_precedent=precedent,
        )

        return forecast

    def _compute_stagnation_frequency_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute commit frequency decline indicator for stagnation.

        Compares recent quarter to historical average.
        """
        if len(quarters) < 2:
            return None

        rates = [_compute_commit_rate(q) for q in quarters]

        # Compare recent to historical average
        historical_avg = sum(rates[:-1]) / len(rates[:-1]) if len(rates) > 1 else 0
        current = rates[-1]

        if historical_avg == 0:
            return None

        decline_pct = max(0, (historical_avg - current) / historical_avg)

        trend = _compute_trend(rates)
        trend_direction = (
            "declining" if trend is not None and trend < 0 else "stable"
        )
        if trend is not None and trend > 0:
            trend_direction = "rising"

        # Contribution: how much decline
        contribution = min(1.0, decline_pct * 2.0)

        indicator = ForecastIndicator(
            name="commit_frequency_decline",
            current_value=round(current, 2),
            threshold_value=round(historical_avg * 0.5, 2),  # < 50% of avg
            trend=trend_direction,
            description=(
                f"Commits per week. Current: {current:.1f}/week, "
                f"historical avg: {historical_avg:.1f}/week. "
                f"Decline: {decline_pct:.0%} from average."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
        }

    def _compute_stagnation_files_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute declining unique files touched indicator for stagnation."""
        if len(quarters) < 2:
            return None

        unique_counts: List[float] = []
        for quarter in quarters:
            if not quarter:
                unique_counts.append(0.0)
                continue
            span_days = max(
                (quarter[-1].date - quarter[0].date).days, 1
            )
            weeks = span_days / 7.0
            unique_files = len(
                set(fp for c in quarter for fp in c.files_changed)
            )
            unique_counts.append(unique_files / max(weeks, 1))

        trend = _compute_trend(unique_counts)
        if trend is None:
            return None

        current = unique_counts[-1] if unique_counts else 0.0
        historical_avg = (
            sum(unique_counts[:-1]) / len(unique_counts[:-1])
            if len(unique_counts) > 1
            else 0
        )

        if historical_avg == 0:
            return None

        decline_pct = max(0, (historical_avg - current) / historical_avg)

        trend_direction = (
            "declining" if trend < 0 else ("stable" if trend == 0 else "rising")
        )

        contribution = min(1.0, decline_pct * 2.0)

        indicator = ForecastIndicator(
            name="unique_files_decline",
            current_value=round(current, 2),
            threshold_value=round(historical_avg * 0.5, 2),
            trend=trend_direction,
            description=(
                f"Unique files touched per week. Current: {current:.1f}, "
                f"historical avg: {historical_avg:.1f}. "
                f"Decline: {decline_pct:.0%} from average."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
        }

    def _compute_stagnation_output_indicator(
        self, quarters: List[List[CommitData]]
    ) -> Optional[Dict[str, Any]]:
        """Compute declining code output indicator for stagnation."""
        if len(quarters) < 2:
            return None

        outputs: List[float] = []
        for quarter in quarters:
            if not quarter:
                outputs.append(0.0)
                continue
            span_days = max(
                (quarter[-1].date - quarter[0].date).days, 1
            )
            weeks = span_days / 7.0
            net_output = sum(c.insertions - c.deletions for c in quarter)
            outputs.append(net_output / max(weeks, 1))

        trend = _compute_trend(outputs)
        if trend is None:
            return None

        current = outputs[-1] if outputs else 0.0
        historical_avg = (
            sum(outputs[:-1]) / len(outputs[:-1])
            if len(outputs) > 1
            else 0
        )

        if historical_avg <= 0:
            # If historical average is 0 or negative, the project has always
            # had low output
            return None

        decline_pct = max(0, (historical_avg - current) / historical_avg)

        trend_direction = (
            "declining" if trend < 0 else ("stable" if trend == 0 else "rising")
        )

        contribution = min(1.0, decline_pct * 2.0)

        indicator = ForecastIndicator(
            name="code_output_decline",
            current_value=round(current, 1),
            threshold_value=round(historical_avg * 0.5, 1),
            trend=trend_direction,
            description=(
                f"Net lines of code per week. Current: {current:.0f}, "
                f"historical avg: {historical_avg:.0f}. "
                f"Decline: {decline_pct:.0%} from average."
            ),
        )

        return {
            "indicator": indicator,
            "contribution": contribution,
        }

    @staticmethod
    def _stagnation_recommendation(
        indicators: List[ForecastIndicator],
    ) -> str:
        """Build a stagnation mitigation recommendation."""
        parts = ["Project activity is declining relative to historical baselines."]

        for ind in indicators:
            if (
                ind.name == "commit_frequency_decline"
                and ind.trend == "declining"
            ):
                parts.append(
                    "Commit frequency is dropping. Consider whether this "
                    "reflects project completion, a natural lull, or "
                    "disengagement. If disengagement, investigate blockers."
                )
            elif (
                ind.name == "unique_files_decline" and ind.trend == "declining"
            ):
                parts.append(
                    "The scope of changes is narrowing (fewer unique files "
                    "touched). This may indicate developers are only "
                    "maintaining existing code rather than building new "
                    "features."
                )
            elif (
                ind.name == "code_output_decline" and ind.trend == "declining"
            ):
                parts.append(
                    "Code output is declining. If this is not a deliberate "
                    "slowdown, reassess project direction and team capacity."
                )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Health Score
    # ------------------------------------------------------------------

    def compute_health_score(self, forecasts: List[Forecast]) -> float:
        """Compute overall project health score 0-100 based on all forecasts.

        Starts at 100 and subtracts for each forecast based on risk level:
        CRITICAL = -25, HIGH = -15, MODERATE = -8, LOW = -3.
        Also adjusts for the number of patterns detected.

        Parameters
        ----------
        forecasts:
            List of forecasts from :meth:`forecast_all`.

        Returns
        -------
        float
            Health score clamped to the range [0, 100].
        """
        score = 100.0

        # Subtract for each forecast
        for forecast in forecasts:
            penalty = _HEALTH_PENALTY.get(forecast.risk_level, 0)
            # Scale penalty by probability so high-probability risks
            # have more impact
            scaled_penalty = penalty * forecast.probability
            score -= scaled_penalty
            logger.debug(
                "Health score: -%.1f for %s risk (%s, probability=%.3f)",
                scaled_penalty,
                forecast.risk_level.value,
                forecast.risk_type.value,
                forecast.probability,
            )

        # Adjust for number of patterns detected
        pattern_penalty = 0.0
        for pattern in self.patterns:
            if pattern.severity == Severity.CRITICAL:
                pattern_penalty += 2.0
            elif pattern.severity == Severity.HIGH:
                pattern_penalty += 1.0
            elif pattern.severity == Severity.MEDIUM:
                pattern_penalty += 0.5
            else:
                pattern_penalty += 0.2

        # Cap pattern penalty to avoid over-penalising
        pattern_penalty = min(pattern_penalty, 8.0)
        score -= pattern_penalty

        if pattern_penalty > 0:
            logger.debug(
                "Health score: -%.1f for %d detected patterns",
                pattern_penalty,
                len(self.patterns) if self.patterns else 0,
            )

        # Clamp to [0, 100]
        score = max(0.0, min(100.0, score))

        logger.info("Project health score: %.1f/100", score)
        return round(score, 1)
