"""Core pattern detection engine for the memoir project.

This module provides the :class:`PatternDetector` class which analyses commit
data and git statistics to detect **real, statistically grounded patterns**.
Every pattern produced must be backed by hard evidence -- no vibes, no
guessing, no "seems like".  If the data does not support a conclusion, the
pattern is not reported.

Detection Rules
~~~~~~~~~~~~~~~
1.  Every pattern must have **hard evidence** -- specific data points, commit
    hashes, dates, and measurements.
2.  **Confidence** is derived from the number and consistency of supporting
    data points.  Low evidence → low confidence.
3.  **Severity** reflects impact: a recurring fix in a test file is LOW; the
    same fix in a payment handler is CRITICAL.
4.  **Minimum thresholds**: at least 3 occurrences are required before calling
    something a "pattern".
5.  **Time-based analysis**: patterns show progression over time (improving or
    worsening), not just cumulative counts.
6.  **Recommendations** are data-driven and actionable, never generic advice.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from memoir.models.commit_data import CommitData, GitStats
from memoir.models.pattern import Pattern, PatternType, Severity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Commit messages that indicate a fix / bug-related change
_FIX_KEYWORD_RE = re.compile(
    r"\b(fix|bug|patch|hotfix|issue|close|closes|closed|resolve|resolves|resolved)\b",
    re.IGNORECASE,
)

# Technical-debt markers inside commit messages
_DEBT_MARKER_RE = re.compile(
    r"\b(TODO|FIXME|HACK|XXX|WORKAROUND|KLUDGE|TEMP|TEMPORARY)\b",
)

# Vague / low-effort commit message detection (mirrors git_analyzer logic)
_VAGUE_RE = re.compile(
    r"^(wip|fix|update|changes?|misc|stuff|cleanups?|tidy|tweaks?"
    r"|adjust|minor|fixes|updates|tmp|temp|hack|x)$",
    re.IGNORECASE,
)
_VAGUE_LENGTH_THRESHOLD = 10

# File-path heuristics for severity classification
_CRITICAL_PATH_RE = re.compile(
    r"(payment|billing|checkout|auth|security|encrypt|decrypt|transaction"
    r"|stripe|paypal|bank|money|invoice|receipt|credential|password|secret)",
    re.IGNORECASE,
)
_HIGH_PATH_RE = re.compile(
    r"(api|route|handler|controller|service|model|middleware|core|engine"
    r"|processor|scheduler|worker|queue|queue)",
    re.IGNORECASE,
)
_TEST_PATH_RE = re.compile(
    r"(test|spec|__tests__|mock|fixture|stub)",
    re.IGNORECASE,
)
_CONFIG_PATH_RE = re.compile(
    r"(\.ya?ml$|\.json$|\.toml$|\.ini$|\.cfg$|\.conf$|\.env|dockerfile"
    r"|makefile|\.lock$)",
    re.IGNORECASE,
)

# After-hours definition: before 07:00 or at/after 20:00 (mirrors git_analyzer)
_AFTER_HOURS_START = 20  # 20:00
_AFTER_HOURS_END = 7  # before 07:00


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _classify_file_severity(file_path: str) -> Severity:
    """Return a severity estimate for a file based on its path.

    This is a *heuristic* used only as a starting point -- detectors may
    promote or demote severity based on additional evidence.

    Parameters
    ----------
    file_path:
        Relative path of the file within the repository.

    Returns
    -------
    Severity
        Estimated severity based on the file's role.
    """
    if _CRITICAL_PATH_RE.search(file_path):
        return Severity.CRITICAL
    if _HIGH_PATH_RE.search(file_path):
        return Severity.HIGH
    if _TEST_PATH_RE.search(file_path):
        return Severity.LOW
    if _CONFIG_PATH_RE.search(file_path):
        return Severity.LOW
    return Severity.MEDIUM


def _is_vague_message(message: str) -> bool:
    """Return True if a commit message is considered vague.

    Mirrors the logic in :class:`GitAnalyzer.get_commit_message_patterns`.
    """
    subject = message.strip().split("\n", 1)[0].strip()
    normalised = subject.lower().rstrip(".!? ")
    return len(subject) < _VAGUE_LENGTH_THRESHOLD or bool(_VAGUE_RE.match(normalised))


def _split_into_quarters(
    commits: List[CommitData],
) -> List[List[CommitData]]:
    """Split commits chronologically into four equal-sized groups.

    If there are fewer than 4 commits, returns a single group containing all
    of them.

    Parameters
    ----------
    commits:
        Commits sorted chronologically (oldest first).

    Returns
    -------
    list[list[CommitData]]
        Up to 4 groups of commits, each spanning roughly 25 % of the time
        range.
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

    A positive slope means the metric is increasing over time; negative means
    decreasing.

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

    # Simple least-squares: slope = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
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


# ---------------------------------------------------------------------------
# PatternDetector
# ---------------------------------------------------------------------------


class PatternDetector:
    """Detects statistically grounded patterns from git commit history.

    Parameters
    ----------
    commits:
        Complete list of commits for the analysed range.
    git_stats:
        Aggregate repository statistics.
    work_pattern:
        Output of :meth:`GitAnalyzer.get_work_pattern`.
    message_patterns:
        Output of :meth:`GitAnalyzer.get_commit_message_patterns`.
    code_churn:
        Output of :meth:`GitAnalyzer.get_code_churn`.
    most_changed_files:
        Output of :meth:`GitAnalyzer.get_most_changed_files`.
    """

    def __init__(
        self,
        commits: List[CommitData],
        git_stats: GitStats,
        work_pattern: Dict[str, Any],
        message_patterns: Dict[str, Any],
        code_churn: Dict[str, Dict[str, int]],
        most_changed_files: List[Tuple[str, int]],
    ) -> None:
        self.commits = sorted(commits, key=lambda c: c.date)
        self.git_stats = git_stats
        self.work_pattern = work_pattern
        self.message_patterns = message_patterns
        self.code_churn = code_churn
        self.most_changed_files = most_changed_files

        # Pre-computed indices used by multiple detectors
        self._fix_commits: Optional[List[CommitData]] = None
        self._file_fix_dates: Optional[Dict[str, List[datetime]]] = None
        self._file_commit_dates: Optional[Dict[str, List[datetime]]] = None
        self._author_file_map: Optional[Dict[str, Dict[str, int]]] = None

        logger.info(
            "PatternDetector initialised with %d commits, %d files in churn data",
            len(self.commits),
            len(self.code_churn) if self.code_churn else 0,
        )

    # ------------------------------------------------------------------
    # Pre-computed indices (lazily built)
    # ------------------------------------------------------------------

    @property
    def fix_commits(self) -> List[CommitData]:
        """Commits whose messages indicate a fix or bug-related change."""
        if self._fix_commits is None:
            self._fix_commits = [
                c
                for c in self.commits
                if _FIX_KEYWORD_RE.search(c.message)
            ]
            logger.debug("Identified %d fix commits", len(self._fix_commits))
        return self._fix_commits

    @property
    def file_fix_dates(self) -> Dict[str, List[datetime]]:
        """Mapping of file path → list of dates when it was fixed."""
        if self._file_fix_dates is None:
            mapping: Dict[str, List[datetime]] = defaultdict(list)
            for c in self.fix_commits:
                for fp in c.files_changed:
                    mapping[fp].append(c.date)
            self._file_fix_dates = dict(mapping)
        return self._file_fix_dates

    @property
    def file_commit_dates(self) -> Dict[str, List[datetime]]:
        """Mapping of file path → list of commit dates (all commits)."""
        if self._file_commit_dates is None:
            mapping: Dict[str, List[datetime]] = defaultdict(list)
            for c in self.commits:
                for fp in c.files_changed:
                    mapping[fp].append(c.date)
            self._file_commit_dates = dict(mapping)
        return self._file_commit_dates

    @property
    def author_file_map(self) -> Dict[str, Dict[str, int]]:
        """Mapping of author → {file_path: commit_count}."""
        if self._author_file_map is None:
            mapping: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for c in self.commits:
                for fp in c.files_changed:
                    mapping[c.author_name][fp] += 1
            self._author_file_map = {
                author: dict(files) for author, files in mapping.items()
            }
        return self._author_file_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_all(self) -> List[Pattern]:
        """Run all pattern detectors and return found patterns.

        Returns
        -------
        list[Pattern]
            All patterns detected across every category, sorted by severity
            (most severe first) and then by confidence (highest first).
        """
        all_patterns: List[Pattern] = []

        detectors = [
            self.detect_recurring_fixes,
            self.detect_technical_debt,
            self.detect_burnout_indicators,
            self.detect_learning_curves,
            self.detect_code_ownership,
            self.detect_anti_patterns,
            self.detect_irregular_hours,
            self.detect_high_churn,
        ]

        for detector in detectors:
            try:
                patterns = detector()
                all_patterns.extend(patterns)
                logger.debug(
                    "Detector %s found %d patterns",
                    detector.__name__,
                    len(patterns),
                )
            except Exception:
                logger.error(
                    "Detector %s failed",
                    detector.__name__,
                    exc_info=True,
                )

        # Sort: CRITICAL first, then HIGH, MEDIUM, LOW; within same severity,
        # higher confidence first.
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        all_patterns.sort(
            key=lambda p: (severity_order.get(p.severity, 99), -p.confidence)
        )

        logger.info("Total patterns detected: %d", len(all_patterns))
        return all_patterns

    # ------------------------------------------------------------------
    # Detector: Recurring Fixes
    # ------------------------------------------------------------------

    def detect_recurring_fixes(self) -> List[Pattern]:
        """Find files/areas that are repeatedly fixed.

        A "fix commit" is one whose message contains keywords like "fix",
        "bug", "patch", "hotfix", or "issue" (case-insensitive).

        A file exhibits a recurring-fix pattern when:
        - It appears in **3 or more** fix commits, **and**
        - Those fixes span at least **2 distinct calendar weeks** (to rule
          out a single debugging session).

        Severity is derived from the file's importance (payment handler >
        source > test > config).  Confidence increases with the number of
        fix occurrences and the time span they cover.

        Returns
        -------
        list[Pattern]
            Detected recurring-fix patterns.
        """
        patterns: List[Pattern] = []
        if len(self.commits) < 5:
            logger.debug("Too few commits (%d) for recurring-fix detection", len(self.commits))
            return patterns

        for file_path, fix_dates in self.file_fix_dates.items():
            if len(fix_dates) < 3:
                continue

            # Ensure fixes span multiple distinct weeks
            distinct_weeks: set = set()
            for d in fix_dates:
                iso = d.isocalendar()
                distinct_weeks.add((iso[0], iso[1]))

            if len(distinct_weeks) < 2:
                continue

            sorted_dates = sorted(fix_dates)
            first_seen = sorted_dates[0]
            last_seen = sorted_dates[-1]
            span_days = (last_seen - first_seen).days

            # Collect the actual fix commits touching this file as evidence
            evidence_commits = [
                c for c in self.fix_commits if file_path in c.files_changed
            ]
            evidence: List[Dict[str, Any]] = [
                {
                    "commit": c.short_hash,
                    "date": c.date.isoformat(),
                    "message_subject": c.message.split("\n", 1)[0][:120],
                }
                for c in evidence_commits
            ]

            # Severity based on file path
            severity = _classify_file_severity(file_path)

            # Confidence: more occurrences and longer span → higher
            # Base: 0.3 for 3 fixes.  +0.05 per additional fix (capped at 0.4).
            # Span bonus: up to +0.2 for a span > 60 days.
            # Distinct-weeks bonus: +0.05 per extra distinct week (capped at +0.1).
            confidence = 0.3
            confidence += min(0.4, (len(fix_dates) - 3) * 0.05)
            confidence += min(0.2, span_days / 300.0)
            confidence += min(0.1, (len(distinct_weeks) - 2) * 0.05)
            confidence = round(min(1.0, confidence), 3)

            # Build actionable recommendation
            rec = self._recurring_fix_recommendation(
                file_path, len(fix_dates), span_days, severity
            )

            pattern = Pattern(
                pattern_type=PatternType.RECURRING_FIX,
                title=f"Recurring fixes in {file_path}",
                description=(
                    f"File '{file_path}' has been fixed {len(fix_dates)} times "
                    f"across {len(distinct_weeks)} distinct weeks over a "
                    f"{span_days}-day span.  This suggests a systemic issue "
                    f"rather than isolated bugs."
                ),
                first_seen=first_seen,
                last_seen=last_seen,
                occurrence_count=len(fix_dates),
                severity=severity,
                evidence=evidence,
                affected_files=[file_path],
                recommendation=rec,
                confidence=confidence,
            )
            patterns.append(pattern)

        logger.debug("Recurring-fix detector found %d patterns", len(patterns))
        return patterns

    @staticmethod
    def _recurring_fix_recommendation(
        file_path: str,
        fix_count: int,
        span_days: int,
        severity: Severity,
    ) -> str:
        """Build a data-driven recommendation for a recurring-fix pattern."""
        parts = [
            f"File '{file_path}' has been fixed {fix_count} times over "
            f"{span_days} days."
        ]

        if severity in (Severity.CRITICAL, Severity.HIGH):
            parts.append(
                "Given the critical nature of this file, consider a focused "
                "rewrite or architectural refactor to address the root cause, "
                "not just the symptoms."
            )
        else:
            parts.append(
                "Consider scheduling a refactor session to identify and "
                "eliminate the common root cause behind these recurring fixes."
            )

        if fix_count >= 5:
            parts.append(
                f"With {fix_count} fix commits, the cost of continued "
                f"patching likely exceeds the cost of a proper fix."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Detector: Technical Debt
    # ------------------------------------------------------------------

    def detect_technical_debt(self) -> List[Pattern]:
        """Detect growing technical debt.

        Evidence tracked:
        - Increasing TODO / FIXME / HACK mentions in commit messages over time.
        - Files with high churn but low (or negative) net growth (being
          rewritten without improvement).
        - Growing codebase size without a proportional increase in test-related
          commits.

        Returns
        -------
        list[Pattern]
            Detected technical-debt patterns.
        """
        patterns: List[Pattern] = []

        if len(self.commits) < 10:
            logger.debug("Too few commits for technical-debt detection")
            return patterns

        # --- Indicator 1: Growing debt markers in messages ---
        debt_marker_patterns = self._detect_growing_debt_markers()
        patterns.extend(debt_marker_patterns)

        # --- Indicator 2: High churn, low net growth files ---
        churn_no_growth_patterns = self._detect_churn_without_growth()
        patterns.extend(churn_no_growth_patterns)

        # --- Indicator 3: Code growing without test growth ---
        test_gap_patterns = self._detect_code_test_gap()
        patterns.extend(test_gap_patterns)

        logger.debug("Technical-debt detector found %d patterns", len(patterns))
        return patterns

    def _detect_growing_debt_markers(self) -> List[Pattern]:
        """Find increasing TODO/FIXME/HACK mentions over time."""
        patterns: List[Pattern] = []

        quarters = _split_into_quarters(self.commits)
        if len(quarters) < 3:
            return patterns

        marker_counts_by_quarter: List[int] = []
        marker_details_by_quarter: List[List[Dict[str, Any]]] = []

        for quarter in quarters:
            count = 0
            details: List[Dict[str, Any]] = []
            for c in quarter:
                matches = _DEBT_MARKER_RE.findall(c.message)
                if matches:
                    count += len(matches)
                    details.append(
                        {
                            "commit": c.short_hash,
                            "date": c.date.isoformat(),
                            "markers": matches,
                            "message_subject": c.message.split("\n", 1)[0][:120],
                        }
                    )
            marker_counts_by_quarter.append(count)
            marker_details_by_quarter.append(details)

        # Normalise by quarter size to get density
        densities = [
            marker_counts_by_quarter[i] / max(len(quarters[i]), 1)
            for i in range(len(quarters))
        ]

        trend = _compute_trend(densities)
        if trend is None or trend <= 0:
            return patterns  # Not growing → no pattern

        # Require at least 3 total debt markers to report
        total_markers = sum(marker_counts_by_quarter)
        if total_markers < 3:
            return patterns

        # Evidence: quarter-by-quarter breakdown
        evidence: List[Dict[str, Any]] = []
        for i, quarter in enumerate(quarters):
            if not quarter:
                continue
            evidence.append(
                {
                    "quarter_index": i + 1,
                    "period_start": quarter[0].date.isoformat(),
                    "period_end": quarter[-1].date.isoformat(),
                    "commits_in_quarter": len(quarter),
                    "debt_marker_count": marker_counts_by_quarter[i],
                    "debt_density": round(densities[i], 4),
                }
            )
        # Include representative examples from the most recent quarter
        if marker_details_by_quarter:
            last_quarter_details = marker_details_by_quarter[-1]
            for d in last_quarter_details[:5]:
                evidence.append({"recent_example": d})

        first_seen = self.commits[0].date
        last_seen = self.commits[-1].date

        # Confidence: based on trend strength and total markers
        confidence = min(1.0, 0.3 + total_markers * 0.02 + abs(trend) * 2.0)
        confidence = round(confidence, 3)

        severity = Severity.MEDIUM
        if total_markers >= 15 and trend > 0.1:
            severity = Severity.HIGH
        if total_markers >= 25 and trend > 0.2:
            severity = Severity.CRITICAL

        pattern = Pattern(
            pattern_type=PatternType.TECHNICAL_DEBT,
            title="Growing technical debt markers in commit messages",
            description=(
                f"TODO/FIXME/HACK mentions increased from "
                f"{marker_counts_by_quarter[0]} in Q1 to "
                f"{marker_counts_by_quarter[-1]} in the latest quarter "
                f"(density trend slope: {trend:.4f}).  This indicates "
                f"accumulating technical debt that is not being addressed."
            ),
            first_seen=first_seen,
            last_seen=last_seen,
            occurrence_count=total_markers,
            severity=severity,
            evidence=evidence,
            affected_files=[],
            recommendation=(
                f"Technical debt markers are increasing (trend slope: "
                f"{trend:.4f}).  Dedicate the next sprint to resolving "
                f"the {total_markers} accumulated TODO/FIXME/HACK items "
                f"before adding new features.  Prioritise items in files "
                f"that also exhibit high churn."
            ),
            confidence=confidence,
        )
        patterns.append(pattern)
        return patterns

    def _detect_churn_without_growth(self) -> List[Pattern]:
        """Find files with high churn but low or negative net growth.

        A file is flagged when:
        - Its total churn (additions + deletions) is in the top 25 % of all
          files, **and**
        - Its net growth (additions - deletions) is less than 10 % of its
          total churn (meaning most changes are rewrites, not additions).

        Returns
        -------
        list[Pattern]
            Patterns for files being rewritten without meaningful improvement.
        """
        patterns: List[Pattern] = []
        if not self.code_churn:
            return patterns

        churn_values = [
            v["additions"] + v["deletions"] for v in self.code_churn.values()
        ]
        if not churn_values:
            return patterns

        churn_values_sorted = sorted(churn_values, reverse=True)
        top_25_threshold = churn_values_sorted[
            max(0, len(churn_values_sorted) // 4)
        ]

        flagged_files: List[Tuple[str, Dict[str, int], int]] = []
        for file_path, stats in self.code_churn.items():
            total_churn = stats["additions"] + stats["deletions"]
            if total_churn < top_25_threshold:
                continue

            net = stats["net"]
            # Flag when net growth is less than 10 % of total churn
            if total_churn > 0 and abs(net) < total_churn * 0.10:
                flagged_files.append((file_path, stats, total_churn))

        if not flagged_files:
            return patterns

        # Sort by churn descending; keep only the top 10
        flagged_files.sort(key=lambda x: x[2], reverse=True)
        flagged_files = flagged_files[:10]

        for file_path, stats, total_churn in flagged_files:
            net = stats["net"]
            severity = _classify_file_severity(file_path)
            if total_churn > 1000:
                severity = Severity.HIGH

            evidence: List[Dict[str, Any]] = [
                {
                    "file": file_path,
                    "additions": stats["additions"],
                    "deletions": stats["deletions"],
                    "net_change": net,
                    "total_churn": total_churn,
                    "churn_to_net_ratio": (
                        round(total_churn / max(abs(net), 1), 1)
                    ),
                }
            ]

            confidence = min(1.0, 0.4 + total_churn / 5000.0)
            confidence = round(confidence, 3)

            commit_dates = self.file_commit_dates.get(file_path, [])
            first_seen = min(commit_dates) if commit_dates else self.git_stats.first_commit_date
            last_seen = max(commit_dates) if commit_dates else self.git_stats.last_commit_date

            pattern = Pattern(
                pattern_type=PatternType.TECHNICAL_DEBT,
                title=f"High churn without growth: {file_path}",
                description=(
                    f"File '{file_path}' has {total_churn} lines of churn "
                    f"({stats['additions']} additions, {stats['deletions']} "
                    f"deletions) but a net change of only {net} lines.  This "
                    f"suggests the file is being repeatedly rewritten without "
                    f"meaningful improvement -- a strong indicator of technical "
                    f"debt or unresolved design problems."
                ),
                first_seen=first_seen,
                last_seen=last_seen,
                occurrence_count=len(commit_dates),
                severity=severity,
                evidence=evidence,
                affected_files=[file_path],
                recommendation=(
                    f"File '{file_path}' has been churned {total_churn} lines "
                    f"with virtually no net growth (ratio: "
                    f"{round(total_churn / max(abs(net), 1), 1)}:1 churn-to-net). "
                    f"Before making further changes, invest time in understanding "
                    f"the root cause of the instability.  Consider a design "
                    f"review or splitting this file into smaller, more stable "
                    f"modules."
                ),
                confidence=confidence,
            )
            patterns.append(pattern)

        return patterns

    def _detect_code_test_gap(self) -> List[Pattern]:
        """Detect growing codebase without proportional test growth.

        Compares the rate of source-file commits vs. test-file commits across
        quarters.  A widening gap indicates accumulating untested code.
        """
        patterns: List[Pattern] = []

        quarters = _split_into_quarters(self.commits)
        if len(quarters) < 3:
            return patterns

        source_counts: List[int] = []
        test_counts: List[int] = []
        evidence: List[Dict[str, Any]] = []

        for i, quarter in enumerate(quarters):
            src = 0
            tst = 0
            for c in quarter:
                for fp in c.files_changed:
                    if _TEST_PATH_RE.search(fp):
                        tst += 1
                    else:
                        src += 1
            source_counts.append(src)
            test_counts.append(tst)
            evidence.append(
                {
                    "quarter_index": i + 1,
                    "source_file_commits": src,
                    "test_file_commits": tst,
                    "ratio": round(tst / max(src, 1), 4),
                }
            )

        # Compute trend of test-to-source ratio
        ratios = [
            test_counts[i] / max(source_counts[i], 1)
            for i in range(len(quarters))
        ]
        ratio_trend = _compute_trend(ratios)

        if ratio_trend is None:
            return patterns

        # Only flag if the ratio is *declining* (tests growing slower than code)
        if ratio_trend >= 0:
            return patterns

        # Require meaningful gap: latest quarter ratio < 0.2
        if ratios[-1] >= 0.2:
            return patterns

        confidence = min(1.0, 0.4 + abs(ratio_trend) * 5.0)
        confidence = round(confidence, 3)

        severity = Severity.MEDIUM
        if ratios[-1] < 0.05:
            severity = Severity.HIGH

        pattern = Pattern(
            pattern_type=PatternType.TECHNICAL_DEBT,
            title="Test coverage not keeping pace with code growth",
            description=(
                f"The test-to-source commit ratio has declined from "
                f"{ratios[0]:.4f} in Q1 to {ratios[-1]:.4f} in the latest "
                f"quarter (trend slope: {ratio_trend:.4f}).  Code is being "
                f"added significantly faster than tests, indicating growing "
                f"untested surface area."
            ),
            first_seen=self.commits[0].date,
            last_seen=self.commits[-1].date,
            occurrence_count=len(quarters),
            severity=severity,
            evidence=evidence,
            affected_files=[],
            recommendation=(
                f"The test-to-source ratio has dropped to {ratios[-1]:.2f}.  "
                f"Prioritise writing tests for new code before adding more "
                f"features.  Focus on the files with the highest churn and "
                f"fewest test commits first."
            ),
            confidence=confidence,
        )
        patterns.append(pattern)
        return patterns

    # ------------------------------------------------------------------
    # Detector: Burnout Indicators
    # ------------------------------------------------------------------

    def detect_burnout_indicators(self) -> List[Pattern]:
        """Detect burnout signals from commit patterns.

        Indicators tracked (need 2+ trending negative to flag):
        1. **After-hours ratio trend**: increasing over time.
        2. **Vague message ratio trend**: increasing over time.
        3. **Burst-then-silence cycles**: high activity followed by extended
           zero-commit periods (7+ days of silence after a burst).
        4. **Declining output**: commits per week decreasing over time.

        Returns
        -------
        list[Pattern]
            Detected burnout-indicator patterns.
        """
        patterns: List[Pattern] = []

        if len(self.commits) < 15:
            logger.debug("Too few commits for burnout detection")
            return patterns

        negative_indicators = 0
        evidence: List[Dict[str, Any]] = []
        indicator_descriptions: List[str] = []

        quarters = _split_into_quarters(self.commits)

        # --- Indicator 1: After-hours ratio trend ---
        after_hours_trend = self._compute_after_hours_trend(quarters)
        if after_hours_trend is not None and after_hours_trend > 0:
            negative_indicators += 1
            indicator_descriptions.append("increasing after-hours ratio")
            evidence.append(
                {
                    "indicator": "after_hours_ratio_trend",
                    "slope": round(after_hours_trend, 4),
                    "direction": "increasing",
                }
            )

        # --- Indicator 2: Vague message ratio trend ---
        vague_trend = self._compute_vague_message_trend(quarters)
        if vague_trend is not None and vague_trend > 0:
            negative_indicators += 1
            indicator_descriptions.append("increasing vague message ratio")
            evidence.append(
                {
                    "indicator": "vague_message_ratio_trend",
                    "slope": round(vague_trend, 4),
                    "direction": "increasing",
                }
            )

        # --- Indicator 3: Burst-then-silence cycles ---
        burst_silence = self._detect_burst_silence_cycles()
        if burst_silence["count"] >= 2:
            negative_indicators += 1
            indicator_descriptions.append(
                f"{burst_silence['count']} burst-then-silence cycles"
            )
            evidence.append(
                {
                    "indicator": "burst_silence_cycles",
                    "count": burst_silence["count"],
                    "details": burst_silence["details"][:5],
                }
            )

        # --- Indicator 4: Declining output ---
        output_trend = self._compute_output_trend(quarters)
        if output_trend is not None and output_trend < 0:
            negative_indicators += 1
            indicator_descriptions.append("declining commit output")
            evidence.append(
                {
                    "indicator": "commit_output_trend",
                    "slope": round(output_trend, 4),
                    "direction": "declining",
                }
            )

        if negative_indicators < 2:
            return patterns

        confidence = min(1.0, 0.3 + negative_indicators * 0.2)
        confidence = round(confidence, 3)

        severity = Severity.MEDIUM
        if negative_indicators >= 3:
            severity = Severity.HIGH
        if negative_indicators >= 4:
            severity = Severity.CRITICAL

        indicator_text = ", ".join(indicator_descriptions)

        pattern = Pattern(
            pattern_type=PatternType.BURNOUT_INDICATOR,
            title="Potential burnout indicators detected",
            description=(
                f"{negative_indicators} burnout indicators observed: "
                f"{indicator_text}.  These signals are derived from commit "
                f"patterns, not self-reporting, so they may reflect project "
                f"pressures rather than personal state."
            ),
            first_seen=self.commits[0].date,
            last_seen=self.commits[-1].date,
            occurrence_count=negative_indicators,
            severity=severity,
            evidence=evidence,
            affected_files=[],
            recommendation=self._burnout_recommendation(
                negative_indicators, indicator_descriptions
            ),
            confidence=confidence,
        )
        patterns.append(pattern)
        return patterns

    def _compute_after_hours_trend(
        self, quarters: List[List[CommitData]]
    ) -> Optional[float]:
        """Compute the trend of after-hours commit ratios across quarters."""
        if len(quarters) < 3:
            return None

        ratios: List[float] = []
        for quarter in quarters:
            if not quarter:
                ratios.append(0.0)
                continue
            after_hours = sum(
                1 for c in quarter if c.date.hour < _AFTER_HOURS_END or c.date.hour >= _AFTER_HOURS_START
            )
            ratios.append(after_hours / len(quarter))

        return _compute_trend(ratios)

    def _compute_vague_message_trend(
        self, quarters: List[List[CommitData]]
    ) -> Optional[float]:
        """Compute the trend of vague message ratios across quarters."""
        if len(quarters) < 3:
            return None

        ratios: List[float] = []
        for quarter in quarters:
            if not quarter:
                ratios.append(0.0)
                continue
            vague = sum(1 for c in quarter if _is_vague_message(c.message))
            ratios.append(vague / len(quarter))

        return _compute_trend(ratios)

    def _detect_burst_silence_cycles(self) -> Dict[str, Any]:
        """Detect burst-then-silence cycles.

        A cycle is defined as: a week with 5+ commits followed by 7+ days
        with 0 commits.
        """
        if len(self.commits) < 10:
            return {"count": 0, "details": []}

        # Group commits by week
        weekly: Dict[str, List[CommitData]] = defaultdict(list)
        for c in self.commits:
            iso_year, iso_week, _ = c.date.isocalendar()
            key = f"{iso_year}-W{iso_week:02d}"
            weekly[key].append(c)

        sorted_weeks = sorted(weekly.keys())
        details: List[Dict[str, Any]] = []
        cycle_count = 0

        for i, week_key in enumerate(sorted_weeks):
            week_commits = len(weekly[week_key])
            if week_commits < 5:
                continue

            # Check for silence after this burst
            burst_end = max(c.date for c in weekly[week_key])

            # Look ahead for 7+ day silence
            next_commits_after_burst = [
                c for c in self.commits if c.date > burst_end
            ]
            if not next_commits_after_burst:
                # End of data -- can't confirm silence
                continue

            next_commit_date = min(c.date for c in next_commits_after_burst)
            silence_days = (next_commit_date - burst_end).days

            if silence_days >= 7:
                cycle_count += 1
                details.append(
                    {
                        "burst_week": week_key,
                        "burst_commit_count": week_commits,
                        "burst_end": burst_end.isoformat(),
                        "silence_days": silence_days,
                        "next_commit": next_commit_date.isoformat(),
                    }
                )

        return {"count": cycle_count, "details": details}

    def _compute_output_trend(
        self, quarters: List[List[CommitData]]
    ) -> Optional[float]:
        """Compute the trend of commit output (commits per week) across quarters."""
        if len(quarters) < 3:
            return None

        rates: List[float] = []
        for quarter in quarters:
            if not quarter:
                rates.append(0.0)
                continue
            span_days = max(
                (quarter[-1].date - quarter[0].date).days, 1
            )
            rates.append(len(quarter) / span_days * 7.0)

        return _compute_trend(rates)

    @staticmethod
    def _burnout_recommendation(
        indicator_count: int,
        indicator_descriptions: List[str],
    ) -> str:
        """Build a burnout-specific recommendation."""
        parts = [
            f"{indicator_count} burnout indicators detected: "
            + "; ".join(indicator_descriptions) + "."
        ]

        if "increasing after-hours ratio" in indicator_descriptions:
            parts.append(
                "After-hours commits are increasing -- consider setting "
                "boundaries around work time or investigating whether "
                "deadline pressure is the root cause."
            )

        if "increasing vague message ratio" in indicator_descriptions:
            parts.append(
                "Vague commit messages are increasing -- this often correlates "
                "with rushing.  Slowing down and writing descriptive messages "
                "can improve both code quality and personal sense of control."
            )

        if any("burst-then-silence" in d for d in indicator_descriptions):
            parts.append(
                "Burst-then-silence cycles suggest unsustainable pacing.  "
                "Aim for a more even distribution of work to avoid "
                "feast-or-famine patterns."
            )

        if "declining commit output" in indicator_descriptions:
            parts.append(
                "Declining output may indicate disengagement or exhaustion.  "
                "If this is a team signal, consider workload rebalancing."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Detector: Learning Curves
    # ------------------------------------------------------------------

    def detect_learning_curves(self) -> List[Pattern]:
        """Detect learning patterns from commit history.

        Evidence for improvement:
        - Decreasing time between similar mistakes (fix commits on the same
          file becoming less frequent over time).
        - Improving commit message quality (vague ratio decreasing).
        - Decreasing churn on specific files over time (stabilising).

        Returns
        -------
        list[Pattern]
            Detected learning-curve patterns (positive indicators).
        """
        patterns: List[Pattern] = []

        if len(self.commits) < 10:
            logger.debug("Too few commits for learning-curve detection")
            return patterns

        # --- Indicator 1: Decreasing fix frequency on specific files ---
        fix_learning = self._detect_fix_learning()
        patterns.extend(fix_learning)

        # --- Indicator 2: Improving message quality ---
        msg_learning = self._detect_message_quality_improvement()
        patterns.extend(msg_learning)

        # --- Indicator 3: Decreasing churn on specific files ---
        churn_learning = self._detect_churn_stabilisation()
        patterns.extend(churn_learning)

        logger.debug("Learning-curve detector found %d patterns", len(patterns))
        return patterns

    def _detect_fix_learning(self) -> List[Pattern]:
        """Find files where fix frequency is decreasing over time."""
        patterns: List[Pattern] = []

        for file_path, fix_dates in self.file_fix_dates.items():
            if len(fix_dates) < 4:
                continue

            sorted_dates = sorted(fix_dates)
            # Compute intervals between consecutive fixes in days
            intervals: List[float] = []
            for i in range(1, len(sorted_dates)):
                delta = (sorted_dates[i] - sorted_dates[i - 1]).total_seconds() / 86400.0
                intervals.append(delta)

            if len(intervals) < 3:
                continue

            trend = _compute_trend(intervals)
            if trend is None or trend <= 0:
                continue  # Intervals not growing → not learning

            # Get fix commits for evidence
            fix_commits_for_file = [
                c for c in self.fix_commits if file_path in c.files_changed
            ]
            evidence: List[Dict[str, Any]] = [
                {
                    "file": file_path,
                    "fix_count": len(fix_dates),
                    "interval_trend_days": round(trend, 2),
                    "first_interval_days": round(intervals[0], 1),
                    "last_interval_days": round(intervals[-1], 1),
                    "fix_dates": [d.isoformat() for d in sorted_dates],
                }
            ]

            confidence = min(1.0, 0.3 + len(fix_dates) * 0.05 + abs(trend) * 0.1)
            confidence = round(confidence, 3)

            first_seen = sorted_dates[0]
            last_seen = sorted_dates[-1]

            pattern = Pattern(
                pattern_type=PatternType.LEARNING_CURVE,
                title=f"Improving stability: {file_path}",
                description=(
                    f"Fix intervals for '{file_path}' are increasing "
                    f"(trend: +{trend:.1f} days between fixes), from "
                    f"{intervals[0]:.1f} days initially to "
                    f"{intervals[-1]:.1f} days recently.  This indicates "
                    f"learning or stabilisation -- the file is breaking less "
                    f"frequently over time."
                ),
                first_seen=first_seen,
                last_seen=last_seen,
                occurrence_count=len(fix_dates),
                severity=Severity.LOW,
                evidence=evidence,
                affected_files=[file_path],
                recommendation=(
                    f"Good news: '{file_path}' is stabilising.  The knowledge "
                    f"gained from earlier fixes is paying off.  Continue "
                    f"current practices and consider documenting the lessons "
                    f"learned for the benefit of other team members."
                ),
                confidence=confidence,
            )
            patterns.append(pattern)

        return patterns

    def _detect_message_quality_improvement(self) -> List[Pattern]:
        """Detect improving commit message quality over time."""
        patterns: List[Pattern] = []

        quarters = _split_into_quarters(self.commits)
        if len(quarters) < 3:
            return patterns

        vague_ratios: List[float] = []
        avg_lengths: List[float] = []
        evidence: List[Dict[str, Any]] = []

        for i, quarter in enumerate(quarters):
            if not quarter:
                vague_ratios.append(0.0)
                avg_lengths.append(0.0)
                continue
            vague = sum(1 for c in quarter if _is_vague_message(c.message))
            vague_ratios.append(vague / len(quarter))
            avg_len = sum(len(c.message) for c in quarter) / len(quarter)
            avg_lengths.append(avg_len)
            evidence.append(
                {
                    "quarter_index": i + 1,
                    "commits": len(quarter),
                    "vague_ratio": round(vague / len(quarter), 4),
                    "avg_message_length": round(avg_len, 1),
                }
            )

        vague_trend = _compute_trend(vague_ratios)
        length_trend = _compute_trend(avg_lengths)

        # Improvement = vague ratio decreasing AND/OR message length increasing
        improving = False
        descriptions: List[str] = []

        if vague_trend is not None and vague_trend < 0:
            improving = True
            descriptions.append(
                f"vague message ratio declining (slope: {vague_trend:.4f})"
            )
        if length_trend is not None and length_trend > 0:
            improving = True
            descriptions.append(
                f"message length increasing (slope: {length_trend:.2f})"
            )

        if not improving:
            return patterns

        confidence = min(1.0, 0.4 + abs(vague_trend or 0) * 3.0 + abs(length_trend or 0) * 0.5)
        confidence = round(confidence, 3)

        pattern = Pattern(
            pattern_type=PatternType.LEARNING_CURVE,
            title="Improving commit message quality",
            description=(
                f"Commit message quality is improving over time: "
                + "; ".join(descriptions)
                + ".  This suggests growing attention to communication and "
                "documentation practices."
            ),
            first_seen=self.commits[0].date,
            last_seen=self.commits[-1].date,
            occurrence_count=len(quarters),
            severity=Severity.LOW,
            evidence=evidence,
            affected_files=[],
            recommendation=(
                "Commit message quality is trending upward.  Reinforce this "
                "behaviour by acknowledging good messages in code review.  "
                "Consider documenting message conventions if not already done."
            ),
            confidence=confidence,
        )
        patterns.append(pattern)
        return patterns

    def _detect_churn_stabilisation(self) -> List[Pattern]:
        """Find files where churn is decreasing over time."""
        patterns: List[Pattern] = []

        # Need at least 10 commits per file for meaningful analysis
        for file_path, commit_dates in self.file_commit_dates.items():
            if len(commit_dates) < 10:
                continue

            # Split file's commits into halves and compare churn
            sorted_dates = sorted(commit_dates)
            mid = len(sorted_dates) // 2
            first_half_end = sorted_dates[mid - 1]
            second_half_start = sorted_dates[mid]

            # Get per-commit churn for this file
            first_half_churn = 0
            second_half_churn = 0
            first_half_count = 0
            second_half_count = 0

            for c in self.commits:
                if file_path not in c.files_changed:
                    continue
                churn = c.insertions + c.deletions
                if c.date <= first_half_end:
                    first_half_churn += churn
                    first_half_count += 1
                else:
                    second_half_churn += churn
                    second_half_count += 1

            if first_half_count < 5 or second_half_count < 5:
                continue

            # Normalise: churn per commit
            first_rate = first_half_churn / first_half_count
            second_rate = second_half_churn / second_half_count

            # Require at least 30 % reduction to call it stabilisation
            if first_rate == 0 or second_rate >= first_rate * 0.7:
                continue

            reduction_pct = round((1 - second_rate / first_rate) * 100, 1)

            evidence: List[Dict[str, Any]] = [
                {
                    "file": file_path,
                    "first_half_avg_churn_per_commit": round(first_rate, 1),
                    "second_half_avg_churn_per_commit": round(second_rate, 1),
                    "reduction_percent": reduction_pct,
                    "first_half_commits": first_half_count,
                    "second_half_commits": second_half_count,
                }
            ]

            confidence = min(1.0, 0.3 + reduction_pct / 200.0)
            confidence = round(confidence, 3)

            pattern = Pattern(
                pattern_type=PatternType.LEARNING_CURVE,
                title=f"Stabilising code: {file_path}",
                description=(
                    f"Average churn per commit on '{file_path}' decreased by "
                    f"{reduction_pct}% (from {first_rate:.1f} to "
                    f"{second_rate:.1f} lines/commit).  The code is settling "
                    f"into a stable state, requiring fewer rewrites."
                ),
                first_seen=sorted_dates[0],
                last_seen=sorted_dates[-1],
                occurrence_count=len(commit_dates),
                severity=Severity.LOW,
                evidence=evidence,
                affected_files=[file_path],
                recommendation=(
                    f"'{file_path}' is stabilising with a {reduction_pct}% "
                    f"reduction in per-commit churn.  This is a positive "
                    f"signal.  The understanding gained from earlier iterations "
                    f"is paying off in more targeted, efficient changes."
                ),
                confidence=confidence,
            )
            patterns.append(pattern)

        return patterns

    # ------------------------------------------------------------------
    # Detector: Code Ownership / Bus Factor
    # ------------------------------------------------------------------

    def detect_code_ownership(self) -> List[Pattern]:
        """Detect bus factor issues.

        A file has a code ownership problem when:
        - One author contributes ≥ 80 % of the commits to that file, **and**
        - The file has been touched by at least 5 commits total.

        The overall bus factor is also computed: the minimum number of
        authors whose departure would leave ≥ 50 % of files without a
        knowledgeable contributor.

        Returns
        -------
        list[Pattern]
            Detected code-ownership patterns.
        """
        patterns: List[Pattern] = []

        if len(self.commits) < 10 or self.git_stats.unique_authors < 2:
            logger.debug("Insufficient data for code-ownership detection")
            return patterns

        # --- Per-file ownership analysis ---
        for file_path, commit_dates in self.file_commit_dates.items():
            if len(commit_dates) < 5:
                continue

            # Count commits per author for this file
            author_commits: Dict[str, int] = defaultdict(int)
            for c in self.commits:
                if file_path in c.files_changed:
                    author_commits[c.author_name] += 1

            if not author_commits:
                continue

            total = sum(author_commits.values())
            dominant_author = max(author_commits, key=author_commits.get)
            dominant_pct = author_commits[dominant_author] / total

            if dominant_pct < 0.8:
                continue

            # Require the file to be touched by at least 5 commits
            if total < 5:
                continue

            severity = _classify_file_severity(file_path)
            if dominant_pct >= 0.95:
                severity = Severity.HIGH
                if _CRITICAL_PATH_RE.search(file_path):
                    severity = Severity.CRITICAL

            other_authors = [
                a for a in author_commits if a != dominant_author
            ]

            evidence: List[Dict[str, Any]] = [
                {
                    "file": file_path,
                    "dominant_author": dominant_author,
                    "dominant_pct": round(dominant_pct * 100, 1),
                    "dominant_commits": author_commits[dominant_author],
                    "total_commits": total,
                    "other_authors": other_authors,
                    "other_author_commits": {
                        a: author_commits[a] for a in other_authors
                    },
                }
            ]

            confidence = min(1.0, 0.4 + dominant_pct * 0.4 + (total / 50.0) * 0.2)
            confidence = round(confidence, 3)

            first_seen = min(commit_dates)
            last_seen = max(commit_dates)

            pattern = Pattern(
                pattern_type=PatternType.CODE_OWNERSHIP,
                title=f"Single-author dominance: {file_path}",
                description=(
                    f"'{file_path}' is {round(dominant_pct * 100, 1)}% "
                    f"authored by {dominant_author} "
                    f"({author_commits[dominant_author]}/{total} commits).  "
                    f"Only {len(other_authors)} other author(s) have touched "
                    f"this file.  This creates a bus-factor risk."
                ),
                first_seen=first_seen,
                last_seen=last_seen,
                occurrence_count=total,
                severity=severity,
                evidence=evidence,
                affected_files=[file_path],
                recommendation=(
                    f"'{file_path}' is effectively owned by {dominant_author} "
                    f"({round(dominant_pct * 100, 1)}% of commits).  "
                    f"If {dominant_author} leaves, this file will have no "
                    f"knowledgeable maintainer.  Schedule pair-programming or "
                    f"code-review rotation to spread knowledge.  "
                    f"Target: at least 2 authors with ≥ 20 % contribution."
                ),
                confidence=confidence,
            )
            patterns.append(pattern)

        # --- Overall bus factor ---
        bus_factor_pattern = self._compute_bus_factor()
        if bus_factor_pattern is not None:
            patterns.append(bus_factor_pattern)

        logger.debug("Code-ownership detector found %d patterns", len(patterns))
        return patterns

    def _compute_bus_factor(self) -> Optional[Pattern]:
        """Compute the overall bus factor for the repository.

        Bus factor = the minimum number of authors whose removal would
        leave ≥ 50 % of frequently-changed files without a primary
        contributor.
        """
        # Consider files with 5+ commits
        significant_files = {
            fp
            for fp, dates in self.file_commit_dates.items()
            if len(dates) >= 5
        }

        if len(significant_files) < 3:
            return None

        # For each file, find the primary author
        file_primary: Dict[str, str] = {}
        for c in self.commits:
            for fp in c.files_changed:
                if fp not in significant_files:
                    continue
                # We'll compute this after collecting all author data
                pass

        # Compute per-file author counts
        file_author_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for c in self.commits:
            for fp in c.files_changed:
                if fp in significant_files:
                    file_author_counts[fp][c.author_name] += 1

        # Determine primary author per file
        for fp, author_counts in file_author_counts.items():
            if author_counts:
                file_primary[fp] = max(author_counts, key=author_counts.get)

        # For each author, count how many files they are primary for
        author_file_count: Dict[str, int] = Counter(file_primary.values())

        # Sort authors by number of files they own (descending)
        sorted_authors = sorted(
            author_file_count.items(), key=lambda x: x[1], reverse=True
        )

        # Bus factor: how many top authors need to leave before ≥ 50 % of
        # files lose their primary contributor?
        total_files = len(file_primary)
        files_lost = 0
        bus_factor = 0

        for author, count in sorted_authors:
            files_lost += count
            bus_factor += 1
            if files_lost >= total_files * 0.5:
                break

        if bus_factor >= min(self.git_stats.unique_authors, 3):
            # Healthy bus factor -- not worth reporting as a problem
            return None

        evidence: List[Dict[str, Any]] = [
            {
                "bus_factor": bus_factor,
                "total_significant_files": total_files,
                "author_file_ownership": [
                    {"author": a, "files_as_primary": c}
                    for a, c in sorted_authors[:10]
                ],
            }
        ]

        severity = Severity.MEDIUM
        if bus_factor == 1:
            severity = Severity.HIGH
        if bus_factor == 1 and _CRITICAL_PATH_RE.search(
            " ".join(file_primary.keys())
        ):
            severity = Severity.CRITICAL

        confidence = min(1.0, 0.5 + total_files / 100.0)
        confidence = round(confidence, 3)

        return Pattern(
            pattern_type=PatternType.CODE_OWNERSHIP,
            title=f"Low bus factor: {bus_factor}",
            description=(
                f"The project's bus factor is {bus_factor} -- only "
                f"{bus_factor} author{'s' if bus_factor != 1 else ''} need"
                f"{'s' if bus_factor == 1 else ''} to leave before 50 % of "
                f"significant files ({total_files} files with 5+ commits) "
                f"lose their primary contributor."
            ),
            first_seen=self.commits[0].date,
            last_seen=self.commits[-1].date,
            occurrence_count=bus_factor,
            severity=severity,
            evidence=evidence,
            affected_files=list(file_primary.keys())[:20],
            recommendation=(
                f"Bus factor is {bus_factor}.  Cross-training is critical: "
                f"ensure every significant file has at least 2 authors with "
                f"meaningful contributions (≥ 20 % of commits).  Start with "
                f"the files owned by the most concentrated single author."
            ),
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Detector: Anti-Patterns
    # ------------------------------------------------------------------

    def detect_anti_patterns(self) -> List[Pattern]:
        """Detect emerging anti-patterns.

        Evidence tracked:
        1. **Files growing rapidly in size** (commit count accelerating).
        2. **Increasing coupling**: sets of files that consistently change
           together across commits.
        3. **God objects**: files touched by an unusually high fraction of
           total commits.

        Returns
        -------
        list[Pattern]
            Detected anti-patterns.
        """
        patterns: List[Pattern] = []

        if len(self.commits) < 10:
            logger.debug("Too few commits for anti-pattern detection")
            return patterns

        # --- Indicator 1: Rapidly growing files ---
        growing = self._detect_rapidly_growing_files()
        patterns.extend(growing)

        # --- Indicator 2: Increasing coupling ---
        coupling = self._detect_increasing_coupling()
        patterns.extend(coupling)

        # --- Indicator 3: God objects ---
        god_objects = self._detect_god_objects()
        patterns.extend(god_objects)

        logger.debug("Anti-pattern detector found %d patterns", len(patterns))
        return patterns

    def _detect_rapidly_growing_files(self) -> List[Pattern]:
        """Find files whose commit frequency is accelerating."""
        patterns: List[Pattern] = []

        quarters = _split_into_quarters(self.commits)
        if len(quarters) < 3:
            return patterns

        # Count commits per file per quarter
        file_quarter_counts: Dict[str, List[int]] = defaultdict(
            lambda: [0] * len(quarters)
        )

        for qi, quarter in enumerate(quarters):
            for c in quarter:
                for fp in c.files_changed:
                    file_quarter_counts[fp][qi] += 1

        for file_path, counts in file_quarter_counts.items():
            total = sum(counts)
            if total < 5:
                continue

            # Check if commit frequency is accelerating
            trend = _compute_trend([float(c) for c in counts])
            if trend is None or trend <= 0:
                continue

            # Require at least a 2x increase from first to last quarter
            if counts[-1] < counts[0] * 2:
                continue

            severity = _classify_file_severity(file_path)
            if counts[-1] >= 3 * counts[0]:
                severity = Severity.HIGH

            evidence: List[Dict[str, Any]] = [
                {
                    "file": file_path,
                    "quarterly_commit_counts": counts,
                    "acceleration_trend": round(trend, 4),
                    "first_quarter": counts[0],
                    "latest_quarter": counts[-1],
                    "growth_factor": round(counts[-1] / max(counts[0], 1), 1),
                }
            ]

            confidence = min(1.0, 0.3 + total * 0.02 + abs(trend) * 2.0)
            confidence = round(confidence, 3)

            commit_dates = self.file_commit_dates.get(file_path, [])
            first_seen = min(commit_dates) if commit_dates else self.commits[0].date
            last_seen = max(commit_dates) if commit_dates else self.commits[-1].date

            pattern = Pattern(
                pattern_type=PatternType.ANTI_PATTERN,
                title=f"Rapidly growing file: {file_path}",
                description=(
                    f"'{file_path}' commit frequency is accelerating: "
                    f"{counts[0]} commits in Q1 → {counts[-1]} in the latest "
                    f"quarter ({round(counts[-1] / max(counts[0], 1), 1)}x "
                    f"growth).  Accelerating commit frequency often indicates "
                    f"a file taking on too many responsibilities."
                ),
                first_seen=first_seen,
                last_seen=last_seen,
                occurrence_count=total,
                severity=severity,
                evidence=evidence,
                affected_files=[file_path],
                recommendation=(
                    f"'{file_path}' is being modified with increasing "
                    f"frequency ({counts[-1]} commits in the latest quarter, "
                    f"up from {counts[0]}).  This often signals a file that "
                    f"has become a catch-all.  Consider splitting it into "
                    f"smaller, focused modules with clear single "
                    f"responsibilities."
                ),
                confidence=confidence,
            )
            patterns.append(pattern)

        return patterns

    def _detect_increasing_coupling(self) -> List[Pattern]:
        """Find sets of files that increasingly change together.

        Two files are "coupled" when they appear in the same commit.  We
        measure whether this coupling is increasing over time.
        """
        patterns: List[Pattern] = []

        quarters = _split_into_quarters(self.commits)
        if len(quarters) < 3:
            return patterns

        # For each pair of frequently-changed files, count co-changes per quarter
        # Only consider files that appear in the most_changed_files list
        significant_files = {fp for fp, _ in self.most_changed_files[:15]}
        if len(significant_files) < 2:
            return patterns

        file_list = sorted(significant_files)

        # Track co-change counts per quarter for each pair
        pair_quarter_counts: Dict[Tuple[str, str], List[int]] = defaultdict(
            lambda: [0] * len(quarters)
        )

        for qi, quarter in enumerate(quarters):
            for c in quarter:
                changed_significant = [
                    fp for fp in c.files_changed if fp in significant_files
                ]
                if len(changed_significant) < 2:
                    continue
                # Count co-changes for all pairs
                for i in range(len(changed_significant)):
                    for j in range(i + 1, len(changed_significant)):
                        pair = tuple(sorted([changed_significant[i], changed_significant[j]]))
                        pair_quarter_counts[pair][qi] += 1

        # Find pairs with increasing co-change frequency
        for pair, counts in pair_quarter_counts.items():
            total = sum(counts)
            if total < 3:
                continue

            trend = _compute_trend([float(c) for c in counts])
            if trend is None or trend <= 0:
                continue

            # Require at least 3 co-changes and increasing trend
            if counts[-1] < 2 or counts[-1] <= counts[0]:
                continue

            severity = Severity.MEDIUM
            # If both files are critical-path, upgrade severity
            if all(_CRITICAL_PATH_RE.search(fp) for fp in pair):
                severity = Severity.HIGH

            evidence: List[Dict[str, Any]] = [
                {
                    "file_pair": list(pair),
                    "quarterly_cochange_counts": counts,
                    "total_cochanges": total,
                    "trend": round(trend, 4),
                }
            ]

            confidence = min(1.0, 0.3 + total * 0.05 + abs(trend) * 2.0)
            confidence = round(confidence, 3)

            commit_dates_a = self.file_commit_dates.get(pair[0], [])
            commit_dates_b = self.file_commit_dates.get(pair[1], [])
            all_dates = commit_dates_a + commit_dates_b
            first_seen = min(all_dates) if all_dates else self.commits[0].date
            last_seen = max(all_dates) if all_dates else self.commits[-1].date

            pattern = Pattern(
                pattern_type=PatternType.ANTI_PATTERN,
                title=f"Increasing coupling: {pair[0]} ↔ {pair[1]}",
                description=(
                    f"'{pair[0]}' and '{pair[1]}' are increasingly changed "
                    f"together ({counts[0]} co-changes in Q1 → "
                    f"{counts[-1]} in the latest quarter).  Growing "
                    f"co-change frequency indicates tight coupling that may "
                    f"make independent changes difficult."
                ),
                first_seen=first_seen,
                last_seen=last_seen,
                occurrence_count=total,
                severity=severity,
                evidence=evidence,
                affected_files=list(pair),
                recommendation=(
                    f"'{pair[0]}' and '{pair[1]}' change together "
                    f"{counts[-1]} times in the latest quarter.  This coupling "
                    f"is increasing (trend: {trend:.2f}).  Consider "
                    f"extracting shared logic into a common module or "
                    f"introducing an interface to decouple them."
                ),
                confidence=confidence,
            )
            patterns.append(pattern)

        # Only report top 5 by total co-changes (to avoid noise)
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
        return patterns[:5]

    def _detect_god_objects(self) -> List[Pattern]:
        """Find files touched by an unusually high fraction of total commits.

        A god object is a file that appears in ≥ 15 % of all commits AND
        has been touched by ≥ 10 commits.
        """
        patterns: List[Pattern] = []
        total = len(self.commits)
        if total < 10:
            return patterns

        threshold_pct = 0.15
        for file_path, count in self.most_changed_files:
            if count < 10:
                continue

            pct = count / total
            if pct < threshold_pct:
                continue

            severity = _classify_file_severity(file_path)
            if pct >= 0.30:
                severity = Severity.HIGH
            if pct >= 0.50:
                severity = Severity.CRITICAL

            # Count how many authors have touched this file
            authors: set = set()
            for c in self.commits:
                if file_path in c.files_changed:
                    authors.add(c.author_name)

            evidence: List[Dict[str, Any]] = [
                {
                    "file": file_path,
                    "commit_count": count,
                    "total_commits": total,
                    "percentage": round(pct * 100, 1),
                    "distinct_authors": len(authors),
                    "author_names": sorted(authors),
                }
            ]

            confidence = min(1.0, 0.4 + pct * 1.0)
            confidence = round(confidence, 3)

            commit_dates = self.file_commit_dates.get(file_path, [])
            first_seen = min(commit_dates) if commit_dates else self.commits[0].date
            last_seen = max(commit_dates) if commit_dates else self.commits[-1].date

            pattern = Pattern(
                pattern_type=PatternType.ANTI_PATTERN,
                title=f"God object: {file_path}",
                description=(
                    f"'{file_path}' appears in {round(pct * 100, 1)} % of "
                    f"all commits ({count}/{total}).  This is an unusually "
                    f"high concentration of changes in a single file, "
                    f"suggesting it may have too many responsibilities."
                ),
                first_seen=first_seen,
                last_seen=last_seen,
                occurrence_count=count,
                severity=severity,
                evidence=evidence,
                affected_files=[file_path],
                recommendation=(
                    f"'{file_path}' is modified in {round(pct * 100, 1)} % "
                    f"of commits, making it a god object.  Break it into "
                    f"smaller, single-responsibility modules.  Start by "
                    f"identifying logical groupings of functions and "
                    f"extracting them into dedicated files."
                ),
                confidence=confidence,
            )
            patterns.append(pattern)

        return patterns

    # ------------------------------------------------------------------
    # Detector: Irregular Hours
    # ------------------------------------------------------------------

    def detect_irregular_hours(self) -> List[Pattern]:
        """Detect concerning work schedule patterns.

        Evidence tracked:
        1. **Late-night commits increasing** (after-hours ratio trending up).
        2. **Weekend work increasing** (weekend ratio trending up).
        3. **No rest days**: longest streak of consecutive days with commits.

        Returns
        -------
        list[Pattern]
            Detected irregular-hours patterns.
        """
        patterns: List[Pattern] = []

        if len(self.commits) < 10:
            logger.debug("Too few commits for irregular-hours detection")
            return patterns

        quarters = _split_into_quarters(self.commits)
        if len(quarters) < 3:
            return patterns

        evidence: List[Dict[str, Any]] = []
        negative_indicators = 0
        indicator_descriptions: List[str] = []

        # --- Indicator 1: After-hours ratio trend ---
        after_hours_ratios: List[float] = []
        for quarter in quarters:
            if not quarter:
                after_hours_ratios.append(0.0)
                continue
            ah = sum(
                1 for c in quarter
                if c.date.hour < _AFTER_HOURS_END or c.date.hour >= _AFTER_HOURS_START
            )
            after_hours_ratios.append(ah / len(quarter))

        ah_trend = _compute_trend(after_hours_ratios)
        if ah_trend is not None and ah_trend > 0:
            negative_indicators += 1
            indicator_descriptions.append("increasing after-hours commits")
            evidence.append(
                {
                    "indicator": "after_hours_trend",
                    "slope": round(ah_trend, 4),
                    "quarterly_ratios": [round(r, 4) for r in after_hours_ratios],
                }
            )

        # --- Indicator 2: Weekend ratio trend ---
        weekend_ratios: List[float] = []
        for quarter in quarters:
            if not quarter:
                weekend_ratios.append(0.0)
                continue
            wk = sum(1 for c in quarter if c.date.weekday() >= 5)
            weekend_ratios.append(wk / len(quarter))

        wk_trend = _compute_trend(weekend_ratios)
        if wk_trend is not None and wk_trend > 0:
            negative_indicators += 1
            indicator_descriptions.append("increasing weekend commits")
            evidence.append(
                {
                    "indicator": "weekend_trend",
                    "slope": round(wk_trend, 4),
                    "quarterly_ratios": [round(r, 4) for r in weekend_ratios],
                }
            )

        # --- Indicator 3: No rest days (longest streak) ---
        streak = self._compute_longest_work_streak()
        if streak["longest_streak"] >= 14:
            negative_indicators += 1
            indicator_descriptions.append(
                f"{streak['longest_streak']}-day work streak without a rest day"
            )
            evidence.append(
                {
                    "indicator": "no_rest_days",
                    "longest_streak_days": streak["longest_streak"],
                    "streak_start": streak.get("streak_start", ""),
                    "streak_end": streak.get("streak_end", ""),
                }
            )

        if negative_indicators == 0:
            return patterns

        confidence = min(1.0, 0.3 + negative_indicators * 0.2)
        confidence = round(confidence, 3)

        severity = Severity.LOW
        if negative_indicators >= 2:
            severity = Severity.MEDIUM
        if negative_indicators >= 3:
            severity = Severity.HIGH

        indicator_text = ", ".join(indicator_descriptions)

        pattern = Pattern(
            pattern_type=PatternType.IRREGULAR_HOURS,
            title="Irregular work schedule patterns",
            description=(
                f"{negative_indicators} irregular-hours indicator(s): "
                f"{indicator_text}.  These patterns may reflect project "
                f"deadline pressure or unsustainable work habits."
            ),
            first_seen=self.commits[0].date,
            last_seen=self.commits[-1].date,
            occurrence_count=negative_indicators,
            severity=severity,
            evidence=evidence,
            affected_files=[],
            recommendation=self._irregular_hours_recommendation(
                negative_indicators, indicator_descriptions, after_hours_ratios, weekend_ratios
            ),
            confidence=confidence,
        )
        patterns.append(pattern)
        return patterns

    def _compute_longest_work_streak(self) -> Dict[str, Any]:
        """Compute the longest streak of consecutive days with commits."""
        if not self.commits:
            return {"longest_streak": 0}

        # Collect unique commit dates (date only, no time)
        commit_dates: set = set()
        for c in self.commits:
            commit_dates.add(c.date.date())

        sorted_dates = sorted(commit_dates)
        if not sorted_dates:
            return {"longest_streak": 0}

        longest = 1
        current = 1
        streak_start = sorted_dates[0]
        best_start = sorted_dates[0]
        best_end = sorted_dates[0]

        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i] - sorted_dates[i - 1]).days == 1:
                current += 1
                if current > longest:
                    longest = current
                    best_start = streak_start
                    best_end = sorted_dates[i]
            elif (sorted_dates[i] - sorted_dates[i - 1]).days > 1:
                current = 1
                streak_start = sorted_dates[i]

        result: Dict[str, Any] = {"longest_streak": longest}
        if longest >= 7:
            result["streak_start"] = best_start.isoformat()
            result["streak_end"] = best_end.isoformat()
        return result

    @staticmethod
    def _irregular_hours_recommendation(
        indicator_count: int,
        indicator_descriptions: List[str],
        after_hours_ratios: List[float],
        weekend_ratios: List[float],
    ) -> str:
        """Build an irregular-hours-specific recommendation."""
        parts = [
            f"{indicator_count} irregular-hours indicator(s) detected: "
            + "; ".join(indicator_descriptions) + "."
        ]

        if "increasing after-hours commits" in indicator_descriptions:
            latest_ah = after_hours_ratios[-1] if after_hours_ratios else 0
            parts.append(
                f"After-hours commits have reached {round(latest_ah * 100, 1)}% "
                f"in the latest quarter.  Consider whether deadline pressure "
                f"is driving late-night work and address the root cause."
            )

        if "increasing weekend commits" in indicator_descriptions:
            latest_wk = weekend_ratios[-1] if weekend_ratios else 0
            parts.append(
                f"Weekend commits have reached {round(latest_wk * 100, 1)}% "
                f"in the latest quarter.  Sustainable pace is critical for "
                f"long-term productivity."
            )

        if any("work streak" in d for d in indicator_descriptions):
            parts.append(
                "Extended work streaks without rest days lead to diminishing "
                "returns.  Protect at least one full rest day per week."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Detector: High Churn
    # ------------------------------------------------------------------

    def detect_high_churn(self) -> List[Pattern]:
        """Detect unstable code areas with high churn.

        A file has high churn when:
        - Its total churn (additions + deletions) is in the top 10 % of all
          files, **or**
        - Its churn is more than 2× the average churn per file.

        A file is *unstable* when its churn is also increasing over time
        (accelerating rewrites).

        Returns
        -------
        list[Pattern]
            Detected high-churn patterns.
        """
        patterns: List[Pattern] = []

        if not self.code_churn:
            return patterns

        churn_values = [
            v["additions"] + v["deletions"] for v in self.code_churn.values()
        ]
        if not churn_values:
            return patterns

        avg_churn = sum(churn_values) / len(churn_values)
        churn_values_sorted = sorted(churn_values, reverse=True)
        top_10_threshold = churn_values_sorted[
            max(0, len(churn_values_sorted) // 10)
        ]

        # Identify high-churn files
        high_churn_files: List[Tuple[str, Dict[str, int], int]] = []
        for file_path, stats in self.code_churn.items():
            total_churn = stats["additions"] + stats["deletions"]
            if total_churn >= top_10_threshold or total_churn > 2 * avg_churn:
                if total_churn >= max(avg_churn * 2, 50):
                    high_churn_files.append((file_path, stats, total_churn))

        if not high_churn_files:
            return patterns

        # Sort by churn descending
        high_churn_files.sort(key=lambda x: x[2], reverse=True)
        # Limit to top 10
        high_churn_files = high_churn_files[:10]

        quarters = _split_into_quarters(self.commits)

        for file_path, stats, total_churn in high_churn_files:
            # Determine if churn is increasing over time
            increasing = False
            churn_trend: Optional[float] = None
            quarterly_churn: List[int] = []

            if len(quarters) >= 3:
                for quarter in quarters:
                    q_churn = 0
                    for c in quarter:
                        if file_path in c.files_changed:
                            q_churn += c.insertions + c.deletions
                    quarterly_churn.append(q_churn)

                if quarterly_churn:
                    churn_trend = _compute_trend([float(c) for c in quarterly_churn])
                    if churn_trend is not None and churn_trend > 0:
                        increasing = True

            severity = _classify_file_severity(file_path)
            if total_churn > 2000:
                severity = Severity.HIGH
            if increasing and total_churn > 1000:
                severity = Severity.HIGH
            if increasing and _CRITICAL_PATH_RE.search(file_path):
                severity = Severity.CRITICAL

            evidence: List[Dict[str, Any]] = [
                {
                    "file": file_path,
                    "additions": stats["additions"],
                    "deletions": stats["deletions"],
                    "net_change": stats["net"],
                    "total_churn": total_churn,
                    "avg_churn_per_file": round(avg_churn, 1),
                    "churn_vs_avg": round(total_churn / max(avg_churn, 1), 1),
                }
            ]

            if quarterly_churn:
                evidence.append(
                    {
                        "quarterly_churn": quarterly_churn,
                        "churn_trend": round(churn_trend, 4) if churn_trend is not None else None,
                        "increasing": increasing,
                    }
                )

            # Confidence: based on churn magnitude and trend consistency
            confidence = min(1.0, 0.3 + total_churn / 3000.0)
            if increasing:
                confidence = min(1.0, confidence + 0.2)
            confidence = round(confidence, 3)

            commit_dates = self.file_commit_dates.get(file_path, [])
            first_seen = min(commit_dates) if commit_dates else self.commits[0].date
            last_seen = max(commit_dates) if commit_dates else self.commits[-1].date

            trend_text = ""
            if increasing:
                trend_text = (
                    f" Churn is also accelerating (trend slope: "
                    f"{churn_trend:.2f}), indicating the file is becoming "
                    f"more unstable over time."
                )

            pattern = Pattern(
                pattern_type=PatternType.HIGH_CHURN,
                title=f"High churn: {file_path}",
                description=(
                    f"'{file_path}' has {total_churn} lines of total churn "
                    f"({stats['additions']} additions, {stats['deletions']} "
                    f"deletions), which is "
                    f"{round(total_churn / max(avg_churn, 1), 1)}× the "
                    f"average per-file churn ({round(avg_churn, 1)})."
                    f"{trend_text}"
                ),
                first_seen=first_seen,
                last_seen=last_seen,
                occurrence_count=len(commit_dates),
                severity=severity,
                evidence=evidence,
                affected_files=[file_path],
                recommendation=self._high_churn_recommendation(
                    file_path, total_churn, avg_churn, increasing, stats
                ),
                confidence=confidence,
            )
            patterns.append(pattern)

        logger.debug("High-churn detector found %d patterns", len(patterns))
        return patterns

    @staticmethod
    def _high_churn_recommendation(
        file_path: str,
        total_churn: int,
        avg_churn: float,
        increasing: bool,
        stats: Dict[str, int],
    ) -> str:
        """Build a data-driven recommendation for a high-churn file."""
        ratio = round(total_churn / max(avg_churn, 1), 1)
        parts = [
            f"'{file_path}' has {ratio}× the average churn ({total_churn} "
            f"lines vs. avg {round(avg_churn, 1)})."
        ]

        if increasing:
            parts.append(
                "Churn is accelerating -- the file is becoming less stable, "
                "not more.  This is a strong signal that the current design "
                "is not working."
            )

        if stats["net"] < 0:
            parts.append(
                f"The file has a net loss of {abs(stats['net'])} lines, "
                f"suggesting it is being trimmed without converging on a "
                f"stable design."
            )
        elif abs(stats["net"]) < total_churn * 0.1:
            parts.append(
                "Net growth is negligible relative to churn -- most changes "
                "are rewrites, not additions.  This is characteristic of "
                "an unsettled design."
            )

        parts.append(
            "Recommendation: freeze non-critical changes to this file and "
            "conduct a design review.  Identify the root cause of the "
            "instability (e.g., changing requirements, poor abstraction, "
            "missing tests) before making further modifications."
        )

        return " ".join(parts)
