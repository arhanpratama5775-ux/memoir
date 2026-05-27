"""Core git repository analyzer for the memoir project.

This module provides the :class:`GitAnalyzer` class -- the primary engine for
extracting **real** commit data and computing aggregate statistics from git
repositories.  Every number it returns comes from the actual git history via
GitPython; nothing is fabricated or estimated.

Typical usage::

    analyzer = GitAnalyzer("/path/to/repo", author_filter="alice")
    commits, stats = analyzer.analyze()
    patterns = analyzer.get_commit_message_patterns(commits)
    work = analyzer.get_work_pattern(commits)
    churn = analyzer.get_code_churn()
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import git
from git.exc import InvalidGitRepositoryError, NoSuchPathError

from memoir.models.commit_data import CommitData, FileChange, GitStats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns used during commit-message analysis
# ---------------------------------------------------------------------------

# Conventional-commit prefixes: feat!, fix, chore(scope), docs, etc.
_CONVENTIONAL_PREFIX_RE = re.compile(
    r"^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\([^)]+\))?\s*:",
    re.IGNORECASE,
)

# Messages considered "vague" -- too short or too generic to convey intent.
_VAGUE_RE = re.compile(
    r"^(wip|fix|update|changes?|misc|stuff|cleanups?|tidy|tweaks?"
    r"|adjust|minor|fixes|updates|tmp|temp|hack|x)$",
    re.IGNORECASE,
)

# Vague threshold: any subject line shorter than this is automatically vague.
_VAGUE_LENGTH_THRESHOLD = 10


class GitAnalyzer:
    """Analyzes a git repository to extract commit data and compute statistics.

    This is the core engine of the **memoir** project.  It reads real data
    from git repositories using *GitPython* and produces structured
    :class:`CommitData` objects and aggregate :class:`GitStats`.

    Parameters
    ----------
    repo_path:
        Absolute or relative path to the git repository on disk.
    author_filter:
        Optional author name or email substring to filter commits.
    since:
        Optional start date (inclusive) for the commit range.
    until:
        Optional end date (inclusive) for the commit range.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        repo_path: str,
        author_filter: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> None:
        self.repo_path = repo_path
        self.author_filter = author_filter
        self.since = since
        self.until = until

        # Lazily initialised internal state
        self._repo: Optional[git.Repo] = None
        self._tag_map: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_repo(self) -> git.Repo:
        """Open and cache the git repository.

        Returns
        -------
        git.Repo
            The opened repository object.

        Raises
        ------
        FileNotFoundError
            If *repo_path* does not exist on disk.
        ValueError
            If *repo_path* exists but is not a valid git repository.
        """
        if self._repo is not None:
            return self._repo

        try:
            self._repo = git.Repo(self.repo_path)
            logger.info("Opened repository at %s", self.repo_path)
            return self._repo
        except NoSuchPathError:
            logger.error("Repository path does not exist: %s", self.repo_path)
            raise FileNotFoundError(
                f"Repository path does not exist: {self.repo_path}"
            )
        except InvalidGitRepositoryError:
            logger.error("Not a valid git repository: %s", self.repo_path)
            raise ValueError(f"Not a valid git repository: {self.repo_path}")

    def _build_tag_map(self) -> Dict[str, List[str]]:
        """Build a mapping from commit hex SHA to associated tag names.

        Returns
        -------
        dict[str, list[str]]
            Mapping of ``commit.hexsha`` -> ``[tag_name, ...]``.
        """
        if self._tag_map is not None:
            return self._tag_map

        repo = self._open_repo()
        tag_map: Dict[str, List[str]] = defaultdict(list)

        try:
            for tag in repo.tags:
                try:
                    # Works for both annotated and lightweight tags
                    commit_hex = tag.commit.hexsha
                    tag_map[commit_hex].append(tag.name)
                except (ValueError, TypeError):
                    logger.debug(
                        "Skipping tag %s: could not resolve target commit",
                        tag.name,
                    )
        except Exception:
            logger.warning(
                "Could not read tags from repository", exc_info=True
            )

        self._tag_map = dict(tag_map)
        logger.debug(
            "Built tag map with %d tagged commits", len(self._tag_map)
        )
        return self._tag_map

    def _iter_commits_kwargs(self) -> Dict[str, Any]:
        """Build keyword arguments for ``repo.iter_commits()``.

        Returns
        -------
        dict[str, Any]
            Keyword arguments reflecting *author_filter*, *since*, *until*.
        """
        kwargs: Dict[str, Any] = {}
        if self.author_filter:
            kwargs["author"] = self.author_filter
            logger.debug("Filtering commits by author: %s", self.author_filter)
        if self.since:
            kwargs["since"] = self.since
            logger.debug(
                "Filtering commits since: %s", self.since.isoformat()
            )
        if self.until:
            kwargs["until"] = self.until
            logger.debug(
                "Filtering commits until: %s", self.until.isoformat()
            )
        return kwargs

    # ------------------------------------------------------------------
    # Public API -- main entry point
    # ------------------------------------------------------------------

    def analyze(self) -> Tuple[List[CommitData], GitStats]:
        """Main analysis method.  Extracts all commits and computes aggregate stats.

        This is the primary entry point for the analyzer.  It opens the
        repository, iterates through all matching commits, builds
        :class:`CommitData` objects with real insertion/deletion/file counts,
        and finally computes :class:`GitStats` from the resulting list.

        Returns
        -------
        tuple[list[CommitData], GitStats]
            A 2-tuple of *(commits, stats)*.

        Raises
        ------
        FileNotFoundError
            If the repository path does not exist.
        ValueError
            If the path is not a git repository or the repository is empty.
        """
        repo = self._open_repo()

        # Guard against empty repositories (no HEAD commit)
        try:
            _ = repo.head.commit  # noqa: F841 -- just checking existence
        except ValueError:
            logger.warning("Repository has no commits: %s", self.repo_path)
            raise ValueError(
                f"Repository has no commits: {self.repo_path}"
            )

        commits = self.get_commits()
        logger.info("Analyzed %d commits", len(commits))

        stats = self.compute_stats(commits)
        return commits, stats

    # ------------------------------------------------------------------
    # Public API -- commit extraction
    # ------------------------------------------------------------------

    def get_commits(self) -> List[CommitData]:
        """Parse all commits from git log using GitPython.

        Applies *author_filter*, *since*, and *until* constraints if they
        were provided at construction time.  Each commit's stats
        (insertions, deletions, files changed) are computed from the real
        git diff data.

        Returns
        -------
        list[CommitData]
            Commits in reverse-chronological order (newest first), matching
            the order returned by ``git log``.

        Raises
        ------
        ValueError
            If the repository cannot be read or the git command fails.
        """
        repo = self._open_repo()
        tag_map = self._build_tag_map()
        kwargs = self._iter_commits_kwargs()

        commits: List[CommitData] = []

        try:
            for commit in repo.iter_commits(**kwargs):
                try:
                    commit_data = self._build_commit_data(commit, tag_map)
                    commits.append(commit_data)
                except Exception:
                    logger.warning(
                        "Failed to process commit %s, skipping",
                        getattr(commit, "hexsha", "unknown")[:7],
                        exc_info=True,
                    )
                    continue
        except git.exc.GitCommandError as exc:
            logger.error("Git command failed while iterating commits: %s", exc)
            raise ValueError(f"Failed to iterate commits: {exc}") from exc

        logger.info("Parsed %d commits from repository", len(commits))
        return commits

    @staticmethod
    def _build_commit_data(
        commit: git.objects.commit.Commit,
        tag_map: Dict[str, List[str]],
    ) -> CommitData:
        """Construct a :class:`CommitData` from a GitPython commit object.

        Parameters
        ----------
        commit:
            A ``git.objects.commit.Commit`` instance.
        tag_map:
            Pre-built mapping of ``hexsha`` -> tag names.

        Returns
        -------
        CommitData
        """
        # commit.stats.total returns a dict with keys:
        #   'files', 'insertions', 'deletions', 'lines'
        # NOTE: Accessing commit.stats triggers a git diff which can fail
        # on shallow clones where the parent commit is missing.
        # Handle gracefully by falling back to zero stats.
        try:
            total_stats = commit.stats.total
            files_list = list(commit.stats.files.keys())
        except Exception:
            logger.debug(
                "Could not retrieve stats for commit %s (shallow clone?)",
                commit.hexsha[:7],
            )
            total_stats = {"insertions": 0, "deletions": 0}
            files_list = []

        return CommitData(
            hash=commit.hexsha,
            short_hash=commit.hexsha[:7],
            author_name=(
                commit.author.name if commit.author else "Unknown"
            ),
            author_email=(
                commit.author.email if commit.author else "unknown"
            ),
            date=commit.committed_datetime,
            message=commit.message.strip(),
            files_changed=files_list,
            insertions=total_stats.get("insertions", 0),
            deletions=total_stats.get("deletions", 0),
            is_merge=len(commit.parents) > 1,
            tags=tag_map.get(commit.hexsha, []),
        )

    # ------------------------------------------------------------------
    # Public API -- file-level changes
    # ------------------------------------------------------------------

    def get_file_changes(self, commit_hash: str) -> List[FileChange]:
        """Get file-level changes for a specific commit.

        Uses ``commit.stats.files`` as the primary data source, falling back
        to a raw diff parse if stats are unavailable (e.g. binary-heavy
        commits in some repo configurations).

        Parameters
        ----------
        commit_hash:
            Full or abbreviated SHA-1 hash of the target commit.

        Returns
        -------
        list[FileChange]
            One :class:`FileChange` per file touched in the commit.

        Raises
        ------
        ValueError
            If the commit hash cannot be resolved.
        """
        repo = self._open_repo()

        # Resolve the commit object
        try:
            commit = repo.commit(commit_hash)
        except git.exc.BadName:
            logger.error("Commit not found: %s", commit_hash)
            raise ValueError(f"Commit not found: {commit_hash}")
        except Exception as exc:
            logger.error("Error looking up commit %s: %s", commit_hash, exc)
            raise ValueError(
                f"Error looking up commit {commit_hash}: {exc}"
            ) from exc

        commit_date = commit.committed_datetime
        file_changes: List[FileChange] = []

        # --- Primary path: commit.stats.files ---
        try:
            for file_path, file_stat in commit.stats.files.items():
                file_changes.append(
                    FileChange(
                        file_path=file_path,
                        insertions=file_stat.get("insertions", 0),
                        deletions=file_stat.get("deletions", 0),
                        commit_hash=commit.hexsha,
                        commit_date=commit_date,
                    )
                )
            if file_changes:
                return file_changes
        except Exception:
            logger.debug(
                "commit.stats.files failed for %s, trying diff approach",
                commit_hash[:7],
                exc_info=True,
            )

        # --- Fallback: parse the diff output ---
        try:
            if commit.parents:
                diffs = commit.diff(commit.parents[0], create_patch=True)
            else:
                # Initial commit -- diff against the empty tree
                diffs = commit.diff(git.NULL_TREE, create_patch=True)

            for diff_item in diffs:
                file_path = diff_item.b_path or diff_item.a_path or ""
                if not file_path:
                    continue

                insertions = 0
                deletions = 0
                if diff_item.diff:
                    for line in diff_item.diff.decode(
                        "utf-8", errors="replace"
                    ).split("\n"):
                        if line.startswith("+") and not line.startswith("+++"):
                            insertions += 1
                        elif line.startswith("-") and not line.startswith("---"):
                            deletions += 1

                file_changes.append(
                    FileChange(
                        file_path=file_path,
                        insertions=insertions,
                        deletions=deletions,
                        commit_hash=commit.hexsha,
                        commit_date=commit_date,
                    )
                )
        except Exception:
            logger.error(
                "All methods failed to get file changes for commit %s",
                commit_hash[:7],
                exc_info=True,
            )

        return file_changes

    # ------------------------------------------------------------------
    # Public API -- aggregate statistics
    # ------------------------------------------------------------------

    def compute_stats(self, commits: List[CommitData]) -> GitStats:
        """Compute aggregated statistics from a list of commits.

        Parameters
        ----------
        commits:
            Non-empty list of :class:`CommitData` objects.

        Returns
        -------
        GitStats
            Fully populated aggregate statistics.

        Raises
        ------
        ValueError
            If *commits* is empty.
        """
        if not commits:
            raise ValueError("Cannot compute stats from an empty commit list")

        # --- Basic aggregates ---
        total_commits = len(commits)
        dates = [c.date for c in commits]
        first_commit_date = min(dates)
        last_commit_date = max(dates)

        author_names = sorted({c.author_name for c in commits})
        unique_authors = len(author_names)

        total_files_changed = sum(len(c.files_changed) for c in commits)
        total_insertions = sum(c.insertions for c in commits)
        total_deletions = sum(c.deletions for c in commits)

        message_lengths = [len(c.message) for c in commits]
        avg_message_length = (
            sum(message_lengths) / len(message_lengths) if message_lengths else 0.0
        )

        # --- Derived distributions ---
        commit_frequency_by_day = self.get_commit_frequency(commits)
        hourly_distribution = self.get_hourly_distribution(commits)
        weekly_commit_counts = self.get_weekly_commits(commits)
        most_changed_files = self.get_most_changed_files(commits)

        return GitStats(
            total_commits=total_commits,
            first_commit_date=first_commit_date,
            last_commit_date=last_commit_date,
            unique_authors=unique_authors,
            author_names=author_names,
            total_files_changed=total_files_changed,
            total_insertions=total_insertions,
            total_deletions=total_deletions,
            avg_message_length=round(avg_message_length, 2),
            commit_frequency_by_day=commit_frequency_by_day,
            hourly_distribution=hourly_distribution,
            weekly_commit_counts=weekly_commit_counts,
            most_changed_files=most_changed_files,
        )

    # ------------------------------------------------------------------
    # Public API -- distribution helpers
    # ------------------------------------------------------------------

    def get_commit_frequency(
        self, commits: List[CommitData]
    ) -> Dict[str, int]:
        """Compute commit count grouped by day of the week.

        Parameters
        ----------
        commits:
            List of commits to analyse.

        Returns
        -------
        dict[str, int]
            Mapping of full weekday names (``"Monday"`` -- ``"Sunday"``)
            to commit counts.  All seven days are always present, even if
            the count is zero.
        """
        day_names = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday",
        ]
        counter: Dict[str, int] = {day: 0 for day in day_names}

        for commit in commits:
            day_name = commit.date.strftime("%A")
            counter[day_name] = counter.get(day_name, 0) + 1

        return counter

    def get_hourly_distribution(
        self, commits: List[CommitData]
    ) -> Dict[int, int]:
        """Compute commit count grouped by hour of the day.

        Parameters
        ----------
        commits:
            List of commits to analyse.

        Returns
        -------
        dict[int, int]
            Mapping of hours (0--23) to commit counts.  All 24 hours are
            always present, even if the count is zero.
        """
        distribution: Dict[int, int] = {h: 0 for h in range(24)}

        for commit in commits:
            hour = commit.date.hour
            distribution[hour] = distribution.get(hour, 0) + 1

        return distribution

    def get_weekly_commits(
        self, commits: List[CommitData]
    ) -> List[Tuple[str, int]]:
        """Compute weekly commit counts.

        Each week is labelled using the ISO week format ``"YYYY-WXX"``
        (e.g. ``"2025-W09"``).

        Parameters
        ----------
        commits:
            List of commits to analyse.

        Returns
        -------
        list[tuple[str, int]]
            Chronologically sorted ``(week_label, count)`` pairs.
        """
        week_counter: Counter = Counter()

        for commit in commits:
            iso_year, iso_week, _ = commit.date.isocalendar()
            week_label = f"{iso_year}-W{iso_week:02d}"
            week_counter[week_label] += 1

        return sorted(week_counter.items())

    def get_most_changed_files(
        self, commits: List[CommitData], top_n: int = 20
    ) -> List[Tuple[str, int]]:
        """Find files with the most changes across all commits.

        A file's "change count" is the number of commits that touched it.

        Parameters
        ----------
        commits:
            List of commits to analyse.
        top_n:
            Maximum number of files to return (default 20).

        Returns
        -------
        list[tuple[str, int]]
            ``(file_path, change_count)`` pairs sorted by count descending.
        """
        file_counter: Counter = Counter()

        for commit in commits:
            for file_path in commit.files_changed:
                file_counter[file_path] += 1

        return file_counter.most_common(top_n)

    # ------------------------------------------------------------------
    # Public API -- code churn
    # ------------------------------------------------------------------

    def get_code_churn(self) -> Dict[str, Dict[str, int]]:
        """Compute per-file code churn across the entire repository.

        Code churn is the cumulative additions, deletions, and net change
        (additions minus deletions) for each file over the analysed commit
        range.

        This method iterates the repository independently (it does **not**
        rely on a pre-built commit list), so it respects the *author_filter*
        and date range supplied at construction time.

        Returns
        -------
        dict[str, dict[str, int]]
            ``{file_path: {"additions": N, "deletions": N, "net": N}}``

        Raises
        ------
        ValueError
            If the git command fails while iterating commits.
        """
        repo = self._open_repo()
        kwargs = self._iter_commits_kwargs()

        churn: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"additions": 0, "deletions": 0, "net": 0}
        )

        try:
            for commit in repo.iter_commits(**kwargs):
                try:
                    for file_path, file_stat in commit.stats.files.items():
                        ins = file_stat.get("insertions", 0)
                        dels = file_stat.get("deletions", 0)
                        churn[file_path]["additions"] += ins
                        churn[file_path]["deletions"] += dels
                        churn[file_path]["net"] += ins - dels
                except Exception:
                    logger.debug(
                        "Skipping churn calculation for commit %s",
                        commit.hexsha[:7],
                        exc_info=True,
                    )
                    continue
        except git.exc.GitCommandError as exc:
            logger.error("Git command failed during churn analysis: %s", exc)
            raise ValueError(
                f"Failed to analyze code churn: {exc}"
            ) from exc

        return dict(churn)

    # ------------------------------------------------------------------
    # Public API -- commit message patterns
    # ------------------------------------------------------------------

    def get_commit_message_patterns(
        self, commits: List[CommitData]
    ) -> Dict[str, Any]:
        """Analyse commit messages for patterns and quality indicators.

        Detects conventional-commit prefixes (``feat:``, ``fix:``, etc.),
        measures average length, and identifies *vague* messages that lack
        meaningful description.

        A message is considered **vague** when its subject line is shorter
        than 10 characters **or** matches common low-effort patterns such as
        ``"wip"``, ``"fix"``, ``"update"``, etc.

        Parameters
        ----------
        commits:
            List of commits to analyse.

        Returns
        -------
        dict[str, Any]
            Keys:

            - ``avg_length`` (*float*) -- average message length in chars.
            - ``conventional_prefix_counts`` (*dict[str, int]*) -- prefix
              -> occurrence count.
            - ``conventional_ratio`` (*float*) -- fraction of commits using
              conventional prefixes.
            - ``vague_count`` (*int*) -- number of vague messages.
            - ``vague_ratio`` (*float*) -- fraction of commits with vague
              messages.
            - ``vague_examples`` (*list[str]*) -- up to 10 subject lines
              flagged as vague.
        """
        if not commits:
            return {
                "avg_length": 0.0,
                "conventional_prefix_counts": {},
                "conventional_ratio": 0.0,
                "vague_count": 0,
                "vague_ratio": 0.0,
                "vague_examples": [],
            }

        total_length = 0
        prefix_counter: Counter = Counter()
        conventional_count = 0
        vague_count = 0
        vague_examples: List[str] = []

        for commit in commits:
            msg = commit.message.strip()
            # Analyse only the subject line (first line)
            subject = msg.split("\n", 1)[0].strip()
            total_length += len(msg)

            # Conventional-commit prefix detection
            match = _CONVENTIONAL_PREFIX_RE.match(subject)
            if match:
                prefix_counter[match.group(1).lower()] += 1
                conventional_count += 1

            # Vague-message detection
            normalised = subject.lower().rstrip(".!? ")
            if len(subject) < _VAGUE_LENGTH_THRESHOLD or _VAGUE_RE.match(
                normalised
            ):
                vague_count += 1
                if len(vague_examples) < 10:
                    vague_examples.append(subject)

        n = len(commits)
        return {
            "avg_length": round(total_length / n, 2),
            "conventional_prefix_counts": dict(prefix_counter.most_common()),
            "conventional_ratio": round(conventional_count / n, 4),
            "vague_count": vague_count,
            "vague_ratio": round(vague_count / n, 4),
            "vague_examples": vague_examples,
        }

    # ------------------------------------------------------------------
    # Public API -- work pattern analysis
    # ------------------------------------------------------------------

    def get_work_pattern(
        self, commits: List[CommitData]
    ) -> Dict[str, Any]:
        """Analyse work patterns from commit timestamps.

        Definitions
        ~~~~~~~~~~~
        - **After-hours**: commits authored before 07:00 or at/after 20:00
          (in the commit's own timezone).
        - **Weekend**: commits authored on Saturday or Sunday.
        - **Burst period**: any 2-hour window containing 5 or more commits.
        - **Average time between commits**: mean interval between consecutive
          commits in hours.

        Parameters
        ----------
        commits:
            List of commits to analyse.

        Returns
        -------
        dict[str, Any]
            Keys:

            - ``after_hours_ratio`` (*float*) -- fraction of after-hours
              commits.
            - ``weekend_ratio`` (*float*) -- fraction of weekend commits.
            - ``burst_periods`` (*list[dict]*) -- each dict has ``start``,
              ``end`` (ISO-format strings), and ``count``.
            - ``avg_time_between_commits_hours`` (*float*) -- mean hours
              between consecutive commits.
        """
        if not commits:
            return {
                "after_hours_ratio": 0.0,
                "weekend_ratio": 0.0,
                "burst_periods": [],
                "avg_time_between_commits_hours": 0.0,
            }

        after_hours_count = 0
        weekend_count = 0

        for commit in commits:
            hour = commit.date.hour
            # After-hours: before 7 AM (hours 0-6) or at/after 8 PM (hours 20-23)
            if hour < 7 or hour >= 20:
                after_hours_count += 1

            # Weekend: Saturday (5) or Sunday (6) per weekday()
            if commit.date.weekday() >= 5:
                weekend_count += 1

        n = len(commits)
        after_hours_ratio = round(after_hours_count / n, 4)
        weekend_ratio = round(weekend_count / n, 4)

        # Burst detection
        burst_periods = self._detect_burst_periods(commits)

        # Average time between consecutive commits
        avg_time_hours = self._avg_time_between_commits(commits)

        return {
            "after_hours_ratio": after_hours_ratio,
            "weekend_ratio": weekend_ratio,
            "burst_periods": burst_periods,
            "avg_time_between_commits_hours": round(avg_time_hours, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers -- burst detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_burst_periods(
        commits: List[CommitData],
    ) -> List[Dict[str, Any]]:
        """Detect burst periods: 5+ commits within any 2-hour window.

        Uses a sliding-window approach over sorted commit timestamps.
        Overlapping bursts are merged so that no commit is double-counted.

        Parameters
        ----------
        commits:
            List of commits (will be sorted by date internally).

        Returns
        -------
        list[dict[str, Any]]
            Each dict contains ``start`` (ISO datetime), ``end`` (ISO
            datetime), and ``count`` (number of commits in the burst).
        """
        if len(commits) < 5:
            return []

        # Sort oldest-first for chronological scanning
        sorted_commits = sorted(commits, key=lambda c: c.date)
        timestamps = [c.date for c in sorted_commits]

        # Phase 1: collect raw burst windows using two-pointer technique
        raw_bursts: List[Tuple[datetime, datetime, int]] = []
        left = 0

        for right in range(len(timestamps)):
            # Advance left pointer until the window fits within 2 hours
            while timestamps[right] - timestamps[left] > timedelta(hours=2):
                left += 1

            window_size = right - left + 1
            if window_size >= 5:
                raw_bursts.append(
                    (timestamps[left], timestamps[right], window_size)
                )

        if not raw_bursts:
            return []

        # Phase 2: merge overlapping bursts
        merged: List[Tuple[datetime, datetime, int]] = [raw_bursts[0]]
        for start, end, count in raw_bursts[1:]:
            prev_start, prev_end, prev_count = merged[-1]
            if start <= prev_end:
                # Overlapping -- extend the previous burst and keep the
                # larger commit count.
                merged[-1] = (
                    prev_start,
                    max(prev_end, end),
                    max(prev_count, count),
                )
            else:
                merged.append((start, end, count))

        return [
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "count": count,
            }
            for start, end, count in merged
        ]

    # ------------------------------------------------------------------
    # Private helpers -- average time between commits
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_time_between_commits(commits: List[CommitData]) -> float:
        """Compute average hours between consecutive commits.

        Parameters
        ----------
        commits:
            List of commits to analyse.

        Returns
        -------
        float
            Average number of hours between consecutive commits.
            Returns ``0.0`` when fewer than 2 commits are provided.
        """
        if len(commits) < 2:
            return 0.0

        sorted_commits = sorted(commits, key=lambda c: c.date)
        total_seconds = 0.0

        for i in range(1, len(sorted_commits)):
            delta = sorted_commits[i].date - sorted_commits[i - 1].date
            total_seconds += delta.total_seconds()

        return total_seconds / (len(sorted_commits) - 1) / 3600.0
