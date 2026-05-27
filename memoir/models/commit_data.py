"""Git commit data models for the memoir project.

Provides dataclasses representing git commits, file changes, and
aggregate repository statistics used throughout the analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple


@dataclass
class CommitData:
    """Represents a single git commit with full metadata.

    Attributes:
        hash: Full commit SHA-1 hash.
        short_hash: Abbreviated commit hash (typically 7 chars).
        author_name: Name of the commit author.
        author_email: Email address of the commit author.
        date: Timestamp of the commit.
        message: Full commit message including subject and body.
        files_changed: List of file paths modified in this commit.
        insertions: Number of lines added.
        deletions: Number of lines removed.
        is_merge: Whether this is a merge commit.
        tags: List of tag names pointing to this commit.
    """

    hash: str
    short_hash: str
    author_name: str
    author_email: str
    date: datetime
    message: str
    files_changed: List[str]
    insertions: int = 0
    deletions: int = 0
    is_merge: bool = False
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the commit to a JSON-compatible dictionary.

        Datetime fields are converted to ISO 8601 format strings.

        Returns:
            A dictionary representation of this commit.
        """
        return {
            "hash": self.hash,
            "short_hash": self.short_hash,
            "author_name": self.author_name,
            "author_email": self.author_email,
            "date": self.date.isoformat(),
            "message": self.message,
            "files_changed": self.files_changed,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "is_merge": self.is_merge,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CommitData:
        """Deserialize a commit from a dictionary.

        Args:
            data: A dictionary, typically produced by ``to_dict``.

        Returns:
            A new ``CommitData`` instance.
        """
        return cls(
            hash=data["hash"],
            short_hash=data["short_hash"],
            author_name=data["author_name"],
            author_email=data["author_email"],
            date=datetime.fromisoformat(data["date"]),
            message=data["message"],
            files_changed=list(data.get("files_changed", [])),
            insertions=data.get("insertions", 0),
            deletions=data.get("deletions", 0),
            is_merge=data.get("is_merge", False),
            tags=list(data.get("tags", [])),
        )


@dataclass
class FileChange:
    """Represents a single file change within a commit.

    Attributes:
        file_path: Path of the changed file relative to the repo root.
        insertions: Number of lines added in this file.
        deletions: Number of lines removed from this file.
        commit_hash: SHA-1 hash of the commit that introduced this change.
        commit_date: Timestamp of the commit.
    """

    file_path: str
    insertions: int
    deletions: int
    commit_hash: str
    commit_date: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the file change to a JSON-compatible dictionary.

        Returns:
            A dictionary representation of this file change.
        """
        return {
            "file_path": self.file_path,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "commit_hash": self.commit_hash,
            "commit_date": self.commit_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileChange:
        """Deserialize a file change from a dictionary.

        Args:
            data: A dictionary, typically produced by ``to_dict``.

        Returns:
            A new ``FileChange`` instance.
        """
        return cls(
            file_path=data["file_path"],
            insertions=data["insertions"],
            deletions=data["deletions"],
            commit_hash=data["commit_hash"],
            commit_date=datetime.fromisoformat(data["commit_date"]),
        )


@dataclass
class GitStats:
    """Aggregate statistics for a git repository.

    Attributes:
        total_commits: Total number of commits in the analysed range.
        first_commit_date: Date of the earliest commit.
        last_commit_date: Date of the most recent commit.
        unique_authors: Number of distinct commit authors.
        author_names: List of all distinct author names.
        total_files_changed: Cumulative count of file-change entries.
        total_insertions: Cumulative lines added across all commits.
        total_deletions: Cumulative lines removed across all commits.
        avg_message_length: Average character length of commit messages.
        commit_frequency_by_day: Mapping of weekday name to commit count.
        hourly_distribution: Mapping of hour (0-23) to commit count.
        weekly_commit_counts: Ordered list of (week_label, count) pairs.
        most_changed_files: Ordered list of (file_path, change_count) pairs.
    """

    total_commits: int
    first_commit_date: datetime
    last_commit_date: datetime
    unique_authors: int
    author_names: List[str]
    total_files_changed: int
    total_insertions: int
    total_deletions: int
    avg_message_length: float
    commit_frequency_by_day: Dict[str, int]
    hourly_distribution: Dict[int, int]
    weekly_commit_counts: List[Tuple[str, int]]
    most_changed_files: List[Tuple[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the git statistics to a JSON-compatible dictionary.

        Datetime fields are converted to ISO 8601 format strings.
        Tuples are converted to lists for JSON compatibility.

        Returns:
            A dictionary representation of these statistics.
        """
        return {
            "total_commits": self.total_commits,
            "first_commit_date": self.first_commit_date.isoformat(),
            "last_commit_date": self.last_commit_date.isoformat(),
            "unique_authors": self.unique_authors,
            "author_names": self.author_names,
            "total_files_changed": self.total_files_changed,
            "total_insertions": self.total_insertions,
            "total_deletions": self.total_deletions,
            "avg_message_length": self.avg_message_length,
            "commit_frequency_by_day": self.commit_frequency_by_day,
            "hourly_distribution": {
                str(k): v for k, v in self.hourly_distribution.items()
            },
            "weekly_commit_counts": [list(item) for item in self.weekly_commit_counts],
            "most_changed_files": [list(item) for item in self.most_changed_files],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GitStats:
        """Deserialize git statistics from a dictionary.

        Args:
            data: A dictionary, typically produced by ``to_dict``.

        Returns:
            A new ``GitStats`` instance.
        """
        return cls(
            total_commits=data["total_commits"],
            first_commit_date=datetime.fromisoformat(data["first_commit_date"]),
            last_commit_date=datetime.fromisoformat(data["last_commit_date"]),
            unique_authors=data["unique_authors"],
            author_names=list(data.get("author_names", [])),
            total_files_changed=data["total_files_changed"],
            total_insertions=data["total_insertions"],
            total_deletions=data["total_deletions"],
            avg_message_length=data["avg_message_length"],
            commit_frequency_by_day=dict(data.get("commit_frequency_by_day", {})),
            hourly_distribution={
                int(k): v for k, v in data.get("hourly_distribution", {}).items()
            },
            weekly_commit_counts=[
                tuple(item) for item in data.get("weekly_commit_counts", [])
            ],
            most_changed_files=[
                tuple(item) for item in data.get("most_changed_files", [])
            ],
        )
