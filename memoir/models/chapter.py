"""Narrative chapter models for the memoir project.

Defines the data structures that compose a developer autobiography:
chapters of various types, and the top-level ``Memoir`` document that
collects them together with repository statistics and forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from memoir.models.commit_data import CommitData, GitStats
    from memoir.models.forecast import Forecast
    from memoir.models.pattern import Pattern


class ChapterType(Enum):
    """Classification of narrative chapter types.

    Each value corresponds to a distinct section of the generated
    developer autobiography.
    """

    PROLOGUE = "prologue"
    PATTERN = "pattern"
    MILESTONE = "milestone"
    CRISIS = "crisis"
    CURRENT_STATE = "current_state"
    FORECAST = "forecast"


@dataclass
class Chapter:
    """A single chapter in the developer autobiography.

    Chapters are the primary narrative building blocks.  Each chapter
    covers a time period, references the patterns and key commits that
    shaped it, and provides aggregated statistics.

    Attributes:
        id: Unique identifier for the chapter.
        chapter_type: The type classification of this chapter.
        title: Human-readable chapter title.
        subtitle: A descriptive subtitle.
        period_start: Start of the time period this chapter covers (None for prologue/forecast).
        period_end: End of the time period this chapter covers (None for prologue/forecast).
        narrative: The generated prose content of the chapter.
        patterns: Patterns detected within this chapter's time period.
        key_commits: Significant commits highlighted in the narrative.
        stats: Arbitrary statistics relevant to this chapter.
        order: Position of this chapter in the memoir (0-indexed).
    """

    id: str
    chapter_type: ChapterType
    title: str
    subtitle: str
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    narrative: str
    patterns: List[Any]  # List[Pattern] at runtime
    key_commits: List[Any]  # List[CommitData] at runtime
    stats: Dict[str, Any]
    order: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the chapter to a JSON-compatible dictionary.

        Datetime fields are converted to ISO 8601 format strings (or None).
        Nested model lists are recursively serialized.

        Returns:
            A dictionary representation of this chapter.
        """
        return {
            "id": self.id,
            "chapter_type": self.chapter_type.value,
            "title": self.title,
            "subtitle": self.subtitle,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "narrative": self.narrative,
            "patterns": [p.to_dict() for p in self.patterns],
            "key_commits": [c.to_dict() for c in self.key_commits],
            "stats": self.stats,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Chapter:
        """Deserialize a chapter from a dictionary.

        Nested model lists (patterns, key_commits) are recursively
        reconstructed using their respective ``from_dict`` methods.

        Args:
            data: A dictionary, typically produced by ``to_dict``.

        Returns:
            A new ``Chapter`` instance.
        """
        # Lazy imports to avoid circular dependencies at import time
        from memoir.models.commit_data import CommitData
        from memoir.models.pattern import Pattern

        period_start = (
            datetime.fromisoformat(data["period_start"])
            if data.get("period_start")
            else None
        )
        period_end = (
            datetime.fromisoformat(data["period_end"])
            if data.get("period_end")
            else None
        )

        patterns = [Pattern.from_dict(p) for p in data.get("patterns", [])]
        key_commits = [CommitData.from_dict(c) for c in data.get("key_commits", [])]

        return cls(
            id=data["id"],
            chapter_type=ChapterType(data["chapter_type"]),
            title=data["title"],
            subtitle=data["subtitle"],
            period_start=period_start,
            period_end=period_end,
            narrative=data["narrative"],
            patterns=patterns,
            key_commits=key_commits,
            stats=dict(data.get("stats", {})),
            order=data["order"],
        )


@dataclass
class Memoir:
    """The top-level developer autobiography document.

    A ``Memoir`` collects repository metadata, aggregate statistics,
    narrative chapters, risk forecasts, and an overall health score
    into a single serializable document.

    Attributes:
        repo_name: Name of the repository.
        repo_path: Absolute or relative path to the repository on disk.
        generated_at: Timestamp when this memoir was generated.
        git_stats: Aggregate git statistics for the repository.
        chapters: Ordered list of narrative chapters.
        forecasts: Risk forecasts derived from the analysis.
        overall_health_score: Composite health score from 0 (critical) to 100 (excellent).
    """

    repo_name: str
    repo_path: str
    generated_at: datetime
    git_stats: Any  # GitStats at runtime
    chapters: List[Chapter]
    forecasts: List[Any]  # List[Forecast] at runtime
    overall_health_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memoir to a JSON-compatible dictionary.

        Datetime fields are converted to ISO 8601 format strings.
        Nested models are recursively serialized.

        Returns:
            A dictionary representation of this memoir.
        """
        return {
            "repo_name": self.repo_name,
            "repo_path": self.repo_path,
            "generated_at": self.generated_at.isoformat(),
            "git_stats": self.git_stats.to_dict(),
            "chapters": [ch.to_dict() for ch in self.chapters],
            "forecasts": [f.to_dict() for f in self.forecasts],
            "overall_health_score": self.overall_health_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Memoir:
        """Deserialize a memoir from a dictionary.

        Nested models (git_stats, chapters, forecasts) are recursively
        reconstructed using their respective ``from_dict`` methods.

        Args:
            data: A dictionary, typically produced by ``to_dict``.

        Returns:
            A new ``Memoir`` instance.
        """
        from memoir.models.commit_data import GitStats
        from memoir.models.forecast import Forecast

        return cls(
            repo_name=data["repo_name"],
            repo_path=data["repo_path"],
            generated_at=datetime.fromisoformat(data["generated_at"]),
            git_stats=GitStats.from_dict(data["git_stats"]),
            chapters=[Chapter.from_dict(ch) for ch in data.get("chapters", [])],
            forecasts=[Forecast.from_dict(f) for f in data.get("forecasts", [])],
            overall_health_score=float(data["overall_health_score"]),
        )
