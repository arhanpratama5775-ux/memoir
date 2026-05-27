"""Pattern detection models for the memoir project.

Defines the enumerations and data structures used to represent behavioural
and code-level patterns discovered during git history analysis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class PatternType(Enum):
    """Classification of detectable development patterns.

    Each value corresponds to a category of recurring behaviour or
    code structure that the analysis engine can identify.
    """

    RECURRING_FIX = "recurring_fix"
    TECHNICAL_DEBT = "technical_debt"
    LEARNING_CURVE = "learning_curve"
    BURNOUT_INDICATOR = "burnout_indicator"
    CODE_OWNERSHIP = "code_ownership"
    ANTI_PATTERN = "anti_pattern"
    IRREGULAR_HOURS = "irregular_hours"
    HIGH_CHURN = "high_churn"


class Severity(Enum):
    """Impact severity level assigned to a detected pattern.

    Ranges from LOW (informational) to CRITICAL (immediate action required).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def _generate_pattern_id(pattern_type: PatternType, timestamp: datetime) -> str:
    """Generate a deterministic pattern identifier.

    The ID is derived from the pattern type and timestamp so that the same
    pattern detected in the same time window always produces the same ID.

    Args:
        pattern_type: The classification of the pattern.
        timestamp: The timestamp used as part of the hash input.

    Returns:
        A hex string uniquely identifying this pattern occurrence.
    """
    raw = f"{pattern_type.value}:{timestamp.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class Pattern:
    """A detected development pattern within a repository's history.

    Instances are produced by the pattern-detection analysis stage and
    consumed by the narrative-generation and forecasting stages.

    Attributes:
        id: Unique identifier, auto-generated from pattern_type and first_seen.
        pattern_type: The category of the detected pattern.
        title: Human-readable short title of the pattern.
        description: Detailed explanation of what was detected.
        first_seen: When the pattern was first observed.
        last_seen: When the pattern was most recently observed.
        occurrence_count: How many times this pattern has been observed.
        severity: Impact severity level.
        evidence: Supporting data points (commit hashes, metrics, etc.).
        affected_files: File paths that exhibit this pattern.
        recommendation: Suggested action to address or monitor the pattern.
        confidence: Detection confidence from 0.0 (uncertain) to 1.0 (certain).
    """

    pattern_type: PatternType
    title: str
    description: str
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int
    severity: Severity
    evidence: List[Dict[str, Any]]
    affected_files: List[str]
    recommendation: str
    confidence: float
    id: str = field(default="")

    def __post_init__(self) -> None:
        """Auto-generate the pattern ID if not explicitly provided."""
        if not self.id:
            self.id = _generate_pattern_id(self.pattern_type, self.first_seen)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the pattern to a JSON-compatible dictionary.

        Datetime fields are converted to ISO 8601 format strings.
        Enums are stored by their value.

        Returns:
            A dictionary representation of this pattern.
        """
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "title": self.title,
            "description": self.description,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "occurrence_count": self.occurrence_count,
            "severity": self.severity.value,
            "evidence": self.evidence,
            "affected_files": self.affected_files,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Pattern:
        """Deserialize a pattern from a dictionary.

        Args:
            data: A dictionary, typically produced by ``to_dict``.

        Returns:
            A new ``Pattern`` instance.
        """
        return cls(
            id=data.get("id", ""),
            pattern_type=PatternType(data["pattern_type"]),
            title=data["title"],
            description=data["description"],
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            occurrence_count=data["occurrence_count"],
            severity=Severity(data["severity"]),
            evidence=list(data.get("evidence", [])),
            affected_files=list(data.get("affected_files", [])),
            recommendation=data["recommendation"],
            confidence=float(data["confidence"]),
        )
