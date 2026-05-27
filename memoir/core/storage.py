"""Local JSON file storage for memoir data.

Manages persistent storage of analysis results, detected patterns,
crisis forecasts, generated memoirs, and project configuration.

Storage layout::

    ~/.memoir/              <- global config
      config.yaml
    <repo>/.memoir/         <- per-repo data
      analysis.json
      patterns.json
      forecasts.json
      memoir.json
      config.yaml
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from memoir.models.chapter import Memoir
from memoir.models.commit_data import CommitData, GitStats
from memoir.models.forecast import Forecast
from memoir.models.pattern import Pattern

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLOBAL_DIR_NAME = ".memoir"
REPO_DIR_NAME = ".memoir"

ANALYSIS_FILE = "analysis.json"
PATTERNS_FILE = "patterns.json"
FORECASTS_FILE = "forecasts.json"
MEMOIR_FILE = "memoir.json"
CONFIG_FILE = "config.yaml"


class Storage:
    """Local JSON file storage for memoir data.

    Each repository gets its own ``.memoir/`` directory at the repo root
    for analysis data, patterns, forecasts, and generated memoirs.  Global
    configuration lives in ``~/.memoir/config.yaml``.

    Args:
        repo_path: Absolute or relative path to the git repository root.
    """

    def __init__(self, repo_path: str) -> None:
        """Initialize storage for a repo.

        Creates ``~/.memoir/`` (global) and ``<repo>/.memoir/`` (per-repo)
        directories if they do not already exist.

        Args:
            repo_path: Path to the git repository root.
        """
        self.repo_path = Path(repo_path).resolve()
        self.global_dir = Path.home() / GLOBAL_DIR_NAME
        self.repo_dir = self.repo_path / REPO_DIR_NAME

        # Ensure directories exist
        try:
            self.global_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured global memoir directory exists: %s", self.global_dir)
        except OSError as exc:
            logger.warning("Could not create global directory %s: %s", self.global_dir, exc)

        try:
            self.repo_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured repo memoir directory exists: %s", self.repo_dir)
        except OSError as exc:
            logger.warning("Could not create repo directory %s: %s", self.repo_dir, exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_json(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Read and parse a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            Parsed dictionary, or ``None`` if the file does not exist or
            cannot be decoded.
        """
        if not filepath.exists():
            logger.debug("JSON file not found: %s", filepath)
            return None
        try:
            text = filepath.read_text(encoding="utf-8")
            data = json.loads(text)
            logger.debug("Loaded JSON from %s", filepath)
            return data
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON from %s: %s", filepath, exc)
            return None
        except OSError as exc:
            logger.error("Failed to read %s: %s", filepath, exc)
            return None

    def _write_json(self, filepath: Path, data: Any) -> None:
        """Serialize data to a JSON file atomically.

        Writes to a temporary file first, then renames to avoid
        partial-write corruption.

        Args:
            filepath: Destination path.
            data: JSON-serialisable data.
        """
        tmp_path = filepath.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            # Atomic rename (same filesystem)
            tmp_path.replace(filepath)
            logger.debug("Wrote JSON to %s", filepath)
        except (OSError, TypeError) as exc:
            logger.error("Failed to write JSON to %s: %s", filepath, exc)
            # Clean up temp file if it lingers
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def _read_yaml(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Read and parse a YAML file.

        Args:
            filepath: Path to the YAML file.

        Returns:
            Parsed dictionary, or ``None`` if the file does not exist or
            cannot be parsed.
        """
        if not filepath.exists():
            logger.debug("YAML file not found: %s", filepath)
            return None
        try:
            text = filepath.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if data is None:
                return {}
            if not isinstance(data, dict):
                logger.warning("YAML file %s did not contain a mapping", filepath)
                return {}
            return data
        except yaml.YAMLError as exc:
            logger.error("Failed to parse YAML from %s: %s", filepath, exc)
            return None
        except OSError as exc:
            logger.error("Failed to read %s: %s", filepath, exc)
            return None

    def _write_yaml(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Serialize data to a YAML file.

        Args:
            filepath: Destination path.
            data: YAML-serialisable dictionary.
        """
        try:
            filepath.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            logger.debug("Wrote YAML to %s", filepath)
        except (OSError, yaml.YAMLError) as exc:
            logger.error("Failed to write YAML to %s: %s", filepath, exc)
            raise

    def _file_mtime(self, filepath: Path) -> Optional[datetime]:
        """Return the modification time of a file, or None if missing."""
        if not filepath.exists():
            return None
        try:
            return datetime.fromtimestamp(filepath.stat().st_mtime)
        except OSError:
            return None

    # ------------------------------------------------------------------
    # Analysis data
    # ------------------------------------------------------------------

    def save_analysis(
        self,
        commits: List[CommitData],
        git_stats: GitStats,
        work_pattern: Dict[str, Any],
        message_patterns: Dict[str, Any],
        code_churn: Dict[str, Any],
        most_changed_files: List[Any],
    ) -> None:
        """Save raw analysis data.

        Args:
            commits: List of analysed commits.
            git_stats: Aggregate repository statistics.
            work_pattern: Work-time pattern data.
            message_patterns: Commit message pattern data.
            code_churn: Code churn metrics.
            most_changed_files: List of most frequently changed files.
        """
        payload: Dict[str, Any] = {
            "saved_at": datetime.now().isoformat(),
            "commits": [c.to_dict() for c in commits],
            "git_stats": git_stats.to_dict(),
            "work_pattern": work_pattern,
            "message_patterns": message_patterns,
            "code_churn": code_churn,
            "most_changed_files": most_changed_files,
        }
        filepath = self.repo_dir / ANALYSIS_FILE
        self._write_json(filepath, payload)
        logger.info("Saved analysis data (%d commits) to %s", len(commits), filepath)

    def load_analysis(self) -> Optional[Dict[str, Any]]:
        """Load saved analysis data.

        Returns:
            A dictionary with keys ``saved_at``, ``commits``, ``git_stats``,
            ``work_pattern``, ``message_patterns``, ``code_churn``,
            ``most_changed_files``, or ``None`` if no analysis file exists.
        """
        filepath = self.repo_dir / ANALYSIS_FILE
        data = self._read_json(filepath)
        if data is None:
            return None

        # Reconstruct typed objects for the main model fields
        try:
            data["commits"] = [CommitData.from_dict(c) for c in data.get("commits", [])]
            data["git_stats"] = GitStats.from_dict(data["git_stats"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Failed to deserialize analysis data: %s", exc)
            return None

        logger.debug("Loaded analysis data from %s", filepath)
        return data

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def save_patterns(self, patterns: List[Pattern]) -> None:
        """Save detected patterns.

        Args:
            patterns: List of detected ``Pattern`` instances.
        """
        payload: Dict[str, Any] = {
            "saved_at": datetime.now().isoformat(),
            "patterns": [p.to_dict() for p in patterns],
        }
        filepath = self.repo_dir / PATTERNS_FILE
        self._write_json(filepath, payload)
        logger.info("Saved %d patterns to %s", len(patterns), filepath)

    def load_patterns(self) -> Optional[List[Pattern]]:
        """Load saved patterns.

        Returns:
            A list of ``Pattern`` instances, or ``None`` if no patterns
            file exists or it cannot be parsed.
        """
        filepath = self.repo_dir / PATTERNS_FILE
        data = self._read_json(filepath)
        if data is None:
            return None

        try:
            patterns = [Pattern.from_dict(p) for p in data.get("patterns", [])]
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Failed to deserialize patterns: %s", exc)
            return None

        logger.debug("Loaded %d patterns from %s", len(patterns), filepath)
        return patterns

    # ------------------------------------------------------------------
    # Forecasts
    # ------------------------------------------------------------------

    def save_forecasts(self, forecasts: List[Forecast]) -> None:
        """Save crisis forecasts.

        Args:
            forecasts: List of ``Forecast`` instances.
        """
        payload: Dict[str, Any] = {
            "saved_at": datetime.now().isoformat(),
            "forecasts": [f.to_dict() for f in forecasts],
        }
        filepath = self.repo_dir / FORECASTS_FILE
        self._write_json(filepath, payload)
        logger.info("Saved %d forecasts to %s", len(forecasts), filepath)

    def load_forecasts(self) -> Optional[List[Forecast]]:
        """Load saved forecasts.

        Returns:
            A list of ``Forecast`` instances, or ``None`` if no forecasts
            file exists or it cannot be parsed.
        """
        filepath = self.repo_dir / FORECASTS_FILE
        data = self._read_json(filepath)
        if data is None:
            return None

        try:
            forecasts = [Forecast.from_dict(f) for f in data.get("forecasts", [])]
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Failed to deserialize forecasts: %s", exc)
            return None

        logger.debug("Loaded %d forecasts from %s", len(forecasts), filepath)
        return forecasts

    # ------------------------------------------------------------------
    # Memoir
    # ------------------------------------------------------------------

    def save_memoir(self, memoir: Memoir) -> None:
        """Save generated memoir.

        Args:
            memoir: A ``Memoir`` instance.
        """
        payload: Dict[str, Any] = {
            "saved_at": datetime.now().isoformat(),
            "memoir": memoir.to_dict(),
        }
        filepath = self.repo_dir / MEMOIR_FILE
        self._write_json(filepath, payload)
        logger.info("Saved memoir to %s", filepath)

    def load_memoir(self) -> Optional[Memoir]:
        """Load saved memoir.

        Returns:
            A ``Memoir`` instance, or ``None`` if no memoir file exists
            or it cannot be parsed.
        """
        filepath = self.repo_dir / MEMOIR_FILE
        data = self._read_json(filepath)
        if data is None:
            return None

        try:
            memoir = Memoir.from_dict(data["memoir"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Failed to deserialize memoir: %s", exc)
            return None

        logger.debug("Loaded memoir from %s", filepath)
        return memoir

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save project-specific config.

        Args:
            config: Configuration dictionary to persist.
        """
        filepath = self.repo_dir / CONFIG_FILE
        self._write_yaml(filepath, config)
        logger.info("Saved project config to %s", filepath)

    def load_config(self) -> Dict[str, Any]:
        """Load project config with defaults.

        Returns:
            A configuration dictionary.  Returns an empty dict if no
            project-level config file exists.
        """
        filepath = self.repo_dir / CONFIG_FILE
        data = self._read_yaml(filepath)
        if data is None:
            return {}
        return data

    # ------------------------------------------------------------------
    # Status & maintenance
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current storage status.

        Returns a dictionary describing what data exists, when each file
        was last updated, and file sizes.

        Returns:
            Status dictionary with keys per data type and a ``repo_dir``
            path string.
        """
        status: Dict[str, Any] = {
            "repo_dir": str(self.repo_dir),
            "repo_dir_exists": self.repo_dir.exists(),
        }

        file_keys: Dict[str, str] = {
            "analysis": ANALYSIS_FILE,
            "patterns": PATTERNS_FILE,
            "forecasts": FORECASTS_FILE,
            "memoir": MEMOIR_FILE,
            "config": CONFIG_FILE,
        }

        for key, filename in file_keys.items():
            filepath = self.repo_dir / filename
            mtime = self._file_mtime(filepath)
            entry: Dict[str, Any] = {
                "exists": filepath.exists(),
                "last_updated": mtime.isoformat() if mtime else None,
            }
            if filepath.exists():
                try:
                    entry["size_bytes"] = filepath.stat().st_size
                except OSError:
                    entry["size_bytes"] = None
            status[key] = entry

        return status

    def clear(self, what: str = "all") -> None:
        """Clear stored data.

        Args:
            what: What to clear.  One of ``'analysis'``, ``'patterns'``,
                ``'forecasts'``, ``'memoir'``, or ``'all'`` (default).

        Raises:
            ValueError: If *what* is not a recognised value.
        """
        valid = {"analysis", "patterns", "forecasts", "memoir", "all"}
        if what not in valid:
            raise ValueError(f"Invalid clear target '{what}'. Must be one of {sorted(valid)}")

        file_map: Dict[str, str] = {
            "analysis": ANALYSIS_FILE,
            "patterns": PATTERNS_FILE,
            "forecasts": FORECASTS_FILE,
            "memoir": MEMOIR_FILE,
        }

        if what == "all":
            targets = list(file_map.keys())
        else:
            targets = [what]

        for target in targets:
            filepath = self.repo_dir / file_map[target]
            if filepath.exists():
                try:
                    filepath.unlink()
                    logger.info("Cleared %s data: %s", target, filepath)
                except OSError as exc:
                    logger.error("Failed to delete %s: %s", filepath, exc)
            else:
                logger.debug("No %s file to clear at %s", target, filepath)
