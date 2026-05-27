"""Configuration management for the memoir project.

Supports a two-layer configuration system:

1. **Global** — ``~/.memoir/config.yaml`` (shared across all repos)
2. **Project** — ``<repo>/.memoir/config.yaml`` (overrides global)

Project-level settings always take precedence over global ones.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLOBAL_DIR_NAME = ".memoir"
CONFIG_FILE = "config.yaml"

# Default configuration values
_DEFAULTS: Dict[str, Any] = {
    "ai_api_key": None,
    "ai_model": "gpt-4o-mini",
    "ai_base_url": None,
    "after_hours_start": 20,  # 8 PM
    "after_hours_end": 7,  # 7 AM
    "min_pattern_occurrences": 3,
    "forecast_confidence_threshold": 0.3,
    "author_filter": None,
    "export_formats": ["markdown"],
}


class MemoirConfig:
    """Configuration manager for the memoir project.

    Loads settings from both the global config (``~/.memoir/config.yaml``)
    and a project-level config (``<repo>/.memoir/config.yaml``).  Project
    settings override global ones, and both fall back to built-in defaults.

    Args:
        repo_path: Optional path to the git repository root.  When
            provided, project-level config is loaded from
            ``<repo>/.memoir/config.yaml``.
    """

    def __init__(self, repo_path: Optional[str] = None) -> None:
        """Load config from ``~/.memoir/config.yaml`` and repo ``.memoir/config.yaml``.

        Args:
            repo_path: Optional path to the git repository root.
        """
        self._global_dir = Path.home() / GLOBAL_DIR_NAME
        self._global_config_path = self._global_dir / CONFIG_FILE

        if repo_path is not None:
            self._repo_path = Path(repo_path).resolve()
            self._project_config_path = self._repo_path / ".memoir" / CONFIG_FILE
        else:
            self._repo_path = None
            self._project_config_path = None

        # Load configs
        self._global_config: Dict[str, Any] = self._load_yaml(self._global_config_path)
        self._project_config: Dict[str, Any] = (
            self._load_yaml(self._project_config_path) if self._project_config_path else {}
        )

        # Staging area for unsaved changes
        self._pending_global: Dict[str, Any] = {}
        self._pending_project: Dict[str, Any] = {}

        logger.debug(
            "MemoirConfig initialised — global: %s, project: %s",
            self._global_config_path,
            self._project_config_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(filepath: Optional[Path]) -> Dict[str, Any]:
        """Load a YAML file and return its contents as a dict.

        Returns an empty dict if the file does not exist or cannot be
        parsed.

        Args:
            filepath: Path to the YAML file, or ``None``.

        Returns:
            Parsed configuration dictionary.
        """
        if filepath is None or not filepath.exists():
            return {}
        try:
            text = filepath.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if data is None:
                return {}
            if not isinstance(data, dict):
                logger.warning("Config file %s did not contain a mapping; ignoring", filepath)
                return {}
            return data
        except yaml.YAMLError as exc:
            logger.error("Failed to parse config %s: %s", filepath, exc)
            return {}
        except OSError as exc:
            logger.error("Failed to read config %s: %s", filepath, exc)
            return {}

    @staticmethod
    def _save_yaml(filepath: Path, data: Dict[str, Any]) -> None:
        """Write a configuration dictionary to a YAML file.

        Creates parent directories as needed.

        Args:
            filepath: Destination path.
            data: Data to serialise.
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            logger.debug("Saved config to %s", filepath)
        except (OSError, yaml.YAMLError) as exc:
            logger.error("Failed to save config to %s: %s", filepath, exc)
            raise

    def _resolve(self, key: str) -> Any:
        """Resolve a config key through the layer stack.

        Priority order (highest first):
        1. Pending project changes
        2. Project config file
        3. Pending global changes
        4. Global config file
        5. Built-in defaults

        Args:
            key: Configuration key.

        Returns:
            The resolved value.
        """
        # 1. Pending project
        if key in self._pending_project:
            return self._pending_project[key]
        # 2. Project config
        if key in self._project_config:
            return self._project_config[key]
        # 3. Pending global
        if key in self._pending_global:
            return self._pending_global[key]
        # 4. Global config
        if key in self._global_config:
            return self._global_config[key]
        # 5. Defaults
        return _DEFAULTS.get(key)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def ai_api_key(self) -> Optional[str]:
        """OpenAI / LLM API key (also checks ``MEMOIR_AI_API_KEY`` env var)."""
        env_val = os.environ.get("MEMOIR_AI_API_KEY")
        if env_val:
            return env_val
        return self._resolve("ai_api_key")

    @property
    def ai_model(self) -> str:
        """LLM model identifier.  Default: ``'gpt-4o-mini'``."""
        return self._resolve("ai_model")

    @property
    def ai_base_url(self) -> Optional[str]:
        """Custom base URL for the LLM API endpoint."""
        return self._resolve("ai_base_url")

    @property
    def after_hours_start(self) -> int:
        """Hour (24-h) marking the start of after-hours work.  Default: 20."""
        return int(self._resolve("after_hours_start"))

    @property
    def after_hours_end(self) -> int:
        """Hour (24-h) marking the end of after-hours work.  Default: 7."""
        return int(self._resolve("after_hours_end"))

    @property
    def min_pattern_occurrences(self) -> int:
        """Minimum occurrences for a pattern to be reported.  Default: 3."""
        return int(self._resolve("min_pattern_occurrences"))

    @property
    def forecast_confidence_threshold(self) -> float:
        """Minimum confidence to include a forecast.  Default: 0.3."""
        return float(self._resolve("forecast_confidence_threshold"))

    @property
    def author_filter(self) -> Optional[str]:
        """Optional author name/email to filter commits by."""
        return self._resolve("author_filter")

    @property
    def export_formats(self) -> List[str]:
        """Supported export formats.  Default: ``['markdown']``."""
        val = self._resolve("export_formats")
        if isinstance(val, list):
            return list(val)
        if isinstance(val, str):
            return [val]
        return list(_DEFAULTS["export_formats"])

    # ------------------------------------------------------------------
    # Get / Set / List
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        Falls back to *default* if the key is not found in any layer
        or the built-in defaults.

        Args:
            key: Configuration key.
            default: Fallback value if the key is unknown.

        Returns:
            The resolved value.
        """
        resolved = self._resolve(key)
        if resolved is None:
            return default
        return resolved

    def set(self, key: str, value: Any, scope: str = "project") -> None:
        """Set a configuration value.

        The change is staged in memory until :meth:`save` is called.

        Args:
            key: Configuration key.
            value: Value to assign.
            scope: ``'project'`` (default) or ``'global'``.

        Raises:
            ValueError: If *scope* is not ``'project'`` or ``'global'``.
        """
        if scope == "project":
            if self._project_config_path is None:
                raise ValueError(
                    "Cannot set project-scoped config without a repo path. "
                    "Use --repo to specify a repository, or use --global to set globally."
                )
            self._pending_project[key] = value
        elif scope == "global":
            self._pending_global[key] = value
        else:
            raise ValueError(f"Invalid scope '{scope}'. Must be 'project' or 'global'.")

        logger.debug("Staged config %s=%s (scope=%s)", key, value, scope)

    def list_all(self) -> Dict[str, Any]:
        """List all configuration values with their effective resolution.

        Returns a dictionary mapping each known key to its resolved value.
        Includes all default keys plus any extra keys found in config files.

        Returns:
            Dictionary of all resolved key-value pairs.
        """
        # Start with defaults, then overlay
        all_keys = set(_DEFAULTS.keys())
        all_keys.update(self._global_config.keys())
        all_keys.update(self._project_config.keys())
        all_keys.update(self._pending_global.keys())
        all_keys.update(self._pending_project.keys())

        result: Dict[str, Any] = {}
        for key in sorted(all_keys):
            result[key] = self._resolve(key)
        return result

    def save(self) -> None:
        """Persist staged changes to disk.

        Merges pending changes into the in-memory config dicts and
        writes both global and project config files as needed.
        """
        # --- Global ---
        if self._pending_global:
            merged_global = {**self._global_config, **self._pending_global}
            self._save_yaml(self._global_config_path, merged_global)
            self._global_config = merged_global
            self._pending_global.clear()
            logger.info("Saved global config to %s", self._global_config_path)

        # --- Project ---
        if self._pending_project and self._project_config_path is not None:
            merged_project = {**self._project_config, **self._pending_project}
            self._save_yaml(self._project_config_path, merged_project)
            self._project_config = merged_project
            self._pending_project.clear()
            logger.info("Saved project config to %s", self._project_config_path)
        elif self._pending_project:
            logger.warning("Cannot save project config: no repo path configured")
