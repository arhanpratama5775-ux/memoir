"""JSON exporter for the memoir project.

Serialises a ``Memoir`` document to a structured JSON file enriched with
export metadata such as the export timestamp, memoir library version, and
format identifier.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import memoir as _memoir_module

from memoir.models.chapter import Memoir

logger = logging.getLogger(__name__)

# The format version is incremented when the output schema changes in a
# way that consumers should be aware of (e.g. renamed keys, structural
# changes).
_FORMAT_VERSION = "1.0.0"


class JsonExporter:
    """Export a ``Memoir`` to a structured JSON file.

    The output contains the full ``memoir.to_dict()`` payload wrapped in
    an envelope that carries export metadata:

    .. code-block:: json

        {
            "metadata": {
                "format_version": "1.0.0",
                "memoir_version": "0.1.0",
                "exported_at": "2025-03-04T12:00:00+00:00",
                "format": "memoir-json"
            },
            "data": { ... }
        }

    Example::

        exporter = JsonExporter()
        path = exporter.export(memoir, "output/memoir.json")
    """

    def export(self, memoir: Memoir, output_path: str) -> str:
        """Export memoir as a structured JSON file.

        Args:
            memoir: The memoir document to export.
            output_path: Destination file path.

        Returns:
            The absolute path of the written file.

        Raises:
            OSError: If the file cannot be written.
            TypeError: If the memoir data contains non-serialisable values.
        """
        logger.info("Exporting memoir to JSON: %s", output_path)

        payload = self._build_payload(memoir)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.error("Failed to serialise memoir to JSON: %s", exc)
            raise

        abs_path = os.path.abspath(output_path)
        logger.info("JSON export complete: %s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payload(memoir: Memoir) -> Dict[str, Any]:
        """Build the full JSON payload with metadata envelope.

        Args:
            memoir: The memoir document.

        Returns:
            A dictionary ready for JSON serialisation.
        """
        metadata: Dict[str, Any] = {
            "format_version": _FORMAT_VERSION,
            "memoir_version": _memoir_module.__version__,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "format": "memoir-json",
        }

        data = memoir.to_dict()

        return {
            "metadata": metadata,
            "data": data,
        }
