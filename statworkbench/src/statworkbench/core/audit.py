"""Audit logging for dataset and project operations.

The :class:`AuditLog` records every significant action performed on a
:class:`Dataset` or project so that analyses remain reproducible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    """An in-memory append-only audit log.

    Each entry captures an action, optional details, and a UTC timestamp.
    The log can be exported to JSON Lines format for persistent storage.

    Example::

        log = AuditLog()
        log.append("variable_rename", {"old": "x", "new": "age"})
        log.append("analysis_run", {"procedure": "t_test", "variables": ["a", "b"]})
        entries = log.to_list()
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def append(self, action: str, details: dict[str, Any] | None = None) -> None:
        """Append a new audit entry.

        Args:
            action: A short identifier for the action (e.g.
                ``variable_rename``, ``analysis_run``).
            details: Optional dictionary with structured details about
                the action.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
        }
        if details is not None:
            entry["details"] = dict(details)
        self._entries.append(entry)

    def to_list(self) -> list[dict[str, Any]]:
        """Return a shallow copy of all entries.

        Returns:
            A list of audit entry dictionaries.
        """
        return list(self._entries)

    def clear(self) -> None:
        """Remove all entries from the log."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"AuditLog(entries={len(self._entries)})"

    # ------------------------------------------------------------------
    # JSON Lines persistence
    # ------------------------------------------------------------------

    def save_jsonl(self, path: str | Path) -> None:
        """Save the log to a JSON Lines file.

        Each entry is written as a single JSON object on its own line.

        Args:
            path: File path to write to. Parent directories are created
                automatically.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for entry in self._entries:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> AuditLog:
        """Load an audit log from a JSON Lines file.

        Args:
            path: File path to read from.

        Returns:
            A new :class:`AuditLog` instance populated with the entries.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        log = cls()
        target = Path(path)
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    log._entries.append(entry)
        return log
