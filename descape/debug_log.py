"""In-memory debug log, viewable via Help > Debug Log. Not written to disk --
a live, in-process aid for seeing what the app just did (scenario loads,
renders, settings changes), not a persistent log file.

Errors are failed requested operations or unexpected exceptions; warnings are
degraded or refused operations. Everything else, including perf lines, is info.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import NamedTuple

INFO = "info"
WARNING = "warning"
ERROR = "error"
LEVELS = (INFO, WARNING, ERROR)


class Entry(NamedTuple):
    timestamp: str
    level: str
    message: str

    def text(self) -> str:
        tag = "" if self.level == INFO else f"{self.level}: "
        return f"[{self.timestamp}] {tag}{self.message}"


_MAX_LINES = 1000
_entries: deque[Entry] = deque(maxlen=_MAX_LINES)


def log(message: str, level: str = INFO) -> None:
    if level not in LEVELS:
        raise ValueError(f"Unknown debug log level: {level!r}")
    timestamp = datetime.now().strftime("%H:%M:%S")
    _entries.append(Entry(timestamp, level, message))


def warning(message: str) -> None:
    log(message, level=WARNING)


def error(message: str) -> None:
    log(message, level=ERROR)


def entries() -> list[Entry]:
    return list(_entries)


def get_log_text() -> str:
    rows = entries()
    return "\n".join(entry.text() for entry in rows) if rows else "(empty)"


def clear() -> None:
    _entries.clear()
