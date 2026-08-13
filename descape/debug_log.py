"""In-memory debug log, viewable via Help > Debug Log. Not written to disk --
a live, in-process aid for seeing what the app just did (scenario loads,
renders, settings changes), not a persistent log file."""

from __future__ import annotations

from collections import deque
from datetime import datetime

_MAX_LINES = 1000
_lines: deque[str] = deque(maxlen=_MAX_LINES)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    _lines.append(f"[{timestamp}] {message}")


def get_log_text() -> str:
    return "\n".join(_lines) if _lines else "(empty)"


def clear() -> None:
    _lines.clear()
