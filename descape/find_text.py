"""Qt-free text matching and CSV-path guarding shared by Find and Replace's
object and trigger finders (GH #144).

compile_pattern() never case-folds the text itself: "ß".casefold() is "ss",
which would shift every match span after it. Case-insensitivity rides the
regex's own IGNORECASE flag instead, so spans index the original string.
"""

from __future__ import annotations

import re
from pathlib import Path

from descape.scenario_io import FORBIDDEN_WRITE_MARKER, is_under_compatdata
from descape.scenario_write import WriteBlockedError


class FindPatternError(ValueError):
    """An invalid regex (or replacement template); the message is the dialog's inline text."""


def compile_pattern(text: str, regex: bool, case_sensitive: bool) -> re.Pattern | None:
    """The compiled find pattern, or None for empty text (no text constraint).

    Substring mode is re.escape(text). Raises FindPatternError, and nothing
    else, for an invalid regex."""
    if not text:
        return None
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return re.compile(text if regex else re.escape(text), flags)
    except re.error as e:
        raise FindPatternError(f"Invalid pattern: {e}") from e


def refuse_forbidden_path(path: Path | str) -> Path:
    """Raises WriteBlockedError for a path under a Proton compatdata/ prefix,
    as written or resolved (AGENTS.md hard rule 1); returns the Path otherwise."""
    path = Path(path)
    resolved = None
    if FORBIDDEN_WRITE_MARKER not in str(path):
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise WriteBlockedError(
                f"Refusing to write to a path that cannot be resolved ({e}), so it can't be"
                f" shown to be outside a {FORBIDDEN_WRITE_MARKER!r} folder: {path}"
            ) from e
    if resolved is None or is_under_compatdata(path, resolved):
        raise WriteBlockedError(
            f"Refusing to write to a path containing {FORBIDDEN_WRITE_MARKER!r}"
            f" (as written or through a symlink): {path}"
        )
    return path
