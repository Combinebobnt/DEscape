"""Parses CHANGELOG.md's version headings and section bodies. Shared by
tests/test_version.py (drift guard) and tools/release_notes.py (release
notes extraction), so there is one heading parser instead of two.

Reads CHANGELOG.md from disk inside each function, not at import time --
packaging/entry_frozen.py's --self-check imports every descape.* submodule,
and CHANGELOG.md isn't in packaging/descape.spec's datas list, so a
module-level read would break the frozen build.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_HEADING_RE = re.compile(r"^## \[(?P<version>[^\]]+)\]")


def _changelog_text() -> str:
    return (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def newest_released_version() -> str:
    for line in _changelog_text().splitlines():
        match = _HEADING_RE.match(line)
        if match and match.group("version") != "Unreleased":
            return match.group("version")
    raise AssertionError("CHANGELOG.md has no released version heading")


def section_for(version: str) -> str:
    lines = _changelog_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match and match.group("version") == version:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"CHANGELOG.md has no section for version {version!r}")

    end = len(lines)
    for i in range(start, len(lines)):
        if _HEADING_RE.match(lines[i]):
            end = i
            break

    return "\n".join(lines[start:end]).strip("\n")
