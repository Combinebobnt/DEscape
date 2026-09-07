"""Drift guard between descape.__version__ and CHANGELOG.md: the version
lives in two places now -- descape/__init__.py holds it as a string the crash
reporter can embed in a dump, and CHANGELOG.md's top heading is what a reader
sees -- and this test is what keeps them from silently disagreeing. Same
consistency-guard genre as test_config_example.py and
test_no_maintainer_refs.py.
"""

from __future__ import annotations

import re
from pathlib import Path

from descape import __version__

ROOT = Path(__file__).resolve().parent.parent
_HEADING_RE = re.compile(r"^## \[(?P<version>[^\]]+)\]")


def _newest_released_version() -> str:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match and match.group("version") != "Unreleased":
            return match.group("version")
    raise AssertionError("CHANGELOG.md has no released version heading")


def test_version_matches_newest_changelog_heading() -> None:
    newest = _newest_released_version()
    assert __version__ == newest, (
        f"descape.__version__ ({__version__!r}) doesn't match CHANGELOG.md's "
        f"newest released heading ({newest!r}) -- bump descape/__init__.py's "
        "__version__ in the same commit that turns [Unreleased] into a new heading."
    )
