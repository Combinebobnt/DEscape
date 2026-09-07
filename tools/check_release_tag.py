#!/usr/bin/env python3
"""Fails a release build fast if a pushed tag doesn't match the version the
release would actually ship. Runs as the first step of the release path, so
a mistyped or premature tag stops before any build minutes are spent.

    python3 tools/check_release_tag.py v0.4
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import __version__, changelog


def check(tag: str) -> str | None:
    """Returns an error message, or None if the tag is consistent."""
    version = tag.removeprefix("v")

    if version != __version__:
        return f"tag {tag!r} (version {version!r}) does not match descape.__version__ {__version__!r}"

    changelog_version = changelog.newest_released_version()
    if version != changelog_version:
        return f"tag {tag!r} (version {version!r}) does not match CHANGELOG.md's newest heading {changelog_version!r}"

    try:
        body = changelog.section_for(version)
    except ValueError:
        return f"CHANGELOG.md has no section for version {version!r}"
    if not body.strip():
        return f"CHANGELOG.md has no section body for version {version!r}"

    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_release_tag.py <tag>", file=sys.stderr)
        return 1

    error = check(sys.argv[1])
    if error is not None:
        print(f"check_release_tag: {error}", file=sys.stderr)
        return 1

    print(f"check_release_tag: {sys.argv[1]} is consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
