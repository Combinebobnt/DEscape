#!/usr/bin/env python3
"""Prints a GitHub Release body for a tag: the tag's CHANGELOG.md section
plus a footer covering GPL-3.0 Corresponding Source. The release job pipes
this straight into `body_path` for softprops/action-gh-release.

    python3 tools/release_notes.py v0.4
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape.changelog import section_for

_FOOTER = """\
---

**Corresponding Source (GPL-3.0):** this release's source is the auto-generated \
"Source code" archive GitHub attaches to this same tag below. Bundled \
dependency versions (PyQt5, AoE2ScenarioParser, and the rest) are pinned in \
`requirements.txt` and `packaging/requirements-build.txt` at that tag, and are \
each installable from PyPI at those exact versions.\
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: release_notes.py <tag>", file=sys.stderr)
        return 1

    version = sys.argv[1].removeprefix("v")
    try:
        body = section_for(version)
    except ValueError as exc:
        print(f"release_notes: {exc}", file=sys.stderr)
        return 1

    print(body)
    print()
    print(_FOOTER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
