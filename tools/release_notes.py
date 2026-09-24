#!/usr/bin/env python3
"""Prints a GitHub Release body for a tag: the tag's CHANGELOG.md section,
collapsed in a <details> block, plus a footer covering GPL-3.0 Corresponding
Source. The release job pipes this straight into `body_path` for
softprops/action-gh-release.

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


def render(version: str) -> str:
    """The Release body for `version`; raises ValueError if CHANGELOG.md has no such section."""
    body = section_for(version)
    # GitHub strips `style`, so a fixed-height scroll box isn't possible; <details> collapses instead.
    # The blank line after </summary> is what makes GitHub render the body as markdown.
    return (
        f"<details>\n<summary><b>Full changelog for v{version}</b> (click to expand)</summary>\n\n"
        f"{body.strip()}\n\n</details>\n\n{_FOOTER}\n"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: release_notes.py <tag>", file=sys.stderr)
        return 1

    try:
        print(render(sys.argv[1].removeprefix("v")), end="")
    except ValueError as exc:
        print(f"release_notes: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
