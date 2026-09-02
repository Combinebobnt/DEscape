"""Guards the one-way-reference rule (maintainer/README.md,
maintainer/AGENTS.md): no file this repo ships publicly may cite a
maintainer/-relative path, since a clone of this repo never has that
directory on disk. A violation isn't a broken link -- it's an unexplained
claim, since the citation is usually standing in for a fact that belongs
inline.

Deliberate exceptions, not violations, exempted by path below:
  - THIS FILE, which has to name the target string to search for it;
  - .gitignore's `/maintainer/` entry, which lists the directory rather
    than citing a fact inside it;
  - AGENTS.md's own description of the maintainer/ convention, which is
    how an agent discovers the private repo exists at all, not a citation
    of a fact only findable there.

git-tracked files only, via `git ls-files` -- that is the actual
definition of "ships publicly" here, and it means a build artifact or a
local scratch file can never trip this.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_EXEMPT = {
    "tests/test_no_maintainer_refs.py",
    ".gitignore",
    "AGENTS.md",
}


def test_no_tracked_file_cites_a_maintainer_path():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    hits = []
    for rel in tracked:
        if rel in _EXEMPT:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "maintainer/" in text:
            hits.append(rel)
    assert not hits, (
        "these tracked files cite a maintainer/-relative path a clone of "
        f"this repo cannot follow -- restate the fact inline instead: {hits}"
    )
