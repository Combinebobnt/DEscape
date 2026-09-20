"""Guards the one-way-reference rule (maintainer/README.md,
maintainer/AGENTS.md): no file this repo ships publicly may cite a
maintainer/-relative path, or name a maintainer-only doc/plan file by its
bare filename, since a clone of this repo never has that directory on
disk. A violation isn't a broken link -- it's an unexplained claim, since
the citation is usually standing in for a fact that belongs inline. The
bare-filename form is worse than the full-path one: it reads as pointing
at a real file (some even look like `docs/PLAN_*.md`, which is doubly
misleading -- this repo ships a real `docs/`, just not that file in it).

Deliberate exceptions, not violations, exempted by path below:
  - THIS FILE, which has to name the target strings to search for them;
  - .gitignore's `/maintainer/` entry, which lists the directory rather
    than citing a fact inside it;
  - AGENTS.md's own description of the maintainer/ convention, which is
    how an agent discovers the private repo exists at all, not a citation
    of a fact only findable there;
  - tools/gen_elevation_reference.py's DOC_PATH, which builds a real
    maintainer/docs/ output path for a maintainer-only generator -- an
    operational target, not a citation of a fact readers need.

git-tracked files only, via `git ls-files` -- that is the actual
definition of "ships publicly" here, and it means a build artifact or a
local scratch file can never trip this.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_LINE_LEADER = re.compile(r"^[ \t]*(?:#|//|\*)?[ \t]*")

_EXEMPT = {
    "tests/test_no_maintainer_refs.py",
    ".gitignore",
    "AGENTS.md",
    "tools/gen_elevation_reference.py",
}


def _tracked_files() -> list[str]:
    return subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()


def _read_text(rel: str) -> str | None:
    path = ROOT / rel
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _joined(text: str) -> str:
    """Reforms a citation a comment/docstring line-wrap split mid-token --
    e.g. `plans/descape-dedup-qapp-stepped-` / `# window.md` on the next
    line -- so the containment check below still catches it. A plain
    substring search over `text` misses this: the wrap point falls inside
    the target string, not between two of them. Strips each line's leading
    indentation and comment marker before concatenating with no separator,
    since that's exactly what a word-wrapped token needs undone; unrelated
    words glue together too, which is harmless for a containment check."""
    return "".join(_LINE_LEADER.sub("", line, count=1) for line in text.split("\n"))


def test_no_tracked_file_cites_a_maintainer_path():
    hits = []
    for rel in _tracked_files():
        if rel in _EXEMPT:
            continue
        text = _read_text(rel)
        if text is not None and "maintainer/" in _joined(text):
            hits.append(rel)
    assert not hits, (
        "these tracked files cite a maintainer/-relative path a clone of "
        f"this repo cannot follow -- restate the fact inline instead: {hits}"
    )


def _maintainer_doc_basenames() -> set[str]:
    """Every maintainer-only doc/plan basename, from TODO.md plus every
    *.md under maintainer/docs/ and maintainer/plans/. Discovered from disk
    rather than hardcoded, since the set grows with every new plan file and
    a hardcoded list would silently stop covering new ones."""
    maintainer = ROOT / "maintainer"
    if not maintainer.is_dir():
        return set()
    names = set()
    todo = maintainer / "TODO.md"
    if todo.is_file():
        names.add(todo.name)
    for sub in ("docs", "plans"):
        subdir = maintainer / sub
        if subdir.is_dir():
            names.update(p.name for p in subdir.glob("*.md"))
    return names


def test_no_tracked_file_cites_a_maintainer_doc_by_bare_filename():
    basenames = _maintainer_doc_basenames()
    if not basenames:
        pytest.skip("maintainer/ not present on disk -- nothing to check basenames against")
    hits = []
    for rel in _tracked_files():
        if rel in _EXEMPT:
            continue
        text = _read_text(rel)
        if text is None:
            continue
        joined = _joined(text)
        cited = sorted(name for name in basenames if name in joined)
        if cited:
            hits.append(f"{rel}: {cited}")
    assert not hits, (
        "these tracked files name a maintainer-only doc/plan by its bare "
        "filename -- a clone without maintainer/ can't resolve it; restate "
        f"the fact inline instead: {hits}"
    )


def test_joined_catches_a_filename_wrapped_mid_token():
    """Regression for the real miss, reproducing the exact wrap that escaped
    both checks in tests/test_lazy_viewport.py before it was fixed by hand
    (814e3a2): the citation was intact character-for-character, just split
    across two `# `-commented lines at a hyphen inside the filename."""
    wrapped = (
        "    # caller that never showed its window. MEASURED, against plans/descape-\n"
        "    # dedup-qapp-stepped-window.md's expectation: forcing show=True here does\n"
    )
    assert "descape-dedup-qapp-stepped-window.md" not in wrapped
    assert "descape-dedup-qapp-stepped-window.md" in _joined(wrapped)
