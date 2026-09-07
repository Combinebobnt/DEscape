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

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

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


def test_no_tracked_file_cites_a_maintainer_path():
    hits = []
    for rel in _tracked_files():
        if rel in _EXEMPT:
            continue
        text = _read_text(rel)
        if text is not None and "maintainer/" in text:
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
        cited = sorted(name for name in basenames if name in text)
        if cited:
            hits.append(f"{rel}: {cited}")
    assert not hits, (
        "these tracked files name a maintainer-only doc/plan by its bare "
        "filename -- a clone without maintainer/ can't resolve it; restate "
        f"the fact inline instead: {hits}"
    )
