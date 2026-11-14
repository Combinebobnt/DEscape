"""render.py's split into per-area modules keeps the import graph in descape/render.py's docstring."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from testkit import render_family

DESCAPE = Path(__file__).resolve().parent.parent / "descape"
_LOW = {"render_paint", "render_units"}
_MID = _LOW | {"render_sprites", "render_bbox"}
# Each render module and the render modules it may import: only those below it.
ALLOWED = {
    "render_paint": set(),
    "render_units": {"render_paint"},
    "render_sprites": _LOW,
    "render_bbox": _LOW,
    "render_iso": _MID,
    "render_sloped": _MID,
    "render_flat": _LOW | {"render_sprites"},
    "render": _MID | {"render_iso", "render_sloped", "render_flat"},
}


def _is_render(name: str) -> bool:
    return (name == "render" or name.startswith("render_")) and not name.startswith("render_cache")


def _render_imports(path: Path) -> set[str]:
    """The render modules a file imports anywhere in it, by any import form."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(a.name.removeprefix("descape.") for a in node.names if a.name.startswith("descape."))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and module.startswith("descape."):
                found.add(module.removeprefix("descape."))
            elif (node.level == 0 and module == "descape") or (node.level and not module):
                found.update(a.name for a in node.names)
            elif node.level:
                found.add(module)
    return {name for name in found if _is_render(name)}


def test_every_render_module_is_in_the_graph_and_the_family():
    on_disk = {p.stem for p in DESCAPE.glob("render*.py") if _is_render(p.stem)}
    assert on_disk == set(ALLOWED)
    assert set(render_family.RENDER_MODULES) == {f"descape.{name}" for name in ALLOWED}


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"render.py": "def f():\n    return 'one file'\n"}, "one file"),
        ({"render.py": "", "render_units.py": "def f():\n    return 'split'\n"}, "split"),
    ],
)
def test_release_render_diff_finds_a_name_in_either_layout(tmp_path, files, expected):
    """tools/release_render_diff.py's worker runs on the release tag's tree and on this one, first on sys.path."""
    (tmp_path / "descape").mkdir()
    (tmp_path / "descape" / "__init__.py").write_text("")
    for name, text in files.items():
        (tmp_path / "descape" / name).write_text(text)
    code = "import sys; sys.path[:0] = sys.argv[1:3]; import release_render_diff as r; print(r._render_fn('f')())"
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(DESCAPE.parent / "tools")], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == expected


def test_each_render_module_imports_only_render_modules_below_it():
    wrong = {name: sorted(extra) for name in ALLOWED if (extra := _render_imports(DESCAPE / f"{name}.py") - ALLOWED[name])}
    assert not wrong
