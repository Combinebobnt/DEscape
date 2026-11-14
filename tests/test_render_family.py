"""Tests and tools set a render or render_cache global only through `testkit.render_family`.

TASK-629 splits `descape/render.py` and `descape/render_cache.py`. A patch on
the old module would then miss every reader that moved, and could still pass.
The scan (`testkit.module_family_scan`) fails on each raw form of
module-dict access through an alias of a family module in tests/, tools/ and
testkit/, and on any use of a render module as a value (a patch list names
the family, `RENDER` or `RENDER_CACHE`, instead). The read rule, a read or
import of a name the module only imports, also runs over descape/ and
native/, so a stale `render.X` fails here even in a file no test imports.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from testkit import render_family
from testkit.module_family import ModuleFamily
from testkit.module_family_scan import module_defines, python_files, scan_files, scan_source

ROOT = Path(__file__).resolve().parent.parent
_HELPERS = tuple(ROOT / "testkit" / name for name in ("render_family.py", "module_family.py", "module_family_scan.py"))

# A family module reached through something the scan cannot follow: file -> {name: module}.
_HAND_LISTED_ALIASES: dict[str, dict[str, str]] = {}


def _fix(module_name: str) -> str:
    stem = module_name.rsplit(".", 1)[-1]
    return f"set it with testkit.render_family.set_{stem}_global(), read it with {stem}_global()"


_BARE_FIX = "name the family (testkit.render_family.RENDER or RENDER_CACHE) and apply it with ModuleFamily.patch()"


def _families() -> list[tuple[str, ...]]:
    return [render_family.RENDER_MODULES, render_family.RENDER_CACHE_MODULES]


def test_no_test_or_tool_reaches_into_a_render_modules_globals() -> None:
    files = python_files(ROOT, ("tests", "tools", "testkit"), exclude=_HELPERS)
    problems = []
    for family in _families():
        problems += scan_files(
            ROOT, files, family, module_defines, _HAND_LISTED_ALIASES, fix=_fix, bare_values=True, bare_fix=_BARE_FIX
        )
    assert not problems, "\n".join(problems)


def test_no_code_reads_a_name_a_render_module_only_imports() -> None:
    files = python_files(ROOT, ("descape", "native"))
    problems = []
    for family in _families():
        problems += scan_files(ROOT, files, family, module_defines, reads_only=True)
    assert not problems, "\n".join(problems)


_FAKE_DEFINES = {"descape.render": frozenset({"render_terrain", "SEAM_SHADE"})}.__getitem__
_RENDER = ("descape.render",)
_IMPORT = "from descape import render\n"


@pytest.mark.parametrize(
    "source",
    [
        _IMPORT + "patches = [(render, 'SEAM_SHADE', 1.0)]",
        _IMPORT + "def module():\n    return render",
        _IMPORT + "apply(render, 'SEAM_SHADE', 1.0)",
        _IMPORT + "for module in (render,):\n    pass",
        _IMPORT + "holder.module = render",
    ],
)
def test_the_scan_flags_a_render_module_used_as_a_value(source: str) -> None:
    found = scan_source(source, _RENDER, _FAKE_DEFINES, bare_values=True, bare_fix="fix")
    assert any("descape.render used as a value" in message for _line, message in found), found


@pytest.mark.parametrize(
    "source",
    [
        _IMPORT + "img = render.render_terrain(scenario)",
        _IMPORT + "path = render.__file__",
        _IMPORT + "from testkit.render_family import RENDER\npatches = [(RENDER, 'SEAM_SHADE', 1.0)]",
        _IMPORT + "monkeypatch.setattr(render.SomeClass, 'method', f)",
    ],
)
def test_the_scan_passes_attribute_use_and_a_named_family(source: str) -> None:
    found = scan_source(source, _RENDER, _FAKE_DEFINES, bare_values=True, bare_fix="fix")
    assert not [m for _line, m in found if "used as a value" in m], found


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from . import render\nspan = render.tile_span(1)", "read of descape.render.tile_span"),
        ("from .render import tile_span", "import of descape.render.tile_span"),
        ("from .. import render\nspan = render.tile_span(1)", "read of descape.render.tile_span"),
        ("from . import render\n\n\ndef f(x: render.tile_span):\n    pass", "read of descape.render.tile_span"),
        ("from . import render\n\n\ndef f(x) -> render.tile_span:\n    pass", "read of descape.render.tile_span"),
    ],
)
def test_the_read_rule_resolves_relative_imports(source: str, expected: str) -> None:
    package = "descape.sub" if "from .." in source else "descape"
    found = scan_source(source, _RENDER, _FAKE_DEFINES, package=package, reads_only=True)
    assert any(expected in message for _line, message in found), found


def test_the_read_rule_ignores_writes_and_defined_names() -> None:
    source = "from . import render\nrender.SEAM_SHADE = 1\nimg = render.render_terrain(s)"
    assert scan_source(source, _RENDER, _FAKE_DEFINES, package="descape", reads_only=True) == []


_HELPERS_BY_FAMILY = {
    "RENDER_MODULES": (
        render_family.RENDER,
        render_family.set_render_global,
        render_family.render_global,
        render_family.render_global_bound,
    ),
    "RENDER_CACHE_MODULES": (
        render_family.RENDER_CACHE,
        render_family.set_render_cache_global,
        render_family.render_cache_global,
        render_family.render_cache_global_bound,
    ),
}


@pytest.fixture(params=sorted(_HELPERS_BY_FAMILY))
def fake_family(request, monkeypatch):
    """Two stand-in modules as one family, the way a cut leaves it; yields (modules, family, set, get, bound)."""
    modules = (types.ModuleType(f"_render_family_fake_a_{request.param}"), types.ModuleType(f"_render_family_fake_b_{request.param}"))
    for module in modules:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(render_family, request.param, tuple(m.__name__ for m in modules))
    return (modules, *_HELPERS_BY_FAMILY[request.param])


def test_a_family_name_is_set_in_every_binder_and_undone(fake_family) -> None:
    (first, second), _family, set_global, get_global, _bound = fake_family
    shared = object()
    first.SHARED = second.SHARED = shared
    first.ONLY_FIRST = "first"
    mp = pytest.MonkeyPatch()
    set_global(mp, "SHARED", 5)
    set_global(mp, "ONLY_FIRST", 6)
    assert (first.SHARED, second.SHARED, first.ONLY_FIRST) == (5, 5, 6)
    assert "ONLY_FIRST" not in vars(second)
    assert get_global("SHARED") == 5
    mp.undo()
    assert first.SHARED is shared and second.SHARED is shared and first.ONLY_FIRST == "first"


def test_an_unbound_name_raises_unless_a_default_is_given(fake_family) -> None:
    (first, second), _family, set_global, get_global, bound = fake_family
    with pytest.raises(LookupError, match=first.__name__):
        set_global(pytest.MonkeyPatch(), "MISSING", 1)
    with pytest.raises(LookupError):
        get_global("MISSING")
    assert get_global("MISSING", None) is None
    assert not bound("MISSING")
    second.MISSING = "bound"
    assert bound("MISSING") and get_global("MISSING", None) == "bound"


def test_bindings_that_disagree_are_refused(fake_family) -> None:
    (first, second), _family, _set, get_global, _bound = fake_family
    first.NAME, second.NAME = object(), object()
    with pytest.raises(AssertionError, match="NAME"):
        get_global("NAME")


def test_patch_and_current_route_a_family_and_pass_any_other_object(fake_family) -> None:
    (first, second), family, _set, _get, _bound = fake_family
    first.X = second.X = "old"
    other = types.SimpleNamespace(X="plain")
    mp = pytest.MonkeyPatch()
    ModuleFamily.patch(mp, family, "X", "new")
    ModuleFamily.patch(mp, other, "X", "also new")
    assert (first.X, second.X, other.X) == ("new", "new", "also new")
    assert ModuleFamily.current(family, "X") == "new" and ModuleFamily.current(other, "X") == "also new"
    mp.undo()
    assert (first.X, second.X, other.X) == ("old", "old", "plain")


def test_a_family_object_refuses_an_attribute_write() -> None:
    """`RENDER.X = 1` would set X on the family and patch no module, so it raises."""
    with pytest.raises(AttributeError):
        render_family.RENDER.SEAM_SHADE = 1.0
    with pytest.raises(AttributeError):
        pytest.MonkeyPatch().setattr(render_family.RENDER_CACHE, "WARM_NATIVE_INPUTS", False, raising=False)


def test_the_real_families_bind_their_module() -> None:
    from descape import render_cache, render_flat

    assert render_family.render_global("render_terrain") is render_flat.render_terrain
    assert render_family.render_cache_global("IsoChunkCache") is render_cache.IsoChunkCache
