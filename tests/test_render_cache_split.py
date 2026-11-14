"""render_cache.py's split into per-area modules and mixins keeps the rules in descape/render_cache.py's docstring."""

from __future__ import annotations

import ast
import itertools
import types
from pathlib import Path

from descape.render_cache_base import _ChunkCacheBase
from descape.render_cache_mip import _ChunkCacheMipMixin
from descape.render_cache_patch import _ChunkCachePatchMixin
from descape.render_cache_units import _ChunkCacheUnitsMixin
from testkit import render_family

DESCAPE = Path(__file__).resolve().parent.parent / "descape"
MIXINS = (_ChunkCacheMipMixin, _ChunkCacheUnitsMixin, _ChunkCachePatchMixin)
_LEAVES = {"render_cache_common", "render_cache_splice"}
_MIXIN_MODULES = {"render_cache_mip", "render_cache_units", "render_cache_patch"}
# Each render_cache module and the render_cache modules it may import: only those below it.
ALLOWED = {
    "render_cache_common": set(),
    "render_cache_splice": set(),
    **dict.fromkeys(_MIXIN_MODULES, _LEAVES),
    "render_cache_base": _LEAVES | _MIXIN_MODULES,
    "render_cache_leveled": _LEAVES | _MIXIN_MODULES | {"render_cache_base"},
    "render_cache": _LEAVES | _MIXIN_MODULES | {"render_cache_base", "render_cache_leveled"},
}


def _defined(cls: type) -> set[str]:
    """Names the class body defines or annotates, Python's own dunder bookkeeping (__module__, __doc__, ...) aside."""
    names = set(vars(cls).get("__annotations__", {}))
    for name, value in vars(cls).items():
        if not (name.startswith("__") and name.endswith("__")) or isinstance(value, types.FunctionType):
            names.add(name)
    return names


def _cache_imports(path: Path) -> set[str]:
    """The render_cache modules a file imports anywhere in it, by any import form."""
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
    return {name for name in found if name.startswith("render_cache")}


def test_every_render_cache_module_is_in_the_graph_and_the_family():
    on_disk = {p.stem for p in DESCAPE.glob("render_cache*.py")}
    assert on_disk == set(ALLOWED)
    assert set(render_family.RENDER_CACHE_MODULES) == {f"descape.{name}" for name in ALLOWED}


def test_each_render_cache_module_imports_only_modules_below_it():
    wrong = {name: sorted(extra) for name in ALLOWED if (extra := _cache_imports(DESCAPE / f"{name}.py") - ALLOWED[name])}
    assert not wrong


def test_the_mixins_follow_the_base_in_table_order():
    mro = _ChunkCacheBase.__mro__
    assert mro[1:4] == MIXINS


def test_no_mixin_defines_init():
    assert not [mixin.__name__ for mixin in MIXINS if "__init__" in vars(mixin)]


def test_no_name_is_defined_by_two_classes():
    clashes = {
        (a.__name__, b.__name__): shared
        for a, b in itertools.combinations([_ChunkCacheBase, *MIXINS], 2)
        if (shared := _defined(a) & _defined(b))
    }
    assert not clashes
