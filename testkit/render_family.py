"""Sets and reads the render and render_cache modules' globals for tests and tools.

TASK-629 splits `descape/render.py` and `descape/render_cache.py` into
several modules each. A patch on the old module would then miss every reader
that moved, while the test still passed. So no test or tool writes to a
render or render_cache module directly: every write goes through
`set_render_global()` or `set_render_cache_global()`, which set the name in
every module of the family that binds it (`testkit.module_family`). Each cut
appends the modules it creates to its family's tuple.

A patch list that mixes families and other objects names a family by
`RENDER` or `RENDER_CACHE`, never by the module object, and applies each
entry with `ModuleFamily.patch()`. `tests/test_render_family.py` fails on the
raw forms, and on any use of a render module as a value.

A tool that loads another checkout (`--root`, `DESCAPE_BENCH_ROOT`) refuses
one older than the last cut that moved a name it imports (`tools/bench_gui_drag.py`'s
`AB_ROOT_FLOOR`), whose render names are not where the tool imports them.
"""

from __future__ import annotations

from typing import Any

from testkit.module_family import ModuleFamily

RENDER_MODULES: tuple[str, ...] = (
    "descape.render",
    "descape.render_paint",
    "descape.render_units",
    "descape.render_sprites",
    "descape.render_bbox",
    "descape.render_iso",
    "descape.render_sloped",
    "descape.render_flat",
)
RENDER_CACHE_MODULES: tuple[str, ...] = (
    "descape.render_cache",
    "descape.render_cache_common",
    "descape.render_cache_splice",
    "descape.render_cache_base",
    "descape.render_cache_leveled",
    "descape.render_cache_mip",
    "descape.render_cache_units",
    "descape.render_cache_patch",
)

RENDER = ModuleFamily("render", lambda: RENDER_MODULES)
RENDER_CACHE = ModuleFamily("render_cache", lambda: RENDER_CACHE_MODULES)

patch = ModuleFamily.patch
current = ModuleFamily.current


def set_render_global(monkeypatch: Any, name: str, value: Any) -> None:
    """Sets `name` to `value` in every render module that binds it, undone with `monkeypatch`."""
    RENDER.set_global(monkeypatch, name, value)


def render_global(name: str, *default: Any) -> Any:
    """The current value of `name`, which every render module binding it must agree on (`default` as for `getattr()`)."""
    return RENDER.get_global(name, *default)


def render_global_bound(name: str) -> bool:
    """Whether any render module binds `name`."""
    return RENDER.is_bound(name)


def set_render_cache_global(monkeypatch: Any, name: str, value: Any) -> None:
    """Sets `name` to `value` in every render_cache module that binds it, undone with `monkeypatch`."""
    RENDER_CACHE.set_global(monkeypatch, name, value)


def render_cache_global(name: str, *default: Any) -> Any:
    """The current value of `name`, which every render_cache module binding it must agree on (`default` as for `getattr()`)."""
    return RENDER_CACHE.get_global(name, *default)


def render_cache_global_bound(name: str) -> bool:
    """Whether any render_cache module binds `name`."""
    return RENDER_CACHE.is_bound(name)
