"""TASK-031.12: a Draw stroke through trees after the viewer's pack warm
(LevelWarmer's pre-derive of a level's UnitPack) renders like a fresh cache.

tests/test_native_composite.py covers an Elevate and a unit move after a
cache-level pack warm, and tests/test_terrain_units_viewer.py covers Draw
through trees with packs derived lazily by paints. This joins the two: a shown
viewer, its own LevelWarmer pre-deriving the neighbour level's pack, then a
Draw that removes and replaces the trees on that level's tiles.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_native_composite import (
    native_backend,  # noqa: F401 -- a fixture
    native_kernel,  # noqa: F401 -- native_backend requests it
)
from test_sprite_toggle_viewer import sprite_install  # noqa: F401 -- tree_sprites requests it
from test_terrain_units_viewer import (
    FOREST_OAK,
    FOREST_PALM,
    SHIFTED_TILES,
    _draw,
    _fresh_cache,
    _splice_any_batch,
    _spy,
    _styled_window,
    tree_sprites,  # noqa: F401 -- fixture, requested by name
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _packs(cache) -> dict[int, object]:
    return {m: cache._unit_pack_of(m, create=False) for m in cache.mip_levels()}


def _whole(cache, mip: int):
    return cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)


@pytest.mark.parametrize("sprites", [False, True], ids=["marks", "sprites"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_draw_through_trees_after_a_pack_warm_matches_a_fresh_render(
    native_backend, style, sprites, monkeypatch, request  # noqa: F811
) -> None:
    """Pass conditions, on a shown viewer after a forest stroke and the viewer's own level warm:
    - the warm pre-derived at least one level's pack that had no pack or pending tiles before it,
      and every resident level's pack has nothing pending afterwards;
    - a second Draw, of another forest across those trees, splices (no wholesale unit rebuild);
    - each pre-derived pack is the same object after the stroke (refreshed, not rebuilt);
    - every resident level's whole canvas equals a fresh cache's render on the native backend,
      and on numpy too.
    """
    from PyQt5.QtWidgets import QApplication

    from descape import composite_backend, settings

    if sprites:
        request.getfixturevalue("tree_sprites")
    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    _splice_any_batch(monkeypatch)
    window = _styled_window(style)
    window.resize(1200, 800)
    window.show()
    QApplication.processEvents()
    try:
        window.show_sprites_action.setChecked(sprites)
        cache = window._cache
        assert composite_backend.active_backend() == "native"
        assert cache.sprites_enabled is sprites
        cache.max_bytes *= 8  # every compared level stays resident
        _draw(window, FOREST_OAK)

        before = _packs(cache)
        window._start_level_warm()
        window._level_warmer.run_to_completion()
        warmed = {}
        for mip, pack in _packs(cache).items():
            if not cache.is_level_resident(mip):
                continue
            assert pack is not None and pack.pending().size == 0, f"mip {mip}: tiles left pending after the warm"
            old = before[mip]
            if old is None or old is not pack or old.pending().size:
                warmed[mip] = pack
        assert warmed, f"the warm derived no pack -- vacuous ({before})"
        levels = sorted(m for m in cache.mip_levels() if cache.is_level_resident(m))
        for mip in levels:
            _whole(cache, mip)  # resident before the edit, so a stale chunk would show
        assert all(cache._unit_pack_of(m, create=False) is p for m, p in warmed.items()), "a paint rebuilt a pack"
        calls = _spy(window, monkeypatch)

        _draw(window, FOREST_PALM, SHIFTED_TILES)

        (changed,) = calls["changed"]
        assert changed, "the stroke end took the wholesale unit invalidation"
        removed = {s.old_own_tile for s in changed if s.new_own_tile is None}
        assert removed, "the stroke removed no tree -- vacuous"
        assert calls["wholesale"] == 0, "the cache fell back to a wholesale source rebuild"
        for mip, pack in warmed.items():
            assert cache._unit_pack_of(mip, create=False) is pack, f"mip {mip}: the stroke rebuilt the warmed pack"
        live = {mip: _whole(cache, mip) for mip in levels}
        diffs = []
        for backend in ("native", "numpy"):
            with composite_backend.use_backend(backend):
                fresh = _fresh_cache(window)
                for mip in levels:
                    want = fresh.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)
                    if not np.array_equal(live[mip], want):
                        ys, xs = np.nonzero(np.any(live[mip] != want, axis=-1))
                        tag = "pre-derived" if mip in warmed else "painted"
                        diffs.append(f"mip {mip} ({tag}) vs {backend}: {ys.size} px, first at (y={ys[0]}, x={xs[0]})")
        assert not diffs, f"{style} after Draw through trees, differs from a fresh render: {diffs}"
    finally:
        window.edit_history.mark_saved()
        window.close()
