"""Track C3's descape/viewer.py wiring -- the Qt-registry half
that tests/test_sloped_chunks.py/test_sloped_geometry.py/test_sloped_render.py
can't reach: switching Elevation View to Sloped actually builds a
SlopedChunkCache and pushes it through MapView.set_source, and the toolbar
ends up in the state Sloped's backends can actually honour.

That second half inverted at Track C4's Step 4. The Terrain tools, Copy/Paste
and hit-testing were pinned here as DISABLED for as long as Sloped had no way
to resolve a click to a tile; C4 gave it one, so they are pinned as enabled
now. Units mode followed at Track C5's Step 4 (see
tests/test_unit_selection_viewer.py, which owns that pin). What is still
pinned disabled is what still has no backend: Isometric View, a Flat-only
transform.

Same technique and default-tier rationale as tests/test_toolbar_params.py/
test_fill_tool.py (real offscreen ViewerWindow, blank 120x120 template, no
examples/ corpus needed).

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's own docstring for why.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _show_and_settle(window) -> None:
    """Same requirement tests/test_lazy_viewport.py's own _show_and_settle()
    documents: showing map_view alone never triggers a real paint cycle,
    only its top-level ancestor being shown does."""
    from PyQt5.QtWidgets import QApplication

    window.resize(300, 300)
    window.show()
    QApplication.processEvents()
    QApplication.processEvents()


def test_sloped_actually_composites_through_a_real_paint_cycle() -> None:
    """The gap the other tests here don't cover: none of them show the
    window, so none of them would catch SlopedChunkCache's mip ladder (the
    coarser levels only, TASK-031.60; see that class's own docstring)
    interacting badly with Track B-D-c/d's real LOD-selection paint() path
    (aa885ee/5547904). Asserts the chunk cache actually gets populated (not
    just constructed) by a real MapCanvasItem.paint() dispatch: at the
    initial fit-to-view scale, at a coarser level, and at a zoomed-in scale
    that forces _select_mip() to read the transform, at level 0, the finest
    level Sloped has."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.terrain_edit_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        _show_and_settle(window)
        assert len(window._cache._cache) > 0, "fit-to-view paint composited no chunks"
        assert min(key[0] for key in window._cache._cache) < 0, "fit-to-view paint stayed at level 0"

        window.map_view.scale(8.0, 8.0)
        QApplication.processEvents()
        assert any(key[0] == 0 for key in window._cache._cache), "zoomed-in paint composited no level-0 chunk"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_switching_to_sloped_builds_sloped_chunk_cache() -> None:
    from descape.render_cache import SlopedChunkCache

    window = conftest.terrain_edit_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        assert window._terrain_style == "sloped"
        assert isinstance(window._cache, SlopedChunkCache)
        assert window._cache.style == "sloped"
        assert window.map_view._terrain_style == "sloped"
        # set_source's assert cache.style == terrain_style already guards
        # cross-wiring; this just confirms the call actually landed rather
        # than silently no-op'ing.
        assert window.map_view._canvas_item is not None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_sloped_hit_testing_resolves_real_tiles() -> None:
    """Track C4 landed, inverting this file's old
    test_sloped_has_no_hit_testing_yet. Its original intent is KEPT, not
    dropped: _pick_tile must resolve through Sloped's own pick plane and
    must still never fall through to Flat's plain int-division lookup,
    which would silently "work" for a style whose screen geometry doesn't
    match a flat grid at all. The off-map half below is what pins that --
    Flat's division answers a real tile for the canvas's top-left corner,
    where Sloped's silhouette has no tile at all."""
    from PyQt5.QtCore import QPointF

    window = conftest.terrain_edit_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        mv = window.map_view
        mm = window.scenario.map_manager
        proj = mv._iso_proj

        centre = QPointF(proj.canvas_w // 2, proj.canvas_h // 2)
        tile = mv._pick_tile(centre)
        assert tile is not None, "the canvas centre must land on a tile"
        assert 0 <= tile[0] < mm.map_width and 0 <= tile[1] < mm.map_height
        assert mv._pos_on_map(centre) is True

        # (0, 0) is outside the map's diamond silhouette but inside the
        # canvas rect -- exactly the case Flat's int division gets wrong.
        corner = QPointF(0, 0)
        assert mv._pick_tile(corner) is None
        assert mv._pos_on_map(corner) is False

        polygon = mv._tile_polygon(*tile)
        assert polygon is not None and polygon.count() >= 4
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("tool", ["draw", "fill", "elevation", "set_level"])
def test_sloped_enables_edit_tools_the_same_way_stepped_does(tool: str) -> None:
    """Track C4's Step 4 released these, inverting this file's old
    test_sloped_disables_edit_tools_regardless_of_write_ok. The intent is
    kept, only turned around: Sloped must not apply a style gate of its own
    on top of the scenario-level write support the blank template already
    satisfies (it is square and terrain-write-supported), so each action's
    enabled state must match its Stepped baseline exactly rather than merely
    being enabled somewhere along the way."""
    window = conftest.terrain_edit_window()
    try:
        action = {
            "draw": window.draw_action,
            "fill": window.fill_action,
            "elevation": window.elevation_action,
            "set_level": window.set_level_action,
        }[tool]

        window.terrain_style_combo.setCurrentText("Stepped")
        assert action.isEnabled(), f"{tool} should be enabled in Stepped as a baseline"

        window.terrain_style_combo.setCurrentText("Sloped")
        assert action.isEnabled(), f"{tool} must be enabled in Sloped now that C4 has landed"

        # Round trip, kept from the old test: switching back off Sloped must
        # not leave anything wedged.
        window.terrain_style_combo.setCurrentText("Stepped")
        assert action.isEnabled(), f"{tool} must stay enabled after leaving Sloped"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_sloped_enables_select_and_copy_paste_round_trip() -> None:
    """Select has no dedicated Sloped gate of its own -- the same has_map/
    write_ok gate as Eyedropper, so it followed the four terrain tools out
    of the Step 4 gate the same way Eyedropper did. Pinned explicitly
    anyway, same reason the old version was: a future edit to
    _update_tool_enabled could silently break that chain with no test here
    catching it.

    Phase 2.8 retired the old tool-scoped clipboard (gated on whichever
    edit tool was active) in favour of one gated on self._region/
    self._region_clipboard alone, so this now exercises a full
    select -> copy -> paste round trip in Sloped instead of pinning a
    tool-enabled chain that no longer exists."""
    window = conftest.terrain_edit_window()
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        assert window.select_action.isEnabled(), "baseline: Select should be enabled in Stepped"

        window.terrain_style_combo.setCurrentText("Sloped")
        assert window.select_action.isEnabled()

        mm = window.scenario.map_manager
        mm.get_tile(10, 10).terrain_id = 2  # BEACH, distinct from the blank template's default

        window.on_region_selected((10, 10, 11, 11))
        window.copy_region()
        assert window._region_clipboard is not None, "Copy in Sloped produced no clipboard entry"
        assert window.paste_action.isEnabled()

        window.on_hover((20, 20))
        window.paste_region()
        assert mm.get_tile(20, 20).terrain_id == 2, "Paste in Sloped must still write real terrain"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_opening_and_new_map_while_in_sloped_keep_sloped() -> None:
    """GH #87 open-in-style: neither load path resets the Elevation View.
    Asserted on the combo and the log line; the cache type is a secondary check."""
    from descape.render_cache import SlopedChunkCache
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.terrain_edit_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")

        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.terrain_style_combo.currentText() == "Sloped"
        last = window.status_log.toPlainText().splitlines()[-1]
        assert last.startswith("Loaded ") and "style=sloped)" in last, last
        assert isinstance(window._cache, SlopedChunkCache)

        window.new_map()
        assert window.terrain_style_combo.currentText() == "Sloped"
        last = window.status_log.toPlainText().splitlines()[-1]
        assert last.startswith("Created ") and "style=sloped)" in last, last
        assert isinstance(window._cache, SlopedChunkCache)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_cycling_every_elevation_view_logs_no_failure() -> None:
    """GH #87 style switch: every combo entry, there and back, renders
    without a "failed" line in the status log."""
    from descape.terrain_style import STYLE_LABELS

    window = conftest.terrain_edit_window()
    try:
        before = len(window.status_log.toPlainText().splitlines())
        for label in (*STYLE_LABELS, *reversed(STYLE_LABELS)):
            window.terrain_style_combo.setCurrentText(label)
            assert window.terrain_style_combo.currentText() == label
        new_lines = window.status_log.toPlainText().splitlines()[before:]
        assert any(line.startswith("Elevation view: ") for line in new_lines), "no style switch was logged"
        assert not [line for line in new_lines if "failed" in line.lower()]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_sloped_disables_isometric_view_checkbox() -> None:
    window = conftest.terrain_edit_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        assert not window.iso_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_sloped_keeps_the_selected_tool_rather_than_forcing_pan() -> None:
    """Inverted at Step 4, from the old
    test_sloped_selected_tool_forced_back_to_pan. Switching styles mid-edit
    used to knock Draw back to Pan, purely as a consequence of Sloped
    disabling it; now that it stays enabled, the tool must survive the switch
    -- and its params with it, since terrain_param_ok derives from
    draw_action.isEnabled().

    The "disabling a checked QAction doesn't uncheck it" hazard
    _update_tool_enabled's own comment documents is NOT untested by this
    inversion: the non-square-file path still exercises the forced-back-to-Pan
    block, and _force_mode's own pan_action.setChecked(True) covers the Units
    route."""
    window = conftest.terrain_edit_window()
    try:
        window.draw_action.setChecked(True)
        assert window.draw_action.isChecked()

        window.terrain_style_combo.setCurrentText("Sloped")
        assert window.draw_action.isChecked(), "Draw must survive the switch into Sloped"
        assert window.draw_action.isEnabled()
        assert not window.pan_action.isChecked()
        assert window.paint_trees_param_action.isVisible(), "Draw's terrain params must come back with it"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_pick_map_point_resolves_the_continuous_point_under_a_pixel(style: str) -> None:
    """Off-grid placement's Stage 2, at the Qt boundary. `_pick_map_point`'s
    per-style branches are the one part of the inverse that the pure-geometry
    tier (tests/test_screen_to_map_point.py) cannot reach: wiring the wrong
    projection or corner_rise in here would ship silently, since the geometry
    itself would still be right.

    Checked as "lands inside the tile `_pick_tile` resolved", not against a
    hand-computed coordinate -- that is the actual contract the two call sites
    rely on, and it holds in all three styles without this file having to
    reproduce each one's projection.

    Flat is absent here and covered elsewhere, not skipped: this file's
    fixture window will not actually leave Stepped for Flat (the combo change
    is gated), and Flat's own branch is exercised end to end by the
    off-grid tests in tests/test_unit_edit_viewer.py, which force Flat
    on purpose."""
    from PyQt5.QtCore import QPointF

    window = conftest.terrain_edit_window()
    try:
        window.terrain_style_combo.setCurrentText(style)
        mv = window.map_view
        assert mv._terrain_style == style.lower(), "the fixture did not switch style"
        proj = mv._iso_proj
        pos = QPointF(proj.canvas_w // 2 + 5, proj.canvas_h // 2 + 3)

        tile = mv._pick_tile(pos)
        assert tile is not None, "fixture assumption: the probed pixel must land on a tile"
        point = mv._pick_map_point(pos)
        assert point is not None, f"{style} resolved a tile but no continuous point"
        assert (int(point[0]), int(point[1])) == tile, (
            f"{style}: {point} is not inside the tile _pick_tile resolved, {tile}"
        )
        # And it is genuinely fractional, not the tile centre in disguise.
        assert (point[0], point[1]) != (tile[0] + 0.5, tile[1] + 0.5)
    finally:
        window.edit_history.mark_saved()
        window.close()


def _bumpy_sloped_window():
    """terrain_edit_window() in Sloped over a 3-level blocked field, so a
    block of tiles carries many distinct corner shapes."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.terrain_edit_window()
    for tile in window.scenario.map_manager.terrain:
        tile.elevation = (tile.x // 2 + tile.y // 3) % 3
    window.terrain_style_combo.setCurrentText("Sloped")
    window.refresh_map()
    QApplication.processEvents()
    assert window.map_view._terrain_style == "sloped", "the fixture did not switch style"
    return window


def _normalized_shape(mv, tx: int, ty: int) -> tuple[int, int, int, int]:
    _ox, _oy, corners = mv._sloped_tile_base(tx, ty)
    d_min = min(corners)
    return tuple(d - d_min for d in corners)


def test_cached_sloped_tile_polygon_matches_a_fresh_build_point_for_point() -> None:
    """TASK-031.38: _tile_polygon's Sloped branch serves a cached tile-local
    polygon translated to the tile. Checked against the pre-cache build
    (iso_geometry's outline placed at _sloped_tile_base's origin) on every
    tile of a mixed-shape block, both forms interleaved on a cold cache so a
    key that confused them would fail here."""
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QPolygonF

    from descape import iso_geometry, map_view

    window = _bumpy_sloped_window()
    try:
        mv = window.map_view
        map_view._sloped_outline_polygon.cache_clear()
        block = [(tx, ty) for ty in range(30, 46) for tx in range(30, 46)]
        assert len({_normalized_shape(mv, tx, ty) for tx, ty in block}) >= 8, "fixture lost its shape mix"
        for tx, ty in block:
            ox, oy, corners = mv._sloped_tile_base(tx, ty)
            for coarse in (False, True):
                fn = iso_geometry.sloped_tile_outline_coarse if coarse else iso_geometry.sloped_tile_outline
                expected = QPolygonF([QPointF(ox + px, oy + py) for px, py in fn(mv._tile_pixels, *corners)])
                got = mv._tile_polygon(tx, ty, coarse=coarse)
                assert got == expected, f"tile ({tx}, {ty}) coarse={coarse}: cached polygon differs"
        assert map_view._sloped_outline_polygon.cache_info().hits > 0, "the block never hit the cache"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_sloped_highlight_rebuild_builds_only_shapes_it_has_not_seen(monkeypatch) -> None:
    """TASK-031.38: a brush-9 highlight at a neighbouring tile calls
    sloped_tile_outline_coarse once per corner shape the cache has not seen,
    not once per tile (81). Counted, not timed."""
    from descape import brush, iso_geometry, map_view

    window = _bumpy_sloped_window()
    try:
        mv = window.map_view
        mv.set_brush(9, brush.BRUSH_SHAPE_SQUARE)
        real = iso_geometry.sloped_tile_outline_coarse
        calls = []

        def counting(*args):
            calls.append(args)
            return real(*args)

        monkeypatch.setattr(iso_geometry, "sloped_tile_outline_coarse", counting)
        map_view._sloped_outline_polygon.cache_clear()

        def shapes(cx: int, cy: int) -> set:
            tiles = brush.brush_tiles(cx, cy, 9, brush.BRUSH_SHAPE_SQUARE, mv._map_width, mv._map_height)
            assert len(tiles) == 81
            return {_normalized_shape(mv, tx, ty) for tx, ty in tiles}

        mv._update_highlight(40, 40)
        first = len(calls)
        assert first == len(shapes(40, 40)) < 81
        mv._update_highlight(41, 40)
        assert len(calls) - first == len(shapes(41, 40) - shapes(40, 40)), "a seen shape was rebuilt"
        assert mv._highlight_outline_item is not None and not mv._highlight_outline_item.path().isEmpty()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_scene_space_reads_stay_at_level_0_while_a_coarser_level_is_shown(monkeypatch) -> None:
    """TASK-031.60: MapView's Sloped readers (tile outline, pick, continuous
    pick, a unit's rise and outline) take scene-space geometry from level 0
    while the view paints a coarser level, and none of them builds level 0.
    The oracle is the same reads with _sloped_cache() swapped for a fresh
    cache; a stand-in carrying the shown level's corner_rise must disagree."""
    from types import SimpleNamespace

    from PyQt5.QtCore import QPointF

    from descape import render, unit_pick
    from descape.render_cache import SlopedChunkCache

    window = conftest.terrain_edit_window()
    try:
        mm = window.scenario.map_manager
        for tile in mm.terrain:
            tile.elevation = (tile.x // 2 + tile.y // 3) % 3
        unit_tile = next((x, y) for y in range(50, 70) for x in range(50, 70) if mm.get_tile(x, y).elevation == 2)
        window._ensure_unit_edits().add(1, 83, unit_tile[0] + 0.5, unit_tile[1] + 0.5)
        window.terrain_style_combo.setCurrentText("Stepped")
        window.terrain_style_combo.setCurrentText("Sloped")
        window._rebuild_unit_index()
        _show_and_settle(window)
        mv, cache = window.map_view, window._cache
        visible = mv.viewport_chunk_target()[0]
        assert visible < 0 and {key[0] for key in cache._cache} == {visible}, "the fit paint did not stay coarse"
        builds: list[int] = []
        real = cache._commit_level
        monkeypatch.setattr(cache, "_commit_level", lambda mip, *a: (builds.append(mip), real(mip, *a))[1])

        (entry,) = [e for e in mv._unit_index.entries if e.unit.unit_const == 83]
        tiles = [(x, y) for y in range(unit_tile[1] - 3, unit_tile[1] + 4) for x in range(unit_tile[0] - 3, unit_tile[0] + 4)]

        def reads():
            polygons = {t: mv._tile_polygon(*t) for t in tiles}
            points = [QPointF(p.boundingRect().center()) for p in polygons.values()]
            return (
                {t: [(q.x(), q.y()) for q in p] for t, p in polygons.items()},
                [mv._pick_tile(p) for p in points],
                [mv._pick_map_point(p) for p in points],
                mv._range_ring_rise_px(entry),
                [list(poly) for poly in (mv._unit_polygons_for(entry) or [])],
            )

        got = reads()
        assert got[4] and None not in got[1], "no unit outline or an unpicked tile -- vacuous"
        assert builds == [] and cache._levels[0].gen == -1, "a scene-space read built level 0"
        assert 0 not in {key[0] for key in cache._cache}, "a scene-space read composited level 0"

        elevations, corner_rise, proj = render.sloped_elevations_and_proj(window.scenario)
        fresh = SlopedChunkCache(window.scenario, elevations, corner_rise, proj, proj.tile_px)

        def stand_in(rise):
            # Only what the readers touch, so no reader can reach a level of its own.
            return SimpleNamespace(corner_rise=rise, pick_tile=fresh.pick_tile, with_units=True, sprites_enabled=False)

        right = stand_in(fresh.corner_rise)
        monkeypatch.setattr(mv, "_sloped_cache", lambda: right)
        assert got == reads(), "a read differs from a fresh level-0 cache's"

        shown_rise = cache._level_corner_rise(cache._levels[visible])
        wrong = stand_in(shown_rise)
        monkeypatch.setattr(mv, "_sloped_cache", lambda: wrong)
        assert unit_pick.unit_rise_px_for(entry, shown_rise) != got[3], "the unit's rise is the same at both levels -- vacuous"
        assert reads()[0] != got[0], "the shown level's corner_rise outlines the same -- vacuous"
    finally:
        window.edit_history.mark_saved()
        window.close()
