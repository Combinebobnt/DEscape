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
    window, so none of them would catch SlopedChunkCache's depth-1 mip
    level set (_init_mip_levels({0: tile_px}), see that class's own
    docstring) interacting badly with Track B-D-c/d's real LOD-selection
    paint() path (aa885ee/5547904), which landed on viewer.py AFTER this
    class was written and was never driven through a Sloped cache before.
    Asserts the chunk cache actually gets populated (not just constructed)
    by a real MapCanvasItem.paint() dispatch, at both the initial
    fit-to-view scale and a zoomed-in scale that forces _select_mip() to
    read the transform."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.terrain_edit_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        _show_and_settle(window)
        assert len(window._cache._cache) > 0, "fit-to-view paint composited no chunks"

        window.map_view.scale(8.0, 8.0)
        QApplication.processEvents()
        assert len(window._cache._cache) > 0, "zoomed-in paint composited no chunks"
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


def test_sloped_enables_copy_paste() -> None:
    """Copy/Paste have no dedicated Sloped gate of their own -- copy_ok
    derives from draw_action/fill_action/elevation_action/set_level_action.
    isEnabled(), so it followed the four tools out of the gate at Step 4 the
    same way it followed them in. Pinned explicitly anyway, same reason the
    old disabled-side version was: a future edit to _update_tool_enabled's
    copy/paste block could silently break that chain without any test here
    catching it.

    Paste stays disabled throughout -- the clipboard is empty and its kind
    gate is what refuses it, which is exactly what makes Copy the load-
    bearing half of this check."""
    window = conftest.terrain_edit_window()
    try:
        window.draw_action.setChecked(True)
        window.terrain_style_combo.setCurrentText("Stepped")
        assert window.copy_action.isEnabled(), "baseline: Copy should be enabled in Stepped"

        window.terrain_style_combo.setCurrentText("Sloped")
        assert window.copy_action.isEnabled()

        window.on_hover((10, 10))  # copy_tile() reads _hover_tile, not a click
        window.copy_tile()
        assert window._clipboard is not None, "Copy in Sloped produced no clipboard entry"
        assert window.paste_action.isEnabled(), "Paste must follow a matching-kind Copy in Sloped"
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
        assert window.terrain_combo.isEnabled(), "Draw's terrain param must come back with it"
    finally:
        window.edit_history.mark_saved()
        window.close()
