"""Track C3's descape/viewer.py wiring (docs/PLAN_V2_6.md's Track C, spec'd
in the maintainer plan's own "## Progress" section) -- the Qt-registry half
that tests/test_sloped_chunks.py/test_sloped_geometry.py/test_sloped_render.py
can't reach: switching Elevation View to Sloped actually builds a
SlopedChunkCache and pushes it through MapView.set_source, and every tool
that has no hit-testing yet (Track C4) stays disabled rather than silently
accepting clicks it can't resolve to a tile.

Same technique and default-tier rationale as tests/test_toolbar_params.py/
test_fill_tool.py (real offscreen ViewerWindow, blank 120x120 template, no
examples/ corpus needed).

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's own docstring for why.
"""

from __future__ import annotations

import pytest

import conftest
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _edit_window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Edit")
    return window


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

    window = _edit_window()
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
    from descape.render import SlopedChunkCache

    window = _edit_window()
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


def test_sloped_has_no_hit_testing_yet() -> None:
    """Track C4's job, not C3's -- _pick_tile/_pos_on_map must not fall
    through to Flat's plain int-division lookup (which would silently
    "work" for a style whose screen geometry doesn't match a flat grid at
    all)."""
    from PyQt5.QtCore import QPointF

    window = _edit_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        mv = window.map_view
        # A point well inside the canvas -- Flat's own division would
        # happily resolve this to a real (x, y) tile.
        pos = QPointF(mv._tile_pixels * 5, mv._tile_pixels * 5)
        assert mv._pick_tile(pos) is None
        assert mv._pos_on_map(pos) is False
        assert mv._tile_polygon(5, 5) is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("tool", ["terrain", "fill", "elevation", "set_level"])
def test_sloped_disables_edit_tools_regardless_of_write_ok(tool: str) -> None:
    """The blank template is square and terrain-write-supported, so in
    Stepped/Flat these actions are enabled in Edit mode -- Sloped must
    override that down to disabled purely because it can't resolve a click
    to a tile yet, not because of scenario-level write support."""
    window = _edit_window()
    try:
        action = {
            "terrain": window.terrain_action,
            "fill": window.fill_action,
            "elevation": window.elevation_action,
            "set_level": window.set_level_action,
        }[tool]

        window.terrain_style_combo.setCurrentText("Stepped")
        assert action.isEnabled(), f"{tool} should be enabled in Stepped as a baseline"

        window.terrain_style_combo.setCurrentText("Sloped")
        assert not action.isEnabled(), f"{tool} must be disabled in Sloped (no hit-testing until C4)"

        # Round trip: switching back off Sloped must not leave anything
        # wedged disabled.
        window.terrain_style_combo.setCurrentText("Stepped")
        assert action.isEnabled(), f"{tool} must re-enable after leaving Sloped"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_sloped_disables_copy_paste() -> None:
    """Copy/Paste have no dedicated Sloped gate of their own -- copy_ok
    derives from the same terrain_action/fill_action/elevation_action/
    set_level_action.isEnabled() checks sloped_editable already gates, so
    this is transitive by construction. Pinned explicitly anyway: a future
    edit to _update_tool_enabled's copy/paste block could silently break
    that chain without any test here catching it."""
    window = _edit_window()
    try:
        window.terrain_action.setChecked(True)
        window.terrain_style_combo.setCurrentText("Stepped")
        assert window.copy_action.isEnabled(), "baseline: Copy should be enabled in Stepped"

        window.terrain_style_combo.setCurrentText("Sloped")
        assert not window.copy_action.isEnabled()
        assert not window.paste_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_sloped_disables_isometric_view_checkbox() -> None:
    window = _edit_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        assert not window.iso_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_sloped_selected_tool_forced_back_to_pan() -> None:
    """Same "disabling a checked QAction doesn't uncheck it" hazard
    _update_tool_enabled's own comment documents for the non-square-file
    case -- Terrain selected, then switching to Sloped must force Pan
    rather than leaving Terrain checked-but-disabled."""
    window = _edit_window()
    try:
        window.terrain_action.setChecked(True)
        assert window.terrain_action.isChecked()

        window.terrain_style_combo.setCurrentText("Sloped")
        assert window.pan_action.isChecked()
        assert not window.terrain_action.isChecked()
    finally:
        window.edit_history.mark_saved()
        window.close()
