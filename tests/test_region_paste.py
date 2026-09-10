"""Paste Region's filter checkboxes and one-undo-step composite behaviour --
phase 2.8. Same offscreen technique tests/test_fill_tool.py documents; every
ViewerWindow() here must call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import pytest

import conftest
from descape.edit_history import CompositeDiffRecord
from descape.scenario_io import BLANK_TEMPLATE_PATH
from descape.unit_model import UnitEditModel

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN_A, _TERRAIN_B = 2, 15  # BEACH, GRASS_1 -- present in every DE version


def _window():
    """The blank template loaded, Terrain mode, Select active. Caller must
    edit_history.mark_saved() + close()."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("select")
    return window


def _make_region(window, sx0: int, sy0: int, sx1: int, sy1: int, terrain_id: int, elevation: int):
    """Builds a region [sx0,sx1)x[sy0,sy1) with a distinct terrain_id and a
    legal elevation (via the real Set Elevation stroke, not a hand-assigned
    value -- see tests/test_region_clipboard.py's own fixture-construction
    comment on why), then copies it. Returns the RegionBlock."""
    window._on_tool_selected("draw")
    window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(terrain_id))
    for y in range(sy0, sy1):
        for x in range(sx0, sx1):
            window.on_edit_stroke_start()
            window.on_edit_stroke_tile(x, y, 0)
            window.on_edit_stroke_end()
    if elevation:
        window._on_tool_selected("set_level")
        window.elevation_level_spin.setValue(elevation)
        for y in range(sy0, sy1):
            for x in range(sx0, sx1):
                window.on_edit_stroke_start()
                window.on_edit_stroke_tile(x, y, 0)
                window.on_edit_stroke_end()
    window._on_tool_selected("select")
    window.on_region_selected((sx0, sy0, sx1, sy1))
    window.copy_region()
    assert window._region_clipboard is not None
    return window._region_clipboard


def test_each_filter_combination_writes_only_its_own_categories() -> None:
    window = _window()
    try:
        mm = window.scenario.map_manager
        _make_region(window, 0, 0, 2, 2, _TERRAIN_A, elevation=1)
        unit_edits = window._ensure_unit_edits()
        assert isinstance(unit_edits, UnitEditModel)
        unit_edits.add(player=1, unit_const=83, x=0.5, y=0.5, z=0.0, rotation=0.0)
        window.edit_history.reset()
        window._on_tool_selected("select")
        window.on_region_selected((0, 0, 2, 2))
        window.copy_region()

        combos = [
            (True, False, False, "terrain"),
            (False, True, False, "elevation"),
            (False, False, True, "units"),
        ]
        for i, (do_t, do_e, do_u, label) in enumerate(combos):
            dx0, dy0 = 10 + i * 4, 10
            window.paste_terrain_check.setChecked(do_t)
            window.paste_elevation_check.setChecked(do_e)
            window.paste_units_check.setChecked(do_u)
            window.on_hover((dx0, dy0))
            window.paste_region()

            got_terrain = mm.get_tile(dx0, dy0).terrain_id
            got_elevation = mm.get_tile(dx0, dy0).elevation
            got_units = [
                u
                for player_units in window.scenario.unit_manager.units
                for u in player_units
                if int(u.x) == dx0 and int(u.y) == dy0
            ]
            assert (got_terrain == _TERRAIN_A) == do_t, label
            assert (got_elevation == 1) == do_e, label
            assert bool(got_units) == do_u, label
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_one_undo_step_reverts_a_terrain_elevation_units_paste() -> None:
    window = _window()
    try:
        mm = window.scenario.map_manager
        _make_region(window, 0, 0, 2, 2, _TERRAIN_A, elevation=1)
        unit_edits = window._ensure_unit_edits()
        assert isinstance(unit_edits, UnitEditModel)
        unit_edits.add(player=1, unit_const=83, x=0.5, y=0.5, z=0.0, rotation=0.0)
        window.edit_history.reset()
        window._on_tool_selected("select")
        window.on_region_selected((0, 0, 2, 2))
        window.copy_region()

        window.paste_terrain_check.setChecked(True)
        window.paste_elevation_check.setChecked(True)
        window.paste_units_check.setChecked(True)
        window.on_hover((20, 20))
        records_before = len(window.edit_history.records)
        window.paste_region()
        assert len(window.edit_history.records) == records_before + 1
        assert isinstance(window.edit_history.records[-1], CompositeDiffRecord)

        assert mm.get_tile(20, 20).terrain_id == _TERRAIN_A
        assert mm.get_tile(20, 20).elevation == 1
        pasted_units = [
            u
            for player_units in window.scenario.unit_manager.units
            for u in player_units
            if int(u.x) == 20 and int(u.y) == 20
        ]
        assert len(pasted_units) == 1

        window.undo()
        assert mm.get_tile(20, 20).terrain_id != _TERRAIN_A or mm.get_tile(20, 20).elevation != 1
        remaining = [
            u
            for player_units in window.scenario.unit_manager.units
            for u in player_units
            if int(u.x) == 20 and int(u.y) == 20
        ]
        assert remaining == []
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_all_unchecked_paste_pushes_nothing() -> None:
    window = _window()
    try:
        _make_region(window, 0, 0, 2, 2, _TERRAIN_A, elevation=0)
        window.paste_terrain_check.setChecked(False)
        window.paste_elevation_check.setChecked(False)
        window.paste_units_check.setChecked(False)
        window.on_hover((20, 20))
        records_before = len(window.edit_history.records)
        window.paste_region()
        assert len(window.edit_history.records) == records_before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_multi_owner_regions_undo_restores_every_owners_list() -> None:
    window = _window()
    try:
        unit_edits = window._ensure_unit_edits()
        assert isinstance(unit_edits, UnitEditModel)
        unit_edits.add(player=1, unit_const=83, x=0.5, y=0.5, z=0.0, rotation=0.0)
        unit_edits.add(player=2, unit_const=83, x=1.5, y=1.5, z=0.0, rotation=0.0)
        window.edit_history.reset()

        window._on_tool_selected("select")
        window.on_region_selected((0, 0, 2, 2))
        window.copy_region()
        assert len(window._region_clipboard.units) == 2
        assert {u.player for u in window._region_clipboard.units} == {1, 2}

        window.paste_terrain_check.setChecked(False)
        window.paste_elevation_check.setChecked(False)
        window.paste_units_check.setChecked(True)
        window.on_hover((30, 30))
        window.paste_region()

        def _units_at(x0, y0, x1, y1):
            return [
                u
                for player_units in window.scenario.unit_manager.units
                for u in player_units
                if x0 <= int(u.x) < x1 and y0 <= int(u.y) < y1
            ]

        pasted = _units_at(30, 30, 32, 32)
        assert {u.player for u in pasted} == {1, 2}

        window.undo()
        assert _units_at(30, 30, 32, 32) == []
        # Both owners' original placements (player 1 and player 2) must
        # still be exactly where they were -- a naive single-player
        # begin_unit_edit() would only snapshot one list.
        assert len(window.scenario.unit_manager.units[1]) == 1
        assert len(window.scenario.unit_manager.units[2]) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paste_past_map_edge_does_not_crash_the_region_overlay() -> None:
    """Regression for the 2026-09-08 crash report: pasting a region whose
    anchor+size runs off the map used to hand MapView.set_region() an
    unclamped rect (paste_region()'s own new_region), and _tile_polygon has
    no bounds check on _iso_elevations -- IndexError on the very next
    repaint. new_region must clamp to the map like the writes themselves
    already do (region_clipboard._clipped_bounds)."""
    window = _window()
    try:
        mm = window.scenario.map_manager
        _make_region(window, 0, 0, 18, 34, _TERRAIN_A, elevation=1)
        window.paste_terrain_check.setChecked(True)
        window.paste_elevation_check.setChecked(True)
        window.paste_units_check.setChecked(True)
        # Anchored so the 18x34 block runs 22 tiles past the bottom edge of
        # a 120-tall map -- the same shape as the crash report's 18x34
        # region pasted near (mm.map_height - 12).
        dx0, dy0 = mm.map_width - 5, mm.map_height - 12
        window.on_hover((dx0, dy0))
        window.paste_region()  # must not raise

        assert window._region is not None
        _, _, tx1, ty1 = window._region
        assert tx1 <= mm.map_width
        assert ty1 <= mm.map_height
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_in_terrain_mode_captures_units_with_no_live_pick_index() -> None:
    """The regression this plan's direct-walk decision exists to prevent:
    MapView._unit_index is None in Terrain mode (built lazily on entering
    Units mode), so a copy that went through unit_pick.units_in_rect would
    silently capture zero units."""
    window = _window()
    try:
        assert window.map_view._unit_index is None
        unit_edits = window._ensure_unit_edits()
        assert isinstance(unit_edits, UnitEditModel)
        unit_edits.add(player=1, unit_const=83, x=0.5, y=0.5, z=0.0, rotation=0.0)
        window.edit_history.reset()

        window.on_region_selected((0, 0, 1, 1))
        window.copy_region()
        assert window.map_view._unit_index is None  # still None -- Terrain mode never builds it
        assert len(window._region_clipboard.units) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()
