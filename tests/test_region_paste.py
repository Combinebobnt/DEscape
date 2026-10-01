"""Paste Region's filter checkboxes and one-undo-step composite behaviour --
phase 2.8. Same offscreen technique tests/test_fill_tool.py documents; every
ViewerWindow() here must call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import pytest

from descape.edit_history import CompositeDiffRecord
from descape.unit_model import UnitEditModel

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN_A, _TERRAIN_B = 2, 15  # BEACH, GRASS_1 -- present in every DE version


def _window():
    """The blank template loaded, Terrain mode, Select active. Caller must
    edit_history.mark_saved() + close()."""
    window = conftest.terrain_edit_window()
    window._on_tool_selected("select")
    return window


def _make_region(window, sx0: int, sy0: int, sx1: int, sy1: int, terrain_id: int, elevation: int):
    """Builds a region [sx0,sx1)x[sy0,sy1) with a distinct terrain_id and a
    legal elevation (via the real Set Elevation stroke, not a hand-assigned
    value -- see tests/test_region_clipboard.py's own fixture-construction
    comment on why), then copies it. Returns the RegionBlock."""
    window._on_tool_selected("draw")
    window.terrain_panel.set_terrain(terrain_id)
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


def test_paste_remaps_garrison_link_to_the_pasted_holders_new_id() -> None:
    window = _window()
    try:
        unit_edits = window._ensure_unit_edits()
        assert isinstance(unit_edits, UnitEditModel)
        holder = unit_edits.add(player=1, unit_const=79, x=0.5, y=0.5, z=0.0, rotation=0.0)
        unit_edits.add(
            player=1, unit_const=4, x=0.5, y=0.5, z=0.0, rotation=0.0, garrisoned_in_id=holder.reference_id
        )
        window.edit_history.reset()

        window._on_tool_selected("select")
        window.on_region_selected((0, 0, 1, 1))
        window.copy_region()

        window.paste_terrain_check.setChecked(False)
        window.paste_elevation_check.setChecked(False)
        window.paste_units_check.setChecked(True)
        window.on_hover((30, 30))
        window.paste_region()

        pasted = [
            u
            for player_units in window.scenario.unit_manager.units
            for u in player_units
            if int(u.x) == 30 and int(u.y) == 30
        ]
        pasted_holder = next(u for u in pasted if u.unit_const == 79)
        pasted_occupant = next(u for u in pasted if u.unit_const == 4)
        assert pasted_occupant.garrisoned_in_id == pasted_holder.reference_id
        assert pasted_occupant.garrisoned_in_id != -1
        assert pasted_occupant.garrisoned_in_id != holder.reference_id
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_garrison_remap_leaves_the_reverse_map_spliced_after_undo() -> None:
    """The discriminating regression test: a post-hoc `set_garrisoned_in_id()`
    patch pass (the design the plan explicitly rejects) would splice a unit's
    garrison link without going through add()'s own warm-map maintenance, so
    a warm _garrison would drift. serialize() and restore() both run
    _check_alignment() -> _check_derived(), which raises RuntimeError on a
    drifted map -- this must pass clean."""
    window = _window()
    try:
        unit_edits = window._ensure_unit_edits()
        assert isinstance(unit_edits, UnitEditModel)
        holder = unit_edits.add(player=1, unit_const=79, x=0.5, y=0.5, z=0.0, rotation=0.0)
        unit_edits.add(
            player=1, unit_const=4, x=0.5, y=0.5, z=0.0, rotation=0.0, garrisoned_in_id=holder.reference_id
        )
        window.edit_history.reset()

        window._on_tool_selected("select")
        window.on_region_selected((0, 0, 1, 1))
        window.copy_region()

        window.paste_terrain_check.setChecked(False)
        window.paste_elevation_check.setChecked(False)
        window.paste_units_check.setChecked(True)
        window.on_hover((30, 30))

        unit_edits.warm_garrison_map()
        window.paste_region()
        window.undo()
        unit_edits.serialize()  # must not raise
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


_BLOCK_LEVEL = 3  # >= 2, or a paste onto flat ground propagates no skirt at all


def _skirt_window(anchor: tuple[int, int]):
    """A 3x3 level-3 block copied from (0, 0), and uneven ground east of
    `anchor`: a level-3 tile two columns past the block's middle row, so the
    tile between them takes elevation_tools' `behind` branch while the rest
    of the ring takes the `> 1` branch. Returns (window, rect, between)."""
    window = _window()
    _make_region(window, 0, 0, 3, 3, _TERRAIN_A, elevation=_BLOCK_LEVEL)
    ax, ay = anchor
    window._on_tool_selected("set_level")
    window.elevation_level_spin.setValue(_BLOCK_LEVEL)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(ax + 4, ay + 1, 0)
    window.on_edit_stroke_end()
    window._on_tool_selected("select")
    window.edit_history.reset()
    window.paste_terrain_check.setChecked(True)
    window.paste_elevation_check.setChecked(True)
    window.paste_units_check.setChecked(False)
    window.on_hover(anchor)
    mm = window.scenario.map_manager
    w = mm.map_width
    rect = {y * w + x for y in range(max(0, ay), min(mm.map_height, ay + 3)) for x in range(max(0, ax), min(w, ax + 3))}
    between = (ay + 1) * w + ax + 3
    assert mm.terrain[between].elevation == _BLOCK_LEVEL - 1  # lower than the plateau behind it
    return window, rect, between


def _paste_record_guard(window):
    """Pastes once and asserts the paste's tile record equals a full-map diff
    against the pre-paste state. Returns the record's changes."""
    from descape.edit_history import tile_state

    mm = window.scenario.map_manager
    start = [tile_state(t) for t in mm.terrain]
    history = window.edit_history
    real_build = history.build_stroke_record
    built = []

    def build(*args, **kwargs):
        built.append(real_build(*args, **kwargs))
        return built[-1]

    history.build_stroke_record = build
    try:
        window.paste_region()
    finally:
        del history.build_stroke_record
    oracle = [(i, start[i], tile_state(t)) for i, t in enumerate(mm.terrain) if tile_state(t) != start[i]]
    (record,) = built
    got = record.changes if record is not None else []
    assert got == oracle, f"missing {sorted({c[0] for c in oracle} - {c[0] for c in got})[:8]}"
    return got


@pytest.mark.parametrize("anchor", [(30, 30), (-1, 50)], ids=["interior", "clipped"])
def test_a_paste_whose_skirt_leaves_the_block_records_the_full_map_diff(anchor) -> None:
    window, rect, between = _skirt_window(anchor)
    try:
        changes = _paste_record_guard(window)
        recorded = {i for i, _old, new in changes}
        assert recorded - rect - {between}, "no skirt outside the block; the guard is vacuous"
        assert between in recorded, "the `behind` branch never wrote; its capture is untested"
        assert window.scenario.map_manager.terrain[between].elevation == _BLOCK_LEVEL
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_paste_record_guard_catches_an_uncaptured_skirt(monkeypatch) -> None:
    """Control: a set_tiles_elevation that writes the skirt without reporting
    it through before_write must fail the guard."""
    import descape.viewer as viewer_module

    real = viewer_module.set_tiles_elevation
    window, _rect, _between = _skirt_window((30, 30))
    monkeypatch.setattr(viewer_module, "set_tiles_elevation", lambda mm, targets, before_write=None: real(mm, targets))
    try:
        with pytest.raises(AssertionError, match="missing"):
            _paste_record_guard(window)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_paste_raising_mid_skirt_still_records_every_tile_it_wrote(monkeypatch) -> None:
    """The raise lands inside the skirt, after some of it is written. The
    stroke closes through _close_stroke_on_error, whose unscoped-looking
    commit diffs the scoped captures: terrain, targets and the written part
    of the skirt."""
    import descape.viewer as viewer_module
    from descape.edit_history import TileDiffRecord, tile_state

    real = viewer_module.set_tiles_elevation
    captured: list[int] = []

    def raising(mm, targets, before_write=None):
        def capture(i):
            if len(captured) == 4:
                raise RuntimeError("boom mid-skirt")
            before_write(i)
            captured.append(i)

        return real(mm, targets, before_write=capture)

    window, rect, _between = _skirt_window((30, 30))
    monkeypatch.setattr(viewer_module, "set_tiles_elevation", raising)
    try:
        mm = window.scenario.map_manager
        start = [tile_state(t) for t in mm.terrain]
        with pytest.raises(RuntimeError, match="boom"):
            window.paste_region()
        assert not window.edit_history.in_stroke
        oracle = [(i, start[i], tile_state(t)) for i, t in enumerate(mm.terrain) if tile_state(t) != start[i]]
        assert len(captured) == 4 and {i for i, _o, _n in oracle} - rect, "the raise came before any skirt write"
        record = window.edit_history.records[-1]
        assert isinstance(record, TileDiffRecord)
        assert record.changes == oracle
    finally:
        window.edit_history.mark_saved()
        window.close()
