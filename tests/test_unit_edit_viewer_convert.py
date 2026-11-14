"""Units mode: the Convert stroke."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from unit_edit_viewer_helpers import (
    _BERRY_BUSH,
    _PLACE_CONST,
    _chunk_for_tile,
    _close,
    _empty_tile,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _pos_for_tile,
    _window,
)

from descape import unit_pick
from descape.terrain_palette import BUILDING_TILE_SPANS

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# --- Convert brush: live reassignment during the drag -----------------------


def _convert_setup(window, count: int = 2):
    """Destination Player 2, brush size 1, and `count` player-1 units on
    distinct tiles whose tile holds nothing but player-1 units."""
    window.units_panel.select_owner(2)
    window.brush_size_spin.setValue(1)
    index = window.map_view._unit_index
    picked = []
    seen_tiles = set()
    for entry in index.entries:
        tile = (entry.own_x, entry.own_y)
        if entry.player_id != 1 or tile in seen_tiles:
            continue
        if entry.unit.unit_const in BUILDING_TILE_SPANS:
            continue
        bucket = [index.entries[o] for o in index.by_tile[tile]]
        if all(e.player_id == 1 for e in bucket):
            seen_tiles.add(tile)
            picked.append(entry)
        if len(picked) == count:
            break
    assert len(picked) == count
    return picked


def test_a_convert_stroke_reassigns_live_and_records_one_undo_step() -> None:
    window = _window()
    try:
        entries = _convert_setup(window)
        before_records = len(window.edit_history.records)
        units = [e.unit for e in entries]

        window._begin_convert_stroke()
        window._convert_stroke_tile(entries[0].own_x, entries[0].own_y)
        # Mutated before mouse-up, not batched to it.
        assert units[0] in window.scenario.unit_manager.units[2]
        window._convert_stroke_tile(entries[1].own_x, entries[1].own_y)
        window._end_convert_stroke()

        assert len(window.edit_history.records) == before_records + 1
        assert all(u in window.scenario.unit_manager.units[2] for u in units)
        assert window.unit_edits._pending is None

        window.undo()
        assert all(u in window.scenario.unit_manager.units[1] for u in units)
        assert len([u for u in window.scenario.unit_manager.units[2] if u in units]) == 0

        # GH #36: redo re-applies the whole stroke without adding a record.
        window.redo()
        assert all(u in window.scenario.unit_manager.units[2] for u in units)
        assert len(window.edit_history.records) == before_records + 1
    finally:
        _close(window)


def test_converting_a_gaia_only_object_to_a_player_is_refused() -> None:
    """GH #32: Convert leaves a GAIA bush with GAIA and says so."""
    window = _window()
    try:
        tile = _empty_tile(window)
        window.units_panel.select_owner(0)
        window.units_panel.select_object(_BERRY_BUSH)
        window.place_unit_action.setChecked(True)
        window.on_unit_place(_pos_for_tile(window, *tile), Qt.NoModifier)
        bush = window.scenario.unit_manager.units[0][-1]
        assert bush.unit_const == _BERRY_BUSH
        window.units_panel.select_owner(2)
        window.brush_size_spin.setValue(1)
        window.convert_source_actions[0].setChecked(True)

        before_records = len(window.edit_history.records)

        window._begin_convert_stroke()
        window._convert_stroke_tile(*tile)
        window._end_convert_stroke()

        assert bush in window.scenario.unit_manager.units[0]
        assert len(window.edit_history.records) == before_records
        status = window.status_log.toPlainText().splitlines()[-1]
        assert status == (
            "Converted 0 units to Player 2; 1 object(s) left as they were (GAIA-only or holding one)"
        )
    finally:
        _close(window)


def test_converting_a_host_holding_a_gaia_only_occupant_to_a_player_is_refused() -> None:
    """GH #32: the occupant would go with its host, so both stay; to GAIA both go."""
    from descape.viewer_common import _gaia_only_refused_note

    window = _window()
    try:
        (host_entry,) = _convert_setup(window, count=1)
        host = host_entry.unit
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "test: bad file", [1]):
            deer = model.add(1, 65, host.x, host.y, garrisoned_in_id=host.reference_id)
        window.convert_source_actions[1].setChecked(True)

        window._begin_convert_stroke()
        window._convert_stroke_tile(int(host.x), int(host.y))
        window._end_convert_stroke()

        # Another player-1 unit on the tile may convert; the host and its deer stay.
        assert host in window.scenario.unit_manager.units[1] and deer in window.scenario.unit_manager.units[1]
        assert window.status_log.toPlainText().splitlines()[-1].endswith(_gaia_only_refused_note(1))
        before_records = len(window.edit_history.records)

        window.units_panel.select_owner(0)
        window._begin_convert_stroke()
        window._convert_stroke_tile(int(host.x), int(host.y))
        window._end_convert_stroke()

        assert host in window.scenario.unit_manager.units[0] and deer in window.scenario.unit_manager.units[0]
        assert len(window.edit_history.records) == before_records + 1
    finally:
        _close(window)


def _refused_host_with_occupants(window):
    """A player-1 host holding a GAIA-only deer plus two player-1 villagers:
    one on the host's tile, drawn ahead of it in by_tile, and one stored on
    an empty tile of its own. Returns (host, deer, on_tile, off_tile, far)."""
    (host_entry,) = _convert_setup(window, count=1)
    host = host_entry.unit
    far = _empty_tile(window)
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "test: bad file", [1]):
        deer = model.add(1, 65, host.x, host.y, garrisoned_in_id=host.reference_id)
        on_tile = model.add(1, _PLACE_CONST, host.x, host.y, garrisoned_in_id=host.reference_id)
        off_tile = model.add(1, _PLACE_CONST, far[0] + 0.5, far[1] + 0.5, garrisoned_in_id=host.reference_id)
    index = window.map_view._unit_index
    index.by_tile[(int(host.x), int(host.y))].sort(key=lambda o: index.entries[o].unit is not on_tile)
    return host, deer, on_tile, off_tile, far


def test_an_occupant_reached_before_its_refused_host_stays_with_it() -> None:
    """TASK-474: GH #32 refuses the host, so no occupant converts ahead of it,
    whether it comes first in the host's own bucket or on an earlier tile."""
    window = _window()
    try:
        host, deer, on_tile, off_tile, far = _refused_host_with_occupants(window)
        window.convert_source_actions[1].setChecked(True)

        window._begin_convert_stroke()
        window._convert_stroke_tile(*far)
        window._convert_stroke_tile(int(host.x), int(host.y))
        window._end_convert_stroke()

        player_1 = window.scenario.unit_manager.units[1]
        assert all(u in player_1 for u in (host, deer, on_tile, off_tile))
    finally:
        _close(window)


def test_an_inspector_owner_change_keeps_an_occupant_selected_with_its_refused_host() -> None:
    """TASK-474: the occupant first in the selection, its GAIA-only-holding host second."""
    window = _window()
    try:
        host, deer, on_tile, _off_tile, _far = _refused_host_with_occupants(window)
        window._selection = [unit_pick.unit_key(1, on_tile), unit_pick.unit_key(1, host)]
        window._update_unit_inspector_from_selection()
        before_records = len(window.edit_history.records)

        combo = window.units_panel.unit_field_editors["player"]
        combo.setCurrentIndex(combo.findData(2))

        player_1 = window.scenario.unit_manager.units[1]
        assert all(u in player_1 for u in (host, deer, on_tile))
        assert len(window.edit_history.records) == before_records
        status = window.status_log.toPlainText().splitlines()[-1]
        assert status == "Not reassigned to Player 2; 1 object(s) left as they were (GAIA-only or holding one)"
    finally:
        _close(window)


def test_an_unticked_source_owner_keeps_its_units() -> None:
    window = _window()
    try:
        entries = _convert_setup(window)
        window.convert_source_actions[1].setChecked(False)
        before_records = len(window.edit_history.records)
        units = [e.unit for e in entries]

        window._begin_convert_stroke()
        for entry in entries:
            window._convert_stroke_tile(entry.own_x, entry.own_y)
        window._end_convert_stroke()

        assert all(u in window.scenario.unit_manager.units[1] for u in units)
        assert len(window.edit_history.records) == before_records
        assert "Converted" not in window.status_log.toPlainText()
    finally:
        _close(window)


def test_convert_pixels_change_before_the_stroke_ends() -> None:
    window = _window()
    try:
        (entry,) = _convert_setup(window, count=1)
        cx, cy = _chunk_for_tile(window, entry.own_x, entry.own_y)
        before = window._cache.get_chunk(0, cx, cy).copy()

        window._begin_convert_stroke()
        window._convert_stroke_tile(entry.own_x, entry.own_y)
        assert window._unit_stroke_refresh_timer.isActive()
        window._unit_stroke_refresh_timer.stop()
        window._flush_unit_stroke_refresh()
        mid = window._cache.get_chunk(0, cx, cy).copy()
        assert window._convert_model is not None  # still mid-stroke
        window._end_convert_stroke()

        assert not np.array_equal(before, mid)
    finally:
        _close(window)


def test_convert_coalesces_the_repaint_across_touched_tiles(monkeypatch) -> None:
    window = _window()
    try:
        entries = _convert_setup(window, count=2)
        calls = []
        original = window._after_unit_mutation

        def spy(*args, **kwargs):
            calls.append(kwargs.get("defer_index", False))
            return original(*args, **kwargs)

        monkeypatch.setattr(window, "_after_unit_mutation", spy)
        window._begin_convert_stroke()
        for entry in entries:
            window._convert_stroke_tile(entry.own_x, entry.own_y)
        assert calls == []  # no per-tile refresh; the timer hasn't fired
        window._end_convert_stroke()
        assert calls == [False]
        assert not window._unit_stroke_refresh_timer.isActive()
    finally:
        _close(window)


def test_a_convert_stroke_that_touches_nothing_records_nothing() -> None:
    window = _window()
    try:
        before_records = len(window.edit_history.records)
        tile = _empty_tile(window)
        window.units_panel.select_owner(2)
        window._begin_convert_stroke()
        window._convert_stroke_tile(*tile)
        window._end_convert_stroke()
        assert len(window.edit_history.records) == before_records
        assert window.unit_edits._pending is None
    finally:
        _close(window)


def test_a_stroke_left_open_is_aborted_by_the_next_one() -> None:
    window = _window()
    try:
        (entry,) = _convert_setup(window, count=1)
        window._begin_convert_stroke()
        window._begin_convert_stroke()
        window._convert_stroke_tile(entry.own_x, entry.own_y)
        window._end_convert_stroke()
        assert window.unit_edits._pending is None
        assert entry.unit in window.scenario.unit_manager.units[2]
    finally:
        _close(window)


def test_a_stroke_left_open_with_a_reassign_is_committed_by_the_next_one() -> None:
    """GH #170 review: its reassign is already live, so the next press records
    it (one undo step that reverts it) rather than dropping the record."""
    window = _window()
    try:
        (entry,) = _convert_setup(window, count=1)
        records = len(window.edit_history.records)
        window._begin_convert_stroke()
        window._convert_stroke_tile(entry.own_x, entry.own_y)
        assert entry.unit in window.scenario.unit_manager.units[2]
        window._begin_convert_stroke()
        assert len(window.edit_history.records) == records + 1
        # The pick may carry a garrison along (GH #115), so the count is not pinned.
        assert window.edit_history.records[-1].label.startswith("Convert ")
        assert window._unit_stroke_pending_splices == []
        window._end_convert_stroke()
        assert window.unit_edits._pending is None
        window.undo()
        assert entry.unit in window.scenario.unit_manager.units[1]
    finally:
        _close(window)


def test_convert_flushes_splices_and_the_stroke_end_rebuilds_the_pick_index(monkeypatch) -> None:
    """Mid-stroke flushes splice the converted units rather than taking the
    wholesale path, and the stroke end still rebuilds the index Convert left
    stale all stroke, so picks report the new owner."""
    window = _window()
    try:
        entries = _convert_setup(window, count=2)
        # The second pick is the House's villager, which the first tile's House
        # would carry along (GH #115). Unloaded so each flush owns one unit.
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Unload", [entries[1].player_id]):
            model.set_garrisoned_in(entries[1].unit, -1)
        calls = []
        original = window._after_unit_mutation

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(window, "_after_unit_mutation", spy)
        window._begin_convert_stroke()
        window._convert_stroke_tile(entries[0].own_x, entries[0].own_y)
        window._unit_stroke_refresh_timer.stop()
        window._flush_unit_stroke_refresh()
        window._convert_stroke_tile(entries[1].own_x, entries[1].own_y)
        window._end_convert_stroke()

        assert [c.get("defer_index", False) for c in calls] == [True, False]
        flushed, final = calls[0]["changed"], calls[1]["changed"]
        # A touched tile can hold more than the picked entry (a building over it).
        flushed_ids, final_ids = {id(s.unit) for s in flushed}, {id(s.unit) for s in final}
        assert id(entries[0].unit) in flushed_ids and id(entries[1].unit) in final_ids
        assert not flushed_ids & final_ids, "a unit was spliced twice"
        for splice in (*flushed, *final):
            assert (splice.player_id, splice.old_player_id) == (2, 1)
            assert window.scenario.unit_manager.units[2][splice.index] is splice.unit
        assert calls[1].get("rebuild_index") is True

        index = window.map_view._unit_index
        converted = {id(e.unit) for e in entries}
        owners = {entry.player_id for entry in index.entries if id(entry.unit) in converted}
        assert owners == {2}
    finally:
        _close(window)
