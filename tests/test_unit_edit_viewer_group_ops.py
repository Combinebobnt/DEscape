"""Units mode: group delete and nudge, and marquee selection."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtWidgets import QMessageBox
from unit_edit_viewer_helpers import (
    _REF_ARCHER_P1,
    _REF_ARCHER_P2,
    _REF_HOUSE,
    _REF_TREE_OAK,
    _REF_TREE_PINE,
    _REF_VILLAGER_P1,
    _REF_VILLAGER_P2,
    _chunk_for_tile,
    _close,
    _hide_garrisoned,
    _last_status_lines,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _orphan_reference_id,
    _select,
    _select_keys,
    _unit_count,
    _unit_layout,
    _warnings,
    _window,
    _window_with_style,
)

from descape import render_paint, render_units

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _rect_for_tile(window, tile_x: int, tile_y: int) -> QRectF:
    """A scene rect covering exactly one tile -- Flat's own tile_px grid,
    same arithmetic _pos_for_tile uses for a single point."""
    tp = window.map_view._tile_pixels
    return QRectF(tile_x * tp, tile_y * tp, tp, tp)


# --- b2: bulk operations (marquee, group move/delete, Convert) -------------
#
# The N-way selection highlight (MapView._draw_unit_selection unioning
# multiple entries into one QPainterPath) is new scene-item code with the
# same exposure b1's own postmortem flagged (test_unit_edit_viewer_place_move.py's
# pixel section): every model-level assertion in the earlier tests would pass unchanged even if the
# cache invalidation for a GROUP edit were broken, since none of them read
# pixels for more than one unit at a time. The delete test below does, at
# two DIFFERENT chunks, so a regression that invalidated only one tile (or
# only the first selected unit) would fail it.


def test_group_delete_removes_every_selected_unit_and_repaints_both_tiles() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[6]
        assert (int(a.unit.x), int(a.unit.y)) != (int(b.unit.x), int(b.unit.y))
        cx_a, cy_a = _chunk_for_tile(window, int(a.unit.x), int(a.unit.y))
        cx_b, cy_b = _chunk_for_tile(window, int(b.unit.x), int(b.unit.y))
        assert (cx_a, cy_a) != (cx_b, cy_b), "fixture must give two units in different chunks or this proves nothing"

        window._selection = [(a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)]
        window.map_view.set_unit_selection([a, b])
        before_a = window._cache.get_chunk(0, cx_a, cy_a).copy()
        before_b = window._cache.get_chunk(0, cx_b, cy_b).copy()
        before_count = _unit_count(window)

        window.on_unit_delete(Qt.NoModifier)

        assert _unit_count(window) == before_count - 2
        assert not np.array_equal(before_a, window._cache.get_chunk(0, cx_a, cy_a))
        assert not np.array_equal(before_b, window._cache.get_chunk(0, cx_b, cy_b))
        assert window._selection == []
        assert window.edit_history.can_undo
        # One record for the whole group, not one per unit.
        assert window.edit_history.peek_undo().kind == "unit"
        window.undo()
        assert _unit_count(window) == before_count
    finally:
        _close(window)


def _refuse_single_remove(window, monkeypatch) -> None:
    """Makes the per-unit remove() fail loudly, so a group delete that slid
    back to looping it can't pass by accident."""
    model = window._ensure_unit_edits()

    def _no_loop(unit):
        raise AssertionError("a 2+ group delete must go through remove_many(), not remove()")

    monkeypatch.setattr(model, "remove", _no_loop)


def test_group_delete_across_three_players_is_one_record_and_undo_restores_exact_indices(monkeypatch) -> None:
    window = _window()
    try:
        _refuse_single_remove(window, monkeypatch)
        # Each target sits mid-list or at the head, so a survivor behind it shifts down.
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2, _REF_TREE_PINE)
        assert len({e.player_id for e in entries}) == 3
        survivors = {
            id(u) for units in window.scenario.unit_manager.units for u in units
        } - {id(e.unit) for e in entries}
        layout_before = _unit_layout(window)
        records_before = len(window.edit_history.records)

        window.on_unit_delete(Qt.NoModifier)

        after = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert after == survivors
        assert len(window.edit_history.records) == records_before + 1
        assert window.edit_history.peek_undo().kind == "unit"
        assert window._selection == []
        for e in entries:
            assert window.map_view._unit_index.entry_for_key((e.player_id, e.unit.reference_id)) is None

        window.undo()
        assert _unit_layout(window) == layout_before
        for e in entries:
            assert window.map_view._unit_index.entry_for_key((e.player_id, e.unit.reference_id)) is not None

        window.redo()
        after_redo = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert after_redo == survivors
    finally:
        _close(window)


def test_a_group_delete_cascades_every_hosts_garrison_in_one_record(monkeypatch) -> None:
    """The group counterpart of the cascade: this file used to assert a
    partial refusal here (GH #42 removed the refusal), so what it now pins is
    that an unselected occupant of a selected host goes in the same record,
    through remove_many() rather than a remove() loop."""
    window = _window()
    try:
        _refuse_single_remove(window, monkeypatch)
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1, _REF_ARCHER_P2)
        host = next(e for e in entries if e.unit.reference_id == _REF_ARCHER_P2)
        garrisoned = next(u for u in window.scenario.unit_manager.units[2] if u.reference_id == _REF_VILLAGER_P2)
        garrisoned.garrisoned_in_id = _REF_ARCHER_P2
        layout_before = _unit_layout(window)
        records_before = len(window.edit_history.records)
        before = _unit_count(window)

        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
        window.on_unit_delete(Qt.NoModifier)

        assert not warned
        assert _unit_count(window) == before - 4  # three selected, plus the occupant
        live = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert id(garrisoned) not in live
        assert not any(id(e.unit) in live for e in entries)
        assert id(host.unit) not in live
        assert len(window.edit_history.records) == records_before + 1
        assert "1 garrisoned" in window.status_log.toPlainText()

        window.undo()
        assert _unit_layout(window) == layout_before
    finally:
        _close(window)


def test_delete_labels_and_status_lines_for_one_unit_and_for_a_group() -> None:
    from descape.viewer_common import _unit_name

    window = _window()
    try:
        (entry,) = _select(window, _REF_ARCHER_P1)
        name = _unit_name(entry.unit.unit_const)
        window.on_unit_delete(Qt.NoModifier)
        assert window.edit_history.peek_undo().label == "Delete unit"
        assert _last_status_lines(window, 2) == ["Delete unit", f"Deleted {name}"]

        _select(window, _REF_ARCHER_P2, _REF_TREE_PINE)
        window.on_unit_delete(Qt.NoModifier)
        assert window.edit_history.peek_undo().label == "Delete 2 units"
        assert _last_status_lines(window, 2) == ["Delete 2 units", "Deleted 2 units"]
    finally:
        _close(window)


def test_delete_status_names_the_host_and_counts_its_garrison() -> None:
    from descape.viewer_common import _unit_name

    window = _window()
    try:
        _hide_garrisoned(window)
        (entry,) = _select(window, _REF_HOUSE)
        name = _unit_name(entry.unit.unit_const)
        window.on_unit_delete(Qt.NoModifier)
        assert window.edit_history.peek_undo().label == "Delete 2 units"
        assert _last_status_lines(window, 2) == ["Delete 2 units", f"Deleted {name} (1 garrisoned)"]
    finally:
        _close(window)


def test_a_wholly_refused_delete_warns_by_count() -> None:
    window = _window()
    try:
        warned = _warnings(window)
        house_key = _orphan_reference_id(window, 1, _REF_HOUSE)
        records_before = len(window.edit_history.records)

        _select_keys(window, [house_key])
        window.on_unit_delete(Qt.NoModifier)
        assert _last_status_lines(window, 1) == ["Delete failed: referenced by a garrison"]
        assert warned == [("Cannot delete unit", "Referenced by another unit's garrison.")]

        archer_key = _orphan_reference_id(window, 2, _REF_ARCHER_P2)
        _select_keys(window, [house_key, archer_key])
        window.on_unit_delete(Qt.NoModifier)
        assert _last_status_lines(window, 1) == ["Delete failed: referenced by a garrison"]
        assert warned[1:] == [
            ("Cannot delete units", "All selected units are referenced by another unit's garrison.")
        ]
        assert len(window.edit_history.records) == records_before
    finally:
        _close(window)


def test_a_partly_refused_delete_says_how_many_it_left() -> None:
    window = _window()
    try:
        warned = _warnings(window)
        house_key = _orphan_reference_id(window, 1, _REF_HOUSE)
        before = _unit_count(window)

        _select_keys(window, [house_key, (1, _REF_ARCHER_P1)])
        window.on_unit_delete(Qt.NoModifier)

        assert _unit_count(window) == before - 1
        assert window.edit_history.peek_undo().label == "Delete unit"
        assert _last_status_lines(window, 2) == ["Delete unit", "Deleted 1; 1 refused (garrison reference)"]
        assert warned == [
            (
                "Some units not deleted",
                "1 unit(s) are referenced by another unit's garrison and were not deleted.",
            )
        ]
    finally:
        _close(window)


def test_group_nudge_moves_every_selected_unit_in_one_undo_record() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[6]
        ax, ay = a.unit.x, a.unit.y
        bx, by = b.unit.x, b.unit.y
        window._selection = [(a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)]
        records_before = len(window.edit_history.records)

        window.on_unit_nudge(1, 0, Qt.NoModifier)

        assert len(window.edit_history.records) == records_before + 1
        assert (a.unit.x, a.unit.y) == (ax + 0.1, ay)
        assert (b.unit.x, b.unit.y) == (bx + 0.1, by)
    finally:
        _close(window)


def test_a_flat_group_move_across_two_blocks_patches_one_rect_per_block(monkeypatch) -> None:
    """Four units in two 16-tile blocks moved one tile in one edit: top-down
    Flat hands patch_rects() one rect per block, not one per touched tile,
    patches the warmed chunks in place, and the canvas equals a fresh cache."""
    from descape.render_cache import FlatChunkCache

    window = _window()
    try:
        cache = window._cache
        assert isinstance(cache, FlatChunkCache)
        entries = _select(window, _REF_TREE_OAK, _REF_TREE_PINE, _REF_ARCHER_P2, _REF_VILLAGER_P2)
        block = render_paint.FLAT_PATCH_BLOCK_TILES
        assert len({(int(e.unit.x) // block, int(e.unit.y) // block) for e in entries}) == 2
        canvas = (0, 0, *cache.canvas_dims(0))
        before = cache.render_rect(*canvas, mip=0).copy()
        warm = {key: id(chunk) for key, chunk in cache._cache.items()}
        handed = []
        real_patch = cache.patch_rects

        def patch_rects(rects):
            handed.append(list(rects))
            return real_patch(rects)

        monkeypatch.setattr(cache, "patch_rects", patch_rects)

        assert window.on_unit_nudge(0, 1, Qt.ShiftModifier)

        (rects,) = handed
        tp = cache.tile_px
        assert rects == [(5 * tp, 5 * tp, 7 * tp, 7 * tp), (20 * tp, 20 * tp, 22 * tp, 22 * tp)], rects
        assert {key: id(chunk) for key, chunk in cache._cache.items()} == warm, "the unit patch evicted a chunk"
        after = cache.render_rect(*canvas, mip=0).copy()
        assert not np.array_equal(after, before), "the move changed no pixel, so the check below proves nothing"
        fresh = FlatChunkCache(
            window.scenario, tp, sprites=cache.sprites_enabled, unit_filter=cache.unit_filter, layers=cache.layers
        )
        assert np.array_equal(after, fresh.render_rect(*canvas, mip=0)), "the patched canvas differs from a fresh render"

        window.undo()
        assert np.array_equal(cache.render_rect(*canvas, mip=0), before), "the undo left a stale tile"
    finally:
        _close(window)


def test_marquee_select_plain_replaces_the_selection() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[3]
        window._selection = [(99, 99)]  # a stale key that must be discarded, not merged

        window.on_marquee_select([(a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)], Qt.NoModifier)

        assert set(window._selection) == {(a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)}
    finally:
        _close(window)


def test_marquee_select_ctrl_toggles_membership() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[3]
        key_a, key_b = (a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)
        window._selection = [key_a]

        # A second marquee covering both A (already selected) and B toggles
        # each independently: A drops out, B is added.
        window.on_marquee_select([key_a, key_b], Qt.ControlModifier)

        assert set(window._selection) == {key_b}
    finally:
        _close(window)


def test_marquee_select_shift_only_adds() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[3]
        key_a, key_b = (a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)
        window._selection = [key_a]

        window.on_marquee_select([key_a, key_b], Qt.ShiftModifier)

        assert set(window._selection) == {key_a, key_b}
    finally:
        _close(window)


def test_marquee_over_a_filtered_out_player_selects_nothing() -> None:
    """The marquee's own filter-respecting requirement (plan hard rule: a
    drag across a forest must not grab every hidden tree) -- exercised here
    against a hidden PLAYER rather than GAIA/trees, since this fixture's
    units are placed by player, not by tree-const. GAIA player 0's units
    sit at tiles (5,5)-(7,5); hiding GAIA and marqueeing exactly that strip
    must return nothing, even though units_in_rect's own tile math would
    otherwise find them."""
    window = _window()
    try:
        window.show_gaia_action.setChecked(False)
        tile_px = window.map_view._tile_pixels
        rect = QRectF(5 * tile_px, 5 * tile_px, 3 * tile_px, tile_px)
        keys = window.map_view._units_in_scene_rect(rect)
        assert keys == []
    finally:
        _close(window)


def test_units_in_scene_rect_finds_units_under_the_marquee() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        entry = index.entries[0]
        rect = _rect_for_tile(window, int(entry.unit.x), int(entry.unit.y))
        keys = window.map_view._units_in_scene_rect(rect)
        assert (entry.player_id, entry.unit.reference_id) in keys
    finally:
        _close(window)


_REF_HOUSE_P1 = 200


def _old_marquee_tile_bbox(map_view, rect: QRectF):
    """The pre-#114 iso marquee's tile bbox: _pick_tile sampled on a tile_px
    grid across rect."""
    step = map_view._tile_pixels
    left, top, right, bottom = int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())
    tiles = [
        tile
        for y in [*range(top, bottom, step), bottom]
        for x in [*range(left, right, step), right]
        if (tile := map_view._pick_tile(QPointF(x, y))) is not None
    ]
    txs, tys = [t[0] for t in tiles], [t[1] for t in tiles]
    return min(txs), min(tys), max(txs) + 1, max(tys) + 1


@pytest.mark.parametrize(
    ("style", "map_view_style"),
    [("Stepped", "stepped"), ("Sloped", "sloped"), ("Flat", "stepped")],
    ids=["stepped", "sloped", "flat-isometric"],
)
def test_iso_marquee_selects_by_on_screen_anchor(style: str, map_view_style: str) -> None:
    """GH #114. Flat keeps Isometric View checked here, so it takes the
    Stepped path. A small rect around the house's footprint centre returns
    exactly it; a wide strip well below every unit returns nothing, though
    the old tile-bbox rule's diamond reached up over the units."""
    from descape import iso_geometry, unit_pick

    window = _window_with_style(style)
    try:
        map_view = window.map_view
        assert map_view._terrain_style == map_view_style
        assert window.iso_action.isChecked()
        assert not map_view._iso_elevations.any(), "fixture is flat, so every anchor has rise 0"
        index = map_view._unit_index
        proj = map_view._iso_proj
        house = next(e for e in index.entries if e.unit.reference_id == _REF_HOUSE_P1)
        tx0, tx1, ty0, ty1 = render_units.unit_tile_bounds(house.unit, map_view._map_width, map_view._map_height)
        ax, ay = iso_geometry.map_point_to_screen((tx0 + tx1) / 2, (ty0 + ty1) / 2, 0, proj)

        keys = map_view._units_in_scene_rect(QRectF(ax - 4, ay - 4, 8, 8))
        assert keys == [(house.player_id, _REF_HOUSE_P1)]

        strip = QRectF(300, ay + 160, 1100, 4)
        assert unit_pick.units_in_rect(index, *_old_marquee_tile_bbox(map_view, strip)), (
            "the old rule's diamond must reach a unit here or this proves nothing"
        )
        assert map_view._units_in_scene_rect(strip) == []
    finally:
        _close(window)
