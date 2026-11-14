"""Units mode: scoped move repaints and stacked-unit click cycling with its
badges.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from unit_edit_viewer_helpers import (
    _REF_VILLAGER_P1,
    _close,
    _make_stack,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _pos_for_tile,
    _select,
    _window,
    _window_with_style,
)

from descape import render_sprites
from descape.terrain_palette import BUILDING_TILE_SPANS
from testkit.render_family import set_render_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# --- Batch D's D5: scoped repaint on Stepped/Sloped -------------------------
#
# The pixel tests in test_unit_edit_viewer_place_move.py run on Flat
# (unit_edit_viewer_helpers.py's docstring), whose _patch_unit_edit_cache()
# branch is patch_rects(), not a dirty_screen_bbox_* call -- so Move's
# Stepped/Sloped branch has no coverage from them at all.
# These pin both: pixels move correctly, AND the splice actually engaged
# (render_sprites.sprite_draws_by_anchor(), the wholesale per-level rebuild D4's
# splice exists to avoid, is never called) -- a call-counted oracle in the
# same spirit as test_invalidate_units_splice.py's own, but through the real
# ViewerWindow wiring D5 added rather than a direct cache call.


def _move_without_wholesale_rebuild(window, monkeypatch) -> None:
    """Drives the Set-field ("x") call site rather than on_unit_move(): the
    latter goes through MapView._pick_tile()'s screen-to-tile inverse, which
    is Flat-only arithmetic in these modules by design (unit_edit_viewer_helpers.py's docstring) --
    Stepped/Sloped's real inverse is test_unit_pick.py's job, not this
    file's. The inspector spin editor sets x directly, still through
    UnitEditModel.set_position() and the same _after_unit_mutation()/
    _patch_unit_edit_cache() tail on_unit_move() itself uses.

    Reads the whole canvas via render_rect(), not a single get_chunk(): a
    chunk's (cx, cy) index is a flat tile_px/chunk_px division in Flat, but
    Stepped/Sloped project a tile to an arbitrary screen diamond, so there
    is no cheap way to know which chunk a unit's sprite lands in without
    duplicating that projection here.

    _REF_VILLAGER_P1, not _REF_ARCHER_P1: the archer's own tile sits inside
    _REF_HOUSE's multi-tile footprint in this fixture, which is exactly
    _splice_eligible()'s documented shared-tile fallback (own-tile sharing
    is common enough that Batch D's D4 measured it directly, 5 of 16 corpus
    files) -- a real edit, correctly NOT spliced, but the wrong unit to
    prove the splice path with. The villager has no such neighbour."""
    (entry,) = _select(window, _REF_VILLAGER_P1)
    window.units_panel.show_unit(entry)
    old_x = entry.unit.x

    canvas_w, canvas_h = window._cache.canvas_dims(0)
    # Builds level 0 once, normally -- the edit itself must not pay for this
    # again (D4's splice deliberately never bumps _source_gen).
    before = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()

    calls = []
    real = render_sprites.sprite_draws_by_anchor
    set_render_global(monkeypatch, "sprite_draws_by_anchor", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    window.units_panel.unit_field_editors["x"].setValue(old_x + 5)
    after = window._cache.render_rect(0, 0, canvas_w, canvas_h)

    assert not np.array_equal(before, after), "moving the unit should change the composited canvas"
    assert calls == [], "a spliced unit edit should not pay for a wholesale sprite-source rebuild"


def test_stepped_move_scopes_the_repaint_and_skips_the_wholesale_rebuild(monkeypatch) -> None:
    window = _window_with_style("Stepped")
    try:
        _move_without_wholesale_rebuild(window, monkeypatch)
    finally:
        _close(window)


def test_sloped_move_scopes_the_repaint_and_skips_the_wholesale_rebuild(monkeypatch) -> None:
    window = _window_with_style("Sloped")
    try:
        _move_without_wholesale_rebuild(window, monkeypatch)
    finally:
        _close(window)


# --- Batch D's D6: scoped undo/redo of a fields_only edit -------------------
#
# D5's own tests above pin the FORWARD edit's scoped path. Undo/redo goes
# through a different method (_move_history()), which D6 gives its own
# UnitSplice-building branch -- unpinned by anything above, since none of it
# calls window.undo()/redo().


def _move_then_undo_redo_without_wholesale_rebuild(window, monkeypatch) -> None:
    (entry,) = _select(window, _REF_VILLAGER_P1)
    window.units_panel.show_unit(entry)
    old_x = entry.unit.x

    window.units_panel.unit_field_editors["x"].setValue(old_x + 5)
    moved_x = entry.unit.x
    assert moved_x != old_x

    calls = []
    real = render_sprites.sprite_draws_by_anchor
    set_render_global(monkeypatch, "sprite_draws_by_anchor", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    window.undo()
    assert entry.unit.x == old_x
    window.redo()
    assert entry.unit.x == moved_x
    assert calls == [], "undo/redo of a fields_only edit should not pay for a wholesale sprite-source rebuild"


def test_stepped_undo_redo_of_a_move_scopes_the_repaint(monkeypatch) -> None:
    window = _window_with_style("Stepped")
    try:
        _move_then_undo_redo_without_wholesale_rebuild(window, monkeypatch)
    finally:
        _close(window)


def test_sloped_undo_redo_of_a_move_scopes_the_repaint(monkeypatch) -> None:
    window = _window_with_style("Sloped")
    try:
        _move_then_undo_redo_without_wholesale_rebuild(window, monkeypatch)
    finally:
        _close(window)


def test_undo_of_a_delete_still_takes_the_wholesale_path() -> None:
    """A membership edit (Delete) never gets fields_only=True, so its own
    UnitDiffRecord carries no unit_field_entries -- undo must still fall back
    to _after_unit_mutation(None) rather than crash trying to build splices
    from a whole-list record."""
    window = _window_with_style("Stepped")
    try:
        (entry,) = _select(window, _REF_VILLAGER_P1)
        window.on_unit_delete(Qt.NoModifier)
        assert window.map_view._unit_index.entry_for_key((entry.player_id, entry.unit.reference_id)) is None

        window.undo()
        assert window.map_view._unit_index.entry_for_key((entry.player_id, entry.unit.reference_id)) is not None
    finally:
        _close(window)


# --- stacked units: badges and repeat-click cycling ------------------------


def test_stack_groups_follow_the_index_and_badges_show_the_count() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        members = window.map_view.stack_group_at(tile)
        assert [(m.player_id, m.unit.reference_id) for m in members] == keys
        item = window.map_view._stack_badge_item
        assert "3" in item.badge_texts()
        assert item.isVisible()
    finally:
        _close(window)


def test_repeat_clicks_walk_the_stack_and_wrap() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        pos = _pos_for_tile(window, *tile)
        first = window.map_view.pick_unit_at(pos)

        selected = [window.on_click_select(pos, Qt.NoModifier) for _ in range(4)]

        assert selected[0] == (first.player_id, first.unit.reference_id) == keys[0]
        assert selected == [keys[0], keys[1], keys[2], keys[0]]
        assert window._selection == [keys[0]]
        assert window._stack_cycle[0] == tile
    finally:
        _close(window)


def test_a_modified_click_resets_the_cycle() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        pos = _pos_for_tile(window, *tile)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_click_select(pos, Qt.ControlModifier)
        assert window._stack_cycle is None
        window._selection = []
        assert window.on_click_select(pos, Qt.NoModifier) == keys[0]
    finally:
        _close(window)


def test_a_selection_changed_elsewhere_restarts_the_cycle() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        pos = _pos_for_tile(window, *tile)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_marquee_select([keys[2]], Qt.NoModifier)
        assert window.on_click_select(pos, Qt.NoModifier) == keys[0]
    finally:
        _close(window)


def test_a_click_on_an_unstacked_unit_is_unchanged() -> None:
    window = _window()
    try:
        entry = next(
            e for e in window.map_view._unit_index.entries
            if window.map_view.stack_group_at((e.own_x, e.own_y)) is None
            and e.unit.unit_const not in BUILDING_TILE_SPANS
        )
        pos = _pos_for_tile(window, entry.own_x, entry.own_y)
        picked = window.map_view.pick_unit_at(pos)
        key = (picked.player_id, picked.unit.reference_id)
        assert window.on_click_select(pos, Qt.NoModifier) == key
        assert window.on_click_select(pos, Qt.NoModifier) == key
    finally:
        _close(window)


def _press_release(window, viewport_pos, tool: str) -> None:
    """A real press-release through MapView under `tool` ("select" or "move")."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    getattr(window, f"{tool}_action").setChecked(True)
    map_view = window.map_view
    for kind, button, buttons in (
        (QEvent.MouseButtonPress, Qt.LeftButton, Qt.LeftButton),
        (QEvent.MouseButtonRelease, Qt.LeftButton, Qt.NoButton),
    ):
        event = QMouseEvent(kind, viewport_pos, button, buttons, Qt.NoModifier)
        if kind == QEvent.MouseButtonPress:
            map_view.mousePressEvent(event)
        else:
            map_view.mouseReleaseEvent(event)


def test_after_cycling_with_select_move_drags_the_selected_unit_not_the_top_one() -> None:
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    window = _window()
    try:
        tile, keys = _make_stack(window)
        map_view = window.map_view
        viewport_pos = conftest.viewport_pos(map_view, *tile)
        _press_release(window, viewport_pos, "select")
        _press_release(window, viewport_pos, "select")
        assert window._selection == [keys[1]]
        window.move_action.setChecked(True)
        map_view.mousePressEvent(
            QMouseEvent(QEvent.MouseButtonPress, viewport_pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        )
        assert window._selection == [keys[1]]
        assert map_view._unit_drag_key == keys[1]
    finally:
        _close(window)


@pytest.mark.parametrize("tool", ["select", "move"])
def test_a_units_mode_press_picks_exactly_once(monkeypatch, tool) -> None:
    """mousePressEvent used to call pick_unit_at() and then hand the same
    scene pos to on_click_select(), which picks again through
    pick_unit_cover_at(). Two full unit picks per click, on a path that
    already costs a plane lookup (Sloped) or a screen_to_tile (Stepped).

    Counted at unit_pick.pick_unit_cover, the single function both routes
    bottom out in, so this stays honest if either wrapper is renamed. Both
    the hit and the miss are covered, per tool (TASK-252): Select arms the
    marquee on a hit and a miss and never a drag key; Move arms a drag key
    on a hit and nothing on a miss."""
    from PyQt5.QtCore import QEvent, QPointF
    from PyQt5.QtGui import QMouseEvent

    from descape import unit_pick

    window = _window()
    try:
        tile, _keys = _make_stack(window)
        getattr(window, f"{tool}_action").setChecked(True)
        map_view = window.map_view
        calls = []
        real = unit_pick.pick_unit_cover
        monkeypatch.setattr(
            unit_pick, "pick_unit_cover", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        )

        hit = conftest.viewport_pos(map_view, *tile)
        map_view.mousePressEvent(
            QMouseEvent(QEvent.MouseButtonPress, hit, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        )
        assert len(calls) == 1, f"a press on a unit picked {len(calls)} times, expected 1"
        assert (map_view._unit_drag_key is not None) == (tool == "move")
        assert (map_view._marquee_start_pos is not None) == (tool == "select")

        calls.clear()
        # Far off the map, so the pick genuinely misses rather than landing
        # on some other unit.
        miss = QPointF(-500.0, -500.0)
        map_view.mousePressEvent(
            QMouseEvent(QEvent.MouseButtonPress, miss, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        )
        assert len(calls) == 1, f"a press on empty ground picked {len(calls)} times, expected 1"
        assert map_view._unit_drag_key is None
        assert (map_view._marquee_start_pos is not None) == (tool == "select")
    finally:
        _close(window)


def test_a_double_click_press_still_cycles() -> None:
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    window = _window()
    try:
        tile, keys = _make_stack(window)
        map_view = window.map_view
        viewport_pos = conftest.viewport_pos(map_view, *tile)
        _press_release(window, viewport_pos, "select")
        map_view.mouseDoubleClickEvent(
            QMouseEvent(QEvent.MouseButtonDblClick, viewport_pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        )
        assert window._selection == [keys[1]]
    finally:
        _close(window)


def test_deleting_a_stack_member_updates_the_badge_and_restarts_the_cycle() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        pos = _pos_for_tile(window, *tile)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_unit_delete(Qt.NoModifier)
        assert window._stack_cycle is None
        assert len(window.map_view.stack_group_at(tile)) == 2
        assert window.on_click_select(pos, Qt.NoModifier) == keys[0]
    finally:
        _close(window)


def test_the_badge_toggle_hides_the_layer_and_persists() -> None:
    from descape import settings

    window = _window()
    try:
        _make_stack(window)
        item = window.map_view._stack_badge_item
        assert item.isVisible()
        window.stack_badges_action.setChecked(False)
        assert not item.isVisible()
        assert settings.get_stack_badges() is False
        window.stack_badges_action.setChecked(True)
        assert item.isVisible()
    finally:
        _close(window)


def test_badges_only_show_in_units_mode() -> None:
    window = _window()
    try:
        _make_stack(window)
        item = window.map_view._stack_badge_item
        window.mode_combo.setCurrentText("View")
        assert not item.isVisible()
        window.mode_combo.setCurrentText("Units")
        assert window.map_view._stack_badge_item.isVisible()
    finally:
        _close(window)
