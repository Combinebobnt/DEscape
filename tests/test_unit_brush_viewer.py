"""The Unit Brush tool in the viewer (GH #170), offscreen: routing, the one
undo record per drag, the rules and Free placement, rotation, the widgets,
the refused picks, the status line, the write path and the hover ghost (GH #270).

Every ViewerWindow() here closes through _close(), which marks the
document saved first (tests/test_fill_tool.py's module docstring)."""

from __future__ import annotations

import pytest

from descape import unit_rotation, unit_variant
from descape.object_catalog import display_name
from descape.place_rotation import PlaceRotation, rotation_for
from descape.render import occupied_tiles_for

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

OAK = 349  # GAIA-only, cyclable
DEER = 65  # GAIA-only, ANGLE
VILLAGER = 83  # ANGLE, mobile
HOUSE = 70  # 2x2
BARRACKS = 12  # 3x3
STONE_WALL = 117
STONE_GATE = 64
WATER_DEEP = 22
P1, GAIA = 1, 0
TILE_CENTRE = (0.5, 0.5)
# Low in the tile, where Place Unit's even-span snap rounds down and the brush's does not.
TILE_LOW = (0.2, 0.2)


def _window(const: int = OAK, owner: int = GAIA, size: int = 3, shown: bool = False):
    """The blank 120x120 template in Units mode, Flat, with Unit Brush checked."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.blank_window()
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    assert window.units_panel.select_owner(owner)
    assert window.units_panel.pick_object(const)
    window.unit_brush_action.setChecked(True)
    assert window._current_tool == "unit_brush"
    window.brush_size_spin.setValue(size)
    if shown:
        window.show()
        QApplication.processEvents()
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _stroke(window, path, button=None, modifiers=None) -> None:
    """A press-drag-release through the viewer's stroke trio, as MapView calls it."""
    from PyQt5.QtCore import Qt

    window.map_view._stroke_button = Qt.LeftButton if button is None else button
    window.on_edit_stroke_start()
    for tile in path:
        window.on_edit_stroke_tiles([tile], Qt.NoModifier if modifiers is None else modifiers)
    window.on_edit_stroke_end()
    window.map_view._stroke_button = None


def _view_pos(window, tile, at=TILE_CENTRE):
    """The viewport point `at` inside a Flat tile."""
    from PyQt5.QtCore import QPointF

    view = window.map_view
    tile_px = view._tile_pixels
    return QPointF(view.mapFromScene(QPointF((tile[0] + at[0]) * tile_px, (tile[1] + at[1]) * tile_px)))


def _hover(window, tile, at=TILE_CENTRE) -> None:
    """A real mouse move with no button down, through MapView (_stroke() bypasses it)."""
    from PyQt5.QtCore import QEvent, Qt

    event = conftest.mouse_event(QEvent.MouseMove, _view_pos(window, tile, at), Qt.NoButton, Qt.NoButton)
    window.map_view.mouseMoveEvent(event)


def _press(window, tile, button=None, at=TILE_CENTRE) -> None:
    from PyQt5.QtCore import QEvent, Qt

    button = Qt.LeftButton if button is None else button
    event = conftest.mouse_event(QEvent.MouseButtonPress, _view_pos(window, tile, at), button, button)
    window.map_view.mousePressEvent(event)


def _held_move(window, tile, button=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    buttons = Qt.LeftButton if button is None else button
    event = conftest.mouse_event(QEvent.MouseMove, _view_pos(window, tile), Qt.NoButton, buttons)
    window.map_view.mouseMoveEvent(event)


def _release(window, tile, button=None, at=TILE_CENTRE) -> None:
    from PyQt5.QtCore import QEvent, Qt

    button = Qt.LeftButton if button is None else button
    event = conftest.mouse_event(QEvent.MouseButtonRelease, _view_pos(window, tile, at), button, Qt.NoButton)
    window.map_view.mouseReleaseEvent(event)


def _drag(window, path, button=None, at=TILE_CENTRE) -> None:
    """A real press-drag-release through MapView, with no move before the press."""
    _press(window, path[0], button, at)
    for tile in path[1:]:
        _held_move(window, tile, button)
    _release(window, path[-1], button, at)


def _units(window) -> list:
    return [(p, u) for p, units in enumerate(window.scenario.unit_manager.units) for u in units]


def _ghost(window):
    return window.map_view._unit_ghost_item


def _ghost_polygons(window) -> list:
    return [[(p.x(), p.y()) for p in polygon] for polygon, _brush, _pen in _ghost(window)._marks]


def _ghost_colors(window) -> set:
    return {brush.color().getRgb()[:3] for _polygon, brush, _pen in _ghost(window)._marks}


def _placed_polygons(window, unit) -> list:
    """The mark a placed `unit` shows, test_place_preview.py's _expected_mark() shape."""
    view = window.map_view
    entry = next(entry for entry in view._unit_index.entries if entry.unit is unit)
    return [[(float(x), float(y)) for x, y in pts] for pts in view._unit_polygons_for(entry)]


def _status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _row(y: int, x0: int, x1: int) -> list[tuple[int, int]]:
    return [(x, y) for x in range(x0, x1)]


# --- routing and the undo record -----------------------------------------------------


def test_a_real_drag_paints_one_undo_record_that_undo_and_redo_replay() -> None:
    from PyQt5.QtCore import QEvent, Qt

    window = _window(shown=True)
    try:
        view = window.map_view
        records = len(window.edit_history.records)

        def event(kind, tile, button, buttons):
            pos = conftest.polygon_viewport_pos(view, *tile)
            return conftest.mouse_event(kind, pos, button, buttons)

        view.mousePressEvent(event(QEvent.MouseButtonPress, (20, 20), Qt.LeftButton, Qt.LeftButton))
        view.mouseMoveEvent(event(QEvent.MouseMove, (30, 20), Qt.NoButton, Qt.LeftButton))
        view.mouseReleaseEvent(event(QEvent.MouseButtonRelease, (30, 20), Qt.LeftButton, Qt.NoButton))
        painted = len(_units(window))
        # A size-3 square along an 11-tile line: 3 rows of 13 tiles, density 100.
        assert painted == 39
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.records[-1].label == "Paint 39 units"
        assert _status(window) == f"Painted 39 {display_name(OAK)} for GAIA"
        window.undo()
        assert _units(window) == []
        window.redo()
        assert len(_units(window)) == painted
        assert window.unit_edits._pending is None
    finally:
        _close(window)


def test_a_stroke_that_places_nothing_records_nothing() -> None:
    window = _window()
    try:
        _stroke(window, _row(10, 10, 15))
        records = len(window.edit_history.records)
        before = len(_units(window))
        # Every tile already holds this stroke's trees, which block a second one.
        _stroke(window, _row(10, 10, 15))
        assert len(window.edit_history.records) == records
        assert len(_units(window)) == before
        assert window.unit_edits._pending is None
        assert _status(window).startswith(f"Painted 0 {display_name(OAK)} for GAIA, 21 refused (Blocked by ")
    finally:
        _close(window)


def test_a_right_press_opens_nothing() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        records = len(window.edit_history.records)
        _stroke(window, _row(10, 10, 15), button=Qt.RightButton)
        assert _units(window) == [] and window._unit_brush is None
        assert len(window.edit_history.records) == records
        assert window.unit_edits is None or window.unit_edits._pending is None
    finally:
        _close(window)


@pytest.mark.parametrize(("const", "word"), [(STONE_WALL, "walls"), (STONE_GATE, "gates")])
def test_a_wall_or_gate_pick_logs_and_opens_no_stroke(const, word) -> None:
    window = _window(const=const, owner=P1)
    try:
        _stroke(window, _row(10, 10, 15))
        assert _units(window) == [] and window._unit_brush is None
        assert "Unit Brush: can't paint" in _status(window) and word in _status(window)
    finally:
        _close(window)


def test_no_object_logs_and_opens_no_stroke() -> None:
    window = _window()
    try:
        window.units_panel._pending_object_const = None
        _stroke(window, _row(10, 10, 15))
        assert _units(window) == [] and window._unit_brush is None
        assert _status(window) == "Unit Brush: choose an object first"
    finally:
        _close(window)


# --- rules, Free placement and the status line ----------------------------------------


def _flood(window, tiles, terrain_id: int = WATER_DEEP) -> None:
    mm = window.scenario.map_manager
    for x, y in tiles:
        mm.terrain[y * mm.map_width + x].terrain_id = terrain_id


def test_free_placement_toggles_the_refusals_and_the_status_counts_them() -> None:
    window = _window(const=VILLAGER, owner=P1, size=1)
    try:
        water = _row(10, 12, 15)
        _flood(window, water)
        _stroke(window, _row(10, 10, 20))
        tiles = {(int(u.x), int(u.y)) for _p, u in _units(window)}
        assert tiles == set(_row(10, 10, 20)) - set(water)
        assert _status(window) == (
            "Painted 7 Villager Male for Player 1, 3 refused "
            "(Villager Male can't stand on Water Deep. Tick Free placement to place them anyway.)"
        )
        window.unrestricted_check.setChecked(True)
        _stroke(window, _row(11, 10, 20))
        assert {(int(u.x), int(u.y)) for _p, u in _units(window)} >= set(_row(11, 10, 20))
        assert _status(window) == "Painted 10 Villager Male for Player 1"
    finally:
        _close(window)


def test_density_misses_are_not_counted_as_refusals() -> None:
    window = _window(const=VILLAGER, owner=P1, size=5)
    try:
        window.unit_brush_density_spin.setValue(30)
        _stroke(window, _row(30, 20, 40))
        placed = len(_units(window))
        assert 20 < placed < 90, placed
        assert _status(window) == f"Painted {placed} Villager Male for Player 1"
    finally:
        _close(window)


def test_a_gaia_only_pick_paints_for_gaia() -> None:
    """Picking one switches the panel's owner to GAIA (the planner still
    coerces a player owner, pinned in test_unit_brush.py)."""
    window = _window(const=DEER, owner=P1, size=1)
    try:
        _stroke(window, _row(10, 10, 14))
        assert {p for p, _u in _units(window)} == {GAIA}, _status(window)
        assert _status(window) == "Painted 4 Deer for GAIA"
    finally:
        _close(window)


def test_a_random_pick_paints_both_consts_under_its_label() -> None:
    from descape.place_random import RANDOM_VILLAGER, choices

    window = _window(const=VILLAGER, owner=P1, size=5)
    try:
        window.units_panel.catalog_view.select(RANDOM_VILLAGER)
        _stroke(window, _row(30, 30, 36))
        assert {u.unit_const for _p, u in _units(window)} == set(choices(RANDOM_VILLAGER))
        label = window.units_panel.selected_object_label()
        assert label and _status(window) == f"Painted {len(_units(window))} {label} for Player 1"
    finally:
        _close(window)


def test_the_status_counts_units_a_filter_hides() -> None:
    window = _window(size=1)
    try:
        window.show_trees_action.setChecked(False)
        _stroke(window, _row(10, 10, 13))
        assert _status(window) == (
            f"Painted 3 {display_name(OAK)} for GAIA (3 hidden by a Filters toggle, so they won't be visible)"
        )
    finally:
        _close(window)


def test_no_two_painted_house_footprints_share_a_tile() -> None:
    window = _window(const=HOUSE, owner=P1, size=5)
    try:
        for free in (False, True):
            window.unrestricted_check.setChecked(free)
            before = len(_units(window))
            _stroke(window, [(20 + i, 40 if free else 20) for i in range(12)])
            new = _units(window)[before:]
            assert len(new) >= 6
            seen: set = set()
            for _p, unit in new:
                tiles = set(occupied_tiles_for(HOUSE, unit.x, unit.y, 120, 120))
                assert not seen & tiles
                seen |= tiles
    finally:
        _close(window)


# --- rotation -------------------------------------------------------------------------


def test_random_ticked_varies_a_tree_stroke_and_unticked_gives_the_pending_rotation() -> None:
    window = _window(size=5)
    try:
        assert window.unit_brush_random_check.isChecked()
        _stroke(window, _row(20, 20, 30))
        rotations = [u.rotation for _p, u in _units(window)]
        count = unit_variant.variant_count_for(OAK)
        assert len(set(rotations)) > 10
        assert all(r == int(r) and 0 <= r < count for r in rotations)
        window.unit_brush_random_check.setChecked(False)
        window._place_rotation = PlaceRotation(variant=7, variant_const=OAK)
        before = len(_units(window))
        _stroke(window, _row(40, 20, 30))
        assert {u.rotation for _p, u in _units(window)[before:]} == {7.0}
        assert _status(window).startswith("Painted ") and ", variant 7/" in _status(window)
    finally:
        _close(window)


def test_unticked_random_gives_an_angle_const_the_pending_facing() -> None:
    window = _window(const=VILLAGER, owner=P1, size=3)
    try:
        window.unit_brush_random_check.setChecked(False)
        pending = PlaceRotation(facing=1.0)
        window._place_rotation = pending
        _stroke(window, _row(20, 20, 25))
        expected = rotation_for(pending, VILLAGER, P1)
        assert expected != 0.0
        assert {u.rotation for _p, u in _units(window)} == {expected}
        count = unit_rotation.angle_count_for(VILLAGER)
        assert f"facing {unit_rotation.rotation_to_facing(expected, count)}/{count}" in _status(window)
    finally:
        _close(window)


# --- widgets ----------------------------------------------------------------------------


def test_the_tool_hides_outside_units_mode_and_shows_its_widgets() -> None:
    window = _window(shown=True)
    try:
        assert window.unit_brush_action.isVisible() and window.unit_brush_action.isEnabled()
        for action in (
            window.brush_size_spin_action,
            window.brush_shape_combo_action,
            window.unit_brush_density_param_action,
            window.unit_brush_random_param_action,
            window.off_grid_param_action,
            window.unrestricted_param_action,
            window.tool_param_separator_action,
        ):
            assert action.isVisible()
        for widget in (
            window.unit_brush_density_spin,
            window.unit_brush_random_check,
            window.off_grid_check,
            window.unrestricted_check,
            window.brush_size_spin,
        ):
            assert widget.isEnabled()
        assert not window.convert_sources_param_action.isVisible()
        assert window.unit_brush_density_spin.value() == 100
        assert window.unit_brush_density_spin.suffix() == " %"
        assert "Place Unit or by the Eyedropper" in window.unit_brush_random_check.toolTip()
        window.mode_combo.setCurrentText("Terrain")
        assert not window.unit_brush_action.isVisible()
        assert window._current_tool != "unit_brush"
        assert not window.unit_brush_density_param_action.isVisible()
        assert not window.unit_brush_density_spin.isEnabled()
        assert not window.unit_brush_random_check.isEnabled()
    finally:
        _close(window)


def test_a_tool_wheel_slot_resizes_the_brush_and_a_rotate_slot_does_nothing() -> None:
    """GH #245: the brush has a size box, and a ghost that does not turn."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        view = window.map_view
        view._wheel_bindings[int(Qt.ControlModifier)] = "tool"
        view._wheel_bindings[int(Qt.ShiftModifier)] = "rotate"
        assert view._wheel_action(Qt.ControlModifier) == "brush_size"
        assert view._wheel_action(Qt.ShiftModifier) == "none"
    finally:
        _close(window)


def test_the_tool_is_disabled_without_a_map() -> None:
    window = conftest.blank_window(load=False)
    try:
        assert not window.unit_brush_action.isEnabled()
    finally:
        _close(window)


# --- the hover ghost (GH #270) -------------------------------------------------------------


@pytest.mark.parametrize("const", [VILLAGER, HOUSE])
def test_the_ghost_is_the_unit_a_size_1_press_places(const) -> None:
    window = _window(const=const, owner=P1, size=1, shown=True)
    try:
        _hover(window, (40, 40), TILE_LOW)
        ghost = _ghost_polygons(window)
        assert ghost
        _drag(window, [(40, 40)], at=TILE_LOW)
        (_player, unit), = _units(window)
        assert ghost == _placed_polygons(window, unit)
    finally:
        _close(window)


@pytest.mark.parametrize("end", ["leave", "pan", "place_unit", "mode"])
def test_the_ghost_goes_with_the_hover_the_tool_and_the_mode(end) -> None:
    from PyQt5.QtCore import QEvent

    window = _window(const=VILLAGER, owner=P1, shown=True)
    try:
        view = window.map_view
        _hover(window, (40, 40))
        assert _ghost(window) is not None
        if end == "leave":
            view.leaveEvent(QEvent(QEvent.Leave))
        elif end == "mode":
            window.mode_combo.setCurrentText("View")
        else:
            getattr(window, f"{end}_action").setChecked(True)
            assert view._tool == end
        assert _ghost(window) is None
        assert view._place_hover is None
        view.refresh_place_ghost()
        assert _ghost(window) is None, "a refresh must not resurrect it"
    finally:
        _close(window)


def test_switching_from_place_unit_to_the_brush_drops_place_units_ghost() -> None:
    window = _window(const=VILLAGER, owner=P1, shown=True)
    try:
        window.place_unit_action.setChecked(True)
        _hover(window, (40, 40))
        assert _ghost(window) is not None
        window.unit_brush_action.setChecked(True)
        assert _ghost(window) is None and window.map_view._place_hover is None
    finally:
        _close(window)


@pytest.mark.parametrize("pick", ["nothing", "wall", "gate", "cliff"])
def test_a_pick_the_press_refuses_shows_no_ghost(pick) -> None:
    from descape import unit_kind

    window = _window(const=VILLAGER, owner=P1, shown=True)
    try:
        panel = window.units_panel
        _hover(window, (40, 40))
        assert _ghost(window) is not None
        if pick == "nothing":
            panel._pending_object_const = None
            window._refresh_place_ghost()
        else:
            panel._set_pending_object(
                {"wall": STONE_WALL, "gate": STONE_GATE, "cliff": min(unit_kind.cliff_consts())}[pick]
            )
        assert _ghost(window) is None
        _hover(window, (41, 41))
        assert _ghost(window) is None
    finally:
        _close(window)


def test_a_random_pick_ghosts_its_roll_unless_any_const_of_it_is_refused(monkeypatch) -> None:
    from descape.place_random import RANDOM_VILLAGER, choices

    window = _window(const=VILLAGER, owner=P1, shown=True)
    try:
        panel = window.units_panel
        panel.catalog_view.select(RANDOM_VILLAGER)
        _hover(window, (40, 40))
        assert _ghost(window) is not None
        # No shipped random pick holds a wall; the press would refuse the whole pick if one did.
        monkeypatch.setattr(panel, "selected_object_choices", lambda: (VILLAGER, STONE_WALL))
        _hover(window, (41, 41))
        assert panel.selected_object_const() in choices(RANDOM_VILLAGER), "vacuous: the roll itself must be paintable"
        assert _ghost(window) is None
        _stroke(window, [(41, 41)])
        assert _units(window) == [] and "can't paint" in _status(window)
    finally:
        _close(window)


def test_a_footprint_past_the_map_edge_shows_no_ghost() -> None:
    window = _window(const=BARRACKS, owner=P1, size=1, shown=True)
    try:
        _hover(window, (0, 0))
        assert _ghost(window) is None
        _hover(window, (1, 1))
        assert _ghost(window) is not None
        _hover(window, (119, 119))
        assert _ghost(window) is None
    finally:
        _close(window)


def test_a_catalog_pick_an_owner_change_and_free_placement_update_the_ghost_with_no_move() -> None:
    window = _window(const=VILLAGER, owner=P1, shown=True)
    try:
        _flood(window, [(70, 70)])
        _hover(window, (70, 70))
        assert _ghost(window).refused
        window.unrestricted_check.setChecked(True)
        assert not _ghost(window).refused
        window.unrestricted_check.setChecked(False)
        assert _ghost(window).refused

        colors = window.scenario.player_colors
        assert colors[1] != colors[2]
        assert _ghost_colors(window) == {tuple(colors[1])}
        window.units_panel.select_owner(2)
        assert _ghost_colors(window) == {tuple(colors[2])}

        before = _ghost(window).sceneBoundingRect()
        window.units_panel.select_object(HOUSE)
        assert _ghost(window).sceneBoundingRect() != before
    finally:
        _close(window)


def test_the_ghost_is_refused_on_water_and_over_a_unit_and_undo_turns_it_back() -> None:
    window = _window(const=VILLAGER, owner=P1, size=1, shown=True)
    try:
        _flood(window, [(70, 70)])
        _hover(window, (70, 70))
        assert _ghost(window).refused
        _hover(window, (40, 40))
        assert not _ghost(window).refused
        _drag(window, [(40, 40)])
        assert len(_units(window)) == 1
        assert _ghost(window).refused, "the unit just painted blocks the next one on its tile"
        _hover(window, (42, 42))
        assert not _ghost(window).refused
        _hover(window, (40, 40))
        assert _ghost(window).refused
        window.undo()
        assert not _ghost(window).refused
    finally:
        _close(window)


def test_a_stroke_hides_the_ghost_and_asks_for_none_until_the_release_at_the_live_point() -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    window = _window(const=VILLAGER, owner=P1, size=1, shown=True)
    try:
        view = window.map_view
        calls = []
        real = view._on_place_preview
        view.set_place_preview(lambda pos, modifiers: (calls.append(pos), real(pos, modifiers))[1])
        _hover(window, (40, 40))
        assert _ghost(window) is not None and len(calls) == 1
        calls.clear()

        _press(window, (40, 40))
        assert _ghost(window) is None
        for tile in ((41, 40), (42, 40)):
            _held_move(window, tile)
            assert _ghost(window) is None
        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Alt, Qt.AltModifier))
        view.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, Qt.Key_Alt, Qt.NoModifier))
        window.unrestricted_check.setChecked(True)
        window.unrestricted_check.setChecked(False)
        assert _ghost(window) is None
        assert calls == []

        _release(window, (42, 40))
        assert calls
        placed = {(int(u.x), int(u.y)): u for _p, u in _units(window)}
        assert sorted(placed) == [(40, 40), (41, 40), (42, 40)]
        assert _ghost_polygons(window) == _placed_polygons(window, placed[42, 40])
    finally:
        _close(window)


@pytest.mark.parametrize("stroke", ["placing", "all_refused", "right_button"])
def test_a_release_restores_the_ghost_with_no_move(stroke) -> None:
    from PyQt5.QtCore import Qt

    window = _window(const=VILLAGER, owner=P1, size=1, shown=True)
    try:
        tile = (40, 40)
        button = Qt.RightButton if stroke == "right_button" else Qt.LeftButton
        if stroke == "all_refused":
            _flood(window, [tile])
        _hover(window, tile)
        assert _ghost(window) is not None
        _press(window, tile, button)
        assert _ghost(window) is None
        _release(window, tile, button)
        assert _ghost(window) is not None
        assert bool(_units(window)) is (stroke == "placing")
    finally:
        _close(window)


@pytest.mark.parametrize("end", ["leave", "tool"])
def test_a_stroke_closed_by_a_leave_or_a_tool_switch_ends_with_no_ghost(end) -> None:
    from PyQt5.QtCore import QEvent

    window = _window(const=VILLAGER, owner=P1, size=1, shown=True)
    try:
        view = window.map_view
        _hover(window, (40, 40))
        _press(window, (40, 40))
        if end == "leave":
            view.leaveEvent(QEvent(QEvent.Leave))
        else:
            window.pan_action.setChecked(True)
        assert not view._stroke_active
        assert len(_units(window)) == 1
        assert _ghost(window) is None and view._place_hover is None
    finally:
        _close(window)


@pytest.mark.parametrize("random_ticked", [True, False], ids=["random", "unticked"])
def test_the_ghost_carries_the_pending_rotation(random_ticked, monkeypatch) -> None:
    from descape import viewer

    window = _window(const=VILLAGER, owner=P1, shown=True)
    try:
        window.unit_brush_random_check.setChecked(random_ticked)
        pending = PlaceRotation(facing=1.0)
        window._place_rotation = pending
        seen = []
        real = viewer.new_unit
        monkeypatch.setattr(viewer, "new_unit", lambda *a, **k: (seen.append(k["rotation"]), real(*a, **k))[1])
        _hover(window, (40, 40))
        assert _ghost(window) is not None
        expected = rotation_for(pending, VILLAGER, P1)
        assert expected != 0.0
        assert seen == [expected]
    finally:
        _close(window)


def test_hovering_builds_no_unit_model_adds_no_unit_and_invalidates_no_cache() -> None:
    """test_place_preview.py's test of the same name: the ghost is an overlay under the brush too."""
    window = _window(const=VILLAGER, owner=P1, shown=True)
    try:
        assert window.unit_edits is None, "the fixture must start with the model unbuilt"
        calls = {"units": 0, "region": 0}
        cache = window._cache
        real_units, real_region = cache.invalidate_units, cache.invalidate_region

        def counted_units(*a, **k):
            calls["units"] += 1
            return real_units(*a, **k)

        def counted_region(*a, **k):
            calls["region"] += 1
            return real_region(*a, **k)

        cache.invalidate_units = counted_units
        cache.invalidate_region = counted_region
        for step in range(8):
            _hover(window, (30 + step, 30 + step))
        window.units_panel.select_owner(3)
        window.unrestricted_check.setChecked(True)
        assert _ghost(window) is not None
        assert window.unit_edits is None
        assert _units(window) == []
        assert calls == {"units": 0, "region": 0}
    finally:
        _close(window)


def test_the_ghost_arms_no_rotate_input() -> None:
    """GH #170's Out of scope stands: rotate inputs turn only Place Unit's ghost."""
    window = _window(const=VILLAGER, owner=P1, shown=True)
    try:
        _hover(window, (40, 40))
        assert _ghost(window) is not None
        assert not window._place_ghost_active()
    finally:
        _close(window)


# --- off-grid -----------------------------------------------------------------------------


def test_off_grid_and_a_held_alt_jitter_inside_the_tile() -> None:
    from PyQt5.QtCore import Qt

    window = _window(size=5)
    try:
        # Jittered tree boxes reach their neighbours' and the rules would refuse them.
        window.unrestricted_check.setChecked(True)
        _stroke(window, [(20, 20)])
        assert {(u.x % 1, u.y % 1) for _p, u in _units(window)} == {(0.5, 0.5)}
        before = len(_units(window))
        _stroke(window, [(40, 40)], modifiers=Qt.AltModifier)
        jittered = _units(window)[before:]
        assert len({(u.x, u.y) for _p, u in jittered}) == len(jittered) == 25
        assert len({(u.x % 1, u.y % 1) for _p, u in jittered}) > 20
        assert {(int(u.x), int(u.y)) for _p, u in jittered} == {
            (x, y) for x in range(38, 43) for y in range(38, 43)
        }
        window.off_grid_check.setChecked(True)
        before = len(_units(window))
        _stroke(window, [(60, 60)])
        assert len({(u.x % 1, u.y % 1) for _p, u in _units(window)[before:]}) > 20
    finally:
        _close(window)


# --- document switch, errors ------------------------------------------------------------------


def test_a_document_switch_mid_stroke_drops_it() -> None:
    from PyQt5.QtCore import Qt

    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = _window()
    try:
        window.map_view._stroke_button = Qt.LeftButton
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(20, 20)], Qt.NoModifier)
        assert window._unit_brush is not None
        window.edit_history.mark_saved()
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window._unit_brush is None and window._unit_stroke_pending_splices == []
        window.on_edit_stroke_tiles([(30, 30)], Qt.NoModifier)
        window.on_edit_stroke_end()
        assert _units(window) == []
        assert len(window.edit_history.records) == 0
    finally:
        _close(window)


def test_a_raise_mid_stroke_commits_what_was_placed(monkeypatch) -> None:
    from PyQt5.QtCore import Qt

    from descape import unit_brush

    window = _window(size=1)
    try:
        window.map_view._stroke_button = Qt.LeftButton
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(20, 20)], Qt.NoModifier)
        real = unit_brush.paint_dab

        def boom(*args, **kwargs):
            real(*args, **kwargs)
            raise RuntimeError("boom")

        monkeypatch.setattr(unit_brush, "paint_dab", boom)
        with pytest.raises(RuntimeError):
            window.on_edit_stroke_tiles([(21, 20)], Qt.NoModifier)
        assert window._unit_brush is None
        assert window.unit_edits._pending is None
        assert window.edit_history.records[-1].label == "Paint 2 units"
        window.on_edit_stroke_end()
        assert len(_units(window)) == 2
    finally:
        _close(window)


@pytest.mark.parametrize("fault", ["add", "footprint"])
def test_a_raise_inside_the_first_dab_records_the_unit_already_added(fault, monkeypatch) -> None:
    """GH #170 review: "add" fails the model's second add(), so the first is
    in the model and on the plan's dab only; "footprint" fails the viewer's
    callback right after the first add(), before the plan sees it."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        records = len(window.edit_history.records)
        window.map_view._stroke_button = Qt.LeftButton
        window.on_edit_stroke_start()
        calls = []
        if fault == "add":
            model = window._unit_brush_model
            real = model.add

            def add(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("boom")
                return real(*args, **kwargs)

            monkeypatch.setattr(model, "add", add)
        else:
            real = window._unit_footprint

            def footprint(unit):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("boom")
                return real(unit)

            monkeypatch.setattr(window, "_unit_footprint", footprint)
        with pytest.raises(RuntimeError):
            window.on_edit_stroke_tiles([(20, 20)], Qt.NoModifier)
        window.on_edit_stroke_end()
        assert len(_units(window)) == 1
        assert window._unit_brush is None and window.unit_edits._pending is None
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.records[-1].label == "Paint unit"
        assert _status(window).startswith("Painted 1 ")
        window.undo()
        assert _units(window) == []
    finally:
        _close(window)


def test_a_convert_press_over_an_open_brush_stroke_commits_and_repaints_it(monkeypatch) -> None:
    """GH #170 review: the two strokes share one pending list, so Convert's
    begin ends the brush's through its own end rather than dropping the list."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        records = len(window.edit_history.records)
        window.map_view._stroke_button = Qt.LeftButton
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(20, 20)], Qt.NoModifier)
        pending = list(window._unit_stroke_pending_splices)
        assert len(pending) == 9, "vacuous: nothing awaits a repaint"
        flushed = []
        real = window._after_unit_mutation

        def spy(*args, **kwargs):
            flushed.append(kwargs.get("changed"))
            return real(*args, **kwargs)

        monkeypatch.setattr(window, "_after_unit_mutation", spy)
        window._begin_convert_stroke()
        assert window._unit_brush is None
        assert flushed and flushed[0] == pending
        assert window.edit_history.records[records].label == "Paint 9 units"
        window._end_convert_stroke()
        assert window._unit_stroke_pending_splices == []
    finally:
        _close(window)


# --- the write path ---------------------------------------------------------------------------


def test_a_brushed_file_saves_and_reloads_every_painted_unit_with_its_rotation(tmp_path) -> None:
    from descape import scenario_io
    from descape.scenario_write import write_scenario

    window = _window(size=5)
    try:
        _stroke(window, _row(30, 30, 36))
        assert window.units_panel.pick_object(DEER)
        _stroke(window, _row(50, 30, 36))
        assert window.units_panel.select_owner(P1)
        assert window.units_panel.pick_object(HOUSE)
        _stroke(window, _row(70, 30, 40))
        painted = sorted((p, u.unit_const, u.x, u.y, u.rotation) for p, u in _units(window))
        assert {c for _p, c, _x, _y, _r in painted} == {OAK, DEER, HOUSE}
        out = tmp_path / "brushed.aoe2scenario"
        write_scenario(window.scenario, out, backup=False, **window._edit_model_kwargs())
        reloaded = scenario_io.load_map_and_units(out)
        found = sorted(
            (p, u.unit_const, u.x, u.y, u.rotation)
            for p, units in enumerate(reloaded.unit_manager.units)
            for u in units
        )
        assert len(found) == len(painted)
        for (p, c, x, y, r), (fp, fc, fx, fy, fr) in zip(painted, found, strict=True):
            assert (p, c) == (fp, fc)
            assert (fx, fy, fr) == pytest.approx((x, y, r), abs=1e-5)
        assert len({r for _p, c, _x, _y, r in found if c == OAK}) > 5
        assert len({r for _p, c, _x, _y, r in found if c == DEER}) > 3
        houses = [(x, y) for _p, c, x, y, _r in found if c == HOUSE]
        assert houses and all(x == int(x) and y == int(y) for x, y in houses)
    finally:
        _close(window)
