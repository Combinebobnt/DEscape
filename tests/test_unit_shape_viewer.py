"""Unit Line and Unit Shape (GH #365) through a real offscreen ViewerWindow:
the press latch, the unit branch's commit, and the options row. The wall
branch's ports of the old Wall Rectangle / Wall Circle tests live in
test_wall_run_viewer.py; the geometry and the candidate tail in
test_unit_shape.py."""

from __future__ import annotations

import pytest
from test_wall_run_viewer import (  # noqa: F401 -- _hold_the_synthetic_button is an autouse fixture
    GATE,
    PALISADE,
    PLAYER,
    VILLAGER,
    WALL,
    _band_polygons,
    _close,
    _drag,
    _hold_the_synthetic_button,
    _move,
    _pick,
    _press,
    _release,
    _shape_window,
    _tile_polygons,
    _unit_count,
    _window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

HOUSE = 70  # 2x2
CLIFF = 264
OTHER_PLAYER = 2


def _units_since(window, before: dict[int, int]) -> list:
    units = window.scenario.unit_manager.units
    return [unit for player, count in before.items() for unit in units[player][count:]]


def _counts(window) -> dict[int, int]:
    return {player: len(units) for player, units in enumerate(window.scenario.unit_manager.units)}


def _key(map_view, key, release: bool = False) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    kind = QEvent.KeyRelease if release else QEvent.KeyPress
    event = QKeyEvent(kind, key, Qt.ShiftModifier if key == Qt.Key_Shift and not release else Qt.NoModifier)
    (map_view.keyReleaseEvent if release else map_view.keyPressEvent)(event)


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def test_a_house_rectangle_dragged_up_left_is_one_undo_record_inside_the_box():
    window = _shape_window(HOUSE, "rectangle")
    try:
        window.unit_shape_spacing_spin.setValue(1)
        before = _counts(window)
        _drag(window.map_view, (40, 40), (31, 31))
        placed = _units_since(window, before)
        # A 10-tile box of 2x2 houses at Spacing 1: 3 cells per axis from the anchor down, the 8 outline cells.
        assert len(placed) == 8
        assert {unit.unit_const for unit in placed} == {HOUSE}
        assert {unit.x for unit in placed} == {34.0, 37.0, 40.0} == {unit.y for unit in placed}
        assert [r.kind for r in window.edit_history.records] == ["unit"]
        window.undo()
        assert _counts(window) == before
    finally:
        _close(window)


def test_a_drag_with_every_candidate_refused_records_nothing_and_says_why():
    window = _shape_window(HOUSE, "rectangle")
    try:
        _drag(window.map_view, (40, 40), (45, 45))
        assert len(window.edit_history.records) == 1
        before = _unit_count(window)
        _drag(window.map_view, (40, 40), (45, 45))
        assert _unit_count(window) == before
        assert len(window.edit_history.records) == 1
        status = _last_status(window)
        assert status.startswith("Placed 0 ") and "refused" in status, status
    finally:
        _close(window)


@pytest.mark.parametrize(
    ("pick", "reason"),
    [
        ((), "pick an object in the catalog"),
        ((GATE,), "gates are placed one at a time"),
        ((CLIFF,), "cliffs are placed as chains"),
        ((WALL, VILLAGER), "pick walls alone"),
        ((WALL, PALISADE), "pick one wall"),
    ],
)
@pytest.mark.parametrize("tool", ["unit_line", "unit_shape"])
def test_a_refused_pick_starts_no_drag_and_says_why(monkeypatch, tool, pick, reason):
    window = _shape_window(VILLAGER, "rectangle")
    try:
        getattr(window, f"{tool}_action").setChecked(True)
        monkeypatch.setattr(window.units_panel, "selected_object_choices", lambda: pick or None)
        before = _unit_count(window)
        _press(window.map_view, (20, 40))
        assert window.map_view._shape_anchor is None and window.map_view._shape_latch is None
        _move(window.map_view, (28, 44))
        _release(window.map_view, (28, 44))
        assert _unit_count(window) == before
        assert not window.edit_history.can_undo
        label = "Unit Line" if tool == "unit_line" else "Unit Shape"
        assert f"{label}: {reason}" in window.status_log.toPlainText()
    finally:
        _close(window)


@pytest.mark.parametrize(("const", "other"), [(VILLAGER, HOUSE), (WALL, PALISADE)])
def test_the_press_latches_the_pick_owner_and_options_for_the_whole_drag(const, other):
    window = _shape_window(const, "rectangle")
    try:
        before = _counts(window)
        _press(window.map_view, (20, 40))
        _move(window.map_view, (24, 44))
        _pick(window, other)
        assert window.units_panel.select_owner(OTHER_PLAYER)
        window.unit_shape_spacing_spin.setValue(3)
        _release(window.map_view, (24, 44))
        after = _counts(window)
        # A 5x5 box at the latched Spacing 0 of 1x1 objects: its 16 outline tiles, all the latched owner's.
        assert after[PLAYER] == before[PLAYER] + 16
        assert sum(after.values()) == sum(before.values()) + 16
        placed = window.scenario.unit_manager.units[PLAYER][before[PLAYER]:]
        assert {unit.unit_const for unit in placed} == {const}
    finally:
        _close(window)


def test_a_mixed_pick_names_the_press_time_pick_and_the_shape(monkeypatch):
    """A random pick's label is read at the press like the rest of the latch, and the undo step names the Shape."""
    window = _shape_window(VILLAGER, "triangle")
    try:
        monkeypatch.setattr(window.units_panel, "selected_object_choices", lambda: (VILLAGER, HOUSE))
        monkeypatch.setattr(window.units_panel, "selected_object_label", lambda: "Press pick")
        _press(window.map_view, (20, 40))
        _move(window.map_view, (30, 46))
        monkeypatch.setattr(window.units_panel, "selected_object_label", lambda: "Later pick")
        _release(window.map_view, (30, 46))
        assert [r.label for r in window.edit_history.records] == ["Draw Triangle"]
        assert " Press pick for Player 1" in _last_status(window)
    finally:
        _close(window)


@pytest.mark.parametrize("cancel", ["escape", "tool switch"])
def test_a_cancelled_shape_drag_leaves_no_latch_for_a_place_unit_wall_run(cancel):
    from PyQt5.QtCore import Qt

    window = _shape_window(VILLAGER, "rectangle")
    try:
        map_view = window.map_view
        _press(map_view, (20, 40))
        _move(map_view, (24, 44))
        assert map_view._shape_latch is not None
        if cancel == "escape":
            _key(map_view, Qt.Key_Escape)
        else:
            window.place_unit_action.setChecked(True)
        assert map_view._shape_latch is None and map_view._shape_anchor is None
        _release(map_view, (24, 44))
        assert not window.edit_history.can_undo
        _pick(window, WALL)
        window.place_unit_action.setChecked(True)
        before = _counts(window)
        _drag(map_view, (30, 30), (35, 30))
        placed = _units_since(window, before)
        assert sorted((int(u.x), int(u.y)) for u in placed) == [(x, 30) for x in range(30, 36)]
        assert {unit.unit_const for unit in placed} == {WALL}
        assert [r.kind for r in window.edit_history.records] == ["unit"]
    finally:
        _close(window)


@pytest.mark.parametrize("const", [WALL, VILLAGER])
def test_shift_with_no_move_rebuilds_the_preview_and_the_commit_matches_it(const):
    from PyQt5.QtCore import Qt

    window = _shape_window(const, "rectangle")
    try:
        map_view = window.map_view
        _press(map_view, (20, 40))
        _move(map_view, (28, 44))
        loose = sorted(map_view._shape_tiles(preview=True))
        _key(map_view, Qt.Key_Shift)
        snapped = sorted(map_view._shape_tiles(preview=True))
        assert snapped != loose
        assert _band_polygons(map_view) == _tile_polygons(map_view, snapped)
        before = _counts(window)
        committed = map_view._shape_tiles(preview=False)
        _release(map_view, (28, 44))
        placed = _units_since(window, before)
        # 1x1 objects with no rule refusals: one per previewed tile.
        assert sorted((int(u.x), int(u.y)) for u in placed) == sorted(committed) == snapped
    finally:
        _close(window)


@pytest.mark.parametrize("adds_first", [False, True])
def test_an_exception_inside_the_unit_commit_leaves_no_edit_pending(monkeypatch, adds_first):
    from descape import unit_brush, unit_shape

    window = _shape_window(VILLAGER, "rectangle")
    try:
        latch = window._shape_press()
        assert latch is not None and latch.branch == unit_shape.UNITS

        real = unit_shape.place_cells
        calls = []

        # Raises on the first call only (monkeypatch.undo() would also drop conftest's config isolation).
        def broken(scenario, latch, cells, add, rng):
            calls.append(cells)
            if len(calls) > 1:
                return real(scenario, latch, cells, add, rng)
            if adds_first:
                add(unit_brush.Spec(PLAYER, VILLAGER, 20.5, 40.5, 0.0))
            raise RuntimeError("boom")

        monkeypatch.setattr(unit_shape, "place_cells", broken)
        with pytest.raises(RuntimeError, match="boom"):
            window.on_shape_commit([(20, 40), (21, 40)], latch=latch)
        model = window._ensure_unit_edits()
        assert model._pending is None
        assert len(window.edit_history.records) == (1 if adds_first else 0)
        before = _unit_count(window)
        _drag(window.map_view, (30, 30), (32, 30))
        assert _unit_count(window) == before + 3
        assert len(window.edit_history.records) == (2 if adds_first else 1)
    finally:
        _close(window)


@pytest.mark.parametrize(("answer", "placed"), [("cancel", 0), ("yes", 16)])
def test_a_shape_over_the_threshold_asks_its_own_large_edit_question(monkeypatch, answer, placed):
    from PyQt5.QtWidgets import QMessageBox

    from descape import viewer

    asked: list[str] = []

    def question(_parent, title, text, *_args):
        asked.append(f"{title}: {text}")
        return QMessageBox.Yes if answer == "yes" else QMessageBox.Cancel

    monkeypatch.setattr(viewer, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 10)
    monkeypatch.setattr(viewer.QMessageBox, "question", staticmethod(question))
    window = _shape_window(VILLAGER, "rectangle")
    try:
        before = _unit_count(window)
        _drag(window.map_view, (20, 40), (24, 44))
        assert asked == ["Large edit: This places up to 16 objects, which may take a while. Continue?"]
        assert _unit_count(window) == before + placed
        _drag(window.map_view, (50, 40), (52, 42))
        assert len(asked) == 1, "8 cells are under the threshold"
    finally:
        _close(window)


def test_the_options_row_shows_each_tools_own_controls():
    """QAction.isVisible(), as test_wall_run_viewer's visibility tests: Shape
    and Filled under Unit Shape alone, Spacing and Random under both."""
    window = _window(VILLAGER)
    try:
        box = ("unit_shape_kind_param_action", "unit_shape_filled_param_action")
        shared = (
            "unit_shape_spacing_label_action",
            "unit_shape_spacing_param_action",
            "unit_shape_random_param_action",
        )

        def shown():
            return {name for name in box + shared if getattr(window, name).isVisible()}

        assert shown() == set()
        window.unit_line_action.setChecked(True)
        assert shown() == set(shared)
        window.unit_shape_action.setChecked(True)
        assert shown() == set(box + shared)
        window.unit_brush_action.setChecked(True)
        assert shown() == set()
    finally:
        _close(window)
