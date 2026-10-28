"""The trigger reference index (unit_references.ReferenceIndex) through a real
offscreen ViewerWindow: a unit edit that carries UnitSplices patches the
held index in place instead of rebuilding it, and every result equals a
fresh build_reference_index() over the same scenario state. An edit with
changed=None still drops the index for a lazy rebuild.

Uses units_120x120.aoe2scenario in Flat, like test_unit_edit_viewer.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_VILLAGER = 83
_REF_ARCHER_P1 = 201
_REF_ARCHER_P2 = 300
_REF_TREE_PINE = 101
_REF_WALL = 102
_GATE_NE = 64


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.show_garrisoned_action.setChecked(True)
    window.mode_combo.setCurrentText("Units")
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _select(window, *reference_ids):
    entries = [e for e in window.map_view._unit_index.entries if e.unit.reference_id in reference_ids]
    assert len(entries) == len(reference_ids), "fixture unit missing from the index"
    window._selection = [(e.player_id, e.unit.reference_id) for e in entries]
    window._refresh_selection_view()
    return entries


def _pos_for_tile(window, tile_x: int, tile_y: int):
    from PyQt5.QtCore import QPointF

    tp = window.map_view._tile_pixels
    return QPointF(tile_x * tp + tp // 2, tile_y * tp + tp // 2)


def _empty_tile(window) -> tuple[int, int]:
    occupied = {(int(u.x), int(u.y)) for units in window.scenario.unit_manager.units for u in units}
    return next((x, y) for x in range(60, 80) for y in range(60, 80) if (x, y) not in occupied)


class _Builds:
    """Counts the viewer's own full builds; the oracle calls `fresh` instead."""

    def __init__(self, monkeypatch) -> None:
        from descape import unit_references

        self.fresh = unit_references.build_reference_index
        self.calls = 0

        def counted(loaded):
            self.calls += 1
            return self.fresh(loaded)

        monkeypatch.setattr(unit_references, "build_reference_index", counted)


def _prime(window):
    index = window._unit_reference_index()
    assert index is window._unit_ref_index is not None
    return index


def _assert_patched(window, builds: _Builds, primed, step: str) -> None:
    """The held index survived the edit (no rebuild) and matches a fresh build."""
    assert window._unit_ref_index is primed, f"{step}: the index was dropped, not patched"
    assert builds.calls == 0, f"{step}: a full build ran"
    fresh = builds.fresh(window.scenario)
    assert dict(primed.by_id) == dict(fresh.by_id), f"{step}: patched index differs from a fresh build"
    assert primed.duplicates == fresh.duplicates, step


def _nudge(window, *reference_ids) -> None:
    from PyQt5.QtCore import Qt

    _select(window, *reference_ids)
    window.on_unit_nudge(1, 0, Qt.ShiftModifier)


def _place(window, owner: int, tile) -> None:
    from PyQt5.QtCore import Qt

    window.units_panel.select_object(_VILLAGER)
    window.units_panel.select_owner(owner)
    window.place_unit_action.setChecked(True)
    window.on_unit_place(_pos_for_tile(window, *tile), Qt.NoModifier)
    window.pan_action.setChecked(True)


def _delete(window, reference_id: int) -> None:
    from PyQt5.QtCore import Qt

    _select(window, reference_id)
    window.on_unit_delete(Qt.NoModifier)


def _make_gate(window) -> None:
    (entry,) = _select(window, _REF_WALL)
    entry.unit.unit_const, entry.unit.x, entry.unit.y = _GATE_NE, 10.0, 5.5
    window._after_unit_mutation()


def _convert_targets(window, count: int = 2):
    """`count` player-1 non-building units on distinct tiles holding only player-1 units."""
    from descape import render

    index = window.map_view._unit_index
    picked, seen = [], set()
    for entry in index.entries:
        tile = (entry.own_x, entry.own_y)
        if entry.player_id != 1 or tile in seen or entry.unit.unit_const in render.BUILDING_TILE_SPANS:
            continue
        if all(index.entries[o].player_id == 1 for o in index.by_tile[tile]):
            seen.add(tile)
            picked.append(entry)
        if len(picked) == count:
            break
    assert len(picked) == count
    return picked


def _convert(window) -> None:
    window.units_panel.select_owner(2)
    window.brush_size_spin.setValue(1)
    first, second = _convert_targets(window)
    window._begin_convert_stroke()
    window._convert_stroke_tile(first.own_x, first.own_y)
    window._unit_stroke_refresh_timer.stop()
    window._flush_unit_stroke_refresh()
    window._convert_stroke_tile(second.own_x, second.own_y)
    window._end_convert_stroke()


# Each step carries UnitSplices end to end, so the held index must be patched.
_PATCHED_STEPS = {
    "nudge": lambda w: _nudge(w, _REF_ARCHER_P1),
    "group nudge across two players": lambda w: _nudge(w, _REF_ARCHER_P1, _REF_ARCHER_P2),
    "undo nudge": lambda w: w.undo(),
    "redo nudge": lambda w: w.redo(),
    "rotate": lambda w: (_select(w, _REF_ARCHER_P1), w.on_unit_rotate(1)),
    "place for player 1": lambda w: _place(w, 1, _empty_tile(w)),
    "place for GAIA": lambda w: _place(w, 0, _empty_tile(w)),
    # Flat's _if_spliceable() sends a membership undo on this small fixture down the wholesale path.
    "switch to Stepped": lambda w: w.terrain_style_combo.setCurrentText("Stepped"),
    "undo GAIA place": lambda w: w.undo(),
    "redo GAIA place": lambda w: w.redo(),
    "delete a player unit": lambda w: _delete(w, _REF_ARCHER_P2),
    "delete a GAIA tree": lambda w: _delete(w, _REF_TREE_PINE),
    "undo GAIA delete": lambda w: w.undo(),
    "redo GAIA delete": lambda w: w.redo(),
    "convert with a mid-stroke flush": _convert,
}


def test_every_splice_carrying_edit_patches_the_held_index_to_match_a_fresh_build(monkeypatch) -> None:
    window = _window()
    try:
        builds = _Builds(monkeypatch)
        primed = _prime(window)
        assert builds.calls == 1
        builds.calls = 0
        for step, run in _PATCHED_STEPS.items():
            run(window)
            _assert_patched(window, builds, primed, step)
    finally:
        _close(window)


def test_a_gate_rotate_patches_the_new_const_and_footprint(monkeypatch) -> None:
    window = _window()
    try:
        _make_gate(window)
        builds = _Builds(monkeypatch)
        primed = _prime(window)
        builds.calls = 0
        before = primed.by_id[_REF_WALL]
        _select(window, _REF_WALL)
        window.on_unit_rotate(1)
        _assert_patched(window, builds, primed, "gate rotate")
        after = primed.by_id[_REF_WALL]
        assert (after.unit_const, after.bounds) != (before.unit_const, before.bounds)
    finally:
        _close(window)


# Each step reaches _after_unit_mutation with changed=None: the index is dropped.
_REBUILT_STEPS = {
    "wholesale refresh": lambda w: w._after_unit_mutation(),
    "undo player 1 place": lambda w: (_place(w, 1, _empty_tile(w)), w.undo()),
    "undo convert": lambda w: (_convert(w), w.undo()),
}


@pytest.mark.parametrize("step", list(_REBUILT_STEPS))
def test_an_edit_with_no_splices_drops_the_index_and_one_read_rebuilds_it(step, monkeypatch) -> None:
    window = _window()
    try:
        builds = _Builds(monkeypatch)
        primed = _prime(window)
        _REBUILT_STEPS[step](window)
        # The panel labels may already have re-read it; either way the primed one is gone.
        assert window._unit_ref_index is not primed
        builds.calls = 0
        index = window._unit_reference_index()
        assert builds.calls <= 1
        assert window._unit_reference_index() is index
        fresh = builds.fresh(window.scenario)
        assert dict(index.by_id) == dict(fresh.by_id) and index.duplicates == fresh.duplicates
    finally:
        _close(window)


def test_the_wholesale_path_drops_the_held_index() -> None:
    window = _window()
    try:
        _prime(window)
        window._after_unit_mutation()
        assert window._unit_ref_index is None
    finally:
        _close(window)


def test_a_nudge_under_a_held_trigger_overlay_moves_the_outline_without_a_build(monkeypatch) -> None:
    """The TODO's case: a trigger naming the unit is held, so every unit
    edit re-derives the overlay, which used to rebuild the whole index."""
    from PyQt5.QtCore import Qt

    gen = conftest.load_verify_module("gen_trigger_fixture")
    window = conftest.shown_window(900, 700)
    try:
        window.load_scenario(Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario")
        window.terrain_style_combo.setCurrentText("Stepped")
        window.mode_combo.setCurrentText("Triggers")
        window.trigger_panel.select_trigger(gen.REFERENCE_TRIGGER)
        window.mode_combo.setCurrentText("Units")
        assert window._trigger_overlay_ref is not None
        builds = _Builds(monkeypatch)
        primed = _prime(window)
        builds.calls = 0

        def house_outlines():
            view = window.map_view
            return [s.coords for s in view._trigger_shapes if s.shape == "unit" and s.fields[0] == "unit_object"]

        before = house_outlines()
        assert before
        _select(window, gen.REF_HOUSE)
        window.on_unit_nudge(1, 0, Qt.ShiftModifier)
        _assert_patched(window, builds, primed, "nudge under a held overlay")
        after = house_outlines()
        ((x0, y0, x1, y1),) = before
        assert after == [(x0 + 1, y0, x1 + 1, y1)], "the outline follows the whole-tile nudge"
    finally:
        conftest.close_window(window)


def test_a_moved_reference_label_follows_a_patched_nudge() -> None:
    """The panel label reads the patched index, so it shows the new tile."""
    from descape import unit_references

    window = _window()
    try:
        primed = _prime(window)
        (entry,) = _select(window, _REF_ARCHER_P1)
        _nudge(window, _REF_ARCHER_P1)
        assert window._unit_ref_index is primed
        text = window._describe_unit_reference(_REF_ARCHER_P1)
        assert text.endswith(f"X{int(entry.unit.x)}, Y{int(entry.unit.y)}]")
        assert text == unit_references.describe(unit_references.build_reference_index(window.scenario), _REF_ARCHER_P1)
    finally:
        _close(window)


# -- the owed in-app passes, offscreen (TASK-088) ------------------------------------


def _held_task_object_window(gen):
    """The trigger fixture in Flat, its Task Object effect selected in Triggers, then Units."""
    window = conftest.shown_window(1100, 760)
    window.load_scenario(Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario")
    assert window.scenario is not None
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_trigger(gen.REFERENCE_TRIGGER)
    window.trigger_panel.select_entry("effect", gen.TASK_OBJECT_EFFECT)
    window.mode_combo.setCurrentText("Units")
    assert window._trigger_overlay_ref is not None
    return window


def _selected_rings(window, gen):
    """The Task Object's selected-objects outlines, as the overlay's inclusive tile rects."""
    view = window.map_view
    return [
        s.coords
        for s in view._trigger_shapes
        if s.shape == "unit" and s.fields[0] == "selected_object_ids" and s.entry_ref == ("effect", gen.TASK_OBJECT_EFFECT)
    ]


def _units_strong_bounds(view):
    (item,) = [i for i in view.trigger_overlay_items() if i.data(0) == "units_strong"]
    return item.path().boundingRect()


def _units_strong_path(view):
    (item,) = [i for i in view.trigger_overlay_items() if i.data(0) == "units_strong"]
    return item.path()


def _selected_objects_label(window) -> str:
    """The text of the panel's own QLabel under the Task Object's selected objects field."""
    from test_trigger_unit_picker_viewer import _row, _texts

    (text,) = _texts(_row(window.trigger_panel, "selected_object_ids"))
    return text


def test_a_move_drag_of_a_referenced_unit_moves_its_outline_and_label_and_undo_brings_both_back() -> None:
    """TASK-088 step 10, the drag half. Pass conditions, the Task Object (archers A and B) held
    from Triggers, then Units with the Move tool: a real press-move-release dragging archer A
    from tile (50, 55) to (54, 55) moves it there, the overlay's ring for it is then exactly
    tile (54, 55), and the panel's selected-objects label line for archer A ends
    "[P1, X54, Y55]"; one Undo puts the unit, the ring and the label back on (50, 55)."""
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtWidgets import QApplication

    gen = conftest.load_verify_module("gen_trigger_fixture")
    window = _held_task_object_window(gen)
    try:
        view = window.map_view
        window.move_action.trigger()
        assert window._current_tool == "move"
        start, end = (50, 55), (54, 55)
        assert tuple(int(c) for c in gen.ARCHER_A_POS) == start
        ring_before = _selected_rings(window, gen)
        assert (*start, *start) in ring_before
        bounds_before = _units_strong_bounds(view)
        label_before = _selected_objects_label(window)
        assert "[P1, X50, Y55]" in label_before.splitlines()[0]
        tp = view._tile_pixels

        def viewport(tile):
            return QPointF(view.mapFromScene(QPointF((tile[0] + 0.5) * tp, (tile[1] + 0.5) * tp)))

        view.centerOn(QPointF(52 * tp, 55.5 * tp))
        QApplication.processEvents()
        press, release = viewport(start), viewport(end)
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton))
        view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, release, Qt.NoButton, Qt.LeftButton))
        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, release, Qt.LeftButton, Qt.NoButton))
        unit = next(u for u in window.scenario.unit_manager.units[1] if u.reference_id == gen.REF_ARCHER_A)
        assert (int(unit.x), int(unit.y)) == end, "the drag did not move archer A"
        rings = _selected_rings(window, gen)
        assert (*end, *end) in rings and (*start, *start) not in rings
        assert _units_strong_bounds(view) != bounds_before, "the drawn outline did not move"
        assert _selected_objects_label(window).splitlines()[0].endswith("[P1, X54, Y55]")

        window.undo()
        assert (int(unit.x), int(unit.y)) == start
        assert _selected_rings(window, gen) == ring_before
        assert _units_strong_bounds(view) == bounds_before
        assert _selected_objects_label(window) == label_before
    finally:
        conftest.close_window(window)


def test_the_panel_label_follows_a_patched_arrow_key_nudge() -> None:
    """TASK-088 steps 10 (arrow keys) and 12, functional only. Pass conditions, the Task
    Object held from Triggers, archer A selected in Units and the reference index built: a
    Shift+Right key press on the map view nudges archer A one tile east, the held index is
    patched in place (the same object, no rebuild), and the panel's own selected-objects
    QLabel line for archer A then ends "[P1, X51, Y55]" (it read "X50" before), and the drawn
    outline moves with it (archer A's ring leaves (50,55) for (51,55), the units_strong bounds
    change); Undo puts the line, the ring and the bounds back."""
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    gen = conftest.load_verify_module("gen_trigger_fixture")
    window = _held_task_object_window(gen)
    try:
        view = window.map_view
        _select(window, gen.REF_ARCHER_A)
        primed = _prime(window)
        before = _selected_objects_label(window)
        ring_before = _selected_rings(window, gen)
        path_before = _units_strong_path(view)
        assert before.splitlines()[0].endswith("[P1, X50, Y55]")
        assert (50, 55, 50, 55) in ring_before
        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.ShiftModifier))
        assert window._unit_ref_index is primed, "the nudge rebuilt the index instead of patching it"
        assert _selected_objects_label(window).splitlines()[0].endswith("[P1, X51, Y55]")
        rings = _selected_rings(window, gen)
        assert (51, 55, 51, 55) in rings and (50, 55, 50, 55) not in rings
        # A one-tile nudge stays inside the rings' union bounds, so compare the drawn path itself.
        assert _units_strong_path(view) != path_before, "the drawn outline did not follow the nudge"
        window.undo()
        assert _selected_objects_label(window) == before
        assert _selected_rings(window, gen) == ring_before
        assert _units_strong_path(view) == path_before
    finally:
        conftest.close_window(window)
