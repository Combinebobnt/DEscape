"""GH #286: the Eraser's Cliffs target, Delete, Cut Units and Find's Delete
re-cap the cliff run ends they leave, as the Cliff tool's right-click does
(GH #247), through the same cliff_chain.erase_rewrites().

Driven through a real offscreen ViewerWindow on test_cliff_tool_viewer's
five-run: five 3x3 GAIA pieces on x 9-11, frames 18, 2, 1, 2, 19 from the
north; piece 3 sits on y 15-17. Removing it leaves piece 2 as the run's
north half's south end (cap 19) and piece 4 as the south half's north end
(cap 18).
"""

from __future__ import annotations

import pytest
from PyQt5.QtCore import Qt
from test_cliff_tool_viewer import (
    _FIVE_RUN,
    GAIA_PLAYER_ID,
    _cliffs_on_the_run,
    _close,
    _five_run,
    _gaia,
    _last_status,
    _shown_window,
    _unit_count,
)
from test_eraser_viewer import _tick

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_CAPPED = [(10.5, 18.0, 18), (13.5, 19.0, 19), (19.5, 18.0, 18), (22.5, 19.0, 19)]
_UNCAPPED = [(10.5, 18.0, 18), (13.5, 2.0, 2), (19.5, 2.0, 2), (22.5, 19.0, 19)]
_TREE = 284
_STONE_WALL = 117


def _eraser(window, mode: str = "Terrain", **targets) -> None:
    window.mode_combo.setCurrentText(mode)
    window.eraser_action.setChecked(True)
    window.brush_size_spin.setValue(1)
    assert window._current_tool == "eraser"
    _tick(window, **targets)


def _erase(window, *paths) -> None:
    window.on_edit_stroke_start()
    for path in paths:
        window.on_edit_stroke_tiles(path, 0)
    window.on_edit_stroke_end()


def _plant(window, x: float, y: float, const: int = _TREE):
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Plant", [GAIA_PLAYER_ID]):
        return model.add(GAIA_PLAYER_ID, const, x, y)


def _key(unit, player: int = GAIA_PLAYER_ID):
    return (player, unit.reference_id)


def _units_mode(window) -> None:
    window.mode_combo.setCurrentText("Units")
    assert window.map_view._unit_index is not None


def _delete(window, unit) -> None:
    _units_mode(window)
    window._selection = [_key(unit)]
    window._refresh_selection_view()
    window.on_unit_delete(Qt.NoModifier)


def _the_next_unit_edit_works(window) -> None:
    before = _unit_count(window)
    _plant(window, 60.5, 60.5)
    assert _unit_count(window) == before + 1
    window.undo()
    assert _unit_count(window) == before


def _counting(monkeypatch, window, name: str) -> list:
    calls = []
    real = getattr(window, name)

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(window, name, spy)
    return calls


# -- the Eraser -------------------------------------------------------------


@pytest.mark.parametrize("mode", ["Terrain", "Units"])
def test_an_eraser_stroke_with_cliffs_ticked_caps_the_ends_it_leaves_in_one_record(mode):
    window = _shown_window()
    try:
        run = _five_run(window)
        _eraser(window, mode, cliffs=True)
        records = len(window.edit_history.records)

        _erase(window, [(10, 16)])

        assert run[2] not in _gaia(window)
        assert _cliffs_on_the_run(window) == _CAPPED
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == "Erase object"
        assert window.unit_edits._pending is None
        assert _last_status(window) == "Erased 1 object(s), 2 cliff(s) reshaped to end caps"

        window.undo()
        assert _cliffs_on_the_run(window) == _FIVE_RUN
        window.redo()
        assert _cliffs_on_the_run(window) == _CAPPED
    finally:
        _close(window)


def test_the_cliffs_target_tooltip_names_the_re_cap():
    window = _shown_window()
    try:
        assert window.eraser_target_actions["cliffs"].toolTip() == (
            "Off, no other target erases a cliff: erasing one piece breaks its chain. "
            "On, a run left with one neighbour gets its end cap back."
        )
    finally:
        _close(window)


def test_an_eraser_stroke_with_cliffs_unticked_erases_no_cliff():
    window = _shown_window()
    try:
        _five_run(window)
        _eraser(window, units=True, buildings=True, trees=True, gaia=True, invisible=True)
        records = len(window.edit_history.records)
        _erase(window, [(10, 16)], [(10, 19)])
        assert _cliffs_on_the_run(window) == _FIVE_RUN
        assert len(window.edit_history.records) == records
    finally:
        _close(window)


def test_an_eraser_stroke_that_erases_no_cliff_never_scans_or_reshapes(monkeypatch):
    """Not an _after_unit_mutation argument check: in Flat, _if_spliceable()
    can return None on its own."""
    window = _shown_window()
    try:
        _five_run(window)
        tree = _plant(window, 13.5, 16.5)
        _eraser(window, cliffs=True, trees=True)
        scans = _counting(monkeypatch, window, "_cliff_existing")
        rewrites = _counting(monkeypatch, window, "_rewrite_cliff_ends")
        seen = []
        real_commit = window._commit_erase_stroke
        monkeypatch.setattr(window, "_commit_erase_stroke", lambda s: (seen.append(s), real_commit(s))[1])

        _erase(window, [(13, 16)])

        assert tree not in _gaia(window)
        assert _cliffs_on_the_run(window) == _FIVE_RUN
        assert _last_status(window) == "Erased 1 object(s)"
        assert seen and seen[0].cliffs is None
        assert scans == [] and rewrites == []
    finally:
        _close(window)


def test_a_mid_drag_undo_commits_the_eraser_s_removals_and_reshapes_no_end():
    window = _shown_window()
    try:
        _five_run(window)
        _plant(window, 60.5, 60.5)
        _eraser(window, cliffs=True)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(10, 16)], 0)
        assert window._erase.erased == 1 and window._erase.cliffs is not None
        window.undo()
        records = len(window.edit_history.records)

        window.on_edit_stroke_end()

        assert window.edit_history.peek_undo().label == "Erase object"
        assert len(window.edit_history.records) == records
        assert window.unit_edits._pending is None
        assert _cliffs_on_the_run(window) == _FIVE_RUN
        assert _last_status(window) == (
            "Erased 1 object(s); the map's units changed during the drag, so no cliff ends were reshaped"
        )
        _the_next_unit_edit_works(window)
    finally:
        _close(window)


def test_cliffs_and_terrain_overlay_in_one_drag_are_one_composite_record():
    window = _shown_window()
    try:
        _five_run(window)
        _eraser(window, cliffs=True, overlay=True)
        mm = window.scenario.map_manager
        terrain = mm.terrain
        index = 16 * mm.map_width + 13
        terrain[index].layer = 3
        records = len(window.edit_history.records)

        _erase(window, [(10, 16)], [(13, 16)])

        assert terrain[index].layer == -1
        assert _cliffs_on_the_run(window) == _CAPPED
        assert len(window.edit_history.records) == records + 1
        assert set(window.edit_history.peek_undo().kinds()) == {"unit", "tile"}
        window.undo()
        assert terrain[index].layer == 3
        assert _cliffs_on_the_run(window) == _FIVE_RUN
        window.redo()
        assert terrain[index].layer == -1
        assert _cliffs_on_the_run(window) == _CAPPED
    finally:
        _close(window)


def test_a_cliff_note_survives_a_wall_note_from_the_same_release(monkeypatch):
    from descape import cliff_chain, wall_run

    window = _shown_window()
    try:
        _five_run(window)
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [1]):
            model.add(1, _STONE_WALL, 13.5, 16.5, rotation=2.0)

        def boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(cliff_chain, "erase_rewrites", boom)
        monkeypatch.setattr(wall_run, "plan_removal_rewrites", boom)
        _eraser(window, "Units", cliffs=True, buildings=True)

        _erase(window, [(10, 16)], [(13, 16)])

        assert _last_status(window) == (
            "Erased 2 object(s); reshaping the cliff ends stopped: RuntimeError: boom; "
            "reshaping the walls beside the erased ones stopped: RuntimeError: boom"
        )
        assert window.unit_edits._pending is None
        window.undo()
        assert _cliffs_on_the_run(window) == _FIVE_RUN
    finally:
        _close(window)


# -- Delete, Cut Units and Find's Delete ------------------------------------


def test_deleting_a_cliff_caps_the_ends_it_leaves_in_one_record():
    window = _shown_window()
    try:
        run = _five_run(window)
        records = len(window.edit_history.records)

        _delete(window, run[2])

        assert run[2] not in _gaia(window)
        assert _cliffs_on_the_run(window) == _CAPPED
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == "Delete unit"
        assert _last_status(window).endswith("; 2 cliff(s) reshaped to end caps")
        window.undo()
        assert _cliffs_on_the_run(window) == _FIVE_RUN
        window.redo()
        assert _cliffs_on_the_run(window) == _CAPPED
    finally:
        _close(window)


def test_deleting_a_lone_tree_never_scans_the_cliffs_and_still_splices(monkeypatch):
    window = _shown_window()
    try:
        _five_run(window)
        tree = _plant(window, 60.5, 60.5)
        _units_mode(window)
        scans = _counting(monkeypatch, window, "_cliff_existing")
        refreshes = []
        real = window._after_unit_mutation
        monkeypatch.setattr(
            window, "_after_unit_mutation", lambda changed=None, **kw: (refreshes.append(changed), real(changed, **kw))[1]
        )

        _delete(window, tree)

        assert tree not in _gaia(window)
        assert scans == []
        assert len(refreshes) == 1 and refreshes[0], "expected one spliced refresh"
        assert "cliff" not in _last_status(window)
    finally:
        _close(window)


def test_a_reshaped_cliff_of_another_owner_is_declared_so_undo_restores_its_frame():
    window = _shown_window()
    try:
        run = _five_run(window)
        piece4 = run[3]
        assert window._reassign_units(window._find_entries([_key(piece4)]), 1)[0] == 1
        assert any(u is piece4 for u in window.scenario.unit_manager.units[1])

        _delete(window, run[2])

        assert (piece4.rotation, piece4.initial_animation_frame) == (18.0, 18)
        window.undo()
        assert (piece4.rotation, piece4.initial_animation_frame) == (2.0, 2)
        window.redo()
        assert (piece4.rotation, piece4.initial_animation_frame) == (18.0, 18)
    finally:
        _close(window)


def test_cut_units_caps_the_ends_and_the_clipboard_holds_the_piece_as_it_was():
    window = _shown_window()
    try:
        run = _five_run(window)
        _units_mode(window)
        window._selection = [_key(run[2])]
        window._refresh_selection_view()

        window.cut_units()

        assert run[2] not in _gaia(window)
        assert _cliffs_on_the_run(window) == _CAPPED
        assert window.edit_history.peek_undo().label == "Cut unit"
        assert _last_status(window).endswith("; 2 cliff(s) reshaped to end caps")
        (copied,) = window._clipboard_history.active_block.units
        assert (copied.unit_const, copied.rotation, copied.initial_animation_frame) == (264, 1.0, 1)
    finally:
        _close(window)


@pytest.mark.parametrize("show_gaia", [True, False], ids=["shown", "hidden"])
def test_finds_delete_caps_the_ends_hidden_survivors_included(show_gaia):
    window = _shown_window()
    try:
        run = _five_run(window)
        window.show_gaia_action.setChecked(show_gaia)
        records = len(window.edit_history.records)

        assert window._find_delete_keys([_key(run[2])]) == (1, 0, 0)

        assert _cliffs_on_the_run(window) == _CAPPED
        assert len(window.edit_history.records) == records + 1
        assert _last_status(window).endswith("; 2 cliff(s) reshaped to end caps")
        window.undo()
        assert _cliffs_on_the_run(window) == _FIVE_RUN
    finally:
        _close(window)


def test_a_cliff_frame_write_that_raises_still_records_the_delete(monkeypatch):
    from descape.unit_model import UnitEditModel

    window = _shown_window()
    try:
        run = _five_run(window)
        records = len(window.edit_history.records)
        before = _unit_count(window)

        def refuse(self, unit, frame):
            raise ValueError("refused")

        monkeypatch.setattr(UnitEditModel, "set_cliff_frame", refuse)
        _delete(window, run[2])

        assert _unit_count(window) == before - 1
        assert len(window.edit_history.records) == records + 1
        assert window.unit_edits._pending is None
        assert _cliffs_on_the_run(window) == _UNCAPPED
        assert _last_status(window).endswith("; reshaping the cliff ends stopped: ValueError: refused")
        window.undo()
        assert _cliffs_on_the_run(window) == _FIVE_RUN
        _the_next_unit_edit_works(window)
    finally:
        _close(window)


@pytest.mark.parametrize("verb", ["Delete", "Cut"])
def test_a_plan_that_raises_still_removes_the_cliff_in_one_record(verb, monkeypatch):
    from descape import cliff_chain

    window = _shown_window()
    try:
        run = _five_run(window)
        records = len(window.edit_history.records)

        def boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(cliff_chain, "erase_rewrites", boom)
        _units_mode(window)
        window._selection = [_key(run[2])]
        window._refresh_selection_view()
        if verb == "Delete":
            window.on_unit_delete(Qt.NoModifier)
        else:
            window.cut_units()
            (copied,) = window._clipboard_history.active_block.units
            assert copied.rotation == 1.0

        assert run[2] not in _gaia(window)
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == f"{verb} unit"
        assert _cliffs_on_the_run(window) == _UNCAPPED
        assert _last_status(window).endswith("; reshaping the cliff ends stopped: RuntimeError: boom")
        window.undo()
        assert _cliffs_on_the_run(window) == _FIVE_RUN
    finally:
        _close(window)


def test_deleting_a_cliff_and_a_wall_reshapes_both_neighbours_in_one_record():
    """GH #340's wall reshape and this cliff re-cap from one Delete: one
    record holds both, and one undo restores both."""
    window = _shown_window()
    try:
        run = _five_run(window)
        # Literal, so the file is integer-only (the fixture's GAIA wall 102 is radian).
        next(u for u in _gaia(window) if u.reference_id == 102).rotation = 2.0
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [1]):
            row = {x: model.add(1, _STONE_WALL, x + 0.5, 40.5, rotation=0.0 if 40 < x < 46 else 2.0) for x in range(40, 47)}
        records = len(window.edit_history.records)
        _units_mode(window)
        window._selection = [_key(run[2]), _key(row[43], 1)]
        window._refresh_selection_view()

        window.on_unit_delete(Qt.NoModifier)

        assert _cliffs_on_the_run(window) == _CAPPED
        assert (row[42].rotation, row[44].rotation) == (2.0, 2.0)
        assert len(window.edit_history.records) == records + 1
        last = _last_status(window)
        assert last == "Deleted 2 units, reshaping 2 adjacent; 2 cliff(s) reshaped to end caps", last
        window.undo()
        assert _cliffs_on_the_run(window) == _FIVE_RUN
        assert (row[42].rotation, row[44].rotation) == (0.0, 0.0)
    finally:
        _close(window)
