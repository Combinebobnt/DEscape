"""Units mode: Rotate, Cycle Variant and gate orientation."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import QEvent, Qt
from unit_edit_viewer_helpers import (
    _GATE_E,
    _GATE_NE,
    _GATE_SE,
    _REF_ARCHER_P1,
    _REF_HOUSE,
    _REF_TREE_OAK,
    _REF_TREE_PINE,
    _REF_VILLAGER_P1,
    _REF_WALL,
    _close,
    _field,
    _make_gate,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _select,
    _window,
    _window_with_style,
)

from descape import render_iso, render_sloped

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# --- b3: rotate ---------------------------------------------------------------


def test_rotate_turns_the_selected_unit_by_one_stored_frame() -> None:
    window = _window()
    try:
        from descape import unit_rotation

        (entry,) = _select(window, _REF_ARCHER_P1)
        key = (entry.player_id, entry.unit.reference_id)
        before = entry.unit.rotation

        window.on_unit_rotate(1)

        after = window.map_view._unit_index.entry_for_key(key).unit.rotation
        assert after == pytest.approx(unit_rotation.rotate_step(before, 16, 1))
        # GH #61: the status line speaks the inspector's facing, not radians.
        expected = unit_rotation.rotation_to_facing(after, 16)
        assert f"to facing {expected}/16" in window.status_log.toPlainText()

        window.undo()
        assert window.map_view._unit_index.entry_for_key(key).unit.rotation == before
    finally:
        _close(window)


def test_rotate_skips_a_wall_and_leaves_it_untouched() -> None:
    """A marquee over a village will always catch walls and doodads. They are
    skipped, not refused -- and the file must not change for them."""
    window = _window()
    try:
        (entry,) = _select(window, _REF_WALL)
        before = entry.unit.rotation

        window.on_unit_rotate(1)

        assert entry.unit.rotation == before
        assert not window.edit_history.is_dirty, "a fully-skipped rotate must record nothing"
        assert "no facing or variant to change" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_rotate_skips_an_inert_const_too() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_HOUSE)
        before = entry.unit.rotation
        window.on_unit_rotate(1)
        assert entry.unit.rotation == before
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_a_mixed_selection_rotates_the_angle_units_and_reports_the_skips() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1, _REF_WALL)
        wall = next(e for e in entries if e.unit.reference_id == _REF_WALL)
        rotatables = [e for e in entries if e.unit.reference_id != _REF_WALL]
        before = {e.unit.reference_id: e.unit.rotation for e in entries}

        window.on_unit_rotate(1)

        assert wall.unit.rotation == before[_REF_WALL]
        for entry in rotatables:
            assert entry.unit.rotation != before[entry.unit.reference_id]
        assert "1 skipped" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_group_rotate_is_a_single_undo_record() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1)
        before = [e.unit.rotation for e in entries]

        window.on_unit_rotate(1)
        assert [e.unit.rotation for e in entries] != before

        window.undo()
        assert [e.unit.rotation for e in entries] == before
    finally:
        _close(window)


def test_a_coarse_rotate_turns_a_quarter_of_the_stored_frames() -> None:
    """angle_count 16, so a quarter turn is 4 whole frames -- not an exact
    90 degrees written raw."""
    window = _window()
    try:
        from descape import unit_rotation

        (entry,) = _select(window, _REF_ARCHER_P1)
        before = entry.unit.rotation

        window.on_unit_rotate_coarse(1)

        assert entry.unit.rotation == pytest.approx(unit_rotation.rotate_step(before, 16, 4))
    finally:
        _close(window)


def test_rotate_does_nothing_outside_units_mode() -> None:
    """The keybind is a QAction shortcut, so it is live in every mode --
    unlike the arrow-key nudge, which MapView only injects in Units mode."""
    window = _window()
    try:
        (entry,) = _select(window, _REF_ARCHER_P1)
        before = entry.unit.rotation
        window.mode_combo.setCurrentText("Terrain")

        window.on_unit_rotate(1)

        assert entry.unit.rotation == before
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_the_rotate_actions_need_a_selection() -> None:
    window = _window()
    try:
        assert all(not a.isEnabled() for a in window._rotate_actions)
        _select(window, _REF_ARCHER_P1)
        assert all(a.isEnabled() for a in window._rotate_actions)
    finally:
        _close(window)


# --- Cycle Variant -------------------------------------------------------------


def test_cycle_variant_steps_the_selected_tree_and_undoes() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        key = (entry.player_id, entry.unit.reference_id)

        window.on_unit_variant(1)

        assert window.map_view._unit_index.entry_for_key(key).unit.rotation == 8.0
        assert "Cycled Tree Oak to variant 8/42" in window.status_log.toPlainText()
        window.undo()
        assert window.map_view._unit_index.entry_for_key(key).unit.rotation == 7.0
    finally:
        _close(window)


def test_cycle_variant_changes_the_resolved_frame() -> None:
    from descape import unit_rotation, unit_variant

    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        const = entry.unit.unit_const
        args = (unit_rotation.angle_count_for(const), unit_variant.variant_count_for(const))
        before = unit_variant.variant_of(entry.unit.rotation, *args)
        window.on_unit_variant(-1)
        assert unit_variant.variant_of(entry.unit.rotation, *args) == before - 1
    finally:
        _close(window)


def test_a_mixed_variant_selection_cycles_the_tree_and_reports_the_skips() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_TREE_OAK, _REF_WALL, _REF_ARCHER_P1)
        before = {e.unit.reference_id: e.unit.rotation for e in entries}

        window.on_unit_variant(1)

        after = {e.unit.reference_id: e.unit.rotation for e in entries}
        assert after[_REF_TREE_OAK] != before[_REF_TREE_OAK]
        assert after[_REF_WALL] == before[_REF_WALL]
        assert after[_REF_ARCHER_P1] == before[_REF_ARCHER_P1]
        assert "2 skipped (this object has no graphic variants)" in window.status_log.toPlainText()
        window.undo()
        assert {e.unit.reference_id: e.unit.rotation for e in entries} == before
    finally:
        _close(window)


def test_a_fully_skipped_variant_cycle_records_nothing() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_WALL)
        before = entry.unit.rotation
        window.on_unit_variant(1)
        assert entry.unit.rotation == before
        assert not window.edit_history.is_dirty
        assert "no graphic variants" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_randomize_variant_is_seedable_and_always_changes() -> None:
    import random

    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        window._variant_rng = random.Random(5)
        expected = random.Random(5).randrange(41)
        expected = float(expected + 1 if expected >= 7 else expected)

        window.on_unit_variant(randomize=True)

        assert entry.unit.rotation == expected
        assert entry.unit.rotation != 7.0
        assert "Randomized Tree Oak to variant" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_group_randomize_is_one_record_that_undoes_and_redoes_exactly() -> None:
    import random

    window = _window()
    try:
        entries = _select(window, _REF_TREE_OAK, _REF_TREE_PINE)
        before = {e.unit.reference_id: e.unit.rotation for e in entries}
        records_before = len(window.edit_history.records)
        window._variant_rng = random.Random(5)

        window.on_unit_variant(randomize=True)

        randomized = {e.unit.reference_id: e.unit.rotation for e in entries}
        assert all(randomized[ref] != before[ref] for ref in before)
        assert len(window.edit_history.records) == records_before + 1

        window.undo()
        assert {e.unit.reference_id: e.unit.rotation for e in entries} == before
        window.redo()
        assert {e.unit.reference_id: e.unit.rotation for e in entries} == randomized
    finally:
        _close(window)


def test_cycle_variant_does_nothing_outside_units_mode() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        window.mode_combo.setCurrentText("Terrain")
        window.on_unit_variant(1)
        assert entry.unit.rotation == 7.0
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_the_variant_actions_need_a_selection() -> None:
    window = _window()
    try:
        assert len(window._variant_actions) == 3
        assert all(not a.isEnabled() for a in window._variant_actions)
        _select(window, _REF_TREE_OAK)
        assert all(a.isEnabled() for a in window._variant_actions)
    finally:
        _close(window)


# --- GH #123: Rotate and the Rotation field cycle a tree's variant -----------


def _variant(unit) -> int:
    from descape import unit_rotation, unit_variant

    const = unit.unit_const
    count = unit_variant.variant_count_for(const)
    return unit_variant.variant_of(unit.rotation, unit_rotation.angle_count_for(const), count)


@pytest.mark.parametrize("direction", [1, -1])
def test_a_fine_rotate_on_a_tree_matches_the_cycle_variant_step(direction) -> None:
    """Rotate ↻ (+1) == Next Variant (+1), so `.` and `'` agree on a tree."""
    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        window.on_unit_variant(direction)
        via_cycle = entry.unit.rotation
        window.undo()

        window.on_unit_rotate(direction)

        assert entry.unit.rotation == via_cycle
        assert entry.unit.rotation == 7.0 + direction
        assert f"Cycled Tree Oak to variant {7 + direction}/42" in window.status_log.toPlainText()
        window.undo()
        assert entry.unit.rotation == 7.0
    finally:
        _close(window)


def test_a_fine_rotate_wraps_at_the_last_variant() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        entry.unit.rotation = 41.0
        window.on_unit_rotate(1)
        assert entry.unit.rotation == 0.0
    finally:
        _close(window)


def test_rotate_writes_a_tree_variant_as_a_literal_integer() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_TREE_OAK, _REF_TREE_PINE)
        for _ in range(3):
            window.on_unit_rotate_coarse(1)
            window.on_unit_rotate(-1)
        for entry in entries:
            assert isinstance(entry.unit.rotation, float)
            assert entry.unit.rotation == int(entry.unit.rotation)
            assert 0 <= entry.unit.rotation < 42
    finally:
        _close(window)


def test_a_coarse_rotate_steps_each_tree_by_its_own_quarter() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        entries = _select(window, _REF_TREE_OAK, _REF_TREE_PINE)
        by_ref = {e.unit.reference_id: e.unit for e in entries}
        oak_before, pine_before = _variant(by_ref[_REF_TREE_OAK]), _variant(by_ref[_REF_TREE_PINE])
        records_before = len(window.edit_history.records)

        window.on_unit_rotate_coarse(1)

        assert _variant(by_ref[_REF_TREE_OAK]) == (oak_before + unit_rotation.quarter_turn_steps(42)) % 42
        assert _variant(by_ref[_REF_TREE_PINE]) == (pine_before + unit_rotation.quarter_turn_steps(27)) % 27
        assert len(window.edit_history.records) == records_before + 1
    finally:
        _close(window)


def test_rotate_on_an_archer_tree_and_wall_turns_cycles_and_skips_in_one_record() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_TREE_OAK, _REF_WALL)
        by_ref = {e.unit.reference_id: e.unit for e in entries}
        before = {ref: unit.rotation for ref, unit in by_ref.items()}
        records_before = len(window.edit_history.records)

        window.on_unit_rotate(1)

        assert by_ref[_REF_ARCHER_P1].rotation == pytest.approx(unit_rotation.rotate_step(before[_REF_ARCHER_P1], 16, 1))
        assert by_ref[_REF_TREE_OAK].rotation == 8.0
        assert by_ref[_REF_WALL].rotation == before[_REF_WALL]
        assert len(window.edit_history.records) == records_before + 1
        status = window.status_log.toPlainText()
        assert "Rotated 1 unit, cycled 1 object's variant" in status
        assert "1 skipped (no facing or variant to change)" in status

        window.undo()
        assert {ref: unit.rotation for ref, unit in by_ref.items()} == before
    finally:
        _close(window)


def test_typing_a_variant_on_a_single_tree_is_one_undoable_record() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        spin = window.units_panel.unit_field_editors["rotation"]
        assert spin.isVisibleTo(window.units_panel.unit_inspector_grid)
        assert (spin.maximum(), spin.value()) == (41, 7)
        records_before = len(window.edit_history.records)

        spin.setValue(30)

        assert entry.unit.rotation == 30.0
        assert len(window.edit_history.records) == records_before + 1
        assert spin.value() == 30
        window.undo()
        assert entry.unit.rotation == 7.0
    finally:
        _close(window)


def test_typing_a_variant_on_a_tree_group_sets_both_in_one_record() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_TREE_OAK, _REF_TREE_PINE)
        before = {e.unit.reference_id: e.unit.rotation for e in entries}
        spin = window.units_panel.unit_field_editors["rotation"]
        assert spin.maximum() == 26
        records_before = len(window.edit_history.records)

        spin.setValue(3)

        assert [e.unit.rotation for e in entries] == [3.0, 3.0]
        assert len(window.edit_history.records) == records_before + 1
        assert "Set variant on 2 units" in [r.label for r in window.edit_history.records][-1]
        window.undo()
        assert {e.unit.reference_id: e.unit.rotation for e in entries} == before
    finally:
        _close(window)


def test_typing_the_current_variant_records_nothing() -> None:
    window = _window()
    try:
        _select(window, _REF_TREE_OAK)
        window._on_unit_field_changed(_field("rotation"), 7)
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


# --- b3: gate orientation cycling --------------------------------------------
#
# The fixture holds no gate, so each of these re-points the GAIA wall at a
# stone gate const and its own anchor first. Every path under test reads the
# live unit's fields, and _after_unit_mutation() is the same index rebuild a
# real edit already runs, so the index agrees with the swapped const.


def _occupied(window, unit):
    from descape import render_units

    mm = window.scenario.map_manager
    return render_units.unit_occupied_tiles(unit, mm.map_width, mm.map_height)


def test_rotate_cycles_a_selected_gate_to_its_next_orientation() -> None:
    """A gate's orientation lives in its const, so "rotating" one swaps the
    const and re-anchors. The footprint really changes: (4, 1) along x becomes
    the sparse 6-tile diagonal, which is asserted on tiles rather than pixels.
    """
    window = _window()
    try:
        entry = _make_gate(window)
        before = _occupied(window, entry.unit)

        window.on_unit_rotate(1)

        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == (_GATE_E, 10.0, 7.0)
        after = _occupied(window, entry.unit)
        assert len(before) == 4 and len(after) == 6
        assert before != after
    finally:
        _close(window)


def test_four_gate_rotates_come_back_to_the_starting_orientation() -> None:
    """GH #12: the four siblings form a cycle, and each re-anchor inverts
    exactly, so no drift accumulates over a full turn."""
    window = _window()
    try:
        entry = _make_gate(window)
        start = (entry.unit.unit_const, entry.unit.x, entry.unit.y)
        start_tiles = _occupied(window, entry.unit)

        for _ in range(4):
            window.on_unit_rotate(1)

        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == start
        assert _occupied(window, entry.unit) == start_tiles
        assert "map edge" not in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_coarse_rotate_cycles_a_gate_by_two_orientations() -> None:
    """One gate step is 45 degrees, so a quarter turn is two of them: ne
    lands on se, not on the diagonal in between."""
    window = _window()
    try:
        entry = _make_gate(window)

        window.on_unit_rotate_coarse(1)

        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == (_GATE_SE, 8.5, 7.0)
    finally:
        _close(window)


def test_a_gate_and_an_archer_rotate_together_in_one_undo_record() -> None:
    window = _window()
    try:
        _make_gate(window)
        entries = _select(window, _REF_WALL, _REF_ARCHER_P1)
        gate = next(e for e in entries if e.unit.reference_id == _REF_WALL)
        archer = next(e for e in entries if e.unit.reference_id == _REF_ARCHER_P1)
        before = (gate.unit.unit_const, archer.unit.rotation)

        window.on_unit_rotate(1)
        assert gate.unit.unit_const == _GATE_E
        assert archer.unit.rotation != before[1]

        window.undo()
        assert (gate.unit.unit_const, archer.unit.rotation) == before
        assert (gate.unit.x, gate.unit.y) == (10.0, 5.5)
    finally:
        _close(window)


def test_a_gate_with_no_room_at_the_map_edge_refuses_and_says_so() -> None:
    """Cycling ne to the diagonal grows the footprint three tiles down the y
    axis. Against the map's far edge that would write a hanging footprint, so
    the viewer, the layer that knows the map's size, skips it."""
    window = _window()
    try:
        entry = _make_gate(window, x=10.0, y=117.5)
        before = (entry.unit.unit_const, entry.unit.x, entry.unit.y)

        window.on_unit_rotate(1)

        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == before
        assert not window.edit_history.is_dirty
        assert "map edge" in window.status_log.toPlainText()
    finally:
        _close(window)


# --- GH #61: a gate's orientation typed into the Rotation field ---------------


def test_typing_2_on_a_gate_matches_two_rotate_presses_in_one_undo_step() -> None:
    """Index order is Rotate's step order, so typing n is n presses from 0."""
    window = _window()
    try:
        entry = _make_gate(window)
        start = (entry.unit.unit_const, entry.unit.x, entry.unit.y)
        spin = window.units_panel.unit_field_editors["rotation"]
        assert spin.isVisibleTo(window.units_panel.unit_inspector_grid)
        assert (spin.minimum(), spin.maximum(), spin.value()) == (0, 3, 0)
        records_before = len(window.edit_history.records)

        spin.setValue(2)

        typed = (entry.unit.unit_const, entry.unit.x, entry.unit.y)
        assert typed == (_GATE_SE, 8.5, 7.0)
        assert len(window.edit_history.records) == records_before + 1
        assert window.edit_history.records[-1].label == "Set unit Rotation"
        assert "Orientation 2 of 4: NW to SE" in window.units_panel.unit_field_editors["rotation"].toolTip()
        window.undo()
        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == start

        window.on_unit_rotate(1)
        window.on_unit_rotate(1)
        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == typed
    finally:
        _close(window)


def test_typing_a_gates_current_orientation_records_nothing() -> None:
    window = _window()
    try:
        _make_gate(window)
        window._on_unit_field_changed(_field("rotation"), 0)
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_a_group_of_gates_at_different_orientations_types_one_orientation() -> None:
    window = _window()
    try:
        _make_gate(window)
        (archer,) = _select(window, _REF_ARCHER_P1)
        # A 4x4 diagonal sibling, anchored at tile + span/2 like every corpus gate.
        archer.unit.unit_const, archer.unit.x, archer.unit.y = _GATE_E, 20.0, 20.0
        window._after_unit_mutation()
        entries = _select(window, _REF_WALL, _REF_ARCHER_P1)
        before = [(e.unit.unit_const, e.unit.x, e.unit.y) for e in entries]
        spin = window.units_panel.unit_field_editors["rotation"]
        assert spin.text() == "(mixed)"
        records_before = len(window.edit_history.records)

        window._on_unit_field_changed(_field("rotation"), 2)

        by_ref = {e.unit.reference_id: (e.unit.unit_const, e.unit.x, e.unit.y) for e in entries}
        assert by_ref[_REF_WALL] == (_GATE_SE, 8.5, 7.0)
        assert by_ref[_REF_ARCHER_P1] == (_GATE_SE, 18.5, 20.0)
        assert len(window.edit_history.records) == records_before + 1
        window.undo()
        assert [(e.unit.unit_const, e.unit.x, e.unit.y) for e in entries] == before
    finally:
        _close(window)


def test_a_typed_facing_on_a_gate_and_archer_still_skips_the_gate() -> None:
    window = _window()
    try:
        _make_gate(window)
        entries = _select(window, _REF_WALL, _REF_ARCHER_P1)
        gate = next(e for e in entries if e.unit.reference_id == _REF_WALL)
        archer = next(e for e in entries if e.unit.reference_id == _REF_ARCHER_P1)
        archer_before = archer.unit.rotation

        window._on_unit_field_changed(_field("rotation"), 2)

        assert archer.unit.rotation != archer_before
        assert (gate.unit.unit_const, gate.unit.x, gate.unit.y) == (_GATE_NE, 10.0, 5.5)
    finally:
        _close(window)


def test_a_typed_gate_orientation_off_the_map_edge_refuses_and_resets_the_field() -> None:
    window = _window()
    try:
        entry = _make_gate(window, x=10.0, y=117.5)
        before = (entry.unit.unit_const, entry.unit.x, entry.unit.y)
        spin = window.units_panel.unit_field_editors["rotation"]

        spin.setValue(1)

        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == before
        assert not window.edit_history.is_dirty
        assert "map edge" in window.status_log.toPlainText()
        assert window.units_panel.unit_field_editors["rotation"].value() == 0
    finally:
        _close(window)


@pytest.fixture
def gate_art(tmp_path, monkeypatch):
    """A synthetic install drawing each stone gate orientation (64/88/659/667)
    as its own solid colour, so a stale orientation is a pixel mismatch."""
    from descape import asset_source, unit_sprites

    from test_unit_sprites import build_sld

    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    table = {}
    for i, const in enumerate((64, 88, 659, 667)):
        name = f"t_gate_{const}_x1"
        art = build_sld(1, canvas=unit_sprites.NATIVE_TILE_W, playercolor=False, colour_base=5 * i)
        (graphics / f"{name}.sld").write_bytes(art)
        table[const] = {"graphic_id": const, "file_name": name, "angle_count": 1,
                        "mirroring_mode": 0, "frame_count": 1}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _fresh_canvas(window, style: str, sprites: bool = True) -> np.ndarray:
    canvas_w, canvas_h = window._cache.canvas_dims(0)
    if style == "Stepped":
        full = render_iso.render_terrain_iso_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    else:
        full = render_sloped.render_terrain_sloped_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    return full[:canvas_h, :canvas_w]


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_four_gate_rotates_repaint_like_a_fresh_render_with_sprites(style, gate_art) -> None:
    """GH #78 step 12 and GH #12 step 3 outside Flat: every orientation swap
    changes the gate's footprint and art, and must leave no stale fragment."""
    from descape.render_cache import IsoChunkCache, SlopedChunkCache

    window = _window_with_style(style)
    try:
        assert isinstance(window._cache, IsoChunkCache if style == "Stepped" else SlopedChunkCache)
        assert window.show_sprites_action.isChecked()
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add gate", [1]):
            gate = model.add(1, _GATE_NE, 60.0, 60.5)
        _select(window, gate.reference_id)
        start_tiles = _occupied(window, gate)
        canvas_w, canvas_h = window._cache.canvas_dims(0)
        start = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(start, _fresh_canvas(window, style))
        assert not np.array_equal(start, _fresh_canvas(window, style, sprites=False)), "the gate drew no sprite"

        previous = start
        for step in range(1, 5):
            window.on_unit_rotate(1)
            now = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
            assert np.array_equal(now, _fresh_canvas(window, style)), f"stale pixels after rotate {step}"
            assert not np.array_equal(now, previous), f"rotate {step} changed nothing on screen"
            previous = now

        assert (gate.unit_const, gate.x, gate.y) == (_GATE_NE, 60.0, 60.5)
        assert _occupied(window, gate) == start_tiles
        assert np.array_equal(previous, start)
    finally:
        _close(window)


def _turned_drag(window, press_tile, end_tile, presses: int) -> None:
    """A real press, move, Rotate key presses and release (GH #280)."""
    view = window.map_view
    start = conftest.polygon_viewport_pos(view, *press_tile)
    end = conftest.polygon_viewport_pos(view, *end_tile)
    view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton))
    view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
    for _ in range(presses):
        window.rotate_cw_action.trigger()
    view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))


def _picked(window, tile):
    view = window.map_view
    return view.pick_unit_at(view.mapToScene(conftest.polygon_viewport_pos(view, *tile).toPoint()))


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_gate_turned_on_a_drag_repaints_and_picks_like_a_fresh_render(style, gate_art) -> None:
    """GH #280, the layer test_unit_drag_preview.py's docstring warns about. A
    drop that captures its old tiles after the turn, or that patches a span
    change into the pick index, leaves stale pixels or dead picks. The canvas is
    rendered in whole right before each step: an evicted chunk would hide one."""
    window = _window_with_style(style)
    try:
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add gate", [1]):
            gate = model.add(1, _GATE_NE, 60.0, 60.5)
            other = model.add(1, 83, 70.5, 60.5)
        window.move_action.setChecked(True)
        _select(window, gate.reference_id)
        canvas_w, canvas_h = window._cache.canvas_dims(0)

        def repainted(step: str) -> None:
            now = window._cache.render_rect(0, 0, canvas_w, canvas_h)
            assert np.array_equal(now, _fresh_canvas(window, style)), f"stale pixels after {step}"

        old_tiles = _occupied(window, gate)
        repainted("the set-up")

        # Two steps: (4, 1) to (1, 4), so no tile of the old span survives but the drop's own.
        _turned_drag(window, (60, 60), (80, 40), 2)
        assert gate.unit_const == _GATE_SE
        new_tiles = _occupied(window, gate)
        assert not set(new_tiles) & set(old_tiles)
        repainted("the single drop")
        assert _picked(window, new_tiles[1]).unit is gate
        assert all(_picked(window, tile) is None for tile in old_tiles)

        window.undo()
        assert gate.unit_const == _GATE_NE and _occupied(window, gate) == old_tiles
        repainted("the single undo")
        assert _picked(window, new_tiles[1]) is None
        assert _picked(window, old_tiles[1]).unit is gate

        _select(window, gate.reference_id, other.reference_id)
        _turned_drag(window, (70, 60), (80, 50), 2)
        assert gate.unit_const == _GATE_SE
        group_tiles = _occupied(window, gate)
        assert not set(group_tiles) & set(old_tiles)
        repainted("the group drop")
        assert _picked(window, group_tiles[1]).unit is gate
        assert all(_picked(window, tile) is None for tile in old_tiles)

        window.undo()
        assert gate.unit_const == _GATE_NE
        repainted("the group undo")
        assert _picked(window, old_tiles[1]).unit is gate
    finally:
        _close(window)
