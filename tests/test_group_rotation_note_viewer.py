"""GH #71's group Rotation note, driven through a real offscreen ViewerWindow:
the selection reconciliation that feeds UnitsPanel.show_group(), the Keybinds
tab's rebind path, and the Rotate action on the same selection. The note's
wording per mode is pinned panel-level in tests/test_units_panel.py; since
GH #198 it leads the Rotation caption's tooltip rather than a label under it.

Uses the units_120x120 fixture, whose GAIA wall (102) or oak (100) is re-pointed
at a stone gate const where a test needs one, as tests/test_unit_edit_viewer.py's
gate tests do: the fixture holds no gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_REF_OAK = 100
_REF_PINE = 101
_REF_WALL = 102
_REF_ARCHER_P1 = 201
_REF_ARCHER_P2 = 300
_GATE_NE, _GATE_E = 64, 659
_REBOUND_KEY = "F9"


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    return window


def _select(window, *reference_ids):
    """Through _refresh_selection_view(), the reconciliation every selection path ends in."""
    index = window.map_view._unit_index
    entries = [e for e in index.entries if e.unit.reference_id in reference_ids]
    assert len(entries) == len(reference_ids), "fixture unit missing from the index"
    window._selection = [(e.player_id, e.unit.reference_id) for e in entries]
    window._refresh_selection_view()
    return entries


def _make_gate(window, reference_id: int, x: float, y: float):
    """Re-points a fixture unit at the stone gate (ne) on free tiles."""
    unit = next(u for u in window.scenario.unit_manager.get_all_units() if u.reference_id == reference_id)
    unit.unit_const, unit.x, unit.y = _GATE_NE, x, y
    window._after_unit_mutation()
    return unit


def _note(window) -> str:
    """The group note leading the Rotation caption's tooltip, "" with the caveat alone."""
    from descape.units_panel import _ROTATION_TOOLTIP

    tip = window.units_panel.unit_rotation_label.toolTip()
    assert tip.endswith(_ROTATION_TOOLTIP)
    return tip.removesuffix(_ROTATION_TOOLTIP).removesuffix("\n\n")


def test_a_rebound_rotate_key_is_named_in_the_group_note_after_a_reselect() -> None:
    """Pass conditions:
    - two archers and an oak, selected in the viewer: the caption tooltip's
      note leads "Typing a Rotation turns the 2 selected units with a facing.
      The other 1 of 3", and names Rotate (. and ,) stepping the trees;
    - Rotate Unit Clockwise rebound to F9 in the Keybinds tab persists and
      re-binds the live action;
    - the same group reselected: the note names Rotate (F9 and ,)."""
    from PyQt5.QtGui import QKeySequence

    from descape import settings
    from descape.viewer import SettingsDialog

    window = _window()
    try:
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2, _REF_OAK)
        note = _note(window)
        assert note.startswith(
            "Typing a Rotation turns the 2 selected units with a facing. "
            "The other 1 of 3 (trees, walls, gates, buildings, scenery) keep theirs."
        )
        assert note.endswith(" Rotate (. and ,) still steps trees and scenery through their variants.")

        assert settings.keybind_holder(_REBOUND_KEY) is None, "the rebind must not clear another action"
        dialog = SettingsDialog(window)
        try:
            dialog._keybind_edits["unit_rotate_cw"].setKeySequence(QKeySequence(_REBOUND_KEY))
        finally:
            dialog.close()
        assert settings.get_keybind("unit_rotate_cw") == _REBOUND_KEY
        assert window.rotate_cw_action.shortcut() == QKeySequence(_REBOUND_KEY)

        window._selection = []
        window._refresh_selection_view()
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2, _REF_OAK)
        assert _note(window).endswith(f" Rotate ({_REBOUND_KEY} and ,) still steps trees and scenery through their variants.")
    finally:
        conftest.close_window(window)


def test_rotate_on_a_gate_and_a_wall_turns_the_gate_though_typing_changes_neither() -> None:
    """Pass conditions:
    - a gate and a wall selected: the note reads "Typing a Rotation changes
      none of the 2 selected" and ends naming Rotate (. and ,) turning gates;
    - the Rotate Clockwise action, fired: the gate takes its next orientation
      const (ne to e), the wall's stored index is unchanged, one undo record;
    - undo puts the gate back on its ne const."""
    window = _window()
    try:
        gate = _make_gate(window, _REF_OAK, 30.0, 30.5)
        wall = next(u for u in window.scenario.unit_manager.get_all_units() if u.reference_id == _REF_WALL)
        _select(window, _REF_OAK, _REF_WALL)
        assert _note(window) == (
            "Typing a Rotation changes none of the 2 selected: walls, cliffs, "
            "gates and single-frame objects keep their stored value. "
            "Rotate (. and ,) still turns gates through their four orientations."
        )
        wall_before = wall.rotation
        records = len(window.edit_history.records)

        window.rotate_cw_action.trigger()

        assert gate.unit_const == _GATE_E
        assert wall.rotation == wall_before
        assert len(window.edit_history.records) == records + 1
        window.undo()
        assert gate.unit_const == _GATE_NE
    finally:
        conftest.close_window(window)


def test_an_oak_a_pine_and_a_gate_note_counts_two_variant_objects() -> None:
    """Pass conditions: an oak, a pine and a gate selected in the viewer give
    the caption tooltip this exact note, plural included: "Typing a Rotation
    picks that variant for the 2 selected objects with graphic variants. The
    other 1 of 3 (walls, cliffs, gates, single-frame objects) keep theirs.
    Rotate (. and ,) still turns gates through their four orientations." """
    window = _window()
    try:
        _make_gate(window, _REF_WALL, 30.0, 30.5)
        _select(window, _REF_OAK, _REF_PINE, _REF_WALL)
        assert _note(window) == (
            "Typing a Rotation picks that variant for the 2 selected objects with graphic variants. "
            "The other 1 of 3 (walls, cliffs, gates, single-frame objects) keep theirs. "
            "Rotate (. and ,) still turns gates through their four orientations."
        )
    finally:
        conftest.close_window(window)
