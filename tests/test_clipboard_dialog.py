"""The clipboard history's Qt half: the dialog itself, and the ViewerWindow
wiring that feeds it. The collection's own rules are covered without Qt in
tests/test_clipboard_history.py.

Same offscreen technique tests/test_fill_tool.py documents; every
ViewerWindow() here must call edit_history.mark_saved() before close().
"""

from __future__ import annotations

import numpy as np
import pytest

from descape.clipboard_dialog import ClipboardHistoryDialog
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN_A, _TERRAIN_B = 2, 15  # BEACH, GRASS_1 -- present in every DE version


def _window():
    window = conftest.terrain_edit_window()
    window._on_tool_selected("select")
    return window


def _copy(window, sx0, sy0, sx1, sy1, terrain_id):
    """Paints a distinct terrain over the rect, selects it and copies, so each
    history entry is a genuinely different block rather than a repeat."""
    window._on_tool_selected("draw")
    window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(terrain_id))
    for y in range(sy0, sy1):
        for x in range(sx0, sx1):
            window.on_edit_stroke_start()
            window.on_edit_stroke_tile(x, y, 0)
            window.on_edit_stroke_end()
    window._on_tool_selected("select")
    window.on_region_selected((sx0, sy0, sx1, sy1))
    window.copy_region()
    return window._clipboard_history.entries[0]


def _rows(count: int):
    return [
        (100 + i, f"entry {i}", 2, 2, i, np.zeros((2, 2, 3), np.uint8)) for i in range(count)
    ]


# -- the dialog on its own ---------------------------------------------------


def _dialog(**callbacks):
    conftest.ensure_qapp()
    return ClipboardHistoryDialog(None, **callbacks)


def test_set_entries_builds_one_row_per_entry_and_stores_ids() -> None:
    from PyQt5.QtCore import Qt

    dialog = _dialog()
    try:
        dialog.set_entries(_rows(3), 101)
        assert dialog.tree.topLevelItemCount() == 3
        ids = [dialog.tree.topLevelItem(r).data(0, Qt.UserRole) for r in range(3)]
        assert ids == [100, 101, 102]
        assert dialog.tree.topLevelItem(1).font(1).bold()
        assert not dialog.tree.topLevelItem(0).font(1).bold()
        assert "entry 1" in dialog.status.text()
    finally:
        dialog.close()


def test_empty_state_label_shows_with_no_rows() -> None:
    dialog = _dialog()
    try:
        dialog.set_entries([], None)
        assert dialog.tree.topLevelItemCount() == 0
        assert dialog.status.text() == dialog._EMPTY
    finally:
        dialog.close()


def test_each_button_reports_the_selected_rows_entry_id() -> None:
    """The discriminating index-vs-id case: the rows are populated so that no
    row's entry_id equals its own index."""
    seen = {}
    dialog = _dialog(
        on_activate=lambda i: seen.__setitem__("activate", i),
        on_delete=lambda i: seen.__setitem__("delete", i),
        on_clear=lambda: seen.__setitem__("clear", True),
        on_rename=lambda i, label: seen.__setitem__("rename", (i, label)),
    )
    try:
        dialog.set_entries(_rows(3), 100)
        dialog.select_entry(102)
        assert dialog.selected_entry_id() == 102
        dialog.activate_button.click()
        assert seen["activate"] == 102
        dialog.delete_button.click()
        assert seen["delete"] == 102
        dialog.clear_button.click()
        assert seen["clear"] is True
    finally:
        dialog.close()


def test_buttons_gate_on_selection_and_on_emptiness() -> None:
    dialog = _dialog()
    try:
        dialog.set_entries([], None)
        assert not dialog.activate_button.isEnabled()
        assert not dialog.rename_button.isEnabled()
        assert not dialog.delete_button.isEnabled()
        assert not dialog.clear_button.isEnabled()

        dialog.set_entries(_rows(2), 100)
        dialog.tree.setCurrentItem(None)
        assert not dialog.activate_button.isEnabled()
        assert not dialog.delete_button.isEnabled()
        assert dialog.clear_button.isEnabled()

        dialog.select_entry(101)
        assert dialog.activate_button.isEnabled()
        assert dialog.rename_button.isEnabled()
        assert dialog.delete_button.isEnabled()
    finally:
        dialog.close()


# -- the window wiring -------------------------------------------------------


def test_the_dialog_is_non_modal_so_paste_still_anchors_on_the_hover_tile() -> None:
    """The test a modal design fails, and the one pinning the _hover_tile
    constraint: exec_() would both block here and freeze the paste anchor."""
    window = _window()
    try:
        _copy(window, 0, 0, 2, 2, _TERRAIN_A)
        window._show_clipboard_history()
        assert window._clipboard_dialog is not None
        assert not window._clipboard_dialog.isModal()

        window.on_hover((10, 12))
        window.paste_region()
        mm = window.scenario.map_manager
        assert mm.get_tile(10, 12).terrain_id == _TERRAIN_A
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paste_uses_the_active_entry_not_merely_the_newest() -> None:
    window = _window()
    try:
        first = _copy(window, 0, 0, 2, 2, _TERRAIN_A)
        _copy(window, 4, 0, 6, 2, _TERRAIN_B)
        mm = window.scenario.map_manager

        window.on_hover((10, 12))
        window.paste_region()
        assert mm.get_tile(10, 12).terrain_id == _TERRAIN_B

        window._on_clipboard_activate(first.entry_id)
        window.on_hover((20, 22))
        window.paste_region()
        assert mm.get_tile(20, 22).terrain_id == _TERRAIN_A
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_region_clipboard_property_tracks_the_active_entry() -> None:
    window = _window()
    try:
        assert window._region_clipboard is None
        first = _copy(window, 0, 0, 2, 2, _TERRAIN_A)
        second = _copy(window, 4, 0, 6, 2, _TERRAIN_B)
        assert window._region_clipboard is second.block

        window._on_clipboard_activate(first.entry_id)
        assert window._region_clipboard is first.block

        window._on_clipboard_delete(first.entry_id)
        assert window._region_clipboard is second.block

        window._on_clipboard_clear()
        assert window._region_clipboard is None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paste_action_enablement_follows_the_history() -> None:
    """i.e. the callbacks really do reach _update_tool_enabled()."""
    window = _window()
    try:
        assert not window.paste_action.isEnabled()
        entry = _copy(window, 0, 0, 2, 2, _TERRAIN_A)
        assert window.paste_action.isEnabled()
        window._on_clipboard_delete(entry.entry_id)
        assert not window.paste_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_history_survives_closing_the_file() -> None:
    window = _window()
    try:
        _copy(window, 0, 0, 2, 2, _TERRAIN_A)
        # _copy paints, so the document is dirty and close_scenario() would
        # otherwise raise a modal discard prompt that hangs an offscreen run.
        window.edit_history.mark_saved()
        window.close_scenario()
        assert len(window._clipboard_history.entries) == 1
        assert not window.paste_action.isEnabled(), "no map loaded"

        window.load_scenario(BLANK_TEMPLATE_PATH)
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("select")
        window.on_hover((10, 12))
        window.paste_region()
        assert window.scenario.map_manager.get_tile(10, 12).terrain_id == _TERRAIN_A
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_copy_with_the_dialog_open_grows_it_and_keeps_the_selected_row() -> None:
    window = _window()
    try:
        first = _copy(window, 0, 0, 2, 2, _TERRAIN_A)
        window._show_clipboard_history()
        dialog = window._clipboard_dialog
        dialog.select_entry(first.entry_id)

        _copy(window, 4, 0, 6, 2, _TERRAIN_B)
        assert dialog.tree.topLevelItemCount() == 2
        assert dialog.selected_entry_id() == first.entry_id, "id-based selection restore"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_eleven_copies_leave_ten_rows_newest_first() -> None:
    from descape import clipboard_history

    window = _window()
    try:
        entries = [
            _copy(window, x, 0, x + 1, 1, _TERRAIN_A if x % 2 else _TERRAIN_B)
            for x in range(clipboard_history.MAX_ENTRIES + 1)
        ]
        window._show_clipboard_history()
        dialog = window._clipboard_dialog
        assert dialog.tree.topLevelItemCount() == clipboard_history.MAX_ENTRIES
        assert window._clipboard_history.entries[0] is entries[-1]
        assert window._region_clipboard is entries[-1].block
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rename_reaches_the_history_without_reordering() -> None:
    window = _window()
    try:
        first = _copy(window, 0, 0, 2, 2, _TERRAIN_A)
        _copy(window, 4, 0, 6, 2, _TERRAIN_B)
        order = [e.entry_id for e in window._clipboard_history.entries]
        window._on_clipboard_rename(first.entry_id, "shoreline")
        assert [e.entry_id for e in window._clipboard_history.entries] == order
        assert window._clipboard_history.entries[1].label == "shoreline"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_clipboard_history_action_is_registered_and_unbound_by_default() -> None:
    from descape import settings

    ids = [row[0] for row in settings.REBINDABLE_ACTIONS]
    assert "edit_clipboard_history" in ids
    assert settings.get_keybind("edit_clipboard_history") == ""
