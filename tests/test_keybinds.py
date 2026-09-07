"""Coverage for the menu-item keybind categories brought into the
rebindable system alongside Modes/Tools/Tool Value -- File/Edit/View/Help.

Same offscreen-ViewerWindow technique tests/test_fill_tool.py's gui tests
already use (QT_QPA_PLATFORM=offscreen, one shared QApplication via
conftest.ensure_qapp()). SettingsDialog is constructed directly rather than
through ViewerWindow._show_settings() -- that method calls dialog.exec_(),
which is modal and would hang an offscreen run.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def test_every_rebindable_action_has_a_matching_keybind_action() -> None:
    """Mirrors test_fill_tool.py's test_fill_registered_as_keybind_and_
    toolbar_action, generalized to loop over all of REBINDABLE_ACTIONS
    rather than just tool_fill."""
    from descape import settings
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        for action_id, _label, _default in settings.REBINDABLE_ACTIONS:
            assert action_id in window._keybind_actions, f"{action_id} missing from _keybind_actions"
        # Reverse direction too: an entry in _keybind_actions without a
        # REBINDABLE_ACTIONS counterpart gets its shortcut silently cleared
        # by apply_keybind() (get_keybind() returns "" for an unknown id),
        # not a loud failure -- so a plain subset check above wouldn't catch
        # a stale/orphaned _keybind_actions entry.
        rebindable_ids = {action_id for action_id, _label, _default in settings.REBINDABLE_ACTIONS}
        assert set(window._keybind_actions) == rebindable_ids
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_redo_default_keybind_is_ctrl_y() -> None:
    from PyQt5.QtGui import QKeySequence

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert window.redo_action.shortcut() == QKeySequence("Ctrl+Y")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_assigning_a_taken_keybind_auto_clears_the_conflict_and_warns() -> None:
    """Pins the mitigation decided 2026-08-28: retargeting one action onto
    a sequence another action already owns auto-clears the other action
    rather than leaving both dead. Also
    pins the underlying Qt behaviour the mitigation exists to prevent -- two
    QActions sharing a sequence fire NEITHER on press -- not just that the
    settings dict ends up collision-free: a dict-only assertion would pass
    against an implementation that never actually fixed the collision.

    Uses mode_view/mode_terrain rather than a real menu action like
    file_new: those two are always-enabled, dialog-free QActions (just
    retarget mode_combo), so firing one for real in an offscreen test can't
    hang on a modal confirmation/input dialog the way e.g. New Map's would.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    from descape import settings
    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        dialog = SettingsDialog(window)
        try:
            # mode_terrain defaults to "Ctrl+E", mode_view to "Ctrl+I" --
            # retarget mode_view onto mode_terrain's sequence.
            assert settings.get_keybind("mode_terrain") == "Ctrl+E"
            dialog._keybind_edits["mode_view"].setKeySequence(QKeySequence("Ctrl+E"))

            assert settings.get_keybind("mode_view") == "Ctrl+E"
            assert settings.get_keybind("mode_terrain") == ""
            assert dialog._keybind_edits["mode_terrain"].keySequence().isEmpty()
            assert dialog._keybind_warning_label.text() == (
                "'Ctrl+E' was already assigned to Terrain Mode -- that binding has been cleared."
            )
        finally:
            dialog.close()

        # Qt's shortcut dispatch only considers WindowShortcut-context
        # actions on the active window, which offscreen show() alone
        # doesn't establish.
        window.show()
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        assert window.isActiveWindow()

        fired = {"view": 0, "terrain": 0}
        window.mode_view_action.triggered.connect(lambda: fired.__setitem__("view", fired["view"] + 1))
        window.mode_terrain_action.triggered.connect(lambda: fired.__setitem__("terrain", fired["terrain"] + 1))
        QTest.keyClick(window, Qt.Key_E, Qt.ControlModifier)
        QApplication.processEvents()
        # Only mode_view's (now "Ctrl+E") shortcut should fire -- mode_terrain's
        # was auto-cleared, not left to collide silently. Had the collision
        # gone unhandled, both actions would still share "Ctrl+E" and Qt's
        # ambiguous-shortcut-overload behaviour would fire NEITHER, which
        # would read here as {"view": 0, "terrain": 0} -- indistinguishable
        # from a broken test setup, not a false pass, since that is not the
        # asserted value.
        assert fired == {"view": 1, "terrain": 0}
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_default_button_refuses_to_clear_another_actions_custom_binding() -> None:
    """2026-09-02 decision: clicking Default is a much weaker statement than
    typing a sequence, and must not outrank a binding the user chose on
    purpose. mode_terrain defaults to "Ctrl+E"; custom-binding mode_view onto
    it first (auto-clearing mode_terrain, which is the existing, correct
    behaviour covered above) sets up the collision this test targets."""
    from PyQt5.QtGui import QKeySequence

    from descape import settings
    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        dialog = SettingsDialog(window)
        try:
            dialog._keybind_edits["mode_view"].setKeySequence(QKeySequence("Ctrl+E"))
            assert settings.get_keybind("mode_view") == "Ctrl+E"
            assert settings.get_keybind("mode_terrain") == ""
            terrain_sequence_before = dialog._keybind_edits["mode_terrain"].keySequence()

            dialog._reset_keybind("mode_terrain")

            assert settings.get_keybind("mode_view") == "Ctrl+E"
            assert settings.get_keybind("mode_terrain") == ""
            assert dialog._keybind_edits["mode_terrain"].keySequence() == terrain_sequence_before
            assert dialog._keybind_warning_label.text() == (
                "'Ctrl+E' is assigned to View Mode -- Terrain Mode's default was not restored."
            )
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_default_button_still_restores_an_uncontested_default() -> None:
    from PyQt5.QtGui import QKeySequence

    from descape import settings
    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        dialog = SettingsDialog(window)
        try:
            assert settings.get_keybind("mode_terrain") == "Ctrl+E"
            dialog._keybind_edits["mode_terrain"].setKeySequence(QKeySequence("Ctrl+F9"))
            assert settings.get_keybind("mode_terrain") == "Ctrl+F9"

            dialog._reset_keybind("mode_terrain")

            assert settings.get_keybind("mode_terrain") == "Ctrl+E"
            assert dialog._keybind_edits["mode_terrain"].keySequence() == QKeySequence("Ctrl+E")
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_player_select_actions_default_to_their_own_digit() -> None:
    from descape import settings
    from descape.viewer import GAIA_PLAYER_ID, MAX_PLAYER_ID, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        for pid in range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1):
            assert settings.get_keybind(f"player_select_{pid}") == str(pid)
            assert window.player_select_actions[pid].shortcut().toString() == str(pid)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_player_select_action_count_matches_max_player_id() -> None:
    """range(9) in REBINDABLE_ACTIONS can't silently drift away from
    MAX_PLAYER_ID -- the per-digit test above would still pass with a
    tenth player quietly missing."""
    from descape import settings, viewer

    count = len([a for a, _, _ in settings.REBINDABLE_ACTIONS if a.startswith("player_select_")])
    assert count == viewer.MAX_PLAYER_ID + 1


def test_select_player_in_units_mode_sets_owner_combo_only() -> None:
    """Pins the "selectors stay independent" decision: setting the Units
    owner combo must not touch the Players/Diplomacy panels' own combos."""
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.mode_combo.setCurrentText("Units")
        players_index_before = window.players_panel.player_combo.currentIndex()
        diplomacy_index_before = window.diplomacy_panel.player_combo.currentIndex()

        window._select_player(3)

        assert window.place_owner_combo.currentData() == 3
        assert window.players_panel.player_combo.currentIndex() == players_index_before
        assert window.diplomacy_panel.player_combo.currentIndex() == diplomacy_index_before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_player_gaia_in_players_mode_is_a_noop() -> None:
    from pathlib import Path

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.load_scenario(Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario")
        window.mode_combo.setCurrentText("Players")
        index_before = window.players_panel.player_combo.currentIndex()

        window._select_player(0)

        assert window.players_panel.player_combo.currentIndex() == index_before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_player_gaia_in_diplomacy_mode_is_a_noop() -> None:
    from pathlib import Path

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.load_scenario(Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario")
        window.mode_combo.setCurrentText("Diplomacy")
        index_before = window.diplomacy_panel.player_combo.currentIndex()

        window._select_player(0)

        assert window.diplomacy_panel.player_combo.currentIndex() == index_before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_player_in_terrain_mode_is_a_noop() -> None:
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.mode_combo.setCurrentText("Terrain")
        # Nothing to assert against except "it doesn't raise" -- Terrain
        # mode has no player-scoped widget at all.
        window._select_player(3)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_keybinds_tab_renders_menu_section_headers() -> None:
    from PyQt5.QtWidgets import QLabel

    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        dialog = SettingsDialog(window)
        try:
            headers = {
                label.text().replace("<b>", "").replace("</b>", "")
                for label in dialog.findChildren(QLabel)
                if label.text().startswith("<b>")
            }
            for expected in ("File", "Edit", "View", "Help", "Modes", "Tools", "Tool Value"):
                assert expected in headers, f"{expected} section header missing from Keybinds tab"
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()
