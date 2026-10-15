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


def test_every_menu_command_is_rebindable_or_explicitly_exempt() -> None:
    """A menu action added with its own QAction but no REBINDABLE_ACTIONS row
    never reaches the Keybinds tab, and nothing else fails. Walks the menu bar
    and the Filters popup; tools are covered structurally via settings.TOOLS."""
    from PyQt5.QtWidgets import QAction

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        bound = {id(action) for action in window._keybind_actions.values()}
        # Each exemption is a deliberate "not a command" call: preset sizes
        # beside file_new/file_new_default, and per-player filter toggles.
        exempt = {id(a) for a in window.new_size_actions if a is not window.new_action}
        exempt |= {id(a) for a in window.player_actions.values()}

        missing: list[str] = []

        def walk(actions: list[QAction]) -> None:
            for action in actions:
                if action.isSeparator():
                    continue
                if action.menu() is not None:
                    # Open Recent is rebuilt from disk on every open, so its
                    # entries are never stable objects to bind.
                    if action.menu() is not window.recent_menu:
                        walk(action.menu().actions())
                    continue
                group = action.actionGroup()
                if group is not None and group.isExclusive():
                    continue  # a radio choice inside a submenu, not a command
                if id(action) not in bound and id(action) not in exempt:
                    missing.append(action.text())

        walk(window.menuBar().actions())
        walk(window.filters_button.menu().actions())
        assert not missing, f"menu actions with no REBINDABLE_ACTIONS row: {missing}"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_recover_autosave_ships_unbound_and_takes_a_binding() -> None:
    from PyQt5.QtGui import QKeySequence

    from descape import settings
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert settings.get_default_keybind("file_recover_autosave") == ""
        assert window.recover_autosave_action.shortcut().isEmpty()
        settings.set_keybind("file_recover_autosave", "Ctrl+F10")
        window.apply_keybind("file_recover_autosave")
        assert window.recover_autosave_action.shortcut() == QKeySequence("Ctrl+F10")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_every_mode_combo_entry_has_a_mode_keybind() -> None:
    """The mode_* rows are hand-listed, so a new mode could otherwise ship
    without a switch-to shortcut."""
    from descape import settings
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        combo = window.mode_combo
        expected = {f"mode_{combo.itemText(i).lower().replace(' ', '_')}" for i in range(combo.count())}
        rows = {a for a, _label, _default in settings.REBINDABLE_ACTIONS if a.startswith("mode_")}
        assert rows == expected
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


# Every checkable, non-radio keybind action. The Filters rows live only in the
# Filters button's popup menu, so they're the ones most likely to stop firing.
_TOGGLE_ACTION_IDS = (
    "view_isometric",
    "view_distance_ticks",
    "view_show_sprites",
    "view_stack_badges",
    "view_grid_overlay",
    "view_grid_follow",
    "view_footprint_outlines",
    "view_footprint_merged",
    "view_footprint_by_owner",
    "view_selection_owner_colour",
    "view_range_rings",
    "view_range_rings_by_owner",
    "view_player_cameras",
    "view_trigger_overlay",
    "view_layer_terrain_textures",
    "view_layer_terrain_blend",
    "view_layer_farm_overlay",
    "view_layer_small_trees",
    "view_layer_hero_glow",
    "help_perf_trace",
    "filter_show_gaia",
    "filter_show_trees",
    "filter_show_walls",
    "filter_show_buildings",
    "filter_show_eye_candy",
    "filter_show_obstacles",
    "filter_show_invisible",
    "filter_show_garrisoned",
)


def _is_toggle(action) -> bool:
    group = action.actionGroup()
    return action.isCheckable() and not (group is not None and group.isExclusive())


def test_the_toggle_list_names_every_checkable_keybind_action() -> None:
    """A new toggle added to _keybind_actions without a row above would
    silently escape the shortcut-toggles test below."""
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        toggles = {aid for aid, action in window._keybind_actions.items() if _is_toggle(action)}
        assert toggles == set(_TOGGLE_ACTION_IDS)
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("action_id", _TOGGLE_ACTION_IDS)
def test_a_bound_shortcut_toggles_its_checkable_action(action_id: str) -> None:
    """Binds an unused Ctrl+Alt+F9 and presses it on a shown, active Stepped
    window: the action's checked state must flip, then flip back on a second press."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    from descape import settings
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    action = window._keybind_actions[action_id]
    start = action.isChecked()
    try:
        if not action.isEnabled():
            pytest.skip(f"{action_id} is disabled on a blank Stepped window with no install")
        taken = [aid for aid in window._keybind_actions if settings.get_keybind(aid) == "Ctrl+Alt+F9"]
        assert not taken, f"Ctrl+Alt+F9 is not free: {taken}"
        settings.set_keybind(action_id, "Ctrl+Alt+F9")
        window.apply_keybind(action_id)
        assert action.shortcut() == QKeySequence("Ctrl+Alt+F9")

        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        assert window.isActiveWindow()

        QTest.keyClick(window, Qt.Key_F9, Qt.ControlModifier | Qt.AltModifier)
        QApplication.processEvents()
        assert action.isChecked() is not start
        QTest.keyClick(window, Qt.Key_F9, Qt.ControlModifier | Qt.AltModifier)
        QApplication.processEvents()
        assert action.isChecked() is start
    finally:
        # Persisted toggles (layers, overlays, filters) write through on toggle.
        if action.isChecked() is not start:
            action.trigger()
        conftest.close_window(window)


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

        assert window.units_panel.owner_id() == 3
        assert window.players_panel.player_combo.currentIndex() == players_index_before
        assert window.diplomacy_panel.player_combo.currentIndex() == diplomacy_index_before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_select_player_gaia_in_players_mode_selects_gaia() -> None:
    """GH #187: GAIA has its own rows in Players mode now."""
    from pathlib import Path

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.load_scenario(Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario")
        window.mode_combo.setCurrentText("Players")
        assert window.players_panel.current_player_id() == 1

        window._select_player(0)

        assert window.players_panel.current_player_id() == 0
        assert window.players_panel.player_combo.currentText() == "GAIA"
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


def test_tool_shortcut_fires_while_overflowed_into_more_tools() -> None:
    """Stage 3 of the tool-overflow plan: a tool action moved into the More
    Tools menu (toolbar.removeAction()) must still fire its shortcut --
    self.addAction() in _build_toolbar() gives every tool action a second,
    permanent host independent of toolbar membership. Cliff has no default
    key, so it's rebound here rather than relying on one.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    import conftest

    window = conftest.shown_terrain_window(width=settings.MIN_WINDOW_WIDTH)
    try:
        assert window.cliff_action in window.more_tools_menu.actions(), (
            "test assumes Cliff overflows at MIN_WINDOW_WIDTH -- see "
            "test_toolbar_overflow.py's own suffix-order test for why"
        )
        settings.set_keybind("tool_cliff", "Ctrl+F11")
        window.apply_keybind("tool_cliff")
        assert window.cliff_action.shortcut() == QKeySequence("Ctrl+F11")

        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        assert window.isActiveWindow()

        fired = {"count": 0}
        window.cliff_action.triggered.connect(lambda: fired.__setitem__("count", fired["count"] + 1))
        QTest.keyClick(window, Qt.Key_F11, Qt.ControlModifier)
        QApplication.processEvents()
        assert fired["count"] == 1
    finally:
        conftest.close_window(window)


def test_keybinds_tab_renders_menu_section_headers() -> None:
    from PyQt5.QtWidgets import QLabel

    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        dialog = SettingsDialog(window)
        try:
            # Scoped to a freshly-built Keybinds tab, not the whole dialog:
            # other tabs carry bold labels of their own, and the duplicate
            # check below would read those as a split section.
            keybinds_tab = dialog._build_keybinds_tab()
            header_list = [
                label.text().replace("<b>", "").replace("</b>", "")
                for label in keybinds_tab.findChildren(QLabel)
                if label.text().startswith("<b>")
            ]
            headers = set(header_list)
            for expected in ("File", "Edit", "View", "Help", "Modes", "Tools", "Tool Value"):
                assert expected in headers, f"{expected} section header missing from Keybinds tab"
            # Presence alone would not catch a SPLIT section: _build_keybinds_tab
            # only compares against the previous row's prefix, so a non-contiguous
            # view_* run emits a second "View" header rather than failing.
            assert len(header_list) == len(headers), f"duplicate section header(s) in {header_list}"
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_keybinds_search_box_filters_rows_by_label_or_key_and_hides_empty_sections() -> None:
    """GH #116. isHidden(), not isVisible(): nothing in an unshown dialog is visible."""
    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        dialog = SettingsDialog(window)
        try:
            sections = {header.text(): (divider, header, rows) for divider, header, rows in dialog._keybind_sections}
            file_divider, file_header, _ = sections["<b>File</b>"]
            edit_divider, edit_header, _ = sections["<b>Edit</b>"]
            map_divider, map_header, _ = sections["<b>Map</b>"]
            wheel_divider, _wheel_header, _ = sections["<b>Mouse Wheel</b>"]
            assert wheel_divider is None and file_divider is not None, "only the first section goes without a divider"

            def shown(action_id: str) -> bool:
                return not dialog._keybind_edits[action_id].isHidden()

            dialog._keybind_filter_edit.setText("UNDO")
            assert shown("edit_undo") and not shown("edit_redo") and not shown("file_open")
            assert file_header.isHidden() and not edit_header.isHidden()
            assert edit_divider.isHidden(), "the first shown section drew a divider above itself"
            assert map_header.isHidden() and map_divider.isHidden(), "a fully filtered section kept its header"
            assert dialog._keybind_no_match_label.isHidden()

            dialog._keybind_filter_edit.setText("ctrl+shift+s")  # the key column, not the label
            assert shown("file_save_as") and not shown("file_save")
            assert not file_header.isHidden() and edit_header.isHidden() and edit_divider.isHidden()

            dialog._keybind_filter_edit.setText("no such keybind anywhere")
            assert not any(shown(a) for a in dialog._keybind_edits)
            assert not dialog._keybind_no_match_label.isHidden()

            dialog._keybind_filter_edit.setText("")
            assert all(shown(a) for a in dialog._keybind_edits)
            assert not edit_divider.isHidden() and not map_divider.isHidden() and not map_header.isHidden()
            assert dialog._keybind_no_match_label.isHidden()
            assert not dialog._keybind_no_match_label.text().startswith("<b>")
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rebinding_a_row_refilters_the_keybinds_search() -> None:
    """A key search must follow a rebind, not keep matching the old binding."""
    from PyQt5.QtGui import QKeySequence

    from descape import settings
    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        dialog = SettingsDialog(window)
        try:
            for key in ("Ctrl+Alt+Shift+F9", "Ctrl+Alt+Shift+F8"):
                assert settings.keybind_holder(key) is None, f"test assumes nothing holds {key}"

            def shown(action_id: str) -> bool:
                return not dialog._keybind_edits[action_id].isHidden()

            dialog._keybind_filter_edit.setText("ctrl+shift+s")
            assert shown("file_save_as")
            dialog._keybind_edits["file_save_as"].setKeySequence(QKeySequence("Ctrl+Alt+Shift+F9"))
            assert not shown("file_save_as"), "a row stayed shown by a key it no longer has"

            dialog._keybind_filter_edit.setText("ctrl+alt+shift+f8")
            assert not dialog._keybind_no_match_label.isHidden()
            dialog._keybind_edits["file_save"].setKeySequence(QKeySequence("Ctrl+Alt+Shift+F8"))
            assert shown("file_save") and dialog._keybind_no_match_label.isHidden()
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()


# --- the Mouse Wheel rows (GH #172, GH #210) --------------------------------------


def _stored_wheel_bindings() -> dict | None:
    """The config file's own wheel_bindings mapping, or None when it has none."""
    import yaml

    from descape import settings

    if not settings.CONFIG_PATH.is_file():
        return None
    return (yaml.safe_load(settings.CONFIG_PATH.read_text()) or {}).get("wheel_bindings")


def _set_wheel_combo(dialog, slot: str, action: str) -> None:
    from descape import settings

    dialog._wheel_combos[slot].setCurrentIndex(settings.WHEEL_ACTIONS.index(action))


def test_mouse_wheel_is_the_first_keybinds_section_and_headers_stay_unique() -> None:
    from PyQt5.QtWidgets import QGridLayout, QLabel

    dialog, window = conftest.dialog_and_window()
    try:
        divider, header, rows = dialog._keybind_sections[0]
        assert header.text() == "<b>Mouse Wheel</b>" and divider is None
        assert rows, "the Mouse Wheel section has no rows"
        grid = header.parentWidget().layout()
        assert isinstance(grid, QGridLayout)
        assert grid.getItemPosition(grid.indexOf(header))[0] == 0, "Mouse Wheel is not the top row of the tab"
        # Every later section now draws a divider, File included.
        assert all(section[0] is not None for section in dialog._keybind_sections[1:])

        # A second build, as test_keybinds_tab_renders_menu_section_headers does: still one header each.
        keybinds_tab = dialog._build_keybinds_tab()
        headers = [label.text() for label in keybinds_tab.findChildren(QLabel) if label.text().startswith("<b>")]
        assert headers.count("<b>Mouse Wheel</b>") == 1
        assert len(headers) == len(set(headers)), f"duplicate section header(s) in {headers}"
        assert not dialog._wheel_note.text().startswith("<b>"), "the note would read as a section header"
        assert dialog._wheel_note.wordWrap()
    finally:
        dialog.close()
        conftest.close_window(window)


def test_the_wheel_combos_list_every_action_and_start_from_the_stored_values() -> None:
    from PyQt5.QtWidgets import QPushButton

    from descape import settings
    from descape.viewer import SettingsDialog, ViewerWindow
    from descape.viewer_common import NoWheelComboBox

    settings.set_wheel_binding("plain", "tool")
    settings.set_wheel_binding("alt", "none")
    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        dialog = SettingsDialog(window)
        try:
            assert list(dialog._wheel_combos) == list(settings.WHEEL_SLOTS)
            rows = dict(dialog._keybind_sections[0][2])
            assert list(rows) == [f"wheel_{slot}" for slot in settings.WHEEL_SLOTS]
            stored = settings.get_wheel_bindings()
            assert stored == {"plain": "tool", "ctrl": "tool", "shift": "zoom", "alt": "none"}
            for slot, combo in dialog._wheel_combos.items():
                assert isinstance(combo, NoWheelComboBox)
                assert [combo.itemData(i) for i in range(combo.count())] == list(settings.WHEEL_ACTIONS)
                assert [combo.itemText(i) for i in range(combo.count())] == [
                    settings.WHEEL_ACTION_LABELS[action] for action in settings.WHEEL_ACTIONS
                ]
                assert combo.currentData() == stored[slot]
                label, row_combo, default_btn = rows[f"wheel_{slot}"]
                assert label.text() == settings.WHEEL_SLOT_LABELS[slot] and row_combo is combo
                assert isinstance(default_btn, QPushButton) and default_btn.text() == "Default"
            # Building the rows is not a change: nothing was logged and nothing rewritten.
            assert "Wheel binding changed" not in window.status_log.toPlainText()
            assert _stored_wheel_bindings() == {"plain": "tool", "alt": "none"}
        finally:
            dialog.close()
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_the_ctrl_wheel_combo_shows_its_whole_default_label() -> None:
    """TASK-356: at the dialog's default size the combo clipped "Brush size or
    rotate preview" (190 px field, 225 px label at 96 DPI, offscreen Qt, pinned test font; 276 px after).
    The wheel combos span the key-box and Default columns; Default sits in Clear's."""
    from PyQt5.QtWidgets import QApplication, QStyle, QStyleOptionComboBox, QTabWidget

    from descape import settings

    dialog, window = conftest.dialog_and_window()
    try:
        tabs = dialog.findChild(QTabWidget)
        dialog.show()
        tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i) == "Keybinds"))
        for _ in range(4):
            QApplication.processEvents()
        combo = dialog._wheel_combos["ctrl"]
        assert combo.currentData() == settings.DEFAULT_WHEEL_BINDINGS["ctrl"]
        option = QStyleOptionComboBox()
        combo.initStyleOption(option)
        field = combo.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, combo)
        need = combo.fontMetrics().horizontalAdvance(combo.currentText())
        assert field.width() - 2 >= need, f"a {field.width()} px field cuts {need} px of {combo.currentText()!r}"
        # Default still sits beside its combo, in the grid row's last column.
        _label, _combo, default_btn = next(w for row_id, w in dialog._keybind_sections[0][2] if row_id == "wheel_ctrl")
        grid = combo.parentWidget().layout()
        combo_row, combo_col, _rows, combo_span = grid.getItemPosition(grid.indexOf(combo))
        assert default_btn.text() == "Default"
        assert grid.getItemPosition(grid.indexOf(default_btn)) == (combo_row, combo_col + combo_span, 1, 1)
        assert default_btn.geometry().left() > combo.geometry().right()
    finally:
        dialog.close()
        conftest.close_window(window)


def test_a_wheel_combo_change_reaches_the_config_file_and_a_real_wheel_event() -> None:
    from test_wheel_bindings import _scale, _terrain_window, _wheel_xy

    from descape.viewer import SettingsDialog

    window = _terrain_window("draw")
    try:
        dialog = SettingsDialog(window)
        try:
            view = window.map_view
            size, scale = window.brush_size_spin.value(), _scale(view)
            _wheel_xy(view, (0, 120))
            assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9), "plain wheel should zoom by default"
            scale = _scale(view)

            _set_wheel_combo(dialog, "plain", "brush_size")
            assert _stored_wheel_bindings() == {"plain": "brush_size"}
            assert "Wheel binding changed: Wheel -> Brush size" in window.status_log.toPlainText()
            _wheel_xy(view, (0, 120))
            assert window.brush_size_spin.value() == size + 1, "the new binding is not live on the map"
            assert _scale(view) == scale
        finally:
            dialog.close()
    finally:
        conftest.close_window(window)


def test_the_wheel_default_button_restores_and_leaves_no_stored_key() -> None:
    from PyQt5.QtCore import Qt

    from descape import settings

    dialog, window = conftest.dialog_and_window()
    try:
        rows = dict(dialog._keybind_sections[0][2])
        _set_wheel_combo(dialog, "plain", "none")
        _set_wheel_combo(dialog, "ctrl", "zoom")
        assert _stored_wheel_bindings() == {"plain": "none", "ctrl": "zoom"}
        assert window.map_view._wheel_bindings[int(Qt.ControlModifier)] == "zoom"

        rows["wheel_plain"][2].click()
        assert dialog._wheel_combos["plain"].currentData() == "zoom"
        assert _stored_wheel_bindings() == {"ctrl": "zoom"}

        rows["wheel_ctrl"][2].click()
        assert dialog._wheel_combos["ctrl"].currentData() == "tool"
        assert _stored_wheel_bindings() is None, "a slot on its default left a stored key behind"
        assert settings.get_wheel_bindings() == settings.DEFAULT_WHEEL_BINDINGS
        assert window.map_view._wheel_bindings[int(Qt.ControlModifier)] == "tool"
        assert window.map_view._wheel_bindings[int(Qt.NoModifier)] == "zoom"
    finally:
        dialog.close()
        conftest.close_window(window)


def test_two_wheel_slots_may_share_an_action_with_no_warning() -> None:
    from descape import settings

    dialog, window = conftest.dialog_and_window()
    try:
        assert settings.get_wheel_bindings()["ctrl"] == "tool"
        _set_wheel_combo(dialog, "plain", "tool")
        _set_wheel_combo(dialog, "shift", "tool")
        bindings = settings.get_wheel_bindings()
        assert (bindings["plain"], bindings["ctrl"], bindings["shift"]) == ("tool", "tool", "tool")
        assert [dialog._wheel_combos[slot].currentData() for slot in ("plain", "ctrl", "shift")] == ["tool"] * 3
        assert dialog._keybind_warning_label.text() == ""
    finally:
        dialog.close()
        conftest.close_window(window)


def test_the_keybinds_search_finds_wheel_rows_by_slot_and_by_action_and_refilters() -> None:
    from descape import settings

    dialog, window = conftest.dialog_and_window()
    try:
        sections = {header.text(): (divider, header) for divider, header, _rows in dialog._keybind_sections}
        _, wheel_header = sections["<b>Mouse Wheel</b>"]
        file_divider, file_header = sections["<b>File</b>"]
        edit_divider, edit_header = sections["<b>Edit</b>"]
        note = dialog._wheel_note

        def shown_slots() -> list[str]:
            return [slot for slot in settings.WHEEL_SLOTS if not dialog._wheel_combos[slot].isHidden()]

        def search(text: str) -> list[str]:
            dialog._keybind_filter_edit.setText(text)
            return shown_slots()

        assert search("WHEEL") == list(settings.WHEEL_SLOTS)
        assert not wheel_header.isHidden() and not note.isHidden()
        assert file_header.isHidden() and file_divider.isHidden()

        assert search("ctrl") == ["ctrl"]  # the slot label
        assert not wheel_header.isHidden() and not note.isHidden()
        assert not file_header.isHidden() and not file_divider.isHidden(), "File follows a shown section"

        assert search("zoom") == ["plain", "shift", "alt"]  # the dropdown's current action
        assert search("brush") == ["ctrl"]
        row_widgets = dict(dialog._keybind_sections[0][2])
        assert all(widget.isHidden() for widget in row_widgets["wheel_plain"])
        assert not any(widget.isHidden() for widget in row_widgets["wheel_ctrl"])

        # Still searching "brush": a change re-filters, both ways.
        _set_wheel_combo(dialog, "plain", "brush_size")
        assert shown_slots() == ["plain", "ctrl"]
        _set_wheel_combo(dialog, "ctrl", "zoom")
        assert shown_slots() == ["plain"], "a row stayed shown by an action it no longer has"

        assert search("undo") == []
        assert wheel_header.isHidden() and note.isHidden(), "the note outlived its hidden section"
        assert not edit_header.isHidden() and edit_divider.isHidden(), "the first shown section drew a divider"

        assert search("no such keybind anywhere") == []
        assert note.isHidden() and not dialog._keybind_no_match_label.isHidden()

        assert search("") == list(settings.WHEEL_SLOTS)
        assert not wheel_header.isHidden() and not note.isHidden()
        assert not file_divider.isHidden() and dialog._keybind_no_match_label.isHidden()
    finally:
        dialog.close()
        conftest.close_window(window)


@pytest.mark.parametrize("slot", ["plain", "ctrl"])
@pytest.mark.parametrize("units", [-120, 120], ids=["down", "up"])
def test_a_wheel_event_over_a_wheel_combo_leaves_it_unchanged(slot: str, units: int) -> None:
    """Scrolling the tab over a row must scroll the tab, never rebind the row.
    Plain sits on index 0, where a stock combo ignores wheel-up anyway, so
    wheel-down is the half that tells the two apart."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent

    from descape import settings

    dialog, window = conftest.dialog_and_window()
    try:
        combo = dialog._wheel_combos[slot]
        before = combo.currentIndex()
        assert before == settings.WHEEL_ACTIONS.index(settings.DEFAULT_WHEEL_BINDINGS[slot])
        center = QPointF(combo.rect().center())
        event = QWheelEvent(
            center, center, QPoint(0, 0), QPoint(0, units), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False
        )
        combo.wheelEvent(event)
        assert combo.currentIndex() == before
        assert not event.isAccepted(), "an accepted event never reaches the scroll area"
        assert settings.get_wheel_bindings() == settings.DEFAULT_WHEEL_BINDINGS
        assert _stored_wheel_bindings() is None
    finally:
        dialog.close()
        conftest.close_window(window)


def test_the_pan_actions_are_registered_and_default_to_the_arrow_keys() -> None:
    from descape import settings

    expected = {
        "view_pan_up": ("Pan Up", "Up"),
        "view_pan_down": ("Pan Down", "Down"),
        "view_pan_left": ("Pan Left", "Left"),
        "view_pan_right": ("Pan Right", "Right"),
    }
    rows = {a: (label, default) for a, label, default in settings.REBINDABLE_ACTIONS}
    for action_id, row in expected.items():
        assert rows.get(action_id) == row


def test_the_view_action_ids_form_one_contiguous_run() -> None:
    """_build_keybinds_tab only compares against the PREVIOUS row's prefix,
    so a view_* id separated from the others emits a second "View" header."""
    from descape import settings

    prefixes = [a.split("_", 1)[0] for a, _label, _default in settings.REBINDABLE_ACTIONS]
    view_positions = [i for i, p in enumerate(prefixes) if p == "view"]
    assert view_positions == list(range(view_positions[0], view_positions[-1] + 1))


def test_gh211_the_trigger_move_shortcuts_are_six_tree_scoped_rows_under_one_header() -> None:
    """Scoped to the trigger tree and the entry tree, never the window: a
    window-wide Alt+Down would take the key from every dropdown in the trigger
    form, where it opens the popup. No auto-repeat: each press is an undo record."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtWidgets import QLabel

    from descape import settings
    from descape.viewer import SettingsDialog, ViewerWindow

    expected = [
        ("trigger_move_up", "Alt+Up"),
        ("trigger_move_down", "Alt+Down"),
        ("trigger_move_up_far", "Alt+Shift+Up"),
        ("trigger_move_down_far", "Alt+Shift+Down"),
        ("trigger_move_top", "Alt+Home"),
        ("trigger_move_bottom", "Alt+End"),
    ]
    rows = [(a, label, default) for a, label, default in settings.REBINDABLE_ACTIONS if a.startswith("trigger_")]
    assert [(a, default) for a, _label, default in rows] == expected
    for _action_id, label, _default in rows:
        assert "Trigger" in label and "Condition/Effect" in label, label
    ids = [a for a, _label, _default in settings.REBINDABLE_ACTIONS]
    start = ids.index("trigger_move_up")
    assert ids[start : start + len(expected)] == [a for a, _ in expected], "one contiguous run, so one header"

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        panel = window.trigger_panel
        for action_id, default in expected:
            action = window._keybind_actions[action_id]
            assert action.shortcut() == QKeySequence(default), action_id
            assert action.shortcut().toString() == default, "the default is in Qt's own spelling"
            assert action.shortcutContext() == Qt.WidgetWithChildrenShortcut, action_id
            assert not action.autoRepeat(), action_id
            assert action.associatedWidgets() == [panel.tree, panel.entry_tree], action_id
            assert action not in window.actions(), action_id
        # The label column is as wide as its widest label, and the key boxes get what is left.
        mine = {label for _action_id, label, _default in rows}
        others = {label for _a, label, _d in settings.REBINDABLE_ACTIONS} - mine
        assert max(QLabel(label).sizeHint().width() for label in mine) <= max(
            QLabel(label).sizeHint().width() for label in others
        ), "a trigger move label widens the Keybinds label column"
        dialog = SettingsDialog(window)
        try:
            # A freshly built tab, as in the section-header test above.
            keybinds_tab = dialog._build_keybinds_tab()
            headers = [label.text() for label in keybinds_tab.findChildren(QLabel) if label.text().startswith("<b>")]
            assert headers.count("<b>Triggers</b>") == 1, headers
            for action_id, default in expected:
                assert dialog._keybind_edits[action_id].keySequence() == QKeySequence(default), action_id
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_pan_ids_reach_map_view_rather_than_a_qaction_shortcut() -> None:
    """apply_keybind's one deliberate deviation from the generic path: these
    four never get a shortcut, so asserting .shortcut() would be asserting
    the wrong thing. The binding landing in MapView is the real contract."""
    from PyQt5.QtCore import Qt

    from descape import settings
    from descape.map_view import MapView
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        for action_id in MapView.PAN_DIRECTIONS:
            assert window._keybind_actions[action_id].shortcut().isEmpty()
        bound = set(window.map_view._pan_bindings.values())
        assert bound == set(MapView.PAN_DIRECTIONS.values())

        settings.set_keybind("view_pan_up", "W")
        window.apply_keybind("view_pan_up")
        assert window._keybind_actions["view_pan_up"].shortcut().isEmpty()
        assert (int(Qt.Key_W), 0) in window.map_view._pan_bindings
    finally:
        window.edit_history.mark_saved()
        window.close()
