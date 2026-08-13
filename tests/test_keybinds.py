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
