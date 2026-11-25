"""Settings > General > Trigger section headers (TASK-486): the group's controls."""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def test_the_controls_reflect_settings_and_a_change_applies_at_once() -> None:
    from descape import settings, trigger_organize

    settings.set_trigger_section_headers(trigger_organize.DividerRule.with_prefixes((">>",), tilde_wrap=False))
    dialog, window = conftest.dialog_and_window()
    try:
        page = dialog.trigger_header_settings
        assert [box.isChecked() for box in page.checkboxes.values()] == [True, True, False]
        assert page.prefix_edit.text() == ">>"
        assert trigger_organize.divider_rule() == settings.get_trigger_section_headers(), "the window applied it"

        page.checkboxes["bare_rule"].setChecked(False)
        assert not settings.get_trigger_section_headers().bare_rule
        assert not trigger_organize.is_divider("------")

        page.prefix_edit.setText("//  >> //")
        page.prefix_edit.editingFinished.emit()
        assert page.prefix_edit.text() == "// >>", "deduped, single spaces"
        assert trigger_organize.is_divider("// Setup")
    finally:
        dialog.close()
        window.close()


def test_an_invalid_prefix_shows_an_error_and_leaves_the_prefixes_alone() -> None:
    from descape import settings, trigger_organize

    dialog, window = conftest.dialog_and_window()
    try:
        page = dialog.trigger_header_settings
        page.prefix_edit.setText(">> [")
        page.prefix_edit.editingFinished.emit()
        assert not page.error_label.isHidden() and "'['" in page.error_label.text()
        assert settings.get_trigger_section_headers() == trigger_organize.DEFAULT_DIVIDER_RULE
        assert not trigger_organize.is_divider(">> Setup")

        page.checkboxes["tilde_wrap"].setChecked(False)
        assert settings.get_trigger_section_headers() == trigger_organize.DividerRule(tilde_wrap=False)
        page.prefix_edit.setText(">>")
        page.prefix_edit.editingFinished.emit()
        assert page.error_label.isHidden()
    finally:
        dialog.close()
        window.close()


def test_cancel_puts_the_rule_back() -> None:
    from descape import trigger_organize

    dialog, window = conftest.dialog_and_window()
    try:
        dialog.show()
        dialog.trigger_header_settings.checkboxes["leading_rule"].setChecked(False)
        assert not trigger_organize.is_divider("----- B")
        dialog._cancel()
        assert trigger_organize.divider_rule() == trigger_organize.DEFAULT_DIVIDER_RULE
    finally:
        dialog.close()
        window.close()


def test_a_prefix_typed_then_cancelled_is_not_applied_as_the_dialog_hides() -> None:
    from descape import settings, trigger_organize

    dialog, window = conftest.dialog_and_window()
    try:
        dialog.show()
        page = dialog.trigger_header_settings
        page.prefix_edit.setText(">>")
        dialog._cancel()
        page.prefix_edit.editingFinished.emit()  # the focus loss as the dialog hides
        assert settings.get_trigger_section_headers() == trigger_organize.DEFAULT_DIVIDER_RULE
        assert trigger_organize.divider_rule() == trigger_organize.DEFAULT_DIVIDER_RULE
    finally:
        dialog.close()
        window.close()


def test_ok_takes_a_typed_prefix_without_enter() -> None:
    from PyQt5.QtWidgets import QDialog

    from descape import trigger_organize

    dialog, window = conftest.dialog_and_window()
    try:
        dialog.show()
        dialog.trigger_header_settings.prefix_edit.setText(">>")
        dialog.done(QDialog.Accepted)
        assert trigger_organize.is_divider(">> Setup")
    finally:
        dialog.close()
        window.close()


def test_reset_on_general_puts_the_default_rule_back_everywhere() -> None:
    from descape import settings, trigger_organize

    dialog, window = conftest.dialog_and_window()
    try:
        dialog.show()
        dialog.trigger_header_settings.checkboxes["bare_rule"].setChecked(False)
        old = dialog.trigger_header_settings
        dialog._tabs.setCurrentIndex(next(i for i in range(dialog._tabs.count()) if dialog._tabs.tabText(i) == "General"))
        dialog._reset_tab()
        old.prefix_edit.editingFinished.emit()  # the old page's focus loss must not re-save
        assert settings.get_trigger_section_headers() == trigger_organize.DEFAULT_DIVIDER_RULE
        assert trigger_organize.divider_rule() == trigger_organize.DEFAULT_DIVIDER_RULE
        assert dialog.trigger_header_settings is not old
        assert dialog.trigger_header_settings.checkboxes["bare_rule"].isChecked()
    finally:
        dialog.close()
        window.close()
