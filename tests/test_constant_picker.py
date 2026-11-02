"""Coverage for descape/constant_picker.py (phase 4d, slice 2):
CatalogLineEdit's type-ahead/raw-integer entry and commit-on-focus-out
behaviour, and CatalogBrowseDialog, the Units catalog over one field's objects
(GH #273). descape/catalog_browser.py's own tests cover the view it hosts.

Standalone widgets, not through a ViewerWindow: neither touches a loaded
scenario, so conftest.ensure_qapp() is all the Qt setup either needs.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """No test here wants the real install, and AOE2DE_INSTALL_PATH outranks
    the config conftest hides, so a shell exporting it would leak one in."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


@pytest.fixture(autouse=True)
def _no_real_modal(monkeypatch):
    """A real exec_() here would hang the run instead of failing the test. A test that expects one patches over this."""
    from descape.constant_picker import CatalogBrowseDialog
    from descape.value_picker import ValueBrowseDialog

    for dialog in (CatalogBrowseDialog, ValueBrowseDialog):
        monkeypatch.setattr(dialog, "exec_", lambda self: pytest.fail(f"{type(self).__name__} opened for real"))


def _catalog():
    from descape import object_catalog

    return object_catalog.objects()


def _line_edit(default_category=""):
    conftest.ensure_qapp()
    from descape.constant_picker import CatalogLineEdit

    return CatalogLineEdit(_catalog(), default_category)


def test_set_value_shows_the_resolved_name() -> None:
    widget = _line_edit()
    widget.set_value(109)
    assert widget.value() == 109
    assert widget.line_edit.text() == "TOWN CENTER"


def test_set_value_falls_back_to_the_raw_id_when_uncovered() -> None:
    widget = _line_edit()
    widget.set_value(999999)
    assert widget.value() == 999999
    assert widget.line_edit.text() == "999999"


def test_set_value_none_clears_the_field() -> None:
    widget = _line_edit()
    widget.set_value(109)
    widget.set_value(None)
    assert widget.value() is None
    assert widget.line_edit.text() == ""


def test_typing_a_name_commits_its_id() -> None:
    widget = _line_edit()
    widget.set_value(None)
    received = []
    widget.committed.connect(received.append)
    widget.line_edit.setText("Town Center")
    widget.line_edit.editingFinished.emit()
    assert received == [109]
    assert widget.value() == 109


def test_typing_a_raw_integer_commits_it_even_if_uncovered() -> None:
    widget = _line_edit()
    widget.set_value(None)
    received = []
    widget.committed.connect(received.append)
    widget.line_edit.setText("999999")
    widget.line_edit.editingFinished.emit()
    assert received == [999999]


def test_an_unrecognised_name_is_rejected_and_the_field_is_restored() -> None:
    widget = _line_edit()
    widget.set_value(109)
    received = []
    widget.committed.connect(received.append)
    widget.line_edit.setText("not a real object")
    widget.line_edit.editingFinished.emit()
    assert received == []
    assert widget.value() == 109
    assert widget.line_edit.text() == "TOWN CENTER"


def test_a_focus_out_with_the_same_value_commits_nothing() -> None:
    """Matches the plain QLineEdit field's own behaviour: editingFinished
    fires on every focus-out whether the text changed or not."""
    widget = _line_edit()
    widget.set_value(109)
    received = []
    widget.committed.connect(received.append)
    widget.line_edit.setText("Town Center")
    widget.line_edit.editingFinished.emit()
    assert received == []


def test_set_enabled_disables_both_the_line_edit_and_the_browse_button() -> None:
    widget = _line_edit()
    widget.setEnabled(False)
    assert not widget.line_edit.isEnabled()
    assert not widget.browse_button.isEnabled()


# -- CatalogBrowseDialog ------------------------------------------------------

_ARCHER = 4
_CASTLE = 82
_TOWN_CENTER = 109
_BLUE_TREE = 768  # a real hide_in_editor=True entry in the shipped object_catalog.json
_GUNPOWDER_KEG = 2607


def _dialog_for(catalog, default_category=""):
    conftest.ensure_qapp()
    from descape.constant_picker import CatalogBrowseDialog

    return CatalogBrowseDialog(catalog, default_category)


def _dialog(default_category=""):
    return _dialog_for(_catalog(), default_category)


def _active_dialog(default_category=""):
    """Shown and active, so keys and focus behave as for a user."""
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    dialog = _dialog(default_category)
    dialog.show()
    QApplication.setActiveWindow(dialog)
    assert QTest.qWaitForWindowActive(dialog)
    return dialog


def _ok_button(dialog):
    from PyQt5.QtWidgets import QDialogButtonBox

    return dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)


def _shown(dialog) -> list[int]:
    from PyQt5.QtCore import Qt

    rows = (dialog.view.list.item(i) for i in range(dialog.view.list.count()))
    return [row.data(Qt.UserRole) for row in rows if not row.isHidden()]


def test_dialog_opens_on_the_default_category() -> None:
    dialog = _dialog(default_category="Buildings")
    assert dialog.windowTitle() == "Browse"
    assert dialog.view.category_buttons["Buildings"].isChecked()
    assert _TOWN_CENTER in _shown(dialog)
    assert _ARCHER not in _shown(dialog)
    assert all(button.isEnabled() for button in dialog.view.category_buttons.values())


def test_select_crosses_categories_and_lands_on_the_row() -> None:
    dialog = _dialog(default_category="Units")
    dialog.select(_TOWN_CENTER)
    assert dialog.view.category_buttons["Buildings"].isChecked()
    assert dialog.view.current_value() == _TOWN_CENTER


def test_activating_a_row_accepts_it() -> None:
    """The view listens for itemActivated (double-click AND Enter), not the
    older itemDoubleClicked."""
    dialog = _dialog()
    dialog.select(_TOWN_CENTER)
    dialog.view.list.itemActivated.emit(dialog.view.list.currentItem())
    assert dialog.result() == dialog.Accepted
    assert dialog.selected_id() == _TOWN_CENTER


def test_ok_with_no_pick_stays_open() -> None:
    dialog = _active_dialog()
    try:
        _ok_button(dialog).click()
        assert dialog.isVisible()
        assert dialog.result() != dialog.Accepted
        assert dialog.selected_id() is None
    finally:
        dialog.close()


def test_ok_after_a_category_switch_returns_the_cards_object() -> None:
    dialog = _dialog()
    dialog.select(_ARCHER)
    dialog.view.category_buttons["Buildings"].click()
    assert dialog.view.current_value() is None, "the switch dropped Qt's current row"
    assert dialog.card.const_label.text() == str(_ARCHER)
    _ok_button(dialog).click()
    assert dialog.result() == dialog.Accepted
    assert dialog.selected_id() == _ARCHER


def test_a_filter_that_hides_the_pick_leaves_ok_on_it() -> None:
    dialog = _dialog()
    dialog.select(_ARCHER)
    dialog.filter_edit.setText("no such object at all")
    assert dialog.view.current_value() is None
    _ok_button(dialog).click()
    assert dialog.selected_id() == _ARCHER


def test_enter_in_the_filter_moves_to_the_first_match_and_never_accepts() -> None:
    """Archer is in Units and "castle" matches nothing there: the sticky pick
    must not be accepted out from under the filter, in either step."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    dialog = _active_dialog()
    try:
        dialog.select(_ARCHER)
        dialog.filter_edit.setFocus()
        dialog.filter_edit.setText("castle")
        assert _shown(dialog) == []
        QTest.keyClick(dialog.filter_edit, Qt.Key_Return)
        assert dialog.isVisible() and dialog.result() != dialog.Accepted

        dialog.view.category_buttons["Buildings"].click()
        assert _CASTLE in _shown(dialog)
        QTest.keyClick(dialog.filter_edit, Qt.Key_Return)
        assert dialog.isVisible() and dialog.result() != dialog.Accepted
        first = _shown(dialog)[0]
        assert dialog.view.current_value() == first
        assert dialog.focusWidget() is dialog.view.list

        QTest.keyClick(dialog.view.list, Qt.Key_Return)
        assert dialog.result() == dialog.Accepted
        assert dialog.selected_id() == first
    finally:
        dialog.close()


def test_filter_hides_non_matching_rows_and_counts_a_hit_elsewhere() -> None:
    bear = next(entry for entry in _catalog() if entry.name == "BROWN BEAR")
    dialog = _dialog()
    dialog.filter_edit.setText("brown bear")
    view = dialog.view
    counts = {category: button.text() for category, button in view.category_buttons.items()}
    assert counts.pop(bear.category).endswith("(1)")
    assert all(text.endswith("(0)") for text in counts.values())
    view.category_buttons[bear.category].click()
    assert _shown(dialog) == [bear.id]


def test_filter_matches_by_id_too() -> None:
    """Deliberate behaviour change (S2 of the units-sidebar-catalog plan):
    filtering now runs over CatalogEntry.search_text ("name id"), not just
    the rendered label, so a digit in the filter matches ids as well as
    names -- strictly more rows than the old rendered-label filter matched.
    109 is TOWN CENTER's id."""
    dialog = _dialog("Buildings")
    dialog.filter_edit.setText(str(_TOWN_CENTER))
    assert _TOWN_CENTER in _shown(dialog)


def test_clearing_the_filter_reshows_everything() -> None:
    dialog = _dialog()
    everything = _shown(dialog)
    dialog.filter_edit.setText("brown bear")
    dialog.filter_edit.setText("")
    assert _shown(dialog) == everything


def test_editor_hidden_objects_are_hidden_by_default() -> None:
    dialog = _dialog("Others")
    assert not dialog.show_hidden_checkbox.isChecked()
    assert _BLUE_TREE in dialog.view._items
    assert _BLUE_TREE not in _shown(dialog)


def test_checking_show_hidden_reveals_editor_hidden_objects() -> None:
    dialog = _dialog("Others")
    dialog.show_hidden_checkbox.setChecked(True)
    assert _BLUE_TREE in _shown(dialog)


def test_the_gunpowder_keg_shows_without_the_hidden_box() -> None:
    """GH #208: the .dat hides 2607, the official editor does not."""
    dialog = _dialog("Others")
    assert not dialog.show_hidden_checkbox.isChecked()
    assert _GUNPOWDER_KEG in _shown(dialog)


def test_unchecking_show_hidden_hides_them_again() -> None:
    dialog = _dialog("Others")
    dialog.show_hidden_checkbox.setChecked(True)
    dialog.show_hidden_checkbox.setChecked(False)
    assert _BLUE_TREE not in _shown(dialog)


def test_select_keeps_a_hidden_value_pinned_with_the_box_unticked() -> None:
    dialog = _dialog()
    dialog.select(_BLUE_TREE)
    assert dialog.view.current_value() == _BLUE_TREE
    assert not dialog.show_hidden_checkbox.isChecked()
    hidden = {entry.id for entry in _catalog() if entry.hidden}
    assert hidden & set(_shown(dialog)) == {_BLUE_TREE}


def test_the_keyboard_reaches_every_control_and_the_filter_starts_with_focus(fake_install) -> None:
    """An exclusive button group is one tab stop, its checked button; arrow keys walk the rest."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    def press(key) -> None:
        QTest.keyClick(QApplication.focusWidget(), key)

    dialog = _active_dialog()
    try:
        view = dialog.view
        assert QApplication.focusWidget() is dialog.filter_edit
        reached = set()
        for _ in range(16):
            press(Qt.Key_Tab)
            reached.add(QApplication.focusWidget())
        assert {view.category_buttons["Units"], view.class_combo, view.display_buttons["icons"], view.list} <= reached

        view.category_buttons["Units"].setFocus()
        press(Qt.Key_Right)
        assert view.category_buttons["Buildings"].isChecked()
        assert _TOWN_CENTER in _shown(dialog)
        press(Qt.Key_Down)
        assert view.category_buttons["Others"].isChecked()
        press(Qt.Key_Left)
        assert view.category_buttons["Heroes"].isChecked()

        view.display_buttons["icons"].setFocus()
        press(Qt.Key_Right)
        assert view.mode() == "names"
        press(Qt.Key_Right)
        assert view.mode() == "grid"
    finally:
        dialog.close()


def test_the_card_shows_the_rows_own_name() -> None:
    from descape import object_catalog

    entry = next(e for e in _catalog() if e.name != object_catalog.display_name(e.id) and not e.hidden)
    dialog = _dialog()
    assert dialog.card.name_label.text(), "the card says nothing is picked yet"
    dialog.select(entry.id)
    assert dialog.card.name_label.text() == entry.name
    assert dialog.card.const_label.text() == str(entry.id)


@pytest.mark.parametrize("close", ["ok", "cancel"])
def test_a_finished_dialog_is_deleted(close) -> None:
    from PyQt5.QtCore import QCoreApplication, QEvent

    dialog = _dialog()
    dialog.select(_ARCHER)
    destroyed = []
    dialog.destroyed.connect(lambda: destroyed.append(True))
    if close == "ok":
        _ok_button(dialog).click()
        assert dialog.selected_id() == _ARCHER, "still readable when exec_() returns"
    else:
        dialog.reject()
    assert not destroyed
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert destroyed


def test_the_card_gets_the_portrait_and_a_row_its_thumbnail(fake_install) -> None:
    dialog = _active_dialog()
    try:
        dialog.select(_ARCHER)
        conftest.settle_thumbs(dialog.card.loader)
        assert dialog.card.portrait_source == "icon"
        assert dialog.card.portrait.pixmap() is not None
        image = dialog.view._items[_ARCHER].icon().pixmap(32, 32).toImage()
        assert any(image.pixelColor(x, y).alpha() for x in range(32) for y in range(32)), "still the blank placeholder"
    finally:
        dialog.close()


def test_the_dialog_keeps_its_own_saved_mode(fake_install, monkeypatch) -> None:
    from PyQt5.QtWidgets import QListView

    from descape import settings

    units_mode = settings.get_units_catalog_mode()
    dialog = _dialog()
    assert dialog.view.mode() == settings.BROWSE_CATALOG_MODE_DEFAULT
    dialog.view.display_buttons["grid"].click()
    monkeypatch.setattr(settings, "_browse_catalog_mode", None)
    monkeypatch.setattr(settings, "_units_catalog_mode", None)
    assert settings.get_browse_catalog_mode() == "grid"
    assert settings.get_units_catalog_mode() == units_mode
    reopened = _dialog()
    assert reopened.view.mode() == "grid"
    assert reopened.view.list.viewMode() == QListView.IconMode


# -- CatalogLineEdit's Browse ---------------------------------------------------


def _accepting_exec(monkeypatch, pick) -> list:
    """exec_ as a user picking `pick` and pressing OK. Returns the dialogs it saw."""
    from descape.constant_picker import CatalogBrowseDialog

    built = []

    def fake_exec(dialog):
        built.append(dialog)
        dialog.view.select(pick)
        _ok_button(dialog).click()
        return dialog.result()

    monkeypatch.setattr(CatalogBrowseDialog, "exec_", fake_exec)
    return built


def test_an_object_fields_browse_is_the_catalog_dialog_and_commits_the_pick(monkeypatch) -> None:
    from descape.constant_picker import CatalogBrowseDialog

    built = _accepting_exec(monkeypatch, _CASTLE)
    widget = _line_edit("Units")
    widget.set_value(_ARCHER)
    received = []
    widget.committed.connect(received.append)
    widget.browse_button.click()
    assert isinstance(built[-1], CatalogBrowseDialog)
    assert received == [_CASTLE]
    assert widget.value() == _CASTLE


def test_browse_opens_on_the_fields_value_pinned(monkeypatch) -> None:
    from descape.constant_picker import CatalogBrowseDialog

    built = []
    monkeypatch.setattr(CatalogBrowseDialog, "exec_", lambda dialog: built.append(dialog) or dialog.Rejected)
    widget = _line_edit()
    widget.set_value(_BLUE_TREE)
    received = []
    widget.committed.connect(received.append)
    widget.browse_button.click()
    assert built[-1].view.current_value() == _BLUE_TREE
    assert not built[-1].show_hidden_checkbox.isChecked()
    assert received == []


def test_set_group_narrows_browse_to_that_category(monkeypatch) -> None:
    built = _accepting_exec(monkeypatch, _CASTLE)
    widget = _line_edit()
    widget.set_group("Buildings")
    widget.browse_button.click()
    view = built[-1].view
    assert {entry.category for entry in view._entries} == {"Buildings"}
    assert view.category_buttons["Buildings"].isChecked()
    assert view.category_block.isHidden()


def test_a_tech_fields_browse_stays_the_plain_tree_with_no_preview(monkeypatch) -> None:
    """Object ids and tech ids are separate spaces that can share a number --
    16 is a tech id and an object const. A portrait, stats or preview would
    answer for an unrelated object, so a tech field shows none."""
    from descape import object_catalog
    from descape.constant_picker import CatalogLineEdit
    from descape.value_picker import ValueBrowseDialog

    built = []
    monkeypatch.setattr(ValueBrowseDialog, "exec_", lambda dialog: built.append(dialog) or dialog.Rejected)
    conftest.ensure_qapp()
    widget = CatalogLineEdit(object_catalog.techs(), hide_internal=True)
    widget.set_value(16)
    widget.browse_button.click()
    dialog = built[-1]
    assert isinstance(dialog, ValueBrowseDialog)
    assert dialog.preview is None
    assert dialog.tree.currentItem() is not None


def _rows(dialog) -> list[tuple[int, bool]]:
    from PyQt5.QtCore import Qt

    return [
        (group.child(c).data(0, Qt.UserRole), group.child(c).isHidden())
        for group in (dialog.tree.topLevelItem(i) for i in range(dialog.tree.topLevelItemCount()))
        for c in range(group.childCount())
    ]


def test_the_tech_picker_hides_internal_techs_until_asked_and_keeps_a_current_one(monkeypatch) -> None:
    from descape import object_catalog
    from descape.constant_picker import INTERNAL_TECHS_LABEL, CatalogLineEdit, items_for
    from descape.value_picker import ValueBrowseDialog

    built = []

    def fake_exec(dialog):
        built.append(dialog)
        return ValueBrowseDialog.Rejected

    monkeypatch.setattr(ValueBrowseDialog, "exec_", fake_exec)
    conftest.ensure_qapp()
    catalog = object_catalog.techs()
    internal = {e.id for e in catalog if e.internal}
    widget = CatalogLineEdit(catalog, hide_internal=True)
    widget._browse()
    dialog = built[-1]
    assert dialog.show_hidden_checkbox.text() == INTERNAL_TECHS_LABEL
    assert not dialog.show_hidden_checkbox.isChecked()
    assert {value for value, hidden in _rows(dialog) if hidden} == internal
    dialog.show_hidden_checkbox.setChecked(True)
    assert not any(hidden for _, hidden in _rows(dialog))

    # A field already holding an internal tech still names it, and Browse opens on it, pickable.
    current = next(e for e in catalog if e.internal)
    widget.set_value(current.id)
    assert widget.line_edit.text() == current.name
    widget._browse()
    dialog = built[-1]
    assert dialog.view.current_value() == current.id
    assert {value for value, hidden in _rows(dialog) if hidden} == internal - {current.id}

    # Only a picker that asks: the default (the Disables dialog's Techs tab) hides nothing.
    assert not any(item.hidden for item in items_for(catalog))
