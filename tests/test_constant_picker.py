"""Coverage for descape/constant_picker.py (phase 4d, slice 2):
CatalogLineEdit's type-ahead/raw-integer entry and commit-on-focus-out
behaviour, and CatalogBrowseDialog's category grouping and filtering.

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


def _dialog_for(catalog, default_category=""):
    conftest.ensure_qapp()
    from descape.constant_picker import CatalogBrowseDialog

    return CatalogBrowseDialog(catalog, default_category)


def _dialog(default_category=""):
    return _dialog_for(_catalog(), default_category)


def test_preview_never_shows_for_a_techs_catalog_even_on_an_id_collision() -> None:
    """Object ids and tech ids are separate spaces that can share a number --
    id 16 is a real object (Anarchy is tech id 16, coincidentally also an
    object const). A Techs dialog must never resolve a preview through the
    object-id sprite lookup for it."""
    from descape import object_catalog

    dialog = _dialog_for(object_catalog.techs())
    dialog.select(16)
    assert dialog.tree.currentItem() is not None
    assert dialog.preview.pixmap() is None or dialog.preview.pixmap().isNull()


def test_dialog_groups_by_category_and_expands_the_default_one() -> None:
    dialog = _dialog(default_category="Buildings")
    groups = {
        dialog.tree.topLevelItem(i).text(0).split(" (")[0]: dialog.tree.topLevelItem(i)
        for i in range(dialog.tree.topLevelItemCount())
    }
    assert set(groups) == {"Units", "Buildings", "Heroes", "Others"}
    assert groups["Buildings"].isExpanded()
    assert not groups["Units"].isExpanded()


def test_select_reveals_and_selects_the_matching_row() -> None:
    from PyQt5.QtCore import Qt

    dialog = _dialog()
    dialog.select(109)
    assert dialog.tree.currentItem() is not None
    assert dialog.tree.currentItem().data(0, Qt.UserRole) == 109


def test_double_clicking_a_row_accepts_it() -> None:
    dialog = _dialog()
    dialog.select(109)
    dialog.tree.itemDoubleClicked.emit(dialog.tree.currentItem(), 0)
    assert dialog.result() == dialog.Accepted
    assert dialog.selected_id() == 109


def test_double_clicking_a_group_heading_does_not_accept() -> None:
    dialog = _dialog()
    group = dialog.tree.topLevelItem(0)
    dialog.tree.setCurrentItem(group)
    dialog.tree.itemDoubleClicked.emit(group, 0)
    assert dialog.result() != dialog.Accepted
    assert dialog.selected_id() is None


def test_filter_hides_non_matching_rows_and_empty_groups() -> None:
    dialog = _dialog()
    dialog.filter_edit.setText("brown bear")
    shown_groups = [
        dialog.tree.topLevelItem(i)
        for i in range(dialog.tree.topLevelItemCount())
        if not dialog.tree.topLevelItem(i).isHidden()
    ]
    assert len(shown_groups) == 1
    group = shown_groups[0]
    visible_children = [group.child(c).text(0) for c in range(group.childCount()) if not group.child(c).isHidden()]
    assert visible_children == ["BROWN BEAR"]


def test_clearing_the_filter_reshows_everything() -> None:
    dialog = _dialog()
    dialog.filter_edit.setText("brown bear")
    dialog.filter_edit.setText("")
    hidden_groups = [i for i in range(dialog.tree.topLevelItemCount()) if dialog.tree.topLevelItem(i).isHidden()]
    assert hidden_groups == []


def _child_for(dialog, object_id):
    from PyQt5.QtCore import Qt

    for i in range(dialog.tree.topLevelItemCount()):
        group = dialog.tree.topLevelItem(i)
        for c in range(group.childCount()):
            child = group.child(c)
            if child.data(0, Qt.UserRole) == object_id:
                return child
    return None


def test_editor_hidden_objects_are_hidden_by_default() -> None:
    """id 768 (BLUE TREE) is a real hide_in_editor=True entry in the shipped
    object_catalog.json."""
    dialog = _dialog()
    assert not dialog.show_hidden_checkbox.isChecked()
    child = _child_for(dialog, 768)
    assert child is not None
    assert child.isHidden()


def test_checking_show_hidden_reveals_editor_hidden_objects() -> None:
    dialog = _dialog()
    dialog.show_hidden_checkbox.setChecked(True)
    child = _child_for(dialog, 768)
    assert child is not None
    assert not child.isHidden()


def test_unchecking_show_hidden_hides_them_again() -> None:
    dialog = _dialog()
    dialog.show_hidden_checkbox.setChecked(True)
    dialog.show_hidden_checkbox.setChecked(False)
    child = _child_for(dialog, 768)
    assert child is not None
    assert child.isHidden()
