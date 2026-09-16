"""Coverage for descape/value_picker.py (S1 of the units-sidebar-catalog
work): PickerItem, ValuePickerView's population/filter/preview/selection,
ValueBrowseDialog's thin-shell aliasing, and ValueLineEdit's type-ahead/
raw-integer commit behaviour.

Standalone widgets, not through a ViewerWindow: conftest.ensure_qapp() is
all the Qt setup any of them need.
"""

from __future__ import annotations

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

import conftest
from descape.value_picker import (
    PickerItem,
    ValueBrowseDialog,
    ValueLineEdit,
    ValuePickerView,
)

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _flat_items():
    return [
        PickerItem(label="Hit Points", value=0),
        PickerItem(label="Attack", value=1),
        PickerItem(label="Line of Sight", value=5),
    ]


def _grouped_items():
    return [
        PickerItem(label="Town Center", value=109, group="Buildings"),
        PickerItem(label="Archer", value=4, group="Units"),
        PickerItem(label="Brown Bear", value=48, group="Units"),
        PickerItem(label="Flag", value=200, group="Others"),
        PickerItem(label="Blue Tree", value=768, group="Others", hidden=True),
    ]


def _view(items=None, **kwargs):
    conftest.ensure_qapp()
    return ValuePickerView(items if items is not None else _flat_items(), **kwargs)


# -- PickerItem ---------------------------------------------------------


def test_search_text_defaults_from_label_and_value() -> None:
    item = PickerItem(label="Town Center", value=109)
    assert item.search_text == "town center 109"


def test_search_text_explicit_value_is_kept() -> None:
    item = PickerItem(label="Town Center", value=109, search_text="custom")
    assert item.search_text == "custom"


# -- ValuePickerView: rendering -------------------------------------------


def test_show_values_false_renders_the_bare_label() -> None:
    view = _view([PickerItem(label="Town Center", value=109)], show_values=False)
    assert view.tree.topLevelItem(0).text(0) == "Town Center"


def test_show_values_true_renders_label_and_value() -> None:
    view = _view([PickerItem(label="Hit Points", value=0)], show_values=True)
    assert view.tree.topLevelItem(0).text(0) == "Hit Points (0)"


# -- ValuePickerView: population ------------------------------------------


def test_flat_items_have_no_group_headings() -> None:
    view = _view(_flat_items())
    assert view.tree.topLevelItemCount() == len(_flat_items())
    assert all(view.tree.topLevelItem(i).childCount() == 0 for i in range(view.tree.topLevelItemCount()))


def test_flat_items_preserve_caller_order() -> None:
    view = _view(_flat_items())
    labels = [view.tree.topLevelItem(i).text(0) for i in range(view.tree.topLevelItemCount())]
    assert labels == ["Hit Points", "Attack", "Line of Sight"]


def test_grouped_items_are_grouped_and_sorted_within_group() -> None:
    view = _view(_grouped_items(), default_group="Units")
    groups = {
        view.tree.topLevelItem(i).text(0).split(" (")[0]: view.tree.topLevelItem(i)
        for i in range(view.tree.topLevelItemCount())
    }
    assert set(groups) == {"Buildings", "Units", "Others"}
    assert groups["Units"].isExpanded()
    assert not groups["Buildings"].isExpanded()
    unit_labels = [groups["Units"].child(c).text(0) for c in range(groups["Units"].childCount())]
    assert unit_labels == ["Archer", "Brown Bear"]


def test_set_items_replaces_and_repopulates() -> None:
    view = _view(_flat_items())
    view.set_items([PickerItem(label="Only One", value=99)])
    assert view.tree.topLevelItemCount() == 1
    assert view.tree.topLevelItem(0).text(0) == "Only One"


# -- ValuePickerView: filtering --------------------------------------------


def test_filter_matches_by_id_via_search_text() -> None:
    view = _view(_grouped_items())
    view.filter_edit.setText("109")
    shown_groups = [
        view.tree.topLevelItem(i)
        for i in range(view.tree.topLevelItemCount())
        if not view.tree.topLevelItem(i).isHidden()
    ]
    assert len(shown_groups) == 1
    group = shown_groups[0]
    visible = [group.child(c).text(0) for c in range(group.childCount()) if not group.child(c).isHidden()]
    assert visible == ["Town Center"]


def test_clearing_the_filter_reshows_everything() -> None:
    view = _view(_grouped_items())
    view.filter_edit.setText("brown bear")
    view.filter_edit.setText("")
    hidden = [i for i in range(view.tree.topLevelItemCount()) if view.tree.topLevelItem(i).isHidden()]
    assert hidden == []


def test_hidden_label_empty_omits_the_checkbox() -> None:
    view = _view(_grouped_items(), hidden_label="")
    assert view.show_hidden_checkbox is None


def test_hidden_items_are_hidden_by_default_and_shown_when_opted_into() -> None:
    view = _view(_grouped_items(), hidden_label="Show hidden")
    assert view.show_hidden_checkbox is not None
    assert not view.show_hidden_checkbox.isChecked()

    def _child(value):
        for i in range(view.tree.topLevelItemCount()):
            group = view.tree.topLevelItem(i)
            for c in range(group.childCount()):
                if group.child(c).data(0, Qt.UserRole) == value:
                    return group.child(c)
        return None

    assert _child(768).isHidden()
    view.show_hidden_checkbox.setChecked(True)
    assert not _child(768).isHidden()
    view.show_hidden_checkbox.setChecked(False)
    assert _child(768).isHidden()


# -- ValuePickerView: selection and current_value --------------------------


def test_select_reveals_and_selects_the_matching_row() -> None:
    view = _view(_grouped_items())
    view.select(48)
    assert view.tree.currentItem() is not None
    assert view.tree.currentItem().data(0, Qt.UserRole) == 48


def test_current_value_returns_the_selected_value() -> None:
    view = _view(_flat_items())
    view.select(1)
    assert view.current_value() == 1


def test_current_value_is_none_for_a_group_heading() -> None:
    view = _view(_grouped_items())
    heading = view.tree.topLevelItem(0)
    view.tree.setCurrentItem(heading)
    assert view.current_value() is None


def test_current_value_is_none_for_a_filtered_out_row() -> None:
    view = _view(_flat_items())
    view.select(1)
    view.filter_edit.setText("nothing matches this")
    assert view.current_value() is None


def test_current_value_zero_is_not_treated_as_falsy() -> None:
    """0 is a legal value -- must survive `is None`-style guards intact."""
    view = _view(_flat_items())
    view.select(0)
    assert view.current_value() == 0


def test_current_changed_emits_on_selection() -> None:
    view = _view(_flat_items())
    received = []
    view.current_changed.connect(received.append)
    view.select(1)
    assert 1 in received


# -- ValuePickerView: activation --------------------------------------------


def test_activated_fires_on_item_activated() -> None:
    view = _view(_flat_items())
    received = []
    view.activated.connect(received.append)
    item = view.tree.topLevelItem(1)
    view.tree.itemActivated.emit(item, 0)
    assert received == [1]


def test_activated_does_not_fire_for_a_group_heading() -> None:
    view = _view(_grouped_items())
    received = []
    view.activated.connect(received.append)
    heading = view.tree.topLevelItem(0)
    view.tree.itemActivated.emit(heading, 0)
    assert received == []


# -- ValuePickerView: preview hook ------------------------------------------


def test_preview_is_none_without_a_preview_hook() -> None:
    view = _view(_flat_items())
    assert view.preview is None


def test_preview_hook_is_called_with_the_picker_item() -> None:
    seen = []

    def hook(item):
        seen.append(item)
        pixmap = QPixmap(4, 4)
        pixmap.fill(Qt.red)
        return pixmap

    view = _view(_flat_items(), preview=hook)
    assert view.preview is not None
    view.select(1)
    assert len(seen) == 1
    assert seen[0].value == 1
    assert not view.preview.pixmap().isNull()


def test_preview_clears_for_a_group_heading() -> None:
    def hook(item):
        pixmap = QPixmap(4, 4)
        pixmap.fill(Qt.red)
        return pixmap

    view = _view(_grouped_items(), preview=hook)
    view.select(4)
    assert not view.preview.pixmap().isNull()
    heading = view.tree.topLevelItem(0)
    view.tree.setCurrentItem(heading)
    assert view.preview.pixmap().isNull()


# -- ValuePickerView: scrolling setup ----------------------------------------


def test_tree_is_configured_for_horizontal_scroll_not_elision() -> None:
    view = _view(_flat_items())
    header = view.tree.header()
    assert header.stretchLastSection() is False
    assert view.tree.textElideMode() == Qt.ElideNone
    assert view.tree.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded


# -- ValueBrowseDialog -------------------------------------------------------


def _dialog(items=None, **kwargs):
    conftest.ensure_qapp()
    return ValueBrowseDialog(items if items is not None else _grouped_items(), **kwargs)


def test_dialog_aliases_the_views_widgets() -> None:
    dialog = _dialog(hidden_label="Show hidden")
    assert dialog.tree is dialog.view.tree
    assert dialog.filter_edit is dialog.view.filter_edit
    assert dialog.show_hidden_checkbox is dialog.view.show_hidden_checkbox


def test_dialog_select_then_accept_reports_the_value() -> None:
    dialog = _dialog()
    dialog.select(48)
    dialog._accept_current()
    assert dialog.result() == dialog.Accepted
    assert dialog.selected_value() == 48


def test_dialog_activating_a_row_accepts_it() -> None:
    dialog = _dialog()
    dialog.select(48)
    dialog.tree.itemActivated.emit(dialog.tree.currentItem(), 0)
    assert dialog.result() == dialog.Accepted
    assert dialog.selected_value() == 48


def test_dialog_activating_a_group_heading_does_not_accept() -> None:
    dialog = _dialog()
    heading = dialog.tree.topLevelItem(0)
    dialog.tree.setCurrentItem(heading)
    dialog.tree.itemActivated.emit(heading, 0)
    assert dialog.result() != dialog.Accepted
    assert dialog.selected_value() is None


# -- ValueLineEdit ------------------------------------------------------------


def _line_edit(items=None, **kwargs):
    conftest.ensure_qapp()
    return ValueLineEdit(items if items is not None else _flat_items(), **kwargs)


def test_set_value_renders_bare_label_when_show_values_false() -> None:
    widget = _line_edit(_grouped_items())
    widget.set_value(109)
    assert widget.line_edit.text() == "Town Center"


def test_set_value_renders_label_and_value_when_show_values_true() -> None:
    widget = _line_edit(show_values=True)
    widget.set_value(1)
    assert widget.line_edit.text() == "Attack (1)"


def test_set_value_uncovered_falls_back_to_raw_id_without_show_values() -> None:
    widget = _line_edit(_grouped_items())
    widget.set_value(999999)
    assert widget.line_edit.text() == "999999"


def test_set_value_uncovered_renders_unknown_with_show_values() -> None:
    widget = _line_edit(show_values=True)
    widget.set_value(999999)
    assert widget.line_edit.text() == "unknown (999999)"


def test_set_value_none_clears_the_field() -> None:
    widget = _line_edit(_grouped_items())
    widget.set_value(109)
    widget.set_value(None)
    assert widget.value() is None
    assert widget.line_edit.text() == ""


def test_typing_a_bare_label_commits_its_value() -> None:
    widget = _line_edit(show_values=True)
    received = []
    widget.committed.connect(received.append)
    widget.line_edit.setText("Attack")
    widget.line_edit.editingFinished.emit()
    assert received == [1]


def test_typing_a_raw_integer_commits_it_even_if_uncovered() -> None:
    widget = _line_edit()
    received = []
    widget.committed.connect(received.append)
    widget.line_edit.setText("999999")
    widget.line_edit.editingFinished.emit()
    assert received == [999999]


def test_an_unrecognised_label_is_rejected_and_the_field_is_restored() -> None:
    widget = _line_edit(_grouped_items())
    widget.set_value(109)
    received = []
    widget.committed.connect(received.append)
    widget.line_edit.setText("not a real object")
    widget.line_edit.editingFinished.emit()
    assert received == []
    assert widget.value() == 109
    assert widget.line_edit.text() == "Town Center"


def test_a_focus_out_with_the_same_value_commits_nothing() -> None:
    widget = _line_edit(_grouped_items())
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
