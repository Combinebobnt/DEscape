"""Trigger panel: facet arrows, object lists, areas, type rows, the button grid
and Allow In Fog.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _ARCHER,
    _CASTLE,
    _CREATE_OBJECT,
    _REPLACE_OBJECT,
    _SCOUT,
    _appended_effects,
    _assert_form_rows_do_not_overlap,
    _effect_type_id,
    _effects0,
    _facet_window,
    _focus_out_every_row,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _record_count,
    _row_widget,
    _select_entries,
    _spec_named,
    _triggers_window,
    _window,
    check_triggers_fit_their_drag_floor,
)

from testkit.qt_window import panel_width, set_panel_width

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- GH #167: the facet direction preview -----------------------------------------


_TREBUCHET = 42


def _facet_preview(panel):
    preview = panel._facet_preview
    assert preview is not None, "no facet preview on this form"
    return preview


def _form_row_of(form, widget) -> int:
    row, _role = form.getWidgetPosition(widget)
    assert row >= 0, f"{widget} is not in the form"
    return row


def test_a_create_object_facet_gets_an_arrow_row_under_it_that_hides_when_unset() -> None:
    from PyQt5.QtWidgets import QApplication

    from descape import unit_sprites

    window, _indices = _facet_window(_CREATE_OBJECT, {"object_list_unit_id": _ARCHER, "facet": 4})
    try:
        panel = window.trigger_panel
        preview = _facet_preview(panel)
        spin = _row_widget(panel, "facet")
        form = panel.property_form
        assert _form_row_of(form, preview) == _form_row_of(form, spin) + 1, "directly under the facet row"
        assert not preview.isHidden() and preview.arrow_shown
        assert unit_sprites.angle_index(preview.rotation, 16) == 4, "an archer's facet 4 is its stored angle 4"
        assert not preview.sprite_shown, "no install, no sprite"
        _assert_form_rows_do_not_overlap(form)

        spin.setValue(-1)
        QApplication.processEvents()
        assert _facet_preview(panel).isHidden(), "unset is randomized, so nothing to show"
        spin = _row_widget(panel, "facet")
        spin.setValue(6)
        QApplication.processEvents()
        preview = _facet_preview(panel)
        assert not preview.isHidden() and unit_sprites.angle_index(preview.rotation, 16) == 6
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_trebuchets_facet_counts_in_its_own_32_angles() -> None:
    """GH #167, measured: facet 24 on a Trebuchet is its stored angle 24 of 32."""
    from descape import unit_sprites

    window, _indices = _facet_window(_CREATE_OBJECT, {"object_list_unit_id": _TREBUCHET, "facet": 24})
    try:
        preview = _facet_preview(window.trigger_panel)
        assert preview.arrow_shown
        assert unit_sprites.angle_index(preview.rotation, 32) == 24
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_unit_with_no_facing_gets_a_note_and_no_arrow() -> None:
    window, _indices = _facet_window(_CREATE_OBJECT, {"object_list_unit_id": _CASTLE, "facet": 0})
    try:
        preview = _facet_preview(window.trigger_panel)
        assert not preview.isHidden()
        assert not preview.arrow_shown
        assert preview.note_label.text() == "Castle has no facing"
        assert not preview.note_label.isHidden()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_an_unset_unit_still_gets_the_arrow() -> None:
    window, _indices = _facet_window(_CREATE_OBJECT, {"facet": 2})
    try:
        preview = _facet_preview(window.trigger_panel)
        assert preview.arrow_shown and preview.note_label.isHidden()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_replace_object_binds_facet2_to_its_second_unit() -> None:
    """The first unit is an archer and the second a castle, so only a preview
    reading object_list_unit_id_2 says the castle has no facing."""
    window, _indices = _facet_window(
        _REPLACE_OBJECT, {"object_list_unit_id": _ARCHER, "object_list_unit_id_2": _CASTLE, "facet2": 4}
    )
    try:
        panel = window.trigger_panel
        preview = _facet_preview(panel)
        form = panel.property_form
        assert _form_row_of(form, preview) == _form_row_of(form, _row_widget(panel, "facet2")) + 1
        assert preview.note_label.text() == "Castle has no facing"
        _row_widget(panel, "object_list_unit_id_2").set_value(_ARCHER)
        _row_widget(panel, "object_list_unit_id_2").committed.emit(_ARCHER)
        preview = _facet_preview(panel)
        assert preview.arrow_shown and preview.note_label.isHidden()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_replace_objects_unit_rows_are_captioned_after_their_player() -> None:
    """GH #212: the first unit is the source player's and the second the
    target player's, and the undo label follows the row caption."""
    window, _indices = _facet_window(_REPLACE_OBJECT, {"object_list_unit_id": _ARCHER, "object_list_unit_id_2": _CASTLE})
    try:
        panel = window.trigger_panel
        form = panel.property_form
        assert form.labelForField(_row_widget(panel, "object_list_unit_id")).text() == "source player unit"
        assert form.labelForField(_row_widget(panel, "object_list_unit_id_2")).text() == "target player unit"

        unit = _row_widget(panel, "object_list_unit_id")
        unit.line_edit.setText(str(_SCOUT))
        unit.line_edit.editingFinished.emit()
        assert window.edit_history.records[-1].label == "Set effect source player unit"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #41: the object list type row, and the object filters above the Set rows --

_MILITIA, _TOWN_CENTER = 74, 109


_UNKNOWN_CONST = 99999


def _object_list_window(effect: str, *per_entry):
    """The trigger window with `effect` effects appended to trigger 0, the first selected."""
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    indices = _appended_effects(window, _effect_type_id(window.trigger_panel, effect), list(per_entry))
    window.trigger_panel.refresh_entries(select=("effect", indices[0]))
    QApplication.processEvents()
    return window, indices


def _form_captions(form) -> list[str]:
    """Every side caption, top to bottom."""
    from PyQt5.QtWidgets import QFormLayout

    captions = []
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.LabelRole)
        if item is not None and item.widget() is not None:
            captions.append(item.widget().text())
    return captions


def _list_type_above(panel, field: str):
    """The object list type combo on the row directly above `field`'s."""
    from PyQt5.QtWidgets import QFormLayout

    from descape.trigger_panel import TriggerPanel

    form = panel.property_form
    row = _form_row_of(form, _row_widget(panel, field)) - 1
    caption = form.itemAt(row, QFormLayout.LabelRole)
    assert caption is not None and caption.widget().text() == TriggerPanel.LIST_TYPE_CAPTION, f"no type row above {field}"
    return form.itemAt(row, QFormLayout.FieldRole).widget()


def _suggestions(picker) -> set[str]:
    return set(picker.line_edit.completer().model().stringList())


def _catalog_names(category: str) -> set[str]:
    from descape import object_catalog

    return {entry.name for entry in object_catalog.objects() if entry.category == category}


def test_kill_object_reads_the_object_filters_above_its_set_rows() -> None:
    window, _indices = _object_list_window("kill_object", {})
    try:
        assert _form_captions(window.trigger_panel.property_form) == [
            "object list type",
            "source player unit",
            "object group",
            "object type",
            "source player",
            "Area",
            "selected object ids",
            "Objects",
            "max units affected",
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_objects_in_area_reads_its_type_row_and_filters_under_its_quantity() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(0)
        index = len(panel._manager().triggers[0].conditions)
        condition = next(i for i, d in panel._vocabulary.conditions.items() if d.name == "objects_in_area")
        window.entry_structural_edit("new", 0, "condition", -1, condition)
        panel.refresh_entries(select=("condition", index))
        QApplication.processEvents()
        assert _form_captions(panel.property_form)[:6] == [
            "quantity",
            "object list type",
            "object list",
            "object group",
            "object type",
            "source player",
        ]
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #277: an area's four boxes, as the in-game editor lays them out -----------


def _area_window(kind: str):
    """The trigger window with a condition or an effect that has an area selected.
    Not objects_in_area: its long checkbox caption alone overflows the narrowest pane."""
    from PyQt5.QtWidgets import QApplication

    if kind == "effect":
        return _object_list_window("kill_object", {})[0]
    window = _triggers_window()
    panel = window.trigger_panel
    panel.select_trigger(0)
    index = len(panel._manager().triggers[0].conditions)
    condition = next(i for i, d in panel._vocabulary.conditions.items() if d.name == "bring_object_to_area")
    window.entry_structural_edit("new", 0, "condition", -1, condition)
    panel.refresh_entries(select=("condition", index))
    QApplication.processEvents()
    return window


def _area_boxes(panel) -> list:
    from descape import trigger_geometry

    return [_row_widget(panel, field) for field in trigger_geometry.AREA_FIELDS]


@pytest.mark.font_sensitive
@pytest.mark.parametrize("kind", ["condition", "effect"])
def test_an_area_shows_two_rows_of_two_boxes_with_their_captions_above(kind: str) -> None:
    from PyQt5.QtWidgets import QLabel

    window = _area_window(kind)
    try:
        panel = window.trigger_panel
        boxes = _area_boxes(panel)
        x1, y1, x2, y2 = boxes
        assert x1.y() == y1.y() < x2.y() == y2.y()
        assert x1.x() == x2.x() < y1.x() == y2.x()

        grid = x1.parentWidget()
        captions = grid.findChildren(QLabel)
        for box, text in zip(boxes, ("X1", "Y1", "X2", "Y2"), strict=True):
            caption = next(label for label in captions if label.buddy() is box)
            assert caption.text() == text
            assert caption.x() == box.x() and caption.geometry().bottom() < box.y()

        assert panel._group_rows["area"].set_button.parentWidget().y() > grid.geometry().bottom()
        _assert_form_rows_do_not_overlap(panel.property_form)

        # A box's inner line edit is in the chain too, so count only the boxes.
        reached, widget = [], x1
        for _ in range(32):
            widget = widget.nextInFocusChain()
            if widget in boxes and widget not in reached:
                reached.append(widget)
        assert reached[:3] == [y1, x2, y2]

        # The floor itself, whatever the font: 9 pt fits a narrow pane even without it.
        # check_triggers_fit_their_drag_floor covers area forms at the real pane floor.
        assert grid.minimumSizeHint().width() < 2 * x1.sizeHint().width()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.font_sensitive
def test_triggers_mode_cannot_be_dragged_below_its_panel() -> None:
    """TASK-560, at the pinned font: check_triggers_fit_their_drag_floor's conditions; leaving
    Triggers gives the pane back its plain floor."""
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    window = _triggers_window()
    try:
        check_triggers_fit_their_drag_floor(window)
        window.mode_combo.setCurrentText("View")
        QApplication.processEvents()
        set_panel_width(window, settings.MIN_SPLIT_PANE)
        QApplication.processEvents()
        assert panel_width(window) == settings.MIN_SPLIT_PANE
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_selection_of_areas_shows_the_boxes_without_a_set_row() -> None:
    window, indices = _object_list_window("kill_object", {}, {})
    try:
        panel = window.trigger_panel
        _select_entries(panel, *[("effect", index) for index in indices])
        x1, y1, x2, y2 = _area_boxes(panel)
        assert x1.y() == y1.y() < x2.y() == y2.y()
        assert x1.x() == x2.x() < y1.x() == y2.x()
        assert "area" not in panel._group_rows
        assert x1.toolTip() == "area x1", "with no Set row, the tooltip is what still says area"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_type_row_shows_the_values_kind_else_units() -> None:
    from descape.trigger_panel import TriggerPanel

    window, indices = _object_list_window(
        "kill_object",
        {"object_list_unit_id": _CASTLE},
        {"object_list_unit_id": _MILITIA},
        {},
        {"object_list_unit_id": _UNKNOWN_CONST},
    )
    try:
        panel = window.trigger_panel
        shown = []
        for index in indices:
            _select_entries(panel, ("effect", index))
            combo = _list_type_above(panel, "object_list_unit_id")
            shown.append(combo.currentText())
            assert combo.isEnabled()
            assert combo.toolTip() == TriggerPanel._LIST_TYPE_TIP
            assert panel.property_form.labelForField(combo).toolTip() == TriggerPanel._LIST_TYPE_TIP
        assert shown == ["Buildings", "Units", "Units", "Units"]
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_choosing_a_type_narrows_the_field_and_writes_nothing() -> None:
    window, indices = _object_list_window("kill_object", {"object_list_unit_id": _MILITIA})
    try:
        panel = window.trigger_panel
        picker = _row_widget(panel, "object_list_unit_id")
        assert _suggestions(picker) == _catalog_names("Units")
        records = len(window.edit_history.records)

        _list_type_above(panel, "object_list_unit_id").setCurrentText("Buildings")
        assert _suggestions(picker) == _catalog_names("Buildings")
        assert len(window.edit_history.records) == records and not window.edit_history.is_dirty
        assert _effects0(window)[indices[0]].object_list_unit_id == _MILITIA
        assert picker.value() == _MILITIA, "the field still shows its value"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_chosen_type_survives_a_repopulate_and_resets_with_the_selection() -> None:
    """The form rebuilds after a Set or Reset; the choice lasts while the same effect stays selected."""
    window, indices = _object_list_window("kill_object", {}, {})
    try:
        panel = window.trigger_panel
        first = _list_type_above(panel, "object_list_unit_id")
        first.setCurrentText("Buildings")

        area = {"area_x1": 2, "area_y1": 3, "area_x2": 6, "area_y2": 7}
        window.set_entry_field_group(0, ("effect", indices[0]), area, "Set area")
        rebuilt = _list_type_above(panel, "object_list_unit_id")
        assert rebuilt is not first, "fixture assumption: the group write repopulates the form"
        assert rebuilt.currentText() == "Buildings"
        assert _suggestions(_row_widget(panel, "object_list_unit_id")) == _catalog_names("Buildings")

        _select_entries(panel, ("effect", indices[1]))
        assert _list_type_above(panel, "object_list_unit_id").currentText() == "Units"
        _select_entries(panel, ("effect", indices[0]))
        assert _list_type_above(panel, "object_list_unit_id").currentText() == "Units"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_committing_a_name_of_another_kind_turns_the_row() -> None:
    from descape import object_catalog

    window, indices = _object_list_window("kill_object", {})
    try:
        panel = window.trigger_panel
        combo = _list_type_above(panel, "object_list_unit_id")
        combo.setCurrentText("Heroes")
        picker = _row_widget(panel, "object_list_unit_id")
        picker.line_edit.setText(object_catalog.name_for(object_catalog.objects(), _CASTLE))
        picker.line_edit.editingFinished.emit()

        assert _effects0(window)[indices[0]].object_list_unit_id == _CASTLE
        assert combo.currentText() == "Buildings"
        assert _suggestions(picker) == _catalog_names("Buildings")
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("accept", [True, False], ids=["pick", "cancel"])
def test_a_browse_pick_of_another_kind_turns_the_row(monkeypatch, accept: bool) -> None:
    """GH #273: Browse shows every category, so its pick can turn the row like a typed name."""
    from descape.constant_picker import CatalogBrowseDialog

    def fake_exec(dialog):
        dialog.view.select(_CASTLE)
        if accept:
            dialog._accept_current()
        else:
            dialog.reject()
        return dialog.result()

    monkeypatch.setattr(CatalogBrowseDialog, "exec_", fake_exec)
    window, indices = _object_list_window("kill_object", {})
    try:
        panel = window.trigger_panel
        combo = _list_type_above(panel, "object_list_unit_id")
        combo.setCurrentText("Heroes")
        picker = _row_widget(panel, "object_list_unit_id")
        picker.browse_button.click()

        kind = "Buildings" if accept else "Heroes"
        assert (_effects0(window)[indices[0]].object_list_unit_id == _CASTLE) == accept
        assert combo.currentText() == kind
        assert _suggestions(picker) == _catalog_names(kind)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_replace_object_has_a_type_row_above_each_unit_field() -> None:
    window, _indices = _object_list_window("replace_object", {"object_list_unit_id": _MILITIA, "object_list_unit_id_2": _CASTLE})
    try:
        panel = window.trigger_panel
        assert _form_captions(panel.property_form).count("object list type") == 2
        source = _list_type_above(panel, "object_list_unit_id")
        target = _list_type_above(panel, "object_list_unit_id_2")
        assert (source.currentText(), target.currentText()) == ("Units", "Buildings")

        source.setCurrentText("Heroes")
        assert _suggestions(_row_widget(panel, "object_list_unit_id")) == _catalog_names("Heroes")
        assert _suggestions(_row_widget(panel, "object_list_unit_id_2")) == _catalog_names("Buildings"), "each row filters its own field"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_build_objects_building_list_starts_on_buildings() -> None:
    """The one BuildingInfo field (1.57+): unset, it starts on its picker's own category."""
    window, _indices = _object_list_window("build_object", {})
    try:
        panel = window.trigger_panel
        assert _list_type_above(panel, "object_list_unit_id").currentText() == "Units"
        assert _list_type_above(panel, "building_list").currentText() == "Buildings"
        assert _suggestions(_row_widget(panel, "building_list")) == _catalog_names("Buildings")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_type_row_is_disabled_on_a_read_only_file() -> None:
    window, indices = _object_list_window("kill_object", {"object_list_unit_id": _MILITIA})
    try:
        panel = window.trigger_panel
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        panel.select_trigger(0)
        _select_entries(panel, ("effect", indices[0]))
        assert not panel._editable
        assert not _list_type_above(panel, "object_list_unit_id").isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_two_effects_with_different_objects_get_the_row_on_units() -> None:
    """Both are buildings, but "(differs)" is no one value's kind."""
    window, indices = _object_list_window("kill_object", {"object_list_unit_id": _CASTLE}, {"object_list_unit_id": _TOWN_CENTER})
    try:
        panel = window.trigger_panel
        _select_entries(panel, ("effect", indices[0]), ("effect", indices[1]))
        assert _row_widget(panel, "object_list_unit_id").line_edit.placeholderText() == panel._DIFFERS
        assert _list_type_above(panel, "object_list_unit_id").currentText() == "Units"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_selection_whose_facets_differ_hides_the_preview() -> None:
    window, indices = _facet_window(
        _CREATE_OBJECT,
        {"object_list_unit_id": _ARCHER, "facet": 4},
        {"object_list_unit_id": _ARCHER, "facet": 8},
        {"object_list_unit_id": _SCOUT, "facet": 4},
    )
    try:
        panel = window.trigger_panel
        _select_entries(panel, ("effect", indices[0]), ("effect", indices[1]))
        assert _row_widget(panel, "facet").is_indeterminate()
        assert _facet_preview(panel).isHidden()

        _select_entries(panel, ("effect", indices[0]), ("effect", indices[2]))
        preview = _facet_preview(panel)
        assert not preview.isHidden() and preview.arrow_shown, "a shared facet shows, the differing unit drops to the arrow"
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_clearing_the_form_detaches_the_facet_preview() -> None:
    from PyQt5.QtWidgets import QApplication

    window, _indices = _facet_window(_CREATE_OBJECT, {"object_list_unit_id": _ARCHER, "facet": 4})
    try:
        panel = window.trigger_panel
        first = _facet_preview(panel)
        _select_entries(panel, ("effect", 0))
        QApplication.processEvents()
        assert panel._facet_preview is None
        assert first._fields == {} and first._on_resize is None, "detached"
        assert first.parent() is None, "unparented with its row"
        _select_entries(panel, ("effect", _indices[0]))
        assert panel._facet_preview is not first
        assert panel.property_host.findChildren(type(first)) == [panel._facet_preview]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_sprite_comes_from_the_install_and_is_cached_per_facet(monkeypatch) -> None:
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    from descape import constant_picker, facet_preview

    calls = []

    def fake(unit, rotation):
        calls.append((unit, rotation))
        pixmap = QPixmap(8, 8)
        pixmap.fill()
        return pixmap

    monkeypatch.setattr(constant_picker, "preview_pixmap", fake)
    window, _indices = _facet_window(_CREATE_OBJECT, {"object_list_unit_id": _ARCHER, "facet": 4})
    try:
        panel = window.trigger_panel
        preview = _facet_preview(panel)
        assert preview.sprite_shown
        assert calls == [(_ARCHER, pytest.approx(facet_preview.facet_rotation(4)))]
        for facet in (5, 4, 5):
            preview._fields["facet"].blockSignals(True)
            preview._fields["facet"].setValue(facet)
            preview._fields["facet"].blockSignals(False)
            preview.refresh()
        QApplication.processEvents()
        assert [unit for unit, _r in calls] == [_ARCHER, _ARCHER], "4 and 5 decode once each"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #169: the button grid under button_location ------------------------------


@pytest.mark.parametrize("type_id", [46, 47, 102])
def test_the_button_grid_sits_under_button_location(type_id: int) -> None:
    window, _indices = _facet_window(type_id, {"button_location": 1})
    try:
        panel = window.trigger_panel
        grid = panel._button_grid
        assert grid is not None
        form = panel.property_form
        assert _form_row_of(form, grid) == _form_row_of(form, _row_widget(panel, "button_location")) + 1
        assert form.labelForField(_row_widget(panel, "button_location")).text() == "button slot"
        _assert_form_rows_do_not_overlap(form)
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_other_effects_get_no_button_grid() -> None:
    from descape.button_grid import ButtonGrid

    window, _indices = _facet_window(_CREATE_OBJECT, {"object_list_unit_id": _ARCHER})
    try:
        panel = window.trigger_panel
        assert panel._button_grid is None
        assert panel.property_host.findChildren(ButtonGrid) == []
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- TASK-059: Allow In Fog on Object Visible Multiplayer -----------------------

# tools/gen_v159_fixture.py: trigger 0 condition 1 holds allow_in_fog 1, trigger 1 condition 0 holds -1.
V159_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "v159_units_triggers.aoe2scenario"


_FOG_TICKED, _FOG_UNSET = (0, 1), (1, 0)


def _fog_window():
    window = _window()
    window.load_scenario(V159_FIXTURE)
    window.mode_combo.setCurrentText("Triggers")
    return window


def _fog_box(panel, where: tuple[int, int]):
    trigger, condition = where
    panel.select_trigger(trigger)
    _select_entries(panel, ("condition", condition))
    return _row_widget(panel, "allow_in_fog")


def _fog(window, where: tuple[int, int]) -> int:
    trigger, condition = where
    return window.trigger_panel._manager().triggers[trigger].conditions[condition].allow_in_fog


def test_allow_in_fog_shows_one_ticked_and_unset_unticked_with_the_games_tooltip() -> None:
    from PyQt5.QtWidgets import QCheckBox

    window = _fog_window()
    try:
        panel = window.trigger_panel
        ticked = _fog_box(panel, _FOG_TICKED)
        assert isinstance(ticked, QCheckBox) and ticked.isChecked() and not ticked.isTristate()
        assert ticked.isEnabled()
        label = panel.property_form.labelForField(ticked)
        assert label is not None and label.text() == "allow in fog"
        assert "passes the condition even if the object is in the Fog of War" in ticked.toolTip()
        assert label.toolTip() == ticked.toolTip()
        entry = panel._manager().triggers[0].conditions[1]
        assert "allow_in_fog=1" in panel._describe("condition", entry)[1]

        unset = _fog_box(panel, _FOG_UNSET)
        assert isinstance(unset, QCheckBox) and not unset.isChecked() and not unset.isTristate()
        assert "allow_in_fog" not in panel._describe("condition", panel._manager().triggers[1].conditions[0])[1]
        assert window.trigger_edits is None, "building the form must record nothing"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_ticking_unset_allow_in_fog_stores_one_and_undo_restores_unset(tmp_path: Path) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _fog_window()
    try:
        panel = window.trigger_panel
        records = _record_count(window)
        _fog_box(panel, _FOG_UNSET).setChecked(True)
        assert (_fog(window, _FOG_UNSET), _record_count(window)) == (1, records + 1)

        window.undo()
        assert _fog(window, _FOG_UNSET) == -1
        assert not window.edit_history.is_dirty
        assert not _fog_box(panel, _FOG_UNSET).isChecked()

        _fog_box(panel, _FOG_UNSET).setChecked(True)
        box = _fog_box(panel, _FOG_UNSET)
        assert box.isChecked()
        box.setChecked(False)
        assert _fog(window, _FOG_UNSET) == 0
        assert not _fog_box(panel, _FOG_UNSET).isChecked()

        out = tmp_path / "fog.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        conditions = parse_triggers(load_map_and_units(out)).triggers
        assert (conditions[0].conditions[1].allow_in_fog, conditions[1].conditions[0].allow_in_fog) == (1, 0)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_populating_and_tabbing_past_unset_allow_in_fog_writes_nothing() -> None:
    window = _fog_window()
    try:
        panel = window.trigger_panel
        records = _record_count(window)
        _fog_box(panel, _FOG_UNSET)
        _focus_out_every_row(panel)
        assert _fog(window, _FOG_UNSET) == -1
        assert _record_count(window) == records
        assert window.trigger_edits is None and not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_selection_of_unset_and_zero_allow_in_fog_shows_one_unticked_box() -> None:
    """1 beside -1 differs; 0 beside -1 shows plain unticked, as each shows
    alone. Ticking that pair writes 1 to both, unticking writes 0 to both."""
    from PyQt5.QtCore import Qt

    window = _fog_window()
    try:
        panel = window.trigger_panel
        window.entry_structural_edit("new", 0, "condition", -1, 27)
        fresh = (0, len(panel._manager().triggers[0].conditions) - 1)
        assert _fog(window, fresh) == -1
        pair = [("condition", _FOG_TICKED[1]), ("condition", fresh[1])]

        panel.select_trigger(0)
        _select_entries(panel, *pair)
        box = _row_widget(panel, "allow_in_fog")
        assert box.isTristate() and box.checkState() == Qt.PartiallyChecked

        entry = panel._manager().triggers[0].conditions[1]
        window.set_entry_field(0, "condition", 1, _spec_named(panel, "condition", entry, "allow_in_fog"), 0)
        _select_entries(panel, *pair)
        box = _row_widget(panel, "allow_in_fog")
        assert not box.isTristate() and box.checkState() == Qt.Unchecked

        records = _record_count(window)
        box.setChecked(True)
        assert (_fog(window, _FOG_TICKED), _fog(window, fresh), _record_count(window)) == (1, 1, records + 1)
        box.setChecked(False)
        assert (_fog(window, _FOG_TICKED), _fog(window, fresh)) == (0, 0)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_unticking_a_differing_allow_in_fog_selection_writes_zero_to_every_entry() -> None:
    from PyQt5.QtCore import Qt

    window = _fog_window()
    try:
        panel = window.trigger_panel
        window.entry_structural_edit("new", 0, "condition", -1, 27)
        fresh = (0, len(panel._manager().triggers[0].conditions) - 1)
        panel.select_trigger(0)
        _select_entries(panel, ("condition", _FOG_TICKED[1]), ("condition", fresh[1]))
        box = _row_widget(panel, "allow_in_fog")
        assert box.checkState() == Qt.PartiallyChecked
        box.setCheckState(Qt.Unchecked)
        assert (_fog(window, _FOG_TICKED), _fog(window, fresh)) == (0, 0)
    finally:
        window.edit_history.mark_saved()
        window.close()
