"""Trigger panel: XS script-call and prose fields, multi-line boxes and their
stored heights.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _assert_form_rows_do_not_overlap,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _plant_prose_effect,
    _row_widget,
    _triggers_window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- XS script fields (P5-a) -------------------------------------------------


def _script_call_id(window, kind: str) -> int:
    from descape import library_compat

    vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
    entries = vocabulary.conditions if kind == "condition" else vocabulary.effects
    return next(e.id for e in entries.values() if e.name == "script_call")


def _plant_script_call(window, kind: str, stored: str) -> str:
    """Add a script_call to trigger 0 of the parsed manager, pre-model, holding
    `stored`, and select it. Returns the XS field's name. Not routed through
    the model on purpose: the assertion is that browsing it builds none."""
    panel = window.trigger_panel
    trigger = panel._manager().triggers[0]
    field = "xs_function" if kind == "condition" else "message"
    if kind == "condition":
        trigger._add_condition(_script_call_id(window, kind))
        entries = trigger.conditions
    else:
        trigger._add_effect(_script_call_id(window, kind))
        entries = trigger.effects
    setattr(entries[-1], field, stored)
    panel.select_trigger(0)
    panel.refresh_entries(select=(kind, len(entries) - 1))
    return field


@pytest.mark.parametrize("kind", ["condition", "effect"])
def test_a_script_call_field_is_a_multi_line_editor_with_its_lines(kind: str) -> None:
    from descape.text_edits import XsTextEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        field = _plant_script_call(window, kind, "void f()\r{\r  int a = 0;\r}")
        widget = _row_widget(panel, field)
        assert isinstance(widget, XsTextEdit)
        assert widget.toPlainText() == "void f()\n{\n  int a = 0;\n}"
        assert widget.document().blockCount() == 4
        assert window.trigger_edits is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("stored", ["a\rb", "a\r\nb", "a\nb"], ids=["CR", "CRLF", "LF"])
def test_an_untouched_xs_focus_out_records_nothing_whatever_the_separator(stored: str) -> None:
    """CRLF and LF are the cases _changed()'s equality check alone would miss:
    they display as "a\\nb" and translate back to "a\\rb", which differs from
    what is stored. Only the latch keeps a bare focus-out a no-op."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        field = _plant_script_call(window, "effect", stored)
        _row_widget(panel, field).editingFinished.emit()
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_typing_into_an_xs_field_records_nothing_until_focus_out() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        field = _plant_script_call(window, "effect", "a\rb")
        widget = _row_widget(panel, field)
        widget.appendPlainText("c")
        widget.appendPlainText("d")
        assert window.trigger_edits is None, "an edit must wait for editingFinished"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_editing_an_xs_field_writes_cr_separated_bytes(tmp_path: Path) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        window.entry_structural_edit("new", 0, "effect", -1, _script_call_id(window, "effect"))
        effect_index = len(window.trigger_edits.manager().triggers[0].effects) - 1
        panel.select_trigger(0)
        panel.refresh_entries(select=("effect", effect_index))
        steps = window.edit_history.cursor

        widget = _row_widget(panel, "message")
        widget.setPlainText("void f()\n{\n  // a comment\n  int a = 0;\n}")
        widget.editingFinished.emit()
        assert window.edit_history.cursor == steps + 1
        widget.editingFinished.emit()
        assert window.edit_history.cursor == steps + 1, "a second focus-out is not a second edit"

        out = tmp_path / "xs.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded = parse_triggers(load_map_and_units(out))
        assert reloaded.triggers[0].effects[effect_index].message == (
            "void f()\r{\r  // a comment\r  int a = 0;\r}"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_locked_xs_field_renders_its_lines() -> None:
    from PyQt5.QtWidgets import QLabel

    from descape.trigger_fields import STR, XS, FieldSpec

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_script_call(window, "effect", "a\rb")
        spec = FieldSpec("message", STR, sentinel=None, read_only=True, multiline=XS)
        label = panel._build_widget(spec, [("effect", 0)], panel._read(panel._current_entry()[2], spec.attribute))
        assert isinstance(label, QLabel)
        assert label.text() == "a\nb"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- prose fields ------------------------------------------------------------

# One long line with no newline in it: the dominant real shape of
# display_instructions.message, and what the XS widget renders as a single
# line behind a horizontal scrollbar.
_LONG_LINE = (
    "Defend the town centre until the timer runs out, then escort the "
    "relic cart to the monastery on the far side of the river before "
    "Attila's cavalry reaches the ford."
)


def _plant_trigger_description(window, stored: str) -> None:
    """The trigger's own description, which flows through the same
    _build_widget/_changed path as an effect field."""
    panel = window.trigger_panel
    panel._manager().triggers[0].description = stored
    panel.select_trigger(0)
    panel.refresh_entries()


def _plant_prose(window, where: str, stored: str):
    if where == "effect message":
        return _plant_prose_effect(window, stored)
    _plant_trigger_description(window, stored)
    return None


_PROSE_WHERE = ["effect message", "trigger description"]


_PROSE_FIELD = {"effect message": "message", "trigger description": "description"}


@pytest.mark.parametrize("where", _PROSE_WHERE)
@pytest.mark.font_sensitive
def test_a_prose_field_is_a_wrapping_multi_line_editor(where: str) -> None:
    """The assertion the XS tests deliberately do not make: a value with no
    newline at all still occupies several visual lines, with no horizontal
    scrollbar. GH #38 is exactly that case."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from descape.text_edits import ProseTextEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose(window, where, _LONG_LINE)
        widget = _row_widget(panel, _PROSE_FIELD[where])
        assert isinstance(widget, ProseTextEdit)
        assert widget.toPlainText() == _LONG_LINE
        QApplication.processEvents()
        assert widget.document().blockCount() == 1, "one paragraph, wrapped -- not split"
        assert widget.document().firstBlock().layout().lineCount() > 1, "did not wrap"
        # The policy, not isVisible(): every widget of a never-shown window
        # answers isVisible() False, so that assertion would hold for the
        # NoWrap widget too and prove nothing.
        assert widget.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert window.trigger_edits is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("where", _PROSE_WHERE)
def test_a_prose_field_shows_its_stored_lines(where: str) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose(window, where, "first line\r\nsecond line\r\nthird line")
        widget = _row_widget(panel, _PROSE_FIELD[where])
        assert widget.toPlainText() == "first line\nsecond line\nthird line"
        assert widget.document().blockCount() == 3
        assert widget.newline_token == "\r\n"
    finally:
        window.edit_history.mark_saved()
        window.close()


def _caption_geometry(panel, text: str):
    from PyQt5.QtWidgets import QLabel

    labels = [w for w in panel.property_host.findChildren(QLabel) if w.text() == text and w.isVisible()]
    assert len(labels) == 1, [w.text() for w in panel.property_host.findChildren(QLabel)]
    return labels[0].geometry()


@pytest.mark.parametrize(
    "where,field",
    [
        ("trigger", "description"),
        ("trigger", "short_description"),
        ("effect message", "message"),
        ("script call", "message"),
    ],
)
def test_a_multi_line_box_sits_full_width_under_its_caption(where: str, field: str) -> None:
    """GH #38: like Messages mode, the caption gets its own row and the box
    spans the form below it. One-line fields keep the side caption."""
    from PyQt5.QtWidgets import QApplication

    from descape.text_edits import _MultiLineEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        if where == "trigger":
            _plant_trigger_description(window, "some text")
        elif where == "effect message":
            _plant_prose_effect(window, "some text")
        else:
            _plant_script_call(window, "effect", "void f() {}")
        QApplication.processEvents()
        widget = _row_widget(panel, field)
        assert isinstance(widget, _MultiLineEdit)
        spec = next(s for s, *_ in panel._rows if s.name == field)
        form = panel.property_form.contentsRect()
        box = widget.geometry()
        assert (box.left(), box.right()) == (form.left(), form.right()), (box, form)
        caption = _caption_geometry(panel, spec.label)
        assert caption.left() == form.left()
        assert caption.bottom() < box.top(), "the caption sits above its box"

        if where == "trigger":
            name = _row_widget(panel, "name")
            name_caption = panel.property_form.labelForField(name)
            assert name_caption is not None and name_caption.geometry().right() < name.geometry().left()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #139: caption buddies, locked multi-line rows, the height grip --------


def _caption_for(panel, widget):
    """The full-width caption QLabel whose buddy is `widget`."""
    from PyQt5.QtWidgets import QLabel

    captions = [w for w in panel.property_host.findChildren(QLabel) if w.buddy() is widget]
    assert len(captions) == 1, [w.text() for w in panel.property_host.findChildren(QLabel)]
    return captions[0]


def _with_trigger_spec(monkeypatch, name: str, **changes) -> None:
    """Swap one trigger spec for a replaced copy, as the form will see it."""
    from dataclasses import replace

    from descape import trigger_fields

    original = trigger_fields.trigger_specs
    monkeypatch.setattr(
        trigger_fields,
        "trigger_specs",
        lambda version: tuple(replace(s, **changes) if s.name == name else s for s in original(version)),
    )


@pytest.mark.parametrize("field", ["description", "short_description"])
def test_a_multi_line_caption_is_its_box_s_buddy(field: str) -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        widget = _row_widget(panel, field)
        assert _caption_for(panel, widget).text() == field.replace("_", " ")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_locked_multi_line_field_takes_its_caption_above_too(monkeypatch) -> None:
    """A read-only spec renders as a QLabel, not a _MultiLineEdit; its caption
    still goes on its own row with the text full width below."""
    from PyQt5.QtWidgets import QApplication, QLabel

    _with_trigger_spec(monkeypatch, "description", read_only=True)
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        widget = _row_widget(panel, "description")
        assert isinstance(widget, QLabel)
        assert panel.property_form.labelForField(widget) is None, "not a side caption"
        caption = _caption_for(panel, widget).geometry()
        form = panel.property_form.contentsRect()
        assert caption.bottom() < widget.geometry().top()
        assert (widget.geometry().left(), widget.geometry().right()) == (form.left(), form.right())
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_full_width_caption_carries_its_spec_s_tooltip(monkeypatch) -> None:
    _with_trigger_spec(monkeypatch, "description", tooltip="What the Objectives panel shows.")
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        widget = _row_widget(panel, "description")
        assert _caption_for(panel, widget).toolTip() == "What the Objectives panel shows."
        assert widget.toolTip() == "What the Objectives panel shows."
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_dragging_a_box_taller_refits_the_form_and_records_nothing() -> None:
    """The host grows by the drag's delta, no row draws over the next, and a
    resize is not a document edit: no history record, nothing dirty."""
    from PyQt5.QtWidgets import QApplication
    from test_text_edits import _drag

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        box = _row_widget(panel, "description")
        # Baseline from a fit at the shown width, the same measurement the drag's refit makes.
        panel._fit_property_height()
        host_before = panel.property_host.minimumHeight()
        box_before = box.height()

        _drag(box.grip(), 4)
        QApplication.processEvents()
        assert box.visible_lines() == box.VISIBLE_LINES + 4
        delta = box.height() - box_before
        assert delta == 4 * box.fontMetrics().lineSpacing()
        assert panel.property_host.minimumHeight() - host_before == delta
        _assert_form_rows_do_not_overlap(panel.property_form)
        assert window.trigger_edits is None
        assert not window.edit_history.can_undo
        assert not window.windowTitle().startswith("*")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_repopulating_the_form_leaves_no_empty_rows_behind() -> None:
    """takeAt() empties a QFormLayout row but keeps the row, so every populate
    used to stack another set of empty rows under the live ones."""
    from PyQt5.QtWidgets import QApplication, QFormLayout

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        form = panel.property_form
        counts = []
        for _ in range(3):
            panel._populate_property_form()
            QApplication.processEvents()
            counts.append(form.rowCount())
        assert counts[0] > 0
        assert counts == [counts[0]] * 3, f"rowCount grew across populates: {counts}"
        roles = (QFormLayout.LabelRole, QFormLayout.FieldRole, QFormLayout.SpanningRole)
        empty = [r for r in range(form.rowCount()) if all(form.itemAt(r, role) is None for role in roles)]
        assert not empty, f"empty rows left behind: {empty}"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_entering_triggers_mode_fits_the_form_to_the_shown_width() -> None:
    """The populate's fit ran while every new row still waited on Qt's queued
    show, so the host kept a 102 px minimum until something else re-fit it."""
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        QApplication.processEvents()
        panel = window.trigger_panel
        width = panel.property_area.viewport().width()
        assert width > 0
        assert panel._rows, "entering the mode no longer builds a form, so this tests nothing"
        panel.property_form.activate()
        needed = panel.property_form.heightForWidth(width)
        assert panel.property_host.minimumHeight() >= needed, (
            f"host minimum {panel.property_host.minimumHeight()} < heightForWidth({width}) {needed}"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_stored_height_is_applied_kept_across_reselection_and_a_drag_reports_once() -> None:
    from PyQt5.QtWidgets import QApplication
    from test_text_edits import _drag

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        stored = {"trigger.description": 12}
        reported = []
        panel.text_box_lines = stored.get
        panel.on_text_box_lines = lambda key, lines: reported.append((key, lines))
        # Trigger 0's form was built on entering the mode, before the injection.
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        assert _row_widget(panel, "description").visible_lines() == 12
        assert _row_widget(panel, "short_description").visible_lines() == 6, "keyed per field"

        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        box = _row_widget(panel, "description")
        assert box.visible_lines() == 12

        _drag(box.grip(), 2)
        assert reported == [("trigger.description", 14)]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_window_wires_text_box_heights_to_settings() -> None:
    """End to end through viewer.py's injection: a saved height builds the
    box, a drag saves the new one, and a double-click reset removes the key
    rather than storing the default."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from test_text_edits import _drag

    from descape import settings

    settings.set_text_box_lines("trigger.description", 9)
    settings.set_text_box_lines("messages.hints", 11)
    window = _triggers_window()
    try:
        window.mode_combo.setCurrentText("Messages")
        assert window.messages_panel.widget_for("hints").visible_lines() == 11
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        QApplication.processEvents()
        box = _row_widget(panel, "description")
        assert box.visible_lines() == 9
        _drag(box.grip(), -3)
        assert settings.get_text_box_lines("trigger.description") == 6
        QTest.mouseDClick(box.grip(), Qt.LeftButton, Qt.NoModifier, QPoint(5, 2))
        # A real double click ends with a release; QTest's leaves the button held process-wide.
        QTest.mouseRelease(box.grip(), Qt.LeftButton, Qt.NoModifier, QPoint(5, 2))
        assert settings.get_text_box_lines("trigger.description") is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    "stored", ["a\rb", "a\r\nb", "a\nb", "one line"], ids=["CR", "CRLF", "LF", "none"]
)
@pytest.mark.parametrize("where", _PROSE_WHERE)
def test_an_untouched_prose_focus_out_records_nothing(where: str, stored: str) -> None:
    """CR and CRLF display as "a\\nb", which differs from what is stored, so
    _changed()'s equality check alone would record a phantom undo step."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose(window, where, stored)
        _row_widget(panel, _PROSE_FIELD[where]).editingFinished.emit()
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    "stored,token",
    [("a\rb", "\r"), ("a\r\nb", "\r\n"), ("a\nb", "\n"), ("one line", "\n")],
    ids=["CR", "CRLF", "LF", "none"],
)
def test_an_edited_prose_field_keeps_its_own_newline_token(
    tmp_path: Path, stored: str, token: str
) -> None:
    """The behaviour the XS path deliberately does not have: XS forces every
    separator to CR, which is right for a script body and would silently
    rewrite (and widen the save diff of) a description stored with LF."""
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        effect_index = _plant_prose_effect(window, stored)
        widget = _row_widget(panel, "message")
        widget.setPlainText("edited\nacross\nthree lines")
        widget.editingFinished.emit()
        assert window.edit_history.is_dirty

        out = tmp_path / "prose.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        reloaded = parse_triggers(load_map_and_units(out))
        assert reloaded.triggers[0].effects[effect_index].message == token.join(
            ("edited", "across", "three lines")
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_second_prose_focus_out_is_not_a_second_edit() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose_effect(window, "a\rb")
        widget = _row_widget(panel, "message")
        widget.setPlainText("rewritten")
        widget.editingFinished.emit()
        steps = window.edit_history.cursor
        widget.editingFinished.emit()
        assert window.edit_history.cursor == steps
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_name_field_stays_one_line() -> None:
    """The other half of the set-membership decision: every *_name message is
    an identifier under 30 characters, not prose."""
    from PyQt5.QtWidgets import QLineEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose_effect(window, "Gatehouse", name="change_object_name")
        assert isinstance(_row_widget(panel, "message"), QLineEdit)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_trigger_s_own_name_stays_one_line() -> None:
    from PyQt5.QtWidgets import QLineEdit

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_trigger_description(window, "a description")
        assert isinstance(_row_widget(panel, "name"), QLineEdit)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_locked_prose_field_renders_its_lines() -> None:
    from PyQt5.QtWidgets import QLabel

    from descape.trigger_fields import PROSE, STR, FieldSpec

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _plant_prose_effect(window, "a\r\nb")
        spec = FieldSpec("message", STR, sentinel=None, read_only=True, multiline=PROSE)
        label = panel._build_widget(spec, [("effect", 0)], panel._read(panel._current_entry()[2], spec.attribute))
        assert isinstance(label, QLabel)
        assert label.text() == "a\nb"
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- GH #139 on the trigger form: the grip both ways, reset, a restart ----------


def _form_top(panel, widget) -> int:
    from PyQt5.QtCore import QPoint

    return widget.mapTo(panel.property_host, QPoint(0, 0)).y()


def _settle() -> None:
    from PyQt5.QtWidgets import QApplication

    # Box, form and host each re-lay out in a later pass.
    for _ in range(5):
        QApplication.processEvents()


def test_a_description_drag_moves_the_rows_below_both_ways_and_a_double_click_resets() -> None:
    """TASK-165 steps 1-2 on the trigger form, Description's grip. Pass conditions:
    - the grip carries the vertical-resize cursor;
    - dragged 4 lines down it steps 7, 8, 9, 10 (one step per whole line) and
      the box grows by exactly 4 line heights;
    - the next form row moves down by the box's growth, and no row overlaps another;
    - dragged 6 lines back up it steps 9 down to 4, the next row follows it up
      by the same amount, and still no row overlaps another;
    - a double-click on the grip puts the box back to 6 lines at its starting
      height, with the next row back where it started."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from test_text_edits import _drag

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        _settle()
        names = [spec.name for spec, *_ in panel._rows]
        box = _row_widget(panel, "description")
        below = panel._rows[names.index("description") + 1][3]
        assert box.grip().cursor().shape() == Qt.SizeVerCursor
        steps: list[int] = []
        box.linesChanged.connect(steps.append)
        line = box.fontMetrics().lineSpacing()
        start_height, start_below = box.height(), _form_top(panel, below)
        assert box.visible_lines() == 6

        _drag(box.grip(), 4)
        _settle()
        assert steps == [7, 8, 9, 10]
        assert box.height() - start_height == 4 * line
        assert _form_top(panel, below) - start_below == 4 * line, "the row below moves down with the box"
        _assert_form_rows_do_not_overlap(panel.property_form)

        del steps[:]
        _drag(box.grip(), -6)
        _settle()
        assert steps == [9, 8, 7, 6, 5, 4]
        assert box.height() - start_height == -2 * line
        assert _form_top(panel, below) - start_below == -2 * line, "the row below moves up with the box"
        _assert_form_rows_do_not_overlap(panel.property_form)

        QTest.mouseDClick(box.grip(), Qt.LeftButton, Qt.NoModifier, QPoint(5, 2))
        # A real double click ends with a release; QTest's leaves the button held process-wide.
        QTest.mouseRelease(box.grip(), Qt.LeftButton, Qt.NoModifier, QPoint(5, 2))
        _settle()
        assert box.visible_lines() == 6
        assert box.height() == start_height
        assert _form_top(panel, below) == start_below
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_dragged_description_height_survives_a_restart_on_every_trigger(monkeypatch) -> None:
    """TASK-165 step 3. Pass conditions:
    - dragging Description from 6 to 12 lines writes 12 under
      trigger.description in config.yaml;
    - after that window closes and settings' memoised values are dropped (a
      restart's cold read of config.yaml), a new window on the file shows
      Description at 12 lines;
    - another trigger's Description is 12 lines too."""
    import yaml
    from test_text_edits import _drag

    from descape import settings
    from testkit import settings_isolation

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        _settle()
        box = _row_widget(panel, "description")
        assert box.visible_lines() == 6
        _drag(box.grip(), 6)
        assert box.visible_lines() == 12
    finally:
        window.edit_history.mark_saved()
        window.close()
    assert yaml.safe_load(settings.CONFIG_PATH.read_text())["text_box_lines"] == {"trigger.description": 12}

    for name in settings_isolation.MEMOIZED_GLOBALS:
        monkeypatch.setattr(settings, name, None)
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        _settle()
        assert _row_widget(panel, "description").visible_lines() == 12
        panel.tree.setCurrentItem(panel.tree.topLevelItem(1))
        _settle()
        assert panel.current_trigger_index() == 1
        assert _row_widget(panel, "description").visible_lines() == 12, "another trigger's Description"
    finally:
        window.edit_history.mark_saved()
        window.close()
