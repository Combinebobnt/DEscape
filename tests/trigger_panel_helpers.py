"""Shared fixtures and helpers for the tests/test_trigger_panel_*.py modules.

Same offscreen-ViewerWindow technique as tests/test_keybinds.py
(QT_QPA_PLATFORM=offscreen, one shared QApplication via conftest.ensure_qapp()).
A module that uses these imports `_no_install_env` by name, so the autouse
fixture applies there too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from testkit.qt_window import panel_width, set_panel_width

import conftest

TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


V158_DONOR = Path(__file__).resolve().parent / "fixtures" / "blank_120x120_v158.aoe2scenario"


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """No trigger panel test wants the real install, and AOE2DE_INSTALL_PATH outranks
    the config conftest hides, so a shell exporting it would leak one in."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _window():
    """A shown, fixed-size offscreen ViewerWindow.

    show() + processEvents() is load-bearing for anything reading the
    splitter: QSplitter.setSizes() is renormalized against the widget's real
    geometry, so on an unshown window every size assertion tests Qt's layout
    fallback rather than the code under test.

    Also releases any modifier an earlier test's QTest.keyClick left held:
    the multi-select tree reads the live modifiers on a plain setCurrentItem(),
    exactly as a click does, so a stale Ctrl turns it into a toggle.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    window = conftest.shown_window(1500, 900)
    for key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
        QTest.keyRelease(window, key)
    return window


def _triggers_window():
    """A window in Triggers mode on the trigger-bearing fixture."""
    window = _window()
    window.load_scenario(TRIGGER_FIXTURE)
    window.mode_combo.setCurrentText("Triggers")
    return window


def _walk_every_row(panel) -> None:
    """Select every trigger, then every row of its detail tree. This is what a
    user does by clicking around, and it must record nothing.

    Walks both a top-level item and its children (4c: a grouped tree's
    sections nest real trigger rows under a header, or under the one
    synthetic "(before the first section)" row, which is skipped -- it isn't
    a trigger and Qt.ItemIsSelectable is off for it). On the flat, unshipped
    default fixture every top-level item has zero children, so this walks
    exactly as it did before 4c existed.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    def select(item) -> None:
        panel.tree.setCurrentItem(item)
        QApplication.processEvents()
        for t in range(panel.entry_tree.topLevelItemCount()):
            top = panel.entry_tree.topLevelItem(t)
            panel.entry_tree.setCurrentItem(top)
            QApplication.processEvents()
            for c in range(top.childCount()):
                panel.entry_tree.setCurrentItem(top.child(c))
                QApplication.processEvents()

    for i in range(panel.tree.topLevelItemCount()):
        top = panel.tree.topLevelItem(i)
        if top.data(0, Qt.UserRole) is not None:
            select(top)
        for c in range(top.childCount()):
            select(top.child(c))


def _trigger_row_count(tree) -> int:
    """Every real trigger row in `tree`, top-level or nested -- the count
    the status line reports, which diverges from topLevelItemCount() once
    4c's grouping is in play."""
    from PyQt5.QtCore import Qt

    total = 0
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        if top.data(0, Qt.UserRole) is not None:
            total += 1
        total += top.childCount()
    return total


def _select_first_condition(panel) -> int:
    """Select the first condition the fixture carries, and return its trigger
    index. Asserts rather than skips: a fixture with no conditions would make
    every entry-level test vacuous while still reporting green."""
    for i in range(panel.tree.topLevelItemCount()):
        panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
        conditions = panel.entry_tree.topLevelItem(1)
        if conditions is not None and conditions.childCount():
            panel.entry_tree.setCurrentItem(conditions.child(0))
            return i
    raise AssertionError("the fixture carries no conditions, so entry tests cannot run")


def _row_widget(panel, field: str):
    """The live widget for one field of whatever the detail tree has selected."""
    for spec, _kind, _index, widget in panel._rows:
        if spec.name == field:
            return widget
    raise AssertionError(f"no {field} row in {[s.name for s, *_ in panel._rows]}")


def _select_armour_split_effect(panel, child: int) -> None:
    """Select effect `child` of "Fixture: armour split" (list index 1)."""
    panel.select_trigger(1)
    panel.entry_tree.setCurrentItem(panel.entry_tree.topLevelItem(2).child(child))


def _group(panel, kind: str):
    """The Conditions or Effects heading row, found by its marker.

    By marker rather than by topLevelItem(1)/(2): those positions are right
    today and wrong the moment the detail tree gains a row.
    """
    from PyQt5.QtCore import Qt

    for i in range(panel.entry_tree.topLevelItemCount()):
        item = panel.entry_tree.topLevelItem(i)
        if item.data(0, Qt.UserRole) == ("group", kind):
            return item
    raise AssertionError(f"no {kind} group row in the detail tree")


def _pick(panel, kind: str, name: str) -> None:
    """Select one row of the open picker by its displayed name."""
    from PyQt5.QtCore import Qt

    for i in range(panel.picker_tree.topLevelItemCount()):
        group = panel.picker_tree.topLevelItem(i)
        for c in range(group.childCount()):
            child = group.child(c)
            if child.data(0, Qt.UserRole)[0] == kind and child.text(0) == name:
                panel.picker_tree.setCurrentItem(child)
                return
    raise AssertionError(f"no {kind} named {name!r} in the picker")


def _tree_order(panel) -> list[int]:
    from PyQt5.QtCore import Qt

    return [panel.tree.topLevelItem(i).data(0, Qt.UserRole) for i in range(panel.tree.topLevelItemCount())]


def _positions(panel) -> list[str]:
    return [
        panel.tree.topLevelItem(i).text(panel._COL_POS) for i in range(panel.tree.topLevelItemCount())
    ]


def _header(panel) -> str:
    return panel.tree.headerItem().text(panel._COL_POS)


def _assert_form_rows_do_not_overlap(form) -> tuple[int, int]:
    """Walk every row, label-and-field, wrapped and full-width (spanning)
    alike: no widget squeezed below its minimum, no row drawing over the one
    above, and a wrapped row's editor below its own label. Returns (rows
    walked, rows wrapped). Rows with no widget are skipped; a spanning widget
    answers FieldRole too."""
    from PyQt5.QtWidgets import QFormLayout, QLabel

    walked = wrapped = 0
    previous_bottom = None
    for row in range(form.rowCount()):
        widgets = {}
        for role in (QFormLayout.LabelRole, QFormLayout.FieldRole, QFormLayout.SpanningRole):
            item = form.itemAt(row, role)
            if item is not None and item.widget() is not None and item.widget() not in widgets.values():
                widgets[role] = item.widget()
        if not widgets:
            continue
        walked += 1
        name = next((w.text() for w in widgets.values() if isinstance(w, QLabel)), f"row {row}")
        for widget in widgets.values():
            assert widget.height() >= widget.minimumSizeHint().height(), f"{name!r} is squeezed below its own minimum height"
        geometries = [widget.geometry() for widget in widgets.values()]
        if previous_bottom is not None:
            assert min(g.top() for g in geometries) >= previous_bottom, f"{name!r} draws over the row above it"
        label, field = widgets.get(QFormLayout.LabelRole), widgets.get(QFormLayout.FieldRole)
        if label is not None and field is not None and field.geometry().left() < label.geometry().right():
            wrapped += 1
            assert field.geometry().top() >= label.geometry().bottom(), f"{name!r}'s wrapped editor overlaps its label"
        previous_bottom = max(g.bottom() for g in geometries)
    return walked, wrapped


def _effect_id(window, name: str) -> int:
    from descape import library_compat

    vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
    return next(e.id for e in vocabulary.effects.values() if e.name == name)


def _plant_prose_effect(window, stored: str, name: str = "display_instructions") -> int:
    """Add a prose-message effect to trigger 0 of the parsed manager,
    pre-model, holding `stored`, and select it. Same deliberate no-model route
    as _plant_script_call()."""
    panel = window.trigger_panel
    trigger = panel._manager().triggers[0]
    trigger._add_effect(_effect_id(window, name))
    trigger.effects[-1].message = stored
    panel.select_trigger(0)
    panel.refresh_entries(select=("effect", len(trigger.effects) - 1))
    return len(trigger.effects) - 1


def _ctrl_select(panel, *indices) -> None:
    """Qt's Ctrl+click, minus the mouse: current first, then add to the set."""
    panel.select_triggers(list(indices))


def trigger_clipboard_refs(trigger):
    from descape.trigger_clipboard import get_trigger_referencing_ce

    return get_trigger_referencing_ce(trigger)


def _key(widget, key, modifiers) -> None:
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    QTest.keyClick(widget, key, modifiers)
    QApplication.processEvents()


def _rename(panel, index: int, name: str) -> None:
    panel.select_trigger(index)
    widget = _row_widget(panel, "name")
    widget.setText(name)
    widget.editingFinished.emit()


def _action(menu, text: str):
    return next(a for a in menu.actions() if a.text() == text)


_TAGGED_NAMES = ["[D1] setup", "[d1]armour split", "[P1][D1] references", "[D1]: variable"]


def _tagged_window(tmp_path: Path, names: list[str] = _TAGGED_NAMES):
    """A window on the trigger fixture with tagged names saved into it: D1 x2,
    a case variant d1, and P1 with D1 only as its chained second tag."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = TriggerEditModel(loaded)
    for index, name in enumerate(names):
        model.manager().triggers[index].name = name
        model.mark_dirty(index)
    base = tmp_path / "tagged.aoe2scenario"
    write_scenario(loaded, base, triggers=model)
    window = _window()
    window.load_scenario(base)
    window.mode_combo.setCurrentText("Triggers")
    return window


def _names(window) -> list[str]:
    return [t.name for t in window.trigger_panel._manager().triggers]


def _answer_dialogs(monkeypatch, text=None, answer=None) -> dict:
    """Stub the panel's two dialogs; record what each was asked."""
    from PyQt5.QtWidgets import QInputDialog, QMessageBox

    asked = {"getText": [], "question": []}

    def get_text(parent, title, label, mode, default):
        asked["getText"].append((title, label, default))
        return (text, True) if text is not None else ("", False)

    def question(parent, title, body, *rest):
        asked["question"].append(body)
        return answer

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(get_text))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def _sectioned_window():
    """The trigger fixture grown to 8 triggers and split into four sections:
    (before) 0, 1 | --RISK!!!-- 2: 3 | --RISK!!!-- 4: 5 | ------ 6: 7. The
    repeated title is the census's duplicate-divider shape."""
    window = _triggers_window()
    for _ in range(4):
        window.trigger_structural_edit("new", [])
    panel = window.trigger_panel
    _rename(panel, 5, "needle target")
    _rename(panel, 2, "--RISK!!!--")
    _rename(panel, 4, "--RISK!!!--")
    _rename(panel, 6, "------")
    assert [s.header_index for s in panel._sections] == [None, 2, 4, 6], "fixture assumption"
    panel.select_trigger(0)
    return window


def _roomy_tree(panel) -> None:
    """Give the trigger tree pane 500 px of the vertical split. At the default
    1500x900 split (equal halves) the tree shows ~10 rows, so a test that
    clicks a real row position needs the room to see it."""
    from PyQt5.QtWidgets import QApplication

    sizes = panel.splitter.sizes()
    panel.splitter.setSizes([500, max(sum(sizes) - 500, 1)])
    QApplication.processEvents()


def _order(window) -> list[int]:
    return list(window.trigger_panel._manager().trigger_display_order)


def _record_count(window) -> int:
    return len(window.edit_history.records)


def _divided_window(name: str = "--A--", at: int = 2):
    """The trigger fixture with trigger `at` renamed to divider `name`: by
    default (before) 0, 1 | --A-- 2: 3."""
    window = _triggers_window()
    _rename(window.trigger_panel, at, name)
    window.trigger_panel.select_trigger(0)
    return window


def _row_widgets(row) -> list:
    return [row.itemAt(i).widget() for i in range(row.count()) if row.itemAt(i).widget() is not None]


def _saved_body(path: Path) -> bytes:
    from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

    from descape.scenario_io import load_map_and_units

    return _decompress_bytes(path.read_bytes()[len(load_map_and_units(path).header_bytes) :])


def _open(path: Path):
    window = _window()
    window.load_scenario(path)
    window.mode_combo.setCurrentText("Triggers")
    return window


# The fixture's "armour split" trigger: one condition, two Modify Attribute effects.
_ARMOUR_TRIGGER = 1


def _select_entries(panel, *refs) -> None:
    """Select these (kind, index) entry rows, Qt's Ctrl+click minus the mouse."""
    from PyQt5.QtWidgets import QApplication

    panel.select_entries(list(refs))
    QApplication.processEvents()


_CREATE_OBJECT = 11


_DISPLAY_INSTRUCTIONS = 20


_ARCHER, _SCOUT = 4, 448


def _spec_named(panel, kind: str, entry, name: str):
    return next(s for s in panel._specs_for(kind, entry) if s.name == name)


def _appended_effects(window, type_id: int, per_entry) -> list[int]:
    """Append one `type_id` effect per dict in `per_entry` to trigger 0, set
    its fields through the real funnel, and return their indices. Built at
    runtime: extending the shared fixture breaks 37 tests pinning its layout."""
    panel = window.trigger_panel
    panel.select_trigger(0)
    start = len(panel._manager().triggers[0].effects)
    for offset, values in enumerate(per_entry):
        window.entry_structural_edit("new", 0, "effect", -1, type_id)
        for name, value in values.items():
            entry = panel._manager().triggers[0].effects[start + offset]
            window.set_entry_field(0, "effect", start + offset, _spec_named(panel, "effect", entry, name), value)
    window.edit_history.mark_saved()
    return list(range(start, start + len(per_entry)))


def _create_objects_window():
    """The issue's own example: three Create Object effects agreeing on unit
    and player, differing only in where they put it."""
    window = _triggers_window()
    rows = [
        {"object_list_unit_id": _ARCHER, "source_player": 1, "location_x": 10 + n, "location_y": 20 + n}
        for n in range(3)
    ]
    return window, _appended_effects(window, _CREATE_OBJECT, rows)


def _effects0(window):
    return window.trigger_panel._manager().triggers[0].effects


def _focus_out_every_row(panel) -> None:
    """Enter and leave every editor, the way tabbing through the form does:
    editingFinished fires on a bare focus-out."""
    from PyQt5.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit, QPlainTextEdit, QWidget

    from descape.value_picker import ValueLineEdit

    for _spec, _kind, _index, widget in panel._rows:
        editors = [widget]
        if not isinstance(widget, (ValueLineEdit, QLineEdit, QPlainTextEdit, QAbstractSpinBox)):
            editors = widget.findChildren(QAbstractSpinBox) or [widget]
        for editor in editors:
            target = editor.line_edit if isinstance(editor, ValueLineEdit) else editor
            if isinstance(target, QWidget) and target.isEnabled():
                target.setFocus()
                QApplication.processEvents()
            if hasattr(target, "editingFinished"):
                target.editingFinished.emit()
            QApplication.processEvents()
    panel.entry_tree.setFocus()
    QApplication.processEvents()


def _status(window) -> str:
    return window.status_log.toPlainText()


def _gh2_window():
    """_triggers_window() whose failure box fails the test: a real one hangs a headless worker."""
    window = _triggers_window()
    window._warn = lambda title, text: pytest.fail(f"{title}: {text}")
    return window


# The fixture's "references" (2): conditions [10], effects [8, 9];
# "variable" (3): conditions [10, 5, 6, 6], effects [56, 19, 11, 14, 15, 12, 19].


def _types_of(window, trigger_index: int, kind: str = "effect") -> list[int]:
    trigger = window.trigger_panel._manager().triggers[trigger_index]
    return [getattr(e, f"{kind}_type") for e in getattr(trigger, f"{kind}s")]


def _active_triggers_window():
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    window.activateWindow()
    QApplication.setActiveWindow(window)
    QApplication.processEvents()
    assert window.isActiveWindow()
    return window


def _focus_on(widget) -> None:
    from PyQt5.QtWidgets import QApplication

    widget.setFocus()
    QApplication.processEvents()
    assert widget.hasFocus()


# The fixture's one Display Instructions effect: trigger 0 "Fixture: setup", effect 0.
_SETUP_TRIGGER = 0


def _instruction_window(**fields):
    """The trigger window with the fixture's Display Instructions effect
    selected, its entry fields first set to `fields` (pre-model, in-test: the
    fixture is not regenerated)."""
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    panel = window.trigger_panel
    panel.select_trigger(_SETUP_TRIGGER)
    entry = panel._manager().triggers[_SETUP_TRIGGER].effects[0]
    assert entry.effect_type == _DISPLAY_INSTRUCTIONS
    for name, value in fields.items():
        setattr(entry, name, value)
    panel.refresh_entries(select=("effect", 0))
    QApplication.processEvents()
    return window


_REPLACE_OBJECT = 43


_CASTLE = 82


def _facet_window(type_id: int, *per_entry):
    """The trigger window with `type_id` effects appended to trigger 0, the
    first one selected."""
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    indices = _appended_effects(window, type_id, list(per_entry))
    # A rebuild: the last append left this same row's form up, built before its fields were set.
    window.trigger_panel.refresh_entries(select=("effect", indices[0]))
    QApplication.processEvents()
    return window, indices


# TASK-560: the widest forms of the 144 condition and effect types, measured 2026-11-02 at the
# panel's minimum from 72 to 168 DPI (the plan's p560 probe), plus an area-bearing one.
_WIDE_FORM_TYPES = (
    ("condition", "own_objects"),
    ("condition", "own_fewer_objects"),
    ("condition", "objects_in_area"),
    ("effect", "task_object"),
    ("effect", "build_object"),
    ("effect", "activate_trigger"),
    ("effect", "deactivate_trigger"),
    ("effect", "kill_object"),
)


def check_triggers_fit_their_drag_floor(window) -> int:
    """TASK-560: Triggers mode with the pane dragged as narrow as the splitter allows. Pass
    conditions:
    - the pane stops above MIN_SPLIT_PANE, at the trigger panel's own minimumSizeHint width;
    - each _WIDE_FORM_TYPES form has nothing to scroll sideways and its scroll area ends inside
      the panel;
    - every visible button in the panel lies wholly inside it at no less than its minimum hint.
    Returns the narrowest pane width."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QAbstractButton, QApplication

    from descape import library_compat, settings

    window.mode_combo.setCurrentText("Triggers")
    QApplication.processEvents()
    set_panel_width(window, settings.MIN_SPLIT_PANE)
    QApplication.processEvents()
    pane = panel_width(window)
    panel = window.trigger_panel
    assert pane > settings.MIN_SPLIT_PANE, "Triggers has no drag floor"
    assert pane == panel.minimumSizeHint().width(), (pane, panel.minimumSizeHint().width())

    vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
    panel.select_trigger(0)
    for kind, name in _WIDE_FORM_TYPES:
        entries = vocabulary.conditions if kind == "condition" else vocabulary.effects
        type_id = next(e.id for e in entries.values() if e.name == name)
        trigger = panel._manager().triggers[0]
        index = len(trigger.conditions if kind == "condition" else trigger.effects)
        window.entry_structural_edit("new", 0, kind, -1, type_id)
        panel.refresh_entries(select=(kind, index))
        QApplication.processEvents()
        area = panel.property_area
        slack = area.viewport().width() - panel.property_host.minimumSizeHint().width()
        assert area.horizontalScrollBar().maximum() == 0, f"{name} scrolls sideways at the floor (slack {slack} px)"
        right = area.mapTo(panel, QPoint(area.width(), 0)).x()
        assert right <= panel.width(), f"{name}'s form ends at {right} px, past the {panel.width()} px panel"
    for button in panel.findChildren(QAbstractButton):
        if not button.isVisible():
            continue
        left = button.mapTo(panel, QPoint(0, 0)).x()
        assert left >= 0 and left + button.width() <= panel.width(), (button.text(), left, button.width(), panel.width())
        # A fixed-width button (a picker's "…") is never wider than its maximum, as Qt's layouts allow.
        assert button.width() >= min(button.minimumSizeHint().width(), button.maximumWidth()), button.text()
    window.edit_history.mark_saved()
    return pane


def _display_names(window) -> list[str]:
    manager = window.trigger_panel._manager()
    return [manager.triggers[i].name for i in manager.trigger_display_order]


def _effect_type_id(panel, name: str) -> int:
    return next(i for i, d in panel._vocabulary.effects.items() if d.name == name)


def _ordered_window(tmp_path: Path):
    """Triggers mode on the fixture resaved with trigger 3 under non-identity
    condition and effect orders (test_trigger_write_path.ENTRY_ORDERS), that
    trigger current. Returns (window, the saved file)."""
    from test_trigger_write_path import ENTRY_ORDER_TRIGGER, write_entry_order_fixture

    base = write_entry_order_fixture(tmp_path)
    window = _window()
    window.load_scenario(base)
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_trigger(ENTRY_ORDER_TRIGGER)
    return window, base


def _entry_rows(panel, kind: str) -> list[int]:
    """The list index each row of one group carries, top to bottom."""
    from PyQt5.QtCore import Qt

    group = _group(panel, kind)
    return [group.child(row).data(0, Qt.UserRole)[1] for row in range(group.childCount())]


def _entry_move_states(panel) -> tuple[bool, bool]:
    return (panel.entry_move_up_button.isEnabled(), panel.entry_move_down_button.isEnabled())


def _entry_menu_move_states(panel, ref) -> tuple[bool, bool]:
    menu = panel.entry_context_menu(panel._entry_item_for(*ref))
    return (_action(menu, "Move Up").isEnabled(), _action(menu, "Move Down").isEnabled())


# More triggers than the shipped fixtures' 4 and 8, for five-place moves and tree overflow.
_LONG_COUNT = 30


def _save_long_fixture(tmp_path: Path, dividers: tuple[int, ...] = (), names: dict[int, str] | None = None) -> Path:
    """The fixture grown to 30 triggers, as a real file. The new ones are
    "t4" to "t29" in display order; `dividers` names those list indices as
    section headers ("--- S10 ---") and `names` overrides any of them."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    wanted = {index: f"--- S{index} ---" for index in dividers} | (names or {})
    assert all(index >= 4 for index in wanted), "only the added triggers are named here"
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    model = TriggerEditModel(loaded)

    def grow(manager) -> None:
        for index in range(len(manager.triggers), _LONG_COUNT):
            manager.add_trigger(wanted.get(index, f"t{index}"))
        manager.trigger_display_order = list(range(_LONG_COUNT))

    model.structural_edit(grow)
    base = tmp_path / "long.aoe2scenario"
    write_scenario(loaded, base, triggers=model)
    return base


def _long_window(tmp_path: Path, dividers: tuple[int, ...] = (), names: dict[int, str] | None = None):
    """Triggers mode on _save_long_fixture()'s file, so the history starts empty."""
    window = _open(_save_long_fixture(tmp_path, dividers, names))
    assert _order(window) == list(range(_LONG_COUNT)), "fixture assumption"
    return window


def _modifier_click(button, modifiers) -> None:
    """A real mouse click whose own events carry `modifiers`, which is where
    the ▲▼ buttons read them. The caller releases the keys in its finally."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    QTest.mouseClick(button, Qt.LeftButton, modifiers)
    QApplication.processEvents()


def _release_modifiers(window) -> None:
    """A QTest click or key with a modifier can leave it held for the next test."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    for key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
        QTest.keyRelease(window, key)


def _row_in_view(tree, item) -> bool:
    rect = tree.visualItemRect(item)
    return rect.height() > 0 and rect.top() >= 0 and rect.bottom() <= tree.viewport().height()


def _last_label(window) -> str:
    return window.edit_history.records[-1].label


def _activate(window) -> None:
    """Shortcut dispatch needs the active window, which offscreen show() alone does not make."""
    from PyQt5.QtWidgets import QApplication

    window.activateWindow()
    QApplication.setActiveWindow(window)
    QApplication.processEvents()
    assert window.isActiveWindow()
