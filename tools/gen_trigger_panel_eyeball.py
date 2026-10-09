#!/usr/bin/env python3
"""Screenshots TriggerPanel (Phase 4c, trigger organization) for a manual
eyeball pass -- always writes PNGs, never pass/fail. This covers the
trigger-organization plan's own verification step: "Screenshot the panel
at 340 px on F7_3_York (18 sections) and old-allies-final-v2 (30 sections):
expanded, collapsed, tag-filtered, and under File order (flat) ... check
indent doesn't clip names at MIN_USEFUL_WIDTH."

Requires PyQt5 -- unlike tools/gen_seam_eyeball.py there is no off-engine
fallback, since the thing under review (grouped-tree rendering, indent,
clipping) only exists in the real widget.

Writes build/trigger_panel_eyeball/, gitignored. No test reads it. No
golden images -- there is no baseline, only visual inspection.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from testkit import settings_isolation

OUT_DIR = ROOT / "build" / "trigger_panel_eyeball"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

# (output-name prefix, examples/ filename, expected section count per the
# plan's census -- logged, not asserted, since this script has no pass/fail).
TARGETS = (
    ("f7_3_york", "F7_3_York (865).aoe2scenario", 18),
    ("old_allies_final_v2", "old-allies-final-v2.aoe2scenario", 30),
)

_QAPP = None


def _ensure_qapp() -> None:
    global _QAPP
    if _QAPP is not None:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(sys.argv[:1])


def _open_window(path: Path):
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.show()
    window.load_scenario(path)
    if window.scenario is None:
        window.close()
        raise SystemExit(f"{path.name} failed to load")
    window.mode_combo.setCurrentText("Triggers")
    QApplication.processEvents()
    return window


def _grab(window, out_path: Path) -> None:
    window.trigger_panel.grab().save(str(out_path))


def _capture_one(prefix: str, filename: str, expected_sections: int, out_dir: Path) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    path = ROOT / "examples" / filename
    if not path.is_file():
        raise SystemExit(f"missing corpus file: {path}")

    written: list[Path] = []
    window = _open_window(path)
    panel = window.trigger_panel
    try:
        print(f"{filename}: grouped={panel._grouped} top_level_rows={panel.tree.topLevelItemCount()} "
              f"(plan expects {expected_sections} sections)")

        # 1. expanded -- the populate-time default (every section starts
        # expanded, see trigger_panel.py's setExpanded(True) call sites).
        expanded_path = out_dir / f"{prefix}_expanded.png"
        _grab(window, expanded_path)
        written.append(expanded_path)

        # 2. collapsed.
        panel.collapse_all_sections()
        QApplication.processEvents()
        collapsed_path = out_dir / f"{prefix}_collapsed.png"
        _grab(window, collapsed_path)
        written.append(collapsed_path)
        panel.expand_all_sections()
        QApplication.processEvents()

        # 3. tag-filtered -- pick the first real tag, if the file has one.
        if panel.tag_combo.count() > 1:
            panel.tag_combo.setCurrentIndex(1)
            QApplication.processEvents()
            tag_path = out_dir / f"{prefix}_tag_filtered.png"
            _grab(window, tag_path)
            written.append(tag_path)
            panel.tag_combo.setCurrentIndex(0)
            QApplication.processEvents()
        else:
            print(f"  (skipping tag_filtered -- {filename} has no [tag]-prefixed trigger names)")

        # 4. File order (trigger ID) -- grouping only exists under Display
        # order, so this must render flat.
        panel.sort_combo.setCurrentText("File order (trigger ID)")
        QApplication.processEvents()
        file_order_path = out_dir / f"{prefix}_file_order.png"
        _grab(window, file_order_path)
        written.append(file_order_path)
        if panel._grouped:
            print(f"  WARNING: {filename} still reports _grouped=True under File order")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _capture_navigation_states(prefix: str, filename: str, out_dir: Path) -> list[Path]:
    """Navigation affordances at MIN_USEFUL_WIDTH: the open Sections menu
    (its own top-level window, so grabbed separately from the panel) and a
    filter active over a fully collapsed tree. Widths and the menu's height
    against the panel's are printed, not eyeballed."""
    from PyQt5.QtWidgets import QApplication

    path = ROOT / "examples" / filename
    if not path.is_file():
        print(f"(skipping navigation states -- missing {filename})")
        return []
    written: list[Path] = []
    window = _open_window(path)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        sizes = panel.splitter.sizes()
        panel.splitter.setSizes([panel.MIN_USEFUL_WIDTH, sizes[1]])
        QApplication.processEvents()
        print(f"{filename}: filter and tag rows at {panel.tree.width()} px (MIN_USEFUL_WIDTH is {panel.MIN_USEFUL_WIDTH}):")
        for label, widget in (
            ("filter", panel.filter_edit),
            ("sort", panel.sort_combo),
            ("tag", panel.tag_combo),
            ("rename", panel.tag_rename_button),
            ("sections", panel.sections_button),
        ):
            print(f"  {label:8s} x={widget.x():4d} width={widget.width():4d} hint={widget.sizeHint().width():4d}")

        menu = panel.sections_menu
        menu.popup(panel.sections_button.mapToGlobal(panel.sections_button.rect().bottomLeft()))
        QApplication.processEvents()
        menu_path = out_dir / f"{prefix}_sections_menu.png"
        menu.grab().save(str(menu_path))
        written.append(menu_path)
        print(f"  menu: {len(panel._section_actions)} sections, {menu.height()} px tall "
              f"(panel {panel.height()} px), {menu.width()} px wide")
        menu.hide()
        QApplication.processEvents()

        # A match inside a collapsed section: the middle section's first member.
        panel.collapse_all_sections()
        middle = next(
            (s for s in panel._sections[len(panel._sections) // 2:] if s.member_indices), None
        )
        if middle is None:
            print("  (skipping filtered_collapsed -- no section below the middle has members)")
        else:
            needle = panel._item_for_index[middle.member_indices[0]].text(panel._COL_NAME)[:14]
            panel.filter_edit.setText(needle)
            QApplication.processEvents()
            filtered_path = out_dir / f"{prefix}_filtered_collapsed.png"
            _grab(window, filtered_path)
            written.append(filtered_path)
            shown = sum(1 for i in panel._item_for_index.values() if not i.isHidden())
            print(f"  filter {needle!r} over a collapsed tree: {shown} rows shown")
            panel.filter_edit.clear()
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


TRIGGER_FIXTURE = ROOT / "tests" / "fixtures" / "triggers_120x120.aoe2scenario"
# trigger_id 60's own first condition (destroy_object, unit_object=14201) --
# the small fixture carries no Unit/Unit[] field at all.
UNIT_PRESENTATION_FILE = ROOT / "examples" / "F7_3_York (865).aoe2scenario"
UNIT_PRESENTATION_TRIGGER_ID = 60


def _select_effect(panel, trigger_list_index: int, child_index: int, scroll_to: str = "") -> None:
    from PyQt5.QtWidgets import QApplication

    panel.select_trigger(trigger_list_index)
    QApplication.processEvents()
    effects = panel.entry_tree.topLevelItem(2)
    panel.entry_tree.setCurrentItem(effects.child(child_index))
    QApplication.processEvents()
    if scroll_to:
        # The property form scrolls independently of the panel -- without
        # this the field under review can sit below the fold, off the
        # grabbed image entirely, the same trap 4b.6a's own overlap defect
        # was found by screenshotting rather than by a widget assertion.
        widget = next(w for s, _k, _i, w in panel._rows if s.name == scroll_to)
        panel.property_area.ensureWidgetVisible(widget)
        QApplication.processEvents()


def _capture_reference_states(out_dir: Path) -> list[Path]:
    """4d (trigger constant pickers): a reference row in each of its distinct
    widget states -- catalog (CatalogLineEdit), document (a trigger/variable
    combo), read-only (every widget disabled), and raw Unit (the unchanged
    spinbox, deferred to phase 3.5b). Plus the CatalogBrowseDialog itself.
    """
    from PyQt5.QtWidgets import QApplication

    written: list[Path] = []
    window = _open_window(TRIGGER_FIXTURE)
    panel = window.trigger_panel
    try:
        # catalog: "Fixture: armour split" (list index 1), first effect --
        # object_list_unit_id = 4 (ARCHER).
        _select_effect(panel, 1, 0, scroll_to="object_list_unit_id")
        path = out_dir / "reference_catalog.png"
        _grab(window, path)
        written.append(path)

        # document: "Fixture: references" (list index 2), first effect --
        # trigger_id = 0.
        _select_effect(panel, 2, 0, scroll_to="trigger_id")
        path = out_dir / "reference_document.png"
        _grab(window, path)
        written.append(path)

        # browse dialog, opened from the catalog row above.
        _select_effect(panel, 1, 0, scroll_to="object_list_unit_id")
        from descape.constant_picker import CatalogBrowseDialog, CatalogLineEdit

        catalog_widget = next(
            w for s, _k, _i, w in panel._rows if isinstance(w, CatalogLineEdit)
        )
        browse_dialog = CatalogBrowseDialog(catalog_widget._catalog, catalog_widget._default_category)
        browse_dialog.select(catalog_widget.value())
        browse_dialog.show()
        QApplication.processEvents()
        path = out_dir / "reference_browse_dialog.png"
        browse_dialog.grab().save(str(path))
        browse_dialog.close()
        written.append(path)

        # read-only: same document row, with the file gated off.
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        _select_effect(panel, 2, 0, scroll_to="trigger_id")
        path = out_dir / "reference_read_only.png"
        _grab(window, path)
        written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()

    # raw Unit: a real corpus file's own condition -- the small fixture has
    # no Unit/Unit[] field to show this state on.
    window = _open_window(UNIT_PRESENTATION_FILE)
    panel = window.trigger_panel
    try:
        manager = panel._manager()
        trigger = next(t for t in manager.triggers if t.trigger_id == UNIT_PRESENTATION_TRIGGER_ID)
        list_index = list(manager.triggers).index(trigger)
        panel.select_trigger(list_index)
        QApplication.processEvents()
        conditions = panel.entry_tree.topLevelItem(1)
        panel.entry_tree.setCurrentItem(conditions.child(0))
        QApplication.processEvents()
        unit_widget = next(w for s, _k, _i, w in panel._rows if s.name == "unit_object")
        panel.property_area.ensureWidgetVisible(unit_widget)
        QApplication.processEvents()
        path = out_dir / "reference_raw_unit.png"
        _grab(window, path)
        written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _capture_enum_states(out_dir: Path) -> list[Path]:
    """Large-ENUM picker: a converted enum row, its flat browse dialog, an
    out-of-vocabulary value, a read-only converted enum, and a small enum
    still rendering as a combo as the control. All on "Fixture: armour split"
    (list index 1)."""
    from PyQt5.QtWidgets import QApplication

    from descape.value_picker import ValueBrowseDialog, ValueLineEdit

    written: list[Path] = []
    window = _open_window(TRIGGER_FIXTURE)
    panel = window.trigger_panel
    try:
        # converted: object_attributes = 8 (ObjectAttribute, 147 members),
        # beside operation (Operation, 5 members) still a combo.
        _select_effect(panel, 1, 0, scroll_to="object_attributes")
        path = out_dir / "enum_picker.png"
        _grab(window, path)
        written.append(path)

        _select_effect(panel, 1, 0, scroll_to="operation")
        path = out_dir / "enum_small_combo.png"
        _grab(window, path)
        written.append(path)

        # flat browse dialog, opened on the converted row's own items.
        _select_effect(panel, 1, 0, scroll_to="object_attributes")
        picker = next(w for s, _k, _i, w in panel._rows if s.name == "object_attributes")
        assert isinstance(picker, ValueLineEdit)
        dialog = ValueBrowseDialog(picker._items, show_values=True, title=picker._browse_title)
        dialog.select(picker.value())
        dialog.show()
        QApplication.processEvents()
        path = out_dir / "enum_browse_dialog.png"
        dialog.grab().save(str(path))
        dialog.close()
        written.append(path)

        # out of vocabulary: typed through the raw-integer escape hatch.
        _select_effect(panel, 1, 0, scroll_to="armour_attack_class")
        widget = next(w for s, _k, _i, w in panel._rows if s.name == "armour_attack_class")
        widget.line_edit.setText("9999")
        widget.line_edit.editingFinished.emit()
        QApplication.processEvents()
        path = out_dir / "enum_out_of_vocabulary.png"
        _grab(window, path)
        written.append(path)

        # read-only: the second effect locks armour_attack_class.
        _select_effect(panel, 1, 1, scroll_to="armour_attack_class")
        path = out_dir / "enum_read_only.png"
        _grab(window, path)
        written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


XS_FILE = ROOT / "examples" / "2_Joan_coop_2_v0_15.aoe2scenario"


def _capture_xs_states(out_dir: Path) -> list[Path]:
    """P5-a: a real Script Call effect's XS body in the multi-line editor,
    then the same row on a read-only file, where it is a label."""
    from PyQt5.QtWidgets import QApplication

    written: list[Path] = []
    window = _open_window(XS_FILE)
    panel = window.trigger_panel
    try:
        # Tall enough that the whole six-line band and its scrollbar fit the
        # property pane, instead of the fold cutting the editor off.
        window.resize(1500, 1100)
        QApplication.processEvents()
        manager = panel._manager()
        list_index, effect_index = next(
            (t, e)
            for t, trigger in enumerate(manager.triggers)
            for e, effect in enumerate(trigger.effects)
            if effect.effect_type == 55
        )
        print(f"{XS_FILE.name}: script_call at trigger list index {list_index}, effect {effect_index}")
        _select_effect(panel, list_index, effect_index, scroll_to="message")
        path = out_dir / "xs_editor.png"
        _grab(window, path)
        written.append(path)

        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        _select_effect(panel, list_index, effect_index, scroll_to="message")
        QApplication.processEvents()
        path = out_dir / "xs_read_only.png"
        _grab(window, path)
        written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _capture_retype_states(out_dir: Path) -> list[Path]:
    """GH #37: the detail pane squeezed to MIN_USEFUL_WIDTH with a fourth
    entry button in the row, and the picker in Change mode.

    4b.6b note 2 measured that three buttons fit the 340 px pane and five do
    not, and note 4 records two defects that only a screenshot of this panel
    caught, so the button widths are printed here rather than eyeballed: a
    clipped label reads as a plausible short one in a PNG.
    """
    from PyQt5.QtGui import QFontMetrics
    from PyQt5.QtWidgets import QApplication

    written: list[Path] = []
    window = _open_window(TRIGGER_FIXTURE)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        sizes = panel.splitter.sizes()
        panel.splitter.setSizes([sizes[0], panel.MIN_USEFUL_WIDTH])
        QApplication.processEvents()

        _select_effect(panel, 0, 0)
        path = out_dir / "retype_button_row.png"
        _grab(window, path)
        written.append(path)

        print(f"entry button row at {panel.detail_stack.width()} px "
              f"(MIN_USEFUL_WIDTH is {panel.MIN_USEFUL_WIDTH}):")
        for button in (
            panel.entry_new_button,
            panel.entry_copy_button,
            panel.entry_delete_button,
            panel.entry_retype_button,
        ):
            needed = QFontMetrics(button.font()).horizontalAdvance(button.text())
            verdict = "CLIPPED" if needed > button.width() - 12 else "fits"
            print(f"  {button.text():8s} width={button.width():4d} text={needed:3d}  {verdict}")

        panel._request_entry_op("retype")
        QApplication.processEvents()
        path = out_dir / "retype_picker.png"
        _grab(window, path)
        written.append(path)
        print(f"  picker button reads {panel.picker_add_button.text()!r}, "
              f"{panel.picker_tree.topLevelItemCount()} group(s)")
        panel._close_picker()
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _capture_cluster_states(out_dir: Path) -> list[Path]:
    """The quantity cluster: both "Fixture: armour split" effects before and
    after an object_attributes switch, with no reselect in between. The live
    slot's rows must be editable and every other cluster row a greyed label."""
    from PyQt5.QtWidgets import QApplication

    written: list[Path] = []
    window = _open_window(TRIGGER_FIXTURE)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        cases = ((0, 0, "armor_to_hit_points", "quantity"), (1, 13, "hit_points_to_work_rate", "quantity_float"))
        for child, value, name, live_row in cases:
            _select_effect(panel, 1, child, scroll_to="quantity")
            path = out_dir / f"cluster_{name}_before.png"
            _grab(window, path)
            written.append(path)
            widget = next(w for s, _k, _i, w in panel._rows if s.name == "object_attributes")
            widget.line_edit.setText(str(value))
            widget.line_edit.editingFinished.emit()
            QApplication.processEvents()
            live = next(w for s, _k, _i, w in panel._rows if s.name == live_row)
            panel.property_area.ensureWidgetVisible(live)
            QApplication.processEvents()
            path = out_dir / f"cluster_{name}_after.png"
            _grab(window, path)
            written.append(path)
            # The whole form, unclipped by its scroll area.
            path = out_dir / f"cluster_{name}_after_form.png"
            panel.property_host.grab().save(str(path))
            written.append(path)
            editable = [s.name for s, *_ in panel._rows if not s.read_only]
            print(f"cluster {name}: editable after switch {editable}")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _capture_multi_select_states(out_dir: Path) -> list[Path]:
    """GH #27/#28: the four-button trigger row at MIN_USEFUL_WIDTH, the
    N-selected detail page, a selection spanning two sections, and a pasted
    block below its anchor. Button text widths are printed, not eyeballed."""
    from PyQt5.QtGui import QFontMetrics
    from PyQt5.QtWidgets import QApplication

    written: list[Path] = []
    window = _open_window(TRIGGER_FIXTURE)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        sizes = panel.splitter.sizes()
        panel.splitter.setSizes([panel.MIN_USEFUL_WIDTH, sizes[1]])
        QApplication.processEvents()

        # A divider name groups the fixture into two sections.
        panel.select_trigger(2)
        QApplication.processEvents()
        name = next(w for s, _k, _i, w in panel._rows if s.name == "name")
        name.setText("--- Second section ---")
        name.editingFinished.emit()
        QApplication.processEvents()
        panel.select_triggers([1, 3])
        QApplication.processEvents()
        path = out_dir / "multi_select_two_sections.png"
        _grab(window, path)
        written.append(path)
        print(f"multi-select: page {panel.detail_stack.currentIndex()}, label {panel.multi_label.text()!r}, "
              f"grouped {panel._grouped}")

        print(f"trigger button row at {panel.tree.width()} px (MIN_USEFUL_WIDTH is {panel.MIN_USEFUL_WIDTH}):")
        for button in (
            panel.trigger_new_button,
            panel.trigger_clipboard_copy_button,
            panel.trigger_paste_button,
            panel.trigger_delete_button,
            panel.trigger_copy_button,
        ):
            needed = QFontMetrics(button.font()).horizontalAdvance(button.text())
            verdict = "CLIPPED" if needed > button.width() - 12 else "fits"
            print(f"  {button.text():8s} width={button.width():4d} text={needed:3d}  {verdict}")

        window.copy_triggers()
        panel.select_trigger(0)
        QApplication.processEvents()
        window.paste_triggers()
        QApplication.processEvents()
        path = out_dir / "multi_select_pasted_block.png"
        _grab(window, path)
        written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


TAG_FILE = ROOT / "examples" / "C2_ElCid_coop_1_v0_16.aoe2scenario"


def _capture_tag_states(out_dir: Path) -> list[Path]:
    """Tag management: Rename tag… disabled on "All tags", enabled on a
    chosen tag, and the tree after renaming it (facet follows). The filter
    row's widget widths are printed, not eyeballed."""
    from PyQt5.QtWidgets import QApplication

    if not TAG_FILE.is_file():
        print(f"(skipping tag states -- missing {TAG_FILE.name})")
        return []
    written: list[Path] = []
    window = _open_window(TAG_FILE)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        sizes = panel.splitter.sizes()
        panel.splitter.setSizes([panel.MIN_USEFUL_WIDTH, sizes[1]])
        QApplication.processEvents()

        path = out_dir / "tag_all_tags_disabled.png"
        _grab(window, path)
        written.append(path)
        print(f"tag: All tags -> enabled={panel.tag_rename_button.isEnabled()} "
              f"tooltip={panel.tag_rename_button.toolTip()!r}")
        print(f"filter row at {panel.tree.width()} px (MIN_USEFUL_WIDTH is {panel.MIN_USEFUL_WIDTH}):")
        for label, widget in (
            ("filter", panel.filter_edit),
            ("tag", panel.tag_combo),
            ("rename", panel.tag_rename_button),
            ("sort", panel.sort_combo),
        ):
            print(f"  {label:7s} x={widget.x():4d} width={widget.width():4d} hint={widget.sizeHint().width():4d}")

        panel.set_tag_filter("D1")
        QApplication.processEvents()
        path = out_dir / "tag_d1_rename_enabled.png"
        _grab(window, path)
        written.append(path)
        print(f"tag: D1 -> enabled={panel.tag_rename_button.isEnabled()} "
              f"tooltip={panel.tag_rename_button.toolTip()!r}")

        window.rename_trigger_tag("D1", "Intro")
        QApplication.processEvents()
        path = out_dir / "tag_after_rename_intro.png"
        _grab(window, path)
        written.append(path)
        print(f"tag: after rename facet={panel.current_tag()!r} "
              f"visible={sum(1 for i in panel._item_for_index.values() if not i.isHidden())}")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


SECTION_FILE = ROOT / "examples" / "F7_2_Dos Pilas (648).aoe2scenario"


def _capture_section_states(out_dir: Path) -> list[Path]:
    """Section management at MIN_USEFUL_WIDTH: the section row with a member
    selected, the open Move to Section menu, the tree after a New Section,
    the disabled states with their tooltips, and the divider-free fixture
    grouped by its first New Section. Widths and tooltips are printed."""
    from PyQt5.QtWidgets import QApplication

    if not SECTION_FILE.is_file():
        print(f"(skipping section states -- missing {SECTION_FILE.name})")
        return []
    written: list[Path] = []

    def shot(window, name: str) -> None:
        QApplication.processEvents()
        path = out_dir / name
        _grab(window, path)
        written.append(path)

    def state(panel, label: str) -> None:
        print(f"  {label}: new={panel.section_new_button.isEnabled()} "
              f"rename={panel.section_rename_action.isEnabled()} "
              f"move={panel.section_move_button.isEnabled()} "
              f"move tip={panel.section_move_button.toolTip()!r} "
              f"new tip={panel.section_new_button.toolTip()!r}")

    window = _open_window(SECTION_FILE)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        sizes = panel.splitter.sizes()
        panel.splitter.setSizes([panel.MIN_USEFUL_WIDTH, sizes[1]])
        QApplication.processEvents()
        print(f"sections: {SECTION_FILE.name} at {panel.tree.width()} px (MIN_USEFUL_WIDTH is {panel.MIN_USEFUL_WIDTH}):")
        for label, widget in (
            ("move up", panel.trigger_move_up_button),
            ("move dn", panel.trigger_move_down_button),
            ("new sec", panel.section_new_button),
            ("move to", panel.section_move_button),
        ):
            print(f"  {label:8s} x={widget.x():4d} width={widget.width():4d} hint={widget.sizeHint().width():4d}")

        start = next(s for s in panel._sections if s.header_index is not None)
        panel.reveal_trigger(start.member_indices[0])
        state(panel, "member selected")
        shot(window, "section_member_selected.png")

        menu = panel.section_move_menu
        menu.popup(panel.section_move_button.mapToGlobal(panel.section_move_button.rect().bottomLeft()))
        QApplication.processEvents()
        menu_path = out_dir / "section_move_menu.png"
        menu.grab().save(str(menu_path))
        written.append(menu_path)
        print(f"  move menu: {[(a.text(), a.isEnabled()) for a in menu.actions()]}")
        menu.hide()

        panel.reveal_trigger(start.header_index)
        state(panel, "header selected")
        shot(window, "section_header_selected.png")

        panel.reveal_trigger(start.member_indices[0])
        window.trigger_structural_edit("new_section", [start.member_indices[0]], "DEscape new section")
        print(f"  new section name={window.trigger_panel._manager().triggers[-1].name!r}")
        shot(window, "section_after_new_section.png")

        panel.sort_combo.setCurrentIndex(1)
        state(panel, "file order")
        shot(window, "section_file_order_disabled.png")
        panel.sort_combo.setCurrentIndex(0)
    finally:
        window.edit_history.mark_saved()
        window.close()

    window = _open_window(TRIGGER_FIXTURE)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        sizes = panel.splitter.sizes()
        panel.splitter.setSizes([panel.MIN_USEFUL_WIDTH, sizes[1]])
        state(panel, "fixture, flat")
        window.trigger_structural_edit("new_section", [0], "Later")
        state(panel, "fixture, after New Section")
        shot(window, "section_fixture_grouped.png")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _capture_multi_entry_states(out_dir: Path) -> list[Path]:
    """GH #60: the intersected form for a selection of conditions/effects.
    Every widget kind that can read "(differs)", the blank spinbox before and
    after a value replaces it, and each case at MIN_USEFUL_WIDTH as well as wide."""
    from PyQt5.QtWidgets import QApplication

    create_object, change_ownership = 11, 18
    written: list[Path] = []
    window = _open_window(TRIGGER_FIXTURE)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        QApplication.processEvents()
        total = sum(window.content_splitter.sizes())
        # The trigger list short, so the form under review gets the height.
        panel.splitter.setSizes([180, 820])

        def shot(name: str, focus_field: str = "") -> None:
            for width, suffix in ((600, ""), (int(panel.MIN_USEFUL_WIDTH), "_narrow")):
                window.content_splitter.setSizes([width, total - width])
                QApplication.processEvents()
                if focus_field:
                    widget = next(w for s, _k, _i, w in panel._rows if s.name == focus_field)
                    panel.property_area.ensureWidgetVisible(widget)
                    QApplication.processEvents()
                path = out_dir / f"multi_entry_{name}{suffix}.png"
                _grab(window, path)
                written.append(path)
            print(f"multi-entry {name}: rows {[s.name for s, *_ in panel._rows]}")

        def add_effects(type_id: int, rows) -> list[tuple[str, int]]:
            panel.select_trigger(0)
            QApplication.processEvents()
            base = len(panel._manager().triggers[0].effects)
            for offset, values in enumerate(rows):
                window.entry_structural_edit("new", 0, "effect", -1, type_id)
                for name, value in values.items():
                    entry = panel._manager().triggers[0].effects[base + offset]
                    spec = next(s for s in panel._specs_for("effect", entry) if s.name == name)
                    window.set_entry_field(0, "effect", base + offset, spec, value)
            return [("effect", base + i) for i in range(len(rows))]

        def select(trigger: int, refs) -> None:
            panel.select_trigger(trigger)
            QApplication.processEvents()
            panel.select_entries(refs)
            QApplication.processEvents()

        # Two Modify Attribute effects on different cluster slots.
        select(1, [("effect", 0), ("effect", 1)])
        shot("uniform_pair")
        # Two types sharing one field, and a condition + effect sharing none.
        select(2, [("effect", 0), ("effect", 1)])
        shot("mixed_types")
        select(0, [("condition", 0), ("effect", 0)])
        shot("nothing_shared")

        # The issue's example: three Create Object effects differing in location.
        refs = add_effects(
            create_object,
            [{"object_list_unit_id": 4, "source_player": 1, "location_x": 10 + n, "location_y": 20 + n} for n in range(3)],
        )
        select(0, refs)
        shot("create_objects_blank_spinbox", "location_x")
        spin = next(w for s, _k, _i, w in panel._rows if s.name == "location_x")
        spin.setValue(30)
        QApplication.processEvents()
        print(f"spinbox after commit: {spin.lineEdit().text()!r}, blank {spin.is_indeterminate()}")
        shot("create_objects_committed_spinbox", "location_x")

        # A tristate checkbox and a blank list field.
        refs = add_effects(
            change_ownership,
            [{"flash_object": 0, "selected_object_ids": [1, 2]}, {"flash_object": 1, "selected_object_ids": [3]}],
        )
        select(0, refs)
        shot("tristate_checkbox", "flash_object")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _capture_instruction_preview_states(out_dir: Path) -> list[Path]:
    """GH #140: the Display Instructions preview, wide and at MIN_USEFUL_WIDTH.
    Set AOE2DE_INSTALL_PATH for the unit icon; without it the icon falls back."""
    from PyQt5.QtWidgets import QApplication

    states = (
        ("middle_tags", {"message": "<BLUE>Scout: over here, <GREY>quietly. <<<<<EPIC!>>>>>",
                         "instruction_panel_position": 1, "object_list_unit_id": 448,
                         "play_sound": 1, "sound_name": "Play_Technology_Researched"}),
        ("bottom_long", {"message": "<AQUA>" + "A long line of narration. " * 12,
                         "instruction_panel_position": 2, "object_list_unit_id": 448,
                         "use_tag_color_for_icon": 1, "display_time": -1}),
        ("unset_position", {"instruction_panel_position": -1, "string_id": 60014}),
    )
    written: list[Path] = []
    window = _open_window(TRIGGER_FIXTURE)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        QApplication.processEvents()
        total = sum(window.content_splitter.sizes())
        panel.splitter.setSizes([180, 820])
        entry = panel._manager().triggers[0].effects[0]
        for name, fields in states:
            for field, value in fields.items():
                setattr(entry, field, value)
            panel.select_trigger(0)
            panel.refresh_entries(select=("effect", 0))
            QApplication.processEvents()
            for width, suffix in ((600, ""), (int(panel.MIN_USEFUL_WIDTH), "_narrow")):
                window.content_splitter.setSizes([width, total - width])
                QApplication.processEvents()
                preview = panel._instruction_preview
                panel.property_area.ensureWidgetVisible(preview)
                QApplication.processEvents()
                path = out_dir / f"instruction_preview_{name}{suffix}.png"
                _grab(window, path)
                # The preview alone too: a short pane scrolls its top or footer out of the panel grab.
                alone = out_dir / f"instruction_preview_{name}{suffix}_alone.png"
                preview.grab().save(str(alone))
                written += [path, alone]
            print(f"instruction preview {name}: icon {preview.icon_source}, tint {preview.tint}, "
                  f"footer {preview.footer_label.text()!r}, note {preview.note_label.text()!r}")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


COLLAPSED_COPY_FILE = ROOT / "examples" / "F7_3_York (865).aoe2scenario"


def _section_summary(panel, around: int) -> list[str]:
    """Each section near position `around`: its title and member names."""
    triggers = panel._manager().triggers
    lines = []
    for position in range(max(0, around - 1), min(len(panel._sections), around + 3)):
        section = panel._sections[position]
        title = "(leading)" if section.header_index is None else triggers[section.header_index].name
        members = [triggers[m].name for m in section.member_indices]
        lines.append(f"    [{position}] {title!r}: {len(members)} members {members}")
    return lines


def _capture_collapsed_copy_states(out_dir: Path) -> list[Path]:
    """GH #134 before/after: Duplicate on a collapsed section header, and Copy
    then Paste onto the next collapsed header. Expected: the copy is its own
    section right after the original (Duplicate) or after the next section's
    end (Paste), with the same member count and no originals interleaved."""
    from PyQt5.QtWidgets import QAbstractItemView, QApplication

    if not COLLAPSED_COPY_FILE.is_file():
        print(f"(skipping collapsed copy states -- missing {COLLAPSED_COPY_FILE.name})")
        return []
    written: list[Path] = []

    def shot(window, header_index: int, name: str) -> None:
        panel = window.trigger_panel
        QApplication.processEvents()
        panel.tree.scrollToItem(panel._item_for_index[header_index], QAbstractItemView.PositionAtTop)
        QApplication.processEvents()
        path = out_dir / name
        _grab(window, path)
        written.append(path)

    for verb in ("duplicate", "paste"):
        window = _open_window(COLLAPSED_COPY_FILE)
        panel = window.trigger_panel
        try:
            window.resize(1500, 1100)
            sizes = panel.splitter.sizes()
            # The tree pane's share of the panel's vertical split: room for a section and its copy.
            panel.splitter.setSizes([600, sizes[1]])
            position = next(
                p for p, s in enumerate(panel._sections)
                if p > 0 and s.header_index is not None and 2 <= len(s.member_indices) <= 6
            )
            section = panel._sections[position]
            panel._item_for_index[section.header_index].setExpanded(False)
            panel.select_trigger(section.header_index)
            print(f"collapsed copy ({verb}): {COLLAPSED_COPY_FILE.name}, section {position}, before:")
            print("\n".join(_section_summary(panel, position)))
            shot(window, section.header_index, f"collapsed_{verb}_before.png")
            if verb == "duplicate":
                panel.trigger_copy_button.click()
            else:
                panel.trigger_clipboard_copy_button.click()
                after = panel._sections[position + 1]
                panel._item_for_index[after.header_index].setExpanded(False)
                panel.select_trigger(after.header_index)
                panel.trigger_paste_button.click()
            print("  after:")
            print("\n".join(_section_summary(panel, position)))
            shot(window, section.header_index, f"collapsed_{verb}_after.png")
            panel.expand_all_sections()
            # Paste lands a whole section further down, so frame the pasted header there.
            triggers = panel._manager().triggers
            focus = section.header_index if verb == "duplicate" else next(
                s.header_index for s in panel._sections
                if s.header_index is not None and triggers[s.header_index].name.endswith("(copy) ---")
            )
            shot(window, focus, f"collapsed_{verb}_after_expanded.png")
        finally:
            window.edit_history.mark_saved()
            window.close()
    return written


def _capture_section_move_states(out_dir: Path) -> list[Path]:
    """GH #133 before/after: Section Down on a small F7_3_York section.
    Expected: the section and the one below trade places whole, each keeps
    its member count, no member leaks across, and the Exec # cells renumber
    to the new display slots."""
    from PyQt5.QtWidgets import QAbstractItemView, QApplication

    if not COLLAPSED_COPY_FILE.is_file():
        print(f"(skipping section move states -- missing {COLLAPSED_COPY_FILE.name})")
        return []
    written: list[Path] = []
    window = _open_window(COLLAPSED_COPY_FILE)
    panel = window.trigger_panel
    try:
        window.resize(1500, 1100)
        sizes = panel.splitter.sizes()
        panel.splitter.setSizes([600, sizes[1]])
        position = next(
            p for p, s in enumerate(panel._sections)
            if p > 0 and s.header_index is not None and 2 <= len(s.member_indices) <= 6
            and p + 1 < len(panel._sections) and len(panel._sections[p + 1].member_indices) <= 6
        )
        moved = panel._sections[position].header_index
        below = panel._sections[position + 1].header_index
        panel.select_trigger(moved)
        print(f"section move: {COLLAPSED_COPY_FILE.name}, Section Down on section {position}, before:")
        print("\n".join(_section_summary(panel, position)))
        for name, top in (("f7_3_york_section_before.png", moved), ("f7_3_york_section_after.png", below)):
            if name.endswith("after.png"):
                panel.section_down_button.click()
                print("  after:")
                print("\n".join(_section_summary(panel, position)))
                print(f"  undo label: {window.edit_history.peek_undo().label!r}")
            QApplication.processEvents()
            panel.tree.scrollToItem(panel._item_for_index[top], QAbstractItemView.PositionAtTop)
            QApplication.processEvents()
            path = out_dir / name
            _grab(window, path)
            written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


_ONLY = {
    "instruction_preview": _capture_instruction_preview_states,
    "collapsed_copy": _capture_collapsed_copy_states,
    "section_move": _capture_section_move_states,
}


def generate(out_dir: Path, only: str | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if only is not None:
        return _ONLY[only](out_dir)
    written: list[Path] = []
    for prefix, filename, expected_sections in TARGETS:
        written += _capture_one(prefix, filename, expected_sections, out_dir)
        written += _capture_navigation_states(prefix, filename, out_dir)
    written += _capture_reference_states(out_dir)
    written += _capture_enum_states(out_dir)
    written += _capture_xs_states(out_dir)
    written += _capture_retype_states(out_dir)
    written += _capture_cluster_states(out_dir)
    written += _capture_multi_select_states(out_dir)
    written += _capture_multi_entry_states(out_dir)
    written += _capture_tag_states(out_dir)
    written += _capture_section_states(out_dir)
    written += _capture_instruction_preview_states(out_dir)
    written += _capture_collapsed_copy_states(out_dir)
    written += _capture_section_move_states(out_dir)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument("--only", choices=sorted(_ONLY), help="Capture one group of states only")
    args = parser.parse_args()

    if not PYQT5_AVAILABLE:
        raise SystemExit("PyQt5 not importable -- this script has no Qt-free fallback")

    import tempfile

    _ensure_qapp()
    with tempfile.TemporaryDirectory() as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        written = generate(args.out_dir, args.only)

    for path in written:
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()
