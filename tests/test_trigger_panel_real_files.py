"""Trigger panel on real scenarios: populate, filter, grouping, byte-identical
browsing.
"""

from __future__ import annotations

import pytest
from trigger_panel_helpers import (
    _no_install_env,  # noqa: F401 -- autouse fixture
    _row_widget,
    _trigger_row_count,
    _walk_every_row,
    _window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def test_panel_populates_and_filters_a_real_scenario(scenario_path) -> None:
    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        tree = panel.tree

        if not window.scenario.trigger_read_supported:
            # The 1.54/trigger-3.9 set: an explanatory message, no rows, and
            # the filter disabled rather than a silently empty list.
            assert tree.topLevelItemCount() == 0
            assert "can't be read" in panel.status.text()
            assert not panel.filter_edit.isEnabled()
            return

        # The status line counts triggers, not tree rows -- these diverge
        # once 4c's grouping turns some rows into section headers with
        # children, so this counts every real trigger row, nested or not.
        count = _trigger_row_count(tree)
        assert f"{count} trigger" in panel.status.text()
        if count == 0:
            return

        all_items = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        all_items += [c for top in all_items for c in (top.child(j) for j in range(top.childCount()))]

        # A filter matching nothing hides every row; clearing restores them.
        panel.filter_edit.setText("zzz-no-such-trigger-zzz")
        assert all(item.isHidden() for item in all_items)
        panel.filter_edit.setText("")
        assert not any(item.isHidden() for item in all_items)

        # Filtering on a real trigger's own name keeps at least that one.
        # tree.topLevelItem(0) may be 4c's synthetic header (no real name of
        # its own), so pick the first item that actually carries a trigger.
        from PyQt5.QtCore import Qt

        real = next(item for item in all_items if item.data(0, Qt.UserRole) is not None)
        panel.filter_edit.setText(real.text(panel._COL_NAME))
        assert not real.isHidden()
    finally:
        window.edit_history.mark_saved()
        window.close()


# Stays corpus: the test map stores the exec-order byte, so it would only skip there.
@pytest.mark.corpus
def test_execution_order_readout_covers_the_no_stored_byte_case(scenario_path) -> None:
    """The third readout state -- "not stored" -- needs a real file to reach:
    F7_3_York is scenario 1.55 but its trigger version is below 4.5, so the
    exec-order byte consumes zero bytes despite the version implying
    otherwise. Not hardcoded to that
    filename -- any corpus file with the same shape exercises this the same
    way, and the corpus tier's own summary already reports skip counts, so a
    corpus with no such file reads as 100% skipped rather than silently
    green.
    """
    from descape.trigger_model import exec_order_value

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if exec_order_value(window.scenario) is not None:
            pytest.skip(f"{scenario_path.name}: this file does store the exec-order byte")

        assert "Execution order is not stored in this file." in window.trigger_panel.status.text()
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- 4c: trigger organization (grouped tree, tag facet) -----------------------
#
# The shipped default-tier fixture has 4 triggers and no dividers, so it
# cannot exercise grouping at all -- see trigger_organize.py's own module
# docstring. These run at corpus tier instead, against real files whose
# authors actually used the `--- Section ---`/`[tag]` conventions this
# feature is built on. Generic over scenario_path rather than hardcoded to
# one corpus filename, with an explicit skip when a given file has nothing to
# exercise -- the corpus tier's own summary already reports skip counts, so
# an empty corpus reads as 100% skipped rather than silently green.


def _panel_sections(panel) -> list:
    from PyQt5.QtCore import Qt

    from descape.trigger_organize import Section

    result = []
    for i in range(panel.tree.topLevelItemCount()):
        top = panel.tree.topLevelItem(i)
        header_index = top.data(0, Qt.UserRole)
        members = tuple(top.child(c).data(0, Qt.UserRole) for c in range(top.childCount()))
        result.append(Section("", header_index, members))
    return result


def test_grouped_tree_matches_trigger_organize_sections_on_a_real_scenario(scenario_path) -> None:
    """Cross-checks the panel's own grouped-tree construction against an
    independent call to trigger_organize.sections() on the same manager data
    -- the panel must not silently diverge from the heuristic it's built on."""
    from descape.trigger_organize import sections as compute_sections

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        manager = panel._manager()
        if manager is None or not manager.triggers:
            pytest.skip(f"{scenario_path.name}: no triggers")

        names = [t.name or "" for t in manager.triggers]
        order = list(manager.trigger_display_order)
        expected = compute_sections(names, order)
        is_grouped = not (len(expected) == 1 and expected[0].header_index is None)

        assert panel._grouped == is_grouped
        if is_grouped:
            # Compare header/member indices only: the tree's own header text
            # comes from _trigger_label() (adds "(disabled)", substitutes
            # "(unnamed)"), not the raw name compute_sections() used.
            got = [(s.header_index, s.member_indices) for s in _panel_sections(panel)]
            want = [(s.header_index, s.member_indices) for s in expected]
            assert got == want
        else:
            assert _trigger_row_count(panel.tree) == panel.tree.topLevelItemCount()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_file_order_is_always_flat_on_a_real_scenario(scenario_path) -> None:
    """4c's grouping exists only under Display order -- switching to File
    order must flatten the tree even on a file whose names are full of
    dividers."""
    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if not panel.tree.topLevelItemCount():
            pytest.skip(f"{scenario_path.name}: no triggers")

        panel.sort_combo.setCurrentIndex(1)  # File order
        assert not panel._grouped
        for i in range(panel.tree.topLevelItemCount()):
            assert panel.tree.topLevelItem(i).childCount() == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_tag_facet_hides_non_matching_rows_on_a_real_scenario(scenario_path) -> None:
    """The tag combo matches the stashed _TAG_ROLE, never the rendered label
    -- see that constant's docstring for why inheriting the label-matching
    false-positive class into a facet would be worse here than in the
    free-text filter."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if panel.tag_combo.count() <= 1:
            pytest.skip(f"{scenario_path.name}: no [tag] names in this file")

        panel.tag_combo.setCurrentIndex(1)  # the first real tag, alphabetical
        wanted = panel.tag_combo.currentData()
        assert wanted is not None

        for i in range(panel.tree.topLevelItemCount()):
            top = panel.tree.topLevelItem(i)
            if top.isHidden():
                continue
            visible_children = [
                top.child(c) for c in range(top.childCount()) if not top.child(c).isHidden()
            ]
            header_matches = top.data(0, Qt.UserRole) is not None and wanted in top.data(0, panel._TAG_ROLE)
            # A visible row must owe its visibility to a real match -- its own
            # tag chain (flat rows and real section headers), or a visible child's.
            assert header_matches or visible_children
            for child in visible_children:
                assert wanted in child.data(0, panel._TAG_ROLE)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_across_the_divider_predicate_regroups_a_real_scenario(scenario_path) -> None:
    """4c's silent-staleness case 1: a rename that flips is_divider() must
    trigger a full repopulate, promoting the renamed trigger to its own
    section header -- a label patch alone would leave the tree structurally
    stale."""
    from descape.trigger_organize import is_divider

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if not panel._editable:
            pytest.skip(f"{scenario_path.name}: triggers are read-only")
        manager = panel._manager()
        if manager is None or not manager.triggers:
            pytest.skip(f"{scenario_path.name}: no triggers")

        ordinary = next((i for i, t in enumerate(manager.triggers) if not is_divider(t.name or "")), None)
        if ordinary is None:
            pytest.skip(f"{scenario_path.name}: every trigger name is already divider-shaped")

        panel.select_trigger(ordinary)
        name = _row_widget(panel, "name")
        name.setText("--- 4c test section ---")
        name.editingFinished.emit()

        assert panel._grouped
        renamed_item = panel._item_for_index[ordinary]
        assert renamed_item.parent() is None, "a divider-named trigger becomes its own section header"
        assert panel.current_trigger_index() == ordinary, "the renamed trigger stays selected"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_opening_a_grouped_real_scenario_selects_a_trigger_not_the_synthetic_header(scenario_path) -> None:
    """4c's silent-staleness case 2: opening a file with a before-first run of
    triggers must land on the first real trigger, not the synthetic
    "(before the first section)" header -- selecting the header would open
    the entry pane empty."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if not panel._grouped or not panel.tree.topLevelItemCount():
            pytest.skip(f"{scenario_path.name}: not grouped")
        first = panel.tree.topLevelItem(0)
        if first.data(0, Qt.UserRole) is not None:
            pytest.skip(f"{scenario_path.name}: no leading run before the first divider")

        assert first.text(panel._COL_ID) == "", "the synthetic header is not a trigger and has no id to show"
        assert first.text(panel._COL_POS) == "", "nor a display position"
        assert panel.current_trigger_index() is not None
        assert panel.tree.currentItem() is first.child(0)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_browsing_a_grouped_real_scenario_saves_byte_identically(scenario_path, tmp_path) -> None:
    """Extends test_browsing_every_row_saves_byte_identically's guarantee to
    grouped browsing: expanding/collapsing every section and switching the
    tag facet must never build a TriggerEditModel, same as plain browsing."""
    from descape.scenario_write import write_scenario

    window = _window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        if not window.scenario.trigger_read_supported:
            pytest.skip(f"{scenario_path.name}: triggers do not parse")
        if not panel._grouped:
            pytest.skip(f"{scenario_path.name}: no dividers, nothing to group")

        from PyQt5.QtWidgets import QApplication

        for i in range(panel.tree.topLevelItemCount()):
            top = panel.tree.topLevelItem(i)
            top.setExpanded(False)
            QApplication.processEvents()
            top.setExpanded(True)
            QApplication.processEvents()

        # The Sections menu: one jump per heuristic section, and collapse,
        # expand and every jump are browsing too.
        assert len(panel._section_actions) == len(panel._sections) == panel.tree.topLevelItemCount()
        panel.collapse_all_action.trigger()
        for action in panel._section_actions:
            action.trigger()
            QApplication.processEvents()
        panel.expand_all_action.trigger()

        for i in range(panel.tag_combo.count()):
            panel.tag_combo.setCurrentIndex(i)
            QApplication.processEvents()
        panel.tag_combo.setCurrentIndex(0)

        _walk_every_row(panel)

        assert window.trigger_edits is None, "browsing must not build an edit model"
        assert not window.edit_history.is_dirty, "browsing must record no undo step"

        out = tmp_path / scenario_path.name
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == scenario_path.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()
