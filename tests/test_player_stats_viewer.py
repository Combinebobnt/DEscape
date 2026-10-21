"""GH #5 and GH #145: View mode's per-player stats combo and the page-0 table,
driven through a real offscreen ViewerWindow. The counts themselves are
tests/test_player_stats.py."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNITS_FIXTURE = FIXTURES / "units_120x120.aoe2scenario"
TRIGGER_FIXTURE = FIXTURES / "triggers_120x120.aoe2scenario"
_PLACE_CONST = 83


def _window(path: Path):
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(path)
    assert window.scenario is not None, f"{path.name} failed to load"
    return window


def _player_items(window) -> dict[tuple[str, str], object]:
    """(parent label, label) -> the Player section's tree item."""
    from PyQt5.QtCore import Qt

    section = window._info_sections["Player"]
    items = {}
    # Child 0 is the combo's row.
    for i in range(1, section.childCount()):
        top = section.child(i)
        items[("", top.data(0, Qt.UserRole))] = top
        for j in range(top.childCount()):
            items[(top.data(0, Qt.UserRole), top.child(j).data(0, Qt.UserRole))] = top.child(j)
    return items


def _row(window, label: str, parent: str = "") -> str:
    """The row's count and note as displayed, checked against the tree item's own text."""
    rows = {(row[3], row[0]): row for row in window.player_stats_rows}
    assert (parent, label) in rows, window.player_stats_rows
    _, count, note, _ = rows[(parent, label)]
    item = _player_items(window)[(parent, label)]
    shown = " ".join(item.text(col) for col in range(3))
    assert all(part in shown for part in (count, note) if part), (shown, count, note)
    return " ".join(part for part in (count, note) if part)


def _labels(window) -> set[str]:
    return {label for _parent, label in _player_items(window)}


def _placements(window) -> int:
    return int(_row(window, "Placements").replace(",", ""))


def _combo_items(window) -> list[tuple[str, int]]:
    combo = window.stats_player_combo
    return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]


def test_the_combo_lists_gaia_then_the_defined_players_and_defaults_to_player_one() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        # GH #130: the fixture's tribe names are empty, so the labels are bare P<n>.
        assert _combo_items(window) == [("GAIA", 0), ("P1", 1), ("P2", 2)]
        assert window.stats_player_combo.currentData() == 1
        assert window.stats_player_combo.currentText() == "P1"
        assert _placements(window) == 3
        assert "Units per player" not in window._info_text()
    finally:
        conftest.close_window(window)


def test_every_combo_item_carries_its_players_colour_swatch() -> None:
    """GH #5 step 2: GAIA included, each item's icon is its player's colour."""
    window = _window(UNITS_FIXTURE)
    try:
        combo = window.stats_player_combo
        assert combo.count() == 3
        for i in range(combo.count()):
            icon = combo.itemIcon(i)
            assert not icon.isNull(), combo.itemText(i)
            pixel = icon.pixmap(16, 16).toImage().pixelColor(8, 8)
            expected = tuple(window.scenario.player_colors[combo.itemData(i)])
            assert (pixel.red(), pixel.green(), pixel.blue()) == expected, combo.itemText(i)
    finally:
        conftest.close_window(window)


def test_lowering_the_player_count_labels_a_slot_that_owns_units_inactive() -> None:
    """GH #5 step 7 through a real Number of Players edit. The fixture sits at
    the spinbox's minimum of 2, so it first gains a P3 unit and a third slot."""
    window = _window(UNITS_FIXTURE)
    try:
        model = window._ensure_unit_edits()
        assert model is not None
        with window._unit_edit(model, "Place for P3", [3]):
            model.add(3, _PLACE_CONST, 30.5, 30.5)
        window.mode_combo.setCurrentText("Players")
        spin = window.players_panel.player_count_spin
        assert spin.isEnabled(), "Number of Players is read-only on this fixture"
        assert spin.value() == 2
        assert _combo_items(window)[-1] == ("P3 (inactive)", 3)

        spin.setValue(3)
        assert _combo_items(window)[-1] == ("P3", 3)
        spin.setValue(2)
        assert _combo_items(window)[-1] == ("P3 (inactive)", 3)

        window.undo()
        assert spin.value() == 3
        assert _combo_items(window)[-1] == ("P3", 3)
    finally:
        conftest.close_window(window)


def test_an_inactive_slot_that_owns_placements_is_listed_and_labelled(monkeypatch) -> None:
    window = _window(UNITS_FIXTURE)
    try:
        monkeypatch.setattr(window, "_current_active_players", lambda: [1])
        window._repopulate_stats_players()
        assert _combo_items(window) == [("GAIA", 0), ("P1", 1), ("P2 (inactive)", 2)]
    finally:
        conftest.close_window(window)


def test_player_select_keybinds_drive_the_combo_in_view_mode() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        assert window.mode == "view"
        window._select_player(0)
        assert window.stats_player_combo.currentData() == 0
        assert window.stats_player_combo.currentText() == "GAIA"
        window._select_player(2)
        assert window.stats_player_combo.currentData() == 2
        assert _placements(window) == 2
        window._select_player(5)  # not defined: selection stays put
        assert window.stats_player_combo.currentData() == 2
    finally:
        conftest.close_window(window)


def test_placing_and_undoing_a_unit_moves_the_count_both_ways() -> None:
    """The regression guard for the old per-player list, which only
    refreshed on load, Save As and a trigger-parse mode switch."""
    from PyQt5.QtCore import QPointF, Qt

    window = _window(UNITS_FIXTURE)
    try:
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        window.mode_combo.setCurrentText("Units")
        assert window.stats_player_combo.currentData() == 1
        before = _placements(window)
        window.unrestricted_check.setChecked(True)  # (10, 10) is under the fixture's House
        window.units_panel.select_object(_PLACE_CONST)
        window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        tp = window.map_view._tile_pixels
        window.on_unit_place(QPointF(10 * tp + tp // 2, 10 * tp + tp // 2), Qt.NoModifier)
        assert _placements(window) == before + 1
        window.undo()
        assert _placements(window) == before
        window.redo()
        assert _placements(window) == before + 1
    finally:
        conftest.close_window(window)


def test_view_mode_never_pays_the_trigger_parse() -> None:
    window = _window(TRIGGER_FIXTURE)
    try:
        assert window.mode == "view"
        window._select_player(0)
        window._select_player(1)
        assert window.scenario.trigger_read_supported is None
        assert _row(window, "Triggers") == "not counted yet (enter Triggers mode or run Map Analysis)"
    finally:
        conftest.close_window(window)


def test_map_analysis_updates_both_info_halves() -> None:
    """_show_analysis() parses the triggers but used to leave the info panel
    saying they weren't, since the mode-switch refresh never saw the flip."""
    window = _window(TRIGGER_FIXTURE)
    try:
        assert "Embedded XS: (unknown until triggers are parsed)" in window._info_text()
        window._show_analysis()
        assert window.scenario.trigger_read_supported is True
        assert "Embedded XS: (none)" in window._info_text()
        assert re.fullmatch(r"2 +\(of 4; 2 reference no player\)", _row(window, "Triggers"))
    finally:
        window._close_analysis_dialog()
        conftest.close_window(window)


def test_entering_triggers_mode_turns_the_hint_into_a_count() -> None:
    window = _window(TRIGGER_FIXTURE)
    try:
        window.mode_combo.setCurrentText("Triggers")
        window.mode_combo.setCurrentText("View")
        assert re.fullmatch(r"2 +\(of 4; 2 reference no player\)", _row(window, "Triggers"))
    finally:
        conftest.close_window(window)


def test_an_unparseable_triggers_section_reads_unavailable_not_zero() -> None:
    from descape import map_analysis

    window = _window(TRIGGER_FIXTURE)
    try:
        window.scenario.trigger_read_supported = False
        window._update_player_stats()
        assert _row(window, "Triggers") == map_analysis._TRIGGERS_UNAVAILABLE
    finally:
        conftest.close_window(window)


def test_closing_the_scenario_clears_the_combo() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        window.close_scenario()
        assert window.stats_player_combo.count() == 0
        assert not window.stats_player_combo.isEnabled()
        assert window.player_stats_rows == []
        assert window._info_sections["Player"].childCount() == 1  # only the combo's row
        assert window._info_sections["File"].childCount() == 0
    finally:
        conftest.close_window(window)


# -- GH #145: the table, its sections and the hide rule ------------------------

_GAIA_SIDE = {"Trees", "Resource piles", "Gold mines", "Animals", "Cliffs", "Relics", "Resources on map"}


def test_a_players_rows_carry_no_gaia_side_rows_and_gaias_do() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        window._select_player(1)
        assert not _labels(window) & _GAIA_SIDE, _labels(window)
        assert _row(window, "Economy", "Units") == "1"
        assert _row(window, "Military", "Units") == "1"
        assert _row(window, "Water", "Units") == "0 also counted above"
        assert _row(window, "Economy", "Buildings") == "1"

        window._select_player(0)
        assert _labels(window).issuperset(_GAIA_SIDE), _labels(window)
        assert _row(window, "Trees") == "2"
        assert _row(window, "Wood", "Resources on map") == "200"  # two trees at 100
        # The other side for GAIA: only non-zero unit and building rows. The fixture's GAIA owns one wall.
        assert _row(window, "Walls & gates", "Buildings") == "1"
        assert ("Units", "Economy") not in _player_items(window)
        assert ("", "Units") not in _player_items(window)
    finally:
        conftest.close_window(window)


def test_placing_a_villager_bumps_economy_and_undo_reverts_it() -> None:
    from PyQt5.QtCore import QPointF, Qt

    window = _window(UNITS_FIXTURE)
    try:
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        window.mode_combo.setCurrentText("Units")
        before = int(_row(window, "Economy", "Units"))
        window.units_panel.select_object(_PLACE_CONST)  # 83, a villager
        window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        tp = window.map_view._tile_pixels
        window.on_unit_place(QPointF(12 * tp + tp // 2, 12 * tp + tp // 2), Qt.NoModifier)
        assert int(_row(window, "Economy", "Units")) == before + 1
        window.undo()
        assert int(_row(window, "Economy", "Units")) == before
    finally:
        conftest.close_window(window)


def test_the_sections_are_file_player_terrain_technical_in_order() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        tree = window.info_tree
        titles = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
        assert titles == ["File", "Player", "Terrain (top 8)", "Technical"]
        assert tree.itemWidget(tree.topLevelItem(1).child(0), 0) is window.stats_player_combo.parentWidget()
    finally:
        conftest.close_window(window)


def test_a_short_window_shows_the_tables_own_scrollbar() -> None:
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1200, 500)
    try:
        window.load_scenario(UNITS_FIXTURE)
        window._select_player(0)
        QApplication.processEvents()
        assert window.info_tree.verticalScrollBar().maximum() > 0
    finally:
        conftest.close_window(window)


def test_the_combo_survives_refreshes_as_the_player_headers_widget() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        combo = window.stats_player_combo
        header = window._info_sections["Player"]
        for _ in range(3):
            window._update_player_stats()
            window._update_info()
        assert window.stats_player_combo is combo
        assert window._info_sections["Player"] is header
        assert window.info_tree.itemWidget(header.child(0), 0) is combo.parentWidget()
        assert combo.count() == 3
    finally:
        conftest.close_window(window)


def test_the_combo_is_wide_enough_for_a_long_player_label() -> None:
    """At the default layout one column gave the combo ~90 px, clipping "P1 - <tribe name>".
    Its own spanned row gives it the tree's width, and the Value column stays tight."""
    from PyQt5.QtWidgets import QApplication

    from descape import player_fields

    window = conftest.shown_window(1500, 900)
    try:
        window.load_scenario(UNITS_FIXTURE)
        spec = {s.field_id: s for s in player_fields.specs_for(window.scenario)}["tribe_name"]
        window.set_player_field(spec, 1, "Horde Army")
        QApplication.processEvents()
        combo = window.stats_player_combo
        tree = window.info_tree
        assert combo.currentText() == "P1 - Horde Army"
        needed = combo.fontMetrics().horizontalAdvance(combo.currentText()) + combo.iconSize().width()
        assert combo.width() >= needed, (combo.width(), needed)
        assert combo.mapTo(tree.viewport(), combo.rect().topRight()).x() < tree.viewport().width()
        assert tree.columnWidth(1) < combo.width()
        window._stats_player_row.setSelected(True)
        assert tree.selected_text() == "Player\tP1 - Horde Army"
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_at_the_view_floor_a_hidden_note_still_copies_and_map_size_reads_as_free_text() -> None:
    """TASK-407. Pass conditions, View entered from a 150 px Units pane (its floor): the Note
    column is hidden; copying Player 1's "Other" units row gives its count and note; the File
    section's Map size is one free-text line, "Map size: 120 x 120"."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1500, 900)
    try:
        window.load_scenario(UNITS_FIXTURE)
        window.mode_combo.setCurrentText("Units")
        total = sum(window.content_splitter.sizes())
        window.content_splitter.setSizes([150, total - 150])
        QApplication.processEvents()
        window.mode_combo.setCurrentText("View")
        QApplication.processEvents()
        assert window.content_splitter.sizes()[0] == window.INFO_PAGE_MIN_USEFUL_WIDTH
        window._select_player(1)
        tree = window.info_tree
        assert tree.header().isSectionHidden(2)
        other = _player_items(window)[("Units", "Other")]
        tree.clearSelection()
        other.setSelected(True)
        count = _row(window, "Other", "Units").split()[0]
        assert tree.selected_text() == f"Other\t{count}\tflags, revealers, markers"
        assert "Map size: 120 x 120" in window._info_text().splitlines()
    finally:
        conftest.close_window(window)


def test_changing_the_combo_rebuilds_only_the_player_children() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        static = {
            title: [window._info_sections[title].child(i) for i in range(window._info_sections[title].childCount())]
            for title in ("File", "Terrain (top 8)", "Technical")
        }
        player_before = list(_player_items(window).values())
        window._select_player(2)
        for title, items in static.items():
            section = window._info_sections[title]
            assert [section.child(i) for i in range(section.childCount())] == items, title
        assert not set(map(id, player_before)) & set(map(id, _player_items(window).values()))
    finally:
        conftest.close_window(window)


def test_ctrl_c_copies_the_selected_rows_tab_separated() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1200, 800)
    try:
        window.load_scenario(UNITS_FIXTURE)
        # A committed region arms the window's Copy Region on the same key, which the tree must win.
        window._region = (0, 0, 4, 4)
        window._update_tool_enabled()
        assert window.copy_action.isEnabled()
        items = _player_items(window)
        items[("", "Placements")].setSelected(True)
        items[("Units", "Economy")].setSelected(True)
        window.info_tree.setFocus()
        QApplication.clipboard().clear()
        key = QKeySequence(QKeySequence.Copy)[0]
        QTest.keyClick(window.info_tree, key & ~Qt.KeyboardModifierMask, Qt.KeyboardModifiers(key & Qt.KeyboardModifierMask))
        assert QApplication.clipboard().text() == "Placements\t3\nEconomy\t1"
    finally:
        conftest.close_window(window)


# -- the owed in-app passes, offscreen (TASK-153) --------------------------------


def _example(name: str) -> Path:
    path = Path(__file__).resolve().parent.parent / "examples" / name
    if not path.is_file():
        pytest.skip(f"{name} not in examples/")
    return path


def _assert_rows(window, expected: dict[tuple[str, str], str]) -> None:
    """Each (parent, label) row shows exactly `text` (count, then note if any)."""
    shown = {key: _row(window, key[1], key[0]) for key in expected}
    assert shown == expected


@pytest.mark.corpus
def test_old_allies_player_five_shows_the_measured_table() -> None:
    """GH #145 owed step (1), the View table as displayed (numbers from the task body).

    Pass conditions, P5 in View mode: Placements 1,910; Units 861 (Economy 47, Military 663,
    Other 151, Water 2, Unique 466, Heroes 2); Buildings 1,039 (Economy 167, Military 109,
    Walls & gates 753, Other 10); Animals 4; Eye candy 6; after a Triggers-mode visit,
    Triggers 95 (of 590).
    """
    window = _window(_example("old-allies-final-v2.aoe2scenario"))
    try:
        window._select_player(5)
        _assert_rows(window, {
            ("", "Placements"): "1,910",
            ("", "Units"): "861",
            ("Units", "Economy"): "47",
            ("Units", "Military"): "663",
            ("Units", "Other"): "151 flags, revealers, markers",
            ("Units", "Water"): "2 also counted above",
            ("Units", "Unique"): "466 also counted above",
            ("Units", "Heroes"): "2 also counted above",
            ("", "Buildings"): "1,039",
            ("Buildings", "Economy"): "167",
            ("Buildings", "Military"): "109",
            ("Buildings", "Walls & gates"): "753",
            ("Buildings", "Other"): "10",
            ("", "Animals"): "4",
            ("", "Eye candy"): "6",
        })
        window.mode_combo.setCurrentText("Triggers")
        window.mode_combo.setCurrentText("View")
        assert window.stats_player_combo.currentData() == 5
        assert re.match(r"95 +\(of 590;", _row(window, "Triggers")), _row(window, "Triggers")
    finally:
        conftest.close_window(window)


@pytest.mark.corpus
def test_old_allies_gaia_shows_the_measured_table() -> None:
    """GH #145 owed step (2). Pass conditions, GAIA in View mode: Placements 8,687; Food
    23,665 / Wood 612,275 / Gold 117,600 / Stone 17,150; Resource piles 309 (Gold 147, Stone
    49, Berry 57, Fish 56); Trees 6,153; Animals 30; Cliffs 907; Relics 1; Units 100;
    Buildings 14; Eye candy 1,173."""
    window = _window(_example("old-allies-final-v2.aoe2scenario"))
    try:
        window._select_player(0)
        _assert_rows(window, {
            ("", "Placements"): "8,687",
            ("Resources on map", "Food"): "23,665",
            ("Resources on map", "Wood"): "612,275",
            ("Resources on map", "Gold"): "117,600",
            ("Resources on map", "Stone"): "17,150",
            ("", "Resource piles"): "309",
            ("Resource piles", "Gold mines"): "147",
            ("Resource piles", "Stone mines"): "49",
            ("Resource piles", "Berry bushes"): "57",
            ("Resource piles", "Fish"): "56",
            ("", "Trees"): "6,153",
            ("", "Animals"): "30",
            ("", "Cliffs"): "907",
            ("", "Relics"): "1",
            ("", "Units"): "100",
            ("", "Buildings"): "14",
            ("", "Eye candy"): "1,173",
        })
    finally:
        conftest.close_window(window)


@pytest.mark.corpus
def test_dos_pilas_gaia_shows_the_measured_table() -> None:
    """GH #145 owed step (3). Pass conditions, GAIA in View mode: Placements 28,466; Food
    251,725 / Wood 2,466,900 / Gold 49,600 / Stone 23,100; Resource piles 1,327; Trees
    24,745; Cliffs 546; Relics 23."""
    window = _window(_example("F7_2_Dos Pilas (648).aoe2scenario"))
    try:
        window._select_player(0)
        _assert_rows(window, {
            ("", "Placements"): "28,466",
            ("Resources on map", "Food"): "251,725",
            ("Resources on map", "Wood"): "2,466,900",
            ("Resources on map", "Gold"): "49,600",
            ("Resources on map", "Stone"): "23,100",
            ("", "Resource piles"): "1,327",
            ("", "Trees"): "24,745",
            ("", "Cliffs"): "546",
            ("", "Relics"): "23",
        })
    finally:
        conftest.close_window(window)


def test_a_short_window_scrolls_the_table_down_to_embedded_xs() -> None:
    """GH #145 owed step (4). Pass conditions, a 1200x500 window with GAIA's (longest) table:
    the Embedded XS row starts below the visible area, and once the table's own scroll bar is
    at its end that row's full height lies inside the viewport (vertical scrolling only: the
    row's width at this pane size is step (6)'s question)."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1200, 500)
    try:
        window.load_scenario(UNITS_FIXTURE)
        window._select_player(0)
        QApplication.processEvents()
        tree = window.info_tree
        (xs,) = tree.findItems("Embedded XS", Qt.MatchStartsWith | Qt.MatchRecursive, 0)
        height = tree.viewport().height()
        assert tree.visualItemRect(xs).top() >= height, "nothing to scroll to"
        bar = tree.verticalScrollBar()
        bar.setValue(bar.maximum())
        QApplication.processEvents()
        rect = tree.visualItemRect(xs)
        assert rect.height() > 0 and rect.top() >= 0 and rect.bottom() < height, (rect, height)
    finally:
        conftest.close_window(window)
