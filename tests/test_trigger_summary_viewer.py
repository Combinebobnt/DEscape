"""GH #388's Detail column in a real offscreen ViewerWindow, against
tests/fixtures/triggers_120x120.aoe2scenario: the sentences, the delegate's
link clicks, following a link on the map, and Settings > Appearance's
condition and effect details."""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
_SETUP, _ARMOUR, _REFERENCES = 0, 1, 2


def _gen():
    return conftest.load_verify_module("gen_trigger_fixture")


def _window(trigger: int):
    window = conftest.shown_window(1100, 760)
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_trigger(trigger)
    return window


def _details(panel) -> dict[tuple[str, int], str]:
    return {key: item.text(1) for key, item in panel._entry_items.items()}


def _last_line(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _centred(window, monkeypatch) -> list[tuple[int, int]]:
    centred: list[tuple[int, int]] = []
    monkeypatch.setattr(window.map_view, "center_on_tile", lambda x, y: centred.append((x, y)))
    return centred


# -- the sentences ------------------------------------------------------------


def test_the_detail_column_reads_sentences_with_the_raw_fields_in_the_tooltip() -> None:
    from descape.trigger_summary_delegate import SUMMARY_ROLE

    gen = _gen()
    window = _window(gen.REFERENCE_TRIGGER)
    try:
        panel = window.trigger_panel
        details = _details(panel)
        assert details[("condition", 0)] == "3 s"
        assert details[("effect", 0)] == '"fixture_var" set to 1'
        assert details[("effect", gen.TASK_OBJECT_EFFECT)] == "2 selected objects to P1 House #502"
        assert details[("effect", gen.PATROL_IDS_EFFECT)] == "P1 Archer #501"
        item = panel._entry_items[("effect", 0)]
        assert "quantity=1" in item.toolTip(1)
        assert item.data(1, SUMMARY_ROLE).text == item.text(1)
    finally:
        conftest.close_window(window)


def test_a_trigger_reference_names_the_trigger() -> None:
    window = _window(_REFERENCES)
    try:
        assert _details(window.trigger_panel)[("effect", 0)] == '"Fixture: setup"'
    finally:
        conftest.close_window(window)


def test_raw_view_and_full_style_re_render_live() -> None:
    from descape import settings
    from descape.trigger_summary_delegate import SUMMARY_ROLE

    window = _window(_REFERENCES)
    try:
        panel = window.trigger_panel
        settings.set_trigger_details({"sentence_style": "full"})
        panel.refresh_trigger_details()
        assert _details(panel)[("effect", 0)] == 'Turn on trigger "Fixture: setup"'
        settings.set_trigger_details({"mode": "raw"})
        panel.refresh_trigger_details()
        item = panel._entry_items[("effect", 0)]
        assert item.text(1) == "trigger_id=0"
        assert item.data(1, SUMMARY_ROLE) is None
    finally:
        conftest.close_window(window)


def test_a_player_name_push_re_renders_the_sentences() -> None:
    from descape import player_labels, settings

    settings.set_trigger_details({"player_text": "name"})
    window = _window(_ARMOUR)
    try:
        panel = window.trigger_panel
        assert _details(panel)[("effect", 1)] == 'P1 Archer "Hit points" set to 45'
        names = ("", "Britons", "", "", "", "", "", "", "")
        panel.refresh_player_labels(player_labels.labels_for(names), window.trigger_panel._player_colors, None)
        assert _details(panel)[("effect", 1)] == 'Britons Archer "Hit points" set to 45'
    finally:
        conftest.close_window(window)


# -- the delegate -------------------------------------------------------------


def _link_point(tree, item, column: int, prefix: str):
    """A viewport point a few pixels into the text after `prefix`."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QFontMetrics

    x = tree.header().sectionViewportPosition(column) + 4 + QFontMetrics(tree.font()).horizontalAdvance(prefix) + 3
    return QPoint(x, tree.visualItemRect(item).center().y())


def test_a_click_on_a_link_emits_it_and_a_click_elsewhere_selects() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QAbstractItemView, QApplication, QTreeWidget, QTreeWidgetItem

    from descape.trigger_summary import Segment, Summary
    from descape.trigger_summary_delegate import SUMMARY_ROLE, TriggerSummaryDelegate

    conftest.ensure_qapp()
    tree = QTreeWidget()
    tree.setHeaderLabels(["Item", "Detail"])
    tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
    delegate = TriggerSummaryDelegate(tree)
    delegate.install(tree, 1)
    emitted = []
    delegate.link_activated.connect(lambda index, link: emitted.append(link))
    rows = []
    for name in ("first", "second", "third"):
        summary = Summary((Segment("go to "), Segment("21,215 24,218", ("area", 21, 215, 24, 218))), "raw")
        item = QTreeWidgetItem([name, summary.text])
        item.setData(1, SUMMARY_ROLE, summary)
        tree.addTopLevelItem(item)
        rows.append(item)
    tree.resize(600, 200)
    tree.setColumnWidth(0, 120)
    tree.show()
    QTest.qWaitForWindowExposed(tree)
    try:
        tree.setCurrentItem(rows[0])
        rows[1].setSelected(True)
        QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, _link_point(tree, rows[1], 1, "go to "))
        assert emitted == [], "emitted after the click returns"
        QApplication.processEvents()
        assert emitted == [("area", 21, 215, 24, 218)]
        assert tree.currentItem() is rows[0], "a link click leaves the selection alone"
        assert rows[0].isSelected() and rows[1].isSelected(), "a multi-selection stays whole"
        # A real double-click: press and release, then the double-click and its release.
        link = _link_point(tree, rows[2], 1, "go to ")
        QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, link)
        QTest.mouseDClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, link)
        QTest.mouseRelease(tree.viewport(), Qt.LeftButton, Qt.NoModifier, link)
        QApplication.processEvents()
        assert len(emitted) == 2, "a double-click follows once"
        assert tree.currentItem() is rows[0] and not rows[2].isSelected()
        QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, _link_point(tree, rows[1], 1, ""))
        QApplication.processEvents()
        assert len(emitted) == 2
        assert tree.currentItem() is rows[1]
        QTest.mouseMove(tree.viewport(), _link_point(tree, rows[0], 1, "go to "))
        assert tree.viewport().cursor().shape() == Qt.PointingHandCursor
        QTest.mouseMove(tree.viewport(), _link_point(tree, rows[0], 1, ""))
        assert tree.viewport().cursor().shape() != Qt.PointingHandCursor
    finally:
        tree.close()


# -- following a link ---------------------------------------------------------


def test_coordinate_and_object_links_centre_the_map(monkeypatch) -> None:
    gen = _gen()
    window = _window(gen.REFERENCE_TRIGGER)
    try:
        centred = _centred(window, monkeypatch)
        x1, y1, x2, y2 = gen.GEOMETRY_PATROL_AREA
        window.follow_trigger_link((gen.REFERENCE_TRIGGER, "effect", 1, ("area", x1, y1, x2, y2)))
        window.follow_trigger_link((gen.REFERENCE_TRIGGER, "effect", 1, ("tile", 60, 50)))
        window.follow_trigger_link((gen.REFERENCE_TRIGGER, "effect", 5, ("unit", gen.REF_HOUSE)))
        window.follow_trigger_link((gen.REFERENCE_TRIGGER, "effect", 5, ("units", gen.TASK_SELECTED_IDS)))
        assert centred == [((x1 + x2) // 2, (y1 + y2) // 2), (60, 50), gen.HOUSE_TILE, (51, 56)]
        assert _last_line(window) == "Centred on the selected objects (51, 56)"
        window.follow_trigger_link((gen.REFERENCE_TRIGGER, "effect", 5, ("unit", gen.DANGLING_REFERENCE_ID)))
        assert _last_line(window) == "Go to: the object is not on the map"
    finally:
        conftest.close_window(window)


def test_an_object_filter_link_steps_through_its_matches(monkeypatch) -> None:
    from descape.trigger_summary import LINK_MATCH, ObjectFilter

    gen = _gen()
    window = _window(gen.REFERENCE_TRIGGER)
    try:
        centred = _centred(window, monkeypatch)
        archers = ObjectFilter(player=1, unit_const=4, label="P1 Archer")
        for _ in range(3):
            window.follow_trigger_link((gen.REFERENCE_TRIGGER, "effect", 2, (LINK_MATCH, archers)))
        assert centred == [(50, 55), (52, 57), (50, 55)]
        assert _last_line(window) == "P1 Archer 1 of 2 (50, 55)"
        window.follow_trigger_link((gen.REFERENCE_TRIGGER, "effect", 2, (LINK_MATCH, ObjectFilter(player=2, label="P2 objects"))))
        assert _last_line(window) == "No P2 objects on the map"
        assert len(centred) == 3
    finally:
        conftest.close_window(window)


def test_a_trigger_link_selects_that_trigger() -> None:
    window = _window(_REFERENCES)
    try:
        panel = window.trigger_panel
        setup_id = panel._manager().triggers[_SETUP].trigger_id
        window.follow_trigger_link((_REFERENCES, "effect", 0, ("trigger", setup_id)))
        assert panel.current_trigger_index() == _SETUP
        window.follow_trigger_link((_SETUP, "effect", 0, ("trigger", 99)))
        assert _last_line(window) == "Trigger link: no trigger has id 99"
        panel.select_trigger(_REFERENCES)
        panel.filter_edit.setText("references")
        panel._apply_filter()
        window.follow_trigger_link((_REFERENCES, "effect", 0, ("trigger", setup_id)))
        assert _last_line(window) == "Trigger link: that trigger is hidden by the filter"
        assert panel.current_trigger_index() == _REFERENCES
    finally:
        conftest.close_window(window)


def test_a_clicked_trigger_link_rebuilds_the_rows_safely() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _window(_REFERENCES)
    try:
        panel = window.trigger_panel
        tree = panel.entry_tree
        item = panel._entry_items[("effect", 0)]
        tree.scrollToItem(item)
        QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, _link_point(tree, item, 1, ""))
        QApplication.processEvents()
        assert panel.current_trigger_index() == _SETUP
    finally:
        conftest.close_window(window)


def test_a_rename_shows_in_the_rows_that_name_the_trigger() -> None:
    window = _window(_REFERENCES)
    try:
        panel = window.trigger_panel
        panel.select_entry("effect", 0)
        panel._manager().triggers[_SETUP].name = "Renamed"
        panel._refresh_labels(_REFERENCES)
        assert _details(panel)[("effect", 0)] == '"Renamed"'
    finally:
        conftest.close_window(window)


def test_a_unit_edit_re_renders_only_in_triggers_mode() -> None:
    from types import SimpleNamespace

    gen = _gen()
    window = _window(gen.REFERENCE_TRIGGER)
    try:
        panel = window.trigger_panel
        moved = SimpleNamespace(player_id=2, unit_const=70, bounds=(0, 0, 1, 1), own_tile=(1, 1))
        panel.unit_reference = lambda ref_id: moved
        panel.refresh_unit_reference_labels(recompute_status=False)
        assert _details(panel)[("effect", gen.PATROL_IDS_EFFECT)] == "P1 Archer #501"
        panel.refresh_unit_reference_labels(recompute_status=True)
        assert _details(panel)[("effect", gen.PATROL_IDS_EFFECT)] == "P2 House #501"
    finally:
        conftest.close_window(window)


def test_the_panel_hands_a_clicked_link_to_the_window_with_its_entry() -> None:
    window = _window(_REFERENCES)
    try:
        panel = window.trigger_panel
        followed = []
        panel.on_summary_link = followed.append
        item = panel._entry_items[("effect", 1)]
        panel._on_summary_link(panel.entry_tree.indexFromItem(item, 1), ("trigger", 1))
        assert followed == [(_REFERENCES, "effect", 1, ("trigger", 1))]
    finally:
        conftest.close_window(window)


# -- Settings > Appearance ----------------------------------------------------


def test_the_settings_controls_store_render_the_preview_and_reset() -> None:
    from descape import settings

    dialog, window = conftest.dialog_and_window()
    try:
        page = dialog.trigger_details_page
        assert page.preview.topLevelItem(1).text(1) == "P2 to P1, visible"
        assert [page.preview.topLevelItem(i).text(1) for i in (4, 5, 6, 7)] == [
            '"Wave 2"', 'P1 "Defend the castle until the reinforceme…", 10 s', "P7 Castle #500", "P1 Archer in 10,20 15,25",
        ]
        page.style_combo.setCurrentIndex(page.style_combo.findData("full"))
        page.colour_combo.setCurrentIndex(page.colour_combo.findData("text"))
        page.cap_spin.setValue(0)
        stored = settings.get_trigger_details()
        assert (stored["sentence_style"], stored["player_colour"], stored["text_cap"]) == ("full", "text", 0)
        assert page.preview.topLevelItem(1).text(1) == "P2 sees P1 as visible"
        page.reset_button.click()
        assert settings.get_trigger_details() == settings.TRIGGER_DETAILS_DEFAULTS
        assert page.style_combo.currentData() == "short"
        assert page.colour_combo.currentData() == "box"
    finally:
        dialog.close()
        conftest.close_window(window)


def test_opening_and_closing_settings_writes_nothing() -> None:
    from descape import settings

    dialog, window = conftest.dialog_and_window()
    dialog.close()
    conftest.close_window(window)
    assert "trigger_details" not in settings._load_config()


def test_a_malformed_stored_option_reads_as_its_default() -> None:
    from descape import settings

    settings._save_config({"trigger_details": {"sentence_style": "loud", "text_cap": -3, "show_ids": True, "x": 1}})
    settings._trigger_details = None
    stored = settings.get_trigger_details()
    assert stored == {**settings.TRIGGER_DETAILS_DEFAULTS, "show_ids": True}
    with pytest.raises(ValueError):
        settings.set_trigger_details({"player_text": "loud"})
    with pytest.raises(ValueError):
        settings.set_trigger_details({"no_such_option": True})


def test_the_delegate_colours_the_box_the_text_or_both() -> None:
    from descape.trigger_summary import Segment, Summary
    from descape.trigger_summary_delegate import TriggerSummaryDelegate

    conftest.ensure_qapp()
    delegate = TriggerSummaryDelegate()
    delegate.player_colors = ((0, 0, 0), (10, 20, 30))
    for mark, box, text in (("box", True, False), ("text", False, True), ("both", True, True)):
        html = delegate._html(Summary((Segment("P1", None, 1, 1, mark),), ""), "#ffffff")
        assert ('rgb(10,20,30)">&#9632;' in html) is box, mark
        assert ('">P1</span>' in html) is text, mark


def test_coloured_text_is_shaded_to_read_on_its_background() -> None:
    import re

    from descape import themes
    from descape.trigger_summary import Segment, Summary
    from descape.trigger_summary_delegate import TriggerSummaryDelegate

    conftest.ensure_qapp()
    delegate = TriggerSummaryDelegate()
    delegate.player_colors = ((0, 0, 0), (255, 255, 0))
    html = delegate._html(Summary((Segment("P1", None, 1, 1, "both"), Segment("hi", rgb=(0, 255, 225))), ""), "#f0e0c0")
    texts = re.findall(r'color:(#[0-9a-f]{6})">(?:P1|hi)<', html)
    assert len(texts) == 2
    assert all(themes.contrast_ratio(colour, "#f0e0c0") >= 4.5 for colour in texts), texts
    assert 'rgb(255,255,0)">&#9632;' in html, "the box keeps the true colour"
