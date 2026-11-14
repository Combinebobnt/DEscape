"""Trigger panel: the Display Instructions preview under the form."""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _SETUP_TRIGGER,
    _activate,
    _assert_form_rows_do_not_overlap,
    _instruction_window,
    _key,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _plant_prose_effect,
    _record_count,
    _release_modifiers,
    _row_widget,
    _select_armour_split_effect,
    _select_entries,
    _triggers_window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- GH #140: the Display Instructions preview ----------------------------------


def _preview(panel):
    preview = panel._instruction_preview
    assert preview is not None, "no preview for a single Display Instructions effect"
    return preview


def _live_previews(panel) -> list:
    from descape.instruction_preview import InstructionPreview

    return panel.property_host.findChildren(InstructionPreview)


def test_a_single_display_instructions_effect_gets_a_preview_under_its_form() -> None:
    from PyQt5.QtWidgets import QFormLayout

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        preview = _preview(panel)
        assert _live_previews(panel) == [preview]
        form = panel.property_form
        last = form.itemAt(form.rowCount() - 1, QFormLayout.SpanningRole)
        assert last is not None and last.widget() is preview, "a full-width row below the fields"
        assert "Fixture scenario loaded." in preview.view.html
        assert preview.view.slot == 0
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_refresh_portraits_shows_the_preview_portrait_the_art_now_gives(monkeypatch) -> None:
    """GH #363: after a Mod Art toggle or a data mod switch (install generation moved) the open preview redraws."""
    import numpy as np

    from descape import asset_source

    shade = [100]

    def portrait(*_args):
        return np.full((8, 8, 4), (shade[0], 0, 0, 255), dtype=np.uint8)

    portrait.cache_clear = lambda: None
    monkeypatch.setattr(asset_source, "get_unit_icon", portrait)
    window = _instruction_window(object_list_unit_id=4)
    try:
        panel = window.trigger_panel
        preview = _preview(panel)
        assert preview.icon_source == "icon"
        shade[0] = 200
        panel.refresh_portraits()
        assert preview._icon_pixmap.toImage().pixelColor(4, 4).red() == 100, "same generation: the memo"
        asset_source.clear_art_caches()
        panel.refresh_portraits()
        assert preview._icon_pixmap.toImage().pixelColor(4, 4).red() == 200
    finally:
        window.edit_history.mark_saved()
        window.close()



def test_no_preview_for_a_trigger_a_condition_another_effect_or_a_multi_select() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.select_trigger(_SETUP_TRIGGER)
        QApplication.processEvents()
        assert panel._instruction_preview is None, "the trigger row"
        panel.refresh_entries(select=("condition", 0))
        QApplication.processEvents()
        assert panel._instruction_preview is None, "a condition"
        _select_armour_split_effect(panel, 0)
        QApplication.processEvents()
        assert panel._instruction_preview is None, "a Modify Attribute effect"
        assert _live_previews(panel) == []

        added = _plant_prose_effect(window, "second")
        assert panel._instruction_preview is not None
        _select_entries(panel, ("effect", 0), ("effect", added))
        assert len(panel._rows) > 0, "the two share a form"
        assert panel._instruction_preview is None, "two Display Instructions effects"
        assert _live_previews(panel) == []
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(("stored", "slot", "note"), [(0, 0, ""), (1, 1, ""), (2, 2, ""), (-1, 0, "unset, drawn as TOP")])
def test_the_position_picks_the_slot(stored: int, slot: int, note: str) -> None:
    window = _instruction_window(instruction_panel_position=stored)
    try:
        preview = _preview(window.trigger_panel)
        assert preview.view.slot == slot
        if note:
            assert note in preview.note_label.text()
        else:
            assert "unset" not in preview.note_label.text()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_typing_in_the_message_updates_the_preview_but_writes_nothing() -> None:
    """Observe-only: per keystroke the preview follows, and nothing reaches
    _changed() until the box's own focus-out commit, exactly as before."""
    from descape.message_markup import TEXT_COLORS

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        preview = _preview(panel)
        entry = panel._manager().triggers[_SETUP_TRIGGER].effects[0]
        _row_widget(panel, "message").setPlainText("<BLUE>Scout: <GREY>over here & <fixture_var>")
        html = preview.view.html
        assert "Scout: " in html and "over here &amp; " in html
        assert "color:#{:02x}{:02x}{:02x}".format(*TEXT_COLORS["BLUE"]) in html
        assert "color:#{:02x}{:02x}{:02x}".format(*TEXT_COLORS["GREY"]) in html
        assert "[fixture_var]" in html, "a trigger variable name is a placeholder"
        _row_widget(panel, "sound_name").setText("Play_Technology_Researched")
        assert "Sound: Play_Technology_Researched" in preview.footer_label.text()

        assert entry.message == "Fixture scenario loaded."
        assert entry.sound_name == ""
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
        assert _record_count(window) == 0
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_real_keystrokes_in_the_message_follow_in_the_preview_and_commit_once_on_focus_out() -> None:
    """GH #140's manual step by real key presses on the activated window.
    Passes when: after each of two typed characters the preview's text already
    ends with them; until the box loses focus nothing is recorded and the entry
    still holds the fixture's message; a real focus move to the filter box then
    records exactly one undo step and writes the typed message; and focusing
    and leaving the box again unchanged records nothing more."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        preview = _preview(panel)
        entry = panel._manager().triggers[_SETUP_TRIGGER].effects[0]
        box = _row_widget(panel, "message")
        _activate(window)
        box.setFocus()
        QApplication.processEvents()
        assert box.hasFocus()
        _key(box, Qt.Key_End, Qt.ControlModifier)

        QTest.keyClicks(box, "X")
        QApplication.processEvents()
        assert "Fixture scenario loaded.X" in preview.view.html
        QTest.keyClicks(box, "Y")
        QApplication.processEvents()
        assert "Fixture scenario loaded.XY" in preview.view.html
        assert box.toPlainText() == "Fixture scenario loaded.XY"
        assert _record_count(window) == 0, "a keystroke recorded an undo step"
        assert entry.message == "Fixture scenario loaded."
        assert not window.edit_history.is_dirty

        panel.filter_edit.setFocus()
        QApplication.processEvents()
        assert not box.hasFocus()
        assert _record_count(window) == 1, "the focus-out commits exactly one step"
        assert panel._manager().triggers[_SETUP_TRIGGER].effects[0].message == "Fixture scenario loaded.XY"

        box = _row_widget(panel, "message")
        box.setFocus()
        QApplication.processEvents()
        panel.filter_edit.setFocus()
        QApplication.processEvents()
        assert _record_count(window) == 1, "an unchanged focus-out recorded again"
    finally:
        _release_modifiers(window)
        window.edit_history.mark_saved()
        window.close()


def test_a_field_edit_records_only_its_own_undo_step_and_the_preview_follows() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        combo = _row_widget(panel, "instruction_panel_position")
        combo.setCurrentIndex(combo.findData(2))
        QApplication.processEvents()
        assert _record_count(window) == 1, "the position edit, and nothing from the preview"
        assert _preview(panel).view.slot == 2

        _row_widget(panel, "play_sound").setChecked(True)
        QApplication.processEvents()
        assert _record_count(window) == 2
        assert "(plays)" in _preview(panel).footer_label.text()
        assert _live_previews(panel) == [panel._instruction_preview]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_reselecting_leaves_no_stale_preview() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        first = _preview(panel)
        _select_armour_split_effect(panel, 0)
        QApplication.processEvents()
        assert panel._instruction_preview is None, "the reference outlived its form"
        assert _live_previews(panel) == []
        assert first.parent() is None, "the old preview is unparented with its row"

        panel.select_trigger(_SETUP_TRIGGER)
        panel.refresh_entries(select=("effect", 0))
        QApplication.processEvents()
        assert _live_previews(panel) == [panel._instruction_preview]
        assert panel._instruction_preview is not first
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    ("play_sound", "words"), [(-1, "(default)"), (0, "(off)"), (1, "(plays)")], ids=["default", "off", "plays"]
)
def test_the_footer_reads_play_sound_from_the_entry(play_sound: int, words: str) -> None:
    """The checkbox shows -1 and 1 alike as checked, so the entry decides."""
    window = _instruction_window(play_sound=play_sound, sound_name="Play_66433", display_time=7)
    try:
        footer = _preview(window.trigger_panel).footer_label.text()
        assert "Sound: Play_66433" in footer and words in footer
        assert "Shows for 7 s" in footer
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(("display_time", "words"), [(0, "Shows for 0 s"), (-1, "Shows for (unset)")])
def test_the_footer_shows_zero_and_unset_display_times(display_time: int, words: str) -> None:
    window = _instruction_window(display_time=display_time)
    try:
        footer = _preview(window.trigger_panel).footer_label.text()
        assert words in footer
        assert "Sound: none" in footer
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_with_no_install_a_string_id_is_not_available_and_the_icon_falls_back() -> None:
    from descape import asset_source

    assert asset_source.get_install_path() is None
    window = _instruction_window(string_id=60014, object_list_unit_id=448)
    try:
        preview = _preview(window.trigger_panel)
        assert "language string #60014 (not available)" in preview.note_label.text()
        assert preview.icon_source in ("sprite", None)
        assert (preview.view.icon is None) == (preview.icon_source is None)
    finally:
        window.edit_history.mark_saved()
        window.close()


def _preview_install(tmp_path: Path) -> Path:
    """A fake install: one string, one unit icon (Scout Cavalry's 64) and a Blue override."""
    import json

    from PIL import Image

    install = tmp_path / "install"
    strings = install / "resources" / "en" / "strings" / "key-value"
    strings.mkdir(parents=True)
    (strings / "key-value-strings-utf8.txt").write_text('60014 "<BLUE>From the table"\n', encoding="utf-8")
    units = install / "widgetui" / "textures" / "ingame" / "units"
    units.mkdir(parents=True)
    Image.new("RGBA", (16, 16), (200, 10, 10, 255)).save(units / "064_50730.DDS", format="DDS")
    colors = {"Blue": {"Text": [1, 2, 3, 255]}}
    (install / "widgetui" / "UIColors.json").write_text(json.dumps({"ColorTables": colors}))
    return install


def test_an_install_supplies_the_string_the_icon_and_the_colours(tmp_path: Path) -> None:
    from descape import asset_source

    asset_source.set_install_path_override(_preview_install(tmp_path))
    try:
        window = _instruction_window(string_id=60014, object_list_unit_id=448)
        try:
            preview = _preview(window.trigger_panel)
            assert "From the table" in preview.view.html, "the string replaces the message"
            assert "Fixture scenario loaded." not in preview.view.html
            assert "color:#010203" in preview.view.html, "the install's Blue"
            assert "language string #60014" in preview.note_label.text()
            assert "not available" not in preview.note_label.text()
            assert preview.icon_source == "icon"
            assert not preview.view.icon.isNull()
        finally:
            window.edit_history.mark_saved()
            window.close()
    finally:
        asset_source.set_install_path_override(None)


def test_a_building_icon_comes_from_the_building_portraits(tmp_path: Path) -> None:
    """GH #148: Castle (82, icon 7) shows buildings/007_castle, not unit portrait 007."""
    from PIL import Image

    from descape import asset_source

    install = _preview_install(tmp_path)
    ingame = install / "widgetui" / "textures" / "ingame"
    (ingame / "buildings").mkdir()
    Image.new("RGBA", (16, 16), (200, 10, 10, 255)).save(ingame / "units" / "007_50730.DDS", format="DDS")
    Image.new("RGBA", (16, 16), (10, 200, 10, 255)).save(ingame / "buildings" / "007_castle.DDS", format="DDS")
    asset_source.set_install_path_override(install)
    try:
        window = _instruction_window(object_list_unit_id=82)
        try:
            preview = _preview(window.trigger_panel)
            assert preview.icon_source == "icon"
            image = preview.view.icon.toImage()
            centre = image.pixelColor(image.width() // 2, image.height() // 2)
            assert (centre.red(), centre.green(), centre.blue()) == (10, 200, 10)
        finally:
            window.edit_history.mark_saved()
            window.close()
    finally:
        asset_source.set_install_path_override(None)


def test_the_icon_frame_takes_the_source_players_colour_or_the_leading_tag_colour() -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.message_markup import TEXT_COLORS

    window = _instruction_window(message="<BLUE>tagged", use_tag_color_for_icon=-1)
    try:
        panel = window.trigger_panel
        colors = window.scenario.player_colors
        assert colors is not None
        assert _preview(panel).view.tint == tuple(colors[1]), "source player ONE's colour"

        panel.refresh_player_labels(panel._player_labels, None)
        QApplication.processEvents()
        assert _preview(panel).view.tint is None, "no colours, no tint"

        entry = panel._manager().triggers[_SETUP_TRIGGER].effects[0]
        entry.use_tag_color_for_icon = 1
        panel.refresh_player_labels(panel._player_labels, colors)
        panel.refresh_entries(select=("effect", 0))
        QApplication.processEvents()
        assert _preview(panel).view.tint == TEXT_COLORS["BLUE"]
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.font_sensitive
def test_a_long_footer_grows_the_preview_without_overlap_or_widening() -> None:
    """A long unbreakable word cannot widen the pane, and typing re-fits the form."""
    from PyQt5.QtWidgets import QApplication

    from descape.trigger_panel import TriggerPanel

    def check(panel) -> None:
        QApplication.processEvents()
        preview = _preview(panel)
        _assert_form_rows_do_not_overlap(panel.property_form)
        assert preview.minimumSizeHint().width() <= TriggerPanel.MIN_USEFUL_WIDTH
        assert not panel.property_area.horizontalScrollBar().isVisible()
        assert panel.property_host.height() >= panel.property_form.heightForWidth(panel.property_host.width())

    window = _instruction_window(sound_name="W" * 120)
    try:
        panel = window.trigger_panel
        check(panel)
        before = _preview(panel).height()
        _row_widget(panel, "sound_name").setText("A long line of narration. " * 30)
        check(panel)
        assert _preview(panel).height() > before, "the rows grew with their text"
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_preview_takes_no_focus() -> None:
    """A focusable preview child would pull focus off the message box and
    fire its focus-out commit."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QWidget

    window = _instruction_window()
    try:
        preview = _preview(window.trigger_panel)
        for widget in [preview, *preview.findChildren(QWidget)]:
            assert widget.focusPolicy() == Qt.NoFocus, widget
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("source_player, red", [(1, 200), (0, 100)], ids=["player", "gaia"])
def test_the_preview_portrait_follows_the_source_players_civ(monkeypatch, source_player, red) -> None:
    """GH #372: a freshly built form's preview takes the source player's civ, GAIA's for GAIA."""
    import numpy as np

    from descape import asset_source, civ_art, object_catalog

    def portrait(icon, _building):
        return np.full((8, 8, 4), (200 if icon == 999 else 100, 0, 0, 255), dtype=np.uint8)

    portrait.cache_clear = lambda: None
    real = object_catalog._dat_entry

    def entry(section, id_):
        row = real(section, id_)
        return {**row, "civ_icons": {"46": 999}} if section == "objects" and id_ == 4 else row

    monkeypatch.setattr(asset_source, "get_unit_icon", portrait)
    monkeypatch.setattr(object_catalog, "_dat_entry", entry)
    monkeypatch.setattr(civ_art, "player_portrait_civ", lambda _scenario, pid: 46 if pid == 1 else None)
    window = _instruction_window(object_list_unit_id=4, source_player=source_player)
    try:
        preview = _preview(window.trigger_panel)
        assert preview._icon_pixmap.toImage().pixelColor(4, 4).red() == red
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_preview_portrait_fills_its_player_colour_region(monkeypatch) -> None:
    """GH #372: the source player's colour, the same one the icon frame takes."""
    import numpy as np

    from descape import asset_source

    def portrait(*_args):
        arr = np.full((8, 8, 4), (0, 0, 0, 255), dtype=np.uint8)
        arr[2:6, 2:6] = (255, 255, 255, 0)  # DE's transparent region, full-bright shading
        return arr

    portrait.cache_clear = lambda: None
    monkeypatch.setattr(asset_source, "get_unit_icon", portrait)
    window = _instruction_window(object_list_unit_id=4, source_player=2)
    try:
        view = _preview(window.trigger_panel).view
        image = view.icon.toImage()
        from descape import catalog_thumbs

        assert view.tint == tuple(window.scenario.player_colors[2][:3])
        expected = catalog_thumbs.tinted(np.array([[[255, 255, 255, 0]]], np.uint8), view.tint)[0, 0, :3].tolist()
        assert list(image.pixelColor(image.width() // 2, image.height() // 2).getRgb()[:3]) == expected
    finally:
        window.edit_history.mark_saved()
        window.close()
