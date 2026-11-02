"""TASK-550: the Display Instructions box drawn over the map.

Geometry comes through panel_scale() and panel_top(), and the art through the
module's own constants, so a corrected slot or colour moves no test here; the
corpus tests compare those constants with the install. The window cases reuse
test_trigger_panel's offscreen fixture, whose trigger 0 effect 0 is one Display
Instructions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

if conftest.PYQT5_AVAILABLE:
    from test_trigger_panel import (
        TRIGGER_FIXTURE,
        _instruction_window,
        _plant_prose_effect,
        _record_count,
        _row_widget,
        _select_armour_split_effect,
        _select_entries,
    )

BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"
# Read here: the autouse fixture below hides it from every test, the corpus ones too.
_INSTALL = os.environ.get("AOE2DE_INSTALL_PATH")


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _overlay(window):
    return window.map_view.instruction_overlay()


def _shown(window) -> bool:
    return not _overlay(window).isHidden()


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _settle() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()


# -- the layout, by hand ----------------------------------------------------------


@pytest.mark.parametrize("slot", [0, 1, 2])
def test_each_slot_keeps_its_share_of_the_height(slot: int) -> None:
    from descape.instruction_overlay import CANVAS_HEIGHT, PANEL_TOP, SLOT_OFFSETS, panel_top

    share = (PANEL_TOP + SLOT_OFFSETS[slot]) / CANVAS_HEIGHT
    for height in (200, 720, 1080, 2160):
        assert panel_top(height, slot) == pytest.approx(height * share, abs=0.5)


def test_scale_follows_the_height_down_to_its_floor_and_up_to_the_width() -> None:
    from descape.instruction_overlay import CANVAS_HEIGHT, MIN_SCALE, PANEL_WIDTH, WIDTH_MARGIN_PX, panel_scale

    assert panel_scale(4000, CANVAS_HEIGHT) == 1.0
    assert panel_scale(4000, CANVAS_HEIGHT // 2) == 0.5
    assert panel_scale(4000, 10) == MIN_SCALE, "the floor"
    narrow = 400
    assert panel_scale(narrow, CANVAS_HEIGHT) * PANEL_WIDTH == pytest.approx(narrow - 2 * WIDTH_MARGIN_PX), "the width cap"
    assert panel_scale(0, 0) > 0, "an unlaid-out view must not paint at scale zero"


# -- when the box shows -----------------------------------------------------------


def test_the_box_shows_centred_for_one_display_instructions_effect() -> None:
    from descape.instruction_overlay import panel_top

    window = _instruction_window()
    try:
        view = window.map_view.viewport().geometry()
        box = _overlay(window).geometry()
        assert _shown(window)
        assert "Fixture scenario loaded." in _overlay(window).view().html
        assert abs(box.center().x() - view.center().x()) <= 1
        assert box.top() == view.y() + panel_top(view.height(), 0)
        assert box.right() < view.right()
    finally:
        _close(window)


def test_another_effect_a_trigger_row_and_a_multi_selection_hide_the_box() -> None:
    window = _instruction_window()
    try:
        panel = window.trigger_panel
        _select_armour_split_effect(panel, 0)
        _settle()
        assert not _shown(window), "a Modify Attribute effect"

        added = _plant_prose_effect(window, "second")
        _settle()
        assert _shown(window)
        _select_entries(panel, ("effect", 0), ("effect", added))
        _settle()
        assert not _shown(window), "two Display Instructions effects"

        panel.refresh_entries(select=("effect", 0))
        _settle()
        assert _shown(window)
        panel.select_trigger(1)
        _settle()
        assert not _shown(window), "another trigger's row"
    finally:
        _close(window)


def test_leaving_triggers_mode_hides_the_box_and_coming_back_returns_it() -> None:
    window = _instruction_window()
    try:
        for mode in ("View", "Terrain", "Units", "Map Options"):
            window.mode_combo.setCurrentText(mode)
            _settle()
            assert not _shown(window), mode
            window.mode_combo.setCurrentText("Triggers")
            _settle()
            assert _shown(window), f"back from {mode}"
    finally:
        _close(window)


def test_a_forced_mode_hides_the_box_and_coming_back_returns_it() -> None:
    window = _instruction_window()
    try:
        assert _shown(window)
        window._force_mode("View")
        _settle()
        assert window.mode == "view"
        assert not _shown(window)
        window.mode_combo.setCurrentText("Triggers")
        _settle()
        assert _shown(window)
    finally:
        _close(window)


def test_the_toggle_hides_the_box_and_persists() -> None:
    from descape import settings

    window = _instruction_window()
    try:
        action = window.instruction_overlay_action
        assert action.isChecked(), "on by default"
        action.setChecked(False)
        assert not _shown(window)
        assert settings.get_instruction_overlay() is False
        settings._instruction_overlay = None
        assert settings.get_instruction_overlay() is False, "read back from the file, not the memo"

        window.mode_combo.setCurrentText("View")
        window.mode_combo.setCurrentText("Triggers")
        _settle()
        assert not _shown(window), "a mode round trip leaves it off"

        action.setChecked(True)
        assert _shown(window)
    finally:
        _close(window)


def test_the_toggle_starts_from_the_setting() -> None:
    from descape import settings

    settings.set_instruction_overlay(False)
    window = _instruction_window()
    try:
        assert not window.instruction_overlay_action.isChecked()
        assert not _shown(window)
    finally:
        _close(window)


def test_turning_the_toggle_on_outside_triggers_mode_shows_nothing() -> None:
    window = _instruction_window()
    try:
        window.instruction_overlay_action.setChecked(False)
        window.mode_combo.setCurrentText("View")
        window.instruction_overlay_action.setChecked(True)
        assert not _shown(window)
    finally:
        _close(window)


def test_closing_the_document_hides_the_box_from_any_mode() -> None:
    for mode in ("Triggers", "View"):
        window = _instruction_window()
        try:
            window.mode_combo.setCurrentText(mode)
            window.edit_history.mark_saved()
            window.close_scenario()
            _settle()
            assert not _shown(window), mode
            assert _overlay(window).view() is None
            window.mode_combo.setCurrentText("Triggers")
            _settle()
            assert not _shown(window), f"{mode}: nothing comes back with no document"
        finally:
            _close(window)


def test_no_stale_box_after_opening_another_file_from_a_different_mode() -> None:
    window = _instruction_window()
    try:
        window.mode_combo.setCurrentText("View")
        window.edit_history.mark_saved()
        window.load_scenario(BLANK_FIXTURE)
        _settle()
        assert not _shown(window)
        window.mode_combo.setCurrentText("Triggers")
        _settle()
        assert not _shown(window), "the old document's box came back"
        assert _overlay(window).view() is None
    finally:
        _close(window)


def test_opening_another_file_in_triggers_mode_drops_the_box() -> None:
    window = _instruction_window()
    try:
        assert _shown(window)
        window.edit_history.mark_saved()
        window.load_scenario(TRIGGER_FIXTURE)
        _settle()
        assert not _shown(window), "nothing is selected in the new document"
    finally:
        _close(window)


# -- following the form -----------------------------------------------------------


def test_changing_the_position_moves_the_box() -> None:
    from descape.instruction_overlay import panel_top

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        tops = {}
        for slot in (2, 1, 0):
            combo = _row_widget(panel, "instruction_panel_position")
            combo.setCurrentIndex(combo.findData(slot))
            _settle()
            view = window.map_view.viewport().geometry()
            tops[slot] = _overlay(window).geometry().top()
            assert tops[slot] == view.y() + panel_top(view.height(), slot)
        assert tops[0] < tops[1] < tops[2]
    finally:
        _close(window)


def test_typing_changes_the_box_with_no_undo_record_and_no_entry_change() -> None:
    window = _instruction_window()
    try:
        panel = window.trigger_panel
        entry = panel._manager().triggers[0].effects[0]
        _row_widget(panel, "message").setPlainText("<BLUE>changed")
        assert "changed" in _overlay(window).view().html
        assert entry.message == "Fixture scenario loaded."
        assert _record_count(window) == 0
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


@pytest.mark.font_sensitive
def test_the_box_grows_with_its_text_up_to_the_surround() -> None:
    from descape.instruction_overlay import PANEL_MAX_HEIGHT, PANEL_MIN_HEIGHT, PANEL_WIDTH, panel_scale

    window = _instruction_window()
    try:
        panel = window.trigger_panel
        overlay = _overlay(window)
        area = window.map_view.viewport().geometry()
        scale = panel_scale(area.width(), area.height())
        assert overlay.height() == round(PANEL_MIN_HEIGHT * scale), "a short message takes one slot"
        _row_widget(panel, "message").setPlainText("A long line of narration. " * 8)
        grown = overlay.height()
        assert grown > round(PANEL_MIN_HEIGHT * scale)
        _row_widget(panel, "message").setPlainText("A long line of narration. " * 300)
        assert overlay.height() == round(PANEL_MAX_HEIGHT * scale), "it stops at the Surround"
        assert overlay.width() == round(PANEL_WIDTH * scale)
    finally:
        _close(window)


# -- fixed to the viewport --------------------------------------------------------


def test_pan_and_zoom_leave_the_box_where_it_is() -> None:
    """Geometry from a real scroll: a child of viewport() fails here, because
    QWidget.scroll() carries the viewport's children with the map."""
    window = _instruction_window()
    try:
        map_view = window.map_view
        map_view.scale(8, 8)
        _settle()
        bars = map_view.horizontalScrollBar(), map_view.verticalScrollBar()
        assert all(bar.maximum() > 40 for bar in bars), "fixture assumption: the map scrolls at this zoom"
        before = _overlay(window).geometry()
        for _step in range(5):
            for bar in bars:
                bar.setValue(bar.value() + 20)
            _settle()
            assert _overlay(window).geometry() == before, "a pan moved the box"
        map_view.scale(1.5, 1.5)
        _settle()
        assert _overlay(window).geometry() == before, "a zoom moved the box"
    finally:
        _close(window)


def test_a_resize_replaces_the_box() -> None:
    from descape.instruction_overlay import panel_top

    window = _instruction_window()
    try:
        map_view = window.map_view
        before = _overlay(window).geometry()
        window.resize(window.width() + 300, window.height() + 200)
        _settle()
        area = map_view.viewport().geometry()
        box = _overlay(window).geometry()
        assert box != before
        assert abs(box.center().x() - area.center().x()) <= 1
        assert box.top() == area.y() + panel_top(area.height(), 0)
    finally:
        _close(window)


def test_scrollbars_coming_and_going_keep_the_box_centred() -> None:
    """They resize the viewport with no resize of the view itself."""
    window = _instruction_window()
    try:
        map_view = window.map_view
        bar = map_view.verticalScrollBar()

        def assert_centred() -> None:
            area = map_view.viewport().geometry()
            assert abs(_overlay(window).geometry().center().x() - area.center().x()) <= 1, (area, bar.isVisible())

        map_view.setSceneRect(map_view.mapToScene(map_view.viewport().rect().adjusted(40, 40, -40, -40)).boundingRect())
        _settle()
        assert not bar.isVisible(), "fixture assumption: this scene rect fits, so the bars go"
        without = map_view.viewport().width()
        assert_centred()

        map_view.scale(8, 8)
        _settle()
        assert bar.isVisible()
        assert map_view.viewport().width() < without, "fixture assumption: the bars took the width"
        assert_centred()
    finally:
        _close(window)


def test_the_mouse_goes_through_the_box_to_the_viewport() -> None:
    from PyQt5.QtCore import Qt

    window = _instruction_window()
    try:
        map_view = window.map_view
        overlay = _overlay(window)
        assert map_view.childAt(overlay.geometry().center()) is map_view.viewport()
        assert overlay.focusPolicy() == Qt.NoFocus
    finally:
        _close(window)


def _pixel(image, x: int, y: int) -> tuple[int, int, int]:
    color = image.pixelColor(x, y)
    return color.red(), color.green(), color.blue()


def test_the_panel_fill_composites_over_the_map() -> None:
    from descape.instruction_overlay import ART_ALPHA, ART_FILL, PANEL_WIDTH, TEXT_RECT

    window = _instruction_window()
    try:
        map_view = window.map_view
        overlay = _overlay(window)
        box = overlay.geometry()
        # Inside the fill: past the text column's start, below a one-line message.
        x = box.left() + round(box.width() * (TEXT_RECT[0] + TEXT_RECT[2] / 2) / PANEL_WIDTH)
        y = box.bottom() - 6
        under = _pixel(map_view.viewport().grab().toImage(), x - map_view.viewport().x(), y - map_view.viewport().y())
        drawn = _pixel(map_view.grab().toImage(), x, y)
        alpha = ART_ALPHA / 255
        expected = [round(fill * alpha + below * (1 - alpha)) for fill, below in zip(ART_FILL, under, strict=True)]
        assert drawn == pytest.approx(expected, abs=2), "the fill over the pixel the map drew"
    finally:
        _close(window)


def test_a_scene_capture_leaves_the_box_out() -> None:
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QColor, QImage, QPainter

    window = _instruction_window()

    def capture() -> QImage:
        scene = window.map_view.scene()
        image = QImage(400, 400, QImage.Format_RGB32)
        image.fill(QColor(255, 0, 255))
        painter = QPainter(image)
        scene.render(painter, QRectF(image.rect()), scene.sceneRect())
        painter.end()
        return image

    try:
        assert _shown(window)
        with_the_box = capture()
        window.map_view.show_instruction(None)
        assert not _shown(window)
        assert with_the_box == capture(), "the box is in the capture"
    finally:
        _close(window)


# -- against the install ----------------------------------------------------------


def _install() -> Path:
    if not _INSTALL or not Path(_INSTALL).is_dir():
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH")
    return Path(_INSTALL)


@pytest.mark.corpus
def test_instruction_constants_equal_the_panel_file() -> None:
    """If H1 corrects SLOT_OFFSETS, the one slot equality below goes, with a comment naming H1."""
    from descape import instruction_overlay as io

    data = json.loads((_install() / "widgetui" / "instructionspanel.json").read_text(encoding="utf-8"))
    collection = data["Collection"]
    assert collection["ViewPort"]["yorigin"] == io.PANEL_TOP
    assert collection["ViewPort"]["width"] == io.PANEL_WIDTH
    widgets = [entry["Widget"] for entry in collection["Widgets"]]
    assert [w["Name"] for w in widgets] == ["EventBackground1", "EventBackground2", "EventBackground3"]
    assert tuple(w["ViewPort"]["yorigin"] for w in widgets) == io.SLOT_OFFSETS
    for surround in widgets:
        assert surround["Box"]["width"] == io.PANEL_WIDTH
        assert surround["Box"]["height"] == io.PANEL_MAX_HEIGHT
        instruction = surround["ChildWidgets"][0]["Widget"]
        image = instruction["Image"]
        assert (image["xorigin"], image["yorigin"], image["width"], image["height"]) == io.ICON_RECT
        text = instruction["ChildWidgets"][0]["Widget"]
        box = text["TextBox"]
        assert (box["xorigin"], box["yorigin"], box["width"], box["height"]) == io.TEXT_RECT
        assert box["linesize"] == io.FONT_PX
        assert text["StateMaterials"]["StateTextNormal"]["Font"]["PointSize"] == io.FONT_PX


@pytest.mark.corpus
def test_instruction_fill_equals_the_panel_art() -> None:
    from PIL import Image

    from descape import instruction_overlay as io

    art = _install() / "widgetui" / "textures" / "ingame" / "panels" / "npanel_c_4k.png"
    with Image.open(art) as source:
        image = source.convert("RGBA")
    assert image.getpixel((image.width // 2, image.height // 2)) == (*io.ART_FILL, io.ART_ALPHA)
