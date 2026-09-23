"""GH #22: Players mode's Set / Go to / Reset View buttons and the View >
Player Cameras markers, driven through a real offscreen ViewerWindow.

Same technique as tests/test_unit_selection_viewer.py: assert on the
window's own state and on MapView's item references, never on widget
isVisible() in a never-shown window. The blank template stores an unset
(-1, -1) view for every player, so every marker here is one this file put
there through the real write path.

Every ViewerWindow() constructed here calls edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

from descape import player_fields
from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _window(markers: bool = True, mode: str = "Players"):
    window = conftest.shown_window(900, 700)
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.player_cameras_action.setChecked(markers)
    window.mode_combo.setCurrentText(mode)
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _marker_tiles(window) -> dict[int, tuple[int, int]]:
    return {item.player_id: item.tile() for item in window.map_view.camera_marker_items()}


# --- the three buttons ------------------------------------------------


def test_set_view_writes_the_viewport_centre_tile() -> None:
    window = _window()
    try:
        expected = window.map_view.viewport_centre_tile()
        assert expected is not None, "the blank template's centre should be on the map"
        window.set_player_view(1)
        assert window._player_view(1) == expected
        assert window.players_panel.current_view() == expected
    finally:
        _close(window)


def test_set_view_is_one_undo_step() -> None:
    """Both coordinates move together: two set_player_field() calls would
    cost two Ctrl+Z and leave a half-moved camera in between."""
    window = _window()
    try:
        window.set_player_view(1)
        assert window._player_view(1) is not None
        window.undo()
        assert window._player_view(1) is None, "one undo must restore BOTH coordinates"
        assert not window.edit_history.can_undo, "Set View pushed more than one record"
        window.redo()
        assert window._player_view(1) is not None
    finally:
        _close(window)


def test_set_view_with_an_off_map_centre_writes_nothing() -> None:
    window = _window()
    try:
        window.map_view.viewport_centre_tile = lambda: None
        window.set_player_view(1)
        assert window._player_view(1) is None
        assert not window.edit_history.can_undo
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_reset_view_writes_minus_one() -> None:
    window = _window()
    try:
        window.set_player_view(1)
        window.reset_player_view(1)
        specs = window._pov_specs()
        raw = [
            window.option_edits.current_value(player_fields.player_field_id(spec.field_id, 1))
            for spec in specs
        ]
        assert raw == [player_fields.POV_UNSET, player_fields.POV_UNSET]
        # -1 reads back as "unset", which is what greys Go to View out.
        assert window._player_view(1) is None
        assert not window.players_panel.go_to_view_button.isEnabled()
    finally:
        _close(window)


def test_go_to_view_centres_on_the_clamped_tile() -> None:
    """An out-of-range stored view still navigates, clamped the way
    _navigate_to_finding clamps rather than refused."""
    window = _window()
    try:
        mm = window.scenario.map_manager
        window._write_player_view(1, mm.map_width + 40, 5, "Set P1 Point of View")
        centred = []
        window.map_view.center_on_tile = lambda x, y: centred.append((x, y))
        window.go_to_player_view(1)
        assert centred == [(mm.map_width - 1, 5)]
    finally:
        _close(window)


# --- the markers ------------------------------------------------------


def test_one_marker_per_player_with_a_view_set() -> None:
    window = _window()
    try:
        assert _marker_tiles(window) == {}
        window.set_player_view(1)
        window.players_panel.select_player(2)
        window.map_view.centerOn(window.map_view.sceneRect().topLeft())
        window.set_player_view(2)
        tiles = _marker_tiles(window)
        assert set(tiles) == {1, 2}
        assert tiles[1] != tiles[2], "the two views were set from different viewports"
    finally:
        _close(window)


def test_markers_follow_the_document_across_save_and_reload(tmp_path, monkeypatch) -> None:
    """The load path pushes each document's own views: a saved view comes
    back as a marker from the file, and opening a file with none drops the
    previous document's markers rather than re-anchoring them."""
    from PyQt5.QtWidgets import QFileDialog

    from descape.scenario_io import load_map_and_units

    out = tmp_path / "pov.aoe2scenario"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    window = _window()
    try:
        # x != y, so a swapped-axes write cannot pass.
        window._write_player_view(1, 12, 30, "Set P1 Point of View")
        window.save_as()
        assert not window.edit_history.is_dirty
        reloaded = load_map_and_units(out)
        specs = {s.field_id: s for s in player_fields.specs_for(reloaded)}
        assert (
            player_fields.current_value(reloaded, specs[player_fields.POV_X_FIELD], 1),
            player_fields.current_value(reloaded, specs[player_fields.POV_Y_FIELD], 1),
        ) == (12, 30)

        window.load_scenario(out)
        assert _marker_tiles(window) == {1: (12, 30)}
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert _marker_tiles(window) == {}, "the previous document's marker survived a load"
    finally:
        _close(window)


def test_no_markers_while_the_toggle_is_off() -> None:
    window = _window(markers=False)
    try:
        window.set_player_view(1)
        assert window.map_view.camera_marker_items() == []
        window.player_cameras_action.setChecked(True)
        assert set(_marker_tiles(window)) == {1}
        window.player_cameras_action.setChecked(False)
        assert window.map_view.camera_marker_items() == []
    finally:
        _close(window)


def test_markers_survive_a_style_switch() -> None:
    """set_source() calls scene().clear(), which destroys the C++ items --
    a stale Python reference would raise RuntimeError on the next rebuild."""
    window = _window()
    try:
        window.set_player_view(1)
        before = _marker_tiles(window)
        window.terrain_style_combo.setCurrentText("Stepped")
        assert _marker_tiles(window) == before
        window.terrain_style_combo.setCurrentText("Flat")
        assert _marker_tiles(window) == before
    finally:
        _close(window)


def test_a_marker_follows_an_elevation_edit_in_stepped() -> None:
    """The overlay is always-on and not selection-tied, so a terrain edit
    under a marker has to re-anchor it."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        window.set_player_view(1)
        item = window.map_view.camera_marker_items()[0]
        tile, before = item.tile(), item.pos()
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(tile[0], tile[1], Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        after = window.map_view.camera_marker_items()[0]
        assert after.tile() == tile
        assert after.pos() != before, "the marker stayed at the pre-edit ground height"
    finally:
        _close(window)


def _device_size(window) -> tuple[float, float]:
    """The marker's size in DEVICE pixels -- what
    ItemIgnoresTransformations is supposed to hold constant."""
    view = window.map_view
    item = view.camera_marker_items()[0]
    rect = item.deviceTransform(view.viewportTransform()).mapRect(item.boundingRect())
    return round(rect.width(), 3), round(rect.height(), 3)


def test_the_glyph_keeps_one_device_size_through_zoom_and_every_style() -> None:
    """Countable, so asserted rather than eyeballed: the glyph is drawn in
    device space, so neither a zoom nor the Flat squash may resize it."""
    window = _window()
    try:
        window.set_player_view(1)
        baseline = _device_size(window)
        window.map_view.scale(3.0, 3.0)
        assert _device_size(window) == baseline
        for style in ("Stepped", "Sloped", "Flat"):
            window.terrain_style_combo.setCurrentText(style)
            assert _device_size(window) == baseline, style
        window.iso_action.setChecked(False)  # Flat, top-down: no squash
        assert _device_size(window) == baseline
    finally:
        _close(window)


def test_emphasis_follows_the_selected_player_and_only_in_players_mode() -> None:
    window = _window()
    try:
        window.set_player_view(1)
        window.players_panel.select_player(2)
        window.set_player_view(2)
        emphasised = {i.player_id: i.is_emphasised() for i in window.map_view.camera_marker_items()}
        assert emphasised == {1: False, 2: True}

        window.players_panel.select_player(1)
        emphasised = {i.player_id: i.is_emphasised() for i in window.map_view.camera_marker_items()}
        assert emphasised == {1: True, 2: False}

        window.mode_combo.setCurrentText("View")
        assert not any(i.is_emphasised() for i in window.map_view.camera_marker_items())
        assert len(window.map_view.camera_marker_items()) == 2, "the markers are not mode-gated"
    finally:
        _close(window)


def test_a_colour_change_recolours_the_marker() -> None:
    window = _window()
    try:
        window.set_player_view(1)
        before = window.map_view.camera_marker_items()[0]._color.name()
        spec = next(s for s in window.players_panel._specs if s.field_id == "color")
        shown = window.players_panel.current_values()["color"]
        window.players_panel.widget_for("color").setCurrentIndex(
            window.players_panel.widget_for("color").findData(0 if shown != 0 else 1)
        )
        after = window.map_view.camera_marker_items()[0]._color.name()
        assert after != before, f"{spec.field_id} edit left the marker on the old colour"
    finally:
        _close(window)


def test_the_toggle_persists_and_is_rebindable() -> None:
    from descape import settings

    window = _window(markers=False)
    try:
        assert settings.get_camera_markers() is False
        window.player_cameras_action.setChecked(True)
        assert settings.get_camera_markers() is True
        assert ("view_player_cameras", "Player Cameras", "") in settings.REBINDABLE_ACTIONS
    finally:
        _close(window)
