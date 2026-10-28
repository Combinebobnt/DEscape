"""TASK-272: the Tool Options row starts each window at Settings > Tools'
defaults, and a toolbar edit lasts the session only, through real offscreen
ViewerWindows.

Each test nulls the settings memo before building the window that must read
config.yaml, or it would pass on a wrong YAML shape. Windows come from
conftest.blank_window(), never stepped_window(), whose pin empties the store.
Every store assertion reads the `tool_options` key only: closeEvent writes
window sizes and load_scenario writes recent files, so the file as a whole
always changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from descape import settings, tool_options

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# Row widgets that are inputs but not tool options, by decision: Target starts on
# Terrain every launch, and Clear rulers is an action, not a setting.
EXCLUDED = ("terrain_target_combo", "clear_rulers_button")


def _stored(tmp_path: Path):
    path = tmp_path / "config.yaml"
    return (yaml.safe_load(path.read_text()) or {}).get("tool_options") if path.is_file() else None


def _fresh_window(monkeypatch, load: bool = True):
    monkeypatch.setattr(settings, "_tool_option_overrides", None)
    return conftest.blank_window(load=load)


def _non_defaults(window) -> dict:
    """A legal non-default value for every option, ids taken from this session's own row lists."""
    from descape import tool_option_bindings

    beach = window.beach_combo.itemData(1)
    walls = [a.data() for a in window.wall_pick_button.menu().actions() if a.data() != tool_options.default("last_wall")]
    family = window.cliff_family_combo.itemData(1)
    pieces = window._cliff_families[family]
    piece = pieces[1].unit_const
    values = {
        "brush_size": 7,
        "brush_shape": "circle_outline",
        "brush_thickness": 2,
        "auto_beach": True,
        "beach_terrain": beach,
        "beach_width": 3,
        "paint_trees": False,
        "paint_eye_candy": True,
        "elevation_level": 6,
        "rect_filled": False,
        "paste_terrain": False,
        "paste_elevation": False,
        "paste_units": False,
        "free_placement": True,
        "off_grid": True,
        "walls_skip_occupied": True,
        "gate_walls": True,
        "last_wall": walls[0],
        "cliff_family": family,
        "cliff_piece": piece,
        "cliff_frame": 2,
        "unit_brush_density": 40,
        "unit_brush_random": False,
        "convert_sources": [1, 2],
        "eraser_targets": ["trees", "cliffs"],
        "area_edit": True,
        "area_move": False,
        "area_resize": False,
        "area_tile_borders": True,
    }
    assert tool_option_bindings.cliff_piece_const(window, family, piece) == piece
    assert list(values) == [spec.name for spec in tool_options.OPTIONS], "a new option needs a value here"
    assert not any(tool_options.is_default(name, value) for name, value in values.items())
    return values


def test_binding_names_are_the_schema_names() -> None:
    window = conftest.blank_window(load=False)
    try:
        assert list(window.tool_option_bindings) == [spec.name for spec in tool_options.OPTIONS]
    finally:
        conftest.close_window(window)


def test_every_input_on_the_row_is_a_tool_option_or_a_named_exclusion() -> None:
    """The drift guard: a widget added to the row fails here until it is classified."""
    from PyQt5.QtWidgets import QCheckBox, QComboBox, QSpinBox, QToolButton

    window = conftest.blank_window(load=False)
    try:
        toolbar = window.tool_options_toolbar
        bound = {id(widget) for binding in window.tool_option_bindings.values() for widget in binding.widgets}
        excluded = {id(getattr(window, name)) for name in EXCLUDED}
        unclassified = []
        inputs = 0
        for action in toolbar.actions():
            widget = toolbar.widgetForAction(action)
            if not isinstance(widget, (QSpinBox, QCheckBox, QComboBox, QToolButton)):
                continue
            inputs += 1
            if id(widget) not in bound | excluded:
                unclassified.append(f"{type(widget).__name__} {widget.text() if hasattr(widget, 'text') else ''}")
        assert unclassified == []
        assert inputs == len(window.tool_option_bindings) + len(EXCLUDED), "a binding's widget is not on the row"
    finally:
        conftest.close_window(window)


def test_a_fresh_window_on_an_empty_config_reads_every_schema_default() -> None:
    window = conftest.blank_window()
    try:
        values = window.tool_option_values()
        assert {name: value for name, value in values.items() if not tool_options.is_default(name, value)} == {}
    finally:
        conftest.close_window(window)


def test_every_option_stored_at_once_reaches_its_control(tmp_path: Path, monkeypatch) -> None:
    """The init-order check: every handler runs during construction without raising."""
    probe = conftest.blank_window(load=False)
    try:
        values = _non_defaults(probe)
    finally:
        conftest.close_window(probe)
    settings.set_tool_option_defaults(values)
    stored = _stored(tmp_path)
    window = _fresh_window(monkeypatch)
    try:
        assert window.tool_option_values() == values
        wall_name = next(a.text() for a in window.wall_pick_button.menu().actions() if a.data() == values["last_wall"])
        assert window.wall_pick_button.text().endswith(wall_name)
        # Under Pan the map view gets size 1 / square by design; Draw takes the row's brush.
        window.mode_combo.setCurrentText("Terrain")
        window.draw_action.setChecked(True)
        view = window.map_view
        assert (view._brush_size, view._brush_shape, view._brush_thickness) == (7, "circle_outline", 2)
        assert (view._area_move, view._area_resize, view._area_tile_borders) == (False, False, True)
    finally:
        conftest.close_window(window)
    assert _stored(tmp_path) == stored, "building a window must not write the store"


def test_a_toolbar_edit_writes_nothing_and_the_next_window_is_back_at_the_default(tmp_path: Path, monkeypatch) -> None:
    window = conftest.blank_window()
    try:
        for name, value in _non_defaults(window).items():
            assert window.apply_tool_option_default(name, value), name
    finally:
        conftest.close_window(window)
    assert _stored(tmp_path) is None
    second = _fresh_window(monkeypatch)
    try:
        values = second.tool_option_values()
        assert {name: value for name, value in values.items() if not tool_options.is_default(name, value)} == {}
    finally:
        conftest.close_window(second)


def _cliff_window(tmp_path: Path, monkeypatch, stored: dict):
    settings.set_tool_option_defaults(stored)
    return _fresh_window(monkeypatch)


def _first_family_pieces():
    probe = conftest.blank_window(load=False)
    try:
        return [piece.unit_const for piece in probe._cliff_families[probe.cliff_family_combo.itemData(0)]]
    finally:
        conftest.close_window(probe)


def test_a_cliff_family_that_does_not_exist_skips_the_piece_and_frame(tmp_path: Path, monkeypatch) -> None:
    # The piece would apply on the row's first family: only the chain stops it.
    piece = _first_family_pieces()[1]
    window = _cliff_window(tmp_path, monkeypatch, {"cliff_family": "no_such_cliff.sld", "cliff_piece": piece, "cliff_frame": 2})
    try:
        assert window.cliff_family_combo.currentIndex() == 0
        assert window.cliff_piece_combo.currentIndex() == 0
        assert window.cliff_frame_spin.value() == 0
    finally:
        conftest.close_window(window)


def test_a_piece_that_did_not_apply_skips_the_frame(tmp_path: Path, monkeypatch) -> None:
    window = _cliff_window(tmp_path, monkeypatch, {"cliff_piece": 999_999, "cliff_frame": 2})
    try:
        assert window.cliff_piece_combo.currentIndex() == 0
        assert window.cliff_frame_spin.value() == 0
    finally:
        conftest.close_window(window)


def test_a_frame_past_the_pieces_range_is_not_applied(tmp_path: Path, monkeypatch) -> None:
    window = _cliff_window(tmp_path, monkeypatch, {"cliff_frame": 500})
    try:
        assert window.cliff_frame_spin.maximum() < 500
        assert window.cliff_frame_spin.value() == 0, "setValue would have clamped it to the maximum"
    finally:
        conftest.close_window(window)


def test_an_id_unknown_this_session_stays_stored_and_leaves_the_control_at_its_default(tmp_path: Path, monkeypatch) -> None:
    settings.set_tool_option_defaults({"last_wall": 1, "cliff_family": "no_such_cliff.sld", "beach_terrain": 999})
    window = _fresh_window(monkeypatch)
    try:
        assert window._last_wall_const == tool_options.default("last_wall")
        assert window.cliff_family_combo.currentIndex() == 0
        assert window.beach_combo.currentIndex() == 0
        settings.set_tool_option_default("brush_size", 4)  # an unrelated edit rewrites the key
    finally:
        conftest.close_window(window)
    assert _stored(tmp_path) == {"brush_size": 4, "last_wall": 1, "cliff_family": "no_such_cliff.sld", "beach_terrain": 999}


def test_target_is_never_restored(tmp_path: Path, monkeypatch) -> None:
    """Target is out of the schema by decision; a hand-written entry for it is ignored, then dropped."""
    from descape import terrain_target

    assert not any("target" in spec.name and spec.group == "terrain" for spec in tool_options.OPTIONS)
    (tmp_path / "config.yaml").write_text("tool_options:\n  terrain_target: layer\n")
    window = _fresh_window(monkeypatch)
    try:
        assert window.terrain_target_combo.currentData() == terrain_target.TERRAIN
    finally:
        conftest.close_window(window)
    settings.set_tool_option_default("brush_size", 2)
    assert _stored(tmp_path) == {"brush_size": 2}
