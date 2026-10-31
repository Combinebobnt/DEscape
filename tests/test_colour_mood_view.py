"""TASK-238: the Colour Mood preview through a real offscreen ViewerWindow.

The canvas takes the grade at paint time and the scene brush takes it around
the map: no cache, chunk or write path sees it, so row off must be byte
identical to a file with no mood at all, and DEscape's overlay items, drawn
after the canvas, keep their colours. colour_mood.apply() is the oracle for
the graded pixels. The preframe path's half lives in tests/test_zoom_preframe.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from descape import colour_mood, grid_overlay, option_fields, settings
from descape.colour_mood import apply, grade_for
from descape.options_model import OptionsEditModel
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.scenario_write import write_scenario
from descape.unit_model import UnitEditModel

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIELD = "map_color_mood"
LAYER = "colour_mood"
CENTRE = 60
NIGHT = grade_for("Night")


def _file(tmp_path: Path, name: str, mood: str | None = None, *, unit: bool = False) -> Path:
    """The blank template, saved with `mood` through the real write path."""
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    options = OptionsEditModel(loaded)
    units = UnitEditModel(loaded)
    if mood is not None:
        options.set_value(FIELD, mood)
    if unit:
        units.add(player=1, unit_const=83, x=CENTRE + 0.5, y=CENTRE + 0.5)
    out = tmp_path / f"{name}.aoe2scenario"
    write_scenario(loaded, out, options=options, units=units, backup=False)
    return out


def _probe_rect(view, tile: int = CENTRE, tiles: int = 8):
    from PyQt5.QtCore import QRectF

    from descape import iso_geometry

    proj = view._iso_proj
    sx, sy = iso_geometry.tile_screen_origin(tile, tile, 0, proj)
    return QRectF(sx, sy, tiles * proj.half_w, tiles * proj.half_h)


def _capture(view, rect) -> np.ndarray:
    from testkit.qt_capture import scene_rect_to_array

    return scene_rect_to_array(view.scene(), rect).astype(int)


def _outside_rect():
    from PyQt5.QtCore import QRectF

    return QRectF(-40, 40, 20, 20)


def _toggle(window, on: bool) -> None:
    window.layer_actions[LAYER].setChecked(on)


def _open(path: Path):
    """`path` in Stepped with the row, off by default, then turned on."""
    window = conftest.stepped_window(path)
    _toggle(window, True)
    return window


def _open_with_row_on(path: Path):
    """The row on before `path` loads, so the open itself takes the grade."""
    window = _open(BLANK_TEMPLATE_PATH)
    window.load_scenario(path)
    return window


def _graded(pixels: np.ndarray, grade: colour_mood.MoodGrade) -> np.ndarray:
    """`apply()` of every pixel, through its unique colours."""
    colours, inverse = np.unique(pixels.reshape(-1, 3), axis=0, return_inverse=True)
    mapped = np.array([apply(grade, tuple(int(c) for c in colour)) for colour in colours])
    return mapped[inverse.reshape(-1)].reshape(pixels.shape)


def _within(got: np.ndarray, want: np.ndarray, tolerance: int = 2) -> bool:
    return bool((np.abs(got - want) <= tolerance).all())


# -- the canvas and the background -------------------------------------------------


def test_row_off_is_byte_identical_to_a_file_with_no_mood(tmp_path: Path) -> None:
    plain_window = _open(_file(tmp_path, "plain"))
    try:
        view = plain_window.map_view
        assert view.colour_grade is None
        plain = _capture(view, _probe_rect(view)), _capture(view, _outside_rect())
    finally:
        conftest.close_window(plain_window)

    window = _open(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        assert view.colour_grade == NIGHT
        _toggle(window, False)
        assert view.colour_grade is None
        off = _capture(view, _probe_rect(view)), _capture(view, _outside_rect())
        assert np.array_equal(off[0], plain[0])
        assert np.array_equal(off[1], plain[1])
    finally:
        conftest.close_window(window)


def test_a_file_with_no_mood_paints_no_fill(tmp_path: Path, monkeypatch) -> None:
    from descape import viewer_canvas

    calls = []
    monkeypatch.setattr(viewer_canvas, "paint_colour_grade", lambda *a: calls.append(a))
    window = _open(_file(tmp_path, "plain"))
    try:
        _capture(window.map_view, _probe_rect(window.map_view))
        assert calls == []
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped", "Flat"])
def test_the_canvas_and_background_take_the_grade(tmp_path: Path, style) -> None:
    from PyQt5.QtCore import QRectF

    window = _open(_file(tmp_path, "night", "Night"))
    try:
        window.terrain_style_combo.setCurrentText(style)
        view = window.map_view
        assert view._canvas_item.grade == NIGHT, "a style switch built a new item without the grade"
        rect = _outside_rect()
        centre = view._map_rect.center()
        terrain = QRectF(int(centre.x()), int(centre.y()), 80, 80)
        on = _capture(view, terrain)
        on_outside = _capture(view, rect)
        _toggle(window, False)
        off = _capture(view, terrain)
        off_outside = _capture(view, rect)
        assert off.any(axis=2).all(), "a void pixel in the probe: it is not all terrain"
        assert not np.array_equal(on, off)
        assert _within(on, _graded(off, NIGHT))
        assert (off_outside == view.OUTSIDE_MAP_COLOR.getRgb()[:3]).all()
        assert (on_outside == apply(NIGHT, view.OUTSIDE_MAP_COLOR.getRgb()[:3])).all()
    finally:
        conftest.close_window(window)


def test_the_background_brush_comes_back_with_the_row_off(tmp_path: Path) -> None:
    window = _open(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        assert view.scene().backgroundBrush().color() != view.OUTSIDE_MAP_COLOR
        _toggle(window, False)
        assert view.scene().backgroundBrush().color() == view.OUTSIDE_MAP_COLOR
    finally:
        conftest.close_window(window)


# -- what grades and what does not ---------------------------------------------------


def test_the_baked_grid_takes_the_grade(tmp_path: Path) -> None:
    """Follow Terrain Elevation on: the grid is chunk pixels, so it grades as the game's would."""
    window = _open(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        rect = _probe_rect(view)
        _toggle(window, False)
        bare = _capture(view, rect)
        window.grid_action.setChecked(True)
        off = _capture(view, rect)
        assert (off != bare).any(), "the baked grid drew nothing, so the check is vacuous"
        _toggle(window, True)
        assert _within(_capture(view, rect), _graded(off, NIGHT))
    finally:
        conftest.close_window(window)


def test_the_follow_off_grid_keeps_its_ink(tmp_path: Path) -> None:
    """A GridItem line is black at some alpha over the canvas, so it scales the
    pixel under it by the same factor graded or not. A line graded with the
    canvas would keep the tint instead, and read lighter over dark terrain."""
    settings.set_grid_blend(grid_overlay.BLEND_MIN)
    window = _open(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        rect = _probe_rect(view)
        on_bare = _capture(view, rect)
        _toggle(window, False)
        off_bare = _capture(view, rect)
        _toggle(window, True)
        view.set_grid_follow_elevation(False)
        view.set_grid_overlay(True)
        on_line = _capture(view, rect)
        _toggle(window, False)
        off_line = _capture(view, rect)
        line = (off_line != off_bare).any(axis=2)
        assert line.any(), "the grid drew nothing, so the check is vacuous"
        off_factor = off_line[line] / off_bare[line]
        on_factor = on_line[line] / on_bare[line]
        assert np.abs(off_factor - on_factor).max() <= 0.05
    finally:
        conftest.close_window(window)


def test_a_selection_outline_keeps_its_colour(tmp_path: Path) -> None:
    from PyQt5.QtCore import QRectF

    window = _open(_file(tmp_path, "night", "Night", unit=True))
    try:
        view = window.map_view
        window._rebuild_unit_index()
        entry = view._unit_index.entries[-1]
        assert entry.unit.unit_const == 83
        window._selection = [(entry.player_id, entry.unit.reference_id)]
        view.set_unit_selection([entry])
        outline_item = next(iter(view._unit_select_groups.values()))[2]
        rect = outline_item.sceneBoundingRect().adjusted(-4, -4, 4, 4).toAlignedRect()
        on = _capture(view, QRectF(rect))
        _toggle(window, False)
        off = _capture(view, QRectF(rect))
        ink = tuple(outline_item.pen().color().getRgb()[:3])
        outline = (off == ink).all(axis=2)
        assert outline.any(), "no outline pixel in the probe, so the check is vacuous"
        assert (on[outline] == ink).all()
    finally:
        window.edit_history.mark_saved()
        conftest.close_window(window)


# -- MapView ---------------------------------------------------------------------------


def test_an_equal_grade_schedules_no_update(tmp_path: Path, monkeypatch) -> None:
    window = _open(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        viewport = view.viewport()
        calls = []
        monkeypatch.setattr(viewport, "update", lambda *a: calls.append(a))
        view.set_colour_grade(grade_for("Night"))
        assert calls == []
        view.set_colour_grade(grade_for("Winter"))
        assert len(calls) == 1
        assert view._canvas_item.grade == grade_for("Winter")
    finally:
        conftest.close_window(window)


# -- the viewer's wiring ---------------------------------------------------------------


def _status(window) -> str:
    return window.status_log.toPlainText()


def _spec(field_id: str) -> option_fields.OptionFieldSpec:
    return next(s for s in option_fields._SPECS if s.field_id == field_id)


def test_opening_a_mood_file_sets_the_grade_and_logs_one_line(tmp_path: Path) -> None:
    window = _open_with_row_on(_file(tmp_path, "night", "Night"))
    try:
        assert window.map_view.colour_grade == NIGHT
        assert window.map_view._canvas_item.grade == NIGHT
        assert [line for line in _status(window).splitlines() if "Colour mood:" in line and "Night" in line]
    finally:
        conftest.close_window(window)


def test_a_night_file_with_the_default_row_draws_ungraded_and_logs_nothing(tmp_path: Path) -> None:
    window = conftest.stepped_window(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        assert not window.layer_actions[LAYER].isChecked()
        assert view.colour_grade is None and view._canvas_item.grade is None
        assert view.scene().backgroundBrush().color() == view.OUTSIDE_MAP_COLOR
        assert "Colour mood:" not in _status(window)
    finally:
        conftest.close_window(window)


def test_a_persisted_row_on_grades_a_file_on_open(tmp_path: Path, monkeypatch) -> None:
    """A real read of config.yaml: stepped_window() pins the rows to their defaults, so not through it."""
    settings.set_view_layer(LAYER, True)
    monkeypatch.setattr(settings, "_view_layers", None)
    window = conftest.blank_window(load=False)
    try:
        assert window.layer_actions[LAYER].isChecked()
        window.load_scenario(_file(tmp_path, "night", "Night"))
        view = window.map_view
        assert view.colour_grade == NIGHT and view._canvas_item.grade == NIGHT
        assert window._cache.layers.colour_mood is True
        assert [line for line in _status(window).splitlines() if "Colour mood:" in line and "Night" in line]
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("mood", [None, "Default", "Empty"], ids=["empty", "default", "empty-name"])
def test_a_no_op_mood_sets_nothing_and_logs_nothing(tmp_path: Path, mood) -> None:
    window = _open_with_row_on(_file(tmp_path, "plain", mood))
    try:
        assert window.map_view.colour_grade is None
        assert "Colour mood:" not in _status(window)
    finally:
        conftest.close_window(window)


def test_a_row_edit_undo_and_redo_move_the_grade(tmp_path: Path) -> None:
    window = _open(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        spec = _spec(FIELD)
        window._set_option_scalar(spec, "Winter")
        assert view.colour_grade == grade_for("Winter")
        window.undo()
        assert view.colour_grade == NIGHT
        window.redo()
        assert view.colour_grade == grade_for("Winter")
        window._set_option_scalar(spec, "Default")
        assert view.colour_grade is None
        window.undo()
        window.undo()
        assert view.colour_grade == NIGHT
        assert not window.edit_history.is_dirty
    finally:
        conftest.close_window(window)


def test_an_unrelated_option_edit_moves_nothing(tmp_path: Path, monkeypatch) -> None:
    window = _open(_file(tmp_path, "night", "Night"))
    try:
        viewport = window.map_view.viewport()
        calls = []
        monkeypatch.setattr(viewport, "update", lambda *a: calls.append(a))
        spec = _spec("collide_and_correct")
        window._set_option_scalar(spec, 1)
        assert window.map_view.colour_grade == NIGHT
        assert calls == []
    finally:
        conftest.close_window(window)


def test_toggling_the_row_clears_and_restores_without_a_set_layers_rebuild(tmp_path: Path, monkeypatch) -> None:
    window = _open(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        cache = window._cache
        applied = []
        monkeypatch.setattr(window, "_apply_layers", lambda *a, **k: applied.append(a))
        _toggle(window, False)
        assert view.colour_grade is None and window._cache is cache
        assert cache.layers.colour_mood is False and settings.get_view_layers().colour_mood is False
        assert _status(window).splitlines()[-1].endswith("off")
        _toggle(window, True)
        assert view.colour_grade == NIGHT and cache.layers.colour_mood is True
        assert applied == [], "a draw-only row took the wait-cursor path"
    finally:
        conftest.close_window(window)


def test_a_row_edit_with_the_layer_off_previews_nothing_until_it_is_on(tmp_path: Path) -> None:
    window = _open(_file(tmp_path, "plain"))
    try:
        view = window.map_view
        _toggle(window, False)
        spec = _spec(FIELD)
        window._set_option_scalar(spec, "Night")
        assert view.colour_grade is None
        _toggle(window, True)
        assert view.colour_grade == NIGHT
    finally:
        conftest.close_window(window)


def test_closing_clears_the_grade_and_the_brush(tmp_path: Path) -> None:
    window = _open(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        window.close_scenario()
        assert view.colour_grade is None
        assert view.scene().backgroundBrush().color() == view.OUTSIDE_MAP_COLOR
    finally:
        conftest.close_window(window)


def test_the_new_documents_grade_is_set_before_its_item_exists(tmp_path: Path, monkeypatch) -> None:
    """The item's first paint can fire inside set_source(), so the refresh goes first."""
    plain, night = _file(tmp_path, "plain"), _file(tmp_path, "night", "Night")
    window = _open(plain)
    try:
        view = window.map_view
        seen = []
        real = view.set_source
        monkeypatch.setattr(view, "set_source", lambda *a, **k: (seen.append(view.colour_grade), real(*a, **k))[1])
        window.load_scenario(night)
        assert seen[-1] == NIGHT
        window.load_scenario(plain)
        assert seen[-1] is None
    finally:
        conftest.close_window(window)


def test_a_rerender_keeps_the_grade_and_a_pending_edit(tmp_path: Path) -> None:
    window = _open(_file(tmp_path, "night", "Night"))
    try:
        view = window.map_view
        spec = _spec(FIELD)
        window._set_option_scalar(spec, "Winter")
        window.refresh_map()
        assert view._canvas_item.grade == grade_for("Winter")
    finally:
        conftest.close_window(window)
