"""ViewerWindow.on_hover()'s tile readout on a non-square map.

Calls the method unbound on a stub `self`, so no window or QApplication is
needed. Every shipped fixture is square, hence the duck-typed map_manager.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import conftest

pytestmark = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")


class _Label:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class _NonSquareMapManager:
    """Mirrors the real MapManager on a non-square map: get_tile_safe()
    returns None for every coordinate, on-map or not."""

    def __init__(self, width: int, height: int) -> None:
        self.map_width = width
        self.map_height = height
        self.terrain = [
            SimpleNamespace(x=i % width, y=i // width, terrain_id=0, elevation=0, layer=-1)
            for i in range(width * height)
        ]

    def get_tile_safe(self, x: int, y: int):
        return None


def _stub_window(mm: _NonSquareMapManager) -> SimpleNamespace:
    from descape.viewer import ViewerWindow

    window = SimpleNamespace(
        _hover_tile=None,
        scenario=SimpleNamespace(map_manager=mm),
        hover_label=_Label(),
        # The Terrain-mode page's copy of the readout (GH #56); _set_hover_text() writes both.
        terrain_panel=SimpleNamespace(set_hover_text=lambda text: None),
    )
    window._set_hover_text = lambda text: ViewerWindow._set_hover_text(window, text)
    window._refresh_hover_text = lambda: ViewerWindow._refresh_hover_text(window)
    return window


def test_hover_readout_updates_on_non_square_map() -> None:
    from descape.viewer import ViewerWindow

    mm = _NonSquareMapManager(width=6, height=3)
    target = mm.terrain[2 * 6 + 5]
    target.terrain_id = 15
    target.elevation = 4
    window = _stub_window(mm)

    ViewerWindow.on_hover(window, (5, 2))

    assert window.hover_label.text.startswith("(5, 2)  ")
    assert window.hover_label.text.endswith("elevation=4")


def test_hover_readout_names_a_layered_terrain_before_the_elevation() -> None:
    from descape.terrain_palette import name_for_terrain_id
    from descape.viewer import ViewerWindow

    mm = _NonSquareMapManager(width=6, height=3)
    target = mm.terrain[2 * 6 + 5]
    target.terrain_id, target.layer, target.elevation = 15, 2, 4
    window = _stub_window(mm)

    ViewerWindow.on_hover(window, (5, 2))
    # The Terrain page elides this line on the right, so the layer sits before the elevation.
    assert window.hover_label.text == (
        f"(5, 2)  {name_for_terrain_id(15)}  layer={name_for_terrain_id(2)}  elevation=4"
    )
    assert name_for_terrain_id(2) != name_for_terrain_id(15)


def test_hover_readout_omits_the_layer_on_an_unlayered_tile() -> None:
    from descape.terrain_palette import name_for_terrain_id
    from descape.viewer import ViewerWindow

    mm = _NonSquareMapManager(width=6, height=3)
    mm.terrain[1].terrain_id = 15
    window = _stub_window(mm)

    ViewerWindow.on_hover(window, (1, 0))
    assert window.hover_label.text == f"(1, 0)  {name_for_terrain_id(15)}  elevation=0"
    assert "layer" not in window.hover_label.text


def test_hover_readout_off_map_on_non_square_map_shows_idle_text() -> None:
    from descape.viewer import ViewerWindow
    from descape.viewer_common import HOVER_IDLE_TEXT

    mm = _NonSquareMapManager(width=6, height=3)
    window = _stub_window(mm)
    window.hover_label.setText("stale")

    ViewerWindow.on_hover(window, (2, 3))

    assert window.hover_label.text == HOVER_IDLE_TEXT


# -- a real window: the readout after an edit under a still pointer (TASK-419) --

_HOVERED = (40, 40)
_NEW_TERRAIN = 15  # any id other than the blank template's own terrain


def _edit_window(tool: str):
    window = conftest.terrain_edit_window()
    window._on_tool_selected(tool)
    window.terrain_panel.set_terrain(_NEW_TERRAIN)
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    return window


def _readouts(window) -> tuple[str, str]:
    return window.hover_label.text(), window.terrain_panel.hover_text()


def _one_tile_stroke(window, tile) -> None:
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(*tile, 0)
    window.on_edit_stroke_end()


def _old_and_new_names(window) -> tuple[str, str]:
    from descape.terrain_palette import name_for_terrain_id

    old = name_for_terrain_id(window.scenario.map_manager.get_tile(*_HOVERED).terrain_id)
    new = name_for_terrain_id(_NEW_TERRAIN)
    assert old != new
    return old, new


@pytest.mark.gui
def test_the_readout_names_the_new_terrain_after_a_draw_stroke_under_a_still_pointer() -> None:
    window = _edit_window("draw")
    try:
        _old, new = _old_and_new_names(window)
        window.on_hover(_HOVERED)
        _one_tile_stroke(window, _HOVERED)
        for text in _readouts(window):
            assert f"  {new}  " in text, f"after the stroke: {text!r}"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
def test_the_readout_follows_undo_and_redo_under_a_still_pointer() -> None:
    window = _edit_window("draw")
    try:
        old, new = _old_and_new_names(window)
        window.on_hover(_HOVERED)
        _one_tile_stroke(window, _HOVERED)
        window.on_hover(_HOVERED)  # a pointer move after the stroke: the readout starts current
        window.undo()
        for text in _readouts(window):
            assert f"  {old}  " in text, f"after undo: {text!r}"
        window.redo()
        for text in _readouts(window):
            assert f"  {new}  " in text, f"after redo: {text!r}"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
def test_the_readout_shows_the_new_elevation_after_an_elevate_stroke() -> None:
    window = _edit_window("elevation")
    try:
        window.on_hover(_HOVERED)
        _one_tile_stroke(window, _HOVERED)
        raised = window.scenario.map_manager.get_tile(*_HOVERED).elevation
        assert raised > 0, "the stroke raised nothing"
        for text in _readouts(window):
            assert text.endswith(f"elevation={raised}"), text
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
def test_the_readout_names_the_filled_terrain_after_a_fill() -> None:
    from descape.terrain_palette import name_for_terrain_id

    window = _edit_window("fill")
    try:
        window.on_hover(_HOVERED)
        window.on_fill(*_HOVERED, 0)
        assert window.scenario.map_manager.get_tile(*_HOVERED).terrain_id == _NEW_TERRAIN
        for text in _readouts(window):
            assert f"  {name_for_terrain_id(_NEW_TERRAIN)}  " in text, text
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
def test_the_readout_stays_idle_after_a_stroke_with_the_pointer_off_the_map() -> None:
    from descape.viewer_common import HOVER_IDLE_TEXT

    window = _edit_window("draw")
    try:
        window.on_hover(None)
        _one_tile_stroke(window, _HOVERED)
        assert _readouts(window) == (HOVER_IDLE_TEXT, HOVER_IDLE_TEXT)
    finally:
        window.edit_history.mark_saved()
        window.close()
