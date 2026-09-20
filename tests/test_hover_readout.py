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
            SimpleNamespace(x=i % width, y=i // width, terrain_id=0, elevation=0) for i in range(width * height)
        ]

    def get_tile_safe(self, x: int, y: int):
        return None


def _stub_window(mm: _NonSquareMapManager) -> SimpleNamespace:
    return SimpleNamespace(
        _hover_tile=None,
        scenario=SimpleNamespace(map_manager=mm),
        hover_label=_Label(),
    )


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


def test_hover_readout_off_map_on_non_square_map_shows_idle_text() -> None:
    from descape.viewer import HOVER_IDLE_TEXT, ViewerWindow

    mm = _NonSquareMapManager(width=6, height=3)
    window = _stub_window(mm)
    window.hover_label.setText("stale")

    ViewerWindow.on_hover(window, (2, 3))

    assert window.hover_label.text == HOVER_IDLE_TEXT
