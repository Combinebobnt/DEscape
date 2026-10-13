"""The range ring's centre follows the drawn footprint (GH #200).

A building stored off its span's parity (a 4x4 Town Center at `.5`) draws its
footprint half a tile from its stored coordinate, and the ring used to stay
on the coordinate. Two layers: render.unit_footprint_centre() against
unit_tile_bounds(), Qt-free, and the live ring item against the live
selection outline in each terrain style.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from descape import range_overlay, render
from descape.scenario_io import BLANK_TEMPLATE_PATH
from descape.terrain_palette import BUILDING_TILE_SPANS

import conftest

_TOWN_CENTER = 109  # 4x4, ranged
_KREPOST = 1251  # 3x3, ranged
_WATCH_TOWER = 79  # 1x1, ranged
_VILLAGER = 83  # not a building: span 1x1, arbitrary float coordinates

# (label, const, stored x, stored y, expected centre)
_HALF_STORED_EVEN = ("4x4 stored at .5", _TOWN_CENTER, 20.5, 20.5, (21.0, 21.0))
_UNCHANGED = [
    ("4x4 at tile + span/2", _TOWN_CENTER, 40.0, 40.0, (40.0, 40.0)),
    ("3x3 at tile + span/2", _KREPOST, 60.5, 60.5, (60.5, 60.5)),
    ("1x1 at .5", _WATCH_TOWER, 80.5, 80.5, (80.5, 80.5)),
]
_OFF_GRID_TOWER = ("1x1 off the grid", _WATCH_TOWER, 100.3, 30.8, (100.3, 30.8))
_CASES = [_HALF_STORED_EVEN, *_UNCHANGED, _OFF_GRID_TOWER]
_IDS = [case[0] for case in _CASES]


def test_the_fixture_consts_are_what_the_cases_assume() -> None:
    assert BUILDING_TILE_SPANS[_TOWN_CENTER] == (4, 4)
    assert BUILDING_TILE_SPANS[_KREPOST] == (3, 3)
    assert BUILDING_TILE_SPANS[_WATCH_TOWER] == (1, 1)
    assert _VILLAGER not in BUILDING_TILE_SPANS
    for const in (_TOWN_CENTER, _KREPOST, _WATCH_TOWER):
        assert range_overlay.ring_radius_for_const(const) is not None


@pytest.mark.parametrize(("label", "const", "x", "y", "expected"), _CASES, ids=_IDS)
def test_footprint_centre_per_placement(label, const, x, y, expected) -> None:
    assert render.unit_footprint_centre(SimpleNamespace(x=x, y=y, unit_const=const)) == expected


@pytest.mark.parametrize(
    ("const", "x", "y"),
    [(_TOWN_CENTER, 20.5, 20.5), (_TOWN_CENTER, 40.0, 40.0), (_KREPOST, 60.5, 60.5), (_KREPOST, 60.0, 61.0)],
)
def test_a_buildings_footprint_centre_is_the_middle_of_its_drawn_tiles(const, x, y) -> None:
    """Tied to unit_tile_bounds(), the box the footprint is painted and
    outlined from, so the two cannot drift if the anchor rule changes."""
    unit = SimpleNamespace(x=x, y=y, unit_const=const)
    x0, x1, y0, y1 = render.unit_tile_bounds(unit, 120, 120)
    assert render.unit_footprint_centre(unit) == ((x0 + x1) / 2, (y0 + y1) / 2)


@pytest.mark.parametrize(("x", "y"), [(50.5, 50.5), (50.13, 50.87), (50.99, 50.01)])
def test_a_free_placed_villager_keeps_its_true_position(x, y) -> None:
    assert render.unit_footprint_centre(SimpleNamespace(x=x, y=y, unit_const=_VILLAGER)) == (x, y)


# --- the live ring item ------------------------------------------------


def _gui(test):
    """The gui-tier marks, per test: the Qt-free half above must run without PyQt5."""
    skip = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
    return pytest.mark.gui(skip(test))



@dataclass
class SyntheticUnit:
    x: float
    y: float
    unit_const: int
    reference_id: int
    rotation: float = 0.0
    z: float = 0.0
    garrisoned_in_id: int = -1


def _ring_window(style: str):
    """Every case's building on P1 of the blank (flat) template, in Units
    mode with View > Range Rings on. Flat means the plain grid, not Flat +
    Isometric, which draws through the Stepped branch."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    units = window.scenario.unit_manager.units
    for i, (_label, const, x, y, _centre) in enumerate(_CASES):
        units[1].append(SyntheticUnit(x=x, y=y, unit_const=const, reference_id=301 + i))
    if style == "Flat":
        window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Units")
    window.range_rings_action.setChecked(True)
    assert window.map_view._terrain_style == style.lower()
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _half_tile_px(map_view) -> float:
    """What a half-tile shift on both axes moves a point by on screen."""
    if map_view._terrain_style == "flat":
        return map_view._tile_pixels / 2
    return float(map_view._iso_proj.half_w)


def _ring_and_footprint_centres(window, reference_id: int):
    mv = window.map_view
    (entry,) = [e for e in mv._unit_index.entries if e.unit.reference_id == reference_id]
    mv.set_unit_selection([entry])
    assert mv._range_ring_item is not None, "no ring drawn"
    return mv._range_ring_item.path().boundingRect().center(), mv._unit_path(entry).boundingRect().center()


@_gui
@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
@pytest.mark.parametrize("case", range(len(_CASES)), ids=_IDS)
def test_the_ring_is_centred_on_the_selection_outline(style: str, case: int) -> None:
    """GH #200: a 4x4 stored at `.5` used to ring half a tile off its own
    outline, in every style. The other placements pin what must not move."""
    window = _ring_window(style)
    try:
        assert _half_tile_px(window.map_view) >= 8, "a half-tile miss would hide inside the tolerance"
        ring, footprint = _ring_and_footprint_centres(window, 301 + case)
        assert (ring.x(), ring.y()) == pytest.approx((footprint.x(), footprint.y()), abs=1.0)
    finally:
        _close(window)


@_gui
@pytest.mark.parametrize("case", [*_UNCHANGED, _OFF_GRID_TOWER], ids=[c[0] for c in (*_UNCHANGED, _OFF_GRID_TOWER)])
def test_a_ring_that_was_already_centred_stays_on_the_stored_coordinate(case) -> None:
    """Flat's scene is (x * tile_px, y * tile_px), so the ring's centre reads
    back as a map point with no footprint in between: the off-grid tower's
    ring sits on its true position, not on its tile's centre."""
    _label, _const, x, y, _centre = case
    window = _ring_window("Flat")
    try:
        ring, _footprint = _ring_and_footprint_centres(window, 301 + _CASES.index(case))
        tile_px = window.map_view._tile_pixels
        assert (ring.x() / tile_px, ring.y() / tile_px) == pytest.approx((x, y), abs=0.02)
    finally:
        _close(window)
