"""render.dirty_screen_bboxes_iso_grouped() against its oracle: one
dirty_screen_bbox_iso() call per group, sized as a unit edit's tight block
rect is (with_sprites off, units_changed set). The grouped pass may only ever
answer what those calls answer, or decline.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from descape import iso_geometry, render
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units


def _scenario():
    """The blank template with a jagged height field, so neighbouring tiles'
    swept ranges differ."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    rng = random.Random(3)
    for tile in scenario.map_manager.terrain:
        tile.elevation = rng.randrange(0, 6)
    return scenario


def _groups(w: int, h: int) -> list[list[int]]:
    rng = random.Random(5)
    block = [y * w + x for y in range(40, 56) for x in range(32, 48)]
    return [
        [0, 1, w],  # the map's first corner: the ring is clipped on two sides
        [w * h - 1, w * h - 2],  # and its last
        block,
        [],  # between two filled groups
        block[:40] + block[:40],  # every tile twice, and shared with the group above
        [70 * w + 70],
        [rng.randrange(w * h) for _ in range(300)],  # scattered, past the array path's 256
        [y * w + (w - 1) for y in range(10, 30)],  # along the right edge
    ]


def _oracle(scenario, groups, elevations, proj, **kwargs):
    return [
        render.dirty_screen_bbox_iso(
            scenario, group, elevations, proj, with_units=True, with_sprites=False, units_changed=True, **kwargs
        )
        for group in groups
    ]


@pytest.mark.parametrize("flatten", [False, True], ids=["stepped", "flat-iso"])
def test_the_grouped_pass_answers_what_one_call_per_group_does(flatten) -> None:
    scenario = _scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    if flatten:
        elevations[:] = 0
    groups = _groups(mm.map_width, mm.map_height)
    held = elevations.copy()

    grouped = render.dirty_screen_bboxes_iso_grouped(scenario, groups, elevations, proj, flatten_elevations=flatten)

    assert np.array_equal(elevations, held)
    expected = _oracle(scenario, groups, elevations, proj, flatten_elevations=flatten)
    assert np.array_equal(elevations, held), "the oracle moved an elevation, so the two are not comparable"
    assert grouped == expected
    assert grouped[3] is None and all(box is not None for i, box in enumerate(grouped) if i != 3)
    assert len(set(grouped)) == len(grouped), "two groups got one bbox, so a mixed-up group would pass"
    if not flatten:
        flat = _oracle(scenario, groups, np.zeros_like(elevations), proj, flatten_elevations=True)
        assert grouped != flat, "the height field changed no bbox, so the sweep was not exercised"


def test_the_grouped_pass_declines_a_tile_whose_elevation_moved_and_writes_nothing() -> None:
    scenario = _scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    groups = _groups(mm.map_width, mm.map_height)
    tile = mm.terrain[groups[5][0]]
    tile.elevation = int(tile.elevation) + 1
    held = elevations.copy()

    assert render.dirty_screen_bboxes_iso_grouped(scenario, groups, elevations, proj) is None
    assert np.array_equal(elevations, held)
    # Flat + Isometric View: the mirror must hold 0 there, whatever the tile says.
    assert render.dirty_screen_bboxes_iso_grouped(scenario, groups, elevations, proj, flatten_elevations=True) is None
    elevations[:] = 0
    assert render.dirty_screen_bboxes_iso_grouped(scenario, groups, elevations, proj, flatten_elevations=True)


def test_the_grouped_pass_declines_an_elevation_outside_the_projection() -> None:
    scenario = _scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    groups = _groups(mm.map_width, mm.map_height)
    index = groups[5][0]
    mm.terrain[index].elevation = iso_geometry.MAX_ELEVATION + 1
    elevations[index // mm.map_width, index % mm.map_width] = iso_geometry.MAX_ELEVATION + 1

    assert render.dirty_screen_bboxes_iso_grouped(scenario, groups, elevations, proj) is None
    assert _oracle(scenario, [groups[5]], elevations, proj) == [None]


def test_no_groups_and_all_empty_groups() -> None:
    scenario = _scenario()
    elevations, proj = render.elevations_and_proj(scenario)
    assert render.dirty_screen_bboxes_iso_grouped(scenario, [], elevations, proj) == []
    assert render.dirty_screen_bboxes_iso_grouped(scenario, [[], []], elevations, proj) == [None, None]
