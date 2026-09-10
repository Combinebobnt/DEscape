"""gate_orientation's derived 4-cycles, and the footprint parity a const swap
has to preserve.

The pure cases run in the default tier: gate_orientation reads only the two
committed JSON tables, with no library import and no configured install. The
corpus-marked test at the bottom is what pins the coordinate-parity rule to
real files rather than to the judgement of the session that wrote it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from descape import gate_orientation, render
from descape.terrain_palette import tile_span

STONE_CLOSED = (64, 659, 88, 667)  # ne, e, se, n; AGENTS.md's own example
PALISADE_CLOSED_NE = 789
COLLIDING_CORNER = 1192  # carries code GTAC2 too, colliding with 81
WALL = 117
ARCHER = 4

_GRAPHIC_MAP_PATH = Path(__file__).resolve().parent.parent / "descape" / "unit_graphic_map.json"
_ORIENTATION_TOKEN_RE = re.compile(r"_(ne|se|e|n)_")
_TOKEN_FOR_ORIENTATION = {"A": "ne", "C": "e", "B": "se", "D": "n"}


def _graphics() -> dict[int, dict]:
    return {int(k): v for k, v in json.loads(_GRAPHIC_MAP_PATH.read_text())["graphics"].items()}


def _run_angle(unit_const: int) -> float | None:
    """The direction this gate's footprint runs, in degrees mod 180, derived
    from the committed span and building-tile tables rather than restated.

    None for the 1x1 corner groups, which are (1, 1) in all four orientations
    and so have no run to measure.
    """
    offsets = render.BUILDING_TILE_OFFSETS.get(unit_const)
    if offsets is None:
        span = tile_span(unit_const, render.NON_BUILDING_SPAN)
        vector = {(4, 1): (1, 0), (1, 4): (0, 1)}.get(span)
    else:
        vector = (1, 1) if (0, 0) in offsets else (1, -1)
    if vector is None:
        return None
    return math.degrees(math.atan2(vector[1], vector[0])) % 180


def test_the_catalog_yields_exactly_24_complete_groups_of_four():
    groups = gate_orientation.groups()
    assert len(groups) == 24
    assert sum(len(consts) for consts in groups.values()) == 96
    for key, consts in groups.items():
        assert len(set(consts)) == 4, key


def test_a_known_family_is_grouped_in_cycle_order():
    assert gate_orientation.groups()[("2", "A")] == STONE_CLOSED
    assert gate_orientation.orientation_siblings(88) == STONE_CLOSED


def test_the_const_whose_code_collides_is_in_no_group():
    """1192 also carries code GTAC2, colliding with 81. It is dropped because
    it has no unit_graphic_map.json entry (and no corpus placement) while 81
    has both, resolving the collision by data rather than by a hand-kept
    exclusion list."""
    assert not gate_orientation.is_gate(COLLIDING_CORNER)
    assert gate_orientation.groups()[("2", "C")][0] == 81


def test_every_group_agrees_with_its_graphics_own_orientation_token():
    """The file_name token (`_ne_`/`_se_`/`_e_`/`_n_`) is an independent read
    of the same fact as the code's middle letter. Zero disagreements is what
    makes the code-parsing derivation trustworthy."""
    graphics = _graphics()
    checked = 0
    for key, consts in gate_orientation.groups().items():
        for orientation, const in zip(("A", "C", "B", "D"), consts):
            match = _ORIENTATION_TOKEN_RE.search(graphics[const]["file_name"])
            if match is None:
                continue
            checked += 1
            assert match.group(1) == _TOKEN_FOR_ORIENTATION[orientation], (key, const)
    assert checked >= 48, "the file_name cross-check found nothing to check"


def test_one_cycle_step_turns_the_footprint_45_degrees():
    """The cycle order is A -> C -> B -> D, not the alphabetical order: by run
    direction those are 0, 45, 90 and 135 degrees in TILE space, which is what
    is asserted below. On screen the same four run in order -26.57, 0, +26.57
    and +90 degrees: still monotonically clockwise, the same sense
    unit_rotation.rotate_step's positive direction has, but no longer evenly
    spaced, because iso_geometry.tile_screen_origin() draws tile +x up-right
    and tile +x+y straight right under a 2:1 vertical squash. Tile space is
    therefore the only frame in which the step is a constant 45 degrees.
    """
    measured = 0
    for key, consts in gate_orientation.groups().items():
        angles = [_run_angle(const) for const in consts]
        if any(angle is None for angle in angles):
            assert all(angle is None for angle in angles), key
            continue
        measured += 1
        for i, angle in enumerate(angles):
            expected = (angles[0] + 45 * i) % 180
            assert angle == pytest.approx(expected), (key, consts[i])
    assert measured == 18, "the 1x1 corner groups are the only ones without a run direction"


def test_no_gate_const_is_also_a_rotatable_angle_const():
    """The viewer partitions a selection into rotatable / cyclable / skipped
    and assumes those cannot overlap. All 96 gate consts store a single frame,
    so none of them is ANGLE, and the partition is genuinely a partition. The
    24-const figure in AGENTS.md's gate rule counts only the visible ones."""
    from descape import unit_rotation

    for consts in gate_orientation.groups().values():
        for const in consts:
            assert unit_rotation.angle_count_for(const) == 1, const
            assert not unit_rotation.rotation_is_angle(const), const


def test_four_steps_return_the_original_const():
    for const in STONE_CLOSED + (PALISADE_CLOSED_NE,):
        assert gate_orientation.cycle_const(const, 4) == const
        assert gate_orientation.cycle_const(const, -4) == const


def test_stepping_backwards_is_the_inverse_of_stepping_forwards():
    for const in STONE_CLOSED:
        assert gate_orientation.cycle_const(gate_orientation.cycle_const(const, 1), -1) == const


def test_a_quarter_turn_is_two_orientation_steps():
    assert gate_orientation.QUARTER_TURN_STEPS == 2
    # ne (64) -> se (88), the 90-degree neighbour, in one coarse press.
    assert gate_orientation.cycle_const(64, gate_orientation.QUARTER_TURN_STEPS) == 88


@pytest.mark.parametrize("unit_const", [WALL, ARCHER, COLLIDING_CORNER])
def test_cycling_a_non_gate_raises_rather_than_no_opping(unit_const):
    """Loud refusal, mirroring unit_rotation.rotate_step's contract: a caller
    that thought it had cycled a wall should hear about it."""
    with pytest.raises(ValueError):
        gate_orientation.cycle_const(unit_const, 1)


@pytest.mark.corpus
def test_every_placed_gate_is_grouped_and_sits_on_its_own_span_anchor(corpus_files):
    """The measurement set_unit_const()'s re-anchoring rests on, re-measured
    rather than trusted: every real gate placement's coordinate is
    `tile + span/2` per axis, which is exactly render.span_anchor(). A
    counterexample would mean a const swap must not re-anchor the way it does.
    """
    from descape import scenario_io

    placements = 0
    ungrouped = []
    off_anchor = []
    for path in corpus_files:
        loaded = scenario_io.load_map_and_units(path)
        for units in loaded.unit_manager.units:
            for unit in units:
                if not gate_orientation.is_gate(unit.unit_const):
                    continue
                placements += 1
                if gate_orientation.orientation_siblings(unit.unit_const) is None:
                    ungrouped.append((path.name, unit.unit_const))
                    continue
                span = tile_span(unit.unit_const, render.NON_BUILDING_SPAN)
                low = (render._span_start(unit.x, span[0]), render._span_start(unit.y, span[1]))
                if (unit.x, unit.y) != render.span_anchor(*low, *span):
                    off_anchor.append((path.name, unit.unit_const, unit.x, unit.y))
    assert placements > 0, "the corpus subset holds no gates, so this test measured nothing"
    assert not ungrouped
    assert not off_anchor
