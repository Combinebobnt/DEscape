"""The rotation Place Unit gives the next unit it places (GH #172), and the
Unit Brush's per-unit one (GH #170, random_rotation()).

Place Unit's hover ghost can be turned before a click, by the wheel's rotate
binding (Ctrl+wheel by default) or the Rotate actions, or set by the
Eyedropper's Units-mode pick (ViewerWindow.pick_unit_value), and every click then
places the unit exactly as the ghost shows it. This module holds that pending
rotation and is the only source of the value both the ghost and the click
write, so the two cannot disagree.

It follows the Rotate rules (ViewerWindow.on_unit_rotate) per const:

- ANGLE (unit_rotation.rotation_is_angle): a facing in radians, stepped one
  stored frame at a time and snapped to each const's own frame grid, so it
  carries across picks with different angle counts.
- Cyclable (unit_variant.is_cyclable): a graphic-variant index, always written
  as a literal integer. It belongs to the const it was chosen on and reads as
  0 for any other pick.
- Gate (gate_orientation.is_gate): the orientation lives in the const, so a
  step returns the sibling const and leaves the state alone.
- Everything else (walls, cliffs, single-variant and INERT consts): nothing to
  turn, and rotation_for() is 0.0, add()'s own default.

The owner does not change any of this: a GAIA creatable turns and places its
facing like any player's, and render.stored_rotation() draws it (GH #217).

Qt-free and install-free (it reads the committed tables through the modules
above), so the default test tier covers it.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass

from descape import gate_orientation, unit_rotation, unit_variant

# step()'s outcomes.
TURNED = "turned"
CYCLED = "cycled"
GATE = "gate"
NOTHING = "nothing"


@dataclass(frozen=True)
class PlaceRotation:
    """`facing` is radians on no particular grid; rotation_for() snaps it per
    const. `variant` is an index into `variant_const`'s variants."""

    facing: float = 0.0
    variant: int = 0
    variant_const: int | None = None


def _snapped_facing(facing: float, const: int) -> float:
    count = unit_rotation.angle_count_for(const)
    return unit_rotation.facing_to_rotation(unit_rotation.rotation_to_facing(facing, count), count)


def variant_for(state: PlaceRotation, const: int) -> int:
    """The pending variant for `const`: the chosen one if it was chosen on this const, else 0."""
    if state.variant_const != const:
        return 0
    return state.variant % unit_variant.variant_count_for(const)


def step(state: PlaceRotation, const: int, owner: int, steps: int) -> tuple[PlaceRotation, int, str]:
    """Turns the pending rotation `steps` along for a `const` pick owned by
    `owner`. Returns (new state, the const to place, outcome). Only a gate
    changes the const; only an ANGLE or cyclable const changes the state.
    `owner` no longer changes the answer (GH #217); callers still pass it."""
    if unit_rotation.rotation_is_angle(const):
        count = unit_rotation.angle_count_for(const)
        facing = unit_rotation.rotate_step(_snapped_facing(state.facing, const), count, steps)
        return dataclasses.replace(state, facing=facing), const, TURNED
    if unit_variant.is_cyclable(const):
        variant = (variant_for(state, const) + steps) % unit_variant.variant_count_for(const)
        return dataclasses.replace(state, variant=variant, variant_const=const), const, CYCLED
    if gate_orientation.is_gate(const):
        return state, gate_orientation.cycle_const(const, steps), GATE
    return state, const, NOTHING


def steps_for_coarse(const: int, direction: int) -> int:
    """A quarter turn for `const`, in on_unit_rotate_coarse's units: frames for
    ANGLE, a quarter of the variants for cyclable, two orientations for a gate."""
    if unit_rotation.rotation_is_angle(const):
        return direction * unit_rotation.quarter_turn_steps(unit_rotation.angle_count_for(const))
    if unit_variant.is_cyclable(const):
        return direction * unit_rotation.quarter_turn_steps(unit_variant.variant_count_for(const))
    if gate_orientation.is_gate(const):
        return direction * gate_orientation.QUARTER_TURN_STEPS
    return direction


def random_rotation(rng: random.Random, const: int, owner: int) -> float:
    """A random facing or variant for one Unit Brush placement (GH #170),
    still through rotation_for(): an ANGLE const rolls a facing on its own
    grid, a cyclable const a variant index, and every other const (walls,
    gates, cliffs, INERT) draws nothing from `rng` and gets 0.0."""
    if unit_rotation.rotation_is_angle(const):
        count = unit_rotation.angle_count_for(const)
        state = PlaceRotation(facing=unit_rotation.facing_to_rotation(rng.randrange(count), count))
    elif unit_variant.is_cyclable(const):
        state = PlaceRotation(variant=rng.randrange(unit_variant.variant_count_for(const)), variant_const=const)
    else:
        state = PlaceRotation()
    return rotation_for(state, const, owner)


def rotation_for(state: PlaceRotation, const: int, owner: int) -> float:
    """The `rotation` a placement of `const` for `owner` stores, and the only
    source of one: the snapped facing for an ANGLE const, GAIA's included
    (GH #217), the literal variant index for a cyclable const, 0.0 for
    everything else (walls, cliffs, gates, INERT). Never a radian-encoded
    variant."""
    if unit_rotation.rotation_is_angle(const):
        return _snapped_facing(state.facing, const)
    if unit_variant.is_cyclable(const):
        return float(variant_for(state, const))
    return 0.0
