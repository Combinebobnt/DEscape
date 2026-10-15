"""The rotation Place Unit gives the next unit it places (GH #172).

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

GAIA owns no camera-relative facing: render.stored_rotation() draws a GAIA
creatable at 0.0 whatever it stores, so an ANGLE pick with owner GAIA does not
turn and places 0.0. GAIA trees and gates still rotate.

Qt-free and install-free (it reads the committed tables through the modules
above), so the default test tier covers it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from descape import gate_orientation, unit_rotation, unit_variant
from descape.unit_filter import GAIA_PLAYER_ID

# step()'s outcomes.
TURNED = "turned"
CYCLED = "cycled"
GATE = "gate"
GAIA_NO_FACING = "gaia_no_facing"
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
    changes the const; only an ANGLE or cyclable const changes the state."""
    if unit_rotation.rotation_is_angle(const):
        if owner == GAIA_PLAYER_ID:
            return state, const, GAIA_NO_FACING
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


def rotation_for(state: PlaceRotation, const: int, owner: int) -> float:
    """The `rotation` a placement of `const` for `owner` stores, and the only
    source of one: the snapped facing for a non-GAIA ANGLE const, the literal
    variant index for a cyclable const, 0.0 for everything else (walls,
    cliffs, gates, INERT). Never a radian-encoded variant."""
    if unit_rotation.rotation_is_angle(const):
        return 0.0 if owner == GAIA_PLAYER_ID else _snapped_facing(state.facing, const)
    if unit_variant.is_cyclable(const):
        return float(variant_for(state, const))
    return 0.0
