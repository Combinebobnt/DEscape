"""The turn a Move drag holds until its release (GH #280).

Rotate inputs during a live drag write nothing: they only add steps here, and
the drop applies them once, in the move's own undo record. Steps rather than
per-unit values, so a group needs no per-member map, and an undo mid-drag
cannot leave a value stale.

turned() is on_unit_rotate's per-const rule: a facing through rotate_step(), a
cyclable const's variant through cycle_step(), a gate's sibling through
cycle_const(), and anything else passes through. Steps that net a whole cycle
for a const hand the stored value back verbatim, so turning and turning back
cannot re-encode a `rotation` (AGENTS.md's GAIA rule).

Qt-free, like its three sources, so the default test tier covers it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from descape import gate_orientation, unit_rotation, unit_variant


@dataclass(frozen=True)
class MoveTurn:
    """The steps pending on a drag. Falsy when all four are 0."""

    facing: int = 0
    gate: int = 0
    variant: int = 0
    # Quarter turns of a variant's own count: its size differs per const (oak 42 -> 11, pine 27 -> 7).
    variant_quarters: int = 0

    def __bool__(self) -> bool:
        return bool(self.facing or self.gate or self.variant or self.variant_quarters)


def fine(turn: MoveTurn, steps: int) -> MoveTurn:
    """`turn` plus the fine Rotate pair's `steps`."""
    return replace(turn, facing=turn.facing + steps, gate=turn.gate + steps, variant=turn.variant + steps)


def coarse(turn: MoveTurn, direction: int, facing_steps: int) -> MoveTurn:
    """`turn` plus a quarter-turn key. `facing_steps` is the caller's, as
    on_unit_rotate_coarse takes one frame count for a whole selection."""
    return replace(
        turn,
        facing=turn.facing + direction * facing_steps,
        gate=turn.gate + direction * gate_orientation.QUARTER_TURN_STEPS,
        variant_quarters=turn.variant_quarters + direction,
    )


def turned(turn: MoveTurn, unit_const: int, rotation: float) -> tuple[int, float]:
    """(const, rotation) of a unit after `turn`, the inputs themselves when it
    doesn't turn or the steps are a whole cycle for it."""
    if unit_rotation.rotation_is_angle(unit_const):
        angle_count = unit_rotation.angle_count_for(unit_const)
        if turn.facing % angle_count == 0:
            return unit_const, rotation
        return unit_const, unit_rotation.rotate_step(rotation, angle_count, turn.facing)
    if gate_orientation.is_gate(unit_const):
        # A whole cycle wraps to the same const, and a gate's rotation is never touched.
        return gate_orientation.cycle_const(unit_const, turn.gate), rotation
    if unit_variant.is_cyclable(unit_const):
        count = unit_variant.variant_count_for(unit_const)
        steps = turn.variant + turn.variant_quarters * unit_rotation.quarter_turn_steps(count)
        if steps % count == 0:
            return unit_const, rotation
        return unit_const, unit_variant.cycle_step(rotation, unit_rotation.angle_count_for(unit_const), count, steps)
    return unit_const, rotation
