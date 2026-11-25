"""Which units draw faded for their Unit State, and how faded.

A unit's `status` byte (the Inspector's Unit State) is 2, Alive, for nearly
every placed unit. Any other labelled state (Foundation, Dead, Remove and the
rest, unit_fields.UNIT_STATE_CHOICES) means the unit is not there as a live
unit at map start, so the map draws it faded: sprites at GHOST_ALPHA, coloured
marks mixed GHOST_MARK_MIX of the way towards GHOST_MARK_GREY. It stays
pickable and editable, and Filters > Show Inactive Units hides it outright.

A raw byte with no label draws normally: what the game does with one is
unmeasured. Editor markers (invisible objects) are DEscape's own badges, not
game art, so the render seams leave them unfaded.

Stdlib-only, like unit_filter.py, so the render modules and unit_filter.py
can all import it with no cycle.
"""

from __future__ import annotations

ALIVE = 2

# Every labelled state but Alive; tests/test_unit_state.py pins it against the labels.
GHOSTED_STATES: frozenset[int] = frozenset({0, 1, 3, 4, 5, 6, 7, 8})

GHOST_ALPHA = 0.4
GHOST_MARK_MIX = 0.5
GHOST_MARK_GREY = (128, 128, 128)


def is_ghosted(unit) -> bool:
    """Whether `unit` draws faded. A unit with no `status` (a test fake) is Alive."""
    return getattr(unit, "status", ALIVE) in GHOSTED_STATES


def ghost_mark(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """A ghosted unit's coloured-mark colour: `color` mixed towards GHOST_MARK_GREY, opaque."""
    return tuple(round(c + (g - c) * GHOST_MARK_MIX) for c, g in zip(color, GHOST_MARK_GREY, strict=True))
